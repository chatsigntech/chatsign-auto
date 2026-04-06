"""Real-time sign language recognition session.

Adapts the inference pipeline from gloss_aware/app_local_pose.py into a
stateful class suitable for WebSocket frame-by-frame processing.

Model + RTMPose extractor are cached at module level (max 2 models) to avoid
expensive reloading on every connection.
"""

import logging
import sys
import threading
import types
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from backend.config import settings

logger = logging.getLogger(__name__)

# Numpy compat shim (required by some gloss_aware dependencies)
if "numpy._core" not in sys.modules:
    try:
        import numpy.core as _npcore

        _mod = types.ModuleType("numpy._core")
        _mod.__dict__.update(_npcore.__dict__)
        sys.modules["numpy._core"] = _mod
        for _sub in ("multiarray", "umath", "_multiarray_umath", "numeric"):
            _full = f"numpy._core.{_sub}"
            if _full not in sys.modules:
                _attr = getattr(_npcore, _sub, None)
                if _attr is not None:
                    sys.modules[_full] = _attr
    except Exception:
        pass

# Default model module — matches the ASL config in infer_sentence.MODEL_CONFIGS.
# The checkpoint's saved args may override this if a different architecture was used.
MODEL_MODULE = "ssl_models_crossvideo_mlp_feature_mean_mean_advance_v4_noconf_clip_nob2b"

_ga_imports_ready = False


def _ensure_ga_imports():
    global _ga_imports_ready
    if _ga_imports_ready:
        return
    ga_path = str(settings.GLOSS_AWARE_PATH.resolve())
    if ga_path not in sys.path:
        sys.path.insert(0, ga_path)
    _ga_imports_ready = True


def _import_ga():
    _ensure_ga_imports()
    from preprocess.pose_extractor import BatchRTMPoseVideoPreprocessor
    from preprocess.filter_pose_pkls import evaluate_frame
    from preprocess.norm_cosign_unified import load_part_kp
    from infer.infer_sentence import (
        PARTS,
        load_prototypes,
        build_vq_model_unified,
        compute_window_code_embedding,
        cosine_retrieve,
        emit_tokens,
        postprocess_text,
    )
    return {
        "BatchRTMPoseVideoPreprocessor": BatchRTMPoseVideoPreprocessor,
        "evaluate_frame": evaluate_frame,
        "load_part_kp": load_part_kp,
        "PARTS": PARTS,
        "load_prototypes": load_prototypes,
        "build_vq_model_unified": build_vq_model_unified,
        "compute_window_code_embedding": compute_window_code_embedding,
        "cosine_retrieve": cosine_retrieve,
        "emit_tokens": emit_tokens,
        "postprocess_text": postprocess_text,
    }


def find_best_checkpoint(ckpt_dir: Path) -> Path | None:
    """Find the best checkpoint in a directory (best_cl > best > latest)."""
    if not ckpt_dir.exists():
        return None
    for name in ("best_cl.pth", "best.pth"):
        candidate = ckpt_dir / name
        if candidate.exists():
            return candidate
    ckpts = sorted(ckpt_dir.glob("*.pth"))
    return ckpts[-1] if ckpts else None


@dataclass
class ModelBundle:
    model: object  # PoseBlockEncoderSignCL
    proto_target: torch.Tensor
    use_centroid: bool
    gloss_centroid_ids: list
    video_to_gloss_id: list
    id_to_token: dict
    parts: list
    device: torch.device
    conf_threshold: float
    window_size: int
    stride: int
    gpu_id: int


_model_cache: OrderedDict[str, ModelBundle] = OrderedDict()
_model_cache_lock = threading.Lock()
MAX_CACHED_MODELS = 2

_extractor: Optional[object] = None
_extractor_lock = threading.Lock()
_ga = None


def _get_extractor():
    global _extractor, _ga
    with _extractor_lock:
        if _extractor is None:
            if _ga is None:
                _ga = _import_ga()
            logger.info("Initializing RTMPose extractor...")
            _extractor = _ga["BatchRTMPoseVideoPreprocessor"](
                batch_size=1, num_workers=1
            )
            if not _extractor.pose_processing_enabled:
                raise RuntimeError("RTMPose failed to initialize")
            logger.info("RTMPose extractor ready")
        return _extractor


def _get_ga():
    global _ga
    if _ga is None:
        _ga = _import_ga()
    return _ga


def _detect_model_config(ckpt: dict) -> tuple[str, int, int]:
    """Infer model_module, window_size, stride from checkpoint saved args."""
    args_ck = ckpt.get("args", {})
    # block_size in training corresponds to window_size in inference
    window_size = args_ck.get("block_size", 20)
    stride = args_ck.get("block_stride", window_size // 2)
    # Detect module from architecture hints
    module = MODEL_MODULE
    return module, window_size, stride


def load_model_bundle(task_id: str, gpu_id: int = 0) -> ModelBundle:
    """Load or retrieve cached model bundle for a task."""
    with _model_cache_lock:
        if task_id in _model_cache:
            _model_cache.move_to_end(task_id)
            return _model_cache[task_id]

    ga = _get_ga()
    phase8_output = settings.SHARED_DATA_ROOT / task_id / "phase_8" / "output"

    ckpt_path = find_best_checkpoint(phase8_output / "checkpoints")
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {phase8_output / 'checkpoints'}")

    proto_dir = phase8_output / "prototypes"
    if not (proto_dir / "prototypes.pt").exists():
        raise FileNotFoundError(f"Prototypes not found at {proto_dir / 'prototypes.pt'}")

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model for task {task_id} from {ckpt_path}")
    proto_data = ga["load_prototypes"](str(proto_dir))
    parts = proto_data.get("parts") or ga["PARTS"]
    num_codes_per_part = proto_data.get("num_codes_per_part")

    # Peek into checkpoint to detect window_size/stride from training args
    ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model_module, window_size, stride = _detect_model_config(ckpt_data)
    del ckpt_data  # free memory, build_vq_model_unified will reload

    model, conf_threshold = ga["build_vq_model_unified"](
        str(ckpt_path), model_module, device, num_codes_per_part
    )

    # Retrieval target: default to video prototypes (same as original app_local_pose.py)
    gloss_centroids = proto_data.get("gloss_centroids")
    gloss_centroid_ids = proto_data.get("gloss_centroid_ids")
    proto_target = proto_data["video_prototypes"].to(device)
    use_centroid = False

    bundle = ModelBundle(
        model=model,
        proto_target=proto_target,
        use_centroid=use_centroid,
        gloss_centroid_ids=gloss_centroid_ids or [],
        video_to_gloss_id=proto_data["video_to_gloss_id"],
        id_to_token=proto_data["id_to_token"],
        parts=parts,
        device=device,
        conf_threshold=conf_threshold,
        window_size=window_size,
        stride=stride,
        gpu_id=gpu_id,
    )

    with _model_cache_lock:
        _model_cache[task_id] = bundle
        _model_cache.move_to_end(task_id)
        while len(_model_cache) > MAX_CACHED_MODELS:
            evicted_id, evicted = _model_cache.popitem(last=False)
            logger.info(f"Evicted model cache for task {evicted_id}")
            del evicted

    logger.info(
        f"Model loaded for task {task_id}: "
        f"{proto_target.shape[0]} targets, window={window_size}, stride={stride}, device={device}"
    )
    return bundle


class RecognitionSession:
    """Per-connection inference state. Model bundle is shared via cache.

    Faithfully reproduces the inference logic from gloss_aware/app_local_pose.py:
    - Frame: RTMPose → evaluate_frame → accumulate raw (1,133,2) arrays
    - Flush: pass list of arrays to load_part_kp → sliding window → VQ → retrieve
    - Emit: voting over streaming_results → postprocess_text
    """

    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.extractor = _get_extractor()
        self._ga = _get_ga()

        # Per-session mutable state — mirrors app_local_pose.py inference_worker
        self.raw_kps_buffer: list[np.ndarray] = []   # list of (1, 133, 2)
        self.raw_scores_buffer: list[np.ndarray] = [] # list of (1, 133)
        self.kps_global: dict = {}
        self.streaming_results: list = []
        self.local_pose_frames: int = 0
        self.pending_new_frames: int = 0
        self.next_window_end: int = bundle.window_size

        self._last_token_count: int = 0

        self.hand_threshold = 0.8
        self.head_threshold = 0.8
        self.hand_height_threshold = 0.1
        self.emit_mode = "voting"
        self.vote_window = 6
        self.punct_model = "pcs_en"

    def process_frame(self, jpeg_bytes: bytes) -> Optional[dict]:
        """Process one JPEG frame. Returns prediction dict if new windows available."""
        img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        kps, scores = self.extractor.process_frame_onnx(frame)
        if kps is None:
            return None

        kps_p = kps[np.newaxis, :, :]    # (1, 133, 2)
        scores_p = scores[np.newaxis, :]  # (1, 133)
        decision = self._ga["evaluate_frame"](
            kps_p, scores_p,
            self.hand_threshold, self.head_threshold,
            self.hand_height_threshold,
            frame_idx=self.local_pose_frames,
        )
        if not decision.keep:
            return None

        self.raw_kps_buffer.append(kps_p.copy())
        self.raw_scores_buffer.append(scores_p.copy())
        self.local_pose_frames += 1
        self.pending_new_frames += 1

        b = self.bundle
        if self.pending_new_frames >= b.stride or self.local_pose_frames == b.window_size:
            return self._flush()
        return None

    def finalize(self) -> Optional[dict]:
        """Flush remaining frames on session end (mirrors original's final flush)."""
        if self.pending_new_frames > 0:
            return self._flush()
        return None

    def _flush(self) -> Optional[dict]:
        """Normalize full buffer and run new sliding windows.

        Matches app_local_pose.py normalize_raw_buffer() + _flush() logic exactly:
        load_part_kp expects a list of (1, 133, 2) arrays, NOT a concatenated array.
        """
        if not self.raw_kps_buffer:
            return None

        b = self.bundle
        ga = self._ga

        # Pass list directly — load_part_kp iterates with zip() and peels
        # the leading dim with skeleton[0], so each element must be (1, 133, 2)
        kps_norm = ga["load_part_kp"](self.raw_kps_buffer, self.raw_scores_buffer)
        self.kps_global = {}
        for part in b.parts:
            if part in kps_norm:
                self.kps_global[part] = torch.from_numpy(kps_norm[part]).float()
        self.pending_new_frames = 0

        actual = (
            self.kps_global[b.parts[0]].shape[0]
            if b.parts[0] in self.kps_global
            else 0
        )

        new_results = []
        while self.next_window_end <= actual:
            r = self._run_one_window(
                self.next_window_end - b.window_size,
                self.next_window_end,
            )
            if r:
                new_results.append(r)
            self.next_window_end += b.stride

        if new_results:
            self.streaming_results.extend(new_results)

        emitted = ga["emit_tokens"](
            self.streaming_results,
            mode=self.emit_mode,
            vote_window=self.vote_window,
        )
        pred_tokens = [t for t, _, _ in emitted]
        self._last_token_count = len(pred_tokens)
        sentence = (
            ga["postprocess_text"](pred_tokens, model_name=self.punct_model)
            if pred_tokens
            else ""
        )

        latest_token = ""
        latest_score = 0.0
        if new_results:
            latest_token, latest_score, _ = new_results[-1]

        return {
            "type": "prediction",
            "tokens": pred_tokens,
            "latest_token": latest_token,
            "latest_score": round(latest_score, 3),
            "sentence": sentence,
            "pose_frames": self.local_pose_frames,
            "windows": len(self.streaming_results),
        }

    def _run_one_window(self, w_start: int, w_end: int):
        """Encode + retrieve for a single sliding window."""
        b = self.bundle

        blocks_by_part = {}
        for p in b.parts:
            if p in self.kps_global:
                blocks_by_part[p] = (
                    self.kps_global[p][w_start:w_end].unsqueeze(0).to(b.device)
                )
        if not blocks_by_part:
            return None

        ga = self._ga
        with torch.no_grad():
            _, _, _, vq_indices, _ = b.model.encode(
                blocks_by_part,
                masks_by_part=None,
                conf_threshold=b.conf_threshold,
                return_sequence=False,
                return_vq=True,
            )
            window_code = ga["compute_window_code_embedding"](
                b.model, vq_indices, 0, b.parts, b.device,
            )
            indices, scores_ret = ga["cosine_retrieve"](
                window_code, b.proto_target, topk=1,
            )

        idx = indices.squeeze().item()
        sim = scores_ret.squeeze().item()
        if b.use_centroid:
            gloss_id = b.gloss_centroid_ids[idx]
        else:
            gloss_id = b.video_to_gloss_id[idx]
        token = b.id_to_token.get(gloss_id, f"<unk_{gloss_id}>")
        center = w_start + b.window_size // 2
        return (token, sim, center)

    def reset(self):
        """Clear all accumulated state for a fresh recognition pass."""
        self.raw_kps_buffer.clear()
        self.raw_scores_buffer.clear()
        self.kps_global.clear()
        self.streaming_results.clear()
        self.local_pose_frames = 0
        self.pending_new_frames = 0
        self.next_window_end = self.bundle.window_size
        self._last_token_count = 0

    def get_status(self) -> dict:
        return {
            "pose_frames": self.local_pose_frames,
            "windows": len(self.streaming_results),
            "tokens": self._last_token_count,
        }
