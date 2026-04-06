"""Phase 4: Train a per-task segmentation model using spamo_segement.

Trains on SENTENCE videos only (not word/gloss videos).
This phase handles feature extraction and model training only.
Segmentation inference is performed in Phase 5.

Steps:
  4.1  Extract CLIP-ViT spatial features from sentence videos
  4.2  Generate annotation files (train_info_ml.npy + val_info_ml.npy)
  4.3  Generate task-specific config YAML
  4.4  Train segmentation model (SpaMo: Flan-T5-XL + LoRA + OT alignment)

Input:  Phase 2 output (videos/ + manifest.json) + Phase 1 output (glosses.json)
Output: checkpoint + config + features + annotations
"""
import json
import logging
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from backend.config import settings
from backend.core.subprocess_runner import run_subprocess
from backend.core.video_utils import make_gpu_env

logger = logging.getLogger(__name__)

SPAMO_ROOT = Path(settings.SPAMO_SEGMENT_PATH).resolve()
SPAMO_PYTHON = sys.executable


def _get_sentence_videos(phase2_output: Path, phase1_output: Path) -> list[dict]:
    """Load sentence-only videos from Phase 2 manifest.

    Dataset mode (has dataset_source field): all videos are sentence-level, return all.
    Manual mode: filters out word/gloss videos by checking if sentence_text
    appears as a key in Phase 1's glosses.json (gloss keys are full sentences).
    """
    manifest = phase2_output / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Phase 2 manifest not found: {manifest}")
    with open(manifest) as f:
        all_entries = json.load(f)

    if not all_entries:
        return all_entries

    # Dataset mode: all entries have dataset_source, all are sentences
    if all_entries[0].get("dataset_source"):
        return all_entries

    # Manual mode: glosses.json keys are the original sentences.
    glosses_file = phase1_output / "glosses.json"
    if not glosses_file.exists():
        return all_entries

    with open(glosses_file) as f:
        sentence_keys = set(k.strip() for k in json.load(f).keys())

    return [e for e in all_entries if e.get("sentence_text", "").strip() in sentence_keys]


def _get_pseudo_glosses(phase1_output: Path) -> dict[str, list[str]]:
    """Load glosses from Phase 1 output."""
    glosses_file = phase1_output / "glosses.json"
    if not glosses_file.exists():
        return {}
    with open(glosses_file) as f:
        return json.load(f)


_make_env = make_gpu_env


async def _extract_clip_features(
    task_id: str, video_dir: Path, output_dir: Path, gpu_id: int = 0
) -> int:
    """Step 4.1: Extract CLIP-ViT spatial features from videos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    script = SPAMO_ROOT / "scripts" / "extract_features" / "extract_clip_from_mp4.py"
    cmd = [
        SPAMO_PYTHON, str(script),
        "--video_dir", str(video_dir),
        "--output_dir", str(output_dir),
        "--device", f"cuda:{gpu_id}",
        "--batch_size", "16",
        "--s2_mode", "s2wrapping",
        "--scales", "1", "2",
    ]

    logger.info(f"[{task_id}] Step 4.1: Extracting CLIP features from {video_dir}")
    rc, stdout, stderr = await run_subprocess(
        cmd, cwd=SPAMO_ROOT, env=_make_env(gpu_id), log_to_file=True
    )
    if rc != 0:
        raise RuntimeError(f"CLIP feature extraction failed (rc={rc}): {(stderr or stdout)[-500:]}")

    count = len(list(output_dir.glob("*_s2wrapping.npy")))
    if count == 0:
        raise RuntimeError(
            f"CLIP feature extraction produced 0 features. "
            f"video_dir={video_dir} (videos: {len(list(video_dir.glob('*.mp4')))}), "
            f"output_dir={output_dir}. Subprocess output: {(stdout or stderr)[-300:]}"
        )
    for split in ("train", "val", "dev", "test"):
        split_dir = output_dir / split
        if not split_dir.exists():
            split_dir.symlink_to(output_dir)

    logger.info(f"[{task_id}] Step 4.1: Extracted features for {count} videos")
    return count


def _build_annotations(
    task_id: str,
    manifest: list[dict],
    glosses: dict[str, list[str]],
    anno_dir: Path,
    val_ratio: float = 0.2,
) -> tuple[int, int]:
    """Step 4.2: Generate train/val annotation files (npy format)."""
    anno_dir.mkdir(parents=True, exist_ok=True)

    gloss_map = {}
    for sent, glist in glosses.items():
        gloss_map[sent.strip()] = " ".join(g.lower() for g in glist)

    entries = []
    for item in manifest:
        text = item.get("sentence_text", "")
        pseudo = gloss_map.get(text.strip(), text)
        filename = item.get("filename", "")
        file_id = Path(filename).stem

        entries.append({
            "fileid": file_id,
            "folder": file_id,
            "text": pseudo,
            "original_text": text,
            "gloss": "",
            "start": 0.0,
            "end": 0.0,
            "video_name": filename,
        })

    random.Random(42).shuffle(entries)

    n_val = max(1, int(len(entries) * val_ratio))
    n_train = len(entries) - n_val

    np.save(anno_dir / "train_info_ml.npy", np.array(entries[:n_train], dtype=object))
    np.save(anno_dir / "val_info_ml.npy", np.array(entries[n_train:], dtype=object))
    np.save(anno_dir / "test_info_ml.npy", np.array(entries, dtype=object))

    logger.info(f"[{task_id}] Step 4.2: Annotations: {n_train} train, {n_val} val")
    return n_train, n_val


def _generate_config(
    task_id: str,
    anno_dir: Path,
    feat_dir: Path,
    video_dir: Path,
    output_dir: Path,
    data_size: int,
) -> Path:
    """Step 4.3: Generate task-specific config YAML from template."""
    template_path = SPAMO_ROOT / "configs" / "how2sign_contrastive_word.yaml"
    config = OmegaConf.load(template_path)

    for split in ("train", "validation", "test"):
        p = config.data.params[split].params
        p.anno_root = str(anno_dir)
        p.feat_root = str(feat_dir)
        p.vid_root = str(video_dir)
        p.mae_feat_root = str(feat_dir)
        p.flat_feat_dir = True

    warmup = max(200, min(2000, data_size * 6))
    config.model.params.warm_up_steps = warmup
    config.lightning.trainer.max_epochs = 80
    config.lightning.trainer.check_val_every_n_epoch = 1
    # Shared HF cache on data disk to avoid re-downloading per task
    shared_hf_cache = settings.SHARED_DATA_ROOT / ".hf_cache"
    shared_hf_cache.mkdir(parents=True, exist_ok=True)
    config.model.params.cache_dir = str(shared_hf_cache)

    config_path = (output_dir / f"config_{task_id}.yaml").resolve()
    OmegaConf.save(config, config_path)

    logger.info(f"[{task_id}] Step 4.3: Config saved (warmup={warmup}, epochs=80)")
    return config_path


async def _train_model(
    task_id: str, config_path: Path, output_dir: Path, gpu_id: int = 0
) -> Path:
    """Step 4.4: Train segmentation model."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        SPAMO_PYTHON, str(SPAMO_ROOT / "main.py"),
        "--config", str(config_path),
        "--train", "true",
        "--logdir", str(log_dir),
        "--name", task_id,
        "--seed", "42",
    ]

    logger.info(f"[{task_id}] Step 4.4: Starting segmentation model training")
    rc, stdout, stderr = await run_subprocess(
        cmd, cwd=SPAMO_ROOT, env=_make_env(gpu_id), log_to_file=True
    )

    ckpt_dir = None
    for d in sorted(log_dir.rglob("checkpoints"), reverse=True):
        if d.is_dir():
            ckpt_dir = d
            break

    if ckpt_dir is None:
        raise RuntimeError(f"Training produced no checkpoints. Output: {(stderr or stdout)[-1000:]}")

    best_ckpt = None
    for pattern in ("best_bleu4*.ckpt", "best_loss*.ckpt", "last*.ckpt"):
        matches = sorted(ckpt_dir.glob(pattern))
        if matches:
            best_ckpt = matches[-1]
            break

    if best_ckpt is None:
        raise RuntimeError(f"No checkpoint files in {ckpt_dir}")

    final_ckpt = output_dir / "segmentation_model.ckpt"
    shutil.copy2(best_ckpt, final_ckpt)

    logger.info(f"[{task_id}] Step 4.4: Training complete -> {final_ckpt.name}")
    return final_ckpt


async def run_phase4_segmentation_train(
    task_id: str,
    phase2_output: Path,
    phase1_output: Path,
    output_dir: Path,
    gpu_id: int = 0,
) -> dict:
    """Run segmentation training pipeline (Phase 4: training only, no inference)."""
    output_dir = output_dir.resolve()
    phase2_output = phase2_output.resolve()
    phase1_output = phase1_output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _get_sentence_videos(phase2_output, phase1_output)
    glosses = _get_pseudo_glosses(phase1_output)

    if not manifest:
        raise RuntimeError("No sentence videos found in Phase 2 output")

    video_dir = phase2_output / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"No videos directory: {video_dir}")

    # Symlink sentence-only videos into a clean directory
    sentence_video_dir = output_dir / "sentence_videos"
    sentence_video_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        fn = entry.get("filename", "")
        src = video_dir / fn
        dst = sentence_video_dir / fn
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve() if src.is_symlink() else src)

    video_count = len(list(sentence_video_dir.glob("*.mp4")))
    logger.info(f"[{task_id}] Phase 4: {video_count} sentence videos selected for training")

    if video_count == 0:
        raise RuntimeError("No sentence video files found on disk")

    feat_dir = output_dir / "features"
    feat_count = await _extract_clip_features(task_id, sentence_video_dir, feat_dir, gpu_id)

    anno_dir = output_dir / "annotations"
    n_train, n_val = _build_annotations(task_id, manifest, glosses, anno_dir)

    config_path = _generate_config(
        task_id, anno_dir, feat_dir, sentence_video_dir, output_dir, n_train + n_val
    )

    ckpt_path = await _train_model(task_id, config_path, output_dir, gpu_id)

    # Save manifest for Phase 5 to reuse
    with open(output_dir / "sentence_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"[{task_id}] Phase 4 completed")
    return {
        "input_videos": video_count,
        "features_extracted": feat_count,
        "train_samples": n_train,
        "val_samples": n_val,
        "checkpoint": str(ckpt_path),
    }
