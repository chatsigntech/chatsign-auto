"""Test video generation: composable augmentation pipeline + concatenation.

Augmentation steps are applied sequentially — output of step N becomes
input of step N+1.  Supported step types:

  cv2d      – 2D CV augmentation (25 types: crop, rotate, color, etc.)
  temporal  – temporal augmentation (7 types: speed change, subsampling)
  3d_view   – EHM-Tracker + GUAVA novel viewpoint rendering (GPU required)

Example pipeline:
  [
    {"type": "temporal", "id": 2},                        # speed 1.25x
    {"type": "3d_view", "yaw": 0.25, "pitch": 0, "zoom": 1.0}  # right turn
  ]
"""

import json
import logging
import random
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from backend.config import settings
from backend.core.video_utils import reencode_to_h264

logger = logging.getLogger(__name__)

GUAVA_PATH = settings.GUAVA_AUG_PATH.resolve()

GAP_FRAMES = 15  # ~0.5s black frames between sentences

# ---------------------------------------------------------------------------
# Lazy-loaded augmentation modules
# ---------------------------------------------------------------------------
_guava_path_added = False
_cv_aug = None
_temporal_aug = None


def _ensure_guava_path():
    global _guava_path_added
    if not _guava_path_added:
        guava_path = str(GUAVA_PATH)
        if guava_path not in sys.path:
            sys.path.insert(0, guava_path)
        _guava_path_added = True


def _get_cv_aug():
    global _cv_aug
    if _cv_aug is None:
        _ensure_guava_path()
        from cv_aug.augment import augment_video, AUGMENTATIONS
        _cv_aug = {"fn": augment_video, "list": AUGMENTATIONS}
    return _cv_aug


def _get_temporal_aug():
    global _temporal_aug
    if _temporal_aug is None:
        _ensure_guava_path()
        from cv_aug.temporal_augment import temporal_augment_video, TEMPORAL_AUGMENTATIONS
        _temporal_aug = {"fn": temporal_augment_video, "list": TEMPORAL_AUGMENTATIONS}
    return _temporal_aug


# ---------------------------------------------------------------------------
# Individual augmentation step runners
# ---------------------------------------------------------------------------

def _apply_cv2d(input_path: Path, output_path: Path, step: dict) -> str:
    aug = _get_cv_aug()
    aug_id = step.get("id")
    if aug_id is None:
        aug_id = random.randint(0, len(aug["list"]) - 1)
    aug_name = aug["list"][aug_id]["name"]
    aug["fn"](str(input_path), str(output_path), aug_id)
    return f"cv2d:{aug_name}"


def _apply_temporal(input_path: Path, output_path: Path, step: dict) -> str:
    aug = _get_temporal_aug()
    aug_id = step.get("id")
    if aug_id is None:
        aug_id = random.randint(0, len(aug["list"]) - 1)
    aug_name = aug["list"][aug_id]["name"]
    aug["fn"](str(input_path), str(output_path), aug_id)
    return f"temporal:{aug_name}"


def _apply_3d_view(
    input_path: Path, output_path: Path, step: dict,
    work_dir: Path, gpu_id: int = 0,
) -> str:
    """Apply 3D viewpoint augmentation via EHM-Tracker + GUAVA render.

    Reuses _run_ehm_tracking and _run_guava_render from the pipeline worker.
    """
    from backend.workers.phase7_augment import _run_ehm_tracking, _run_guava_render

    yaw = step.get("yaw", 0.25)
    pitch = step.get("pitch", 0.0)
    zoom = step.get("zoom", 1.0)
    view_name = step.get("name", f"y{yaw}_p{pitch}_z{zoom}")

    video_stem = input_path.stem

    # EHM-Tracker expects a directory of videos
    track_input = work_dir / "track_input"
    track_input.mkdir(parents=True, exist_ok=True)
    track_src = track_input / f"{video_stem}.mp4"
    shutil.copy2(str(input_path), str(track_src))

    tracked_dir = work_dir / "tracked"
    tracked_videos = _run_ehm_tracking(
        "test_video", track_input, tracked_dir, gpu_id,
    )

    if not tracked_videos:
        raise RuntimeError(f"EHM-Tracker produced no tracked data for {video_stem}")

    viewpoint = {"name": view_name, "yaw": yaw, "pitch": pitch, "zoom": zoom}
    render_dir = work_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)

    rendered = _run_guava_render(
        "test_video", tracked_videos[0], render_dir,
        tracked_videos[0].name, viewpoint, gpu_id,
    )

    if rendered is None:
        raise RuntimeError(f"GUAVA render produced no output for {view_name}")

    shutil.copy2(str(rendered), str(output_path))
    return f"3d_view:{view_name}"


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _apply_pipeline(
    input_path: Path, output_path: Path, pipeline: list[dict],
    work_dir: Path, gpu_id: int = 0,
) -> str:
    """Apply a sequence of augmentation steps to a single video.

    Returns a combined description like "temporal:speed_1.25x+3d_view:yaw_right".
    """
    if not pipeline:
        shutil.copy2(str(input_path), str(output_path))
        return "none"

    descriptions = []
    current_input = input_path

    for i, step in enumerate(pipeline):
        step_type = step.get("type", "cv2d")
        is_last = (i == len(pipeline) - 1)
        step_output = output_path if is_last else work_dir / f"step_{i}.mp4"

        if step_type == "cv2d":
            desc = _apply_cv2d(current_input, step_output, step)
        elif step_type == "temporal":
            desc = _apply_temporal(current_input, step_output, step)
        elif step_type == "3d_view":
            step_work = work_dir / f"step_{i}_work"
            step_work.mkdir(parents=True, exist_ok=True)
            desc = _apply_3d_view(
                current_input, step_output, step, step_work, gpu_id,
            )
        else:
            raise ValueError(f"Unknown augmentation type: {step_type}")

        descriptions.append(desc)
        current_input = step_output

    return "+".join(descriptions)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    "random_cv2d": [
        {"type": "cv2d"},
    ],
    "random_temporal": [
        {"type": "temporal"},
    ],
    "3d_yaw_right": [
        {"type": "3d_view", "name": "yaw_right", "yaw": 0.25, "pitch": 0.0, "zoom": 1.0},
    ],
    "3d_yaw_left": [
        {"type": "3d_view", "name": "yaw_left", "yaw": -0.25, "pitch": 0.0, "zoom": 1.0},
    ],
    "temporal_then_3d": [
        {"type": "temporal", "id": 2},
        {"type": "3d_view", "name": "yaw_right", "yaw": 0.25, "pitch": 0.0, "zoom": 1.0},
    ],
    "3d_then_cv2d": [
        {"type": "3d_view", "name": "yaw_right", "yaw": 0.25, "pitch": 0.0, "zoom": 1.0},
        {"type": "cv2d"},
    ],
}


def _get_sentence_entries(task_id: str) -> list[dict]:
    manifest_path = (
        settings.SHARED_DATA_ROOT / task_id / "phase_2" / "output" / "manifest.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Phase 2 manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        entries = json.load(f)

    sentence_entries = [
        e for e in entries if e.get("filename", "").startswith("sentence_")
    ]
    if not sentence_entries:
        raise ValueError(f"No sentence videos found for task {task_id}")

    return sentence_entries


def get_available_presets() -> dict:
    return {name: steps for name, steps in PRESETS.items()}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_test_video(
    task_id: str,
    job_id: str,
    pipeline: list[dict] | None = None,
    preset: str | None = None,
    gpu_id: int = 0,
    on_progress=None,
) -> dict:
    """Generate a test video by augmenting and concatenating sentence videos."""
    if pipeline is None:
        preset = preset or "random_cv2d"
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
        pipeline = PRESETS[preset]

    sentence_entries = _get_sentence_entries(task_id)
    videos_dir = settings.SHARED_DATA_ROOT / task_id / "phase_2" / "output" / "videos"

    output_dir = settings.SHARED_DATA_ROOT / task_id / "test_videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{job_id}.mp4"

    sentences_timeline = []
    augmented_paths = []
    total_sentences = len(sentence_entries)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        for i, entry in enumerate(sentence_entries):
            src = videos_dir / entry["filename"]
            if not src.exists():
                logger.warning(f"Sentence video not found: {src}, skipping")
                continue

            aug_output = tmp / f"aug_{i}.mp4"
            step_work = tmp / f"work_{i}"
            step_work.mkdir(parents=True, exist_ok=True)

            try:
                desc = _apply_pipeline(
                    src, aug_output, pipeline, step_work, gpu_id,
                )
                augmented_paths.append({
                    "path": aug_output,
                    "sentence_text": entry.get("sentence_text", ""),
                    "aug_desc": desc,
                    "index": i,
                })
                logger.info(f"[{task_id}] Sentence {i+1}/{total_sentences}: {desc}")
            except Exception as e:
                logger.error(f"Augmentation failed for {src.name}: {e}")

            if on_progress:
                on_progress((i + 1) / total_sentences * 0.8)

        if not augmented_paths:
            raise RuntimeError("No sentence videos were successfully augmented")

        # Concatenate with black frame gaps
        ref_cap = cv2.VideoCapture(str(augmented_paths[0]["path"]))
        fps = ref_cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ref_cap.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        total_frames = 0
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)

        for idx, item in enumerate(augmented_paths):
            start_time = total_frames / fps

            cap = cv2.VideoCapture(str(item["path"]))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
                total_frames += 1
            cap.release()

            end_time = total_frames / fps
            sentences_timeline.append({
                "index": item["index"],
                "sentence_text": item["sentence_text"],
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "aug_desc": item["aug_desc"],
            })

            if idx < len(augmented_paths) - 1:
                for _ in range(GAP_FRAMES):
                    writer.write(black_frame)
                    total_frames += 1

        writer.release()

    if on_progress:
        on_progress(0.9)

    reencode_to_h264(output_path)

    logger.info(
        f"Test video generated for task {task_id}: "
        f"{len(sentences_timeline)} sentences, {total_frames} frames"
    )

    return {
        "video_path": str(output_path),
        "fps": fps,
        "sentences": sentences_timeline,
        "total_frames": total_frames,
        "duration": round(total_frames / fps, 3),
        "pipeline": pipeline,
    }
