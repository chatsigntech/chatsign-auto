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
from typing import Literal

import cv2
import numpy as np

from backend.config import settings
from backend.core.video_utils import reencode_to_h264

logger = logging.getLogger(__name__)

GUAVA_PATH = settings.GUAVA_AUG_PATH.resolve()

GAP_FRAMES = 125  # ~5s black frames between sentences (at 25fps)

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


def _run_tracking(input_path: Path, work_dir: Path, gpu_id: int = 0) -> Path:
    """Run EHM-Tracker on a single video. Returns tracked data path."""
    from backend.workers.phase7_augment import _run_ehm_tracking

    track_input = work_dir / "track_input"
    track_input.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(input_path), str(track_input / f"{input_path.stem}.mp4"))

    tracked_dir = work_dir / "tracked"
    tracked_videos = _run_ehm_tracking("test_video", track_input, tracked_dir, gpu_id)

    if not tracked_videos:
        raise RuntimeError(f"EHM-Tracker produced no tracked data for {input_path.stem}")
    return tracked_videos[0]


def _apply_3d_view(
    input_path: Path, output_path: Path, step: dict,
    work_dir: Path, gpu_id: int = 0,
) -> str:
    """Apply 3D viewpoint augmentation via EHM-Tracker + GUAVA render."""
    from backend.workers.phase7_augment import _run_guava_render

    yaw = step.get("yaw", 0.25)
    pitch = step.get("pitch", 0.0)
    zoom = step.get("zoom", 1.0)
    view_name = step.get("name", f"y{yaw}_p{pitch}_z{zoom}")

    tracked_data = _run_tracking(input_path, work_dir, gpu_id)

    viewpoint = {"name": view_name, "yaw": yaw, "pitch": pitch, "zoom": zoom}
    render_dir = work_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)

    rendered = _run_guava_render(
        "test_video", tracked_data, render_dir,
        tracked_data.name, viewpoint, gpu_id,
    )
    if rendered is None:
        raise RuntimeError(f"GUAVA render produced no output for {view_name}")

    shutil.copy2(str(rendered), str(output_path))
    return f"3d_view:{view_name}"


def _apply_identity(
    input_path: Path, output_path: Path, step: dict,
    work_dir: Path, gpu_id: int = 0,
) -> str:
    """Apply identity cross-reenactment: template person + input video motion."""
    from backend.workers.phase7_augment import _run_guava_cross_reenact, TEMPLATES_DIR

    template_dir = step.get("template_dir", "")
    template_path = TEMPLATES_DIR / template_dir
    if not template_path.exists():
        raise FileNotFoundError(f"Identity template not found: {template_path}")

    template_label = Path(template_dir).stem.replace(" ", "_")

    tracked_data = _run_tracking(input_path, work_dir, gpu_id)

    render_dir = work_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)

    rendered = _run_guava_cross_reenact(
        "test_video", tracked_data, template_path, render_dir,
        tracked_data.name, template_label, gpu_id=gpu_id,
    )
    if rendered is None:
        raise RuntimeError(f"Identity cross-reenact failed for template {template_label}")

    shutil.copy2(str(rendered), str(output_path))
    return f"identity:{template_label}"


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
        elif step_type == "identity":
            step_work = work_dir / f"step_{i}_work"
            step_work.mkdir(parents=True, exist_ok=True)
            desc = _apply_identity(
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
    "original": [],
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


AVAILABLE_STEPS = [
    # 2D CV
    {"key": "cv2d_random", "label": "2D Random", "step": {"type": "cv2d"}},
    # Temporal
    {"key": "temporal_random", "label": "Temporal Random", "step": {"type": "temporal"}},
    {"key": "temporal_0.75x", "label": "Speed 0.75x", "step": {"type": "temporal", "id": 1}},
    {"key": "temporal_1.25x", "label": "Speed 1.25x", "step": {"type": "temporal", "id": 2}},
    {"key": "temporal_1.5x", "label": "Speed 1.5x", "step": {"type": "temporal", "id": 3}},
    # 3D View
    {"key": "3d_yaw_right", "label": "3D Yaw Right", "step": {"type": "3d_view", "name": "yaw_right", "yaw": 0.25, "pitch": 0.0, "zoom": 1.0}},
    {"key": "3d_yaw_left", "label": "3D Yaw Left", "step": {"type": "3d_view", "name": "yaw_left", "yaw": -0.25, "pitch": 0.0, "zoom": 1.0}},
    {"key": "3d_pitch_up", "label": "3D Pitch Up", "step": {"type": "3d_view", "name": "pitch_up", "yaw": 0.0, "pitch": -0.25, "zoom": 1.0}},
    {"key": "3d_pitch_down", "label": "3D Pitch Down", "step": {"type": "3d_view", "name": "pitch_down", "yaw": 0.0, "pitch": 0.25, "zoom": 1.0}},
    {"key": "3d_zoom_in", "label": "3D Zoom In", "step": {"type": "3d_view", "name": "zoom_in", "yaw": 0.0, "pitch": 0.0, "zoom": 0.85}},
    {"key": "3d_zoom_out", "label": "3D Zoom Out", "step": {"type": "3d_view", "name": "zoom_out", "yaw": 0.0, "pitch": 0.0, "zoom": 1.15}},
    # Identity cross-reenactment
    {"key": "identity_asian_female", "label": "Identity: Asian Female", "step": {"type": "identity", "template_dir": "01_Asian_Female_White_Shirt_Palms_Front.jpeg"}},
    {"key": "identity_african_female", "label": "Identity: African Female", "step": {"type": "identity", "template_dir": "02_African_Female_Yellow_Top_Palms_Open.jpeg"}},
    {"key": "identity_caucasian_male", "label": "Identity: Caucasian Male", "step": {"type": "identity", "template_dir": "03_Caucasian_Male_Denim_Shirt_Hands_Up.jpeg"}},
    {"key": "identity_latino_male", "label": "Identity: Latino Male", "step": {"type": "identity", "template_dir": "04_Latino_Male_Grey_Sweater_Palms_Vertical.jpeg"}},
    {"key": "identity_elderly_male_me", "label": "Identity: Elderly Male ME", "step": {"type": "identity", "template_dir": "05_elderly_male_middleeastern"}},
    {"key": "identity_elderly_female_na", "label": "Identity: Elderly Female NA", "step": {"type": "identity", "template_dir": "06_elderly_female_nativeamerican"}},
    {"key": "identity_elderly_female_c", "label": "Identity: Elderly Female C", "step": {"type": "identity", "template_dir": "07_elderly_female_caucasian"}},
    {"key": "identity_child_female", "label": "Identity: Child Female", "step": {"type": "identity", "template_dir": "08_child_female_caucasian"}},
]


def get_available_steps() -> list[dict]:
    return AVAILABLE_STEPS


def _find_word_video(videos_dir: Path, gloss: str) -> Path | None:
    p = videos_dir / f"word_{gloss.upper()}.mp4"
    if p.exists():
        return p
    matches = list(videos_dir.glob(f"word_{gloss}.mp4"))
    return matches[0] if matches else None


def _concat_videos_cv2(paths: list[Path], output_path: Path) -> None:
    """Frame-by-frame mp4 concat at the first video's resolution."""
    ref = cv2.VideoCapture(str(paths[0]))
    fps = ref.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref.release()
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    for p in paths:
        cap = cv2.VideoCapture(str(p))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
        cap.release()
    writer.release()


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
    source: Literal["sentence", "gloss"] = "sentence",
) -> dict:
    """Generate a test video by augmenting and concatenating per-sentence sources.

    source="sentence" (default): each sentence_i is its own mp4 from phase_2.
    source="gloss": each sentence_i is a composite of word_{GLOSS}.mp4 clips.
    """
    if pipeline is None:
        preset = preset or "random_cv2d"
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
        pipeline = PRESETS[preset]

    sentence_entries = _get_sentence_entries(task_id)
    videos_dir = settings.SHARED_DATA_ROOT / task_id / "phase_2" / "output" / "videos"

    # Load gloss mapping for GT display
    glosses = {}
    glosses_path = settings.SHARED_DATA_ROOT / task_id / "phase_1" / "output" / "glosses.json"
    if glosses_path.exists():
        with open(glosses_path) as f:
            glosses = json.load(f)

    output_dir = settings.SHARED_DATA_ROOT / task_id / "test_videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{job_id}.mp4"

    sentences_timeline = []
    augmented_paths = []
    total_sentences = len(sentence_entries)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        for i, entry in enumerate(sentence_entries):
            if source == "gloss":
                sentence_text = entry.get("sentence_text", "")
                word_paths = [
                    p for p in (_find_word_video(videos_dir, g)
                                for g in glosses.get(sentence_text, []))
                    if p is not None
                ]
                if not word_paths:
                    logger.warning(f"[{task_id}] No word videos for sentence {i}, skipping")
                    continue
                src = tmp / f"composite_{i}.mp4"
                _concat_videos_cv2(word_paths, src)
            else:
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
                sentence_text = entry.get("sentence_text", "")
                gloss_list = glosses.get(sentence_text, [])
                augmented_paths.append({
                    "path": aug_output,
                    "sentence_text": sentence_text,
                    "expected_gloss": " ".join(gloss_list),
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

            last_frame = None
            cap = cv2.VideoCapture(str(item["path"]))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
                last_frame = frame
                total_frames += 1
            cap.release()

            # Write pause frames (hold last frame) before recording end_time
            # so the pause period falls within this sentence's time range.
            # This keeps recognition results visible during the 5s pause.
            if idx < len(augmented_paths) - 1 and last_frame is not None:
                for _ in range(GAP_FRAMES):
                    writer.write(last_frame)
                    total_frames += 1

            end_time = total_frames / fps
            sentences_timeline.append({
                "index": item["index"],
                "sentence_text": item["sentence_text"],
                "expected_gloss": item.get("expected_gloss", ""),
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "aug_desc": item["aug_desc"],
            })

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


def generate_gloss_test_video(*args, **kwargs) -> dict:
    return generate_test_video(*args, **kwargs, source="gloss")
