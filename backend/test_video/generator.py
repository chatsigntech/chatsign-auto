"""Test video generation: augment and concatenate sentence videos."""

import json
import logging
import random
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from backend.config import settings
from backend.core.video_utils import reencode_to_h264

logger = logging.getLogger(__name__)

# Lazy-load augmentation module from guava-aug
_aug_module = None


def _get_aug_module():
    global _aug_module
    if _aug_module is None:
        guava_path = str(settings.GUAVA_AUG_PATH.resolve())
        if guava_path not in sys.path:
            sys.path.insert(0, guava_path)
        from cv_aug.augment import augment_video, AUGMENTATIONS
        _aug_module = {"augment_video": augment_video, "AUGMENTATIONS": AUGMENTATIONS}
    return _aug_module


GAP_FRAMES = 15  # ~0.5s black frames between sentences


def _get_sentence_entries(task_id: str) -> list[dict]:
    """Read Phase 2 manifest and return sentence video entries."""
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
        raise ValueError(f"No sentence videos found in Phase 2 output for task {task_id}")

    return sentence_entries


def generate_test_video(task_id: str, job_id: str, on_progress=None) -> dict:
    """Generate a test video by augmenting and concatenating sentence videos.

    Args:
        on_progress: optional callback(float) with progress in [0, 1].

    Returns dict with video_path, fps, and sentences timeline.
    """
    aug = _get_aug_module()
    augment_video = aug["augment_video"]
    augmentations = aug["AUGMENTATIONS"]

    sentence_entries = _get_sentence_entries(task_id)
    videos_dir = settings.SHARED_DATA_ROOT / task_id / "phase_2" / "output" / "videos"

    # Output directory
    output_dir = settings.SHARED_DATA_ROOT / task_id / "test_videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{job_id}.mp4"

    sentences_timeline = []
    augmented_paths = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Step 1: Augment each sentence video
        for i, entry in enumerate(sentence_entries):
            src = videos_dir / entry["filename"]
            if not src.exists():
                logger.warning(f"Sentence video not found: {src}, skipping")
                continue

            aug_id = random.randint(0, len(augmentations) - 1)
            aug_name = augmentations[aug_id]["name"]
            aug_path = tmp / f"aug_{i}_{aug_name}.mp4"

            try:
                augment_video(str(src), str(aug_path), aug_id)
                augmented_paths.append({
                    "path": aug_path,
                    "sentence_text": entry.get("sentence_text", ""),
                    "aug_name": aug_name,
                    "index": i,
                })
                if on_progress:
                    on_progress((i + 1) / len(sentence_entries) * 0.8)
            except Exception as e:
                logger.error(f"Augmentation failed for {src.name}: {e}")

        if not augmented_paths:
            raise RuntimeError("No sentence videos were successfully augmented")

        # Step 2: Concatenate with black frame gaps
        # Use first video's properties as reference
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
            start_frame = total_frames
            start_time = total_frames / fps

            cap = cv2.VideoCapture(str(item["path"]))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Resize if dimensions differ from reference
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
                "aug_name": item["aug_name"],
            })

            # Insert black gap (except after last sentence)
            if idx < len(augmented_paths) - 1:
                for _ in range(GAP_FRAMES):
                    writer.write(black_frame)
                    total_frames += 1

        writer.release()

    # Step 3: Re-encode to H.264 for browser compatibility
    reencode_to_h264(output_path)

    logger.info(
        f"Test video generated for task {task_id}: "
        f"{len(sentences_timeline)} sentences, {total_frames} frames, {output_path}"
    )

    return {
        "video_path": str(output_path),
        "fps": fps,
        "sentences": sentences_timeline,
        "total_frames": total_frames,
        "duration": round(total_frames / fps, 3),
    }
