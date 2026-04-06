"""Phase 7: Segment augmented sentence videos using Phase 5 split points.

No model inference needed. Reuses the split points from Phase 5:
- 2D CV augmented sentences: directly reuse original split points
- Temporal augmented sentences: scale split points by speed ratio

Input:  Phase 6 augmented sentence videos + Phase 5 split_points.json + Phase 6 temporal_params.json
Output: aug_segment_videos/ (word-level clips from augmented sentences) + labels
"""
import json
import logging
from pathlib import Path

from backend.core.video_utils import cut_video_at_split_points

logger = logging.getLogger(__name__)


def _scale_segments(segments: list[dict], speed_ratio: float) -> list[dict]:
    """Scale segment boundaries by a speed ratio.

    If speed_ratio > 1.0, the video plays faster (shorter duration).
    If speed_ratio < 1.0, the video plays slower (longer duration).
    Split points scale inversely with speed.
    """
    scaled = []
    for seg in segments:
        scaled.append({
            **seg,
            "start": seg.get("start", 0.0) / speed_ratio,
            "end": seg.get("end", 0.0) / speed_ratio,
        })
    return scaled


def _find_original_stem(aug_filename: str, split_points: dict) -> str | None:
    """Find which original video an augmented video was derived from.

    Augmented filenames follow patterns like:
      cv_aug_rotation_<orig_stem>.mp4
      temporal_aug_slow_motion_<orig_stem>.mp4
    """
    stem = Path(aug_filename).stem
    for orig_stem in split_points:
        if orig_stem in stem:
            return orig_stem
    return None


async def run_phase7_aug_segment(
    task_id: str,
    phase6_output: Path,
    phase5_output: Path,
    output_dir: Path,
) -> dict:
    """Segment augmented sentence videos using Phase 5 split points."""
    output_dir = output_dir.resolve()
    phase6_output = phase6_output.resolve()
    phase5_output = phase5_output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load split points from Phase 5
    split_points_path = phase5_output / "split_points.json"
    if not split_points_path.exists():
        raise FileNotFoundError(f"Phase 5 split_points.json not found: {split_points_path}")
    with open(split_points_path) as f:
        split_points = json.load(f)

    if not split_points:
        logger.warning(f"[{task_id}] Phase 7: No split points available, skipping")
        return {"input_videos": 0, "output_clips": 0}

    # Load temporal transform parameters from Phase 6
    temporal_params_path = phase6_output / "temporal_params.json"
    temporal_params = {}
    if temporal_params_path.exists():
        with open(temporal_params_path) as f:
            temporal_params = json.load(f)

    aug_segment_dir = output_dir / "aug_segment_videos"
    aug_segment_dir.mkdir(parents=True, exist_ok=True)

    total_input = 0
    total_clips = 0
    clip_manifest = []

    # Process augmented sentence videos from Phase 6
    # Look for sentence augmentation subdirectories
    aug_sentence_dirs = []
    for subdir_name in ("sentence_cv_aug", "sentence_temporal_aug"):
        subdir = phase6_output / subdir_name
        if subdir.exists():
            aug_sentence_dirs.append(subdir)

    # Also check flat augmentation dirs that contain sentence videos
    for subdir_name in ("cv_aug", "temporal_aug"):
        subdir = phase6_output / subdir_name
        if subdir.exists():
            aug_sentence_dirs.append(subdir)

    for aug_dir in aug_sentence_dirs:
        is_temporal = "temporal" in aug_dir.name

        for video_path in sorted(aug_dir.glob("*.mp4")):
            orig_stem = _find_original_stem(video_path.name, split_points)
            if orig_stem is None:
                continue

            total_input += 1
            orig_data = split_points[orig_stem]
            segments = orig_data["segments"]

            if is_temporal:
                # Get speed ratio for this specific augmented video
                speed_ratio = temporal_params.get(video_path.stem, {}).get("speed_ratio", 1.0)
                segments = _scale_segments(segments, speed_ratio)

            clips, _ = cut_video_at_split_points(
                video_path, segments, aug_segment_dir, video_path.stem
            )
            total_clips += len(clips)

            for clip in clips:
                clip["source_aug_type"] = aug_dir.name
                clip["original_video"] = orig_stem
                clip_manifest.append(clip)

    # Save manifest
    manifest_path = output_dir / "aug_segment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "input_aug_sentences": total_input,
            "output_clips": total_clips,
            "clips": clip_manifest,
        }, f, indent=2, ensure_ascii=False)

    logger.info(
        f"[{task_id}] Phase 7 completed: {total_input} augmented sentences -> "
        f"{total_clips} segment clips"
    )

    return {
        "input_aug_sentences": total_input,
        "output_clips": total_clips,
    }
