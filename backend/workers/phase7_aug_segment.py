"""Phase 7: Segment augmented sentence videos using Phase 5 split points.

No model inference needed. Reuses the split points from Phase 5:
- 2D CV augmented sentences: directly reuse original split points
- Temporal augmented sentences: scale split points by speed ratio
- 3D view augmented sentences: scale split points by fps ratio (25→30fps)
- Identity augmented sentences: scale split points by fps ratio (25→30fps)

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
    # Directory structure: phase6_output/sentence/{cv_aug,temporal_aug}/<aug_type>/*.mp4
    #                      phase6_output/sentence/{3d_views,identity}/<render_dir>/<video>/*.mp4
    sentence_aug_root = phase6_output / "sentence"

    def _process_flat_aug_dir(aug_dir: Path, aug_type: str):
        """Process cv_aug/temporal_aug: videos in <aug_type>/<aug_name>/*.mp4"""
        nonlocal total_input, total_clips
        is_temporal = "temporal" in aug_type

        for aug_type_dir in sorted(aug_dir.iterdir()):
            if not aug_type_dir.is_dir():
                continue
            for video_path in sorted(aug_type_dir.glob("*.mp4")):
                if video_path.name.endswith(".h264.mp4"):
                    continue
                orig_stem = _find_original_stem(video_path.name, split_points)
                if orig_stem is None:
                    continue

                total_input += 1
                orig_data = split_points[orig_stem]
                segments = orig_data["segments"]

                if is_temporal:
                    speed_ratio = temporal_params.get(video_path.stem, {}).get("speed_ratio", 1.0)
                    segments = _scale_segments(segments, speed_ratio)

                unique_stem = f"{aug_type}_{aug_type_dir.name}_{video_path.stem}"
                clips, _ = cut_video_at_split_points(
                    video_path, segments, aug_segment_dir, unique_stem
                )
                total_clips += len(clips)

                for clip in clips:
                    clip["source_aug_type"] = aug_type
                    clip["original_video"] = orig_stem
                    clip_manifest.append(clip)

    def _process_3d_aug_dir(aug_dir: Path, aug_type: str):
        """Process 3d_views/identity: videos in <render_dir>/<video>/*.mp4

        These videos are rendered at 30fps (vs original 25fps).
        Scale split points by fps ratio to compensate.
        """
        nonlocal total_input, total_clips
        orig_fps = 25.0
        render_fps = 30.0
        fps_ratio = render_fps / orig_fps  # 1.2

        for video_path in sorted(aug_dir.rglob("*.mp4")):
            if video_path.name.endswith(".h264.mp4"):
                continue
            # Only sentence videos, not word
            orig_stem = _find_original_stem(video_path.name, split_points)
            if orig_stem is None:
                continue
            # Only process sentence-origin videos
            if not orig_stem.startswith("sentence_"):
                continue

            total_input += 1
            orig_data = split_points[orig_stem]
            segments = orig_data["segments"]

            # Scale split points for fps difference: same frame count at higher fps = shorter duration
            # Original 1.0s at 25fps → 30fps output = 1.0/1.2 = 0.833s
            segments = _scale_segments(segments, fps_ratio)

            # Build unique stem from directory structure
            rel = video_path.relative_to(aug_dir)
            unique_stem = f"{aug_type}_{'_'.join(rel.parent.parts)}_{video_path.stem}"
            # Truncate if too long
            if len(unique_stem) > 200:
                import hashlib
                h = hashlib.md5(unique_stem.encode()).hexdigest()[:12]
                unique_stem = f"{unique_stem[:180]}_{h}"

            clips, _ = cut_video_at_split_points(
                video_path, segments, aug_segment_dir, unique_stem
            )
            total_clips += len(clips)

            for clip in clips:
                clip["source_aug_type"] = aug_type
                clip["original_video"] = orig_stem
                clip_manifest.append(clip)

    if sentence_aug_root.exists():
        for subdir_name in ("cv_aug", "temporal_aug"):
            subdir = sentence_aug_root / subdir_name
            if subdir.exists():
                _process_flat_aug_dir(subdir, subdir_name)

        for subdir_name in ("3d_views", "identity"):
            subdir = sentence_aug_root / subdir_name
            if subdir.exists():
                _process_3d_aug_dir(subdir, subdir_name)

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
