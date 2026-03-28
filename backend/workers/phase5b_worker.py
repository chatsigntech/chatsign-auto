"""Phase 6: Data augmentation using guava-aug.

Stage 1 (current): 2D CV + Temporal augmentation (CPU only, no extra deps)
  - 25 types of 2D augmentation (geometric + color)
  - 7 types of temporal augmentation (speed + fps)
  - Total: up to 32 variants per input video

Stage 2 (future): 3D novel view rendering requires pytorch3d + GUAVA model,
  disabled by default until environment is ready.
"""
import json
import logging
import os
import sys
from pathlib import Path

from backend.config import settings

logger = logging.getLogger(__name__)

GUAVA_PATH = settings.GUAVA_AUG_PATH.resolve()


def _find_videos(input_dir: Path) -> list[Path]:
    """Find all video files in a directory (non-recursive)."""
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = []
    if input_dir.exists():
        for f in sorted(input_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                videos.append(f)
    return videos


async def run_2d_augmentation(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
    aug_ids: list[int] | None = None,
) -> int:
    """
    Apply 2D CV augmentations directly using guava-aug functions.

    Args:
        aug_ids: List of augmentation IDs (0-24) to apply.
                 None = apply all 25 types.
    Returns:
        Number of augmented videos generated.
    """
    sys.path.insert(0, str(GUAVA_PATH))
    from cv_aug.augment import augment_video, AUGMENTATIONS

    if aug_ids is None:
        aug_ids = list(range(len(AUGMENTATIONS)))

    videos = _find_videos(input_dir)
    if not videos:
        logger.warning(f"[{task_id}] Phase 6: No videos found in {input_dir}")
        return 0

    cv_output = output_dir / "cv_aug"
    count = 0

    for video_path in videos:
        video_name = video_path.stem
        for aug_id in aug_ids:
            aug_name = AUGMENTATIONS[aug_id]["name"]
            out_path = cv_output / aug_name / f"{video_name}.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists():
                count += 1
                continue

            try:
                augment_video(str(video_path), str(out_path), aug_id, video_name=video_name)
                count += 1
            except Exception as e:
                logger.error(f"[{task_id}] 2D aug {aug_name} failed for {video_name}: {e}")

    logger.info(f"[{task_id}] Phase 6: 2D augmentation done, {count} videos generated")
    return count


async def run_temporal_augmentation(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
    aug_ids: list[int] | None = None,
) -> int:
    """
    Apply temporal augmentations directly using guava-aug functions.

    Args:
        aug_ids: List of temporal augmentation IDs (0-6) to apply.
                 None = apply all 7 types.
    Returns:
        Number of augmented videos generated.
    """
    sys.path.insert(0, str(GUAVA_PATH))
    from cv_aug.temporal_augment import temporal_augment_video, TEMPORAL_AUGMENTATIONS

    if aug_ids is None:
        aug_ids = list(range(len(TEMPORAL_AUGMENTATIONS)))

    videos = _find_videos(input_dir)
    if not videos:
        logger.warning(f"[{task_id}] Phase 6: No videos found in {input_dir}")
        return 0

    temporal_output = output_dir / "temporal_aug"
    count = 0

    for video_path in videos:
        video_name = video_path.stem
        for aug_id in aug_ids:
            aug_name = TEMPORAL_AUGMENTATIONS[aug_id]["name"]
            out_path = temporal_output / aug_name / f"{video_name}.mp4"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists():
                count += 1
                continue

            try:
                temporal_augment_video(str(video_path), str(out_path), aug_id)
                count += 1
            except Exception as e:
                logger.error(f"[{task_id}] Temporal aug {aug_name} failed for {video_name}: {e}")

    logger.info(f"[{task_id}] Phase 6: Temporal augmentation done, {count} videos generated")
    return count


async def run_phase5b(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
    gpu_id: int = 0,
    enable_3d: bool = False,
    enable_2d: bool = True,
    enable_temporal: bool = True,
    cv_aug_ids: list[int] | None = None,
    temporal_aug_ids: list[int] | None = None,
) -> bool:
    """
    Run data augmentation pipeline.

    Args:
        enable_3d: 3D novel view rendering (disabled, requires pytorch3d + GUAVA model)
        enable_2d: 2D CV augmentation (25 types)
        enable_temporal: Temporal augmentation (7 types)
        cv_aug_ids: Specific 2D aug IDs to apply (None = all 25)
        temporal_aug_ids: Specific temporal aug IDs to apply (None = all 7)

    Output structure:
        output_dir/
        ├── cv_aug/
        │   ├── center_crop_80/      (video1.mp4, video2.mp4, ...)
        │   ├── rotate_p5/           (video1.mp4, video2.mp4, ...)
        │   ├── brightness_up/       (video1.mp4, video2.mp4, ...)
        │   └── ... (25 directories)
        ├── temporal_aug/
        │   ├── speed_0.5x/          (video1.mp4, video2.mp4, ...)
        │   ├── speed_2.0x/          (video1.mp4, video2.mp4, ...)
        │   └── ... (7 directories)
        └── manifest.json            (augmentation metadata)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_2d = 0
    total_temporal = 0

    if enable_3d:
        logger.warning(f"[{task_id}] Phase 6: 3D rendering disabled (requires pytorch3d + GUAVA model)")

    if enable_2d:
        total_2d = await run_2d_augmentation(task_id, input_dir, output_dir, aug_ids=cv_aug_ids)

    if enable_temporal:
        total_temporal = await run_temporal_augmentation(task_id, input_dir, output_dir, aug_ids=temporal_aug_ids)

    # Write manifest
    manifest = {
        "input_dir": str(input_dir),
        "input_videos": len(_find_videos(input_dir)),
        "augmentations": {
            "2d_cv": {"enabled": enable_2d, "count": total_2d},
            "temporal": {"enabled": enable_temporal, "count": total_temporal},
            "3d_views": {"enabled": enable_3d, "count": 0},
        },
        "total_generated": total_2d + total_temporal,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"[{task_id}] Phase 6 completed: {total_2d} 2D + {total_temporal} temporal = "
                f"{total_2d + total_temporal} total augmented videos")
    return True
