"""Phase 6: Data augmentation using guava-aug.

Stage 1 (current): 2D CV + Temporal augmentation (CPU only, no extra deps)
  - 25 types of 2D augmentation (geometric + color)
  - 7 types of temporal augmentation (speed + fps)
  - Total: up to 32 variants per input video

Stage 2 (future): 3D novel view rendering requires pytorch3d + GUAVA model,
  disabled by default until environment is ready.
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

from backend.config import settings

logger = logging.getLogger(__name__)

GUAVA_PATH = settings.GUAVA_AUG_PATH.resolve()

# Add guava-aug to Python path once at module level
if str(GUAVA_PATH) not in sys.path:
    sys.path.insert(0, str(GUAVA_PATH))

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _find_videos(input_dir: Path) -> list[Path]:
    """Find all video files in a directory (non-recursive)."""
    if not input_dir.exists():
        return []
    return [f for f in sorted(input_dir.iterdir())
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS]


def _run_2d_augmentation(
    task_id: str,
    videos: list[Path],
    output_dir: Path,
    aug_ids: list[int] | None = None,
) -> int:
    """Apply 2D CV augmentations (synchronous, CPU-bound)."""
    from cv_aug.augment import augment_video, AUGMENTATIONS

    if aug_ids is None:
        aug_ids = list(range(len(AUGMENTATIONS)))

    if not videos:
        return 0

    cv_output = output_dir / "cv_aug"
    # Pre-create directories once per aug type
    for aug_id in aug_ids:
        (cv_output / AUGMENTATIONS[aug_id]["name"]).mkdir(parents=True, exist_ok=True)

    count = 0
    for video_path in videos:
        video_name = video_path.stem
        for aug_id in aug_ids:
            aug_name = AUGMENTATIONS[aug_id]["name"]
            out_path = cv_output / aug_name / f"{video_name}.mp4"

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


def _run_temporal_augmentation(
    task_id: str,
    videos: list[Path],
    output_dir: Path,
    aug_ids: list[int] | None = None,
) -> int:
    """Apply temporal augmentations (synchronous, CPU-bound)."""
    from cv_aug.temporal_augment import temporal_augment_video, TEMPORAL_AUGMENTATIONS

    if aug_ids is None:
        aug_ids = list(range(len(TEMPORAL_AUGMENTATIONS)))

    if not videos:
        return 0

    temporal_output = output_dir / "temporal_aug"
    for aug_id in aug_ids:
        (temporal_output / TEMPORAL_AUGMENTATIONS[aug_id]["name"]).mkdir(parents=True, exist_ok=True)

    count = 0
    for video_path in videos:
        video_name = video_path.stem
        for aug_id in aug_ids:
            aug_name = TEMPORAL_AUGMENTATIONS[aug_id]["name"]
            out_path = temporal_output / aug_name / f"{video_name}.mp4"

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
    """Run data augmentation pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = _find_videos(input_dir)
    total_2d = 0
    total_temporal = 0

    if enable_3d:
        logger.warning(f"[{task_id}] Phase 6: 3D rendering disabled (requires pytorch3d + GUAVA model)")

    # Run CPU-bound augmentations in thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()

    if enable_2d:
        total_2d = await loop.run_in_executor(
            None, _run_2d_augmentation, task_id, videos, output_dir, cv_aug_ids
        )

    if enable_temporal:
        total_temporal = await loop.run_in_executor(
            None, _run_temporal_augmentation, task_id, videos, output_dir, temporal_aug_ids
        )

    manifest = {
        "input_dir": str(input_dir),
        "input_videos": len(videos),
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
