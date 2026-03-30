"""Phase 4: Person transfer using MimicMotion (on raw original videos).

Per the author's pipeline: transfer FIRST, then filter.
Uses raw videos from Phase 1, not preprocessed ones.
"""
import logging
import os
import subprocess
import sys
from pathlib import Path

from backend.config import settings

logger = logging.getLogger(__name__)

PYTHON = sys.executable
UNISIGN = settings.UNISIGN_PATH.resolve()
SCRIPT = UNISIGN / "scripts" / "inference" / "inference_raw_batch_cache.py"
CONFIG = UNISIGN / "configs" / "test.yaml"


async def run_phase4_transfer(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
    gpu_id: int = 0,
) -> bool:
    """
    Run MimicMotion person transfer on each video individually.
    Skips videos that fail (e.g., too few frames) and continues.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not SCRIPT.exists():
        raise FileNotFoundError(f"MimicMotion script not found: {SCRIPT}")

    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No videos found in {input_dir}")

    logger.info(f"[{task_id}] Phase 4: Person transfer on {len(videos)} videos, GPU {gpu_id}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(UNISIGN)

    success = 0
    failed = 0

    for i, video in enumerate(videos):
        # Skip if already generated
        if list(output_dir.rglob(f"{video.stem}_*.mp4")) or list(output_dir.rglob(f"{video.stem}.mp4")):
            success += 1
            continue

        # Create per-video temp input dir
        tmp_in = Path(f"/tmp/phase4_single_{video.stem}")
        tmp_in.mkdir(exist_ok=True)
        link = tmp_in / video.name
        if not link.exists():
            link.symlink_to(video.resolve())

        logger.info(f"[{task_id}] Phase 4: [{i+1}/{len(videos)}] {video.name}")

        try:
            result = subprocess.run(
                [PYTHON, str(SCRIPT),
                 "--batch_folder", str(tmp_in),
                 "--output_dir", str(output_dir),
                 "--inference_config", str(CONFIG),
                 "--num_inference_steps", "10"],
                cwd=str(UNISIGN),
                env=env,
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                success += 1
            else:
                failed += 1
                logger.warning(f"[{task_id}] Phase 4: {video.name} failed: {result.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            failed += 1
            logger.warning(f"[{task_id}] Phase 4: {video.name} timed out")
        finally:
            link.unlink(missing_ok=True)
            try:
                tmp_in.rmdir()
            except OSError:
                pass

    logger.info(f"[{task_id}] Phase 4 completed: {success} success, {failed} failed out of {len(videos)}")

    generated = list(output_dir.rglob("*.mp4"))
    if not generated:
        raise RuntimeError(f"Phase 4: No videos generated (all {len(videos)} failed)")

    return True
