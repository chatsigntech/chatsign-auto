"""Phase 4: Video preprocessing using UniSignMimicTurbo scripts.

Pipeline follows the author's run_from_scratch.md workflow:
4.1 Extract frames from videos
4.2 Filter duplicate/static frames
4.3 Filter by pose quality (hand + head detection)
4.4 Resize frames to target resolution
4.5 Generate videos from filtered frames
"""
import logging
from pathlib import Path

from backend.config import settings
from backend.core.subprocess_runner import run_subprocess

logger = logging.getLogger(__name__)

SCRIPTS_DIR = settings.UNISIGN_PATH.resolve() / "scripts" / "sentence"
UNISIGN_CWD = settings.UNISIGN_PATH.resolve()


async def run_phase4(task_id: str, input_dir: Path, output_dir: Path) -> bool:
    """Run 5-step video preprocessing pipeline.

    Args:
        input_dir: Directory containing video files (mp4) from Phase 3
        output_dir: Root output directory for all steps
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 4.1: Extract frames from videos
    step_4_1 = output_dir / "step_4.1_frames"
    step_4_1.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Phase 4 Step 4.1: Extracting frames")
    rc, _, stderr = await run_subprocess(
        [
            "python", str(SCRIPTS_DIR / "extract_all_frames_seq.py"),
            "--mp4-root", str(input_dir),
            "--out-root", str(step_4_1),
        ],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 4 Step 4.1 failed: {stderr}")

    # Step 4.2: Filter duplicate frames
    step_4_2 = output_dir / "step_4.2_dedup"
    step_4_2.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Phase 4 Step 4.2: Filtering duplicates")
    rc, _, stderr = await run_subprocess(
        [
            "python", str(SCRIPTS_DIR / "filter_duplicate_frames.py"),
            "--frames-dir", str(step_4_1),
            "--output-dir", str(step_4_2),
            "--save-cleaned-frames",
            "--duplicate-threshold", "3.0",
            "--min-duplicate-length", "2",
        ],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 4 Step 4.2 failed: {stderr}")

    # Step 4.3: Filter by pose (hand + head detection)
    step_4_3 = output_dir / "step_4.3_pose"
    step_4_3.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Phase 4 Step 4.3: Filtering by pose")
    rc, _, stderr = await run_subprocess(
        [
            "python", str(SCRIPTS_DIR / "filter_frames_by_pose.py"),
            "--frames-dir", str(step_4_2),
            "--output-dir", str(step_4_3),
            "--save-filtered",
            "--hand-threshold", "0.8",
            "--head-threshold", "0.9",
            "--hand-height-threshold", "0.1",
            "--device", "cuda",
        ],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 4 Step 4.3 failed: {stderr}")

    # Step 4.4: Resize frames
    step_4_4 = output_dir / "step_4.4_resized"
    step_4_4.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Phase 4 Step 4.4: Resizing frames")
    rc, _, stderr = await run_subprocess(
        [
            "python", str(SCRIPTS_DIR / "resize_frames.py"),
            "--in-root", str(step_4_3),
            "--out-root", str(step_4_4),
            "--width", "512",
            "--height", "320",
        ],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 4 Step 4.4 failed: {stderr}")

    # Step 4.5: Generate videos from frames
    step_4_5 = output_dir / "step_4.5_videos"
    step_4_5.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Phase 4 Step 4.5: Generating videos")
    rc, _, stderr = await run_subprocess(
        [
            "python", str(SCRIPTS_DIR / "generate_videos_from_frames.py"),
            "--frames-dir", str(step_4_4),
            "--output-dir", str(step_4_5),
            "--fps", "25",
        ],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 4 Step 4.5 failed: {stderr}")

    logger.info(f"[{task_id}] Phase 4 completed: preprocessed videos in {step_4_5}")
    return True
