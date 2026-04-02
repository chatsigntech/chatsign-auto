"""Phase 5: Video processing on person-transferred videos.

Per the author's pipeline, this runs AFTER person transfer:
5.1 Extract frames
5.2 Filter duplicate/static frames
5.3 Filter by pose quality (hand + head)
5.4 Resize frames
5.5 Extract boundary frames (for FramerTurbo interpolation)
5.6 Generate cleaned videos from filtered frames
"""
import logging
import sys
from pathlib import Path

from backend.config import settings
from backend.core.subprocess_runner import run_subprocess

logger = logging.getLogger(__name__)

SCRIPTS_DIR = settings.UNISIGN_PATH.resolve() / "scripts" / "sentence"
UNISIGN_CWD = settings.UNISIGN_PATH.resolve()


async def run_phase5_process(task_id: str, input_dir: Path, output_dir: Path) -> bool:
    """Run video processing pipeline on person-transferred videos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all mp4 from input (may be in subdirs from MimicMotion output)
    videos = list(input_dir.rglob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No videos found in {input_dir}")
    logger.info(f"[{task_id}] Phase 5: Processing {len(videos)} transferred videos")

    # If videos are in subdirs, reorganize into sentence-level dirs for scripts
    organized = output_dir / "organized_input"
    organized.mkdir(exist_ok=True)
    for v in videos:
        # Create per-video subdir (scripts expect subdirectories)
        sent_dir = organized / v.stem
        sent_dir.mkdir(exist_ok=True)
        dst = sent_dir / v.name
        if not dst.exists():
            dst.symlink_to(v.resolve())

    # Step 5.1: Extract frames
    step1 = output_dir / "step1_frames"
    step1.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Phase 5.1: Extracting frames")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "extract_all_frames_seq.py"),
         "--mp4-root", str(organized), "--out-root", str(step1)],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 5.1 failed: {stderr}")

    # Step 5.2: Filter duplicate frames
    step2 = output_dir / "step2_dedup"
    step2.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Phase 5.2: Filtering duplicates")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "filter_duplicate_frames.py"),
         "--frames-dir", str(step1), "--output-dir", str(step2),
         "--save-cleaned-frames",
         "--duplicate-threshold", "3.0", "--min-duplicate-length", "2"],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 5.2 failed: {stderr}")

    # Step 5.3: Filter by pose quality
    step3 = output_dir / "step3_pose"
    step3.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Phase 5.3: Filtering by pose")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "filter_frames_by_pose.py"),
         "--frames-dir", str(step2), "--output-dir", str(step3),
         "--save-filtered",
         "--hand-threshold", "0.8", "--head-threshold", "0.9",
         "--hand-height-threshold", "0.1", "--device", "cuda"],
        cwd=UNISIGN_CWD,
        timeout=7200,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 5.3 failed: {stderr}")

    # Step 5.4: Resize frames
    step4 = output_dir / "step4_resized"
    step4.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Phase 5.4: Resizing frames")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "resize_frames.py"),
         "--in-root", str(step3), "--out-root", str(step4),
         "--width", "512", "--height", "320"],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 5.4 failed: {stderr}")

    # Step 5.5: Extract boundary frames (for FramerTurbo)
    step5 = output_dir / "step5_boundary"
    step5.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Phase 5.5: Extracting boundary frames")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "extract_boundary_frames.py"),
         "--frames-root", str(step4), "--out-root", str(step5)],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        logger.warning(f"[{task_id}] Phase 5.5 boundary extraction failed (non-critical): {stderr[:200]}")

    # Step 5.6: Generate cleaned videos from filtered frames
    step6 = output_dir / "videos"
    step6.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Phase 5.6: Generating cleaned videos")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "generate_videos_from_frames.py"),
         "--frames-dir", str(step4), "--output-dir", str(step6), "--fps", "25"],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 5.6 failed: {stderr}")

    logger.info(f"[{task_id}] Phase 5 completed")
    return True
