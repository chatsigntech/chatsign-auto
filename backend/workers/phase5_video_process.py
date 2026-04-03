"""Phase 6: Post-processing on person-transferred videos.

Steps 1-2 (extract frames, dedup) and resize already done in Phase 5 preprocess.
This phase handles the remaining steps:
6.1 Filter by pose quality (hand + head)
6.2 Resize frames to 512x320 (FramerTurbo target)
6.3 Extract boundary frames (for FramerTurbo interpolation)
6.4 Generate cleaned videos from filtered frames
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
    """Run post-processing pipeline on person-transferred videos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all mp4 from input (MimicMotion output)
    videos = list(input_dir.rglob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No videos found in {input_dir}")
    logger.info(f"[{task_id}] Phase 6: Processing {len(videos)} transferred videos")

    # Organize into per-video subdirs for scripts
    organized = output_dir / "organized_input"
    organized.mkdir(exist_ok=True)
    for v in videos:
        sent_dir = organized / v.stem
        sent_dir.mkdir(exist_ok=True)
        dst = sent_dir / v.name
        if not dst.exists():
            try:
                dst.symlink_to(v.resolve())
            except OSError:
                import shutil
                shutil.copy2(v, dst)

    # Step 6.1: Extract frames (from MimicMotion output)
    step1 = output_dir / "step1_frames"
    step1.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Phase 6.1: Extracting frames")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "extract_all_frames_seq.py"),
         "--mp4-root", str(organized), "--out-root", str(step1)],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 6.1 failed: {stderr}")

    # Step 6.2: Filter by pose quality
    step2 = output_dir / "step2_pose"
    step2.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Phase 6.2: Filtering by pose")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "filter_frames_by_pose.py"),
         "--frames-dir", str(step1), "--output-dir", str(step2),
         "--save-filtered",
         "--hand-threshold", "0.8", "--head-threshold", "0.9",
         "--hand-height-threshold", "0.1", "--device", "cuda"],
        cwd=UNISIGN_CWD,
        timeout=7200,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 6.2 failed: {stderr}")

    # Step 6.3: Resize frames to 512x320 (FramerTurbo target)
    step3 = output_dir / "step3_resized"
    step3.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Phase 6.3: Resizing frames to 512x320")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "resize_frames.py"),
         "--in-root", str(step2), "--out-root", str(step3),
         "--width", "512", "--height", "320"],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 6.3 failed: {stderr}")

    # Step 6.4: Extract boundary frames (for FramerTurbo)
    step4 = output_dir / "step4_boundary"
    step4.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Phase 6.4: Extracting boundary frames")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "extract_boundary_frames.py"),
         "--frames-root", str(step3), "--out-root", str(step4)],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        logger.warning(f"[{task_id}] Phase 6.4 boundary extraction failed (non-critical): {stderr[:200]}")

    # Step 6.5: Generate cleaned videos from filtered frames
    step5 = output_dir / "videos"
    step5.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Phase 6.5: Generating cleaned videos")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "generate_videos_from_frames.py"),
         "--frames-dir", str(step3), "--output-dir", str(step5), "--fps", "25"],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 6.5 failed: {stderr}")

    logger.info(f"[{task_id}] Phase 6 completed")
    return True
