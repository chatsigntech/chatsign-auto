"""Pre-process videos before MimicMotion: extract frames, dedup, resize to 576, regenerate.

This reduces video resolution and removes redundant frames so MimicMotion
runs much faster (576p instead of 720p/1080p, fewer frames).
"""
import logging
import sys
from pathlib import Path

from backend.config import settings
from backend.core.subprocess_runner import run_subprocess

logger = logging.getLogger(__name__)

SCRIPTS_DIR = settings.UNISIGN_PATH.resolve() / "scripts" / "sentence"
UNISIGN_CWD = settings.UNISIGN_PATH.resolve()


async def preprocess_videos(task_id: str, input_dir: Path, output_dir: Path) -> Path:
    """Pre-process raw videos: extract frames → dedup → resize 576 → regenerate.

    Args:
        input_dir: Directory with raw mp4 videos
        output_dir: Working directory for intermediate outputs

    Returns:
        Path to directory containing preprocessed mp4 videos
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list(input_dir.rglob("*.mp4"))
    if not videos:
        logger.warning(f"[{task_id}] Preprocess: No videos found, skipping")
        return input_dir

    logger.info(f"[{task_id}] Preprocess: {len(videos)} videos from {input_dir}")

    # Organize into per-video subdirs (scripts expect this structure)
    organized = output_dir / "organized"
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

    # Step 1: Extract frames
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Preprocess 1/4: Extracting frames")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "extract_all_frames_seq.py"),
         "--mp4-root", str(organized), "--out-root", str(frames_dir)],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        logger.warning(f"[{task_id}] Preprocess: frame extraction failed, using original videos")
        return input_dir

    # Step 2: Dedup frames
    dedup_dir = output_dir / "dedup"
    dedup_dir.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Preprocess 2/4: Deduplicating frames")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "filter_duplicate_frames.py"),
         "--frames-dir", str(frames_dir), "--output-dir", str(dedup_dir),
         "--save-cleaned-frames",
         "--duplicate-threshold", "3.0", "--min-duplicate-length", "2"],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        logger.warning(f"[{task_id}] Preprocess: dedup failed, using extracted frames")
        dedup_dir = frames_dir

    # Step 3: Resize to 576 (MimicMotion resolution)
    resized_dir = output_dir / "resized_576"
    resized_dir.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Preprocess 3/4: Resizing to 576p")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "resize_frames.py"),
         "--in-root", str(dedup_dir), "--out-root", str(resized_dir),
         "--width", "576", "--height", "576"],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        logger.warning(f"[{task_id}] Preprocess: resize failed, using dedup frames")
        resized_dir = dedup_dir

    # Step 4: Regenerate videos from processed frames
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Preprocess 4/4: Generating preprocessed videos")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "generate_videos_from_frames.py"),
         "--frames-dir", str(resized_dir), "--output-dir", str(videos_dir), "--fps", "25"],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        logger.warning(f"[{task_id}] Preprocess: video generation failed, using original")
        return input_dir

    generated = list(videos_dir.glob("*.mp4"))
    logger.info(f"[{task_id}] Preprocess complete: {len(generated)} videos at 576p")
    return videos_dir
