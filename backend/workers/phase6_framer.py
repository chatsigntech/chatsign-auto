"""Phase 6: FramerTurbo interpolation + combine into final videos.

Uses boundary frames from Phase 5 to generate smooth transitions
between words, then combines cleaned frames + interpolations into
final sentence-level videos.

NOTE: Requires FramerTurbo checkpoint (checkpoints/framer_512x320/).
If checkpoint is not available, this phase is skipped and Phase 5
cleaned videos are passed through directly.
"""
import logging
import os
import shutil
from pathlib import Path

from backend.config import settings
from backend.core.subprocess_runner import run_subprocess

logger = logging.getLogger(__name__)

UNISIGN = settings.UNISIGN_PATH.resolve()
FRAMER_DIR = UNISIGN / "FramerTurbo"
FRAMER_SCRIPT = FRAMER_DIR / "scripts" / "inference" / "cli_infer_576x576.py"
FRAMER_CKPT = FRAMER_DIR / "checkpoints" / "framer_512x320"
COMBINE_SCRIPT = UNISIGN / "scripts" / "sentence" / "combine_frames_and_interp.py"


async def run_phase6_framer(
    task_id: str,
    phase5_output: Path,
    output_dir: Path,
    gpu_id: int = 0,
) -> bool:
    """
    Run FramerTurbo interpolation and combine with cleaned frames.

    If FramerTurbo checkpoint is not available, pass through Phase 5
    cleaned videos directly.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_out = output_dir / "videos"
    videos_out.mkdir(exist_ok=True)

    boundary_dir = phase5_output / "step5_boundary"
    frames_dir = phase5_output / "step4_resized"
    p5_videos = phase5_output / "videos"

    # Check if FramerTurbo is available
    if not FRAMER_CKPT.exists():
        logger.warning(f"[{task_id}] Phase 6: FramerTurbo checkpoint not found at {FRAMER_CKPT}")
        logger.info(f"[{task_id}] Phase 6: Passing through Phase 5 cleaned videos")

        # Copy/link Phase 5 videos directly
        if p5_videos.exists():
            for v in p5_videos.glob("*.mp4"):
                dst = videos_out / v.name
                if not dst.exists():
                    dst.symlink_to(v.resolve())

        count = len(list(videos_out.glob("*.mp4")))
        logger.info(f"[{task_id}] Phase 6 completed (passthrough): {count} videos")
        return True

    # FramerTurbo interpolation
    if boundary_dir.exists() and list(boundary_dir.iterdir()):
        interp_dir = output_dir / "interp_frames"
        interp_dir.mkdir(exist_ok=True)

        logger.info(f"[{task_id}] Phase 6: Running FramerTurbo interpolation")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        rc, _, stderr = await run_subprocess(
            ["python", str(FRAMER_SCRIPT),
             "--input_dir", str(boundary_dir),
             "--model", str(FRAMER_CKPT),
             "--output_dir", str(interp_dir),
             "--scheduler", "euler"],
            cwd=str(FRAMER_DIR),
            env=env,
            timeout=3600 * 4,
        )
        if rc != 0:
            logger.warning(f"[{task_id}] Phase 6: FramerTurbo failed: {stderr[:200]}")
            logger.info(f"[{task_id}] Phase 6: Falling back to Phase 5 videos")
            # Fallback: use Phase 5 videos
            if p5_videos.exists():
                for v in p5_videos.glob("*.mp4"):
                    dst = videos_out / v.name
                    if not dst.exists():
                        dst.symlink_to(v.resolve())
            return True

        # Combine frames + interpolation into final videos
        logger.info(f"[{task_id}] Phase 6: Combining frames and interpolations")
        rc, _, stderr = await run_subprocess(
            ["python", str(COMBINE_SCRIPT),
             "--frames-root", str(frames_dir),
             "--interp-root", str(interp_dir),
             "--out-root", str(videos_out),
             "--fps", "25"],
            cwd=str(UNISIGN),
            timeout=3600,
        )
        if rc != 0:
            logger.warning(f"[{task_id}] Phase 6: Combine failed: {stderr[:200]}")
    else:
        logger.info(f"[{task_id}] Phase 6: No boundary frames, using Phase 5 videos")
        if p5_videos.exists():
            for v in p5_videos.glob("*.mp4"):
                dst = videos_out / v.name
                if not dst.exists():
                    dst.symlink_to(v.resolve())

    count = len(list(videos_out.glob("*.mp4")))
    logger.info(f"[{task_id}] Phase 6 completed: {count} final videos")
    return True
