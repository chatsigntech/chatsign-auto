"""Phase 5: Person transfer using UniSignMimicTurbo.

Takes sign language videos and transfers the motions onto a different
reference person image, generating new videos with the same gestures
but a different identity.
"""
import logging
import os
from pathlib import Path

from backend.config import settings
from backend.core.subprocess_runner import run_subprocess

logger = logging.getLogger(__name__)


async def run_phase5(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
    gpu_id: int = 0,
    config_path: Path = None,
) -> bool:
    """
    Run person transfer: ref_video (sign language) + ref_image (target person) → new video.

    Uses inference_raw_batch_cache.py which processes a batch folder of videos,
    transferring each video's motion onto the reference person image defined in config.

    Args:
        input_dir: Directory containing sign language videos (from Phase 4)
        output_dir: Output directory for generated videos
        gpu_id: GPU device to use
        config_path: YAML config with ref_image_path, model paths, etc.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    unisign = settings.UNISIGN_PATH.resolve()
    config_path = config_path or (unisign / "configs" / "test.yaml")

    script = unisign / "scripts" / "inference" / "inference_raw_batch_cache.py"
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info(f"[{task_id}] Phase 5: Person transfer on GPU {gpu_id}, "
                f"input={input_dir}, config={config_path}")

    returncode, stdout, stderr = await run_subprocess(
        [
            "python", str(script),
            "--batch_folder", str(input_dir),
            "--output_dir", str(output_dir),
            "--inference_config", str(config_path),
        ],
        cwd=unisign,
        env=env,
        timeout=3600 * 12,
    )
    if returncode != 0:
        raise RuntimeError(f"Phase 5 (person transfer) failed: {stderr}")

    logger.info(f"[{task_id}] Phase 5 completed: person transfer done")
    return True
