"""Phase 5: Data augmentation using UniSignMimicTurbo inference_raw_batch_cache.py."""
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
    """Run video augmentation with specified GPU and config."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_path or settings.AUGMENTATION_CONFIG

    script = settings.UNISIGN_PATH / "scripts" / "inference" / "inference_raw_batch_cache.py"
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info(f"[{task_id}] Phase 5: augmenting on GPU {gpu_id}")
    returncode, stdout, stderr = await run_subprocess(
        [
            "python", str(script),
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--inference_config", str(config_path),
        ],
        cwd=settings.UNISIGN_PATH,
        env=env,
        timeout=3600 * 12,  # 12 hours max
    )
    if returncode != 0:
        raise RuntimeError(f"Phase 5 failed: {stderr}")

    logger.info(f"[{task_id}] Phase 5 completed")
    return True
