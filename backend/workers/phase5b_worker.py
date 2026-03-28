"""Phase 5B: Data augmentation using guava-aug.

Applies 3 types of augmentation to expand the training dataset:
1. 3D novel view rendering (yaw/pitch/zoom) - requires GPU + GUAVA model
2. 2D CV augmentation (geometric + color, 25 types) - CPU only
3. Temporal augmentation (speed/fps changes, 7 types) - CPU only
"""
import logging
import os
from pathlib import Path

from backend.config import settings
from backend.core.subprocess_runner import run_subprocess

logger = logging.getLogger(__name__)

GUAVA_PATH = settings.GUAVA_AUG_PATH


async def run_3d_augmentation(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
    gpu_id: int = 0,
    viewpoints: list[dict] | None = None,
) -> bool:
    """
    Generate novel view renderings using GUAVA 3D Gaussian Avatar.

    Args:
        viewpoints: List of dicts with keys: yaw, pitch, zoom.
                    Defaults to a standard set of viewpoints.
    """
    if viewpoints is None:
        viewpoints = [
            {"yaw": 0.15, "pitch": 0.0, "zoom": 1.0},    # slight right
            {"yaw": -0.15, "pitch": 0.0, "zoom": 1.0},   # slight left
            {"yaw": 0.0, "pitch": 0.10, "zoom": 1.0},    # slight up
            {"yaw": 0.0, "pitch": -0.10, "zoom": 1.0},   # slight down
            {"yaw": 0.0, "pitch": 0.0, "zoom": 0.85},    # closer
            {"yaw": 0.0, "pitch": 0.0, "zoom": 1.15},    # further
        ]

    script = GUAVA_PATH / "main" / "test.py"
    if not script.exists():
        logger.warning(f"[{task_id}] Phase 5B: GUAVA test.py not found, skipping 3D augmentation")
        return False

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    for i, vp in enumerate(viewpoints):
        vp_name = f"yaw{vp['yaw']:.2f}_pitch{vp['pitch']:.2f}_zoom{vp['zoom']:.2f}"
        vp_output = output_dir / "3d_views" / vp_name
        vp_output.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{task_id}] Phase 5B: 3D view {i+1}/{len(viewpoints)}: {vp_name}")

        returncode, stdout, stderr = await run_subprocess(
            [
                "python", str(script),
                "-d", str(gpu_id),
                "-m", "assets/GUAVA",
                "-s", str(vp_output),
                "--data_path", str(input_dir),
                "--skip_self_act",
                "--render_fixed_viewpoint",
                "--fixed_yaw", str(vp['yaw']),
                "--fixed_pitch", str(vp['pitch']),
                "--fixed_zoom", str(vp['zoom']),
            ],
            cwd=GUAVA_PATH,
            env=env,
            timeout=3600 * 4,
        )
        if returncode != 0:
            logger.error(f"[{task_id}] Phase 5B: 3D view {vp_name} failed: {stderr[:200]}")

    return True


async def run_2d_augmentation(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
) -> bool:
    """Apply 2D CV augmentations (25 types: geometric + color)."""
    script = GUAVA_PATH / "cv_aug" / "run_cv_aug.py"
    if not script.exists():
        logger.warning(f"[{task_id}] Phase 5B: run_cv_aug.py not found, skipping 2D augmentation")
        return False

    cv_output = output_dir / "cv_aug"
    cv_output.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{task_id}] Phase 5B: 2D augmentation (25 types)")

    returncode, stdout, stderr = await run_subprocess(
        [
            "python", str(script),
            "--input_dir", str(input_dir),
            "--output_dir", str(cv_output),
        ],
        cwd=GUAVA_PATH,
        timeout=3600 * 2,
    )
    if returncode != 0:
        raise RuntimeError(f"Phase 5B (2D augmentation) failed: {stderr}")

    logger.info(f"[{task_id}] Phase 5B: 2D augmentation completed")
    return True


async def run_temporal_augmentation(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
) -> bool:
    """Apply temporal augmentations (7 types: speed + fps changes)."""
    script = GUAVA_PATH / "cv_aug" / "run_temporal_aug.py"
    if not script.exists():
        logger.warning(f"[{task_id}] Phase 5B: run_temporal_aug.py not found, skipping temporal augmentation")
        return False

    temporal_output = output_dir / "temporal_aug"
    temporal_output.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{task_id}] Phase 5B: Temporal augmentation (7 types)")

    returncode, stdout, stderr = await run_subprocess(
        [
            "python", str(script),
            "--input_dir", str(input_dir),
            "--output_dir", str(temporal_output),
        ],
        cwd=GUAVA_PATH,
        timeout=3600 * 2,
    )
    if returncode != 0:
        raise RuntimeError(f"Phase 5B (temporal augmentation) failed: {stderr}")

    logger.info(f"[{task_id}] Phase 5B: Temporal augmentation completed")
    return True


async def run_phase5b(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
    gpu_id: int = 0,
    enable_3d: bool = True,
    enable_2d: bool = True,
    enable_temporal: bool = True,
) -> bool:
    """
    Run all augmentation types.

    Output structure:
        output_dir/
        ├── 3d_views/          (novel viewpoint renderings)
        │   ├── yaw0.15_pitch0.00_zoom1.00/
        │   └── ...
        ├── cv_aug/            (2D geometric + color augmentations)
        └── temporal_aug/      (speed + fps augmentations)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if enable_3d:
        await run_3d_augmentation(task_id, input_dir, output_dir, gpu_id=gpu_id)

    if enable_2d:
        await run_2d_augmentation(task_id, input_dir, output_dir)

    if enable_temporal:
        await run_temporal_augmentation(task_id, input_dir, output_dir)

    logger.info(f"[{task_id}] Phase 5B: All augmentation completed")
    return True
