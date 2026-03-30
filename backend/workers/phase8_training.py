"""Phase 8: Model training using gloss_aware scripts."""
import logging
import os
from pathlib import Path

from backend.config import settings
from backend.core.subprocess_runner import run_subprocess

logger = logging.getLogger(__name__)

STEPS = [
    ("6.1", "preprocess/pose_extractor.py", "Extracting poses"),
    ("6.2", "preprocess/filter_pose_pkls.py", "Filtering pose files"),
    ("6.3", "preprocess/batch_norm_cosign_padding.py", "Normalizing poses"),
]


async def run_phase8_training(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
    gpu_id: int = 0,
) -> bool:
    """Run model training pipeline: preprocess -> train -> build prototypes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    ga_path = settings.GLOSS_AWARE_PATH.resolve()

    # Steps 6.1 - 6.3: Preprocessing
    for step_id, script_rel, description in STEPS:
        logger.info(f"[{task_id}] Phase 8 Step {step_id}: {description}")
        script = ga_path / script_rel
        if not script.exists():
            raise FileNotFoundError(f"Script not found: {script}")

        returncode, stdout, stderr = await run_subprocess(
            ["python", str(script), "--input-dir", str(input_dir), "--output-dir", str(output_dir)],
            cwd=ga_path,
            env=env,
        )
        if returncode != 0:
            raise RuntimeError(f"Phase 8 Step {step_id} failed: {stderr}")

    # Step 8.4: Training (torchrun for distributed)
    logger.info(f"[{task_id}] Phase 8 Step 8.4: Training model")
    train_script = ga_path / "ssl_models_WLASL_advance_v4_v1_noconf.py"
    returncode, stdout, stderr = await run_subprocess(
        [
            "torchrun", "--nproc_per_node=1",
            str(train_script),
            "--data-dir", str(output_dir),
            "--output-dir", str(output_dir / "checkpoints"),
        ],
        cwd=ga_path,
        env=env,
        timeout=3600 * 24,  # 24 hours max
    )
    if returncode != 0:
        raise RuntimeError(f"Phase 8 training failed: {stderr}")

    # Step 8.5: Build prototypes
    logger.info(f"[{task_id}] Phase 8 Step 8.5: Building prototypes")
    proto_script = ga_path / "build_prototypes_asl_clip_nob2b.py"
    returncode, stdout, stderr = await run_subprocess(
        [
            "python", str(proto_script),
            "--checkpoint", str(output_dir / "checkpoints" / "best_cl.pth"),
            "--output-dir", str(output_dir / "prototypes"),
        ],
        cwd=ga_path,
        env=env,
    )
    if returncode != 0:
        raise RuntimeError(f"Phase 8 prototype building failed: {stderr}")

    logger.info(f"[{task_id}] Phase 8 completed")
    return True
