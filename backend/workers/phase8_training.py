"""Phase 8: Model training using gloss_aware scripts."""
import logging
import os
import sys
from pathlib import Path

from backend.config import settings
from backend.core.subprocess_runner import run_subprocess

logger = logging.getLogger(__name__)

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

    # Collect all videos from input (may include augmented videos in subdirs)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    for mp4 in input_dir.rglob("*.mp4"):
        dst = videos_dir / mp4.name
        if not dst.exists():
            try:
                dst.symlink_to(mp4.resolve())
            except OSError:
                import shutil
                shutil.copy2(mp4, dst)

    pose_dir = output_dir / "poses_raw"
    pose_filtered = output_dir / "poses_filtered"
    pose_normed = output_dir / "poses_normed"

    # Step 8.1: Extract poses (positional args: input_dir output_dir)
    logger.info(f"[{task_id}] Phase 8 Step 8.1: Extracting poses")
    script = ga_path / "preprocess" / "pose_extractor.py"
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(script), str(videos_dir), str(pose_dir)],
        cwd=ga_path, env=env, timeout=7200,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 8 Step 8.1 failed: {stderr}")

    # Step 8.2: Filter pose files (--in-dir, --out-dir)
    logger.info(f"[{task_id}] Phase 8 Step 8.2: Filtering pose files")
    script = ga_path / "preprocess" / "filter_pose_pkls.py"
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(script), "--in-dir", str(pose_dir), "--out-dir", str(pose_filtered)],
        cwd=ga_path, env=env,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 8 Step 8.2 failed: {stderr}")

    # Step 8.3: Normalize poses (positional args: input_dir output_dir)
    logger.info(f"[{task_id}] Phase 8 Step 8.3: Normalizing poses")
    script = ga_path / "preprocess" / "batch_norm_cosign_padding.py"
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(script), str(pose_filtered), str(pose_normed)],
        cwd=ga_path, env=env,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 8 Step 8.3 failed: {stderr}")

    # Step 8.4: Training (torchrun for distributed)
    logger.info(f"[{task_id}] Phase 8 Step 8.4: Training model")
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    train_script = ga_path / "ssl_pretraining_crossvideo_mlp_feature_mean_mean_advance_v4_noconf_clip_nob2b.py"
    returncode, stdout, stderr = await run_subprocess(
        [
            str(Path(sys.executable).parent / "torchrun"), "--nproc_per_node=1",
            str(train_script),
            "--dataset", str(pose_normed),
            "--output_dir", str(ckpt_dir),
            "--epochs", "5",
            "--batch-size", "32",
        ],
        cwd=ga_path,
        env=env,
        timeout=3600 * 24,  # 24 hours max
    )
    if returncode != 0:
        raise RuntimeError(f"Phase 8 training failed: {stderr}")

    # Find the best checkpoint
    best_ckpt = ckpt_dir / "best_cl.pth"
    if not best_ckpt.exists():
        ckpts = sorted(ckpt_dir.glob("*.pth"))
        if ckpts:
            best_ckpt = ckpts[-1]
        else:
            logger.warning(f"[{task_id}] No checkpoint produced, skipping prototype building")
            return True

    # Step 8.5: Build prototypes
    logger.info(f"[{task_id}] Phase 8 Step 8.5: Building prototypes")
    proto_script = ga_path / "build_prototypes_asl_clip_nob2b.py"
    returncode, stdout, stderr = await run_subprocess(
        [
            sys.executable, str(proto_script),
            "--ckpt", str(best_ckpt),
            "--dataset", str(pose_normed),
            "--output-dir", str(output_dir / "prototypes"),
        ],
        cwd=ga_path,
        env=env,
    )
    if returncode != 0:
        raise RuntimeError(f"Phase 8 prototype building failed: {stderr}")

    logger.info(f"[{task_id}] Phase 8 completed")
    return True
