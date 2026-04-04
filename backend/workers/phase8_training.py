"""Phase 8: Model training using gloss_aware scripts.

Pipeline:
  8.1  Extract poses from augmented videos (RTMPose)
  8.2  Filter low-quality pose files
  8.3  Normalize pose keypoints
  8.4  Generate train/dev JSONL + register dataset in gloss_aware config
  8.5  Self-supervised pre-training (torchrun)
  8.6  Build gloss prototypes
"""
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from backend.config import settings
from backend.core.subprocess_runner import run_subprocess

logger = logging.getLogger(__name__)

DATASET_NAME = "Pipeline"  # registered name in gloss_aware/config.py


def _build_video_gloss_map(task_data_root: Path) -> dict[str, list[str]]:
    """Build a mapping from original video stem → gloss tokens.

    Reads Phase 3 annotations.json which contains:
      [{"filename": "xxx.mp4", "glosses": ["WORD"]}]
    """
    ann_path = task_data_root / "phase_4" / "output" / "annotations.json"
    mapping = {}
    if ann_path.exists():
        with open(ann_path) as f:
            for entry in json.load(f):
                stem = Path(entry["filename"]).stem
                mapping[stem] = entry.get("glosses", [])

    # Fallback: Phase 1 manifest
    if not mapping:
        manifest = task_data_root / "phase_1" / "output" / "manifest.json"
        if manifest.exists():
            with open(manifest) as f:
                for entry in json.load(f):
                    stem = Path(entry["filename"]).stem
                    text = entry.get("sentence_text", "")
                    mapping[stem] = [w.upper() for w in text.split()] if text else []

    return mapping


def _match_gloss_for_augmented(pkl_stem: str, gloss_map: dict[str, list[str]]) -> list[str]:
    """Match an augmented video's pkl filename back to its original gloss.

    Augmented filenames contain the original stem as a prefix, e.g.:
      school_unmatch_4_en_1_..._20260402113056.pkl          (Phase 4 output)
      school_unmatch_4_en_1_..._fixed_viewpoint_video.pkl   (3D augmented)
      school_unmatch_4_en_1_..._speed_050x.pkl              (temporal aug)
    """
    for orig_stem, glosses in gloss_map.items():
        if orig_stem in pkl_stem:
            return glosses
    return []


def _generate_dataset_jsonl(
    pose_dir: Path,
    gloss_map: dict[str, list[str]],
    output_dir: Path,
    dev_ratio: float = 0.1,
) -> tuple[Path, Path]:
    """Generate train.jsonl and dev.jsonl from normalized pose pkl files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for pkl in sorted(pose_dir.glob("*.pkl")):
        stem = pkl.stem
        tokens = _match_gloss_for_augmented(stem, gloss_map)
        if not tokens:
            logger.warning(f"No gloss found for {stem}, skipping")
            continue
        entries.append({
            "utterance_id": stem,
            "tokens": tokens,
            "pose_path": str(pkl),
        })

    if not entries:
        total_pkls = len(list(pose_dir.glob("*.pkl")))
        raise RuntimeError(
            f"No valid entries for training dataset. "
            f"Poses: {total_pkls} normalized pkl files, "
            f"Gloss map: {len(gloss_map)} video→gloss mappings. "
            f"No pose filenames matched any gloss map entries."
        )

    # Split train/dev
    n_dev = max(1, int(len(entries) * dev_ratio))
    dev_entries = entries[:n_dev]
    train_entries = entries[n_dev:] if len(entries) > n_dev else entries

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"

    for path, data in [(train_path, train_entries), (dev_path, dev_entries)]:
        with open(path, "w") as f:
            for e in data:
                f.write(json.dumps(e) + "\n")

    logger.info(f"Dataset JSONL: {len(train_entries)} train, {len(dev_entries)} dev")
    return train_path, dev_path


def _register_dataset(
    ga_path: Path,
    dataset_name: str,
    train_jsonl: Path,
    dev_jsonl: Path,
    pose_dir: Path,
):
    """Dynamically add our dataset to gloss_aware/config.py."""
    config_path = ga_path / "config.py"
    config_text = config_path.read_text()

    # Check if already registered
    if f'"{dataset_name}"' in config_text:
        # Update existing entries
        import re
        config_text = re.sub(
            rf'"{dataset_name}":\s*"[^"]*"(,?\s*\n\s*}})',
            f'"{dataset_name}": "{train_jsonl}"\\1',
            config_text,
        )
    else:
        # Insert into train_label_paths
        config_text = config_text.replace(
            'train_label_paths = {',
            f'train_label_paths = {{\n'
            f'                    "{dataset_name}": "{train_jsonl}",',
        )
        # Insert into dev_label_paths
        config_text = config_text.replace(
            'dev_label_paths = {',
            f'dev_label_paths = {{\n'
            f'                    "{dataset_name}": "{dev_jsonl}",',
        )
        # Insert into pose_dirs
        config_text = config_text.replace(
            'pose_dirs = {',
            f'pose_dirs = {{\n'
            f'            "{dataset_name}": "{pose_dir}",',
        )

    config_path.write_text(config_text)
    logger.info(f"Registered dataset '{dataset_name}' in {config_path}")


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
    # Add cuDNN/CUDA libs to LD_LIBRARY_PATH for ONNX Runtime GPU support
    nvidia_lib = Path(sys.executable).parent.parent / "lib" / "python3.10" / "site-packages" / "nvidia"
    cudnn_lib = nvidia_lib / "cudnn" / "lib"
    cuda_lib = nvidia_lib / "cuda_runtime" / "lib"
    extra_paths = ":".join(str(p) for p in [cudnn_lib, cuda_lib] if p.exists())
    if extra_paths:
        env["LD_LIBRARY_PATH"] = extra_paths + ":" + env.get("LD_LIBRARY_PATH", "")
    ga_path = settings.GLOSS_AWARE_PATH.resolve()

    # Resolve task data root: input_dir is phase_7/output, so root is 2 levels up
    task_data_root = input_dir.parent.parent
    if not (task_data_root / "phase_4").exists():
        # Fallback: try 3 levels up (in case input_dir is phase_8/input)
        task_data_root = input_dir.parent.parent.parent

    # Collect all videos from input (may include augmented videos in subdirs)
    # Use subdir name as prefix to avoid filename collisions across aug types
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    for mp4 in input_dir.rglob("*.mp4"):
        rel = mp4.relative_to(input_dir)
        if len(rel.parts) > 1:
            prefix = "_".join(rel.parts[:-1])
            unique_name = f"{prefix}_{mp4.name}"
        else:
            unique_name = mp4.name
        dst = videos_dir / unique_name
        if not dst.exists():
            try:
                dst.symlink_to(mp4.resolve())
            except OSError:
                shutil.copy2(mp4, dst)

    total_videos = len(list(videos_dir.glob("*.mp4")))
    logger.info(f"[{task_id}] Phase 9: Collected {total_videos} videos for training")

    if total_videos == 0:
        raise RuntimeError("Phase 9: No videos found in augmentation output")

    pose_dir = output_dir / "poses_raw"
    pose_filtered = output_dir / "poses_filtered"
    pose_normed = output_dir / "poses_normed"

    # Step 9.1: Extract poses (log_to_file=True to avoid PIPE deadlock on large output)
    logger.info(f"[{task_id}] Phase 9 Step 9.1: Extracting poses from {total_videos} videos")
    script = ga_path / "preprocess" / "pose_extractor.py"
    rc, stdout, stderr = await run_subprocess(
        [sys.executable, str(script), str(videos_dir.resolve()), str(pose_dir.resolve())],
        cwd=ga_path, env=env, timeout=7200, log_to_file=True,
    )
    poses_raw_count = len(list(pose_dir.glob("*.pkl"))) if pose_dir.exists() else 0
    logger.info(f"[{task_id}] Phase 9 Step 9.1: {poses_raw_count}/{total_videos} poses extracted")
    if rc != 0:
        raise RuntimeError(f"Phase 9 Step 9.1 pose extraction failed ({poses_raw_count}/{total_videos} extracted): {stderr[-500:]}")
    if poses_raw_count == 0:
        raise RuntimeError(
            f"Phase 9 Step 9.1: Pose extraction produced 0 results from {total_videos} videos. "
            f"Possible causes: videos too short, no person detected, or invalid video format."
        )

    # Step 9.2: Filter pose files
    logger.info(f"[{task_id}] Phase 9 Step 9.2: Filtering {poses_raw_count} pose files")
    script = ga_path / "preprocess" / "filter_pose_pkls.py"
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(script), "--in-dir", str(pose_dir.resolve()), "--out-dir", str(pose_filtered.resolve())],
        cwd=ga_path, env=env,
    )
    poses_filtered_count = len(list(pose_filtered.glob("*.pkl"))) if pose_filtered.exists() else 0
    logger.info(f"[{task_id}] Phase 9 Step 9.2: {poses_filtered_count}/{poses_raw_count} poses passed quality filter")
    if rc != 0:
        raise RuntimeError(f"Phase 9 Step 9.2 pose filter failed: {stderr[-500:]}")
    if poses_filtered_count == 0:
        raise RuntimeError(
            f"Phase 9 Step 9.2: All {poses_raw_count} poses filtered out. "
            f"Videos may lack visible hands or face (hand_threshold=0.8, head_threshold=0.9)."
        )

    # Step 9.3: Normalize poses
    logger.info(f"[{task_id}] Phase 9 Step 9.3: Normalizing {poses_filtered_count} poses")
    script = ga_path / "preprocess" / "batch_norm_cosign_padding.py"
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(script), str(pose_filtered.resolve()), str(pose_normed.resolve())],
        cwd=ga_path, env=env,
    )
    poses_normed_count = len(list(pose_normed.glob("*.pkl"))) if pose_normed.exists() else 0
    logger.info(f"[{task_id}] Phase 9 Step 9.3: {poses_normed_count}/{poses_filtered_count} poses normalized")
    if rc != 0:
        raise RuntimeError(f"Phase 9 Step 9.3 normalization failed: {stderr[-500:]}")

    # Step 9.4: Validate pkl files and remove corrupt ones
    logger.info(f"[{task_id}] Phase 9 Step 9.4: Validating pose pkl files")
    corrupt_files = []
    for pkl in list(pose_normed.glob("*.pkl")):
        try:
            import pickle
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            # Basic sanity check: must be a dict with expected keys
            if not isinstance(data, dict):
                raise ValueError(f"Expected dict, got {type(data)}")
        except Exception as e:
            corrupt_files.append({"file": pkl.name, "error": str(e)})
            pkl.unlink()
    poses_valid_count = len(list(pose_normed.glob("*.pkl")))
    if corrupt_files:
        logger.warning(f"[{task_id}] Phase 9 Step 9.4: Removed {len(corrupt_files)} corrupt pkl files, "
                       f"{poses_valid_count} valid remaining")
        # Save corrupt file list for frontend display
        with open(output_dir / "corrupt_poses.json", "w") as f:
            json.dump(corrupt_files, f, indent=2)
    else:
        logger.info(f"[{task_id}] Phase 9 Step 9.4: All {poses_valid_count} pkl files valid")

    if poses_valid_count == 0:
        raise RuntimeError("Phase 9 Step 9.4: All pose files are corrupt")

    # Step 9.5: Generate dataset JSONL and register
    logger.info(f"[{task_id}] Phase 9 Step 9.5: Generating dataset index")
    gloss_map = _build_video_gloss_map(task_data_root)
    if not gloss_map:
        raise RuntimeError("No video→gloss mapping found from Phase 3 annotations")

    dataset_name = f"{DATASET_NAME}_{task_id}"

    # Put dataset files where the training script expects: data/{dataset_name}/
    dataset_dir = ga_path / "data" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl, dev_jsonl = _generate_dataset_jsonl(
        pose_normed, gloss_map, dataset_dir,
    )

    # Build vocab.json from all glosses
    all_tokens = set()
    for glosses in gloss_map.values():
        all_tokens.update(glosses)
    token_to_id = {"<pad>": 0, "<unk>": 1}
    for i, tok in enumerate(sorted(all_tokens), start=2):
        token_to_id[tok] = i
    vocab_data = {"token_to_id": token_to_id, "id_to_token": {v: k for k, v in token_to_id.items()}}
    with open(dataset_dir / "vocab.json", "w") as f:
        json.dump(vocab_data, f, indent=2)
    logger.info(f"Vocab: {len(token_to_id)} tokens")

    # Also copy jsonl to output for reference
    shutil.copy2(train_jsonl, output_dir / "train.jsonl")

    _register_dataset(ga_path, dataset_name, train_jsonl, dev_jsonl, pose_normed.resolve())

    # Verify files exist before training
    assert train_jsonl.exists(), f"train.jsonl not found at {train_jsonl}"
    assert (dataset_dir / "vocab.json").exists(), f"vocab.json not found"

    # Step 9.6: Training (torchrun for distributed)
    # Run a background cleanup task to prevent checkpoint disk bloat:
    # the training script saves every epoch, so we periodically remove old ones.
    logger.info(f"[{task_id}] Phase 9 Step 9.6: Training model")
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    cleanup_stop = asyncio.Event()

    async def _cleanup_checkpoints():
        """Periodically remove old intermediate checkpoints during training."""
        keep_recent = 2  # keep last N intermediate checkpoints
        while not cleanup_stop.is_set():
            await asyncio.sleep(60)
            intermediates = sorted(ckpt_dir.glob("ssl_checkpoint_*.pth"), key=lambda f: f.stat().st_mtime)
            if len(intermediates) > keep_recent:
                for old in intermediates[:-keep_recent]:
                    old.unlink()
                    logger.debug(f"Cleaned old checkpoint: {old.name}")

    cleanup_task = asyncio.create_task(_cleanup_checkpoints())

    train_script = ga_path / "ssl_pretraining_crossvideo_mlp_feature_mean_mean_advance_v4_noconf_clip_nob2b.py"
    rc, stdout, stderr = await run_subprocess(
        [
            str(Path(sys.executable).parent / "torchrun"), "--nproc_per_node=1",
            str(train_script),
            "--dataset", dataset_name,
            "--output_dir", str(ckpt_dir.resolve()),
            "--epochs", "100",
            "--batch-size", "128",
        ],
        cwd=ga_path,
        env=env,
        timeout=3600 * 24,  # 24 hours max
        log_to_file=True,
    )

    cleanup_stop.set()
    await cleanup_task

    if rc != 0:
        raise RuntimeError(f"Phase 8 training failed: {stderr}")

    # Final cleanup: keep only best checkpoints
    best_ckpt = ckpt_dir / "best_cl.pth"
    if not best_ckpt.exists():
        best_ckpt = ckpt_dir / "best.pth"
    if not best_ckpt.exists():
        ckpts = sorted(ckpt_dir.glob("*.pth"))
        if ckpts:
            best_ckpt = ckpts[-1]
        else:
            logger.warning(f"[{task_id}] No checkpoint produced, skipping prototype building")
            return True

    for f in ckpt_dir.glob("ssl_checkpoint_*.pth"):
        if f.name != best_ckpt.name:
            f.unlink()

    # Step 9.7: Build prototypes
    logger.info(f"[{task_id}] Phase 9 Step 9.7: Building prototypes")
    proto_dir = output_dir / "prototypes"
    proto_script = ga_path / "build_prototypes_asl_clip_nob2b.py"
    rc, _, stderr = await run_subprocess(
        [
            sys.executable, str(proto_script),
            "--ckpt", str(best_ckpt.resolve()),
            "--dataset", dataset_name,
            "--output-dir", str(proto_dir.resolve()),
        ],
        cwd=ga_path,
        env=env,
        log_to_file=True,
    )
    if rc != 0:
        logger.warning(f"[{task_id}] Phase 9 Step 9.7: Prototype building failed: {stderr[-500:]}")
    else:
        proto_files = list(proto_dir.glob("*")) if proto_dir.exists() else []
        logger.info(f"[{task_id}] Phase 9 Step 9.7: {len(proto_files)} prototype files generated")

    logger.info(f"[{task_id}] Phase 8 completed")
    return True
