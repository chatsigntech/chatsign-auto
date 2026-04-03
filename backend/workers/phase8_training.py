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
        raise RuntimeError("No valid entries for training dataset")

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
    ga_path = settings.GLOSS_AWARE_PATH.resolve()

    # Resolve task data root: input_dir is phase_7/output, so root is 2 levels up
    task_data_root = input_dir.parent.parent
    if not (task_data_root / "phase_4").exists():
        # Fallback: try 3 levels up (in case input_dir is phase_8/input)
        task_data_root = input_dir.parent.parent.parent

    # Collect all videos from input (may include augmented videos in subdirs)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    for mp4 in input_dir.rglob("*.mp4"):
        dst = videos_dir / mp4.name
        if not dst.exists():
            try:
                dst.symlink_to(mp4.resolve())
            except OSError:
                shutil.copy2(mp4, dst)

    pose_dir = output_dir / "poses_raw"
    pose_filtered = output_dir / "poses_filtered"
    pose_normed = output_dir / "poses_normed"

    # Step 8.1: Extract poses
    logger.info(f"[{task_id}] Phase 8 Step 8.1: Extracting poses")
    script = ga_path / "preprocess" / "pose_extractor.py"
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(script), str(videos_dir), str(pose_dir)],
        cwd=ga_path, env=env, timeout=7200,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 8 Step 8.1 failed: {stderr}")

    # Step 8.2: Filter pose files
    logger.info(f"[{task_id}] Phase 8 Step 8.2: Filtering pose files")
    script = ga_path / "preprocess" / "filter_pose_pkls.py"
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(script), "--in-dir", str(pose_dir), "--out-dir", str(pose_filtered)],
        cwd=ga_path, env=env,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 8 Step 8.2 failed: {stderr}")

    # Step 8.3: Normalize poses
    logger.info(f"[{task_id}] Phase 8 Step 8.3: Normalizing poses")
    script = ga_path / "preprocess" / "batch_norm_cosign_padding.py"
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(script), str(pose_filtered), str(pose_normed)],
        cwd=ga_path, env=env,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 8 Step 8.3 failed: {stderr}")

    # Step 8.4: Generate dataset JSONL and register
    logger.info(f"[{task_id}] Phase 8 Step 8.4: Generating dataset index")
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

    _register_dataset(ga_path, dataset_name, train_jsonl, dev_jsonl, pose_normed)

    # Verify files exist before training
    assert train_jsonl.exists(), f"train.jsonl not found at {train_jsonl}"
    assert (dataset_dir / "vocab.json").exists(), f"vocab.json not found"

    # Step 8.5: Training (torchrun for distributed)
    logger.info(f"[{task_id}] Phase 8 Step 8.5: Training model")
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    train_script = ga_path / "ssl_pretraining_crossvideo_mlp_feature_mean_mean_advance_v4_noconf_clip_nob2b.py"
    rc, stdout, stderr = await run_subprocess(
        [
            str(Path(sys.executable).parent / "torchrun"), "--nproc_per_node=1",
            str(train_script),
            "--dataset", dataset_name,
            "--output_dir", str(ckpt_dir),
            "--epochs", "100",
            "--batch-size", "128",
        ],
        cwd=ga_path,
        env=env,
        timeout=3600 * 24,  # 24 hours max
    )
    if rc != 0:
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

    # Step 8.6: Build prototypes
    logger.info(f"[{task_id}] Phase 8 Step 8.6: Building prototypes")
    proto_script = ga_path / "build_prototypes_asl_clip_nob2b.py"
    rc, _, stderr = await run_subprocess(
        [
            sys.executable, str(proto_script),
            "--ckpt", str(best_ckpt),
            "--dataset", dataset_name,
            "--output-dir", str(output_dir / "prototypes"),
        ],
        cwd=ga_path,
        env=env,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 8 prototype building failed: {stderr}")

    logger.info(f"[{task_id}] Phase 8 completed")
    return True
