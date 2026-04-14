"""Phase 8: Model training using gloss_aware scripts.

Input sources (video + corresponding label text):
  - Phase 2: original videos (sentence + word)
  - Phase 5: segmented word clips
  - Phase 6: augmented videos (sentence + word + segment)
  - Phase 7: augmented segment clips

Pipeline:
  8.1  Extract poses from all input videos (RTMPose)
  8.2  Filter low-quality pose files
  8.3  Normalize pose keypoints
  8.4  Generate train/dev JSONL + register dataset in gloss_aware config
  8.5  Self-supervised pre-training (torchrun)
  8.6  Build gloss prototypes
"""
import asyncio
import csv
import hashlib
import json
import logging
import os
import pickle
import shutil
import sys
from pathlib import Path

from sqlmodel import Session, select

from backend.config import settings
from backend.core.subprocess_runner import run_subprocess
from backend.database import engine
from backend.models.task import PipelineTask

logger = logging.getLogger(__name__)

DATASET_NAME = "Pipeline"  # registered name in gloss_aware/config.py


def _build_video_gloss_map(phase2_output: Path, phase1_output: Path) -> dict[str, list[str]]:
    """Build a mapping from original video stem → gloss tokens.

    Reads Phase 2 annotations.json which contains:
      [{"filename": "xxx.mp4", "glosses": ["WORD"]}]
    """
    ann_path = phase2_output / "annotations.json"
    mapping = {}
    if ann_path.exists():
        with open(ann_path) as f:
            for entry in json.load(f):
                stem = Path(entry["filename"]).stem
                mapping[stem] = entry.get("glosses", [])

    # Fallback: Phase 2 manifest
    if not mapping:
        manifest = phase2_output / "manifest.json"
        if manifest.exists():
            with open(manifest) as f:
                for entry in json.load(f):
                    stem = Path(entry["filename"]).stem
                    text = entry.get("sentence_text", "")
                    mapping[stem] = [w.upper() for w in text.split()] if text else []

    # Fallback: Phase 1 glosses
    if not mapping:
        glosses_file = phase1_output / "glosses.json"
        if glosses_file.exists():
            with open(glosses_file) as f:
                for sent, glist in json.load(f).items():
                    mapping[sent.strip()] = glist

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


def _generate_annotations_csv(
    pose_dir: Path,
    gloss_map: dict[str, list[str]],
    output_path: Path,
) -> int:
    """Generate annotations CSV from pose PKLs + gloss_map for make_asl_labels.py.

    Returns the number of entries written.
    """
    rows = []
    for pkl in sorted(pose_dir.glob("*.pkl")):
        stem = pkl.stem
        tokens = _match_gloss_for_augmented(stem, gloss_map)
        if not tokens:
            continue
        # make_asl_labels.py splits gloss by delimiter; join multi-token with "+"
        gloss_str = "+".join(tokens) if tokens else ""
        rows.append({"ref": stem, "gloss": gloss_str})

    if not rows:
        total_pkls = len(list(pose_dir.glob("*.pkl")))
        raise RuntimeError(
            f"No valid entries for training dataset. "
            f"Poses: {total_pkls} normalized pkl files, "
            f"Gloss map: {len(gloss_map)} video→gloss mappings. "
            f"No pose filenames matched any gloss map entries."
        )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ref", "gloss"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Annotations CSV: {len(rows)} entries → {output_path}")
    return len(rows)


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


def _merge_vocab(prev_vocab_path: Path, new_vocab_path: Path, output_path: Path) -> dict:
    """Merge two vocab.json files, preserving old IDs and appending new tokens.

    Returns the merged token_to_id dict.
    """
    with open(prev_vocab_path) as f:
        prev = json.load(f)
    with open(new_vocab_path) as f:
        new = json.load(f)

    # Start from previous vocab (preserves all old ID assignments)
    merged_t2i = dict(prev["token_to_id"])
    if not merged_t2i:
        raise RuntimeError(f"Previous vocab is empty: {prev_vocab_path}")

    # Normalize legacy <pad> → <blank> (old tasks used <pad> for id=0)
    if "<pad>" in merged_t2i and "<blank>" not in merged_t2i:
        merged_t2i["<blank>"] = merged_t2i.pop("<pad>")
        logger.info("Vocab compat: renamed <pad> → <blank> (id=0)")

    max_id = max(merged_t2i.values())

    # Append new tokens that aren't in old vocab
    new_count = 0
    for token in sorted(new["token_to_id"]):
        # Skip <pad> from new vocab if we already have <blank> (same role, id=0)
        if token == "<pad>" and "<blank>" in merged_t2i:
            continue
        if token not in merged_t2i:
            max_id += 1
            merged_t2i[token] = max_id
            new_count += 1

    # Build list-format id_to_token
    merged_i2t = [""] * (max_id + 1)
    for token, tid in merged_t2i.items():
        merged_i2t[tid] = token

    merged = {"id_to_token": merged_i2t, "token_to_id": merged_t2i}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    logger.info(f"Vocab merged: {len(prev['token_to_id'])} old + {new_count} new = {len(merged_t2i)} total")
    return merged_t2i


def _merge_jsonl(prev_jsonl: Path, new_jsonl: Path, output_path: Path) -> int:
    """Concatenate two JSONL files. Returns total entry count."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for src in [prev_jsonl, new_jsonl]:
            if not src.exists():
                continue
            with open(src, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.write(line + "\n")
                        count += 1
    return count


def _resolve_prev_task_outputs(prev_task_id: str) -> dict:
    """Locate the previous task's Phase 8 outputs needed for incremental training.

    Returns dict with paths: checkpoint, vocab, train_jsonl, dev_jsonl, poses_normed.
    Raises RuntimeError if critical files are missing.
    """
    prev_dir = settings.SHARED_DATA_ROOT / prev_task_id / "phase_8" / "output"
    if not prev_dir.exists():
        raise RuntimeError(f"Previous task Phase 8 output not found: {prev_dir}")

    # Find best checkpoint
    ckpt_dir = prev_dir / "checkpoints"
    prev_ckpt = ckpt_dir / "best_cl.pth"
    if not prev_ckpt.exists():
        prev_ckpt = ckpt_dir / "best.pth"
    if not prev_ckpt.exists():
        raise RuntimeError(f"Previous task checkpoint not found in {ckpt_dir}")

    prev_vocab = prev_dir / "vocab.json"
    if not prev_vocab.exists():
        raise RuntimeError(f"Previous task vocab.json not found: {prev_vocab}")

    prev_train = prev_dir / "train.jsonl"
    prev_dev = prev_dir / "dev.jsonl"

    prev_poses = prev_dir / "poses_normed"

    return {
        "checkpoint": prev_ckpt,
        "vocab": prev_vocab,
        "train_jsonl": prev_train,
        "dev_jsonl": prev_dev,
        "poses_normed": prev_poses,
    }


async def run_phase8_training(
    task_id: str,
    phase2_output: Path,
    phase5_output: Path,
    phase6_output: Path,
    phase7_output: Path,
    output_dir: Path,
    gpu_id: int = 0,
    prev_task_id: str | None = None,
) -> bool:
    """Run model training pipeline: preprocess -> train -> build prototypes.

    Input sources:
      - phase2_output: original videos (sentence + word)
      - phase5_output: segmented word clips
      - phase6_output: augmented videos
      - phase7_output: augmented segment clips

    If prev_task_id is set, runs incremental training:
      - Merges previous vocab + new vocab (preserving old IDs)
      - Merges previous JSONL + new JSONL
      - Warm-starts training from previous checkpoint (--pretrained)
      - Rebuilds prototypes on full merged dataset
    """
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

    # Collect all videos from multiple input sources
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    def _link_videos(source_dir: Path, prefix: str = ""):
        """Symlink videos from source into videos_dir with unique names."""
        if not source_dir.exists():
            return 0
        count = 0
        for mp4 in source_dir.rglob("*.mp4"):
            rel = mp4.relative_to(source_dir)
            if len(rel.parts) > 1:
                sub_prefix = "_".join(rel.parts[:-1])
                unique_name = f"{prefix}{sub_prefix}_{mp4.name}" if prefix else f"{sub_prefix}_{mp4.name}"
            else:
                unique_name = f"{prefix}{mp4.name}" if prefix else mp4.name
            # Truncate long filenames (Linux 255 char limit)
            if len(unique_name) > 250:
                name_hash = hashlib.md5(unique_name.encode()).hexdigest()[:12]
                unique_name = f"{unique_name[:200]}_{name_hash}.mp4"
            dst = videos_dir / unique_name
            if not dst.exists():
                try:
                    dst.symlink_to(mp4.resolve())
                except OSError:
                    shutil.copy2(mp4, dst)
            count += 1
        return count

    # Phase 2: original videos
    p2_videos = phase2_output / "videos"
    _link_videos(p2_videos, "orig_")

    # Phase 5: segmented word clips
    p5_segments = phase5_output / "segment_videos"
    _link_videos(p5_segments, "seg_")

    # Phase 6: augmented videos (sentence/word/segment categories)
    for cat in ("sentence", "word", "segment"):
        cat_dir = phase6_output / cat
        if cat_dir.exists():
            _link_videos(cat_dir, f"aug_{cat}_")

    # Phase 7: augmented segment clips
    p7_segments = phase7_output / "aug_segment_videos"
    _link_videos(p7_segments, "augseg_")

    total_videos = len(list(videos_dir.glob("*.mp4")))
    logger.info(f"[{task_id}] Phase 8: Collected {total_videos} videos for training")

    if total_videos == 0:
        raise RuntimeError("Phase 8: No videos found from input sources")

    pose_dir = output_dir / "poses_raw"
    pose_filtered = output_dir / "poses_filtered"
    pose_normed = output_dir / "poses_normed"

    # Step 8.1: Extract poses (log_to_file=True to avoid PIPE deadlock on large output)
    logger.info(f"[{task_id}] Phase 8 Step 8.1: Extracting poses from {total_videos} videos")
    script = ga_path / "preprocess" / "pose_extractor.py"
    rc, stdout, stderr = await run_subprocess(
        [sys.executable, str(script), str(videos_dir.resolve()), str(pose_dir.resolve())],
        cwd=ga_path, env=env, timeout=None, log_to_file=True,
    )
    poses_raw_count = len(list(pose_dir.glob("*.pkl"))) if pose_dir.exists() else 0
    logger.info(f"[{task_id}] Phase 8 Step 8.1: {poses_raw_count}/{total_videos} poses extracted")
    if rc != 0:
        raise RuntimeError(f"Phase 8 Step 8.1 pose extraction failed ({poses_raw_count}/{total_videos} extracted): {stderr[-500:]}")
    if poses_raw_count == 0:
        raise RuntimeError(
            f"Phase 8 Step 8.1: Pose extraction produced 0 results from {total_videos} videos. "
            f"Possible causes: videos too short, no person detected, or invalid video format."
        )

    # Step 8.2: Filter pose files
    logger.info(f"[{task_id}] Phase 8 Step 8.2: Filtering {poses_raw_count} pose files")
    script = ga_path / "preprocess" / "filter_pose_pkls.py"
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(script), "--in-dir", str(pose_dir.resolve()), "--out-dir", str(pose_filtered.resolve())],
        cwd=ga_path, env=env,
    )
    poses_filtered_count = len(list(pose_filtered.glob("*.pkl"))) if pose_filtered.exists() else 0
    logger.info(f"[{task_id}] Phase 8 Step 8.2: {poses_filtered_count}/{poses_raw_count} poses passed quality filter")
    if rc != 0:
        raise RuntimeError(f"Phase 8 Step 8.2 pose filter failed: {stderr[-500:]}")
    if poses_filtered_count == 0:
        raise RuntimeError(
            f"Phase 8 Step 8.2: All {poses_raw_count} poses filtered out. "
            f"Videos may lack visible hands or face (hand_threshold=0.8, head_threshold=0.9)."
        )

    # Step 8.3: Normalize poses
    logger.info(f"[{task_id}] Phase 8 Step 8.3: Normalizing {poses_filtered_count} poses")
    script = ga_path / "preprocess" / "batch_norm_cosign_padding.py"
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(script), str(pose_filtered.resolve()), str(pose_normed.resolve())],
        cwd=ga_path, env=env,
    )
    poses_normed_count = len(list(pose_normed.glob("*.pkl"))) if pose_normed.exists() else 0
    logger.info(f"[{task_id}] Phase 8 Step 8.3: {poses_normed_count}/{poses_filtered_count} poses normalized")
    if rc != 0:
        raise RuntimeError(f"Phase 8 Step 8.3 normalization failed: {stderr[-500:]}")

    # Step 8.4: Validate pkl files — remove corrupt and too-short ones in a single pass
    BLOCK_SIZE = 12
    logger.info(f"[{task_id}] Phase 8 Step 8.4: Validating pose pkl files")
    corrupt_files = []
    short_files = []
    for pkl in list(pose_normed.glob("*.pkl")):
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Expected dict, got {type(data)}")
            T = data["body"].shape[0]
            if T < BLOCK_SIZE:
                short_files.append({"file": pkl.name, "frames": T})
                pkl.unlink()
        except Exception as e:
            corrupt_files.append({"file": pkl.name, "error": str(e)})
            pkl.unlink()

    poses_valid_count = len(list(pose_normed.glob("*.pkl")))

    if corrupt_files:
        logger.warning(f"[{task_id}] Phase 8 Step 8.4: Removed {len(corrupt_files)} corrupt pkl files")
        with open(output_dir / "corrupt_poses.json", "w") as f:
            json.dump(corrupt_files, f, indent=2)
    if short_files:
        logger.warning(f"[{task_id}] Phase 8 Step 8.4: Removed {len(short_files)} poses "
                       f"shorter than {BLOCK_SIZE} frames")
        with open(output_dir / "short_poses.json", "w") as f:
            json.dump(short_files, f, indent=2)

    logger.info(f"[{task_id}] Phase 8 Step 8.4: {poses_valid_count} valid poses remaining")

    if poses_valid_count == 0:
        raise RuntimeError("Phase 8 Step 8.4: All pose files are corrupt or too short")

    # Step 8.5: Generate dataset JSONL and register
    logger.info(f"[{task_id}] Phase 8 Step 8.5: Generating dataset index")
    # Resolve Phase 1 output from Phase 2 path
    phase1_output = phase2_output.parent.parent / "phase_1" / "output"
    gloss_map = _build_video_gloss_map(phase2_output, phase1_output)
    if not gloss_map:
        raise RuntimeError("No video→gloss mapping found from Phase 2 annotations")

    dataset_name = f"{DATASET_NAME}_{task_id}"

    # Put dataset files where the training script expects: data/{dataset_name}/
    dataset_dir = ga_path / "data" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Generate annotations CSV from pose PKLs + gloss_map
    csv_path = output_dir / "annotations.csv"
    _generate_annotations_csv(pose_normed, gloss_map, csv_path)

    # Use upstream make_asl_labels.py to build vocab + JSONL
    make_labels_script = ga_path / "tools" / "make_asl_labels.py"
    rc, stdout, stderr = await run_subprocess(
        [
            sys.executable, str(make_labels_script),
            "--csv-path", str(csv_path),
            "--pose-dir", str(pose_normed.resolve()),
            "--out-dir", str(dataset_dir),
            "--id-col", "ref",
            "--gloss-col", "gloss",
            "--gloss-split", "+",
            "--min-freq", "1",
        ],
        cwd=ga_path, env=env,
    )
    if rc != 0:
        raise RuntimeError(f"make_asl_labels.py failed: {stderr[-500:]}")

    train_jsonl = dataset_dir / "train.jsonl"
    dev_jsonl = dataset_dir / "dev.jsonl"
    if not train_jsonl.exists():
        raise RuntimeError(f"make_asl_labels.py did not produce train.jsonl in {dataset_dir}")

    # Log stats for current task's data
    vocab_path = dataset_dir / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path) as f:
            vocab_data = json.load(f)
        logger.info(f"Current task vocab: {len(vocab_data.get('token_to_id', {}))} tokens")

    # --- Step 8.5b: Incremental merge (if prev_task_id is set) ---
    prev_checkpoint = None
    if prev_task_id:
        logger.info(f"[{task_id}] Phase 8 Step 8.5b: Merging with previous task {prev_task_id}")
        prev = _resolve_prev_task_outputs(prev_task_id)
        prev_checkpoint = prev["checkpoint"]

        # Merge vocab: preserve old IDs, append new tokens
        merged_vocab_path = dataset_dir / "vocab.json"
        _merge_vocab(prev["vocab"], vocab_path, merged_vocab_path)
        vocab_path = merged_vocab_path

        # Merge training data
        merged_train = dataset_dir / "train_merged.jsonl"
        merged_dev = dataset_dir / "dev_merged.jsonl"
        n_train = _merge_jsonl(prev["train_jsonl"], train_jsonl, merged_train)
        n_dev = _merge_jsonl(prev["dev_jsonl"], dev_jsonl, merged_dev)

        # Replace with merged files
        shutil.move(str(merged_train), str(train_jsonl))
        if merged_dev.exists():
            shutil.move(str(merged_dev), str(dev_jsonl))

        logger.info(f"[{task_id}] Merged dataset: {n_train} train, {n_dev} dev entries")

        # Register merged pose dirs: previous + current
        # The training script reads pose_path from JSONL which has absolute paths,
        # so we register the current task's pose dir (prev task poses are already
        # referenced by their absolute paths in the JSONL).

    # Copy dataset files to output for reference and incremental reuse
    shutil.copy2(train_jsonl, output_dir / "train.jsonl")
    if dev_jsonl.exists():
        shutil.copy2(dev_jsonl, output_dir / "dev.jsonl")
    shutil.copy2(vocab_path, output_dir / "vocab.json")

    _register_dataset(ga_path, dataset_name, train_jsonl, dev_jsonl, pose_normed.resolve())

    # Verify files exist before training
    assert train_jsonl.exists(), f"train.jsonl not found at {train_jsonl}"
    assert vocab_path.exists(), f"vocab.json not found"

    # Step 8.6: Training (torchrun for distributed)
    # Run a background cleanup task to prevent checkpoint disk bloat:
    # the training script saves every epoch, so we periodically remove old ones.
    logger.info(f"[{task_id}] Phase 8 Step 8.6: Training model")
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
    train_cmd = [
        str(Path(sys.executable).parent / "torchrun"), "--nproc_per_node=1",
        str(train_script),
        "--dataset", dataset_name,
        "--output_dir", str(ckpt_dir.resolve()),
        "--epochs", "150",
        "--batch-size", "16",
        "--block-size", "12",
        "--block-stride", "6",
    ]
    if prev_checkpoint:
        train_cmd += ["--pretrained", str(prev_checkpoint.resolve())]
        logger.info(f"[{task_id}] Warm start from: {prev_checkpoint}")

    rc, stdout, stderr = await run_subprocess(
        train_cmd,
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

    # Step 8.7: Build prototypes
    logger.info(f"[{task_id}] Phase 8 Step 8.7: Building prototypes")
    proto_dir = output_dir / "prototypes"
    proto_script = ga_path / "build_prototypes_asl_clip_nob2b.py"
    rc, _, stderr = await run_subprocess(
        [
            sys.executable, str(proto_script),
            "--ckpt", str(best_ckpt.resolve()),
            "--dataset", dataset_name,
            "--output-dir", str(proto_dir.resolve()),
            "--block-size", "12",
            "--block-stride", "6",
            "--l2norm",
        ],
        cwd=ga_path,
        env=env,
        log_to_file=True,
    )
    if rc != 0:
        logger.warning(f"[{task_id}] Phase 8 Step 8.7: Prototype building failed: {stderr[-500:]}")
    else:
        proto_files = list(proto_dir.glob("*")) if proto_dir.exists() else []
        logger.info(f"[{task_id}] Phase 8 Step 8.7: {len(proto_files)} prototype files generated")

    # Step 8.8: Clean up old tasks' intermediate data (retention policy)
    _cleanup_old_training_data(task_id)

    logger.info(f"[{task_id}] Phase 8 completed")
    return True


def _cleanup_old_training_data(current_task_id: str):
    """Remove disposable intermediate data from old tasks beyond retention limit.

    Always kept (needed for incremental training):
      checkpoints/, prototypes/, vocab.json, train.jsonl, dev.jsonl, poses_normed/

    Cleaned up:
      poses_raw/, poses_filtered/, videos/, annotations.csv, corrupt/short_poses.json
    """
    retention = settings.TRAINING_DATA_RETENTION
    if retention <= 0:
        return

    with Session(engine) as session:
        tasks = session.exec(
            select(PipelineTask).where(
                PipelineTask.status == "completed"
            ).order_by(PipelineTask.created_at.desc())
        ).all()

    # Only consider tasks with Phase 8 output
    completed_ids = []
    for t in tasks:
        p8_dir = settings.SHARED_DATA_ROOT / t.task_id / "phase_8" / "output"
        if p8_dir.exists():
            completed_ids.append(t.task_id)

    if len(completed_ids) <= retention:
        return

    # Tasks beyond retention limit (oldest first)
    to_clean = completed_ids[retention:]
    disposable_dirs = ["poses_raw", "poses_filtered", "videos"]
    disposable_files = ["annotations.csv", "corrupt_poses.json", "short_poses.json"]

    for tid in to_clean:
        p8_out = settings.SHARED_DATA_ROOT / tid / "phase_8" / "output"
        for dirname in disposable_dirs:
            d = p8_out / dirname
            if d.exists():
                shutil.rmtree(d)
                logger.info(f"Cleaned {d} (retention policy)")
        for fname in disposable_files:
            f = p8_out / fname
            if f.exists():
                f.unlink()
                logger.debug(f"Cleaned {f}")
