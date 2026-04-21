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
import re
import shutil
import sys
from pathlib import Path
from typing import Literal

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


def _build_segment_label_map(phase5_output: Path) -> dict[str, str]:
    """Build segment filename stem → gloss label from Phase 5 split_points.json.

    Returns e.g. {"sentence_1_seg003_app": "APP", "sentence_0_seg001_apply.": "APPLY"}
    Labels are uppercased and punctuation-stripped.
    """
    sp_path = phase5_output / "split_points.json"
    if not sp_path.exists():
        return {}

    with open(sp_path) as f:
        split_points = json.load(f)

    label_map = {}
    for vid_stem, info in split_points.items():
        for i, seg in enumerate(info.get("segments", [])):
            raw_label = seg.get("label", "")
            # Segment filename: {vid_stem}_seg{NNN}_{label}
            seg_stem = f"{vid_stem}_seg{i:03d}_{raw_label}"
            # Clean label: uppercase + strip punctuation
            clean = re.sub(r'[.,!?;:]+$', '', raw_label.upper())
            if clean:
                label_map[seg_stem] = clean

    return label_map


def _extract_single_gloss(
    pkl_stem: str,
    gloss_map: dict[str, list[str]],
    segment_labels: dict[str, str],
) -> str | None:
    """Extract the single-word gloss label for a video based on its type.

    Video types and labeling strategy:
      - Segment videos: label from Phase 5 split_points.json (via segment_labels)
      - Word videos: label from gloss_map (word_APPLE → ["APPLE"])
      - Sentence videos: EXCLUDED (multi-word, not suitable for word-level training)

    Returns the single gloss string (uppercased), or None to skip this video.
    """

    # 1. Segment videos: look up in split_points label map
    for seg_stem, label in segment_labels.items():
        if seg_stem in pkl_stem:
            return label

    # 2. Word videos: match against known word stems from gloss_map
    best_word_match = None
    for orig_stem, glosses in gloss_map.items():
        if orig_stem.startswith("word_") and orig_stem in pkl_stem:
            if best_word_match is None or len(orig_stem) > len(best_word_match[0]):
                best_word_match = (orig_stem, glosses)
    if best_word_match:
        return best_word_match[1][0].upper() if best_word_match[1] else None

    # 3. Sentence videos: skip (contain multiple words)
    for orig_stem in gloss_map:
        if orig_stem.startswith("sentence_") and orig_stem in pkl_stem:
            return None

    # 4. Fallback: single-gloss entries from gloss_map
    for orig_stem, glosses in gloss_map.items():
        if orig_stem in pkl_stem and len(glosses) == 1:
            return glosses[0]

    return None


def _generate_annotations_csv(
    pose_dir: Path,
    gloss_map: dict[str, list[str]],
    segment_labels: dict[str, str],
    output_path: Path,
) -> int:
    """Generate annotations CSV from pose PKLs + gloss_map for make_asl_labels.py.

    Only includes word-level and segment-level videos with single-gloss labels.
    Sentence-level videos are excluded (multi-word, not suitable for word-level training).

    Returns the number of entries written.
    """

    rows = []
    skipped_sentence = 0
    skipped_unmatched = 0
    for pkl in sorted(pose_dir.glob("*.pkl")):
        stem = pkl.stem
        gloss = _extract_single_gloss(stem, gloss_map, segment_labels)
        if gloss is None:
            if "sentence_" in stem:
                skipped_sentence += 1
            else:
                skipped_unmatched += 1
            continue
        # Strip trailing punctuation (e.g. "APPLY." → "APPLY", "STORE." → "STORE")
        gloss = re.sub(r'[.,!?;:]+$', '', gloss)
        if not gloss:
            continue
        rows.append({"ref": stem, "gloss": gloss})

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

    logger.info(
        f"Annotations CSV: {len(rows)} entries, "
        f"skipped {skipped_sentence} sentence videos, "
        f"{skipped_unmatched} unmatched → {output_path}"
    )
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
    env["WANDB_DISABLED"] = "true"
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

    def _link_videos(
        source_dir: Path,
        prefix: str = "",
        level: Literal["word", "sentence"] | None = None,
        filter_prefix: str | None = None,
    ) -> int:
        """Symlink videos from source into videos_dir with unique names.

        level='word' / 'sentence' forces each output stem to end with
        '_<level>' so split-level training and prototype code can read the
        level from the filename suffix (name.endswith('_sentence')).
        filter_prefix restricts linking to files whose basename starts with
        this prefix (used to keep word_*.mp4 and drop sentence_*.mp4 from
        Phase 2's mixed videos/ directory).
        """
        if not source_dir.exists():
            return 0
        count = 0
        for mp4 in source_dir.rglob("*.mp4"):
            if filter_prefix and not mp4.name.startswith(filter_prefix):
                continue
            rel = mp4.relative_to(source_dir)
            if len(rel.parts) > 1:
                sub = "_".join(rel.parts[:-1])
                stem = f"{prefix}{sub}_{mp4.stem}"
            else:
                stem = f"{prefix}{mp4.stem}"
            if level and not stem.endswith(f"_{level}"):
                stem = f"{stem}_{level}"
            name = f"{stem}.mp4"
            # Truncate long filenames (Linux 255 char limit)
            if len(name) > 250:
                h = hashlib.md5(stem.encode()).hexdigest()[:12]
                stem = f"{stem[:200]}_{h}"
                name = f"{stem}.mp4"
            dst = videos_dir / name
            if not dst.exists():
                try:
                    dst.symlink_to(mp4.resolve())
                except OSError:
                    shutil.copy2(mp4, dst)
            count += 1
        return count

    # Video sources (level-tagged per junyi/chatsign convention):
    #   orig_<...>_word       Phase 2 word_*.mp4 (sentence_*.mp4 skipped)
    #   seg_<...>_sentence    Phase 5 segment_videos (sliced from sentences)
    #   aug_word_<...>_word   Phase 6 word/ (isolated-sign augment)
    #   aug_segment_<...>_sentence  Phase 6 segment/ (sentence-derived augment)
    #   augseg_<...>_sentence Phase 7 aug_segment_videos (path B)
    # Whole sentence videos (Phase 2 sentence_*, Phase 6 sentence/) are
    # intentionally skipped — they don't enter word-level training and would
    # only waste pose-extraction time.
    _link_videos(phase2_output / "videos", "orig_", level="word", filter_prefix="word_")
    _link_videos(phase5_output / "segment_videos", "seg_", level="sentence")
    _link_videos(phase6_output / "word", "aug_word_", level="word")
    _link_videos(phase6_output / "segment", "aug_segment_", level="sentence")
    _link_videos(phase7_output / "aug_segment_videos", "augseg_", level="sentence")

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
        cwd=ga_path, env=env, timeout=None, log_to_file=True, task_id=task_id,
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
        cwd=ga_path, env=env, task_id=task_id,
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
    script = ga_path / "preprocess" / "batch_norm_cosign_unified.py"
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(script), str(pose_filtered.resolve()), str(pose_normed.resolve())],
        cwd=ga_path, env=env, task_id=task_id,
    )
    poses_normed_count = len(list(pose_normed.glob("*.pkl"))) if pose_normed.exists() else 0
    logger.info(f"[{task_id}] Phase 8 Step 8.3: {poses_normed_count}/{poses_filtered_count} poses normalized")
    if rc != 0:
        raise RuntimeError(f"Phase 8 Step 8.3 normalization failed: {stderr[-500:]}")

    # Step 8.4: Validate pkl files — remove corrupt and too-short ones in a single pass
    BLOCK_SIZE = 6  # matches --block-size passed to training and prototype scripts
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

    # Build segment labels from Phase 5 split_points.json
    segment_labels = _build_segment_label_map(phase5_output)
    logger.info(f"Segment labels from split_points.json: {len(segment_labels)} entries")

    # Generate annotations CSV from pose PKLs + gloss_map + segment labels
    csv_path = output_dir / "annotations.csv"
    _generate_annotations_csv(pose_normed, gloss_map, segment_labels, csv_path)

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
        cwd=ga_path, env=env, task_id=task_id,
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
    # Fully sync junyi/chatsign README training command. All other hyperparams
    # (batch-size 128, lr 1e-3, hidden-dim 256, block-size 6 / stride 3,
    # arcface scale 30 margin 0.5, center-loss, etc.) come from upstream script
    # defaults.
    train_cmd = [
        str(Path(sys.executable).parent / "torchrun"),
        "--standalone", "--nproc_per_node=1",
        str(train_script),
        "--dataset", dataset_name,
        "--output_dir", str(ckpt_dir.resolve()),
        "--epochs", "150",
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
        task_id=task_id,
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

    # Step 8.7: Build prototypes via junyi/chatsign canonical
    # build_prototypes_asl_clip_nob2b.py (README Step 8).
    logger.info(f"[{task_id}] Phase 8 Step 8.7: Building prototypes")
    proto_dir = output_dir / "prototypes"
    proto_script = ga_path / "build_prototypes_asl_clip_nob2b.py"
    proto_cmd = [
        sys.executable, str(proto_script),
        "--dataset", dataset_name,
        "--ckpt", str(best_ckpt.resolve()),
        "--output-dir", str(proto_dir.resolve()),
        "--l2norm",
    ]

    rc, _, stderr = await run_subprocess(
        proto_cmd, cwd=ga_path, env=env, log_to_file=True, task_id=task_id,
    )
    if rc != 0:
        logger.warning(f"[{task_id}] Phase 8 Step 8.7: Prototype building failed: {stderr[-500:]}")
    else:
        proto_files = list(proto_dir.glob("*")) if proto_dir.exists() else []
        logger.info(
            f"[{task_id}] Phase 8 Step 8.7: {len(proto_files)} prototype files generated"
        )

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
