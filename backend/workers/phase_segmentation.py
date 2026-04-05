"""Phase 5: Train a per-task segmentation model using spamo_segement.

Trains on SENTENCE videos only (not word/gloss videos).
Phase 1 extracts both glosses (words) and sentences. Phase 3 collects videos
for both. This phase filters for sentence-level videos only, since the
segmentation model learns to split sentences into word-level temporal segments.

Steps:
  5.1  Extract CLIP-ViT spatial features from sentence videos
  5.2  Generate annotation files (train_info_ml.npy + val_info_ml.npy)
  5.3  Generate task-specific config YAML
  5.4  Train segmentation model (SpaMo: Flan-T5-XL + LoRA + OT alignment)
  5.5  Run segmentation inference on all sentence videos
  5.6  Output per-word temporal segments

Input:  Phase 3 output (videos/ + manifest.json) + Phase 1 output (glosses.json, sentences.json)
Output: checkpoint + segmentation_results.json
"""
import asyncio
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from backend.config import settings

logger = logging.getLogger(__name__)

SPAMO_ROOT = Path(settings.SPAMO_SEGMENT_PATH).resolve()
SPAMO_PYTHON = sys.executable  # Use current Python interpreter


def _get_sentence_videos(phase3_output: Path, phase1_output: Path) -> list[dict]:
    """Load sentence-only videos from Phase 3 manifest.

    Filters out word/gloss videos by matching against Phase 1's sentences.json.
    For dataset mode, all videos are sentence-level so no filtering needed.
    """
    manifest = phase3_output / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Phase 3 manifest not found: {manifest}")
    with open(manifest) as f:
        all_entries = json.load(f)

    # Load original sentences from Phase 1
    sentences_file = phase1_output / "sentences.json"
    if sentences_file.exists():
        with open(sentences_file) as f:
            sentences = set(s.strip() for s in json.load(f))
    else:
        sentences = None  # No sentences file → treat all as sentences (dataset mode)

    if sentences is None:
        return all_entries

    # Filter: keep only entries whose sentence_text matches an original sentence
    # (not a single gloss word)
    sentence_entries = []
    for entry in all_entries:
        text = entry.get("sentence_text", "").strip()
        if text in sentences:
            sentence_entries.append(entry)

    return sentence_entries


def _get_pseudo_glosses(phase1_output: Path) -> dict[str, list[str]]:
    """Load glosses from Phase 1 output."""
    glosses_file = phase1_output / "glosses.json"
    if not glosses_file.exists():
        return {}
    with open(glosses_file) as f:
        return json.load(f)


async def _extract_clip_features(
    task_id: str, video_dir: Path, output_dir: Path, gpu_id: int = 0
) -> int:
    """Step 5.1: Extract CLIP-ViT spatial features from videos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    script = SPAMO_ROOT / "scripts" / "extract_features" / "extract_clip_from_mp4.py"
    cmd = [
        SPAMO_PYTHON, str(script),
        "--video_dir", str(video_dir),
        "--output_dir", str(output_dir),
        "--device", f"cuda:{gpu_id}",
        "--batch_size", "16",
        "--s2_mode", "s2wrapping",
        "--scales", "1", "2",
    ]

    logger.info(f"[{task_id}] Step 5.1: Extracting CLIP features from {video_dir}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(SPAMO_ROOT),
        env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"CLIP feature extraction failed (rc={proc.returncode}): {stdout.decode()[-500:]}"
        )

    count = len(list(output_dir.glob("*_s2wrapping.npy")))
    logger.info(f"[{task_id}] Step 5.1: Extracted features for {count} videos")
    return count


def _build_annotations(
    task_id: str,
    manifest: list[dict],
    glosses: dict[str, list[str]],
    anno_dir: Path,
    val_ratio: float = 0.2,
) -> tuple[int, int]:
    """Step 5.2: Generate train/val annotation files (npy format)."""
    anno_dir.mkdir(parents=True, exist_ok=True)

    # Build sentence → pseudo-gloss mapping
    gloss_map = {}
    for sent, glist in glosses.items():
        gloss_map[sent.strip()] = " ".join(g.lower() for g in glist)

    entries = []
    for item in manifest:
        text = item.get("sentence_text", "")
        pseudo = gloss_map.get(text.strip(), text)
        filename = item.get("filename", "")
        file_id = Path(filename).stem

        entries.append({
            "fileid": file_id,
            "folder": file_id,
            "text": pseudo,
            "original_text": text,
            "gloss": "",
            "start": 0.0,
            "end": 0.0,
            "video_name": filename,
        })

    # Split into train / val
    n_val = max(1, int(len(entries) * val_ratio))
    n_train = len(entries) - n_val
    train_entries = entries[:n_train]
    val_entries = entries[n_train:]

    np.save(anno_dir / "train_info_ml.npy", np.array(train_entries, dtype=object))
    np.save(anno_dir / "val_info_ml.npy", np.array(val_entries, dtype=object))
    # Also save full set as test for inference
    np.save(anno_dir / "test_info_ml.npy", np.array(entries, dtype=object))

    logger.info(
        f"[{task_id}] Step 5.2: Generated annotations: "
        f"{n_train} train, {n_val} val, {len(entries)} test"
    )
    return n_train, n_val


def _generate_config(
    task_id: str,
    anno_dir: Path,
    feat_dir: Path,
    video_dir: Path,
    output_dir: Path,
    data_size: int,
) -> Path:
    """Step 5.3: Generate task-specific config YAML from template."""
    # Load original config as template
    template_path = SPAMO_ROOT / "configs" / "how2sign_contrastive_word.yaml"
    config = OmegaConf.load(template_path)

    # Adjust paths
    anno_root = str(anno_dir)
    feat_root = str(feat_dir)
    vid_root = str(video_dir)
    mae_feat_root = str(feat_dir)  # Not used but required by dataset class

    for split in ("train", "validation", "test"):
        split_cfg = config.data.params[split]
        split_cfg.params.anno_root = anno_root
        split_cfg.params.feat_root = feat_root
        split_cfg.params.vid_root = vid_root
        split_cfg.params.mae_feat_root = mae_feat_root
        split_cfg.params.flat_feat_dir = True  # Features are in flat dir, not split subdirs

    # Adjust training params for small dataset
    # Scale warmup proportionally (original: 40000 for ~31K samples)
    warmup = max(200, min(2000, data_size * 6))
    config.model.params.warm_up_steps = warmup

    # Fewer epochs, more frequent validation
    config.lightning.trainer.max_epochs = 80
    config.lightning.trainer.check_val_every_n_epoch = 1

    # Use local HuggingFace cache
    config.model.params.cache_dir = str(output_dir / "hf_cache")

    config_path = output_dir / f"config_{task_id}.yaml"
    OmegaConf.save(config, config_path)

    logger.info(f"[{task_id}] Step 5.3: Config saved to {config_path} "
                f"(warmup={warmup}, epochs=80)")
    return config_path


async def _train_model(
    task_id: str, config_path: Path, output_dir: Path, gpu_id: int = 0
) -> Path:
    """Step 5.4: Train segmentation model."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        SPAMO_PYTHON, str(SPAMO_ROOT / "main.py"),
        "--config", str(config_path),
        "--train", "true",
        "--logdir", str(log_dir),
        "--name", task_id,
        "--seed", "42",
    ]

    logger.info(f"[{task_id}] Step 5.4: Starting segmentation model training")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(SPAMO_ROOT),
        env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
    )
    stdout, _ = await proc.communicate()

    # Find best checkpoint
    ckpt_dir = None
    for d in sorted(log_dir.rglob("checkpoints"), reverse=True):
        if d.is_dir():
            ckpt_dir = d
            break

    if ckpt_dir is None:
        raise RuntimeError(
            f"Training finished but no checkpoints found. "
            f"Output: {stdout.decode()[-1000:]}"
        )

    # Prefer best_bleu4, fallback to best_loss, then last
    best_ckpt = None
    for pattern in ("best_bleu4*.ckpt", "best_loss*.ckpt", "last*.ckpt"):
        matches = sorted(ckpt_dir.glob(pattern))
        if matches:
            best_ckpt = matches[-1]
            break

    if best_ckpt is None:
        raise RuntimeError(f"No checkpoint files found in {ckpt_dir}")

    # Copy to output
    final_ckpt = output_dir / "segmentation_model.ckpt"
    shutil.copy2(best_ckpt, final_ckpt)

    logger.info(f"[{task_id}] Step 5.4: Training complete, checkpoint: {final_ckpt.name}")
    return final_ckpt


async def _run_segmentation(
    task_id: str,
    ckpt_path: Path,
    config_path: Path,
    output_dir: Path,
    gpu_id: int = 0,
) -> Path:
    """Step 5.5: Run segmentation inference on all videos."""
    seg_output = output_dir / "segmentation"
    seg_output.mkdir(parents=True, exist_ok=True)

    script = SPAMO_ROOT / "scripts" / "segment_alignment.py"
    cmd = [
        SPAMO_PYTHON, str(script),
        "--ckpt", str(ckpt_path),
        "--config", str(config_path),
        "--mode", "test",  # Use test split which contains all videos
        "--num_samples", "0",  # All samples
        "--output_dir", str(seg_output),
    ]

    logger.info(f"[{task_id}] Step 5.5: Running segmentation inference")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(SPAMO_ROOT),
        env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Segmentation inference failed (rc={proc.returncode}): {stdout.decode()[-500:]}"
        )

    result_file = seg_output / "segmentation_log.json"
    if not result_file.exists():
        raise RuntimeError(f"Segmentation output not found: {result_file}")

    logger.info(f"[{task_id}] Step 5.5: Segmentation results saved to {result_file}")
    return result_file


async def run_phase_segmentation(
    task_id: str,
    phase3_output: Path,
    phase1_output: Path,
    output_dir: Path,
    gpu_id: int = 0,
) -> dict:
    """Run the full segmentation pipeline.

    Args:
        task_id: Pipeline task ID
        phase3_output: Phase 3 output dir (videos/ + manifest.json)
        phase1_output: Phase 1 output dir (glosses.json)
        output_dir: Output directory for this phase
        gpu_id: GPU device ID

    Returns:
        dict with summary of results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs — sentence videos only (exclude word/gloss videos)
    manifest = _get_sentence_videos(phase3_output, phase1_output)
    glosses = _get_pseudo_glosses(phase1_output)

    if not manifest:
        raise RuntimeError("No sentence videos found in Phase 3 output (only word/gloss videos?)")

    # Prepare sentence-only video directory (symlink filtered videos)
    video_dir = phase3_output / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"No videos directory in Phase 3 output: {video_dir}")

    sentence_video_dir = output_dir / "sentence_videos"
    sentence_video_dir.mkdir(parents=True, exist_ok=True)
    sentence_filenames = set()
    for entry in manifest:
        fn = entry.get("filename", "")
        src = video_dir / fn
        if src.exists():
            dst = sentence_video_dir / fn
            if not dst.exists():
                dst.symlink_to(src.resolve() if src.is_symlink() else src)
            sentence_filenames.add(fn)

    video_count = len(sentence_filenames)
    logger.info(f"[{task_id}] Phase 5: {video_count} sentence videos "
                f"(filtered from {len(list(video_dir.glob('*.mp4')))} total)")

    if video_count == 0:
        raise RuntimeError("No sentence video files found on disk")

    # Step 5.1: Extract CLIP features (sentence videos only)
    feat_dir = output_dir / "features"
    feat_count = await _extract_clip_features(task_id, sentence_video_dir, feat_dir, gpu_id)

    # Step 5.2: Generate annotations
    anno_dir = output_dir / "annotations"
    n_train, n_val = _build_annotations(task_id, manifest, glosses, anno_dir)

    # Step 5.3: Generate config
    config_path = _generate_config(
        task_id, anno_dir, feat_dir, video_dir, output_dir, n_train + n_val
    )

    # Step 5.4: Train model
    ckpt_path = await _train_model(task_id, config_path, output_dir, gpu_id)

    # Step 5.5: Run segmentation
    seg_result_path = await _run_segmentation(
        task_id, ckpt_path, config_path, output_dir, gpu_id
    )

    # Step 5.6: Parse and summarize results
    with open(seg_result_path) as f:
        seg_results = json.load(f)

    total_segments = sum(len(r.get("segments", [])) for r in seg_results)

    summary = {
        "input_videos": video_count,
        "features_extracted": feat_count,
        "train_samples": n_train,
        "val_samples": n_val,
        "checkpoint": str(ckpt_path),
        "segmented_videos": len(seg_results),
        "total_word_segments": total_segments,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"[{task_id}] Segmentation phase complete: "
        f"{len(seg_results)} videos → {total_segments} word segments"
    )
    return summary
