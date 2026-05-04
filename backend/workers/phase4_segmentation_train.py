"""Phase 4: Train a per-task segmentation model using test_real/phase4_seg_train.

Trains on SENTENCE videos only (not word/gloss videos).
This phase handles feature extraction and model training only.
Segmentation inference is performed in Phase 5.

Steps:
  4.1  Extract CLIP-ViT spatial features from sentence videos
  4.2  Generate annotation files (train_info_ml.npy + val_info_ml.npy)
  4.2.5 concat-aug data synthesis (test_real 06b port; B + C class variants)
  4.3  Generate task-specific config YAML (chatsign_concat_aug_colent)
  4.4  Train segmentation model (SpaMo: Flan-T5-XL + LoRA + OT alignment + colent loss)

Input:  Phase 2 output (videos/ + manifest.json) + Phase 1 output (glosses.json)
Output: checkpoint + config + features + annotations
"""
import csv
import json
import logging
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from sqlmodel import Session, select

from backend.config import settings
from backend.core.dataset_videos import H2S_DIR, OPENASL_DIR
from backend.core.sentence_search import H2S_FILTERED, OPENASL_FILTERED
from backend.core.subprocess_runner import run_subprocess
from backend.core.video_utils import make_gpu_env
from backend.database import engine
from backend.models.task import PipelineTask
from backend.workers.phase4_concat_aug import run_concat_aug

logger = logging.getLogger(__name__)

P4_ROOT = (settings.TEST_REAL_PATH / "phase4_seg_train").resolve()
SPAMO_PYTHON = sys.executable

# S2_MODE must match the --s2_mode CLI arg passed to extract_clip_from_mp4.py;
# the extractor writes files as f"{stem}_{s2_mode}.npy" so the cache suffix
# must track the same value.
S2_MODE = "s2wrapping"
CLIP_FEATURE_SUFFIX = f"_{S2_MODE}"
CLIP_FEATURE_CACHE_DIR = settings.VIDEO_DATA_ROOT / "clip_features"
_VIDEO_DATA_ROOT_RESOLVED = settings.VIDEO_DATA_ROOT.resolve()

# Phase 1 sometimes emits sentence keys with an extra trailing '.' (e.g.
# "Brazil!." vs manifest's "Brazil!"). Normalize on lookup so the mismatch
# doesn't break gloss resolution.
_TRAILING_PUNCT = ".!?。！？"


def _norm_sentence(s: str) -> str:
    return s.strip().rstrip(_TRAILING_PUNCT)


def _get_sentence_videos(phase2_output: Path, phase1_output: Path) -> list[dict]:
    """Load sentence-only videos from Phase 2 manifest.

    Dataset mode: returns entries whose filename starts with 'sentence_' (drops
    word_*.mp4, which belong to Phase 8 recognition training — not Phase 4
    segmentation, where they would corrupt val/BLEU as single-word targets).
    Manual mode: filters out word/gloss videos by checking if sentence_text
    appears as a key in Phase 1's glosses.json (gloss keys are full sentences).
    """
    manifest = phase2_output / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Phase 2 manifest not found: {manifest}")
    with open(manifest) as f:
        all_entries = json.load(f)

    if not all_entries:
        return all_entries

    if all_entries[0].get("dataset_source"):
        return [e for e in all_entries if e.get("filename", "").startswith("sentence_")]

    # Manual mode: glosses.json keys are the original sentences.
    glosses_file = phase1_output / "glosses.json"
    if not glosses_file.exists():
        return all_entries

    with open(glosses_file) as f:
        sentence_keys = set(k.strip() for k in json.load(f).keys())

    return [e for e in all_entries if e.get("sentence_text", "").strip() in sentence_keys]


def _get_pseudo_glosses(phase1_output: Path) -> dict[str, list[str]]:
    """Load glosses from Phase 1 output."""
    glosses_file = phase1_output / "glosses.json"
    if not glosses_file.exists():
        return {}
    with open(glosses_file) as f:
        return json.load(f)


_make_env = make_gpu_env


def _source_to_cache_key(source_path: Path) -> str | None:
    """Videos outside VIDEO_DATA_ROOT have unstable identity across tasks — skip them."""
    try:
        rel = source_path.resolve().relative_to(_VIDEO_DATA_ROOT_RESOLVED)
    except ValueError:
        return None
    return str(rel.with_suffix(""))


def _iter_feature_pairs(sentence_video_dir: Path, feat_dir: Path):
    """Yield (local_path, cached_path, key) for each cacheable video in the dir."""
    for mp4 in sentence_video_dir.glob("*.mp4"):
        key = _source_to_cache_key(mp4.resolve())
        if not key:
            continue
        local = feat_dir / f"{mp4.stem}{CLIP_FEATURE_SUFFIX}.npy"
        cached = CLIP_FEATURE_CACHE_DIR / f"{key}{CLIP_FEATURE_SUFFIX}.npy"
        yield local, cached, key


def _prepopulate_feature_cache(sentence_video_dir: Path, feat_dir: Path) -> int:
    """Reuse features extracted by prior tasks via symlink to skip re-extraction."""
    if not CLIP_FEATURE_CACHE_DIR.exists():
        return 0
    feat_dir.mkdir(parents=True, exist_ok=True)
    hits = 0
    for local, cached, key in _iter_feature_pairs(sentence_video_dir, feat_dir):
        if not cached.exists():
            continue
        try:
            local.symlink_to(cached)
            hits += 1
        except FileExistsError:
            pass
        except OSError as e:
            logger.debug(f"cache symlink failed for {key}: {e}")
    return hits


def _save_features_to_cache(sentence_video_dir: Path, feat_dir: Path) -> int:
    """Populate the shared cache so future tasks can skip re-extraction."""
    CLIP_FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for local, cached, key in _iter_feature_pairs(sentence_video_dir, feat_dir):
        if not local.exists() or local.is_symlink() or cached.exists():
            continue
        cached.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(local, cached)
            saved += 1
        except OSError as e:
            logger.debug(f"cache save failed for {key}: {e}")
    return saved


async def _extract_clip_features(
    task_id: str, video_dir: Path, output_dir: Path, gpu_id: int = 0
) -> int:
    """Step 4.1: Extract CLIP-ViT spatial features from videos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    script = P4_ROOT / "scripts" / "extract_features" / "extract_clip_from_mp4.py"
    cmd = [
        SPAMO_PYTHON, str(script),
        "--video_dir", str(video_dir),
        "--output_dir", str(output_dir),
        "--device", f"cuda:{gpu_id}",
        "--batch_size", "16",
        "--s2_mode", S2_MODE,
        "--scales", "1", "2",
    ]

    logger.info(f"[{task_id}] Step 4.1: Extracting CLIP features from {video_dir}")
    rc, stdout, stderr = await run_subprocess(
        cmd, cwd=P4_ROOT, env=_make_env(gpu_id), log_to_file=True
    )
    if rc != 0:
        raise RuntimeError(f"CLIP feature extraction failed (rc={rc}): {(stderr or stdout)[-500:]}")

    count = len(list(output_dir.glob(f"*{CLIP_FEATURE_SUFFIX}.npy")))
    if count == 0:
        raise RuntimeError(
            f"CLIP feature extraction produced 0 features. "
            f"video_dir={video_dir} (videos: {len(list(video_dir.glob('*.mp4')))}), "
            f"output_dir={output_dir}. Subprocess output: {(stdout or stderr)[-300:]}"
        )
    for split in ("train", "val", "dev", "test"):
        split_dir = output_dir / split
        if not split_dir.exists():
            split_dir.symlink_to(output_dir)

    logger.info(f"[{task_id}] Step 4.1: Extracted features for {count} videos")
    return count


def _iter_prev_task_sentences(current_task_id: str):
    """Yield (sentence_text, video_path) from previous tasks' phase_2 manifests."""
    with Session(engine) as session:
        task_ids = session.exec(
            select(PipelineTask.task_id).where(
                PipelineTask.task_id != current_task_id
            ).order_by(PipelineTask.created_at.desc())
        ).all()

    for tid in task_ids:
        manifest_path = settings.SHARED_DATA_ROOT / tid / "phase_2" / "output" / "manifest.json"
        videos_dir = settings.SHARED_DATA_ROOT / tid / "phase_2" / "output" / "videos"
        try:
            with open(manifest_path) as f:
                entries = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"Skipping prev task {tid}: {e}")
            continue
        for e in entries:
            text = (e.get("sentence_text") or "").strip()
            fn = e.get("filename") or ""
            if not text or not fn or fn.startswith("word_"):
                continue
            vp = videos_dir / fn
            if vp.exists():
                yield text, vp.resolve()


def _iter_csv_sentences(csv_path: Path, video_dir: Path, text_col: str, vid_col: str):
    """Yield (text, path) from a CSV/TSV with the given column names."""
    if not csv_path.exists():
        return
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            text = (row.get(text_col) or "").strip()
            vid = (row.get(vid_col) or "").strip()
            if not text or len(text) < 5 or not vid:
                continue
            p = video_dir / f"{vid}.mp4"
            if p.exists():
                yield text, p


def _iter_dataset_sentences():
    """Yield (sentence_text, video_path) from how2sign then openasl annotations."""
    yield from _iter_csv_sentences(H2S_FILTERED, H2S_DIR, "SENTENCE", "SENTENCE_NAME")
    yield from _iter_csv_sentences(OPENASL_FILTERED, OPENASL_DIR, "raw-text", "vid")


def _build_pad_entries(
    task_id: str,
    current_manifest: list[dict],
    target_count: int,
) -> list[dict]:
    """Build pad entries to reach target_count.

    Priority: previous tasks → how2sign → openasl. Dedup by lowercase sentence_text.
    """
    if target_count <= 0:
        return []
    need = target_count - len(current_manifest)
    if need <= 0:
        return []

    seen = set()
    for e in current_manifest:
        t = (e.get("sentence_text") or "").strip().lower()
        if t:
            seen.add(t)

    pad_entries: list[dict] = []
    counts = {"prev_task": 0, "dataset": 0}

    sources = [
        ("prev_task", _iter_prev_task_sentences(task_id)),
        ("dataset", _iter_dataset_sentences()),
    ]
    for source_name, iterator in sources:
        if len(pad_entries) >= need:
            break
        for text, path in iterator:
            if len(pad_entries) >= need:
                break
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            idx = len(pad_entries)
            pad_entries.append({
                "video_id": f"pad_{idx}",
                "filename": f"pad_{idx:05d}.mp4",
                "sentence_text": text,
                "language": "en",
                "__pad_source": source_name,
                "__pad_source_path": str(path),
            })
            counts[source_name] += 1

    logger.info(
        f"[{task_id}] Phase 4 pad: current={len(current_manifest)}, "
        f"target={target_count}, added={len(pad_entries)} "
        f"(prev_task={counts['prev_task']}, dataset={counts['dataset']}), "
        f"short={max(0, need - len(pad_entries))}"
    )
    return pad_entries


def _build_annotations(
    task_id: str,
    current_manifest: list[dict],
    pad_entries: list[dict],
    glosses: dict[str, list[str]],
    anno_dir: Path,
) -> tuple[int, int]:
    """Step 4.2: Generate train/val/test annotation files.

    Val is held out from pad at 9% of the full pool. Train mixes current-task
    + remaining pad. Test stays current-task-only so Phase 5's --mode test
    only segments videos we need.
    """
    anno_dir.mkdir(parents=True, exist_ok=True)

    gloss_map = {
        _norm_sentence(sent): " ".join(g.lower() for g in glist)
        for sent, glist in glosses.items()
    }

    # Fail fast for current-task sentences — they must have a pseudo-gloss.
    # Pad sentences (prev_task / how2sign / openasl) are pulled purely to
    # reach PHASE4_MIN_TRAINING_SENTENCES and may lack a gloss entry; fall
    # back to the raw text in that case.
    current_missing = [
        (item.get("filename", ""), item.get("sentence_text", ""))
        for item in current_manifest
        if _norm_sentence(item.get("sentence_text", "")) not in gloss_map
    ]
    if current_missing:
        preview = "\n".join(f"  {fn}  <- {txt!r}" for fn, txt in current_missing[:20])
        more = f"\n  ... (+{len(current_missing) - 20} more)" if len(current_missing) > 20 else ""
        raise RuntimeError(
            f"[{task_id}] {len(current_missing)} current-task sentence(s) have no "
            f"pseudo-gloss in Phase 1 glosses.json. SpaMo's 'text' field must be "
            f"pseudo-gloss (never raw English). Re-run Phase 1 so every sentence "
            f"in the manifest is covered.\n\nMissing:\n{preview}{more}"
        )

    pad_fallback = sum(
        1
        for item in pad_entries
        if _norm_sentence(item.get("sentence_text", "")) not in gloss_map
    )
    if pad_fallback:
        logger.warning(
            f"[{task_id}] {pad_fallback}/{len(pad_entries)} pad sentences lack "
            f"pseudo-gloss; using raw text as SpaMo target for those entries."
        )

    def to_entry(item):
        text = item.get("sentence_text", "")
        filename = item.get("filename", "")
        file_id = Path(filename).stem
        pseudo = gloss_map.get(_norm_sentence(text), text)
        return {
            "fileid": file_id,
            "folder": file_id,
            "text": pseudo,
            "original_text": text,
            "gloss": "",
            "start": 0.0,
            "end": 0.0,
            "video_name": filename,
        }

    current_entries = [to_entry(it) for it in current_manifest]
    pad_annotations = [to_entry(it) for it in pad_entries]

    total_n = len(current_entries) + len(pad_annotations)
    n_val = max(1, round(total_n * 0.09)) if total_n else 0
    n_val = min(n_val, len(pad_annotations))  # never take val from current_entries

    rng = random.Random(42)
    rng.shuffle(pad_annotations)
    val_part = pad_annotations[:n_val]
    train_pad = pad_annotations[n_val:]

    train_part = current_entries + train_pad
    test_part = current_entries

    rng.shuffle(train_part)

    np.save(anno_dir / "train_info_ml.npy", np.array(train_part, dtype=object))
    np.save(anno_dir / "val_info_ml.npy", np.array(val_part, dtype=object))
    np.save(anno_dir / "test_info_ml.npy", np.array(test_part, dtype=object))

    logger.info(
        f"[{task_id}] Step 4.2: Annotations — "
        f"train={len(train_part)} ({len(current_entries)} current + {len(train_pad)} padded), "
        f"val={len(val_part)} (held out from pad), "
        f"test={len(test_part)} (current only)"
    )
    return len(train_part), len(val_part)


def _generate_config(
    task_id: str,
    anno_dir: Path,
    feat_dir: Path,
    video_dir: Path,
    output_dir: Path,
) -> Path:
    """Step 4.3: Generate task-specific config YAML from template.

    Uses the COLENT variant (token-level OT + word continuity + column-entropy
    + position-contrastive losses) — assumes step 4.2.5 already produced
    concat-aug data with known per-token boundaries. warm_up_steps and
    max_epochs are not overridden here: test_real/phase4_seg_train/main.py
    auto-scales warm_up_steps from N_train at launch, and
    EarlyStopping(patience=50) handles early termination under the yaml
    default max_epochs=500.
    """
    template_path = P4_ROOT / "configs" / "chatsign_concat_aug_colent.yaml"
    config = OmegaConf.load(template_path)

    for split in ("train", "validation", "test"):
        p = config.data.params[split].params
        p.anno_root = str(anno_dir)
        p.feat_root = str(feat_dir)
        p.vid_root = str(video_dir)
        p.mae_feat_root = str(feat_dir)

    config.lightning.trainer.check_val_every_n_epoch = 2
    shared_hf_cache = (settings.SHARED_DATA_ROOT / ".hf_cache").resolve()
    shared_hf_cache.mkdir(parents=True, exist_ok=True)
    config.model.params.cache_dir = str(shared_hf_cache)

    config_path = (output_dir / f"config_{task_id}.yaml").resolve()
    OmegaConf.save(config, config_path)

    logger.info(
        f"[{task_id}] Step 4.3: Config saved "
        f"(variant=concat_aug_colent, max_epochs=yaml-default, warmup=auto via main.py)"
    )
    return config_path


async def _train_model(
    task_id: str, config_path: Path, output_dir: Path, gpu_id: int = 0
) -> Path:
    """Step 4.4: Train segmentation model."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        SPAMO_PYTHON, str(P4_ROOT / "main.py"),
        "--config", str(config_path),
        "--train", "true",
        "--logdir", str(log_dir),
        "--name", task_id,
        "--seed", "42",
    ]

    logger.info(f"[{task_id}] Step 4.4: Starting segmentation model training")
    rc, stdout, stderr = await run_subprocess(
        cmd, cwd=P4_ROOT, env=_make_env(gpu_id), log_to_file=True
    )

    ckpt_dir = None
    for d in sorted(log_dir.rglob("checkpoints"), reverse=True):
        if d.is_dir():
            ckpt_dir = d
            break

    if ckpt_dir is None:
        raise RuntimeError(f"Training produced no checkpoints. Output: {(stderr or stdout)[-1000:]}")

    best_ckpt = None
    for pattern in ("best_bleu4*.ckpt", "best_loss*.ckpt", "last*.ckpt"):
        matches = sorted(ckpt_dir.glob(pattern))
        if matches:
            best_ckpt = matches[-1]
            break

    if best_ckpt is None:
        raise RuntimeError(f"No checkpoint files in {ckpt_dir}")

    final_ckpt = output_dir / "segmentation_model.ckpt"
    shutil.copy2(best_ckpt, final_ckpt)

    logger.info(f"[{task_id}] Step 4.4: Training complete -> {final_ckpt.name}")
    return final_ckpt


async def run_phase4_segmentation_train(
    task_id: str,
    phase2_output: Path,
    phase1_output: Path,
    output_dir: Path,
    gpu_id: int = 0,
) -> dict:
    """Run segmentation training pipeline (Phase 4: training only, no inference)."""
    output_dir = output_dir.resolve()
    phase2_output = phase2_output.resolve()
    phase1_output = phase1_output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    current_manifest = _get_sentence_videos(phase2_output, phase1_output)
    glosses = _get_pseudo_glosses(phase1_output)

    if not current_manifest:
        raise RuntimeError("No sentence videos found in Phase 2 output")

    video_dir = phase2_output / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"No videos directory: {video_dir}")

    pad_entries = _build_pad_entries(
        task_id, current_manifest, settings.PHASE4_MIN_TRAINING_SENTENCES
    )

    # Symlink sentence-only videos into a clean directory
    sentence_video_dir = output_dir / "sentence_videos"
    sentence_video_dir.mkdir(parents=True, exist_ok=True)
    for entry in current_manifest:
        fn = entry.get("filename", "")
        src = video_dir / fn
        dst = sentence_video_dir / fn
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve() if src.is_symlink() else src)
    for entry in pad_entries:
        fn = entry["filename"]
        src = Path(entry["__pad_source_path"])
        dst = sentence_video_dir / fn
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())

    video_count = len(list(sentence_video_dir.glob("*.mp4")))
    logger.info(
        f"[{task_id}] Phase 4: {video_count} total videos "
        f"({len(current_manifest)} current + {len(pad_entries)} padded)"
    )

    if video_count == 0:
        raise RuntimeError("No sentence video files found on disk")

    feat_dir = output_dir / "features"
    try:
        hits = _prepopulate_feature_cache(sentence_video_dir, feat_dir)
        if hits:
            logger.info(f"[{task_id}] CLIP cache: {hits}/{video_count} pre-populated from shared cache")
    except Exception as e:
        logger.warning(f"[{task_id}] CLIP cache prepopulate failed (ignored): {e}")

    feat_count = await _extract_clip_features(task_id, sentence_video_dir, feat_dir, gpu_id)

    try:
        saved = _save_features_to_cache(sentence_video_dir, feat_dir)
        if saved:
            logger.info(f"[{task_id}] CLIP cache: {saved} new features saved to shared cache")
    except Exception as e:
        logger.warning(f"[{task_id}] CLIP cache save failed (ignored): {e}")

    anno_dir = output_dir / "annotations"
    n_train, n_val = _build_annotations(
        task_id, current_manifest, pad_entries, glosses, anno_dir
    )

    # Step 4.2.5: concat-aug — synthesize 36x training data from train_info_ml
    # base sentences + B (chatsign-accuracy approved word videos) + C (ASL-27K
    # dictionary). Auto-fallback to 21x if ORG hit-rate < 40%. C-class variants
    # silently drop when ASL-27K precompute features are missing — degraded mode
    # is still strictly more data than the 1x non-aug baseline.
    aug_anno_dir = output_dir / "aug_annotations"
    aug_feat_dir = output_dir / "aug_features"
    aug_anno_dir, aug_feat_dir, aug_summary = run_concat_aug(
        task_id=task_id,
        base_anno=anno_dir,
        base_feat=feat_dir,
        aug_anno_out=aug_anno_dir,
        aug_feat_out=aug_feat_dir,
        base_sentences_npy="train_info_ml.npy",
    )

    config_path = _generate_config(
        task_id, aug_anno_dir, aug_feat_dir, sentence_video_dir, output_dir
    )

    ckpt_path = await _train_model(task_id, config_path, output_dir, gpu_id)

    # Save the CURRENT task's manifest only — Phase 5 must not process padded videos.
    with open(output_dir / "sentence_manifest.json", "w") as f:
        json.dump(current_manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"[{task_id}] Phase 4 completed")
    return {
        "input_videos": len(current_manifest),
        "padded_sentences": len(pad_entries),
        "total_training_videos": video_count,
        "features_extracted": feat_count,
        "train_samples": n_train,
        "val_samples": n_val,
        "checkpoint": str(ckpt_path),
    }
