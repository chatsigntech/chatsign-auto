"""Prepare videos from OpenASL/How2Sign datasets for the pipeline.

When task source is "dataset", this module:
1. Locates sentence-level videos from OpenASL/How2Sign
2. Prepares Phase 2 output format (manifest.json + videos/ symlinks)
"""
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

from backend.config import settings
from backend.core.io_utils import read_jsonl

# Dataset video directories
OPENASL_DIR = settings.VIDEO_DATA_ROOT / "opensl_data"
H2S_DIR = settings.VIDEO_DATA_ROOT / "how2sign_data"

# H2S / OpenASL upstream artifacts (test_real preprocess outputs, mirrored
# locally) — used by P4 to read pseudo-gloss text and reuse pre-computed
# CLIP features for pad entries.
H2S_INFO_ML_TRAIN = H2S_DIR / "spamo_anno" / "train_info_ml.npy"
H2S_FEATS_TRAIN = settings.VIDEO_DATA_ROOT / "clip_features" / "how2sign_data" / "train"
OPENASL_TRAIN_TSV = OPENASL_DIR / "annotations" / "openasl-v1.0-train.tsv"
OPENASL_FEATS = settings.VIDEO_DATA_ROOT / "clip_features" / "opensl_data"

# Uni-Sign ASL pool — C-class source for P4 concat-aug (matches test_real
# upstream's word_lib/<WORD>/asl_*).
UNISIGN_ASL_DIR = settings.VIDEO_DATA_ROOT / "unisign_asl_data"
UNISIGN_ASL_VIDEOS = UNISIGN_ASL_DIR / "videos"
UNISIGN_ASL_TRAIN_JSONL = UNISIGN_ASL_DIR / "train.jsonl"
UNISIGN_ASL_FEATS = settings.VIDEO_DATA_ROOT / "clip_features" / "unisign_asl_data"

# chatsign-accuracy reviewer-uploaded word videos (ORG / B-class)
ORG_UPLOADS_DIR = settings.CHATSIGN_ACCURACY_DATA / "uploads" / "videos"
ORG_REVIEW_DECISIONS = settings.CHATSIGN_ACCURACY_DATA / "reports" / "review-decisions.jsonl"
ORG_FEATS = settings.VIDEO_DATA_ROOT / "clip_features" / "accuracy_word_uploads"


def load_approved_video_filenames() -> set[str]:
    """Set of videoFileNames that reviewers approved.

    Falls back from videoInfo.videoFileName to top-level videoFileName
    because the schema changed over time.
    """
    approved: set[str] = set()
    for r in read_jsonl(ORG_REVIEW_DECISIONS):
        if r.get("decision") != "approved":
            continue
        vinfo = r.get("videoInfo") or {}
        fn = vinfo.get("videoFileName") or r.get("videoFileName")
        if fn:
            approved.add(fn)
    return approved


def extract_tokens_from_entries(entries) -> list[str]:
    """Sorted unique whitespace-split tokens from a list of anno entries.

    Tokens are already lowercased for current_entries by phase4_segmentation_train,
    may be raw text for pad entries lacking pseudo-gloss (those tokens won't
    match resolver libs and get dropped).
    """
    tokens: set[str] = set()
    for entry in entries:
        if isinstance(entry, dict):
            tokens.update(entry.get("text", "").split())
    return sorted(tokens)


def extract_tokens_from_anno(base_anno: Path, filename: str = "test_info_ml.npy") -> list[str]:
    """Sorted unique tokens from <base_anno>/<filename>.

    Convenience wrapper: loads the npy then calls extract_tokens_from_entries.
    Callers that already have the entries loaded should call the entries
    helper directly to avoid re-reading the file.
    """
    path = base_anno / filename
    if not path.exists():
        raise RuntimeError(f"{filename} not found at {path}")
    return extract_tokens_from_entries(np.load(path, allow_pickle=True))


def normalize_gloss_token(token: str) -> str:
    """Anno text token (lowercase_underscore) → resource-key form (lower with spaces).

    P1 emits glosses as UPPER_UNDERSCORE; phase4_segmentation_train lowercases
    them when joining into anno text. Both forms collapse to "lower with spaces"
    for Uni-Sign / accuracy-uploads matching.

    Examples:
        more_than -> "more than"
        MORE_THAN -> "more than"   (idempotent on case)
        home      -> "home"
    """
    return token.strip().lower().replace("_", " ")


_unisign_asl_gloss_map: dict[str, list[str]] | None = None


def _load_unisign_asl_gloss_map() -> dict[str, list[str]]:
    """phrase → [utterance_id, ...] from Uni-Sign train.jsonl.

    Each line schema: {"utterance_id": str, "tokens": list[str], ...}.
    Phrase key = " ".join(tokens).lower(), matching normalize_gloss_token output.
    """
    global _unisign_asl_gloss_map
    if _unisign_asl_gloss_map is not None:
        return _unisign_asl_gloss_map

    _unisign_asl_gloss_map = {}
    if not UNISIGN_ASL_TRAIN_JSONL.exists():
        logger.warning(f"Uni-Sign train.jsonl not found: {UNISIGN_ASL_TRAIN_JSONL}")
        return _unisign_asl_gloss_map

    for r in read_jsonl(UNISIGN_ASL_TRAIN_JSONL):
        phrase = " ".join(r.get("tokens", [])).strip().lower()
        uid = r.get("utterance_id", "").strip()
        if phrase and uid:
            _unisign_asl_gloss_map.setdefault(phrase, []).append(uid)

    logger.info(f"Uni-Sign ASL gloss map loaded: {len(_unisign_asl_gloss_map)} unique phrases")
    return _unisign_asl_gloss_map


def _find_video(vid: str, source: str) -> Path | None:
    """Locate video file by vid and source dataset."""
    if source == "openasl":
        path = OPENASL_DIR / f"{vid}.mp4"
    elif source == "how2sign":
        path = H2S_DIR / f"{vid}.mp4"
    else:
        return None
    return path if path.exists() else None


def prepare_dataset_videos(
    task_id: str,
    dataset_videos: list[dict],
    phase2_output: Path,
) -> dict:
    """Symlink dataset videos into Phase 2 output format.

    Args:
        task_id: Pipeline task ID
        dataset_videos: List of {text, vid, source} for sentence videos
        phase2_output: Phase 2 output directory to populate

    Returns:
        dict with video_count, sentences, missing, manifest_path
    """
    phase2_output.mkdir(parents=True, exist_ok=True)
    videos_dir = phase2_output / "videos"
    videos_dir.mkdir(exist_ok=True)

    manifest = []
    sentences = set()
    found = 0
    missing = 0

    for i, entry in enumerate(dataset_videos):
        vid = entry.get("vid", "")
        source = entry.get("source", "")
        text = entry.get("text", "")

        src_path = _find_video(vid, source)
        if not src_path:
            logger.warning(f"[{task_id}] Dataset video not found: {vid} ({source})")
            missing += 1
            continue

        filename = f"sentence_{i}.mp4"
        dst = videos_dir / filename

        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src_path.resolve())

        manifest.append({
            "video_id": f"ds_{i}",
            "filename": filename,
            "sentence_id": i,
            "sentence_text": text,
            "language": "en",
            "dataset_source": source,
            "dataset_vid": vid,
        })
        sentences.add(text)
        found += 1

    manifest_path = phase2_output / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    with open(phase2_output / "sentences.txt", "w", encoding="utf-8") as f:
        for s in sorted(sentences):
            f.write(s + "\n")

    logger.info(
        f"[{task_id}] Dataset videos prepared: {found} linked, {missing} missing"
    )

    return {
        "video_count": found,
        "missing": missing,
        "sentences": sorted(sentences),
        "manifest_path": str(manifest_path),
    }
