"""Prepare videos from OpenASL/How2Sign/ASL-27K datasets for the pipeline.

When task source is "dataset", this module:
1. Locates sentence-level videos from OpenASL/How2Sign
2. Matches gloss words to ASL-27K word-level videos
3. Prepares Phase 2 output format (manifest.json + videos/ symlinks)
"""
import csv
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
ASL27K_DIR = settings.VIDEO_DATA_ROOT / "ASL-final-27K-202603"
ASL27K_VIDEOS = ASL27K_DIR / "videos"
ASL27K_GLOSS_CSV = ASL27K_DIR / "gloss.csv"
ASL27K_FEATS = settings.VIDEO_DATA_ROOT / "clip_features" / "ASL-final-27K-202603" / "videos"

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


def extract_tokens_from_anno(base_anno: Path, filename: str = "test_info_ml.npy") -> list[str]:
    """Sorted unique tokens from <base_anno>/<filename>.

    Tokens are anno-text whitespace splits — already lowercased for current_entries
    by phase4_segmentation_train, may be raw text for pad entries lacking
    pseudo-gloss (those tokens won't match resolver libs and get dropped).
    """
    path = base_anno / filename
    if not path.exists():
        raise RuntimeError(f"{filename} not found at {path}")
    tokens: set[str] = set()
    for entry in np.load(path, allow_pickle=True):
        if isinstance(entry, dict):
            tokens.update(entry.get("text", "").split())
    return sorted(tokens)


def normalize_gloss_token(token: str) -> str:
    """Anno text token (lowercase_underscore) → gloss-csv key (lower with spaces).

    P1 emits glosses as UPPER_UNDERSCORE; phase4_segmentation_train lowercases
    them when joining into anno text. Both forms collapse to "lower with spaces"
    for csv / alternate_words matching.

    Examples:
        more_than -> "more than"
        MORE_THAN -> "more than"   (idempotent on case)
        home      -> "home"
    """
    return token.strip().lower().replace("_", " ")


# Cached gloss→video mapping (loaded once)
_asl27k_gloss_map: dict[str, list[str]] | None = None


def _load_asl27k_gloss_map() -> dict[str, list[str]]:
    """Load ASL-27K gloss.csv into a word→[filenames] mapping.

    Keys are lowercased for case-insensitive matching.
    """
    global _asl27k_gloss_map
    if _asl27k_gloss_map is not None:
        return _asl27k_gloss_map

    _asl27k_gloss_map = {}
    if not ASL27K_GLOSS_CSV.exists():
        logger.warning(f"ASL-27K gloss.csv not found: {ASL27K_GLOSS_CSV}")
        return _asl27k_gloss_map

    with open(ASL27K_GLOSS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get("word", "").strip().lower()
            ref = row.get("ref", "").strip()
            if word and ref:
                _asl27k_gloss_map.setdefault(word, []).append(ref)

    logger.info(f"ASL-27K gloss map loaded: {len(_asl27k_gloss_map)} unique words")
    return _asl27k_gloss_map


def _find_video(vid: str, source: str) -> Path | None:
    """Locate video file by vid and source dataset."""
    if source == "openasl":
        path = OPENASL_DIR / f"{vid}.mp4"
    elif source == "how2sign":
        path = H2S_DIR / f"{vid}.mp4"
    else:
        return None
    return path if path.exists() else None


def _find_gloss_videos(gloss: str, max_per_gloss: int = 1) -> list[Path]:
    """Find ASL-27K videos matching a gloss word.

    Returns up to max_per_gloss video paths.
    """
    gloss_map = _load_asl27k_gloss_map()
    key = gloss.strip().lower()
    filenames = gloss_map.get(key, [])

    results = []
    for fn in filenames[:max_per_gloss]:
        path = ASL27K_VIDEOS / fn
        if path.exists():
            results.append(path)
    return results


def prepare_dataset_videos(
    task_id: str,
    dataset_videos: list[dict],
    phase2_output: Path,
    glosses: list[str] | None = None,
) -> dict:
    """Symlink dataset videos into Phase 2 output format.

    Args:
        task_id: Pipeline task ID
        dataset_videos: List of {text, vid, source} for sentence videos
        phase2_output: Phase 2 output directory to populate
        glosses: List of gloss words to match against ASL-27K

    Returns:
        dict with video_count, sentences, gloss_videos, missing, manifest_path
    """
    phase2_output.mkdir(parents=True, exist_ok=True)
    videos_dir = phase2_output / "videos"
    videos_dir.mkdir(exist_ok=True)

    manifest = []
    sentences = set()
    found = 0
    missing = 0

    # 1. Sentence videos from OpenASL/How2Sign
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

    # 2. Word videos from ASL-27K (matched by gloss)
    gloss_found = 0
    gloss_missing = 0
    if glosses:
        for gloss in glosses:
            videos = _find_gloss_videos(gloss)
            if not videos:
                gloss_missing += 1
                continue

            for vid_path in videos:
                # Sanitize gloss for filename
                safe_gloss = "".join(c if c.isalnum() or c in "_-" else "_" for c in gloss)
                filename = f"word_{safe_gloss}.mp4"
                dst = videos_dir / filename

                if dst.exists() or dst.is_symlink():
                    # Already linked (duplicate gloss)
                    gloss_found += 1
                    continue

                dst.symlink_to(vid_path.resolve())

                manifest.append({
                    "video_id": f"gloss_{safe_gloss}",
                    "filename": filename,
                    "sentence_text": gloss,
                    "language": "en",
                    "dataset_source": "asl27k",
                    "dataset_vid": vid_path.stem,
                    "glosses": [gloss.upper()],
                })
                gloss_found += 1
                found += 1

    # Write manifest.json
    manifest_path = phase2_output / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Write sentences.txt
    with open(phase2_output / "sentences.txt", "w", encoding="utf-8") as f:
        for s in sorted(sentences):
            f.write(s + "\n")

    logger.info(
        f"[{task_id}] Dataset videos prepared: {found} linked "
        f"({found - gloss_found} sentences, {gloss_found} glosses), "
        f"{missing} sentence missing, {gloss_missing} gloss missing"
    )

    return {
        "video_count": found,
        "sentence_videos": found - gloss_found,
        "gloss_videos": gloss_found,
        "gloss_missing": gloss_missing,
        "missing": missing,
        "sentences": sorted(sentences),
        "manifest_path": str(manifest_path),
    }
