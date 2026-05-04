"""Resolve chatsign-accuracy approved word videos for a gloss list.

Provides resolve_org_resources() — pure function used by build_concat_aug to
get B-class (org_*) clips for token concatenation.

Inputs are lowercase_underscore tokens. Returns dict keyed by the same form so
build_concat_aug looks up directly.

Data sources (chatsign-accuracy):
  reports/word-glosses.json     — {filename: {alternate_words, gloss, synset_*}}
  reports/review-decisions.jsonl — approved videos
  uploads/videos/<reviewer>/*.mp4 — actual mp4 files
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from backend.config import settings
from backend.core.dataset_videos import normalize_gloss_token
from backend.core.io_utils import read_jsonl

logger = logging.getLogger(__name__)

ACCURACY_DATA = settings.CHATSIGN_ACCURACY_DATA
WORD_GLOSSES_JSON = ACCURACY_DATA / "reports" / "word-glosses.json"
REVIEW_DECISIONS = ACCURACY_DATA / "reports" / "review-decisions.jsonl"
UPLOADS_DIR = ACCURACY_DATA / "uploads" / "videos"
ORG_FEATS = settings.VIDEO_DATA_ROOT / "clip_features" / "accuracy_word_uploads"

_index_cache: dict[str, list[tuple[Path, Path]]] | None = None


def _build_org_index() -> dict[str, list[tuple[Path, Path]]]:
    """phrase → [(mp4_path, npy_path), ...]   only approved + mp4 exists.

    Caches the result for repeated calls within the same process.
    """
    global _index_cache
    if _index_cache is not None:
        return _index_cache

    # 1. approved set (deduped)
    approved: set[str] = set()
    for r in read_jsonl(REVIEW_DECISIONS):
        if r.get("decision") != "approved":
            continue
        vinfo = r.get("videoInfo") or {}
        fn = vinfo.get("videoFileName") or r.get("videoFileName")
        if fn:
            approved.add(fn)

    # 2. existing mp4s (per-reviewer subdirs)
    fn_to_path: dict[str, Path] = {}
    if not UPLOADS_DIR.exists():
        logger.warning(f"uploads dir not found: {UPLOADS_DIR}")
    else:
        for reviewer_dir in UPLOADS_DIR.iterdir():
            if not reviewer_dir.is_dir():
                continue
            for mp4 in reviewer_dir.glob("*.mp4"):
                fn_to_path[mp4.name] = mp4

    # 3. join: mp4 exists ∩ approved ∩ in word-glosses metadata
    if not WORD_GLOSSES_JSON.exists():
        logger.warning(f"word-glosses.json not found: {WORD_GLOSSES_JSON}")
        _index_cache = {}
        return _index_cache

    with open(WORD_GLOSSES_JSON) as f:
        glosses_meta = json.load(f)

    index: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    n_qualified = 0
    for fn, info in glosses_meta.items():
        if fn not in approved or fn not in fn_to_path:
            continue
        n_qualified += 1
        mp4_path = fn_to_path[fn]
        npy_path = ORG_FEATS / mp4_path.parent.name / f"{mp4_path.stem}_s2wrapping.npy"
        # split alternate_words: 'hello, hullo, hi' → 'hello' / 'hullo' / 'hi'
        alt = info.get("alternate_words") or ""
        for w in alt.split(","):
            w = w.strip().lower()
            if w:
                index[w].append((mp4_path, npy_path))

    _index_cache = dict(index)
    logger.info(
        f"org index built: {len(_index_cache)} unique words from "
        f"{n_qualified} approved mp4s "
        f"(approved={len(approved)}, mp4_exists={len(fn_to_path)}, "
        f"meta={len(glosses_meta)})"
    )
    return _index_cache


def resolve_org_resources(
    glosses: list[str],
    max_per_gloss: int = 5,
) -> dict:
    """For each token, find approved chatsign-accuracy mp4 + cached npy.

    Args:
        glosses: list of lowercase_underscore tokens (from anno text)
        max_per_gloss: cap on candidates returned per gloss

    Returns:
        {
            "resources": {token: [(mp4_path, npy_path), ...]},
            "missing": [token, ...],
            "feat_missing_files": [mp4_filename, ...],
            "n_glosses_hit": int,
            "n_clips_total": int,
        }
    """
    index = _build_org_index()
    out = {
        "resources": {},
        "missing": [],
        "feat_missing_files": [],
        "n_glosses_hit": 0,
        "n_clips_total": 0,
    }
    for gloss in glosses:
        key = normalize_gloss_token(gloss)
        candidates = index.get(key, [])
        if not candidates:
            out["missing"].append(gloss)
            continue
        pairs = []
        for mp4_path, npy_path in candidates[:max_per_gloss]:
            if npy_path.exists():
                pairs.append((mp4_path, npy_path))
            else:
                out["feat_missing_files"].append(mp4_path.name)
        if pairs:
            out["resources"][gloss] = pairs
            out["n_glosses_hit"] += 1
            out["n_clips_total"] += len(pairs)
    hit_rate = out["n_glosses_hit"] / max(len(glosses), 1)
    logger.info(
        f"accuracy org resolved: {out['n_glosses_hit']}/{len(glosses)} glosses ({hit_rate:.1%}), "
        f"{out['n_clips_total']} clips total; missing={len(out['missing'])}, "
        f"feat_missing={len(out['feat_missing_files'])}"
    )
    if out["feat_missing_files"]:
        logger.warning(
            f"{len(out['feat_missing_files'])} videos lack precomputed features; "
            f"re-run precompute_accuracy_word_features.py to fill"
        )
    return out


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--glosses",
        type=str,
        help="Comma-separated tokens (lowercase_underscore)",
    )
    ap.add_argument(
        "--from-anno",
        type=Path,
        help="Path to base_anno dir; will read test_info_ml.npy and extract tokens",
    )
    ap.add_argument("--max-per-gloss", type=int, default=5)
    ap.add_argument("--show-missing", action="store_true")
    ap.add_argument(
        "--show-index-stats",
        action="store_true",
        help="Print index size + sample words after building",
    )
    args = ap.parse_args()

    if args.show_index_stats:
        idx = _build_org_index()
        print(f"index size: {len(idx)} unique words")
        print(f"sample (first 20): {sorted(idx.keys())[:20]}")
        sys.exit(0)

    if args.glosses:
        tokens = sorted({t.strip() for t in args.glosses.split(",") if t.strip()})
    elif args.from_anno:
        import numpy as np
        path = args.from_anno / "test_info_ml.npy"
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            sys.exit(2)
        tokens_set = set()
        for entry in np.load(path, allow_pickle=True):
            text = entry.get("text", "") if isinstance(entry, dict) else ""
            tokens_set.update(text.split())
        tokens = sorted(tokens_set)
        print(f"Extracted {len(tokens)} unique tokens from {path}")
    else:
        ap.error("must pass --glosses OR --from-anno OR --show-index-stats")

    result = resolve_org_resources(tokens, max_per_gloss=args.max_per_gloss)
    print(json.dumps({k: v if k != "resources" else f"<{len(v)} entries>"
                      for k, v in result.items()}, indent=2, default=str))
    if args.show_missing and result["missing"]:
        print("\nMissing glosses:")
        for g in result["missing"][:50]:
            print(f"  {g}")
