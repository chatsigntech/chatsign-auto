"""Resolve Uni-Sign ASL pool mp4 + precomputed feature paths for a gloss list.

Provides resolve_asl_resources() — pure function used by build_concat_aug to
get C-class (asl_*) clips for token concatenation. Source matches test_real
upstream's `word_lib/<WORD>/asl_*.mp4` flow (populated by import_asl_videos.py
from the Uni-Sign / SLRT ASL pool).

Inputs are lowercase_underscore tokens (from base_anno entry["text"].split()).
Returns dict keyed by the same form so build_concat_aug looks up directly.
"""
from __future__ import annotations

import logging
from pathlib import Path

from backend.core.dataset_videos import (
    UNISIGN_ASL_FEATS,
    UNISIGN_ASL_VIDEOS,
    _load_unisign_asl_gloss_map,
    normalize_gloss_token,
)

logger = logging.getLogger(__name__)


def resolve_asl_resources(
    glosses: list[str],
    max_per_gloss: int = 5,
) -> dict:
    """For each token, find Uni-Sign ASL pool mp4 + cached *_s2wrapping.npy.

    Args:
        glosses: list of lowercase_underscore tokens (from anno text)
        max_per_gloss: cap on candidates returned per gloss

    Returns:
        {
            "resources": {token: [(mp4_path, npy_path), ...]},  # key matches anno tokens
            "missing": [token, ...],
            "feat_missing_files": [mp4_filename, ...],
            "n_glosses_hit": int,
            "n_clips_total": int,
        }

    Note: build_concat_aug only consumes npy_path; mp4_path may not exist
    locally (videos are 6.4G, optional for runtime). We gate on npy existence.
    """
    gloss_map = _load_unisign_asl_gloss_map()  # {phrase: [utterance_ids]}
    out = {
        "resources": {},
        "missing": [],
        "feat_missing_files": [],
        "n_glosses_hit": 0,
        "n_clips_total": 0,
    }
    for gloss in glosses:
        key = normalize_gloss_token(gloss)
        uids = gloss_map.get(key, [])
        if not uids:
            out["missing"].append(gloss)
            continue
        pairs = []
        for uid in uids[:max_per_gloss]:
            src_npy = UNISIGN_ASL_FEATS / f"{uid}_s2wrapping.npy"
            if not src_npy.exists():
                out["feat_missing_files"].append(src_npy.name)
                continue
            src_mp4 = UNISIGN_ASL_VIDEOS / f"{uid}.mp4"
            pairs.append((src_mp4, src_npy))
        if pairs:
            out["resources"][gloss] = pairs
            out["n_glosses_hit"] += 1
            out["n_clips_total"] += len(pairs)
    hit_rate = out["n_glosses_hit"] / max(len(glosses), 1)
    logger.info(
        f"Uni-Sign ASL resolved: {out['n_glosses_hit']}/{len(glosses)} glosses ({hit_rate:.1%}), "
        f"{out['n_clips_total']} clips total; missing={len(out['missing'])}, "
        f"feat_missing={len(out['feat_missing_files'])}"
    )
    if out["feat_missing_files"]:
        logger.warning(
            f"{len(out['feat_missing_files'])} clips lack precomputed features; "
            f"sync from LAN (data-003/spamo/asl/features/) or rerun feature extraction"
        )
    return out


if __name__ == "__main__":
    import argparse
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--glosses",
        type=str,
        help="Comma-separated tokens (lowercase_underscore), e.g. 'home,more_than,go'",
    )
    ap.add_argument(
        "--from-anno",
        type=Path,
        help="Path to base_anno dir; will read test_info_ml.npy and extract tokens",
    )
    ap.add_argument("--max-per-gloss", type=int, default=5)
    ap.add_argument("--show-missing", action="store_true")
    args = ap.parse_args()

    if args.glosses:
        tokens = sorted({t.strip() for t in args.glosses.split(",") if t.strip()})
    elif args.from_anno:
        from backend.core.dataset_videos import extract_tokens_from_anno
        tokens = extract_tokens_from_anno(args.from_anno)
        print(f"Extracted {len(tokens)} unique tokens from {args.from_anno}")
    else:
        ap.error("must pass --glosses OR --from-anno")

    result = resolve_asl_resources(tokens, max_per_gloss=args.max_per_gloss)
    print(json.dumps({k: v if k != "resources" else f"<{len(v)} entries>"
                      for k, v in result.items()}, indent=2, default=str))
    if args.show_missing and result["missing"]:
        print("\nMissing glosses:")
        for g in result["missing"][:50]:
            print(f"  {g}")
