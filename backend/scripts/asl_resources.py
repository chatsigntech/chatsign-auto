"""Resolve ASL-27K mp4 + precomputed feature paths for a gloss list.

Provides resolve_asl_resources() — pure function used by build_concat_aug to
get C-class (asl_*) clips for token concatenation. No worker, no word_lib
materialization.

Inputs are lowercase_underscore tokens (from base_anno entry["text"].split()).
Returns dict keyed by the same form so build_concat_aug looks up directly.
"""
from __future__ import annotations

import logging
from pathlib import Path

from backend.core.dataset_videos import (
    ASL27K_FEATS,
    ASL27K_VIDEOS,
    _load_asl27k_gloss_map,
    normalize_gloss_token,
)

logger = logging.getLogger(__name__)


def resolve_asl_resources(
    glosses: list[str],
    max_per_gloss: int = 5,
) -> dict:
    """For each token, find ASL-27K mp4 + cached *_s2wrapping.npy.

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
    """
    gloss_map = _load_asl27k_gloss_map()  # {lowercase_word: [filenames]}
    out = {
        "resources": {},
        "missing": [],
        "feat_missing_files": [],
        "n_glosses_hit": 0,
        "n_clips_total": 0,
    }
    for gloss in glosses:
        key = normalize_gloss_token(gloss)
        filenames = gloss_map.get(key, [])
        if not filenames:
            out["missing"].append(gloss)
            continue
        pairs = []
        for fn in filenames[:max_per_gloss]:
            src_mp4 = ASL27K_VIDEOS / fn
            if not src_mp4.exists():
                continue
            src_npy = ASL27K_FEATS / f"{src_mp4.stem}_s2wrapping.npy"
            if src_npy.exists():
                pairs.append((src_mp4, src_npy))
            else:
                out["feat_missing_files"].append(src_mp4.name)
        if pairs:
            out["resources"][gloss] = pairs
            out["n_glosses_hit"] += 1
            out["n_clips_total"] += len(pairs)
    hit_rate = out["n_glosses_hit"] / max(len(glosses), 1)
    logger.info(
        f"ASL-27K resolved: {out['n_glosses_hit']}/{len(glosses)} glosses ({hit_rate:.1%}), "
        f"{out['n_clips_total']} clips total; missing={len(out['missing'])}, "
        f"feat_missing={len(out['feat_missing_files'])}"
    )
    if out["feat_missing_files"]:
        logger.warning(
            f"{len(out['feat_missing_files'])} videos lack precomputed features; "
            f"re-run precompute_asl27k_features.py to fill"
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
        ap.error("must pass --glosses OR --from-anno")

    result = resolve_asl_resources(tokens, max_per_gloss=args.max_per_gloss)
    print(json.dumps({k: v if k != "resources" else f"<{len(v)} entries>"
                      for k, v in result.items()}, indent=2, default=str))
    if args.show_missing and result["missing"]:
        print("\nMissing glosses:")
        for g in result["missing"][:50]:
            print(f"  {g}")
