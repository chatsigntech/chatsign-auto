"""One-time precompute of CLIP features for chatsign-accuracy reviewer-uploaded word videos.

Outputs to /mnt/data/chatsign-auto-videos/clip_features/accuracy_word_uploads/<reviewer>/<stem>_s2wrapping.npy
which is what backend/scripts/org_resources.py reads.

Per-reviewer iteration because chatsign-accuracy uploads are split across reviewer subdirs:
  Tareq/ Heba/ Rawdah/ chatsign2026admin/ test/

extract_clip_from_mp4 only does flat *.mp4 glob (non-recursive), so we feed
each reviewer subdir as a separate (video_dir, output_dir) pair to the shared
extractor.

By default, --approved-only filters to ~838 mp4 (those already approved by
review), saving ~57% GPU compute vs processing all 1969. Pass --include-pending
to precompute everything (useful if you expect approvals to come in later).

Total ≈ 838 mp4 × ~60 frames ÷ 80-120 fps ≈ 10-15 min on 5090 Laptop.
Idempotent (skip-existing per-reviewer).

Usage:
    conda activate spamo
    python -m backend.scripts.precompute_accuracy_word_features --gpu 0
    # All reviewers, include pending (slower but caches future approvals):
    python -m backend.scripts.precompute_accuracy_word_features --gpu 0 --include-pending
    # Single reviewer:
    python -m backend.scripts.precompute_accuracy_word_features --gpu 0 --reviewer Tareq
"""
from __future__ import annotations

import argparse
import logging
import sys

from backend.config import settings
from backend.core.io_utils import read_jsonl
from backend.scripts._clip_extract import precompute_features_for_dirs

UPLOADS_DIR = settings.CHATSIGN_ACCURACY_DATA / "uploads" / "videos"
ORG_FEATS = settings.VIDEO_DATA_ROOT / "clip_features" / "accuracy_word_uploads"
REVIEW_DECISIONS = settings.CHATSIGN_ACCURACY_DATA / "reports" / "review-decisions.jsonl"


def _load_approved_filenames() -> set[str]:
    approved: set[str] = set()
    for r in read_jsonl(REVIEW_DECISIONS):
        if r.get("decision") != "approved":
            continue
        vinfo = r.get("videoInfo") or {}
        fn = vinfo.get("videoFileName") or r.get("videoFileName")
        if fn:
            approved.add(fn)
    return approved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--reviewer",
        type=str,
        default=None,
        help="Limit to a single reviewer subdir (e.g. Tareq). Default: all.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap processed mp4s per reviewer (sanity check)",
    )
    ap.add_argument(
        "--include-pending",
        action="store_true",
        help="Process all mp4s (default: approved-only via review-decisions.jsonl)",
    )
    ap.add_argument("--s2-mode", type=str, default="s2wrapping")
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not UPLOADS_DIR.exists():
        logging.getLogger().error(f"accuracy uploads dir not found: {UPLOADS_DIR}")
        return 2

    reviewer_dirs = sorted(
        d for d in UPLOADS_DIR.iterdir()
        if d.is_dir() and (args.reviewer is None or d.name == args.reviewer)
    )
    if not reviewer_dirs:
        logging.getLogger().error(
            f"No reviewer subdirs found "
            f"({'matching --reviewer ' + args.reviewer if args.reviewer else 'in uploads/videos/'})"
        )
        return 2

    extra_filter = None
    if not args.include_pending:
        approved = _load_approved_filenames()
        logging.getLogger().info(
            f"approved-only filter: {len(approved)} approved mp4s "
            f"(pass --include-pending to process all)"
        )
        extra_filter = lambda mp4: mp4.name in approved

    dirs = [(d, ORG_FEATS / d.name) for d in reviewer_dirs]
    result = precompute_features_for_dirs(
        dirs=dirs,
        gpu=args.gpu,
        batch_size=args.batch_size,
        s2_mode=args.s2_mode,
        scales=args.scales,
        limit_per_dir=args.limit,
        failed_log_root=ORG_FEATS,
        extra_filter=extra_filter,
    )
    return 0 if result["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
