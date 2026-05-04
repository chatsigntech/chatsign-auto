"""One-time precompute of CLIP-ViT-L/14 + S2-wrapping features for ASL-27K.

Outputs to /mnt/data/chatsign-auto-videos/clip_features/ASL-final-27K-202603/videos/<stem>_s2wrapping.npy
which is what backend/scripts/asl_resources.py reads.

The bulk run is 27090 mp4s × ~70 frames × dual-scale ≈ 1.9M frames ≈ 5-9 hr on 5090 Laptop.
Use --limit N for sanity check (50 ≈ 3 min).

Idempotent: skips videos whose npy already exists (via SPAMO's get_pending_videos).
Failed mp4 (corrupt / 0 frame): SPAMO's extract_one logs a warning and continues.

Usage:
    conda activate spamo
    python -m backend.scripts.precompute_asl27k_features --gpu 0 --limit 50  # sanity
    python -m backend.scripts.precompute_asl27k_features --gpu 0              # full
"""
from __future__ import annotations

import argparse
import logging
import sys

from backend.core.dataset_videos import ASL27K_FEATS, ASL27K_VIDEOS
from backend.scripts._clip_extract import precompute_features_for_dirs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N pending mp4s (sanity check). None = full 27090.",
    )
    ap.add_argument("--s2-mode", type=str, default="s2wrapping")
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not ASL27K_VIDEOS.exists():
        logging.getLogger().error(f"ASL-27K videos dir not found: {ASL27K_VIDEOS}")
        return 2

    result = precompute_features_for_dirs(
        dirs=[(ASL27K_VIDEOS, ASL27K_FEATS)],
        gpu=args.gpu,
        batch_size=args.batch_size,
        s2_mode=args.s2_mode,
        scales=args.scales,
        limit_per_dir=args.limit,
    )
    return 0 if result["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
