"""Shared SPAMO CLIP-ViT-L/14 extraction wrapper.

Used by precompute_asl27k_features.py and precompute_accuracy_word_features.py.

Centralizes:
- sys.path patching for SPAMO submodule
- Module imports (with friendly error if 'spamo' env not active)
- Per-mp4 extraction loop with progress logging + ETA
- Failure tracking → failed.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from backend.config import settings

logger = logging.getLogger(__name__)


def _import_spamo_extractor():
    """Lazy import of SPAMO extractor (requires sys.path patch + GPU env)."""
    sys.path.insert(0, str(settings.SPAMO_SEGMENT_PATH))
    try:
        from scripts.extract_features.extract_clip_from_mp4 import (  # type: ignore
            extract_one,
            get_pending_videos,
        )
        from scripts.extract_features.vit_extract_feature import (  # type: ignore
            ViTFeatureReader,
        )
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import SPAMO extract_features module ({e}). "
            "Ensure conda env 'spamo' (or compatible: torch + transformers + "
            "open-clip-torch + einops + opencv) is active."
        ) from e
    return get_pending_videos, extract_one, ViTFeatureReader


def precompute_features_for_dirs(
    dirs: list[tuple[Path, Path]],
    *,
    gpu: int = 0,
    batch_size: int = 32,
    s2_mode: str = "s2wrapping",
    scales: list[int] = [1, 2],
    limit_per_dir: int | None = None,
    failed_log_root: Path | None = None,
    extra_filter: callable | None = None,
) -> dict:
    """Run CLIP feature extraction for one or more (video_dir, output_dir) pairs.

    Args:
        dirs: list of (video_dir, output_dir) — output is sibling-flat per dir
        gpu: cuda device index
        batch_size: extract_one batch size
        s2_mode: 's2wrapping' (default) or '' for no S2
        scales: S2 scales (default [1, 2] = dual-scale)
        limit_per_dir: cap pending mp4s per dir (sanity check)
        failed_log_root: where to write failed.json (default: first output_dir's parent)
        extra_filter: optional (mp4_path) -> bool predicate to filter pending list

    Returns:
        {"n_done": int, "n_failed": int, "failed": [(dir_name, mp4_name, err_str), ...]}
    """
    get_pending_videos, extract_one, ViTFeatureReader = _import_spamo_extractor()
    suffix = f"_{s2_mode}" if s2_mode else ""

    # Pre-flight: count pending across all dirs (no model load yet)
    plan: list[tuple[Path, Path, list[Path]]] = []
    for video_dir, out_dir in dirs:
        if not video_dir.exists():
            logger.warning(f"  skipping (missing): {video_dir}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        pending = get_pending_videos(video_dir, out_dir, suffix)
        if extra_filter is not None:
            pending = [p for p in pending if extra_filter(p)]
        if limit_per_dir is not None:
            pending = pending[:limit_per_dir]
        total = len(list(video_dir.glob("*.mp4")))
        plan.append((video_dir, out_dir, pending))
        logger.info(f"  {video_dir.name}: {len(pending)} pending / {total} total")

    total_pending = sum(len(p) for _, _, p in plan)
    if total_pending == 0:
        logger.info("Nothing to do.")
        return {"n_done": 0, "n_failed": 0, "failed": []}

    logger.info(
        f"Total pending: {total_pending}. "
        f"Initializing CLIP-ViT-L/14 (s2_mode={s2_mode}, scales={scales}) "
        f"on cuda:{gpu}, batch_size={batch_size}"
    )
    reader = ViTFeatureReader(
        model_name="openai/clip-vit-large-patch14",
        device=f"cuda:{gpu}",
        s2_mode=s2_mode,
        scales=scales,
    )

    failed: list[tuple[str, str, str]] = []
    t0 = time.time()
    n_done = 0
    for video_dir, out_dir, pending in plan:
        for mp4 in pending:
            out_file = out_dir / f"{mp4.stem}{suffix}.npy"
            try:
                extract_one(reader, mp4, out_file, batch_size)
                n_done += 1
            except Exception as e:
                failed.append((video_dir.name, mp4.name, str(e)))
                logger.warning(f"FAILED {video_dir.name}/{mp4.name}: {e}")
                continue
            if n_done % 50 == 0 or n_done == total_pending:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-9)
                eta = (total_pending - n_done) / max(rate, 1e-9)
                logger.info(
                    f"[{n_done}/{total_pending}] {rate:.1f} mp4/s "
                    f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)"
                )

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.0f}s. {len(failed)} failures")

    if failed:
        log_root = failed_log_root or (dirs[0][1].parent if dirs else Path.cwd())
        log_root.mkdir(parents=True, exist_ok=True)
        failed_log = log_root / "failed.json"
        existing = []
        if failed_log.exists():
            try:
                with open(failed_log) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"failed.json unreadable, starting fresh: {e}")
        existing.extend(
            [{"dir": d, "file": fn, "error": err} for d, fn, err in failed]
        )
        with open(failed_log, "w") as f:
            json.dump(existing, f, indent=2)
        logger.warning(f"Failure log: {failed_log}")

    return {"n_done": n_done, "n_failed": len(failed), "failed": failed}
