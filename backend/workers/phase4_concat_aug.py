"""Phase 4 helper: concat-aug data synthesis (test_real 06b port).

Called from phase4_segmentation_train.run_phase4_segmentation_train after
annotations exist and before SpaMo config generation.

Resolves B-class (chatsign-accuracy approved word videos) and C-class
(ASL-27K dictionary) resources from in-memory dicts, then calls
backend.scripts.build_concat_aug to synthesize the 36x training data.
Auto-fallback to 21x preset if ORG (B-class) hit-rate < 40%.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from backend.scripts.asl_resources import resolve_asl_resources
from backend.scripts.build_concat_aug import build_concat_aug
from backend.scripts.org_resources import resolve_org_resources

logger = logging.getLogger(__name__)


def _extract_tokens_from_anno(base_anno: Path) -> list[str]:
    """Extract unique tokens (lowercase_underscore) from test_info_ml.npy.

    Matches test_real 06b — uses test_info_ml as the "all sentences to augment"
    source (orchestrator's test_info_ml = current_entries, same semantic).
    Tokens are already lowercased when phase4_segmentation_train joins them
    into anno text.
    """
    path = base_anno / "test_info_ml.npy"
    if not path.exists():
        raise RuntimeError(f"test_info_ml.npy not found at {path}")
    tokens: set[str] = set()
    for entry in np.load(path, allow_pickle=True):
        if isinstance(entry, dict):
            tokens.update(entry.get("text", "").split())
    return sorted(tokens)


def run_concat_aug(
    task_id: str,
    base_anno: Path,
    base_feat: Path,
    aug_anno_out: Path,
    aug_feat_out: Path,
    *,
    preset: str = "36x",
    val_fraction: float = 0.1,
    aug_seed: int = 1337,
    split_seed: int = 42,
    org_fallback_threshold: float = 0.4,
) -> tuple[Path, Path, dict]:
    """Step 4.2.5: resolve B/C resources + build concat-aug training data.

    Returns:
        (aug_anno_out, aug_feat_out, build_summary_dict)

    Side effects:
        - Writes aug_anno_out / {train,val,test}_info_ml.npy + aug_recipe.json
          + build_summary.json
        - Writes aug_feat_out / *_s2wrapping.npy (mix of symlinks and physical)
        - Creates aug_feat_out / {train,val,dev,test}/ self-symlinks
        - Auto-fallback to 21x if ORG hit-rate < threshold
    """
    glosses = _extract_tokens_from_anno(base_anno)
    logger.info(
        f"[{task_id}] Step 4.2.5: extracted {len(glosses)} unique tokens from test_info_ml"
    )

    asl = resolve_asl_resources(glosses, max_per_gloss=5)
    org = resolve_org_resources(glosses, max_per_gloss=5)

    for which, res in [("ASL", asl), ("ORG", org)]:
        if res["feat_missing_files"]:
            logger.warning(
                f"[{task_id}] {which}: {len(res['feat_missing_files'])} videos lack "
                f"precomputed features; run precompute_*_features.py to fill"
            )

    # Auto-fallback when ORG hit-rate too low (B-class would mostly drop)
    org_hit_rate = org["n_glosses_hit"] / max(len(glosses), 1)
    asl_hit_rate = asl["n_glosses_hit"] / max(len(glosses), 1)
    effective_preset = preset
    if preset == "36x" and org_hit_rate < org_fallback_threshold:
        logger.warning(
            f"[{task_id}] ORG hit-rate {org_hit_rate:.1%} < "
            f"{org_fallback_threshold:.0%} threshold; falling back to 21x preset "
            f"(B-class skipped, C-class only)"
        )
        effective_preset = "21x"

    logger.info(
        f"[{task_id}] Step 4.2.5: ASL hit {asl['n_glosses_hit']}/{len(glosses)} "
        f"({asl_hit_rate:.1%}), ORG hit {org['n_glosses_hit']}/{len(glosses)} "
        f"({org_hit_rate:.1%}); preset={effective_preset}"
    )

    summary = build_concat_aug(
        base_anno=base_anno,
        base_feat=base_feat,
        gloss_resources_org=org["resources"],
        gloss_resources_asl=asl["resources"],
        anno_out=aug_anno_out,
        feat_out=aug_feat_out,
        preset=effective_preset,
        val_fraction=val_fraction,
        aug_seed=aug_seed,
        split_seed=split_seed,
    )

    logger.info(
        f"[{task_id}] Step 4.2.5: concat-aug done — "
        f"train={summary['n_train']} val={summary['n_val']} dropped={summary['n_dropped']} "
        f"(preset={effective_preset})"
    )
    return aug_anno_out, aug_feat_out, summary
