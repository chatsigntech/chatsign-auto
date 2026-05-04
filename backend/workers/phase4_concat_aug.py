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

from backend.core.dataset_videos import extract_tokens_from_entries
from backend.scripts.asl_resources import resolve_asl_resources
from backend.scripts.build_concat_aug import build_concat_aug
from backend.scripts.org_resources import resolve_org_resources

logger = logging.getLogger(__name__)


def run_concat_aug(
    task_id: str,
    base_anno: Path,
    base_feat: Path,
    aug_anno_out: Path,
    aug_feat_out: Path,
    *,
    base_sentences_npy: str = "train_info_ml.npy",
    preset: str = "36x",
    val_fraction: float = 0.1,
    aug_seed: int = 1337,
    split_seed: int = 42,
    org_fallback_threshold: float = 0.4,
) -> dict:
    """Step 4.2.5: resolve B/C resources + build concat-aug training data.

    base_sentences_npy: which info_ml.npy under base_anno is treated as the
        source of base sentences to augment. Default "train_info_ml.npy" so
        every training entry (current task + pad) gets the 36x augmentation;
        pass "test_info_ml.npy" to limit augmentation to current_entries.

    Returns: build_summary dict (n_train, n_val, n_dropped, preset, etc.)

    Side effects:
        - Writes aug_anno_out / {train,val,test}_info_ml.npy + aug_recipe.json
          + build_summary.json
        - Writes aug_feat_out / *_s2wrapping.npy (mix of symlinks and physical)
        - Creates aug_feat_out / {train,val,dev,test}/ self-symlinks
        - Auto-fallback to 21x if ORG hit-rate < threshold
    """
    sentences = list(np.load(base_anno / base_sentences_npy, allow_pickle=True))
    glosses = extract_tokens_from_entries(sentences)
    logger.info(
        f"[{task_id}] Step 4.2.5: extracted {len(glosses)} unique tokens from {base_sentences_npy}"
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
        sentences=sentences,
    )

    logger.info(
        f"[{task_id}] Step 4.2.5: concat-aug done — "
        f"train={summary['n_train']} val={summary['n_val']} dropped={summary['n_dropped']} "
        f"(preset={effective_preset})"
    )
    return summary
