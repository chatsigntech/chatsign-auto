"""Build concat-augmented training data for SpaMo (orchestrator port of test_real 06b).

Synthesizes new training "sentences" by concatenating word-level features
in token order — both real word videos (B class, `org_*`) and ASL-dictionary
videos (C class, `asl_*`).

Per base sentence, emit three categories of variants:

  A — `1 + N_A` × :  base sentence feature (no concat)
                     • v=base : original text       (no fileid suffix)
                     • v=0..N_A-1 : derangement-shuffled text
                     • feature is a SYMLINK to the base sentence feature

  B — `N_B` × :       concat from gloss_resources_org[token] features
                     (always picks first available org_* per token)
                     • v=0 : original token order
                     • v=1..N_B-1 : derangement-shuffled token order
                     • feature is a PHYSICAL .npy (concatenated time axis)

  C — `N_C` × :       concat from gloss_resources_asl[token] features
                     (random pick when a word has multiple asl_* options)
                     • v=0 : original token order
                     • v=1..N_C-1 : derangement-shuffled token order
                     • feature is a PHYSICAL .npy

Total per sentence: 1 + N_A + N_B + N_C  (default 36x: 1 + 10 + 10 + 15 = 36).

The full pool is then shuffled with --split-seed and split top
--val-fraction → val (= test, leaky on purpose), rest → train.

Output layout (matches test_real 06b 1:1):
  <anno-out>/
    train_info_ml.npy
    val_info_ml.npy
    test_info_ml.npy           # = val_info_ml.npy (leaky on purpose)
    aug_recipe.json            # per-fileid audit trail
    build_summary.json         # counts + drop tallies + frame stats

  <feat-out>/
    sentence_<id>_s2wrapping.npy            symlink → base sentence feat
    sentence_<id>_a<n>_s2wrapping.npy       symlink → base sentence feat
    sentence_<id>_bv<v>_s2wrapping.npy      PHYSICAL  (concat org_*)
    sentence_<id>_cv<v>_s2wrapping.npy      PHYSICAL  (concat random asl_*)
    train/ val/ dev/ test/                  symlinks → feat_out itself
                                            (REQUIRED: SpaMo dataloader looks
                                             at <feat_root>/<split>/*.npy)

Tokens with no matching word are dropped from a variant; if every token
drops the variant is skipped (counted in build_summary.json).

Differences from test_real 06b:
  - Input changed from --word-lib (filesystem dir) to two in-memory dicts
    (gloss_resources_org + gloss_resources_asl), produced by
    backend.scripts.{org_resources,asl_resources}.resolve_*_resources()
  - tok_to_word() / build_word_index() removed (not needed: dicts are
    already keyed by lowercase_underscore matching anno text tokens)
  - dead code base_feat_rel removed (test_real 06b:192-194 was unused)

Presets:
  36x  →  N_A=10  N_B=10  N_C=15   (default; matches test_real default)
  21x  →  N_A=10  N_B=0   N_C=10   (auto-fallback when ORG hit-rate < 40%)
  custom: pass n_a / n_b / n_c explicitly

See TEST_REAL_UPGRADE_PLAN.md §3.3 Step 2 for full design.
"""
from __future__ import annotations

import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


PRESETS = {
    "36x": {"N_A": 10, "N_B": 10, "N_C": 15},
    "21x": {"N_A": 10, "N_B": 0, "N_C": 10},
}


def derangement(rng: random.Random, tokens: list[str], max_tries: int = 200) -> list[str]:
    """Permute tokens so no element stays at its original index. Falls back
    to single-position rotation when impossible (length<=1 or all-same).

    1:1 port from test_real 06b:89-101.
    """
    if len(tokens) <= 1:
        return list(tokens)
    if len(set(tokens)) == 1:
        return list(tokens)
    for _ in range(max_tries):
        cand = list(tokens)
        rng.shuffle(cand)
        if all(cand[i] != tokens[i] for i in range(len(tokens))):
            return cand
    return tokens[1:] + tokens[:1]


def _concat_features(
    tokens: list[str],
    kind: str,
    gloss_resources: dict[str, list[tuple[Path, Path]]],
    rng: random.Random,
    drop_log: Counter,
    npy_cache: dict[Path, np.ndarray],
) -> tuple[Optional[np.ndarray], list[str]]:
    """Concat per-token features along time axis, dropping missing tokens.

    For 'asl' (C class), randomly pick one of the available .npy when multiple exist.
    For 'org' (B class), always pick the first (deterministic).

    Adapted from test_real 06b:116-144 — but takes pre-resolved dict instead
    of word_index from filesystem scan, and reads-through `npy_cache` so the
    same npy file isn't reloaded across the ~36 variants per sentence.

    Args:
        tokens: lowercase_underscore tokens (from anno entry["text"].split())
        kind: 'org' or 'asl'
        gloss_resources: {token: [(mp4_path, npy_path), ...]}
        rng: shared RNG (sequential consumption preserves test_real determinism)
        drop_log: Counter to record per-token drop reasons
        npy_cache: shared {npy_path: ndarray} cache; mutated as features load
    """
    feats: list[np.ndarray] = []
    used: list[str] = []
    for t in tokens:
        cands = gloss_resources.get(t)
        if not cands:
            drop_log[f"no_{kind}:{t}"] += 1
            continue
        # B class always picks first; C class picks random
        if kind == "asl":
            _mp4, npy_path = rng.choice(cands)
        else:
            _mp4, npy_path = cands[0]
        arr = npy_cache.get(npy_path)
        if arr is None:
            try:
                arr = np.load(npy_path).astype(np.float32)
            except Exception as e:
                drop_log[f"load_fail:{t}"] += 1
                logger.debug(f"  skip {t}: load failed for {npy_path}: {e}")
                continue
            if arr.shape[0] == 0:
                drop_log[f"empty:{t}"] += 1
                # cache empty too so we don't retry-load it
                npy_cache[npy_path] = arr
                continue
            npy_cache[npy_path] = arr
        elif arr.shape[0] == 0:
            drop_log[f"empty:{t}"] += 1
            continue
        feats.append(arr)
        used.append(npy_path.stem.replace("_s2wrapping", ""))
    if not feats:
        return None, []
    return np.concatenate(feats, axis=0), used


def build_concat_aug(
    base_anno: Path,
    base_feat: Path,
    gloss_resources_org: dict[str, list[tuple[Path, Path]]],
    gloss_resources_asl: dict[str, list[tuple[Path, Path]]],
    anno_out: Path,
    feat_out: Path,
    *,
    preset: str = "36x",
    n_a: Optional[int] = None,
    n_b: Optional[int] = None,
    n_c: Optional[int] = None,
    aug_seed: int = 1337,
    split_seed: int = 42,
    val_fraction: float = 0.10,
    sentences: Optional[list[dict]] = None,
) -> dict:
    """Build concat-augmented anno + feat dirs for SpaMo training.

    Args:
        base_anno: dir containing test_info_ml.npy (used as base sentences source)
        base_feat: dir containing <fileid>_s2wrapping.npy for every base sentence
        gloss_resources_org: {token: [(mp4, npy), ...]} for B class
        gloss_resources_asl: {token: [(mp4, npy), ...]} for C class
        anno_out: output anno dir
        feat_out: output feat dir
        preset: '36x' or '21x'
        n_a/n_b/n_c: override preset
        aug_seed: RNG seed for shuffles + asl random pick (test_real default 1337)
        split_seed: RNG seed for train/val split (test_real default 42)
        val_fraction: top fraction of shuffled pool → val == test

    Returns: build_summary dict (also written to anno_out/build_summary.json)
    """
    n_a = n_a if n_a is not None else PRESETS[preset]["N_A"]
    n_b = n_b if n_b is not None else PRESETS[preset]["N_B"]
    n_c = n_c if n_c is not None else PRESETS[preset]["N_C"]
    logger.info(f"counts: A={n_a}  B={n_b}  C={n_c}   (preset={preset})")

    # === Step 1: load base sentences from test_info_ml.npy (or use caller-supplied) ===
    if sentences is None:
        test_info_path = base_anno / "test_info_ml.npy"
        if not test_info_path.exists():
            raise RuntimeError(
                f"base_anno/test_info_ml.npy not found: {test_info_path}"
            )
        sentences = list(np.load(test_info_path, allow_pickle=True))
    logger.info(f"base sentences (test_info_ml): {len(sentences)}")

    # === Step 2: prepare output dirs ===
    anno_out.mkdir(parents=True, exist_ok=True)
    feat_out.mkdir(parents=True, exist_ok=True)

    # base feature symlinks (one per base sentence)
    for entry in sentences:
        base_fid = entry["fileid"]
        dst = feat_out / f"{base_fid}_s2wrapping.npy"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        src = (base_feat / f"{base_fid}_s2wrapping.npy").resolve()
        dst.symlink_to(src)

    # === Step 3: generate A/B/C augmentations ===
    aug_rng = random.Random(aug_seed)
    aug_all: list[dict] = []
    recipe: dict = {
        "method": {
            "aug_seed": aug_seed,
            "split_seed": split_seed,
            "val_fraction": val_fraction,
            "A": f"1 base (no suffix) + {n_a} shuffles, real sentence feature symlink",
            "B": f"{n_b} variants, concat org_* features (gloss_resources_org), v=0 orig text",
            "C": f"{n_c} variants, concat random asl_* features (gloss_resources_asl), v=0 orig text",
            "drop_policy": "drop tokens missing required kind (skip variant only if ALL drop)",
            "input_source": "in-memory dict (NOT filesystem word_lib scan)",
        },
        "samples": {},
    }
    drop_log_b: Counter = Counter()
    drop_log_c: Counter = Counter()
    skipped: Counter = Counter()
    frame_stats: dict[str, list[int]] = {"A": [], "B": [], "C": []}
    # Shared npy load cache (per-build): same npy is touched ~10× across B/C
    # variants per token; without cache that's thousands of redundant np.load.
    npy_cache: dict[Path, np.ndarray] = {}

    for entry in sentences:
        base_fid = entry["fileid"]
        text = entry["text"]
        tokens_lc = text.split()  # already lowercase_underscore from phase4 worker:321

        # Inline A-frame stat (avoids a second pass over base_feat)
        try:
            base_T = int(np.load(
                base_feat / f"{base_fid}_s2wrapping.npy", mmap_mode="r"
            ).shape[0])
        except Exception:
            base_T = 0
        for _ in range(1 + n_a):
            frame_stats["A"].append(base_T)

        # ===== A — base entry + N_A shuffles (real-feature symlinks) =====
        aug_all.append(dict(entry))  # bare base
        for n in range(n_a):
            order = derangement(aug_rng, tokens_lc)
            new_fid = f"{base_fid}_a{n}"
            ne = dict(entry)
            ne["fileid"] = new_fid
            ne["folder"] = new_fid
            ne["text"] = " ".join(order)
            aug_all.append(ne)

            link = feat_out / f"{new_fid}_s2wrapping.npy"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to((base_feat / f"{base_fid}_s2wrapping.npy").resolve())
            recipe["samples"][new_fid] = {
                "cat": "A",
                "variant": n,
                "order": order,
                "feat_symlink_to": f"{base_fid}_s2wrapping.npy",
            }

        # ===== B — N_B variants, concat org_* =====
        for v in range(n_b):
            order = list(tokens_lc) if v == 0 else derangement(aug_rng, tokens_lc)
            feat, used = _concat_features(
                order, "org", gloss_resources_org, aug_rng, drop_log_b, npy_cache
            )
            if feat is None:
                skipped[f"B_v{v}"] += 1
                continue
            new_fid = f"{base_fid}_bv{v}"
            np.save(feat_out / f"{new_fid}_s2wrapping.npy", feat)
            ne = dict(entry)
            ne["fileid"] = new_fid
            ne["folder"] = new_fid
            ne["text"] = " ".join(order)
            aug_all.append(ne)
            frame_stats["B"].append(int(feat.shape[0]))
            recipe["samples"][new_fid] = {
                "cat": "B",
                "variant": v,
                "order": order,
                "used_uids": used,
                "frame_count": int(feat.shape[0]),
            }

        # ===== C — N_C variants, concat random asl_* =====
        for v in range(n_c):
            order = list(tokens_lc) if v == 0 else derangement(aug_rng, tokens_lc)
            feat, used = _concat_features(
                order, "asl", gloss_resources_asl, aug_rng, drop_log_c, npy_cache
            )
            if feat is None:
                skipped[f"C_v{v}"] += 1
                continue
            new_fid = f"{base_fid}_cv{v}"
            np.save(feat_out / f"{new_fid}_s2wrapping.npy", feat)
            ne = dict(entry)
            ne["fileid"] = new_fid
            ne["folder"] = new_fid
            ne["text"] = " ".join(order)
            aug_all.append(ne)
            frame_stats["C"].append(int(feat.shape[0]))
            recipe["samples"][new_fid] = {
                "cat": "C",
                "variant": v,
                "order": order,
                "used_uids": used,
                "frame_count": int(feat.shape[0]),
            }

    expected = len(sentences) * (1 + n_a + n_b + n_c)
    logger.info(f"total aug entries: {len(aug_all)}    (expected {expected})")
    if skipped:
        logger.info(f"skipped variants (all-tokens-missing): {dict(skipped)}")

    # === Step 5: split train/val (val == test, leaky on purpose) ===
    logger.info(
        f"splitting train/val (seed={split_seed}, "
        f"top {val_fraction*100:.0f}% → val=test) ..."
    )
    split_rng = random.Random(split_seed)
    indices = list(range(len(aug_all)))
    split_rng.shuffle(indices)
    n_val = int(round(len(aug_all) * val_fraction))
    val_idx = sorted(indices[:n_val])
    train_idx = sorted(indices[n_val:])
    train_arr = [aug_all[i] for i in train_idx]
    val_arr = [aug_all[i] for i in val_idx]
    test_arr = list(val_arr)  # leaky on purpose
    logger.info(f"  train: {len(train_arr)}    val: {len(val_arr)}  (= test)")

    # === Step 6: save ===
    np.save(anno_out / "train_info_ml.npy", np.array(train_arr, dtype=object))
    np.save(anno_out / "val_info_ml.npy", np.array(val_arr, dtype=object))
    np.save(anno_out / "test_info_ml.npy", np.array(test_arr, dtype=object))
    with open(anno_out / "aug_recipe.json", "w") as f:
        json.dump(recipe, f, indent=2)

    # Look up per-fileid cat from recipe (set explicitly when each variant emitted).
    # Don't infer from filename — would re-couple semantics to naming, which the
    # repo's no-filename-semantics rule explicitly forbids.
    samples = recipe["samples"]

    def cat_of(fid: str) -> str:
        meta = samples.get(fid)
        if meta is None:
            return "A_base"  # base entries aren't in samples
        return f"A_shuf" if meta["cat"] == "A" else meta["cat"]

    summary = {
        "preset": preset,
        "counts": {"N_A": n_a, "N_B": n_b, "N_C": n_c},
        "aug_seed": aug_seed,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "n_base_sentences": len(sentences),
        "total_built": len(aug_all),
        "n_train": len(train_arr),
        "n_val": len(val_arr),
        "n_test": len(test_arr),
        "n_dropped": int(sum(skipped.values())),
        "split": {
            "train": len(train_arr),
            "val": len(val_arr),
            "test": len(test_arr),
        },
        "cat_breakdown_per_split": {
            split_name: dict(Counter(cat_of(r["fileid"]) for r in arr))
            for split_name, arr in [("train", train_arr), ("val", val_arr)]
        },
        "skipped_variants_total": dict(skipped),
        "B_drops_top20": dict(drop_log_b.most_common(20)),
        "C_drops_top20": dict(drop_log_c.most_common(20)),
        "frame_stats": {
            cat: (
                {
                    "n": len(L),
                    "min": min(L),
                    "max": max(L),
                    "mean": float(np.mean(L)),
                    "median": int(np.median(L)),
                }
                if L
                else {}
            )
            for cat, L in frame_stats.items()
        },
    }
    with open(anno_out / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # === Step 7: feat_out split symlinks (REQUIRED: SpaMo dataloader needs them) ===
    # 1:1 port of test_real 06b:351-356.
    feat_out_abs = feat_out.resolve()
    for split_name in ("train", "val", "dev", "test"):
        link = feat_out_abs / split_name
        if link.is_symlink() or link.exists():
            continue
        try:
            link.symlink_to(feat_out_abs)
        except OSError as e:
            logger.warning(f"failed to create split symlink {link}: {e}")

    logger.info(f"build_summary saved to {anno_out / 'build_summary.json'}")
    return summary


# ---------------- CLI ----------------


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-anno", type=Path, required=True)
    ap.add_argument("--base-feat", type=Path, required=True)
    ap.add_argument(
        "--anno-out",
        type=Path,
        required=True,
        help="Output anno dir (will overwrite train/val/test_info_ml.npy)",
    )
    ap.add_argument(
        "--feat-out",
        type=Path,
        required=True,
        help="Output feat dir (will populate symlinks + physical npy)",
    )
    ap.add_argument(
        "--from-anno-tokens",
        action="store_true",
        help="Auto-derive gloss list from base_anno/test_info_ml.npy and call "
        "asl_resources/org_resources to resolve. Default: requires explicit dicts via Python API.",
    )
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="36x")
    ap.add_argument("--n-a", type=int, default=None)
    ap.add_argument("--n-b", type=int, default=None)
    ap.add_argument("--n-c", type=int, default=None)
    ap.add_argument("--aug-seed", type=int, default=1337)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument(
        "--max-per-gloss",
        type=int,
        default=5,
        help="Cap on candidates per gloss in resolve_*_resources (default 5)",
    )
    args = ap.parse_args()

    if not args.from_anno_tokens:
        print(
            "ERROR: CLI mode requires --from-anno-tokens (auto-resolve via "
            "asl_resources + org_resources from base_anno/test_info_ml).\n"
            "       For programmatic use, import build_concat_aug() and pass dicts directly.",
            file=sys.stderr,
        )
        sys.exit(2)

    from backend.scripts.asl_resources import resolve_asl_resources
    from backend.scripts.org_resources import resolve_org_resources

    # Extract tokens from base_anno (1:1 with test_real 06b — only test_info_ml)
    test_info_path = args.base_anno / "test_info_ml.npy"
    if not test_info_path.exists():
        print(f"ERROR: {test_info_path} not found", file=sys.stderr)
        sys.exit(2)
    tokens_set = set()
    for entry in np.load(test_info_path, allow_pickle=True):
        if isinstance(entry, dict):
            tokens_set.update(entry.get("text", "").split())
    tokens = sorted(tokens_set)
    logger.info(f"extracted {len(tokens)} unique tokens from test_info_ml.npy")

    asl = resolve_asl_resources(tokens, max_per_gloss=args.max_per_gloss)
    org = resolve_org_resources(tokens, max_per_gloss=args.max_per_gloss)

    summary = build_concat_aug(
        base_anno=args.base_anno,
        base_feat=args.base_feat,
        gloss_resources_org=org["resources"],
        gloss_resources_asl=asl["resources"],
        anno_out=args.anno_out,
        feat_out=args.feat_out,
        preset=args.preset,
        n_a=args.n_a,
        n_b=args.n_b,
        n_c=args.n_c,
        aug_seed=args.aug_seed,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
    )
    print(json.dumps(summary, indent=2))
