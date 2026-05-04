# Phase 8 worker upgrade — patch notes (REVISED after --help verification)

This describes draft changes to `backend/workers/phase8_training.py` for split-level dual-centroid per `TEST_REAL_UPGRADE_PLAN.md` §4.2.

**Status**: Drafted, NOT applied. **Verified args via --help on 2026-05-03 — found 2 important deviations from earlier doc estimate.**

## ⚠️ Important: actual gloss_aware HEAD differs from doc estimate

### Deviation 1: split_level lacks `--pretrained` (warm-start broken)

Old `ssl_pretraining_crossvideo_mlp_*` had `--pretrained` for warm-start:
```python
parser.add_argument('--pretrained', default=None, type=str,
                    help='Previous checkpoint for warm start')
```

`ssl_pretraining_glosspose_split_level.py` **does NOT have `--pretrained`** (verified by grep + --help). The worker's warm-start logic at line 663-665 must skip --pretrained when using split_level:

```python
# Before (line 663):
if prev_checkpoint:
    train_cmd += ["--pretrained", str(prev_checkpoint.resolve())]
    logger.info(f"[{task_id}] Warm start from: {prev_checkpoint}")

# After: warm-start unavailable in split_level
if prev_checkpoint:
    logger.warning(
        f"[{task_id}] Warm-start requested ({prev_checkpoint}) but split_level "
        f"script doesn't support --pretrained. Skipping warm-start; training "
        f"from scratch. Either add --pretrained support to "
        f"ssl_pretraining_glosspose_split_level.py upstream, or accept cold start."
    )
```

### Deviation 2: build_prototypes_both arg names differ from doc estimate

Old `build_prototypes_asl_clip_nob2b.py` accepted `--ckpt` `--dataset` `--output-dir` `--l2norm`.

`build_prototypes_both.py` has **dual prefix**:
- `--dual-ckpt` (instead of `--ckpt`)
- `--dual-dataset` (instead of `--dataset`)
- `--dual-output-dir` (instead of `--output-dir`)
- `--single-*` siblings (skipped via `--skip-single`)
- `--l2norm` ✓ same

So the proto_cmd needs full rename:

```python
# Before (line 703-709):
proto_cmd = [
    sys.executable, str(proto_script),
    "--dataset", dataset_name,
    "--ckpt", str(best_ckpt.resolve()),
    "--output-dir", str(proto_dir.resolve()),
    "--l2norm",
]

# After:
proto_cmd = [
    sys.executable, str(proto_script),
    "--dual-dataset", dataset_name,
    "--dual-ckpt", str(best_ckpt.resolve()),
    "--dual-output-dir", str(proto_dir.resolve()),
    "--l2norm",
    "--skip-single",
]
```

## Patch summary — 5 actual changes

### Line 650 — train script path

```python
# Before:
train_script = ga_path / "ssl_pretraining_crossvideo_mlp_feature_mean_mean_advance_v4_noconf_clip_nob2b.py"
# After:
train_script = ga_path / "ssl_pretraining_glosspose_split_level.py"
```

### Line 663-665 — handle missing --pretrained

```python
# Before:
if prev_checkpoint:
    train_cmd += ["--pretrained", str(prev_checkpoint.resolve())]
    logger.info(f"[{task_id}] Warm start from: {prev_checkpoint}")

# After:
if prev_checkpoint:
    logger.warning(
        f"[{task_id}] Warm-start requested but ssl_pretraining_glosspose_split_level.py "
        f"doesn't support --pretrained; training from scratch. (prev_checkpoint={prev_checkpoint})"
    )
```

### Line 702 — proto script path

```python
# Before:
proto_script = ga_path / "build_prototypes_asl_clip_nob2b.py"
# After:
proto_script = ga_path / "build_prototypes_both.py"
```

### Line 703-709 — proto_cmd full rename

```python
proto_cmd = [
    sys.executable, str(proto_script),
    "--dual-dataset", dataset_name,
    "--dual-ckpt", str(best_ckpt.resolve()),
    "--dual-output-dir", str(proto_dir.resolve()),
    "--l2norm",
    "--skip-single",
]
```

## Verified compatibility (no changes needed)

These existing args still work in split_level:
- `--dataset` ✓ (same name)
- `--output_dir` ✓ (same name)
- `--epochs` ✓ (same name, default 100 vs 150 — keep `--epochs 150` from worker)
- `torchrun` invocation ✓ (no change)
- `slrt1` env (Python 3.8.20) ✓ (split_level doesn't use `BooleanOptionalAction` confirmed earlier)

## ⚠️ NOT YET VERIFIED (require GPU run)

- Whether the new proto outputs use the same .pkl filename pattern that downstream P5 inference expects
- Whether dual-centroid prototypes are loadable by sign_stream / sign_video_generator inference paths
- Whether 21x/36x augmented training data (from #1 P4 colent) actually produces a checkpoint that split_level pretraining can read (data format chain)

## Rollback

`git revert` this commit. Old scripts remain in gloss_aware HEAD, so reverted worker still runs single-center.

## Action items for verification before applying

- [ ] Verify P5 inference / sign_stream / sign_video_generator can load `prototypes/word_*.pkl` + `prototypes/sentence_*.pkl` from build_prototypes_both output
- [ ] Decide whether warm-start is critical (if yes, split_level upgrade blocked until `--pretrained` is added upstream); if not, accept cold-start
- [ ] Validate train_cmd args list against latest split_level --help (already done 2026-05-03 — but recheck before commit if gloss_aware bumped)
