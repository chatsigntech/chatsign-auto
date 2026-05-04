# Phase 4 worker upgrade — patch notes

This describes the **draft** changes to `backend/workers/phase4_segmentation_train.py` to enable colent + 36x concat-aug per `TEST_REAL_UPGRADE_PLAN.md` §3.3 Step 3.

**Status**: Drafted but NOT applied. Apply via the patch below after:
1. spamo_segement submodule fork has colent files (§1.2 B)
2. ASL-27K + accuracy precompute done
3. phase4_concat_aug.py is in place (already done)

## Required orchestrator changes

### 1. Template path change (§3.3 Step 3 (a))

```python
# Before (line ~416):
template_path = SPAMO_ROOT / "configs" / "how2sign_contrastive_single.yaml"

# After:
template_path = SPAMO_ROOT / "configs" / "chatsign_concat_aug_colent.yaml"
```

Or — to support reverting cleanly via env var when needed (NOT introducing a new env var per user direction; use git revert instead):

Just change the constant. Rollback path = git revert this commit.

### 2. New step 4.2.5: insert _run_concat_aug after _build_annotations

In `run_phase4_segmentation_train`, after `_build_annotations()` returns, insert:

```python
from backend.workers.phase4_concat_aug import run_concat_aug

# Step 4.2.5: concat-aug (colent uses 36x augmented data)
aug_anno_dir = output_dir / "annotations_aug"
aug_feat_dir = output_dir / "features_aug"
aug_anno_dir, aug_feat_dir, aug_summary = await asyncio.to_thread(
    run_concat_aug,
    task_id=task_id,
    base_anno=anno_dir,
    base_feat=feat_dir,
    aug_anno_out=aug_anno_dir,
    aug_feat_out=aug_feat_dir,
    preset="36x",
)
```

Note: `run_concat_aug` is sync (no asyncio inside), wrap in `asyncio.to_thread` to keep the async worker non-blocking.

### 3. _generate_config call: pass aug paths instead of base

```python
# Before:
config_path = _generate_config(
    task_id, anno_dir, feat_dir, sentence_video_dir, output_dir
)

# After:
config_path = _generate_config(
    task_id, aug_anno_dir, aug_feat_dir, sentence_video_dir, output_dir
)
```

This makes SpaMo training read the augmented data (anno_root / feat_root point to aug dirs).

### 4. Schema check on chatsign_concat_aug_colent.yaml

`_generate_config` does:
```python
for split in ("train", "validation", "test"):
    p = config.data.params[split].params
    p.anno_root = str(anno_dir)
    p.feat_root = str(feat_dir)
    p.vid_root = str(video_dir)
    p.mae_feat_root = str(feat_dir)
```

Verify that the new template `chatsign_concat_aug_colent.yaml` has the same `data.params.{train,validation,test}.params.{anno_root, feat_root, vid_root, mae_feat_root}` schema. If different, the config-patching loop must adapt.

**Action item before applying**: cat the new yaml after submodule fork and confirm schema. If schema differs (e.g., new fields added), update _generate_config accordingly.

## Side-effect analysis

| Field | Before | After |
|---|---|---|
| Training BLEU target | base 85.58 | colent 36x (target ~94, actual待实测) |
| Training data volume | N_train sentences | N_train × 36 augmented entries |
| anno_root for SpaMo | output_dir/annotations | output_dir/annotations_aug |
| feat_root for SpaMo | output_dir/features | output_dir/features_aug |
| Disk usage per task | ~base feature size | ~(1 + N_C + N_B) × per-sentence feat size, much larger; B/C are physical npy concat |
| Test set semantic | output_dir/annotations/test_info_ml.npy = current_entries (P5 reads this) | aug_anno_dir/test_info_ml.npy = leaky (=val); but **P5 still reads original anno_dir/test_info_ml.npy**, NOT aug |

The original `anno_dir/test_info_ml.npy` is preserved (we only add aug_*, don't overwrite). P5 segment is unaffected — it walks anno_dir, not aug_anno_dir. Do not change P5 worker.

## Rollback

`git revert` the commit applying these changes. Submodule pin (containing colent files) can stay — the old `how2sign_contrastive_single.yaml` is still in spamo_segement after fork, so reverted worker still runs base.

## Open questions for review

- [ ] Confirm `chatsign_concat_aug_colent.yaml` schema matches what `_generate_config` expects (run after submodule fork)
- [ ] Disk space estimate per task with 36x aug (depends on sentence count × frames × 2048 dim)
- [ ] Whether to clean up `aug_anno_dir/aug_feat_dir` after training to save disk (maybe keep for reproducibility)
