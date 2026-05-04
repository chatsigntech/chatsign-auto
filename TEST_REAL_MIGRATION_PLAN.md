# test_real 迁移 — P4 + P8 训练 worker

**日期**：2026-05-04
**作者**：yfang + Claude
**取代**：`backend/workers/PHASE4_PATCH_NOTES.md`、`backend/workers/PHASE8_PATCH_NOTES.md`（两份草稿都以 `gloss_aware/` + `spamo_segement/` 为目标，方向偏；本 plan 直接以 `test_real/` 为目标重写）

---

## 1. 范围

| 改动类型 | 范围 |
|---|---|
| **核心逻辑** | 仅 P4（分割训练）+ P8（识别训练）两个 worker |
| **I/O 格式适配** | 仅在和 P4/P8 接壤处做最小 shim — P3→P4 输入、P4→P5 ckpt、P7→P8 manifest、P8→inference |
| **不动** | P1/P2/P3/P5/P6/P7 核心逻辑、UI、API、`backend/recognition/api.py:_find_phase8_outputs` 接口契约 |

---

## 2. 关键事实清单（已验证）

- `spamo_segement/main.py` 与 `test_real/phase4_seg_train/main.py` **二进制相同**（`diff -q` 空）；`extract_clip_from_mp4.py` 同理 → P4 几乎纯粹是路径切换
- `test_real/phase4_seg_train/configs/` 包含 `chatsign_concat_aug_colent.yaml`（split-level 训练用）+ `chatsign_concat_aug.yaml` + `chatsign_concat_aug_e1e2.yaml`
- `test_real/phase8_training/` 8 个脚本路径与 `gloss_aware/` 平行（preprocess + tools + ssl_pretraining_* + build_prototypes_*），布局完全可平移
- test_real 的 `ssl_pretraining_glosspose_split_level.py`：
  - **没有** `--pretrained` —— upstream 故意不支持 warm-start（README §3.4 指引用 `add_prototypes.py` 走 prototype 层增量）
  - 多了 `--adaptive-schedule`（默认 ON）：n>50000 → 200/15，否则 150/10。覆盖 `--epochs`
  - `--batch-size choices=[128]` 锁死
- test_real 的 `build_prototypes_both.py` 参数名带 `--dual-` 前缀：`--dual-ckpt`、`--dual-dataset`、`--dual-output-dir`，加 `--skip-single`
- DB 19/0 个 task 用过 `prev_task_id`（incremental training 实际未投产）
- 接壤点 `_find_phase8_outputs` 只查 SHARED_DATA_ROOT 下的 `checkpoints/`、`prototypes/prototypes.pt`、`vocab.json` —— 路径在 worker 控制下，**不依赖 test_real 内部路径**

---

## 3. 落地前必须验证（PRE-CONDITIONS）— 已完成

### 3.1 P8 prototype 输出 → inference 兼容性

**✅ PASS — 不需要 shim**

`build_prototypes_both.py` 是 dispatcher，`--skip-single` 时调用 `build_prototypes_glosspose_splitlvl.py`，line 627 输出 `<output_dir>/prototypes.pt`。`_find_phase8_outputs` 找的就是 `prototypes/prototypes.pt`，路径一致。

### 3.2 P4→P5 ckpt 命名 + P5 worker 路径

**⚠️ 需要小补丁（plan §6 新增）**

P5 worker `phase5_segment.py:22` 同样硬编码了 `SPAMO_SEGMENT_PATH`，读 P4 ckpt 的同时还要调 `SPAMO_ROOT/scripts/segment_alignment.py`。
- ckpt 名 `segmentation_model.ckpt`：test_real main.py 二进制与 spamo_segement 相同 → 一致 ✓
- `test_real/phase4_seg_train/scripts/segment_alignment.py`：已确认存在 ✓
- **P5 worker 必须同步改 `SPAMO_ROOT` 指向**（plan §6）

### 3.3 P7→P8 manifest schema

**✅ PASS — 不需要 shim**

实际 P7→P8 边界是 **mp4 视频**（`phase8_training.py:440` `_link_videos(phase7_output / "aug_segment_videos", ...)`），不是 CSV。P8 step 8.4 自己 `_generate_csv()` 从 pose PKL 生成 `ref,gloss` 列，schema 匹配 `make_asl_labels.py`。

---

## 4. P4 patch（小）

### 4.1 `backend/config.py`

```python
# 4.1 加新设置（保留 SPAMO_SEGMENT_PATH 作向后兼容，但 P4 worker 不再读它）
TEST_REAL_PATH: Path = Path(os.getenv("TEST_REAL_PATH", str(BASE_DIR / "test_real")))
```

### 4.2 `backend/workers/phase4_segmentation_train.py`

```python
# Before line 38:
SPAMO_ROOT = Path(settings.SPAMO_SEGMENT_PATH).resolve()
# After:
SPAMO_ROOT = (settings.TEST_REAL_PATH / "phase4_seg_train").resolve()
```

其余 `SPAMO_ROOT / "scripts" / "extract_features" / "extract_clip_from_mp4.py"`、`SPAMO_ROOT / "main.py"` **路径不变**（test_real 同名）。

### 4.3 接入 concat-aug helper（昨夜成果）

参考 `backend/workers/phase4_concat_aug.py`，在 P4 worker 的 step 4.2.5 调用 `run_concat_aug()`。具体插入点等 step 4 实施时再定（通常在 anno 生成之后、config 生成之前）。

### 4.4 配置模板切到 colent

P4 worker step 4.3 现在用什么 yaml 模板？查清楚后改成 `chatsign_concat_aug_colent.yaml`。

---

## 5. P8 patch

### 5.1 `phase8_training.py:377` — root 路径

```python
# Before:
ga_path = settings.GLOSS_AWARE_PATH.resolve()
# After:
ga_path = (settings.TEST_REAL_PATH / "phase8_training").resolve()
```

8 处下游引用（`preprocess/`、`tools/`、`config.py`、`data/`、`ssl_pretraining_*`、`build_prototypes_*`）路径名一致，无需改。

### 5.2 `:650` — 训练脚本

```python
# Before:
train_script = ga_path / "ssl_pretraining_crossvideo_mlp_feature_mean_mean_advance_v4_noconf_clip_nob2b.py"
# After:
train_script = ga_path / "ssl_pretraining_glosspose_split_level.py"
```

### 5.3 `:663-665` — warm-start 跳过（对齐 upstream 设计）

```python
# Before:
if prev_checkpoint:
    train_cmd += ["--pretrained", str(prev_checkpoint.resolve())]
    logger.info(f"[{task_id}] Warm start from: {prev_checkpoint}")
# After:
if prev_checkpoint:
    logger.warning(
        f"[{task_id}] Warm-start skipped: ssl_pretraining_glosspose_split_level.py "
        f"(test_real upstream) doesn't support --pretrained by design. "
        f"prev_task_id incremental data still merged into vocab+JSONL; backbone "
        f"trained from scratch. See test_real/phase8_training/README.md §3.4 — "
        f"upstream's incremental path is via add_prototypes.py at the prototype "
        f"layer. (prev_checkpoint={prev_checkpoint})"
    )
```

### 5.4 `:702` — proto 脚本

```python
# Before:
proto_script = ga_path / "build_prototypes_asl_clip_nob2b.py"
# After:
proto_script = ga_path / "build_prototypes_both.py"
```

### 5.5 `:703-709` — proto args 全套改名

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

### 5.6 `:661` — `--epochs 150` 兼容性

`test_real` split_level 默认 `--adaptive-schedule` ON，会覆盖 `--epochs`。
- n_samples ≤ 50000 → epochs 150（一致，无效改动）
- n_samples > 50000 → epochs **强制 200**（被覆盖）

**决策**：保留传 `--epochs 150`（行为不变 ≤50000，>50000 时被覆盖也合理）。或者改成 `--no-adaptive-schedule + --epochs 150` 锁死。

**建议**：信任 upstream 自适应，保持现状不加 `--no-adaptive-schedule`。在 worker 注释说明。

### 5.7 boundary §3.1 — prototype 文件名 shim（如需要）

视 §3.1 验证结果：若 `build_prototypes_both.py --skip-single` 产出不是 `prototypes.pt`，在 step 8.7 末尾加最小 shim：

```python
# 仅在 §3.1 验证不通过时启用：
expected = proto_dir / "prototypes.pt"
if not expected.exists():
    actual = next(proto_dir.glob("*.pt"), None)  # 或对应实际命名
    if actual:
        expected.symlink_to(actual.name)
```

---

## 6. P5 patch（边界 I/O 适配，非核心逻辑）

按 yfang 「其它 phase 顶多做 I/O 格式适配」原则，P5 worker 必须同步改：

### 6.0 `backend/workers/phase5_segment.py:22`

```python
# Before:
SPAMO_ROOT = Path(settings.SPAMO_SEGMENT_PATH).resolve()
# After:
SPAMO_ROOT = (settings.TEST_REAL_PATH / "phase4_seg_train").resolve()
```

`scripts/segment_alignment.py`、ckpt 名 `segmentation_model.ckpt`、`config_*.yaml` 命名约定均与 spamo_segement 一致（已验证），无其它改动。

---

## 7. 测试计划

### 7.1 静态验证（不跑 GPU）

- [ ] 应用 patch 后跑 `python -c "from backend.workers import phase4_segmentation_train, phase8_training"` 确认 import 不炸
- [ ] `python -c "from backend.config import settings; print(settings.TEST_REAL_PATH)"` 输出正确
- [ ] `which torchrun` 在 slrt1 env 下可用

### 7.2 P4 小 task E2E（用户在场）

- 用一个已有的 phase2 输出做 source，跑 P4
- 期望：feature extraction → train → 产出 best ckpt + config → 不报错
- 失败可能点：`chatsign_concat_aug_colent.yaml` schema 是否被 worker 的 `_generate_config()` 兼容

### 7.3 P8 小 task E2E（用户在场）

- 用 P4 产出 + 一个 phase7 manifest 做 source，跑 P8
- 期望：8.1 pose extract → 8.2 filter → 8.3 norm → 8.4 labels → 8.5 register → 8.6 train → 8.7 prototypes → produces `checkpoints/best_cl.pth` + `prototypes/prototypes.pt` + `vocab.json`
- 失败可能点：
  - § 3.1 prototype filename mismatch
  - § 3.3 P7 manifest schema mismatch
  - adaptive-schedule 跑过 200 epoch 时间超 24h timeout

### 7.4 Inference 兼容性

- [ ] `_find_phase8_outputs(<test_task_id>)` 能找到模型
- [ ] WebSocket 推理通路能加载新 ckpt + prototypes 不报错（P5 推理路径同样核对）

---

## 8. 回滚

每一步都是 `git revert` 友好。最坏情况：
- `config.py` 那一行恢复 `SPAMO_SEGMENT_PATH`、`GLOSS_AWARE_PATH`
- worker 路径常量恢复
- old patch notes 仍在 git history 可参考

submodule 内容不动（`gloss_aware/`、`spamo_segement/` 仍在树上，只是 worker 不再引用）。

---

## 9. 落地步骤（按顺序）

1. ✅ 写本 plan
2. ✅ §3 PRE-CONDITION 验证（§3.1 PASS、§3.2 需 P5 小补丁、§3.3 PASS）
3. ✅ commit 昨夜成果（77f0bf3）
4. ✅ 应用 §4 P4 patch（**路径部分**；concat-aug 接入 §4.3 + yaml 切换 §4.4 deferred）
5. ✅ 应用 §5 P8 patch（ebe03a1）
6. ✅ 应用 §6 P5 patch（边界适配，ebe03a1）
7. ✅ §7.1 静态验证（所有 worker import + 8 个 P8 子脚本路径全部 OK）
8. ⬜ §7.2/7.3/7.4 E2E（用户在场启动）
9. ✅ §4.3 concat-aug 接入 + §4.4 colent yaml 切换（B 路线：augment train_info_ml；ASL precompute 进度独立，degraded 21x 模式可跑）

---

## 10. 已废弃的草稿

- `backend/workers/PHASE4_PATCH_NOTES.md` —— 以 `spamo_segement/` 为目标，方向偏
- `backend/workers/PHASE8_PATCH_NOTES.md` —— 以 `gloss_aware/` 为目标，方向偏

两份保留在 commit 里作为历史，不再更新。
