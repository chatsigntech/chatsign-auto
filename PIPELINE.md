# ChatSign Pipeline — 8-Phase Sign Language Processing

## Pipeline Overview

```
Phase 1 → Phase 2 (阻塞：等待人工录制审核) ─┬── Phase 3 (独立分支，不参与训练)
提取推送    采集整理                           │
                                             └── Phase 4 → Phase 5 → Phase 6 → Phase 7 → Phase 8
                                                 分割训练    句子分割    数据增广    增广分割    模型训练
```

---

## Data Flow

```
Phase 1: Gloss 提取与推送
    │
    ▼
Phase 2: 视频采集与整理 (阻塞点：等待人工录制和审核)
    │
    │   Phase 2 输出包含：句子视频 + 单词视频
    │
    ├── Phase 3: 标准手语视频生成 (独立分支，仅用于导出)
    │
    ├── Phase 4: 分割模型训练 ◄── Phase 2 句子视频 + Phase 1 glosses
    │       │
    │       ▼
    ├── Phase 5: 原生句子分割 ◄── Phase 4 模型 + Phase 2 句子视频
    │       │                      输出：分割词视频文件 + 分割点数据
    │       │
    │       ▼
    ├── Phase 6: 数据增广     ◄── Phase 2 句子视频 + Phase 2 单词视频 + Phase 5 分割词视频
    │       │
    │       ▼
    ├── Phase 7: 增广句子分割 ◄── Phase 6 增广句子视频 + Phase 5 分割点数据 (不调用模型)
    │       │
    │       ▼
    └── Phase 8: 模型训练     ◄── Phase 2 原始视频 + Phase 5 分割词视频
                                   + Phase 6 增广视频 + Phase 7 增广分割视频
                                   (每类视频 + 对应标注文本)
```

---

## Phase Descriptions

### Phase 1: Gloss 提取与推送

**合并自旧 Phase 1 + Phase 2**

- **Step 1.1**: 用 spaCy `en_core_web_sm` 对用户输入文本做 POS 过滤，提取伪 gloss（NOUN, VERB, ADJ, ADV, NUM, PRON, PROPN → 大写）
- **Step 1.2**: 将 gloss 和句子推送到 chatsign-accuracy 系统，生成录制批次，等待人工录制和审核
- **Input**: 用户输入文本
- **Output**: `glosses.json`, `vocab.json`, 推送到 accuracy 系统的录制批次
- **GPU**: 不需要

### Phase 2: 视频采集与整理

**合并自旧 Phase 3 + Phase 4**

**阻塞点**：正式运行时阻塞在此阶段，等待人工完成录制和审核后继续。

- **Step 2.1**: 从 chatsign-accuracy 收集已审核通过的视频（句子视频 + 单词视频），生成 `manifest.json` + `sentences.txt`
- **Step 2.2**: 合并 manifest 与 glosses，生成 `annotations.json`
- **Step 2.3**: 视频预处理（提取帧 → 去重 → 缩放 576 → 重新生成视频）
- **Input**: Phase 1 推送的批次（审核完成后）
- **Output**: `manifest.json`, `annotations.json`, `videos/`（句子视频 + 单词视频）, `sentences.txt`
- **GPU**: 不需要

### Phase 3: 标准手语视频生成

**合并自旧 Phase 6 + Phase 7 + Phase 8**

独立分支，生成可导出的手语视频，用于外部 text-to-sign 程序。**不参与后续训练流程。**
可与 Phase 4~8 串行或并行执行，根据 GPU 资源决定。

- **Step 3.1**: 换人（MimicMotion）— 将视频中的人替换为目标人物
  - 视频 < 16 帧：跳过
  - 首次失败：自动降低 `num_inference_steps` 重试
  - 所有尝试失败：排除
- **Step 3.2**: 视频处理 — 提取帧 → 姿态质量过滤（hand ≥ 0.8, head ≥ 0.9）→ 缩放 512×320 → 提取边界帧 → 生成清理视频
- **Step 3.3**: 帧插值（FramerTurbo）
  - Word-level: 生成 rest↔sign 手部回归过渡（intro + word + outro）
  - Sentence-level: 词间平滑插值
  - Fallback: checkpoint 不存在时直接透传
- **Input**: Phase 2 预处理视频
- **Output**: 可导出的标准手语视频
- **GPU**: 需要（MimicMotion + FramerTurbo）

### Phase 4: 分割模型训练 (SpaMo)

**来自旧 Phase 5，保留训练部分**

原代码 `phase_segmentation.py`（已删除）将训练和推理写在一起（Step 5.1~5.6）。新 Pipeline 拆分为：
- Phase 4 负责特征提取 + 模型训练（原 Step 5.1 ~ 5.4）
- Phase 5 负责推理分割（原 Step 5.5 ~ 5.6）

子步骤：
- **Step 4.1**: 提取 CLIP-ViT 空间特征（S2Wrapping, scales 1+2）
- **Step 4.2**: 生成标注文件（train_info_ml.npy + val_info_ml.npy）
- **Step 4.3**: 生成任务配置 YAML
- **Step 4.4**: 训练分割模型（SpaMo: Flan-T5-XL + LoRA + OT alignment）
- **Input**: Phase 2 句子视频 + Phase 1 glosses
- **Output**: 训练好的分割模型 checkpoint
- **GPU**: 需要

### Phase 5: 原生句子分割

**新增阶段（从旧 Phase 5 推理部分独立出来）**

用 Phase 4 训练好的分割模型，对 Phase 2 采集的原始句子视频做时序分割。

- **Step 5.1**: 对 Phase 2 的句子视频运行分割推理
- **Step 5.2**: 按分割点将句子视频切割成独立的词片段视频文件
- **Step 5.3**: 记录每个句子视频的分割点数据（供 Phase 7 复用）
- **Input**: Phase 4 分割模型 checkpoint + Phase 2 句子视频
- **Output**:
  - 分割词视频文件（独立的词片段 mp4）
  - 分割点数据（每个句子视频的词级时间边界，供 Phase 7 直接使用）
- **GPU**: 需要

### Phase 6: 数据增广

**来自旧 Phase 9，扩展增广范围**

三类增广：

| 增广类型 | 说明 | 数据来源 |
|---------|------|---------|
| 句子增广 | 对完整句子视频做 2D/时序增广 | Phase 2 句子视频 |
| 词语增广 | 对单词级视频做 2D/时序增广 | Phase 2 单词视频 |
| 分割词增广 | 对分割产出的词片段视频做 2D/时序增广 | Phase 5 分割词视频 |

现有增广方法（保持不变）：
- 2D CV 增广：25 种（12 几何 + 13 色彩），CPU-only
- 时序增广：7 种（5 速度 + 2 FPS），CPU-only
- 3D 视角增广：6 固定视角，GPU（当前默认禁用）
- 身份交叉增广：GUAVA cross-act，GPU（当前默认禁用）

- **Input**: Phase 2 句子视频 + Phase 2 单词视频 + Phase 5 分割词视频
- **Output**: 增广后的视频（句子 + 词语 + 分割词），每类视频保留对应的标注文本
- **GPU**: 3D/身份增广需要（当前默认禁用）

### Phase 7: 增广句子分割

**新增阶段**

对 Phase 6 增广产出的句子视频做分割。**不调用分割模型**，直接复用 Phase 5 记录的原视频分割点进行切割：

- **2D CV 增广句子**：直接复用原始分割点（几何/色彩变换不改变时序）
- **时序增广句子**：按变速比例换算分割点（Phase 6 记录了每个输出视频的时序变换参数）

- **Input**: Phase 6 增广句子视频 + Phase 5 分割点数据 + Phase 6 时序变换参数
- **Output**: 增广句子的词级分割视频 + 对应标注文本
- **GPU**: 不需要（无模型推理，仅按分割点切割视频）

### Phase 8: 模型训练

**来自旧 Phase 10，输入源调整**

输入为视频 + 对应标注文本。训练过程中从输入视频提取姿态数据。

- **Step 8.1**: 姿态提取（RTMPose）— 从所有输入视频提取关键点
- **Step 8.2**: 姿态过滤 — 按质量阈值过滤
- **Step 8.3**: 姿态归一化 — 133 关键点 → 4 部分格式（body, hand, face, pose）
- **Step 8.4**: 数据验证 — 去除损坏和过短的 pkl 文件
- **Step 8.5**: 数据集生成 — 构建 train.jsonl / dev.jsonl / vocab.json
- **Step 8.6**: SSL 预训练（SignCL）
- **Step 8.7**: 构建 gloss 原型
- **Input**（视频 + 对应标注文本）:
  - Phase 2 原始视频（句子 + 单词）
  - Phase 5 原生分割词视频
  - Phase 6 增广视频（句子 + 词语 + 分割词）
  - Phase 7 增广分割视频
- **Output**: `best.pth`, `prototypes/`, `vocab.json`, `train.jsonl`
- **GPU**: 需要

---

## Execution Order

Phase 2 完成后存在两条路径：

```
Phase 2 完成 (阻塞点解除)
    │
    ├── Phase 3 (独立分支，可导出手语视频)
    │
    └── Phase 4 → Phase 5 → Phase 6 → Phase 7 → Phase 8 (训练主线)
```

- Phase 3 与 Phase 4~8 可串行或并行，根据 GPU 资源决定
- Phase 4~8 之间为严格串行依赖

---

## Error Handling

### Phase 3 换人报告 (`phase3_report.json`)

| Status | Meaning | Action |
|--------|---------|--------|
| `success` | 首次换人成功 | 继续后续处理 |
| `retry_success` | 降参后成功 | 继续后续处理 |
| `skipped_short` | 视频过短 (< 16 帧) | **排除** |
| `failed` | 所有重试失败 | **排除** |

---

## External Dependencies

| Component | Project | Required For |
|-----------|---------|-------------|
| spaCy `en_core_web_sm` | — | Phase 1 |
| MimicMotion model | `models/MimicMotion_1-1.pth` (2.8GB) | Phase 3 |
| SVD-XT model | `models/svd-xt-1-1/` | Phase 3 |
| FramerTurbo checkpoint | `FramerTurbo/checkpoints/framer_512x320/` | Phase 3 (optional) |
| SpaMo segmentation | `spamo_segement/` | Phase 4, 5 |
| GUAVA 3D model | `guava-aug/assets/GUAVA/` | Phase 6 3D views (disabled) |
| RTMPose ONNX | `~/.cache/rtmlib/` (auto-downloaded) | Phase 3, 8 |
