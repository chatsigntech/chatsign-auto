# Pipeline 各阶段独立性设计

**设计日期**: 2026-03-22
**原则**: 各阶段相对独立，通过标准输入/输出接口通信，允许单独升级

---

## 📋 6阶段Pipeline总览

```
Phase 1        Phase 2        Phase 3        Phase 4        Phase 5        Phase 6
上传语句  →  词汇拆解  →  手语录制  →  视频预处理  →  数据增广  →  模型训练
(手动)      (自动)      (手动)       (自动)         (GPU自动)    (手动触发)
```

---

## Phase 1: 上传场景语句 (手动)

### ✅ 独立性：最高

| 方面 | 说明 |
|------|------|
| **输入** | 用户上传 CSV 文件 |
| **处理** | 验证格式、计数、存储 |
| **输出** | `phase1/inputs/sentences.csv` |
| **依赖** | 无 |
| **后续升级** | ✅ 可独立升级（仅改变验证逻辑） |

### 📄 输出文件格式

```csv
# sentences.csv
sentence_1,英文句子1
sentence_2,英文句子2
sentence_3,英文句子3
...
```

### 🔄 状态转移

```
pending → 用户上传CSV
        ↓
    验证和解析
        ↓
completed → 自动触发 Phase 2
```

---

## Phase 2: 词汇拆解 (自动)

### ✅ 独立性：高

| 方面 | 说明 |
|------|------|
| **输入** | `phase1/inputs/sentences.csv` |
| **处理** | 调用外部模块（pseudo-gloss-English 或 inline subprocess） |
| **输出** | `phase2/outputs/glosses.json` + `glosses_for_import.csv` |
| **依赖** | pseudo-gloss-English (但通过 subprocess，非代码耦合) |
| **后续升级** | ✅ 可独立升级词汇提取逻辑（更换 NLP 模型） |

### 📄 输出文件格式

```json
// glosses.json
{
  "vocabulary": [
    {"gloss_id": "apple", "count": 5, "sources": ["sentence_1", "sentence_2"]},
    {"gloss_id": "banana", "count": 3, "sources": ["sentence_3"]},
    ...
  ],
  "total_unique": 85,
  "total_instances": 150
}
```

```csv
# glosses_for_import.csv
gloss_id,description,count
apple,apple fruit,5
banana,yellow fruit,3
...
```

### 🔄 状态转移

```
pending → 读取 Phase 1 输出
        ↓
    调用 NLP 模块
        ↓
completed → 自动触发 Phase 3 等待状态
```

### 🔧 升级点

```
可以独立替换：
- NLP 引擎（pseudo-gloss-English → 其他词汇提取库）
- 词汇过滤规则
- 去重和统计逻辑
- 输出格式（只要保持 JSON/CSV 接口）
```

---

## Phase 3: 手语录制审核 (手动)

### ✅ 独立性：中

| 方面 | 说明 |
|------|------|
| **输入** | `phase2/outputs/glosses_for_import.csv` |
| **处理** | 外链到 chatsign-accuracy，用户手动录制审核 |
| **输出** | `/data/chatsign/shared/{task_id}/gloss_db/gloss_to_video_database.json` + `recordings/*.mp4` |
| **依赖** | chatsign-accuracy (外链，非代码耦合) |
| **后续升级** | ⚠️ 受限：需改 chatsign-accuracy 导出逻辑（但仅涉及配置改动） |

### 📄 输出文件格式

```json
// gloss_to_video_database.json
{
  "metadata": {
    "task_id": "abc123",
    "created_at": "2026-03-22T10:00:00",
    "total_glosses": 85,
    "total_recorded": 85
  },
  "glosses": {
    "apple": {
      "video_file": "recordings/apple.mp4",
      "duration": 2.5,
      "quality": "approved",
      "reviewer": "user@example.com"
    },
    "banana": {...},
    ...
  }
}
```

### 🔄 状态转移

```
waiting → 用户点击"打开录制网站"
       ↓
   (在 chatsign-accuracy 完成录制)
       ↓
   用户确认 Phase 3 完成
       ↓
verified → 自动触发 Phase 4
```

### 🔧 升级点

```
可以独立改进：
- 录制网站 UI/UX（chatsign-accuracy 内部改动）
- 审核标准（质量检查规则）
- 导出格式（只要保持 JSON 接口和文件位置）
- 支持多人录制（修改数据库结构，但接口可保持兼容）
```

---

## Phase 4: 视频预处理 (自动)

### ✅ 独立性：高

| 方面 | 说明 |
|------|------|
| **输入** | `/data/chatsign/shared/{task_id}/recordings/*.mp4` + `gloss_db/gloss_to_video_database.json` |
| **处理** | 5步子流程：提取帧 → 去重 → 过滤姿态 → 缩放 → 生成清洁视频 |
| **输出** | `phase4/outputs/{gloss_id}/video_clean.mp4` + `ref_image.jpg` + `preprocessing_report.json` |
| **依赖** | UniSignMimicTurbo 的 5 个脚本（通过 subprocess，非代码耦合） |
| **后续升级** | ✅ 可独立升级每一步（替换预处理算法） |

### 📄 输出文件结构

```
phase4/outputs/
├── apple/
│   ├── video_clean.mp4      (清洁视频)
│   └── ref_image.jpg        (参考图片)
├── banana/
│   ├── video_clean.mp4
│   └── ref_image.jpg
└── preprocessing_report.json
```

```json
// preprocessing_report.json
{
  "total_processed": 85,
  "successful": 84,
  "failed": 1,
  "skipped": 0,
  "details": {
    "apple": {"status": "success", "frames": 75, "duration": 2.5},
    "banana": {"status": "success", "frames": 60, "duration": 2.0},
    "cherry": {"status": "failed", "reason": "corrupted_video"}
  },
  "statistics": {
    "avg_frame_count": 65,
    "avg_duration": 2.2,
    "processing_time": 3600
  }
}
```

### 🔄 状态转移

```
pending → 读取 Phase 3 输出
       ↓
   Step 4.1-4.5 依次执行
       ↓
completed → 自动排队 Phase 5
```

### 🔧 升级点

```
可以独立替换：
- Step 4.1: 帧提取方法（fps、采样策略）
- Step 4.2: 去重算法（阈值、方法）
- Step 4.3: 姿态过滤（模型、阈值）
- Step 4.4: 缩放方法（分辨率、插值）
- Step 4.5: 编码参数（codec、bitrate）

升级时无需改动其他阶段
```

---

## Phase 5: 数据增广 (GPU自动)

### ✅ 独立性：高

| 方面 | 说明 |
|------|------|
| **输入** | `phase4/outputs/{gloss_id}/video_clean.mp4` + `augmentation_config.yaml` |
| **处理** | 多GPU并行：MimicMotion 生成增广变体 |
| **输出** | `phase5/outputs/{gloss_id}/variant_*.mp4` + `manifest.json` |
| **依赖** | UniSignMimicTurbo/MimicMotion (通过 subprocess，非代码耦合) |
| **后续升级** | ✅ 可独立升级（增广参数、GPU分配、生成模型） |

### 📄 输入配置文件

```yaml
# phase5/config/augmentation_config.yaml (任务专属副本)
preset: "medium"
seeds: [0, 1, ..., 19]                    # 20个
noise_strengths: [0.0, 0.1, ..., 1.0]    # 11个
num_inference_steps: [15, 25]             # 2个
guidance_scales: [2.0, 5.0]               # 2个
schedulers: ["AnimateLCM_SVD"]            # 1个

execution:
  num_parallel_gpus: 8
  batch_size_per_gpu: 1
  enable_feature_cache: true
  force_refresh_cache: false
```

### 📄 输出文件格式

```
phase5/outputs/
├── apple/
│   ├── variant_0_0.0_15_2.0.mp4
│   ├── variant_0_0.1_15_2.0.mp4
│   └── ... (880个变体)
├── banana/
│   └── ... (880个变体)
└── manifest.json
```

```json
// manifest.json
{
  "task_id": "task_abc123",
  "preset": "medium",
  "total_variants": 74800,
  "total_glosses": 85,
  "variants_per_gloss": 880,
  "timestamp": "2026-03-22T10:00:00",
  "glosses": {
    "apple": {
      "source_video": "phase4/outputs/apple/video_clean.mp4",
      "total_variants": 880,
      "variants": [
        {"variant_id": "variant_0_0.0_15_2.0", "seed": 0, "noise": 0.0, "steps": 15, "guidance": 2.0},
        {"variant_id": "variant_0_0.1_15_2.0", "seed": 0, "noise": 0.1, "steps": 15, "guidance": 2.0},
        ...
      ]
    }
  }
}
```

### 🔄 状态转移

```
pending → 生成参数组合
       ↓
   按GPU数量切分
       ↓
   启动N个GPU子进程 (并行)
       ↓
progress_update → 每3秒通过WebSocket推送进度
       ↓
completed → 用户确认质量，手动触发 Phase 6
```

### 🔧 升级点

```
可以独立替换：
- 增广参数（preset: light/medium/heavy/custom）
- GPU并行策略（数量、batch size）
- 生成模型（MimicMotion → 其他模型）
- 缓存策略（特征缓存启/禁）
- 断点续传逻辑

升级时仅需更新：
- augmentation_config.yaml
- phase5_worker.py 内的参数生成逻辑
- 其他阶段无需改动
```

---

## Phase 6: 模型训练 (手动触发)

### ✅ 独立性：高

| 方面 | 说明 |
|------|------|
| **输入** | `phase5/outputs/` + `phase2/outputs/glosses.json` |
| **处理** | 5步子流程：标注生成 → pose提取 → 过滤 → 归一化 → 训练 → 构建原型 |
| **输出** | `phase6/outputs/best.pth` + `prototypes.pt` + `gloss_code_stats.pkl` + `vq_codebook.pt` |
| **依赖** | gloss_aware 模块（通过 subprocess，非代码耦合） |
| **后续升级** | ✅ 可独立升级（训练配置、模型架构、优化参数） |

### 📄 标注文件格式

```
phase6/annotations/
├── vocab.json              (词表)
├── train.jsonl             (训练集：增广视频)
├── dev.jsonl               (验证集：原始清洁视频)
└── test.jsonl              (可选：测试集)
```

```json
// vocab.json
{
  "token_to_id": {
    "apple": 0,
    "banana": 1,
    ...
  },
  "id_to_token": {
    "0": "apple",
    "1": "banana",
    ...
  },
  "total_tokens": 85
}
```

```json
// train.jsonl (每行一条)
{"utterance_id": "apple_v000", "tokens": ["apple"], "pose_path": "dataset/asl/pose_format_clip/apple_v000.pkl", "split": "train"}
{"utterance_id": "apple_v001", "tokens": ["apple"], "pose_path": "dataset/asl/pose_format_clip/apple_v001.pkl", "split": "train"}
...
```

### 📄 Phase 6 子步骤

```
Step 6.0: 标注生成 (编排器内部)
          → 读取 Phase 5 manifest.json
          → 生成 vocab.json, train.jsonl, dev.jsonl

Step 6.1: Pose 提取 (pose_extractor.py)
          输入: phase5/outputs/{gloss_id}/variant_*.mp4
          输出: phase6/pose_pkls/raw/{utterance_id}.pkl

Step 6.2: 置信度过滤 (filter_pose_pkls.py)
          输入: phase6/pose_pkls/raw/*.pkl
          输出: phase6/pose_pkls/clip/*.pkl

Step 6.3: 归一化 (batch_norm_cosign_padding.py)
          输入: phase6/pose_pkls/clip/*.pkl
          输出: phase6/pose_pkls/norm/*.pkl

Step 6.4: 模型训练 (torchrun ssl_pretraining_*.py)
          输入: 标注 + pose files
          输出: best.pth, best_cl.pth

Step 6.5: 构建原型库 (build_prototypes_asl_clip_nob2b.py)
          输入: best.pth
          输出: prototypes.pt, gloss_code_stats.pkl, vq_codebook.pt
```

### 🔄 状态转移

```
pending → 用户选择训练配置（epochs, batch-size等）
       ↓
   Step 6.0-6.3: 准备数据 (可再用已有缓存)
       ↓
   Step 6.4: 训练 (进度通过日志文件轮询推送)
       ↓
   Step 6.5: 构建原型
       ↓
completed → 版本号自动递增（v1 → v2）
         → 文件通过 SCP 同步到推理服务器
```

### 🔧 升级点

```
可以独立替换：
- Step 6.1-6.3: 姿态处理算法（模型、过滤阈值）
- Step 6.4: 训练脚本（模型架构、优化器、损失函数）
- Step 6.5: 原型构建逻辑（聚合方法）
- 训练超参数（epochs, batch-size, lr, scheduler）

升级时仅需更新：
- phase6_worker.py 中的参数配置
- gloss_aware 中的脚本
- 其他阶段无需改动
```

---

## 🔗 阶段间的数据接口

### 数据流向

```
Phase 1          Phase 2              Phase 3
输出 CSV    →   输出 JSON         →   输出 MP4 + JSON
sentences.csv   glosses.json        gloss_db.json
                                     recordings/

                                    Phase 4
                                    输出 MP4 + 报告
                                    phase4/outputs/

                                    Phase 5
                                    输出 MP4 + manifest
                                    phase5/outputs/

                                    Phase 6
                                    输出模型文件
                                    phase6/outputs/
```

### 关键约定（稳定的接口）

| 接口 | 格式 | 位置 | 说明 |
|------|------|------|------|
| Phase 1 → 2 | CSV | `phase1/inputs/sentences.csv` | 行格式：`id,sentence` |
| Phase 2 → 3 | CSV | `phase2/outputs/glosses_for_import.csv` | 行格式：`gloss_id,description,count` |
| Phase 2 内 | JSON | `phase2/outputs/glosses.json` | 标准词汇表格式 |
| Phase 3 → 4 | JSON + MP4 | `/data/chatsign/shared/{task_id}/` | gloss_db.json + recordings/ |
| Phase 4 → 5 | MP4 + JSON | `phase4/outputs/` | {gloss_id}/video_clean.mp4 + preprocessing_report.json |
| Phase 5 → 6 | MP4 + JSON | `phase5/outputs/` | {gloss_id}/variant_*.mp4 + manifest.json |
| Phase 6 输出 | .pt/.pkl | `phase6/outputs/` | best.pth, prototypes.pt, gloss_code_stats.pkl, vq_codebook.pt |

---

## ✅ 独立性验证清单

### Phase 1: 上传语句
- [x] 完全独立，无外部依赖
- [x] 输出格式固定（CSV）
- [x] 可单独升级验证逻辑

### Phase 2: 词汇拆解
- [x] 通过 subprocess 调用外部，松耦合
- [x] 输出为 JSON/CSV，格式标准
- [x] 可替换 NLP 引擎（保持接口）

### Phase 3: 录制审核
- [x] 通过外链到 chatsign-accuracy，松耦合
- [x] 输出为 JSON + MP4，格式标准
- [x] chatsign-accuracy 内改动，orchestrator 无需改

### Phase 4: 预处理
- [x] 通过 subprocess 调用 UniSignMimicTurbo，松耦合
- [x] 5步子流程可独立升级
- [x] 输出为 MP4 + 报告，格式标准

### Phase 5: 增广
- [x] 通过 subprocess 调用 MimicMotion，松耦合
- [x] 配置驱动（augmentation_config.yaml）
- [x] 可独立升级增广策略或参数

### Phase 6: 训练
- [x] 通过 subprocess 调用 gloss_aware，松耦合
- [x] 5步子流程可独立升级
- [x] 可独立改进训练配置和模型

---

## 🎯 单独升级示例

### 示例1：升级 Phase 5 增广策略

```
当前：Medium 预设（880变体/词）
目标：改为 Heavy 预设（4400变体/词）

改动：
✅ 修改 augmentation_config.yaml
✅ 无需改其他阶段
✅ Phase 4 输出不变，Phase 6 输入不变
✅ 后台重新运行 Phase 5 即可
```

### 示例2：升级 Phase 4 姿态过滤阈值

```
当前：hand_threshold=0.8, head_threshold=0.9
目标：改为 hand_threshold=0.7, head_threshold=0.85

改动：
✅ 修改 Phase 4 脚本的参数
✅ 无需改其他阶段
✅ Phase 3 输出不变，Phase 5 输入格式不变
✅ 后台重新运行 Phase 4 即可
```

### 示例3：升级 Phase 6 训练参数

```
当前：epochs=100, batch-size=64
目标：改为 epochs=150, batch-size=32

改动：
✅ 修改 phase6_worker.py 的参数
✅ 无需改其他阶段
✅ Phase 5 输出不变，Phase 6 输出格式不变
✅ 后台重新运行 Phase 6 即可
```

---

## 📝 总结

| 阶段 | 独立性 | 外部依赖 | 升级难度 | 推荐周期 |
|------|--------|---------|---------|---------|
| Phase 1 | ⭐⭐⭐⭐⭐ | 无 | 简 | 按需 |
| Phase 2 | ⭐⭐⭐⭐ | pseudo-gloss | 简 | 按需 |
| Phase 3 | ⭐⭐⭐ | chatsign-accuracy | 中 | 按需 |
| Phase 4 | ⭐⭐⭐⭐ | UniSignMimicTurbo | 简 | 定期 |
| Phase 5 | ⭐⭐⭐⭐ | MimicMotion | 简 | 定期 |
| Phase 6 | ⭐⭐⭐⭐ | gloss_aware | 中 | 定期 |

**设计原则成立** ✅：
- 各阶段通过明确的数据接口通信
- 可以独立升级任何阶段
- 无需改动其他阶段
- 整个Pipeline保持向后兼容

