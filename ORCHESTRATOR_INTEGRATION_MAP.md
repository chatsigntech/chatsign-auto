# Orchestrator与5个子项目的集成关系

**梳理日期**: 2026-03-22
**问题**: 哪些子项目需要改动？

---

## 📊 集成关系总览

```
┌─────────────────────────────────────────────────────────┐
│          chatsign-orchestrator (新建)                   │
│  独立管理后台 + 任务管理 + Pipeline编排                │
│          Phase 1-6 核心流程                             │
└─────────────────────────────────────────────────────────┘
                    │
                    ├──→ Phase 2: pseudo-gloss-English
                    ├──→ Phase 3: chatsign-accuracy
                    ├──→ Phase 4: UniSignMimicTurbo
                    ├──→ Phase 5: UniSignMimicTurbo (增广)
                    ├──→ Phase 6: gloss_aware (训练)
                    └──→ Phase 6: inference-service (模型部署)
```

---

## 🔗 5个子项目详细关系

### 1️⃣ pseudo-gloss-English（词汇拆解）

| 项目 | 位置 | Phase |
|------|------|-------|
| 项目 | `/opt/chatsign/pseudo-gloss-English` | Phase 2 |
| 功能 | 英文句子 → 词汇列表 | 自动 |

**集成方式**: subprocess调用
```python
# 脚本所在：pseudo-gloss-English/pseudo_gloss_en.py
# 问题：脚本无argparse，无法直接CLI调用

# 解决方案：inline subprocess (直接嵌入Python代码)
INLINE_SERVER = """
import spacy, json
nlp = spacy.load('en_core_web_sm')
...
"""
subprocess.Popen(['python', '-c', INLINE_SERVER], ...)
```

**需要改动原项目文件？** ❌ **NO**
- 仅调用pseudo_gloss_en.py中的spacy逻辑
- 通过inline subprocess方式避免修改原文件
- 仅需确保 `en_core_web_sm` 模型已安装

**需要的.env配置**:
```bash
PSEUDO_GLOSS_PATH=/opt/chatsign/pseudo-gloss-English
```

---

### 2️⃣ chatsign-accuracy（手语录制和审核）

| 项目 | 位置 | Phase |
|------|------|-------|
| 项目 | `/opt/chatsign/chatsign-accuracy` | Phase 3 |
| 功能 | 手语视频录制→审核→导出 | 手动 |

**集成方式**: 外链 + 共享文件系统

```
录制网站链接: /recording/?task_id={task_id}
导出路径: /data/chatsign/shared/{task_id}/
  ├─ gloss_db/gloss_to_video_database.json
  └─ recordings/*.mp4
```

**需要改动原项目文件？** ✅ **YES - 2处小改动**

#### 改动1️⃣: 支持task_id参数 (.env配置)

**文件**: `chatsign-accuracy/backend/.env`

```bash
# 原配置
EXPORT_BASE_DIR=/data/chatsign/exports

# 新配置（需改动）
EXPORT_BASE_DIR=/data/chatsign/shared
EXPORT_USE_TASK_ID=true  # 启用task_id目录隔离
```

#### 改动2️⃣: 导出时使用task_id作为子目录 (可选)

**文件**: `chatsign-accuracy/backend/routes/export.js` (或类似)

**改动类型**: 薄的配置逻辑，不改核心功能

```javascript
// 原逻辑
const exportPath = `${EXPORT_BASE_DIR}/export_${timestamp}`;

// 新逻辑（改动，但简单）
const taskId = req.query.task_id || req.body.task_id;
const exportPath = taskId
  ? `${EXPORT_BASE_DIR}/${taskId}`
  : `${EXPORT_BASE_DIR}/export_${timestamp}`;
```

**影响范围**:
- ❌ 不影响录制功能
- ❌ 不影响审核功能
- ✅ 仅改变导出目录结构
- ✅ 可向后兼容（不传task_id时用时间戳）

**需要的.env配置**:
```bash
PHASE3_SHARED_DIR=/data/chatsign/shared
RECORDING_SITE_URL=/recording/
```

---

### 3️⃣ UniSignMimicTurbo（预处理 + 增广）

| 项目 | 位置 | Phase |
|------|------|-------|
| 项目 | `/opt/chatsign/UniSignMimicTurbo` | Phase 4, 5 |
| 功能 | 视频预处理(5步) + 数据增广 | 自动 |

**集成方式**: subprocess调用已有脚本

#### Phase 4预处理（5步）

```python
# 调用链（全部有argparse）
Step 4.1: scripts/sentence/extract_all_frames_seq.py
Step 4.2: scripts/sentence/filter_duplicate_frames.py
Step 4.3: scripts/sentence/filter_frames_by_pose.py
Step 4.4: scripts/sentence/resize_frames.py
Step 4.5: scripts/sentence/generate_videos_from_frames.py
```

**需要改动原项目文件？** ❌ **NO**
- 所有脚本已有argparse
- orchestrator直接调用，无需改动
- 仅需通过参数传入路径

#### Phase 5增广（多GPU并行）

```python
# 调用：scripts/inference/inference_raw_batch_cache.py
# 脚本已有argparse，支持--inference_config参数
# orchestrator生成配置文件，脚本读取执行
```

**需要改动原项目文件？** ❌ **NO**
- 脚本已支持YAML配置
- orchestrator生成augmentation_config.yaml副本
- 脚本无需改动，直接调用

**需要的.env配置**:
```bash
UNISIGN_PATH=/opt/chatsign/UniSignMimicTurbo
```

---

### 4️⃣ gloss_aware（模型训练）

| 项目 | 位置 | Phase |
|------|------|-------|
| 项目 | `/opt/chatsign/chatsign-auto/gloss_aware` | Phase 6 |
| 功能 | 训练VQ-VAE + 构建原型库 | 手动触发 |

**集成方式**: subprocess调用 Phase 6 的5个脚本

```python
# Step 6.0: 生成标注 (orchestrator内部)
# Step 6.1: gloss_aware/preprocess/pose_extractor.py (subprocess)
# Step 6.2: gloss_aware/preprocess/filter_pose_pkls.py (subprocess)
# Step 6.3: gloss_aware/preprocess/batch_norm_cosign_padding.py (subprocess)
# Step 6.4: gloss_aware/ssl_pretraining_crossvideo_mlp_*.py (torchrun)
# Step 6.5: gloss_aware/build_prototypes_asl_clip_nob2b.py (subprocess)
```

**需要改动原项目文件？** ✅ **NO - 已验证所有脚本都有argparse**

#### 验证结果（已确认）

| 脚本 | argparse? | 改动需求 |
|------|----------|---------|
| pose_extractor.py | ✅ 有 (line 238) | ❌ 无需改动 |
| filter_pose_pkls.py | ✅ 有 (line 47) | ❌ 无需改动 |
| batch_norm_cosign_padding.py | ✅ 有 (line 15) | ❌ 无需改动 |
| ssl_pretraining_crossvideo_mlp_*.py | ✅ 有 (line 71) | ❌ 无需改动 |
| build_prototypes_asl_clip_nob2b.py | ✅ 有 (line 44) | ❌ 无需改动 |

**结论**: gloss_aware 所有脚本都完整支持 CLI 调用，**无需任何改动**！

**需要的.env配置**:
```bash
GLOSS_AWARE_PATH=/opt/chatsign/chatsign-auto/gloss_aware
```

---

### 5️⃣ 推理服务（模型部署）

| 项目 | 位置 | 用途 |
|------|------|------|
| 项目 | `/opt/gloss_aware/models/asl_clip/v{N}/` | 推理 |
| 功能 | 版本管理 + 增量部署 | Phase 6输出 |

**集成方式**: 文件部署（不涉及代码改动）

**需要改动原项目文件？** ❌ **NO**
- 推理脚本（infer_asl_clip_nob2b_accuracy.py）无需改动
- 仅需更新推理服务的配置文件指向新版本路径
- Phase 6 output → 拷贝prototypes.pt + gloss_code_stats.pkl → 推理服务

**版本管理结构**:
```
/opt/gloss_aware/models/asl_clip/
├── v1/
│   ├── best_cl.pth
│   ├── prototypes.pt
│   ├── gloss_code_stats.pkl
│   ├── vq_codebook.pt
│   └── MANIFEST.json (v1: initial)
├── v2/
│   ├── best_cl.pth
│   ├── prototypes.pt (新)
│   ├── gloss_code_stats.pkl (新)
│   ├── vq_codebook.pt
│   └── MANIFEST.json (v2: from v1 + new 100 vocab)
└── v3/ ...
```

**部署脚本示例** (orchestrator生成):
```bash
#!/bin/bash
# Phase 6完成后执行
INFERENCE_SERVER="inference_user@inference_server"
INFERENCE_PATH="/opt/gloss_aware/models/asl_clip/v{new_version}/"

# 仅拷贝2个文件到推理服务器
scp {TRAIN_PATH}/prototypes.pt $INFERENCE_SERVER:$INFERENCE_PATH/
scp {TRAIN_PATH}/gloss_code_stats.pkl $INFERENCE_SERVER:$INFERENCE_PATH/

# 推理脚本无需改动，自动使用新文件
```

**需要的.env配置**:
```bash
GLOSS_AWARE_INFERENCE_PATH=/opt/gloss_aware/models/asl_clip
INFERENCE_SERVER=user@host
INFERENCE_DEPLOY_KEY=/path/to/ssh/key
```

---

## 📋 改动总结表

| 项目 | 需要改动? | 改动文件 | 改动类型 | 影响范围 |
|------|---------|--------|--------|--------|
| pseudo-gloss-English | ❌ NO | — | — | — |
| chatsign-accuracy | ✅ YES | .env, routes/export.js | 配置+薄逻辑 | 仅导出目录结构 |
| UniSignMimicTurbo | ❌ NO | — | — | — |
| gloss_aware | ⚠️ 可能 | preprocess/*.py | argparse添加 | 仅CLI接口 |
| 推理服务 | ❌ NO | — | — | — |

---

## ✅ 改动原则（核心）

```
❌ 禁止：修改子项目核心业务逻辑
✅ 允许：添加薄的配置/CLI接口
✅ 允许：改变文件组织/目录结构
❌ 禁止：改变外部项目的数据处理算法
```

---

## 🚀 实施步骤

### 第1步：gloss_aware脚本检查
```bash
# 检查哪些脚本需要添加argparse
grep -l "if __name__ == '__main__':" gloss_aware/preprocess/*.py
grep -l "argparse" gloss_aware/preprocess/*.py

# 对比找出缺argparse的脚本
```

### 第2步：chatsign-accuracy改动
```bash
# 添加配置支持task_id的导出目录
# 改动: export.js 或对应的导出路由
# 完成后可向后兼容
```

### 第3步：测试集成
```bash
# Phase 2: pseudo-gloss-English (inline)
# Phase 3: chatsign-accuracy (with task_id)
# Phase 4: UniSignMimicTurbo (existing CLI)
# Phase 5: UniSignMimicTurbo (inference_raw_batch_cache.py)
# Phase 6: gloss_aware (check/add argparse)
```

---

## 📋 最终检查清单

### 在开始编码前，需要确认：

- [ ] **pseudo-gloss-English**: en_core_web_sm 模型已安装
- [ ] **chatsign-accuracy**: 确认导出逻辑，是否支持custom路径
- [ ] **UniSignMimicTurbo**: 所有4步脚本都有 `--input-dir`, `--output-dir` 等参数
- [ ] **gloss_aware**:
  - [ ] pose_extractor.py 是否有 argparse?
  - [ ] filter_pose_pkls.py 是否有 argparse?
  - [ ] batch_norm_cosign_padding.py 是否有 argparse?
  - [ ] ssl_pretraining_*.py 已验证有 argparse ✓
  - [ ] build_prototypes_asl_clip_nob2b.py 已验证有 argparse ✓
- [ ] **推理服务**: MANIFEST.json 版本管理机制是否已准备

---

**结论**:
- ❌ **不需要大改**：子项目核心代码保持不动
- ✅ **小改动范围**：仅chatsign-accuracy的导出配置 + gloss_aware脚本的CLI接口（如需要）
- ✅ **充分兼容**：所有改动向后兼容，不影响原有功能

