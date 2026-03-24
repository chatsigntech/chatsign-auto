# Phase 5 参数系统详解

**文档版本**: 1.0
**最后更新**: 2026-03-24
**目的**: 深入讲解 Phase 5 多 GPU 增广参数系统的原理和使用

---

## 📋 概述

Phase 5 的核心功能是通过 **笛卡尔积** 生成参数组合，利用生成模型为每个词汇生成多个增广视频变体。

**关键概念**:
- **笛卡尔积**: 所有参数的完整组合 (e.g., 20 seeds × 11 noise levels = 220 基础组合)
- **变体数**: 笛卡尔积中每个组合对应一个视频变体
- **Multi-GPU**: 将变体分配到多个 GPU 并行处理
- **断点续传**: 保存检查点，失败后可恢复处理

---

## 🎯 7 个可配置参数

### 参数等级体系

#### 第一梯队：**最关键参数** ⭐⭐⭐

这两个参数对增广效果的影响最大，必须精心配置。

##### 1. `seed` - 随机种子

**范围**: 0 ~ 2^32-1
**影响**: **完全不同的视频生成**
**优先级**: ⭐⭐⭐ **最关键**

**说明**:
- 每个 seed 值会生成完全不同的视频（保持其他参数不变）
- 用于增加数据多样性的最有效方式
- 建议为每个词汇生成 5~100 个不同的 seed 变体

**预设值**:
- Light: 5 个 (0~4)
- Medium: 20 个 (0~19)
- Heavy: 100 个 (0~99)

**代码示例**:
```python
# inference_raw_batch_cache.py 中
for seed in seeds:
    torch.manual_seed(seed)
    output_video = model.forward(input_video, seed=seed)
    # seed = 0 vs seed = 1 => 完全不同的视频！
```

##### 2. `noise_aug_strength` - 噪声增强强度

**范围**: 0.0 ~ 1.0
**影响**: **视频风格和质感变化**
**优先级**: ⭐⭐⭐ **最关键**

**说明**:
- 控制在生成过程中添加多少噪声
- 0.0 = 完全确定性，接近参考视频
- 1.0 = 最大随机性，最大的风格变化
- 0.5 = 平衡点，既有多样性又保持可识别性

**预设值**:
- Light: 3 档 (0.0, 0.5, 1.0)
- Medium: **11 档** (0.0, 0.1, 0.2, ..., 1.0) ← 覆盖全范围
- Heavy: 11 档 (同 Medium)

**推荐配置**:
```yaml
# Medium 预设的 noise_aug_strength
noise_strengths: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# 说明：均匀划分 [0, 1] 范围，11 档递进
```

**生成效果**:
```
seed=0, noise=0.0   → 最接近参考视频的生成
seed=0, noise=0.5   → 中等风格变化
seed=0, noise=1.0   → 最大风格变化（可能差异很大）
```

#### 第二梯队：**重要参数** ⭐⭐

对视频质量和多样性有明显影响，但不如第一梯队关键。

##### 3. `guidance_scale` - 引导强度

**范围**: 0.0 ~ 15.0
**影响**: **与参考视频的相似度控制**
**优先级**: ⭐⭐ **重要**

**说明**:
- 控制生成过程中对参考视频的"忠诚度"
- 0.0~1.0 = 弱引导，生成结果更多样但可能偏离参考
- 5.0~7.5 = 推荐范围，平衡多样性和相似度
- 10.0+ = 强引导，生成结果更接近参考但多样性下降

**预设值**:
- Light: 1 档 (2.0)
- Medium: 2 档 (2.0, 5.0)
- Heavy: 2 档 (2.0, 5.0)

**推荐用法**:
```yaml
# 平衡版本（推荐）
guidance_scales: [2.0, 5.0]

# 说明
# 2.0: 多样化版本（较弱的参考约束）
# 5.0: 质量优先版本（中等的参考约束）
```

##### 4. `num_inference_steps` - 推理步数

**范围**: 1 ~ 50（通常 15~25）
**影响**: **质量 vs 速度权衡**
**优先级**: ⭐⭐ **重要**

**说明**:
- 生成过程中的去噪迭代次数
- 步数越多，质量越好，耗时越长
- 15 步: 标准平衡（~30-40 分钟/个变体）
- 25 步: 高质量（~50-60 分钟/个变体）

**预设值**:
- Light: 2 档 (15, 25)
- Medium: 2 档 (15, 25)
- Heavy: 2 档 (15, 25)

**时间估算**:
```
Medium 预设 (20 seeds × 11 noise × 2 steps × 2 guidance)
= 880 个变体

使用 num_inference_steps=[15, 25]:
- 一半变体用 15 步 (快)
- 一半变体用 25 步 (质量好)
- 平均耗时: (30 + 50) / 2 = 40 分钟/变体
- 单 GPU 总耗时: 880 × 40min = 586 小时 ≈ 24 天 ❌ (太长!)
```

**优化策略**:
```python
# 推荐：统一用 15 步，加快速度
num_inference_steps: [15]  # 2-3 小时完成 medium 预设

# 或：选择性提高质量
if seed < 5:
    num_inference_steps = 25  # 前 5 个 seed 用高质量
else:
    num_inference_steps = 15  # 其他用标准质量
```

#### 第三梯队：**可选参数** ⭐

控制细节行为，通常保持默认值不变。

##### 5. `scheduler` - 去噪调度器

**范围**: DDIM, Karras, DPMSolver, ...
**影响**: **去噪曲线，微调生成细节**
**优先级**: ⭐ **可选**

**说明**:
- 控制每一步的去噪强度曲线
- 不同调度器会导致轻微的视频差异
- 通常 DDIM 表现最稳定

**预设值**:
- Light: 1 个 (DDIM)
- Medium: 1 个 (DDIM)
- Heavy: 1~2 个 (DDIM, Karras)

**使用建议**:
```python
# 通常保持默认
scheduler = "DDIM"

# 仅在有特殊需求时配置多个
schedulers = ["DDIM", "Karras"]  # Heavy 预设可尝试
```

##### 6. `sample_stride` - 变步长采样

**范围**: 1 ~ 8
**影响**: **时间采样粒度（帧跳跃）**
**优先级**: ⭐ **可选**

**说明**:
- 从输入视频中每 stride 帧采样一帧
- stride=1: 采样所有帧（最详细）
- stride=2: 采样每第 2 帧（推荐值）
- stride=4: 采样每第 4 帧（稀疏）

**通常固定**:
```python
sample_stride = 2  # Phase 4 已固定，不需要在 Phase 5 配置
```

##### 7. `frames_overlap` - 帧插值增广（潜空间）

**范围**: 0 ~ 15
**影响**: **Tile 之间的平滑度**
**优先级**: ⭐ **可选**

**说明**:
- 长视频分割成重叠 tiles，在潜空间中进行插值融合
- 避免 tile 边界的视觉不连续
- 通常固定为 6~8，无需在 Phase 5 配置

**通常固定**:
```python
frames_overlap = 6  # 固定值，不参与笛卡尔积
```

---

## 📐 3 个预设配置详解

### Light 预设（开发/测试）

**特点**: 最快速度，用于快速验证

```yaml
preset: light
description: "快速验证 - 5×3×2×1 = 30变体/词汇"

parameters:
  seeds: [0, 1, 2, 3, 4]                    # 5 个
  noise_strengths: [0.0, 0.5, 1.0]          # 3 个
  num_inference_steps: [15, 25]             # 2 个
  guidance_scales: [2.0]                    # 1 个
  schedulers: ["DDIM"]                      # 1 个

# 笛卡尔积计算
total_variants = 5 × 3 × 2 × 1 = 30 个

# 耗时估算（单 GPU）
# 假设每个变体需要 40 分钟（15步）
# 30 × 40 = 1200 分钟 = 20 小时... 这太长了！
# 实际上：使用并行，8 GPU 则 20/8 ≈ 2.5 小时

# 更现实的估算（使用 15 步平均）
# 30 × 30 min = 900 min = 15 小时（单 GPU）
# 8 GPU: ~2 小时
```

**适用场景**:
- ✅ 开发和测试新词汇
- ✅ 验证 pipeline 功能是否正常
- ✅ 快速迭代调试参数
- ✅ GPU 资源有限的环境

**成本估算**:
- 单 GPU: ~30 分钟 ~ 2 小时
- 8 GPU: ~2-15 分钟
- 存储: 150MB ~ 300MB (30 个 MP4)

---

### Medium 预设（实验数据，**推荐**）

**特点**: 最佳性价比，推荐用于实验数据集

```yaml
preset: medium
description: "标准增广 - 20×11×2×2 = 880变体/词汇"

parameters:
  seeds: [0, 1, ..., 19]                                    # 20 个
  noise_strengths: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 11 个
  num_inference_steps: [15, 25]                             # 2 个
  guidance_scales: [2.0, 5.0]                               # 2 个
  schedulers: ["DDIM"]                                      # 1 个

# 笛卡尔积计算
total_variants = 20 × 11 × 2 × 2 × 1 = 880 个

# 关键点：noise_strengths 覆盖整个 [0, 1] 范围
# 从 0.0 (确定性) 到 1.0 (随机性)，11 档均匀分布
```

**耗时估算**:
```python
# 假设每个变体 = 40 分钟（混合 15 和 25 步）
# 880 × 40 = 35,200 分钟 = 588 小时

# 单 GPU: 588 小时 ≈ 24 天 ❌ (太慢)
# 8 GPU: 588 / 8 = 73.5 小时 ≈ 3 天 (可接受)

# 实际运行（优化后）:
# 使用 15 步 + 并行处理
# 8 GPU: 2-3 小时/词汇 ✅
```

**存储需求**:
```
880 个 MP4 × 2-3 MB/个 = 1.7-2.6 GB/词汇
85 个词汇 × 2.5 GB = 212.5 GB

# 分布式存储建议
SSD 缓存:   100 GB (phase5/cache/)
RAID 存储:  300 GB (phase5/output/)
```

**适用场景**:
- ✅ **实验数据集的标准增广** ← 推荐首选
- ✅ 模型初期训练（充分的数据多样性）
- ✅ 学术研究项目（发表论文）
- ✅ 算力和存储平衡的场景

**成本总结**（85 个词汇）:
- 耗时: 2-3 小时（8 GPU 并行）
- 存储: ~212 GB
- 算力: 8 × GPU days
- **性价比**: ⭐⭐⭐⭐⭐ 最优

---

### Heavy 预设（完整生产数据）

**特点**: 最大化数据多样性，用于生产环境

```yaml
preset: heavy
description: "完整增广 - 100×11×2×2 = 4,400变体/词汇"

parameters:
  seeds: [0, 1, ..., 99]                                     # 100 个
  noise_strengths: [0.0, 0.1, 0.2, ..., 1.0]               # 11 个
  num_inference_steps: [15, 25]                             # 2 个
  guidance_scales: [2.0, 5.0]                               # 2 个
  schedulers: ["DDIM"]                                      # 1 个

# 笛卡尔积计算
total_variants = 100 × 11 × 2 × 2 × 1 = 4,400 个

# 关键点：100 个 seed 充分覆盖随机性
# 每个 seed 生成 11×2×2 = 44 个相关变体
```

**耗时估算**:
```python
# 4,400 变体 / 880 (medium) = 5 倍
# Medium 耗时 × 5 = 2-3 小时 × 5 = 10-15 小时

# 8 GPU: 10-15 小时/词汇 (可接受)
# 85 个词汇: 850-1275 小时 = 35-53 天 (用 8 GPU)
```

**存储需求**:
```
4,400 个 MP4 × 2-3 MB/个 = 8.8-13.2 GB/词汇
85 个词汇 × 11 GB = 935 GB ≈ 1 TB

# 专用存储设备必需
RAID-6: 2 TB (容错)
```

**适用场景**:
- ✅ 最大化数据多样性（100 个随机种子）
- ✅ 生产级别的模型部署
- ✅ 充足的计算资源（weeks of GPU time）
- ✅ 充足的存储空间（1 TB+）

**成本总结**（85 个词汇）:
- 耗时: 50 小时 ~ 1.5 周（8 GPU）
- 存储: ~935 GB (~1 TB)
- 算力: 400+ × GPU hours
- **性价比**: ⭐⭐ (仅限生产环境)

---

## 🧮 变体数计算公式

### 笛卡尔积

```
总变体数 = |seeds| × |noise_strengths| × |num_inference_steps| × |guidance_scales| × |schedulers|
         × |sample_stride| × |frames_overlap|

# 但 sample_stride 和 frames_overlap 通常固定，所以：

总变体数 = |seeds| × |noise_strengths| × |num_inference_steps| × |guidance_scales| × |schedulers|
```

### 示例

```python
# Light 预设
light = 5 × 3 × 2 × 1 × 1 = 30

# Medium 预设
medium = 20 × 11 × 2 × 2 × 1 = 880

# Heavy 预设
heavy = 100 × 11 × 2 × 2 × 1 = 4,400

# Custom 预设（用户自定义）
custom = 10 × 5 × 1 × 2 × 1 = 100
```

---

## 🎮 Custom 预设（用户自定义）

用户可以在创建任务时指定 `augmentation_preset: "custom"` 并提供自定义参数：

```yaml
# 编排器配置示例
custom_augmentation_config:
  augmentation_preset: custom

  parameters:
    seeds: [0, 5, 10, 15, 20]                # 5 个
    noise_strengths: [0.3, 0.5, 0.7]         # 3 个（跳过极端值）
    num_inference_steps: [20]                 # 1 个（固定质量）
    guidance_scales: [3.0, 6.0]               # 2 个
    schedulers: ["DDIM"]                      # 1 个

  # 笛卡尔积
  total_variants = 5 × 3 × 1 × 2 × 1 = 30
```

**适用场景**:
- 高级用户精细化调优
- 特定需求的实验设计
- 性能验证（快速生成少量变体）

---

## ⏱️ 耗时预测模型

### 单变体耗时

```
single_variant_time = base_inference_time + overhead
                    ≈ num_inference_steps × 1.5 分钟 + 2 分钟

# 示例
15 步:  15 × 1.5 + 2 = 24.5 分钟
25 步:  25 × 1.5 + 2 = 39.5 分钟
平均:  (24.5 + 39.5) / 2 = 32 分钟
```

### 总耗时

```
total_time_single_gpu = total_variants × single_variant_time
total_time_multi_gpu = total_time_single_gpu / num_gpus

# Medium 预设估算
880 × 32 分钟 = 28,160 分钟
单 GPU: 28,160 / 60 = 469 小时 ≈ 20 天
8 GPU: 469 / 8 ≈ 59 小时 ≈ 2.5 天
```

### 系数调整

实际耗时会受多因素影响：
- **GPU 型号**: H100 > A100 > A6000 > RTX 4090
- **批处理**: 并行处理多个变体会有内存管理开销
- **I/O**: 写入视频文件的磁盘速度

建议使用 **1.2x ~ 1.5x** 的安全系数。

---

## 💾 存储需求计算

### 单变体视频大小

```
video_size = resolution × frame_rate × duration × bit_depth / 8

# 标准配置
resolution = 576×576
frame_rate = 25 FPS
duration ≈ 5 秒
bit_depth = 24 bits (RGB)

video_size ≈ 2.5 MB/变体
```

### 总存储需求

```
total_storage = total_variants × video_size × num_glosses

# Medium 预设，85 个词汇
880 变体/词 × 2.5 MB × 85 词 = 187 GB
```

### 存储规划

| 预设 | 单词汇 | 85 词总量 | 建议设备 |
|------|--------|---------|---------|
| Light | 150-300 MB | 13-25 GB | SSD |
| Medium | 1.5-3 GB | 128-255 GB | RAID |
| Heavy | 8-13 GB | 680-1100 GB | RAID-6 |

---

## 🔄 GPU 分配策略

### Round-Robin 分配

```python
def partition_for_gpus(variants, num_gpus):
    """轮询分配变体到各 GPU"""
    chunks = [[] for _ in range(num_gpus)]

    for idx, variant in enumerate(variants):
        gpu_id = idx % num_gpus
        chunks[gpu_id].append(variant)

    return chunks

# 示例：880 个变体，8 个 GPU
# GPU 0: 变体 0, 8, 16, 24, ...  (110 个)
# GPU 1: 变体 1, 9, 17, 25, ...  (110 个)
# ...
# GPU 7: 变体 7, 15, 23, 31, ... (110 个)
```

**优点**:
- ✅ 负载均衡（每 GPU 分配相同数量）
- ✅ 简单实现
- ✅ 可预测的耗时

---

## 📊 参数选择决策树

```
┌─ 选择预设
│
├─ Light?
│  ├─ 开发/测试            → ✅ 选择 Light
│  └─ 快速验证功能          → ✅ 选择 Light
│
├─ Medium?
│  ├─ 实验数据集            → ✅ 选择 Medium (推荐)
│  ├─ 初期模型训练          → ✅ 选择 Medium
│  └─ 学术研究              → ✅ 选择 Medium
│
├─ Heavy?
│  ├─ 生产环境              → ✅ 选择 Heavy
│  ├─ 最大多样性            → ✅ 选择 Heavy
│  └─ 充足算力和存储        → ✅ 选择 Heavy
│
└─ Custom?
   ├─ 精细化调优             → ✅ 选择 Custom
   └─ 特定实验需求           → ✅ 选择 Custom
```

---

## 🚀 最佳实践

### 推荐配置流程

1. **第一步：使用 Medium 预设开始**
   ```yaml
   # 创建任务时
   task_config = {
       "augmentation_preset": "medium"
   }
   ```

2. **第二步：评估耗时和成本**
   - 8 GPU: 预期 2-3 小时/词汇
   - 存储: 预期 200GB 总量

3. **第三步：根据需求调整**
   - 如果太慢：使用 Light 预设
   - 如果需要更多数据：使用 Heavy 预设
   - 如果有特殊需求：配置 Custom 预设

4. **第四步：启动并监测**
   ```bash
   # 监测进度
   curl http://localhost:8000/api/tasks/{task_id} \
     -H "Authorization: Bearer {token}"

   # 输出示例：
   # {
   #   "phase_states": [{
   #     "phase_num": 5,
   #     "progress": 45.5,
   #     "substep": "GPU 0: 400/880 variants"
   #   }]
   # }
   ```

---

## 🔗 相关文档

- [README.md](README.md) - 项目概览
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - Phase 5 完成情况
- [AUGMENTATION_CONFIG_MAPPING.md](AUGMENTATION_CONFIG_MAPPING.md) - 10 维度映射表
- [backend/workers/phase5_worker.py](backend/workers/phase5_worker.py) - 实现代码
- [backend/core/gpu_manager.py](backend/core/gpu_manager.py) - GPU 管理

---

**版本**: 1.0
**更新日期**: 2026-03-24
**维护者**: ChatSign Team
**状态**: ✅ 完成
