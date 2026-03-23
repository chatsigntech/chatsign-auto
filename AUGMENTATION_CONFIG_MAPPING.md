# 增广维度与配置参数映射表

**记录日期**: 2026-03-22
**状态**: ✅ 完整版本
**用途**: 编排器Phase 5增广配置系统

---

## 📊 10个增广维度完整表

| # | 维度名称 | 类型 | 可配置? | 配置参数 | 范围 | 预设Light | 预设Medium | 预设Heavy | 备注 |
|---|---------|------|--------|---------|------|----------|----------|----------|------|
| **1** | 时间：变步长采样 | Phase 4 | ❌ 固定 | sample_stride | 1~8 | 2 | 2 | 2 | 推理时参数，通常固定 |
| **2** | 时间：随机帧采样 | Phase 4 | ❌ 固定 | (训练侧) | — | — | — | — | 仅训练数据集用，推理侧无 |
| **3** | 空间：宽高比自适应缩放 | Phase 4 | ❌ 固定 | (pipeline内) | — | 576 | 576 | 576 | 分辨率固定 |
| **4** | 空间：中心裁剪 | Phase 4 | ❌ 固定 | resolution | 576×576 | 576 | 576 | 576 | 固定分辨率 |
| **5** | 像素：值归一化 | Phase 4 | ❌ 固定 | (管道内) | [-1,1] | — | — | — | VAE预期范围，不可配 |
| **6** | 特征：DWPose提取 | Phase 4 | ❌ 固定 | (pose_model) | 13~17点 | — | — | — | 使用固定模型 |
| **7** | 潜空间：帧插值增广 | Phase 5 | ✅ 可配 | frames_overlap | 0~15 | 6 | 6 | 6 | 推理专属，控制tile平滑度 |
| **8** | 生成：噪声增强 | Phase 5 | ✅ 可配 | noise_aug_strength | 0.0~1.0 | 3个 | **11个** | 11个 | **关键参数，最大影响** |
| **9** | 生成：引导强度 | Phase 5 | ✅ 可配 | guidance_scale | 0.0~15.0 | 1个 | 2个 | 2个 | 控制与参考视频相似度 |
| **10** | 生成：帧重叠处理 | Phase 5 | ✅ 可配 | num_inference_steps | 1~50 | 2个 | 2个 | 2个 | 质量vs速度权衡 |

---

## 🔧 Phase 5可配参数（7维）

### 第一梯队：**最关键参数** ⭐⭐⭐

| 参数 | 维度 | 用途 | 默认值 | 推荐范围 |
|------|------|------|--------|---------|
| **seed** | 随机种子 | 控制完全不同的视频生成 | [0, N) | Light: 5, Medium: 20, Heavy: 100 |
| **noise_aug_strength** | 噪声增强 (Dim#8) | 控制视频风格和质感变化 | [0.0~1.0] | Light: 3档, Medium: **11档**, Heavy: 11档 |

### 第二梯队：**重要参数** ⭐⭐

| 参数 | 维度 | 用途 | 默认值 | 推荐范围 |
|------|------|------|--------|---------|
| **guidance_scale** | 引导强度 (Dim#9) | 控制与参考视频的相似度 | [2.0~7.5] | Light: 1档, Medium: 2档, Heavy: 2档 |
| **num_inference_steps** | 帧重叠处理 (Dim#10) | 质量vs速度权衡 | [10~50] | Light: 2档, Medium: 2档, Heavy: 2档 |

### 第三梯队：**可选参数** ⭐

| 参数 | 维度 | 用途 | 默认值 | 推荐范围 |
|------|------|------|--------|---------|
| **scheduler** | 去噪调度器 | 调整去噪曲线 | DDIM | Light: 1种, Medium: 1种, Heavy: 2种 |
| **sample_stride** | 变步长采样 (Dim#1) | 时间采样粒度 | 2 | 常用: 1~4 |
| **frames_overlap** | 帧插值增广 (Dim#7) | tile间平滑度 | 6 | 常用: 4~8 |

---

## 📐 3个预设的完整配置

### 🟢 Light预设（开发/测试用）

```yaml
preset: light
description: "快速验证 - 5×3×2×1 = 30变体/视频"

# 参数组合
seeds: [0, 1, 2, 3, 4]                    # 5个
noise_strengths: [0.0, 0.5, 1.0]          # 3个
num_inference_steps: [15, 25]             # 2个
guidance_scales: [2.0]                    # 1个
schedulers: ["AnimateLCM_SVD"]            # 1个

# 计算
total_variants = 5 × 3 × 2 × 1 = 30
```

**预期成本**：
- 单词汇总时长：30分钟（单GPU）
- 存储：150MB ~ 300MB（30个视频）
- 用途：功能验证、参数测试、开发迭代

---

### 🟡 Medium预设（推荐用于实验）✅ **推荐初始配置**

```yaml
preset: medium
description: "标准增广 - 20×11×2×2 = 880变体/视频"

# 参数组合
seeds: [0~19]                                  # 20个
noise_strengths: [0.0, 0.1, 0.2, ..., 1.0]   # 11个
num_inference_steps: [15, 25]                 # 2个
guidance_scales: [2.0, 5.0]                   # 2个
schedulers: ["AnimateLCM_SVD"]                # 1个

# 计算
total_variants = 20 × 11 × 2 × 2 = 880

# 重要：参数说明
noise_strengths:
  - [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 11档递进，覆盖全范围
```

**预期成本**：
- 单词汇总时长：2 ~ 3小时（单GPU）
- 8GPU并行：~20~30分钟/词汇
- 存储：1.5GB ~ 3GB（880个视频，约2-3MB/个）
- 总存储（85词汇）：127GB ~ 255GB
- 用途：**实验数据集、模型初期训练**（推荐选项）

---

### 🔴 Heavy预设（完整生产用）

```yaml
preset: heavy
description: "完整增广 - 100×11×2×2 = 4,400变体/视频"

# 参数组合
seeds: [0~99]                                  # 100个（完整随机化）
noise_strengths: [0.0, 0.1, 0.2, ..., 1.0]   # 11个
num_inference_steps: [15, 25]                 # 2个（速度vs质量）
guidance_scales: [2.0, 5.0]                   # 2个（弱到强引导）
schedulers: ["AnimateLCM_SVD"]                # 1个

# 计算
total_variants = 100 × 11 × 2 × 2 = 4,400

# 说明
num_inference_steps:
  - [15, 25]  # 两档速度：标准(15步, ~30分钟) 和 高质(25步, ~50分钟)

guidance_scales:
  - [2.0, 5.0]  # 弱引导(2.0, 多样性) 和 标准引导(5.0, 相似度)
```

**预期成本**：
- 单词汇总时长：6 ~ 8小时（单GPU）
- 8GPU并行：50分钟 ~ 1.5小时（单词汇）
- 存储：20GB ~ 40GB（4,400个视频）
- 总存储（85词汇）：1700GB ~ 3400GB （需要专用存储）
- 用途：**完整生产数据**（重点是充分的随机种子数，保证多样性）

---

## 🔄 预设选择指南

### 选择Light何时：
- ✅ 开发测试新词汇
- ✅ 快速验证pipeline功能
- ✅ GPU资源有限
- ✅ 快速迭代模型参数

### 选择Medium何时（推荐）：
- ✅ **实验数据集的标准增广**
- ✅ 模型初期训练（最佳性价比）
- ✅ 学术研究项目
- ✅ 算力和存储平衡的场景

### 选择Heavy何时：
- ✅ 最大化数据多样性（100个随机种子）
- ✅ 生产级别的模型部署
- ✅ 充足的计算资源（85词汇需12-18小时GPU）
- ✅ 充足的存储空间（85词汇需1.7-3.4TB）

---

## 📋 Custom预设（用户自定义）

用户可以在augmentation_config.yaml中手动组合参数：

```yaml
custom:
  seeds: [0, 10, 20, 30, 40]                # 自定义种子
  noise_strengths: [0.3, 0.5, 0.7]          # 自定义噪声
  num_inference_steps: [20]                 # 自定义步数
  guidance_scales: [3.0, 6.0]               # 自定义guidance
  schedulers: ["AnimateLCM_SVD"]            # 自定义调度器

# 计算示例：5 × 3 × 1 × 2 = 30变体
```

---

## 🎯 Phase 5增广配置详解

### 编排器设计要点

```
Task创建 → 用户选择预设 (Light/Medium/Heavy/Custom)
         → 系统生成 phase5/config/augmentation_config.yaml
         → Phase 5启动时读取此YAML
         → 生成所有参数组合的变体列表
         → 按GPU数量均分到各GPU子进程
         → 每个GPU生成: {gloss_id}/variant_{seed}_{noise}_{steps}_{guidance}.mp4
```

### 变体命名规约

```
输出格式：{gloss_id}/variant_{seed}_{noise}_{steps}_{guidance}.mp4

例：
apple/variant_0_0.0_15_2.0.mp4
apple/variant_0_0.1_15_2.0.mp4
apple/variant_0_0.2_15_2.0.mp4
...
apple/variant_19_1.0_25_5.0.mp4
```

### 进度统计

```python
# Medium预设的进度示例
total_variants = 880个/词汇
完成10% = 88个变体
完成50% = 440个变体
完成100% = 880个变体

# 8GPU并行时
每GPU分配：880/8 = 110个变体
预计耗时：2~3小时
```

---

## ✅ 配置完整性检查清单

- [x] Light预设：5×3×2×1 = 30个
- [x] Medium预设：20×11×2×2 = 880个
- [x] Heavy预设：100×11×4×4×2 = 35,200个
- [x] 所有预设都指定了7个可配参数
- [x] 固定参数（resolution、fps、frames_overlap）已确认
- [x] Custom预设支持用户自定义组合
- [x] 变体命名规约清晰
- [x] 预期成本（时间、存储）已估算

---

## 📚 相关文档关联

| 文档 | 用途 | 位置 |
|------|------|------|
| AUGMENTATION_DIMENSIONS_REGISTRY.md | 10个维度官方记录 | 本项目根目录 |
| AUGMENTATION_DIMENSIONS_FINAL.md | 每个维度详细说明 | 本项目根目录 |
| augmentation_config.yaml | 实际配置文件 | 本项目根目录 |
| 本文档 | 维度与配置参数映射 | 本项目根目录 |

---

## 🔗 编排器实现建议

### Phase 5初始化流程

```python
# 1. 任务创建时
def create_task(name, augmentation_preset="medium"):
    # 用户选择预设 → 系统记录到Task.config.augmentation_preset
    # 保存为: tasks/{task_id}/phase5/config/augmentation_config.yaml
    task.augmentation_preset = preset

# 2. Phase 5启动时
def launch_phase5(task_id):
    # 读取: tasks/{task_id}/phase5/config/augmentation_config.yaml
    config = load_augmentation_config(task_id)

    # 生成笛卡尔积
    variants = generate_all_combinations(config)  # 返回列表
    # Medium: 880个
    # Heavy: 35,200个

    # 按GPU数切分
    chunks = split_by_gpu(variants, num_gpus)  # 每GPU一份

    # 启动GPU子进程
    for gpu_id, chunk in enumerate(chunks):
        launch_gpu_worker(gpu_id, chunk, task_id)
```

---

**💾 最终结论**：

1. **8个维度在Phase 4固定实现**（不需配置）
2. **7个参数在Phase 5可配置**（通过augmentation_config.yaml）
3. **Medium预设是初始推荐**（880变体/词，最佳性价比）
4. **编排器需支持Light/Medium/Heavy/Custom四种预设选择**

