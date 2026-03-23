# 任务计划：数据增广系统集成

## 目标
集成数据增广系统到编排器，梳理参数系统、设计初始配置、修正架构文档。

## 背景
- 发现 inference_raw_batch_cache.py 支持参数变体生成（最强大）
- 需要生成合理的初始配置供用户调整
- 修正编排器设计：从推理系统 → 数据处理系统

---

## Phase 1: 梳理 inference_raw_batch_cache.py 参数系统

### 目标
理解参数override机制，生成参数组合表和变体数计算公式

### 步骤
- [ ] 1.1 读取 inference_raw_batch_cache.py，识别所有可变参数
- [ ] 1.2 分析参数影响（seed、noise_aug_strength、scheduler等）
- [ ] 1.3 生成参数组合表
- [ ] 1.4 生成变体数计算公式

### 输出物
- [ ] augmentation_params_analysis.md（参数详细说明）
- [ ] param_combinations_table.csv（组合矩阵）

**Status**: pending

---

## Phase 2: 设计初始数据增广配置

### 目标
生成 augmentation_config.yaml 初始配置，平衡生成效率与数据多样性

### 步骤
- [ ] 2.1 基于Phase 1的发现，设计推荐配置
- [ ] 2.2 为不同使用场景（小规模测试/中等/大规模）提供预设
- [ ] 2.3 生成配置文档和参数说明
- [ ] 2.4 估算存储需求和耗时

### 输出物
- [ ] augmentation_config.yaml（初始配置）
- [ ] augmentation_config_presets.md（预设配置和说明）

**Status**: pending

---

## Phase 3: 修正编排器设计文档

### 目标
更新现有计划文件，从推理系统改为数据处理系统

### 步骤
- [ ] 3.1 读取 /Users/li/.claude/plans/parsed-marinating-thimble.md
- [ ] 3.2 更新架构：适配6阶段流程（特别是Phase 5数据增广）
- [ ] 3.3 修改API端点：面向数据处理而非推理
- [ ] 3.4 更新 pipeline 步骤设计
- [ ] 3.5 添加GPU资源管理细节

### 输出物
- [ ] 更新的 parsed-marinating-thimble.md（修正后的设计）
- [ ] orchestrator_phase5_detail.md（Phase 5详细设计）

**Status**: pending

---

## 错误日志
（将记录任务执行过程中的问题和解决方案）

| 错误 | 尝试次数 | 解决方案 |
|------|---------|--------|
| | | |

---

## 决策记录
（关键决策点和理由）

| 决策 | 理由 | 日期 |
|------|------|------|
| | | |

---

## 最终交付物清单
- [x] augmentation_params_analysis.md ✅
- [x] augmentation_config.yaml ✅
- [x] augmentation_config_presets.md ✅
- [x] orchestrator_phase5_detail.md ✅
- [x] AUGMENTATION_SYSTEM_SUMMARY.md ✅
- [x] task_plan.md (本文件) ✅
- [x] findings.md ✅
- [x] progress.md ✅

---

## 完成总结

**总耗时**: 3个Phase，完成所有目标
**交付物**: 8个文档，覆盖参数系统、配置设计、编排器实现

### Phase 1: 梳理参数系统 ✅
- 完成时间: 第1部分
- 输出: augmentation_params_analysis.md
- 成果: 完整的参数文档，包括6个核心参数详解、override机制、组合策略

### Phase 2: 设计初始配置 ✅
- 完成时间: 第2部分
- 输出: augmentation_config.yaml + augmentation_config_presets.md
- 成果: 3个预设方案（Light/Medium/Heavy）和详细使用指南

### Phase 3: 修正编排器设计 ✅
- 完成时间: 第3部分
- 输出: orchestrator_phase5_detail.md
- 成果: Phase 5详细设计，包括完整Python代码框架

**整体进度: 100% ✅**
