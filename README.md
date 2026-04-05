# ChatSign Orchestrator

**编排管理系统** - 统一管理 6 阶段视频处理和模型训练工作流

## 📋 项目概述

ChatSign Orchestrator 是一个完整的编排管理系统，负责管理手语视频数据处理、增广和模型训练的整个 6 阶段流程。系统支持多任务并发、GPU 并行处理、断点续传、以及实时进度监测。

**核心功能**：
- **Phase 1-3**: 视频采集、预处理、标注整理
- **Phase 4**: 视频标准化处理（帧提取、去重、姿态过滤、尺度标准化）
- **Phase 5**: 数据增广（多 GPU 并行生成增广视频）
- **Phase 6**: 模型训练和原型构建

---

## 📚 文档导航

| 文档 | 位置 | 阅读场景 |
|------|------|--------|
| **PROJECT_STATUS.md** | 根目录 | 了解 6 个阶段的功能完成情况 |
| **ORCHESTRATOR_INTEGRATION_MAP.md** | 根目录 | 理解与 5 个子项目的集成关系 |
| **PIPELINE_STAGE_INDEPENDENCE.md** | 根目录 | 学习各 Stage 接口规范和数据流 |
| **PHASE3_INTEGRATION.md** | 根目录 | 了解 Phase 3 与 chatsign-accuracy 的共享目录规范 |
| **PHASE5_PARAMETER_SYSTEM.md** | 根目录 | 深入了解多 GPU 增广参数系统 |
| **AUGMENTATION_CONFIG_MAPPING.md** | 根目录 | 查阅 10 个增广维度与配置参数的映射表 |
| **DEPLOYMENT_GUIDE.md** | deployment/ | 生产环境部署指南 |

---

## 🚀 快速开始

### 环境准备

```bash
# 克隆项目
git clone <repo-url> chatsign-auto
cd chatsign-auto

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp backend/.env.example .env
# 编辑 .env 文件：
#   - DATABASE_URL：数据库路径
#   - SECRET_KEY：JWT 签名密钥（改为强随机值）
#   - DEFAULT_ADMIN_PASSWORD：admin 初始密码
#   - 外部项目路径：PSEUDO_GLOSS_PATH, UNISIGN_PATH, GLOSS_AWARE_PATH
```

### 启动服务

```bash
# 开发环境
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 访问
# - WebUI: http://localhost:8000
# - API 文档: http://localhost:8000/docs
# - 数据库管理: http://localhost:8000/admin
```

### 创建任务

```bash
# 通过 API 创建新任务
curl -X POST http://localhost:8000/api/tasks \
  -H "Authorization: Bearer <jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Sign Language Dataset",
    "config": {
      "augmentation_preset": "medium"
    }
  }'

# 查看任务进度
curl http://localhost:8000/api/tasks/<task_id> \
  -H "Authorization: Bearer <jwt_token>"
```

---

## 📁 数据集

项目相关的视频数据集存储在数据盘 `/mnt/data/chatsign-auto-videos/`，共约 63GB：

| 数据集 | 路径 | 文件数 | 大小 | 说明 |
|--------|------|--------|------|------|
| **ASL-final-27K** | `ASL-final-27K-202603/` | 27,079 视频 + gloss.csv | 21G | ASL 最终数据集（2026-03） |
| **How2Sign** | `how2sign_data/` | 31,048 视频 | 31G | How2Sign 公开数据集 |
| **OpenSL** | `opensl_data/` | 50,008 视频 | 11G | OpenSL 公开数据集 |

---

## 🔄 工作流概览

### 6 个处理阶段

| 阶段 | 名称 | 输入 | 输出 | 时间 |
|------|------|------|------|------|
| **1** | 视频采集 | 手语视频 | 词汇 JSON | 人工 |
| **2** | 格式标准化 | 原始视频 | 标准化视频 + 词汇表 | 自动 |
| **3** | 标注整理 | 标准化视频 | gloss_to_video_database.json | 自动 |
| **4** | 视频预处理 | 原始视频 | 标准尺寸清晰视频 | 30min-2h |
| **5** | 数据增广 | 清晰视频 | 880-4400 个变体视频/词汇 | 2-8h (单 GPU) |
| **6** | 模型训练 | 增广视频 + 标注 | 训练好的模型 + 原型嵌入 | 4-12h |

### 数据流

```
Phase 1 (采集)
    ↓ glosses.json
Phase 2 (标准化)
    ↓ standardized_videos/
Phase 3 (标注)
    ↓ gloss_to_video_database.json
Phase 4 (预处理)
    ↓ cleaned_videos/
Phase 5 (增广, Multi-GPU)
    ↓ variant_*.mp4 (Light: 30, Medium: 880, Heavy: 4400 个/词汇)
Phase 6 (训练, Distributed)
    ↓ best.pth + prototypes.pt
```

---

## ⚙️ 系统架构

### 核心模块

```
backend/
├── main.py              # FastAPI 应用入口
├── config.py            # 配置管理（环境变量、路径、预设）
├── database.py          # SQLite 数据库会话
├── models/              # SQLModel 数据模型
│   ├── task.py         # 任务模型
│   ├── phase.py        # Phase 状态模型
│   └── user.py         # 用户认证模型
├── api/                 # FastAPI 路由
│   ├── tasks.py        # 任务管理 API
│   ├── phases.py       # Phase 状态 API
│   └── auth.py         # 认证 API
├── core/                # 业务逻辑核心
│   ├── file_manager.py           # 文件路径管理
│   ├── phase_state_manager.py    # Phase 状态机
│   ├── subprocess_runner.py       # 子进程执行
│   ├── gpu_manager.py            # GPU 管理和调度
│   └── file_lock.py              # 文件锁（Phase 3 同步）
└── workers/             # Phase 处理器
    ├── phase4_worker.py          # 视频预处理
    ├── phase5_worker.py          # 数据增广
    └── phase6_worker.py          # 模型训练
```

### 关键特性

- **并发安全**: 使用 fcntl 文件锁保护共享目录访问
- **GPU 调度**: Round-robin 负载均衡，支持暂停/恢复
- **断点续传**: 检查点机制跳过已完成任务
- **实时监测**: WebSocket 进度广播
- **错误隔离**: 单个 gloss 失败不影响其他 gloss 处理

---

## 📊 配置系统

### 数据增广预设

编排器提供 3 种增广预设，通过 `augmentation_preset` 参数选择：

| 预设 | 变体数 | 用途 | 单 GPU 时间 |
|------|--------|------|-----------|
| **light** | 30/词汇 | 开发测试 | ~30 分钟 |
| **medium** | 880/词汇 | 实验数据（推荐） | ~2-3 小时 |
| **heavy** | 4400/词汇 | 生产数据 | ~6-8 小时 |

详见 [AUGMENTATION_CONFIG_MAPPING.md](AUGMENTATION_CONFIG_MAPPING.md)

### 训练参数

```python
# Phase 6 默认值
batch_size = 64      # 批处理大小
epochs = 100         # 训练轮数
hidden_dim = 256     # 隐层维度
```

---

## 🔐 身份验证

系统使用 JWT 令牌认证，默认 admin 账户在启动时自动创建：

```bash
# 登录
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# 响应包含 JWT token
{"access_token": "eyJ0eX...", "token_type": "bearer"}

# 使用 token 访问保护的 API
curl http://localhost:8000/api/tasks \
  -H "Authorization: Bearer eyJ0eX..."
```

**生产环境必须**：
- 修改 `.env` 中的 `DEFAULT_ADMIN_PASSWORD`
- 修改 `.env` 中的 `SECRET_KEY` 为强随机值

---

## 🛠️ 开发指南

### 添加新 Phase

1. 在 `backend/models/phase.py` 中添加 Phase 定义
2. 在 `backend/workers/` 中创建 `phase{N}_worker.py`
3. 在 `backend/main.py` 中注册 Phase 处理器
4. 实现 Phase 接口：`async def process_task(task_id, session) -> bool`

### 处理 Phase 失败

```python
# Phase 状态机
PENDING → RUNNING → COMPLETED/FAILED/PAUSED

# 标记失败
await PhaseStateManager.mark_failed(phase, session, error_msg)

# 标记暂停（仅 Phase 5 支持）
await PhaseStateManager.mark_paused(phase, session)

# 恢复执行
await PhaseStateManager.mark_running(phase, session)
```

---

## 🐛 故障排除

### Phase 4 或 5 失败

检查：
1. 磁盘空间（Phase 5 可能需要 100GB+）
2. GPU 可用性（`nvidia-smi` 检查）
3. 外部脚本路径（`.env` 中的 `UNISIGN_PATH`, `GLOSS_AWARE_PATH`）

### 文件锁超时

如果看到 "Failed to acquire lock" 错误：
1. 检查是否有其他任务在访问同一 Phase 3 目录
2. 查看 `/data/chatsign/shared/{task_id}/.phase3.lock`
3. 任务完成或失败后锁会自动释放

### GPU 内存不足

减少 batch_size 或使用 light 预设：
```bash
curl -X POST http://localhost:8000/api/tasks \
  -d '{"augmentation_preset": "light"}'
```

---

## 📖 相关资源

- [PHASE3_INTEGRATION.md](PHASE3_INTEGRATION.md) - 了解 Phase 3 与 chatsign-accuracy 的共享机制
- [ORCHESTRATOR_INTEGRATION_MAP.md](ORCHESTRATOR_INTEGRATION_MAP.md) - 5 个子项目如何集成
- [deployment/DEPLOYMENT_GUIDE.md](deployment/DEPLOYMENT_GUIDE.md) - 生产部署步骤
- Git Commits - 查看 `git log` 了解最近的修复和改进

---

**最后更新**: 2026-03-24
**项目状态**: ✅ Phase 1-6 全部实现完成
**维护者**: ChatSign Team
