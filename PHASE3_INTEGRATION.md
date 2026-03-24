# Phase 3 集成规范：与 chatsign-accuracy 的共享目录

**文档版本**: 1.0
**最后更新**: 2026-03-24
**状态**: ✅ 完成

---

## 📋 概述

Phase 3 是编排器与 **chatsign-accuracy** 项目之间的集成点。Phase 3 负责：
1. 读取 Phase 2 生成的标准化视频
2. 基于词汇列表构建数据库
3. 生成 `gloss_to_video_database.json`
4. 为 Phase 4 的视频预处理提供清晰的词汇-视频映射

此文档定义了两个系统间的共享目录规范、文件格式和并发访问规则。

---

## 🏗️ 目录结构

### 编排器侧（chatsign-auto）

```
chatsign-auto/
├── backend/
│   ├── workers/
│   │   └── phase3_worker.py        # Phase 3 处理器
│   └── core/
│       └── file_lock.py            # 共享目录访问控制
└── data/
    └── chatsign/
        └── shared/                 # ← 共享根目录
            └── {task_id}/
                ├── gloss_to_video_database.json  # Phase 3 输出
                └── .phase3.lock                  # 文件锁
```

### chatsign-accuracy 侧（采集网站）

```
chatsign-accuracy/
├── uploads/                        # 用户上传的视频
└── tasks/
    └── {task_id}/
        ├── standardized/           # Phase 2 输出（标准化视频）
        └── glosses.json            # Phase 2 输出（词汇表）
```

---

## 📊 共享数据格式

### 1. glosses.json（Phase 2 输出，Phase 3 输入）

**来源**: Phase 2 处理结果
**位置**: `/data/chatsign/shared/{task_id}/glosses.json`
**用途**: 作为 Phase 3 构建词汇-视频映射的输入

**格式**:
```json
{
  "task_id": "task_12345",
  "created_at": "2026-03-20T10:30:00Z",
  "glosses": [
    {
      "gloss": "apple",
      "video_urls": [
        "https://chatsign-accuracy.local/api/videos/apple-1",
        "https://chatsign-accuracy.local/api/videos/apple-2"
      ],
      "count": 2,
      "duration": 8.5
    },
    {
      "gloss": "ball",
      "video_urls": [
        "https://chatsign-accuracy.local/api/videos/ball-1"
      ],
      "count": 1,
      "duration": 4.2
    }
  ]
}
```

**字段说明**:
- `gloss` (string): 词汇文本（唯一标识符）
- `video_urls` (array): 该词汇的所有视频 URL
- `count` (int): 该词汇的视频个数
- `duration` (float): 总时长（秒）

### 2. gloss_to_video_database.json（Phase 3 输出，Phase 4 输入）

**生成者**: Phase 3 处理器
**位置**: `/data/chatsign/shared/{task_id}/gloss_to_video_database.json`
**用途**: Phase 4 使用此文件了解每个词汇的视频位置

**格式**:
```json
{
  "task_id": "task_12345",
  "created_at": "2026-03-20T14:45:00Z",
  "glosses": {
    "apple": {
      "video_paths": [
        "path/to/phase2/standardized/task_12345/apple/video_1.mp4",
        "path/to/phase2/standardized/task_12345/apple/video_2.mp4"
      ],
      "count": 2,
      "total_duration": 8.5
    },
    "ball": {
      "video_paths": [
        "path/to/phase2/standardized/task_12345/ball/video_1.mp4"
      ],
      "count": 1,
      "total_duration": 4.2
    }
  }
}
```

**字段说明**:
- `task_id`: 任务 ID
- `created_at`: 创建时间戳
- `glosses`: 词汇映射字典
  - `{gloss_id}`: 词汇 ID（键）
    - `video_paths`: 本地文件路径数组
    - `count`: 视频个数
    - `total_duration`: 总时长

---

## 🔄 数据流

### Phase 3 处理流程

```
chatsign-accuracy 采集数据
  ↓
Phase 2: 标准化视频，生成 glosses.json
  ↓
Phase 2 输出存储在共享目录
  ↓
编排器检测到新任务 (task_id)
  ↓
Phase 3: 读取 glosses.json
  ↓
Phase 3: 基于词汇列表构建词汇-视频数据库
  ↓
Phase 3: 写入 gloss_to_video_database.json
  ↓
Phase 4: 读取 gloss_to_video_database.json，开始处理每个词汇
```

### 多任务隔离

编排器支持多个任务并发运行。每个任务的数据完全隔离：

```
/data/chatsign/shared/
├── task_001/
│   ├── glosses.json
│   ├── gloss_to_video_database.json
│   └── .phase3.lock
├── task_002/
│   ├── glosses.json
│   ├── gloss_to_video_database.json
│   └── .phase3.lock
└── task_003/
    ├── glosses.json
    ├── gloss_to_video_database.json
    └── .phase3.lock
```

**重要**: 每个任务的词汇 ID 可能重复（如多个任务都有"apple"词汇），但它们存储在不同的目录中，不会互相干扰。

---

## 🔐 并发访问控制

### 文件锁机制

Phase 3 和 chatsign-accuracy 可能同时访问共享目录。编排器使用 **fcntl 文件锁** 保证并发安全：

```python
# 编排器端（Phase 3）
from backend.core.file_lock import acquire_phase3_lock

with acquire_phase3_lock(task_id):
    # 读取 glosses.json
    with open(gloss_db_path, "r") as f:
        gloss_data = json.load(f)

    # 构建 gloss_to_video_database.json
    database = {...}

    # 写入结果
    with open(result_path, "w") as f:
        json.dump(database, f, indent=2)
```

**锁的特点**:
- **非阻塞**: 如果无法立即获得锁，立即失败（不会无限等待）
- **进程级**: 利用操作系统的 fcntl.flock() 实现，所有进程都能看到锁
- **自动释放**: 进程结束或显式释放时自动清除
- **Unix-only**: 仅在 macOS 和 Linux 上支持

**锁文件位置**: `/data/chatsign/shared/{task_id}/.phase3.lock`

### 访问规则

| 操作 | 执行方 | 频率 | 锁 | 说明 |
|------|-------|------|----|----|
| 写入 glosses.json | Phase 2 | 1 次 | 需要 | Phase 2 完成后，chatsign-accuracy 写入一次 |
| 读取 glosses.json | Phase 3 | 1 次 | 需要 | Phase 3 启动时读取 |
| 写入 gloss_to_video_database.json | Phase 3 | 1 次 | 需要 | Phase 3 处理完毕后写入一次 |
| 读取 gloss_to_video_database.json | Phase 4 | 1 次 | 不需要 | Phase 4 读取（Phase 3 已完成，无竞争） |

**设计原则**:
- Phase 3 必须在 Phase 2 完成后执行
- 每个文件最多写入一次（幂等性）
- 读取需要文件锁保护（避免读到部分写入的数据）

---

## 🚀 使用指南

### 编排器创建任务

```python
# 创建新任务
task = await create_task(
    name="Chinese Sign Language Dataset",
    task_config={
        "source": "chatsign-accuracy",
        "augmentation_preset": "medium"
    }
)
task_id = task.task_id  # e.g., "task_20260320_101"
```

### Phase 3 工作流

```python
# Phase 3 处理逻辑
async def run_phase3(task_id: str, session: AsyncSession) -> bool:
    phase = None
    try:
        # 1. 获取 Phase 3 状态
        phase_result = await session.execute(
            select(PhaseState)
            .where(PhaseState.task_id == task_id)
            .where(PhaseState.phase_num == 3)
        )
        phase = phase_result.scalars().first()

        # 2. 标记为运行中
        await PhaseStateManager.mark_running(phase, session)

        # 3. 获取 glosses.json（带文件锁保护）
        with acquire_phase3_lock(task_id):
            gloss_db_path = FileManager.get_phase3_gloss_db_dir(task_id) / "glosses.json"
            with open(gloss_db_path, "r") as f:
                gloss_data = json.load(f)

        # 4. 构建 gloss_to_video_database.json
        database = {
            "task_id": task_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "glosses": {}
        }

        for gloss_item in gloss_data.get("glosses", []):
            gloss_id = gloss_item["gloss"]
            database["glosses"][gloss_id] = {
                "video_paths": [...],  # 根据 glosses.json 构建本地路径
                "count": gloss_item["count"],
                "total_duration": gloss_item["duration"]
            }

        # 5. 写入结果（带文件锁保护）
        with acquire_phase3_lock(task_id):
            output_path = FileManager.get_phase3_gloss_db_dir(task_id) / "gloss_to_video_database.json"
            with open(output_path, "w") as f:
                json.dump(database, f, indent=2)

        # 6. 标记完成
        await PhaseStateManager.mark_completed(phase, session)
        return True

    except Exception as e:
        logger.error(f"Phase 3 failed: {e}")
        if phase:
            await PhaseStateManager.mark_failed(phase, session, str(e)[:500])
        return False
```

### Phase 4 读取词汇数据库

```python
# Phase 4 在启动时读取 gloss_to_video_database.json
gloss_db_path = FileManager.get_phase3_gloss_db_dir(task_id) / "gloss_to_video_database.json"

if not gloss_db_path.exists():
    raise FileNotFoundError(f"Gloss database not found: {gloss_db_path}")

with open(gloss_db_path, "r") as f:
    gloss_db = json.load(f)

# 遍历每个词汇并处理
for gloss_id, video_info in gloss_db["glosses"].items():
    video_paths = video_info["video_paths"]

    # Phase 4 的 5 步处理...
    # Step 1: 提取帧
    # Step 2: 去重
    # Step 3: 姿态过滤
    # Step 4: 尺度标准化
    # Step 5: 生成最终视频
```

---

## 📂 文件路径规范

### 共享目录路径

```
{SHARED_ROOT}/
└── {task_id}/
    ├── glosses.json                    # Phase 2 输出
    ├── gloss_to_video_database.json    # Phase 3 输出
    └── .phase3.lock                    # 文件锁（隐藏文件）
```

**SHARED_ROOT** 定义：
```python
# backend/config.py
SHARED_DATA_ROOT = "/data/chatsign/shared"  # 可通过环境变量覆盖
```

### 标准化视频路径（chatsign-accuracy）

```
{ACCURACY_ROOT}/
└── tasks/
    └── {task_id}/
        └── standardized/
            ├── apple/
            │   ├── video_1.mp4
            │   └── video_2.mp4
            ├── ball/
            │   └── video_1.mp4
            └── ...
```

### Phase 4 输出路径

```
{ORCHESTRATOR_ROOT}/
└── tasks/
    └── {task_id}/
        └── phase4/
            └── output/
                ├── apple/
                │   └── cleaned_video.mp4
                ├── ball/
                │   └── cleaned_video.mp4
                └── ...
```

---

## ✅ 检查清单

### Phase 2 完成后

- [ ] `glosses.json` 成功写入 `/data/chatsign/shared/{task_id}/`
- [ ] 所有标准化视频存储在相应路径
- [ ] 词汇列表格式正确（包含 gloss, video_urls, count, duration）

### Phase 3 执行前

- [ ] glosses.json 已存在
- [ ] 共享目录权限正确（可读/可写）
- [ ] 无其他进程正在访问该任务的共享目录

### Phase 3 完成后

- [ ] gloss_to_video_database.json 成功写入
- [ ] 格式验证正确（包含 task_id, created_at, glosses）
- [ ] 所有词汇都有对应的视频路径列表
- [ ] video_paths 中的路径都存在且可读

### Phase 4 启动前

- [ ] gloss_to_video_database.json 已成功生成
- [ ] 所有词汇都在数据库中
- [ ] 标准化视频文件都存在

---

## 🐛 故障排除

### 问题 1: "Failed to acquire lock"

**原因**: 另一个进程正在访问同一任务的共享目录

**解决**:
1. 检查是否有其他 Phase 3 任务在运行
2. 查看 `.phase3.lock` 文件的修改时间
3. 如果锁文件超过 1 小时未修改，说明进程已崩溃，手动删除锁文件

```bash
rm /data/chatsign/shared/{task_id}/.phase3.lock
```

### 问题 2: "Gloss database not found"

**原因**: Phase 3 未完成或失败

**检查步骤**:
1. 查看 Phase 3 状态：`SELECT * FROM phase_states WHERE task_id = ? AND phase_num = 3`
2. 查看是否有错误日志
3. 确认 glosses.json 存在且可读

### 问题 3: video_paths 指向不存在的文件

**原因**: chatsign-accuracy 中的视频路径与 gloss_to_video_database.json 不匹配

**检查步骤**:
1. 核实标准化视频是否都存储在正确的目录
2. 确认路径分隔符（/ vs \）一致
3. 验证文件权限（读取权限）

### 问题 4: 多任务词汇冲突

**场景**: 多个任务都有"apple"词汇，是否会互相干扰？

**答案**: **不会**。每个任务的数据完全隔离在自己的 task_id 目录下：
```
/data/chatsign/shared/
├── task_001/glosses.json  (包含 "apple")
├── task_002/glosses.json  (也包含 "apple")
```

它们分别处理，互不影响。

---

## 🔗 相关文档

- [README.md](README.md) - 项目概览
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - Phase 3 完成情况
- [PIPELINE_STAGE_INDEPENDENCE.md](PIPELINE_STAGE_INDEPENDENCE.md) - Stage 接口规范
- [backend/workers/phase3_worker.py](backend/workers/phase3_worker.py) - 实现代码
- [backend/core/file_lock.py](backend/core/file_lock.py) - 文件锁实现

---

**版本**: 1.0
**更新日期**: 2026-03-24
**维护者**: ChatSign Team
**状态**: ✅ 完成
