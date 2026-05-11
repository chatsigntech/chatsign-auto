# 制品库与升级推送（Release Registry）— 设计文档

**状态**: 草案 v0.1
**日期**: 2026-04-29
**作者**: yfang + Claude
**关联文档**: `PIPELINE.md`, `ORCHESTRATOR_INTEGRATION_MAP.md`, `PHASE3_INTEGRATION.md`

---

## 0. 文档说明

本文档描述将 `chatsign-auto`（pipeline）与 `chatsign-accuracy`（审核站）两套系统现有的"推送"能力整合为**统一的制品库 + 升级推送中心**的方案。

**目标读者**: 开发与运维。
**与既有文档的关系**:
- `PIPELINE.md` 描述 8 阶段流水线"产出什么"。
- 本文档描述这些产出（加上 accuracy 站审核完成的视频 + gloss）"如何打包、版本化、推送到下游手语双向翻译服务器"。
- 不涉及单机服务部署，那部分见 `deployment/DEPLOYMENT_GUIDE.md`。

**重要约定**: 本文档为方案草稿，未实施。落地前需就 §13 的待决问题与作者确认。

---

## 1. 背景与诉求

`chatsign-auto`（流水线）+ `chatsign-accuracy`（审核站）两套系统的最终服务对象，是一组**手语双向翻译服务器**（如 ChatSignAvatar 等）。这些下游服务器需要两类东西才能工作：

| 用途 | 需要的素材 |
|---|---|
| **文本 → 手语** | 一组"标准手语视频" + 对应的 `gloss.csv`（视频文件名 ↔ 词/句的映射） |
| **手语 → 文本** | 一套训练好的推理模型（权重 + 词表 + 原型嵌入） |

目前这两类素材已经在不同的位置生成，但分发给下游服务器的方式割裂、不可追溯，且 Phase 8 模型完全没有自动化通道。

**诉求**: 一个集中管理三类升级品（视频、Phase 3 标准视频、Phase 8 模型）的"标准库房"，并提供"一键升级"到任意下游服务器的能力，类似 accuracy 站现在的 PublishTab 但覆盖更全、有版本与回滚。

---

## 2. 现状盘点

### 2.1 三类升级品的来源与落盘

| 类别 | 产出方 | 落盘位置 | 关键文件 |
|---|---|---|---|
| **A. 已审核录制视频 + gloss** | accuracy（人工录制 + 人工审核） | `chatsign-accuracy/backend/data/` | `reports/pending-videos.jsonl`, `reports/review-decisions.jsonl`, 实际视频在 `uploads/videos/<userId>/` 或 `review/generated/`，gloss 由 `scripts/buildWordGlosses.py` 生成在 `reports/word-glosses.json` |
| **B. Phase 3 标准手语视频** | pipeline | `SHARED_DATA_ROOT/<task_id>/phase_3/output/` | `videos/word_*.mp4`, `phase3_report.json` |
| **C. Phase 8 推理模型套件** | pipeline | `SHARED_DATA_ROOT/<task_id>/phase_8/output/` | `checkpoints/best_cl.pth`（或 `best.pth`）, `prototypes/prototypes.pt`, `vocab.json`, `train.jsonl`, `dev.jsonl` |

### 2.2 已有的两套推送通道

| 通道 | 入口 | 后端 | 传输 | 推送内容 |
|---|---|---|---|---|
| **accuracy PublishTab** | `chatsign-accuracy/frontend/src/components/admin/PublishTab.jsx` | `chatsign-accuracy/backend/services/publishService.js`、路由 `backend/routes/adminRoutes.js`（POST `/api/admin/publish`） | `sshpass -f <600 模式临时文件>` + `scp`（并行 4，每文件 60s 超时） | A 类视频，重命名为 `<md5(videoId)[:10]>_hiya.mp4`；并同步 `gloss.csv`（27K schema：`ref, word, sourceid, synset_id, gloss, alternate_words`，下载远端版本→按 ref 合并→上传） |
| **Phase 3 远端发布** | API `/api/publish-servers` + pipeline 工作流 | `chatsign-auto/backend/workers/phase3_remote_publish.py`、API 在 `backend/api/publish_servers.py` | `sshpass`（`SSHPASS` 环境变量）+ `scp`，参数走 `backend/core/validation_patterns.py` 正则白名单，known_hosts 持久化在 `~/.ssh/known_hosts_chatsign_publish` | B 类视频 + gloss.csv |
| **Phase 3 → accuracy 回灌** | pipeline 内部 | `chatsign-auto/backend/workers/phase3_publish.py` | 本地文件拷贝 | 把 Phase 3 视频复制到 accuracy `review/generated/` 并写 `pending-videos.jsonl`，使其进入审核队列 |

### 2.3 共用配置

- 服务器列表唯一来源: `chatsign-auto/backend/data/publish_servers.json`，accuracy 端通过 `CHATSIGN_AUTO_DATA` 环境变量挂过来（默认相对路径）。
- 字段示例：
  ```json
  {
    "name": "chatsign-avatar-58",
    "host": "10.225.72.58",
    "port": 22,
    "username": "chatsign",
    "password": "chatsign2025",
    "default_target_dir": "/Applications/ChatSignAvatar.app/Contents/Resources/app/text_to_sign/videos/"
  }
  ```

### 2.4 当前缺口

1. **Phase 8 模型零自动化通道** —— 必须人工 `scp`/`rsync`，最大缺口。
2. **三类制品逻辑各自为政** —— accuracy 推 A，pipeline 推 B，C 没人管。
3. **没有"一次发版"概念** —— 看不到"这一次给某下游服务器更新了哪批视频 + 哪个 Phase 3 任务 + 哪个模型版本"。
4. **没有版本/回滚** —— 当前都是覆盖式推送，远端没保留历史版本，出了问题无法快速回退。
5. **凭据策略不一致** —— accuracy 用 `sshpass -f` 写临时文件；Phase 3 用 `SSHPASS` 环境变量法（更安全）。
6. **命名规则不一致** —— accuracy 视频 `<hex10>_hiya.mp4`；Phase 3 视频 `<task>_hiya_<word>.mp4`。
7. **`gloss.csv` 合并器只在 accuracy 一侧** —— `update-on-diff` + `sourceid=200` + `synset_id=md5(batchFile)` 这一套逻辑埋在 `publishService.js` 里，Phase 3 没有等价物。

---

## 3. 设计目标 / 非目标

### 3.1 目标

- **G1**: 三类制品（A/B/C）有统一的落地目录与元数据格式
- **G2**: Phase 8 模型支持一键推送到任意下游服务器
- **G3**: 单次"发版"可以同时携带 A+B+C 的指定版本
- **G4**: 远端保留 N 个历史版本，可一键回滚
- **G5**: 沿用 `publish_servers.json` 与现有 SSH 通道，不引入新的基础设施依赖（如 S3、Artifactory）
- **G6**: 凭据走 `SSHPASS` 环境变量法，统一收口
- **G7**: 所有发布操作可审计（谁、何时、推到哪、推了什么）

### 3.2 非目标

- **N1**: 不引入云对象存储（除非未来明确需要）
- **N2**: 不重写 accuracy PublishTab 的视频审核数据流，只让它消费/复用统一推送通道
- **N3**: 不做 CI/CD（如 GitHub Actions 触发的发布）；本期只做手动触发
- **N4**: 不解决 Phase 4–7 中间产物的归档/分发问题，那些是流水线内部状态

---

## 4. 概念模型

```
                 ┌──────────────────────────────┐
   accuracy ──►  │  Artifact (制品)              │  ─►  Push  ─►  下游服务器
   pipeline ──►  │   - kind: video|t2s|model    │
                 │   - version (immutable id)   │
                 │   - manifest (sha256 + meta) │
                 └──────────────────────────────┘
                              │
                              ▼
                 ┌──────────────────────────────┐
                 │  Release (发版)               │  ─►  Push  ─►  下游服务器
                 │   - release_id               │     (按 release 整体推)
                 │   - artifact refs            │
                 └──────────────────────────────┘
```

- **Artifact**: 一个制品的不可变快照（一批视频、一个 Phase 3 任务的产出、一个 Phase 8 模型套件）。一旦写入 registry，内容固定。
- **Release**: 一个"发版清单"，引用具体版本的 A/B/C 制品。Release 可单独推送某一类制品，也可整包推送。
- **Target Server**: 来自 `publish_servers.json`。

---

## 5. 仓库目录结构

引入新的 registry 根 `$RELEASE_ROOT`（默认 `/mnt/data/chatsign-release/`，env `RELEASE_ROOT` 覆盖）。**待 §13 Q2 确认。**

```
$RELEASE_ROOT/
├── artifacts/
│   ├── videos/<batch_id>/                    # 来自 accuracy 已审核
│   │   ├── <hex10>_hiya.mp4 ...
│   │   ├── gloss.csv                         # 27K schema
│   │   └── MANIFEST.json
│   ├── text_to_sign/<task_id>-<UTCts>/       # 来自 Phase 3
│   │   ├── word_*.mp4
│   │   ├── gloss.csv
│   │   └── MANIFEST.json
│   └── models/phase8/<task_id>-<UTCts>/      # 来自 Phase 8
│       ├── best_cl.pth (或 best.pth)
│       ├── prototypes.pt
│       ├── vocab.json
│       ├── train.jsonl
│       ├── dev.jsonl
│       └── MANIFEST.json
├── releases/
│   └── <release_id>.json                     # 引用上述具体 artifact 版本
└── audit/
    └── push-log.jsonl                        # 每次推送一行（append-only）
```

**约定**:
- artifact 子目录名一旦创建不再变（不可变）。
- 同一来源（task_id / batch_id）多次产出 → 多个时间戳子目录共存。
- `MANIFEST.json` 中包含每个被发布文件的 `sha256` + 来源元数据 + 创建时间。

---

## 6. Manifest 规范

### 6.1 视频类 Artifact Manifest

```jsonc
{
  "kind": "videos",
  "artifact_id": "videos/batch_701",
  "created_at": "2026-04-29T10:11:12Z",
  "source": {
    "system": "chatsign-accuracy",
    "batch_file": "001_hello.jsonl",
    "review_decisions_count": 234
  },
  "files": [
    {
      "path": "abcdef0123_hiya.mp4",
      "sha256": "...",
      "size_bytes": 312456,
      "video_id": "<accuracy videoId>",
      "sentence_id": 42,
      "sentence_text": "...",
      "translator_id": "...",
      "language": "en"
    }
  ],
  "gloss_csv": {
    "path": "gloss.csv",
    "sha256": "...",
    "schema": "27K",
    "sourceid": 200,
    "synset_id": "<md5(batch_file) 9-digit>"
  }
}
```

### 6.2 Text-to-Sign Artifact Manifest

```jsonc
{
  "kind": "text_to_sign",
  "artifact_id": "text_to_sign/task_38-20260427T1130Z",
  "created_at": "...",
  "source": {
    "system": "chatsign-auto",
    "task_id": "task_38",
    "phase3_report": "...摘要..."
  },
  "files": [ ... ],
  "gloss_csv": { ... }
}
```

### 6.3 Phase 8 Model Artifact Manifest

```jsonc
{
  "kind": "model_phase8",
  "artifact_id": "models/phase8/task_42-20260428T0930Z",
  "created_at": "...",
  "source": {
    "system": "chatsign-auto",
    "task_id": "task_42",
    "training_config_hash": "...",
    "input_artifacts": ["videos/batch_701", "text_to_sign/task_38-..."]
  },
  "files": [
    { "path": "checkpoints/best_cl.pth", "sha256": "...", "size_bytes": ... },
    { "path": "prototypes/prototypes.pt", "sha256": "..." },
    { "path": "vocab.json", "sha256": "..." },
    { "path": "train.jsonl", "sha256": "..." },
    { "path": "dev.jsonl", "sha256": "..." }
  ],
  "metrics": { "best_dev_wer": ..., "step": ... }
}
```

### 6.4 Release Manifest

```jsonc
{
  "release_id": "2026-04-29-r1",
  "created_at": "...",
  "created_by": "yfang",
  "notes": "...",
  "artifacts": {
    "videos":        ["videos/batch_701", "videos/batch_702"],
    "text_to_sign":  "text_to_sign/task_38-20260427T1130Z",
    "model_phase8":  "models/phase8/task_42-20260428T0930Z"
  }
}
```

### 6.5 Push 审计行（`audit/push-log.jsonl`，append-only）

```jsonc
{
  "ts": "2026-04-29T10:30:00Z",
  "operator": "yfang",
  "release_id": "2026-04-29-r1",
  "artifact_id": "models/phase8/task_42-20260428T0930Z",
  "target": "chatsign-avatar-58",
  "result": "ok",
  "bytes_sent": 1834567890,
  "duration_sec": 142.3,
  "remote_version_dir": "20260429T1030Z"
}
```

---

## 7. 推送通道设计

### 7.1 统一传输模块

新建 `chatsign-auto/backend/services/release_publish.py`（或同等路径），提供：

```python
def push_artifact(artifact_id: str, target: ServerSpec, *,
                  resume: bool = True, dry_run: bool = False) -> PushResult: ...

def push_release(release_id: str, target: ServerSpec) -> list[PushResult]: ...
```

实现要点：
- 凭据统一走 `SSHPASS` 环境变量法（参考现有 `phase3_remote_publish.py`），不写临时文件。
- 大文件传输优先 `rsync --partial --inplace --append-verify` 而非 `scp`，支持断点续传。`scp` 作为兜底。
- 分类用不同命令更稳妥：
  - 视频/T2S：每个文件单独传，可并行（沿用 accuracy 现有的 4 worker 并行模型）。
  - Phase 8 模型：先打包 `tar.zst`（或 `tar.gz`）再单流传输 + 远端校验 sha256，避免多文件中断后状态不清。**待 §13 Q3 确认**。
- 远端目标路径：`<server.targets.<kind>>/<version_dir>/`，其中 `version_dir` 默认是 UTC 时间戳。
- 推送结束后远端 `current/` 软链切到新版本目录（**原子化操作**，给回滚留路径）。
- 所有 SSH 参数走现有正则白名单（`backend/core/validation_patterns.py`），防注入。

### 7.2 服务器配置扩展

`publish_servers.json` 每条记录扩展为：

```jsonc
{
  "name": "chatsign-avatar-58",
  "host": "10.225.72.58",
  "port": 22,
  "username": "chatsign",
  "password": "chatsign2025",
  "default_target_dir": "/.../text_to_sign/videos/",   // 兼容老字段
  "targets": {
    "videos":       "/.../release/videos/",
    "text_to_sign": "/.../release/text_to_sign/",
    "model_phase8": "/.../release/models/phase8/"
  },
  "retain_versions": 3
}
```

向后兼容：缺 `targets` 时降级到 `default_target_dir`。

### 7.3 `gloss.csv` 合并器抽出

把 accuracy `publishService.js` 里的"download → update-on-diff merge → upload"逻辑抽到独立模块（无论是 JS 共享还是 Python 重写一份，**待 §13 Q4**），让 Phase 3 的视频推送也能复用。

---

## 8. UI / API 设计

### 8.1 入口位置（**§13 Q1，待确认**）

两条路：

- **方案 A**: 扩展 accuracy 的 `PublishTab.jsx`，新增"模型"与"标准手语视频"子页签。
  - 优点: 用户已熟悉，前端框架就位。
  - 缺点: accuracy 本职是"数据源/审核"，让它代理推 Phase 8 模型职责模糊；需要 accuracy 反向读取 `SHARED_DATA_ROOT`。

- **方案 B（倾向）**: 在 chatsign-auto 主台新建 `ReleaseTab`。
  - 优点: pipeline 自然知道 B/C 的位置；与流水线生命周期一致。
  - 缺点: 用户要在两个站点之间切换；前端从零写。

### 8.2 后端 API（无论 A/B 入口）

```
GET  /api/release/artifacts?kind=videos|text_to_sign|model_phase8
GET  /api/release/artifacts/:artifact_id
POST /api/release/artifacts/import   # 触发从源系统(accuracy / phase3 / phase8)采集到 registry
GET  /api/release/releases
POST /api/release/releases           # 创建 release（指定 artifact 版本组合）
GET  /api/release/releases/:release_id
POST /api/release/push               # body: { release_id|artifact_id, target_name }
GET  /api/release/push/jobs/:job_id
GET  /api/release/push/jobs?limit=50

# 沿用既有
GET/POST/PUT/DELETE  /api/publish-servers
```

推送任务沿用 accuracy 现有的"job + poll"模型（前端 2s 轮询）。

### 8.3 前端关键交互

1. 选择/创建 Release（指定 A/B/C 版本）→ 创建按钮
2. 选择 Release + 一个或多个目标服务器 → 推送
3. 实时进度（每文件状态 + 总进度）
4. 历史 Release 列表 + 单 Release 推送历史 + 回滚按钮（切远端 `current/` 软链）

---

## 9. 凭据与安全

- `publish_servers.json` 仍是凭据真源，文件权限 `0600`，仅 backend 用户可读。
- 传输时使用 `SSHPASS` 环境变量（不写临时文件、不出现在命令行 argv 里）。
- 远端命令参数走正则白名单（已有 `validation_patterns.py`）。
- known_hosts 持久化在 `~/.ssh/known_hosts_chatsign_publish`，首次连接需要人工确认 fingerprint（或预置）。
- 审计日志不记录密码、不记录文件内容，只记录元数据（host、artifact_id、bytes、耗时、结果码）。
- 未来迁移到 SSH 密钥（`-i`），密码字段留空时自动走密钥。

---

## 10. 版本化与回滚

远端目录约定：

```
<server.targets.<kind>>/
├── 20260427T1130Z/          # 历史版本
├── 20260428T0930Z/
├── 20260429T1030Z/          # 最新
└── current  →  20260429T1030Z   # 软链
```

- 推送流程：上传到新时间戳目录 → 远端校验 sha256 → 原子切 `current` 软链。
- 回滚：把 `current` 软链切到上一个时间戳目录（一次 ssh 命令，秒级）。
- 远端清理：保留最近 `retain_versions`（默认 3）个版本，旧的删除。
- 下游服务（如 ChatSignAvatar）只读 `current/`，对它来说切版本是原子的。**前提是下游服务支持热加载或重启时读最新软链 —— 见 §13 Q5。**

---

## 11. 与现有系统的迁移路径

| 现有 | 新方案下的位置 |
|---|---|
| accuracy `PublishTab` 推视频 | 短期：保留并继续工作；中期：底层调用统一 push 模块；长期：作为 release 的一部分 |
| accuracy `publishService.js` | 视频/gloss 合并逻辑保留并抽公共部分；SCP 逻辑由统一模块替代 |
| `phase3_remote_publish.py` | 收编为统一模块的"transport 实现"，对外 API 不变 |
| `phase3_publish.py`（→ accuracy 回灌） | 不变，与本方案正交 |
| `publish_servers.json` | 字段向后兼容扩展 |

---

## 12. 分阶段落地计划

| 阶段 | 范围 | 价值 | 预计周期 |
|---|---|---|---|
| **Phase A** | 给 Phase 8 加发布通道：复用 `phase3_remote_publish.py` 模式，新增 `phase8_publish.py` + `/api/publish-models`，先做"一键推送一个 task 的模型套件 + sha256 校验"。先不做 release/版本/UI，命令行 + 简单页面即可。 | 立即补上最大缺口（C 类） | 3–5 天 |
| **Phase B** | 抽 release registry 落地：建 `$RELEASE_ROOT`、定义 manifest、写 import 工具（accuracy 视频 / phase 3 / phase 8 → registry）。 | 让"发了什么"可追溯 | 5–7 天 |
| **Phase C** | Release 概念 + 一键发版 + 版本目录 + `current` 软链 + 回滚。 | 真正的"升级"语义 | 5–7 天 |
| **Phase D** | 统一前端入口（A/B 二选一落地），把 accuracy PublishTab 接入新通道。 | 收口体验 | 4–6 天 |
| **Phase E** | gloss.csv 合并器抽公共、凭据全面切到 SSHPASS env、known_hosts 整理、审计完善。 | 清理技术债 | 3–4 天 |

各阶段独立可发布；Phase A 单独做完即可显著缓解眼前痛点。

---

## 13. 待决问题（必须用户确认）

1. **Q1 入口** — accuracy 的 PublishTab 扩展（A） vs chatsign-auto 新建 ReleaseTab（B）？倾向 B。
2. **Q2 `$RELEASE_ROOT` 路径** — 用 `/mnt/data/chatsign-release/`，还是挂在 `SHARED_DATA_ROOT` 下当一个子目录？是否要走 `pipeline-data` 那个外置盘？
3. **Q3 Phase 8 模型传输** — 直接传 `best.pth`（1–2 GB）走 rsync 断点续传，还是先 `tar.zst` 打包成单文件传？倾向打包，原子性更好。
4. **Q4 `gloss.csv` 合并器位置** — JS 模块给 accuracy 用 + Python 模块给 pipeline 用（重复实现），还是统一收到 chatsign-auto 一侧（accuracy 通过 HTTP 调用）？倾向后者。
5. **Q5 下游服务的热加载** — 下游服务器（如 ChatSignAvatar）拿到新视频/新模型后，是自己定时扫盘，还是需要我们 SSH 调它的重启/reload 命令？这直接决定推送流程要不要带"远端 hook"步骤。
6. **Q6 凭据切换节奏** — 把 accuracy `sshpass -f` 替换成 `SSHPASS` env 是否纳入 Phase A？还是延后到 Phase E？涉及现有逻辑改动。
7. **Q7 retain_versions 默认值** — 远端保留几份历史？默认 3 是否够？大模型场景下盘吃紧。
8. **Q8 release 是否强制三类齐全** — 只升级模型不升级视频是否允许？倾向允许（每类独立可推），release 中允许字段缺省。

---

## 14. 参考代码位置（已存在，无需新建）

- accuracy 推送逻辑: `chatsign-accuracy/backend/services/publishService.js`
- accuracy 推送 UI: `chatsign-accuracy/frontend/src/components/admin/PublishTab.jsx`
- accuracy 路由: `chatsign-accuracy/backend/routes/adminRoutes.js`
- pipeline Phase 3 远端推送: `chatsign-auto/backend/workers/phase3_remote_publish.py`
- pipeline Phase 3 → accuracy 回灌: `chatsign-auto/backend/workers/phase3_publish.py`
- pipeline 服务器配置 API: `chatsign-auto/backend/api/publish_servers.py`
- 服务器列表存储: `chatsign-auto/backend/data/publish_servers.json`
- SSH 参数白名单: `chatsign-auto/backend/core/validation_patterns.py`
- Phase 8 训练入口: `chatsign-auto/backend/workers/phase8_training.py`
- Phase 8 输出发现 / 模型 API: `chatsign-auto/backend/recognition/api.py`（`_find_phase8_outputs`）
- 配置中心: `chatsign-auto/backend/config.py`（`SHARED_DATA_ROOT`, `CHATSIGN_ACCURACY_DATA`, `SIGN_VIDEO_OUTPUT_DIR`, `TRAINING_DATA_RETENTION` 等）

---

## 修订历史

| 版本 | 日期 | 作者 | 摘要 |
|---|---|---|---|
| v0.1 | 2026-04-29 | yfang + Claude | 初稿，盘点现状 + 提出三层方案（artifact/release/push），列出 8 个待决问题 |
