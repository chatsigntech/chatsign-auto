# test_real → orchestrator 算法回流升级方案

**Date:** 2026-05-03
**Author:** drafted with Claude Code
**Status:** Draft — pending review before any code change
**Scope:** 把 `test_real` 里的 3 项算法增量**移植**到 chatsign-auto orchestrator 自身代码树（submodule + workers + scripts），不改 web/服务层

> **架构原则（强制）**
> `test_real` 仅作为**参考标杆**（reference / oracle），用于：
> 1. 算法源代码 / 配置的 ground truth
> 2. 评测集 (`selected_100_*.json`) 与 BLEU 数据的对照
>
> orchestrator 运行时**不得**调用 `test_real/` 路径下的任何脚本或 import 任何 test_real 模块。所有算法必须**移植到 chatsign-auto 自身的 submodule (`spamo_segement` / `gloss_aware`) 或 `backend/scripts/`** 中执行。

---

## 0. 背景

`test_real` (junyi2005/test_real, 已注册为 submodule，仅作参考) 是同一条 8-phase 流水线的 CLI reproducer，里面包含 3 处算法 / 数据处理升级，是 orchestrator 现有 worker 还没有的：

| # | 升级项 | 来源文件 | 已有 BLEU 收益 |
|---|---|---|---|
| 1 | SpaMo colent 变体 + **36× concat-aug** | `phase4_seg_train/` + `preprocess/06b_build_concat_aug.py` | base 85.58 → colent **94.19** (chatsign 21x test, ep207) |
| 2 | 双中心 split-level 识别预训练 | `phase8_training/` (`ssl_pretraining_glosspose_split_level.py` + `build_prototypes_both.py`) | 词级 / 句级中心分别建，分类更准 |
| 3 | **双源词级数据**（**#1 的前置**）：ASL-27K (C 类源) + chatsign-accuracy 已审词视频 (B 类源) | `preprocess/import_asl_videos.py` + `03b_extract_word_features.py` | 解锁 #1 的 36x concat-aug，使 BLEU 94 收益可达 |

orchestrator P1/P2/P3/P5/P6/P7 算法核心与 test_real 一致，**不需要替换**。

> **决策记录（2026-05-03）**
> - 帧像素去重 (test_real 02b) — **不做**。chatsign-accuracy 录制视频未出现卡顿 / 重复帧问题，无防御必要。
> - 现有 conda 环境 (`spamo` / `slrt1` / `sign_yijie` / `GUAVA`) **完全不动**：不升级 Python 版本、不重装包、不引入新 env。所有升级在现有 env 内完成。
> - 特征预热 — **本机 5090 Laptop 单卡夜里挂跑**，不调用 DGX / 外部 GPU 资源（ASL-27K 27090 mp4 + accuracy 838 mp4 的预热都在本机做）。
> - **C 类拼接源用 ASL-27K**（不引入 Uni-Sign / SLRT），**B 类拼接源用 chatsign-accuracy 已审词视频**（reviewer Heba/Tareq 等实录的 838 个 approved mp4，覆盖约 1806 独立词）。test_real 原版的 Uni-Sign（`~/junyi/unisign/Uni-Sign/data/ASL/train.jsonl`）不引入。
> - **目标 preset = 36x**（1 base + 10 A-shuffle + 10 B-org + 15 C-asl）—— B 和 C 两类源都齐了，36x 全套配方可达；BLEU 收益**仍待实测**（数据源仍与 test_real Uni-Sign 不完全相同），但接近 94 的概率比 21x 大。

---

## 1. 总览

### 1.1 实施顺序

依赖链：`#3 (双源词级数据) → #1 (SpaMo colent) → #2 (P8 split-level)`。

| 顺序 | 升级项 | 改动面 |
|---|---|---|
| 第 1 步 a | **#3a ASL-27K 资源就绪 (C 类源)** | (a) 一次性预热 `backend/scripts/precompute_asl27k_features.py`（夜里挂跑 5-9 小时）；(b) `backend/scripts/asl_resources.py:resolve_asl_resources(glosses) -> dict`（in-memory） |
| 第 1 步 b | **#3b accuracy 词视频资源就绪 (B 类源)** | (a) 一次性预热 `backend/scripts/precompute_accuracy_word_features.py`（838 mp4，几十分钟）；(b) `backend/scripts/org_resources.py:resolve_org_resources(glosses) -> dict`（按 word-glosses.json + review-decisions.jsonl 联合查询） |
| 第 2 步 | **#1 SpaMo colent + 36x concat-aug** | 升级 `spamo_segement` submodule（移植 colent 三件文件）+ 新增 `backend/scripts/build_concat_aug.py`（输入接口接受 `gloss_resources_asl` + `gloss_resources_org` **两个** dict）+ 改 `phase4_segmentation_train.py` |
| 第 3 步 | **#2 P8 split-level dual-centroid** | 改 `phase8_training.py` 调 `gloss_aware` submodule 已有的 split-level 脚本（HEAD 已含 split-level，dataset 接口已确认通过 filename suffix 支持，详见 §4.2） |

**注**：#3a 和 #3b 互不依赖，可并行实施。在 #1 启动前需要两者都完成。

### 1.2 关键架构决策

**A. 算法代码归属：哪些文件落到哪里**

orchestrator 运行时**不得**调用 test_real。所有需要的算法都按"模型/配置 → submodule，胶水/预处理 → backend/scripts/" 的原则归位：

| test_real 来源 | 性质 | 移植目标 | 备注 |
|---|---|---|---|
| `phase4_seg_train/configs/chatsign_concat_aug_colent.yaml` | SpaMo 训练 config | `spamo_segement/configs/` | submodule fork 升级 |
| `phase4_seg_train/spamo/t5_slt_colent.py` | SpaMo 模型类 | `spamo_segement/spamo/` | submodule fork 升级 |
| `phase4_seg_train/spamo/ot_sinkhorn_colent.py` | OT 算子 | `spamo_segement/spamo/` | submodule fork 升级 |
| `phase4_seg_train/configs/chatsign_concat_aug_e1e2.yaml` + 配套 | e1e2 ablation | 同上（一并带过去） | 备用对照 |
| `preprocess/06b_build_concat_aug.py` | 训练数据合成 | `backend/scripts/build_concat_aug.py` | 重写为 chatsign-auto 内部脚本，**输入接受两个 dict**（`gloss_resources_asl` + `gloss_resources_org`），不读 word_lib 目录 |
| `preprocess/03b_extract_word_features.py` | 词级 CLIP 特征 | 拆为两个预热脚本 | ASL-27K → `backend/scripts/precompute_asl27k_features.py`；accuracy 词视频 → `backend/scripts/precompute_accuracy_word_features.py` |
| `preprocess/import_asl_videos.py` (Uni-Sign 用) | C 类（asl）数据导入 | **不移植** | 我们用 ASL-27K；改写成 `backend/scripts/asl_resources.py` 的 `resolve_asl_resources()` |
| **N/A**（test_real 假设 org_* 已存在） | B 类（org）数据导入 | `backend/scripts/org_resources.py:resolve_org_resources()` | 新增：从 chatsign-accuracy 已审词视频组装，无对应 test_real 源文件 |
| `phase8_training/...` 的 `ssl_pretraining_glosspose_split_level.py` + `build_prototypes_both.py` | P8 split-level | `gloss_aware` submodule **HEAD 已有，不动** | 不需移植，只改 worker 调用 |

**注**：`gloss_aware` submodule HEAD 的 split-level 脚本（777 行）比 test_real 版本（800 行）少 30 行 diff，缺 test_real 新加的 `--adaptive-schedule` 开关。**接受这个差异**，不 fork gloss_aware —— 这样 P8 升级零环境改动。

**移植方式**：
- **submodule 文件**：在 `spamo_segement` 仓建一个分支（建议 `chatsign-auto/colent`），把 colent/e1e2 三件文件 commit 进去，本仓 `git submodule update` 指向新 commit。**这是 fork 不是 PR upstream** —— 我们自管自，避免外部仓阻塞。
- **scripts 文件**：从 test_real 复制源码 → 放到 `backend/scripts/` → 删掉 test_real 风格的 CLI banner 和 `from __future__` 不必要的导入 → 适配 chatsign-auto 的 logger / config（`from backend.config import settings`）

**禁止的写法**（用于 review 时检查）：

```python
# BAD — orchestrator 不得 import 或调用 test_real
TEST_REAL_ROOT = ... / "test_real"             # 禁止
subprocess.run(["python", str(test_real_path / "...")])   # 禁止
sys.path.insert(0, str(... / "test_real"))     # 禁止
```

```python
# GOOD — 只引 chatsign-auto 自身资源
SPAMO_ROOT = ... / "spamo_segement"
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
```

**B. submodule 升级流程**（针对 spamo_segement）

> **⚠️ 文件规模警告**：colent 不是"小补丁"。实测：
> - `spamo/t5_slt.py` (base) = 934 行
> - `spamo/t5_slt_colent.py` (colent) = **1584 行**（+650 行新增 / 改动）
> - 类似比例的 `ot_sinkhorn_colent.py` 也是新增的较大文件
>
> 移植不是简单 copy 7 个文件就完事，需要 **import chain 兜底验证**：colent 可能依赖 spamo 内部其他 utils / loss helper。如果 test_real 的 colent 引用了 base 仓没有的 helper，需要把那些 helper 也带过去。

```bash
# 假设我们 fork 了 spamo_segement 到 chatsignavatar 组织
cd spamo_segement
git checkout -b chatsign-auto/colent

# Step B1: 复制核心 7 个文件
cp ../test_real/phase4_seg_train/configs/chatsign_concat_aug_colent.yaml configs/
cp ../test_real/phase4_seg_train/configs/chatsign_concat_aug_e1e2.yaml configs/
cp ../test_real/phase4_seg_train/configs/chatsign_concat_aug.yaml configs/
cp ../test_real/phase4_seg_train/spamo/t5_slt_colent.py spamo/
cp ../test_real/phase4_seg_train/spamo/t5_slt_e1e2.py spamo/
cp ../test_real/phase4_seg_train/spamo/ot_sinkhorn_colent.py spamo/
cp ../test_real/phase4_seg_train/spamo/ot_sinkhorn_e1e2.py spamo/

# Step B2: 默认 default.yaml 软链
ln -sf chatsign_concat_aug_colent.yaml configs/default.yaml

# Step B3: ★ Sanity check import chain（关键，不要跳）
conda activate spamo
python -c "from spamo.t5_slt_colent import FlanT5SLT; print('t5_slt_colent OK')"
python -c "from spamo.ot_sinkhorn_colent import sinkhorn; print('ot_sinkhorn_colent OK')"
python -c "from spamo.t5_slt_e1e2 import FlanT5SLT; print('t5_slt_e1e2 OK')"
python -c "from spamo.ot_sinkhorn_e1e2 import sinkhorn; print('ot_sinkhorn_e1e2 OK')"
# ↑ 若报 ModuleNotFoundError 或缺函数，说明 colent 依赖了 base 仓没有的 helper，
#   去 test_real/phase4_seg_train/spamo/ 找出缺失模块一并补过来

# Step B4: 提交 + 推 fork
git add -A && git commit -m "feat: import colent/e1e2 variants from test_real reference"
git push origin chatsign-auto/colent

# Step B5: 回主仓更新 submodule pin
cd ..
git add spamo_segement
git commit -m "Bump spamo_segement: add colent/e1e2 variants"
```

注意 **不要让 spamo_segement submodule 反过来引用 test_real**。检查移植后的 colent 文件 `import` 语句中是否有 test_real 路径残留。

**C. Python / conda 环境（明确不动）**

| Phase | env | 现状 | 升级后 | 备注 |
|---|---|---|---|---|
| P4 colent | `spamo` | PyTorch + Lightning + open_clip + omegaconf | **同** | colent 的 import 与 base 一致，无新包 |
| P8 split-level | `slrt1` (Python 3.8.20) | 跑 `ssl_pretraining_crossvideo_mlp_*` | **同** | gloss_aware HEAD 的 split_level 脚本不用 `BooleanOptionalAction`，3.8 即可跑 |
| #3 ASL 导入脚本 | 任意 | — | 用 `backend/` 主 env | pure Python + ffmpeg / cv2 |

**不需要**的事：
- 装 Python 3.10
- 重装 PyTorch / Lightning
- 升级 conda env
- 引入新的 Python 解释器路径（PHASE8_PYTHON310 / PHASE8_TORCHRUN 这些 env var **作废**）

---

## 2. 升级 #3：双源词级数据就绪（#1 的前置）

#3 拆成两半，互不依赖，可并行实施：
- **#3a**: ASL-27K 字典视频 → C 类拼接源 `asl_*`
- **#3b**: chatsign-accuracy 已审词视频 → B 类拼接源 `org_*`

两者共同支撑 §3.3 的 `build_concat_aug` 36x preset。

---

### 2.A #3a ASL-27K 资源就绪（C 类源）

#### 2.A.1 现状（重要发现）

orchestrator 实际上**已经在用 ASL-27K**，路径和查询代码都现成。位置：

```
/mnt/data/chatsign-auto-videos/ASL-final-27K-202603/
├── gloss.csv          27080 行（25440 个唯一词）
└── videos/            27090 个 .mp4，哈希文件名 + CSV 索引

/mnt/data/chatsign-auto-videos/clip_features/ASL-final-27K-202603/videos/
└── ... 22 个 *_s2wrapping.npy （旧 task 的副产物，远未完整）
```

CSV schema：`ref, word, sourceid, synset_id, gloss, alternate_words`，其中 `ref` 是 mp4 文件名，`word` 是 gloss 词。

**已有查询 / 缓存代码**（`backend/core/dataset_videos.py` + `backend/workers/phase4_segmentation_train.py`）：
- `ASL27K_VIDEOS` / `ASL27K_GLOSS_CSV` 路径常量
- `_load_asl27k_gloss_map()` — 加载 csv → `{word_lowercase: [filenames]}`
- `_find_gloss_videos(gloss, max_per_gloss)` — 给一个词返回 N 个匹配视频路径
- `CLIP_FEATURE_CACHE_DIR = settings.VIDEO_DATA_ROOT / "clip_features"` — 全局共享特征缓存
- `_prepopulate_feature_cache()` / `_save_features_to_cache()` — 缓存命中软链 / 落盘
- `_source_to_cache_key()` — 视频路径 → cache key（相对 `VIDEO_DATA_ROOT`）

**特征 shape**：`(T, 2048) float32`，CLIP-ViT-L/14 + S2 wrapping（`--scales 1 2`，patch 拼接）。22 个已缓存样本：T 范围 57-117，平均 70 帧/视频。

**结论**：#3 的实施大幅简化 —— 不需要重写 csv 解析、不需要 `ASL_POOL_DIR` env var、不需要从零移植 test_real 的 import_asl_videos.py。**部分**复用 orchestrator 现成函数：
- ✅ `_load_asl27k_gloss_map()` 加载 csv → `{lowercase_word: [filenames]}`，复用
- ✅ `ASL27K_VIDEOS` / `CLIP_FEATURE_CACHE_DIR` 路径常量，复用
- ⚠️ `_find_gloss_videos(gloss)` **不**直接用：它内部只做 `.lower()`，不处理 P1 输出的 `MORE_THAN` 类下划线 token；我们在 `asl_resources.py` 里加 `_normalize_for_asl27k()` 做下划线→空格转换，详见 §2.2 Step 2

#### 2.A.2 改动

简化设计：**不在 pipeline 里加 P2.5a 阶段**，不物化 `word_lib/<WORD>/asl_*.mp4` 软链树，因为 ASL-27K 是全局共享数据，所有 task 应该复用同一份。`build_concat_aug` 直接从 `gloss_resources` 内存 dict 消费。

**Step 1：一次性预热 ASL-27K 全部 27090 个特征**

新增 `backend/scripts/precompute_asl27k_features.py` —— 离线批跑脚本，把 ASL-27K 全部视频的 CLIP-ViT-L/14 + S2 双尺度特征写到 `clip_features/ASL-final-27K-202603/videos/<stem>_s2wrapping.npy`。复用 SPAMO submodule 自带的 `scripts/extract_features/extract_clip_from_mp4.py` 做实际提取，不引入新的模型加载代码。

设计要点：
- **断点续算（idempotent）**：跑前先扫已有 `*_s2wrapping.npy`，存在即跳过 → 现在的 22 个会自动复用，中断 Ctrl+C 之后重跑只补差额
- **失败兜底**：损坏 / 0 帧的 mp4 记录到 `failed.json`，不阻塞批次继续
- **进度日志**：每 100 条打一行（计数 + ETA）
- **CLI**：`--limit N` 用于先跑一小段做 sanity check

跑法（**本机单卡 5090 Laptop**，夜里挂上）：

```bash
conda activate spamo
python -m backend.scripts.precompute_asl27k_features --gpu 0 --batch-size 32
```

时间预估：单 5090 Laptop **5-9 小时**（CLIP-ViT-L/14 双尺度 ~80-120 fps，27090 视频 × 70 帧均值 ≈ 1.9M 帧）。dev 机夜里挂跑，第二天醒来就齐了。后续跑实测 micro-bench 后给准确数。

完成后磁盘占用约 **15 GB**（27090 × ~580 KB/file），落到 `/mnt/data/chatsign-auto-videos/clip_features/ASL-final-27K-202603/videos/`。

**Step 2：新增 `backend/scripts/asl_resources.py` —— gloss → 资源 dict 的解析函数**

不创建独立 worker、不修改 pipeline 序列、不物化 `word_lib`。只提供一个**纯函数**给 `build_concat_aug.py` 在 P4 内部直接调用。

**关键：命名约定转换**

| 数据源 | gloss 表现形态 | 例子 |
|---|---|---|
| orchestrator P1 输出（`combined_pipeline.py:66`） | UPPER_UNDERSCORE | `MORE_THAN`, `ABU_DHABI` |
| ASL-27K `gloss.csv` `word` 列 | lowercase with spaces | `more than`, `abu dhabi`, `(movie/tv) credits` |
| `dataset_videos._find_gloss_videos()` 内部查找 key | `gloss.strip().lower()`（仅小写，**不替换下划线**） | `more_than` → 查不到 `more than` ❌ |

**含义**：直接用 `_find_gloss_videos(P1_token)` 对多词 token 全部 miss。必须先做 `_` → ` ` 转换：

```python
"""Resolve ASL-27K mp4 + precomputed feature paths for a gloss list."""
from pathlib import Path
from backend.core.dataset_videos import _load_asl27k_gloss_map, ASL27K_VIDEOS
from backend.config import settings
import logging

logger = logging.getLogger(__name__)
ASL27K_FEATS = settings.VIDEO_DATA_ROOT / "clip_features" / "ASL-final-27K-202603" / "videos"


def _normalize_for_asl27k(token: str) -> str:
    """Convert orchestrator P1 token (UPPER_UNDERSCORE) to ASL-27K csv key (lower with spaces).

    Examples:
        MORE_THAN     -> "more than"
        ABU_DHABI     -> "abu dhabi"
        HOME          -> "home"
        #ALL          -> "#all"

    NOTE: do NOT call dataset_videos._find_gloss_videos directly — its lookup
    only does .lower() and would miss multi-word tokens.
    """
    return token.strip().lower().replace("_", " ")


def resolve_asl_resources(
    glosses: list[str],
    max_per_gloss: int = 5,
) -> dict:
    """
    Returns:
        {
            "resources": {gloss: [(mp4_path, npy_path), ...]},  # gloss 原文当 key（保持下划线形态），不做目录化
            "missing": [gloss, ...],                             # ASL-27K 没找到的词
            "feat_missing_files": [mp4_filename, ...],           # 视频找到但特征缓存缺失（debug 用）
            "n_glosses_hit": int,
            "n_clips_total": int,
        }
    """
    gloss_map = _load_asl27k_gloss_map()      # {lowercase_word: [filenames]}
    out = {"resources": {}, "missing": [], "feat_missing_files": [],
           "n_glosses_hit": 0, "n_clips_total": 0}
    for gloss in glosses:
        key = _normalize_for_asl27k(gloss)
        filenames = gloss_map.get(key, [])
        if not filenames:
            out["missing"].append(gloss)
            continue
        pairs = []
        for fn in filenames[:max_per_gloss]:
            src_mp4 = ASL27K_VIDEOS / fn
            if not src_mp4.exists():
                continue
            src_npy = ASL27K_FEATS / f"{src_mp4.stem}_s2wrapping.npy"
            if src_npy.exists():
                pairs.append((src_mp4, src_npy))
            else:
                out["feat_missing_files"].append(src_mp4.name)
        if pairs:
            out["resources"][gloss] = pairs   # key 用原 gloss（带下划线），下游用同样的 key 查
            out["n_glosses_hit"] += 1
            out["n_clips_total"] += len(pairs)
    hit_rate = out["n_glosses_hit"] / max(len(glosses), 1)
    logger.info(
        f"ASL-27K resolved: {out['n_glosses_hit']}/{len(glosses)} glosses ({hit_rate:.1%}), "
        f"{out['n_clips_total']} clips total; missing={len(out['missing'])}, "
        f"feat_missing={len(out['feat_missing_files'])}"
    )
    if out["feat_missing_files"]:
        logger.warning(
            f"{len(out['feat_missing_files'])} videos lack precomputed features; "
            f"re-run precompute_asl27k_features.py to fill"
        )
    return out
```

约 60 行。**gloss 原文（含下划线）直接做 dict key**，下游 `build_concat_aug` 按 P1 token 原文查 dict —— 不做目录化转换，无 `gloss.upper().replace()` 类潜在 bug。

**为什么不复用 `_find_gloss_videos`**：它内部 `gloss.strip().lower()` 不替换下划线，会让 `MORE_THAN` 类多词 token 全部 miss。我们需要 `_normalize_for_asl27k` 做"下划线 → 空格"的关键转换，所以直接走 `_load_asl27k_gloss_map` 自己查。

**Step 3：`build_concat_aug` 接口**

移植后的 `backend/scripts/build_concat_aug.py` 在内部调 `resolve_asl_resources(glosses)` 拿到 dict，然后按 06b 算法生成 C 类拼接样本（21x preset 仅用 asl_*，不需要 org_*）。具体改动详见 §3.3 Step 2。

#### 2.A.3 前置数据：已就绪 + 一次性预热

| 项 | 状态 |
|---|---|
| ASL-27K 视频 | 就绪 `/mnt/data/chatsign-auto-videos/ASL-final-27K-202603/videos/` (27090 个) |
| ASL-27K 索引 | 就绪 `gloss.csv` (27080 行 / 25440 唯一词) |
| 查询代码 | 就绪 `backend/core/dataset_videos.py:_find_gloss_videos()` |
| 词汇覆盖 | 抽样 hello/home/school/today/tomorrow/yesterday 全命中 |
| **特征缓存** | **待预热**：22/27090 已就位，需跑一次 `precompute_asl27k_features.py` 补齐剩余 27068 个（单 5090 Laptop 5-9 小时） |
| cleanup 状态 | filename 含 `_hiya_complete`，已经过 hiya 处理流水线 |

不需要 `ASL_POOL_DIR` env var（路径走 `settings.VIDEO_DATA_ROOT`）；不需要外部 GPU 资源（本机 5090 Laptop 即可）；**不需要 `word_lib` 目录树**（in-memory dict 即可）。

#### 2.A.4 验证

- 跑预热脚本，确认 27090 个 `*_s2wrapping.npy` 全部就位
- 拿一份典型 task 的 P1 输出 gloss 列表（如 all-glosses task 的 `glosses.json`），调 `resolve_asl_resources(glosses)`：
  - 检查 `n_glosses_hit / len(glosses) ≥ 0.8`（命中率门槛）
  - 检查 `feat_missing_files == []`（预热完整）
  - 抽 3 个命中词肉眼看 mp4 内容确实对应

#### 2.A.5 回滚

**不引入新 env var**。回滚通过 `git revert` worker 改动（恢复到调用 base config 的状态），预热好的 `*_s2wrapping.npy` 缓存保留不删（其他 phase 的 ASL-27K fallback 仍依赖它）。

---

### 2.B #3b chatsign-accuracy 词视频资源就绪（B 类源）

#### 2.B.1 现状

chatsign-accuracy 上 reviewer 已经录了 **838 个 approved 词视频**，覆盖约 **1806 个独立词**。这是 test_real `org_*` 在我们这边的对应物。

**数据位置**：

```
chatsign-accuracy/backend/data/
├── uploads/videos/<reviewer>/*.mp4         实录 mp4，1969 个总数
│   ├── Heba/        (677)
│   ├── Tareq/       (1212)
│   ├── Rawdah/      (73)
│   └── 其他 (~7)
├── reports/word-glosses.json              22120 条元数据
│   每条: {alternate_words: "inclusive", gloss: "...", synset_pos: "a", synset_id: ...}
│   key 是 mp4 文件名，value 是该视频对应的 gloss 信息
└── reports/review-decisions.jsonl         review 决策日志
   approved 视频 19470 条（去重）
```

**交集统计**：

| 维度 | 数量 |
|---|---|
| mp4 实际存在 | 1969 |
| ∩ word-glosses metadata | 1471 |
| ∩ approved | **838** ← 实际可用 |
| 覆盖独立词（alternate_words 拆分） | **1806** |

#### 2.B.2 改动

**Step 1：一次性预热 838 个 accuracy 词视频的 CLIP 特征**

新增 `backend/scripts/precompute_accuracy_word_features.py` —— 类比 ASL-27K 预热，但跑 accuracy uploads 这 838 个 mp4。

**关键差异**：accuracy uploads **不在** `VIDEO_DATA_ROOT` 下（在 `chatsign-accuracy/backend/data/uploads/...`），所以**不**走 `phase4_segmentation_train._prepopulate_feature_cache` 那套基于 `VIDEO_DATA_ROOT` 相对路径的命名约定。改用独立缓存目录：

```
clip_features/accuracy_word_uploads/<reviewer>/<filename>_s2wrapping.npy
```

设计要点：
- 复用 SPAMO `extract_clip_from_mp4.py` 做实际提取（不引入新模型加载）
- 断点续算：扫已存在 npy 跳过
- 失败兜底：损坏 mp4 记录到 `failed.json`
- CLI：`--reviewer Tareq` 限制单 reviewer，`--limit N` 限制条数

跑法（本机单卡）：

```bash
conda activate spamo
python -m backend.scripts.precompute_accuracy_word_features --gpu 0 --batch-size 32
```

**时间预估**：838 mp4 × 平均 60 帧 × 双尺度 / 80-120 fps ≈ **10-15 分钟**（远比 ASL-27K 27090 视频快）。磁盘约 480 MB（838 × ~580 KB）。

**Step 2：新增 `backend/scripts/org_resources.py` —— gloss → org 资源 dict**

类比 `asl_resources.py`，但查询源是 `word-glosses.json` + `review-decisions.jsonl`：

```python
"""Resolve chatsign-accuracy approved word videos for a gloss list."""
import json
from pathlib import Path
from collections import defaultdict
from backend.config import settings
import logging

logger = logging.getLogger(__name__)

ACCURACY_DATA = settings.CHATSIGN_ACCURACY_DATA   # already configured
WORD_GLOSSES_JSON = ACCURACY_DATA / "reports" / "word-glosses.json"
REVIEW_DECISIONS = ACCURACY_DATA / "reports" / "review-decisions.jsonl"
UPLOADS_DIR = ACCURACY_DATA / "uploads" / "videos"
ORG_FEATS = settings.VIDEO_DATA_ROOT / "clip_features" / "accuracy_word_uploads"

_index_cache = None


def _build_org_index() -> dict[str, list[Path]]:
    """phrase → [(mp4_path, npy_path), ...]   只包含 approved 且 mp4 存在的条目."""
    global _index_cache
    if _index_cache is not None:
        return _index_cache

    # 1. 取 approved set
    approved = set()
    with open(REVIEW_DECISIONS) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("decision") == "approved":
                    fn = r.get("videoInfo", {}).get("videoFileName") or r.get("videoFileName")
                    if fn:
                        approved.add(fn)
            except: pass

    # 2. 找 reviewer (uploads 目录结构)
    fn_to_path = {}
    for reviewer_dir in UPLOADS_DIR.iterdir():
        if reviewer_dir.is_dir():
            for mp4 in reviewer_dir.glob("*.mp4"):
                fn_to_path[mp4.name] = mp4

    # 3. 联合查询：mp4 存在 ∩ approved ∩ 在 word-glosses.json
    with open(WORD_GLOSSES_JSON) as f:
        glosses_meta = json.load(f)

    index: dict[str, list] = defaultdict(list)
    for fn, info in glosses_meta.items():
        if fn not in approved or fn not in fn_to_path:
            continue
        mp4_path = fn_to_path[fn]
        npy_path = ORG_FEATS / mp4_path.parent.name / f"{mp4_path.stem}_s2wrapping.npy"
        # alternate_words 拆同义词，每个都加索引
        alt = info.get("alternate_words", "")
        for w in alt.split(","):
            w = w.strip().lower()
            if w:
                index[w].append((mp4_path, npy_path))

    _index_cache = dict(index)
    logger.info(f"org index built: {len(_index_cache)} unique words from {len(approved & fn_to_path.keys())} approved mp4s")
    return _index_cache


def _normalize_for_org(token: str) -> str:
    """同 _normalize_for_asl27k：UPPER_UNDERSCORE → lower with space."""
    return token.strip().lower().replace("_", " ")


def resolve_org_resources(glosses: list[str], max_per_gloss: int = 5) -> dict:
    index = _build_org_index()
    out = {"resources": {}, "missing": [], "feat_missing_files": [],
           "n_glosses_hit": 0, "n_clips_total": 0}
    for gloss in glosses:
        key = _normalize_for_org(gloss)
        candidates = index.get(key, [])
        if not candidates:
            out["missing"].append(gloss)
            continue
        pairs = []
        for mp4_path, npy_path in candidates[:max_per_gloss]:
            if npy_path.exists():
                pairs.append((mp4_path, npy_path))
            else:
                out["feat_missing_files"].append(mp4_path.name)
        if pairs:
            out["resources"][gloss] = pairs
            out["n_glosses_hit"] += 1
            out["n_clips_total"] += len(pairs)
    hit_rate = out["n_glosses_hit"] / max(len(glosses), 1)
    logger.info(
        f"accuracy org resolved: {out['n_glosses_hit']}/{len(glosses)} glosses ({hit_rate:.1%}), "
        f"{out['n_clips_total']} clips total; missing={len(out['missing'])}, "
        f"feat_missing={len(out['feat_missing_files'])}"
    )
    return out
```

约 80 行。**跟 `asl_resources.py` 同样的接口形态**，下游 `build_concat_aug` 同样的方式消费。

#### 2.B.3 前置数据：已就绪 + 待预热

| 项 | 状态 |
|---|---|
| accuracy uploads mp4 | 就绪 1969 个总，838 个 approved 可用 |
| word-glosses.json 元数据 | 就绪 22120 条，对 838 个 approved 全覆盖（mp4 ∩ metadata ∩ approved） |
| review-decisions.jsonl | 就绪 |
| 词覆盖（alternate_words 拆分） | 1806 unique words |
| **特征缓存** | **待预热**：跑 `precompute_accuracy_word_features.py`，~10-15 分钟 |

#### 2.B.4 验证

- 跑预热脚本，确认 `clip_features/accuracy_word_uploads/<reviewer>/*_s2wrapping.npy` 至少 838 个就位
- 拿同一份 task P1 输出 gloss 列表，调 `resolve_org_resources(glosses)`：
  - 检查 `n_glosses_hit / len(glosses) ≥ 0.4`（**门槛比 ASL-27K 低**：1806 vs 25440 词覆盖差距大）
  - 与 `resolve_asl_resources(glosses)` 比较 missing 列表，看 org 命中率是否符合预期
- 抽 3 条 mp4 + 对应 word-glosses metadata 肉眼比对，确认 alternate_words 拆分逻辑正确

#### 2.B.5 回滚

不引入新 env var。回滚通过 `git revert` `org_resources.py` 引用，让 `build_concat_aug` 回退到 21x preset（仅 asl_* concat，无 org_*）。预热好的 `accuracy_word_uploads/*` 缓存保留不删。

---

## 3. 升级 #1：SpaMo colent + 36x concat-aug — 主菜

### 3.1 现状

`backend/workers/phase4_segmentation_train.py:416` 当前用：

```python
template_path = SPAMO_ROOT / "configs" / "how2sign_contrastive_single.yaml"
```

- 模型类：`spamo.t5_slt.FlanT5SLT`（base 变体）
- OT 算子：标准 `ot_sinkhorn_null_noglobal_order_windownomean.py`
- 训练数据量：1×（仅 `train_info_ml.npy`，不做 concat-aug）

BLEU 基线：~85.58（按 test_real README 在 chatsign 21x test 集的复现数）

### 3.2 升级目标

| 维度 | 现 | 新 |
|---|---|---|
| `SPAMO_ROOT` | `chatsign-auto/spamo_segement`（保持不变） | `chatsign-auto/spamo_segement`（**仅 submodule pin 升级**） |
| 训练 config | `configs/how2sign_contrastive_single.yaml` | `configs/chatsign_concat_aug_colent.yaml` |
| 模型类 | `spamo.t5_slt.FlanT5SLT` | `spamo.t5_slt_colent.FlanT5SLT`（升级后的 submodule 中） |
| OT 算子 | 默认 sinkhorn | `spamo.ot_sinkhorn_colent`（升级后的 submodule 中） |
| 训练数据 | 当前 task 的 N 条标注 | **36x（默认）**：1 base + 10 A-shuffle + 10 B-org + 15 C-asl，由 `backend/scripts/build_concat_aug.py` 生成；C 类源 = ASL-27K（27090 词），B 类源 = chatsign-accuracy 已审词视频（838 mp4，1806 词）。**21x（fallback）**：B 类命中率太低或预热未完成时退化用，1 + 10 + 0 + 10 |

### 3.3 改动

**Step 1：移植 colent/e1e2 文件到 `spamo_segement` submodule**（参见 §1.2 末尾的 fork 命令 + sanity check）

升级后的 `spamo_segement` 应包含：
- `configs/chatsign_concat_aug_colent.yaml` (默认 default.yaml 软链)
- `configs/chatsign_concat_aug_e1e2.yaml`
- `configs/chatsign_concat_aug.yaml`
- `spamo/t5_slt_colent.py` (1584 行) + `spamo/t5_slt_e1e2.py`
- `spamo/ot_sinkhorn_colent.py` + `spamo/ot_sinkhorn_e1e2.py`

**注意**：`t5_slt_colent.py` (1584 行) 比 base `t5_slt.py` (934 行) 多了 650 行 —— 不是"小改动"。先按 §1.2 B Step B3 的 sanity check 跑通 import，再 commit submodule pin。

**Step 2：移植 `06b_build_concat_aug.py` 到 `backend/scripts/build_concat_aug.py`**

源码参考 test_real `preprocess/06b_build_concat_aug.py` (364 行)：1 base + 10 A-shuffle (符号链接) + 10 B-org (concat 物理 npy) + 15 C-asl (concat 物理 npy) = 36×/句。21x fallback = 1+10+0+10。

适配要点：
- 删 banner、test_real 风格注释
- **接口改为双 dict 输入，不读 `--word-lib` 目录**：

  ```python
  def build_concat_aug(
      base_anno, base_feat,
      gloss_resources_org,   # ← 来自 resolve_org_resources(),  B 类
      gloss_resources_asl,   # ← 来自 resolve_asl_resources(),  C 类
      anno_out, feat_out,
      preset="36x", val_fraction=0.1, split_seed=42,
  ) -> dict
  ```
- 两个 dict key 都是 P1 原 token（带下划线），value 是 `[(mp4_path, npy_path), ...]`
- 删除 test_real 的 `tok_to_word()` / `build_word_index()` / 文件系统扫描相关代码
- B 类拼接：从 `gloss_resources_org[token]` 取，**always picks first**（test_real 06b:concat_features 行为）
- C 类拼接：从 `gloss_resources_asl[token]` 取，**random pick**（保留 test_real 06b 的 `random.Random(seed)` 实例化）
- 保留 CLI 入口（手工 debug 用）
- 标准 logger

**⚠️ 必须保留 deterministic 行为**：

test_real `06b_build_concat_aug.py:concat_features()` 用 `rng = random.Random(...)` 做 C 类候选随机挑选。**移植时按源码原样保留 seed 推导逻辑**（具体 seed 派生公式以 06b 源码为准 —— 实施时打开文件直接抄那段 RNG 实例化代码，不要凭印象重写）。

不保留 deterministic 的后果：
- 同样的 task 跑两次 BLEU 数对不上 → 评估困难
- 与 test_real 的对照实验可重复性丢失

verifiable 的可重复性测试：连跑两次 `build_concat_aug` 同输入，diff 输出 npy + manifest，应当**字节级一致**。如果不一致，说明 seed 移植没做对，回去对照 test_real 06b 源码核查。

`--split-seed` 控制 train/val 划分（默认 42），这一项明确，按 test_real 默认值即可。

**Step 3：改 `phase4_segmentation_train.py`**

**(a) 模板路径**

```python
# from line ~416:
template_path = SPAMO_ROOT / "configs" / "how2sign_contrastive_single.yaml"

# to:
template_path = SPAMO_ROOT / "configs" / "chatsign_concat_aug_colent.yaml"
```

`SPAMO_ROOT` 不变，仍指 `spamo_segement` submodule（升级后里面就有 colent 文件）。

注意 `chatsign_concat_aug_colent.yaml` 的 schema 与 `how2sign_contrastive_single.yaml` 是否完全一致需要核对（`config.data.params.{train,val,test}.params.{anno_root,feat_root,vid_root,mae_feat_root}` 这几个字段必须存在）。如果不一致，本 worker 里 `for split in ("train", "validation", "test")` 那段循环要按新 schema 调整。

**(b) 新增 36× concat-aug 子步骤**

```python
from backend.scripts.build_concat_aug import build_concat_aug
from backend.scripts.asl_resources import resolve_asl_resources
from backend.scripts.org_resources import resolve_org_resources

async def _run_concat_aug(
    task_id: str,
    base_anno: Path, base_feat: Path,
    glosses: list[str],                 # 来自 P1 输出
    aug_anno_out: Path, aug_feat_out: Path,
    *, preset: str = "36x", val_fraction: float = 0.1, split_seed: int = 42,
) -> tuple[Path, Path]:
    """Step 4.2.5: resolve B/C resources + build concat-aug training data."""
    asl = resolve_asl_resources(glosses, max_per_gloss=5)
    org = resolve_org_resources(glosses, max_per_gloss=5)

    for which, res in [("ASL", asl), ("ORG", org)]:
        if res["feat_missing_files"]:
            logger.warning(
                f"[{task_id}] {which}: {len(res['feat_missing_files'])} videos lack "
                f"precomputed features; run precompute_*_features.py to fill"
            )

    # B 类命中率太低时降级到 21x preset（仅 C 类）
    org_hit_rate = org["n_glosses_hit"] / max(len(glosses), 1)
    effective_preset = preset
    if preset == "36x" and org_hit_rate < 0.4:
        logger.warning(
            f"[{task_id}] org hit rate {org_hit_rate:.1%} < 0.4 threshold, "
            f"falling back to 21x preset"
        )
        effective_preset = "21x"

    summary = build_concat_aug(
        base_anno=base_anno, base_feat=base_feat,
        gloss_resources_org=org["resources"],
        gloss_resources_asl=asl["resources"],
        anno_out=aug_anno_out, feat_out=aug_feat_out,
        preset=effective_preset, val_fraction=val_fraction, split_seed=split_seed,
    )
    logger.info(
        f"[{task_id}] Step 4.2.5: ASL hit {asl['n_glosses_hit']}/{len(glosses)} "
        f"({asl['n_glosses_hit']/max(len(glosses),1):.1%}); "
        f"ORG hit {org['n_glosses_hit']}/{len(glosses)} ({org_hit_rate:.1%}); "
        f"concat-aug preset={effective_preset} -> "
        f"train={summary['n_train']} val={summary['n_val']} dropped={summary['n_dropped']}"
    )
    return aug_anno_out, aug_feat_out
```

调用顺序变成：
1. `_extract_clip_features()` (Step 4.1) → `feat_dir`
2. `_generate_annotations()` (Step 4.2) → `anno_dir/{train,val,test}_info_ml.npy`
3. **NEW** `_run_concat_aug()` (Step 4.2.5) → `aug_anno_dir/`, `aug_feat_dir/`（内部解析 ASL + ORG 双源）
4. `_generate_config(anno_dir=aug_anno_dir, feat_dir=aug_feat_dir, ...)` (Step 4.3)
5. `_train_model()` (Step 4.4)

**自动降级机制**：如果 ORG 命中率 < 40%（accuracy 已审词没覆盖到当前 task 的关键 token），自动从 36x 降级到 21x，避免 B 类大量 drop 影响数据质量。

**(c) preset 选择 & 数据来源**

简化设计下，`build_concat_aug` 不再读 `word_lib/<WORD>/` 目录，改为：
- 接收**两个**内存 dict：`gloss_resources_org`（来自 `resolve_org_resources()`） + `gloss_resources_asl`（来自 `resolve_asl_resources()`）
- B 类拼接源 = chatsign-accuracy 已审词视频（reviewer 实录）
- C 类拼接源 = ASL-27K 字典视频（已经过 hiya pipeline）

| Preset | N_A | N_B | N_C | 总放大倍数 | 需要 |
|---|---|---|---|---|---|
| **36x（默认）** | 10 | 10 | 15 | 1 + 35 = 36 | #3a + #3b 都做完（本文档计划全部完成） |
| **21x（自动降级）** | 10 | 0 | 10 | 1 + 20 = 21 | ORG 命中率 < 40% 时自动退化（不需要 B 类） |

**默认走 36x**，命中率不达标时 worker 自动降级到 21x，无需人工干预。

### 3.4 spamo env 依赖

升级后的 `spamo_segement` submodule 里的 colent 模型类 import 与 base 一致，应不需要新装包。但建议在 submodule 升级落地后做 sanity check：

```bash
conda activate spamo
cd /home/chatsign/lizh/chatsign-auto/spamo_segement
python -c "from spamo.t5_slt_colent import FlanT5SLT; print('ok')"
python -c "from spamo.ot_sinkhorn_colent import sinkhorn; print('ok')"
```

### 3.5 验证

A/B 对比流程（建立 baseline → 升级 → 比较）：

1. **建立 base 基线**：在升级 colent 之前，先跑一次 base config (`how2sign_contrastive_single.yaml`) 在选定的 task 数据上，记录最佳 val BLEU 数 `BLEU_base`
2. **升级后跑 colent + 21x preset**：同一份 task 数据，记录 BLEU `BLEU_colent_21x`
3. **门槛**：`BLEU_colent_21x ≥ BLEU_base + 5`（绝对值不强求 test_real 报告的 94.19）
4. **辅助参考**：用 `test_real/phase4_seg_train/selected_100_*.json` 评测集（**只读取数据，不调代码**）做横向对照，确认输出 token 序列合理
5. **下游传导**：P5 切片质量人眼抽查 5-10 条，边界对得上

### 3.6 回滚

**不引入新 env var**。回滚通过 `git revert` 恢复 `phase4_segmentation_train.py`（template 路径回 `how2sign_contrastive_single.yaml`、删 `_run_concat_aug` 调用）。`spamo_segement` submodule 即使已经 fork 升级、含 colent 文件，**旧的 `how2sign_contrastive_single.yaml` 仍然在 submodule 里保留**，所以 base 路径随时可走。

如果担心 colent 训练效果不好想保留对照能力，可以在同一个 task 跑完 colent 后，**手动**用旧 worker 代码（git stash 临时切回去）跑一次 base 做对比 —— 不需要 env var。

---

## 4. 升级 #2：P8 split-level dual-centroid

### 4.1 现状

`backend/workers/phase8_training.py:650, 702`：

```python
train_script = ga_path / "ssl_pretraining_crossvideo_mlp_feature_mean_mean_advance_v4_noconf_clip_nob2b.py"
proto_script = ga_path / "build_prototypes_asl_clip_nob2b.py"
```

单中心、CLIP-based。`gloss_aware` submodule **已经包含** split-level 版本（已确认存在 `ssl_pretraining_glosspose_split_level.py` 和 `build_prototypes_both.py`），只是 worker 没调到。

### 4.2 改动 — `phase8_training.py`

**实测 gloss_aware HEAD 已经完整支持 split-level**（不需 fork submodule）：

```python
# gloss_aware/ssl_pretraining_glosspose_split_level.py:320, 449
# composite = gloss * 2 + level (0=word, 1=sentence)
levels_list = [1 if n.endswith('_sentence') else 0 for n in name_batch]

# gloss_aware/build_prototypes_both.py:6
# level = 1 if utterance_id ends with '_sentence', else 0 (word)
```

**Level 是从 pkl 文件名后缀（`_word` / `_sentence`）读取的**，不需要 dataset class 单独提供 `composite_label` 字段。而 orchestrator `phase8_training.py:386-440` 的 `_link_videos(... level="word"|"sentence" ...)` **已经在按这个约定打 suffix**：

```python
_link_videos(phase2_output / "videos", "orig_", level="word", filter_prefix="word_")
_link_videos(phase5_output / "segment_videos", "seg_", level="sentence")
_link_videos(phase6_output / "word", "aug_word_", level="word")
_link_videos(phase6_output / "segment", "aug_segment_", level="sentence")
_link_videos(phase7_output / "aug_segment_videos", "augseg_", level="sentence")
```

**结论**：P8 升级**就是改两条脚本路径 + 加 `--skip-single` 参数**，不需要 fork gloss_aware、不需要改 dataset 接口、不需要 worker 准备 pkl 时加 level 字段（已经在做）。`torchrun` 入口、Python 解释器、conda env 全部不动。

**(a) 训练脚本路径**

```python
# line ~650:
train_script = ga_path / "ssl_pretraining_glosspose_split_level.py"
```

`torchrun` 入口保持现状：`Path(sys.executable).parent / "torchrun"`（slrt1 env 自带）。

**(b) 训练参数核对**

split_level 脚本可能有新增 / 重命名的参数。在切换前跑：

```bash
conda activate slrt1
cd /home/chatsign/lizh/chatsign-auto/gloss_aware
python ssl_pretraining_glosspose_split_level.py --help
python build_prototypes_both.py --help
```

按 help 输出对照 worker 现在的 `train_cmd` 参数（`--epochs 150 --batch-size 128 --hidden-dim 256 --block-size 6` 等）有无差异。通用 hyperparam 应该保留，只补 split-level 新增的（如 `--num-classes` 改算法等）。

**(c) 原型构造脚本**

```python
# line ~702:
proto_script = ga_path / "build_prototypes_both.py"
```

`build_prototypes_both.py` 要求 `--skip-single` 开关。worker 里 `proto_cmd` 要补：

```python
proto_cmd = [
    sys.executable, str(proto_script),
    "--dataset", dataset_name,
    "--ckpt", str(best_ckpt.resolve()),
    "--output-dir", str(proto_dir.resolve()),
    "--l2norm",
    "--skip-single",  # NEW
]
```

### 4.3 dataset 准备 — 已就绪，无需额外改动

split_level 脚本对每个样本计算 `composite_label = gloss_id × 2 + level`。`level` 由 pkl 文件名后缀决定（`_word` → 0，`_sentence` → 1），不需要 dataset class 单独提供字段。

orchestrator 现状（`phase8_training.py:386-440`）已经在按这个约定打 suffix（见 §4.2 引用的 `_link_videos(...level="word"|"sentence"...)` 调用）。pkl 经 `pose_extractor.py` → `filter_pose_pkls.py` → `batch_norm_cosign_unified.py` 处理后保留原 stem 后缀，最终落到 `build_prototypes_both.py` / `ssl_pretraining_glosspose_split_level.py` 的输入时 level 标签自然存在。

**含义**：升级时**不改 worker 的 pkl 准备逻辑**，整链路自动跑通。

### 4.4 验证

- 训练 log 里 `composite_label` 出现，证明 split-level 走通了
- `prototypes/` 输出文件包含两套中心
- 推理时词级和句级各自的 top-1 准确率比单中心基线持平或更高

### 4.5 回滚

**不引入新 env var**。`gloss_aware` submodule HEAD 同时包含新旧两套脚本（split-level 和 crossvideo_mlp），所以回滚通过 `git revert` 恢复 `phase8_training.py:650, 702` 即可，submodule 不动。

旧 ckpt 在 task 输出目录里保留（不被 submodule 升级影响），可继续用作对照评估。

---

## 5. 总体实施时间线（推荐节奏）

依赖链：`#3 → #1 → #2`。

| 周 | 任务 | 验收标准 |
|---|---|---|
| W1 day 1 | **#3a 预热 ASL-27K 特征**：写 `precompute_asl27k_features.py`，本机 5090 Laptop 夜里挂跑一次 | 27090 个 `*_s2wrapping.npy` 全部落到 `clip_features/ASL-final-27K-202603/videos/` |
| W1 day 2 上午 | **#3b 预热 accuracy 词视频特征**：写 `precompute_accuracy_word_features.py`，本机单卡跑（约 10-15 分钟） | 838 个 `*_s2wrapping.npy` 落到 `clip_features/accuracy_word_uploads/<reviewer>/` |
| W1 day 2 下午 | **#3a + #3b helper**：新增 `asl_resources.py:resolve_asl_resources()` 和 `org_resources.py:resolve_org_resources()`；用同一份典型 task 的 P1 gloss 列表跑通两个 dict 解析 | ASL 命中率 ≥ 80%；ORG 命中率 ≥ 40%；两边 `feat_missing_files == []` |
| W2 | **#1 SpaMo colent + 36x preset**：升级 `spamo_segement` submodule（fork + 推 colent 文件）；移植 `build_concat_aug.py`（双 dict 接口）；改 P4 worker（先跑 base baseline 再跑 colent）；端到端跑一个小 task 比 A/B | 端到端跑通；同数据下 BLEU ≥ base + 5（绝对值待实测，目标接近 94） |
| W3 | **#2 P8 split-level**：先跑 `--help` 核对 split_level 脚本参数；改 `phase8_training.py:650, 702` 两条路径 + 加 `--skip-single`；端到端跑通 | P8 端到端跑通，`prototypes/word_*.pkl` + `prototypes/sentence_*.pkl` 都生成 |
| W4 | 全管线 (e.g. all-glosses) 端到端跑通；新版 vs 旧版做 A/B 对比 | 最终识别准确率提升；`git revert` 演练 worker 仍能跑 base 路径 |

---

## 6. 风险与回滚

| 风险 | 触发场景 | 缓解 |
|---|---|---|
| **B/C 数据源差异** | test_real 用 Uni-Sign（多词短语录像）拼出 94.19；我们 B 类用 accuracy 实录词、C 类用 ASL-27K 字典 —— 数据风格组合不同 | **接受不确定性**：实施 #1 后做 A/B 对比（base vs colent + 36x），目标 BLEU ≥ base + 5；如远低于 94，再讨论是否引入 Uni-Sign |
| **accuracy 词命中率不足** | P1 输出的 token 在 1806 词覆盖外（非常用词 / 多词短语）→ B 类多 drop | 自动降级机制：worker 内部 ORG 命中率 < 40% 时自动转 21x preset，避免 B 类大量 drop 影响数据质量 |
| `spamo_segement` submodule fork 后维护成本 | 主 spamo_segement 仓后续有更新需要 merge | fork 分支命名清晰 (`chatsign-auto/colent`)；定期 rebase upstream |
| spamo env 缺包 | colent 模型 import 失败 | submodule 升级落地后跑 `python -c "from spamo.t5_slt_colent import *"` |
| split_level 训练参数与现 worker 不兼容 | `--help` 输出新增必填参数 | 实施前先跑 `--help`（详见 §4.2 (b)），按需补 `train_cmd` / `proto_cmd` |
| ASL-27K 数据未挂载 | `/mnt/data/chatsign-auto-videos` 不可访问（盘掉了 / 权限问题） | `dataset_videos.py:_load_asl27k_gloss_map` 已有 fail-fast 警告；新 worker 沿用 |
| word-glosses.json 未来 schema 变化 | accuracy 后端如果调整 metadata 字段（如改 `alternate_words` 名字） | `org_resources.py` 的字段引用集中在一处；监测到变化时改一处即可 |
| 移植脚本与 test_real 行为漂移 | 后续 test_real 升级后 orchestrator 没跟上 | 定期对照 `selected_100_*.json` 做回归 BLEU 对比；偏差超 1pt 时 audit |
| 移植代码残留 test_real 引用 | review 漏掉 `from test_real...` 或 `sys.path.insert` | CI 加 lint（见 §7） |

**回滚策略**：**不引入新的环境变量**。回滚通过 `git revert` 三个 worker 改动 + 必要时 submodule pin 回退实现：

| 升级项 | 回滚方式 |
|---|---|
| #3 ASL 资源解析 | 不需单独回滚 —— 它只是被 #1 调用的 helper，#1 回滚后自然不再被调用；预热好的缓存保留不删 |
| #1 P4 colent | `git revert` `phase4_segmentation_train.py`（template 路径回 base + 删 `_run_concat_aug` 调用）。submodule 升级后旧 `how2sign_contrastive_single.yaml` 仍在 submodule 里，base 路径仍可走 |
| #2 P8 split-level | `git revert` `phase8_training.py:650, 702`。`gloss_aware` HEAD 同时含新旧两套脚本，submodule 不需回退 |

如果担心训练效果不如 base 想保留对照能力，**手动跑两次**（先 git stash 切回旧版跑一次，再切回新版跑一次），不需要 env var。

---

## 7. 验收 checklist

- [ ] **代码隔离**：`grep -rE "from\s+test_real|import\s+test_real|sys\.path.*test_real|subprocess.*test_real" backend/` 0 命中（"# 参考 test_real" 这种注释允许）
- [ ] **submodule pin**：`spamo_segement` HEAD 含 colent/e1e2 文件，推上 fork 分支；主仓 submodule pin 已更新
- [ ] **gloss_aware 不动**：submodule HEAD 与升级前一致（`git submodule status gloss_aware` 无 `+` 标记）
- [ ] **conda env 不动**：`spamo` / `slrt1` 这两个 env 中 PyTorch / Lightning / open_clip / omegaconf / sentencepiece 等核心包版本不变（无关包系统级变化忽略）
- [ ] **ASL-27K 预热完整**：`ls /mnt/data/chatsign-auto-videos/clip_features/ASL-final-27K-202603/videos/*.npy | wc -l` ≥ 27090（容许少量 failed）
- [ ] **accuracy 词预热完整**：`find /mnt/data/chatsign-auto-videos/clip_features/accuracy_word_uploads -name "*.npy" | wc -l` ≥ 838（容许少量 failed）
- [ ] **#3a 跑通**：给定一个 gloss 列表（取 all-glosses task 的 P1 输出 `glosses.json`），调 `resolve_asl_resources(glosses)` 满足 `n_glosses_hit / len(glosses) ≥ 0.8` 且 `feat_missing_files == []`
- [ ] **#3b 跑通**：同 gloss 列表调 `resolve_org_resources(glosses)` 满足 `n_glosses_hit / len(glosses) ≥ 0.4` 且 `feat_missing_files == []`
- [ ] **#1 (P4) 跑通**：先跑 base config 拿到 `BLEU_base`，再跑 colent + 36x preset 拿 `BLEU_colent`；`BLEU_colent ≥ BLEU_base + 5`（绝对值待实测，目标接近 94）；如自动降级到 21x，`BLEU_colent ≥ BLEU_base + 4`
- [ ] **#2 (P8) 参数兼容性核对**：`gloss_aware/ssl_pretraining_glosspose_split_level.py --help` 与 `build_prototypes_both.py --help` 输出对照，worker 现有 `train_cmd` / `proto_cmd` 都对得上（gloss_aware HEAD 已支持 split-level via filename suffix，不需要 fork）
- [ ] **#2 (P8) 跑通**：`prototypes/word_*.pkl` 和 `prototypes/sentence_*.pkl` 都生成
- [ ] **P5 切片**肉眼抽查 5-10 条，边界合理
- [ ] **all-glosses task 全管线**：用新 pipeline 重跑，最终识别准确率 ≥ 旧版
- [ ] **回滚演练**：`git revert` `phase4_segmentation_train.py` 后 base 路径仍能跑通；`git revert` `phase8_training.py` 后单中心路径仍能跑通

---

## 8. 附录

### 8.1 文件路径速查

**要改的 orchestrator workers**
- `backend/workers/phase4_segmentation_train.py` ← #1（template 路径 + 新增 `_run_concat_aug` 子步骤 4.2.5）
- `backend/workers/phase8_training.py` ← #2（两条脚本路径，待 dataset 接口验证后决定是否还要加 level 字段）

**新增 scripts（chatsign-auto 自身代码树）**
- `backend/scripts/precompute_asl27k_features.py` ← #3a 一次性离线预热脚本（27090 mp4，5-9 hr）
- `backend/scripts/precompute_accuracy_word_features.py` ← #3b 一次性预热脚本（838 mp4，10-15 min）
- `backend/scripts/asl_resources.py` ← #3a helper，`resolve_asl_resources(glosses) -> dict`
- `backend/scripts/org_resources.py` ← #3b helper，`resolve_org_resources(glosses) -> dict`（读 word-glosses.json + review-decisions.jsonl）
- `backend/scripts/build_concat_aug.py` ← 移植自 test_real/preprocess/06b_build_concat_aug.py，输入接口改为**双 dict**（`gloss_resources_org` + `gloss_resources_asl`）

**不实现 / 不新增**
- ~~`backend/workers/phase2a_import_asl.py`~~ ← 简化设计删除（无独立 worker）
- ~~`backend/scripts/import_asl_clips.py`~~ ← 简化设计删除（功能并入 `asl_resources.py`）
- `backend/scripts/extract_word_features.py` ← 不实现（拆为两个 precompute_* 脚本，分别处理 ASL-27K 和 accuracy 两个数据源）

**升级的 submodule**
- `spamo_segement/configs/chatsign_concat_aug_colent.yaml` (NEW)
- `spamo_segement/configs/chatsign_concat_aug_e1e2.yaml` (NEW)
- `spamo_segement/configs/chatsign_concat_aug.yaml` (NEW)
- `spamo_segement/spamo/t5_slt_colent.py` (NEW)
- `spamo_segement/spamo/t5_slt_e1e2.py` (NEW)
- `spamo_segement/spamo/ot_sinkhorn_colent.py` (NEW)
- `spamo_segement/spamo/ot_sinkhorn_e1e2.py` (NEW)

**已有不动的 submodule（仅 worker 改调用）**
- `gloss_aware/ssl_pretraining_glosspose_split_level.py`
- `gloss_aware/build_prototypes_both.py`

**test_real（仅作参考标杆，运行时不调用）**
- `test_real/preprocess/*.py` ← 移植源代码参考
- `test_real/phase4_seg_train/configs/*.yaml` + `spamo/*.py` ← 移植源代码参考
- `test_real/phase4_seg_train/selected_100_*.json` + `test_30_manifest.json` ← 评测数据（**只读**对照）
- `test_real/README.md` ← BLEU 数据 + 命令行参数对照

### 8.2 BLEU 数据出处

来自 `test_real/README.md`：
- 第 363 行：`aug36x_new_ep65_bleu85.58.ckpt` (base + 36x)
- 第 437 行：colent variant — **BLEU 94.19 @ ep207** (chatsign 21x test, trained-end eps=0.06 inference)
- 第 450 行：e1e2 variant — BLEU 92.71 @ ep209

### 8.3 不在本文档范围内（明确不做）

- **P1 pseudo-gloss 算法升级** — orchestrator 后接 accuracy 人工审，人工修正比算法升级更可靠
- **P2.5b 帧像素去重 (test_real 02b)** — 用户确认 chatsign-accuracy 录制视频未出现卡顿 / 重复帧问题（决策日期 2026-05-03）
- **P3 标准视频生成** — UniSignMimicTurbo 路径两边一致
- **P6 GUAVA 240× 增广** — 共用 `guava-aug` submodule，已经一致
- **P7 augmented-segment 切片** — 纯 ffmpeg，复用 P5 split_points，无算法差异
- **web/服务层**（FastAPI、accuracy 网站、publish 流程） — 本次不动
- **conda env 升级**（Python 3.10 / 重装 PyTorch / 新建 env） — 用户确认环境保持现状（决策日期 2026-05-03）
- **gloss_aware submodule fork 升级**（拿 test_real 那版的 `--adaptive-schedule`） — 接受 HEAD 现状以避免 Python 3.10 依赖
- **物化 word_lib 目录树** — 简化设计走 in-memory dict，不在每个 task 下生成 `word_lib/<WORD>/{org,asl}_*.mp4` 软链树

### 8.4 后续讨论项 / 待用户确认

- **`spamo_segement` fork 仓位置**（GitHub org / 分支命名约定）
- **accuracy 词视频质量审计**：1969 个 mp4 中 838 个已 approved，但是否所有 reviewer 录制风格 / 光线 / 角度都满足 SpaMo 训练要求？建议预热完后抽 5-10 条肉眼看
- **多词 token 命中率**：P1 输出可能含 `MORE_THAN` 这种多词 token，accuracy 词视频 alternate_words 字段是否包含多词形态？需实测
- **batch 数据敏感性**：1806 词覆盖率对不同 batch 可能不同（如 school batch 命中率高，但医学 batch 可能低）；自动降级到 21x 已 mitigate，但需监控
- **e1e2 变体**是否值得作为 colent 的 ablation 跑出
- **CI lint 规则**：是否加 `grep` 规则禁止 backend/ 引用 test_real
- **test_real submodule** 是否最终从 chatsign-auto 移除（移植完成 + 评测数据备份后即可）
