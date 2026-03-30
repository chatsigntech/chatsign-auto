# ChatSign Pipeline — 8-Phase Sign Language Processing

## Pipeline Overview

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7 → Phase 8
采集       注解       整理       换人       处理       插值       增广       训练
```

Pipeline follows the author's flow from [openhe-hub/UniSignMimicTurbo](https://github.com/openhe-hub/UniSignMimicTurbo):
**Person transfer FIRST (on raw videos), THEN filter the transferred output.**

---

## Phase Descriptions

### Phase 1: Video Collection
- **Source**: chatsign-accuracy local filesystem
- **Input**: Approved videos from `pending-videos.jsonl` + `review-decisions.jsonl`
- **Output**: `videos/` (symlinks), `manifest.json`, `sentences.txt`, `gloss.csv`
- **Filter**: Optional `batch_name` (e.g., `school_unmatch`)

### Phase 2: Pseudo-gloss Extraction
- **Tool**: spaCy `en_core_web_sm`
- **Input**: `sentences.txt` from Phase 1
- **Output**: `glosses.json`, `vocab.json` (frequency statistics)
- **Method**: POS filter → keep NOUN, VERB, ADJ, ADV, NUM, PRON, PROPN → uppercase

### Phase 3: Annotation Organization
- **Input**: Phase 1 manifest + Phase 2 glosses
- **Output**: `annotations.json` (merged), `videos/` (symlinks)

### Phase 4: Person Transfer (MimicMotion)
- **Tool**: UniSignMimicTurbo `inference_raw_batch_cache.py`
- **Input**: Raw original videos from Phase 1 (NOT preprocessed)
- **Output**: Person-transferred videos + `phase4_report.json`
- **GPU**: Required (RTX 5090, ~2-5 min/video)
- **Error handling**:
  - Videos < 16 frames: skipped (logged as `skipped_short`)
  - First attempt fails: auto-retry with reduced `num_inference_steps`
  - All attempts fail: excluded from pipeline (logged as `failed`)
  - **Failed/skipped videos do NOT enter subsequent phases**
- **Report**: `phase4_report.json` contains per-video status, frame count, duration, error details

### Phase 5: Video Processing
- **Tool**: UniSignMimicTurbo `scripts/sentence/` scripts
- **Input**: Person-transferred videos from Phase 4
- **Steps**:
  1. Extract frames from transferred videos
  2. Filter duplicate/static frames (threshold 3.0%)
  3. Filter by pose quality (hand ≥ 0.8, head ≥ 0.9)
  4. Resize to 512×320
  5. Extract boundary frames (for Phase 6 interpolation)
  6. Generate cleaned videos from filtered frames

### Phase 6: Frame Interpolation (FramerTurbo)
- **Tool**: UniSignMimicTurbo `FramerTurbo/scripts/inference/cli_infer_576x576.py`
- **Input**: Boundary frames from Phase 5 + cleaned frames
- **Output**: Final sentence-level videos with smooth word transitions
- **Fallback**: If FramerTurbo checkpoint not available, Phase 5 videos pass through directly
- **Required model**: `FramerTurbo/checkpoints/framer_512x320/` (not included in repo)

### Phase 7: Data Augmentation
- **Tool**: guava-aug `cv_aug/`
- **Input**: Final videos from Phase 6
- **Output**: 32x augmented videos
  - 25 types 2D (12 geometric + 13 color)
  - 7 types temporal (5 speed + 2 fps)
- **GPU**: Not required (CPU-only, OpenCV)

### Phase 8: Model Training (gloss_aware)
- **Tool**: gloss_aware scripts
- **Steps**:
  1. Pose extraction (RTMPose) from all videos (original + augmented)
  2. Filter pose PKLs by quality
  3. Normalize keypoints (cosign padding)
  4. SSL pretraining (SignCL)
  5. Build gloss prototypes (VQ codebook + TF-IDF)
- **Output**: `best.pth`, `prototypes.pt`, `vq_codebook.pt`, `gloss_code_stats.pkl`

---

## Error Handling

### Phase 4 Report (`phase4_report.json`)

```json
{
  "results": {
    "sentence_100": {
      "filename": "sentence_100.mp4",
      "video_info": {"frames": 74, "width": 1280, "height": 720, "fps": 30, "duration": 2.5},
      "status": "success",
      "attempts": [{"status": "success", "time": 158.2, "steps": 10}]
    },
    "sentence_3": {
      "filename": "sentence_3.mp4",
      "video_info": {"frames": 12, "width": 1280, "height": 720, "fps": 30, "duration": 0.4},
      "status": "skipped_short",
      "note": "Only 12 frames, need >= 16"
    }
  },
  "summary": {
    "success": 80,
    "retry_success": 5,
    "skipped_short": 41,
    "failed": 18,
    "total_generated": 85,
    "total_excluded": 59
  }
}
```

### Status definitions
| Status | Meaning | Action |
|--------|---------|--------|
| `success` | Transferred on first attempt | Continues to Phase 5+ |
| `retry_success` | Failed first, succeeded with reduced params | Continues to Phase 5+ |
| `skipped_short` | Video too short for MimicMotion | **Excluded from pipeline** |
| `failed` | All retry attempts failed | **Excluded from pipeline** |

---

## Data Flow

```
accuracy data (144 approved videos)
    │
    ├── Phase 1 → 144 videos (symlinks)
    │
    ├── Phase 4 → N transferred videos (N < 144, failures excluded)
    │
    ├── Phase 5 → M cleaned videos (M ≤ N, quality filtered)
    │
    ├── Phase 6 → M final videos (interpolated or passthrough)
    │
    ├── Phase 7 → M × 32 augmented videos
    │
    └── Phase 8 → trained model + prototypes
```

---

## External Dependencies

| Component | Project | Required For |
|-----------|---------|-------------|
| MimicMotion model | `models/MimicMotion_1-1.pth` (2.8GB) | Phase 4 |
| SVD-XT model | `models/svd-xt-1-1/` | Phase 4 |
| FramerTurbo checkpoint | `FramerTurbo/checkpoints/framer_512x320/` | Phase 6 (optional) |
| GUAVA 3D model | `guava-aug/assets/GUAVA/` | Phase 7 3D views (disabled) |
| RTMPose ONNX | `~/.cache/rtmlib/` (auto-downloaded) | Phase 5, 8 |
