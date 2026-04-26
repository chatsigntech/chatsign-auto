# dgx-pipeline-test

Self-contained sandbox for verifying the **post-MimicMotion DGX cleanup chain**
on already-generated mimicmotion outputs (no mimic re-run).

## Pipeline (current 4-stage chain)

```
input.mp4
  │
  ▼  ① infer_dgx_filter.sh   (FILTER_HEAD_TAIL=true, FILTER_DUPLICATE=false, FILTER_POSE=false)
       /media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/infer_dgx_filter.sh
       trim_inactive_frames.py  →  output/input.mp4   (cuts pre/post-static frames only)
  │
  ▼  ② infer_dgx_tail_glitch.sh   (rtmlib Wholebody)
       /media/cvpr/zhewen/cv/tail_glitch/infer_dgx_tail_glitch.sh
       detect_tail_glitch.py     →  tail_glitch.mp4   (cuts trailing glitch frames)
  │
  ▼  ③ infer_dgx_rvm.sh         (BG=white)
       /media/cvpr/zhewen/cv/RVM/infer_dgx_rvm.sh     →  rvm.mp4         (white background)
  │
  ▼  ④ infer_dgx_realesr.sh     (OUTSCALE=2.0)
       /media/cvpr/zhewen/cv/RealESR/infer_dgx_realesr.sh →  sr.mp4    (576² → 1152²)
```

**Why DUPLICATE=false:** The upstream dedup (`filter_duplicate_frames.py`,
threshold 3%) drops 25–47% of legitimate slow-motion frames in sign-language
clips. Real "ghost frames" cluster at the tail and are handled instead by
`detect_tail_glitch.py` (rtmlib body/hand confidence + wrist-jump signals).

## Files

| File | Role |
|---|---|
| `run_pipeline.sh` | Orchestrator: per-video, submits the 4 sbatches in series + scp out + transcode + DB push |
| `transcode_h264.sh` | Convert mp4v → H.264 sidecars under `web/` (browser-playable) |
| `insert_to_phase3_test_db.py` | Upsert `phase3_test_jobs` rows so results show on `/phase3-test` |
| `push_to_accuracy.py` | Push final `web/<base>_sr.mp4` into `chatsign-accuracy` for review |
| `detect_tail_glitch.py` | Tail-glitch detector + cv2 trim (the **source** copy; DGX has its own deploy at `/media/cvpr/zhewen/cv/tail_glitch/`) |
| `inputs/manifest.tsv` | 10 sample videos + gloss labels |

## Tail glitch detection rules

Three independent signal categories — flagged as glitch if **≥2** fire (default):

1. **body** — body confidence drops > 13% from middle-of-video reference
2. **hand** — both hands lost / one hand lost + opposite wrist jumps
3. **wrist** — both wrists move > 25% of frame side simultaneously

`MIN_CATEGORIES` tunable via `--min-cats` flag or sbatch env.

## DGX deploy locations (independent of upstream UniSignMimicTurbo)

```
/media/cvpr/zhewen/cv/RealESR/      ← super-resolution
/media/cvpr/zhewen/cv/RVM/          ← background matting
/media/cvpr/zhewen/cv/tail_glitch/  ← tail-glitch trim (deployed from this dir)
```

`detect_tail_glitch.py` is **cv2-only** (no ffmpeg dependency) — works on DGX
aarch64 without any extra binary install. It uses rtmlib at the standard
location `/media/cvpr/zhewen/UniSignMimicTurbo/rtmlib/` via PYTHONPATH.

## Run

```bash
# Full chain on the 10 manifest videos
./run_pipeline.sh inputs/*.mp4

# Single video
./run_pipeline.sh inputs/0nzcdjzng6_hiya.mp4

# Resume after Ctrl+C: re-running with the same args skips any video whose
# outputs/<base>_sr.mp4 is already present. Phase 5 (transcode + DB upsert)
# always runs at the end so the page still reflects newly-completed rows.

# Output naming per video <base>:
#   outputs/<base>_filter.mp4   filter sbatch direct (mp4v)
#   outputs/<base>_tg.mp4       tail_glitch sbatch direct (mp4v)
#   outputs/<base>_rvm.mp4      RVM sbatch direct (H.264)
#   outputs/<base>_sr.mp4       RealESR sbatch direct (mp4v)
#   web/<base>_orig.mp4         H.264 transcode of original (browser-playable)
#   web/<base>_sr.mp4           H.264 transcode of final SR
```

## Detection-only (no actual trim, no sbatch)

```bash
# local dry run
/home/chatsign/miniconda3/envs/mimicmotion_dgx/bin/python detect_tail_glitch.py --dir inputs
```

## Local fallback (no DGX, no SLURM)

```bash
./run_pipeline_local.sh inputs/0nzcdjzng6_hiya.mp4
# outputs_local/<base>_{trim,tg,rvm,sr}.mp4   stage outputs
```

Runs the same 4 stages on the local GPU (RTX 5090, mimicmotion_dgx env)
in ~40 s/video. Requires one-time setup of `~/lizh/cv_local/` (RVM +
RealESR repos rsync'd from DGX, plus a pip `--no-deps` extras dir for
basicsr/realesrgan/gfpgan/facexlib with two compat patches — see
`run_pipeline_local.sh` for env paths).

## Reports

- `tail_glitch_report.tsv` — per-video glitch verdict + numeric metrics
- `all_rejections_batch_20260423.tsv` — accuracy reviewer rejections (separate analysis)
- `rejected_review_batch_20260423.tsv` — same data, video-keyed

## Performance (per video, GB10 spark partition)

| stage | time |
|---|---|
| filter (trim_inactive) | ~30 s |
| tail_glitch (rtmlib) | ~30 s (incl. model load) |
| RVM | ~25 s |
| RealESR | ~15 s |
| **per-video chain total** | **~1:40** |
