#!/bin/bash
# Local 4-stage pipeline (no DGX, no SLURM). Mirrors run_pipeline.sh's chain
# but runs everything on the local GPU with the mimicmotion_dgx env.
#
#   1. trim_inactive   (frames-dir flow → wrapped video-in / video-out)
#   2. tail_glitch     (detect_tail_glitch.py --apply)
#   3. RVM             (white background, fp16, mobilenetv3)
#   4. RealESR         (OUTSCALE=2.0)
#
# Requires:
#   - /home/chatsign/lizh/cv_local/{RVM,RealESR}/ (rsync'd from DGX)
#   - /home/chatsign/lizh/cv_local/realesr_extras/ (pip --no-deps target)
#   - mimicmotion_dgx env with torch+CUDA, cv2, rtmlib, av
#
# Usage: ./run_pipeline_local.sh <input.mp4> [<input2.mp4> ...]
set -euo pipefail

PY=/home/chatsign/miniconda3/envs/mimicmotion_dgx/bin/python
CV_LOCAL=/home/chatsign/lizh/cv_local
EXTRAS=$CV_LOCAL/realesr_extras
RVM_DIR=$CV_LOCAL/RVM
SR_DIR=$CV_LOCAL/RealESR
UNISIGN=/home/chatsign/lizh/chatsign-auto/UniSignMimicTurbo

THIS_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR=$THIS_DIR/outputs_local
LOG=$THIS_DIR/logs/local_$(date +%H%M%S).log
mkdir -p "$OUT_DIR" "$THIS_DIR/logs"

[ $# -eq 0 ] && { echo "usage: $0 <input.mp4> [...]"; exit 2; }

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

for src in "$@"; do
  base=$(basename "$src" .mp4)
  log "=== $base ==="

  # Per-video scratch dir for stages 1 + 2; cleaned at end of iteration.
  SCRATCH=$(mktemp -d -t local_pipe.XXXXXX)

  # Stage 1: trim_inactive (extract → trim → reassemble).
  # The upstream script is frame-folder-based, so we wrap it with cv2.
  log "  [1/4] trim_inactive (extract → trim → reassemble)"
  PYTHONPATH="$UNISIGN" "$PY" "$THIS_DIR/_trim_inactive_video.py" \
    "$src" "$OUT_DIR/${base}_trim.mp4" --work "$SCRATCH/trim" >>"$LOG" 2>&1

  # Stage 2: tail_glitch. Output goes to a sibling temp dir so it doesn't
  # clobber the stage-1 file (detect_tail_glitch writes <apply-dir>/<input-basename>).
  log "  [2/4] tail_glitch (rtmlib body+hand+wrist)"
  TG_APPLY="$SCRATCH/tg"
  mkdir -p "$TG_APPLY"
  RTMLIB_PARENT="$UNISIGN" "$PY" "$THIS_DIR/detect_tail_glitch.py" \
    "$OUT_DIR/${base}_trim.mp4" \
    --apply --apply-dir "$TG_APPLY" \
    --out "$THIS_DIR/logs/tg_${base}.tsv" >>"$LOG" 2>&1
  if [ -f "$TG_APPLY/${base}_trim.mp4" ]; then
    mv "$TG_APPLY/${base}_trim.mp4" "$OUT_DIR/${base}_tg.mp4"
  else
    cp "$OUT_DIR/${base}_trim.mp4" "$OUT_DIR/${base}_tg.mp4"
  fi
  rm -rf "$SCRATCH"

  # Stage 3: RVM (white bg, fp16, mobilenetv3)
  log "  [3/4] RVM (white bg, fp16)"
  cd "$RVM_DIR"
  "$PY" matting_video.py \
    --input "$OUT_DIR/${base}_tg.mp4" \
    --output "$OUT_DIR/${base}_rvm.mp4" \
    --bg white --variant mobilenetv3 --downsample-ratio 0.4 --fp16 >>"$LOG" 2>&1

  # Stage 4: RealESR 2x
  log "  [4/4] RealESR (2x → 1152²)"
  cd "$SR_DIR"
  PYTHONPATH="$EXTRAS" "$PY" "$THIS_DIR/_upscale_video_local.py" \
    "$OUT_DIR/${base}_rvm.mp4" "$OUT_DIR/${base}_sr.mp4" 2.0 >>"$LOG" 2>&1

  log "  done -> $OUT_DIR/${base}_sr.mp4 ($(stat -c%s "$OUT_DIR/${base}_sr.mp4") B)"
done

log "ALL DONE. log: $LOG"
