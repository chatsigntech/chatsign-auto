#!/bin/bash
# 4-stage DGX pipeline test (skipping mimicmotion), STAGE-PARALLEL:
#   each stage submits all videos to SLURM at once → all 10 jobs run in parallel
#   on whichever GPU nodes are free; stages serialize via wait-for-all.
#
#   1. trim_inactive          (FILTER_HEAD_TAIL=true, FILTER_DUPLICATE=false, FILTER_POSE=false)
#   2. tail_glitch            (rtmlib body+hand+wrist multi-signal trim)
#   3. RVM                    (BG=white)
#   4. RealESR                (OUTSCALE=2.0)
#
# Usage:
#   ./run_pipeline.sh <input.mp4> [<input2.mp4> ...]
set -euo pipefail

DGX=dgx-login
TASK_ROOT=/media/cvpr/zhewen/api_tasks
FILTER_SBATCH=/media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/infer_dgx_filter.sh
TG_SBATCH=/media/cvpr/zhewen/cv/tail_glitch/infer_dgx_tail_glitch.sh
RVM_SBATCH=/media/cvpr/zhewen/cv/RVM/infer_dgx_rvm.sh
SR_SBATCH=/media/cvpr/zhewen/cv/RealESR/infer_dgx_realesr.sh
EXTRA_EXCLUDE=ADUAED21029WKLX12,ADUAED21043WKLX04,ADUAED21026WKLX26
THIS_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR=$THIS_DIR/outputs
LOG_DIR=$THIS_DIR/logs
mkdir -p "$OUT_DIR" "$LOG_DIR"

[ $# -eq 0 ] && { echo "usage: $0 <input.mp4> [...]"; exit 2; }

# Manifest columns: base \t task_id \t src \t remote
MANIFEST=$LOG_DIR/parallel_manifest.tsv
: > "$MANIFEST"

ts() { date '+%H:%M:%S'; }

# Wait for a comma-separated job list to fully drain from squeue.
wait_jobs() {
  local label=$1 csv=$2
  local elapsed=0
  while :; do
    local rem
    rem=$(ssh -q "$DGX" "squeue -j $csv -h -o '%i' 2>/dev/null | wc -l")
    [ "$rem" = "0" ] && break
    elapsed=$((elapsed + 30))
    echo "  [$label] still running ($rem queued/active, ${elapsed}s)"
    sleep 30
  done
}

# ---- Phase 0: setup remote dirs + upload inputs in parallel ----
echo "[$(ts)] === Phase 0: prepare $# remote task dirs + upload ==="
UPLOAD_PIDS=()
for src in "$@"; do
  base=$(basename "$src" .mp4)
  task_id=$(uuidgen)
  remote="$TASK_ROOT/$task_id"
  printf '%s\t%s\t%s\t%s\n' "$base" "$task_id" "$src" "$remote" >> "$MANIFEST"
  (
    ssh -q "$DGX" "mkdir -p $remote/output"
    scp -q "$src" "$DGX:$remote/input_video.mp4"
  ) &
  UPLOAD_PIDS+=($!)
done
for pid in "${UPLOAD_PIDS[@]}"; do wait $pid; done
echo "[$(ts)]   uploaded $#"

# ---- Phase 1: filter (parallel sbatch) ----
echo "[$(ts)] === Phase 1: filter (FILTER_HEAD_TAIL=true) ==="
FILTER_JOBS=""
# --exclude on the wrapper because the upstream filter sbatch (mimicmotion repo)
# doesn't bake in the bad-node list — the other 3 stage sbatches do.
while IFS=$'\t' read -r base task_id src remote; do
  job=$(ssh -q "$DGX" "sbatch --parsable --chdir=/tmp \
    --exclude=$EXTRA_EXCLUDE \
    --export=ALL,TASK_ID=$task_id,FILTER_HEAD_TAIL=true,FILTER_DUPLICATE=false,FILTER_POSE=false \
    $FILTER_SBATCH")
  printf '%s\t%s\n' "$base" "$job" >> "$LOG_DIR/jobs_filter.tsv"
  FILTER_JOBS="${FILTER_JOBS:+$FILTER_JOBS,}$job"
  echo "  filter $base -> job=$job"
done < "$MANIFEST"
wait_jobs filter "$FILTER_JOBS"

# Pull all filter outputs (parallel scp)
echo "[$(ts)]   pulling filter outputs..."
PULL_PIDS=()
while IFS=$'\t' read -r base task_id src remote; do
  scp -q "$DGX:$remote/output/input.mp4" "$OUT_DIR/${base}_filter.mp4" &
  PULL_PIDS+=($!)
done < "$MANIFEST"
for pid in "${PULL_PIDS[@]}"; do wait $pid; done

# ---- Phase 2: tail_glitch (parallel sbatch) ----
echo "[$(ts)] === Phase 2: tail_glitch ==="
TG_JOBS=""
while IFS=$'\t' read -r base task_id src remote; do
  job=$(ssh -q "$DGX" "sbatch --parsable --chdir=/tmp \
    --export=ALL,INPUT_VIDEO=$remote/output/input.mp4,OUTPUT_VIDEO=$remote/tail_glitch.mp4 \
    $TG_SBATCH")
  printf '%s\t%s\n' "$base" "$job" >> "$LOG_DIR/jobs_tg.tsv"
  TG_JOBS="${TG_JOBS:+$TG_JOBS,}$job"
  echo "  tail_glitch $base -> job=$job"
done < "$MANIFEST"
wait_jobs tail_glitch "$TG_JOBS"

# Pull tail_glitch outputs
echo "[$(ts)]   pulling tail_glitch outputs..."
PULL_PIDS=()
while IFS=$'\t' read -r base task_id src remote; do
  scp -q "$DGX:$remote/tail_glitch.mp4" "$OUT_DIR/${base}_tg.mp4" &
  PULL_PIDS+=($!)
done < "$MANIFEST"
for pid in "${PULL_PIDS[@]}"; do wait $pid; done

# ---- Phase 3: RVM (parallel sbatch) ----
echo "[$(ts)] === Phase 3: RVM (BG=white) ==="
RVM_JOBS=""
while IFS=$'\t' read -r base task_id src remote; do
  job=$(ssh -q "$DGX" "sbatch --parsable --chdir=/tmp \
    --export=ALL,INPUT_VIDEO=$remote/tail_glitch.mp4,OUTPUT_VIDEO=$remote/rvm.mp4,BG=white \
    $RVM_SBATCH")
  printf '%s\t%s\n' "$base" "$job" >> "$LOG_DIR/jobs_rvm.tsv"
  RVM_JOBS="${RVM_JOBS:+$RVM_JOBS,}$job"
  echo "  rvm $base -> job=$job"
done < "$MANIFEST"
wait_jobs rvm "$RVM_JOBS"

# Pull RVM outputs
echo "[$(ts)]   pulling rvm outputs..."
PULL_PIDS=()
while IFS=$'\t' read -r base task_id src remote; do
  scp -q "$DGX:$remote/rvm.mp4" "$OUT_DIR/${base}_rvm.mp4" &
  PULL_PIDS+=($!)
done < "$MANIFEST"
for pid in "${PULL_PIDS[@]}"; do wait $pid; done

# ---- Phase 4: RealESR (parallel sbatch) ----
echo "[$(ts)] === Phase 4: RealESR (OUTSCALE=2.0) ==="
SR_JOBS=""
while IFS=$'\t' read -r base task_id src remote; do
  job=$(ssh -q "$DGX" "sbatch --parsable --chdir=/tmp \
    --export=ALL,INPUT_VIDEO=$remote/rvm.mp4,OUTPUT_VIDEO=$remote/sr.mp4,OUTSCALE=2.0 \
    $SR_SBATCH")
  printf '%s\t%s\n' "$base" "$job" >> "$LOG_DIR/jobs_sr.tsv"
  SR_JOBS="${SR_JOBS:+$SR_JOBS,}$job"
  echo "  realesr $base -> job=$job"
done < "$MANIFEST"
wait_jobs realesr "$SR_JOBS"

# Pull SR outputs
echo "[$(ts)]   pulling sr outputs..."
PULL_PIDS=()
while IFS=$'\t' read -r base task_id src remote; do
  scp -q "$DGX:$remote/sr.mp4" "$OUT_DIR/${base}_sr.mp4" &
  PULL_PIDS+=($!)
done < "$MANIFEST"
for pid in "${PULL_PIDS[@]}"; do wait $pid; done

# ---- Phase 5: local post-processing (transcode + DB) ----
echo "[$(ts)] === Phase 5: transcode H.264 + write phase3-test rows ==="
bash "$THIS_DIR/transcode_h264.sh" >> "$LOG_DIR/transcode.log" 2>&1 || true
# DB engine uses a relative path under chatsign-auto/data/tasks.db; cd there.
( cd "$THIS_DIR/.." && /home/chatsign/miniconda3/envs/chatsign/bin/python \
  "$THIS_DIR/insert_to_phase3_test_db.py" >> "$LOG_DIR/insert.log" 2>&1 ) || true

echo "[$(ts)] ALL DONE."
