#!/bin/bash
# Production 4-stage DGX pipeline for ASL-27K-middle (post-MimicMotion outputs).
# Same algorithm as run_pipeline.sh; differences are all in the orchestration:
#   - batched submission (BATCH_SIZE) so we never exceed SLURM MaxJobCount=10000
#   - throttled scp (SCP_PARALLEL) so sshd doesn't get steamrolled
#   - 1 ssh per phase per batch (remote bash loop) instead of N
#   - stable task_id "asl27k-<base>" so resume finds prior remote intermediates
#   - per-video success/failure logged to progress.tsv (no abort on single fail)
#   - intermediates kept locally; remote per-task dirs cleaned per batch
#   - no Phase3TestJob DB writes (27K is for training, not review UI)
#   - H.264 sidecars produced for all SR outputs at the end
#
# Usage:
#   ./run_pipeline_27k.sh <input_dir> [LIMIT]
# LIMIT=0 (default) processes all; LIMIT=100 takes first 100 alphabetically.
set -eo pipefail

INPUT_DIR=${1:?"usage: $0 <input_dir> [limit]"}
LIMIT=${2:-0}

OUT_ROOT=/mnt/data/chatsign-auto-videos/ASL-27K-final
DGX=dgx-login
TASK_ROOT=/media/cvpr/zhewen/api_tasks
EXTRA_EXCLUDE=${EXTRA_EXCLUDE:-ADUAED21029WKLX12,ADUAED21043WKLX04,ADUAED21026WKLX26}
# NODELIST overrides EXTRA_EXCLUDE: only schedule on listed nodes (positive
# selection). Used when most cluster nodes have broken autofs.
NODELIST=${NODELIST:-}
if [ -n "$NODELIST" ]; then
  NODE_FLAG="--nodelist=$NODELIST"
else
  NODE_FLAG="--exclude=$EXTRA_EXCLUDE"
fi
BATCH_SIZE=${BATCH_SIZE:-200}
SCP_PARALLEL=${SCP_PARALLEL:-16}

FILTER_SBATCH=/media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/infer_dgx_filter.sh
TG_SBATCH=/media/cvpr/zhewen/cv/tail_glitch/infer_dgx_tail_glitch.sh
RVM_SBATCH=/media/cvpr/zhewen/cv/RVM/infer_dgx_rvm.sh
SR_SBATCH=/media/cvpr/zhewen/cv/RealESR/infer_dgx_realesr.sh
FFMPEG=/home/chatsign/lizh/chatsign-auto/bin/ffmpeg

mkdir -p "$OUT_ROOT"/{filter,tg,rvm,sr,web,logs}
RUN_DIR="$OUT_ROOT/logs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
PROGRESS="$OUT_ROOT/progress.tsv"
[ -s "$PROGRESS" ] || printf 'ts\tbase\tstatus\tbatch\tjob_filter\tjob_tg\tjob_rvm\tjob_sr\n' > "$PROGRESS"
MAIN_LOG="$RUN_DIR/main.log"

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }

# Wait for a comma-separated job list to fully drain from squeue.
wait_jobs() {
  local label=$1 csv=$2
  [ -z "$csv" ] && return
  local elapsed=0
  while :; do
    local rem
    rem=$(ssh -q "$DGX" "squeue -j $csv -h -o '%i' 2>/dev/null | wc -l")
    [ "$rem" = "0" ] && break
    elapsed=$((elapsed + 30))
    log "    [$label] still running ($rem queued/active, ${elapsed}s)"
    sleep 30
  done
}

# Pull a per-stage output file in parallel.
# Args: manifest, remote_relpath, local_subdir
pull_stage() {
  local manifest=$1 remote_relpath=$2 local_subdir=$3
  while IFS=$'\t' read -r base task_id src remote; do
    printf '%s\0%s\0' "${DGX}:${remote}/${remote_relpath}" "${OUT_ROOT}/${local_subdir}/${base}.mp4"
  done < "$manifest" | xargs -0 -n 2 -P "$SCP_PARALLEL" scp -q 2>/dev/null || true
}

process_batch() {
  local batch_id=$1; shift
  local n=$#
  local bdir="$RUN_DIR/batch_${batch_id}"
  mkdir -p "$bdir"
  local manifest="$bdir/manifest.tsv"
  : > "$manifest"
  for src in "$@"; do
    local base task_id remote
    base=$(basename "$src" .mp4)
    task_id="asl27k-$base"
    remote="$TASK_ROOT/$task_id"
    printf '%s\t%s\t%s\t%s\n' "$base" "$task_id" "$src" "$remote" >> "$manifest"
  done

  log ">>> batch $batch_id ($n videos)"

  # Stage manifest on DGX (one scp + ssh per batch — not per video).
  local manifest_remote="$TASK_ROOT/_run_$$_batch${batch_id}.tsv"
  scp -q "$manifest" "${DGX}:${manifest_remote}"
  ssh -q "$DGX" "awk -F'\t' '{print \$4}' $manifest_remote | xargs -I{} mkdir -p {}/output"

  # Phase 0: upload inputs (parallel scp, throttled).
  log "  [0/4] upload $n inputs"
  while IFS=$'\t' read -r base task_id src remote; do
    printf '%s\0%s\0' "$src" "${DGX}:${remote}/input_video.mp4"
  done < "$manifest" | xargs -0 -n 2 -P "$SCP_PARALLEL" scp -q

  # Phase 1: filter (FILTER_HEAD_TAIL=true, dedup off, pose off — same as 10-video run).
  log "  [1/4] filter sbatch"
  ssh -q "$DGX" "
    while IFS=\$'\t' read -r base task_id src remote; do
      sbatch --parsable --chdir=/tmp $NODE_FLAG \
        --export=ALL,TASK_ID=\$task_id,FILTER_HEAD_TAIL=true,FILTER_DUPLICATE=false,FILTER_POSE=false \
        $FILTER_SBATCH
    done < $manifest_remote
  " > "$bdir/jobs_filter.txt"
  local jobs_filter
  jobs_filter=$(tr '\n' ',' < "$bdir/jobs_filter.txt" | sed 's/,$//')
  wait_jobs filter "$jobs_filter"
  pull_stage "$manifest" "output/input.mp4" "filter"

  # Phase 2: tail_glitch.
  log "  [2/4] tail_glitch sbatch"
  ssh -q "$DGX" "
    while IFS=\$'\t' read -r base task_id src remote; do
      sbatch --parsable --chdir=/tmp $NODE_FLAG \
        --export=ALL,INPUT_VIDEO=\$remote/output/input.mp4,OUTPUT_VIDEO=\$remote/tail_glitch.mp4 \
        $TG_SBATCH
    done < $manifest_remote
  " > "$bdir/jobs_tg.txt"
  local jobs_tg
  jobs_tg=$(tr '\n' ',' < "$bdir/jobs_tg.txt" | sed 's/,$//')
  wait_jobs tg "$jobs_tg"
  pull_stage "$manifest" "tail_glitch.mp4" "tg"

  # Phase 3: RVM (white background).
  log "  [3/4] rvm sbatch"
  ssh -q "$DGX" "
    while IFS=\$'\t' read -r base task_id src remote; do
      sbatch --parsable --chdir=/tmp $NODE_FLAG \
        --export=ALL,INPUT_VIDEO=\$remote/tail_glitch.mp4,OUTPUT_VIDEO=\$remote/rvm.mp4,BG=white \
        $RVM_SBATCH
    done < $manifest_remote
  " > "$bdir/jobs_rvm.txt"
  local jobs_rvm
  jobs_rvm=$(tr '\n' ',' < "$bdir/jobs_rvm.txt" | sed 's/,$//')
  wait_jobs rvm "$jobs_rvm"
  pull_stage "$manifest" "rvm.mp4" "rvm"

  # Phase 4: RealESR 2x.
  log "  [4/4] realesr sbatch"
  ssh -q "$DGX" "
    while IFS=\$'\t' read -r base task_id src remote; do
      sbatch --parsable --chdir=/tmp $NODE_FLAG \
        --export=ALL,INPUT_VIDEO=\$remote/rvm.mp4,OUTPUT_VIDEO=\$remote/sr.mp4,OUTSCALE=2.0 \
        $SR_SBATCH
    done < $manifest_remote
  " > "$bdir/jobs_sr.txt"
  local jobs_sr
  jobs_sr=$(tr '\n' ',' < "$bdir/jobs_sr.txt" | sed 's/,$//')
  wait_jobs sr "$jobs_sr"
  pull_stage "$manifest" "sr.mp4" "sr"

  # H.264 sidecar transcode for THIS batch's SR outputs (browser-playable).
  # Done per-batch so accuracy push (next step) has playable files immediately.
  while IFS=$'\t' read -r base task_id src remote; do
    sr="$OUT_ROOT/sr/${base}.mp4"
    out="$OUT_ROOT/web/${base}.mp4"
    [ -s "$sr" ] || continue
    [ -s "$out" ] && [ "$out" -nt "$sr" ] && continue
    printf '%s\0%s\0' "$sr" "$out"
  done < "$manifest" | xargs -0 -n 2 -P "$SCP_PARALLEL" bash -c '
    '"$FFMPEG"' -y -i "$1" -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
      -movflags +faststart -an "$2" 2>/dev/null || true
  ' _

  # Optional: push completed videos to chatsign-accuracy (env-gated).
  if [ -n "${ACC_PUSH:-}" ]; then
    awk -F'\t' '{print $1}' "$manifest" | \
      /home/chatsign/miniconda3/envs/chatsign/bin/python \
        "$THIS_DIR/push_27k_to_accuracy.py" >> "$RUN_DIR/acc_push.log" 2>&1 || true
  fi

  # Per-video status to progress.tsv.
  paste "$manifest" "$bdir/jobs_filter.txt" "$bdir/jobs_tg.txt" \
        "$bdir/jobs_rvm.txt" "$bdir/jobs_sr.txt" 2>/dev/null \
    | while IFS=$'\t' read -r base task_id src remote jf jt jr js; do
        if [ -s "$OUT_ROOT/sr/${base}.mp4" ]; then status=ok; else status=failed; fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$(ts)" "$base" "$status" "$batch_id" "$jf" "$jt" "$jr" "$js" >> "$PROGRESS"
      done

  # Cleanup remote task dirs for this batch (intermediates already on local).
  ssh -q "$DGX" "
    awk -F'\t' '{print \$4}' $manifest_remote | xargs -r rm -rf
    rm -f $manifest_remote
  "

  local n_ok n_fail
  n_ok=$(awk -F'\t' -v b="$batch_id" '$4==b && $3=="ok"' "$PROGRESS" | wc -l)
  n_fail=$(awk -F'\t' -v b="$batch_id" '$4==b && $3=="failed"' "$PROGRESS" | wc -l)
  log "  batch $batch_id done: ok=$n_ok failed=$n_fail"
}

# === Build TODO ===
mapfile -t ALL_ARR < <(find "$INPUT_DIR" -maxdepth 1 -name '*.mp4' | sort)
if [ "$LIMIT" -gt 0 ] && [ "${#ALL_ARR[@]}" -gt "$LIMIT" ]; then
  ALL_ARR=( "${ALL_ARR[@]:0:$LIMIT}" )
fi
N_ALL=${#ALL_ARR[@]}

TODO_ARR=()
for src in "${ALL_ARR[@]}"; do
  base=$(basename "$src" .mp4)
  [ -s "$OUT_ROOT/sr/${base}.mp4" ] && continue
  TODO_ARR+=("$src")
done
N_TODO=${#TODO_ARR[@]}
N_SKIP=$((N_ALL - N_TODO))

log "input dir: $INPUT_DIR (limit=$LIMIT)"
log "scanned $N_ALL videos; skip $N_SKIP already-done; processing $N_TODO"
log "batch_size=$BATCH_SIZE scp_parallel=$SCP_PARALLEL"
log "out_root=$OUT_ROOT  run_dir=$RUN_DIR"

if [ "$N_TODO" -eq 0 ]; then
  log "nothing to do."
  exit 0
fi

# === Process in batches ===
total_batches=$(( (N_TODO + BATCH_SIZE - 1) / BATCH_SIZE ))
batch_num=0
for ((i=0; i<N_TODO; i+=BATCH_SIZE)); do
  batch_num=$((batch_num + 1))
  batch=( "${TODO_ARR[@]:i:BATCH_SIZE}" )
  log "============ batch $batch_num/$total_batches ============"
  process_batch "$batch_num" "${batch[@]}"
done

n_ok=$(awk -F'\t' '$3=="ok"' "$PROGRESS" | wc -l)
n_fail=$(awk -F'\t' '$3=="failed"' "$PROGRESS" | wc -l)
log "ALL DONE. cumulative: ok=$n_ok failed=$n_fail"
log "logs: $RUN_DIR    progress: $PROGRESS"
