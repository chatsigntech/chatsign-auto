#!/bin/bash
# Persistent retry loop for the 8 stubborn letters using the repo-standard
# infer_dgx_total.sh. Keeps resubmitting failures until all 8 land or
# MAX_ROUNDS hit.
set -euo pipefail

DGX=dgx-login
TASK_ROOT=/media/cvpr/zhewen/api_tasks
SBATCH_SCRIPT=/media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/infer_dgx_total.sh
REF_IMG=/media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/data/ref_images/test4.jpg
SRC_DIR=/home/chatsign/lizh/chatsign-auto/chatsign-accuracy/backend/data/uploads/videos/Tareq
OUT_DIR=/mnt/data/chatsign-generated-videos/letters
LOG_DIR=/home/chatsign/lizh/chatsign-auto/logs/letters_8_retry
mkdir -p "$OUT_DIR" "$LOG_DIR"
MAX_ROUNDS=20

declare -A SID=(
  [a]=0 [q]=16 [r]=17 [u]=20 [v]=21 [x]=23 [y]=24 [z]=25
)
PENDING=(a q r u v x y z)

submit_letter() {
  local letter=$1 src task_id remote job_id
  src=$(ls "$SRC_DIR"/letters_${SID[$letter]}_en_*.mp4 2>/dev/null | head -1)
  task_id=$(uuidgen)
  remote="$TASK_ROOT/$task_id"
  ssh -q "$DGX" "mkdir -p $remote/videos $remote/output && cp -f $REF_IMG $remote/input_image.png"
  scp -q "$src" "$DGX:$remote/videos/input.mp4"
  job_id=$(ssh -q "$DGX" "sbatch --parsable --chdir=/tmp --export=ALL,TASK_ID=$task_id $SBATCH_SCRIPT")
  echo "$letter $task_id $job_id"
}

for round in $(seq 1 $MAX_ROUNDS); do
  if [ "${#PENDING[@]}" -eq 0 ]; then break; fi
  MAP="$LOG_DIR/round${round}_map.tsv"
  : > "$MAP"
  echo "[$(date '+%H:%M:%S')] === Round $round: ${#PENDING[@]} letters [${PENDING[*]}] ==="
  for letter in "${PENDING[@]}"; do
    line=$(submit_letter "$letter")
    echo "$line" >> "$MAP"
    echo "  $line"
  done
  JOBIDS=$(awk '{print $3}' "$MAP" | tr '\n' ',' | sed 's/,$//')
  while :; do
    rem=$(ssh -q "$DGX" "squeue -j $JOBIDS -h -o '%i' 2>/dev/null | wc -l")
    [ "$rem" = "0" ] && break
    sleep 60
  done
  NEW_PENDING=()
  while IFS=' ' read -r letter task_id job_id; do
    out_remote="$TASK_ROOT/$task_id/output_filter/input.mp4"
    out_local="$OUT_DIR/${letter}.mp4"
    if scp -q "$DGX:$out_remote" "$out_local" 2>/dev/null; then
      echo "  ${letter}.mp4 OK ($(stat -c%s "$out_local") B)"
    else
      echo "  ${letter} FAILED — retry"
      NEW_PENDING+=("$letter")
    fi
  done < "$MAP"
  PENDING=("${NEW_PENDING[@]}")
done

if [ "${#PENDING[@]}" -gt 0 ]; then
  echo "[$(date '+%H:%M:%S')] HIT MAX_ROUNDS, still pending: ${PENDING[*]}"
else
  echo "[$(date '+%H:%M:%S')] ALL 26 LETTERS NOW UNIFIED via infer_dgx_total.sh"
fi
