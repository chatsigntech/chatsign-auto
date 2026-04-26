#!/bin/bash
# Unified letter generation: all 26 letters via repo-standard infer_dgx_total.sh
# (mimic + trim_inactive + filter_duplicate + filter_pose). Auto-retries letters
# that fail at slurmstepd-chdir or autofs level; max 5 rounds.
set -euo pipefail

DGX=dgx-login
TASK_ROOT=/media/cvpr/zhewen/api_tasks
SBATCH_SCRIPT=/media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/infer_dgx_total.sh
REF_IMG=/media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/data/ref_images/test4.jpg
SRC_DIR=/home/chatsign/lizh/chatsign-auto/chatsign-accuracy/backend/data/uploads/videos/Tareq
OUT_DIR=/mnt/data/chatsign-generated-videos/letters
LOG_DIR=/home/chatsign/lizh/chatsign-auto/logs/letters_unified
mkdir -p "$OUT_DIR" "$LOG_DIR"
MAX_ROUNDS=5

# letter -> sentenceId
declare -A SID=(
  [a]=0 [b]=1 [c]=2 [d]=3 [e]=4 [f]=5 [g]=6 [h]=7 [i]=8 [j]=9
  [k]=10 [l]=11 [m]=12 [n]=13 [o]=14 [p]=15 [q]=16 [r]=17 [s]=18 [t]=19
  [u]=20 [v]=21 [w]=22 [x]=23 [y]=24 [z]=25
)

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

PENDING=(a b c d e f g h i j k l m n o p q r s t u v w x y z)

for round in $(seq 1 $MAX_ROUNDS); do
  if [ "${#PENDING[@]}" -eq 0 ]; then break; fi
  MAP="$LOG_DIR/round${round}_map.tsv"
  : > "$MAP"
  echo "[$(date '+%H:%M:%S')] === Round $round: ${#PENDING[@]} letters -> $MAP ==="
  for letter in "${PENDING[@]}"; do
    line=$(submit_letter "$letter")
    echo "$line" >> "$MAP"
    echo "  $line"
  done

  JOBIDS=$(awk '{print $3}' "$MAP" | tr '\n' ',' | sed 's/,$//')
  echo "[$(date '+%H:%M:%S')] Round $round polling: $JOBIDS"
  while :; do
    rem=$(ssh -q "$DGX" "squeue -j $JOBIDS -h -o '%i' 2>/dev/null | wc -l")
    [ "$rem" = "0" ] && break
    echo "  [$(date '+%H:%M:%S')] running: $rem"
    sleep 60
  done

  # Pull successes; build new PENDING from failures
  echo "[$(date '+%H:%M:%S')] Round $round: pulling outputs..."
  NEW_PENDING=()
  while IFS=' ' read -r letter task_id job_id; do
    out_remote="$TASK_ROOT/$task_id/output_filter/input.mp4"
    out_local="$OUT_DIR/${letter}.mp4"
    if scp -q "$DGX:$out_remote" "$out_local" 2>/dev/null; then
      echo "  ${letter}.mp4 OK ($(stat -c%s "$out_local") B)"
    else
      echo "  ${letter} FAILED — will retry next round"
      NEW_PENDING+=("$letter")
    fi
  done < "$MAP"
  PENDING=("${NEW_PENDING[@]}")
done

if [ "${#PENDING[@]}" -gt 0 ]; then
  echo "[$(date '+%H:%M:%S')] DONE with permanent failures: ${PENDING[*]}"
else
  echo "[$(date '+%H:%M:%S')] DONE — all 26 letters generated via repo-standard infer_dgx_total.sh"
fi
