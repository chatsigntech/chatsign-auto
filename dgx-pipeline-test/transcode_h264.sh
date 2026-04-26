#!/bin/bash
# Transcode mp4v → H.264 so browsers can play.
# Mirrors originals + outputs into web/ sidecars next to the source.
set -euo pipefail
FFMPEG=/home/chatsign/lizh/chatsign-auto/bin/ffmpeg
TEST_DIR=$(cd "$(dirname "$0")" && pwd)
WEB_DIR=$TEST_DIR/web
mkdir -p "$WEB_DIR"

transcode() {
  local in=$1 out=$2
  if [ -f "$out" ] && [ "$out" -nt "$in" ]; then
    echo "  skip (up to date): $(basename "$out")"
    return 0
  fi
  "$FFMPEG" -hide_banner -loglevel warning -y -i "$in" \
    -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
    -movflags +faststart -an "$out"
  echo "  ok: $(basename "$out") ($(stat -c%s "$out") B)"
}

# Originals (50K) → web/<base>_orig.mp4
while IFS=$'\t' read -r fname word; do
  [ "$fname" = "fname" ] && continue
  base=${fname%.mp4}
  src=$(ls /mnt/data/chatsign-auto-videos/50Kfull_v2/csv_*/"$fname" 2>/dev/null | head -1)
  [ -z "$src" ] && src=$TEST_DIR/inputs/$fname
  [ -f "$src" ] || { echo "  miss orig: $fname"; continue; }
  transcode "$src" "$WEB_DIR/${base}_orig.mp4"
done < "$TEST_DIR/inputs/manifest.tsv"

# SR outputs → web/<base>_sr.mp4
for src in "$TEST_DIR/outputs"/*_sr.mp4; do
  [ -f "$src" ] || continue
  base=$(basename "$src" .mp4)
  transcode "$src" "$WEB_DIR/${base}.mp4"
done

echo "done."
