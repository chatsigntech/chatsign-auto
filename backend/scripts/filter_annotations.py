"""Filter OpenASL and How2Sign annotation files to keep only entries with existing videos.

Reads the original TSV/CSV annotations, checks each entry against the video directory,
and writes filtered versions alongside the originals.

Output:
  - openasl-v1.0-filtered.tsv  (same dir as original)
  - how2sign_train-filtered.csv (same dir as original)
"""
import csv
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
VIDEO_ROOT = Path("/mnt/data/chatsign-auto-videos")

OPENASL_TSV = VIDEO_ROOT / "opensl_data" / "annotations" / "openasl-v1.0.tsv"
OPENASL_VIDEO_DIR = VIDEO_ROOT / "opensl_data"
OPENASL_OUT = OPENASL_TSV.with_name("openasl-v1.0-filtered.tsv")

H2S_CSV = VIDEO_ROOT / "how2sign_data" / "annotations" / "en" / "raw_text" / "how2sign_train.csv"
H2S_VIDEO_DIR = VIDEO_ROOT / "how2sign_data"
H2S_OUT = H2S_CSV.with_name("how2sign_train-filtered.csv")


def filter_openasl():
    """Filter OpenASL: keep rows whose vid maps to an existing .mp4 file."""
    print(f"Reading {OPENASL_TSV}")
    video_files = {f.stem for f in OPENASL_VIDEO_DIR.glob("*.mp4")}
    print(f"  Found {len(video_files)} video files")

    kept, dropped = 0, 0
    with open(OPENASL_TSV, encoding="utf-8") as fin, \
         open(OPENASL_OUT, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            vid = row["vid"]
            if vid in video_files:
                writer.writerow(row)
                kept += 1
            else:
                dropped += 1

    print(f"  Kept {kept}, dropped {dropped} → {OPENASL_OUT.name}")


def filter_how2sign():
    """Filter How2Sign: keep rows whose SENTENCE_NAME maps to an existing .mp4 file."""
    print(f"Reading {H2S_CSV}")
    video_files = {f.stem for f in H2S_VIDEO_DIR.glob("*.mp4")}
    print(f"  Found {len(video_files)} video files")

    kept, dropped = 0, 0
    with open(H2S_CSV, encoding="utf-8") as fin, \
         open(H2S_OUT, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            sentence_name = row["SENTENCE_NAME"]
            if sentence_name in video_files:
                writer.writerow(row)
                kept += 1
            else:
                dropped += 1

    print(f"  Kept {kept}, dropped {dropped} → {H2S_OUT.name}")


if __name__ == "__main__":
    filter_openasl()
    filter_how2sign()
    print("Done.")
