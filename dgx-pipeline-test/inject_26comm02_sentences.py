#!/usr/bin/env python3
"""Inject 168 26commencement-02-sentences DGX-rendered videos into chatsign-accuracy.

Inputs:
    /home/chatsign/lizh/chatsign-auto/dgx-pipeline-test/26commencement_02_sentences_manifest.jsonl
    /mnt/data/chatsign-phase3-test/26commencement-02-render/videos/<sid>.mp4 (one per row)

Outputs (all under chatsign-accuracy/):
    backend/data/texts/26commencement-02-render.jsonl     (168 sentences, sid 1..168)
    backend/data/review/generated/26commencement-02-render/<sid>.mp4   (copied videos)
    backend/data/reports/pending-videos.jsonl   (168 new entries appended, source=generated)

Idempotent: skips entries whose videoId already exists in pending-videos.jsonl.

NOTE: This is the no-auto-assign variant — review-assignments are NOT written.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from _inject_helpers import compute_description, load_existing_video_ids, load_master_descriptions, video_filename


REPO = Path("/home/chatsign/lizh/chatsign-auto")
ACC = REPO / "chatsign-accuracy" / "backend" / "data"
MANIFEST = REPO / "dgx-pipeline-test" / "26commencement_02_sentences_manifest.jsonl"
STAGE_VIDS = Path("/mnt/data/chatsign-phase3-test/26commencement-02-render/videos")

BATCH_NAME = "26commencement-02-render.jsonl"
OUT_VIDEO_DIR = ACC / "review" / "generated" / "26commencement-02-render"
TEXTS_FILE = ACC / "texts" / BATCH_NAME
PENDING_FILE = ACC / "reports" / "pending-videos.jsonl"


def main():
    records = []
    with MANIFEST.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"manifest entries: {len(records)}")

    for r in records:
        r["_video_id"] = f"comm26_02s_{r['new_sid']:04d}"
        r["_fname"] = video_filename(r["_video_id"])

    # Short-circuit when there's nothing new to inject — avoids the pandas
    # master-CSV load and the TextPipeline cold-start (~30s for sentence-kind).
    existing_vids = load_existing_video_ids(PENDING_FILE)
    new_records = [r for r in records if r["_video_id"] not in existing_vids]
    if not new_records:
        print(f"nothing to inject — all {len(records)} videoIds already in pending-videos")
        return

    OUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    TEXTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # description per record: master row if present, else ASL gloss tokens
    master_lookup = load_master_descriptions()
    descriptions = {
        r["new_sid"]: compute_description(r["text"], kind="sentence", master_lookup=master_lookup)
        for r in records
    }

    # 1. Build texts/<batch>.jsonl
    text_rows = [
        {"id": r["new_sid"], "language": "en", "text": r["text"], "description": descriptions[r["new_sid"]]}
        for r in records
    ]
    with TEXTS_FILE.open("w") as f:
        for row in text_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {TEXTS_FILE} ({len(text_rows)} sentences)")

    # 2. Copy videos
    copied = skipped = missing = 0
    for r in records:
        src = STAGE_VIDS / r["_fname"]
        dst = OUT_VIDEO_DIR / r["_fname"]
        if not src.exists():
            missing += 1
            continue
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    print(f"videos: copied={copied}, skipped (already present)={skipped}, missing={missing}")

    # 3. Append to pending-videos.jsonl
    new_entries = []
    for r in new_records:
        if not (OUT_VIDEO_DIR / r["_fname"]).exists():
            continue  # don't list a video that wasn't produced
        new_entries.append({
            "videoId": r["_video_id"],
            "sentenceId": r["new_sid"],
            "sentenceText": r["text"],
            "translatorId": "generated",
            "language": "en",
            "videoPath": f"review/generated/26commencement-02-render/{r['_fname']}",
            "videoFileName": r["_fname"],
            "localPath": f"26commencement-02-render/{r['_fname']}",
            "source": "generated",
            "addedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "batchFile": BATCH_NAME,
            "description": descriptions[r["new_sid"]],
            # Audit trail back to the recording it was derived from:
            "_origin": {
                "src_videoId": r["src_videoId"],
                "src_batchFile": r["src_batchFile"],
                "src_sid": r["src_sid"],
                "src_translator": r["translator"],
                "src_sentence": r["text"],
            },
        })

    # Use compact JSON to match the file's existing convention (no spaces).
    with PENDING_FILE.open("a") as f:
        for e in new_entries:
            f.write(json.dumps(e, ensure_ascii=False, separators=(',', ':')) + "\n")
    print(f"appended {len(new_entries)} entries to pending-videos.jsonl")
    print(f"   batchFile: {BATCH_NAME}")
    print(f"   source:    generated")


if __name__ == "__main__":
    main()
