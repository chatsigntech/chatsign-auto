#!/usr/bin/env python3
"""Inject the 701 mimic+filter outputs into chatsign-accuracy as a new review batch.

Inputs:
    /home/chatsign/lizh/chatsign-auto/dgx-pipeline-test/commence_701_manifest.jsonl
    /mnt/data/chatsign-phase3-test/commence-701-260428-render/videos/<sid>.mp4 (one per row)

Outputs (all under chatsign-accuracy/):
    backend/data/texts/commence-701-260428-render.jsonl     (701 sentences, sid 1..701)
    backend/data/review/generated/commence-701-260428-render/<sid>.mp4   (copied videos)
    backend/data/reports/pending-videos.jsonl   (701 new entries appended, source=generated)

Optional --assign-to <reviewerId>: also POST review-assignments for the
videoIds added in THIS run. Idempotent (server dedups).

Idempotent: skips entries whose videoId already exists in pending-videos.jsonl.
"""

import argparse
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from _inject_helpers import compute_description, load_master_descriptions


REPO = Path("/home/chatsign/lizh/chatsign-auto")
ACC = REPO / "chatsign-accuracy" / "backend" / "data"
MANIFEST = REPO / "dgx-pipeline-test" / "commence_701_manifest.jsonl"
STAGE_VIDS = Path("/mnt/data/chatsign-phase3-test/commence-701-260428-render/videos")

BATCH_NAME = "commence-701-260428-render.jsonl"
OUT_VIDEO_DIR = ACC / "review" / "generated" / "commence-701-260428-render"
TEXTS_FILE = ACC / "texts" / BATCH_NAME
PENDING_FILE = ACC / "reports" / "pending-videos.jsonl"


def assign_videos(video_ids: list[str], reviewer_id: str, host: str) -> None:
    """POST /api/admin/review-assignments. Server dedups existing pairs."""
    if not video_ids:
        print(f"[assign] nothing to assign to {reviewer_id}")
        return
    body = json.dumps({"videoIds": video_ids, "reviewerId": reviewer_id}).encode()
    req = urllib.request.Request(
        f"{host}/api/admin/review-assignments",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-User-Id": "chatsign2026admin",
        },
        method="POST",
    )
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = json.loads(resp.read())
        d = data.get("data", {})
        print(f"[assign] reviewer={reviewer_id}: created={d.get('created')} skipped={d.get('skipped')} selfSkipped={d.get('selfSkipped')}")
    except Exception as e:
        print(f"[assign] FAILED: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign-to", default=None, help="reviewerId to auto-assign newly injected videoIds")
    ap.add_argument("--accuracy-host", default="https://localhost:5443")
    args = ap.parse_args()

    records = []
    with MANIFEST.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"manifest entries: {len(records)}")

    OUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    TEXTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Resolve description per record: master row if present, else WordNet
    # (commence-701 is single-word entries; sentence-level batches use kind='sentence')
    master_lookup = load_master_descriptions()
    descriptions = {
        r["new_sid"]: compute_description(r["text"], kind="word", master_lookup=master_lookup)
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
        src = STAGE_VIDS / f"{r['new_sid']:04d}.mp4"
        dst = OUT_VIDEO_DIR / f"{r['new_sid']:04d}.mp4"
        if not src.exists():
            missing += 1
            continue
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    print(f"videos: copied={copied}, skipped (already present)={skipped}, missing={missing}")

    # 3. Append to pending-videos.jsonl (skip already-present videoIds)
    existing_vids = set()
    if PENDING_FILE.exists():
        with PENDING_FILE.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_vids.add(json.loads(line).get("videoId"))
                except Exception:
                    pass

    new_entries = []
    for r in records:
        new_video_id = f"commence701_{r['new_sid']:04d}"
        if new_video_id in existing_vids:
            continue
        if not (OUT_VIDEO_DIR / f"{r['new_sid']:04d}.mp4").exists():
            continue  # don't list a video that wasn't produced
        new_entries.append({
            "videoId": new_video_id,
            "sentenceId": r["new_sid"],
            "sentenceText": r["text"],
            "translatorId": "generated",
            "language": "en",
            "videoPath": f"review/generated/commence-701-260428-render/{r['new_sid']:04d}.mp4",
            "videoFileName": f"{r['new_sid']:04d}.mp4",
            "localPath": f"commence-701-260428-render/{r['new_sid']:04d}.mp4",
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

    # compact JSON to match existing pending-videos.jsonl convention
    with PENDING_FILE.open("a") as f:
        for e in new_entries:
            f.write(json.dumps(e, ensure_ascii=False, separators=(',', ':')) + "\n")
    print(f"appended {len(new_entries)} entries to pending-videos.jsonl")
    print(f"   batchFile: {BATCH_NAME}")
    print(f"   source:    generated")

    if args.assign_to and new_entries:
        assign_videos([e["videoId"] for e in new_entries], args.assign_to, args.accuracy_host)


if __name__ == "__main__":
    main()
