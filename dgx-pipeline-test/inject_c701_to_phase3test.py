#!/usr/bin/env python3
"""Inject all completed c701 mimic+filter results into the phase3-test page.

For each (sid) where both raw recording (original) and filter output exist
locally, write one Phase3TestJob row pointing to:
  - source_video_path: the original human recording
  - generated_video_path: the post-filter (final) mp4

Idempotent: skips job_ids already in phase3_test_jobs.
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _naming import video_filename

DB_PATH = "/home/chatsign/lizh/chatsign-auto/data/chatsign/orchestrator/tasks.db"
ACC_UPLOADS = Path("/home/chatsign/lizh/chatsign-auto/chatsign-accuracy/backend/data/uploads/videos")
STAGE_ROOT = Path("/mnt/data/chatsign-phase3-test/commence-701-260428-render")
MANIFEST = "/home/chatsign/lizh/chatsign-auto/dgx-pipeline-test/commence_701_manifest.jsonl"


def main():
    records = {}
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[f"{r['new_sid']:04d}"] = r
    print(f"manifest entries: {len(records)}")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    existing = {row[0] for row in cur.execute("SELECT job_id FROM phase3_test_jobs")}
    print(f"existing phase3_test_jobs: {len(existing)}")

    inserted = skipped_exists = skipped_missing = 0
    now = datetime.utcnow().isoformat()
    rows = []
    for sid, r in sorted(records.items(), key=lambda x: int(x[0])):
        job_id = f"c701-{sid}"
        if job_id in existing:
            skipped_exists += 1
            continue
        gen = STAGE_ROOT / "videos" / video_filename(f"commence701_{sid}")
        if not gen.exists():
            skipped_missing += 1
            continue
        src = Path(r["localPath"])
        if not src.exists():
            skipped_missing += 1
            continue
        rows.append((
            job_id,
            "completed",
            r["src_videoId"],                         # video_id
            f"{r['text']} (c701 sid={sid} src={r['src_batchFile']})",  # sentence_text
            r["translator"],                          # translator_id
            str(src),                                 # source_video_path
            r["src_filename"],                        # source_filename
            str(STAGE_ROOT),                          # output_dir
            str(gen),                                 # generated_video_path
            None, None, None, None, None,             # timing + error
            now, now,
        ))

    if rows:
        cur.executemany(
            """INSERT INTO phase3_test_jobs
               (job_id, status, video_id, sentence_text, translator_id,
                source_video_path, source_filename, output_dir, generated_video_path,
                duration_sec, transfer_time_sec, process_time_sec, framer_time_sec,
                error_message, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        con.commit()
        inserted = len(rows)
    con.close()

    print(f"inserted: {inserted}")
    print(f"skipped (already in DB): {skipped_exists}")
    print(f"skipped (missing files): {skipped_missing}")


if __name__ == "__main__":
    main()
