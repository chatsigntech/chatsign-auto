#!/usr/bin/env python3
"""Recover injection step after a successful DGX run whose DB insert failed.

Reads `tests/dgx_smoke/out_makeup/phase3_report.json` + the generated mp4s
under `out_makeup/videos/`, writes a Phase3TestJob row per completed result.
"""
import json
import os
import re
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("/home/chatsign/lizh/chatsign-auto")
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session

from backend.config import settings
from backend.database import engine
from backend.models.phase3_test import Phase3TestJob


ACCURACY_DIR = REPO_ROOT / "chatsign-accuracy/backend/data/uploads/videos/Heba"
TEXTS_PATH = REPO_ROOT / "chatsign-accuracy/backend/data/texts/makeup.jsonl"
PENDING_PATH = REPO_ROOT / "chatsign-accuracy/backend/data/reports/pending-videos.jsonl"
REPORT = REPO_ROOT / "tests/dgx_smoke/out_makeup/phase3_report.json"
VIDEOS_DIR = REPO_ROOT / "tests/dgx_smoke/out_makeup/videos"


def load_metadata() -> dict[int, dict]:
    m = {}
    with TEXTS_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            m[row["id"]] = {"text": row["text"], "video_id": None}
    if PENDING_PATH.exists():
        with PENDING_PATH.open() as f:
            for line in f:
                row = json.loads(line)
                mm = re.match(r"makeup_(\d+)_", row.get("videoFileName", ""))
                if mm:
                    sid = int(mm.group(1))
                    if sid in m:
                        m[sid]["video_id"] = row.get("videoId", "")
    return m


def main() -> int:
    if not REPORT.exists():
        print(f"[FAIL] no report at {REPORT}")
        return 1
    report = json.loads(REPORT.read_text())
    meta = load_metadata()

    n = 0
    for r in report.get("results", []):
        if r.get("status") != "completed":
            continue
        sid_match = re.match(r"makeup_(\d+)_", r["filename"])
        sid = int(sid_match.group(1)) if sid_match else None
        info = meta.get(sid, {}) if sid is not None else {}

        gen_src = VIDEOS_DIR / r["filename"]
        if not gen_src.exists():
            print(f"  [skip] generated mp4 missing: {r['filename']}")
            continue

        db_id = f"makeup_{uuid.uuid4().hex[:8]}"
        job_dir = settings.PHASE3_TEST_OUTPUT_DIR / db_id
        (job_dir / "input").mkdir(parents=True, exist_ok=True)
        src_original = ACCURACY_DIR / r["filename"]
        staged = job_dir / "input" / r["filename"]
        if not staged.exists():
            shutil.copy2(src_original, staged)
        gen_dst = job_dir / "generated.mp4"
        shutil.copy2(gen_src, gen_dst)

        row = Phase3TestJob(
            job_id=db_id,
            status="completed",
            video_id=info.get("video_id") or f"makeup_{sid}",
            sentence_text=(info.get("text") or "")[:300],
            translator_id="Heba",
            source_video_path=str(staged),
            source_filename=r["filename"],
            output_dir=str(job_dir),
            generated_video_path=str(gen_dst),
            duration_sec=r.get("wall_sec"),
            transfer_time_sec=r.get("wall_sec"),
            process_time_sec=0.0,
            framer_time_sec=0.0,
            error_message=f"[makeup batch test] DGX JOBID={r.get('job_id')} "
                          f"sub_task={r.get('sub_task_id')}",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        with Session(engine) as session:
            session.add(row)
            session.commit()
        n += 1
        print(f"  [ok] sid={sid:<2}  db={db_id}  '{(info.get('text') or '')[:60]}…'")

    print(f"\ninjected {n} rows. page: https://auto.chatsign.ai/phase3-test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
