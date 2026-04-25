#!/usr/bin/env python3
"""Submit the 6 single-word recordings to DGX for A/B vs local versions.

Same source videos as words_local_test.py / wordsl2_*. New jobs use prefix
`wdgx6_` so we can distinguish from earlier single-word DGX runs.

Pinned to known-good nodes (WKLX08/11/13) via DGX_NODELIST env var — newly
recovered nodes lack /home/cvpr/enerverse_arm and would silently fail.
"""
import asyncio
import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("/home/chatsign/lizh/chatsign-auto")
os.chdir(REPO_ROOT)
# Pin to nodes that have BASE_PY installed (verified working on prior batches)
os.environ["DGX_NODELIST"] = "ADUAED21042WKLX08,ADUAED21024WKLX13,ADUAED21037WKLX10,ADUAED21027WKLX14,ADUAED21035WKLX15,ADUAED21028WKLX16"
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session

from backend.config import settings
from backend.database import engine
from backend.models.phase3_test import Phase3TestJob
from backend.workers.phase3_dgx_client import run_phase3_on_dgx


TARGETS = [
    ("WELCOME",    "Tareq/26commencement-02_9_en_7_2026041908250000_2026041908251600.mp4"),
    ("STUDENT",    "Tareq/26commencement-02_61_en_17_2026042208161700_2026042208162900.mp4"),
    ("STAFF",      "Tareq/26commencement-02_5_en_47_2026041407435800_2026041407441600.mp4"),
    ("FACULTY",    "Tareq/26commencement-02_4_en_28_2026041606331200_2026041606332700.mp4"),
    ("REMARKABLE", "Tareq/school_match_323_en_61_2026040207084000_2026040207090000.mp4"),
    ("EVENT",      "Tareq/school_match_152_en_1_2026040207174500_2026040207180800.mp4"),
]
ACCURACY_BASE = REPO_ROOT / "chatsign-accuracy/backend/data/uploads/videos"


async def main() -> int:
    staging = Path(tempfile.mkdtemp(prefix="words_dgx_"))
    print(f"[1/3] staging {len(TARGETS)} word videos via symlink")
    word_to_src = {}
    for word, rel in TARGETS:
        src = ACCURACY_BASE / rel
        if not src.exists():
            print(f"      [FAIL] missing: {src}"); return 1
        (staging / f"word_{word}.mp4").symlink_to(src.resolve())
        word_to_src[word] = src

    out_dir = REPO_ROOT / "tests/dgx_smoke/out_words_dgx" / uuid.uuid4().hex[:8]
    out_dir.mkdir(parents=True, exist_ok=True)
    task_id = f"wdgx6_{uuid.uuid4().hex[:8]}"

    async def _progress(pct: float):
        print(f"      [{datetime.now().strftime('%H:%M:%S')}] progress={pct:.1f}%", flush=True)

    print(f"[2/3] submitting {len(TARGETS)} videos to DGX")
    print(f"      task_id  = {task_id}")
    print(f"      nodelist = {os.environ['DGX_NODELIST']}")
    t0 = datetime.now()
    try:
        report = await run_phase3_on_dgx(task_id, staging, out_dir, progress_cb=_progress)
    except Exception as e:
        print(f"[FAIL] {e}")
        return 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    wall = (datetime.now() - t0).total_seconds()
    print(f"[3/3] done in {wall/60:.1f} min  "
          f"success={report['transfer_success']}/{report['input_videos']}")

    n_ok = 0
    for r in report.get("results", []):
        word = r["filename"].replace("word_", "").replace(".mp4", "")
        if r.get("status") != "completed":
            print(f"      [skip] {word}: {r.get('status')} ({r.get('stage')}) "
                  f"{(r.get('error') or '')[:120]}")
            continue
        db_id = f"wdgx6_{uuid.uuid4().hex[:8]}"
        job_dir = settings.PHASE3_TEST_OUTPUT_DIR / db_id
        (job_dir / "input").mkdir(parents=True, exist_ok=True)
        src_orig = word_to_src[word]
        staged = job_dir / "input" / r["filename"]
        shutil.copy2(src_orig, staged)
        gen_dst = job_dir / "generated.mp4"
        shutil.copy2(r["output_path"], gen_dst)

        row = Phase3TestJob(
            job_id=db_id, status="completed",
            video_id=f"word_{word.lower()}", sentence_text=word,
            translator_id="Tareq",
            source_video_path=str(staged), source_filename=r["filename"],
            output_dir=str(job_dir), generated_video_path=str(gen_dst),
            duration_sec=r.get("wall_sec"), transfer_time_sec=r.get("wall_sec"),
            process_time_sec=0.0, framer_time_sec=0.0,
            error_message=f"[words DGX A/B] sub={r.get('sub_task_id')} job={r.get('job_id')}",
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        )
        with Session(engine) as session:
            session.add(row); session.commit()
        n_ok += 1
        print(f"      [ok] {word:<12s} db={db_id}  job={r.get('job_id')}  {r.get('output_size',0)//1024} KB")

    print(f"\n=== A/B URLS ===")
    print(f"local nightly (wordsl2_*): https://auto.chatsign.ai/api/phase3-test/bundle/wordsl2")
    print(f"DGX           (wdgx6_*):   https://auto.chatsign.ai/api/phase3-test/bundle/wdgx6")
    return 0 if report["transfer_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
