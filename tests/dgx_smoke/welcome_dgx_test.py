#!/usr/bin/env python3
"""Submit the same WELCOME source video to DGX for A/B vs local result.

Local result was `wordsl2_a6be55c7` (sub_id=word_WELCOME_920420).
This script submits the SAME source mp4 to DGX, injects with prefix `wdgx_`.
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
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session

from backend.config import settings
from backend.database import engine
from backend.models.phase3_test import Phase3TestJob
from backend.workers.phase3_dgx_client import run_phase3_on_dgx


SOURCE = REPO_ROOT / "chatsign-accuracy/backend/data/uploads/videos/Heba/Tareq" \
         / "26commencement-02_9_en_7_2026041908250000_2026041908251600.mp4"
# Actually Tareq is its own dir, fix path:
SOURCE = REPO_ROOT / "chatsign-accuracy/backend/data/uploads/videos/Tareq" \
         / "26commencement-02_9_en_7_2026041908250000_2026041908251600.mp4"


async def main() -> int:
    if not SOURCE.exists():
        print(f"[FAIL] source missing: {SOURCE}")
        return 1

    staging = Path(tempfile.mkdtemp(prefix="welcome_dgx_"))
    (staging / "word_WELCOME.mp4").symlink_to(SOURCE.resolve())
    out_dir = REPO_ROOT / "tests/dgx_smoke/out_welcome_dgx" / uuid.uuid4().hex[:8]
    out_dir.mkdir(parents=True, exist_ok=True)
    task_id = f"wdgx_{uuid.uuid4().hex[:8]}"

    async def _progress(pct: float):
        print(f"      [{datetime.now().strftime('%H:%M:%S')}] progress={pct:.1f}%", flush=True)

    print(f"[1/2] submitting WELCOME to DGX  task_id={task_id}")
    print(f"      source = {SOURCE.name}")
    t0 = datetime.now()
    try:
        report = await run_phase3_on_dgx(task_id, staging, out_dir, progress_cb=_progress)
    except Exception as e:
        print(f"[FAIL] {e}")
        return 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    wall = (datetime.now() - t0).total_seconds()
    print(f"[2/2] DGX done in {wall/60:.1f} min  "
          f"success={report['transfer_success']}/{report['input_videos']}")

    n_ok = 0
    for r in report.get("results", []):
        if r.get("status") != "completed":
            print(f"      [skip] {r.get('stage')}  {(r.get('error') or '')[:200]}")
            continue
        db_id = f"wdgx_{uuid.uuid4().hex[:8]}"
        job_dir = settings.PHASE3_TEST_OUTPUT_DIR / db_id
        (job_dir / "input").mkdir(parents=True, exist_ok=True)
        staged = job_dir / "input" / r["filename"]
        shutil.copy2(SOURCE, staged)
        gen_dst = job_dir / "generated.mp4"
        shutil.copy2(r["output_path"], gen_dst)

        row = Phase3TestJob(
            job_id=db_id, status="completed",
            video_id="word_welcome", sentence_text="WELCOME",
            translator_id="Tareq",
            source_video_path=str(staged), source_filename=r["filename"],
            output_dir=str(job_dir), generated_video_path=str(gen_dst),
            duration_sec=r.get("wall_sec"), transfer_time_sec=r.get("wall_sec"),
            process_time_sec=0.0, framer_time_sec=0.0,
            error_message=f"[welcome DGX A/B] sub={r.get('sub_task_id')} job={r.get('job_id')}",
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        )
        with Session(engine) as session:
            session.add(row); session.commit()
        n_ok += 1
        print(f"      [ok] db={db_id}  {r.get('output_size',0)//1024} KB")

    print(f"\n=== A/B URLS ===")
    print(f"local (wordsl2_a6be55c7): https://auto.chatsign.ai/api/phase3-test/jobs/wordsl2_a6be55c7/generated-video")
    print(f"DGX  (just generated):    https://auto.chatsign.ai/api/phase3-test/bundle/wdgx")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
