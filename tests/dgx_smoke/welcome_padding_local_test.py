#!/usr/bin/env python3
"""Single-video local Phase 3 test on the front+back-padded WELCOME source.

Same code path as words_local_test.py / makeup_local_test.py — calls
phase3_local_client.run_phase3_on_local unchanged. Only the input video
is different (50 frames cloned at front+back via ffmpeg tpad).

For A/B vs the un-padded version (`wordsl2_a6be55c7`).
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
from backend.workers.phase3_local_client import run_phase3_on_local


PADDED_SRC = REPO_ROOT / "tests/dgx_smoke/inputs/word_WELCOME_padding.mp4"


async def main() -> int:
    if not PADDED_SRC.exists():
        print(f"[FAIL] padded source missing: {PADDED_SRC}")
        return 1

    staging = Path(tempfile.mkdtemp(prefix="welcome_pad_"))
    (staging / "word_WELCOME_padding.mp4").symlink_to(PADDED_SRC.resolve())
    out_dir = REPO_ROOT / "tests/dgx_smoke/out_welcome_padding" / uuid.uuid4().hex[:8]
    out_dir.mkdir(parents=True, exist_ok=True)
    task_id = f"wpad_{uuid.uuid4().hex[:8]}"

    async def _progress(pct: float):
        print(f"      [{datetime.now().strftime('%H:%M:%S')}] progress={pct:.1f}%", flush=True)

    print(f"[1/2] running local Phase 3 on padded WELCOME")
    print(f"      task_id = {task_id}")
    print(f"      source  = {PADDED_SRC.name}  (50 frames cloned front+back)")
    t0 = datetime.now()
    try:
        report = await run_phase3_on_local(task_id, staging, out_dir, progress_cb=_progress)
    except Exception as e:
        print(f"[FAIL] {e}")
        return 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    wall = (datetime.now() - t0).total_seconds()
    print(f"[2/2] done in {wall/60:.1f} min  "
          f"success={report['transfer_success']}/{report['input_videos']}")

    n_ok = 0
    for r in report.get("results", []):
        if r.get("status") != "completed":
            print(f"      [skip] {r.get('stage')} {(r.get('error') or '')[:200]}")
            continue
        db_id = f"wpad_{uuid.uuid4().hex[:8]}"
        job_dir = settings.PHASE3_TEST_OUTPUT_DIR / db_id
        (job_dir / "input").mkdir(parents=True, exist_ok=True)
        staged = job_dir / "input" / r["filename"]
        shutil.copy2(PADDED_SRC, staged)
        gen_dst = job_dir / "generated.mp4"
        shutil.copy2(r["output_path"], gen_dst)

        row = Phase3TestJob(
            job_id=db_id, status="completed",
            video_id="word_welcome_padded", sentence_text="WELCOME (padded ±50)",
            translator_id="Tareq",
            source_video_path=str(staged), source_filename=r["filename"],
            output_dir=str(job_dir), generated_video_path=str(gen_dst),
            duration_sec=r.get("wall_sec"), transfer_time_sec=r.get("wall_sec"),
            process_time_sec=0.0, framer_time_sec=0.0,
            error_message=f"[welcome padded test] backend=local sub_id={r.get('sub_id')}",
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        )
        with Session(engine) as session:
            session.add(row); session.commit()
        n_ok += 1
        print(f"      [ok] db={db_id}  {r.get('output_size',0)//1024} KB")

    print(f"\n=== A/B URLS ===")
    print(f"WITHOUT padding (orig):  https://auto.chatsign.ai/api/phase3-test/jobs/wordsl2_a6be55c7/generated-video")
    print(f"WITH padding ±50 frames: https://auto.chatsign.ai/api/phase3-test/bundle/wpad")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
