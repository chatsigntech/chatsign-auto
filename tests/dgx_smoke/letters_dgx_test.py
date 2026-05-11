#!/usr/bin/env python3
"""Submit all 26 letter recordings (A-Z, batch=letters.jsonl, by Tareq) to DGX.

Pinned to 6 verified-working nodes (have /home/cvpr/enerverse_arm) for
parallelism without hitting silently-broken nodes. 26 / 6 ≈ 5 rounds → ~50 min
wall (each MimicMotion ~10 min serial inside a node).

DB rows use prefix `wdgxa_` (a = alphabet) so they're distinguishable.
"""
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("/home/chatsign/lizh/chatsign-auto")
os.chdir(REPO_ROOT)
os.environ["DGX_NODELIST"] = (
    "ADUAED21042WKLX08,ADUAED21024WKLX13,ADUAED21037WKLX10,"
    "ADUAED21027WKLX14,ADUAED21035WKLX15,ADUAED21028WKLX16"
)
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session

from backend.config import settings
from backend.database import engine
from backend.models.phase3_test import Phase3TestJob
from backend.workers.phase3_dgx_client import run_phase3_on_dgx


ACCURACY_BASE = REPO_ROOT / "chatsign-accuracy/backend/data/uploads/videos/Tareq"
PENDING = REPO_ROOT / "chatsign-accuracy/backend/data/reports/pending-videos.jsonl"


def collect_letter_sources() -> dict[str, Path]:
    """Map LETTER → absolute path of Tareq's letters_<sid>_en_*.mp4."""
    by_letter: dict[str, Path] = {}
    with PENDING.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("batchFile") != "letters.jsonl":
                continue
            if r.get("translatorId") != "Tareq":
                continue
            txt = (r.get("sentenceText") or "").strip().upper()
            if len(txt) != 1 or not txt.isalpha():
                continue
            fn = r.get("videoFileName")
            if not fn:
                continue
            p = ACCURACY_BASE / fn
            if p.exists() and txt not in by_letter:
                by_letter[txt] = p
    return by_letter


async def main() -> int:
    sources = collect_letter_sources()
    missing = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in sources]
    if missing:
        print(f"[FAIL] missing letters: {missing}")
        return 1
    print(f"[1/3] all 26 letters located")

    staging = Path(tempfile.mkdtemp(prefix="letters_dgx_"))
    for L, src in sources.items():
        (staging / f"word_{L}.mp4").symlink_to(src.resolve())

    out_dir = REPO_ROOT / "tests/dgx_smoke/out_letters_dgx" / uuid.uuid4().hex[:8]
    out_dir.mkdir(parents=True, exist_ok=True)
    task_id = f"wdgxa_{uuid.uuid4().hex[:8]}"

    async def _progress(pct: float):
        print(f"      [{datetime.now().strftime('%H:%M:%S')}] progress={pct:.1f}%", flush=True)

    print(f"[2/3] submitting 26 videos to DGX")
    print(f"      task_id  = {task_id}")
    print(f"      nodelist = 6 verified-good nodes (5 rounds expected)")
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
        letter = r["filename"].replace("word_", "").replace(".mp4", "").upper()
        if r.get("status") != "completed":
            print(f"      [skip] {letter}: {r.get('status')} ({r.get('stage')}) "
                  f"{(r.get('error') or '')[:120]}")
            continue
        db_id = f"wdgxa_{uuid.uuid4().hex[:8]}"
        job_dir = settings.PHASE3_TEST_OUTPUT_DIR / db_id
        (job_dir / "input").mkdir(parents=True, exist_ok=True)
        src_orig = sources[letter]
        staged = job_dir / "input" / r["filename"]
        shutil.copy2(src_orig, staged)
        gen_dst = job_dir / "generated.mp4"
        shutil.copy2(r["output_path"], gen_dst)

        row = Phase3TestJob(
            job_id=db_id, status="completed",
            video_id=f"letter_{letter.lower()}", sentence_text=letter,
            translator_id="Tareq",
            source_video_path=str(staged), source_filename=r["filename"],
            output_dir=str(job_dir), generated_video_path=str(gen_dst),
            duration_sec=r.get("wall_sec"), transfer_time_sec=r.get("wall_sec"),
            process_time_sec=0.0, framer_time_sec=0.0,
            error_message=f"[26 letters DGX] sub={r.get('sub_task_id')} job={r.get('job_id')}",
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        )
        with Session(engine) as session:
            session.add(row); session.commit()
        n_ok += 1
        print(f"      [ok] {letter}  db={db_id}  job={r.get('job_id')}  {r.get('output_size',0)//1024} KB")

    print(f"\n=== SUMMARY ===")
    print(f"wall     = {wall/60:.1f} min")
    print(f"injected = {n_ok}/26")
    print(f"page     = https://auto.chatsign.ai/phase3-test")
    print(f"bundle   = https://auto.chatsign.ai/api/phase3-test/bundle/wdgxa")
    return 0 if report["transfer_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
