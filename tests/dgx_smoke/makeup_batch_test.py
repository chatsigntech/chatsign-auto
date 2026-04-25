#!/usr/bin/env python3
"""End-to-end test: drive 14 `makeup` accuracy videos through the production
phase3_dgx_client, then inject each result as a Phase3TestJob row so they're
viewable at /phase3-test.

Stages 14 makeup_*.mp4 from accuracy/uploads/Heba/ into a temp input dir
(symlinks, no copy), invokes run_phase3_on_dgx (single call processes all 14
in parallel on DGX), then for each completed result registers a DB row pointing
the page at the local filesystem result.
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

# .env uses relative sqlite path — chdir before importing settings so the
# DB URL resolves to the real orchestrator DB instead of an empty fallback.
REPO_ROOT = Path("/home/chatsign/lizh/chatsign-auto")
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
from sqlmodel import Session

from backend.config import settings
from backend.database import engine
from backend.models.phase3_test import Phase3TestJob
from backend.workers.phase3_dgx_client import run_phase3_on_dgx


ACCURACY_DIR = Path("/home/chatsign/lizh/chatsign-auto/chatsign-accuracy/backend/data/uploads/videos/Heba")
TEXTS_PATH = Path("/home/chatsign/lizh/chatsign-auto/chatsign-accuracy/backend/data/texts/makeup.jsonl")
PENDING_PATH = Path("/home/chatsign/lizh/chatsign-auto/chatsign-accuracy/backend/data/reports/pending-videos.jsonl")


def load_makeup_metadata() -> dict[int, dict]:
    """Load {sentence_id: {text, video_id_in_pending}} from accuracy data."""
    sentences = {}
    with TEXTS_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            sentences[row["id"]] = {"text": row["text"], "video_id": None}

    if PENDING_PATH.exists():
        with PENDING_PATH.open() as f:
            for line in f:
                row = json.loads(line)
                fn = row.get("videoFileName", "")
                m = re.match(r"makeup_(\d+)_", fn)
                if m:
                    sid = int(m.group(1))
                    if sid in sentences:
                        sentences[sid]["video_id"] = row.get("videoId", "")
                        sentences[sid]["filename"] = fn
    return sentences


def parse_sentence_id(filename: str) -> int | None:
    m = re.match(r"makeup_(\d+)_", filename)
    return int(m.group(1)) if m else None


async def main() -> int:
    sentences = load_makeup_metadata()
    mp4s = sorted(ACCURACY_DIR.glob("makeup_*.mp4"))
    if not mp4s:
        print(f"[FAIL] no makeup videos under {ACCURACY_DIR}")
        return 1

    print(f"[1/4] staging {len(mp4s)} videos via symlink into temp input dir")
    staging = Path(tempfile.mkdtemp(prefix="makeup_batch_"))
    for v in mp4s:
        (staging / v.name).symlink_to(v.resolve())

    output_dir = Path(__file__).resolve().parent / "out_makeup"
    output_dir.mkdir(exist_ok=True)
    task_id = f"makeup_{uuid.uuid4().hex[:8]}"
    print(f"      task_id={task_id}")
    print(f"      staging={staging}")
    print(f"      output_dir={output_dir}")

    async def _progress(pct: float):
        print(f"      [{datetime.now().strftime('%H:%M:%S')}] progress={pct:.1f}%")

    print(f"[2/4] invoking run_phase3_on_dgx (this is the production code path)")
    t0 = datetime.now()
    try:
        report = await run_phase3_on_dgx(task_id, staging, output_dir, progress_cb=_progress)
    except Exception as e:
        print(f"[FAIL] run_phase3_on_dgx raised: {e}")
        return 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    wall = (datetime.now() - t0).total_seconds()
    print(f"[3/4] DGX run completed in {wall/60:.1f} min")
    print(f"      success={report['transfer_success']}/{report['input_videos']}  "
          f"failed={report['transfer_failed']}")

    # ── Inject each completed result as a Phase3TestJob row ────────────
    print(f"[4/4] injecting DB rows so results appear on /phase3-test")
    n_injected = 0
    for r in report.get("results", []):
        if r.get("status") != "completed":
            print(f"      [skip] {r.get('filename')}: {r.get('status')} ({r.get('stage')}) "
                  f"{(r.get('error') or '')[:80]}")
            continue

        sid = parse_sentence_id(r["filename"])
        meta = sentences.get(sid, {}) if sid is not None else {}
        sentence_text = (meta.get("text") or "")[:300]
        video_id = meta.get("video_id") or f"makeup_{sid}"

        db_job_id = f"makeup_{uuid.uuid4().hex[:8]}"
        job_dir = settings.PHASE3_TEST_OUTPUT_DIR / db_job_id
        (job_dir / "input").mkdir(parents=True, exist_ok=True)

        src_video = ACCURACY_DIR / r["filename"]
        staged_src = job_dir / "input" / r["filename"]
        if not staged_src.exists():
            shutil.copy2(src_video, staged_src)

        # Move (or copy) the generated mp4 from output_dir/videos to job_dir
        gen_src = Path(r["output_path"])
        gen_dst = job_dir / "generated.mp4"
        shutil.copy2(gen_src, gen_dst)

        row = Phase3TestJob(
            job_id=db_job_id,
            status="completed",
            video_id=video_id,
            sentence_text=sentence_text,
            translator_id="Heba",
            source_video_path=str(staged_src),
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
        n_injected += 1
        print(f"      [ok]   sid={sid:<2}  job={db_job_id}  '{sentence_text[:60]}…'")

    print(f"\n=== SUMMARY ===")
    print(f"wall          = {wall/60:.1f} min")
    print(f"completed     = {report['transfer_success']}/{report['input_videos']}")
    print(f"injected to DB= {n_injected}")
    print(f"page          = https://auto.chatsign.ai/phase3-test")
    return 0 if report["transfer_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
