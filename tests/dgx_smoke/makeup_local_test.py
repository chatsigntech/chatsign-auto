#!/usr/bin/env python3
"""Re-run the 14 makeup batch sentences via local Phase 3 (DGX-replica) for
direct A/B comparison with the DGX-generated `makeup_*` jobs from yesterday.

Same source videos, same MimicMotion code+weights+ref_image — only difference
is hardware/kernel (local RTX 5090 sm_120 vs DGX GB10 sm_120).

New jobs use prefix `makeupl_` (l = local).
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
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session

from backend.config import settings
from backend.database import engine
from backend.models.phase3_test import Phase3TestJob
from backend.workers.phase3_local_client import run_phase3_on_local


ACCURACY_DIR = REPO_ROOT / "chatsign-accuracy/backend/data/uploads/videos/Heba"
TEXTS_PATH = REPO_ROOT / "chatsign-accuracy/backend/data/texts/makeup.jsonl"
PENDING_PATH = REPO_ROOT / "chatsign-accuracy/backend/data/reports/pending-videos.jsonl"


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


def parse_sid(filename: str) -> int | None:
    m = re.match(r"makeup_(\d+)_", filename)
    return int(m.group(1)) if m else None


async def main() -> int:
    metadata = load_metadata()
    mp4s = sorted(ACCURACY_DIR.glob("makeup_*.mp4"))
    if not mp4s:
        print(f"[FAIL] no makeup videos under {ACCURACY_DIR}")
        return 1

    print(f"[1/4] staging {len(mp4s)} makeup videos via symlink")
    staging = Path(tempfile.mkdtemp(prefix="makeup_local_"))
    for v in mp4s:
        (staging / v.name).symlink_to(v.resolve())

    out_dir = REPO_ROOT / "tests/dgx_smoke/out_makeup_local" / uuid.uuid4().hex[:8]
    out_dir.mkdir(parents=True, exist_ok=True)
    task_id = f"mklocal_{uuid.uuid4().hex[:8]}"
    print(f"      task_id={task_id}  out={out_dir}")

    async def _progress(pct: float):
        print(f"      [{datetime.now().strftime('%H:%M:%S')}] progress={pct:.1f}%", flush=True)

    n_injected = [0]  # mutable counter for closure

    def _inject_one(rec: dict) -> None:
        """Called immediately after each video finishes. Injects DB row so it
        appears on /phase3-test live, without waiting for the rest."""
        if rec.get("status") != "completed":
            print(f"      [skip] {rec.get('filename')}: {rec.get('status')} ({rec.get('stage')}) "
                  f"{(rec.get('error') or '')[:120]}", flush=True)
            return
        sid = parse_sid(rec["filename"])
        info = metadata.get(sid, {}) if sid is not None else {}

        db_id = f"makeupl_{uuid.uuid4().hex[:8]}"
        job_dir = settings.PHASE3_TEST_OUTPUT_DIR / db_id
        (job_dir / "input").mkdir(parents=True, exist_ok=True)
        src_orig = ACCURACY_DIR / rec["filename"]
        staged = job_dir / "input" / rec["filename"]
        if not staged.exists():
            shutil.copy2(src_orig, staged)
        gen_dst = job_dir / "generated.mp4"
        shutil.copy2(rec["output_path"], gen_dst)

        row = Phase3TestJob(
            job_id=db_id,
            status="completed",
            video_id=info.get("video_id") or f"makeup_{sid}",
            sentence_text=(info.get("text") or "")[:300],
            translator_id="Heba",
            source_video_path=str(staged),
            source_filename=rec["filename"],
            output_dir=str(job_dir),
            generated_video_path=str(gen_dst),
            duration_sec=rec.get("wall_sec"),
            transfer_time_sec=rec.get("wall_sec"),
            process_time_sec=0.0,
            framer_time_sec=0.0,
            error_message=f"[makeup local test] backend=local sub_id={rec.get('sub_id')}",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        with Session(engine) as session:
            session.add(row)
            session.commit()
        n_injected[0] += 1
        print(f"      [LIVE] sid={sid:<2}  db={db_id}  injected  "
              f"({n_injected[0]}/{len(mp4s)})  '{(info.get('text') or '')[:50]}…'", flush=True)

    print(f"[2/3] invoking run_phase3_on_local on {len(mp4s)} videos (serial,边跑边 inject)")
    t0 = datetime.now()
    try:
        report = await run_phase3_on_local(
            task_id, staging, out_dir,
            progress_cb=_progress,
            on_video_done=_inject_one,
        )
    except Exception as e:
        print(f"[FAIL] run_phase3_on_local: {e}")
        return 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    wall = (datetime.now() - t0).total_seconds()
    n_ok = n_injected[0]

    print(f"\n=== SUMMARY ===")
    print(f"wall          = {wall/60:.1f} min")
    print(f"completed     = {report['transfer_success']}/{report['input_videos']}")
    print(f"injected      = {n_ok}")
    print(f"page          = https://auto.chatsign.ai/phase3-test")
    print(f"bundle (local 14) = https://auto.chatsign.ai/api/phase3-test/bundle/makeupl")
    print(f"bundle (DGX 14,昨天) = https://auto.chatsign.ai/api/phase3-test/bundle/makeup")
    return 0 if report["transfer_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
