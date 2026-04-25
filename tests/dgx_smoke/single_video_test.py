#!/usr/bin/env python3
"""Single-video end-to-end test against current production phase3_dgx_client.

Stages one video into a temp input dir and calls `run_phase3_on_dgx`, which
goes through the full scheduler path:
    ssh mkdir → scp video → ssh cp DGX's test4.jpg → sbatch → squeue poll →
    scp back → h264 remux → remote cleanup → phase3_report.json

No local image upload (DGX_REF_IMAGE on DGX is the canonical ref).
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

from backend.workers.phase3_dgx_client import run_phase3_on_dgx


DEFAULT_VIDEO = REPO_ROOT / "chatsign-accuracy/backend/data/review/generated/001_hello.mp4"


async def main() -> int:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO
    if not video.exists():
        print(f"[FAIL] video not found: {video}")
        return 1

    staging = Path(tempfile.mkdtemp(prefix="single_dgx_"))
    (staging / video.name).symlink_to(video.resolve())
    out_dir = REPO_ROOT / "tests/dgx_smoke/out_single" / uuid.uuid4().hex[:8]
    out_dir.mkdir(parents=True, exist_ok=True)
    task_id = f"single_{uuid.uuid4().hex[:8]}"

    async def _progress(pct: float):
        print(f"      [{datetime.now().strftime('%H:%M:%S')}] progress={pct:.1f}%", flush=True)

    print(f"[1/2] calling run_phase3_on_dgx")
    print(f"      task_id = {task_id}")
    print(f"      video   = {video}")
    print(f"      out_dir = {out_dir}")
    try:
        report = await run_phase3_on_dgx(task_id, staging, out_dir, progress_cb=_progress)
    except Exception as e:
        print(f"[FAIL] {e}")
        return 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"[2/2] done in {report['total_wall_seconds']:.0f}s  "
          f"success={report['transfer_success']}/{report['input_videos']}")
    for r in report["results"]:
        if r.get("status") == "completed":
            print(f"      OUTPUT: {r['output_path']}  ({r['output_size']/1024:.0f} KB)")
        else:
            print(f"      FAILED stage={r.get('stage')}  {r.get('error','')[:200]}")
    return 0 if report["transfer_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
