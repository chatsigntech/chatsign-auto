#!/usr/bin/env python3
"""Drive commence-701-260428-render through DGX (mimic+filter), pull back, h264 remux.

Stages:
    1. read manifest
    2. stage inputs on DGX (parallel scp, capped at SSH_CONCURRENCY)
    3. submit sbatch infer_dgx_total.sh per task
    4. poll until queue drains (or per-job sacct OK)
    5. pull output_filter/input.mp4 back
    6. h264 remux to local staging dir
    7. write a status JSON; injection into accuracy is done by inject_commence_701.py

Idempotent: each step skips work already done (TASK dir exists, output exists, etc).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _naming import video_filename


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "dgx-pipeline-test" / "commence_701_manifest.jsonl"
STAGING_ROOT = Path("/mnt/data/chatsign-phase3-test/commence-701-260428-render")
FFMPEG = REPO_ROOT / "bin" / "ffmpeg"

DGX_HOST = "dgx-login"
DGX_TASKS_ROOT = "/media/cvpr/zhewen/api_tasks"
DGX_REF_IMAGE = "/media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/data/ref_images/test4.jpg"
DGX_LOGS_DIR = "/media/cvpr/zhewen/logs"
DGX_SBATCH = "/media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/infer_dgx_total.sh"
DGX_OUTPUT_RELPATH = "output_filter/input.mp4"

SSH_CONCURRENCY = 6
FFMPEG_CONCURRENCY = 4
POLL_INTERVAL = 30


def task_id_for(rec: dict) -> str:
    return f"c701_{rec['new_sid']:04d}"


def run(cmd: list[str], **kw) -> tuple[int, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, (r.stdout + r.stderr)


def ssh(cmd: str) -> tuple[int, str]:
    return run(["ssh", DGX_HOST, cmd])


def scp_up(local: Path, remote: str) -> tuple[int, str]:
    return run(["scp", "-q", str(local), f"{DGX_HOST}:{remote}"])


def scp_down(remote: str, local: Path) -> tuple[int, str]:
    return run(["scp", "-q", f"{DGX_HOST}:{remote}", str(local)])


def stage_one(rec: dict) -> dict:
    tid = task_id_for(rec)
    tdir = f"{DGX_TASKS_ROOT}/{tid}"
    rc, out = ssh(
        f"mkdir -p '{tdir}/videos' '{tdir}/output' '{tdir}/output_filter' && "
        f"if [ ! -f '{tdir}/input_image.png' ]; then cp '{DGX_REF_IMAGE}' '{tdir}/input_image.png'; fi && "
        f"echo OK"
    )
    if rc != 0 or "OK" not in out:
        return {"new_sid": rec["new_sid"], "task_id": tid, "stage": "mkdir", "ok": False, "err": out[:200]}

    # Skip scp if input already there (idempotent re-runs).
    rc, out = ssh(f"test -s '{tdir}/videos/input.mp4' && echo HAS")
    if "HAS" in out:
        return {"new_sid": rec["new_sid"], "task_id": tid, "stage": "stage", "ok": True, "skipped": True}

    rc, out = scp_up(Path(rec["localPath"]), f"{tdir}/videos/input.mp4")
    return {
        "new_sid": rec["new_sid"], "task_id": tid, "stage": "stage",
        "ok": rc == 0, "err": "" if rc == 0 else out[:200],
    }


def stage_all(records: list[dict]) -> list[dict]:
    print(f"[stage] uploading {len(records)} videos to DGX (parallel={SSH_CONCURRENCY})…")
    results = []
    with ThreadPoolExecutor(max_workers=SSH_CONCURRENCY) as ex:
        futs = {ex.submit(stage_one, r): r for r in records}
        done = 0
        for f in as_completed(futs):
            res = f.result()
            results.append(res)
            done += 1
            if done % 25 == 0 or done == len(records):
                fail = sum(1 for r in results if not r.get("ok"))
                print(f"  staged {done}/{len(records)} (fail={fail})")
    fails = [r for r in results if not r.get("ok")]
    if fails:
        print(f"[stage] {len(fails)} failures:")
        for r in fails[:5]:
            print(f"    sid={r['new_sid']}: {r.get('err','')}")
    return results


def submit_one(rec: dict) -> dict:
    tid = task_id_for(rec)
    cmd = (
        f"cd /media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion && "
        f"sbatch --parsable --export=ALL,TASK_ID={tid} infer_dgx_total.sh"
    )
    rc, out = ssh(cmd)
    out = out.strip()
    if rc != 0 or not out.split("\n")[-1].isdigit():
        return {"new_sid": rec["new_sid"], "task_id": tid, "ok": False, "err": out[:200]}
    job_id = out.split("\n")[-1]
    return {"new_sid": rec["new_sid"], "task_id": tid, "job_id": job_id, "ok": True}


def submit_all(records: list[dict]) -> list[dict]:
    print(f"[submit] sbatch {len(records)} jobs sequentially…")
    results = []
    for i, r in enumerate(records, 1):
        results.append(submit_one(r))
        if i % 25 == 0 or i == len(records):
            fail = sum(1 for x in results if not x.get("ok"))
            print(f"  submitted {i}/{len(records)} (fail={fail})")
    return results


def poll_until_done(job_ids: list[str]) -> None:
    print(f"[poll] waiting on {len(job_ids)} jobs…")
    pending = set(job_ids)
    t0 = time.time()
    while pending:
        # squeue accepts comma list; chunk to avoid huge args.
        active = set()
        for chunk_start in range(0, len(pending), 200):
            chunk = list(pending)[chunk_start:chunk_start + 200]
            rc, out = ssh(f"squeue -j {','.join(chunk)} -h -o %i 2>/dev/null")
            for line in out.splitlines():
                jid = line.strip()
                if jid:
                    active.add(jid)
        done_now = pending - active
        pending = active
        elapsed = int(time.time() - t0)
        print(f"  t+{elapsed:>5}s  active={len(pending)}  done={len(job_ids) - len(pending)}")
        if not pending:
            break
        time.sleep(POLL_INTERVAL)


def fetch_one(rec: dict) -> dict:
    tid = task_id_for(rec)
    sid = rec["new_sid"]
    out_dir = STAGING_ROOT / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / video_filename(f"commence701_{sid:04d}")
    if final.exists() and final.stat().st_size > 0:
        return {"new_sid": sid, "task_id": tid, "ok": True, "skipped": True, "path": str(final)}
    raw = out_dir / f".raw_{sid:04d}.mp4"
    rc, out = scp_down(f"{DGX_TASKS_ROOT}/{tid}/{DGX_OUTPUT_RELPATH}", raw)
    if rc != 0 or not raw.exists() or raw.stat().st_size == 0:
        return {"new_sid": sid, "task_id": tid, "ok": False, "stage": "fetch", "err": out[:200]}
    rc, out = run([
        str(FFMPEG), "-y", "-loglevel", "error",
        "-i", str(raw),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(final),
    ])
    raw.unlink(missing_ok=True)
    if rc != 0 or not final.exists() or final.stat().st_size == 0:
        return {"new_sid": sid, "task_id": tid, "ok": False, "stage": "remux", "err": out[:200]}
    return {"new_sid": sid, "task_id": tid, "ok": True, "path": str(final)}


def fetch_all(records: list[dict]) -> list[dict]:
    print(f"[fetch] pulling {len(records)} outputs + h264 remux (parallel={FFMPEG_CONCURRENCY})…")
    results = []
    with ThreadPoolExecutor(max_workers=FFMPEG_CONCURRENCY) as ex:
        futs = {ex.submit(fetch_one, r): r for r in records}
        done = 0
        for f in as_completed(futs):
            res = f.result()
            results.append(res)
            done += 1
            if done % 25 == 0 or done == len(records):
                fail = sum(1 for r in results if not r.get("ok"))
                print(f"  fetched {done}/{len(records)} (fail={fail})")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage", "submit", "poll", "fetch", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="process only first N records (debug)")
    ap.add_argument("--state", default="/home/chatsign/lizh/chatsign-auto/dgx-pipeline-test/commence_701_state.json")
    args = ap.parse_args()

    records = []
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit > 0:
        records = records[: args.limit]
    print(f"loaded {len(records)} records from manifest")

    state_path = Path(args.state)
    state = {}
    if state_path.exists():
        with state_path.open() as f:
            state = json.load(f)

    def save_state():
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with state_path.open("w") as f:
            json.dump(state, f, indent=2)

    if args.stage in ("stage", "all"):
        state["stage_results"] = stage_all(records)
        save_state()

    if args.stage in ("submit", "all"):
        # Only submit for stages that succeeded (or were already staged).
        ok_sids = {r["new_sid"] for r in state.get("stage_results", []) if r.get("ok")}
        to_submit = [r for r in records if r["new_sid"] in ok_sids] if ok_sids else records
        state["submit_results"] = submit_all(to_submit)
        save_state()

    if args.stage in ("poll", "all"):
        job_ids = [r["job_id"] for r in state.get("submit_results", []) if r.get("ok")]
        if job_ids:
            poll_until_done(job_ids)

    if args.stage in ("fetch", "all"):
        ok_sids = {r["new_sid"] for r in state.get("submit_results", []) if r.get("ok")}
        to_fetch = [r for r in records if r["new_sid"] in ok_sids] if ok_sids else records
        state["fetch_results"] = fetch_all(to_fetch)
        save_state()

    print("done.")


if __name__ == "__main__":
    main()
