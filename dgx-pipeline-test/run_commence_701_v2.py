#!/usr/bin/env python3
"""Commence-701 V2: mimic + 27K-style filter chain (no RVM) for all 701 sids.

Flow per sid:
- If raw mimic exists (output/input_hiya.mp4):
    cp it to input_video.mp4 → sbatch filter → TG → SR (3-stage)
- Else:
    sbatch total (mimic+filter, 27K params) → TG → SR (3-stage)

All sbatch use --exclude=ADUAED21041WKLX30 (bad autofs node).

Stages:
    --stage prep   : ssh DGX, cp input_hiya.mp4 → input_video.mp4 for all 549 sids
    --stage submit : submit chain per sid, store SR job_id in state
    --stage poll   : block until all SR jobs leave queue
    --stage fetch  : pull sr.mp4 back, h264-remux to videos/<sid>.mp4
    --stage all    : prep+submit+poll+fetch
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _naming import video_filename


REPO = Path("/home/chatsign/lizh/chatsign-auto")
MANIFEST = REPO / "dgx-pipeline-test" / "commence_701_manifest.jsonl"
STATE = REPO / "dgx-pipeline-test" / "commence_701_state_v2.json"
STAGING = Path("/mnt/data/chatsign-phase3-test/commence-701-260428-render/videos")
FFMPEG = REPO / "bin" / "ffmpeg"

DGX = "dgx-login"
TR = "/media/cvpr/zhewen/api_tasks"
EX = "ADUAED21041WKLX30"

FILTER_SBATCH = "/media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/infer_dgx_filter.sh"
TOTAL_SBATCH  = "/media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/infer_dgx_total.sh"
TG_SBATCH     = "/media/cvpr/zhewen/cv/tail_glitch/infer_dgx_tail_glitch.sh"
SR_SBATCH     = "/media/cvpr/zhewen/cv/RealESR/infer_dgx_realesr.sh"

SSH_CONC = 8
POLL = 30


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, r.stdout + r.stderr


def ssh(cmd):
    return run(["ssh", DGX, cmd])


def load_records():
    rs = []
    with MANIFEST.open() as f:
        for line in f:
            line = line.strip()
            if line: rs.append(json.loads(line))
    return rs


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=2))


def query_dgx_mimic_set():
    """Return set of c701_NNNN that already have output/input_hiya.mp4."""
    rc, out = ssh("ls /media/cvpr/zhewen/api_tasks/c701_*/output/input_hiya.mp4 2>/dev/null "
                  "| sed -E 's:.*/c701_::; s:/output/input_hiya.mp4::'")
    return set(out.split())


def cmd_prep():
    """Pre-stage: cp output/input_hiya.mp4 → input_video.mp4 for tasks that have mimic.
    Filter sbatch (used for the 549 with-mimic group) consumes input_video.mp4.
    """
    have_mimic = query_dgx_mimic_set()
    print(f"prep: {len(have_mimic)} tasks have mimic; copying input_hiya → input_video on DGX")
    if not have_mimic:
        return
    cmd = "for sid in " + " ".join(sorted(have_mimic)) + ' ; do '
    cmd += '  d="/media/cvpr/zhewen/api_tasks/c701_$sid"; '
    cmd += '  [ -s "$d/output/input_hiya.mp4" ] && cp -f "$d/output/input_hiya.mp4" "$d/input_video.mp4"; '
    cmd += "done; echo done"
    rc, out = ssh(cmd)
    print(out.strip().splitlines()[-1] if out else "(no out)")


def submit_one(sid_int: int, has_mimic: bool) -> dict:
    """Submit the filter→TG→SR chain (or total→TG→SR for non-mimic). Returns SR job_id."""
    sid = f"{sid_int:04d}"
    tid = f"c701_{sid}"
    task_dir = f"{TR}/{tid}"

    if has_mimic:
        # 3-stage: filter → TG → SR. Filter consumes input_video.mp4 (prepped).
        chain = f"""
J1=$(sbatch --parsable --exclude={EX} --chdir=/tmp \
  --export=ALL,TASK_ID={tid},FILTER_HEAD_TAIL=true,FILTER_DUPLICATE=false,FILTER_POSE=false,ACTIVITY_THRESHOLD=0.7 \
  {FILTER_SBATCH})
J2=$(sbatch --parsable --exclude={EX} --chdir=/tmp --dependency=afterok:$J1 \
  --export=ALL,INPUT_VIDEO={task_dir}/output/input.mp4,OUTPUT_VIDEO={task_dir}/tail_glitch.mp4 \
  {TG_SBATCH})
J3=$(sbatch --parsable --exclude={EX} --chdir=/tmp --dependency=afterok:$J2 \
  --export=ALL,INPUT_VIDEO={task_dir}/tail_glitch.mp4,OUTPUT_VIDEO={task_dir}/sr.mp4 \
  {SR_SBATCH})
echo "$J1 $J2 $J3"
"""
    else:
        # 4-stage: total (mimic+filter) → TG → SR. Output of total is output_filter/input.mp4.
        chain = f"""
J1=$(sbatch --parsable --exclude={EX} --chdir=/tmp \
  --export=ALL,TASK_ID={tid},FILTER_HEAD_TAIL=true,FILTER_DUPLICATE=false,FILTER_POSE=false,ACTIVITY_THRESHOLD=0.7 \
  {TOTAL_SBATCH})
J2=$(sbatch --parsable --exclude={EX} --chdir=/tmp --dependency=afterok:$J1 \
  --export=ALL,INPUT_VIDEO={task_dir}/output_filter/input.mp4,OUTPUT_VIDEO={task_dir}/tail_glitch.mp4 \
  {TG_SBATCH})
J3=$(sbatch --parsable --exclude={EX} --chdir=/tmp --dependency=afterok:$J2 \
  --export=ALL,INPUT_VIDEO={task_dir}/tail_glitch.mp4,OUTPUT_VIDEO={task_dir}/sr.mp4 \
  {SR_SBATCH})
echo "$J1 $J2 $J3"
"""
    rc, out = ssh(chain.strip())
    line = (out.splitlines() or [""])[-1].strip()
    parts = line.split()
    if rc == 0 and len(parts) == 3 and all(p.isdigit() for p in parts):
        return {"sid": sid_int, "task_id": tid, "has_mimic": has_mimic,
                "j1": parts[0], "j2": parts[1], "j3": parts[2], "ok": True}
    return {"sid": sid_int, "task_id": tid, "has_mimic": has_mimic,
            "ok": False, "err": (out or "")[:300]}


def cmd_submit(records):
    have_mimic = query_dgx_mimic_set()
    print(f"submit: {len(records)} records, {len(have_mimic)} have mimic, "
          f"{len(records) - len(have_mimic)} need full pipeline")
    results = []
    with ThreadPoolExecutor(max_workers=SSH_CONC) as ex:
        futs = {ex.submit(submit_one, r["new_sid"], f"{r['new_sid']:04d}" in have_mimic): r
                for r in records}
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 25 == 0 or i == len(records):
                ok = sum(1 for r in results if r.get("ok"))
                print(f"  submitted {i}/{len(records)}  ok={ok}")
    return results


def cmd_poll(submit_results):
    sr_jobs = [r["j3"] for r in submit_results if r.get("ok")]
    print(f"poll: waiting on {len(sr_jobs)} SR jobs (last stage of each chain)")
    pending = set(sr_jobs)
    t0 = time.time()
    while pending:
        active = set()
        for chunk_start in range(0, len(pending), 200):
            chunk = list(pending)[chunk_start:chunk_start + 200]
            rc, out = ssh(f"squeue -j {','.join(chunk)} -h -o %i 2>/dev/null")
            for line in out.splitlines():
                jid = line.strip()
                if jid: active.add(jid)
        pending = active
        elapsed = int(time.time() - t0)
        print(f"  t+{elapsed:>5}s  active SR jobs={len(pending)}  done={len(sr_jobs) - len(pending)}")
        if not pending: break
        time.sleep(POLL)


def fetch_one(sid_int: int):
    sid = f"{sid_int:04d}"
    tid = f"c701_{sid}"
    final = STAGING / video_filename(f"commence701_{sid}")
    if final.exists() and final.stat().st_size > 0:
        return {"sid": sid_int, "ok": True, "skipped": True}
    raw = STAGING / f".raw_{sid}.mp4"
    rc, _ = run(["scp", "-q", f"{DGX}:{TR}/{tid}/sr.mp4", str(raw)])
    if rc != 0 or not raw.exists() or raw.stat().st_size == 0:
        if raw.exists(): raw.unlink()
        return {"sid": sid_int, "ok": False, "stage": "scp"}
    rc, _ = run([
        str(FFMPEG), "-y", "-loglevel", "error",
        "-i", str(raw),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(final),
    ])
    raw.unlink(missing_ok=True)
    if rc != 0 or not final.exists() or final.stat().st_size == 0:
        return {"sid": sid_int, "ok": False, "stage": "remux"}
    return {"sid": sid_int, "ok": True, "path": str(final)}


def cmd_fetch(records):
    STAGING.mkdir(parents=True, exist_ok=True)
    print(f"fetch: pulling sr.mp4 + h264 remux for {len(records)} records (parallel=4)")
    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(fetch_one, r["new_sid"]) for r in records]
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 25 == 0 or i == len(records):
                ok = sum(1 for x in results if x.get("ok"))
                print(f"  fetched {i}/{len(records)} ok={ok}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "submit", "poll", "fetch", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    records = load_records()
    if args.limit > 0: records = records[: args.limit]
    state = load_state()

    if args.stage in ("prep", "all"):
        cmd_prep()

    if args.stage in ("submit", "all"):
        state["submit_results"] = cmd_submit(records)
        save_state(state)

    if args.stage in ("poll", "all"):
        cmd_poll(state.get("submit_results", []))

    if args.stage in ("fetch", "all"):
        state["fetch_results"] = cmd_fetch(records)
        save_state(state)

    print("done.")


if __name__ == "__main__":
    main()
