#!/usr/bin/env python3
"""hiya_keynote_2026 (63) → DGX 4-stage chain (total → TG → SR).

Same flow as run_26potential_script_approved.py for the 63 sentence-level
recordings of the HiYa keynote demo script (38 MC + 25 guest, mirrored
2026-05-07; sources: drive-download + SecondPart-Tariq zips).

Per-sid pipeline (no cached mimic, all fresh):
    sbatch infer_dgx_total.sh   (J1: mimic + filter, output_filter/input.mp4)
    sbatch infer_dgx_tail_glitch.sh --dependency=afterok:J1   (J2: tail_glitch.mp4)
    sbatch infer_dgx_realesr.sh    --dependency=afterok:J2   (J3: sr.mp4)
    local: scp sr.mp4 → h264 remux → STAGING/<md5(c_hiya26_<sid>)[:10]>_hiya.mp4

Stages:
    --stage upload  : mkdir DGX task dir, copy ref image, scp input.mp4 (parallel)
    --stage submit  : submit total→TG→SR chain per sid, store J3 in state
    --stage poll    : wait until all J3 (SR) jobs leave the queue
    --stage fetch   : scp sr.mp4, h264 remux to videos/<sid>.mp4
    --stage all     : upload + submit + poll + fetch

Excludes:
    ADUAED21041WKLX30  (autofs broken — per memory)
    ADUAED21034WKLX25  (down per current sinfo)

Idempotent: skips uploads / fetches that are already complete.
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _naming import video_filename


REPO = Path("/home/chatsign/lizh/chatsign-auto")
MANIFEST = REPO / "dgx-pipeline-test" / "hiya_keynote_2026_manifest.jsonl"
STATE = REPO / "dgx-pipeline-test" / "hiya_keynote_2026_state.json"
STAGING = Path("/mnt/data/chatsign-phase3-test/hiya_keynote_2026-render-20260507/videos")
FFMPEG = REPO / "bin" / "ffmpeg"

DGX = "dgx-login"
TR = "/media/cvpr/zhewen/api_tasks"
DGX_REF_IMAGE = "/media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/data/ref_images/test4.jpg"

# Sbatch chains (matches commence-701 v2)
TOTAL_SBATCH  = "/media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/infer_dgx_total.sh"
TG_SBATCH     = "/media/cvpr/zhewen/cv/tail_glitch/infer_dgx_tail_glitch.sh"
SR_SBATCH     = "/media/cvpr/zhewen/cv/RealESR/infer_dgx_realesr.sh"

# Excludes
EX = "ADUAED21041WKLX30,ADUAED21034WKLX25,ADUAED21018WKLX24"

# Per-sid task id
PREFIX = "c_hiya26"

SSH_CONC = 8
POLL = 30


def task_id_for(rec):
    return f"{PREFIX}_{rec['new_sid']:04d}"


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r.returncode, r.stdout + r.stderr


def ssh(cmd):
    return run(["ssh", DGX, cmd])


def scp_up(local, remote):
    return run(["scp", "-q", str(local), f"{DGX}:{remote}"])


def scp_down(remote, local):
    return run(["scp", "-q", f"{DGX}:{remote}", str(local)])


def load_records():
    rs = []
    with MANIFEST.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rs.append(json.loads(line))
    return rs


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=2))


# ── Stage 1: upload ─────────────────────────────────────────────────

def stage_one(rec):
    tid = task_id_for(rec)
    tdir = f"{TR}/{tid}"
    rc, out = ssh(
        f"mkdir -p '{tdir}/videos' '{tdir}/output' '{tdir}/output_filter' && "
        f"if [ ! -f '{tdir}/input_image.png' ]; then cp '{DGX_REF_IMAGE}' '{tdir}/input_image.png'; fi && "
        f"echo OK"
    )
    if rc != 0 or "OK" not in out:
        return {"new_sid": rec["new_sid"], "task_id": tid, "stage": "mkdir", "ok": False, "err": out[:300]}

    # idempotent: skip scp if input already there
    rc, out = ssh(f"test -s '{tdir}/videos/input.mp4' && echo HAS")
    if "HAS" in out:
        return {"new_sid": rec["new_sid"], "task_id": tid, "stage": "upload", "ok": True, "skipped": True}

    rc, out = scp_up(Path(rec["localPath"]), f"{tdir}/videos/input.mp4")
    return {
        "new_sid": rec["new_sid"], "task_id": tid, "stage": "upload",
        "ok": rc == 0, "err": "" if rc == 0 else out[:300],
    }


def cmd_upload(records):
    print(f"[upload] {len(records)} videos to DGX (parallel={SSH_CONC})…")
    results = []
    with ThreadPoolExecutor(max_workers=SSH_CONC) as ex:
        futs = {ex.submit(stage_one, r): r for r in records}
        done = 0
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 25 == 0 or done == len(records):
                fail = sum(1 for r in results if not r.get("ok"))
                skip = sum(1 for r in results if r.get("skipped"))
                print(f"  uploaded {done}/{len(records)} (fail={fail}, skipped={skip})")
    fails = [r for r in results if not r.get("ok")]
    if fails:
        print(f"[upload] {len(fails)} failures:")
        for r in fails[:5]:
            print(f"    sid={r['new_sid']}: {r.get('err','')[:200]}")
    return results


# ── Stage 2: submit chain (total → TG → SR) ─────────────────────────

def submit_one(rec):
    sid = rec["new_sid"]
    tid = task_id_for(rec)
    tdir = f"{TR}/{tid}"
    chain = f"""
J1=$(sbatch --parsable --exclude={EX} --chdir=/tmp \
  --export=ALL,TASK_ID={tid},FILTER_HEAD_TAIL=true,FILTER_DUPLICATE=false,FILTER_POSE=false,ACTIVITY_THRESHOLD=0.7 \
  {TOTAL_SBATCH})
J2=$(sbatch --parsable --exclude={EX} --chdir=/tmp --dependency=afterok:$J1 \
  --export=ALL,INPUT_VIDEO={tdir}/output_filter/input.mp4,OUTPUT_VIDEO={tdir}/tail_glitch.mp4 \
  {TG_SBATCH})
J3=$(sbatch --parsable --exclude={EX} --chdir=/tmp --dependency=afterok:$J2 \
  --export=ALL,INPUT_VIDEO={tdir}/tail_glitch.mp4,OUTPUT_VIDEO={tdir}/sr.mp4 \
  {SR_SBATCH})
echo "$J1 $J2 $J3"
"""
    rc, out = ssh(chain.strip())
    line = (out.splitlines() or [""])[-1].strip()
    parts = line.split()
    if rc == 0 and len(parts) == 3 and all(p.isdigit() for p in parts):
        return {"sid": sid, "task_id": tid,
                "j1": parts[0], "j2": parts[1], "j3": parts[2], "ok": True}
    return {"sid": sid, "task_id": tid, "ok": False, "err": (out or "")[:400]}


def cmd_submit(records):
    print(f"[submit] {len(records)} chains (parallel={SSH_CONC})…")
    results = []
    with ThreadPoolExecutor(max_workers=SSH_CONC) as ex:
        futs = {ex.submit(submit_one, r): r for r in records}
        done = 0
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 25 == 0 or done == len(records):
                ok = sum(1 for r in results if r.get("ok"))
                print(f"  submitted {done}/{len(records)} ok={ok}")
    fails = [r for r in results if not r.get("ok")]
    if fails:
        print(f"[submit] {len(fails)} failures:")
        for r in fails[:5]:
            print(f"    sid={r['sid']}: {r.get('err','')[:200]}")
    return results


# ── Stage 3: poll until SR jobs done ────────────────────────────────

def cmd_poll(submit_results):
    sr_jobs = [r["j3"] for r in submit_results if r.get("ok")]
    print(f"[poll] waiting on {len(sr_jobs)} SR jobs (J3)…")
    pending = set(sr_jobs)
    t0 = time.time()
    while pending:
        active = set()
        for chunk_start in range(0, len(pending), 200):
            chunk = list(pending)[chunk_start:chunk_start + 200]
            rc, out = ssh(f"squeue -j {','.join(chunk)} -h -o %i 2>/dev/null")
            for line in out.splitlines():
                jid = line.strip()
                if jid:
                    active.add(jid)
        pending = active
        elapsed = int(time.time() - t0)
        print(f"  t+{elapsed:>5}s  active SR jobs={len(pending)}  done={len(sr_jobs) - len(pending)}")
        if not pending:
            break
        time.sleep(POLL)


# ── Stage 4: fetch sr.mp4 + h264 remux ──────────────────────────────

def fetch_one(rec):
    sid = rec["new_sid"]
    tid = task_id_for(rec)
    final = STAGING / video_filename(f"{PREFIX}_{sid:04d}")
    if final.exists() and final.stat().st_size > 0:
        return {"sid": sid, "task_id": tid, "ok": True, "skipped": True}
    raw = STAGING / f".raw_{sid:04d}.mp4"
    rc, out = scp_down(f"{TR}/{tid}/sr.mp4", raw)
    if rc != 0 or not raw.exists() or raw.stat().st_size == 0:
        if raw.exists():
            raw.unlink()
        return {"sid": sid, "task_id": tid, "ok": False, "stage": "scp", "err": out[:200]}
    rc, out = run([
        str(FFMPEG), "-y", "-loglevel", "error",
        "-i", str(raw),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(final),
    ])
    raw.unlink(missing_ok=True)
    if rc != 0 or not final.exists() or final.stat().st_size == 0:
        return {"sid": sid, "task_id": tid, "ok": False, "stage": "remux", "err": out[:200]}
    return {"sid": sid, "task_id": tid, "ok": True, "path": str(final)}


def cmd_fetch(records):
    STAGING.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {len(records)} sids (parallel=4)…")
    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(fetch_one, r) for r in records]
        done = 0
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 25 == 0 or done == len(records):
                ok = sum(1 for x in results if x.get("ok"))
                skip = sum(1 for x in results if x.get("skipped"))
                print(f"  fetched {done}/{len(records)} ok={ok} skipped={skip}")
    fails = [r for r in results if not r.get("ok")]
    if fails:
        print(f"[fetch] {len(fails)} failures:")
        for r in fails[:5]:
            print(f"    sid={r['sid']}: stage={r.get('stage')} err={r.get('err','')[:200]}")
    return results


# ── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["upload", "submit", "poll", "fetch", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0,
                    help="run on first N records only (smoke test)")
    ap.add_argument("--manifest", default=None,
                    help="override default MANIFEST path (e.g. for redo sub-manifests)")
    args = ap.parse_args()

    if args.manifest:
        global MANIFEST
        MANIFEST = Path(args.manifest)
        print(f"using manifest override: {MANIFEST}")
    records = load_records()
    if args.limit > 0:
        records = records[: args.limit]
    print(f"manifest: {len(records)} records")
    state = load_state()

    if args.stage in ("upload", "all"):
        state["upload_results"] = cmd_upload(records)
        save_state(state)

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
