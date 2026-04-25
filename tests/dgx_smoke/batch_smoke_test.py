#!/usr/bin/env python3
"""Parallel DGX smoke test: submit 5 sentence videos concurrently, watch DGX
run them (up to however many the spark partition allows simultaneously),
fetch results, h264-remux, and inject each as a Phase3TestJob.

Tests three things end-to-end:
  1. DGX partition actually runs jobs in parallel (squeue shows >1 RUNNING)
  2. Our local scheduler fires 5 independent sbatch submissions without stepping
     on each other (unique task_ids, unique JOBIDs, no mixed files)
  3. All 5 finish and their outputs come back viewable on /phase3-test

Assumes SSH key auth to dgx-login already set up.
"""
import shlex
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/home/chatsign/lizh/chatsign-auto")
from sqlmodel import Session

from backend.config import settings
from backend.database import engine
from backend.models.phase3_test import Phase3TestJob


DGX_HOST = "dgx-login"
DGX_TASKS_ROOT = "/media/cvpr/zhewen/api_tasks"
DGX_SBATCH = "/media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/infer_dgx_task.sh"
DGX_LOGS = "/media/cvpr/zhewen/logs"
POLL_SEC = 30

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out_batch"
OUT_DIR.mkdir(exist_ok=True)
FFMPEG = "/home/chatsign/lizh/chatsign-auto/bin/ffmpeg"
REF_IMAGE = Path("/home/chatsign/lizh/chatsign-auto/UniSignMimicTurbo/assets/example_data/images/test2.jpg")
ACCURACY_ROOT = Path("/home/chatsign/lizh/chatsign-auto/chatsign-accuracy/backend/data/review/generated")

# (video_id, filename, sentence_text)
TARGETS = [
    ("gen_006", "006_sorry_i_m_late.mp4",          "Sorry, I'm late."),
    ("gen_023", "023_what_time_is_boarding.mp4",   "What time is boarding?"),
    ("gen_030", "030_is_this_seat_taken.mp4",      "Is this seat taken?"),
    ("gen_067", "067_please_wait_for_your_turn.mp4", "Please wait for your turn."),
    ("gen_070", "070_please_sign_here.mp4",        "Please sign here."),
]

_print_lock = threading.Lock()
def log(msg: str) -> None:
    with _print_lock:
        print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


def run(cmd: list[str], check: bool = True) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{out}")
    return p.returncode, out


def ssh(cmd: str, check: bool = True) -> tuple[int, str]:
    return run(["ssh", DGX_HOST, cmd], check=check)


def scp_up(local: Path, remote: str) -> None:
    run(["scp", "-q", str(local), f"{DGX_HOST}:{remote}"])


def scp_down(remote: str, local: Path) -> int:
    rc, _ = run(["scp", "-q", f"{DGX_HOST}:{remote}", str(local)], check=False)
    return rc


def submit_one(video_id: str, filename: str) -> dict:
    """Upload inputs + sbatch, return {task_id, job_id, t_submit}."""
    task_id = f"bt_{uuid.uuid4().hex[:10]}"
    remote_dir = f"{DGX_TASKS_ROOT}/{task_id}"
    src = ACCURACY_ROOT / filename

    log(f"[{video_id}] task_id={task_id}  uploading ({src.stat().st_size//1024}KB)")
    ssh(f"mkdir -p {shlex.quote(remote_dir + '/videos')} {shlex.quote(remote_dir + '/output')}")
    scp_up(src, f"{remote_dir}/videos/input.mp4")
    scp_up(REF_IMAGE, f"{remote_dir}/input_image.png")

    _, out = ssh(
        f"sbatch --parsable --export=ALL,TASK_ID={shlex.quote(task_id)} "
        f"{shlex.quote(DGX_SBATCH)}"
    )
    job_id = out.strip().splitlines()[-1].split(";")[0].strip()
    if not job_id.isdigit():
        raise RuntimeError(f"bad JOBID for {video_id}: {out!r}")
    t_submit = time.time()
    log(f"[{video_id}] JOBID={job_id} submitted")
    return {
        "video_id": video_id,
        "filename": filename,
        "task_id": task_id,
        "job_id": job_id,
        "remote_dir": remote_dir,
        "t_submit": t_submit,
    }


def poll_one(job: dict) -> dict:
    """Poll squeue until job leaves queue, then fetch result + remux h264."""
    video_id = job["video_id"]
    job_id = job["job_id"]
    task_id = job["task_id"]
    t_start_run = None

    while True:
        time.sleep(POLL_SEC)
        _, out = ssh(f"squeue -j {job_id} -h -o %T 2>/dev/null", check=False)
        state = (out.strip().splitlines() or [""])[-1].strip()
        elapsed = int(time.time() - job["t_submit"])
        if not state:
            log(f"[{video_id}] t+{elapsed}s  job left queue")
            break
        if state == "RUNNING" and t_start_run is None:
            t_start_run = time.time()
        log(f"[{video_id}] t+{elapsed}s  state={state}")

    # Fetch result
    local_raw = OUT_DIR / f"{task_id}_raw.mp4"
    rc = scp_down(f"{job['remote_dir']}/output/input_hiya.mp4", local_raw)
    if rc != 0 or not local_raw.exists() or local_raw.stat().st_size == 0:
        log(f"[{video_id}] FAIL — no output mp4")
        _, logs = ssh(
            f"echo '===OUT==='; tail -40 {DGX_LOGS}/mm_imitate_{job_id}.out 2>/dev/null; "
            f"echo '===ERR==='; tail -30 {DGX_LOGS}/mm_imitate_{job_id}.err 2>/dev/null "
            f"| tr '\\r' '\\n' | tail -30",
            check=False,
        )
        job.update(status="failed", error=logs[-1500:])
        return job

    # h264 remux for browser
    local_h264 = OUT_DIR / f"{task_id}_h264.mp4"
    run([FFMPEG, "-y", "-loglevel", "error", "-i", str(local_raw),
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(local_h264)])

    total = time.time() - job["t_submit"]
    log(f"[{video_id}] DONE  total={total:.0f}s  output={local_h264.stat().st_size//1024}KB")
    job.update(
        status="completed",
        raw_mp4=str(local_raw),
        h264_mp4=str(local_h264),
        total_sec=round(total, 1),
    )
    return job


def inject_db(job: dict, sentence_text: str) -> str:
    """Insert Phase3TestJob row. Returns db job_id."""
    db_job_id = f"dgxbt_{uuid.uuid4().hex[:8]}"
    output_dir = settings.PHASE3_TEST_OUTPUT_DIR / db_job_id
    (output_dir / "input").mkdir(parents=True, exist_ok=True)

    src = ACCURACY_ROOT / job["filename"]
    staged_src = output_dir / "input" / job["filename"]
    import shutil
    shutil.copy2(src, staged_src)
    gen_path = output_dir / "generated.mp4"
    shutil.copy2(job["h264_mp4"], gen_path)

    row = Phase3TestJob(
        job_id=db_job_id,
        status=job["status"],
        video_id=job["video_id"],
        sentence_text=sentence_text,
        translator_id="generated",
        source_video_path=str(staged_src),
        source_filename=job["filename"],
        output_dir=str(output_dir),
        generated_video_path=str(gen_path),
        duration_sec=job.get("total_sec"),
        transfer_time_sec=job.get("total_sec"),
        process_time_sec=0.0,
        framer_time_sec=0.0,
        error_message=f"[DGX batch smoke] sbatch JOBID={job['job_id']} task={job['task_id']}",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    with Session(engine) as session:
        session.add(row)
        session.commit()
    return db_job_id


def main() -> int:
    t0 = time.time()
    log(f"=== submit {len(TARGETS)} jobs in parallel ===")
    submits = []
    with ThreadPoolExecutor(max_workers=len(TARGETS)) as pool:
        futures = [pool.submit(submit_one, vid, fn) for vid, fn, _ in TARGETS]
        for f in as_completed(futures):
            submits.append(f.result())

    log(f"all {len(submits)} submitted in {time.time() - t0:.0f}s")

    # Let DGX see them, then check queue state
    time.sleep(5)
    _, q = ssh("squeue -u cvpr -h -o '%i %T %M %N' 2>/dev/null", check=False)
    log(f"squeue snapshot just after submit:\n{q}")

    log(f"=== poll {len(submits)} jobs in parallel ===")
    results = []
    with ThreadPoolExecutor(max_workers=len(submits)) as pool:
        futures = [pool.submit(poll_one, j) for j in submits]
        for f in as_completed(futures):
            results.append(f.result())

    log(f"\n=== inject DB rows ===")
    text_by_vid = {vid: txt for vid, _, txt in TARGETS}
    ok = fail = 0
    for r in results:
        if r["status"] != "completed":
            fail += 1
            log(f"[{r['video_id']}] SKIP inject (failed)")
            continue
        db_id = inject_db(r, text_by_vid[r["video_id"]])
        ok += 1
        log(f"[{r['video_id']}] injected db_job_id={db_id}")

    wall = time.time() - t0
    log(f"\n=== SUMMARY ===")
    log(f"wall={wall:.0f}s  ok={ok}  failed={fail}")
    for r in sorted(results, key=lambda x: x.get("total_sec", 0)):
        if r["status"] == "completed":
            log(f"  {r['video_id']:8s} total={r['total_sec']:.0f}s  job={r['job_id']}")
        else:
            log(f"  {r['video_id']:8s} FAILED  job={r['job_id']}")
    log(f"page: https://auto.chatsign.ai/phase3-test")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
