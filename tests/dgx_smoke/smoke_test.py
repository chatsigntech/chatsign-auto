#!/usr/bin/env python3
"""DGX smoke test: send one video + reference image, wait for result mp4.

End-to-end check that the Phase 3 scheduler path works:
    ssh mkdir -> scp inputs -> sbatch -> poll squeue -> scp back result

Everything talks to `dgx-login` (see ~/.ssh/config). Assumes SSH key auth
already set up (no password).

Usage:
    python smoke_test.py
    python smoke_test.py --video /path/to/input.mp4 --image /path/to/ref.jpg
"""
import argparse
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path


DGX_HOST = "dgx-login"
DGX_TASKS_ROOT = "/media/cvpr/zhewen/api_tasks"
DGX_SBATCH = "/media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/infer_dgx_task.sh"
DGX_LOGS = "/media/cvpr/zhewen/logs"
POLL_SEC = 30

HERE = Path(__file__).resolve().parent
DEFAULT_VIDEO = Path("/home/chatsign/lizh/chatsign-auto/chatsign-accuracy/backend/data/review/generated/001_hello.mp4")
DEFAULT_IMAGE = Path("/home/chatsign/lizh/chatsign-auto/UniSignMimicTurbo/assets/example_data/images/test2.jpg")


def run(cmd: list[str], check: bool = True) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        print(f"[FAIL] {' '.join(cmd)}\n{out}")
        sys.exit(1)
    return p.returncode, out


def ssh(cmd: str, check: bool = True) -> tuple[int, str]:
    return run(["ssh", DGX_HOST, cmd], check=check)


def scp_up(local: Path, remote: str) -> None:
    print(f"  scp ↑ {local.name} → {remote}")
    run(["scp", "-q", str(local), f"{DGX_HOST}:{remote}"])


def scp_down(remote: str, local: Path) -> int:
    rc, _ = run(["scp", "-q", f"{DGX_HOST}:{remote}", str(local)], check=False)
    return rc


_DWPOSE_RE = re.compile(r"DWPose:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)")
_DENOISE_RE = re.compile(r"(?<!DWPose:)\s(\d+)%\|[^|]*\|\s*(\d+)/(\d+)")


def parse_progress(log_tail: str) -> str:
    """Return a short human-readable progress line from tqdm in stderr."""
    dwpose = denoise = None
    for line in log_tail.splitlines():
        m = _DWPOSE_RE.search(line)
        if m:
            dwpose = f"DWPose {m.group(1)}/{m.group(2)}"
        m = _DENOISE_RE.search(line)
        if m:
            denoise = f"denoise {m.group(2)}/{m.group(3)}"
    return denoise or dwpose or "…"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--out", type=Path, default=HERE / "out")
    args = ap.parse_args()

    for p, name in [(args.video, "video"), (args.image, "image")]:
        if not p.exists():
            print(f"[FAIL] {name} not found: {p}")
            return 1

    task_id = f"smoke_{uuid.uuid4().hex[:12]}"
    remote_dir = f"{DGX_TASKS_ROOT}/{task_id}"
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[1/6] task_id = {task_id}")
    print(f"      video   = {args.video}  ({args.video.stat().st_size/1024:.0f} KB)")
    print(f"      image   = {args.image}")

    print(f"[2/6] mkdir remote: {remote_dir}")
    ssh(f"mkdir -p {shlex.quote(remote_dir + '/videos')} {shlex.quote(remote_dir + '/output')}")

    print("[3/6] upload inputs")
    scp_up(args.video, f"{remote_dir}/videos/input.mp4")
    scp_up(args.image, f"{remote_dir}/input_image.png")

    print("[4/6] sbatch submit")
    rc, out = ssh(
        f"sbatch --parsable --export=ALL,TASK_ID={shlex.quote(task_id)} "
        f"{shlex.quote(DGX_SBATCH)}"
    )
    job_id = out.strip().splitlines()[-1].split(";")[0].strip()
    if not job_id.isdigit():
        print(f"[FAIL] bad JOBID: {out!r}")
        return 1
    print(f"      JOBID = {job_id}")

    print(f"[5/6] poll squeue every {POLL_SEC}s (job, state, progress)")
    last_state = ""
    while True:
        time.sleep(POLL_SEC)
        _, out = ssh(f"squeue -j {job_id} -h -o %T 2>/dev/null", check=False)
        state = (out.strip().splitlines() or [""])[-1].strip()
        elapsed = int(time.time() - t0)
        if not state:
            print(f"      t+{elapsed}s: job left queue")
            break
        _, tail = ssh(
            f"tail -c 4000 {DGX_LOGS}/mm_imitate_{job_id}.err 2>/dev/null "
            f"| tr '\\r' '\\n' | tail -20",
            check=False,
        )
        prog = parse_progress(tail)
        if state != last_state or prog != "…":
            print(f"      t+{elapsed}s: state={state}  progress={prog}")
            last_state = state

    print("[6/6] fetch result")
    _, state_final = ssh(
        f"sacct -j {job_id} -X --noheader -P -o State 2>&1 | head -1",
        check=False,
    )
    state_final = state_final.strip() or "UNKNOWN"
    print(f"      sacct final state = {state_final}")

    local_result = args.out / f"{task_id}_result.mp4"
    rc = scp_down(f"{remote_dir}/output/input_hiya.mp4", local_result)
    if rc != 0 or not local_result.exists() or local_result.stat().st_size == 0:
        print(f"[FAIL] result not fetched. Dumping log tails:")
        _, logs = ssh(
            f"echo '===OUT==='; tail -60 {DGX_LOGS}/mm_imitate_{job_id}.out 2>/dev/null; "
            f"echo '===ERR==='; tail -60 {DGX_LOGS}/mm_imitate_{job_id}.err 2>/dev/null "
            f"| tr '\\r' '\\n' | tail -60",
            check=False,
        )
        print(logs[-3000:])
        return 1

    size_kb = local_result.stat().st_size / 1024
    total = int(time.time() - t0)
    print(f"\n[OK] result: {local_result}  ({size_kb:.0f} KB)  wall={total}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
