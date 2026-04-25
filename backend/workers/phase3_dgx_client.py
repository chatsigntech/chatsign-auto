"""Phase 3 DGX client — pure scheduler.

Sends N word videos (+ a fixed reference image) to the DGX `spark` partition
in parallel via sbatch, polls every job in a single aggregated `squeue`,
fetches the resulting standardised sign-language mp4s back to
`output_dir/videos/`, and writes `phase3_report.json`.

This module owns nothing about MimicMotion / DWPose / FramerTurbo internals —
that all lives in `infer_dgx_task.sh` on DGX. We just submit, wait, and pull.

Assumes:
  - SSH key auth to `dgx-login` is already set up (no password prompt).
  - DGX has the sbatch entrypoint at $DGX_SBATCH (default points at the
    UniSignMimicTurbo MimicMotion task wrapper).
  - The shared task root $DGX_TASKS_ROOT is writable by the cvpr user.
"""
import asyncio
import json
import logging
import os
import shlex
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


DGX_HOST = os.environ.get("DGX_HOST", "dgx-login")
DGX_TASKS_ROOT = os.environ.get("DGX_TASKS_ROOT", "/media/cvpr/zhewen/api_tasks")
DGX_SBATCH_SCRIPT = os.environ.get(
    "DGX_SBATCH_SCRIPT",
    "/media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/infer_dgx_task.sh",
)
DGX_LOGS_DIR = os.environ.get("DGX_LOGS_DIR", "/media/cvpr/zhewen/logs")
POLL_INTERVAL_SEC = int(os.environ.get("DGX_POLL_INTERVAL_SEC", "30"))
FFMPEG_CONCURRENCY = int(os.environ.get("DGX_FFMPEG_CONCURRENCY", "4"))
# Cap concurrent ssh/scp to stay under sshd MaxStartups (default 10:30:100).
SSH_CONCURRENCY = int(os.environ.get("DGX_SSH_CONCURRENCY", "6"))

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FFMPEG = _REPO_ROOT / "bin" / "ffmpeg"

# Reference image lives on DGX itself (shipped with UniSignMimicTurbo).
# Local never needs to upload one — every per-task `input_image.png` is just
# a symlink to this path.
DGX_REF_IMAGE = os.environ.get(
    "DGX_REF_IMAGE",
    "/media/cvpr/zhewen/UniSignMimicTurbo/mimicmotion/data/ref_images/test4.jpg",
)


ProgressCb = Callable[[float], Awaitable[None] | None]

_ssh_sem: asyncio.Semaphore | None = None


def _get_ssh_sem() -> asyncio.Semaphore:
    """Lazily build the SSH concurrency semaphore inside the running loop."""
    global _ssh_sem
    if _ssh_sem is None:
        _ssh_sem = asyncio.Semaphore(SSH_CONCURRENCY)
    return _ssh_sem


async def _run(cmd: list[str]) -> tuple[int, str]:
    async with _get_ssh_sem():
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace")


async def _ssh(cmd: str) -> tuple[int, str]:
    return await _run(["ssh", DGX_HOST, cmd])


async def _scp_up(local: Path, remote: str) -> tuple[int, str]:
    return await _run(["scp", "-q", str(local), f"{DGX_HOST}:{remote}"])


async def _scp_down(remote: str, local: Path) -> tuple[int, str]:
    return await _run(["scp", "-q", f"{DGX_HOST}:{remote}", str(local)])


async def _emit_progress(progress_cb: ProgressCb | None, value: float, last: list[float]) -> None:
    """Emit progress, deduplicating consecutive identical values."""
    if progress_cb is None:
        return
    if last and abs(value - last[-1]) < 0.5:
        return
    last.append(value)
    result = progress_cb(value)
    if asyncio.iscoroutine(result):
        await result


def _failed(filename: str, stage: str, error: str) -> dict:
    return {"filename": filename, "status": "failed", "stage": stage, "error": error[-500:]}


async def _submit_one(task_id: str, video: Path, shared_ref_remote: str) -> dict:
    """Upload one video, symlink the shared ref image, submit sbatch."""
    sub_task_id = f"{task_id}_{video.stem}_{uuid.uuid4().hex[:6]}"
    remote_dir = f"{DGX_TASKS_ROOT}/{sub_task_id}"

    rc, out = await _ssh(
        f"mkdir -p {shlex.quote(remote_dir + '/videos')} {shlex.quote(remote_dir + '/output')}"
    )
    if rc != 0:
        return _failed(video.name, "mkdir", out)

    # Video upload + ref image copy can run in parallel.
    # `cp` (not `ln -s`) — the SMB share rejects symlink creation despite
    # `symlink=native` in the mount options. Server-side cp is fast.
    (rc_v, out_v), (rc_c, out_c) = await asyncio.gather(
        _scp_up(video, f"{remote_dir}/videos/input.mp4"),
        _ssh(f"cp {shlex.quote(shared_ref_remote)} {shlex.quote(remote_dir + '/input_image.png')}"),
    )
    if rc_v != 0:
        return _failed(video.name, "scp_video", out_v)
    if rc_c != 0:
        return _failed(video.name, "cp_image", out_c)

    rc, out = await _ssh(
        f"sbatch --parsable --export=ALL,TASK_ID={shlex.quote(sub_task_id)} "
        f"{shlex.quote(DGX_SBATCH_SCRIPT)}"
    )
    if rc != 0:
        return _failed(video.name, "sbatch", out)
    job_id = out.strip().splitlines()[-1].split(";")[0].strip()
    if not job_id.isdigit():
        return _failed(video.name, "sbatch_parse", f"bad JOBID: {out[-300:]}")

    return {
        "filename": video.name,
        "status": "submitted",
        "sub_task_id": sub_task_id,
        "remote_dir": remote_dir,
        "job_id": job_id,
        "t_submit": time.time(),
    }


async def _poll_all_until_done(
    submitted: list[dict],
    progress_cb: ProgressCb | None,
    n_total: int,
    last_emitted: list[float],
) -> None:
    """Single aggregated `squeue` per cycle marks all jobs done as they leave the queue.

    Mutates each submitted dict in place: sets `wall_sec` when the job finishes.
    """
    pending = {s["job_id"]: s for s in submitted}
    while pending:
        await asyncio.sleep(POLL_INTERVAL_SEC)
        ids_csv = ",".join(pending.keys())
        rc, out = await _ssh(f"squeue -j {ids_csv} -h -o %i 2>/dev/null")
        if rc != 0:
            continue
        active = {line.strip() for line in out.splitlines() if line.strip()}
        finished = [jid for jid in pending if jid not in active]
        for jid in finished:
            sub = pending.pop(jid)
            sub["wall_sec"] = round(time.time() - sub["t_submit"], 1)
        if finished:
            n_done = n_total - len(pending)
            await _emit_progress(progress_cb, 5.0 + 65.0 * n_done / max(1, n_total), last_emitted)


async def _fetch_one(sub: dict, output_videos_dir: Path, ffmpeg_sem: asyncio.Semaphore) -> dict:
    """Pull result mp4 back, h264-remux to output_videos_dir using original filename."""
    target_name = sub["filename"]
    raw_local = output_videos_dir / f".raw_{target_name}"
    final_local = output_videos_dir / target_name

    rc, out = await _scp_down(
        f"{sub['remote_dir']}/output/input_hiya.mp4", raw_local,
    )
    if rc != 0 or not raw_local.exists() or raw_local.stat().st_size == 0:
        _, logs = await _ssh(
            f"echo '===OUT==='; tail -40 {DGX_LOGS_DIR}/mm_imitate_{sub['job_id']}.out 2>/dev/null; "
            f"echo '===ERR==='; tail -30 {DGX_LOGS_DIR}/mm_imitate_{sub['job_id']}.err 2>/dev/null "
            f"| tr '\\r' '\\n' | tail -30",
        )
        sub.update(_failed(target_name, "fetch", out + "\n" + logs))
        return sub

    async with ffmpeg_sem:
        rc, _ = await _run([
            str(FFMPEG), "-y", "-loglevel", "error",
            "-i", str(raw_local),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(final_local),
        ])
    raw_local.unlink(missing_ok=True)
    if rc != 0 or not final_local.exists() or final_local.stat().st_size == 0:
        sub.update(_failed(target_name, "remux", "h264 remux failed"))
        return sub

    sub["status"] = "completed"
    sub["output_path"] = str(final_local)
    sub["output_size"] = final_local.stat().st_size
    return sub


async def _cleanup_remote(sub_task_id: str) -> None:
    """Best-effort: remove the per-task dir on DGX. Failure is non-fatal."""
    remote = f"{DGX_TASKS_ROOT}/{sub_task_id}"
    await _ssh(f"rm -rf {shlex.quote(remote)}")


async def run_phase3_on_dgx(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
    progress_cb: ProgressCb | None = None,
) -> dict:
    """Run full Phase 3 on DGX for every *.mp4 in input_dir.

    Reference image lives on DGX at $DGX_REF_IMAGE — never uploaded from local.

    Args:
        task_id: pipeline task id (used as a logging prefix and as a
            namespace for sub-task ids on DGX).
        input_dir: directory of input *.mp4 word videos.
        output_dir: directory where `videos/<original_name>.mp4` and
            `phase3_report.json` will be written.
        progress_cb: optional async/sync callable receiving 0-100 floats.

    Returns the report dict (also written to disk).
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    videos_out = output_dir / "videos"
    videos_out.mkdir(parents=True, exist_ok=True)

    all_mp4 = sorted(input_dir.glob("*.mp4"))
    videos = [v for v in all_mp4 if not v.name.startswith("sentence_")]
    skipped = len(all_mp4) - len(videos)
    if skipped:
        logger.info(f"[{task_id}] Phase 3 (DGX): skipped {skipped} sentence_* videos")
    if not videos:
        raise RuntimeError(f"Phase 3 (DGX): no word videos found in {input_dir}")

    logger.info(f"[{task_id}] Phase 3 (DGX): submitting {len(videos)} videos to {DGX_HOST} "
                f"(ref={DGX_REF_IMAGE})")
    t_start = time.time()
    last_emitted: list[float] = []
    await _emit_progress(progress_cb, 1.0, last_emitted)

    # ── Stage 1: parallel submit ───────────────────────────────────────
    submissions = await asyncio.gather(
        *[_submit_one(task_id, v, DGX_REF_IMAGE) for v in videos]
    )
    submitted = [s for s in submissions if s["status"] == "submitted"]
    pre_failed = [s for s in submissions if s["status"] != "submitted"]

    n_total = len(submissions)
    logger.info(f"[{task_id}] Phase 3 (DGX): submitted {len(submitted)}/{n_total} "
                f"({len(pre_failed)} failed at submit)")
    await _emit_progress(progress_cb, 5.0, last_emitted)

    # ── Stage 2: aggregated poll until every job leaves the queue ──────
    await _poll_all_until_done(submitted, progress_cb, n_total, last_emitted)
    await _emit_progress(progress_cb, 70.0, last_emitted)

    # ── Stage 3: parallel fetch + h264 remux (capped) ──────────────────
    ffmpeg_sem = asyncio.Semaphore(FFMPEG_CONCURRENCY)
    fetched = await asyncio.gather(*[_fetch_one(s, videos_out, ffmpeg_sem) for s in submitted])
    await _emit_progress(progress_cb, 95.0, last_emitted)

    # ── Stage 4: best-effort remote cleanup ────────────────────────────
    await asyncio.gather(
        *[_cleanup_remote(s["sub_task_id"]) for s in submitted],
        return_exceptions=True,
    )

    # ── Build report (compatible with old phase4/phase6 summary keys) ──
    all_results = pre_failed + fetched
    n_completed = sum(1 for r in all_results if r.get("status") == "completed")
    n_failed = n_total - n_completed

    wall = round(time.time() - t_start, 1)
    report = {
        "task_id": task_id,
        "backend": "dgx",
        "input_videos": n_total,
        "transfer_success": n_completed,
        "transfer_failed": n_failed,
        "transfer_skipped": skipped,
        "videos_generated": n_completed,
        "interpolation_mode": "dgx",
        "total_wall_seconds": wall,
        "results": all_results,
    }
    report_path = output_dir / "phase3_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"[{task_id}] Phase 3 (DGX) done: {n_completed}/{n_total} videos "
                f"in {wall/60:.1f} min")
    await _emit_progress(progress_cb, 100.0, last_emitted)

    if n_completed == 0:
        raise RuntimeError(f"Phase 3 (DGX): all {n_total} videos failed")

    return report
