"""Phase 3 LOCAL client — same algorithm as DGX, runs on local GPU.

Mirrors `phase3_dgx_client.run_phase3_on_dgx`'s contract (interface,
output layout, report shape) so callers can swap backends with a flag.
Internally invokes the **same** `mimicmotion/batch_process.py` script that
runs on DGX, against a local replica of UniSignMimicTurbo at MIMIC_ROOT.

Identical to DGX:
  - Source: rsync'd from /media/cvpr/zhewen/UniSignMimicTurbo (DGX)
  - Weights: MimicMotion_1-1.pth + DWPose + svd_cache (same files)
  - CLI args: --mode square --crop-anchor top  (matches infer_dgx_task.sh)
  - Reference image: a single image used for every video (default test4.jpg
    from the DGX repo's data/ref_images/, mirrored locally)

Different from DGX:
  - Hardware: local sm_120 vs DGX GB10 — minor numeric drift in CUDA kernels,
    no algorithmic difference.
  - Concurrency: serial per-video (1 GPU). Caller can spawn parallel tasks
    if there's enough VRAM, but defaults to one-at-a-time.
"""
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


MIMIC_ROOT = Path(os.environ.get(
    "PHASE3_MIMIC_ROOT", "/mnt/data/UniSignMimicTurbo_dgx",
))
MIMIC_PYTHON = Path(os.environ.get(
    "PHASE3_MIMIC_PYTHON",
    "/home/chatsign/miniconda3/envs/mimicmotion_dgx/bin/python",
))
DEFAULT_REF_IMAGE = Path(os.environ.get(
    "PHASE3_LOCAL_REF_IMAGE",
    str(MIMIC_ROOT / "mimicmotion" / "data" / "ref_images" / "test4.jpg"),
))
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FFMPEG = _REPO_ROOT / "bin" / "ffmpeg"
FFMPEG_CONCURRENCY = int(os.environ.get("PHASE3_LOCAL_FFMPEG_CONCURRENCY", "4"))


ProgressCb = Callable[[float], Awaitable[None] | None]


async def _emit_progress(progress_cb: ProgressCb | None, value: float, last: list[float]) -> None:
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


def _mimic_env() -> dict:
    """Env vars mirroring DGX's infer_dgx_task.sh — pin HF cache to local svd
    snapshot, run fully offline (no network lookup at model load)."""
    env = os.environ.copy()
    env["HF_HUB_CACHE"] = str(MIMIC_ROOT / "mimicmotion" / "models" / "svd_cache")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


async def _run(cmd: list[str], cwd: Path | None = None,
               env: dict | None = None) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(cwd) if cwd else None,
        env=env,
    )
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace")


async def _process_one(video: Path, ref_image: Path, work_dir: Path,
                       videos_out: Path) -> dict:
    """Stage one video into work_dir/videos/ + symlinked ref, run batch_process,
    h264-remux output, return result record."""
    sub_id = f"{video.stem}_{int(time.time()*1000) % 1000000}"
    sub_work = work_dir / sub_id
    in_videos = sub_work / "videos"
    out_dir = sub_work / "output"
    in_videos.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage video as `input.mp4` (matches DGX layout — batch_process picks any
    # mp4 in --videos dir; we mirror the DGX naming for consistency).
    staged = in_videos / "input.mp4"
    if staged.exists() or staged.is_symlink():
        staged.unlink()
    staged.symlink_to(video.resolve())

    t_start = time.time()
    rc, out = await _run(
        [str(MIMIC_PYTHON), "batch_process.py",
         "--videos", str(in_videos),
         "--image", str(ref_image),
         "--output", str(out_dir),
         "--mode", "square", "--crop-anchor", "top"],
        cwd=MIMIC_ROOT / "mimicmotion",
        env=_mimic_env(),
    )
    wall = round(time.time() - t_start, 1)
    if rc != 0:
        return _failed(video.name, "batch_process", out) | {"wall_sec": wall, "sub_id": sub_id}

    raw = out_dir / "input_hiya.mp4"
    if not raw.exists() or raw.stat().st_size == 0:
        return _failed(video.name, "no_output", out) | {"wall_sec": wall, "sub_id": sub_id}

    return {
        "filename": video.name,
        "status": "raw_done",
        "sub_id": sub_id,
        "raw_path": str(raw),
        "wall_sec": wall,
    }


async def _remux_one(rec: dict, videos_out: Path, ffmpeg_sem: asyncio.Semaphore) -> dict:
    if rec["status"] != "raw_done":
        return rec
    final = videos_out / rec["filename"]
    async with ffmpeg_sem:
        rc, out = await _run([
            str(FFMPEG), "-y", "-loglevel", "error",
            "-i", rec["raw_path"],
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(final),
        ])
    if rc != 0 or not final.exists() or final.stat().st_size == 0:
        rec.update(_failed(rec["filename"], "remux", out))
        return rec
    rec["status"] = "completed"
    rec["output_path"] = str(final)
    rec["output_size"] = final.stat().st_size
    return rec


VideoDoneCb = Callable[[dict], Awaitable[None] | None]


async def run_phase3_on_local(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
    progress_cb: ProgressCb | None = None,
    on_video_done: VideoDoneCb | None = None,
) -> dict:
    """Run full Phase 3 locally for every *.mp4 in input_dir.

    Output layout and report keys mirror `run_phase3_on_dgx` so callers don't
    care which backend produced the report.

    `on_video_done(record)` is invoked after each video is fully processed
    (denoise + h264 remux) — callers can incrementally publish results
    without waiting for the whole batch.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    videos_out = output_dir / "videos"
    videos_out.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_local_scratch"
    work_dir.mkdir(parents=True, exist_ok=True)

    if not DEFAULT_REF_IMAGE.exists():
        raise FileNotFoundError(f"reference image not found: {DEFAULT_REF_IMAGE}")
    if not (MIMIC_ROOT / "mimicmotion" / "batch_process.py").exists():
        raise FileNotFoundError(f"MIMIC_ROOT/mimicmotion/batch_process.py missing under {MIMIC_ROOT}")

    all_mp4 = sorted(input_dir.glob("*.mp4"))
    videos = [v for v in all_mp4 if not v.name.startswith("sentence_")]
    skipped = len(all_mp4) - len(videos)
    if skipped:
        logger.info(f"[{task_id}] Phase 3 (local): skipped {skipped} sentence_* videos")
    if not videos:
        raise RuntimeError(f"Phase 3 (local): no word videos found in {input_dir}")

    logger.info(f"[{task_id}] Phase 3 (local): processing {len(videos)} videos with {MIMIC_ROOT}")
    last_emitted: list[float] = []
    t_start = time.time()
    await _emit_progress(progress_cb, 1.0, last_emitted)

    # Serial denoise; remux + on_video_done callback inline so each video
    # becomes available to consumers as soon as it's ready (no batched wait).
    ffmpeg_sem = asyncio.Semaphore(FFMPEG_CONCURRENCY)
    fetched: list[dict] = []
    for i, video in enumerate(videos):
        rec = await _process_one(video, DEFAULT_REF_IMAGE, work_dir, videos_out)
        rec = await _remux_one(rec, videos_out, ffmpeg_sem)
        fetched.append(rec)
        pct = 5.0 + 90.0 * (i + 1) / len(videos)
        await _emit_progress(progress_cb, pct, last_emitted)
        if on_video_done is not None:
            try:
                result = on_video_done(rec)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"[{task_id}] on_video_done callback raised: {e}")

    # Cleanup scratch (non-fatal)
    try:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass

    n_total = len(videos)
    n_completed = sum(1 for r in fetched if r.get("status") == "completed")
    wall = round(time.time() - t_start, 1)
    report = {
        "task_id": task_id,
        "backend": "local",
        "input_videos": n_total,
        "transfer_success": n_completed,
        "transfer_failed": n_total - n_completed,
        "transfer_skipped": skipped,
        "videos_generated": n_completed,
        "interpolation_mode": "local",
        "total_wall_seconds": wall,
        "results": fetched,
    }
    (output_dir / "phase3_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )

    logger.info(f"[{task_id}] Phase 3 (local) done: {n_completed}/{n_total} videos "
                f"in {wall/60:.1f} min")
    await _emit_progress(progress_cb, 100.0, last_emitted)

    if n_completed == 0:
        raise RuntimeError(f"Phase 3 (local): all {n_total} videos failed")

    return report
