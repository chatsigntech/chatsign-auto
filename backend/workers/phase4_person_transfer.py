"""Phase 4: Person transfer using MimicMotion (on raw original videos).

Per the author's pipeline: transfer FIRST, then filter.
Uses raw videos from Phase 1, not preprocessed ones.

Features:
- Auto-parallel: detects GPU memory, runs optimal number of workers
- Per-video error handling with detailed logging
- Auto-retry with reduced params on failure
- Generates phase4_report.json with full audit trail
- Videos that fail all retries are excluded from pipeline
"""
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2

from backend.config import settings
from backend.core.gpu_auto_parallel import calculate_optimal_workers, get_gpu_info
from backend.core.subprocess_runner import run_subprocess

logger = logging.getLogger(__name__)

PYTHON = sys.executable
UNISIGN = settings.UNISIGN_PATH.resolve()
SCRIPT = UNISIGN / "scripts" / "inference" / "inference_raw_batch_cache.py"
CONFIG = UNISIGN / "configs" / "test.yaml"

# Retry strategy: first try normal quality, then reduce inference steps.
# Fewer steps = faster + less memory, but lower visual quality.
MIN_FRAMES = 20
DEFAULT_PAD = 5

RETRY_CONFIGS = [
    {"num_inference_steps": 25, "min_frames": MIN_FRAMES},
    {"num_inference_steps": 10, "min_frames": MIN_FRAMES},
]


def _get_video_info(path: Path) -> dict:
    """Get video metadata using OpenCV."""
    cap = cv2.VideoCapture(str(path))
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 0,
    }
    cap.release()
    info["duration"] = round(info["frames"] / max(1, info["fps"]), 2)
    return info


def _pad_video(video: Path, output_path: Path, min_frames: int = MIN_FRAMES, default_pad: int = DEFAULT_PAD) -> tuple[Path, int, int]:
    """Pad a short video by duplicating first/last frames to reach min_frames.

    Returns (padded_video_path, pad_front, pad_back).
    If video already has enough frames, returns (original_path, 0, 0).
    """
    cap = cv2.VideoCapture(str(video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if n_frames >= min_frames:
        cap.release()
        return video, 0, 0

    # Calculate padding
    pad_front = default_pad
    pad_back = default_pad
    total = n_frames + pad_front + pad_back
    if total < min_frames:
        extra = min_frames - total
        pad_front += (extra + 1) // 2
        pad_back += extra // 2

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return video, 0, 0

    # Write padded video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for _ in range(pad_front):
        writer.write(frames[0])
    for f in frames:
        writer.write(f)
    for _ in range(pad_back):
        writer.write(frames[-1])
    writer.release()

    logger.info(f"Padded {video.name}: {n_frames} → {n_frames + pad_front + pad_back} frames "
                f"(+{pad_front} front, +{pad_back} back)")
    return output_path, pad_front, pad_back


def _trim_video(video: Path, output_path: Path, trim_front: int, trim_back: int):
    """Trim padded frames from the output video after person transfer."""
    cap = cv2.VideoCapture(str(video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    keep_start = trim_front
    keep_end = n_frames - trim_back

    if keep_start >= keep_end:
        cap.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if keep_start <= i < keep_end:
            writer.write(frame)
    cap.release()
    writer.release()

    logger.info(f"Trimmed {video.name}: {n_frames} → {keep_end - keep_start} frames "
                f"(-{trim_front} front, -{trim_back} back)")


async def _run_single_transfer(video: Path, output_dir: Path, gpu_id: int,
                               num_inference_steps: int, task_id: str,
                               timeout: int = 1800) -> dict:
    """Run MimicMotion on a single video. Returns result dict."""
    tmp_in = Path(f"/tmp/phase4_{task_id}_{video.stem}")
    tmp_in.mkdir(exist_ok=True)
    link = tmp_in / video.name
    if not link.exists():
        link.symlink_to(video.resolve())

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(UNISIGN)

    t0 = time.time()
    try:
        returncode, stdout, stderr = await run_subprocess(
            [PYTHON, str(SCRIPT),
             "--batch_folder", str(tmp_in),
             "--output_dir", str(output_dir),
             "--inference_config", str(CONFIG),
             "--num_inference_steps", str(num_inference_steps),
             "--scheduler", "EulerDiscreteScheduler"],
            cwd=str(UNISIGN),
            env=env,
            timeout=timeout,
        )
        elapsed = time.time() - t0

        if returncode == 0:
            return {"status": "success", "time": round(elapsed, 1), "steps": num_inference_steps}
        else:
            error_lines = [l for l in stderr.strip().split("\n") if l.strip()]
            error_summary = error_lines[-1] if error_lines else "Unknown error"
            return {
                "status": "failed", "time": round(elapsed, 1), "steps": num_inference_steps,
                "error": error_summary,
                "stderr_tail": "\n".join(error_lines[-5:]),
            }
    except Exception as e:
        return {"status": "error", "time": round(time.time() - t0, 1),
                "steps": num_inference_steps, "error": str(e)}
    finally:
        link.unlink(missing_ok=True)
        try:
            tmp_in.rmdir()
        except OSError:
            pass


async def _process_one_video(video: Path, output_dir: Path, gpu_id: int,
                             task_id: str, index: int, total: int) -> dict:
    """Process a single video with retry logic. Returns report entry."""
    entry = {"filename": video.name, "attempts": []}

    # Skip if already generated
    existing = list(output_dir.rglob(f"{video.stem}_*.mp4")) or \
               list(output_dir.rglob(f"{video.stem}.mp4"))
    if existing:
        entry["status"] = "success"
        entry["note"] = "already exists"
        return entry

    video_info = _get_video_info(video)
    entry["video_info"] = video_info

    # Pad short videos (duplicate first/last frames to reach MIN_FRAMES)
    pad_front, pad_back = 0, 0
    actual_video = video
    if video_info["frames"] < MIN_FRAMES and video_info["frames"] > 0:
        padded_path = Path(f"/tmp/phase4_{task_id}_{video.stem}_padded.mp4")
        actual_video, pad_front, pad_back = _pad_video(video, padded_path)
        if pad_front > 0:
            entry["padding"] = {"front": pad_front, "back": pad_back,
                                "original_frames": video_info["frames"]}

    # Try with each config
    for attempt_idx, cfg in enumerate(RETRY_CONFIGS):
        logger.info(f"[{task_id}] Phase 4: [{index}/{total}] {video.name} "
                    f"(attempt {attempt_idx+1}, steps={cfg['num_inference_steps']})")

        result = await _run_single_transfer(
            actual_video, output_dir, gpu_id, cfg["num_inference_steps"], task_id
        )
        entry["attempts"].append(result)

        if result["status"] == "success":
            # Trim padded frames from output if we added padding
            if pad_front > 0 or pad_back > 0:
                generated = list(output_dir.rglob(f"{video.stem}*.mp4")) + \
                            list(output_dir.rglob(f"{actual_video.stem}*.mp4"))
                for out_mp4 in generated:
                    trimmed = out_mp4.with_suffix(".trimmed.mp4")
                    _trim_video(out_mp4, trimmed, pad_front, pad_back)
                    if trimmed.exists() and trimmed.stat().st_size > 0:
                        trimmed.replace(out_mp4)

            # Clean up padded temp file
            if actual_video != video and actual_video.exists():
                actual_video.unlink(missing_ok=True)

            entry["status"] = "success" if attempt_idx == 0 else "retry_success"
            return entry
        else:
            logger.warning(f"[{task_id}] Phase 4: {video.name} attempt {attempt_idx+1} "
                           f"failed: {result.get('error', 'unknown')[:100]}")

    entry["status"] = "failed"
    logger.error(f"[{task_id}] Phase 4: {video.name} FAILED all attempts")

    # Clean up padded temp file
    if actual_video != video and actual_video.exists():
        actual_video.unlink(missing_ok=True)

    return entry


def _write_live_progress(output_dir: Path, counters: dict, total: int):
    """Write live progress summary for the frontend to poll."""
    done = sum(counters.values())
    summary = {
        "total": total,
        "done": done,
        "success": counters.get("success", 0),
        "retry_success": counters.get("retry_success", 0),
        "failed": counters.get("failed", 0),
        "skipped_short": counters.get("skipped_short", 0),
        "in_progress": total - done,
    }
    try:
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass  # non-critical


async def _worker(worker_id: int, queue: asyncio.Queue, output_dir: Path,
                  gpu_id: int, task_id: str, total: int,
                  results: dict, counters: dict):
    """Worker coroutine that pulls videos from queue and processes them."""
    while True:
        try:
            index, video = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        entry = await _process_one_video(video, output_dir, gpu_id,
                                         task_id, index, total)
        results[video.stem] = entry

        status = entry.get("status", "unknown")
        counters[status] = counters.get(status, 0) + 1

        logger.info(f"[{task_id}] Worker {worker_id}: {video.name} → {status} "
                    f"(done: {sum(counters.values())}/{total})")

        _write_live_progress(output_dir, counters, total)

        queue.task_done()


async def run_phase4_transfer(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
    gpu_id: int = 0,
    max_workers: int = 0,
) -> bool:
    """
    Run MimicMotion person transfer with auto-parallel and reporting.

    Args:
        max_workers: 0 = auto-detect from GPU memory, >0 = force N workers
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not SCRIPT.exists():
        raise FileNotFoundError(f"MimicMotion script not found: {SCRIPT}")

    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No videos found in {input_dir}")

    # Auto-detect parallelism
    if max_workers <= 0:
        rec = calculate_optimal_workers(gpu_id)
        num_workers = rec["workers"]
        logger.info(f"[{task_id}] Phase 4: Auto-parallel → {num_workers} workers. "
                    f"{rec['reasoning']}")
    else:
        num_workers = max_workers
        logger.info(f"[{task_id}] Phase 4: Forced {num_workers} workers")

    logger.info(f"[{task_id}] Phase 4: Person transfer on {len(videos)} videos, "
                f"GPU {gpu_id}, {num_workers} parallel worker(s)")

    # Build queue
    queue = asyncio.Queue()
    for i, video in enumerate(videos):
        queue.put_nowait((i + 1, video))

    results = {}
    counters = {}
    start_all = time.time()

    # Launch workers
    workers = [
        asyncio.create_task(
            _worker(w, queue, output_dir, gpu_id, task_id, len(videos), results, counters)
        )
        for w in range(num_workers)
    ]
    await asyncio.gather(*workers)

    total_time = time.time() - start_all

    # Build report
    success = counters.get("success", 0)
    retry_success = counters.get("retry_success", 0)
    skipped_short = counters.get("skipped_short", 0)
    failed = counters.get("failed", 0)

    report = {
        "task_id": task_id,
        "input_dir": str(input_dir),
        "total_input": len(videos),
        "num_workers": num_workers,
        "gpu_info": get_gpu_info(gpu_id),
        "results": results,
        "summary": {
            "success": success,
            "retry_success": retry_success,
            "skipped_short": skipped_short,
            "failed": failed,
            "total_generated": success + retry_success,
            "total_excluded": skipped_short + failed,
            "total_time_seconds": round(total_time, 1),
            "avg_time_per_video": round(total_time / max(1, success + retry_success), 1),
        },
    }

    report_path = output_dir / "phase4_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    generated = list(output_dir.rglob("*.mp4"))
    logger.info(f"[{task_id}] Phase 4 completed: {success}+{retry_success} success, "
                f"{skipped_short} skipped, {failed} failed → "
                f"{len(generated)} videos in {total_time/60:.1f}min "
                f"({num_workers} workers)")

    if not generated:
        raise RuntimeError(f"Phase 4: No videos generated (all {len(videos)} failed/skipped)")

    # Fix moov atom for browser streaming (faststart)
    ffmpeg = str(Path(__file__).resolve().parent.parent.parent / "bin" / "ffmpeg")
    if Path(ffmpeg).exists():
        for mp4 in generated:
            tmp = mp4.with_suffix(".tmp.mp4")
            rc, _, _ = await run_subprocess([ffmpeg, "-y", "-i", str(mp4), "-c", "copy", "-movflags", "+faststart", str(tmp)])
            if rc == 0 and tmp.exists():
                tmp.replace(mp4)
            elif tmp.exists():
                tmp.unlink()
        logger.info(f"[{task_id}] Fixed faststart for {len(generated)} videos")

    return True
