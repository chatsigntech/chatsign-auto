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
RETRY_CONFIGS = [
    {"num_inference_steps": 10, "min_frames": 20},
    {"num_inference_steps": 6, "min_frames": 16},
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
             "--num_inference_steps", str(num_inference_steps)],
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

    # Check minimum frame count
    if video_info["frames"] < RETRY_CONFIGS[-1]["min_frames"]:
        entry["status"] = "skipped_short"
        entry["note"] = f"Only {video_info['frames']} frames, need >= {RETRY_CONFIGS[-1]['min_frames']}"
        logger.info(f"[{task_id}] Phase 4: [{index}/{total}] {video.name} SKIPPED ({video_info['frames']} frames)")
        return entry

    # Try with each config
    for attempt_idx, cfg in enumerate(RETRY_CONFIGS):
        if video_info["frames"] < cfg["min_frames"]:
            continue

        logger.info(f"[{task_id}] Phase 4: [{index}/{total}] {video.name} "
                    f"(attempt {attempt_idx+1}, steps={cfg['num_inference_steps']})")

        result = await _run_single_transfer(
            video, output_dir, gpu_id, cfg["num_inference_steps"], task_id
        )
        entry["attempts"].append(result)

        if result["status"] == "success":
            entry["status"] = "success" if attempt_idx == 0 else "retry_success"
            return entry
        else:
            logger.warning(f"[{task_id}] Phase 4: {video.name} attempt {attempt_idx+1} "
                           f"failed: {result.get('error', 'unknown')[:100]}")

    entry["status"] = "failed"
    logger.error(f"[{task_id}] Phase 4: {video.name} FAILED all attempts")
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

    return True
