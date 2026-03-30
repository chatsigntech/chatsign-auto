"""Phase 4: Person transfer using MimicMotion (on raw original videos).

Per the author's pipeline: transfer FIRST, then filter.
Uses raw videos from Phase 1, not preprocessed ones.

Features:
- Per-video error handling with detailed logging
- Auto-retry with reduced params on failure
- Generates phase4_report.json with full audit trail
- Videos that fail all retries are excluded from pipeline
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2

from backend.config import settings
from backend.core.subprocess_runner import run_subprocess

logger = logging.getLogger(__name__)

PYTHON = sys.executable
UNISIGN = settings.UNISIGN_PATH.resolve()
SCRIPT = UNISIGN / "scripts" / "inference" / "inference_raw_batch_cache.py"
CONFIG = UNISIGN / "configs" / "test.yaml"

# Retry strategy: first try normal quality, then reduce inference steps
# for videos where the full pipeline triggers OOM or index errors.
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
                               timeout: int = 600) -> dict:
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


async def run_phase4_transfer(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
    gpu_id: int = 0,
) -> bool:
    """
    Run MimicMotion person transfer with retry and reporting.

    Status definitions:
    - success: transferred on first attempt → continues to Phase 5+
    - retry_success: failed first, succeeded with reduced params → continues
    - skipped_short: too few frames for MimicMotion → excluded
    - failed: all attempts failed → excluded
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not SCRIPT.exists():
        raise FileNotFoundError(f"MimicMotion script not found: {SCRIPT}")

    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No videos found in {input_dir}")

    logger.info(f"[{task_id}] Phase 4: Person transfer on {len(videos)} videos, GPU {gpu_id}")

    report = {
        "task_id": task_id,
        "input_dir": str(input_dir),
        "total_input": len(videos),
        "results": {},
        "summary": {},
    }

    success = 0
    retry_success = 0
    skipped_short = 0
    failed = 0
    start_all = time.time()

    for i, video in enumerate(videos):
        entry = {"filename": video.name, "attempts": []}

        # Skip if already generated (fast check before opening video)
        existing = list(output_dir.rglob(f"{video.stem}_*.mp4")) or list(output_dir.rglob(f"{video.stem}.mp4"))
        if existing:
            entry["status"] = "success"
            entry["note"] = "already exists"
            report["results"][video.stem] = entry
            success += 1
            continue

        # Get video info only for videos that need processing
        video_info = _get_video_info(video)
        entry["video_info"] = video_info

        # Check minimum frame count
        if video_info["frames"] < RETRY_CONFIGS[-1]["min_frames"]:
            entry["status"] = "skipped_short"
            entry["note"] = f"Only {video_info['frames']} frames, need >= {RETRY_CONFIGS[-1]['min_frames']}"
            report["results"][video.stem] = entry
            skipped_short += 1
            logger.info(f"[{task_id}] Phase 4: [{i+1}/{len(videos)}] {video.name} "
                        f"SKIPPED ({video_info['frames']} frames)")
            continue

        # Try with each config
        transferred = False
        for attempt_idx, cfg in enumerate(RETRY_CONFIGS):
            if video_info["frames"] < cfg["min_frames"]:
                continue

            logger.info(f"[{task_id}] Phase 4: [{i+1}/{len(videos)}] {video.name} "
                        f"(attempt {attempt_idx+1}, steps={cfg['num_inference_steps']})")

            result = await _run_single_transfer(
                video, output_dir, gpu_id, cfg["num_inference_steps"], task_id
            )
            entry["attempts"].append(result)

            if result["status"] == "success":
                entry["status"] = "success" if attempt_idx == 0 else "retry_success"
                if attempt_idx == 0:
                    success += 1
                else:
                    retry_success += 1
                transferred = True
                break
            else:
                logger.warning(f"[{task_id}] Phase 4: {video.name} attempt {attempt_idx+1} "
                               f"failed: {result.get('error', 'unknown')[:100]}")

        if not transferred:
            entry["status"] = "failed"
            failed += 1
            logger.error(f"[{task_id}] Phase 4: {video.name} FAILED all attempts")

        report["results"][video.stem] = entry

    total_time = time.time() - start_all
    report["summary"] = {
        "success": success,
        "retry_success": retry_success,
        "skipped_short": skipped_short,
        "failed": failed,
        "total_generated": success + retry_success,
        "total_excluded": skipped_short + failed,
        "total_time_seconds": round(total_time, 1),
    }

    report_path = output_dir / "phase4_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    generated = list(output_dir.rglob("*.mp4"))
    logger.info(f"[{task_id}] Phase 4 completed: {success} success, {retry_success} retry, "
                f"{skipped_short} skipped, {failed} failed → {len(generated)} videos in {total_time/60:.1f}min")

    if not generated:
        raise RuntimeError(f"Phase 4: No videos generated (all {len(videos)} failed/skipped)")

    return True
