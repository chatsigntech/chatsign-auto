"""Test video generation API endpoints."""

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from backend.api.auth import get_current_user
from backend.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/test-video", tags=["test-video"])

MAX_JOBS = 50  # evict oldest jobs beyond this limit


@dataclass
class TestVideoJob:
    job_id: str
    task_id: str
    status: str = "pending"  # pending | generating | completed | failed
    progress: float = 0.0
    error: str | None = None
    video_path: str | None = None
    fps: float = 0.0
    sentences: list = field(default_factory=list)
    duration: float = 0.0


_jobs: dict[str, TestVideoJob] = {}
_jobs_lock = threading.Lock()


def _evict_old_jobs():
    """Remove oldest completed/failed jobs when store exceeds MAX_JOBS.

    Must be called with _jobs_lock held.
    """
    if len(_jobs) <= MAX_JOBS:
        return
    # Sort by insertion order (dict preserves order in Python 3.7+)
    removable = [
        jid for jid, j in _jobs.items()
        if j.status in ("completed", "failed")
    ]
    while len(_jobs) > MAX_JOBS and removable:
        _jobs.pop(removable.pop(0), None)


def _run_test_job(job: TestVideoJob):
    """Run test video generation in a background thread."""
    from backend.test_video.generator import generate_test_video

    try:
        job.status = "generating"
        job.progress = 0.0

        def _update_progress(pct):
            job.progress = pct

        result = generate_test_video(job.task_id, job.job_id, on_progress=_update_progress)

        job.video_path = result["video_path"]
        job.fps = result["fps"]
        job.sentences = result["sentences"]
        job.duration = result["duration"]
        job.status = "completed"
        job.progress = 1.0

        logger.info(f"Test video job {job.job_id} completed for task {job.task_id}")

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        logger.exception(f"Test video job {job.job_id} failed: {e}")


@router.post("/generate/{task_id}")
async def generate_test_video_endpoint(
    task_id: str,
    _user=Depends(get_current_user),
):
    """Start test video generation for a task."""
    # Validate Phase 2 output exists
    phase2_dir = settings.SHARED_DATA_ROOT / task_id / "phase_2" / "output"
    if not (phase2_dir / "manifest.json").exists():
        raise HTTPException(404, f"Phase 2 output not found for task {task_id}")

    job_id = uuid.uuid4().hex[:12]
    job = TestVideoJob(job_id=job_id, task_id=task_id)

    with _jobs_lock:
        _jobs[job_id] = job
        _evict_old_jobs()

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_test_job, job)

    return {"job_id": job_id, "status": job.status}


@router.get("/status/{job_id}")
async def get_test_video_status(
    job_id: str,
    _user=Depends(get_current_user),
):
    """Poll test video generation status."""
    job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(404, "Job not found")

    result = {
        "job_id": job.job_id,
        "task_id": job.task_id,
        "status": job.status,
        "progress": job.progress,
    }

    if job.status == "completed":
        result.update({
            "video_url": f"/api/test-video/video/{job.job_id}",
            "fps": job.fps,
            "sentences": job.sentences,
            "duration": job.duration,
        })
    elif job.status == "failed":
        result["error"] = job.error

    return result


@router.get("/video/{job_id}")
async def serve_test_video(job_id: str):
    """Serve generated test video file.

    No Bearer auth — video tags cannot send Authorization headers.
    Access is gated by the unguessable job_id (96-bit UUID).
    """
    job = _jobs.get(job_id)

    if job is None or job.video_path is None:
        raise HTTPException(404, "Video not found")

    path = Path(job.video_path)
    if not path.exists():
        raise HTTPException(404, "Video file missing")

    return FileResponse(path, media_type="video/mp4", filename=f"test_{job.task_id}.mp4")
