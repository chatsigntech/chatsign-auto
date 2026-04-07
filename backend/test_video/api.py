"""Test video generation API endpoints."""

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.api.auth import get_current_user
from backend.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/test-video", tags=["test-video"])

MAX_JOBS = 50


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    preset: Optional[str] = None
    pipeline: Optional[list[dict]] = None
    gpu_id: int = 0


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

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
    pipeline_desc: str = ""


_jobs: dict[str, TestVideoJob] = {}
_jobs_lock = threading.Lock()


def _evict_old_jobs():
    if len(_jobs) <= MAX_JOBS:
        return
    removable = [
        jid for jid, j in _jobs.items()
        if j.status in ("completed", "failed")
    ]
    while len(_jobs) > MAX_JOBS and removable:
        _jobs.pop(removable.pop(0), None)


def _run_test_job(job: TestVideoJob, pipeline, preset, gpu_id):
    from backend.test_video.generator import generate_test_video

    try:
        job.status = "generating"
        job.progress = 0.0

        def _update_progress(pct):
            job.progress = pct

        result = generate_test_video(
            job.task_id, job.job_id,
            pipeline=pipeline, preset=preset, gpu_id=gpu_id,
            on_progress=_update_progress,
        )

        job.video_path = result["video_path"]
        job.fps = result["fps"]
        job.sentences = result["sentences"]
        job.duration = result["duration"]
        job.pipeline_desc = str(result.get("pipeline", ""))
        job.status = "completed"
        job.progress = 1.0

        logger.info(f"Test video job {job.job_id} completed for task {job.task_id}")

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        logger.exception(f"Test video job {job.job_id} failed: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/presets")
async def list_presets(_user=Depends(get_current_user)):
    """List available augmentation pipeline presets."""
    from backend.test_video.generator import get_available_presets
    return get_available_presets()


@router.post("/generate/{task_id}")
async def generate_test_video_endpoint(
    task_id: str,
    body: GenerateRequest = Body(GenerateRequest()),
    _user=Depends(get_current_user),
):
    """Start test video generation for a task.

    Body (all optional):
      preset: preset name (e.g. "3d_yaw_right", "temporal_then_3d")
      pipeline: custom pipeline steps (overrides preset)
      gpu_id: GPU device for 3D augmentation (default 0)
    """
    phase2_dir = settings.SHARED_DATA_ROOT / task_id / "phase_2" / "output"
    if not (phase2_dir / "manifest.json").exists():
        raise HTTPException(404, f"Phase 2 output not found for task {task_id}")

    job_id = uuid.uuid4().hex[:12]
    job = TestVideoJob(job_id=job_id, task_id=task_id)

    with _jobs_lock:
        _jobs[job_id] = job
        _evict_old_jobs()

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        None, _run_test_job, job, body.pipeline, body.preset, body.gpu_id,
    )

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
    """Serve generated test video file."""
    job = _jobs.get(job_id)
    if job is None or job.video_path is None:
        raise HTTPException(404, "Video not found")

    path = Path(job.video_path)
    if not path.exists():
        raise HTTPException(404, "Video file missing")

    return FileResponse(path, media_type="video/mp4", filename=f"test_{job.task_id}.mp4")
