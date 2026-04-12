"""API router for standalone Phase 3 testing on accuracy videos."""

import asyncio
import json
import logging
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from backend.config import settings
from backend.core.io_utils import read_jsonl
from backend.database import get_session
from backend.models.phase3_test import Phase3TestJob

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/phase3-test", tags=["phase3-test"])

ACCURACY_DATA = settings.CHATSIGN_ACCURACY_DATA
PENDING_VIDEOS_PATH = ACCURACY_DATA / "reports" / "pending-videos.jsonl"


def _resolve_accuracy_video(video_path_str: str) -> Path | None:
    """Resolve accuracy videoPath to an absolute filesystem path."""
    if video_path_str.startswith("/"):
        candidate = ACCURACY_DATA / video_path_str.lstrip("/")
    else:
        candidate = ACCURACY_DATA / video_path_str
    return candidate if candidate.is_file() else None


# ── List accuracy videos ──────────────────────────────────────────────

@router.get("/videos")
def list_accuracy_videos():
    """List all submitted videos from accuracy system."""
    entries = read_jsonl(PENDING_VIDEOS_PATH)
    videos = []
    for v in entries:
        if v.get("source") not in ("submission", "generated"):
            continue
        vp = v.get("videoPath", "")
        resolved = _resolve_accuracy_video(vp)
        videos.append({
            "video_id": v.get("videoId", ""),
            "sentence_id": v.get("sentenceId"),
            "sentence_text": v.get("sentenceText", ""),
            "translator_id": v.get("translatorId", ""),
            "filename": v.get("videoFileName", ""),
            "source": v.get("source", ""),
            "added_at": v.get("addedAt", ""),
            "exists": resolved is not None,
        })
    return {"videos": videos}


# ── Start a Phase 3 test job ──────────────────────────────────────────

class Phase3RunRequest(BaseModel):
    video_id: str


@router.post("/run")
def start_phase3_test(
    req: Phase3RunRequest,
    session: Session = Depends(get_session),
):
    entries = read_jsonl(PENDING_VIDEOS_PATH)
    video_entry = None
    for v in entries:
        if v.get("videoId") == req.video_id:
            video_entry = v
            break
    if not video_entry:
        raise HTTPException(404, f"Video {req.video_id} not found in accuracy")

    resolved = _resolve_accuracy_video(video_entry.get("videoPath", ""))
    if not resolved:
        raise HTTPException(404, "Video file not found on disk")

    job_id = uuid.uuid4().hex[:12]
    output_dir = settings.PHASE3_TEST_OUTPUT_DIR / job_id
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    dst = input_dir / resolved.name
    shutil.copy2(resolved, dst)

    job = Phase3TestJob(
        job_id=job_id,
        video_id=req.video_id,
        sentence_text=video_entry.get("sentenceText", ""),
        translator_id=video_entry.get("translatorId", ""),
        source_video_path=str(dst),
        source_filename=resolved.name,
        output_dir=str(output_dir),
        status="pending",
    )
    session.add(job)
    session.commit()

    threading.Thread(target=_run_phase3_test, args=(job_id,), daemon=True).start()

    return {"job_id": job_id, "status": "pending"}


# ── Background worker ─────────────────────────────────────────────────

def _run_phase3_test(job_id: str):
    from backend.api.tasks import gpu_manager
    from backend.database import engine
    from backend.workers.phase4_person_transfer import run_phase4_transfer
    from backend.workers.phase5_video_process import run_phase5_process
    from backend.workers.phase6_framer import run_phase6_framer

    def _save(session, job, status=None):
        if status:
            job.status = status
        job.updated_at = datetime.utcnow()
        session.add(job)
        session.commit()

    gpu_id = None
    with Session(engine) as session:
        job = session.exec(
            select(Phase3TestJob).where(Phase3TestJob.job_id == job_id)
        ).first()
        if not job:
            return

        try:
            output_dir = Path(job.output_dir)
            input_dir = output_dir / "input"
            transfer_dir = output_dir / "phase4_transfer"
            process_dir = output_dir / "phase5_process"
            framer_dir = output_dir / "phase6_framer"
            for d in [transfer_dir, process_dir, framer_dir]:
                d.mkdir(parents=True, exist_ok=True)

            # Acquire GPU (spin-wait with timeout)
            MAX_GPU_WAIT = 1800  # 30 minutes
            waited = 0
            while gpu_id is None:
                if waited >= MAX_GPU_WAIT:
                    raise TimeoutError("GPU not available within 30 minutes")
                gpu_id = gpu_manager.acquire(job_id)
                if gpu_id is None:
                    time.sleep(10)
                    waited += 10

            t_total = time.time()

            # Step 1: MimicMotion person transfer
            _save(session, job, "transfer")
            t0 = time.time()
            asyncio.run(run_phase4_transfer(job_id, input_dir, transfer_dir, gpu_id=gpu_id))
            job.transfer_time_sec = round(time.time() - t0, 1)
            _save(session, job)

            # Step 2: Video processing (resize, pose filter, dedup)
            _save(session, job, "processing")
            t0 = time.time()
            asyncio.run(run_phase5_process(job_id, transfer_dir, process_dir))
            job.process_time_sec = round(time.time() - t0, 1)
            _save(session, job)

            # Step 3: FramerTurbo interpolation
            _save(session, job, "framer")
            t0 = time.time()
            asyncio.run(run_phase6_framer(job_id, process_dir, framer_dir, gpu_id=gpu_id))
            job.framer_time_sec = round(time.time() - t0, 1)

            # Find generated video
            gen_videos = list((framer_dir / "videos").rglob("*.mp4")) if (framer_dir / "videos").exists() else []
            if gen_videos:
                job.generated_video_path = str(gen_videos[0])
            else:
                # Fallback: check process_dir for videos
                proc_videos = list(process_dir.rglob("*.mp4"))
                if proc_videos:
                    job.generated_video_path = str(proc_videos[0])

            job.duration_sec = round(time.time() - t_total, 1)
            _save(session, job, "completed")
            logger.info("Phase 3 test %s completed in %.1fs", job_id, job.duration_sec)

        except Exception as e:
            job.error_message = str(e)[:1000]
            _save(session, job, "failed")
            logger.exception("Phase 3 test %s failed", job_id)
        finally:
            if gpu_id is not None:
                gpu_manager.release(gpu_id)


# ── List / detail / serve / delete ────────────────────────────────────

@router.get("/jobs")
def list_jobs(
    status: str | None = None,
    session: Session = Depends(get_session),
):
    stmt = select(Phase3TestJob).order_by(Phase3TestJob.created_at.desc())
    if status:
        stmt = stmt.where(Phase3TestJob.status == status)
    jobs = session.exec(stmt).all()
    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status,
                "video_id": j.video_id,
                "sentence_text": j.sentence_text,
                "translator_id": j.translator_id,
                "source_filename": j.source_filename,
                "duration_sec": j.duration_sec,
                "transfer_time_sec": j.transfer_time_sec,
                "process_time_sec": j.process_time_sec,
                "framer_time_sec": j.framer_time_sec,
                "error_message": j.error_message,
                "created_at": j.created_at.isoformat(),
                "updated_at": j.updated_at.isoformat(),
            }
            for j in jobs
        ]
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str, session: Session = Depends(get_session)):
    job = session.exec(
        select(Phase3TestJob).where(Phase3TestJob.job_id == job_id)
    ).first()
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "video_id": job.video_id,
        "sentence_text": job.sentence_text,
        "translator_id": job.translator_id,
        "source_filename": job.source_filename,
        "generated_video_path": job.generated_video_path,
        "duration_sec": job.duration_sec,
        "transfer_time_sec": job.transfer_time_sec,
        "process_time_sec": job.process_time_sec,
        "framer_time_sec": job.framer_time_sec,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }


@router.get("/jobs/{job_id}/original-video")
def serve_original(job_id: str, session: Session = Depends(get_session)):
    job = session.exec(
        select(Phase3TestJob).where(Phase3TestJob.job_id == job_id)
    ).first()
    if not job:
        raise HTTPException(404, "Job not found")
    p = Path(job.source_video_path)
    if not p.is_file():
        raise HTTPException(404, "Original video not found")
    return FileResponse(p, media_type="video/mp4")


@router.get("/jobs/{job_id}/generated-video")
def serve_generated(job_id: str, session: Session = Depends(get_session)):
    job = session.exec(
        select(Phase3TestJob).where(Phase3TestJob.job_id == job_id)
    ).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.generated_video_path:
        raise HTTPException(400, "Video not ready")
    p = Path(job.generated_video_path)
    if not p.is_file():
        raise HTTPException(404, "Generated video not found")
    return FileResponse(p, media_type="video/mp4")


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str, session: Session = Depends(get_session)):
    job = session.exec(
        select(Phase3TestJob).where(Phase3TestJob.job_id == job_id)
    ).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status in ("pending", "transfer", "processing", "framer"):
        raise HTTPException(400, "Cannot delete a running job")
    if job.output_dir:
        p = Path(job.output_dir)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    session.delete(job)
    session.commit()
    return {"deleted": True}
