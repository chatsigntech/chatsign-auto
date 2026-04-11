"""API router for sign language video generation."""

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel
from sqlmodel import Session, select

from backend.database import get_session
from backend.models.sign_video import SignVideoGeneration
from backend.core.sign_video_generator import run_generation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sign-video", tags=["sign-video"])


class SignVideoCreateRequest(BaseModel):
    title: str
    text: str


class SignVideoItem(BaseModel):
    job_id: str
    title: str
    input_text: str
    status: str
    gloss_count: int
    matched_count: int
    duration_sec: float | None
    error_message: str | None
    created_at: str
    updated_at: str


@router.post("/generate")
def create_sign_video(
    req: SignVideoCreateRequest,
    session: Session = Depends(get_session),
):
    title = (req.title or "").strip()
    text = (req.text or "").strip()
    if not title:
        raise HTTPException(400, "Title is required")
    if not text:
        raise HTTPException(400, "Text is required")

    job_id = uuid.uuid4().hex[:12]
    job = SignVideoGeneration(
        job_id=job_id,
        title=title,
        input_text=text,
        status="pending",
    )
    session.add(job)
    session.commit()

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, run_generation, job_id, title, text)

    return {"job_id": job_id, "status": "pending"}


@router.get("/")
def list_sign_videos(
    status: str | None = None,
    session: Session = Depends(get_session),
):
    stmt = select(SignVideoGeneration).order_by(SignVideoGeneration.created_at.desc())
    if status:
        stmt = stmt.where(SignVideoGeneration.status == status)
    jobs = session.exec(stmt).all()
    return {
        "jobs": [
            SignVideoItem(
                job_id=j.job_id,
                title=j.title,
                input_text=j.input_text,
                status=j.status,
                gloss_count=j.gloss_count,
                matched_count=j.matched_count,
                duration_sec=j.duration_sec,
                error_message=j.error_message,
                created_at=j.created_at.isoformat(),
                updated_at=j.updated_at.isoformat(),
            )
            for j in jobs
        ]
    }


@router.get("/{job_id}")
def get_sign_video(job_id: str, session: Session = Depends(get_session)):
    job = session.exec(
        select(SignVideoGeneration).where(SignVideoGeneration.job_id == job_id)
    ).first()
    if not job:
        raise HTTPException(404, "Job not found")

    return {
        "job_id": job.job_id,
        "title": job.title,
        "input_text": job.input_text,
        "status": job.status,
        "gloss_count": job.gloss_count,
        "matched_count": job.matched_count,
        "duration_sec": job.duration_sec,
        "glosses": json.loads(job.glosses_json) if job.glosses_json else [],
        "match_result": json.loads(job.match_result_json) if job.match_result_json else [],
        "unmatched": json.loads(job.unmatched_glosses) if job.unmatched_glosses else [],
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }


@router.get("/{job_id}/video")
def get_sign_video_file(job_id: str, session: Session = Depends(get_session)):
    job = session.exec(
        select(SignVideoGeneration).where(SignVideoGeneration.job_id == job_id)
    ).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "completed" or not job.video_path:
        raise HTTPException(400, "Video not ready")

    video_path = Path(job.video_path)
    if not video_path.is_file():
        raise HTTPException(404, "Video file not found on disk")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=job.video_filename or f"{job_id}.mp4",
    )


@router.delete("/{job_id}")
def delete_sign_video(job_id: str, session: Session = Depends(get_session)):
    job = session.exec(
        select(SignVideoGeneration).where(SignVideoGeneration.job_id == job_id)
    ).first()
    if not job:
        raise HTTPException(404, "Job not found")

    if job.video_path:
        p = Path(job.video_path)
        if p.is_file():
            p.unlink()

    session.delete(job)
    session.commit()
    return {"deleted": True}
