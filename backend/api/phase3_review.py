"""Phase 3 review-stats + publish-to-remote API.

Side-branch feature, fully isolated from the pipeline:
- Read-only against accuracy data files (jsonl)
- POST /publish forks `sshpass scp` to push approved videos + gloss.csv
- Failures here cannot affect Phase 3 worker, other phases, or accuracy.
"""
import logging
import re
import threading

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from backend.api.auth import get_current_user
from backend.config import settings
from backend.core.io_utils import read_jsonl
from backend.database import get_session
from backend.models.task import PipelineTask
from backend.models.user import User
from backend.workers.phase3_publish import _safe
from backend.workers.phase3_remote_publish import publish_to_remote

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tasks", tags=["phase3-review"])


# ── input validation patterns (publish) ──────────────────────────────
_USER_OK = re.compile(r"^[A-Za-z0-9_.-]+$")
_HOST_OK = re.compile(r"^[A-Za-z0-9.-]+$")
_PATH_OK = re.compile(r"^/[A-Za-z0-9/_.-]+$")  # absolute path only


# ── per-task publish lock (in-memory; OK because publish is interactive) ──
_in_flight: set[str] = set()
_in_flight_lock = threading.Lock()


def _acquire_lock(task_id: str) -> bool:
    with _in_flight_lock:
        if task_id in _in_flight:
            return False
        _in_flight.add(task_id)
        return True


def _release_lock(task_id: str) -> None:
    with _in_flight_lock:
        _in_flight.discard(task_id)


# ── helpers ──────────────────────────────────────────────────────────
def _get_task(session: Session, task_id: str) -> PipelineTask:
    task = session.exec(select(PipelineTask).where(PipelineTask.task_id == task_id)).first()
    if not task:
        raise HTTPException(404, "task not found")
    return task


def _compute_review_stats(task_name: str) -> dict:
    """Join accuracy pending-videos.jsonl + review-decisions.jsonl for one task.

    Filters to source=generated videos whose videoId starts with our prefix.
    Returns {approved, rejected, pending, *_videos: [...]}.
    """
    safe_task = _safe(task_name)
    prefix = f"gen_{safe_task}_hiya_"
    accuracy_data = settings.CHATSIGN_ACCURACY_DATA
    reports = accuracy_data / "reports"

    pending = read_jsonl(reports / "pending-videos.jsonl")
    our_videos = {}
    for v in pending:
        vid = v.get("videoId", "")
        if v.get("source") == "generated" and vid.startswith(prefix):
            our_videos[vid] = v

    decisions = read_jsonl(reports / "review-decisions.jsonl")
    decided = {}
    for d in decisions:
        vid = d.get("videoId")
        if vid in our_videos:
            decided[vid] = d  # last-write-wins if duplicates

    approved, rejected, pending_list = [], [], []
    for vid, v in our_videos.items():
        word = vid[len(prefix):]
        entry = {
            "videoId": vid,
            "word": word,
            "sentenceText": v.get("sentenceText", ""),
            "videoPath": v.get("videoPath", ""),
            "filename": v.get("videoFileName", ""),
        }
        d = decided.get(vid)
        if d and d.get("decision") == "approved":
            entry["decided_at"] = d.get("timestamp", "")
            approved.append(entry)
        elif d and d.get("decision") == "rejected":
            entry["decided_at"] = d.get("timestamp", "")
            entry["comments"] = d.get("comments", "")
            rejected.append(entry)
        else:
            pending_list.append(entry)

    return {
        "approved": len(approved),
        "rejected": len(rejected),
        "pending": len(pending_list),
        "approved_videos": approved,
        "rejected_videos": rejected,
        "pending_videos": pending_list,
        "batch_prefix": prefix,
    }


# ── endpoint: review-stats ───────────────────────────────────────────
@router.get("/{task_id}/phases/3/review-stats")
def get_review_stats(
    task_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    task = _get_task(session, task_id)
    try:
        return _compute_review_stats(task.name)
    except FileNotFoundError as e:
        # accuracy files missing — return empty stats rather than 500
        logger.warning(f"[{task_id}] review-stats: accuracy file missing: {e}")
        return {"approved": 0, "rejected": 0, "pending": 0,
                "approved_videos": [], "rejected_videos": [], "pending_videos": [],
                "batch_prefix": "", "note": "accuracy data unavailable"}


# ── endpoint: publish ────────────────────────────────────────────────
class PublishRequest(BaseModel):
    host: str
    port: int = 22
    username: str
    password: str       # never logged, never stored
    target_dir: str     # absolute path on remote


@router.post("/{task_id}/phases/3/publish")
def publish_phase3(
    task_id: str,
    body: PublishRequest,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    task = _get_task(session, task_id)

    # Strict input validation (defense against shell injection in scp remote spec)
    if not _HOST_OK.match(body.host):
        raise HTTPException(400, "invalid host (allowed: A-Za-z0-9.-)")
    if not (1 <= body.port <= 65535):
        raise HTTPException(400, "invalid port")
    if not _USER_OK.match(body.username):
        raise HTTPException(400, "invalid username (allowed: A-Za-z0-9._-)")
    if not _PATH_OK.match(body.target_dir):
        raise HTTPException(400, "target_dir must be absolute path matching [A-Za-z0-9/_.-]")
    if not body.password:
        raise HTTPException(400, "password required")

    # Compute approved list fresh (always live from accuracy)
    try:
        stats = _compute_review_stats(task.name)
    except FileNotFoundError:
        raise HTTPException(503, "accuracy data unavailable")

    if not stats["approved_videos"]:
        raise HTTPException(400, "no approved videos to publish")

    if not _acquire_lock(task_id):
        raise HTTPException(409, "publish already running for this task")
    try:
        result = publish_to_remote(
            stats["approved_videos"],
            settings.CHATSIGN_ACCURACY_DATA,
            body.host, body.port, body.username, body.password,
            body.target_dir,
        )
        # Audit log (no password, no host details beyond name)
        logger.info(f"[{task_id}] publish to {body.host}:{body.port}{body.target_dir}: "
                    f"success={result['success']}/{result['total_videos']} "
                    f"failed={result['failed']} gloss={result.get('gloss_uploaded')}")
        return result
    finally:
        _release_lock(task_id)
