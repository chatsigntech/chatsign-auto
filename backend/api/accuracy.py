"""API to preview chatsign-accuracy collection & review status."""
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from backend.config import settings
from backend.models.user import User
from backend.api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/accuracy", tags=["accuracy"])

DATA_DIR = settings.CHATSIGN_ACCURACY_DATA
REPORTS_DIR = DATA_DIR / "reports"
TEXTS_DIR = DATA_DIR / "texts"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


@router.get("/status")
def get_accuracy_status(
    batch: str | None = Query(None, description="Filter by batch name"),
    user: User = Depends(get_current_user),
):
    """
    Preview the current collection & review status from accuracy data.
    Reads directly from accuracy's local filesystem.
    """
    # Load all submissions
    pending = _read_jsonl(REPORTS_DIR / "pending-videos.jsonl")
    submissions = [v for v in pending if v.get("source") == "submission"]

    # Filter by batch if specified
    if batch:
        prefix = batch + "_"
        submissions = [v for v in submissions if v.get("videoFileName", "").startswith(prefix)]

    # Load review decisions
    decisions = _read_jsonl(REPORTS_DIR / "review-decisions.jsonl")
    decision_map = {}
    for d in decisions:
        vid = d.get("videoId")
        if vid:
            decision_map[vid] = d.get("decision")

    # Classify submissions
    approved = []
    rejected = []
    pending_review = []
    for v in submissions:
        vid = v.get("videoId")
        status = decision_map.get(vid)
        entry = {
            "video_id": vid,
            "sentence_id": v.get("sentenceId"),
            "sentence_text": v.get("sentenceText", ""),
            "translator": v.get("translatorId", ""),
            "filename": v.get("videoFileName", ""),
            "added_at": v.get("addedAt", ""),
        }
        if status == "approved":
            approved.append(entry)
        elif status == "rejected":
            rejected.append(entry)
        else:
            entry["review_status"] = "pending"
            pending_review.append(entry)

    return {
        "summary": {
            "total_submissions": len(submissions),
            "approved": len(approved),
            "rejected": len(rejected),
            "pending_review": len(pending_review),
            "ready_for_pipeline": len(approved) > 0,
        },
        "approved_videos": approved,
        "pending_review": pending_review,
        "rejected_videos": rejected,
    }


@router.get("/batches")
def get_batches(user: User = Depends(get_current_user)):
    """List available sentence batches from accuracy."""
    batches = []
    if TEXTS_DIR.exists():
        for f in sorted(TEXTS_DIR.glob("*.jsonl")):
            sentences = _read_jsonl(f)
            batches.append({
                "name": f.stem,
                "sentence_count": len(sentences),
            })
    return {"batches": batches}


@router.get("/sentences")
def get_sentences(
    batch: str = Query(..., description="Batch name"),
    user: User = Depends(get_current_user),
):
    """List sentences in a batch with their recording status."""
    batch_file = TEXTS_DIR / f"{batch}.jsonl"
    if not batch_file.exists():
        return {"sentences": [], "error": f"Batch '{batch}' not found"}

    sentences = _read_jsonl(batch_file)

    # Check which sentences have submissions
    pending = _read_jsonl(REPORTS_DIR / "pending-videos.jsonl")
    decisions = _read_jsonl(REPORTS_DIR / "review-decisions.jsonl")
    decision_map = {d.get("videoId"): d.get("decision") for d in decisions}

    sentence_status = {}
    for v in pending:
        if v.get("source") != "submission":
            continue
        sid = v.get("sentenceId")
        vid = v.get("videoId")
        if sid not in sentence_status:
            sentence_status[sid] = {"recorded": 0, "approved": 0, "rejected": 0, "pending": 0}
        sentence_status[sid]["recorded"] += 1
        status = decision_map.get(vid)
        if status == "approved":
            sentence_status[sid]["approved"] += 1
        elif status == "rejected":
            sentence_status[sid]["rejected"] += 1
        else:
            sentence_status[sid]["pending"] += 1

    result = []
    for s in sentences:
        sid = s.get("id")
        stats = sentence_status.get(sid, {"recorded": 0, "approved": 0, "rejected": 0, "pending": 0})
        result.append({
            "id": sid,
            "text": s.get("text", ""),
            **stats,
        })

    return {"sentences": result, "batch": batch}
