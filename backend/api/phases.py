from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from backend.database import get_session
from backend.models.phase import PhaseState
from backend.models.user import User
from backend.api.auth import get_current_user

router = APIRouter(prefix="/api/phases", tags=["phases"])


@router.get("/{task_id}")
def get_phases(
    task_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    phases = session.exec(
        select(PhaseState).where(PhaseState.task_id == task_id).order_by(PhaseState.phase_num)
    ).all()
    if not phases:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"phases": phases}
