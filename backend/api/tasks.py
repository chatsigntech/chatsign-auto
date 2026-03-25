import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from backend.database import get_session
from backend.models.task import PipelineTask
from backend.models.phase import PhaseState
from backend.models.user import User
from backend.api.auth import get_current_user

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

NUM_PHASES = 6


class TaskCreate(BaseModel):
    name: str
    augmentation_preset: str = "medium"


class TaskResponse(BaseModel):
    task_id: str
    name: str
    status: str
    current_phase: int
    augmentation_preset: str
    created_at: datetime


@router.post("/", response_model=TaskResponse)
def create_task(
    body: TaskCreate,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    task_id = str(uuid.uuid4())[:8]
    task = PipelineTask(
        task_id=task_id,
        name=body.name,
        augmentation_preset=body.augmentation_preset,
    )
    session.add(task)

    for phase_num in range(1, NUM_PHASES + 1):
        phase = PhaseState(task_id=task_id, phase_num=phase_num)
        session.add(phase)

    session.commit()
    session.refresh(task)
    return TaskResponse(
        task_id=task.task_id,
        name=task.name,
        status=task.status,
        current_phase=task.current_phase,
        augmentation_preset=task.augmentation_preset,
        created_at=task.created_at,
    )


@router.get("/")
def list_tasks(
    status: Optional[str] = None,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    query = select(PipelineTask)
    if status:
        query = query.where(PipelineTask.status == status)
    tasks = session.exec(query.order_by(PipelineTask.created_at.desc())).all()
    return {"tasks": tasks}


@router.get("/{task_id}")
def get_task(
    task_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    task = session.exec(select(PipelineTask).where(PipelineTask.task_id == task_id)).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    phases = session.exec(
        select(PhaseState).where(PhaseState.task_id == task_id).order_by(PhaseState.phase_num)
    ).all()
    return {"task": task, "phases": phases}
