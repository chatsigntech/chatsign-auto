import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from backend.database import engine, get_session
from backend.models.task import PipelineTask
from backend.models.phase import PhaseState
from backend.models.user import User
from backend.api.auth import get_current_user
from backend.core.phase_state_manager import PhaseStateManager
from backend.core.gpu_manager import GPUManager
from backend.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

NUM_PHASES = 6

gpu_manager = GPUManager(
    max_gpus=settings.MAX_GPUS,
    device_ids=[int(d) for d in settings.CUDA_VISIBLE_DEVICES.split(",") if d.strip()],
)

# Track running tasks for pause support
_running_tasks: dict[str, bool] = {}  # task_id -> cancelled flag


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


def _run_pipeline_sync(task_id: str):
    """Sync wrapper to run the async pipeline in a background thread."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run_pipeline(task_id))
    finally:
        loop.close()


async def _run_pipeline(task_id: str):
    """Execute pipeline phases sequentially in the background."""
    from backend.workers.phase2_worker import run_phase2
    from backend.workers.phase4_worker import run_phase4
    from backend.workers.phase5_worker import run_phase5
    from backend.workers.phase6_worker import run_phase6

    _running_tasks[task_id] = False
    data_root = settings.SHARED_DATA_ROOT / task_id

    try:
        with Session(engine) as session:
            task = session.exec(
                select(PipelineTask).where(PipelineTask.task_id == task_id)
            ).first()
            if not task:
                return
            start_phase = task.current_phase or 1
            task.status = "running"
            task.updated_at = datetime.utcnow()
            session.add(task)
            session.commit()

        for phase_num in range(start_phase, NUM_PHASES + 1):
            # Check for pause/cancel
            if _running_tasks.get(task_id):
                with Session(engine) as session:
                    task = session.exec(
                        select(PipelineTask).where(PipelineTask.task_id == task_id)
                    ).first()
                    if task:
                        task.status = "paused"
                        task.current_phase = phase_num
                        task.updated_at = datetime.utcnow()
                        session.add(task)
                        session.commit()
                logger.info(f"[{task_id}] Pipeline paused at phase {phase_num}")
                return

            with Session(engine) as session:
                task = session.exec(
                    select(PipelineTask).where(PipelineTask.task_id == task_id)
                ).first()
                if task:
                    task.current_phase = phase_num
                    task.updated_at = datetime.utcnow()
                    session.add(task)
                    session.commit()

            with Session(engine) as session:
                PhaseStateManager.mark_running(task_id, phase_num, session)

            phase_input = data_root / f"phase_{phase_num}" / "input"
            phase_output = data_root / f"phase_{phase_num}" / "output"
            phase_input.mkdir(parents=True, exist_ok=True)
            phase_output.mkdir(parents=True, exist_ok=True)

            try:
                gpu_id = None
                if phase_num in (5, 6):
                    gpu_id = gpu_manager.acquire(task_id)
                    if gpu_id is None:
                        gpu_id = 0  # fallback

                if phase_num == 1:
                    # TODO: Phase 1 - integrate with chatsign-accuracy to pull collected videos
                    logger.info(f"[{task_id}] Phase 1: Skipped (video data managed externally)")
                elif phase_num == 2:
                    # TODO: Phase 2 - load sentences from Phase 1 output
                    await run_phase2(task_id, [])
                elif phase_num == 3:
                    # TODO: Phase 3 - integrate annotation organization logic
                    logger.info(f"[{task_id}] Phase 3: Skipped (annotations managed externally)")
                elif phase_num == 4:
                    await run_phase4(task_id, phase_input, phase_output)
                elif phase_num == 5:
                    await run_phase5(task_id, phase_input, phase_output, gpu_id=gpu_id)
                elif phase_num == 6:
                    await run_phase6(task_id, phase_input, phase_output, gpu_id=gpu_id)

                if gpu_id is not None and phase_num in (5, 6):
                    gpu_manager.release(gpu_id)

                with Session(engine) as session:
                    PhaseStateManager.mark_completed(task_id, phase_num, session)

            except Exception as e:
                if gpu_id is not None and phase_num in (5, 6):
                    gpu_manager.release(gpu_id)
                with Session(engine) as session:
                    PhaseStateManager.mark_failed(task_id, phase_num, session, str(e))
                logger.error(f"[{task_id}] Phase {phase_num} failed: {e}")
                return

        # All phases completed
        with Session(engine) as session:
            task = session.exec(
                select(PipelineTask).where(PipelineTask.task_id == task_id)
            ).first()
            if task:
                task.status = "completed"
                task.updated_at = datetime.utcnow()
                session.add(task)
                session.commit()
        logger.info(f"[{task_id}] Pipeline completed successfully")

    except Exception as e:
        logger.error(f"[{task_id}] Pipeline error: {e}")
        with Session(engine) as session:
            task = session.exec(
                select(PipelineTask).where(PipelineTask.task_id == task_id)
            ).first()
            if task:
                task.status = "failed"
                task.error_message = str(e)
                task.updated_at = datetime.utcnow()
                session.add(task)
                session.commit()
    finally:
        _running_tasks.pop(task_id, None)


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


@router.post("/{task_id}/run")
def run_task(
    task_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Start pipeline execution for a task."""
    task = session.exec(select(PipelineTask).where(PipelineTask.task_id == task_id)).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status == "running":
        raise HTTPException(status_code=409, detail="Task is already running")
    if task.status == "completed":
        raise HTTPException(status_code=409, detail="Task is already completed")

    import threading
    t = threading.Thread(target=_run_pipeline_sync, args=(task_id,), daemon=True)
    t.start()
    return {"message": "Pipeline started", "task_id": task_id}


@router.post("/{task_id}/pause")
def pause_task(
    task_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Pause a running task. Takes effect before the next phase starts."""
    task = session.exec(select(PipelineTask).where(PipelineTask.task_id == task_id)).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != "running":
        raise HTTPException(status_code=409, detail="Task is not running")

    _running_tasks[task_id] = True  # signal pause
    return {"message": "Pause signal sent", "task_id": task_id}


@router.post("/{task_id}/resume")
def resume_task(
    task_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Resume a paused task from its current phase."""
    task = session.exec(select(PipelineTask).where(PipelineTask.task_id == task_id)).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != "paused":
        raise HTTPException(status_code=409, detail="Task is not paused")

    import threading
    t = threading.Thread(target=_run_pipeline_sync, args=(task_id,), daemon=True)
    t.start()
    return {"message": "Pipeline resumed", "task_id": task_id}


@router.delete("/{task_id}")
def delete_task(
    task_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Delete a task and all its phase states."""
    task = session.exec(select(PipelineTask).where(PipelineTask.task_id == task_id)).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running task. Pause it first.")

    # Delete phase states
    phases = session.exec(select(PhaseState).where(PhaseState.task_id == task_id)).all()
    for phase in phases:
        session.delete(phase)
    session.delete(task)
    session.commit()
    return {"message": "Task deleted", "task_id": task_id}
