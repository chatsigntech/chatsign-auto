import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, delete, select

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

NUM_PHASES = 8

gpu_manager = GPUManager(
    max_gpus=settings.MAX_GPUS,
    device_ids=settings.cuda_device_ids,
)

# Track running tasks for pause support (single-worker only)
_running_tasks: dict[str, bool] = {}


class TaskCreate(BaseModel):
    name: str
    input_text: str  # source text to convert to sign language
    batch_name: Optional[str] = None  # accuracy batch filter (e.g. "school_unmatch")


class TaskResponse(BaseModel):
    task_id: str
    name: str
    status: str
    current_phase: int
    created_at: datetime


def _get_task_or_404(session: Session, task_id: str) -> PipelineTask:
    task = session.exec(select(PipelineTask).where(PipelineTask.task_id == task_id)).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


def _fetch_task(session: Session, task_id: str) -> Optional[PipelineTask]:
    return session.exec(select(PipelineTask).where(PipelineTask.task_id == task_id)).first()


def _update_task_status(task_id: str, status: str, **fields):
    with Session(engine) as session:
        task = _fetch_task(session, task_id)
        if task:
            task.status = status
            task.updated_at = datetime.utcnow()
            for k, v in fields.items():
                setattr(task, k, v)
            session.add(task)
            session.commit()


def _run_pipeline_sync(task_id: str):
    """Sync wrapper to run the async pipeline in a background thread."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run_pipeline(task_id))
    finally:
        loop.close()


async def _run_pipeline(task_id: str):
    """Execute pipeline phases sequentially in the background."""
    from backend.workers.phase1_worker import run_phase1
    from backend.workers.phase2_worker import run_phase2
    from backend.workers.phase3_worker import run_phase3
    from backend.workers.phase4_person_transfer import run_phase4_transfer
    from backend.workers.phase5_video_process import run_phase5_process
    from backend.workers.phase6_framer import run_phase6_framer
    from backend.workers.phase7_augment import run_phase7_augment
    from backend.workers.phase8_training import run_phase8_training

    _running_tasks[task_id] = False
    data_root = settings.SHARED_DATA_ROOT / task_id

    try:
        with Session(engine) as session:
            task = _fetch_task(session, task_id)
            if not task:
                return
            start_phase = task.current_phase or 1
            task_config = json.loads(task.config_json) if task.config_json else {}
            batch_name = task_config.get("batch_name")
            task.status = "running"
            task.updated_at = datetime.utcnow()
            session.add(task)
            session.commit()

        phase_outputs = {i: data_root / f"phase_{i}" / "output" for i in range(1, NUM_PHASES + 1)}

        for phase_num in range(start_phase, NUM_PHASES + 1):
            if _running_tasks.get(task_id):
                _update_task_status(task_id, "paused", current_phase=phase_num)
                logger.info(f"[{task_id}] Pipeline paused at phase {phase_num}")
                return

            with Session(engine) as session:
                task = _fetch_task(session, task_id)
                if task:
                    task.current_phase = phase_num
                    task.updated_at = datetime.utcnow()
                    session.add(task)
                PhaseStateManager.mark_running(task_id, phase_num, session)
                session.commit()

            phase_input = data_root / f"phase_{phase_num}" / "input"
            phase_output = phase_outputs[phase_num]
            phase_input.mkdir(parents=True, exist_ok=True)
            phase_output.mkdir(parents=True, exist_ok=True)

            try:
                gpu_id = None
                if phase_num in (4, 5, 6, 8):
                    gpu_id = gpu_manager.acquire(task_id)
                    if gpu_id is None:
                        gpu_id = 0

                if phase_num == 1:
                    # Phase 1: Gloss extraction from user input text
                    input_text = task_config.get("input_text", "")
                    sentences = [input_text] if input_text else []
                    await run_phase2(task_id, sentences, output_dir=phase_output)
                    logger.info(f"[{task_id}] Phase 1: gloss extracted from input text")

                elif phase_num == 2:
                    # Phase 2: Video collection - match glosses to accuracy videos
                    result = await run_phase1(task_id, phase_output,
                                              batch_name=batch_name,
                                              gloss_filter=phase_outputs[1])
                    logger.info(f"[{task_id}] Phase 2: collected {result['video_count']} videos")

                elif phase_num == 3:
                    # Phase 3: Annotation organization
                    await run_phase3(task_id, phase_outputs[2], phase_outputs[1], phase_output)

                elif phase_num == 4:
                    # Phase 4: Person transfer (MimicMotion) on collected videos
                    p2_videos = phase_outputs[2] / "videos"
                    input_dir = p2_videos if p2_videos.exists() else phase_input
                    await run_phase4_transfer(task_id, input_dir, phase_output, gpu_id=gpu_id)

                elif phase_num == 5:
                    # Phase 5: Video processing (extract frames → dedup → pose filter → resize → boundary frames)
                    await run_phase5_process(task_id, phase_outputs[4], phase_output)

                elif phase_num == 6:
                    # Phase 6: FramerTurbo interpolation + combine into final videos
                    await run_phase6_framer(task_id, phase_outputs[5], phase_output, gpu_id=gpu_id)

                elif phase_num == 7:
                    # Phase 7: Data augmentation (guava-aug: 2D + temporal + 3D novel views)
                    p6_videos = phase_outputs[6] / "videos"
                    input_dir = p6_videos if p6_videos.exists() else phase_outputs[6]
                    await run_phase7_augment(task_id, input_dir, phase_output)

                elif phase_num == 8:
                    # Phase 8: Model training (gloss_aware)
                    await run_phase8_training(task_id, phase_outputs[7], phase_output, gpu_id=gpu_id)

                if gpu_id is not None and phase_num in (4, 5, 6, 8):
                    gpu_manager.release(gpu_id)

                with Session(engine) as session:
                    PhaseStateManager.mark_completed(task_id, phase_num, session)

            except Exception as e:
                if gpu_id is not None and phase_num in (4, 5, 6, 8):
                    gpu_manager.release(gpu_id)
                with Session(engine) as session:
                    PhaseStateManager.mark_failed(task_id, phase_num, session, str(e))
                logger.error(f"[{task_id}] Phase {phase_num} failed: {e}")
                return

        _update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] Pipeline completed successfully")

    except Exception as e:
        logger.error(f"[{task_id}] Pipeline error: {e}")
        _update_task_status(task_id, "failed", error_message=str(e))
    finally:
        _running_tasks.pop(task_id, None)


def _start_pipeline_thread(task_id: str):
    t = threading.Thread(target=_run_pipeline_sync, args=(task_id,), daemon=True)
    t.start()


@router.post("/", response_model=TaskResponse)
def create_task(
    body: TaskCreate,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    task_id = str(uuid.uuid4())[:8]
    config = {"input_text": body.input_text}
    if body.batch_name:
        config["batch_name"] = body.batch_name
    task = PipelineTask(
        task_id=task_id,
        name=body.name,
        config_json=json.dumps(config),
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
    task = _get_task_or_404(session, task_id)
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
    task = _get_task_or_404(session, task_id)
    if task.status == "running":
        raise HTTPException(status_code=409, detail="Task is already running")
    if task.status == "completed":
        raise HTTPException(status_code=409, detail="Task is already completed")

    _start_pipeline_thread(task_id)
    return {"message": "Pipeline started", "task_id": task_id}


@router.post("/{task_id}/pause")
def pause_task(
    task_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Pause a running task. Takes effect before the next phase starts."""
    task = _get_task_or_404(session, task_id)
    if task.status != "running":
        raise HTTPException(status_code=409, detail="Task is not running")

    _running_tasks[task_id] = True
    return {"message": "Pause signal sent", "task_id": task_id}


@router.post("/{task_id}/resume")
def resume_task(
    task_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Resume a paused task from its current phase."""
    task = _get_task_or_404(session, task_id)
    if task.status != "paused":
        raise HTTPException(status_code=409, detail="Task is not paused")

    _start_pipeline_thread(task_id)
    return {"message": "Pipeline resumed", "task_id": task_id}


@router.delete("/{task_id}")
def delete_task(
    task_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Delete a task and all its phase states."""
    task = _get_task_or_404(session, task_id)
    if task.status == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running task. Pause it first.")

    session.exec(delete(PhaseState).where(PhaseState.task_id == task_id))
    session.delete(task)
    session.commit()
    return {"message": "Task deleted", "task_id": task_id}
