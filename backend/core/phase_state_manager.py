from datetime import datetime
from sqlmodel import Session, select

from backend.models.phase import PhaseState
from backend.models.task import PipelineTask


class PhaseStateManager:

    @staticmethod
    def mark_running(task_id: str, phase_num: int, session: Session):
        phase = session.exec(
            select(PhaseState).where(
                PhaseState.task_id == task_id,
                PhaseState.phase_num == phase_num,
            )
        ).first()
        if phase:
            phase.status = "running"
            phase.started_at = datetime.utcnow()
            session.add(phase)
            session.commit()

    @staticmethod
    def mark_completed(task_id: str, phase_num: int, session: Session):
        phase = session.exec(
            select(PhaseState).where(
                PhaseState.task_id == task_id,
                PhaseState.phase_num == phase_num,
            )
        ).first()
        if phase:
            phase.status = "completed"
            phase.progress = 100.0
            phase.completed_at = datetime.utcnow()
            session.add(phase)
            session.commit()

    @staticmethod
    def mark_failed(task_id: str, phase_num: int, session: Session, error_msg: str):
        phase = session.exec(
            select(PhaseState).where(
                PhaseState.task_id == task_id,
                PhaseState.phase_num == phase_num,
            )
        ).first()
        if phase:
            phase.status = "failed"
            phase.error_message = error_msg
            session.add(phase)
            session.commit()

        task = session.exec(
            select(PipelineTask).where(PipelineTask.task_id == task_id)
        ).first()
        if task:
            task.status = "failed"
            task.error_message = error_msg
            task.updated_at = datetime.utcnow()
            session.add(task)
            session.commit()

    @staticmethod
    def update_progress(task_id: str, phase_num: int, session: Session, progress: float):
        phase = session.exec(
            select(PhaseState).where(
                PhaseState.task_id == task_id,
                PhaseState.phase_num == phase_num,
            )
        ).first()
        if phase:
            phase.progress = progress
            session.add(phase)
            session.commit()
