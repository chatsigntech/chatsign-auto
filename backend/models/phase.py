from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class PhaseState(SQLModel, table=True):
    __tablename__ = "phase_states"

    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: str = Field(index=True)
    phase_num: int
    status: str = Field(default="pending")  # pending, running, completed, failed, paused
    progress: float = Field(default=0.0)  # 0.0 - 100.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    gpu_id: Optional[int] = None
