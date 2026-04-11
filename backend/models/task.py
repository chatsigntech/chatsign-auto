from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class PipelineTask(SQLModel, table=True):
    __tablename__ = "pipeline_tasks"

    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: str = Field(index=True, unique=True)
    name: str
    status: str = Field(default="pending")  # pending, running, completed, failed, paused
    current_phase: int = Field(default=1)
    config_json: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
