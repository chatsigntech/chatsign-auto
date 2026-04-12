from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Phase3TestJob(SQLModel, table=True):
    __tablename__ = "phase3_test_jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(index=True, unique=True)
    status: str = Field(default="pending")

    video_id: str
    sentence_text: str = ""
    translator_id: str = ""
    source_video_path: str
    source_filename: str

    output_dir: Optional[str] = None
    generated_video_path: Optional[str] = None

    duration_sec: Optional[float] = None
    transfer_time_sec: Optional[float] = None
    process_time_sec: Optional[float] = None
    framer_time_sec: Optional[float] = None

    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
