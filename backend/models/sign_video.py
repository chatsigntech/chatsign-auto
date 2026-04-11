from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class SignVideoGeneration(SQLModel, table=True):
    __tablename__ = "sign_video_generations"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(index=True, unique=True)
    title: str
    input_text: str
    status: str = Field(default="pending")
    glosses_json: Optional[str] = None
    match_result_json: Optional[str] = None
    unmatched_glosses: Optional[str] = None
    video_path: Optional[str] = None
    video_filename: Optional[str] = None
    duration_sec: Optional[float] = None
    gloss_count: int = Field(default=0)
    matched_count: int = Field(default=0)
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
