#!/usr/bin/env python3
"""Inject the DGX smoke test result as a Phase3TestJob row so it's viewable
at https://auto.chatsign.ai/phase3-test.

Copies the h264-remuxed mp4 into PHASE3_TEST_OUTPUT_DIR/<job_id>/generated.mp4
and inserts a completed Phase3TestJob pointing at it.
"""
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/home/chatsign/lizh/chatsign-auto")
from sqlmodel import Session

from backend.config import settings
from backend.database import engine
from backend.models.phase3_test import Phase3TestJob


SMOKE_DIR = Path(__file__).resolve().parent
SMOKE_H264 = SMOKE_DIR / "out" / "smoke_8bb48253c61e_h264.mp4"
SOURCE_VIDEO = Path(
    "/home/chatsign/lizh/chatsign-auto/chatsign-accuracy/backend/data/review/generated/001_hello.mp4"
)

# From pending-videos.jsonl lookup earlier
VIDEO_ID = "gen_001"
SENTENCE_TEXT = "Hello."
TRANSLATOR_ID = "generated"

# From the smoke test run
TOTAL_WALL_SEC = 633.0


def main() -> int:
    if not SMOKE_H264.exists():
        print(f"[FAIL] h264 remux missing: {SMOKE_H264}")
        return 1
    if not SOURCE_VIDEO.exists():
        print(f"[FAIL] source accuracy video missing: {SOURCE_VIDEO}")
        return 1

    job_id = f"dgx_{uuid.uuid4().hex[:8]}"
    output_dir = settings.PHASE3_TEST_OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage the source (matches what /run does) and the generated video
    input_dir = output_dir / "input"
    input_dir.mkdir(exist_ok=True)
    staged_source = input_dir / SOURCE_VIDEO.name
    shutil.copy2(SOURCE_VIDEO, staged_source)

    generated = output_dir / "generated.mp4"
    shutil.copy2(SMOKE_H264, generated)

    now = datetime.utcnow()
    job = Phase3TestJob(
        job_id=job_id,
        status="completed",
        video_id=VIDEO_ID,
        sentence_text=SENTENCE_TEXT,
        translator_id=TRANSLATOR_ID,
        source_video_path=str(staged_source),
        source_filename=SOURCE_VIDEO.name,
        output_dir=str(output_dir),
        generated_video_path=str(generated),
        duration_sec=TOTAL_WALL_SEC,
        transfer_time_sec=TOTAL_WALL_SEC,  # DGX end-to-end, all attributed here
        process_time_sec=0.0,
        framer_time_sec=0.0,
        error_message="[DGX smoke test] single-video MimicMotion via sbatch. "
                      "No local process/framer steps.",
        created_at=now,
        updated_at=now,
    )

    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)

    print(f"[OK] inserted job_id={job_id}")
    print(f"     page:      https://auto.chatsign.ai/phase3-test")
    print(f"     job JSON:  https://auto.chatsign.ai/api/phase3-test/jobs/{job_id}")
    print(f"     video:     https://auto.chatsign.ai/api/phase3-test/jobs/{job_id}/generated-video")
    return 0


if __name__ == "__main__":
    sys.exit(main())
