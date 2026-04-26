"""Push completed dgx-pipeline-test outputs into Phase3TestJob table so the
existing /phase3-test UI lists them like any other job.

source_video_path = original 50K mp4
generated_video_path = sr final (1152x1152 white-bg)

Idempotent: re-running upserts each row by deterministic job_id `dgx-test-<base>`.
"""
import csv
import sys
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlmodel import Session, select  # noqa: E402

from backend.database import engine, init_db  # noqa: E402
from backend.models.phase3_test import Phase3TestJob  # noqa: E402

TEST_DIR = Path(__file__).resolve().parent
MANIFEST = TEST_DIR / "inputs" / "manifest.tsv"
INPUTS = TEST_DIR / "inputs"
OUTPUTS = TEST_DIR / "outputs"
WEB_DIR = TEST_DIR / "web"  # H.264 transcodes the browser can play


def find_orig_path(base: str) -> Path | None:
    """Prefer the H.264 sidecar; fall back to mp4v source if transcoding hasn't run."""
    p = WEB_DIR / f"{base}_orig.mp4"
    if p.exists():
        return p
    for sub in sorted(Path("/mnt/data/chatsign-auto-videos/50Kfull_v2").glob("csv_*")):
        p = sub / f"{base}.mp4"
        if p.exists():
            return p
    p = INPUTS / f"{base}.mp4"
    return p if p.exists() else None


def find_sr_path(base: str) -> Path | None:
    p = WEB_DIR / f"{base}_sr.mp4"
    if p.exists():
        return p
    p = OUTPUTS / f"{base}_sr.mp4"
    return p if p.exists() else None


def main():
    init_db()
    if not MANIFEST.exists():
        print(f"missing {MANIFEST}")
        return 1
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)

    inserted = updated = skipped = 0
    with Session(engine) as session:
        for r in rows:
            fname = (r.get("fname") or "").strip()
            word = (r.get("word") or "").strip()
            base = Path(fname).stem
            sr_path = find_sr_path(base)
            if sr_path is None:
                skipped += 1
                continue
            orig = find_orig_path(base)
            if orig is None:
                print(f"  WARN: no original for {fname}")
                skipped += 1
                continue

            job_id = f"dgx-test-{base}"
            existing = session.exec(
                select(Phase3TestJob).where(Phase3TestJob.job_id == job_id)
            ).first()
            now = datetime.utcnow()
            if existing is None:
                job = Phase3TestJob(
                    job_id=job_id,
                    status="completed",
                    video_id=base,
                    sentence_text=word,
                    translator_id="dgx-pipeline-test",
                    source_video_path=str(orig),
                    source_filename=fname,
                    output_dir=str(OUTPUTS),
                    generated_video_path=str(sr_path),
                    duration_sec=None,
                    created_at=now,
                    updated_at=now,
                )
                session.add(job)
                inserted += 1
            else:
                existing.status = "completed"
                existing.source_video_path = str(orig)
                existing.generated_video_path = str(sr_path)
                existing.sentence_text = word
                existing.created_at = now   # bump so rerun rises to top of /phase3-test listing
                existing.updated_at = now
                updated += 1
        session.commit()

    print(f"inserted={inserted} updated={updated} skipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
