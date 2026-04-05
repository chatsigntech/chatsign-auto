"""Prepare videos from OpenASL/How2Sign datasets for the pipeline.

When task source is "dataset", this module locates the original video files
and prepares them in Phase 3 output format (manifest.json + videos/ symlinks),
allowing the pipeline to skip Phase 2 and 3 and go directly to Phase 4.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from backend.config import settings

# Dataset video directories
OPENASL_DIR = settings.VIDEO_DATA_ROOT / "opensl_data"
H2S_DIR = settings.VIDEO_DATA_ROOT / "how2sign_data"


def _find_video(vid: str, source: str) -> Path | None:
    """Locate video file by vid and source dataset."""
    if source == "openasl":
        path = OPENASL_DIR / f"{vid}.mp4"
    elif source == "how2sign":
        path = H2S_DIR / f"{vid}.mp4"
    else:
        return None
    return path if path.exists() else None


def prepare_dataset_videos(
    task_id: str,
    dataset_videos: list[dict],
    phase3_output: Path,
) -> dict:
    """Symlink dataset videos into Phase 3 output format.

    Args:
        task_id: Pipeline task ID
        dataset_videos: List of {text, vid, source} from task config
        phase3_output: Phase 3 output directory to populate

    Returns:
        dict with video_count, sentences, manifest_path
    """
    phase3_output.mkdir(parents=True, exist_ok=True)
    videos_dir = phase3_output / "videos"
    videos_dir.mkdir(exist_ok=True)

    manifest = []
    sentences = set()
    found = 0
    missing = 0

    for i, entry in enumerate(dataset_videos):
        vid = entry.get("vid", "")
        source = entry.get("source", "")
        text = entry.get("text", "")

        src_path = _find_video(vid, source)
        if not src_path:
            logger.warning(f"[{task_id}] Dataset video not found: {vid} ({source})")
            missing += 1
            continue

        # Use a clean filename: sentence_{i}.mp4
        filename = f"sentence_{i}.mp4"
        dst = videos_dir / filename

        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src_path.resolve())

        manifest.append({
            "video_id": f"ds_{i}",
            "filename": filename,
            "sentence_id": i,
            "sentence_text": text,
            "language": "en",
            "dataset_source": source,
            "dataset_vid": vid,
        })
        sentences.add(text)
        found += 1

    # Write manifest.json
    manifest_path = phase3_output / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Write sentences.txt
    with open(phase3_output / "sentences.txt", "w", encoding="utf-8") as f:
        for s in sorted(sentences):
            f.write(s + "\n")

    logger.info(f"[{task_id}] Dataset videos prepared: {found} linked, {missing} missing")

    return {
        "video_count": found,
        "missing": missing,
        "sentences": sorted(sentences),
        "manifest_path": str(manifest_path),
    }
