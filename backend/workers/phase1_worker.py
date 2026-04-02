"""Phase 3 worker: Collect approved videos from chatsign-accuracy matching extracted glosses."""
import csv
import json
import logging
import shutil
from pathlib import Path

from backend.config import settings
from backend.core.io_utils import read_jsonl

logger = logging.getLogger(__name__)

ACCURACY_DATA = settings.CHATSIGN_ACCURACY_DATA
REPORTS_DIR = ACCURACY_DATA / "reports"
TEXTS_DIR = ACCURACY_DATA / "texts"


def _get_approved_video_ids() -> set[str]:
    decisions = read_jsonl(REPORTS_DIR / "review-decisions.jsonl")
    return {d["videoId"] for d in decisions if d.get("decision") == "approved"}


def _get_pending_videos(batch_name: str | None = None) -> list[dict]:
    videos = read_jsonl(REPORTS_DIR / "pending-videos.jsonl")
    if batch_name:
        prefix = batch_name + "_"
        videos = [v for v in videos if v.get("source") == "submission"
                  and v.get("videoFileName", "").startswith(prefix)]
    else:
        videos = [v for v in videos if v.get("source") == "submission"]
    return videos


def _load_glosses(gloss_dir: Path) -> set[str]:
    """Load extracted glosses from Phase 1 output to filter videos."""
    glosses = set()
    glosses_file = gloss_dir / "glosses.json"
    if glosses_file.exists():
        with open(glosses_file) as f:
            data = json.load(f)
        for sent_glosses in data.values():
            glosses.update(g.lower() for g in sent_glosses)
    return glosses


async def run_phase1(task_id: str, output_dir: Path,
                     batch_name: str | None = None,
                     gloss_filter: Path | None = None) -> dict:
    """
    Collect approved videos from accuracy, optionally filtered by glosses.

    Args:
        task_id: Pipeline task identifier
        output_dir: Where to place collected videos and manifest
        batch_name: Optional batch filter (e.g. "school_unmatch")
        gloss_filter: Path to Phase 1 (gloss extraction) output dir.
                      If provided, only collect videos whose sentenceText
                      matches one of the extracted glosses.

    Returns:
        dict with keys: video_count, sentences, manifest_path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_out = output_dir / "videos"
    videos_out.mkdir(exist_ok=True)

    # Load gloss filter if provided (from Phase 1 gloss extraction)
    target_glosses = set()
    if gloss_filter:
        target_glosses = _load_glosses(gloss_filter)
        logger.info(f"[{task_id}] Phase 2: filtering by {len(target_glosses)} glosses: {target_glosses}")

    # Get approved video IDs
    approved_ids = _get_approved_video_ids()
    logger.info(f"[{task_id}] Phase 2: {len(approved_ids)} approved videos in review system")

    # Get submission videos (optionally filtered by batch)
    pending = _get_pending_videos(batch_name)
    logger.info(f"[{task_id}] Phase 2: {len(pending)} submission videos"
                + (f" for batch '{batch_name}'" if batch_name else ""))

    # Intersect: only approved submissions
    approved_videos = [v for v in pending if v.get("videoId") in approved_ids]

    # Filter by glosses if provided
    if target_glosses:
        approved_videos = [
            v for v in approved_videos
            if v.get("sentenceText", "").lower() in target_glosses
        ]

    logger.info(f"[{task_id}] Phase 2: {len(approved_videos)} videos to collect")

    if not approved_videos:
        logger.warning(f"[{task_id}] Phase 1: No approved videos found")
        return {"video_count": 0, "sentences": [], "manifest_path": None}

    # Load sentence metadata for CSV
    sentence_map = {}
    if batch_name:
        batch_file = f"{batch_name}.jsonl"
        sentence_map = _get_sentences(batch_file)
    else:
        # Load all text batches
        if TEXTS_DIR.exists():
            for tf in TEXTS_DIR.glob("*.jsonl"):
                for sid, sdata in _get_sentences(tf.name).items():
                    sentence_map[sid] = sdata

    # Copy videos and build manifest
    manifest = []
    sentences_collected = set()

    for video in approved_videos:
        video_rel = video.get("videoPath", "")
        if video_rel.startswith("/"):
            video_rel = video_rel[1:]
        src = ACCURACY_DATA / video_rel

        if not src.exists():
            logger.warning(f"[{task_id}] Phase 1: Video file not found: {src}")
            continue

        filename = video.get("videoFileName", src.name)
        dst = videos_out / filename

        # Symlink to avoid copying large files
        if dst.exists():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)

        sentence_id = video.get("sentenceId")
        sentence_info = sentence_map.get(sentence_id, {})
        sentence_text = sentence_info.get("text", video.get("sentenceText", ""))

        manifest.append({
            "video_id": video.get("videoId"),
            "filename": filename,
            "sentence_id": sentence_id,
            "sentence_text": sentence_text,
            "language": video.get("language", "en"),
        })
        if sentence_text:
            sentences_collected.add(sentence_text)

    # Write manifest.json
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Write gloss.csv (compatible with accuracy export format)
    csv_path = output_dir / "gloss.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gloss", "description", "video_filename"])
        for entry in manifest:
            writer.writerow([
                entry["sentence_text"],
                "",
                entry["filename"],
            ])

    # Write sentences.txt (one per line, for Phase 2)
    sentences_path = output_dir / "sentences.txt"
    with open(sentences_path, "w", encoding="utf-8") as f:
        for s in sorted(sentences_collected):
            f.write(s + "\n")

    logger.info(f"[{task_id}] Phase 1 completed: {len(manifest)} videos, "
                f"{len(sentences_collected)} unique sentences")

    return {
        "video_count": len(manifest),
        "sentences": sorted(sentences_collected),
        "manifest_path": str(manifest_path),
    }
