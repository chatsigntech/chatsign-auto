"""Phase 3 publish — push generated standardised sign videos to chatsign-accuracy
for human review.

Each completed Phase 3 video (DGX or local backend, doesn't matter — the
Phase 3 backend is a separate concern) is copied into accuracy's
``review/generated/`` directory and registered in ``pending-videos.jsonl``
as a ``source: "generated"`` entry. Reviewer assignments are NOT written
here — admins assign reviewers through the accuracy UI.

Idempotent: re-runs skip videoIds already present in ``pending-videos.jsonl``.
The mp4 itself is overwritten so a re-published Phase 3 video replaces the
older copy for the same (task_name, word).
"""
import json
import logging
import re
import shutil
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from backend.config import settings

logger = logging.getLogger(__name__)


_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _safe(s: str) -> str:
    """Sanitize for filenames/videoIds — collapse non-[A-Za-z0-9_-] runs to _."""
    return _SAFE_RE.sub("_", s).strip("_") or "x"


def _existing_video_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                vid = json.loads(line).get("videoId")
                if vid:
                    out.add(vid)
            except Exception:
                pass
    return out


def publish_one_to_accuracy(
    task_name: str,
    video_path: Path,
    word: str,
    *,
    existing_ids: set[str] | None = None,
) -> bool:
    """Copy a Phase 3 video to accuracy ``review/generated/`` and register it.

    Returns True on success, False on any failure. Never raises — failures
    are logged at WARNING and Phase 3 progression continues.

    ``existing_ids`` lets callers hoist the pending-videos.jsonl scan once
    per batch instead of paying it per video. The set is mutated on append
    so within-batch dedup works. Pass None for one-off callers (re-scans).

    Naming:
        videoFileName: ``<safe_task>_hiya_<safe_word>.mp4``  (accuracy batch
                       filter ``<safe_task>_hiya`` matches by prefix + ``_``)
        videoId:       ``gen_<safe_task>_hiya_<safe_word>``
    """
    try:
        accuracy_root = Path(settings.CHATSIGN_ACCURACY_DATA)
        gen_dir = accuracy_root / "review" / "generated"
        pending = accuracy_root / "reports" / "pending-videos.jsonl"

        if not video_path.exists() or video_path.stat().st_size == 0:
            logger.warning(f"[publish] missing/empty source video: {video_path}")
            return False

        gen_dir.mkdir(parents=True, exist_ok=True)
        pending.parent.mkdir(parents=True, exist_ok=True)

        safe_task = _safe(task_name)
        safe_word = _safe(word)
        review_name = f"{safe_task}_hiya_{safe_word}.mp4"
        video_id = f"gen_{safe_task}_hiya_{safe_word}"

        # Copy mp4 first — overwrites so newer Phase 3 outputs win.
        shutil.copy2(video_path, gen_dir / review_name)

        if existing_ids is None:
            existing_ids = _existing_video_ids(pending)
        if video_id in existing_ids:
            logger.info(f"[publish] {video_id} already registered, file refreshed")
            return True

        # Synthetic sentenceId — these videos don't map to a real sentence,
        # but accuracy's schema wants an int. Stable per videoId via crc32.
        sentence_id = 90000 + (zlib.crc32(video_id.encode()) % 9000)

        entry = {
            "videoId": video_id,
            "sentenceId": sentence_id,
            "sentenceText": word,
            "translatorId": "generated",
            "language": "en",
            "videoPath": f"review/generated/{review_name}",
            "videoFileName": review_name,
            "source": "generated",
            "addedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "localPath": review_name,
            "metadata": {"origin": "phase3", "task_name": task_name, "word": word},
        }
        with open(pending, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        existing_ids.add(video_id)
        logger.info(f"[publish] registered {video_id} -> {review_name}")
        return True

    except Exception as e:
        logger.warning(f"[publish] failed for {video_path.name}: {e}")
        return False


def make_phase3_publisher(task_name: str) -> Callable[[dict], None]:
    """Build an ``on_video_done`` callback for one Phase 3 batch.

    Reads ``pending-videos.jsonl`` once at construction and shares the
    resulting videoId set across every per-video publish in the batch —
    avoids re-scanning a multi-MB file N times for an N-video batch.
    """
    pending = Path(settings.CHATSIGN_ACCURACY_DATA) / "reports" / "pending-videos.jsonl"
    existing_ids = _existing_video_ids(pending)

    def _on_video_done(rec: dict) -> None:
        if rec.get("status") != "completed":
            return
        out = rec.get("output_path")
        if not out:
            return
        word = Path(rec.get("filename", "")).stem or "x"
        publish_one_to_accuracy(task_name, Path(out), word, existing_ids=existing_ids)

    return _on_video_done
