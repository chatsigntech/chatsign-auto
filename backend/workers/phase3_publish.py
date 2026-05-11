"""Phase 3 publish — push generated standardised sign videos to chatsign-accuracy
for human review.

Each completed Phase 3 video (DGX or local backend, doesn't matter — the
Phase 3 backend is a separate concern) is copied into accuracy's
``review/generated/`` directory and registered in ``pending-videos.jsonl``
as a ``source: "generated"`` entry. Reviewer assignments are NOT written
here — admins assign reviewers through the accuracy UI.

Per-day independent batches: videoId is suffixed with the render date
(e.g. ``gen_<src>_20260511``), so re-running Phase 3 on a different day
produces a fresh batch with its own pending-videos rows + render-dir
files. Same-day re-runs are idempotent — dedup by full videoId fires
and the mp4 is overwritten in place.
"""
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from backend.config import settings
from backend.core.io_utils import read_jsonl
from backend.core.video_naming import video_filename

logger = logging.getLogger(__name__)


_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _safe(s: str) -> str:
    """Sanitize for filenames/videoIds — collapse non-[A-Za-z0-9_-] runs to _."""
    return _SAFE_RE.sub("_", s).strip("_") or "x"


def _existing_video_ids(path: Path) -> set[str]:
    return {e["videoId"] for e in read_jsonl(path) if e.get("videoId")}


def publish_one_to_accuracy(
    *,
    video_path: Path,
    annotation: dict,
    batch_file: str,
    existing_ids: set[str] | None = None,
) -> bool:
    """Copy a Phase 3 rendered video to accuracy and register it with proper metadata.

    Returns True on success, False on any failure. Never raises — failures
    are logged at WARNING and Phase 3 progression continues.

    ``annotation`` MUST contain (from Phase 2's annotations.json):
        - video_id      (source recording videoId, e.g. "sub_Tareq_<batch>_<sid>_<dur>")
        - sentence_id   (real source sid, an int)
        - sentence_text (the actual sentence the recording was for)
        - filename      (original mp4 file name)
        - language      (optional; defaults "en")

    ``batch_file`` is the rendered batch's accuracy texts/ filename
    (e.g. ``chatsign-opening-introduction-render-20260508.jsonl``).
    Used as ``batchFile`` for the new pending-videos entry so admin#publish
    UI can select this batch.

    Naming convention (mirrors inject_*.py post-render scripts):
        videoFileName: ``<md5(videoId)[:10]>_hiya.mp4``
        videoId:       ``gen_<source_video_id>_<render_date>``
        videoPath:     ``review/generated/<batch_dir>/<videoFileName>``
    """
    try:
        accuracy_root = Path(settings.CHATSIGN_ACCURACY_DATA)
        batch_dir = batch_file.removesuffix(".jsonl")
        gen_dir = accuracy_root / "review" / "generated" / batch_dir
        pending = accuracy_root / "reports" / "pending-videos.jsonl"

        if not video_path.exists() or video_path.stat().st_size == 0:
            logger.warning(f"[publish] missing/empty source video: {video_path}")
            return False

        gen_dir.mkdir(parents=True, exist_ok=True)
        pending.parent.mkdir(parents=True, exist_ok=True)

        src_video_id = annotation.get("video_id") or "unknown"
        # Batch-scoped videoId: every Phase 3 rerun on a different day produces
        # an independent batch (own pending-videos rows, own render-dir files),
        # so admin can review/publish each generation separately. Re-running
        # twice on the same day is treated as the same batch (dedup fires).
        m = re.search(r"-(\d{8})$", batch_dir)
        render_date = m.group(1) if m else "unknown"
        video_id = f"gen_{src_video_id}_{render_date}"
        review_name = video_filename(video_id)

        shutil.copy2(video_path, gen_dir / review_name)

        if existing_ids is None:
            existing_ids = _existing_video_ids(pending)
        if video_id in existing_ids:
            logger.info(f"[publish] {video_id} already registered, file refreshed")
            return True

        entry = {
            "videoId": video_id,
            "sentenceId": annotation.get("sentence_id"),
            "sentenceText": annotation.get("sentence_text", ""),
            "translatorId": "generated",
            "language": annotation.get("language", "en"),
            "videoPath": f"review/generated/{batch_dir}/{review_name}",
            "videoFileName": review_name,
            "source": "generated",
            "addedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "localPath": f"{batch_dir}/{review_name}",
            "batchFile": batch_file,
            # Audit trail back to the source recording — mirrors inject_*.py
            "_origin": {
                "src_videoId": src_video_id,
                "src_sid": annotation.get("sentence_id"),
                "src_sentence": annotation.get("sentence_text", ""),
                "src_filename": annotation.get("filename", ""),
            },
        }
        with open(pending, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        existing_ids.add(video_id)
        logger.info(f"[publish] registered {video_id} → {review_name} (batch={batch_dir})")
        return True

    except Exception as e:
        logger.warning(f"[publish] failed for {video_path.name}: {e}")
        return False


def make_phase3_publisher(
    task_name: str,
    *,
    batch_name: str | None = None,
    phase2_output: Path | None = None,
) -> Callable[[dict], None]:
    """Build an ``on_video_done`` callback for one Phase 3 batch.

    Loads phase 2's ``annotations.json`` so each rendered video carries its
    real ``sentenceId`` / ``sentenceText``. Writes
    ``texts/<batch>-render-<YYYYMMDD>.jsonl`` once at startup so accuracy's
    batch picker lists this Phase 3 output as a selectable batch.

    A video whose filename has no matching annotation is logged + skipped
    (not published) — the only legitimate source of input filenames is
    Phase 2's annotated set, so a miss is an upstream bug.
    """
    accuracy_root = Path(settings.CHATSIGN_ACCURACY_DATA)
    pending = accuracy_root / "reports" / "pending-videos.jsonl"
    existing_ids = _existing_video_ids(pending)

    annot_by_filename: dict[str, dict] = {}
    if phase2_output:
        ann_file = phase2_output / "annotations.json"
        if ann_file.exists():
            try:
                with open(ann_file, encoding="utf-8") as f:
                    ann_list = json.load(f)
                for a in ann_list:
                    fn = a.get("filename")
                    if fn:
                        annot_by_filename[fn] = a
                logger.info(
                    f"[publish] loaded {len(annot_by_filename)} annotations from {ann_file}"
                )
            except Exception as e:
                logger.warning(f"[publish] failed to read annotations.json: {e}")

    base_batch = batch_name or task_name or "phase3"
    render_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    batch_file = f"{_safe(base_batch).lower()}-render-{render_date}.jsonl"

    texts_path = accuracy_root / "texts" / batch_file
    if not texts_path.exists() and annot_by_filename:
        try:
            texts_path.parent.mkdir(parents=True, exist_ok=True)
            seen_sids: set[int] = set()
            rows = []
            for a in annot_by_filename.values():
                sid = a.get("sentence_id")
                if sid is None or sid in seen_sids:
                    continue
                seen_sids.add(sid)
                rows.append({
                    "id": sid,
                    "language": a.get("language", "en"),
                    "text": a.get("sentence_text", ""),
                })
            rows.sort(key=lambda r: r["id"])
            with open(texts_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
            logger.info(f"[publish] wrote texts/{batch_file} ({len(rows)} sentences)")
        except Exception as e:
            logger.warning(f"[publish] failed to write texts/{batch_file}: {e}")

    def _on_video_done(rec: dict) -> None:
        if rec.get("status") != "completed":
            return
        out = rec.get("output_path")
        filename = rec.get("filename", "")
        if not out or not filename:
            return
        annot = annot_by_filename.get(filename)
        if not annot:
            logger.warning(
                f"[publish] no annotation for {filename}; skipping (upstream Phase 2 issue)"
            )
            return
        publish_one_to_accuracy(
            video_path=Path(out),
            annotation=annot,
            batch_file=batch_file,
            existing_ids=existing_ids,
        )

    return _on_video_done
