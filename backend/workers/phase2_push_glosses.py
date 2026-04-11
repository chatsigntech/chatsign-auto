"""Phase 2: Push glosses and sentences to chatsign-accuracy for sign language recording.

Reads glosses and sentences from Phase 1 output, generates a CSV with
text + description, and uploads it to the accuracy system via
POST /api/admin/sentences/import.

Both glosses (individual sign vocabulary) and sentences (original English
sentences) are pushed — accuracy treats them identically as recording tasks.
The pipeline then pauses, waiting for human recording and review.
"""
import base64
import csv as csv_mod
import io
import json
import logging
from pathlib import Path

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

ACCURACY_API = settings.ACCURACY_API_URL


def _build_csv(
    glosses: dict[str, list[str]],
    sentences: list[str],
    descriptions: dict[str, str] | None = None,
    asl_descriptions: dict[str, str] | None = None,
) -> tuple[str, int, int]:
    """Build CSV content with both glosses and sentences for accuracy import.

    Returns:
        (csv_content, gloss_count, sentence_count)
    """
    if descriptions is None:
        descriptions = {}
    if asl_descriptions is None:
        asl_descriptions = {}

    buf = io.StringIO()
    writer = csv_mod.writer(buf)
    writer.writerow(["text", "description", "type"])

    seen = set()
    gloss_count = 0
    sentence_count = 0

    for sent_glosses in glosses.values():
        for g in sent_glosses:
            g_lower = g.lower()
            if g_lower not in seen:
                seen.add(g_lower)
                desc = descriptions.get(g, descriptions.get(g.upper(), ""))
                writer.writerow([g_lower, desc, "gloss"])
                gloss_count += 1

    for sent in sentences:
        sent_stripped = sent.strip()
        if sent_stripped and sent_stripped not in seen:
            seen.add(sent_stripped)
            asl_desc = asl_descriptions.get(sent_stripped, "")
            writer.writerow([sent_stripped, asl_desc, "sentence"])
            sentence_count += 1

    return buf.getvalue(), gloss_count, sentence_count


async def run_phase2_push(
    task_id: str,
    phase1_output: Path,
    output_dir: Path,
    batch_title: str | None = None,
    language: str = "en",
) -> dict:
    """Push glosses and sentences to accuracy system for sign language recording.

    Args:
        task_id: Pipeline task ID
        phase1_output: Path to Phase 1 output (contains glosses.json, sentences.json)
        output_dir: Phase 2 output directory
        batch_title: Title for the sentence batch in accuracy (default: task_id)
        language: Language code (default: "en")

    Returns:
        dict with item_count, gloss_count, sentence_count, batch_title, status
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load glosses from Phase 1
    glosses_file = phase1_output / "glosses.json"
    if not glosses_file.exists():
        raise FileNotFoundError(f"Phase 1 glosses not found: {glosses_file}")
    with open(glosses_file) as f:
        glosses = json.load(f)

    # Load sentences from Phase 1
    sentences = []
    sentences_file = phase1_output / "sentences.json"
    if sentences_file.exists():
        with open(sentences_file) as f:
            sentences = json.load(f)

    if not glosses and not sentences:
        logger.warning(f"[{task_id}] Phase 2: No glosses or sentences to push")
        return {"item_count": 0, "gloss_count": 0, "sentence_count": 0,
                "batch_title": "", "status": "empty"}

    # Load descriptions from Phase 1
    descriptions = {}
    desc_file = phase1_output / "descriptions.json"
    if desc_file.exists():
        with open(desc_file) as f:
            descriptions = json.load(f)

    # Build ASL-reordered gloss descriptions for sentences
    asl_descriptions = {}
    try:
        from backend.core.sign_video_generator import reorder_glosses_asl
    except ImportError as e:
        reorder_glosses_asl = None
        logger.warning(f"[{task_id}] Phase 2: ASL reorder unavailable ({e})")
    if reorder_glosses_asl:
        for sent, sent_glosses in glosses.items():
            if sent_glosses:
                try:
                    reordered = reorder_glosses_asl(sent_glosses, sent)
                    asl_descriptions[sent] = " ".join(reordered)
                except Exception as e:
                    logger.warning(f"[{task_id}] ASL reorder failed for '{sent[:40]}': {e}")

    # Build CSV with both glosses and sentences
    csv_content, gloss_count, sentence_count = _build_csv(glosses, sentences, descriptions, asl_descriptions)
    item_count = gloss_count + sentence_count
    title = batch_title or f"pipeline_{task_id}"

    csv_path = output_dir / "upload.csv"
    with open(csv_path, "w") as f:
        f.write(csv_content)

    # Push to accuracy system via API
    csv_base64 = base64.b64encode(csv_content.encode("utf-8")).decode("ascii")
    payload = {
        "csvBase64": csv_base64,
        "title": title,
        "language": language,
    }

    try:
        async with httpx.AsyncClient(verify=False, timeout=30) as client:
            resp = await client.post(
                f"{ACCURACY_API}/api/admin/sentences/import",
                json=payload,
                headers={"X-User-Id": "chatsign2026admin"},
            )

        if resp.status_code == 200:
            logger.info(f"[{task_id}] Phase 2: Pushed {gloss_count} glosses + "
                        f"{sentence_count} sentences to accuracy as batch '{title}'")
            status = "pushed"
        else:
            logger.error(f"[{task_id}] Phase 2: Accuracy API returned {resp.status_code}: {resp.text[:500]}")
            status = "api_error"
            if "already exists" in resp.text:
                logger.info(f"[{task_id}] Phase 2: Batch '{title}' already exists, continuing")
                status = "exists"
    except Exception as e:
        logger.warning(f"[{task_id}] Phase 2: Could not reach accuracy API ({e}), "
                       f"CSV saved locally at {csv_path}")
        status = "offline"

    meta = {
        "task_id": task_id,
        "batch_title": title,
        "item_count": item_count,
        "gloss_count": gloss_count,
        "sentence_count": sentence_count,
        "language": language,
        "status": status,
        "csv_path": str(csv_path),
    }
    with open(output_dir / "push_result.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta
