"""Phase 2: Push extracted glosses to chatsign-accuracy for human recording.

Reads glosses from Phase 1 output, generates a CSV with gloss + description,
and uploads it to the accuracy system via POST /api/admin/sentences/import.
The pipeline then pauses, waiting for human recording and review.
"""
import base64
import io
import json
import logging
from pathlib import Path

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

ACCURACY_API = settings.ACCURACY_API_URL


def _build_csv_from_glosses(glosses: dict[str, list[str]], descriptions: dict[str, str] | None = None) -> str:
    """Build CSV content from gloss extraction output.

    Args:
        glosses: {sentence: [GLOSS1, GLOSS2, ...]}
        descriptions: optional {GLOSS: description} for each gloss

    Returns:
        CSV string with columns: text, description
    """
    if descriptions is None:
        descriptions = {}

    # Collect unique glosses (lowercase) preserving order
    seen = set()
    unique_glosses = []
    for sent_glosses in glosses.values():
        for g in sent_glosses:
            g_lower = g.lower()
            if g_lower not in seen:
                seen.add(g_lower)
                unique_glosses.append(g_lower)

    lines = ["text,description"]
    for gloss in unique_glosses:
        desc = descriptions.get(gloss, descriptions.get(gloss.upper(), ""))
        # Escape CSV: quote fields containing commas
        gloss_escaped = f'"{gloss}"' if "," in gloss else gloss
        desc_escaped = f'"{desc}"' if "," in desc else desc
        lines.append(f"{gloss_escaped},{desc_escaped}")

    return "\n".join(lines) + "\n"


async def run_phase2_push(
    task_id: str,
    phase1_output: Path,
    output_dir: Path,
    batch_title: str | None = None,
    language: str = "en",
) -> dict:
    """Push glosses to accuracy system for human recording.

    Args:
        task_id: Pipeline task ID
        phase1_output: Path to Phase 1 output (contains glosses.json)
        output_dir: Phase 2 output directory
        batch_title: Title for the sentence batch in accuracy (default: task_id)
        language: Language code (default: "en")

    Returns:
        dict with gloss_count, batch_title, status
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load glosses from Phase 1
    glosses_file = phase1_output / "glosses.json"
    if not glosses_file.exists():
        raise FileNotFoundError(f"Phase 1 glosses not found: {glosses_file}")

    with open(glosses_file) as f:
        glosses = json.load(f)

    if not glosses:
        logger.warning(f"[{task_id}] Phase 2: No glosses to push")
        return {"gloss_count": 0, "batch_title": "", "status": "empty"}

    # Load descriptions from Phase 1 (generated alongside glosses)
    descriptions = {}
    desc_file = phase1_output / "descriptions.json"
    if desc_file.exists():
        with open(desc_file) as f:
            descriptions = json.load(f)

    # Build CSV
    csv_content = _build_csv_from_glosses(glosses, descriptions)
    title = batch_title or f"pipeline_{task_id}"

    # Save CSV locally for reference
    csv_path = output_dir / "glosses_upload.csv"
    with open(csv_path, "w") as f:
        f.write(csv_content)

    # Count unique glosses
    gloss_count = csv_content.strip().count("\n")  # minus header

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
            result = resp.json()
            logger.info(f"[{task_id}] Phase 2: Pushed {gloss_count} glosses to accuracy "
                        f"as batch '{title}'")
            status = "pushed"
        else:
            logger.error(f"[{task_id}] Phase 2: Accuracy API returned {resp.status_code}: {resp.text[:500]}")
            status = "api_error"
            # If batch already exists, treat as success
            if "already exists" in resp.text:
                logger.info(f"[{task_id}] Phase 2: Batch '{title}' already exists, continuing")
                status = "exists"
    except Exception as e:
        logger.warning(f"[{task_id}] Phase 2: Could not reach accuracy API ({e}), "
                       f"CSV saved locally at {csv_path}")
        status = "offline"

    # Save metadata
    meta = {
        "task_id": task_id,
        "batch_title": title,
        "gloss_count": gloss_count,
        "language": language,
        "status": status,
        "csv_path": str(csv_path),
    }
    with open(output_dir / "push_result.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta
