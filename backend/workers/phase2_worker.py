"""Phase 1: Gloss extraction (delegates to chatsign-pipeline).

Writes 6 JSON files: glosses.json, descriptions.json, sentences.json,
unmatched.json, match_details.json, vocab.json.
"""
import json
import logging
from pathlib import Path

from chatsign_pipeline import TextPipeline

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_GLOSS_CSV = _PROJECT_ROOT / "data" / "gloss.csv"

# TextPipeline init is heavy (lemmatizes 27k vocab rows ~80s) — cache across calls.
_pipeline_cache: dict[str, TextPipeline] = {}


def _get_pipeline(gloss_csv: Path) -> TextPipeline:
    key = str(gloss_csv)
    if key not in _pipeline_cache:
        _pipeline_cache[key] = TextPipeline(gloss_csv_path=gloss_csv, mode='train')
    return _pipeline_cache[key]


async def run_phase2(
    task_id: str,
    input_text: str,
    output_dir: Path | None = None,
    gloss_csv: Path | str | None = None,
) -> dict:
    """Split text into sentences and extract glosses (train mode).

    Returns:
        dict mapping sentence -> list of matched glosses (uppercase) — same as
        legacy phase2_worker_OLD.run_phase2.
    """
    csv_path = Path(gloss_csv) if gloss_csv else _DEFAULT_GLOSS_CSV
    input_text = (input_text or "").strip()

    if not input_text:
        logger.warning(f"[{task_id}] Phase 1: No input text")
        if output_dir:
            _write_empty_outputs(output_dir)
        return {}

    pipeline = _get_pipeline(csv_path)
    extraction = pipeline.extract_glosses_per_sentence(input_text)

    logger.info(
        f"[{task_id}] Phase 1: {len(extraction.sentences)} clauses, "
        f"vocab size {extraction.vocab.get('size', 0)}, "
        f"{len(extraction.unmatched)} unmatched"
    )
    if extraction.unmatched:
        logger.info(
            f"[{task_id}] Phase 1: unmatched sample: {extraction.unmatched[:20]}"
        )

    if output_dir:
        _write_outputs(output_dir, extraction)

    return extraction.glosses


def _write_outputs(output_dir: Path, extraction) -> None:
    """Write 6 JSON files in the same shape as legacy phase2_worker_OLD."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _dump(output_dir / "sentences.json", extraction.sentences)
    _dump(output_dir / "glosses.json", extraction.glosses)
    _dump(output_dir / "descriptions.json", extraction.descriptions)
    _dump(output_dir / "vocab.json", extraction.vocab)
    if extraction.match_details:
        _dump(output_dir / "match_details.json", extraction.match_details)
    if extraction.unmatched:
        _dump(output_dir / "unmatched.json", extraction.unmatched)


def _write_empty_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, val in [
        ("sentences.json", []), ("glosses.json", {}),
        ("descriptions.json", {}),
        ("vocab.json", {"size": 0, "total_tokens": 0, "frequency": {}}),
    ]:
        _dump(output_dir / name, val)


def _dump(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
