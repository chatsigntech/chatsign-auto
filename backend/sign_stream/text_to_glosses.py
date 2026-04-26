"""Text → ASL gloss list using the existing pseudo-gloss-English pipeline.

Wraps `combined_pipeline.combined_gloss_pipeline` to preserve word order
(unlike `sign_video_generator.extract_ordered_glosses` which applies ASL
grammar reordering). Reuses the spaCy + GlossVocab caches that already
live in `sign_video_generator` so we don't double-load either.
"""
import sys
from pathlib import Path

from backend.core.sign_video_generator import _get_nlp_sm, _get_vocab_db

_PGE_ROOT = Path(__file__).resolve().parent.parent.parent / "pseudo-gloss-English"
for _p in (_PGE_ROOT, _PGE_ROOT / "asl_gloss_seprate"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def warmup() -> None:
    _get_vocab_db()
    _get_nlp_sm()


def text_to_glosses(text: str) -> list[str]:
    """Return ordered list of UPPERCASE glosses for the input English text."""
    from combined_pipeline import combined_gloss_pipeline  # noqa: WPS433
    return combined_gloss_pipeline(text, _get_vocab_db(), _get_nlp_sm())
