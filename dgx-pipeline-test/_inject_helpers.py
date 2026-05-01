"""Shared helpers for inject_*.py scripts.

Centralizes TextPipeline init + per-record description lookup so each inject
script doesn't reinvent the wheel. Description fallback chain:

    master gloss.csv hit → master row's description
    OOV sentence        → ' '.join(text_to_gloss_tokens(sent))   (ASL gloss)
    OOV word            → WordNet first-synset definition
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

# Re-export video_filename so inject_*.py can grab everything from one module.
# Kept in a separate light file so run_*.py can import it without paying the
# chatsign-pipeline cold-start cost.
from _naming import video_filename  # noqa: F401

# chatsign-auto root on sys.path so `from backend.config import settings` works
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from chatsign_pipeline import TextPipeline
from chatsign_pipeline.gloss_dict import (
    TYPE_SENTENCE,
    TYPE_WORD,
    load_dict,
    wordnet_first_definition,
)
from backend.config import settings


_pipeline: Optional[TextPipeline] = None


def get_pipeline() -> TextPipeline:
    """Lazy-init a generate-mode TextPipeline (cold start ~30s)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = TextPipeline(
            gloss_csv_path=settings.GLOSS_CSV_PATH,
            letters_dir=settings.SIGN_VIDEO_OUTPUT_DIR / "letters",
            model_dir=settings.SENTENCE_TRANSFORMER_MODEL_DIR,
            embedding_cache_dir=settings.EMBEDDING_CACHE_DIR,
            mode='generate',
            enable_fingerspell_gate=False,  # batch-extraction mode: surface every OOV
        )
    return _pipeline


def load_master_descriptions() -> dict[str, str]:
    """Return {text_lower: description} from master gloss.csv (validated NEW schema)."""
    df = load_dict(settings.GLOSS_CSV_PATH)
    return dict(zip(df['text'].str.strip().str.lower(), df['description']))


def compute_description(
    text: str, *, kind: Literal['word', 'sentence'], master_lookup: dict[str, str],
) -> str:
    """Description fallback chain: master row → ASL gloss tokens → WordNet."""
    text_lower = text.strip().lower()
    if master_lookup.get(text_lower):
        return master_lookup[text_lower]
    if kind == TYPE_SENTENCE:
        return ' '.join(get_pipeline().text_to_gloss_tokens(text))
    return wordnet_first_definition(text_lower)


def load_existing_video_ids(pending_path) -> set[str]:
    """Read pending-videos.jsonl and collect every existing videoId (deduped)."""
    import json
    from pathlib import Path
    p = Path(pending_path)
    if not p.exists():
        return set()
    out: set[str] = set()
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vid = json.loads(line).get('videoId')
            except Exception:
                continue
            if vid:
                out.add(vid)
    return out
