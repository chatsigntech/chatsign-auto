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
from typing import Optional

# chatsign-auto root on sys.path so `from backend.config import settings` works
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd

from chatsign_pipeline import TextPipeline
from chatsign_pipeline.gloss_dict import _wordnet_first_definition
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
    """Return {text_lower: description} from master gloss.csv (NEW schema)."""
    df = pd.read_csv(settings.GLOSS_CSV_PATH, dtype=str, keep_default_na=False)
    df['text'] = df['text'].str.strip().str.lower()
    return dict(zip(df['text'], df['description']))


def compute_description(text: str, *, kind: str, master_lookup: dict[str, str]) -> str:
    """Pick the best description for an inject record.

    kind: 'word' or 'sentence' (decides the fallback path)
    master_lookup: from load_master_descriptions()
    """
    text_lower = text.strip().lower()
    if text_lower in master_lookup and master_lookup[text_lower]:
        return master_lookup[text_lower]
    if kind == 'sentence':
        tokens = get_pipeline().text_to_gloss_tokens(text)
        return ' '.join(tokens)
    return _wordnet_first_definition(text_lower)
