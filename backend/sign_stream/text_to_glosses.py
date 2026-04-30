"""Text → ASL gloss list — delegated to chatsign-pipeline.

Same TextPipeline as Phase 2 / sign_video_generator, but instantiated in
`generate` mode so OOV words fingerspell, digits decompose, and L4 semantic
search runs (sign-stream takes arbitrary user input, not curated training text).
"""
from chatsign_pipeline import TextPipeline

from backend.config import settings

_pipeline: TextPipeline | None = None


def _get_pipeline() -> TextPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TextPipeline(
            gloss_csv_path=settings.GLOSS_CSV_PATH,
            letters_dir=settings.SIGN_VIDEO_OUTPUT_DIR / "letters",
            model_dir=settings.SENTENCE_TRANSFORMER_MODEL_DIR,
            embedding_cache_dir=settings.EMBEDDING_CACHE_DIR,
            mode='generate',
        )
    return _pipeline


def warmup() -> None:
    _get_pipeline()


def text_to_glosses(text: str) -> list[str]:
    """Return ordered list of UPPERCASE glosses for the input English text.

    Output contract:
      - normal word/phrase  → "WORD" / "GOOD_MORNING"
      - number components   → "20", "HUNDRED", "24"
      - fingerspell letters → emitted as separate single-char tokens
                              ("NYU" → ["N", "Y", "U"])
    """
    if not text or not text.strip():
        return []
    return _get_pipeline().text_to_gloss_tokens(text)
