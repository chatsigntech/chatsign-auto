"""Semantic sentence search over filtered OpenASL and How2Sign annotations.

Loads filtered annotation files, builds spaCy doc vectors on first call,
then uses cosine similarity to find sentences matching a topic query.
"""
import csv
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VIDEO_ROOT = Path("/mnt/data/chatsign-auto-videos")
OPENASL_FILTERED = VIDEO_ROOT / "opensl_data" / "annotations" / "openasl-v1.0-filtered.tsv"
H2S_FILTERED = VIDEO_ROOT / "how2sign_data" / "annotations" / "en" / "raw_text" / "how2sign_train-filtered.csv"

# Singleton cache
_cache: Optional[dict] = None


def _load_sentences() -> list[dict]:
    """Load sentences from both filtered annotation files."""
    sentences = []

    # OpenASL
    if OPENASL_FILTERED.exists():
        with open(OPENASL_FILTERED, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = (row.get("raw-text") or "").strip()
                if text and len(text) > 5:
                    sentences.append({
                        "text": text,
                        "source": "openasl",
                        "vid": row.get("vid", ""),
                    })
        logger.info(f"Loaded {sum(1 for s in sentences if s['source'] == 'openasl')} OpenASL sentences")

    # How2Sign
    h2s_start = len(sentences)
    if H2S_FILTERED.exists():
        with open(H2S_FILTERED, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = (row.get("SENTENCE") or "").strip()
                if text and len(text) > 5:
                    sentences.append({
                        "text": text,
                        "source": "how2sign",
                        "vid": row.get("SENTENCE_NAME", ""),
                    })
        logger.info(f"Loaded {len(sentences) - h2s_start} How2Sign sentences")

    return sentences


def _get_cache():
    """Build or return cached sentence vectors."""
    global _cache
    if _cache is not None:
        return _cache

    import spacy
    logger.info("Building sentence search index (first call, may take a minute)...")
    nlp = spacy.load("en_core_web_md", disable=["ner", "parser", "lemmatizer"])

    sentences = _load_sentences()
    if not sentences:
        _cache = {"sentences": [], "vectors": np.array([]), "nlp": nlp}
        return _cache

    # Batch process with nlp.pipe for efficiency
    texts = [s["text"] for s in sentences]
    vectors = []
    valid_sentences = []
    for doc, sent in zip(nlp.pipe(texts, batch_size=512, n_process=1), sentences):
        if doc.has_vector and np.any(doc.vector):
            vectors.append(doc.vector)
            valid_sentences.append(sent)

    vectors_np = np.array(vectors, dtype=np.float32)
    # L2 normalize for cosine similarity via dot product
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors_np = vectors_np / norms

    _cache = {
        "sentences": valid_sentences,
        "vectors": vectors_np,
        "nlp": nlp,
    }
    logger.info(f"Sentence search index ready: {len(valid_sentences)} sentences")
    return _cache


def search(query: str, count: int = 50) -> list[dict]:
    """Find sentences semantically similar to query.

    Args:
        query: Topic description (e.g. "shopping mall")
        count: Number of sentences to return

    Returns:
        List of {text, source, vid, score} dicts, sorted by relevance.
    """
    cache = _get_cache()
    if len(cache["sentences"]) == 0:
        return []

    nlp = cache["nlp"]
    query_doc = nlp(query)
    if not query_doc.has_vector or not np.any(query_doc.vector):
        return []

    query_vec = query_doc.vector.astype(np.float32)
    query_vec = query_vec / (np.linalg.norm(query_vec) or 1)

    # Cosine similarity via dot product (vectors are already normalized)
    scores = cache["vectors"] @ query_vec
    top_indices = np.argsort(scores)[::-1][:count]

    results = []
    for idx in top_indices:
        s = cache["sentences"][idx]
        results.append({
            "text": s["text"],
            "source": s["source"],
            "vid": s["vid"],
            "score": round(float(scores[idx]), 4),
        })

    return results
