"""Phase 1: Gloss extraction using pseudo-gloss-English combined pipeline.

Uses the combined pipeline (ASL vocabulary phrase merging + POS filtering)
from the pseudo-gloss-English submodule. This replaces the previous
vocabulary-only matching with a two-stage approach:

  1. ASL vocabulary phrase merging — multi-word signs recognized and merged
  2. POS filtering fallback — unmatched words filtered by part-of-speech,
     keeping NOUN/VERB/ADJ/ADV/NUM/PRON/PROPN

The combined pipeline provides higher coverage than vocabulary-only matching
because words not in gloss.csv can still be kept if they are content words.
"""
import collections
import json
import logging
import re
import sys
from pathlib import Path

import spacy

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PGE_ROOT = _PROJECT_ROOT / "pseudo-gloss-English"
_PGE_ASL = _PGE_ROOT / "asl_gloss_seprate"

for _p in (_PGE_ROOT, _PGE_ASL):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from combined_pipeline import combined_gloss_pipeline  # noqa: E402
from asl_gloss_extract import GlossVocab, expand_contractions, STOP_WORDS  # noqa: E402

_DEFAULT_GLOSS_CSV = _PROJECT_ROOT / "data" / "gloss.csv"


async def run_phase2(
    task_id: str,
    sentences: list[str],
    output_dir: Path | None = None,
    gloss_csv: Path | str | None = None,
) -> dict:
    """Extract glosses from sentences using the combined pipeline.

    Args:
        task_id: Pipeline task ID
        sentences: List of English sentences
        output_dir: If provided, write glosses.json, descriptions.json, vocab.json, etc.
        gloss_csv: Path to gloss.csv (default: data/gloss.csv)

    Returns:
        dict mapping sentence -> list of matched glosses (uppercase)
    """
    csv_path = Path(gloss_csv) if gloss_csv else _DEFAULT_GLOSS_CSV

    if not sentences:
        logger.warning(f"[{task_id}] Phase 1: No sentences to process")
        glosses = {}
        descriptions = {}
        vocab = {"size": 0, "total_tokens": 0, "frequency": {}}
        match_details = []
        unmatched = []
    else:
        logger.info(f"[{task_id}] Phase 1: processing {len(sentences)} sentences "
                     f"via combined pipeline (vocab merge + POS filter)")

        # Load shared resources once
        vocab_db = GlossVocab(csv_path)
        nlp = spacy.load("en_core_web_sm")

        glosses = {}
        descriptions = {}
        vocab_counter = collections.Counter()
        match_details = []
        unmatched_tokens = set()

        for sent in sentences:
            # Combined pipeline: vocab phrase merge + POS filter
            sent_glosses = combined_gloss_pipeline(sent, vocab_db, nlp)
            glosses[sent] = sent_glosses
            vocab_counter.update(sent_glosses)

            # Collect detailed match info from vocabulary layer
            expanded = expand_contractions(sent)
            tokens = vocab_db.tokenize_with_phrases(expanded)
            for token in tokens:
                token_lower = token.lower().strip()
                if not token_lower or token_lower in STOP_WORDS:
                    continue
                if re.fullmatch(r'\d+', token_lower):
                    continue
                result = vocab_db.lookup(token)
                if result:
                    gloss_word = result["matched_to"].upper()
                    if gloss_word not in descriptions and result.get("gloss"):
                        descriptions[gloss_word] = result["gloss"]
                    match_details.append({
                        "input": token_lower,
                        "matched_to": result["matched_to"],
                        "ref": result["ref"],
                        "match_type": result["match_type"],
                        "confidence": result["confidence"],
                    })
                else:
                    unmatched_tokens.add(token_lower)

        vocab = {
            "size": len(vocab_counter),
            "total_tokens": sum(vocab_counter.values()),
            "frequency": dict(vocab_counter.most_common()),
        }
        unmatched = sorted(unmatched_tokens)

        if unmatched:
            logger.info(f"[{task_id}] Phase 1: {len(unmatched)} tokens unmatched in vocab "
                        f"(may be retained by POS filter): {unmatched[:20]}")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "glosses.json", "w", encoding="utf-8") as f:
            json.dump(glosses, f, ensure_ascii=False, indent=2)
        with open(output_dir / "descriptions.json", "w", encoding="utf-8") as f:
            json.dump(descriptions, f, ensure_ascii=False, indent=2)
        with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        if match_details:
            with open(output_dir / "match_details.json", "w", encoding="utf-8") as f:
                json.dump(match_details, f, ensure_ascii=False, indent=2)
        if unmatched:
            with open(output_dir / "unmatched.json", "w", encoding="utf-8") as f:
                json.dump(unmatched, f, ensure_ascii=False, indent=2)

    logger.info(f"[{task_id}] Phase 1 completed: {len(glosses)} sentences, "
                f"vocab size {vocab['size']}")
    return glosses
