"""Phase 1: Gloss extraction using pseudo-gloss-English submodule.

Combines ASL vocabulary matching with POS filtering fallback:
  1. Expand contractions, tokenize with phrase awareness
  2. Words/phrases matched in gloss.csv vocabulary -> keep directly
  3. Unmatched words -> POS filter (keep NOUN/VERB/ADJ/ADV/NUM/PRON/PROPN)

This gives higher coverage than vocabulary-only matching because content
words not in gloss.csv are still retained via POS fallback.
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
_PGE_ASL = _PROJECT_ROOT / "pseudo-gloss-English" / "asl_gloss_seprate"

if str(_PGE_ASL) not in sys.path:
    sys.path.insert(0, str(_PGE_ASL))

from asl_gloss_extract import GlossVocab, expand_contractions, STOP_WORDS  # noqa: E402

_DEFAULT_GLOSS_CSV = _PROJECT_ROOT / "data" / "gloss.csv"

# POS tags to keep for unmatched words (same as combined_pipeline.py)
_SELECTED_POS = {"NOUN", "NUM", "ADV", "PRON", "PROPN", "ADJ", "VERB"}


def _extract_sentence_glosses(
    sent: str, vocab_db: GlossVocab, nlp,
) -> tuple[list[str], list[dict], set[str]]:
    """Extract glosses from a single sentence.

    Strategy: vocabulary match first (both phrases and single words),
    POS filter only for words not in vocabulary.

    Returns: (gloss_list, match_detail_list, unmatched_set)
    """
    expanded = expand_contractions(sent)
    tokens = vocab_db.tokenize_with_phrases(expanded)

    # Classify tokens: vocab-matched or needs POS filtering
    token_plan = []  # (type, value): ("vocab", gloss_upper) or ("word", index)
    single_words = []  # words needing POS check
    details = []
    unmatched = set()

    for token in tokens:
        token_clean = token.strip(".,!?;:\"'()[]{}—–-")
        if not token_clean:
            continue
        token_lower = token_clean.lower()
        if not token_lower or token_lower in STOP_WORDS:
            continue
        if re.fullmatch(r'\d+', token_lower):
            continue

        result = vocab_db.lookup(token_clean)
        if result:
            gloss_word = result["matched_to"].upper()
            token_plan.append(("vocab", gloss_word))
            details.append({
                "input": token_lower,
                "matched_to": result["matched_to"],
                "ref": result["ref"],
                "match_type": result["match_type"],
                "confidence": result["confidence"],
            })
        else:
            idx = len(single_words)
            single_words.append(token_clean)
            token_plan.append(("word", idx))
            unmatched.add(token_lower)

    # POS filter unmatched words using full sentence context
    pos_results = {}
    if single_words:
        doc = nlp(" ".join(single_words))
        for i, tok in enumerate(doc):
            if i < len(single_words):
                pos_results[i] = (tok.lemma_.upper(), tok.pos_)

    # Assemble final gloss list
    glosses = []
    for entry_type, value in token_plan:
        if entry_type == "vocab":
            glosses.append(value)
        else:
            idx = value
            if idx in pos_results:
                lemma, pos = pos_results[idx]
                if pos in _SELECTED_POS:
                    glosses.append(lemma)

    return glosses, details, unmatched


async def run_phase2(
    task_id: str,
    sentences: list[str],
    output_dir: Path | None = None,
    gloss_csv: Path | str | None = None,
) -> dict:
    """Extract glosses from sentences using vocabulary matching + POS filter.

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
                     f"(vocab match + POS filter)")

        vocab_db = GlossVocab(csv_path)
        nlp = spacy.load("en_core_web_sm")

        glosses = {}
        descriptions = {}
        vocab_counter = collections.Counter()
        match_details = []
        all_unmatched = set()

        for sent in sentences:
            sent_glosses, details, sent_unmatched = _extract_sentence_glosses(sent, vocab_db, nlp)
            glosses[sent] = sent_glosses
            vocab_counter.update(sent_glosses)
            match_details.extend(details)
            all_unmatched.update(sent_unmatched)

            # Collect descriptions from matched glosses
            for d in details:
                gloss_word = d["matched_to"].upper()
                if gloss_word not in descriptions:
                    result = vocab_db.lookup(d["matched_to"])
                    if result and result.get("gloss"):
                        descriptions[gloss_word] = result["gloss"]

        vocab = {
            "size": len(vocab_counter),
            "total_tokens": sum(vocab_counter.values()),
            "frequency": dict(vocab_counter.most_common()),
        }
        unmatched = sorted(all_unmatched)

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
