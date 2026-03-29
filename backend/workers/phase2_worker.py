"""Phase 2: Pseudo-gloss extraction using spaCy (inline subprocess).

Extracts content words from English sentences as uppercase pseudo-glosses,
filtering out function words (articles, prepositions, conjunctions, etc.).
Includes per-sentence error handling and vocabulary frequency statistics.
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

from backend.config import settings

logger = logging.getLogger(__name__)

# Subprocess script: per-sentence try/catch + vocab stats
INLINE_GLOSS_SCRIPT = '''
import spacy, json, sys, collections

nlp = spacy.load("en_core_web_sm")
sentences = json.loads(sys.argv[1])
content_pos = {"NOUN", "VERB", "ADJ", "ADV", "NUM", "PRON", "PROPN"}

glosses = {}
errors = []
vocab_counter = collections.Counter()

for sent in sentences:
    try:
        doc = nlp(sent)
        sent_glosses = [token.lemma_.upper() for token in doc if token.pos_ in content_pos]
        glosses[sent] = sent_glosses
        vocab_counter.update(sent_glosses)
    except Exception as e:
        glosses[sent] = []
        errors.append({"sentence": sent, "error": str(e)})

output = {
    "glosses": glosses,
    "errors": errors,
    "vocab": {
        "size": len(vocab_counter),
        "total_tokens": sum(vocab_counter.values()),
        "frequency": dict(vocab_counter.most_common()),
    },
}
print(json.dumps(output))
'''


async def run_phase2(task_id: str, sentences: list[str], output_dir: Path | None = None) -> dict:
    """Extract pseudo-glosses from English sentences.

    Args:
        task_id: Pipeline task ID
        sentences: List of English sentences
        output_dir: If provided, write glosses.json and vocab.json

    Returns:
        dict mapping sentence -> list of glosses
    """
    if not sentences:
        logger.warning(f"[{task_id}] Phase 2: No sentences to process")
        glosses = {}
        vocab = {"size": 0, "total_tokens": 0, "frequency": {}}
        errors = []
    else:
        logger.info(f"[{task_id}] Phase 2: processing {len(sentences)} sentences")
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", INLINE_GLOSS_SCRIPT, json.dumps(sentences),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Phase 2 failed: {stderr.decode()}")

        output = json.loads(stdout.decode())
        glosses = output["glosses"]
        vocab = output["vocab"]
        errors = output["errors"]

        if errors:
            logger.warning(f"[{task_id}] Phase 2: {len(errors)} sentences failed, skipped")
            for e in errors:
                logger.warning(f"[{task_id}]   {e['sentence'][:60]}... → {e['error']}")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "glosses.json", "w", encoding="utf-8") as f:
            json.dump(glosses, f, ensure_ascii=False, indent=2)
        with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

    logger.info(f"[{task_id}] Phase 2 completed: {len(glosses)} glosses, "
                f"vocab size {vocab['size']}, {len(errors)} errors")
    return glosses
