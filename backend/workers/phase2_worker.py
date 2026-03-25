"""Phase 2: Pseudo-gloss extraction using spaCy (inline subprocess)."""
import asyncio
import json
import logging
import sys
from pathlib import Path

from backend.config import settings

logger = logging.getLogger(__name__)

INLINE_GLOSS_SCRIPT = '''
import spacy, json, sys

nlp = spacy.load("en_core_web_sm")
sentences = json.loads(sys.argv[1])
content_pos = {"NOUN", "VERB", "ADJ", "ADV", "NUM", "PRON", "PROPN"}

result = {}
for sent in sentences:
    doc = nlp(sent)
    glosses = [token.lemma_.upper() for token in doc if token.pos_ in content_pos]
    result[sent] = glosses

print(json.dumps(result))
'''


async def run_phase2(task_id: str, sentences: list[str]) -> dict:
    """Extract pseudo-glosses from English sentences."""
    logger.info(f"[{task_id}] Phase 2: processing {len(sentences)} sentences")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", INLINE_GLOSS_SCRIPT, json.dumps(sentences),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Phase 2 failed: {stderr.decode()}")
    return json.loads(stdout.decode())
