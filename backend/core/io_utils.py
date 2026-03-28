"""Shared I/O utilities."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skipping malformed lines with a warning."""
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSONL line %d in %s: %s", lineno, path, e)
    return entries
