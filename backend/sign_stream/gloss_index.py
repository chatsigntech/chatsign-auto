"""Gloss → mp4 path index.

Build order (later sources override earlier ones; highest-priority last):
  1. ASL-27K base vocabulary
  2. accuracy raw translator recordings (source=submission) — phase 2 outputs
  3. accuracy admin-uploaded reference videos (source=generated)
  4. shared task phase-3 DGX-converted outputs (data/shared/<task>/phase_3/...)
  5. DGX letters dir — the 26 a-z letter clips (highest, used for single-char glosses)

Out-of-vocabulary multi-char glosses are spelled out using the letter clips.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from backend.api.accuracy import REPORTS_DIR
from backend.config import settings
from backend.core.io_utils import read_jsonl
from backend.core.sign_video_generator import _load_asl27k_index, scan_phase3_videos

logger = logging.getLogger(__name__)

SOURCE_ASL27K = "asl27k"
SOURCE_SUBMISSION = "submission"
SOURCE_GENERATED = "generated"
SOURCE_PHASE3 = "phase3"
SOURCE_LETTERS = "letters"

_LETTERS_DIR = settings.SIGN_VIDEO_OUTPUT_DIR / "letters"
_ACCURACY_UPLOADS_ROOT = settings.CHATSIGN_ACCURACY_DATA
_ACCURACY_GENERATED = _ACCURACY_UPLOADS_ROOT / "review" / "generated"
_PENDING_VIDEOS = REPORTS_DIR / "pending-videos.jsonl"


@dataclass
class GlossClip:
    gloss: str
    paths: list[Path]
    source: str


_lock = Lock()
_index: dict[str, tuple[Path, str]] | None = None
_letters_cache: dict[str, Path] | None = None


def _norm(s: str) -> str:
    """Normalize for index lookup. Pipeline emits 'GOOD_MORNING'; gloss.csv stores 'good morning'."""
    return s.strip().lower().replace("_", " ")


def _load_asl27k(idx: dict[str, tuple[Path, str]]) -> int:
    n = 0
    for upper, path in _load_asl27k_index().items():
        idx.setdefault(_norm(upper), (path, SOURCE_ASL27K))
        n += 1
    return n


def _load_accuracy_pending(idx: dict[str, tuple[Path, str]]) -> tuple[int, int]:
    sub_n = gen_n = 0
    for e in read_jsonl(_PENDING_VIDEOS):
        text = _norm(e.get("sentenceText") or "")
        src = e.get("source")
        if not text:
            continue
        if src == SOURCE_SUBMISSION:
            rel = (e.get("videoPath") or "").lstrip("/")
            if not rel.startswith("uploads/"):
                continue
            mp4 = _ACCURACY_UPLOADS_ROOT / rel
            if mp4.exists():
                idx[text] = (mp4, SOURCE_SUBMISSION)
                sub_n += 1
        elif src == SOURCE_GENERATED:
            local = e.get("localPath") or ""
            mp4 = _ACCURACY_GENERATED / local if local else None
            if mp4 and mp4.exists():
                idx[text] = (mp4, SOURCE_GENERATED)
                gen_n += 1
    return sub_n, gen_n


def _build_letters_cache() -> dict[str, Path]:
    cache: dict[str, Path] = {}
    if not _LETTERS_DIR.exists():
        return cache
    for p in _LETTERS_DIR.glob("*.mp4"):
        stem = p.stem.lower()
        if len(stem) == 1 and stem.isalpha():
            cache[stem] = p
    return cache


def _load_phase3_shared(idx: dict[str, tuple[Path, str]]) -> int:
    """Phase 3 DGX-converted task outputs (data/shared/<task>/phase_3/...)."""
    n = 0
    for upper, path in scan_phase3_videos().items():
        idx[_norm(upper)] = (path, SOURCE_PHASE3)
        n += 1
    return n


def _load_phase3_letters(idx: dict[str, tuple[Path, str]], letters: dict[str, Path]) -> int:
    """The 26 DGX-processed letter clips override raw accuracy recordings of single letters."""
    n = 0
    for ch, p in letters.items():
        idx[ch] = (p, SOURCE_PHASE3)
        n += 1
    return n


def _build() -> dict[str, tuple[Path, str]]:
    idx: dict[str, tuple[Path, str]] = {}
    base = _load_asl27k(idx)
    sub, gen = _load_accuracy_pending(idx)
    letters = _build_letters_cache()
    p3_shared = _load_phase3_shared(idx)
    p3_letters = _load_phase3_letters(idx, letters)
    logger.info(
        "gloss index built: asl27k=%d, submissions=%d, generated=%d, "
        "phase3_shared=%d, phase3_letters=%d, total=%d",
        base, sub, gen, p3_shared, p3_letters, len(idx),
    )
    return idx


def get_index() -> dict[str, tuple[Path, str]]:
    global _index, _letters_cache
    if _index is not None:
        return _index
    with _lock:
        if _index is None:
            _letters_cache = _build_letters_cache()
            _index = _build()
    return _index


def reload():
    global _index, _letters_cache
    with _lock:
        _index = None
        _letters_cache = None
    return get_index()


def _letter_clips(token: str) -> list[Path]:
    cache = _letters_cache or {}
    return [cache[ch] for ch in token.lower() if ch in cache]


def resolve(gloss: str) -> GlossClip:
    """Resolve a single gloss to clip(s). Letter fallback on miss."""
    idx = get_index()
    key = _norm(gloss)
    if key in idx:
        path, src = idx[key]
        return GlossClip(gloss=gloss, paths=[path], source=src)
    return GlossClip(gloss=gloss, paths=_letter_clips(gloss), source=SOURCE_LETTERS)


def resolve_many(glosses: list[str]) -> list[GlossClip]:
    return [resolve(g) for g in glosses]


def build_plan(glosses: list[str]) -> tuple[list[dict], list[Path]]:
    """Resolve glosses to a UI plan + flat clip list. Used by both preview and ws."""
    plan: list[dict] = []
    flat: list[Path] = []
    for g in glosses:
        c = resolve(g)
        plan.append({"gloss": g, "source": c.source, "n_clips": len(c.paths),
                     "paths": [p.name for p in c.paths]})
        flat.extend(c.paths)
    return plan, flat
