"""Phase 2: Vocabulary-based gloss extraction using gloss.csv.

Replaces the previous spaCy POS-filter approach with multi-level
vocabulary matching against the sign language video corpus (gloss.csv).

Each extracted gloss is guaranteed to have a corresponding video in the
corpus, ensuring downstream phases only work with actionable vocabulary.

Matching strategy (4 levels, best-first):
  1. Exact match          (confidence 0.95)
  2. Lemma match          (confidence 0.90)  — input lemmatized
  3. Double-lemma match   (confidence 0.85)  — both sides lemmatized
  4. Semantic search       (confidence ≥ 0.70) — cosine similarity (optional)
"""
import asyncio
import collections
import csv
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default path to gloss.csv (can be overridden via argument)
# ---------------------------------------------------------------------------
_DEFAULT_GLOSS_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "gloss.csv"

# ---------------------------------------------------------------------------
# Contraction expansion (from gloss_matcher)
# ---------------------------------------------------------------------------
_CONTRACTIONS = {
    "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
    "'d": " would", "'m": " am", "'s": " is",
}
_CONTRACTION_RE = re.compile(
    "(" + "|".join(re.escape(k) for k in sorted(_CONTRACTIONS, key=len, reverse=True)) + ")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Stop words — these rarely have sign language equivalents
# ---------------------------------------------------------------------------
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "shall", "may", "might", "can", "must",
    "i", "me", "my", "mine", "we", "us", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs",
    "this", "that", "these", "those",
    "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "up", "as",
    "into", "about", "between", "through", "after", "before",
    "not", "no", "if", "then", "than", "when", "while",
    "who", "whom", "which", "what", "where", "how",
    "all", "each", "every", "both", "few", "more", "most",
    "some", "any", "such", "only", "own", "same", "too", "very",
    "just", "also", "now", "here", "there",
}


def _normalize_apostrophes(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u2018", "'")


def _expand_contractions(text: str) -> str:
    text = _normalize_apostrophes(text)
    return _CONTRACTION_RE.sub(lambda m: _CONTRACTIONS[m.group(0).lower()], text)


# ---------------------------------------------------------------------------
# GlossVocab — loads gloss.csv and provides multi-level lookup
# ---------------------------------------------------------------------------
class GlossVocab:
    """Sign language vocabulary loaded from gloss.csv."""

    def __init__(self, csv_path: Path | str):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {csv_path}")

        # word (lowercase) → list of {ref, gloss, alternate_words, ...}
        self.word_to_entries: dict[str, list[dict]] = {}
        # All multi-word phrases (lowercase), sorted longest first
        self.phrases: list[str] = []
        # Lemmatized word → original word mapping
        self._lemma_to_words: dict[str, set[str]] = {}
        # spaCy model (lazy loaded)
        self._nlp = None

        self._load_csv(csv_path)
        self._build_lemma_index()

    def _load_csv(self, csv_path: Path):
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = (row.get("word") or "").strip()
                if not word:
                    continue
                word_lower = word.lower()
                entry = {
                    "ref": row.get("ref", ""),
                    "word": word,
                    "sourceid": row.get("sourceid", ""),
                    "synset_id": row.get("synset_id", ""),
                    "gloss": row.get("gloss", ""),
                    "alternate_words": row.get("alternate_words", ""),
                }
                self.word_to_entries.setdefault(word_lower, []).append(entry)

        # Collect multi-word phrases (2+ words), sorted longest first
        self.phrases = sorted(
            [w for w in self.word_to_entries if " " in w],
            key=len, reverse=True,
        )

        logger.info(f"GlossVocab loaded: {len(self.word_to_entries)} entries, "
                     f"{len(self.phrases)} multi-word phrases")

    def _get_nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def _lemmatize(self, word: str) -> str:
        nlp = self._get_nlp()
        doc = nlp(word)
        return doc[0].lemma_.lower() if doc else word.lower()

    def _build_lemma_index(self):
        """Build lemma → original words index for double-lemma matching."""
        nlp = self._get_nlp()
        for word_lower in list(self.word_to_entries.keys()):
            if " " in word_lower:
                continue  # skip phrases for lemma index
            doc = nlp(word_lower)
            if doc:
                lemma = doc[0].lemma_.lower()
                if lemma != word_lower:
                    self._lemma_to_words.setdefault(lemma, set()).add(word_lower)

    def tokenize_with_phrases(self, text: str) -> list[str]:
        """Tokenize text, recognizing multi-word phrases from the vocabulary.

        Longer phrases are matched first (greedy), remaining text is split
        on whitespace.
        """
        text_lower = text.lower()
        # Mark phrase positions
        used = [False] * len(text_lower)
        tokens = []

        for phrase in self.phrases:
            start = 0
            while True:
                idx = text_lower.find(phrase, start)
                if idx == -1:
                    break
                # Check word boundaries
                before_ok = (idx == 0 or not text_lower[idx - 1].isalnum())
                end_idx = idx + len(phrase)
                after_ok = (end_idx >= len(text_lower) or not text_lower[end_idx].isalnum())
                if before_ok and after_ok and not any(used[idx:end_idx]):
                    tokens.append(text[idx:end_idx])
                    for i in range(idx, end_idx):
                        used[i] = True
                start = idx + 1

        # Remaining: split on whitespace, filter out already-used chars
        remaining = []
        current = []
        for i, ch in enumerate(text):
            if used[i]:
                if current:
                    remaining.append("".join(current))
                    current = []
            elif ch.isspace():
                if current:
                    remaining.append("".join(current))
                    current = []
            else:
                current.append(ch)
        if current:
            remaining.append("".join(current))

        tokens.extend(remaining)

        # Clean: strip punctuation from edges
        cleaned = []
        for t in tokens:
            t = t.strip(".,!?;:\"'()[]{}—–-")
            if t:
                cleaned.append(t)

        return cleaned

    def lookup(self, word: str) -> dict | None:
        """Multi-level lookup for a single word/phrase.

        Returns dict with: ref, word, gloss, match_type, confidence, matched_to
        or None if no match found.
        """
        word_lower = word.lower().strip()
        if not word_lower:
            return None

        # Level 1: Exact match
        entries = self.word_to_entries.get(word_lower)
        if entries:
            e = entries[0]
            return {
                "ref": e["ref"], "word": word_lower,
                "gloss": e["gloss"], "alternate_words": e["alternate_words"],
                "match_type": "exact", "confidence": 0.95,
                "matched_to": e["word"],
            }

        # Level 2: Lemma match (lemmatize input, look up in vocab)
        input_lemma = self._lemmatize(word_lower)
        if input_lemma != word_lower:
            entries = self.word_to_entries.get(input_lemma)
            if entries:
                e = entries[0]
                return {
                    "ref": e["ref"], "word": word_lower,
                    "gloss": e["gloss"], "alternate_words": e["alternate_words"],
                    "match_type": "lemma", "confidence": 0.90,
                    "matched_to": e["word"],
                }

        # Level 3: Double-lemma match (lemmatize input, match against lemmatized vocab)
        vocab_words = self._lemma_to_words.get(input_lemma, set())
        for vw in vocab_words:
            entries = self.word_to_entries.get(vw)
            if entries:
                e = entries[0]
                return {
                    "ref": e["ref"], "word": word_lower,
                    "gloss": e["gloss"], "alternate_words": e["alternate_words"],
                    "match_type": "lemma_lemma", "confidence": 0.85,
                    "matched_to": e["word"],
                }

        return None


# ---------------------------------------------------------------------------
# Module-level vocab cache (loaded once per process)
# ---------------------------------------------------------------------------
_vocab_cache: GlossVocab | None = None


def _get_vocab(csv_path: Path | str | None = None) -> GlossVocab:
    global _vocab_cache
    if _vocab_cache is None:
        path = Path(csv_path) if csv_path else _DEFAULT_GLOSS_CSV
        _vocab_cache = GlossVocab(path)
    return _vocab_cache


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def run_phase2(
    task_id: str,
    sentences: list[str],
    output_dir: Path | None = None,
    gloss_csv: Path | str | None = None,
) -> dict:
    """Extract glosses from sentences by matching against gloss.csv vocabulary.

    Args:
        task_id: Pipeline task ID
        sentences: List of English sentences
        output_dir: If provided, write glosses.json, descriptions.json, vocab.json
        gloss_csv: Path to gloss.csv (default: data/tmp/gloss.csv)

    Returns:
        dict mapping sentence → list of matched glosses (uppercase)
    """
    if not sentences:
        logger.warning(f"[{task_id}] Phase 2: No sentences to process")
        glosses = {}
        descriptions = {}
        vocab = {"size": 0, "total_tokens": 0, "frequency": {}}
        match_details = []
    else:
        logger.info(f"[{task_id}] Phase 2: processing {len(sentences)} sentences")

        vocab_db = _get_vocab(gloss_csv)
        glosses = {}
        descriptions = {}
        vocab_counter = collections.Counter()
        match_details = []

        for sent in sentences:
            # Expand contractions: didn't → did not
            expanded = _expand_contractions(sent)
            # Tokenize with phrase awareness
            tokens = vocab_db.tokenize_with_phrases(expanded)

            sent_glosses = []
            for token in tokens:
                token_lower = token.lower().strip()
                # Skip stop words and pure numbers
                if not token_lower or token_lower in STOP_WORDS:
                    continue
                if re.fullmatch(r'\d+', token_lower):
                    continue

                result = vocab_db.lookup(token)
                if result:
                    gloss_word = result["matched_to"].upper()
                    sent_glosses.append(gloss_word)

                    # Record description (from gloss.csv gloss field)
                    if gloss_word not in descriptions and result.get("gloss"):
                        descriptions[gloss_word] = result["gloss"]

                    # Track match details
                    match_details.append({
                        "input": token_lower,
                        "matched_to": result["matched_to"],
                        "ref": result["ref"],
                        "match_type": result["match_type"],
                        "confidence": result["confidence"],
                    })

            glosses[sent] = sent_glosses
            vocab_counter.update(sent_glosses)

        vocab = {
            "size": len(vocab_counter),
            "total_tokens": sum(vocab_counter.values()),
            "frequency": dict(vocab_counter.most_common()),
        }

        unmatched_tokens = set()
        for sent in sentences:
            expanded = _expand_contractions(sent)
            tokens = vocab_db.tokenize_with_phrases(expanded)
            for token in tokens:
                tl = token.lower().strip()
                if not tl or tl in STOP_WORDS or re.fullmatch(r'\d+', tl):
                    continue
                if not vocab_db.lookup(token):
                    unmatched_tokens.add(tl)

        if unmatched_tokens:
            logger.info(f"[{task_id}] Phase 2: {len(unmatched_tokens)} unmatched tokens: "
                        f"{sorted(unmatched_tokens)[:20]}")

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
        if unmatched_tokens:
            with open(output_dir / "unmatched.json", "w", encoding="utf-8") as f:
                json.dump(sorted(unmatched_tokens), f, ensure_ascii=False, indent=2)

    logger.info(f"[{task_id}] Phase 2 completed: {len(glosses)} sentences, "
                f"{len(descriptions)} descriptions, vocab size {vocab['size']}")
    return glosses
