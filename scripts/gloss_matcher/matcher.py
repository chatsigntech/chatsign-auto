#!/usr/bin/env python3
"""
Gloss Batch Matcher - Match words from .docx documents against gloss.csv vocabulary

Delegates the 4-layer lookup + semantic search to chatsign-pipeline's
TextPipeline (mode='generate'); this file owns only the .docx ingest, batch
aggregation across files, and CSV output formatting.

Usage:
    python3 scripts/gloss_matcher/matcher.py file1.docx [file2.docx ...]
    python3 scripts/gloss_matcher/matcher.py ~/Desktop/*.docx

Output:
    scripts/gloss_matcher/output/gloss_match_result.csv
"""

import csv
import glob
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Contraction expansion map
# ---------------------------------------------------------------------------
CONTRACTIONS = {
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
    "'s": " is",
}

# Build a regex that matches any contraction suffix (longest first)
_CONTRACTION_RE = re.compile(
    "(" + "|".join(re.escape(k) for k in sorted(CONTRACTIONS, key=len, reverse=True)) + ")",
    re.IGNORECASE,
)

# Words to skip (stop words, punctuation-only tokens, etc.)
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
    """Normalize Unicode smart quotes to ASCII apostrophe."""
    # U+2019 RIGHT SINGLE QUOTATION MARK, U+2018 LEFT SINGLE QUOTATION MARK
    return text.replace("\u2019", "'").replace("\u2018", "'")


def expand_contractions(text: str) -> str:
    """Expand English contractions: didn't -> did not, I've -> I have, etc."""
    text = _normalize_apostrophes(text)
    return _CONTRACTION_RE.sub(lambda m: CONTRACTIONS[m.group(0).lower()], text)


def extract_sentences(text: str) -> list:
    """Split text into sentences on .!? boundaries, filtering blanks."""
    parts = re.split(r'[.!?]+', text)
    return [s.strip() for s in parts if s.strip()]


def extract_text_from_docx(filepath: str) -> str:
    """Extract all paragraph text from a .docx file."""
    from docx import Document
    doc = Document(filepath)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _normalize_unicode(text: str) -> str:
    """Normalize problematic Unicode characters to ASCII equivalents."""
    replacements = {
        "\u2019": "'",   # RIGHT SINGLE QUOTATION MARK
        "\u2018": "'",   # LEFT SINGLE QUOTATION MARK
        "\u201c": '"',   # LEFT DOUBLE QUOTATION MARK
        "\u201d": '"',   # RIGHT DOUBLE QUOTATION MARK
        "\u2014": "-",   # EM DASH
        "\u2013": "-",   # EN DASH
        "\u2026": "...", # HORIZONTAL ELLIPSIS
        "\u00a0": " ",   # NON-BREAKING SPACE
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


class GlossMatcher:
    """Batch-match words from documents against gloss.csv vocabulary."""

    def __init__(self):
        # Make `backend.config.settings` importable so we share the same paths
        # (gloss.csv, sentence-transformer model, embedding cache) as the
        # running sign-stream service. Resolving from this file: scripts/<name>/<file>
        # → repo root is two levels up.
        repo_root = Path(__file__).resolve().parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from backend.config import settings
        from chatsign_pipeline import TextPipeline

        print("[GlossMatcher] Initializing pipeline...")
        self.pipeline = TextPipeline(
            gloss_csv_path=settings.GLOSS_CSV_PATH,
            letters_dir=settings.SIGN_VIDEO_OUTPUT_DIR / "letters",
            model_dir=settings.SENTENCE_TRANSFORMER_MODEL_DIR,
            embedding_cache_dir=settings.EMBEDDING_CACHE_DIR,
            mode='generate',
        )
        print("[GlossMatcher] Pipeline ready.\n")

        # Output directory
        self.output_dir = Path(__file__).resolve().parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Threshold
        self.threshold = 0.70

        # Load spaCy once for POS tagging
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        except Exception:
            self._nlp = None

    def _generate_explanation(self, word: str) -> dict:
        """Generate gloss explanation and synonyms for an unmatched word.

        Uses the pipeline's semantic model to find similar vocabulary words,
        then builds a definition from the closest match + spaCy POS.
        Returns dict with 'gloss' and 'alternate_words'.
        """
        import numpy as np

        # Get POS tag from spaCy
        pos_label = ""
        if self._nlp:
            doc = self._nlp(word)
            if doc and len(doc) > 0:
                pos_label = doc[0].pos_.lower()

        # Use semantic model to find similar words in vocabulary
        p = self.pipeline
        if p.semantic_model is not None and p.vocab_words and p.vocab_embeddings is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            word_emb = p.semantic_model.encode([word], show_progress_bar=False)[0]
            sims = cosine_similarity([word_emb], p.vocab_embeddings)[0]
            top_indices = np.argsort(sims)[::-1][:5]

            # Closest match for definition reference
            best_idx = top_indices[0]
            best_word = p.vocab_words[best_idx]
            best_sim = sims[best_idx]

            # Get definition of closest word from gloss.csv
            refs = p.word_to_refs.get(best_word, [])
            if refs and best_sim >= 0.40:
                ref_gloss = refs[0].get("gloss", "")
                # Look up full definition from DataFrame
                ref_val = refs[0].get("ref", "")
                full_row = self._lookup_gloss_row(ref_val)
                full_def = full_row.get("gloss", "") or ref_gloss
                if pos_label:
                    gloss = f"({pos_label}) {full_def} [similar to: {best_word}]"
                else:
                    gloss = f"{full_def} [similar to: {best_word}]"
            else:
                gloss = f"({pos_label}) {word}" if pos_label else word

            # Collect top similar words as alternate_words
            alternates = []
            for idx in top_indices:
                if sims[idx] < 0.35:
                    break
                candidate = p.vocab_words[idx]
                if candidate.lower() != word.lower():
                    alternates.append(candidate)
                if len(alternates) >= 3:
                    break
            alternate_words = ", ".join(alternates)
        else:
            gloss = f"({pos_label}) {word}" if pos_label else word
            alternate_words = ""

        return {"gloss": gloss, "alternate_words": alternate_words}

    @staticmethod
    def _clean_nan(value):
        """Convert pandas NaN / 'nan' to empty string."""
        import math
        if value is None:
            return ""
        try:
            if math.isnan(value):
                return ""
        except (TypeError, ValueError):
            pass
        s = str(value)
        if s.lower() == "nan":
            return ""
        # sourceid/synset_id: drop .0 from float representation
        if s.endswith(".0") and s[:-2].isdigit():
            return s[:-2]
        return s

    def _lookup_gloss_row(self, ref_value: str) -> dict:
        """Look up the original gloss.csv row by ref value."""
        if not ref_value:
            return {}
        mask = self.pipeline.data["ref"] == ref_value
        rows = self.pipeline.data[mask]
        if rows.empty:
            return {}
        row = rows.iloc[0]
        return {
            "sourceid": self._clean_nan(row.get("sourceid", "")),
            "synset_id": self._clean_nan(row.get("synset_id", "")),
            "gloss": self._clean_nan(row.get("gloss", "")),
            "alternate_words": self._clean_nan(row.get("alternate_words", "")),
        }

    def process_files(self, file_paths: list) -> str:
        """Process multiple .docx files and write a merged CSV.

        Returns the output CSV path.
        """
        # word_key (lowercase) -> record dict
        records: dict = {}

        for fpath in file_paths:
            fname = os.path.basename(fpath)
            print(f"[Processing] {fname}")
            try:
                text = extract_text_from_docx(fpath)
            except Exception as e:
                print(f"  ERROR reading {fname}: {e}")
                continue

            sentences = extract_sentences(text)
            print(f"  Sentences: {len(sentences)}")

            for sentence in sentences:
                expanded = expand_contractions(sentence)
                tokens = self.pipeline._tokenize_with_phrases(expanded)

                for token in tokens:
                    token_lower = token.lower().strip()
                    if not token_lower or token_lower in STOP_WORDS:
                        continue
                    # Skip pure numbers
                    if re.fullmatch(r'\d+', token_lower):
                        continue

                    if token_lower in records:
                        # Already seen — bump count, track additional source files
                        records[token_lower]["count"] += 1
                        if fname not in records[token_lower]["_source_files"]:
                            records[token_lower]["_source_files"].add(fname)
                        continue

                    # First occurrence — do lookup
                    result = self.pipeline.multi_level_lookup(token, expanded)

                    if result and result.get("confidence", 0) >= self.threshold:
                        ref_val = result.get("ref", "")
                        gloss_row = self._lookup_gloss_row(ref_val)
                        records[token_lower] = {
                            "ref": ref_val,
                            "word": token_lower,
                            "sourceid": gloss_row.get("sourceid", ""),
                            "synset_id": gloss_row.get("synset_id", ""),
                            "gloss": gloss_row.get("gloss", ""),
                            "alternate_words": gloss_row.get("alternate_words", ""),
                            "status": "matched",
                            "match_type": result.get("match_type", ""),
                            "confidence": round(result["confidence"], 2),
                            "matched_to": result.get("lemma", ""),
                            "count": 1,
                            "source_sentence": _normalize_unicode(sentence.strip()),
                            "source_file": fname,
                            "_source_files": {fname},
                        }
                    else:
                        # Unmatched
                        conf = round(result["confidence"], 2) if result and result.get("confidence") else ""
                        records[token_lower] = {
                            "ref": "",
                            "word": token_lower,
                            "sourceid": "",
                            "synset_id": "",
                            "gloss": "",  # placeholder, filled later
                            "alternate_words": "",
                            "status": "unmatched",
                            "match_type": "",
                            "confidence": conf if conf else "",
                            "matched_to": "",
                            "count": 1,
                            "source_sentence": _normalize_unicode(sentence.strip()),
                            "source_file": fname,
                            "_source_files": {fname},
                        }

        # Generate gloss explanations and synonyms for unmatched words
        unmatched = [k for k, v in records.items() if v["status"] == "unmatched"]
        if unmatched:
            print(f"\n[Gloss] Generating explanations for {len(unmatched)} unmatched words...")
            for key in unmatched:
                info = self._generate_explanation(records[key]["word"])
                records[key]["gloss"] = info["gloss"]
                records[key]["alternate_words"] = info["alternate_words"]

        # Merge source_file for words appearing in multiple docs
        for rec in records.values():
            if len(rec["_source_files"]) > 1:
                rec["source_file"] = "; ".join(sorted(rec["_source_files"]))

        # Sort: matched first (alphabetical), then unmatched (alphabetical)
        sorted_records = sorted(
            records.values(),
            key=lambda r: (0 if r["status"] == "matched" else 1, r["word"]),
        )

        # Write CSV — first 6 columns match gloss.csv, then extra columns
        output_path = self.output_dir / "gloss_match_result.csv"
        fieldnames = [
            "ref", "word", "sourceid", "synset_id", "gloss", "alternate_words",
            "status", "match_type", "confidence", "matched_to",
            "count", "source_sentence", "source_file",
        ]

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted_records)

        # Summary
        total = len(sorted_records)
        matched = sum(1 for r in sorted_records if r["status"] == "matched")
        unmatched_count = total - matched
        pct = (matched / total * 100) if total else 0

        print(f"\n{'='*50}")
        print(f"  Total unique words : {total}")
        print(f"  Matched            : {matched} ({pct:.1f}%)")
        print(f"  Unmatched          : {unmatched_count}")
        print(f"  Output             : {output_path}")
        print(f"{'='*50}\n")

        return str(output_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 matcher.py <file1.docx> [file2.docx ...]")
        print("       python3 matcher.py ~/Desktop/*.docx")
        sys.exit(1)

    # Expand globs (shell usually does this, but handle it for safety)
    file_paths = []
    for arg in sys.argv[1:]:
        expanded = glob.glob(arg)
        if expanded:
            file_paths.extend(expanded)
        else:
            file_paths.append(arg)

    # Filter to .docx only
    docx_files = [f for f in file_paths if f.lower().endswith(".docx") and os.path.isfile(f)]

    if not docx_files:
        print("ERROR: No valid .docx files found.")
        print(f"  Provided: {file_paths}")
        sys.exit(1)

    print(f"[GlossMatcher] Processing {len(docx_files)} file(s):")
    for f in docx_files:
        print(f"  - {os.path.basename(f)}")
    print()

    matcher = GlossMatcher()
    matcher.process_files(docx_files)


if __name__ == "__main__":
    main()
