#!/usr/bin/env python3
"""Enhance gloss quality for unmatched words in gloss_match_result.csv.

Applies hand-crafted, context-aware glosses from claude_glosses.json
to unmatched words. Matched words are left completely untouched.

Usage:
    python3 scripts/gloss_matcher/enhance_glosses.py

Gloss source:
    output/claude_glosses.json — crafted by Claude based on each word's
    source_sentence context. To update, edit the JSON and re-run.
"""

from __future__ import annotations

import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "output", "gloss_match_result.csv")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "output", "gloss_match_result_enhanced.csv")
GLOSSES_JSON = os.path.join(SCRIPT_DIR, "output", "claude_glosses.json")


def enhance(input_csv: str, output_csv: str, glosses_path: str):
    """Read CSV, apply claude glosses to unmatched rows, write output."""

    with open(glosses_path, "r", encoding="utf-8") as f:
        glosses = json.load(f)
    # Remove metadata keys
    glosses.pop("_comment", None)
    print(f"Loaded {len(glosses)} glosses from {glosses_path}")

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    applied = 0
    missing = []

    for row in rows:
        if row.get("status") != "unmatched":
            continue

        word = row.get("word", "").strip().lower()
        entry = glosses.get(word)

        if entry is None:
            missing.append(word)
            continue

        row["gloss"] = entry["gloss"]
        row["alternate_words"] = entry.get("alternate_words", "")
        applied += 1

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Stats
    total = len(rows)
    unmatched = sum(1 for r in rows if r.get("status") == "unmatched")
    print(f"\n{'='*50}")
    print(f"Total rows:          {total}")
    print(f"Matched (untouched): {total - unmatched}")
    print(f"Unmatched:           {unmatched}")
    print(f"  Applied glosses:   {applied}")
    print(f"  Missing in JSON:   {len(missing)}")
    if missing:
        print(f"  Words: {', '.join(missing)}")
    print(f"Output: {output_csv}")
    print(f"{'='*50}\n")


def main():
    input_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    output_csv = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    glosses_path = sys.argv[3] if len(sys.argv) > 3 else GLOSSES_JSON

    for path, label in [(input_csv, "Input CSV"), (glosses_path, "Glosses JSON")]:
        if not os.path.exists(path):
            print(f"Error: {label} not found: {path}")
            sys.exit(1)

    print(f"Input:   {input_csv}")
    print(f"Glosses: {glosses_path}")
    print(f"Output:  {output_csv}")

    enhance(input_csv, output_csv, glosses_path)


if __name__ == "__main__":
    main()
