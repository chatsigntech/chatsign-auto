#!/usr/bin/env python3
"""Append pending-videos.jsonl entries for a render batch (idempotent).

Complements the chatsign-accuracy admin bulk-upload flow for the case where the
videos are already on disk under data/review/generated/<batch-tag>/. Safe to
re-run: existing videoIds are skipped.

Usage (defaults are set for the 26commencement-02-render batch):
    python3 bin/seed_render_batch.py \
        [--videos-dir PATH] [--texts-file PATH] [--pending-path PATH] \
        [--batch-tag STR] [--translator-id STR] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULTS = {
    "videos_dir": REPO / "chatsign-accuracy/backend/data/review/generated/26commencement-02-render",
    "texts_file": REPO / "chatsign-accuracy/backend/data/texts/26commencement-02.jsonl",
    "pending_path": REPO / "chatsign-accuracy/backend/data/reports/pending-videos.jsonl",
    "batch_tag": "26commencement-02-render",
    "batch_file": "26commencement-02.jsonl",
    "translator_id": "render",
    "language": "en",
    "video_id_prefix": "render_26c02",
}

# Matches 26commencement-02_<sid>_<lang>_<tokens>_<start>_<end>_hiya.mp4 and similar.
FILENAME_RE = re.compile(r"^26commencement-02_(\d+)_[a-z]+_\d+_\d+_\d+.*\.mp4$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--videos-dir", type=Path, default=DEFAULTS["videos_dir"])
    p.add_argument("--texts-file", type=Path, default=DEFAULTS["texts_file"])
    p.add_argument("--pending-path", type=Path, default=DEFAULTS["pending_path"])
    p.add_argument("--batch-tag", default=DEFAULTS["batch_tag"],
                   help="source label in pending-videos entries (admin UI uses this)")
    p.add_argument("--batch-file", default=DEFAULTS["batch_file"],
                   help="texts/*.jsonl that sentenceIds belong to")
    p.add_argument("--translator-id", default=DEFAULTS["translator_id"],
                   help="translatorId value — keep distinct from reviewer ids to avoid self-review block")
    p.add_argument("--language", default=DEFAULTS["language"])
    p.add_argument("--video-id-prefix", default=DEFAULTS["video_id_prefix"])
    p.add_argument("--dry-run", action="store_true", help="print plan, do not modify files")
    return p.parse_args()


def load_text_map(texts_file: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with texts_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[int(obj["id"])] = obj["text"]
    return out


def load_existing_video_ids(pending_path: Path) -> set[str]:
    existing: set[str] = set()
    if not pending_path.exists():
        return existing
    with pending_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = obj.get("videoId")
            if vid:
                existing.add(vid)
    return existing


def build_entry(args: argparse.Namespace, sid: int, filename: str, text: str, now_iso: str) -> dict:
    return {
        "videoId": f"{args.video_id_prefix}_{sid}",
        "sentenceId": sid,
        "sentenceText": text,
        "translatorId": args.translator_id,
        "language": args.language,
        "videoPath": f"review/generated/{args.batch_tag}/{filename}",
        "videoFileName": filename,
        "source": args.batch_tag,
        "addedAt": now_iso,
        "batchFile": args.batch_file,
        "localPath": f"{args.batch_tag}/{filename}",
    }


def main() -> int:
    args = parse_args()

    # Preconditions
    if not args.videos_dir.is_dir():
        print(f"ERROR: videos_dir not found: {args.videos_dir}", file=sys.stderr)
        return 2
    if not args.texts_file.is_file():
        print(f"ERROR: texts_file not found: {args.texts_file}", file=sys.stderr)
        return 2
    args.pending_path.parent.mkdir(parents=True, exist_ok=True)

    text_map = load_text_map(args.texts_file)
    existing_ids = load_existing_video_ids(args.pending_path)
    now_iso = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")

    mp4_files = sorted(p.name for p in args.videos_dir.glob("*.mp4"))
    if not mp4_files:
        print(f"ERROR: no *.mp4 under {args.videos_dir}", file=sys.stderr)
        return 2

    to_append: list[dict] = []
    skipped_existing = 0
    unmatched: list[str] = []
    missing_text: list[tuple[int, str]] = []

    for fn in mp4_files:
        m = FILENAME_RE.match(fn)
        if not m:
            unmatched.append(fn)
            continue
        sid = int(m.group(1))
        text = text_map.get(sid)
        if text is None:
            missing_text.append((sid, fn))
            continue
        video_id = f"{args.video_id_prefix}_{sid}"
        if video_id in existing_ids:
            skipped_existing += 1
            continue
        to_append.append(build_entry(args, sid, fn, text, now_iso))

    print(f"mp4 files found:     {len(mp4_files)}")
    print(f"unmatched filenames: {len(unmatched)}")
    print(f"missing sentence id: {len(missing_text)}")
    print(f"skipped (existing):  {skipped_existing}")
    print(f"to append:           {len(to_append)}")
    if unmatched:
        print("  unmatched sample:", unmatched[:3])
    if missing_text:
        print("  missing sample:", missing_text[:3])

    if unmatched or missing_text:
        print("ERROR: refusing to append while inputs are inconsistent", file=sys.stderr)
        return 3

    if args.dry_run:
        print("dry-run — no file writes")
        if to_append:
            print("first entry preview:")
            print(json.dumps(to_append[0], ensure_ascii=False, indent=2))
        return 0

    if not to_append:
        print("nothing to do")
        return 0

    # Append atomically — write to temp sibling then rename, so a kill mid-write
    # never leaves a half-line stuck in pending-videos.jsonl.
    tmp = args.pending_path.with_suffix(args.pending_path.suffix + ".seed-tmp")
    with tmp.open("w", encoding="utf-8") as out:
        if args.pending_path.exists():
            with args.pending_path.open(encoding="utf-8") as src:
                for line in src:
                    out.write(line if line.endswith("\n") else line + "\n")
        for entry in to_append:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    tmp.replace(args.pending_path)

    print(f"appended {len(to_append)} entries to {args.pending_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
