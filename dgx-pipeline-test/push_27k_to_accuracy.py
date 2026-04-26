"""Push ASL-27K final SR videos into chatsign-accuracy as generated/pending.

Reads a list of bases (one per line) from --bases-file or stdin, copies
web/<base>.mp4 (H.264 SR sidecar) into chatsign-accuracy/.../review/generated/
as `asl27k-<base>.mp4`, and appends a pending-videos.jsonl row.

Idempotent: existing `gen_asl27k_<base>` entries are skipped. No
review-assignments are written — 27K is too many for the existing reviewer
pool; assignment is left for a separate step.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings  # noqa: E402
from backend.core.io_utils import read_jsonl  # noqa: E402

WEB_SR = Path("/mnt/data/chatsign-auto-videos/ASL-27K-final/web")
GEN_DIR = settings.CHATSIGN_ACCURACY_DATA / "review" / "generated"
PENDING = settings.CHATSIGN_ACCURACY_DATA / "reports" / "pending-videos.jsonl"

# Reserve sentenceId 100000-127999 for the asl27k batch (50K source IDs are
# below that). Hash the base into a stable id within range.
SENT_ID_BASE = 100000


def stable_sent_id(base: str) -> int:
    return SENT_ID_BASE + (hash(base) & 0x7FFFFFFF) % 27_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases-file", help="file with one base per line; default stdin")
    args = ap.parse_args()

    if args.bases_file:
        bases = [b.strip() for b in Path(args.bases_file).read_text().splitlines() if b.strip()]
    else:
        bases = [b.strip() for b in sys.stdin.read().splitlines() if b.strip()]

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    have = {e["videoId"] for e in read_jsonl(PENDING) if e.get("videoId")}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    copied = added = skipped = 0
    with open(PENDING, "a") as fp:
        for base in bases:
            sr = WEB_SR / f"{base}.mp4"
            if not sr.exists() or sr.stat().st_size == 0:
                skipped += 1
                continue
            review_name = f"asl27k-{base}.mp4"
            dest = GEN_DIR / review_name
            if not dest.exists() or dest.stat().st_size != sr.stat().st_size:
                shutil.copy2(sr, dest)
                copied += 1

            video_id = f"gen_asl27k_{base}"
            if video_id in have:
                continue
            word = base.split("_", 1)[1] if "_" in base else base
            entry = {
                "videoId": video_id,
                "sentenceId": stable_sent_id(base),
                "sentenceText": word,
                "translatorId": "generated",
                "language": "en",
                "videoPath": f"review/generated/{review_name}",
                "videoFileName": review_name,
                "source": "generated",
                "addedAt": now,
                "localPath": review_name,
                "batchFile": "asl27k.jsonl",
                "metadata": {"origin": "asl27k"},
            }
            fp.write(json.dumps(entry, separators=(",", ":")) + "\n")
            have.add(video_id)
            added += 1

    print(f"copied={copied} pending_added={added} skipped_missing={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
