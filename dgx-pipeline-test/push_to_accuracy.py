"""Push the 10 final SR outputs into chatsign-accuracy as generated videos
and assign them to two reviewers (ay2710 + maggie).

Idempotent: re-running is safe — already-existing pending-videos rows are skipped,
and assignments are upserted by (videoId, reviewerId).
"""
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings  # noqa: E402
from backend.core.io_utils import read_jsonl  # noqa: E402

TEST_DIR = Path(__file__).resolve().parent
WEB = TEST_DIR / "web"
MANIFEST = TEST_DIR / "inputs" / "manifest.tsv"

GEN_DIR = settings.CHATSIGN_ACCURACY_DATA / "review" / "generated"
PENDING = settings.CHATSIGN_ACCURACY_DATA / "reports" / "pending-videos.jsonl"
ASSIGN = settings.CHATSIGN_ACCURACY_DATA / "reports" / "review-assignments.jsonl"

REVIEWERS = ["ay2710", "maggie"]
# Single batch timestamp so all rows from one push group together when sorted.
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def main():
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    have_videos = {e["videoId"] for e in read_jsonl(PENDING) if e.get("videoId")}
    have_assigns = {
        (e["videoId"], e["reviewerId"])
        for e in read_jsonl(ASSIGN)
        if e.get("videoId") and e.get("reviewerId")
    }

    with open(MANIFEST) as f, open(PENDING, "a") as fp, open(ASSIGN, "a") as fa:
        reader = csv.DictReader(f, delimiter="\t")
        copied = vids_added = assigns_added = 0
        for i, row in enumerate(reader):
            fname = (row.get("fname") or "").strip()
            word = (row.get("word") or "").strip()
            base = Path(fname).stem
            src = WEB / f"{base}_sr.mp4"
            if not src.exists():
                print(f"  skip (no sr): {fname}")
                continue
            video_id = f"gen_dgx_{base}"
            review_name = f"dgx-test-{base}.mp4"
            dest = GEN_DIR / review_name
            if not dest.exists():
                shutil.copy2(src, dest)
                copied += 1

            if video_id not in have_videos:
                entry = {
                    "videoId": video_id,
                    "sentenceId": 90000 + i,
                    "sentenceText": word,
                    "translatorId": "generated",
                    "language": "en",
                    "videoPath": f"review/generated/{review_name}",
                    "videoFileName": review_name,
                    "source": "generated",
                    "addedAt": NOW,
                    "localPath": review_name,
                    "metadata": {"origin": "dgx-pipeline-test"},
                }
                fp.write(json.dumps(entry, separators=(",", ":")) + "\n")
                have_videos.add(video_id)
                vids_added += 1

            for rev in REVIEWERS:
                if (video_id, rev) in have_assigns:
                    continue
                a = {
                    "videoId": video_id,
                    "reviewerId": rev,
                    "assignedBy": "dgx-pipeline-test",
                    "assignedAt": NOW,
                }
                fa.write(json.dumps(a, separators=(",", ":")) + "\n")
                have_assigns.add((video_id, rev))
                assigns_added += 1

    print(f"copied={copied}, pending-videos added={vids_added}, assignments added={assigns_added}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
