"""Wrap trim_inactive_frames.py (frames-dir flow) with cv2 extract/reassemble
so it works on a single mp4 in/out. Used by run_pipeline_local.sh stage 1.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

THIS = Path(__file__).resolve().parent
UNISIGN = Path("/home/chatsign/lizh/chatsign-auto/UniSignMimicTurbo")
TRIM = UNISIGN / "scripts" / "sentence" / "trim_inactive_frames.py"


def extract_frames(video: Path, out_dir: Path) -> tuple[float, tuple[int, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        cv2.imwrite(str(out_dir / f"{i:06d}_x.jpg"), f)
    cap.release()
    return fps, (w, h)


def reassemble(frames_dir: Path, out_video: Path, fps: float, size: tuple[int, int]):
    paths = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.name.split("_")[0]))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, size)
    for p in paths:
        writer.write(cv2.imread(str(p)))
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--work", required=True, help="scratch dir")
    args = ap.parse_args()

    work = Path(args.work)
    src = Path(args.input)
    base = src.stem
    raw = work / "raw" / base
    trim = work / "trim"

    fps, size = extract_frames(src, raw)
    print(f"[trim] extracted {sum(1 for _ in raw.iterdir())} frames @ {fps:.1f}fps {size}")

    # trim_inactive_frames.py expects --frames-dir = parent of one-or-more subdirs.
    cmd = [
        sys.executable, str(TRIM),
        "--frames-dir", str(work / "raw"),
        "--output-dir", str(trim),
        "--device", "cuda",
        "--mode", "balanced",
    ]
    subprocess.run(cmd, check=True)

    out_frames = trim / base
    if not out_frames.exists() or not any(out_frames.iterdir()):
        # No active region detected — pass through original.
        shutil.copy2(src, args.output)
        print(f"[trim] no active region; passthrough -> {args.output}")
        return

    reassemble(out_frames, Path(args.output), fps, size)
    print(f"[trim] wrote {args.output}")


if __name__ == "__main__":
    main()
