"""Detect tail-frame glitches in MimicMotion outputs.

Glitch signature (observed on 50K outputs):
  - body confidence drops sharply at the last 1-2 frames (~14-21%)
  - one or both hand confidences collapse (≥50% drop, often <0.25)
  - hand keypoint position can jump suddenly (out-of-pose)
  - face confidence dips slightly
  - pixel diff vs prev frame can be huge (>30%) — but not always

Uses rtmlib Wholebody (same as DGX `trim_inactive_frames.py` /
`filter_frames_by_pose.py`) so the metrics line up with the upstream pipeline.

Usage:
  python detect_tail_glitch.py <video.mp4> [<video2.mp4> ...]
  python detect_tail_glitch.py --dir <folder of mp4s>

Output: TSV at <out>/tail_glitch_report.tsv (+ a `flagged` column)
"""
import argparse
import csv
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# rtmlib parent dir: env override (DGX deployment) or default to local fork
_rtmlib_parent = os.environ.get("RTMLIB_PARENT") or str(PROJECT_ROOT / "UniSignMimicTurbo")
sys.path.insert(0, _rtmlib_parent)

import cv2
import numpy as np
from rtmlib import Wholebody

# Wholebody keypoint slices (133 total, COCO-Wholebody)
BODY = slice(0, 17)
FACE = slice(23, 91)
LHAND = slice(91, 112)
RHAND = slice(112, 133)
LWRIST_KP = 9     # body keypoint #9 = left wrist
RWRIST_KP = 10    # body keypoint #10 = right wrist

# Detection thresholds (tuned on the 4 confirmed glitches in dgx-pipeline-test)
BODY_DROP_PCT = 0.13        # body conf falls > 13% from middle-of-video mean
HAND_DROP_PCT = 0.50        # one hand conf falls > 50%
HAND_ABS_LOW = 0.30         # hand conf < 0.30 absolute is hard fail
WRIST_JUMP_PX_FRAC = 0.10   # wrist xy moves > 10% side — corroborating signal only
BOTH_WRIST_BIG_FRAC = 0.25  # require both wrists > 25% side to flag on jump alone

# Default for "frame is a glitch only if N of {body, hand, wrist} fire."
# 1 = max recall; 2 = safer / fewer false positives.
DEFAULT_MIN_CATEGORIES = 2


def read_all_frames(video_path: Path):
    """Read every frame of an mp4 with cv2. Returns (frames_list, fps, (w,h))."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps, (w, h)


def trim_frames_to_mp4(frames, fps, size, n_keep: int, out_path: Path):
    """Write the first n_keep frames to a new mp4 (mp4v codec — same as DGX
    filter / RealESR outputs; downstream stages handle codec normalization)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    try:
        for f in frames[:n_keep]:
            writer.write(f)
    finally:
        writer.release()


def per_frame_features(wb, img):
    """Return dict of per-frame features or None if no person detected."""
    kp, sc = wb(img)
    if sc is None or len(sc) == 0:
        return None
    s = sc[0] if len(sc.shape) > 1 else sc
    k = kp[0] if len(kp.shape) > 1 else kp
    return {
        "body": float(np.mean(s[BODY])),
        "face": float(np.mean(s[FACE])),
        "lhand": float(np.mean(s[LHAND])),
        "rhand": float(np.mean(s[RHAND])),
        "lwrist_xy": (float(k[LWRIST_KP][0]), float(k[LWRIST_KP][1])),
        "rwrist_xy": (float(k[RWRIST_KP][0]), float(k[RWRIST_KP][1])),
    }


def _is_glitch_vs_ref(frame, ref, min_cats: int):
    """Decide if `frame` (features dict) is a glitch given a reference dict.

    `ref` keys: body, lhand, rhand, lwrist_xy, rwrist_xy, side.
    """
    if frame is None:
        return True, ["no_pose"], None

    body_drop = (ref["body"] - frame["body"]) / ref["body"] if ref["body"] > 0 else 0
    lhand_drop = (ref["lhand"] - frame["lhand"]) / ref["lhand"] if ref["lhand"] > 0 else 0
    rhand_drop = (ref["rhand"] - frame["rhand"]) / ref["rhand"] if ref["rhand"] > 0 else 0
    lwrist_jump = math.dist(frame["lwrist_xy"], ref["lwrist_xy"])
    rwrist_jump = math.dist(frame["rwrist_xy"], ref["rwrist_xy"])

    body_fired = body_drop > BODY_DROP_PCT
    lhand_fired = lhand_drop > HAND_DROP_PCT or frame["lhand"] < HAND_ABS_LOW
    rhand_fired = rhand_drop > HAND_DROP_PCT or frame["rhand"] < HAND_ABS_LOW
    big_l = lwrist_jump > ref["side"] * WRIST_JUMP_PX_FRAC
    big_r = rwrist_jump > ref["side"] * WRIST_JUMP_PX_FRAC
    both_wrists_huge = (
        lwrist_jump > ref["side"] * BOTH_WRIST_BIG_FRAC
        and rwrist_jump > ref["side"] * BOTH_WRIST_BIG_FRAC
    )
    cat_body = body_fired
    cat_hand = (lhand_fired and rhand_fired) or (lhand_fired and big_r) or (rhand_fired and big_l)
    cat_wrist = both_wrists_huge
    n_fired = int(cat_body) + int(cat_hand) + int(cat_wrist)

    flags = []
    if body_fired: flags.append(f"body_drop={body_drop:.1%}")
    if lhand_fired: flags.append(f"lhand_lost(conf={frame['lhand']:.2f},drop={lhand_drop:.1%})")
    if rhand_fired: flags.append(f"rhand_lost(conf={frame['rhand']:.2f},drop={rhand_drop:.1%})")
    if big_l: flags.append(f"lwrist_jump={lwrist_jump:.0f}px")
    if big_r: flags.append(f"rwrist_jump={rwrist_jump:.0f}px")

    metrics = {
        "body_drop": body_drop, "lhand_drop": lhand_drop, "rhand_drop": rhand_drop,
        "lwrist_jump_px": lwrist_jump, "rwrist_jump_px": rwrist_jump,
        "n_categories": n_fired,
    }
    return n_fired >= min_cats, flags, metrics


def analyze_video(video_path: Path, wb, min_cats: int):
    """Return (frames, fps, size, feats, verdict).

    verdict includes:
      flagged       -> last frame is a glitch
      cut_at_frame  -> 1-based frame index from which to cut (keep [1, cut_at_frame)).
                       == n_total + 1 if no glitch (keep all)
      n_drop        -> number of trailing frames flagged
    """
    frames, fps, (w, h) = read_all_frames(video_path)
    side = max(w, h)
    n_total = len(frames)
    feats = [per_frame_features(wb, img) for img in frames]

    if n_total < 4:
        return frames, fps, (w, h), feats, {
            "flagged": False, "cut_at_frame": n_total + 1, "n_drop": 0,
            "reason": "too_short",
        }

    # Reference window: middle 50% of frames (skip first/last 25% to avoid both
    # mimicmotion ref-padding bursts and trailing glitches).
    lo = max(1, n_total // 4)
    hi = max(lo + 1, n_total - n_total // 4)
    mid = [f for f in feats[lo:hi] if f is not None]
    if not mid:
        return frames, fps, (w, h), feats, {
            "flagged": False, "cut_at_frame": n_total + 1, "n_drop": 0,
            "reason": "no_pose_in_middle",
        }
    ref = {
        "body":  float(np.mean([f["body"]  for f in mid])),
        "lhand": float(np.mean([f["lhand"] for f in mid])),
        "rhand": float(np.mean([f["rhand"] for f in mid])),
        "lwrist_xy": tuple(np.mean([f["lwrist_xy"] for f in mid], axis=0)),
        "rwrist_xy": tuple(np.mean([f["rwrist_xy"] for f in mid], axis=0)),
        "side": side,
    }

    # Walk backward from last frame; cut while frame is glitch vs the middle ref.
    cut_at = n_total + 1   # 1-based; keep [1, cut_at)
    last_metrics = None
    last_flags = []
    drop_indices = []
    for i in range(n_total - 1, n_total // 2, -1):  # walk back, only check tail half
        is_glitch, flags, metrics = _is_glitch_vs_ref(feats[i], ref, min_cats)
        if i == n_total - 1:
            last_metrics, last_flags = metrics, flags
        if is_glitch:
            cut_at = i + 1   # ffmpeg -frames:v counts 1-based
            drop_indices.append(i + 1)
        else:
            break

    n_drop = (n_total + 1) - cut_at if cut_at <= n_total else 0
    flagged = n_drop > 0

    verdict = {
        "flagged": flagged,
        "n_total_frames": n_total,
        "cut_at_frame": cut_at,
        "n_drop": n_drop,
        "drop_frames": ",".join(str(i) for i in sorted(drop_indices)),
        "reasons": ";".join(last_flags),
        "ref_body": ref["body"], "ref_lhand": ref["lhand"], "ref_rhand": ref["rhand"],
        "last_body": feats[-1]["body"] if feats[-1] else None,
        "last_lhand": feats[-1]["lhand"] if feats[-1] else None,
        "last_rhand": feats[-1]["rhand"] if feats[-1] else None,
    }
    if last_metrics:
        verdict.update({
            "body_drop": last_metrics["body_drop"],
            "lwrist_jump_px": last_metrics["lwrist_jump_px"],
            "rwrist_jump_px": last_metrics["rwrist_jump_px"],
        })
    return frames, fps, (w, h), feats, verdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="*", help="mp4 paths to scan")
    p.add_argument("--dir", help="alternatively scan a directory of mp4s")
    p.add_argument("--out", default=str(Path(__file__).parent / "tail_glitch_report.tsv"))
    p.add_argument("--mode", default="lightweight", choices=["lightweight","balanced","performance"])
    p.add_argument("--min-cats", type=int, default=DEFAULT_MIN_CATEGORIES,
                   help="how many of {body, hand, wrist} must fire (default 2 = safer; 1 = more sensitive)")
    p.add_argument("--apply", action="store_true",
                   help="actually write trimmed mp4s (mp4v) to --apply-dir")
    p.add_argument("--apply-dir", default=None,
                   help="output directory for trimmed mp4s (default: <video_dir>/_trimmed)")
    args = p.parse_args()

    paths: list[Path] = [Path(v) for v in args.videos]
    if args.dir:
        paths += sorted(Path(args.dir).glob("*.mp4"))
    if not paths:
        sys.exit("no inputs (provide videos or --dir)")

    print(f"loading rtmlib Wholebody (mode={args.mode}, cpu)... [min-cats={args.min_cats}]")
    wb = Wholebody(to_openpose=False, mode=args.mode, backend="onnxruntime", device="cpu")
    print("model ready.")

    rows = []
    flagged = 0
    trimmed = 0
    for v in paths:
        frames, fps, size, _feats, verdict = analyze_video(v, wb, args.min_cats)
        if verdict.get("flagged"):
            flagged += 1
        n_total = verdict.get("n_total_frames", 0)
        cut_at = verdict.get("cut_at_frame", n_total + 1)
        n_drop = verdict.get("n_drop", 0)
        msg = f"flagged={verdict.get('flagged')} keep[1,{cut_at}) drop={n_drop} reasons: {verdict.get('reasons','') or verdict.get('reason','')}"
        print(f"  {v.name}: {msg}")

        if args.apply and n_drop > 0:
            out_dir = Path(args.apply_dir) if args.apply_dir else (v.parent / "_trimmed")
            out_path = out_dir / v.name
            try:
                trim_frames_to_mp4(frames, fps, size, cut_at - 1, out_path)
                trimmed += 1
                print(f"    → wrote {out_path}")
            except Exception as e:
                print(f"    → trim FAILED: {e}")

        rows.append({
            "video": str(v),
            "flagged": verdict.get("flagged"),
            "n_total_frames": n_total,
            "cut_at_frame": cut_at,
            "n_drop": n_drop,
            "drop_frames": verdict.get("drop_frames", ""),
            "reasons": verdict.get("reasons", verdict.get("reason", "")),
            "ref_body": verdict.get("ref_body"),
            "last_body": verdict.get("last_body"),
            "body_drop": verdict.get("body_drop"),
            "ref_lhand": verdict.get("ref_lhand"),
            "last_lhand": verdict.get("last_lhand"),
            "ref_rhand": verdict.get("ref_rhand"),
            "last_rhand": verdict.get("last_rhand"),
            "lwrist_jump_px": verdict.get("lwrist_jump_px"),
            "rwrist_jump_px": verdict.get("rwrist_jump_px"),
        })

    out = Path(args.out)
    with open(out, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    print(f"\nflagged {flagged}/{len(rows)} videos. trimmed={trimmed}. report -> {out}")


if __name__ == "__main__":
    main()
