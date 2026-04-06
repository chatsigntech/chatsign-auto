"""Shared video processing utilities."""
import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def cut_video_at_split_points(
    video_path: Path,
    segments: list[dict],
    output_dir: Path,
    video_stem: str,
    fps_override: float | None = None,
) -> tuple[list[dict], float]:
    """Cut a video into clips based on segment boundaries.

    Each segment should have 'start' and 'end' (in seconds) and optionally 'label'.

    Returns (clips metadata list, video fps).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return [], 0.0

    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clips = []
    for idx, seg in enumerate(segments):
        start_sec = seg.get("start", 0.0)
        end_sec = seg.get("end", 0.0)
        label = seg.get("label", seg.get("text", f"seg_{idx}"))

        start_frame = int(start_sec * fps)
        end_frame = min(int(end_sec * fps), total_frames)

        if end_frame <= start_frame:
            continue

        clip_name = f"{video_stem}_seg{idx:03d}_{label}.mp4"
        clip_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in clip_name)
        clip_path = output_dir / clip_name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()

        clips.append({
            "filename": clip_name,
            "source_video": video_path.name,
            "segment_index": idx,
            "label": label,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "fps": fps,
        })

    cap.release()
    return clips, fps


def make_gpu_env(gpu_id: int, **extra) -> dict:
    """Build environment dict with CUDA_VISIBLE_DEVICES set."""
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["WANDB_MODE"] = "disabled"
    env.update(extra)
    return env
