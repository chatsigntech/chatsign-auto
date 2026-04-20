"""Phase 5: Segment original sentence videos using Phase 4 trained model.

Steps:
  5.1  Run segmentation inference on Phase 2 sentence videos
  5.2  Cut sentence videos into word-level video clips at split points
  5.3  Record split point data for Phase 7 reuse

Input:  Phase 4 checkpoint + config + Phase 2 sentence videos
Output: segment_videos/ (word-level mp4 clips) + split_points.json
"""
import json
import logging
import sys
from pathlib import Path

from backend.config import settings
from backend.core.subprocess_runner import run_subprocess
from backend.core.video_utils import cut_video_at_split_points, make_gpu_env

logger = logging.getLogger(__name__)

SPAMO_ROOT = Path(settings.SPAMO_SEGMENT_PATH).resolve()
SPAMO_PYTHON = sys.executable


async def _run_segmentation(
    task_id: str,
    ckpt_path: Path,
    config_path: Path,
    output_dir: Path,
    gpu_id: int = 0,
) -> Path:
    """Step 5.1: Run segmentation inference on all videos."""
    seg_output = output_dir / "segmentation"
    seg_output.mkdir(parents=True, exist_ok=True)

    # Pass explicitly so our behavior is pinned even if upstream argparse defaults shift.
    cmd = [
        SPAMO_PYTHON, str(SPAMO_ROOT / "scripts" / "segment_alignment.py"),
        "--ckpt", str(ckpt_path),
        "--config", str(config_path),
        "--mode", "test",
        "--num_samples", "0",
        "--min_word_frames", "8",
        "--max_word_frames", "75",
        "--output_dir", str(seg_output),
    ]

    logger.info(f"[{task_id}] Step 5.1: Running segmentation inference")
    rc, stdout, stderr = await run_subprocess(
        cmd, cwd=SPAMO_ROOT, env=make_gpu_env(gpu_id), log_to_file=True
    )
    if rc != 0:
        raise RuntimeError(f"Segmentation failed (rc={rc}): {(stderr or stdout)[-500:]}")

    result_file = seg_output / "segmentation_log.json"
    if not result_file.exists():
        raise RuntimeError(f"Segmentation output not found: {result_file}")

    logger.info(f"[{task_id}] Step 5.1: Segmentation results -> {result_file}")
    return result_file


async def run_phase5_segment(
    task_id: str,
    phase4_output: Path,
    phase2_output: Path,
    output_dir: Path,
    gpu_id: int = 0,
) -> dict:
    """Run segmentation inference and cut videos into word-level clips."""
    output_dir = output_dir.resolve()
    phase4_output = phase4_output.resolve()
    phase2_output = phase2_output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate Phase 4 outputs
    ckpt_path = phase4_output / "segmentation_model.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Phase 4 checkpoint not found: {ckpt_path}")

    # Find config file
    config_files = list(phase4_output.glob("config_*.yaml"))
    if not config_files:
        raise FileNotFoundError(f"Phase 4 config not found in {phase4_output}")
    config_path = config_files[0]

    # Step 5.1: Run segmentation inference
    seg_result_path = await _run_segmentation(
        task_id, ckpt_path, config_path, output_dir, gpu_id
    )

    with open(seg_result_path) as f:
        seg_results = json.load(f)

    # Step 5.2 & 5.3: Cut videos and record split points
    segment_videos_dir = output_dir / "segment_videos"
    segment_videos_dir.mkdir(parents=True, exist_ok=True)

    video_dir = phase2_output / "videos"
    split_points = {}
    total_clips = 0
    total_segments = 0

    for result in seg_results:
        video_name = result.get("video_name", result.get("fileid", ""))
        segments = result.get("segments", [])

        if not segments or not video_name:
            continue

        # Skip word videos — they are already word-level, segmenting them is redundant
        if video_name.startswith("word_"):
            continue

        total_segments += len(segments)

        # Find the source video
        video_path = video_dir / video_name
        if not video_path.exists():
            video_path = video_dir / f"{video_name}.mp4"
        if not video_path.exists():
            logger.warning(f"[{task_id}] Source video not found: {video_name}")
            continue

        video_stem = video_path.stem

        # Get video fps for frame→seconds conversion
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        # Convert SpaMo's frame-based segments to seconds-based format
        # SpaMo outputs: orig_start/orig_end (frame numbers), token (label)
        converted_segments = []
        for seg in segments:
            converted_segments.append({
                "start": seg.get("orig_start", 0) / fps,
                "end": seg.get("orig_end", 0) / fps,
                "label": seg.get("token", ""),
            })

        # Cut video into clips
        clips, _ = cut_video_at_split_points(
            video_path, converted_segments, segment_videos_dir, video_stem
        )
        total_clips += len(clips)

        # Record split points with both formats (for Phase 7 reuse)
        split_points[video_stem] = {
            "video_name": video_path.name,
            "fps": fps,
            "segments": converted_segments,
            "raw_segments": segments,
        }

    # Save split points for Phase 7
    split_points_path = output_dir / "split_points.json"
    with open(split_points_path, "w") as f:
        json.dump(split_points, f, indent=2, ensure_ascii=False)

    logger.info(
        f"[{task_id}] Phase 5 completed: {len(seg_results)} videos segmented, "
        f"{total_segments} segments found, {total_clips} clips generated"
    )

    return {
        "input_videos": len(seg_results),
        "segmented_videos": len(split_points),
        "total_segments": total_segments,
        "total_clips": total_clips,
    }
