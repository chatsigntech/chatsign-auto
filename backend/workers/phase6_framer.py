"""Phase 6: FramerTurbo transition generation.

Two modes based on input:
- WORD-LEVEL (single word per video): Hand regression
  Each word gets rest→sign transition at start and sign→rest at end.
  Template image (ref_image) = person in natural rest pose.

- SENTENCE-LEVEL (multiple words per video): Word-to-word transitions
  Smooth interpolation between word boundaries.
  Uses extract_boundary_frames.py for multi-ref_id boundaries.

Requires FramerTurbo checkpoint (checkpoints/framer_512x320/).
If not available, falls back to passthrough.
"""
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2

from backend.config import settings
from backend.core.subprocess_runner import run_subprocess

logger = logging.getLogger(__name__)

UNISIGN = settings.UNISIGN_PATH.resolve()
FRAMER_DIR = UNISIGN / "FramerTurbo"
FRAMER_SCRIPT = FRAMER_DIR / "scripts" / "inference" / "cli_infer_576x576.py"
FRAMER_CKPT = FRAMER_DIR / "checkpoints" / "framer_512x320"
COMBINE_SCRIPT = UNISIGN / "scripts" / "sentence" / "combine_frames_and_interp.py"
BOUNDARY_SCRIPT = UNISIGN / "scripts" / "sentence" / "extract_boundary_frames.py"

# FramerTurbo adapted to work with current diffusers (0.37+)
FRAMER_PYTHON = Path(sys.executable)

# Template = ref_image used in MimicMotion (person in rest pose)
DEFAULT_TEMPLATE = UNISIGN / "assets" / "example_data" / "images" / "test2.jpg"


def _extract_frame(video_path: Path, frame_idx: int, output_path: Path):
    """Extract a single frame from a video."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(str(output_path), frame)
        return True
    return False


def _get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _detect_mode(phase5_output: Path) -> str:
    """Detect whether input is word-level or sentence-level.

    Word-level: each video dir has only 1 ref_id in frame filenames.
    Sentence-level: some video dirs have 2+ ref_ids.
    """
    frames_dir = phase5_output / "step4_resized"
    if not frames_dir.exists():
        # No frames available, check videos directly
        videos = phase5_output / "videos"
        if videos.exists():
            return "word"
        return "word"

    multi_ref = 0
    total = 0
    for d in frames_dir.iterdir():
        if not d.is_dir():
            continue
        total += 1
        ref_ids = set()
        for f in d.glob("*.jpg"):
            parts = f.stem.split("_", 1)
            if len(parts) >= 2:
                ref_ids.add(parts[1])
        if len(ref_ids) >= 2:
            multi_ref += 1

    if multi_ref > total * 0.5:
        return "sentence"
    return "word"


async def _run_word_level_regression(
    task_id: str,
    phase5_output: Path,
    output_dir: Path,
    gpu_id: int,
    template_path: Path,
) -> int:
    """
    Generate hand regression transitions for word-level videos.

    For each word video, creates:
    - intro: template (rest) → first frame (start signing)
    - outro: last frame (end signing) → template (rest)

    Then combines: intro + word frames + outro → final video
    """
    p5_videos = phase5_output / "videos"
    if not p5_videos.exists():
        return 0

    videos = sorted(p5_videos.glob("*.mp4"))
    if not videos:
        return 0

    # Prepare boundary frame pairs for FramerTurbo
    boundary_dir = output_dir / "boundary_pairs"
    boundary_dir.mkdir(parents=True, exist_ok=True)

    pair_count = 0
    video_map = {}  # track which pairs belong to which video

    for video in videos:
        n_frames = _get_frame_count(video)
        if n_frames < 2:
            continue

        name = video.stem

        # Pair A (intro): template → first frame of word
        intro_dir = boundary_dir / f"{name}_intro"
        intro_dir.mkdir(exist_ok=True)
        shutil.copy2(str(template_path), str(intro_dir / f"0_start.jpg"))
        _extract_frame(video, 0, intro_dir / f"0_end.jpg")

        # Pair B (outro): last frame of word → template
        outro_dir = boundary_dir / f"{name}_outro"
        outro_dir.mkdir(exist_ok=True)
        _extract_frame(video, n_frames - 1, outro_dir / f"0_start.jpg")
        shutil.copy2(str(template_path), str(outro_dir / f"0_end.jpg"))

        video_map[name] = {"video": video, "frames": n_frames}
        pair_count += 2

    logger.info(f"[{task_id}] Phase 6: Prepared {pair_count} boundary pairs "
                f"for {len(video_map)} word videos")

    if pair_count == 0:
        return 0

    # Run FramerTurbo on all pairs
    interp_dir = output_dir / "interp_results"
    interp_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(FRAMER_DIR)

    logger.info(f"[{task_id}] Phase 6: Running FramerTurbo on {pair_count} pairs")

    rc, _, stderr = await run_subprocess(
        [str(FRAMER_PYTHON), str(FRAMER_SCRIPT),
         "--input_dir", str(boundary_dir),
         "--model", str(FRAMER_CKPT),
         "--output_dir", str(interp_dir),
         "--scheduler", "euler"],
        cwd=str(FRAMER_DIR),
        env=env,
        timeout=3600 * 6,
    )

    if rc != 0:
        logger.warning(f"[{task_id}] Phase 6: FramerTurbo failed: {stderr[:300]}")
        return 0

    # Combine: intro_transition + word_frames + outro_transition → final video
    videos_out = output_dir / "videos"
    videos_out.mkdir(exist_ok=True)
    combined = 0

    for name, info in video_map.items():
        video = info["video"]

        # Find intro and outro interpolation results
        intro_interp = _find_interp_result(interp_dir, f"{name}_intro")
        outro_interp = _find_interp_result(interp_dir, f"{name}_outro")

        out_path = videos_out / f"{name}.mp4"

        # Read original video frames
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        word_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            word_frames.append(frame)
        cap.release()

        if not word_frames:
            continue

        # Write combined video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        # 1. Intro transition (rest → sign)
        if intro_interp:
            for frame in _read_interp_frames(intro_interp, (width, height)):
                writer.write(frame)

        # 2. Word frames
        for frame in word_frames:
            writer.write(frame)

        # 3. Outro transition (sign → rest)
        if outro_interp:
            for frame in _read_interp_frames(outro_interp, (width, height)):
                writer.write(frame)

        writer.release()
        combined += 1

    return combined


def _find_interp_result(interp_dir: Path, pair_name: str):
    """Find interpolation output (GIF or video) for a pair."""
    for ext in ("*.gif", "*.mp4", "*.avi"):
        results = list((interp_dir / pair_name).rglob(ext))
        if results:
            return results[0]
    # Check if result is directly in interp_dir
    for ext in ("*.gif", "*.mp4"):
        results = list(interp_dir.glob(f"{pair_name}{ext}"))
        if results:
            return results[0]
    return None


def _read_interp_frames(path: Path, target_size: tuple):
    """Read frames from an interpolation result (GIF or video)."""
    if path.suffix == ".gif":
        try:
            import imageio
            reader = imageio.get_reader(str(path))
            for frame in reader:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if (frame_bgr.shape[1], frame_bgr.shape[0]) != target_size:
                    frame_bgr = cv2.resize(frame_bgr, target_size)
                yield frame_bgr
        except Exception:
            return
    else:
        cap = cv2.VideoCapture(str(path))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame.shape[1], frame.shape[0]) != target_size:
                frame = cv2.resize(frame, target_size)
            yield frame
        cap.release()


async def _run_sentence_level_interpolation(
    task_id: str,
    phase5_output: Path,
    output_dir: Path,
    gpu_id: int,
) -> int:
    """
    Sentence-level: use author's original boundary extraction + FramerTurbo + combine.
    """
    boundary_dir = phase5_output / "step5_boundary"
    frames_dir = phase5_output / "step4_resized"
    videos_out = output_dir / "videos"
    videos_out.mkdir(exist_ok=True)

    if not boundary_dir.exists() or not list(boundary_dir.iterdir()):
        logger.warning(f"[{task_id}] Phase 6: No boundary frames found")
        return 0

    interp_dir = output_dir / "interp_frames"
    interp_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(FRAMER_DIR)

    logger.info(f"[{task_id}] Phase 6: Running sentence-level FramerTurbo")

    rc, _, stderr = await run_subprocess(
        [str(FRAMER_PYTHON), str(FRAMER_SCRIPT),
         "--input_dir", str(boundary_dir),
         "--model", str(FRAMER_CKPT),
         "--output_dir", str(interp_dir),
         "--scheduler", "euler"],
        cwd=str(FRAMER_DIR),
        env=env,
        timeout=3600 * 4,
    )
    if rc != 0:
        logger.warning(f"[{task_id}] Phase 6: FramerTurbo failed: {stderr[:200]}")
        return 0

    logger.info(f"[{task_id}] Phase 6: Combining frames and interpolations")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(COMBINE_SCRIPT),
         "--frames-root", str(frames_dir),
         "--interp-root", str(interp_dir),
         "--out-root", str(videos_out),
         "--fps", "25"],
        cwd=str(UNISIGN),
        timeout=3600,
    )
    if rc != 0:
        logger.warning(f"[{task_id}] Phase 6: Combine failed: {stderr[:200]}")

    return len(list(videos_out.glob("*.mp4")))


def _passthrough(task_id: str, phase5_output: Path, output_dir: Path) -> int:
    """Fallback: link Phase 5 videos directly."""
    videos_out = output_dir / "videos"
    videos_out.mkdir(exist_ok=True)
    p5_videos = phase5_output / "videos"
    if p5_videos.exists():
        for v in p5_videos.glob("*.mp4"):
            dst = videos_out / v.name
            if not dst.exists():
                dst.symlink_to(v.resolve())
    return len(list(videos_out.glob("*.mp4")))


async def run_phase6_framer(
    task_id: str,
    phase5_output: Path,
    output_dir: Path,
    gpu_id: int = 0,
    template_path: Path = None,
) -> bool:
    """
    Phase 6: Generate transitions with automatic mode detection.

    - Word-level: hand regression (rest ↔ sign transitions)
    - Sentence-level: word-to-word interpolation
    - Fallback: passthrough if FramerTurbo not available
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    template = template_path or DEFAULT_TEMPLATE

    # Check FramerTurbo availability
    if not FRAMER_CKPT.exists():
        logger.warning(f"[{task_id}] Phase 6: FramerTurbo checkpoint not found at {FRAMER_CKPT}")
        count = _passthrough(task_id, phase5_output, output_dir)
        logger.info(f"[{task_id}] Phase 6 completed (passthrough): {count} videos")
        return True

    # Detect mode
    mode = _detect_mode(phase5_output)
    logger.info(f"[{task_id}] Phase 6: Detected mode = {mode}")

    if mode == "word":
        if not template.exists():
            logger.warning(f"[{task_id}] Phase 6: Template image not found at {template}")
            count = _passthrough(task_id, phase5_output, output_dir)
        else:
            logger.info(f"[{task_id}] Phase 6: Word-level hand regression (template: {template.name})")
            count = await _run_word_level_regression(
                task_id, phase5_output, output_dir, gpu_id, template
            )
            if count == 0:
                logger.warning(f"[{task_id}] Phase 6: No regression videos generated, falling back")
                count = _passthrough(task_id, phase5_output, output_dir)
    else:
        logger.info(f"[{task_id}] Phase 6: Sentence-level interpolation")
        count = await _run_sentence_level_interpolation(
            task_id, phase5_output, output_dir, gpu_id
        )
        if count == 0:
            logger.warning(f"[{task_id}] Phase 6: No sentence videos generated, falling back")
            count = _passthrough(task_id, phase5_output, output_dir)

    # Write report
    report = {
        "mode": mode,
        "framer_available": True,
        "template": str(template) if mode == "word" else None,
        "videos_generated": count,
    }
    with open(output_dir / "phase6_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"[{task_id}] Phase 6 completed ({mode} mode): {count} videos")
    return True
