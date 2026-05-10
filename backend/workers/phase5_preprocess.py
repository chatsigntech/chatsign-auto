"""Pre-process videos before MimicMotion: extract frames, dedup, resize to 576, regenerate.

This reduces video resolution and removes redundant frames so MimicMotion
runs much faster (576p instead of 720p/1080p, fewer frames).
"""
import logging
import shutil
import sys
from pathlib import Path

from backend.config import settings
from backend.core.subprocess_runner import run_subprocess

logger = logging.getLogger(__name__)

SCRIPTS_DIR = settings.UNISIGN_PATH.resolve() / "scripts" / "sentence"
UNISIGN_CWD = settings.UNISIGN_PATH.resolve()
_FFMPEG = str(Path(__file__).resolve().parent.parent.parent / "bin" / "ffmpeg")


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink src→dst, falling back to copy on filesystems that block symlinks."""
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _center_crop_frames(in_root: Path, out_root: Path, size: int, logger, task_id: str):
    """Scale short edge to `size`, then center crop to size x size.

    Preserves the directory structure: in_root/<subdir>/*.jpg → out_root/<subdir>/*.jpg
    """
    import cv2

    total = 0
    for subdir in sorted(in_root.iterdir()):
        if not subdir.is_dir():
            continue
        out_sub = out_root / subdir.name
        out_sub.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(subdir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Scale so short edge = size
            if w < h:
                new_w = size
                new_h = int(h * size / w)
            else:
                new_h = size
                new_w = int(w * size / h)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # Center crop to size x size
            y0 = (new_h - size) // 2
            x0 = (new_w - size) // 2
            img = img[y0:y0 + size, x0:x0 + size]

            cv2.imwrite(str(out_sub / img_path.name), img)
            total += 1

    logger.info(f"[{task_id}] Center crop: {total} frames → {size}x{size}")


async def _ensure_25fps(src: Path, dst: Path) -> bool:
    """Place a 25-fps version of `src` at `dst`.

    Probes source fps via cv2; symlinks if already 25 (±0.1), else re-encodes
    with ffmpeg `-vf fps=25`. Returns False if the re-encode fails (caller
    falls back to raw symlink).
    """
    import cv2
    cap = cv2.VideoCapture(str(src))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if src_fps and abs(src_fps - 25.0) < 0.1:
        _link_or_copy(src, dst)
        return True

    tmp = dst.with_suffix(".tmp.mp4")
    rc, _, _ = await run_subprocess(
        [_FFMPEG, "-y", "-i", str(src),
         "-vf", "fps=25", "-an",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         str(tmp)],
    )
    if rc == 0 and tmp.exists():
        tmp.replace(dst)
        return True
    tmp.unlink(missing_ok=True)
    return False


async def preprocess_videos(task_id: str, input_dir: Path, output_dir: Path) -> Path:
    """Pre-process raw videos: extract frames → dedup → resize 576 → regenerate.

    Args:
        input_dir: Directory with raw mp4 videos
        output_dir: Working directory for intermediate outputs

    Returns:
        Path to directory containing preprocessed mp4 videos
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list(input_dir.rglob("*.mp4"))
    if not videos:
        logger.warning(f"[{task_id}] Preprocess: No videos found, skipping")
        return input_dir

    logger.info(f"[{task_id}] Preprocess: {len(videos)} videos from {input_dir}")

    # Organize into per-video subdirs (scripts expect this structure).
    # fps→25 normalization happens here so dedup operates on time-uniform
    # frames: per-frame diff% has stable physical meaning (~40ms motion),
    # and downstream 25fps regen plays at original speed when keep_ratio=100%.
    organized = output_dir / "organized"
    organized.mkdir(exist_ok=True)
    for v in videos:
        sent_dir = organized / v.stem
        sent_dir.mkdir(exist_ok=True)
        dst = sent_dir / v.name
        if dst.exists():
            continue
        ok = await _ensure_25fps(v, dst)
        if not ok:
            logger.warning(f"[{task_id}] Preprocess: fps normalize failed for {v.name}, using raw")
            _link_or_copy(v, dst)

    # Step 1: Extract frames
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Preprocess 1/4: Extracting frames")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "extract_all_frames_seq.py"),
         "--mp4-root", str(organized), "--out-root", str(frames_dir)],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        logger.warning(f"[{task_id}] Preprocess: frame extraction failed, using original videos")
        return input_dir

    # Step 2: Dedup frames
    dedup_dir = output_dir / "dedup"
    dedup_dir.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Preprocess 2/4: Deduplicating frames")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "filter_duplicate_frames.py"),
         "--frames-dir", str(frames_dir), "--output-dir", str(dedup_dir),
         "--save-cleaned-frames",
         "--duplicate-threshold", "2.0", "--min-duplicate-length", "2"],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        logger.warning(f"[{task_id}] Preprocess: dedup failed, using extracted frames")
        dedup_dir = frames_dir

    # Step 3: Scale + center crop to 576x576 (MimicMotion resolution)
    # Scale short edge to 576, then center crop to square — no distortion
    resized_dir = output_dir / "resized_576"
    resized_dir.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Preprocess 3/4: Scale + center crop to 576x576")
    _center_crop_frames(dedup_dir, resized_dir, 576, logger, task_id)
    if not any(resized_dir.rglob("*.jpg")):
        logger.warning(f"[{task_id}] Preprocess: crop produced no frames, using dedup")
        resized_dir = dedup_dir

    # Step 4: Regenerate videos from processed frames
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    logger.info(f"[{task_id}] Preprocess 4/4: Generating preprocessed videos")
    rc, _, stderr = await run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "generate_videos_from_frames.py"),
         "--frames-dir", str(resized_dir), "--output-dir", str(videos_dir), "--fps", "25"],
        cwd=UNISIGN_CWD,
    )
    if rc != 0:
        logger.warning(f"[{task_id}] Preprocess: video generation failed, using original")
        return input_dir

    generated = list(videos_dir.glob("*.mp4"))

    # Step 5: Re-encode to H.264 + faststart for browser playback
    # OpenCV's mp4v codec is not supported by modern browsers
    logger.info(f"[{task_id}] Preprocess 5/5: Re-encoding to H.264 for browser playback")
    for mp4 in generated:
        tmp = mp4.with_suffix(".tmp.mp4")
        rc2, _, _ = await run_subprocess(
            [_FFMPEG, "-y", "-i", str(mp4),
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-movflags", "+faststart", "-pix_fmt", "yuv420p",
             str(tmp)],
        )
        if rc2 == 0 and tmp.exists():
            tmp.replace(mp4)
        else:
            tmp.unlink(missing_ok=True)

    logger.info(f"[{task_id}] Preprocess complete: {len(generated)} videos at 576p")
    return videos_dir
