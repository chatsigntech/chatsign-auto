"""Phase 7: Data augmentation using guava-aug.

Three parallel augmentation streams (matching upstream author's pipeline):
  1. 2D CV augmentation  – 25 types (geometric + color), CPU only
  2. Temporal augmentation – 7 types (speed + fps), CPU only
  3. 3D novel view rendering – EHM-Tracker + GUAVA, GPU required
     - Fixed viewpoint: 6 camera angles (yaw left/right, pitch up/down, zoom in/out)
     - Optionally cross-reenactment (disabled by default)
"""
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from backend.config import settings

logger = logging.getLogger(__name__)

GUAVA_PATH = settings.GUAVA_AUG_PATH.resolve()
EHM_TRACKER_PATH = GUAVA_PATH / "EHM-Tracker"

# Add guava-aug to Python path once at module level
if str(GUAVA_PATH) not in sys.path:
    sys.path.insert(0, str(GUAVA_PATH))

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _load_augmentation_config() -> dict:
    """Load augmentation config from JSON file, falling back to defaults."""
    config_path = settings.AUGMENTATION_CONFIG_PATH
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load augmentation config from {config_path}: {e}")
    return {}


# Default viewpoints for 3D augmentation (from upstream run_demo25_all.sh)
DEFAULT_VIEWPOINTS = [
    {"name": "yaw_right",  "yaw":  0.25, "pitch": 0.0,   "zoom": 1.0},
    {"name": "yaw_left",   "yaw": -0.25, "pitch": 0.0,   "zoom": 1.0},
    {"name": "pitch_up",   "yaw": 0.0,   "pitch": -0.25, "zoom": 1.0},
    {"name": "pitch_down", "yaw": 0.0,   "pitch":  0.25, "zoom": 1.0},
    {"name": "zoom_in",    "yaw": 0.0,   "pitch": 0.0,   "zoom": 0.85},
    {"name": "zoom_out",   "yaw": 0.0,   "pitch": 0.0,   "zoom": 1.15},
]


def _find_videos(input_dir: Path) -> list[Path]:
    """Find all video files in a directory (non-recursive)."""
    if not input_dir.exists():
        return []
    return [f for f in sorted(input_dir.iterdir())
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS]


# ---------------------------------------------------------------------------
# 2D CV augmentation
# ---------------------------------------------------------------------------

def _run_2d_augmentation(
    task_id: str,
    videos: list[Path],
    output_dir: Path,
    aug_ids: list[int] | None = None,
) -> int:
    """Apply 2D CV augmentations (synchronous, CPU-bound)."""
    from cv_aug.augment import augment_video, AUGMENTATIONS

    if aug_ids is None:
        aug_ids = list(range(len(AUGMENTATIONS)))

    if not videos:
        return 0

    cv_output = output_dir / "cv_aug"
    for aug_id in aug_ids:
        (cv_output / AUGMENTATIONS[aug_id]["name"]).mkdir(parents=True, exist_ok=True)

    count = 0
    for video_path in videos:
        video_name = video_path.stem
        for aug_id in aug_ids:
            aug_name = AUGMENTATIONS[aug_id]["name"]
            out_path = cv_output / aug_name / f"{video_name}.mp4"

            if out_path.exists():
                count += 1
                continue

            try:
                augment_video(str(video_path), str(out_path), aug_id, video_name=video_name)
                count += 1
            except Exception as e:
                logger.error(f"[{task_id}] 2D aug {aug_name} failed for {video_name}: {e}")

    logger.info(f"[{task_id}] Phase 7: 2D augmentation done, {count} videos generated")
    return count


# ---------------------------------------------------------------------------
# Temporal augmentation
# ---------------------------------------------------------------------------

def _run_temporal_augmentation(
    task_id: str,
    videos: list[Path],
    output_dir: Path,
    aug_ids: list[int] | None = None,
) -> int:
    """Apply temporal augmentations (synchronous, CPU-bound)."""
    from cv_aug.temporal_augment import temporal_augment_video, TEMPORAL_AUGMENTATIONS

    if aug_ids is None:
        aug_ids = list(range(len(TEMPORAL_AUGMENTATIONS)))

    if not videos:
        return 0

    temporal_output = output_dir / "temporal_aug"
    for aug_id in aug_ids:
        (temporal_output / TEMPORAL_AUGMENTATIONS[aug_id]["name"]).mkdir(parents=True, exist_ok=True)

    count = 0
    for video_path in videos:
        video_name = video_path.stem
        for aug_id in aug_ids:
            aug_name = TEMPORAL_AUGMENTATIONS[aug_id]["name"]
            out_path = temporal_output / aug_name / f"{video_name}.mp4"

            if out_path.exists():
                count += 1
                continue

            try:
                temporal_augment_video(str(video_path), str(out_path), aug_id)
                count += 1
            except Exception as e:
                logger.error(f"[{task_id}] Temporal aug {aug_name} failed for {video_name}: {e}")

    logger.info(f"[{task_id}] Phase 7: Temporal augmentation done, {count} videos generated")
    return count


# ---------------------------------------------------------------------------
# 3D novel view rendering (EHM-Tracker + GUAVA)
# ---------------------------------------------------------------------------

def _run_ehm_tracking(
    task_id: str,
    input_dir: Path,
    tracked_dir: Path,
    gpu_id: int = 0,
) -> list[Path]:
    """Run EHM-Tracker on raw videos to produce tracked body/face/hand data.

    Returns list of tracked data directories (one per video).
    """
    tracked_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(EHM_TRACKER_PATH)
    env["XFORMERS_DISABLED"] = "1"

    cmd = [
        sys.executable, "tracking_video.py",
        "--in_root", str(input_dir),
        "--output_dir", str(tracked_dir),
        "--check_hand_score", "0.0",
        "-n", "1",
        "-v", "0",
    ]

    logger.info(f"[{task_id}] Phase 7: Starting EHM-Tracker on {input_dir}")
    result = subprocess.run(
        cmd,
        cwd=str(EHM_TRACKER_PATH),
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,  # 2 hour timeout
    )

    if result.returncode != 0:
        logger.error(f"[{task_id}] EHM-Tracker stderr:\n{result.stderr[-2000:]}")

    # Collect successfully tracked directories
    tracked_videos = []
    if tracked_dir.exists():
        for d in sorted(tracked_dir.iterdir()):
            if d.is_dir() and (d / "optim_tracking_ehm.pkl").exists():
                tracked_videos.append(d)

    logger.info(f"[{task_id}] Phase 7: EHM-Tracker done, {len(tracked_videos)} videos tracked")
    return tracked_videos


def _run_guava_render(
    task_id: str,
    tracked_data_path: Path,
    output_dir: Path,
    video_name: str,
    viewpoint: dict,
    gpu_id: int = 0,
) -> Path | None:
    """Render a single video from a fixed viewpoint using GUAVA.

    Returns path to the rendered video, or None on failure.
    """
    view_name = viewpoint["name"]
    save_name = f"{video_name}_{view_name}"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(GUAVA_PATH)
    env["XFORMERS_DISABLED"] = "1"

    cmd = [
        sys.executable, "main/test.py",
        "-d", "0",
        "-m", "assets/GUAVA",
        "--data_path", str(tracked_data_path),
        "-s", str(output_dir),
        "--skip_self_act",
        "--render_fixed_viewpoint",
        "--fixed_yaw", str(viewpoint["yaw"]),
        "--fixed_pitch", str(viewpoint["pitch"]),
        "--fixed_zoom", str(viewpoint["zoom"]),
        "-n", save_name,
    ]

    result = subprocess.run(
        cmd,
        cwd=str(GUAVA_PATH),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour timeout per video
    )

    if result.returncode != 0:
        logger.error(f"[{task_id}] GUAVA render {save_name} failed:\n{result.stderr[-1000:]}")
        return None

    # Find the output video
    expected = (output_dir / f"{save_name}_fixed_viewpoint" / video_name
                / f"{video_name}_fixed_viewpoint_video.mp4")
    if expected.exists():
        return expected

    # Fallback: search for any mp4 in the output
    render_dir = output_dir / f"{save_name}_fixed_viewpoint"
    if render_dir.exists():
        mp4s = list(render_dir.rglob("*.mp4"))
        if mp4s:
            return mp4s[0]

    logger.warning(f"[{task_id}] GUAVA render completed but no output video found for {save_name}")
    return None


def _run_3d_augmentation(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
    gpu_id: int = 0,
    viewpoints: list[dict] | None = None,
) -> int:
    """Run full 3D augmentation: EHM tracking → GUAVA multi-view rendering."""
    if viewpoints is None:
        viewpoints = DEFAULT_VIEWPOINTS

    tracked_dir = output_dir / "tracked"
    render_dir = output_dir / "3d_views"
    render_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: EHM-Tracker
    tracked_videos = _run_ehm_tracking(task_id, input_dir, tracked_dir, gpu_id)

    if not tracked_videos:
        logger.warning(f"[{task_id}] Phase 7: No videos tracked, skipping 3D rendering")
        return 0

    # Step 2: GUAVA fixed viewpoint rendering for each tracked video × each viewpoint
    count = 0
    for tracked_path in tracked_videos:
        video_name = tracked_path.name
        for vp in viewpoints:
            rendered = _run_guava_render(
                task_id, tracked_path, render_dir, video_name, vp, gpu_id
            )
            if rendered:
                count += 1

    logger.info(f"[{task_id}] Phase 7: 3D augmentation done, {count} videos rendered")
    return count


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_phase7_augment(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
    gpu_id: int = 0,
    enable_3d: bool = True,
    enable_2d: bool = True,
    enable_temporal: bool = True,
    cv_aug_ids: list[int] | None = None,
    temporal_aug_ids: list[int] | None = None,
    viewpoints: list[dict] | None = None,
) -> bool:
    """Run data augmentation pipeline.

    Three parallel streams:
      1. 2D CV augmentations (CPU) – 25 types
      2. Temporal augmentations (CPU) – 7 types
      3. 3D novel view rendering (GPU) – 6 viewpoints via EHM-Tracker + GUAVA
    """
    # Load augmentation config and override parameters from it
    config = _load_augmentation_config()

    if config:
        # Override enable flags from config sections
        if "cv2d" in config:
            enable_2d = config["cv2d"].get("enabled", enable_2d)
        if "temporal" in config:
            enable_temporal = config["temporal"].get("enabled", enable_temporal)
        if "view3d" in config:
            enable_3d = config["view3d"].get("enabled", enable_3d)

        # Collect enabled cv2d augmentation IDs
        if cv_aug_ids is None and "cv2d" in config:
            augs = config["cv2d"].get("augmentations", [])
            if augs:
                cv_aug_ids = [a["id"] for a in augs if a.get("enabled", True)]

        # Collect enabled temporal augmentation IDs
        if temporal_aug_ids is None and "temporal" in config:
            augs = config["temporal"].get("augmentations", [])
            if augs:
                temporal_aug_ids = [a["id"] for a in augs if a.get("enabled", True)]

        # Collect enabled 3D viewpoints
        if viewpoints is None and "view3d" in config:
            vp_list = config["view3d"].get("viewpoints", [])
            if vp_list:
                viewpoints = [
                    {"name": v["name"], "yaw": v["yaw"], "pitch": v["pitch"], "zoom": v["zoom"]}
                    for v in vp_list if v.get("enabled", True)
                ]

        # Identity/cross-reenactment: warn if enabled but templates missing
        identity_cfg = config.get("identity", {})
        if identity_cfg.get("enabled", False):
            templates_dir = settings.GUAVA_AUG_PATH / "identity_templates"
            if not templates_dir.exists():
                logger.warning(
                    f"Identity/cross-reenactment enabled in config but "
                    f"templates directory not found: {templates_dir}"
                )

    output_dir.mkdir(parents=True, exist_ok=True)

    videos = _find_videos(input_dir)
    if not videos:
        logger.warning(f"[{task_id}] Phase 7: No input videos found in {input_dir}")
        return True

    loop = asyncio.get_event_loop()
    total_2d = 0
    total_temporal = 0
    total_3d = 0

    # Launch CPU tasks (2D + temporal) in parallel with GPU task (3D)
    tasks = []

    if enable_2d:
        tasks.append(loop.run_in_executor(
            None, _run_2d_augmentation, task_id, videos, output_dir, cv_aug_ids
        ))

    if enable_temporal:
        tasks.append(loop.run_in_executor(
            None, _run_temporal_augmentation, task_id, videos, output_dir, temporal_aug_ids
        ))

    if enable_3d:
        tasks.append(loop.run_in_executor(
            None, _run_3d_augmentation, task_id, input_dir, output_dir, gpu_id, viewpoints
        ))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Unpack results in order
    idx = 0
    if enable_2d:
        r = results[idx]; idx += 1
        total_2d = r if isinstance(r, int) else 0
        if isinstance(r, Exception):
            logger.error(f"[{task_id}] 2D augmentation failed: {r}")

    if enable_temporal:
        r = results[idx]; idx += 1
        total_temporal = r if isinstance(r, int) else 0
        if isinstance(r, Exception):
            logger.error(f"[{task_id}] Temporal augmentation failed: {r}")

    if enable_3d:
        r = results[idx]; idx += 1
        total_3d = r if isinstance(r, int) else 0
        if isinstance(r, Exception):
            logger.error(f"[{task_id}] 3D augmentation failed: {r}")

    manifest = {
        "input_dir": str(input_dir),
        "input_videos": len(videos),
        "augmentations": {
            "2d_cv": {"enabled": enable_2d, "count": total_2d},
            "temporal": {"enabled": enable_temporal, "count": total_temporal},
            "3d_views": {"enabled": enable_3d, "count": total_3d},
        },
        "total_generated": total_2d + total_temporal + total_3d,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total = total_2d + total_temporal + total_3d
    logger.info(f"[{task_id}] Phase 7 completed: {total_2d} 2D + {total_temporal} temporal + "
                f"{total_3d} 3D = {total} total augmented videos")
    return True
