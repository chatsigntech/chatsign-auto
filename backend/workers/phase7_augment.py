"""Phase 6: Data augmentation using guava-aug.

Three categories of input videos, each augmented with same methods:
  - Sentence videos (from Phase 2)
  - Word videos (from Phase 2)
  - Segment videos (from Phase 5)

Four augmentation streams per category:
  1. 2D CV augmentation  – 25 types (geometric + color), CPU only
  2. Temporal augmentation – 7 types (speed + fps), CPU only
  3. 3D novel view rendering – EHM-Tracker + GUAVA fixed viewpoint, GPU required
  4. Identity cross-reenactment – GUAVA cross-act with tracked templates, GPU required

Records temporal transform parameters (speed_ratio) for Phase 7 split point scaling.
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
    temporal_params: dict | None = None,
) -> int:
    """Apply temporal augmentations (synchronous, CPU-bound).

    If temporal_params dict is provided, records speed_ratio for each output video
    (used by Phase 7 to scale split points).
    """
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
            aug_info = TEMPORAL_AUGMENTATIONS[aug_id]
            aug_name = aug_info["name"]
            out_path = temporal_output / aug_name / f"{video_name}.mp4"

            if out_path.exists():
                count += 1
                # Still record params for existing files
                if temporal_params is not None:
                    temporal_params[out_path.stem] = {
                        "speed_ratio": aug_info.get("speed_ratio", 1.0),
                        "aug_name": aug_name,
                        "source_video": video_path.name,
                    }
                continue

            try:
                temporal_augment_video(str(video_path), str(out_path), aug_id)
                count += 1
                if temporal_params is not None:
                    temporal_params[out_path.stem] = {
                        "speed_ratio": aug_info.get("speed_ratio", 1.0),
                        "aug_name": aug_name,
                        "source_video": video_path.name,
                    }
            except Exception as e:
                logger.error(f"[{task_id}] Temporal aug {aug_name} failed for {video_name}: {e}")

    logger.info(f"[{task_id}] Phase 6: Temporal augmentation done, {count} videos generated")
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
        "--in_root", str(input_dir.resolve()),
        "--output_dir", str(tracked_dir.resolve()),
        "--check_hand_score", "0.0",
        "-n", "1",
        "-v", "0",
    ]

    videos_in = list(input_dir.glob("*.mp4")) if input_dir.exists() else []
    logger.info(f"[{task_id}] EHM-Tracker: input_dir={input_dir} ({len(videos_in)} mp4), "
                f"output_dir={tracked_dir}, cwd={EHM_TRACKER_PATH}")
    result = subprocess.run(
        cmd,
        cwd=str(EHM_TRACKER_PATH),
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,  # 2 hour timeout
    )

    if result.returncode != 0:
        logger.error(f"[{task_id}] EHM-Tracker failed (rc={result.returncode}):\n{result.stderr[-2000:]}")
    else:
        logger.info(f"[{task_id}] EHM-Tracker completed (rc=0)")

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
        "--data_path", str(tracked_data_path.resolve()),
        "-s", str(output_dir.resolve()),
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


TEMPLATES_DIR = GUAVA_PATH / "assets" / "tracked_templates"


def _run_guava_cross_reenact(
    task_id: str,
    tracked_data_path: Path,
    template_path: Path,
    output_dir: Path,
    video_name: str,
    template_name: str,
    viewpoint: dict | None = None,
    gpu_id: int = 0,
) -> Path | None:
    """Render cross-reenactment: template identity + driving video motion.

    Optionally render from a fixed viewpoint after cross-reenactment.
    Returns path to rendered video, or None on failure.
    """
    vp_suffix = f"_{viewpoint['name']}" if viewpoint and viewpoint["name"] != "original" else ""
    save_name = f"x_{video_name}_{template_name}{vp_suffix}"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(GUAVA_PATH)
    env["XFORMERS_DISABLED"] = "1"

    cmd = [
        sys.executable, "main/test.py",
        "-d", "0",
        "-m", "assets/GUAVA",
        "--data_path", str(tracked_data_path.resolve()),
        "--source_data_path", str(template_path.resolve()),
        "-s", str(output_dir.resolve()),
        "--skip_self_act",
        "--render_cross_act",
        "-n", save_name,
    ]

    # If viewpoint is not "original", also render fixed viewpoint
    if viewpoint and viewpoint["name"] != "original":
        cmd.extend([
            "--render_fixed_viewpoint",
            "--fixed_yaw", str(viewpoint.get("yaw", 0.0)),
            "--fixed_pitch", str(viewpoint.get("pitch", 0.0)),
            "--fixed_zoom", str(viewpoint.get("zoom", 1.0)),
        ])

    result = subprocess.run(
        cmd,
        cwd=str(GUAVA_PATH),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if result.returncode != 0:
        logger.error(f"[{task_id}] Cross-reenact {save_name} failed:\n{result.stderr[-1000:]}")
        return None

    # Find output video
    render_dir = output_dir / f"{save_name}_cross_act"
    if not render_dir.exists():
        render_dir = output_dir
    mp4s = list(render_dir.rglob("*.mp4"))
    if mp4s:
        return mp4s[0]

    logger.warning(f"[{task_id}] Cross-reenact completed but no output video for {save_name}")
    return None


def _run_identity_augmentation(
    task_id: str,
    tracked_dir: Path,
    output_dir: Path,
    identity_config: dict,
    gpu_id: int = 0,
) -> int:
    """Run identity cross-reenactment for tracked videos × templates × viewpoints."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not TEMPLATES_DIR.exists():
        logger.warning(f"[{task_id}] Identity templates dir not found: {TEMPLATES_DIR}")
        return 0

    # Collect tracked videos
    tracked_videos = []
    if tracked_dir.exists():
        for d in sorted(tracked_dir.iterdir()):
            if d.is_dir() and (d / "optim_tracking_ehm.pkl").exists():
                tracked_videos.append(d)

    if not tracked_videos:
        logger.warning(f"[{task_id}] No tracked videos for identity augmentation")
        return 0

    templates = identity_config.get("templates", [])
    enabled_templates = [t for t in templates if t.get("enabled", True)]

    count = 0
    for tracked_path in tracked_videos:
        video_name = tracked_path.name
        for tpl in enabled_templates:
            tpl_dir = TEMPLATES_DIR / tpl["template_dir"]
            if not tpl_dir.exists():
                logger.warning(f"[{task_id}] Template not found: {tpl_dir}")
                continue
            tpl_label = tpl["template_dir"].replace(".jpeg", "").replace(" ", "_")[:20]

            for vp in tpl.get("viewpoints", [{"name": "original"}]):
                rendered = _run_guava_cross_reenact(
                    task_id, tracked_path, tpl_dir, output_dir,
                    video_name, tpl_label, vp, gpu_id,
                )
                if rendered:
                    count += 1

    logger.info(f"[{task_id}] Phase 7: Identity augmentation done, {count} videos rendered")
    return count


def _build_render_manifest(
    tracked_videos: list[Path],
    viewpoints: list[dict],
    identity_cfg: dict | None,
    render_dir: Path,
    identity_dir: Path,
) -> list[dict]:
    """Build a combined manifest of 3D viewpoint + identity render jobs."""
    jobs = []

    # 3D fixed viewpoint jobs
    for tracked_path in tracked_videos:
        video_name = tracked_path.name
        for vp in viewpoints:
            save_name = f"{video_name}_{vp['name']}"
            jobs.append({
                "type": "fixed_viewpoint",
                "data_path": str(tracked_path.resolve()),
                "save_path": str(render_dir.resolve()),
                "save_name": save_name,
                "fixed_yaw": vp.get("yaw", 0.0),
                "fixed_pitch": vp.get("pitch", 0.0),
                "fixed_zoom": vp.get("zoom", 1.0),
            })

    # Identity cross-reenactment jobs
    if identity_cfg and identity_cfg.get("enabled", False):
        templates = identity_cfg.get("templates", [])
        enabled_templates = [t for t in templates if t.get("enabled", True)]

        for tracked_path in tracked_videos:
            video_name = tracked_path.name
            for tpl in enabled_templates:
                tpl_dir = TEMPLATES_DIR / tpl["template_dir"]
                if not tpl_dir.exists():
                    continue
                tpl_label = tpl["template_dir"].replace(".jpeg", "").replace(" ", "_")[:20]

                for vp in tpl.get("viewpoints", [{"name": "original"}]):
                    vp_suffix = f"_{vp['name']}" if vp.get("name") != "original" else ""
                    save_name = f"x_{video_name}_{tpl_label}{vp_suffix}"
                    job = {
                        "type": "cross_reenact",
                        "data_path": str(tracked_path.resolve()),
                        "source_data_path": str(tpl_dir.resolve()),
                        "save_path": str(identity_dir.resolve()),
                        "save_name": save_name,
                    }
                    if vp.get("name") != "original":
                        job["fixed_yaw"] = vp.get("yaw", 0.0)
                        job["fixed_pitch"] = vp.get("pitch", 0.0)
                        job["fixed_zoom"] = vp.get("zoom", 1.0)
                    jobs.append(job)

    return jobs


def _partition_manifest(jobs: list[dict], num_workers: int) -> list[list[dict]]:
    """Partition jobs into N worker groups, keeping same data_path together.

    Uses greedy load-balancing: groups by data_path, then assigns each group
    to the worker with the fewest jobs so far.
    """
    from collections import defaultdict

    # Group by data_path
    groups = defaultdict(list)
    for job in jobs:
        groups[job["data_path"]].append(job)

    # Sort groups by size (largest first) for better balancing
    sorted_groups = sorted(groups.values(), key=len, reverse=True)

    # Greedy assignment to workers
    workers = [[] for _ in range(num_workers)]
    worker_counts = [0] * num_workers

    for group in sorted_groups:
        # Assign to worker with fewest jobs
        min_idx = worker_counts.index(min(worker_counts))
        workers[min_idx].extend(group)
        worker_counts[min_idx] += len(group)

    return [w for w in workers if w]  # Remove empty workers


def _run_batch_render(
    task_id: str,
    jobs: list[dict],
    gpu_id: int = 0,
    num_workers: int = 3,
    work_dir: Path | None = None,
) -> int:
    """Run batch rendering with multiple parallel workers.

    Returns total number of successfully rendered videos.
    """
    if not jobs:
        return 0

    if work_dir is None:
        work_dir = Path("/tmp") / f"batch_render_{task_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    BATCH_SCRIPT = GUAVA_PATH / "batch_render.py"

    # Partition jobs across workers
    partitions = _partition_manifest(jobs, num_workers)
    actual_workers = len(partitions)

    logger.info(
        f"[{task_id}] Batch render: {len(jobs)} jobs across {actual_workers} workers"
    )

    # Write manifests and launch workers
    processes = []
    status_files = []

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(GUAVA_PATH)
    env["XFORMERS_DISABLED"] = "1"

    for i, partition in enumerate(partitions):
        manifest_path = work_dir / f"manifest_worker_{i}.json"
        status_file = work_dir / f"status_worker_{i}.json"

        with open(manifest_path, "w") as f:
            json.dump({"jobs": partition}, f)

        cmd = [
            sys.executable, str(BATCH_SCRIPT),
            "--manifest", str(manifest_path),
            "--model_path", "assets/GUAVA",
            "--device", str(gpu_id),
            "--status_file", str(status_file),
        ]

        proc = subprocess.Popen(
            cmd,
            cwd=str(GUAVA_PATH),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes.append(proc)
        status_files.append(status_file)

    # Wait for all workers to complete
    total_completed = 0
    for i, proc in enumerate(processes):
        try:
            stdout, stderr = proc.communicate(timeout=3600 * 8)  # 8 hour timeout
            if proc.returncode != 0:
                logger.error(
                    f"[{task_id}] Batch worker {i} failed (rc={proc.returncode}): "
                    f"{stderr.decode()[-500:]}"
                )
        except subprocess.TimeoutExpired:
            proc.kill()
            logger.error(f"[{task_id}] Batch worker {i} timed out")

        # Read final status
        sf = status_files[i]
        if sf.exists():
            with open(sf) as f:
                status = json.load(f)
                total_completed += status.get("completed", 0)

    logger.info(f"[{task_id}] Batch render done: {total_completed}/{len(jobs)} jobs")
    return total_completed


def _run_3d_and_identity_augmentation(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
    gpu_id: int = 0,
    viewpoints: list[dict] | None = None,
    identity_cfg: dict | None = None,
    num_workers: int = 3,
) -> tuple[int, int]:
    """Run 3D + identity augmentation with batch rendering.

    Returns (3d_count, identity_count).
    """
    if viewpoints is None:
        viewpoints = DEFAULT_VIEWPOINTS

    tracked_dir = output_dir / "tracked"
    render_dir = output_dir / "3d_views"
    identity_dir = output_dir / "identity"
    render_dir.mkdir(parents=True, exist_ok=True)
    identity_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: EHM-Tracker
    tracked_videos = _run_ehm_tracking(task_id, input_dir, tracked_dir, gpu_id)

    if not tracked_videos:
        logger.warning(f"[{task_id}] No videos tracked, skipping 3D/identity rendering")
        return 0, 0

    # Step 2: Build combined manifest
    jobs = _build_render_manifest(
        tracked_videos, viewpoints, identity_cfg, render_dir, identity_dir
    )

    if not jobs:
        return 0, 0

    n_3d = sum(1 for j in jobs if j["type"] == "fixed_viewpoint")
    n_identity = sum(1 for j in jobs if j["type"] == "cross_reenact")
    logger.info(f"[{task_id}] Manifest: {n_3d} 3D + {n_identity} identity = {len(jobs)} jobs")

    # Step 3: Batch render with auto-downgrade on OOM
    work_dir = output_dir / ".batch_work"
    count = 0
    for workers in (num_workers, max(1, num_workers - 1), 1):
        try:
            count = _run_batch_render(task_id, jobs, gpu_id, workers, work_dir)
            break
        except Exception as e:
            if workers > 1:
                logger.warning(f"[{task_id}] Batch render with {workers} workers failed: {e}, retrying with fewer")
            else:
                logger.error(f"[{task_id}] Batch render failed with 1 worker: {e}")
                # Fallback to sequential per-job rendering
                logger.info(f"[{task_id}] Falling back to sequential rendering")
                count_3d = 0
                for tracked_path in tracked_videos:
                    video_name = tracked_path.name
                    for vp in viewpoints:
                        rendered = _run_guava_render(
                            task_id, tracked_path, render_dir, video_name, vp, gpu_id
                        )
                        if rendered:
                            count_3d += 1
                return count_3d, 0

    # Count actual output videos
    actual_3d = len(list(render_dir.rglob("*_fixed_viewpoint_video.mp4")))
    actual_identity = len(list(identity_dir.rglob("*.mp4")))

    logger.info(
        f"[{task_id}] 3D+Identity done: {actual_3d} 3D views, {actual_identity} identity videos"
    )
    return actual_3d, actual_identity


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _augment_category(
    task_id: str,
    category: str,
    videos: list[Path],
    output_dir: Path,
    gpu_id: int,
    enable_2d: bool,
    enable_temporal: bool,
    enable_3d: bool,
    enable_identity: bool,
    cv_aug_ids: list[int] | None,
    temporal_aug_ids: list[int] | None,
    viewpoints: list[dict] | None,
    identity_cfg: dict | None,
    temporal_params: dict,
) -> dict:
    """Run augmentation for a single category of videos (synchronous)."""
    cat_dir = output_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    totals = {"2d_cv": 0, "temporal": 0, "3d_views": 0, "identity": 0}

    if enable_2d:
        totals["2d_cv"] = _run_2d_augmentation(task_id, videos, cat_dir, cv_aug_ids)

    if enable_temporal:
        totals["temporal"] = _run_temporal_augmentation(
            task_id, videos, cat_dir, temporal_aug_ids, temporal_params
        )

    if enable_3d or (enable_identity and identity_cfg):
        input_dir = videos[0].parent if videos else cat_dir
        views_count, identity_count = _run_3d_and_identity_augmentation(
            task_id, input_dir, cat_dir, gpu_id,
            viewpoints=viewpoints if enable_3d else [],
            identity_cfg=identity_cfg if enable_identity else None,
            num_workers=3,
        )
        if enable_3d:
            totals["3d_views"] = views_count
        if enable_identity:
            totals["identity"] = identity_count

    return totals


async def run_phase6_augment(
    task_id: str,
    phase2_output: Path,
    phase5_output: Path,
    output_dir: Path,
    gpu_id: int = 0,
) -> bool:
    """Run data augmentation pipeline on three categories of input.

    Categories:
      - sentence: Phase 2 sentence videos
      - word: Phase 2 word videos
      - segment: Phase 5 segmented word clips

    Each category is augmented with the same methods (2D CV, temporal, 3D, identity).
    Records temporal_params.json for Phase 7 split point scaling.
    """
    config = _load_augmentation_config()

    enable_2d = True
    enable_temporal = True
    enable_3d = False
    enable_identity = False
    cv_aug_ids = None
    temporal_aug_ids = None
    viewpoints = None
    identity_cfg = {}

    if config:
        if "cv2d" in config:
            enable_2d = config["cv2d"].get("enabled", enable_2d)
        if "temporal" in config:
            enable_temporal = config["temporal"].get("enabled", enable_temporal)
        if "view3d" in config:
            enable_3d = config["view3d"].get("enabled", enable_3d)

        if cv_aug_ids is None and "cv2d" in config:
            augs = config["cv2d"].get("augmentations", [])
            if augs:
                cv_aug_ids = [a["id"] for a in augs if a.get("enabled", True)]

        if temporal_aug_ids is None and "temporal" in config:
            augs = config["temporal"].get("augmentations", [])
            if augs:
                temporal_aug_ids = [a["id"] for a in augs if a.get("enabled", True)]

        if viewpoints is None and "view3d" in config:
            vp_list = config["view3d"].get("viewpoints", [])
            if vp_list:
                viewpoints = [
                    {"name": v["name"], "yaw": v["yaw"], "pitch": v["pitch"], "zoom": v["zoom"]}
                    for v in vp_list if v.get("enabled", True)
                ]

        identity_cfg = config.get("identity", {})
        enable_identity = identity_cfg.get("enabled", False)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect videos by category using filename prefix convention:
    #   sentence_*.mp4 → sentence videos
    #   word_*.mp4     → word videos
    phase2_videos_dir = phase2_output / "videos"

    sentence_videos = []
    word_videos = []

    for v in _find_videos(phase2_videos_dir):
        if v.name.startswith("word_"):
            word_videos.append(v)
        else:
            sentence_videos.append(v)

    # Phase 5 segment videos
    segment_videos = []
    if phase5_output.exists():
        seg_dir = phase5_output / "segment_videos"
        if seg_dir.exists():
            segment_videos = _find_videos(seg_dir)

    logger.info(
        f"[{task_id}] Phase 6: Input videos - "
        f"{len(sentence_videos)} sentences, {len(word_videos)} words, "
        f"{len(segment_videos)} segments"
    )

    # Temporal params dict (shared, filled by _run_temporal_augmentation)
    temporal_params = {}

    loop = asyncio.get_event_loop()
    category_results = {}

    # Progress tracking: each category with videos is one unit
    categories = [
        ("sentence", sentence_videos),
        ("word", word_videos),
        ("segment", segment_videos),
    ]
    active_cats = [(n, v) for n, v in categories if v]
    total_cats = len(active_cats)

    def _update_progress(done_cats: int, current_cat: str = ""):
        """Write progress to summary.json for frontend polling."""
        pct = round(done_cats / total_cats * 100) if total_cats else 0
        progress = {
            "status": "running",
            "progress_pct": pct,
            "completed_categories": done_cats,
            "total_categories": total_cats,
            "current_category": current_cat,
            "categories": category_results,
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(progress, f, indent=2)

    # Update DB progress
    from sqlmodel import Session as DbSession
    from backend.database import engine
    from backend.core.phase_state_manager import PhaseStateManager

    for cat_idx, (cat_name, cat_videos) in enumerate(categories):
        if not cat_videos:
            category_results[cat_name] = {"2d_cv": 0, "temporal": 0, "3d_views": 0, "identity": 0}
            continue

        _update_progress(cat_idx, cat_name)
        with DbSession(engine) as session:
            pct = round(cat_idx / total_cats * 100) if total_cats else 0
            PhaseStateManager.update_progress(task_id, 6, session, pct)

        result = await loop.run_in_executor(
            None, _augment_category,
            task_id, cat_name, cat_videos, output_dir, gpu_id,
            enable_2d, enable_temporal, enable_3d, enable_identity,
            cv_aug_ids, temporal_aug_ids, viewpoints, identity_cfg,
            temporal_params,
        )
        category_results[cat_name] = result

    _update_progress(total_cats, "done")

    # Save temporal params for Phase 7
    with open(output_dir / "temporal_params.json", "w") as f:
        json.dump(temporal_params, f, indent=2)

    # Build manifest
    total_by_type = {"2d_cv": 0, "temporal": 0, "3d_views": 0, "identity": 0}
    for cat_result in category_results.values():
        for k in total_by_type:
            total_by_type[k] += cat_result.get(k, 0)

    total = sum(total_by_type.values())

    manifest = {
        "input_sentences": len(sentence_videos),
        "input_words": len(word_videos),
        "input_segments": len(segment_videos),
        "categories": category_results,
        "augmentations": {
            "2d_cv": {"enabled": enable_2d, "count": total_by_type["2d_cv"]},
            "temporal": {"enabled": enable_temporal, "count": total_by_type["temporal"]},
            "3d_views": {"enabled": enable_3d, "count": total_by_type["3d_views"]},
            "identity": {"enabled": enable_identity, "count": total_by_type["identity"]},
        },
        "total_generated": total,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"[{task_id}] Phase 6 completed: "
        f"{total_by_type['2d_cv']} 2D + {total_by_type['temporal']} temporal + "
        f"{total_by_type['3d_views']} 3D + {total_by_type['identity']} identity = {total} total"
    )
    return True
