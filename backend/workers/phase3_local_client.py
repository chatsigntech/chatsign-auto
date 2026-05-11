"""Phase 3 LOCAL client — DGX-equivalent fallback, runs on local GPU.

Mirrors `phase3_dgx_client.run_phase3_on_dgx`'s 3-stage chain on a single
local GPU, so an output mp4 from this backend is the same artifact as the
DGX `sr.mp4` (modulo minor numeric drift between sm_120 vs DGX GB10).

Stages per video (matches DGX `infer_dgx_total.sh` + `infer_dgx_tail_glitch.sh`
+ `infer_dgx_realesr.sh`):
  J1a mimic        → batch_process.py            → output/input_hiya.mp4
  J1b filter       → trim_inactive_frames.py     → output_filter/input.mp4
                     (head/tail trim; dedup + pose filter off, mirroring
                     phase3_dgx_client's filter_env)
  J2  tail_glitch  → detect_tail_glitch.py       → tail_glitch.mp4
  J3  SR upscale   → upscale_video.py x2         → sr.mp4
  remux            → ffmpeg libx264              → videos_out/<original>.mp4

Identical to DGX:
  - Source: rsync'd from /media/cvpr/zhewen/UniSignMimicTurbo (DGX)
  - Weights: MimicMotion_1-1.pth + DWPose + svd_cache + RealESR x4v3 (same files)
  - CLI args: --mode square --crop-anchor top --sample-stride 1
  - Filter env: FILTER_HEAD_TAIL=true, ACTIVITY_THRESHOLD=0.7, MARGIN=3,
    FILTER_DUPLICATE=false, FILTER_POSE=false
  - J2 min-cats=2, J3 OUTSCALE=2

Different from DGX:
  - Hardware: local sm_120 vs DGX GB10 — minor numeric drift in CUDA kernels,
    no algorithmic difference.
  - Concurrency: serial per-video (1 GPU). Caller can spawn parallel tasks
    if there's enough VRAM, but defaults to one-at-a-time.
"""
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


MIMIC_ROOT = Path(os.environ.get(
    "PHASE3_MIMIC_ROOT", "/mnt/data/UniSignMimicTurbo_dgx",
))
MIMIC_PYTHON = Path(os.environ.get(
    "PHASE3_MIMIC_PYTHON",
    "/home/chatsign/miniconda3/envs/mimicmotion_dgx/bin/python",
))
DEFAULT_REF_IMAGE = Path(os.environ.get(
    "PHASE3_LOCAL_REF_IMAGE",
    str(MIMIC_ROOT / "mimicmotion" / "data" / "ref_images" / "test4.jpg"),
))
# J2/J3 toolchain (local mirror of /media/cvpr/zhewen/cv/).
CV_LOCAL_ROOT = Path(os.environ.get(
    "PHASE3_CV_LOCAL_ROOT", "/home/chatsign/lizh/cv_local",
))
TAIL_GLITCH_SCRIPT = CV_LOCAL_ROOT / "tail_glitch" / "detect_tail_glitch.py"
REALESR_SCRIPT = CV_LOCAL_ROOT / "RealESR" / "upscale_video.py"
REALESR_EXTRAS = CV_LOCAL_ROOT / "realesr_extras"

# Filter knobs — defaults mirror phase3_dgx_client's filter_env so the two
# backends produce comparable outputs without per-call configuration.
FILTER_HEAD_TAIL = os.environ.get("PHASE3_FILTER_HEAD_TAIL", "true").lower() == "true"
FILTER_DUPLICATE = os.environ.get("PHASE3_FILTER_DUPLICATE", "false").lower() == "true"
FILTER_POSE = os.environ.get("PHASE3_FILTER_POSE", "false").lower() == "true"
ACTIVITY_THRESHOLD = float(os.environ.get("PHASE3_ACTIVITY_THRESHOLD", "0.7"))
MARGIN = int(os.environ.get("PHASE3_MARGIN", "3"))
FPS = int(os.environ.get("PHASE3_FPS", "25"))
SAMPLE_STRIDE = int(os.environ.get("PHASE3_SAMPLE_STRIDE", "1"))
TAIL_GLITCH_MIN_CATS = int(os.environ.get("PHASE3_TG_MIN_CATS", "2"))
REALESR_OUTSCALE = float(os.environ.get("PHASE3_SR_OUTSCALE", "2.0"))

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FFMPEG = _REPO_ROOT / "bin" / "ffmpeg"
FFMPEG_CONCURRENCY = int(os.environ.get("PHASE3_LOCAL_FFMPEG_CONCURRENCY", "4"))


ProgressCb = Callable[[float], Awaitable[None] | None]


async def _emit_progress(progress_cb: ProgressCb | None, value: float, last: list[float]) -> None:
    if progress_cb is None:
        return
    if last and abs(value - last[-1]) < 0.5:
        return
    last.append(value)
    result = progress_cb(value)
    if asyncio.iscoroutine(result):
        await result


def _failed(filename: str, stage: str, error: str) -> dict:
    return {"filename": filename, "status": "failed", "stage": stage, "error": error[-500:]}


def _mimic_env() -> dict:
    """Env for stage J1 (MimicMotion batch_process).

    Mirrors DGX `infer_dgx_total.sh`: pin HF cache to local svd snapshot
    + offline mode so no network lookup at model load.
    """
    env = os.environ.copy()
    env["HF_HUB_CACHE"] = str(MIMIC_ROOT / "mimicmotion" / "models" / "svd_cache")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def _filter_env() -> dict:
    """Env for J1b filter helpers (trim_inactive_frames etc.).

    rtmlib lives in UniSignMimicTurbo; add it to PYTHONPATH + set
    RTMLIB_PARENT so `from rtmlib import Wholebody` resolves. Mirrors
    DGX `infer_dgx_total.sh`'s PYTHONPATH overlay.
    """
    env = _mimic_env()
    env["PYTHONPATH"] = f"{MIMIC_ROOT}:{env.get('PYTHONPATH', '')}"
    env["RTMLIB_PARENT"] = str(MIMIC_ROOT)
    return env


def _tg_env() -> dict:
    """Env for J2 tail_glitch — same rtmlib overlay as filter helpers."""
    return _filter_env()


def _sr_env() -> dict:
    """Env for J3 RealESR — overlay realesr_extras (pip --target install
    of basicsr + facexlib + realesrgan), mirroring DGX `infer_dgx_realesr.sh`.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REALESR_EXTRAS}:{env.get('PYTHONPATH', '')}"
    env["HF_HUB_OFFLINE"] = "1"
    return env


async def _run(cmd: list[str], cwd: Path | None = None,
               env: dict | None = None) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(cwd) if cwd else None,
        env=env,
    )
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace")


_EXTRACT_FRAMES_PY = '''
import sys, cv2
src, out_dir, stem = sys.argv[1], sys.argv[2], sys.argv[3]
cap = cv2.VideoCapture(src)
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imwrite(f"{out_dir}/{i}_{stem}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    i += 1
cap.release()
print(f"extracted {i} frames", flush=True)
'''


async def _stage_j1_filter(mimic_out: Path, sub_work: Path) -> tuple[Path | None, str]:
    """Apply trim_inactive_frames + reassemble. Returns (final_mp4, err).

    Pure pass-through (just copies mimic_out) when no filter is enabled,
    so callers always get back a single canonical "post-filter" path.
    """
    stem = "input"
    scratch = sub_work / "filter_scratch"
    raw_dir = scratch / "00_raw" / stem
    raw_dir.mkdir(parents=True, exist_ok=True)

    rc, out = await _run(
        [str(MIMIC_PYTHON), "-c", _EXTRACT_FRAMES_PY,
         str(mimic_out), str(raw_dir), stem],
        env=_filter_env(),
    )
    if rc != 0:
        return None, f"extract_frames: {out[-300:]}"

    current = scratch / "00_raw"
    scripts = MIMIC_ROOT / "scripts" / "sentence"
    step = 0

    if FILTER_HEAD_TAIL:
        step += 1
        nxt = scratch / f"{step:02d}_trim"
        rc, out = await _run(
            [str(MIMIC_PYTHON), str(scripts / "trim_inactive_frames.py"),
             "--frames-dir", str(current),
             "--output-dir", str(nxt),
             "--activity-threshold", str(ACTIVITY_THRESHOLD),
             "--margin", str(MARGIN),
             "--device", "cuda"],
            env=_filter_env(),
        )
        if rc != 0 or not (nxt / stem).exists():
            return None, f"trim: {out[-300:]}"
        current = nxt

    out_filter = sub_work / "output_filter"
    out_filter.mkdir(parents=True, exist_ok=True)
    rc, out = await _run(
        [str(MIMIC_PYTHON), str(scripts / "generate_videos_from_frames.py"),
         "--frames-dir", str(current),
         "--output-dir", str(out_filter),
         "--fps", str(FPS)],
        env=_filter_env(),
    )
    final = out_filter / f"{stem}.mp4"
    if rc != 0 or not final.exists():
        return None, f"reassemble: {out[-300:]}"
    return final, ""


async def _stage_j2_tail_glitch(in_mp4: Path, sub_work: Path) -> tuple[Path | None, str]:
    """Trim trailing glitched frames (rtmlib body/hand/wrist analysis)."""
    tg_dir = sub_work / "tg_out"
    tg_dir.mkdir(parents=True, exist_ok=True)
    rc, out = await _run(
        [str(MIMIC_PYTHON), str(TAIL_GLITCH_SCRIPT), str(in_mp4),
         "--apply", "--apply-dir", str(tg_dir),
         "--min-cats", str(TAIL_GLITCH_MIN_CATS)],
        env=_tg_env(),
    )
    if rc != 0:
        return None, f"tail_glitch: {out[-300:]}"
    trimmed = tg_dir / in_mp4.name
    if trimmed.exists():
        return trimmed, ""
    # Pass-through when no glitch detected — match DGX wrapper behavior.
    target = sub_work / "tail_glitch.mp4"
    import shutil
    shutil.copy2(in_mp4, target)
    return target, ""


async def _stage_j3_realesr(in_mp4: Path, sub_work: Path) -> tuple[Path | None, str]:
    """Real-ESRGAN x2 super-resolution (576x576 → 1152x1152)."""
    sr_out = sub_work / "sr.mp4"
    rc, out = await _run(
        [str(MIMIC_PYTHON), str(REALESR_SCRIPT),
         str(in_mp4), str(sr_out), str(REALESR_OUTSCALE)],
        cwd=REALESR_SCRIPT.parent,
        env=_sr_env(),
    )
    if rc != 0 or not sr_out.exists() or sr_out.stat().st_size == 0:
        return None, f"realesr: {out[-300:]}"
    return sr_out, ""


async def _process_one(video: Path, ref_image: Path, work_dir: Path,
                       videos_out: Path) -> dict:
    """Run J1 (mimic+filter) → J2 (tail_glitch) → J3 (SR) for one video."""
    sub_id = f"{video.stem}_{int(time.time()*1000) % 1000000}"
    sub_work = work_dir / sub_id
    in_videos = sub_work / "videos"
    out_dir = sub_work / "output"
    in_videos.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage video as `input.mp4` (matches DGX layout — batch_process picks any
    # mp4 in --videos dir; we mirror the DGX naming for consistency).
    staged = in_videos / "input.mp4"
    if staged.exists() or staged.is_symlink():
        staged.unlink()
    staged.symlink_to(video.resolve())

    t_start = time.time()

    # === J1a: MimicMotion mimic ===
    rc, out = await _run(
        [str(MIMIC_PYTHON), "batch_process.py",
         "--videos", str(in_videos),
         "--image", str(ref_image),
         "--output", str(out_dir),
         "--mode", "square", "--crop-anchor", "top",
         "--sample-stride", str(SAMPLE_STRIDE)],
        cwd=MIMIC_ROOT / "mimicmotion",
        env=_mimic_env(),
    )
    if rc != 0:
        wall = round(time.time() - t_start, 1)
        return _failed(video.name, "j1_mimic", out) | {"wall_sec": wall, "sub_id": sub_id}
    mimic_out = out_dir / "input_hiya.mp4"
    if not mimic_out.exists() or mimic_out.stat().st_size == 0:
        wall = round(time.time() - t_start, 1)
        return _failed(video.name, "j1_no_output", out) | {"wall_sec": wall, "sub_id": sub_id}

    # === J1b: filter (head/tail trim) ===
    filtered, err = await _stage_j1_filter(mimic_out, sub_work)
    if filtered is None:
        wall = round(time.time() - t_start, 1)
        return _failed(video.name, "j1_filter", err) | {"wall_sec": wall, "sub_id": sub_id}

    # === J2: tail_glitch ===
    tg_out, err = await _stage_j2_tail_glitch(filtered, sub_work)
    if tg_out is None:
        wall = round(time.time() - t_start, 1)
        return _failed(video.name, "j2_tail_glitch", err) | {"wall_sec": wall, "sub_id": sub_id}

    # === J3: RealESR ===
    sr_out, err = await _stage_j3_realesr(tg_out, sub_work)
    if sr_out is None:
        wall = round(time.time() - t_start, 1)
        return _failed(video.name, "j3_realesr", err) | {"wall_sec": wall, "sub_id": sub_id}

    wall = round(time.time() - t_start, 1)
    return {
        "filename": video.name,
        "status": "raw_done",
        "sub_id": sub_id,
        "raw_path": str(sr_out),
        "wall_sec": wall,
    }


async def _remux_one(rec: dict, videos_out: Path, ffmpeg_sem: asyncio.Semaphore) -> dict:
    if rec["status"] != "raw_done":
        return rec
    final = videos_out / rec["filename"]
    async with ffmpeg_sem:
        rc, out = await _run([
            str(FFMPEG), "-y", "-loglevel", "error",
            "-i", rec["raw_path"],
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(final),
        ])
    if rc != 0 or not final.exists() or final.stat().st_size == 0:
        rec.update(_failed(rec["filename"], "remux", out))
        return rec
    rec["status"] = "completed"
    rec["output_path"] = str(final)
    rec["output_size"] = final.stat().st_size
    return rec


VideoDoneCb = Callable[[dict], Awaitable[None] | None]


async def run_phase3_on_local(
    task_id: str,
    input_dir: Path,
    output_dir: Path,
    progress_cb: ProgressCb | None = None,
    on_video_done: VideoDoneCb | None = None,
) -> dict:
    """Run full Phase 3 locally for every *.mp4 in input_dir.

    Output layout and report keys mirror `run_phase3_on_dgx` so callers don't
    care which backend produced the report.

    `on_video_done(record)` is invoked after each video is fully processed
    (3-stage chain + h264 remux) — callers can incrementally publish results
    without waiting for the whole batch.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    videos_out = output_dir / "videos"
    videos_out.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_local_scratch"
    work_dir.mkdir(parents=True, exist_ok=True)

    if not DEFAULT_REF_IMAGE.exists():
        raise FileNotFoundError(f"reference image not found: {DEFAULT_REF_IMAGE}")
    if not (MIMIC_ROOT / "mimicmotion" / "batch_process.py").exists():
        raise FileNotFoundError(f"MIMIC_ROOT/mimicmotion/batch_process.py missing under {MIMIC_ROOT}")
    if not TAIL_GLITCH_SCRIPT.exists():
        raise FileNotFoundError(f"tail_glitch script missing: {TAIL_GLITCH_SCRIPT}")
    if not REALESR_SCRIPT.exists():
        raise FileNotFoundError(f"realesr script missing: {REALESR_SCRIPT}")
    if not REALESR_EXTRAS.exists():
        raise FileNotFoundError(f"realesr_extras dir missing: {REALESR_EXTRAS}")

    all_mp4 = sorted(input_dir.glob("*.mp4"))
    videos = [v for v in all_mp4 if not v.name.startswith("sentence_")]
    skipped = len(all_mp4) - len(videos)
    if skipped:
        logger.info(f"[{task_id}] Phase 3 (local): skipped {skipped} sentence_* videos")
    if not videos:
        raise RuntimeError(f"Phase 3 (local): no word videos found in {input_dir}")

    logger.info(f"[{task_id}] Phase 3 (local): processing {len(videos)} videos with {MIMIC_ROOT} "
                f"(3-stage chain: mimic+filter → tail_glitch → realesr x{REALESR_OUTSCALE})")
    last_emitted: list[float] = []
    t_start = time.time()
    await _emit_progress(progress_cb, 1.0, last_emitted)

    # Serial chain; remux + on_video_done callback inline so each video
    # becomes available to consumers as soon as it's ready (no batched wait).
    ffmpeg_sem = asyncio.Semaphore(FFMPEG_CONCURRENCY)
    fetched: list[dict] = []
    for i, video in enumerate(videos):
        rec = await _process_one(video, DEFAULT_REF_IMAGE, work_dir, videos_out)
        rec = await _remux_one(rec, videos_out, ffmpeg_sem)
        fetched.append(rec)
        pct = 5.0 + 90.0 * (i + 1) / len(videos)
        await _emit_progress(progress_cb, pct, last_emitted)
        if on_video_done is not None:
            try:
                result = on_video_done(rec)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"[{task_id}] on_video_done callback raised: {e}")

    # Cleanup scratch (non-fatal)
    try:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass

    n_total = len(videos)
    n_completed = sum(1 for r in fetched if r.get("status") == "completed")
    wall = round(time.time() - t_start, 1)
    report = {
        "task_id": task_id,
        "backend": "local",
        "input_videos": n_total,
        "transfer_success": n_completed,
        "transfer_failed": n_total - n_completed,
        "transfer_skipped": skipped,
        "videos_generated": n_completed,
        "interpolation_mode": "local",
        "total_wall_seconds": wall,
        "results": fetched,
    }
    (output_dir / "phase3_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )

    logger.info(f"[{task_id}] Phase 3 (local) done: {n_completed}/{n_total} videos "
                f"in {wall/60:.1f} min")
    await _emit_progress(progress_cb, 100.0, last_emitted)
    return report
