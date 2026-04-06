import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, delete, select

from backend.database import engine, get_session
from backend.models.task import PipelineTask
from backend.models.phase import PhaseState
from backend.models.user import User
from backend.api.auth import get_current_user
from backend.core.phase_state_manager import PhaseStateManager
from backend.core.gpu_manager import GPUManager
from backend.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

NUM_PHASES = 8
GPU_PHASES = frozenset({3, 4, 5, 6, 8})  # Phases requiring GPU

gpu_manager = GPUManager(
    max_gpus=settings.MAX_GPUS,
    device_ids=settings.cuda_device_ids,
)

# Track running tasks for pause support (single-worker only)
_running_tasks: dict[str, bool] = {}


class SuggestSentencesRequest(BaseModel):
    topic: str
    count: int = 50


class DatasetVideo(BaseModel):
    text: str
    vid: str
    source: str  # "openasl" or "how2sign"


class TaskCreate(BaseModel):
    name: str
    input_text: str  # source text to convert to sign language
    batch_name: Optional[str] = None  # accuracy batch filter (e.g. "school_unmatch")
    source: Optional[str] = None  # "dataset" if from suggest-sentences
    dataset_videos: Optional[list[DatasetVideo]] = None


class TaskResponse(BaseModel):
    task_id: str
    name: str
    status: str
    current_phase: int
    created_at: datetime


def _get_task_or_404(session: Session, task_id: str) -> PipelineTask:
    task = session.exec(select(PipelineTask).where(PipelineTask.task_id == task_id)).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


def _fetch_task(session: Session, task_id: str) -> Optional[PipelineTask]:
    return session.exec(select(PipelineTask).where(PipelineTask.task_id == task_id)).first()


def _update_task_status(task_id: str, status: str, **fields):
    with Session(engine) as session:
        task = _fetch_task(session, task_id)
        if task:
            task.status = status
            task.updated_at = datetime.utcnow()
            for k, v in fields.items():
                setattr(task, k, v)
            session.add(task)
            session.commit()


def _run_pipeline_sync(task_id: str):
    """Sync wrapper to run the async pipeline in a background thread."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run_pipeline(task_id))
    finally:
        loop.close()


async def _run_pipeline(task_id: str):
    """Execute pipeline phases sequentially in the background."""
    from backend.workers.phase2_worker import run_phase2 as run_gloss_extract
    from backend.workers.phase2_push_glosses import run_phase2_push
    from backend.workers.phase1_worker import run_phase1 as run_video_collect
    from backend.workers.phase3_worker import run_phase3
    from backend.workers.phase4_person_transfer import run_phase4_transfer
    from backend.workers.phase5_video_process import run_phase5_process
    from backend.workers.phase6_framer import run_phase6_framer
    from backend.workers.phase4_segmentation_train import run_phase4_segmentation_train
    from backend.workers.phase5_segment import run_phase5_segment
    from backend.workers.phase7_augment import run_phase6_augment
    from backend.workers.phase7_aug_segment import run_phase7_aug_segment
    from backend.workers.phase8_training import run_phase8_training

    _running_tasks[task_id] = False
    data_root = settings.SHARED_DATA_ROOT / task_id

    try:
        with Session(engine) as session:
            task = _fetch_task(session, task_id)
            if not task:
                return
            start_phase = task.current_phase or 1
            task_config = json.loads(task.config_json) if task.config_json else {}
            batch_name = task_config.get("batch_name")
            task.status = "running"
            task.updated_at = datetime.utcnow()
            session.add(task)
            session.commit()

        phase_outputs = {i: data_root / f"phase_{i}" / "output" for i in range(1, NUM_PHASES + 1)}

        for phase_num in range(start_phase, NUM_PHASES + 1):
            if _running_tasks.get(task_id):
                _update_task_status(task_id, "paused", current_phase=phase_num)
                logger.info(f"[{task_id}] Pipeline paused at phase {phase_num}")
                return

            with Session(engine) as session:
                # Skip phases already completed (e.g. dataset mode skips Phase 2/3)
                phase_state = session.exec(
                    select(PhaseState).where(
                        PhaseState.task_id == task_id,
                        PhaseState.phase_num == phase_num,
                    )
                ).first()
                if phase_state and phase_state.status == "completed":
                    logger.info(f"[{task_id}] Phase {phase_num} already completed, skipping")
                    continue

                task = _fetch_task(session, task_id)
                if task:
                    task.current_phase = phase_num
                    task.updated_at = datetime.utcnow()
                    session.add(task)
                PhaseStateManager.mark_running(task_id, phase_num, session)
                session.commit()

            phase_input = data_root / f"phase_{phase_num}" / "input"
            phase_output = phase_outputs[phase_num]
            phase_input.mkdir(parents=True, exist_ok=True)
            phase_output.mkdir(parents=True, exist_ok=True)

            try:
                gpu_id = None
                if phase_num in GPU_PHASES:
                    gpu_id = gpu_manager.acquire(task_id)
                    if gpu_id is None:
                        gpu_id = 0

                summary = {}

                is_dataset = task_config.get("source") == "dataset"

                if phase_num == 1:
                    # Phase 1: Gloss extraction + push to accuracy
                    input_text = task_config.get("input_text", "")
                    glosses = await run_gloss_extract(task_id, input_text, output_dir=phase_output)
                    all_glosses = []
                    for g_list in glosses.values():
                        all_glosses.extend(g_list)

                    # Step 1.2: Push to accuracy
                    push_result = await run_phase2_push(
                        task_id, phase_output, phase_output,
                        batch_title=batch_name or task_config.get("batch_name", task_id),
                    )

                    summary = {
                        "input_text": input_text,
                        "sentences": list(glosses.keys()),
                        "sentence_count": len(glosses),
                        "unique_glosses": len(set(all_glosses)),
                        "glosses": list(set(all_glosses)),
                        "glosses_pushed": push_result.get("gloss_count", 0),
                        "batch_title": push_result.get("batch_title", ""),
                        "source": "dataset" if is_dataset else "user",
                    }

                    # Dataset mode: skip Phase 2, prepare videos directly
                    if is_dataset:
                        from backend.core.dataset_videos import prepare_dataset_videos
                        dataset_videos = task_config.get("dataset_videos", [])

                        phase_outputs[2].mkdir(parents=True, exist_ok=True)
                        result = prepare_dataset_videos(task_id, dataset_videos, phase_outputs[2])
                        with open(phase_outputs[2] / "summary.json", "w") as f:
                            json.dump({
                                "status": "dataset",
                                "videos_collected": result.get("video_count", 0),
                                "missing": result.get("missing", 0),
                            }, f, indent=2)
                        with Session(engine) as session:
                            PhaseStateManager.mark_completed(task_id, 2, session)

                        summary["dataset_videos"] = result.get("video_count", 0)
                        summary["dataset_missing"] = result.get("missing", 0)
                        logger.info(f"[{task_id}] Dataset mode: skipped Phase 2, "
                                    f"{result.get('video_count', 0)} videos prepared")
                    else:
                        # Normal mode: pause for human recording/review
                        _running_tasks[task_id] = True
                        summary["message"] = "Waiting for human recording and review"

                elif phase_num == 2:
                    # Phase 2: Video collection + annotation organization + preprocessing
                    result = await run_video_collect(task_id, phase_output,
                                                     batch_name=batch_name or task_id,
                                                     gloss_filter=phase_outputs[1])
                    videos_collected = result.get("video_count", 0)

                    # Step 2.2: Annotation organization
                    await run_phase3(task_id, phase_output, phase_outputs[1], phase_output)
                    ann_file = phase_output / "annotations.json"
                    ann_count = 0
                    if ann_file.exists():
                        with open(ann_file) as f:
                            ann_count = len(json.load(f))

                    # Step 2.3: Preprocess videos
                    from backend.workers.phase5_preprocess import preprocess_videos
                    raw_dir = phase_output / "videos"
                    preprocess_dir = phase_output / "preprocess"
                    preprocessed = await preprocess_videos(task_id, raw_dir, preprocess_dir)
                    preprocessed_count = len(list(preprocessed.glob("*.mp4"))) if preprocessed.is_dir() else 0

                    summary = {
                        "videos_collected": videos_collected,
                        "unique_sentences": len(result.get("sentences", [])),
                        "annotated_videos": ann_count,
                        "preprocessed_videos": preprocessed_count,
                    }

                elif phase_num == 3:
                    # Phase 3: Standard sign language video generation (independent branch)
                    # Step 3.1: Person transfer
                    p2_preprocessed = phase_outputs[2] / "preprocess" / "videos"
                    if not p2_preprocessed.exists():
                        p2_preprocessed = phase_outputs[2] / "videos"
                    transfer_dir = phase_output / "transfer"
                    transfer_dir.mkdir(parents=True, exist_ok=True)
                    await run_phase4_transfer(task_id, p2_preprocessed, transfer_dir, gpu_id=gpu_id)

                    # Step 3.2: Video processing
                    process_dir = phase_output / "processed"
                    process_dir.mkdir(parents=True, exist_ok=True)
                    await run_phase5_process(task_id, transfer_dir, process_dir)

                    # Step 3.3: Frame interpolation
                    await run_phase6_framer(task_id, process_dir, phase_output, gpu_id=gpu_id)

                    # Build summary from transfer report + final videos
                    report = transfer_dir / "phase4_report.json"
                    transfer_summary = {}
                    if report.exists():
                        r = json.loads(report.read_text())
                        s = r.get("summary", {})
                        transfer_summary = {
                            "input_videos": r.get("total_input", 0),
                            "transfer_success": s.get("success", 0) + s.get("retry_success", 0),
                            "transfer_failed": s.get("failed", 0),
                            "transfer_skipped": s.get("skipped_short", 0),
                        }

                    framer_report = phase_output / "phase6_report.json"
                    if framer_report.exists():
                        fr = json.loads(framer_report.read_text())
                        transfer_summary["interpolation_mode"] = fr.get("mode", "")
                        transfer_summary["videos_generated"] = fr.get("videos_generated", 0)
                    else:
                        vids = list((phase_output / "videos").rglob("*.mp4")) if (phase_output / "videos").exists() else []
                        transfer_summary["videos_generated"] = len(vids)

                    summary = transfer_summary

                elif phase_num == 4:
                    # Phase 4: Segmentation model training (SpaMo)
                    result = await run_phase4_segmentation_train(
                        task_id,
                        phase2_output=phase_outputs[2],
                        phase1_output=phase_outputs[1],
                        output_dir=phase_output,
                        gpu_id=gpu_id,
                    )
                    summary = {
                        "input_videos": result.get("input_videos", 0),
                        "features_extracted": result.get("features_extracted", 0),
                        "train_samples": result.get("train_samples", 0),
                        "val_samples": result.get("val_samples", 0),
                    }

                elif phase_num == 5:
                    # Phase 5: Segment original sentence videos
                    result = await run_phase5_segment(
                        task_id,
                        phase4_output=phase_outputs[4],
                        phase2_output=phase_outputs[2],
                        output_dir=phase_output,
                        gpu_id=gpu_id,
                    )
                    summary = {
                        "segmented_videos": result.get("segmented_videos", 0),
                        "total_segments": result.get("total_segments", 0),
                        "total_clips": result.get("total_clips", 0),
                    }

                elif phase_num == 6:
                    # Phase 6: Data augmentation (sentence + word + segment)
                    await run_phase6_augment(
                        task_id,
                        phase2_output=phase_outputs[2],
                        phase5_output=phase_outputs[5],
                        output_dir=phase_output,
                        gpu_id=gpu_id,
                    )
                    manifest_file = phase_output / "manifest.json"
                    if manifest_file.exists():
                        m = json.loads(manifest_file.read_text())
                        aug = m.get("augmentations", {})
                        summary = {
                            "input_sentences": m.get("input_sentences", 0),
                            "input_words": m.get("input_words", 0),
                            "input_segments": m.get("input_segments", 0),
                            "2d_cv": aug.get("2d_cv", {}).get("count", 0),
                            "temporal": aug.get("temporal", {}).get("count", 0),
                            "3d_views": aug.get("3d_views", {}).get("count", 0),
                            "identity": aug.get("identity", {}).get("count", 0),
                            "total_generated": m.get("total_generated", 0),
                        }

                elif phase_num == 7:
                    # Phase 7: Segment augmented sentence videos (no GPU)
                    result = await run_phase7_aug_segment(
                        task_id,
                        phase6_output=phase_outputs[6],
                        phase5_output=phase_outputs[5],
                        output_dir=phase_output,
                    )
                    summary = {
                        "input_aug_sentences": result.get("input_aug_sentences", 0),
                        "output_clips": result.get("output_clips", 0),
                    }

                elif phase_num == 8:
                    # Phase 8: Model training
                    await run_phase8_training(
                        task_id,
                        phase2_output=phase_outputs[2],
                        phase5_output=phase_outputs[5],
                        phase6_output=phase_outputs[6],
                        phase7_output=phase_outputs[7],
                        output_dir=phase_output,
                        gpu_id=gpu_id,
                    )
                    ckpts = list((phase_output / "checkpoints").glob("*.pth")) if (phase_output / "checkpoints").exists() else []
                    vids = len(list((phase_output / "videos").rglob("*.mp4"))) if (phase_output / "videos").exists() else 0
                    poses_raw = len(list((phase_output / "poses_raw").glob("*.pkl"))) if (phase_output / "poses_raw").exists() else 0
                    poses_filtered = len(list((phase_output / "poses_filtered").glob("*.pkl"))) if (phase_output / "poses_filtered").exists() else 0
                    poses_normed = len(list((phase_output / "poses_normed").glob("*.pkl"))) if (phase_output / "poses_normed").exists() else 0
                    corrupt_path = phase_output / "corrupt_poses.json"
                    corrupt_count = 0
                    if corrupt_path.exists():
                        corrupt_count = len(json.loads(corrupt_path.read_text()))
                    protos = list((phase_output / "prototypes").glob("*")) if (phase_output / "prototypes").exists() else []
                    dataset_files = []
                    for name in ["train.jsonl", "vocab.json"]:
                        if (phase_output / name).exists():
                            dataset_files.append(name)
                    summary = {
                        "input_videos": vids,
                        "poses_extracted": poses_raw,
                        "poses_filtered": poses_filtered,
                        "poses_normalized": poses_normed,
                        "poses_corrupt": corrupt_count,
                        "checkpoints": len(ckpts),
                        "prototypes": len(protos),
                        "dataset_files": dataset_files,
                    }

                # Write summary for this phase
                if summary:
                    with open(phase_output / "summary.json", "w") as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)

                if gpu_id is not None and phase_num in GPU_PHASES:
                    gpu_manager.release(gpu_id)

                with Session(engine) as session:
                    PhaseStateManager.mark_completed(task_id, phase_num, session)

            except Exception as e:
                if gpu_id is not None and phase_num in GPU_PHASES:
                    gpu_manager.release(gpu_id)
                with Session(engine) as session:
                    PhaseStateManager.mark_failed(task_id, phase_num, session, str(e))
                logger.error(f"[{task_id}] Phase {phase_num} failed: {e}")
                return

        _update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] Pipeline completed successfully")

    except Exception as e:
        logger.error(f"[{task_id}] Pipeline error: {e}")
        _update_task_status(task_id, "failed", error_message=str(e))
    finally:
        _running_tasks.pop(task_id, None)


def _start_pipeline_thread(task_id: str):
    t = threading.Thread(target=_run_pipeline_sync, args=(task_id,), daemon=True)
    t.start()


@router.post("/suggest-sentences")
def suggest_sentences(
    body: SuggestSentencesRequest,
    user: User = Depends(get_current_user),
):
    """Find sentences from OpenASL/How2Sign related to a topic via semantic search."""
    topic = body.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")
    count = max(1, min(body.count, 200))

    from backend.core.sentence_search import search
    results = search(topic, count=count)
    return {
        "topic": topic,
        "count": len(results),
        "details": results,
    }


@router.post("/", response_model=TaskResponse)
def create_task(
    body: TaskCreate,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    # Check for duplicate task name
    existing = session.exec(select(PipelineTask).where(PipelineTask.name == body.name)).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Task name '{body.name}' already exists")

    task_id = str(uuid.uuid4())[:8]
    config = {"input_text": body.input_text}
    if body.batch_name:
        config["batch_name"] = body.batch_name
    if body.source == "dataset" and body.dataset_videos:
        config["source"] = "dataset"
        config["dataset_videos"] = [v.model_dump() for v in body.dataset_videos]
    task = PipelineTask(
        task_id=task_id,
        name=body.name,
        config_json=json.dumps(config),
    )
    session.add(task)

    for phase_num in range(1, NUM_PHASES + 1):
        phase = PhaseState(task_id=task_id, phase_num=phase_num)
        session.add(phase)

    session.commit()
    session.refresh(task)
    return TaskResponse(
        task_id=task.task_id,
        name=task.name,
        status=task.status,
        current_phase=task.current_phase,
        created_at=task.created_at,
    )


@router.get("/")
def list_tasks(
    status: Optional[str] = None,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    query = select(PipelineTask)
    if status:
        query = query.where(PipelineTask.status == status)
    tasks = session.exec(query.order_by(PipelineTask.created_at.desc())).all()
    return {"tasks": tasks}


@router.get("/{task_id}")
def get_task(
    task_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    task = _get_task_or_404(session, task_id)
    phases = session.exec(
        select(PhaseState).where(PhaseState.task_id == task_id).order_by(PhaseState.phase_num)
    ).all()
    return {"task": task, "phases": phases}


# Text file extensions that can be displayed in the UI
_TEXT_EXTS = {".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml", ".log"}


@router.get("/{task_id}/phases/{phase_num}/accuracy-progress")
def get_accuracy_progress(
    task_id: str,
    phase_num: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Get recording and review progress from accuracy system for Phase 2."""
    task = _get_task_or_404(session, task_id)
    task_config = json.loads(task.config_json) if task.config_json else {}
    batch_name = task_config.get("batch_name", task_id)

    accuracy_data = settings.CHATSIGN_ACCURACY_DATA
    from backend.core.io_utils import read_jsonl

    # Count total glosses from accuracy batch file (actual uploaded count)
    batch_file = accuracy_data / "texts" / f"{batch_name}.jsonl"
    total_glosses = 0
    if batch_file.exists():
        with open(batch_file) as f:
            total_glosses = sum(1 for _ in f)

    # Count recordings (pending-videos matching batch)
    pending_path = accuracy_data / "reports" / "pending-videos.jsonl"
    batch_videos = []
    if pending_path.exists():
        for entry in read_jsonl(pending_path):
            fn = entry.get("videoFileName", "")
            if fn.startswith(batch_name + "_") or batch_name in fn:
                batch_videos.append(entry)

    # Count review decisions
    decisions_path = accuracy_data / "reports" / "review-decisions.jsonl"
    video_ids = {v.get("videoId") for v in batch_videos}
    approved = 0
    rejected = 0
    if decisions_path.exists():
        for entry in read_jsonl(decisions_path):
            if entry.get("videoId") in video_ids:
                if entry.get("decision") == "approved":
                    approved += 1
                elif entry.get("decision") == "rejected":
                    rejected += 1

    return {
        "total_glosses": total_glosses,
        "recorded": len(batch_videos),
        "reviewed": approved + rejected,
        "approved": approved,
        "rejected": rejected,
        "pending_review": len(batch_videos) - approved - rejected,
        "pending_recording": max(0, total_glosses - len(batch_videos)),
    }


@router.get("/{task_id}/phases/{phase_num}/summary")
def get_phase_summary(
    task_id: str,
    phase_num: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Get summary data for a completed phase."""
    _get_task_or_404(session, task_id)
    summary_path = settings.SHARED_DATA_ROOT / task_id / f"phase_{phase_num}" / "output" / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {}


@router.get("/{task_id}/phases/{phase_num}/files")
def get_phase_files(
    task_id: str,
    phase_num: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """List output files for a specific phase."""
    _get_task_or_404(session, task_id)
    phase_dir = settings.SHARED_DATA_ROOT / task_id / f"phase_{phase_num}" / "output"
    if not phase_dir.exists():
        return {"files": []}

    files = []
    for f in sorted(phase_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = str(f.relative_to(phase_dir))
        size = f.stat().st_size
        is_text = f.suffix.lower() in _TEXT_EXTS
        files.append({"path": rel, "size": size, "is_text": is_text})

    return {"files": files}


@router.get("/{task_id}/phases/{phase_num}/files/{file_path:path}")
def get_phase_file_content(
    task_id: str,
    phase_num: int,
    file_path: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Read content of a text file from phase output."""
    _get_task_or_404(session, task_id)
    full_path = settings.SHARED_DATA_ROOT / task_id / f"phase_{phase_num}" / "output" / file_path

    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    if full_path.suffix.lower() not in _TEXT_EXTS:
        raise HTTPException(status_code=400, detail="Only text files can be read")

    # Limit to 500KB
    if full_path.stat().st_size > 512_000:
        content = full_path.read_text(encoding="utf-8", errors="replace")[:512_000]
        content += "\n\n... (truncated)"
    else:
        content = full_path.read_text(encoding="utf-8", errors="replace")

    return {"path": file_path, "content": content}


@router.get("/{task_id}/phases/{phase_num}/download/{file_path:path}")
def download_phase_file(
    task_id: str,
    phase_num: int,
    file_path: str,
    session: Session = Depends(get_session),
):
    """Download any file from phase output."""
    _get_task_or_404(session, task_id)
    phase_out = settings.SHARED_DATA_ROOT / task_id / f"phase_{phase_num}" / "output"
    full_path = phase_out / file_path

    # Resolve symlinks
    if full_path.is_symlink():
        full_path = full_path.resolve()

    # Fallback: search recursively
    if not full_path.exists():
        for found in phase_out.rglob(file_path.split("/")[-1]):
            if found.is_file():
                full_path = found.resolve() if found.is_symlink() else found
                break

    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(full_path, filename=full_path.name)


@router.get("/{task_id}/phases/{phase_num}/videos")
def get_phase_videos(
    task_id: str,
    phase_num: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """List videos with gloss metadata for a phase.

    Reads manifest.json or annotations.json (Phase 4) to return video entries
    with sentence_text/glosses, filename, and a streamable URL.
    """
    _get_task_or_404(session, task_id)
    phase_dir = settings.SHARED_DATA_ROOT / task_id / f"phase_{phase_num}" / "output"

    # Try annotations.json first, then manifest.json
    entries = []
    ann_path = phase_dir / "annotations.json"
    manifest_path = phase_dir / "manifest.json"

    if ann_path.exists():
        with open(ann_path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                entries = data
    elif manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                entries = data

    # Check for preprocessed videos (Phase 2 preprocess output)
    preprocess_videos_dir = phase_dir / "preprocess" / "videos"
    has_preprocessed = preprocess_videos_dir.is_dir() and any(preprocess_videos_dir.glob("*.mp4"))

    # Build gloss lookup from Phase 2 annotations (for later phases)
    gloss_lookup = {}
    if not entries:
        task_root = settings.SHARED_DATA_ROOT / task_id
        p2_ann = task_root / "phase_2" / "output" / "annotations.json"
        if p2_ann.exists():
            with open(p2_ann, encoding="utf-8") as f:
                for ann in json.load(f):
                    stem = ann.get("filename", "").rsplit(".", 1)[0]
                    gloss_lookup[stem] = {
                        "glosses": ann.get("glosses", []),
                        "sentence_text": ann.get("sentence_text", ""),
                    }

    if entries:
        # Phase 1/2: build from manifest/annotations
        videos = []
        for entry in entries:
            filename = entry.get("filename", "")
            glosses = entry.get("glosses", [])
            if has_preprocessed:
                video_path = preprocess_videos_dir / filename
            else:
                video_path = phase_dir / "videos" / filename

            videos.append({
                "video_id": entry.get("video_id", ""),
                "filename": filename,
                "sentence_text": entry.get("sentence_text", ""),
                "glosses": glosses,
                "language": entry.get("language", "en"),
                "preprocessed": has_preprocessed,
                "exists": video_path.exists() or (video_path.is_symlink() and video_path.resolve().exists()),
                "size": video_path.stat().st_size if video_path.exists() else 0,
                "url": f"/api/tasks/{task_id}/phases/{phase_num}/video/{filename}",
            })
    else:
        # Phase 3-8: scan for mp4 files (in videos/ or subdirectories)
        mp4_files = sorted(phase_dir.rglob("*.mp4"))
        # Exclude intermediate files (logs, preprocess intermediates)
        mp4_files = [f for f in mp4_files if "preprocess" not in str(f.relative_to(phase_dir))]
        if not mp4_files:
            return {"videos": []}

        videos = []
        for vf in mp4_files:
            filename = vf.name
            stem = filename.rsplit(".", 1)[0]
            # Match gloss from Phase 2 annotations by checking if stem contains original stem
            matched = {}
            for orig_stem, info in gloss_lookup.items():
                if orig_stem in stem:
                    matched = info
                    break

            videos.append({
                "filename": filename,
                "sentence_text": matched.get("sentence_text", ""),
                "glosses": matched.get("glosses", []),
                "exists": True,
                "size": vf.stat().st_size,
                "url": f"/api/tasks/{task_id}/phases/{phase_num}/video/{filename}",
            })

    return {"videos": videos}


@router.get("/{task_id}/phases/{phase_num}/video/{filename:path}")
def stream_phase_video(
    task_id: str,
    phase_num: int,
    filename: str,
    session: Session = Depends(get_session),
):
    """Stream a video file from phase output."""
    _get_task_or_404(session, task_id)
    phase_out = settings.SHARED_DATA_ROOT / task_id / f"phase_{phase_num}" / "output"

    # Search for the video in multiple locations (including subdirectories)
    candidates = [
        phase_out / "preprocess" / "videos" / filename,
        phase_out / "videos" / filename,
        phase_out / filename,
    ]
    video_path = None
    for c in candidates:
        resolved = c.resolve() if c.is_symlink() else c
        if resolved.exists() and resolved.is_file():
            video_path = resolved
            break

    # Fallback: search recursively in subdirectories
    if not video_path:
        for found in phase_out.rglob(filename):
            if found.is_file():
                video_path = found.resolve() if found.is_symlink() else found
                break

    if not video_path:
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(video_path, media_type="video/mp4", filename=filename)


@router.post("/{task_id}/run")
def run_task(
    task_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Start pipeline execution for a task."""
    task = _get_task_or_404(session, task_id)
    if task.status == "running":
        raise HTTPException(status_code=409, detail="Task is already running")
    if task.status == "completed":
        raise HTTPException(status_code=409, detail="Task is already completed")

    _start_pipeline_thread(task_id)
    return {"message": "Pipeline started", "task_id": task_id}


@router.post("/{task_id}/pause")
def pause_task(
    task_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Pause a running task. Takes effect before the next phase starts."""
    task = _get_task_or_404(session, task_id)
    if task.status != "running":
        raise HTTPException(status_code=409, detail="Task is not running")

    _running_tasks[task_id] = True
    return {"message": "Pause signal sent", "task_id": task_id}


@router.post("/{task_id}/resume")
def resume_task(
    task_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Resume a paused task from its current phase."""
    task = _get_task_or_404(session, task_id)
    if task.status != "paused":
        raise HTTPException(status_code=409, detail="Task is not paused")

    _start_pipeline_thread(task_id)
    return {"message": "Pipeline resumed", "task_id": task_id}


@router.delete("/{task_id}")
def delete_task(
    task_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Delete a task and all its phase states."""
    task = _get_task_or_404(session, task_id)
    if task.status == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running task. Pause it first.")

    session.exec(delete(PhaseState).where(PhaseState.task_id == task_id))
    session.delete(task)
    session.commit()
    return {"message": "Task deleted", "task_id": task_id}
