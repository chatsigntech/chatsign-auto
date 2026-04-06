"""Recognition API: list available models and WebSocket real-time inference."""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select

from backend.api.auth import get_current_user, decode_token
from backend.config import settings
from backend.database import get_session
from backend.models.task import PipelineTask
from backend.workers.recognition_session import find_best_checkpoint

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recognition", tags=["recognition"])


def _find_phase8_outputs(task_id: str) -> dict | None:
    """Check if a task has usable Phase 8 outputs. Returns info dict or None."""
    phase8_output = settings.SHARED_DATA_ROOT / task_id / "phase_8" / "output"
    if not phase8_output.exists():
        return None

    ckpt = find_best_checkpoint(phase8_output / "checkpoints")
    if ckpt is None:
        return None

    if not (phase8_output / "prototypes" / "prototypes.pt").exists():
        return None

    vocab_size = 0
    vocab_path = phase8_output / "vocab.json"
    if vocab_path.exists():
        try:
            with open(vocab_path) as f:
                vocab_data = json.load(f)
            vocab_size = len(vocab_data.get("token_to_id", {}))
        except Exception:
            pass

    return {
        "checkpoint": ckpt.name,
        "vocab_size": vocab_size,
    }


@router.get("/models")
def list_models(
    _user=Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """List tasks that have completed Phase 8 with usable model outputs."""
    tasks = session.exec(
        select(PipelineTask).where(
            PipelineTask.status.in_(["completed", "failed", "paused"])
        )
    ).all()

    models = []
    for task in tasks:
        info = _find_phase8_outputs(task.task_id)
        if info is None:
            continue
        models.append({
            "task_id": task.task_id,
            "task_name": task.name,
            "checkpoint": info["checkpoint"],
            "vocab_size": info["vocab_size"],
            "created_at": task.created_at.isoformat() if task.created_at else None,
        })

    return models


@router.websocket("/ws/{task_id}")
async def recognition_ws(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time sign language recognition.

    Protocol:
      1. Client connects -> server loads model -> sends {"type": "ready"}
      2. Client sends binary JPEG frames
      3. Server sends {"type": "prediction", ...} when new windows are available
      4. Client sends text {"type": "reset"} to clear session state
      5. On disconnect, session is cleaned up
    """
    await websocket.accept()

    token = websocket.query_params.get("token")
    if not token:
        await websocket.send_json({"type": "error", "message": "Missing token"})
        await websocket.close(code=4001)
        return

    username = decode_token(token)
    if username is None:
        await websocket.send_json({"type": "error", "message": "Invalid token"})
        await websocket.close(code=4001)
        return

    try:
        await websocket.send_json({"type": "loading", "message": "Loading model..."})

        from backend.workers.recognition_session import (
            load_model_bundle,
            RecognitionSession,
        )

        # Offload heavy model loading to a thread to avoid blocking the event loop
        bundle = await asyncio.to_thread(load_model_bundle, task_id, 0)
        session = RecognitionSession(bundle)

        await websocket.send_json({"type": "ready"})
        logger.info(f"Recognition session started for task {task_id} (user={username})")

    except FileNotFoundError as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close(code=4004)
        return
    except Exception as e:
        logger.exception(f"Failed to load model for task {task_id}")
        await websocket.send_json({
            "type": "error",
            "message": f"Model loading failed: {e}",
        })
        await websocket.close(code=4000)
        return

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                # Offload inference to thread to avoid blocking other connections
                result = await asyncio.to_thread(
                    session.process_frame, message["bytes"]
                )
                if result is not None:
                    await websocket.send_json(result)

            elif "text" in message and message["text"]:
                try:
                    cmd = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                if cmd.get("type") == "reset":
                    session.reset()
                    await websocket.send_json({
                        "type": "reset_ack",
                        "message": "Session reset",
                    })
                elif cmd.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong", **session.get_status()
                    })

    except WebSocketDisconnect:
        logger.info(f"Recognition session disconnected for task {task_id}")
    except Exception as e:
        logger.exception(f"Recognition WebSocket error for task {task_id}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        del session
        logger.info(f"Recognition session cleaned up for task {task_id}")
