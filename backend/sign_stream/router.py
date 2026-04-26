"""FastAPI router for the speech→sign streaming test feature.

Single WebSocket endpoint:
  1. Client opens WS, sends a JSON text frame: {"text": "..."}
  2. Server replies with a JSON metadata frame: {"glosses": [...], "session_id": "..."}
  3. Server then streams binary fMP4 chunks until ffmpeg exits.
  4. Server sends a final JSON frame {"done": true} and closes.
"""
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from . import gloss_index, text_to_glosses
from .stream_session import StreamSession, new_session_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sign-stream", tags=["sign-stream"])


class PreviewRequest(BaseModel):
    text: str


@router.get("/health")
def health():
    return {"index_size": len(gloss_index.get_index())}


@router.post("/preview")
def preview(req: PreviewRequest):
    text = req.text.strip()
    if not text:
        return {"glosses": [], "plan": []}
    glosses = text_to_glosses.text_to_glosses(text)
    plan, _ = gloss_index.build_plan(glosses)
    return {"glosses": glosses, "plan": plan}


@router.websocket("/ws")
async def stream_ws(ws: WebSocket):
    await ws.accept()
    session: StreamSession | None = None
    try:
        msg = await ws.receive_text()
        try:
            req = json.loads(msg)
        except json.JSONDecodeError:
            await ws.send_json({"error": "expected JSON {text:'...'}"})
            return
        text = (req.get("text") or "").strip()
        if not text:
            await ws.send_json({"error": "empty text"})
            return

        glosses = text_to_glosses.text_to_glosses(text)
        plan, clips = gloss_index.build_plan(glosses)
        if not clips:
            await ws.send_json({"error": "no clips resolved", "glosses": glosses})
            return
        sid = new_session_id()
        await ws.send_json({"session_id": sid, "glosses": glosses, "plan": plan})

        session = StreamSession(sid, clips)
        await session.start()
        async for chunk in session.iter_chunks():
            await ws.send_bytes(chunk)
        await ws.send_json({"done": True})
    except WebSocketDisconnect:
        logger.info("client disconnected")
    except ValueError as e:
        logger.warning("sign-stream invalid input: %s", e)
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass
    except Exception:
        logger.exception("sign-stream WS error")
        try:
            await ws.send_json({"error": "internal error"})
        except Exception:
            pass
    finally:
        if session is not None:
            await session.close()
        try:
            await ws.close()
        except Exception:
            pass
