"""Video filename naming convention shared by run_*.py and inject_*.py.

Kept stdlib-only so run_*.py (DGX driver scripts) can import it without
loading the chatsign-pipeline TextPipeline / pandas / spacy stack.
"""
from __future__ import annotations

import hashlib


def video_filename(video_id: str) -> str:
    """md5(videoId)[:10] + '_hiya.mp4' — unified naming across DGX-rendered batches.

    Why: matches publishService.js convention and avoids the legacy `<sid>.mp4`
    namespace collision when batches share sentence indices.
    """
    return hashlib.md5(video_id.encode()).hexdigest()[:10] + "_hiya.mp4"
