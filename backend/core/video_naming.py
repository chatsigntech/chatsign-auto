"""Canonical filename for DGX-rendered videos in pending-videos / terminal sync.

Mirrors `chatsign-accuracy/backend/services/publishService.js` (the JS side
that ships these files to the terminal) and is duplicated by
`dgx-pipeline-test/_naming.py` (a stdlib-only twin that DGX driver scripts
import without paying the chatsign-pipeline cold-start). All three must
stay in sync.
"""
from __future__ import annotations

import hashlib


def video_filename(video_id: str) -> str:
    return hashlib.md5(video_id.encode()).hexdigest()[:10] + "_hiya.mp4"
