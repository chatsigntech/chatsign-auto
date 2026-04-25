"""DISABLED — legacy post-processor for the old local Phase 3 (Step 3.2).

Kept as a stub so the import path still resolves but any call fails loudly.
Phase 3 is fully delegated to DGX (see `backend/workers/phase3_dgx_client.py`)
and this module is no longer part of any live pipeline path.

Original behaviour: extract frames → trim inactive head/tail → resize 576×576
→ extract boundary frames → regenerate mp4 → h264 remux. Git history has
the full implementation if it ever needs to come back.
"""
from pathlib import Path


async def run_phase5_process(task_id: str, input_dir: Path, output_dir: Path) -> bool:
    raise NotImplementedError(
        "phase5_video_process.run_phase5_process is disabled — Phase 3 is "
        "handled by phase3_dgx_client.run_phase3_on_dgx."
    )
