"""Phase 4: Video preprocessing using UniSignMimicTurbo scripts."""
import logging
from pathlib import Path

from backend.config import settings
from backend.core.subprocess_runner import run_subprocess

logger = logging.getLogger(__name__)

SCRIPTS_DIR = settings.UNISIGN_PATH / "scripts" / "sentence"

STEPS = [
    ("4.1", "extract_all_frames_seq.py", "Extracting frames"),
    ("4.2", "filter_duplicate_frames.py", "Filtering duplicates"),
    ("4.3", "filter_frames_by_pose.py", "Filtering by pose"),
    ("4.4", "resize_frames.py", "Resizing frames"),
    ("4.5", "generate_videos_from_frames.py", "Generating videos"),
]


async def run_phase4(task_id: str, input_dir: Path, output_dir: Path) -> bool:
    """Run 5-step video preprocessing pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    current_input = input_dir

    for step_id, script_name, description in STEPS:
        logger.info(f"[{task_id}] Phase 4 Step {step_id}: {description}")
        step_output = output_dir / f"step_{step_id}"
        step_output.mkdir(parents=True, exist_ok=True)

        script_path = SCRIPTS_DIR / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        returncode, stdout, stderr = await run_subprocess(
            ["python", str(script_path), "--input-dir", str(current_input), "--output-dir", str(step_output)],
            cwd=settings.UNISIGN_PATH,
        )
        if returncode != 0:
            raise RuntimeError(f"Phase 4 Step {step_id} failed: {stderr}")

        current_input = step_output

    logger.info(f"[{task_id}] Phase 4 completed")
    return True
