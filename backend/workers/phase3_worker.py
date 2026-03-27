"""Phase 3: Organize annotations - merge Phase 1 manifest with Phase 2 glosses.

Creates the directory structure expected by Phase 4 (video preprocessing).
"""
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


async def run_phase3(task_id: str, phase1_output: Path, phase2_output: Path, output_dir: Path) -> bool:
    """
    Organize Phase 1 videos + Phase 2 glosses into Phase 4 input format.

    Expected inputs:
        phase1_output/manifest.json  — video file list with sentence mappings
        phase1_output/videos/        — symlinked/copied video files
        phase2_output/glosses.json   — sentence → gloss mapping from Phase 2

    Output structure (for Phase 4):
        output_dir/videos/           — organized video files
        output_dir/annotations.json  — merged annotations (video → sentence → glosses)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_out = output_dir / "videos"
    videos_out.mkdir(exist_ok=True)

    # Load Phase 1 manifest
    manifest_path = phase1_output / "manifest.json"
    if not manifest_path.exists():
        logger.warning(f"[{task_id}] Phase 3: No manifest.json from Phase 1")
        return True

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Load Phase 2 glosses (may not exist if Phase 2 had no input)
    glosses = {}
    glosses_path = phase2_output / "glosses.json"
    if glosses_path.exists():
        with open(glosses_path, encoding="utf-8") as f:
            glosses = json.load(f)

    # Merge: for each video, attach sentence + extracted glosses
    annotations = []
    for entry in manifest:
        filename = entry["filename"]
        sentence = entry.get("sentence_text", "")
        src = phase1_output / "videos" / filename

        if src.exists():
            dst = videos_out / filename
            if dst.exists():
                dst.unlink()
            # Symlink to Phase 1's video (which itself may symlink to accuracy data)
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)

        annotations.append({
            "video_id": entry.get("video_id"),
            "filename": filename,
            "sentence_id": entry.get("sentence_id"),
            "sentence_text": sentence,
            "language": entry.get("language", "en"),
            "glosses": glosses.get(sentence, []),
        })

    # Write merged annotations
    annotations_path = output_dir / "annotations.json"
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    logger.info(f"[{task_id}] Phase 3 completed: {len(annotations)} annotated videos")
    return True
