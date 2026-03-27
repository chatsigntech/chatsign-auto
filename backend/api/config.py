import logging
from fastapi import APIRouter, Depends
from backend.config import settings
from backend.models.user import User
from backend.api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("/presets")
def get_presets(user: User = Depends(get_current_user)):
    """Return available augmentation presets from config."""
    config_path = settings.AUGMENTATION_CONFIG
    presets = []

    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                data = yaml.safe_load(f)
            for key in ("light", "medium", "heavy", "custom"):
                if key in data:
                    entry = {"name": key}
                    if isinstance(data[key], dict):
                        entry["description"] = data[key].get("description", "")
                        entry["estimated_variants_per_video"] = data[key].get(
                            "estimated_variants_per_video", None
                        )
                        entry["estimated_duration_per_video"] = data[key].get(
                            "estimated_duration_per_video", None
                        )
                    presets.append(entry)
        except Exception as e:
            logger.warning(f"Failed to parse augmentation config: {e}")

    if not presets:
        presets = [
            {"name": "light", "description": "Quick validation"},
            {"name": "medium", "description": "Standard augmentation"},
            {"name": "heavy", "description": "Full augmentation for production"},
            {"name": "custom", "description": "Custom configuration"},
        ]

    return {"presets": presets}


@router.get("/gpu")
def get_gpu_status(user: User = Depends(get_current_user)):
    """Return GPU availability info."""
    from backend.api.tasks import gpu_manager
    return {
        "max_gpus": settings.MAX_GPUS,
        "device_ids": [int(d) for d in settings.CUDA_VISIBLE_DEVICES.split(",") if d.strip()],
        "available": gpu_manager.available,
    }
