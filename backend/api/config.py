import json
import logging
from functools import lru_cache

from fastapi import APIRouter, Body, Depends
from backend.config import settings
from backend.models.user import User
from backend.api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["config"])


@lru_cache(maxsize=1)
def _load_presets():
    config_path = settings.AUGMENTATION_CONFIG
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                data = yaml.safe_load(f)
            presets = []
            for key in ("light", "medium", "heavy", "custom"):
                if key in data and isinstance(data[key], dict):
                    presets.append({
                        "name": key,
                        "description": data[key].get("description", ""),
                        "estimated_variants_per_video": data[key].get("estimated_variants_per_video"),
                        "estimated_duration_per_video": data[key].get("estimated_duration_per_video"),
                    })
            if presets:
                return presets
        except Exception as e:
            logger.warning(f"Failed to parse augmentation config: {e}")

    return [
        {"name": "light", "description": "Quick validation"},
        {"name": "medium", "description": "Standard augmentation"},
        {"name": "heavy", "description": "Full augmentation for production"},
        {"name": "custom", "description": "Custom configuration"},
    ]


@router.get("/presets")
def get_presets(user: User = Depends(get_current_user)):
    """Return available augmentation presets from config."""
    return {"presets": _load_presets()}


@router.get("/augmentation")
async def get_augmentation_config():
    """Return the current augmentation config. Generate defaults if file missing."""
    config_path = settings.AUGMENTATION_CONFIG_PATH
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    else:
        # Return empty defaults - the file should be created by setup
        return {"error": "Config file not found", "path": str(config_path)}


@router.put("/augmentation")
async def update_augmentation_config(config: dict = Body(...)):
    """Update augmentation config."""
    config_path = settings.AUGMENTATION_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config


@router.get("/gpu")
def get_gpu_status(user: User = Depends(get_current_user)):
    """Return GPU availability info."""
    from backend.api.tasks import gpu_manager
    return {
        "max_gpus": settings.MAX_GPUS,
        "device_ids": settings.cuda_device_ids,
        "available": gpu_manager.available,
    }
