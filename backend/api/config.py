import json
import logging

from fastapi import APIRouter, Body, Depends
from backend.config import settings
from backend.models.user import User
from backend.api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["config"])


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


@router.get("/accuracy-url")
def get_accuracy_url():
    """Return the accuracy system URL for frontend navigation."""
    return {"url": settings.CHATSIGN_ACCURACY_URL}


@router.get("/gpu")
def get_gpu_status(user: User = Depends(get_current_user)):
    """Return GPU availability info."""
    from backend.api.tasks import gpu_manager
    return {
        "max_gpus": settings.MAX_GPUS,
        "device_ids": settings.cuda_device_ids,
        "available": gpu_manager.available,
    }
