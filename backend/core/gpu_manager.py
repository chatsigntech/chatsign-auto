import logging
from typing import Optional

from backend.core.gpu_auto_parallel import get_gpu_info

logger = logging.getLogger(__name__)

# Minimum free GPU memory (MB) required before assigning a GPU
MIN_FREE_MB = 8000


class GPUManager:
    """Round-robin GPU assignment with actual memory check."""

    def __init__(self, max_gpus: int = 1, device_ids: Optional[list[int]] = None):
        self.device_ids = device_ids or list(range(max_gpus))
        self._next = 0
        self._in_use: dict[int, str] = {}

    def acquire(self, task_id: str, min_free_mb: int = MIN_FREE_MB) -> Optional[int]:
        for _ in range(len(self.device_ids)):
            gpu_id = self.device_ids[self._next]
            self._next = (self._next + 1) % len(self.device_ids)
            if gpu_id in self._in_use:
                continue
            info = get_gpu_info(gpu_id)
            free = info["free_mb"] if info else 0
            if free < min_free_mb:
                logger.warning(f"GPU {gpu_id} has only {free}MB free (<{min_free_mb}MB), skipping")
                continue
            self._in_use[gpu_id] = task_id
            logger.info(f"GPU {gpu_id} assigned to task {task_id} ({free}MB free)")
            return gpu_id
        return None

    def release(self, gpu_id: int):
        self._in_use.pop(gpu_id, None)
        logger.info(f"GPU {gpu_id} released")

    @property
    def available(self) -> list[int]:
        return [g for g in self.device_ids if g not in self._in_use]
