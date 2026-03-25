import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GPUManager:
    """Simple round-robin GPU assignment."""

    def __init__(self, max_gpus: int = 1, device_ids: Optional[list[int]] = None):
        self.device_ids = device_ids or list(range(max_gpus))
        self._next = 0
        self._in_use: dict[int, str] = {}

    def acquire(self, task_id: str) -> Optional[int]:
        for _ in range(len(self.device_ids)):
            gpu_id = self.device_ids[self._next]
            self._next = (self._next + 1) % len(self.device_ids)
            if gpu_id not in self._in_use:
                self._in_use[gpu_id] = task_id
                logger.info(f"GPU {gpu_id} assigned to task {task_id}")
                return gpu_id
        return None

    def release(self, gpu_id: int):
        self._in_use.pop(gpu_id, None)
        logger.info(f"GPU {gpu_id} released")

    @property
    def available(self) -> list[int]:
        return [g for g in self.device_ids if g not in self._in_use]
