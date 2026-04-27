"""Append-only JSONL log of Phase 3 publish attempts.

One record per `POST /api/tasks/{task_id}/phases/3/publish` call (success or
failure of any kind), so the user can see what was published when, to which
server, with what outcome.

Failure semantics: append errors are logged at WARNING and swallowed — a
publish itself is not failed by an audit-log failure.
"""
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class PublishHistoryStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[publish_history] append failed: {e}")

    def list_for_task(self, task_id: str, limit: int = 50) -> list[dict]:
        """Return most-recent-first records for the given task_id, capped at `limit`."""
        if not self.path.exists():
            return []
        out: list[dict] = []
        try:
            with self._lock, open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("task_id") == task_id:
                        out.append(r)
        except Exception as e:
            logger.warning(f"[publish_history] read failed: {e}")
            return []
        out.reverse()
        return out[:limit]
