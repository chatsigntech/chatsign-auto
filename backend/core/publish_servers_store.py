"""Publish-server profiles: file-backed store of {name, host, port, user,
password, default_target_dir} records used by the Phase 3 publish feature.

Plaintext on disk for now (TODO: encrypt at rest). The file path is
gitignored. Concurrency is single-process (in-memory lock + atomic rename).
"""
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_REQUIRED_FIELDS = {"name", "host", "port", "username", "password", "default_target_dir"}


class PublishServerStore:
    """Thread-safe file-backed list of server profiles, keyed by unique name."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _read_unlocked(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"[publish_servers] read failed ({e}); treating as empty")
            return []

    def _write_unlocked(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        fd = os.open(str(tmp), os.O_RDONLY); os.fsync(fd); os.close(fd)
        os.replace(tmp, self.path)

    def list_full(self) -> list[dict]:
        with self._lock:
            return list(self._read_unlocked())

    def get(self, name: str) -> dict | None:
        with self._lock:
            for r in self._read_unlocked():
                if r.get("name") == name:
                    return dict(r)
        return None

    def add(self, record: dict) -> None:
        missing = _REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"missing fields: {sorted(missing)}")
        with self._lock:
            records = self._read_unlocked()
            if any(r.get("name") == record["name"] for r in records):
                raise KeyError(f"server name already exists: {record['name']}")
            records.append({k: record[k] for k in _REQUIRED_FIELDS})
            self._write_unlocked(records)

    def update(self, name: str, patch: dict) -> dict:
        """Patch fields. patch values that are None are ignored (keep existing).
        'name' field cannot be changed. Returns the updated record."""
        with self._lock:
            records = self._read_unlocked()
            for i, r in enumerate(records):
                if r.get("name") == name:
                    for k, v in patch.items():
                        if k == "name" or v is None:
                            continue
                        if k in _REQUIRED_FIELDS:
                            r[k] = v
                    records[i] = r
                    self._write_unlocked(records)
                    return dict(r)
            raise KeyError(f"server not found: {name}")

    def delete(self, name: str) -> bool:
        with self._lock:
            records = self._read_unlocked()
            new = [r for r in records if r.get("name") != name]
            if len(new) == len(records):
                return False
            self._write_unlocked(new)
            return True


def redact(record: dict[str, Any]) -> dict[str, Any]:
    """Strip password before returning to the client."""
    return {k: v for k, v in record.items() if k != "password"}
