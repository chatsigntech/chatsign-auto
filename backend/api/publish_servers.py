"""CRUD for publish-server profiles. Side-branch feature; failure here
never affects the pipeline."""
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.auth import get_current_user
from backend.core.publish_servers_store import PublishServerStore, redact
from backend.core.validation_patterns import HOST_OK, NAME_OK, PATH_OK, USER_OK
from backend.models.user import User

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/publish-servers", tags=["publish-servers"])

# Lazy-init store so module import is cheap (and main.py try/except can catch
# any path-related issue at request time, not import time).
_store: PublishServerStore | None = None


def get_publish_servers_store() -> PublishServerStore:
    global _store
    if _store is None:
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent.parent
        _store = PublishServerStore(repo_root / "backend" / "data" / "publish_servers.json")
    return _store


# ── Pydantic models ──────────────────────────────────────────────────
class PublishServerCreate(BaseModel):
    name: str
    host: str
    port: int = 22
    username: str
    password: str
    default_target_dir: str


class PublishServerUpdate(BaseModel):
    host: str | None = None
    port: int | None = None
    username: str | None = None
    password: str | None = None  # None / "" = keep existing
    default_target_dir: str | None = None


def _validate_fields(*, host=None, port=None, username=None, target_dir=None):
    """Validate any provided (non-None) field against shell-injection-safe whitelists.

    The remote shell on the SCP target evaluates the path component of
    `user@host:/path`, so even with subprocess.run (no local shell) these
    fields can be exploited. Patterns are intentionally narrow.
    """
    if host is not None and not HOST_OK.match(host):
        raise HTTPException(400, "invalid host (allowed: A-Za-z0-9.-)")
    if port is not None and not (1 <= port <= 65535):
        raise HTTPException(400, "invalid port")
    if username is not None and not USER_OK.match(username):
        raise HTTPException(400, "invalid username (allowed: A-Za-z0-9._-)")
    if target_dir is not None and not PATH_OK.match(target_dir):
        raise HTTPException(400, "target dir must be absolute path matching [A-Za-z0-9/_.-]")


# ── endpoints ─────────────────────────────────────────────────────────
@router.get("")
def list_servers(user: User = Depends(get_current_user)):
    return [redact(r) for r in get_publish_servers_store().list_full()]


@router.post("", status_code=201)
def add_server(body: PublishServerCreate, user: User = Depends(get_current_user)):
    if not NAME_OK.match(body.name):
        raise HTTPException(400, "invalid name (allowed: A-Za-z0-9 ._- up to 64 chars)")
    _validate_fields(host=body.host, port=body.port, username=body.username,
                     target_dir=body.default_target_dir)
    if not body.password:
        raise HTTPException(400, "password required")
    try:
        get_publish_servers_store().add(body.model_dump())
    except KeyError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    logger.info(f"publish-server added: name={body.name} host={body.host}")
    return redact(body.model_dump())


@router.put("/{name}")
def update_server(name: str, body: PublishServerUpdate, user: User = Depends(get_current_user)):
    if not NAME_OK.match(name):
        raise HTTPException(400, "invalid name")
    _validate_fields(host=body.host, port=body.port, username=body.username,
                     target_dir=body.default_target_dir)
    # Empty-string password from the UI means "don't change"
    patch = body.model_dump()
    if patch.get("password") == "":
        patch["password"] = None
    try:
        updated = get_publish_servers_store().update(name, patch)
    except KeyError as e:
        raise HTTPException(404, str(e))
    logger.info(f"publish-server updated: name={name}")
    return redact(updated)


@router.delete("/{name}")
def delete_server(name: str, user: User = Depends(get_current_user)):
    if not NAME_OK.match(name):
        raise HTTPException(400, "invalid name")
    if not get_publish_servers_store().delete(name):
        raise HTTPException(404, "server not found")
    logger.info(f"publish-server deleted: name={name}")
    return {"deleted": name}
