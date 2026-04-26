"""Phase 3 publish-to-remote — sshpass + scp the approved videos and a
generated gloss.csv to a user-specified remote directory.

Runs strictly out-of-band of the pipeline: failure here only affects the
publish action, never the Phase 3 worker or any other phase.

Security notes:
- Password is passed via SSHPASS env var to the child only (never on
  command line, never logged, never written to disk).
- target_dir / username / host must be pre-validated by the caller.
- Persistent known_hosts under ~/.ssh/known_hosts_chatsign_publish gives
  TOFU semantics (first connection accepts, subsequent verify).
"""
import csv
import io
import logging
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


_KNOWN_HOSTS = Path.home() / ".ssh" / "known_hosts_chatsign_publish"
_GLOSS_HEADER = ["ref", "word", "sourceid", "synset_id", "gloss", "alternate_words"]
_PER_FILE_TIMEOUT_SEC = 600


def _build_gloss_csv(approved: list[dict]) -> str:
    """ASL-27K-compatible 6-column CSV. Empty cols filled blank."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(_GLOSS_HEADER)
    for v in approved:
        w.writerow([v["filename"], v["word"], "", "", "", ""])
    return buf.getvalue()


def _ssh_opts(port: int) -> list[str]:
    _KNOWN_HOSTS.parent.mkdir(parents=True, exist_ok=True)
    return [
        "-P", str(port),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={_KNOWN_HOSTS}",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=15",
    ]


def _sanitize_err(stderr: str, password: str) -> str:
    """Strip anything that might contain the password (paranoia) + truncate."""
    cleaned = (stderr or "").replace(password, "***") if password else (stderr or "")
    return cleaned.strip()[:300]


def _scp_one(local_path: str, remote_target: str, port: int,
             env: dict, password: str) -> tuple[bool, str]:
    """Run a single scp; returns (ok, error_msg_or_empty)."""
    cmd = ["sshpass", "-e", "scp", *_ssh_opts(port), local_path, remote_target]
    try:
        rc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            timeout=_PER_FILE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {_PER_FILE_TIMEOUT_SEC}s"
    except FileNotFoundError as e:
        return False, f"sshpass or scp not found: {e}"
    if rc.returncode == 0:
        return True, ""
    return False, _sanitize_err(rc.stderr, password)


def publish_to_remote(
    approved: list[dict],
    accuracy_data_root: Path,
    host: str,
    port: int,
    username: str,
    password: str,
    target_dir: str,
) -> dict:
    """Upload all approved videos + a generated gloss.csv to remote target_dir.

    `approved` items must each have keys: filename, word, videoPath (relative
    to accuracy_data_root, may start with '/').

    Returns a dict suitable for direct JSON response:
        {success: int, failed: int, total_videos: int, gloss_uploaded: bool,
         errors: [{filename, msg}, ...]}
    Never raises — exceptions are converted to error entries.
    """
    if not approved:
        return {"success": 0, "failed": 0, "total_videos": 0,
                "gloss_uploaded": False, "errors": [],
                "note": "no approved videos to publish"}

    env = {**os.environ, "SSHPASS": password}
    # target_dir was validated by caller; still quote for safety in remote spec
    remote_base = f"{username}@{host}:{shlex.quote(target_dir.rstrip('/'))}/"

    errors: list[dict] = []
    success = 0
    gloss_uploaded = False

    gloss_csv = _build_gloss_csv(approved)
    gloss_tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tf:
            tf.write(gloss_csv)
            gloss_tmp = tf.name
        ok, msg = _scp_one(gloss_tmp, remote_base + "gloss.csv", port, env, password)
        if ok:
            gloss_uploaded = True
            logger.info(f"[publish] gloss.csv uploaded ({len(approved)} entries)")
        else:
            # Gloss failure is fatal — abort to avoid partial state.
            errors.append({"filename": "gloss.csv", "msg": msg})
            return {"success": 0, "failed": 1 + len(approved),
                    "total_videos": len(approved), "gloss_uploaded": False,
                    "errors": errors,
                    "note": "aborted: gloss.csv upload failed"}
    finally:
        if gloss_tmp:
            try: Path(gloss_tmp).unlink(missing_ok=True)
            except Exception: pass

    for v in approved:
        rel = (v.get("videoPath") or "").lstrip("/")
        if not rel:
            errors.append({"filename": v.get("filename", "?"), "msg": "missing videoPath"})
            continue
        src = accuracy_data_root / rel
        if not src.exists():
            errors.append({"filename": v["filename"], "msg": f"local file missing: {src}"})
            continue
        ok, msg = _scp_one(str(src), remote_base + v["filename"], port, env, password)
        if ok:
            success += 1
        else:
            errors.append({"filename": v["filename"], "msg": msg})

    return {
        "success": success,
        "failed": len(errors),
        "total_videos": len(approved),
        "gloss_uploaded": gloss_uploaded,
        "errors": errors,
    }
