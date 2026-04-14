import asyncio
import logging
import os
import signal
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Registry: task_id -> set of active subprocess PIDs
_active_pids: dict[str, set[int]] = {}


def register_pid(task_id: str, pid: int):
    _active_pids.setdefault(task_id, set()).add(pid)


def unregister_pid(task_id: str, pid: int):
    if task_id in _active_pids:
        _active_pids[task_id].discard(pid)
        if not _active_pids[task_id]:
            del _active_pids[task_id]


def kill_task_subprocesses(task_id: str):
    """Kill all active subprocesses for a task."""
    pids = _active_pids.pop(task_id, set())
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Killed subprocess {pid} for task {task_id}")
        except ProcessLookupError:
            pass


async def run_subprocess(
    cmd: list[str],
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    timeout: Optional[int] = None,
    log_to_file: bool = False,
    task_id: Optional[str] = None,
) -> tuple[int, str, str]:
    """Run an external process and return (returncode, stdout, stderr).

    Args:
        log_to_file: If True, redirect stdout/stderr to temp files instead of
            PIPE. Use this for long-running processes that produce lots of
            output to avoid PIPE buffer deadlocks.
        task_id: If set, register the subprocess PID so it can be killed
            on task restart.
    """
    logger.info(f"Running: {' '.join(cmd)}")

    if log_to_file:
        stdout_path = tempfile.mktemp(suffix='.stdout')
        stderr_path = tempfile.mktemp(suffix='.stderr')
        with open(stdout_path, 'w') as fout, open(stderr_path, 'w') as ferr:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=cwd, env=env, stdout=fout, stderr=ferr,
            )
            if task_id:
                register_pid(task_id, proc.pid)
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return -1, "", "Process timed out"
            finally:
                if task_id:
                    unregister_pid(task_id, proc.pid)
        stdout_text = Path(stdout_path).read_text(errors='replace')[-10000:]
        stderr_text = Path(stderr_path).read_text(errors='replace')[-10000:]
        Path(stdout_path).unlink(missing_ok=True)
        Path(stderr_path).unlink(missing_ok=True)
        return proc.returncode, stdout_text, stderr_text

    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if task_id:
        register_pid(task_id, proc.pid)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return -1, "", "Process timed out"
    finally:
        if task_id:
            unregister_pid(task_id, proc.pid)

    return proc.returncode, stdout.decode(), stderr.decode()
