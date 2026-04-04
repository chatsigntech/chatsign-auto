import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


async def run_subprocess(
    cmd: list[str],
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    timeout: Optional[int] = None,
    log_to_file: bool = False,
) -> tuple[int, str, str]:
    """Run an external process and return (returncode, stdout, stderr).

    Args:
        log_to_file: If True, redirect stdout/stderr to temp files instead of
            PIPE. Use this for long-running processes that produce lots of
            output to avoid PIPE buffer deadlocks.
    """
    logger.info(f"Running: {' '.join(cmd)}")

    if log_to_file:
        stdout_path = tempfile.mktemp(suffix='.stdout')
        stderr_path = tempfile.mktemp(suffix='.stderr')
        with open(stdout_path, 'w') as fout, open(stderr_path, 'w') as ferr:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=cwd, env=env, stdout=fout, stderr=ferr,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return -1, "", "Process timed out"
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
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return -1, "", "Process timed out"

    return proc.returncode, stdout.decode(), stderr.decode()
