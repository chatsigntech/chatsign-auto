"""Per-request ffmpeg streaming session.

Takes a flat list of mp4 paths, runs them through one ffmpeg subprocess that
concats + transcodes to a unified profile + emits fragmented MP4 chunks on
stdout. The router consumes those chunks and pushes them over WebSocket so
the browser MediaSource can `appendBuffer` continuously.

One ffmpeg per session — no pool, no reuse (this is a test-only feature).
"""
import asyncio
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FFMPEG = _REPO_ROOT / "bin" / "ffmpeg"

# Unified output profile — chosen to play back identically regardless of
# input source (ASL-27K @ varied res, letters @ 576x576, submissions @ phone res).
TARGET_W = 576
TARGET_H = 576
TARGET_FPS = 25
CHUNK_BYTES = 64 * 1024


class StreamSession:
    def __init__(self, session_id: str, clips: list[Path]):
        self.session_id = session_id
        self.clips = list(clips)
        self._proc: asyncio.subprocess.Process | None = None
        self._tmpdir: Path | None = None
        self._concat_list: Path | None = None

    def _write_concat_list(self) -> Path:
        self._tmpdir = Path(tempfile.mkdtemp(prefix=f"sign_stream_{self.session_id}_"))
        listfile = self._tmpdir / "list.txt"
        with open(listfile, "w") as f:
            for clip in self.clips:
                # ffmpeg concat demuxer requires escaped paths; safest is single-quote
                f.write(f"file '{clip.as_posix()}'\n")
        self._concat_list = listfile
        return listfile

    async def start(self) -> None:
        if not self.clips:
            raise ValueError("no clips to stream")
        listfile = self._write_concat_list()
        cmd = [
            str(FFMPEG),
            "-hide_banner", "-loglevel", "warning",
            "-fflags", "+genpts+igndts",
            "-f", "concat", "-safe", "0", "-i", str(listfile),
            "-vf", f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
                   f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2,"
                   f"fps={TARGET_FPS},setpts=N/{TARGET_FPS}/TB",
            "-c:v", "libx264", "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-g", "25", "-keyint_min", "25",
            "-avoid_negative_ts", "make_zero",
            "-an",
            "-movflags", "frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset",
            "-frag_duration", "500000",
            "-flush_packets", "1",
            "-f", "mp4",
            "pipe:1",
        ]
        logger.info(
            "session=%s starting ffmpeg with %d clips, list=%s",
            self.session_id, len(self.clips), listfile,
        )
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def iter_chunks(self) -> AsyncIterator[bytes]:
        if self._proc is None:
            raise RuntimeError("call start() first")
        assert self._proc.stdout is not None
        try:
            while True:
                chunk = await self._proc.stdout.read(CHUNK_BYTES)
                if not chunk:
                    break
                yield chunk
        finally:
            await self._drain_stderr()

    async def _drain_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        try:
            err = await self._proc.stderr.read()
            if err:
                tail = err.decode("utf-8", errors="replace")[-2000:]
                logger.info("session=%s ffmpeg stderr tail:\n%s", self.session_id, tail)
        except Exception:
            pass

    async def close(self) -> None:
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    logger.warning("session=%s ffmpeg did not exit after kill", self.session_id)
            except ProcessLookupError:
                pass
        if self._tmpdir and self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]
