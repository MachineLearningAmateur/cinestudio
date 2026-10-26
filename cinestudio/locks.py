"""Exclusive file locks held by the operating system.

The lock belongs to the open file handle: it ends when the holder closes it OR when the holder's
process dies (crash, kill, power off), so a dead worker never leaves a stale lock behind.
Windows: msvcrt.locking on byte 0; POSIX: fcntl.flock.

    with FileLock(path, timeout=0):      # timeout 0 = fail at once if another process holds it
        ...
"""

from __future__ import annotations

import errno
import json
import os
import sys
import time
from pathlib import Path


class LockBusy(RuntimeError):
    """Another process holds the lock."""

    def __init__(self, path: Path, holder: dict | None) -> None:
        who = f" (held by pid {holder.get('pid')} since {holder.get('since')})" if holder else ""
        super().__init__(f"{path.name} is held by another process{who}")
        self.path, self.holder = path, holder


class FileLock:
    def __init__(self, path: str | Path, timeout: float = 0, poll: float = 0.25, note: dict | None = None) -> None:
        self.path = Path(path)
        self.timeout, self.poll, self.note = timeout, poll, note or {}
        self._fh = None

    def _try(self) -> bool:
        fh = open(self.path, "a+b")
        try:
            if sys.platform.startswith("win"):
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            if sys.platform.startswith("win") or exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                return False
            # a filesystem without flock (some network volumes: ENOLCK, EOPNOTSUPP) must not read as "busy": every
            # queue, job and engine lock would then wait for a holder that does not exist (RunPod check, 2026-10-26)
            raise OSError(exc.errno, f"file locks do not work in {self.path.parent} ({exc.strerror}); put "
                                     "CINESTUDIO_HOME on a disk that supports flock") from exc
        self._fh = fh
        return True

    def acquire(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while not self._try():
            if time.monotonic() >= deadline:
                raise LockBusy(self.path, self.holder())
            time.sleep(self.poll)
        # who holds it, for error messages only (the lock itself is the OS lock, not this text)
        info = {"pid": os.getpid(), "since": time.strftime("%Y-%m-%dT%H:%M:%S"), **self.note}
        try:
            self.path.with_suffix(self.path.suffix + ".who").write_text(json.dumps(info), encoding="utf-8")
        except OSError:
            pass
        return self

    def holder(self) -> dict | None:
        try:
            return json.loads(self.path.with_suffix(self.path.suffix + ".who").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if sys.platform.startswith("win"):
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None

    @property
    def held(self) -> bool:
        return self._fh is not None

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


def is_locked(path: str | Path) -> bool:
    """True if another process holds the lock now (tries to take it and gives it back at once)."""
    lock = FileLock(path, timeout=0)
    try:
        lock.acquire()
    except LockBusy:
        return True
    lock.release()
    return False


def pid_alive(pid: int) -> bool:
    """True if a process with this id runs now (for records that name their owner process)."""
    if not pid or pid <= 0:
        return False
    if sys.platform.startswith("win"):
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259                                          # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_ctime(pid: int) -> float | None:
    """Creation time of a process (with the pid, the identity of that process: Windows reuses pids)."""
    try:
        import psutil
        return psutil.Process(int(pid)).create_time()
    except Exception:  # noqa: BLE001 - gone, no access, or no psutil
        return None
