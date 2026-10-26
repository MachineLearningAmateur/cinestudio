"""Start a background worker that does not belong to the process that started it.

A worker started directly by the app is a child of the app: stopping the app's process tree (a
terminal that closes, a tool that kills the tree, a restart) also stops the worker. So a short-lived
launcher starts the worker and exits at once: the worker has no living parent, and on Windows it also
leaves the parent's job object when Windows allows it (CREATE_BREAKAWAY_FROM_JOB).

    pid = spawn.detached([sys.executable, "-m", "cinestudio", "job", "work", job_id], log_path, cwd)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

WIN = sys.platform.startswith("win")
BREAKAWAY = 0x01000000                     # CREATE_BREAKAWAY_FROM_JOB
PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _flags(breakaway: bool) -> dict:
    if not WIN:
        return {"start_new_session": True}
    f = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP   # hidden console: no window pops up
    return {"creationflags": f | (BREAKAWAY if breakaway else 0)}


def _popen(cmd: list[str], **kw) -> tuple[subprocess.Popen, bool]:
    """(process, left the parent's job object). Without the breakaway, a job object that kills its
    processes when it closes can still stop the worker: the caller reports that as a warning."""
    try:
        return subprocess.Popen(cmd, **kw, **_flags(True)), WIN
    except OSError:                        # the job object does not allow a breakaway
        return subprocess.Popen(cmd, **kw, **_flags(False)), not WIN


def detached(cmd: list[str], log: str | Path, cwd: str | Path) -> tuple[int, bool]:
    """Start `cmd` as an independent background process (output to `log`). Returns (pid, independent):
    independent is False when Windows refused the breakaway from the parent's job object."""
    # the launcher runs in the cinestudio folder (so that it can import cinestudio); the worker runs in `cwd`
    launcher, outer = _popen([sys.executable, "-m", "cinestudio.spawn", str(log), str(cwd), *cmd],
                             cwd=str(PACKAGE_ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    out, _ = launcher.communicate(timeout=60)
    text = out.decode(errors="replace").strip()
    last = text.splitlines()[-1].split() if text.splitlines() else []
    if launcher.returncode != 0 or len(last) != 2 or not last[0].isdigit():
        raise OSError(f"the worker did not start: {text[-300:]}")
    return int(last[0]), outer and last[1] == "1"


def _main(argv: list[str]) -> int:
    log, cwd, cmd = argv[0], argv[1], argv[2:]
    with open(log, "a", encoding="utf-8") as fh:
        p, inner = _popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT)
    print(p.pid, 1 if inner else 0, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
