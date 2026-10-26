"""Find Blender and run cinestudio's Blender jobs in background mode (no window).

Blender is a setup requirement: the shot (camera, set, placeholder character and its animation)
lives in a saved .blend file, and bpy_jobs/render_passes.py renders the raw passes from it.
Blender does not need to be open; each job starts `blender -b` and ends when the job is done.

Search order for the Blender executable:
  1. CINESTUDIO_BLENDER (full path to blender / blender.exe)
  2. `blender` on PATH
  3. the default install folders (newest version first)
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BPY_JOBS = ROOT / "bpy_jobs"
MIN_VERSION = (4, 2)

_DEFAULT_GLOBS = {
    "win32": [r"C:\Program Files\Blender Foundation\Blender *\blender.exe"],
    "darwin": ["/Applications/Blender*.app/Contents/MacOS/Blender"],
    "linux": ["/usr/bin/blender", "/usr/local/bin/blender", "/opt/blender*/blender", "/snap/bin/blender"],
}


class BlenderError(RuntimeError):
    """Blender is missing, too old, or a Blender job failed."""


def _version_key(path: str) -> tuple:
    m = re.search(r"(\d+)\.(\d+)", path)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def find_blender() -> str:
    env = os.environ.get("CINESTUDIO_BLENDER")
    if env:
        if not Path(env).is_file():
            raise BlenderError(f"CINESTUDIO_BLENDER={env!r} is not a file")
        return env
    on_path = shutil.which("blender")
    if on_path:
        return on_path
    key = "win32" if sys.platform.startswith("win") else "darwin" if sys.platform == "darwin" else "linux"
    found = [p for pattern in _DEFAULT_GLOBS[key] for p in glob.glob(pattern)]
    if found:
        return sorted(found, key=_version_key)[-1]
    raise BlenderError("Blender not found: install Blender (https://www.blender.org/download/) "
                       "or set CINESTUDIO_BLENDER to the full path of the Blender executable")


def version(exe: str | None = None) -> tuple[int, int, int]:
    exe = exe or find_blender()
    out = subprocess.run([exe, "-b", "--factory-startup", "--version"], capture_output=True, text=True, timeout=120).stdout
    m = re.search(r"Blender (\d+)\.(\d+)(?:\.(\d+))?", out)
    if not m:
        raise BlenderError(f"cannot read the Blender version from {exe}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)


def check() -> dict:
    """For `cinestudio doctor`: path and version, or the error."""
    try:
        exe = find_blender()
        v = version(exe)
    except (BlenderError, OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    ok = v[:2] >= MIN_VERSION
    return {"ok": ok, "path": exe, "version": ".".join(map(str, v)),
            **({} if ok else {"error": f"Blender {'.'.join(map(str, MIN_VERSION))} or newer is needed"})}


def run_job(script: str, blend: str | Path | None, args: list[str], log: str | Path,
            timeout: float | None = None) -> None:
    """Run bpy_jobs/<script> in background Blender. Arguments go as a list (no shell). The whole
    Blender output goes to `log`; on failure the last lines are in the error."""
    exe = find_blender()
    cmd = [exe, "-b"]
    if blend is not None:
        cmd.append(str(blend))
    else:
        cmd.append("--factory-startup")
    cmd += ["--python-exit-code", "1", "--python", str(BPY_JOBS / script), "--", *map(str, args)]
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if proc.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        raise BlenderError(f"Blender job {script} failed (exit {proc.returncode}); log: {log}\n" + "\n".join(tail))
