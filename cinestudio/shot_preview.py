"""Browser-playable previews for the app.

- shot_preview(blend): a small Workbench render of a shot (Video form). Workbench is light enough to run next to a
  GPU job (AGENTS.md); EEVEE is not used here. Kept per shot version (path, size and modification time).
- run_media(run_id): web copies of what a run really used: the shot's Blender render (the pass frames) and the
  control video. The run's own files are lossless RGB H.264 (4:4:4), which browsers do not play.
All jobs go through ONE background worker with a first-in first-out queue (one Blender process at a time); a job is
known by its destination file, so two requests for the same file share one job. Callers poll until "ready".
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import uuid
from collections import deque
from pathlib import Path

from . import blender, passes, runner

PREVIEW_PERCENT = 33           # of the shot's render size: 1536x896 -> about 507x296
WEB_WIDTH = 768                # web copies of a run's render and control
_cv = threading.Condition()
_queue: deque[tuple[str, object, tuple]] = deque()     # (key, job, args), oldest first
_queued: set[str] = set()                              # keys waiting in the queue
_running: str | None = None                            # the key in work now
_errors: dict[str, str] = {}
_worker: threading.Thread | None = None


def previews_dir() -> Path:
    return runner.home() / "previews"


def _flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _encode(args: list[str], dst: Path) -> Path:
    """ffmpeg <args> -> dst as H.264 4:2:0 (plays in every browser), through a temporary file of its own."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}.tmp.mp4")
    cmd = [passes.ffmpeg(), "-loglevel", "error", "-y", *args, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
           "-preset", "veryfast", "-movflags", "+faststart", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, creationflags=_flags())
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[-300:]}")
    os.replace(tmp, dst)
    return dst


def _even(width: int | None) -> list[str]:
    return ["-vf", f"scale={width}:-2" if width else "scale=trunc(iw/2)*2:trunc(ih/2)*2"]


def encode_frames(pattern: Path, start: int, fps: float, dst: Path, width: int | None = None) -> Path:
    return _encode(["-framerate", str(fps), "-start_number", str(start), "-i", str(pattern), *_even(width)], dst)


def web_copy(src: Path, dst: Path, width: int | None = WEB_WIDTH) -> Path:
    if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    return _encode(["-i", str(src), *_even(width)], dst)


def _work() -> None:
    """The one worker: take the oldest job, run it without holding the lock, record an error, repeat."""
    global _running
    while True:
        with _cv:
            while not _queue:
                _cv.wait()
            key, job, args = _queue.popleft()
            _queued.discard(key)
            _running = key
        try:
            job(*args)
        except Exception as exc:  # noqa: BLE001 - reported to the app on the next poll
            with _cv:
                _errors[key] = str(exc)[-400:]
        finally:
            with _cv:
                _running = None


def _enqueue(key: str, job, *args) -> str:
    """Queue job(*args) once per key; its state now: "making" (in work), "queued" or "error" (reported once)."""
    global _worker
    with _cv:
        if key in _errors:
            return "error"
        if key == _running:
            return "making"
        if key not in _queued:
            _queue.append((key, job, args))
            _queued.add(key)
            if _worker is None or not _worker.is_alive():
                _worker = threading.Thread(target=_work, name="cinestudio-previews", daemon=True)
                _worker.start()
            _cv.notify()
        return "queued"


def _take_error(key: str) -> str | None:
    with _cv:
        return _errors.pop(key, None)


def _shot_key(blend: Path) -> str:
    st = blend.stat()
    return hashlib.sha256(f"{blend.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:16]


def shot_preview(blend: str | Path) -> dict:
    """{"status": "ready", "path"} or {"status": "making"|"queued"} or {"status": "error", "error"}."""
    blend = Path(blend)
    if not blend.is_file() or blend.suffix.lower() != ".blend":
        raise ValueError(f"not a saved .blend file: {blend}")
    dst = previews_dir() / f"shot_{_shot_key(blend)}.mp4"
    if dst.is_file():
        return {"status": "ready", "path": str(dst)}
    key = str(dst.resolve())
    state = _enqueue(key, _make_shot, blend, dst)
    return {"status": "error", "error": _take_error(key)} if state == "error" else {"status": state}


def _make_shot(blend: Path, dst: Path) -> None:
    tmp = previews_dir() / f".{dst.stem}.{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True)
    try:
        blender.run_job("preview_shot.py", blend, ["--out", tmp, "--percent", PREVIEW_PERCENT], tmp / "blender.log",
                        timeout=1800)
        info = json.loads((tmp / "preview.json").read_text())
        encode_frames(tmp / "f_%04d.png", info["frame_start"], info["fps"], dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _media(key_src: Path | None, dst: Path, missing: str, job, *args) -> dict:
    """The state of one web copy: ready (path), making / queued, unavailable (why), or error (once)."""
    if dst.is_file():
        return {"status": "ready", "path": str(dst)}
    if key_src is None:
        return {"status": "unavailable", "note": missing}
    key = str(dst.resolve())
    state = _enqueue(key, job, *args)
    return {"status": "error", "note": _take_error(key)} if state == "error" else {"status": state}


def _render_job(pdir: Path, dst: Path) -> None:
    info = json.loads((pdir / "passes.json").read_text())
    encode_frames(pdir / "rgb" / "rgb_%04d.png", info["frame_start"], info["fps"], dst, WEB_WIDTH)


def run_media(run_id: str) -> dict:
    """The run's Blender render and control as web copies, each with its own state:
    {"render": {status, path|note}, "control": {status, path|note} | None, "status": overall}.
    A layout run uses no control video (the model sees the render itself): its "control" is None."""
    d = runner.run_dir(run_id)
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    st = json.loads((d / "state.json").read_text(encoding="utf-8"))
    pdir = runner.home() / "shots" / m["pass_key"] / "passes"
    outs = st.get("steps", {}).get("controls", {}).get("outputs") or {}
    ctl = outs.get("control_full") or outs.get("control")          # older runs kept only the clip's slice
    control_src = Path(ctl) if ctl and Path(ctl).is_file() else None
    render = _media(pdir if (pdir / "passes.json").is_file() else None, pdir / "web" / "render.mp4",
                    "the Blender render is not made yet (the run's first step)", _render_job, pdir, pdir / "web" / "render.mp4")
    control = None if m.get("render_mode") == "layout" else \
        _media(control_src, d / "work" / "web_control.mp4", "the control is not made yet (the run's second step)",
               web_copy, control_src, d / "work" / "web_control.mp4")
    states = {render["status"]} | ({control["status"]} if control else set())
    overall = "making" if states & {"making", "queued"} else "ready" if "ready" in states else \
        "error" if "error" in states else "unavailable"
    return {"status": overall, "render": render, "control": control}
