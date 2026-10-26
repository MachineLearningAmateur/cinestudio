"""Templates: Blender shots made from a song and a dance (the app's Templates tab), ready for the Video tab.

    plan_dance(song_record, routines, loop, seconds, start) -> the frames, beats and tempo of a dance shot (no files)
    start_dance(name, song, routines, ...)                   -> writes the record and starts a detached worker
    make_dance(name)                                         -> the worker: shot.blend + preview.mp4 (+ poster.jpg)
    list_shots() / load(name) / delete(name)

A template lives in <home>/shots/<name>/ (shot.json, shot.blend, preview.mp4, poster.jpg). Hand-made shots in that
folder (a shot.blend without shot.json) are listed too, read-only. The dance is keyed by templates/funk_dance/build.py
on the song's beats; the song goes into the .blend as a sound strip, so a run uses it as "the template's sound".
Only Blender Workbench and ffmpeg run here (no GPU job), one build at a time (a file lock).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from . import blender, runner, song, video
from .locks import FileLock, LockBusy, pid_alive

FPS = 24
BUILD = blender.ROOT / "templates" / "funk_dance" / "build.py"
BUSY = ("queued", "running")
ROUTINES = {                       # name -> (counts, label); "X~" is routine X mirrored
    "S1": (8, "Shuffle + kicks"),
    "S2": (8, "Points + step-touch"),
    "H1": (8, "Gallop + lasso"),
    "H2": (8, "Side gallop + hey"),
    "A": (8, "Locking 8-count"),
    "B": (8, "Groove 8-count"),
    "F": (5, "Finale"),
}
LOOP_BEATS = (8, 16, 32, 48, 64)
LOOP_TAIL = 24                     # frames past a loop's end (the next cycle): the run cross-fades them into the start
COUNT_ONE_FRAME = 13               # a shot that is not a loop: frame 1 stands still, count 1 is on frame 13
START_GRACE = 90.0                 # a queued record without a worker pid is a fresh reservation for this long


class ShotError(RuntimeError):
    pass


def shots_dir() -> Path:
    return runner.home() / "shots"


BUILTIN = Path(__file__).resolve().parent.parent / "templates" / "library"


def builtin_dir() -> Path:
    """The templates that ship with the repository (read-only; setup/export_template.py adds one). Each folder has
    shot.json (file paths relative to the folder), shot.blend (sounds packed) and preview.mp4."""
    return BUILTIN


def _builtin_record(name: str) -> dict | None:
    d = builtin_dir() / name
    p = d / "shot.json"
    if not p.is_file():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    files = {k: (str(d / v) if v else None) for k, v in (rec.get("files") or {}).items()}
    return {**rec, "name": name, "builtin": True, "status": "done", "files": files}


def library_record(blend: str | Path) -> dict | None:
    """The record of a .blend that belongs to the template library (the user's or the built-in one), else None."""
    p = Path(blend).resolve()
    for i, root in enumerate((shots_dir(), builtin_dir())):
        try:
            if p.parent.parent == root.resolve() and song.NAME_RE.match(p.parent.name):
                if i == 1:                 # the built-in's own record, also when a user template
                    return _builtin_record(p.parent.name)  # of the same name hides it in the list
                return load(p.parent.name)
        except (ShotError, OSError):
            return None
    return None


def default_background(blend: str | Path | None) -> str | None:
    """The default background of a library template (shot.json "background", e.g. a built-in template's backplate
    drawn over its set), when that background is in the library; else None."""
    if not blend:
        return None
    try:
        rec = library_record(blend) or {}
    except Exception:  # noqa: BLE001 - not a template
        return None
    name = rec.get("background")
    if not isinstance(name, str) or not name:
        return None
    from . import background
    try:
        background.path(name)
    except background.BackgroundError:
        return None
    return name


NO_BACKGROUND = ("none", "own", "set")       # `run new --background none`: the template's own set (no default)


def run_background(blend: str | Path | None, background: str | None, background_text: str | None = None,
                   background_image: str | None = None, look_scene: str | None = None) -> tuple[str | None, str | None]:
    """The CLI's and MCP's `background` of a new run: (value for runner.new_run, the default background used or None).
    Only when NOTHING about the background was asked (no --background, --background-text, --background-image or
    --look-scene) does a template's default background (shot.json "background") come in as "saved:NAME"; "none" asks
    for the template's own set. A template without a default keeps the old behaviour (its own set). The app's form
    preselects the default itself and always sends its choice."""
    if isinstance(background, str) and background.strip().lower() in NO_BACKGROUND:
        return "", None
    if background or background_text or background_image or look_scene:
        return background, None
    name = default_background(blend)
    return (f"saved:{name}", name) if name else (background, None)


def shot_dir(name: str) -> Path:
    if not song.NAME_RE.match(name or ""):
        raise ShotError(f"template name {name!r}: use letters, digits, '-' and '_' (max 64)")
    return shots_dir() / name


def check_new_name(name: str) -> None:
    shot_dir(name)
    if name.lower().startswith("mask-"):         # the queue's figure-selection jobs use "mask-<template>" (workqueue)
        raise ShotError(f"{name!r}: names that start with 'mask-' are kept for the figure selection: choose another")
    if (builtin_dir() / name / "shot.json").is_file():
        raise ShotError(f"{name!r} is the name of a built-in template: choose another name")


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def load(name: str) -> dict:
    d = shot_dir(name)
    p = d / "shot.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ShotError(f"template {name!r}: unreadable record") from exc
    if (d / "shot.blend").is_file():
        return _handmade(d)
    rec = _builtin_record(name)
    if rec:
        return rec
    raise ShotError(f"no template {name!r}")


def _handmade(d: Path) -> dict:
    preview = next((d / f for f in ("preview.mp4", "dance_preview.mp4") if (d / f).is_file()), None)
    return {"name": d.name, "status": "done", "handmade": True, "kind": "blender",
            "added": (d / "shot.blend").stat().st_mtime,
            "files": {"blend": str(d / "shot.blend"), "preview": str(preview) if preview else None}}


def _worker_alive(rec: dict) -> bool:
    return bool(rec.get("pid")) and pid_alive(int(rec["pid"]))


def _busy(rec: dict) -> bool:
    """Queued or running with a live worker, or a fresh reservation whose worker has not started yet."""
    if rec.get("status") not in BUSY:
        return False
    if rec.get("pid"):
        return _worker_alive(rec)
    # "requeued": a template from a video queued again after the user's choice of people (video_template.pick)
    return time.time() - float(rec.get("requeued") or rec.get("added") or 0) < START_GRACE


def life_lock(name: str) -> FileLock:
    """The per-name lifecycle lock (non-blocking): held for check-and-reserve, for a whole build and for a delete.
    The name is checked FIRST (shot_dir): every caller takes this lock before it loads the record, so an unchecked
    name ("../x") would put the lock file outside the library (Codex review 2026-10-25)."""
    shot_dir(name)
    return FileLock(shots_dir() / ".locks" / f"{name}.lock", timeout=0, note={"shot": name})


class _Life:
    """`with _Life(name):` = the lifecycle lock, or ShotError at once when someone else holds it."""

    def __init__(self, name: str, busy: str) -> None:
        self.lock, self.busy = life_lock(name), busy

    def __enter__(self):
        try:
            self.lock.acquire()
        except LockBusy as exc:
            raise ShotError(self.busy) from exc
        return self

    def __exit__(self, *exc):
        self.lock.release()
        return False


def show(name: str) -> dict:
    """One template's record as `list_shots` shows it: a queued or running template whose worker is gone is
    "interrupted" (Codex review 2026-10-22: `shot show` kept showing a dead worker as running)."""
    rec = load(name)
    if rec.get("status") in BUSY and not _busy(rec):
        rec["status"] = "interrupted"
    return rec


def list_shots() -> list[dict]:
    """Every template and hand-made shot (a folder with shot.json or shot.blend), newest first. A queued or running
    template whose worker is gone shows as "interrupted" (the record itself does not change). The pass caches of runs
    (folders without either file) are not shots."""
    out = []
    root = shots_dir()
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if not d.is_dir() or not song.NAME_RE.match(d.name):
            continue
        try:
            rec = load(d.name)
        except ShotError:
            continue
        if rec.get("status") in BUSY and not _busy(rec):
            rec["status"] = "interrupted"
        if rec.get("status") == "canceled":            # not shown; cleanup_canceled deletes it
            continue
        rec.setdefault("name", d.name)
        out.append(rec)
    out.sort(key=lambda r: r.get("added") or 0, reverse=True)
    mine = {r["name"] for r in out}
    built = [r for r in (_builtin_record(d.name) for d in (sorted(builtin_dir().iterdir()) if builtin_dir().is_dir() else [])
                         if d.is_dir() and song.NAME_RE.match(d.name) and d.name not in mine) if r]
    built.sort(key=lambda r: r.get("order", 99))
    return out + built                                  # the user's templates first, then the built-in ones


def cleanup_canceled() -> list[str]:
    """Delete every canceled template whose worker has stopped (user, 2026-10-21). Returns the deleted names."""
    gone = []
    root = shots_dir()
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if not d.is_dir() or not song.NAME_RE.match(d.name):
            continue
        try:
            if load(d.name).get("status") != "canceled":
                continue
            delete(d.name)
            gone.append(d.name)
        except (ShotError, OSError):
            continue
    return gone


def delete(name: str) -> dict:
    """Remove a template's folder (its .blend, preview and record). Refused while it is being built. Runs keep their
    own frozen copy of the shot, so old runs still work."""
    d = shot_dir(name)
    if not d.exists() and _builtin_record(name):
        raise ShotError(f"{name!r} is a built-in template (it ships with cinestudio) and cannot be deleted")
    with _Life(name, f"template {name!r} is being built: wait until it ends"):
        rec = load(name)
        if _busy(rec):
            raise ShotError(f"template {name!r} is being built: wait until it ends")
        shutil.rmtree(d)
    return {"deleted": name}


# ---------------------------------------------------------------- the plan (pure: testable without Blender)
def parse_routines(text: str | list[str]) -> list[str]:
    items = text if isinstance(text, list) else [x.strip() for x in str(text or "").split(",")]
    items = [x for x in items if x]
    if not items:
        raise ShotError("choose at least one routine")
    for x in items:
        if x.rstrip("~") not in ROUTINES:
            raise ShotError(f"unknown routine {x!r}: use {', '.join(ROUTINES)} (add ~ for the mirrored routine)")
    return items


def plan_dance(rec: dict, routines: list[str], loop: bool = True, seconds: float = 10.0,
               start: float | None = None) -> dict:
    """Frames, beats and tempo of a dance shot on the song `rec` (a done song record with its beats).

    A loop has n beats (8, 16, 32 ... the one nearest to `seconds` at the song's tempo); the routines are repeated
    to fill it (S1,S2 = 16 beats). The song's best n-beat section is stretched (pitch kept, a few percent at most)
    to a whole number of frames, so beat k is on frame 1 + k * frames / n and the last frame repeats the first.
    Not a loop: the song plays from `start` (default: half a second before its first downbeat), count 1 is on
    frame 13 and the shot ends one beat after the last routine."""
    if rec.get("status") != "done":
        raise ShotError(f"song {rec.get('name')!r} is not ready ({rec.get('status')})")
    b = rec.get("beats") or {}
    bpm = float(b.get("bpm") or rec.get("bpm") or 120)
    counts = sum(ROUTINES[r.rstrip("~")][0] for r in routines)
    if loop:
        if any(r.rstrip("~") == "F" for r in routines):
            raise ShotError("the Finale ends the dance: it cannot be part of a loop")
        want = max(1.0, float(seconds)) * bpm / 60
        n = min(LOOP_BEATS, key=lambda k: abs(k - want))
        if n % counts:
            raise ShotError(f"a {seconds:g} s loop at {bpm:g} bpm has {n} beats ({n // 8} 8-counts): choose routines "
                            f"that fill {n} beats (now {counts})")
        full = routines * (n // counts)
        frames = n * max(1, round(60 / bpm * FPS))     # a whole number of frames per beat: every beat on a frame,
        #                                               and 16 beats at ~96 bpm = 240 frames, a 241-frame shot = 2 clips
        grid = FPS * 60 * n / frames
        return {"loop": True, "routines": full, "n_beats": n, "frames": frames + 1 + LOOP_TAIL, "loop_frames": frames,
                "loop_tail": LOOP_TAIL,
                "bpm": round(grid, 4), "song_bpm": bpm, "seconds": round(frames / FPS, 3),
                "beat_frames": [1 + round(k * frames / n) for k in range(n + 1)]}
    beats, downs = b.get("beats") or [], b.get("downbeats") or []
    if not beats:
        raise ShotError(f"song {rec.get('name')!r} has no beats")
    if start is None:
        first = next((d for d in downs if d - (COUNT_ONE_FRAME - 1) / FPS >= 0), None)
        if first is None:
            raise ShotError("the song has no downbeat after its first half second")
        start = first - (COUNT_ONE_FRAME - 1) / FPS
    frames_all = song.frames(beats, float(start), FPS)
    grid = [f for f in frames_all if f >= COUNT_ONE_FRAME - 2]
    if len(grid) < counts + 1:
        raise ShotError(f"the song is too short after {start:g} s for {counts} counts")
    step = grid[1] - grid[0]
    frames = grid[counts] + step
    if frames / FPS > float((b.get("seconds") or 1e9)) - float(start):
        raise ShotError("the song ends before the dance")
    return {"loop": False, "routines": routines, "n_beats": counts, "frames": frames, "start": round(float(start), 3),
            "bpm": bpm, "song_bpm": bpm, "seconds": round(frames / FPS, 3), "beat_frames": grid[:counts + 2]}


# ---------------------------------------------------------------- start (the app / CLI) and the worker
def prepare_dance(name: str, song_name: str, routines: str | list[str], loop: bool = True, seconds: float = 10.0,
                  start: float | None = None, night: bool = False) -> dict:
    """Check everything that can be checked at once and write the template's record ("queued"), all under the
    template's lifecycle lock (two requests for one name cannot both pass)."""
    with _Life(name, f"template {name!r} is already being built"):
        return _prepare_locked(name, song_name, routines, loop, seconds, start, night)


def _prepare_locked(name: str, song_name: str, routines: str | list[str], loop: bool, seconds: float,
                    start: float | None, night: bool) -> dict:
    d = shot_dir(name)
    if d.exists():
        try:
            old = load(name)
        except ShotError:
            old = {}
        if old.get("handmade") or (d / "shot.blend").is_file() and not (d / "shot.json").is_file():
            raise ShotError(f"{name!r} is a hand-made shot: choose another name")
        if _busy(old):
            raise ShotError(f"template {name!r} is already being built")
        if old.get("status") == "done":
            raise ShotError(f"a template named {name!r} exists: choose another name or delete it first")
        if not old and any(d.iterdir()):
            raise ShotError(f"the folder of {name!r} is used by something else: choose another name")
    rs = parse_routines(routines)
    rec_song = song.load(song_name)
    plan = plan_dance(rec_song, rs, loop, seconds, start)
    rec = {"name": name, "kind": "dance", "status": "queued", "added": time.time(), "pid": None,
           "params": {"song": song_name, "routines": rs, "loop": bool(loop), "seconds": float(seconds),
                      "start": start, "night": bool(night)}, "plan": plan}
    _write(d / "shot.json", rec)
    return rec


def start_dance(name: str, song_name: str, routines: str | list[str], loop: bool = True, seconds: float = 10.0,
                start: float | None = None, night: bool = False, spawner: Callable | None = None) -> dict:
    """prepare_dance, then the detached worker (`python -m cinestudio shot work NAME`): an app restart does not
    stop it."""
    from . import spawn
    rec = prepare_dance(name, song_name, routines, loop, seconds, start, night)
    plan, d = rec["plan"], shot_dir(name)
    log = d / "build.log"
    cmd = [sys.executable, "-m", "cinestudio", "shot", "work", name]
    pid, independent = (spawner or spawn.detached)(cmd, log, blender.ROOT)
    rec = load(name)                                     # the worker writes its own pid (never the parent: no race)
    return {"name": name, "pid": pid, "status": rec.get("status"), "plan": plan,
            **({} if independent else {"warning": "the worker may stop when this app stops"})}


def _run_blender(script: Path, blend: Path | None, args: list, log: Path) -> None:
    """Background Blender with any script (build.py is not in bpy_jobs). The whole output goes to `log`."""
    cmd = [blender.find_blender(), "-b"] + ([str(blend)] if blend else ["--factory-startup"])
    cmd += ["--python-exit-code", "1", "--python", str(script), "--", *map(str, args)]
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if proc.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        raise ShotError(f"Blender ({script.name}) failed (exit {proc.returncode}); log: {log}\n" + "\n".join(tail))


def _ffmpeg(args: list[str]) -> None:
    subprocess.run([video.ffmpeg(), "-nostdin", "-y", "-v", "error", *args], check=True, stdin=subprocess.DEVNULL,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def make_preview(d: Path, blend: Path, log: Path, loop: tuple[int, int] | None = None) -> dict:
    """A Workbench preview of the shot with its own sound: preview.mp4 and poster.jpg in `d` (frames removed after)."""
    frames_dir = d / "preview_frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    _run_blender(blender.BPY_JOBS / "preview_shot.py", blend, ["--out", frames_dir, "--percent", 50], log)
    sound_dir = d / "sound"
    shutil.rmtree(sound_dir, ignore_errors=True)
    _run_blender(blender.BPY_JOBS / "export_sound.py", blend, ["--out", sound_dir], log)
    info = json.loads((sound_dir / "sound.json").read_text(encoding="utf-8"))
    wav = video.mix_strips(info, sound_dir / "sound.wav") if info.get("sound") else None
    first = int(info.get("frame_start") or 1)
    pngs = sorted(frames_dir.glob("f_*.png"))
    if not pngs:
        raise ShotError("the preview has no frames")
    tmp = d / "preview.tmp.mp4"
    args = ["-framerate", str(FPS), "-start_number", str(first), "-i", str(frames_dir / "f_%04d.png")]
    args += (["-i", str(wav)] if wav else []) + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                                                  "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    args += (["-c:a", "aac", "-b:a", "192k", "-shortest"] if wav else []) + ["-movflags", "+faststart", str(tmp)]
    _ffmpeg(args)
    if loop:                                              # a loop shot: the preview loops seamlessly too
        closed = d / "preview.loop.tmp.mp4"
        video.loop_close(tmp, closed, loop[0], loop[1], FPS)
        os.replace(closed, tmp)
    os.replace(tmp, d / "preview.mp4")
    _ffmpeg(["-i", str(pngs[0]), "-q:v", "3", str(d / "poster.jpg")])
    shutil.rmtree(frames_dir, ignore_errors=True)
    return {"preview": str(d / "preview.mp4"), "poster": str(d / "poster.jpg"), "sound": str(wav) if wav else None,
            "frames": len(pngs)}


def make_dance(name: str, on_event: Callable[[dict], None] | None = None) -> dict:
    """The worker: build the dance shot of a queued template record. It holds the template's lifecycle lock for the
    whole build (a second worker for the same name stops at once and changes nothing); one build at a time overall
    (the build lock)."""
    with _Life(name, f"template {name!r} is already being built by another worker"):
        return _make_dance_locked(name, on_event)


def _make_dance_locked(name: str, on_event: Callable[[dict], None] | None) -> dict:
    emit = on_event or (lambda e: None)
    d = shot_dir(name)
    rec = load(name)
    if rec.get("handmade"):
        raise ShotError(f"{name!r} is a hand-made shot")
    rec.update(pid=os.getpid(), status="queued")
    _write(d / "shot.json", rec)
    p = rec["params"]
    log = d / "build.log"
    steps = ["song", "dance", "sound", "preview"]

    def step(i: int, text: str) -> None:
        rec["progress"] = {"step": steps[i], "i": i + 1, "n": len(steps), "text": text}
        _write(d / "shot.json", rec)
        emit({"event": "progress", "step": steps[i], "value": i, "max": len(steps), "text": text})

    lock = FileLock(shots_dir() / ".build.lock", timeout=6 * 3600, note={"shot": name})
    try:
        lock.acquire()
        try:
            rec.update(status="running", started=time.time())
            plan = plan_dance(song.load(p["song"]), p["routines"], p["loop"], p["seconds"], p["start"])
            rec["plan"] = plan
            step(0, "The song section")
            sound_file, start = None, float(plan.get("start") or 0.0)
            if plan["loop"]:
                lp = song.loop(p["song"], plan["n_beats"], plan["loop_frames"], FPS,
                               tail_frames=plan.get("loop_tail", 0) + 1)
                sound_file, start = lp["file"], float(lp["start"])
                rec["loop"] = {k: lp[k] for k in ("start", "end", "tempo_change", "file") if k in lp}
            step(1, "Blender keys the dance")
            args = ["--out", d / "dance.blend", "--fps", FPS, "--frames", plan["frames"],
                    "--routine", ",".join(plan["routines"])]
            if plan["loop"]:
                args += ["--bpm", repr(plan["bpm"]), "--first-beat", 1, "--loop", "--loop-frames", plan["loop_frames"]]
            else:
                args += ["--beats", ",".join(map(str, plan["beat_frames"]))]
            if p.get("night"):
                args += ["--night"]
            _run_blender(BUILD, None, args, log)
            step(2, "The song goes into the shot")
            song.place(d / "dance.blend", p["song"], start, d / "shot.blend", log=d / "place.log", sound_file=sound_file)
            (d / "dance.blend").unlink(missing_ok=True)
            step(3, "Blender makes a preview (Workbench)")
            files = make_preview(d, d / "shot.blend", log, loop=(plan["loop_frames"], plan.get("loop_tail", 0))
                                 if plan["loop"] and plan.get("loop_tail") else None)
            rec.pop("progress", None)
            rec.update(status="done", ended=time.time(), song_start=start,
                       files={"blend": str(d / "shot.blend"), **files},
                       frames=int(files.get("frames") or plan["frames"]))     # the render's length (shot_frames)
            _store_check(d / "shot.blend", rec, emit)
            _write(d / "shot.json", rec)
            emit({"event": "done", "name": name})
            return rec
        finally:
            lock.release()
    except Exception as exc:
        rec.update(status="failed", ended=time.time(), error=str(exc))
        rec.pop("progress", None)
        _write(d / "shot.json", rec)
        raise


# ---------------------------------------------------------------- any Blender shot as a template (agents, imports)
def add(name: str, blend: str | Path, description: str = "", move: bool = False, preview: bool = True,
        on_event: Callable[[dict], None] | None = None) -> dict:
    """Store a Blender shot as a template: the .blend is copied (or moved) to <home>/shots/<name>/shot.blend, the record
    says what the shot is (`description`), and a Workbench preview with the shot's own sound is made (preview.mp4).
    A shot that is already <home>/shots/<name>/shot.blend (an older hand-made shot) is adopted in place.
    Every Blender shot that an agent builds belongs here: the Video tab takes its shots from this library."""
    check_new_name(name)
    with _Life(name, f"template {name!r} is being built: choose another name or wait"):
        return _add_locked(name, blend, description, move, preview, on_event)


def _add_locked(name: str, blend: str | Path, description: str, move: bool, preview: bool,
                on_event: Callable[[dict], None] | None) -> dict:
    emit = on_event or (lambda e: None)
    src = Path(blend).resolve()
    if not src.is_file() or src.suffix.lower() != ".blend":
        raise ShotError(f"not a saved .blend file: {src}")
    d = shot_dir(name)
    in_place = src == (d / "shot.blend").resolve()
    if (d / "shot.json").is_file():
        raise ShotError(f"a template named {name!r} exists: choose another name or delete it first")
    if not in_place and d.exists() and any(d.iterdir()):
        raise ShotError(f"the folder of {name!r} is used by something else: choose another name")
    d.mkdir(parents=True, exist_ok=True)
    if not in_place:
        (shutil.move if move else shutil.copy2)(str(src), str(d / "shot.blend"))
    rec = {"name": name, "kind": "blender", "status": "running" if preview else "done", "added": time.time(),
           "pid": os.getpid(), "description": description.strip(), "source": str(src),
           "files": {"blend": str(d / "shot.blend")}}
    _write(d / "shot.json", rec)
    if not preview:
        return rec
    lock = FileLock(shots_dir() / ".build.lock", timeout=6 * 3600, note={"shot": name})
    try:
        lock.acquire()
        try:
            emit({"event": "progress", "step": "preview", "text": "Blender makes a preview (Workbench)"})
            files = make_preview(d, d / "shot.blend", d / "build.log")
        finally:
            lock.release()
    except Exception as exc:
        rec.update(status="failed", ended=time.time(), error=str(exc))
        _write(d / "shot.json", rec)
        raise
    rec.update(status="done", ended=time.time(), frames=files["frames"], seconds=round(files["frames"] / FPS, 3),
               files={"blend": str(d / "shot.blend"), **files})
    try:                                               # the layers (characters, props, set) and their warnings
        scan = scan_layers(d / "shot.blend")
        rec["layers"] = {"scan": scan, "descriptions": {"set": "", "props": {}}, "warnings": layer_warnings(scan)}
        for w in rec["layers"]["warnings"]:
            emit({"event": "layer_warning", "text": w})
    except Exception as exc:  # noqa: BLE001 - the template works without the scan; `shot layers NAME` again
        emit({"event": "layer_warning", "text": f"the layer scan failed: {exc}"})
    _store_check(d / "shot.blend", rec, emit)          # the template rules (shot check): the summary in shot.json
    _write(d / "shot.json", rec)
    return rec


def _store_check(blend: Path, rec: dict, emit: Callable[[dict], None] | None = None) -> None:
    """`shot check` at the end of a build (shot_check.after_build): rec["check"] = {errors, warnings, at, sha}; the
    template works without it when the check cannot run."""
    from . import shot_check
    summary = shot_check.after_build(blend, rec, emit)
    if summary:
        rec["check"] = summary


# ---------------------------------------------------------------- a video template (made from text, no .blend)
def add_video(name: str, video_file: str | Path, description: str = "", meta: dict | None = None) -> dict:
    """Store a template VIDEO (a stand-in figure in a scene, made from text with the open models) as a template of kind
    "video": <home>/shots/<name>/template.mp4 (also its preview) and poster.jpg. A run made from it reads the stand-in's
    pose and the set edges from the video (cinestudio/video_passes.py) and replaces the stand-in with the character."""
    check_new_name(name)
    with _Life(name, f"template {name!r} is being built: choose another name or wait"):
        src = Path(video_file).resolve()
        if not src.is_file() or src.suffix.lower() not in (".mp4", ".mov", ".mkv", ".webm"):
            raise ShotError(f"not a video file: {src}")
        d = shot_dir(name)
        if (d / "shot.json").is_file() or (d.exists() and any(d.iterdir())):
            raise ShotError(f"a template named {name!r} exists: choose another name or delete it first")
        d.mkdir(parents=True, exist_ok=True)
        files = store_video_files(d, src)
        rec = {"name": name, "kind": "video", "status": "done", "added": time.time(), "ended": time.time(),
               "description": description.strip(), "source": str(src), "frames": files.pop("frames"),
               "seconds": files.pop("seconds"), "meta": meta or {}, "files": files}
        _write(d / "shot.json", rec)
        return rec


def store_video_files(d: Path, src: str | Path) -> dict:
    """The files of a video template in its folder: template.mp4 (also its preview) and poster.jpg (the first frame).
    Returns the file paths plus "frames" and "seconds"."""
    tmp = d / "template.tmp.mp4"
    shutil.copy2(src, tmp)
    os.replace(tmp, d / "template.mp4")
    _ffmpeg(["-i", str(d / "template.mp4"), "-frames:v", "1", "-q:v", "3", str(d / "poster.jpg")])
    p = video.probe(d / "template.mp4")
    return {"video": str(d / "template.mp4"), "preview": str(d / "template.mp4"), "poster": str(d / "poster.jpg"),
            "frames": p.frames, "seconds": round(p.frames / p.fps, 3)}


def timeline(name: str) -> dict:
    """The template's motion events from its rig (cinestudio/timeline.py), stored in its shot.json ("timeline") with
    the per-frame data in timeline.json next to it. A template author can read and correct the events; runs with
    --clip-text auto write their clip texts from the rig."""
    from . import timeline as _tl
    rec = load(name)
    blend = Path((rec.get("files") or {}).get("blend") or "")
    if not blend.is_file():
        raise ShotError(f"template {name!r} has no .blend (a video template has no rig)")
    d = blend.parent
    raw = _tl.blend_timeline(blend, d / "timeline.json")
    ev = _tl.events(raw)
    p = d / "shot.json"
    data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    data["timeline"] = {"events": ev, "frames": [raw["frame_start"], raw["frame_end"]], "file": "timeline.json"}
    _write(p, data)
    return {"name": name, "timeline": data["timeline"]}


# ---------------------------------------------------------------- layers: characters, props, set (2026-10-24)
# User: "is it worth further breaking down our templates to give more control over background vs props vs characters?"
# Every template has three layers by Blender collection: Mannequin* (the characters' rigs), Props (objects that move or
# that the characters touch) and Set (the static place: the source of a backplate drawn over it, `background
# from-set`). shot.json "layers" = {"scan": what Blender found, "descriptions": {"set": TEXT, "props": {NAME or
# PATTERN: TEXT}}, "warnings": [...]}: the same words then go into the background, the video and the look texts.

def scan_layers(blend: str | Path) -> dict:
    """bpy_jobs/scan_layers.py on a .blend: {"characters", "props", "set", "loose", "camera_moves", "frames"}."""
    import tempfile

    from . import blender
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "layers.json"
        blender.run_job("scan_layers.py", blend, ["--out", str(out)], log=Path(td) / "scan_layers.log")
        return json.loads(out.read_text(encoding="utf-8"))


def layer_warnings(scan: dict) -> list[str]:
    w = []
    if not scan.get("characters"):
        w.append("no character layer: a template needs a Mannequin collection with its rig MannequinRig")
    for c in scan.get("characters") or []:
        if not c.get("rig"):
            w.append(f"{c['collection']} has no rig")
    if scan.get("loose"):
        names = ", ".join(scan["loose"][:8]) + (" ..." if len(scan["loose"]) > 8 else "")
        w.append(f"objects in no layer (put them in Set or Props): {names}")
    if not scan.get("set"):
        w.append("no Set collection: a backplate cannot be drawn over the set (use background plate from text)")
    if scan.get("camera_moves"):
        w.append("the camera moves: a backplate (from text or from the set) needs a still camera; use a 360 panorama")
    return w


def layers(name: str, set_description: str | None = None, props: dict | None = None, rescan: bool = False) -> dict:
    """Read, scan or describe a template's layers (stored in its shot.json). `props` maps a prop's object name or a
    pattern ("Ball_*") to its words ("plain red rubber dodgeballs"); a pattern that matches no prop is refused."""
    with _Life(name, f"template {name!r} is being built or changed now: try again when it ends"):
        return _layers_locked(name, set_description, props, rescan)


def _layers_locked(name: str, set_description: str | None, props: dict | None, rescan: bool) -> dict:
    import fnmatch
    rec = load(name)
    changes = set_description is not None or bool(props) or rescan
    if rec.get("builtin") and (changes or not (rec.get("layers") or {}).get("scan")):
        raise ShotError(f"{name!r} is a built-in template: its layers are set in the repo (templates/library/{name}/"
                        "shot.json, setup/export_template.py); copy it to your library to change them")
    blend = Path((rec.get("files") or {}).get("blend") or "")
    if not blend.is_file():
        raise ShotError(f"template {name!r} has no .blend (a video template has no layers)")
    p = blend.parent / "shot.json"
    data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else dict(rec)
    lay = dict(data.get("layers") or {})
    if rescan or not lay.get("scan"):
        lay["scan"] = scan_layers(blend)
    desc = lay.setdefault("descriptions", {})
    desc.setdefault("set", "")
    desc.setdefault("props", {})
    if set_description is not None:
        desc["set"] = set_description.strip()
    desc.pop("style", None)                             # a template has no look of its own (2026-10-25)
    names = [x["name"] for x in lay["scan"].get("props") or []]
    for key, text in (props or {}).items():
        if not (text and text.strip()):                 # removal: also a key that no longer matches (after a rescan)
            desc["props"].pop(key, None)
            continue
        if not any(fnmatch.fnmatchcase(n, key) for n in names):
            raise ShotError(f"no prop matches {key!r}; the props are: {', '.join(names) or 'none'}")
        desc["props"][key] = text.strip()
    lay["warnings"] = layer_warnings(lay["scan"]) + [
        f"the prop words {k!r} match no prop (rescan?): remove them with --prop '{k}='"
        for k in desc["props"] if not any(fnmatch.fnmatchcase(n, k) for n in names)]
    data["layers"] = lay
    _write(p, data)
    return {"name": name, "layers": lay}


def props_sentence(lay: dict | None) -> str:
    """The props' words for a prompt ("In the scene: plain red rubber dodgeballs."), each description once, in the
    order they were given; "" without described props."""
    seen = []
    for text in ((lay or {}).get("descriptions") or {}).get("props", {}).values():
        t = (text or "").strip().rstrip(".").replace(";", ",")
        if t and t not in seen:                         # "red balls" and "red balls." are one description
            seen.append(t)
    return f"In the scene: {'; '.join(seen)}." if seen else ""


# ---------------------------------------------------------------- slots: who goes where (2026-10-25)
# User: "each template should have an indicator of how many characters, and have it mapped so we can select our own
# character for each reference character in a template" and "if a video only has 1 character there should only be one
# character option, if it has three then three options". A slot = one Mannequin* layer of the template, numbered in
# the order of the collections (Mannequin = 1, Mannequin2 = 2, ...; a run maps slot 1 -> --character, 2 ->
# --character2). Read from the recorded layer scan only: the list never starts Blender.

def _coll_number(name: str) -> int:
    tail = name[len("Mannequin"):] if name.startswith("Mannequin") else name
    return int(tail) if tail.isdigit() else 1


def _sides(xs: list) -> list:
    """"left" / "centre" / "right" as seen in the picture, from each slot's x (None when any x is unknown)."""
    n = len(xs)
    if n == 1:
        return ["centre"]
    if any(x is None for x in xs):
        return [None] * n
    order = sorted(range(n), key=lambda i: xs[i])
    out: list = [None] * n
    for rank, i in enumerate(order):
        out[i] = "left" if rank == 0 else "right" if rank == n - 1 else "centre"
    return out


def mannequin_slots(rec: dict | None) -> dict:
    """The character slots of a template record: {"slots": [{"index", "collection", "rig", "label", "side",
    "place" [x, y] m or None, "screen" {"head": [u, v], "feet": [u, v]} (0..1 of the poster frame, v from the top)
    or None, "height_m"}], "scanned": bool, "markers": bool (every slot has a screen place)}. No scan (a hand-made
    shot, an old template): ONE slot "Mannequin", scanned False. A side ("left"/"right" as seen in the picture) only
    from a measured place: the scan's projected head, else a from-video person's floor place; else None."""
    rec = rec or {}
    if rec.get("kind") == "video":                     # a template video: one grey stand-in
        return {"slots": [{"index": 1, "collection": None, "rig": None, "label": "Character 1", "side": "centre",
                           "place": None, "screen": None, "height_m": None}], "scanned": True, "markers": False,
                "video": True}
    scan = ((rec.get("layers") or {}).get("scan")) or {}
    chars = sorted((c for c in scan.get("characters") or [] if c.get("collection")),
                   key=lambda c: _coll_number(c["collection"]))
    if not chars:
        return {"slots": [{"index": 1, "collection": "Mannequin", "rig": "MannequinRig", "label": "Character 1",
                           "side": "centre", "place": None, "screen": None, "height_m": None}],
                "scanned": bool(scan), "markers": False}
    people = {p.get("collection"): p for p in ((rec.get("people") or {}).get("chosen") or []) if p.get("collection")}
    slots = []
    for i, c in enumerate(chars, 1):
        screen = c.get("screen") if isinstance(c.get("screen"), dict) and c["screen"].get("head") else None
        place = c.get("place") or (people.get(c["collection"]) or {}).get("place")
        slots.append({"index": i, "collection": c["collection"], "rig": c.get("rig"), "label": f"Character {i}",
                      "place": [round(float(v), 3) for v in place[:2]] if place else None, "screen": screen,
                      "height_m": c.get("height_m")})
    xs = [s["screen"]["head"][0] if s["screen"] else None for s in slots]
    if any(x is None for x in xs):                     # no projected place: the floor places of a from-video template
        xs = [s["place"][0] if s["place"] else None for s in slots]   # (its camera looks along +Y: x = picture right)
    for s, side in zip(slots, _sides(xs)):
        s["side"] = side
    return {"slots": slots, "scanned": True, "markers": all(s["screen"] for s in slots)}


def slot_name(slot: dict) -> str:
    """"Character 2 (right, Mannequin2)" for messages."""
    side = slot.get("side") if slot.get("side") and slot.get("side") != "centre" else None
    bits = [b for b in (side, slot.get("collection")) if b]
    return f"{slot['label']} ({', '.join(bits)})" if bits else slot["label"]


# ---------------------------------------------------------------- prop groups: the objects a run can re-word
# User: "the generating video template should also expose the objects in the template. that way we can map our own
# props to the object (for example the ball)". A group = one pattern of the template's prop words (shot layers
# --prop "Ball_*=..."), or, for props without words, the objects that share a name prefix. A run's `props`
# {pattern: words} replaces the template's words of those groups in its clip prompts (runner.props_sentence).

PROP_WORDS_MAX = 300


def _prop_parts(props: list[dict]) -> set[str]:
    """Objects that are a part of another prop (a ball's seam): parented to a prop, or named "<prop>_<part>"."""
    names = {p["name"] for p in props}
    parts = set()
    for p in props:
        if p.get("parent") in names:
            parts.add(p["name"])
        elif "parent" not in p and any(p["name"].startswith(o + "_") for o in names if o != p["name"]):
            parts.add(p["name"])
    return parts


def _prefix_pattern(names: list[str]) -> tuple[str, str]:
    """(group key, pattern) of objects named alike: "Ball_Dodge1", "Ball_Kick2" -> "Ball_*"."""
    import re
    first = names[0]
    pre = re.split(r"[_.\d]", first, maxsplit=1)[0] or first
    if all(n == pre for n in names):
        return pre, pre
    if all(n.startswith(pre + "_") for n in names):
        return pre, pre + "_*"
    return pre, pre + "*"


def _prop_label(pattern: str, count: int) -> str:
    """A short name of a group for the form: "Ball_*" -> "Balls"; "Photo" -> "Photo"."""
    import re
    base = re.split(r"[*?\[]", pattern, maxsplit=1)[0].rstrip("_.- ")
    base = re.sub(r"\d+$", "", base).rstrip("_.- ") or pattern
    base = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", base).replace("_", " ").strip()
    if count > 1 and not base.lower().endswith("s"):
        if re.search(r"[^aeiou]y$", base):
            base = base[:-1] + "ies"
        else:
            base += "es" if re.search(r"(x|ch|sh)$", base) else "s"
    return base[:1].upper() + base[1:]


prop_label = _prop_label


def prop_groups(rec: dict | None) -> list[dict]:
    """The template's prop groups: [{"pattern", "label", "count" (whole props, parts left out), "words" (the
    template's; "" = none), "objects"}], the described patterns first (in their order), then the other props by
    name prefix. [] without a Props layer (or a scan)."""
    import fnmatch
    lay = (rec or {}).get("layers") or {}
    props = [p for p in ((lay.get("scan") or {}).get("props") or []) if p.get("name")]
    if not props:
        return []
    words = ((lay.get("descriptions") or {}).get("props") or {})
    parts = _prop_parts(props)
    left = [p["name"] for p in props]
    out = []
    for pat, text in words.items():
        objs = [n for n in left if fnmatch.fnmatchcase(n, pat)]
        if not objs:                                   # words of a prop that is gone (rescan): not a group
            continue
        left = [n for n in left if n not in objs]
        whole = [n for n in objs if n not in parts] or objs
        out.append({"pattern": pat, "label": _prop_label(pat, len(whole)), "count": len(whole),
                    "words": (text or "").strip(), "objects": objs})
    groups: dict[str, list[str]] = {}
    for n in left:                                     # props without words: by name prefix
        groups.setdefault(_prefix_pattern([n])[0], []).append(n)
    for objs in groups.values():
        pat = _prefix_pattern(objs)[1]
        whole = [n for n in objs if n not in parts] or objs
        out.append({"pattern": pat, "label": _prop_label(pat, len(whole)), "count": len(whole), "words": "",
                    "objects": objs})
    return out


def check_prop_pattern(rec: dict | None, pat: str) -> str:
    """A run's prop pattern (for words or a picture): one of the template's prop groups, or matching one of its
    props; else ShotError (the old check of check_prop_words, shared with the run's prop pictures)."""
    import fnmatch
    lay = (rec or {}).get("layers") or {}
    names = [p["name"] for p in ((lay.get("scan") or {}).get("props") or []) if p.get("name")]
    if not names:
        raise ShotError("this template has no props (no Props layer recorded: shot layers NAME --rescan)")
    pats = [g["pattern"] for g in prop_groups(rec)]
    if pat not in pats and not any(fnmatch.fnmatchcase(n, pat) for n in names):
        raise ShotError(f"no prop matches {pat!r}; the template's props: {', '.join(pats) or 'none'}")
    return pat


def check_prop_words(rec: dict | None, props: dict | None) -> dict:
    """A run's prop words {pattern: words}, cleaned: empty words are left out (= the template's); each pattern must
    be one of the template's prop groups or match one of its props; words at most PROP_WORDS_MAX characters."""
    if not props:
        return {}
    if not isinstance(props, dict):
        raise ShotError('props must map a prop pattern to its words, e.g. {"Ball_*": "red beach balls"}')
    out = {}
    for pat, text in props.items():
        pat, text = str(pat or "").strip(), " ".join(str(text or "").split())
        if not text:
            continue
        check_prop_pattern(rec, pat)
        if len(text) > PROP_WORDS_MAX:
            raise ShotError(f"the words for {pat!r} are longer than {PROP_WORDS_MAX} characters")
        out[pat] = text
    return out


def effective_layers(lay: dict | None, props: dict | None) -> dict | None:
    """The frozen layer words with a run's prop words over the template's (the same pattern replaced, a new one
    added after them)."""
    if not props:
        return lay
    desc = dict((lay or {}).get("descriptions") or {})
    desc["props"] = {**(desc.get("props") or {}), **props}
    return {**(lay or {}), "descriptions": desc}


# ---------------------------------------------------------------- clip texts (2026-10-25)
# User: "when we create new clips each clip should have its own set of prompts ... whatever happened in clip 1 should
# not be described in clip 3" and "context that is needed throughout each clip should still be present but actions
# specific to that scene should not be used in other clips". A template author writes, once, from the rig timeline:
# shot.json "clip_texts" {"1": ACTIONS, "2": ...} (only what that clip's frames show, for the default clip split) and
# "clip_context" (lasting states and orientation for EVERY clip: "{character} faces the camera the whole time").
# "{character}" is replaced by the run's words for its character (runner.fill_character). A run's clip k gets: the
# run's main text (shared context: place, light, look) + the template's context + clip k's actions.
CLIP_TEXT_MAX = 1500
CLIP_CONTEXT_MAX = 600


def shot_frames(rec: dict | None) -> int | None:
    """The number of frames of a template (its record: frames, the preview's frames, the dance plan's frames (a loop
    with its tail), the timeline's or the layer scan's frame range, else its seconds at 24 fps); None when the record
    does not say. The seconds come last: a dance loop's seconds leave out its loop tail (Codex review 2026-10-25:
    233 render frames read as 208, so clip 3 lost its text)."""
    rec = rec or {}
    for fr in (rec.get("frames"), (rec.get("files") or {}).get("frames"), (rec.get("plan") or {}).get("frames")):
        if isinstance(fr, int) and not isinstance(fr, bool) and fr > 0:
            return fr
    for fr in ((rec.get("timeline") or {}).get("frames"), ((rec.get("layers") or {}).get("scan") or {}).get("frames")):
        if isinstance(fr, list) and len(fr) == 2:
            return int(fr[1]) - int(fr[0]) + 1
    secs = rec.get("seconds") or (rec.get("plan") or {}).get("seconds")
    return int(round(float(secs) * FPS)) if secs else None


def clip_plan(frames: int | None, clip_frames: int | None = None, overlap: int | None = None) -> list[dict]:
    """The clips a run cuts a shot of `frames` frames into (runner.Run.clip_format): [{"clip", "first", "last" (shot
    frames, 1-based, the last one clamped to the shot), "start_s", "end_s"}]; [] when the length is unknown."""
    if not frames:
        return []
    most = int(clip_frames or runner.MAX_CLIP_FRAMES)
    ov = int(runner.OVERLAP if overlap is None else overlap)
    n_frames = (min(frames, most) - 1) // 8 * 8 + 1
    n = 1 if frames <= most else video.n_clips(frames, n_frames, ov)
    out = []
    for k in range(1, n + 1):
        a = video.clip_start(k, n_frames, ov) + 1
        b = min(a + n_frames - 1, frames)
        out.append({"clip": k, "first": a, "last": b, "start_s": round((a - 1) / FPS, 2), "end_s": round(b / FPS, 2)})
    return out


def clip_texts_view(rec: dict | None) -> dict:
    """A template's clip texts for the app, the MCP and the CLI: {"clip_texts" {k: text}, "clip_context", "clips"
    [the default split, each with its frame span and the template's text]}."""
    rec = rec or {}
    texts = {str(k): v for k, v in (rec.get("clip_texts") or {}).items() if isinstance(v, str) and v.strip()}
    plan = clip_plan(shot_frames(rec))
    for c in plan:
        c["text"] = texts.get(str(c["clip"]), "")
    return {"clip_texts": texts, "clip_context": (rec.get("clip_context") or "").strip(), "clips": plan}


def clip_texts(name: str, texts: dict | None = None, context: str | None = None) -> dict:
    """Read or set a template's clip texts. `texts` {clip number: ACTIONS} (an empty text removes that clip's text);
    `context` the words for every clip ("" removes them). Built-in templates are read-only (their texts are set in the
    repo: templates/library/NAME/shot.json, setup/export_template.py)."""
    with _Life(name, f"template {name!r} is being built or changed now: try again when it ends"):
        return _clip_texts_locked(name, texts or {}, context)


def _clip_texts_locked(name: str, texts: dict, context: str | None) -> dict:
    rec = load(name)
    changes = bool(texts) or context is not None
    if not changes:
        return {"name": name, **clip_texts_view(rec)}
    if rec.get("builtin"):
        raise ShotError(f"{name!r} is a built-in template: its clip texts are set in the repo (templates/library/{name}/"
                        "shot.json, setup/export_template.py); copy it to your library to change them")
    blend = Path((rec.get("files") or {}).get("blend") or "")
    if not blend.is_file():
        raise ShotError(f"template {name!r} has no .blend (clip texts are for Blender templates)")
    n = len(clip_plan(shot_frames(rec))) or None
    p = blend.parent / "shot.json"
    data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else dict(rec)
    cur = {str(k): v for k, v in (data.get("clip_texts") or {}).items()}
    for k, text in texts.items():
        if not str(k).strip().isdigit() or int(k) < 1:
            raise ShotError(f"clip text: {k!r} is not a clip number (1, 2, 3 ...)")
        k = str(int(k))
        if n and int(k) > n:
            raise ShotError(f"clip {k}: the template has {n} clip{'s' if n != 1 else ''}")
        text = " ".join(str(text or "").split())
        if len(text) > CLIP_TEXT_MAX:
            raise ShotError(f"the text of clip {k} is longer than {CLIP_TEXT_MAX} characters")
        if text:
            cur[k] = text
        else:
            cur.pop(k, None)
    data["clip_texts"] = dict(sorted(cur.items(), key=lambda kv: int(kv[0])))
    if context is not None:
        context = " ".join(context.split())
        if len(context) > CLIP_CONTEXT_MAX:
            raise ShotError(f"the clip context is longer than {CLIP_CONTEXT_MAX} characters")
        data["clip_context"] = context
    if not data["clip_texts"]:
        data.pop("clip_texts")
    if not data.get("clip_context"):
        data.pop("clip_context", None)
    _write(p, data)
    return {"name": name, **clip_texts_view(data)}
