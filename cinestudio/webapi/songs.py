"""Songs: the music library and new songs."""
from __future__ import annotations

import sys

from aiohttp import web

from .. import cli, song, spawn, workqueue
from .common import _bg, _cleanup_canceled, _int_or_none, _ok, file_url, routes


# ------------------------------------------------------------------ music (ACE-Step songs)


def _song_view(s: dict) -> dict:
    d = song.songs_dir() / s["name"]
    audio = next((d / f for f in ("song.flac", "song.wav") if (d / f).is_file()), None) if s.get("status") == "done" else None
    b = s.get("beats") or {}
    return {k: s.get(k) for k in ("name", "status", "tags", "lyrics", "bpm", "seconds", "key", "seed", "language", "added",
                                  "started", "ended", "seconds_used", "error", "progress")} | {
        "audio_url": file_url(audio) if audio else None,
        "beats": {"bpm": b.get("bpm"), "count": len(b.get("beats") or []),
                  "first_downbeat": (b.get("downbeats") or [None])[0], "seconds": b.get("seconds")} if b else None}


@routes.get("/api/songs")
async def songs(request: web.Request):
    await _bg(_cleanup_canceled)
    return _ok({"songs": [_song_view(s) for s in song.library()], "keys": list(song.KEYS)})


@routes.post("/api/songs/new")
async def song_new(request: web.Request):
    f = await request.json()
    return _ok(await _bg(_song_new, f))


def _song_new(f: dict, wait: float = 20) -> dict:
    """Start one song as an independent background worker (`song make`), so an app restart does not stop it."""
    import time
    from ..locks import pid_alive
    name = (f.get("name") or "").strip()
    d = song.song_dir(name)                                # checks the name
    if (d / "song.json").is_file():
        old = song.load(name)
        if old.get("status") in song.BUSY and song._worker_alive(old):
            raise song.SongError(f"song {name!r} is already being made")
        if old.get("status") in ("done", "unresolved"):
            raise song.SongError(f"a song named {name!r} exists ({old['status']}): choose another name or delete it first")
    tags = (f.get("tags") or "").strip()
    if not tags:
        raise ValueError("describe the style (tags)")
    key = f.get("key") or "E minor"
    if key not in song.KEYS:
        raise ValueError(f"key must be one of {list(song.KEYS)}")
    bpm, seconds = int(f.get("bpm") or 120), float(f.get("seconds") or 30)
    if not 30 <= bpm <= 300 or not song.MIN_SECONDS <= seconds <= 600:
        raise ValueError("BPM must be 30..300 and seconds 4..600")
    seed = _int_or_none(f.get("seed"))
    # free text as --opt=value: a text that starts with "-" must not be read as an option
    cmd = [sys.executable, "-m", "cinestudio", "song", "make", name, f"--tags={tags}",
           f"--lyrics={(f.get('lyrics') or '').strip() or '[instrumental]'}", "--bpm", str(bpm), "--seconds", str(seconds),
           "--key", key, "--seed", str(31 if seed is None else seed), f"--language={(f.get('language') or 'en').strip()}"]
    log = song.songs_dir() / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    workqueue.add_new("song", name, f"{name} · song")     # only this request owns its reservation
    try:
        pid, independent = spawn.detached(cmd, log, cli.ROOT)
    except BaseException:
        workqueue.done("song", name)
        raise
    deadline = time.time() + wait
    while time.time() < deadline:                          # the worker writes its record ("queued") at once
        try:
            rec = song.load(name)
            if (rec.get("added") or 0) >= started - 1 and rec.get("pid") == pid:
                return {"name": name, "pid": pid, "status": rec.get("status"),
                        **({} if independent else {"warning": "the worker may stop when this app stops"})}
        except song.SongError:
            pass
        if not pid_alive(pid):
            tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-2:] if log.is_file() else []
            raise song.SongError("the song worker ended at once: " + " | ".join(tail))
        time.sleep(0.25)
    return {"name": name, "pid": pid, "status": "starting"}


@routes.post("/api/songs/{name}/delete")
async def song_delete(request: web.Request):
    return _ok(await _bg(song.delete, request.match_info["name"]))
