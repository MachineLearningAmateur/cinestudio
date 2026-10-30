"""Original songs for a shot (ACE-Step 1.5, MIT: the output may be used commercially), their beats, and the song
in a Blender shot.

    make(name, tags, lyrics, bpm, seconds, ...)  -> <home>/songs/<name>/song.flac + song.wav + song.json
    beats(path, bpm)                              -> beat and downbeat times (numpy + ffmpeg; no extra package)
    place(blend, name, start, out)                -> a copy of the shot with the song as a Video Sequencer strip

A song job follows the same rules as every GPU job: a ticket at the end of the shared queue, the GPU lock from the
model-family check until the job ends, the prompt id recorded as ours, a job-scoped cancel only.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import server, workqueue
from .locks import FileLock, LockBusy

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
KEYS = tuple(f"{n} {m}" for m in ("major", "minor")
             for n in ("C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"))


class SongError(RuntimeError):
    pass


class SongUnresolved(SongError):
    """A submitted song job could not be confirmed as ended (ComfyUI did not answer): it may still run."""


def life_lock(name: str) -> FileLock:
    """The per-name lifecycle lock: a song worker holds it for its whole job; deletes and new requests check it.
    Non-blocking: a second worker for the same name gets LockBusy at once. The name is checked first (song_dir)."""
    song_dir(name)
    return FileLock(songs_dir() / ".locks" / f"{name}.lock", timeout=0, note={"song": name})


def songs_dir() -> Path:
    from .runner import home
    return home() / "songs"


def song_dir(name: str) -> Path:
    if not NAME_RE.match(name or ""):
        raise SongError(f"song name {name!r}: use letters, digits, '-' and '_' (max 64)")
    return songs_dir() / name


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def load(name: str) -> dict:
    p = song_dir(name) / "song.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SongError(f"no song {name!r}") from exc


def list_songs() -> list[dict]:
    out = []
    for p in sorted(songs_dir().glob("*/song.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({k: s.get(k) for k in ("name", "status", "bpm", "seconds", "tags", "file", "beats")})
    return out


def request_cancel(name: str) -> dict:
    """Explicit cancel of one song (queued or running): the worker sees the flag and cancels its own job only."""
    d = song_dir(name)
    if not (d / "song.json").is_file():
        raise SongError(f"no song {name!r}")
    (d / "cancel").write_text(str(time.time()), encoding="utf-8")
    return {"song": name, "cancel_requested": True}


# ---------------------------------------------------------------- the GPU job
MIN_SECONDS = 4.0
DRAIN_SECONDS = 3600.0          # how long a worker waits for an unconfirmed job before it calls it "unresolved"
CANCEL_POLL_SECONDS = 2.0
REPLACEABLE = ("failed", "canceled")


def make(name: str, tags: str, lyrics: str = "[instrumental]", bpm: int = 120, seconds: float = 30.0,
         key: str = "E minor", seed: int = 31, language: str = "en", on_event: Callable[[dict], None] | None = None,
         backend_factory: Callable[[], Any] | None = None, *, replace: bool = False) -> dict:
    """One original song. Waits for its turn in the shared queue, then runs ACE-Step in ComfyUI.

    The worker holds the song's lifecycle lock for the whole job (a second worker for the same name stops at once
    and touches nothing). A done (or unresolved) song is never overwritten unless `replace=True`; a failed or
    canceled one, or one whose worker died, may be made again."""
    d = song_dir(name)
    if key not in KEYS:
        raise SongError(f"key must be one of {list(KEYS)}")
    if not 30 <= int(bpm) <= 300 or not MIN_SECONDS <= float(seconds) <= 600:
        raise SongError(f"bpm must be 30..300 and seconds {MIN_SECONDS:g}..600")
    life = life_lock(name)
    try:
        life.acquire()
    except LockBusy as exc:
        raise SongError(f"song {name!r} is already being made by another worker") from exc
    try:
        return _make_locked(d, name, tags, lyrics, bpm, seconds, key, seed, language, on_event, backend_factory,
                            replace)
    finally:
        life.release()


def _make_locked(d: Path, name: str, tags: str, lyrics: str, bpm: int, seconds: float, key: str, seed: int,
                 language: str, on_event: Callable[[dict], None] | None, backend_factory: Callable[[], Any] | None,
                 replace: bool) -> dict:
    from .backend import SdkBackend, UnresolvedSubmission
    from .workflow import ProfileWorkflow
    label = f"{name} · song"
    if (d / "song.json").is_file():
        old = load(name)
        st = old.get("status")
        if st in BUSY and _worker_alive(old) and int(old.get("pid") or 0) != os.getpid():
            raise SongError(f"song {name!r} is already being made")
        if not replace and st not in REPLACEABLE + BUSY:          # done, unresolved, anything unknown: keep it
            workqueue.done("song", name)                         # our own reservation (the web app's), if any
            raise SongError(f"a song named {name!r} exists ({st}): choose another name or delete it first")
    emit = on_event or (lambda e: None)
    params = {"tags": tags, "lyrics": lyrics, "bpm": int(bpm), "seconds": float(seconds), "key": key, "seed": int(seed),
              "language": language}
    from . import models as _models
    from . import recipe
    tier = _models.job_tier()
    rec = {"name": name, "status": "queued", "added": time.time(), "pid": os.getpid(), **params}
    (d / "cancel").unlink(missing_ok=True)
    _write(d / "song.json", rec)
    canceled = lambda: (d / "cancel").is_file()       # noqa: E731
    try:
        bundle = _models.bundle(tier)
        wf = ProfileWorkflow("acestep", recipe.ACE_VARIANT, models=bundle)
        if "cfg" in wf.bind["inputs"]:
            params = dict(params, cfg=recipe.ace_cfg(bundle))
        token = secrets.token_hex(3)
        prefix, submit_key = f"cinestudio/song_{name}_{token}", f"song-{name}-{token}"
        graph = wf.build(dict(params, output_prefix=prefix))
        from .workflow import graph_cache, save_graph      # the exact graph sent (the app's Workflows view)
        save_graph(graph_cache(), f"song_{name}", graph, keep=200,
                   meta={"tag": f"song_{name}", "profile": "acestep", "variant": recipe.ACE_VARIANT, "prefix": prefix})
        workqueue.claim("song", name, label)
        workqueue.wait_turn("song", name, label, should_stop=canceled,
                            on_wait=lambda n: emit({"event": "queued", "ahead": n}))
        from . import comfy
        if backend_factory is None and not server.alive():
            emit({"event": "starting_comfyui"})
            comfy.ensure_running()
        lock = server.gpu_lock(note={"song": name})
        try:
            lock.acquire()
        except LockBusy:
            emit({"event": "waiting_for_gpu", "holder": lock.holder()})
            lock.timeout = 6 * 3600
            lock.acquire()
        try:
            if canceled():
                raise workqueue.QueueCanceled(f"song {name} was canceled")
            rec.update(status="running", started=time.time())
            _write(d / "song.json", rec)
            switch = server.ensure_family(wf.family_key())
            if switch.get("restarted"):
                emit({"event": "comfyui_restarted", **switch})
            backend = (backend_factory or SdkBackend)()
            t0 = time.time()
            if canceled():                                         # the last check before the job exists
                raise workqueue.QueueCanceled(f"song {name} was canceled")
            rec.update(prefix=prefix, submit_key=submit_key)      # written BEFORE the submit: a lost answer can be found
            _write(d / "song.json", rec)
            jid = None
            try:
                jid = backend.submit(graph, {}, key=submit_key, prefix=prefix)
                server.note_owned(jid)
                rec["job"] = jid
                _write(d / "song.json", rec)
                emit({"event": "comfy_job", "job": jid})
                flac = _watch(backend, jid, wf, d, rec, canceled, emit)
            except BaseException as exc:
                # the job may still run on the GPU: keep the GPU lock and the queue ticket until ComfyUI confirms
                # that it ended (or that it never existed); otherwise the next job could start next to it
                pending = jid or (exc.key if isinstance(exc, UnresolvedSubmission) else None)
                if _drain(prefix, jid, backend) != "ended":
                    workqueue.hold("song", name, prefix=prefix, job=jid)
                    raise SongUnresolved(f"song {name}: the ComfyUI job ({pending or prefix}) could not be confirmed "
                                         f"as ended after {DRAIN_SECONDS:.0f} s ({exc}); check ComfyUI's queue before "
                                         "starting more GPU work") from exc
                raise
            seconds_used = round(time.time() - t0, 1)
        finally:
            lock.release()
        _ffmpeg(["-i", str(flac), "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(d / "song.wav")])
        found = beats(d / "song.wav", int(bpm))
        rec.update(status="done", ended=time.time(), seconds_used=seconds_used, file=str(d / "song.wav"),
                   flac=str(flac), beats=found)
        rec.pop("progress", None)
        _write(d / "song.json", rec)
        return rec
    except workqueue.QueueCanceled:
        rec.update(status="canceled", ended=time.time())
        _write(d / "song.json", rec)
        raise
    except SongUnresolved as exc:
        rec.update(status="unresolved", ended=time.time(), error=str(exc))
        _write(d / "song.json", rec)
        raise
    except Exception as exc:
        rec.update(status="failed", ended=time.time(), error=str(exc))
        _write(d / "song.json", rec)
        raise
    finally:
        workqueue.done("song", name)


def _watch(backend: Any, jid: str, wf: Any, d: Path, rec: dict, canceled: Callable[[], bool],
           emit: Callable[[dict], None]) -> Path:
    """Watch the job to its end with a cancel watcher next to it (a cancel does not wait for a progress event)."""
    stop = threading.Event()

    def watcher() -> None:
        while not stop.wait(CANCEL_POLL_SECONDS):
            if canceled():
                try:
                    backend.cancel(jid)                        # job-scoped, once
                except Exception:  # noqa: BLE001 - the watch below sees the outcome either way
                    continue                                  # transient failure: retry until the watch ends
                return
    t = threading.Thread(target=watcher, name=f"song-cancel-{jid}", daemon=True)
    t.start()

    def progress(kind: str, data: dict) -> None:
        if kind == "progress" and data.get("steps"):
            rec["progress"] = {"value": data.get("step"), "max": data.get("steps")}
            _write(d / "song.json", rec)
            emit({"event": "progress", "value": data.get("step"), "max": data.get("steps")})
    try:
        st = backend.watch(jid, progress)
    finally:
        stop.set()
        t.join(timeout=5)
    if canceled():
        raise workqueue.QueueCanceled(f"song {rec['name']} was canceled")
    if not st.ok:
        raise SongError(f"ComfyUI job {jid} {st.status}: {st.error}")
    return server.download_output(server.comfyui_url(), jid, wf.bind["outputs"]["audio"], d / "song.flac")


def _drain(prefix: str, jid: str | None, backend: Any = None, limit: float | None = None, poll: float = 5.0,
           sleep: Callable[[float], None] = time.sleep) -> str:
    """Wait until a submitted song job has surely ended: "ended" when ComfyUI (answering) shows it finished or
    knows nothing of it in its queue, its history AND the proxy three checks in a row; "unknown" after `limit`
    seconds without such evidence (ComfyUI or the proxy did not answer)."""
    url = server.comfyui_url()
    end = time.time() + (DRAIN_SECONDS if limit is None else limit)
    absent = 0
    while True:
        try:
            if server.alive(url):
                pid = jid or server.find_prompt(prefix, url)
                proxy_state = None
                if backend is not None:
                    try:
                        found = pid or backend.find(prefix)
                        proxy_state = backend.state(found) if found else None
                        pid = pid or found
                    except Exception:  # noqa: BLE001 - the proxy did not answer: no evidence from it
                        proxy_state = "no answer"
                if pid:
                    st = server.prompt_status(url, pid)
                    if st in ("done_ok", "done_error"):
                        return "ended"
                    if st == "unknown" and proxy_state not in (None, "no answer") and proxy_state.done:
                        return "ended"
                    absent = absent + 1 if st == "unknown" and proxy_state is None else 0
                else:
                    absent = absent + 1 if proxy_state is None else 0
                if absent >= 3:
                    return "ended"
        except Exception:  # noqa: BLE001 - ComfyUI went away during the check: no evidence
            absent = 0
        if time.time() >= end:
            return "unknown"
        sleep(poll)


def _ffmpeg(args: list[str]) -> None:
    from .video import ffmpeg as ffmpeg_exe
    subprocess.run([ffmpeg_exe(), "-nostdin", "-y", "-v", "error", *args], check=True, stdin=subprocess.DEVNULL,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


# ---------------------------------------------------------------- beats
SR, HOP, NFFT = 22050, 256, 1024


def _mono(path: Path) -> np.ndarray:
    from .video import ffmpeg as ffmpeg_exe
    raw = subprocess.run([ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         check=True, capture_output=True,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _onsets(y: np.ndarray, max_hz: float | None = None) -> np.ndarray:
    """Spectral-flux onset strength, one value per HOP samples (the standard log-magnitude flux)."""
    n = 1 + max(0, len(y) - NFFT) // HOP
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(y[idx] * np.hanning(NFFT), axis=1))
    if max_hz:
        spec = spec[:, : int(max_hz * NFFT / SR) + 1]
    flux = np.maximum(0.0, np.diff(np.log1p(100 * spec), axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    k = int(SR / HOP)                                   # remove the slow loudness trend (1 s window)
    if len(flux) > 2 * k:                               # a very short sound has no trend to remove
        flux = flux - np.convolve(flux, np.ones(k) / k, mode="same")
    flux = np.maximum(flux, 0.0)
    return flux / (flux.std() + 1e-9)


def _tempo(env: np.ndarray, bpm: float) -> float:
    """The tempo near `bpm` (+-8 %) with the strongest autocorrelation of the onset strength."""
    fr = SR / HOP
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    best, score = bpm, -1.0
    for cand in np.arange(bpm * 0.92, bpm * 1.08, 0.1):
        lag = 60.0 * fr / cand
        s = sum(np.interp(lag * m, np.arange(len(ac)), ac) for m in (1, 2, 4))
        if s > score:
            best, score = float(cand), s
    return best


def _track(env: np.ndarray, period: float, tightness: float = 100.0) -> np.ndarray:
    """Dynamic-programming beat tracker (Ellis 2007, the method behind librosa.beat): beats on strong onsets, spaced
    close to `period` frames."""
    n = len(env)
    score, back = env.astype(float).copy(), np.full(n, -1)
    lo, hi = max(1, int(round(period / 2))), max(1, int(round(period * 2)))
    for t in range(lo, n):
        prev = np.arange(max(0, t - hi), t - lo + 1)
        if len(prev) == 0:
            continue
        cost = -tightness * np.log((t - prev) / period) ** 2
        cand = score[prev] + cost
        i = int(np.argmax(cand))
        if cand[i] > 0:
            score[t] += cand[i]
            back[t] = prev[i]
    w = max(1, min(n, int(period * 2)))
    t = int(np.argmax(score[-w:])) + n - w
    out = [t]
    while back[out[-1]] >= 0:
        out.append(int(back[out[-1]]))
    return np.array(out[::-1])


def beats(path: str | Path, bpm: float = 120.0) -> dict:
    """Beat times (seconds), the measured tempo and the downbeats (the bar phase with the most low-end onsets: the
    kick usually falls on count 1)."""
    y = _mono(Path(path))
    seconds = round(len(y) / SR, 2)
    if len(y) / SR < 4 * 60.0 / float(bpm) or len(y) < NFFT * 4:     # under 4 beats: no grid can be found
        return {"bpm": None, "beats": [], "downbeats": [], "seconds": seconds, "beat_strength": None}
    env = _onsets(y)
    tempo = _tempo(env, float(bpm))
    period = 60.0 * SR / HOP / tempo
    b = _track(env, period)
    b = b[b >= 0]
    t = (b * HOP + NFFT / 2) / SR                    # the time of a frame is its window's centre
    low = _onsets(y, max_hz=150.0)
    phase = max(range(4), key=lambda p: float(low[b[p::4]].sum()))
    on_beat = float(env[b].mean() / (env.mean() + 1e-9))
    return {"bpm": round(tempo, 2), "beats": [round(float(x), 3) for x in t],
            "downbeats": [round(float(x), 3) for x in t[phase::4]], "seconds": round(len(y) / SR, 2),
            "beat_strength": round(on_beat, 2)}


def frames(beat_times: list[float], start: float, fps: float, first_frame: int = 1, last_frame: int | None = None) -> list[int]:
    """Beat times (seconds in the song) as shot frames, when the song's second `start` plays on `first_frame`."""
    out = [first_frame + int(round((bt - start) * fps)) for bt in beat_times if bt >= start - 1e-6]
    return [f for f in out if last_frame is None or f <= last_frame]


# ---------------------------------------------------------------- a loop
def _spec(y: np.ndarray, n: int = 2048, h: int = 512) -> np.ndarray:
    idx = np.arange(n)[None, :] + h * np.arange(max(0, len(y) - n) // h)[:, None]
    return np.log1p(np.abs(np.fft.rfft(y[idx] * np.hanning(n), axis=1))[:, :400])


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def loop_sections(path: str | Path, beat_times: list[float], downbeats: list[float], n_beats: int = 16) -> list[dict]:
    """Every section of n_beats that starts on a downbeat, scored for a seamless loop: `repeat` = how much the next
    n_beats sound the same (a section that repeats loops well), `seam` = the sound just before its end against the
    sound just before its start (what the loop plays across the cut). Best first."""
    y = _mono(Path(path))
    S, h = _spec(y), 512
    fr = lambda t: int(t * SR / h)                  # noqa: E731
    out = []
    for d in downbeats:
        if d not in beat_times:
            continue
        i = beat_times.index(d)
        if i + 2 * n_beats >= len(beat_times):
            break
        a0, a1, a2 = beat_times[i], beat_times[i + n_beats], beat_times[i + 2 * n_beats]
        A, B = S[fr(a0):fr(a1)], S[fr(a1):fr(a2)]
        m = min(len(A), len(B))
        k = 8
        repeat, seam = _cos(A[:m], B[:m]), _cos(S[fr(a1) - k:fr(a1)], S[fr(a0) - k:fr(a0)]) if fr(a0) >= k else 0.0
        out.append({"start": a0, "end": a1, "repeat": round(repeat, 4), "seam": round(seam, 4),
                    "score": round(repeat + seam, 4)})
    return sorted(out, key=lambda r: -r["score"])


def loop(name: str, n_beats: int = 16, frames: int = 240, fps: float = 24.0, start: float | None = None,
         tail_frames: int = 1) -> dict:
    """A seamless loop of n_beats from the song, stretched (pitch kept) to exactly `frames` frames, so the beats fall on
    a fixed frame grid: beat k on frame 1 + k * frames / n_beats. The file has one frame more at the end (the loop's
    first frame again) for a shot whose last frame repeats its first. `start` picks the section; else the best one."""
    s = load(name)
    if s.get("status") != "done":
        raise SongError(f"song {name!r} is not ready ({s.get('status')})")
    b = s["beats"]
    if start is None:
        ranked = loop_sections(s["file"], b["beats"], b["downbeats"], n_beats)
        if not ranked:
            raise SongError(f"song {name!r} is too short for a loop of {n_beats} beats")
        best = ranked[0]
    else:
        i = min(range(len(b["beats"])), key=lambda j: abs(b["beats"][j] - start))
        if i + n_beats >= len(b["beats"]):
            raise SongError("the loop does not fit in the song after that start")
        best = {"start": b["beats"][i], "end": b["beats"][i + n_beats]}
    length = frames / fps
    factor = (best["end"] - best["start"]) / length          # > 1: the music plays a little faster
    if not 0.8 <= factor <= 1.25:
        raise SongError(f"the loop would change the tempo by {abs(factor - 1) * 100:.0f} %: make the song at "
                        f"{60 * n_beats / length:.0f} bpm")
    d = song_dir(name)
    out = d / (f"loop_{best['start']:.3f}_{n_beats}b_{frames}f_{fps:g}fps"
               + (f"_tail{tail_frames}" if tail_frames != 1 else "") + ".wav")
    tag = f"{os.getpid()}.{secrets.token_hex(3)}"
    once, part = d / f"loop.{tag}.once.tmp.wav", d / f"loop.{tag}.part.tmp.wav"
    _ffmpeg(["-ss", f"{best['start']:.6f}", "-t", f"{best['end'] - best['start']:.6f}", "-i", s["file"],
             "-af", f"atempo={factor:.6f},apad,atrim=end={length:.6f},afade=t=in:d=0.004,"
                    f"afade=t=out:st={length - 0.004:.6f}:d=0.004",
             "-t", f"{length:.6f}", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(once)])
    _ffmpeg(["-i", str(once), "-i", str(once), "-filter_complex",
             f"[1:a]atrim=end={tail_frames / fps:.6f}[w];[0:a][w]concat=n=2:v=0:a=1[a]", "-map", "[a]", "-c:a", "pcm_s16le",
             str(part)])
    once.unlink(missing_ok=True)
    os.replace(part, out)
    return {**best, "file": str(out), "frames": frames, "fps": fps, "tempo_change": round(factor, 4),
            "bpm": round(60 * n_beats / length, 3), "beat_frames": [1 + round(k * frames / n_beats)
                                                                    for k in range(n_beats + 1)]}


# ---------------------------------------------------------------- the song in a shot
def place(blend: str | Path, name: str, start: float, out: str | Path, log: str | Path | None = None,
          sound_file: str | Path | None = None) -> dict:
    """A copy of the shot with the song from second `start` as a Video Sequencer sound strip on the first frame
    (the run's 'sound: auto' then puts this song on the video). The original .blend does not change."""
    from . import blender
    s = load(name)
    if s.get("status") != "done":
        raise SongError(f"song {name!r} is not ready ({s.get('status')})")
    d = song_dir(name)
    cut = Path(sound_file) if sound_file else d / f"from_{start:.3f}.wav"
    if not cut.is_file():
        _ffmpeg(["-ss", f"{start:.6f}", "-i", s["file"], "-c:a", "pcm_s16le", str(cut)])
    blender.run_job("place_sound.py", blend, ["--sound", cut, "--out", out], log or d / "place.log")
    return {"blend": str(out), "sound": str(cut), "start": start}


# ---------------------------------------------------------------- the app's song library
BUSY = ("queued", "running")


def _worker_alive(rec: dict) -> bool:
    from .locks import pid_alive
    pid = rec.get("pid")
    return bool(pid) and pid_alive(int(pid))


def library() -> list[dict]:
    """Every song with its full record, newest first. A queued or running song whose worker process is gone is shown
    as "interrupted" (the record itself does not change)."""
    out = []
    for p in songs_dir().glob("*/song.json"):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        s.setdefault("name", p.parent.name)
        if s.get("status") in BUSY and not _worker_alive(s):
            s["status"] = "interrupted"
        if s.get("status") == "canceled":              # not shown; cleanup_canceled deletes it
            continue
        out.append(s)
    out.sort(key=lambda s: s.get("added") or 0, reverse=True)
    return out


def cleanup_canceled() -> list[str]:
    """Delete every canceled song whose worker has stopped (user, 2026-10-21: canceled tasks are not shown and are
    cleaned up). Returns the deleted names."""
    gone = []
    for p in songs_dir().glob("*/song.json"):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if s.get("status") != "canceled":
            continue
        try:
            delete(p.parent.name)
            gone.append(p.parent.name)
        except (SongError, OSError):
            continue
    return gone


def delete(name: str) -> dict:
    """Remove a song's folder (its audio, record and cuts) and its worker log. Refused while it is queued or being made."""
    import shutil
    d = song_dir(name)
    life = life_lock(name)
    try:
        life.acquire()
    except LockBusy as exc:
        raise SongError(f"song {name!r} is being made: stop it first (queue) or wait until it ends") from exc
    try:
        rec = load(name)
        if rec.get("status") in BUSY and _worker_alive(rec):
            raise SongError(f"song {name!r} is {rec['status']}: stop it first (queue) or wait until it ends")
        shutil.rmtree(d)
        (songs_dir() / f"{name}.log").unlink(missing_ok=True)
    finally:
        life.release()
    return {"deleted": name}
