"""A film from several scenes with ONE continuous score (2026-10-24, user: "is there a way to create a bgm that is
coherent across multiple blender renders?").

    python -m cinestudio film assemble NAME --scene RUN_ID --scene RUN_ID ... [--score SONG] [--music-db -8]

Two sound layers, made apart and joined only here:
- the SCENE sound: each scene's own video sound (LTX makes the ambience and the effects with the picture; the scene's
  sound text should say "no music, no one speaks"); the scenes are joined with hard cuts, each with its own sound;
- the SCORE: one music file for the whole film (a library song, e.g. ACE-Step), laid under the whole timeline from 0,
  so it never restarts at a cut.
The music is mixed below the scene sound (music_db), ducked a little when a strong effect plays (sidechain
compression), faded out at the end, and the whole film is loudness-normalized once (EBU R128, ffmpeg loudnorm).
The film goes to <home>/films/NAME/film.mp4 with film.json: the scenes, their start times (the cue sheet for fitting
the music to the cuts) and the settings.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import video


class FilmError(ValueError):
    pass


NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def films_dir() -> Path:
    from .runner import home
    d = home() / "films"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _scene_file(scene: str) -> tuple[Path, str | None]:
    """A scene = a run ID (its final.mp4) or a video file. Returns (file, run id or None)."""
    p = Path(scene)
    if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm") and p.is_file():
        return p, None
    from . import runner
    try:
        d = runner.run_dir(scene)
    except Exception as exc:  # noqa: BLE001 - an unknown run id: say what is missing
        raise FilmError(f"scene {scene!r}: not a video file and not a run ({exc})") from exc
    f = d / "final.mp4"
    if not f.is_file():
        raise FilmError(f"scene {scene!r}: the run has no final.mp4 yet")
    return f, scene


def _score_file(score: str | None, score_file: str | None) -> tuple[Path | None, str | None]:
    if score and score_file:
        raise FilmError("give a library song (score) or a file (score_file), not both")
    if score_file:
        p = Path(score_file)
        if not p.is_file():
            raise FilmError(f"score file {score_file} does not exist")
        return p, None
    if score:
        from . import song
        s = song.load(score)
        if s.get("status") not in (None, "done") or not s.get("file") or not Path(s["file"]).is_file():
            raise FilmError(f"song {score!r} is not finished")
        return Path(s["file"]), score
    return None, None


def assemble(name: str, scenes: list[str], score: str | None = None, score_file: str | None = None,
             score_start: float = 0.0, music_db: float = -8.0, scene_db: float = 0.0, duck: bool = True,
             fade_out: float = 2.0, loudness: float = -16.0, replace: bool = False) -> dict:
    """Join `scenes` (run IDs or video files, in order) with hard cuts and lay one score under all of them. The size and
    frame rate of the first scene are the film's; other sizes are scaled and padded (letterbox). An existing film of
    that name is replaced only with `replace`; one film name is made by one process at a time (a lock)."""
    if not NAME_RE.match(name or ""):
        raise FilmError("film name: lowercase letters, digits, - and _ (at most 64)")
    from .locks import FileLock, LockBusy
    lock = FileLock(films_dir() / f".{name}.lock")
    try:
        lock.acquire()
    except LockBusy as exc:
        raise FilmError(f"film {name!r} is being made now by another process") from exc
    try:
        return _assemble(name, scenes, score, score_file, score_start, music_db, scene_db, duck, fade_out, loudness,
                         replace)
    finally:
        lock.release()


def _assemble(name, scenes, score, score_file, score_start, music_db, scene_db, duck, fade_out, loudness,
              replace) -> dict:
    if not NAME_RE.match(name or ""):
        raise FilmError("film name: lowercase letters, digits, - and _ (at most 64)")
    if not scenes:
        raise FilmError("a film needs at least one scene")
    if not -40.0 <= float(music_db) <= 6.0 or not -40.0 <= float(scene_db) <= 6.0:
        raise FilmError("music_db and scene_db must be between -40 and 6")
    if not -30.0 <= float(loudness) <= -8.0:
        raise FilmError("loudness must be between -30 and -8 LUFS")
    files = [_scene_file(s) for s in scenes]
    probes = [video.probe(f) for f, _ in files]
    fps = probes[0].fps
    for (f, _), p in zip(files, probes):
        if abs(p.fps - fps) > 0.01:
            raise FilmError(f"{f}: {p.fps} fps; the film is {fps} fps (render every scene at the same rate)")
        if p.frames < 1:
            raise FilmError(f"{f}: no frames")
    w, h = probes[0].width, probes[0].height
    durs = [p.frames / fps for p in probes]
    total = sum(durs)
    sfile, sname = _score_file(score, score_file)

    out_dir = films_dir() / name
    dst = out_dir / "film.mp4"
    if dst.exists() and not replace:
        raise FilmError(f"film {name!r} exists already (replace=True / --replace to make it again)")
    out_dir.mkdir(parents=True, exist_ok=True)
    import secrets
    token = secrets.token_hex(4)
    tmp = out_dir / f".film_{token}.final.mp4"
    inputs, parts = [], []
    for i, ((f, _), p) in enumerate(zip(files, probes)):
        inputs += ["-i", str(f)]
        parts.append(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
                     f"setsar=1,format=yuv420p,trim=end_frame={p.frames},setpts=PTS-STARTPTS[v{i}]")   # (no fps filter: every
        #                                                       scene has the film's rate; fps= dropped the last frame)
        if video.has_audio(f):
            parts.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                         f"asetpts=PTS-STARTPTS,apad,atrim=end={durs[i]:.6f}[s{i}]")
        else:
            parts.append(f"anullsrc=r=48000:cl=stereo,aformat=sample_fmts=fltp,atrim=end={durs[i]:.6f}[s{i}]")
    n = len(files)
    parts.append("".join(f"[v{i}][s{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vc][sc]")
    if sfile is not None:
        k = n
        inputs += ["-i", str(sfile)]
        fo = max(0.0, min(float(fade_out), total / 2))
        parts.append(f"[{k}:a]atrim=start={max(0.0, float(score_start)):.6f},asetpts=PTS-STARTPTS,aresample=48000,"
                     f"aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=end={total:.6f},"
                     f"afade=t=out:st={max(0.0, total - fo):.6f}:d={fo:.6f},volume={float(music_db):.2f}dB[mus]")
        parts.append(f"[sc]volume={float(scene_db):.2f}dB,asplit=2[sx][side]")
        if duck:      # the music gives way a little while a strong effect plays (the scene sound is the sidechain)
            parts.append("[mus][side]sidechaincompress=threshold=0.05:ratio=3:attack=20:release=400[musd]")
        else:
            parts.append("[side]anullsink;[mus]anull[musd]")
        parts.append("[sx][musd]amix=inputs=2:normalize=0:duration=first[mix]")
        mix = "mix"
    else:
        parts.append(f"[sc]volume={float(scene_db):.2f}dB[mix]")
        mix = "mix"
    # pass 1: the joined picture and the mixed sound (not yet normalized) as two files
    tmp_v, tmp_a = out_dir / f".film_{token}.mp4", out_dir / f".film_{token}.wav"
    parts.append(f"[{mix}]aresample=48000[ac]")
    video._run([*inputs, "-filter_complex", ";".join(parts),
                "-map", "[vc]", "-frames:v", str(sum(q.frames for q in probes)), "-c:v", "libx264", "-crf", "14",
                "-preset", "slow", "-pix_fmt", "yuv420p", "-r", str(fps), "-an", str(tmp_v),
                "-map", "[ac]", "-t", f"{total:.6f}", "-c:a", "pcm_s16le", str(tmp_a)])
    # pass 2: measure, then normalize with the measured values (two-pass loudnorm); a silent mix stays silent
    # (Codex review: a noise floor through one-pass loudnorm turned a silent film into noise at -10.7 LUFS)
    m = _measure(tmp_a, loudness)
    af = (f"loudnorm=I={float(loudness):.1f}:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true,"
          f"aresample=48000" if m else "anull")
    video._run(["-i", str(tmp_v), "-i", str(tmp_a), "-map", "0:v", "-map", "1:a", "-af", af, "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.6f}", str(tmp)])
    tmp_v.unlink(missing_ok=True)
    tmp_a.unlink(missing_ok=True)
    t, cues = 0.0, []
    for (f, rid), p, d in zip(files, probes, durs):
        cues.append({"scene": rid or str(f), "file": str(f), "start": round(t, 4), "seconds": round(d, 4),
                     "frames": p.frames, "size": [p.width, p.height]})
        t += d
    rec = {"name": name, "file": str(dst), "seconds": round(total, 4), "fps": fps, "size": [w, h], "scenes": cues,
           "score": {"song": sname, "file": str(sfile) if sfile else None, "start": float(score_start),
                     "music_db": float(music_db)} if sfile else None,
           "settings": {"scene_db": float(scene_db), "duck": bool(duck), "fade_out": float(fade_out),
                        "loudness": float(loudness)},
           "made": time.time()}
    jtmp = out_dir / f".film_{token}.json"
    jtmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    tmp.replace(dst)                                  # the picture and its record are published together
    jtmp.replace(out_dir / "film.json")
    return rec


def _measure(wav: Path, target: float) -> dict | None:
    """ffmpeg loudnorm's first pass on the mix: its JSON measurements, or None when the mix is (near) silent (no
    finite loudness above -70 LUFS: nothing to normalize)."""
    import math
    import subprocess
    r = subprocess.run([video.ffmpeg(), "-nostdin", "-hide_banner", "-i", str(wav), "-af",
                        f"loudnorm=I={float(target):.1f}:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=video.FFMPEG_TIMEOUT)
    txt = r.stderr
    try:
        m = json.loads(txt[txt.rindex("{"):txt.rindex("}") + 1])
        i = float(m["input_i"])
    except (ValueError, KeyError):
        return None
    if not math.isfinite(i) or i < -70.0:
        return None
    return m


def list_films() -> list[dict]:
    out = []
    for p in sorted(films_dir().glob("*/film.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({k: r.get(k) for k in ("name", "file", "seconds", "scenes", "score")})
    return out
