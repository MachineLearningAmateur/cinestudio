"""film assemble: scenes joined with hard cuts, each with its own sound, and ONE score under the whole film that does
not restart or stop at a cut. No GPU."""
import json
import subprocess

import numpy as np
import pytest

from cinestudio import film, video


def _scene(path, colour, seconds, tone=None):
    ff = video.ffmpeg()
    args = [ff, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:s=64x32:r=24:d={seconds}"]
    if tone:
        args += ["-f", "lavfi", "-i", f"sine=f={tone}:r=48000:d={seconds}", "-c:a", "aac", "-shortest"]
    subprocess.run(args + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def _band(path, t0, t1, f0):
    """Energy of frequency f0 (+-20 Hz) in [t0, t1] s of the file's sound, relative to all energy there."""
    raw = subprocess.run([video.ffmpeg(), "-v", "error", "-i", str(path), "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}",
                          "-vn", "-f", "f32le", "-ac", "1", "-ar", "48000", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / 48000)
    return spec[(freqs > f0 - 20) & (freqs < f0 + 20)].sum() / (spec.sum() + 1e-12), np.sqrt(np.mean(x ** 2))


def test_scenes_are_joined_and_one_score_runs_under_all_of_them(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    a = _scene(tmp_path / "a.mp4", "red", 2, tone=1000)
    b = _scene(tmp_path / "b.mp4", "blue", 1.5)                       # a scene without sound
    c = _scene(tmp_path / "c.mp4", "lime", 2, tone=1500)
    score = tmp_path / "score.wav"
    subprocess.run([video.ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=220:r=48000:d=12", str(score)],
                   check=True)
    rec = film.assemble("test-film", [str(a), str(b), str(c)], score_file=str(score), fade_out=0.5)
    out = tmp_path / "home" / "films" / "test-film" / "film.mp4"
    assert rec["file"] == str(out) and out.is_file()
    assert video.probe(out).frames == 48 + 36 + 48
    assert [s["start"] for s in rec["scenes"]] == [0.0, 2.0, 3.5]                 # the cue sheet
    assert json.loads((out.parent / "film.json").read_text())["seconds"] == pytest.approx(5.5, abs=0.01)
    for cut in (2.0, 3.5):                                            # the music has no gap at the cuts
        share, rms = _band(out, cut - 0.15, cut + 0.15, 220)
        assert share > 0.05 and rms > 0.01                            # (ducked under scene a's own sound)
    assert _band(out, 0.5, 1.5, 1000)[0] > 0.2                        # scene a keeps its own sound
    assert _band(out, 2.5, 3.2, 220)[0] > 0.9                         # scene b: the music alone
    assert _band(out, 4.0, 4.8, 1500)[0] > 0.2                        # scene c keeps its own sound
    assert film.list_films()[0]["name"] == "test-film"


def test_a_film_refuses_bad_input(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    a = _scene(tmp_path / "a.mp4", "red", 1)
    with pytest.raises(film.FilmError, match="name"):
        film.assemble("Bad Name", [str(a)])
    with pytest.raises(film.FilmError, match="not a video file and not a run"):
        film.assemble("x", ["no-such-run"])
    with pytest.raises(film.FilmError, match="not both"):
        film.assemble("x", [str(a)], score="s", score_file=str(a))
    rec = film.assemble("no-score", [str(a)])                         # without a score: the scene sound alone
    assert rec["score"] is None and video.probe(rec["file"]).frames == 24


def test_a_silent_film_stays_silent_and_a_film_is_not_replaced_by_mistake(tmp_path, monkeypatch):
    """Codex review: a noise floor through one-pass loudnorm turned a silent film into noise (-10.7 LUFS). And an
    existing film was replaced without a word."""
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    a = _scene(tmp_path / "a.mp4", "red", 1)                          # no sound at all
    rec = film.assemble("quiet", [str(a)])
    assert _band(rec["file"], 0.1, 0.9, 220)[1] < 1e-4                # still silent (RMS)
    with pytest.raises(film.FilmError, match="exists already"):
        film.assemble("quiet", [str(a)])
    assert film.assemble("quiet", [str(a)], replace=True)["file"] == rec["file"]
    assert not list((tmp_path / "home" / "films" / "quiet").glob(".film_*"))   # no temporary files left
