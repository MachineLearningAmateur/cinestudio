"""Songs: the beat finder on a known click track, beat frames, and the queued GPU job (fake ComfyUI)."""
import wave
from pathlib import Path

import numpy as np
import pytest

from cinestudio import song, workqueue
from cinestudio.backend import JobState


def click_track(path, bpm=118.0, offset=0.37, seconds=16, sr=48000):
    y = np.random.default_rng(0).normal(0, 0.01, sr * seconds)
    t = np.arange(int(0.15 * sr)) / sr
    kick = np.sin(2 * np.pi * 55 * t) * np.exp(-t * 18)
    click = np.random.default_rng(1).normal(0, 1, len(t)) * np.exp(-t * 80) * 0.3
    k = 0
    while offset + k * 60 / bpm < seconds - 0.2:
        i = int((offset + k * 60 / bpm) * sr)
        y[i:i + len(t)] += kick if k % 4 == 0 else click
        k += 1
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32000).astype(np.int16).tobytes())
    return [offset + i * 60 / bpm for i in range(k)]


def test_beats_find_tempo_phase_and_downbeat(tmp_path):
    true = click_track(tmp_path / "c.wav")
    b = song.beats(tmp_path / "c.wav", 120)
    assert abs(b["bpm"] - 118.0) < 0.5
    errs = [min(abs(x - t) for t in true) for x in b["beats"]]
    assert max(errs) < 0.021                     # under half a frame at 24 fps
    assert abs(b["downbeats"][0] - true[0]) < 0.021 and abs(b["downbeats"][1] - true[4]) < 0.021


def test_frames():
    assert song.frames([0.5, 1.0, 1.5, 2.0], start=0.5, fps=24) == [1, 13, 25, 37]
    assert song.frames([0.5, 1.0, 1.5], start=0.75, fps=24, first_frame=1, last_frame=10) == [7]


class FakeBackend:
    def __init__(self):
        self.submits, self.canceled = [], []

    def submit(self, graph, files, key, prefix=None):
        self.submits.append(graph)
        return "p1"

    def watch(self, jid, on_event):
        on_event("progress", {"step": 8, "steps": 8})
        return JobState(jid, "succeeded")

    def cancel(self, jid):
        self.canceled.append(jid)


class FreeLock:
    def acquire(self):
        pass

    def release(self):
        pass


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    owned = []
    monkeypatch.setattr(song.server, "ensure_family", lambda fam, url=None: {"restarted": False, "fam": fam})
    monkeypatch.setattr(song.server, "note_owned", lambda pid, url=None: owned.append(pid))
    monkeypatch.setattr(song.server, "gpu_lock", lambda url=None, timeout=0, note=None: FreeLock())
    monkeypatch.setattr(song.server, "alive", lambda url=None: True)
    monkeypatch.setattr(song, "_drain", lambda prefix, jid, backend=None, **k: "ended")   # the job ended (no ComfyUI)

    def no_comfyui(*a, **k):
        raise AssertionError("a test must not ask the real ComfyUI")
    monkeypatch.setattr(song.server, "prompt_status", no_comfyui)
    monkeypatch.setattr(song.server, "find_prompt", no_comfyui)
    track = tmp_path / "src.wav"
    click_track(track)
    monkeypatch.setattr(song.server, "download_output",
                        lambda url, pid, node, dst: Path(dst).write_bytes(track.read_bytes()) and Path(dst))
    return {"owned": owned, "backend": FakeBackend()}


def test_make_runs_one_queued_job(fake):
    r = song.make("groove", "funk, slap bass", bpm=118, seconds=16, backend_factory=lambda: fake["backend"])
    g = fake["backend"].submits[0]
    assert g["enc"]["inputs"]["bpm"] == 118 and g["lat"]["inputs"]["seconds"] == 16.0
    assert g["enc"]["inputs"]["seed"] == g["ks"]["inputs"]["seed"] == 31
    assert g["save"]["inputs"]["filename_prefix"].startswith("cinestudio/song_groove_")
    assert fake["owned"] == ["p1"]
    assert r["status"] == "done" and Path(r["file"]).is_file() and abs(r["beats"]["bpm"] - 118) < 0.5
    assert song.load("groove")["status"] == "done"
    assert workqueue.find("song", "groove") is None              # the ticket is gone after the job


def test_make_refuses_bad_input(fake):
    with pytest.raises(song.SongError):
        song.make("bad name!", "funk")
    with pytest.raises(song.SongError):
        song.make("ok", "funk", key="H minor")


def test_cancel_while_queued(fake, monkeypatch):
    def wait(kind, ref, label="", on_wait=None, should_stop=None, poll=2.0):
        song.request_cancel(ref)
        if should_stop():
            raise workqueue.QueueCanceled("canceled")
    monkeypatch.setattr(song.workqueue, "wait_turn", wait)
    with pytest.raises(workqueue.QueueCanceled):
        song.make("late", "funk", backend_factory=lambda: fake["backend"])
    assert song.load("late")["status"] == "canceled" and not fake["backend"].submits


def test_mix_strips_places_trims_and_pads(tmp_path):
    from cinestudio import video
    sr = 48000
    tone = (np.sin(2 * np.pi * 440 * np.arange(sr * 4) / sr) * 16000).astype(np.int16)
    src = tmp_path / "tone.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(tone.tobytes())
    # the sound starts on frame -23 (1 s before the shot); its part from frame 25 (2 s into the shot) is cut away
    info = {"fps": 24, "frame_start": 1, "frame_end": 96, "strips": [
        {"file": str(src), "frame_start": -23, "frame_final_start": -23, "frame_final_end": 25, "volume": 0.5}]}
    out = video.mix_strips(info, tmp_path / "mix.wav")
    y = song._mono(out)
    assert abs(len(y) / song.SR - 4.0) < 0.01                    # 96 frames at 24 fps, padded with silence
    first, later = np.abs(y[: song.SR]).max(), np.abs(y[int(1.1 * song.SR):]).max()
    assert 0.2 < first < 0.3 and later < 1e-3                     # volume 0.5 on 1 s, then silence
    assert video.mix_strips({**info, "strips": []}, tmp_path / "none.wav") is None


def test_loop_stretches_to_the_frame_grid(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    d = song.song_dir("clicks")
    d.mkdir(parents=True)
    click_track(d / "song.wav", bpm=118.0, offset=0.37, seconds=24)
    b = song.beats(d / "song.wav", 120)
    song._write(d / "song.json", {"name": "clicks", "status": "done", "file": str(d / "song.wav"), "beats": b})
    r = song.loop("clicks", n_beats=16, frames=192, fps=24)
    assert r["bpm"] == 120.0 and abs(r["tempo_change"] - 1.0167) < 0.01
    y = song._mono(r["file"])
    assert abs(len(y) / song.SR - (192 + 1) / 24) < 0.005             # the loop + one wrap frame
    got = song.beats(r["file"], 120)
    assert abs(got["bpm"] - 120) < 0.5
    grid = [k * 0.5 for k in range(17)]
    inner = [t for t in got["beats"] if t > 0.1]                       # no onset can be found at the file's first frame
    assert len(inner) >= 15 and max(min(abs(t - g) for g in grid) for t in inner) < 0.025
    with pytest.raises(song.SongError, match="tempo"):
        song.loop("clicks", n_beats=16, frames=320, fps=24)


def test_cut_audio_starts_at_the_second(tmp_path):
    from cinestudio import video
    sr = 48000
    y = np.zeros(sr * 3, np.int16)
    y[sr * 2:] = 12000                                          # loud from second 2
    with wave.open(str(tmp_path / "s.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())
    video.cut_audio(tmp_path / "s.wav", 2.0, tmp_path / "c.wav")
    z = song._mono(tmp_path / "c.wav")
    assert abs(len(z) / song.SR - 1.0) < 0.01 and np.abs(z[: song.SR // 2]).min() > 0.2


# ---------------------------------------------------------------- the review fixes (2026-10-20)
def test_a_done_song_is_not_overwritten(fake):
    song.make("keep", "funk", seconds=16, backend_factory=lambda: fake["backend"])
    before = song.load("keep")
    with pytest.raises(song.SongError, match="exists"):
        song.make("keep", "other", seconds=16, backend_factory=lambda: fake["backend"])
    assert song.load("keep") == before and len(fake["backend"].submits) == 1
    r = song.make("keep", "other", seconds=16, backend_factory=lambda: fake["backend"], replace=True)
    assert r["tags"] == "other" and r["status"] == "done"


def test_a_second_worker_for_the_same_name_touches_nothing(fake):
    workqueue.add("song", "busy", "busy · song")                    # the owner's reservation
    rec = {"name": "busy", "status": "queued", "pid": None, "added": 1.0}
    song._write(song.song_dir("busy") / "song.json", rec)
    owner = song.life_lock("busy").acquire()
    try:
        with pytest.raises(song.SongError, match="another worker"):
            song.make("busy", "funk", seconds=16, backend_factory=lambda: fake["backend"])
        with pytest.raises(song.SongError, match="being made"):
            song.delete("busy")
    finally:
        owner.release()
    assert song.load("busy") == rec and workqueue.find("song", "busy") is not None
    assert not fake["backend"].submits


def test_a_dead_workers_song_and_a_failed_song_can_be_made_again(fake):
    song._write(song.song_dir("lost") / "song.json", {"name": "lost", "status": "running", "pid": 999999, "added": 1})
    assert song.make("lost", "funk", seconds=16, backend_factory=lambda: fake["backend"])["status"] == "done"
    song._write(song.song_dir("bad") / "song.json", {"name": "bad", "status": "failed", "pid": 999999, "added": 1})
    assert song.make("bad", "funk", seconds=16, backend_factory=lambda: fake["backend"])["status"] == "done"


def test_a_setup_error_does_not_leave_a_queued_record(fake, monkeypatch):
    import cinestudio.workflow as wfmod

    def broken(*a, **k):
        raise RuntimeError("graph file missing")
    monkeypatch.setattr(wfmod, "ProfileWorkflow", broken)
    with pytest.raises(RuntimeError):
        song.make("broken", "funk", seconds=16, backend_factory=lambda: fake["backend"])
    assert song.load("broken")["status"] == "failed"
    with pytest.raises(song.SongError, match="seconds"):
        song.make("tiny", "funk", seconds=2)


class TrackedLock:
    def __init__(self, log):
        self.log = log

    def acquire(self):
        self.log.append("lock")

    def release(self):
        self.log.append("release")


def test_a_lost_watch_drains_before_the_gpu_lock_is_released(fake, monkeypatch):
    log = []
    monkeypatch.setattr(song.server, "gpu_lock", lambda url=None, timeout=0, note=None: TrackedLock(log))

    class LostWatch(FakeBackend):
        def watch(self, jid, on_event):
            raise ConnectionError("event stream lost")
    monkeypatch.setattr(song, "_drain", lambda prefix, jid, backend=None, **k: log.append(("drain", jid)) or "ended")
    with pytest.raises(ConnectionError):
        song.make("lost1", "funk", seconds=16, backend_factory=LostWatch)
    assert log == ["lock", ("drain", "p1"), "release"]
    r = song.load("lost1")
    assert r["status"] == "failed" and r["job"] == "p1" and r["prefix"].startswith("cinestudio/song_lost1_")
    assert r["submit_key"].startswith("song-lost1-")
    monkeypatch.setattr(song, "_drain", lambda prefix, jid, backend=None, **k: "unknown")
    with pytest.raises(song.SongUnresolved):
        song.make("lost2", "funk", seconds=16, backend_factory=LostWatch)
    assert song.load("lost2")["status"] == "unresolved"
    with pytest.raises(song.SongError, match="exists"):                 # an unresolved job is not replaced silently
        song.make("lost2", "funk", seconds=16, backend_factory=lambda: fake["backend"])


def test_drain_needs_evidence_from_an_answering_comfyui(monkeypatch, tmp_path):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    states = iter(["queued", "queued", "done_ok"])
    monkeypatch.setattr(song.server, "alive", lambda url=None: True)
    monkeypatch.setattr(song.server, "prompt_status", lambda url, pid: next(states))
    assert song._drain("pfx", "p1", limit=60, sleep=lambda s: None) == "ended"
    monkeypatch.setattr(song.server, "alive", lambda url=None: False)          # ComfyUI never answers
    assert song._drain("pfx", "p1", limit=0.05, poll=0.01, sleep=lambda s: None) == "unknown"
    monkeypatch.setattr(song.server, "alive", lambda url=None: True)           # never submitted: absent 3 times
    monkeypatch.setattr(song.server, "find_prompt", lambda prefix, url=None: None)
    assert song._drain("pfx", None, limit=60, sleep=lambda s: None) == "ended"


def test_cancel_reaches_a_silent_job(fake):
    import threading

    class Silent(FakeBackend):
        def __init__(self):
            super().__init__()
            self.stopped = threading.Event()

        def watch(self, jid, on_event):                    # no progress events at all (a long model load)
            self.stopped.wait(15)
            return JobState(jid, "canceled")

        def cancel(self, jid):
            super().cancel(jid)
            self.stopped.set()
    b = Silent()
    threading.Timer(0.3, lambda: song.request_cancel("quiet")).start()
    with pytest.raises(workqueue.QueueCanceled):
        song.make("quiet", "funk", seconds=16, backend_factory=lambda: b)
    assert b.canceled == ["p1"] and song.load("quiet")["status"] == "canceled"


def test_short_sounds_give_an_empty_grid_not_an_error(tmp_path):
    for secs, bpm in ((1, 120), (2, 30)):
        p = tmp_path / f"s{secs}.wav"
        click_track(p, bpm=bpm, offset=0.1, seconds=secs)
        b = song.beats(p, bpm)
        assert b["beats"] == [] and b["bpm"] is None
    p = tmp_path / "silence.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\0\0" * 48000 * 9)
    b = song.beats(p, 30)
    assert all(t >= 0 for t in b["beats"])


def test_loop_file_name_has_the_fps(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    d = song.song_dir("clicks")
    d.mkdir(parents=True)
    click_track(d / "song.wav", bpm=118.0, offset=0.37, seconds=24)
    song._write(d / "song.json", {"name": "clicks", "status": "done", "file": str(d / "song.wav"),
                                  "beats": song.beats(d / "song.wav", 120)})
    a = song.loop("clicks", n_beats=8, frames=96, fps=24)
    b = song.loop("clicks", n_beats=8, frames=96, fps=25)
    assert a["file"] != b["file"] and a["file"].endswith("_24fps.wav") and b["file"].endswith("_25fps.wav")
    assert not list(d.glob("*.tmp.wav"))


def test_mix_strips_mixes_two_strips(tmp_path):
    from cinestudio import video
    sr = 48000

    def tone(path, hz, secs):
        y = (np.sin(2 * np.pi * hz * np.arange(int(sr * secs)) / sr) * 12000).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(y.tobytes())
    tone(tmp_path / "a.wav", 440, 2)
    tone(tmp_path / "b.wav", 880, 2)
    # a plays from frame 1 for 1 s; b starts at frame 25 (1 s in) and plays 1 s
    info = {"fps": 24, "frame_start": 1, "frame_end": 72, "strips": [
        {"file": str(tmp_path / "a.wav"), "frame_start": 1, "frame_final_start": 1, "frame_final_end": 25, "volume": 1},
        {"file": str(tmp_path / "b.wav"), "frame_start": 25, "frame_final_start": 25, "frame_final_end": 49,
         "volume": 1}]}
    y = song._mono(video.mix_strips(info, tmp_path / "mix.wav"))
    n = song.SR

    def peak_hz(seg):
        f = np.abs(np.fft.rfft(seg))
        return np.argmax(f) * song.SR / len(seg)
    assert abs(len(y) / n - 3.0) < 0.01
    assert abs(peak_hz(y[int(0.2 * n):int(0.8 * n)]) - 440) < 10
    assert abs(peak_hz(y[int(1.2 * n):int(1.8 * n)]) - 880) < 10
    assert np.abs(y[int(2.2 * n):]).max() < 1e-3
