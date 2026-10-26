"""Templates (dance shots from a song): the plan, the worker with fake Blender, the library (no Blender, no GPU)."""
import json
import os
from pathlib import Path

import pytest

from cinestudio import shots, song


def song_rec(bpm=118.1, seconds=30.0, name="groove"):
    step = 60 / bpm
    beats = [round(0.05 + k * step, 3) for k in range(int((seconds - 0.1) / step))]
    return {"name": name, "status": "done", "bpm": 120, "file": "x.wav",
            "beats": {"bpm": bpm, "beats": beats, "downbeats": beats[2::4], "seconds": seconds}}


def test_loop_plan_fills_the_nearest_beat_count():
    p = shots.plan_dance(song_rec(), ["S1", "S2"], loop=True, seconds=10)
    assert p["n_beats"] == 16 and p["routines"] == ["S1", "S2"]
    assert p["loop_frames"] == 16 * 12 and p["frames"] == p["loop_frames"] + 1 + shots.LOOP_TAIL   # + the tail
    assert p["beat_frames"] == [1 + 12 * k for k in range(17)]                       # the loop ends on count 1 again
    assert abs(p["bpm"] - 118.1) < 2.5                                               # a small tempo change only (120)
    one = shots.plan_dance(song_rec(), ["S1"], loop=True, seconds=10)                 # one routine plays twice
    assert one["routines"] == ["S1", "S1"]
    with pytest.raises(shots.ShotError, match="Finale"):
        shots.plan_dance(song_rec(), ["A", "F"], loop=True)
    with pytest.raises(shots.ShotError, match="fill 16 beats"):
        shots.plan_dance(song_rec(), ["S1", "S2", "A"], loop=True, seconds=10)


def test_plan_without_loop_puts_count_one_on_frame_13():
    r = song_rec()
    p = shots.plan_dance(r, ["A", "F"], loop=False)
    first_down = r["beats"]["downbeats"][0] if r["beats"]["downbeats"][0] >= 0.5 else r["beats"]["downbeats"][1]
    assert abs(p["start"] - (first_down - 0.5)) < 1e-6
    assert p["beat_frames"][0] == 13 and p["n_beats"] == 13 and p["frames"] > p["beat_frames"][13]
    with pytest.raises(shots.ShotError, match="too short"):
        shots.plan_dance(song_rec(seconds=6), ["A", "B", "A"], loop=False)
    with pytest.raises(shots.ShotError, match="not ready"):
        shots.plan_dance({**r, "status": "queued"}, ["A"], loop=False)


def test_parse_routines():
    assert shots.parse_routines("S1, S2~") == ["S1", "S2~"]
    with pytest.raises(shots.ShotError):
        shots.parse_routines("X1")
    with pytest.raises(shots.ShotError):
        shots.parse_routines("")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    d = song.song_dir("groove")
    d.mkdir(parents=True)
    song._write(d / "song.json", song_rec())
    return tmp_path


def test_worker_builds_the_shot_with_fake_blender(home, monkeypatch):
    calls = []

    def fake_blender(script, blend, args, log):
        calls.append((Path(script).name, [str(a) for a in args]))
        Path(args[args.index("--out") + 1]).write_bytes(b"blend")
    monkeypatch.setattr(shots, "_run_blender", fake_blender)
    monkeypatch.setattr(song, "loop", lambda name, n, frames, fps, **kw: {"file": str(home / "loop.wav"), "start": 5.0,
                                                                    "end": 13.0, "tempo_change": 1.002})
    placed = []
    monkeypatch.setattr(song, "place", lambda blend, name, start, out, log=None, sound_file=None:
                        (placed.append((name, start, sound_file)), Path(out).write_bytes(b"blend")))
    monkeypatch.setattr(shots, "make_preview", lambda d, blend, log, **kw: {"preview": str(d / "preview.mp4"),
                                                                       "poster": str(d / "poster.jpg"), "frames": 196})
    spawned = []
    r = shots.start_dance("loop1", "groove", "S1,S2", loop=True, seconds=10,
                          spawner=lambda cmd, log, cwd: (spawned.append(cmd), (4242, True))[1])
    assert spawned[0][1:] == ["-m", "cinestudio", "shot", "work", "loop1"] and r["status"] == "queued"
    assert shots.load("loop1")["pid"] is None                    # the worker writes its own pid, never the parent
    done = shots.make_dance("loop1")
    assert done["status"] == "done" and done["pid"] == os.getpid()
    build = next(a for s, a in calls if s == "build.py")
    assert "--loop" in build and build[build.index("--routine") + 1] == "S1,S2"
    assert build[build.index("--frames") + 1] == str(done["plan"]["frames"])
    assert placed == [("groove", 5.0, str(home / "loop.wav"))]
    assert done["files"]["blend"].endswith("shot.blend") and not (shots.shot_dir("loop1") / "dance.blend").exists()
    assert done["frames"] == 196 and shots.shot_frames(shots.load("loop1")) == 196    # the render's length is stored
    with pytest.raises(shots.ShotError, match="exists"):                              # no overwrite of a done template
        shots.start_dance("loop1", "groove", "S1,S2", spawner=lambda *a: (1, True))


def test_worker_records_a_failure(home, monkeypatch):
    def broken(script, blend, args, log):
        raise shots.ShotError("Blender failed")
    monkeypatch.setattr(shots, "_run_blender", broken)
    shots.prepare_dance("bad", "groove", "A,B", loop=False)
    with pytest.raises(shots.ShotError):
        shots.make_dance("bad")
    rec = shots.load("bad")
    assert rec["status"] == "failed" and "Blender failed" in rec["error"]


def test_library_lists_hand_made_shots_read_only(home):
    hm = shots.shots_dir() / "my_shot"
    hm.mkdir(parents=True)
    (hm / "shot.blend").write_bytes(b"b")
    (hm / "dance_preview.mp4").write_bytes(b"v")
    (shots.shots_dir() / "0123456789abcdef0123" / "passes").mkdir(parents=True)     # a run's pass cache: not a shot
    lost = shots.shot_dir("lost")
    shots._write(lost / "shot.json", {"name": "lost", "status": "running", "pid": 999999, "added": 5})
    got = {r["name"]: r for r in shots.list_shots()}
    assert set(got) == {"my_shot", "lost"}
    assert got["my_shot"]["handmade"] and got["my_shot"]["files"]["preview"].endswith("dance_preview.mp4")
    assert got["lost"]["status"] == "interrupted"
    with pytest.raises(shots.ShotError, match="hand-made"):              # a new template never overwrites it
        shots.prepare_dance("my_shot", "groove", "S1,S2")
    assert shots.delete("lost") == {"deleted": "lost"} and not lost.exists()
    assert json.loads(json.dumps(got["my_shot"]))["status"] == "done"


def test_add_stores_any_blend_as_a_template(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    src = tmp_path / "mine.blend"
    src.write_bytes(b"BLENDER")
    monkeypatch.setattr(shots, "make_preview", lambda d, blend, log, **kw: {"preview": str(d / "preview.mp4"),
                                                                       "poster": str(d / "poster.jpg"), "sound": None,
                                                                       "frames": 48})
    r = shots.add("orbit", src, "one mannequin, the camera orbits")
    d = shots.shot_dir("orbit")
    assert r["status"] == "done" and r["kind"] == "blender" and r["seconds"] == 2.0 and src.is_file()
    assert (d / "shot.blend").read_bytes() == b"BLENDER" and shots.load("orbit")["description"].startswith("one")
    with pytest.raises(shots.ShotError, match="exists"):
        shots.add("orbit", src)
    # an older hand-made shot (shot.blend, no record) is adopted in place, then it can be deleted
    old = shots.shot_dir("old")
    old.mkdir(parents=True)
    (old / "shot.blend").write_bytes(b"B")
    assert shots.add("old", old / "shot.blend", "by hand")["status"] == "done"
    assert shots.delete("old") == {"deleted": "old"} and not old.exists()
    with pytest.raises(shots.ShotError, match="not a saved .blend"):
        shots.add("x", tmp_path / "missing.blend")


def test_one_name_has_one_owner(home, monkeypatch):
    """The lifecycle lock: a second worker, a second request and a delete all stop at once and change nothing."""
    import time
    monkeypatch.setattr(shots, "make_preview", lambda d, blend, log, **kw: {"frames": 1})
    rec = shots.prepare_dance("busy", "groove", "S1,S2", loop=True, seconds=10)
    assert rec["pid"] is None
    with pytest.raises(shots.ShotError, match="already being built"):           # a fresh reservation is busy
        shots.prepare_dance("busy", "groove", "S1,S2", loop=True, seconds=10)
    with pytest.raises(shots.ShotError, match="being built"):
        shots.delete("busy")
    owner = shots.life_lock("busy").acquire()
    try:
        before = shots.load("busy")
        with pytest.raises(shots.ShotError, match="another worker"):
            shots.make_dance("busy")
        assert shots.load("busy") == before
    finally:
        owner.release()
    old = {**shots.load("busy"), "added": time.time() - 3600}                   # an old reservation: its worker died
    shots._write(shots.shot_dir("busy") / "shot.json", old)
    assert [r["status"] for r in shots.list_shots() if r["name"] == "busy"] == ["interrupted"]
    assert shots.delete("busy") == {"deleted": "busy"}


def test_dance_templates_get_their_own_start_noise(tmp_path, monkeypatch):
    from cinestudio import runner
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    for name, kind in (("shuffle", "dance"), ("orbit", "blender")):
        d = shots.shots_dir() / name
        d.mkdir(parents=True)
        (d / "shot.json").write_text(json.dumps({"name": name, "kind": kind, "status": "done"}))
        (d / "shot.blend").write_bytes(b"BLENDER")
    assert runner._library_kind(shots.shots_dir() / "shuffle" / "shot.blend") == "dance"
    assert runner._library_kind(shots.shots_dir() / "orbit" / "shot.blend") == "blender"
    assert runner._library_kind(tmp_path / "elsewhere.blend") is None
    assert runner.DANCE_SOURCE_DENOISE > runner.DEFAULTS["source_denoise"]


def test_built_in_templates_are_listed_read_only(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    lib = tmp_path / "library"
    d = lib / "party"
    d.mkdir(parents=True)
    (d / "shot.blend").write_bytes(b"BLENDER")
    (d / "preview.mp4").write_bytes(b"MP4")
    (d / "shot.json").write_text(json.dumps({"name": "party", "kind": "dance", "order": 1, "description": "a dance",
                                             "files": {"blend": "shot.blend", "preview": "preview.mp4"},
                                             "plan": {"loop": True, "loop_frames": 240, "loop_tail": 24}}))
    monkeypatch.setattr(shots, "BUILTIN", lib)
    rec = shots.load("party")
    assert rec["builtin"] and Path(rec["files"]["blend"]) == d / "shot.blend"
    assert [r["name"] for r in shots.list_shots()] == ["party"]
    assert shots.library_record(d / "shot.blend")["plan"]["loop_frames"] == 240
    with pytest.raises(shots.ShotError, match="built-in"):
        shots.delete("party")
    with pytest.raises(shots.ShotError, match="built-in"):
        shots.check_new_name("party")
    from cinestudio import runner
    assert runner._library_kind(d / "shot.blend") == "dance"


# ---------------------------------------------------------------- clip texts (one prompt per clip, 2026-10-25)
def test_clip_plan_follows_the_runs_clip_split():
    plan = shots.clip_plan(276)
    assert [(c["first"], c["last"]) for c in plan] == [(1, 121), (97, 217), (193, 276)]
    assert plan[1]["start_s"] == 4.0 and plan[2]["end_s"] == 11.5
    assert len(shots.clip_plan(121)) == 1 and shots.clip_plan(None) == []
    assert [c["first"] for c in shots.clip_plan(276, 193)] == [1, 169]          # longer clips: fewer of them
    assert shots.shot_frames({"timeline": {"frames": [1, 276]}}) == 276
    assert shots.shot_frames({"seconds": 10}) == 240 and shots.shot_frames({}) is None


def test_the_built_in_dodgeball_template_has_a_text_per_clip(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(shots, "BUILTIN", Path(shots.__file__).resolve().parent.parent / "templates" / "library")
    v = shots.clip_texts_view(shots._builtin_record("shan_shen_bu_balls"))
    assert sorted(v["clip_texts"]) == ["1", "2", "3"] and len(v["clips"]) == 3
    ctx = v["clip_context"]                                                     # text B (AGENTS.md 7.2)
    assert "faces the camera" in ctx and "the camera remains static" in ctx and "flash step" in ctx
    c1, c2, c3 = (v["clips"][i]["text"] for i in range(3))
    assert "slides toward the right of the picture" in c1 and "left foot" in c1 and "catch" not in c1   # its own actions
    assert "flies" not in c1                                                    # no threat story for layout props
    assert "claps" in c2 and "dodge" not in c2.replace("dodgeball", "") and "catch" not in c2
    assert "catches" in c3 and "kick" not in c3 and "slides" not in c3
    assert not any("faces the camera" in t for t in (c1, c2, c3))                 # the lasting state is the context
    with pytest.raises(shots.ShotError, match="built-in"):
        shots.clip_texts("shan_shen_bu_balls", {"1": "x"})


def test_clip_texts_of_a_user_template_are_stored_and_edited(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    d = shots.shots_dir() / "mine"
    d.mkdir(parents=True)
    (d / "shot.blend").write_bytes(b"BLENDER")
    (d / "shot.json").write_text(json.dumps({"name": "mine", "kind": "blender", "status": "done", "frames": 276,
                                             "files": {"blend": str(d / "shot.blend")}}), encoding="utf-8")
    r = shots.clip_texts("mine", {"1": "  {character}  waves. ", "3": "it bows"}, "{character} faces the camera")
    assert r["clip_texts"] == {"1": "{character} waves.", "3": "it bows"}
    assert [c["text"] for c in r["clips"]] == ["{character} waves.", "", "it bows"]
    assert json.loads((d / "shot.json").read_text())["clip_context"] == "{character} faces the camera"
    r = shots.clip_texts("mine", {"3": ""}, "")
    assert r["clip_texts"] == {"1": "{character} waves."} and r["clip_context"] == ""
    assert shots.clip_texts("mine")["clip_texts"] == {"1": "{character} waves."}            # read only
    with pytest.raises(shots.ShotError, match="has 3 clips"):
        shots.clip_texts("mine", {"4": "x"})
    with pytest.raises(shots.ShotError, match="not a clip number"):
        shots.clip_texts("mine", {"x": "y"})


# ---------------------------------------------------------------- Codex review 2026-10-25 (round 1)
def test_a_bad_template_name_never_reaches_a_path(tmp_path, monkeypatch):
    """P1: the lifecycle lock was made before the name was checked: a "../x" name put a lock file outside the
    library."""
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    for bad in (r"..\..\review-probe", "../review-probe", "../../x"):
        with pytest.raises(shots.ShotError, match="template name"):
            shots.life_lock(bad)
        with pytest.raises(shots.ShotError, match="template name"):
            shots.clip_texts(bad)
        with pytest.raises(shots.ShotError, match="template name"):
            shots.layers(bad)
        with pytest.raises(shots.ShotError, match="template name"):
            shots.delete(bad)
    assert not list(tmp_path.rglob("*.lock"))                       # no lock file anywhere, inside or outside


def test_the_other_libraries_check_the_name_before_their_locks(tmp_path, monkeypatch):
    from cinestudio import character, props, runner
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    for bad in (r"..\..\probe", "../probe"):
        with pytest.raises(character.CharacterError):
            with character.char_lock(bad):
                pass
        with pytest.raises(props.PropError):
            with props.prop_lock(bad):
                pass
        with pytest.raises(song.SongError):
            song.life_lock(bad)
        with pytest.raises(runner.RunError, match="bad run id"):
            runner._life_lock(bad)
    assert not list(tmp_path.rglob("*.lock"))


def test_the_api_refuses_a_traversal_name_for_clip_texts(tmp_path, monkeypatch):
    import asyncio

    from aiohttp.test_utils import TestClient, TestServer

    from cinestudio import webapp
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))

    async def go():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as c:
            for path in ("/api/templates/..%5C..%5Creview-probe/clip-texts",
                         "/api/templates/..%5Creview-probe/layers"):
                r = await c.get(path)
                assert r.status == 400 and not (await r.json())["ok"]
            r = await c.post("/api/templates/..%5C..%5Creview-probe/clip-texts", json={"clip_texts": {"1": "x"}})
            assert r.status == 400
    asyncio.run(go())
    assert not list(tmp_path.rglob("review-probe*"))


def test_a_dance_loop_counts_its_loop_tail(tmp_path, monkeypatch):
    """P1: a dance record has no top-level frames; its seconds leave out the loop tail (233 render frames read as
    208): the clip-text setter refused clip 3 and runs lost its text."""
    assert shots.shot_frames({"plan": {"frames": 233, "seconds": 8.667, "loop": True}}) == 233
    assert shots.shot_frames({"files": {"frames": 233}, "seconds": 8.667}) == 233
    assert shots.shot_frames({"frames": 233, "files": {"frames": 9}}) == 233          # the record's own first
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    d = shots.shots_dir() / "loop2"
    d.mkdir(parents=True)
    (d / "shot.blend").write_bytes(b"blend")
    (d / "shot.json").write_text(json.dumps({"name": "loop2", "kind": "dance", "status": "done",
                                             "plan": {"loop": True, "frames": 233, "seconds": 8.667, "loop_tail": 24},
                                             "files": {"blend": str(d / "shot.blend"), "frames": 233}}))
    view = shots.clip_texts("loop2", {"3": "{character} lands the loop."})
    assert len(view["clips"]) == 3 and view["clip_texts"]["3"] == "{character} lands the loop."
