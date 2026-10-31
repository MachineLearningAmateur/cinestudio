"""Web app API: JSON envelope, file serving limits, character and run endpoints (no ComfyUI, no GPU)."""

import asyncio
import json
from pathlib import Path

import aiohttp
import cv2
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from cinestudio import character, jobs, webapp


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(jobs.spawn, "detached", lambda *a, **k: (0, True))    # no background workers in tests
    return tmp_path


def call(fn):
    async def run():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as client:
            return await fn(client)
    return asyncio.run(run())


def test_stuck_gpu_meter_does_not_block_apps_or_page(home, monkeypatch):
    import threading
    import types
    from cinestudio import sysinfo
    from cinestudio.webapi import setup

    release = threading.Event()
    sampler = sysinfo._GpuSampler()
    calls = []

    def stuck_driver():
        calls.append(1)
        release.wait()
        return None

    monkeypatch.setattr(sysinfo, "gpu", stuck_driver)
    monkeypatch.setattr(sysinfo, "_gpu_sampler", sampler)
    monkeypatch.setattr(setup, "_gpu", lambda: types.SimpleNamespace(list_apps=lambda: {"apps": []}))

    async def go(c):
        for _ in range(3):
            r = await asyncio.wait_for(c.get("/api/system"), timeout=2)
            data = (await r.json())["data"]
            assert data["gpu"] is None and "cpu" in data and "ram_total" in data
        assert len(calls) == 1
        r = await asyncio.wait_for(c.get("/api/settings/gpu"), timeout=2)
        assert (await r.json())["data"]["apps"] == []
        assert (await asyncio.wait_for(c.get("/"), timeout=2)).status == 200

    try:
        call(go)
    finally:
        pending = sampler._pending
        release.set()
        if pending is not None:
            assert pending.wait(2)


def test_characters_list_and_show(home):
    img = home / "sheet.png"
    cv2.imwrite(str(img), np.zeros((896, 1536, 3), np.uint8))
    character.import_sheet("enemy", img, "Left panel: ...")

    async def go(c):
        r = await c.get("/api/characters")
        body = await r.json()
        assert body["ok"] and body["data"][0]["name"] == "enemy"
        r = await c.get("/api/characters/enemy")
        d = (await r.json())["data"]
        assert d["status"] == "ready" and d["urls"]["sheet"].startswith("/files/characters/enemy/sheet.png")
        r = await c.get(d["urls"]["sheet"])
        assert r.status == 200
        r = await c.get("/api/characters/nobody")
        body = await r.json()
        assert r.status == 400 and body["ok"] is False and body["error_type"] == "CharacterError"
    call(go)


def test_files_cannot_leave_the_library(home):
    (home / "secret.json").write_text("{}")
    character.library().mkdir(parents=True)

    async def go(c):
        assert (await c.get("/files/characters/..%2Fsecret.json")).status in (403, 404)
        assert (await c.get("/files/other/x.png")).status == 404
    call(go)


def test_create_from_text_queues_a_job(home):
    async def go(c):
        r = await c.post("/api/characters", json={"name": "fox", "source": "text", "description": "a fox", "candidates": 2})
        d = (await r.json())["data"]
        assert d["job"]["action"] == "create" and d["job"]["status"] == "queued"
        r = await c.get(f"/api/jobs/{d['job']['id']}")
        assert (await r.json())["data"]["character"] == "fox"
    call(go)


def test_run_needs_a_ready_character(home):
    async def go(c):
        r = await c.post("/api/runs", json={"blend": "x.blend", "character": "nobody", "video": "v"})
        body = await r.json()
        assert r.status == 400 and body["ok"] is False
    call(go)


def test_interrupted_jobs_are_marked_at_start(home):
    jobs.jobs_dir()
    (jobs.jobs_dir() / "j1.json").write_text(json.dumps({"id": "j1", "status": "running", "events": []}))
    jobs.JobRunner()
    assert jobs.load("j1")["status"] == "interrupted"


def test_delete_endpoints(home):
    from cinestudio import runner
    d = runner.runs_dir() / "r-x"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"id": "r-x", "created": "now", "quality": "preview"}))
    (d / "state.json").write_text(json.dumps({"status": "done", "steps": {}}))
    img = home / "sheet.png"
    cv2.imwrite(str(img), np.zeros((64, 64, 3), np.uint8))
    character.import_sheet("enemy", img, "x")

    async def go(c):
        assert (await (await c.post("/api/runs/r-x/delete")).json())["ok"]
        assert not d.exists()
        assert (await (await c.post("/api/characters/enemy/delete")).json())["ok"]
        assert (await (await c.get("/api/characters")).json())["data"] == []
    call(go)


def test_changes_need_the_app_header_and_origin(home):
    async def go():
        async with TestClient(TestServer(webapp.make_app())) as c:            # no app header
            r = await c.post("/api/characters/x/delete")
            assert r.status == 403
            r = await c.post("/api/characters/x/delete", headers={"X-Cinestudio": "1", "Origin": "https://evil.example"})
            assert r.status == 403
            r = await c.get("/api/characters")                               # reading needs no header
            assert r.status == 200
    asyncio.run(go())


def test_host_must_be_the_app(home):
    async def go():
        async with TestClient(TestServer(webapp.make_app(port=8765)), headers={"X-Cinestudio": "1"}) as c:
            r = await c.get("/api/characters", headers={"Host": "rebind.example:8765"})
            assert r.status == 403
            r = await c.get("/api/characters", headers={"Host": "127.0.0.1:8765"})
            assert r.status == 200
    asyncio.run(go())


def test_reference_path_must_stay_in_the_library(home, tmp_path):
    img = home / "sheet.png"
    cv2.imwrite(str(img), np.zeros((64, 64, 3), np.uint8))
    character.import_sheet("enemy", img, "x")
    outside = tmp_path.parent / "secret.png"

    async def go(c):
        r = await c.post("/api/characters/enemy/edit", data={"instruction": "x", "target": "front",
                                                             "reference_path": str(outside)})
        body = await r.json()
        assert r.status == 400 and "reference_path" in body["error"]
    call(go)


def test_files_serve_only_media_with_fixed_types(home):
    lib = character.library() / "x"
    lib.mkdir(parents=True)
    (lib / "page.html").write_text("<script>alert(1)</script>")
    cv2.imwrite(str(lib / "a.png"), np.zeros((4, 4, 3), np.uint8))

    async def go(c):
        assert (await c.get("/files/characters/x/page.html")).status == 404
        r = await c.get("/files/characters/x/a.png")
        assert r.status == 200 and r.headers["Content-Type"].startswith("image/png")
        assert r.headers["X-Content-Type-Options"] == "nosniff"
    call(go)


def test_run_media_reports_what_is_not_made_yet(home, monkeypatch):
    from cinestudio import runner, shot_preview
    d = runner.runs_dir() / "r9"
    (d / "work").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"id": "r9", "pass_key": "k"}))
    (d / "state.json").write_text(json.dumps({"steps": {}}))
    r = shot_preview.run_media("r9")
    assert r["status"] == "unavailable" and r["render"]["status"] == "unavailable" and r["control"]["status"] == "unavailable"


def _song_record(home, name, **rec):
    d = home / "songs" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "song.json").write_text(json.dumps({"name": name, "tags": "funk", "bpm": 120, "seconds": 30.0,
                                             "key": "E minor", "seed": 31, **rec}))
    return d


def test_song_new_starts_a_detached_worker(home, monkeypatch):
    import os
    import time
    calls = []

    def fake_detached(cmd, log, cwd):                      # the worker writes its record at once, like song.make
        calls.append(cmd)
        name = cmd[cmd.index("make") + 1]
        from cinestudio import workqueue
        assert workqueue.find("song", name) is not None    # the queue place is reserved BEFORE the worker starts
        _song_record(home, name, status="queued", pid=os.getpid(), added=time.time())
        return os.getpid(), True
    monkeypatch.setattr(webapp.spawn, "detached", fake_detached)

    async def go(c):
        r = await c.post("/api/songs/new", json={"name": "groove", "tags": "funk, slap bass", "bpm": 96, "seconds": 20,
                                                 "key": "A minor", "seed": 7, "lyrics": ""})
        body = await r.json()
        assert body["ok"] and body["data"]["status"] == "queued"
        cmd = calls[0]
        assert cmd[1:5] == ["-m", "cinestudio", "song", "make"] and cmd[5] == "groove"
        assert cmd[cmd.index("--bpm") + 1] == "96" and cmd[cmd.index("--key") + 1] == "A minor"
        assert "--lyrics=[instrumental]" in cmd and "--tags=funk, slap bass" in cmd and cmd[cmd.index("--seed") + 1] == "7"
        songs = (await (await c.get("/api/songs")).json())["data"]
        assert songs["songs"][0]["name"] == "groove" and songs["songs"][0]["status"] == "queued"
        assert "E minor" in songs["keys"]
        r = await c.post("/api/songs/new", json={"name": "groove", "tags": "funk"})       # already being made
        assert r.status == 400 and (await r.json())["error_type"] == "SongError"
        r = await c.post("/api/songs/groove/delete")                                       # refused while queued
        assert r.status == 400 and (home / "songs" / "groove").is_dir()
        for bad in ({"name": "no spaces", "tags": "funk"}, {"name": "ok", "tags": "funk", "key": "H minor"},
                    {"name": "ok", "tags": ""}, {"name": "ok", "tags": "funk", "bpm": 5}):
            r = await c.post("/api/songs/new", json=bad)
            assert r.status == 400, bad
        assert len(calls) == 1
    call(go)


def test_song_audio_is_served_and_deleted(home):
    import wave
    d = _song_record(home, "done1", status="done", added=1.0,
                     beats={"bpm": 118.1, "beats": [0.5, 1.0, 1.5], "downbeats": [0.5], "seconds": 30.0})
    with wave.open(str(d / "song.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\0\0" * 800)
    _song_record(home, "lost", status="running", pid=999999, added=2.0)                    # its worker is gone

    async def go(c):
        songs = {s["name"]: s for s in (await (await c.get("/api/songs")).json())["data"]["songs"]}
        s = songs["done1"]
        assert s["audio_url"].startswith("/files/songs/done1/song.wav")
        assert s["beats"] == {"bpm": 118.1, "count": 3, "first_downbeat": 0.5, "seconds": 30.0}
        assert songs["lost"]["status"] == "interrupted" and songs["lost"]["audio_url"] is None
        r = await c.get(s["audio_url"])
        assert r.status == 200 and r.headers["Content-Type"] == "audio/wav"
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        r = await c.post("/api/songs/new", json={"name": "done1", "tags": "funk"})        # a done song is not overwritten
        assert r.status == 400
        assert (await (await c.post("/api/songs/done1/delete")).json())["ok"]
        assert not d.exists()
        assert (await (await c.post("/api/songs/lost/delete")).json())["ok"]               # an interrupted song can go
        assert (await c.get("/files/songs/..%2Fsecret.wav")).status in (403, 404)
    call(go)


def test_templates_endpoints(home, monkeypatch):
    import os
    import time
    from cinestudio import shots, song
    _song_record(home, "groove", status="done", added=1.0, file=str(home / "songs" / "groove" / "song.wav"),
                 beats={"bpm": 118.1, "beats": [round(0.05 + k * 60 / 118.1, 3) for k in range(50)],
                        "downbeats": [round(0.05 + k * 60 / 118.1, 3) for k in range(2, 50, 4)], "seconds": 30.0})
    hm = home / "shots" / "by_hand"
    hm.mkdir(parents=True)
    (hm / "shot.blend").write_bytes(b"b")
    spawned = []

    def fake_detached(cmd, log, cwd):
        spawned.append(cmd)
        return os.getpid(), True
    from cinestudio import spawn
    monkeypatch.setattr(spawn, "detached", fake_detached)

    async def go(c):
        d = (await (await c.get("/api/templates")).json())["data"]
        assert [s["name"] for s in d["songs"]] == ["groove"] and {r["id"] for r in d["routines"]} >= {"S1", "S2", "F"}
        assert d["shots"][0]["name"] == "by_hand" and d["shots"][0]["handmade"]
        r = await c.post("/api/templates/new", json={"name": "loop1", "song": "groove", "routines": "S1,S2",
                                                    "loop": True, "seconds": 10})
        assert r.status == 200, await r.text()
        assert spawned[0][-3:] == ["shot", "work", "loop1"]
        d = (await (await c.get("/api/templates")).json())["data"]
        t = next(s for s in d["shots"] if s["name"] == "loop1")
        assert t["status"] == "queued" and t["beats"] == 16 and t["routines"] == ["S1", "S2"]
        r = await c.post("/api/templates/new", json={"name": "loop2", "song": "groove", "routines": "S1,S2,A",
                                                    "loop": True, "seconds": 10})
        assert r.status == 400 and "fill 16 beats" in (await r.json())["error"]
        r = await c.post("/api/templates/loop1/delete")                     # its worker (this process) is alive
        assert r.status == 400
        shots._write(shots.shot_dir("loop1") / "shot.json",
                     {**shots.load("loop1"), "status": "done", "added": time.time(),
                      "files": {"blend": str(hm / "shot.blend")}})
        listed = (await (await c.get("/api/shots")).json())["data"]
        assert {(s["name"], s["template"]) for s in listed} == {("loop1", True), ("by_hand", False)}
        r = await c.post("/api/templates/loop1/delete")
        assert r.status == 200 and not (home / "shots" / "loop1").exists()
        r = await c.post("/api/templates/by_hand/delete")                  # every template can be deleted
        assert r.status == 200 and not hm.exists()
    call(go)


def test_template_mask_endpoints(home, monkeypatch):
    import subprocess
    from cinestudio import figure_mask, shots, video
    src = home / "src.mp4"
    subprocess.run([video.ffmpeg(), "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x32:r=24:d=1",
                    "-frames:v", "24", "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True,
                   stdin=subprocess.DEVNULL)
    shots.add_video("tv", src, "stand-in")
    hm = home / "shots" / "by_hand"
    hm.mkdir(parents=True)
    (hm / "shot.blend").write_bytes(b"b")
    started = []
    monkeypatch.setattr(figure_mask, "start", lambda name, clicks: started.append((name, clicks)) or {"status": "queued"})

    async def go(c):
        r = await c.get("/api/templates/tv/mask")
        d = (await r.json())["data"]
        assert d["video"]["width"] == 64 and d["video"]["url"].startswith("/files/shots/tv/") and d["mask"] is None
        r = await c.post("/api/templates/tv/mask", json={"clicks": [{"frame": 1, "x": 2, "y": 3, "positive": True}]})
        assert r.status == 200 and started == [("tv", [{"frame": 1, "x": 2, "y": 3, "positive": True}])]
        r = await c.get("/api/templates/by_hand/mask")                     # a Blender shot has no figure mask
        assert r.status == 400
        cards = {t["name"]: t for t in (await (await c.get("/api/templates")).json())["data"]["shots"]}
        assert cards["tv"]["figure"] is None and cards["by_hand"]["figure"] is None
    call(go)


def test_runs_come_in_pages_and_all(home):
    for i in range(30):
        d = home / "runs" / f"r20261020-1000{i:02d}-abcd"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"id": d.name, "created": "2026-10-20T10:00:00", "quality": "final"}))
        (d / "state.json").write_text(json.dumps({"status": "done"}))

    async def go(c):
        d = (await (await c.get("/api/runs?offset=0&limit=12")).json())["data"]
        assert d["total"] == 30 and len(d["items"]) == 12 and d["items"][0]["id"].endswith("100029-abcd")   # newest first
        d = (await (await c.get("/api/runs?offset=24&limit=12")).json())["data"]
        assert len(d["items"]) == 6 and d["items"][-1]["id"].endswith("100000-abcd")                        # the last page
        d = (await (await c.get("/api/runs?limit=0")).json())["data"]
        assert len(d["items"]) == 30                                                                         # all
    call(go)


def test_import_a_video_template_and_refuse_other_files(home, tmp_path):
    import subprocess
    from aiohttp import FormData
    from cinestudio import video
    clip = tmp_path / "standin.mp4"
    subprocess.run([video.ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x32:r=24:d=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)

    async def go(c):
        fd = FormData()
        fd.add_field("name", "my_standin")
        fd.add_field("description", "one person waves")
        fd.add_field("file", clip.read_bytes(), filename="standin.mp4", content_type="video/mp4")
        r = await c.post("/api/templates/import", data=fd)
        body = await r.json()
        assert r.status == 200 and body["data"]["name"] == "my_standin" and body["data"]["kind"] == "video"
        assert not list((home / "uploads").glob("template_*"))             # the upload was moved or removed
        fd = FormData()
        fd.add_field("name", "bad")
        fd.add_field("file", b"x", filename="notes.txt", content_type="text/plain")
        r = await c.post("/api/templates/import", data=fd)
        assert r.status == 400 and "unsupported" in (await r.json())["error"]
    call(go)


def test_new_video_form_choices_are_checked():
    from cinestudio.webapi.runs import _choice
    assert _choice({"proxy": ""}, "proxy", ("auto", "off")) is None              # empty = the run default
    assert _choice({"background_ground": "dome"}, "background_ground", ("dome", "depth")) == "dome"
    with pytest.raises(ValueError):
        _choice({"look_keyframes": "sometimes"}, "look_keyframes", ("auto", "off"))


def test_character_list_has_style_and_proxy(home):
    img = home / "sheet.png"
    cv2.imwrite(str(img), np.zeros((896, 1536, 3), np.uint8))
    character.import_sheet("enemy", img, "Left panel: ...")

    async def go(c):
        row = (await (await c.get("/api/characters")).json())["data"][0]
        assert "style" in row and row["proxy"] is False
    call(go)


def test_new_character_takes_a_body_type(home, tmp_path):
    img = tmp_path / "front.png"
    cv2.imwrite(str(img), np.full((300, 200, 3), 120, np.uint8))

    async def go(c):
        def form(name, body):
            f = aiohttp.FormData()
            f.add_field("name", name); f.add_field("source", "images"); f.add_field("body", body)
            f.add_field("front", img.read_bytes(), filename="front.png", content_type="image/png")
            return f
        d = (await (await c.post("/api/characters", data=form("bot", "other"))).json())["data"]
        assert character.load("bot")["body"] == "other" and d["character"]
        r = await c.post("/api/characters", data=form("bad", "crab"))
        assert r.status == 400
        r = await c.post("/api/characters/bot/repose")
        assert r.status >= 400                                          # the app has no repose action any more
    call(go)



def test_a_display_name_makes_the_id_and_a_rename_moves_nothing(home):
    """User, 2026-10-24: names with spaces ("automatically replace spaces as needed for the id"); the ID stays in every
    path (sheets, views, jobs, runs) and the app's rename changes only the display name."""
    import aiohttp
    assert character.slugify("Masked Wanderer 3D") == "masked-wanderer-3d"
    assert character.slugify("  Kitsune  (Rest) ") == "kitsune-rest"
    assert character.slugify("Ünïcode Fox!") == "unicode-fox"
    assert character.slugify("!!!") == "character"
    assert len(character.slugify("x" * 200)) == 48
    ok, buf = cv2.imencode(".png", np.zeros((896, 1536, 3), np.uint8))

    async def go(c):
        ids = []
        for _ in range(2):                                    # the same display name twice: two IDs
            fd = aiohttp.FormData()
            fd.add_field("name", "Masked Wanderer 3D")
            fd.add_field("source", "sheet")
            fd.add_field("image", buf.tobytes(), filename="sheet.png", content_type="image/png")
            d = (await (await c.post("/api/characters", data=fd)).json())["data"]
            ids.append(d["character"]["name"])
            assert d["character"]["display_name"] == "Masked Wanderer 3D"
        assert ids == ["masked-wanderer-3d", "masked-wanderer-3d-2"]
        sheet = character.load(ids[0])["files"]["sheet"]
        r = await c.post(f"/api/characters/{ids[0]}/rename", json={"new_name": "The  Wanderer"})
        d = (await r.json())["data"]["character"]
        assert d["name"] == ids[0] and d["display_name"] == "The Wanderer"
        assert character.load(ids[0])["files"]["sheet"] == sheet and Path(sheet).is_file()   # nothing moved
        rows = (await (await c.get("/api/characters")).json())["data"]
        assert {r["name"]: r["display_name"] for r in rows}[ids[0]] == "The Wanderer"
        assert (await c.post(f"/api/characters/{ids[0]}/rename", json={"new_name": "   "})).status == 400
    call(go)


def test_new_video_form_passes_the_video_model(monkeypatch):
    from cinestudio import cli, runner, workqueue
    from cinestudio.webapi import runs as runs_api
    seen = {}
    monkeypatch.setattr(runs_api.character, "for_run", lambda name: ("sheet.png", "text"))
    monkeypatch.setattr(runner, "new_run", lambda *a, **k: seen.update(k) or {"id": "r1"})
    monkeypatch.setattr(runner, "queue_label", lambda m: "label")
    monkeypatch.setattr(workqueue, "add", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_start_detached", lambda rid: {"ok": True})
    form = {"blend": "shot.blend", "character": "k", "video": "v"}
    runs_api._run_new({**form, "video_model": "distilled"})
    assert seen["video_model"] == "distilled"
    with pytest.raises(ValueError, match="video_model"):
        runs_api._run_new({**form, "video_model": "full"})
    runs_api._run_new(form)
    assert seen["video_model"] is None                        # empty = the run default (distilled for layout)
    with pytest.raises(ValueError):
        runs_api._run_new({**form, "video_model": "turbo"})
