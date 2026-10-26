"""The app's clip-by-clip loop and the other parity routes (cinestudio/webapi): takes, retake, use-take, continue, score
of a run; a song and the clip gauntlet in a new run; repose and the sheet redraw of a character; the backplate over a
template's set. No GPU: the detached workers are replaced by recorders."""
import asyncio
import json

import cv2
import numpy as np
import pytest
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

from cinestudio import character, cli, jobs, runner, webapp, workqueue
from cinestudio.webapi import backgrounds as bg_api
from cinestudio.webapi import runs as runs_api

RID = "r20261025-100000-abcd"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(jobs.spawn, "detached", lambda *a, **k: (0, True))      # no background workers in tests
    started = []
    monkeypatch.setattr(cli, "_start_detached", lambda rid, new_attempt=False, wait=30:
                        started.append((rid, new_attempt)) or {"ok": True, "id": rid, "worker": "running"})
    return {"tmp": tmp_path, "started": started}


def call(fn):
    async def run():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as client:
            return await fn(client)
    return asyncio.run(run())


def _run(tmp, status="paused", clips=2, made=1, layout=True, takes=None, limit=1):
    """A layout run folder with `made` of `clips` clips made (each with a clip file and saved latents)."""
    d = tmp / "runs" / RID
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "id": RID, "created": "2026-10-25T10:00:00", "quality": "preview", "prompt": {"video": "a wave"},
        "render_mode": "layout" if layout else "v2v", "relay": "latent" if layout else "off"}))
    parts = {}
    for k in range(1, made + 1):
        f = d / "work" / f"clip_{k}.mp4"
        f.parent.mkdir(exist_ok=True)
        f.write_bytes(b"clip")
        parts[str(k)] = {"outputs": {"clip": str(f)}, "relay": {"s1": f"relay_{k}_s1.latent"}}
    st = {"status": status, "steps": {"controls": {"status": "done", "outputs": {"clips": clips}},
                                      "clip": {"status": status, "parts": parts}}}
    if takes:
        st["takes"] = takes
    if limit:
        st["clip_limit"] = limit
    (d / "state.json").write_text(json.dumps(st))
    return d


def test_takes_lists_each_clip_with_its_takes_and_links(home):
    d = _run(home["tmp"])
    tf = d / "takes" / "clip1" / "take1" / "clip.mp4"
    tf.parent.mkdir(parents=True)
    tf.write_bytes(b"take")
    st = json.loads((d / "state.json").read_text())
    st["takes"] = {"1": [{"n": 1, "status": "done", "overrides": {"seed": 7}, "note": "seed", "outputs": {"clip": str(tf)}}]}
    (d / "state.json").write_text(json.dumps(st))

    async def go(c):
        r = (await (await c.get(f"/api/runs/{RID}/takes")).json())["data"]
        assert r["clips_total"] == 2 and r["retake"] is True and r["worker_alive"] is False and r["clip_limit"] == 1
        c1, c2 = r["clips"]
        assert c1["made"] and c1["url"].startswith("/files/runs/") and not c2["made"] and c2["url"] is None
        t = c1["takes"][0]
        assert t["n"] == 1 and t["overrides"] == {"seed": 7} and t["url"].startswith("/files/runs/") and "outputs" not in t
        assert (await c.get(t["url"])).status == 200
    call(go)


def test_a_retake_records_the_take_and_starts_the_worker(home):
    _run(home["tmp"])

    async def go(c):
        r = await c.post(f"/api/runs/{RID}/retake", json={"clip": 1, "seed": "7", "keyframe": "off", "adain": "",
                                                          "ingredients_strength": "0.8", "video": " ", "note": "new seed"})
        d = (await r.json())["data"]
        assert d["take"] == 1 and d["changes"] == {"seed": 7, "keyframe": "off", "ingredients_strength": 0.8}
        assert home["started"] == [(RID, False)]
        assert (await c.post(f"/api/runs/{RID}/retake", json={"clip": 2, "seed": 3})).status == 400   # not made yet
        assert (await c.post(f"/api/runs/{RID}/retake", json={"clip": 1, "keyframe": "maybe"})).status == 400
        assert (await c.post(f"/api/runs/{RID}/retake", json={"seed": 3})).status == 400              # no clip
        t = runner.list_takes(RID)["takes"]["1"]
        assert [x["n"] for x in t] == [1] and t[0]["note"] == "new seed"
    call(go)


def test_a_retake_of_a_stopped_run_without_a_limit_pauses_after_the_made_clips(home):
    d = _run(home["tmp"], status="interrupted", clips=3, made=2, limit=None)

    async def go(c):
        r = (await (await c.post(f"/api/runs/{RID}/retake", json={"clip": 2, "seed": 5})).json())["data"]
        assert r["clip_limit"] == 2                                   # not the take AND clip 3
        assert json.loads((d / "state.json").read_text())["clip_limit"] == 2
    call(go)


def test_a_retake_takes_an_uploaded_look_image_and_drops_it_when_refused(home):
    _run(home["tmp"])
    png = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))[1].tobytes()

    async def go(c):
        fd = FormData()
        fd.add_field("clip", "1")
        fd.add_field("look_image", png, filename="look.png", content_type="image/png")
        d = (await (await c.post(f"/api/runs/{RID}/retake", data=fd)).json())["data"]
        assert d["changes"]["look_image"].endswith(".png")
        fd = FormData()
        fd.add_field("clip", "2")                                                                     # refused
        fd.add_field("look_image", png, filename="look.png", content_type="image/png")
        assert (await c.post(f"/api/runs/{RID}/retake", data=fd)).status == 400
        assert len(list((home["tmp"] / "uploads").glob("upload_*"))) == 1                             # only the kept one
    call(go)


def test_retakes_need_a_layout_run_and_a_stopped_worker(home, monkeypatch):
    _run(home["tmp"], layout=False)

    async def go(c):
        r = await c.get(f"/api/runs/{RID}/takes")
        assert (await r.json())["data"]["retake"] is False
        assert (await c.post(f"/api/runs/{RID}/retake", json={"clip": 1, "seed": 2})).status == 400
        monkeypatch.setattr(runner, "worker_alive", lambda rid: True)
        r = await c.post(f"/api/runs/{RID}/use-take", json={"clip": 1, "take": 1})
        assert r.status == 400 and "stopped" in (await r.json())["error"]
        assert (await c.post(f"/api/runs/{RID}/continue", json={})).status == 400
    call(go)
    assert home["started"] == []


def test_use_take_calls_the_runner(home, monkeypatch):
    _run(home["tmp"])
    seen = []
    monkeypatch.setattr(runner, "use_take", lambda rid, clip, take: seen.append((rid, clip, take)) or {"id": rid})

    async def go(c):
        assert (await c.post(f"/api/runs/{RID}/use-take", json={"clip": "1", "take": "2"})).status == 200
    call(go)
    assert seen == [(RID, 1, 2)]


def test_continue_sets_the_clip_limit_and_starts_the_worker(home):
    d = _run(home["tmp"])

    async def go(c):
        r = await c.post(f"/api/runs/{RID}/continue", json={"through_clip": 2})          # the next clip
        assert (await r.json())["data"]["start"]["ok"]
        assert json.loads((d / "state.json").read_text())["clip_limit"] == 2
        await c.post(f"/api/runs/{RID}/continue", json={"through_clip": 0})              # the rest
        assert "clip_limit" not in json.loads((d / "state.json").read_text())
        runner.set_clip_limit(RID, 1)
        await c.post(f"/api/runs/{RID}/continue")                                        # paused: the next clip
        assert json.loads((d / "state.json").read_text())["clip_limit"] == 2
        st = json.loads((d / "state.json").read_text())
        st["status"] = "interrupted"
        (d / "state.json").write_text(json.dumps(st))
        await c.post(f"/api/runs/{RID}/continue")                                        # stopped: keep the limit
        assert json.loads((d / "state.json").read_text())["clip_limit"] == 2
        rows = (await (await c.get("/api/runs")).json())["data"]["items"]
        assert rows[0]["worker_alive"] is False                                          # the card's Continue
    call(go)
    assert home["started"] == [(RID, False)] * 4


def test_a_done_run_does_not_continue_but_an_unresolved_one_can_submit_again(home):
    d = _run(home["tmp"], status="done", limit=None)

    async def go(c):
        assert (await c.post(f"/api/runs/{RID}/continue", json={})).status == 400
        st = json.loads((d / "state.json").read_text())
        st["status"] = "unresolved"
        (d / "state.json").write_text(json.dumps(st))
        assert (await c.post(f"/api/runs/{RID}/continue", json={"new_attempt": True})).status == 200
    call(go)
    assert home["started"] == [(RID, True)]


def test_score_measures_the_picture_only(home, monkeypatch):
    _run(home["tmp"])
    from cinestudio import clip_score
    seen = []
    monkeypatch.setattr(clip_score, "score", lambda rid, clip, take=None, with_motion=True, **k:
                        seen.append((rid, clip, take, with_motion)) or {"picture": {"sharpness": 50}})

    async def go(c):
        r = await c.post(f"/api/runs/{RID}/score", json={"clip": 1, "take": ""})
        assert (await r.json())["data"]["picture"]["sharpness"] == 50
    call(go)
    assert seen == [(RID, 1, None, False)]


def _ready_character(tmp, name="fox", body=True):
    img = tmp / "sheet.png"
    cv2.imwrite(str(img), np.zeros((64, 64, 3), np.uint8))
    character.import_sheet(name, img, "x")
    c = character.load(name)
    if body:
        c["mannequin"] = {}
    c["files"]["front"] = c["files"]["sheet"]
    character.save(c)
    return c


def test_new_run_passes_the_song_and_the_clip_limit(home, monkeypatch):
    _ready_character(home["tmp"])
    seen = {}

    def fake_new_run(*a, **k):
        seen.update(k)
        d = home["tmp"] / "runs" / RID
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps({"id": RID, "prompt": {"video": "v"}}))
        (d / "state.json").write_text(json.dumps({"status": "new", "steps": {}}))
        return {"id": RID, "prompt": {"video": "v"}}
    monkeypatch.setattr(runner, "new_run", fake_new_run)
    monkeypatch.setattr(workqueue, "add", lambda *a, **k: None)

    async def go(c):
        r = await c.post("/api/runs", json={"blend": "x.blend", "character": "fox", "video": "v", "style": "s",
                                            "song": "funk", "song_start": "2.5", "through_clip": "1"})
        assert (await r.json())["ok"]
        assert seen["song"] == "funk" and seen["song_start"] == 2.5
        assert json.loads((home["tmp"] / "runs" / RID / "state.json").read_text())["clip_limit"] == 1
        seen.clear()
        r = await c.post("/api/runs", json={"blend": "template:tv", "character": "fox", "video": "v", "song": "funk"})
        assert r.status == 400 and "Blender template" in (await r.json())["error"]                # a video template
    call(go)


def test_repose_needs_a_front_and_a_body(home):
    _ready_character(home["tmp"], "fox")
    _ready_character(home["tmp"], "owl", body=False)

    async def go(c):
        r = await c.post("/api/characters/owl/repose", json={})
        assert r.status == 400 and "mannequin body" in (await r.json())["error"]
        d = (await (await c.post("/api/characters/fox/repose", json={})).json())["data"]
        assert d["job"]["action"] == "repose" and d["job"]["status"] == "queued"
    call(go)


def test_redraw_sheet_calls_the_rescale(home, monkeypatch):
    c0 = _ready_character(home["tmp"])
    seen = []
    monkeypatch.setattr(character, "rescale_sheet", lambda name: seen.append(name) or c0)

    async def go(c):
        d = (await (await c.post("/api/characters/fox/redraw-sheet", json={})).json())["data"]
        assert d["character"]["name"] == "fox"
    call(go)
    assert seen == ["fox"]


def test_a_backplate_over_the_set_queues_a_setplate_job(home, monkeypatch):
    from cinestudio import shots
    blend = home["tmp"] / "shots" / "still" / "shot.blend"
    blend.parent.mkdir(parents=True)
    blend.write_bytes(b"blend")
    rec = {"name": "still", "status": "done", "files": {"blend": str(blend)},
           "layers": {"descriptions": {"set": ""}}}
    monkeypatch.setattr(shots, "list_shots", lambda: [rec])
    monkeypatch.setattr(shots, "library_record", lambda p: rec)
    started = []
    monkeypatch.setattr(bg_api.spawn, "detached", lambda cmd, log, cwd: started.append(cmd) or (0, True))
    monkeypatch.setattr(bg_api.time, "sleep", lambda s: None)

    async def go(c):
        assert (await c.post("/api/backgrounds/from-set", json={"shot": ""})).status == 400
        assert (await c.post("/api/backgrounds/from-set", json={"shot": str(home["tmp"] / "x.blend")})).status == 400
        r = await c.post("/api/backgrounds/from-set", json={"shot": str(blend)})                 # no place, no Set words
        assert r.status == 400 and "Set" in (await r.json())["error"]
        rec["layers"]["descriptions"]["set"] = "a market street"
        r = await c.post("/api/backgrounds/from-set", json={"shot": str(blend), "style": "3D", "seed": 4,
                                                             "name": "market"})
        ref = (await r.json())["data"]["ref"]
        assert ref == "bg-setplate-market"
        cmd = started[-1]
        assert cmd[cmd.index("background") + 1] == "from-set" and f"--shot={blend}" in cmd and "--scene=" in cmd
        assert cmd[-2:] == ["--ref", ref] and cmd[cmd.index("--seed") + 1] == "4"
        d = (await (await c.get("/api/backgrounds")).json())["data"]
        assert [(j["ref"], j["kind"]) for j in d["jobs"]] == [(ref, "setplate")]
        assert (await c.post("/api/backgrounds/from-set", json={"shot": str(blend), "name": "market"})).status == 400
    call(go)


def test_a_set_backplate_card_has_its_set_picture(home):
    from cinestudio import background
    src = home["tmp"] / "plate.png"
    cv2.imwrite(str(src), np.full((36, 64, 3), 90, np.uint8))
    rec = background.save_plate(src, "x", 1, scene="a street", name="street", source="set", shot="still")
    cv2.imwrite(str(home["tmp"] / "backgrounds" / "street" / "set.png"), np.zeros((36, 64, 3), np.uint8))
    assert rec["name"] == "street"

    async def go(c):
        b = (await (await c.get("/api/backgrounds")).json())["data"]["backgrounds"][0]
        assert b["source"] == "set" and b["shot"] == "still" and b["set_url"].startswith("/files/backgrounds/street/set.png")
    call(go)


def test_song_args_are_optional():
    assert runs_api._song_args({}, None) == {}
    assert runs_api._song_args({"song": "a", "song_start": ""}, None) == {"song": "a", "song_start": 0.0}
