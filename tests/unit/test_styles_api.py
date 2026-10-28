"""The Styles tab's API (cinestudio/webapi/styles.py): the style library, its use and staleness, the reference picture,
and the style of a character or a template. No GPU, no Blender (the template scan is faked)."""
import asyncio
import json

import aiohttp
import cv2
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from cinestudio import background, character, shots, styles, webapp
from cinestudio.webapi import styles as api

SCAN = {"characters": [{"collection": "Mannequin", "rig": "MannequinRig", "objects": 10}], "props": [],
        "set": ["Floor"], "loose": [], "camera_moves": False, "frames": [1, 100]}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    d = character.library() / "hero"
    d.mkdir(parents=True)
    (d / "character.json").write_text(json.dumps({"name": "hero", "status": "ready", "source": "text", "files": {},
                                                  "style": "old words"}), encoding="utf-8")
    t = shots.shots_dir() / "gym"
    t.mkdir(parents=True)
    (t / "shot.blend").write_bytes(b"BLENDER")
    (t / "shot.json").write_text(json.dumps({"name": "gym", "kind": "blender", "status": "done",
                                             "files": {"blend": str(t / "shot.blend")}}), encoding="utf-8")
    monkeypatch.setattr(shots, "scan_layers", lambda blend: json.loads(json.dumps(SCAN)))
    return tmp_path


def call(fn):
    async def run():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as client:
            return await fn(client)
    return asyncio.run(run())


async def data(r, status=200):
    body = await r.json()
    assert r.status == status, body
    return body.get("data") if status == 200 else body


def png(value=90):
    ok, buf = cv2.imencode(".png", np.full((40, 30, 3), value, np.uint8))
    return buf.tobytes()


def test_the_list_has_the_builtins_the_default_and_the_card_fields(home):
    async def go(c):
        d = await data(await c.get("/api/styles"))
        assert d["default"] == "3d_animated"
        by = {s["name"]: s for s in d["styles"]}
        s = by["3d_animated"]
        assert s["builtin"] and s["has_builtin"] and s["default"] and s["version"] == 1
        assert s["used_by"] == {"characters": [], "backgrounds": []} and s["stale"] == s["used_by"]
        assert s["text"] == styles.text("3d_animated") and s["reference_url"] is None
    call(go)


def test_new_edit_and_version(home):
    async def go(c):
        r = await data(await c.post("/api/styles/new", json={"name": "clay", "label": "Clay", "words": "claymation",
                                                              "light": "soft studio light",
                                                              "lora": {"name": "clay.safetensors", "strength": 0.5}}))
        s = r["style"]
        assert s["version"] == 1 and not s["builtin"] and not s["has_builtin"] and s["lora"]["strength"] == 0.5
        r = await data(await c.post("/api/styles/clay/edit", json={"words": "claymation, stop-motion", "lora": None}))
        assert r["style"]["version"] == 2 and r["style"]["lora"] is None and r["style"]["light"] == "soft studio light"
        fd = aiohttp.FormData({"label": "", "lora_name": "x.safetensors", "lora_strength": "0.3"})   # a form
        r = await data(await c.post("/api/styles/clay/edit", data=fd))
        assert r["style"]["label"] == "clay" and r["style"]["lora"] == {"name": "x.safetensors", "strength": 0.3}
        await data(await c.post("/api/styles/new", json={"name": "clay", "words": "x"}), 400)     # exists
    call(go)


def test_editing_a_builtin_makes_a_copy_and_deleting_it_brings_the_builtin_back(home):
    async def go(c):
        base = styles.load("anime_cel")["version"]
        s = (await data(await c.post("/api/styles/anime_cel/edit", json={"light": "soft rim light"})))["style"]
        assert not s["builtin"] and s["has_builtin"] and s["version"] == base + 1
        assert (home / "styles" / "anime_cel.json").is_file()
        r = await data(await c.post("/api/styles/anime_cel/delete", json={}))
        assert r["back_to_builtin"] is True
        assert styles.load("anime_cel")["builtin"] and styles.load("anime_cel")["version"] == base
        r = await data(await c.post("/api/styles/anime_cel/delete", json={}), 400)           # a built-in stays
        assert "built-in" in r["error"]
    call(go)


def test_set_default(home):
    async def go(c):
        await data(await c.post("/api/styles/anime_cel/default", json={}))
        d = await data(await c.get("/api/styles"))
        assert d["default"] == "anime_cel" and {s["name"]: s["default"] for s in d["styles"]}["anime_cel"]
        await data(await c.post("/api/styles/nope/default", json={}), 400)
    call(go)


def test_a_character_takes_a_style_shows_as_used_and_stale_and_can_drop_it(home):
    async def go(c):
        r = await data(await c.post("/api/characters/hero/style", json={"style": "anime_cel"}))
        assert r["character"]["style_ref"] == "anime_cel" and r["character"]["style"] == styles.text("anime_cel", light=False)
        words = r["character"]["style"]
        d = {s["name"]: s for s in (await data(await c.get("/api/styles")))["styles"]}
        assert d["anime_cel"]["used_by"]["characters"] == ["hero"] and d["anime_cel"]["stale"]["characters"] == []
        await data(await c.post("/api/styles/anime_cel/edit", json={"light": "soft rim light"}))
        d = {s["name"]: s for s in (await data(await c.get("/api/styles")))["styles"]}
        assert d["anime_cel"]["stale"]["characters"] == ["hero"]
        await data(await c.post("/api/characters/hero/use-style", json={"style": "nope"}), 400)
        r = await data(await c.post("/api/characters/hero/use-style", json={"style": ""}))    # none
        assert "style_ref" not in r["character"] and r["character"]["style"] == words       # its words stay
    call(go)


def test_the_old_style_lora_job_still_goes_to_the_job_queue(home, monkeypatch):
    seen = []
    from cinestudio import jobs
    monkeypatch.setattr(jobs.JobRunner, "submit", lambda self, action, name, **kw: seen.append((action, kw)) or {"id": "j1"})

    async def go(c):
        r = await data(await c.post("/api/characters/hero/style", data=aiohttp.FormData({"style_lora": "a.safetensors",
                                                                                          "keep": "0.5"})))
        assert r == {"job": {"id": "j1"}} and seen[0][0] == "style"
    call(go)


def test_a_template_has_no_style_of_its_own(home):
    """The user (2026-10-25): "shouldnt the style be applied by the background and video gen": a template is grey clay
    geometry and motion, reusable in any style. No template style route, no template counts on the style cards."""
    async def go(c):
        r = await c.post("/api/templates/gym/style", json={"style": "3d_animated"})
        assert r.status in (404, 405)
        s = (await data(await c.get("/api/styles")))["styles"][0]
        assert "templates" not in s["used_by"]
    call(go)


def test_a_reference_picture_is_stored_served_replaced_and_removed(home):
    async def go(c):
        fd = aiohttp.FormData({"name": "ink", "label": "Ink", "words": "ink wash painting"})
        fd.add_field("reference", png(), filename="crop.png", content_type="image/png")
        s = (await data(await c.post("/api/styles/new", data=fd)))["style"]
        first = s["reference"]
        assert s["reference_url"].startswith("/files/styles/") and (await c.get(s["reference_url"])).status == 200
        assert first.startswith(str((home / "styles" / "refs").resolve())) and "ink." in first
        fd = aiohttp.FormData()
        fd.add_field("reference", png(200), filename="b.jpg", content_type="image/jpeg")
        s = (await data(await c.post("/api/styles/ink/reference", data=fd)))["style"]
        assert s["reference"] != first and s["version"] == 2
        assert sorted(p.name for p in (home / "styles" / "refs").iterdir()) == [s["reference"].split("\\")[-1].split("/")[-1]]
        dup = aiohttp.FormData({"name": "ink2", "words": "ink wash painting", "copy_reference_from": "ink"})
        s2 = (await data(await c.post("/api/styles/new", data=dup)))["style"]
        assert s2["reference"] and s2["reference"] != s["reference"]          # its own copy
        s = (await data(await c.post("/api/styles/ink/edit", data=aiohttp.FormData({"remove_reference": "1"}))))["style"]
        assert s["reference"] is None and s["reference_url"] is None
        assert [p.name.split(".")[0] for p in (home / "styles" / "refs").iterdir()] == ["ink2"]
        await data(await c.post("/api/styles/ink2/delete", json={}))
        assert not list((home / "styles" / "refs").iterdir())
        bad = aiohttp.FormData()
        bad.add_field("reference", b"not a picture", filename="x.png", content_type="image/png")
        await data(await c.post("/api/styles/ink/reference", data=bad), 400)
        await data(await c.post("/api/styles/ink/reference", data=aiohttp.FormData({"x": "1"})), 400)
    call(go)
    assert not (home / "uploads").is_dir() or not list((home / "uploads").iterdir())


def test_a_failed_save_keeps_the_old_picture_and_leaves_no_files(home):
    async def go(c):
        fd = aiohttp.FormData({"name": "ink", "words": "ink wash"})
        fd.add_field("reference", png(), filename="a.png", content_type="image/png")
        old = (await data(await c.post("/api/styles/new", data=fd)))["style"]["reference"]
        fd = aiohttp.FormData({"words": "x" * 2001})
        fd.add_field("reference", png(10), filename="b.png", content_type="image/png")
        await data(await c.post("/api/styles/ink/edit", data=fd), 400)
        assert styles.load("ink")["reference"] == old and len(list((home / "styles" / "refs").iterdir())) == 1
    call(go)
    assert not (home / "uploads").is_dir() or not list((home / "uploads").iterdir())


@pytest.mark.parametrize("path,body", [
    ("/api/styles/new", {"name": "Bad Name", "words": "x"}),
    ("/api/styles/new", {"name": "ok", "words": ""}),
    ("/api/styles/new", {"name": "ok", "words": "x", "extra": 1}),
    ("/api/styles/new", {"name": "ok", "words": "x", "lora": {"name": "a", "strength": 5}}),
    ("/api/styles/new", {"name": "ok", "words": "x", "lora": {"name": "", "strength": 0.4}}),
    ("/api/styles/new", {"name": "ok", "words": "x", "lora": {"name": "a", "strength": "lots"}}),
    ("/api/styles/new", {"name": "ok", "words": "x", "reference": "C:/Windows/win.ini"}),
    ("/api/styles/new", {"name": "ok", "words": 5}),
    ("/api/styles/nope/edit", {"words": "x"}),
    ("/api/styles/anime_cel/edit", {"words": ""}),
    ("/api/styles/nope/delete", {}),
    ("/api/characters/hero/use-style", {"style": 3}),
    ("/api/characters/nobody/use-style", {"style": "anime_cel"}),
])
def test_invalid_inputs_are_refused_with_400(home, path, body):
    async def go(c):
        r = await data(await c.post(path, json=body), 400)
        assert r["ok"] is False and r["error"]
    call(go)
    assert not (home / "styles" / "ok.json").exists()


def test_a_style_select_and_a_style_text(home):
    assert api.preset_or_words("", "own words") == (None, "own words")
    assert api.preset_or_words("anime_cel", "") == ("anime_cel", "")
    assert api.preset_or_words("anime_cel", "  " + styles.text("anime_cel") + ". ") == ("anime_cel", "")
    assert api.preset_or_words("anime_cel", "my own look") == (None, "my own look")      # changed text: own words
    with pytest.raises(styles.StyleError):
        api.preset_or_words("nope", "")


def test_a_new_background_passes_the_style_preset(home, monkeypatch):
    from cinestudio.webapi import backgrounds as bapi
    started = []
    monkeypatch.setattr(bapi.spawn, "detached", lambda cmd, log, cwd: started.append(cmd) or (0, True))
    monkeypatch.setattr(bapi.time, "sleep", lambda s: None)

    async def go(c):
        await data(await c.post("/api/backgrounds/new", json={"scene": "a harbour", "name": "h1",
                                                              "style": styles.text("anime_cel"), "style_preset": "anime_cel"}))
        assert "--style-preset=anime_cel" in started[-1] and "--style=" in started[-1]
        await data(await c.post("/api/backgrounds/new", json={"scene": "a harbour", "name": "h2", "style": "my look",
                                                              "style_preset": "anime_cel"}))
        assert not any(a.startswith("--style-preset") for a in started[-1]) and "--style=my look" in started[-1]
        await data(await c.post("/api/backgrounds/new", json={"scene": "a harbour", "name": "h3", "style_preset": "nope"}), 400)
    call(go)
    assert background.list_backgrounds() == []
