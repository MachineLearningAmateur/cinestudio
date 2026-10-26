"""The prop library (2026-10-25, the Cast tab's Props): records, picture jobs with stand-ins, edits and undo, skins, the
run's prop_refs, the API, the CLI and the MCP tools (no ComfyUI, no GPU)."""
import argparse
import asyncio
import json
import os
import time

import cv2
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from cinestudio import character, cli, jobs, mcp_server, props, styles, webapp

WORDS = "matte deep violet rubber dodgeball with a thin magenta seam"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(jobs.spawn, "detached", lambda *a, **k: (0, True))    # no background workers in tests
    return tmp_path


def _pic(path, colour=(120, 40, 90), size=(64, 64)):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    cv2.circle(img, (size[0] // 2, size[1] // 2), size[0] // 3, colour, -1)
    cv2.imwrite(str(path), img)
    return path


def _ball(home, **kw):
    a, b = _pic(home / "ball_s505.png"), _pic(home / "ball_s202.png", (200, 30, 30))
    return props.import_picture("dodgeball", a, WORDS, style="anime_cel", candidates=[b], display_name="Dodgeball", **kw)


@pytest.fixture
def fake_gpu(monkeypatch):
    """A stand-in image job: it writes a picture to `out` and records its arguments."""
    calls = []

    def image_job(profile, variant, values, files, out, tag, **kw):
        calls.append({"profile": profile, "variant": variant, "values": values, "files": files, "tag": tag})
        out.parent.mkdir(parents=True, exist_ok=True)
        _pic(out, (10, 200, 10), (64, 64))
        return {"job": "c1", "seconds": 1.5, "file": str(out)}
    monkeypatch.setattr(character, "image_job", image_job)
    return calls


def test_import_list_pick_undo_and_words(home):
    p = _ball(home)
    assert p["name"] == "dodgeball" and p["status"] == "ready" and p["style_ref"] == "anime_cel"
    assert p["seed"] == 505 and [c["seed"] for c in p["candidates"]] == [505, 202]
    assert (props.library() / "dodgeball" / "candidates" / "candidates.jpg").is_file()
    rows = props.list_props()
    assert rows[0]["display_name"] == "Dodgeball" and rows[0]["words"] == WORDS and rows[0]["thumb"].endswith("front.png")
    front = props.library() / "dodgeball" / "front.png"
    before = cv2.imread(str(front))
    p = props.pick("dodgeball", 202)                                   # the old front stays undoable
    assert p["seed"] == 202 and p["edits"][-1]["mode"] == "pick" and not np.array_equal(cv2.imread(str(front)), before)
    p = props.undo("dodgeball")
    assert p["seed"] == 505 and np.array_equal(cv2.imread(str(front)), before) and p["edits"][-1]["undone"]
    assert (props.library() / "dodgeball" / "history" / "front_undone_v1.png").is_file()
    with pytest.raises(props.PropError, match="no change"):
        props.undo("dodgeball")
    p = props.set_words("dodgeball", "  red rubber ball. ", "a round red ball")
    assert p["words"] == "red rubber ball" and p["description"] == "a round red ball"
    with pytest.raises(props.PropError, match="concrete words"):
        props.set_words("dodgeball", " ")
    assert props.set_display_name("dodgeball", "Red  ball")["display_name"] == "Red ball"
    assert props.new_id("Dodgeball") == "dodgeball-2"                 # taken: the next free ID
    with pytest.raises(props.PropError, match="exists already"):
        _ball(home)
    with pytest.raises(props.PropError, match="no prop"):
        props.load("nothing")


def test_an_imported_picture_has_no_default_style(home):
    p = props.import_picture("cone", _pic(home / "cone.png"), "orange traffic cone")
    assert "style_ref" not in p and p["style"] == ""


def test_create_queues_candidates_with_the_prop_prompt(home, fake_gpu):
    p = props.create("ball", WORDS, description="a round matte deep violet rubber ball", style="anime_cel",
                     candidates=2, seed=202)
    assert p["status"] == "candidates" and [c["seed"] for c in p["candidates"]] == [202, 203]
    c = fake_gpu[0]
    assert c["profile"] == "qwenimage" and c["variant"] == "pano" and c["values"]["lora_strength"] == 0.0
    prompt = c["values"]["prompt"]
    assert prompt.startswith("One single object shown alone as a prop reference picture: a round matte deep violet")
    assert styles.text("anime_cel", light=False)[1:40] in prompt and "pure black everywhere" in prompt
    assert c["values"]["negative"].startswith(props.PROP_NEGATIVE) and c["values"]["seed"] == 202
    assert (props.library() / "ball" / "candidates" / "candidates.jpg").is_file()
    p = props.pick("ball", 203)
    assert p["status"] == "ready" and p["files"]["front"].endswith("front.png") and "edits" not in p
    z = props.create("cone", "orange traffic cone", model="z-image", candidates=1)
    assert fake_gpu[-1]["profile"] == "zimage" and "lora_strength" not in fake_gpu[-1]["values"] and z["model"] == "z-image"
    with pytest.raises(props.PropError, match="model must be"):
        props.create("x", "a box", model="flux")


def test_edit_region_and_undo_share_the_character_core(home, fake_gpu):
    _ball(home)
    front = props.library() / "dodgeball" / "front.png"
    orig = cv2.imread(str(front))
    p = props.edit("dodgeball", "make the seam thicker. Keep everything else exactly the same.", region="8,8,40,40",
                   seed=7, settings={"denoise": 0.85})
    e = p["edits"][-1]
    assert e["region"] == [8, 8, 40, 40] and e["settings"]["seed"] == 7 and e["settings"]["denoise"] == 0.85
    assert fake_gpu[-1]["profile"] == "qwenedit" and fake_gpu[-1]["tag"] == "prop_dodgeball_front_edit1"
    assert fake_gpu[-1]["values"].get("cfg_tail_start") in (None, 1.0)
    now = cv2.imread(str(front))
    assert np.array_equal(now[50:, 50:], orig[50:, 50:]) and not np.array_equal(now[20:28, 20:28], orig[20:28, 20:28])
    assert props.undo("dodgeball")["edits"][-1]["undone"] and np.array_equal(cv2.imread(str(front)), orig)
    with pytest.raises(props.PropError, match="no side picture"):
        props.edit("dodgeball", "x", target="side")
    with pytest.raises(props.PropError, match="cannot edit"):
        props.edit("dodgeball", "x", target="back")


def test_a_new_skin_redraws_the_front_in_another_style(home, fake_gpu):
    _ball(home)
    r = props.new_skin("dodgeball", "3d_animated", seed=3)
    new = r["skin"]
    assert new == "dodgeball-3d_animated" and r["skin_of"] == "dodgeball" and r["style_ref"] == "3d_animated"
    assert r["status"] == "ready" and r["words"] == WORDS and r["display_name"] == "Dodgeball"
    ins = fake_gpu[-1]["values"]["instruction"]
    assert ins.startswith("Redraw the object of Picture 1 in this rendering style") and "pure black" in ins
    assert fake_gpu[-1]["values"]["negative"].startswith(props.PROP_NEGATIVE)
    assert [x["name"] for x in props.skins(new)] == ["dodgeball", new]
    with pytest.raises(props.PropError, match="has a skin in"):
        props.new_skin(new, "3d_animated")
    with pytest.raises(props.PropError, match="other skin"):
        props.delete("dodgeball")
    props.rename("dodgeball", "ball")                                  # its skin follows the main record's new ID
    assert props.load(new)["skin_of"] == "ball" and props.load("ball")["files"]["front"].startswith(
        str(props.library() / "ball"))
    props.delete(new)
    props.delete("ball")
    assert props.list_props() == []


def test_prop_jobs_keep_their_own_subject_and_lock(home):
    assert {"prop_create", "prop_edit", "prop_skin"} <= jobs.LONG
    j = jobs._new_job("prop_edit", "dodgeball", {"instruction": "x"}, "cli")
    assert j["prop"] == "dodgeball" and j["character"] is None and jobs.label(j).startswith("dodgeball · prop_edit")
    j.update(owner={"pid": os.getpid(), "via": "cli"}, status="running", created=time.time())
    jobs._write(j)
    assert [r["id"] for r in jobs.list_jobs(None, prop_name="dodgeball")] == [j["id"]]
    assert jobs.list_jobs("dodgeball") == []                           # not a character's job
    _ball(home)
    with pytest.raises(props.PropError, match="waiting or running"):
        props.delete("dodgeball")


def test_prop_refs_fill_the_words_and_the_picture(home):
    _ball(home)
    r = props.resolve_refs({"Ball_*": "dodgeball"})
    assert r["props"] == {"Ball_*": WORDS} and r["prop_images"]["Ball_*"].endswith("front.png")
    words, pics = props.merge_refs({"Ball_*": "dodgeball"}, {"Ball_*": "own words"}, {})
    assert words == {"Ball_*": "own words"} and pics["Ball_*"].endswith("front.png")   # own words win
    assert props.merge_refs(None, {"A": "x"}, None) == ({"A": "x"}, {})
    with pytest.raises(props.PropError, match="bad prop name"):
        props.resolve_refs({"Ball_*": "../x"})
    with pytest.raises(props.PropError, match="no prop"):
        props.resolve_refs({"Ball_*": "nothing"})
    assert props.parse_ref_args(["Ball_*=dodgeball"]) == {"Ball_*": "dodgeball"}
    with pytest.raises(props.PropError, match="PATTERN=PROP_ID"):
        props.parse_ref_args(["dodgeball"])


def call(fn):
    async def run():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as client:
            return await fn(client)
    return asyncio.run(run())


def test_the_props_api(home):
    _ball(home)

    async def go(c):
        rows = (await (await c.get("/api/props")).json())["data"]
        assert rows[0]["name"] == "dodgeball" and rows[0]["thumb_url"].startswith("/files/props/dodgeball/front.png")
        d = (await (await c.get("/api/props/dodgeball")).json())["data"]
        assert d["urls"]["front"].startswith("/files/props/") and d["skins"][0]["name"] == "dodgeball"
        assert (await c.get(d["urls"]["front"])).status == 200
        r = await (await c.post("/api/props/dodgeball/words", json={"words": "red ball"})).json()
        assert r["data"]["prop"]["words"] == "red ball"
        r = await (await c.post("/api/props/dodgeball/pick", json={"seed": 202})).json()
        assert r["data"]["prop"]["edits"][0]["before_url"]
        r = await (await c.post("/api/props/dodgeball/undo", json={"target": "front"})).json()
        assert r["data"]["prop"]["seed"] == 505
        r = await (await c.post("/api/props", json={"name": "Big Cone", "words": "orange cone", "style": "anime_cel",
                                                     "candidates": 2})).json()
        assert r["ok"] and r["data"]["name"] == "big-cone" and r["data"]["job"]["action"] == "prop_create"
        assert r["data"]["job"]["prop"] == "big-cone" and r["data"]["job"]["args"]["style"] == "anime_cel"
        r = await (await c.post("/api/props/dodgeball/edit", json={"instruction": "a thicker seam", "region": "1,1,40,40"})).json()
        assert r["data"]["job"]["action"] == "prop_edit" and r["data"]["job"]["args"]["region"] == "1,1,40,40"
        r = await (await c.post("/api/props/dodgeball/edit", json={"instruction": "x", "steps": 500})).json()
        assert not r["ok"] and "steps" in r["error"]
        r = await (await c.post("/api/props/dodgeball/new-skin", json={"style": "3d_animated"})).json()
        assert r["data"]["job"]["action"] == "prop_skin"
        r = await (await c.post("/api/props/nothing/delete", json={})).json()
        assert not r["ok"] and r["error_type"] == "PropError"
    call(go)


def test_the_run_form_resolves_library_props(home):
    from cinestudio.webapi import runs
    _ball(home)
    out = runs._props_args({"prop_refs": {"Ball_*": "dodgeball"}, "props": {"Box": "a crate"}}, None)
    assert out["props"] == {"Ball_*": WORDS, "Box": "a crate"} and out["prop_images"]["Ball_*"].endswith("front.png")
    with pytest.raises(ValueError, match="Blender templates"):
        runs._props_args({"prop_refs": {"Ball_*": "dodgeball"}}, "a video template")
    with pytest.raises(props.PropError):
        runs._props_args({"prop_refs": {"Ball_*": "../../etc"}}, None)


def test_the_prop_cli(home, capsys):
    img = _pic(home / "cone.png")
    assert cli.main(["prop", "import", "cone", "--image", str(img), "--words", "orange cone", "--display-name", "Cone"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["ok"] and out["prop"]["display_name"] == "Cone"
    assert cli.main(["prop", "words", "cone", "--words", "tall orange cone"]) == 0
    capsys.readouterr()
    assert cli.main(["prop", "list"]) == 0
    assert json.loads(capsys.readouterr().out)["props"][0]["words"] == "tall orange cone"
    assert cli.main(["prop", "undo", "cone"]) == 2                     # nothing to undo: a refusal with its reason
    assert "no change" in json.loads(capsys.readouterr().out)["error"]
    a = argparse.Namespace(prop=None, prop_image=["Box=box.png"], prop_ref=["Ball_*=cone"])
    opts = cli._run_prop_options(a)
    assert opts["props"] == {"Ball_*": "tall orange cone"} and opts["prop_images"]["Box"] == "box.png"
    ns = cli._parser().parse_args(["run", "new", "--blend", "x.blend", "--character", "k", "--video", "v",
                                   "--prop-ref", "Ball_*=cone"])
    assert ns.prop_ref == ["Ball_*=cone"]


def test_the_prop_mcp_tools(home, monkeypatch):
    _ball(home)
    seen = []
    monkeypatch.setattr(mcp_server.jobs, "submit", lambda action, name, via="app", **kw: seen.append((action, name, kw)) or {"id": "j1"})
    mcp_server.create_prop("cone", "orange cone", style="anime_cel", candidates=2)
    mcp_server.edit_prop("dodgeball", "thicker seam", region="1,1,30,30", denoise=0.8)
    mcp_server.prop_skins("dodgeball", new_style="3d_animated")
    assert [s[0] for s in seen] == ["prop_create", "prop_edit", "prop_skin"]
    assert seen[1][2]["settings"] == {"denoise": 0.8} and seen[0][2]["candidates"] == 2
    with pytest.raises(ValueError, match="not one of"):
        mcp_server.create_prop("cone", "orange cone", model="flux")
    assert mcp_server.list_props()["props"][0]["name"] == "dodgeball"
    assert mcp_server.set_prop_words("dodgeball", words="red ball", display_name="Red ball")["prop"]["words"] == "red ball"
    with pytest.raises(ValueError, match="confirm"):
        mcp_server.delete_prop("dodgeball")
    assert mcp_server._prop_options({"Box": "a crate"}, None, {"Ball_*": "dodgeball"})["props"] == {
        "Box": "a crate", "Ball_*": "red ball"}
