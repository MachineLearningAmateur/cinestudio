"""Built-in backgrounds (backgrounds/library, read-only, 2026-10-25) and a template's default background (shot.json
"background"): load/list with the user's library first, nothing writes into a built-in, the API card, the CLI/MCP
default of `run new`, the export of a built-in template. No GPU, no Blender."""
import asyncio
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from cinestudio import background, pano_depth, shots, webapp


@pytest.fixture
def lib(tmp_path, monkeypatch):
    """A user home with one panorama, and a built-in library with a backplate (drawn over a set) and a panorama."""
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    built = tmp_path / "builtin_bg"
    monkeypatch.setattr(background, "BUILTIN", built)
    d = built / "gym"
    d.mkdir(parents=True)
    img = np.full((112, 192, 3), 70, np.uint8)
    cv2.imwrite(str(d / "plate.png"), img)
    cv2.imwrite(str(d / "set.png"), np.full((112, 192, 3), 128, np.uint8))
    (d / "bg.json").write_text(json.dumps({"kind": "plate", "source": "set", "shot": "balls", "scene": "a gym",
                                           "sha1": background.pixels_sha1(img), "created": 1.0, "order": 1}))
    p = built / "sky"
    p.mkdir()
    cv2.imwrite(str(p / "pano.png"), np.full((64, 128, 3), 200, np.uint8))
    (p / "bg.json").write_text(json.dumps({"scene": "a sky", "created": 2.0, "order": 2}))
    src = tmp_path / "pano.png"
    cv2.imwrite(str(src), np.full((64, 128, 3), 90, np.uint8))
    background.save(src, "a rooftop", 1, scene="a rooftop", name="roof")
    return {"tmp": tmp_path, "built": built, "src": src}


def call(fn):
    async def run():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as client:
            return await fn(client)
    return asyncio.run(run())


def test_builtins_load_and_list_after_the_users_own(lib):
    rows = background.list_backgrounds()
    assert [r["name"] for r in rows] == ["roof", "gym", "sky"]
    assert [r["builtin"] for r in rows] == [False, True, True]
    g = background.load("gym")
    assert g["kind"] == "plate" and g["builtin"] and Path(g["file"]) == lib["built"] / "gym" / "plate.png"
    assert background.path("sky") == lib["built"] / "sky" / "pano.png"
    assert [r["name"] for r in background.list_backgrounds("plate")] == ["gym"]
    assert background.is_builtin("gym") and not background.is_builtin("roof") and not background.is_builtin("nope")
    assert background.taken("gym") and background.taken("roof") and not background.taken("free")


def test_a_user_background_of_the_same_name_wins(lib):
    d = background.library_dir() / "sky"                   # e.g. the user's own copy made before it was built in
    d.mkdir()
    cv2.imwrite(str(d / "pano.png"), np.full((64, 128, 3), 10, np.uint8))
    (d / "bg.json").write_text(json.dumps({"scene": "my sky", "created": 5.0}))
    rows = background.list_backgrounds()
    assert [r["name"] for r in rows].count("sky") == 1
    sky = background.load("sky")
    assert not sky["builtin"] and sky["scene"] == "my sky" and not background.is_builtin("sky")
    assert background.has_builtin("sky")
    background.delete("sky")                               # the user's copy goes; the built-in comes back
    assert background.load("sky")["builtin"]


def test_nothing_writes_into_a_builtin(lib):
    before = sorted(p.name for p in (lib["built"] / "gym").iterdir())
    with pytest.raises(background.BackgroundError, match="built-in"):
        background.delete("gym")
    with pytest.raises(background.BackgroundError, match="built-in"):
        background.stamp_style("gym", "anime_cel")
    with pytest.raises(background.BackgroundError, match="built-in"):
        background.save(lib["src"], "x", 1, name="sky")
    with pytest.raises(background.BackgroundError, match="built-in"):
        background.save_plate(lib["src"], name="gym")
    with pytest.raises(background.BackgroundError, match="built-in"):
        background.import_plate(lib["src"], name="gym")
    with pytest.raises(background.BackgroundError, match="not taken"):
        background.make_plate("a place", lib["src"], name="gym")
    with pytest.raises(background.BackgroundError, match="not taken"):
        background.make_set_plate("a place", lib["src"], name="sky")
    with pytest.raises(background.BackgroundError, match="exists already"):
        background.make("a place", name="gym")
    assert sorted(p.name for p in (lib["built"] / "gym").iterdir()) == before
    assert not (background.library_dir() / "gym").exists() and not (background.library_dir() / "sky").exists()


def test_the_same_picture_saved_without_a_name_is_the_builtin_itself(lib, monkeypatch):
    img = cv2.imread(str(lib["built"] / "gym" / "plate.png"))
    sha = background.pixels_sha1(img)
    monkeypatch.setattr(background, "BUILTIN", lib["built"])
    (lib["built"] / f"plate_{sha[:12]}").mkdir()
    cv2.imwrite(str(lib["built"] / f"plate_{sha[:12]}" / "plate.png"), img)
    (lib["built"] / f"plate_{sha[:12]}" / "bg.json").write_text(json.dumps({"kind": "plate", "sha1": sha}))
    r = background.save_plate(lib["built"] / "gym" / "plate.png")
    assert r["builtin"] and r["name"] == f"plate_{sha[:12]}"


def test_a_builtin_panoramas_depth_goes_to_the_users_cache(lib):
    pano = background.path("sky")
    c = pano_depth.cached(pano)
    home = Path(background.library_dir()).parent
    assert home in c.parents and lib["built"] not in c.parents and c.name == "pano_depth.npy"
    user = background.path("roof")
    assert pano_depth.cached(user) == user.with_name("pano_depth.npy")      # the user's: next to the picture


def test_the_api_card_says_builtin_and_the_delete_is_refused(lib):
    async def go(c):
        d = (await (await c.get("/api/backgrounds")).json())["data"]
        by = {b["name"]: b for b in d["backgrounds"]}
        assert by["gym"]["builtin"] is True and by["roof"]["builtin"] is False
        assert by["gym"]["shot"] == "balls" and by["gym"]["source"] == "set"
        assert by["gym"]["url"].startswith("/files/backgrounds_builtin/")
        assert (await c.get(by["gym"]["url"])).status == 200
        assert by["gym"]["set_url"] and (await c.get(by["gym"]["set_url"])).status == 200
        r = await c.post("/api/backgrounds/gym/delete", json={})
        assert r.status == 400 and "built-in" in (await r.json())["error"]
        r = await c.post("/api/backgrounds/new", json={"scene": "a harbour", "name": "gym"})
        assert r.status == 400                                          # a built-in's name is taken
        assert (lib["built"] / "gym" / "plate.png").is_file()
    call(go)


def _template(root: Path, name: str, rec: dict) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "shot.blend").write_bytes(b"BLENDER")
    (d / "shot.json").write_text(json.dumps({"name": name, "kind": "blender", "status": "done",
                                             "files": {"blend": "shot.blend"}, **rec}))
    return d / "shot.blend"


def test_a_templates_default_background_and_the_run_new_default(lib, monkeypatch):
    tpl = lib["tmp"] / "builtin_tpl"
    monkeypatch.setattr(shots, "BUILTIN", tpl)
    blend = _template(tpl, "balls", {"background": "gym"})
    plain = _template(tpl, "plain", {})
    gone = _template(tpl, "gone", {"background": "nothing"})
    assert shots.default_background(blend) == "gym"
    assert shots.default_background(plain) is None and shots.default_background(gone) is None
    assert shots.default_background(lib["tmp"] / "elsewhere.blend") is None
    # nothing asked: the default; anything asked: as asked; "none": the template's own set
    assert shots.run_background(blend, None) == ("saved:gym", "gym")
    assert shots.run_background(blend, "none") == ("", None)
    assert shots.run_background(blend, "saved:roof") == ("saved:roof", None)
    assert shots.run_background(blend, None, background_text="a beach") == (None, None)
    assert shots.run_background(blend, None, look_scene="a beach") == (None, None)
    assert shots.run_background(plain, None) == (None, None)            # no default: the old behaviour
    # a user template of the same name hides the built-in in the list, but the built-in .blend keeps its own record
    user = shots.shots_dir() / "balls"
    user.mkdir(parents=True)
    (user / "shot.blend").write_bytes(b"BLENDER")
    (user / "shot.json").write_text(json.dumps({"name": "balls", "kind": "blender", "status": "done",
                                                "files": {"blend": str(user / "shot.blend")}}))
    listed = [r for r in shots.list_shots() if r["name"] == "balls"]
    assert len(listed) == 1 and not listed[0].get("builtin")
    assert shots.library_record(blend)["builtin"] and shots.default_background(blend) == "gym"
    assert shots.default_background(user / "shot.blend") is None


def test_the_shots_api_and_the_template_card_carry_the_default_background(lib, monkeypatch):
    tpl = lib["tmp"] / "builtin_tpl"
    monkeypatch.setattr(shots, "BUILTIN", tpl)
    _template(tpl, "balls", {"background": "gym", "builtin": True})

    async def go(c):
        rows = (await (await c.get("/api/shots")).json())["data"]
        assert [r["background"] for r in rows if r["name"] == "balls"] == ["gym"]
        d = (await (await c.get("/api/templates")).json())["data"]
        card = next(t for t in d["shots"] if t["name"] == "balls")
        assert card["builtin"] and card["background"]["name"] == "gym" and card["background"]["kind"] == "plate"
        assert card["background"]["url"].startswith("/files/backgrounds_builtin/")
    call(go)


def test_the_cli_run_new_takes_the_default_and_says_so(lib, monkeypatch, capsys):
    from cinestudio import character, cli, runner
    tpl = lib["tmp"] / "builtin_tpl"
    monkeypatch.setattr(shots, "BUILTIN", tpl)
    blend = _template(tpl, "balls", {"background": "gym"})
    seen = []
    monkeypatch.setattr(character, "for_run", lambda name: ("sheet.png", "a hero"))
    monkeypatch.setattr(runner, "new_run", lambda *a, **k: seen.append(k) or {"id": "r1"})
    base = ["run", "new", "--blend", str(blend), "--character", "hero", "--video", "it dodges"]
    assert cli.main(base) == 0
    assert seen[-1]["background"] == "saved:gym"
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip()]
    assert lines[0]["event"] == "note" and "saved:gym" in lines[0]["note"] and lines[-1]["ok"]
    assert cli.main(base + ["--background", "none"]) == 0 and seen[-1]["background"] == ""
    assert cli.main(base + ["--background", "text"]) == 0 and seen[-1]["background"] == "text"


def test_export_writes_the_default_background_and_refuses_a_user_one(lib, monkeypatch, tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "setup"))
    import export_template
    from cinestudio import runner
    out = tmp_path / "tpl_out"
    monkeypatch.setattr(shots, "BUILTIN", out)
    monkeypatch.setattr(runner, "snapshot_blend", lambda src, dst, rep: dst.write_bytes(src.read_bytes()))
    monkeypatch.setattr(export_template.shot_check, "check", lambda name: {"checks": [], "errors": 0})   # no Blender
    monkeypatch.setattr(export_template.shot_check, "check_blend",          # the shipped .blend's own check
                        lambda blend, rec, name: {"checks": [], "errors": 0, "warnings": 0, "at": 1.0, "sha": "s"})
    d = shots.shots_dir() / "balls"
    d.mkdir(parents=True)
    (d / "shot.blend").write_bytes(b"BLENDER")
    (d / "shot.json").write_text(json.dumps({"name": "balls", "kind": "blender", "status": "done", "pid": 4,
                                             "source": "C:/Users/me/x.blend", "layers": {"descriptions": {"set": "a gym"}},
                                             "files": {"blend": str(d / "shot.blend")}}))
    with pytest.raises(SystemExit, match="built-in background"):
        export_template.export("balls", 4, "a dodge", background="roof")       # a user background: not shipped
    dst = export_template.export("balls", 4, "a dodge", background="gym")
    rec = json.loads((dst / "shot.json").read_text())
    assert rec["background"] == "gym" and rec["builtin"] and rec["order"] == 4
    assert rec["layers"] == {"descriptions": {"set": "a gym"}} and "source" not in rec and "pid" not in rec
    assert rec["files"] == {"blend": "shot.blend"}
