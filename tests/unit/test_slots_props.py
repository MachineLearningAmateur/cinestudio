"""Template slots (one character select per Mannequin* layer) and prop groups (the run's own words for the template's
objects), 2026-10-25. No Blender: the layer scans are written by hand."""
import asyncio
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from cinestudio import cli, jobs, mcp_server, runner, shots, webapp
from cinestudio.webapi import runs as wr

BALLS = [{"name": n, "animated": True, "parent": n[:-5] if n.endswith("_Seam") else None}     # a ball and its seam
         for k in ("Dodge1", "Dodge2", "Catch") for n in (f"Ball_{k}", f"Ball_{k}_Seam")]


def duo_record(screen=True):
    chars = [{"collection": "Mannequin2", "rig": "MannequinRig2"}, {"collection": "Mannequin", "rig": "MannequinRig"}]
    if screen:
        chars[0]["screen"] = {"head": [0.67, 0.24], "feet": [0.66, 0.74]}
        chars[1]["screen"] = {"head": [0.29, 0.16], "feet": [0.30, 0.92]}
    return {"name": "duo", "kind": "blender", "status": "done",
            "layers": {"scan": {"characters": chars, "props": BALLS + [{"name": "Photo", "animated": True}]},
                       "descriptions": {"props": {"Ball_*": "matte violet dodgeballs"}}}}


# ---------------------------------------------------------------- slots
def test_slots_follow_the_mannequin_layers_and_their_picture_side():
    sl = shots.mannequin_slots(duo_record())
    assert [s["collection"] for s in sl["slots"]] == ["Mannequin", "Mannequin2"]          # numbered by collection
    assert [s["side"] for s in sl["slots"]] == ["left", "right"] and sl["markers"] and sl["scanned"]
    assert shots.slot_name(sl["slots"][1]) == "Character 2 (right, Mannequin2)"
    # no projected place: the floor places of a template from a video (camera along +Y), no markers
    rec = duo_record(screen=False)
    rec["people"] = {"chosen": [{"collection": "Mannequin", "place": [0.9, 0.7]},
                                {"collection": "Mannequin2", "place": [-0.9, -0.7]}]}
    sl = shots.mannequin_slots(rec)
    assert [s["side"] for s in sl["slots"]] == ["right", "left"] and not sl["markers"]
    # nothing measured: no side is guessed
    assert [s["side"] for s in shots.mannequin_slots(duo_record(screen=False))["slots"]] == [None, None]


def test_unscanned_video_and_three_figure_templates():
    sl = shots.mannequin_slots({"name": "old", "kind": "blender"})
    assert len(sl["slots"]) == 1 and sl["slots"][0]["collection"] == "Mannequin" and sl["scanned"] is False
    v = shots.mannequin_slots({"kind": "video"})
    assert len(v["slots"]) == 1 and v.get("video")
    three = {"layers": {"scan": {"characters": [
        {"collection": c, "screen": {"head": [x, 0.2], "feet": [x, 0.9]}}
        for c, x in (("Mannequin", 0.5), ("Mannequin2", 0.2), ("Mannequin10", 0.8))]}}}
    sl = shots.mannequin_slots(three)
    assert [s["collection"] for s in sl["slots"]] == ["Mannequin", "Mannequin2", "Mannequin10"]
    assert [s["side"] for s in sl["slots"]] == ["centre", "left", "right"]


def test_a_run_refusal_names_every_slot(monkeypatch):
    monkeypatch.setattr(shots, "library_record", lambda blend: duo_record())
    msg = runner._slots_refusal("x.blend", ["Mannequin", "Mannequin2"], 1)
    assert "Character 1 (left, Mannequin) = --character #1" in msg
    assert "Character 2 (right, Mannequin2) = --character #2" in msg and "1 given" in msg
    msg = runner._slots_refusal("x.blend", ["Mannequin", "Mannequin2", "Mannequin3"], 2)
    assert "3 characters" in msg and "Mannequin3) = --character #3" in msg and "2 given" in msg
    msg = runner._slots_refusal("x.blend", [f"Mannequin{i}" for i in ("", 2, 3, 4, 5)], 5)
    assert "at most 4" in msg


def test_prop_groups_count_whole_props_and_group_the_rest_by_name():
    g = shots.prop_groups(duo_record())
    assert [(x["pattern"], x["label"], x["count"], x["words"]) for x in g] == [
        ("Ball_*", "Balls", 3, "matte violet dodgeballs"), ("Photo", "Photo", 1, "")]
    assert len(g[0]["objects"]) == 6                                     # the seams belong to the group
    # an old scan without parents: "<prop>_<part>" names are parts
    old = {"layers": {"scan": {"props": [{"name": n} for n in ("Cup_1", "Cup_1_Lid", "Cup_2", "Box")]}}}
    assert [(x["pattern"], x["label"], x["count"]) for x in shots.prop_groups(old)] == [
        ("Cup_*", "Cups", 2), ("Box", "Box", 1)]
    assert shots.prop_groups({"layers": {}}) == []


def test_prop_words_are_checked():
    rec = duo_record()
    assert shots.check_prop_words(rec, {"Ball_*": "  red   beach balls ", "Photo": ""}) == {"Ball_*": "red beach balls"}
    assert shots.check_prop_words(rec, {"Ball_Catch": "a gold ball"}) == {"Ball_Catch": "a gold ball"}   # one object
    with pytest.raises(shots.ShotError, match="no prop matches 'Hoop'"):
        shots.check_prop_words(rec, {"Hoop": "a hoop"})
    with pytest.raises(shots.ShotError, match="longer than 300"):
        shots.check_prop_words(rec, {"Ball_*": "x" * 301})
    with pytest.raises(shots.ShotError, match="no props"):
        shots.check_prop_words({"layers": {"scan": {"props": []}}}, {"Ball_*": "balls"})


def test_the_runs_prop_words_replace_the_templates_in_the_clip_text():
    lay = {"descriptions": {"set": "a gym", "props": {"Ball_*": "matte violet dodgeballs", "Photo": "a framed photo"}}}
    m = {"render_mode": "layout", "prompt_format": "ing25", "sound_text": "", "layers": lay,
         "prompt": {"character": "on black: a fox", "style": "3D", "video": "the fox dances"},
         "props": {"Ball_*": "red and white beach balls"}}
    assert runner.props_sentence(m) == " In the scene: red and white beach balls; a framed photo."
    assert "red and white beach balls" in runner.video_prompt(m) and "violet" not in runner.video_prompt(m)
    assert runner.props_label(m["props"]) == "Balls -> red and white beach balls"
    assert lay["descriptions"]["props"]["Ball_*"] == "matte violet dodgeballs"   # the frozen words are not changed


# ---------------------------------------------------------------- the New video form -> API
def test_the_forms_slots_map_to_the_characters_list():
    f = wr._slot_characters({"blend": "b", "characters": [
        {"slot": 2, "collection": "Mannequin2", "character": "fox"},
        {"slot": 1, "collection": "Mannequin", "character": "masked"},
        {"slot": 3, "collection": "Mannequin3", "character": "kit"}], "spacing": 3})
    assert f["character"] == "masked" and "character2" not in f
    assert f["slot_characters"] == [{"name": "masked", "collection": "Mannequin"},
                                    {"name": "fox", "collection": "Mannequin2"},
                                    {"name": "kit", "collection": "Mannequin3"}]
    assert "spacing" not in f and "characters" not in f
    assert wr._group_args(f, None) == {"characters": f["slot_characters"]}
    one = wr._slot_characters({"characters": [{"slot": 1, "collection": "Mannequin", "character": "masked"}]})
    assert one["character"] == "masked" and "slot_characters" not in one and wr._group_args(one, None) == {}
    assert wr._slot_characters({"character": "old-form"}) == {"character": "old-form"}      # the old fields still work
    assert wr._group_args({"character": "a", "character2": "b", "spacing": "2"}, None) ==         {"character2": {"name": "b"}, "spacing": 2.0}
    with pytest.raises(ValueError, match="This template has 5 characters; videos support up to 4 for now"):
        wr._slot_characters({"characters": [{"slot": k, "character": f"c{k}"} for k in (1, 2, 3, 4, 5)]})
    with pytest.raises(ValueError, match="choose a character for Character 2"):
        wr._slot_characters({"characters": [{"slot": 1, "character": "a"}, {"slot": 2, "character": ""}]})
    with pytest.raises(ValueError, match="without gaps"):
        wr._slot_characters({"characters": [{"slot": 1, "character": "a"}, {"slot": 3, "character": "b"}]})
    twins = wr._slot_characters({"characters": [{"slot": 1, "character": "a"}, {"slot": 2, "character": "a"}]})
    assert [r["name"] for r in wr._group_args(twins, None)["characters"]] == ["a", "a"]   # twins: allowed
    with pytest.raises(ValueError, match="video template"):
        wr._group_args(f, "beach")
    assert wr._props_args({"props": {"Ball_*": " red balls ", "Photo": ""}}, None) == {"props": {"Ball_*": "red balls"}}
    with pytest.raises(ValueError, match="video template"):
        wr._props_args({"props": {"Ball_*": "red balls"}}, "beach")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(jobs.spawn, "detached", lambda *a, **k: (0, True))
    return tmp_path


def test_the_shots_list_has_slots_and_props(home):
    d = shots.shots_dir() / "duo"
    d.mkdir(parents=True)
    (d / "shot.blend").write_bytes(b"BLENDER")
    rec = {**duo_record(), "files": {"blend": str(d / "shot.blend")}}
    (d / "shot.json").write_text(json.dumps(rec), encoding="utf-8")
    old = shots.shots_dir() / "handmade"                                 # a shot.blend without a record: not scanned
    old.mkdir()
    (old / "shot.blend").write_bytes(b"BLENDER")

    async def go():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as c:
            listed = {s["name"]: s for s in (await (await c.get("/api/shots")).json())["data"]}
            cards = {t["name"]: t for t in (await (await c.get("/api/templates")).json())["data"]["shots"]}
            return listed, cards
    listed, cards = asyncio.run(go())
    duo = listed["duo"]
    assert [m["side"] for m in duo["mannequins"]] == ["left", "right"] and duo["markers"] and duo["slots_scanned"]
    assert duo["props"][0]["label"] == "Balls" and duo["props"][0]["count"] == 3
    assert len(listed["handmade"]["mannequins"]) == 1 and listed["handmade"]["slots_scanned"] is False
    assert len(cards["duo"]["mannequins"]) == 2 and cards["duo"]["props"][1]["pattern"] == "Photo"


# ---------------------------------------------------------------- CLI and MCP
def test_the_cli_prop_flag():
    assert cli.prop_args(["Ball_*=red and white beach balls", "Photo = a map"]) == {
        "Ball_*": "red and white beach balls", "Photo": "a map"}
    with pytest.raises(cli.CliInputError, match="PATTERN=WORDS"):
        cli.prop_args(["Ball_*"])
    with pytest.raises(cli.CliInputError, match="give the words"):
        cli.prop_args(["Ball_*="])
    assert cli.prop_args(["Ball_*="], allow_empty=True) == {"Ball_*": ""}


def test_cli_run_new_and_fork_pass_the_props(monkeypatch, tmp_path, capsys):
    seen = {}
    blend = tmp_path / "shot.blend"
    blend.write_bytes(b"x")
    monkeypatch.setattr(cli.character, "for_run", lambda name: ("sheet.png", "text"))
    monkeypatch.setattr(cli.runner, "new_run", lambda *a, **k: seen.update(new=k) or {"id": "r1"})
    monkeypatch.setattr(cli.runner, "fork_run", lambda *a, **k: seen.update(fork=k) or {"id": "r2", "forked_from": {}})
    monkeypatch.setattr(cli, "_start_detached", lambda rid, *a: {"ok": True})
    cli.main(["run", "new", "--blend", str(blend), "--character", "fox", "--video", "v",
              "--prop", "Ball_*=red beach balls", "--prop", "Photo=a map"])
    assert seen["new"]["props"] == {"Ball_*": "red beach balls", "Photo": "a map"}
    cli.main(["run", "fork", "r1", "--from-clip", "2", "--prop", "Ball_*="])
    assert seen["fork"]["props"] == {"Ball_*": ""}


def test_mcp_new_video_run_and_fork_pass_the_props(tmp_path, monkeypatch):
    seen = {}
    blend = tmp_path / "shot.blend"
    blend.write_bytes(b"x")
    monkeypatch.setattr(mcp_server.character, "for_run", lambda name: ("sheet.png", "text"))
    monkeypatch.setattr(mcp_server.runner, "new_run", lambda *a, **k: seen.update(new=k) or {"id": "r1"})
    monkeypatch.setattr(mcp_server.runner, "fork_run", lambda *a, **k: seen.update(fork=k) or
                        {"id": "r2", "forked_from": {}})
    monkeypatch.setattr(mcp_server.runner, "queue_label", lambda m: "label")
    monkeypatch.setattr(mcp_server.workqueue, "add", lambda *a: None)
    monkeypatch.setattr(cli, "_start_detached", lambda rid, *a: {"ok": True})
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    mcp_server.new_video_run(str(blend), "fox", "v", props={"Ball_*": "red beach balls"})
    assert seen["new"]["props"] == {"Ball_*": "red beach balls"}
    mcp_server.new_video_run(str(blend), "fox", "v")
    assert "props" not in seen["new"]                                    # None = the template's words
    mcp_server.fork_run("r1", 2, props={"Ball_*": "gold balls"})
    assert seen["fork"]["props"] == {"Ball_*": "gold balls"}


def test_proxy_kit_job_arguments_for_one_and_several_characters(tmp_path):
    """One kit keeps the classic --spec (a single-character run broke when it got --kits before proxy_kit.py knew it,
    2026-10-25); several kits go to ONE Blender job with --kits."""
    import json as _json
    one = runner.proxy_kit_args(tmp_path, [{"spec": {"mask": {}}, "rig": "MannequinRig", "collection": "Mannequin"}],
                                {"mask": {}}, tmp_path / "o.blend")
    assert one[:2] == ["--spec", str(tmp_path / "proxy.json")] and "--kits" not in one
    assert one[one.index("--rig") + 1] == "MannequinRig" and one[one.index("--collection") + 1] == "Mannequin"
    assert _json.loads((tmp_path / "proxy.json").read_text()) == {"mask": {}}
    todo = [{"spec": {"mask": {}}, "rig": "MannequinRig", "collection": "Mannequin"},
            {"spec": {"hair": {}}, "rig": "MannequinRig2", "collection": "Mannequin2"},
            {"spec": {"cape": {}}, "rig": "MannequinRig3", "collection": "Mannequin3"}]
    many = runner.proxy_kit_args(tmp_path, todo, todo[0]["spec"], tmp_path / "o.blend")
    assert many == ["--kits", str(tmp_path / "proxy_kits.json"), "--out", str(tmp_path / "o.blend")]
    assert [k["rig"] for k in _json.loads((tmp_path / "proxy_kits.json").read_text())] == \
        ["MannequinRig", "MannequinRig2", "MannequinRig3"]
    only2 = runner.proxy_kit_args(tmp_path, todo[1:2], None, tmp_path / "o.blend")   # slot 1 without a kit
    assert only2[0] == "--kits"
    import re
    src = (runner.ROOT / "bpy_jobs" / "proxy_kit.py").read_text(encoding="utf-8")
    assert re.search(r'add_argument\("--kits"', src) and re.search(r'add_argument\("--spec", default=None', src)


def test_cli_run_new_takes_one_character_per_slot(monkeypatch, tmp_path):
    seen = {}
    blend = tmp_path / "shot.blend"
    blend.write_bytes(b"x")
    monkeypatch.setattr(cli.character, "for_run", lambda name: seen.update(sheet_of=name) or ("sheet.png", "text"))
    monkeypatch.setattr(cli.runner, "new_run", lambda *a, **k: seen.update(new=k) or {"id": "r1"})
    monkeypatch.setattr(cli, "_start_detached", lambda rid, *a: {"ok": True})
    cli.main(["run", "new", "--blend", str(blend), "--character", "fox", "--character", "masked", "--character", "fox",
              "--video", "v"])
    assert seen["sheet_of"] == "fox" and seen["new"]["character_name"] == "fox"
    assert seen["new"]["characters"] == [{"name": "fox"}, {"name": "masked"}, {"name": "fox"}]
    cli.main(["run", "new", "--blend", str(blend), "--character", "fox", "--character2", "masked",
              "--character2-rig", "R2", "--video", "v"])                      # the old form = slot 2
    assert seen["new"]["characters"] == [{"name": "fox"}, {"name": "masked", "collection": None, "rig": "R2"}]
    cli.main(["run", "new", "--blend", str(blend), "--character", "fox", "--video", "v"])
    assert seen["new"].get("characters") is None
