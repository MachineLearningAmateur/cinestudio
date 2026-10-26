"""`shot check NAME` (cinestudio/shot_check.py): every rule on synthetic facts (the data bpy_jobs/check_shot.py writes),
the sha256 cache, the stored summary, the CLI / MCP / API and the export refusal. No Blender."""
import asyncio
import copy
import json
import math
import sys
from pathlib import Path

import pytest

from cinestudio import cli, mcp_server, shot_check as sc, shots

BONES = {"head": [0, 0, 1.55], "hips": [0, 0, 0.95], "shoulder.L": [0.19, 0, 1.42], "shoulder.R": [-0.19, 0, 1.42],
         "hand.L": [0.28, -0.03, 0.8], "hand.R": [-0.28, -0.03, 0.8], "foot.L": [0.11, 0.01, 0.09],
         "foot.R": [-0.11, 0.01, 0.09]}
JOINTS = {"head": [0.5, 0.2], "hips": [0.5, 0.5], "hand.L": [0.55, 0.55], "hand.R": [0.45, 0.55],
          "ankle.L": [0.52, 0.79], "ankle.R": [0.48, 0.79], "toe.L": [0.52, 0.81], "toe.R": [0.48, 0.81]}
CAM = {"loc": [0.0, -4.0, 1.45], "fwd": [0.0, 0.9913, -0.1314], "lens": 30.0, "name": "ShotCam"}


def facts(n=100, **kw):
    """A clean template: a figure standing still, facing a still front camera at 4 m, a grey set, nothing loose."""
    tl = [{"frame": 1 + i, "bones": copy.deepcopy(BONES), "camera": [0.0, -4.0, 1.45]} for i in range(n)]
    f = {"version": sc.FACTS_VERSION, "frames": [1, n], "fps": 24.0, "resolution": [1536, 896],
         "layers": {"characters": [{"collection": "Mannequin", "rig": "MannequinRig", "objects": 107,
                                    "screen": {"head": [0.5, 0.2], "feet": [0.5, 0.81]}}],
                    "props": [], "set": ["Floor"], "loose": [], "camera_moves": False, "frames": [1, n]},
         "mannequin_foreign": {}, "in_two_layers": [],
         "rigs": {"MannequinRig": {"collection": "Mannequin", "timeline": tl,
                                   "joints": {j: [list(p) for _ in range(n)] for j, p in JOINTS.items()},
                                   "scale_leaks": []}},
         "camera": {"frames": [dict(CAM) for _ in range(n)], "moves": False}, "props": [],
         "set_materials": [{"object": "Floor", "material": "Clay", "color": [0.3, 0.3, 0.3], "saturation": 0.0,
                            "value": 0.3, "textured": False}],
         "sounds": [{"name": "song", "file": "song.wav", "packed": True, "frame_start": 1, "mute": False}]}
    f.update(kw)
    return f


def rig(f):
    return f["rigs"]["MannequinRig"]


def by_id(checks, cid):
    return [c for c in checks if c["id"] == cid]


def bad(checks):
    return [(c["id"], c["severity"]) for c in checks if c["severity"] in ("error", "warning")]


def test_a_clean_template_passes_every_rule():
    checks = sc.rules(facts())
    assert bad(checks) == []
    ids = {c["id"] for c in checks}
    assert {"bone_scale", "shared_frames", "end_hold", "first_frame", "camera_speed", "camera_still", "layers",
            "figure_size", "set_neutral", "clip_texts", "sound", "joint_accel"} <= ids
    assert all(set(c) == {"id", "severity", "message", "frames", "fix", "data"} for c in checks)
    assert by_id(checks, "camera_still")[0]["message"].startswith("still camera: a backplate is possible")


def test_bone_scale_leaks_are_errors_and_float_drift_a_warning():
    f = facts()
    rig(f)["scale_leaks"] = [{"bone": "hand.L", "first": 20, "last": 90, "max": 1.35}]
    (c,) = by_id(sc.rules(f), "bone_scale")
    assert c["severity"] == "error" and "hand.L (1.35)" in c["message"] and c["frames"] == [[20, 90]] and c["fix"]
    rig(f)["scale_leaks"] = [{"bone": "hand.R", "first": 1, "last": 360, "max": 1.0016}]
    assert by_id(sc.rules(f), "bone_scale")[0]["severity"] == "warning"


def raise_hands(f, first, last):
    for fr in rig(f)["timeline"][first - 1:last]:
        for s in ("L", "R"):
            fr["bones"][f"hand.{s}"] = [0.3 if s == "L" else -0.3, -0.03, 1.8]


def test_shared_frames_flags_an_action_across_a_clip_join():
    assert [w["frames"] for w in sc.shared_windows(276)] == [[97, 121], [193, 217]]
    assert sc.shared_windows(121) == []
    f = facts(276, )
    raise_hands(f, 209, 222)                         # the old built-in's clap above the head
    w = by_id(sc.rules(f, {"clip_texts": {"1": "a", "2": "b", "3": "c"}, "clip_context": "x"}), "shared_frames")
    assert {c["severity"] for c in w} == {"warning"} and len(w) == 1 and "both hands" in w[0]["message"]
    assert w[0]["frames"] == [[209, 222], [193, 217]] and "between clips 2 and 3" in w[0]["message"]
    f = facts(276)
    raise_hands(f, 130, 160)                         # inside clip 2 only
    assert by_id(sc.rules(f), "shared_frames")[0]["severity"] == "ok"


def test_motion_events_find_kicks_jumps_turns_and_dashes():
    f = facts(120)
    tl = rig(f)["timeline"]
    for fr in tl[20:26]:
        fr["bones"]["foot.L"] = [0.4, -0.2, 0.55]    # a high side kick
    for fr in tl[50:55]:
        for s in ("L", "R"):
            fr["bones"][f"foot.{s}"][2] = 0.3        # both feet up: a jump
    for i, fr in enumerate(tl[80:90]):               # 90 deg turn in 10 frames
        a = math.radians(9 * (i + 1))
        fr["bones"]["shoulder.L"] = [0.19 * math.cos(a), 0.19 * math.sin(a), 1.42]
        fr["bones"]["shoulder.R"] = [-0.19 * math.cos(a), -0.19 * math.sin(a), 1.42]
    for fr in tl[90:]:
        fr["bones"]["shoulder.L"], fr["bones"]["shoulder.R"] = [0, 0.19, 1.42], [0, -0.19, 1.42]
    for i, fr in enumerate(tl[100:106]):             # the hips dash sideways 10 cm/frame
        fr["bones"]["hips"] = [0.1 * (i + 1), 0, 0.95]
    for fr in tl[106:]:
        fr["bones"]["hips"] = [0.6, 0, 0.95]
    what = [e["what"].split(" (")[0] for e in sc.motion_events(tl)]
    assert "kicks with its left foot" in what and "jumps" in " ".join(what)
    assert any(w.startswith("turns fast") for w in what) and "fast step" in what


def test_end_hold_first_frame_and_loops():
    f = facts()
    for i, fr in enumerate(rig(f)["timeline"][-6:]):
        fr["bones"]["hand.L"] = [0.28, -0.03, 0.8 + 0.05 * (i % 2)]      # still waving at the end
    (c,) = by_id(sc.rules(f), "end_hold")
    assert c["severity"] == "warning" and c["frames"] == [[89, 100]]
    assert by_id(sc.rules(f, {"plan": {"loop": True}}), "end_hold")[0]["severity"] == "info"
    f = facts()
    for fr in rig(f)["timeline"]:                    # turned to profile: the camera sees its right side
        fr["bones"]["shoulder.L"], fr["bones"]["shoulder.R"] = [0, 0.19, 1.42], [0, -0.19, 1.42]
    (c,) = by_id(sc.rules(f), "first_frame")
    assert c["severity"] == "warning" and "90 deg" in c["message"]
    f = facts()
    for i, fr in enumerate(rig(f)["timeline"][:4]):
        fr["bones"]["hand.R"] = [-0.28, -0.03, 0.8 + 0.1 * i]           # moving from frame 1
    assert "moving" in by_id(sc.rules(f), "first_frame")[0]["message"]


def orbit(f, deg_per_frame):
    for i, c in enumerate(f["camera"]["frames"]):
        a = math.radians(-90 + deg_per_frame * i)
        c["loc"] = [4 * math.cos(a), 4 * math.sin(a), 1.45]
        c["fwd"] = [-math.cos(a), -math.sin(a), 0.0]
    f["camera"]["moves"] = True


def test_camera_speed_and_still():
    f = facts()
    orbit(f, 1.22)
    cs = sc.rules(f)
    (c,) = by_id(cs, "camera_speed")
    assert c["severity"] == "error" and "1.22" in c["message"]
    assert by_id(cs, "camera_still")[0]["message"].startswith("moving camera: use a panorama")
    f = facts()
    orbit(f, 0.8)
    assert by_id(sc.rules(f), "camera_speed")[0]["severity"] == "warning"
    f = facts()
    orbit(f, 0.5)
    assert by_id(sc.rules(f), "camera_speed")[0]["severity"] == "ok"
    f = facts()
    f["camera"]["frames"] = []
    assert by_id(sc.rules(f), "camera_speed")[0]["severity"] == "error"


def test_layers_rules():
    f = facts()
    f["mannequin_foreign"] = {"Mannequin": ["Ball"]}
    f["layers"]["loose"] = ["Cube"]
    f["layers"]["set"] = []
    cs = by_id(sc.rules(f), "layers")
    assert ("error" in {c["severity"] for c in cs} and any("Ball" in c["message"] for c in cs if c["severity"] == "error"))
    assert any("no layer: Cube" in c["message"] and c["severity"] == "warning" for c in cs)
    assert any("no Set" in c["message"] for c in cs)
    f = facts()
    f["layers"]["characters"] = []
    f["rigs"] = {}
    assert by_id(sc.rules(f), "layers")[0]["severity"] == "error"


def test_figure_size_and_the_body_leaving_the_frame():
    f = facts()
    f["layers"]["characters"][0]["screen"] = {"head": [0.5, 0.45], "feet": [0.5, 0.6]}
    (c,) = [x for x in by_id(sc.rules(f), "figure_size") if x["severity"] != "ok"]
    assert c["severity"] == "warning" and "15%" in c["message"]
    f = facts()
    for p in rig(f)["joints"]["hand.L"][40:45]:
        p[0] = 1.2
    (c,) = [x for x in by_id(sc.rules(f), "figure_size") if x["severity"] != "ok"]
    assert "leaves the frame" in c["message"] and c["frames"] == [[41, 45]]


def test_prop_size_set_colour_sound_and_parts():
    f = facts()
    f["props"] = [{"name": "Pea", "parent": None, "part": False, "boxes": [[0.5, 0.5, 0.505, 0.505]] * 100},
                  {"name": "Ball", "parent": None, "part": False, "boxes": [[0.4, 0.4, 0.45, 0.48]] * 100},
                  {"name": "Ball_Seam", "parent": "Ball", "part": True, "boxes": [[0.4, 0.4, 0.401, 0.401]] * 100},
                  {"name": "Parked", "parent": None, "part": False, "boxes": [None] * 100}]
    cs = by_id(sc.rules(f), "prop_size")
    sev = {c["message"].split()[0]: c["severity"] for c in cs}
    assert sev["Pea"] == "warning" and sev["Ball:"] == "ok" and sev["Parked"] == "info" and "Ball_Seam" not in sev
    f["set_materials"].append({"object": "Wall", "material": "Red", "color": [0.8, 0.05, 0.05], "saturation": 0.94,
                               "value": 0.8, "textured": False})
    (c,) = by_id(sc.rules(f), "set_neutral")
    assert c["severity"] == "info" and "Wall" in c["message"] and "neutral grey" in c["message"]
    assert "packed" in by_id(sc.rules(f), "sound")[0]["message"]
    f["sounds"] = []
    assert by_id(sc.rules(f), "sound")[0]["message"].startswith("no sound")


def test_clip_texts_rule_reads_the_record():
    f = facts(276)
    cs = by_id(sc.rules(f, {}), "clip_texts")
    assert [c["severity"] for c in cs] == ["warning", "warning"] and "clips 1, 2, 3" in cs[0]["message"]
    cs = by_id(sc.rules(f, {"clip_texts": {"1": "a", "3": "c"}, "clip_context": "{character} faces us"}), "clip_texts")
    assert len(cs) == 1 and "clip 2" in cs[0]["message"]
    full = {"clip_texts": {"1": "a", "2": "b", "3": "c"}, "clip_context": "{character} faces us"}
    assert by_id(sc.rules(f, full), "clip_texts")[0]["severity"] == "ok"
    assert by_id(sc.rules(facts(100), {}), "clip_texts")[0]["severity"] == "ok"        # one clip


def test_joint_pops():
    f = facts()
    rig(f)["joints"]["hand.R"][50] = [0.6, 0.55]     # one frame 115 px off: a pop
    (c,) = by_id(sc.rules(f), "joint_accel")
    assert c["severity"] == "warning" and "hand.R" in c["message"] and [50, 52] in c["frames"]
    assert sc.joint_accels(rig(facts()))["head"][1] == 0.0


def test_a_broken_rule_does_not_hide_the_others(monkeypatch):
    def boom(f, r):
        raise KeyError("x")
    monkeypatch.setattr(sc, "RULES", (boom,) + sc.RULES[1:])
    cs = sc.rules(facts())
    assert any(c["id"] == "check_incomplete" and c["severity"] == "error" and c["data"]["rule"] == "boom"
               and "could not run" in c["message"] for c in cs) and by_id(cs, "end_hold")     # an error, not a pass


# ---------------------------------------------------------------- the cache, the record, the interfaces
@pytest.fixture
def tpl(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    d = shots.shots_dir() / "gym"
    d.mkdir(parents=True)
    (d / "shot.blend").write_bytes(b"BLENDER-1")
    (d / "shot.json").write_text(json.dumps({"name": "gym", "kind": "blender", "status": "done",
                                             "files": {"blend": str(d / "shot.blend")}}), encoding="utf-8")
    calls = []

    def measure(blend):
        calls.append(str(blend))
        f = facts(100)
        rig(f)["scale_leaks"] = [{"bone": "hand.L", "first": 1, "last": 100, "max": 1.3}]
        return f
    monkeypatch.setattr(sc, "measure", measure)
    return calls


def test_the_check_is_cached_by_sha_and_stored(tpl):
    r = sc.check("gym")
    assert r["ok"] and r["template"] == "gym" and r["errors"] == 1 and not r["cached"] and len(tpl) == 1
    again = sc.check("gym")
    assert again["cached"] and len(tpl) == 1 and again["errors"] == 1          # instant: no Blender
    stored = shots.load("gym")["check"]
    assert stored["errors"] == 1 and stored["sha"] == r["sha"] and set(stored) == {"errors", "warnings", "at", "sha", "fp"}
    assert sc.chip(shots.load("gym"))["stale"] is False
    blend = shots.shot_dir("gym") / "shot.blend"
    blend.write_bytes(b"BLENDER-2 changed")                                    # the .blend changed: stale, then a new read
    chip = sc.chip(shots.load("gym"))
    assert chip["stale"] is True and chip["errors"] == 1
    assert not sc.check("gym")["cached"] and len(tpl) == 2
    sc.check("gym", fresh=True)
    assert len(tpl) == 3


def test_check_refuses_video_and_unfinished_templates(tpl):
    d = shots.shots_dir() / "vid"
    d.mkdir()
    (d / "shot.json").write_text(json.dumps({"name": "vid", "kind": "video", "status": "done", "files": {}}))
    with pytest.raises(sc.CheckError, match="video template"):
        sc.check("vid")
    rec = shots.load("gym")
    rec["status"] = "running"
    (shots.shot_dir("gym") / "shot.json").write_text(json.dumps(rec))
    with pytest.raises(sc.CheckError, match="not done"):
        sc.check("gym")


def test_after_build_never_fails_a_build(tpl, monkeypatch):
    events = []
    rec = shots.load("gym")
    s = sc.after_build(shots.shot_dir("gym") / "shot.blend", rec, events.append)
    assert s["errors"] == 1 and any(e.get("id") == "bone_scale" for e in events)
    monkeypatch.setattr(sc, "measure", lambda b: (_ for _ in ()).throw(RuntimeError("no blender")))
    (shots.shot_dir("gym") / "shot.blend").write_bytes(b"other")
    events.clear()
    assert sc.after_build(shots.shot_dir("gym") / "shot.blend", rec, events.append) is None
    assert "could not run" in events[0]["text"]


def test_cli_and_mcp(tpl, capsys):
    assert cli.main(["shot", "check", "gym"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["ok"] and out["errors"] == 1 and out["template"] == "gym" and "facts" not in out
    assert cli.main(["shot", "check", "gym", "--json"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["facts"]["frames"] == [1, 100]
    assert cli.main(["shot", "check", "nope"]) != 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["ok"] is False and out["error"]
    r = mcp_server.check_template("gym")
    assert r["ok"] and r["errors"] == 1 and r["checks"][0]["severity"] == "error"


def test_the_api_checks_and_the_cards_show_the_chip(tpl):
    from aiohttp.test_utils import TestClient, TestServer

    from cinestudio import webapp

    async def go():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as c:
            card = next(t for t in (await (await c.get("/api/templates")).json())["data"]["shots"] if t["name"] == "gym")
            assert card["check"] is None                                       # never checked
            d = (await (await c.post("/api/templates/gym/check", json={})).json())["data"]
            assert d["errors"] == 1 and d["checks"][0]["id"] == "bone_scale"
            card = next(t for t in (await (await c.get("/api/templates")).json())["data"]["shots"] if t["name"] == "gym")
            assert card["check"]["errors"] == 1 and card["check"]["stale"] is False
            s = next(x for x in (await (await c.get("/api/shots")).json())["data"] if x["name"] == "gym")
            assert s["check"]["errors"] == 1
            r = await c.post("/api/templates/nope/check", json={})
            assert r.status == 400
    asyncio.run(go())


def test_export_refuses_on_errors_unless_forced(tpl, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "setup"))
    import export_template
    with pytest.raises(SystemExit, match="1 template check error"):
        export_template.check_before_export("gym")
    assert export_template.check_before_export("gym", force=True)["errors"] == 1
    monkeypatch.setattr(sc, "measure", lambda b: (_ for _ in ()).throw(RuntimeError("no blender")))
    (shots.shot_dir("gym") / "shot.blend").write_bytes(b"other")
    with pytest.raises(SystemExit, match="could not run"):
        export_template.check_before_export("gym")
    assert export_template.check_before_export("gym", force=True) == {}


def test_builds_store_the_summary(tpl):
    rec = shots.load("gym")
    shots._store_check(shots.shot_dir("gym") / "shot.blend", rec)
    assert rec["check"]["errors"] == 1 and rec["check"]["sha"] == sc.blend_sha(shots.shot_dir("gym") / "shot.blend")


# ---------------------------------------------------------------- Codex review 2026-10-25 (round 1)
def test_two_checks_at_once_read_the_blend_once(tmp_path, monkeypatch):
    """P3: the cache's temp name was per process (two API threads collided) and both threads ran Blender."""
    import threading
    import time as _time
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    blend = tmp_path / "shot.blend"
    blend.write_bytes(b"BLENDER-concurrent")
    calls, results, errors = [], [], []

    def slow(b):
        calls.append(1)
        _time.sleep(0.3)
        return facts(100)

    def go():
        try:
            results.append(sc.facts_for(blend, measurer=slow))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    ts = [threading.Thread(target=go) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors and len(calls) == 1 and len(results) == 3
    assert sorted(r[2] for r in results) == [False, True, True]              # one read, two cache hits
    assert not list(sc.cache_dir().glob("*.tmp"))


def test_missing_measurements_make_the_check_incomplete():
    """Verified (item 8): a rig without the tracked bones has an empty timeline; the joins, the end hold, the first
    frame and the feet were skipped silently and the check passed."""
    f = facts(276)
    raise_hands(f, 209, 222)                                   # would be a join warning if it could be measured
    rig(f)["timeline"] = []
    cs = sc.rules(f, {"clip_texts": {"1": "a", "2": "b", "3": "c"}, "clip_context": "x"})
    inc = by_id(cs, "check_incomplete")
    assert inc and inc[0]["severity"] == "error" and "tracked bones" in inc[0]["message"]
    f = facts()
    f["layers"]["characters"].append({"collection": "Mannequin2", "rig": "MannequinRig2", "screen": None})
    assert any(c["data"].get("rig") == "MannequinRig2" for c in by_id(sc.rules(f), "check_incomplete"))
    assert by_id(sc.rules(facts()), "check_incomplete") == []                  # a clean template: complete


def test_the_export_enforces_the_shipped_blends_check(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "setup"))
    import export_template
    err = {"checks": [{"id": "check_incomplete", "severity": "error", "message": "m", "fix": "", "frames": [],
                       "data": {}}], "errors": 1, "warnings": 0, "at": 1.0, "sha": "s"}
    monkeypatch.setattr(export_template.shot_check, "check_blend", lambda blend, rec, name: err)
    with pytest.raises(SystemExit, match="exported .blend has 1 template check error"):
        export_template.check_shipped(tmp_path / "shot.blend", {}, "gym")
    assert export_template.check_shipped(tmp_path / "shot.blend", {}, "gym", force=True) is err
    monkeypatch.setattr(export_template.shot_check, "check_blend",
                        lambda blend, rec, name: (_ for _ in ()).throw(RuntimeError("no blender")))
    with pytest.raises(SystemExit, match="could not run"):
        export_template.check_shipped(tmp_path / "shot.blend", {}, "gym")
    assert export_template.check_shipped(tmp_path / "shot.blend", {}, "gym", force=True) is None


def test_a_pose_held_across_a_join_is_not_an_action():
    """Verified (item 9): a raised arm HELD still across the shared frames was flagged like a move."""
    texts = {"clip_texts": {"1": "a", "2": "b", "3": "c"}, "clip_context": "x"}
    f = facts(276)
    raise_hands(f, 80, 140)                                    # raised before 97, still through 121, lowered after
    w = by_id(sc.rules(f, texts), "shared_frames")
    assert not [c for c in w if c["severity"] == "warning"]
    assert any(c["severity"] == "info" and c["data"].get("held") for c in w)
    assert any(c["severity"] == "ok" for c in w)
    f = facts(276)
    raise_hands(f, 105, 140)                                   # the raise itself inside the shared frames
    assert [c["severity"] for c in by_id(sc.rules(f, texts), "shared_frames")] == ["warning"]
    f = facts(276)
    raise_hands(f, 80, 140)
    for i, fr in enumerate(rig(f)["timeline"][99:119]):        # waving inside the shared frames: a move
        fr["bones"]["hand.L"] = [0.3 + (0.1 if i % 2 else -0.1), -0.03, 1.8]
    assert "warning" in [c["severity"] for c in by_id(sc.rules(f, texts), "shared_frames")]


def test_screen_sizes_use_the_runs_preview_size_for_portrait_shots():
    """Verified (item 10): pixel thresholds assumed 768x448; a portrait run previews at 448x768."""
    assert sc.preview_size(facts()) == (768, 448)
    portrait = facts(resolution=[896, 1536])
    assert sc.preview_size(portrait) == (448, 768)
    prop = {"name": "Card", "part": False, "boxes": [[0.5, 0.5, 0.525, 0.51]] * 100}   # wide and flat on screen
    land = by_id(sc.rules(facts(props=[prop])), "prop_size")[0]
    port = by_id(sc.rules(facts(props=[prop], resolution=[896, 1536])), "prop_size")[0]
    assert land["severity"] == "ok" and land["data"]["median_px"] == 19.2
    assert port["severity"] == "warning" and port["data"]["median_px"] == 11.2


def test_a_stored_check_is_stale_when_the_records_words_change(tpl):
    """Verified (item 11): without cached facts the card showed the stored summary as fresh after the clip texts or
    the loop changed."""
    sc.check("gym")
    rec = shots.load("gym")
    for p in sc.cache_dir().glob("*.json"):                    # no cached facts: the stored summary is shown
        p.unlink()
    assert sc.chip(rec)["stale"] is False
    rec["clip_texts"] = {"1": "{character} waves."}
    assert sc.chip(rec)["stale"] is True
    old = {k: v for k, v in rec["check"].items() if k != "fp"}            # a summary from before the fingerprint
    assert sc.chip({**rec, "check": old})["stale"] is False
