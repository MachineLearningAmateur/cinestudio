"""Templates and runs with ANY number of characters (2026-10-25, user: "if a video only has 1 character there should
only be one character option, if it has three then three options"): the characters list and the old manifests, the N
bodies and kits in the run's shot, the group sheet (sizes, the prop column), the crop-and-stitch planner for N figures
(with the overlap fallback), the clip text, the run details and a fork. No Blender, no GPU."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cinestudio import character, charsheet, runner, shots

COLOURS = {"masked": (30, 160, 30), "fox": (160, 30, 30), "kit": (30, 30, 160)}
HEIGHTS = {"masked": 1.0, "fox": 0.72, "kit": 0.9}
DESCS = {"masked": "a slim masked figure with a smooth white oval mask that covers the whole face; a black bodysuit.",
         "fox": "a slim young wanderer whose face is hidden behind a white kitsune mask: two tall ears, a teal cape.",
         "kit": "A small knight in round silver armour with a red plume. Brown boots."}


def figure(h: int, w: int, colour) -> np.ndarray:
    img = np.zeros((h + 40, w + 40, 3), np.uint8)
    cv2.rectangle(img, (20, 20), (20 + w, 20 + h), colour, -1)
    return img


# ---------------------------------------------------------------- the characters list and old manifests
def test_run_characters_reads_the_list_and_old_manifests():
    assert runner.run_characters({}) == []
    assert runner.run_characters({"character_name": "a"}) == [
        {"name": "a", "collection": "Mannequin", "rig": "MannequinRig"}]
    old = {"character_name": "a", "character_collection": "Mannequin", "rig": "MannequinRig",
           "character2": {"name": "b", "collection": "Mannequin2", "rig": "MannequinRig2"},
           "views": {"front": "inputs/views/front.png"}, "views2": {"front": "inputs/views2/front.png"},
           "prompt": {"duo": {"left": "a one.", "right": "b two.", "names": ["a", "b"], "heights": [1.0, 0.8]}}}
    assert [r["name"] for r in runner.run_characters(old)] == ["a", "b"]
    assert runner.run_characters(old)[1]["rig"] == "MannequinRig2"
    assert runner.run_views(old) == [{"front": "inputs/views/front.png"}, {"front": "inputs/views2/front.png"}]
    g = runner.run_group(old)
    assert g["order"] == [0, 1] and g["views"] == list(runner.DUO_VIEWS) and g["figures"][1]["height"] == 0.8
    new = {"characters": [{"name": "a"}, {"name": "b", "collection": "Mannequin2"}, {"name": "a"}]}
    assert runner.run_characters(new) == [{"name": "a", "collection": "Mannequin", "rig": "MannequinRig"},
                                          {"name": "b", "collection": "Mannequin2", "rig": "MannequinRig2"},
                                          {"name": "a", "collection": "Mannequin3", "rig": "MannequinRig3"}]
    assert runner.slot_names(3) == ("Mannequin4", "MannequinRig4")
    assert runner.queue_label({**new, "prompt": {"video": "they dance"}}).startswith("a + b · ")


def test_the_characters_option_and_its_old_form():
    o = {"characters": ["a", {"name": "b", "rig": "R2"}]}
    rows = runner._characters_option(o)
    assert o["character_name"] == "a" and "characters" not in o
    assert rows == [{"name": "a", "collection": None, "rig": None}, {"name": "b", "collection": None, "rig": "R2"}]
    o = {"character_name": "a", "character2": {"name": "b"}}
    assert [r["name"] for r in runner._characters_option(o)] == ["a", "b"] and "character2" not in o
    with pytest.raises(runner.RunError, match="not with character2"):
        runner._characters_option({"characters": ["a", "b"], "character2": {"name": "c"}})
    with pytest.raises(runner.RunError, match="first of the characters"):
        runner._characters_option({"character_name": "x", "characters": ["a", "b"]})
    with pytest.raises(runner.RunError, match="needs a character"):
        runner._characters_option({"characters": ["a", " "]})


# ---------------------------------------------------------------- the clip text
def _group(order=(0, 1, 2), views=("front", "back")):
    figs = [{"name": n, "description": DESCS[n].rstrip("."), "marks": runner.target_marks(DESCS[n]),
             "height": HEIGHTS[n]} for n in ("masked", "fox", "kit")]
    return {"figures": figs, "order": list(order), "views": list(views)}


def test_two_characters_read_exactly_like_the_old_duo_text():
    duo = {"left": "a masked figure in a plum jacket.", "right": "A wanderer in a fox mask and a teal cape.",
           "names": ["masked3d_rest", "kitsune"]}
    old = ("**Panels 1-3 (Character 1):** the first character in full body (front, right side, back) on black: "
           "a masked figure in a plum jacket. **Panels 4-6 (Character 2):** the second character in full body (front, "
           "right side, back) on black: A wanderer in a fox mask and a teal cape.")
    assert runner.duo_sheet_text(duo) == old
    m = {"prompt": {"character": old, "video": "Both dance.", "style": "", "duo": duo}, "prompt_format": "ing25",
         "sound_text": ""}
    assert ("Two characters from the reference sheet side by side: on the left, the first character, a masked figure "
            "in a plum jacket; on the right, the second character, a wanderer in a fox mask and a teal cape. "
            "Both dance.") in runner.video_prompt(m)


def test_three_characters_by_their_place_and_key_marks():
    g = _group(order=(1, 0, 2))                         # the fox stands on the left in the picture
    sheet = runner.group_sheet_text(g)
    assert sheet.startswith("**Panels 1-2 (Character 1):** the first character in full body (front, back) on black: "
                            "a slim young wanderer")
    assert "**Panels 3-4 (Character 2):** the second character in full body (front, back) on black: a slim masked" in sheet
    assert "**Panels 5-6 (Character 3):**" in sheet and "red plume" in sheet
    m = {"prompt": {"character": sheet, "video": "They dance.", "style": "3D animation", "group": g},
         "prompt_format": "ing25", "sound_text": ""}
    text = runner.video_prompt(m)
    target = text.split("### Target Description ", 1)[1]
    assert ("Three characters from the reference sheet side by side: on the left, the first character, a slim young "
            "wanderer whose face is hidden behind a white kitsune mask; in the middle, the second character, a slim "
            "masked figure with a smooth white oval mask that covers the whole face; on the right, the third "
            "character, a small knight in round silver armour with a red plume. They dance.") in target
    assert "teal cape" not in target                    # only the key marks in the target
    four = {"figures": g["figures"] + [g["figures"][0]], "order": [0, 1, 2, 3], "views": ["front"]}
    t4 = runner.group_sheet_text(four)
    assert "**Panel 4 (Character 4):** the fourth character in full body, from the front, on black:" in t4
    tg = runner._group_target(four)
    assert "second from the left" in tg and "second from the right" in tg and tg.startswith("Four characters")


def test_target_marks_and_views_per_count():
    assert runner.target_marks("a fox: tall ears; a cape") == "a fox"
    masked = ("a slim masked figure with a smooth white oval mask that covers the whole face, two black almond-shaped "
              "eye holes angled upward like snake eyes and no mouth opening; short hair")
    assert runner.target_marks(masked) == "a slim masked figure with a smooth white oval mask that covers the whole face"
    long = " ".join(["word"] * 40) + " the end"
    assert len(runner.target_marks(long).split()) == runner.TARGET_MARK_WORDS
    assert runner.target_marks("a hero with a sword and the") == "a hero with a sword and the"   # short: as it is
    assert runner.group_views(2) == runner.DUO_VIEWS and runner.group_views(3) == ("front", "back")
    assert runner.group_views(4) == ("front",)
    with pytest.raises(runner.RunError):
        runner.group_views(5)
    assert runner.picture_order([0.8, 0.2, 0.5], 3) == [1, 2, 0]
    assert runner.picture_order([0.8, None, 0.5], 3) == [0, 1, 2] and runner.picture_order(None, 2) == [0, 1]


# ---------------------------------------------------------------- crop and stitch for N figures
def _mask(boxes, size=(448, 768)):
    m = np.zeros(size, bool)
    for x0, y0, x1, y1 in boxes:
        m[y0:y1, x0:x1] = True
    return m


def test_three_figures_that_stand_apart_get_one_crop_each():
    m = _mask([(80, 100, 160, 400), (330, 140, 400, 400), (560, 90, 650, 400)])
    plan = runner.plan_figure_crops(m, 3, xs=[365, 120, 600])      # slot 1 stands in the middle
    assert [c["slots"] for c in plan] == [[1], [0], [2]] and not any(c["shared"] for c in plan)
    for c, (a, b) in zip(plan, ((80, 160), (330, 400), (560, 650))):
        x0, y0, x1, y1 = c["box"]
        assert x0 < a and x1 > b                                     # each box holds its figure ...
    assert plan[0]["box"][2] <= plan[1]["box"][0] and plan[1]["box"][2] <= plan[2]["box"][0]   # ... and no other
    plain = runner.plan_figure_crops(m, 3)                           # no screen places: slot order left to right
    assert [c["slots"] for c in plain] == [[0], [1], [2]]


def test_touching_figures_share_one_crop_the_fallback():
    m = _mask([(80, 100, 160, 400), (160, 120, 230, 400), (560, 90, 650, 400)])   # 1 and 2 touch
    plan = runner.plan_figure_crops(m, 3, xs=[120, 195, 600])
    assert [c["slots"] for c in plan] == [[0, 1], [2]] and [c["shared"] for c in plan] == [True, False]
    blind = runner.plan_figure_crops(m, 3)                           # without places: the widest run takes two
    assert [c["slots"] for c in blind] == [[0, 1], [2]]
    stray = _mask([(80, 100, 160, 400), (175, 300, 180, 305), (560, 90, 650, 400)])   # a loose part of figure 1
    plan = runner.plan_figure_crops(stray, 2, xs=[120, 600])
    assert [c["slots"] for c in plan] == [[0], [1]] and plan[0]["box"][2] > 180
    with pytest.raises(runner.RunError, match="no character"):
        runner.plan_figure_crops(np.zeros((10, 10), bool), 2)


def test_the_group_look_edits_each_crop_and_a_shared_crop_in_turn(tmp_path):
    frame = np.full((448, 768, 3), 120, np.uint8)
    cv2.imwrite(str(tmp_path / "f.png"), frame)
    m = _mask([(80, 100, 160, 400), (160, 120, 230, 400), (560, 90, 650, 400)])
    cv2.imwrite(str(tmp_path / "m.png"), m.astype(np.uint8) * 255)
    calls = []

    def figure_look(step, name, src, slot, out, place=None):
        calls.append((name, slot, place, Path(src).name))
        img = cv2.imread(str(src))
        img[:] = (0, 0, 255) if slot == 2 else (255, 0, 0)
        cv2.imwrite(str(out), img)
    events = []
    run = SimpleNamespace(m={"characters": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
                             "mannequins": {"screen_x": [120 / 768, 195 / 768, 600 / 768]}},
                          work=tmp_path, _figure_look=figure_look, event=lambda kind, **k: events.append((kind, k)))
    run._figure_xs = lambda w: runner.Run._figure_xs(run, w)
    step = {}
    runner.Run._group_look(run, step, tmp_path / "f.png", tmp_path / "m.png", tmp_path / "look.png")
    assert [(c[0], c[1], c[2]) for c in calls] == [("look", 0, "on the left"), ("look_fig2", 1, "on the right"),
                                                    ("look_fig3", 2, None)]
    assert calls[1][3] == "look_g1_f1.png"                           # the second edit works on the first's result
    assert step["look_fallback"] == "shared crop" and events[-1][0] == "group_look"
    look = cv2.imread(str(tmp_path / "look.png"))
    assert np.abs(look[250, 600].astype(int) - (0, 0, 255)).max() <= 2       # pasted back on the figures
    assert np.abs(look[250, 150].astype(int) - (255, 0, 0)).max() <= 2
    assert tuple(look[20, 400]) == (120, 120, 120)                   # the rest of the frame stays
    assert np.abs(look[45, 600].astype(int) - 120).max() <= 12       # inside the crop box but away from the figure:
    #                                                                  the original frame (a crop's wall became sky)


# ---------------------------------------------------------------- the group sheet
def test_the_group_sheet_for_three_and_four_with_a_prop_column():
    def groups(n, views):
        return [[charsheet.cutout(figure(900, 260, c)) for _ in range(views)]
                for c in list(COLOURS.values())[:n] + [(120, 120, 30)] * (n - 3)]
    ball = charsheet.cutout(figure(300, 300, (200, 60, 150)))
    for n in (3, 4):
        k = len(runner.group_views(n))
        plain, lay = {}, {}
        a = charsheet.assemble_groups(groups(n, k), ratios=[1.0, 0.72, 0.9, 1.0][:n], layout=plain)
        b = charsheet.assemble_groups(groups(n, k), ratios=[1.0, 0.72, 0.9, 1.0][:n], props=[ball], layout=lay)
        assert a.shape == b.shape == (896, 1536, 3)
        assert plain["body_scale"] >= 0.9                            # six panels or fewer: about full size
        assert lay["prop_places"] == ["bottom-right"] and lay["body_scale"] >= 0.75   # the props gave way first
        x0 = lay["prop_boxes"][0][0]
        for c in list(COLOURS.values())[:3]:                         # every group left of the prop column
            cols = np.where((np.abs(b.astype(int) - c).max(axis=2) < 10).any(axis=0))[0]
            assert cols.size and cols.max() < x0
    # nine panels (three groups x three views) would have shrunk the bodies: the reason for GROUP_VIEWS
    crowded = {}
    charsheet.assemble_groups(groups(3, 3), layout=crowded)
    assert crowded["body_scale"] < 1.0


# ---------------------------------------------------------------- a run with three characters
@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    blend = tmp_path / "shot.blend"
    blend.write_bytes(b"BLENDER-fake")
    recs = {}
    for name, col in COLOURS.items():
        d = tmp_path / name
        d.mkdir()
        files = {}
        for v in ("front", "right", "back", "left", "face"):
            cv2.imwrite(str(d / f"{v}.png"), figure(900 if v != "face" else 400, 260, col))
            files[v] = str(d / f"{v}.png")
        cv2.imwrite(str(d / "sheet.png"), np.zeros((896, 1536, 3), np.uint8))
        files["sheet"] = str(d / "sheet.png")
        recs[name] = {"name": name, "status": "ready", "files": files, "description": DESCS[name],
                      "proxy": {"mask": {"shape": "round"}} if name != "kit" else None,
                      "mannequin": {"scale": 0.8} if name == "fox" else {}}
    monkeypatch.setattr(character, "load", lambda name: json.loads(json.dumps(recs[name])))
    monkeypatch.setattr(character, "require_video_ready", lambda name: {})
    monkeypatch.setattr(character, "ensure_height", lambda rec: rec)
    monkeypatch.setattr(character, "height_ratio", lambda rec: HEIGHTS[rec["name"]])
    three = {"name": "trio", "kind": "blender", "status": "done",
             "layers": {"scan": {"characters": [{"collection": c, "rig": r} for c, r in
                                                (("Mannequin", "MannequinRig"), ("Mannequin2", "MannequinRig2"),
                                                 ("Mannequin3", "MannequinRig3"))],
                                 "props": [{"name": "Ball_1"}]},
                        "descriptions": {"set": "", "props": {"Ball_*": "red balls"}}}}
    monkeypatch.setattr(shots, "library_record", lambda b: three)

    def fake_snapshot(src, dst, report, allow_external=False):
        dst.write_bytes(src.read_bytes())
        r = {"ok": True, "packed": [], "missing": [], "unsupported": []}
        report.write_text(json.dumps(r))
        return r
    monkeypatch.setattr(runner, "snapshot_blend", fake_snapshot)
    monkeypatch.setattr(runner.blender, "check", lambda: {"ok": True, "version": "5.2.0"})
    calls = []

    def fake_job(script, blend_in, args, log=None):
        args = list(map(str, args))
        calls.append((script, args))
        Path(args[args.index("--out") + 1]).write_bytes(Path(blend_in).read_bytes() + b"+" + script.encode())
        if script == "duplicate_mannequin.py":
            n = int(args[args.index("--count") + 1])
            Path(log).write_text("DUPLICATED " + json.dumps({"made": 0, "unused": [], "bodies": [None] * n,
                                                             "screen_x": [0.5, 0.2, 0.8][:n], "rigs": []}) + "\n")
    monkeypatch.setattr(runner.blender, "run_job", fake_job)
    ball = tmp_path / "ball.png"
    img = np.full((256, 256, 3), 4, np.uint8)
    cv2.circle(img, (128, 128), 90, (200, 60, 150), -1)
    cv2.imwrite(str(ball), img)
    return {"blend": blend, "sheet": tmp_path / "masked" / "sheet.png", "calls": calls, "ball": ball}


def new3(env, **kw):
    sheet, text = env["sheet"], "Left panel: ... on black: " + DESCS["masked"]
    return runner.new_run(env["blend"], sheet, text, "They dance.", characters=["masked", "fox", "kit"], **kw)


def test_a_three_character_run_freezes_bodies_kits_views_sheet_and_text(env):
    m = new3(env)
    d = runner.run_dir(m["id"])
    calls = env["calls"]
    assert [c[0] for c in calls] == ["duplicate_mannequin.py", "proxy_kit.py"]
    dup = calls[0][1]
    assert dup[dup.index("--count") + 1] == "3"
    assert json.loads(dup[dup.index("--names") + 1]) == [["Mannequin", "MannequinRig"], ["Mannequin2", "MannequinRig2"],
                                                         ["Mannequin3", "MannequinRig3"]]
    assert json.loads(dup[dup.index("--bodies") + 1]) == [{}, {"scale": 0.8}, {}]
    kit = calls[1][1]
    assert kit[0] == "--kits"                                        # ONE Blender job for the kits
    assert Path(kit[1]).name == "proxy_kits.json"
    todo = json.loads((d / "inputs" / "proxy_kits.json").read_text())
    assert [k["rig"] for k in todo] == ["MannequinRig", "MannequinRig2"]       # kit has no proxy kit
    assert [r["name"] for r in m["characters"]] == ["masked", "fox", "kit"] and m["character_name"] == "masked"
    assert m["look_keyframes"] == "off" and m["proxy_kits"][2] is None
    assert [sorted(v) for v in m["views_all"]] == [sorted(["front", "right", "back", "left", "face"])] * 3
    assert m["views_all"][2]["front"] == "inputs/views3/front.png" and (d / m["views_all"][2]["front"]).is_file()
    assert m["sheet"]["layout"] == "group" and m["sheet"]["file"] == "inputs/sheet.png"
    g = m["prompt"]["group"]
    assert g["order"] == [1, 0, 2] and g["views"] == ["front", "back"]          # fox is on the left (screen x 0.2)
    assert m["prompt"]["character"].startswith("**Panels 1-2 (Character 1):** the first character in full body "
                                               "(front, back) on black: a slim young wanderer")
    sheet = cv2.imread(str(d / "inputs" / "sheet.png"))
    assert sheet.shape == (896, 1536, 3)
    cols = {n: np.where((np.abs(sheet.astype(int) - c).max(axis=2) < 10).any(axis=0))[0] for n, c in COLOURS.items()}
    assert cols["fox"].max() < cols["masked"].min() and cols["masked"].max() < cols["kit"].min()   # picture order
    rows = {n: np.where((np.abs(sheet.astype(int) - c).max(axis=2) < 10).any(axis=1))[0] for n, c in COLOURS.items()}
    assert rows["fox"].min() > rows["masked"].min()                  # the shorter one is drawn shorter
    assert [s["side"] for s in m["slots"]] == ["centre", "left", "right"]
    assert m["pass_key_source"]["collections"] == ["Mannequin", "Mannequin2", "Mannequin3"]
    det = runner._character_rows(m)
    assert det == [("Character 1 (centre)", "masked"), ("Character 2 (left)", "fox"), ("Character 3 (right)", "kit")]
    text = runner.video_prompt(m)
    assert "Three characters from the reference sheet side by side: on the left, the first character, a slim young" in text


def test_a_three_character_run_with_a_prop_picture_and_a_fork(env):
    m = new3(env, prop_images={"Ball_*": str(env["ball"])})
    sh = m["sheet"]
    assert sh["prop_layout"]["drawn"] == "group" and sh["prop_places"] == {"Ball_*": "bottom-right"}
    d = runner.run_dir(m["id"])
    w = d / "work"
    w.mkdir(exist_ok=True)
    for f in ("clip.mp4", "clip_2.mp4"):
        (w / f).write_bytes(b"x")
    st = json.loads((d / "state.json").read_text())
    st["steps"].update({"controls": {"status": "done", "outputs": {"clips": 2}},
                        "clip": {"status": "done", "outputs": {"clip": str(w / "clip.mp4")},
                                 "parts": {"1": {"outputs": {"clip": str(w / "clip.mp4")}, "relay": {"s1": "x"}},
                                           "2": {"outputs": {"clip": str(w / "clip_2.mp4")}}}}})
    (d / "state.json").write_text(json.dumps(st))
    f = runner.fork_run(m["id"], 2, prop_images={"Ball_*": str(env["ball"])}, video="They bow.")
    assert runner.run_characters(f) == runner.run_characters(m) and f["prompt"]["group"] == m["prompt"]["group"]
    assert f["sheet"]["prop_layout"]["drawn"] == "group"            # the fork draws the N groups again
    assert "characters" not in runner.FORK_FIELDS                   # a fork keeps its characters (its kept clips)


def test_runs_refuse_too_many_characters_and_video_templates(env):
    with pytest.raises(runner.RunError, match="at most 4"):
        runner.new_run(env["blend"], env["sheet"], "t", "v", characters=["masked", "fox", "kit", "fox", "kit"])
    with pytest.raises(runner.RunError, match="keyframe looks"):
        new3(env, look_keyframes="auto")
    with pytest.raises(runner.RunError, match="own mannequin collection"):
        runner.new_run(env["blend"], env["sheet"], "t", "v",
                       characters=[{"name": "masked"}, {"name": "fox", "collection": "Mannequin"}])


def test_crop_instructions_name_the_background_not_a_sky():
    """A crop of a gym wall came back with a blue sky when the instruction said "keep ... the sky" (trio_test)."""
    for text in (runner.Run.CROP_LOOK_INSTRUCTION, runner.Run.GROUP_LOOK_INSTRUCTION):
        assert "sky" not in text and "as they are in image 1" in text
    assert "{place}" in runner.Run.GROUP_LOOK_INSTRUCTION
