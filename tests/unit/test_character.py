"""Character library: the three ways in, the sheet, and what a run receives (fake image jobs, no ComfyUI)."""

import cv2
import numpy as np
from pathlib import Path
import pytest

from cinestudio import character


def figure(colour=(200, 150, 40), scarf=True, w=200, h=300):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.rectangle(img, (70, 30), (130, 280), colour, -1)            # body
    if scarf:
        cv2.rectangle(img, (60, 90), (140, 120), (30, 30, 200), -1)  # red scarf
    return img


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    calls = []

    def fake_job(profile, variant, values, files, out, tag, **kw):
        calls.append((profile, variant, values.get("instruction") or values.get("seed")))
        lost_scarf = "back view" in str(values.get("instruction"))
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), figure(scarf=not lost_scarf))
        return {"job": f"job{len(calls)}", "seconds": 1.0, "file": str(out)}
    monkeypatch.setattr(character, "image_job", fake_job)
    from cinestudio import blender
    renders = []

    def fake_blender(script, blend, args, log, timeout=None):      # render_turnaround.py: the bare mannequin
        assert script == "render_turnaround.py"
        renders.append(Path(blend))
        out = Path(args[0]); out.mkdir(parents=True, exist_ok=True)
        for v in ("front", "right", "back", "left"):
            img = np.zeros((600, 400, 3), np.uint8); img[50:560, 150:250] = 220       # a tall thin mannequin
            cv2.imwrite(str(out / f"{v}.png"), img)
    monkeypatch.setattr(blender, "run_job", fake_blender)
    monkeypatch.setattr(character, "default_pose_shot", lambda: tmp_path / "shot.blend")
    return {"tmp": tmp_path, "calls": calls, "renders": renders}


def test_text_path_to_ready(lib):
    c = character.create("kitsune", "a masked wanderer with a red scarf", candidates=3, seed=5)
    assert [x["seed"] for x in c["candidates"]] == [5, 6, 7]
    assert (character.char_dir("kitsune") / "candidates" / "candidates.jpg").is_file()
    with pytest.raises(character.CharacterError):
        character.pick("kitsune", 99)
    character.pick("kitsune", 6)
    with pytest.raises(character.CharacterError, match="fast mode"):     # the Lightning LoRA is not downloaded
        character.make_views("kitsune", "fast")
    c = character.make_views("kitsune")
    assert c["status"] == "sheet" and {"right", "back", "left", "face"} <= set(c["files"])
    assert any("back" in w for w in c["checks"]["warnings"])           # the scarf is missing from the back view
    sheet = cv2.imread(c["files"]["sheet"])
    assert sheet.shape == (896, 1536, 3)
    with pytest.raises(character.CharacterError, match="not ready"):
        character.for_run("kitsune")
    character.finish("kitsune")
    path, text = character.for_run("kitsune")
    assert path.endswith("sheet.png") and "red scarf" in text and "Panels 2-5 (Character)" in text and "left side" in text
    qwen = [x for x in lib["calls"] if x[0] == "qwenedit"]
    assert qwen[0][1] == "anypose"                                     # body type human: the front is posed first
    qwen = [x for x in qwen if x[1] != "anypose"]
    assert len(qwen) == 3 and all(str(x[2]).startswith("<sks>") for x in qwen)   # the face is cut from the front


def test_image_path_skips_candidates(lib, tmp_path):
    img = tmp_path / "front.png"
    cv2.imwrite(str(img), figure())
    character.import_image("hero", img, "a hero")
    assert character.make_views("hero")["status"] == "sheet"
    assert not [x for x in lib["calls"] if x[0] == "zimage"]


def test_sheet_path_is_ready_at_once(lib, tmp_path):
    sheet = tmp_path / "sheet.png"
    cv2.imwrite(str(sheet), np.zeros((896, 1536, 3), np.uint8))
    character.import_sheet("enemy", sheet, "Left panel: ... Next three panels: ...")
    path, text = character.for_run("enemy")
    assert path.endswith("sheet.png") and text.startswith("Left panel")    # the user's own panel text, unchanged
    assert lib["calls"] == []


def test_names_and_duplicates(lib):
    for bad in ("", "Has Space", "../up", "UPPER"):
        with pytest.raises(character.CharacterError):
            character.char_dir(bad, must_exist=False)
    character.create("one", "x", candidates=1)
    with pytest.raises(character.CharacterError, match="exists"):
        character.create("one", "x", candidates=1)
    assert [c["name"] for c in character.list_characters()] == ["one"]


def test_edit_keeps_the_old_front_and_clears_old_views(lib, tmp_path):
    img = tmp_path / "front.png"
    cv2.imwrite(str(img), figure())
    character.import_image("hero", img, "a hero", body="other")       # no pose step: the edit numbers start at 1
    character.make_views("hero")
    ref = tmp_path / "ref.png"
    cv2.imwrite(str(ref), figure(colour=(40, 150, 200)))
    c = character.edit_front("hero", "make the belt like in image 2", reference=ref)
    d = character.char_dir("hero")
    assert (d / "front_v1.png").is_file() and c["edits"][0]["instruction"].startswith("make the belt")
    assert c["status"] == "front" and "sheet" not in c["files"] and "back" not in c["files"]
    c = character.edit_front("hero", "fix the hands")
    assert (d / "front_v2.png").is_file() and len(c["edits"]) == 2
    edit_calls = [x for x in lib["calls"] if x[1] == "edit"]
    assert [x[2] for x in edit_calls] == ["make the belt like in image 2", "fix the hands"]


def test_delete_character(lib, tmp_path):
    img = tmp_path / "front.png"
    cv2.imwrite(str(img), figure())
    character.import_image("hero", img, "a hero")
    assert character.delete_character("hero")["deleted"]
    assert character.list_characters() == []
    with pytest.raises(character.CharacterError):
        character.delete_character("hero")


def test_cli_jobs_share_the_app_job_list(lib, tmp_path, capsys):
    import json as _json
    from cinestudio import cli, jobs
    img = tmp_path / "front.png"
    cv2.imwrite(str(img), figure())
    character.import_image("hero", img, "a hero")
    assert cli.main(["character", "views", "hero"]) == 0
    rows = jobs.list_jobs("hero")
    assert rows[0]["action"] == "views" and rows[0]["status"] == "done" and rows[0]["via"] == "cli"
    last = _json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert last["ok"] and last["character"]["status"] == "sheet"


def test_slots_make_only_the_missing_views(lib, tmp_path):
    paths = {}
    for slot in ("front", "back"):
        paths[slot] = tmp_path / f"{slot}.jpg"
        cv2.imwrite(str(paths[slot]), figure())
    c = character.import_slots("duo", paths, "a hero")
    assert c["provided"] == ["front", "back"] and c["status"] == "front"
    c = character.make_views("duo")
    made = [x[2] for x in lib["calls"] if x[0] == "qwenedit"]
    assert not any("back view" in str(m) for m in made) and len(made) == 2        # right, left (face: from the front)
    assert c["status"] == "sheet"
    c = character.edit_front("duo", "fix the hands")
    assert "back" in c["files"] and "right" not in c["files"]                     # the user's view stays


def test_all_body_slots_give_a_sheet_without_gpu(lib, tmp_path):
    paths = {}
    for slot in ("front", "right", "back", "left"):
        paths[slot] = tmp_path / f"{slot}.png"
        cv2.imwrite(str(paths[slot]), figure())
    c = character.import_slots("full", paths, "a hero")
    assert c["status"] == "sheet" and lib["calls"] == []
    with pytest.raises(character.CharacterError):
        character.import_slots("bad", {"back": paths["back"]}, "no front")


def test_pick_over_an_edited_front_can_be_undone(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    import json as _json
    c = character._new("fox", "a fox", "", "text")
    d = character.char_dir("fox")
    (d / "candidates").mkdir()
    for s, v in ((1, 10), (2, 200)):
        cv2.imwrite(str(d / "candidates" / f"front_s{s}.png"), np.full((8, 8, 3), v, np.uint8))
    c["candidates"] = [{"seed": s, "file": str(d / "candidates" / f"front_s{s}.png")} for s in (1, 2)]
    character.save(c)
    character.pick("fox", 1)
    cv2.imwrite(str(d / "front.png"), np.full((8, 8, 3), 77, np.uint8))     # the user's edited front
    c = character.load("fox")
    c["files"]["sheet"] = str(d / "sheet.png")
    character.save(c)
    c = character.pick("fox", 2)
    assert c["edits"][-1]["mode"] == "pick" and "sheet" not in c["files"]
    assert cv2.imread(str(d / "front.png"))[0, 0, 0] == 200
    character.undo_edit("fox")
    assert cv2.imread(str(d / "front.png"))[0, 0, 0] == 77                 # the edited front is back


def test_rename_moves_the_folder_and_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    img = tmp_path / "sheet.png"
    cv2.imwrite(str(img), np.zeros((64, 64, 3), np.uint8))
    character.import_sheet("fox", img, "a fox")
    c = character.rename("fox", "kitsune2")
    assert c["name"] == "kitsune2" and not (character.library() / "fox").exists()
    assert c["files"]["sheet"].startswith(str(character.char_dir("kitsune2")))
    assert character.load("kitsune2")["files"]["sheet"] == c["files"]["sheet"]
    with pytest.raises(character.CharacterError):
        character.rename("kitsune2", "Bad Name")


def _kit(tmp_path, monkeypatch):
    """A character with a front, four views and a sheet; image_job writes a flat image of a known value."""
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    c = character._new("fox", "a fox", "", "text")
    d = character.char_dir("fox")
    (d / "views").mkdir()
    fr = np.zeros((96, 64, 3), np.uint8)
    fr[20:80, 20:44] = 90
    cv2.imwrite(str(d / "front.png"), fr)
    c["files"]["front"] = str(d / "front.png")
    for i, v in enumerate(character.VIEWS, 1):
        img = np.zeros((96, 64, 3), np.uint8)
        img[20:80, 20:44] = 40 + i
        cv2.imwrite(str(d / "views" / f"{v}.png"), img)
        c["files"][v] = str(d / "views" / f"{v}.png")
    character.save(c)
    value = {"v": 200}

    def fake_job(profile, variant, values, files, out, tag, **kw):
        img = np.zeros((96, 64, 3), np.uint8)
        img[20:80, 20:44] = value["v"]
        cv2.imwrite(str(out), img)
        return {"job": "x", "seconds": 1.0, "file": str(out)}
    monkeypatch.setattr(character, "image_job", fake_job)
    return d, value


def test_edit_and_redo_one_view_then_undo(tmp_path, monkeypatch):
    d, value = _kit(tmp_path, monkeypatch)
    left = d / "views" / "left.png"
    before = cv2.imread(str(left))[50, 30, 0]
    c = character.edit_image("fox", "give the hand five fingers", target="left")
    assert c["edits"][-1]["target"] == "left" and cv2.imread(str(left))[50, 30, 0] == 200
    assert c["files"]["front"] and c["files"]["right"] and c["status"] == "sheet"     # other images stay; new sheet
    value["v"] = 150
    c = character.redo_view("fox", "left", seed=7)
    assert c["edits"][-1]["mode"] == "redo" and cv2.imread(str(left))[50, 30, 0] == 150
    character.undo_edit("fox", "left")
    assert cv2.imread(str(left))[50, 30, 0] == 200                       # back to the edited view
    character.undo_edit("fox", "left")
    assert cv2.imread(str(left))[50, 30, 0] == before                    # back to the first view
    with pytest.raises(character.CharacterError):
        character.undo_edit("fox", "left")
    with pytest.raises(character.CharacterError):
        character.undo_edit("fox", "right")                              # each image has its own history


def test_region_edit_changes_only_the_box(tmp_path, monkeypatch):
    d, value = _kit(tmp_path, monkeypatch)
    left = d / "views" / "left.png"
    before = cv2.imread(str(left))
    seen = {}

    def fake_job(profile, variant, values, files, out, tag, **kw):
        src = cv2.imread(str(list(files.values())[0]))
        seen["shape"] = src.shape
        cv2.imwrite(str(out), np.full_like(src, 250))              # the model paints the crop white
        return {"job": "x", "seconds": 1.0, "file": str(out)}
    monkeypatch.setattr(character, "image_job", fake_job)
    c = character.edit_image("fox", "give the hand five fingers", target="left", region="10,10,50,60")
    after = cv2.imread(str(left))
    assert max(seen["shape"][:2]) == 1024                          # the model saw the crop large
    assert (after[:10] == before[:10]).all() and (after[:, 50:] == before[:, 50:]).all()   # outside: unchanged
    assert after[35, 30, 0] == 250                                 # inside: changed
    assert c["edits"][-1]["region"] == [10, 10, 50, 60]


def test_import_needs_no_text_and_gets_no_default_style(lib, tmp_path):
    """Imported images or a sheet carry their own look: no anime default, and the description is optional."""
    front = tmp_path / "front.jpg"
    cv2.imwrite(str(front), figure())
    c = character.import_slots("plain", {"front": front}, "")
    assert c["description"] == "" and c["style"] == ""
    sheet = tmp_path / "sheet.png"
    cv2.imwrite(str(sheet), np.zeros((448, 768, 3), np.uint8))
    c = character.import_sheet("plainsheet", sheet, "")
    assert c["style"] == ""
    with pytest.raises(character.CharacterError):
        character.create("textless", "   ")


def test_view_prompts_use_the_chosen_distance():
    assert character.view_prompt("right") == "<sks> right side view eye-level shot medium shot"
    assert character.view_prompt("back", "wide shot") == "<sks> back view eye-level shot wide shot"
    assert character.view_prompt("face") == "<sks> front view eye-level shot close-up"     # the face keeps its close-up
    with pytest.raises(character.CharacterError):
        character.view_prompt("left", "far away")


def test_the_face_panel_is_cut_from_the_front():
    import numpy as np
    front = np.zeros((1536, 1024, 3), np.uint8)
    front[200:1400, 400:620] = 120                                  # a body
    front[200:380, 450:570] = 250                                   # a white head at the top
    face = character.face_from_front(front)
    assert face.shape == (1536, 1024, 3)
    assert face[300:700, 400:620].mean() > 200                      # the head fills the upper middle of the panel
    with pytest.raises(character.CharacterError):
        character.face_from_front(np.zeros((64, 64, 3), np.uint8))


def test_views_of_a_human_pose_the_front_first(lib, tmp_path):
    """Order A (2026-10-22): body type human -> `views` poses the FRONT like the mannequin (AnyPose), then makes the
    views from the posed front; a second views run does not pose again; undo keeps the old front."""
    img = tmp_path / "front.png"
    cv2.imwrite(str(img), figure())
    character.import_image("posed", img, "a hero in a red coat.")
    c = character.make_views("posed")
    kinds = [x[1] for x in lib["calls"] if x[0] == "qwenedit"]
    assert kinds[0] == "anypose" and kinds[1:] and set(kinds[1:]) == {"angle"}
    pose_text = next(x[2] for x in lib["calls"] if x[1] == "anypose")
    assert "white mannequin in Picture 2" in pose_text and "red coat" in pose_text
    assert c["status"] == "sheet" and c["reposed"]["views"] == ["front"] and character.front_is_posed(c)
    edit = next(e for e in c["edits"] if e.get("mode") == "repose")
    assert edit["target"] == "front" and Path(edit["before"]).is_file()
    aligned = cv2.imread(str(character.char_dir("posed") / "repose" / "front_aligned.png"))
    ys = np.nonzero(aligned.max(axis=2) > 12)[0]
    assert abs(ys.min() - 50) <= 2 and abs(ys.max() - 559) <= 2                      # the mannequin's height and feet line
    n = len(lib["calls"])
    character.make_views("posed")                                                     # the front is posed already
    assert [x[1] for x in lib["calls"][n:]] and "anypose" not in [x[1] for x in lib["calls"][n:]]


def test_views_of_other_bodies_keep_the_drawn_pose(lib, tmp_path):
    img = tmp_path / "front.png"
    cv2.imwrite(str(img), figure())
    character.import_slots("robot", {"front": img}, "a robot", body="other")
    c = character.make_views("robot")
    assert "anypose" not in [x[1] for x in lib["calls"]] and not lib["renders"] and c["body"] == "other"
    with pytest.raises(character.CharacterError):
        character.import_slots("crab", {"front": img}, "a crab", body="crab")


def test_repose_poses_the_front_then_makes_the_views(lib, tmp_path):
    img = tmp_path / "front.png"
    cv2.imwrite(str(img), figure())
    character.import_slots("old", {"front": img}, "a hero", body="other")
    c = character.repose("old", shot=tmp_path / "my_shot.blend")
    kinds = [x[1] for x in lib["calls"] if x[0] == "qwenedit"]
    assert kinds.count("anypose") == 1 and kinds[0] == "anypose" and "angle" in kinds
    assert lib["renders"] == [(tmp_path / "my_shot.blend").resolve()] and c["status"] == "sheet"


def test_instant_actions_skip_the_gpu_queue(lib, tmp_path, monkeypatch):
    """An undo, pick or finish uses no GPU: no queue ticket, no wait for a turn (user, 2026-10-22: "we should only queue
    up tasks that require gpu usage"); a GPU action still takes its turn."""
    from cinestudio import jobs, workqueue
    img = tmp_path / "front.png"
    cv2.imwrite(str(img), figure())
    character.import_image("quick", img, "a hero", body="other")
    added, waited = [], []
    monkeypatch.setattr(workqueue, "add", lambda kind, ref, label: added.append(ref))
    monkeypatch.setattr(workqueue, "wait_turn", lambda *a, **k: waited.append(a[1]))
    monkeypatch.setattr(workqueue, "done", lambda kind, ref: None)
    monkeypatch.setitem(jobs.ACTIONS, "noop", lambda name, **kw: {"name": name})
    job = jobs.run_now("noop", "quick")
    assert job["status"] == "done" and not added and not waited
    job = jobs.run_now("views", "quick")                             # a GPU action: a ticket and a turn
    assert job["status"] == "done" and added == [job["id"]] and waited == [job["id"]]
