"""A run's prop pictures (2026-10-25, user: "generate a ball as a prop ... and swap it out with the ball in the blender
reference video"): frozen into the run, drawn into a prop panel of the run's own copy of the sheet, named in the
Reference Sheet Description. No Blender, no GPU: synthetic pictures and a hand-written template record."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from cinestudio import character, charsheet, cli, jobs, mcp_server, runner, shots, webapp
from cinestudio.webapi import runs as wr

PROP_BGR = (200, 60, 150)                          # the test ball's colour (BGR): violet


def ball_picture(path: Path, size: int = 512, alpha: bool = False) -> Path:
    """A ball on a near-black backdrop whose underside is nearly black too (the case that a loose cut-out ate)."""
    img = np.full((size, size, 3), 4, np.uint8)
    c, r = size // 2, int(size * 0.36)
    cv2.circle(img, (c, c), r, PROP_BGR, -1)
    cv2.circle(img, (c, c + r // 2), r // 2, (11, 8, 13), -1)                      # the dark underside
    cv2.circle(img, (c, c), r, (11, 8, 13), 3)
    if alpha:
        a = np.zeros((size, size), np.uint8)
        cv2.circle(a, (c, c), r + 2, 255, -1)
        img = np.dstack([img, a])
    cv2.imwrite(str(path), img)
    return path


def figure(h: int, w: int, colour) -> np.ndarray:
    img = np.zeros((h + 40, w + 40, 3), np.uint8)
    cv2.rectangle(img, (20, 20), (20 + w, 20 + h), colour, -1)
    return img


def cut(img):
    return charsheet.cutout(img)


# ---------------------------------------------------------------- the sheet drawing
def test_a_prop_column_keeps_the_sheet_size_and_the_bodies_scale():
    face = cut(figure(400, 280, (90, 90, 90)))                   # about the masked sheet's proportions
    views = [cut(figure(900, 230, (30, 160, 30))) for _ in range(4)]
    ball = charsheet.prop_cutout(cv2.imread(str(ball_picture(Path(__import__("tempfile").mkdtemp()) / "b.png"))))
    plain, lay0 = {}, {}
    a = character.sheet_image(face, views, 1.0, layout=lay0)
    b = character.sheet_image(face, views, 1.0, props=[ball], layout=plain)
    assert a.shape == b.shape == (896, 1536, 3)
    assert plain["body_scale"] == 1.0                            # the face gave way, not the bodies
    assert plain["face_px"][0] < lay0["face_px"][0]
    (x0, y0, x1, y1), = plain["prop_boxes"]
    assert plain["prop_places"] == ["bottom-right"] and x1 <= 1536 - 12 and y0 > 896 // 2 - 20   # the bottom cell
    assert x1 - x0 >= 300                                        # readable: about 0.86 of half the panel height
    patch = b[y0:y1, x0:x1].reshape(-1, 3)
    assert (np.abs(patch.astype(int) - PROP_BGR).max(axis=1) < 30).mean() > 0.2     # the ball is there
    body_cols = np.where((np.abs(b.astype(int) - (30, 160, 30)).max(axis=2) < 10).any(axis=0))[0]
    assert body_cols.max() < x0                                  # the bodies stay left of the prop column
    area = (x1 - x0) * (y1 - y0) / (1536 * 896)
    assert 0.05 < area < 0.25


def test_two_props_take_both_cells_and_a_crowded_row_gives_way_in_order():
    face = cut(figure(400, 380, (90, 90, 90)))
    ball = cut(figure(300, 300, PROP_BGR))
    wide = [cut(figure(900, 560, (30, 160, 30))) for _ in range(4)]        # wide costumes: no room at all
    lay = {}
    img = character.sheet_image(face, wide, 1.0, props=[ball, ball], layout=lay)
    assert img.shape == (896, 1536, 3) and lay["prop_places"] == ["top-right", "bottom-right"]
    top, bottom = lay["prop_boxes"]
    assert top[3] <= bottom[1]
    assert lay["prop_side"] == int(872 * charsheet.PROP_MIN)     # the props gave way to their smallest size ...
    assert lay["face_px"][1] <= int(872 * 0.3) + 1               # ... after the face went to its smallest
    assert lay["body_scale"] < 1.0                               # and only then everything together
    with pytest.raises(ValueError, match="1 to 2"):
        character.sheet_image(face, wide, 1.0, props=[ball] * 3)


def test_the_duo_sheet_and_any_other_sheet_get_a_prop_column():
    groups = [[cut(figure(900, 200, (30, 160, 30))) for _ in range(3)], [cut(figure(700, 200, (160, 30, 30)))] * 3]
    ball = cut(figure(300, 300, PROP_BGR))
    lay = {}
    img = charsheet.assemble_groups(groups, ratios=[1.0, 0.8], props=[ball], layout=lay)
    assert img.shape == (896, 1536, 3) and lay["prop_places"] == ["bottom-right"]
    assert lay["body_scale"] >= 0.85                             # the character panels keep most of their size
    x0 = lay["prop_boxes"][0][0]
    assert not (np.abs(img[:, x0:].astype(int) - (30, 160, 30)).max(axis=2) < 10).any()
    # an imported sheet (no views to draw again): its content moves left for the column
    sheet = np.zeros((896, 1536, 3), np.uint8)
    sheet[100:800, 40:1500] = (30, 160, 30)
    lay = {}
    out = charsheet.add_prop_column(sheet, [ball], layout=lay)
    assert out.shape == sheet.shape and lay["body_scale"] < 1.0
    assert (np.abs(out[:, lay["prop_boxes"][0][0]:].astype(int) - (30, 160, 30)).max(axis=2) < 10).sum() == 0


def test_prop_cutout_keeps_a_dark_underside_and_reads_alpha(tmp_path):
    raw = cv2.imread(str(ball_picture(tmp_path / "b.png")))
    c = charsheet.prop_cutout(raw)
    assert abs(c.shape[0] - c.shape[1]) <= 4 and c.shape[0] >= 0.72 * 512   # the whole ball, underside included
    a = charsheet.prop_cutout(cv2.imread(str(ball_picture(tmp_path / "a.png", alpha=True)), cv2.IMREAD_UNCHANGED))
    assert a.shape[2] == 3 and a.shape[0] < 512
    with pytest.raises(ValueError, match="no prop"):
        charsheet.prop_cutout(np.full((64, 64, 3), 7, np.uint8))


# ---------------------------------------------------------------- the run option
def template_record():
    return {"name": "dodge", "kind": "blender", "status": "done",
            "layers": {"scan": {"characters": [{"collection": "Mannequin", "rig": "MannequinRig"}],
                                "props": [{"name": "Ball_1"}, {"name": "Ball_2"}, {"name": "Hoop"}]},
                       "descriptions": {"set": "a gym", "props": {"Ball_*": "matte violet dodgeballs"}}}}


def test_prop_pictures_are_checked(tmp_path):
    rec, ball = template_record(), ball_picture(tmp_path / "ball.png")
    assert runner.check_prop_images(rec, {"Ball_*": str(ball), "Hoop": ""}) == {"Ball_*": ball.resolve()}
    with pytest.raises(runner.RunError, match="no prop matches 'Bat'"):
        runner.check_prop_images(rec, {"Bat": str(ball)})
    with pytest.raises(runner.RunError, match="no prop picture"):
        runner.check_prop_images(rec, {"Ball_*": str(tmp_path / "missing.png")})
    (tmp_path / "x.png").write_bytes(b"not a picture")
    with pytest.raises(runner.RunError, match="not a picture"):
        runner.check_prop_images(rec, {"Ball_*": str(tmp_path / "x.png")})
    cv2.imwrite(str(tmp_path / "tiny.png"), np.full((16, 16, 3), 200, np.uint8))
    with pytest.raises(runner.RunError, match="32..8192"):
        runner.check_prop_images(rec, {"Ball_*": str(tmp_path / "tiny.png")})
    rec["layers"]["scan"]["props"].append({"name": "Cone"})
    with pytest.raises(runner.RunError, match="at most 2"):
        runner.check_prop_images(rec, {"Ball_*": str(ball), "Hoop": str(ball), "Cone": str(ball)})
    with pytest.raises(runner.RunError, match="no props"):
        runner.check_prop_images({"layers": {}}, {"Ball_*": str(ball)})


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A library-like character with a panels5 sheet and its views, a template record with balls, no Blender."""
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    blend = tmp_path / "shot.blend"
    blend.write_bytes(b"BLENDER-fake")
    cdir = tmp_path / "char"
    cdir.mkdir()
    files = {}
    for k, (h, w, col) in {"front": (900, 260, (30, 160, 30)), "right": (900, 200, (30, 150, 30)),
                           "back": (900, 260, (30, 140, 30)), "left": (900, 200, (30, 130, 30)),
                           "face": (400, 380, (90, 90, 90))}.items():
        cv2.imwrite(str(cdir / f"{k}.png"), figure(h, w, col))
        files[k] = str(cdir / f"{k}.png")
    sheet = character.sheet_image(cut(cv2.imread(files["face"])),
                                  [cut(cv2.imread(files[k])) for k in character.SHEET_PANELS], 1.0)
    cv2.imwrite(str(cdir / "sheet.png"), sheet)
    files["sheet"] = str(cdir / "sheet.png")
    crec = {"name": "masked", "status": "ready", "files": files, "sheet_layout": "panels5", "sheet_scale": 1.0,
            "description": "on black: a masked figure"}
    monkeypatch.setattr(character, "load", lambda name: dict(crec))
    monkeypatch.setattr(character, "require_video_ready", lambda name: {})
    monkeypatch.setattr(shots, "library_record", lambda b: template_record())

    def fake_snapshot(src, dst, report, allow_external=False):
        dst.write_bytes(src.read_bytes())
        r = {"ok": True, "packed": [], "missing": [], "unsupported": []}
        report.write_text(json.dumps(r))
        return r
    monkeypatch.setattr(runner, "snapshot_blend", fake_snapshot)
    monkeypatch.setattr(runner.blender, "check", lambda: {"ok": True, "version": "5.2.0"})
    return {"blend": blend, "sheet": cdir / "sheet.png", "tmp": tmp_path, "ball": ball_picture(tmp_path / "ball.png")}


def new(env, **kw):
    return runner.new_run(env["blend"], env["sheet"], "on black: a masked figure", "she throws the ball",
                          character_name="masked", proxy="off", **kw)


def test_a_run_freezes_its_prop_picture_and_draws_its_own_sheet(env):
    m = new(env, prop_images={"Ball_*": str(env["ball"])}, props={"Ball_*": "a matte violet dodgeball"})
    d = runner.run_dir(m["id"])
    rec = m["prop_images"]["Ball_*"]
    assert rec["source"] == str(env["ball"].resolve()) and (d / rec["file"]).is_file()      # frozen
    env["ball"].unlink()                                         # the run does not need the original any more
    sh = m["sheet"]
    assert sh["layout"] == "panels5" and sh["props_file"] == "inputs/sheet_props.png"
    assert sh["prop_places"] == {"Ball_*": "bottom-right"} and sh["prop_layout"]["drawn"] == "panels5"
    plain, props = cv2.imread(str(d / sh["file"])), cv2.imread(str(d / sh["props_file"]))
    assert plain.shape == props.shape                            # LTX gets the same shape
    assert np.array_equal(plain, cv2.imread(str(env["sheet"])))  # the character's sheet stays as it was
    assert sh["prop_layout"]["body_scale"] == 1.0
    # the sheet step prepares the prop sheet; the keyframes keep cutting panels from the character's sheet
    step = {}
    runner.Run.step_sheet(SimpleNamespace(m=m, d=d, work=d / "work"), step)
    black = cv2.imread(step["outputs"]["sheet"])
    x0, y0, x1, y1 = sh["prop_layout"]["prop_boxes"][0]
    assert (np.abs(black[y0:y1, x0:x1].astype(int) - PROP_BGR).max(axis=2) < 30).mean() > 0.2
    lost = (np.abs(black[y0:y1, x0:x1].astype(int) - props[y0:y1, x0:x1].astype(int)).max(axis=2) > 0).mean()
    assert lost < 0.01                                           # the near-black underside survives the black-out
    # the prompt: the prop line in the sheet part, the props sentence in the target
    p = runner.video_prompt(m)
    head, target = p.split("### Target Description ")
    assert "**Prop:** a matte violet dodgeball, shown in the bottom-right panel." in head
    assert "In the scene: a matte violet dodgeball." in target
    det = runner.run_details(m["id"])
    assert det["sheet_file"].endswith("sheet_props.png")
    assert ("Prop pictures", "Balls -> ball.png") in [tuple(x) for x in det["form"]]
    # no picture: no prop sheet, no line
    m0 = new(env)
    assert m0["prop_images"] is None and "props_file" not in m0["sheet"] and "**Prop:**" not in runner.video_prompt(m0)


def test_the_prop_line_takes_the_templates_words_or_the_groups_name(env):
    m = new(env, prop_images={"Ball_*": str(env["ball"])})
    assert "**Prop:** matte violet dodgeballs, shown in the bottom-right panel." in runner.video_prompt(m)
    m["layers"] = None
    assert "**Prop:** ball, shown in" in runner.video_prompt(m)
    m["prompt_format"] = None                                     # an old run's format keeps the line too
    assert runner.video_prompt(m).startswith("Reference sheet: on black: a masked figure **Prop:** ball")


def test_a_prop_picture_needs_a_template_with_props(env, tmp_path):
    env_rec = template_record()
    env_rec["layers"]["scan"]["props"] = []
    shots_lr = shots.library_record
    try:
        shots.library_record = lambda b: env_rec
        with pytest.raises(runner.RunError, match="no props"):
            new(env, prop_images={"Ball_*": str(env["ball"])})
    finally:
        shots.library_record = shots_lr
    with pytest.raises(runner.RunError, match="no prop matches"):
        new(env, prop_images={"Bat": str(env["ball"])})


def test_a_fork_changes_or_removes_the_prop_picture(env, tmp_path):
    m = new(env, prop_images={"Ball_*": str(env["ball"])})
    d = runner.run_dir(m["id"])
    runner.Run.step_sheet(SimpleNamespace(m=m, d=d, work=d / "work"), st_sheet := {})
    w = d / "work"
    for f in ("clip.mp4", "clip_2.mp4"):
        (w / f).write_bytes(b"x")
    st = json.loads((d / "state.json").read_text())
    st["steps"].update({"sheet": {"status": "done", **st_sheet},
                        "controls": {"status": "done", "outputs": {"clips": 2}},
                        "clip": {"status": "done", "outputs": {"clip": str(w / "clip.mp4")},
                                 "parts": {"1": {"outputs": {"clip": str(w / "clip.mp4")}, "relay": {"s1": "x"}},
                                           "2": {"outputs": {"clip": str(w / "clip_2.mp4")}}}}})
    (d / "state.json").write_text(json.dumps(st))
    gold = tmp_path / "gold.png"
    img = cv2.imread(str(ball_picture(tmp_path / "g.png")))
    img[np.abs(img.astype(int) - PROP_BGR).max(axis=2) < 5] = (0, 200, 255)                # a gold ball
    cv2.imwrite(str(gold), img)
    f = runner.fork_run(m["id"], 2, prop_images={"Ball_*": str(gold)})
    d2 = runner.run_dir(f["id"])
    assert f["prop_images"]["Ball_*"]["source"] == str(gold.resolve())
    st2 = json.loads((d2 / "state.json").read_text())
    black = cv2.imread(st2["steps"]["sheet"]["outputs"]["sheet"])
    assert str(d2) in st2["steps"]["sheet"]["outputs"]["sheet"]
    x0, y0, x1, y1 = f["sheet"]["prop_layout"]["prop_boxes"][0]
    assert (np.abs(black[y0:y1, x0:x1].astype(int) - (0, 200, 255)).max(axis=2) < 30).mean() > 0.2
    # the parent is untouched
    assert m["prop_images"] == json.loads((d / "manifest.json").read_text())["prop_images"]
    # an empty picture removes it: the fork's sheet is the character's sheet again
    f2 = runner.fork_run(m["id"], 2, prop_images={"Ball_*": ""})
    d3 = runner.run_dir(f2["id"])
    assert f2["prop_images"] is None and "props_file" not in f2["sheet"] and not (d3 / "inputs" / "sheet_props.png").exists()
    st3 = json.loads((d3 / "state.json").read_text())
    assert np.array_equal(cv2.imread(st3["steps"]["sheet"]["outputs"]["sheet"]),
                          cv2.imread(str(d3 / "work" / "sheet" / "sheet_black.png")))
    assert "**Prop:**" not in runner.video_prompt(f2)
    with pytest.raises(runner.RunError, match="prop images must map"):
        runner.fork_run(m["id"], 2, prop_images=["x"])


# ---------------------------------------------------------------- CLI, MCP and the app
def test_the_cli_prop_image_flag(monkeypatch, tmp_path):
    assert cli.prop_image_args(["Ball_*=C:/x/ball.png", 'Hoop = "h.png"']) == {"Ball_*": "C:/x/ball.png", "Hoop": "h.png"}
    with pytest.raises(cli.CliInputError, match="PATTERN=PICTURE"):
        cli.prop_image_args(["ball.png"])
    with pytest.raises(cli.CliInputError, match="give the picture"):
        cli.prop_image_args(["Ball_*="])
    assert cli.prop_image_args(["Ball_*="], allow_empty=True) == {"Ball_*": ""}
    seen = {}
    blend = tmp_path / "shot.blend"
    blend.write_bytes(b"x")
    monkeypatch.setattr(cli.character, "for_run", lambda name: ("sheet.png", "text"))
    monkeypatch.setattr(cli.runner, "new_run", lambda *a, **k: seen.update(new=k) or {"id": "r1"})
    monkeypatch.setattr(cli.runner, "fork_run", lambda *a, **k: seen.update(fork=k) or {"id": "r2", "forked_from": {}})
    monkeypatch.setattr(cli, "_start_detached", lambda rid, *a: {"ok": True})
    cli.main(["run", "new", "--blend", str(blend), "--character", "fox", "--video", "v", "--prop-image", "Ball_*=b.png"])
    assert seen["new"]["prop_images"] == {"Ball_*": "b.png"}
    cli.main(["run", "new", "--blend", str(blend), "--character", "fox", "--video", "v"])
    assert "prop_images" not in seen["new"]
    cli.main(["run", "fork", "r1", "--from-clip", "2", "--prop-image", "Ball_*="])
    assert seen["fork"]["prop_images"] == {"Ball_*": ""}


def test_mcp_passes_the_prop_pictures(tmp_path, monkeypatch):
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
    mcp_server.new_video_run(str(blend), "fox", "v", prop_images={"Ball_*": "C:/b.png"})
    assert seen["new"]["prop_images"] == {"Ball_*": "C:/b.png"}
    mcp_server.new_video_run(str(blend), "fox", "v")
    assert "prop_images" not in seen["new"]
    mcp_server.fork_run("r1", 2, prop_images={"Ball_*": ""})
    assert seen["fork"]["prop_images"] == {"Ball_*": ""}


def test_the_app_uploads_a_prop_picture_and_sends_it_with_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(jobs.spawn, "detached", lambda *a, **k: (0, True))
    ball = ball_picture(tmp_path / "ball.png").read_bytes()

    async def go():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as c:
            from aiohttp import FormData
            fd = FormData()
            fd.add_field("image", ball, filename="ball.png", content_type="image/png")
            ok = await (await c.post("/api/prop-images", data=fd)).json()
            fd = FormData()
            fd.add_field("image", b"nope", filename="x.png", content_type="image/png")
            bad = await (await c.post("/api/prop-images", data=fd)).json()
            return ok, bad
    ok, bad = asyncio.run(go())
    assert ok["ok"] and Path(ok["data"]["path"]).is_file() and ok["data"]["width"] == 512
    assert Path(ok["data"]["path"]).parent == wr.prop_uploads()
    assert not bad["ok"] and "not a picture" in bad["error"]
    assert list((tmp_path / "home" / "uploads").glob("upload_*")) == []          # the raw uploads are gone
    path = ok["data"]["path"]
    assert wr._props_args({"prop_images": {"Ball_*": path, "Hoop": ""}}, None) == {"prop_images": {"Ball_*": path}}
    with pytest.raises(ValueError, match="not an uploaded prop picture"):
        wr._props_args({"prop_images": {"Ball_*": str(tmp_path / "ball.png")}}, None)
    with pytest.raises(ValueError, match="video template"):
        wr._props_args({"prop_images": {"Ball_*": path}}, "beach")
    assert wr._prop_image_paths({"Ball_*": ""}, allow_empty=True) == {"Ball_*": ""}
