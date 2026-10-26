"""The Backgrounds tab's API (cinestudio/webapi/backgrounds.py): list, new, depth, style redraw, delete. No GPU: the
detached `cinestudio background ...` worker is replaced by a recorder, and the queue tickets stay unclaimed."""
import asyncio

import cv2
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from cinestudio import background, webapp, workqueue
from cinestudio.webapi import backgrounds as api


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    started = []
    monkeypatch.setattr(api.spawn, "detached", lambda cmd, log, cwd: started.append(cmd) or (0, True))
    monkeypatch.setattr(api.time, "sleep", lambda s: None)
    src = tmp_path / "pano.png"
    cv2.imwrite(str(src), np.full((64, 128, 3), 90, np.uint8))
    background.save(src, "a rooftop", 1, scene="a rooftop", name="roof")
    return started


def call(fn):
    async def run():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as client:
            return await fn(client)
    return asyncio.run(run())


def test_the_list_has_the_card_fields_and_the_flat_form(home):
    async def go(c):
        d = (await (await c.get("/api/backgrounds")).json())["data"]
        assert d["jobs"] == []
        b = d["backgrounds"][0]
        assert b["name"] == "roof" and b["source"] == "text" and b["depth"] is False and b["runs"] == 0
        assert b["busy"] == [] and b["url"].startswith("/files/")
        assert (await c.get(b["url"])).status == 200
        flat = (await (await c.get("/api/backgrounds?flat=1")).json())["data"]
        assert flat == [{k: b[k] for k in ("name", "kind", "scene", "seed", "created", "url", "camera")}]
        assert b["kind"] == "pano" and flat[0]["camera"] == ""
    call(go)


def test_a_new_background_queues_a_worker_and_shows_as_a_job(home):
    async def go(c):
        r = await c.post("/api/backgrounds/new", json={"scene": "", "style": ""})
        assert r.status == 400                                         # no place: refused before the queue
        r = await c.post("/api/backgrounds/new", json={"scene": "a harbour at dawn", "name": "roof"})
        assert r.status == 400                                         # the name is taken
        r = await c.post("/api/backgrounds/new", json={"scene": "a harbour at dawn", "name": "harbour", "seed": 5,
                                                        "depth": False})
        ref = (await r.json())["data"]["ref"]
        assert ref == "bg-new-harbour"
        cmd = home[-1]
        assert cmd[-2:] == ["--ref", ref] and "--scene=a harbour at dawn" in cmd and "--no-depth" in cmd
        assert cmd[cmd.index("--seed") + 1] == "5"
        d = (await (await c.get("/api/backgrounds")).json())["data"]
        assert [j["ref"] for j in d["jobs"]] == [ref] and d["jobs"][0]["status"] == "queued"
    call(go)


def test_depth_and_style_jobs_are_refused_twice_and_block_the_delete(home):
    async def go(c):
        assert (await c.post("/api/backgrounds/nobody/depth", json={})).status == 400
        r = await c.post("/api/backgrounds/roof/depth", json={})
        assert (await r.json())["data"]["ref"] == "bg-depth-roof"
        assert (await c.post("/api/backgrounds/roof/depth", json={})).status == 400     # already in the queue
        b = (await (await c.get("/api/backgrounds")).json())["data"]["backgrounds"][0]
        assert [j["kind"] for j in b["busy"]] == ["depth"]
        r = await c.post("/api/backgrounds/roof/delete", json={})
        assert r.status == 400 and "job" in (await r.json())["error"]
        assert (await c.post("/api/backgrounds/roof/style", json={"keep": 0.3})).status == 400   # no LoRA
        r = await c.post("/api/backgrounds/roof/style", json={"style_lora": "3d.safetensors", "keep": 0.3,
                                                               "strength": 1.0, "style": "3dsrx"})
        assert (await r.json())["data"]["ref"] == "bg-style-roof"
        cmd = home[-1]
        assert cmd[cmd.index("style") + 1] == "roof" and "--style-lora=3d.safetensors" in cmd
        assert cmd[cmd.index("--keep") + 1] == "0.3"
        assert (await c.post("/api/backgrounds/roof/other", json={})).status == 404
    call(go)


def test_delete_removes_the_panorama_when_no_job_waits(home):
    async def go(c):
        r = await c.post("/api/backgrounds/roof/delete", json={})
        assert r.status == 200
        d = (await (await c.get("/api/backgrounds")).json())["data"]
        assert d["backgrounds"] == []
    call(go)


def test_a_failed_worker_start_frees_the_ticket(home, monkeypatch):
    def boom(*a, **k):
        raise OSError("no python")
    monkeypatch.setattr(api.spawn, "detached", boom)

    async def go(c):
        assert (await c.post("/api/backgrounds/roof/depth", json={})).status >= 400
        assert not [t for t in workqueue.tickets() if t["ref"] == "bg-depth-roof"]
    call(go)


def test_a_failed_job_shows_until_it_is_dismissed(home):
    background.record_job("bg-depth-roof", "failed", "panorama roof: 360 depth", "RuntimeError: ComfyUI is down")

    async def go(c):
        d = (await (await c.get("/api/backgrounds")).json())["data"]
        assert [f["ref"] for f in d["failed"]] == ["bg-depth-roof"] and "ComfyUI" in d["failed"][0]["error"]
        assert (await c.post("/api/background-jobs/bg-depth-roof/dismiss", json={})).status == 200
        assert (await (await c.get("/api/backgrounds")).json())["data"]["failed"] == []
        assert (await c.post("/api/background-jobs/..%2Fx/dismiss", json={})).status in (400, 404)
        # a new start of the same job removes its old failure record
        background.record_job("bg-depth-roof", "failed", "x", "old")
        await c.post("/api/backgrounds/roof/depth", json={})
        assert (await (await c.get("/api/backgrounds")).json())["data"]["failed"] == []
    call(go)


def test_the_queue_refuses_a_second_ticket_for_the_same_ref(home):
    workqueue.add_new("lab", "bg-depth-roof", "x")
    with pytest.raises(workqueue.QueueBusy):
        workqueue.add_new("lab", "bg-depth-roof", "x")


def test_delete_is_refused_while_a_job_waits_also_from_the_cli(home):
    workqueue.add_new("lab", "bg-style-roof", "x")
    with pytest.raises(background.BackgroundError, match="job in the queue"):
        background.delete("roof")


def test_an_old_empty_reserved_folder_is_swept(home):
    import os
    import time
    d = background.library_dir() / "half_done"
    d.mkdir()
    os.utime(d, (time.time() - 7200, time.time() - 7200))
    (background.library_dir() / "fresh").mkdir()
    assert background.sweep_reserved() == ["half_done"]
    assert (background.library_dir() / "fresh").is_dir()


def test_a_failing_worker_leaves_a_record_and_frees_its_ticket(home, monkeypatch, capsys):
    from cinestudio import cli

    def boom(name, on_event=None):
        raise RuntimeError("no MoGe model")
    monkeypatch.setattr(background, "depth", boom)
    workqueue.add_new("lab", "bg-depth-roof", "panorama roof: 360 depth")
    rc = cli.main(["background", "depth", "roof", "--ref", "bg-depth-roof"])
    assert rc != 0 and '"ok": false' in capsys.readouterr().out
    assert [f["error"] for f in background.failed_jobs()] == ["RuntimeError: no MoGe model"]
    assert not [t for t in workqueue.tickets() if t["ref"] == "bg-depth-roof"]


def test_a_reservation_is_kept_while_a_style_pass_waits(home):
    import os
    import time
    d = background.library_dir() / "roof_3d"
    d.mkdir()
    (d / background.RESERVED).write_text("{}")
    os.utime(d, (time.time() - 7200, time.time() - 7200))
    workqueue.add_new("lab", "bg-style-roof", "x")                # its ticket names the SOURCE, not the destination
    assert background.sweep_reserved() == [] and d.is_dir()
    workqueue.done("lab", "bg-style-roof")
    assert background.sweep_reserved() == ["roof_3d"]


CAM = {"resolution": [1536, 896], "height_m": 1.45, "pitch_deg": -7.5, "horizon_from_top": 0.31, "moves": False}


def test_backplates_share_the_library_with_panoramas(home, tmp_path):
    """User, 2026-10-24: the Backgrounds tab includes both panoramas and regular (one-picture) backgrounds."""
    src = tmp_path / "plate_src.png"
    cv2.imwrite(str(src), np.full((560, 960, 3), 120, np.uint8))
    rec = background.save_plate(src, "a forest", 7, scene="a forest", name="forest", camera=CAM, shot="shan_shen_bu")
    assert rec["kind"] == "plate" and rec["file"].endswith("plate.png") and rec["camera"]["height_m"] == 1.45
    assert [r["name"] for r in background.list_backgrounds("plate")] == ["forest"]
    assert {r["name"] for r in background.list_backgrounds()} == {"forest", "roof"}
    with pytest.raises(background.BackgroundError, match="backplate"):
        background.require_pano("forest")
    with pytest.raises(background.BackgroundError, match="backplate"):
        background.depth("forest")
    with pytest.raises(background.BackgroundError, match="exists"):
        background.import_plate(src, "roof")
    own = background.import_plate(src, "mine", "my garden")
    assert own["kind"] == "plate" and own["source"] == "import" and own["camera"] is None

    async def go(c):
        d = (await (await c.get("/api/backgrounds")).json())["data"]
        by = {b["name"]: b for b in d["backgrounds"]}
        assert by["forest"]["kind"] == "plate" and by["forest"]["depth"] is False
        assert "1.45 m" in by["forest"]["camera"] and by["forest"]["shot"] == "shan_shen_bu"
        assert (await c.post("/api/backgrounds/forest/depth", json={})).status == 400
        assert (await c.post("/api/backgrounds/forest/style", json={"style_lora": "x.safetensors"})).status == 400
        flat = (await (await c.get("/api/backgrounds?flat=1")).json())["data"]
        assert {b["name"]: b["kind"] for b in flat} == {"forest": "plate", "mine": "plate", "roof": "pano"}
    call(go)


def test_a_picture_upload_becomes_a_backplate(home, tmp_path):
    import aiohttp
    ok, buf = cv2.imencode(".png", np.full((300, 500, 3), 60, np.uint8))

    async def go(c):
        fd = aiohttp.FormData()
        fd.add_field("image", buf.tobytes(), filename="garden.png", content_type="image/png")
        fd.add_field("name", "garden")
        fd.add_field("scene", "my garden")
        r = await c.post("/api/backgrounds/import", data=fd)
        assert (await r.json())["data"]["background"] == {"name": "garden", "kind": "plate"}
        r = await c.post("/api/backgrounds/import", data=aiohttp.FormData({"name": "x"}))
        assert r.status == 400                                         # no picture
    call(go)
    assert background.load("garden")["scene"] == "my garden"
    assert not list((tmp_path / "uploads").glob("*")) if (tmp_path / "uploads").is_dir() else True


def test_a_new_backplate_needs_a_template_camera(home, monkeypatch, tmp_path):
    from cinestudio import shots
    blend = tmp_path / "shots" / "dance" / "shot.blend"
    blend.parent.mkdir(parents=True)
    blend.write_bytes(b"BLENDER")
    monkeypatch.setattr(shots, "list_shots", lambda: [{"name": "dance", "files": {"blend": str(blend)}}])

    async def go(c):
        r = await c.post("/api/backgrounds/new", json={"kind": "plate", "scene": "a forest"})
        assert r.status == 400                                         # no camera chosen
        r = await c.post("/api/backgrounds/new", json={"kind": "plate", "scene": "a forest", "shot": str(tmp_path / "x.blend")})
        assert r.status == 400                                         # not a template's .blend
        r = await c.post("/api/backgrounds/new", json={"kind": "plate", "scene": "a forest", "shot": str(blend),
                                                        "name": "forest2"})
        ref = (await r.json())["data"]["ref"]
        assert ref == "bg-plate-forest2"
        cmd = home[-1]
        assert cmd[cmd.index("background") + 1] == "plate" and f"--shot={blend}" in cmd and "--no-depth" not in cmd
        d = (await (await c.get("/api/backgrounds")).json())["data"]
        assert [j["kind"] for j in d["jobs"]] == ["plate"]
    call(go)


def test_the_plate_prompt_names_the_camera():
    t = background.plate_prompt("a forest clearing", "stylized 3D", CAM)
    assert t.startswith("a forest clearing. A wide view") and "1.4 m" in t or "1.5 m" in t
    assert "31% down from the top" in t and "slightly down (8 degrees)" in t and "stylized 3D. " in t
    assert background.plate_size(CAM) == (1920, 1120)
    with pytest.raises(background.BackgroundError):
        background.plate_prompt("", "", CAM)



def test_unnamed_backplates_are_named_by_content_and_never_replace_another(home, tmp_path):
    """Codex review 2026-10-24: a name from the file size let two different pictures replace each other."""
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    im = np.zeros((40, 60, 3), np.uint8)
    cv2.imwrite(str(a), im)
    im2 = im.copy()
    im2[5, 5] = 255                                          # another picture, the same size on disk is likely
    cv2.imwrite(str(b), im2)
    ra, rb = background.import_plate(a), background.import_plate(b)
    assert ra["name"] != rb["name"] and ra["name"].startswith("plate_")
    assert background.import_plate(a)["name"] == ra["name"]  # the same picture again: its record, no second copy
    assert len(background.list_backgrounds("plate")) == 2
    with pytest.raises(background.BackgroundError, match="exists"):
        background.save_plate(b, name=ra["name"])             # a named save never replaces
    assert cv2.imread(ra["file"]).max() == 0                  # the first picture is untouched


def test_a_failed_or_duplicate_upload_leaves_no_files(home, tmp_path):
    import aiohttp
    ok, buf = cv2.imencode(".png", np.full((30, 40, 3), 60, np.uint8))

    async def go(c):
        fd = aiohttp.FormData()
        fd.add_field("image", buf.tobytes(), filename="a.png", content_type="image/png")
        fd.add_field("image", buf.tobytes(), filename="b.png", content_type="image/png")
        assert (await c.post("/api/backgrounds/import", data=fd)).status == 400
        fd = aiohttp.FormData()
        fd.add_field("image", buf.tobytes(), filename="a.png", content_type="image/png")
        fd.add_field("extra", buf.tobytes(), filename="c.png", content_type="image/png")
        assert (await c.post("/api/backgrounds/import", data=fd)).status == 400
    call(go)
    up = tmp_path / "uploads"
    assert not up.is_dir() or not list(up.iterdir())



def test_a_lab_job_shows_its_sampling_progress(home):
    """The queue's bar for experiments and background jobs moved never (frac was always 0)."""
    from cinestudio import lab, workqueue as wq
    with lab.turn("bg-new-prog", "new panorama: test"):
        lab.note_progress(10, 40)
        item = next(i for i in wq.view()["items"] if i["ref"] == "bg-new-prog")
        assert item["status"] == "running" and item["frac"] == 0.25
    assert lab.progress("bg-new-prog") == 0.0                   # the record goes with the ticket
    lab.note_progress(5, 10)                                    # outside a turn: nothing is written
    assert not list(lab.lab_dir().glob("*.progress.json"))


def test_a_lab_job_counts_its_time_from_its_turn_not_from_its_ticket(home, monkeypatch):
    """A lab job that waited behind another one showed "1 h left" at 22%: the elapsed time began when the ticket was
    taken. Now it begins when the job gets its turn."""
    import time
    from cinestudio import lab, workqueue as wq
    t0 = time.time()
    with lab.turn("k3d-test", "front: test"):
        item = next(i for i in wq.view()["items"] if i["ref"] == "k3d-test")
        assert item["started"] >= t0 and lab.started("k3d-test") == item["started"]
    assert lab.started("k3d-test") is None or lab.started("k3d-test") != item["started"]   # the record goes too


def test_a_lab_turn_with_two_jobs_times_each_job_from_its_own_start(home, monkeypatch):
    """A lab turn can run several jobs (each bar restarts at 0): the estimate used the turn's start for the second job
    too (Codex review 2026-10-24). The progress record keeps the current job's start."""
    import time
    from cinestudio import lab
    clock = [1000.0]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    with lab.turn("two-jobs", "two jobs"):
        lab.note_progress(1, 40)
        clock[0] = 1100.0
        lab.note_progress(40, 40)                    # job 1 ends
        clock[0] = 1200.0
        lab.note_progress(1, 40)                     # job 2 starts: its bar restarts
        clock[0] = 1250.0
        lab.note_progress(10, 40)
        assert lab.started("two-jobs") == 1200.0


def test_a_backplate_is_drawn_over_the_templates_own_set(home, tmp_path, monkeypatch):
    """background from-set (2026-10-24): Blender renders the set alone from the still camera; Qwen-Image-Edit redraws
    it with Picture 1 only; the result is resized to the frame and stored as a plate with its set render."""
    import json as _json
    from pathlib import Path
    from cinestudio import blender, character
    shot = tmp_path / "shot.blend"
    shot.write_bytes(b"BLENDER")
    cam = {"moves": False, "resolution": [1536, 896], "height_m": 1.5, "horizon_from_top": 0.3, "pitch_deg": -8}
    monkeypatch.setattr(background, "camera_info", lambda s: dict(cam))
    jobs = []

    def fake_run_job(script, blend, args, log, timeout=None):
        a = dict(zip(args[::2], args[1::2]))
        assert script == "render_set.py" and Path(blend) == shot
        assert (int(a["--width"]), int(a["--height"])) == background.edit_size(*background.plate_size(cam))
        cv2.imwrite(a["--out"], np.full((int(a["--height"]), int(a["--width"]), 3), 120, np.uint8))
        Path(log).write_text("SET " + _json.dumps({"hidden": ["Body"], "kept": ["Wall", "Hoop", "Cart"]}))
    monkeypatch.setattr(blender, "run_job", fake_run_job)

    def fake_image_job(profile, variant, values, files, out, tag, **kw):
        jobs.append((profile, variant, values, files))
        from cinestudio.workflow import ProfileWorkflow
        g = ProfileWorkflow("qwenedit", "edit").build({"instruction": "x", "seed": 1}, files=files)
        kw["graph_edit"](g)                                            # the Kontext centre crop is taken out
        assert "scale" not in g and g["enc"]["inputs"]["pixels"] == ["img", 0]
        assert g["pos_txt"]["inputs"]["image1"] == ["img", 0] and "image2" not in g["pos_txt"]["inputs"]
        src = cv2.imread(str(next(iter(files.values()))))
        cv2.imwrite(str(out), np.full(src.shape, 200, np.uint8))       # the set render's own size (1344x784)
        return {"seconds": 1}
    monkeypatch.setattr(character, "image_job", fake_image_job)
    rec = background.make_set_plate("a school gym", shot, style="stylized 3D animation", seed=5, name="gym_set")
    (profile, variant, values, files), = jobs
    assert (profile, variant) == ("qwenedit", "edit") and len(files) == 1        # Picture 1 only: the set render
    assert "a school gym" in values["instruction"] and "keeps exactly its place" in values["instruction"]
    assert "mannequin" in values["negative"]
    img = cv2.imread(rec["file"])
    w, h = background.plate_size(cam)
    assert (img.shape[1], img.shape[0]) == (w, h) and rec["source"] == "set" and rec["kind"] == "plate"
    assert (Path(rec["file"]).parent / "set.png").is_file() and rec["set_objects"] == ["Wall", "Hoop", "Cart"]
    monkeypatch.setattr(background, "camera_info", lambda s: (_ for _ in ()).throw(
        background.BackgroundError("a backplate needs a still camera")))
    import pytest
    with pytest.raises(background.BackgroundError, match="still camera"):
        background.make_set_plate("x", shot, name="moving")
