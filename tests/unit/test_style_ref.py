"""Style reference pictures made by the video model (cinestudio/style_ref.py): the txt2img graph and its binding, the
job arguments, the sharpest-frame pick, use_reference (the user's copy of a built-in style), the crop helpers, the CLI,
the MCP tools and the API routes."""
import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cinestudio import character, cli, model_pairs, style_ref, styles
from cinestudio.workflow import ProfileWorkflow


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def _pic(path, w=768, h=1152, sharp=True, value=120):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = np.full((h, w, 3), value, np.uint8)
    if sharp:
        im[::4, :, :] = 255                                   # fine stripes: a high Laplacian variance
    cv2.imwrite(str(path), im)
    return path


def _with_lora(name="3d_animated", lora="3dsrx_1250.safetensors"):
    return styles.save(name, {"lora": {"name": lora, "strength": 0.4}})


# ---------------------------------------------------------------- the graph
@pytest.mark.parametrize("prof", ["ltx25", "ltx25dev"])
def test_the_txt2img_graph_binds_text_size_frames_and_an_optional_lora(prof):
    wf = ProfileWorkflow(prof, "txt2img")
    st = {"words": "stylized 3D", "light": "", "lora": {"name": "3dsrx_1250.safetensors", "strength": 0.4}}
    v = style_ref.job_values(st, None, 7, wf.capabilities["recipe"])
    g = wf.build(dict(v, output_prefix="cinestudio/x"), files={})
    assert g["lat"]["inputs"] | {} == {"width": 768, "height": 1152, "length": 17, "batch_size": 1}
    assert g["alat"]["inputs"]["frames_number"] == 17 and g["noise1"]["inputs"]["noise_seed"] == 7
    assert g["lora_style"]["inputs"]["lora_name"] == "3dsrx_1250.safetensors" and g["guider1"]["inputs"]["model"] == ["lora_style", 0]
    assert g["pos"]["inputs"]["text"].startswith("3dsrx, stylized 3D.")
    assert not any(n["class_type"] in ("LoadImage", "LTXAddVideoICLoRAGuide", "LTXICLoRALoaderModelOnly")
                   for n in g.values())                                       # text only: no guide, no IC-LoRA
    assert [g[n]["inputs"]["filename_prefix"] for n in wf.bind["outputs"]["frames"]] == ["cinestudio/x"] * 5
    assert [g[n.replace("save", "pick")]["inputs"]["batch_index"] for n in wf.bind["outputs"]["frames"]] == [0, 4, 8, 12, 16]
    sig = g["sigmas1"]["inputs"]["sigmas"].split(",")
    assert float(sig[0]) == 1.0 and float(sig[-1]) == 0.0
    assert len(sig) == (9 if prof == "ltx25" else 31)                         # each model's own whole schedule
    g2 = wf.build(dict(style_ref.job_values({**st, "lora": None}, None, 7, wf.capabilities["recipe"]),
                       output_prefix="x"), files={})
    assert "lora_style" not in g2 and g2["guider1"]["inputs"]["model"] == ["ckpt", 0]   # no LoRA: LTX's own look
    with pytest.raises(ValueError, match="multiple of 64"):
        wf.build(dict(v, width=700), files={})


def test_the_txt2img_pair_is_registered_with_its_generator():
    pair = model_pairs.PAIRS["txt2img"]
    assert pair.generator == "setup/make_txt2img_graphs.py" and not model_pairs.pair_diff(pair)


# ---------------------------------------------------------------- prompt, frames, crops
def test_the_prompt_and_the_trigger_word():
    st = {"words": "stylized cinematic 3D animation.", "light": "evenly lit from the front",
          "lora": {"name": "3dsrx_1250.safetensors", "strength": 0.4}}
    p = style_ref.prompt_for(st)
    assert p.startswith("3dsrx, stylized cinematic 3D animation. A still shot of a young woman in a hooded jacket")
    assert "full body, evenly lit from the front." in p
    assert style_ref.trigger_word({"lora": {"name": "anime_v2.safetensors"}}) is None
    assert style_ref.trigger_word({"lora": {"name": "x.safetensors", "trigger": "zzstyle"}}) == "zzstyle"
    assert style_ref.trigger_word({"lora": None}) is None
    assert not style_ref.prompt_for({"words": "flat", "lora": None}, "a red ball").startswith(",")
    with pytest.raises(style_ref.StyleRefError):
        style_ref.prompt_for(st, "   ")


def test_the_sharpest_frame_is_picked(tmp_path):
    soft = _pic(tmp_path / "a.png", 64, 64, sharp=False)
    crisp = _pic(tmp_path / "b.png", 64, 64, sharp=True)
    soft2 = _pic(tmp_path / "c.png", 64, 64, sharp=False)
    best, score = style_ref.sharpest([soft, crisp, soft2])
    assert best == crisp and score > style_ref.sharpness(soft)
    assert style_ref.sharpest([soft, soft2])[0] == soft                       # a tie: the first frame
    with pytest.raises(style_ref.StyleRefError):
        style_ref.sharpest([])


def test_the_crop_helpers(tmp_path):
    assert style_ref.default_crop(768, 1152) == [192, 476, 576, 860]          # half the width, on the clothes
    assert style_ref.default_crop(100, 100, side=2.0) == [0, 0, 100, 100]     # kept inside the picture
    assert styles.check_crop("10, 20, 110, 220") == [10, 20, 110, 220]
    for bad in ("1,2,3", "0,0,10,10", "50,0,10,100", "a,b,c,d"):
        with pytest.raises(styles.StyleError):
            styles.check_crop(bad)
    pic = _pic(tmp_path / "p.png", 200, 100)
    with pytest.raises(styles.StyleError, match="outside"):
        styles.check_crop([0, 0, 300, 50], pic)
    out = style_ref.crop_file(pic, [10, 10, 74, 90], tmp_path / "c.png")
    assert cv2.imread(str(out)).shape[:2] == (80, 64)


# ---------------------------------------------------------------- use_reference and the Qwen crop
def test_use_reference_makes_the_users_copy_of_a_builtin_style_and_keeps_a_crop(home, tmp_path):
    assert styles.load("3d_animated")["builtin"]
    cand = _pic(tmp_path / "cand.png")
    r = style_ref.use_reference("3d_animated", cand)
    s = styles.load("3d_animated")
    assert not s["builtin"] and (styles.user_dir() / "3d_animated.json").is_file()      # the user's copy
    assert Path(s["reference"]).parent == styles.refs_dir() and Path(s["reference"]).is_file()
    assert cv2.imread(s["reference"]).shape[:2] == (1152, 768)                           # the whole picture is kept
    assert s["reference_crop"] == r["crop"] == style_ref.default_crop(768, 1152)
    q = style_ref.reference_for_qwen("3d_animated")
    assert cv2.imread(str(q)).shape[:2] == (384, 384)                                    # Qwen gets the crop
    first = s["reference"]
    style_ref.use_reference("3d_animated", cand, "0,0,200,300")
    s2 = styles.load("3d_animated")
    assert s2["reference_crop"] == [0, 0, 200, 300] and not Path(first).exists()         # the old copy is dropped
    assert cv2.imread(str(style_ref.reference_for_qwen("3d_animated"))).shape[:2] == (300, 200)
    styles.save("3d_animated", {"reference": str(_pic(tmp_path / "up.png", 64, 64))})    # an upload: no crop box
    assert styles.load("3d_animated")["reference_crop"] is None
    assert style_ref.reference_for_qwen("3d_animated") == Path(styles.load("3d_animated")["reference"])
    with pytest.raises(style_ref.StyleRefError):
        style_ref.use_reference("3d_animated", tmp_path / "missing.png")
    with pytest.raises(styles.StyleError, match="outside"):
        style_ref.use_reference("3d_animated", cand, [0, 0, 900, 900])


def test_new_skin_sends_the_crop_as_picture_2(home, monkeypatch, tmp_path):
    d = character.library() / "kitsune"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "front.png"), np.zeros((64, 48, 3), np.uint8))
    (d / "character.json").write_text(json.dumps({"name": "kitsune", "status": "ready", "source": "text",
                                                  "files": {"front": str(d / "front.png")}, "description": "a fox."}))
    style_ref.use_reference("3d_animated", _pic(tmp_path / "cand.png"))
    seen = []
    monkeypatch.setattr(character, "edit_image", lambda name, ins, reference=None, **kw: seen.append((ins, reference)))
    monkeypatch.setattr(character, "make_views", lambda name, **kw: character.load(name))
    character.new_skin("kitsune", "3d_animated")
    ins, ref = seen[0]
    assert "rendering style of Picture 2" in ins and cv2.imread(str(ref)).shape[:2] == (384, 384)


# ---------------------------------------------------------------- the job
def test_make_reference_runs_one_text_job_per_seed_and_keeps_the_sharpest_frame(home, monkeypatch):
    _with_lora()
    calls = []

    def job(profile, variant, values, files, out, tag, extra_outputs=None, **kw):
        calls.append((profile, variant, values, files, tag, sorted(extra_outputs)))
        sharp_node = "save_8" if values["seed"] == 202 else "save_0"
        for node, f in [("save_0", out), *extra_outputs.items()]:
            _pic(f, 96, 144, sharp=node == sharp_node)
        return {"file": str(out), "seconds": 1.0}
    monkeypatch.setattr(character, "image_job", job)
    events = []
    r = style_ref.make_reference("3d_animated", "a knight in a forest", [202, 303], on_event=events.append)
    assert [c[:2] for c in calls] == [("ltx25", "txt2img")] * 2 and [c[3] for c in calls] == [{}, {}]
    v = calls[0][2]
    assert v["seed"] == 202 and v["style_lora"] == "3dsrx_1250.safetensors" and v["style_lora_strength"] == 0.4
    assert (v["width"], v["height"], v["frames"]) == (768, 1152, 17) and "a knight in a forest" in v["prompt"]
    assert calls[0][5] == ["save_12", "save_16", "save_4", "save_8"]
    assert [c["frame"] for c in r["candidates"]] == [2, 0]                       # frame index in PICKS (8 -> 2)
    for c in r["candidates"]:
        assert Path(c["file"]).parent == style_ref.candidates_dir("3d_animated") and Path(c["file"]).is_file()
    assert not (style_ref.candidates_dir("3d_animated") / ".work").exists()
    assert [e["seed"] for e in events if e["event"] == "candidate"] == [202, 303]
    assert {c["seed"] for c in style_ref.candidates("3d_animated")} == {202, 303}
    with pytest.raises(style_ref.StyleRefError):
        style_ref.make_reference("3d_animated", seeds="1,1")
    with pytest.raises(styles.StyleError):
        style_ref.make_reference("no_such_style")


def test_image_job_downloads_the_extra_outputs(home, monkeypatch, tmp_path):
    from cinestudio.backend import JobState
    monkeypatch.setattr(character.server, "ensure_family", lambda fam, url=None: {"restarted": False})
    monkeypatch.setattr(character.server, "note_owned", lambda pid, url=None: None)
    from cinestudio.locks import FileLock                    # never the real GPU lock (shared with the live queue)
    monkeypatch.setattr(character.server, "gpu_lock", lambda url=None, timeout=0, note=None:
                        FileLock(tmp_path / "gpu.lock", timeout=timeout, note=note))
    got = []

    class Fake:
        def submit(self, graph, files, key, prefix=None):
            return "j1"

        def watch(self, jid, on_event):
            return JobState(jid, "succeeded")

        def download(self, jid, node, dst):
            got.append(node)
            _pic(dst, 8, 8)
            return dst
    v = style_ref.job_values({"words": "x", "lora": None}, None, 1, "distilled")
    r = character.image_job("ltx25", "txt2img", v, {}, tmp_path / "f0.png", "t", backend_factory=Fake,
                            extra_outputs={"save_4": tmp_path / "f4.png", "save_8": tmp_path / "f8.png"})
    assert got == ["save_0", "save_4", "save_8"] and set(r["extra"]) == {"save_4", "save_8"}


def test_submit_queues_a_detached_worker_and_the_status_follows_it(home, monkeypatch):
    from cinestudio import spawn, workqueue
    started = []
    monkeypatch.setattr(spawn, "detached", lambda cmd, log, cwd: started.append(cmd) or (4242, True))
    r = style_ref.submit("3d_animated", "a knight", [1, 2], via="mcp")
    assert r["ref"].startswith("sref-3d_animated-") and r["seeds"] == [1, 2]
    cmd = started[0]
    assert cmd[2:6] == ["cinestudio", "style", "make-reference", "3d_animated"] and "--subject=a knight" in cmd
    assert cmd[cmd.index("--ref") + 1] == r["ref"] and cmd[cmd.index("--seeds") + 1] == "1,2"
    assert any(t["ref"] == r["ref"] for t in workqueue.tickets())
    assert style_ref.job_status("3d_animated")["status"] in ("queued", "running")
    with pytest.raises(style_ref.StyleRefError, match="in the queue already"):
        style_ref.submit("3d_animated")
    workqueue.done("lab", r["ref"])
    style_ref.record_job("3d_animated", ref=r["ref"], status="done")
    assert style_ref.job_status("3d_animated")["status"] == "done"


# ---------------------------------------------------------------- CLI, MCP, API
def _cli(capsys, *argv):
    code = cli.main(list(argv))
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip()]
    return code, lines[-1]


def test_cli_make_and_use_reference(home, monkeypatch, capsys, tmp_path):
    seen = {}

    def fake_run(name, subject, seeds, ref, on_event=None, video_model=None):
        seen.update(name=name, subject=subject, seeds=seeds, ref=ref, model=video_model)
        return {"style": name, "candidates": []}
    monkeypatch.setattr(style_ref, "run_job", fake_run)
    code, out = _cli(capsys, "style", "make-reference", "3d_animated", "--subject", "a cat", "--seeds", "5,6",
                     "--video-model", "full")
    assert code == 0 and out["ok"]
    assert seen == {"name": "3d_animated", "subject": "a cat", "seeds": "5,6", "ref": None, "model": "full"}
    assert style_ref.profile_of("full") == "ltx25dev" and style_ref.profile_of(None) is None
    code, out = _cli(capsys, "style", "use-reference", "3d_animated", str(_pic(tmp_path / "c.png")), "--crop", "0,0,64,64")
    assert code == 0 and out["crop"] == [0, 0, 64, 64]
    code, out = _cli(capsys, "style", "show", "3d_animated")
    assert out["style"]["reference_crop"] == [0, 0, 64, 64] and out["reference_candidates"] == []


def test_mcp_tools(home, monkeypatch, tmp_path):
    from cinestudio import mcp_server
    monkeypatch.setattr(style_ref, "submit", lambda name, subject, seeds, via, video_model=None:
                        {"ref": "sref-x", "via": via, "seeds": seeds})
    assert mcp_server.make_style_reference("3d_animated", seeds=[1]) == {"ok": True, "ref": "sref-x", "via": "mcp",
                                                                         "seeds": [1]}
    r = mcp_server.use_style_reference("3d_animated", str(_pic(tmp_path / "c.png")), [0, 0, 100, 100])
    assert r["ok"] and r["crop"] == [0, 0, 100, 100]
    assert mcp_server.show_style("3d_animated")["style"]["reference_crop"] == [0, 0, 100, 100]


def test_api_routes(home, monkeypatch, tmp_path):
    from aiohttp.test_utils import TestClient, TestServer

    from cinestudio import webapp
    monkeypatch.setattr(style_ref, "submit", lambda name, subject, seeds, via: {"ref": "sref-y", "seeds": seeds,
                                                                               "subject": subject})
    cand = _pic(style_ref.candidates_dir("3d_animated") / "ref_x_s1.png")
    style_ref._add_candidates("3d_animated", [{"file": str(cand), "seed": 1, "frame": 0, "sharpness": 9.0, "made": 1.0}])
    outside = _pic(tmp_path / "elsewhere.png")

    async def go():
        client = TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"})
        await client.start_server()
        try:
            r = await client.get("/api/styles/3d_animated/reference-candidates")
            d = (await r.json())["data"]
            assert d["candidates"][0]["seed"] == 1 and d["candidates"][0]["url"].startswith("/files/style_files/")
            assert d["subject"] == style_ref.SUBJECT and d["job"] is None
            r = await client.post("/api/styles/3d_animated/make-reference", json={"subject": "a cat", "seeds": [3]})
            assert (await r.json())["data"] == {"ref": "sref-y", "seeds": [3], "subject": "a cat"}
            r = await client.post("/api/styles/3d_animated/make-reference", json={"bogus": 1})
            assert r.status == 400
            r = await client.post("/api/styles/3d_animated/use-reference", json={"file": str(outside)})
            assert r.status == 400                                   # only this style's candidates
            r = await client.post("/api/styles/3d_animated/use-reference", json={"file": str(cand)})
            d = (await r.json())["data"]
            assert d["crop"] == style_ref.default_crop(768, 1152) and d["crop_url"].startswith("/files/style_crops/")
            assert d["style"]["reference_crop_url"] and d["style"]["reference_url"]
            r = await client.get(d["crop_url"])
            assert r.status == 200
        finally:
            await client.close()
    asyncio.run(go())
