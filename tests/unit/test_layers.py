"""Template layers (2026-10-24): Mannequin* = characters, Props, Set; their words in shot.json go into the prompts.
No Blender: shots.scan_layers is replaced by a fake scan."""
import json

import pytest

from cinestudio import runner, shots

SCAN = {"characters": [{"collection": "Mannequin", "rig": "MannequinRig", "objects": 107}],
        "props": [{"name": f"Ball_{k}", "animated": True} for k in ("Dodge1", "Dodge2", "Kick1", "Kick2")],
        "set": ["Floor", "BackWall", "Hoop", "BallCart"], "loose": [], "camera_moves": False, "frames": [1, 276]}


@pytest.fixture
def tpl(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    d = shots.shots_dir() / "gym"
    d.mkdir(parents=True)
    (d / "shot.blend").write_bytes(b"BLENDER")
    (d / "shot.json").write_text(json.dumps({"name": "gym", "kind": "blender", "status": "done",
                                             "files": {"blend": str(d / "shot.blend")}}), encoding="utf-8")
    calls = []
    monkeypatch.setattr(shots, "scan_layers", lambda blend: calls.append(blend) or json.loads(json.dumps(SCAN)))
    return calls


def test_layers_are_scanned_once_and_described(tpl):
    r = shots.layers("gym", set_description="a school gym with a ball cart at the far wall",
                     props={"Ball_*": "plain red rubber dodgeballs"})
    lay = r["layers"]
    assert lay["descriptions"]["set"].startswith("a school gym") and lay["warnings"] == []
    assert lay["descriptions"]["props"] == {"Ball_*": "plain red rubber dodgeballs"}
    shots.layers("gym", props={"Ball_Kick1": "a plain red rubber dodgeball"})
    assert len(tpl) == 1                                              # the scan is kept; --rescan reads it again
    rec = shots.load("gym")
    assert shots.props_sentence(rec["layers"]) == ("In the scene: plain red rubber dodgeballs; "
                                                    "a plain red rubber dodgeball.")
    shots.layers("gym", props={"Ball_Kick1": ""})                    # an empty text removes it
    assert "Ball_Kick1" not in shots.load("gym")["layers"]["descriptions"]["props"]
    with pytest.raises(shots.ShotError, match="no prop matches"):
        shots.layers("gym", props={"Hoop": "a hoop"})                 # the hoop is Set, not a prop


def test_layer_warnings_name_what_is_missing():
    w = shots.layer_warnings({"characters": [], "props": [], "set": [], "loose": ["Cube", "Lamp"],
                              "camera_moves": True})
    text = " | ".join(w)
    assert "no character layer" in text and "Cube, Lamp" in text and "no Set collection" in text
    assert "camera moves" in text
    assert shots.layer_warnings(SCAN) == []


def _m(video, layers):
    return {"render_mode": "layout", "prompt": {"character": "on black: a masked dancer", "style": "3D animation",
                                                "video": video}, "prompt_format": "ing25",
            "layers": layers, "sound_text": ""}


def test_the_prop_words_reach_the_clip_prompt_once():
    lay = {"descriptions": {"set": "a gym", "props": {"Ball_*": "plain red rubber dodgeballs"}}}
    p = runner.video_prompt(_m("The dancer dodges two throws and kicks two balls away.", lay))
    assert "In the scene: plain red rubber dodgeballs." in p
    p2 = runner.video_prompt(_m("The dancer dodges plain red rubber dodgeballs.", lay))
    assert "In the scene" not in p2                                   # the video text says it already
    assert "In the scene" not in runner.video_prompt(_m("x", None))   # a template without layers: as before


def test_every_common_frame_shape_has_an_edit_layout_the_encoder_keeps():
    """Fable review: edit_size refused 16:10, 4:3, 21:9, 2:1 and 5:4 frames. edit_layout pads them instead; the size
    sent to Qwen is one the encoder keeps, and the frame's box has the frame's exact shape."""
    from cinestudio import background as b
    for cam in [(1920, 1200), (1440, 1080), (2560, 1080), (1920, 960), (1280, 1024), (1920, 1080), (1536, 896),
                (1080, 1920), (1024, 1024)]:
        w, h = b.plate_size({"resolution": list(cam)})
        ew, eh, (x0, y0, iw, ih) = b.edit_layout(w, h)
        assert ew % 16 == 0 and eh % 16 == 0 and b._stable(ew, eh), cam
        assert 0 <= x0 and 0 <= y0 and x0 + iw <= ew and y0 + ih <= eh
        assert abs(iw / ih - w / h) / (w / h) < 0.002, cam                  # no stretch
        assert 1 - iw * ih / (ew * eh) < 0.03, cam                          # at most a thin padded band
    assert b.edit_layout(1920, 1120) == (1344, 784, (0, 0, 1344, 784))     # our 768x448 cameras: exact


def test_the_kontext_crop_is_removed_from_the_real_edit_graph():
    from cinestudio import background as b
    from cinestudio.workflow import ProfileWorkflow
    wf = ProfileWorkflow("qwenedit", "edit")
    g = wf.build({"instruction": "x", "seed": 1}, files={wf.targets("image")[0]: "set.png"})
    b._no_kontext_crop(g)
    assert "scale" not in g and "ref" not in g and "image2" not in g["pos_txt"]["inputs"]
    for n in g.values():
        assert not any(isinstance(v, list) and v[:1] == ["scale"] for v in n["inputs"].values())
    g2 = wf.build({"instruction": "x", "seed": 1}, files={wf.targets("image")[0]: "set.png"})
    g2["other"] = {"class_type": "X", "inputs": {"image": ["scale", 0]}}   # a future graph change
    with pytest.raises(b.BackgroundError, match="still uses the Kontext scale node"):
        b._no_kontext_crop(g2)


def test_a_built_in_templates_layers_are_not_written(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    lib = tmp_path / "library" / "dance"
    lib.mkdir(parents=True)
    (lib / "shot.blend").write_bytes(b"BLENDER")
    (lib / "shot.json").write_text(json.dumps({"name": "dance", "kind": "blender", "files": {"blend": "shot.blend"}}),
                                   encoding="utf-8")
    monkeypatch.setattr(shots, "BUILTIN", tmp_path / "library")
    with pytest.raises(shots.ShotError, match="built-in template"):
        shots.layers("dance", set_description="a gym")
    assert "layers" not in json.loads((lib / "shot.json").read_text(encoding="utf-8"))   # the repo file is untouched


def test_props_words_dedupe_and_stale_keys(tpl):
    shots.layers("gym", props={"Ball_*": "plain red rubber dodgeballs."})
    shots.layers("gym", props={"Ball_Kick1": "plain red rubber dodgeballs"})
    assert shots.props_sentence(shots.load("gym")["layers"]) == "In the scene: plain red rubber dodgeballs."
    d = shots.shots_dir() / "gym" / "shot.json"
    data = json.loads(d.read_text(encoding="utf-8"))
    data["layers"]["descriptions"]["props"]["Old_*"] = "old words"          # a key from before a rescan
    d.write_text(json.dumps(data), encoding="utf-8")
    r = shots.layers("gym", props={"Old_*": ""})                             # can be removed
    assert "Old_*" not in r["layers"]["descriptions"]["props"]


def test_a_saved_backplate_names_its_place_in_the_clip_text():
    m = _m("The dancer dodges.", None)
    m.update(background="plate", background_text="a school gym with a ball cart")
    assert "The place: a school gym with a ball cart." in runner.video_prompt(m)


def test_a_slightly_zoomed_and_shifted_backplate_is_aligned_back_onto_the_set():
    import cv2
    import numpy as np
    from cinestudio import background as b
    rng = np.random.default_rng(3)
    ref = np.full((448, 768, 3), 90, np.uint8)
    for _ in range(25):                                               # boxes and lines like a clay set
        x, y = int(rng.integers(0, 700)), int(rng.integers(0, 400))
        cv2.rectangle(ref, (x, y), (x + int(rng.integers(20, 120)), y + int(rng.integers(20, 90))),
                      tuple(int(v) for v in rng.integers(30, 230, 3)), -1)
    m = np.float32([[0.97, 0, 14], [0, 0.96, 22]])                     # Qwen's drift: smaller and shifted
    drifted = cv2.warpAffine(ref, m, (768, 448), borderMode=cv2.BORDER_REPLICATE)
    out, info = b.align_to_set(ref, drifted)
    assert info["applied"] and info["fit_after"] > info["fit_before"] + 0.1
    assert abs(info["scale"][1] - 0.96) < 0.01
    same, info2 = b.align_to_set(ref, ref.copy())                     # already on the set: unchanged
    assert not info2["applied"] and np.array_equal(same, ref)


def test_the_api_lists_and_edits_a_templates_clip_texts(tpl):
    import asyncio

    from aiohttp.test_utils import TestClient, TestServer

    from cinestudio import webapp
    rec_path = shots.shots_dir() / "gym" / "shot.json"
    rec = json.loads(rec_path.read_text())
    rec["frames"] = 276
    rec_path.write_text(json.dumps(rec))

    async def go():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as c:
            r = await c.post("/api/templates/gym/clip-texts",
                             json={"clip_texts": {"1": "{character} dodges", "2": "it kicks"}, "context": "faces us"})
            d = (await r.json())["data"]
            assert d["clip_texts"] == {"1": "{character} dodges", "2": "it kicks"} and d["clip_context"] == "faces us"
            g = (await (await c.get("/api/templates/gym/clip-texts")).json())["data"]
            assert [x["text"] for x in g["clips"]] == ["{character} dodges", "it kicks", ""]
            s = next(x for x in (await (await c.get("/api/shots")).json())["data"] if x["name"] == "gym")
            assert s["clip_context"] == "faces us" and [x["first"] for x in s["clips"]] == [1, 97, 193]
            bad = await c.post("/api/templates/gym/clip-texts", json={"clip_texts": {"9": "x"}})
            assert bad.status >= 400
    asyncio.run(go())
