"""The Mannequin view (phase 1, 2026-10-24): Blender previews of a character's own mannequin, their cache, the height
reuse, the one Save and the editor's schema. No Blender: blender.run_job is replaced by a fake that writes the files."""
import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from cinestudio import blender, character, jobs, mannequin_preview, webapp


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(jobs.spawn, "detached", lambda *a, **k: (0, True))
    shot = tmp_path / "template.blend"
    shot.write_bytes(b"BLENDER")
    monkeypatch.setattr(character, "default_pose_shot", lambda: shot)
    runs = []

    def fake_run_job(script, blend, args, log, timeout=None):     # what bpy_jobs/mannequin_preview.py writes
        assert script == "mannequin_preview.py"
        a = dict(zip(args[::2], args[1::2]))
        body = json.loads(Path(a["--body"]).read_text())
        out = Path(a["--out"])
        out.mkdir(parents=True)
        for v in ("front", "right", "back"):
            cv2.imwrite(str(out / f"{v}.png"), np.zeros((640, 384, 4), np.uint8))
        h = 1.8057 * float(body.get("scale", 1.0))
        (out / "info.json").write_text(json.dumps({"height": round(h, 4), "ref": 1.8057, "frame_m": 2.2,
                                                    "size": [384, 640], "feet_row": 625.5,
                                                    "views": ["front", "right", "back"]}))
        Path(log).write_text("ok")
        runs.append(body)
    monkeypatch.setattr(blender, "run_job", fake_run_job)
    d = character.library() / "kit"
    d.mkdir(parents=True)
    front = np.zeros((600, 300, 3), np.uint8)
    front[60:560, 100:200] = (40, 200, 90)
    cv2.imwrite(str(d / "front.png"), front)
    (d / "character.json").write_text(json.dumps({"name": "kit", "display_name": "Kit", "status": "ready",
                                                  "source": "text", "files": {"front": str(d / "front.png")},
                                                  "proxy": {"mask": {"shape": "round"}}}))
    return runs


def call(fn):
    async def run():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as client:
            return await fn(client)
    return asyncio.run(run())


def test_a_preview_is_rendered_once_then_served_from_the_cache(home):
    r = mannequin_preview.preview("kit", {"scale": 0.72}, {"mask": {"shape": "round"}})
    assert r["height_m"] == pytest.approx(1.3001, abs=1e-3) and r["ratio"] == pytest.approx(0.72, abs=1e-3)
    assert not r["cached"] and set(r["files"]) == {"front", "right", "back", "character"}
    pic = cv2.imread(r["files"]["character"], cv2.IMREAD_UNCHANGED)
    assert pic.shape == (640, 384, 4) and pic[..., 3].max() == 255      # the character at the same scale
    ys = np.where(pic[..., 3].max(1) > 0)[0]
    assert abs(ys.max() + 1 - 625.5) <= 2                              # its feet on the renders' feet row
    assert abs((ys.max() - ys.min() + 1) - 1.3001 * 640 / 2.2) <= 4     # its height = the body's height
    r2 = mannequin_preview.preview("kit", {"scale": 0.72, "head": 1.0}, {"mask": {"shape": "round"}})
    assert r2["cached"] and r2["key"] == r["key"] and len(home) == 1    # 1.0 values do not change the key
    mannequin_preview.preview("kit", {"scale": 0.72}, {})               # another kit: another render
    assert len(home) == 2
    with pytest.raises(character.CharacterError):
        mannequin_preview.preview("kit", {"tail": 2}, {})
    with pytest.raises(character.CharacterError):
        mannequin_preview.preview("kit", {}, {"mask": True})


def test_the_height_of_a_previewed_body_is_reused_without_blender(home, monkeypatch):
    mannequin_preview.preview("kit", {"scale": 0.8}, {})
    monkeypatch.setattr(blender, "run_job", lambda *a, **k: pytest.fail("no Blender for a cached height"))
    h = character.measure_height({"scale": 0.8})
    assert h["ratio"] == pytest.approx(0.8, abs=1e-3) and h["body"] == {"scale": 0.8}


def test_a_changed_script_gives_a_new_key(home, monkeypatch):
    k1 = mannequin_preview.key({"scale": 0.8}, {})
    monkeypatch.setattr(mannequin_preview, "_versions", lambda: "another")
    assert mannequin_preview.key({"scale": 0.8}, {}) != k1


def test_the_editor_endpoints(home, monkeypatch):
    monkeypatch.setattr(character, "build_sheet", lambda name, c=None, *a, **k: c)   # no views in this fixture

    async def go(c):
        s = (await (await c.get("/api/mannequin/schema")).json())["data"]
        assert [f["key"] for f in s["body"]] == ["scale", "head", "neck", "torso", "legs", "arms"]
        assert {p["id"] for p in s["presets"]} >= {"adult", "chibi"} and s["limits"] == [0.3, 2.0]
        assert "kitsune" in s["mark_presets"]
        r = await c.post("/api/characters/kit/mannequin-preview", json={"body": json.dumps({"scale": 0.9})})
        d = (await r.json())["data"]
        assert d["ratio"] == pytest.approx(0.9, abs=1e-3) and d["urls"]["front"].startswith("/files/mannequin/")
        assert (await c.get(d["urls"]["front"])).status == 200
        assert (await c.post("/api/characters/kit/mannequin-preview", json={"body": '{"scale": 9}'})).status == 400
        r = await c.post("/api/characters/kit/stand-in", json={"body": json.dumps({"scale": 0.9})})
        ch = (await r.json())["data"]["character"]
        assert ch["mannequin"] == {"scale": 0.9} and ch["proxy"] == {"mask": {"shape": "round"}}   # the kit stays
        assert ch["height"]["ratio"] == pytest.approx(0.9, abs=1e-3) and ch["missing"] == []
    call(go)


def test_the_rules_file_gives_the_body_math():
    """bpy_jobs/mannequin_rules.json is the one source of the body math (mannequin_body.py, the live view later)."""
    rules = json.loads((Path(mannequin_preview.ROOT) / "bpy_jobs" / "mannequin_rules.json").read_text())

    def scales(body):
        out = {}
        for bone, r in rules["bones"].items():
            f = 1.0
            for k in r.get("mul", []):
                f *= body.get(k, 1.0)
            for k in r.get("div", []):
                f /= body.get(k, 1.0)
            out[bone] = f
        return out
    s = scales({"head": 1.55, "neck": 0.35, "torso": 0.8, "legs": 0.7, "arms": 0.85})
    assert s["spine"] == pytest.approx(0.8) and s["neck"] == pytest.approx(0.35 / 0.8)
    assert s["head"] == pytest.approx(1.55 / 0.35) and s["thigh.L"] == pytest.approx(0.7)
    assert s["upper_arm.R"] == pytest.approx(0.85 / 0.8)
    assert tuple(rules["keys"]) == character.MANNEQUIN_KEYS and tuple(rules["limits"]) == character.MANNEQUIN_SCALE
