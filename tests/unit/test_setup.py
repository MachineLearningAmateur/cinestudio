"""Quick setup: tokens (never shown), settings, model tiers, the verified download, frozen models (no network)."""

import asyncio
import hashlib
import json
import threading

import pytest
from aiohttp.test_utils import TestClient, TestServer

from cinestudio import models, settings, webapp
from cinestudio.workflow import freeze_models

SECRET = "hf_abcdefghijklmnopqrstuvwxyz123456"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("COMFY_API_KEY", raising=False)
    return tmp_path


def test_tokens_are_saved_but_never_reported(home, monkeypatch):
    assert settings.status()["hf"]["set"] is False
    settings.set_token("hf", "  " + SECRET + "\n")
    st = settings.status()
    assert st["hf"] == {**st["hf"], "set": True, "source": "saved"}
    assert SECRET not in json.dumps(st)
    assert settings.token("hf") == SECRET
    monkeypatch.setenv("HF_TOKEN", "hf_fromtheenvironment")                 # the environment wins
    assert settings.token("hf") == "hf_fromtheenvironment" and settings.status()["hf"]["source"] == "environment"
    monkeypatch.delenv("HF_TOKEN")
    settings.clear_token("hf")
    assert settings.token("hf") is None and not settings.status()["hf"]["set"]


@pytest.mark.parametrize("value,msg", [("", "empty"), ("abc def", "spaces"), ("xyz123", "starts with 'hf_'")])
def test_bad_tokens_are_refused(home, value, msg):
    with pytest.raises(settings.SettingsError, match=msg):
        settings.set_token("hf", value)


def test_cloud_needs_the_comfy_key_and_falls_back_when_removed(home):
    with pytest.raises(settings.SettingsError, match="Comfy Cloud API key"):
        settings.set_backend("cloud")
    settings.set_token("comfy", "comfyui-0123456789abcdef")
    assert settings.set_backend("cloud")["backend"] == "cloud"
    settings.clear_token("comfy")
    assert settings.get()["backend"] == "local"


def test_video_tiers_use_distilled_int8_with_pinned_files():
    for tier in models.tier_names():
        bundle = models.bundle(tier)
        for slot in {s for f in models.features().values() for s in f}:
            entry = bundle[slot]
            assert entry["size"] > 0 and len(entry["sha256"]) == 64 and "/resolve/" in entry["url"]
        assert "int8" in bundle["ltx25_distilled"]["file"]
        assert "int8" in bundle["ltx25_text"]["file"]
        assert "int8" in bundle["qwen_edit_model"]["file"]
        assert "ltx25_model" not in bundle
        assert "ltx25_distilled_lora" not in bundle
    assert "full_stage2" not in models.features()


def _fake_tier(monkeypatch, tmp_path, content: bytes):
    manifest = {"model_tiers": {"features": {"video": ["m"]}, "common": {},
                "tiers": {"t1": {"label": "T", "vram_gb": 8, "ram_gb": 32, "about": "",
                                 "slots": {"m": {"dir": "checkpoints", "file": "m.safetensors",
                                                 "url": "https://huggingface.co/x/y/resolve/abc/m.safetensors",
                                                 "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}}}}}}
    p = tmp_path / "models.json"
    p.write_text(json.dumps(manifest))
    monkeypatch.setattr(models, "MANIFEST", p)
    md = tmp_path / "models"
    md.mkdir()
    settings.set_models_dir(str(md))
    return md


def test_download_verifies_and_renames(home, monkeypatch):
    data = b"model-bytes" * 1000
    md = _fake_tier(monkeypatch, home, data)
    assert models.status("t1")["files"][0]["state"] == "missing"
    seen = {}

    def fake_curl(url, part, token, stop):
        seen["token"] = token
        part.write_bytes(data)
        return 200
    monkeypatch.setattr(models, "_curl_one", fake_curl)
    settings.set_token("hf", SECRET)
    r = models.download("t1")
    assert r["ok"] and (md / "checkpoints" / "m.safetensors").read_bytes() == data
    assert not (md / "checkpoints" / "m.safetensors.part").exists()
    assert seen["token"] == SECRET                      # the token goes to curl for huggingface.co
    st = models.status("t1")
    assert st["ready"] and st["missing_bytes"] == 0


def test_download_rejects_a_bad_hash_and_reports_auth(home, monkeypatch):
    data = b"x" * 100
    md = _fake_tier(monkeypatch, home, data)

    def wrong(url, part, token, stop):
        part.write_bytes(b"y" * 100)                    # right size, wrong content
        return 200
    monkeypatch.setattr(models, "_curl_one", wrong)
    r = models.download("t1")
    assert not r["ok"] and r["failed"][0]["error_type"] == "hash"
    assert not (md / "checkpoints" / "m.safetensors.part").exists()        # a bad full copy is removed
    monkeypatch.setattr(models, "_curl_one", lambda url, part, token, stop: 401)
    r = models.download("t1")
    assert r["failed"][0]["error_type"] == "auth" and "token" in r["failed"][0]["error"]


def test_download_refuses_without_disk_space(home, monkeypatch):
    _fake_tier(monkeypatch, home, b"z" * 10)
    monkeypatch.setattr(models, "DISK_MARGIN", 10 ** 18)
    with pytest.raises(models.ModelsError, match="disk space"):
        models.download("t1", stop=threading.Event())


def test_freeze_models_writes_every_graph_with_model_slots(tmp_path):
    """The recipe chooses the sampling, the tier the files: distilled graphs take the tier too (Codex review
    2026-10-22); a full graph needs every slot, another graph keeps its file for a slot the tier lacks."""
    def graph(name, recipe):
        (tmp_path / f"{name}_api.json").write_text(json.dumps({"ckpt": {"class_type": "CheckpointLoaderSimple",
                                                                       "inputs": {"ckpt_name": "old.safetensors"}}}))
        (tmp_path / f"{name}.bind.json").write_text(json.dumps({"capabilities": {"recipe": recipe},
                                                                "models": {"ltx_model": [["ckpt", "ckpt_name"]]}}))
    graph("p_full", "full")
    graph("p_distilled", "distilled")
    files = {"ltx_model": {"dir": "checkpoints", "file": "dev.safetensors", "size": 1, "sha256": "a" * 64,
                           "url": "u"}}
    used = freeze_models(tmp_path, files)
    assert used == {"ltx_model": {"dir": "checkpoints", "file": "dev.safetensors", "size": 1, "sha256": "a" * 64}}
    assert json.loads((tmp_path / "p_full_api.json").read_text())["ckpt"]["inputs"]["ckpt_name"] == "dev.safetensors"
    assert json.loads((tmp_path / "p_distilled_api.json").read_text())["ckpt"]["inputs"]["ckpt_name"] == "dev.safetensors"
    with pytest.raises(ValueError, match="no file for slot"):
        freeze_models(tmp_path, {})
    (tmp_path / "p_full.bind.json").unlink(); (tmp_path / "p_full_api.json").unlink()
    assert freeze_models(tmp_path, {}) == {}                       # the distilled graph keeps its own file


def test_setup_api_never_returns_a_token(home, monkeypatch):
    monkeypatch.setattr(settings, "check_token", lambda name, timeout=15: {"ok": True, "detail": "works"})
    _fake_tier(monkeypatch, home, b"q" * 10)

    async def go():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as c:
            r = await c.post("/api/setup/token", json={"name": "hf", "value": SECRET})
            body = await r.text()
            assert r.status == 200 and SECRET not in body and json.loads(body)["data"]["check"]["ok"]
            r = await c.get("/api/setup")
            body = await r.text()
            d = json.loads(body)["data"]
            assert SECRET not in body and d["tokens"]["hf"]["set"] and d["tiers"][0]["tier"] == "t1"
            r = await c.post("/api/setup/tier", json={"tier": "t1"})
            assert r.status == 400                                          # not downloaded yet
            r = await c.post("/api/setup/token/clear", json={"name": "hf"})
            assert (await r.json())["data"]["set"] is False
    asyncio.run(go())


def test_download_state_survives_parallel_writers(home):
    import threading
    errors = []

    def w(i):
        try:
            for k in range(50):
                models._write_state({"state": "running", "i": i, "k": k})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    ts = [threading.Thread(target=w, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors and models.download_state()["state"] in ("running", "stopped")


def test_the_old_24gb_tier_name_still_works(tmp_path, monkeypatch):
    """The 24gb tier was renamed "large" (2026-10-24: its bf16 files need an 80 GB+ GPU). Old settings and runs that
    name "24gb" get the large tier."""
    from cinestudio import models, settings
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    assert "24gb" not in models.tier_names() and "large" in models.tier_names()
    assert models.bundle("24gb") == models.bundle("large")
    assert models.graph_settings("24gb") == models.graph_settings("large")
    assert settings.set_tier("24gb")["model_tier"] == "large"


def test_a_saved_24gb_setting_reads_as_large(tmp_path, monkeypatch):
    """A settings.json written before the rename holds "24gb": the Settings page compares it with the tier list
    ("In use") and new runs store it, so get() gives the current name."""
    import json
    from cinestudio import settings
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    (tmp_path / "settings.json").write_text(json.dumps({"model_tier": "24gb"}), encoding="utf-8")
    assert settings.get()["model_tier"] == "large"


def test_a_fork_keeps_its_parents_node_settings(tmp_path):
    """A fork wrote today's tier settings into its graphs; it now writes the parent's ({} for a run from before)."""
    import json
    import shutil
    from cinestudio import runner
    from cinestudio.workflow import WORKFLOWS
    for name in ("ltx25_layout_api.json", "ltx25_layout.bind.json"):
        shutil.copy(WORKFLOWS / name, tmp_path / name)
    out = runner._freeze_tier(tmp_path, "large", {}, settings={})
    g = json.loads((tmp_path / "ltx25_layout_api.json").read_text(encoding="utf-8"))
    assert "settings" not in out and all(n["inputs"]["tile_size"] == 512 for n in g.values()
                                         if n["class_type"] == "VAEDecodeTiled")
    out = runner._freeze_tier(tmp_path, "8gb", {}, settings={"VAEDecodeTiled": {"tile_size": 768}})
    g = json.loads((tmp_path / "ltx25_layout_api.json").read_text(encoding="utf-8"))
    assert out["settings"]["VAEDecodeTiled"]["tile_size"] == 768
    assert out["tier"] == "8gb" and runner._freeze_tier(tmp_path, "24gb", {}, settings={})["tier"] == "large"


def test_every_pinned_model_file_has_a_real_sha256():
    """The distilled LTX-2.5 file had a placeholder hash (64 '*'): a new install failed its check of the default video
    model. Every pinned entry must have 64 hex characters."""
    import json
    import re
    from pathlib import Path
    data = json.loads((Path(__file__).resolve().parents[2] / "setup" / "models.json").read_text(encoding="utf-8"))
    bad = []

    def walk(x, where):
        if isinstance(x, dict):
            if "sha256" in x and not re.fullmatch(r"[0-9a-f]{64}", str(x["sha256"])):
                bad.append(where)
            for k, v in x.items():
                walk(v, f"{where}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{where}[{i}]")
    walk(data, "models")
    assert not bad, bad
