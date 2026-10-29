"""Installer failure gates: no GPU, downloads, services or Blender required."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_linux", ROOT / "setup/check_linux.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def report():
    return {"ok": False, "checks": {
        "blender": {"ok": True}, "ffmpeg": {"ok": True}, "models": {"ok": True},
        "model_tier": {"ok": True, "ready": True}, "tokens": {"ok": True},
        "comfyui": {"ok": False}, "proxy": {"ok": False},
    }}


def test_only_offline_services_are_expected():
    r = report()
    assert check.assess_doctor(r)["ok"]
    assert "skipped" not in r["checks"]["comfyui"]  # no mutation of the original diagnostic


@pytest.mark.parametrize("name", ["blender", "ffmpeg", "models", "tokens", "future_check"])
def test_real_failures_are_never_swallowed(name):
    r = report()
    r["checks"][name] = {"ok": False, "error": "broken"}
    result = check.assess_doctor(r)
    assert not result["ok"] and name in result["failed"]


def test_incomplete_tier_fails_even_when_video_models_are_ready():
    r = report()
    r["checks"]["model_tier"]["ready"] = False
    assert check.assess_doctor(r)["failed"] == ["model_tier"]
    assert check.assess_doctor(r, skip_models=True)["ok"]
    r["checks"]["blender"]["ok"] = False
    assert check.assess_doctor(r, skip_models=True)["failed"] == ["blender"]


def test_missing_checks_cannot_look_like_success():
    with pytest.raises(KeyError):
        check.assess_doctor({"ok": True})
    assert "blender" in check.assess_doctor({"checks": {}})["failed"]


def fake_torch(monkeypatch, version="2.12.1+cu130", cuda="13.0", available=True):
    module = types.SimpleNamespace(__version__=version, version=types.SimpleNamespace(cuda=cuda),
                                  cuda=types.SimpleNamespace(is_available=lambda: available,
                                                            get_device_name=lambda n: "test GPU"))
    monkeypatch.setitem(sys.modules, "torch", module)
    monkeypatch.setitem(sys.modules, "torchvision", types.ModuleType("torchvision"))
    monkeypatch.setitem(sys.modules, "torchaudio", types.ModuleType("torchaudio"))


def test_cuda_checks_the_final_build_and_gpu(monkeypatch):
    fake_torch(monkeypatch)
    assert check.check_cuda("2.12.1", "cu130")["gpu"] == "test GPU"
    fake_torch(monkeypatch, available=False)
    with pytest.raises(RuntimeError, match="cannot use CUDA"):
        check.check_cuda("2.12.1", "cu130")
    fake_torch(monkeypatch, cuda=None)
    with pytest.raises(RuntimeError, match="CUDA 13.0 wheels"):
        check.check_cuda("2.12.1", "cu130")
    fake_torch(monkeypatch, version="2.14.1+cu130")
    with pytest.raises(RuntimeError, match="expected torch"):
        check.check_cuda("2.12.1", "cu130")


def test_an_import_failure_is_a_nonzero_json_result(monkeypatch, capsys):
    fake_torch(monkeypatch)
    monkeypatch.setitem(sys.modules, "torchaudio", None)
    assert check.main(["cuda", "--torch-version", "2.12.1", "--cuda", "cu130"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert not result["ok"] and "torchaudio" in result["error"]


def test_locks_exclude_a_second_process_and_release(tmp_path):
    assert check.check_locks(tmp_path)["release"]
    assert not list(tmp_path.iterdir())


def test_a_filesystem_that_ignores_locks_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(check.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stderr=""))
    with pytest.raises(RuntimeError, match="do not exclude"):
        check.check_locks(tmp_path)


def test_render_refuses_a_busy_queue_without_starting_blender(tmp_path, monkeypatch):
    from cinestudio import blender, server, workqueue
    from contextlib import nullcontext
    monkeypatch.setattr(server, "gpu_lock", lambda **kw: nullcontext())
    monkeypatch.setattr(workqueue, "tickets", lambda: [{"kind": "run"}])
    monkeypatch.setattr(blender, "run_job", lambda *a, **kw: pytest.fail("must not render"))
    with pytest.raises(RuntimeError, match="queued or running"):
        check.check_render()


@pytest.mark.parametrize("python", ["3.12", "3.13"])
@pytest.mark.parametrize("cuda", ["cu128", "cu130"])
def test_all_shipped_dependency_locks_are_current_and_hash_pinned(python, cuda):
    check.check_pins(python, cuda)
    runtime = json.loads((ROOT / "setup/runtime.lock.json").read_text())
    engine = ROOT / f"setup/linux/comfy-py{python}-{cuda}.txt"
    assert f"torch=={runtime['torch']['linux_versions'][cuda]}+{cuda}" in engine.read_text()
    # A requirements-file index resets pip's command-line extra index, hiding CUDA wheels.
    assert "--index-url " not in engine.read_text()
    for path in (ROOT / f"setup/linux/app-py{python}.txt", engine, ROOT / "setup/linux/sageattention.txt"):
        blocks = path.read_text().replace("\\\n", " ").splitlines()
        for block in blocks:
            if not block or block.startswith(("#", "--")):
                continue
            assert re.match(r"^[a-zA-Z0-9_.-]+==[^ ;]+", block), block
            assert re.search(r"--hash=sha256:[a-f0-9]{64}", block), block


def test_changed_inputs_force_lock_regeneration(tmp_path, monkeypatch):
    setup = tmp_path / "setup"
    (setup / "linux").mkdir(parents=True)
    (setup / "requirements.txt").write_text("changed")
    (setup / "linux/manifest.json").write_text(json.dumps({
        "setup/requirements.txt": hashlib.sha256(b"original").hexdigest(),
    }))
    monkeypatch.setattr(check, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="lock is stale"):
        check.check_pins("3.13", "cu130")
