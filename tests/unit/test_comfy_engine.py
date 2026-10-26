"""The ComfyUI engine: identity by pid + creation time, a stop that never interrupts work."""

import pytest

from cinestudio import comfy, server


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("COMFYUI_URL", f"http://127.0.0.1:1/{tmp_path.name}")   # its own GPU lock
    return tmp_path


def test_a_reused_pid_is_not_our_engine(home, monkeypatch):
    comfy._save({"started_pid": 1234, "started_ctime": 1000.0})
    monkeypatch.setattr(comfy, "_ctime", lambda pid: 2000.0)          # same pid, another process
    assert not comfy.owns_engine()
    monkeypatch.setattr(comfy, "_ctime", lambda pid: 1000.2)
    assert comfy.owns_engine()


def test_stop_refuses_while_a_step_uses_comfyui(home, monkeypatch):
    monkeypatch.setattr(comfy, "owns_engine", lambda: True)
    monkeypatch.setattr(comfy, "_kill_own", lambda: pytest.fail("must not stop"))
    held = server.gpu_lock()
    held.acquire()
    try:
        with pytest.raises(comfy.EngineError, match="a step uses ComfyUI"):
            comfy.stop()
    finally:
        held.release()


def test_stop_refuses_an_engine_started_elsewhere(home, monkeypatch):
    monkeypatch.setattr(comfy, "owns_engine", lambda: False)
    with pytest.raises(comfy.EngineError, match="not started by cinestudio"):
        comfy.stop()


def test_autostart_off_everywhere(home, monkeypatch):
    monkeypatch.setattr(server, "alive", lambda url=None: False)
    monkeypatch.setenv("CINESTUDIO_AUTOSTART_COMFYUI", "0")
    monkeypatch.setattr(comfy, "start", lambda: pytest.fail("must not start"))
    with pytest.raises(comfy.EngineError, match="AUTOSTART"):
        comfy.ensure_running()
