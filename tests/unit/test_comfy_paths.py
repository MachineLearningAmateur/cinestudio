"""ComfyUI folder rules for a Linux install (Codex review 2026-10-22)."""
from pathlib import Path

from cinestudio import comfy, settings


def test_io_dir_follows_main_py(monkeypatch, tmp_path):
    # COMFYUI_DIR = the folder WITH main.py (a normal Linux install): no extra ComfyUI level
    monkeypatch.setattr(comfy, "load_config", lambda: {"cwd": str(tmp_path / "ComfyUI"), "main": "main.py", "args": []})
    assert comfy.io_dir("output") == tmp_path / "ComfyUI" / "output"
    # Comfy Desktop without its flags: cwd/ComfyUI/main.py
    monkeypatch.setattr(comfy, "load_config", lambda: {"cwd": str(tmp_path), "main": "ComfyUI/main.py", "args": []})
    assert comfy.io_dir("input") == tmp_path / "ComfyUI" / "input"
    # the saved flag wins
    monkeypatch.setattr(comfy, "load_config", lambda: {"cwd": str(tmp_path), "main": "main.py",
                                                       "args": ["--input-directory", str(tmp_path / "in")]})
    assert comfy.io_dir("input") == tmp_path / "in"


def test_models_folder_of_the_settings_reaches_comfyui(monkeypatch, tmp_path):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    cfg = {"cwd": str(tmp_path / "ComfyUI"), "main": "main.py", "args": ["--enable-manager"]}
    monkeypatch.setattr(settings, "get", lambda: {"models_dir": None})
    assert comfy.model_path_args(cfg) == []
    vol = tmp_path / "workspace" / "models"
    monkeypatch.setattr(settings, "get", lambda: {"models_dir": str(vol)})
    args = comfy.model_path_args(cfg)
    assert args[0] == "--extra-model-paths-config"
    text = Path(args[1]).read_text(encoding="utf-8")
    assert f"base_path: {vol.as_posix()}" in text and "diffusion_models: diffusion_models/" in text
    assert args == comfy.base_args(cfg)[-2:]
    # Comfy Desktop has its own paths file; ComfyUI's own models folder needs nothing
    assert comfy.model_path_args({**cfg, "args": ["--extra-model-paths-config", "x.yaml"]}) == []
    monkeypatch.setattr(settings, "get", lambda: {"models_dir": str(tmp_path / "ComfyUI" / "models")})
    assert comfy.model_path_args(cfg) == []
