"""The panorama style pass (background.style_pass) with the GPU job mocked: it must finish, save a NEW panorama and
never overwrite one (Codex review 2026-10-22: a NameError after the GPU job; a 64-character name overwrote the source)."""
import cv2
import numpy as np
import pytest

from cinestudio import background, character


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    src = tmp_path / "pano.png"
    cv2.imwrite(str(src), np.full((1024, 2048, 3), 90, np.uint8))
    background.save(src, "a rooftop", 1, scene="a rooftop", name="roof")
    calls = []

    def job(profile, variant, values, files, out, tag, on_event=None):
        calls.append((profile, variant, values))
        cv2.imwrite(str(out), np.full((1024, 2304, 3), 140, np.uint8))
        return {}

    monkeypatch.setattr(character, "image_job", job)
    return calls


def test_the_style_pass_saves_a_new_panorama_with_the_redraw_graph(lib):
    r = background.style_pass("roof", "my.lora v2.safetensors", keep=0.3, strength=1.0, seed=7)
    assert r["name"] != "roof" and background.NAME_RE.match(r["name"])
    assert lib[0][1] == "redraw" and lib[0][2]["source_sigmas"].startswith("0.7,")
    assert cv2.imread(background.load("roof")["file"]).mean() == pytest.approx(90, abs=1)    # the source is kept
    r2 = background.style_pass("roof", "my.lora v2.safetensors", keep=0.3, strength=1.0, seed=7)
    assert r2["name"] != r["name"]                                                           # never the same folder


def test_an_existing_name_is_refused_before_the_gpu_job(lib):
    with pytest.raises(background.BackgroundError, match="exists already"):
        background.style_pass("roof", "x.safetensors", new_name="roof")
    with pytest.raises(background.BackgroundError):
        background.style_pass("roof", "x.safetensors", keep=0.99)
    assert lib == []


def test_a_failed_job_frees_the_reserved_name(lib, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("GPU lost")

    monkeypatch.setattr(character, "image_job", boom)
    with pytest.raises(RuntimeError):
        background.style_pass("roof", "x.safetensors", new_name="roof_try")
    assert not (background.library_dir() / "roof_try").exists()
