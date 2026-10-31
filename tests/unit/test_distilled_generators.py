"""The supported graph generators do not need retired dev graphs or model weights."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / "workflows"


def _load(script):
    spec = importlib.util.spec_from_file_location(script, ROOT / "setup" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("variant,ingredients", [("layout", False), ("layouting", True)])
def test_layout_generation_needs_only_distilled_source(tmp_path, monkeypatch, variant, ingredients):
    for suffix in ("_api.json", ".bind.json"):
        shutil.copy2(WORKFLOWS / f"ltx25_v2v{suffix}", tmp_path)
    generator = _load("make_layout_graphs")
    monkeypatch.setattr(generator, "W", tmp_path)
    graph, bind = generator.build(ingredients)
    assert graph == json.loads((WORKFLOWS / f"ltx25_{variant}_api.json").read_text(encoding="utf-8"))
    assert bind == json.loads((WORKFLOWS / f"ltx25_{variant}.bind.json").read_text(encoding="utf-8"))
    assert not list(tmp_path.glob("*dev*"))


@pytest.mark.parametrize("script,source,variant", [
    ("make_redraw_graphs", "restyle", "redraw"),
    ("make_txt2img_graphs", "redraw", "txt2img"),
])
def test_image_graph_generation_needs_only_distilled_source(tmp_path, monkeypatch, script, source, variant):
    for suffix in ("_api.json", ".bind.json"):
        shutil.copy2(WORKFLOWS / f"ltx25_{source}{suffix}", tmp_path)
    generator = _load(script)
    monkeypatch.setattr(generator, "WF", tmp_path)
    graph, bind = generator.build("ltx25")
    assert graph == json.loads((WORKFLOWS / f"ltx25_{variant}_api.json").read_text(encoding="utf-8"))
    assert bind == json.loads((WORKFLOWS / f"ltx25_{variant}.bind.json").read_text(encoding="utf-8"))
    assert not list(tmp_path.glob("*dev*"))
