import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("sdpose_backend", Path(__file__).resolve().parents[2] / "setup" / "sdpose_backend.py")
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)


def test_boxes_are_widened_to_the_models_shape():
    x0, y0, x1, y1 = sb.SDPose.model_box([100, 50, 160, 350])          # a tall figure: 60 x 300
    assert abs((x1 - x0) / (y1 - y0) - 0.75) < 1e-6
    assert abs((y1 - y0) - 300 * 1.25) < 1e-6 and abs((x0 + x1) / 2 - 130) < 1e-6


def test_padding_keeps_boxes_whole_at_the_image_edge():
    img = np.full((100, 200, 3), 7, np.uint8)
    out, boxes, (ox, oy) = sb.SDPose._pad(img, [[-20, -10, 50, 120]])
    assert (ox, oy) == (20, 10) and out.shape == (130, 220, 3)
    assert boxes == [[0, 0, 70, 130]] and (out[10:110, 20:220] == 7).all() and out[0, 0].sum() == 0
    same, b2, off = sb.SDPose._pad(img, [[10, 10, 50, 50]])
    assert same is img and off == (0, 0)


def test_bad_device_and_low_ram_are_refused(monkeypatch):
    with pytest.raises(ValueError, match="cpu' or 'cuda"):
        sb.SDPose("gpu0")
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("M", (), {"available": 2e9})())
    with pytest.raises(RuntimeError, match="free RAM"):
        sb.SDPose("cpu")


def test_fractional_edges_are_padded_and_restored():
    img = np.zeros((100, 200, 3), np.uint8)
    out, boxes, (ox, oy) = sb.SDPose._pad(img, [[-0.4, 10, 200.6, 50]])
    assert (ox, oy) == (1, 0) and out.shape[1] == 202 and boxes[0][0] >= 0 and boxes[0][2] <= out.shape[1]
