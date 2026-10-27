"""The pinned look image (look_pin.py, 2026-10-26): the soft figure mask, the graph change and the background gate."""
import json
from pathlib import Path

import cv2
import numpy as np

from cinestudio import look_pin


def _figure(h=448, w=768):
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (360, 80), (410, 380), 255, -1)                # a standing figure
    return m


def test_the_soft_mask_covers_the_figure_with_a_ring_and_a_shadow():
    fig = _figure() > 0
    soft = look_pin.soft_mask(fig, 448)
    assert soft[fig].min() == 1.0                                   # every figure pixel may change fully
    assert 0 < soft[230, 360 - int(0.03 * 448) - 10] < 1             # the ring is soft
    assert soft[5, 5] == 0                                          # the room is pinned
    assert 0.2 < soft[392, 385] <= 1.0                              # the contact shadow under the feet


def test_prepare_writes_the_frame_and_mask_at_the_edit_size(tmp_path):
    frame = np.full((448, 768, 3), 90, np.uint8)
    cv2.imwrite(str(tmp_path / "f.png"), frame)
    cv2.imwrite(str(tmp_path / "m.png"), _figure())
    r = look_pin.prepare(tmp_path / "f.png", tmp_path / "m.png", tmp_path / "out", "look")
    assert r["size"] == [1344, 784]
    assert cv2.imread(r["frame"]).shape == (784, 1344, 3) and cv2.imread(r["mask"], 0).shape == (784, 1344)


def test_the_graph_edit_pins_the_latent_and_drops_the_crop():
    g = json.loads((Path(__file__).resolve().parents[2] / "workflows" / "qwenedit_edit_api.json").read_text("utf-8"))
    look_pin.graph_edit("look_mask")(g)
    assert "scale" not in g and g["enc"]["inputs"]["pixels"] == ["img", 0]
    assert g["pos_txt"]["inputs"]["image1"] == ["img", 0] and g["neg_txt"]["inputs"]["image1"] == ["img", 0]
    assert g["ks"]["inputs"]["latent_image"] == ["look_noise", 0]
    assert g["look_noise"]["class_type"] == "SetLatentNoiseMask" and g["look_noise"]["inputs"]["samples"] == ["enc", 0]
    assert g["ks"]["inputs"]["model"] == ["look_dd", 0] and g["look_dd"]["inputs"]["model"] == ["cfg_tail", 0]
    assert g["look_mask"]["class_type"] == "LoadImage" and g["look_mask_m"]["inputs"]["channel"] == "red"


def test_the_gate_passes_a_kept_room_and_fails_a_lost_one(tmp_path):
    rng = np.random.default_rng(1)
    room = np.full((448, 768, 3), 70, np.uint8)
    for _ in range(40):                                             # signs, doors, lines: edges to keep
        x, y = rng.integers(0, 700), rng.integers(0, 400)
        cv2.rectangle(room, (int(x), int(y)), (int(x) + 50, int(y) + 30), (200, 200, 60), 2)
    cv2.imwrite(str(tmp_path / "frame.png"), room)
    cv2.imwrite(str(tmp_path / "mask.png"), _figure())
    kept = room.copy()
    cv2.rectangle(kept, (360, 80), (410, 380), (150, 40, 150), -1)  # only the figure changed
    cv2.imwrite(str(tmp_path / "kept.png"), cv2.resize(kept, (1344, 784)))
    lost = cv2.GaussianBlur(room, (0, 0), 9)                        # the room redrawn without its details
    cv2.imwrite(str(tmp_path / "lost.png"), cv2.resize(lost, (1344, 784)))
    ok = look_pin.check(tmp_path / "kept.png", tmp_path / "frame.png", tmp_path / "mask.png")
    bad = look_pin.check(tmp_path / "lost.png", tmp_path / "frame.png", tmp_path / "mask.png")
    assert ok["ok"], ok
    assert not bad["ok"] and bad["reasons"], bad


def test_offset_moves_the_look_without_resampling(tmp_path):
    img = np.random.default_rng(1).integers(0, 255, (784, 1344, 3), dtype=np.uint8)
    p = tmp_path / "look.png"
    cv2.imwrite(str(p), img)
    assert look_pin.offset(p) == (6, 4)                      # the move that measured bright at the edit size
    moved = cv2.imread(str(p))
    assert moved.shape == img.shape
    assert (moved[4:, 6:] == img[:-4, :-6]).all()            # the same pixels, only moved
    assert (moved[4:, :6] == img[:-4, :1]).all() and (moved[:4] == moved[4:5]).all()   # the edges repeat


def test_offset_scales_with_the_picture(tmp_path):
    p = tmp_path / "look.png"
    cv2.imwrite(str(p), np.zeros((448, 768, 3), np.uint8))
    assert look_pin.offset(p) == (3, 2)


def test_the_run_moves_the_kept_look_and_records_it(tmp_path, monkeypatch):
    from cinestudio import runner

    frame = np.full((448, 768, 3), 90, np.uint8)
    cv2.rectangle(frame, (100, 60), (200, 120), (30, 200, 220), -1)    # a banner the look must keep
    fp, mp = tmp_path / "frame.png", tmp_path / "mask.png"
    cv2.imwrite(str(fp), frame)
    cv2.imwrite(str(mp), _figure())

    class Stub:
        m = {"look_pin": True, "seed": 303}
        events: list = []
        _pinned_look = runner.Run._pinned_look
        _offset_look = runner.Run._offset_look

        def _qwen_look(self, step, name, frame_, view, out, mask=None, seed=None):
            cv2.imwrite(str(out), frame)                                # a look that kept the room
            return None

        def event(self, kind, **kw):
            self.events.append((kind, kw))

        def save(self):
            pass

    stub, step, out = Stub(), {}, tmp_path / "look.png"
    assert stub._pinned_look(step, "look", fp, mp, "front", out) is None
    assert step["look_checks"]["look"][0]["ok"]
    assert step["look_offsets"]["look"] == [3, 2]
    assert (cv2.imread(str(out))[2:, 3:] == frame[:-2, :-3]).all()      # the kept look is moved
    assert (cv2.imread(str(tmp_path / "look_try1.png")) == frame).all()  # the gate's copy is not
    assert [k for k, _ in stub.events] == ["look_check", "look_offset"]
