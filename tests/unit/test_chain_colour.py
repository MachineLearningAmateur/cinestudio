"""chain_colour.lock: the chained look's dark costume pixels get the look image's colour cast (2026-10-22)."""
import cv2
import numpy as np

from cinestudio import chain_colour


def _lab_bgr(L, a, b, shape=(120, 160)):
    lab = np.zeros((*shape, 3), np.uint8)
    lab[..., 0], lab[..., 1], lab[..., 2] = int(L * 255 / 100), 128 + a, 128 + b
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def test_the_dark_suit_gets_the_look_cast_and_the_rest_stays(tmp_path):
    chained = _lab_bgr(60, 5, 20)                       # warm background
    chained[30:90, 60:100] = _lab_bgr(18, 12, 12, (60, 40))      # a BROWN suit
    look = _lab_bgr(60, 5, 20)
    look[30:90, 60:100] = _lab_bgr(18, 4, -10, (60, 40))         # the black (cool) suit of clip 1
    mask = np.zeros((120, 160), np.uint8); mask[25:95, 55:105] = 255
    for name, img in (("c.png", chained), ("l.png", look), ("m.png", mask)):
        cv2.imwrite(str(tmp_path / name), img)
    r = chain_colour.lock(tmp_path / "c.png", tmp_path / "l.png", tmp_path / "m.png", tmp_path / "m.png")
    assert r["done"] and r["shift"][1] < -15                      # b* moves toward the cool look
    out = cv2.cvtColor(cv2.imread(str(tmp_path / "c.png")), cv2.COLOR_BGR2LAB).astype(int)
    assert abs(out[60, 80, 2] - 128 - (-10)) <= 4                  # the suit's b* is the look's now
    assert abs(out[5, 5, 2] - 128 - 20) <= 2                       # the background is unchanged


def test_nothing_happens_without_enough_dark_pixels(tmp_path):
    img = _lab_bgr(70, 0, 0)
    mask = np.full((120, 160), 255, np.uint8)
    for name, im in (("c.png", img), ("l.png", img), ("m.png", mask)):
        cv2.imwrite(str(tmp_path / name), im)
    assert chain_colour.lock(tmp_path / "c.png", tmp_path / "l.png", tmp_path / "m.png", tmp_path / "m.png")["done"] is False


def test_match_moves_the_whole_frame_toward_the_reference_palette(tmp_path):
    chained = _lab_bgr(50, 20, 25)                                 # a warm, magenta-red drifted frame
    ref = _lab_bgr(55, 2, -5)
    cv2.imwrite(str(tmp_path / "c.png"), chained); cv2.imwrite(str(tmp_path / "r.png"), ref)
    r = chain_colour.match(tmp_path / "c.png", tmp_path / "r.png", strength=1.0)
    assert r["done"] and r["method"] == "reinhard_lab"
    out = cv2.cvtColor(cv2.imread(str(tmp_path / "c.png")), cv2.COLOR_BGR2LAB).astype(int)
    assert abs(out[10, 10, 1] - 128 - 2) <= 2 and abs(out[10, 10, 2] - 128 + 5) <= 2


def test_add_adain_can_take_clip_1s_latent_as_the_reference():
    from cinestudio import runner
    g = {"crop": {"class_type": "LTXVCropGuides", "inputs": {}}, "up": {"class_type": "X", "inputs": {"samples": ["crop", 2]}},
         "look_load": {"class_type": "LoadImage", "inputs": {}}}
    runner.add_adain(g, 0.3, 768, 448, ref_latent="relay_r1_1_s1.latent")
    assert g["adain_ref"] == {"class_type": "LoadLatent", "inputs": {"latent": "relay_r1_1_s1.latent"}}
    assert "adain_img" not in g and g["up"]["inputs"]["samples"] == ["adain_crop", 0]
    assert g["adain_crop"]["inputs"]["factor"] == 0.3
    g2 = {"crop": {"class_type": "LTXVCropGuides", "inputs": {}}, "look_load": {"class_type": "LoadImage", "inputs": {}}}
    runner.add_adain(g2, 0.5, 768, 448)
    assert g2["adain_ref"]["class_type"] == "VAEEncode"                # the default: the run look's latent
