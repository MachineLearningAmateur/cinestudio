"""The chained look's picture fixes (2026-10-23): lightness anchored to the first look, sharpening only up to clip 1's
sharpness, and an exact frame extraction (no darkening)."""
import subprocess

import cv2
import numpy as np

from cinestudio import video


def _noise_picture(path, level=110, seed=1, blur=0.0):
    rng = np.random.default_rng(seed)
    im = np.clip(rng.normal(level, 30, (180, 320, 3)), 0, 255).astype(np.uint8)
    if blur:
        im = cv2.GaussianBlur(im, (0, 0), blur)
    cv2.imwrite(str(path), im)
    return path


def _L(path, mask=None):
    lab = cv2.cvtColor(cv2.imread(str(path)).astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    return float(lab[..., 0].mean() if mask is None else lab[..., 0][mask].mean())


def test_match_lightness_brings_the_background_to_the_reference(tmp_path):
    ref = _noise_picture(tmp_path / "ref.png", 120)
    dark = _noise_picture(tmp_path / "dark.png", 113, seed=2)
    r = video.match_lightness(dark, ref)
    assert r["gain"] > 1 and not r["clamped"]
    assert abs(_L(dark) - _L(ref)) < 1.0                     # within 1 L (the soft roll-off keeps highlights)


def test_match_lightness_is_limited_and_follows_the_layout_ratio(tmp_path):
    ref = _noise_picture(tmp_path / "ref.png", 160)
    dark = _noise_picture(tmp_path / "dark.png", 60, seed=2)
    r = video.match_lightness(dark, ref, max_gain=0.10)
    assert r["clamped"] and r["gain"] == 1.1
    ref2 = _noise_picture(tmp_path / "ref2.png", 120)
    same = _noise_picture(tmp_path / "same.png", 120, seed=3)
    r = video.match_lightness(same, ref2, ratio=0.95)          # the layout itself got 5% darker: so may the picture
    assert 0.94 < r["gain"] < 0.97


def test_match_lightness_measures_only_the_background(tmp_path):
    ref = _noise_picture(tmp_path / "ref.png", 120)
    im = cv2.imread(str(_noise_picture(tmp_path / "p.png", 120, seed=4)))
    im[40:140, 120:200] = 250                                  # a bright character in the middle
    cv2.imwrite(str(tmp_path / "p.png"), im)
    mask = np.zeros(im.shape[:2], np.uint8)
    mask[40:140, 120:200] = 255
    cv2.imwrite(str(tmp_path / "mask.png"), mask)
    r = video.match_lightness(tmp_path / "p.png", ref, mask=tmp_path / "mask.png")
    assert abs(r["gain"] - 1.0) < 0.02 and r["background_share"] < 0.8


def test_sharpen_to_stops_at_the_target(tmp_path):
    soft = _noise_picture(tmp_path / "soft.png", 120, blur=1.5)
    target = video.sharpness(soft) * 2.0
    r = video.sharpen_to(soft, target, max_amount=3.0)
    assert 0 < r["amount"] < 3.0 and abs(r["after"] - target) / target < 0.1
    sharp = _noise_picture(tmp_path / "sharp.png", 120)
    r = video.sharpen_to(sharp, video.sharpness(sharp) * 0.5, max_amount=1.0)
    assert r["amount"] == 0.0                                  # already sharper than the target: untouched


def test_frame_png_returns_the_encoded_brightness(tmp_path):
    src = _noise_picture(tmp_path / "src.png", 120, blur=2.0)
    clip = tmp_path / "c.mp4"
    subprocess.run([video.ffmpeg(), "-nostdin", "-y", "-v", "error", "-loop", "1", "-i", str(src), "-t", "0.5", "-r", "24",
                    "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-color_range", "tv", "-c:v", "libx264",
                    "-crf", "1", str(clip)], check=True, stdin=subprocess.DEVNULL)
    video.frame_png(clip, tmp_path / "f.png", 3)
    assert abs(_L(tmp_path / "f.png") - _L(src)) < 0.2


def test_the_gain_is_solved_with_the_roll_off(tmp_path):
    """Codex review: a bright background (L ~85) needs more gain than target/before because the roll-off damps it."""
    import pytest
    ref = tmp_path / "ref.png"
    cur = tmp_path / "cur.png"
    cv2.imwrite(str(ref), np.full((90, 160, 3), 228, np.uint8))     # L ~ 89.6
    cv2.imwrite(str(cur), np.full((90, 160, 3), 215, np.uint8))     # L ~ 85
    r = video.match_lightness(cur, ref)
    assert r["target_met"] and abs(r["residual_L"]) < 0.3 and not r["clamped"]
    assert r["gain"] > r["target_L"] / r["before_L"]                # more than the naive ratio
    with pytest.raises(ValueError):
        video.match_lightness(tmp_path / "missing.png", ref)


def test_a_fallback_on_one_side_only_changes_nothing(tmp_path):
    ref = _noise_picture(tmp_path / "ref.png", 140)
    cur = _noise_picture(tmp_path / "cur.png", 110, seed=5)
    mask = np.zeros((180, 320), np.uint8)
    mask[60:120, 140:180] = 255
    cv2.imwrite(str(tmp_path / "m.png"), mask)
    before = cv2.imread(str(cur)).copy()
    r = video.match_lightness(cur, ref, mask=tmp_path / "m.png", ref_mask=None)   # a real background vs "no mask"
    assert r["gain"] == 1.0 and "skipped" in r and np.array_equal(cv2.imread(str(cur)), before)
    r = video.match_lightness(cur, ref, ratio=0.0)                   # a black layout: no usable ratio
    assert "skipped" in r


def test_frame_png_needs_both_sizes(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        video.frame_png(tmp_path / "x.mp4", tmp_path / "f.png", 0, width=100)


def test_match_lightness_refuses_a_bad_limit(tmp_path):
    import pytest
    a = _noise_picture(tmp_path / "a.png", 120)
    b = _noise_picture(tmp_path / "b.png", 110, seed=2)
    for bad in (-0.1, float("nan")):
        with pytest.raises(ValueError):
            video.match_lightness(b, a, max_gain=bad)


def test_tone_restores_colourfulness_and_contrast(tmp_path):
    rng = np.random.default_rng(3)
    base = np.clip(rng.normal(0, 1, (120, 200, 3)) * 25 + np.array([70, 150, 120]), 0, 255).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "ref.png"), base)
    lab = cv2.cvtColor(base.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    lab[..., 1:] *= 0.9                                              # 10% less colour
    m = lab[..., 0].mean()
    lab[..., 0] = m + (lab[..., 0] - m) * 0.94                      # 6% less contrast
    faded = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) * 255, 0, 255).round().astype(np.uint8)
    cv2.imwrite(str(tmp_path / "cur.png"), faded)
    r = video.match_lightness(tmp_path / "cur.png", tmp_path / "ref.png", tone=True)
    assert 1.08 < r["chroma_gain"] < 1.13 and 1.03 < r["contrast_gain"] < 1.09
    out = cv2.cvtColor(cv2.imread(str(tmp_path / "cur.png")).astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    ref = cv2.cvtColor(base.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    assert abs(np.hypot(out[..., 1], out[..., 2]).mean() - np.hypot(ref[..., 1], ref[..., 2]).mean()) < 0.6
