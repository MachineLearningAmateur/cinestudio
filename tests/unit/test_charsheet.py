"""Turntable sheet: exact frame picks and the sheet layout (synthetic frames, no AI)."""

import cv2
import numpy as np
import pytest

from cinestudio import charsheet


def test_frame_for_exact_angles():
    assert [charsheet.frame_for(a, 121, 360) for a in (0, 45, 90, 180, 360)] == [0, 15, 30, 60, 120]
    assert charsheet.frame_for(90, 121, 180) == 60
    with pytest.raises(ValueError):
        charsheet.frame_for(200, 121, 180)


def test_make_sheet_from_a_clip(tmp_path):
    clip = tmp_path / "tt.mp4"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 24, (144, 256))
    for i in range(25):                       # a bar whose width shrinks: a figure seen from more and more aside
        f = np.zeros((256, 144, 3), np.uint8)
        half = 40 - i
        cv2.rectangle(f, (72 - half, 30), (72 + half, 230), (200, 120, 60), -1)
        vw.write(f)
    vw.release()
    r = charsheet.make_sheet(clip, tmp_path / "sheet.png", degrees=180, angles=(0, 90, 180))
    assert [v["frame"] for v in r["views"]] == [0, 12, 24]
    sheet = cv2.imread(r["sheet"])
    assert sheet.shape == (896, 1536, 3)
    assert sheet[:5].max() < 10 and sheet[:, -5:].max() < 10        # black border
    assert (tmp_path / "sheet_view_090.png").is_file()


def test_measured_angles_win_over_nominal():
    az = [0, 10, 44, 46, 90, 120, 180]
    assert charsheet.pick_frames((0, 45, 90, 180), 7, azimuths=az) == {0: 0, 45: 2, 90: 4, 180: 6}
    with pytest.raises(ValueError, match="trimmed"):
        charsheet.pick_frames((0,), 5, azimuths=az)
    with pytest.raises(ValueError, match="no frame near"):
        charsheet.pick_frames((270,), 7, azimuths=az)


def test_short_clip_is_refused(tmp_path):
    clip = tmp_path / "tt.mp4"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 64))
    for _ in range(10):
        vw.write(np.full((64, 64, 3), 200, np.uint8))
    vw.release()
    with pytest.raises(ValueError, match="expected 121"):
        charsheet.make_sheet(clip, tmp_path / "s.png", expected_frames=121)



def test_fill_specks_never_makes_a_speck_white():
    from cinestudio import refsheet
    img = np.zeros((40, 40, 3), np.uint8)
    img[19:21, 19:21] = 120                                       # a grey speck on pure black
    out, n = refsheet.fill_specks(img)
    assert n > 0 and out[19:21, 19:21].max() == 0
