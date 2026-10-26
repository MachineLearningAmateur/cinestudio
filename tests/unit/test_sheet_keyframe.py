"""The sheet keyframe (cinestudio/sheet_keyframe.py): a sheet panel pasted into the mannequin's box."""
import cv2
import numpy as np

from cinestudio import sheet_keyframe


def test_the_back_panel_lands_in_the_mannequin_box_with_its_black_legs(tmp_path):
    sheet = np.zeros((400, 1000, 3), np.uint8)
    for i in range(5):                                     # five panels: a coloured body + pure-black legs
        x = 40 + i * 190
        cv2.rectangle(sheet, (x, 40), (x + 100, 200), (40 * i + 40, 0, 120), -1)    # poncho
        cv2.rectangle(sheet, (x + 20, 300), (x + 40, 360), (220, 220, 220), -1)      # left shin bandage
        cv2.rectangle(sheet, (x + 60, 300), (x + 80, 360), (220, 220, 220), -1)      # right shin bandage
    rgb = np.full((448, 768, 3), 200, np.uint8)
    mask = np.zeros((448, 768), np.uint8)
    cv2.rectangle(mask, (350, 100), (420, 400), 255, -1)
    for name, img in (("sheet.png", sheet), ("rgb.png", rgb), ("mask.png", mask)):
        cv2.imwrite(str(tmp_path / name), img)
    out = sheet_keyframe.make(tmp_path / "rgb.png", tmp_path / "mask.png", tmp_path / "sheet.png", "back",
                              tmp_path / "kf.png", fill_gaps=True)
    kf = cv2.imread(str(out))
    assert kf.shape == rgb.shape
    assert abs(int(kf[150, 385, 0]) - (40 * 3 + 40)) < 25          # the BACK panel's poncho colour, inside the box
    thigh = kf[260, 360:410]                                      # between the hem and the bandages: filled charcoal
    assert (thigh.max(axis=1) < 80).any()
    assert kf[50, 100].tolist() == [200, 200, 200]                # outside the box: the plate is untouched


def test_no_gap_fill_by_default_and_a_foreign_layout_is_refused(tmp_path):
    """Codex review 2026-10-22: the gap fill is opt-in (it can join a sleeve to a skirt), and a sheet that is not the
    5-panel layout raises LayoutError (the run then skips the keyframe) instead of an IndexError."""
    import pytest
    sheet = np.zeros((400, 1000, 3), np.uint8)
    for i in range(5):
        x = 40 + i * 190
        cv2.rectangle(sheet, (x, 40), (x + 100, 200), (120, 0, 120), -1)
        cv2.rectangle(sheet, (x + 20, 300), (x + 40, 360), (220, 220, 220), -1)
    fig, fgm = sheet_keyframe.sheet_panel(sheet, "back")
    assert not fgm[165:255].any()                              # the black gap (rows 205-295) stays background
    four = sheet[:, :760]                                      # only four panels
    with pytest.raises(sheet_keyframe.LayoutError):
        sheet_keyframe.sheet_panel(four, "back")
    white = np.full((400, 1000, 3), 255, np.uint8)
    with pytest.raises(sheet_keyframe.LayoutError):
        sheet_keyframe.sheet_panel(white, "back")


def test_a_view_image_keeps_its_dark_eye_holes(tmp_path):
    view = np.zeros((300, 200, 3), np.uint8)
    cv2.rectangle(view, (50, 20), (150, 280), (200, 200, 200), -1)
    cv2.rectangle(view, (80, 60), (95, 70), (0, 0, 0), -1)          # a black eye hole inside the figure
    fig, fgm = sheet_keyframe.view_panel(view)
    assert fgm.all()                                               # cropped to the figure, the hole is figure
