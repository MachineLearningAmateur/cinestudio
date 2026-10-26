"""Two characters in one layout run (2026-10-23): the sheet in groups and the clip text."""
import numpy as np

from cinestudio import charsheet, runner


def _figure(h, w, colour):
    img = np.zeros((h + 20, w + 20, 3), np.uint8)
    img[10:10 + h, 10:10 + w] = colour
    return img


def test_assemble_groups_puts_every_figure_on_one_feet_line():
    groups = [[_figure(300, 80, (200, 50, 50)) for _ in range(3)], [_figure(200, 120, (50, 200, 50)) for _ in range(3)]]
    sheet = charsheet.assemble_groups(groups, size=(1536, 896))
    assert sheet.shape == (896, 1536, 3)
    red = np.where((sheet[..., 0] > 150) & (sheet[..., 1] < 100))
    green = np.where((sheet[..., 1] > 150) & (sheet[..., 0] < 100))
    assert red[0].size and green[0].size
    assert abs(int(red[0].max()) - int(green[0].max())) <= 18         # one feet line (the scaled cut-out border)
    assert red[1].max() < green[1].min()                                # group 1 left, group 2 right


def test_duo_prompt_names_each_character_by_its_place():
    duo = {"left": "a masked figure in a plum jacket.", "right": "A wanderer in a fox mask and a teal cape.",
           "names": ["masked3d_rest", "kitsune"]}
    m = {"prompt": {"character": runner.duo_sheet_text(duo), "video": "Both dance.", "style": "3D animation",
                    "duo": duo}, "prompt_format": "ing25", "sound_text": ""}
    text = runner.video_prompt(m)
    assert "**Panels 1-3 (Character 1):**" in text and "**Panels 4-6 (Character 2):**" in text
    assert "on the left, the first character, a masked figure in a plum jacket;" in text
    assert "on the right, the second character, a wanderer in a fox mask and a teal cape." in text
    assert text.index("### Reference Sheet Description") < text.index("### Target Description")


def test_duo_boxes_split_two_figures_at_the_gap():
    m = np.zeros((448, 768), bool)
    m[80:400, 200:300] = True                      # the left figure
    m[140:400, 480:560] = True                     # the shorter right figure
    boxes, split = runner.duo_boxes(m)
    assert 300 <= split <= 480
    lx0, ly0, lx1, ly1 = boxes["left"]
    rx0, ry0, rx1, ry1 = boxes["right"]
    assert lx0 < 200 and lx1 > 300 and lx1 <= split and ly0 < 80
    assert rx0 >= split and rx0 < 480 and rx1 > 560 and ry0 < 140 and ry0 > ly0     # each box fits its own figure
    import pytest
    with pytest.raises(runner.RunError):
        runner.duo_boxes(np.zeros((10, 10), bool))
    touching = np.zeros((100, 100), bool); touching[10:90, 20:80] = True
    with pytest.raises(runner.RunError):
        runner.duo_boxes(touching)
