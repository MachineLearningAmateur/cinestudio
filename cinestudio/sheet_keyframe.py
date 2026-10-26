"""A keyframe look made from the character SHEET itself (no GPU): the Blender frame with the mannequin painted out
(OpenCV inpaint over its grown mask) and one sheet panel (front / right / back / left) pasted into the mannequin's
mask box (same height, same feet line).

Qwen-Image-Edit drew the FRONT when asked for a back view (R3 look_k2, 2026-10-21), so back-view clips had no keyframe.
Gauntlet R3: the sheet's side panel at clip 1 frame 88 kept the poncho through the side views, and the back panel at
clip 2 frame 60 (chained from that clip 1) gave a full mid-thigh poncho in every frame.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PANELS = {"face": 0, "front": 1, "right": 2, "back": 3, "left": 4}


class LayoutError(RuntimeError):
    """The sheet is not cinestudio's 5-panel layout (an imported sheet): no panel can be cut from it."""


def view_panel(img, fill_gaps: bool = False):
    """The figure of one view image (the character library's views/<view>.png, on black): cropped to the figure."""
    return _finish(img, ~_border_dark(img), fill_gaps)


def _border_dark(panel):
    """The black background: dark (<= 10) areas that touch the image border (dark parts inside the figure, such as the
    mask's eye holes, stay figure)."""
    dark = (panel.max(axis=2) <= 10).astype(np.uint8)
    n, lab = cv2.connectedComponents(dark, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    return np.isin(lab, list(border)) & (dark > 0)


def sheet_panel(sheet, view, fill_gaps: bool = False):
    fg = sheet.max(axis=2) > 25
    cols = fg.any(axis=0)
    segs, start = [], None
    for x, v in enumerate(list(cols) + [False]):
        if v and start is None:
            start = x
        if not v and start is not None:
            if x - start > 20:
                segs.append((start, x))
            start = None
    if len(segs) != len(PANELS):                          # Codex review 2026-10-22: a white or 4-panel sheet crashed
        raise LayoutError(f"the sheet has {len(segs)} panels, not cinestudio's {len(PANELS)}")
    x0, x1 = segs[PANELS[view]]
    pad = 6
    x0, x1 = max(0, x0 - pad), min(sheet.shape[1], x1 + pad)
    panel = sheet[:, x0:x1]
    return _finish(panel, ~_border_dark(panel), fill_gaps)


def _finish(panel, fgp, fill_gaps: bool):
    fgp = cv2.morphologyEx(fgp.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    if fill_gaps:
        # OPT-IN (Codex review 2026-10-22: a wide sleeve above a flared skirt would be joined by invented cloth). The
        # 2D cloaked sheet drew the black bodysuit (thighs between the hem and the shin bandages) as pure black, like
        # the background: join the parts above and below over each leg and paint the gap charcoal.
        closed = cv2.morphologyEx(fgp.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((160, 9), np.uint8)) > 0
        gap = closed & ~fgp & (panel.max(axis=2) <= 10)
        panel = panel.copy()
        panel[gap] = (34, 30, 30)
        fgp = fgp | gap
    if not fgp.any():
        raise LayoutError("no figure in the panel")
    rows = np.where(fgp.any(axis=1))[0]
    cols = np.where(fgp.any(axis=0))[0]
    y0, y1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    return panel[y0:y1, c0:c1], fgp[y0:y1, c0:c1]


def make(rgb_path: str | Path, mask_path: str | Path, sheet_path: str | Path, view: str, out: str | Path,
         view_image: str | Path | None = None, fill_gaps: bool = False) -> Path:
    """`view_image`: the character's own view (the run's copy); else the panel is cut from the sheet (LayoutError when
    the sheet is not cinestudio's 5-panel layout)."""
    rgb = cv2.imread(str(rgb_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if rgb is None or mask is None:
        raise RuntimeError(f"cannot read {rgb_path} or {mask_path}")
    mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
    if not mask.any():
        raise RuntimeError(f"no character in the mask {mask_path}")
    ys, xs = np.where(mask)
    by0, by1, bx0, bx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    grown = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((15, 15), np.uint8))
    plate = cv2.inpaint(rgb, grown, 9, cv2.INPAINT_TELEA)
    vimg = cv2.imread(str(view_image)) if view_image else None
    if vimg is not None:
        fig, fgm = view_panel(vimg, fill_gaps)
    else:
        sh = cv2.imread(str(sheet_path))
        if sh is None:
            raise LayoutError(f"cannot read the sheet {sheet_path}")
        fig, fgm = sheet_panel(sh, view, fill_gaps)
    h = by1 - by0
    w = int(round(fig.shape[1] * h / fig.shape[0]))
    fig = cv2.resize(fig, (w, h), interpolation=cv2.INTER_AREA)
    alpha = cv2.GaussianBlur(cv2.resize(fgm.astype(np.float32), (w, h)), (0, 0), 1.0)[..., None]
    cx = (bx0 + bx1) // 2
    x0 = max(0, cx - w // 2)
    x1 = min(plate.shape[1], x0 + w)
    roi = plate[by0:by1, x0:x1].astype(np.float32)
    a = alpha[:, : x1 - x0]
    plate[by0:by1, x0:x1] = (fig[:, : x1 - x0] * a + roi * (1 - a)).astype(np.uint8)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), plate)
    return Path(out)
