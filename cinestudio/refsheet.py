"""Prepare a character reference sheet for the video models.

    python -m cinestudio.refsheet <sheet.png> --out <dir> [--tolerance 14]

Writes into <dir>:
  sheet_black.png   the sheet with its plain background changed to black (the LTX-2.3 Ingredients
                    IC-LoRA was trained on reference sheets on black)
  sheet_mask.png    white = character, black = background
  view_<n>.png      each separate view (panel) of the sheet on its original background, n = 1.. from left
  view_<n>_black.png  the same view on black
Prints a JSON summary.

The background is found by a flood fill from the image border with a colour tolerance, so it works for
plain or slightly uneven backgrounds and stops at the character's outline. Views are split at columns
that contain only background.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def background_mask(img: np.ndarray, tolerance: int = 14) -> np.ndarray:
    """True where the pixel belongs to the background connected to the image border."""
    h, w = img.shape[:2]
    fill = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    work = img.copy()
    step = max(1, min(h, w) // 64)
    seeds = [(x, 0) for x in range(0, w, step)] + [(x, h - 1) for x in range(0, w, step)] + \
            [(0, y) for y in range(0, h, step)] + [(w - 1, y) for y in range(0, h, step)]
    t = (tolerance,) * 3
    # start only from border pixels that have the background colour (the median of the border): a
    # character that touches the edge (a mask in a close-up) must not become a start point
    border = np.array([img[y, x] for x, y in seeds], np.float32)
    bg_colour = np.median(border, axis=0)
    seeds = [s for s, px in zip(seeds, border) if np.abs(px - bg_colour).max() <= 2 * tolerance]
    for x, y in seeds:
        if fill[y + 1, x + 1]:
            continue
        cv2.floodFill(work, fill, (x, y), 0, t, t, flags)
    bg = fill[1:-1, 1:-1] > 0
    # remove single-pixel specks inside the background and close tiny holes in the character
    k = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx((~bg).astype(np.uint8), cv2.MORPH_OPEN, k)
    return fg == 0


def split_views(fg: np.ndarray, min_gap: int = 8, min_width: int = 20) -> list[tuple[int, int]]:
    """(x0, x1) column ranges of the separate views."""
    cols = fg.any(axis=0)
    views, start, gap = [], None, 0
    for x, on in enumerate(cols):
        if on:
            if start is None:
                start = x
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                if x - gap + 1 - start >= min_width:
                    views.append((start, x - gap + 1))
                start, gap = None, 0
    if start is not None and len(cols) - start >= min_width:
        views.append((start, len(cols)))
    return views


def prepare(sheet: str | Path, out: str | Path, tolerance: int = 14, margin: int = 24) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(sheet), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {sheet}")
    bg = background_mask(img, tolerance)
    fg = ~bg
    black = img.copy()
    black[bg] = 0
    cv2.imwrite(str(out / "sheet_black.png"), black)
    cv2.imwrite(str(out / "sheet_mask.png"), fg.astype(np.uint8) * 255)
    views = split_views(fg)
    h = img.shape[0]
    files = []
    for i, (x0, x1) in enumerate(views, 1):
        rows = np.where(fg[:, x0:x1].any(axis=1))[0]
        y0, y1 = max(0, rows[0] - margin), min(h, rows[-1] + 1 + margin)
        a, b = max(0, x0 - margin), min(img.shape[1], x1 + margin)
        cv2.imwrite(str(out / f"view_{i}.png"), img[y0:y1, a:b])
        cv2.imwrite(str(out / f"view_{i}_black.png"), black[y0:y1, a:b])
        files.append({"view": i, "x": [int(a), int(b)], "y": [int(y0), int(y1)]})
    return {"sheet_black": str(out / "sheet_black.png"), "background_fraction": round(float(bg.mean()), 3),
            "views": files}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance", type=int, default=14, help="colour distance still counted as background")
    a = ap.parse_args(argv)
    print(json.dumps(prepare(a.sheet, a.out, a.tolerance), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


def clean_edges(img: np.ndarray, old_bg: tuple[int, int, int] | None = None, width: float = 2.5) -> tuple[np.ndarray, int]:
    """Remove the light halo that an old light background leaves around a character on black.

    1. Light grey pixels next to the black background take the colour of the darkest pixel near them
       (the outline), so the edge stays smooth.
    2. With `old_bg` (BGR of the old background): small pockets of that colour that dark pixels surround
       (gaps between gloved fingers) become black. Pockets next to light parts (a white mask,
       bandages) stay. Returns (image, number of changed pixels)."""
    bg = background_mask(img, 10)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dist = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 3)
    work = img.copy()
    work[bg] = 255
    dark = cv2.erode(work, np.ones((5, 5), np.uint8))
    fix = (~bg) & (dist <= width) & (hsv[..., 2] >= 70) & (hsv[..., 1] <= 45) & (dark.max(axis=2) < hsv[..., 2].astype(int) - 40)
    out = img.copy()
    out[fix] = dark[fix]
    changed = int(fix.sum())
    if old_bg is not None:
        diff = np.abs(out.astype(np.int16) - np.array(old_bg, np.int16)).max(axis=2)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        cand = ((diff <= 26) & (hsv[..., 1] <= 18)).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        small = np.zeros(cand.shape, bool)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] > 600:
                continue
            comp = (lab == i).astype(np.uint8)
            ring = cv2.dilate(comp, np.ones((9, 9), np.uint8)).astype(bool) & ~cv2.dilate(comp, np.ones((5, 5), np.uint8)).astype(bool)
            if ring.any() and (hsv[..., 2][ring] >= 120).mean() < 0.08:
                small |= comp.astype(bool)
        grow = cv2.dilate(small.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        pockets = small | (grow & (hsv[..., 2] >= 90) & (hsv[..., 1] <= 30) & (diff <= 60))
        out[pockets] = 0
        changed += int(pockets.sum())
    out, specks = fill_specks(out)
    return out, changed + specks


def fill_specks(img: np.ndarray, max_area: int = 160, light_v: int = 95, max_sat: int = 40, dark_v: int = 85,
                dark_share: float = 0.7) -> tuple[np.ndarray, int]:
    """Step 3 of clean_edges: small light, grey spots that dark pixels mostly surround (old background
    mixed into the gaps between gloved fingers) take the colour of the dark pixels around them. Large
    light parts (a mask, bandages) are not small, so they stay."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v, s = hsv[..., 2], hsv[..., 1]
    light = ((v >= light_v) & (s <= max_sat)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(light, connectivity=8)
    fix = np.zeros(light.shape, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            continue
        x, y, ww, hh = stats[i, :4]
        x0, y0, x1, y1 = max(0, x - 4), max(0, y - 4), min(img.shape[1], x + ww + 4), min(img.shape[0], y + hh + 4)
        comp = lab[y0:y1, x0:x1] == i
        ring = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~comp
        rv = v[y0:y1, x0:x1][ring]
        if rv.size and (rv < dark_v).mean() >= dark_share:
            fix[y0:y1, x0:x1] |= comp
    fix |= cv2.dilate(fix.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (v >= 70) & (s <= max_sat)
    work = img.copy()
    work[fix | (v < 8)] = 255                                 # neither the spot nor pure black is a glove colour
    dark = cv2.erode(work, np.ones((7, 7), np.uint8))
    dark[(dark >= 250).all(axis=2)] = 0                       # no glove colour near: black, never white
    out = img.copy()
    out[fix] = dark[fix]
    return out, int(fix.sum())
