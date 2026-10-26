"""Does anything other than the stand-in move in a template? (Moving shapes in the background — kicked-up snow, birds,
people — become phantom runners and animals in the run: blind A/B 2026-10-20.)

    background_motion(video, pass_dir) -> {"moving_share": max share of the frame that moves outside the figure,
                                           "frames": [frames above the limit], "flag": bool, "per_frame": [...]}

The figure's area comes from the template's passes (video_passes.py: the hull of the stand-in's DWPose points, made
larger). The camera's own motion is removed first: consecutive frames are aligned with ORB features outside the figure
(a similarity transform, RANSAC), so a push, pull or pan does not count. What still changes outside the figure, above a
grey-level threshold and as connected blobs of a minimum size, is "moving".
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DIFF = 28                  # grey levels: below = noise, grain, a light flicker
MIN_BLOB = 0.0015          # a blob smaller than this share of the frame is ignored (sparkles, snowflakes)
LIMIT = 0.004              # a frame with more moving share than this outside the figure is flagged
GROW = 0.15                # the figure's area is made larger by this share of its height (its shadow, loose cloth)


def _figure(mask_png: Path, shape) -> np.ndarray:
    m = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(shape, bool)
    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 127
    if m.any():
        ys = np.nonzero(m)[0]
        r = max(3, int((ys.max() - ys.min()) * GROW))
        m = cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    return m


def background_motion(video: str | Path, pass_dir: str | Path, size: int = 384) -> dict:
    pass_dir = Path(pass_dir)
    masks = sorted((pass_dir / "mask").glob("mask_*.png"))
    cap = cv2.VideoCapture(str(video))
    orb = cv2.ORB_create(800)
    prev = prev_fig = None
    per_frame = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h = int(f.shape[0] * size / f.shape[1])
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f, (size, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY),
                             (5, 5), 0)
        fig = _figure(masks[i], g.shape) if i < len(masks) else np.zeros(g.shape, bool)
        if prev is not None:
            free = (~fig & ~prev_fig).astype(np.uint8) * 255
            k1, d1 = orb.detectAndCompute(prev, free)
            k2, d2 = orb.detectAndCompute(g, free)
            warped = prev
            if d1 is not None and d2 is not None and len(k1) >= 8 and len(k2) >= 8:
                matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
                if len(matches) >= 8:
                    a = np.float32([k1[m.queryIdx].pt for m in matches])
                    b = np.float32([k2[m.trainIdx].pt for m in matches])
                    mtx, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if mtx is not None:
                        warped = cv2.warpAffine(prev, mtx, (g.shape[1], g.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            diff = (cv2.absdiff(g, warped) > DIFF) & ~fig & ~prev_fig
            diff[:4], diff[-4:], diff[:, :4], diff[:, -4:] = False, False, False, False     # warp borders
            n, lab, stats, _ = cv2.connectedComponentsWithStats(diff.astype(np.uint8))
            big = sum(int(stats[j, cv2.CC_STAT_AREA]) for j in range(1, n)
                      if stats[j, cv2.CC_STAT_AREA] >= MIN_BLOB * g.size)
            per_frame.append(round(big / g.size, 5))
        prev, prev_fig = g, fig
        i += 1
    frames = [j + 1 for j, v in enumerate(per_frame) if v > LIMIT]
    return {"moving_share": max(per_frame, default=0.0), "frames": frames, "flag": bool(frames),
            "per_frame": per_frame}
