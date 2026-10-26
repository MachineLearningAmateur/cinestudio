"""Colour lock for the chained look image of layout clips (look_chain "prev").

Clip k > 1 takes clip k-1's frame at its own start as its frame -1 look image. When that frame has drifted (masked3d
orbit, 2026-10-22: the black bodysuit turned bronze, then brown leather, clip by clip; the blind critic: "each later
clip inherits and amplifies the previous clip's errors"), the drift is passed on. `lock` moves the colour CAST of the
character's dark pixels (the suit, gloves, shoes: Lab L < 30 inside the Blender mask) back to the cast of the same
pixels in the run's look image (clip 1's palette): a mean shift of a* and b* only, with a soft weight, so the shading,
the angle and the background stay as the chained frame has them."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DARK_L = 30.0          # Lab lightness (0..100) of the "dark" costume pixels


def _lab(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def _mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 127
    return cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def lock(chained: str | Path, look: str | Path, chained_mask: str | Path, look_mask: str | Path,
         strength: float = 1.0, min_pixels: int = 400) -> dict:
    """Rewrite `chained` in place. Returns the measured casts and the shift (or a reason when nothing was done)."""
    img, ref = cv2.imread(str(chained)), cv2.imread(str(look))
    if img is None or ref is None:
        raise FileNotFoundError(chained if img is None else look)
    lab, rlab = _lab(img), _lab(ref)
    cm, rm = _mask(chained_mask, img.shape[:2]), _mask(look_mask, ref.shape[:2])
    dark, rdark = cm & (lab[..., 0] < DARK_L), rm & (rlab[..., 0] < DARK_L)
    if dark.sum() < min_pixels or rdark.sum() < min_pixels:
        return {"done": False, "reason": "too few dark costume pixels", "dark": int(dark.sum()), "ref_dark": int(rdark.sum())}
    cast = lab[dark][:, 1:].mean(axis=0)
    want = rlab[rdark][:, 1:].mean(axis=0)
    shift = (want - cast) * float(strength)
    # the weight: 1 for clearly dark costume pixels, fading out up to L 45 and at the mask edge
    w = np.clip((45.0 - lab[..., 0]) / 15.0, 0.0, 1.0) * cm.astype(np.float32)
    w = cv2.GaussianBlur(w, (0, 0), 1.5)
    out = lab.copy()
    out[..., 1] += w * shift[0]
    out[..., 2] += w * shift[1]
    out[..., 0] *= 255.0 / 100.0
    out[..., 1:] += 128.0
    cv2.imwrite(str(chained), cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR))
    return {"done": True, "cast": [round(float(x), 2) for x in cast], "target": [round(float(x), 2) for x in want],
            "shift": [round(float(x), 2) for x in shift], "dark": int(dark.sum())}


def match(chained: str | Path, reference: str | Path, strength: float = 0.8) -> dict:
    """Whole-frame colour match (Reinhard in Lab: per-channel mean and spread) of `chained` toward `reference` (clip 1's
    first frame: the shot's palette), mixed in at `strength`. Rewrites `chained` in place. The pose and the place stay
    (it is the same picture); only the colours move. The research's experiment C (docs/ltx_long_drift_research.md)."""
    img, ref = cv2.imread(str(chained)), cv2.imread(str(reference))
    if img is None or ref is None:
        raise FileNotFoundError(chained if img is None else reference)
    lab, rlab = _lab(img), _lab(ref)
    out = lab.copy()
    stats = {}
    for c in range(3):
        m, sd = float(lab[..., c].mean()), float(lab[..., c].std()) + 1e-6
        rm, rs = float(rlab[..., c].mean()), float(rlab[..., c].std()) + 1e-6
        moved = (lab[..., c] - m) * (rs / sd) + rm
        out[..., c] = lab[..., c] + float(strength) * (moved - lab[..., c])
        stats["Lab"[c]] = [round(m, 2), round(rm, 2)]
    out[..., 0] *= 255.0 / 100.0
    out[..., 1:] += 128.0
    cv2.imwrite(str(chained), cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR))
    return {"done": True, "method": "reinhard_lab", "strength": float(strength), "mean_before_after_ref": stats}
