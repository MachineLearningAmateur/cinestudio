"""The LOOK IMAGE keeps the background (2026-10-26, docs/look_image_research.md; user: "preserve everything ideally
without using this stitch method cause it can introduce halo issues shadow issues").

A look image is a Qwen-Image-Edit-2511 edit of a Blender frame (the mannequin -> the character). The plain edit
redrew the WHOLE frame: a final run lost the hoop, a printed banner and both EXIT signs, and the next video followed
it. Three changes, all on the look step only (character edits and set redraws are unchanged):

1. GEOMETRY: the frame goes to Qwen at an encoder-stable size of its own shape (background.edit_layout: 768x448 ->
   1344x784) and the graph's FluxKontextImageScale centre crop is taken out (background._no_kontext_crop): before,
   every look was drawn ~0.6% narrower and ~1% shorter than the layout.
2. PIN: the figure's mask goes in as a latent NOISE MASK (core SetLatentNoiseMask + DifferentialDiffusion): at every
   sampling step the frame's own latent is put back outside the mask, so the room cannot change, while the model
   still sees the whole frame (light, contact shadow). No pixel paste. The mask = the mannequin + kit silhouette
   dilated by ~3% of the frame height (value 1), a soft ring (1 -> 0.3 -> 0) and a soft contact-shadow ellipse under
   the feet (0.35): with DifferentialDiffusion a soft value can change only late in the schedule (tone, not shape).
3. GATE: check() compares the look with the layout frame outside the (wider) figure area: ECC alignment, SSIM, edge
   recall and lost tiles; the run makes the look again with the next seed when it fails (at most 3 tries).
4. OFFSET: the kept look is moved a few pixels (offset()) before LTX sees it. A look whose background lines up
   with the layout pixel for pixel made the whole video ~35% darker (diagnosis 2026-10-26, preview clip 1, seed
   303, .cinestudio/gauntlet/look_pin_ab: aligned 27.9 / 27.0 (the same look twice), the old unaligned look 42.7,
   aligned + 6/4 px move 39.9, aligned + resize 39.5, aligned + blur 30.5: the alignment, not the VAE texture).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CORE_DILATE = 0.03          # of the frame height: room for a costume wider than the mannequin + kit
RING_PX = 28                # the soft ring outside the core (at the edit size; >= 16 px = one transformer patch)
SHADOW_VALUE = 0.35         # the contact-shadow ellipse under the feet
GATE_EXCLUDE = 0.04         # the gate ignores the figure dilated by this part of the frame height
MAX_TRIES = 3
GATE_WIDTH = 768            # the gate measures at the video's preview width
OFFSET = (6 / 1344, 4 / 784)   # the move of the kept look (part of the width, height): 6 px right, 4 px down
                               # at the edit size = the move that measured bright (docs: point 4 above)


def offset(look: str | Path) -> tuple[int, int]:
    """Move the look image OFFSET to the right and down, in place (the uncovered left columns and top rows repeat
    the edge). The pixels are moved, not resampled. -> (dx, dy) in pixels."""
    img = cv2.imread(str(look), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"not a picture: {look}")
    h, w = img.shape[:2]
    dx, dy = max(1, round(OFFSET[0] * w)), max(1, round(OFFSET[1] * h))
    moved = np.empty_like(img)
    moved[dy:, dx:] = img[:h - dy, :w - dx]
    moved[dy:, :dx] = img[:h - dy, :1]
    moved[:dy] = moved[dy:dy + 1]
    if not cv2.imwrite(str(look), moved):
        raise OSError(f"could not write {look}")
    return dx, dy


def edit_size(w: int, h: int) -> tuple[int, int, tuple[int, int, int, int]]:
    from .background import edit_layout
    return edit_layout(w, h)


def _figure(mask_png: str | Path) -> np.ndarray:
    m = cv2.imread(str(mask_png), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"no figure mask {mask_png}")
    if m.ndim == 3:
        m = m[..., 3] if m.shape[2] == 4 and m[..., 3].max() > 0 and m[..., :3].max() == 0 else m[..., :3].max(axis=2)
    return m > 127


def _fit(img: np.ndarray, ew: int, eh: int, box: tuple[int, int, int, int], interp) -> np.ndarray:
    x0, y0, iw, ih = box
    out = np.zeros((eh, ew) + img.shape[2:], img.dtype)
    out[y0:y0 + ih, x0:x0 + iw] = cv2.resize(img, (iw, ih), interpolation=interp)
    return out


def prepare(frame: str | Path, mask_png: str | Path, work: Path, tag: str) -> dict:
    """The frame at the edit size and its soft figure mask (PNG, white = may change). -> {frame, mask, size, box}."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(frame), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"no frame {frame}")
    h, w = img.shape[:2]
    ew, eh, box = edit_size(w, h)
    big = _fit(img, ew, eh, box, cv2.INTER_LANCZOS4)
    fig = _fit(_figure(mask_png).astype(np.uint8) * 255, ew, eh, box, cv2.INTER_NEAREST) > 127
    soft = soft_mask(fig, eh)
    f_out, m_out = work / f"{tag}_frame.png", work / f"{tag}_mask.png"
    cv2.imwrite(str(f_out), big)
    cv2.imwrite(str(m_out), np.clip(soft * 255 + 0.5, 0, 255).astype(np.uint8))
    return {"frame": str(f_out), "mask": str(m_out), "size": [ew, eh], "box": list(box)}


def soft_mask(fig: np.ndarray, frame_h: int) -> np.ndarray:
    """The figure (bool) -> a float mask 0..1: a dilated core (1), a soft ring (1 -> 0.3 -> 0) and a contact-shadow
    ellipse under the feet (SHADOW_VALUE). Every figure pixel is inside the core (no white clay edge can stay)."""
    out = np.zeros(fig.shape, np.float32)
    if not fig.any():
        return out
    r = max(2, int(round(CORE_DILATE * frame_h)))
    core = cv2.dilate(fig.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
    t = np.clip(dist / RING_PX, 0, 1)
    ring = np.where(t < 0.5, 1.0 - 1.4 * t, 0.6 * (1.0 - t))   # 1 at the core, 0.3 halfway, 0 at RING_PX
    out = np.where(core, 1.0, np.where(dist <= RING_PX, ring, 0.0)).astype(np.float32)
    ys, xs = np.nonzero(fig)
    feet_y = ys.max()
    low = ys > feet_y - 0.08 * (ys.max() - ys.min() + 1)       # the lowest 8% of the figure = the feet
    fx0, fx1 = xs[low].min(), xs[low].max()
    cx, w = (fx0 + fx1) / 2, max(fx1 - fx0, 8) * 0.7
    shadow = np.zeros_like(out)
    cv2.ellipse(shadow, (int(cx), int(feet_y)), (int(w), max(4, int(0.06 * frame_h))), 0, 0, 360, SHADOW_VALUE, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 6)
    return np.maximum(out, shadow)


def graph_edit(mask_node: str = "look_mask"):
    """A change of the built qwenedit edit graph: no Kontext centre crop; the sampler's latent gets the noise mask
    (the uploaded mask picture -> ImageToMask -> SetLatentNoiseMask on the VAEEncode of the frame) and its model goes
    through DifferentialDiffusion. -> a function(graph) for Run._comfy(graph_edit=...)."""
    def edit(graph: dict) -> None:
        from .background import _no_kontext_crop
        _no_kontext_crop(graph)
        ks = graph["ks"]["inputs"]
        graph[mask_node] = {"class_type": "LoadImage", "inputs": {"image": "MASK.png"}}
        graph[mask_node + "_m"] = {"class_type": "ImageToMask", "inputs": {"image": [mask_node, 0], "channel": "red"}}
        graph["look_noise"] = {"class_type": "SetLatentNoiseMask",
                               "inputs": {"samples": ks["latent_image"], "mask": [mask_node + "_m", 0]}}
        graph["look_dd"] = {"class_type": "DifferentialDiffusion", "inputs": {"model": ks["model"], "strength": 1.0}}
        ks["latent_image"] = ["look_noise", 0]
        ks["model"] = ["look_dd", 0]
    return edit


def check(look: str | Path, frame: str | Path, mask_png: str | Path) -> dict:
    """Does the look keep the layout frame's background? Outside the figure (dilated by GATE_EXCLUDE): ECC affine fit
    of the look to the frame (shift, scale), then SSIM, edge recall (layout edges found within 3 px) and lost tiles
    (tiles with many layout edges of which < 40% are kept). -> {ok, ssim, edge_recall, lost_tiles, shift_px, scale,
    reasons}. Thresholds from 22 past looks (docs/look_image_research.md option 3)."""
    ref = cv2.imread(str(frame), cv2.IMREAD_COLOR)
    img = cv2.imread(str(look), cv2.IMREAD_COLOR)
    h0, w0 = ref.shape[:2]
    w, h = GATE_WIDTH, int(round(GATE_WIDTH * h0 / w0))          # measured at video scale (the tolerances are px)
    ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    fig = _figure(mask_png)
    if fig.shape != (h, w):
        fig = cv2.resize(fig.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    r = max(2, int(round(GATE_EXCLUDE * h)))
    excl = cv2.dilate(fig.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    keep = (~excl).astype(np.uint8)
    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(cv2.GaussianBlur(g_ref, (0, 0), 1.5), cv2.GaussianBlur(g_img, (0, 0), 1.5),
                                       warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), keep, 5)
    except cv2.error:
        pass
    aligned = cv2.warpAffine(img, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
    g_al = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    scale = float(np.sqrt(abs(np.linalg.det(warp[:, :2]))))
    c = np.array([w / 2, h / 2, 1.0], np.float32)
    moved = warp @ c                                              # the centre's displacement (not the corner's)
    shift = float(np.hypot(moved[0] - c[0], moved[1] - c[1]))
    ssim = _ssim(g_ref, g_al, keep > 0)
    e_ref = cv2.Canny((g_ref * 255).astype(np.uint8), 60, 140) > 0
    e_img = cv2.dilate((cv2.Canny((g_al * 255).astype(np.uint8), 60, 140) > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    e_ref &= keep > 0
    recall = float((e_ref & e_img).sum() / max(e_ref.sum(), 1))
    tile = max(16, int(round(64 * w / 1328)))
    need = max(8, int(round(60 * (tile / 64) ** 2)))
    lost = 0
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            t = e_ref[y:y + tile, x:x + tile]
            if t.sum() >= need and (t & e_img[y:y + tile, x:x + tile]).sum() < 0.4 * t.sum():
                lost += 1
    textured = e_ref.sum() / max(keep.sum(), 1) > 0.12        # foliage and the like: softer SSIM / recall bars
    # limits measured 2026-10-26 at 768 px on 4 old looks (made WITH the old centre crop): 3 that kept the room
    # (SSIM 0.957-0.966, recall 0.83-0.86, 15-21 lost tiles, 3.3 px) and 1 that lost the hoop, a banner and the EXIT
    # signs (SSIM 0.770, recall 0.64, 41 tiles, 20.8 px); pinned looks should be far better
    reasons = []
    if lost >= 30:
        reasons.append(f"{lost} background tiles lost")
    if shift > 8:
        reasons.append(f"shifted {shift:.1f} px")
    if abs(scale - 1) > 0.025:
        reasons.append(f"scaled {scale:.3f}")
    if ssim < (0.80 if textured else 0.90):
        reasons.append(f"SSIM {ssim:.3f}")
    if recall < (0.60 if textured else 0.75):
        reasons.append(f"edge recall {recall:.2f}")
    return {"ok": not reasons, "ssim": round(ssim, 4), "edge_recall": round(recall, 3), "lost_tiles": lost,
            "shift_px": round(shift, 2), "scale": round(scale, 4), "textured": bool(textured), "reasons": reasons}


def _ssim(a: np.ndarray, b: np.ndarray, where: np.ndarray) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), 1.5)              # noqa: E731
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(s[where].mean()) if where.any() else 1.0
