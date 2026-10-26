"""Character sheet from a turntable clip (templates/turntable): views at exact angles + a face panel.

The turntable template moves the camera around the character at a constant speed, so the camera angle
of every frame is known: angle(i) = degrees * i / (frames - 1) for the 0-based frame i. The sheet has
the layout that the LTX Ingredients IC-LoRA was trained on: on black, a large face close-up on the
left, then full-body views of the same height.

    python -m cinestudio.charsheet <turntable.mp4> --out sheet.png [--degrees 360] [--views 0,45,90,180]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .refsheet import background_mask

VIEW_NAMES = {0: "front", 45: "three-quarter", 90: "side", 135: "three-quarter back", 180: "back", 270: "other side"}


def frame_for(angle: float, frames: int, degrees: float) -> int:
    """0-based frame index whose camera angle is closest to `angle`."""
    if not 0 <= angle <= degrees:
        raise ValueError(f"angle {angle} is outside the turntable range 0..{degrees}")
    return int(round(angle / degrees * (frames - 1)))


def read_frames(clip: str | Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(clip))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise ValueError(f"no frames in {clip}")
    return frames


def cutout(img: np.ndarray, tolerance: int = 18, pad: int = 10) -> np.ndarray:
    """The character on pure black, cropped to its bounding box (background found from the border)."""
    bg = background_mask(img, tolerance)
    out = img.copy()
    out[bg] = 0
    ys, xs = np.where(~bg)
    if not len(ys):
        raise ValueError("no character found in a view (the whole frame looks like background)")
    return out[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]


def assemble(views: list[np.ndarray], size: tuple[int, int] = (1536, 896), face_share: float = 0.33,
             face_width: int = 360, margin: int = 16, face: np.ndarray | None = None,
             common_scale: bool = True) -> np.ndarray:
    """Sheet: face close-up + all views side by side, standing on one ground line.

    face: a separate close-up image (e.g. made by the angle model); without it, the top `face_share`
    of the first view. common_scale=True: one scale for all views, so the character keeps the size it
    has in the source (right for a turntable clip, where the camera distance does not change).
    common_scale=False: every view is scaled to the same height (right for views made one by one,
    where each image can show the character at another size)."""
    w, h = size
    canvas = np.zeros((h, w, 3), np.uint8)
    head = face if face is not None else views[0][: max(1, int(views[0].shape[0] * face_share))]
    cols = np.where(head.max(axis=2).max(axis=0) > 0)[0]
    if len(cols):
        head = head[:, max(0, cols.min() - 8): cols.max() + 8]
    fw = face_width
    fh = int(head.shape[0] * fw / head.shape[1])
    if fh > h - 2 * margin:
        fh, fw = h - 2 * margin, int(head.shape[1] * (h - 2 * margin) / head.shape[0])
    head = cv2.resize(head, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
    y0 = (h - fh) // 2
    canvas[y0:y0 + fh, margin // 2:margin // 2 + fw] = head
    x = margin // 2 + fw + margin
    slot = (w - x - margin // 2) // len(views)
    body_h = h - 2 * margin
    if common_scale:
        common = min(body_h / max(v.shape[0] for v in views), (slot - 4) / max(v.shape[1] for v in views))
        scales = [common] * len(views)
        widths = [slot] * len(views)
    else:
        # the same height for every figure; each panel is as wide as its figure (a spread cape gets a wider
        # panel instead of a smaller figure); if the row does not fit, all heights shrink together
        avail = w - x - margin // 2
        gap = 8
        height = body_h
        need = sum(v.shape[1] * height / v.shape[0] for v in views) + gap * (len(views) - 1)
        if need > avail:
            height *= (avail - gap * (len(views) - 1)) / (need - gap * (len(views) - 1))
        scales = [height / v.shape[0] for v in views]
        widths = [int(v.shape[1] * s) + gap for v, s in zip(views, scales)]
        x += max(0, (avail - sum(widths)) // 2)
    for v, scale, slot_w in zip(views, scales, widths):
        v = cv2.resize(v, (max(1, int(v.shape[1] * scale)), max(1, int(v.shape[0] * scale))),
                       interpolation=cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA)
        oy = margin + body_h - v.shape[0]                  # feet on one ground line
        ox = x + (slot_w - v.shape[1]) // 2
        region = canvas[oy:oy + v.shape[0], ox:ox + v.shape[1]]
        np.copyto(region, v, where=v.max(axis=2, keepdims=True) > 0)
        x += slot_w
    return canvas


PANEL_GREY = (205, 205, 205)


def max_ratio(size: tuple[int, int] = (1536, 896), gutter: int = 12, fill: float = 0.94) -> float:
    """The largest height ratio a one-character sheet can draw: the space above its feet line / the template's body
    height (a cut-out's image height is the body height, and its soles never lie below the image's bottom)."""
    ph = size[1] - 2 * gutter
    full_h = ph * fill
    return (int((ph + full_h) / 2)) / full_h


def _sole(mask: np.ndarray) -> int:
    """The row just below the figure's lowest pixel (its soles) in a scaled figure mask; the cut-out's margin below the
    feet scales with the body, so the feet line is set by the figure, not by the picture's edge."""
    rows = np.where((mask[..., 0] if mask.ndim == 3 else mask).max(1) > 0.5)[0]
    return int(rows.max()) + 1 if rows.size else mask.shape[0]


def _ratio(r) -> float:
    """A height ratio for a sheet: a finite number in 0.2..2.5 (else 1.0, the template mannequin's height)."""
    try:
        r = float(r)
    except (TypeError, ValueError):
        return 1.0
    return r if 0.2 <= r <= 2.5 else 1.0


def assemble_panels(face: np.ndarray, views: list[np.ndarray], size: tuple[int, int] = (1536, 896),
                    gutter: int = 12, fill: float = 0.94, panel_bg: tuple[int, int, int] = PANEL_GREY,
                    face_fill: float = 0.98, body_ratio: float = 1.0, min_face: float = 0.3,
                    props: list[np.ndarray] | None = None, layout: dict | None = None) -> np.ndarray:
    """A Lightricks-style sheet (2026-10-22; the Ingredients 2.5 card and its example sheet): one row of panels on a
    black canvas with black gutters, each a light-grey panel with one image: the face close-up, then the full-body
    views at one height on one feet line, filling `fill` of the panel height. Panel widths follow the figures, the face
    panel keeps its own aspect; the row is scaled to fit the width. `face`/`views` come from cutout() (black around).
    `body_ratio` = the character's height / the template mannequin's (character.height_ratio; user, 2026-10-24:
    "standardize the sheets so they can properly reflect the difference in heights"): the bodies are drawn at that part
    of the full height on the same feet line, so every sheet has one scale. The face close-up keeps its size.
    `props` (prop_cutout() images: a run's prop pictures): a PROP COLUMN at the right end (PropColumn); the face panel
    gives way first, then the props down to PROP_MIN, and only then everything shrinks. `layout` (a dict) is filled
    with the prop boxes, their places and the scale of the bodies against the standard (1.0 = unchanged)."""
    w, h = size
    canvas = np.zeros((h, w, 3), np.uint8)
    ph = h - 2 * gutter
    items = [face] + list(views)
    fg = [~background_mask(it, 2) for it in items]              # the figure (black inside it stays figure)
    full_h = ph * fill                                          # the template mannequin's height on every sheet
    feet = gutter + int((ph + full_h) / 2)                      # one feet line for every height
    # the tallest body that fits above the feet line (Codex review: 1/fill did not fit): 1.14x the template at the
    # default size; a taller character is drawn at this limit (documented: max_ratio)
    body_h = full_h * min(_ratio(body_ratio), max_ratio(size, gutter, fill))
    scales = [ph * face_fill / face.shape[0]] + [body_h / v.shape[0] for v in views]
    widths = [int(it.shape[1] * sc) + 2 * gutter for it, sc in zip(items, scales)]
    body0 = scales[1] if len(scales) > 1 else 1.0
    col = PropColumn(props, ph, gutter) if props else None
    base = w - gutter * (len(items) + 1)
    avail = base - (col.width() if col else 0)
    if sum(widths) > avail:                                     # too wide: the FACE panel gives way first (the
        room = avail - sum(widths[1:]) - 2 * gutter             # bodies keep the one scale of every sheet), down to
        fs = max(ph * min_face / face.shape[0], room / face.shape[1])   # `min_face` of the height
        if fs < scales[0]:
            scales[0] = fs
            widths[0] = int(face.shape[1] * fs) + 2 * gutter
    if col and sum(widths) > avail:                             # then the props, down to PROP_MIN of the height
        col.shrink(sum(widths) - avail)
        avail = base - col.width()
    if sum(widths) > avail:                                     # still too wide: everything smaller together (this
        k = avail / sum(widths)                                 # sheet's scale is then smaller than the standard)
        scales = [sc * k for sc in scales]
        widths = [int(wd * k) for wd in widths]
    else:                                                       # spare width goes to the panels (wider grey room)
        extra = (avail - sum(widths)) // len(items)
        widths = [wd + extra for wd in widths]
    x = gutter + max(0, (avail - sum(widths)) // 2)
    for i, (it, m, sc, pw) in enumerate(zip(items, fg, scales, widths)):
        canvas[gutter:gutter + ph, x:x + pw] = panel_bg
        img = cv2.resize(it, (max(1, int(it.shape[1] * sc)), max(1, int(it.shape[0] * sc))),
                         interpolation=cv2.INTER_LANCZOS4 if sc > 1 else cv2.INTER_AREA)
        mk = cv2.resize(m.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)[..., None]
        oy = gutter + (ph - img.shape[0]) // 2 if i == 0 else feet - _sole(mk)   # the lowest figure row on the line
        ox = x + (pw - img.shape[1]) // 2
        roi = canvas[oy:oy + img.shape[0], ox:ox + img.shape[1]].astype(np.float32)
        canvas[oy:oy + img.shape[0], ox:ox + img.shape[1]] = (img * mk + roi * (1 - mk)).astype(np.uint8)
        x += pw + gutter
    if col:
        col.draw(canvas, x, gutter, panel_bg)
    if layout is not None:
        layout.update(body_scale=round(scales[1] / body0, 4) if len(scales) > 1 else 1.0,
                      face_px=[int(face.shape[1] * scales[0]), int(face.shape[0] * scales[0])],
                      **(col.info() if col else {}))
    return canvas


def assemble_groups(groups: list, size: tuple[int, int] = (1536, 896), gutter: int = 12, gap: int = 48,
                    fill: float = 0.94, panel_bg: tuple[int, int, int] = (0, 0, 0),
                    ratios: list[float] | None = None, props: list[np.ndarray] | None = None,
                    layout: dict | None = None) -> np.ndarray:
    """One row of full-body panels in GROUPS (one group per character; 2026-10-23, two characters in one shot): the
    figures on one feet line, black gutters, a wider `gap` between the groups. `groups` = lists of cutout() images.
    `ratios` = each character's height / the template mannequin's (character.height_ratio): the tallest group fills the
    height and the others keep their real size against it (a short character is drawn short, as in the layout).
    `props`/`layout`: a prop column at the right end, as in assemble_panels (no face here: the props give way first,
    down to PROP_MIN, then everything shrinks)."""
    w, h = size
    canvas = np.zeros((h, w, 3), np.uint8)
    ph = h - 2 * gutter
    items = [(gi, it) for gi, g in enumerate(groups) for it in g]
    full_h = ph * fill
    rs = [_ratio(r) for r in (ratios or [1.0] * len(groups))]
    if len(rs) != len(groups):
        raise ValueError("one height ratio per group")
    top = max(rs)
    feet = gutter + int((ph + full_h) / 2)
    scales = [full_h * rs[gi] / top / it.shape[0] for gi, it in items]
    widths = [int(it.shape[1] * sc) + 2 * gutter for (_, it), sc in zip(items, scales)]
    gaps = gap * (len(groups) - 1)
    body0 = scales[0]
    col = PropColumn(props, ph, gutter) if props else None
    base = w - gutter * (len(items) + 1) - gaps
    avail = base - (col.width() if col else 0)
    if col and sum(widths) > avail:
        col.shrink(sum(widths) - avail)
        avail = base - col.width()
    if sum(widths) > avail:
        k = avail / sum(widths)
        scales = [sc * k for sc in scales]
        widths = [int(wd * k) for wd in widths]
    else:
        extra = (avail - sum(widths)) // len(items)
        widths = [wd + extra for wd in widths]
    x = gutter + max(0, (avail - sum(widths)) // 2)
    for i, ((gi, it), sc, pw) in enumerate(zip(items, scales, widths)):
        if i and items[i - 1][0] != gi:
            x += gap
        canvas[gutter:gutter + ph, x:x + pw] = panel_bg
        img = cv2.resize(it, (max(1, int(it.shape[1] * sc)), max(1, int(it.shape[0] * sc))),
                         interpolation=cv2.INTER_LANCZOS4 if sc > 1 else cv2.INTER_AREA)
        m = (~background_mask(it, 2)).astype(np.float32)
        mk = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)[..., None]
        oy = feet - _sole(mk)
        ox = x + (pw - img.shape[1]) // 2
        roi = canvas[oy:oy + img.shape[0], ox:ox + img.shape[1]].astype(np.float32)
        canvas[oy:oy + img.shape[0], ox:ox + img.shape[1]] = (img * mk + roi * (1 - mk)).astype(np.uint8)
        x += pw + gutter
    if col:
        col.draw(canvas, x, gutter, panel_bg)
    if layout is not None:
        layout.update(body_scale=round(scales[0] / body0, 4), **(col.info() if col else {}))
    return canvas


# ---------------------------------------------------------------- prop panels (a run's prop pictures, 2026-10-25)
# User: "generate a ball as a prop ... and swap it out with the ball in the blender reference video". The Ingredients
# 2.5 card reads the sheet "to match characters, props, and setting" and its example sheets put extra items in their
# own panels; "bigger panels carry over better" (docs/style_consistency_research.md 2.7). So a run's prop picture goes
# into a COLUMN at the right end of the run's copy of the sheet: the column is split into two cells (top and bottom);
# one prop takes the BOTTOM cell, two props take both. A prop is drawn at a readable size, not at the body's scale (a
# 22 cm ball beside a 1.8 m body would be ~90 px). The sheet keeps its size; no text is drawn. Not A/B-tested yet.
PROP_CELL_FILL = 0.86          # a prop's longest side = this part of its cell (the cell = half the panel height)
PROP_MIN = 0.22                # the smallest prop side (part of the panel height) before the characters shrink
MAX_PROP_PANELS = 2
PROP_DARK_FLOOR = 16           # the darkest value inside a prop (refsheet.prepare's tolerance is 14)


def prop_cutout(img: np.ndarray) -> np.ndarray:
    """A prop picture on pure black, cropped to the prop with a small margin (a transparent PNG is put on black
    first; a plain light or dark background is found from the border, as for a character view)."""
    if img is None or img.size == 0:
        raise ValueError("an empty prop picture")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pad = max(4, int(0.03 * max(img.shape[:2])))
    if img.shape[2] == 4:                                      # alpha: on black
        a = img[..., 3].astype(np.float32) / 255.0
        img = (img[..., :3].astype(np.float32) * a[..., None]).astype(np.uint8)
        if (a < 0.5).mean() > 0.02:                            # a real cut-out: its own alpha is the outline
            ys, xs = np.where(a >= 0.5)
            if not len(ys):
                raise ValueError("the prop picture is fully transparent")
            return img[max(0, ys.min() - pad):ys.max() + 1 + pad, max(0, xs.min() - pad):xs.max() + 1 + pad]
    h, w = img.shape[:2]                                       # a dark backdrop: a tight tolerance (a ball's shaded
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, w - 1]])   # underside is nearly black: 18 ate it)
    tol = 6 if float(np.median(border.max(axis=1))) < 40 else 18
    try:
        return cutout(img, tolerance=tol, pad=pad)
    except ValueError:
        raise ValueError("no prop found in the picture (it looks like plain background)") from None


class PropColumn:
    """The prop column of a run's sheet: its width, how it gives way, where each prop goes (see PROP_CELL_FILL)."""

    def __init__(self, props: list[np.ndarray], ph: int, gutter: int):
        if not 1 <= len(props) <= MAX_PROP_PANELS:
            raise ValueError(f"a sheet takes 1 to {MAX_PROP_PANELS} prop pictures")
        self.props, self.ph, self.g = list(props), ph, gutter
        cells = max(2, len(props))
        self.cell_h = (ph - gutter * (cells - 1)) / cells
        self.rows = list(range(cells - len(props), cells))      # one prop: the bottom cell
        self.side = self.cell_h * PROP_CELL_FILL
        self.min_side = min(self.side, ph * PROP_MIN)
        self.boxes: list = []

    def _inner(self) -> int:
        return max(max(1, int(p.shape[1] * self.side / max(p.shape[:2]))) for p in self.props)

    def panel_w(self) -> int:
        return self._inner() + 2 * self.g

    def width(self) -> int:
        """The width the column takes from the row: its panel and the gutter before it."""
        return self.panel_w() + self.g

    def shrink(self, need: float) -> None:
        """Give back `need` px of width, down to the smallest prop."""
        inner = self._inner()
        self.side = max(self.min_side, self.side * max(1.0, inner - need) / inner)

    def places(self) -> list[str]:
        return [("top-right", "bottom-right")[r] for r in self.rows]

    def draw(self, canvas: np.ndarray, x: int, top: int, panel_bg=(0, 0, 0)) -> list:
        pw = self.panel_w()
        self.boxes = []
        for p, r in zip(self.props, self.rows):
            cy = top + r * (self.cell_h + self.g)
            if tuple(panel_bg) != (0, 0, 0):
                canvas[int(cy):int(cy + self.cell_h), x:x + pw] = panel_bg
            sc = self.side / max(p.shape[:2])
            img = cv2.resize(p, (max(1, int(p.shape[1] * sc)), max(1, int(p.shape[0] * sc))),
                             interpolation=cv2.INTER_LANCZOS4 if sc > 1 else cv2.INTER_AREA)
            m = (~background_mask(p, 2)).astype(np.float32)
            mk = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)[..., None]
            # the run's sheet step blacks out everything within 14 of black that touches the border (refsheet.prepare):
            # a ball's shaded underside lost a ragged piece. Inside the prop, near-black is lifted to PROP_DARK_FLOOR.
            inside = mk[..., 0] > 0.5
            lift = np.clip(PROP_DARK_FLOOR - img.max(axis=2).astype(np.int16), 0, None)
            img = np.where(inside[..., None], np.clip(img.astype(np.int16) + lift[..., None], 0, 255), img).astype(np.uint8)
            oy, ox = int(cy + (self.cell_h - img.shape[0]) / 2), x + (pw - img.shape[1]) // 2
            roi = canvas[oy:oy + img.shape[0], ox:ox + img.shape[1]].astype(np.float32)
            canvas[oy:oy + img.shape[0], ox:ox + img.shape[1]] = (img * mk + roi * (1 - mk)).astype(np.uint8)
            self.boxes.append([int(ox), int(oy), int(ox + img.shape[1]), int(oy + img.shape[0])])
        return self.boxes

    def info(self) -> dict:
        return {"prop_boxes": self.boxes, "prop_places": self.places(), "prop_side": int(self.side)}


def add_prop_column(sheet: np.ndarray, props: list[np.ndarray], gutter: int = 12,
                    layout: dict | None = None) -> np.ndarray:
    """A prop column on a sheet that cannot be drawn again from its views (an imported or restyled sheet): the
    sheet's content (its non-black columns) moves left, scaled down only when the column needs the room (after the
    props gave way to PROP_MIN). Same size as `sheet`."""
    h, w = sheet.shape[:2]
    col = PropColumn(props, h - 2 * gutter, gutter)
    cols = np.where(sheet.max(axis=2).max(axis=0) > 25)[0]
    x0, x1 = (int(cols.min()), int(cols.max()) + 1) if cols.size else (0, w)
    content = sheet[:, x0:x1]
    room = w - 2 * gutter - col.width()
    if content.shape[1] > room:
        col.shrink(content.shape[1] - room)
        room = w - 2 * gutter - col.width()
    k = min(1.0, room / max(1, content.shape[1]))
    canvas = np.zeros_like(sheet)
    if k < 1.0:
        content = cv2.resize(content, (max(1, int(content.shape[1] * k)), max(1, int(h * k))),
                             interpolation=cv2.INTER_AREA)
    oy, ox = (h - content.shape[0]) // 2, gutter + (room - content.shape[1]) // 2
    canvas[oy:oy + content.shape[0], ox:ox + content.shape[1]] = content
    col.draw(canvas, w - gutter - col.panel_w(), gutter)
    if layout is not None:
        layout.update(body_scale=round(k, 4), **col.info())
    return canvas


def pick_frames(angles, frames: int, degrees: float | None = None, azimuths: list[float] | None = None) -> dict:
    """{angle: 0-based frame}. With `azimuths` (the measured camera angle of every frame, from the
    run's passes) the frame whose angle is closest; else the nominal constant-speed mapping."""
    if azimuths is not None:
        if len(azimuths) != frames:
            raise ValueError(f"{len(azimuths)} camera angles for {frames} frames: the clip was trimmed or changed")
        out = {}
        for a in angles:
            i = min(range(frames), key=lambda k: abs(azimuths[k] - a))
            if abs(azimuths[i] - a) > 5:
                raise ValueError(f"no frame near {a} degrees (closest: {azimuths[i]:.1f} at frame {i})")
            out[a] = i
        return out
    if degrees is None:
        raise ValueError("give the turntable degrees or the measured camera angles")
    return {a: frame_for(a, frames, degrees) for a in angles}


def make_sheet(clip: str | Path, out: str | Path, degrees: float | None = 360.0, angles: tuple[float, ...] = (0, 45, 90, 180),
               tolerance: int = 18, azimuths: list[float] | None = None, expected_frames: int | None = None) -> dict:
    frames = read_frames(clip)
    if expected_frames is not None and len(frames) != expected_frames:
        raise ValueError(f"{clip}: decoded {len(frames)} frames, expected {expected_frames} (damaged or trimmed clip)")
    picks = pick_frames(angles, len(frames), degrees, azimuths)
    views = [cutout(frames[i], tolerance) for i in picks.values()]
    sheet = assemble(views)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    for a, i in picks.items():
        cv2.imwrite(str(out.with_name(f"{out.stem}_view_{int(a):03d}.png")), frames[i])
    return {"sheet": str(out), "frames": len(frames), "angles_from": "measured" if azimuths is not None else "nominal",
            "views": [{"angle": a, "frame": i, "name": VIEW_NAMES.get(int(a), f"{a:g} deg"),
                       **({"camera_deg": azimuths[i]} if azimuths is not None else {})} for a, i in picks.items()]}


def from_run(run_id: str, out: str | Path, angles: tuple[float, ...] = (0, 45, 90, 180)) -> dict:
    """Sheet from a finished turntable run: its final clip, its frame count and the camera angles
    that render_passes measured for exactly those frames."""
    from . import runner
    st = runner.status(run_id)
    if st["status"] != "done":
        raise ValueError(f"run {run_id} is {st['status']}, not done")
    clip = st["steps"]["clip"]["outputs"]
    az = clip.get("camera_azimuth_deg")
    if not az:
        raise ValueError(f"run {run_id} has no measured camera angles (made before pass format 2, or no rig)")
    return make_sheet(st["final"], out, degrees=None, angles=angles, azimuths=az, expected_frames=clip["frames"])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--degrees", type=float, default=360.0)
    ap.add_argument("--views", default="0,45,90,180", help="camera angles of the views, in degrees")
    ap.add_argument("--run", action="store_true", help="CLIP is a run id: use its measured camera angles")
    a = ap.parse_args(argv)
    views = tuple(float(v) for v in a.views.split(","))
    print(json.dumps(from_run(a.clip, a.out, views) if a.run else make_sheet(a.clip, a.out, a.degrees, views)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


def palette_check(front: np.ndarray, views: dict[str, np.ndarray], colours: int = 6, min_share: float = 0.04,
                  keep: float = 0.3, radius: float = 40.0) -> dict:
    """Do the other views keep the front view's main colours? A part that disappears in a view (a cape,
    a scarf) shows as a main colour whose share drops a lot. Returns per view the colours that dropped
    below `keep` x their front share. Some drops are expected (e.g. a face mask is not visible from the
    back), so this is a warning list for a person or an agent, not a pass/fail gate."""
    def pixels(img):
        p = img.reshape(-1, 3)
        return p[p.max(axis=1) > 0].astype(np.float32)
    fp = pixels(front)
    if len(fp) < colours * 10:
        return {"ok": True, "note": "front view too small to check"}
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(fp, colours, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(labels.ravel(), minlength=colours) / len(fp)
    main = [(i, float(share[i])) for i in range(colours) if share[i] >= min_share]
    out, warnings = {}, []
    for name, img in views.items():
        vp = pixels(img)
        if not len(vp):
            out[name] = {"error": "empty view"}
            continue
        d = np.linalg.norm(vp[:, None, :] - centers[None, :, :], axis=2)
        near = d.min(axis=1) < radius
        lab = d.argmin(axis=1)
        vshare = np.bincount(lab[near], minlength=colours) / len(vp)
        dropped = []
        for i, s in main:
            if vshare[i] < s * keep:
                bgr = [int(x) for x in centers[i]]
                dropped.append({"bgr": bgr, "front_share": round(s, 3), "view_share": round(float(vshare[i]), 3)})
        out[name] = {"dropped": dropped}
        if dropped:
            warnings.append(f"{name}: {len(dropped)} main colour(s) of the front view mostly missing")
    return {"ok": not warnings, "warnings": warnings, "views": out,
            "front_colours": [{"bgr": [int(x) for x in centers[i]], "share": round(s, 3)} for i, s in main]}
