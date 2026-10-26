"""Control videos from raw Blender passes (see bpy_jobs/render_passes.py).

    python -m cinestudio.passes <pass_dir> [--kinds pose,depth,canny,canny_env,pose_env]

Kinds:
  pose       OpenPose (COCO-18) skeleton on black, drawn from the rig's bones (exact, no detector)
  depth      8-bit depth, near = white, one near/far range for the whole shot (no flicker)
  canny      edges of the whole render
  canny_env  edges of the set only: the character's area is removed, so the AI is not bound to the
             placeholder's outline and can show your character with its own shape
  pose_env   canny_env in grey + the pose skeleton: camera motion from the set, the character from the pose
Each kind becomes <pass_dir>/controls/<kind>.mp4 (lossless, libx264rgb) plus a contact sheet.

With `size` (the generation size, e.g. 768x448), every control is drawn at that size: the render is
first made smaller (cover + centre crop, area filter: a clean, anti-aliased picture), then the edges
are found and the skeleton is drawn. Edges and skeleton lines are then sharp, with their normal
widths. (Drawn at the render size and made smaller later, a 1-pixel edge became a faint 0.6-pixel line.)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from .video import ffmpeg

CANNY = (3, 40, 100)                 # blur, low, high (same values for every frame: no flicker)
DEPTH_QUANTILES = (0.01, 0.99)
MASK_DILATE = 15                     # px removed around the character in canny_env
# OpenPose COCO-18 limbs and colours (BGR), as used by the common pose ControlNets
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
         (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(0, 0, 255), (0, 85, 255), (0, 170, 255), (0, 255, 255), (0, 255, 170), (0, 255, 85), (0, 255, 0),
          (85, 255, 0), (170, 255, 0), (255, 255, 0), (255, 170, 0), (255, 85, 0), (255, 0, 0), (255, 0, 85),
          (255, 0, 170), (255, 0, 255), (170, 0, 255), (85, 0, 255)]


# OpenPose hand-21: wrist -> 4 points per finger; one rainbow colour per edge, joints as small dots
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
HAND_COLORS = [tuple(int(c) for c in cv2.cvtColor(np.uint8([[[int(180 * i / len(HAND_EDGES)), 255, 255]]]),
                                                  cv2.COLOR_HSV2BGR)[0, 0]) for i in range(len(HAND_EDGES))]
HAND_DOT = (255, 0, 0)                 # BGR blue, as in the OpenPose hand drawing


def load(pass_dir: Path) -> dict:
    info = json.loads((pass_dir / "passes.json").read_text(encoding="utf-8"))
    info["rgb"] = sorted((pass_dir / "rgb").glob("rgb_*.png"))
    info["mask"] = sorted((pass_dir / "mask").glob("mask_*.png")) if (pass_dir / "mask").exists() else []
    kp = pass_dir / "keypoints.json"
    info["keypoints"] = json.loads(kp.read_text(encoding="utf-8")) if kp.exists() else None
    if len(info["rgb"]) != info["frames"]:
        raise ValueError(f"{pass_dir}: {len(info['rgb'])} RGB frames, passes.json says {info['frames']}")
    if info["mask"] and len(info["mask"]) != info["frames"]:
        raise ValueError(f"{pass_dir}: {len(info['mask'])} mask frames, passes.json says {info['frames']}")
    if info["keypoints"] and len(info["keypoints"]["frames"]) != info["frames"]:
        raise ValueError(f"{pass_dir}: keypoints for {len(info['keypoints']['frames'])} frames, passes.json says {info['frames']}")
    for side, frames in ((info["keypoints"] or {}).get("hands") or {}).items():
        if side != "format" and len(frames) != info["frames"]:
            raise ValueError(f"{pass_dir}: hand {side} points for {len(frames)} frames, passes.json says {info['frames']}")
    return info


def canny(img: np.ndarray) -> np.ndarray:
    b, lo, hi = CANNY
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (b, b), 0)
    return cv2.Canny(g, lo, hi, L2gradient=True)


def mask_of(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return (m[:, :, 3] if m.ndim == 3 and m.shape[2] == 4 else m) > 127


def draw_pose(points: list, w: int, h: int, hands: list | None = None) -> np.ndarray:
    """Body skeleton (COCO-18) and, if given, hand skeletons (lists of 21 points) on black."""
    img = np.zeros((h, w, 3), np.uint8)
    width = max(2, round(w / 200))
    for i, (a, b) in enumerate(LIMBS):
        pa, pb = points[a], points[b]
        if pa[2] and pb[2]:
            cv2.line(img, (round(pa[0]), round(pa[1])), (round(pb[0]), round(pb[1])), COLORS[i], width * 2, cv2.LINE_AA)
    for i, p in enumerate(points):
        if p[2]:
            cv2.circle(img, (round(p[0]), round(p[1])), width * 2, COLORS[i], -1, cv2.LINE_AA)
    hw = max(1, round(w / 640))
    for hand in hands or []:
        for i, (a, b) in enumerate(HAND_EDGES):
            pa, pb = hand[a], hand[b]
            if pa[2] and pb[2]:
                cv2.line(img, (round(pa[0]), round(pa[1])), (round(pb[0]), round(pb[1])), HAND_COLORS[i], hw * 2, cv2.LINE_AA)
        for p in hand:
            if p[2]:
                cv2.circle(img, (round(p[0]), round(p[1])), hw * 2, HAND_DOT, -1, cv2.LINE_AA)
    return img


# The standard pose drawing of comfyui_controlnet_aux (DWPose node; util.draw_bodypose / draw_handpose, Apache-2.0),
# ported here so the run draws it without ComfyUI's Python. Colours are given in RGB there; this canvas is BGR.
STD_LIMBS = [(2, 3), (2, 6), (3, 4), (4, 5), (6, 7), (7, 8), (2, 9), (9, 10), (10, 11), (2, 12), (12, 13), (13, 14),
             (2, 1), (1, 15), (15, 17), (1, 16), (16, 18)]
STD_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
              (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
              (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]
STD_HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
                  (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def draw_pose_standard(points: list, w: int, h: int, hands: list | None = None) -> np.ndarray:
    """The skeleton as the DWPose node draws it: thin ellipse limbs (4 px) in 60% colours, 4 px joint dots, thin
    rainbow hand lines. For characters with standard human proportions (body type "standard"). Hands are drawn in the
    given order (a later hand on top); the node draws the left hand, then the right."""
    import colorsys
    import math
    img = np.zeros((h, w, 3), np.uint8)
    for (a, b), rgb in zip(STD_LIMBS, STD_COLORS):
        pa, pb = points[a - 1], points[b - 1]
        if not (pa and pb and pa[2] and pb[2]):
            continue
        X, Y = np.array([pa[1], pb[1]], float), np.array([pa[0], pb[0]], float)
        length = float(np.hypot(X[0] - X[1], Y[0] - Y[1]))
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        poly = cv2.ellipse2Poly((int(Y.mean()), int(X.mean())), (int(length / 2), 4), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(img, poly, [int(c * 0.6) for c in rgb[::-1]])
    for p, rgb in zip(points, STD_COLORS):
        if p and p[2]:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, rgb[::-1], thickness=-1)
    for hand in hands or []:
        if not hand:
            continue
        for ie, (e1, e2) in enumerate(STD_HAND_EDGES):
            k1, k2 = hand[e1], hand[e2]
            if not (k1 and k2 and k1[2] and k2[2]):
                continue
            x1, y1, x2, y2 = int(k1[0]), int(k1[1]), int(k2[0]), int(k2[1])
            if min(x1, y1, x2, y2) > 0:
                r, g, b = colorsys.hsv_to_rgb(ie / len(STD_HAND_EDGES), 1.0, 1.0)
                cv2.line(img, (x1, y1), (x2, y2), (b * 255, g * 255, r * 255), thickness=2)
        for k in hand:
            if k and k[2] and int(k[0]) > 0 and int(k[1]) > 0:
                cv2.circle(img, (int(k[0]), int(k[1])), 4, (255, 0, 0), thickness=-1)
    return img


BODY_TYPES = ("chibi", "standard")   # chibi: our thick skeleton (big-head characters); standard: the DWPose node's


def control_name(kind: str, body: str = "chibi") -> str:
    """File name of a drawn control: pose_env.mp4 (chibi, as before) or pose_env_standard.mp4."""
    return f"{kind}.mp4" if body == "chibi" else f"{kind}_{body}.mp4"


def hands_at(kp: dict | None, i: int) -> list:
    """The hand skeletons of frame i (empty if the rig has no finger bones)."""
    if not kp or "hands" not in kp:
        return []
    return [kp["hands"][s][i] for s in ("R", "L") if s in kp["hands"]]


def over(skeleton: np.ndarray, base: np.ndarray) -> np.ndarray:
    """The skeleton's drawn pixels on top of a base picture (both BGR, the same size)."""
    return np.where(skeleton.any(axis=2, keepdims=True), skeleton, base)


PEOPLE_DIFF = 18       # a pixel that differs this much between two characters' alone-renders is a character or its shadow


def people_mask(alone: list[np.ndarray]) -> np.ndarray:
    """Where the characters (and their shadows) are, from the renders of each character ALONE (render_people):
    each one shows the same set, so the set is equal in all of them and only the characters differ. Needs 2+."""
    m = np.zeros(alone[0].shape[:2], bool)
    for a in range(len(alone)):
        for b in range(a + 1, len(alone)):
            m |= cv2.absdiff(alone[a], alone[b]).max(axis=2) > PEOPLE_DIFF
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(m, np.ones((MASK_DILATE, MASK_DILATE), np.uint8)) > 0


def env_grey(edges: np.ndarray) -> np.ndarray:
    """Set edges in grey (60%), the base of pose_env: the skeleton's colours stay clear on top."""
    return cv2.cvtColor((edges * 0.6).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def people_env(pass_dir: Path, people_dir: Path, size: tuple[int, int]):
    """Per frame: the set edges of the shot render without the characters' areas (pose_env for several characters).
    The edges move with the camera, so the model sees that the CAMERA turns (with a skeleton on black only, LTX kept
    the background still and turned the characters round instead)."""
    info = load(pass_dir)
    people = json.loads((people_dir / "people.json").read_text())
    t = fit_transform(info["width"], info["height"], size)
    n = len(people["characters"])
    for i in range(info["frames"]):
        f = people["frame_start"] + i
        edges = canny(fit_image(cv2.imread(str(info["rgb"][i])), t))
        alone = [cv2.imread(str(people_dir / "iso" / str(c) / f"rgb_{f:04d}.png")) for c in range(n)]
        if any(a is None for a in alone):
            raise FileNotFoundError(f"missing alone-render of frame {f} in {people_dir / 'iso'}")
        # the alone-renders are made at a rounded percent of the shot: cover + cut them like the main render
        alone = [fit_image(a, fit_transform(a.shape[1], a.shape[0], size)) for a in alone]
        edges[people_mask(alone)] = 0
        yield env_grey(edges)


def depth_frames(pass_dir: Path):
    """8-bit depth frames (BGR), one near/far range for the whole shot. The file is read from disk frame by frame
    (a 360-frame shot at 1536x896 is about 2 GB of float32) and the range comes from a sample of it."""
    depth = np.load(pass_dir / "depth_m.npy", mmap_mode="r")
    sample = np.asarray(depth[::max(1, len(depth) // 60), ::4, ::4], dtype=np.float64)
    valid = sample[np.isfinite(sample) & (sample > 0)]
    near, far = np.quantile(valid, DEPTH_QUANTILES) if valid.size else (0.1, 10.0)
    return _DepthFrames(depth, float(near), float(far))


class _DepthFrames:
    """A list-like view: frame i is made when it is read."""

    def __init__(self, depth, near: float, far: float):
        self.depth, self.near, self.far = depth, near, far

    def __len__(self) -> int:
        return len(self.depth)

    def __getitem__(self, i: int) -> np.ndarray:
        z = np.asarray(self.depth[i], dtype=np.float32)
        z = np.where(np.isfinite(z), z, 0.0)              # inf / NaN = no surface
        v = 255 * (1 - np.clip((z - self.near) / max(1e-6, self.far - self.near), 0, 1))
        v[z <= 0] = 0                     # no surface = far
        return cv2.cvtColor(np.round(v).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def write_video(frames: list[np.ndarray], dst: Path, fps: float) -> None:
    """Encode to a temporary file, check the frame count, then move it into place: a stopped encode
    never leaves a short video under the final name (a later run would take it as finished)."""
    from .video import probe
    dst = Path(dst)
    tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.tmp{dst.suffix}")
    h, w = frames[0].shape[:2]
    p = subprocess.Popen([ffmpeg(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264rgb", "-crf", "0", "-preset", "veryfast", str(tmp)],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {dst}")
    got = probe(tmp).frames
    if got != len(frames):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{dst.name}: encoded {got} frames, expected {len(frames)}")
    os.replace(tmp, dst)


def fit_transform(w: int, h: int, size: tuple[int, int] | None) -> tuple[float, float, float, int, int]:
    """(scale, x offset, y offset, out w, out h) that covers `size` with a w x h picture and cuts the
    overflow equally on both sides (the same rule as video.slice_control: never stretched)."""
    if not size:
        return 1.0, 0.0, 0.0, w, h
    W, H = size
    s = max(W / w, H / h)
    return s, (w * s - W) / 2, (h * s - H) / 2, W, H


def fit_image(img: np.ndarray, t: tuple, interp: int = cv2.INTER_AREA) -> np.ndarray:
    s, ox, oy, W, H = t
    if s == 1.0 and not ox and not oy:
        return img
    big = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=interp)
    x, y = int(round(ox)), int(round(oy))
    return big[y:y + H, x:x + W]


def fit_points(points: list | None, t: tuple) -> list | None:
    """Keypoints [x, y, visible] moved by the same scale and cut. A point that the cut moves outside the
    picture stays visible: its limb is then drawn up to the edge (OpenCV clips the line), not dropped."""
    if points is None:
        return None
    s, ox, oy, W, H = t
    out = []
    for p in points:
        if p is None:
            out.append(None)
            continue
        x, y = p[0] * s - ox, p[1] * s - oy
        out.append([x, y, bool(p[2])] + list(p[3:]))
    return out


def make_controls(pass_dir: str | Path, kinds: list[str], size: tuple[int, int] | None = None,
                  body: str = "chibi") -> dict:
    """body: "chibi" draws the skeleton thick and bright (draw_pose); "standard" as the DWPose node (draw_pose_standard).
    Tested 2026-10-19 on a chibi character: the thick skeleton kept the sheet's proportions and mask, the standard one
    gave a taller, mannequin-like body."""
    if body not in BODY_TYPES:
        raise ValueError(f"body type must be one of {BODY_TYPES}")
    draw = draw_pose if body == "chibi" else draw_pose_standard
    pass_dir = Path(pass_dir)
    info = load(pass_dir)
    t = fit_transform(info["width"], info["height"], size)
    fps, n = info["fps"], info["frames"]
    w, h = t[3], t[4]
    need_mask = {"canny_env", "pose_env"} & set(kinds)
    if need_mask and not info["mask"]:
        raise ValueError("canny_env / pose_env need the character mask (render with a character collection)")
    if {"pose", "pose_env"} & set(kinds) and not info["keypoints"]:
        raise ValueError("pose needs keypoints (a rig with the cinestudio bone names)")
    out_dir = pass_dir / "controls"
    out_dir.mkdir(exist_ok=True)
    frames: dict[str, list[np.ndarray]] = {k: [] for k in kinds}
    dep = depth_frames(pass_dir) if "depth" in kinds else None
    kernel = np.ones((MASK_DILATE, MASK_DILATE), np.uint8)
    for i in range(n):
        rgb = fit_image(cv2.imread(str(info["rgb"][i])), t)
        edges = canny(rgb) if {"canny", "canny_env", "pose_env"} & set(kinds) else None
        env = None
        if need_mask:
            m = cv2.dilate((fit_image(mask_of(info["mask"][i]).astype(np.uint8) * 255, t) > 127).astype(np.uint8), kernel) > 0
            env = edges.copy()
            env[m] = 0
        hand_list = [fit_points(hp, t) for hp in hands_at(info["keypoints"], i)]       # [R, L]
        if body == "standard":
            hand_list = hand_list[::-1]                                                  # as the node: L, then R
        pose = draw(fit_points(info["keypoints"]["frames"][i], t), w, h, hand_list) \
            if info["keypoints"] and {"pose", "pose_env"} & set(kinds) else None
        for k in kinds:
            if k == "pose":
                frames[k].append(pose)
            elif k == "depth":
                frames[k].append(fit_image(dep[i], t))
            elif k == "canny":
                frames[k].append(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
            elif k == "canny_env":
                frames[k].append(cv2.cvtColor(env, cv2.COLOR_GRAY2BGR))
            elif k == "pose_env":
                frames[k].append(over(pose, env_grey(env)))
            else:
                raise ValueError(f"unknown control kind {k!r}")
    result = {}
    for k, fr in frames.items():
        dst = out_dir / control_name(k, body)
        write_video(fr, dst, fps)
        result[k] = str(dst)
    picks = [0, n // 3, 2 * n // 3, n - 1]
    rows = [np.hstack([cv2.resize(fit_image(cv2.imread(str(info["rgb"][i])), t), (w // 3, h // 3))] +
                      [cv2.resize(frames[k][i], (w // 3, h // 3)) for k in kinds]) for i in picks]
    cv2.imwrite(str(out_dir / "contact_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    result["contact_sheet"] = str(out_dir / "contact_sheet.jpg")
    return result


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pass_dir")
    ap.add_argument("--kinds", default="pose,depth,canny,canny_env,pose_env")
    a = ap.parse_args(argv)
    print(json.dumps(make_controls(a.pass_dir, [k.strip() for k in a.kinds.split(",") if k.strip()]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
