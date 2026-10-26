"""Passes for a VIDEO template (a template made from text, not in Blender): the same files a Blender shot gives, so a
run treats both the same way (the source slices, the drawn skeleton in chibi or standard style, the set edges).

    <ComfyUI python> cinestudio/video_passes.py VIDEO PASS_DIR [--aux DIR]

From VIDEO: rgb/rgb_####.png (the frames), keypoints.json (the standard DWPose with its own person finder, in the
format of bpy_jobs/render_passes.py: coco18 body points [x, y, shown] in pixels and openpose_hand21 hands; the figure
with the most points when there are several), mask/mask_####.png (the figure's area: the hull of its body, hand and
face points, made larger), people/people.json (one character: the runner draws it like a one-mannequin shot) and
passes.json. A frame without a figure keeps the last figure's points (marked not shown).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def _find_aux() -> str:
    """The comfyui_controlnet_aux folder: CONTROLNET_AUX, else next to the ComfyUI whose python runs this script
    (<ComfyUI>/.venv/{Scripts,bin}/python -> <ComfyUI>/custom_nodes/comfyui_controlnet_aux)."""
    import os as _os
    import sys as _sys
    env = _os.environ.get("CONTROLNET_AUX")
    if env:
        return env
    for parent in Path(_sys.executable).resolve().parents:
        cand = parent / "custom_nodes" / "comfyui_controlnet_aux"
        if cand.is_dir():
            return str(cand)
    return ""


AUX_DEFAULT = _find_aux()
NAMES = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "r_hip", "r_knee",
         "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]
SHOWN = 0.3                       # the DWPose node's own threshold for drawing a point
MASK_GROW = 0.06                  # the mask hull grows by this part of the figure's height


def load_dwpose(aux: Path):
    sys.path.insert(0, str(aux / "src"))
    from custom_controlnet_aux.dwpose import DwposeDetector
    return DwposeDetector.from_pretrained("hr16/DWPose-TorchScript-BatchSize5", "hr16/yolox-onnx",
                                          det_filename="yolox_l.torchscript.pt",
                                          pose_filename="dw-ll_ucoco_384_bs5.torchscript.pt", torchscript_device="cpu")


def pts(kps, n):
    out = []
    for i in range(n):
        k = kps[i] if kps and i < len(kps) else None
        if k is None:
            out.append([0.0, 0.0, 0])
        else:
            # detect_poses gives pixel coordinates already (its own drawing divides by the size); keep them as they are
            out.append([round(float(k.x), 2), round(float(k.y), 2), 1 if float(k.score) >= SHOWN else 0])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("pass_dir")
    ap.add_argument("--aux", default=AUX_DEFAULT)
    ap.add_argument("--mask", default=None, help="the figure mask video (SAM 2, figure_mask.py): the person inside it "
                                                 "is the figure on every frame; its silhouette (made larger) is the mask")
    a = ap.parse_args(argv)
    out = Path(a.pass_dir)
    for sub in ("rgb", "mask", "people"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    det = load_dwpose(Path(a.aux))
    cap = cv2.VideoCapture(a.video)
    mcap = cv2.VideoCapture(a.mask) if a.mask else None
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    body, hands_r, hands_l, found = [], [], [], 0
    last = None
    i = 0
    w = h = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        i += 1
        h, w = bgr.shape[:2]
        cv2.imwrite(str(out / "rgb" / f"rgb_{i:04d}.png"), bgr)
        poses = det.detect_poses(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        sel = None
        if mcap is not None:                         # the selected figure: the person with most points in the mask
            okm, mf = mcap.read()
            sel = (cv2.resize(mf, (w, h), interpolation=cv2.INTER_NEAREST)[:, :, 0] > 127) if okm else None
        if sel is not None:
            def inside(q):
                return sum(1 for k in q.body.keypoints if k is not None and 0 <= int(k.y) < h and 0 <= int(k.x) < w
                           and sel[int(k.y), int(k.x)])
            cand = [q for q in poses if inside(q) >= 3]
            p = max(cand, key=inside, default=None)
        else:
            p = max(poses, key=lambda q: sum(1 for k in q.body.keypoints if k is not None), default=None)
        if p is not None:
            found += 1
            b = pts(p.body.keypoints, 18)
            hr, hl = pts(p.right_hand, 21), pts(p.left_hand, 21)
            face = pts(p.face, 70) if p.face else []
            last = (b, hr, hl)
        elif last is not None:                       # no figure: keep the place, hide the points
            b, hr, hl = ([[x, y, 0] for x, y, _ in q] for q in last)
            face = []
        else:
            b, hr, hl, face = [[0.0, 0.0, 0]] * 18, [[0.0, 0.0, 0]] * 21, [[0.0, 0.0, 0]] * 21, []
        body.append(b)
        hands_r.append(hr)
        hands_l.append(hl)
        mask = np.zeros((h, w), np.uint8)
        xy = np.array([[x, y] for x, y, s in b + hr + hl + face if s], np.float32)
        if sel is not None and sel.any():           # the SAM silhouette, made larger (edges near the figure go too)
            ys = np.nonzero(sel)[0]
            grow = max(3, int((ys.max() - ys.min()) * MASK_GROW))
            mask = cv2.dilate(sel.astype(np.uint8) * 255, np.ones((grow * 2 + 1, grow * 2 + 1), np.uint8))
        elif len(xy) >= 3:
            hull = cv2.convexHull(xy.astype(np.int32))
            tall = float(xy[:, 1].max() - xy[:, 1].min()) or h * 0.5
            grow = max(3, int(tall * MASK_GROW))
            cv2.fillConvexPoly(mask, hull, 255)
            mask = cv2.dilate(mask, np.ones((grow * 2 + 1, grow * 2 + 1), np.uint8))
        cv2.imwrite(str(out / "mask" / f"mask_{i:04d}.png"), mask)
    if not i:
        raise SystemExit(f"{a.video}: no frames")
    kp = {"format": "coco18", "names": NAMES, "frames": body,
          "hands": {"format": "openpose_hand21", "R": hands_r, "L": hands_l}}
    (out / "keypoints.json").write_text(json.dumps(kp), encoding="utf-8")
    (out / "people" / "people.json").write_text(json.dumps({"characters": []}), encoding="utf-8")
    info = {"frame_start": 1, "frame_end": i, "frames": i, "fps": round(float(fps), 3), "width": w, "height": h,
            "mask": True, "keypoints": True, "source": "video", "video": str(Path(a.video).resolve()),
            "figure_found": found}
    (out / "passes.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print("PASSES DONE", json.dumps({k: info[k] for k in ("frames", "fps", "width", "height", "figure_found")}))


if __name__ == "__main__":
    main()
