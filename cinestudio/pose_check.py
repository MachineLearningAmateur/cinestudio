"""How many people does the standard pose detector see in each picture? (DWPose with its own person finder, YOLOX:
the first-frame check of a template made from text.) Run with ComfyUI's Python (it has controlnet_aux):

    <ComfyUI python> cinestudio/pose_check.py OUT.json IMG [IMG ...]

Writes OUT.json: {"<image stem>": {"persons": n, "big": large figures, "body": points shown of 18, "score": mean body
score}}.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_passes import AUX_DEFAULT, load_dwpose  # noqa: E402


BIG = 0.2      # a figure whose shown body points span at least this part of the frame height counts as "big"


def _span(pose, h: int) -> float:
    ys = [k.y for k in pose.body.keypoints if k is not None]
    return (max(ys) - min(ys)) / h if len(ys) >= 2 else 0.0


def read(det, img: np.ndarray) -> dict:
    """persons = every figure the person finder found; big = figures at least BIG of the frame tall (tiny people far
    away in a park or street do not fail a good frame, a second performer in front does)."""
    poses = det.detect_poses(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    h = img.shape[0]
    out = {"persons": len(poses), "big": sum(1 for q in poses if _span(q, h) >= BIG), "body": 0, "score": 0.0}
    if poses:
        p = max(poses, key=lambda q: sum(1 for k in q.body.keypoints if k is not None))
        kps = [k for k in p.body.keypoints if k is not None]
        out["body"] = len(kps)
        out["score"] = round(float(np.mean([k.score for k in kps])), 3) if kps else 0.0
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        raise SystemExit("usage: pose_check.py OUT.json IMG [IMG ...]")
    det = load_dwpose(Path(AUX_DEFAULT))
    res = {Path(p).stem: read(det, cv2.imread(p)) for p in argv[1:]}
    Path(argv[0]).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("CHECK DONE", json.dumps(res))


if __name__ == "__main__":
    main()
