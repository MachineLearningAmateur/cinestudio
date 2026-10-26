"""Step 1 of the motion follow score: DWPose reads each character in the generated VIDEO (inside the character's
Blender box) and saves its joints next to the rig's joints. Run with ComfyUI's Python (it has DWPose):

    <ComfyUI python> cinestudio/motion_follow.py RUN_DIR [--out RUN_DIR/follow_dets.json] [--aux DIR]

Step 2 scores them with the official COCO keypoint evaluation (pycocotools): python -m cinestudio.follow_eval.
This is the protocol of the pose-guided video papers (a standard detector on the output, compared with the target
poses); the target here is the Blender rig, not the drawn control. The quick summary printed here is only a check:

The reference is the run's pass folder: people/people.json for a shot with several characters, else the single
rig's keypoints.json with boxes from the mask pass. Per character and frame:
  body     share of the rig's body joints (in the frame) with a detected joint within 10% of the character's box
           height (looser than the detector bench: a drawn character has other proportions than the mannequin)
  body_lr  the same with left and right swapped if that fits better (position only)
  swap     the video's character has left and right the wrong way round (it faces the other way)
  facing   the video shows a face where the rig's head faces away, or no face where it faces the camera
  found    a pose was found in the box at all
The summary has the means, the swap and facing error counts, and the worst frames (to look at first).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pose_people as pp  # noqa: E402

FACE = (0, 14, 15, 16, 17)
BODY = tuple(i for i in range(18) if i not in FACE)


def reference(run_dir: Path, people_json: Path | None = None):
    """[(name, keypoints, boxes, ref width, ref height, hidden, control)] from the run's passes. hidden: per frame the
    18 flags of joints that ANOTHER character hides (several characters) or None; control: per frame the 18 points of
    the drawn control (pose_people's control_points.json) or None."""
    m = json.loads((run_dir / "state.json").read_text())
    pdir = Path(m["steps"]["passes"]["outputs"]["passes"])
    people = Path(people_json) if people_json else pdir / "people" / "people.json"
    if people.is_file():
        info = json.loads(people.read_text())
        if info["characters"]:
            out = []
            cp = people.parent / "frames" / "control_points.json"
            ctrl = json.loads(cp.read_text())["frames"] if cp.is_file() else None
            fr = range(info["frame_start"], info["frame_end"] + 1)
            for n, c in enumerate(info["characters"]):
                boxes = [info["frames"][str(f)][n]["box"] for f in fr]
                hidden = [info["frames"][str(f)][n].get("body") for f in fr]
                control = [(ctrl.get(str(f)) or {}).get(c["name"]) for f in fr] if ctrl else None
                out.append((c["name"], info["keypoints"][n], boxes, info["width"], info["height"], hidden, control))
            return out
    import cv2
    pinfo = json.loads((pdir / "passes.json").read_text())
    kp = json.loads((pdir / "keypoints.json").read_text())
    boxes = []
    for mf in sorted((pdir / "mask").glob("mask_*.png")):
        ys, xs = np.nonzero(cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE) > 127)
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())] if len(xs) else None)
    # a one-character run without --control-video draws its control from these rig points (pose_env)
    control = kp["frames"] if not json.loads((run_dir / "manifest.json").read_text()).get("control_video") else None
    return [("character", kp, boxes, pinfo["width"], pinfo["height"], None, control)]


def score(pose, rig, h):
    b = np.array([[k.x, k.y] if k is not None else [np.nan, np.nan] for k in pose.body.keypoints])
    r = np.array(rig, float)
    shown = [i for i in BODY if r[i, 2] > 0]
    if not shown:
        return None

    def pck(mp):
        ok = sum(1 for i in shown if not np.isnan(b[mp(i), 0]) and np.linalg.norm(b[mp(i)] - r[i, :2]) < 0.1 * h)
        return ok / len(shown)
    straight, swapped = pck(lambda i: i), pck(lambda i: pp.LR.get(i, i))
    rig_face = r[0, 2] > 0                        # the rig's face looks toward the camera (render_passes rule)
    det_face = not np.isnan(b[0, 0]) and (not np.isnan(b[14, 0]) or not np.isnan(b[15, 0]))
    return {"body": straight, "body_lr": max(straight, swapped), "swap": int(swapped > straight + 0.2),
            "facing": int(rig_face != det_face), "found": int(not np.isnan(b[:, 0]).all())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out")
    ap.add_argument("--people", help="people.json of the same shot (a run made with --control-video has none)")
    ap.add_argument("--aux", default=pp.AUX_DEFAULT)
    ap.add_argument("--video", help="score this video instead of RUN_DIR/final.mp4 (one clip or one take)")
    ap.add_argument("--start", type=int, default=0, help="the shot frame of the video's first frame (clip k's start)")
    a = ap.parse_args(argv)
    import cv2
    run = Path(a.run_dir)
    refs = reference(run, a.people)
    det = pp.load_dwpose(Path(a.aux))
    cap = cv2.VideoCapture(str(Path(a.video) if a.video else run / "final.mp4"))
    rows, dets, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        img = np.ascontiguousarray(frame[:, :, ::-1])
        vi, i = i, i + a.start                          # the video frame, the shot frame it shows
        for name, kp, boxes, rw, rh, hidden, control in refs:
            if i >= len(boxes) or i >= len(kp["frames"]) or not boxes[i]:
                continue
            # the pipeline's own mapping (passes.fit_transform): cover the frame, cut the overflow equally
            k = max(W / rw, H / rh)
            ox, oy = (rw * k - W) / 2, (rh * k - H) / 2
            b = boxes[i]
            box = [b[0] * k - ox, b[1] * k - oy, b[2] * k - ox, b[3] * k - oy]
            hid = hidden[i] if hidden else None
            rig = [[x * k - ox, y * k - oy, int(bool(s) and not (hid and hid[j]))]
                   for j, (x, y, s) in enumerate(kp["frames"][i])]
            pose = pp.pose_in_box(det, img, box)
            s = score(pose, rig, (box[3] - box[1]))
            if s:
                rows.append({"frame": i, "name": name, **s})
            row = {"frame": i, "name": name, "box": box, "rig": rig,
                   "det": [[q.x, q.y, q.score] if q is not None else None for q in pose.body.keypoints]}
            if control is not None and i < len(control) and control[i] is not None:
                row["ctrl"] = [[x * k - ox, y * k - oy, int(bool(v))] for x, y, v in control[i]]
            dets.append(row)
        i = vi + 1

    def mean(key, sel):
        return round(float(np.mean([r[key] for r in sel])), 3) if sel else None
    worst = sorted(rows, key=lambda r: r["body_lr"])[:8]
    summary = {"frames": i, "rows": len(rows), "body": mean("body", rows), "body_lr": mean("body_lr", rows),
               "swap": sum(r["swap"] for r in rows), "facing": sum(r["facing"] for r in rows),
               "per_character": {n: {"body": mean("body", [r for r in rows if r["name"] == n]),
                                     "swap": sum(r["swap"] for r in rows if r["name"] == n),
                                     "facing": sum(r["facing"] for r in rows if r["name"] == n)}
                                 for n in sorted({r["name"] for r in rows})},
               "worst": [(r["frame"], r["name"], round(r["body_lr"], 2)) for r in worst]}
    out = Path(a.out) if a.out else run / "follow_dets.json"
    expected = max(len(r[2]) for r in refs) if refs else 0
    out.write_text(json.dumps({"video": str(Path(a.video) if a.video else run / "final.mp4"), "start": a.start, "size": [W, H] if i else None, "dets": dets,
                               "frames": {"decoded": i, "expected": expected,
                                          "complete": i >= expected}, "quick": summary}))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
