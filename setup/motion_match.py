"""How closely a template's mannequin follows a reference dance video (the gauntlet's numbers for a template made with
setup/pose_from_video.py + templates/flash_step/build.py).

    python setup/motion_match.py MOTION.json PROJECTED.json [--per-second]

MOTION.json gives the reference window (its pass folder and start frame); PROJECTED.json comes from
bpy_jobs/rig_project.py on the template. Both skeletons are compared as 2D LIMB ANGLES (scale and place free: the
reference camera is low and the template camera is at chest height): upper arms, forearms, thighs, shins, the torso,
the shoulder line and the hip line. Prints the mean angle error per limb and overall (degrees), the worst frames, and
the foot slide (ankle movement on the mannequin while the reference ankle is still and low)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_from_video as pv  # noqa: E402

N = {n: i for i, n in enumerate(pv.NAMES[:14])}
LIMBS = {"upper_arm.R": ("r_shoulder", "r_elbow"), "forearm.R": ("r_elbow", "r_wrist"),
         "upper_arm.L": ("l_shoulder", "l_elbow"), "forearm.L": ("l_elbow", "l_wrist"),
         "thigh.R": ("r_hip", "r_knee"), "shin.R": ("r_knee", "r_ankle"),
         "thigh.L": ("l_hip", "l_knee"), "shin.L": ("l_knee", "l_ankle"),
         "shoulders": ("r_shoulder", "l_shoulder"), "hips": ("r_hip", "l_hip")}


def angles(P, a, b):
    d = P[:, N[b]] - P[:, N[a]]
    return np.degrees(np.arctan2(d[:, 1], d[:, 0]))


def torso(P):
    mid_hip = (P[:, N["r_hip"]] + P[:, N["l_hip"]]) / 2
    d = P[:, N["neck"]] - mid_hip
    return np.degrees(np.arctan2(d[:, 1], d[:, 0]))


def diff(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("projected")
    ap.add_argument("--per-second", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    mo = json.loads(Path(a.motion).read_text(encoding="utf-8"))
    pr = json.loads(Path(a.projected).read_text(encoding="utf-8"))
    body, hr, hl, src_fps, _ = pv.load(Path(mo["source"]))
    ref = pv.smooth(pv.resample(pv.fill(body), src_fps, mo["fps"]))[mo["start_frame"]:mo["start_frame"] + len(mo["frames"])]
    R = ref[:, :14, :2].astype(np.float64)
    M = np.array([f["coco"] for f in pr["frames"]], np.float64)[:len(R)]
    R = R[:len(M)]
    errs = {k: diff(angles(R, *v), angles(M, *v)) for k, v in LIMBS.items()}
    errs["torso"] = diff(torso(R), torso(M))
    allm = np.mean([e for e in errs.values()], axis=0)
    # foot slide: the mannequin's ankle moves (world, metres per frame) while the reference ankle is still and low
    slide = []
    for side, n in (("r_ankle", "r_ankle"), ("l_ankle", "l_ankle")):
        w = np.array([f["ankles"][side] for f in pr["frames"]])[:len(R)]
        sp_m = np.linalg.norm(np.diff(w, axis=0), axis=1)
        sp_r = np.linalg.norm(np.diff(R[:, N[n]], axis=0), axis=1)
        low = R[1:, N[n], 1] > np.percentile(R[:, N[n], 1], 60)
        still = (sp_r < np.percentile(sp_r, 40)) & low
        mpp = mo["scale"]["metres_per_pixel"]               # excess: the mannequin's move minus the dancer's own
        slide.append(float((sp_m[still] - sp_r[still] * mpp).mean() * 100) if still.any() else 0.0)
    # pops (Codex review): a joint's frame-to-frame step on screen, in torso lengths, beyond the dancer's own step
    def steps(P):
        mid = (P[:, N["r_hip"]] + P[:, N["l_hip"]]) / 2
        tl = np.median(np.linalg.norm(P[:, N["neck"]] - mid, axis=1))
        return np.linalg.norm(np.diff(P, axis=0), axis=2) / max(1e-6, tl)
    ex = steps(M) - steps(R)
    pops = ex.max(axis=1)
    out = {"mean_deg": round(float(allm.mean()), 2),
           "pop_max_torso": round(float(pops.max()), 3), "pop_p99_torso": round(float(np.percentile(pops, 99)), 3),
           "pop_frames": [int(i) + 2 for i in np.argsort(-pops)[:5]],
           "limbs_deg": {k: round(float(v.mean()), 1) for k, v in errs.items()},
           "worst_frames": [int(i) + 1 for i in np.argsort(-allm)[:8]],
           "foot_slide_excess_cm_per_frame": [round(s, 2) for s in slide]}
    if a.per_second:
        fps = int(mo["fps"])
        out["per_second_deg"] = [round(float(allm[i:i + fps].mean()), 1) for i in range(0, len(allm), fps)]
    print(json.dumps(out))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
