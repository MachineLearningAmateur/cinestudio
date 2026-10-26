"""Measure the joins of a finished run: is the stitched video smooth where clips meet, and are later clips as sharp?

    python setup/join_check.py RUN_ID [--out DIR]

For the run's final video (and its clips):
- the frame change (mean absolute grey difference of consecutive frames, 192x112) at every join and at the loop point,
  as a multiple of the video's median frame change: about 1 = a seamless join; above ~2.5 = a visible jump;
- the centre sharpness (Laplacian variance) of every clip, before the stitch, and of the stitched video per clip span;
- a contact sheet of the frames around each join (join_check.jpg) to look at.
Prints one JSON line.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cinestudio import runner, video  # noqa: E402


def frames_of(path):
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def sharp(f):
    g = cv2.cvtColor(cv2.resize(f, (768, int(768 * f.shape[0] / f.shape[1]))), cv2.COLOR_BGR2GRAY)
    w = g.shape[1]
    return float(cv2.Laplacian(g[:, w // 4: 3 * w // 4], cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = runner.run_dir(a.run)
    st = json.loads((d / "state.json").read_text())
    m = json.loads((d / "manifest.json").read_text())
    clip = st["steps"]["clip"]
    fmt = st["steps"]["controls"]["outputs"]
    n, ov, frames = fmt.get("clips", 1), int(fmt.get("overlap", 1)), fmt["frames"]
    clips = [clip.get("outputs", {}).get("clip") if k == 1 else (clip.get("parts", {}).get(str(k), {}).get("outputs") or {}).get("clip")
             for k in range(1, n + 1)]
    final = d / "final.mp4"
    fr = frames_of(final)
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (192, 112)).astype(np.float32) for f in fr]
    diff = [float(np.abs(small[i + 1] - small[i]).mean()) for i in range(len(small) - 1)]
    med = float(np.median(diff)) or 1.0
    starts = [video.clip_start(k, frames, ov) for k in range(2, n + 1)]
    joins = []
    for s in starts:                                  # the blend window: the overlap frames s .. s+ov-1
        lo, hi = max(0, s - 2), min(len(diff), s + ov + 2)
        win = diff[lo:hi]
        joins.append({"frame": s, "max_x_median": round(max(win) / med, 2) if win else None,
                      "mean_x_median": round(float(np.mean(win)) / med, 2) if win else None})
    loop = m.get("loop")
    seam = round((float(np.abs(small[0] - small[-1]).mean()) / med), 2) if loop and len(small) > 2 else None
    per_clip_src = []
    for c in clips:
        if c and Path(c).is_file():
            cf = frames_of(c)
            per_clip_src.append(round(float(np.mean([sharp(f) for f in cf[::4]])), 1))
        else:
            per_clip_src.append(None)
    spans = [0] + starts + [len(fr)]
    per_clip_final = [round(float(np.mean([sharp(f) for f in fr[spans[i]:spans[i + 1]:4]])), 1)
                      for i in range(len(spans) - 1) if spans[i + 1] > spans[i]]
    out = Path(a.out) if a.out else d
    tiles = []
    for s in starts[:4]:
        row = []
        for i in (s - 6, s, s + ov // 2, s + ov - 1, s + ov + 6):
            i = max(0, min(len(fr) - 1, i))
            t = cv2.resize(fr[i], (320, int(320 * fr[i].shape[0] / fr[i].shape[1])))
            cv2.putText(t, str(i), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            row.append(t)
        tiles.append(np.hstack(row))
    if tiles:
        cv2.imwrite(str(out / "join_check.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(json.dumps({"run": a.run, "frames": len(fr), "median_change": round(med, 2), "joins": joins,
                      "loop_seam_x_median": seam, "sharpness_clips_raw": per_clip_src,
                      "sharpness_final_by_clip": per_clip_final, "sheet": str(out / "join_check.jpg") if tiles else None}))


if __name__ == "__main__":
    main()
