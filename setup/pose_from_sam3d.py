"""SAM 3D Body GLB -> a motion file for templates/flash_step/build.py (a command-line wrapper of cinestudio/mocap.py).

    python setup/pose_from_sam3d.py POSE.glb OUT_JSON [--seconds 10] [--fps 24] [--start 2.5]
                                    [--track N] [--min-feet 0] [--smooth 5] [--legs-smooth 9] [--root-depth]

The product path is `cinestudio shot from-video` (cinestudio/video_template.py), which runs SAM 3D Body itself."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cinestudio import mocap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=None, help="the window's length (default: to the end)")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--start", type=float, default=0.0, help="window start in seconds of the GLB")
    ap.add_argument("--track", type=int, default=None, help="the person (default: the one that moves the most)")
    ap.add_argument("--min-feet", type=float, default=0.0, help="push the ankles apart to this gap (m) when both stand")
    ap.add_argument("--smooth", type=int, default=5, help="Savitzky-Golay window of the joints (frames, odd)")
    ap.add_argument("--legs-smooth", type=int, default=9, help="the window of the hip, knee and ankle joints")
    ap.add_argument("--root-depth", action="store_true", help="keep the root's depth move (noisy; default off)")
    a = ap.parse_args()
    rec = mocap.motion_from_glb(a.glb, a.out, a.start, a.seconds, a.fps, a.track, a.smooth, a.legs_smooth,
                                a.min_feet, a.root_depth)
    print(json.dumps({"frames": len(rec["frames"]), "start_seconds": rec["start_seconds"], "tracks": rec["tracks"],
                      "track": rec["track"], **rec["scale"],
                      "contact_frames": {s: sum(f["contact"][s] for f in rec["frames"]) for s in (".R", ".L")}}))


if __name__ == "__main__":
    main()
