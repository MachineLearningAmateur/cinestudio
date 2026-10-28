"""Where each PROP is on screen, frame by frame (run inside Blender, headless; no render).

    blender -b shot.blend --python bpy_jobs/prop_track.py -- --out PROPS.json [--collection Props]

For every frame of the scene range and every visible mesh in the `Props` collection: the camera-view box of its
world bounding box, normalised 0..1 with y down (x0, y0, x1, y1), and whether it is in front of the camera. A prop
keyframe (runner._prop_keyframes, 2026-10-27, user: "we really need to find a way to insert props into the scene
without making them magically appear at the start") picks, per clip, the frame where a prop is largest and redraws
it with the character, so the prop is pictured where and when the layout has it.

Writes {"frame_start", "frame_end", "frames": [[{"name", "box": [x0, y0, x1, y1], "front": bool}, ...], ...]}.
"""
import argparse
import json
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--collection", default="Props")
    return ap.parse_args(argv)


def main():
    a = args()
    sc = bpy.context.scene
    if sc.camera is None:
        raise SystemExit("prop_track: the shot has no camera")
    coll = bpy.data.collections.get(a.collection)
    obs = [o for o in (coll.all_objects if coll else []) if o.type == "MESH" and not o.hide_render]
    frames = []
    for f in range(sc.frame_start, sc.frame_end + 1):
        sc.frame_set(f)
        row = []
        for o in obs:
            pts = [world_to_camera_view(sc, sc.camera, o.matrix_world @ Vector(c)) for c in o.bound_box]
            front = all(p.z > 0 for p in pts)
            xs, ys = [p.x for p in pts], [1.0 - p.y for p in pts]
            row.append({"name": o.name, "box": [round(min(xs), 4), round(min(ys), 4), round(max(xs), 4),
                                                round(max(ys), 4)], "front": front})
        frames.append(row)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"frame_start": sc.frame_start, "frame_end": sc.frame_end, "props": [o.name for o in obs],
                   "frames": frames}, fh)
    print("PROP_TRACK", len(obs), "props,", len(frames), "frames")


if __name__ == "__main__":
    main()
