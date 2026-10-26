"""The rig's motion facts per frame, for clip texts written from Blender (cinestudio/timeline.py).

blender -b SHOT.blend --python bpy_jobs/rig_timeline.py -- OUT.json [--rig MannequinRig]

Per frame (world space, metres): the head, both shoulders, hands and feet, the hips, and the camera. No render:
only the pose is evaluated, so it is fast and safe next to a GPU job."""
import argparse
import json
import sys

import bpy

BONES = ("head", "hips", "shoulder.L", "shoulder.R", "hand.L", "hand.R", "foot.L", "foot.R")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--rig", default="MannequinRig")
    a = ap.parse_args(argv)
    sc = bpy.context.scene
    rig = bpy.data.objects.get(a.rig)
    if rig is None or rig.type != "ARMATURE":
        raise SystemExit(f"rig_timeline: no armature {a.rig!r}")
    missing = [b for b in BONES if b not in rig.pose.bones]
    if missing:
        raise SystemExit(f"rig_timeline: the rig has no bones {missing}")
    frames = []
    for f in range(sc.frame_start, sc.frame_end + 1):
        sc.frame_set(f)
        pts = {}
        for b in BONES:
            pb = rig.pose.bones[b]
            head = rig.matrix_world @ pb.head
            tail = rig.matrix_world @ pb.tail
            pts[b] = [round(v, 4) for v in (tail if b.startswith("hand") else head)]   # hands: the fingers' end
        cam = sc.camera.matrix_world.translation if sc.camera else None
        frames.append({"frame": f, "bones": pts, "camera": [round(v, 4) for v in cam] if cam is not None else None})
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "rig": a.rig, "fps": sc.render.fps, "frame_start": sc.frame_start,
                   "frame_end": sc.frame_end, "frames": frames}, fh)
    print("RIG_TIMELINE", a.out, len(frames))


main()
