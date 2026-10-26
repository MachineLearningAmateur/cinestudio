"""The height of a character's own mannequin (its body on the template's mannequin, rest pose) and of the template's
mannequin itself, in metres (Blender, background, no render).

    blender -b SHOT.blend --python bpy_jobs/mannequin_height.py -- --body JSON [--rig MannequinRig] [--collection Mannequin]

Prints: HEIGHT {"height": h, "ref": h0}. User, 2026-10-24: "standardize the sheets so they can properly reflect the
difference in heights": every sheet is drawn at one scale (the template mannequin fills the panel), so a character's
sheet height is height / ref of the panel (character.height_ratio)."""
import argparse
import json
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mannequin_body  # noqa: E402


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="{}")
    ap.add_argument("--rig", default="MannequinRig")
    ap.add_argument("--collection", default="Mannequin")
    return ap.parse_args(argv)


def height(coll) -> float:
    bpy.context.view_layer.update()
    zs = [(o.matrix_world @ Vector(c)).z for o in coll.all_objects if o.type == "MESH" for c in o.bound_box]
    return max(zs) - min(zs)


def rest(rig) -> None:
    if rig.animation_data:
        rig.animation_data.action = None
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)
    rig.data.pose_position = "POSE"


def main():
    a = args()
    rig = bpy.data.objects[a.rig]
    coll = bpy.data.collections[a.collection]
    rest(rig)
    rig.scale = (1, 1, 1)
    ref = height(coll)
    mannequin_body.apply(rig, json.loads(a.body or "{}"))
    print("HEIGHT", json.dumps({"height": round(height(coll), 4), "ref": round(ref, 4)}))


main()
