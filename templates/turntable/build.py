"""Turntable template: the camera circles the mannequin (A-pose) at a constant speed, 360 degrees by default.

Used to make a character sheet: the AI draws the character over this motion (front view as frame 0),
and because the angle of every frame is known, the code takes the views at exact angles.

    blender -b --factory-startup --python templates/turntable/build.py -- --out turntable.blend \
        [--frames 121] [--degrees 360]

Camera angle of frame f (1-based): degrees * (f - 1) / (frames - 1), written to the scene as the
custom property "cs_turntable" ({"frames", "degrees"}) and to <out>.json. 0 = in front of the
figure; the camera moves counter-clockwise seen from above, so at 90 degrees it sees the figure's
left side and at 180 degrees its back. The lights move with the camera: the character keeps the
same lighting in every view, as on a turntable.
Everything is built from code (no external assets).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from blender_lib import mannequin as mq  # noqa: E402

FPS = 24
COLORS = {"mannequin": (0.72, 0.66, 0.57), "joint": (0.42, 0.38, 0.33), "stripe": (0.15, 0.17, 0.22)}
RELAXED = (0.30, 0.0)
# A-pose like the front-view prompt: arms about 35 degrees away from the body
A_POSE = {"upper_arm.L": (0.55, 0.0, -0.84), "upper_arm.R": (-0.55, 0.0, -0.84),
          "forearm.L": (0.55, -0.05, -0.83), "forearm.R": (-0.55, -0.05, -0.83),
          "hand.L": (0.5, -0.05, -0.86), "hand.R": (-0.5, -0.05, -0.86)}


def material(name, color):
    m = bpy.data.materials.new("cs_" + name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = 0.6
    return m


def build(frames=121, degrees=360.0, width=720, height=1280):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, frames
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = width, height, 100
    rig, _ = mq.build({k: material(k, v) for k, v in COLORS.items()})
    mq.key_pose(rig, 1, dirs=A_POSE, hands={".L": RELAXED, ".R": RELAXED})
    # camera on a constant-speed orbit (linear keys on a pivot: no ease in or out); lights ride along
    pivot = bpy.data.objects.new("OrbitPivot", None)
    sc.collection.objects.link(pivot)
    cam = bpy.data.objects.new("ShotCam", bpy.data.cameras.new("ShotCam"))
    cam.data.lens = 50
    sc.collection.objects.link(cam)
    cam.parent = pivot
    cam.location = (0, -3.3, 0.92)       # the figure fills about 80% of the frame height
    cam.rotation_euler = (math.radians(90), 0, 0)
    sc.camera = cam
    from mathutils import Vector
    for name, loc, energy in (("Key", (-2.5, -3.0, 3.0), 400), ("Fill", (3.0, -2.5, 2.0), 200), ("Rim", (0, 2.5, 3.0), 250)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy, ld.size = energy, 2.5
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        lo.rotation_euler = (Vector((0, 0, 1.0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        lo.parent = pivot
        sc.collection.objects.link(lo)
    pivot.rotation_mode = "XYZ"
    for f, a in ((1, 0.0), (frames, degrees)):
        pivot.rotation_euler = (0, 0, math.radians(a))
        pivot.keyframe_insert("rotation_euler", frame=f)
    _linear_keys(pivot)
    w = bpy.data.worlds.new("World")
    sc.world = w
    w.use_nodes = True
    next(n for n in w.node_tree.nodes if n.type == "BACKGROUND").inputs["Color"].default_value = (0, 0, 0, 1)
    sc["cs_turntable"] = json.dumps({"frames": frames, "degrees": degrees})
    sc.frame_set(1)
    return sc


def _linear_keys(ob):
    """Make every key of the object's own action linear (Blender 4.4+ layered actions and older actions)."""
    act = ob.animation_data.action
    curves = []
    if hasattr(act, "layers") and act.layers:
        for layer in act.layers:
            for strip in layer.strips:
                for bag in getattr(strip, "channelbags", []):
                    curves += list(bag.fcurves)
    if not curves and hasattr(act, "fcurves"):
        curves = list(act.fcurves)
    for fc in curves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=121)
    ap.add_argument("--degrees", type=float, default=360.0)
    a = ap.parse_args(argv)
    build(a.frames, a.degrees)
    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as fh:
        json.dump({"frames": a.frames, "degrees": a.degrees}, fh)
    print("SAVED", a.out)


if __name__ == "__main__":
    main()
