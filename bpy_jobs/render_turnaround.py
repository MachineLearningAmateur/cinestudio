"""Render the (kitted) mannequin of a shot as a character-sheet turnaround: rest pose, on black, even light, from the
front, its right side, its back and its left side (2026-10-22: a character sheet drawn FROM the template's mannequin,
so the views have the rig's pose, proportions and exact angles; with the proxy kit the costume's places show too).

    blender -b SHOT.blend --python bpy_jobs/render_turnaround.py -- OUT_DIR [--rig MannequinRig]
        [--collection Mannequin] [--width 832] [--height 1248] [--samples 32] [--body JSON]

Writes OUT_DIR/{front,right,back,left}.png. Cycles on the CPU (safe next to a GPU job). The mannequin faces -Y, its
left side is +X. The file names follow the SHEET's slots, not the anatomy: views/right.png shows the figure facing the
PICTURE'S LEFT (runner.step_look: a body that faces the picture's left takes the "right" panel), so "right" puts the
camera at +X. The first version had it at -X: the reposed right view faced the other way (2026-10-22)."""
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
    ap.add_argument("out")
    ap.add_argument("--rig", default="MannequinRig")
    ap.add_argument("--collection", default="Mannequin")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=1248)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--body", default="", help="the character's mannequin body (JSON, mannequin_body.py): the rest "
                    'pose with its proportions, e.g. {"head": 1.4, "legs": 0.75}')
    return ap.parse_args(argv)


def main():
    a = args()
    sc = bpy.context.scene
    rig = bpy.data.objects[a.rig]
    coll = bpy.data.collections[a.collection]
    keep = set(coll.all_objects) | {rig}
    for o in sc.objects:                                         # only the mannequin (and its kit) is rendered
        if o not in keep:
            o.hide_render = True
    body = json.loads(a.body) if a.body else {}
    if body:                                                     # the rest pose WITH the body's bone scales (REST
        if rig.animation_data:                                   # ignores pose-bone scales): no action, every bone
            rig.animation_data.action = None                     # at identity except the body's scales
        for pb in rig.pose.bones:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.location = (0, 0, 0)
            pb.scale = (1, 1, 1)
        rig.data.pose_position = "POSE"
        mannequin_body.apply(rig, {k: v for k, v in body.items() if k != "scale"})   # a sheet has no absolute size
    else:
        rig.data.pose_position = "REST"
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ Vector(c) for o in coll.all_objects if o.type == "MESH" and not o.hide_render
           for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    centre = (lo + hi) / 2
    height = hi.z - lo.z
    try:
        sc.render.engine = "CYCLES"
    except TypeError as e:
        raise SystemExit(f"render_turnaround: no Cycles ({e})")
    sc.cycles.device = "CPU"
    sc.cycles.samples = a.samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = a.width, a.height, 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("TurnaroundWorld")
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 0.0
    sc.world = world
    cam_data = bpy.data.cameras.new("TurnCam")
    cam_data.lens = 85
    cam_data.sensor_fit = "VERTICAL"                             # the picture's height is the sensor's height
    cam_data.sensor_height = 36.0
    cam = bpy.data.objects.new("TurnCam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    dist = (height * 1.12) / 2 / (cam_data.sensor_height / 2 / cam_data.lens)   # the body fills ~89% of the height
    fill = bpy.data.lights.new("TurnFill", "AREA")               # an even, soft light from the camera's side
    fill.energy, fill.size = 250.0, 4.0
    fill_ob = bpy.data.objects.new("TurnFill", fill)
    sc.collection.objects.link(fill_ob)
    top = bpy.data.lights.new("TurnTop", "AREA")
    top.energy, top.size = 80.0, 3.0
    top_ob = bpy.data.objects.new("TurnTop", top)
    sc.collection.objects.link(top_ob)
    views = {"front": Vector((0, -1, 0)), "right": Vector((1, 0, 0)), "back": Vector((0, 1, 0)), "left": Vector((-1, 0, 0))}
    for name, d in views.items():
        cam.location = centre + d * dist
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        fill_ob.location = centre + d * (dist * 0.8) + Vector((0, 0, height * 0.15))
        fill_ob.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        top_ob.location = centre + Vector((0, 0, height)) + d * 0.5
        top_ob.rotation_euler = (0, 0, 0)
        sc.render.filepath = f"{a.out}/{name}.png"
        bpy.ops.render.render(write_still=True)
    print("TURNAROUND", a.out, {"height": round(height, 3), "distance": round(dist, 2)})


main()
