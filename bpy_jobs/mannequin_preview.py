"""A character's own mannequin (its body + its proxy kit) rendered for the app's Mannequin view (Blender, background,
Workbench: light, a few seconds, safe next to a GPU job).

    blender -b SHOT.blend --python bpy_jobs/mannequin_preview.py -- --body BODY.json --spec SPEC.json --out DIR
        [--size 384x640] [--frame 2.2]

Writes DIR/front.png, DIR/right.png, DIR/back.png (transparent) and DIR/info.json, prints PREVIEW {json}:
{"height": m (the body, without the kit), "ref": the template's height, "frame_m", "size", "feet_row"}.
User, 2026-10-24: "some kind of rendered live image of what the mannequin would look like in the mannequin config".
An ORTHOGRAPHIC camera at a FIXED scale (`--frame` metres tall, the feet on the same pixel row): two characters' renders
compare at their real heights. A thin ring marks the template mannequin's height. "right" follows the sheet's slot
(the figure faces the picture's left: the camera at +X, as bpy_jobs/render_turnaround.py). The order is the run's: the
body first (pose-bone scales), then the kit (built in the rest pose on the bones, so the cape follows the legs).
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mannequin_body  # noqa: E402
import proxy_kit  # noqa: E402  (its CLI runs only as __main__)

FLOOR_MARGIN = 0.05          # metres below the feet in the frame


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, help="a JSON file: the body (mannequin_body.py keys)")
    ap.add_argument("--spec", required=True, help="a JSON file: the proxy kit spec ({} = none)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="384x640")
    ap.add_argument("--frame", type=float, default=2.2, help="the frame's height in metres (the same for everyone)")
    ap.add_argument("--rig", default="MannequinRig")
    ap.add_argument("--collection", default="Mannequin")
    return ap.parse_args(argv)


def height(coll) -> tuple[float, float]:
    bpy.context.view_layer.update()
    zs = [(o.matrix_world @ Vector(c)).z for o in coll.all_objects if o.type == "MESH" and not o.hide_render
          for c in o.bound_box]
    return min(zs), max(zs)


def rest(rig) -> None:
    if rig.animation_data:
        rig.animation_data.action = None
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)
    rig.data.pose_position = "POSE"


def ring(name, centre, radius, z, colour, coll):
    """A thin horizontal ring (a line in every side view): the template's height."""
    bpy.ops.mesh.primitive_torus_add(major_radius=radius, minor_radius=0.004, major_segments=96, minor_segments=6,
                                     location=(centre.x, centre.y, z))
    ob = bpy.context.active_object
    ob.name = name
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*colour, 1.0)
    ob.data.materials.append(m)
    for c in ob.users_collection:
        c.objects.unlink(ob)
    coll.objects.link(ob)
    return ob


def main():
    a = args()
    body = json.load(open(a.body, encoding="utf-8"))
    spec = json.load(open(a.spec, encoding="utf-8"))
    w, h = (int(x) for x in a.size.lower().split("x"))
    os.makedirs(a.out, exist_ok=True)
    sc = bpy.context.scene
    rig = bpy.data.objects[a.rig]
    coll = bpy.data.collections[a.collection]
    keep = set(coll.all_objects) | {rig}
    for o in sc.objects:                                          # only the mannequin (and its kit) is rendered
        if o not in keep:
            o.hide_render = True
    rest(rig)
    rig.scale = (1, 1, 1)
    lo, hi = height(coll)
    ref = hi - lo
    base = rig.location.copy()
    mannequin_body.apply(rig, body, base)                         # the body first (as in a run) ...
    lo2, hi2 = height(coll)
    tall = hi2 - lo2
    if spec:
        proxy_kit.build(spec, rig, coll)                          # ... then the kit on the scaled bones
    bpy.context.view_layer.update()
    centre = Vector((base.x, base.y, 0.0))
    guides = bpy.data.collections.new("PreviewGuides")
    sc.collection.children.link(guides)
    ring("TemplateHeight", centre, 0.55, lo + ref, (0.17, 0.70, 0.64), guides)
    ring("Floor", centre, 0.55, lo, (0.35, 0.37, 0.42), guides)
    try:
        sc.render.engine = "BLENDER_WORKBENCH"
    except TypeError as e:
        raise SystemExit(f"mannequin_preview: no Workbench ({e})")
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "MATERIAL"
    sc.display.shading.show_cavity = True
    sc.render.film_transparent = True
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = w, h, 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    cam = bpy.data.objects.new("PreviewCam", bpy.data.cameras.new("PreviewCam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = a.frame * max(1.0, w / h)              # the larger side (the height for a portrait frame)
    cz = lo - FLOOR_MARGIN + a.frame / 2
    views = {"front": Vector((0, -1, 0)), "right": Vector((1, 0, 0)), "back": Vector((0, 1, 0))}
    for name, d in views.items():
        cam.location = Vector((centre.x, centre.y, cz)) + d * 6.0
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = os.path.join(a.out, f"{name}.png")
        bpy.ops.render.render(write_still=True)
    info = {"height": round(tall, 4), "ref": round(ref, 4), "frame_m": a.frame, "size": [w, h],
            "feet_row": round(h * (1 - FLOOR_MARGIN / a.frame), 1), "views": list(views)}
    with open(os.path.join(a.out, "info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh)
    print("PREVIEW", json.dumps(info))


main()
