"""Render a template's static SET alone from its camera (2026-10-24, user: "we should be generating the backgrounds
based off of the blender template"): every mannequin collection (Mannequin, Mannequin2, ...), the moving props
(collection Props), armatures and everything parented to them are hidden; the rest (walls, floor, furniture) is rendered
with Workbench (studio light, material colours, soft shadows, cavity) at frame 1. A Qwen edit then redraws this render
as the finished place, keeping every object's place and outline (cinestudio/background.py make_set_plate).

    blender -b SHOT.blend --python bpy_jobs/render_set.py -- --out SET.png --width 1920 --height 1120 [--frame 1]

Prints SET <json> with the hidden and the kept object names.
"""
import argparse
import json
import sys

import bpy


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--frame", type=int, default=None, help="default: the scene's frame_start")
    ap.add_argument("--cover", default=None, help="a pivot camera's cover pose (JSON file: plate_world.py --info "
                                                   "'cover'): the set is rendered from it")
    return ap.parse_args(argv)


def moving_or_character(o) -> bool:
    for c in o.users_collection:
        if c.name.startswith("Mannequin") or c.name == "Props":
            return True
    p = o.parent
    while p is not None:
        if p.type == "ARMATURE":
            return True
        p = p.parent
    return o.type == "ARMATURE"


def main():
    a = args()
    sc = bpy.context.scene
    if sc.camera is None:
        raise SystemExit("the shot has no camera")
    hidden, kept = [], []
    set_objs = ({o.name for o in bpy.data.collections["Set"].all_objects} if "Set" in bpy.data.collections else None)
    for o in bpy.data.objects:
        if o.type in ("CAMERA", "LIGHT", "EMPTY"):
            continue
        # a template with a Set layer: only its objects; an older template: everything that is not a character,
        # a prop or a rig (shots.py layers)
        drop = (o.name not in set_objs) if set_objs is not None else moving_or_character(o)
        if drop:
            o.hide_render = True
            hidden.append(o.name)
        elif not o.hide_render:
            kept.append(o.name)
    sc.frame_set(a.frame if a.frame is not None else sc.frame_start)
    if a.cover:                                  # a pivot camera: render from its cover pose (the plate's projector)
        from mathutils import Matrix, Quaternion, Vector
        with open(a.cover, encoding="utf-8") as fh:
            cv = json.load(fh)
        cd = bpy.data.cameras.new("SetCover")
        cd.lens, cd.sensor_width, cd.sensor_fit = cv["lens_mm"], cv["sensor_mm"], "HORIZONTAL"
        ob = bpy.data.objects.new("SetCover", cd)
        ob.matrix_world = Matrix.Translation(Vector(cv["location"])) @ Quaternion(cv["rotation_quat"]).to_matrix().to_4x4()
        sc.collection.objects.link(ob)
        sc.camera = ob
    try:
        sc.render.engine = "BLENDER_WORKBENCH"
    except TypeError:
        sc.render.engine = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
                            if "WORKBENCH" in e.identifier][0]
    sh = sc.display.shading
    sh.light = "STUDIO"
    sh.color_type = "MATERIAL"
    sh.show_shadows = False                    # flat light: a shadow pattern would be painted into the picture
    sh.show_cavity = True
    sh.background_type = "VIEWPORT"
    sh.background_color = (0.06, 0.06, 0.07)
    sc.render.resolution_x, sc.render.resolution_y = a.width, a.height
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = a.out
    bpy.ops.render.render(write_still=True)
    print("SET " + json.dumps({"hidden": hidden, "kept": kept, "frame": a.frame, "size": [a.width, a.height]}))


main()
