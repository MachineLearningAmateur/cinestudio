"""The LAYERS of a template (2026-10-24, user: "is it worth further breaking down our templates to give more control over
background vs props vs characters?"): every template has three layers, by Blender collection:

    characters  collections Mannequin, Mannequin2, ... (each with its rig MannequinRig, MannequinRig2, ...)
    props       collection Props: objects that move or that the characters touch (balls, a photo)
    set         collection Set: the static place (walls, floor, furniture): the source of a backplate drawn over it

    blender -b SHOT.blend --python bpy_jobs/scan_layers.py -- --out LAYERS.json

Writes {"characters": [{"collection", "rig", "objects", "place", "screen", "height_m"}], "props": [{"name",
"animated", "parent"}], "set": [names], "loose": [mesh objects in no layer], "camera_moves": bool, "frames": [start,
end]}. Per character at the first frame (the poster's frame): "place" = the hips' floor place [x, y] in m, "screen" =
{"head": [u, v], "feet": [u, v]} = the head top and the feet projected by the scene camera (0..1 of the frame, v from
the top; the New video form's numbered markers, 2026-10-25), "height_m" = head top above the feet. Missing (None)
when the rig has no such bones or the point is behind the camera.
"""
import argparse
import json
import sys

import bpy

SKIP_TYPES = ("CAMERA", "LIGHT", "EMPTY", "ARMATURE", "SPEAKER", "LIGHT_PROBE")


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def animated(o) -> bool:
    if o.animation_data and o.animation_data.action:
        return True
    p = o.parent
    while p is not None:                     # a prop held in a hand moves with the rig
        if p.type == "ARMATURE" or (p.animation_data and p.animation_data.action):
            return True
        p = p.parent
    return False


def figure_points(sc, rig_name):
    """(place [x, y], screen {"head", "feet"} or None, height in m) of a rig at the current frame."""
    rig = bpy.data.objects.get(rig_name) if rig_name else None
    if rig is None or rig.type != "ARMATURE" or rig.pose is None:
        return None, None, None
    pb, mw = rig.pose.bones, rig.matrix_world
    head = mw @ pb["head"].tail if "head" in pb else None
    hips = mw @ pb["hips"].head if "hips" in pb else None
    feet = [mw @ pb[n].head for n in ("foot.L", "foot.R", "toe.L", "toe.R") if n in pb]
    if head is None or hips is None:
        return None, None, None
    place = [round(hips.x, 4), round(hips.y, 4)]
    low = min((f.z for f in feet), default=hips.z - 0.9) - 0.03     # the sole is a little below the ankle
    height = round(head.z - low, 3)
    screen = None
    if sc.camera is not None:
        from bpy_extras.object_utils import world_to_camera_view
        from mathutils import Vector
        ph = world_to_camera_view(sc, sc.camera, head)
        pf = world_to_camera_view(sc, sc.camera, Vector((hips.x, hips.y, low)))
        if ph.z > 0 and pf.z > 0:
            screen = {"head": [round(ph.x, 4), round(1 - ph.y, 4)], "feet": [round(pf.x, 4), round(1 - pf.y, 4)]}
    return place, screen, height


def scan(sc) -> dict:
    """The layer scan of the open .blend (also imported by bpy_jobs/check_shot.py: `shot check`)."""
    chars, in_layer = [], set()
    for c in sorted(bpy.data.collections, key=lambda c: c.name):
        if c.name.startswith("Mannequin") and c.name != "MannequinHidden":
            objs = [o.name for o in c.all_objects]
            rig = next((o.name for o in c.all_objects if o.type == "ARMATURE"), None)
            if rig is None:                  # the rig can live beside the collection (MannequinRig2 for Mannequin2)
                want = "MannequinRig" + c.name[len("Mannequin"):]
                rig = want if want in bpy.data.objects else None
            chars.append({"collection": c.name, "rig": rig, "objects": len(objs)})
            in_layer |= set(objs)
    props = []
    if "Props" in bpy.data.collections:
        for o in bpy.data.collections["Props"].all_objects:
            props.append({"name": o.name, "animated": animated(o), "parent": o.parent.name if o.parent else None})
            in_layer.add(o.name)
    set_objs = []
    if "Set" in bpy.data.collections:
        set_objs = sorted(o.name for o in bpy.data.collections["Set"].all_objects)
        in_layer |= set(set_objs)
    loose = sorted(o.name for o in bpy.data.objects
                   if o.type not in SKIP_TYPES and o.name not in in_layer
                   and not (o.parent and o.parent.type == "ARMATURE"))
    sc.frame_set(sc.frame_start)            # the poster is the first frame
    for ch in chars:
        try:
            ch["place"], ch["screen"], ch["height_m"] = figure_points(sc, ch["rig"])
        except Exception as exc:  # noqa: BLE001 - the slots work without places
            print(f"place of {ch['collection']}: {exc}")
    moves = False
    if sc.camera is not None:                # the same still-camera test as the backplates (every frame, lens,
        import os                            # sensor, shift, camera switches: plate_world.camera_info)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from plate_world import camera_info
            moves = bool(camera_info(sc).get("moves"))
        except Exception as exc:  # noqa: BLE001 - a camera the backplate code refuses: report it as moving
            print(f"camera check: {exc}")
            moves = True
    return {"characters": chars, "props": props, "set": set_objs, "loose": loose, "camera_moves": moves,
            "frames": [sc.frame_start, sc.frame_end]}


def main():
    a = args()
    out = scan(bpy.context.scene)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("LAYERS " + json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in out.items()}))


if __name__ == "__main__":                   # Blender runs --python scripts as __main__; check_shot.py imports scan()
    main()
