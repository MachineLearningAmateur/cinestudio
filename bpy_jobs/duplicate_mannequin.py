"""Put more mannequins beside the first, with the same motion (Blender, background, no render).

    blender -b SHOT.blend --python bpy_jobs/duplicate_mannequin.py -- --out OUT.blend [--spacing 1.5] [--count 2]
        [--scales 1.0,0.85] [--bodies JSON] [--names '[["Mannequin", "MannequinRig"], ["Mannequin2", "MannequinRig2"]]']

Why (2026-10-23, user: "would our existing pipeline work if we add in a 2nd character ... two character models side by
side doing the same motions"): the template's mannequin (collection "Mannequin", rig "MannequinRig") is copied to
"Mannequin2" / "MannequinRig2" (the names of templates/three_dancers), sharing the rig's action, and the rigs are spread
along the camera's sideways axis, `spacing` m apart and centred where the first one stood. The meshes keep their
armature modifiers, pointed at their own rig. render_people.py finds every rig with the mannequin's bones.
`--scales` = one uniform scale per mannequin, left to right (user, 2026-10-23: "can we make our character models
shorter" for kitsune, a short chibi character): the rig scales about its origin on the floor, so the feet stay on the
floor and the motion (steps, dodges) scales with the body. The proxy kit reads the scaled head, so its parts fit.
The run calls this on its own copy of the shot (runner.new_run: `--character2`, each character's `mannequin.scale`), so a
template keeps ONE mannequin with the video's motion and every character brings its own body (user, 2026-10-23: "swap
the model and apply the motion from the video"). A rig that already exists (MannequinRig2 ...) is kept with its place.
2026-10-25 (N characters, user: "if it has three then three options"): `--names` gives every mannequin's collection
and rig (slot order; default: the first's names + 2, 3, ...); the copies of a shot that already has several mannequins
go to the right of the rightmost one, `spacing` apart. The result line also gives each rig's place in the PICTURE at
the first frame ("screen_x": the hips' x, 0..1 from the left; None behind the camera): the run orders its group
sheet and its look crops by it.
"""
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--count", type=int, default=2)
    ap.add_argument("--collection", default="Mannequin")
    ap.add_argument("--rig", default="MannequinRig")
    ap.add_argument("--scales", default="", help="comma-separated uniform scales, one per mannequin (left to right)")
    ap.add_argument("--bodies", default="", help='JSON list, one body per mannequin (left to right), e.g. '
                    '[{}, {"scale": 0.8, "head": 1.4, "legs": 0.75}] (mannequin_body.py); replaces --scales')
    ap.add_argument("--names", default="", help='JSON list of [collection, rig], one per mannequin in slot order '
                    '(the first = --collection/--rig); default: the first + "2", "3", ...')
    return ap.parse_args(argv)


def copy_character(col, rig, suffix, name=None, rig_name=None):
    """A copy of the collection with its own rig (same action) and meshes bound to it (named `name` / `rig_name`, else
    the originals + suffix)."""
    new_col = bpy.data.collections.new(name or col.name + suffix)
    for parent in bpy.data.collections:
        if col.name in parent.children:
            parent.children.link(new_col)
            break
    else:
        bpy.context.scene.collection.children.link(new_col)
    new_rig = rig.copy()
    new_rig.data = rig.data.copy()
    new_rig.name = rig_name or rig.name + suffix
    new_rig.data.name = rig.data.name + suffix
    if rig.animation_data and rig.animation_data.action:
        new_rig.animation_data.action = rig.animation_data.action      # the same motion
    new_col.objects.link(new_rig)
    copies = {rig: new_rig}
    for ob in col.all_objects:
        if ob is rig:
            continue
        c = ob.copy()
        if ob.data is not None:
            c.data = ob.data.copy()
        c.name = ob.name + suffix
        copies[ob] = c
        new_col.objects.link(c)
    for ob, c in copies.items():                              # every link to an object of the character goes to its
        if ob is rig:                                         # copy (a nested child kept its ORIGINAL parent before)
            continue
        if c.parent in copies:
            c.parent = copies[c.parent]                       # matrix_parent_inverse and parent_bone are kept
        for mod in c.modifiers:
            if getattr(mod, "object", None) in copies:
                mod.object = copies[mod.object]
        for con in getattr(c, "constraints", []):
            if getattr(con, "target", None) in copies:
                con.target = copies[con.target]
    return new_col, new_rig


def picture_x(sc, rig):
    """The rig's hips (else its origin) in the picture at the current frame: x 0..1 from the left, None behind the
    camera or without a camera."""
    if sc.camera is None:
        return None
    from bpy_extras.object_utils import world_to_camera_view
    pb = rig.pose.bones.get("hips") if rig.pose else None
    p = rig.matrix_world @ pb.head if pb is not None else rig.matrix_world.translation
    v = world_to_camera_view(sc, sc.camera, p)
    return round(v.x, 4) if v.z > 0 else None


def main():
    a = args()
    sc = bpy.context.scene
    col = bpy.data.collections[a.collection]
    rig = bpy.data.objects[a.rig]
    cam = sc.camera
    sc.frame_set(sc.frame_start)
    fwd = (cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0)))
    side = Vector((fwd.y, -fwd.x, 0.0))
    side = side.normalized() if side.length > 1e-6 else Vector((1.0, 0.0, 0.0))
    names = [tuple(x) for x in json.loads(a.names)] if a.names else \
        [(a.collection, a.rig)] + [(a.collection + str(i), a.rig + str(i)) for i in range(2, a.count + 1)]
    if len(names) != a.count or names[0] != (a.collection, a.rig):
        raise SystemExit("--names needs one [collection, rig] per mannequin, the first = --collection/--rig")
    rigs, made, kept = [rig], 0, [True]
    for i in range(2, a.count + 1):                           # a shot that already has this mannequin keeps it (and its
        cname, rname = names[i - 1]                           # place): a template with several mannequins
        old = bpy.data.objects.get(rname)
        if old is not None and old.type == "ARMATURE":
            rigs.append(old)
            kept.append(True)
        else:
            rigs.append(copy_character(col, rig, str(i), cname, rname)[1])
            kept.append(False)
            made += 1
    if a.bodies:
        bodies = json.loads(a.bodies)
    else:
        bodies = [{"scale": float(x)} for x in a.scales.split(",") if x.strip()] if a.scales else []
    if bodies and len(bodies) != len(rigs):
        raise SystemExit(f"--bodies/--scales needs {len(rigs)} entries (one per mannequin)")
    places = [r.location.copy() for r in rigs]
    if made and made == len(rigs) - 1:                        # copies of one mannequin: left to right in the picture,
        base = rig.location.copy()                            # centred on the old place
        places = [base + side * (a.spacing * (i - (len(rigs) - 1) / 2)) for i in range(len(rigs))]
    elif made:                                                # a template with several: its own stay; the copies go to
        right = max((p for p, k in zip(places, kept) if k), key=lambda p: p.dot(side))   # the right of the rightmost
        step = 0
        for i, k in enumerate(kept):
            if not k:
                step += 1
                places[i] = right + side * (a.spacing * step)
    applied = []
    for i, r in enumerate(rigs):                              # the character's own body (absolute, about the floor)
        r.location = places[i]
        applied.append(mannequin_body.apply(r, bodies[i], places[i]) if bodies else None)
    unused = sorted(o.name for o in bpy.data.objects if o.type == "ARMATURE" and o not in rigs
                    and all(b in o.pose.bones for b in ("hips", "head", "thigh.L", "thigh.R")))
    sc.frame_set(sc.frame_start)
    bpy.context.view_layer.update()
    screen_x = [picture_x(sc, r) for r in rigs]
    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    print("DUPLICATED", json.dumps({"made": made, "unused": unused, "bodies": applied, "screen_x": screen_x,
                                    "names": [list(n) for n in names],
                                    "rigs": [(r.name, [round(v, 3) for v in r.location], round(r.scale.x, 3))
                                             for r in rigs]}))


main()
