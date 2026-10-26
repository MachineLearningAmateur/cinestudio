"""A shot with several mannequins from one-mannequin shots with DIFFERENT motions (Blender, background, no render).

    blender -b A.blend --python bpy_jobs/merge_duo.py -- --other B.blend [--other C.blend ...] --out OUT.blend
        [--spacing 1.5] [--places "X,Y;X,Y;X,Y"]

User, 2026-10-23: the duo template ends in an arms-crossed pose, the two characters turned outward (mirrored, ~30 deg),
so the second mannequin needs its own end motion (duplicate_mannequin.py shares ONE action). The mannequin collection of
each other shot ("Mannequin", with its rig "MannequinRig" and its action) is appended to A as "Mannequin2" /
"MannequinRig2", "Mannequin3" / "MannequinRig3", ... (the names runner.new_run and duplicate_mannequin.py expect); A's
mannequin goes to the picture's left and the others follow to the right, `spacing` m apart along the camera's sideways
axis, centred on A's old place. A keeps its camera, set and sound.
`--places "X,Y;X,Y;..."` (one per mannequin, A first; 2026-10-25, templates from a video with several people:
cinestudio/video_template.py) puts each rig at that floor place in metres instead (X = the picture's right, Y = away
from the camera), so the dancers keep the distance and the depth order they had in the video. `--place-a` / `--place-b`
(the two-shot form) still work. 2026-10-25 (user: "if it has three then three options"): any number of --other.
"""
import argparse
import json
import sys

import bpy
from mathutils import Vector


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--other", required=True, action="append", help="another one-mannequin shot (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--collection", default="Mannequin")
    ap.add_argument("--rig", default="MannequinRig")
    ap.add_argument("--places", default="", help="X,Y;X,Y;... floor places in metres, one per mannequin (A first)")
    ap.add_argument("--place-a", default="", help="X,Y floor place of A's rig in metres (two shots)")
    ap.add_argument("--place-b", default="", help="X,Y floor place of B's rig in metres (two shots)")
    return ap.parse_args(argv)


def _xy(text):
    x, y = (float(v) for v in text.split(","))
    return x, y


def append_mannequin(path, collection, rig_name, rig_a, k):
    """The mannequin collection of `path` appended as collection + k, its rig named rig + k; -> (collection, rig, new)."""
    before = set(bpy.data.objects)
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        if collection not in src.collections:
            raise SystemExit(f"{path} has no collection {collection!r}")
        dst.collections = [collection]
    col_b = dst.collections[0]
    col_b.name = collection + str(k)
    new = [o for o in col_b.all_objects if o not in before]
    rig_b = next((o for o in new if o.type == "ARMATURE"), None)
    if rig_b is None:
        raise SystemExit(f"the mannequin collection of {path} has no rig")
    rig_b.name, rig_b.data.name = rig_name + str(k), (rig_a.data.name + str(k))
    for o in new:                                             # clean names: Part.001 -> Part2
        if o is not rig_b:
            base = o.name.rsplit(".", 1)[0] if o.name.rsplit(".", 1)[-1].isdigit() else o.name
            o.name = base + str(k)
    return col_b, rig_b, new


def main():
    a = args()
    sc = bpy.context.scene
    col_a, rig_a = bpy.data.collections[a.collection], bpy.data.objects[a.rig]
    if a.collection + "2" in bpy.data.collections or a.rig + "2" in bpy.data.objects:
        raise SystemExit("the shot already has a second mannequin")
    parent = next((p for p in bpy.data.collections if col_a.name in p.children), None)
    rigs, count = [rig_a], 0
    for k, other in enumerate(a.other, 2):
        col_b, rig_b, new = append_mannequin(other, a.collection, a.rig, rig_a, k)
        (parent.children if parent else sc.collection.children).link(col_b)
        rigs.append(rig_b)
        count += len(new)
    cam = sc.camera
    sc.frame_set(sc.frame_start)
    fwd = cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
    side = Vector((fwd.y, -fwd.x, 0.0))
    side = side.normalized() if side.length > 1e-6 else Vector((1.0, 0.0, 0.0))
    base = rig_a.location.copy()
    places = [_xy(t) for t in a.places.split(";") if t.strip()] if a.places else []
    if a.place_a or a.place_b:
        if not (a.place_a and a.place_b):
            raise SystemExit("--place-a and --place-b go together")
        places = [_xy(a.place_a), _xy(a.place_b)]
    if places:
        if len(places) != len(rigs):
            raise SystemExit(f"{len(places)} places for {len(rigs)} mannequins")
        for r, (x, y) in zip(rigs, places):
            r.location = Vector((x, y, base.z))
    else:
        n = len(rigs)
        for i, r in enumerate(rigs):
            r.location = base + side * (a.spacing * (i - (n - 1) / 2))
    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    acts = [r.animation_data.action.name if r.animation_data and r.animation_data.action else None for r in rigs[1:]]
    print("MERGED", json.dumps({"rigs": [[r.name, [round(v, 3) for v in r.location]] for r in rigs],
                                "objects": count, "action2": acts[0] if acts else None, "actions": acts}))


main()
