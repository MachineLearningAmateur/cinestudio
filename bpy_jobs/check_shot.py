"""Everything `shot check` (cinestudio/shot_check.py) needs from a template .blend, in ONE headless pass (no render:
only the pose and the camera are evaluated, so it is fast and safe next to a GPU job).

    blender -b SHOT.blend --python bpy_jobs/check_shot.py -- --out FACTS.json

FACTS = {"version", "frames" [start, end], "fps", "resolution" [w, h],
         "layers": the layer scan (bpy_jobs/scan_layers.py scan()),
         "mannequin_foreign": {collection: [objects in a Mannequin* layer that are not part of a figure]},
         "in_two_layers": [objects linked in more than one layer],
         "rigs": {rig: {"collection", "timeline" (bpy_jobs/rig_timeline.py frames: head, hips, shoulders, hands,
                  feet + the camera, world m), "joints" {joint: [[u, v] or None per frame]} (screen, 0..1, v from the
                  top), "scale_leaks" [{"bone", "first", "last", "max"}]}},
         "camera": {"frames": [{"loc", "fwd", "lens", "name"}] or [], "moves"},
         "props": [{"name", "parent", "parent_type", "parent_bone", "part", "animated", "boxes" [[u0, v0, u1, v1] or
                    None per frame]}],
         "set_materials": [{"object", "material", "color" [r, g, b] linear, "saturation", "value", "textured"}],
         "sounds": [{"name", "file", "packed", "frame_start", "mute"}]}
Every frame is evaluated (scales and joint speeds need them all)."""
import argparse
import colorsys
import json
import os
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_layers import scan  # noqa: E402

VERSION = 1
TIMELINE_BONES = ("head", "hips", "shoulder.L", "shoulder.R", "hand.L", "hand.R", "foot.L", "foot.R")
# joints on screen for the acceleration check: (name, bone, "head"|"tail")
JOINTS = (("head", "head", "tail"), ("neck", "neck", "head"), ("hips", "hips", "head"),
          ("elbow.L", "forearm.L", "head"), ("elbow.R", "forearm.R", "head"),
          ("wrist.L", "hand.L", "head"), ("wrist.R", "hand.R", "head"),
          ("hand.L", "hand.L", "tail"), ("hand.R", "hand.R", "tail"),
          ("knee.L", "shin.L", "head"), ("knee.R", "shin.R", "head"),
          ("ankle.L", "foot.L", "head"), ("ankle.R", "foot.R", "head"),
          ("toe.L", "toe.L", "tail"), ("toe.R", "toe.R", "tail"))
SCALE_TOL = 1e-3


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def r4(v):
    return [round(float(x), 4) for x in v]


def in_armature_chain(o) -> bool:
    p = o.parent
    while p is not None:
        if p.type == "ARMATURE":
            return True
        p = p.parent
    return False


def layer_of(o) -> list:
    out = []
    for c in bpy.data.collections:
        if o.name in c.all_objects:
            if c.name.startswith("Mannequin") and c.name != "MannequinHidden":
                out.append("Mannequin")
            elif c.name in ("Props", "Set"):
                out.append(c.name)
    return sorted(set(out))


def project(sc, cam, p):
    v = world_to_camera_view(sc, cam, p)
    return [round(v.x, 4), round(1 - v.y, 4)] if v.z > 0 else None


def material_color(m):
    """(linear rgb, textured) of a material: the Principled Base Color (not linked) or the viewport colour."""
    if m.use_nodes and m.node_tree:
        bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is not None:
            inp = bsdf.inputs.get("Base Color") or bsdf.inputs[0]
            if inp.is_linked:
                return list(m.diffuse_color)[:3], True
            return list(inp.default_value)[:3], False
    return list(m.diffuse_color)[:3], False


def sound_strips(sc):
    ed = sc.sequence_editor
    if ed is None:
        return []
    strips = getattr(ed, "sequences_all", None) or getattr(ed, "strips_all", None) or []
    out = []
    for s in strips:
        if s.type != "SOUND" or s.sound is None:
            continue
        out.append({"name": s.name, "file": bpy.path.basename(s.sound.filepath or ""),
                    "packed": s.sound.packed_file is not None, "frame_start": int(s.frame_start),
                    "mute": bool(s.mute)})
    return out


def main():
    a = args()
    sc = bpy.context.scene
    layers = scan(sc)                                  # also sets the frame to frame_start
    rigs = {}
    for ch in layers.get("characters") or []:
        rig = bpy.data.objects.get(ch.get("rig") or "")
        if rig is not None and rig.type == "ARMATURE" and rig.pose is not None:
            rigs[rig.name] = {"obj": rig, "collection": ch["collection"], "timeline": [],
                              "joints": {j[0]: [] for j in JOINTS if j[1] in rig.pose.bones}, "leaks": {}}
    parts_names = {p["name"] for p in layers.get("props") or []}
    props = []
    for p in layers.get("props") or []:
        o = bpy.data.objects.get(p["name"])
        if o is None:
            continue
        props.append({"obj": o, "name": o.name, "parent": o.parent.name if o.parent else None,
                      "parent_type": o.parent_type if o.parent else None,
                      "parent_bone": o.parent_bone if o.parent and o.parent_type == "BONE" else None,
                      "part": bool(o.parent and o.parent.name in parts_names), "animated": p.get("animated"),
                      "boxes": []})
    cam_frames = []
    for f in range(sc.frame_start, sc.frame_end + 1):
        sc.frame_set(f)
        cam = sc.camera
        if cam is not None:
            mw = cam.matrix_world
            fwd = (mw.to_3x3() @ Vector((0, 0, -1))).normalized()
            cam_frames.append({"loc": r4(mw.translation), "fwd": r4(fwd), "lens": round(cam.data.lens, 3),
                               "name": cam.name})
        for name, R in rigs.items():
            rig = R["obj"]
            pb, mw = rig.pose.bones, rig.matrix_world
            for b in pb:
                dev = max(abs(s - 1.0) for s in b.scale)
                if dev > SCALE_TOL:
                    L = R["leaks"].setdefault(b.name, {"bone": b.name, "first": f, "last": f, "max": 1.0})
                    L["last"] = f
                    worst = max(b.scale, key=lambda s: abs(s - 1.0))
                    if abs(worst - 1.0) > abs(L["max"] - 1.0):
                        L["max"] = round(float(worst), 4)
            if all(b in pb for b in TIMELINE_BONES):
                pts = {b: r4(mw @ (pb[b].tail if b.startswith("hand") else pb[b].head)) for b in TIMELINE_BONES}
                R["timeline"].append({"frame": f, "bones": pts,
                                      "camera": r4(cam.matrix_world.translation) if cam is not None else None})
            for jn, bone, end in JOINTS:
                if bone in pb:
                    w = mw @ (pb[bone].tail if end == "tail" else pb[bone].head)
                    R["joints"][jn].append(project(sc, cam, w) if cam is not None else None)
        if cam is not None:
            for P in props:
                o = P["obj"]
                pts = [project(sc, cam, o.matrix_world @ Vector(c)) for c in o.bound_box]
                if any(p is None for p in pts):
                    P["boxes"].append(None)
                    continue
                us, vs = [p[0] for p in pts], [p[1] for p in pts]
                P["boxes"].append([round(min(us), 4), round(min(vs), 4), round(max(us), 4), round(max(vs), 4)])
    # objects in a Mannequin layer that are not a figure's parts (the rig or objects parented into it)
    foreign = {}
    for c in bpy.data.collections:
        if c.name.startswith("Mannequin") and c.name != "MannequinHidden":
            bad = sorted(o.name for o in c.all_objects if o.type != "ARMATURE" and not in_armature_chain(o)
                         and o.type not in ("EMPTY", "CAMERA", "LIGHT"))
            props_too = set(bpy.data.collections["Props"].all_objects.keys()) if "Props" in bpy.data.collections else set()
            bad += sorted(o.name for o in c.all_objects if o.name in props_too and o.name not in bad)
            if bad:
                foreign[c.name] = bad
    two = sorted(o.name for o in bpy.data.objects if len(layer_of(o)) > 1)
    mats, seen = [], set()
    if "Set" in bpy.data.collections:
        for o in bpy.data.collections["Set"].all_objects:
            for slot in getattr(o, "material_slots", []):
                m = slot.material
                if m is None or (o.name, m.name) in seen:
                    continue
                seen.add((o.name, m.name))
                rgb, tex = material_color(m)
                h, s, v = colorsys.rgb_to_hsv(*[max(0.0, min(1.0, x)) for x in rgb])
                mats.append({"object": o.name, "material": m.name, "color": r4(rgb), "saturation": round(s, 3),
                             "value": round(v, 3), "textured": tex})
    out = {"version": VERSION, "frames": [sc.frame_start, sc.frame_end],
           "fps": round(sc.render.fps / (sc.render.fps_base or 1.0), 3),
           "resolution": [sc.render.resolution_x, sc.render.resolution_y],
           "layers": layers, "mannequin_foreign": foreign, "in_two_layers": two,
           "rigs": {n: {"collection": R["collection"], "timeline": R["timeline"], "joints": R["joints"],
                        "scale_leaks": sorted(R["leaks"].values(), key=lambda x: x["bone"])} for n, R in rigs.items()},
           "camera": {"frames": cam_frames, "moves": bool(layers.get("camera_moves"))},
           "props": [{k: v for k, v in P.items() if k != "obj"} for P in props],
           "set_materials": mats, "sounds": sound_strips(sc)}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print("CHECK_SHOT", a.out, sc.frame_end - sc.frame_start + 1, "frames", len(rigs), "rigs")


if __name__ == "__main__":
    main()
