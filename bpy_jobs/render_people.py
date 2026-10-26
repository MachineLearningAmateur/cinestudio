"""Detector input for shots with several characters (run inside Blender on the run's frozen shot):

    blender -b shot.blend --python bpy_jobs/render_people.py -- --out <dir> --percent 50

Finds every cinestudio mannequin rig in the scene. With fewer than two, it only writes people.json with
"characters": [] (the run then draws the control from the rig as before). With two or more, for every frame:
  iso/<n>/rgb_####.png   character n ALONE (the other characters hidden; the same camera, light and set), EEVEE
  people.json            per character: name, rig, the rig's points (render_passes.keypoints format, so the face
                         rule is in the visibility flags); per frame and character: the box of its projected mesh,
                         its distance to the camera, and the points that ANOTHER character hides (ray test)
setup/... is not needed: cinestudio/pose_people.py reads this folder.
"""
import argparse
import json
import os
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_passes  # noqa: E402

NEEDED_BONES = ("hips", "head", "hand.L", "hand.R", "thumb_01.L", "foot.R")


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--percent", type=int, default=50, help="render size in % of the shot (the control size)")
    ap.add_argument("--samples", type=int, default=16)
    return ap.parse_args(argv)


def mannequins():
    """(collection name, rig name) of every mannequin rig, in a stable order."""
    out = []
    for ob in sorted(bpy.data.objects, key=lambda o: o.name):
        if ob.type == "ARMATURE" and all(b in ob.pose.bones for b in NEEDED_BONES) and ob.users_collection:
            col = next((c for c in ob.users_collection if c.name != "Scene Collection"), ob.users_collection[0])
            out.append((col.name, ob.name))
    return out


def box(sc, col, w, h):
    dg = bpy.context.evaluated_depsgraph_get()
    xs, ys, inside = [], [], False
    for ob in col.all_objects:
        if ob.type != "MESH" or ob.hide_render:
            continue
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        mw = ev.matrix_world
        for v in me.vertices:
            c = world_to_camera_view(sc, sc.camera, mw @ v.co)
            if c.z > 0:
                inside = inside or (0 <= c.x <= 1 and 0 <= c.y <= 1)
                xs.append(min(max(c.x, 0.0), 1.0) * w)
                ys.append(min(max(1 - c.y, 0.0), 1.0) * h)
        ev.to_mesh_clear()
    if not inside:
        return None
    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


def joint_points(rig):
    """World positions of the body (COCO-18) and hand (21 each) points, as render_passes.keypoints places them."""
    face = json.loads(rig["cs_face_points"]) if "cs_face_points" in rig.keys() else {}
    out = {"body": []}
    for name, (bone, rest) in render_passes.KEYPOINTS:
        pb = rig.pose.bones[bone]
        rest = tuple(face[name]) if name in face and rest is not None else rest
        out["body"].append(rig.matrix_world @ pb.head if rest is None else
                           rig.matrix_world @ pb.matrix @ pb.bone.matrix_local.inverted() @ Vector(rest))
    for s in (".R", ".L"):
        pts = render_passes.hand_points(s)
        if all(b in rig.pose.bones for b, _ in pts):
            out[s.strip(".")] = [rig.matrix_world @ getattr(rig.pose.bones[b], end) for b, end in pts]
    return out


def occlusion(sc, chars):
    """Per character: distance camera -> hips, and per point True when ANOTHER character hides it (the first
    surface on the ray from the camera belongs to another character). Perspective cameras only."""
    dg = bpy.context.evaluated_depsgraph_get()
    cam = sc.camera.matrix_world.translation
    owner = {ob.name: c for c, _ in chars for ob in bpy.data.collections[c].all_objects}
    out = []
    for c, r in chars:
        rig = bpy.data.objects[r]
        res = {"depth": round((rig.matrix_world @ rig.pose.bones["hips"].head - cam).length, 3)}
        for part, ps in joint_points(rig).items():
            flags = []
            for wp in ps:
                d = wp - cam
                hit, _, _, _, ob, _ = sc.ray_cast(dg, cam, d.normalized(), distance=max(d.length - 0.01, 0.0))
                flags.append(bool(hit and ob is not None and owner.get(ob.original.name, c) != c))
            res[part] = flags
        out.append(res)
    return out


def main():
    a = args()
    sc = bpy.context.scene
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    chars = mannequins()
    if len(chars) < 2:
        with open(os.path.join(out, "people.json"), "w") as fh:
            json.dump({"characters": []}, fh)
        print("PEOPLE DONE", json.dumps({"characters": len(chars)}))
        return
    if sc.camera is None or sc.camera.data.type != "PERSP":
        sys.exit("render_people: the occlusion test needs a perspective camera")
    render_passes.set_engine(sc)
    sc.eevee.taa_render_samples = a.samples
    sc.render.resolution_percentage = a.percent
    sc.render.image_settings.file_format = "PNG"
    w, h = render_passes.out_size(sc)
    frames = {}
    old = {c: bpy.data.collections[c].hide_render for c, _ in chars}
    try:
        for f in range(sc.frame_start, sc.frame_end + 1):
            sc.frame_set(f)
            boxes = [box(sc, bpy.data.collections[c], w, h) for c, _ in chars]
            occ = occlusion(sc, chars)
            frames[f] = [{"box": b, **o} for b, o in zip(boxes, occ)]
            for n in range(len(chars)):
                for m, (c2, _) in enumerate(chars):
                    bpy.data.collections[c2].hide_render = old[c2] or m != n
                sc.render.filepath = os.path.join(out, "iso", str(n), f"rgb_{f:04d}.png")
                bpy.ops.render.render(write_still=True)
    finally:
        for c2, v in old.items():
            bpy.data.collections[c2].hide_render = v
    kps = [render_passes.keypoints(sc, r) for _, r in chars]
    info = {"characters": [{"name": c, "rig": r} for c, r in chars], "width": w, "height": h,
            "frame_start": sc.frame_start, "frame_end": sc.frame_end, "keypoints": kps, "frames": frames}
    with open(os.path.join(out, "people.json"), "w") as fh:
        json.dump(info, fh)
    print("PEOPLE DONE", json.dumps({"characters": len(chars), "frames": len(frames), "size": [w, h]}))


if __name__ == "__main__":
    main()
