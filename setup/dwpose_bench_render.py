"""The DWPose bench render (run inside Blender): a shot with one or more mannequins, a few frames, on the CPU.

    blender -b SHOT.blend --factory-startup --python setup/dwpose_bench_render.py -- --out DIR
        [--looks googly_eyes,hair] [--every 8] [--percent 50] [--samples 16]
        [--characters Mannequin:MannequinRig,Mannequin2:MannequinRig2]

Writes a folder that setup/dwpose_score.py reads:
  rgb/rgb_####.png   the frames (Cycles on the CPU, so a GPU job can run at the same time)
  boxes.json         per frame and per character: the box of the character's projected mesh [x0, y0, x1, y1]
  keypoints.json     the true points from each rig ("characters": [...], format of bpy_jobs/render_passes.py)
  passes.json        frame range, size, camera azimuth, looks
The camera, the light and the animation of the shot do not change: only the mannequin's look (--looks).
"""

import argparse
import json
import os
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "blender_lib"), os.path.join(ROOT, "bpy_jobs")]

import mannequin_looks  # noqa: E402
import render_passes  # noqa: E402


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--looks", default="", help="comma list of mannequin_looks.LOOKS")
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--percent", type=int, default=50, help="render size in % of the shot (50: 1536x896 -> 768x448)")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--characters", default="Mannequin:MannequinRig", help="Collection:Rig, comma separated")
    ap.add_argument("--occlusion", action="store_true", help="write occlusion.json: per frame and character, the "
                    "points that another character hides, and the distance to the camera (drawing order)")
    ap.add_argument("--isolate", action="store_true", help="also render each character alone (the other characters "
                    "hidden) to iso/<n>/rgb_####.png: a detector input with one person")
    return ap.parse_args(argv)


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
    if not inside:              # no part of the character is in the frame
        return None
    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


def joint_points(sc, rig):
    """World positions of the scored points of one rig at the current frame: body (COCO-18, as
    render_passes.keypoints) and hands (OpenPose hand-21)."""
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
    surface on the ray from the camera belongs to another character; the character's own body does not count)."""
    dg = bpy.context.evaluated_depsgraph_get()
    cam = sc.camera.matrix_world.translation
    owner = {ob.name: c for c, _ in chars for ob in bpy.data.collections[c].all_objects}
    out = []
    for c, r in chars:
        rig = bpy.data.objects[r]
        pts = joint_points(sc, rig)
        res = {"depth": round((rig.matrix_world @ rig.pose.bones["hips"].head - cam).length, 3)}
        for part, ps in pts.items():
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
    os.makedirs(os.path.join(out, "rgb"), exist_ok=True)
    chars = [tuple(c.split(":")) for c in a.characters.split(",") if c]
    looks = [x for x in a.looks.split(",") if x]
    for col_name, rig_name in chars:
        mannequin_looks.apply(bpy.data.objects[rig_name], bpy.data.collections[col_name], looks)

    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = a.samples
    sc.cycles.use_denoising = True
    sc.render.resolution_percentage = a.percent
    sc.render.image_settings.file_format = "PNG"
    w, h = render_passes.out_size(sc)
    frames = list(range(sc.frame_start, sc.frame_end + 1, a.every))
    if frames[-1] != sc.frame_end:
        frames.append(sc.frame_end)

    boxes, occ = {}, {}
    for f in frames:
        sc.frame_set(f)
        sc.render.filepath = os.path.join(out, "rgb", f"rgb_{f:04d}.png")
        bpy.ops.render.render(write_still=True)
        boxes[f] = [box(sc, bpy.data.collections[c], w, h) for c, _ in chars]
        if a.occlusion:
            occ[f] = occlusion(sc, chars)
        if a.isolate:              # each character alone (with one character: the same picture again)
            old = {c2: bpy.data.collections[c2].hide_render for c2, _ in chars}
            try:
                for n, (c, _) in enumerate(chars):
                    for m, (c2, _) in enumerate(chars):
                        bpy.data.collections[c2].hide_render = old[c2] or m != n
                    sc.render.filepath = os.path.join(out, "iso", str(n), f"rgb_{f:04d}.png")
                    bpy.ops.render.render(write_still=True)
            finally:
                for c2, v in old.items():
                    bpy.data.collections[c2].hide_render = v
    people = []
    for col_name, rig_name in chars:
        kp = render_passes.keypoints(sc, rig_name)
        kp["name"] = col_name
        people.append(kp)
    with open(os.path.join(out, "keypoints.json"), "w") as fh:
        json.dump({"format": "coco18", "characters": people}, fh)
    if a.occlusion:
        with open(os.path.join(out, "occlusion.json"), "w") as fh:
            json.dump({"characters": [c for c, _ in chars], "frames": occ}, fh)
    with open(os.path.join(out, "boxes.json"), "w") as fh:
        json.dump({"characters": [c for c, _ in chars], "frames": boxes}, fh)
    info = {"frame_start": sc.frame_start, "frame_end": sc.frame_end, "frames": sc.frame_end - sc.frame_start + 1,
            "rendered": frames, "width": w, "height": h, "looks": looks, "engine": "CYCLES CPU",
            "samples": a.samples, "blend": bpy.data.filepath, "isolate": a.isolate,
            "camera_azimuth_deg": render_passes.camera_azimuths(sc, chars[0][1]),
            "character_azimuth_deg": [render_passes.camera_azimuths(sc, r) for _, r in chars]}
    with open(os.path.join(out, "passes.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print("BENCH DONE", json.dumps({k: info[k] for k in ("rendered", "width", "height", "looks")}))


if __name__ == "__main__":
    main()
