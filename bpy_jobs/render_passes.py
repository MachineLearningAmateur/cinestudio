"""Render the raw passes of a shot (run inside Blender, headless).

    blender -b shot.blend --python bpy_jobs/render_passes.py -- --out <dir> [--character Mannequin --rig MannequinRig]

Writes into <dir>:
  rgb/rgb_####.png        the Blender render (EEVEE)
  depth_m.npy             float32 (frames, H, W) camera-space Z in metres (0 = no surface)
  mask/mask_####.png      white where the character is visible (other objects hold it out)
  keypoints.json          per frame: 18 OpenPose (COCO) keypoints [x, y, in_frame] in pixels, from the rig's bones;
                          with finger bones (thumb/index/middle/ring/pinky_01..03.L/R) also 21 hand points per hand
  passes.json             frame range, fps, size, and what was rendered
The character's rig must use the cinestudio bone names (hips, spine, chest, neck, head, upper_arm.L/R,
forearm.L/R, hand.L/R, thigh.L/R, shin.L/R, foot.L/R); other rigs get no keypoints.
Face points (nose, eyes, ears) are fixed points on the head bone. The defaults fit the Cinema
Studio mannequin; another rig can set its own as a custom property on the armature object:
rig["cs_face_points"] = '{"nose": [x, y, z], "r_eye": [...], ...}' (rest armature space, metres).

Old pass files in <dir> are deleted first, and passes.json is written last, so a stopped render
never leaves a mix of old and new passes that looks complete. Scene settings changed for the
depth and mask passes are restored even if a render fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

# the middle of the head (rest armature space): face points are seen when they lie on its camera side
HEAD_CENTER = (0.0, 0.0, 1.66)
# minimum cos(angle) between "centre -> point" and "centre -> camera" for a face point to count as seen:
# the nose stays in a profile view; an eye goes when the head turns that side away; ears go only when the
# head turns that ear away (a back view keeps both ears and loses nose and eyes, as in OpenPose)
FACE_VISIBLE_DOT = {"nose": -0.2, "eye": -0.1, "ear": -0.35}
# COCO-18 order used by OpenPose: name, (bone, point in rest/armature space or None = bone head)
KEYPOINTS = [
    ("nose", ("head", (0.0, -0.12, 1.66))), ("neck", ("neck", None)),
    ("r_shoulder", ("upper_arm.R", None)), ("r_elbow", ("forearm.R", None)), ("r_wrist", ("hand.R", None)),
    ("l_shoulder", ("upper_arm.L", None)), ("l_elbow", ("forearm.L", None)), ("l_wrist", ("hand.L", None)),
    ("r_hip", ("thigh.R", None)), ("r_knee", ("shin.R", None)), ("r_ankle", ("foot.R", None)),
    ("l_hip", ("thigh.L", None)), ("l_knee", ("shin.L", None)), ("l_ankle", ("foot.L", None)),
    ("r_eye", ("head", (-0.04, -0.10, 1.69))), ("l_eye", ("head", (0.04, -0.10, 1.69))),
    ("r_ear", ("head", (-0.11, 0.0, 1.66))), ("l_ear", ("head", (0.11, 0.0, 1.66))),
]


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--character", default="Mannequin", help="collection of the character (mask); "
                    "comma-separated for several characters")
    ap.add_argument("--rig", default="MannequinRig", help="armature object for keypoints")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--exclude-from-depth", default="", help="comma-separated objects that read as 'far' (sky, backdrops)")
    return ap.parse_args(argv)


def out_size(sc):
    """The size Blender really renders (resolution percentage included)."""
    pct = sc.render.resolution_percentage / 100
    return int(sc.render.resolution_x * pct), int(sc.render.resolution_y * pct)


def clean(out):
    """Remove the passes of an earlier run."""
    for name in ("passes.json", "keypoints.json", "depth_m.npy"):
        path = os.path.join(out, name)
        if os.path.exists(path):
            os.remove(path)
    for sub in ("rgb", "mask", "_depth_exr", "controls"):
        d = os.path.join(out, sub)
        if os.path.isdir(d):
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))


def set_engine(sc):
    try:
        sc.render.engine = "BLENDER_EEVEE"
    except TypeError:
        pass


def render_rgb(sc, out, samples):
    d = os.path.join(out, "rgb")
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    sc.eevee.taa_render_samples = samples
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.render.filepath = os.path.join(d, "rgb_####")
    bpy.ops.render.render(animation=True)


def render_depth(sc, out, exclude):
    import numpy as np
    exr = os.path.join(out, "_depth_exr")
    os.makedirs(exr, exist_ok=True)
    dm = bpy.data.materials.new("_cs_depth")
    dm.use_nodes = True
    nt = dm.node_tree
    nt.nodes.clear()
    cd, em, mo = nt.nodes.new("ShaderNodeCameraData"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(cd.outputs["View Z Depth"], em.inputs["Color"])
    nt.links.new(em.outputs[0], mo.inputs["Surface"])
    vl = sc.view_layers[0]
    override = vl.material_override
    hidden = [o for o in bpy.data.objects if o.name in exclude and not o.hide_render]
    world = sc.world
    samples, filt = sc.eevee.taa_render_samples, sc.render.filter_size
    try:
        vl.material_override = dm
        for o in hidden:
            o.hide_render = True
        sc.world = None
        sc.eevee.taa_render_samples, sc.render.filter_size = 1, 0.0
        s = sc.render.image_settings
        s.file_format, s.color_mode, s.color_depth, s.exr_codec = "OPEN_EXR", "RGB", "32", "ZIP"
        sc.render.filepath = os.path.join(exr, "depth_####")
        bpy.ops.render.render(animation=True)
    finally:
        vl.material_override = override
        for o in hidden:
            o.hide_render = False
        sc.world = world
        sc.eevee.taa_render_samples, sc.render.filter_size = samples, filt
        bpy.data.materials.remove(dm)
    frames = []
    for f in range(sc.frame_start, sc.frame_end + 1):
        img = bpy.data.images.load(os.path.join(exr, f"depth_{f:04d}.exr"), check_existing=False)
        img.colorspace_settings.name = "Non-Color"
        a = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
        img.pixels.foreach_get(a)
        z = a.reshape(img.size[1], img.size[0], 4)[::-1, :, 0].copy()
        z[~np.isfinite(z) | (z < 0)] = 0.0          # no valid surface = "far"
        frames.append(z)
        bpy.data.images.remove(img)
        os.remove(os.path.join(exr, f"depth_{f:04d}.exr"))
    os.rmdir(exr)
    np.save(os.path.join(out, "depth_m.npy"), np.stack(frames))


def render_mask(sc, out, character):
    d = os.path.join(out, "mask")
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    keep = {o.name for c in character.split(",") if c.strip() in bpy.data.collections      # "A,B": several characters
            for o in bpy.data.collections[c.strip()].all_objects}                         # (one mask for all of them)
    white = bpy.data.materials.new("_cs_white")
    white.use_nodes = True
    nt = white.node_tree
    nt.nodes.clear()
    em, mo = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], mo.inputs["Surface"])
    vl = sc.view_layers[0]
    override = vl.material_override
    holdouts = [o for o in bpy.data.objects if o.type in {"MESH", "CURVE"} and o.name not in keep and not o.is_holdout]
    transparent, samples = sc.render.film_transparent, sc.eevee.taa_render_samples
    try:
        vl.material_override = white
        for o in holdouts:
            o.is_holdout = True
        sc.render.film_transparent = True
        sc.eevee.taa_render_samples = 4
        s = sc.render.image_settings
        s.file_format, s.color_mode, s.color_depth = "PNG", "RGBA", "8"
        sc.render.filepath = os.path.join(d, "mask_####")
        bpy.ops.render.render(animation=True)
    finally:
        for o in holdouts:
            o.is_holdout = False
        vl.material_override = override
        sc.render.film_transparent = transparent
        sc.eevee.taa_render_samples = samples
        bpy.data.materials.remove(white)


def hand_points(side):
    """OpenPose hand-21 order: wrist, then thumb, index, middle, ring, pinky (4 points each, base to tip).
    Each point is (bone, "head"|"tail")."""
    pts = [("hand" + side, "head")]
    for f in ("thumb", "index", "middle", "ring", "pinky"):
        pts += [(f"{f}_01{side}", "head"), (f"{f}_02{side}", "head"), (f"{f}_03{side}", "head"), (f"{f}_03{side}", "tail")]
    return pts


def _yaw(sc, forward, to_cam):
    """[degrees, side]: the angle between the direction a part points (the face: head centre -> nose; the body: the
    chest's front) and the direction to the camera (0 = it looks into the camera, 90 = a profile, 180 = seen from
    behind), and the side of the picture it points to ("left" / "right"). The face-point flags cannot give this:
    an eye counts as seen up to about 75 degrees (OpenPose's convention), so a three-quarter view looks frontal."""
    import math
    if forward.length < 1e-6:
        return [0.0, "right"]
    deg = math.degrees(forward.normalized().angle(to_cam))
    side = "right" if (sc.camera.matrix_world.to_3x3().inverted() @ forward).x > 0 else "left"
    return [round(deg, 1), side]


def keypoints(sc, rig_name):
    """Body (COCO-18) keypoints, plus hand-21 keypoints when the rig has the cinestudio finger bones."""
    rig = bpy.data.objects.get(rig_name)
    if rig is None or any(b not in rig.pose.bones for _, (b, _) in KEYPOINTS):
        return None
    face = json.loads(rig["cs_face_points"]) if "cs_face_points" in rig.keys() else {}
    points = [(n, (b, tuple(face[n]) if n in face and p is not None else p)) for n, (b, p) in KEYPOINTS]
    hands = {s: hand_points(s) for s in (".R", ".L") if all(b in rig.pose.bones for b, _ in hand_points(s))}
    w, h = out_size(sc)

    def project(p):
        c = world_to_camera_view(sc, sc.camera, p)
        inside = 0.0 <= c.x <= 1.0 and 0.0 <= c.y <= 1.0 and c.z > 0
        return [round(c.x * w, 2), round((1 - c.y) * h, 2), 1 if inside else 0]

    face_names = {"nose", "r_eye", "l_eye", "r_ear", "l_ear"}
    head_center = tuple(face["head_center"]) if "head_center" in face else HEAD_CENTER

    def fixed(pb, rest_point):     # a point fixed to the bone: rest armature space -> posed world space
        return rig.matrix_world @ pb.matrix @ pb.bone.matrix_local.inverted() @ Vector(rest_point)

    nose_rest = dict(points)["nose"][1]
    frames, hand_frames, head_yaw, body_yaw = [], {s: [] for s in hands}, [], []
    for f in range(sc.frame_start, sc.frame_end + 1):
        sc.frame_set(f)
        pts = []
        head = rig.pose.bones["head"]
        center = fixed(head, head_center)
        to_cam = (sc.camera.matrix_world.translation - center).normalized()
        head_yaw.append(_yaw(sc, fixed(head, nose_rest) - center, to_cam))
        # the body's front, level: the shoulder line (right -> left) crossed with up (the rest figure faces -Y)
        across = (rig.matrix_world @ rig.pose.bones["upper_arm.L"].head) - (rig.matrix_world @ rig.pose.bones["upper_arm.R"].head)
        chest = (rig.matrix_world @ rig.pose.bones["upper_arm.L"].head + rig.matrix_world @ rig.pose.bones["upper_arm.R"].head) / 2
        level = lambda v: Vector((v.x, v.y, 0.0))  # noqa: E731
        body_yaw.append(_yaw(sc, level(across.cross(Vector((0.0, 0.0, 1.0)))),
                                  level(sc.camera.matrix_world.translation - chest).normalized()))
        for name, (bone, rest_point) in points:
            pb = rig.pose.bones[bone]
            p = rig.matrix_world @ pb.head if rest_point is None else fixed(pb, rest_point)
            q = project(p)
            if name in face_names and q[2]:
                # OpenPose convention: a face point is drawn only when it is on the side of the head that faces
                # the camera. A back view then has no nose and no eyes (else the model turns the face round).
                off = (p - center)
                if off.length > 1e-6 and off.normalized().dot(to_cam) < FACE_VISIBLE_DOT[name[-3:] if "_" in name else name]:
                    q[2] = 0
            pts.append(q)
        frames.append(pts)
        for s, hp in hands.items():
            hand_frames[s].append([project(rig.matrix_world @ getattr(rig.pose.bones[b], end)) for b, end in hp])
    sc.frame_set(sc.frame_start)
    out = {"format": "coco18", "names": [n for n, _ in KEYPOINTS], "frames": frames, "head_yaw": head_yaw,
           "body_yaw": body_yaw}
    if hands:      # "R" / "L": per frame 21 points [x, y, in_frame]
        out["hands"] = {"format": "openpose_hand21", **{s.strip("."): v for s, v in hand_frames.items()}}
    return out


def camera_azimuths(sc, rig_name):
    """Per frame: the camera's angle around the character, in degrees, measured in the rig's own
    frame (0 = in front of the figure, which faces -Y; counter-clockwise seen from above = positive).
    Unwrapped, so a full orbit ends at 360, not 0. None without a rig."""
    import math
    rig = bpy.data.objects.get(rig_name)
    if rig is None:
        return None
    out, prev = [], None
    for f in range(sc.frame_start, sc.frame_end + 1):
        sc.frame_set(f)
        local = rig.matrix_world.inverted() @ sc.camera.matrix_world.translation
        a = math.degrees(math.atan2(local.x, -local.y))
        if prev is not None:
            while a - prev > 180:
                a -= 360
            while a - prev < -180:
                a += 360
        out.append(round(a, 3))
        prev = a
    sc.frame_set(sc.frame_start)
    return out


def main():
    a = args()
    sc = bpy.context.scene
    if sc.frame_step != 1:
        sys.exit(f"render_passes: frame_step is {sc.frame_step}; control videos need every frame (set it to 1)")
    if sc.camera is None:
        sys.exit("render_passes: the scene has no active camera")
    set_engine(sc)
    a.out = os.path.abspath(a.out)       # Blender resolves relative render paths differently from Python
    os.makedirs(a.out, exist_ok=True)
    clean(a.out)
    exclude = {s.strip() for s in a.exclude_from_depth.split(",") if s.strip()}
    render_rgb(sc, a.out, a.samples)
    render_depth(sc, a.out, exclude)
    has_char = any(c.strip() in bpy.data.collections for c in a.character.split(","))
    if has_char:
        render_mask(sc, a.out, a.character)
    kp = keypoints(sc, a.rig)
    if kp:
        with open(os.path.join(a.out, "keypoints.json"), "w") as fh:
            json.dump(kp, fh)
    info = {"frame_start": sc.frame_start, "frame_end": sc.frame_end, "frames": sc.frame_end - sc.frame_start + 1,
            "fps": sc.render.fps / sc.render.fps_base, "width": out_size(sc)[0], "height": out_size(sc)[1],
            "mask": has_char, "keypoints": bool(kp), "blend": bpy.data.filepath,
            "camera_azimuth_deg": camera_azimuths(sc, a.rig)}
    with open(os.path.join(a.out, "passes.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print("PASSES DONE", json.dumps(info))


if __name__ == "__main__":
    main()
