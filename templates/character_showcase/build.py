"""Character showcase template: a neutral mannequin, a simple set, a slow orbiting camera.

The AI later replaces the mannequin with your character (reference image) in your style; the
mannequin only gives pose and placement, the camera gives the angles.

    blender -b --factory-startup --python templates/character_showcase/build.py -- --out shot.blend [--frames 121]

Everything is built from code (no external assets), so the template is reproducible and free to share.
Units: metres. The mannequin faces -Y; its right side is -X. Camera speed is kept low (about 0.75 deg/frame)
because video models smear fast camera turns.
"""

from __future__ import annotations

import argparse
import math
import sys

import os

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from blender_lib import mannequin as mq  # noqa: E402

# ---------------------------------------------------------------- configuration
FPS = 24
CAM_RADIUS, CAM_TARGET_Z = 4.0, 0.92   # whole figure in frame, a little head room
# Degrees around the mannequin; 0 = straight in front (-Y side). 45 degrees in 120 frames with the easing
# below peaks at ~0.49 deg/frame: all tested video models smear the picture above ~0.6 deg/frame.
# More angles = more clips (one orbit segment per clip), not a faster camera.
CAM_AZ = (-22.5, 22.5)
ORBIT = [45.0]                          # set by build(); read by animate()
CAM_Z = (1.25, 1.65)
COLORS = {"mannequin": (0.72, 0.66, 0.57), "joint": (0.42, 0.38, 0.33), "stripe": (0.15, 0.17, 0.22),
          "floor": (0.10, 0.10, 0.11), "backdrop": (0.30, 0.31, 0.33), "prop": (0.45, 0.20, 0.16), "rug": (0.18, 0.22, 0.30)}
RELAXED = (0.30, 0.0)        # finger curl, spread: hands at rest
OPEN = (0.05, 8.0)           # open hand with the fingers a little apart (the wave)


# ---------------------------------------------------------------- helpers
def material(name, color, roughness=0.6):
    m = bpy.data.materials.new("cs_" + name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = roughness
    return m


def collection(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


def add_mesh(op, col, mat, **kw):
    op(**kw)
    ob = bpy.context.active_object
    for c in ob.users_collection:
        c.objects.unlink(ob)
    col.objects.link(ob)
    ob.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return ob


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- scene
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = FPS
    sc.unit_settings.system = "METRIC"
    return sc


def build_set(M):
    col = collection("Set")
    add_mesh(bpy.ops.mesh.primitive_plane_add, col, M["floor"], size=30, location=(0, 0, 0))
    # curved backdrop (cyclorama) behind the mannequin, open toward the camera side
    bpy.ops.curve.primitive_bezier_circle_add(radius=7.0, location=(0, 0, 0))
    curve = bpy.context.active_object
    curve.data.extrude = 2.5
    curve.data.resolution_u = 32
    curve.location.z = 2.5
    for c in curve.users_collection:
        c.objects.unlink(curve)
    col.objects.link(curve)
    curve.data.materials.append(M["backdrop"])
    add_mesh(bpy.ops.mesh.primitive_cylinder_add, col, M["rug"], radius=1.1, depth=0.01, location=(0, 0, 0.005))
    # props at different depths: they show the camera motion (parallax)
    if ORBIT[0] >= 180:
        # full orbit: every prop stands OUTSIDE the camera circle (radius CAM_RADIUS), so the camera never
        # passes next to one and none ever hides the mannequin; spread around the whole circle
        r = CAM_RADIUS + 1.6
        # no prop near 180 degrees: it would stand right behind the figure in the first and last frames
        for deg, kind in ((40, "pillar"), (95, "box"), (150, "low"), (215, "pillar"), (265, "box"), (320, "low")):
            a = math.radians(deg)
            x, y = math.sin(a) * r, -math.cos(a) * r
            if kind == "pillar":
                add_mesh(bpy.ops.mesh.primitive_cylinder_add, col, M["prop"], radius=0.25, depth=2.4, location=(x, y, 1.2))
            elif kind == "box":
                add_mesh(bpy.ops.mesh.primitive_cube_add, col, M["prop"], size=1, location=(x, y, 0.6), scale=(0.5, 0.5, 1.2))
            else:
                add_mesh(bpy.ops.mesh.primitive_cube_add, col, M["stripe"], size=1, location=(x, y, 0.35), scale=(0.7, 0.35, 0.7))
        return col
    add_mesh(bpy.ops.mesh.primitive_cube_add, col, M["prop"], size=1, location=(-2.2, 1.4, 0.6), scale=(0.5, 0.5, 1.2))
    add_mesh(bpy.ops.mesh.primitive_cylinder_add, col, M["prop"], radius=0.25, depth=2.4, location=(2.4, 0.8, 1.2))
    add_mesh(bpy.ops.mesh.primitive_cube_add, col, M["stripe"], size=1, location=(1.4, 2.6, 0.35), scale=(0.7, 0.35, 0.7))
    return col


def _rest_dir(bone):
    head, tail = mq.REST[bone]
    return (Vector(tail) - Vector(head)).normalized()


def _wave_phases(frames):
    """Key poses of the right arm: (frame, {bone: direction}, hand (curl, spread)). Between two key poses the
    directions are blended smoothly, so the arm never snaps (an earlier version dropped in one jump)."""
    F = frames
    # a full orbit waves early, while the camera passes the front (it starts 60 degrees to the left)
    r0, w0, w1 = (1, int(F * 0.20), int(F * 0.42)) if ORBIT[0] >= 180 else (int(F * 0.22), int(F * 0.40), int(F * 0.78))
    rest = {b: _rest_dir(b) for b in ("upper_arm.R", "forearm.R", "hand.R")}
    up = Vector((-0.82, -0.18, 0.52))                   # upper arm out to the side, a little up
    phases = [(1, rest, RELAXED), (r0, rest, RELAXED),
              (r0 + (w0 - r0) // 2, {"upper_arm.R": Vector((-0.60, -0.45, -0.10)),      # elbow bends first
                                     "forearm.R": Vector((-0.20, -0.60, 0.75)),
                                     "hand.R": Vector((-0.20, -0.55, 0.80))}, OPEN)]
    k = 0
    for f in range(w0, w1 + 1, 6):                       # the wave: the forearm swings left and right
        sw = -0.42 if k % 2 == 0 else 0.42
        phases.append((f, {"upper_arm.R": up, "forearm.R": Vector((sw, -0.18, 0.89)),
                           "hand.R": Vector((sw * 1.3, -0.20, 0.85))}, OPEN))
        k += 1
    last = phases[-1][0]
    phases += [(last + 11, {"upper_arm.R": Vector((-0.55, -0.15, -0.82)),                 # down to the side (the
                           "forearm.R": Vector((-0.35, -0.35, -0.87)),                    # forearm never points
                           "hand.R": Vector((-0.30, -0.30, -0.90))}, RELAXED),           # at the camera)
               (last + 22, rest, RELAXED), (F, rest, RELAXED)]
    return phases


def _arm_at(phases, f):
    """Blended right-arm directions, open-hand amount and whether the palm faces front at frame f."""
    for (fa, pa, ha), (fb, pb, hb) in zip(phases, phases[1:]):
        if fa <= f <= fb:
            t = smoothstep((f - fa) / max(1, fb - fa))
            dirs = {b: (pa[b].normalized().lerp(pb[b].normalized(), t)).normalized() for b in pa}
            hand = (ha[0] + (hb[0] - ha[0]) * t, ha[1] + (hb[1] - ha[1]) * t)
            wa, wb = (1.0 if ha is OPEN else 0.0), (1.0 if hb is OPEN else 0.0)
            return dirs, hand, wa + (wb - wa) * t          # palm-to-front weight, blended like the arm
    return phases[-1][1], phases[-1][2], 0.0


def animate(rig, frames, cam_dir_at):
    """Idle stand; the head follows the camera; the right hand waves in the middle: the arm rises with the
    elbow bent, the forearm swings left and right with the palm to the front (not to the camera: in a full
    orbit that twisted the wrist round), and the arm comes down through a middle pose."""
    phases = _wave_phases(frames)
    keys = sorted(set(range(1, frames + 1, 3)) | {p[0] for p in phases} | {frames})
    for f in keys:
        d = cam_dir_at(f)
        az = math.degrees(math.atan2(d.x, -d.y))          # camera azimuth; the face points to -Y at rest
        if ORBIT[0] >= 180:   # full orbit: the head turns toward the camera smoothly, at most ~55 degrees
            look = 55.0 * math.sin(math.radians(az))
            yaws = {"neck": look * 0.35, "head": look * 0.65}
        else:
            yaws = {"neck": az * 0.25, "head": az * 0.45}
        dirs, hand, palm_w = _arm_at(phases, f)
        palms = {"hand.R": (Vector((-0.15, -1.0, 0.0)), palm_w)} if palm_w > 0 else {}
        mq.key_pose(rig, f, dirs, yaws, palms, {".L": RELAXED, ".R": hand})


def build_camera(frames, target):
    cam_data = bpy.data.cameras.new("ShotCam")
    cam_data.lens = 30
    cam = bpy.data.objects.new("ShotCam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis, track.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"

    full = ORBIT[0] >= 180

    def cam_pos(f):
        lin = (f - 1) / max(1, frames - 1)
        t = lin if full else smoothstep(lin) * 0.6 + lin * 0.4    # full orbit: constant speed
        az = math.radians(CAM_AZ[0] + (CAM_AZ[1] - CAM_AZ[0]) * t)
        z = sum(CAM_Z) / 2 if full else CAM_Z[0] + (CAM_Z[1] - CAM_Z[0]) * t
        return Vector((math.sin(az) * CAM_RADIUS, -math.cos(az) * CAM_RADIUS, z))

    for f in range(1, frames + 1, 1 if full else 4):
        cam.location = cam_pos(f)
        cam.keyframe_insert("location", frame=f)
    cam.location = cam_pos(frames)
    cam.keyframe_insert("location", frame=frames)

    def cam_dir_at(f):        # from the mannequin's head toward the camera
        return (cam_pos(f) - Vector((0, 0, 1.65))).normalized()
    return cam, cam_dir_at


def build_lights():
    for name, loc, energy, size, color in (("Key", (-2.5, -3.0, 3.2), 450, 2.0, (1.0, 0.95, 0.88)),
                                           ("Fill", (3.0, -2.0, 2.0), 150, 3.0, (0.85, 0.9, 1.0)),
                                           ("Rim", (0.5, 2.5, 3.0), 300, 1.5, (1.0, 1.0, 1.0))):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy, ld.size, ld.color = energy, size, color
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        lo.rotation_euler = (Vector((0, 0, 1.2)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.collection.objects.link(lo)
    w = bpy.data.worlds.new("World")
    bpy.context.scene.world = w
    w.use_nodes = True
    next(n for n in w.node_tree.nodes if n.type == "BACKGROUND").inputs["Color"].default_value = (0.05, 0.05, 0.06, 1)


NIGHT = {"floor": (0.035, 0.037, 0.05), "backdrop": (0.012, 0.018, 0.045), "rug": (0.10, 0.10, 0.12),
         "prop": (0.10, 0.09, 0.09)}


def night_set():
    """Light the SET like a moonlit night (the mannequin keeps its colour and stays lit): with video-to-video about 2%
    of the render stays in the video, and a bright grey room leaked into the clips (and made the clip jump away from a
    night first frame). Dark blue sky and walls, a dark stone floor, cold moonlight as the key light, warm lantern
    lights near the edge of the set."""
    for name, rgb in NIGHT.items():
        m = bpy.data.materials.get("cs_" + name)
        if m:
            bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
            bsdf.inputs["Base Color"].default_value = (*rgb, 1)
    w = bpy.context.scene.world
    next(n for n in w.node_tree.nodes if n.type == "BACKGROUND").inputs["Color"].default_value = (0.006, 0.009, 0.025, 1)
    looks = {"Key": ((0.62, 0.72, 1.0), 1.0), "Fill": ((0.45, 0.55, 0.9), 0.35), "Rim": ((0.7, 0.8, 1.0), 1.2)}
    for name, (color, k) in looks.items():
        ld = bpy.data.lights.get(name)
        if ld:
            ld.color, ld.energy = color, ld.energy * k
    col = bpy.context.scene.collection
    for i, deg in enumerate((60, 180, 300)):                 # warm lanterns round the set (outside the camera path)
        a = math.radians(deg)
        ld = bpy.data.lights.new(f"Lantern{i + 1}", "POINT")
        ld.color, ld.energy, ld.shadow_soft_size = (1.0, 0.62, 0.3), 120.0, 0.15
        lo = bpy.data.objects.new(f"Lantern{i + 1}", ld)
        lo.location = (math.sin(a) * 6.2, -math.cos(a) * 6.2, 1.2)
        col.objects.link(lo)


def build(frames=121, width=1536, height=896, orbit=45.0):
    """orbit: degrees the camera travels around the mannequin. Below 180: centred on the front with an
    eased speed (the original showcase). 180 or more: a full-turn style orbit starting in front, at a
    constant speed and height (a 360 needs about 600 frames to stay under ~0.6 deg/frame)."""
    global CAM_AZ
    # a full orbit starts at the front: the first frame then matches the front panel of the character sheet
    # (a start 60 degrees to the left made the video's first pose differ from the mannequin's)
    CAM_AZ = (0.0, float(orbit)) if orbit >= 180 else (-orbit / 2, orbit / 2)
    ORBIT[0] = float(orbit)
    sc = reset()
    sc.frame_start, sc.frame_end = 1, frames
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = width, height, 100
    M = {k: material(k, v) for k, v in COLORS.items()}
    build_set(M)
    rig, _ = mq.build(M)
    target = bpy.data.objects.new("CamTarget", None)
    target.location = (0, 0, CAM_TARGET_Z)
    sc.collection.objects.link(target)
    cam, cam_dir_at = build_camera(frames, target)
    animate(rig, frames, cam_dir_at)
    build_lights()
    sc.frame_set(1)
    return sc


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=121)
    ap.add_argument("--width", type=int, default=1536, help="2 x the 768x448 generation size (same shape)")
    ap.add_argument("--height", type=int, default=896)
    ap.add_argument("--orbit", type=float, default=45.0, help="camera orbit in degrees (360 = all the way around)")
    ap.add_argument("--night", action="store_true", help="light the set like a moonlit night (see night_set)")
    a = ap.parse_args(argv)
    build(a.frames, a.width, a.height, a.orbit)
    if a.night:
        night_set()
    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    print("SAVED", a.out)


if __name__ == "__main__":
    main()
