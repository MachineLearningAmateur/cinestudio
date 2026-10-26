"""Built-in template: one mannequin waves while the camera orbits it once (360 degrees).

    blender -b --factory-startup --python templates/orbit_wave/build.py -- --out shot.blend [--frames 408] [--hold 48]

Version 2 (2026-10-22): the shot is SEGMENTED for the video's clips (121 frames, 25 shared with the next clip, so clip k
covers frames 96(k-1)+1 .. 96(k-1)+121), so nothing in the layout misleads a clip:
- the camera HOLDS at the front for `hold` frames at the start and at the end, and orbits 360 degrees between them at a
  near-constant speed (eased in and out over 16 frames; about 1.2 degrees per frame at the defaults);
- each wave happens only while the camera holds at the front and lies inside ONE clip, away from the shared frames:
  wave 1 in clip 1 (frames ~8-40), wave 2 in clip 4's own frames (~362-392, rising from ~348 when the body is already
  within ~10 degrees of the camera). Version 1 started wave 2 with the body at ~76 degrees and the head already turned
  toward the camera: LTX turned the whole body to the camera early and STEPPED to do it ("the model walks");
- the head follows the body (no look toward the camera): version 1 turned it up to 55 degrees toward the camera;
- 408 frames = 4 clips with real frames to the end (version 1: 360 frames, the last 49 frames of clip 4 held still);
- landmarks of different shapes and heights in every direction (no near-opposite pairs: those made the front and the
  back look alike, so the video could not see the turn). They stand outside the camera circle. A run with a panorama
  background hides them (bpy_jobs/pano_world.py).
Version 1 (360 frames, waves at 3-19% and 81-97%, the head looking at the camera) is kept in
.cinestudio/backup/orbit_wave_360_v1 (not in the repo).
"""
import argparse
import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "character_showcase"))
import build as showcase  # noqa: E402
from blender_lib import mannequin as mq  # noqa: E402

HOLD = 48                                  # front hold at each end (frames)
RAMP = 16                                  # the orbit eases in and out over this many frames
WAVE_LEN = 32                              # frames of forearm swings per wave
# (degrees around the figure, kind, size): every direction looks different
LANDMARKS = ((28, "pillar", 2.8), (72, "box", 1.0), (118, "low", 1.4), (152, "pillar", 1.6), (205, "arch", 2.2),
             (248, "thin", 3.2), (292, "stack", 1.0), (332, "low", 0.8))
SET = {"frames": 408, "hold": HOLD}         # set by main(); read by the camera and the waves


def orbit_deg(f, frames, hold):
    """Camera azimuth in degrees at frame f: 0 during the first hold, 360 during the last, and between them a
    trapezoid speed profile (RAMP frames of easing at each end, constant speed in the middle)."""
    n = frames - 2 * hold                  # orbit frames
    t = min(max(f - hold, 0), n)
    r = min(RAMP, n // 2)
    v = 360.0 / (n - r)                    # top speed (degrees per frame)
    if t <= r:
        return 0.5 * v * t * t / r
    if t >= n - r:
        u = n - t
        return 360.0 - 0.5 * v * u * u / r
    return 0.5 * v * r + v * (t - r)


def waves(frames, hold):
    """The two wave windows (first and last swing frame): each inside a front hold, inside one clip."""
    w1 = (8, 8 + WAVE_LEN)
    w2 = (frames - hold + 2, frames - hold + 2 + WAVE_LEN - 2)
    return w1, w2


def wave_phases(frames):
    """Right-arm key poses: rest, then for each window rise (elbow first), swing the forearm left/right every 6 frames
    with the palm to the front, come down through a middle pose, rest."""
    F = frames
    rest = {b: showcase._rest_dir(b) for b in ("upper_arm.R", "forearm.R", "hand.R")}
    up = Vector((-0.82, -0.18, 0.52))
    raise_mid = {"upper_arm.R": Vector((-0.60, -0.45, -0.10)), "forearm.R": Vector((-0.20, -0.60, 0.75)),
                 "hand.R": Vector((-0.20, -0.55, 0.80))}
    down_mid = {"upper_arm.R": Vector((-0.55, -0.15, -0.82)), "forearm.R": Vector((-0.35, -0.35, -0.87)),
                "hand.R": Vector((-0.30, -0.30, -0.90))}
    phases = [(1, rest, showcase.RELAXED)]
    for w0, w1 in waves(F, SET["hold"]):
        r0 = max(phases[-1][0] + 2, w0 - 14)
        phases += [(r0, rest, showcase.RELAXED), (r0 + (w0 - r0) // 2, raise_mid, showcase.OPEN)]
        k = 0
        for f in range(w0, w1 + 1, 6):
            sw = -0.42 if k % 2 == 0 else 0.42
            phases.append((f, {"upper_arm.R": up, "forearm.R": Vector((sw, -0.18, 0.89)),
                               "hand.R": Vector((sw * 1.3, -0.20, 0.85))}, showcase.OPEN))
            k += 1
        last = phases[-1][0]
        phases += [(min(F - 1, last + 7), down_mid, showcase.RELAXED), (min(F, last + 14), rest, showcase.RELAXED)]
    if phases[-1][0] < F:
        phases.append((F, rest, showcase.RELAXED))
    out, seen = [], set()
    for p in phases:                                   # strictly increasing frames
        if p[0] not in seen and (not out or p[0] > out[-1][0]):
            out.append(p)
            seen.add(p[0])
    return out


def build_camera(frames, target):
    """The orbit camera with a front hold at each end (constant height, radius and target of the showcase)."""
    cam_data = bpy.data.cameras.new("ShotCam")
    cam_data.lens = 30
    cam = bpy.data.objects.new("ShotCam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis, track.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    z = sum(showcase.CAM_Z) / 2

    def cam_pos(f):
        az = math.radians(orbit_deg(f, frames, SET["hold"]))
        return Vector((math.sin(az) * showcase.CAM_RADIUS, -math.cos(az) * showcase.CAM_RADIUS, z))

    for f in range(1, frames + 1):
        cam.location = cam_pos(f)
        cam.keyframe_insert("location", frame=f)

    def cam_dir_at(f):
        return (cam_pos(f) - Vector((0, 0, 1.65))).normalized()
    return cam, cam_dir_at


def animate(rig, frames, cam_dir_at):
    """Idle stand, feet planted, the head straight along the body (no look toward the camera), the right hand waves
    in the two front holds."""
    phases = wave_phases(frames)
    keys = sorted(set(range(1, frames + 1, 3)) | {p[0] for p in phases} | {frames})
    for f in keys:
        dirs, hand, palm_w = showcase._arm_at(phases, f)
        palms = {"hand.R": (Vector((-0.15, -1.0, 0.0)), palm_w)} if palm_w > 0 else {}
        mq.key_pose(rig, f, dirs, {"neck": 0.0, "head": 0.0}, palms, {".L": showcase.RELAXED, ".R": hand})


def build_set(M):
    col = showcase.collection("Set")
    showcase.add_mesh(bpy.ops.mesh.primitive_plane_add, col, M["floor"], size=30, location=(0, 0, 0))
    showcase.add_mesh(bpy.ops.mesh.primitive_cylinder_add, col, M["rug"], radius=1.1, depth=0.01, location=(0, 0, 0.005))
    r = showcase.CAM_RADIUS + 1.7
    for deg, kind, size in LANDMARKS:
        a = math.radians(deg)
        x, y = math.sin(a) * r, -math.cos(a) * r
        if kind == "pillar":
            showcase.add_mesh(bpy.ops.mesh.primitive_cylinder_add, col, M["prop"], radius=0.25, depth=size,
                              location=(x, y, size / 2))
        elif kind == "thin":
            showcase.add_mesh(bpy.ops.mesh.primitive_cylinder_add, col, M["stripe"], radius=0.09, depth=size,
                              location=(x, y, size / 2))
        elif kind == "box":
            showcase.add_mesh(bpy.ops.mesh.primitive_cube_add, col, M["prop"], size=1, location=(x, y, 0.5 * size),
                              scale=(0.5 * size, 0.5 * size, 0.5 * size))
        elif kind == "low":
            showcase.add_mesh(bpy.ops.mesh.primitive_cube_add, col, M["stripe"], size=1, location=(x, y, 0.2 * size),
                              scale=(0.6 * size, 0.3 * size, 0.2 * size))
        elif kind == "stack":
            for i, s in enumerate((0.8, 0.6, 0.4)):
                showcase.add_mesh(bpy.ops.mesh.primitive_cube_add, col, M["prop"], size=1,
                                  location=(x, y, 0.4 + i * 0.7), scale=(s / 2, s / 2, 0.35))
        elif kind == "arch":
            side = Vector((math.cos(a), math.sin(a), 0)) * 0.6
            for sgn in (-1, 1):
                showcase.add_mesh(bpy.ops.mesh.primitive_cylinder_add, col, M["prop"], radius=0.14, depth=size,
                                  location=(x + sgn * side.x, y + sgn * side.y, size / 2))
            showcase.add_mesh(bpy.ops.mesh.primitive_cube_add, col, M["prop"], size=1, location=(x, y, size + 0.1),
                              scale=(0.9, 0.18, 0.12), rotation=(0, 0, a))
    return col


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=408, help="408 = 48 hold + 312 orbit + 48 hold (17 s, 4 clips)")
    ap.add_argument("--hold", type=int, default=HOLD, help="front hold at each end, in frames (a wave in each)")
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=896)
    a = ap.parse_args(argv)
    if a.frames - 2 * a.hold < 4 * RAMP or a.hold < 8 + WAVE_LEN + 8:
        raise SystemExit("--frames must leave an orbit of at least 64 frames, and --hold must hold a wave (48+)")
    SET.update(frames=a.frames, hold=a.hold)
    showcase.build_set = build_set              # this template's set, camera and animation on the showcase parts
    showcase.build_camera = build_camera
    showcase.animate = animate
    showcase.build(a.frames, a.width, a.height, orbit=360.0)
    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    print("SAVED", a.out, {"frames": a.frames, "hold": a.hold, "waves": waves(a.frames, a.hold),
                           "top_speed_deg_per_frame": round(360.0 / (a.frames - 2 * a.hold - RAMP), 3)})


if __name__ == "__main__":
    main()
