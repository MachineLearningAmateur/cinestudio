"""Three-dancer template: three mannequins on the edge of the ring, a camera that goes once round them and stops
in front of each dancer, and a spotlight on the dancer in focus. For now each dancer waves in its spotlight;
dance motion (motion capture) comes later.

    blender -b --factory-startup --python templates/three_dancers/build.py -- --out shot.blend
        [--dwell 40] [--move-speed 1.5] [--radius 0.8] [--look 55] [--landmarks] [--night]
        [--dancers 1 --frames 360]     # one dancer in the centre, the camera goes once round it, front to front

Timeline: for each dancer (0, 120 and 240 degrees round the centre; each faces outwards): the camera stops in
front of it for `dwell` frames (a slow drift of a few degrees), then moves 120 degrees to the next at
`move-speed` degrees per frame (eased). The last move goes back to the start, so the shot is a full circle.
Built from code only (no external assets), like templates/character_showcase.
"""
from __future__ import annotations

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

DANCERS = (0.0, 120.0, 240.0)          # degrees round the centre; 0 = in front (toward -Y)
CAM_RADIUS, CAM_TARGET_Z, CAM_Z = 4.0, 0.92, 1.45
DRIFT = 8.0                            # degrees the camera drifts during a stop (a still camera looks frozen)


def timeline(dwell: int, move_speed: float):
    """[(start, end, kind, dancer)] and the total frame count. kind: "stop" or "move"."""
    move = max(8, round(120.0 / move_speed))
    out, f = [], 1
    for i in range(len(DANCERS)):
        out.append((f, f + dwell - 1, "stop", i))
        f += dwell
        out.append((f, f + move - 1, "move", i))
        f += move
    return out, f - 1


def solo_timeline(frames: int):
    """One dancer: one constant-speed orbit from the front back to the front (the last frame = the first angle)."""
    return [(1, frames, "orbit", 0)], frames


def cam_az(f, tl):
    """Camera azimuth (degrees) at frame f: in front of a dancer during its stop, eased moves between; or a
    constant-speed full orbit (solo) that starts and ENDS at the front."""
    if tl and tl[0][2] == "orbit":
        start, end = tl[0][0], tl[0][1]
        return 360.0 * (f - start) / max(1, end - start)
    for start, end, kind, i in tl:
        if start <= f <= end:
            t = (f - start) / max(1, end - start)
            base = DANCERS[i]
            if kind == "stop":
                return base - DRIFT / 2 + DRIFT * t
            e = showcase.smoothstep(t)
            return base + DRIFT / 2 + (120.0 - DRIFT) * e
    return DANCERS[0] + 360.0


def place(rig, deg, radius):
    a = math.radians(deg)
    rig.location = (radius * math.sin(a), -radius * math.cos(a), 0.0)
    rig.rotation_mode = "XYZ"
    rig.rotation_euler.z = a                     # faces outwards (its front points away from the centre)


def wave_phases(start, end, total):
    """The showcase wave keys, fitted between `start` and `end` (the dancer's stop)."""
    span = end - start
    rest = {b: showcase._rest_dir(b) for b in ("upper_arm.R", "forearm.R", "hand.R")}
    up = Vector((-0.82, -0.18, 0.52))
    r0, w0 = start, start + max(4, span // 5)
    w1 = end - max(8, span // 4)
    ph = [(1, rest, showcase.RELAXED), (r0, rest, showcase.RELAXED),
          (r0 + (w0 - r0) // 2, {"upper_arm.R": Vector((-0.60, -0.45, -0.10)), "forearm.R": Vector((-0.20, -0.60, 0.75)),
                                 "hand.R": Vector((-0.20, -0.55, 0.80))}, showcase.OPEN)]
    k = 0
    for f in range(w0, max(w0 + 1, w1) + 1, 6):
        sw = -0.42 if k % 2 == 0 else 0.42
        ph.append((f, {"upper_arm.R": up, "forearm.R": Vector((sw, -0.18, 0.89)), "hand.R": Vector((sw * 1.3, -0.20, 0.85))},
                   showcase.OPEN))
        k += 1
    last = ph[-1][0]
    ph += [(last + 11, {"upper_arm.R": Vector((-0.55, -0.15, -0.82)), "forearm.R": Vector((-0.35, -0.35, -0.87)),
                        "hand.R": Vector((-0.30, -0.30, -0.90))}, showcase.RELAXED),
           (last + 22, rest, showcase.RELAXED), (max(total, last + 23), rest, showcase.RELAXED)]
    return ph


def animate(rig, deg, phases, total, cam_pos, look_max=55.0):
    """Idle stand; the head turns toward the camera (at most `look_max` degrees; 0 = the head follows the body);
    the right hand waves in the stop."""
    keys = sorted(set(range(1, total + 1, 3)) | {p[0] for p in phases if p[0] <= total} | {total})
    for f in keys:
        local = rig.matrix_world.inverted() @ cam_pos(f)
        az = math.degrees(math.atan2(local.x, -local.y))       # the camera seen from the dancer (0 = in front)
        look = look_max * math.sin(math.radians(max(-90.0, min(90.0, az))))
        dirs, hand, palm_w = showcase._arm_at(phases, f)
        palms = {"hand.R": (Vector((-0.15, -1.0, 0.0)), palm_w)} if palm_w > 0 else {}
        mq.key_pose(rig, f, dirs, {"neck": look * 0.35, "head": look * 0.65}, palms,
                    {".L": showcase.RELAXED, ".R": hand})


def spotlights(tl, radius, total):
    """One spot per dancer, from above and in front; bright during its stop, dim otherwise (12-frame fades)."""
    for i, deg in enumerate(DANCERS):
        a = math.radians(deg)
        pos = Vector((math.sin(a) * (radius + 1.6), -math.cos(a) * (radius + 1.6), 3.6))
        aim = Vector((math.sin(a) * radius, -math.cos(a) * radius, 1.0))
        ld = bpy.data.lights.new(f"Spot{i + 1}", "SPOT")
        ld.spot_size, ld.spot_blend, ld.color = math.radians(32), 0.35, (1.0, 0.93, 0.80)
        lo = bpy.data.objects.new(f"Spot{i + 1}", ld)
        lo.location = pos
        lo.rotation_euler = (aim - pos).to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.collection.objects.link(lo)
        stop = next((s, e) for s, e, kind, j in tl if kind == "stop" and j == i)
        for f, energy in ((1, 60), (stop[0] - 12, 60), (stop[0], 1400), (stop[1], 1400), (stop[1] + 12, 60), (total, 60)):
            if 1 <= f <= total:
                ld.energy = energy
                ld.keyframe_insert("energy", frame=f)


def landmarks(radius=0.8, night=False):
    """Fixed things all round the set, so the picture shows that the CAMERA turns: a tiled floor, low stones round
    the stage (not where a dancer stands) and pillars of different heights near the back wall. With the plain set the
    control had few edges and the round stage looked the same from every side; LTX then kept the background still
    and turned the dancers round instead (2026-10-19)."""
    col = bpy.data.collections["Set"]
    prop, stripe = bpy.data.materials["cs_prop"], bpy.data.materials["cs_stripe"]
    # tiled floor: a checker in object space, 0.8 m tiles
    floor = next(o for o in col.objects if o.type == "MESH" and o.dimensions.x > 20)
    m = bpy.data.materials.new("cs_tiles")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = 0.8
    tc, ck = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeTexChecker")
    ck.inputs["Scale"].default_value = 1.25
    a, b = ((0.035, 0.04, 0.06), (0.07, 0.075, 0.10)) if night else ((0.09, 0.09, 0.10), (0.17, 0.17, 0.18))
    ck.inputs["Color1"].default_value, ck.inputs["Color2"].default_value = (*a, 1), (*b, 1)
    nt.links.new(tc.outputs["Object"], ck.inputs["Vector"])
    nt.links.new(ck.outputs["Color"], bsdf.inputs["Base Color"])
    floor.data.materials.clear()
    floor.data.materials.append(m)
    # low stones round the stage, between the dancers (a dancer stands at 0, 120 and 240 degrees)
    for k in range(12):
        deg = 15 + k * 30
        if radius > 0 and min(abs((deg - d + 180) % 360 - 180) for d in DANCERS) < 20:
            continue
        r = 1.3
        t = math.radians(deg)
        showcase.add_mesh(bpy.ops.mesh.primitive_cube_add, col, stripe if k % 2 else prop, size=1,
                          location=(math.sin(t) * r, -math.cos(t) * r, 0.06), rotation=(0, 0, t), scale=(0.28, 0.16, 0.12))
    # pillars near the back wall: 14, uneven heights, so no two views look alike
    for k in range(14):
        t = math.radians(k * 360 / 14 + 7)
        h = 1.4 + 1.6 * ((k * 7) % 5) / 4
        showcase.add_mesh(bpy.ops.mesh.primitive_cylinder_add, col, prop if k % 3 else stripe, radius=0.16, depth=h,
                          vertices=12, location=(math.sin(t) * 6.4, -math.cos(t) * 6.4, h / 2))


def solo_spotlight(total):
    """One spot from above and in front of the single dancer, on for the whole shot."""
    pos, aim = Vector((0.0, -2.2, 3.8)), Vector((0.0, 0.0, 1.0))
    ld = bpy.data.lights.new("Spot1", "SPOT")
    ld.spot_size, ld.spot_blend, ld.color, ld.energy = math.radians(34), 0.35, (1.0, 0.93, 0.80), 1400
    lo = bpy.data.objects.new("Spot1", ld)
    lo.location = pos
    lo.rotation_euler = (aim - pos).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(lo)


def build(dwell=40, move_speed=1.5, radius=0.8, width=1536, height=896, look=55.0, dancers=3, frames=360):
    global DANCERS
    solo = dancers == 1
    if solo:
        DANCERS = (0.0,)
        radius = 0.0
        look = 0.0     # a full orbit passes behind the dancer: a head that follows the camera would snap side to side
        tl, total = solo_timeline(frames)
    elif dancers == 3:
        tl, total = timeline(dwell, move_speed)
    else:
        raise SystemExit("--dancers must be 1 or 3")
    showcase.ORBIT[0] = 360.0                    # the set: props outside the camera circle (a full orbit)
    sc = showcase.reset()
    sc.frame_start, sc.frame_end = 1, total
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = width, height, 100
    M = {k: showcase.material(k, v) for k, v in showcase.COLORS.items()}
    showcase.build_set(M)
    target = bpy.data.objects.new("CamTarget", None)
    target.location = (0, 0, CAM_TARGET_Z)
    sc.collection.objects.link(target)

    def cam_pos(f):
        a = math.radians(cam_az(f, tl))
        return Vector((math.sin(a) * CAM_RADIUS, -math.cos(a) * CAM_RADIUS, CAM_Z))
    cam_data = bpy.data.cameras.new("ShotCam")
    cam_data.lens = 30
    cam = bpy.data.objects.new("ShotCam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    tr = cam.constraints.new("TRACK_TO")
    tr.target, tr.track_axis, tr.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
    for f in range(1, total + 1):
        cam.location = cam_pos(f)
        cam.keyframe_insert("location", frame=f)
    for i, deg in enumerate(DANCERS):
        suffix = "" if i == 0 else str(i + 1)
        rig, _ = mq.build(M, collection_name=f"Mannequin{suffix}", rig_name=f"MannequinRig{suffix}")
        place(rig, deg, radius)
        bpy.context.view_layer.update()
        if solo:     # the wave while the camera passes the front: about the first 70 degrees of the orbit
            stop = (5, max(40, round(total * 70 / 360)))
        else:
            stop = next((s, e) for s, e, kind, j in tl if kind == "stop" and j == i)
        animate(rig, deg, wave_phases(stop[0], stop[1], total), total, cam_pos, look)
    showcase.build_lights()
    for name in ("Key", "Fill", "Rim"):          # the room light lower: the spotlights lead
        bpy.data.lights[name].energy *= 0.45
    if solo:
        solo_spotlight(total)
    else:
        spotlights(tl, radius, total)
    sc.frame_set(1)
    return sc, tl, total


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dwell", type=int, default=40, help="frames the camera stops in front of each dancer")
    ap.add_argument("--move-speed", type=float, default=1.5, help="degrees per frame between dancers (eased)")
    ap.add_argument("--radius", type=float, default=0.8, help="the dancers' circle (the ring has 1.1 m)")
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=896)
    ap.add_argument("--look", type=float, default=55.0,
                    help="degrees each head turns toward the camera at most (0 = the head follows the body: a dancer "
                         "that turns away shows the back of its head)")
    ap.add_argument("--dancers", type=int, default=3, choices=(1, 3),
                    help="1 = one dancer in the centre and a constant-speed orbit from the front back to the front")
    ap.add_argument("--frames", type=int, default=360, help="--dancers 1: frames of the orbit (360 = 1 degree per frame)")
    ap.add_argument("--landmarks", action="store_true",
                    help="tiled floor, stones round the stage and pillars at the back wall: they show the camera's turn")
    ap.add_argument("--night", action="store_true", help="light the set like a moonlit night (showcase.night_set)")
    a = ap.parse_args(argv)
    sc, tl, total = build(a.dwell, a.move_speed, a.radius, a.width, a.height, a.look, a.dancers, a.frames)
    if a.night:
        showcase.night_set()
    if a.landmarks:
        landmarks(0.0 if a.dancers == 1 else a.radius, a.night)
    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    print("SAVED", a.out, "frames", total, "timeline", tl)


if __name__ == "__main__":
    main()
