"""Template from a reference dance: the mannequin copies a dancer's moves read from a video, with a STILL front camera.

    blender -b --factory-startup --python templates/flash_step/build.py -- --motion MOTION.json --out shot.blend
        [--audio CLIP.wav] [--camera-distance 4.6]

MOTION.json comes from setup/pose_from_video.py (DWPose of a front-facing dancer with a still camera -> per frame the
hip move and drop, bone directions, ankle positions on the mannequin's scale). Made 2026-10-22 for a "flash step"
(sideways dodge) template. Per frame: the hips move sideways and drop, spine/chest/head and both arms (upper arm,
forearm, hand) point along the read directions (set_bone_dir from the current direction, quaternion signs kept
continuous), the legs follow the ankles with the two-bone IK of templates/funk_dance (the knees bend forward), the hands
stay relaxed. The camera does not move (a still camera keeps the dodges sharp in the video and gives the run nothing to
misread); it looks at the figure from the front at full body.
`--audio` puts a sound file in as a strip at frame 1 (packed in the .blend). Do NOT ship a template whose audio is
someone else's recording in templates/library (the contest rule: no unlicensed IP): keep it in the user's library.
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "character_showcase"))
import build as showcase  # noqa: E402
from blender_lib import mannequin as mq  # noqa: E402

THIGH, SHIN = 0.43, 0.43                   # mannequin leg bone lengths (blender_lib/mannequin.py REST)
MOTION = {"frames": []}
CAM = {"distance": 4.0}


def leg_ik(hip, ankle, forward=Vector((0, -1, 0))):
    """Knee position for a two-bone leg from hip to ankle (the knee bends forward; templates/funk_dance/build.py)."""
    d = ankle - hip
    if d.length < 1e-6:                         # Codex: hip and ankle at one place gave a division by zero
        return hip + forward.normalized() * THIGH
    dist = min(d.length, THIGH + SHIN - 1e-4)
    u = d.normalized()
    cos_a = max(-1.0, min(1.0, (THIGH ** 2 + dist ** 2 - SHIN ** 2) / (2 * THIGH * dist)))
    w = (forward - u * forward.dot(u))
    w = w.normalized() if w.length > 1e-6 else Vector((0, -1, 0))
    return hip + THIGH * (cos_a * u + math.sqrt(max(0.0, 1 - cos_a ** 2)) * w)


def build_camera(frames, target):
    """A still camera in front of the figure (the showcase's height and lens, a little farther for the sideways moves)."""
    cam_data = bpy.data.cameras.new("ShotCam")
    cam_data.lens = 30
    cam = bpy.data.objects.new("ShotCam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis, track.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    cam.location = Vector((0.0, -CAM["distance"], sum(showcase.CAM_Z) / 2))
    cam.keyframe_insert("location", frame=1)

    def cam_dir_at(f):
        return (cam.location - Vector((0, 0, 1.65))).normalized()
    return cam, cam_dir_at


ANKLE_Z = 0.09
DECAY = 0.92                                   # per frame, while no foot stands
OPT = {"legs": "dirs", "pelvis": True, "kicks": True}


def _ankles(rig):
    return {s: rig.matrix_world @ rig.pose.bones["foot" + s].head for s in (".R", ".L")}


def animate(rig, frames, cam_dir_at):
    """Per frame: hips, spine/chest/head, arms and legs along the read directions; then the ROOT correction (round 1
    of the gauntlet): the hips move so a standing foot (contact in the reference) stays where it was set down (no
    slide), and so the lowest standing foot touches the floor. `--legs ik` = the old two-bone IK to the read ankles."""
    locked = {}
    pins, release = {}, {}
    offset = Vector((0.0, 0.0, 0.0))
    for f, p in enumerate(MOTION["frames"], 1):
        for pb in rig.pose.bones:
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.location = (0, 0, 0)
            pb.scale = (1, 1, 1)               # 2026-10-23 (user: "what happened to the hands.."): the scale was never
            #                                    reset; set_bone_dir's matrices carried the last frame's rounding into the
            #                                    next, it grew into a stretch, and the last frame's (unkeyed) scale held
            #                                    for the whole shot: hands as flat blades (6.7 x 0.07 x 2.6)
        hips = rig.pose.bones["hips"]
        to_local = hips.bone.matrix_local.to_3x3().inverted()
        delta = Vector((p["hips"][0], 0.0, p["hips"][1]))
        if p.get("hips_fwd"):                              # the hips forward along the body's facing (mocap.pose_frame:
            ya = math.radians(p.get("yaw") or 0.0)         # the b-boy lean back); the body faces -Y before its yaw
            delta += Vector((math.sin(ya), -math.cos(ya), 0.0)) * p["hips_fwd"]
        hips.location = to_local @ delta
        bpy.context.view_layer.update()
        if OPT["pelvis"] and p.get("pelvis_roll") is not None:   # round 4: the hip line tilts as in the video
            m = hips.matrix.copy()
            rot = Matrix.Rotation(math.radians(-p["pelvis_roll"]), 4, "Y")
            hips.matrix = Matrix.Translation(m.to_translation()) @ rot @ m.to_3x3().to_4x4()
            bpy.context.view_layer.update()
        if p.get("yaw"):                                   # the whole body turned about the vertical (mocap.pose_frame:
            m = hips.matrix.copy()                         # a finishing pose angled to one side)
            rot = Matrix.Rotation(math.radians(p["yaw"]), 4, "Z")
            hips.matrix = Matrix.Translation(m.to_translation()) @ rot @ m.to_3x3().to_4x4()
            bpy.context.view_layer.update()
        d = p["dirs"]
        for name in ("spine", "chest", "head"):            # parents first (Codex review: the collarbones were set
            mq.set_bone_dir(rig, name, Vector(d[name]))   # before the chest, which then carried them off)
        for name in ("shoulder.R", "shoulder.L"):
            if OPT["pelvis"] and name in d:
                mq.set_bone_dir(rig, name, Vector(d[name]))
        for name in ("upper_arm.R", "forearm.R", "hand.R", "upper_arm.L", "forearm.L", "hand.L"):
            mq.set_bone_dir(rig, name, Vector(d[name]))
        bpy.context.view_layer.update()
        if OPT["legs"] == "pin":
            pin_legs(rig, p, pins, release)
        for s in (() if OPT["legs"] == "pin" else (".R", ".L")):
            if OPT["legs"] == "ik" or "thigh" + s not in d:
                thigh = rig.pose.bones["thigh" + s]
                hip = rig.matrix_world @ thigh.head
                ankle = rig.matrix_world @ Vector(p["feet"][s])
                knee = leg_ik(hip, ankle)
                mq.set_bone_dir(rig, "thigh" + s, knee - hip)
                bpy.context.view_layer.update()
                mq.set_bone_dir(rig, "shin" + s, ankle - knee)
            else:
                mq.set_bone_dir(rig, "thigh" + s, Vector(d["thigh" + s]))
                bpy.context.view_layer.update()
                mq.set_bone_dir(rig, "shin" + s, Vector(d["shin" + s]))
            bpy.context.view_layer.update()
            rest_foot = foot_dir(s, p)
            mq.set_bone_dir(rig, "foot" + s, rest_foot)
            bpy.context.view_layer.update()
        if OPT["legs"] == "dirs":
            # a PERSISTENT root offset (round 2: a per-frame correction jumped back when a contact ended)
            hips.location = to_local @ (delta + offset)
            bpy.context.view_layer.update()
            contact = p.get("contact") or {}
            ank = _ankles(rig)
            planted = [s for s in (".R", ".L") if contact.get(s)]
            for s in (".R", ".L"):
                if s in planted and s not in locked:
                    locked[s] = Vector((ank[s].x, ank[s].y, 0.0))   # set down here (after the carried offset)
                elif s not in planted:
                    locked.pop(s, None)
            if planted:
                offset.x += sum(locked[s].x - ank[s].x for s in planted) / len(planted)
                offset.y += sum(locked[s].y - ank[s].y for s in planted) / len(planted)
                offset.z += ANKLE_Z - min(ank[s].z for s in planted)
            else:                                           # both feet move: ease back toward the read hip path
                offset.x *= DECAY
                offset.y *= DECAY
                offset.z *= DECAY
            hips.location = to_local @ (delta + offset)
            bpy.context.view_layer.update()
            low = min(a.z for a in _ankles(rig).values())
            if low < ANKLE_Z - 1e-4:                        # never below the floor
                offset.z += ANKLE_Z - low
                hips.location = to_local @ (delta + offset)
                bpy.context.view_layer.update()
        curl, spread = showcase.RELAXED[0], showcase.RELAXED[1]
        if p.get("grip"):                                  # the crossed arms' hands hold the other arm (mocap.pose_frame)
            curl, spread = curl + (0.7 - curl) * p["grip"], spread * (1 - p["grip"])
        mq.curl_fingers(rig, ".R", curl, spread, *showcase.RELAXED[2:])
        mq.curl_fingers(rig, ".L", curl, spread, *showcase.RELAXED[2:])
        for pb in rig.pose.bones:
            q = pb.rotation_quaternion.copy()
            key = (rig.name, pb.name)
            if key in mq._LAST_Q and q.dot(mq._LAST_Q[key]) < 0:
                q.negate()
                pb.rotation_quaternion = q
            mq._LAST_Q[key] = q
            pb.keyframe_insert("rotation_quaternion", frame=f)
        hips.keyframe_insert("location", frame=f)
    for pb in rig.pose.bones:                  # the scale is not keyed: leave every bone at 1 for the whole shot
        pb.scale = (1, 1, 1)


def foot_dir(s, p):
    """The foot's rest direction, turned with the body's yaw (the feet point where the body faces)."""
    v = Vector(mq.REST["foot" + s][1]) - Vector(mq.REST["foot" + s][0])
    return (Matrix.Rotation(math.radians(p["yaw"]), 3, "Z") @ v) if p.get("yaw") else v


RELEASE = 3                                    # frames a released foot blends from its pin to the read place
REACH = (0.43 + 0.43) * 0.97                   # a leg keeps a small knee bend
MAX_DROP = 0.15


def pin_legs(rig, p, pins, release):
    """Round 8 (critic 2 + Codex review): the free ankle place = the hip + the read thigh and shin directions at the
    mannequin's leg lengths (knees at mid-leg, back flicks from the depth pass); a STANDING foot (contact, with
    hysteresis) keeps the place where it was set down (on the floor); a released foot blends back over RELEASE frames;
    two-bone IK reaches the target with the knee bending toward the read knee. The root follows only the read hip path:
    no root shifts (they made 12 cm pops and could lift the supporting foot)."""
    d = p["dirs"]
    contact = p.get("contact") or {}
    if (p.get("settle") or 0) >= 0.999:                   # the end's REST hold (mocap.settle): the legs exactly along
        for s in (".R", ".L"):                            # the rest directions, no IK (it keeps a knee bend: REACH,
            for bone in ("thigh", "shin"):                # and with a near-straight leg its knee turned inward:
                mq.set_bone_dir(rig, bone + s, Vector(d[bone + s]))     # user, "why is the legs still slightly
                bpy.context.view_layer.update()                          # angled at rest position")
            rest_foot = foot_dir(s, p)
            mq.set_bone_dir(rig, "foot" + s, rest_foot)
            bpy.context.view_layer.update()
            LAST_ANKLE[s] = (rig.matrix_world @ rig.pose.bones["foot" + s].head).copy()
            pins.pop(s, None)
            release.pop(s, None)
        return
    # round 9 (Codex): a pinned foot out of reach -> the hips go DOWN by the missing distance (at most MAX_DROP)
    need = 0.0
    for s in (".R", ".L"):
        if contact.get(s) and s in pins:
            hip = rig.matrix_world @ rig.pose.bones["thigh" + s].head
            gap = (pins[s] - hip).length - REACH
            if gap > 0:
                dz = pins[s].z - hip.z                    # negative: the pin is below the hip
                horiz = (pins[s] - hip).to_2d().length
                # the hip must stand sqrt(REACH^2 - horiz^2) above the pin: the drop = (hip - pin) - that height
                drop = -dz - math.sqrt(max(0.0, REACH ** 2 - horiz ** 2))
                need = max(need, min(MAX_DROP, max(0.0, drop)))
    if need > 0:
        hips = rig.pose.bones["hips"]
        hips.location = hips.location + hips.bone.matrix_local.to_3x3().inverted() @ Vector((0, 0, -need))
        bpy.context.view_layer.update()
    for s in (".R", ".L"):
        thigh = rig.pose.bones["thigh" + s]
        hip = rig.matrix_world @ thigh.head
        t_dir, s_dir = Vector(d["thigh" + s]).normalized(), Vector(d["shin" + s]).normalized()
        free_knee = hip + THIGH * t_dir
        free = free_knee + SHIN * s_dir
        free.z = max(free.z, ANKLE_Z)
        if contact.get(s):
            if s not in pins:                             # round 12: the pin starts where the foot IS (no snap) ...
                start = LAST_ANKLE.get(s, free)
                pins[s] = Vector((start.x, start.y, max(ANKLE_Z, start.z)))
            else:                                         # ... and eases down to the floor
                pins[s].z = ANKLE_Z + (pins[s].z - ANKLE_Z) * 0.5
            target = pins[s].copy()
            release.pop(s, None)
        else:
            if s in pins:
                release[s] = [pins.pop(s), RELEASE]
            target = free
            if s in release:
                pin_at, left = release[s]
                k = left / (RELEASE + 1)
                target = pin_at * k + free * (1 - k)
                release[s][1] -= 1
                if release[s][1] <= 0:
                    release.pop(s)
        w = (p.get("kick") or {}).get(s, 0.0) if OPT["kicks"] else 0.0
        if w > 0 and not contact.get(s):                  # round 13: the kicking leg straightens along hip -> foot
            kd = (p.get("kick_dir") or {}).get(s)
            line = Vector(kd) if kd else (free - hip)       # the dancer's hip -> ankle line (round 13b)
            if line.length > 1e-6:
                straight = hip + line.normalized() * (THIGH + SHIN) * 0.995
                straight.z = max(straight.z, ANKLE_Z)
                target = target.lerp(straight, w)
        pole = (free_knee - hip)
        axis = (target - hip)
        if axis.length > 1e-6:
            u = axis.normalized()
            pole = pole - u * pole.dot(u)
        pole = pole.normalized() if pole.length > 1e-6 else Vector((0, -1, 0))
        knee = leg_ik(hip, target, forward=pole)
        mq.set_bone_dir(rig, "thigh" + s, knee - hip)
        bpy.context.view_layer.update()
        mq.set_bone_dir(rig, "shin" + s, target - knee)
        bpy.context.view_layer.update()
        rest_foot = foot_dir(s, p)
        mq.set_bone_dir(rig, "foot" + s, rest_foot)
        bpy.context.view_layer.update()
        LAST_ANKLE[s] = (rig.matrix_world @ rig.pose.bones["foot" + s].head).copy()


LAST_ANKLE = {}                                # the foot's real place on the frame before (round 12)


def build_set(M):
    """Only a plain floor (user, 2026-10-22: "keep it clean and let the panorama handle it"; a panorama run hides the
    template's set anyway, bpy_jobs/pano_world.py)."""
    col = showcase.collection("Set")
    showcase.add_mesh(bpy.ops.mesh.primitive_plane_add, col, M["floor"], size=30, location=(0, 0, 0))
    return col


def add_audio(path):
    sc = bpy.context.scene
    se = sc.sequence_editor or sc.sequence_editor_create()
    strips = getattr(se, "strips", None)            # an EMPTY collection is falsy: test for None (Blender 5: strips)
    if strips is None:
        strips = se.sequences
    strip = strips.new_sound("song", os.path.abspath(path), 1, sc.frame_start)
    strip.sound.pack()
    return strip


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--camera-distance", type=float, default=4.0)
    ap.add_argument("--legs", choices=("pin", "dirs", "ik"), default="pin", help="dirs (default): leg directions from the "
                    "video + the root keeps standing feet still; ik: two-bone IK to the read ankles (round 0)")
    ap.add_argument("--no-pelvis", action="store_true", help="no pelvis tilt and no collarbones (rounds 0-3)")
    ap.add_argument("--no-kicks", action="store_true", help="do not straighten the legs in the found kicks (round 13)")
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=896)
    a = ap.parse_args(argv)
    MOTION.update(json.loads(open(a.motion, encoding="utf-8").read()))
    CAM["distance"] = a.camera_distance
    OPT["legs"] = a.legs
    OPT["pelvis"] = not a.no_pelvis
    OPT["kicks"] = not a.no_kicks
    frames = len(MOTION["frames"])
    showcase.build_camera = build_camera
    showcase.animate = animate
    showcase.build_set = build_set
    sc = showcase.build(frames, a.width, a.height, orbit=0.0)
    sc.render.fps = int(round(MOTION.get("fps", 24)))
    if a.audio:
        add_audio(a.audio)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
    xs = [p["hips"][0] for p in MOTION["frames"]]
    print("SAVED", a.out, {"frames": frames, "hips_x": [round(min(xs), 3), round(max(xs), 3)],
                           "audio": bool(a.audio)})


if __name__ == "__main__":
    main()
