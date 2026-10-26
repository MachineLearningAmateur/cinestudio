"""The cinestudio mannequin: a jointed wooden-style figure with a full human rig (run inside Blender).

51 bones with the cinestudio bone names, so bpy_jobs/render_passes.py can export body keypoints
(OpenPose COCO-18) and hand keypoints (OpenPose hand-21) from any shot that uses it:
  torso     hips, spine, chest, neck, head
  per side  shoulder, upper_arm, forearm, hand, thumb_01..03, index_01..03, middle_01..03,
            ring_01..03, pinky_01..03, thigh, shin, foot, toe            (suffix .L / .R)
The figure stands in an A-pose and faces -Y; its right side is -X. Units: metres.

Every finger bone has its roll set so that its local Z axis is the palm normal: a positive rotation
about local X curls the finger toward the palm (see `curl_fingers`).
"""

from __future__ import annotations

import math

import bpy
from mathutils import Matrix, Quaternion, Vector

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# finger: (base offset from the hand bone tail in (out, forward, down) hand space, direction, 3 phalanx lengths)
# out = away from the body (+X for .L), forward = -Y, down = -Z
_FINGER_SPEC = {
    "thumb": ((-0.012, 0.030, 0.075), (-0.25, 0.55, -0.80), (0.034, 0.028, 0.024)),
    "index": ((0.0, 0.026, 0.0), (0.0, 0.10, -1.0), (0.040, 0.025, 0.020)),
    "middle": ((0.0, 0.008, 0.0), (0.0, 0.0, -1.0), (0.045, 0.028, 0.022)),
    "ring": ((0.0, -0.010, 0.0), (0.0, -0.06, -1.0), (0.042, 0.026, 0.020)),
    "pinky": ((0.0, -0.026, 0.006), (0.0, -0.14, -1.0), (0.032, 0.020, 0.018)),
}
FINGER_RADIUS = {"thumb": 0.0105, "index": 0.0090, "middle": 0.0092, "ring": 0.0088, "pinky": 0.0078}


def _rest() -> tuple[dict, dict]:
    """REST {bone: (head, tail)} in parent-before-child order, and PARENT {bone: parent}."""
    rest = {
        "hips": ((0, 0, 0.95), (0, 0, 1.08)),
        "spine": ((0, 0, 1.08), (0, 0, 1.26)),
        "chest": ((0, 0, 1.26), (0, 0, 1.45)),
        "neck": ((0, 0, 1.45), (0, 0, 1.55)),
        "head": ((0, 0, 1.55), (0, 0, 1.78)),
    }
    parent = {"spine": "hips", "chest": "spine", "neck": "chest", "head": "neck"}
    for s, x in ((".L", 1), (".R", -1)):
        rest["shoulder" + s] = ((0.03 * x, -0.01, 1.42), (0.19 * x, 0, 1.42))
        rest["upper_arm" + s] = ((0.19 * x, 0, 1.42), (0.25 * x, 0, 1.15))
        rest["forearm" + s] = ((0.25 * x, 0, 1.15), (0.27 * x, -0.02, 0.90))
        rest["hand" + s] = ((0.27 * x, -0.02, 0.90), (0.28 * x, -0.03, 0.80))
        parent.update({"shoulder" + s: "chest", "upper_arm" + s: "shoulder" + s, "forearm" + s: "upper_arm" + s,
                       "hand" + s: "forearm" + s})
        hand_tail = Vector(rest["hand" + s][1])
        for f, ((o, fwd, down), d, lengths) in _FINGER_SPEC.items():
            head = hand_tail + Vector((o * x, -fwd, down))
            direction = Vector((d[0] * x, -d[1], d[2])).normalized()
            prev = "hand" + s
            for i, length in enumerate(lengths, 1):
                name = f"{f}_{i:02d}{s}"
                tail = head + direction * length
                rest[name] = (tuple(head), tuple(tail))
                parent[name] = prev
                prev, head = name, tail
        rest["thigh" + s] = ((0.10 * x, 0, 0.95), (0.11 * x, 0, 0.52))
        rest["shin" + s] = ((0.11 * x, 0, 0.52), (0.11 * x, 0.01, 0.09))
        rest["foot" + s] = ((0.11 * x, 0.01, 0.09), (0.12 * x, -0.10, 0.03))
        rest["toe" + s] = ((0.12 * x, -0.10, 0.03), (0.12 * x, -0.16, 0.025))
        parent.update({"thigh" + s: "hips", "shin" + s: "thigh" + s, "foot" + s: "shin" + s, "toe" + s: "foot" + s})
    return rest, parent


REST, PARENT = _rest()
ORDER = list(REST)                     # parents come before children
RADIUS = {"hips": 0.15, "spine": 0.12, "chest": 0.17, "neck": 0.05, "head": 0.11, "upper_arm": 0.05,
          "forearm": 0.042, "thigh": 0.075, "shin": 0.055, "foot": 0.045, "toe": 0.035}


def side_of(name: str) -> int:
    return 1 if name.endswith(".L") else -1


def palm_normal(name: str) -> Vector:
    """Rest palm normal of a hand or finger bone: toward the body (arms hang down, palms inward)."""
    return Vector((-side_of(name), 0, 0))


def _part(op, col, mat, **kw):
    op(**kw)
    ob = bpy.context.active_object
    for c in ob.users_collection:
        c.objects.unlink(ob)
    col.objects.link(ob)
    ob.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return ob


def build(materials: dict, collection_name: str = "Mannequin", rig_name: str = "MannequinRig"):
    """Build the rig and its rigid parts. materials: {"mannequin", "joint", "stripe"} -> bpy materials."""
    col = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(col)
    arm = bpy.data.armatures.new(rig_name)
    rig = bpy.data.objects.new(rig_name, arm)
    col.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for name, (h, t) in REST.items():
        eb = arm.edit_bones.new(name)
        eb.head, eb.tail = h, t
        if name.startswith(("hand",) + FINGERS):
            eb.align_roll(palm_normal(name))
    for name, parent in PARENT.items():
        arm.edit_bones[name].parent = arm.edit_bones[parent]
        arm.edit_bones[name].use_connect = (Vector(REST[name][0]) - Vector(REST[parent][1])).length < 1e-4
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
    M = materials
    for name, (h, t) in REST.items():
        h, t = Vector(h), Vector(t)
        base = name.split(".")[0]
        mid, length, d = (h + t) / 2, (t - h).length, (t - h).normalized()
        if base == "head":
            ob = _part(bpy.ops.mesh.primitive_uv_sphere_add, col, M["mannequin"], radius=1, location=mid + Vector((0, 0, 0.01)))
            ob.scale = (0.11, 0.121, length / 2 * 1.05)
            nose = _part(bpy.ops.mesh.primitive_cone_add, col, M["mannequin"], radius1=0.022, depth=0.05,
                         location=mid + Vector((0, -0.123, -0.01)), rotation=(math.radians(90), 0, 0))
            parts = [ob, nose]
        elif base in ("chest", "hips", "spine"):
            r = RADIUS[base]
            ob = _part(bpy.ops.mesh.primitive_uv_sphere_add, col, M["mannequin"], radius=1, location=mid)
            ob.scale = (r * 1.15, r * 0.75, length / 2 * 1.25)
            parts = [ob]
            if base == "chest":   # a dark stripe on the front: shows which way the body faces
                s = _part(bpy.ops.mesh.primitive_cube_add, col, M["stripe"], size=1, location=mid + Vector((0, -r * 0.72, 0)))
                s.scale = (0.05, 0.02, length * 0.8)
                parts.append(s)
        elif base == "shoulder":
            parts = []        # inside the chest
        elif base == "hand":  # a flat palm: thin along the palm normal
            ob = _part(bpy.ops.mesh.primitive_uv_sphere_add, col, M["mannequin"], radius=1, location=mid)
            ob.rotation_mode = "QUATERNION"
            ob.rotation_quaternion = rig.data.bones[name].matrix_local.to_quaternion()
            ob.scale = (0.040, length / 2 * 1.15, 0.016)     # local X = across the palm, Y = along, Z = palm normal
            joint = _part(bpy.ops.mesh.primitive_uv_sphere_add, col, M["joint"], radius=0.03, location=h)
            parts = [ob, joint]
        else:
            r = FINGER_RADIUS[base.split("_")[0]] if base.split("_")[0] in FINGERS else RADIUS[base]
            ob = _part(bpy.ops.mesh.primitive_cylinder_add, col, M["mannequin"], radius=r, depth=length * 0.9,
                       location=mid, vertices=16 if r < 0.02 else 32)
            ob.rotation_mode = "QUATERNION"
            ob.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d)
            joint = _part(bpy.ops.mesh.primitive_uv_sphere_add, col, M["joint"], radius=r * 1.12, location=h,
                          segments=12 if r < 0.02 else 32, ring_count=8 if r < 0.02 else 16)
            parts = [ob, joint]
            if base.endswith("_03"):       # round fingertip
                parts.append(_part(bpy.ops.mesh.primitive_uv_sphere_add, col, M["mannequin"], radius=r * 0.95,
                                   location=t - d * length * 0.05, segments=12, ring_count=8))
        bpy.context.view_layer.update()
        for p in parts:
            mw = p.matrix_world.copy()
            p.parent, p.parent_type, p.parent_bone = rig, "BONE", name
            p.matrix_world = mw
    return rig, col


# ---------------------------------------------------------------- posing (armature space)
def set_bone_dir(rig, name, direction):
    """Pose `name` so it points along `direction`, keeping its head where the parent puts it.

    The turn starts from the bone's current direction (where its posed parent carries it), not from its
    rest direction: a forearm that rests pointing down and must point up would otherwise need a turn of
    almost 180 degrees, where the axis of the shortest turn is unstable and the bone twists suddenly."""
    pb = rig.pose.bones[name]
    m = pb.matrix.copy()                       # the parent's pose with this bone's own rotation at rest
    cur_dir = m.col[1].xyz.normalized()
    swing = cur_dir.rotation_difference(Vector(direction).normalized())
    head = m.to_translation()
    pb.matrix = Matrix.Translation(head) @ swing.to_matrix().to_4x4() @ m.to_3x3().to_4x4()
    bpy.context.view_layer.update()


def yaw_bone(rig, name, degrees):
    """Turn bone `name` (and its children) about the vertical axis through its head."""
    pb = rig.pose.bones[name]
    m = pb.matrix.copy()
    head = m.to_translation()
    pb.matrix = Matrix.Translation(head) @ Matrix.Rotation(math.radians(degrees), 4, "Z") @ Matrix.Translation(-head) @ m
    bpy.context.view_layer.update()


_LAST_TWIST: dict = {}       # (rig, hand) -> the palm twist of the previous key (radians)
_LAST_Q: dict = {}           # (rig, bone) -> the quaternion of the previous key


def face_palm(rig, hand, target, weight=1.0):
    """Twist `hand` about its own axis so its palm (local Z) faces `target` as far as the bone direction allows."""
    pb = rig.pose.bones[hand]
    m = pb.matrix.copy()
    y = m.col[1].xyz.normalized()
    z = m.col[2].xyz.normalized()
    t = Vector(target)
    t = (t - y * t.dot(y))
    if t.length < 1e-6:
        return
    t.normalize()
    angle = math.atan2(y.dot(z.cross(t)), z.dot(t))
    # continuous from key to key (no jump from +179 to -179 degrees) and inside a natural wrist range
    key = (rig.name, hand)
    if key in _LAST_TWIST:
        last = _LAST_TWIST[key]
        angle += 2 * math.pi * round((last - angle) / (2 * math.pi))
    angle = max(-math.radians(110), min(math.radians(110), angle))
    _LAST_TWIST[key] = angle
    angle *= max(0.0, min(1.0, weight))          # blended in and out, never switched on in one frame
    head = m.to_translation()
    pb.matrix = Matrix.Translation(head) @ Matrix.Rotation(angle, 4, y) @ Matrix.Translation(-head) @ m
    bpy.context.view_layer.update()


def curl_fingers(rig, side, curl=0.0, spread=0.0, thumb=None):
    """curl: 0 = straight, 1 = fist (degrees per joint scale with the phalanx); spread: degrees between fingers.
    Rotations are local (relative to the hand), so call this after the hand is posed."""
    thumb = curl if thumb is None else thumb
    joint_max = {1: 70, 2: 95, 3: 70}
    fan = {"index": -1.0, "middle": 0.0, "ring": 1.0, "pinky": 2.0, "thumb": -1.5}
    for f in FINGERS:
        amount = thumb if f == "thumb" else curl
        for i in (1, 2, 3):
            pb = rig.pose.bones[f"{f}_{i:02d}{side}"]
            q = Quaternion((1, 0, 0), math.radians(joint_max[i] * amount * (0.6 if f == "thumb" else 1.0)))
            if i == 1 and spread:
                q = Quaternion((0, 0, 1), math.radians(spread * fan[f] * side_of(side))) @ q
            pb.rotation_quaternion = q
    bpy.context.view_layer.update()


def key_pose(rig, frame, dirs=None, yaws=None, palms=None, hands=None):
    """Key a full pose. dirs {bone: direction}; yaws {bone: degrees}; palms {hand bone: target normal};
    hands {".L"/".R": (curl, spread)}. Bones not named stay at rest."""
    dirs, yaws, palms, hands = dirs or {}, yaws or {}, palms or {}, hands or {}
    for pb in rig.pose.bones:
        pb.rotation_quaternion = (1, 0, 0, 0)
    bpy.context.view_layer.update()
    for name in ORDER:
        if name in dirs:
            set_bone_dir(rig, name, dirs[name])
        if name in yaws:
            yaw_bone(rig, name, yaws[name])
        if name in palms:
            target = palms[name]
            face_palm(rig, name, *(target if isinstance(target, tuple) else (target,)))
    for side, (curl, spread) in hands.items():
        curl_fingers(rig, side, curl, spread)
    for pb in rig.pose.bones:
        # q and -q are the same rotation: key the one nearer to the previous key, so Blender never turns a
        # bone the long way round between two keys (that showed as a one-frame twitch)
        q = pb.rotation_quaternion.copy()
        key = (rig.name, pb.name)
        if key in _LAST_Q and q.dot(_LAST_Q[key]) < 0:
            q.negate()
            pb.rotation_quaternion = q
        _LAST_Q[key] = q
        pb.keyframe_insert("rotation_quaternion", frame=frame)
