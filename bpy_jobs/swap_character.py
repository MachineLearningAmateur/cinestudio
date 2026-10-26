"""Put your own rigged 3D character into a shot in place of the mannequin (Blender, background).

    blender -b SHOT.blend --python swap_character.py -- CHARACTER.blend OUT.blend [--rig NAME] [--frames A B]

The character's armature must use the cinestudio bone names for the main bones (hips, spine, chest, neck, head,
shoulder/upper_arm/forearm/hand .L/.R, thigh/shin/foot .L/.R). The shot's MannequinRig drives it: for every frame and
every shared bone, the mannequin bone's rotation away from ITS rest pose is applied to the character bone's own rest
pose (so a different rest pose - A-pose vs T-pose, other bone rolls - does not twist the limbs), and the hips follow
the mannequin's hips (scaled). The character is scaled to the mannequin's height. Keys are baked.

The mannequin's meshes move to the hidden collection "MannequinHidden" (excluded from the render); the character's
meshes go into the collection "Mannequin", so the passes (mask, people) use the character's real silhouette. The
MannequinRig stays (it gives the skeleton points).
"""
import sys

import bpy
from mathutils import Matrix, Vector

BONES = ["hips", "spine", "chest", "neck", "head",
         "shoulder.L", "upper_arm.L", "forearm.L", "hand.L", "shoulder.R", "upper_arm.R", "forearm.R", "hand.R",
         "thigh.L", "shin.L", "foot.L", "thigh.R", "shin.R", "foot.R"]


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    src, out = a[0], a[1]
    rig_name, frames = None, None
    if "--rig" in a:
        rig_name = a[a.index("--rig") + 1]
    if "--frames" in a:
        i = a.index("--frames")
        frames = (int(a[i + 1]), int(a[i + 2]))
    return src, out, rig_name, frames


def main():
    src, out, rig_name, frames = args()
    scene = bpy.context.scene
    man = bpy.data.objects["MannequinRig"]
    with bpy.data.libraries.load(src, link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    new = [o for o in data_to.objects if o is not None and o.type in ("MESH", "ARMATURE", "CURVE")]
    rigs = [o for o in new if o.type == "ARMATURE" and (rig_name is None or o.name == rig_name)]
    if not rigs:
        raise SystemExit("the character file has no armature")
    rig = rigs[0]
    meshes = [o for o in new if o.type != "ARMATURE"]
    # collections: the mannequin's meshes out of the render, the character in "Mannequin"
    mcoll = bpy.data.collections["Mannequin"]
    hidden = bpy.data.collections.get("MannequinHidden") or bpy.data.collections.new("MannequinHidden")
    if hidden.name not in scene.collection.children:
        scene.collection.children.link(hidden)
    for o in list(mcoll.objects):
        if o.type == "MESH":
            mcoll.objects.unlink(o)
            hidden.objects.link(o)
            o.hide_render = True
            o.hide_viewport = True
    hidden.hide_render = True
    for lc in bpy.context.view_layer.layer_collection.children:
        if lc.collection is hidden:
            lc.exclude = True
    for o in [rig, *meshes]:
        mcoll.objects.link(o)
    # place and scale: the character's rest height to the mannequin's
    bpy.context.view_layer.update()
    mh = (man.matrix_world @ man.data.bones["head"].tail_local).z - (man.matrix_world @ man.data.bones["foot.L"].tail_local).z
    ch = (rig.data.bones["head"].tail_local.z - rig.data.bones["foot.L"].tail_local.z)
    s = mh / ch if ch > 1e-6 else 1.0
    rig.matrix_world = Matrix.Translation(man.matrix_world.translation) @ Matrix.Diagonal((s, s, s, 1.0))
    bpy.context.view_layer.update()
    names = [b for b in BONES if b in rig.pose.bones and b in man.pose.bones]
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
    start, end = frames or (scene.frame_start, scene.frame_end)
    rest_m = {b: (man.matrix_world @ man.data.bones[b].matrix_local).to_3x3().normalized() for b in names}
    rest_c = {b: (rig.matrix_world @ rig.data.bones[b].matrix_local).to_3x3().normalized() for b in names}
    hips_rest_m = man.matrix_world @ man.data.bones["hips"].head_local
    hips_rest_c = rig.matrix_world @ rig.data.bones["hips"].head_local
    order = sorted(names, key=lambda b: len(rig.pose.bones[b].parent_recursive))
    for f in range(start, end + 1):
        scene.frame_set(f)
        for b in order:
            mpb = man.pose.bones[b]
            cur_m = (man.matrix_world @ mpb.matrix).to_3x3().normalized()
            delta = cur_m @ rest_m[b].inverted()                    # the mannequin bone's turn away from its rest
            want = delta @ rest_c[b]                                # the same turn on the character's own rest
            aim = cur_m.col[1].normalized()                         # then point the bone exactly where the
            have = want.col[1].normalized()                         # mannequin's bone points (rest poses differ:
            want = have.rotation_difference(aim).to_matrix() @ want  # A- vs T-pose); the twist stays from delta
            pb = rig.pose.bones[b]
            head_w = rig.matrix_world @ pb.matrix.translation
            if b == "hips":
                head_w = hips_rest_c + (man.matrix_world @ mpb.head - hips_rest_m)
            world = Matrix.LocRotScale(head_w, want.to_quaternion(), None)
            pb.matrix = rig.matrix_world.inverted() @ world @ Matrix.Diagonal((s, s, s, 1.0))
            bpy.context.view_layer.update()
            pb.keyframe_insert("rotation_quaternion", frame=f)
            if b == "hips":
                pb.keyframe_insert("location", frame=f)
    bpy.ops.wm.save_as_mainfile(filepath=out)
    print("SWAP_OK", rig.name, len(meshes), "meshes", "scale", round(s, 3), "bones", len(names), "frames", start, end)


main()
