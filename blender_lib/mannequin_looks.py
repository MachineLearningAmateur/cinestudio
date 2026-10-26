"""Looks for the cinestudio mannequin that make it easier for a standard pose detector (DWPose) to read
(run inside Blender). The rig and the body shape do not change; only parts and materials are added.

    import mannequin_looks
    mannequin_looks.apply(rig, collection, ["googly_eyes", "hair"])

Looks (any combination; see LOOKS):
  googly_eyes  white eyes with black pupils on the front of the head (front vs back of the head)
  face         dark eyes and a mouth on the front of the head
  mouth        a distinct mouth: wide, dark red, low on the face (with googly_eyes: a clear face from far away)
  hair         a dark cap on the back and the top of the head (a back view looks like a back view)
  hair_full    a fuller hairstyle: the cap, and hair at the back of the head down to the nape
  ears         small ears at the sides of the head (the ear points of OpenPose)
  clothes      skin, a shirt with long sleeves, trousers and shoes in different colours (a human figure)
  lr_tint      a warm tint on the left limbs and a cool tint on the right limbs

The figure faces -Y and its left side is +X (blender_lib/mannequin.py). Parts are made in the rest pose and
parented to their bones, so they follow every animation.
"""
from __future__ import annotations

import bpy
from mathutils import Vector

import mannequin

LOOKS = ("googly_eyes", "face", "mouth", "hair", "hair_full", "ears", "clothes", "lr_tint")

# the head ellipsoid of mannequin.build (rest space): centre and semi-axes
_HEAD = mannequin.REST["head"]
HEAD_C = (Vector(_HEAD[0]) + Vector(_HEAD[1])) / 2 + Vector((0, 0, 0.01))
HEAD_R = Vector((0.11, 0.121, (Vector(_HEAD[1]) - Vector(_HEAD[0])).length / 2 * 1.05))

COLORS = {"white": (0.92, 0.92, 0.90), "black": (0.01, 0.01, 0.01), "hair": (0.10, 0.06, 0.04),
          "skin": (0.80, 0.58, 0.45), "shirt": (0.12, 0.30, 0.65), "trousers": (0.20, 0.20, 0.23),
          "shoes": (0.04, 0.04, 0.04), "mouth": (0.35, 0.08, 0.08)}
# clothes: the first matching group of the bone (name without the side) gives the material
CLOTHES = (("skin", ("head", "neck", "hand", "thumb", "index", "middle", "ring", "pinky")),
           ("shirt", ("chest", "spine", "shoulder", "upper_arm", "forearm")),
           ("trousers", ("hips", "thigh", "shin")),
           ("shoes", ("foot", "toe")))


def _material(name, color, roughness=0.6):
    m = bpy.data.materials.get("csl_" + name)
    if m is None:
        m = bpy.data.materials.new("csl_" + name)
        m.use_nodes = True
        bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = (*color, 1)
        bsdf.inputs["Roughness"].default_value = roughness
    return m


def _color_of(mat):
    bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if mat and mat.use_nodes else None
    return tuple(bsdf.inputs["Base Color"].default_value[:3]) if bsdf else (0.7, 0.7, 0.7)


def head_surface(x, z, out=0.0):
    """The rest-space point on the front of the head at (x, z), moved `out` metres further to the front."""
    k = 1 - (x / HEAD_R.x) ** 2 - ((z - HEAD_C.z) / HEAD_R.z) ** 2
    y = -HEAD_R.y * max(k, 0.0) ** 0.5
    return Vector((x, HEAD_C.y + y - out, z))


class _Adder:
    """Adds parts in the rest pose and parents them to bones (keeps their world placement)."""

    def __init__(self, rig, col):
        self.rig, self.col = rig, col

    def __enter__(self):
        self.old = self.rig.data.pose_position
        self.rig.data.pose_position = "REST"
        bpy.context.view_layer.update()
        return self

    def __exit__(self, *exc):
        self.rig.data.pose_position = self.old
        bpy.context.view_layer.update()

    def add(self, op, mat, bone, location, scale=(1, 1, 1), **kw):
        ob = mannequin._part(op, self.col, mat, location=self.rig.matrix_world @ Vector(location), **kw)
        ob.name = f"csl_{bone}_{ob.name}"
        ob.scale = scale
        bpy.context.view_layer.update()
        mw = ob.matrix_world.copy()
        ob.parent, ob.parent_type, ob.parent_bone = self.rig, "BONE", bone
        ob.matrix_world = mw
        return ob


def googly_eyes(a):
    """White eyes with a dark outline (the head is white too) and large black pupils."""
    for x in (-0.042, 0.042):
        p = head_surface(x, HEAD_C.z + 0.025)
        a.add(bpy.ops.mesh.primitive_uv_sphere_add, _material("black", COLORS["black"], 0.3), "head",
              p + Vector((0, 0.006, 0)), scale=(0.036, 0.010, 0.036), segments=24, ring_count=12, radius=1)
        a.add(bpy.ops.mesh.primitive_uv_sphere_add, _material("white", COLORS["white"], 0.3), "head",
              p + Vector((0, 0.001, 0)), scale=(0.030, 0.012, 0.030), segments=24, ring_count=12, radius=1)
        a.add(bpy.ops.mesh.primitive_uv_sphere_add, _material("black", COLORS["black"], 0.3), "head",
              p + Vector((0, -0.011, -0.004)), scale=(0.017, 0.006, 0.017), segments=16, ring_count=8, radius=1)


def face(a):
    for x in (-0.040, 0.040):
        a.add(bpy.ops.mesh.primitive_uv_sphere_add, _material("black", COLORS["black"], 0.3), "head",
              head_surface(x, HEAD_C.z + 0.025, -0.003), scale=(0.016, 0.006, 0.020), segments=16, ring_count=8, radius=1)
    a.add(bpy.ops.mesh.primitive_uv_sphere_add, _material("mouth", COLORS["mouth"]), "head",
          head_surface(0.0, HEAD_C.z - 0.060, -0.002), scale=(0.030, 0.006, 0.008), segments=16, ring_count=8, radius=1)


def mouth(a):
    a.add(bpy.ops.mesh.primitive_uv_sphere_add, _material("mouth_dark", (0.25, 0.02, 0.03), 0.5), "head",
          head_surface(0.0, HEAD_C.z - 0.065, -0.002), scale=(0.045, 0.008, 0.014), segments=24, ring_count=10, radius=1)


def hair(a):
    a.add(bpy.ops.mesh.primitive_uv_sphere_add, _material("hair", COLORS["hair"], 0.8), "head",
          HEAD_C + Vector((0, 0.035, 0.030)), scale=tuple(HEAD_R * 1.06), segments=32, ring_count=16, radius=1)


def hair_full(a):
    """A fuller hairstyle: the cap, a mass at the back down to the nape and over the ears' tops."""
    mat = _material("hair", COLORS["hair"], 0.8)
    a.add(bpy.ops.mesh.primitive_uv_sphere_add, mat, "head", HEAD_C + Vector((0, 0.030, 0.030)),
          scale=tuple(HEAD_R * 1.08), segments=32, ring_count=16, radius=1)
    a.add(bpy.ops.mesh.primitive_uv_sphere_add, mat, "head", HEAD_C + Vector((0, 0.045, -0.055)),
          scale=(HEAD_R.x * 1.02, HEAD_R.y * 0.85, HEAD_R.z * 0.75), segments=32, ring_count=16, radius=1)


def ears(a, mat=None):
    for x in (-1, 1):
        a.add(bpy.ops.mesh.primitive_uv_sphere_add, mat or _material("ear", _color_of(bpy.data.materials.get("cs_mannequin"))),
              "head", HEAD_C + Vector((x * (HEAD_R.x + 0.006), 0.005, -0.005)), scale=(0.012, 0.022, 0.034),
              segments=16, ring_count=8, radius=1)


def _bone_base(name):
    return name.split(".")[0].split("_0")[0]


def clothes(rig, col):
    for ob in col.all_objects:
        if ob.type != "MESH" or not ob.parent_bone or ob.name.startswith("csl_"):
            continue
        base = _bone_base(ob.parent_bone)
        for kind, bones in CLOTHES:
            if base in bones:
                ob.data.materials[0] = _material(kind, COLORS[kind])
                break


def lr_tint(rig, col):
    tint = {".L": (1.15, 0.85, 0.75), ".R": (0.75, 0.90, 1.15)}
    for ob in col.all_objects:
        if ob.type != "MESH" or not ob.parent_bone or ob.parent_bone[-2:] not in tint or not ob.data.materials:
            continue
        old = ob.data.materials[0]
        k = tint[ob.parent_bone[-2:]]
        c = tuple(min(1.0, v * t) for v, t in zip(_color_of(old), k))
        ob.data.materials[0] = _material(f"{old.name}{ob.parent_bone[-2:]}", c)


def apply(rig, col, looks):
    """Apply the named looks (see LOOKS) to the mannequin `rig` / `col`. The order of the names does not matter."""
    looks = list(dict.fromkeys(looks))
    bad = [n for n in looks if n not in LOOKS]
    if bad:
        raise ValueError(f"unknown looks {bad}; known: {LOOKS}")
    if "clothes" in looks:          # first: the added parts below then keep their own colours
        clothes(rig, col)
    with _Adder(rig, col) as a:
        if "googly_eyes" in looks:
            googly_eyes(a)
        if "face" in looks:
            face(a)
        if "mouth" in looks:
            mouth(a)
        if "hair" in looks:
            hair(a)
        if "hair_full" in looks:
            hair_full(a)
        if "ears" in looks:
            ears(a, _material("skin", COLORS["skin"]) if "clothes" in looks else None)
    if "lr_tint" in looks:
        lr_tint(rig, col)
    return looks
