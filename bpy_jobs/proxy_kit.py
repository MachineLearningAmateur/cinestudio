"""Put a character's PROXY KIT on the shot's mannequin (Blender, background): simple clay parts that give the layout
render the character's outline, so Layout-To-Render keeps it (research 2026-10-22: the layout keeps "the placement of
objects"; a bare mannequin carries no mask, hair or costume, and the video drifted from the costume and turned the face
toward the camera in back views).

    blender -b SHOT.blend --python bpy_jobs/proxy_kit.py -- --spec SPEC.json --out OUT.blend [--rig MannequinRig]

SPEC (the character's "proxy" record), every part optional:
    {"mask": {"eyes": true, "shade": 0.92}, "hair": {"shade": 0.22, "length": "short"},
     "suit": {"color": [r, g, b]},  "shoes": {"color": [r, g, b]},
     "bands": [{"bone": "forearm.L", "from": 0.15, "to": 1.0, "color": [r, g, b], "scale": 1.3}, ...],
     "jacket": {"color": [r, g, b], "sleeve": 0.55, "bottom": 0.75},
     "cape": {"color": [r, g, b], "length": "hips|thigh|knee", "width": 1.25, "flare": 1.7, "wrap": 0.25}}
mask.ears = two pointed fox ears on top of the mask (a kitsune mask; ear_height / ear_width in head radii, default 0.95 /
0.30); mask.shape "round" = a round, wide mask without the pointed chin (kitsune), default the pointed oval (masked3d).
hair.locks "chin" | "shoulder" | "chest" = two flat strands hanging at the sides of the head down to that height;
hair.back (same values) = a full-width back of hair from the back of the head down to that height. cape = the back half of a tapered tube from the
shoulders down to `length`, on the chest bone (2026-10-23, user: "add a proxy kit for kitsune with mask and cape").
Colours are linear RGB 0..1 ("shade" = a grey). suit = the mannequin's own body parts, shoes = its feet and toes;
bands = tubes round a part of a limb bone (0 = the bone's head, 1 = its tail: bandages, cuffs); jacket = a shell round
the chest from the shoulders down to `bottom` (0 = the neck, 1 = the hips) with short sleeves (`sleeve` = the part of
the upper arm they cover); jacket.open (0..0.9) = an open front: a gap of that part of the half width shows the suit. Costume parts give Qwen (sheet views from the kitted mannequin) and the layout render the
costume's places and values (2026-10-22, user: "with the proxy kit i think the drawing can be more accurate").
The parts are parented to the rig's head bone (they follow every template's motion), sized from the mannequin's own head
mesh, in clay tones (light mask, dark hair: the value contrast shows front and back), in the "Mannequin" collection so
the passes' character mask includes them.
"""
import argparse
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Vector


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=None, help="the kit (JSON file) for --rig / --collection")
    ap.add_argument("--kits", default=None, help='JSON file: [{"spec": {...}, "rig": "MannequinRig2", "collection": '
                    '"Mannequin2"}, ...]: every character\'s kit in ONE Blender run (2026-10-25, N characters)')
    ap.add_argument("--out", required=True)
    ap.add_argument("--rig", default="MannequinRig")
    ap.add_argument("--collection", default="Mannequin")
    return ap.parse_args(argv)


def clay(name, shade):
    """A matte material: `shade` is a grey 0..1 or a linear [r, g, b]."""
    rgb = tuple(float(x) for x in shade) if isinstance(shade, (list, tuple)) else (float(shade),) * 3
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.8
    spec = float(os.environ.get("CINESTUDIO_KIT_SPEC", 0.0))      # NO specular shine (kit gauntlet 2026-10-26: the
    for key in ("Specular IOR Level", "Specular"):                # default 0.5 added a white shine that turned the
        if key in bsdf.inputs:                                    # dark plum hair grey-lavender in the layout; at 0
            bsdf.inputs[key].default_value = spec                 # every part renders in its own hue; head match
            break                                                 # 0.108 -> 0.222 for masked)
    m.diffuse_color = (*rgb, 1.0)                                 # Workbench previews
    return m


def paint(o, mat):
    """Every material slot of `o` (object- or mesh-linked) gets `mat`; a mesh without slots gets one."""
    if not o.material_slots:
        o.data.materials.append(mat)
    for slot in o.material_slots:
        slot.material = mat


def colour_of(part, default):
    return part.get("color", part.get("shade", default))


def bone_meshes(rig, coll, bone):
    return [o for o in coll.all_objects if o.type == "MESH" and o.parent == rig and o.parent_type == "BONE"
            and o.parent_bone == bone]


def bone_radius(rig, coll, bone):
    """The thickness of the mannequin's mesh on `bone` (rest pose): the largest distance of its vertices from the axis."""
    pb = rig.data.bones[bone]
    a, b = rig.matrix_world @ pb.head_local, rig.matrix_world @ pb.tail_local
    ax = (b - a).normalized()
    best = 0.0
    for o in bone_meshes(rig, coll, bone):
        for v in o.data.vertices:
            w = o.matrix_world @ v.co
            d = (w - a) - ax * (w - a).dot(ax)
            best = max(best, d.length)
    return best or 0.05


def tube(name, a, b, radius, mat, coll, rig, bone, sx=1.0, sy=1.0, segments=32):
    """A closed tube from world point a to b (rest pose), parented to `bone`."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments, radius1=radius, radius2=radius,
                          depth=(b - a).length)
    for v in bm.verts:
        v.co.x *= sx
        v.co.y *= sy
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.location = (a + b) / 2
    ob.rotation_euler = Vector((0, 0, 1)).rotation_difference((b - a).normalized()).to_euler()
    ob.data.materials.append(mat)
    for poly in ob.data.polygons:
        poly.use_smooth = True
    link(ob, coll, rig, bone)
    return ob


def cone(name, base, tip, radius, mat, coll, rig, sy=1.0, bone="head"):
    """A pointed cone from `base` to `tip` (world, rest pose), flattened in its local y by `sy`, on `bone`."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=True, segments=24, radius1=radius, radius2=0.0,
                          depth=(tip - base).length)
    for v in bm.verts:
        v.co.y *= sy
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.location = (base + tip) / 2
    ob.rotation_euler = Vector((0, 0, 1)).rotation_difference((tip - base).normalized()).to_euler()
    ob.data.materials.append(mat)
    link(ob, coll, rig, bone)
    return ob


def cape_shell(name, top, bottom, w_top, w_bottom, depth, mat, coll, rig, wrap=0.25, segments=48, bone="chest"):
    """The back of a tapered elliptic tube (the body faces -Y): the part with y > -wrap x depth stays, so the cape
    wraps a little round the sides; open at the front, the top and the bottom. Parented to the chest bone."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    h = (top - bottom).length
    bmesh.ops.create_cone(bm, cap_ends=False, segments=segments, radius1=1.0, radius2=1.0, depth=h)
    for v in bm.verts:                                            # z from -h/2 (bottom) to +h/2 (top)
        t = (v.co.z + h / 2) / h                                  # 0 at the bottom, 1 at the top
        w = w_bottom + (w_top - w_bottom) * t
        v.co.x *= w
        v.co.y *= depth * (1.0 + 0.35 * (1 - t))
    bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=Vector((0, -wrap * depth, 0)),
                           plane_no=Vector((0, 1, 0)), clear_inner=True)
    bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=0.012)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.location = (top + bottom) / 2
    ob.data.materials.append(mat)
    for p in ob.data.polygons:
        p.use_smooth = True
    link(ob, coll, rig, bone)
    return ob


def head_mesh(rig, coll):
    """The mannequin's head sphere: the largest mesh parented to the head bone."""
    best = None
    for o in coll.all_objects:
        if o.type == "MESH" and o.parent == rig and o.parent_type == "BONE" and o.parent_bone == "head":
            size = max(o.dimensions)
            if best is None or size > max(best.dimensions):
                best = o
    if best is None:
        raise SystemExit("proxy_kit: no head mesh on the rig's head bone")
    return best


def link(obj, coll, rig, bone="head"):
    world = obj.matrix_basis.copy()                               # its location/rotation (not linked yet: no parent)
    coll.objects.link(obj)
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone
    bpy.context.view_layer.update()
    obj.matrix_world = world                                      # keep the rest-pose place


def shell(name, centre, radii, cuts, mat, coll, rig, segments=64, rings=48, rot=(0.0, 0.0, 0.0)):
    """A UV-sphere shell (ellipsoid radii) cut by planes on the UNIT sphere: each cut (point, normal) keeps the side
    the normal points to (clean edges, no vertex steps)."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=1.0)
    for co, no in cuts:
        bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=Vector(co),
                               plane_no=Vector(no).normalized(), clear_inner=True)
    for v in bm.verts:
        v.co = Vector((v.co.x * radii[0], v.co.y * radii[1], v.co.z * radii[2]))
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.location = centre
    ob.rotation_euler = rot
    ob.data.materials.append(mat)
    for p in ob.data.polygons:
        p.use_smooth = True
    link(ob, coll, rig)
    return ob


# mask MARKS (2026-10-24, user: "can we update the proxy kits to be more distinctive? like have the actual marks on the
# masks?"): a plain white mask on a white mannequin head looks the same from the front and the back, and the duo videos
# turned the characters round at the end. Each mark is a flat sticker on the mask's front surface: x, z = the place on
# the mask (-1..1 of its half width and half height, 0 = the middle), w, h = half size (same units), roll = the turn in
# the face plane (degrees, + = counter-clockwise as seen from the front), color = linear RGB.
BLACK = [0.015, 0.01, 0.01]
# the presets live in bpy_jobs/mask_marks.json (the app's character.validate_proxy checks the same marks)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mask_marks.json"), encoding="utf-8") as _fh:
    MARK_PRESETS = {k: v for k, v in json.load(_fh).items() if not k.startswith("_")}


def decal(name, centre, radii, mark, mat, coll, rig, bone="head"):
    """A flat sticker on the mask ellipsoid (centre, radii; the face toward -Y) at the mark's place, size and roll."""
    x, z = float(mark.get("x", 0.0)), float(mark.get("z", 0.0))
    if x * x + z * z > 0.95:                                      # the app checks the footprint (character.validate_proxy)
        raise SystemExit(f"proxy_kit: mark {name} at x {x}, z {z} lies off the mask")
    s = math.sqrt(1.0 - x * x - z * z)
    p = Vector((x, -s, z))                                        # on the unit sphere
    e = Vector((p.x * radii[0], p.y * radii[1], p.z * radii[2]))  # on the ellipsoid
    n = Vector((p.x / radii[0], p.y / radii[1], p.z / radii[2])).normalized()
    thick = 0.08 * min(radii)
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=10, radius=1.0)
    for v in bm.verts:                                            # local x = across, y = the normal, z = up
        v.co = Vector((v.co.x * float(mark.get("w", 0.1)) * radii[0], v.co.y * thick,
                       v.co.z * float(mark.get("h", 0.05)) * radii[2]))
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    q = Vector((0.0, -1.0, 0.0)).rotation_difference(n)           # the sticker's back on the surface, facing out
    from mathutils import Quaternion
    # n points toward the viewer of the mask's front, so a positive angle about it is counter-clockwise for that viewer
    q = Quaternion(n, math.radians(float(mark.get("roll", 0.0)))) @ q
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = q
    ob.location = centre + e + n * (thick * 0.35)
    ob.data.materials.append(mat)
    for poly in ob.data.polygons:
        poly.use_smooth = True
    link(ob, coll, rig, bone)
    return ob


def main():
    a = args()
    if bool(a.spec) == bool(a.kits):
        raise SystemExit("proxy_kit: give --spec (one kit) or --kits (a list), not both")
    if a.spec:
        todo = [{"spec": json.loads(open(a.spec, encoding="utf-8").read()), "rig": a.rig, "collection": a.collection}]
    else:
        todo = json.loads(open(a.kits, encoding="utf-8").read())
    for kit in todo:                                              # each kit on its own mannequin
        made, info = build(kit["spec"], bpy.data.objects[kit["rig"]], bpy.data.collections[kit["collection"]])
        print("PROXY_KIT", {"rig": kit["rig"], "parts": [o.name for o in made], **info})
    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    print("PROXY_KIT_SAVED", {"out": a.out, "kits": len(todo)})


def build(spec: dict, rig, coll) -> tuple[list, dict]:
    """Put the kit `spec` on `rig` (its parts in `coll`, parented to its bones); nothing is saved. A character's own
    body (mannequin_body.apply) must be on the rig BEFORE: the cape ends at the scaled leg's knee. Reused by
    bpy_jobs/mannequin_preview.py (2026-10-24). -> (the parts, {head_centre, head_radii})."""
    sc = bpy.context.scene
    keep_frame = sc.frame_current
    # work in the REST pose: the parts are placed on the rest head and then follow the bone
    rig.data.pose_position = "REST"
    bpy.context.view_layer.update()
    head = head_mesh(rig, coll)
    bb = [head.matrix_world @ Vector(c) for c in head.bound_box]
    c = sum(bb, Vector()) / 8.0
    rx = (max(v.x for v in bb) - min(v.x for v in bb)) / 2
    ry = (max(v.y for v in bb) - min(v.y for v in bb)) / 2
    rz = (max(v.z for v in bb) - min(v.z for v in bb)) / 2
    made = []
    if "mask" in spec:                                            # the face points toward -Y on the mannequin
        m = spec["mask"] or {}
        mat = clay("proxy_mask", colour_of(m, 0.92))
        round_mask = m.get("shape") == "round"                    # a round, wide mask (kitsune) instead of the pointed
        mc = Vector((c.x, c.y - ry * 0.30, c.z + (rz * 0.06 if round_mask else -rz * 0.05)))   # oval (masked3d); the round
        mr = (rx * 1.04, ry * 0.95, rz * 1.14) if round_mask else (rx * 0.92, ry * 0.95, rz * 1.12)   # one covers the
        #                                                         head up to its top (user: "no hair should be exposed
        #                                                         from the top")
        cuts = [((0, -0.05, 0), (0, -1, 0))]                      # the front half
        if m.get("top") is not None:                              # the mask ends at the brow (head radii above the
            zt = (float(m["top"]) * rz - (mc.z - c.z)) / mr[2]     # head centre): hair shows above it from the front
            cuts.append(((0, 0, zt), (0, 0, -1)))
        made.append(shell("ProxyMask", mc, mr, cuts, mat, coll, rig))
        mk = made[-1]                                             # a pointed lower end: pull the bottom to a point
        for v in ([] if round_mask else mk.data.vertices):        # (not for a round mask: user, "looks too tall")
            if v.co.z < -0.45 * mr[2]:
                t = (-0.45 * mr[2] - v.co.z) / (0.55 * mr[2])
                v.co.x *= max(0.15, 1.0 - 0.85 * t)
        if m.get("ears"):                                         # a fox (kitsune) mask: two pointed ears on top,
            eh = float(m.get("ear_height", 0.95))                 # part of the mask (white, tilted outward);
            ew = float(m.get("ear_width", 0.30))                  # ear_height/ear_width in head radii (user, 2026-10-24:
            for side in (-1, 1):                                  # "the mask of the kitsune ... looks too tall")
                base = Vector((mc.x + side * rx * 0.52, mc.y - ry * 0.05, c.z + rz * 0.62))
                tip = base + Vector((side * rx * 0.30, 0.0, rz * eh))
                made.append(cone(f"ProxyEar{'L' if side > 0 else 'R'}", base, tip, rx * ew, mat, coll, rig, sy=0.45))
                if m.get("ear_inner"):                            # the inside of the ear (a kitsune's pink): a
                    inner = clay("proxy_ear_inner", m["ear_inner"])   # thinner cone ON the ear's front face: its
                    depth = rx * ew * 0.45                            # base sits forward by the ear's front depth and
                    b_in = base + Vector((0.0, -depth * 0.80, rz * 0.08))   # its tip at the ear's tip (the cone narrows
                    t_in = tip - (tip - base) * 0.12 + Vector((0.0, -depth * 0.10, 0.0))    # there: Codex review, a
                    made.append(cone(f"ProxyEarIn{'L' if side > 0 else 'R'}", b_in, t_in, rx * ew * 0.55, inner,   # fixed
                                     coll, rig, sy=0.25))                                          # offset floated)
        if m.get("eyes", True):
            dark = clay("proxy_eyes", 0.03)
            es = float(m.get("eye_size", 1.0))                    # larger eyes read better at video size
            for side in (-1, 1):                                  # almond eye holes, angled like snake eyes
                ex = mc.x + side * rx * 0.36
                shell(f"ProxyEye{'L' if side > 0 else 'R'}", Vector((ex, mc.y - mr[1] * 0.93, mc.z + rz * 0.12)),
                      (rx * 0.20 * es, ry * 0.08, rz * 0.10 * es), [], dark, coll, rig, 24, 12,
                      rot=(0.0, side * math.radians(-22), 0.0))
        marks = m.get("marks") or []
        if isinstance(marks, str):
            if marks not in MARK_PRESETS:
                raise SystemExit(f"proxy_kit: unknown mask marks {marks!r} (use {', '.join(MARK_PRESETS)} or a list)")
            marks = MARK_PRESETS[marks]
        mats = {}
        for i, mk_ in enumerate(marks):                           # the mask's own marks, as stickers on its front
            col = tuple(float(v) for v in mk_.get("color", BLACK))
            mats.setdefault(col, clay(f"proxy_mark_{len(mats)}", list(col)))
            made.append(decal(f"ProxyMark{i}", mc, mr, mk_, mats[col], coll, rig))
    if "hair" in spec:                                            # a short hair cap over the top and the back
        h = spec["hair"] or {}
        mat = clay("proxy_hair", colour_of(h, 0.22))
        low = {"short": -0.30, "medium": -0.60, "long": -0.95}.get(h.get("length", "short"), -0.30)
        front = float(h.get("front", 0.45))                       # the hairline at the forehead (lower = a fringe)
        made.append(shell("ProxyHair", Vector((c.x, c.y + ry * 0.06, c.z + rz * 0.06)), (rx * 1.10, ry * 1.08, rz * 1.06),
                          [((0, 0, front + (low - front) / 2), (0, -(low - front) / 2, 1))], mat, coll, rig))
        # one slanted hairline: at the forehead (front, y -1) `front` up, at the back (y +1) down to `low`
        if h.get("back"):                                         # a full back of hair down to that height (user:
            chest_top = rig.matrix_world @ rig.data.bones["chest"].tail_local      # "the back of the hair to be full
            to = {"chin": c.z - rz * 1.2, "shoulder": chest_top.z - rz * 0.25,      # length across rather than 2
                  "chest": chest_top.z - rz * 1.4}.get(h["back"], chest_top.z - rz * 0.25)  # ponytails")
            made.append(cape_shell("ProxyHairBack", Vector((c.x, c.y, c.z - rz * 0.1)), Vector((c.x, c.y, to)),
                                   rx * 1.06, rx * 1.12, ry * 1.04, mat, coll, rig, wrap=0.0, segments=32, bone="head"))
        if h.get("locks"):                                        # side locks (user, 2026-10-24: "the hair placement
            chest_top = rig.matrix_world @ rig.data.bones["chest"].tail_local      # of that model is a bit off"):
            to = {"chin": c.z - rz * 1.2, "shoulder": chest_top.z - rz * 0.25,      # kitsune's hair frames the mask
                  "chest": chest_top.z - rz * 1.4}.get(h["locks"], chest_top.z - rz * 0.25)   # down to the shoulders
            for side in (-1, 1):
                top = Vector((c.x + side * rx * 0.98, c.y + ry * 0.12, c.z + rz * 0.15))
                bot = Vector((c.x + side * rx * 1.02, c.y + ry * 0.22, to))
                made.append(tube(f"ProxyLock{'L' if side > 0 else 'R'}", bot, top, rx * 0.32, mat, coll, rig, "head",
                                 sx=0.55, sy=1.0, segments=16))
    if "suit" in spec:                                            # the mannequin's own parts in the suit's colour
        mat = clay("proxy_suit", colour_of(spec["suit"] or {}, 0.05))
        for o in coll.all_objects:
            if o.type == "MESH" and o.parent == rig and o.parent_type == "BONE" and not o.name.startswith("Proxy") \
                    and o.parent_bone not in ("head",):
                paint(o, mat)
    if "shoes" in spec:
        mat = clay("proxy_shoes", colour_of(spec["shoes"] or {}, 0.08))
        for side in (".L", ".R"):
            for bone in ("foot" + side, "toe" + side):
                if bone in rig.data.bones:
                    for o in bone_meshes(rig, coll, bone):
                        paint(o, mat)
    for i, band in enumerate(spec.get("bands") or []):           # bandages, cuffs: tubes round part of a limb
        bone = band["bone"]
        pb = rig.data.bones[bone]
        h, t = rig.matrix_world @ pb.head_local, rig.matrix_world @ pb.tail_local
        p0 = h + (t - h) * float(band.get("from", 0.0))
        p1 = h + (t - h) * float(band.get("to", 1.0))
        r = bone_radius(rig, coll, bone) * float(band.get("scale", 1.25))
        made.append(tube(f"ProxyBand{i}_{bone}", p0, p1, r, clay(f"proxy_band{i}", colour_of(band, 0.85)),
                         coll, rig, bone))
    if "jacket" in spec:                                          # a short jacket: torso shell + short sleeves
        j = spec["jacket"] or {}
        mat = clay("proxy_jacket", colour_of(j, [0.15, 0.02, 0.33]))
        chest = rig.data.bones["chest"]
        top = rig.matrix_world @ chest.tail_local                  # the neck end
        hips = rig.matrix_world @ rig.data.bones["hips"].head_local
        bottom = top + (hips - top) * float(j.get("bottom", 0.75))
        cm = bone_meshes(rig, coll, "chest") + bone_meshes(rig, coll, "spine")
        xs = [(o.matrix_world @ v.co).x for o in cm for v in o.data.vertices]
        ys = [(o.matrix_world @ v.co).y for o in cm for v in o.data.vertices]
        rw, rd = (max(xs) - min(xs)) / 2 if xs else 0.17, (max(ys) - min(ys)) / 2 if ys else 0.12
        made.append(tube("ProxyJacket", bottom, top - (top - bottom) * 0.08, 1.0, mat, coll, rig, "chest",
                         sx=rw * float(j.get("fit", 1.04)), sy=rd * float(j.get("fit", 1.04))))
        op = float(j.get("open", 0.0))                            # an OPEN jacket (user, 2026-10-26: the kit must
        if op > 0:                                                # look like the reference): the front faces within
            jk = made[-1]                                         # `open` x the half width and the caps go, so the
            bm = bmesh.new()                                      # suit shows down the middle from the front
            bm.from_mesh(jk.data)
            hw = rw * float(j.get("fit", 1.04))
            kill = [f for f in bm.faces if len(f.verts) > 4
                    or (f.calc_center_median().y < 0 and abs(f.calc_center_median().x) < op * hw)]
            bmesh.ops.delete(bm, geom=kill, context="FACES")
            bm.to_mesh(jk.data)
            bm.free()
        for side in (".L", ".R"):
            ua = rig.data.bones["upper_arm" + side]
            h, t = rig.matrix_world @ ua.head_local, rig.matrix_world @ ua.tail_local
            made.append(tube("ProxySleeve" + side, h, h + (t - h) * float(j.get("sleeve", 0.55)),
                             bone_radius(rig, coll, "upper_arm" + side) * float(j.get("sleeve_fit", 1.15)), mat, coll,
                             rig, "upper_arm" + side))
    if "cape" in spec:                                            # a cape: the BACK half of a tapered tube from the
        cp = spec["cape"] or {}                                   # shoulders down (hips / thigh / knee), on the chest
        mat = clay("proxy_cape", colour_of(cp, [0.02, 0.18, 0.15]))
        chest = rig.data.bones["chest"]
        top = rig.matrix_world @ chest.tail_local
        ends = {"hips": rig.matrix_world @ rig.data.bones["hips"].head_local}
        for side in (".L", ".R"):
            th = rig.data.bones.get("thigh" + side)
            if th is not None:
                h0, k = rig.matrix_world @ th.head_local, rig.matrix_world @ th.tail_local
                legs = rig.pose.bones["thigh" + side].scale[0]   # a character's own body (mannequin_body.py): the
                k = Vector((k.x, k.y, h0.z - legs * (h0.z - k.z)))   # knee of the scaled leg (the kit is built in REST)
                ends.setdefault("knee", Vector((top.x, top.y, k.z)))
                ends.setdefault("thigh", Vector((top.x, top.y, (k.z + ends["hips"].z) / 2)))
        length = cp.get("length", "thigh")
        bottom = ends.get(length) if isinstance(length, str) else top + (ends["hips"] - top) * float(length)
        bottom = Vector((top.x, top.y, bottom.z))
        cm = bone_meshes(rig, coll, "chest") + bone_meshes(rig, coll, "spine")
        xs = [(o.matrix_world @ v.co).x for o in cm for v in o.data.vertices]
        ys = [(o.matrix_world @ v.co).y for o in cm for v in o.data.vertices]
        rw, rd = (max(xs) - min(xs)) / 2 if xs else 0.17, (max(ys) - min(ys)) / 2 if ys else 0.12
        made.append(cape_shell("ProxyCape", top - Vector((0, 0, (top.z - bottom.z) * 0.04)), bottom,
                               rw * float(cp.get("width", 1.25)), rw * float(cp.get("flare", 1.7)), rd * 1.35, mat,
                               coll, rig, float(cp.get("wrap", 0.25))))
    rig.data.pose_position = "POSE"
    sc.frame_set(keep_frame)
    return made, {"head_centre": [round(x, 3) for x in c], "head_radii": [round(rx, 3), round(ry, 3), round(rz, 3)]}


if __name__ == "__main__":                                       # blender --python runs it as __main__; an import
    main()                                                        # (mannequin_preview.py) only gets build()
