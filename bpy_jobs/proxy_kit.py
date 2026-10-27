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
0.30); mask.shape "round" = a round, wide mask without the pointed chin (kitsune), default the pointed oval (masked3d);
mask.width / mask.length / mask.depth (0.5..1.6, default 1) = the mask's width, height and how far it stands out;
head {width, depth, length (0.6..1.4), neck: colour} = the mannequin's own head mesh scaled (the mask and hair follow
it) and the neck painted (a dark collar);
mask.eye_height (-0.5..0.6, default 0.12 head radii above the mask centre) = where the eye holes sit;
mask.chin_back (0..0.8, default 0) = a pointed chin curves back toward the face (0 = a forward blade from the side).
hair.locks "chin" | "shoulder" | "chest" = two flat strands hanging at the sides of the head down to that height;
hair.cover (true) = the mannequin's own head in the hair colour (only the mask stays light);
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
from mathutils import Matrix, Vector


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


def _interp(levels, vals, v):
    """Linear interpolation of a per-level value at height v (levels go from top to bottom)."""
    if v >= levels[0]:
        return vals[0]
    for i in range(1, len(levels)):
        if v >= levels[i]:
            t = (v - levels[i]) / ((levels[i - 1] - levels[i]) or 1e-9)
            return vals[i] + (vals[i - 1] - vals[i]) * t
    return vals[-1]


def _inside(poly, x, y):
    """Point in polygon (ray casting)."""
    hit = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            hit = not hit
        j = i
    return hit


def _loft(name, rows, c, rx, ry, rz, segments, power, ridge=0.0):
    """A closed loft of superellipse rings: rows = [(v, half width u, front w, back w[, x offset u]), ...] top to
    bottom, in head units around the head centre `c`; `ridge` pushes the front centre line forward (a crease down a
    mask: its two halves catch the light differently). -> a bmesh (the caller assigns materials, builds the object)."""
    bm = bmesh.new()
    ring_verts = []
    for row in rows:
        v, au, f, b = row[:4]
        xo = row[4] if len(row) > 4 else 0.0
        au = max(au, 0.02)
        wc, wd = (f + b) / 2, max((b - f) / 2, 0.02)
        row = []
        for k in range(segments):
            t = 2 * math.pi * k / segments
            ct, st = math.cos(t), math.sin(t)
            x = au * math.copysign(abs(ct) ** (2 / power), ct)
            w = wc + wd * math.copysign(abs(st) ** (2 / power), st)
            if ridge and st < 0:
                w -= ridge * max(0.0, 1.0 - abs(x) / au) ** 1.5 * (-st)
            row.append(bm.verts.new((c.x + (x + xo) * rx, c.y + w * ry, c.z + v * rz)))
        ring_verts.append(row)
    for i in range(len(rows) - 1):
        for k in range(segments):
            k2 = (k + 1) % segments
            bm.faces.new((ring_verts[i][k], ring_verts[i][k2], ring_verts[i + 1][k2], ring_verts[i + 1][k]))
    for rv, row, top in ((ring_verts[0], rows[0], True), (ring_verts[-1], rows[-1], False)):
        v, _au, f, b = row[:4]
        xo = row[4] if len(row) > 4 else 0.0
        cap = bm.verts.new((c.x + xo * rx, c.y + (f + b) / 2 * ry, c.z + v * rz))
        for k in range(segments):
            k2 = (k + 1) % segments
            bm.faces.new((rv[k2], rv[k], cap) if top else (rv[k], rv[k2], cap))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def _obj_from(name, bm, mats, coll, rig):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    for m in mats:
        ob.data.materials.append(m)
    for poly in ob.data.polygons:
        poly.use_smooth = True
    link(ob, coll, rig, "head")
    return ob


def _cap_rows(row, cap, n=14):
    """Rows above `row` (v, half width, front, back[, x offset]) that close it with a round dome `cap` head units
    high (kitsune head gauntlet 2026-10-26: a fit that starts at the dome between the ears had a flat top)."""
    v, au, f, b = row[:4]
    rest = tuple(row[4:])
    wc, wd = (f + b) / 2, (b - f) / 2
    out = []
    for k in range(n, 0, -1):
        t = k / n
        q = math.sqrt(max(1.0 - t * t, 0.0))
        out.append((v + cap * t, max(au * q, 0.02), wc - wd * q, wc + wd * q) + rest)
    return out


def fitted_head(fit, c, rx, ry, rz, mats, coll, rig, rings=160, segments=192, power=2.4):
    """A head FITTED to the character's own views (head gauntlet 2026-10-26, user: "making the headshape part of the
    proxy kit"), in head units of the mannequin's head (u = x/rx, v = (z-cz)/rz, w = (y-cy)/ry; the face looks to -w):
    - the HAIR: a loft of superellipse rings of the measured outline (front half width `a`, side front/back `yf`, `yb`);
      at the mask's levels its front starts just in front of the mask's back edge (the mask lies over it);
    - the MASK: its own loft at the mask's levels: the measured front-view half width `hw`, from the side outline's
      front edge back to the mask's back edge `back` (so it wraps the cheeks as in the side view); its faces inside
      the mirrored eye outline are the eye holes.
    `mats` = (hair, mask, eye). -> [hair object, mask object]."""
    lv = [float(x) for x in fit["v"]]
    a, yf, yb = ([float(x) for x in fit[k]] for k in ("a", "yf", "yb"))
    mk = fit.get("mask") or {}
    mhw = [float(x) for x in mk.get("hw", [0.0] * len(lv))]
    mback = [float(x) for x in mk.get("back", [-9.0] * len(lv))]
    v0, v1 = float(mk.get("v0", -9)), float(mk.get("v1", -9))
    eye = [(float(u), float(v)) for u, v in (fit.get("eye") or [])]
    axis = float(fit.get("eye_axis", 0.0))                    # the eyes mirror about the mask's own centre line
    es = float(fit.get("eye_scale", 1.0))                     # larger eye holes read at video size (~37 px head)
    shade = float(fit["shade"]) if fit.get("shade") is not None else None
    shade_side = fit.get("shade_side", "left")                # the darker half, by PICTURE side in the front view
    if eye and es != 1.0:
        eu, ev = sum(p[0] for p in eye) / len(eye), sum(p[1] for p in eye) / len(eye)
        eye = [(eu + (u - eu) * es, ev + (v - ev) * es) for u, v in eye]
    vs = [lv[0] + (lv[-1] - lv[0]) * i / (rings - 1) for i in range(rings)]
    cover = float(fit.get("side_cover", 0.0))                 # how far the hair comes forward over the mask's sides
    hv = float(fit.get("hair_volume", 1.0))                   # LOOSER hair (user 2026-10-26: "a bit too tight around
    hair_rows = []                                            # the head"): wider, deeper at the back, a higher crown
    he = fit.get("hair_end")                                  # the hair ends this far down the mask (0 top .. 1 chin)
    v_hair_end = v0 - float(he) * (v0 - v1) if he is not None else None
    for v in vs:
        if v_hair_end is not None and v < v_hair_end:
            break
        if fit.get("helmet_v") is not None and v > float(fit["helmet_v"]) + 0.08:   # a helmet-top mask: the hair
            continue                                              # starts just above its band (none inside the mask)
        f, b = _interp(lv, yf, v), _interp(lv, yb, v)
        in_mask = v1 <= v <= v0 and _interp(lv, mhw, v) > 0.05
        if in_mask:
            f = max(f, min(_interp(lv, mback, v) - 0.12, b - 0.2))   # the hair starts under the mask's back edge
            if cover > 0:                                      # the hair hangs OVER the mask's sides (user
                f_mask = _interp(lv, yf, v) - 0.015 - float(fit.get("proud", 0.0))   # 2026-10-26: "the hair covers a
                # strongest at the mask's TOP, fading out toward the chin (user: "higher up on the mask rather than
                # at the bottom"): full cover down to `cover_full`, none below `cover_end` (fractions of the mask's
                # height from its top)
                hgt = max(v0 - v1, 1e-6)
                t = (v0 - v) / hgt
                c_full, c_end = float(fit.get("cover_full", 0.3)), float(fit.get("cover_end", 0.6))
                wgt = 1.0 if t <= c_full else max(0.0, 1.0 - (t - c_full) / max(c_end - c_full, 1e-6))
                cov = cover * (wgt * wgt * (3 - 2 * wgt))              # a smooth fade
                f = max(min(f, _interp(lv, mback, v) - cov), f_mask + 0.12)    # the face's centre stays free
                aw = max(_interp(lv, a, v) * hv, (_interp(lv, mhw, v) + 0.12 * wgt))
        elif hv != 1.0:
            f -= (hv - 1.0) * abs(f)                          # the fringe and the crown come forward too
        b += (hv - 1.0) * abs(b)
        if fit.get("helmet_v") is not None and v >= float(fit["helmet_v"]):   # under a helmet-top mask the hair
            b = min(b, _interp(lv, yb, v) - 0.06)                 # stays inside it (it starts below the band)
        width = aw if (in_mask and cover > 0) else _interp(lv, a, v) * hv
        if fit.get("helmet_v") is not None and v >= float(fit["helmet_v"]):
            width = min(width, _interp(lv, mhw, v) * 0.97) if _interp(lv, mhw, v) > 0.05 else width
        hair_rows.append((v * (1.0 + (hv - 1.0) * 0.8) if v > 0 else v, width, f, b))
    hb = fit.get("hair_below") or []                               # the hair below the mask, down to the shoulders
    if hb and v_hair_end is None:                                  # (measured rows [v, a, f, b]: the loft goes on)
        hair_rows += [(float(r_[0]), float(r_[1]) * hv, float(r_[2]), float(r_[3]) + (hv - 1.0) * abs(float(r_[3])))
                      for r_ in hb if float(r_[0]) < hair_rows[-1][0]]
    drop = fit.get("hair_drop")                                    # a BOB below the last hair row: down `drop` head
    if drop and hair_rows:                                         # units, narrowing a little, its front behind the
        v_, a_, f_, b_ = hair_rows[-1][:4]                          # neck, its back drawn in (kitsune critics: "about
        n_ = 10                                                    # as wide as the head, ends at the shoulders")
        for k in range(1, n_ + 1):
            t = k / n_
            hair_rows.append((v_ - float(drop) * t, a_ * (1.0 - float(fit.get("hair_taper", 0.12)) * t),
                              f_ + (max(f_, -0.05) - f_) * min(1.0, 2 * t), b_ - 0.25 * t * abs(b_)))
    cap = float(fit.get("cap", 0.0))
    if cap > 0 and hair_rows:
        hair_rows = _cap_rows(hair_rows[0], cap) + hair_rows
    out = [_obj_from("ProxyHead", _loft("ProxyHead", hair_rows, c, rx, ry, rz, segments, power), mats[:1], coll, rig)]
    if fit.get("under") and len(mats) > 3:                    # a BLACK under-layer (user 2026-10-26: "a black layer for
        top_u = (v_hair_end if v_hair_end is not None else v1) + 0.35   # the part not covered by the hair"): the suit's
        rows_u = []                                           # hood/collar from inside the hair down into the neck,
        for i in range(rings // 2):                           # behind the mask, so where the hair ends black shows
            v = top_u - (top_u - (v1 - 0.35)) * i / (rings // 2 - 1)
            f_m = _interp(lv, yf, v) - 0.015 - float(fit.get("proud", 0.0))
            in_m = v1 <= v <= v0 and _interp(lv, mhw, v) > 0.05
            f = max(_interp(lv, mback, v) - 0.08, f_m + 0.18) if in_m else _interp(lv, yf, min(v, v1)) + 0.25
            b = _interp(lv, yb, v) * 0.9
            au = max(_interp(lv, a, v) * 0.9, (_interp(lv, mhw, v) + 0.06) if in_m else 0.0)
            if v < v1:                                        # below the chin: narrow down to the neck
                k = min((v1 - v) / 0.35, 1.0)
                au *= 1.0 - 0.45 * k
                b = f + (b - f) * (1.0 - 0.4 * k)
            rows_u.append((v, au, f, max(b, f + 0.1)))
        out.append(_obj_from("ProxyUnder", _loft("ProxyUnder", rows_u, c, rx, ry, rz, segments, power), mats[3:4],
                             coll, rig))
    mrows = []
    mv = [v for v in vs if v1 <= v <= v0 and _interp(lv, mhw, v) > 0.03]
    if len(mv) >= 3:
        mv = [mv[0] + (mv[-1] - mv[0]) * i / (rings - 1) for i in range(rings)]
        ml, mr_ = mk.get("left"), mk.get("right")
        if fit.get("round_top"):                                   # a broad rounded forehead edge (under a fringe),
            ve = sum(p[1] for p in eye) / len(eye) if eye else (mv[0] + mv[-1]) / 2   # not a spike: above the eyes
            he = _interp(lv, mhw, ve)                              # the width follows an arc up to the mask's top
            top = mv[0]
            def _round(v, h):
                if v <= ve or top <= ve:
                    return h
                t = min((v - ve) / (top - ve), 1.0)
                return max(h, he * math.sqrt(max(1.0 - t * t, 0.0)))
        else:
            _round = lambda v, h: h                                # noqa: E731
        asym = bool(fit.get("asymmetric", True)) and ml and mr_
        proud = float(fit.get("proud", 0.0))                     # the mask stands out in front of the hair
        helmet = fit.get("helmet_v")                              # the mask covers the whole top of the head (a
        sn = fit.get("snout") or {}                                # the SNOUT pushed forward around the nose (a
        for v in mv:                                               # kitsune mask: white over the top and the back
            f = _interp(lv, yf, v) - 0.015 - proud                 # down to its band; hair below)
            if sn:
                d_ = (v - float(sn.get("v", -0.4))) / max(float(sn.get("span", 0.3)), 1e-6)
                f -= float(sn.get("amount", 0.1)) * max(0.0, 1.0 - d_ * d_) ** 2
            b = max(_interp(lv, mback, v) + 0.12, f + 0.1)          # deep under the hair: a clean side edge
            if helmet is not None and v >= float(helmet):
                b = max(b, _interp(lv, yb, v) + 0.03)
            if asym:                                               # the visible mask's own edges (the fringe's line)
                L, R = _interp(lv, [float(x) for x in ml], v), _interp(lv, [float(x) for x in mr_], v)
                mrows.append((v, max((L + R) / 2, 0.02), f, b, (R - L) / 2))
            else:
                mrows.append((v, _round(v, _interp(lv, mhw, v)), f, b))
        cc = float(fit.get("chin_cap", 0.0))                      # a ROUND chin under the snout (not a flat cut)
        if cc > 0 and mrows:
            last = mrows[-1]
            mrows = mrows + [(r_[0] - 2 * (r_[0] - last[0]),) + tuple(r_[1:]) for r_ in reversed(_cap_rows(last, cc))]
        if cap > 0 and mrows:                                      # the mask's own round top (under the ears)
            mrows = _cap_rows(mrows[0], cap * (1.04 if helmet is not None else 0.9)) + mrows
        drop = float(fit.get("chin_drop", 0.0))                   # the chin continues down to a point below the hair
        if drop > 0 and mrows:
            v_end, h_end, f_end, b_end = mrows[-1][:4]
            for i in range(1, 13):
                t = i / 12
                mrows.append((v_end - drop * t, max(h_end * (1 - t) ** 0.75, 0.02), f_end - 0.04 * t,
                              max(f_end - 0.04 * t + (b_end - f_end) * (1 - t) ** 1.5, f_end - 0.04 * t + 0.04)))
        bm = _loft("ProxyMask", mrows, c, rx, ry, rz, segments * 2, 2.0, float(fit.get("ridge", 0.0)))
        mmats = [mats[1], mats[2]]                                 # mask, eyes (+ the shaded half, + paint colours)
        if shade is not None:
            col = tuple(mats[1].diffuse_color[:3])               # clay() keeps the base colour here too
            mmats.append(clay("proxy_mask_shade", [x * shade for x in col]))
        paint = []                                                 # PAINT (kitsune head gauntlet 2026-10-26): the
        for pr in fit.get("paint") or []:                          # mask's own marks as coloured regions of its front
            col = tuple(round(float(x), 4) for x in pr["color"])   # faces: polygons in (u, v) measured on the front
            key = f"proxy_paint_{len(mmats)}"                      # view (red marks, black slits / nose / mouth); the
            idx = next((i for i, m in enumerate(mmats) if tuple(round(x, 4) for x in m.diffuse_color[:3]) == col
                        and m.name.startswith("proxy_paint")), None)   # later regions win (black slits over red)
            if idx is None:
                mmats.append(clay(key, list(col)))
                idx = len(mmats) - 1
            paint.append(([(float(u), float(v)) for u, v in pr["poly"]], idx))
        paint_side = []                                            # the SIDE marks, measured on the right view in
        for pr in fit.get("paint_side") or []:                     # (w, v); painted on the faces that look sideways,
            col = tuple(round(float(x), 4) for x in pr["color"])   # mirrored onto the left side
            idx = next((i for i, m in enumerate(mmats) if tuple(round(x, 4) for x in m.diffuse_color[:3]) == col
                        and m.name.startswith("proxy_paint")), None)
            if idx is None:
                mmats.append(clay(f"proxy_paint_{len(mmats)}", list(col)))
                idx = len(mmats) - 1
            paint_side.append(([(float(w_), float(v)) for w_, v in pr["poly"]], idx))
        for fc in bm.faces:
            p = fc.calc_center_median()
            u, v, w = (p.x - c.x) / rx, (p.z - c.z) / rz, (p.y - c.y) / ry
            front = fc.normal.y < -0.25                          # faces that look toward the front (the mask wraps)
            if eye and front and (_inside(eye, u, v) or _inside(eye, 2 * axis - u, v)):
                fc.material_index = 1
            elif shade is not None and (u < axis) == (shade_side == "left"):   # one half in grey (the reference's
                fc.material_index = 2                              # lit / shaded halves at the ridge; picture side)
            else:
                fc.material_index = 0
            nx, ny = fc.normal.x, fc.normal.y
            if paint_side and abs(nx) > 0.35 and abs(nx) >= abs(ny):
                hit = next((i for poly, i in reversed(paint_side) if _inside(poly, w, v)), None)
                if hit is not None:
                    fc.material_index = hit
            elif paint and ny < -0.1:
                hit = next((i for poly, i in reversed(paint) if _inside(poly, u, v)), None)
                if hit is not None:
                    fc.material_index = hit
        out.append(_obj_from("ProxyMask", bm, mmats, coll, rig))
    band = fit.get("band")                                         # the BAND (the mask's tie): a ribbon around the
    if band and band.get("color"):                                 # BACK of the head at band.v (the front half, in
        bv_ = float(band.get("v", fit.get("helmet_v", 0.2)))        # front of the mask's back edge, is cut away)
        bh = float(band.get("height", 0.06)) / 2
        rows_b = []
        for vv in (bv_ + bh, bv_ - bh):
            f_ = _interp(lv, yf, vv)
            b_ = max(_interp(lv, yb, vv), _interp(lv, mback, vv) + 0.12) + float(band.get("out", 0.04))
            rows_b.append((vv, max(_interp(lv, a, vv), _interp(lv, mhw, vv)) + float(band.get("out", 0.04)), f_, b_))
        bmb = _loft("ProxyBand", rows_b, c, rx, ry, rz, segments, power)
        cut = _interp(lv, mback, bv_) + float(band.get("from_back", 0.0))
        bmesh.ops.delete(bmb, geom=[fc for fc in bmb.faces if (fc.calc_center_median().y - c.y) / ry < cut],
                         context="FACES")
        out.append(_obj_from("ProxyBand", bmb, [clay("proxy_band", band["color"])], coll, rig))
    ears = fit.get("ears") or {}                                   # EARS (a fox mask): flattened cones on the mask's
    for i, e in enumerate(ears.get("list") or []):                 # top, from the measured base (down to the lowest
        bu = float(e["base"][0])                                   # row of the ear's red inside) to the tip; the
        bv = float(e["base"][1]) - float(ears.get("sink", 0.12))   # inside in its own colour (the red continuing
        #                                                           down the forehead is the mask's paint)
        tu, tv = (float(x) for x in e["tip"])
        dome_v0 = float(e["base"][1])                              # ears.height: the tip this much higher above the
        tv = dome_v0 + (tv - dome_v0) * float(ears.get("height", 1.0))   # dome; ears.lean: less outward lean (0..1)
        tu = bu + (tu - bu) * float(ears.get("lean", 1.0))
        vv = min(max(bv, lv[-1]), lv[0])
        wf = _interp(lv, yf, vv)
        wb = _interp(lv, mback, vv) if v1 <= vv <= v0 and _interp(lv, mback, vv) > -9 else _interp(lv, yb, vv)
        w = wf + float(ears.get("depth", 0.35)) * (wb - wf)        # the ear stands this far back on the mask's top
        dome_v = float(e["base"][1])                               # the measured width is at the dome line: the
        frac = min(max((dome_v - bv) / max(tv - bv, 1e-6), 0.0), 0.8)   # cone is wider at its sunk base
        r = float(e.get("half_width", 0.3)) * float(ears.get("width", 1.0)) * rx / (1.0 - frac)
        base = Vector((c.x + bu * rx, c.y + w * ry, c.z + bv * rz))
        tip = Vector((c.x + tu * rx, c.y + (w + float(ears.get("tip_back", 0.05))) * ry, c.z + tv * rz))
        side = "L" if bu > 0 else "R"
        out.append(cone(f"ProxyEar{side}", base, tip, r, mats[1], coll, rig, sy=float(ears.get("flat", 0.45))))
        if ears.get("inner"):
            inner = clay("proxy_ear_inner", ears["inner"])
            depth = r * float(ears.get("flat", 0.45))
            b_in = base + Vector((0.0, -depth * 0.80, rz * 0.06))
            t_in = tip - (tip - base) * 0.10 + Vector((0.0, -depth * 0.10, 0.0))
            out.append(cone(f"ProxyEarIn{side}", b_in, t_in, r * float(ears.get("inner_width", 0.62)), inner, coll,
                            rig, sy=0.25))
    return out

def head_asset(asset, c, rx, ry, rz, coll, rig):
    """A HEAD ASSET (kitsune head gauntlet 2026-10-26, user: "construct the head seperately and then throw it onto the
    head"): every mesh object of the .blend `asset["file"]` (made in HEAD UNITS: the mannequin head's centre at the
    origin, its radius 1 on each axis, the face toward -Y) is appended, scaled by the head's radii (rx, ry, rz) about
    the head centre `c` and parented to the head bone. Its own materials come along. -> the objects."""
    path = asset["file"] if isinstance(asset, dict) else asset
    if not os.path.isfile(path):
        raise SystemExit(f"proxy_kit: the head asset {path!r} is not a file")
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.objects = [n for n in src.objects]
    k = float(asset.get("scale", 1.0)) if isinstance(asset, dict) else 1.0
    S = Matrix.Diagonal((rx * k, ry * k, rz * k, 1.0))
    T = Matrix.Translation(c)
    out = []
    for ob in dst.objects:
        if ob is None or ob.type != "MESH":
            continue
        ob.matrix_basis = T @ S @ ob.matrix_basis
        ob.name = "Proxy" + ob.name if not ob.name.startswith("Proxy") else ob.name
        link(ob, coll, rig, "head")
        out.append(ob)
    return out


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
    if "head" in spec:                                            # the HEAD SHAPE (kit gauntlet 2026-10-26, user:
        hd = spec["head"] or {}                                   # "hotswap the headshape"): the mannequin's ball head
        sx, sy, sz = (float(hd.get(k, 1.0)) for k in ("width", "depth", "length"))   # scaled about its centre (the
        if (sx, sy, sz) != (1.0, 1.0, 1.0):                       # mesh only: the bones and the motion stay), its own
            head.data = head.data.copy()                          # mesh copy (a copied mannequin shares the data)
            mw = head.matrix_world
            inv = mw.inverted()
            hb = [mw @ Vector(v) for v in head.bound_box]
            hc = sum(hb, Vector()) / 8.0
            for v in head.data.vertices:
                w = mw @ v.co
                v.co = inv @ Vector((hc.x + (w.x - hc.x) * sx, hc.y + (w.y - hc.y) * sy, hc.z + (w.z - hc.z) * sz))
            head.data.update()
        if hd.get("neck") is not None:                            # the neck in a colour (a dark collar under a white
            nm = clay("proxy_neck", hd["neck"])                   # mask: a white neck merged with the mask's chin)
            for o in bone_meshes(rig, coll, "neck"):
                paint(o, nm)
    bb = [head.matrix_world @ Vector(c) for c in head.bound_box]
    c = sum(bb, Vector()) / 8.0
    rx = (max(v.x for v in bb) - min(v.x for v in bb)) / 2
    ry = (max(v.y for v in bb) - min(v.y for v in bb)) / 2
    rz = (max(v.z for v in bb) - min(v.z for v in bb)) / 2
    made = []
    asset = (spec.get("head") or {}).get("asset")
    if asset:                                                     # the kit brings a whole modelled head (its mask,
        made += head_asset(asset, c, rx, ry, rz, coll, rig)       # ears, hair): the ball head is hidden and the
        for o in bone_meshes(rig, coll, "head"):                  # mask and hair parts are not built
            if not o.name.startswith("Proxy"):
                o.hide_render = True
        spec = {k: v for k, v in spec.items() if k not in ("mask", "hair")}
    fit = None if asset else (spec.get("head") or {}).get("fit")
    if fit:                                                       # the kit brings its OWN head (fitted to the views):
        hair_c = colour_of(spec.get("hair") or {}, 0.22)          # the ball head is hidden, the mask and hair shells
        mats = (clay("proxy_hair", hair_c), clay("proxy_mask", colour_of(spec.get("mask") or {}, 0.92)),   # are not
                clay("proxy_eyes", 0.0),                          # built (the fitted head carries both); the under-
                clay("proxy_under", (spec.get("head") or {}).get("neck") or 0.02))   # layer takes the neck colour
        made += fitted_head(fit, c, rx, ry, rz, mats, coll, rig)
        for o in bone_meshes(rig, coll, "head"):                  # the ball head and its nose cone are hidden
            if not o.name.startswith("Proxy"):
                o.hide_render = True
        spec = {k: v for k, v in spec.items() if k != "mask"}
        if "hair" in spec:                                        # the hair's back and side locks stay (below the
            spec["hair"] = dict(spec["hair"] or {}, _fit=True)    # fitted head: kitsune's shoulder-length hair)
    if "mask" in spec:                                            # the face points toward -Y on the mannequin
        m = spec["mask"] or {}
        mat = clay("proxy_mask", colour_of(m, 0.92))
        round_mask = m.get("shape") == "round"                    # a round, wide mask (kitsune) instead of the pointed
        mc = Vector((c.x, c.y - ry * 0.30, c.z + (rz * 0.06 if round_mask else -rz * 0.05)))   # oval (masked3d); the round
        mr = (rx * 1.04, ry * 0.95, rz * 1.14) if round_mask else (rx * 0.92, ry * 0.95, rz * 1.12)   # one covers the
        mr = (mr[0] * float(m.get("width", 1.0)), mr[1] * float(m.get("depth", 1.0)),           # kit gauntlet
              mr[2] * float(m.get("length", 1.0)))                                               # 2026-10-26
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
                v.co.y *= max(0.2, 1.0 - float(m.get("chin_back", 0.0)) * t)   # the chin curves back to the face
                #                                                 (user 2026-10-26: "from the side view it shouldnt be
                #                                                 that sharp": x alone made a forward blade)
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
            dark = clay("proxy_eyes", 0.0)                        # pure black (0.03 rendered mid-grey: gauntlet)
            es = float(m.get("eye_size", 1.0))                    # larger eyes read better at video size
            for side in (-1, 1):                                  # almond eye holes, angled like snake eyes
                ex = mc.x + side * rx * 0.36
                ez = mc.z + rz * float(m.get("eye_height", 0.12))   # eyes inside the mask, below a cut top
                shell(f"ProxyEye{'L' if side > 0 else 'R'}", Vector((ex, mc.y - mr[1] * 0.93, ez)),
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
        if not h.get("_fit"):                                     # (a fitted head carries the cap itself)
            made.append(shell("ProxyHair", Vector((c.x, c.y + ry * 0.06, c.z + rz * 0.06)),
                              (rx * 1.10, ry * 1.08, rz * 1.06),
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
                made.append(tube(f"ProxyLock{'L' if side > 0 else 'R'}", bot, top,
                                 rx * 0.32 * float(h.get("lock_width", 1.0)), mat, coll, rig, "head",
                                 sx=0.55, sy=1.0, segments=16))
        if h.get("cover") and not h.get("_fit"):                  # the mannequin's own head in the hair colour (kit
            for o in bone_meshes(rig, coll, "head"):              # gauntlet 2026-10-26: the white head showed beside
                if not o.name.startswith("Proxy"):                # and behind the hair, a "second face" from the side
                    paint(o, mat)                                 # and the back; only the mask stays white)
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
