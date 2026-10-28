"""Rooftop Assignment: a 14 s silent-comedy scene in three shots (user, 2026-10-27).

    blender -b --python templates/rooftop_assignment/build.py -- --cut 1|2|3|all --out OUT.blend
            [--portrait PORTRAIT.png] [--kits] [--res 960x540]

ONE continuous animation of 336 frames (24 fps) drives both characters and the photograph; each shot is a frame range
of it, so the last pose of a shot IS the first pose of the next:
    A_BriefingHandoff  story frames   1-144   the main camera: 35 mm wide, eases in to a 58 mm two-shot (handoff)
    B_TargetReveal     story frames 145-216   85 mm insert over SHORT's right shoulder, slow push-in (the one cut)
    C_Reaction         story frames 217-336   the main camera: pushes in on SHORT for the slash, eases back out
The main camera stays in ONE place and only turns and zooms (MAIN_KEYS): one backplate from its cover camera fits
every frame of A and C (bpy_jobs/plate_world.py, a pivot camera).
--cut N: a TEMPLATE file for one shot (its frames start at 1; collections Mannequin = TALL, Mannequin2 = SHORT, Props
= the photograph, Set = BG_01_ROOFTOP; one camera; the cinestudio template layers). --cut all: one file with the three
scenes (linked copies: the same objects, each scene its own camera and frame range) for reference renders.
--kits: the characters' proxy kits (heads, kitsune's cape) on the mannequins (reference renders only: a run adds them).

Every frame is SOLVED (no in-between of rotations): the photograph is the master object, the hands reach its edges by
two-bone IK, the feet stay planted (leg IK), so the photograph never teleports and the hands meet it in every frame.
TALL = the masked character (the template mannequin, 1.81 m); SHORT = kitsune-3d's body (1.30 m).
"""
import argparse
import json
import math
import os
import random
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bpy_jobs"))
from blender_lib import mannequin as mq  # noqa: E402

FPS = 24
REACH = []                                       # (wrist-to-target error m, story frame, arm)
# THREE shots (user, 2026-10-27: "try practicing moving and zooming the camera around, we cut only if it warrants a
# jump in scene"): A and C are seen by ONE camera that stays in one place and only turns and zooms (a pivot camera:
# one backplate fits every frame, bpy_jobs/plate_world.py); B is the insert over SHORT's shoulder (the photograph's
# picture side cannot be seen from the main camera: a real jump)
CUTS = {1: ("A_BriefingHandoff", 1, 144), 2: ("B_TargetReveal", 145, 216), 3: ("C_Reaction", 217, 336)}
MAIN_AT = (0.04, -4.2, 1.30)                      # the main camera's one place
# (story frame, aim point, lens mm): eased keys; 35 mm wide -> a 52 mm two-shot for the handoff; C pushes in on SHORT
# (72 mm: TALL's head leaves the top of the frame) for the slash and eases back out when TALL steps in
MAIN_KEYS = [(1, (0.04, 0.0, 0.93), 35.0), (44, (0.04, 0.0, 0.93), 35.0), (78, (0.0, 0.0, 1.05), 44.0),
             (104, (-0.02, 0.0, 1.12), 52.0), (144, (-0.02, 0.0, 1.12), 52.0),
             (217, (-0.02, 0.0, 1.12), 52.0), (232, (-0.02, 0.0, 1.12), 52.0), (256, (0.40, 0.0, 0.98), 72.0),
             (272, (0.40, 0.0, 0.98), 72.0), (290, (0.0, 0.0, 1.12), 52.0), (336, (0.0, 0.0, 1.12), 52.0)]
TALL_BODY = {}
SHORT_BODY = {"scale": 0.896, "head": 1.3, "legs": 0.7, "arms": 0.85, "neck": 0.35, "torso": 0.8}
TALL_AT, SHORT_AT = Vector((-0.43, 0.0, 0.0)), Vector((0.43, 0.0, 0.0))
TALL_YAW, SHORT_YAW = 14.0, -16.0                # degrees: both face the camera (-Y), turned toward each other
PHOTO_W, PHOTO_H = 0.13, 0.18                    # a portrait print (m): ~30 px tall in the S01 frame at preview


# ---------------------------------------------------------------- helpers
def smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    if isinstance(a, (tuple, list, Vector)):
        return Vector(a).lerp(Vector(b), t)
    return a + (b - a) * t


def track(keys, f):
    """A parameter's value at story frame f from [(frame, value), ...]: held before/after, eased between keys."""
    if f <= keys[0][0]:
        return keys[0][1]
    for (f0, v0), (f1, v1) in zip(keys, keys[1:]):
        if f0 <= f <= f1:
            return lerp(v0, v1, smooth((f - f0) / max(1e-6, f1 - f0)))
    return keys[-1][1]


def material(name, color, rough=0.6, metal=0.0, emit=None, strength=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = strength
    m.diffuse_color = (*color, 1)
    return m


def coll(name, parent=None):
    c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    p = parent or bpy.context.scene.collection
    if c.name not in [x.name for x in p.children]:
        p.children.link(c)
    return c


def box(col, mat, name, loc, size, rot_z=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = size
    ob.rotation_euler.z = math.radians(rot_z)
    for c in ob.users_collection:
        c.objects.unlink(ob)
    col.objects.link(ob)
    ob.data.materials.append(mat)
    return ob


def cyl(col, mat, name, loc, r, depth, rot=(0, 0, 0), verts=24):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth, location=loc, rotation=rot, vertices=verts)
    ob = bpy.context.active_object
    ob.name = name
    for c in ob.users_collection:
        c.objects.unlink(ob)
    col.objects.link(ob)
    ob.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return ob


# ---------------------------------------------------------------- BG_01_ROOFTOP
def window_material(name, base, lit, seed):
    """A tower facade: a dark wall with a grid of windows, a random share of them lit (warm or cool)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = (*base, 1)
    b.inputs["Roughness"].default_value = 0.55
    tc = nt.nodes.new("ShaderNodeNewGeometry")         # world position: windows ~2.4 m apart, floors ~3.3 m
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (0.42, 0.30, 1.0)
    mp.inputs["Location"].default_value = (seed * 0.37, seed * 0.11, 0.0)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")        # facades are vertical: (x + y) along, z up
    nt.links.new(tc.outputs["Position"], sep.inputs[0])
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    nt.links.new(sep.outputs["X"], add.inputs[0])
    nt.links.new(sep.outputs["Y"], add.inputs[1])
    cmb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(add.outputs["Value"], cmb.inputs["X"])
    nt.links.new(sep.outputs["Z"], cmb.inputs["Y"])
    nt.links.new(cmb.outputs["Vector"], mp.inputs["Vector"])
    br = nt.nodes.new("ShaderNodeTexBrick")          # the window grid (mortar = wall between the windows)
    br.inputs["Scale"].default_value = 1.0
    br.inputs["Mortar Size"].default_value = 0.22
    br.offset = 0.0
    br.inputs["Brick Width"].default_value = 0.9
    br.inputs["Row Height"].default_value = 1.1
    br.inputs["Color1"].default_value = (1, 1, 1, 1)
    br.inputs["Color2"].default_value = (1, 1, 1, 1)
    br.inputs["Mortar"].default_value = (0, 0, 0, 1)
    nt.links.new(mp.outputs["Vector"], br.inputs["Vector"])
    vor = nt.nodes.new("ShaderNodeTexWhiteNoise")    # which windows are lit (per cell)
    vor.noise_dimensions = "3D"
    snap = nt.nodes.new("ShaderNodeVectorMath")
    snap.operation = "SNAP"
    snap.inputs[1].default_value = (0.9, 1.1, 1.0)
    nt.links.new(mp.outputs["Vector"], snap.inputs[0])
    nt.links.new(snap.outputs["Vector"], vor.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = 1.0 - lit
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    nt.links.new(vor.outputs["Value"], ramp.inputs["Fac"])
    inv = nt.nodes.new("ShaderNodeMath")              # the brick Fac is 1 on the mortar: the windows = 1 - Fac
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    nt.links.new(br.outputs["Fac"], inv.inputs[1])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(inv.outputs["Value"], mul.inputs[0])
    nt.links.new(ramp.outputs["Color"], mul.inputs[1])
    hue = nt.nodes.new("ShaderNodeValToRGB")         # warm yellow mostly, some cool white-blue
    hue.color_ramp.elements[0].color = (1.0, 0.62, 0.25, 1)
    hue.color_ramp.elements[1].color = (0.65, 0.78, 1.0, 1)
    hue.color_ramp.elements[1].position = 0.85
    nt.links.new(vor.outputs["Color"], hue.inputs["Fac"])
    em = b.inputs["Emission Color"]
    nt.links.new(hue.outputs["Color"], em)
    amp = nt.nodes.new("ShaderNodeMath")
    amp.operation = "MULTIPLY"
    amp.inputs[1].default_value = 2.5
    nt.links.new(mul.outputs["Value"], amp.inputs[0])
    nt.links.new(amp.outputs["Value"], b.inputs["Emission Strength"])
    cd = nt.nodes.new("ShaderNodeCameraData")            # the night haze: far towers fade into violet-blue
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 40.0
    mr.inputs["From Max"].default_value = 320.0
    mr.inputs["To Min"].default_value = 0.0
    mr.inputs["To Max"].default_value = 0.8
    nt.links.new(cd.outputs["View Distance"], mr.inputs["Value"])
    haze = nt.nodes.new("ShaderNodeEmission")
    haze.inputs["Color"].default_value = (0.06, 0.05, 0.13, 1)
    haze.inputs["Strength"].default_value = 1.0
    ms = nt.nodes.new("ShaderNodeMixShader")
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    nt.links.new(mr.outputs["Result"], ms.inputs["Fac"])
    nt.links.new(b.outputs["BSDF"], ms.inputs[1])
    nt.links.new(haze.outputs["Emission"], ms.inputs[2])
    nt.links.new(ms.outputs["Shader"], out.inputs["Surface"])
    return m


def build_rooftop():
    """BG_01_ROOFTOP: a flat wet concrete roof (24 x 24 m) with a low parapet on all sides (the rear one behind the
    characters), ventilation machinery far left, towers with lit windows all around (the S03 insert looks the other
    way), all in neutral grey clay under white light (Qwen gives the look: background from-set)."""
    rng = random.Random(7)
    col = coll("Set")
    # BG_01 v2 (user, 2026-10-27: "use gpt reference image to create the background of our blender render"): every
    # part is placed by a ray through the S01 camera at its place in the reference picture (rooftop_bg.png, made by
    # GPT): u = 0..1 from the left, v = 0..1 from the top. Qwen gives the set its final look later (background from-set).
    cam_loc, cam_tgt = Vector((0.04, -4.2, 1.30)), Vector((0.04, 0.0, 0.93))
    fwd = (cam_tgt - cam_loc).normalized()
    right = fwd.cross(Vector((0, 0, 1))).normalized()
    up = right.cross(fwd).normalized()
    tx, ty = 18.0 / 35.0, 18.0 / 35.0 * 9 / 16

    def ray(u, v):
        return (fwd + right * ((u - 0.5) * 2 * tx) + up * ((0.5 - v) * 2 * ty)).normalized()

    def at_y(u, v, y):                                  # where the ray of (u, v) meets the plane y = const
        d = ray(u, v)
        return cam_loc + d * ((y - cam_loc.y) / d.y)

    def on_floor(u, v):                                 # where the ray of (u, v) meets the floor z = 0
        d = ray(u, v)
        return cam_loc + d * (-cam_loc.z / d.z)

    concrete = floor_material()
    # neutral grey clay (user, 2026-10-27: "make blender preview videos stay neutral grey"; AGENTS.md 6: a template
    # has no style): the shapes carry the place, Qwen (background from-set) adds the windows, lights and colours
    parapet = material("BG_parapet", (0.26, 0.26, 0.26), rough=0.8)
    metal = material("BG_metal", (0.32, 0.32, 0.32), rough=0.6)
    dark = material("BG_dark", (0.14, 0.14, 0.14), rough=0.7)
    box(col, concrete, "Roof", (0, 2, -0.1), (24, 24, 0.2))
    # the rear parapet: its foot at v 0.595 and its top at v 0.405 in the reference
    py = on_floor(0.5, 0.595).y
    ph = max(0.8, at_y(0.5, 0.405, py).z)
    for name, loc, size in (("Parapet_Rear", (0, py + 0.15, ph / 2), (24, 0.3, ph)),
                            ("Parapet_Front", (0, -9.85, ph / 2), (24, 0.3, ph)),
                            ("Parapet_Left", (-11.85, 2, ph / 2), (0.3, 24, ph)),
                            ("Parapet_Right", (11.85, 2, ph / 2), (0.3, 24, ph))):
        box(col, parapet, name, loc, size)
        box(col, metal, name + "_Cap", (loc[0], loc[1], ph + 0.02), (size[0] + 0.05, size[1] + 0.05, 0.05))
    # the ventilation unit at the far left edge: its right side at u 0.099, its top at v 0.305, its foot at v 0.62
    hy = min(on_floor(0.05, 0.62).y, py - 0.75)
    hx_r = at_y(0.099, 0.5, hy).x
    hh = at_y(0.05, 0.305, hy).z
    box(col, metal, "HVAC_Unit", (hx_r - 0.9, hy + 0.3, hh / 2), (1.8, 0.6, hh))
    fr = min(0.32, hh * 0.18)
    for i, zz in enumerate((hh * 0.7, hh * 0.32)):      # two round fan grilles on its front
        cyl(col, dark, f"HVAC_Fan{i}", (hx_r - 0.55, hy - 0.01, zz), fr, 0.04, rot=(math.radians(90), 0, 0))
        cyl(col, metal, f"HVAC_Hub{i}", (hx_r - 0.55, hy - 0.04, zz), 0.06, 0.04, rot=(math.radians(90), 0, 0))
    # the vent stack: centre u 0.175, cap top v 0.17, foot v 0.60; an elbow pipe to the unit at v 0.55
    sy = min(on_floor(0.175, 0.60).y, py - 0.35)
    sx = at_y(0.175, 0.5, sy).x
    stop = at_y(0.175, 0.17, sy).z
    cyl(col, metal, "Vent_Stack", (sx, sy, stop / 2), 0.17, stop)
    cyl(col, metal, "Vent_Cap", (sx, sy, stop + 0.04), 0.27, 0.08)
    cyl(col, metal, "Vent_Collar", (sx, sy, stop * 0.62), 0.2, 0.06)
    pz = at_y(0.12, 0.55, sy).z
    cyl(col, metal, "Vent_Pipe", ((sx + hx_r) / 2, sy, pz), 0.14, abs(sx - hx_r) + 0.2, rot=(0, math.radians(90), 0))
    # the low box near the right edge: its left side at u 0.855, front foot at v 0.665, front top at v 0.51
    bpos = on_floor(0.855, 0.665)
    bh = max(0.3, at_y(0.855, 0.51, bpos.y).z)
    box(col, dark, "Roof_Box", (bpos.x + 0.9, bpos.y + 0.75, bh / 2), (1.8, 1.5, bh))
    box(col, metal, "Roof_Box_Top", (bpos.x + 0.9, bpos.y + 0.75, bh + 0.015), (1.85, 1.55, 0.03))
    print("BG_PLACES", {"parapet_y": round(py, 2), "parapet_h": round(ph, 2), "hvac": (round(hx_r, 2), round(hy, 2),
          round(hh, 2)), "stack": (round(sx, 2), round(sy, 2), round(stop, 2)), "box": (round(bpos.x, 2),
          round(bpos.y, 2), round(bh, 2))})
    # towers: a dense skyline behind the rear parapet (a tall near tower at the left, others across, far ones in the
    # haze), and fewer all around (the S03 insert looks the other way)
    facades = [material(f"BG_tower_{i}", (0.16 + 0.03 * i,) * 3, rough=0.85) for i in range(4)]
    n = 0
    # the skyline the S01 camera sees (its view: about +-27 deg around +Y): far towers whose tops stand between the
    # top of the frame and the parapet (height / distance 0.03..0.15), a tall near tower at the left edge
    sky = [(-34, 75, 18, 16, 34), (-12, 110, 12, 12, 16)]
    for i in range(64):
        d = rng.uniform(120, 420)
        x = rng.uniform(-0.75, 0.75) * d
        sky.append((x, d, rng.uniform(10, 26), rng.uniform(10, 24), d * rng.uniform(0.05, 0.24)))
    for x, y, w_, d_, h in sky:
        box(col, facades[n % 4], f"Tower_{n:03d}", (x, y, h / 2 - 40), (w_, d_, h + 80), rot_z=rng.uniform(-8, 8))
        n += 1
    # the other directions (the S03 insert looks back over the roof): rings of towers outside the front sector
    for ring, count, dmin, dmax, hmin, hmax in ((1, 40, 80, 160, 20, 90), (2, 36, 170, 320, 40, 160)):
        for i in range(count):
            a = 2 * math.pi * (i + rng.random() * 0.7) / count
            if math.cos(a) > 0.5:                        # the front sector has the skyline above
                continue
            d = rng.uniform(dmin, dmax)
            x, y = math.sin(a) * d, math.cos(a) * d + 2
            w_, dep, h = rng.uniform(8, 22), rng.uniform(8, 22), rng.uniform(hmin, hmax)
            box(col, facades[n % 4], f"Tower_{n:03d}", (x, y, h / 2 - 40), (w_, dep, h + 80), rot_z=rng.uniform(-20, 20))
            n += 1
    # lights (in Set: every cut sees the same light)
    def light(name, kind, loc, energy, color, size=1.0, target=(0, 0, 1.2)):
        ld = bpy.data.lights.new(name, kind)
        ld.energy, ld.color = energy, color
        if kind == "AREA":
            ld.size = size
        else:
            ld.shadow_soft_size = size
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        lo.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.collection.objects.link(lo)   # at the scene root, not in Set: a backplate run hides Set
        return lo
    light("Key_Blue", "AREA", (-4.0, -3.5, 3.8), 520, (1.0, 1.0, 1.0), size=3.0)
    light("Fill_Blue", "AREA", (3.5, -4.0, 2.2), 110, (1.0, 1.0, 1.0), size=4.0)
    light("Rim_Purple", "AREA", (1.0, 3.6, 3.0), 260, (1.0, 1.0, 1.0), size=2.0)
    light("Rim_Purple2", "AREA", (-1.6, 3.2, 2.4), 130, (1.0, 1.0, 1.0), size=1.5)
    light("Mask_Fill", "AREA", (0.0, -2.2, 1.9), 60, (1.0, 1.0, 1.0), size=1.0, target=(0, 0, 1.5))
    light("Warm_Bounce", "POINT", (9.0, 12.0, 2.5), 160, (1.0, 1.0, 1.0), size=0.5)
    w = bpy.data.worlds.new("BG_01_Night")
    w.use_nodes = True
    nt = w.node_tree
    bg = next(n for n in nt.nodes if n.type == "BACKGROUND")

    def sock(sockets, ident):
        return next(x for x in sockets if x.identifier == ident)

    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")           # the sky: lighter grey at the horizon, dark above
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.09, 0.09, 0.09, 1)
    ramp.color_ramp.elements[1].position = 0.4
    ramp.color_ramp.elements[1].color = (0.03, 0.03, 0.03, 1)
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    noise = nt.nodes.new("ShaderNodeTexNoise")          # faint clouds
    noise.inputs["Scale"].default_value = 3.0
    mul = nt.nodes.new("ShaderNodeMix")
    mul.data_type = "RGBA"
    mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 0.6
    nt.links.new(ramp.outputs["Color"], sock(mul.inputs, "A_Color"))
    nt.links.new(noise.outputs["Color"], sock(mul.inputs, "B_Color"))
    nt.links.new(sock(mul.outputs, "Result_Color"), bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.6
    return col, w


def floor_material():
    """the wet concrete: neutral grey, smooth puddles (low roughness) among rougher dry patches, joint lines every
    1.5 m (the reference shows strong reflections of the windows)"""
    m = material("BG_concrete_wet", (0.2, 0.2, 0.2), rough=0.3)
    nt = m.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 0.35
    nz.inputs["Detail"].default_value = 6.0
    nt.links.new(geo.outputs["Position"], nz.inputs["Vector"])
    rr = nt.nodes.new("ShaderNodeMapRange")              # puddles: roughness 0.04, dry: 0.45
    rr.inputs["From Min"].default_value = 0.45
    rr.inputs["From Max"].default_value = 0.6
    rr.inputs["To Min"].default_value = 0.04
    rr.inputs["To Max"].default_value = 0.45
    nt.links.new(nz.outputs["Fac"], rr.inputs["Value"])
    nt.links.new(rr.outputs["Result"], b.inputs["Roughness"])
    br = nt.nodes.new("ShaderNodeTexBrick")              # joint lines
    br.inputs["Scale"].default_value = 0.33
    br.inputs["Mortar Size"].default_value = 0.006
    br.inputs["Color1"].default_value = (0.2, 0.2, 0.2, 1)
    br.inputs["Color2"].default_value = (0.19, 0.19, 0.19, 1)
    br.inputs["Mortar"].default_value = (0.15, 0.15, 0.15, 1)
    nt.links.new(geo.outputs["Position"], br.inputs["Vector"])
    nt.links.new(br.outputs["Color"], b.inputs["Base Color"])
    return m


# ---------------------------------------------------------------- characters
def mannequin_materials():
    return {"mannequin": material("cs_mannequin", (0.72, 0.66, 0.57)), "joint": material("cs_joint", (0.42, 0.38, 0.33)),
            "stripe": material("cs_stripe", (0.15, 0.17, 0.22))}


def make_character(M, col_name, rig_name, at, yaw, body):
    import mannequin_body
    rig, col = mq.build(M, col_name, rig_name)
    rig.location = at
    rig.rotation_euler.z = math.radians(yaw)
    bpy.context.view_layer.update()
    mannequin_body.apply(rig, body, at)
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
    bpy.context.view_layer.update()
    return rig, col


_SCALES = {}                                     # (rig, bone) -> the body's own pose-bone scale


def keep_scale(rig, name):
    """set_bone_dir / yaw_bone / face_palm set a full matrix: under a scaled parent that also sets a scale (a hand at
    0.81 failed the template check's bone_scale rule); put the bone's own (body) scale back at once"""
    sc = _SCALES.get((rig.name, name))
    if sc is not None:
        rig.pose.bones[name].scale = sc
        bpy.context.view_layer.update()


def _wrap(fn):
    def w(rig, name, *a, **k):
        fn(rig, name, *a, **k)
        keep_scale(rig, name)
    return w


mq.set_bone_dir, mq.yaw_bone, mq.face_palm = _wrap(mq.set_bone_dir), _wrap(mq.yaw_bone), _wrap(mq.face_palm)


class Body:
    """World-space helpers for one rig (directions are converted to armature space for mq.set_bone_dir)."""

    def __init__(self, rig):
        self.rig = rig
        for pb in rig.pose.bones:
            _SCALES[(rig.name, pb.name)] = pb.scale.copy()
        self.R = rig.matrix_world.to_3x3().normalized()
        self.Ri = self.R.inverted()
        pb = rig.pose.bones
        mw = rig.matrix_world
        self.len = {}
        for b in ("upper_arm.L", "forearm.L", "upper_arm.R", "forearm.R", "thigh.L", "shin.L", "hand.L"):
            self.len[b] = (mw @ pb[b].tail - mw @ pb[b].head).length
        self.rest_hips = pb["hips"].head.copy()
        self.foot_rest = {s: mw @ pb["foot" + s].head for s in (".L", ".R")}
        self.ankle_h = self.foot_rest[".L"].z
        self.fwd = self.R @ Vector((0, -1, 0))          # where the body faces (world)
        self.side = {".L": self.R @ Vector((1, 0, 0)), ".R": self.R @ Vector((-1, 0, 0))}   # out to each side

    def w(self, name, tail=False):
        pb = self.rig.pose.bones[name]
        return self.rig.matrix_world @ (pb.tail if tail else pb.head)

    def dir(self, name, d_world):
        mq.set_bone_dir(self.rig, name, self.Ri @ Vector(d_world))


def two_bone(root, target, l1, l2, pole):
    """the elbow / knee of a two-bone chain from root toward target (clamped), bent toward `pole`"""
    d = target - root
    dist = max(1e-4, min(d.length, l1 + l2 - 1e-4))
    u = d.normalized()
    cos_a = max(-1.0, min(1.0, (l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist)))
    p = pole - u * pole.dot(u)
    p = p.normalized() if p.length > 1e-6 else Vector((0, 0, -1))
    return root + l1 * (cos_a * u + math.sqrt(max(0.0, 1 - cos_a * cos_a)) * p), root + u * dist


def pose_body(b: Body, P: dict):
    """Pose one rig from the parameters P at the current frame (all bones from rest: nothing is carried over)."""
    rig = b.rig
    for pb in rig.pose.bones:
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
        if (rig.name, pb.name) in _SCALES:
            pb.scale = _SCALES[(rig.name, pb.name)]
    bpy.context.view_layer.update()
    # the hips: forward/side shift (local) and drop (the knee bend)
    sh = Vector(P.get("hips", (0, 0, 0)))               # (side +x local, forward, down)
    hp = rig.pose.bones["hips"]
    m = hp.matrix.copy()
    off_arm = Vector((sh.x, -sh.y, -sh.z)) / max(1e-6, rig.scale.x)
    m.translation = m.translation + off_arm
    hp.matrix = m
    bpy.context.view_layer.update()
    yaw_hips = P.get("hips_yaw", 0.0)
    if yaw_hips:
        mq.yaw_bone(rig, "hips", yaw_hips)
    # the spine: lean forward (+) / back (-), to the side
    lean, side = math.radians(P.get("lean", 0.0)), math.radians(P.get("lean_side", 0.0))
    up = Vector((math.sin(side), -math.sin(lean), math.cos(lean) * math.cos(side))).normalized()
    chest_lean = math.radians(P.get("lean", 0.0) + P.get("chest", 0.0))
    up2 = Vector((math.sin(side), -math.sin(chest_lean), math.cos(chest_lean))).normalized()
    mq.set_bone_dir(rig, "spine", up)
    mq.set_bone_dir(rig, "chest", up2)
    if P.get("chest_yaw"):
        mq.yaw_bone(rig, "chest", P["chest_yaw"])
    for s in (".L", ".R"):                               # the shoulders (a slump lowers both)
        slump = P.get("slump", 0.0)
        o = Vector((1 if s == ".L" else -1, 0, -slump * 4.0)).normalized()
        mq.set_bone_dir(rig, "shoulder" + s, rig.pose.bones["chest"].matrix.to_3x3() @ (
            rig.pose.bones["chest"].matrix.to_3x3().inverted() @ o))
    # the head: pitch down (+), tilt (roll, + = toward its left), then yaw (turn)
    pitch, tilt = math.radians(P.get("head_pitch", 0.0)), math.radians(P.get("head_tilt", 0.0))
    nd = Vector((math.sin(tilt) * 0.6, -math.sin(pitch) * 0.6 - math.sin(chest_lean) * 0.4, 1.0)).normalized()
    mq.set_bone_dir(rig, "neck", nd)
    hd = Vector((math.sin(tilt), -math.sin(pitch), math.cos(pitch) * math.cos(tilt))).normalized()
    mq.set_bone_dir(rig, "head", hd)
    if P.get("head_yaw"):
        mq.yaw_bone(rig, "neck", P["head_yaw"] * 0.4)
        mq.yaw_bone(rig, "head", P["head_yaw"] * 0.6)
    # legs: feet planted where P says (world), knees forward
    for s in (".L", ".R"):
        foot = Vector(P["feet"][s])
        ankle = Vector((foot.x, foot.y, b.ankle_h + P.get("lift" + s, 0.0)))
        hip = b.w("thigh" + s)
        knee, reach = two_bone(hip, ankle, b.len["thigh.L"], b.len["shin.L"], b.fwd + Vector((0, 0, 0.1)))
        b.dir("thigh" + s, knee - hip)
        b.dir("shin" + s, reach - knee)
        b.dir("foot" + s, b.fwd * 0.9 + Vector((0, 0, -0.45)))
    # arms: a target (world) for the wrist, or hanging
    for s in (".L", ".R"):
        a = P["arms"][s]
        sh_w = b.w("upper_arm" + s)
        if a.get("target") is None:
            out = a.get("out", 0.12)
            fwd_ = a.get("fwd", 0.05)
            d1 = (b.side[s] * out + b.fwd * fwd_ + Vector((0, 0, -1))).normalized()
            b.dir("upper_arm" + s, d1)
            bend = a.get("bend", 0.15)
            d2 = (d1 + b.fwd * bend).normalized()
            b.dir("forearm" + s, d2)
            b.dir("hand" + s, (d2 + b.fwd * 0.05).normalized())
        else:
            target = Vector(a["target"])
            pole = Vector(a.get("pole", tuple(b.side[s] * 0.6 + Vector((0, 0, -1)))))
            elbow, wrist = two_bone(sh_w, target, b.len["upper_arm.L"], b.len["forearm.L"], pole)
            b.dir("upper_arm" + s, elbow - sh_w)
            b.dir("forearm" + s, wrist - elbow)
            hd = a.get("hand_dir")
            b.dir("hand" + s, Vector(hd) if hd is not None else (wrist - elbow).normalized())
        if a.get("palm") is not None:
            mq.face_palm(rig, "hand" + s, b.Ri @ Vector(a["palm"]), a.get("palm_w", 1.0))
        c = a.get("curl", 0.3)
        mq.curl_fingers(rig, s, c, a.get("spread", 0.0), a.get("thumb"))
        if a.get("pinch"):                               # ring and little fingers tucked (not on the photo)
            for fn, amt in (("ring", 0.7), ("pinky", 0.8)):
                for i, jm in ((1, 70), (2, 95), (3, 70)):
                    rig.pose.bones[f"{fn}_{i:02d}{s}"].rotation_quaternion = Quaternion((1, 0, 0), math.radians(jm * amt))
            bpy.context.view_layer.update()


def fit_pinch(b: Body, P: dict) -> bool:
    """move each pinching wrist so the thumb tip rests on the picture side of its corner (1.2 cm in front of the
    plane) and the photo plane runs between the thumb and the index finger; True when a target moved"""
    moved = False
    for sd in (".L", ".R"):
        a = P["arms"][sd]
        if not a.get("pinch") or a.get("target") is None:
            continue
        m = Matrix(a["photo"])
        mi = m.inverted()
        n = m.col[2].xyz.normalized()
        thumb = mi @ b.w("thumb_03" + sd, tail=True)
        index = mi @ ((b.w("index_02" + sd) + b.w("index_03" + sd, tail=True)) / 2)
        want = mi @ Vector(a["corner"])
        d_in = Vector((want.x - thumb.x, want.y - thumb.y, 0.0))          # thumb tip to the corner, in the plane
        mid = (thumb.z + index.z) / 2                                       # the plane between thumb and finger
        shift = m.to_3x3() @ d_in - n * mid
        if shift.length > 0.004:
            a = dict(a)
            a["target"] = tuple(Vector(a["target"]) + shift)
            P["arms"] = dict(P["arms"])
            P["arms"][sd] = a
            moved = True
    return moved


def solve(b: Body, P: dict, tries: int = 6):
    """pose_body, and when a wrist cannot reach its target, move the hips (and so the upper body) toward the target
    and pose again (the feet stay planted: the legs bend): the reach assist (contacts at 0.86 m apart)"""
    P = dict(P)
    for _ in range(tries + 4):
        pose_body(b, P)
        worst = Vector((0, 0, 0))
        for sd in (".L", ".R"):
            tg = P["arms"][sd].get("target")
            if tg is not None:
                e = Vector(tg) - b.w("hand" + sd)
                if e.length > worst.length:
                    worst = e
        if worst.length < 0.012:
            if not fit_pinch(b, P):                      # the pinching hands fitted to their corners
                break
            continue
        lx = worst.dot(b.R @ Vector((1, 0, 0)))          # to the body's local axes (hips: side, forward, down)
        lf = worst.dot(b.fwd)
        h = Vector(P.get("hips", (0, 0, 0)))
        h = Vector((max(-0.35, min(0.35, h.x + lx * 0.9)), max(-0.2, min(0.25, h.y + lf * 0.9)), h.z))
        P["hips"] = tuple(h)
    pose_body(b, P)
    return P


def key_all(rig, frame):
    for pb in rig.pose.bones:
        pb.keyframe_insert("rotation_quaternion", frame=frame)
    rig.pose.bones["hips"].keyframe_insert("location", frame=frame)


# ---------------------------------------------------------------- the story (story frames 1..336)
def photo_frame(center, normal, up):
    """the photograph's world matrix: its front (+Z local, the picture) along `normal`, its top (+Y local) along up"""
    n = Vector(normal).normalized()
    u = (Vector(up) - n * Vector(up).dot(n)).normalized()
    x = u.cross(n)
    m = Matrix((x, u, n)).transposed().to_4x4()
    m.translation = Vector(center)
    return m


def photo_points(m):
    """grip points of the photograph (world): its left/right edge middles, corners (as seen from its front)"""
    def p(u, v):
        return m @ Vector((u * PHOTO_W / 2, v * PHOTO_H / 2, 0))
    return {"left": p(-1.25, 0), "right": p(1.25, 0), "low_left": p(-1.1, -0.95), "low_right": p(1.1, -0.95),
            "bottom": p(0, -1), "top": p(0, 1)}


def pinch(b, m, sx, sy, side, thumb_front=True):
    """ONE-hand photo grip at a corner (user, 2026-10-27: "it should be held by the thumb pinching the corners of
    the photo, and one hand is usually enough"): the thumb pad on the picture side near corner (sx, sy) (+-1 in the
    photo's own x, y as seen from its front), the index and middle fingers flat on its back, the ring and little
    fingers tucked; the hand below the corner, its fingers toward the photo's centre. Returns the arm parameters.
    thumb_front=False: SHOWING it to someone (the user's reference picture, 2026-10-27: cards held up by hands
    from below): the thumb stays on the holder's side (the back), the fingers point up over the bottom edge on the
    picture side, near the middle of that edge."""
    n = m.col[2].xyz.normalized()
    if not thumb_front:
        corner = m @ Vector((sx * 0.02, -(PHOTO_H / 2 - 0.012), 0))
        d = m.col[1].xyz.normalized()
        hl = b.len.get("hand" + side, 0.1)
        wrist = corner - d * (hl * 0.6) + n * 0.02
        return {"target": tuple(wrist), "hand_dir": tuple(d), "palm": tuple(-n), "curl": 0.3, "thumb": 0.35,
                "pinch": True, "photo": [tuple(r) for r in m], "corner": tuple(corner)}
    corner = m @ Vector((sx * (PHOTO_W / 2 - 0.016), sy * (PHOTO_H / 2 - 0.016), 0))
    inward, upward = m.col[0].xyz * -sx, m.col[1].xyz * -sy
    centre = (inward + upward).normalized()
    # the palm faces the picture side from behind (+n), so the thumb's side is D x n (right hand) or n x D (left
    # hand): take the finger direction whose thumb lies over the photo. Holding it up for another to see, the fingers
    # point up behind it; looking at it oneself, they point in along its bottom edge and the thumb comes up on front
    cands = [(inward * 0.5 + upward * 0.85).normalized(), (inward * 0.95 + upward * 0.3).normalized()]

    def thumb_side(dv):
        return dv.cross(n) if side.endswith(".R") else n.cross(dv)

    d = max(cands, key=lambda dv: thumb_side(dv).dot(centre))
    hl = b.len.get("hand" + side, 0.1)
    wrist = corner - d * (hl * 0.6) - n * 0.02          # a first guess: solve() fits the thumb and fingers to it
    return {"target": tuple(wrist), "hand_dir": tuple(d), "palm": tuple(n), "curl": 0.0, "thumb": 0.35,
            "pinch": True, "photo": [tuple(r) for r in m], "corner": tuple(corner)}


class Story:
    """Every parameter of both characters and the photograph as eased key tracks over the 336 story frames."""

    def __init__(self, tall: Body, short: Body):
        self.T, self.S = tall, short
        self.tall_feet = {".L": TALL_AT + tall.side[".L"] * 0.17 + Vector((0, 0.02, 0)),
                          ".R": TALL_AT + tall.side[".R"] * 0.17 + Vector((0, -0.02, 0))}
        self.short_feet0 = {".L": SHORT_AT + short.side[".L"] * 0.12, ".R": SHORT_AT + short.side[".R"] * 0.12}
        self.short_step = (SHORT_AT - TALL_AT).normalized() * -0.18   # a small step toward TALL (S01 49-72)

    # -- the photograph: who holds it and where (world), per story frame
    def photo(self, f):
        T, S = self.T, self.S
        chest_t = TALL_AT + Vector((0.03, -0.32, 1.20))   # in front of TALL's chest, the elbow at ~90 deg
        held = TALL_AT + Vector((0.13, -0.34, 1.18))      # a little toward SHORT (S01 end)
        short_face = SHORT_AT + Vector((0, 0, 1.02)) + S.fwd * 0.42 + S.side[".R"] * 0.02   # ~22 cm in front of
        #                                                  his mask (his big head leans forward), at chin height
        short_out = SHORT_AT + Vector((0, 0, 0.94)) + S.fwd * 0.48 + S.side[".L"] * 0.04   # S04: arms extended
        to_short = (SHORT_AT - TALL_AT).normalized()
        inside = TALL_AT + Vector((0.05, -0.10, 1.28))   # at the opening of TALL's jacket
        hand_over = Vector((-0.12, -0.36, 1.00))          # S02: lowered to SHORT's chest height, between them
        keys_c = [(1, inside), (24, inside + Vector((0.0, -0.06, 0.0))), (40, chest_t), (48, chest_t),
                  (72, held),                                           # S01 end: held between them
                  (96, hand_over), (104, hand_over + Vector((0.02, 0, 0.01))),   # S02 handoff
                  (120, hand_over + Vector((0.16, 0.02, 0.06))),
                  (144, short_face), (168, short_face), (216, short_face),       # S02 end, S03 held
                  (229, short_out), (312, short_out), (336, short_out + Vector((0, 0, -0.06)))]   # S04: held out
        # the picture faces: TALL shows it to SHORT (front toward SHORT and the camera), SHORT looks at it (front to
        # his mask), then it is held out at arm's length, still facing him
        to_t = -to_short
        keys_n = [(1, T.fwd), (40, (to_short + T.fwd * 0.8).normalized()), (72, (to_short + T.fwd * 0.8).normalized()),
                  (96, (to_short + T.fwd * 0.5).normalized()), (120, (to_t * 0.3 + S.fwd * -1.0).normalized()),
                  (144, (-S.fwd + Vector((0, 0, 1.4))).normalized()), (216, (-S.fwd + Vector((0, 0, 1.4))).normalized()),
                  (229, (-S.fwd + Vector((0, 0, 1.0))).normalized()),
                  (336, (-S.fwd + Vector((0, 0, 0.8))).normalized())]
        roll = [(1, 0.0), (145, -5.0), (168, 0.0), (336, 0.0)]           # S03: turned 5 deg clockwise, straightened
        c = track(keys_c, f)
        n = track(keys_n, f).normalized()
        up = Matrix.Rotation(math.radians(track(roll, f)), 3, n) @ Vector((0, 0, 1))
        return photo_frame(c, n, up)

    def tall(self, f):
        T = self.T
        stp = track([(284, 0.0), (298, 1.0), (316, 1.0), (332, 0.0)], f)   # S04: a half step toward SHORT and back
        step_v = (SHORT_AT - TALL_AT).normalized() * 0.30
        lift_r = 0.05 * math.sin(math.pi * max(0.0, min(1.0, (f - 284) / 14.0))) if 284 <= f <= 298 else (
            0.05 * math.sin(math.pi * max(0.0, min(1.0, (f - 316) / 16.0))) if 316 <= f <= 332 else 0.0)
        feet = {".L": self.tall_feet[".L"] + step_v * stp, ".R": self.tall_feet[".R"] + step_v * stp * 0.6}
        P = {"feet": feet, "lift.L": lift_r, "hips": (0.025, 0.0, 0.01), "lean": 0.0, "head_yaw": 10.0, "head_pitch": 4.0,
             "arms": {".L": {"out": 0.1, "bend": 0.1, "curl": 0.35}, ".R": {"out": 0.1, "bend": 0.12, "curl": 0.35}}}
        P["head_pitch"] = track([(1, 4), (24, 14), (48, 8), (72, 6), (144, 4), (217, 4), (264, 5), (336, 6)], f)
        P["head_yaw"] = track([(1, 10), (48, 14), (72, 16), (144, 14), (217, 12), (240, 12), (250, 12), (268, 26),
                               (290, 26), (312, 22), (336, 22)], f)
        P["hips"] = tuple(track([(1, (0.025, 0, 0.012)), (72, (0.03, 0, 0.012)), (90, (0.06, 0.02, 0.02)),
                                 (112, (0.06, 0.02, 0.02)), (130, (0.02, 0, 0.008)), (284, (0.02, 0, 0.008)),
                                 (298, (0.04, 0.02, 0.01)), (314, (0.04, 0.02, 0.01)), (330, (0.02, 0, 0.008)),
                                 (336, (0.02, 0, 0.008))], f))
        P["chest_yaw"] = track([(1, 0), (40, 8), (72, 12), (88, 26), (104, 28), (128, 6), (144, 0), (284, 0), (298, 30),
                                (314, 30), (330, 4), (336, 4)], f)
        P["lean_side"] = track([(72, 0), (90, 12), (104, 12), (128, 0), (284, 0), (298, 4), (314, 4), (330, 0), (336, 0)], f)
        P["lean"] = track([(72, 0), (90, 6), (104, 6), (128, 0), (336, 0)], f)
        m = self.photo(f)
        hang = T.w("upper_arm.R") + Vector((0, 0, -0.50)) + T.fwd * 0.07 + T.side[".R"] * 0.05
        P["arms"][".R"] = {"target": tuple(hang), "pole": tuple(T.side[".R"] * 0.5 + Vector((0, 0.3, -1))),
                           "curl": 0.35}            # the right arm is ALWAYS solved (no switch from a hanging pose)
        # the right hand: to the jacket (1-24), takes the photograph out (25-48), shows it, hands it over (73-120)
        if f <= 118:
            pc = pinch(T, m, -1, -1, ".R", thumb_front=False)   # he shows it: the thumb behind
            grip = Vector(pc["target"]) if f > 20 else T.w("chest") + Vector((0.06, -0.12, 0.1))
            reach = track([(1, 0.0), (18, 1.0)], f) if f <= 20 else 1.0
            rest_w = hang
            tgt = lerp(rest_w, grip, reach)
            if f > 100:                                   # releases after SHORT pinches it (S02 frames 28-46)
                away = track([(100, 0.0), (118, 1.0)], f)
                tgt = lerp(Vector(pc["target"]), rest_w, away)
            arm = {"target": tuple(tgt), "pole": tuple(T.side[".R"] * 0.6 + Vector((0, 0.35, -1))), "curl": 0.2,
                   "thumb": 0.2, "palm": tuple(T.fwd * -1)}
            if 22 <= f <= 100:
                arm.update({k: v for k, v in pc.items() if k != "target"})
            P["arms"][".R"] = arm

        # S04: TALL takes SHORT's raised right wrist (281-296), lowers it (297-312), releases (313-320), arm back
        if 281 <= f:
            wrist = self.S.w("hand.R")
            t_in = track([(281, 0.0), (296, 1.0)], f)
            t_out = track([(312, 0.0), (324, 1.0)], f)
            rest_w = hang
            tgt = lerp(lerp(rest_w, wrist + Vector((0, 0, 0.02)), t_in), rest_w, t_out)
            P["arms"][".R"] = {"target": tuple(tgt), "pole": tuple(T.side[".R"] * 0.4 + Vector((0, 0.3, -1))),
                               "palm": (0, 0, -1), "palm_w": t_in * (1 - t_out), "curl": 0.25 + 0.4 * t_in * (1 - t_out)}
        return P

    def short(self, f):
        S = self.S
        st = track([(49, 0.0), (66, 1.0)], f)               # the small step toward TALL (left foot leads)
        feet = {".L": self.short_feet0[".L"] + self.short_step * st, ".R": self.short_feet0[".R"] + self.short_step * st * 0.5}
        lift_l = 0.06 * math.sin(math.pi * max(0.0, min(1.0, (f - 49) / 17.0))) if 49 <= f <= 66 else 0.0
        P = {"feet": feet, "lift.L": lift_l}
        P["hips"] = tuple(track([(1, (0.0, 0.02, 0.05)), (10, (0.012, 0.025, 0.055)), (20, (-0.01, 0.02, 0.05)),
                                 (49, (0.0, 0.03, 0.05)), (72, (-0.06, 0.06, 0.06)), (144, (-0.05, 0.05, 0.05)),
                                 (217, (-0.05, 0.05, 0.05)), (232, (-0.04, 0.0, 0.0)), (264, (-0.04, -0.02, 0.0)),
                                 (312, (-0.04, 0.0, 0.01)), (336, (-0.04, 0.02, 0.04))], f))
        P["lean"] = track([(1, 10), (48, 10), (72, 15), (96, 12), (144, 8), (217, 8), (232, -6), (250, -10),
                           (264, -10), (312, -6), (324, 4), (336, 6)], f)
        P["chest"] = track([(217, 0), (232, -6), (264, -6), (312, -2), (336, 2)], f)
        P["lean_side"] = track([(72, 0), (92, 8), (112, 8), (134, 0), (336, 0)], f)
        P["slump"] = track([(312, 0.0), (330, 0.05)], f)
        P["head_pitch"] = track([(1, 2), (48, 2), (60, 10), (72, 12), (120, 8), (144, 15), (217, 15), (229, -12),
                                 (250, -18), (264, -16), (312, -10), (330, 15), (336, 15)], f)
        P["head_tilt"] = track([(25, 0), (40, 20), (60, 14), (72, 8), (100, 0), (336, 0)], f)
        P["head_yaw"] = track([(1, -18), (72, -14), (130, -6), (144, 0), (217, 0), (240, -4), (312, -6), (336, -14)], f)
        m = self.photo(f)
        rest = {s: S.w("upper_arm" + s) + Vector((0, 0, -0.42)) + S.fwd * 0.1 for s in (".L", ".R")}
        waist = {s: SHORT_AT + S.side[s] * 0.12 + S.fwd * 0.2 + Vector((0, 0, 0.62)) for s in (".L", ".R")}
        # S01: hands near the waist, fidget; S02: the right hand pinches the photo's lower right corner (one hand)
        r_on = track([(80, 0.0), (100, 1.0)], f)
        fid = 0.02 * math.sin(f * 0.35)
        r_free = waist[".R"] + Vector((0, 0, fid)) + Vector((0, 0, 0.12)) * track([(60, 0.0), (72, 1.0)], f)
        l_free = waist[".L"] + Vector((0, 0, -fid))
        pr, pl = pinch(S, m, 1, -1, ".R"), pinch(S, m, -1, -1, ".L")
        R = {"target": tuple(lerp(r_free, Vector(pr["target"]), r_on)),
             "pole": tuple(S.side[".R"] * 0.5 + Vector((0, 0.15, -1))), "curl": 0.3, "thumb": 0.3}
        if r_on > 0.6:
            R.update({k: v for k, v in pr.items() if k != "target"})
        L = {"target": tuple(l_free), "pole": tuple(S.side[".L"] * 0.5 + Vector((0, 0.15, -1))), "curl": 0.35}
        if f >= 217:
            # S04 (user, 2026-10-27: "kitsune holds the photo in front of her with arms extended and then have the
            # other arm do the thumb across neck gesture from camera's left to right assuming camera right's hand is
            # holding photo"): the left hand (camera right) joins the photo's other lower corner (217-229), the
            # right hand (camera left) lets go and does the slash: a fist, knuckles up, palm to the neck, so the
            # thumb points across toward his left (camera right), dragged from his right to his left
            l_on = track([(217, 0.0), (229, 1.0)], f)
            L["target"] = tuple(lerp(waist[".L"], Vector(pl["target"]), l_on))
            if l_on > 0.6:
                L.update({k: v for k, v in pl.items() if k != "target"})
            neck = S.w("head") + S.fwd * 0.15 + S.side[".R"] * 0.10 + Vector((0, 0, -0.06))   # the throat under
            #                                                  the mask (his chibi neck is short: its base is low)
            sweep_a = neck + S.side[".R"] * 0.06 + Vector((0, 0, 0.02))
            sweep_b = neck + S.side[".L"] * 0.2
            raised = S.w("upper_arm.R") + S.side[".R"] * 0.22 + Vector((0, 0, 0.16)) + S.fwd * 0.12
            frozen = neck + S.fwd * 0.12 + S.side[".R"] * 0.06 + Vector((0, 0, -0.04))   # caught: the fist
            #                                                  comes back in front of the throat and freezes there
            front = SHORT_AT + S.fwd * 0.24 + S.side[".R"] * 0.12 + Vector((0, 0, 0.98))   # in TALL's reach
            r_track = [(229, Vector(pr["target"])), (240, raised), (249, sweep_a), (259, sweep_b), (269, frozen),
                       (280, frozen)]
            if f >= 229:
                R = {"target": tuple(track(r_track, f)), "pole": tuple(S.side[".R"] * 0.5 + Vector((0, 0.15, -1))),
                     "curl": 0.5, "thumb": 0.3}
                if 245 <= f <= 280:
                    R.update({"hand_dir": (0.0, 0.0, 1.0), "palm": tuple(-S.fwd), "curl": 0.95, "thumb": 0.0,
                              "spread": 0.0})
                if f >= 281:                               # TALL holds the wrist (296) and lowers it in front of
                    R["target"] = tuple(lerp(frozen, front, track([(296, 0.0), (312, 1.0)], f)))   # him
                    if f > 312:                            # let go: it falls to his side
                        R["target"] = tuple(lerp(front, rest[".R"], track([(314, 0.0), (326, 1.0)], f)))
        P["arms"] = {".R": R, ".L": L}
        return P


# ---------------------------------------------------------------- cameras
def camera(name, loc, target, lens, scene=None):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = 36.0
    cd.sensor_fit = "HORIZONTAL"
    cd.clip_start, cd.clip_end = 0.05, 800
    co = bpy.data.objects.new(name, cd)
    co.location = loc
    co.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    (scene or bpy.context.scene).collection.objects.link(co)
    return co


def build_cameras(story):
    cams = {}
    main = camera("Cam_Main", MAIN_AT, MAIN_KEYS[0][1], MAIN_KEYS[0][2])
    for f, aim, lens in MAIN_KEYS:                      # turns and zooms only: its place never changes
        main.rotation_euler = (Vector(aim) - Vector(MAIN_AT)).to_track_quat("-Z", "Y").to_euler()
        main.keyframe_insert("rotation_euler", frame=f)
        main.data.lens = lens
        main.data.keyframe_insert("lens", frame=f)
    cams[1] = cams[3] = main
    # S03: over SHORT's right shoulder, a little above, looking at the photograph; a slow push-in (frames 25-60)
    S = story.S
    m = story.photo(145)
    c = m.translation - m.col[1].xyz.normalized() * 0.03   # a little low: the thumb on the lower corner shows
    n = m.col[2].xyz.normalized()                      # along the photograph's own facing (tilted back ~45 deg:
    d = -(n + S.side[".R"] * 0.28 + Vector((0, 0, 0.25))).normalized()   # it clears his head; a little to his right
    cam = camera("Cam_S03", c - d * 0.98, c, 85)
    up = m.col[1].xyz                                    # the photograph stands upright in the frame
    z = (-d).normalized()
    x = up.cross(z).normalized()
    cam.rotation_euler = Matrix((x, z.cross(x), z)).transposed().to_euler()
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 0.9
    cam.data.dof.aperture_fstop = 2.0
    for sf, lens in ((145, 85.0), (169, 85.0), (205, 99.0), (216, 99.0)):   # the push-in is a ZOOM: the camera
        cam.data.lens = lens                             # keeps its place (a pivot camera: one backplate fits)
        cam.data.keyframe_insert("lens", frame=sf)
    cams[2] = cam
    return cams


# ---------------------------------------------------------------- the photograph
def build_photo(portrait):
    col = coll("Props")
    bpy.ops.mesh.primitive_plane_add(size=1)
    ob = bpy.context.active_object
    ob.name = "Photo"
    for c in ob.users_collection:
        c.objects.unlink(ob)
    col.objects.link(ob)
    ob.scale = (PHOTO_W, PHOTO_H, 1)
    bpy.ops.object.transform_apply(scale=True)
    so = ob.modifiers.new("thick", "SOLIDIFY")
    so.thickness, so.offset = 0.002, 0.0
    front = material("Photo_Front", (0.35, 0.12, 0.14), rough=0.25)
    if portrait and os.path.isfile(portrait):
        nt = front.node_tree
        b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(portrait, check_existing=True)
        tex.image.pack()
        nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
    back = material("Photo_Back", (0.82, 0.8, 0.76), rough=0.6)
    ob.data.materials.append(front)
    ob.data.materials.append(back)
    so.material_offset = 1
    so.material_offset_rim = 1
    ob.rotation_mode = "QUATERNION"
    return ob


# ---------------------------------------------------------------- build
def build(cut, portrait=None, kits=False, res=(1280, 720)):
    if bpy.app.background:                               # a headless build starts from an empty file; in a live
        bpy.ops.wm.read_factory_settings(use_empty=True)   # Blender (MCP) a reset would stop the MCP add-on: a new
        sc = bpy.context.scene                          # scene there instead
    else:
        sc = bpy.data.scenes.new("Rooftop_Build")
        bpy.context.window.scene = sc
    sc.render.fps = FPS
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = res[0], res[1], 100
    set_col, world = build_rooftop()
    sc.world = world
    M = mannequin_materials()
    tall_rig, tall_col = make_character(M, "Mannequin", "MannequinRig", TALL_AT, TALL_YAW, TALL_BODY)
    short_rig, short_col = make_character(M, "Mannequin2", "MannequinRig2", SHORT_AT, SHORT_YAW, SHORT_BODY)
    photo = build_photo(portrait)
    T, S = Body(tall_rig), Body(short_rig)
    story = Story(T, S)
    cams = build_cameras(story)
    if cut == "all":
        rng = (1, 336)
    else:
        rng = (CUTS[cut][1], CUTS[cut][2])
    shift = 0 if cut == "all" else rng[0] - 1
    for f in range(rng[0], rng[1] + 1):                  # every frame solved and keyed (SHORT first: TALL takes his wrist)
        ps = solve(S, story.short(f))
        pt = solve(T, story.tall(f))
        for nm, b_, p_ in (("SHORT", S, ps), ("TALL", T, pt)):   # the contact check: wrist vs its target
            for sd in (".L", ".R"):
                tg = p_["arms"][sd].get("target")
                if tg is not None:
                    err = (b_.w("hand" + sd) - Vector(tg)).length
                    REACH.append((err, f, nm + sd))
        key_all(short_rig, f - shift)
        key_all(tall_rig, f - shift)
        m = story.photo(f)
        photo.matrix_world = m
        photo.keyframe_insert("location", frame=f - shift)
        photo.keyframe_insert("rotation_quaternion", frame=f - shift)
    for ob in (tall_rig, short_rig, photo):
        if ob.animation_data and ob.animation_data.action:
            for fc in _fcurves(ob.animation_data.action):
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
    if kits:
        add_kits(tall_rig, tall_col, short_rig, short_col)
        costume(tall_rig, TALL_COSTUME)
        costume(short_rig, SHORT_COSTUME)
    if cut == "all":
        scenes = {}
        base = sc
        for k, (name, f0, f1) in CUTS.items():
            if k == 1:
                s2 = base
                s2.name = name
            else:                                         # the SAME collections and objects linked into each scene
                s2 = bpy.data.scenes.new(name)
                for c in base.collection.children:
                    s2.collection.children.link(c)
                for o in base.collection.objects:
                    s2.collection.objects.link(o)
                s2.world = base.world
                s2.render.fps = base.render.fps
                s2.render.resolution_x, s2.render.resolution_y = base.render.resolution_x, base.render.resolution_y
                s2.render.resolution_percentage = 100
            s2.frame_start, s2.frame_end = f0, f1
            s2.camera = cams[k]
            scenes[k] = s2
        return scenes
    sc.name = CUTS[cut][0]
    sc.frame_start, sc.frame_end = 1, rng[1] - rng[0] + 1
    sc.camera = cams[cut]
    keep = cams[cut].name
    for name in {co.name for co in cams.values()}:       # one camera per template file (A and C share one)
        if name != keep and name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    if shift:                                            # the camera keys were made on story frames
        cam = cams[cut]
        # Blender 5: the object and its camera data share ONE layered action (a slot each): shift each action once
        actions = {ad.action.name: ad.action for ad in (cam.animation_data, cam.data.animation_data)
                   if ad is not None and ad.action is not None}
        for act in actions.values():
            for fc in _fcurves(act):
                for kp in fc.keyframe_points:
                    kp.co.x -= shift
                    kp.handle_left.x -= shift
                    kp.handle_right.x -= shift
    sc.frame_set(1)
    return {cut: sc}


def _fcurves(action):
    """fcurves of an action (Blender 5 layered actions keep them in slots/channelbags)"""
    fcs = getattr(action, "fcurves", None)
    if fcs is not None and len(fcs):
        return list(fcs)
    out = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                out.extend(bag.fcurves)
    return out


TALL_COSTUME = {"jacket": ((0.20, 0.04, 0.24), ("chest", "shoulder", "upper_arm")), "wrap": ((0.80, 0.78, 0.74), ("forearm", "shin")),
                "glove": ((0.02, 0.02, 0.022), ("hand", "thumb", "index", "middle", "ring", "pinky")),
                "suit": ((0.03, 0.03, 0.035), ("hips", "spine", "thigh", "neck")), "shoe": ((0.04, 0.04, 0.045), ("foot", "toe"))}
SHORT_COSTUME = {"tunic": ((0.62, 0.55, 0.40), ("chest", "spine", "hips", "shoulder", "upper_arm", "forearm")),
                 "glove": ((0.22, 0.11, 0.05), ("hand", "thumb", "index", "middle", "ring", "pinky")),
                 "trousers": ((0.09, 0.09, 0.10), ("thigh",)), "boot": ((0.20, 0.11, 0.05), ("shin", "foot", "toe")),
                 "scarf": ((0.45, 0.03, 0.03), ("neck",))}


def costume(rig, spec):
    """reference renders only: the mannequin's parts coloured by the bone they follow (a costume read at a glance)"""
    mats = {k: material(f"costume_{rig.name}_{k}", c, rough=0.6) for k, (c, _) in spec.items()}
    for o in rig.children:
        if o.type != "MESH" or o.parent_type != "BONE" or o.name.startswith("Proxy"):
            continue
        base = o.parent_bone.split(".")[0].split("_0")[0]
        for k, (_, bones) in spec.items():
            if base in bones:
                o.data = o.data.copy()
                o.data.materials.clear()
                o.data.materials.append(mats[k])
                break


def add_kits(tall_rig, tall_col, short_rig, short_col):
    """the characters' proxy kits on the mannequins (reference renders): masked's and kitsune-3d's"""
    import proxy_kit
    sys.path.insert(0, ROOT)
    for name, rig, col in (("masked", tall_rig, tall_col), ("kitsune-3d", short_rig, short_col)):
        rec = json.load(open(os.path.join(ROOT, "characters", "library", name, "character.json"), encoding="utf-8"))
        spec = json.loads(json.dumps(rec.get("proxy") or {}).replace('"./', '"' + os.path.join(ROOT, "characters", "library", name).replace("\\", "/") + "/"))
        proxy_kit.build(spec, rig, col)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--portrait", default=None)
    ap.add_argument("--kits", action="store_true")
    ap.add_argument("--res", default="1280x720")
    a = ap.parse_args(argv)
    cut = "all" if a.cut == "all" else int(a.cut)
    w, h = (int(x) for x in a.res.lower().split("x"))
    build(cut, a.portrait, a.kits, (w, h))
    worst = sorted(REACH, reverse=True)[:6]
    print("REACH_WORST", [(round(e, 3), f, n) for e, f, n in worst])
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
    print("ROOFTOP_SAVED", a.out)


if __name__ == "__main__" and "--" in sys.argv:
    main()
