"""The photograph's picture: a stylized 3D crime boss portrait (rooftop assignment, CUT 03).

    blender -b --python templates/rooftop_assignment/boss_portrait.py -- --out PORTRAIT.png [--size 520x720]

A large bald man with broad shoulders and an intimidating expression (heavy brow, small hard eyes, a down-turned mouth,
a thick neck), a burgundy-red silk robe with lapels over a dark shirt, a dark neutral background, a soft key light
from the left and a rim light (a printed photograph: no text)."""
import argparse
import math
import sys

import bpy
from mathutils import Vector


def mat(name, color, rough=0.5, sheen=0.0, spec=0.5, sss=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    for k in ("Sheen Weight", "Sheen"):
        if k in b.inputs:
            b.inputs[k].default_value = sheen
            break
    if "Subsurface Weight" in b.inputs:
        b.inputs["Subsurface Weight"].default_value = sss
    return m


def add(op, m, name, **kw):
    op(**kw)
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(m)
    bpy.ops.object.shade_smooth()
    return o


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="520x720")
    a = ap.parse_args(argv)
    w, h = (int(x) for x in a.size.split("x"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    skin = mat("skin", (0.42, 0.27, 0.20), rough=0.45, sss=0.1)
    dark = mat("brow", (0.06, 0.045, 0.04), rough=0.7)
    eye = mat("eye", (0.02, 0.02, 0.02), rough=0.2)
    robe = mat("robe", (0.20, 0.012, 0.03), rough=0.26, sheen=0.5)
    shirt = mat("shirt", (0.03, 0.03, 0.035), rough=0.6)
    bg = mat("bg", (0.035, 0.035, 0.04), rough=0.9)
    # head: a broad egg, heavy jaw, thick neck
    head = add(bpy.ops.mesh.primitive_uv_sphere_add, skin, "Head", radius=1, location=(0, 0, 1.62), segments=48, ring_count=32)
    head.scale = (0.125, 0.135, 0.155)
    jaw = add(bpy.ops.mesh.primitive_uv_sphere_add, skin, "Jaw", radius=1, location=(0, -0.02, 1.52), segments=48, ring_count=32)
    jaw.scale = (0.122, 0.118, 0.085)
    neck = add(bpy.ops.mesh.primitive_cylinder_add, skin, "Neck", radius=0.105, depth=0.22, location=(0, 0.01, 1.40))
    for x in (-1, 1):                                     # ears
        e = add(bpy.ops.mesh.primitive_uv_sphere_add, skin, f"Ear{x}", radius=1, location=(x * 0.125, 0.0, 1.62))
        e.scale = (0.018, 0.03, 0.045)
    nose = add(bpy.ops.mesh.primitive_uv_sphere_add, skin, "Nose", radius=1, location=(0, -0.135, 1.60))
    nose.scale = (0.03, 0.035, 0.045)
    for x in (-1, 1):                                     # heavy angled brows (an angry V), small hard eyes
        b = add(bpy.ops.mesh.primitive_cube_add, dark, f"Brow{x}", size=1, location=(x * 0.046, -0.13, 1.664))
        b.scale = (0.055, 0.018, 0.016)
        b.rotation_euler = (0, math.radians(-x * 22), 0)         # inner ends low: an angry V
        ey = add(bpy.ops.mesh.primitive_uv_sphere_add, eye, f"Eye{x}", radius=1, location=(x * 0.045, -0.122, 1.648))
        ey.scale = (0.014, 0.008, 0.008)
        bag = add(bpy.ops.mesh.primitive_uv_sphere_add, skin, f"Bag{x}", radius=1, location=(x * 0.045, -0.122, 1.635))
        bag.scale = (0.022, 0.012, 0.008)
    for x in (-1, 0, 1):                                   # a hard, slightly down-turned line of a mouth
        mo = add(bpy.ops.mesh.primitive_cube_add, dark, f"Mouth{x}", size=1,
                 location=(x * 0.022, -0.139, 1.522 - abs(x) * 0.006))
        mo.scale = (0.026, 0.008, 0.0065)
        mo.rotation_euler = (0, math.radians(x * 14), 0)
    # broad shoulders in a burgundy silk robe with lapels, a dark shirt at the V
    torso = add(bpy.ops.mesh.primitive_uv_sphere_add, robe, "Torso", radius=1, location=(0, 0.05, 1.05), segments=48, ring_count=32)
    torso.scale = (0.42, 0.2, 0.36)
    shirt_v = add(bpy.ops.mesh.primitive_cone_add, shirt, "ShirtV", radius1=0.12, depth=0.3, location=(0, -0.145, 1.2))
    shirt_v.rotation_euler = (math.radians(180), 0, 0)
    shirt_v.scale = (1.0, 0.25, 1.0)
    for x in (-1, 1):
        lap = add(bpy.ops.mesh.primitive_cube_add, robe, f"Lapel{x}", size=1, location=(x * 0.1, -0.16, 1.2))
        lap.scale = (0.06, 0.02, 0.2)
        lap.rotation_euler = (0, math.radians(x * 22), 0)
    back = add(bpy.ops.mesh.primitive_plane_add, bg, "Backdrop", size=4, location=(0, 0.8, 1.4))
    back.rotation_euler = (math.radians(90), 0, 0)
    # camera and light (a portrait print)
    cd = bpy.data.cameras.new("cam")
    cd.lens = 85
    co = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(co)
    co.location = (0.0, -1.9, 1.52)
    co.rotation_euler = (Vector((0, 0, 1.44)) - co.location).to_track_quat("-Z", "Y").to_euler()
    sc.camera = co
    for name, loc, en, col in (("key", (-1.0, -1.0, 2.2), 70, (1.0, 0.86, 0.74)), ("rim", (0.9, 0.6, 1.9), 60, (0.9, 0.5, 0.45)),
                               ("fill", (1.0, -1.3, 1.3), 8, (0.7, 0.75, 1.0))):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy, ld.color, ld.size = en, col, 1.0
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        lo.rotation_euler = (Vector((0, 0, 1.4)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        sc.collection.objects.link(lo)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    next(n for n in world.node_tree.nodes if n.type == "BACKGROUND").inputs["Color"].default_value = (0.01, 0.01, 0.012, 1)
    sc.world = world
    try:
        sc.view_settings.look = "AgX - Punchy"
    except TypeError:
        pass
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 64
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = w, h, 100
    sc.render.filepath = a.out
    bpy.ops.render.render(write_still=True)
    print("PORTRAIT", a.out)


if __name__ == "__main__":
    main()
