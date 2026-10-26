"""A BACKPLATE for a still camera (Blender, background): one picture of the place instead of a 360 panorama.

    blender -b SHOT.blend --python bpy_jobs/plate_world.py -- --info OUT.json
    blender -b SHOT.blend --python bpy_jobs/plate_world.py -- --plate IMG --out OUT.blend [--character Mannequin]

Why (2026-10-23, user: "if it's a still camera there's no need for a panorama right?"): the template camera (30 mm on a
36 mm sensor) sees ~62 deg of a 2048 px panorama = ~350 px across the frame, magnified 2.2x at preview and 4.4x at
final quality. A picture made at the frame's own shape gives the background 4-9x more pixels.

--info: the scene camera at the first frame (location, rotation, lens, sensor, resolution, the pitch, the horizon's
place in the picture from the top 0..1, and the fields of view) and whether it MOVES (its world matrix compared at the
start, the middle and the end: a single key on a still camera is not a move).
--plate: the template's set hidden; a large floor (z = 0) and a far wall in front of the camera, both showing the
picture by WINDOW coordinates (a camera projection: exact from a still camera), full-strength emission mixed with 30%
diffuse (the character's shadow falls on the floor, the lit picture stays close to the original); the world = the picture's mean colour (soft ambient
light). The picture must have the frame's aspect (the runner crops it).
"""
import argparse
import json
import math
import sys

import bpy
from mathutils import Vector


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", default=None)
    ap.add_argument("--plate", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--character", default="Mannequin")
    ap.add_argument("--wall", type=float, default=150.0, help="distance of the far wall from the camera (m)")
    return ap.parse_args(argv)


def camera_info(sc) -> dict:
    # still or moving: EVERY frame of the range, the whole projection (Codex review 2026-10-24: three samples missed an
    # out-and-back move, and lens, sensor, shift and a camera switch by markers were not compared)
    def state(c):
        d = c.data
        return (c.name, tuple(round(v, 5) for row in c.matrix_world for v in row), d.type, round(d.lens, 5),
                round(d.sensor_width, 5), round(d.sensor_height, 5), d.sensor_fit, round(d.shift_x, 6),
                round(d.shift_y, 6))
    sc.frame_set(sc.frame_start)
    cam = sc.camera
    if cam is None:
        raise SystemExit("the shot has no scene camera")
    if cam.data.type != "PERSP":
        raise SystemExit(f"a backplate needs a perspective camera (this one is {cam.data.type})")
    first, moves = state(cam), False
    for f in range(sc.frame_start, sc.frame_end + 1):
        sc.frame_set(f)
        c = sc.camera                                      # a marker can switch the camera
        if c is None or state(c) != first:
            moves = True
            break
    sc.frame_set(sc.frame_start)
    m = cam.matrix_world.copy()
    fwd = (m.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    pitch = math.asin(max(-1.0, min(1.0, fwd.z)))
    rx, ry = sc.render.resolution_x, sc.render.resolution_y
    d = cam.data
    fit = d.sensor_fit if d.sensor_fit != "AUTO" else ("HORIZONTAL" if rx >= ry else "VERTICAL")
    sensor = d.sensor_width if fit == "HORIZONTAL" or d.sensor_fit == "AUTO" else d.sensor_height
    if fit == "HORIZONTAL":
        hfov = 2 * math.atan(sensor / 2 / d.lens)
        vfov = 2 * math.atan(math.tan(hfov / 2) * ry / rx)
    else:
        vfov = 2 * math.atan(sensor / 2 / d.lens)
        hfov = 2 * math.atan(math.tan(vfov / 2) * rx / ry)
    horizon = 0.5 + math.tan(pitch) / (2 * math.tan(vfov / 2))      # 0 = top of the picture, 1 = bottom
    pa = sc.render.pixel_aspect_x / sc.render.pixel_aspect_y         # a lens shift up moves the horizon down; the
    horizon += d.shift_y * ((rx * pa / ry) if fit == "HORIZONTAL" else 1.0)   # shift is in units of the FITTED side
    return {"location": [round(v, 4) for v in m.to_translation()], "forward": [round(v, 4) for v in fwd],
            "pitch_deg": round(math.degrees(pitch), 3), "lens_mm": d.lens, "sensor_mm": sensor, "fit": fit,
            "resolution": [rx, ry], "hfov_deg": round(math.degrees(hfov), 3), "vfov_deg": round(math.degrees(vfov), 3),
            "horizon_from_top": round(horizon, 4), "height_m": round(m.to_translation().z, 3), "moves": moves}

PROPS = "Props"      # templates: `Mannequin*` = characters, `Props` = moving objects, `Set` = the static place


def hide_set(character):
    """Hide everything that is not the character, a camera, a light or a rig (the template's set), as pano_world.
    The `Props` collection (moving or touched objects, e.g. thrown balls) stays visible too."""
    keep = set()
    for c in str(character).split(","):                          # "Mannequin,Mannequin2,...": several characters
        if c.strip() in bpy.data.collections:
            keep |= {o.name for o in bpy.data.collections[c.strip()].all_objects}
    if PROPS in bpy.data.collections:                            # the template's props play in the run's place
        keep |= {o.name for o in bpy.data.collections[PROPS].all_objects}
    hidden = []
    for o in bpy.data.objects:
        if o.name in keep or o.type in ("CAMERA", "LIGHT", "ARMATURE", "EMPTY"):
            continue
        if o.parent and o.parent.type == "ARMATURE":
            continue
        o.hide_render = True
        o.hide_viewport = True
        hidden.append(o.name)
    return hidden


SHADOW_MIX = 0.3     # the share of diffuse in the plate material: the character's shadow shows, the lit picture stays
#                      close to the original (the panorama ground's "0.6 emission + diffuse" rendered the plate 5 L
#                      brighter and paler, 2026-10-23)


def plate_material(img):
    """The picture seen through the camera (Window coordinates): full-strength emission mixed with SHADOW_MIX diffuse."""
    mat = bpy.data.materials.new("PlateMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = "EXTEND"
    diffuse = nt.nodes.new("ShaderNodeBsdfDiffuse")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = SHADOW_MIX
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tc.outputs["Window"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], diffuse.inputs["Color"])
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], mix.inputs[1])
    nt.links.new(diffuse.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def plane(name, verts, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], [(0, 1, 2, 3)])
    me.update()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    o.data.materials.append(mat)
    return o


def mean_colour(img):
    px = img.pixels[:]
    n = len(px) // 4
    step = max(1, n // 20000)
    r = g = b = 0.0
    k = 0
    for i in range(0, n, step):
        r += px[4 * i]
        g += px[4 * i + 1]
        b += px[4 * i + 2]
        k += 1
    return (r / k, g / k, b / k)


def build(a, sc):
    info = camera_info(sc)
    if info["moves"]:
        raise SystemExit("a backplate needs a still camera: this shot's camera moves (use a 360 panorama)")
    img = bpy.data.images.load(a.plate, check_existing=True)
    img.pack()                                                    # the shot carries its picture
    hidden = hide_set(a.character)
    mat = plate_material(img)
    cam = sc.camera
    c = cam.matrix_world.to_translation()
    fwd = Vector((info["forward"][0], info["forward"][1], 0.0))
    fwd = fwd.normalized() if fwd.length > 1e-6 else Vector((0.0, 1.0, 0.0))
    side = Vector((fwd.y, -fwd.x, 0.0))
    big = 400.0
    floor = plane("PlateFloor", [(c.x - big, c.y - big, 0.0), (c.x + big, c.y - big, 0.0),
                                 (c.x + big, c.y + big, 0.0), (c.x - big, c.y + big, 0.0)], mat)
    centre = Vector((c.x, c.y, 0.0)) + fwd * a.wall
    half = a.wall * math.tan(math.radians(info["hfov_deg"]) / 2) * 1.6 + 10.0
    top = c.z + a.wall * math.tan(math.radians(info["vfov_deg"]) / 2 + abs(math.radians(info["pitch_deg"]))) * 1.6 + 10
    p0, p1 = centre - side * half, centre + side * half
    wall = plane("PlateWall", [(p0.x, p0.y, -1.0), (p1.x, p1.y, -1.0), (p1.x, p1.y, top), (p0.x, p0.y, top)], mat)
    w = bpy.data.worlds.new("PlateWorld")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    col = mean_colour(img)
    bg.inputs["Color"].default_value = (*col, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    print("PLATE_WORLD", json.dumps({"out": a.out, "hidden": len(hidden), "parts": [floor.name, wall.name],
                                     "world": [round(x, 3) for x in col], "camera": info}))


def main():
    a = args()
    sc = bpy.context.scene
    if a.info:
        info = camera_info(sc)
        with open(a.info, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=1)
        print("CAMERA_INFO", json.dumps(info))
        return
    if not (a.plate and a.out):
        raise SystemExit("--info OUT.json, or --plate IMG --out OUT.blend")
    build(a, sc)


if __name__ == "__main__":                  # Blender --python runs it as __main__ (a test imports it)
    main()
