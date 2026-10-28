"""A BACKPLATE for a still camera (Blender, background): one picture of the place instead of a 360 panorama.

    blender -b SHOT.blend --python bpy_jobs/plate_world.py -- --info OUT.json
    blender -b SHOT.blend --python bpy_jobs/plate_world.py -- --plate IMG --out OUT.blend [--character Mannequin]

Why (2026-10-23, user: "if it's a still camera there's no need for a panorama right?"): the template camera (30 mm on a
36 mm sensor) sees ~62 deg of a 2048 px panorama = ~350 px across the frame, magnified 2.2x at preview and 4.4x at
final quality. A picture made at the frame's own shape gives the background 4-9x more pixels.

--info: the scene camera at the first frame (location, rotation, lens, sensor, resolution, the pitch, the horizon's
place in the picture from the top 0..1, and the fields of view) and whether it MOVES (its world matrix compared at the
start, the middle and the end: a single key on a still camera is not a move).
A PIVOT camera (2026-10-27, user: "practicing moving and zooming the camera around"): it moves, but it is the same
camera at the same place on every frame and only turns or zooms (no lens shift). Its info then describes a COVER
camera at that place (the mean view direction, wide enough for the corners of every frame + 3%, the frame's shape):
the plate is made for the cover camera and projected from it (exact for any turn or zoom about that place), and
`cover` holds its pose for bpy_jobs/render_set.py and --plate.
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
from mathutils import Matrix, Quaternion, Vector


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", default=None)
    ap.add_argument("--plate", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--character", default="Mannequin")
    ap.add_argument("--wall", type=float, default=150.0, help="distance of the far wall from the camera (m)")
    ap.add_argument("--cover", default=None, help="JSON file: the cover camera the plate was MADE for (a saved pivot "
                                                   "plate; the same camera place): the picture is projected from it")
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
    rx, ry = sc.render.resolution_x, sc.render.resolution_y
    loc0 = cam.matrix_world.to_translation().copy()
    pivot, views = cam.data.shift_x == 0 and cam.data.shift_y == 0, []
    for f in range(sc.frame_start, sc.frame_end + 1):
        sc.frame_set(f)
        c = sc.camera                                      # a marker can switch the camera
        if c is None or state(c) != first:
            moves = True
        if (c is None or c.name != cam.name or (c.matrix_world.to_translation() - loc0).length > 1e-4
                or c.data.shift_x != 0 or c.data.shift_y != 0):
            pivot = False
        if not pivot and moves:
            break
        views.append((c.matrix_world.to_3x3().normalized(), half_tans(c.data, rx, ry)))
    sc.frame_set(sc.frame_start)
    if moves and pivot:
        cover = cover_camera(loc0, views, rx / ry)
        if cover is not None:
            return describe(cover["matrix"], cover["lens_mm"], cover["sensor_mm"], "HORIZONTAL", 0.0, rx, ry, sc,
                            moves=True, pivot=True,
                            cover={k: v for k, v in cover.items() if k != "matrix"})
    m = cam.matrix_world.copy()
    d = cam.data
    return describe(m, d.lens, d.sensor_width if d.sensor_fit in ("HORIZONTAL", "AUTO") else d.sensor_height,
                    d.sensor_fit, d.shift_y, rx, ry, sc, moves=moves, pivot=False)


def half_tans(d, rx, ry):
    """tan of the half fields of view (x, y) of a perspective camera data block at the frame's shape"""
    fit = d.sensor_fit if d.sensor_fit != "AUTO" else ("HORIZONTAL" if rx >= ry else "VERTICAL")
    if fit == "HORIZONTAL":
        tx = d.sensor_width / 2 / d.lens
        return tx, tx * ry / rx
    ty = (d.sensor_height if d.sensor_fit == "VERTICAL" else d.sensor_width) / 2 / d.lens
    return ty * rx / ry, ty


def cover_camera(loc, views, aspect, margin=1.03, max_half_deg=60.0):
    """the pivot's cover camera: at `loc`, looking along the mean view direction with the frames' mean roll (a camera
    rolled to follow a tilted object keeps that roll), the smallest view of the frame's shape that holds the four
    corners of every frame (`views`: [(rotation 3x3, (tan x, tan y))]); None when a frame looks more than 90 deg away
    or the cover would be wider than 2 x max_half_deg"""
    mean, up = Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0))
    for r, _ in views:
        mean += r @ Vector((0.0, 0.0, -1.0))
        up += r @ Vector((0.0, 1.0, 0.0))
    if mean.length < 1e-6:
        return None
    z = -mean.normalized()
    x = up.cross(z)
    if x.length < 1e-6:
        return None
    x.normalize()
    rc = Matrix((x, z.cross(x), z)).transposed()
    q = rc.to_quaternion()
    ri = rc.transposed()
    tx_c = ty_c = 0.0
    for r, (tx, ty) in views:
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                p = ri @ (r @ Vector((sx * tx, sy * ty, -1.0)))
                if p.z > -1e-6:
                    return None
                tx_c, ty_c = max(tx_c, abs(p.x / -p.z)), max(ty_c, abs(p.y / -p.z))
    tx_c = max(tx_c, ty_c * aspect) * margin
    if math.degrees(math.atan(tx_c)) > max_half_deg:
        return None
    sensor = 36.0
    m = Matrix.Translation(loc) @ rc.to_4x4()
    return {"matrix": m, "location": [round(v, 5) for v in loc], "rotation_quat": [round(v, 7) for v in q],
            "lens_mm": round(sensor / 2 / tx_c, 4), "sensor_mm": sensor, "frames": len(views)}


def describe(m, lens, sensor, sensor_fit, shift_y, rx, ry, sc, moves, pivot, cover=None):
    fwd = (m.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    pitch = math.asin(max(-1.0, min(1.0, fwd.z)))
    fit = sensor_fit if sensor_fit != "AUTO" else ("HORIZONTAL" if rx >= ry else "VERTICAL")
    if fit == "HORIZONTAL":
        hfov = 2 * math.atan(sensor / 2 / lens)
        vfov = 2 * math.atan(math.tan(hfov / 2) * ry / rx)
    else:
        vfov = 2 * math.atan(sensor / 2 / lens)
        hfov = 2 * math.atan(math.tan(vfov / 2) * rx / ry)
    horizon = 0.5 + math.tan(pitch) / (2 * math.tan(vfov / 2))      # 0 = top of the picture, 1 = bottom
    pa = sc.render.pixel_aspect_x / sc.render.pixel_aspect_y         # a lens shift up moves the horizon down; the
    horizon += shift_y * ((rx * pa / ry) if fit == "HORIZONTAL" else 1.0)   # shift is in units of the FITTED side
    info = {"location": [round(v, 4) for v in m.to_translation()], "forward": [round(v, 4) for v in fwd],
            "pitch_deg": round(math.degrees(pitch), 3), "lens_mm": lens, "sensor_mm": sensor, "fit": fit,
            "resolution": [rx, ry], "hfov_deg": round(math.degrees(hfov), 3), "vfov_deg": round(math.degrees(vfov), 3),
            "horizon_from_top": round(horizon, 4), "height_m": round(m.to_translation().z, 3), "moves": moves}
    if pivot:
        info["pivot"] = True
        info["cover"] = cover
    return info


def cover_object(info):
    """a camera object at the pivot's cover pose (not the render camera): the plate's projector"""
    cv = info["cover"]
    cd = bpy.data.cameras.new("PlateCover")
    cd.lens, cd.sensor_width, cd.sensor_fit = cv["lens_mm"], cv["sensor_mm"], "HORIZONTAL"
    ob = bpy.data.objects.new("PlateCover", cd)
    ob.matrix_world = Matrix.Translation(Vector(cv["location"])) @ Quaternion(cv["rotation_quat"]).to_matrix().to_4x4()
    bpy.context.scene.collection.objects.link(ob)
    return ob

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


def plate_material(img, projector=None, tans=None):
    """The picture seen through the camera (Window coordinates): full-strength emission mixed with SHADOW_MIX diffuse.
    A pivot camera: the picture is projected from `projector` (the cover camera object; `tans` its half tangents)."""
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
    if projector is None:
        nt.links.new(tc.outputs["Window"], tex.inputs["Vector"])
    else:                                    # u = 0.5 + x / (-z) / (2 tan_x), v = 0.5 + y / (-z) / (2 tan_y)
        tc.object = projector
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["Object"], sep.inputs[0])
        comb = nt.nodes.new("ShaderNodeCombineXYZ")
        for axis, t in (("X", tans[0]), ("Y", tans[1])):
            div = nt.nodes.new("ShaderNodeMath")
            div.operation = "DIVIDE"
            nt.links.new(sep.outputs[axis], div.inputs[0])
            nt.links.new(sep.outputs["Z"], div.inputs[1])
            ma = nt.nodes.new("ShaderNodeMath")
            ma.operation = "MULTIPLY_ADD"
            nt.links.new(div.outputs["Value"], ma.inputs[0])
            ma.inputs[1].default_value = -0.5 / t
            ma.inputs[2].default_value = 0.5
            nt.links.new(ma.outputs["Value"], comb.inputs[axis])
        nt.links.new(comb.outputs["Vector"], tex.inputs["Vector"])
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
    if a.cover:                              # the plate's own cover camera (made for a wider set of frames)
        with open(a.cover, encoding="utf-8") as fh:
            cv = json.load(fh)
        m = Matrix.Translation(Vector(cv["location"])) @ Quaternion(cv["rotation_quat"]).to_matrix().to_4x4()
        rx, ry = sc.render.resolution_x, sc.render.resolution_y
        info = describe(m, cv["lens_mm"], cv["sensor_mm"], "HORIZONTAL", 0.0, rx, ry, sc, moves=info["moves"],
                        pivot=True, cover=cv)
    if info["moves"] and not info.get("pivot"):
        raise SystemExit("a backplate needs a camera that stays in one place (it may turn and zoom): this shot's "
                         "camera changes its place (use a 360 panorama)")
    img = bpy.data.images.load(a.plate, check_existing=True)
    img.pack()                                                    # the shot carries its picture
    hidden = hide_set(a.character)
    if info.get("pivot"):
        tx = math.tan(math.radians(info["hfov_deg"]) / 2)
        mat = plate_material(img, cover_object(info), (tx, math.tan(math.radians(info["vfov_deg"]) / 2)))
    else:
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
