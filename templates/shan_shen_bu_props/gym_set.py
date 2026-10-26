"""A blocked-out school GYM as the static `Set` of a template (Blender; used by add_balls.py).

Layers of a template (user, 2026-10-24): `Mannequin`/`Mannequin2` = the characters, `Props` = moving or touched
objects, `Set` = the static place. This builds the place in `Set` (the template's floor `Plane` is moved into it),
blocked out after a reference photo of a school gym seen from the court (its layout only) (Gahr High School gym,
Wikimedia Commons "Basketball court view.jpg", Aphaxia, CC BY-SA 4.0; only looked at, nothing of it is used). A
regulation (FIBA) court centred on the camera's axis with its basket end toward the back wall: the floor, a painted
key, the key, free-throw circle, no-charge arc and three-point lines, the baseline; the basket centred (the backboard
hung from the roof 2.7 m in front of the wall); a back wall 8 m behind the dancer with padded mats on its lower 2 m, a
stripe, two symmetric double doors with exit signs, two plain banners (left and right); side walls just
outside the frame; folded wooden bleachers along the right wall (outside the frame); a metal ball cart with ten balls
in the back left corner. Every material is neutral grey (no hue); the props bring the colour. Plain blocks with flat
colours (material base colour + viewport colour for Workbench), no textures, no animation. Nothing stands in the
dancer's space or in the balls' paths (add_balls.py checks every ball against every set object). The camera looks
along +y from y = -4 at the dancer at the origin; x = screen right.
"""
import math

import bmesh
import bpy

BACK_Y = 8.0          # back wall's front face
SIDE_X = 8.0          # side walls' inner faces (just outside the frame)
WALL_H = 7.0
LINE_W, LINE_H = 0.05, 0.003

COLOURS = {                     # NEUTRAL grey clay only (user, 2026-10-25: "make it more bland, let the AI dictate the
    #                             colours and aesthetics"): R = G = B, value steps so the parts read apart
    "wood": (0.45, 0.45, 0.45, 1.0), "line": (0.85, 0.85, 0.85, 1.0), "wall": (0.65, 0.65, 0.65, 1.0),
    "line_navy": (0.85, 0.85, 0.85, 1.0), "line_green": (0.85, 0.85, 0.85, 1.0), "key": (0.38, 0.38, 0.38, 1.0),
    "stripe": (0.5, 0.5, 0.5, 1.0), "door": (0.25, 0.25, 0.25, 1.0), "exit": (0.5, 0.5, 0.5, 1.0),
    "score": (0.2, 0.2, 0.2, 1.0), "digit": (0.6, 0.6, 0.6, 1.0), "banner": (0.55, 0.55, 0.55, 1.0),
    "mat": (0.35, 0.35, 0.35, 1.0), "board": (0.75, 0.75, 0.75, 1.0), "board_line": (0.2, 0.2, 0.2, 1.0),
    "rim": (0.2, 0.2, 0.2, 1.0), "net": (0.8, 0.8, 0.8, 1.0), "steel": (0.3, 0.3, 0.3, 1.0),
    "bleacher": (0.5, 0.5, 0.5, 1.0), "bleacher_dark": (0.42, 0.42, 0.42, 1.0), "rubber": (0.15, 0.15, 0.15, 1.0),
}


def mat(key, red=None):
    name = "Gym_" + key
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    col = red or COLOURS[key]
    m.diffuse_color = col
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = col
    bsdf.inputs["Roughness"].default_value = 0.6 if key != "steel" else 0.35
    if key == "steel":
        bsdf.inputs["Metallic"].default_value = 0.8
    return m


def _obj(name, bm, material, coll, smooth=False):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    me.materials.append(material)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def box(name, x0, x1, y0, y1, z0, z1, material, coll):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x = x0 + (v.co.x + 0.5) * (x1 - x0)
        v.co.y = y0 + (v.co.y + 0.5) * (y1 - y0)
        v.co.z = z0 + (v.co.z + 0.5) * (z1 - z0)
    return _obj(name, bm, material, coll)


def ring(name, cx, cy, r, w, material, coll, a0=0.0, a1=2 * math.pi, seg=64, z=LINE_H, clip_x=None):
    """a flat annulus (or arc) on the floor; `clip_x` = (lo, hi) leaves out the pieces outside that x range"""
    bm = bmesh.new()
    n = max(4, int(seg * (a1 - a0) / (2 * math.pi)))
    inner, outer = [], []
    for i in range(n + 1):
        a = a0 + (a1 - a0) * i / n
        inner.append(bm.verts.new((cx + (r - w / 2) * math.cos(a), cy + (r - w / 2) * math.sin(a), z)))
        outer.append(bm.verts.new((cx + (r + w / 2) * math.cos(a), cy + (r + w / 2) * math.sin(a), z)))
    for i in range(n):
        if clip_x and not all(clip_x[0] <= v.co.x <= clip_x[1] for v in (inner[i], outer[i + 1])):
            continue
        bm.faces.new((inner[i], outer[i], outer[i + 1], inner[i + 1]))
    for v in [v for v in bm.verts if not v.link_faces]:
        bm.verts.remove(v)
    return _obj(name, bm, material, coll)


def torus(name, c, R, r, material, coll, seg=32, sides=10):
    bm = bmesh.new()
    rows = []
    for i in range(seg):
        a = 2 * math.pi * i / seg
        row = []
        for j in range(sides):
            b = 2 * math.pi * j / sides
            d = R + r * math.cos(b)
            row.append(bm.verts.new((c[0] + d * math.cos(a), c[1] + d * math.sin(a), c[2] + r * math.sin(b))))
        rows.append(row)
    for i in range(seg):
        for j in range(sides):
            bm.faces.new((rows[i][j], rows[(i + 1) % seg][j], rows[(i + 1) % seg][(j + 1) % sides],
                          rows[i][(j + 1) % sides]))
    return _obj(name, bm, material, coll, smooth=True)


def cone(name, c, r_top, r_bottom, height, material, coll):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, segments=24, radius1=r_bottom, radius2=r_top, depth=height)
    for v in bm.verts:
        v.co.x += c[0]
        v.co.y += c[1]
        v.co.z += c[2]
    return _obj(name, bm, material, coll, smooth=True)


def sphere(name, c, r, material, coll):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=r)
    for v in bm.verts:
        v.co.x += c[0]
        v.co.y += c[1]
        v.co.z += c[2]
    return _obj(name, bm, material, coll, smooth=True)


def build(red, ball_r, floor="Plane"):
    """the gym in the collection `Set`; returns the objects made. `red` = the props' RGBA, `ball_r` their radius."""
    sc = bpy.context.scene
    coll = bpy.data.collections.get("Set") or bpy.data.collections.new("Set")
    if coll.name not in [c.name for c in sc.collection.children]:
        sc.collection.children.link(coll)
    made = []
    # the Workbench preview's shadow light (view space: x right, y up, z toward the camera): high, a little from the
    # front left; lower lights let the 7 m side walls shade a wedge of the floor and the back wall
    sc.display.light_direction = (-0.2, 0.85, 0.5)
    if floor in bpy.data.objects:                         # the template's floor becomes the gym floor
        fl = bpy.data.objects[floor]
        for c in list(fl.users_collection):
            if c != coll:
                c.objects.unlink(fl)
        if coll not in fl.users_collection:
            coll.objects.link(fl)
        fl.data.materials.clear()
        fl.data.materials.append(mat("wood"))
        made.append(fl)
    # a regulation court (FIBA) centred on the camera's axis, its basket end toward the back wall: baseline 1.5 m in
    # front of the wall, the backboard 1.2 m inside the baseline, the rim's centre 1.575 m inside it, a 4.9 m painted
    # key to the free-throw line 5.8 m from the baseline (the dancer stands just in front of it), the free-throw circle
    # (r 1.8, its inner half dashed), the no-charge arc (r 1.25), the three-point line (r 6.75, straight 0.9 m inside the
    # sidelines); the sidelines (x = +-7.5) lie outside the frame
    base_y = BACK_Y - 1.5
    ft_y, lane, rim_y = base_y - 5.8, 2.45, base_y - 1.575
    made.append(box("Line_KeyPaint", -lane, lane, ft_y, base_y, 0.0, 0.0015, mat("key"), coll))
    wl, nl = mat("line"), mat("line_navy")
    corner_x = 7.5 - 0.9
    corner_y = rim_y - math.sqrt(6.75 ** 2 - corner_x ** 2)
    lines = [("Line_Baseline", -7.5, 7.5, base_y - LINE_W, base_y, wl),
             ("Line_Side_L", -7.5 - LINE_W, -7.5, -8.0, base_y, wl),
             ("Line_Side_R", 7.5, 7.5 + LINE_W, -8.0, base_y, wl),
             ("Line_Lane_L", -lane - LINE_W, -lane, ft_y, base_y, nl),
             ("Line_Lane_R", lane, lane + LINE_W, ft_y, base_y, nl),
             ("Line_FreeThrow", -lane, lane, ft_y - LINE_W, ft_y, nl),
             ("Line_Three_L", -corner_x - LINE_W / 2, -corner_x + LINE_W / 2, corner_y, base_y, nl),
             ("Line_Three_R", corner_x - LINE_W / 2, corner_x + LINE_W / 2, corner_y, base_y, nl),
             ("Line_NoCharge_L", -1.25 - LINE_W / 2, -1.25 + LINE_W / 2, rim_y, rim_y + 0.375, nl),
             ("Line_NoCharge_R", 1.25 - LINE_W / 2, 1.25 + LINE_W / 2, rim_y, rim_y + 0.375, nl)]
    for name, x0, x1, y0, y1, m_ in lines:
        made.append(box(name, x0, x1, y0, y1, 0.0, LINE_H, m_, coll))
    made.append(ring("Line_FreeThrowCircle", 0.0, ft_y, 1.8, LINE_W, nl, coll, a0=math.pi, a1=2 * math.pi))
    for k in range(8):                                     # the inner half of the circle, dashed
        a0 = k * math.pi / 8 + 0.04
        made.append(ring(f"Line_FreeThrowDash_{k + 1}", 0.0, ft_y, 1.8, LINE_W, nl, coll, a0=a0, a1=a0 + math.pi / 16,
                         seg=128))
    made.append(ring("Line_NoChargeArc", 0.0, rim_y, 1.25, LINE_W, nl, coll, a0=math.pi, a1=2 * math.pi))
    made.append(ring("Line_ThreeArc", 0.0, rim_y, 6.75, LINE_W, nl, coll, a0=math.pi, a1=2 * math.pi, seg=256,
                     clip_x=(-corner_x, corner_x)))
    # walls: off-white above, blue pads on the lower 2 m with a gold stripe on top, two double doors (symmetric) with
    # red exit signs, a plain banner left and right (symmetric about the hoop)
    wm = mat("wall")
    made.append(box("Wall_Back", -SIDE_X - 0.2, SIDE_X + 0.2, BACK_Y, BACK_Y + 0.25, 0.0, WALL_H, wm, coll))
    made.append(box("Wall_Left", -SIDE_X - 0.25, -SIDE_X, -6.0, BACK_Y, 0.0, WALL_H, wm, coll))
    made.append(box("Wall_Right", SIDE_X, SIDE_X + 0.25, -6.0, BACK_Y, 0.0, WALL_H, wm, coll))
    doors = [(-5.4, -3.6), (3.6, 5.4)]
    dm, ex = mat("door"), mat("exit")
    for i, (x0, x1) in enumerate(doors, 1):
        made.append(box(f"WallDoor_{i}", x0, x1, BACK_Y - 0.05, BACK_Y, 0.0, 2.15, dm, coll))
        made.append(box(f"WallDoor_{i}_Gap", (x0 + x1) / 2 - 0.01, (x0 + x1) / 2 + 0.01, BACK_Y - 0.055, BACK_Y - 0.05,
                        0.0, 2.15, mat("rubber"), coll))
        made.append(box(f"WallExit_{i}", (x0 + x1) / 2 - 0.18, (x0 + x1) / 2 + 0.18, BACK_Y - 0.06, BACK_Y, 2.3, 2.45,
                        ex, coll))
    mm = mat("mat")
    i = 1
    edges = [-SIDE_X + 0.05] + [e for d in doors for e in d] + [SIDE_X - 0.05]
    for s0, s1 in zip(edges[::2], edges[1::2]):           # the wall between the doors, in panels of about 1.2 m
        n = max(1, round((s1 - s0) / 1.2))
        pw = (s1 - s0) / n
        for k in range(n):
            made.append(box(f"WallMat_{i:02d}", s0 + k * pw + 0.015, s0 + (k + 1) * pw - 0.015, BACK_Y - 0.08, BACK_Y,
                            0.08, 2.0, mm, coll))
            i += 1
    made.append(box("WallStripe", -SIDE_X, SIDE_X, BACK_Y - 0.085, BACK_Y, 2.0, 2.12, mat("stripe"), coll))
    # two EMPTY banners, mirror images about the hoop (user, 2026-10-25: the right board is "a banner ... similar to
    # comfy", not a glowing scoreboard), so the background (and the AI) decides what they show; digit boxes made the
    # backplate draw paper squares
    for side, sx in (("", -1), ("_Right", 1)):
        x0, x1 = sorted((sx * 2.8, sx * 4.6))
        made.append(box(f"Banner{side}", x0, x1, BACK_Y - 0.03, BACK_Y, 2.6, 3.55, mat("banner"), coll))
        made.append(box(f"Banner{side}_Border", x0 - 0.06, x1 + 0.06, BACK_Y - 0.02, BACK_Y, 2.54, 3.61, mat("mat"),
                        coll))
    # the basket, centred: the backboard hung from the roof on a steel frame, its red square, the rim and a net
    st = mat("steel")
    by = base_y - 1.2                                      # the backboard's face
    for name, x0 in (("Hoop_Frame_L", -0.62), ("Hoop_Frame_R", 0.58)):
        made.append(box(name, x0, x0 + 0.04, by + 0.05, by + 0.09, 3.95, WALL_H, st, coll))
    made.append(box("Hoop_Frame_Top", -0.62, 0.62, by + 0.05, by + 0.09, 3.95, 4.0, st, coll))
    made.append(box("Hoop_Backboard", -0.9, 0.9, by, by + 0.05, 2.9, 3.95, mat("board"), coll))
    br = mat("board_line")
    for name, x0, x1, z0, z1 in (("Hoop_Square_T", -0.295, 0.295, 3.45, 3.5), ("Hoop_Square_B", -0.295, 0.295, 3.05, 3.1),
                                 ("Hoop_Square_L", -0.295, -0.245, 3.05, 3.5), ("Hoop_Square_R", 0.245, 0.295, 3.05, 3.5)):
        made.append(box(name, x0, x1, by - 0.01, by, z0, z1, br, coll))
    made.append(box("Hoop_Arm", -0.05, 0.05, rim_y + 0.2, by, 3.02, 3.07, st, coll))
    made.append(torus("Hoop_Rim", (0.0, rim_y, 3.05), 0.225, 0.012, mat("rim"), coll))
    made.append(cone("Hoop_Net", (0.0, rim_y, 2.85), 0.22, 0.15, 0.4, mat("net"), coll))
    # folded wooden bleachers along the right wall (a stack of steps, the high side against the wall), just outside
    # the frame (a sliver at the frame's edge read as a mistake)
    bm_, bd = mat("bleacher"), mat("bleacher_dark")
    for k in range(6):
        made.append(box(f"Bleacher_Step_{k + 1}", SIDE_X - 1.3 + k * 0.2, SIDE_X, -3.0, BACK_Y - 2.0,
                        k * 0.42, (k + 1) * 0.42 - 0.04, bm_ if k % 2 == 0 else bd, coll))
    # the ball cart against the back wall on the left, with ten red balls (the props' red)
    st, rb = mat("steel"), mat("rubber")
    cx, cy, w, d, h = -5.9, BACK_Y - 0.55, 1.1, 0.6, 0.75     # the back left corner, inside the frame
    made.append(box("Cart_Bottom", cx - w / 2, cx + w / 2, cy - d / 2, cy + d / 2, 0.12, 0.16, st, coll))
    for name, x0, x1, y0, y1 in (("Cart_Side_F", cx - w / 2, cx + w / 2, cy - d / 2, cy - d / 2 + 0.03),   # low rims:
                                 ("Cart_Side_B", cx - w / 2, cx + w / 2, cy + d / 2 - 0.03, cy + d / 2),   # the balls
                                 ("Cart_Side_L", cx - w / 2, cx - w / 2 + 0.03, cy - d / 2, cy + d / 2),   # show
                                 ("Cart_Side_R", cx + w / 2 - 0.03, cx + w / 2, cy - d / 2, cy + d / 2)):
        made.append(box(name, x0, x1, y0, y1, 0.16, 0.30, st, coll))
    for i, (px, py) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)), 1):          # corner posts and the top rails
        made.append(box(f"Cart_Post_{i}", cx + px * w / 2 - 0.02, cx + px * w / 2 + 0.02,
                        cy + py * d / 2 - 0.02, cy + py * d / 2 + 0.02, 0.16, h, st, coll))
    made.append(box("Cart_Rail_F", cx - w / 2, cx + w / 2, cy - d / 2 - 0.02, cy - d / 2 + 0.02, h - 0.03, h, st, coll))
    made.append(box("Cart_Rail_B", cx - w / 2, cx + w / 2, cy + d / 2 - 0.02, cy + d / 2 + 0.02, h - 0.03, h, st, coll))
    for i, (wx, wy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)), 1):
        made.append(box(f"Cart_Wheel_{i}", cx + wx * (w / 2 - 0.08) - 0.03, cx + wx * (w / 2 - 0.08) + 0.03,
                        cy + wy * (d / 2 - 0.08) - 0.06, cy + wy * (d / 2 - 0.08) + 0.06, 0.0, 0.12, rb, coll))
    red_m = mat("ball", red=red)
    r = ball_r
    spots = [(cx - 0.33, cy - 0.13, 0.16 + r), (cx, cy - 0.13, 0.16 + r), (cx + 0.33, cy - 0.13, 0.16 + r),
             (cx - 0.33, cy + 0.13, 0.16 + r), (cx, cy + 0.13, 0.16 + r), (cx + 0.33, cy + 0.13, 0.16 + r),
             (cx - 0.17, cy, 0.16 + 2.5 * r), (cx + 0.17, cy, 0.16 + 2.5 * r),
             (cx - 0.33, cy - 0.1, 0.16 + 3.0 * r + 0.06), (cx + 0.05, cy + 0.05, 0.16 + 4.0 * r)]
    for i, c in enumerate(spots, 1):
        made.append(sphere(f"Cart_Ball_{i:02d}", c, r, red_m, coll))
    return made
