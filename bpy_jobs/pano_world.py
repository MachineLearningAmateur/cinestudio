"""Put a 360-degree panorama (equirectangular, 2:1) into a shot as its world: the sky all round, and a ground that shows
the panorama's own floor, so the feet stand on the scene of the panorama. The template's grey set is hidden; the
mannequin, its rig, the camera and the lights stay. A run calls this on its OWN copy of the shot (the template never
changes).

    blender -b SHOT.blend --python bpy_jobs/pano_world.py -- --pano PANO.png --out OUT.blend [--eye 1.6] [--yaw 0]
            [--keep-set] [--strength 1.0]

The ground (--ground dome, the default): a floor disc out to where the panorama's floor ends (the strongest long
horizontal edge 2-30 degrees below the horizon, seen from the centre at `--eye`) and an open wall there up to
`--wall-top` degrees, both coloured by the panorama from the centre: low walls, tank bases and railings stand up instead
of lying flat on the ground (checked 2026-10-21: the rooftop panorama, R 19.3 m, camera 4 m from the centre).
--ground plane (the old one): a large plane whose colour is the panorama looked up in the direction from the panorama's centre (at
`--eye` metres above the floor) to each point of the plane. Seen from a camera near that centre, the plane lines up with
the floor in the panorama; it also catches the mannequin's shadow from the lights (diffuse + a little emission).
`--yaw` turns the panorama round the vertical axis (degrees), to choose what is behind the character at frame 1.
--ground depth --depth FILE.npy (2026-10-22): a mesh with the panorama's OWN distance in every direction (MoGe-2,
cinestudio/pano_depth.py: relative distances, row 0 = the top): each pixel a vertex in Blender's panorama convention,
scaled so that the floor (25-75 degrees below the horizon) lies `--eye` below the centre; floor points within 0.3 m are
put on the floor; the sky and stretched edges (a distance jump over --edge-ratio) get no faces, so the world (the
panorama at infinity) shows there. Same projective colour as the dome: lines up from the centre, real parallax off it.
"""
import argparse
import math
import sys

import bpy


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--pano", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eye", type=float, default=1.6, help="height of the panorama's centre above the floor (metres)")
    ap.add_argument("--yaw", type=float, default=0.0, help="turn the panorama (degrees)")
    ap.add_argument("--strength", type=float, default=1.0, help="brightness of the panorama as the world light")
    ap.add_argument("--keep-set", action="store_true", help="keep the template's set objects visible")
    ap.add_argument("--character", default="Mannequin")
    ap.add_argument("--depth", default=None, help="--ground depth: the panorama's distance map (.npy)")
    ap.add_argument("--depth-cols", type=int, default=512, help="--ground depth: mesh columns (rows = half)")
    ap.add_argument("--edge-ratio", type=float, default=0.0,
                    help="--ground depth: no face across a larger distance jump (0 = keep every face: a hole shows the "
                         "world at infinity, which tore the skyline in the first test)")
    ap.add_argument("--far", type=float, default=60.0,
                    help="--ground depth: the sky and everything farther lie on a sphere of this radius (metres)")
    ap.add_argument("--ground", choices=("plane", "dome", "depth"), default="dome",
                    help="plane = a flat 200 m ground; dome = a floor disc + a wall at radius R (walls and props below "
                         "the horizon stand up instead of lying flat on the ground)")
    ap.add_argument("--radius", type=float, default=0.0, help="dome radius in metres (0 = found in the panorama)")
    ap.add_argument("--wall-top", type=float, default=30.0,
                    help="dome wall height above the eye, in degrees seen from the centre (the far skyline stays on "
                         "the world sphere at infinity above it)")
    return ap.parse_args(argv)


def mapping(nt, yaw):
    """Texture coordinate -> a rotation round Z (the panorama's yaw)."""
    m = nt.nodes.new("ShaderNodeMapping")
    m.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(yaw))
    return m


def world(img, yaw, strength):
    w = bpy.data.worlds.new("PanoWorld")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = mapping(nt, yaw)
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = img
    env.projection = "EQUIRECTANGULAR"
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def pano_material(img, eye, yaw, name):
    """A material whose colour is the panorama seen from (0, 0, eye) toward each surface point (+ diffuse for the
    character's shadow)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sub = nt.nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = (0.0, 0.0, eye)
    norm = nt.nodes.new("ShaderNodeVectorMath")
    norm.operation = "NORMALIZE"
    mp = mapping(nt, yaw)
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = img
    env.projection = "EQUIRECTANGULAR"
    diffuse = nt.nodes.new("ShaderNodeBsdfDiffuse")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 0.6
    add = nt.nodes.new("ShaderNodeAddShader")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(geo.outputs["Position"], sub.inputs[0])
    nt.links.new(sub.outputs["Vector"], norm.inputs[0])
    nt.links.new(norm.outputs["Vector"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], diffuse.inputs["Color"])
    nt.links.new(env.outputs["Color"], emit.inputs["Color"])
    nt.links.new(diffuse.outputs["BSDF"], add.inputs[0])
    nt.links.new(emit.outputs["Emission"], add.inputs[1])
    nt.links.new(add.outputs["Shader"], out.inputs["Surface"])
    return mat


def camera_reach():
    """The largest horizontal distance of the scene camera from the centre over the shot's frames."""
    sc = bpy.context.scene
    cam = sc.camera
    if cam is None:
        return 0.0
    keep = sc.frame_current
    reach = 0.0
    for f in range(sc.frame_start, sc.frame_end + 1, 5):
        sc.frame_set(f)
        p = cam.matrix_world.translation
        reach = max(reach, math.hypot(p.x, p.y))
    sc.frame_set(keep)
    return reach


def floor_radius(img, eye):
    """Where the panorama's floor ends: the strongest horizontal edge between 2 and 30 degrees below the horizon
    (median over columns), as a distance from the centre at eye height. None when no clear edge."""
    import numpy as np
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    lum = px.reshape(h, w, 4)[::-1, :, :3].mean(axis=2)          # row 0 = the top of the picture
    top, bot = int(h * (0.5 + 2 / 180)), int(h * (0.5 + 30 / 180))
    grad = np.abs(np.diff(lum[top:bot + 1], axis=0))
    rows = grad.argmax(axis=0) + top
    strong = grad.max(axis=0) > np.percentile(grad, 90)
    if strong.mean() < 0.2:
        return None
    angle = (np.median(rows[strong]) / h - 0.5) * math.pi        # radians below the horizon
    return eye / math.tan(angle) if angle > 0 else None


def dome(img, eye, yaw, radius, wall_top_deg, floor_z=0.001):
    """A floor disc of `radius` and an open wall round it up to `wall_top_deg` above the eye, both coloured by the
    panorama from (0, 0, eye): what stands below the horizon in the panorama (low walls, tank bases, railings) stands
    up on the wall at the right distance instead of lying flat on the ground."""
    import bmesh
    me = bpy.data.meshes.new("PanoDome")
    bm = bmesh.new()
    seg = 128
    top = eye + radius * math.tan(math.radians(wall_top_deg))
    ring0 = [bm.verts.new((radius * math.cos(2 * math.pi * i / seg), radius * math.sin(2 * math.pi * i / seg), floor_z))
             for i in range(seg)]
    ring1 = [bm.verts.new((v.co.x, v.co.y, top)) for v in ring0]
    centre = bm.verts.new((0, 0, floor_z))
    for i in range(seg):
        j = (i + 1) % seg
        bm.faces.new((centre, ring0[i], ring0[j]))               # the floor disc
        bm.faces.new((ring0[i], ring1[i], ring1[j], ring0[j]))   # the wall
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("PanoGround", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.data.materials.append(pano_material(img, eye, yaw, "PanoGround"))
    return ob


def depth_geometry(eye, yaw, npy, cols, far_r=60.0):
    """The numbers of the depth layer (no Blender data): every pixel's point, the scale and the floor edge."""
    import numpy as np
    d0 = np.load(npy).astype(np.float64)
    if d0.ndim != 2 or min(d0.shape) < 4:
        raise ValueError(f"--ground depth: {npy} is not a 2-D distance map")
    R0, C0 = d0.shape
    C = max(16, min(cols, C0))
    ci = np.unique(np.linspace(0, C0 - 1, C).round().astype(int))           # exactly `cols` columns (Codex 10-22)
    ri = np.unique(np.linspace(0, R0 - 1, max(8, min(R0, C // 2))).round().astype(int))   # incl. both poles
    d = d0[np.ix_(ri, ci)]
    R, C = d.shape
    v = ri / max(R0 - 1, 1)                                     # 0 = the top row
    u = ci / max(C0 - 1, 1)
    theta = (0.5 - v)[:, None] * math.pi                        # elevation (+ up)
    # Blender's equirect lookup: u = 0.5 - atan2(y, x) / 2pi (checked by a render 2026-10-22: u 0.5 = +X, 0.25 = +Y,
    # 0 = -X, 0.75 = -Y), so a pixel's direction has the angle (0.5 - u) * 2pi; the mapping node turns the LOOKUP by
    # +yaw, so the mesh is turned by -yaw
    phi = (u - 0.5)[None, :] * 2 * math.pi + math.radians(yaw)
    cos_t = np.cos(theta)
    dirs = np.stack([np.cos(phi) * cos_t, -np.sin(phi) * cos_t, np.sin(theta) * np.ones_like(phi)], axis=-1)
    ok_d = np.isfinite(d) & (d > 0)
    if not ok_d.any():
        raise ValueError("--ground depth: no positive distance in the map")
    far = np.nanmax(np.where(ok_d, d, np.nan))
    sky = ~ok_d | (d >= 0.97 * far)
    band = (theta < math.radians(-25)) & (theta > math.radians(-75)) & ~sky
    h = (d * np.sin(-theta))[band]
    h = h[np.isfinite(h) & (h > 0)]
    if h.size < 50 or float(np.median(h)) <= 0:
        raise ValueError("--ground depth: too few floor points in the depth map (25-75 degrees below the horizon)")
    scale = eye / float(np.median(h))           # assumes the band is mostly level floor (a known limit)
    dist = np.where(sky, np.inf, d * scale)   # NO far limit (Codex review 2026-10-22: a limit before the near test
    #                                           pulled far points inside the dome); far points never make near faces
    raw_z = dirs[..., 2] * np.where(sky, 0.0, dist) + eye
    rh0 = np.hypot(dirs[..., 0], dirs[..., 1]) * np.where(sky, np.inf, dist)
    # floor points (below the eye, within max(0.3 m, 4% of their distance) of the floor, above OR below it; the city
    # far below a rooftop is not floor) go ONTO the floor ALONG THEIR OWN RAY: the colour comes from the direction seen
    # from the centre, so moving only the height would fetch another pixel (Codex review 2026-10-22)
    down = (theta < -1e-6) & ~sky
    floor = down & (np.abs(raw_z) < np.maximum(0.3, 0.04 * rh0))
    t_floor = np.where(down, (eye - 0.001) / np.maximum(np.sin(-theta) * np.ones_like(phi), 1e-9), dist)
    floor &= np.abs(t_floor - dist) <= 0.10 * dist              # a small move only: a wall's base stays at the wall
    floor |= down & (theta < math.radians(-40))                 # steeply down: the floor (MoGe is weak there: a raised
    #                                                             slab under the feet in the rooftop render 2026-10-22)
    dist = np.where(floor, t_floor, dist)                       # exactly on its own ray, 1 mm above the floor
    pts = dirs * np.where(sky, 0.0, dist)[..., None]
    pts[..., 2] += eye
    rh = np.hypot(pts[..., 0], pts[..., 1])
    fl = floor & (rh < far_r)
    info = {"scale": round(scale, 4), "grid": [R, C], "sky_px": int(sky.sum()), "floor_px": int(floor.sum()),
            "floor_edge": round(1.05 * float(np.percentile(rh[fl], 99.5)), 2) if fl.any() else None,   # + 5%: the
            # wall that ends the floor is inside the near layer (its filter is 0.97 x the dome radius)
            "median_dist": round(float(np.median(dist[np.isfinite(dist)])), 2)}
    return {"pts": pts, "dist": dist, "sky": sky, "floor": floor, "R": R, "C": C}, info


def depth_mesh(img, eye, yaw, geo, edge_ratio, near_r):
    """The NEAR layer from depth_geometry(): faces whose four points are nearer than the dome wall (horizontal
    distance < 0.97 * near_r) and not sky; across a distance jump over `edge_ratio` (0 = off) no face."""
    import numpy as np
    pts, dist, sky, floor, R, C = geo["pts"], geo["dist"], geo["sky"], geo["floor"], geo["R"], geo["C"]
    idx = np.arange(R * C).reshape(R, C)
    a, b = idx[:-1, :], np.roll(idx, -1, axis=1)[:-1, :]        # the seam column joins the first one
    c, e = np.roll(idx, -1, axis=1)[1:, :], idx[1:, :]
    quad = np.stack([a.ravel(), b.ravel(), c.ravel(), e.ravel()], axis=1)
    qd = dist.reshape(-1)[quad]
    rh = np.hypot(pts[..., 0], pts[..., 1]).reshape(-1)[quad]   # the dome wall is a cylinder: horizontal distance
    ok = np.all(np.isfinite(qd), axis=1) & np.all(rh < 0.97 * near_r, axis=1) & ~np.any(sky.reshape(-1)[quad], axis=1)
    if edge_ratio > 0:
        ok &= np.max(qd, axis=1) <= edge_ratio * np.min(qd, axis=1)
    # things stand on something (Codex review 2026-10-22: a wrongly finite sky patch inside the dome made faces): keep
    # only the face groups that reach the floor (tanks, walls, the floor itself); a floating patch goes to the dome
    on = ok.reshape(R - 1, C)
    rh_all = np.hypot(pts[..., 0], pts[..., 1])
    grounded = (floor | ((pts[..., 2] < np.minimum(np.maximum(0.5, 0.06 * rh_all), 0.9 * eye)) & ~sky)).reshape(-1)
    # on or near the floor, and below the eye (Codex 10-22: beyond ~27 m, 6% was higher than the eye, so a sky patch
    # just above the horizon counted as standing on the floor)
    touches = np.any(grounded[quad], axis=1).reshape(R - 1, C)
    keep = np.zeros_like(on)
    from collections import deque
    seen = np.zeros_like(on)
    for r0, c0 in zip(*np.nonzero(on & touches)):
        if seen[r0, c0]:
            continue
        q = deque([(r0, c0)])
        seen[r0, c0] = True
        while q:
            r, c = q.popleft()
            keep[r, c] = True
            for rr, cc in ((r - 1, c), (r + 1, c), (r, (c - 1) % C), (r, (c + 1) % C)):
                if 0 <= rr < R - 1 and on[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
    floating = int((on & ~keep).sum())
    ok = keep.reshape(-1)
    quad = quad[ok]
    me = bpy.data.meshes.new("PanoDepth")
    me.from_pydata(pts.reshape(-1, 3).tolist(), [], quad[:, ::-1].tolist())   # faces toward the centre
    me.update()
    ob = bpy.data.objects.new("PanoGround", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.data.materials.append(pano_material(img, eye, yaw, "PanoGround"))
    return ob, {"faces": int(len(quad)), "dropped_faces": int((~ok).sum()), "floating_faces": floating}


def ground(img, eye, yaw):
    """A 200 m plane whose colour is the panorama seen from (0, 0, eye) toward each surface point."""
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0.001))
    plane = bpy.context.active_object
    plane.name = "PanoGround"
    for c in plane.users_collection:
        c.objects.unlink(plane)
    bpy.context.scene.collection.objects.link(plane)
    mat = bpy.data.materials.new("PanoGround")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sub = nt.nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = (0.0, 0.0, eye)
    norm = nt.nodes.new("ShaderNodeVectorMath")
    norm.operation = "NORMALIZE"
    mp = mapping(nt, yaw)
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = img
    env.projection = "EQUIRECTANGULAR"
    diffuse = nt.nodes.new("ShaderNodeBsdfDiffuse")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 0.6
    add = nt.nodes.new("ShaderNodeAddShader")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(geo.outputs["Position"], sub.inputs[0])
    nt.links.new(sub.outputs["Vector"], norm.inputs[0])
    nt.links.new(norm.outputs["Vector"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], diffuse.inputs["Color"])
    nt.links.new(env.outputs["Color"], emit.inputs["Color"])
    nt.links.new(diffuse.outputs["BSDF"], add.inputs[0])
    nt.links.new(emit.outputs["Emission"], add.inputs[1])
    nt.links.new(add.outputs["Shader"], out.inputs["Surface"])
    plane.data.materials.append(mat)
    return plane


PROPS = "Props"      # templates: `Mannequin*` = characters, `Props` = moving objects, `Set` = the static place


def hide_set(character):
    """Hide everything that is not the character, a camera, a light or the panorama ground (the template's set).
    The `Props` collection (moving or touched objects, e.g. thrown balls) stays visible too."""
    keep = set()
    for c in str(character).split(","):                          # "Mannequin,Mannequin2,...": several characters
        if c.strip() in bpy.data.collections:
            keep |= {o.name for o in bpy.data.collections[c.strip()].all_objects}
    if PROPS in bpy.data.collections:                            # the template's props play in the run's place
        keep |= {o.name for o in bpy.data.collections[PROPS].all_objects}
    hidden = []
    for o in bpy.data.objects:
        if o.name in keep or o.type in ("CAMERA", "LIGHT", "ARMATURE", "EMPTY") or o.name == "PanoGround":
            continue
        if o.parent and o.parent.type == "ARMATURE":             # parts of a rig (another character)
            continue
        o.hide_render = True
        o.hide_viewport = True
        hidden.append(o.name)
    return hidden


def main():
    a = args()
    if a.radius < 0 or not 1.0 <= a.wall_top <= 80.0:
        raise SystemExit("--radius must be >= 0 and --wall-top 1..80 degrees")
    img = bpy.data.images.load(a.pano, check_existing=True)
    img.pack()                                                    # the shot carries its panorama
    world(img, a.yaw, a.strength)
    hidden = [] if a.keep_set else hide_set(a.character)
    info = {"ground": a.ground}
    if a.ground == "dome":
        reach = camera_reach()
        found = floor_radius(img, a.eye)
        r = a.radius or found or 8.0
        r = min(max(r, 1.5 * reach, 3.0), 80.0)                   # the camera well inside; a far edge is not a floor
        dome(img, a.eye, a.yaw, r, a.wall_top)
        info.update(radius=round(r, 2), found=found and round(found, 2), camera_reach=round(reach, 2))
    elif a.ground == "depth":
        if not a.depth:
            raise SystemExit("--ground depth needs --depth FILE.npy")
        # two layers (2026-10-22): one closed depth mesh warped the far skyline at every building/sky jump, and holes
        # tore it; the dome carries the far background (skyline, sky) cleanly, the depth mesh only what is NEARER
        # than the dome wall (the floor, low walls, tanks), at their own distances
        reach = camera_reach()
        try:
            geo, dinfo = depth_geometry(a.eye, a.yaw, a.depth, a.depth_cols, a.far)
        except (ValueError, OSError) as e:                    # no usable floor in the depth: the dome alone
            geo, dinfo = None, {"depth_error": str(e)}
        edge = dinfo.get("floor_edge")
        r = min(max(a.radius or edge or floor_radius(img, a.eye) or 8.0, 1.5 * reach, 3.0), 80.0)
        dome(img, a.eye, a.yaw, r, a.wall_top, floor_z=-0.02 if geo is not None else 0.001)
        # (under the depth layer's floor: the same height made the two floors shadow each other, a darker floor)
        if geo is not None:
            dinfo.update(depth_mesh(img, a.eye, a.yaw, geo, a.edge_ratio, r)[1])
        dinfo.update(radius=round(r, 2), camera_reach=round(reach, 2))
        cam = bpy.context.scene.camera
        if cam is not None and cam.data.clip_end < 300:          # the far parts of the mesh must not be clipped
            cam.data.clip_end = 300.0
        info.update(dinfo)
    else:
        ground(img, a.eye, a.yaw)
    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    print("PANO_WORLD", {"out": a.out, "hidden": hidden, **info})


if __name__ == "__main__":
    main()
