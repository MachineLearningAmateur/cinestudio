"""Four red dodgeballs for the shan_shen_bu template (a private candidate: the built-in stays unchanged).

    blender -b templates/shan_shen_bu_props/base_shot.blend --python templates/shan_shen_bu_props/add_balls.py --
            --out OUT.blend [--report REPORT.json] [--no-gym]

It also builds the static gym in the collection `Set` (gym_set.py; the template's floor moves into it).

User request (2026-10-24): "the dancer is moving to dodge something being thrown ... they dodge it twice, and then for
the kicks another ball enters the scene and times perfectly with their left and right kicks. so in total 4 balls".

Everything is found from the RIG of the open shot (no frame numbers typed in):
- the two DASHES = the fastest sideways moves of the hips (hips x speed, frames 1..120), one to each side; a dash starts
  where the hips' x speed turns to the dash's sign. Ball_Dodge1/2 is tossed from the camera side (4-4.5 m/s, 24 deg
  sideways), from the side the dancer dashes AWAY from, at the top side of the head as it was just before the dash (it
  would hit: 9 cm overlap), and crosses that place 8-9 frames after the dash starts, when the head has moved away. It
  falls behind the dancer, bounces and rolls to a stop on the floor, clear of the dancer on screen (a ball that flies
  away from a front camera never leaves the frame).
- the two KICKS = per leg, the frame where the foot is farthest out (from the hips) and high (out + up of the foot's
  centre). Ball_Kick1 (left leg, screen right) / Ball_Kick2 (right leg, screen left) comes in from that side, touches
  the foot's outer side at the kick's peak frame (centre R + 5 mm from the nearest foot vertex) and is kicked up
  and out of the frame on that side.
- every path is ballistic (g = 9.81 m/s^2) with floor bounces, keyed on EVERY frame (linear); a ball that is not in play
  is parked far away under the floor (constant keys), never hidden with hide_render.
The balls live in the collection `Props` (never in `Mannequin`: the character mask holds them out, the proxy kit does
not see them; bpy_jobs/plate_world.py and pano_world.py keep a `Props` collection visible in a run).
The script measures every frame (the gap to every vertex of the mannequin's meshes, on-screen, shared clip frames) and
writes the numbers to --report; it refuses to save when a dodge ball touches the body or a kick ball goes into the leg.
"""
import argparse
import json
import math
import sys

import os

import bmesh
import bpy
import numpy as np
from mathutils import Quaternion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gym_set  # noqa: E402

R = 0.11                       # ball radius (a 0.22 m dodgeball)
G = 9.81
RED = (0.80, 0.03, 0.03, 1.0)                  # (the first look; the props are violet now and the Set is grey)
VIOLET = (0.22, 0.06, 0.30, 1.0)               # "anime noir" props (user): a muted deep violet ball
MAGENTA = (0.70, 0.05, 0.40, 1.0)              # with a thin magenta seam ring
SET_BALL = (0.5, 0.5, 0.5, 1.0)                # the balls in the Set's cart: neutral grey like the whole Set
PARK = 40.0                    # parked balls: 40 m to the side, 40 m back, 20 m under the floor
SHARED = [(97, 121), (193, 217)]
# The dodges (user, 2026-10-25: "from the front again and throw fast balls to properly time when the ball should be next
# to the head when the body shifts for the dash"): each ball is a fast flat throw (17.5 m/s) from the front, the thrower
# just beside the camera, 12-14 deg off the camera's axis, aimed where the head was at the dash's start (the ball's
# centre 15-19 cm from the head's centre the frame before: it would hit the top/side of the head). Its closest approach
# (a 2.7-3 cm surface gap to the nearest vertex at the frames, >= 1.5 cm between the frames with the body moving
# linearly) comes at the frame of the dash's peak sideways speed or 1 frame after it. Coming from the front, its disk
# passes over the head's silhouette for 1 frame just before. Then it flies on, bounces off the back wall's mats and
# rolls out of the frame at the side. Tuned on the built-in r26 by a grid search (crossing time in quarter frames,
# angle, speed, drop, aim).
DODGE = [{"cross_after": 3.0, "aim_side": 0.11, "aim_up": 0.07, "angle_deg": -12.0, "speed": 17.5, "vz": -1.0},
         {"cross_after": 4.0, "aim_side": 0.08, "aim_up": 0.14, "angle_deg": -14.0, "speed": 17.5, "vz": -1.0},
         {"cross_after": 4.0, "aim_side": 0.12, "aim_up": 0.10, "angle_deg": -14.0, "speed": 17.5, "vz": -1.0}]
# the catch at the end pose (user: "bounce a ball into the palm down raised hand? and the character catches it?", then
# "the ball feels like it's moving in slow mo, the speed should be consistent" and "slowly attracted to the hand rather
# than actually bouncing into it"): real ballistic paths and a real floor bounce on a keyed frame; it pops up into the
# palm in 5 frames (0.21 s) and hits it still rising fast (8.7 m/s, 6.7 m/s up) on the frame the raised hand STOPS
# moving on screen ("i dont like the pause before the ball hits hand, it should be closely timed": the hand had waited
# empty 5 frames, 261-265), then stays in the hand (it rides the hand's last ~2 cm of easing). Lessons (2026-10-25,
# measured in preview px/frame at 768 wide):
# - the old pass (8.3 frames from the bounce up to the palm, arriving at 3 m/s up) slowed on screen from 51 to 28
#   px/frame: a ball that rises 1.6 m decelerates by g * flight time, so keep the rise SHORT (whole frames: the
#   bounce is a keyed frame, the floor contact shows) and arrive far from the apex;
# - a ball that travels AWAY from the camera after the bounce shrinks and slows on screen on top of gravity (ratio last/
#   first post-bounce step 0.6-0.75); one that comes slightly TOWARD the camera (dir y -0.3) keeps a constant screen
#   speed (54-57 px/frame after, 62-73 before the bounce; the kicked balls fly at ~42, at most 101);
# - time the catch to the hand's VISIBLE stop (its steps fall below 1 cm/frame, ~2 px), not to its exact settle (the
#   ease-out creeps < 2 px/frame for 3 more frames: a ball that waits for it reads as a pause)
# the GRIP of the catch (user, 2026-10-26: "why exactly is the ball ending up on top of the hand rather than under it
# compared to our blender render"): the old end pose held the ball BESIDE an upright open hand (the fingers up, the palm
# facing out; the ball 14 cm out and 6 cm below the wrist) and nothing held it, so LTX drew the common hold instead (the
# palm up, the ball on top). Now the wrist bends toward the palm before the catch (the palm faces down, the fingers
# point out), the ball comes up under the palm (catch_path: along the palm normal), and after the catch every finger and
# the thumb curl around it until they touch it (TOUCH from the ball's surface).
# The ball also moves FORWARD along the hand (toward the fingers) until it is arm_clear from the forearm and the upper
# arm (user, 2026-10-26: "shouldnt the ball be further from the arm? right now it looks like it's not the palm actually
# grabbing the ball"): under the palm block's centre, the 22 cm ball reached back under the wrist and touched the
# forearm, so the wrist seemed to hold it.
# Then PALMED, not held by the fingertips (user, 2026-10-26: "i feel like the ball should be palmed. not fingered"):
# the hand opens and spreads before the catch (open_curl, spread), the ball comes up into the PALM's centre (it may
# move forward only forward_max, to stay arm_clear from the forearm), and after the catch the spread fingers and the
# thumb curl down over its sides until they touch it.
# Then UNDER THE PALM ONLY (user, 2026-10-27: "the ball ... directly under the arm ... properly distance the ball from
# contacting the arm and should strictly contact the palm only"): with wrist 60 deg + forward 4.5 cm the ball touched
# the THUMB TIP (3 mm) and sat 1.5 cm from the forearm, so the video drew it against the forearm. Now choose_wrist
# tries every wrist bend (wrist_degs) on the catch frame's geometry: the ball goes along the palm normal to `gap` from
# the PALM BLOCK only (fingers and thumb are not the contact; they open away and curl after the catch), and the bend
# kept is the one whose palm faces most downward with the ball >= arm_clear from the forearm and upper arm (3D) and
# >= arm_clear_px from the arm's projection at preview, and >= body_clear from the rest of the body.
GRIP = {"wrist_deg": 60.0, "wrist_degs": tuple(range(0, 95, 5)), "ease": 10, "curl_frames": 3, "touch": 0.004,
        "arm_clear": 0.04, "arm_clear_px": 4.0, "forwards": tuple(i / 100 for i in range(0, 13)), "body_clear": 0.03, "open_curl": 0.0, "spread": 12.0,
        "thumb_spread": 35.0, "thumb_open": -0.25}
FAN = {"index": -1.0, "middle": 0.0, "ring": 1.0, "pinky": 2.0}   # blender_lib/mannequin.curl_fingers' fan
JOINT_MAX = {1: 70.0, 2: 95.0, 3: 70.0}            # blender_lib/mannequin.curl_fingers: degrees per joint at curl 1
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CATCH = {"after_stop": 0, "stop_step": 0.01, "flight_frames": 5, "post_speed": 5.5, "dir": (0.95, -0.3),
         "restitution": 0.8, "keep": 0.9, "gap": 0.003}
# the kicks: the ball comes from the side the foot flicks toward (more sideways than the flick: x 2.0, y 0.4), is on
# screen ~10 frames, touches the foot's front-outer side at the flick's fullest frame, and is kicked up and away
KICK = {"lead": 15, "in_dist": 2.6, "in_up": 0.6, "mix": (2.0, 0.4), "contact_up": 0.15, "out_speed": 9.0,
        "out_up": 4.5}
DASH_MIN_SPEED = 0.65          # m/s: a dash's peak hips speed (the small shifts before the dashes peak at <= 0.6)
E_N, E_T, ROLL = 0.5, 0.85, 1.5   # bounce: normal restitution, kept tangential speed; rolling deceleration (m/s^2)
E_W = 0.55                     # bounce off a wall (the padded mats)
WALLS = None                   # set when the gym is built: the back wall's face, the side walls, the bleachers' front


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--rig", default="MannequinRig")
    ap.add_argument("--character", default="Mannequin")
    ap.add_argument("--no-gym", action="store_true", help="keep the template's plain floor (no gym in `Set`)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- the rig and the body, per frame
def read_body(sc, rig, coll):
    objs = [o for o in bpy.data.collections[coll].all_objects if o.type == "MESH"]
    parts = []
    for o in objs:
        parts += [o.parent_bone or o.name] * len(o.data.vertices)
    parts = np.array(parts)
    frames = list(range(sc.frame_start, sc.frame_end + 1))
    verts = np.zeros((len(frames), len(parts), 3), np.float32)
    bones = {}
    for i, f in enumerate(frames):
        sc.frame_set(f)
        k = 0
        for o in objs:
            co = np.empty(len(o.data.vertices) * 3, np.float32)
            o.data.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            m = np.array(o.matrix_world, np.float32)
            verts[i, k:k + len(co)] = co @ m[:3, :3].T + m[:3, 3]
            k += len(co)
        mw = rig.matrix_world
        for b in ("hips", "head"):
            pb = rig.pose.bones[b]
            bones.setdefault(b, []).append(np.array(mw @ ((pb.head + pb.tail) / 2)))
        bones.setdefault("hips_head", []).append(np.array(mw @ rig.pose.bones["hips"].head))
        for b in ("hand.L", "hand.R"):                     # the hands' world matrices (the end pose's catch)
            bones.setdefault(b + "_matrix", []).append(np.array(mw @ rig.pose.bones[b].matrix))
    return frames, verts, parts, {k: np.array(v) for k, v in bones.items()}


class Camera:
    def __init__(self, sc):
        cam = sc.camera
        sc.frame_set(sc.frame_start)
        m = np.array(cam.matrix_world)
        self.pos = m[:3, 3]
        self.right, self.up, self.fwd = m[:3, 0], m[:3, 1], -m[:3, 2]
        rx, ry = sc.render.resolution_x, sc.render.resolution_y
        d = cam.data
        fit = d.sensor_fit if d.sensor_fit != "AUTO" else ("HORIZONTAL" if rx >= ry else "VERTICAL")
        if fit == "HORIZONTAL":
            self.tx = d.sensor_width / 2 / d.lens
            self.ty = self.tx * ry / rx
        else:
            self.ty = d.sensor_height / 2 / d.lens
            self.tx = self.ty * rx / ry
        self.w = 768                                   # preview pixels (the run's preview width)

    def uvz(self, p):
        d = np.asarray(p, float) - self.pos
        z = d @ self.fwd
        return (d @ self.right) / (z * self.tx), (d @ self.up) / (z * self.ty), z

    def visible(self, p):
        """any part of the ball inside the frame"""
        u, v, z = self.uvz(p)
        if z <= 0.05:
            return False
        return abs(u) < 1 + R / (z * self.tx) and abs(v) < 1 + R / (z * self.ty)

    def px(self, p):
        u, v, z = self.uvz(p)
        return np.array([(u + 1) / 2 * self.w, (1 - v) / 2 * self.w * self.ty / self.tx]), R / (z * self.tx) * self.w / 2, z


# ---------------------------------------------------------------- events from the rig
def find_dashes(frames, bones):
    """the dashes = the peaks of the hips' sideways speed above DASH_MIN_SPEED (frames 1..120); a dash starts where the
    speed turns to the dash's sign and ends where it falls below 0.1 m/s"""
    x = bones["hips_head"][:, 0]
    vx = np.diff(x, prepend=x[0]) * 24.0
    out = []
    for i, f in enumerate(frames):
        if f > 120 or i == 0 or i + 1 >= len(frames) or abs(vx[i]) < DASH_MIN_SPEED:
            continue
        if abs(vx[i]) < abs(vx[i - 1]) or abs(vx[i]) < abs(vx[i + 1]):
            continue
        sign = 1 if vx[i] > 0 else -1
        s = i
        while s > 0 and sign * vx[s - 1] > 0:
            s -= 1
        e = i
        while e + 1 < len(frames) and sign * vx[e + 1] > 0.1:
            e += 1
        out.append({"direction": "right" if sign > 0 else "left", "sign": sign, "start": frames[s], "end": frames[e],
                    "peak_speed_frame": f, "peak_speed": round(float(vx[i]), 3),
                    "hips_x_start": round(float(x[s - 1]), 3), "hips_x_end": round(float(x[e]), 3)})
    return out


def find_kicks(frames, verts, parts, bones):
    """per leg, the flick's fullest frame: the foot's centre farthest from the hips (sideways + forward) plus its height"""
    out = []
    for leg in (".L", ".R"):
        m = np.isin(parts, ["foot" + leg, "toe" + leg])
        c = verts[:, m].mean(axis=1)
        reach = np.hypot(c[:, 0] - bones["hips_head"][:, 0], c[:, 1] - bones["hips_head"][:, 1])
        i = int((reach + c[:, 2]).argmax())
        d = c[i, :2] - bones["hips_head"][i, :2]
        lo = max(0, i - 6)
        out.append({"leg": "left" if leg == ".L" else "right", "bone": "foot" + leg, "peak": frames[i],
                    "highest_frame": frames[lo + int(c[lo:i + 7, 2].argmax())],
                    "foot_centre": c[i].round(3).tolist(), "reach_m": round(float(reach[i]), 3),
                    "flick_dir": (d / np.linalg.norm(d)).round(3).tolist()})
    out.sort(key=lambda k: k["peak"])
    return out


# ---------------------------------------------------------------- physics
def ballistic(p, v, t):
    return p + v * t + np.array([0, 0, -0.5 * G]) * t * t


def simulate(p, v, f0, f_end, sub=24):
    """per-frame centres from frame f0 to f_end with floor bounces and rolling; also the frame it comes to rest"""
    p, v = np.array(p, float), np.array(v, float)
    out, rest = {f0: p.copy()}, None
    dt = 1.0 / 24.0 / sub
    for f in range(f0 + 1, f_end + 1):
        for _ in range(sub):
            if p[2] <= R + 1e-5 and abs(v[2]) < 0.35:          # rolling on the floor
                v[2], p[2] = 0.0, R
                hs = math.hypot(v[0], v[1])
                if hs > 1e-9:
                    v[:2] *= max(0.0, hs - ROLL * dt) / hs
            else:
                v[2] -= G * dt
            p += v * dt
            if p[2] < R:
                p[2] = R
                if v[2] < 0:
                    v[2] = -v[2] * E_N
                    v[:2] *= E_T
            if WALLS:                                         # the gym's back wall (mats), side walls, bleachers
                if p[1] > WALLS["back"] - R and v[1] > 0:
                    p[1], v[1] = WALLS["back"] - R, -v[1] * E_W
                    v[0] *= 0.9
                    v[2] *= 0.9
                if p[0] > WALLS["side"] - R and v[0] > 0:
                    p[0], v[0] = WALLS["side"] - R, -v[0] * E_W
                if p[0] < -WALLS["side"] + R and v[0] < 0:
                    p[0], v[0] = -WALLS["side"] + R, -v[0] * E_W
                for lo, hi in WALLS["boxes"]:                 # the bleachers, the cart, the hoop: bounce off a box
                    if np.all(p > lo - R) and np.all(p < hi + R):
                        pen = np.minimum(p - (lo - R), (hi + R) - p)
                        k = int(pen.argmin())
                        out_side = 1.0 if (p[k] - (lo[k] - R)) > ((hi[k] + R) - p[k]) else -1.0
                        p[k] = hi[k] + R if out_side > 0 else lo[k] - R
                        if v[k] * out_side < 0:
                            v[k] = -v[k] * E_W
        out[f] = p.copy()
        if rest is None and p[2] <= R + 1e-5 and v[2] == 0.0 and math.hypot(v[0], v[1]) < 0.02:
            rest = f
            v[:] = 0.0
    return out, rest


def gap(verts, parts, fi, c, mask=None):
    d = np.linalg.norm((verts[fi] if mask is None else verts[fi][mask]) - c, axis=1)
    i = int(d.argmin())
    return float(d[i] - R), str((parts if mask is None else parts[mask])[i])


# ---------------------------------------------------------------- the four paths
def dodge_path(d, prm, frames, verts, parts, bones, cam):
    fs = frames.index(d["start"])                          # the head (its mesh's centre) at the dash's start
    aim = verts[fs][parts == "head"].mean(axis=0) + np.array([-d["sign"] * prm["aim_side"], 0.0, prm["aim_up"]])
    head_before = verts[fs - 1][parts == "head"].mean(axis=0)
    would_hit = -float(np.linalg.norm(aim - head_before))   # the ball's centre this far from the head's centre
    tc = d["start"] + prm["cross_after"]                   # may fall between two frames
    a = math.radians(prm["angle_deg"])
    # from the front side, wide: away from the camera and across, from the side the dancer dashes away from
    vel = np.array([d["sign"] * prm["speed"] * math.sin(a), prm["speed"] * math.cos(a), prm["vz"]])
    path = {}
    f = int(math.floor(tc))
    while True:                                            # back in time to the throw (fully off-screen)
        p = ballistic(aim, vel, (f - tc) / 24.0)
        path[f] = p
        if not cam.visible(p) or f <= frames[0] - 40:
            break
        f -= 1
    throw = f
    f0 = int(math.ceil(tc + 1e-9))
    t0 = (f0 - tc) / 24.0
    fwd, rest = simulate(ballistic(aim, vel, t0), vel + np.array([0.0, 0.0, -G * t0]), f0, frames[-1] + 60)
    exit_f = next((g for g in sorted(fwd) if g > tc and not cam.visible(fwd[g])), None)
    path.update({g: q for g, q in fwd.items() if exit_f is None or g <= exit_f})
    sub = 9.0                                              # between the frames (the body moved linearly)
    for t10 in range((throw + 1) * 10, (f0 + 2) * 10):
        t = t10 / 10.0
        g = int(math.floor(t))
        if g not in frames or g + 1 not in frames:
            continue
        w = t - g
        body = verts[frames.index(g)] * (1 - w) + verts[frames.index(g + 1)] * w
        sub = min(sub, float(np.linalg.norm(body - ballistic(aim, vel, (t - tc) / 24.0), axis=1).min() - R))
    return {"aim": aim, "cross": tc, "throw": throw, "rest": None if exit_f else rest, "exit": exit_f, "path": path,
            "would_hit_overlap": -would_hit, "throw_from": path[throw], "sub_gap": sub}


def kick_path(k, frames, verts, parts, cam):
    fi = frames.index(k["peak"])
    m = np.isin(parts, ["foot" + k["bone"][-2:], "toe" + k["bone"][-2:]])
    centre = np.array(k["foot_centre"])
    fd = np.array(k["flick_dir"])
    side = np.array([fd[0] * KICK["mix"][0], fd[1] * KICK["mix"][1]])
    side /= np.linalg.norm(side)                           # where the ball comes from (and is kicked back to)
    dvec = np.array([side[0], side[1], KICK["contact_up"]])
    dvec /= np.linalg.norm(dvec)
    lo, hi = 0.0, 1.0                                      # touch: centre R + 5 mm from the nearest body vertex
    for _ in range(40):
        mid = (lo + hi) / 2
        g, _ = gap(verts, parts, fi, centre + dvec * mid)
        lo, hi = (mid, hi) if g < 0.005 else (lo, mid)
    contact = centre + dvec * hi
    T = KICK["lead"] / 24.0
    start = contact + np.array([side[0] * KICK["in_dist"], side[1] * KICK["in_dist"], KICK["in_up"]])
    v_in = (contact - start - 0.5 * np.array([0, 0, -G]) * T * T) / T
    path = {}
    for f in range(k["peak"] - KICK["lead"], k["peak"] + 1):
        path[f] = ballistic(start, v_in, (f - (k["peak"] - KICK["lead"])) / 24.0)
    v_out = np.array([side[0] * KICK["out_speed"], side[1] * KICK["out_speed"], KICK["out_up"]])
    fwd, _ = simulate(contact, v_out, k["peak"], min(frames[-1] + 30, k["peak"] + 60))
    path.update(fwd)
    throw = max(f for f in path if f < k["peak"] and not cam.visible(path[f]))
    exit_f = min(f for f in path if f > k["peak"] and not cam.visible(path[f]))
    path = {f: p for f, p in path.items() if throw <= f <= exit_f}
    foot_d = float(np.linalg.norm(verts[fi][m] - contact, axis=1).min())
    return {"contact": contact, "throw": throw, "exit": exit_f, "path": path, "v_in": v_in, "v_out": v_out,
            "foot_vertex_dist": foot_d, "foot_centre_dist": float(np.linalg.norm(contact - centre))}


def find_raised_hand(frames, bones, verts, parts):
    """the end pose's raised hand (the higher hand at the last frame), the frame it settles (from then on its matrix
    stays within 1 mm of the final one) and the frame it STOPS on screen: from then on the hand mesh's centre moves less
    than CATCH stop_step per frame (1 cm: ~2 preview px; the ease-out's last creep is not seen as motion)"""
    hand = max(("hand.L", "hand.R"), key=lambda b: bones[b + "_matrix"][-1][2, 3])
    ms = bones[hand + "_matrix"]
    i = len(frames) - 1
    while i > 0 and np.abs(ms[i - 1][:3, 3] - ms[-1][:3, 3]).max() < 0.001 and np.abs(ms[i - 1][:3, :3] - ms[-1][:3, :3]).max() < 0.002:
        i -= 1
    c = verts[:, parts == hand].mean(axis=1)
    step = np.linalg.norm(np.diff(c, axis=0), axis=1)          # step[j]: frame j -> j + 1
    j = len(frames) - 1
    while j > 0 and step[j - 1] < CATCH["stop_step"]:
        j -= 1
    return {"hand": hand, "settle": frames[i], "stop": frames[j], "palm_normal": ms[i][:3, 2].round(3).tolist(),
            "steps_m_before_stop": step[max(0, j - 3):j + 3].round(4).tolist()}


def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = math.radians(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)


def palm_mask(parts, hand):
    """the hand's palm surface: the palm block and the fingers' base segments (not the thumb)"""
    side = hand[-2:]
    return (parts == hand) | np.isin(parts, [f"{f}_01{side}" for f in FINGERS if f != "thumb"])


def palm_ball(verts_f, parts, hand, n, y=None, forward=0.0):
    """the ball's centre along the palm normal n from the palm block's centre (moved `forward` along the fingers'
    direction y), `gap` from the palm surface only (palm_mask)"""
    palm_m = palm_mask(parts, hand)
    palm = verts_f[parts == hand].mean(axis=0) + (0.0 if y is None else y * forward)
    lo, hi = 0.0, 0.6
    for _ in range(40):
        mid = (lo + hi) / 2
        g = float(np.linalg.norm(verts_f[palm_m] - (palm + n * mid), axis=1).min()) - R
        lo, hi = (mid, hi) if g < CATCH["gap"] else (lo, mid)
    return palm + n * hi


def choose_wrist(h, frames, verts, parts, bones, cam):
    """the wrist bend for the catch (see GRIP): every bend in wrist_degs is tried on the catch frame by turning the hand
    and finger vertices about the hand bone's head and local X (what bend_wrist's local +X turn does)"""
    hand, side = h["hand"], h["hand"][-2:]
    fi = frames.index(h["stop"] + CATCH["after_stop"])
    M = bones[hand + "_matrix"][fi]
    head, ax = M[:3, 3], M[:3, 0]
    moved = np.array([p == hand or (p.endswith(side) and p.split("_")[0] in FINGERS) for p in parts])
    arm = np.isin(parts, ["forearm" + side, "upper_arm" + side])
    rest = ~moved & ~arm
    arm_px = np.array([cam.px(v)[0] for v in verts[fi][arm]])
    tried = []
    for deg in GRIP["wrist_degs"]:
        Rm = _rot(ax, deg)
        vf = verts[fi].copy()
        vf[moved] = (vf[moved] - head) @ Rm.T + head
        n = Rm @ M[:3, 2]
        n /= np.linalg.norm(n)
        y = Rm @ M[:3, 1]
        y /= np.linalg.norm(y)
        for fw in GRIP["forwards"]:                       # the least forward place that is clear of the arm
            c = palm_ball(vf, parts, hand, n, y, fw)
            arm_gap = float(np.linalg.norm(vf[arm] - c, axis=1).min()) - R
            body_gap = float(np.linalg.norm(vf[rest] - c, axis=1).min()) - R
            cp, rp, _ = cam.px(c)
            px_gap = float(np.linalg.norm(arm_px - cp, axis=1).min()) - rp
            ok = arm_gap >= GRIP["arm_clear"] and px_gap >= GRIP["arm_clear_px"] and body_gap >= GRIP["body_clear"]
            if ok:
                break
        tried.append({"deg": deg, "forward": fw, "down": round(float(-n[2]), 3), "arm_gap_m": round(arm_gap, 3),
                      "arm_gap_px": round(float(px_gap), 1), "body_gap_m": round(body_gap, 3), "ok": ok})
    good = [t for t in tried if t["ok"]]
    if not good:
        raise SystemExit(f"add_balls: no wrist bend keeps the caught ball clear of the arm: {tried}")
    # the palm as much downward as possible (the ball hangs under it), then the ball as near the palm's centre
    best = max(good, key=lambda t: (round(t["down"], 2), -t["forward"], -t["deg"]))
    return best["deg"], best["forward"], tried


def bend_wrist(sc, rig, h, deg=None):
    """Bend the raised hand toward its palm (a local +X turn moves the bone's Y toward its palm normal Z) by `deg`
    (choose_wrist; default GRIP wrist_deg), eased in over GRIP ease frames up to the frame the hand stops, held to the
    end. Every frame's own rotation is read first: keys written while reading would change the curve between the old
    keys."""
    deg = GRIP["wrist_deg"] if deg is None else deg
    pb = rig.pose.bones[h["hand"]]
    f0, f1 = h["stop"] - GRIP["ease"], h["stop"]
    frames = list(range(f0, sc.frame_end + 1))
    orig = []
    for f in frames:
        sc.frame_set(f)
        orig.append(pb.rotation_quaternion.copy())
    for f, q in zip(frames, orig):
        t = min(1.0, max(0.0, (f - f0) / (f1 - f0)))
        w = t * t * (3.0 - 2.0 * t)
        pb.rotation_quaternion = q @ Quaternion((1.0, 0.0, 0.0), math.radians(deg * w))
        pb.keyframe_insert("rotation_quaternion", frame=f)
    sc.frame_set(sc.frame_end)
    m = rig.matrix_world @ pb.matrix
    return {"from": f0, "to": f1, "deg": deg, "palm_normal_end": [round(x, 3) for x in m.col[2][:3]],
            "fingers_dir_end": [round(x, 3) for x in m.col[1][:3]]}


def _set_curl(rig, side, finger, c, spread=0.0):
    """mannequin.curl_fingers for one finger, with the palm grip's spread on the first joint (the thumb: away from the
    index finger)"""
    sgn = 1.0 if side == ".L" else -1.0
    for i in (1, 2, 3):
        q = Quaternion((1.0, 0.0, 0.0), math.radians(JOINT_MAX[i] * c * (0.6 if finger == "thumb" else 1.0)))
        if i == 1 and spread:
            fan = -GRIP["thumb_spread"] / GRIP["spread"] if finger == "thumb" else FAN[finger]   # thumb: thumb_spread deg
            q = Quaternion((0.0, 0.0, 1.0), math.radians(spread * fan * sgn)) @ q
        rig.pose.bones[f"{finger}_{i:02d}{side}"].rotation_quaternion = q


def open_hand(sc, rig, h):
    """Open and spread the raised hand over the same frames as the wrist bend (bend_wrist), held to the end: the palm
    takes the ball, not the fingertips. Every frame's own finger rotations are read first (see bend_wrist)."""
    side = h["hand"][-2:]
    f0, f1 = h["stop"] - GRIP["ease"], h["stop"]
    frames = list(range(f0, sc.frame_end + 1))
    bones = [f"{f}_{i:02d}{side}" for f in FINGERS for i in (1, 2, 3)]
    orig = []
    for f in frames:
        sc.frame_set(f)
        orig.append({b: rig.pose.bones[b].rotation_quaternion.copy() for b in bones})
    for f, qs in zip(frames, orig):
        t = min(1.0, max(0.0, (f - f0) / (f1 - f0)))
        w = t * t * (3.0 - 2.0 * t)
        for fn in FINGERS:
            _set_curl(rig, side, fn, GRIP["thumb_open"] if fn == "thumb" else GRIP["open_curl"], GRIP["spread"])
            for i in (1, 2, 3):
                b = rig.pose.bones[f"{fn}_{i:02d}{side}"]
                b.rotation_quaternion = qs[b.name].slerp(b.rotation_quaternion, w)
                b.keyframe_insert("rotation_quaternion", frame=f)


def grip_fingers(sc, rig, coll, h, ball, catch):
    """After the catch, curl every finger (and the thumb) of the raised hand around the ball: each one as far as it goes
    before its mesh comes within GRIP touch of the ball's surface (the ball rides the hand bone, so the test at the
    last frame holds for every frame after the catch), eased in over GRIP curl_frames and held to the end."""
    side = h["hand"][-2:]
    objs = {f: [o for o in bpy.data.collections[coll].all_objects if o.type == "MESH"
                and (o.parent_bone or "").startswith(f + "_") and o.parent_bone.endswith(side)] for f in FINGERS}
    end = sc.frame_end
    sc.frame_set(catch)
    start = {f: GRIP["thumb_open"] if f == "thumb" else GRIP["open_curl"] for f in FINGERS}   # open_hand
    sc.frame_set(end)
    c = np.array(ball.matrix_world.translation)

    def clearance(f):
        bpy.context.view_layer.update()
        d = 9.0
        for o in objs[f]:
            mw = np.array(o.matrix_world)
            co = np.array([tuple(v.co) for v in o.data.vertices])
            w = co @ mw[:3, :3].T + mw[:3, 3]
            d = min(d, float(np.linalg.norm(w - c, axis=1).min()) - R)
        return d

    target = {}
    for f in FINGERS:
        best = start[f]
        for k in range(0, 41):
            cv = start[f] + (1.0 - start[f]) * k / 40.0
            _set_curl(rig, side, f, cv, GRIP["spread"])
            if clearance(f) < GRIP["touch"]:
                break
            best = cv
        target[f] = best
    keys = list(range(catch, end + 1))
    for fr in keys:
        t = min(1.0, (fr - catch) / GRIP["curl_frames"])
        w = t * t * (3.0 - 2.0 * t)
        for f in FINGERS:
            _set_curl(rig, side, f, start[f] + (target[f] - start[f]) * w, GRIP["spread"])
            for i in (1, 2, 3):
                rig.pose.bones[f"{f}_{i:02d}{side}"].keyframe_insert("rotation_quaternion", frame=fr)
    worst = 9.0
    for fr in keys:                                         # the check: no finger goes into the ball
        sc.frame_set(fr)
        c = np.array(ball.matrix_world.translation)
        for f in FINGERS:
            worst = min(worst, clearance(f))
    return {"curl_from": {f: round(v, 3) for f, v in start.items()}, "curl_to": {f: round(v, 3) for f, v in target.items()},
            "min_finger_gap_m": round(worst, 4)}


def catch_path(h, frames, verts, parts, bones, cam):
    """the ball's centre at the catch: from the palm's centre along the palm's normal (the hand bone's +z: the side the
    fingers curl to; it faces down and out) to CATCH gap from the nearest body vertex. Real physics all the way: a fast
    bounce pass in from the screen edge on the hand's side, one floor bounce on a keyed frame (vertical speed x
    restitution, horizontal x keep), a ballistic rise of `flight_frames` frames (post_speed across, the vertical speed
    solved so it meets the palm: still rising fast) that reaches the palm `after_stop` frames after the hand has
    stopped on screen; it stops there in one frame (caught) and then moves with the hand"""
    hand = h["hand"]
    fc = h["stop"] + CATCH["after_stop"]
    fi = frames.index(fc)
    n = bones[hand + "_matrix"][fi][:3, 2]                 # the palm's normal at the catch frame (the bent wrist)
    n = n / np.linalg.norm(n)
    y = bones[hand + "_matrix"][fi][:3, 1]                 # the fingers' direction
    y = y / np.linalg.norm(y)
    forward = h.get("forward", 0.0)
    c = palm_ball(verts[fi], parts, hand, n, y, forward)   # against the palm surface only (choose_wrist)
    arm = np.isin(parts, ["forearm" + hand[-2:], "upper_arm" + hand[-2:]])
    arm_gap = float(np.linalg.norm(verts[fi][arm] - c, axis=1).min()) - R
    side = 1.0 if c[0] < 0 else -1.0                       # it travels toward the hand from the hand's outer side
    d = np.array([side * CATCH["dir"][0], CATCH["dir"][1]])
    d /= np.linalg.norm(d)
    t2 = CATCH["flight_frames"] / 24.0                     # bounce -> palm in whole frames: the bounce is a keyed frame
    vbz = (c[2] - R + 0.5 * G * t2 * t2) / t2              # up, just after the bounce
    vc = vbz - G * t2                                      # still rising fast at the palm (far from the apex)
    vh_post = d * CATCH["post_speed"]
    b = np.array([c[0] - vh_post[0] * t2, c[1] - vh_post[1] * t2, R])
    tb = fc - CATCH["flight_frames"]
    vh_pre = vh_post / CATCH["keep"]
    vz_pre = -vbz / CATCH["restitution"]
    path = {}
    f = int(math.floor(tb))
    while True:                                            # back from the bounce to off-screen
        t = (f - tb) / 24.0
        p = np.array([b[0] + vh_pre[0] * t, b[1] + vh_pre[1] * t, R + vz_pre * t - 0.5 * G * t * t])
        path[f] = p
        if not cam.visible(p) or f < tb - 60:
            break
        f -= 1
    throw = f
    for f in range(int(math.floor(tb)) + 1, fc):           # up from the bounce
        t = (f - tb) / 24.0
        path[f] = np.array([b[0] + vh_post[0] * t, b[1] + vh_post[1] * t, R + vbz * t - 0.5 * G * t * t])
    path[fc] = c
    m0 = bones[hand + "_matrix"][fi]
    local = np.linalg.inv(m0) @ np.append(c, 1.0)
    for f in range(fc + 1, frames[-1] + 1):                # caught: it moves with the hand
        path[f] = (bones[hand + "_matrix"][frames.index(f)] @ local)[:3]
    palm_gap, part = gap(verts, parts, fi, c)
    speeds = {"in": float(math.hypot(np.linalg.norm(vh_pre), vz_pre)),
              "after_bounce": float(math.hypot(CATCH["post_speed"], vbz)),
              "at_catch": float(math.hypot(CATCH["post_speed"], vc)), "at_catch_up": float(vc)}
    return {"path": path, "throw": throw, "bounce": tb, "catch": fc, "centre": c, "palm_gap": palm_gap,
            "forward": forward, "arm_gap": arm_gap,
            "palm_gap_part": part, "speeds": speeds, "bounce_at": b}


# ---------------------------------------------------------------- Blender objects
def material(name, col, rough=0.55):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = col                                # Workbench (MATERIAL colour) previews
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = col
    bsdf.inputs["Roughness"].default_value = rough
    return mat


def make_ball(name, coll, mat, seam_mat):
    """a ball (radius R, smooth) and its seam ring NAME_Seam (a thin torus on the equator, parented: it spins with
    the ball; it stands at most 2 mm out of the ball's outline)"""
    for n in (name + "_Seam", name):
        if n in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[n], do_unlink=True)
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=R)
    bm.to_mesh(me)
    bm.free()
    for poly in me.polygons:
        poly.use_smooth = True
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    bm = bmesh.new()
    rows, seg, sides, big, small = [], 48, 8, R - 0.004, 0.006
    for i in range(seg):
        a = 2 * math.pi * i / seg
        rows.append([bm.verts.new(((big + small * math.cos(2 * math.pi * j / sides)) * math.cos(a),
                                   (big + small * math.cos(2 * math.pi * j / sides)) * math.sin(a),
                                   small * math.sin(2 * math.pi * j / sides))) for j in range(sides)])
    for i in range(seg):
        for j in range(sides):
            bm.faces.new((rows[i][j], rows[(i + 1) % seg][j], rows[(i + 1) % seg][(j + 1) % sides],
                          rows[i][(j + 1) % sides]))
    sm = bpy.data.meshes.new(name + "_Seam")
    bm.to_mesh(sm)
    bm.free()
    for poly in sm.polygons:
        poly.use_smooth = True
    sm.materials.append(seam_mat)
    seam = bpy.data.objects.new(name + "_Seam", sm)
    coll.objects.link(seam)
    seam.parent = ob
    seam.rotation_euler = (math.radians(70), 0.0, 0.0)     # tilted, so the spin shows
    return ob


def fcurves(ob):
    act = ob.animation_data.action
    try:
        return list(act.fcurves)
    except AttributeError:                                 # Blender 5: layered actions
        from bpy_extras import anim_utils
        return list(anim_utils.action_get_channelbag_for_slot(act, ob.animation_data.action_slot).fcurves)


def key_ball(ob, path, f_first, f_last, park):
    """keys on every frame of the scene: parked before f_first and after f_last (constant), the path between (linear)"""
    sc = bpy.context.scene
    lo, hi = sc.frame_start, sc.frame_end
    keys = {}
    if f_first > lo:
        keys[lo] = park
        keys[f_first - 1] = park
    for f in range(max(lo, f_first), min(hi, f_last) + 1):
        keys[f] = path[f] if f in path else path[max(k for k in path if k <= f)]
    if f_last < hi:
        keys[f_last + 1] = park
        keys[hi] = park
    from mathutils import Quaternion, Vector
    ob.rotation_mode = "QUATERNION"
    q, prev = Quaternion(), None
    for f, p in sorted(keys.items()):
        ob.location = tuple(float(c) for c in p)
        ob.keyframe_insert("location", frame=f)
        if p is not park and prev is not None and prev is not park:    # it spins/rolls with its way along the floor
            dh = Vector((float(p[0] - prev[0]), float(p[1] - prev[1]), 0.0))
            if dh.length > 1e-6:
                q = Quaternion(Vector((0.0, 0.0, 1.0)).cross(dh).normalized(), dh.length / R) @ q
        ob.rotation_quaternion = q
        ob.keyframe_insert("rotation_quaternion", frame=f)
        prev = p
    parked = {f for f, p in keys.items() if p is park} | {min(hi, f_last)}   # no blend into or out of the park
    for fc in fcurves(ob):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT" if int(round(kp.co[0])) in parked else "LINEAR"
    return keys


# ---------------------------------------------------------------- measuring
def in_play(keys, park):
    return sorted(f for f, p in keys.items() if p is not park)


def wall_frame(keys, park):
    """the frame nearest the back-wall bounce: where the ball is deepest, if that is at the wall"""
    play = in_play(keys, park)
    if not WALLS or not play:
        return None
    f = max(play, key=lambda g: keys[g][1])
    return f if keys[f][1] >= WALLS["back"] - R - 0.4 and f != play[-1] else None


def entry_edge(cam, p):
    u, v, _ = cam.uvz(p)
    return ("right" if u > 0 else "left") if abs(u) >= abs(v) else ("top" if v > 0 else "bottom")


def screen_check(keys, park, frames, verts, parts, cam, at, w=1536):
    """in the full-size camera view (w px wide): the frames where the ball's disk overlaps the head's projected
    vertices (the head's silhouette) or any body vertex, and the gap in px between the ball's disk and the head /
    the body at the frames `at`"""
    s = w / cam.w
    head = parts == "head"
    out = {"head_overlap_frames": [], "body_overlap_frames": []}
    for f in in_play(keys, park):
        if f not in frames or not cam.visible(keys[f]):
            continue
        fi = frames.index(f)
        c, rp, _ = cam.px(keys[f])
        c, rp = c * s, rp * s
        hp = np.array([cam.px(v)[0] for v in verts[fi][head]]) * s
        bp = np.array([cam.px(v)[0] for v in verts[fi][::3]]) * s
        dh = float(np.linalg.norm(hp - c, axis=1).min() - rp)
        db = float(np.linalg.norm(bp - c, axis=1).min() - rp)
        if dh < 0:
            out["head_overlap_frames"].append(f)
        if db < 0:
            out["body_overlap_frames"].append(f)
        if f in at:
            out[f"screen_gap_px_head_f{f}"] = round(dh, 1)
            out[f"screen_gap_px_body_f{f}"] = round(db, 1)
    return out


def measure(name, keys, park, frames, verts, parts, cam, rest=None, contact=None):
    play = in_play(keys, park)
    seen = [f for f in play if cam.visible(keys[f])]
    flying = [f for f in play if rest is None or f < rest]
    res = {"name": name, "first_key_off_screen": not cam.visible(keys[play[0]]), "first_visible": seen[0] if seen else None,
           "last_visible_in_flight": max(f for f in seen if f in flying) if seen else None,
           "rest_frame": rest, "exit_frame": None if rest else (play[-1] if not cam.visible(keys[play[-1]]) else None),
           "shared_frames_in_flight": [f for f in flying if f in seen and any(a <= f <= b for a, b in SHARED)]}
    worst, before = (9.0, None, ""), (9.0, None, "")
    pxgap = 9e9
    cover = []
    for f in play:
        if f not in frames:
            continue
        fi = frames.index(f)
        g, part = gap(verts, parts, fi, keys[f])
        if contact is None or f != contact:
            if g < worst[0]:
                worst = (g, f, part)
        if contact is not None and f < contact and g < before[0]:
            before = (g, f, part)
        if cam.visible(keys[f]):                           # on screen: the gap in preview pixels, and covering
            c, rp, z = cam.px(keys[f])
            bp = np.array([cam.px(v)[0] for v in verts[fi][::7]])
            bz = (verts[fi][::7] - cam.pos) @ cam.fwd
            d = np.linalg.norm(bp - c, axis=1)
            j = int(d.argmin())
            pxgap = min(pxgap, float(d[j] - rp))
            if d[j] < rp and z < bz[j]:
                cover.append(f)
    if rest is not None:                                   # a ball at rest: its smallest on-screen gap to the body
        rg = 9e9
        for f in [f for f in play if f >= rest and f in frames][::3]:
            fi = frames.index(f)
            c, rp, z = cam.px(keys[f])
            bp = np.array([cam.px(v)[0] for v in verts[fi][::7]])
            rg = min(rg, float(np.linalg.norm(bp - c, axis=1).min() - rp))
        res["rest_min_screen_gap_px"] = round(rg, 1)
    res["min_gap_m"] = round(worst[0], 3)
    res["min_gap_frame"] = worst[1]
    res["min_gap_part"] = worst[2]
    res["min_gap_centre_m"] = round(worst[0] + R, 3)
    res["min_screen_gap_px"] = round(pxgap, 1)
    res["frames_covering_body"] = cover
    if contact is not None:
        res["min_gap_before_contact_m"] = round(before[0], 3)
        res["min_gap_before_contact_frame"] = before[1]
    return res


def main():
    a = args()
    sc = bpy.context.scene
    rig = bpy.data.objects[a.rig]
    frames, verts, parts, bones = read_body(sc, rig, a.character)
    cam = Camera(sc)
    dashes = find_dashes(frames, bones)
    kicks = find_kicks(frames, verts, parts, bones)
    raised = find_raised_hand(frames, bones, verts, parts)
    deg, fw, tried = choose_wrist(raised, frames, verts, parts, bones, cam)   # the bend that keeps the ball off the arm
    wrist = dict(bend_wrist(sc, rig, raised, deg), forward=fw, tried=tried)           # the palm turns down over the ball
    open_hand(sc, rig, raised)                             # and opens: the palm takes the ball
    frames, verts, parts, bones = read_body(sc, rig, a.character)                # the body with the bent wrist

    global WALLS
    gym = [] if a.no_gym else gym_set.build(SET_BALL, R)
    if gym:
        boxes = []                                         # every set object but the floor, its lines and the walls
        for o in gym:
            if o.name == "Plane" or o.name.startswith(("Line_", "Wall")):     # walls: handled as planes
                continue
            cs = np.array([tuple(o.matrix_world @ v.co) for v in o.data.vertices])
            boxes.append((cs.min(axis=0), cs.max(axis=0)))
        WALLS = {"back": gym_set.BACK_Y - 0.08, "side": gym_set.SIDE_X, "boxes": boxes}
    if len(dashes) != len(DODGE):
        raise SystemExit(f"add_balls: {len(dashes)} dashes found, {len(DODGE)} dodge throws tuned")
    coll = bpy.data.collections.get("Props") or bpy.data.collections.new("Props")
    if coll.name not in sc.collection.children:
        sc.collection.children.link(coll)
    mat = material("Dodgeball_Violet", VIOLET)
    seam_mat = material("Dodgeball_Seam_Magenta", MAGENTA, rough=0.4)
    report = {"dashes": dashes, "kicks": kicks, "balls": [], "radius": R, "params": {"dodge": DODGE, "kick": KICK, "catch": CATCH}}

    for i, d in enumerate(dashes, 1):
        p = dodge_path(d, DODGE[i - 1], frames, verts, parts, bones, cam)
        park = np.array([-d["sign"] * (PARK + 5 * i), PARK, -20.0])   # one park place per ball
        ob = make_ball(f"Ball_Dodge{i}", coll, mat, seam_mat)
        keys = key_ball(ob, p["path"], p["throw"], p["exit"] or frames[-1], park)
        m = measure(ob.name, keys, park, frames, verts, parts, cam, rest=p["rest"])
        cf = int(round(p["cross"]))
        m.update({"dash": d["direction"], "dash_start": d["start"], "aim": p["aim"].round(3).tolist(),
                  "cross_time": p["cross"], "cross_frame": cf, "cross_after_dash_start": p["cross"] - d["start"],
                  "frames_on_screen_before_cross": cf - m["first_visible"] if m["first_visible"] else None,
                  **screen_check(keys, park, frames, verts, parts, cam, [cf, m["min_gap_frame"]]),
                  "throw_frame": p["throw"], "throw_from": p["throw_from"].round(2).tolist(),
                  "aim_to_head_centre_before_dash_m": round(p["would_hit_overlap"], 3),
                  "min_gap_between_frames_m": round(p["sub_gap"], 3),
                  "entry_edge": entry_edge(cam, keys[m["first_visible"]]) if m["first_visible"] else None,
                  "wall_bounce_frame": wall_frame(keys, park),
                  "rest_at": p["path"][p["rest"]].round(2).tolist() if p["rest"] else None,
                  "path_every_4": {f: keys[f].round(2).tolist() for f in in_play(keys, park)[::4]}})
        report["balls"].append(m)
    for i, k in enumerate(kicks, 1):
        p = kick_path(k, frames, verts, parts, cam)
        park = np.array([(PARK + 5 * (i + 3)) * (1 if k["leg"] == "left" else -1), PARK, -20.0])
        ob = make_ball(f"Ball_Kick{i}", coll, mat, seam_mat)
        keys = key_ball(ob, p["path"], p["throw"], p["exit"], park)
        m = measure(ob.name, keys, park, frames, verts, parts, cam, contact=k["peak"])
        m.update({"leg": k["leg"], "contact_frame": k["peak"], "foot_highest_frame": k["highest_frame"],
                  "frames_on_screen_before_contact": k["peak"] - m["first_visible"] if m["first_visible"] else None, "contact": p["contact"].round(3).tolist(),
                  "foot_vertex_to_centre_m": round(p["foot_vertex_dist"], 3),
                  "foot_centre_to_centre_m": round(p["foot_centre_dist"], 3), "throw_frame": p["throw"],
                  "exit_frame": p["exit"], "v_in": p["v_in"].round(2).tolist(), "v_out": p["v_out"].round(2).tolist()})
        report["balls"].append(m)
    hand = dict(find_raised_hand(frames, bones, verts, parts), forward=fw)
    p = catch_path(hand, frames, verts, parts, bones, cam)
    park = np.array([-(PARK + 30), PARK, -20.0])
    ob = make_ball("Ball_Catch", coll, mat, seam_mat)
    keys = key_ball(ob, p["path"], p["throw"], frames[-1], park)
    m = measure(ob.name, keys, park, frames, verts, parts, cam, contact=p["catch"])
    m.update({"hand": hand["hand"], "hand_settle_frame": hand["settle"], "hand_stop_frame": hand["stop"], "catch_frame": p["catch"],
              "bounce_time": round(p["bounce"], 2), "bounce_frame": int(round(p["bounce"])),
              "frames_on_screen_before_bounce": int(round(p["bounce"])) - m["first_visible"] if m["first_visible"] else None,
              "palm_gap_m": round(p["palm_gap"], 4), "palm_gap_part": p["palm_gap_part"],
              "catch_centre": p["centre"].round(3).tolist(), "bounce_at": p["bounce_at"].round(3).tolist(),
              "speeds_m_s": {k: round(v, 2) for k, v in p["speeds"].items()}, "throw_frame": p["throw"],
              **screen_check(keys, park, frames, verts, parts, cam, [p["catch"]])})
    m["grip"] = grip_fingers(sc, rig, a.character, hand, ob, p["catch"])
    m["grip"].update({"forward_m": round(p["forward"], 3), "arm_gap_m": round(p["arm_gap"], 3)})
    m["wrist"] = wrist
    report["balls"].append(m)
    report["raised_hand"] = hand

    names = [b["name"] for b in report["balls"]]
    obs = [bpy.data.objects[n] for n in names]
    closest = (9.0, None, "")
    for f in frames:                                      # the balls never touch each other
        sc.frame_set(f)
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                dd = (obs[i].matrix_world.translation - obs[j].matrix_world.translation).length - 2 * R
                if dd < closest[0]:
                    closest = (round(dd, 3), f, f"{names[i]}-{names[j]}")
    report["closest_ball_ball_gap"] = closest
    # the balls never go into the set (a box check against every set object but the floor, every frame in play)
    boxes = []
    for o in gym:
        if o.name == "Plane" or o.name.startswith("Line_"):
            continue
        cs = [o.matrix_world @ v.co for v in o.data.vertices]
        boxes.append((o.name, [min(c[k] for c in cs) for k in range(3)], [max(c[k] for c in cs) for k in range(3)]))
    set_gap = (9.0, None, "")
    for n, ob in zip(names, obs):
        for f in frames:
            sc.frame_set(f)
            p = ob.matrix_world.translation
            if p.z < 0:                                    # parked under the floor
                continue
            for bn, lo, hi in boxes:
                dd = math.sqrt(sum(max(lo[k] - p[k], 0, p[k] - hi[k]) ** 2 for k in range(3))) - R
                if dd < set_gap[0]:
                    set_gap = (round(dd, 3), f, f"{n}-{bn}")
    report["closest_ball_set_gap"] = set_gap
    report["set_objects"] = [o.name for o in gym]
    bad = [b["name"] for b in report["balls"] if b["name"].startswith("Ball_Dodge")
           and (b["min_gap_m"] <= 0.0 or b["min_gap_between_frames_m"] <= 0.0)]
    bad += [b["name"] for b in report["balls"] if b["name"] == "Ball_Catch"
            and (not 0.0 <= b["palm_gap_m"] <= 0.006 or b["min_gap_before_contact_m"] < 0.0
                 or b["grip"]["min_finger_gap_m"] < 0.0)]
    if closest[0] <= 0.0:
        bad.append(closest[2])
    if set_gap[0] < -0.005:                                # touching a wall in a bounce is fine; going in is not
        bad.append(set_gap[2])
    bad += [b["name"] for b in report["balls"] if b["name"].startswith("Ball_Kick") and b["min_gap_before_contact_m"] < 0.0]
    report["ok"] = not bad
    txt = json.dumps(report, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("BALLS REPORT", txt)
    if a.report:
        with open(a.report, "w") as fh:
            fh.write(txt)
    if bad:
        raise SystemExit(f"add_balls: {bad} touch the body, each other or the set: not saved")
    sc.frame_set(sc.frame_start)
    bpy.ops.wm.save_as_mainfile(filepath=a.out, compress=True)
    print("BALLS SAVED", a.out)


if __name__ == "__main__":
    main()
