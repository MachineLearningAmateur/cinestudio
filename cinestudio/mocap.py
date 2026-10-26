"""Motion from a video for the mannequin: SAM 3D Body's 3D joints -> a motion file for templates/flash_step/build.py.

SAM 3D Body (ComfyUI core nodes; graph workflows/sam3d_body) reads a person's METRIC 3D joints in every frame of a
video and BuildPoseFile writes them as an animated GLB: one root node per person ("trackNN": the person's move relative
to its first frame) with one child node per OpenPose-18 body joint and 2 x 21 hand joints. A joint that moves has a
translation channel (keys at the track's own frame times / fps); a joint that never moves has its translation on the
node and no channel (comfy_extras/sam3d_body/export/glb_openpose.py).

Our graph gives SAM3DBody_Predict no person boxes, so it uses ONE full-frame box: film one dancer. `pick_track` only
matters for a GLB with more tracks (a graph with a person tracker).

`motion_from_glb` turns one track into the motion format that templates/flash_step/build.py reads (the same format as
setup/pose_from_video.py, the older DWPose path): per frame the hip move, unit bone directions, ankle positions, foot
contacts, the pelvis roll and the kick weights.

Why 3D (shan shen bu gauntlet round 14, 2026-10-22): DWPose gives 2D points, so the bone depths were guessed from screen
lengths and fast or crossed limbs were lost; with SAM 3D a blind critic chose the copy clearly (pose 7 vs 5.5,
readability 7.5 vs 5) and the joint acceleration peaks halved."""
from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path

import numpy as np

BODY = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "RKnee", "RAnkle",
        "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"]
HAND = [f"{s}_{k}" for s in ("R", "L") for k in
        ["wrist"] + [f"{f}{i}" for f in ("thumb", "index", "middle", "ring", "pinky") for i in (1, 2, 3, 4)]]
# mannequin bone lengths (blender_lib/mannequin.py REST)
L_THIGH, L_SHIN = 0.43, 0.43
ANKLE_Z = 0.09
STILL_M, LOW_M = 0.018, 0.08                   # a standing foot: moves < 1.8 cm per frame, < 8 cm above the floor
KICK_MIN_PEAK_M, KICK_LEN = 0.55, (4, 30)       # a kick: the foot off the floor, a high peak, a short swing
FLOOR_MIN_SPREAD_M = 0.03                       # the standing ankles must spread this much in BOTH floor directions
BODY_UP_MAX_DEG = 20.0                          # without a floor: a mean body lean up to this is taken as camera tilt


class MocapError(ValueError):
    pass


# ---------------------------------------------------------------- the GLB
def _interp(t_src: np.ndarray, v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation of v[len(t_src), 3] at the times t (held at the ends)."""
    if len(t_src) == 1:
        return np.repeat(v[:1], len(t), 0)
    return np.stack([np.interp(t, t_src, v[:, k]) for k in range(v.shape[1])], 1)


def read_glb(path: str | Path) -> dict:
    """-> {"tracks": [{"name", "t": seconds[n], "joints": {name: xyz[n, 3]}, "root": xyz[n, 3]}]} in the GLB's axes
    (y up, the person faces +z). Each track keeps its own times (a person may appear late or leave); joints without a
    channel take their node translation."""
    d = Path(path).read_bytes()
    if d[:4] != b"glTF" or len(d) < 28:
        raise MocapError(f"not a GLB file: {path}")
    n = struct.unpack_from("<I", d, 12)[0]
    j = json.loads(d[20:20 + n])
    off = 20 + n
    blen = struct.unpack_from("<I", d, off)[0]
    binb = d[off + 8: off + 8 + blen]

    def acc(i):
        a = j["accessors"][i]
        bv = j["bufferViews"][a["bufferView"]]
        k = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[a["type"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        return np.frombuffer(binb, np.float32, a["count"] * k, start).reshape(a["count"], k).astype(np.float64)

    nodes = j.get("nodes") or []
    owner, tracks = {}, {}                               # node index -> its track node index
    for ti, nd in enumerate(nodes):
        if re.fullmatch(r"track\d+", nd.get("name", "")):
            tracks[ti] = {"name": nd["name"], "chan": {}, "const": {}}
            stack = [ti]
            while stack:
                k = stack.pop()
                owner[k] = ti
                stack.extend(nodes[k].get("children") or [])

    def key(k: int) -> str | None:                       # "__root__", a joint name, or None (limbs, meshes)
        name = nodes[k].get("name", "")
        if k in tracks:
            return "__root__"
        if name.startswith("openpose_") and "limb" not in name:
            return name[len("openpose_"):]
        return None

    for k, ti in owner.items():                          # constant joints (and a still root) live on the nodes
        kk = key(k)
        if kk:
            tracks[ti]["const"][kk] = np.array(nodes[k].get("translation") or [0.0, 0.0, 0.0], float)
    for an in j.get("animations") or []:
        for c in an.get("channels") or []:
            k = c["target"].get("node")
            if k is None or c["target"].get("path") != "translation" or k not in owner or not key(k):
                continue
            s = an["samplers"][c["sampler"]]
            tracks[owner[k]]["chan"][key(k)] = (acc(s["input"])[:, 0], acc(s["output"]))
    out = []
    for tr in tracks.values():
        names = set(tr["chan"]) | set(tr["const"])
        if not all(b in names for b in BODY):
            continue
        times = [tt for tt, _ in tr["chan"].values()]
        t = np.unique(np.concatenate(times)) if times else np.zeros(1)
        val = lambda kk: (_interp(*tr["chan"][kk], t) if kk in tr["chan"]          # noqa: E731
                          else np.tile(tr["const"].get(kk, np.zeros(3)), (len(t), 1)))
        out.append({"name": tr["name"], "t": t, "joints": {kk: val(kk) for kk in names if kk != "__root__"},
                    "root": val("__root__")})
    if not out:
        raise MocapError("the GLB has no complete OpenPose body track (no person found in the video?)")
    return {"tracks": sorted(out, key=lambda x: x["name"])}


# ---------------------------------------------------------------- helpers
def to_blender(v: np.ndarray) -> np.ndarray:
    """GLB (x right of the viewer, y up, z toward the viewer) -> Blender (x, y away from the camera, z up)."""
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], -1)


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(1e-9, float(np.linalg.norm(v)))


def _sg(x: np.ndarray, win: int = 7) -> np.ndarray:
    """Savitzky-Golay (quadratic) of a 1D signal, edges padded with the end values. `win` must be odd."""
    k = win // 2
    A = np.vander(np.arange(-k, k + 1), 3, increasing=True)
    w = np.linalg.pinv(A)[0]
    pad = np.concatenate([np.repeat(x[:1], k), x, np.repeat(x[-1:], k)])
    return np.array([float(np.dot(w, pad[i:i + win])) for i in range(len(x))])


def sg(a: np.ndarray, win: int) -> np.ndarray:
    """Savitzky-Golay along axis 0 of an array of any shape (an even window is made odd; < 3 = no smoothing)."""
    win = int(win) | 1
    if win < 3 or len(a) < win:
        return a
    flat = a.reshape(len(a), -1)
    return np.stack([_sg(flat[:, k], win) for k in range(flat.shape[1])], 1).reshape(a.shape)


def _align(v: np.ndarray) -> np.ndarray:
    """The rotation that turns the unit vector v onto +Z (Rodrigues)."""
    z = np.array([0.0, 0.0, 1.0])
    ax = np.cross(v, z)
    s, c = np.linalg.norm(ax), float(np.dot(v, z))
    if s < 1e-9:
        return np.eye(3)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def _plane_visible(A: np.ndarray) -> bool:
    """Floor points that define a plane: at least 8, spread FLOOR_MIN_SPREAD_M (std) in both floor directions."""
    if len(A) < 8:
        return False
    return float((np.linalg.svd(A[:, :2] - A[:, :2].mean(0), compute_uv=False) / np.sqrt(len(A))).min()) \
        >= FLOOR_MIN_SPREAD_M


def floor_rotation(P: np.ndarray) -> tuple[np.ndarray, str]:
    """The rotation that makes the floor level: it removes the tilt of the filming camera (a phone held low and looking
    up: 15.5 deg in the first video). The floor plane is fitted to the lower ankle of each frame, outliers dropped in
    up to 4 passes; EVERY pass's points must spread in both floor directions (outliers can give the second direction and
    then go: Codex review). A dancer who stands still or moves on one line gives no plane: then the body's mean up
    direction (hips -> neck, over the whole clip) is made vertical when it is within BODY_UP_MAX_DEG (a camera tilt);
    a larger mean lean is more likely the dance itself (a held bow), and nothing is turned.
    -> (rotation, "floor" | "body" | "none")."""
    J = {n: i for i, n in enumerate(BODY)}
    ia, ib = J["RAnkle"], J["LAnkle"]
    A = np.where((P[:, ia, 2] < P[:, ib, 2])[:, None], P[:, ia], P[:, ib])
    keep, coef = np.ones(len(A), bool), None
    for _ in range(4):
        if not _plane_visible(A[keep]):
            coef = None
            break
        M = np.c_[A[keep, 0], A[keep, 1], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(M, A[keep, 2], rcond=None)
        res = A[:, 2] - (A[:, 0] * coef[0] + A[:, 1] * coef[1] + coef[2])
        new = res < max(0.03, float(np.percentile(res, 60)))
        if (new == keep).all():
            break
        keep = new
    if coef is not None:
        return _align(unit(np.array([-coef[0], -coef[1], 1.0]))), "floor"
    up = unit((P[:, J["Neck"]] - (P[:, J["RHip"]] + P[:, J["LHip"]]) / 2).mean(0))
    if np.degrees(np.arccos(np.clip(up[2], -1, 1))) <= BODY_UP_MAX_DEG:
        return _align(up), "body"
    return np.eye(3), "none"


def pelvis_roll(hl: np.ndarray) -> float:
    """The tilt of the hip line (left hip minus right hip) in degrees, + = the left hip up, as the front-facing
    mannequin shows it: the line's height angle, signed by the side of the picture its left end is on (blended near a
    side view, where the tilt is not visible). A plain atan2(z, x) jumped 0 -> 180 deg when the dancer turned past 90."""
    n = float(np.linalg.norm(hl))
    if n < 1e-9:
        return 0.0
    side = float(np.clip(hl[0] / (0.3 * n), -1.0, 1.0))
    return float(np.degrees(np.arctan2(hl[2], np.hypot(hl[0], hl[1])))) * side


def hysteresis(frames: list, enter: int = 2, leave: int = 2, min_len: int = 3) -> None:
    """A contact starts after `enter` still frames and ends after `leave` moving frames; a contact shorter than
    `min_len` frames is dropped (one-frame flicker re-pinned a foot at a new place). A contact at frame 1 counts from
    frame 1 (a late start snapped the foot to its pin)."""
    for side in (".R", ".L"):
        raw = [f["contact"][side] for f in frames]
        out, on, run = [], bool(raw[0]) if raw else False, 0
        for r in raw:
            run = run + 1 if r != on else 0
            if (not on and run >= enter) or (on and run >= leave):
                on, run = not on, 0
            out.append(on)
        i = 0
        while i < len(out):
            if out[i]:
                j = i
                while j < len(out) and out[j]:
                    j += 1
                if j - i < min_len:
                    out[i:j] = [False] * (j - i)
                i = j
            else:
                i += 1
        for f, c in zip(frames, out):
            f["contact"][side] = c


def kicks(frames: list) -> list:
    """Kick segments (no contact, foot above 0.2 m, peak >= KICK_MIN_PEAK_M, KICK_LEN frames, not touching the ends)
    get a weight per frame: 0 -> 1 over the first third, 1 in the middle third, back to 0 (a short kick is widened by 3
    frames on each side so the leg can straighten smoothly). The builder straightens the leg by it."""
    T = len(frames)
    for f in frames:
        f["kick"] = {".R": 0.0, ".L": 0.0}
    found = []
    for side in (".R", ".L"):
        z = np.array([f["feet"][side][2] for f in frames])
        off = np.array([(not f["contact"][side]) and f["feet"][side][2] > 0.2 for f in frames])
        i = 0
        while i < T:
            if not off[i]:
                i += 1
                continue
            j = i
            while j < T and off[j]:
                j += 1
            if KICK_LEN[0] <= j - i <= KICK_LEN[1] and z[i:j].max() >= KICK_MIN_PEAK_M and 0 < i and j < T:
                a, b = i, j
                if b - a < 12:
                    a, b = max(0, a - 3), min(T, b + 3)
                n = b - a
                for t in range(a, b):
                    u = (t - a + 0.5) / n
                    w = min(1.0, u * 3, (1 - u) * 3)
                    w = w * w * (3 - 2 * w)                 # smoothstep
                    frames[t]["kick"][side] = round(max(frames[t]["kick"][side], w), 3)
                found.append({"side": side, "frames": [a + 1, b], "peak_z": round(float(z[i:j].max()), 2)})
            i = j
    return found


def accent(frames: list, side: str, first: int, last: int, lift: float = 1.0, thigh_lift: float = 1.0,
           turn_deg: float = 0.0, max_deg: float = 110.0, ease: float = 0.33) -> int:
    """Emphasize one leg move in place (shan shen bu round 15, user: "the second kick on the tiqiu part needs to be more
    emphasized"): over the 1-based frames first..last (eased in and out over the outer `ease` share), the shin's angle from
    straight down is multiplied by `lift` and the thigh's by `thigh_lift` (at most `max_deg`), and the shin's heading
    turns up to `turn_deg` toward the leg's own side (.L: +X, the picture's right). A flick that goes BACK, behind the
    knee, hides behind the leg from a front camera; turned to the side it reads like the first kick. -> frames changed."""
    if side not in (".R", ".L") or last < first:
        raise MocapError("accent: side .R or .L and first <= last")
    target = 0.0 if side == ".L" else math.pi
    n, changed = last - first + 1, 0

    def rebuild(v, scale, turn):
        a = math.acos(max(-1.0, min(1.0, -v[2])))                    # the angle from straight down
        h = math.atan2(v[1], v[0])
        if turn:
            dh = (target - h + math.pi) % (2 * math.pi) - math.pi
            h += max(-turn, min(turn, dh))
        a = min(a * scale, math.radians(max_deg))
        return [round(math.sin(a) * math.cos(h), 4), round(math.sin(a) * math.sin(h), 4), round(-math.cos(a), 4)]

    for t in range(max(1, first), min(len(frames), last) + 1):
        u = (t - first + 0.5) / n
        w = min(1.0, u / ease, (1 - u) / ease)
        w = w * w * (3 - 2 * w)                                          # smoothstep in the outer `ease` share
        if w <= 0:
            continue
        d = frames[t - 1]["dirs"]
        d["thigh" + side] = rebuild(d["thigh" + side], 1 + (thigh_lift - 1) * w, 0.0)
        d["shin" + side] = rebuild(d["shin" + side], 1 + (lift - 1) * w, math.radians(turn_deg) * w)
        frames[t - 1].setdefault("accent", {})[side] = round(w, 3)
        changed += 1
    return changed


def _mirror(v: list) -> list:
    """A direction or place seen in a mirror at the body's middle (x -> -x)."""
    return [-v[0], v[1], v[2]]


def mirror_legs(frames: list, src_first: int, src_last: int, dst_first: int, blend: int = 4) -> int:
    """Copy the LEGS of frames src_first..src_last (1-based) to dst_first.., left and right swapped and mirrored (user,
    shan shen bu round 16: "mirror the kick towards the right with the kick towards the left"): the thighs, shins,
    feet, contacts, kick weights and kick lines; the pelvis roll is mirrored too. The arms, torso and hip path of the
    destination stay. The copy fades in and out over `blend` frames at each end (directions mixed, contacts and kicks
    switched at the middle of the fade). -> frames changed."""
    n = src_last - src_first + 1
    if n < 2 or src_first < 1 or src_last > len(frames) or dst_first < 1 or dst_first + n - 1 > len(frames):
        raise MocapError("mirror_legs: both windows must lie inside the motion")
    if not (dst_first > src_last or dst_first + n - 1 < src_first):
        raise MocapError("mirror_legs: the windows must not overlap")
    other = {".R": ".L", ".L": ".R"}
    for k in range(n):
        s, d = frames[src_first - 1 + k], frames[dst_first - 1 + k]
        w = min(1.0, (k + 0.5) / max(1, blend), (n - k - 0.5) / max(1, blend))
        w = w * w * (3 - 2 * w)
        for side in (".R", ".L"):
            o = other[side]
            for bone in ("thigh", "shin"):
                v = np.array(_mirror(s["dirs"][bone + o])) * w + np.array(d["dirs"][bone + side]) * (1 - w)
                d["dirs"][bone + side] = unit(v).round(4).tolist()
        # the kick weights fade with the copy (switched at once, the leg snapped straight: an ankle spike 2.6x the
        # clip's maximum); the kick line is the copy's where the copy kicks
        kd_src = {other[side]: _mirror(v) for side, v in (s.get("kick_dir") or {}).items()}
        kick = {side: round(s["kick"][other[side]] * w + (d.get("kick") or {}).get(side, 0.0) * (1 - w), 3)
                for side in (".R", ".L")}
        kd = {side: kd_src[side] if side in kd_src and s["kick"][other[side]] > 0 else v
              for side, v in {**(d.get("kick_dir") or {}), **kd_src}.items()}
        d["kick"] = kick
        d.pop("kick_dir", None)
        if kd:
            d["kick_dir"] = kd
        if w >= 0.5:
            d["contact"] = {side: s["contact"][other[side]] for side in (".R", ".L")}
            d["feet"] = {side: _mirror(s["feet"][other[side]]) for side in (".R", ".L")}
        d["pelvis_roll"] = round(-s["pelvis_roll"] * w + d["pelvis_roll"] * (1 - w), 2)
        d["mirrored_from"] = src_first + k
    return n


# the mannequin's REST pose as bone directions (blender_lib/mannequin.py _rest(): tail - head, normalized; that module
# needs bpy): the body upright, the arms down a little away from the body, the legs straight down (the sheets' pose)
REST_DIRS = {"spine": [0.0, 0.0, 1.0], "chest": [0.0, 0.0, 1.0], "head": [0.0, 0.0, 1.0],
             "shoulder.L": [0.998, 0.0624, 0.0], "shoulder.R": [-0.998, 0.0624, 0.0],
             "upper_arm.L": [0.2169, 0.0, -0.9762], "upper_arm.R": [-0.2169, 0.0, -0.9762],
             "forearm.L": [0.0795, -0.0795, -0.9937], "forearm.R": [-0.0795, -0.0795, -0.9937],
             "hand.L": [0.099, -0.099, -0.9901], "hand.R": [-0.099, -0.099, -0.9901],
             "thigh.L": [0.0233, 0.0, -0.9997], "thigh.R": [-0.0233, 0.0, -0.9997],
             "shin.L": [0.0, 0.0233, -0.9997], "shin.R": [0.0, 0.0233, -0.9997]}


# an arms-crossed finishing pose (user, 2026-10-23: "an arms crossed flair type of look with them angled at a side"),
# in the body's own frame (the mannequin faces -Y, +X = its left): the upper arms down, a little forward and in, the
# forearms across the lower chest (the left one in front), the hands on the other upper arm
# v2 (user: "what happened to the hands.."): the first hands pointed up and back INTO the chest (a flat blade across it,
# fingers in the body); now the forearms cross further forward (the left one above), each hand on the outside of the
# other upper arm, a little down, its fingers curled (grip)
CROSSED_ARMS = {"upper_arm.L": [-0.08, -0.5, -0.86], "upper_arm.R": [0.08, -0.55, -0.83],
                "forearm.L": [-0.93, -0.3, 0.2], "forearm.R": [0.92, -0.35, 0.12],
                "hand.L": [-0.72, 0.6, -0.35], "hand.R": [0.72, 0.6, -0.35]}
UPPER_BODY = ("spine", "chest", "head", "shoulder.L", "shoulder.R", "upper_arm.L", "upper_arm.R", "forearm.L",
              "forearm.R", "hand.L", "hand.R")
# a quirky "ta-da" finish for one dancer (user, 2026-10-24: the single shan shen bu "doesnt do the arm pose, you can
# decide on what pose it does make it quirky"): the right arm thrown up and out in a high diagonal, the left hand on
# the hip with the elbow out (body frame: faces -Y, +X = its left)
TADA_ARMS = {"upper_arm.R": [-0.55, -0.12, 0.83], "forearm.R": [-0.42, -0.18, 0.89], "hand.R": [-0.36, -0.22, 0.91],
             "upper_arm.L": [0.62, 0.22, -0.75], "forearm.L": [-0.72, -0.12, -0.68], "hand.L": [-0.45, -0.05, -0.89]}


def _yawed(v, deg: float) -> list:
    """A direction turned about the vertical (Z) by `deg` degrees (negative = toward the picture's left: the body faces
    -Y, so its facing vector becomes (sin a, -cos a))."""
    a = math.radians(deg)
    x, y, z = v
    return [round(x * math.cos(a) - y * math.sin(a), 4), round(x * math.sin(a) + y * math.cos(a), 4), z]


def _leaned(v, deg: float) -> list:
    """A direction tilted sideways in the picture plane (about the camera's Y axis): positive = the top toward +X (the
    picture's right)."""
    a = math.radians(deg)
    x, y, z = v
    return [round(x * math.cos(a) + z * math.sin(a), 4), y, round(-x * math.sin(a) + z * math.cos(a), 4)]


LEG_BONE_M = 0.43                                  # the mannequin's thigh and shin (blender_lib/mannequin.py REST)


def _backed(v, deg: float) -> list:
    """A direction tilted backward about the body's sideways (X) axis: positive = the top toward +Y (behind; the body
    faces -Y)."""
    a = math.radians(deg)
    x, y, z = v
    return [x, round(y * math.cos(a) + z * math.sin(a), 4), round(-y * math.sin(a) + z * math.cos(a), 4)]


def pose_frame(like: dict, arms: str = "rest", yaw_deg: float = 0.0, lean_deg: float = 0.0, hip_shift: float = 0.0,
               knee_deg: float = 0.0, back_deg: float = 0.0, chin_deg: float = 0.0, hips_fwd: float = 0.0,
               head_tilt_deg: float = 0.0, knee_pop: str | None = None, knee_pop_deg: float = 22.0) -> dict:
    """A finishing pose: the rest pose (arms "rest") or crossed arms ("crossed", with a finger grip), the upper body
    tilted sideways by `lean_deg` (positive = toward the picture's right; user: "they should lean their upper body more
    towards the inside"), then the whole body turned by `yaw_deg` (the hips turn in templates/flash_step/build.py;
    every direction here is turned with it). `hip_shift` = the hips' sideways move in m (+ = the picture's right): a
    strong lean looks balanced with the hips the other way (user: "lean more into it towards the middle"). `knee_deg` =
    soft knees (user: "have their thighs bend slightly backwards too ... a lean into it type of thing"): each thigh
    slants forward to the knee and each shin back by the same angle, so the foot stays under its hip, and the hips sink
    by exactly what the bent legs lose. A b-boy finish (user: "i just think of bboys when i hear the lean into it arms
    crossed" -> lean back + toward the middle): `back_deg` = the upper body leans back (chest out), `chin_deg` = the
    head a further tilt back (chin up), `hips_fwd` = the hips forward in m along the body's facing; the legs slant so
    the feet stay where they were (the shin takes the difference). Arms "tada" = TADA_ARMS (one arm up, one hand on
    the hip); `head_tilt_deg` = the head alone tilted sideways after the lean (+ = toward the picture's right);
    `knee_pop` ".L"/".R" = that knee bent forward and inward by `knee_pop_deg` with its foot off the floor (the other
    leg straight carries the hips; the popped foot has no contact, so the hold does not pin it)."""
    f = rest_frame(like)
    a = math.radians(knee_deg)
    sc = max(-0.95, min(0.95, math.sin(a) + float(hips_fwd) / LEG_BONE_M))   # the foot `hips_fwd` behind the hip
    c = math.asin(sc)
    drop = LEG_BONE_M * (2 - math.cos(a) - math.cos(c)) if (knee_deg or hips_fwd) else 0.0
    f["hips"] = [float(hip_shift), round(-drop, 4)]
    if hips_fwd:
        f["hips_fwd"] = float(hips_fwd)
    if knee_deg or hips_fwd:
        for s, x in ((".L", 0.0233), (".R", -0.0233)):
            f["dirs"]["thigh" + s] = [x, round(-math.sin(a), 4), round(-math.cos(a), 4)]     # the knee forward (-Y)
            f["dirs"]["shin" + s] = [0.0, round(math.sin(c), 4), round(-math.cos(c), 4)]     # the foot back
    if arms == "crossed":
        f["dirs"].update({k: list(v) for k, v in CROSSED_ARMS.items() if k in f["dirs"]})
        f["grip"] = 1.0
    elif arms == "tada":
        f["dirs"].update({k: list(unit(np.array(v)).round(4)) for k, v in TADA_ARMS.items() if k in f["dirs"]})
    elif arms != "rest":
        raise MocapError(f"unknown arms pose {arms!r}")
    if back_deg:
        f["dirs"].update({k: _backed(v, back_deg) for k, v in f["dirs"].items() if k in UPPER_BODY})
    if chin_deg:
        f["dirs"]["head"] = _backed(f["dirs"]["head"], chin_deg)
    if lean_deg:
        f["dirs"].update({k: _leaned(v, lean_deg) for k, v in f["dirs"].items() if k in UPPER_BODY})
    if head_tilt_deg:
        f["dirs"]["head"] = _leaned(f["dirs"]["head"], head_tilt_deg)
    if knee_pop:
        if knee_pop not in (".L", ".R") or knee_deg or hips_fwd:
            raise MocapError("knee_pop is '.L' or '.R', without knee_deg or hips_fwd")
        p = math.radians(knee_pop_deg)
        inward = -1.0 if knee_pop == ".L" else 1.0                         # .L is +X: inward = -X
        thigh = unit(np.array([inward * 0.8 * math.sin(p), -math.sin(p), -math.cos(p)]))   # inward: it shows from the front
        shin = unit(np.array([0.0, math.sin(p), -math.cos(p)]))
        rest = np.array(f["dirs"]["thigh" + knee_pop]) + np.array(f["dirs"]["shin" + knee_pop])
        f["dirs"]["thigh" + knee_pop] = thigh.round(4).tolist()
        f["dirs"]["shin" + knee_pop] = shin.round(4).tolist()
        move = LEG_BONE_M * (thigh + shin - rest)                           # where the ankle goes from its rest place
        foot = f["feet"][knee_pop]
        f["feet"][knee_pop] = [round(float(foot[i] + move[i]), 4) for i in range(3)]
        f["contact"][knee_pop] = False
    if yaw_deg:
        f["dirs"] = {k: _yawed(v, yaw_deg) for k, v in f["dirs"].items()}
        f["feet"] = {s: _yawed(v[:2] + [v[2]], yaw_deg) for s, v in f["feet"].items()}
    f["yaw"] = float(yaw_deg)
    return f


def rest_frame(like: dict) -> dict:
    """A motion frame in the mannequin's rest pose (hips centred, both feet down), with the keys of `like`."""
    feet = like.get("feet") or {}
    y = float(np.mean([v[1] for v in feet.values()])) if feet else 0.0
    return {"dirs": {k: list(REST_DIRS[k]) for k in (like.get("dirs") or REST_DIRS) if k in REST_DIRS},
            "hips": [0.0, 0.0], "pelvis_roll": 0.0, "contact": {".R": True, ".L": True},
            "feet": {".R": [-0.11, y, 0.09], ".L": [0.11, y, 0.09]}, "kick": {".R": 0.0, ".L": 0.0}}


def settle(frames: list, n: int = 16, target: int | dict = 1, near_floor: float = 0.03, hold: int = 0,
           free_feet: bool = False) -> int:
    """Ease the last `n` frames into the pose of frame `target` (1-based; frame 1 = the dancer's first, standing pose),
    so a template cut from a video ends on a stand and not in the middle of a step (2026-10-23, user: "what's with the
    pose at the end of the renders": the shan shen bu window ended on one leg). Weights rise smoothly (smoothstep) and
    reach 1 on the last frame. The directions, hips, pelvis roll, feet and kick weights are mixed; a foot takes the
    target's contact only when its mixed height is within `near_floor` m of the target's (a lifted foot pins after it
    comes down, so the pin does not hold it in the air). `target` = a frame number, or a frame (rest_frame(): the
    mannequin's rest pose; user: "they should just transition into their rest position"); `hold` = the last frames
    stay exactly on the target (a still end: without one, LTX carried the fast steps on to the last frame);
    `free_feet` = no contact during the ease, so the feet follow the legs to the target's places (pinned feet stayed
    crossed where the ease began), and both feet down in the hold. -> frames changed."""
    n, hold = int(n), max(0, int(hold))
    if n <= 0:
        return 0
    if isinstance(target, dict):
        t = target
    else:
        if target < 1 or target > len(frames):
            raise MocapError("settle: the target must lie inside the motion")
        t = frames[target - 1]
    if n + hold >= len(frames):
        raise MocapError("settle: the ease and the hold must be shorter than the motion")
    first = len(frames) - n - hold
    for k in range(n + hold):
        d = frames[first + k]
        w = min(1.0, (k + 1) / n)
        w = w * w * (3 - 2 * w)
        for bone, v in t["dirs"].items():
            if bone in d["dirs"]:
                d["dirs"][bone] = unit(np.array(v) * w + np.array(d["dirs"][bone]) * (1 - w)).round(4).tolist()
        d["hips"] = [round(a * w + b * (1 - w), 4) for a, b in zip(t["hips"], d["hips"])]
        if t.get("pelvis_roll") is not None and d.get("pelvis_roll") is not None:
            d["pelvis_roll"] = round(t["pelvis_roll"] * w + d["pelvis_roll"] * (1 - w), 2)
        if t.get("hips_fwd") or d.get("hips_fwd"):         # the hips forward along the facing (pose_frame)
            d["hips_fwd"] = round(float(t.get("hips_fwd") or 0.0) * w + float(d.get("hips_fwd") or 0.0) * (1 - w), 4)
        if t.get("grip") or d.get("grip"):                 # the fingers' grip of the crossed arms (pose_frame)
            d["grip"] = round(float(t.get("grip") or 0.0) * w + float(d.get("grip") or 0.0) * (1 - w), 3)
        if t.get("yaw") or d.get("yaw"):                   # the body's turn about the vertical (pose_frame)
            d["yaw"] = round(float(t.get("yaw") or 0.0) * w + float(d.get("yaw") or 0.0) * (1 - w), 2)
        contact = dict(d.get("contact") or {})
        for side in (".R", ".L"):
            if side in (t.get("feet") or {}) and side in (d.get("feet") or {}):
                d["feet"][side] = [round(a * w + b * (1 - w), 4) for a, b in zip(t["feet"][side], d["feet"][side])]
                if (t.get("contact") or {}).get(side) and d["feet"][side][2] - t["feet"][side][2] <= near_floor:
                    contact[side] = True
                elif w >= 0.999:
                    contact[side] = bool((t.get("contact") or {}).get(side))
            kick = d.get("kick") or {}
            if side in kick:
                kick[side] = round(kick[side] * (1 - w) + (t.get("kick") or {}).get(side, 0.0) * w, 3)
        if free_feet:                                      # both feet down in the hold, unless the target lifts one
            contact = {s: w >= 0.999 and bool((t.get("contact") or {}).get(s, True)) for s in (".R", ".L")}
        d["contact"] = contact
        d["settle"] = round(w, 3)
    return n + hold


# ---------------------------------------------------------------- the conversion
def _frames(P: np.ndarray, H: np.ndarray, min_feet: float) -> tuple[list, dict]:
    """P[frames, 18, 3], H[frames, 42, 3] in Blender axes (metres) -> motion frames + the scale facts."""
    J = {n: i for i, n in enumerate(BODY)}
    g = lambda t, n: P[t, J[n]]                                          # noqa: E731
    thigh = float(np.median(np.linalg.norm(P[:, J["RKnee"]] - P[:, J["RHip"]], axis=1)))
    shin = float(np.median(np.linalg.norm(P[:, J["RAnkle"]] - P[:, J["RKnee"]], axis=1)))
    s = (L_THIGH + L_SHIN) / max(1e-3, thigh + shin)                     # dancer metres -> mannequin metres
    mid_hip = (P[:, J["RHip"]] + P[:, J["LHip"]]) / 2
    ank_rest = float(np.percentile(np.minimum(P[:, J["RAnkle"], 2], P[:, J["LAnkle"], 2]), 20))   # a standing ankle
    x_ref = float(np.median(mid_hip[:, 0]))
    hip_z = (mid_hip[:, 2] - ank_rest) * s + ANKLE_Z
    stand = float(np.percentile(hip_z, 75))
    T = len(P)
    frames = []
    r4 = lambda v: unit(v).round(4).tolist()                             # noqa: E731
    for t in range(T):
        d = {}
        sp = unit(g(t, "Neck") - mid_hip[t])
        d["spine"] = d["chest"] = sp.round(4).tolist()
        ears = (g(t, "REar") + g(t, "LEar")) / 2
        h = unit(ears + (ears - g(t, "Neck")) * 0.5 - g(t, "Neck"))
        d["head"] = r4(sp + np.clip(h - sp, -0.35, 0.35))                # the head stays near the body's line
        for side, pre, hoff in ((".R", "R", 0), (".L", "L", 21)):
            d["shoulder" + side] = r4(g(t, pre + "Shoulder") - g(t, "Neck"))
            d["upper_arm" + side] = r4(g(t, pre + "Elbow") - g(t, pre + "Shoulder"))
            d["forearm" + side] = r4(g(t, pre + "Wrist") - g(t, pre + "Elbow"))
            d["hand" + side] = r4(H[t, hoff + 9] - H[t, hoff])          # wrist -> middle knuckle
            d["thigh" + side] = r4(g(t, pre + "Knee") - g(t, pre + "Hip"))
            d["shin" + side] = r4(g(t, pre + "Ankle") - g(t, pre + "Knee"))
        feet, contact = {}, {}
        for side, pre in ((".R", "R"), (".L", "L")):
            k = J[pre + "Ankle"]
            an = P[t, k]
            feet[side] = [round(float((an[0] - x_ref) * s), 4), round(float(an[1] * s), 4),
                          round(float(max(ANKLE_Z, (an[2] - ank_rest) * s + ANKLE_Z)), 4)]
            v = (P[min(t + 1, T - 1), k] - P[max(t - 1, 0), k]) / 2       # the speed without depth (the noisy axis)
            contact[side] = bool(np.hypot(v[0], v[2]) * s < STILL_M and (an[2] - ank_rest) * s < LOW_M)
        if min_feet > 0:                                                 # optional no-cross clamp (both feet down)
            gap = feet[".L"][0] - feet[".R"][0]
            if gap < min_feet and all(feet[x][2] - ANKLE_Z < LOW_M for x in feet):
                push = (min_feet - gap) / 2
                feet[".L"][0] += push
                feet[".R"][0] -= push
        frames.append({"pelvis_roll": round(pelvis_roll(g(t, "LHip") - g(t, "RHip")), 2), "contact": contact,
                       "hips": [round(float((mid_hip[t, 0] - x_ref) * s), 4),
                                round(float(np.clip(hip_z[t] - stand, -0.30, 0.05)), 4)],
                       "dirs": d, "feet": feet})
    hysteresis(frames)
    found = kicks(frames)
    for t, f in enumerate(frames):                                       # the kick line = the dancer's 3D hip -> ankle
        for side, pre in ((".R", "R"), (".L", "L")):
            if f["kick"][side] > 0:
                f.setdefault("kick_dir", {})[side] = r4(g(t, pre + "Ankle") - g(t, pre + "Hip"))
    return frames, {"scale_dancer_to_mannequin": round(s, 4), "dancer_leg_m": round(thigh + shin, 3),
                    "ankle_rest_m": round(ank_rest, 3), "kicks": found}


def pick_track(tracks: list, index: int | None = None) -> int:
    """The person to copy: `index`, else the one that moves the most in the picture's plane (per second)."""
    if index is not None:
        if not 0 <= index < len(tracks):
            raise MocapError(f"track {index} does not exist: the GLB has {len(tracks)} person track(s)")
        return index

    def energy(tr):
        P = np.stack([tr["joints"][b] for b in BODY], 1) + tr["root"][:, None]
        span = max(1e-3, float(tr["t"][-1] - tr["t"][0]))
        return float(np.abs(np.diff(P[..., :2], axis=0)).sum()) / span
    return int(np.argmax([energy(tr) for tr in tracks]))


def motion_from_glb(glb: str | Path, out: str | Path | None = None, start: float = 0.0, seconds: float | None = None,
                    fps: float = 24.0, track: int | None = None, smooth: int = 5, legs_smooth: int = 9,
                    min_feet: float = 0.0, root_depth: bool = False, keep: np.ndarray | None = None,
                    rotation: np.ndarray | None = None, on_ray: bool = False) -> dict:
    """SAM 3D Body GLB -> the motion record (written to `out` as JSON when given). `start`/`seconds` choose the window
    in the GLB's time (seconds); a window that begins before the person appears starts where it appears. The root's
    DEPTH move is dropped unless `root_depth` (depth from one camera is noisy).
    `keep` = one flag per video frame (frame i at i / fps): only the GLB samples of kept frames are used, the others
    are filled by interpolation (several people: in a frame where SAM 3 lost a person, SAM 3D Body still predicted
    something from a full-frame box). "place_m" = the person's median floor place over the window in metres (x = the
    picture's right, y = away from the camera; meaningful for a GLB with ABSOLUTE roots: workflows/sam3d_body_multi).
    For such a GLB: `on_ray` = each root slides along its camera ray to the person's median depth (ray_root) instead of
    losing its depth, and `rotation` = the floor shared by all people (shared_floor) instead of one fitted to this
    person alone."""
    finite = all(math.isfinite(float(x)) for x in (fps, start) + (() if seconds is None else (seconds,)))
    if not finite or fps <= 0 or start < 0 or (seconds is not None and seconds <= 0):
        raise MocapError("fps and seconds must be positive numbers and start not negative")
    tracks = read_glb(glb)["tracks"]
    ti = pick_track(tracks, track)
    tr = tracks[ti]
    if keep is not None:
        tr = _kept(tr, np.asarray(keep, bool), fps)
    t_src = tr["t"]
    root = to_blender(ray_root(tr["root"]) if on_ray else tr["root"])
    Pj = to_blender(np.stack([tr["joints"][n] for n in BODY], 1))
    hips_full = (Pj[:, BODY.index("RHip")] + Pj[:, BODY.index("LHip")]) / 2 + root     # with the root's depth
    hand = [tr["joints"].get(n) for n in HAND]
    if any(h is None for h in hand):                                     # no hand points: the hand follows the forearm
        wr = {"R": tr["joints"]["RWrist"], "L": tr["joints"]["LWrist"]}
        el = {"R": tr["joints"]["RElbow"], "L": tr["joints"]["LElbow"]}
        hand = [wr[n[0]] + (wr[n[0]] - el[n[0]]) * (0.0 if n.endswith("wrist") else 0.4) for n in HAND]
    Hj = to_blender(np.stack(hand, 1))
    # the root's depth move is dropped BEFORE the floor fit: depth from one camera is noisy, and a fit with it tilted the
    # first video's dancer back by ~4 deg (11.5 deg found against 15.5; the critic-checked round 14 fits without it)
    if not root_depth and not on_ray:                    # (on_ray: the depth is one constant per person already)
        root[:, 1] = 0.0
    if rotation is not None:                             # one floor for several people (shared_floor)
        R, floor_from = np.asarray(rotation, float), "shared"
    else:
        R, floor_from = floor_rotation(Pj + root[:, None])
    P, H = (Pj + root[:, None]) @ R.T, (Hj + root[:, None]) @ R.T
    hips_full = hips_full @ R.T                                          # the floor place (camera tilt removed)
    if on_ray:                                           # the person's own depth out of the motion (its place keeps it)
        dy = float(np.median((P[:, BODY.index("RHip"), 1] + P[:, BODY.index("LHip"), 1]) / 2))
        P[..., 1] -= dy
        H[..., 1] -= dy
    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    # the window on the output frame grid, at absolute times (each track has its own times)
    t0, t1 = float(t_src[0]), float(t_src[-1])
    if len(t_src) == 1 and seconds is not None:          # a wholly still person (no channel at all): hold the pose
        t1 = max(t1, float(start) + float(seconds))
    begin = max(float(start), t0)
    if seconds is None:                                  # every grid time up to the person's last key
        n = int(math.floor((t1 - begin) * fps + 1e-3)) + 1          # 1e-3 frame: the GLB keeps float32 times
    else:                                                # grid times before the window's end and within half a frame
        end = min(float(start) + float(seconds), t1 + 0.5 / fps)     # of the last key
        n = int(math.ceil((end - begin) * fps - 1e-3))
    if n < 2:
        raise MocapError(f"the person is in the video from {t0:.2f} to {t1:.2f} s: the window {start:g} s + "
                         f"{(seconds or 0):g} s has no frames of it")
    # smoothed with a margin of real frames around the window (the window's first and last frames keep neighbours)
    m = max(int(smooth), int(legs_smooth)) // 2 + 1
    a = min(m, int(np.floor((begin - t0) * fps + 1e-3)))
    b = min(m, int(np.floor((t1 - (begin + (n - 1) / fps)) * fps + 1e-3)))
    grid = begin + np.arange(-a, n + max(0, b)) / fps
    P = np.stack([_interp(t_src, P[:, j], grid) for j in range(P.shape[1])], 1)
    H = np.stack([_interp(t_src, H[:, j], grid) for j in range(H.shape[1])], 1)
    P, H = sg(P, smooth), sg(H, smooth)
    legs = [BODY.index(x) for x in ("RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle")]
    P[:, legs] = sg(P[:, legs], legs_smooth)
    P, H = P[a:a + n], H[a:a + n]
    place = np.median(_interp(t_src, hips_full, grid[a:a + n]), 0)
    frames, scale = _frames(P, H, min_feet)
    rec = {"fps": fps, "frames": frames, "start_frame": int(round(begin * fps)), "start_seconds": round(begin, 3),
           "seconds": round(n / fps, 3), "source": str(Path(glb).resolve()), "detector": "sam3d_body",
           "tracks": len(tracks), "track": ti, "place_m": [round(float(place[0]), 4), round(float(place[1]), 4)],
           "scale": {**scale, "floor_tilt_deg": round(tilt, 2), "floor_from": floor_from}}
    if out:
        Path(out).write_text(json.dumps(rec), encoding="utf-8")
    return rec


# ---------------------------------------------------------------- several people (workflows/sam3d_body_multi)
# User, 2026-10-25: "for the from a video framework of templates, is there anyway to detect multiple people and extract
# the motion of multiple people?". SAM 3.1 finds every "person" and follows each through the video (object k = GLB
# track k); SAM 3D Body reads each one inside its own mask; the GLB roots are ABSOLUTE (camera space), so the people keep
# their distances. A run takes up to PEOPLE_MAX characters (the group sheet, crop-and-stitch looks: AGENTS.md 7.3;
# 2026-10-25, user: "if it has three then three options"), so a template is made for up to that many people.
TRACK_COLORS = [(0.12, 0.47, 0.71), (1.0, 0.5, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16),
                (0.58, 0.4, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.5, 0.5, 0.5),
                (0.74, 0.74, 0.13), (0.09, 0.75, 0.81), (0.94, 0.76, 0.06), (0.42, 0.68, 0.84)]   # SAM3_TrackPreview
LABEL_MAX_DIST = 60.0               # 0..255 RGB distance of a label pixel to its object's colour (H.264 shifts colours)
LABEL_WIDTH = 320                   # the label video is read at this width (presence and boxes need no detail)
PRESENT_MIN_FRAC = 0.002            # a person covers at least this share of the frame ...
PRESENT_REL = 0.25                  # ... and at least this share of its own median visible area (else: hidden)
PEOPLE_MAX = 4                      # characters per run (runner.MAX_CHARACTERS: --character repeated)
MIN_APART_M = 0.5                   # two mannequins never stand closer than this (their bodies would cut each other)
DEPTH_MAX_M = 1.5                   # a depth difference beyond this is clamped: depth from one camera is a guess, and
#                                     a short (chibi) character stood 1.9 m "behind" its partner at the same depth (test
#                                     2026-10-25, multi_test_duo)
SPAN_MIN_S = 2.0                    # the shared time of the chosen people


def track_color(k: int) -> str:
    r, g, b = TRACK_COLORS[k % len(TRACK_COLORS)]
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


def _kept(tr: dict, keep: np.ndarray, fps: float) -> dict:
    """A track with only the samples of kept video frames (frame = round(t * fps)); at least two must remain."""
    idx = np.clip(np.round(tr["t"] * fps).astype(int), 0, max(0, len(keep) - 1))
    m = keep[idx] if len(keep) else np.zeros(len(idx), bool)
    if m.sum() < 2:
        raise MocapError(f"{tr['name']}: the person is visible in fewer than 2 frames")
    return {"name": tr["name"], "t": tr["t"][m], "root": tr["root"][m],
            "joints": {k: v[m] for k, v in tr["joints"].items()}}


def ray_root(root: np.ndarray) -> np.ndarray:
    """ABSOLUTE roots (GLB axes: x right, y up, z toward the viewer; the camera at 0) slid along their camera rays to
    the person's median depth. SAM 3D Body's depth jitters by decimetres and its x / height are pixel x depth, so each
    jitter moves the root along the slanted ray; dropping only the depth left that slant in x and height, and the floor
    fit read it as a tilted floor (multi_test_duo 2026-10-25: a body standing 7.9 deg off vertical was turned to 18.5 deg,
    user: "why are the mannequins slanted"). On its ray the root keeps its place in the picture."""
    root = np.asarray(root, float)
    depth = -root[:, 2]
    if not (depth > 0.05).all():                         # not absolute (a "centered" GLB): as it is
        return root
    return root * (float(np.median(depth)) / depth)[:, None]


FLOOR_AGREE_DEG = 5.0              # the feet's floor counts only within this of the people's mean body up


def shared_floor(glb: str | Path, tracks: list[int], pres: np.ndarray, fps: float = 24.0) -> tuple[np.ndarray, str]:
    """ONE level rotation for all chosen people (their relative places stay right). Candidates: the floor plane under
    every person's lower ankle (floor_rotation, roots on their rays) and the people's mean body up (each person's mean
    hips -> neck, averaged). The floor is used only when it agrees with the body up within FLOOR_AGREE_DEG: in
    multi_test_duo (2026-10-25) the bodies agreed with the template camera's 7.5 deg pitch (5.1 and 7.9 deg) but the feet
    plane was 13.6 deg off, dragged by a wrong depth (a chibi character read 1.7 m too far: on her ray "farther" is also
    "lower") and her noisy feet; leveled by it, both mannequins leaned 12-15 deg. Otherwise the body up (within
    BODY_UP_MAX_DEG, as floor_rotation's own fallback). -> (rotation, "floor" | "body" | "floor-unchecked" | "none")."""
    data = read_glb(glb)["tracks"]
    parts, ups = [], []
    J = {n: i for i, n in enumerate(BODY)}
    for k in tracks:
        tr = _kept(data[k], np.asarray(pres[k], bool), fps)
        Pj = np.stack([tr["joints"][n] for n in BODY], 1)
        P = to_blender(Pj + ray_root(tr["root"])[:, None])
        parts.append(P)
        ups.append(unit((P[:, J["Neck"]] - (P[:, J["RHip"]] + P[:, J["LHip"]]) / 2).mean(0)))
    R, src = floor_rotation(np.concatenate(parts, 0))
    up = unit(np.mean(ups, 0))
    body_ok = np.degrees(np.arccos(np.clip(up[2], -1, 1))) <= BODY_UP_MAX_DEG
    if src == "floor" and np.degrees(np.arccos(np.clip(float(np.dot(R[2], up)), -1, 1))) <= FLOOR_AGREE_DEG:
        return R, "floor"
    if body_ok:
        return _align(up), "body"
    return R, ("floor-unchecked" if src == "floor" else src)


def torso_tilt(frames: list) -> float:
    """Degrees between the mean spine direction of motion frames and the vertical (a standing dancer: a few)."""
    sp = unit(np.mean([f["dirs"]["spine"] for f in frames], 0))
    return round(float(np.degrees(np.arccos(np.clip(sp[2], -1, 1)))), 1)


def read_labels(path: str | Path, objects: int) -> dict:
    """The label video of workflows/sam3d_body_multi (SAM3_TrackPreview without images: object k's mask in
    TRACK_COLORS[k] on black) -> {"frames", "size": [w, h], "area": [objects][frames] (share of the frame),
    "box": [objects][frames] ([x0, y0, x1, y1] in 0..1, or None)}. Overlaps go to the lower number (as drawn)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise MocapError(f"cannot read the label video {path}")
    cols = np.array(TRACK_COLORS[:max(1, objects)], np.float32) * 255.0
    area, box, size = [[] for _ in range(objects)], [[] for _ in range(objects)], None
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            size = size or [w, h]
            sw = min(LABEL_WIDTH, w)
            sh = max(1, round(h * sw / w))
            rgb = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_NEAREST)[..., ::-1].astype(np.float32)
            d = np.linalg.norm(rgb[:, :, None, :] - cols[None, None], axis=-1)        # (h, w, objects)
            best = d.argmin(-1)
            hit = (d.min(-1) < LABEL_MAX_DIST) & (rgb.max(-1) > 40)
            for k in range(objects):
                m = hit & (best == k)
                n = int(m.sum())
                area[k].append(n / float(sw * sh))
                if n:
                    ys, xs = np.nonzero(m)
                    box[k].append([round(xs.min() / sw, 4), round(ys.min() / sh, 4),
                                   round((xs.max() + 1) / sw, 4), round((ys.max() + 1) / sh, 4)])
                else:
                    box[k].append(None)
    finally:
        cap.release()
    if size is None:
        raise MocapError(f"the label video {path} has no frames")
    return {"frames": len(area[0]) if objects else 0, "size": size, "area": area, "box": box}


def _drop_short(x: np.ndarray, min_len: int) -> np.ndarray:
    """x (bool) with every True run shorter than min_len set to False."""
    x = x.copy()
    i = 0
    while i < len(x):
        if not x[i]:
            i += 1
            continue
        j = i
        while j < len(x) and x[j]:
            j += 1
        if j - i < min_len:
            x[i:j] = False
        i = j
    return x


def _close_holes(x: np.ndarray, max_len: int) -> np.ndarray:
    """x (bool) with every False run of at most max_len frames BETWEEN two True frames set to True."""
    x = x.copy()
    on = np.nonzero(x)[0]
    for a, b in zip(on[:-1], on[1:]):
        if 1 < b - a <= max_len + 1:
            x[a:b] = True
    return x


def presence(labels: dict) -> np.ndarray:
    """[objects, frames] bool: the person is well visible (PRESENT_MIN_FRAC of the frame and PRESENT_REL of its own
    median area); flicker shorter than 3 frames is dropped and holes up to 2 frames are closed."""
    a = np.array(labels["area"], float).reshape(len(labels["area"]), -1)
    out = np.zeros(a.shape, bool)
    for k in range(len(a)):
        seen = a[k][a[k] >= PRESENT_MIN_FRAC]
        if not len(seen):
            continue
        p = a[k] >= max(PRESENT_MIN_FRAC, PRESENT_REL * float(np.median(seen)))
        out[k] = _close_holes(_drop_short(p, 3), 2)
    return out


def people_summary(labels: dict, fps: float = 24.0) -> list[dict]:
    """One line per SAM 3 object: its number, colour, the share of frames it is visible, its first/last second, its
    median height and centre in the picture (0..1). The app's "who is who" list."""
    pres = presence(labels)
    out = []
    for k in range(len(pres)):
        fr = np.nonzero(pres[k])[0]
        boxes = [b for b, p in zip(labels["box"][k], pres[k]) if b and p]
        out.append({"track": k, "color": track_color(k),
                    "visible": round(float(pres[k].mean()), 3) if pres.shape[1] else 0.0,
                    "first_s": round(fr[0] / fps, 2) if len(fr) else None,
                    "last_s": round(fr[-1] / fps, 2) if len(fr) else None,
                    "height": round(float(np.median([b[3] - b[1] for b in boxes])), 3) if boxes else 0.0,
                    "x": round(float(np.median([(b[0] + b[2]) / 2 for b in boxes])), 3) if boxes else None})
    return out


def choose_people(summary: list[dict], count: int) -> list[int]:
    """The `count` people to copy when the user did not pick: the ones seen longest and largest (visible share x
    height; at least a quarter of the clip), left to right in the picture."""
    cand = [s for s in summary if s["visible"] >= 0.25 and s["x"] is not None]
    cand.sort(key=lambda s: s["visible"] * s["height"], reverse=True)
    return [s["track"] for s in sorted(cand[:count], key=lambda s: s["x"])]


def shared_span(pres: np.ndarray, tracks: list[int], fps: float = 24.0) -> tuple[int, int]:
    """The frames (first, last) in which ALL chosen people are in the video: from the last one's first frame to the
    first one's last frame. The template is TRIMMED to it (a mannequin that holds a pose while its dancer is not in the
    video would look frozen); short hidden moments inside are filled by interpolation."""
    firsts, lasts = [], []
    for k in tracks:
        fr = np.nonzero(pres[k])[0]
        if not len(fr):
            raise MocapError(f"person {k} is not visible in the video")
        firsts.append(int(fr[0]))
        lasts.append(int(fr[-1]))
    a, b = max(firsts), min(lasts)
    if (b - a + 1) / fps < SPAN_MIN_S:
        raise MocapError(f"the chosen people are in the video together for {max(0, b - a + 1) / fps:.1f} s only "
                         f"(at least {SPAN_MIN_S:g} s): choose other people or another part of the video")
    return a, b


def check_tracks(tracks: list[int], objects: int) -> list[int]:
    try:
        tracks = [int(t) for t in tracks]
    except (TypeError, ValueError):
        raise MocapError("people are given by their numbers (0, 1, ...)") from None
    if not tracks:
        raise MocapError("choose at least one person")
    if len(set(tracks)) != len(tracks):
        raise MocapError("choose each person once")
    if len(tracks) > PEOPLE_MAX:
        raise MocapError(f"a run takes at most {PEOPLE_MAX} characters: choose 1 to {PEOPLE_MAX} people")
    bad = [t for t in tracks if not 0 <= t < objects]
    if bad:
        raise MocapError(f"no person {bad[0]}: the video has {objects} (numbered 0 to {objects - 1})")
    return tracks


def place_people(places_m: list, scales: list) -> list[list[float]]:
    """Floor places (metres, in the video) -> mannequin places centred on the people's middle: one scale for all (the
    mean dancer -> mannequin scale, so the distance keeps its proportion to the bodies), the depth SPREAD of all the
    people limited to DEPTH_MAX_M (centred on the middle of the range and scaled down, so the depth ORDER of three or
    more people stays; for two people the same as clamping each to +-DEPTH_MAX_M / 2), and neighbours (left to right)
    that stand closer than MIN_APART_M moved apart sideways."""
    P = np.array(places_m, float).reshape(-1, 2) * float(np.mean(scales))
    P -= P.mean(0)
    if len(P) > 1:
        y0, y1 = float(P[:, 1].min()), float(P[:, 1].max())
        P[:, 1] -= (y0 + y1) / 2
        if y1 - y0 > DEPTH_MAX_M:
            P[:, 1] *= DEPTH_MAX_M / (y1 - y0)
        order = sorted(range(len(P)), key=lambda i: P[i, 0])
        xs = [float(P[i, 0]) for i in order]
        moved = False
        for k in range(1, len(order)):              # left to right: each one at least MIN_APART_M from its neighbour
            a, b = P[order[k - 1]], P[order[k]]
            if float(np.hypot(xs[k] - xs[k - 1], b[1] - a[1])) < MIN_APART_M and xs[k] - xs[k - 1] < MIN_APART_M:
                xs[k] = xs[k - 1] + MIN_APART_M
                moved = True
        if moved:                                   # keep the group's middle where it was
            mid = (min(float(P[i, 0]) for i in order) + max(float(P[i, 0]) for i in order)) / 2
            shift = mid - (xs[0] + xs[-1]) / 2
            for i, x in zip(order, xs):
                P[i, 0] = x + shift
    return [[round(float(x), 3), round(float(y), 3)] for x, y in P]


CAM_TAN_HALF_W, CAM_TAN_HALF_H = 0.6, 0.35     # templates/flash_step: lens 30 mm, sensor 36 mm, 1536 x 896
CAM_HEIGHT_NEED = 1.05                          # half the height the camera must see around its target (0.92 m)


def camera_distance(places: list, motions: list, base: float = 4.0, reach: float = 0.6, far: float = 9.0) -> float:
    """How far the still front camera (looking at the floor's middle) stands so every person stays in the frame: the
    widest hip place + `reach` (the arms) within 90% of the half width at that person's distance, and the nearest
    person's whole height. At least `base` (the one-person template's 4 m)."""
    need = base
    for (x, y), m in zip(places, motions):
        xs = [abs(x + f["hips"][0]) for f in m["frames"]] or [abs(x)]
        need = max(need, (max(xs) + reach) / (CAM_TAN_HALF_W * 0.9) - y, CAM_HEIGHT_NEED / CAM_TAN_HALF_H - y)
    return round(min(far, need), 2)


def people_motions(glb: str | Path, labels: dict, tracks: list[int], fps: float = 24.0) -> dict:
    """The motions of the chosen people (SAM 3 object numbers), trimmed to the time all are in the video, each with
    its mannequin place; ordered LEFT to RIGHT in the picture (the run's character 1 = the left figure).
    -> {"people": [{"track", "place", "color", "filled"}], "motions": [...], "span": [first, last] (frames),
    "start_s", "seconds", "camera_distance"}. `filled` = frames of the span in which that person was hidden
    (interpolated)."""
    objects = len(read_glb(glb)["tracks"])
    tracks = check_tracks(tracks, objects)
    pres = presence(labels)
    if len(pres) < objects:
        raise MocapError(f"the label video shows {len(pres)} people, the GLB {objects}")
    a, b = shared_span(pres, tracks, fps)
    start, seconds = a / fps, (b - a + 1) / fps
    R, floor_from = shared_floor(glb, tracks, pres, fps)
    motions = [motion_from_glb(glb, None, start=start, seconds=seconds, fps=fps, track=k, keep=pres[k], rotation=R,
                               on_ray=True) for k in tracks]
    for m in motions:
        m["scale"]["floor_from"] = f"shared:{floor_from}"
        m["scale"]["torso_tilt_deg"] = torso_tilt(m["frames"])
    n = min(len(m["frames"]) for m in motions)
    for m in motions:                                    # equal lengths (float rounding of the window)
        del m["frames"][n:]
    places = place_people([m["place_m"] for m in motions], [m["scale"]["scale_dancer_to_mannequin"] for m in motions])
    order = sorted(range(len(tracks)), key=lambda i: places[i][0])
    people = [{"track": tracks[i], "place": places[i], "color": track_color(tracks[i]),
               "filled": int((~pres[tracks[i]][a:b + 1]).sum())} for i in order]
    motions = [motions[i] for i in order]
    for p, m in zip(people, motions):
        m["person"] = p
    return {"people": people, "motions": motions, "span": [a, b], "start_s": round(start, 3),
            "seconds": round(n / fps, 3), "camera_distance": camera_distance([p["place"] for p in people], motions)}
