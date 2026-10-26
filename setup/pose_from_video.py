"""Turn a dance video's 2D poses into a motion file for the mannequin (2026-10-22: a template that copies a reference
dance; the reference must face the camera with a still camera).

    python setup/pose_from_video.py PASS_DIR OUT_JSON [--seconds 10] [--fps 24] [--start S] [--overlay OUT.mp4]

PASS_DIR = the output of cinestudio/video_passes.py on the video (keypoints.json: coco18 body + hand21 in pixels,
passes.json: fps). Steps: resample to `fps` (linear), fill frames without a figure, smooth lightly (a 5-frame
Savitzky-Golay: the hard stop of a fast dodge stays), choose the window (`--start`, else the start whose clip joins -
frames 97-121 and 193-217 of a 121/25 clip layout - have the least motion), and give per frame:
- hips: the hip centre's sideways move and its drop from the standing height (metres on the mannequin's scale);
- dirs: unit directions for spine, chest, head, upper_arm/forearm/hand .L/.R (the person's right = the rig's .R);
  an arm bone shorter on screen than on the mannequin is turned toward the camera by the missing length;
- feet: ankle positions (x, 0, z) for the two-bone leg IK.
The image right is the world +X (the camera looks along +Y at a figure facing -Y)."""
import argparse
import json
from pathlib import Path

import numpy as np

NAMES = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "r_hip", "r_knee",
         "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]
J = {n: i for i, n in enumerate(NAMES)}
# mannequin bone lengths (blender_lib/mannequin.py REST)
L_UPPER, L_FORE, L_HAND = 0.2766, 0.2516, 0.1005
L_THIGH, L_SHIN = 0.43, 0.43
L_CLAV = 0.16                                  # shoulder bone (neck side -> shoulder joint)
TORSO = 0.50                                   # neck (1.45) above the hip centre (0.95)
ANKLE_Z = 0.09
CLIP, OVERLAP = 121, 25
MIN_FEET_M = 0.08                              # round 6: ankles never closer or crossed (an X of shins read badly)
STILL_M, LOW_M = 0.018, 0.08                   # a standing foot: moves < 1.8 cm per frame, < 8 cm above the floor
#                                                (gauntlet round 3; 1.2 cm / 6 cm caught only 41 and 36 of 240 frames)


def load(pass_dir: Path):
    kp = json.loads((pass_dir / "keypoints.json").read_text(encoding="utf-8"))
    info = json.loads((pass_dir / "passes.json").read_text(encoding="utf-8"))
    body = np.array(kp["frames"], np.float32)                       # [T, 18, 3]
    hr = np.array(kp["hands"]["R"], np.float32)                     # [T, 21, 3]
    hl = np.array(kp["hands"]["L"], np.float32)
    return body, hr, hl, float(info["fps"]), (info["width"], info["height"])


def fill(a):
    """Points not shown get the nearest shown value in time (per joint), so smoothing has no holes."""
    out = a.copy()
    T, N = a.shape[:2]
    for j in range(N):
        ok = a[:, j, 2] > 0
        if not ok.any():
            continue
        idx = np.arange(T)
        for c in (0, 1):
            out[:, j, c] = np.interp(idx, idx[ok], a[ok, j, c])
        out[:, j, 2] = ok
    return out


def resample(a, src_fps, fps):
    T = a.shape[0]
    t_src = np.arange(T) / src_fps
    t = np.arange(int(T / src_fps * fps)) / fps
    out = np.empty((len(t),) + a.shape[1:], np.float32)
    for j in range(a.shape[1]):
        for c in range(a.shape[2]):
            out[:, j, c] = np.interp(t, t_src, a[:, j, c])
    return out


def smooth(a, win=5):
    """Savitzky-Golay (order 2) along time on x and y."""
    k = win // 2
    x = np.arange(-k, k + 1)
    A = np.vander(x, 3, increasing=True)
    w = np.linalg.pinv(A)[0]                                         # the value at the centre
    out = a.copy()
    pad = np.concatenate([a[:1].repeat(k, 0), a, a[-1:].repeat(k, 0)])
    for c in (0, 1):
        out[..., c] = sum(w[i] * pad[i:i + len(a), ..., c] for i in range(win))
    return out


def motion(body):
    v = np.linalg.norm(np.diff(body[:, :14, :2], axis=0), axis=2).mean(axis=1)
    return np.concatenate([[0], v])


def pick_start(energy, frames):
    """The start whose shared clip frames (97-121, 193-217, 1-based) move least."""
    shared = [s for k in range(2, 20) for s in range((k - 1) * (CLIP - OVERLAP), (k - 1) * (CLIP - OVERLAP) + OVERLAP)
              if s < frames]
    best, cost = 0, None
    for st in range(0, max(1, len(energy) - frames + 1)):
        c = float(energy[[st + s for s in shared]].sum())
        if cost is None or c < cost:
            best, cost = st, c
    return best


def bone_dir(p0, p1, s, length, forward=True):
    """Unit direction (x, y, z) of a bone from 2D points (pixels; y down) at `s` metres per pixel."""
    dx, dz = (p1[0] - p0[0]) * s, -(p1[1] - p0[1]) * s
    planar = float(np.hypot(dx, dz))
    dy = -np.sqrt(max(0.0, length ** 2 - planar ** 2)) if forward and planar < length else 0.0
    v = np.array([dx, dy, dz])
    n = np.linalg.norm(v)
    return (v / n).round(4).tolist() if n > 1e-6 else [0.0, 0.0, -1.0]


def knee_shin(thigh, p0, p1, s):
    """The shin's direction with the KNEE RULE (gauntlet round 5): a knee bends only backward, so a foreshortened shin
    takes the depth sign that puts it behind the thigh's line (+Y = away from the camera). Always turning it toward
    the camera (bone_dir) drew back kicks as legs crossed in front (critic 1: an X of shins)."""
    dx, dz = (p1[0] - p0[0]) * s, -(p1[1] - p0[1]) * s
    planar = float(np.hypot(dx, dz))
    d = float(np.sqrt(max(0.0, L_SHIN ** 2 - planar ** 2))) if planar < L_SHIN else 0.0
    best = None
    for dy in (d, -d):                                  # the sign whose shin lies behind the thigh, else the smaller bend
        v = np.array([dx, dy, dz]) / max(1e-6, float(np.linalg.norm([dx, dy, dz])))
        behind = v[1] >= thigh[1] - 0.05
        bend = float(np.degrees(np.arccos(np.clip(np.dot(v, thigh), -1, 1))))
        key = (0 if behind else 1, bend)
        if best is None or key < best[0]:
            best = (key, v)
    return best[1].round(4).tolist()


def flat_dir(p0, p1):
    v = np.array([p1[0] - p0[0], 0.0, -(p1[1] - p0[1])])
    n = np.linalg.norm(v)
    return (v / n).round(4).tolist() if n > 1e-6 else [0.0, 0.0, 1.0]


def convert(body, hr, hl):
    mid_hip = (body[:, J["r_hip"], :2] + body[:, J["l_hip"], :2]) / 2
    neck = body[:, J["neck"], :2]
    torso_px = float(np.median(np.linalg.norm(neck - mid_hip, axis=1)))
    s = TORSO / max(1.0, torso_px)                                  # metres per pixel
    ankles_y = np.maximum(body[:, J["r_ankle"], 1], body[:, J["l_ankle"], 1])
    ground = float(np.percentile(ankles_y, 90))                     # the floor line (feet down most of the time)
    x_ref = float(np.median(mid_hip[:, 0]))
    hip_z = (ground - mid_hip[:, 1]) * s + ANKLE_Z
    stand = float(np.percentile(hip_z, 75))                         # the standing hip height on this scale
    frames = []
    min_gap = MIN_FEET_M / s
    planar, hand_w = {}, {}
    for t in range(len(body)):
        b = body[t]
        P = {n: b[J[n], :2].copy() for n in NAMES}
        gap = P["l_ankle"][0] - P["r_ankle"][0]           # the person's left foot is on the picture's right
        both_down = all((ground - P[k][1]) * s < LOW_M for k in ("l_ankle", "r_ankle"))
        if gap < min_gap and both_down:                   # crossed or touching ON THE FLOOR (DWPose mixes feet-together
            #                                               frames): move both apart; a foot in the air may cross
            #                                               (round 13c: the clamp cancelled a kick across the body)
            push = (min_gap - gap) / 2
            P["l_ankle"][0] += push
            P["r_ankle"][0] -= push
        d = {}
        d["spine"] = d["chest"] = flat_dir(mid_hip[t], neck[t])
        ears = [P[e] for e in ("r_ear", "l_ear") if b[J[e], 2] > 0]
        top = np.mean(ears, axis=0) if ears else P["nose"]
        h = np.array(flat_dir(neck[t], top + (top - neck[t]) * 0.5))
        sp = np.array(d["spine"])
        h = sp + np.clip(h - sp, -0.35, 0.35)                       # the head stays near the body's line
        d["head"] = (h / np.linalg.norm(h)).round(4).tolist()
        for side, pre, hand in ((".R", "r_", hr[t]), (".L", "l_", hl[t])):
            d["shoulder" + side] = bone_dir(P["neck"], P[pre + "shoulder"], s, L_CLAV, forward=False)   # round 4
            d["upper_arm" + side] = bone_dir(P[pre + "shoulder"], P[pre + "elbow"], s, L_UPPER)
            d["forearm" + side] = bone_dir(P[pre + "elbow"], P[pre + "wrist"], s, L_FORE)
            if hand[9, 2] > 0 and hand[0, 2] > 0:                    # wrist -> middle-finger knuckle
                d["hand" + side] = bone_dir(hand[0, :2], hand[9, :2], s, L_HAND)
            else:
                d["hand" + side] = d["forearm" + side]
        for side, pre in ((".R", "r_"), (".L", "l_")):           # legs by direction too (round 1 of the gauntlet:
            d["thigh" + side] = bone_dir(P[pre + "hip"], P[pre + "knee"], s, L_THIGH)   # the IK put the knee where the
            d["shin" + side] = knee_shin(d["thigh" + side], P[pre + "knee"], P[pre + "ankle"], s)   # legs allowed)
        feet = {}
        for side, pre in ((".R", "r_"), (".L", "l_")):
            a = P[pre + "ankle"]
            feet[side] = [round(float((a[0] - x_ref) * s), 4), 0.0,
                          round(float(max(ANKLE_Z, (ground - a[1]) * s + ANKLE_Z)), 4)]
        contact = {}
        for side, pre in ((".R", "r_"), (".L", "l_")):           # the foot stands: low and nearly still
            j = J[pre + "ankle"]
            sp = float(np.linalg.norm(body[min(t + 1, len(body) - 1), j, :2] - body[max(t - 1, 0), j, :2])) / 2
            contact[side] = bool(sp * s < STILL_M and (ground - body[t, j, 1]) * s < LOW_M)
        hl_ = P["l_hip"] - P["r_hip"]                            # the pelvis tilt (round 4), degrees, + = left up
        for side, pre, hand in ((".R", "r_", hr[t]), (".L", "l_", hl[t])):   # round 7: the raw projected vectors
            for bone, a_, b_ in (("upper_arm", pre + "shoulder", pre + "elbow"), ("forearm", pre + "elbow", pre + "wrist"),
                                 ("thigh", pre + "hip", pre + "knee"), ("shin", pre + "knee", pre + "ankle")):
                planar.setdefault(bone + side, []).append(((P[b_][0] - P[a_][0]) * s, -(P[b_][1] - P[a_][1]) * s))
            planar.setdefault("hand" + side, []).append(((hand[9, 0] - hand[0, 0]) * s, -(hand[9, 1] - hand[0, 1]) * s))
            planar.setdefault("leg" + side, []).append(((P[pre + "ankle"][0] - P[pre + "hip"][0]) * s,
                                                        -(P[pre + "ankle"][1] - P[pre + "hip"][1]) * s))
            hand_w.setdefault(side, []).append(float(min(hand[0, 2], hand[9, 2])))
        frames.append({"pelvis_roll": round(float(np.degrees(np.arctan2(-hl_[1], hl_[0]))), 2), "contact": contact, "hips": [round(float((mid_hip[t, 0] - x_ref) * s), 4),
                                round(float(np.clip(hip_z[t] - stand, -0.30, 0.05)), 4)],
                       "dirs": d, "feet": feet})
    depth_pass(frames, planar, hand_w)
    hysteresis(frames)
    found = kicks(frames, planar)
    return frames, {"metres_per_pixel": s, "ground_px": ground, "x_ref_px": x_ref, "kicks": found}


def hysteresis(frames, enter=2, leave=2, min_len=3):
    """Round 8 (Codex review): a contact starts after `enter` still frames and ends after `leave` moving frames, and a
    contact shorter than `min_len` frames is dropped (one-frame flicker re-pinned a foot at a new place)."""
    for side in (".R", ".L"):
        raw = [f["contact"][side] for f in frames]
        out, on, run = [], bool(raw[0]), 0       # round 12: a contact at the start counts from frame 1
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


KICK_MIN_PEAK_M, KICK_LEN = 0.55, (4, 30)       # a kick: the foot off the floor, a high peak, a short swing


def kicks(frames, planar=None):
    """Round 13 (user: "the leg that is trying to kick should extend out completely for the middle part ... both
    kicks"): kick segments (no contact, foot above 0.2 m, peak >= KICK_MIN_PEAK_M, KICK_LEN frames) get a weight per
    frame: 0 -> 1 over the first third, 1 in the middle third, back to 0 (a short kick is widened by 3 frames on each
    side so the leg can straighten smoothly). The builder straightens the leg along its hip -> foot line by it."""
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
            if KICK_LEN[0] <= j - i <= KICK_LEN[1] and z[i:j].max() >= KICK_MIN_PEAK_M and 0 < i and j < T:  # whole kicks
                a, b = i, j
                if b - a < 12:                              # widen a short kick for the ramps
                    a, b = max(0, a - 3), min(T, b + 3)
                n = b - a
                for t in range(a, b):
                    u = (t - a + 0.5) / n
                    w = min(1.0, u * 3, (1 - u) * 3)
                    w = w * w * (3 - 2 * w)                 # smoothstep
                    frames[t]["kick"][side] = round(max(frames[t]["kick"][side], w), 3)
                if planar and ("leg" + side) in planar:      # the leg points along the DANCER's hip -> ankle line
                    LG = np.array(planar["leg" + side], np.float64)
                    full = max(1e-3, float(np.percentile(np.hypot(LG[:, 0], LG[:, 1]), 98)))
                    for t in range(a, b):
                        dx, dz = LG[t] / full                   # in leg lengths
                        pl = min(1.0, float(np.hypot(dx, dz)))
                        v = np.array([dx, -np.sqrt(max(0.0, 1 - pl ** 2)), dz])   # the rest goes toward the camera
                        frames[t]["kick_dir"] = frames[t].get("kick_dir", {})
                        frames[t]["kick_dir"][side] = (v / np.linalg.norm(v)).round(4).tolist()
                found.append({"side": side, "frames": [a + 1, b], "peak_z": round(float(z[i:j].max()), 2)})
            i = j
    return found


def _sg(x, win=7):
    k = win // 2
    A = np.vander(np.arange(-k, k + 1), 3, increasing=True)
    w = np.linalg.pinv(A)[0]
    pad = np.concatenate([np.repeat(x[:1], k), x, np.repeat(x[-1:], k)])
    return np.array([float(np.dot(w, pad[i:i + win])) for i in range(len(x))])


def _limit_turn(dirs, max_deg):
    """At most `max_deg` of turn per frame (a sudden flip of a small bone, e.g. a hand, becomes a short sweep)."""
    out = [dirs[0]]
    lim = np.radians(max_deg)
    for v in dirs[1:]:
        prev = out[-1]
        ang = float(np.arccos(np.clip(np.dot(prev, v), -1, 1)))
        if ang > lim:
            t = lim / ang
            v = prev * (1 - t) + v * t
            v = v / max(1e-6, np.linalg.norm(v))
        out.append(v)
    return out


def depth_pass(frames, planar, hand_w):
    """Round 7 (critic 2 + Codex review): the depth of each limb bone from the DANCER's own projected length (the 98th
    percentile over the window), not the mannequin's: a low camera makes a raised arm look short, and the mannequin's
    length read that as an arm pointing at the camera (arms at shoulder height where the dancer reached overhead);
    the depth magnitude smoothed over time (sqrt(L^2 - p^2) jumps near full length: Codex saw 12.7 cm root pops);
    arms and thighs toward the camera, shins behind the thigh (the knee rule); hands blended with the forearm by the
    hand's detection confidence and limited to 25 degrees of turn per frame."""
    T = len(frames)
    vecs = {}
    for bone, pl in planar.items():
        if bone.startswith("leg"):
            continue
        P = np.array(pl, np.float64)
        p = np.hypot(P[:, 0], P[:, 1])
        L = max(1e-3, float(np.percentile(p, 98)))
        depth = _sg(np.sqrt(np.clip(L ** 2 - p ** 2, 0, None)))
        depth = np.clip(depth, 0, None)
        vecs[bone] = (P, depth)
    for side in (".R", ".L"):
        out = {}
        for bone in ("upper_arm", "forearm", "thigh"):
            P, dep = vecs[bone + side]
            out[bone] = [np.array([P[t, 0], -dep[t], P[t, 1]]) for t in range(T)]
        P, dep = vecs["shin" + side]
        shin, sign, other_run = [], None, 0
        for t in range(T):                                # round 9: the sign changes only after 3 frames that ask
            th = out["thigh"][t] / max(1e-6, np.linalg.norm(out["thigh"][t]))   # for it (a per-frame choice flipped the
            keys = {}                                                           # foot 60-75 cm front <-> back)
            for sg in (1, -1):
                v = np.array([P[t, 0], sg * dep[t], P[t, 1]])
                v = v / max(1e-6, np.linalg.norm(v))
                keys[sg] = ((0 if v[1] >= th[1] - 0.05 else 1, float(np.arccos(np.clip(np.dot(v, th), -1, 1)))), v)
            best = min(keys, key=lambda k: keys[k][0])
            if sign is None:
                sign = best
            elif best != sign:
                other_run += 1
                if other_run >= 3:
                    sign, other_run = best, 0
            else:
                other_run = 0
            shin.append(sign)
        # round 11: the SIGNED depth smoothed over 11 frames: a sign change becomes a swing through the side plane
        # (the 3-frame hysteresis alone still switched in one frame: the foot jumped 50-63 cm front <-> back)
        ys = _sg(np.array([sg * dep[t] for t, sg in enumerate(shin)]), 11)
        out["shin"] = [np.array([P[t, 0], ys[t], P[t, 1]]) for t in range(T)]
        for bone in ("upper_arm", "forearm", "thigh", "shin"):          # round 9: smooth the 3D directions in time
            V = np.array([v / max(1e-6, np.linalg.norm(v)) for v in out[bone]])   # round 12: legs over 9 frames (the
            win = 9 if bone in ("thigh", "shin") else 5                            # feet zig-zagged on blurred DWPose)
            V = np.stack([_sg(V[:, c], win) for c in range(3)], axis=1)
            out[bone] = [v / max(1e-6, np.linalg.norm(v)) for v in V]
        P, dep = vecs["hand" + side]
        w = np.clip(_sg(np.array(hand_w[side])), 0, 1)
        hands = []
        for t in range(T):
            fo = out["forearm"][t] / max(1e-6, np.linalg.norm(out["forearm"][t]))
            hv = np.array([P[t, 0], -dep[t], P[t, 1]])
            hv = hv / max(1e-6, np.linalg.norm(hv)) if np.linalg.norm(hv) > 1e-6 else fo
            v = w[t] * hv + (1 - w[t]) * fo
            hands.append(v / max(1e-6, np.linalg.norm(v)))
        out["hand"] = _limit_turn(hands, 25.0)
        for bone, seq in out.items():
            for t in range(T):
                v = seq[t] / max(1e-6, np.linalg.norm(seq[t]))
                frames[t]["dirs"][bone + side] = v.round(4).tolist()


def overlay(video, body_win, src_start, src_fps, fps, out):
    import cv2
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    pairs = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (0, 1)]
    for i, b in enumerate(body_win):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((src_start + i) / fps * src_fps)))
        ok, im = cap.read()
        if not ok:
            break
        for a, c in pairs:
            col = (0, 0, 255) if NAMES[c].startswith("r_") else (255, 120, 0)
            cv2.line(im, tuple(int(v) for v in b[a, :2]), tuple(int(v) for v in b[c, :2]), col, 2)
        cv2.putText(im, str(i + 1), (6, 18), 0, 0.5, (0, 255, 255), 1)
        vw.write(im)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pass_dir")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--start", type=float, default=None, help="window start in seconds (default: calm clip joins)")
    ap.add_argument("--overlay", default=None, help="write the skeleton over the video window (a check)")
    a = ap.parse_args()
    pd = Path(a.pass_dir)
    body, hr, hl, src_fps, size = load(pd)
    body, hr, hl = (smooth(resample(fill(x), src_fps, a.fps)) for x in (body, hr, hl))
    n = int(round(a.seconds * a.fps))
    if len(body) < n:
        raise SystemExit(f"the video has {len(body)} frames at {a.fps} fps, fewer than {n}")
    st = int(round(a.start * a.fps)) if a.start is not None else pick_start(motion(body), n)
    win = slice(st, st + n)
    frames, scale = convert(body[win], hr[win], hl[win])
    rec = {"fps": a.fps, "frames": frames, "start_frame": st, "start_seconds": round(st / a.fps, 3),
           "seconds": a.seconds, "source": str(pd.resolve()), "scale": scale, "size": size}
    Path(a.out).write_text(json.dumps(rec), encoding="utf-8")
    if a.overlay:
        info = json.loads((pd / "passes.json").read_text(encoding="utf-8"))
        overlay(info["video"], body[win], st, src_fps, a.fps, a.overlay)
    e = motion(body[win])
    print(json.dumps({"frames": n, "start_seconds": rec["start_seconds"], "metres_per_pixel": round(scale["metres_per_pixel"], 5),
                      "motion_mean": round(float(e.mean()), 2),
                      "motion_at_joins": [round(float(e[96:121].mean()), 2), round(float(e[192:217].mean()), 2)]}))


if __name__ == "__main__":
    main()
