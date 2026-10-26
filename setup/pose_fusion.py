"""Merge two pose detectors (DWPose and SDPose) into one skeleton per character, PART BY PART, and score it.

    python setup/pose_fusion.py --passes DIR --a dets_dwpose.json --b dets_sdpose.json
        [--choose owner|rig|avg] [--owner torso=b,arms=b,legs=b,head=a,hands=a,face=a]
        [--out score.json] [--save-fused fused.json] [--control DIR]

The dets files come from setup/dwpose_score.py --save-dets on the SAME render with Blender boxes (blender or
blender_iso; other box modes are refused, because their character names come from matching with the truth).
Every point of the merged skeleton comes from one detector; the merge only CHOOSES:
  parts      torso (neck, shoulders, hips), right arm, left arm, right leg, left leg, head (nose, eyes, ears),
             right hand, left hand, face (68 + 2). A part is taken whole from one detector, so an elbow and a
             wrist never come from different models.
  left/right before choosing, the two skeletons are lined up: when one fits the other better swapped, the one
             with fewer points is relabelled (a detector's own scores are not comparable between models)
  hands      per detector, its two hands go one-to-one to the two wrists of the merged body (2 permutations,
             the smaller total distance; a hand further than 15% of the height from its wrist is dropped)
  --choose   owner = a fixed detector per part (--owner);
             rig   = per part, the detector whose points lie closer to the Blender rig (the rig only chooses
                     between two real detections). NOTE: scoring this against the same rig is optimistic; judge
                     it with a blind critic and the motion follow score of a video too;
             avg   = the old rule (mean where both agree), for comparison
  joins      a limb from another detector than the torso must stay connected: its first bone (shoulder-elbow,
             hip-knee) in the merged skeleton keeps 0.75-1.33 x its length and 25 degrees of its direction in its own
             detector, and a missing end means NOT connected; else the torso detector's limb is used ("relinked"),
             and if no complete, connected limb exists the limb is left out ("dropped": nothing floats)
  head/face  the rig's face rule first (no nose or eyes where the rig's head faces away; ears may stay, as in
             OpenPose back views), then a head point needs both detectors; the dense face needs both detectors and
             the rig's face toward the camera, and its points come from ONE detector (no average)
  rig mode   each detector's left/right is checked against the rig on its own; hands are chosen by all 21 points
             against the rig's hand; a hand must start within 8% of the height of its final body wrist
Prints the same summary as dwpose_score.py, plus the share of parts taken from each detector.
"""
import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dwpose_score as ds  # noqa: E402

PARTS = {"torso": (1, 2, 5, 8, 11), "arm_r": (3, 4), "arm_l": (6, 7), "leg_r": (9, 10), "leg_l": (12, 13),
         "head": (0, 14, 15, 16, 17)}
# the join of each part with the rest of the body: (point on the torso / neck, first point of the part)
JOINS = {"arm_r": (2, 3), "arm_l": (5, 6), "leg_r": (8, 9), "leg_l": (11, 12), "head": (1, 0)}
OWNER_KEYS = {"torso": "torso", "arm_r": "arms", "arm_l": "arms", "leg_r": "legs", "leg_l": "legs", "head": "head"}
LR = {a: b for a, b in ds.PAIRS + ds.FACE_PAIRS} | {b: a for a, b in ds.PAIRS + ds.FACE_PAIRS}
ACCEPTED_BOXES = ("blender", "blender_iso")


def arr(v, n):
    """A detector's points as an (n, 3) array; a point is missing (NaN) unless x, y and its score are finite."""
    if v is None:
        return None
    a = np.array([[np.nan if x is None else x for x in p] for p in v], float).reshape(n, 3)
    bad = ~np.isfinite(a).all(axis=1)
    a[bad] = np.nan
    return None if bad.all() else a


def load(path):
    d = json.loads(Path(path).read_text())
    if d.get("box") not in ACCEPTED_BOXES:
        sys.exit(f"{path}: made with --box {d.get('box')}; fusion accepts only {ACCEPTED_BOXES}")
    def body(v):                                  # an all-invalid or missing body is all NaN, never None
        a = arr(v, 18) if v is not None else None
        return a if a is not None else np.full((18, 3), np.nan)
    return {f: {name: None if p is None else {"body": body(p["body"]),
                                              "L": arr(p["L"], 21), "R": arr(p["R"], 21), "face": arr(p["face"], 70)}
                for name, p in chars.items()}
            for f, chars in d["frames"].items()}


def has(p, i):
    return p is not None and not np.isnan(p[i, 0])


def swap_sides(p):
    q = dict(p)
    q["body"] = p["body"][[LR.get(i, i) for i in range(18)]]
    q["L"], q["R"] = p["R"], p["L"]
    return q


def _fits(body, ref, swapped):
    """Sum of distances of the limb pairs to ref (straight or swapped) and the number of pairs used."""
    d, n = 0.0, 0
    for x, y in ds.PAIRS:
        if has(body, x) and has(body, y) and not np.isnan(ref[[x, y], 0]).any() and ref[x, 2] > 0 and ref[y, 2] > 0:
            n += 1
            if swapped:
                d += ds._dist(body[x], ref[y]) + ds._dist(body[y], ref[x])
            else:
                d += ds._dist(body[x], ref[x]) + ds._dist(body[y], ref[y])
    return d, n


def orient(p, ref, margin=0.15):
    """Relabel p when its limbs fit ref clearly better with left and right swapped (>= 2 pairs, margin)."""
    st, n = _fits(p["body"], ref, False)
    sw, _ = _fits(p["body"], ref, True)
    if n >= 2 and sw < (1 - margin) * st:
        return swap_sides(p), True
    return p, False


def align(a, b, rig=None):
    """Line up left and right. With the rig: each detector against the rig, on its own (a mirrored detector cannot
    flip a correct one). Without the rig: detector agreement cannot tell which one is right; the one with fewer
    points is relabelled (a weak rule, for the owner/avg ablations only)."""
    if rig is not None:
        a, ra = orient(a, rig)
        b, rb = orient(b, rig)
        return a, b, ra or rb
    ref = a["body"].copy()
    ref[:, 2] = np.where(np.isnan(ref[:, 0]), 0, 1)
    st, n = _fits(b["body"], ref, False)
    sw, _ = _fits(b["body"], ref, True)
    if n < 2 or sw >= 0.85 * st:
        return a, b, False
    na, nb = int(np.sum(~np.isnan(a["body"][:, 0]))), int(np.sum(~np.isnan(b["body"][:, 0])))
    return (a, swap_sides(b), True) if na >= nb else (swap_sides(a), b, True)


def part_cost(pts, rig, idx, h):
    """Mean distance (as a share of h) of a part's points to the rig; a missing point costs 0.5."""
    c = [min(ds._dist(pts[i], rig[i]) / h, 0.5) if not np.isnan(pts[i, 0]) else 0.5 for i in idx if rig[i, 2] > 0]
    return float(np.mean(c)) if c else 0.5


def linked(fj, fc, dj, dc, ratio=(0.75, 1.33), max_deg=25.0):
    """Is the merged bone fj->fc close to the detector's own bone dj->dc (length and direction)? A missing end
    means NOT linked (a limb without its anchor would float)."""
    if np.isnan([fj[0], fj[1], fc[0], fc[1], dj[0], dj[1], dc[0], dc[1]]).any():
        return False
    v, w = np.array(fc[:2]) - fj[:2], np.array(dc[:2]) - dj[:2]
    lv, lw = np.linalg.norm(v), np.linalg.norm(w)
    if lw < 1e-6 or lv < 1e-6:
        return False
    cos = float(np.dot(v, w) / (lv * lw))
    return ratio[0] <= lv / lw <= ratio[1] and cos >= np.cos(np.radians(max_deg))


def complete(p, idx):
    return sum(1 for i in idx if has(p, i))


HAND_GAP = 0.08          # a hand must start within this share of the height from its body wrist


def assign_hands(p, body, h):
    """The detector's hands one-to-one to the merged wrists (L = 7, R = 4), the pair of sides with the smaller total
    distance; a hand must start within HAND_GAP * h of its wrist, and a missing wrist takes no hand."""
    hands = [x for x in (p["L"], p["R"]) if x is not None and has(x, 0)]
    ok = lambda hand, sd: has(body, 7 if sd == "L" else 4) and         ds._dist(hand[0], body[7 if sd == "L" else 4]) <= HAND_GAP * h  # noqa: E731
    best = None                                   # the pairing with the most attached hands, then the smallest gaps
    for perm in permutations(("L", "R", None, None)[:max(2, len(hands)) + len(hands)], len(hands)):
        if any(sd is not None and not ok(hand, sd) for hand, sd in zip(hands, perm)):
            continue
        if len([sd for sd in perm if sd]) != len({sd for sd in perm if sd}):
            continue
        kept = sum(1 for sd in perm if sd)
        cost = sum(ds._dist(hand[0], body[7 if sd == "L" else 4]) for hand, sd in zip(hands, perm) if sd)
        if best is None or (kept, -cost) > (best[0], -best[1]):
            best = (kept, cost, perm)
    out = {"L": None, "R": None}
    for hand, sd in zip(hands, best[2] if best else ()):
        if sd:
            out[sd] = hand
    return out


def hand_cost(hand, rig_hand, h):
    if hand is None:
        return 1.0
    if rig_hand is None:
        return 0.5
    rh = np.asarray(rig_hand, float)
    d = [min(ds._dist(hand[i], rh[i]) / h, 0.5) if has(hand, i) else 0.5 for i in range(21)]
    return float(np.mean(d))


LIMBS = {"arm_r": (2, 3, 4), "arm_l": (5, 6, 7), "leg_r": (8, 9, 10), "leg_l": (11, 12, 13)}


def connect(body, rig, src):
    """The final check: no point is drawn without its link. In a limb (anchor, middle, end) a missing anchor drops
    the limb and a missing middle drops the end; in the head a missing nose drops the eyes - except the OpenPose back
    view, where the rig hides the nose and eyes and the ears may stand alone."""
    for part, (j, m, e) in LIMBS.items():
        if not has(body, j):
            if has(body, m) or has(body, e):
                src["dropped_" + part] = True
            body[[m, e]] = np.nan
        elif not has(body, m) and has(body, e):
            body[e] = np.nan
            src["dropped_" + part] = True
    if not has(body, 1):                          # no neck: the head has nothing to hang on
        body[[0, 14, 15, 16, 17]] = np.nan
        return
    if not has(body, 0) and rig[0, 2] > 0:        # a front/side view without a nose: no lone eyes or ears
        body[[14, 15, 16, 17]] = np.nan
    for eye, ear in ((14, 16), (15, 17)):
        if not has(body, eye) and rig[0, 2] > 0:
            body[ear] = np.nan


def fuse(a, b, h, rig, choose, owner, rig_hands=None):
    """One character in one frame. rig: the true body (18 x 3); rig_hands: {"L"/"R": 21 x 3} or None. The rig is used
    by choose == "rig" (choosing, left/right) and by the face rule (no face where the rig's head faces away)."""
    src = {}
    rig = np.asarray(rig, float)
    rig_face_shown = rig[0, 2] > 0
    if a is None or b is None:                            # one detector only: its skeleton, without a face
        p = a if a is not None else b
        if p is None:
            return None, {"absent": True}
        p, _ = orient(p, rig)
        q = dict(p)
        q["body"] = p["body"].copy()
        q["body"][list(PARTS["head"])] = np.nan
        connect(q["body"], rig, {})
        q["face"] = None
        hands = assign_hands(p, q["body"], h)
        q["L"], q["R"] = hands["L"], hands["R"]
        return q, {"whole": "a" if a is not None else "b"}
    a, b, relabelled = align(a, b, rig if choose == "rig" else None)
    src["relabelled"] = relabelled
    tol = 0.05 * h
    body = np.full((18, 3), np.nan)
    pick = {}
    for part, idx in PARTS.items():
        idx = list(idx)
        if choose == "avg":
            for i in idx:
                pa, pb = a["body"][i], b["body"][i]
                if has(a["body"], i) and has(b["body"], i) and ds._dist(pa, pb) < tol:
                    body[i, :2], body[i, 2] = (pa[:2] + pb[:2]) / 2, max(pa[2], pb[2])
                else:
                    body[i] = pa if (owner[OWNER_KEYS[part]] == "a" or not has(b["body"], i)) and has(a["body"], i) else pb
            src[part] = "avg"
            continue
        if choose == "rig":
            ca, cb = part_cost(a["body"], rig, idx, h), part_cost(b["body"], rig, idx, h)
            pick[part] = "a" if ca <= cb else "b"
        else:
            pick[part] = owner[OWNER_KEYS[part]]
            if complete((a if pick[part] == "a" else b)["body"], idx) == 0:
                pick[part] = "b" if pick[part] == "a" else "a"     # the owner has nothing for this part
        body[idx] = (a if pick[part] == "a" else b)["body"][idx]
        src[part] = pick[part]
    # the head: the face rule of the rig first (no nose or eyes where the rig's head faces away; ears may stay),
    # then the head points must agree between the two detectors (a face both see)
    for i in PARTS["head"]:
        if i in (0, 14, 15) and not (rig[i, 2] > 0):
            body[i] = np.nan
        elif not (has(a["body"], i) and has(b["body"], i) and ds._dist(a["body"][i], b["body"][i]) <= tol):
            body[i] = np.nan
    if choose != "avg":                                   # every part stays connected to the torso
        for part, (j, c) in JOINS.items():
            own, tsrc = pick.get(part), pick.get("torso")
            if part == "head":                            # the head: its neck-nose bone must stay linked
                own_det = a if own == "a" else b
                if own != tsrc and has(body, 0) and not linked(body[1], body[0], own_det["body"][1], own_det["body"][0]):
                    body[[0, 14, 15]] = np.nan
                    src["dropped_head"] = True
                continue
            options = [own] + ([tsrc] if tsrc != own else [])
            chosen = None
            for o in options:
                det = a if o == "a" else b
                if complete(det["body"], PARTS[part]) == 0:
                    continue
                child = det["body"][c]
                if o == tsrc or linked(body[j], child, det["body"][j], child):
                    if has(body, j):                      # the anchor on the torso must exist
                        chosen = o
                        break
            idx = list(PARTS[part])
            if chosen is None:
                body[idx] = np.nan                        # nothing connects: no floating limb
                src["dropped_" + part] = True
            else:
                body[idx] = (a if chosen == "a" else b)["body"][idx]
                if chosen != own:
                    src["relinked_" + part] = True
                src[part] = chosen
    connect(body, rig, src)
    out = {"body": body}
    ha, hb = assign_hands(a, body, h), assign_hands(b, body, h)
    for s in ("L", "R"):
        if choose == "rig":
            rh = (rig_hands or {}).get(s)
            pick_h = "a" if hand_cost(ha[s], rh, h) <= hand_cost(hb[s], rh, h) else "b"
        else:
            pick_h = owner["hands"]
        hand = ha[s] if pick_h == "a" else hb[s]
        if hand is None:                                  # the chosen detector has no attached hand: the other's
            pick_h = "b" if pick_h == "a" else "a"
            hand = ha[s] if pick_h == "a" else hb[s]
        out[s] = hand
        src["hand_" + s] = pick_h if hand is not None else None
    # the dense face: only where the rig's face looks at the camera and BOTH detectors give a face; the points come
    # from ONE detector (the head's donor, else the face owner), never an average
    donor = src.get("head") if src.get("head") in ("a", "b") else owner.get("face", "a")
    fa, fb = a["face"], b["face"]
    if rig_face_shown and fa is not None and fb is not None:
        f = (fa if donor == "a" else fb).copy()
        other = fb if donor == "a" else fa
        for i in range(70):
            if not (has(f, i) and has(other, i) and ds._dist(f[i], other[i]) < tol):
                f[i] = np.nan
        out["face"] = None if np.isnan(f[:, 0]).all() else f
        src["face"] = donor
    else:
        out["face"] = None
    return out, src


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--passes", required=True)
    ap.add_argument("--a", required=True, help="dets of detector a (DWPose)")
    ap.add_argument("--b", required=True, help="dets of detector b (SDPose)")
    ap.add_argument("--choose", choices=("owner", "rig", "avg"), default="rig")
    ap.add_argument("--owner", default="torso=b,arms=b,legs=b,head=a,hands=a,face=a")
    ap.add_argument("--out")
    ap.add_argument("--save-fused", help="the merged skeletons (with the source of each part), JSON")
    ap.add_argument("--control", help="folder for the merged control pictures (needs ComfyUI's Python)")
    a = ap.parse_args(argv)
    owner = dict(kv.split("=") for kv in a.owner.split(","))
    p = Path(a.passes)
    kp = json.loads((p / "keypoints.json").read_text())
    info = json.loads((p / "passes.json").read_text())
    boxes = json.loads((p / "boxes.json").read_text())
    az = info.get("camera_azimuth_deg") or [0.0] * info["frames"]
    da, db = load(a.a), load(a.b)
    if set(da) != set(db):
        sys.exit(f"the two dets files cover different frames: {sorted(set(da) ^ set(db))[:10]}")
    rows, frames, saved, sources = [], {}, {}, {}
    for f in sorted(da, key=int):
        i = int(f) - info["frame_start"]
        bframe = [b if b and b[2] - b[0] >= 4 and b[3] - b[1] >= 4 else None for b in boxes["frames"][f]]
        truths = ds.truth(kp, i, bframe, info.get("character_azimuth_deg"))
        found = 0
        for t in truths:
            if not t["box"]:
                continue
            fused, src = fuse(da[f].get(t["name"]), db[f].get(t["name"]), max(t["h"], 1.0), t["body"], a.choose, owner,
                              {"L": t["L"], "R": t["R"]})
            found += fused is not None
            for k, v in src.items():
                if k.startswith("relinked_") or k.startswith("dropped_"):
                    key = k.split("_")[0]
                    sources[key] = sources.get(key, 0) + 1
                elif k != "relabelled":
                    sources[f"{k}={v}"] = sources.get(f"{k}={v}", 0) + 1
            if a.save_fused:
                saved.setdefault(f, {})[t["name"]] = None if fused is None else {
                    **{k: (None if v is None else np.where(np.isnan(v), None, v).tolist()) for k, v in fused.items()},
                    "source": src}
            r = ds.score_person(t, fused)
            caz = t["az"] if t["az"] is not None else az[i]
            r.update(frame=int(f), az=caz, side=ds.side(caz), name=t["name"])
            rows.append(r)
        frames[int(f)] = {"people": found, "expected": sum(1 for t in truths if t["box"]), "side": ds.side(az[i]),
                          "yolox_person": None}
    result = {"all": {**ds.summary(rows, frames), "choose": a.choose, "sources": dict(sorted(sources.items()))}}
    for s in ("front", "left", "back", "right"):
        sel = [r for r in rows if r["side"] == s]
        result[s] = ds.summary(sel, {f: frames[f] for f in sorted({r["frame"] for r in sel})})
    if a.save_fused:
        Path(a.save_fused).write_text(json.dumps({"choose": a.choose, "owner": owner, "frames": saved}))
    if a.control:
        draw_controls(saved if a.save_fused else None, a, p, da, db, info)
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": result, "rows": rows, "settings": vars(a)}, indent=1, default=float))
    print(json.dumps(result))


def draw_controls(saved, a, p, da, db, info):
    """The merged skeletons drawn as the DWPose node draws them (one picture per frame)."""
    if saved is None:
        sys.exit("--control needs --save-fused")
    import cv2
    sys.path.insert(0, str(ds.AUX / "src"))
    from custom_controlnet_aux.dwpose import draw_poses
    from custom_controlnet_aux.dwpose.types import BodyResult, Keypoint, PoseResult
    out = Path(a.control)
    out.mkdir(parents=True, exist_ok=True)
    H, W = info["height"], info["width"]

    def kps(v):
        return None if v is None else [None if x[0] is None else Keypoint(x[0], x[1], x[2] or 1.0) for x in v]
    for f, chars in saved.items():
        poses = [PoseResult(BodyResult(kps(c["body"]), 0.0, 18), kps(c["L"]), kps(c["R"]), kps(c["face"]))
                 for c in chars.values() if c]
        canvas = draw_poses(poses, H, W, draw_body=True, draw_hand=True, draw_face=True)
        cv2.imwrite(str(out / f"rgb_{int(f):04d}.png"), canvas[:, :, ::-1])


if __name__ == "__main__":
    main()
