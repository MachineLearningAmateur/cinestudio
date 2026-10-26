"""Funk dance template: one mannequin does an original Locking-style 8-count (1970s funk: points, wrist rolls, claps,
lock freezes, kicks) over a constant knee bounce, timed to a beat grid. The camera arcs slowly in front of it.

    blender -b --factory-startup --python templates/funk_dance/build.py -- --out shot.blend
        [--bpm 120] [--fps 24] [--frames 121] [--first-beat 13] [--beats "13,25,..."] [--night]

The moves sit on the beats: `--beats` gives exact beat frames (from the song's beat detection); otherwise a regular grid
from `--first-beat` at `--bpm`. Every move is a real, human move: points, a lock (arms bent, fists up), wrist rolls,
claps, knee kicks; the feet stay on the floor except in the kicks (two-bone leg IK keeps them planted while the knees
bend). Built from code only, like templates/character_showcase.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "character_showcase"))
import build as showcase  # noqa: E402
from blender_lib import mannequin as mq  # noqa: E402

CAM_RADIUS, CAM_Z, CAM_TARGET_Z = 3.6, 1.30, 0.95
ARC = 24.0                 # degrees the camera travels (well under 0.6 degrees per frame)
BOUNCE = 0.085             # metres the hips drop on each beat
SWAY = 0.035               # metres the hips sway to the side, left and right on alternate beats
THIGH, SHIN = 0.43, 0.43   # mannequin leg bone lengths (blender_lib/mannequin.py REST)
FIST, OPEN, LOOSE = (1.0, 0.0), (0.05, 8.0), (0.35, 2.0)


def V(*c):
    return Vector(c).normalized()


def mirror(d):
    return Vector((-d.x, d.y, d.z))


def arms(r_upper, r_fore, l_upper=None, l_fore=None):
    """Arm directions; the right side is given (character's right = -X, forward = -Y); left mirrors when omitted."""
    l_upper = l_upper if l_upper is not None else mirror(r_upper)
    l_fore = l_fore if l_fore is not None else mirror(r_fore)
    return {"upper_arm.R": r_upper, "forearm.R": r_fore, "hand.R": r_fore,
            "upper_arm.L": l_upper, "forearm.L": l_fore, "hand.L": l_fore}


# ---- arm poses (right side; mirrored for the left unless both are given)
RELAX = (V(-0.15, 0.0, -1), V(-0.10, -0.25, -1))
HIP = (V(-0.55, 0.25, -0.80), V(0.60, -0.30, -0.50))              # hand on hip, elbow out
POINT_FWD = (V(-0.60, -0.50, 0.62), V(-0.62, -0.48, 0.62))       # the point: up and out, toward the camera side
RECOIL = (V(-0.25, 0.05, -0.95), V(0.45, -0.60, 0.65))          # fist pulled back to the chest
SIDE = (V(-0.97, -0.10, 0.05), V(-0.98, -0.10, 0.05))            # arm straight out to the side
ROLL_A = (V(-0.30, -0.55, -0.78), V(0.70, -0.55, 0.40))          # wrist roll: forearm across the chest, high
ROLL_B = (V(-0.30, -0.55, -0.78), V(0.70, -0.62, -0.15))         # ... low
CLAP = (V(-0.20, -0.62, -0.75), V(0.62, -0.72, 0.30))            # both hands meet at the chest
LOCK = (V(-0.95, 0.05, -0.15), V(-0.10, -0.20, 0.97))            # the lock: elbows out, fists up by the head
SWING_F = (V(-0.10, -0.75, -0.65), V(-0.05, -0.85, -0.50))       # arm swing forward (kick step)
SWING_B = (V(-0.10, 0.55, -0.83), V(-0.10, 0.40, -0.90))         # arm swing back
VUP = (V(-0.55, -0.15, 0.82), V(-0.55, -0.15, 0.82))             # arms up in a V
KNEES = (V(-0.10, -0.20, -1.0), V(0.10, -0.10, -1.0))            # hands on knees (with the deep forward lean)


def pose(arm_pair_r, arm_pair_l=None, *, spine=V(0, 0, 1), chest=None, head=None, hands=(LOOSE, LOOSE),
         drop=0.0, sway=0.0, feet=None):
    """One upper-body pose + extra hip drop/sway and optional foot targets."""
    l_up, l_fo = (mirror(arm_pair_r[0]), mirror(arm_pair_r[1])) if arm_pair_l is None else arm_pair_l
    return {"arms": arms(arm_pair_r[0], arm_pair_r[1], l_up, l_fo), "spine": spine, "chest": chest or spine,
            "head": head or V(0, -0.05, 1), "hands": hands, "drop": drop, "sway": sway, "feet": feet or {}}


def l_side(pair):
    return (mirror(pair[0]), mirror(pair[1]))


# ---- more arm poses (right side)
STEP_R = (V(-0.60, -0.20, -0.78), V(-0.65, -0.30, -0.70))       # both arms swing low toward the stepping side ...
STEP_L_CROSS = (V(-0.25, -0.50, -0.83), V(-0.55, -0.45, -0.70)) # ... the other arm crosses the body (given for .L)
SHIMMY = (V(-0.35, 0.15, -0.92), V(0.10, -0.95, 0.12))          # elbows bent at the ribs, fists forward
TWIRL_A = (V(-0.22, -0.30, 0.93), V(0.80, -0.45, 0.40))         # wrist twirl above the head, high ...
TWIRL_B = (V(-0.22, -0.30, 0.93), V(0.80, -0.55, 0.05))        # ... low

FEET_OPEN = {".R": Vector((-0.22, 0.0, 0.09)), ".L": Vector((0.22, 0.0, 0.09))}


def mirror_pose(p):
    """The same pose with left and right swapped (a mirrored routine)."""
    arms_m = {}
    for k, v in p["arms"].items():
        other = k[:-2] + (".L" if k.endswith(".R") else ".R")
        arms_m[other] = mirror(v)
    feet = {(".L" if s == ".R" else ".R"): mirror(v) for s, v in p["feet"].items()}
    return {**p, "arms": arms_m, "spine": mirror(p["spine"]), "chest": mirror(p["chest"]), "head": mirror(p["head"]),
            "hands": (p["hands"][1], p["hands"][0]), "sway": -p["sway"], "feet": feet}


def routine_a(b):
    """The first 8-count (Locking): point, recoil + side arm, wrist rolls, clap, LOCK (count 5), side kicks R and L, V."""
    half = lambda i: (b[i] + b[i + 1]) // 2        # noqa: E731 - the "and" between two counts
    kick_r = {".R": Vector((-0.40, -0.30, 0.38))}      # kicks go out to the side (they read from the front)
    kick_l = {".L": Vector((0.40, -0.30, 0.38))}
    knee_r = {".R": Vector((-0.20, -0.22, 0.32))}
    knee_l = {".L": Vector((0.20, -0.22, 0.32))}
    return [
        (b[0], pose(POINT_FWD, l_side(HIP), spine=V(0, -0.18, 1), head=V(0, -0.10, 1), hands=(OPEN, LOOSE), drop=0.03), 3),
        (half(0), pose(RECOIL, l_side(HIP), spine=V(0, -0.05, 1), hands=(FIST, LOOSE)), 1),
        (b[1], pose(RECOIL, l_side(SIDE), spine=V(0.12, -0.05, 1), head=V(0.15, -0.05, 1), hands=(FIST, OPEN), sway=0.05), 3),
        (b[2], pose(ROLL_A, l_side(ROLL_B), spine=V(0, -0.10, 1), hands=(FIST, FIST)), 0),
        (b[2] + 3, pose(ROLL_B, l_side(ROLL_A), spine=V(0, -0.10, 1), hands=(FIST, FIST)), 0),
        (b[2] + 6, pose(ROLL_A, l_side(ROLL_B), spine=V(0, -0.10, 1), hands=(FIST, FIST)), 0),
        (b[2] + 9, pose(ROLL_B, l_side(ROLL_A), spine=V(0, -0.10, 1), hands=(FIST, FIST)), 0),
        (b[3], pose(CLAP, spine=V(0, -0.22, 1), head=V(0, -0.20, 1), hands=(OPEN, OPEN), drop=0.10), 2),
        (b[4], pose(LOCK, spine=V(0, 0.12, 1), head=V(0.0, 0.05, 1), hands=(FIST, FIST), drop=0.04), 6),
        (b[5], pose(SWING_B, l_side(SWING_F), spine=V(0, 0.05, 1), hands=(LOOSE, LOOSE), feet=knee_r), 0),
        (b[5] + 4, pose(SWING_B, l_side(SWING_F), spine=V(0, 0.10, 1), hands=(LOOSE, LOOSE), feet=kick_r), 2),
        (b[6], pose(SWING_F, l_side(SWING_B), spine=V(0, 0.05, 1), hands=(LOOSE, LOOSE), feet=knee_l), 0),
        (b[6] + 4, pose(SWING_F, l_side(SWING_B), spine=V(0, 0.10, 1), hands=(LOOSE, LOOSE), feet=kick_l), 2),
        (b[7], pose(VUP, spine=V(0, 0.10, 1), head=V(0, 0.12, 1), hands=(OPEN, OPEN), drop=-0.01, feet=FEET_OPEN), 6),
    ]


def routine_b(b):
    """The groove 8-count: step-touch R + clap, step-touch L + clap, shoulder shimmy (2 counts), point left, twirls."""
    step = pose(STEP_R, (STEP_L_CROSS[0], STEP_L_CROSS[1]), spine=V(-0.10, -0.08, 1), head=V(-0.12, -0.08, 1),
                hands=(OPEN, OPEN), sway=-0.07, drop=0.03, feet={".R": Vector((-0.32, 0.0, 0.09))})
    clap = pose(CLAP, spine=V(0, -0.15, 1), head=V(0, -0.12, 1), hands=(OPEN, OPEN), drop=0.07)
    keys = [(b[0], step, 2), (b[1], clap, 2), (b[2], mirror_pose(step), 2), (b[3], clap, 2)]
    span = b[6] - b[4]
    for i in range(8):                               # the shimmy: the chest rocks left and right 8 times in 2 counts
        x = 0.28 if i % 2 == 0 else -0.28
        keys.append((b[4] + round(i * span / 8), pose(SHIMMY, spine=V(0, 0.08, 1), chest=V(x, 0.10, 1),
                                                      head=V(x * 0.4, 0.05, 1), hands=(FIST, FIST), drop=0.06), 0))
    keys.append((b[6], mirror_pose(pose(POINT_FWD, l_side(HIP), spine=V(0, -0.18, 1), head=V(0, -0.10, 1),
                                        hands=(OPEN, LOOSE), drop=0.03)), 5))
    span = (b[8] if len(b) > 8 else b[7] + (b[7] - b[6])) - b[7]
    for i in range(4):
        a_, b_ = (TWIRL_A, TWIRL_B) if i % 2 == 0 else (TWIRL_B, TWIRL_A)
        keys.append((b[7] + round(i * span / 4), pose(a_, l_side(b_), spine=V(0, 0.08, 1), head=V(0, 0.10, 1),
                                                      hands=(FIST, FIST)), 0))
    return keys


def routine_f(b):
    """The finale: hands on knees, point, clap, lock, and a V held to the end."""
    knees = pose(KNEES, spine=V(0, -0.85, 1), chest=V(0, -0.95, 1), head=V(0, -0.55, 1), hands=(OPEN, OPEN), drop=0.16,
                 feet=FEET_OPEN)
    seq = [(knees, 4),
           (pose(POINT_FWD, l_side(HIP), spine=V(0, -0.10, 1), head=V(0, -0.10, 1), hands=(OPEN, LOOSE), drop=0.02), 4),
           (pose(CLAP, spine=V(0, -0.22, 1), head=V(0, -0.20, 1), hands=(OPEN, OPEN), drop=0.10), 2),
           (pose(LOCK, spine=V(0, 0.12, 1), head=V(0.0, 0.05, 1), hands=(FIST, FIST), drop=0.04), 6),
           (pose(VUP, spine=V(0, 0.10, 1), head=V(0, 0.12, 1), hands=(OPEN, OPEN), drop=-0.01, feet=FEET_OPEN), 999)]
    return [(b[i], p_, h) for i, (p_, h) in enumerate(seq) if i < len(b)]


# ---- the shuffle (running man): feet per half beat; arms punch in opposition to the lifted knee
PUNCH = (V(-0.45, -0.50, 0.30), V(-0.45, -0.45, 0.77))          # fist punches up and out on a diagonal
PULL = (V(-0.35, 0.25, -0.90), V(0.35, -0.25, -0.90))           # the other fist pulled down to the hip
CLAP_LOW = (V(-0.15, -0.55, -0.82), V(0.35, -0.55, -0.75))      # hands meet low in front (under a lifted knee)
DOUBLE_PUNCH = (V(-0.10, -0.97, 0.05), V(-0.05, -0.99, 0.05))    # both fists punch straight forward


def run_man(state):
    """The four half-beat states of the running man. A: R steps down, L slid back; B: L knee up, R slides back, R arm
    punches; C: L down, R back; D: R knee up, L slides back, L arm punches."""
    feet = {"A": {".R": Vector((-0.12, -0.05, 0.09)), ".L": Vector((0.12, 0.17, 0.12))},
            "B": {".R": Vector((-0.12, 0.10, 0.09)), ".L": Vector((0.12, -0.20, 0.44))},
            "C": {".R": Vector((-0.12, 0.17, 0.12)), ".L": Vector((0.12, -0.05, 0.09))},
            "D": {".R": Vector((-0.12, -0.20, 0.44)), ".L": Vector((0.12, 0.10, 0.09))}}[state]
    arm_r = {"A": RELAX, "B": PUNCH, "C": RELAX, "D": PULL}[state]
    arm_l = l_side({"A": RELAX, "B": PULL, "C": RELAX, "D": PUNCH}[state])
    return pose(arm_r, arm_l, spine=V(0, -0.12, 1), head=V(0, -0.08, 1), hands=(FIST, FIST), feet=feet,
                drop=0.05 if state in ("A", "C") else 0.0)


def lock_hit():
    """The lock freeze of the shuffle routines (fists up beside the head, a small dip, feet apart)."""
    return pose(LOCK, spine=V(0, 0.12, 1), head=V(0.0, 0.05, 1), hands=(FIST, FIST), drop=0.06, feet=FEET_OPEN)


def routine_s1(b):
    """Shuffle 8-count 1: running man x4 (arms punch in sync), side kick R (arms swing left), side kick L (arms swing
    right), high knee R with a clap under it, LOCK on 8 (held across the next count 1)."""
    h = lambda i: (b[i] + b[i + 1]) // 2           # noqa: E731
    keys = []
    for i in range(4):
        keys += [(b[i], run_man("A" if i % 2 == 0 else "C"), 0, 99),
                 (h(i), run_man("B" if i % 2 == 0 else "D"), 0, 99)]
    swing_left = (V(0.45, -0.45, -0.77), V(0.60, -0.45, -0.66))      # right arm swings across to the left ...
    left_out = (V(0.55, -0.25, -0.80), V(0.75, -0.25, -0.62))         # ... the left arm swings out to the left
    kick_r = pose(swing_left, left_out, spine=V(0.10, 0.05, 1), head=V(0.05, -0.05, 1), hands=(OPEN, OPEN), sway=0.03,
                  feet={".R": Vector((-0.42, -0.25, 0.40))})
    keys += [(b[4], kick_r, 3),
             (h(4), pose(RELAX, hands=(LOOSE, LOOSE), feet={".L": Vector((0.20, -0.20, 0.32))}, drop=0.03), 0),
             (b[5], mirror_pose(kick_r), 3),
             (b[6], pose(CLAP_LOW, spine=V(0, -0.35, 1), head=V(0, -0.30, 1), hands=(OPEN, OPEN),
                         feet={".R": Vector((-0.10, -0.28, 0.50))}), 3),
             (b[7], lock_hit(), max(0, (b[8] - b[7]) - 4))]  # held until 4 frames before the next count 1, so
    return keys                                               # what follows (S2's held lock, S1, the loop end) winds in


def routine_s2(b):
    """Shuffle 8-count 2: (1 = the lock held), point R, point L, clap with a dip, step-touch R and L with arm swings
    and claps, front kick R with a double punch, back into the running man (the loop's first pose)."""
    h = lambda i: (b[i] + b[i + 1]) // 2           # noqa: E731
    point_r = pose(POINT_FWD, l_side(HIP), spine=V(0, -0.15, 1), head=V(-0.10, -0.10, 1), hands=(OPEN, LOOSE),
                   sway=-0.05, drop=0.03)
    recoil = pose(RECOIL, l_side(HIP), spine=V(0, -0.05, 1), hands=(FIST, LOOSE))
    clap = pose(CLAP, spine=V(0, -0.20, 1), head=V(0, -0.18, 1), hands=(OPEN, OPEN), drop=0.10)
    step = pose(STEP_R, (STEP_L_CROSS[0], STEP_L_CROSS[1]), spine=V(-0.10, -0.08, 1), head=V(-0.12, -0.08, 1),
                hands=(OPEN, OPEN), sway=-0.07, drop=0.03, feet={".R": Vector((-0.32, 0.0, 0.09))})
    touch = pose(CLAP, spine=V(0, -0.12, 1), hands=(OPEN, OPEN), drop=0.06)
    kick = pose(DOUBLE_PUNCH, spine=V(0, 0.10, 1), hands=(FIST, FIST), feet={".R": Vector((-0.12, -0.45, 0.45))})
    return [(b[0], lock_hit(), 6),                     # count 1: the lock held (after S1 it spans the clip join)
            (b[1], point_r, 3), (h(1), recoil, 0), (b[2], mirror_pose(point_r), 3), (h(2), mirror_pose(recoil), 0),
            (b[3], clap, 2),
            (b[4], step, 2), (h(4), touch, 1), (b[5], mirror_pose(step), 2), (h(5), touch, 1),
            (b[6], kick, 3),
            (b[7], run_man("C"), 0), (h(7), run_man("D"), 0, 99)]


# ---- the horse-riding party dance (original moves in a well-known party style: the gallop step with the wrists crossed
#      as if holding reins, a lasso arm over the head, side gallops, fist pumps, a "hey" pose)
REINS_R = (V(-0.20, -0.55, -0.81), V(0.80, -0.45, 0.40))       # right forearm across the chest (wrists crossed)
REINS_L = (V(0.20, -0.55, -0.81), V(-0.80, -0.38, 0.47))       # left forearm across, a little higher (on top)
HEY = (V(-0.50, -0.10, 0.86), V(-0.42, -0.10, 0.90))           # arm up in a V, fist
PUMP = (V(-0.30, -0.35, 0.89), V(-0.25, -0.30, 0.92))          # fist pumped straight up
PUMP_DOWN = (V(-0.35, -0.10, -0.93), V(0.40, -0.55, 0.73))     # fist pulled down to the chest


def gallop(state, side=0.0, lasso=None):
    """The four half-beat states of the gallop: A = R steps down (L heel kicked back), B = hop (R knee up, L down),
    C and D = the same on the other foot. `side` moves both feet sideways (a side gallop, metres); `lasso` (angle in
    degrees) puts the right arm over the head with the forearm at that angle of the circle."""
    import math
    feet = {"A": {".R": Vector((-0.13 + side, -0.06, 0.09)), ".L": Vector((0.13 + side, 0.14, 0.24))},
            "B": {".R": Vector((-0.13 + side, -0.16, 0.27)), ".L": Vector((0.13 + side, 0.02, 0.09))},
            "C": {".R": Vector((-0.13 + side, 0.14, 0.24)), ".L": Vector((0.13 + side, -0.06, 0.09))},
            "D": {".R": Vector((-0.13 + side, 0.02, 0.09)), ".L": Vector((0.13 + side, -0.16, 0.27))}}[state]
    if lasso is None:
        arm_r = REINS_R
    else:
        a = math.radians(lasso)
        arm_r = (V(-0.25, -0.10, 0.96), V(0.78 * math.cos(a), -0.78 * math.sin(a), 0.62))
    return pose(arm_r, REINS_L, spine=V(-side * 0.6, -0.10, 1), head=V(0, -0.08, 1), hands=(FIST, FIST), feet=feet,
                drop=0.07 if state in ("A", "C") else 0.0, sway=-side * 0.5)


def routine_h1(b):
    """Horse 8-count 1: the gallop in place with crossed wrists (counts 1-4), then the gallop with the right arm
    swinging a lasso over the head, one full circle per two counts (counts 5-8)."""
    h = lambda i: (b[i] + b[i + 1]) // 2           # noqa: E731
    keys = []
    for i in range(8):
        lasso = None if i < 4 else (i - 4) * 180.0
        keys += [(b[i], gallop("A" if i % 2 == 0 else "C", lasso=lasso), 0, 99),
                 (h(i), gallop("B" if i % 2 == 0 else "D", lasso=None if lasso is None else lasso + 90.0), 0, 99)]
    return keys


def routine_h2(b):
    """Horse 8-count 2: side gallop to the right (counts 1-2) and to the left (3-4), fist pumps in a deep bounce
    (5-6), the "hey" pose with both arms up (7), back into the gallop with the reins (8: the loop's first pose)."""
    h = lambda i: (b[i] + b[i + 1]) // 2           # noqa: E731
    keys = []
    for i, side in ((0, -0.10), (1, -0.20), (2, -0.10), (3, 0.0)):
        keys += [(b[i], gallop("A" if i % 2 == 0 else "C", side=side), 0, 99),
                 (h(i), gallop("B" if i % 2 == 0 else "D", side=side + (-0.05 if i < 2 else 0.05)), 0, 99)]
    pump_r = pose(PUMP, l_side(PUMP_DOWN), spine=V(-0.08, -0.12, 1), head=V(0, -0.10, 1), hands=(FIST, FIST), drop=0.12,
                  feet=FEET_OPEN)
    keys += [(b[4], pump_r, 1), (h(4), pose(PUMP_DOWN, spine=V(0, -0.15, 1), hands=(FIST, FIST), drop=0.05,
                                             feet=FEET_OPEN), 0),
             (b[5], mirror_pose(pump_r), 1), (h(5), pose(PUMP_DOWN, spine=V(0, -0.15, 1), hands=(FIST, FIST), drop=0.05,
                                                         feet=FEET_OPEN), 0),
             (b[6], pose(HEY, spine=V(0, 0.15, 1), head=V(0, 0.12, 1), hands=(FIST, FIST), drop=0.02, feet=FEET_OPEN), 4),
             (b[7], gallop("A"), 0), (h(7), gallop("B"), 0, 99)]
    return keys


ROUTINES = {"A": (routine_a, 8), "B": (routine_b, 8), "F": (routine_f, 5), "S1": (routine_s1, 8), "S2": (routine_s2, 8),
            "H1": (routine_h1, 8), "H2": (routine_h2, 8)}


def choreography(beats, routines="A,B,A~,A,F", loop=False):
    """[(frame, pose, hold_frames)]: frame 1 stands relaxed, then the routines one after the other from beats[0] (count
    1). "X~" is routine X mirrored. A routine that does not fit in the beats left is cut (the finale may be partial)."""
    keys = [] if loop else [(1, pose(RELAX, head=V(0, -0.15, 1)), 0)]
    i = 0
    for name in [r.strip() for r in routines.split(",") if r.strip()]:
        fn, n = ROUTINES[name.rstrip("~")]
        if i >= len(beats) or (name.rstrip("~") != "F" and i + n > len(beats)):
            break
        part = fn(beats[i:])
        if name.endswith("~"):
            part = [(k[0], mirror_pose(k[1])) + tuple(k[2:]) for k in part]
        keys += part
        i += n
    keys.sort(key=lambda k: k[0])
    if loop:                                       # the loop ends on its first pose (beat i = the next count 1)
        if i >= len(beats):
            raise ValueError("--loop needs one more beat after the last routine (the loop's end)")
        period = beats[i] - beats[0]
        # the next cycle again after the end: frames past the loop (a tail) repeat the first frames, so a video made
        # from the shot can be cross-faded into its own start (a seamless loop)
        keys += [(k[0] + period,) + tuple(k[1:]) for k in keys]
        keys.sort(key=lambda k: k[0])
    return keys


def lerp_dir(a, b, t):
    v = a.lerp(b, t)
    return v.normalized() if v.length > 1e-6 else b


def blend(p, q, t):
    """Pose between p and q (t in 0..1): directions normalized-lerped, numbers lerped, hands by the nearer one."""
    out = {"arms": {k: lerp_dir(p["arms"][k], q["arms"][k], t) for k in p["arms"]}}
    for k in ("spine", "chest", "head"):
        out[k] = lerp_dir(p[k], q[k], t)
    out["hands"] = tuple((a[0] + (c[0] - a[0]) * t, a[1] + (c[1] - a[1]) * t) for a, c in zip(p["hands"], q["hands"]))
    out["drop"] = p["drop"] + (q["drop"] - p["drop"]) * t
    out["sway"] = p["sway"] + (q["sway"] - p["sway"]) * t
    feet = {}
    for s in (".R", ".L"):
        a, c = p["feet"].get(s), q["feet"].get(s)
        if a is not None or c is not None:
            base = Vector((0.13 * (-1 if s == ".R" else 1), 0.0, 0.09))
            feet[s] = (a or base).lerp(c or base, t)
    out["feet"] = feet
    return out


def pose_at(f, keys, wind=4):
    """The pose at frame f: each key is reached `wind` frames after leaving the previous hold (a sharp hit), held for
    its hold frames, then the next move starts."""
    for i, (kf, p, hold, *opt) in enumerate(keys):
        nxt = keys[i + 1] if i + 1 < len(keys) else None
        if f <= kf + hold or nxt is None:
            if i == 0 or f >= kf:
                return p
            pk, pp, ph = keys[i - 1][:3]
            start = max(pk + ph, kf - (opt[0] if opt else wind))   # a 4th value: this key's own wind (99 = glide)
            if f <= start:
                return pp
            t = (f - start) / max(1, kf - start)
            t = t * t * (3 - 2 * t)                     # ease in and out
            return blend(pp, p, t)
    return keys[-1][1]


def bounce(f, beats, fps):
    """The constant funk bounce: down on every beat, up on every 'and' (a smooth wave between the beats)."""
    if f <= beats[0]:
        prev, nxt = beats[0] - (beats[1] - beats[0]), beats[0]
    else:
        prev = max(b for b in beats if b <= f)
        later = [b for b in beats if b > f]
        nxt = later[0] if later else prev + (beats[-1] - beats[-2])
    t = (f - prev) / max(1, nxt - prev)
    return BOUNCE * (0.5 + 0.5 * math.cos(2 * math.pi * t))     # 1 at each beat, 0 half way


def sway(f, beats):
    """The hips go to one side on each beat and to the other side on the next one (a smooth wave)."""
    step = beats[1] - beats[0]
    return SWAY * math.sin(math.pi * (f - beats[0]) / step)


def leg_ik(hip, ankle, forward=Vector((0, -1, 0))):
    """Knee position for a two-bone leg from hip to ankle (the knee bends forward)."""
    d = ankle - hip
    dist = min(d.length, THIGH + SHIN - 1e-4)
    u = d.normalized()
    cos_a = max(-1.0, min(1.0, (THIGH ** 2 + dist ** 2 - SHIN ** 2) / (2 * THIGH * dist)))
    w = (forward - u * forward.dot(u))
    w = w.normalized() if w.length > 1e-6 else Vector((0, -1, 0))
    return hip + THIGH * (cos_a * u + math.sqrt(max(0.0, 1 - cos_a ** 2)) * w)


def apply(rig, f, p, extra_drop, extra_sway=0.0):
    """Key the full pose at frame f: hips (drop/sway), spine, arms, hands, and IK legs with planted feet."""
    for pb in rig.pose.bones:
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
    hips = rig.pose.bones["hips"]
    delta = Vector((p["sway"] + extra_sway, 0.0, -(p["drop"] + extra_drop)))
    hips.location = hips.bone.matrix_local.to_3x3().inverted() @ delta
    bpy.context.view_layer.update()
    mq.set_bone_dir(rig, "spine", p["spine"])
    mq.set_bone_dir(rig, "chest", p["chest"])
    mq.set_bone_dir(rig, "head", p["head"])
    for name in ("upper_arm.R", "forearm.R", "hand.R", "upper_arm.L", "forearm.L", "hand.L"):
        mq.set_bone_dir(rig, name, p["arms"][name])
    bpy.context.view_layer.update()
    for s, x in ((".R", -1), (".L", 1)):
        ankle = p["feet"].get(s, Vector((0.13 * x, 0.0, 0.09)))
        thigh = rig.pose.bones["thigh" + s]
        hip = rig.matrix_world @ thigh.head
        ankle_w = rig.matrix_world @ ankle
        knee = leg_ik(hip, ankle_w)
        mq.set_bone_dir(rig, "thigh" + s, knee - hip)
        bpy.context.view_layer.update()
        mq.set_bone_dir(rig, "shin" + s, ankle_w - knee)
        bpy.context.view_layer.update()
        rest_foot = Vector(mq.REST["foot" + s][1]) - Vector(mq.REST["foot" + s][0])
        mq.set_bone_dir(rig, "foot" + s, rest_foot)
        bpy.context.view_layer.update()
    mq.curl_fingers(rig, ".R", *p["hands"][0])
    mq.curl_fingers(rig, ".L", *p["hands"][1])
    for pb in rig.pose.bones:
        q = pb.rotation_quaternion.copy()
        key = (rig.name, pb.name)
        if key in mq._LAST_Q and q.dot(mq._LAST_Q[key]) < 0:
            q.negate()
            pb.rotation_quaternion = q
        mq._LAST_Q[key] = q
        pb.keyframe_insert("rotation_quaternion", frame=f)
    hips.keyframe_insert("location", frame=f)


def build(bpm=120.0, fps=24, frames=121, first_beat=13, beats=None, width=1536, height=896, routines="A,B,A~,A,F",
          loop=False, loop_frames=None):
    showcase.ORBIT[0] = 45.0
    sc = showcase.reset()
    sc.frame_start, sc.frame_end = 1, frames
    sc.render.fps, sc.render.fps_base = fps, 1.0
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = width, height, 100
    M = {k: showcase.material(k, v) for k, v in showcase.COLORS.items()}
    showcase.build_set(M)
    if not beats:
        step = 60.0 * fps / bpm
        beats = [round(first_beat + k * step) for k in range(int(frames / step) + 2) if first_beat + k * step <= frames + step]
    rig, _ = mq.build(M)
    keys = choreography(beats, routines, loop)
    for f in range(1, frames + 1, 2):
        apply(rig, f, pose_at(f, keys), bounce(f, beats, fps), sway(f, beats))
    if frames % 2 == 0:
        apply(rig, frames, pose_at(frames, keys), bounce(frames, beats, fps), sway(frames, beats))
    target = bpy.data.objects.new("CamTarget", None)
    target.location = (0, 0, CAM_TARGET_Z)
    sc.collection.objects.link(target)
    cam = bpy.data.objects.new("ShotCam", bpy.data.cameras.new("ShotCam"))
    cam.data.lens = 35
    sc.collection.objects.link(cam)
    sc.camera = cam
    tr = cam.constraints.new("TRACK_TO")
    tr.target, tr.track_axis, tr.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
    period = loop_frames or (frames - 1)
    for f in range(1, frames + 1):
        t = (f - 1) / max(1, frames - 1)
        if loop:
            t = (f - 1) / max(1, period)                  # periodic: the tail past the loop end repeats the start                                           # a loop: a small sway that ends where it starts
            a = math.radians(-6.0 * math.sin(2 * math.pi * t))
            r = CAM_RADIUS - 0.15 * math.sin(math.pi * t) ** 2
        elif frames <= 121:                                # one clip: one slow sweep and a push in
            a = math.radians(-ARC / 2 + ARC * (t * t * (3 - 2 * t)))
            r = CAM_RADIUS - 0.25 * t
        else:                                              # a long shot: a slow sway left and right, in and out
            a = math.radians(-14.0 * math.sin(2 * math.pi * 0.75 * t))
            r = CAM_RADIUS - 0.20 * math.sin(math.pi * t)
        cam.location = (math.sin(a) * r, -math.cos(a) * r, CAM_Z)
        cam.keyframe_insert("location", frame=f)
    showcase.build_lights()
    sc.frame_set(1)
    return sc, beats


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--frames", type=int, default=121)
    ap.add_argument("--first-beat", type=int, default=13, help="frame of count 1 when no --beats are given")
    ap.add_argument("--beats", default="", help="comma-separated beat frames (from the song's beat detection)")
    ap.add_argument("--night", action="store_true")
    ap.add_argument("--loop", action="store_true", help="the last frame repeats the first (pose, bounce, camera)")
    ap.add_argument("--loop-frames", type=int, default=None,
                    help="with --loop: the loop length; frames past it (a tail) repeat the first frames")
    ap.add_argument("--routine", default="A,B,A~,A,F", help="routines in order: A (locking), B (groove), F (finale); X~ = mirrored")
    a = ap.parse_args(argv)
    beats = [int(x) for x in a.beats.split(",") if x.strip()] if a.beats else None
    sc, beats = build(a.bpm, a.fps, a.frames, a.first_beat, beats, routines=a.routine, loop=a.loop,
                      loop_frames=a.loop_frames)
    if a.night:
        showcase.night_set()
    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    print("SAVED", a.out, "beats", beats)


if __name__ == "__main__":
    main()
