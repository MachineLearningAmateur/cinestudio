"""A character's own MANNEQUIN BODY on a mannequin rig (imported by duplicate_mannequin.py and render_turnaround.py).

User, 2026-10-23: "each mannequin needs its own proportions". kitsune (a chibi character: its mask is ~1/4 of its
height) was posed with AnyPose against the full-size adult mannequin, and the posed front took the mannequin's
proportions (a small head, long legs). The body record (character.json "mannequin"):

    {"scale": 0.8, "head": 1.4, "neck": 0.4, "legs": 0.75, "arms": 0.85}   every key optional, 1.0 = the template's

- scale = the whole rig (the object scale, about its origin on the floor);
- head / legs / arms = a UNIFORM pose-bone scale of the head, both thighs (the shins and feet follow) and both upper
  arms (the forearms and hands follow); neck = the neck bone (user: "the mannequin's neck seems too long compared to
  our generated image": kitsune's mask sits on its scarf). The head hangs on the neck, so the head bone gets head / neck
  and the head keeps the size asked; torso = the spine bone (the chest follows: user, "torso of the blender model
  still seems too tall"); the neck and the upper arms get 1 / torso, so the head and the arms keep their own sizes; the
  shoulders narrow with the chest (they stuck out at full width). A uniform torso scale also makes it narrower. Uniform: a non-uniform parent scale shears the turned children. The templates
  key only rotation_quaternion (every bone) and the hips' location, so a pose-bone scale stays for the whole shot;
- the legs' change lowers the rig (its object location) by the height the feet rise in the rest pose, so the feet stay
  on the floor (in a deep knee bend a planted foot sinks by (1 - legs) x the hip drop: ~1.5 cm for 0.85 and 10 cm).
Parts parented to a bone (the mannequin's meshes, the proxy kit) follow its scale.
"""
import json
import os

import bpy
from mathutils import Vector

# the body math lives in mannequin_rules.json (one table for Blender, the app's live view and the tests, 2026-10-24)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mannequin_rules.json"), encoding="utf-8") as _fh:
    RULES = json.load(_fh)
KEYS = tuple(RULES["keys"])
LIMITS = tuple(RULES["limits"])


def bone_scales(body: dict) -> dict:
    """{bone: uniform pose scale} for a checked body (RULES["bones"]: product of `mul` / product of `div`)."""
    out = {}
    for bone, r in RULES["bones"].items():
        f = 1.0
        for k in r.get("mul", []):
            f *= body.get(k, 1.0)
        for k in r.get("div", []):
            f /= body.get(k, 1.0)
        out[bone] = f
    return out


def check(body: dict) -> dict:
    out = {}
    for k, v in (body or {}).items():
        if k not in KEYS:
            raise SystemExit(f"mannequin body: unknown key {k!r} (use {', '.join(KEYS)})")
        v = float(v)
        if not LIMITS[0] <= v <= LIMITS[1]:
            raise SystemExit(f"mannequin body: {k} {v} is outside {LIMITS[0]}..{LIMITS[1]}")
        out[k] = v
    return out


def apply(rig, body: dict, base_location=None) -> dict:
    """Give `rig` the body (see the module text). `base_location` = where the rig stands (default: its location now)."""
    body = check(body)
    s = body.get("scale", 1.0)
    rig.scale = (s, s, s)
    pb = rig.pose.bones
    for name, f in bone_scales(body).items():                      # the shoulders narrow WITH the chest; the arms and
        if name in pb:                                             # the head keep their sizes (RULES["bones"])
            pb[name].scale = (f, f, f)
    balls = RULES["balls"]
    factor = body.get(balls["factor"], 1.0)
    if abs(factor - 1.0) > 1e-9:                                   # the shoulder BALLS (the round joint at the top of
        for o in rig.children:                                     # each upper arm) shrink with the chest: at full
            if o.type == "MESH" and o.parent_type == "BONE" and o.parent_bone in balls["bones"]:
                dims = sorted(o.dimensions)                        # size they stood above a shorter chest (user: "why
                if dims[0] > 1e-6 and dims[2] / dims[0] < balls["max_aspect"]:   # is the shoulder pointing out"); a
                    o.scale = o.scale * factor                     # sphere, not the arm's cylinder
    dr = RULES["drop"]
    legs = body.get(dr["key"], 1.0)
    drop = 0.0
    if abs(legs - 1.0) > 1e-9 and dr["bone"] in rig.data.bones:   # the thigh's head height above the rig's floor (rest)
        drop = (1.0 - legs) * rig.data.bones[dr["bone"]].head_local.z * s
    loc = Vector(base_location) if base_location is not None else rig.location.copy()
    rig.location = (loc.x, loc.y, loc.z - drop)
    bpy.context.view_layer.update()
    return {"body": body, "drop": round(drop, 4)}
