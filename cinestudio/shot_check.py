"""`shot check NAME`: the template rules (AGENTS.md section 5, docs/TEMPLATE_GUIDE.md) as automatic checks.

    check(name, fresh=False, store=True) -> {"ok", "template", "errors", "warnings", "infos", "checks", "sha", ...}
    check_blend(blend, rec)              -> the same for any .blend (export, `shot add`, tests)
    rules(facts, rec)                    -> the checks from the measured facts (pure: tested without Blender)
    summary(result) / chip(rec)          -> {"errors", "warnings", "at", "sha"} for shot.json "check" and the cards

ONE headless Blender pass (bpy_jobs/check_shot.py: no render, a few seconds, safe next to a GPU job) measures every
frame: pose-bone scales, the rig's joints (world and screen), the camera, the layers (scan_layers), the props' screen
boxes, the Set's material colours and the sound strips. The facts are cached by the .blend's sha256
(<home>/cache/shot_check/<sha>.json), so a repeat check is instant; the rules run again each time (they also read the
record: clip texts, a loop).

Each check: {"id", "severity" error|warning|info|ok, "message", "frames" [[first, last], ...] (Blender frames),
"fix" (one line), "data"}. Errors block the export (setup/export_template.py --force overrides)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import tempfile
import threading
import time
from pathlib import Path

from . import blender, runner, shots, timeline

FACTS_VERSION = 1                 # bpy_jobs/check_shot.py VERSION: the cache key (a new scan when it changes)
PREVIEW_W, PREVIEW_H = 768, 448   # the run's landscape preview size (preview_size: a portrait shot is 448x768)
CAMERA_WARN, CAMERA_ERROR = 0.6, 1.0      # deg/frame about the subject (an orbit backup at 1.22 failed)
FRONT_DEG = 30.0                  # frame 1: the body within this of the camera
FIRST_SPEED = 0.015               # m/frame: frame 1 is a still pose (the look image is made from it)
HOLD_FRAMES, HOLD_SPEED = 12, 0.005       # the last 12 frames hold a stance (< 5 mm/frame)
SHARED_HOLD_SPEED = 0.008         # m/frame: a raised pose held across a join's shared frames (no move in them) is fine
FIG_MIN, FIG_MAX = 0.35, 0.85     # the figure's screen height at the poster frame (built-in: 0.62)
PROP_MIN_PX = 12                  # a prop smaller than this at preview does not read
SET_SATURATION, SET_VALUE = 0.25, 0.05    # a coloured Set material (HSV of its linear colour)
ACCEL_POP = 40.0                  # px/frame^2 at preview: a pop (the built-in's hard flash-step stops peak at ~31)
KICK_HEIGHT = 0.30                # m above the foot's floor level, the other foot down: a kick
JUMP_HEIGHT = 0.08                # m: both feet off the floor
TURN_DEG, TURN_WINDOW = 45.0, 8   # a body turn of 45 deg within 8 frames: a fast turn
DASH_SPEED = 0.06                 # m/frame of the hips on the floor plane (1.4 m/s): a fast step / dash
SLIDE_SPEED, SLIDE_FLOOR = 0.018, 0.03    # foot slide (info): > 1.8 cm/frame within 3 cm of the floor
SCALE_ERROR = 0.01                # a pose-bone scale off by more than 1 % is an error (0.1-1 %: a warning, float drift)
SEVERITIES = ("error", "warning", "info", "ok")


class CheckError(shots.ShotError):
    """A template that cannot be checked (a video template, not done, no .blend)."""


# ---------------------------------------------------------------- the facts (Blender, cached by sha256)
_SHA_MEMO: dict[tuple, str] = {}


def blend_sha(path: str | Path) -> str:
    """sha256 of a file, remembered per (path, size, mtime) in this process (the cards ask for it on every list)."""
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SHA_MEMO:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _SHA_MEMO[key] = h.hexdigest()
    return _SHA_MEMO[key]


def cache_dir() -> Path:
    return runner.home() / "cache" / "shot_check"


def _cached(sha: str) -> dict | None:
    p = cache_dir() / f"{sha}.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if d.get("version") == FACTS_VERSION else None


def measure(blend: str | Path) -> dict:
    """bpy_jobs/check_shot.py on a .blend (headless, no render)."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "facts.json"
        blender.run_job("check_shot.py", blend, ["--out", str(out)], log=Path(td) / "check_shot.log", timeout=600)
        return json.loads(out.read_text(encoding="utf-8"))


def facts_for(blend: str | Path, fresh: bool = False, measurer=None) -> tuple[dict, str, bool]:
    """(facts, sha, from the cache) of a .blend."""
    blend = Path(blend)
    if not blend.is_file():
        raise CheckError(f"no .blend at {blend}")
    sha = blend_sha(blend)
    if not fresh:
        hit = _cached(sha)
        if hit is not None:
            return hit, sha, True
    t0 = time.time()
    with _sha_lock(sha):                               # one Blender read per .blend at a time in this process (the
        if not fresh:                                  # API's executor threads): a second check waits and takes the
            hit = _cached(sha)                         # first one's facts (Codex review 2026-10-25)
            if hit is not None:
                return hit, sha, True
        elif _measured_at.get(sha, 0.0) >= t0:         # a fresh read that started while we waited counts as ours
            hit = _cached(sha)
            if hit is not None:
                return hit, sha, False
        facts = (measurer or measure)(blend)
        if facts.get("version") != FACTS_VERSION:
            raise CheckError(f"the scan script wrote version {facts.get('version')}, expected {FACTS_VERSION}")
        d = cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"{sha}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}.tmp"   # unique per writer
        try:
            tmp.write_text(json.dumps(facts), encoding="utf-8")
            os.replace(tmp, d / f"{sha}.json")
        finally:
            tmp.unlink(missing_ok=True)
        _measured_at[sha] = time.time()
        return facts, sha, False


_SHA_LOCKS: dict[str, threading.Lock] = {}
_SHA_LOCKS_GUARD = threading.Lock()
_measured_at: dict[str, float] = {}


def _sha_lock(sha: str) -> threading.Lock:
    with _SHA_LOCKS_GUARD:
        return _SHA_LOCKS.setdefault(sha, threading.Lock())


# ---------------------------------------------------------------- small helpers
def _c(cid: str, severity: str, message: str, fix: str = "", frames: list | None = None, data: dict | None = None) -> dict:
    assert severity in SEVERITIES
    return {"id": cid, "severity": severity, "message": message, "frames": frames or [], "fix": fix,
            "data": data or {}}


def _segments(mask: list[bool], min_len: int = 1, gap: int = 0) -> list[list[int]]:
    """Index runs [first, last] where mask is True (runs closer than `gap` are joined; shorter than min_len dropped)."""
    runs, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if runs and start - runs[-1][1] - 1 <= gap:
                runs[-1][1] = i - 1
            else:
                runs.append([start, i - 1])
            start = None
    return [r for r in runs if r[1] - r[0] + 1 >= min_len]


def _span_text(frames: list) -> str:
    return ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in frames[:8]) + (" ..." if len(frames) > 8 else "")


def _dist(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _speeds(tl: list[dict]) -> list[float]:
    """Per frame step (i -> i+1): the largest move of a tracked joint (m/frame)."""
    out = []
    for f0, f1 in zip(tl, tl[1:]):
        out.append(max(_dist(f0["bones"][b], f1["bones"][b]) for b in f0["bones"]))
    return out


def _yaw(f: dict) -> float:
    l, r = f["bones"]["shoulder.L"], f["bones"]["shoulder.R"]
    return math.degrees(math.atan2(l[1] - r[1], l[0] - r[0]))


def _unwrap(deg: list[float]) -> list[float]:
    out = []
    for d in deg:
        if out:
            while d - out[-1] > 180:
                d -= 360
            while d - out[-1] < -180:
                d += 360
        out.append(d)
    return out


def _percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    k = (len(s) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def shared_windows(n_frames: int, frame_start: int = 1) -> list[dict]:
    """The shared frames of each clip join (runner's default split: clip k = shot frames 96(k-1)+1 .. +121):
    [{"join": k (between clip k and k+1), "frames": [first, last] Blender frames}]."""
    plan = shots.clip_plan(n_frames)
    off = frame_start - 1
    return [{"join": a["clip"], "frames": [b["first"] + off, a["last"] + off]} for a, b in zip(plan, plan[1:])]


def motion_events(tl: list[dict], frame_start: int = 1) -> list[dict]:
    """The rig's events that must not cross a clip join: raised hands, kicks, jumps, fast turns, fast steps (dashes).
    [{"what", "frames": [first, last]}] in Blender frames."""
    if len(tl) < 3:
        return []
    raw = {"frames": tl, "frame_start": frame_start, "frame_end": frame_start + len(tl) - 1}
    ev = []
    segs = timeline.raised_segments(raw)
    right = list(segs.get("R") or [])
    for a, b, waves in segs.get("L") or []:              # both hands up together: one event
        both = next((r for r in right if min(b, r[1]) - max(a, r[0]) + 1 >= 0.5 * max(b - a + 1, r[1] - r[0] + 1)),
                    None)
        if both:
            right.remove(both)
            ev.append({"what": ("waves" if waves or both[2] else "raises") + " both hands",
                       "frames": [min(a, both[0]) + frame_start, max(b, both[1]) + frame_start]})
        else:
            ev.append({"what": ("waves" if waves else "raises") + " its left hand",
                       "frames": [a + frame_start, b + frame_start]})
    for a, b, waves in right:
        ev.append({"what": ("waves" if waves else "raises") + " its right hand",
                   "frames": [a + frame_start, b + frame_start]})
    z = {s: [f["bones"][f"foot.{s}"][2] for f in tl] for s in ("L", "R")}
    low = {s: _percentile(z[s], 0.05) for s in z}
    up = {s: [v - low[s] for v in z[s]] for s in z}
    for s, o in (("L", "R"), ("R", "L")):
        mask = [up[s][i] > KICK_HEIGHT and up[o][i] < JUMP_HEIGHT for i in range(len(tl))]
        for a, b in _segments(mask, 3, 2):
            ev.append({"what": f"kicks with its {'left' if s == 'L' else 'right'} foot",
                       "frames": [a + frame_start, b + frame_start]})
    for a, b in _segments([up["L"][i] > JUMP_HEIGHT and up["R"][i] > JUMP_HEIGHT for i in range(len(tl))], 3, 2):
        ev.append({"what": "jumps (both feet off the floor)", "frames": [a + frame_start, b + frame_start]})
    yaw = _unwrap([_yaw(f) for f in tl])
    turn = [False] * len(tl)
    for i in range(len(tl) - TURN_WINDOW):
        if abs(yaw[i + TURN_WINDOW] - yaw[i]) > TURN_DEG:
            for j in range(i, i + TURN_WINDOW + 1):
                turn[j] = True
    for a, b in _segments(turn, 1, 2):
        ev.append({"what": f"turns fast ({abs(yaw[b] - yaw[a]):.0f} deg)", "frames": [a + frame_start, b + frame_start]})
    hips = [f["bones"]["hips"] for f in tl]
    dash = [math.hypot(h1[0] - h0[0], h1[1] - h0[1]) > DASH_SPEED for h0, h1 in zip(hips, hips[1:])]
    for a, b in _segments(dash, 2, 2):
        ev.append({"what": "fast step (the hips dash)", "frames": [a + frame_start, b + 1 + frame_start]})
    return sorted(ev, key=lambda e: e["frames"][0])


# ---------------------------------------------------------------- the rules
def _rigs(facts: dict) -> list[tuple[str, dict, str]]:
    """[(rig, data, label)]: the label names the collection when there are several figures."""
    rigs = facts.get("rigs") or {}
    many = len(rigs) > 1
    return [(n, r, f"{r.get('collection') or n}: " if many else "") for n, r in rigs.items()]


def rule_bone_scale(facts: dict, rec: dict) -> list[dict]:
    out = []
    for name, r, lab in _rigs(facts):
        leaks = r.get("scale_leaks") or []             # every bone off by more than 0.1 % (bpy_jobs/check_shot.py)
        if leaks:
            bones = ", ".join(f"{x['bone']} ({x['max']:g})" for x in leaks[:6]) + (" ..." if len(leaks) > 6 else "")
            big = any(abs(x["max"] - 1.0) > SCALE_ERROR for x in leaks)
            out.append(_c("bone_scale", "error" if big else "warning", f"{lab}pose-bone scales are not 1: {bones}",
                          "key rotations only: clear the scale keys (Alt+S on the pose bones) and key the scale at 1; "
                          "the run sets each character's body scales itself",
                          [[x["first"], x["last"]] for x in leaks], {"rig": name, "bones": leaks}))
    if not out and facts.get("rigs"):
        out.append(_c("bone_scale", "ok", "every pose bone's scale is 1 on every frame"))
    return out


def rule_shared_frames(facts: dict, rec: dict) -> list[dict]:
    f0, f1 = facts["frames"]
    wins = shared_windows(f1 - f0 + 1, f0)
    if not wins:
        return [_c("shared_frames", "ok", "one clip: no clip joins")]
    out = []
    for name, r, lab in _rigs(facts):
        tl = r.get("timeline") or []
        for e in motion_events(tl, f0):
            a, b = e["frames"]
            for w in wins:
                wa, wb = w["frames"]
                if a <= wb and b >= wa:
                    if a < wa and b > wb and _still(tl, wa - f0, wb - f0):   # a pose HELD across the join (it began
                        out.append(_c("shared_frames", "info",              # before and ends after, still inside):
                                      f"{lab}{e['what']} at frames {a}-{b}: a still pose held across the shared "   # fine
                                      f"frames {wa}-{wb} (no move inside them)", frames=[[a, b], [wa, wb]],
                                      data={"event": e["what"], "join": w["join"], "held": True}))
                        continue
                    out.append(_c("shared_frames", "warning",
                                  f"{lab}{e['what']} at frames {a}-{b} crosses the shared frames {wa}-{wb} of the join "
                                  f"between clips {w['join']} and {w['join'] + 1}",
                                  f"move it fully inside one clip (end before {wa} or start after {wb}), or hold a pose "
                                  "across the join", [[a, b], [wa, wb]], {"event": e["what"], "join": w["join"]}))
    if not any(c["severity"] == "warning" for c in out):
        out.append(_c("shared_frames", "ok", "no hand raise, kick, jump, fast turn or dash crosses a clip join's shared "
                      f"frames ({_span_text([w['frames'] for w in wins])})"))
    return out


def _still(tl: list[dict], i0: int, i1: int) -> bool:
    """No tracked joint moves more than SHARED_HOLD_SPEED per frame between timeline indexes i0 and i1."""
    seg = tl[max(0, i0):i1 + 1]
    return len(seg) >= 2 and max(_speeds(seg)) < SHARED_HOLD_SPEED


def rule_end_hold(facts: dict, rec: dict) -> list[dict]:
    if (rec.get("plan") or {}).get("loop"):
        return [_c("end_hold", "info", "a loop: the end runs on into the start (no end hold needed)")]
    out = []
    for name, r, lab in _rigs(facts):
        tl = r.get("timeline") or []
        if len(tl) < HOLD_FRAMES + 1:
            continue
        sp = _speeds(tl)
        tail = max(sp[-HOLD_FRAMES:])
        held = 0
        for v in reversed(sp):
            if v >= HOLD_SPEED:
                break
            held += 1
        last = tl[-1]["frame"]
        if tail >= HOLD_SPEED:
            out.append(_c("end_hold", "warning",
                          f"{lab}the end does not hold a stance: a joint moves {tail * 1000:.0f} mm/frame in the last "
                          f"{HOLD_FRAMES} frames (held {held} frames; < {HOLD_SPEED * 1000:.0f} mm/frame wanted)",
                          "ease into the end pose over >= 24 frames, then hold it still for >= 12 frames (mocap.settle)",
                          [[last - HOLD_FRAMES + 1, last]], {"max_m_per_frame": round(tail, 4), "held_frames": held}))
        else:
            out.append(_c("end_hold", "ok", f"{lab}the end holds a stance for {held} frames",
                          data={"held_frames": held}))
    return out


def rule_first_frame(facts: dict, rec: dict) -> list[dict]:
    out = []
    for name, r, lab in _rigs(facts):
        tl = r.get("timeline") or []
        if len(tl) < 4 or not tl[0].get("camera"):
            continue
        ang, _ = timeline.camera_angle(tl[0])
        speed = max(_speeds(tl[:4]))
        f = tl[0]["frame"]
        bad = []
        if abs(ang) >= FRONT_DEG:
            bad.append(f"the body is turned {abs(ang):.0f} deg from the camera ({timeline.view_words(ang)})")
        if speed >= FIRST_SPEED:
            bad.append(f"it is moving ({speed * 100:.1f} cm/frame)")
        if bad:
            out.append(_c("first_frame", "warning", f"{lab}frame {f} is not a clean front pose: " + "; ".join(bad) +
                          ". The look image is made from it",
                          "start on a still pose facing the camera (within 30 deg); begin the move a few frames later",
                          [[f, f + 3]], {"angle_deg": round(ang, 1), "speed_m_per_frame": round(speed, 4)}))
        else:
            out.append(_c("first_frame", "ok", f"{lab}frame {f} is a still front pose ({abs(ang):.0f} deg)",
                          data={"angle_deg": round(ang, 1)}))
    return out


def camera_speeds(facts: dict) -> list[float]:
    """deg/frame of the camera per step: the larger of its view turn and its orbit about the figures' mean hips."""
    cams = (facts.get("camera") or {}).get("frames") or []
    hips = [f["bones"]["hips"] for _, r, _ in _rigs(facts) for f in (r.get("timeline") or [])]
    cx = sum(h[0] for h in hips) / len(hips) if hips else 0.0
    cy = sum(h[1] for h in hips) / len(hips) if hips else 0.0
    out = []
    for c0, c1 in zip(cams, cams[1:]):
        u, v = c0["fwd"], c1["fwd"]                     # atan2(|u x v|, u.v): exact 0 for equal (rounded) vectors
        cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        view = math.degrees(math.atan2(math.sqrt(sum(x * x for x in cross)), sum(a * b for a, b in zip(u, v))))
        a0 = math.atan2(c0["loc"][1] - cy, c0["loc"][0] - cx)
        a1 = math.atan2(c1["loc"][1] - cy, c1["loc"][0] - cx)
        orbit = abs(math.degrees((a1 - a0 + math.pi) % (2 * math.pi) - math.pi))
        out.append(0.0 if c0.get("name") != c1.get("name") else max(view, orbit))   # a camera switch is a cut
    return out


def rule_camera(facts: dict, rec: dict) -> list[dict]:
    cams = (facts.get("camera") or {}).get("frames") or []
    if not cams:
        return [_c("camera_speed", "error", "the scene has no camera", "add a camera and make it the scene camera")]
    sp = camera_speeds(facts)
    top = max(sp) if sp else 0.0
    f0 = facts["frames"][0]
    out = []
    if top >= CAMERA_WARN:
        sev = "error" if top > CAMERA_ERROR else "warning"
        fast = _segments([v >= CAMERA_WARN for v in sp], 1, 2)
        frames = [[a + f0, b + 1 + f0] for a, b in fast]
        out.append(_c("camera_speed", sev, f"the camera turns up to {top:.2f} deg/frame about the subject (keep it under "
                      f"{CAMERA_WARN}; over {CAMERA_ERROR} is an error) at frames {_span_text(frames)}",
                      "slow the move (a 90 deg arc needs >= 150 frames) or make the shot longer (more clips)", frames,
                      {"max_deg_per_frame": round(top, 3)}))
    else:
        out.append(_c("camera_speed", "ok", f"the camera turns at most {top:.2f} deg/frame",
                      data={"max_deg_per_frame": round(top, 3)}))
    if (facts.get("camera") or {}).get("moves") or (facts.get("layers") or {}).get("camera_moves"):
        out.append(_c("camera_still", "info", "moving camera: use a panorama (a 360 background; a backplate needs a still camera)"))
    else:
        out.append(_c("camera_still", "info", "still camera: a backplate is possible (background plate / from-set)"))
    return out


def rule_layers(facts: dict, rec: dict) -> list[dict]:
    scan = facts.get("layers") or {}
    out = []
    if not scan.get("characters"):
        out.append(_c("layers", "error", "no character layer: a template needs a Mannequin collection with its rig "
                      "MannequinRig", "put the figure and its rig in a collection named Mannequin (rig MannequinRig)"))
    for c in scan.get("characters") or []:
        if not c.get("rig"):
            out.append(_c("layers", "error", f"{c['collection']} has no rig",
                          f"put the armature (MannequinRig{c['collection'][len('Mannequin'):]}) in {c['collection']}"))
    for coll, objs in (facts.get("mannequin_foreign") or {}).items():
        out.append(_c("layers", "error", f"objects in {coll} that are not part of the figure (props in the character "
                      f"layer): {', '.join(objs[:8])}{' ...' if len(objs) > 8 else ''}",
                      "move them to Props (a held prop parented to the hand bone) or Set: the character mask and the "
                      "proxy kit read the Mannequin collection", data={"collection": coll, "objects": objs}))
    if facts.get("in_two_layers"):
        objs = facts["in_two_layers"]
        out.append(_c("layers", "warning", f"objects in more than one layer: {', '.join(objs[:8])}",
                      "link each object to one layer only (Mannequin*, Props or Set)", data={"objects": objs}))
    if scan.get("loose"):
        objs = scan["loose"]
        out.append(_c("layers", "warning", f"objects in no layer: {', '.join(objs[:8])}{' ...' if len(objs) > 8 else ''}",
                      "put the static place in a Set collection and moving or touched objects in Props",
                      data={"objects": objs}))
    if not scan.get("set"):
        out.append(_c("layers", "warning", "no Set collection: a backplate cannot be drawn over the set",
                      "put the floor and the place in a collection named Set (or use background plate from text)"))
    if not out:
        n = len(scan.get("characters") or [])
        out.append(_c("layers", "ok", f"layers: {n} character layer{'s' if n != 1 else ''}, "
                      f"{len(scan.get('props') or [])} prop objects, {len(scan.get('set') or [])} set objects; nothing "
                      "loose"))
    return out


def rule_figure_size(facts: dict, rec: dict) -> list[dict]:
    out = []
    chars = (facts.get("layers") or {}).get("characters") or []
    many = len(chars) > 1
    poster = facts["frames"][0]
    for c in chars:
        lab = f"{c['collection']}: " if many else ""
        sc = c.get("screen")
        if not sc:
            out.append(_c("figure_size", "warning", f"{lab}the figure is not in front of the camera at frame {poster}",
                          "aim the camera at the figure", [[poster, poster]]))
            continue
        h = sc["feet"][1] - sc["head"][1]
        if h < FIG_MIN or h > FIG_MAX:
            out.append(_c("figure_size", "warning", f"{lab}the figure fills {h * 100:.0f}% of the frame height at frame "
                          f"{poster} (keep it between {FIG_MIN * 100:.0f}% and {FIG_MAX * 100:.0f}%)",
                          "move the camera closer or farther (the built-in: 30 mm lens, 4 m, head at 0.20, feet at "
                          "0.81)", [[poster, poster]], {"height": round(h, 3)}))
        else:
            out.append(_c("figure_size", "ok", f"{lab}the figure fills {h * 100:.0f}% of the frame height",
                          data={"height": round(h, 3)}))
    # every frame: the whole body inside the frame (head top, ankles and toes)
    f0 = facts["frames"][0]
    for name, r, lab in _rigs(facts):
        js = r.get("joints") or {}
        n = max((len(v) for v in js.values()), default=0)
        outside = []
        for i in range(n):
            pts = [js[j][i] for j in ("head", "ankle.L", "ankle.R", "toe.L", "toe.R", "hand.L", "hand.R")
                   if j in js and i < len(js[j])]
            outside.append(any(p is None or not (-0.005 <= p[0] <= 1.005 and -0.005 <= p[1] <= 1.005) for p in pts))
        segs = [[a + f0, b + f0] for a, b in _segments(outside)]
        if segs:
            out.append(_c("figure_size", "warning", f"{lab}part of the body leaves the frame at frames {_span_text(segs)}",
                          "keep the whole body, feet and raised hands included, inside the frame on every frame "
                          "(widen the lens or move the camera back)", segs))
    return out


def rule_prop_size(facts: dict, rec: dict) -> list[dict]:
    out = []
    f0 = facts["frames"][0]
    pw, ph = preview_size(facts)
    for p in facts.get("props") or []:
        if p.get("part"):
            continue
        sizes, frames = [], []
        for i, b in enumerate(p.get("boxes") or []):
            if not b:
                continue
            cu, cv = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if 0 <= cu <= 1 and 0 <= cv <= 1:
                sizes.append(max((b[2] - b[0]) * pw, (b[3] - b[1]) * ph))
                frames.append(i + f0)
        if not sizes:
            out.append(_c("prop_size", "info", f"{p['name']} is never in the frame", "fine for a parked prop; else "
                          "check its path"))
            continue
        med = _percentile(sizes, 0.5)
        if med < PROP_MIN_PX:
            segs = [[a, b] for a, b in _segments_from_frames(frames)]
            out.append(_c("prop_size", "warning", f"{p['name']} is {med:.0f} px on screen at preview (median of the "
                          f"{len(sizes)} frames it is in view): too small to read (>= {PROP_MIN_PX} px wanted)",
                          "make it bigger or bring it nearer the camera (at 4 m, 1 px ~ 6 mm at preview)", segs,
                          {"median_px": round(med, 1), "max_px": round(max(sizes), 1)}))
        else:
            out.append(_c("prop_size", "ok", f"{p['name']}: {med:.0f} px on screen at preview (median)",
                          data={"median_px": round(med, 1), "min_px": round(min(sizes), 1)}))
    return out


def _segments_from_frames(frames: list[int]) -> list[list[int]]:
    out = []
    for f in frames:
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def rule_set_neutral(facts: dict, rec: dict) -> list[dict]:
    mats = facts.get("set_materials") or []
    loud = [m for m in mats if (m["saturation"] > SET_SATURATION and m["value"] > SET_VALUE) or m.get("textured")]
    if loud:
        names = sorted({m["object"] for m in loud})
        return [_c("set_neutral", "info", f"{len(names)} Set object{'s are' if len(names) != 1 else ' is'} coloured or "
                   f"textured ({', '.join(names[:6])}{' ...' if len(names) > 6 else ''}): a neutral grey set lets the "
                   "background decide the look", "use neutral grey clay (R = G = B) in the Set; the props bring the colour",
                   data={"objects": names})]
    if not facts.get("layers", {}).get("set"):
        return []
    return [_c("set_neutral", "ok", "the Set is neutral grey")]


def rule_clip_texts(facts: dict, rec: dict) -> list[dict]:
    f0, f1 = facts["frames"]
    plan = shots.clip_plan(f1 - f0 + 1)
    if len(plan) <= 1:
        return [_c("clip_texts", "ok", "one clip: the run's video text describes it")]
    texts = {str(k): v for k, v in (rec.get("clip_texts") or {}).items() if isinstance(v, str) and v.strip()}
    missing = [c["clip"] for c in plan if str(c["clip"]) not in texts]
    out = []
    if missing:
        out.append(_c("clip_texts", "warning", f"{len(plan)} clips, but no clip text for clip"
                      f"{'s' if len(missing) > 1 else ''} {', '.join(map(str, missing))}: every clip would get the "
                      "whole-shot text", "write one text per clip with only what its frames show: "
                      "shot clip-texts NAME --clip K \"...\" (AGENTS.md 7.2)",
                      [[c["first"] + f0 - 1, c["last"] + f0 - 1] for c in plan if c["clip"] in missing],
                      {"missing": missing}))
    if not (rec.get("clip_context") or "").strip():
        out.append(_c("clip_texts", "warning", "no shared clip context (the lasting states for every clip, e.g. "
                      "\"{character} faces the camera the whole time\")",
                      "shot clip-texts NAME --context \"...\" (AGENTS.md 7.2)"))
    if not out:
        out.append(_c("clip_texts", "ok", f"clip texts for all {len(plan)} clips and a shared context"))
    return out


def rule_sound(facts: dict, rec: dict) -> list[dict]:
    snd = [s for s in facts.get("sounds") or [] if not s.get("mute")]
    if not snd:
        return [_c("sound", "info", "no sound: a run makes its sound from its sound text, or takes a library song "
                   "(--song)")]
    unpacked = [s["name"] for s in snd if not s.get("packed")]
    msg = f"sound: {', '.join(s['name'] for s in snd[:4])}" + (" (packed)" if not unpacked else
                                                                 f" ({len(unpacked)} not packed: the export packs it)")
    return [_c("sound", "info", msg + "; it is the run's music unless the run names --song. Someone else's recording "
               "keeps the template private", data={"sounds": snd})]


def preview_size(facts: dict) -> tuple[int, int]:
    """The run's preview size for this shot (runner.control_size: 768x448, or 448x768 for a portrait render):
    screen sizes and accelerations are measured at it (Codex review 2026-10-25)."""
    res = facts.get("resolution") or []
    if len(res) == 2 and res[0] and res[1]:
        return runner.control_size({"width": int(res[0]), "height": int(res[1])})
    return PREVIEW_W, PREVIEW_H


def joint_accels(r: dict, size: tuple[int, int] = (PREVIEW_W, PREVIEW_H)) -> dict[str, list[float | None]]:
    """Per joint: screen acceleration (px/frame^2 at the preview size) per frame (None at the ends / off screen)."""
    pw, ph = size
    out = {}
    for j, pts in (r.get("joints") or {}).items():
        acc = [None] * len(pts)
        for i in range(1, len(pts) - 1):
            a, b, c = pts[i - 1], pts[i], pts[i + 1]
            if a and b and c:
                acc[i] = math.hypot((a[0] - 2 * b[0] + c[0]) * pw, (a[1] - 2 * b[1] + c[1]) * ph)
        out[j] = acc
    return out


def rule_joint_accel(facts: dict, rec: dict) -> list[dict]:
    out = []
    f0 = facts["frames"][0]
    for name, r, lab in _rigs(facts):
        acc = joint_accels(r, preview_size(facts))
        if not acc:
            continue
        top_j, top_v, top_f = None, 0.0, None
        pops = []
        for j, a in acc.items():
            for i, v in enumerate(a):
                if v is None:
                    continue
                if v > top_v:
                    top_j, top_v, top_f = j, v, i + f0
                if v > ACCEL_POP:
                    pops.append((i + f0, j, v))
        if pops:
            segs = _segments_from_frames(sorted({f for f, _, _ in pops}))
            worst = sorted(pops, key=lambda x: -x[2])[:4]
            out.append(_c("joint_accel", "warning", f"{lab}joint pops (screen acceleration over {ACCEL_POP:g} px/frame^2 "
                          f"at preview): " + ", ".join(f"{j} {v:.0f} at frame {f}" for f, j, v in worst),
                          "ease those keys (or smooth the motion: mocap.sg) and look at the frames; a hard stop on a "
                          "beat can stay if a blind critic passes it", segs,
                          {"max": round(top_v, 1), "joint": top_j, "frame": top_f}))
        else:
            out.append(_c("joint_accel", "ok", f"{lab}no pops (largest screen acceleration {top_v:.0f} px/frame^2, "
                          f"{top_j} at frame {top_f})", data={"max": round(top_v, 1), "joint": top_j, "frame": top_f}))
    return out


def rule_foot_slide(facts: dict, rec: dict) -> list[dict]:
    out = []
    for name, r, lab in _rigs(facts):
        tl = r.get("timeline") or []
        if len(tl) < 3:
            continue
        n = 0
        for s in ("L", "R"):
            z = [f["bones"][f"foot.{s}"][2] for f in tl]
            low = min(z)
            for f_0, f_1 in zip(tl, tl[1:]):
                a, b = f_0["bones"][f"foot.{s}"], f_1["bones"][f"foot.{s}"]
                if a[2] < low + SLIDE_FLOOR and math.hypot(b[0] - a[0], b[1] - a[1]) > SLIDE_SPEED:
                    n += 1
        if n:
            out.append(_c("foot_slide", "info", f"{lab}a foot slides on the floor in {n} frame steps (> 1.8 cm/frame "
                          "within 3 cm of the floor); fine when it is a deliberate slide",
                          "lift the foot for a step, or plant it", data={"steps": n}))
    return out


TRACKED_BONES = ("head", "hips", "shoulder.L", "shoulder.R", "hand.L", "hand.R", "foot.L", "foot.R")  # check_shot.py


def rule_complete(facts: dict, rec: dict) -> list[dict]:
    """The measurements the motion rules need: every character layer's rig measured, with its tracked bones on every
    frame. Without them the joins, the end hold, the first frame and the feet were silently skipped (Codex review
    2026-10-25): an incomplete check is an error, not a pass."""
    out = []
    f0, f1 = facts["frames"]
    n = f1 - f0 + 1
    rigs = facts.get("rigs") or {}
    for c in (facts.get("layers") or {}).get("characters") or []:
        rig = c.get("rig")
        if rig and rig not in rigs:
            out.append(_c("check_incomplete", "error", f"{c['collection']}: its rig {rig} was not measured (not an "
                          "armature with a pose?): the motion checks could not run",
                          "make the rig an armature in the layer, then check again", data={"rig": rig}))
    for name, r, lab in _rigs(facts):
        got = len(r.get("timeline") or [])
        if got != n:
            out.append(_c("check_incomplete", "error", f"{lab}the motion checks could not run: the rig {name} lacks "
                          f"the tracked bones ({', '.join(TRACKED_BONES)}) on {n - got} of {n} frames",
                          "use the cinestudio bone names (the mannequin rig), then check again",
                          data={"rig": name, "frames": n, "tracked": got}))
    return out


RULES = (rule_complete, rule_bone_scale, rule_shared_frames, rule_end_hold, rule_first_frame, rule_camera,
         rule_layers, rule_figure_size, rule_prop_size, rule_set_neutral, rule_clip_texts, rule_sound,
         rule_joint_accel, rule_foot_slide)
RULE_IDS = ("check_incomplete", "bone_scale", "shared_frames", "end_hold", "first_frame", "camera_speed",
            "camera_still", "layers", "figure_size", "prop_size", "set_neutral", "clip_texts", "sound", "joint_accel",
            "foot_slide")


def rules(facts: dict, rec: dict | None = None) -> list[dict]:
    """Every check on the measured facts and the template's record, worst first (a stable order within a level). A
    rule that raises is an error ("check_incomplete"): the others still run, but the check does not pass."""
    rec = rec or {}
    out = []
    for fn in RULES:
        try:
            out += fn(facts, rec)
        except Exception as exc:  # noqa: BLE001 - one broken rule must not hide the others
            rid = fn.__name__.replace("rule_", "")
            out.append(_c("check_incomplete", "error", f"the {rid} check could not run: {exc}",
                          "fix the scene (or report the bug), then check again", data={"rule": rid}))
    order = {s: i for i, s in enumerate(SEVERITIES)}
    return sorted(out, key=lambda c: order[c["severity"]])


# ---------------------------------------------------------------- the check of a template
RULES_VERSION = 2                 # raise it when the rules change: a summary stored by older rules shows as stale


def record_fingerprint(rec: dict | None) -> str:
    """What the rules read from the record besides the .blend (clip texts, the clip context, a loop) and the rules'
    version: a stored summary with another fingerprint is stale (Codex review 2026-10-25: the card kept a check made
    before the clip texts changed)."""
    rec = rec or {}
    data = {"v": RULES_VERSION, "clip_context": (rec.get("clip_context") or "").strip(),
            "clip_texts": {str(k): v for k, v in (rec.get("clip_texts") or {}).items()},
            "loop": bool((rec.get("plan") or {}).get("loop"))}
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _result(name: str | None, checks: list[dict], sha: str, cached: bool, facts: dict | None = None,
            rec: dict | None = None) -> dict:
    count = {s: sum(1 for c in checks if c["severity"] == s) for s in SEVERITIES}
    f0, f1 = (facts or {}).get("frames") or [None, None]
    return {"ok": True, "template": name, "errors": count["error"], "warnings": count["warning"],
            "infos": count["info"], "checks": checks, "sha": sha, "fp": record_fingerprint(rec), "cached": cached,
            "at": time.time(),
            "frames": [f0, f1], "clips": len(shots.clip_plan(f1 - f0 + 1)) if f0 is not None else None}


def check_blend(blend: str | Path, rec: dict | None = None, name: str | None = None, fresh: bool = False,
                measurer=None, with_facts: bool = False) -> dict:
    facts, sha, cached = facts_for(blend, fresh, measurer)
    res = _result(name, rules(facts, rec or {}), sha, cached, facts, rec)
    if with_facts:
        res["facts"] = facts
    return res


def summary(res: dict) -> dict:
    """The short form stored in shot.json ("check") and shown on the cards."""
    out = {"errors": res["errors"], "warnings": res["warnings"], "at": round(res.get("at") or time.time(), 1),
           "sha": res["sha"]}
    if res.get("fp"):
        out["fp"] = res["fp"]
    return out


def check(name: str, fresh: bool = False, store: bool = True, measurer=None, with_facts: bool = False) -> dict:
    """`shot check NAME`: the rules on a library template (a Blender template; a video template has no rig). The
    summary is stored in the user template's shot.json ("check"); a built-in's record is read-only (the cache keeps
    the result)."""
    rec = shots.load(name)
    if rec.get("kind") == "video":
        raise CheckError(f"template {name!r} is a video template: the checks are for Blender templates")
    if rec.get("status") not in (None, "done"):
        raise CheckError(f"template {name!r} is not done ({rec.get('status')})")
    blend = Path((rec.get("files") or {}).get("blend") or "")
    if not blend.is_file():
        raise CheckError(f"template {name!r} has no .blend")
    res = check_blend(blend, rec, name, fresh, measurer, with_facts)
    if store and not rec.get("builtin") and not rec.get("handmade"):
        try:
            with shots._Life(name, f"template {name!r} is being changed now"):
                record(blend.parent / "shot.json", res)
        except shots.ShotError:
            pass                                       # busy: the cache still has the result
    return res


def record(shot_json: Path, res: dict) -> None:
    """Write the summary into a shot.json (the caller holds the template's lifecycle lock or builds it)."""
    p = Path(shot_json)
    data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    data["check"] = summary(res)
    shots._write(p, data)


def after_build(blend: str | Path, rec: dict, emit=None) -> dict | None:
    """The check at the end of `shot add` / from-video / make-dance: the summary for rec["check"] (None when the check
    cannot run; the template works without it). Events "check" name the errors and warnings."""
    try:
        res = check_blend(blend, rec, rec.get("name"))
    except Exception as exc:  # noqa: BLE001 - never fail a build over its check
        if emit:
            emit({"event": "check", "text": f"the template check could not run: {exc}"})
        return None
    if emit:
        for c in res["checks"]:
            if c["severity"] in ("error", "warning"):
                emit({"event": "check", "severity": c["severity"], "id": c["id"], "text": c["message"]})
    return summary(res)


def chip(rec: dict) -> dict | None:
    """The card's check state: {"errors", "warnings", "at", "sha", "stale"} from the cached facts of the CURRENT .blend
    (rules re-run on the record: instant), else the stored summary, marked stale when the .blend, the record's words
    (record_fingerprint) or the rules changed; None when the template was never checked (or is not a Blender
    template)."""
    blend = (rec.get("files") or {}).get("blend")
    if not blend or rec.get("kind") == "video" or rec.get("status") not in (None, "done"):
        return None
    try:
        sha = blend_sha(blend)
    except OSError:
        return None
    facts = _cached(sha)
    st = rec.get("check")
    if facts is not None:
        res = _result(rec.get("name"), rules(facts, rec), sha, True, facts, rec)
        if isinstance(st, dict) and st.get("sha") == sha and st.get("at"):
            res["at"] = st["at"]
        else:
            try:
                res["at"] = (cache_dir() / f"{sha}.json").stat().st_mtime
            except OSError:
                pass
        return {**summary(res), "stale": False}
    if isinstance(st, dict) and "errors" in st:          # the stored summary: stale when the .blend, the record's
        stale = st.get("sha") != sha or bool(st.get("fp") and st["fp"] != record_fingerprint(rec))   # words or the
        return {**st, "stale": stale}                                                              # rules changed
    return None
