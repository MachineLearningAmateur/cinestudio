"""Clip texts written from Blender: what the rig and the camera do in each clip (2026-10-22).

A long shot is made clip by clip, and each clip should get a text that says what ITS frames show (a whole-shot text
made a middle clip wave where the layout does not; the masked3d orbit's last clip stepped during the wave because
nothing said "feet planted"). bpy_jobs/rig_timeline.py exports the rig's bones and the camera per frame; this module
turns them into events (which hand is raised and waves, whether the feet stay on one spot, crouches) and into one short
text per clip in the order of Lightricks' prompting guide: the shot and the place, the action in time order, then the
camera and what it sees at the END of its move (docs.ltx.io prompting guide; the Layout-To-Render card asks for short
prompts that describe the finished shot and match the first frame; no "clay" or "3D").

Storage: a template keeps its events in shot.json ("timeline"); a run writes its clip texts into the manifest
(clip_videos) at `run new`, where they can be read and changed before the start."""
from __future__ import annotations

import math

RAISE = 0.10            # a hand counts as raised when it is this far (metres) above its shoulder
PLANTED = 0.06          # feet that move less than this (metres, on the floor plane) stay "on the same spot"
WALK = 0.5              # feet that move more than this walk
MIN_EVENT = 8           # frames: shorter raises are ignored


def _v(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _frames(raw: dict) -> list[dict]:
    return raw["frames"]


def _side_word(side: str) -> str:
    return "left" if side == "L" else "right"


def raised_segments(raw: dict) -> dict[str, list[tuple[int, int, bool]]]:
    """Per side ("L", "R"): [(first index, last index, waves)] of the frames where the hand is above the shoulder.
    Indexes count from 0 = the shot's first frame. "waves" = the hand swings sideways at least twice."""
    out = {}
    fr = _frames(raw)
    for side in ("L", "R"):
        up = [f["bones"][f"hand.{side}"][2] > f["bones"][f"shoulder.{side}"][2] + RAISE for f in fr]
        segs, start = [], None
        for i, u in enumerate(up + [False]):
            if u and start is None:
                start = i
            elif not u and start is not None:
                if i - start >= MIN_EVENT:
                    segs.append([start, i - 1])
                start = None
        merged = []
        for s in segs:                                   # join raises with a short gap (a wave that dips)
            if merged and s[0] - merged[-1][1] <= 6:
                merged[-1][1] = s[1]
            else:
                merged.append(s)
        res = []
        for a, b in merged:
            xs = [fr[i]["bones"][f"hand.{side}"][0] - fr[i]["bones"][f"shoulder.{side}"][0] for i in range(a, b + 1)]
            turns, last = 0, 0
            for i in range(1, len(xs)):
                d = xs[i] - xs[i - 1]
                if abs(d) > 0.004:
                    sgn = 1 if d > 0 else -1
                    if last and sgn != last:
                        turns += 1
                    last = sgn
            res.append((a, b, turns >= 2))
        out[side] = res
    return out


def feet_move(raw: dict, a: int, b: int) -> float:
    """The largest distance (metres, on the floor plane) that a foot moves from its place at frame a, in a..b."""
    fr = _frames(raw)
    most = 0.0
    for side in ("L", "R"):
        x0, y0 = fr[a]["bones"][f"foot.{side}"][:2]
        for i in range(a, b + 1):
            x, y = fr[i]["bones"][f"foot.{side}"][:2]
            most = max(most, math.hypot(x - x0, y - y0))
    return most


def camera_angle(f: dict) -> tuple[float, float]:
    """(angle, distance) of the camera seen from the character: angle 0 = in front of it, +90 = at its LEFT side,
    -90 = at its right side, +-180 = behind it. The front comes from the shoulders (left minus right, turned 90 deg)."""
    b = f["bones"]
    sl, sr, hips = b["shoulder.L"], b["shoulder.R"], b["hips"]
    lx, ly = sl[0] - sr[0], sl[1] - sr[1]
    n = math.hypot(lx, ly) or 1.0
    lx, ly = lx / n, ly / n
    fx, fy = ly, -lx                                     # the left axis turned -90 deg = the front (a -Y facing rig)
    cx, cy = f["camera"][0] - hips[0], f["camera"][1] - hips[1]
    ang = math.degrees(math.atan2(cx * lx + cy * ly, cx * fx + cy * fy))
    return ang, math.hypot(cx, cy)


def view_words(ang: float) -> str:
    a = abs(ang)
    side = "left" if ang > 0 else "right"
    if a < 30:
        return "from the front"
    if a < 65:
        return f"from the front, three-quarter from its {side}"
    if a < 115:
        return f"from its {side} side, in profile"
    if a < 150:
        return f"from behind, over its {side} shoulder"
    return "from behind"


def clip_ranges(shot_frames: int, clip_frames: int, overlap: int, clips: int) -> list[tuple[int, int, int]]:
    """[(first, last shown, last of the clip)] shot indexes per clip; frames past the shot's end hold its last frame."""
    out = []
    for k in range(clips):
        a = k * (clip_frames - overlap)
        out.append((a, min(a + clip_frames - 1, shot_frames - 1), a + clip_frames - 1))
    return out


def clip_text(raw: dict, a: int, b: int, end: int, place: str, who: str = "the character",
              segs: dict | None = None) -> str:
    """One clip's text: the place, the action in time order, the camera and its end view."""
    fr = _frames(raw)
    segs = segs if segs is not None else raised_segments(raw)
    n = b - a + 1
    beats = []
    for side, lst in segs.items():
        for s, e, waves in lst:
            if e < a or s > b:
                continue
            hand = f"its {_side_word(side)} hand"
            when = "" if s <= a else ("at first, " if (s - a) < 0.15 * n else
                                      "near the end, " if (s - a) > 0.7 * n else "then ")
            if s < a:                                    # raised before this clip starts
                act = f"keeps {hand} raised" + (" and waving" if waves else "")
            else:
                act = f"{when}raises {hand} high" + (" and waves it" if waves else "")
            if e <= b:
                act += " and then lowers it to its side"
            beats.append((max(s, a), act))
    beats.sort()
    name = f"{who[0].upper()}{who[1:]}"
    move = feet_move(raw, a, b)
    if move < PLANTED:
        sents = [f"{name} stands on the same spot with both feet planted."]
    elif move < WALK:
        sents = [f"{name} shifts its feet a little but stays in place."]
    else:
        sents = [f"{name} walks."]
    if beats:
        for i, (_, act) in enumerate(beats):
            act = act.replace("then ", "", 1) if i == 0 and act.startswith("then ") else act
            sents.append(f"It {act.replace('at first, ', '').replace('near the end, ', '')}."
                         if not act.startswith(("at first", "near the end"))
                         else f"{act[0].upper()}{act[1:].replace(', ', ' it ', 1)}.")
    else:
        sents.append("Its arms stay relaxed at its sides.")
    if end > b:
        sents.append("Then it holds this last pose without moving.")
    a0, d0 = camera_angle(fr[a])
    a1, d1 = camera_angle(fr[b])
    turn = a1 - a0
    turn = (turn + 180) % 360 - 180 if abs(turn) > 180 else turn
    if abs(turn) < 5:
        cam = f"The camera holds still and sees it {view_words(a0)}."
    else:
        cam = (f"The camera circles slowly around it: at the start it sees it {view_words(a0)}, "
               f"and by the end {view_words(a1)}.")
    if d1 > d0 * 1.15:
        cam += " The camera pulls back as it moves."
    elif d1 < d0 * 0.87:
        cam += " The camera moves closer as it moves."
    place = place.strip().rstrip(".")
    head = f"A full-body shot on {place[0].lower()}{place[1:]}. " if place else ""
    return head + " ".join(sents) + " " + cam


def clip_texts(raw: dict, place: str, shot_frames: int, clip_frames: int, overlap: int, clips: int,
               who: str = "the character") -> dict[str, str]:
    segs = raised_segments(raw)
    return {str(k + 1): clip_text(raw, a, b, e, place, who, segs)
            for k, (a, b, e) in enumerate(clip_ranges(shot_frames, clip_frames, overlap, clips))}


def events(raw: dict) -> list[dict]:
    """The template's events (for shot.json): raised / waving hands with their frames, and whether the feet move."""
    out = []
    for side, lst in raised_segments(raw).items():
        for s, e, waves in lst:
            out.append({"what": ("waves" if waves else "raises") + f" its {_side_word(side)} hand",
                        "frames": [s + raw["frame_start"], e + raw["frame_start"]]})
    n = len(_frames(raw))
    out.append({"what": "feet planted" if feet_move(raw, 0, n - 1) < PLANTED else "feet move",
                "frames": [raw["frame_start"], raw["frame_end"]]})
    return sorted(out, key=lambda x: x["frames"][0])


def blend_timeline(blend, out_json, rig: str = "MannequinRig", log=None) -> dict:
    """Run bpy_jobs/rig_timeline.py on a .blend (no render) and return its data."""
    import json
    from pathlib import Path
    from . import blender
    out_json = Path(out_json)
    blender.run_job("rig_timeline.py", Path(blend), [str(out_json), "--rig", rig],
                    log=log or out_json.with_suffix(".log"))
    return json.loads(out_json.read_text(encoding="utf-8"))


def run_clip_texts(raw: dict, place: str, clip_frames: int, overlap: int, who: str = "the character") -> dict[str, str]:
    """The clip texts of a run, with the run's own clip split (runner.Run.clip_format)."""
    from . import video
    shot = raw["frame_end"] - raw["frame_start"] + 1
    frames = (min(shot, clip_frames) - 1) // 8 * 8 + 1
    clips = 1 if shot <= clip_frames else video.n_clips(shot, frames, overlap)
    return clip_texts(raw, place, shot, frames, overlap, clips, who)
