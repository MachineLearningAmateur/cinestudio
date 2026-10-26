"""Clip prompts that say where each character stands and which way it faces, read from the Blender rig.

LTX follows the text as well as the control. Its prompt guide asks for explicit placement with several subjects
("left vs right, foreground vs background, facing toward vs away") and a camera described relative to the subjects.
Blender knows both for every frame, so the run can write it. render_passes exports two angles per frame, each
[degrees to the camera, picture side it points to]:
  body_yaw   the chest's front          head_yaw   the face (head centre -> nose)
The body angle gives the main words (below 45 degrees: faces the camera; 45 to 135: in profile; above 135: its back
to the camera). When the head turns clearly more toward the camera than the body, a note says so ("looks over its
shoulder toward the camera"): the 2D skeleton cannot tell this, and a model that sees a face tends to turn the
whole body round.
Older pass folders have no angles; then the face-point flags decide (nose and both eyes = front, one eye =
profile, no nose = back). That fallback calls a three-quarter view "front": OpenPose's rule keeps an eye up to
about 75 degrees.
"""
from __future__ import annotations

NOSE, NECK, R_EYE, L_EYE = 0, 1, 14, 15
FRONT_MAX, BACK_MIN = 45.0, 135.0
HEAD_TURN = 30.0        # the head counts as turned toward the camera when it is this much closer to it than the body


def _shown(p):
    return p is not None and len(p) > 2 and bool(p[2])


def from_yaw(yaw: list) -> str:
    deg, side = yaw
    return "front" if deg < FRONT_MAX else "back" if deg > BACK_MIN else f"profile_{side}"


def facing(points: list, yaw: list | None = None) -> str | None:
    """'front', 'profile_left', 'profile_right', 'back' or None (no head points) for one rig frame (COCO-18).
    yaw: [degrees, "left"|"right"] from render_passes (body_yaw or head_yaw); it decides when given."""
    if not points or not (_shown(points[NECK]) or _shown(points[NOSE])):
        return None
    if yaw:
        return from_yaw(yaw)
    if not _shown(points[NOSE]):
        return "back"
    if _shown(points[R_EYE]) and _shown(points[L_EYE]):
        return "front"
    if _shown(points[NECK]):
        return "profile_left" if points[NOSE][0] < points[NECK][0] else "profile_right"
    return "profile_left"


def place(x: float, width: float) -> str:
    return "on the left" if x < 0.4 * width else "on the right" if x > 0.6 * width else "in the centre"


WORDS = {"front": "faces the camera", "profile_left": "is seen in profile, facing left",
         "profile_right": "is seen in profile, facing right", "back": "has its back to the camera"}


def head_note(body: list | None, head: list | None) -> str:
    """Words for a head that turns toward the camera more than the body does ("" if it does not)."""
    if not body or not head or body[0] < FRONT_MAX or head[0] > body[0] - HEAD_TURN:
        return ""
    if body[0] > BACK_MIN:
        return " and looks back over its shoulder toward the camera"
    return " and turns its head to look at the camera" if head[0] < FRONT_MAX else " and turns its head toward the camera"


def _names(chars: list[tuple], width: float) -> list[str]:
    """'the character on the left' ...; two or more in the same place get 'in front' / 'behind' from their depth."""
    if len(chars) == 1:
        return ["the character"]
    where = [place(c[1], width) for c in chars]
    out = []
    for i, c in enumerate(chars):
        same = [j for j, w in enumerate(where) if w == where[i]]
        d = c[4] if len(c) > 4 else None
        if len(same) > 1 and d is not None and all(len(chars[j]) > 4 and chars[j][4] is not None for j in same):
            depths = sorted(chars[j][4] for j in same)
            extra = " in front" if d == depths[0] else " at the back" if d == depths[-1] else ""
            out.append(f"the character {where[i]}{extra}")
        else:
            out.append(f"the character {where[i]}")
    return out


def describe_frame(chars: list[tuple], width: float) -> str:
    """chars: [(rig points, x in the picture[, head yaw[, body yaw[, distance to the camera]]])] -> one sentence."""
    parts = []
    chars = sorted(chars, key=lambda c: c[1])
    for c, who in zip(chars, _names(chars, width)):
        pts = c[0]
        head = c[2] if len(c) > 2 else None
        body = c[3] if len(c) > 3 else None
        f = facing(pts, body or head)
        if f:
            parts.append(f"{who} {WORDS[f]}{head_note(body, head)}")
    if not parts:
        return ""
    s = "; ".join(parts)
    return s[0].upper() + s[1:]


def _at(seq, i):
    return seq[i] if seq and i < len(seq) else None


def clip_text(char_frames: list[list], xs: list[list], width: float, first: int, last: int,
              yaws: list[list] | None = None, bodies: list[list] | None = None,
              depths: list[list] | None = None) -> str:
    """The facing text of one clip (shot frames first..last): the start, the middle and the end.
    char_frames[c][i] = rig points of character c in shot frame i; xs[c][i] = its x in the picture (None = not in it);
    yaws[c][i] / bodies[c][i] = its head / body yaw (render_passes head_yaw / body_yaw), None = not known;
    depths[c][i] = its distance to the camera (people.json), for 'in front' / 'at the back'."""
    moments = [("At the start", first), ("In the middle", (first + last) // 2), ("At the end", last)]
    n = len(char_frames)
    yaws, bodies, depths = yaws or [None] * n, bodies or [None] * n, depths or [None] * n
    out, prev = [], None
    for label, i in moments:
        chars = [(cf[i], x[i], _at(y, i), _at(b, i), _at(d, i))
                 for cf, x, y, b, d in zip(char_frames, xs, yaws, bodies, depths) if i < len(cf) and x[i] is not None]
        s = describe_frame(chars, width)
        if s and s != prev:
            out.append(f"{label}: {s[0].lower() + s[1:]}.")
        prev = s
    return " ".join(out)
