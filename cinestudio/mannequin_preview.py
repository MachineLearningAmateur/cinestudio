"""The Mannequin view's previews: a character's own mannequin (body + proxy kit) rendered by Blender Workbench
(bpy_jobs/mannequin_preview.py), cached, and the editor's table of labels, ranges and presets.

User, 2026-10-24: "some kind of rendered live image of what the mannequin would look like in the mannequin config";
research and plan: docs/mannequin_preview_research.md (phase 1: exact Blender previews; phase 2 adds a live 3D view).

    <home>/cache/mannequin_preview/<key>/front.png right.png back.png info.json char_<front sha>.png
    <home>/cache/mannequin_height/<body key>.json      the height of a body (reused by character.measure_height)

The key covers the body, the kit spec, the Blender scripts, the rules, the mark presets and the template .blend, so a
change of any of them gives a new preview. One Blender preview runs at a time; a newer request for the same character
supersedes one that still waits (a running Blender is never killed: its result is only dropped).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ("mannequin_preview.py", "proxy_kit.py", "mannequin_body.py", "mannequin_rules.json", "mask_marks.json")
FRAME_M = 2.2                    # the preview frame's height in metres (one scale for every character)
SIZE = (384, 640)
KEEP = 200                       # cache entries kept (least recently used go)

# the editor's body controls (docs/mannequin_preview_research.md 5.3): slider range, typed range = the rules' limits
BODY_UI = [
    {"key": "scale", "label": "Overall height", "min": 0.5, "max": 1.3, "step": 0.01,
     "help": "the whole figure; 1 = the template mannequin"},
    {"key": "head", "label": "Head size", "min": 0.7, "max": 2.0, "step": 0.01,
     "help": "with the mask and the hair parts; chibi about 1.5"},
    {"key": "neck", "label": "Neck length", "min": 0.2, "max": 1.5, "step": 0.01, "help": "the head keeps its size"},
    {"key": "torso", "label": "Torso length", "min": 0.6, "max": 1.4, "step": 0.01,
     "help": "spine and chest; the arms and the head keep their sizes"},
    {"key": "legs", "label": "Leg length", "min": 0.5, "max": 1.4, "step": 0.01, "help": "the feet stay on the floor"},
    {"key": "arms", "label": "Arm length", "min": 0.6, "max": 1.4, "step": 0.01, "help": "both arms"},
]
PRESETS = [
    {"id": "adult", "label": "Adult", "body": {}, "note": "the template mannequin (1.81 m)"},
    {"id": "teen", "label": "Teen", "body": {"scale": 0.93, "head": 1.08, "legs": 0.97},
     "note": "about 1.67 m (a proposal, not yet tested in a video)"},
    {"id": "child", "label": "Child", "body": {"scale": 0.72, "head": 1.3, "torso": 0.92, "legs": 0.85, "arms": 0.92},
     "note": "about 1.23 m (a proposal, not yet tested in a video)"},
    {"id": "chibi", "label": "Chibi", "body": {"scale": 0.86, "head": 1.55, "neck": 0.35, "torso": 0.8, "legs": 0.7,
                                               "arms": 0.85},
     "note": "about 1.30 m: kitsune's body (used in videos)"},
]

_RUN = threading.Lock()          # one Blender preview at a time in this process
_STATE = threading.Lock()        # guards _LATEST / _SEQ (never held while Blender runs)
_LATEST: dict[str, int] = {}     # the newest request per character (an older waiting one is superseded)
_SEQ = [0]


def cache_dir() -> Path:
    from .runner import home
    return home() / "cache"


def _canon(v) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


def _versions() -> str:
    """The scripts' and the template's identity: a changed script or template gives new previews and heights."""
    from . import character
    h = hashlib.sha256()
    for name in SCRIPTS:
        h.update(name.encode())
        h.update((ROOT / "bpy_jobs" / name).read_bytes())
    t = character.default_pose_shot()
    st = t.stat()
    h.update(f"{t}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()


def _asset_sig(spec: dict) -> str:
    """A kit with a modelled head (head.asset) changes when its .blend changes at the same path: the file's size and
    time are part of the key (2026-10-27: a new kitsune head showed the old cached preview)."""
    a = ((spec or {}).get("head") or {}).get("asset")
    f = a.get("file") if isinstance(a, dict) else a
    if not f:
        return ""
    try:
        st = Path(f).stat()
    except OSError:
        return f"{f}|missing"
    return f"{f}|{st.st_size}|{st.st_mtime_ns}"


def key(body: dict, spec: dict) -> str:
    return hashlib.sha256(f"{_canon(body)}|{_canon(spec)}|{_asset_sig(spec)}|{_versions()}".encode()).hexdigest()[:24]


def body_key(body: dict) -> str:
    return hashlib.sha256(f"{_canon(body)}|{_versions()}".encode()).hexdigest()[:24]


def cached_height(body: dict) -> dict | None:
    """The height of `body` from a preview or an earlier measure (no Blender), or None."""
    f = cache_dir() / "mannequin_height" / f"{body_key(body)}.json"
    try:
        r = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return r if r.get("body") == body and (r.get("ref_m") or 0) > 0 else None


def store_height(body: dict, m: float, ref: float) -> dict:
    rec = {"m": round(float(m), 4), "ref_m": round(float(ref), 4), "ratio": round(float(m) / float(ref), 4),
           "body": dict(body)}
    d = cache_dir() / "mannequin_height"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{body_key(body)}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    os.replace(tmp, d / f"{body_key(body)}.json")
    return rec


def _prune(root: Path) -> None:
    dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_atime, reverse=True)
    for p in dirs[KEEP:]:
        shutil.rmtree(p, ignore_errors=True)


def _character_picture(name: str, info: dict, out_dir: Path) -> str | None:
    """The character's own front picture cut out and scaled to the preview's metres-per-pixel, its feet on the same row
    (a side-by-side comparison at one scale). The figure's height is the mannequin's height (the drawing's hair and ears
    stand above it). -> the file, or None without a front."""
    import cv2
    import numpy as np

    from . import character, charsheet
    try:
        c = character.load(name)
    except character.CharacterError:
        return None
    front = Path((c.get("files") or {}).get("front") or "")
    if not front.is_file():
        return None
    sha = hashlib.sha256(front.read_bytes()).hexdigest()[:12]
    out = out_dir / f"char_{sha}.png"
    if out.is_file():
        return str(out)
    img = cv2.imread(str(front))
    if img is None:
        return None
    try:
        cut = charsheet.cutout(img)
    except ValueError:
        return None
    w, h = info["size"]
    fig = (~charsheet.background_mask(cut, 2)).astype(np.uint8)
    ys = np.where(fig.any(1))[0]
    top, bot = int(ys.min()), int(ys.max()) + 1
    want = info["height"] * h / info["frame_m"]                    # the body's pixel height in the renders
    k = want / max(1, bot - top)
    small = cv2.resize(cut, (max(1, int(cut.shape[1] * k)), max(1, int(cut.shape[0] * k))),
                       interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    alpha = cv2.resize(fig * 255, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((h, w, 4), np.uint8)
    feet = int(round(info["feet_row"]))
    oy = feet - int(round(bot * k))
    ox = (w - small.shape[1]) // 2
    y0, y1 = max(0, oy), min(h, oy + small.shape[0])
    x0, x1 = max(0, ox), min(w, ox + small.shape[1])
    if y1 > y0 and x1 > x0:
        canvas[y0:y1, x0:x1, :3] = small[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        canvas[y0:y1, x0:x1, 3] = alpha[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    cv2.imwrite(str(out), canvas)
    return str(out)


def preview(name: str, body: dict, spec: dict) -> dict:
    """The preview of a DRAFT body and kit for character `name` (both checked first; nothing is saved).
    -> {key, height_m, ref_m, ratio, files: {front, right, back, character}, cached} or {superseded: True}."""
    from . import blender, character
    body = character.check_body(body)
    character.validate_proxy(spec)
    k = key(body, spec)
    root = cache_dir() / "mannequin_preview"
    d = root / k
    with _STATE:
        _SEQ[0] += 1
        token = _SEQ[0]
        _LATEST[name] = token
    cached = (d / "info.json").is_file()
    if not cached:
        with _RUN:                                                # one Blender preview at a time
            with _STATE:
                newer = _LATEST.get(name) != token
            if newer:                                             # a newer draft came while this one waited
                return {"superseded": True}
            root.mkdir(parents=True, exist_ok=True)
            work = root / f".{k}.{os.getpid()}.work"
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir()
            (work / "body.json").write_text(_canon(body), encoding="utf-8")
            (work / "spec.json").write_text(_canon(spec), encoding="utf-8")
            try:
                blender.run_job("mannequin_preview.py", character.default_pose_shot(),
                                ["--body", str(work / "body.json"), "--spec", str(work / "spec.json"),
                                 "--out", str(work / "out"), "--size", f"{SIZE[0]}x{SIZE[1]}", "--frame", str(FRAME_M)],
                                log=work / "preview.log", timeout=180)
                if d.exists():                                    # another request made it meanwhile
                    shutil.rmtree(work, ignore_errors=True)
                else:
                    os.replace(work / "out", d)
                    shutil.rmtree(work, ignore_errors=True)
            except Exception:
                tail = ""
                try:
                    tail = (work / "preview.log").read_text(encoding="utf-8", errors="replace")[-600:]
                finally:
                    shutil.rmtree(work, ignore_errors=True)
                raise character.CharacterError(f"the Blender preview failed: {tail.strip()[-300:]}") from None
            _prune(root)
    info = json.loads((d / "info.json").read_text(encoding="utf-8"))
    os.utime(d)                                                   # recently used (the LRU keeps it)
    store_height(body, info["height"], info["ref"])
    files = {v: str(d / f"{v}.png") for v in info.get("views", ("front", "right", "back"))}
    pic = _character_picture(name, info, d)
    if pic:
        files["character"] = pic
    return {"key": k, "height_m": info["height"], "ref_m": info["ref"], "ratio": round(info["height"] / info["ref"], 4),
            "frame_m": info["frame_m"], "feet_row": info["feet_row"], "size": info["size"], "files": files,
            "cached": cached, "made": time.time()}


def schema() -> dict:
    """What the editor reads: the body controls, the presets, the rules' limits and the mark preset names."""
    from . import character
    rules = json.loads((ROOT / "bpy_jobs" / "mannequin_rules.json").read_text(encoding="utf-8"))
    return {"body": BODY_UI, "presets": PRESETS, "limits": rules["limits"], "keys": rules["keys"],
            "mark_presets": sorted(character.mark_presets()), "band_bones": list(character.BAND_BONES),
            "frame_m": FRAME_M, "size": list(SIZE)}
