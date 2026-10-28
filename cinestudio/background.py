"""Backgrounds for Blender shots: a 360-degree panorama made from the scene text, saved in a small library and put into
the run's own copy of the shot as its world (bpy_jobs/pano_world.py).

Research 2026-10-21 (AGENTS.md "Dynamic backgrounds"): the video keeps about 1.5% of the Blender render, and a grey set
gave grey, changing scenes; a panorama world gives every clip the same place, turning exactly with the camera (Blender puts it
round the shot as a dome: bpy_jobs/pano_world.py; no mask holds it). Model: Qwen-Image-2512 (full model, GGUF Q8) + the
qwen-360-diffusion v2 LoRA (MIT), 2048x1024, graph workflows/qwenimage_pano.

    <home>/backgrounds/<name>/pano.png + bg.json {prompt, scene, style, seed, created}

Two kinds (2026-10-24, user: "can you implement the backplates to the background tab"; "include both panoramas and
regular scene backgrounds"): a 360 PANORAMA (pano.png, any camera) and a BACKPLATE (plate.png + bg.json kind "plate"
and the camera it was made for: one picture at the frame's shape for a still camera, 4-9x sharper than the part of a
panorama that camera sees; runner._plate_background puts it behind the shot with bpy_jobs/plate_world.py).

BUILT-IN backgrounds (2026-10-25, user: "turn it into a built in template" with its backplate): the repo folder
backgrounds/library/<name>/ (the same files; bg.json without personal paths) ships read-only, like the built-in styles
and templates. `load`, `path` and `list_backgrounds` see both libraries; a user background of the same name wins (the
user's library first). Nothing writes into a built-in: delete, stamps and saves under its name are refused; a built-in
panorama's 360 depth goes to the user's cache (pano_depth.cached); a style redraw is a new USER background.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

PANO_PROMPT = ("360 degree panorama with equirectangular projection. {scene}. {style}. The camera stands in the middle "
               "of an open, flat, empty floor area at eye height; the whole place is visible all around, with the horizon "
               "level. No people, no animals, no text.")
NEGATIVE = ("people, person, crowd, character, figure, animal, text, watermark, logo, signature, blurry, low quality, "
            "visible seam, tilted horizon, distorted horizon, fisheye border, camera, tripod, photographer")
# (2026-10-22: a panorama drew a camera on a tripod in the floor: "camera, tripod" added to the negative)
WRAP = 128      # style pass: the panorama is padded with this many wrapped columns on each side (the seam stays whole)
WIDTH, HEIGHT = 2048, 1024
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class BackgroundError(ValueError):
    pass


def library_dir() -> Path:
    """The USER's library (<home>/backgrounds): every new background is written here."""
    from .runner import home
    d = home() / "backgrounds"
    d.mkdir(parents=True, exist_ok=True)
    return d


BUILTIN = Path(__file__).resolve().parent.parent / "backgrounds" / "library"


def builtin_dir() -> Path:
    """The backgrounds that ship with the repository (read-only)."""
    return BUILTIN


def prompt_for(scene: str, style: str = "") -> str:
    scene = (scene or "").strip().rstrip(".")
    if not scene:
        raise BackgroundError("describe the place for the background")
    style = (style or "cinematic fantasy anime illustration, painted background art").strip().rstrip(".")
    return PANO_PROMPT.format(scene=scene[:1].upper() + scene[1:], style=style[:1].upper() + style[1:])


def auto_name(prompt: str, seed: int) -> str:
    return "bg_" + hashlib.sha1(f"{prompt}|{seed}".encode()).hexdigest()[:10]


FILES = {"pano": "pano.png", "plate": "plate.png"}       # the picture of each kind


def _file_of(d: Path) -> Path | None:
    for f in FILES.values():
        if (d / f).is_file():
            return d / f
    return None


def path(name: str) -> Path:
    """The picture of a background: the user's library first, then the built-in one."""
    if not NAME_RE.match(name or ""):
        raise BackgroundError(f"background name {name!r}: letters, digits, '-' and '_'")
    p = _file_of(library_dir() / name) or _file_of(BUILTIN / name)
    if p is None:
        raise BackgroundError(f"no background {name!r}")
    return p


def is_builtin(name: str) -> bool:
    """True when `name` is a built-in background that no user background of the same name hides."""
    if not NAME_RE.match(name or ""):
        return False
    return _file_of(library_dir() / name) is None and _file_of(BUILTIN / name) is not None


def has_builtin(name: str) -> bool:
    """A built-in background of this name exists (also when a user background of the same name hides it)."""
    return bool(NAME_RE.match(name or "")) and _file_of(BUILTIN / name) is not None


def taken(name: str) -> bool:
    """The name is used in the user's library (also a reserved folder) or by a built-in background."""
    return (library_dir() / name).exists() or (BUILTIN / name).exists()


def refuse_builtin(name: str, what: str = "changed") -> None:
    """Built-in backgrounds are read-only (they ship with cinestudio)."""
    if is_builtin(name):
        raise BackgroundError(f"{name!r} is a built-in background (it ships with cinestudio: backgrounds/library/"
                              f"{name}) and cannot be {what}")


def load(name: str) -> dict:
    p = path(name)
    try:
        rec = json.loads((p.parent / "bg.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rec = {}
    builtin = p.parent.parent.resolve() == BUILTIN.resolve()
    return {**rec, "name": name, "file": str(p), "kind": "plate" if p.name == FILES["plate"] else "pano",
            "builtin": builtin}


def list_backgrounds(kind: str | None = None) -> list[dict]:
    """The user's backgrounds (newest first), then the built-in ones that no user background hides (by name)."""
    out, built = [], []
    for root, rows in ((library_dir(), out), (BUILTIN, built)):
        for d in sorted(root.iterdir()) if root.is_dir() else []:
            if not (d.is_dir() and NAME_RE.match(d.name) and _file_of(d) is not None):
                continue
            if rows is built and _file_of(library_dir() / d.name) is not None:
                continue                                      # the user's background of the same name wins
            r = load(d.name)
            if kind is None or r["kind"] == kind:
                rows.append(r)
    out.sort(key=lambda r: r.get("created") or 0, reverse=True)
    built.sort(key=lambda r: (r.get("order", 99), r["name"]))
    return out + built


def require_pano(name: str) -> dict:
    """The record of a 360 panorama; a backplate is refused (no 360 depth, no wrap-round redraw)."""
    r = load(name)
    if r["kind"] != "pano":
        raise BackgroundError(f"{name} is a backplate (one picture for a still camera), not a 360 panorama")
    return r


def save(png: str | Path, prompt: str, seed: int, scene: str = "", style: str = "", name: str | None = None) -> dict:
    """Store a panorama in the library (a copy); the name is derived from the prompt and the seed when not given."""
    name = name or auto_name(prompt, seed)
    if (BUILTIN / name).exists():
        raise BackgroundError(f"{name!r} is the name of a built-in background: choose another name")
    d = library_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "pano.tmp.png"
    shutil.copyfile(png, tmp)
    os.replace(tmp, d / "pano.png")
    (d / "bg.json").write_text(json.dumps({"prompt": prompt, "scene": scene, "style": style, "seed": int(seed),
                                           "created": time.time()}, indent=1), encoding="utf-8")
    return load(name)


# ---- backplates (one picture for a still camera)
PLATE_PROMPT = ("{scene}. A wide view of the place from one still camera at eye level, about {h:.1f} m above the ground "
                "and looking {look}; the horizon line is {where} of the picture. The middle of the picture is open, "
                "flat, empty ground from the near foreground to about 10 m away, where a person could dance. "
                "{style}The image contains no people, no animals and no recognizable text.")
PLATE_WIDTH = 1920


def plate_size(cam: dict) -> tuple[int, int]:
    """The backplate's size for a camera (bpy_jobs/plate_world.py --info): 1920 wide, the frame's shape."""
    rx, ry = cam["resolution"]
    return PLATE_WIDTH, max(512, int(round(PLATE_WIDTH * ry / rx / 16)) * 16)


def plate_prompt(scene: str, style: str, cam: dict) -> str:
    """The Qwen-Image text of a backplate: the place, then the camera's height, look direction and horizon place."""
    scene = (scene or "").strip().rstrip(".")
    if not scene:
        raise BackgroundError("describe the place for the backplate")
    hz = float(cam.get("horizon_from_top", 0.5))
    if abs(hz - 0.5) < 0.04:
        where = "at the middle"
    elif 0.0 < hz < 1.0:
        where = f"{round(hz * 100)}% down from the top"
    else:
        where = "above the top" if hz <= 0.0 else "below the bottom"
    pitch = float(cam.get("pitch_deg", 0.0))
    if abs(pitch) < 3:
        look = "straight ahead"
    else:
        look = f"slightly down ({abs(pitch):.0f} degrees)" if pitch < 0 else f"slightly up ({pitch:.0f} degrees)"
    style = (style or "").strip().rstrip(".")
    return PLATE_PROMPT.format(scene=scene, h=float(cam.get("height_m", 1.5)), look=look, where=where,
                               style=(style + ". ") if style else "")


def camera_info(shot: str | Path) -> dict:
    """The still camera of a .blend (bpy_jobs/plate_world.py --info). A PIVOT camera (one place, it only turns or
    zooms) is described by its cover camera (`pivot`, `cover`); a camera that changes its place is refused."""
    import tempfile

    from . import blender
    shot = Path(shot)
    if not shot.is_file():
        raise BackgroundError(f"no shot {shot}")
    with tempfile.TemporaryDirectory() as td:
        info = Path(td) / "camera.json"
        blender.run_job("plate_world.py", shot, ["--info", str(info)], log=Path(td) / "plate_world.log")
        cam = json.loads(info.read_text(encoding="utf-8"))
    if cam.get("moves") and not cam.get("pivot"):
        raise BackgroundError("a backplate needs a camera that stays in one place (it may turn and zoom): this shot's "
                              "camera changes its place (use a 360 panorama)")
    return cam


def camera_summary(cam: dict | None) -> str:
    if not cam:
        return ""
    return (f"camera {float(cam.get('height_m', 0)):.2f} m high, looking {float(cam.get('pitch_deg', 0)):+.0f} deg, "
            f"horizon {round(float(cam.get('horizon_from_top', 0.5)) * 100)}% from the top")


def pixels_sha1(img) -> str:
    """The SHA-1 of a decoded picture: its shape and channel layout, then its pixels (two pictures with the same bytes
    but another shape differ)."""
    h = hashlib.sha1(f"{img.shape}|{img.dtype}".encode())
    h.update(img.tobytes())
    return h.hexdigest()


def save_plate(png: str | Path, prompt: str = "", seed: int | None = None, scene: str = "", style: str = "",
               name: str | None = None, camera: dict | None = None, source: str = "text", **extra) -> dict:
    """Store a backplate in the library (a copy) with the camera it was made for. Without a name, the name comes from
    the picture's CONTENT ("plate_" + the pixels' SHA-1): the same picture saved again gives back its record, never a
    second copy (Codex review 2026-10-24: a name from the file size let two pictures replace each other). An existing
    background is NEVER replaced: its folder is reserved atomically (mkdir without exist_ok)."""
    import secrets as _secrets

    import cv2
    img = cv2.imread(str(png), cv2.IMREAD_COLOR)
    if img is None:
        raise BackgroundError(f"cannot read the picture {png}")
    sha = pixels_sha1(img)
    auto = not name
    name = name or "plate_" + sha[:12]
    if not NAME_RE.match(name):
        raise BackgroundError(f"background name {name!r}: letters, digits, '-' and '_' (at most 64)")
    if (BUILTIN / name).exists():                         # a built-in background is never hidden or replaced
        if auto:
            try:
                have = load(name)
            except BackgroundError:
                have = {}
            if have.get("kind") == "plate" and have.get("sha1") == sha:
                return have
        raise BackgroundError(f"{name!r} is the name of a built-in background: choose another name")
    d = library_dir() / name
    try:
        d.mkdir()                                         # the reservation: fails when the name exists
    except FileExistsError:
        if auto:
            try:
                have = load(name)
            except BackgroundError:
                have = {}
            if have.get("kind") == "plate" and have.get("sha1") == sha:
                return have                               # the same picture: already in the library
        raise BackgroundError(f"a background {name!r} exists already: choose another name") from None
    try:
        tmp = d / f"plate.{_secrets.token_hex(4)}.tmp.png"
        if not cv2.imwrite(str(tmp), img):
            raise BackgroundError(f"cannot write {tmp}")
        meta = d / f"bg.{_secrets.token_hex(4)}.tmp.json"
        meta.write_text(json.dumps({"kind": "plate", "prompt": prompt, "scene": scene, "style": style,
                                    "seed": None if seed is None else int(seed), "camera": camera, "sha1": sha,
                                    "source": source, "created": time.time(), **extra}, indent=1), encoding="utf-8")
        os.replace(meta, d / "bg.json")
        os.replace(tmp, d / FILES["plate"])               # the picture last: a listed plate is always complete
    except BaseException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    return load(name)


def stamp_style(name: str, style_ref: str | None) -> dict:
    """Record the library style a background was made in (bg.json style_ref + style_version)."""
    if not style_ref:
        return load(name)
    refuse_builtin(name, "stamped with another style")
    from . import styles
    p = path(name).parent / "bg.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec.update(styles.stamp(style_ref))
    tmp = p.with_suffix(".style.tmp")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return load(name)


def style_words(style: str = "", style_ref: str | None = None) -> str:
    """The look words of a background job: a library style's words when `style_ref` is given (and no own words)."""
    if style_ref and not (style or "").strip():
        from . import styles
        try:
            return styles.text(style_ref)
        except styles.StyleError as exc:
            raise BackgroundError(str(exc)) from exc
    return style or ""


def make_plate(scene: str, shot: str | Path, style: str = "", seed: int | None = None, name: str | None = None,
               on_event=None) -> dict:
    """A new library backplate for the still camera of `shot` (Qwen-Image-2512 through the panorama graph with its
    360 LoRA at 0, the same recipe as a run's plate). A GPU job: the caller holds a queue turn (cli `background
    plate`). Returns the saved record."""
    import secrets as _secrets

    from . import character
    if name and (not NAME_RE.match(name) or taken(name)):
        raise BackgroundError(f"name {name!r}: letters, digits, '-' and '_', and not taken")
    cam = camera_info(shot)
    prompt = plate_prompt(scene, style, cam)
    seed = int(seed) if seed is not None else _secrets.randbelow(2 ** 31)
    w, h = plate_size(cam)
    work = library_dir() / ".work"
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"plate_{_secrets.token_hex(4)}.png"
    try:
        character.image_job("qwenimage", "pano", {"prompt": prompt, "negative": NEGATIVE, "seed": seed, "width": w,
                                                  "height": h, "lora_strength": 0.0}, {}, out, "bg_plate",
                            on_event=on_event)
        return save_plate(out, prompt, seed, scene=scene.strip(), style=(style or "").strip(), name=name, camera=cam,
                          source="text", shot=Path(shot).parent.name)
    finally:
        out.unlink(missing_ok=True)


SET_INSTRUCTION = (
    "Redraw Picture 1, a simple clay render of a 3D set seen from one still camera, as a finished picture of this "
    "place: {scene}. Every object in Picture 1 keeps exactly its place, size, shape and outline, and the camera, the "
    "floor, the walls and the horizon stay exactly where they are; only the materials, colours, textures and light "
    "change, so each clay shape becomes the real object it stands for. {style}. The place is empty: no people and no "
    "characters. Keep everything else exactly the same.")
SET_NEGATIVE = ("people, person, character, figure, mannequin, crowd, moved objects, extra furniture, different layout, "
                "different camera angle, watermark, blurry, low quality")
# no fixed text rule (2026-10-27, user: "the background creation prompt should be flexible ... as long as the look
# matches the blender reference it's good and we can always direct what banners contain what text"): the scene words
# decide the signs (quote their words)


def _stable(ww: int, hh: int) -> bool:
    """TextEncodeQwenImageEditPlus keeps ww x hh (its ~1 MP rescale rounds to multiples of 8: nodes_qwen.py)."""
    import math
    s = math.sqrt(1024 * 1024 / (ww * hh))
    return round(ww * s / 8) * 8 == ww and round(hh * s / 8) * 8 == hh


def edit_layout(w: int, h: int) -> tuple[int, int, tuple[int, int, int, int]]:
    """(ew, eh, (x0, y0, iw, ih)): the size the set render goes to Qwen at (multiples of 16, ~1 MP, kept as it is by
    the text encoder: no crop, no stretch) and the box inside it that holds the frame with its exact shape. For 16:9
    and 1:1 frames the box is the whole picture; other shapes (16:10, 4:3, 21:9, ...) have no such pair, so the frame
    is placed inside the nearest pair and the rest is padded (Fable review 2026-10-24: edit_size refused them)."""
    aspect = w / h
    best = None
    for hh in range(16 * 30, 16 * 121, 16):
        for ww in range(16 * 30, 16 * 121, 16):
            if abs(ww * hh - 1024 * 1024) > 0.02 * 1024 * 1024 or not _stable(ww, hh):
                continue
            if ww / hh >= aspect:                      # pad left and right
                ih, iw = hh, min(ww, int(round(hh * aspect)))
            else:                                      # pad top and bottom
                iw, ih = ww, min(hh, int(round(ww / aspect)))
            waste = 1 - (iw * ih) / (ww * hh)
            err = abs((iw / ih) - aspect) / aspect
            key = (round(waste, 4) + err * 10, abs(ww * hh - 1024 * 1024))
            if best is None or key < best[0]:
                best = (key, ww, hh, ((ww - iw) // 2, (hh - ih) // 2, iw, ih))
    if best is None:
        raise BackgroundError(f"no edit size for a {w}x{h} frame")
    return best[1], best[2], best[3]


def edit_size(w: int, h: int) -> tuple[int, int]:
    """The frame's box inside edit_layout (the size the set is RENDERED at)."""
    return edit_layout(w, h)[2][2:]


def _no_kontext_crop(graph: dict) -> None:
    """The set render goes to Qwen at its own size: the FluxKontextImageScale node ("scale": a centre crop to one of 17
    fixed sizes) is taken out; both text encoders and the sampling latent read the loaded picture."""
    for node, field in (("pos_txt", "image1"), ("neg_txt", "image1"), ("enc", "pixels")):
        if node in graph and graph[node]["inputs"].get(field) == ["scale", 0]:
            graph[node]["inputs"][field] = ["img", 0]
    left = [f"{k}.{f}" for k, n in graph.items() if isinstance(n, dict)
            for f, v in (n.get("inputs") or {}).items() if isinstance(v, (list, tuple)) and v and v[0] == "scale"]
    if left:                                           # a changed graph: fail here, not inside ComfyUI
        raise BackgroundError(f"the edit graph still uses the Kontext scale node at {left}")
    graph.pop("scale", None)


def edge_fit(ref, img, tol: int = 3) -> float:
    """How well the edges of `img` lie on the edges of `ref` (an F-score of Canny edges matched within `tol` px; 1.0 =
    the same edges). The finished picture adds texture edges, so about 0.5 is a good fit for a set backplate."""
    import cv2
    import numpy as np

    def edges(i):
        return cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), (5, 5), 0), 40, 120) > 0
    a, b = edges(ref), edges(img)
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    p = (b & (cv2.dilate(a.astype(np.uint8), k) > 0)).sum() / max(1, b.sum())
    r = (a & (cv2.dilate(b.astype(np.uint8), k) > 0)).sum() / max(1, a.sum())
    return float(2 * p * r / max(1e-9, p + r))


def align_to_set(set_img, plate) -> tuple:
    """Qwen edits can draw the whole picture a few percent smaller or shifted (gym_from_set_v3: x0.98 / x0.96 and 44 px
    lower; edge fit 0.26). An affine fit (ECC on edge-strength maps, coarse to fine) brings the picture back onto the
    set render; the thin band it uncovers is inpainted. Used only when the change is small (scale 0.9-1.1, rotation
    < 2 deg, shift < 8% of the size) and the edge fit gets better by at least 0.02. Returns (picture, info)."""
    import math

    import cv2
    import numpy as np

    def strength(i):
        x = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = cv2.GaussianBlur(np.sqrt(cv2.Sobel(x, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(x, cv2.CV_32F, 0, 1) ** 2), (0, 0), 3)
        return m / (m.max() + 1e-6)
    h, w = plate.shape[:2]
    a, b = strength(set_img), strength(plate)
    warp = np.eye(2, 3, dtype=np.float32)
    info = {"applied": False, "fit_before": round(edge_fit(set_img, plate), 3)}
    try:
        for s in (0.25, 0.5, 1.0):
            ww = warp.copy()
            ww[:, 2] *= s
            _, ww = cv2.findTransformECC(cv2.resize(a, None, fx=s, fy=s), cv2.resize(b, None, fx=s, fy=s), ww,
                                         cv2.MOTION_AFFINE,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
            warp = ww.copy()
            warp[:, 2] /= s
    except cv2.error as exc:
        info["note"] = f"no fit: {str(exc)[:80]}"
        return plate, info
    sx, sy = float(np.linalg.norm(warp[:, 0])), float(np.linalg.norm(warp[:, 1]))
    rot = math.degrees(math.atan2(float(warp[1, 0]), float(warp[0, 0])))
    info.update(scale=[round(sx, 4), round(sy, 4)], rotation_deg=round(rot, 3),
                shift=[round(float(warp[0, 2]), 1), round(float(warp[1, 2]), 1)])
    if not (0.9 <= sx <= 1.1 and 0.9 <= sy <= 1.1 and abs(rot) < 2 and abs(warp[0, 2]) < 0.08 * w
            and abs(warp[1, 2]) < 0.08 * h):
        info["note"] = "the fit is too large a change: not applied"
        return plate, info
    flags = cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP
    out = cv2.warpAffine(plate, warp, (w, h), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), warp, (w, h), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                           borderValue=0)
    hole = cv2.dilate((valid == 0).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    if hole.any():
        out = cv2.inpaint(out, hole, 5, cv2.INPAINT_TELEA)
    after = edge_fit(set_img, out)
    info["fit_after"] = round(after, 3)
    if after < info["fit_before"] + 0.02:
        info["note"] = "no real gain: not applied"
        return plate, info
    info["applied"] = True
    return out, info


def render_set(shot: str | Path, out: str | Path, width: int, height: int, frame: int | None = None,
               cover: dict | None = None) -> dict:
    """The template's static set alone from its camera (bpy_jobs/render_set.py: mannequins, props and rigs hidden,
    Workbench, frame 1). Returns {"hidden", "kept"} (the object names)."""
    import tempfile

    from . import blender
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "render_set.log"
        extra = []
        if cover:                                             # a pivot camera: the set seen by its cover camera
            cf = Path(td) / "cover.json"
            cf.write_text(json.dumps(cover), encoding="utf-8")
            extra = ["--cover", str(cf)]
        blender.run_job("render_set.py", shot, ["--out", str(out), "--width", str(int(width)),
                                                "--height", str(int(height)),
                                                *(["--frame", str(int(frame))] if frame is not None else []), *extra],
                        log=log)
        line = next((ln for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()
                     if ln.startswith("SET ")), None)
    if not Path(out).is_file():
        raise BackgroundError("the set render was not written")
    return json.loads(line[4:]) if line else {"hidden": [], "kept": []}


def make_set_plate(scene: str, shot: str | Path, style: str = "", seed: int | None = None, name: str | None = None,
                   on_event=None, negative_extra: str = "") -> dict:
    """A backplate DRAWN OVER THE TEMPLATE'S OWN SET (2026-10-24, user: "we should be generating the backgrounds based
    off of the blender template ... right now it feels like we are praying that the background would fit"): Blender
    renders the set alone from the template's still camera; Qwen-Image-Edit redraws that render as the place in the
    given style, keeping every object's place and outline (Qwen kept the Blender props' places in the first-frame tests).
    The picture then agrees with the layout's geometry. A GPU job: the caller holds a queue turn. Returns the record."""
    import secrets as _secrets

    import cv2

    from . import character
    if name and (not NAME_RE.match(name) or taken(name)):
        raise BackgroundError(f"name {name!r}: letters, digits, '-' and '_', and not taken")
    if not (scene or "").strip():                              # the template's own words for its Set layer
        try:
            from . import shots
            scene = (((shots.library_record(Path(shot)) or {}).get("layers") or {}).get("descriptions") or {}).get("set") or ""
        except Exception:  # noqa: BLE001
            scene = ""
        if not scene.strip():
            raise BackgroundError("say what the set stands for (--scene), or describe the template's Set layer: "
                                  "shot layers NAME --set \"...\"")
    cam = camera_info(shot)                                    # refuses a moving camera
    w, h = plate_size(cam)
    ew, eh, (bx, by, bw, bh) = edit_layout(w, h)              # Qwen gets ew x eh; the frame is the box inside
    seed = int(seed) if seed is not None else _secrets.randbelow(2 ** 31)
    work = library_dir() / ".work"
    work.mkdir(parents=True, exist_ok=True)
    tok = _secrets.token_hex(4)
    set_png, out = work / f"set_{tok}.png", work / f"setplate_{tok}.png"
    try:
        info = render_set(shot, set_png, bw, bh, frame=cam.get("frame_start"), cover=cam.get("cover"))
        if (bw, bh) != (ew, eh):                               # pad to the stable size (the border repeated)
            pic = cv2.imread(str(set_png))
            pic = cv2.copyMakeBorder(pic, by, eh - bh - by, bx, ew - bw - bx, cv2.BORDER_REPLICATE)
            cv2.imwrite(str(set_png), pic)
        if on_event:
            on_event({"event": "set_rendered", "kept": len(info.get("kept") or []), "hidden": len(info.get("hidden") or [])})
        if len(info.get("kept") or []) <= 1 and on_event:     # only a floor: the picture is mostly invented
            on_event({"event": "set_sparse", "note": "the template's set has (almost) no objects: a text backplate "
                                                     "(background plate) gives the same result"})
        instruction = SET_INSTRUCTION.format(scene=scene.strip().rstrip("."),
                                             style=(style or "").strip().rstrip(".") or "The same rendering style")
        # the style's own excluded looks too (2026-10-27: a 2D style needs its 3D/photo negatives; cfg 3 applies them)
        neg = ", ".join(x for x in (SET_NEGATIVE, (negative_extra or "").strip().rstrip(".")) if x)
        values, _ = character.quality_values("quality", {"negative": neg}, character.EDIT_SETTINGS)
        values.update(instruction=instruction, seed=seed)
        wf = character.ProfileWorkflow("qwenedit", "edit")
        character.image_job("qwenedit", "edit", values, {wf.targets("image")[0]: set_png}, out, "bg_setplate",
                            on_event=on_event, graph_edit=_no_kontext_crop)
        img = cv2.imread(str(out), cv2.IMREAD_COLOR)
        if img is None:
            raise BackgroundError("the set backplate was not written")
        if (img.shape[1], img.shape[0]) != (ew, eh):
            raise BackgroundError(f"Qwen returned {img.shape[1]}x{img.shape[0]}, not the set render's {ew}x{eh}: the "
                                  "geometry would not match")
        img = img[by:by + bh, bx:bx + bw]                       # the frame's box (the padding cut off)
        set_box = cv2.imread(str(set_png))[by:by + bh, bx:bx + bw]
        img, fit = align_to_set(set_box, img)                   # Qwen's small zoom or shift taken back
        if on_event:
            on_event({"event": "set_fit", **fit})
        if (bw, bh) != (w, h):                                  # up to the frame's size, the same shape (no crop)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(out), img)
        rec = save_plate(out, instruction, seed, scene=scene.strip(), style=(style or "").strip(), name=name,
                         camera=cam, source="set", shot=Path(shot).parent.name, set_objects=info.get("kept") or [],
                         set_fit=fit)
        shutil.copyfile(set_png, Path(rec["file"]).parent / "set.png")   # the clay render it was drawn over
        return rec
    finally:
        set_png.unlink(missing_ok=True)
        out.unlink(missing_ok=True)


def import_plate(png: str | Path, name: str | None = None, scene: str = "") -> dict:
    """Your own picture of the place as a library backplate (it is fitted to the frame's shape in each run)."""
    return save_plate(png, scene=scene.strip(), name=name, source="import")   # refuses an existing name


def jobs_waiting(name: str) -> list[str]:
    """The queue refs of background jobs for this panorama (app refs "bg-<kind>-<name>")."""
    from . import workqueue
    return [t["ref"] for t in workqueue.tickets() if t.get("kind") == "lab" and str(t.get("ref", "")).count("-") >= 2
            and str(t["ref"]).startswith("bg-") and str(t["ref"]).split("-", 2)[2] == name]


def delete(name: str) -> dict:
    p = path(name)
    refuse_builtin(name, "deleted")
    if jobs_waiting(name):                                 # a waiting depth or style worker would fail on a missing file
        raise BackgroundError(f"{name} has a job in the queue: let it finish first")
    shutil.rmtree(p.parent)
    return {"deleted": name}


# ---- job records of the app's background workers (the Backgrounds tab shows a failed job until it is dismissed)
JOB_KEEP = 24 * 3600


def _job_file(ref: str) -> Path:
    if not re.match(r"^bg-[a-z]+-[A-Za-z0-9_-]{1,64}$", ref):
        raise BackgroundError(f"bad job ref {ref!r}")
    return library_dir() / ".work" / f"{ref}.job.json"


def record_job(ref: str, status: str, label: str = "", error: str = "") -> None:
    f = _job_file(ref)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"ref": ref, "status": status, "label": label, "error": error[:600], "t": time.time()}),
                 encoding="utf-8")


def failed_jobs() -> list[dict]:
    """Failed jobs of the last day, newest first (older records are removed)."""
    out = []
    for f in (library_dir() / ".work").glob("bg-*.job.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if time.time() - float(r.get("t") or 0) > JOB_KEEP:
            f.unlink(missing_ok=True)
        elif r.get("status") == "failed":
            out.append(r)
    return sorted(out, key=lambda r: -float(r.get("t") or 0))


def dismiss_job(ref: str) -> dict:
    _job_file(ref).unlink(missing_ok=True)
    return {"dismissed": ref}


RESERVED = "reserved.json"        # a style pass's name reservation (written before its GPU job, removed when saved)


def sweep_reserved(age: float = 3600) -> list[str]:
    """Name folders that a killed style pass left behind: empty, or holding only its reservation file. They are
    removed after `age` seconds when NO style pass waits in the queue (a waiting pass holds the reservation of a
    destination name that its ticket does not show), so the name is free again. A folder with a picture stays."""
    from . import workqueue
    gone = []
    if not library_dir().is_dir():
        return gone
    if any(t.get("kind") == "lab" and str(t.get("ref", "")).startswith(("bg-style-", "bgstyle_"))
           for t in workqueue.tickets()):
        return gone
    for d in library_dir().iterdir():
        try:
            if not (d.is_dir() and NAME_RE.match(d.name)):
                continue
            files = [f.name for f in d.iterdir()]
            if files in ([], [RESERVED]) and time.time() - d.stat().st_mtime > age:
                (d / RESERVED).unlink(missing_ok=True)
                d.rmdir()
                gone.append(d.name)
        except OSError:
            continue
    return gone


def style_pass(name: str, style_lora: str, keep: float = 0.5, strength: float = 0.8, seed: int = 333,
               style: str = "", new_name: str | None = None, on_event=None, profile: str | None = None) -> dict:
    """Redraw a library panorama with LTX-2.5 and a style LoRA (the video's LoRA; the panorama model is another family
    and cannot load it), so the background has the video's look. The panorama is padded with WRAP columns from its
    other side (the seam stays continuous through the model), cropped back, and its seam closed. Saved as a NEW
    library panorama (default name: NAME_<lora stem>); the original stays.

    Image-to-image with the "redraw" graph (setup/make_redraw_graphs.py): the panorama is encoded and the sampler starts
    at noise S = 1 - keep, so about `keep` of its structure stays. (Before 2026-10-22 the restyle graph held the picture
    as frame 0 of a still clip with a noise mask: keep 0.5 and 0.2 both returned the same photo.) `profile`: default the distilled
    LTX-2.5 (runner.LAYOUT_PROFILE)."""
    import secrets

    import cv2
    import numpy as np

    from . import models as _models
    from .workflow import ProfileWorkflow
    rec = require_pano(name)                               # a backplate has no wrap-round seam to keep
    src = path(name)
    if not 0.0 <= keep <= 0.95:
        raise BackgroundError("keep must be between 0 and 0.95 (about this part of the panorama's structure stays)")
    # the destination: never an existing panorama (Codex 10-22: a 64-character source name made the default name equal
    # to the source; a second pass with the same LoRA overwrote the first). Reserved before the GPU job.
    if new_name:
        nn = new_name
    else:
        stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(style_lora).stem)[:12].strip("-") or "lora"
        nn = f"{name[:30]}_{stem}_k{round(float(keep) * 100):02d}_s{abs(int(seed)) % 10**6}"[:58]
        while taken(nn):                                   # 58 + "_" + 4 hex characters = at most 63
            nn = f"{nn[:58]}_{secrets.token_hex(2)}"
    if not NAME_RE.match(nn):
        raise BackgroundError(f"new name {nn!r}: letters, digits, '-' and '_' (at most 64)")
    dest = library_dir() / nn
    if taken(nn):
        raise BackgroundError(f"a panorama {nn!r} exists already: give another --new-name")
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise BackgroundError(f"cannot read {src}")
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    padded = np.concatenate([img[:, -WRAP:], img, img[:, :WRAP]], axis=1)
    work = library_dir() / ".work" / f"style_{name}"     # never next to the source (a built-in is read-only)
    work.mkdir(parents=True, exist_ok=True)
    pin = work / "padded.png"
    cv2.imwrite(str(pin), padded)
    scene = rec.get("scene") or rec.get("prompt") or "the place"
    prompt = (f"A 360 degree panorama with equirectangular projection: {scene}. {style}. A still image: nothing moves, "
              f"the camera does not move. No people, no camera, no tripod.")
    from . import runner
    prof = profile or runner.LAYOUT_PROFILE
    start = round(1.0 - float(keep), 4)
    wf = ProfileWorkflow(prof, "redraw", models=_models.job_bundle())
    sig = (runner.dev_source_sigmas(start, WIDTH + 2 * WRAP, HEIGHT, 9) if wf.capabilities.get("recipe") == "full"
           else runner.source_sigmas(start))
    values = {"prompt": prompt, "negative": "worst quality, blurry, distorted, text, watermark, people, camera, tripod",
              "width": WIDTH + 2 * WRAP, "height": HEIGHT, "seed": int(seed), "source_sigmas": sig,
              "style_lora": style_lora, "style_lora_strength": float(strength)}
    token = secrets.token_hex(3)
    out = work / "styled_padded.png"
    library_dir().mkdir(parents=True, exist_ok=True)
    try:
        dest.mkdir()                                       # reserved: a parallel pass cannot take the same name
        (dest / RESERVED).write_text(json.dumps({"from": name, "t": time.time()}), encoding="utf-8")
    except FileExistsError as exc:
        raise BackgroundError(f"a panorama {nn!r} exists already: give another --new-name") from exc
    try:
        return _finish_style(name, nn, rec, prof, values, wf, pin, out, work, prompt, seed, style, style_lora, keep,
                             strength, start, token, on_event, scene)
    except BaseException:
        (dest / RESERVED).unlink(missing_ok=True)          # the job failed: free the reserved name
        if dest.is_dir() and not any(dest.iterdir()):
            dest.rmdir()
        raise


def _finish_style(name, nn, rec, prof, values, wf, pin, out, work, prompt, seed, style, style_lora, keep, strength,
                  start, token, on_event, scene):
    import cv2
    from . import character
    character.image_job(prof, "redraw", dict(values, output_prefix_last=f"cinestudio/bg_{name}_style_{token}_last"),
                        {wf.targets("reference")[0]: pin}, out, f"bg_{name}_style", on_event=on_event)
    styled = cv2.imread(str(out), cv2.IMREAD_COLOR)
    if styled is None:
        raise BackgroundError("the style pass wrote no image")
    styled = cv2.resize(styled, (WIDTH + 2 * WRAP, HEIGHT), interpolation=cv2.INTER_CUBIC)[:, WRAP:WRAP + WIDTH]
    final = work / "styled.png"
    cv2.imwrite(str(final), styled)
    fix_seam(final)
    saved = save(final, prompt, seed, scene=scene, style=style or rec.get("style") or "", name=nn)
    meta = json.loads((library_dir() / nn / "bg.json").read_text(encoding="utf-8"))
    meta.update(styled_from=name, style_lora=style_lora, keep=float(keep), lora_strength=float(strength),
                start_noise=start, profile=prof, method="redraw")
    (library_dir() / nn / "bg.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (library_dir() / nn / RESERVED).unlink(missing_ok=True)
    return saved


def make(scene: str, style: str = "", seed: int | None = None, name: str | None = None, on_event=None) -> dict:
    """A new library panorama from a place description (Qwen-Image-2512 + the 360 LoRA, graph workflows/
    qwenimage_pano; the same recipe as a run's text background), its seam closed. A GPU job: the caller holds a queue
    turn (cli `background new`). Returns the saved record."""
    import secrets as _secrets

    from . import character
    prompt = prompt_for(scene, style)
    seed = int(seed) if seed is not None else _secrets.randbelow(2 ** 31)
    nn = name or auto_name(prompt, seed)
    if not NAME_RE.match(nn):
        raise BackgroundError(f"background name {nn!r}: letters, digits, '-' and '_' (at most 64)")
    if taken(nn):
        raise BackgroundError(f"a panorama {nn!r} exists already: choose another name")
    work = library_dir() / ".work"
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"{nn}_{_secrets.token_hex(3)}.png"
    try:
        character.image_job("qwenimage", "pano", {"prompt": prompt, "negative": NEGATIVE, "seed": seed, "width": WIDTH,
                                                  "height": HEIGHT}, {}, out, f"bg_{nn}", on_event=on_event)
        fix_seam(out)
        save(out, prompt, seed, scene=scene.strip(), style=style.strip(), name=nn)
    finally:
        out.unlink(missing_ok=True)
    meta_f = library_dir() / nn / "bg.json"
    meta = json.loads(meta_f.read_text(encoding="utf-8"))
    meta["source"] = "text"
    meta_f.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return load(nn)


def depth(name: str, on_event=None, pano: str | Path | None = None) -> Path:
    """The 360 depth of a library panorama (or of the file `pano`) with MoGe-2 (cinestudio/pano_depth.py): a GPU job
    (graph workflows/moge_pano), kept next to the panorama as <stem>_depth.npy (a BUILT-IN panorama's in the user's
    cache: pano_depth.cached) and reused while the pixels are the same. Returns the .npy file."""
    import hashlib

    from . import character, pano_depth
    src = Path(pano) if pano else Path(require_pano(name)["file"])   # a backplate has no 360 depth
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    have = pano_depth.load_if_current(src, sha)
    if have:
        return have
    glb = pano_depth.cached(src).with_name(src.stem + "_depth.glb")     # beside the depth (never in the repo)
    character.image_job("moge", "pano", {}, {("load", "image"): src}, glb, f"bg_{name}_depth", on_event=on_event)
    return pano_depth.save(pano_depth.distance_map(glb), src, sha)


def fix_seam(png: str | Path, width: int = 12) -> None:
    """Make the left and right edges meet (the panorama wraps round): the outer `width` columns on each side are
    cross-faded toward their common average, so no vertical line shows behind the character at azimuth 180."""
    import cv2
    import numpy as np
    img = cv2.imread(str(png), cv2.IMREAD_COLOR)
    if img is None or img.shape[1] < 4 * width:
        return
    f = img.astype(np.float32)
    mean = (f[:, :1] + f[:, -1:]) / 2
    for i in range(width):
        w = 1.0 - i / width                                       # 1 at the edge, 0 inside
        f[:, i:i + 1] = f[:, i:i + 1] * (1 - w) + mean * w
        f[:, -1 - i:f.shape[1] - i] = f[:, -1 - i:f.shape[1] - i] * (1 - w) + mean * w
    cv2.imwrite(str(png), np.clip(f, 0, 255).astype(np.uint8))
