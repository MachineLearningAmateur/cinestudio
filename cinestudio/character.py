"""Character library: one folder per character, reusable in every shot.

<home>/characters/<name>/
  character.json     name, description, style, source, chosen seed, files, checks, status
  candidates/        front views made from text (one per seed) + candidates.jpg (all of them)
  front.png          the chosen (or imported) front view
  views/             right.png, back.png, left.png, face.png
  sheet.png          face panel + front / right side / back / left side on black (the Ingredients layout)
  sheet_styled.png   the sheet after the LTX style pass with the scene's style LoRA (optional)

Ways in:
  text   -> create (Z-Image-Turbo front views) -> pick -> views -> [style] -> ready
  image  -> import --image (one front view)            -> views -> [style] -> ready
  sheet  -> import --sheet (a full sheet: used as it is)                   -> ready
The views come from Qwen-Image-Edit-2511 with the Multiple-Angles LoRA, each made from the front view.
The style pass redraws the sheet with LTX-2.5 (ltx25_redraw) and the scene's style LoRA, so the character and the
scene come from the same model and LoRA.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import math
import os
import threading
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from . import charsheet, server, video
from .locks import FileLock, LockBusy
from .workflow import ProfileWorkflow

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
FRONT_PROMPT = ("A full-body character design illustration of ONE original character: {description} The character "
                "stands straight and faces the viewer in a neutral A-pose, arms about 35 degrees away from the body, relaxed "
                "open hands with exactly five clearly separated fingers each, legs slightly apart. The whole body is visible from head to feet with a margin, centered, on a "
                "plain flat black background. No text, no scenery, no floor shadow, no other characters. Style: {style}.")
VIEWS = {   # fal Multiple-Angles LoRA format: <sks> [azimuth] [elevation] [distance]
    "right": "<sks> right side view eye-level shot wide shot",
    "back": "<sks> back view eye-level shot wide shot",
    "left": "<sks> left side view eye-level shot wide shot",
    "face": "<sks> front view eye-level shot close-up",
}
# the LoRA card's distances: "close-up" x0.6, "medium shot" x1.0 ("balanced, standard": the input's own framing), "wide
# shot" x1.8 ("context, environment"; the LoRA learned from full-scene renders: a likely cause of a ROOM behind the
# character). Default since 2026-10-22 (image prompt research, docs/image_prompt_research.md): medium shot.
VIEW_DISTANCES = ("medium shot", "wide shot")
VIEW_DISTANCE = "medium shot"
FACE_SOURCES = ("front", "qwen")   # the face panel: cut from the front (default; cloaked3d: the Qwen close-up drew a
                                   # wrong small full body) or the Qwen close-up


def view_prompt(view: str, distance: str | None = None) -> str:
    """The Multiple-Angles prompt of a body view at the chosen distance (the face keeps its close-up)."""
    distance = distance or VIEW_DISTANCE
    if distance not in VIEW_DISTANCES:
        raise CharacterError(f"view distance must be one of {VIEW_DISTANCES}")
    ins = VIEWS[view]
    return ins if view == "face" else ins.replace("wide shot", distance)


def face_from_front(front: np.ndarray) -> np.ndarray:
    """The face panel cut from the front view: the head and shoulders (the top 30% of the figure, centred on the head),
    at 2:3, enlarged to the front's size, lightly sharpened. Exact and consistent with the front, no model."""
    fig = front.max(axis=2) > 12
    ys, xs = np.nonzero(fig)
    if not len(ys):
        raise CharacterError("the front view has no figure on its background")
    top, bottom = ys.min(), ys.max()
    hgt = bottom - top
    head = fig[top:top + max(4, int(hgt * 0.12))]
    hx = np.nonzero(head)[1]
    cx = int(np.median(hx)) if len(hx) else int(np.median(xs))
    ch = int(hgt * 0.30)
    cw = int(ch * 2 / 3)
    y0 = max(0, top - int(ch * 0.06))
    x0 = int(np.clip(cx - cw // 2, 0, max(0, front.shape[1] - cw)))
    crop = front[y0:y0 + ch, x0:x0 + cw]
    big = cv2.resize(crop, (front.shape[1], int(front.shape[1] * 1.5)), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(big, (0, 0), 2.0)
    return cv2.addWeighted(big, 1.5, blur, -0.5, 0)
SHEET_ORDER = ["front", "right", "back", "left"]      # a classic turnaround: four angles 90 degrees apart
# the sheet since 2026-10-22 (the Ingredients 2.5 card: "The more space an element takes up in the reference image, the
# more faithfully it carries over"; Lightricks' example: face, front, 3/4, back): 4 large panels, the figures filling
# the height on one feet line; BOTH sides (the card's example: "front, sides, back, 3/4 view"; user: "one of the side
# views is no longer visible"); the face panel at 60% of the height gives the bodies their full height
SHEET_PANELS = ["front", "right", "back", "left"]
SHEET_LAYOUT = "panels5"
SLOTS = ["front", "right", "back", "left", "face"]     # images a user can give; empty ones are made from the front
# quality (the default): 40 steps, cfg 3, as in the ComfyUI template. (cfg 4 with a second reference image gave
# pixel errors on 2026-10-19; 40 steps was the user's choice over 50.)
# fast: the Lightning LoRA, only for quick checks.
QUALITY = {"quality": {"steps": 40, "cfg": 3.0},
           "fast": {"steps": 4, "cfg": 1.0, "fast": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"}}
# values of the image graphs that change quality; the graph defaults are used for the ones not given
GRAPH_DEFAULTS = {"sampler": "euler", "scheduler": "simple", "shift": 3.1, "denoise": 1.0, "negative": "",
                  "angles_strength": 0.9}
SETTING_RULES = {"steps": (int, 1, 100), "cfg": (float, 0.0, 15.0), "shift": (float, 0.0, 10.0),
                 "denoise": (float, 0.05, 1.0), "angles_strength": (float, 0.0, 2.0), "sampler": (str,), "scheduler": (str,),
                 "negative": (str,)}
EDIT_SETTINGS = ("steps", "cfg", "sampler", "scheduler", "shift", "denoise", "negative")
VIEW_SETTINGS = ("steps", "cfg", "sampler", "scheduler", "shift", "negative", "angles_strength")


def quality_values(mode: str, settings: dict | None, allowed: tuple[str, ...]) -> tuple[dict, dict]:
    """(values for the graph, the complete settings used, for the history) of a mode plus the user's
    changes. Each value is checked: type and range."""
    if mode not in QUALITY:
        raise CharacterError(f"mode must be one of {sorted(QUALITY)}")
    from . import recipe
    if mode == "fast" and not recipe.FAST_EDITS:
        raise CharacterError("the fast mode uses a distilled shortcut (Lightning LoRA), which is turned off: "
                             "the full model only (use mode 'quality')")
    values = dict(QUALITY[mode])
    for k, v in (settings or {}).items():
        if v is None or v == "":
            continue
        if k not in allowed:
            raise CharacterError(f"{k!r} is not a setting of this step; use {list(allowed)}")
        rule = SETTING_RULES[k]
        try:
            v = rule[0](v)
        except (TypeError, ValueError) as exc:
            raise CharacterError(f"{k} must be a {rule[0].__name__}") from exc
        if len(rule) == 3 and not rule[1] <= v <= rule[2]:
            raise CharacterError(f"{k} must be between {rule[1]} and {rule[2]}")
        values[k] = v
    used = {"mode": mode, **{k: GRAPH_DEFAULTS[k] for k in allowed if k in GRAPH_DEFAULTS}, **values}
    used.pop("fast", None)
    return values, used


DEFAULT_STYLE = ("cinematic fantasy anime cel-shaded illustration with hand-drawn linework, painted shading and "
                 "stylized fantasy character design")


_HELD: dict = {}                 # (library folder, name, thread id) -> [FileLock, depth]
_HELD_GUARD = threading.Lock()


@contextlib.contextmanager
def char_lock(name: str, timeout: float = 0):
    """The lock of one character: every change (edit, redo, undo, views, style, pick, finish, rename, delete,
    touch-up) holds it, so two changes never mix (a pick during an edit overwrote the edit's backup).
    The lock file is outside the character folder, so a delete can hold it. Re-entrant in one thread."""
    with item_lock(library(), name, timeout):
        yield


@contextlib.contextmanager
def item_lock(folder: Path, name: str, timeout: float = 0):
    """The lock of one library item (a character, a prop: cinestudio/props.py) in <folder>/.locks/<name>.lock.
    Re-entrant in one thread; a busy lock raises CharacterError (props: PropError, a subclass)."""
    if not isinstance(name, str) or not NAME_RE.match(name):    # before any path is made from it (Codex 10-25)
        raise CharacterError(f"bad name {name!r}: use lowercase letters, digits, '-' or '_' (max 48)")
    key = (str(folder), name, threading.get_ident())
    with _HELD_GUARD:
        held = _HELD.get(key)
        if held:
            held[1] += 1
    if not held:
        lk = FileLock(Path(folder) / ".locks" / f"{name}.lock", timeout=timeout)
        try:
            lk.acquire()
        except LockBusy as exc:
            raise CharacterError(f"{name} is busy: another change runs now (a job or an edit); "
                                 "try again when it ends") from exc
        with _HELD_GUARD:
            _HELD[key] = held = [lk, 1]
    try:
        yield
    finally:
        with _HELD_GUARD:
            held[1] -= 1
            last = held[1] == 0
            if last:
                del _HELD[key]
        if last:
            held[0].release()


def _locked(fn):
    """Run a character function under its character's lock (the first argument is the name). Every locked function
    CHANGES the character, so a built-in character (read-only) is refused here, before any file is written."""
    @functools.wraps(fn)
    def wrapper(name, *args, **kwargs):
        refuse_builtin(name)
        with char_lock(name):
            return fn(name, *args, **kwargs)
    return wrapper


class StepCanceled(RuntimeError):
    """The user canceled the step (the only reason a step may cancel its ComfyUI job)."""


class CharacterError(RuntimeError):
    pass


def library() -> Path:
    from .runner import home
    return home() / "characters"


# BUILT-IN characters (2026-10-26, user: "let's move it in as a built in character ... ship it in with proxy kits
# too"): characters/library/<id>/ in the repo, read-only like the built-in templates, backgrounds and styles. Their
# character.json keeps paths RELATIVE to its folder (load() resolves them); a user character of the same ID wins.
BUILTIN = Path(__file__).resolve().parent.parent / "characters" / "library"


def is_builtin(name: str) -> bool:
    """True when `name` is a built-in character that no user character of the same ID hides."""
    if not isinstance(name, str) or not NAME_RE.match(name):
        return False
    return not (library() / name / "character.json").is_file() and (BUILTIN / name / "character.json").is_file()


def refuse_builtin(name: str, what: str = "changed") -> None:
    if is_builtin(name):
        raise CharacterError(f"{name!r} is a built-in character (it ships with cinestudio: characters/library/{name}) "
                             f"and cannot be {what}; make a skin of it or a new character")


def char_dir(name: str, must_exist: bool = True) -> Path:
    if not NAME_RE.match(name or ""):
        raise CharacterError(f"bad character name {name!r}: use lowercase letters, digits, '-' or '_' (max 48)")
    d = library() / name
    if must_exist and not (d / "character.json").is_file():
        if (BUILTIN / name / "character.json").is_file():
            return BUILTIN / name
        raise CharacterError(f"no character {name!r} in {library()}")
    return d


def _resolve_paths(v: Any, base: Path) -> Any:
    """A built-in record's relative file paths -> absolute paths in its folder (dicts and lists, recursively)."""
    if isinstance(v, dict):
        return {k: _resolve_paths(x, base) for k, x in v.items()}
    if isinstance(v, list):
        return [_resolve_paths(x, base) for x in v]
    if isinstance(v, str) and v.startswith("./"):
        return str(base / v[2:])
    return v


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + f".{secrets.token_hex(3)}.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


def load(name: str) -> dict:
    d = char_dir(name)
    c = json.loads((d / "character.json").read_text(encoding="utf-8"))
    if d.parent.resolve() == BUILTIN.resolve():
        c = _resolve_paths(c, d)
        c["builtin"] = True
    return c


def save(c: dict) -> None:
    if c.get("builtin") or is_builtin(c.get("name", "")):
        raise CharacterError(f"{c.get('name')!r} is a built-in character (characters/library) and cannot be changed")
    c["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_json(char_dir(c["name"], must_exist=False) / "character.json", c)


def list_characters() -> list[dict]:
    dirs = sorted(library().iterdir()) if library().is_dir() else []
    mine = {d.name for d in dirs if (d / "character.json").is_file()}
    dirs += [d for d in (sorted(BUILTIN.iterdir()) if BUILTIN.is_dir() else []) if d.name not in mine]
    out = []
    for d in dirs:
        try:
            c = load(d.name) if d.parent.resolve() == BUILTIN.resolve() else                 json.loads((d / "character.json").read_text(encoding="utf-8"))
            row = {k: c.get(k) for k in ("name", "status", "source", "description", "style", "updated", "style_ref",
                                         "style_version", "skin_of")}
            row["display_name"] = display_name(c)         # what the app shows; "name" stays the ID
            row["proxy"] = bool(c.get("proxy"))           # the New video form: style text and the proxy kit switch
            row["missing"] = video_missing(c)             # what a Blender run still needs (mannequin, proxy, sheet)
            row["mannequin"] = c.get("mannequin")
            row["builtin"] = bool(c.get("builtin"))       # ships with cinestudio (read-only)
            f = c.get("files", {})
            cands = c.get("candidates") or [{}]
            row["thumb"] = f.get("front") or f.get("sheet") or cands[0].get("file")   # a picture for the library
            out.append(row)
        except (OSError, ValueError):
            continue
    return out


DISPLAY_MAX = 80


def slugify(text: str, fallback: str = "character") -> str:
    """An ID from a display name (user, 2026-10-24: "automatically replace spaces as needed for the id"): lowercase,
    every run of other characters -> "-", at most 48 characters, starting with a letter or digit ("Masked Wanderer 3D"
    -> "masked-wanderer-3d"). The ID names the folder, the files, the jobs and the runs; the display name only shows."""
    import unicodedata
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9_]+", "-", t).strip("-_")[:48].strip("-_")
    return t or fallback


def free_id(display: str, folder: Path, fallback: str = "character") -> str:
    """A free ID in a library folder: slugify(display), then -2, -3 ... when the ID is taken (characters, props)."""
    base = slugify(display, fallback)
    cand, n = base, 2
    while (Path(folder) / cand).exists() or (Path(folder) == library() and (BUILTIN / cand).exists()):
        suffix = f"-{n}"
        cand, n = base[:48 - len(suffix)] + suffix, n + 1
    return cand


def new_id(display: str) -> str:
    """A free ID for a new character: slugify(display), then -2, -3 ... when the ID is taken."""
    return free_id(display, library())


def clean_display(text: str) -> str:
    t = " ".join((text or "").split())                   # no control characters, single spaces
    if not t:
        raise CharacterError("the name cannot be empty")
    if len(t) > DISPLAY_MAX:
        raise CharacterError(f"the name is longer than {DISPLAY_MAX} characters")
    return t


def display_name(c: dict) -> str:
    """The name the app shows (any text); old characters show their ID."""
    return c.get("display_name") or c["name"]


def set_style(name: str, style: str) -> dict:
    """Give the character a style from the library (instant): its LOOK words go into the character's own style field
    (front views and edits use it; a sheet stays flatly lit, so the style's light words are left out) and the record
    keeps the style's name and version (styles.stamp). Runs take the full style (look + light) from the name. Finished
    pictures do not change; a sheet made in another look must be converted (AGENTS.md 4.9)."""
    from . import styles
    try:
        words, stamp = styles.text(style, light=False), styles.stamp(style)   # a sheet stays flatly lit
    except styles.StyleError as exc:
        raise CharacterError(str(exc)) from exc
    with char_lock(name):
        c = load(name)
        c["style"] = words
        c.update(stamp)
        save(c)
    return c


# ---------------------------------------------------------------- skins: one character in more than one style
# (user, 2026-10-25: "think of it as like switching skins for a character ... we should combine kitsune 3d into
# kitsune since they are the same character but just different styles"). A skin is a full character record of its
# own (its ID, pictures, sheet, history, body and kit, so runs, jobs and locks do not change) with "skin_of": the ID of
# the MAIN record. The main record has no "skin_of". The app shows one character per family and a row of skin cards that
# switch between the skins or make a new one.
SKIN_COPY = ("description", "body", "mannequin", "proxy", "height", "sheet_scale", "sheet_layout", "view_distance",
             "source", "seed", "front_size", "front_model", "exclude", "display_name")


def base_of(c: dict) -> str:
    """The ID of the character's main record (itself when it is not a skin)."""
    return c.get("skin_of") or c["name"]


def skins(name: str) -> list[dict]:
    """The family of a character (the list rows): the main record first, then its skins by ID."""
    return family_rows(list_characters(), base_of(load(name)))


def family_rows(rows: list[dict], base: str) -> list[dict]:
    """The list rows of one skin family (characters or props): the main record first, then its skins by ID."""
    fam = [r for r in rows if (r.get("skin_of") or r["name"]) == base]
    return sorted(fam, key=lambda r: (bool(r.get("skin_of")), r["name"]))


def _skins_of(base: str) -> list[str]:
    return [r["name"] for r in list_characters() if r.get("skin_of") == base]


def link_skin(name: str, base: str) -> dict:
    """Make `name` a skin of `base` (instant; nothing else changes)."""
    if name == base:
        raise CharacterError("a character cannot be a skin of itself")
    b = load(base)
    if b.get("skin_of"):
        raise CharacterError(f"{base} is itself a skin of {b['skin_of']}: link to {b['skin_of']}")
    own = _skins_of(name)
    if own:
        raise CharacterError(f"{name} has skins of its own ({', '.join(own)}): it cannot become a skin")
    with char_lock(name):
        c = load(name)
        c["skin_of"] = base
        save(c)
    return c


def unlink_skin(name: str) -> dict:
    """Make a skin a character of its own again (instant)."""
    with char_lock(name):
        c = load(name)
        c.pop("skin_of", None)
        save(c)
    return c


def skin_instruction(c: dict, style: dict, with_reference: bool) -> str:
    """The Qwen instruction of a style conversion (AGENTS.md 4.9; the background as its own sentence, 4.4)."""
    look = (style.get("words") or "").strip().rstrip(".")
    head = (f"Redraw the character of Picture 1 in the rendering style of Picture 2: {look}." if with_reference
            else f"Redraw the character of Picture 1 in this rendering style: {look}.")
    design = (c.get("description") or "").strip().rstrip(".")
    return (f"{head} The background is pure black everywhere, empty, without floor, shadow or horizon, exactly as in "
            f"Picture 1. The character's design stays exactly the same: {design}, and the pose. Keep the colours, "
            f"the proportions and every part of the costume exactly the same.")


SKIN_NEGATIVE = ("white background, grey background, gradient background, studio backdrop, seamless paper backdrop, "
                 "room, wall, floor, ground shadow, cast shadow, floor reflection, vignette, spotlight, rim light, "
                 "white outline around the figure, extra arms, extra hands, extra fingers, missing fingers, fused "
                 "fingers, duplicate character, second figure, cropped feet, cropped head, text, watermark, logo, "
                 "blurry, low quality, plastic toy, vinyl figure, product photo")


def skin_negative(src: dict, style: dict, base: str = SKIN_NEGATIVE) -> str:
    """The conversion's negative: the base list (4.4), the style's own excluded looks and the SOURCE look's phrases
    that share no word with the target look (2D -> 3D: "cel-shaded illustration", "hand-drawn linework"). Props pass
    their own base list (props.PROP_NEGATIVE)."""
    target = set(re.findall(r"[a-z]{4,}", (style.get("words") or "").lower()))
    old = [p.strip() for p in (src.get("style") or "").split(",") if p.strip()]
    drop = [p for p in old if not set(re.findall(r"[a-z]{4,}", p.lower())) & target]
    parts = [base, (style.get("negative_image") or "").strip(), ", ".join(drop)]
    return ", ".join(p for p in parts if p)


def new_skin(name: str, style: str, seed: int | None = None, mode: str = "quality", **kw) -> dict:
    """A new skin of the character in a library style (a GPU job): a new record (ID "<main>-<style>") with the
    character's description, body and kit; its front = this skin's front redrawn by Qwen-Image-Edit in the style (the
    style's reference picture, cut to its crop box, as Picture 2 when it has one; the pose stays); then the views and the sheet from that
    front. Check the result, then finish it (Use as is) like any character."""
    from . import styles
    src = load(name)
    try:
        st = styles.load(style)
    except styles.StyleError as exc:
        raise CharacterError(str(exc)) from exc
    base = base_of(src)
    fam = skins(base)
    if any(r.get("style_ref") == style for r in fam):
        raise CharacterError(f"{display_name(load(base))} has a skin in {st.get('label') or style} already")
    front = Path((src.get("files") or {}).get("front") or "")
    if not front.is_file():
        raise CharacterError(f"{name} has no front view to redraw")
    new = new_id(f"{base} {style}")
    d = char_dir(new, must_exist=False)
    d.mkdir(parents=True)
    shutil.copyfile(front, d / "front.png")
    c = {k: src[k] for k in SKIN_COPY if k in src}
    c.update(name=new, skin_of=base, copied_from=name, status="front", files={"front": str(d / "front.png")},
             checks={}, created=time.strftime("%Y-%m-%dT%H:%M:%S"), style=styles.text(style, light=False),
             **styles.stamp(style))
    save(c)
    from . import style_ref                        # Picture 2: the reference's tight crop (AGENTS.md 4.3), or None
    ref = style_ref.reference_for_qwen(style)
    edit_image(new, skin_instruction(src, st, bool(ref)), target="front", reference=ref, mode=mode, seed=seed,
               settings={"negative": skin_negative(src, st)}, **kw)
    # the redrawn front keeps the source's pose: an AnyPose front stays posed (no second repose)
    c = make_views(new, mode=mode, seed=seed, pose=False if front_is_posed(src) else None, **kw)
    return {**c, "skin": new}


def set_display_name(name: str, text: str) -> dict:
    """Rename in the app: only the display name changes (instant; no folder, file, job or run changes)."""
    t = clean_display(text)
    with char_lock(name):
        c = load(name)
        c["display_name"] = t
        save(c)
    return c


def _new(name: str, description: str, style: str, source: str, body: str = "human", display: str = "") -> dict:
    if body not in BODY_TYPES:
        raise CharacterError(f"body must be one of {BODY_TYPES}")
    d = char_dir(name, must_exist=False)
    if (d / "character.json").exists():
        raise CharacterError(f"character {name!r} exists already")
    if source == "text" and not description.strip():
        raise CharacterError("a description of the character is needed (it is also used in every video prompt)")
    d.mkdir(parents=True)
    # Imported images or a sheet carry their own look: no default style text (2026-10-22: the anime default was put on
    # imported 3D art); the description is optional there, but the video prompt and the look images use it when given.
    style = (style or (DEFAULT_STYLE if source == "text" else "")).strip()
    c = {"name": name, "display_name": clean_display(display) if (display or "").strip() else name,
         "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source, "status": "new",
         "description": (description or "").strip(), "style": style, "body": body, "files": {}, "checks": {}}
    save(c)
    return c


# ---------------------------------------------------------------- one ComfyUI image job
def image_job(profile: str, variant: str, values: dict, files: dict, out: Path, tag: str,
              backend_factory: Callable[[], Any] | None = None, on_event: Callable[[dict], None] | None = None,
              item: tuple[int, int] = (1, 1), graph_edit: Callable[[dict], None] | None = None,
              extra_outputs: dict[str, Path] | None = None) -> dict:
    """Run one image graph with the same rules as the runner: GPU lock, model-family check, owned prompt
    id, job-scoped watch. `out` receives the image of the graph's output node; `extra_outputs` ({node id: file})
    receives the images of other save nodes of the same job (style_ref: several frames of one clip)."""
    from .backend import SdkBackend
    from . import models as _models
    wf = ProfileWorkflow(profile, variant, models=_models.job_bundle())
    token = secrets.token_hex(3)
    prefix = f"cinestudio/char_{tag}_{token}"
    graph = wf.build(dict(values, output_prefix=prefix), files=files)
    if graph_edit is not None:                       # a caller's change of the built graph (background.make_set_plate)
        graph_edit(graph)
    from .workflow import graph_cache, save_graph    # the exact graph sent (the app's Workflows view)
    save_graph(graph_cache(), tag, graph, files=files, keep=200,
               meta={"tag": tag, "profile": profile, "variant": variant, "prefix": prefix,
                     "edited": graph_edit is not None})
    from . import comfy
    if backend_factory is None and not server.alive():       # ComfyUI not open: start it in the background
        if on_event:
            on_event({"event": "starting_comfyui"})
        comfy.ensure_running()
    lock = server.gpu_lock(note={"character": tag})
    try:
        lock.acquire()
    except LockBusy:
        if on_event:
            on_event({"event": "waiting_for_gpu", "holder": lock.holder()})
        lock.timeout = 6 * 3600
        lock.acquire()
    try:
        switch = server.ensure_family(wf.family_key())
        if switch.get("restarted") and on_event:
            on_event({"event": "comfyui_restarted", **switch})
        backend = (backend_factory or SdkBackend)()
        t0 = time.time()
        jid = backend.submit(graph, files, key=f"char-{tag}-{token}", prefix=prefix)
        server.note_owned(jid)
        try:
            return _watch_image(backend, jid, wf, out, t0, tag, item, on_event, extra_outputs)
        except StepCanceled:
            try:                                          # an explicit cancel: stop this ComfyUI job too
                if not backend.state(jid).done:
                    backend.cancel(jid)
            except Exception:  # noqa: BLE001 - jobs.cancel also cancels it through ComfyUI directly
                pass
            raise
        # any other error (a lost connection, a failed download): the ComfyUI job is NOT canceled. It goes on;
        # its id is in the job's events ("comfy_job"), so its result can still be recovered.
    finally:
        lock.release()


def _watch_image(backend: Any, jid: str, wf: ProfileWorkflow, out: Path, t0: float, tag: str,
                 item: tuple[int, int], on_event: Callable[[dict], None] | None,
                 extra_outputs: dict[str, Path] | None = None) -> dict:
    if on_event:
        on_event({"event": "comfy_job", "job": jid, "tag": tag})
    last = [0.0]

    def progress(kind: str, data: dict) -> None:     # sampling step i of n, image `item` of the job
        if kind == "progress" and data.get("steps") and time.time() - last[0] > 1.0:
            last[0] = time.time()
            from . import lab                        # a lab job (experiments, backgrounds): the queue's bar
            lab.note_progress(data.get("step"), data.get("steps"))
            if on_event:
                on_event({"event": "progress", "value": data.get("step"), "max": data.get("steps"),
                          "item": item[0], "items": item[1], "tag": tag})
    st = backend.watch(jid, progress)
    if not st.ok:
        raise CharacterError(f"ComfyUI job {jid} {st.status}: {st.error}")
    out.parent.mkdir(parents=True, exist_ok=True)
    backend.download(jid, wf.bind["outputs"]["image"], out)   # (a GLB falls back to ComfyUI's history: backend.py)
    res = {"job": jid, "seconds": round(time.time() - t0, 1), "file": str(out)}
    if extra_outputs:
        for node, dst in extra_outputs.items():
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            backend.download(jid, node, Path(dst))
        res["extra"] = {node: str(dst) for node, dst in extra_outputs.items()}
    return res


# ---------------------------------------------------------------- text -> front-view candidates
@_locked
def create(name: str, description: str, style: str = "", candidates: int = 8, seed: int = 1, width: int = 1024,
           height: int = 1536, style_lora: str | None = None, style_lora_strength: float = 0.8, body: str = "human",
           display_name: str = "", **kw) -> dict:
    """Front-view candidates from text with Z-Image (the full model; one per seed). Choose one with `pick`."""
    c = _new(name, description, style, "text", body, display_name)
    d = char_dir(name)
    prompt = FRONT_PROMPT.format(description=c["description"], style=c["style"])
    values = {"prompt": prompt, "width": width, "height": height}
    if style_lora:
        values |= {"style_lora": style_lora, "style_lora_strength": style_lora_strength}
    made = []
    for i, s in enumerate(range(seed, seed + candidates), 1):
        from . import recipe
        r = image_job("zimage", recipe.ZIMAGE_VARIANT, dict(values, seed=s), {}, d / "candidates" / f"front_s{s}.png", f"{name}_s{s}",
                      item=(i, candidates), **kw)
        made.append({"seed": s, **r})
        if kw.get("on_event"):
            kw["on_event"]({"event": "candidate", "seed": s, "file": r["file"], "seconds": r["seconds"]})
    _contact(d / "candidates", [m["file"] for m in made], [f"seed {m['seed']}" for m in made])
    c.update(status="candidates", front_prompt=prompt, candidates=[{"seed": m["seed"], "file": m["file"]} for m in made],
             front_model="Z-Image" if recipe.ZIMAGE_VARIANT == "base" else "Z-Image-Turbo", front_size=[width, height])
    save(c)
    return c


def _contact(folder: Path, files: list[str], labels: list[str], height: int = 512) -> Path:
    tiles = []
    for f, label in zip(files, labels):
        im = cv2.imread(f)
        im = cv2.resize(im, (int(im.shape[1] * height / im.shape[0]), height), interpolation=cv2.INTER_AREA)
        cv2.putText(im, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(im)
    out = folder / "candidates.jpg"
    cv2.imwrite(str(out), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


@_locked
def pick(name: str, seed: int) -> dict:
    c = load(name)
    hit = [x for x in c.get("candidates", []) if x["seed"] == seed]
    if not hit:
        raise CharacterError(f"{name}: no candidate with seed {seed} (candidates: {[x['seed'] for x in c.get('candidates', [])]})")
    d = char_dir(name)
    front = d / "front.png"
    if front.is_file():
        # a front view exists (maybe with edits): keep it in the edit history, so `undo_edit` brings it back
        version = len(c.get("edits", [])) + 1
        shutil.copyfile(front, d / f"front_v{version}.png")
        c.setdefault("edits", []).append({"version": version, "instruction": f"use candidate seed {seed}",
                                          "reference": None, "before": str(d / f"front_v{version}.png"),
                                          "seconds": 0, "mode": "pick", "seed_before": c.get("seed")})
        _drop_generated(c)                                    # views and sheets of the old front
    shutil.copyfile(hit[0]["file"], front)
    c.update(status="front", seed=seed)
    c["files"]["front"] = str(front)
    save(c)
    return c


# ---------------------------------------------------------------- imports
def _drop_generated(c: dict) -> None:
    """Forget the views and sheets that were made from an older front view; keep the ones the user gave."""
    given = set(c.get("provided", []))
    for k in (*VIEWS, "sheet", "sheet_styled"):
        if k not in given:
            c["files"].pop(k, None)


@_locked
def import_slots(name: str, images: dict[str, str | Path], description: str, style: str = "",
                 body: str = "human", display_name: str = "") -> dict:
    """A character from the user's own images: front (required), right, back, left, face (optional).
    Empty slots are made from the front view by `make_views`; with right, back and left all given,
    the sheet is made at once (no GPU job)."""
    unknown = set(images) - set(SLOTS)
    if unknown:
        raise CharacterError(f"unknown slots {sorted(unknown)}; use {SLOTS}")
    if not images.get("front"):
        raise CharacterError("the front slot is needed")
    for slot, path in images.items():
        if path and cv2.imread(str(path)) is None:
            raise CharacterError(f"cannot read the {slot} image {path}")
    c = _new(name, description, style, "images", body, display_name)
    d = char_dir(name)
    (d / "views").mkdir(exist_ok=True)
    given = []
    for slot, path in images.items():
        if not path:
            continue
        dst = d / "front.png" if slot == "front" else d / "views" / f"{slot}.png"
        cv2.imwrite(str(dst), cv2.imread(str(path)))           # always PNG, whatever the input format
        c["files"][slot] = str(dst)
        given.append(slot)
    c.update(provided=given, status="front")
    save(c)
    if all(s in given for s in ("right", "back", "left")):
        return build_sheet(name)
    return load(name)


@_locked
def import_image(name: str, image: str | Path, description: str, style: str = "", body: str = "human",
                 display_name: str = "") -> dict:
    """One image of the character (best: the full body from the front). The views come from it."""
    image = Path(image)
    if cv2.imread(str(image)) is None:
        raise CharacterError(f"cannot read the image {image}")
    c = _new(name, description, style, "image", body, display_name)
    shutil.copyfile(image, char_dir(name) / "front.png")
    c.update(status="front")
    c["files"]["front"] = str(char_dir(name) / "front.png")
    save(c)
    return c


@_locked
def import_sheet(name: str, sheet: str | Path, description: str, style: str = "", display_name: str = "") -> dict:
    """A finished reference sheet (face + several angles): used as it is, no views are made."""
    sheet = Path(sheet)
    if cv2.imread(str(sheet)) is None:
        raise CharacterError(f"cannot read the sheet {sheet}")
    c = _new(name, description, style, "sheet", display=display_name)
    shutil.copyfile(sheet, char_dir(name) / "sheet.png")
    c.update(status="ready")
    c["files"]["sheet"] = str(char_dir(name) / "sheet.png")
    save(c)
    return c


# ---------------------------------------------------------------- front -> views -> sheet
@_locked
def make_views(name: str, mode: str = "quality", seed: int | None = None, distance: str | None = None,
               face: str = "front", pose: bool | None = None, **kw) -> dict:
    """The right side, back and left side views, each made from the front view with Qwen-Image-Edit-2511 + the
    Multiple-Angles LoRA (`distance`: "medium shot" default, or "wide shot"), and the face panel (`face`: "front" =
    cut from the front, default; "qwen" = the LoRA's close-up); then the colour check and the sheet."""
    if face not in FACE_SOURCES:
        raise CharacterError(f"face must be one of {FACE_SOURCES}")
    distance = distance or VIEW_DISTANCE
    view_prompt("right", distance)                            # refuses an unknown distance before any GPU work
    c = load(name)
    d = char_dir(name)
    front = d / "front.png"
    if not front.is_file():
        raise CharacterError(f"{name}: no front view yet (create + pick, or import --image)")
    if mode not in QUALITY:
        raise CharacterError(f"mode must be one of {sorted(QUALITY)}")
    seed = seed if seed is not None else c.get("seed", 1)
    if pose is None:                                         # body type "human": the front takes the rig's rest pose
        #                                                       first (order A, 2026-10-22); not when the user gave
        #                                                       their own side or back images (they keep their pose)
        own = set(c.get("provided") or []) & {"right", "back", "left"}
        pose = c.get("body") == "human" and not own and not front_is_posed(c)
    if pose:
        c = pose_front(name, **kw)
    wf = ProfileWorkflow("qwenedit", "angle")
    timings = {}
    given = set(c.get("provided", []))
    todo = [(v, view_prompt(v, distance)) for v in VIEWS if v not in given or not Path(c["files"].get(v, "")).is_file()]
    if face == "front" and "face" in [v for v, _ in todo]:   # the face panel from the front: no GPU job
        todo = [(v, ins) for v, ins in todo if v != "face"]
        prev = d / "views" / "face.png"
        if prev.is_file():
            (d / "views" / "history").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prev, d / "views" / "history" / f"face_{time.strftime('%Y%m%d-%H%M%S')}.png")
        (d / "views").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "views" / "face.png"), face_from_front(cv2.imread(str(front))))
        c["files"]["face"] = str(d / "views" / "face.png")
    c["view_distance"] = distance
    for i, (view, instruction) in enumerate(todo, 1):
        values = {"instruction": instruction, "seed": seed, **quality_values(mode, {}, VIEW_SETTINGS)[0]}
        prev = d / "views" / f"{view}.png"
        if prev.is_file():                                    # keep the view that this run replaces
            (d / "views" / "history").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prev, d / "views" / "history" / f"{view}_{time.strftime('%Y%m%d-%H%M%S')}.png")
        r = image_job("qwenedit", "angle", values, {wf.targets("front")[0]: front}, d / "views" / f"{view}.png",
                      f"{name}_{view}", item=(i, len(todo)), **kw)
        timings[view] = r["seconds"]
        c["files"][view] = r["file"]
        if kw.get("on_event"):
            kw["on_event"]({"event": "view", "view": view, "file": r["file"], "seconds": r["seconds"]})
    return build_sheet(name, c, timings, mode)


BODY_TYPES = ("human", "other")
POSE_INSTRUCTION = ("Make the character in Picture 1 take the exact same pose as the white mannequin in Picture 2: "
                    "standing straight and upright, facing the viewer squarely, shoulders level, the weight on both feet, "
                    "both arms hanging relaxed close to the sides of the body, feet hip-width apart. The pose should be "
                    "pixel accurate to the mannequin's pose, with the same position and size. Keep everything about the "
                    "character from Picture 1 exactly the same{what}: the colours, the materials and the style. Keep the "
                    "plain black background of Picture 1.")


def align_to(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """`img`'s figure scaled and moved onto `ref`'s figure: the same height, feet line and centre, on black."""
    def box(a):
        ys, xs = np.nonzero(a.max(axis=2) > 12)
        if not len(ys):
            raise CharacterError("an image has no figure on its background")
        return xs.min(), ys.min(), xs.max(), ys.max()
    rx0, ry0, rx1, ry1 = box(ref)
    sx0, sy0, sx1, sy1 = box(img)
    s = (ry1 - ry0) / max(1, sy1 - sy0)
    m = np.float32([[s, 0, (rx0 + rx1) / 2 - s * (sx0 + sx1) / 2], [0, s, ry1 - s * sy1]])
    return cv2.warpAffine(img, m, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LANCZOS4, borderValue=(0, 0, 0))


def default_pose_shot() -> Path:
    """The built-in shan_shen_bu_balls template: its mannequin (blender_lib/mannequin.py) is the rig of every built-in
    template; the repose, the height measure and the mannequin preview render ONLY its Mannequin collection in the
    rest pose (its balls and gym set stay hidden). (orbit_wave_360, then shan_shen_bu held this role until they were
    removed from the built-in templates on 2026-10-24 and 2026-10-25; the same rig.)"""
    from . import shots
    return Path(shots.load("shan_shen_bu_balls")["files"]["blend"])


def measure_height(body: dict, shot: Path | None = None) -> dict:
    """The height of a mannequin body (rest pose) and of the template mannequin, in metres (bpy_jobs/
    mannequin_height.py on the built-in templates' rig: default_pose_shot). -> {m, ref_m, ratio, body}."""
    import tempfile

    from . import blender, mannequin_preview
    if shot is None:                                     # a preview or an earlier measure of this body: no Blender
        have = mannequin_preview.cached_height(body or {})
        if have:
            return have
    shot = Path(shot) if shot else default_pose_shot()
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "height.log"
        blender.run_job("mannequin_height.py", shot, ["--body", json.dumps(body or {})], log=log, timeout=180)
        line = next((x for x in log.read_text(encoding="utf-8", errors="replace").splitlines()
                     if x.startswith("HEIGHT ")), None)
    if line is None:
        raise CharacterError("mannequin_height.py gave no result")
    r = json.loads(line[len("HEIGHT "):])
    if not (r.get("ref") or 0) > 0:
        raise CharacterError("the template mannequin has no height")
    if shot == default_pose_shot():
        return mannequin_preview.store_height(body or {}, r["height"], r["ref"])
    return {"m": r["height"], "ref_m": r["ref"], "ratio": round(r["height"] / r["ref"], 4), "body": dict(body or {})}


def height_ratio(c: dict) -> float:
    """The character's height / the template mannequin's (1.0 when unknown): the scale of its bodies on every sheet
    (user, 2026-10-24: "standardize the sheets so they can properly reflect the difference in heights"). Only a height
    measured for the CURRENT body counts."""
    h = c.get("height") or {}
    if not isinstance(c.get("mannequin"), dict) or h.get("body") != c["mannequin"]:
        return 1.0
    try:
        return float(h["ratio"])
    except (KeyError, TypeError, ValueError):
        return 1.0


def ensure_height(c: dict) -> dict:
    """Measure the height of the character's current mannequin body when it is missing or was measured for another
    body (in place; the caller saves). A Blender failure leaves no height (the sheet then uses 1.0) and a note."""
    body = c.get("mannequin")
    if not isinstance(body, dict):
        c.pop("height", None)
        return c
    if (c.get("height") or {}).get("body") == body:
        return c
    if not body:                                   # the template's body: the template's height, no Blender needed
        c["height"] = {"ratio": 1.0, "body": {}}
        return c
    try:
        c["height"] = measure_height(body)
    except Exception as exc:  # noqa: BLE001 - the sheet falls back to the template height; the note says why
        c.pop("height", None)
        c["height_error"] = f"{type(exc).__name__}: {exc}"[:300]
    else:
        c.pop("height_error", None)
    return c


def _file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def front_is_posed(c: dict) -> bool:
    """True when the current front is the one pose_front made (a new pick or edit of the front changes the file)."""
    front = char_dir(c["name"]) / "front.png"
    return bool(c.get("front_posed")) and front.is_file() and c["front_posed"] == _file_hash(front)


@_locked
def pose_front(name: str, shot: str | Path | None = None, seed: int | None = None, rig: str = "MannequinRig",
               collection: str = "Mannequin", **kw) -> dict:
    """The FRONT view in the template mannequin's REST pose (order A, user 2026-10-22: pose the front, then make the
    views from it; AnyPose on each view put the MASK on the back of the head). Blender renders the bare mannequin of
    `shot` (bpy_jobs/render_turnaround.py, CPU); the front is aligned to it (height, feet line, centre) and
    Qwen-Image-Edit-2511 + the AnyPose LoRAs (serotoninboi/AnyPose, Apache-2.0) gives it the mannequin's pose:
    Picture 1 = the character (the identity), Picture 2 = the mannequin (the pose). The old front stays in the history
    (undo); the views are out of date after it (make_views makes them from the posed front)."""
    from . import blender
    c = load(name)
    d = char_dir(name)
    front = d / "front.png"
    if not front.is_file():
        raise CharacterError(f"{name}: no front view yet (create + pick, or import --image)")
    shot = Path(shot) if shot else default_pose_shot()
    work = d / "repose"
    work.mkdir(parents=True, exist_ok=True)
    # the pose source has the character's own proportions (user: "each mannequin needs its own proportions"), not its
    # height: a sheet has no absolute size
    body = {k: v for k, v in mannequin_body(name).items() if k != "scale"}
    blender.run_job("render_turnaround.py", shot.resolve(), [str((work / "mannequin").resolve()), "--rig", rig,
                                                              "--collection", collection]
                    + (["--body", json.dumps(body)] if body else []), log=work / "turnaround.log")
    desc = (c.get("description") or "").strip().rstrip(".")
    what = f" ({desc[0].lower()}{desc[1:]})" if desc else ""
    seed = int(seed) if seed not in (None, "") else int(c.get("seed") or 17)
    mann = work / "mannequin" / "front.png"
    aligned = work / "front_aligned.png"
    cv2.imwrite(str(aligned), align_to(cv2.imread(str(front)), cv2.imread(str(mann))))
    values, _ = quality_values("quality", {}, EDIT_SETTINGS)
    values.update(instruction=POSE_INSTRUCTION.format(what=what), seed=seed)
    wf = ProfileWorkflow("qwenedit", "anypose")
    out = work / "front_posed.png"
    r = image_job("qwenedit", "anypose", values, {wf.targets("image")[0]: aligned, wf.targets("reference")[0]: mann},
                  out, f"{name}_pose_front", item=(1, 1), **kw)
    c = load(name)
    version = len(c.get("edits", [])) + 1
    before = _keep_before(c, "front", version)
    shutil.copyfile(out, front)
    c.setdefault("edits", []).append({"version": version, "target": "front", "instruction": "pose like the template "
                                      f"mannequin (AnyPose, {shot.name})", "reference": str(mann), "before": str(before),
                                      "seconds": r["seconds"], "mode": "repose", "seed": seed})
    c["front_posed"] = _file_hash(front)
    c["reposed"] = {"shot": str(shot), "views": ["front"], "method": "anypose", **({"body": body} if body else {})}
    save(c)
    if kw.get("on_event"):
        kw["on_event"]({"event": "posed", "view": "front", "file": str(front), "seconds": r["seconds"]})
    return c


@_locked
def repose(name: str, shot: str | Path | None = None, seed: int | None = None, mode: str = "quality",
           distance: str | None = None, face: str = "front", **kw) -> dict:
    """Order A in one step: pose the front like the template mannequin (pose_front), then make the right, back and
    left views and the sheet from the posed front (make_views, the Multiple-Angles LoRA)."""
    pose_front(name, shot=shot, seed=seed, **kw)
    return make_views(name, mode=mode, distance=distance, face=face, pose=False, **kw)


# the standard scale of every sheet: the template mannequin's height = this part of the panel height (2026-10-24). The
# old 0.94 never fitted: the row was too wide and shrank to ~0.79 by each character's width, so sheets had no common
# scale; at 0.78 the bodies keep one scale and the face close-up gives way when a wide costume needs the room.
SHEET_FILL = 0.78


def sheet_image(face, views: list, ratio: float, size: tuple[int, int] = (1536, 896), props: list | None = None,
                layout: dict | None = None):
    """The panels5 sheet picture from cut-outs (face + SHEET_PANELS views) at the character's height ratio; the one
    drawing of build_sheet and of a run's prop sheet (runner.prop_sheet: `props` = the prop column)."""
    return charsheet.assemble_panels(face, views, size=size, panel_bg=(0, 0, 0), face_fill=0.6, fill=SHEET_FILL,
                                     body_ratio=ratio, props=props, layout=layout)


def build_sheet(name: str, c: dict | None = None, timings: dict | None = None, mode: str | None = None) -> dict:
    c = c or load(name)
    d = char_dir(name)
    order = SHEET_ORDER
    imgs = {k: cv2.imread(c["files"][k]) for k in order}
    if any(v is None for v in imgs.values()):
        raise CharacterError(f"{name}: missing view images {[k for k, v in imgs.items() if v is None]}")
    cut = {k: charsheet.cutout(v) for k, v in imgs.items()}
    face = charsheet.cutout(cv2.imread(c["files"]["face"])) if c["files"].get("face") else None
    ensure_height(c)                                     # one scale for every sheet: the character's real height
    ratio = height_ratio(c)
    if face is not None:                                 # the panel layout (black panels: the black bodysuit and a
        sheet = sheet_image(face, [cut[k] for k in SHEET_PANELS], ratio)   # background cannot be told apart on grey)
        c["sheet_layout"] = SHEET_LAYOUT
    else:
        sheet = charsheet.assemble([cut[k] for k in order], face=face, common_scale=False)
        c["sheet_layout"] = "row5"
    if c["sheet_layout"] == SHEET_LAYOUT:
        c["sheet_scale"] = ratio                         # the height ratio this sheet was drawn for
    else:                                                # the old row layout (no face) does not scale its bodies
        c.pop("sheet_scale", None)
    tmp = d / f"sheet.{secrets.token_hex(4)}.tmp.png"    # a failed write must not count as a redraw (Codex review)
    if not cv2.imwrite(str(tmp), sheet):
        tmp.unlink(missing_ok=True)
        raise CharacterError(f"{name}: cannot write the sheet")
    os.replace(tmp, d / "sheet.png")
    checks = charsheet.palette_check(cut["front"], {k: cut[k] for k in order[1:]})
    c["files"]["sheet"] = str(d / "sheet.png")
    old_styled = c["files"].pop("sheet_styled", None)    # styled from the old sheet: a video must not use it any more;
    if old_styled:                                       # the file stays (the app says to run the style pass again)
        c["files"]["sheet_styled_old"] = old_styled
    c.update(status="sheet", checks=checks, views_model="Qwen-Image-Edit-2511 + Multiple-Angles LoRA",
             views_mode=mode, views_seconds=timings)
    save(c)
    return c


@_locked
def style_pass(name: str, style_lora: str, keep: float = 0.5, strength: float = 0.8, seed: int = 333, **kw) -> dict:
    """Redraw the sheet with LTX-2.5 and the scene's style LoRA (the same model and LoRA as the video; Qwen cannot load
    an LTX LoRA). Image-to-image with the "redraw" graph: the sheet is encoded and sampled from S = 1 - keep (about
    `keep` of its structure stays). Before 2026-10-22 the restyle graph held the sheet as frame 0 of a still clip and
    returned a near copy: the pass had almost no effect. A low keep can change design details (the panorama test at
    keep 0.15 added domes and clouds): check every panel after a pass."""
    c = load(name)
    d = char_dir(name)
    sheet = d / "sheet.png"
    if not sheet.is_file():
        raise CharacterError(f"{name}: no sheet yet")
    fitted = d / "styled_in.png"
    video.fit_image(sheet, fitted, 1536, 896, mode="pad", color="black")
    from . import models as _models
    from . import runner
    if not 0.0 <= keep <= 0.95:
        raise CharacterError("keep must be between 0 and 0.95 (about this part of the sheet's structure stays)")
    prof = runner.LAYOUT_PROFILE
    wf = ProfileWorkflow(prof, "redraw", models=_models.job_bundle())
    start = round(1.0 - float(keep), 4)
    sig = (runner.dev_source_sigmas(start, 1536, 896, 9) if wf.capabilities.get("recipe") == "full"
           else runner.source_sigmas(start))
    look = c.get("style") or "the same style as the input sheet"
    who = f": {c['description']}" if c.get("description") else "."
    prompt = (f"A character reference sheet in {look}, on a plain black background. Left panel: a close-up of the "
              f"character's face. Next four panels: the same character in full body (front, right side, back, left side)"
              f"{who} A still image: nothing moves, the camera does not move.")
    values = {"prompt": prompt, "negative": "worst quality, blurry, distorted, text, watermark", "width": 1536,
              "height": 896, "seed": seed, "source_sigmas": sig, "style_lora": style_lora, "style_lora_strength": strength}
    token = secrets.token_hex(3)
    old = d / "sheet_styled.png"
    if old.is_file():                              # keep the earlier styled sheet (no change is lost)
        (d / "history").mkdir(exist_ok=True)
        shutil.copyfile(old, d / "history" / f"sheet_styled_{time.strftime('%Y%m%d-%H%M%S')}.png")
    r = image_job(prof, "redraw", dict(values, output_prefix_last=f"cinestudio/char_{name}_style_{token}_last"),
                  {wf.targets("reference")[0]: fitted}, d / "sheet_styled.png", f"{name}_style", **kw)
    c["files"]["sheet_styled"] = r["file"]
    c.update(status="ready", style_lora=style_lora, style_keep=keep)
    save(c)
    return c


@_locked
def finish(name: str) -> dict:
    """Mark a sheet as ready without a style pass."""
    c = load(name)
    if not c["files"].get("sheet"):
        raise CharacterError(f"{name}: no sheet yet")
    if c.get("sheet_stale"):
        raise CharacterError(f"{name}: the sheet does not show the current mannequin height yet: draw it again "
                             f"(character mannequin {name} ..., or the Mannequin block's Save body)")
    c["status"] = "ready"
    save(c)
    return c


def for_run(name: str) -> tuple[str, str]:
    """(sheet file, panel description) for `run new --character`."""
    c = load(name)
    if c["status"] != "ready":
        raise CharacterError(f"character {name!r} is {c['status']}, not ready (finish the views and the sheet first)")
    if c.get("sheet_stale"):
        raise CharacterError(f"character {name!r}: the sheet does not show the current mannequin height yet (save its "
                             f"body again to draw it)")
    sheet = c["files"].get("sheet_styled") or c["files"]["sheet"]
    desc = (c.get("description") or "").strip()
    if c["source"] == "sheet":
        text = desc or "A character reference sheet: the same character in every panel, on black."
    elif c.get("sheet_layout") == SHEET_LAYOUT:         # the Ingredients 2.5 card's panel wording
        text = ("**Left Panel (Character):** a close-up of the character's face. **Panels 2-5 (Character):** the same "
                "character in full body shown from multiple angles (front, right side, back, left side) on black"
                + (f": {desc}" if desc else "."))
    else:
        text = ("Left panel: a close-up of the character's face. Next four panels: the same character in full body "
                "(front, right side, back, left side) on black" + (f": {desc}" if desc else "."))
    return sheet, text


MANNEQUIN_SCALE = (0.3, 2.0)
MANNEQUIN_KEYS = ("scale", "head", "neck", "torso", "legs", "arms")


def mannequin_body(name: str | None) -> dict:
    """The character's own MANNEQUIN BODY (character.json "mannequin": {"scale", "head", "legs", "arms"}, each 1.0 =
    the template's mannequin; bpy_jobs/mannequin_body.py). User, 2026-10-23: "swap the model and apply the motion from
    the video" (the template keeps the motion, the character brings its body) and "each mannequin needs its own
    proportions" (AnyPose against the adult mannequin gave chibi kitsune a small head and long legs). Only keys that
    differ from 1.0; {} = the template's mannequin as it is."""
    if not name:
        return {}
    try:
        rec = load(name).get("mannequin") or {}
    except CharacterError:
        return {}
    lo, hi = MANNEQUIN_SCALE
    out = {}
    for k, v in rec.items():
        if k not in MANNEQUIN_KEYS:
            raise CharacterError(f"character {name!r}: unknown mannequin key {k!r}")
        v = float(v)
        if not lo <= v <= hi:
            raise CharacterError(f"character {name!r}: mannequin {k} {v} is outside {lo}..{hi}")
        if abs(v - 1.0) > 1e-9:
            out[k] = v
    return out


def video_missing(c: dict) -> list[str]:
    """What a character still needs before a Blender run can use it (user, 2026-10-24: "each character should have its
    own associated mannequin ... a character isnt ready to be used until a mannequin is set for the character with its
    respective proxy kit"): "sheet" (finish the views and the sheet) and "mannequin" (character.json "mannequin", {} =
    the template's body as it is; a key that is present counts as set). The proxy kit is OPTIONAL (user, 2026-10-24:
    "by default models shouldnt have a proxy kit, it should only be assigned thru the ai mcp option"): none by default,
    an AI agent adds it with the MCP tool set_proxy_kit."""
    miss = []
    if c.get("status") != "ready" or c.get("sheet_stale"):   # sheet_stale: drawn for another height (set_mannequin)
        miss.append("sheet")
    if not isinstance(c.get("mannequin"), dict):
        miss.append("mannequin")
    return miss


VIDEO_STEPS = {"sheet": "finish its views and sheet (character finish NAME)",
               "mannequin": "set its mannequin body (the Mannequin view, or character mannequin NAME --scale S ... "
                            "or --default)"}


PROXY_PARTS = ("mask", "hair", "suit", "shoes", "bands", "jacket", "cape", "head")     # bpy_jobs/proxy_kit.py


def check_body(body: dict | None) -> dict:
    """A body checked (known keys, finite values in the limits) and cleaned: values of 1.0 are dropped ({} = the
    template's body)."""
    lo, hi = MANNEQUIN_SCALE
    if body is not None and not isinstance(body, dict):
        raise CharacterError("the mannequin body is an object of values, for example {\"scale\": 0.86}")
    out = {}
    for k, v in (body or {}).items():
        if k not in MANNEQUIN_KEYS:
            raise CharacterError(f"unknown mannequin key {k!r} (use {', '.join(MANNEQUIN_KEYS)})")
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise CharacterError(f"the mannequin {k} must be a number") from None
        if not (math.isfinite(v) and lo <= v <= hi):
            raise CharacterError(f"the mannequin {k} must be {lo}..{hi}")
        if abs(v - 1.0) > 1e-9:
            out[k] = round(v, 4)
    return out


def set_stand_in(name: str, body: dict, spec: dict) -> dict:
    """The Mannequin editor's ONE save (2026-10-24): the whole body (not merged) and the whole kit, both checked first,
    then in one locked step: body, kit, height (from the preview cache when there is one), sheet redraw."""
    body = check_body(body)
    validate_proxy(spec)
    with char_lock(name):
        c = load(name)
        c["mannequin"] = body
        c["proxy"] = spec
        ensure_height(c)
        return _rescale_locked(name, c)


def set_mannequin(name: str, body: dict | None = None, default: bool = False) -> dict:
    """Set the character's mannequin body: the given keys change, the others stay; `default` = the template's
    mannequin as it is ({}: set, every value 1.0). -> the record."""
    lo, hi = MANNEQUIN_SCALE
    given = {}
    for k, v in (body or {}).items():
        if k not in MANNEQUIN_KEYS:
            raise CharacterError(f"unknown mannequin key {k!r} (use {', '.join(MANNEQUIN_KEYS)})")
        v = float(v)
        if not (math.isfinite(v) and lo <= v <= hi):
            raise CharacterError(f"the mannequin {k} must be {lo}..{hi}")
        given[k] = round(v, 4)
    if not given and not default:
        raise CharacterError("give at least one mannequin value, or the default body")
    if given and default:
        raise CharacterError("the default body sets every value to 1.0: give it alone")
    with char_lock(name):                               # ONE locked step: body, height, sheet (Codex review: a
        c = load(name)                                  # job between the save and the redraw left a mismatch)
        merged = {} if default else {**(c.get("mannequin") or {}), **given}
        c["mannequin"] = {k: v for k, v in merged.items() if abs(v - 1.0) > 1e-9}   # {} = the template's body: SET
        ensure_height(c)
        return _rescale_locked(name, c)


def rescale_sheet(name: str) -> dict:
    """Draw the sheet again at the character's current height (after a body change); a ready character stays ready.
    Nothing is drawn without the views and the face (a sheet that is still being made), or when the sheet already
    shows this height."""
    with char_lock(name):
        c = load(name)
        ensure_height(c)
        return _rescale_locked(name, c)


def _rescale_locked(name: str, c: dict) -> dict:
    """(char_lock held) Save `c`; when its sheet was drawn for another height, mark it out of date FIRST (saved: a crash
    cannot leave a ready character with the wrong sheet), draw it again, then clear the mark and keep `ready`. A failed
    redraw leaves the mark and status "sheet" (finish refuses until a redraw succeeds)."""
    files = c.get("files") or {}
    drawable = files.get("sheet") and files.get("face") and all(files.get(k) for k in SHEET_ORDER)
    want = height_ratio(c)
    if not drawable or (not c.get("sheet_stale") and abs(float(c.get("sheet_scale") or -1.0) - want) < 1e-6):
        save(c)
        return c
    ready = c.get("status") == "ready"
    c["sheet_stale"] = True
    save(c)
    try:
        c = build_sheet(name, c)
    except BaseException:
        c = load(name)
        c.update(status="sheet", sheet_stale=True)
        save(c)
        raise
    c.pop("sheet_stale", None)
    if ready:
        c["status"] = "ready"
    save(c)
    return c


def mark_presets() -> dict:
    """The mask MARK presets (bpy_jobs/mask_marks.json, shared with bpy_jobs/proxy_kit.py)."""
    f = Path(__file__).resolve().parent.parent / "bpy_jobs" / "mask_marks.json"
    return {k: v for k, v in json.loads(f.read_text(encoding="utf-8")).items() if not k.startswith("_")}
BAND_BONES = tuple(f"{b}.{s}" for b in ("upper_arm", "forearm", "hand", "thigh", "shin", "foot") for s in "LR") + \
    ("spine", "chest", "neck", "hips")


def _num(where: str, v, lo: float, hi: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise CharacterError(f"proxy {where}: a number is needed, not {v!r}") from None
    if isinstance(v, bool) or not (math.isfinite(f) and lo <= f <= hi):
        raise CharacterError(f"proxy {where}: {v!r} is outside {lo}..{hi}")
    return f


def _colour(where: str, part: dict, key: str = "color") -> None:
    if key not in part:
        return
    v = part[key]
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        _num(where + "." + key, v, 0.0, 1.0)                 # a grey
        return
    if not (isinstance(v, (list, tuple)) and len(v) == 3):
        raise CharacterError(f"proxy {where}.{key}: three numbers 0..1 (linear RGB) or one grey")
    for i, x in enumerate(v):
        _num(f"{where}.{key}[{i}]", x, 0.0, 1.0)


def _choice(where: str, v, allowed) -> None:
    if v not in allowed:
        raise CharacterError(f"proxy {where}: {v!r} (use {', '.join(map(str, allowed))})")


def _validate_head_fit(fit) -> None:
    """head.fit = a head FITTED to the character's views (proxy_kit.fitted_head): levels v (top to bottom) with the
    outline a / yf / yb, the mask region {v0, v1, hw, back} and the eye outline [[u, v], ...], in head units."""
    if not isinstance(fit, dict):
        raise CharacterError("proxy head.fit: an object (from the head fit of the character's views)")
    lv = fit.get("v")
    if not (isinstance(lv, list) and 4 <= len(lv) <= 400):
        raise CharacterError("proxy head.fit.v: a list of 4..400 levels")
    nums = lambda xs: all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)  # noqa: E731
                          and abs(x) < 20 for x in xs)
    if not nums(lv) or any(lv[i] <= lv[i + 1] for i in range(len(lv) - 1)):
        raise CharacterError("proxy head.fit.v: numbers from the top down (decreasing)")
    for k in ("a", "yf", "yb"):
        if not (isinstance(fit.get(k), list) and len(fit[k]) == len(lv) and nums(fit[k])):
            raise CharacterError(f"proxy head.fit.{k}: one number per level")
    mk = fit.get("mask") or {}
    for k in ("hw", "back", "left", "right"):
        if k in mk and not (isinstance(mk[k], list) and len(mk[k]) == len(lv) and nums(mk[k])):
            raise CharacterError(f"proxy head.fit.mask.{k}: one number per level")
    if "eye_scale" in fit:
        _num("head.fit.eye_scale", fit["eye_scale"], 0.5, 2.5)
    if "ridge" in fit:
        _num("head.fit.ridge", fit["ridge"], 0.0, 0.5)
    if "eye_axis" in fit:
        _num("head.fit.eye_axis", fit["eye_axis"], -1.0, 1.0)
    if "shade" in fit:
        _num("head.fit.shade", fit["shade"], 0.2, 1.0)
    if "side_cover" in fit:
        _num("head.fit.side_cover", fit["side_cover"], 0.0, 1.5)
    if "under" in fit and not isinstance(fit["under"], bool):
        raise CharacterError("proxy head.fit.under: true or false")
    for k in ("cover_full", "cover_end", "hair_end"):
        if k in fit:
            _num("head.fit." + k, fit[k], 0.0, 1.0)
    if "hair_volume" in fit:
        _num("head.fit.hair_volume", fit["hair_volume"], 0.8, 1.5)
    for k, hi in (("proud", 0.5), ("chin_drop", 0.8)):
        if k in fit:
            _num("head.fit." + k, fit[k], 0.0, hi)
    if "shade_side" in fit and fit["shade_side"] not in ("left", "right"):
        raise CharacterError("proxy head.fit.shade_side: left or right (the picture side of the darker half)")
    if "round_top" in fit and not isinstance(fit["round_top"], bool):
        raise CharacterError("proxy head.fit.round_top: true or false")
    if "asymmetric" in fit and not isinstance(fit["asymmetric"], bool):
        raise CharacterError("proxy head.fit.asymmetric: true or false")
    eye = fit.get("eye") or []
    if not (isinstance(eye, list) and len(eye) <= 200 and all(isinstance(p, list) and len(p) == 2 and nums(p) for p in eye)):
        raise CharacterError("proxy head.fit.eye: a list of [u, v] points")


def validate_proxy(spec) -> dict:
    """The proxy kit spec checked in full before it is saved (Codex review 2026-10-24: {"mask": true} counted as ready
    and failed later inside Blender): every part a dict of known types, numbers finite and in range, colours three
    numbers 0..1, preset names known, band bones real, mask marks inside the mask (their footprint on the unit disk,
    above a pointed mask's chin and below a cut top). -> spec."""
    if not isinstance(spec, dict) or set(spec) - set(PROXY_PARTS):
        raise CharacterError(f"the proxy spec is a JSON object with the parts {', '.join(PROXY_PARTS)}")
    for part in ("mask", "hair", "suit", "shoes", "jacket", "cape", "head"):
        if part in spec and not isinstance(spec[part], dict):
            raise CharacterError(f"proxy {part}: an object is needed, for example {{}}")
    m = spec.get("mask")
    if m is not None:
        for k in ("eyes", "ears"):
            if k in m and not isinstance(m[k], bool):
                raise CharacterError(f"proxy mask.{k}: true or false")
        if "shape" in m:
            _choice("mask.shape", m["shape"], ("round", "pointed"))
        for k, lo, hi in (("top", -1.0, 1.5), ("ear_height", 0.05, 3.0), ("ear_width", 0.05, 2.0),
                          ("eye_size", 0.3, 3.0), ("shade", 0.0, 1.0), ("width", 0.5, 1.6), ("length", 0.5, 1.6),
                          ("depth", 0.5, 1.6), ("chin_back", 0.0, 0.8), ("eye_height", -0.5, 0.6)):
            if k in m and not (k == "shade" and isinstance(m[k], (list, tuple))):
                _num("mask." + k, m[k], lo, hi)
        _colour("mask", m)
        _colour("mask", m, "shade")
        _colour("mask", m, "ear_inner")
        marks = m.get("marks")
        if isinstance(marks, str):                        # a preset: its marks get the same checks as a list
            presets = mark_presets()
            _choice("mask.marks", marks, tuple(presets))
            marks = presets[marks]
        if marks is not None:
            if not isinstance(marks, list):
                raise CharacterError("proxy mask.marks: a preset name or a list of marks")
            round_mask = m.get("shape") == "round"
            z_top = None                                  # the mask's cut top on its unit sphere (proxy_kit.py)
            if m.get("top") is not None:
                z_top = (float(m["top"]) - (0.06 if round_mask else -0.05)) / (1.14 if round_mask else 1.12)
            for i, mk in enumerate(marks):
                w = f"mask.marks[{i}]"
                if not isinstance(mk, dict):
                    raise CharacterError(f"proxy {w}: an object with x, z, w, h, roll, color")
                x, z = _num(w + ".x", mk.get("x", 0.0), -1.0, 1.0), _num(w + ".z", mk.get("z", 0.0), -1.0, 1.0)
                hw, hh = _num(w + ".w", mk.get("w", 0.1), 0.005, 0.6), _num(w + ".h", mk.get("h", 0.05), 0.005, 0.6)
                _num(w + ".roll", mk.get("roll", 0.0), -360.0, 360.0)
                if "color" in mk and not isinstance(mk["color"], (list, tuple)):
                    raise CharacterError(f"proxy {w}.color: three numbers 0..1 (linear RGB)")
                _colour(w, mk)
                reach = max(hw, hh)                       # the footprint, whatever the roll
                if (abs(x) + reach) ** 2 + (abs(z) + reach) ** 2 > 0.92:
                    raise CharacterError(f"proxy {w}: it reaches past the mask's edge (keep x, z and its size smaller)")
                if not round_mask and z - reach < -0.45:
                    raise CharacterError(f"proxy {w}: too low for a pointed mask (its chin narrows below z -0.45)")
                if z_top is not None and z + reach > z_top - 0.02:
                    raise CharacterError(f"proxy {w}: above the mask's cut top (mask.top)")
    hd = spec.get("head")
    if hd is not None:                                    # the head shape + the neck colour (proxy_kit.py)
        unknown = set(hd) - {"width", "depth", "length", "neck", "fit", "asset"}
        if unknown:
            raise CharacterError(f"proxy head: unknown keys {sorted(unknown)} (use width, depth, length, neck, fit, "
                                 "asset)")
        if "fit" in hd:
            _validate_head_fit(hd["fit"])
        if "asset" in hd:                                 # a modelled head (.blend in head units; proxy_kit.head_asset)
            a = hd["asset"]
            a = {"file": a} if isinstance(a, str) else a
            if not (isinstance(a, dict) and isinstance(a.get("file"), str) and a["file"].lower().endswith(".blend")):
                raise CharacterError("proxy head.asset: {\"file\": \"<path>.blend\"} (a head modelled in head units)")
            if set(a) - {"file", "scale"}:
                raise CharacterError("proxy head.asset: only the keys file and scale")
            if not Path(a["file"]).is_file():
                raise CharacterError(f"proxy head.asset: the file {a['file']!r} does not exist")
            if "scale" in a:
                _num("head.asset.scale", a["scale"], 0.5, 2.0)
        for k in ("width", "depth", "length"):
            if k in hd:
                _num("head." + k, hd[k], 0.6, 1.4)
        if "neck" in hd:
            _colour("head", {"color": hd["neck"]})
    h = spec.get("hair")
    if h is not None:
        if "length" in h:
            _choice("hair.length", h["length"], ("short", "medium", "long"))
        for k in ("locks", "back"):
            if k in h:
                _choice("hair." + k, h[k], ("chin", "shoulder", "chest"))
        if "front" in h:
            _num("hair.front", h["front"], -1.5, 1.5)
        if "cover" in h and not isinstance(h["cover"], bool):
            raise CharacterError("proxy hair.cover: true or false")
        _colour("hair", h)
        _colour("hair", h, "shade")
    for part in ("suit", "shoes"):
        if spec.get(part) is not None:
            _colour(part, spec[part])
            _colour(part, spec[part], "shade")
    j = spec.get("jacket")
    if j is not None:
        _colour("jacket", j)
        _colour("jacket", j, "shade")
        for k in ("sleeve", "bottom"):
            if k in j:
                _num("jacket." + k, j[k], 0.0, 1.0)
        if "open" in j:                                   # an open front (proxy_kit.py: the suit shows in the gap)
            _num("jacket.open", j["open"], 0.0, 0.9)
        for k in ("fit", "sleeve_fit"):                   # how far the shell stands off the body (proxy_kit.py)
            if k in j:
                _num("jacket." + k, j[k], 0.5, 3.0)
    cp = spec.get("cape")
    if cp is not None:
        _colour("cape", cp)
        _colour("cape", cp, "shade")
        if "length" in cp:
            _choice("cape.length", cp["length"], ("hips", "thigh", "knee"))
        for k, lo, hi in (("width", 0.2, 4.0), ("flare", 0.2, 4.0), ("wrap", 0.0, 1.0)):
            if k in cp:
                _num("cape." + k, cp[k], lo, hi)
    if "bands" in spec:
        if not isinstance(spec["bands"], list):
            raise CharacterError("proxy bands: a list of bands")
        for i, b in enumerate(spec["bands"]):
            w = f"bands[{i}]"
            if not isinstance(b, dict):
                raise CharacterError(f"proxy {w}: an object with bone, from, to, color, scale")
            _choice(w + ".bone", b.get("bone"), BAND_BONES)
            f0, f1 = _num(w + ".from", b.get("from", 0.0), 0.0, 1.0), _num(w + ".to", b.get("to", 1.0), 0.0, 1.0)
            if f0 >= f1:
                raise CharacterError(f"proxy {w}: from must be below to")
            if "scale" in b:
                _num(w + ".scale", b["scale"], 0.5, 3.0)
            _colour(w, b)
            _colour(w, b, "shade")
    return spec


def set_proxy(name: str, spec: dict) -> dict:
    """Set the character's proxy kit (bpy_jobs/proxy_kit.py parts); {} = no stand-in parts (set). -> the record."""
    validate_proxy(spec)
    with char_lock(name):
        c = load(name)
        c["proxy"] = spec
        save(c)
    return c


def require_video_ready(name: str) -> dict:
    """The character record, or CharacterError naming what is missing for a Blender run."""
    c = load(name)
    miss = video_missing(c)
    if miss:
        raise CharacterError(f"character {name!r} is not ready for a video: "
                             + "; ".join(VIDEO_STEPS[m].replace("NAME", name) for m in miss))
    return c


def mannequin_scale(name: str | None) -> float:
    """The character's whole-mannequin scale (1.0 = the template's height)."""
    return float(mannequin_body(name).get("scale", 1.0))


def black_background(img: np.ndarray, tolerance: int = 18) -> np.ndarray:
    """The image with its border-connected background set to pure black."""
    from .refsheet import background_mask
    out = img.copy()
    out[background_mask(img, tolerance)] = 0
    return out


TARGETS = ["front", *VIEWS]                         # images that can be edited: the front view and every view


def _target_path(c: dict, target: str) -> Path:
    if target not in TARGETS:
        raise CharacterError(f"cannot edit {target!r}; use one of {TARGETS}")
    d = char_dir(c["name"])
    return d / "front.png" if target == "front" else Path(c["files"].get(target) or d / "views" / f"{target}.png")


def _keep_before(c: dict, target: str, version: int) -> Path:
    """Copy the image that a change replaces, so the change can be undone."""
    src = _target_path(c, target)
    d = char_dir(c["name"])
    dst = d / f"front_v{version}.png" if target == "front" else d / "views" / "history" / f"{target}_v{version}.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def _after_change(name: str, c: dict, target: str) -> dict:
    """After a change of one image: a new front makes the views out of date; a new view only needs a
    new sheet (made at once, no GPU), and a styled sheet must be made again."""
    if target == "front":
        _drop_generated(c)
        c.update(status="front")
        save(c)
        return c
    c["files"].pop("sheet_styled", None)
    if all(Path(c["files"].get(k) or "").is_file() for k in SHEET_ORDER):
        return build_sheet(name, c, c.get("views_seconds"), c.get("views_mode"))
    save(c)
    return c


def _parse_region(region, w: int, h: int) -> tuple[int, int, int, int] | None:
    if region in (None, ""):
        return None
    x0, y0, x1, y1 = (int(round(float(v))) for v in (region.split(",") if isinstance(region, str) else region))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise CharacterError("the region must be at least 16 x 16 pixels inside the image")
    return x0, y0, x1, y1


def edit_picture(img: Path, instruction: str, work: Path, tag: str, reference: str | Path | None = None,
                 mode: str = "quality", seed: int | None = None, settings: dict | None = None,
                 region: str | tuple | None = None, keep: Callable[[], Any] | None = None, **kw) -> dict:
    """One Qwen-Image-Edit-2511 change of the picture `img`, written back into `img` (the shared core of character and
    prop edits: cinestudio/props.py). Picture 1 = `img` (its border-connected background set to pure black first),
    Picture 2 = the optional `reference`. `region` "x0,y0,x1,y1": crop and stitch (only that box changes). `keep` is
    called after the checks and before the GPU job (the caller saves the picture before the change there).
    Returns {"seconds", "seed", "used" (the complete settings with the seed), "box" (the region or None)}."""
    seed = int(seed) if seed not in (None, "") else secrets.randbelow(2 ** 31)
    values, used = quality_values(mode, settings, EDIT_SETTINGS)
    wf = ProfileWorkflow("qwenedit", "edit")
    if float(values.get("denoise") or 1.0) < 1.0 and "cfg_tail_start" in wf.bind["inputs"]:
        # the CFG tail is an ABSOLUTE noise range (CFGOverride): with denoise < 1 the truncated schedule falls mostly or
        # wholly inside it (0.5: ~28 of 40 steps without the negative). Partial edits keep full CFG (Codex review)
        values["cfg_tail_start"] = 1.0
    ref = Path(reference) if reference else None              # without a reference: Picture 1 only
    if ref is not None and cv2.imread(str(ref)) is None:
        raise CharacterError(f"cannot read the reference {ref}")
    full = cv2.imread(str(img))
    if full is None:
        raise CharacterError(f"cannot read the picture {img}")
    box = _parse_region(region, full.shape[1], full.shape[0])
    if keep is not None:
        keep()
    # pure black background before the edit: faint noise in a near-black background grows with each
    # edit and became pixel bars after a chain of edits (2026-10-19)
    clean_in = Path(work) / "edit_in.png"
    if box:
        x0, y0, x1, y1 = box
        k = 1024 / max(x1 - x0, y1 - y0)
        # the crop's border is the picture (a tunic), not the background: no black background here
        cv2.imwrite(str(clean_in), cv2.resize(full[y0:y1, x0:x1], (round((x1 - x0) * k), round((y1 - y0) * k)),
                                              interpolation=cv2.INTER_CUBIC))
    else:
        cv2.imwrite(str(clean_in), black_background(full))
    out = Path(work) / "edit_out.png"
    r = image_job("qwenedit", "edit", {"instruction": instruction, "seed": seed, **values},
                  {wf.targets("image")[0]: clean_in, **({wf.targets("reference")[0]: ref} if ref else {})},
                  out, tag, **kw)
    if box:
        x0, y0, x1, y1 = box
        res = cv2.resize(cv2.imread(str(out)), (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        f = max(3, min(x1 - x0, y1 - y0) // 10)               # soft edge: no visible seam at the box border
        ramp = lambda n: np.minimum(np.minimum(np.arange(n), np.arange(n)[::-1]) / f, 1.0)
        m = (ramp(y1 - y0)[:, None] * ramp(x1 - x0)[None, :])[..., None]
        part = full[y0:y1, x0:x1].astype(np.float32)
        full[y0:y1, x0:x1] = np.clip(part * (1 - m) + res.astype(np.float32) * m, 0, 255).astype(np.uint8)
        cv2.imwrite(str(img), full)
        out.unlink(missing_ok=True)
    else:
        out.replace(img)
    return {"seconds": r["seconds"], "seed": seed, "used": {**used, "seed": seed}, "box": box}


@_locked
def edit_image(name: str, instruction: str, target: str = "front", reference: str | Path | None = None,
               mode: str = "quality", seed: int | None = None, settings: dict | None = None,
               region: str | tuple | None = None, **kw) -> dict:
    """Change details of one image (the front view or a view) by an instruction (Qwen-Image-Edit-2511),
    optionally with a second image as a reference ("give the hand five fingers like in image 2").
    The earlier image is kept, so every edit can be undone. After a front edit the views must be made
    again; after a view edit the sheet is made again at once.

    region "x0,y0,x1,y1" (crop and stitch): only that box is edited. It is cut out and made large (about
    1024 px), so the model sees a small detail (a hand) big; the result is made small again and pasted back
    with soft edges. Nothing outside the box can change."""
    c = load(name)
    d = char_dir(name)
    img = _target_path(c, target)
    if not img.is_file():
        raise CharacterError(f"{name}: no {target} image yet")
    version = len(c.get("edits", [])) + 1
    kept: dict = {}
    r = edit_picture(img, instruction, d, f"{name}_{target}_edit{version}", reference, mode, seed, settings, region,
                     keep=lambda: kept.setdefault("before", _keep_before(c, target, version)), **kw)
    box = r["box"]
    c = load(name)                                            # reread: only this change is applied
    c["files"][target] = str(img)
    c.setdefault("edits", []).append({"version": version, "target": target, "instruction": instruction,
                                      "reference": str(reference) if reference else None,
                                      "before": str(kept["before"]), "seconds": r["seconds"], "mode": mode,
                                      "settings": r["used"], **({"region": list(box)} if box else {})})
    return _after_change(name, c, target)


def edit_front(name: str, instruction: str, reference: str | Path | None = None, mode: str = "quality",
               seed: int | None = None, **kw) -> dict:
    """Change details of the front view (see edit_image)."""
    return edit_image(name, instruction, "front", reference, mode, seed, **kw)


@_locked
def redo_view(name: str, view: str, seed: int | None = None, mode: str = "quality", settings: dict | None = None,
              **kw) -> dict:
    """Make one view again from the front view with another seed (for example when a hand came out with
    four fingers). The earlier view is kept, so `undo_edit(name, view)` brings it back."""
    if view not in VIEWS:
        raise CharacterError(f"redo works on the views {list(VIEWS)}, not {view!r}")
    c = load(name)
    d = char_dir(name)
    front = d / "front.png"
    if not front.is_file():
        raise CharacterError(f"{name}: no front view yet")
    values, used = quality_values(mode, settings, VIEW_SETTINGS)
    seed = int(seed) if seed not in (None, "") else secrets.randbelow(2 ** 31)
    img = d / "views" / f"{view}.png"
    version = len(c.get("edits", [])) + 1
    before = _keep_before(c, view, version) if img.is_file() else None
    wf = ProfileWorkflow("qwenedit", "angle")
    r = image_job("qwenedit", "angle", {"instruction": view_prompt(view, c.get("view_distance") or "wide shot"),
                                        "seed": seed, **values},
                  {wf.targets("front")[0]: front}, img, f"{name}_{view}_redo{version}", **kw)
    c = load(name)
    c["files"][view] = r["file"]
    was_given = view in c.get("provided", [])
    c["provided"] = [p for p in c.get("provided", []) if p != view]
    if before:
        c.setdefault("edits", []).append({"version": version, "target": view, "instruction": f"make the {view} view again (seed {seed})",
                                          "reference": None, "before": str(before), "seconds": r["seconds"],
                                          "mode": "redo", "seed": seed, "settings": {**used, "seed": seed},
                                          "provided_before": was_given})
    return _after_change(name, c, view)


@_locked
def undo_edit(name: str, target: str = "front") -> dict:
    """Undo the last change of one image (an edit, a redo or a pick): the image before it comes back;
    the change stays in the history, marked as undone."""
    c = load(name)
    last = restore_before(c, target, _target_path(c, target) if target in TARGETS else None)
    if last.get("mode") == "pick":
        c["seed"] = last.get("seed_before")
    if last.get("provided_before") and target not in c.get("provided", []):
        c.setdefault("provided", []).append(target)   # the user's own image is back: keep it as theirs
    return _after_change(name, c, target)


def restore_before(c: dict, target: str, img: Path | None, err: type = None) -> dict:
    """The file part of an undo (characters and props): the picture before the last live change of `target` comes
    back into `img`; the picture undone is kept beside the "before" file as <target>_undone_v<N>.png (the app's
    history shows it) and the change is marked undone in c["edits"]. Returns that change (the caller saves c).
    `err`: the error class of a refusal (default CharacterError; props: PropError)."""
    err = err or CharacterError
    live = [e for e in c.get("edits", []) if not e.get("undone") and e.get("target", "front") == target]
    if not live:
        raise err(f"{c['name']}: no change of the {target} image to undo")
    last = live[-1]
    before = Path(last["before"])
    if not before.is_file():
        raise err(f"{c['name']}: the image before change {last['version']} is missing ({before})")
    if img is None:
        raise err(f"cannot undo {target!r}")
    shutil.copyfile(img, before.with_name(f"{target}_undone_v{last['version']}.png"))
    shutil.copyfile(before, img)
    last["undone"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return last


@_locked
def touch_up(name: str, image: np.ndarray, note: str, target: str = "front") -> dict:
    """Save an image changed by code (not by a model) as the new front view or view, with an entry in the
    edit history, so `undo_edit` works the same as for model edits."""
    c = load(name)
    img = _target_path(c, target)
    if not img.is_file():
        raise CharacterError(f"{name}: no {target} image yet")
    version = len(c.get("edits", [])) + 1
    before = _keep_before(c, target, version)
    cv2.imwrite(str(img), image)
    c["files"][target] = str(img)
    c.setdefault("edits", []).append({"version": version, "target": target, "instruction": note, "reference": None,
                                      "before": str(before), "seconds": 0, "mode": "manual"})
    return _after_change(name, c, target)


@_locked
def delete_character(name: str) -> dict:
    """Delete a character for good. Video runs keep their own copy of the sheet, so they are not
    affected. A character with a queued or running job is refused."""
    d = char_dir(name)
    from . import jobs
    busy = [j["id"] for j in jobs.list_jobs(name, limit=50) if j["status"] in ("queued", "running")]
    if busy:
        raise CharacterError(f"{name} has a job that is waiting or running ({busy[0]}); let it finish first")
    own = _skins_of(name)
    if own:
        raise CharacterError(f"{name} has {len(own)} other skin(s) ({', '.join(own)}): delete them first")
    shutil.rmtree(d)
    return {"name": name, "deleted": True}


@_locked
def rename(name: str, new_name: str) -> dict:
    """Give a character a new name (its folder moves; stored paths follow). Video runs keep their own
    copy of the sheet, so they are not affected. Refused while a job of the character waits or runs."""
    if new_name == name:
        return load(name)
    old = char_dir(name)
    new = char_dir(new_name, must_exist=False)
    if new.exists():
        raise CharacterError(f"a character named {new_name!r} exists already")
    from . import jobs
    busy = [j["id"] for j in jobs.list_jobs(name, limit=50) if j["status"] in ("queued", "running")]
    if busy:
        raise CharacterError(f"{name} has a job that is waiting or running ({busy[0]}); rename it after the job")
    with char_lock(new_name):
        c = _rename(name, new_name, old, new)
    for s in _skins_of(name):                                  # its skins follow the main record's new ID
        with char_lock(s):
            r = load(s)
            r["skin_of"] = new_name
            save(r)
    return c


def _rename(name: str, new_name: str, old: Path, new: Path) -> dict:
    os.replace(old, new)
    c = move_paths(load(new_name), old, new)
    c["name"] = new_name
    save(c)
    return c


def move_paths(v: Any, old: Path, new: Path) -> Any:
    """A record with every stored path under the folder `old` moved to `new` (a rename: characters and props)."""
    old_s, new_s = str(old), str(new)
    if isinstance(v, str) and v.startswith(old_s):
        return new_s + v[len(old_s):]
    if isinstance(v, list):
        return [move_paths(x, old, new) for x in v]
    if isinstance(v, dict):
        return {k: move_paths(x, old, new) for k, x in v.items()}
    return v


def _guard_builtin(fn):
    """A change that is not locked (it takes the character's ID first) refuses a built-in character before it writes
    any file (rescale_sheet draws the sheet before it saves)."""
    @functools.wraps(fn)
    def wrapper(name, *args, **kwargs):
        refuse_builtin(name)
        return fn(name, *args, **kwargs)
    return wrapper


for _fn in ("set_style", "link_skin", "unlink_skin", "set_display_name", "set_stand_in", "set_mannequin",
            "rescale_sheet", "set_proxy"):
    globals()[_fn] = _guard_builtin(globals()[_fn])
