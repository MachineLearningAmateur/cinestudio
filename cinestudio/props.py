"""Prop library: one folder per prop (an object a character holds, kicks or throws), reusable in every run.

<home>/props/<id>/
  prop.json          name (the ID), display_name, words (used in the prompts), description (the picture prompt's
                     detail), style (the look words), style_ref/style_version, source, status, files, candidates, seed,
                     edits (the history), skin_of (another style of the same prop)
  front.png          the prop alone on pure black, flatly lit (side.png: an optional second view)
  candidates/        pictures made from text (one per seed) + candidates.jpg
  history/           the pictures before each change (front_v<N>.png) and the undone ones (front_undone_v<N>.png)

A prop shares the character tools (cinestudio/character.py: IDs and display names, the ComfyUI image job, the
Qwen-Image-Edit change with region edits and undo, the style stamp, skins), with a simpler record: no views, sheet,
face, mannequin body, repose or proxy kit (the template's Blender object is the prop's stand-in in the layout).
A run uses a prop by its ID (`run new --prop-ref "Ball_*=dodgeball"`): its front becomes the run's prop picture
(`prop_images`: the prop panel of the run's sheet) and its words the prop group's words (`props`). AGENTS.md 4.10.

    python -m cinestudio prop create NAME --words "..." [--style STYLE] [--candidates 4]   # a GPU job (the queue)
    python -m cinestudio prop pick NAME --seed N | import NAME --image FILE --words "..."  # instant
    python -m cinestudio prop edit NAME --instruction "..." [--region x0,y0,x1,y1]       # a GPU job; undo: prop undo
"""
from __future__ import annotations

import contextlib
import functools
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import cv2

from . import character as ch

TARGETS = ("front", "side")                      # pictures of a prop that can be given and edited
MODELS = {"qwen-image": ("qwenimage", "pano"),   # Qwen-Image-2512 through the backplate graph, its 360 LoRA at 0
          "z-image": ("zimage", None)}           # Z-Image (the character fronts' model); variant: recipe.ZIMAGE_VARIANT
DEFAULT_MODEL = "qwen-image"                     # the first prop (the anime noir dodgeball) came from Qwen-Image
SIZE = (1024, 1024)
MAX_CANDIDATES = 8
WORDS_MAX = 300                                  # the words go into every clip prompt: short and concrete
DESCRIPTION_MAX = 1200
# the prompt of a prop picture (from the dodgeball prompt that worked, 2026-10-25; AGENTS.md 4.2 and 4.4: objective
# and concrete, the background as its own sentence, flat front light, no meta words)
PROP_PROMPT = ("One single object shown alone as a prop reference picture: {description}, seen at eye level, centred "
               "in the picture and filling about half of its height.{look} The background is the same solid pure "
               "black everywhere, empty and seamless, without floor, shadow or horizon. The light is flat and even "
               "from the front, soft and neutral.")
PROP_NEGATIVE = ("text, watermark, logo, letters, numbers, floor, table, ground shadow, cast shadow, reflection, white "
                 "background, grey background, gradient background, studio backdrop, room, wall, second object, "
                 "several objects, duplicate object, hand, person, character, cropped object, blurry, low quality")


class PropError(ch.CharacterError):
    pass


# ---------------------------------------------------------------- the library
def library() -> Path:
    from .runner import home
    return home() / "props"


def prop_dir(name: str, must_exist: bool = True) -> Path:
    if not ch.NAME_RE.match(name or ""):
        raise PropError(f"bad prop name {name!r}: use lowercase letters, digits, '-' or '_' (max 48)")
    d = library() / name
    if must_exist and not (d / "prop.json").is_file():
        raise PropError(f"no prop {name!r} in {library()}")
    return d


def check_id(name: str) -> str:
    """A prop ID that exists (a run's prop_refs, the API): the ID or a PropError."""
    prop_dir(str(name or ""))
    return name


@contextlib.contextmanager
def prop_lock(name: str, timeout: float = 0):
    """The lock of one prop (every change holds it; a job waits for a short change). Re-entrant in one thread."""
    prop_dir(name if isinstance(name, str) else "", must_exist=False)     # a bad name: PropError before any path
    try:
        with ch.item_lock(library(), name, timeout):
            yield
    except ch.CharacterError as exc:
        if isinstance(exc, PropError) or "is busy" not in str(exc):
            raise
        raise PropError(str(exc)) from exc


def _locked(fn):
    """Run a prop function under its prop's lock (the first argument is the ID)."""
    @functools.wraps(fn)
    def wrapper(name, *args, **kwargs):
        with prop_lock(name):
            return fn(name, *args, **kwargs)
    return wrapper


def load(name: str) -> dict:
    return json.loads((prop_dir(name) / "prop.json").read_text(encoding="utf-8"))


def save(p: dict) -> None:
    p["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    ch._write_json(prop_dir(p["name"], must_exist=False) / "prop.json", p)


def new_id(display: str) -> str:
    """A free ID for a new prop from its display name ("Dodgeball" -> "dodgeball", then -2, -3 ...)."""
    return ch.free_id(display, library(), fallback="prop")


def display_name(p: dict) -> str:
    return p.get("display_name") or p["name"]


def list_props() -> list[dict]:
    if not library().is_dir():
        return []
    out = []
    for d in sorted(library().iterdir()):
        try:
            p = json.loads((d / "prop.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        row = {k: p.get(k) for k in ("name", "status", "source", "words", "updated", "style_ref", "style_version",
                                     "skin_of")}
        row["display_name"] = display_name(p)
        f = p.get("files") or {}
        cands = p.get("candidates") or [{}]
        row["thumb"] = f.get("front") or cands[0].get("file")
        out.append(row)
    return out


def base_of(p: dict) -> str:
    return ch.base_of(p)


def skins(name: str) -> list[dict]:
    """The prop's family (list rows): the main record first, then its skins (the same prop in other styles)."""
    return ch.family_rows(list_props(), base_of(load(name)))


def _skins_of(base: str) -> list[str]:
    return [r["name"] for r in list_props() if r.get("skin_of") == base]


def _clean_words(words: str, required: bool = True) -> str:
    w = " ".join((words or "").split()).strip().rstrip(".")
    if required and not w:
        raise PropError("say what the prop is in a few concrete words (they go into every video prompt), e.g. "
                        "\"matte deep violet rubber dodgeball with a thin magenta seam\"")
    if len(w) > WORDS_MAX:
        raise PropError(f"the words are longer than {WORDS_MAX} characters: keep them short (the description takes "
                        "the detail)")
    return w


def _clean_description(text: str | None) -> str:
    t = " ".join((text or "").split()).strip()
    if len(t) > DESCRIPTION_MAX:
        raise PropError(f"the description is longer than {DESCRIPTION_MAX} characters")
    return t


def _look(style: str | None, look: str | None, default: bool = True) -> tuple[str, dict]:
    """(the look words, the style stamp) of a new prop: a library style by name (default: the library's default
    style when `default`), or the caller's own words without a stamp."""
    from . import styles
    if (look or "").strip() and not style:
        return look.strip().rstrip("."), {}
    name = style or (styles.default_style() if default else None)
    if not name:
        return "", {}
    try:
        return styles.text(name, light=False), styles.stamp(name)    # a prop picture stays flatly lit, like a sheet
    except styles.StyleError as exc:
        raise PropError(str(exc)) from exc


def _new(name: str, words: str, description: str, style: str | None, look: str | None, source: str,
         display: str) -> dict:
    d = prop_dir(name, must_exist=False)
    if d.exists():
        raise PropError(f"prop {name!r} exists already")
    words_c = _clean_words(words)
    look_w, stamp = _look(style, look, default=source == "text")     # an imported picture has its own look
    d.mkdir(parents=True)
    p = {"name": name, "display_name": ch.clean_display(display) if (display or "").strip() else name,
         "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source, "status": "new", "words": words_c,
         "description": _clean_description(description), "style": look_w, **stamp, "files": {}}
    save(p)
    return p


def prompt_for(p: dict) -> str:
    """The picture prompt of a prop: its description (or its words), its look, the black background and flat light."""
    desc = (p.get("description") or p.get("words") or "").strip().rstrip(".")
    look = (p.get("style") or "").strip().rstrip(".")
    return PROP_PROMPT.format(description=desc, look=f" {look[:1].upper() + look[1:]}." if look else "")


def negative_for(p: dict, extra: str = "") -> str:
    from . import styles
    parts = [PROP_NEGATIVE]
    if p.get("style_ref"):
        try:
            parts.append((styles.load(p["style_ref"]).get("negative_image") or "").strip())
        except styles.StyleError:
            pass
    parts.append((extra or "").strip())
    return ", ".join(x for x in parts if x)


# ---------------------------------------------------------------- text -> candidates (a GPU job: jobs "prop_create")
@_locked
def create(name: str, words: str, description: str = "", style: str | None = None, look: str | None = None,
           candidates: int = 4, seed: int = 202, model: str = DEFAULT_MODEL, display_name: str = "",
           negative: str = "", width: int = SIZE[0], height: int = SIZE[1], **kw) -> dict:
    """Prop picture candidates from text (one per seed: Qwen-Image-2512 by default, or Z-Image), each the prop alone on
    pure black, flatly lit, in the library style `style` (default: the library's default style). Choose one with
    `pick`."""
    if model not in MODELS:
        raise PropError(f"model must be one of {sorted(MODELS)}")
    if not 1 <= int(candidates) <= MAX_CANDIDATES:
        raise PropError(f"candidates must be between 1 and {MAX_CANDIDATES}")
    p = _new(name, words, description, style, look, "text", display_name)
    d = prop_dir(name)
    prompt, neg = prompt_for(p), negative_for(p, negative)
    profile, variant = MODELS[model]
    if variant is None:
        from . import recipe
        variant = recipe.ZIMAGE_VARIANT
    values = {"prompt": prompt, "negative": neg, "width": int(width), "height": int(height)}
    if profile == "qwenimage":
        values["lora_strength"] = 0.0                     # the backplate recipe: no 360 panorama look
    made = []
    for i, s in enumerate(range(int(seed), int(seed) + int(candidates)), 1):
        r = ch.image_job(profile, variant, dict(values, seed=s), {}, d / "candidates" / f"front_s{s}.png",
                         f"prop_{name}_s{s}", item=(i, int(candidates)), **kw)
        made.append({"seed": s, "file": r["file"]})
        if kw.get("on_event"):
            kw["on_event"]({"event": "candidate", "seed": s, "file": r["file"], "seconds": r["seconds"]})
    ch._contact(d / "candidates", [m["file"] for m in made], [f"seed {m['seed']}" for m in made])
    p = load(name)
    p.update(status="candidates", prompt=prompt, negative=neg, model=model, candidates=made, size=[width, height])
    save(p)
    return p


@_locked
def pick(name: str, seed: int) -> dict:
    """Use one candidate as the prop's front (instant; an earlier front stays in the history: undo brings it back)."""
    p = load(name)
    hit = [x for x in p.get("candidates", []) if int(x["seed"]) == int(seed)]
    if not hit:
        raise PropError(f"{name}: no candidate with seed {seed} (candidates: {[x['seed'] for x in p.get('candidates', [])]})")
    d = prop_dir(name)
    front = d / "front.png"
    if front.is_file():
        version = len(p.get("edits", [])) + 1
        before = _keep_before(p, "front", version)
        p.setdefault("edits", []).append({"version": version, "target": "front", "instruction": f"use candidate seed {seed}",
                                          "reference": None, "before": str(before), "seconds": 0, "mode": "pick",
                                          "seed_before": p.get("seed")})
    shutil.copyfile(hit[0]["file"], front)
    p["files"]["front"] = str(front)
    p.update(status="ready", seed=int(seed))
    save(p)
    return p


def _read_picture(path: str | Path, what: str):
    img = cv2.imread(str(path))
    if img is None:
        raise PropError(f"cannot read the {what} picture {path}")
    return img


def import_picture(name: str, image: str | Path, words: str, description: str = "", style: str | None = None,
                   look: str | None = None, side: str | Path | None = None, candidates: list | None = None,
                   display_name: str = "") -> dict:
    """A prop from the user's own picture (instant): `image` = the front (best: the prop alone on black, in the
    characters' style), `side` an optional second view, `candidates` more pictures to choose from later (pick).
    `style` names the library style the picture is drawn in (its stamp; default: none = the picture's own look)."""
    front = _read_picture(image, "front")
    side_img = _read_picture(side, "side") if side else None
    extra = [(Path(c), _read_picture(c, "candidate")) for c in (candidates or [])]
    with prop_lock(name):
        p = _new(name, words, description, style, look, "image", display_name)
        d = prop_dir(name)
        cv2.imwrite(str(d / "front.png"), front)               # always PNG, whatever the input format
        p["files"]["front"] = str(d / "front.png")
        if side_img is not None:
            cv2.imwrite(str(d / "side.png"), side_img)
            p["files"]["side"] = str(d / "side.png")
        if extra:
            (d / "candidates").mkdir(exist_ok=True)
            rows = []
            for i, (src, img) in enumerate([(Path(image), front), *extra], 1):
                seed = _seed_in(src.stem) or i
                while any(r["seed"] == seed for r in rows):
                    seed += 1000
                dst = d / "candidates" / f"front_s{seed}.png"
                cv2.imwrite(str(dst), img)
                rows.append({"seed": seed, "file": str(dst)})
            ch._contact(d / "candidates", [r["file"] for r in rows], [f"seed {r['seed']}" for r in rows])
            p["candidates"] = rows
            p["seed"] = rows[0]["seed"]
        p["status"] = "ready"
        save(p)
        return p


def _seed_in(stem: str) -> int | None:
    m = re.search(r"_s(\d+)$", stem)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- edits, undo, history
def _target_path(p: dict, target: str) -> Path:
    if target not in TARGETS:
        raise PropError(f"cannot edit {target!r}; use one of {list(TARGETS)}")
    return prop_dir(p["name"]) / f"{target}.png"


def _keep_before(p: dict, target: str, version: int) -> Path:
    dst = prop_dir(p["name"]) / "history" / f"{target}_v{version}.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_target_path(p, target), dst)
    return dst


@_locked
def edit(name: str, instruction: str, target: str = "front", reference: str | Path | None = None,
         mode: str = "quality", seed: int | None = None, settings: dict | None = None,
         region: str | tuple | None = None, **kw) -> dict:
    """Change one picture of the prop by an instruction (Qwen-Image-Edit-2511; Picture 1 = the prop, Picture 2 = an
    optional reference showing ONE detail). One change per edit; `region` "x0,y0,x1,y1" edits only that box. Every
    edit can be undone (undo). The same core as a character edit (character.edit_picture)."""
    if not (instruction or "").strip():
        raise PropError("say the one change (end with: Keep everything else exactly the same.)")
    p = load(name)
    img = _target_path(p, target)
    if not img.is_file():
        raise PropError(f"{name}: no {target} picture yet")
    version = len(p.get("edits", [])) + 1
    kept: dict = {}
    r = ch.edit_picture(img, instruction, prop_dir(name), f"prop_{name}_{target}_edit{version}", reference, mode, seed,
                        settings, region, keep=lambda: kept.setdefault("before", _keep_before(p, target, version)), **kw)
    p = load(name)                                            # reread: only this change is applied
    p["files"][target] = str(img)
    p.setdefault("edits", []).append({"version": version, "target": target, "instruction": instruction,
                                      "reference": str(reference) if reference else None,
                                      "before": str(kept["before"]), "seconds": r["seconds"], "mode": mode,
                                      "settings": r["used"], **({"region": list(r["box"])} if r["box"] else {})})
    save(p)
    return p


@_locked
def undo(name: str, target: str = "front") -> dict:
    """Undo the last change of one picture (an edit, a pick or a touch-up; instant): the picture before it comes back;
    the change stays in the history, marked as undone."""
    p = load(name)
    last = ch.restore_before(p, target, _target_path(p, target) if target in TARGETS else None, PropError)
    if last.get("mode") == "pick":
        p["seed"] = last.get("seed_before")
    save(p)
    return p


@_locked
def touch_up(name: str, image, note: str, target: str = "front") -> dict:
    """Save a picture changed by code (not by a model), undoable like an edit."""
    p = load(name)
    img = _target_path(p, target)
    if not img.is_file():
        raise PropError(f"{name}: no {target} picture yet")
    version = len(p.get("edits", [])) + 1
    before = _keep_before(p, target, version)
    cv2.imwrite(str(img), image)
    p["files"][target] = str(img)
    p.setdefault("edits", []).append({"version": version, "target": target, "instruction": note, "reference": None,
                                      "before": str(before), "seconds": 0, "mode": "manual"})
    save(p)
    return p


# ---------------------------------------------------------------- words, names, style, delete
@_locked
def set_words(name: str, words: str | None = None, description: str | None = None) -> dict:
    """The prop's words (the video prompts and the run sheet's "**Prop:**" line) and/or its picture description
    (instant; finished pictures do not change)."""
    p = load(name)
    if words is not None:
        p["words"] = _clean_words(words)
    if description is not None:
        p["description"] = _clean_description(description)
    save(p)
    return p


@_locked
def set_display_name(name: str, text: str) -> dict:
    p = load(name)
    p["display_name"] = ch.clean_display(text)
    save(p)
    return p


@_locked
def set_style(name: str, style: str) -> dict:
    """Name the library style the prop's pictures are drawn in (instant: its look words and stamp; the pictures stay)."""
    from . import styles
    try:
        words, stamp = styles.text(style, light=False), styles.stamp(style)
    except styles.StyleError as exc:
        raise PropError(str(exc)) from exc
    p = load(name)
    p["style"] = words
    p.update(stamp)
    save(p)
    return p


def _busy(name: str) -> list[str]:
    from . import jobs
    return [j["id"] for j in jobs.list_jobs(None, limit=60, prop_name=name) if j["status"] in jobs.ACTIVE]


@_locked
def delete(name: str) -> dict:
    """Delete a prop for good. Runs keep their own copy of its picture. Refused while a job of the prop waits or runs,
    and for a main record that still has skins."""
    d = prop_dir(name)
    busy = _busy(name)
    if busy:
        raise PropError(f"{name} has a job that is waiting or running ({busy[0]}); let it finish first")
    own = _skins_of(name)
    if own:
        raise PropError(f"{name} has {len(own)} other skin(s) ({', '.join(own)}): delete them first")
    shutil.rmtree(d)
    return {"name": name, "deleted": True}


@_locked
def rename(name: str, new_name: str) -> dict:
    """Give a prop a new ID (its folder moves; stored paths follow; its skins follow). Runs keep their own copies."""
    if new_name == name:
        return load(name)
    old, new = prop_dir(name), prop_dir(new_name, must_exist=False)
    if new.exists():
        raise PropError(f"a prop named {new_name!r} exists already")
    busy = _busy(name)
    if busy:
        raise PropError(f"{name} has a job that is waiting or running ({busy[0]}); rename it after the job")
    with prop_lock(new_name):
        os.replace(old, new)
        p = ch.move_paths(load(new_name), old, new)
        p["name"] = new_name
        save(p)
    for s in _skins_of(name):
        with prop_lock(s):
            r = load(s)
            r["skin_of"] = new_name
            save(r)
    return p


# ---------------------------------------------------------------- skins: the same prop in another library style
def skin_instruction(p: dict, style: dict, with_reference: bool) -> str:
    look = (style.get("words") or "").strip().rstrip(".")
    head = (f"Redraw the object of Picture 1 in the rendering style of Picture 2: {look}." if with_reference
            else f"Redraw the object of Picture 1 in this rendering style: {look}.")
    design = (p.get("description") or p.get("words") or "").strip().rstrip(".")
    return (f"{head} The background is pure black everywhere, empty, without floor, shadow or horizon, exactly as in "
            f"Picture 1. The object's design stays exactly the same: {design}. Keep its shape, its colours, its "
            f"proportions, its size in the picture and every part of it exactly the same.")


def new_skin(name: str, style: str, seed: int | None = None, mode: str = "quality", **kw) -> dict:
    """A new skin of the prop in a library style (a GPU job, jobs "prop_skin"): a new record (ID "<main>-<style>") with
    the prop's words; its front = this skin's front redrawn by Qwen-Image-Edit in the style (the style's reference
    crop as Picture 2 when it has one). Look at it before a run."""
    from . import style_ref, styles
    src = load(name)
    try:
        st = styles.load(style)
    except styles.StyleError as exc:
        raise PropError(str(exc)) from exc
    base = base_of(src)
    if any(r.get("style_ref") == style for r in skins(base)):
        raise PropError(f"{display_name(load(base))} has a skin in {st.get('label') or style} already")
    front = Path((src.get("files") or {}).get("front") or "")
    if not front.is_file():
        raise PropError(f"{name} has no front picture to redraw")
    new = new_id(f"{base} {style}")
    d = prop_dir(new, must_exist=False)
    with prop_lock(new):
        d.mkdir(parents=True)
        shutil.copyfile(front, d / "front.png")
        p = {k: src[k] for k in ("words", "description", "display_name") if k in src}
        p.update(name=new, skin_of=base, copied_from=name, source="skin", status="new", files={"front": str(d / "front.png")},
                 created=time.strftime("%Y-%m-%dT%H:%M:%S"), style=styles.text(style, light=False), **styles.stamp(style))
        save(p)
        ref = style_ref.reference_for_qwen(style)
        neg = ch.skin_negative(src, st, base=PROP_NEGATIVE)
        p = edit(new, skin_instruction(src, st, bool(ref)), target="front", reference=ref, mode=mode, seed=seed,
                 settings={"negative": neg}, **kw)
        p["status"] = "ready"
        save(p)
    return {**p, "skin": new}


# ---------------------------------------------------------------- runs: prop_refs {pattern: prop ID}
def resolve_refs(refs: dict | None) -> dict:
    """A run's library props: {pattern: prop ID} -> {"props": {pattern: the prop's words}, "prop_images": {pattern:
    its front picture}} (runner.new_run's own options; the pattern is checked against the template there)."""
    if not refs:
        return {}
    if not isinstance(refs, dict):
        raise PropError("prop_refs must map a template prop pattern (e.g. Ball_*) to a prop ID")
    words, pics = {}, {}
    for pat, pid in refs.items():
        pat, pid = str(pat).strip(), str(pid or "").strip()
        if not pat:
            raise PropError("prop_refs: a pattern is empty")
        if not pid:
            continue
        p = load(check_id(pid))
        front = Path((p.get("files") or {}).get("front") or "")
        if not front.is_file():
            raise PropError(f"prop {pid!r} has no picture yet (pick a candidate first)")
        words[pat] = p.get("words") or display_name(p)
        pics[pat] = str(front)
    return {**({"props": words} if words else {}), **({"prop_images": pics} if pics else {})}


def merge_refs(refs: dict | None, props: dict | None, prop_images: dict | None) -> tuple[dict, dict]:
    """The run's prop words and pictures with the library props of `refs` filled in; the caller's own words or own
    picture for the same pattern win (the "own words / own picture" fallback). -> (props, prop_images)."""
    r = resolve_refs(refs)
    words = {**r.get("props", {}), **{k: v for k, v in (props or {}).items() if str(v or "").strip()}}
    pics = {**r.get("prop_images", {}), **{k: v for k, v in (prop_images or {}).items() if str(v or "").strip()}}
    return words, pics


def parse_ref_args(items: list[str] | None) -> dict:
    """--prop-ref "PATTERN=PROP_ID" (repeatable) -> {pattern: prop ID} (the CLI)."""
    out = {}
    for it in items or []:
        pat, sep, pid = str(it).partition("=")
        if not sep or not pat.strip() or not pid.strip():
            raise PropError(f"--prop-ref {it!r}: write PATTERN=PROP_ID, e.g. --prop-ref \"Ball_*=dodgeball\"")
        out[pat.strip()] = check_id(pid.strip())
    return out


ACTIONS: dict[str, Callable[..., Any]] = {"prop_create": create, "prop_edit": edit, "prop_skin": new_skin}
