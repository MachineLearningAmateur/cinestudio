"""The style library (2026-10-25, user: "we should use the same set of key words for styles across characters, templates,
backgrounds, and objects"): ONE named entry holds the words of a look, and every asset refers to it by name instead of
keeping its own copy of the words.

    python -m cinestudio style list | show NAME | new NAME --label ... --words ... | edit NAME ... | delete NAME
    python -m cinestudio style set-default NAME

A style: {"name", "label", "words" (the look: "stylized cinematic 3D animation, ..."), "light" ("evenly lit by soft
neutral light from the front"), "negative_image" (excluded looks for Qwen / Z-Image jobs), "lora" ({"name", "strength"}
for LTX runs, or null; an optional "trigger" word), "reference" (a picture file in this style, Picture 2 of a style
conversion, or null), "reference_crop" ([x0, y0, x1, y1] in the reference's pixels: the tight crop Qwen gets as
Picture 2, AGENTS.md 4.3; null = the whole picture, which is then itself a tight crop), "reference_source" (the file
the picture was copied from, e.g. a candidate of style_ref), "version" (+1 at each edit),
"updated"}. A reference can be made by the VIDEO model and the style's LoRA from text (cinestudio/style_ref.py). Built-in styles ship in styles/library/NAME.json (read-only); the user's
are in <home>/styles/NAME.json (a user style can take a built-in's name: it then wins).

An asset stores {"style_ref": NAME, "style_version": N}; its words are written into the asset's own text fields when the
style is assigned (so every existing prompt path uses them), and an asset whose version is older than the style's is
reported "made with an older version of this style" (finished pictures never change by themselves).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILTIN = ROOT / "styles" / "library"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
FIELDS = ("label", "words", "light", "negative_image", "lora", "reference", "reference_crop", "reference_source")
REF_EXT = (".png", ".jpg", ".jpeg", ".webp")


class StyleError(ValueError):
    pass


def user_dir() -> Path:
    from .runner import home
    d = home() / "styles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load(name: str) -> dict:
    if not NAME_RE.match(name or ""):
        raise StyleError(f"style name {name!r}: lowercase letters, digits, - and _ (at most 64)")
    for p, builtin in ((user_dir() / f"{name}.json", False), (BUILTIN / f"{name}.json", True)):
        s = _read(p)
        if s:
            return {**_blank(name), **s, "name": name, "builtin": builtin}
    raise StyleError(f"no style {name!r}")


def _blank(name: str) -> dict:
    return {"name": name, "label": name, "words": "", "light": "", "negative_image": "", "lora": None,
            "reference": None, "reference_crop": None, "reference_source": None, "version": 1}


def list_styles() -> list[dict]:
    names = {p.stem for d in (BUILTIN, user_dir()) if d.is_dir() for p in d.glob("*.json")}
    out = []
    for n in sorted(names):
        try:
            out.append(load(n))
        except StyleError:
            continue
    default = default_style()
    for s in out:
        s["default"] = s["name"] == default
    return out


def default_style() -> str | None:
    from . import settings
    name = settings.get().get("default_style")
    if name:
        try:
            load(name)
            return name
        except StyleError:
            pass
    return "3d_animated" if (BUILTIN / "3d_animated.json").is_file() else None


def set_default(name: str) -> dict:
    load(name)
    from . import settings
    settings._set("default_style", name)
    return {"default_style": name}


def _check(data: dict) -> dict:
    out = {}
    for k in FIELDS:
        if k not in data:
            continue
        v = data[k]
        if k == "lora":
            if v in (None, "", {}):
                out[k] = None
                continue
            if not isinstance(v, dict) or not str(v.get("name") or "").strip():
                raise StyleError("lora must be {\"name\": FILE, \"strength\": 0..2} or empty")
            st = float(v.get("strength", 0.4))
            if not 0.0 <= st <= 2.0:
                raise StyleError("lora strength must be between 0 and 2")
            out[k] = {"name": str(v["name"]).strip(), "strength": st}
            trig = str(v.get("trigger") or "").strip()
            if trig:
                if len(trig) > 64:
                    raise StyleError("the LoRA's trigger word is too long (at most 64 characters)")
                out[k]["trigger"] = trig
        elif k == "reference":
            if v in (None, ""):
                out[k] = None
            elif not Path(str(v)).is_file():
                raise StyleError(f"reference picture {v} does not exist")
            else:
                out[k] = str(Path(str(v)).resolve())
        elif k == "reference_crop":
            out[k] = None if v in (None, "", []) else check_crop(v)
        else:
            v = str(v or "").strip()
            if len(v) > 2000:
                raise StyleError(f"{k} is too long (at most 2000 characters)")
            out[k] = v
    return out


def save(name: str, data: dict, new: bool = False) -> dict:
    """Make (new=True) or change a user style; changing a built-in one makes a user copy with that name."""
    if not NAME_RE.match(name or ""):
        raise StyleError(f"style name {name!r}: lowercase letters, digits, - and _ (at most 64)")
    p = user_dir() / f"{name}.json"
    try:
        cur = load(name)
    except StyleError:
        cur = None
    if new and cur:
        raise StyleError(f"a style {name!r} exists already")
    if not new and not cur:
        raise StyleError(f"no style {name!r}")
    rec = {k: v for k, v in (cur or _blank(name)).items() if k not in ("builtin", "default", "name")}
    checked = _check(data)
    if "reference" in checked and checked["reference"] != rec.get("reference"):
        checked.setdefault("reference_crop", None)       # a new picture: the old crop box does not fit it ...
        checked.setdefault("reference_source", None)     # ... and it came from elsewhere
    rec.update(checked)
    if rec.get("reference_crop") and not rec.get("reference"):
        raise StyleError("a crop box needs a reference picture")
    if rec.get("reference_crop") and rec.get("reference"):
        check_crop(rec["reference_crop"], rec["reference"])
    if not rec.get("words"):
        raise StyleError("a style needs its words (the look)")
    rec["version"] = int((cur or {}).get("version", 0)) + 1 if cur else 1
    rec["updated"] = time.time()
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return load(name)


def check_crop(box, picture: str | Path | None = None) -> list[int]:
    """A crop box [x0, y0, x1, y1] (pixels; a list or "x0,y0,x1,y1"), at least 32 px each way; inside `picture` when
    given."""
    if isinstance(box, str):
        box = [p for p in box.replace(" ", "").split(",") if p]
    try:
        x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    except (TypeError, ValueError) as exc:
        raise StyleError("a crop box is four numbers: x0,y0,x1,y1 (pixels)") from exc
    if x0 < 0 or y0 < 0 or x1 - x0 < 32 or y1 - y0 < 32:
        raise StyleError(f"crop box {[x0, y0, x1, y1]}: x0 < x1 and y0 < y1, at least 32 pixels each way")
    if picture is not None:
        import cv2
        im = cv2.imread(str(picture), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise StyleError(f"cannot read the picture {picture}")
        h, w = im.shape[:2]
        if x1 > w or y1 > h:
            raise StyleError(f"crop box {[x0, y0, x1, y1]} is outside the picture ({w}x{h})")
    return [x0, y0, x1, y1]


def refs_dir() -> Path:
    """<home>/styles/refs: the stored reference pictures, NAME.<id>.<ext> (a style name has no ".")."""
    return user_dir() / "refs"


def store_reference(name: str, picture: str | Path) -> Path:
    """A checked copy of a picture as refs/NAME.<id>.<ext> (a new id each time: a failed save never replaces the
    picture in use)."""
    import secrets
    import shutil

    import cv2
    src = Path(picture)
    ext = src.suffix.lower()
    if ext not in REF_EXT:
        raise StyleError(f"unsupported picture type {ext!r} (png, jpg or webp)")
    if cv2.imread(str(src)) is None:
        raise StyleError("the reference file is not a picture that can be read")
    d = refs_dir()
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"{name}.{secrets.token_hex(4)}{ext}"
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return dst


def drop_references(name: str, keep: str | None = None) -> None:
    """Remove this style's stored pictures except the one in use."""
    d = refs_dir()
    if not d.is_dir():
        return
    k = Path(keep).resolve() if keep else None
    for p in d.glob(f"{name}.*"):
        if p.suffix.lower() in REF_EXT and (k is None or p.resolve() != k):
            p.unlink(missing_ok=True)


def delete(name: str) -> dict:
    s = load(name)
    if s["builtin"] and not (user_dir() / f"{name}.json").is_file():
        raise StyleError(f"{name!r} is a built-in style: it cannot be deleted")
    (user_dir() / f"{name}.json").unlink(missing_ok=True)
    return {"deleted": name}


def text(name: str, light: bool = True) -> str:
    """The style's words for a prompt: the look, then the light ("stylized 3D ..., evenly lit by ..."). light=False:
    the look only, for a character sheet, which stays flatly lit whatever the style's scene light (AGENTS.md 4.2)."""
    s = load(name)
    parts = (s.get("words"), s.get("light")) if light else (s.get("words"),)
    return ", ".join(x.strip().rstrip(".") for x in parts if x and x.strip())


def stamp(name: str) -> dict:
    """What an asset stores when the style is assigned to it."""
    s = load(name)
    return {"style_ref": name, "style_version": int(s.get("version", 1))}


def is_stale(asset: dict) -> bool:
    """True when the asset was made with an older version of its style."""
    ref = asset.get("style_ref")
    if not ref:
        return False
    try:
        return int(asset.get("style_version", 0)) < int(load(ref).get("version", 1))
    except StyleError:
        return False


def used_by() -> dict[str, dict[str, list[str]]]:
    """{style: {"characters": [...], "backgrounds": [...]}} from the assets' style_ref (a template has no look of
    its own: the user, 2026-10-25, "the style should be applied by the background and video gen")."""
    out: dict[str, dict[str, list[str]]] = {}

    def add(ref, kind, name):
        if ref:
            out.setdefault(ref, {"characters": [], "backgrounds": []})[kind].append(name)
    try:
        from . import character
        for c in character.list_characters():
            add(c.get("style_ref"), "characters", c.get("name"))
    except Exception:  # noqa: BLE001 - a broken record must not hide the library
        pass
    try:
        from . import background
        for b in background.list_backgrounds():
            add(b.get("style_ref"), "backgrounds", b.get("name"))
    except Exception:  # noqa: BLE001
        pass
    return out
