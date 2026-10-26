"""The Styles tab: the style library (cinestudio/styles.py) - list with use and staleness, new, edit, delete, default,
a reference picture per style, and the style of a character or a template.

A style's reference picture is kept as <home>/styles/refs/NAME.<id>.<ext> (a style name has no ".", so NAME.* is only
this style's files); a new picture gets a new id, so a failed save never replaces the picture in use."""
from __future__ import annotations

import json
from pathlib import Path

from aiohttp import web

from .. import character, shots, styles
from .common import _bg, _form, _ok, file_url, routes

TEXT_FIELDS = ("label", "words", "light", "negative_image")
FORM_FIELDS = {"name", *TEXT_FIELDS, "lora", "lora_name", "lora_strength", "remove_reference", "copy_reference_from"}
REF_EXT = styles.REF_EXT
KINDS = ("characters", "backgrounds")


def refs_dir() -> Path:
    return styles.refs_dir()


def _same_words(a: str, b: str) -> bool:
    return " ".join((a or "").split()).rstrip(".") == " ".join((b or "").split()).rstrip(".")


def preset_or_words(preset: str | None, words: str | None) -> tuple[str | None, str]:
    """The style of a form that has a style select AND a style text (the select fills the text): the library style when
    the text is empty or still its words (then the words are sent empty: the style gives them); the user's own words when
    they changed the text (the style is then not used)."""
    preset, words = (preset or "").strip() or None, words or ""
    if not preset:
        return None, words
    text = styles.text(preset)                           # refuses an unknown style (StyleError -> 400)
    if words.strip() and not _same_words(words, text):
        return None, words
    return preset, ""


def _empty() -> dict:
    return {k: [] for k in KINDS}


def _stale() -> dict[str, dict[str, list[str]]]:
    """{style: {kind: [names made with an older version]}}; templates keep no version (never stale)."""
    out: dict[str, dict[str, list[str]]] = {}
    try:
        for c in character.list_characters():
            if styles.is_stale(c):
                out.setdefault(c["style_ref"], _empty())["characters"].append(c["name"])
    except Exception:  # noqa: BLE001 - a broken record must not hide the library
        pass
    try:
        from .. import background
        for b in background.list_backgrounds():
            if styles.is_stale(b):
                out.setdefault(b["style_ref"], _empty())["backgrounds"].append(b["name"])
    except Exception:  # noqa: BLE001
        pass
    return out


def _view(s: dict, used: dict, stale: dict) -> dict:
    name = s["name"]
    out = {k: s.get(k) for k in ("name", "label", "words", "light", "negative_image", "lora", "reference", "version",
                                 "updated", "builtin", "default")}
    out["has_builtin"] = (styles.BUILTIN / f"{name}.json").is_file()   # builtin False + has_builtin = the user's copy
    out["text"] = ", ".join(x.strip().rstrip(".") for x in (s.get("words"), s.get("light")) if x and x.strip())
    out["reference_url"] = file_url(s.get("reference"))
    out["reference_crop"] = s.get("reference_crop")
    out["reference_source"] = s.get("reference_source")
    out["reference_crop_url"] = None
    if s.get("reference_crop"):
        try:                                              # the crop Qwen gets as Picture 2
            from .. import style_ref
            out["reference_crop_url"] = file_url(style_ref.reference_for_qwen(name))
        except Exception:  # noqa: BLE001 - a broken picture must not hide the library
            pass
    out["used_by"] = {k: list((used.get(name) or {}).get(k) or []) for k in KINDS}
    out["stale"] = {k: list((stale.get(name) or {}).get(k) or []) for k in KINDS}
    return out


def _list() -> dict:
    used, stale = styles.used_by(), _stale()
    return {"default": styles.default_style(), "styles": [_view(s, used, stale) for s in styles.list_styles()]}


def _one(name: str) -> dict:
    s = styles.load(name)
    s["default"] = s["name"] == styles.default_style()
    return _view(s, styles.used_by(), _stale())


def _flag(v) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _data(fields: dict, new: bool, name: str) -> dict:
    """The style fields of a request (JSON or a multipart form), checked; unknown fields are refused."""
    unknown = set(fields) - FORM_FIELDS
    if unknown:
        raise styles.StyleError(f"unknown fields: {', '.join(sorted(unknown))}")
    data = {}
    for k in TEXT_FIELDS:
        if k in fields:
            if fields[k] is not None and not isinstance(fields[k], str):
                raise styles.StyleError(f"{k} must be text")
            data[k] = (fields[k] or "").strip()
    if "label" in data and not data["label"]:           # no label: the name shows
        if new:
            data.pop("label")
        else:
            data["label"] = name
    if "lora" in fields:                                  # JSON: {"name", "strength"} or null
        lora = fields["lora"]
        if isinstance(lora, str):
            try:
                lora = json.loads(lora) if lora.strip() else None
            except ValueError as exc:
                raise styles.StyleError("lora must be {\"name\": FILE, \"strength\": 0..2} or empty") from exc
        data["lora"] = lora
    elif "lora_name" in fields or "lora_strength" in fields:   # a form: two fields
        lname = str(fields.get("lora_name") or "").strip()
        if lname:
            try:
                strength = float(fields.get("lora_strength") or 0.4)
            except (TypeError, ValueError) as exc:
                raise styles.StyleError("lora strength must be a number between 0 and 2") from exc
            data["lora"] = {"name": lname, "strength": strength}
        else:
            data["lora"] = None
    if "lora" in data and data["lora"] is not None:
        if not isinstance(data["lora"], dict):
            raise styles.StyleError("lora must be {\"name\": FILE, \"strength\": 0..2} or empty")
        try:
            data["lora"] = {**data["lora"], "strength": float(data["lora"].get("strength", 0.4))}
        except (TypeError, ValueError) as exc:
            raise styles.StyleError("lora strength must be a number between 0 and 2") from exc
    styles._check(data)                                   # refuse bad values before a picture is stored
    return data


_store_ref = styles.store_reference          # the storage lives in styles.py (style_ref.use_reference uses it too)
_drop_refs = styles.drop_references


def _save(name: str, fields: dict, upload: Path | None, new: bool) -> dict:
    if not styles.NAME_RE.match(name or ""):
        raise styles.StyleError(f"style name {name!r}: lowercase letters, digits, - and _ (at most 64)")
    exists = True
    try:
        styles.load(name)
    except styles.StyleError:
        exists = False
    if new and exists:
        raise styles.StyleError(f"a style {name!r} exists already")
    if not new and not exists:
        raise styles.StyleError(f"no style {name!r}")
    data = _data(fields, new, name)
    if new and not data.get("words"):
        raise styles.StyleError("a style needs its words (the look)")
    if "words" in data and not data["words"]:
        raise styles.StyleError("a style needs its words (the look)")
    src = None
    if upload is not None:
        src = upload
    elif new and fields.get("copy_reference_from"):       # Duplicate: the new style gets its own copy of the picture
        ref = styles.load(str(fields["copy_reference_from"])).get("reference")
        if ref and Path(ref).is_file():
            src = Path(ref)
    stored = None
    if src is not None:
        stored = _store_ref(name, src)
        data["reference"] = str(stored)
    elif _flag(fields.get("remove_reference")):
        data["reference"] = None
    try:
        s = styles.save(name, data, new=new)
    except BaseException:
        if stored:
            stored.unlink(missing_ok=True)
        raise
    if "reference" in data:
        _drop_refs(name, keep=s.get("reference"))
    return _one(name)


def _delete(name: str) -> dict:
    s = styles.load(name)
    builtin_copy = not s["builtin"] and (styles.BUILTIN / f"{name}.json").is_file()
    r = styles.delete(name)
    _drop_refs(name)
    if builtin_copy:
        r["back_to_builtin"] = True
    return r


def assign_character(name: str, style) -> dict:
    """Give a character a library style ("" or "none": no library style; its own style words stay)."""
    if style is not None and not isinstance(style, str):
        raise character.CharacterError("style must be a style name, or empty for none")
    style = (style or "").strip()
    if style and style != "none":
        return character.set_style(name, style)
    with character.char_lock(name):
        c = character.load(name)
        c.pop("style_ref", None)
        c.pop("style_version", None)
        character.save(c)
    return c


async def _fields(request: web.Request) -> tuple[dict, dict[str, Path]]:
    fields, files = await _form(request)
    try:
        if not isinstance(fields, dict):
            raise styles.StyleError("send the style as a JSON object or a form")
        if set(files) - {"reference"}:
            raise styles.StyleError(f"unexpected file fields: {', '.join(sorted(set(files) - {'reference'}))}")
    except BaseException:
        for f in files.values():
            f.unlink(missing_ok=True)
        raise
    return fields, files


@routes.get("/api/styles")
async def styles_list(request: web.Request):
    """The style library: each style with its use (characters, templates, backgrounds) and the assets made with an
    older version of it ("stale")."""
    return _ok(await _bg(_list))


@routes.post("/api/styles/new")
async def style_new(request: web.Request):
    """A new user style (JSON, or multipart with an optional "reference" picture)."""
    fields, files = await _fields(request)
    try:
        return _ok({"style": await _bg(_save, str(fields.get("name") or "").strip(), fields, files.get("reference"),
                                       True)})
    finally:
        for f in files.values():
            f.unlink(missing_ok=True)


@routes.post("/api/styles/{name}/edit")
async def style_edit(request: web.Request):
    """Change a style (version +1); a built-in one becomes the user's own copy. Multipart may carry a new "reference"
    picture; remove_reference=1 takes it away."""
    fields, files = await _fields(request)
    try:
        return _ok({"style": await _bg(_save, request.match_info["name"], fields, files.get("reference"), False)})
    finally:
        for f in files.values():
            f.unlink(missing_ok=True)


@routes.post("/api/styles/{name}/reference")
async def style_reference(request: web.Request):
    """Only the reference picture: multipart "reference" (or "image") sets it, remove=1 takes it away."""
    fields, files = await _form(request)
    try:
        if set(files) - {"reference", "image"} or len(files) > 1:
            raise styles.StyleError("send one picture in the field \"reference\"")
        up = files.get("reference") or files.get("image")
        if up is None and not _flag(fields.get("remove")):
            raise styles.StyleError("choose a picture (or remove=1)")
        f = {} if up is not None else {"remove_reference": "1"}
        return _ok({"style": await _bg(_save, request.match_info["name"], f, up, False)})
    finally:
        for p in files.values():
            p.unlink(missing_ok=True)


@routes.post("/api/styles/{name}/delete")
async def style_delete(request: web.Request):
    """Delete a user style; for the user's copy of a built-in style, the built-in comes back."""
    return _ok(await _bg(_delete, request.match_info["name"]))


@routes.post("/api/styles/{name}/default")
async def style_default(request: web.Request):
    return _ok(await _bg(styles.set_default, request.match_info["name"]))



# ---------------------------------------------------------------- a reference picture made by the video model
def _candidates(name: str) -> dict:
    from .. import style_ref
    styles.load(name)
    rows = [{k: r.get(k) for k in ("file", "seed", "frame", "sharpness", "subject", "made")} | {"url": file_url(r["file"])}
            for r in style_ref.candidates(name)]
    return {"candidates": rows, "job": style_ref.job_status(name), "subject": style_ref.SUBJECT,
            "seeds": list(style_ref.SEEDS)}


def _make_reference(name: str, fields: dict) -> dict:
    from .. import style_ref
    if not isinstance(fields, dict):
        raise styles.StyleError("send a JSON object")
    unknown = set(fields) - {"subject", "seeds"}
    if unknown:
        raise styles.StyleError(f"unknown fields: {', '.join(sorted(unknown))}")
    subject = fields.get("subject")
    if subject is not None and not isinstance(subject, str):
        raise styles.StyleError("subject must be text")
    return style_ref.submit(name, subject, fields.get("seeds"), via="app")


def _use_reference(name: str, fields: dict) -> dict:
    """Only a candidate of this style (a file of its candidates folder) can be chosen here."""
    from .. import style_ref
    if not isinstance(fields, dict):
        raise styles.StyleError("send a JSON object")
    f = Path(str(fields.get("file") or ""))
    folder = style_ref.candidates_dir(name).resolve()
    try:
        f.resolve().relative_to(folder)
    except ValueError as exc:
        raise styles.StyleError("choose one of this style's candidates") from exc
    r = style_ref.use_reference(name, f, fields.get("crop"))
    return {"style": _one(name), "crop": r["crop"], "crop_url": file_url(r["crop_file"])}


@routes.get("/api/styles/{name}/reference-candidates")
async def style_reference_candidates(request: web.Request):
    """The reference candidates made by the video model (newest first) and the state of the style's reference job."""
    return _ok(await _bg(_candidates, request.match_info["name"]))


@routes.post("/api/styles/{name}/make-reference")
async def style_make_reference(request: web.Request):
    """Queue reference candidates made by LTX-2.5 + the style's LoRA from text: {"subject": TEXT, "seeds": [N, ...]}
    (both optional). Returns the queue ref at once."""
    fields, _ = await _form(request)
    return _ok(await _bg(_make_reference, request.match_info["name"], fields))


@routes.post("/api/styles/{name}/use-reference")
async def style_use_reference(request: web.Request):
    """Make one candidate the style's reference: {"file": PATH, "crop": [x0, y0, x1, y1] (optional)}."""
    fields, _ = await _form(request)
    return _ok(await _bg(_use_reference, request.match_info["name"], fields))
