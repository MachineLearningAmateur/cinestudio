"""Props: the prop library (the Cast tab's Props switch), new props, edits, undo, words, skins and their jobs."""
from __future__ import annotations

from pathlib import Path

from aiohttp import web

from .. import character, jobs, props, runner
from .common import _bg, _form, _history, _inside, _int_or_none, _ok, file_url, routes


def _prop_view(p: dict) -> dict:
    d = props.prop_dir(p["name"])
    out = dict(p)
    out["display_name"] = props.display_name(p)
    out["urls"] = {k: file_url(v) for k, v in (p.get("files") or {}).items()}
    out["candidates"] = [{**x, "url": file_url(x["file"])} for x in p.get("candidates", [])]
    out["edits"] = _history(p)
    out["jobs"] = jobs.list_jobs(None, limit=8, prop_name=p["name"])
    out["contact_url"] = file_url(d / "candidates" / "candidates.jpg")
    out["missing"] = [] if (p.get("files") or {}).get("front") else ["front"]
    try:                                                     # the skin menu: the skins of this prop (main first)
        out["skins"] = [{k: r.get(k) for k in ("name", "status", "style_ref", "style_version", "skin_of",
                                               "display_name")} | {"missing": [], "thumb_url": file_url(r.get("thumb"))}
                        for r in props.skins(p["name"])]
    except Exception:  # noqa: BLE001 - a broken sibling record must not hide this prop
        out["skins"] = []
    out["skin_jobs"] = _skin_jobs({s["name"] for s in out["skins"]} | {p["name"]})
    return out


def _skin_jobs(family: set[str]) -> list[dict]:
    """The new skins being made for this prop family (queued or running prop_skin jobs)."""
    out = []
    for row in jobs.list_jobs(None, limit=40):
        if row.get("action") != "prop_skin" or row.get("status") not in jobs.ACTIVE or row.get("prop") not in family:
            continue
        try:
            style = (jobs.load(row["id"]).get("args") or {}).get("style")
        except (OSError, ValueError, KeyError):
            style = None
        out.append({"id": row["id"], "status": row["status"], "character": row["prop"], "prop": row["prop"],
                    "style": style})
    return out


@routes.get("/api/props")
async def props_list(request: web.Request):
    rows = props.list_props()
    for r in rows:
        r["thumb_url"] = file_url(r.pop("thumb", None))
    return _ok(rows)


@routes.get("/api/props/{name}")
async def prop_show(request: web.Request):
    return _ok(_prop_view(props.load(request.match_info["name"])))


@routes.post("/api/props")
async def prop_new(request: web.Request):
    """New prop: source=text (a job: candidates from the words in a library style) or image (instant)."""
    fields, files = await _form(request)
    return _ok(await _bg(_prop_new, request.app, fields, files))


def _prop_new(app, fields: dict, files: dict) -> dict:
    shown = character.clean_display(fields.get("name", ""))   # the form's name is the DISPLAY name; the ID comes from it
    name = props.new_id(shown)
    words, desc = fields.get("words", ""), fields.get("description", "")
    style = (fields.get("style") or "").strip() or None
    if (fields.get("source") or "text") == "text":
        props._clean_words(words)                              # refuse empty words now, not in the job
        n = int(fields.get("candidates") or 4)
        if not 1 <= n <= props.MAX_CANDIDATES:
            raise ValueError(f"candidates must be between 1 and {props.MAX_CANDIDATES}")
        model = fields.get("model") or props.DEFAULT_MODEL
        if model not in props.MODELS:
            raise ValueError(f"model must be one of {sorted(props.MODELS)}")
        if style:
            from .. import styles
            styles.load(style)                                 # an unknown style is refused now
        seed = _int_or_none(fields.get("seed"))
        job = app["jobs"].submit("prop_create", name, words=words, description=desc, style=style, candidates=n,
                                 seed=202 if seed is None else seed, model=model, display_name=shown)
        return {"job": job, "name": name}
    if "image" not in files:
        raise ValueError("a picture of the prop is needed")
    p = props.import_picture(name, files["image"], words, desc, style, side=files.get("side"), display_name=shown)
    return {"prop": _prop_view(p)}


@routes.post("/api/props/{name}/{action}")
async def prop_action(request: web.Request):
    name, action = request.match_info["name"], request.match_info["action"]
    fields, files = await _form(request)
    return _ok(await _bg(_prop_action, request.app, name, action, fields, files))


def _prop_action(app, name: str, action: str, fields: dict, files: dict) -> dict:
    j = app["jobs"]
    props.check_id(name)
    if action == "delete":
        return props.delete(name)
    if action == "pick":
        return {"prop": _prop_view(props.pick(name, int(fields["seed"])))}
    if action == "undo":
        return {"prop": _prop_view(props.undo(name, fields.get("target") or "front"))}
    if action == "rename":                          # the app renames the DISPLAY name only (the ID stays)
        return {"prop": _prop_view(props.set_display_name(name, fields.get("new_name", "")))}
    if action == "words":
        return {"prop": _prop_view(props.set_words(name, fields.get("words"), fields.get("description")))}
    if action == "use-style":
        return {"prop": _prop_view(props.set_style(name, fields.get("style") or ""))}
    if action == "new-skin":
        if not fields.get("style"):
            raise ValueError("choose a style for the new skin")
        return {"job": j.submit("prop_skin", name, style=fields["style"], seed=_int_or_none(fields.get("seed")))}
    if action == "edit":
        settings = {k: fields[k] for k in character.EDIT_SETTINGS if fields.get(k) not in (None, "")}
        mode = fields.get("mode") or "quality"
        character.quality_values(mode, settings, character.EDIT_SETTINGS)        # refuse bad values now
        ref = files.get("reference") or (fields.get("reference_path") or None)
        if ref and not files.get("reference") and not _inside(Path(ref), props.library(), runner.home() / "uploads"):
            raise ValueError("reference_path must be a file in the prop library or the uploads folder")
        target = fields.get("target") or "front"
        if target not in props.TARGETS:
            raise ValueError(f"target must be one of {list(props.TARGETS)}")
        if not (fields.get("instruction") or "").strip():
            raise ValueError("say the one change")
        return {"job": j.submit("prop_edit", name, instruction=fields["instruction"], reference=ref, target=target,
                                region=fields.get("region") or None, mode=mode, seed=_int_or_none(fields.get("seed")),
                                settings=settings)}
    raise ValueError(f"unknown action {action!r}")
