"""Characters: the library, new characters, edits, views, sheets and their background jobs."""
from __future__ import annotations

from pathlib import Path

from aiohttp import web

from .. import character, jobs, runner
from .common import _SAMPLERS, _bg, _char_view, _form, _inside, _int_or_none, _ok, file_url, routes


def _edit_modes() -> dict:
    """The edit modes the app offers: "fast" (the Lightning LoRA, a distilled shortcut) only when the recipe allows it."""
    from .. import recipe
    return {k: {x: y for x, y in v.items() if x != "fast"} for k, v in character.QUALITY.items()
            if k != "fast" or recipe.FAST_EDITS}


@routes.get("/api/image-settings")
async def image_settings(request: web.Request):
    """Defaults of the quality and fast modes, and the samplers and schedulers that ComfyUI has."""
    if not _SAMPLERS:
        import asyncio
        try:
            from .. import server
            info = await asyncio.get_running_loop().run_in_executor(
                None, lambda: server._get(f"{server.comfyui_url()}/object_info/KSampler", timeout=10))
            req = info["KSampler"]["input"]["required"]
            _SAMPLERS.update(samplers=req["sampler_name"][0], schedulers=req["scheduler"][0])
        except Exception:  # noqa: BLE001 - ComfyUI down: the common ones
            return _ok({"modes": _edit_modes(), "defaults": character.GRAPH_DEFAULTS,
                        "samplers": ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "res_multistep", "uni_pc", "ddim"],
                        "schedulers": ["simple", "normal", "karras", "exponential", "sgm_uniform", "beta"]})
    return _ok({"modes": _edit_modes(), "defaults": character.GRAPH_DEFAULTS, **_SAMPLERS})


@routes.get("/api/characters")
async def characters(request: web.Request):
    rows = character.list_characters()
    for r in rows:
        r["thumb_url"] = file_url(r.pop("thumb", None))
    return _ok(rows)


@routes.get("/api/mannequin/schema")
async def mannequin_schema(request: web.Request):
    """The Mannequin editor's controls: body sliders, presets, limits (cinestudio/mannequin_preview.schema)."""
    from .. import mannequin_preview
    return _ok(await _bg(mannequin_preview.schema))


@routes.get("/api/characters/{name}")
async def character_show(request: web.Request):
    return _ok(_char_view(character.load(request.match_info["name"])))


@routes.post("/api/characters")
async def character_new(request: web.Request):
    """New character: source=text (job: candidates), image or sheet (instant)."""
    fields, files = await _form(request)
    return _ok(await _bg(_character_new, request.app, fields, files))


def _character_new(app, fields: dict, files: dict) -> dict:
    # the form's name is the DISPLAY name (any text); the ID (folder, files, jobs, runs) is made from it (2026-10-24)
    shown, source = character.clean_display(fields.get("name", "")), fields.get("source", "text")
    name = character.new_id(shown)
    text, style = fields.get("description", ""), fields.get("style", "")
    body = fields.get("body") or "human"
    if body not in character.BODY_TYPES:
        raise ValueError(f"body must be one of {list(character.BODY_TYPES)}")
    if source == "text":
        n = int(fields.get("candidates") or 8)
        if not 1 <= n <= 16:
            raise ValueError("candidates must be between 1 and 16")
        character.char_dir(name, must_exist=False)               # (the made ID always passes; kept as a guard)
        if not text.strip():
            raise ValueError("a description of the character is needed")
        seed = _int_or_none(fields.get("seed"))
        job = app["jobs"].submit("create", name, description=text, style=style, candidates=n,
                                 seed=1 if seed is None else seed, body=body, display_name=shown)
        return {"job": job}
    if source == "sheet":
        if "image" not in files:
            raise ValueError("a sheet image is needed")
        return {"character": _char_view(character.import_sheet(name, files["image"], text, style, display_name=shown))}
    slots = {s: files[s] for s in character.SLOTS if s in files}
    if "image" in files and "front" not in slots:
        slots["front"] = files["image"]
    return {"character": _char_view(character.import_slots(name, slots, text, style, body, display_name=shown))}


@routes.post("/api/characters/{name}/{action}")
async def character_action(request: web.Request):
    name, action = request.match_info["name"], request.match_info["action"]
    fields, files = await _form(request)
    return _ok(await _bg(_character_action, request.app, name, action, fields, files))


def _character_action(app, name: str, action: str, fields: dict, files: dict) -> dict:
    j = app["jobs"]
    if action == "delete":
        return character.delete_character(name)
    if action == "mannequin-preview":               # the Mannequin view: a DRAFT body (+ the saved kit) in Blender
        from .. import mannequin_preview                # Workbench, cached; nothing is saved
        import json as _json
        body = fields.get("body") or {}
        body = _json.loads(body) if isinstance(body, str) else body
        spec = character.load(name).get("proxy") or {}
        r = mannequin_preview.preview(name, body, spec)
        if r.get("superseded"):
            return r
        r["urls"] = {k: file_url(v) for k, v in r.pop("files").items()}
        return r
    if action == "stand-in":                        # the Mannequin view's ONE save: the whole body (the kit stays;
        import json as _json                        # an AI agent sets it through MCP)
        body = fields.get("body") or {}
        body = _json.loads(body) if isinstance(body, str) else body
        c = character.load(name)
        return {"character": _char_view(character.set_stand_in(name, body, c.get("proxy") or {}))}
    if action == "mannequin":                       # instant: the body of the character's own mannequin
        import json as _json
        body = fields.get("body") or {}
        body = _json.loads(body) if isinstance(body, str) else body
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
        default = str(fields.get("default") or "").lower() in ("1", "true", "yes", "on")
        return {"character": _char_view(character.set_mannequin(name, body, default=default))}
    if action == "proxy":                           # instant: the stand-in parts on the mannequin
        import json as _json
        spec = fields.get("spec")
        spec = _json.loads(spec) if isinstance(spec, str) else spec
        return {"character": _char_view(character.set_proxy(name, spec))}
    if action == "rename":                          # the app renames the DISPLAY name only (instant, nothing moves);
        return {"character": _char_view(character.set_display_name(name, fields.get("new_name", "")))}   # the ID stays
    if action == "pick":
        return {"character": _char_view(character.pick(name, int(fields["seed"])))}
    if action == "finish":
        return {"character": _char_view(character.finish(name))}
    settings = {k: fields[k] for k in ("steps", "cfg", "sampler", "scheduler", "shift", "denoise", "negative",
                                         "angles_strength") if fields.get(k) not in (None, "")}
    quality = {"mode": fields.get("mode") or "quality", "seed": _int_or_none(fields.get("seed")), "settings": settings}
    if action == "edit":
        ref = files.get("reference") or (fields.get("reference_path") or None)
        if ref and not files.get("reference") and not _inside(Path(ref), character.library(), runner.home() / "uploads"):
            raise ValueError("reference_path must be a file in the character library or the uploads folder")
        quality["settings"] = {k: v for k, v in settings.items() if k in character.EDIT_SETTINGS}
        character.quality_values(quality["mode"], quality["settings"], character.EDIT_SETTINGS)   # refuse bad values now
        return {"job": j.submit("edit", name, instruction=fields["instruction"], reference=ref,
                                    target=fields.get("target") or "front", region=fields.get("region") or None,
                                    **quality)}
    if action == "undo":
        return {"job": j.submit("undo", name, target=fields.get("target") or "front")}
    if action == "redo":
        quality["settings"] = {k: v for k, v in settings.items() if k in character.VIEW_SETTINGS}
        character.quality_values(quality["mode"], quality["settings"], character.VIEW_SETTINGS)
        return {"job": j.submit("redo_view", name, view=fields["view"], **quality)}
    if action == "views":
        return {"job": j.submit("views", name)}
    if action == "repose":                          # AnyPose: the front posed like its own mannequin, then the views
        c = character.load(name)
        if not (c.get("files") or {}).get("front"):
            raise ValueError(f"{name} has no front view to pose")
        if not isinstance(c.get("mannequin"), dict):
            raise ValueError(f"{name} has no mannequin body yet: set it in the Mannequin view first")
        return {"job": j.submit("repose", name, seed=_int_or_none(fields.get("seed")))}
    if action == "redraw-sheet":                    # a sheet redraw that failed after a body change: try it again
        return {"character": _char_view(character.rescale_sheet(name))}
    if action == "new-skin":                        # a new skin in a library style (GPU job; the pose stays)
        if not fields.get("style"):
            raise ValueError("choose a style for the new skin")
        return {"job": j.submit("new_skin", name, style=fields["style"], seed=_int_or_none(fields.get("seed")))}
    if action == "link-skin":                       # instant: make this character a skin of another
        return {"character": _char_view(character.link_skin(name, fields.get("base") or ""))}
    if action == "unlink-skin":
        return {"character": _char_view(character.unlink_skin(name))}
    if action == "use-style" or (action == "style" and "style_lora" not in fields):   # a style from the library
        from .styles import assign_character                                          # (instant; "" = none)
        return {"character": _char_view(assign_character(name, fields.get("style")))}
    if action == "style":
        return {"job": j.submit("style", name, style_lora=fields["style_lora"],
                                    keep=float(fields.get("keep", 0.5)))}
    raise ValueError(f"unknown action {action!r}")


@routes.get("/api/jobs/{id}")
async def job_show(request: web.Request):
    return _ok(jobs.load(request.match_info["id"]))
