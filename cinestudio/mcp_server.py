"""cinestudio as an MCP server (stdio): the same functions as the JSON CLI, for agents.

    python -m cinestudio mcp

Every generation workflow of the CLI has a tool here (tests/unit/test_mcp_server.py keeps the two in step: a new CLI
command needs a tool or a reasoned entry in its CLI-only list). Long GPU steps (character images, views, videos,
backgrounds, songs, templates) go into the shared queue and return at once with a job, run, ref or name; poll them
with job_status / run_status / list_backgrounds / show_song / show_template / queue_list, at most once a minute.
Parameter checks are the CLI's: choices are read from the CLI's own parser (_check_cli). The tips for good output
are in the resource cinestudio://agents (AGENTS.md): read it first.
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import character, jobs, runner, workqueue

ROOT = Path(__file__).resolve().parent.parent
READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
QUEUE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)
CHANGE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)    # instant, no queue
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)

INSTRUCTIONS = (
    "cinestudio turns a Blender shot (a mannequin and a camera) into an AI video of the user's own character "
    "(open-weight models in ComfyUI). Read the resource cinestudio://agents first: it has the hard rules and the "
    "tips for good output. Typical flow: doctor -> create_character -> (job_status) -> pick_candidate -> "
    "edit_image (optional) -> set_mannequin_body / set_proxy_kit (preview_mannequin, look at the PNGs) -> "
    "repose_character -> finish_character -> new_video_run (quality preview) -> run_status. For a long shot: "
    "new_video_run(through_clip=1) -> score_clip -> retake_clip / use_take -> continue_run (the clip gauntlet). "
    "Backgrounds (new_backplate, backplate_from_set, new_panorama), templates (template_from_video, "
    "make_dance_template, add_template), songs (make_song) and films (assemble_film) have their own tools. "
    "Rules: poll long work at most once a minute; new work waits at the end of the shared queue (queue_list); never "
    "cancel or delete the user's work unless the user asked for it in this conversation; look at the output files "
    "before you say that a result is good; tokens are never set or shown here.")

mcp = MCPServer("cinestudio", instructions=INSTRUCTIONS)


def _ok(**kw: Any) -> dict:
    return {"ok": True, **kw}


def _tool(annotations: ToolAnnotations):
    """Register a tool whose refusals reach the agent WITH their reason. The SDK hides the text of any exception other
    than ToolError ("Error executing tool X"), so a refused value, a missing character or a busy run would come back
    without saying why; the CLI prints every error's text, and so does this server. The module keeps the plain
    function (direct calls raise the original exception)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except ToolError:
                raise
            except Exception as exc:  # noqa: BLE001 - every failure as a readable tool error, as in the CLI
                raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        mcp.tool(annotations=annotations)(wrapper)
        return fn
    return deco


@functools.lru_cache(maxsize=None)
def _cli_actions(path: tuple[str, ...]) -> dict[str, argparse.Action]:
    """The CLI parser's arguments of one command (`("run", "new")`), by destination name."""
    from . import cli
    p: argparse.ArgumentParser = cli._parser()
    for name in path:
        sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
        p = sub.choices[name]
    return {a.dest: a for a in p._actions}


def _check_cli(path: tuple[str, ...], **values: Any) -> None:
    """Refuse a value the CLI would refuse: the same choices (and number types) as `python -m cinestudio <path>`."""
    acts = _cli_actions(path)
    for key, v in values.items():
        if v is None:
            continue
        a = acts[key]                                   # a KeyError here is a bug in this file, not bad input
        if a.type in (int, float) and (isinstance(v, bool) or not isinstance(v, (int, float))
                                       or (a.type is int and not isinstance(v, int))):
            raise ValueError(f"{key}: {v!r} is not {'an integer' if a.type is int else 'a number'}")
        if a.choices is not None and v not in a.choices:
            raise ValueError(f"{key}: {v!r} is not one of {', '.join(map(str, a.choices))}")


def _confirm(confirm: bool, what: str) -> None:
    if confirm is not True:
        raise ValueError(f"{what} is deleted for good: call again with confirm=true, and only when the user asked for "
                         "this delete in this conversation")


def _template_blend(shot: str) -> str:
    """A template's .blend from its path or its library name (list_templates)."""
    if Path(shot).is_file():
        return shot
    from . import shots
    blend = (shots.load(shot).get("files") or {}).get("blend")
    if not blend:
        raise ValueError(f"template {shot!r} has no .blend (a video template)")
    return blend


# ---------------------------------------------------------------- information, setup, engine
@mcp.resource("cinestudio://agents", name="AGENTS.md", description="Hard rules and tips for good output",
              mime_type="text/markdown")
def agents_md() -> str:
    return (ROOT / "AGENTS.md").read_text(encoding="utf-8")


@_tool(READ)
def doctor() -> dict:
    """Check Blender, ffmpeg, ComfyUI, the proxy, the models and which tokens are set (never their values)."""
    from . import cli
    return cli.doctor()


@_tool(READ)
def list_loras() -> dict:
    """LoRA files that ComfyUI can load (for style_lora; ComfyUI must be running)."""
    from . import cli
    return _ok(loras=cli.loras())


@_tool(READ)
def settings_status() -> dict:
    """The settings (model tier for new work, backend, models folder, default style), the tier names, and which access
    tokens are set and where they come from. Token values are never shown, and tokens are never set through MCP: the
    user sets them in the app (Settings) or with `settings token set NAME` in a terminal. Changing the tier, backend or
    models folder is the user's choice (CLI `settings ...` or the app)."""
    from . import models, settings
    return _ok(settings=settings.get(), tiers=models.tier_names(),
               tokens={k: {"set": v["set"], "source": v["source"]} for k, v in settings.status().items()})


@_tool(READ)
def models_status(tier: str | None = None, features: list[str] | None = None) -> dict:
    """The model files of a tier (default: the tier in the settings): each file's state, the bytes missing, whether the
    tier is ready; `features` limits it to video, character, music, select, pose. Also the background download's
    progress ("download")."""
    from . import models
    return _ok(**models.status(tier, features), download=models.download_state())


@_tool(QUEUE)
def download_models(tier: str | None = None, features: list[str] | None = None) -> dict:
    """Start the background download of a tier's missing model files (free; resumable, SHA-256 checked; gated files use
    the user's own hf token when it is set). Returns at once; poll models_status ("download") at most once a minute.
    Downloads are large (8gb tier ~154 GB): ask the user first when the disk may be short."""
    from . import models, settings
    return _ok(**models.start_download(tier or settings.get()["model_tier"], features))


@_tool(DESTRUCTIVE)
def stop_model_download() -> dict:
    """Stop the background model download (files already done stay; a later download continues).
    ONLY when the user asked for this stop."""
    from . import models
    return _ok(**models.request_stop())


@_tool(READ)
def engine_status() -> dict:
    """The ComfyUI engine: running or not, whether cinestudio started it, its log. cinestudio starts it by itself
    when a job needs it; starting and stopping it by hand stays in the CLI (`comfy start|stop`)."""
    from . import comfy
    return _ok(**comfy.status())


# ---------------------------------------------------------------- the shared queue
@_tool(READ)
def queue_list() -> dict:
    """The shared queue: what runs now and what waits (character steps, videos, songs, backgrounds, templates)."""
    return _ok(**workqueue.view())


@_tool(DESTRUCTIVE)
def release_queue_hold(kind: str, ref: str, confirm: bool = False) -> dict:
    """Release an unresolved queue hold only when the user explicitly asks and confirms the job was checked.
    The user must check the full proxy queue; ComfyUI and the proxy must be reachable with no observed active jobs.
    The song record is kept."""
    _check_cli(("queue", "release-hold"), kind=kind)
    return _ok(**workqueue.release_hold(kind, ref, confirm=confirm))


@_tool(DESTRUCTIVE)
def cancel_queued(kind: str, ref: str) -> dict:
    """Cancel one queued or running request of the shared queue (kind character|video|song|lab|template; ref = the job
    id, run id, song name, background ref "bg-..." or template name, as queue_list shows). ONLY when the user asked
    for this cancel: it stops the user's work."""
    _check_cli(("queue", "cancel"), kind=kind)
    return _ok(**workqueue.cancel(kind, ref))


# ---------------------------------------------------------------- styles (one set of look words per look)
@_tool(READ)
def list_styles() -> dict:
    """The style library: each style's words, light words, image negative words, video LoRA, which characters,
    templates and backgrounds use it, and the default. Use the SAME style for every asset of a shot (characters,
    background, template, props): cinestudio://agents section 6."""
    from . import styles
    used = styles.used_by()
    return _ok(default=styles.default_style(),
               styles=[{**s, "used_by": used.get(s["name"], {})} for s in styles.list_styles()])


@_tool(READ)
def show_style(name: str) -> dict:
    """One style: its words, light, negative words, LoRA, reference picture (and its crop box), version, the assets that
    use it, the reference candidates made by the video model and the state of its reference job."""
    from . import style_ref, styles
    return _ok(style=styles.load(name), used_by=styles.used_by().get(name, {}),
               reference_candidates=style_ref.candidates(name), reference_job=style_ref.job_status(name))


def _save_style(name: str, new: bool, label, words, light, negative_image, lora, lora_strength, reference) -> dict:
    """The CLI's `style new|edit` fields -> styles.save (a LoRA "" = none; a strength alone changes the current LoRA)."""
    from . import styles
    data = {k: v for k, v in (("label", label), ("words", words), ("light", light),
                              ("negative_image", negative_image), ("reference", reference)) if v is not None}
    if lora is not None:
        data["lora"] = {"name": lora, "strength": lora_strength if lora_strength is not None else 0.4} if lora else None
    elif lora_strength is not None and not new:
        cur = styles.load(name).get("lora")
        if cur:
            data["lora"] = {**cur, "strength": lora_strength}
    return _ok(style=styles.save(name, data, new=new))


@_tool(CHANGE)
def new_style(name: str, words: str, label: str | None = None, light: str | None = None,
              negative_image: str | None = None, lora: str | None = None, lora_strength: float | None = None,
              reference: str | None = None) -> dict:
    """Make a style (instant): `words` = the look ("stylized cinematic 3D animation, ..."; no meta words such as
    "premium" or "8K"), `light` ("evenly lit by soft neutral light from the front"), `negative_image` (looks to exclude
    in image jobs), `lora` (an LTX style LoRA file; strength default 0.4), `reference` (a picture: a tight crop in this
    style)."""
    return _save_style(name, True, label, words, light, negative_image, lora, lora_strength, reference)


@_tool(CHANGE)
def edit_style(name: str, words: str | None = None, label: str | None = None, light: str | None = None,
               negative_image: str | None = None, lora: str | None = None, lora_strength: float | None = None,
               reference: str | None = None) -> dict:
    """Change a style (instant; given fields change, others stay; lora "" = none). Its version goes up: assets made
    before are reported as older (their pictures never change by themselves). Editing a built-in style makes the
    user's copy of it."""
    return _save_style(name, False, label, words, light, negative_image, lora, lora_strength, reference)


@_tool(DESTRUCTIVE)
def delete_style(name: str, confirm: bool = False) -> dict:
    """Delete one of the user's styles (built-in styles stay). ONLY when the user asked for this delete; needs
    confirm=true."""
    from . import styles
    _confirm(confirm, f"style {name!r}")
    return _ok(**styles.delete(name))


@_tool(CHANGE)
def set_default_style(name: str) -> dict:
    """The style new assets get by default (instant)."""
    from . import styles
    return _ok(**styles.set_default(name))


@_tool(QUEUE)
def make_style_reference(name: str, subject: str | None = None, seeds: list[int] | None = None,
                         video_model: str | None = None) -> dict:
    """Queue reference-picture candidates for a style made by the VIDEO model (a GPU job): LTX-2.5 + the style's LoRA
    draw a still shot of `subject` from text only (default: a young woman in a hooded jacket in a city street, full
    body; a person, cloth and a place), one candidate per seed (default 202, 303, 404, 505; the sharpest frame of each).
    A style without a LoRA shows LTX's own look for its words. `video_model`: distilled (the model of the
    layout videos) or full. Returns the queue ref at once; poll show_style
    (reference_job, reference_candidates) at most once a minute; LOOK at the pictures, then use_style_reference."""
    from . import style_ref
    return _ok(**style_ref.submit(name, subject, seeds, via="mcp", video_model=video_model))


@_tool(CHANGE)
def use_style_reference(name: str, file: str, crop: list[int] | None = None) -> dict:
    """Make a picture (a candidate of make_style_reference, or any picture in the style) the style's reference
    (instant; a built-in style gets the user's copy; the version goes up). The whole picture is kept with a crop box
    `crop` [x0, y0, x1, y1] in its pixels (default: a square on the clothes below the face); Qwen style conversions get only
    that crop as Picture 2 (one detail, tightly cropped). Look at crop_file before you use it."""
    from . import style_ref
    return _ok(**style_ref.use_reference(name, file, crop))


@_tool(CHANGE)
def set_character_style(name: str, style: str) -> dict:
    """Give a character a style from the library (instant): its words become the character's style (the look words of
    its runs and edits). Finished pictures do not change; a character drawn in another look must be converted first."""
    return _ok(character=character.set_style(name, style))


@_tool(QUEUE)
def character_skins(name: str, new_style: str | None = None, link_to: str | None = None, unlink: bool = False,
                    seed: int | None = None) -> dict:
    """A character's SKINS (the same character in different styles; each skin is its own record with `skin_of` = the
    main record's ID). Without arguments: the list (main first). `new_style`: queue a GPU job that makes a new skin in
    that library style (this skin's front redrawn by Qwen, the pose kept, then the views and the sheet; finish it after
    a look; poll job_status at most once a minute). `link_to`: make this existing character a skin of `link_to`
    (instant). `unlink`: make this skin a character of its own again (instant)."""
    if sum(map(bool, (new_style, link_to, unlink))) > 1:
        raise ValueError("give one of new_style, link_to or unlink")
    if link_to:
        return _ok(character=character.link_skin(name, link_to))
    if unlink:
        return _ok(character=character.unlink_skin(name))
    if new_style:
        return _ok(job=jobs.submit("new_skin", name, via="mcp", style=new_style, seed=seed))
    return _ok(skins=character.skins(name))


# ---------------------------------------------------------------- characters
@_tool(READ)
def list_characters() -> dict:
    """The character library (ID, display name, status, thumbnail path, what a Blender video still needs)."""
    return _ok(characters=character.list_characters())


@_tool(READ)
def show_character(name: str) -> dict:
    """One character: its images, candidates, history, status, mannequin body and proxy kit."""
    return _ok(character=character.load(name))


@_tool(QUEUE)
def create_character(name: str, description: str, style: str = "", candidates: int = 8, seed: int = 1,
                     body: str = "human", display_name: str = "", style_lora: str | None = None) -> dict:
    """Queue front-view candidates from text (Z-Image; a GPU job). `name` is the ID (lowercase, "-"; used in every
    path), `display_name` the shown name (any text). The description: objective and concrete, head to feet, colours,
    counts and sides; "exactly five clearly separated fingers" (cinestudio://agents 4.2). Candidates use seed,
    seed+1, ... Returns a job: poll job_status at most once a minute, LOOK at candidates/candidates.jpg, then
    pick_candidate."""
    _check_cli(("character", "create"), body=body)
    return _ok(job=jobs.submit("create", name, via="mcp", description=description, style=style, candidates=candidates,
                               seed=seed, body=body, display_name=display_name, style_lora=style_lora))


@_tool(CHANGE)
def import_character(name: str, image: str | None = None, sheet: str | None = None, right: str | None = None,
                     back: str | None = None, left: str | None = None, face: str | None = None,
                     description: str = "", style: str = "", body: str = "human", display_name: str = "") -> dict:
    """A character from the user's own pictures (instant, no GPU): `image` = the front view (then make_views or
    repose_character makes the missing views; `right`, `back`, `left`, `face` are optional own pictures), OR `sheet` =
    a finished sheet used as it is (ready at once). `description`: the costume in exact words (the video and look texts
    use it). Then set a mannequin body before a Blender video."""
    _check_cli(("character", "import"), body=body)
    if bool(image) == bool(sheet):
        raise ValueError("give image (the front view) or sheet (a finished sheet), one of them")
    slots = {k: v for k, v in (("right", right), ("back", back), ("left", left), ("face", face)) if v}
    if sheet:
        if slots:
            raise ValueError("right, back, left and face go with image, not with a finished sheet")
        return _ok(character=character.import_sheet(name, sheet, description, style, display_name=display_name),
                   next="ready")
    c = character.import_slots(name, {"front": image, **slots}, description, style, body, display_name=display_name)
    return _ok(character=c, next="make_views or repose_character")


@_tool(QUEUE)
def pick_candidate(name: str, seed: int) -> dict:
    """Queue the choice of the front view among the candidates (by its seed); it waits behind earlier steps of
    the character. The old front stays as an undoable version."""
    return _ok(job=jobs.submit("pick", name, via="mcp", seed=seed))


@_tool(QUEUE)
def edit_image(name: str, instruction: str, target: str = "front", region: str | None = None,
               denoise: float | None = None, mode: str = "quality", seed: int | None = None,
               reference: str | None = None, settings: dict | None = None) -> dict:
    """Queue an edit of one image (Qwen-Image-Edit; a GPU job, ~4 min). One change per edit; end with "Keep everything
    else exactly the same." target: front, right, back, left or face. region "x0,y0,x1,y1" edits only that box (crop
    and stitch: always use it for small fixes such as a hand; describe the crop). A lower denoise (e.g. 0.5) changes
    less. `reference`: a second picture (Picture 2) showing ONE detail, tightly cropped. `settings`: steps, cfg,
    sampler, scheduler, shift, negative (default: the mode's values). Every edit can be undone (undo_edit). Poll
    job_status at most once a minute and LOOK at the result before the next edit."""
    _check_cli(("character", "edit"), target=target, mode=mode)
    settings = {**(settings or {}), **({"denoise": denoise} if denoise is not None else {})}
    character.quality_values(mode, settings, character.EDIT_SETTINGS)          # refuse bad values now
    return _ok(job=jobs.submit("edit", name, via="mcp", instruction=instruction, target=target, reference=reference,
                               region=region, mode=mode, seed=seed, settings=settings))


@_tool(QUEUE)
def undo_edit(name: str, target: str = "front") -> dict:
    """Queue an undo of the last change (edit, redo, pick or touch-up) of one image."""
    _check_cli(("character", "undo"), target=target)
    return _ok(job=jobs.submit("undo", name, via="mcp", target=target))


@_tool(QUEUE)
def make_views(name: str, mode: str = "quality", seed: int | None = None, distance: str = "medium",
               face: str = "front") -> dict:
    """Queue the right, back, left and face views and the sheet (a GPU job). Body "human" first poses the front like
    the character's own mannequin (AnyPose) unless it is already posed or own side/back pictures were given.
    `distance`: medium (default) or wide (drew rooms); `face`: front (cut from the front; default) or qwen. Poll
    job_status at most once a minute; then check every view (the side, the finger count, missing colours, no face on
    the back of the head)."""
    _check_cli(("character", "views"), mode=mode, distance=distance, face=face)
    return _ok(job=jobs.submit("views", name, via="mcp", mode=mode, seed=seed, distance=f"{distance} shot", face=face))


@_tool(QUEUE)
def redo_view(name: str, view: str, seed: int | None = None, mode: str = "quality",
              settings: dict | None = None) -> dict:
    """Queue one view again with another seed (right, back, left or face; a GPU job). The earlier view stays undoable.
    `settings`: steps, cfg, sampler, scheduler, shift, negative, angles_strength."""
    _check_cli(("character", "redo-view"), view=view, mode=mode)
    settings = dict(settings or {})
    character.quality_values(mode, settings, character.VIEW_SETTINGS)          # refuse bad values now
    return _ok(job=jobs.submit("redo_view", name, via="mcp", view=view, seed=seed, mode=mode, settings=settings))


@_tool(QUEUE)
def repose_character(name: str, shot: str | None = None, seed: int | None = None) -> dict:
    """Queue AnyPose (order A, a GPU job): the FRONT is posed like the character's own mannequin (Blender renders the
    bare mannequin with its body; `shot` = another .blend whose mannequin gives the pose), then the views and the
    sheet are made from it. Needs a front and a mannequin body (set_mannequin_body). Repose from the ORIGINAL front
    (undo an old posed front first). Poll job_status at most once a minute; compare the result with the original
    (AnyPose can shrink the head or redraw small parts)."""
    c = character.load(name)
    if not (c.get("files") or {}).get("front"):
        raise ValueError(f"{name} has no front view to pose")
    if not isinstance(c.get("mannequin"), dict):
        raise ValueError(f"{name} has no mannequin body yet: set_mannequin_body first")
    return _ok(job=jobs.submit("repose", name, via="mcp", shot=shot, seed=seed))


@_tool(QUEUE)
def restyle_sheet(name: str, style_lora: str, keep: float = 0.5) -> dict:
    """Queue a redraw of the sheet with LTX-2.5 and a style LoRA (a GPU job); about `keep` of the sheet stays. A low
    keep can change design details: check every panel. The character is then ready."""
    return _ok(job=jobs.submit("style", name, via="mcp", style_lora=style_lora, keep=keep))


@_tool(CHANGE)
def redraw_sheet(name: str) -> dict:
    """Draw the sheet again at the character's current height (instant, no GPU): after a body change whose sheet
    redraw failed. A ready character stays ready."""
    return _ok(character=character.rescale_sheet(name))


@_tool(QUEUE)
def finish_character(name: str) -> dict:
    """Queue: use the sheet as it is; the character is then ready for videos (after earlier queued steps). A Blender
    video also needs a mannequin body (set_mannequin_body)."""
    return _ok(job=jobs.submit("finish", name, via="mcp"))


@_tool(READ)
def preview_mannequin(name: str, body: dict | None = None, spec: dict | None = None) -> dict:
    """Render the character's own mannequin (Blender Workbench, a few seconds, cached; nothing is saved): its body (or
    the draft `body`) with its proxy kit (or the draft `spec`) from the front, its right side and the back, at one fixed
    scale (a ring marks the template's 1.81 m). Returns the PNG paths and the height: LOOK at the PNGs (and at the
    character's sheet) before you set a kit."""
    c = character.load(name)
    from . import mannequin_preview
    r = mannequin_preview.preview(name, (c.get("mannequin") or {}) if body is None else body,
                                  (c.get("proxy") or {}) if spec is None else spec)
    return _ok(**r)


@_tool(CHANGE)
def set_proxy_kit(name: str, spec: dict) -> dict:
    """Give the character its PROXY KIT: simple clay parts on its mannequin in every Blender video, so the layout
    carries the character's outline (the mask, the hair, a cape...). The app has no editor for it: an AI agent sets it
    here (user, 2026-10-24). Make it from the character's sheet: its colours (linear RGB 0..1), mask shape, hair length,
    cape length. Parts and fields: see AGENTS.md "Proxy kit" and character.validate_proxy (mask {shape round|pointed,
    eyes, eye_size, ears, ear_height, ear_width, ear_inner, top, marks: a preset name or a list, color}, hair {length
    short|medium|long, front, locks, back, color}, cape {length hips|thigh|knee, width, flare, wrap, color}, jacket,
    bands, suit, shoes). Check it with preview_mannequin(spec=...) first. {} removes every part. The spec is checked in
    full before it is saved."""
    c = character.set_proxy(name, spec)
    return _ok(name=name, proxy=c["proxy"], missing=character.video_missing(c))


@_tool(CHANGE)
def set_mannequin_body(name: str, body: dict) -> dict:
    """Set the character's whole mannequin BODY (keys scale, head, neck, torso, legs, arms; 1.0 = the template's; {} =
    the template's body). The sheet is drawn again at the new height; the pose changes only after repose_character. A
    Blender video needs a body. Ask before you replace a body the character already has."""
    rec = character.load(name)
    c = character.set_stand_in(name, body, rec.get("proxy") or {})
    return _ok(name=name, mannequin=c["mannequin"], height=c.get("height"), missing=character.video_missing(c))


@_tool(CHANGE)
def rename_character(name: str, new_name: str) -> dict:
    """Give a character a new ID (its folder moves; stored paths follow; runs keep their own copy). Refused while a
    job of the character waits or runs. For the shown name only, use set_display_name."""
    return _ok(character=character.rename(name, new_name))


@_tool(CHANGE)
def set_display_name(name: str, text: str) -> dict:
    """The name the app shows (any text, at most 80 characters); the ID stays."""
    return _ok(character=character.set_display_name(name, text))


@_tool(DESTRUCTIVE)
def delete_character(name: str, confirm: bool = False) -> dict:
    """Delete a character for good (runs keep their own sheet copy). ONLY when the user asked for this delete in this
    conversation; needs confirm=true. Refused while a job of the character waits or runs."""
    _confirm(confirm, f"character {name!r}")
    return _ok(**character.delete_character(name))


@_tool(READ)
def job_status(job_id: str) -> dict:
    """A character job: status (queued, running, done, error, canceled, interrupted), progress, result. Poll at most
    once a minute."""
    j = jobs.load(job_id)
    return _ok(job={**j, "status": jobs.live_status(j)})


@_tool(READ)
def list_jobs(character_name: str | None = None) -> dict:
    """The latest character jobs (all, or one character's)."""
    return _ok(jobs=jobs.list_jobs(character_name))


@_tool(DESTRUCTIVE)
def cancel_job(job_id: str) -> dict:
    """Cancel a character job. ONLY when the user asked for this cancel: it stops the user's work."""
    return _ok(job=jobs.cancel(job_id))


# ---------------------------------------------------------------- props (cinestudio://agents 4.10)
@_tool(READ)
def list_props() -> dict:
    """The prop library (ID, display name, words, style, status, thumbnail path): objects for a template's Props layer,
    used by new_video_run(prop_refs={"Ball_*": "dodgeball"})."""
    from . import props
    return _ok(props=props.list_props())


@_tool(READ)
def show_prop(name: str) -> dict:
    """One prop: its pictures (front.png: the prop alone on black), candidates, words, style and history."""
    from . import props
    return _ok(prop=props.load(name))


@_tool(QUEUE)
def create_prop(name: str, words: str, description: str = "", style: str | None = None, candidates: int = 4,
                seed: int = 202, model: str = "qwen-image", display_name: str = "", negative: str = "") -> dict:
    """Queue prop picture candidates from text (a GPU job; Qwen-Image-2512 by default, or model "z-image"): the prop
    alone on pure black, flatly lit, in the library style `style` (default: the default style; use the shot's
    characters' style). `name` = the ID; `words` = a few concrete words that every video prompt uses ("matte deep violet
    rubber dodgeball with a thin magenta seam"); `description` = more detail for the picture only. Poll job_status at
    most once a minute, LOOK at candidates/candidates.jpg, then pick_prop_candidate."""
    from . import props
    _check_cli(("prop", "create"), model=model)
    props._clean_words(words)
    return _ok(job=jobs.submit("prop_create", name, via="mcp", words=words, description=description, style=style,
                               candidates=candidates, seed=seed, model=model, display_name=display_name,
                               negative=negative))


@_tool(CHANGE)
def import_prop(name: str, image: str, words: str, description: str = "", style: str | None = None,
                side: str | None = None, candidates: list[str] | None = None, display_name: str = "") -> dict:
    """A prop from the user's own picture (instant): `image` = the front (the prop alone on black, in the characters'
    style), `side` an optional second view, `candidates` more pictures to pick from later, `style` the library style
    the picture is drawn in."""
    from . import props
    return _ok(prop=props.import_picture(name, image, words, description, style, side=side, candidates=candidates,
                                         display_name=display_name))


@_tool(CHANGE)
def pick_prop_candidate(name: str, seed: int) -> dict:
    """Use one candidate (by its seed) as the prop's front (instant; the old front stays undoable)."""
    from . import props
    return _ok(prop=props.pick(name, seed))


@_tool(QUEUE)
def edit_prop(name: str, instruction: str, target: str = "front", region: str | None = None,
              denoise: float | None = None, mode: str = "quality", seed: int | None = None,
              reference: str | None = None, settings: dict | None = None) -> dict:
    """Queue an edit of one prop picture (Qwen-Image-Edit; a GPU job; Picture 1 = the prop). One change per edit; end
    with "Keep everything else exactly the same." `region` "x0,y0,x1,y1" edits only that box; `reference` = Picture 2
    (one detail, tightly cropped). Undo with undo_prop_edit. LOOK at the result before the next edit."""
    from . import props
    _check_cli(("prop", "edit"), target=target, mode=mode)
    settings = {**(settings or {}), **({"denoise": denoise} if denoise is not None else {})}
    character.quality_values(mode, settings, character.EDIT_SETTINGS)          # refuse bad values now
    props.check_id(name)
    return _ok(job=jobs.submit("prop_edit", name, via="mcp", instruction=instruction, target=target,
                               reference=reference, region=region, mode=mode, seed=seed, settings=settings))


@_tool(CHANGE)
def undo_prop_edit(name: str, target: str = "front") -> dict:
    """Undo the last change (edit, pick or touch-up) of one prop picture (instant)."""
    from . import props
    _check_cli(("prop", "undo"), target=target)
    return _ok(prop=props.undo(name, target))


@_tool(CHANGE)
def set_prop_words(name: str, words: str | None = None, description: str | None = None, style: str | None = None,
                   display_name: str | None = None) -> dict:
    """Change a prop's words (the video prompts and the run sheet's "**Prop:**" line), its picture description, the
    library style its pictures are drawn in, and/or its display name (instant; pictures do not change)."""
    from . import props
    if all(v is None for v in (words, description, style, display_name)):
        raise ValueError("give words, description, style or display_name")
    p = props.load(name)
    if words is not None or description is not None:
        p = props.set_words(name, words, description)
    if style:
        p = props.set_style(name, style)
    if display_name is not None:
        p = props.set_display_name(name, display_name)
    return _ok(prop=p)


@_tool(QUEUE)
def prop_skins(name: str, new_style: str | None = None, seed: int | None = None) -> dict:
    """A prop's SKINS (the same prop in other library styles; each skin its own record with `skin_of`). Without
    `new_style`: the list (main first). `new_style`: queue a GPU job that redraws this skin's front in that style
    (Qwen-Image-Edit; a new record "<main>-<style>"); LOOK at it before a run."""
    from . import props
    if new_style:
        props.check_id(name)
        return _ok(job=jobs.submit("prop_skin", name, via="mcp", style=new_style, seed=seed))
    return _ok(skins=props.skins(name))


@_tool(CHANGE)
def rename_prop(name: str, new_name: str) -> dict:
    """Give a prop a new ID (its folder moves; its skins follow; runs keep their own copies). For the shown name only,
    use set_prop_words(display_name=...)."""
    from . import props
    return _ok(prop=props.rename(name, new_name))


@_tool(DESTRUCTIVE)
def delete_prop(name: str, confirm: bool = False) -> dict:
    """Delete a prop for good (runs keep their own copy of its picture). ONLY when the user asked for this delete in
    this conversation; needs confirm=true. Refused while a job of the prop waits or runs, or while it has skins."""
    from . import props
    _confirm(confirm, f"prop {name!r}")
    return _ok(**props.delete(name))


# ---------------------------------------------------------------- backgrounds (360 panoramas and backplates)
@_tool(READ)
def list_backgrounds() -> dict:
    """The background library (panoramas and backplates: name, kind, scene, style, camera, file), the background jobs
    waiting or running in the queue, and failed jobs with their error."""
    from . import background
    from .webapi import backgrounds as wb
    jobs_now = wb._jobs()
    return _ok(backgrounds=background.list_backgrounds(), jobs=jobs_now,
               failed=[f for f in background.failed_jobs() if not any(j["ref"] == f["ref"] for j in jobs_now)])


@_tool(READ)
def show_background(name: str) -> dict:
    """One saved background (its picture file, kind, scene words, style, the camera a backplate was made for)."""
    from . import background
    return _ok(background=background.load(name))


@_tool(QUEUE)
def new_panorama(scene: str, style: str = "", style_preset: str | None = None, seed: int | None = None,
                 name: str | None = None, depth: bool = True) -> dict:
    """Queue a new 360 panorama of a place (Qwen-Image + the 360 LoRA, then its 360 depth; a GPU job) for any camera,
    also orbits. `style_preset` (list_styles) or `style` words: use the shot's characters' style. Returns the queue ref
    at once; poll list_backgrounds at most once a minute; LOOK at the picture next to the character sheet."""
    from .webapi import backgrounds as wb
    return _ok(**wb._new({"scene": scene, "style": style, "style_preset": style_preset, "seed": seed, "name": name,
                          "depth": depth}))


@_tool(QUEUE)
def new_backplate(scene: str, shot: str, style: str = "", style_preset: str | None = None, seed: int | None = None,
                  name: str | None = None) -> dict:
    """Queue a BACKPLATE (a GPU job): one picture of the place for the STILL camera of a template (`shot`: its name or
    its .blend), 4-9x sharper than a panorama there. Returns the queue ref at once; poll list_backgrounds at most
    once a minute. Use it in new_video_run(background="saved:NAME")."""
    from .webapi import backgrounds as wb
    return _ok(**wb._new({"kind": "plate", "scene": scene, "shot": _template_blend(shot), "style": style,
                          "style_preset": style_preset, "seed": seed, "name": name}))


@_tool(QUEUE)
def backplate_from_set(shot: str, scene: str = "", style: str = "", style_preset: str | None = None,
                       seed: int | None = None, name: str | None = None) -> dict:
    """Queue a backplate drawn OVER the template's own Blender set (a GPU job): Blender renders the set alone from the
    still camera, Qwen-Image-Edit redraws it as the place with every object kept in its place. `scene` default: the
    template's Set layer words (template_layers). Use it when the template has its own set. Returns the queue ref at
    once; poll list_backgrounds at most once a minute."""
    from .webapi import backgrounds as wb
    return _ok(**wb._from_set({"shot": _template_blend(shot), "scene": scene, "style": style,
                               "style_preset": style_preset, "seed": seed, "name": name}))


@_tool(CHANGE)
def import_background(file: str, name: str | None = None, scene: str = "") -> dict:
    """The user's own picture of the place as a backplate (instant; fitted to the frame in each run)."""
    from . import background
    return _ok(**background.import_plate(file, name, scene))


@_tool(QUEUE)
def panorama_depth(name: str) -> dict:
    """Queue the 360 depth (MoGe-2) of a saved panorama (a GPU job; the ground mesh of depth runs). Returns the queue
    ref at once; poll list_backgrounds at most once a minute."""
    from .webapi import backgrounds as wb
    wb._check_pano(name)
    return _ok(**wb._start("depth", name, ["depth", name], f"panorama {name}: 360 depth (MoGe-2)"))


@_tool(QUEUE)
def restyle_panorama(name: str, style_lora: str, keep: float | None = None, strength: float | None = None,
                     style: str = "") -> dict:
    """Queue a redraw of a saved panorama with LTX-2.5 and a style LoRA (a GPU job), saved as a new panorama (the
    original stays). It changes little and can invent details (keep 0.15-0.3 tested). Returns the queue ref at once."""
    from .webapi import backgrounds as wb
    return _ok(**wb._style(name, {"style_lora": style_lora, "keep": keep, "strength": strength, "style": style}))


@_tool(DESTRUCTIVE)
def delete_background(name: str, confirm: bool = False) -> dict:
    """Delete a saved background (runs keep their own copy; built-ins stay). ONLY when the user asked for this delete;
    needs confirm=true."""
    from . import background
    _confirm(confirm, f"background {name!r}")
    return _ok(**background.delete(background.load(name)["name"]))


# ---------------------------------------------------------------- templates (Blender shots)
@_tool(READ)
def list_templates() -> dict:
    """The template library (built-in and the user's): name, status, kind, description, files (blend, preview),
    layers, default background, people of a template from a video."""
    from . import shots
    return _ok(templates=shots.list_shots())


@_tool(READ)
def show_template(name: str) -> dict:
    """One template: its record (status while it is being made, files, plan, timeline, layers, people scan)."""
    from . import shots
    return _ok(template=shots.show(name))


@_tool(CHANGE)
def add_template(name: str, blend: str, description: str = "", move: bool = False, preview: bool = True) -> dict:
    """Store a Blender shot as a template (the .blend is copied, or moved with move=true; a Workbench preview with its
    sound is made: Blender, blocks for a little while). Every shot an agent builds belongs in the library. Rules for a
    template that does not mislead the video: cinestudio://agents section 5."""
    from . import shots
    return _ok(template=shots.add(name, blend, description, move, preview))


@_tool(CHANGE)
def template_layers(name: str, set_description: str | None = None, props: dict[str, str] | None = None,
                    rescan: bool = False) -> dict:
    """A template's three layers (Mannequin* = characters, Props, Set) and their words. Without arguments: read them
    (a user template without a scan is scanned now: Blender, a few seconds). `set_description`: what the Set stands for
    ("a school gym with a ball cart at the far wall"); `props`: {prop name or pattern such as "Ball_*": words; "" removes
    it}; `rescan`: read the layers from the .blend again. Runs freeze the words into their texts."""
    from . import shots
    return _ok(**shots.layers(name, set_description, props or {}, rescan))


@_tool(CHANGE)
def template_clip_texts(name: str, clip_texts: dict[str, str] | None = None, context: str | None = None) -> dict:
    """A template's clip texts: one ACTION text per clip (only what that clip's frames show; clip k = frames
    96(k-1)+1 .. +121) and a `context` for EVERY clip (lasting states and orientation, e.g. "{character} faces the
    camera the whole time"; "{character}" becomes the run's words for its character). Without arguments: read them
    with each clip's frame span. `clip_texts` {"2": "..."} sets those clips ("" removes one); `context` "" removes it.
    Built-in templates are read-only. A multi-clip run takes a clip's actions from its own clip_videos, else from here;
    the run's main video text is then the shared context (who, place, light, look). cinestudio://agents 7.2."""
    from . import shots
    return _ok(**shots.clip_texts(name, clip_texts or {}, context))


@_tool(CHANGE)
def check_template(name: str, fresh: bool = False) -> dict:
    """Check a Blender template against the template rules (instant when cached; else one headless Blender read, a few
    seconds, no render, no GPU job): pose-bone scales = 1, no action across a clip join's 25 shared frames, an end
    hold, frame 1 a still front pose, camera < 0.6 deg/frame, layers (nothing loose, no prop in Mannequin), figure and
    prop sizes on screen, a neutral grey Set, clip texts + context for every clip, the sound, joint pops. Returns
    {errors, warnings, checks: [{id, severity error|warning|info|ok, message, frames, fix}]}; the summary is stored in
    a user template's shot.json ("check"). `fresh`: read the .blend again. Fix errors before a run or an export;
    cinestudio://agents section 5 and docs/TEMPLATE_GUIDE.md."""
    from . import shot_check
    res = shot_check.check(name, fresh=fresh)
    return _ok(**{k: v for k, v in res.items() if k != "ok"})


@_tool(CHANGE)
def template_timeline(name: str) -> dict:
    """The template's motion events from its rig (which hand waves when, feet planted; Blender, a few seconds), stored
    in its shot.json; runs use them with clip_text "auto"."""
    from . import shots
    return _ok(**shots.timeline(name))


@_tool(QUEUE)
def template_from_video(name: str, video: str, start: float = 0.0, seconds: float | None = None,
                        sound: str = "video", people: str = "1", tracks: str | list[int] | None = None,
                        description: str = "") -> dict:
    """Queue a Blender template from a dance VIDEO the user supplied (SAM 3D Body reads the 3D motion, the mannequin
    copies it; still front camera, plain floor; a GPU job). The video should face a still camera. `seconds` default 10,
    at most 30. `sound` video = the video's own sound (the template is then PRIVATE) or none. `people`: 1 (one dancer;
    default), 2 / 3 / 4 (SAM 3 finds every person; that many seen longest and largest become Mannequin = the picture's
    left, Mannequin2, ... to the right), all (every person seen long enough, up to 4: a run takes at most 4 characters)
    or pick (stop after the scan with status "pick" and a numbered sheet in people/sheet.jpg, then
    pick_template_people). `tracks` (e.g. "0,1,2"): the people by their SAM 3 numbers when known. Returns at once; poll
    show_template at most once a minute. Never download a video from YouTube."""
    from . import video_template
    _check_cli(("shot", "from-video"), sound=sound, people=people)
    return _ok(template=video_template.start(name, video, start, seconds, sound, description, people=people,
                                             tracks=tracks))


@_tool(QUEUE)
def pick_template_people(name: str, tracks: str | list[int]) -> dict:
    """After template_from_video(people="pick"): build the template from 1 to 4 chosen people (SAM 3 numbers from
    people/sheet.jpg, e.g. "0,2,1"). No second GPU job; Blender in a detached worker. Poll show_template."""
    from . import video_template
    return _ok(template=video_template.pick(name, tracks))


@_tool(QUEUE)
def make_dance_template(name: str, song: str, routine: str = "S1,S2", loop: bool = True, seconds: float = 10.0,
                        start: float | None = None, night: bool = False) -> dict:
    """Build a dance template on a done library song's beats (a detached Blender worker; poll show_template).
    `routine`: routines in order (S1 shuffle + kicks, S2 points + step-touch, A locking, B groove, F finale; add ~ for
    the mirrored routine). `loop`: a seamless loop of ~`seconds` (default) or a dance from the song's second `start`."""
    from . import shots
    return _ok(template=shots.start_dance(name, song, routine, loop, seconds, None if loop else start, night))


@_tool(DESTRUCTIVE)
def delete_template(name: str, confirm: bool = False) -> dict:
    """Delete one of the user's templates (built-ins stay). ONLY when the user asked for this delete; needs
    confirm=true."""
    from . import shots
    _confirm(confirm, f"template {name!r}")
    return _ok(**shots.delete(name))


# ---------------------------------------------------------------- songs and films
@_tool(READ)
def list_songs() -> dict:
    """The song library (name, status, bpm, seconds, tags, file, measured beats)."""
    from . import song
    return _ok(songs=song.list_songs())


@_tool(READ)
def show_song(name: str) -> dict:
    """One song: status (queued, running, done, failed...), progress, file, beats."""
    from . import song
    return _ok(song=song.load(name))


@_tool(QUEUE)
def make_song(name: str, tags: str, lyrics: str = "[instrumental]", bpm: int = 120, seconds: float = 30.0,
              key: str = "E minor", seed: int = 31, language: str = "en") -> dict:
    """Queue an original song (ACE-Step 1.5; a GPU job). `tags`: style, instruments, mood (traits only, never song or
    artist names); the tempo goes in `bpm`. For a film: ONE instrumental score of the whole film length
    (cinestudio://agents section 8). Returns at once; poll show_song at most once a minute; then song_beats."""
    from .webapi import songs as ws
    return _ok(song=ws._song_new({"name": name, "tags": tags, "lyrics": lyrics, "bpm": bpm, "seconds": seconds,
                                  "key": key, "seed": seed, "language": language}))


@_tool(READ)
def song_beats(name: str, bpm: float | None = None, start: float = 0.0, fps: float = 24.0) -> dict:
    """Measured beat and downbeat times of a song (or a sound file), and as shot frames (`start` = the song's second on
    frame 1). Time a dance on the MEASURED beats, not the asked bpm."""
    from . import song
    p = Path(name)
    if p.is_file():
        bpm = bpm or 120.0
    else:
        s = song.load(name)
        p, bpm = Path(s["file"]), bpm or s.get("bpm") or 120.0
    b = song.beats(p, bpm)
    return _ok(**b, beat_frames=song.frames(b["beats"], start, fps), downbeat_frames=song.frames(b["downbeats"], start,
                                                                                                   fps))


@_tool(CHANGE)
def song_loop(name: str, beats: int = 16, frames: int = 240, fps: float = 24.0, start: float | None = None) -> dict:
    """Cut a seamless loop of `beats` beats from a song, stretched to exactly `frames` frames (CPU, seconds)."""
    from . import song
    return _ok(**song.loop(name, beats, frames, fps, start))


@_tool(CHANGE)
def place_song(blend: str, name: str, out: str, start: float = 0.0, file: str | None = None) -> dict:
    """A copy of a .blend (`out`) with the song as its sound (`start` = the song's second on the shot's first frame;
    `file` = a sound file of this song instead, e.g. a loop). The run then uses it."""
    from . import song
    return _ok(**song.place(blend, name, start, out, sound_file=file))


@_tool(CHANGE)
def assemble_film(name: str, scenes: list[str], score: str | None = None, score_file: str | None = None,
                  score_start: float = 0.0, music_db: float = -8.0, scene_db: float = 0.0, duck: bool = True,
                  fade_out: float = 2.0, loudness: float = -16.0, replace: bool = False) -> dict:
    """Join scenes (run IDs or video files, in film order) with hard cuts and lay ONE score under the whole film (a
    library song `score` or a `score_file`; it never restarts at a cut), ducked under strong effects, one loudness
    normalization. CPU (ffmpeg): it blocks until the film is made. `replace` makes an existing film again: only when
    the user asked. Returns the film file and its cue sheet."""
    from . import film
    return _ok(**film.assemble(name, scenes, score=score, score_file=score_file, score_start=score_start,
                               music_db=music_db, scene_db=scene_db, duck=duck, fade_out=fade_out, loudness=loudness,
                               replace=replace))


@_tool(READ)
def list_films() -> dict:
    """The films made so far (film assemble)."""
    from . import film
    return _ok(films=film.list_films())


# ---------------------------------------------------------------- videos
# run new options that new_video_run takes in `advanced` (the CLI's names; the rest are its own parameters)
ADVANCED_RUN_OPTIONS = ("look_extra", "relay", "relay_strength", "clip_text", "look_keyframes", "prop_keyframes",
                        "pose_strength",
                        "pose_body", "adain", "adain_ref", "chain_light", "chain_sharpen", "chain_sharpen_mode",
                        "chain_sharpen_target", "chain_colour", "look_chain", "clip_frames", "proxy", "negative",
                        "steps", "guidance", "profile", "character_collection", "rig", "character2_collection",
                        "character2_rig", "allow_external", "target_text")


def _prop_options(props: dict | None, prop_images: dict | None, prop_refs: dict | None) -> dict:
    """new_video_run: the prop words and pictures with the library props of `prop_refs` filled in (own ones win)."""
    if prop_refs:
        from . import props as _props
        props, prop_images = _props.merge_refs(prop_refs, props, prop_images)
    return {"props": props or None, "prop_images": prop_images or None}


@_tool(QUEUE)
def new_video_run(blend: str, character_name: str, video: str, style: str = "", style_lora: str | None = None,
                  quality: str = "preview", seed: int | None = None, style_lora_strength: float | None = None,
                  control_video: str | None = None, describe_facing: bool | None = None,
                  body_type: str | None = None, sound: str | None = None, song: str | None = None,
                  song_start: float = 0.0, through_clip: int | None = None, style_preset: str | None = None,
                  video_model: str | None = None, background: str | None = None,
                  background_text: str | None = None, background_image: str | None = None,
                  background_seed: int | None = None, background_ground: str | None = None,
                  look_scene: str | None = None, look_text: str | None = None, character2: str | None = None,
                  characters: list[str] | None = None,
                  spacing: float | None = None, clip_videos: dict[str, str] | None = None,
                  sound_text: str | None = None, audio_relay: str | None = None,
                  ingredients_strength: float | None = None, note: str | None = None,
                  props: dict[str, str] | None = None, prop_images: dict[str, str] | None = None,
                  prop_refs: dict[str, str] | None = None, clip_context: str | None = None,
                  character_words: str | None = None,
                  advanced: dict | None = None) -> dict:
    """Freeze a Blender template (`blend`: its .blend or its library name) and a READY library character into a new
    layout video run, queue it and start its worker (the GPU work waits in the shared queue). Returns the run id at
    once; poll run_status at most once a minute. `video`: one paragraph of present-tense prose (style, place and light,
    the character by 2-3 visible marks, the action in time order; no camera words for a template). Shots longer than
    5 s become more clips that are joined: every clip gets ONE PROMPT OF ITS OWN = `video` (the SHARED context: who,
    place, light, look, lasting states; no actions of single clips) + the clip context + that clip's ACTIONS:
    `clip_videos` {"2": "..."} (only what ITS frames show), else the template's clip texts (template_clip_texts).
    `clip_context`: words for every clip instead of the template's ("" = none); `character_words`: what "{character}"
    in the template's texts becomes (default "the character"). The result's `prompt_warnings` names clips without
    their own actions and actions of single clips in `video`: fix them before the run starts (cinestudio://agents 7.2). `quality`: preview (default: stage 1, 768x448, about half the time; use it for tests) or final (a chosen
    demo). `through_clip` K: stop ("paused") after clip K for the clip gauntlet (cinestudio://agents 7.5).
    `style_preset`: a library style (its words and video LoRA); without `style` and `style_preset` the run takes the
    character's style. `background`: "saved:NAME" (a panorama or backplate from the library), "text" (a new panorama
    from `background_text`), "plate" (a backplate from `background_text` or the user's `background_image`), "none" (the
    template's own set); default: the template's default background when it has one. `look_scene`: no panorama, the
    place in words is put into the look image. `look_text`: the costume in exact words when the description and the
    sheet disagree. `characters`: SEVERAL ready characters, one per template slot in slot order (the list's "slots";
    the first = `character_name`; at most 4; the same character may fill two slots): one group sheet, one look crop
    per figure, keyframe looks off; on a one-mannequin template the mannequin is copied for each (`spacing` m apart).
    `character2`: the old form of a two-item `characters`. `sound`: auto (the shot's own sound) or ltx; `song`: a
    library song from its second `song_start`; `sound_text`: an audible sound sentence for LTX (never "quiet"); `audio_relay` carry|off.
    `video_model`: distilled (the supported LTX-2.5 INT8 video model). `describe_facing`,
    `body_type` chibi|standard, `control_video`, `ingredients_strength`: as in `run new`. `note`: what this run tests.
    `props` {"Ball_*": "red and white beach balls"}: the run's own words for the template's prop groups (the groups:
    template_layers / the list's "props"; only the given ones change, the others keep the template's words).
    `prop_images` {"Ball_*": "C:/.../ball.png"}: a picture of a template prop (at most 2; a prop on black in the
    characters' style): frozen into the run and drawn into a prop panel of the run's sheet with a "**Prop:**" line
    (its words: `props` or the template's). Not A/B-tested yet; the look image still shows the Blender prop's colour.
    `prop_refs` {"Ball_*": "dodgeball"}: a LIBRARY prop per prop group (list_props): its front becomes the group's
    prop picture and its words the group's words (`props` / `prop_images` for the same pattern win).
    `advanced`: other `run new` options by their CLI names (look_chain, chain_light, adain, adain_ref, relay_strength,
    clip_frames, proxy, negative, steps, guidance, pose_strength, ...)."""
    from . import cli, shots
    adv = dict(advanced or {})
    bad = set(adv) - set(ADVANCED_RUN_OPTIONS)
    if bad:
        raise ValueError(f"advanced: unknown options {sorted(bad)}; allowed: {list(ADVANCED_RUN_OPTIONS)}")
    _check_cli(("run", "new"), quality=quality, sound=sound, body_type=body_type, video_model=video_model,
               background_ground=background_ground, audio_relay=audio_relay, **adv)
    if characters:
        characters = [str(c).strip() for c in characters]
        if character2:
            raise ValueError("give characters (every slot) or character2, not both")
        if characters[0] != character_name:
            raise ValueError(f"characters[0] must be character_name ({character_name!r}): the list is every slot")
    if style_lora:
        cli.check_style_lora(style_lora)
    blend = _template_blend(blend)
    sheet, text = character.for_run(character_name)
    bg, bg_default = shots.run_background(blend, background, background_text, background_image, look_scene)
    c2 = {"name": character2, "collection": adv.pop("character2_collection", None),
          "rig": adv.pop("character2_rig", None)} if character2 else None
    rows = [{"name": nm} for nm in characters] if characters and len(characters) > 1 else None
    allow_external = bool(adv.pop("allow_external", False))
    profile = adv.pop("profile", None)
    clips = {}
    for k, t in (clip_videos or {}).items():
        if not str(k).strip().isdigit() or int(k) < 1:
            raise ValueError(f"clip_videos: {k!r} is not a clip number (1, 2, 3 ...)")
        if str(t or "").strip():
            clips[str(int(k))] = str(t).strip()
    options = {"quality": quality, "seed": seed, "style_lora_strength": style_lora_strength,
               "character_name": character_name, "control_video": control_video, "describe_facing": describe_facing,
               "body_type": body_type, "sound": sound, "song": song, "song_start": song_start,
               "style_preset": style_preset, "video_model": video_model, "background": bg,
               "background_text": background_text, "background_image": background_image,
               "background_seed": background_seed, "background_ground": background_ground, "look_scene": look_scene,
               "look_text": look_text, "character2": c2, "characters": rows, "spacing": spacing,
               "sound_text": sound_text,
               "audio_relay": audio_relay, "ingredients_strength": ingredients_strength,
               "clip_videos": clips or None, "clip_context": clip_context, "character_words": character_words,
               **_prop_options(props, prop_images, prop_refs), **adv}
    m = runner.new_run(blend, sheet, text, video, style, style_lora, False, None, allow_external, profile=profile,
                       **{k: v for k, v in options.items() if v is not None})
    if note:
        runner.set_note(m["id"], note)
    if through_clip:
        runner.set_clip_limit(m["id"], through_clip)
    workqueue.add("video", m["id"], runner.queue_label(m))
    started = cli._start_detached(m["id"])
    if not started.get("ok"):          # the run exists, but its worker did not start: say so
        return {"ok": False, "id": m["id"], "start": started,
                "error": f"run {m['id']} was made but its worker did not start: {started.get('error')}"}
    return _ok(id=m["id"], start=started, clip_sources=m.get("clip_sources"),
               prompt_warnings=m.get("prompt_warnings") or [],
               **({"background": f"saved:{bg_default} (the template's default)"} if bg_default else {}))


@_tool(READ)
def run_status(run_id: str) -> dict:
    """A video run: status, steps, clip limit, takes, and the path of final.mp4 when it is done. Poll at most once a
    minute."""
    return _ok(**runner.status(run_id))


@_tool(READ)
def run_details(run_id: str) -> dict:
    """What each clip of a run was given (the full prompt, negative, sheet, first-frame image, main settings), the
    run's note, its library style and style warnings, and whether it is retired (an old LTX-2.3 run: it can be
    watched, not continued, retaken or forked)."""
    m = runner._read_json(runner.run_dir(run_id) / "manifest.json")
    note = runner.run_dir(run_id) / "note.json"
    return _ok(details=runner.run_details(run_id), retired=runner.retired(m), style_ref=m.get("style_ref"),
               style_warnings=m.get("style_warnings") or [], prompt_warnings=m.get("prompt_warnings") or [],
               note=json.loads(note.read_text(encoding="utf-8")) if note.is_file() else None)


@_tool(READ)
def run_events(run_id: str, since: int = 0, limit: int = 200, progress: bool = False) -> dict:
    """A run's events from number `since` on (at most `limit`; progress events only with progress=true) and `next`,
    the number to pass as `since` next time."""
    from . import cli
    lines = cli._complete_lines(runner.run_dir(run_id) / "events.jsonl")
    evs = [json.loads(x) for x in lines[since:]]
    if not progress:
        evs = [e for e in evs if e.get("event") != "progress"]
    return _ok(id=run_id, events=evs[-limit:] if limit else evs, next=len(lines))


@_tool(READ)
def list_runs(limit: int = 20) -> dict:
    """The latest video runs."""
    return _ok(runs=runner.list_runs(limit))


@_tool(QUEUE)
def continue_run(run_id: str, through_clip: int = 0) -> dict:
    """Start a stopped, paused or never-started run again (it waits in the queue). `through_clip` K > 0: stop after
    clip K; 0: no limit (the rest of the clips and the final video). A retired run is refused."""
    from . import cli
    runner.check_runnable(run_id)                  # an old LTX-2.3 run: a clear refusal, no queue ticket
    runner.set_clip_limit(run_id, through_clip or None)
    return cli._start_detached(run_id)


@_tool(QUEUE)
def fork_run(run_id: str, from_clip: int, video: str | None = None, style: str | None = None,
             describe_facing: bool | None = None, look_chain: str | None = None, chain_colour: str | None = None,
             chain_light: str | None = None, chain_sharpen_mode: str | None = None,
             chain_sharpen_target: float | None = None, adain: float | None = None, adain_ref: str | None = None,
             keyframe: str | None = None, relay_strength: float | None = None, clip_text: str | None = None,
             audio_relay: str | None = None, chain_light_ref: str | None = None, chain_grade: str | None = None,
             props: dict[str, str] | None = None,
             prop_images: dict[str, str] | None = None, clip_videos: dict[str, str] | None = None,
             note: str | None = None) -> dict:
    """"Build on this video": a NEW run that keeps clips 1..from_clip-1 of a run and makes the rest again with the given
    changes (queued and started; the original run is not touched). The fork keeps each clip's own actions; `video`
    changes the shared text; `clip_videos` {"3": "..."} gives the clips it makes new actions ("" = none). `props` {pattern: words}: new prop words (""
    = the template's words again). `prop_images` {pattern: picture}: new prop pictures ("" removes the parent's; the
    fork's sheet is drawn again). Returns the new run id; poll run_status."""
    from . import cli
    _check_cli(("run", "fork"), look_chain=look_chain, chain_colour=chain_colour, chain_light=chain_light,
               chain_light_ref=chain_light_ref, chain_grade=chain_grade, chain_sharpen_mode=chain_sharpen_mode, adain_ref=adain_ref, keyframe=keyframe, clip_text=clip_text)
    _check_cli(("run", "new"), audio_relay=audio_relay)
    m = runner.fork_run(run_id, from_clip, note, video=video, style=style, describe_facing=describe_facing,
                        look_chain=look_chain, chain_colour=chain_colour, adain=adain, adain_ref=adain_ref,
                        keyframe=keyframe, relay_strength=relay_strength, clip_text=clip_text, chain_light=chain_light,
                        chain_light_ref=chain_light_ref, chain_grade=chain_grade, chain_sharpen_mode=chain_sharpen_mode, chain_sharpen_target=chain_sharpen_target,
                        audio_relay=audio_relay, props=props, prop_images=prop_images, clip_videos=clip_videos)
    started = cli._start_detached(m["id"])                 # it also takes its place in the queue
    if not started.get("ok"):
        return {"ok": False, "id": m["id"], "forked_from": m["forked_from"], "start": started,
                "error": f"run {m['id']} was made but its worker did not start: {started.get('error')}"}
    return _ok(id=m["id"], forked_from=m["forked_from"], start=started)


@_tool(QUEUE)
def retake_clip(run_id: str, clip: int, seed: int | None = None, video: str | None = None,
                ingredients_strength: float | None = None, relay_strength: float | None = None,
                adain: float | None = None, keyframe: str | None = None, end_anchor: str | None = None,
                anchor_strength: float | None = None, note: str | None = None, start: bool = True,
                advanced: dict | None = None) -> dict:
    """Layout runs: make clip `clip` again with ONE change (a new take beside the original; it starts from the
    previous clip's saved latents). The run's worker makes it at once (start=false: only record it); the video is not
    changed until use_take. keyframe / end_anchor: "on" or "off". `advanced`: other retake fields by their names
    (style, character, negative, style_lora, style_lora_strength, refine_sigmas, look_image, keyframe_image,
    keyframe_frame, keyframe_images [[file, frame], ...], look_chain, chain_sharpen, chain_sharpen_mode,
    chain_sharpen_target, chain_light, chain_colour, adain_ref, pose_strength, pose_body, pose_stage2 on|off). First
    2-3 seeds, then one knob; after ~4 takes without a pass, stop and report."""
    from . import cli
    adv = dict(advanced or {})
    changes = {"seed": seed, "video": video, "ingredients_strength": ingredients_strength,
               "relay_strength": relay_strength, "adain": adain, "keyframe": keyframe, "end_anchor": end_anchor,
               "anchor_strength": anchor_strength}
    clash = set(adv) & {k for k, v in changes.items() if v is not None}
    if clash:
        raise ValueError(f"give {sorted(clash)} once (as a parameter or in advanced)")
    acts = _cli_actions(("run", "retake"))
    _check_cli(("run", "retake"), **{k: v for k, v in {**changes, **adv}.items()
                                     if k in acts and not (k == "pose_stage2" and isinstance(v, bool))})
    if isinstance(adv.get("pose_stage2"), str):
        adv["pose_stage2"] = adv["pose_stage2"] == "on"
    r = runner.add_take(run_id, clip, {**changes, **adv}, note)
    if start:
        r["start"] = cli._start_detached(run_id)
    return _ok(**r)


@_tool(READ)
def list_takes(run_id: str) -> dict:
    """The takes of a run per clip (settings, status, files, scores) and which take each clip uses."""
    return _ok(**runner.list_takes(run_id))


@_tool(QUEUE)
def use_take(run_id: str, clip: int, take: int) -> dict:
    """Put take `take` into the video for clip `clip` (the original stays as take 0; the previous final video is
    kept) and join the clips again. Only while the run's worker is stopped (paused or done)."""
    return _ok(**runner.use_take(run_id, clip, take))


@_tool(READ)
def score_clip(run_id: str, clip: int, take: int | None = None, motion: bool = True,
               critic_dir: str | None = None, tag: str | None = None) -> dict:
    """The gauntlet numbers of one clip or take: shared-frame agreement with the previous clip (overlap), sharpness,
    flicker, motion follow vs the rig (DWPose, ~1-2 min on the CPU). critic_dir + tag: also the blind critic's
    material (docs/gauntlet_critic.md; keep the tag -> take map outside critic_dir)."""
    from . import clip_score
    return _ok(**clip_score.score(run_id, clip, take, with_motion=motion, critic_dir=critic_dir, tag=tag))


@_tool(DESTRUCTIVE)
def cancel_run(run_id: str) -> dict:
    """Cancel a video run. ONLY when the user asked for this cancel: it stops the user's work."""
    return _ok(**runner.request_cancel(run_id))


@_tool(DESTRUCTIVE)
def delete_run(run_id: str, confirm: bool = False) -> dict:
    """Delete a run and its video for good (a run with a live worker is refused). ONLY when the user asked for this
    delete in this conversation; needs confirm=true."""
    _confirm(confirm, f"run {run_id}")
    return _ok(**runner.delete_run(run_id))


# ---------------------------------------------------------------- workflows (the user's copies of the built-in graphs)
@_tool(READ)
def list_workflows(name: str | None = None) -> dict:
    """Without `name`: every built-in ComfyUI graph (group, features that use it, model family) with the user's copies
    and which graph new jobs use. With `name`: that graph's copies and the active choice (None = the built-in)."""
    from . import user_workflows as uw
    if name:
        return _ok(copies=uw.list_copies(name), active=uw.active_choice(name))
    from .webapi import workflows as ww
    rows = [{**ww._summary(ww.WORKFLOWS / f"{n}_api.json", n), **ww._copies_info(n)}
            for n in ww._graph_names(ww.WORKFLOWS)]
    order = {g: i for i, g in enumerate(ww.GROUPS)}
    rows.sort(key=lambda r: (order.get(r["group"], 99), r["name"]))
    return _ok(workflows=rows)


@_tool(READ)
def show_workflow(name: str, copy: str | None = None) -> dict:
    """A built-in graph (its nodes, the inputs the job fills itself) or one of the user's copies (its graph, the values
    that differ from the built-in, problems). Use the node ids and input names with set_workflow_value."""
    from . import user_workflows as uw
    if copy:
        from .webapi import workflows as ww
        v = ww._copy_view(name, copy)
        return _ok(**{k: v[k] for k in ("builtin", "id", "meta", "graph", "custom", "bound", "problems", "active",
                                        "features")})
    return _ok(name=name, graph=uw.builtin_graph(name), bound=sorted(uw.bound_inputs(uw.builtin_bind(name))),
               copies=uw.list_copies(name), active=uw.active_choice(name))


@_tool(CHANGE)
def copy_workflow(name: str, label: str | None = None) -> dict:
    """Make the user's copy of a built-in graph (the built-in is only read). New jobs use it only after use_workflow."""
    from . import user_workflows as uw
    return _ok(copy=uw.make_copy(name, label))


@_tool(CHANGE)
def set_workflow_value(name: str, copy: str, node: str, input: str, value: Any) -> dict:
    """Set one literal value in a copy (inputs the job fills itself and links are refused). Changes what every new job
    of that graph does once the copy is in use: tell the user what you changed."""
    from . import user_workflows as uw
    return _ok(**uw.save_values(name, copy, {node: {input: value}}))


@_tool(CHANGE)
def reset_workflow(name: str, copy: str, node: str | None = None) -> dict:
    """The built-in's values again in a copy (one node, or all)."""
    from . import user_workflows as uw
    return _ok(**uw.reset_values(name, copy, node))


@_tool(CHANGE)
def use_workflow(name: str, copy: str) -> dict:
    """Which graph NEW jobs use: a copy id, or "builtin". Running jobs and frozen runs keep their graphs."""
    from . import user_workflows as uw
    return _ok(**uw.set_choice(name, None if copy == "builtin" else copy))


def serve() -> None:
    mcp.run("stdio")
