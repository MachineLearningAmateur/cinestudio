"""The background library (the Backgrounds tab): 360 panoramas and backplates (one picture for a still camera); list,
new from text, import a picture (backplate), 360 depth and style redraw (panoramas), delete.

GPU work runs as detached `cinestudio background ...` workers in the shared queue (kind "lab", refs "bg-..."): the app
adds the ticket in request order, the worker claims it (lab.turn), so an app restart does not stop a job."""
from __future__ import annotations

import json
import secrets
import sys
import time
from pathlib import Path

from aiohttp import web

from .. import cli, lab, runner, spawn, styles, workqueue
from .common import _bg, _float_or_none, _form, _int_or_none, _ok, file_url, routes

KINDS = {"new": "new panorama", "plate": "new backplate", "depth": "360 depth", "style": "style redraw",
         "setplate": "backplate over the set"}     # no hyphen in a kind: _jobs() splits the ref on "-"
NEW_KINDS = ("new", "plate", "setplate")             # jobs that make a NEW background (a card of their own)


def _jobs() -> list[dict]:
    """The queue tickets of background jobs (waiting or running), first in line first. Only our own tickets: the tab
    polls this every few seconds, and workqueue.view() would also ask ComfyUI for its queue each time."""
    out = []
    for i, t in enumerate(workqueue.tickets()):
        ref = str(t.get("ref", ""))
        if t.get("kind") != "lab" or not ref.startswith("bg-") or ref.count("-") < 2:
            continue
        _, kind, name = ref.split("-", 2)
        status = "running" if i == 0 and t.get("pid") else "queued"
        if lab.canceled(ref):                             # Stop was pressed: the worker ends after this step
            status = "stopping"
        out.append({"ref": ref, "label": t.get("label") or ref, "position": i, "kind": kind, "name": name,
                    "status": status})
    return out


def _runs_using() -> dict[str, int]:
    """How many runs used each saved panorama (their manifests: background "saved:NAME", or the library name a text
    background was saved under)."""
    count: dict[str, int] = {}
    for d in runner.runs_dir().glob("r*"):
        try:
            m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        names = set()
        bgm = m.get("background") or ""
        if isinstance(bgm, str) and bgm.startswith("saved:"):
            names.add(bgm[6:])
        try:
            st = json.loads((d / "state.json").read_text(encoding="utf-8"))
            step = (st.get("steps") or {}).get("background") or {}
            lib = step.get("library") or (step.get("outputs") or {}).get("library")
            if lib:
                names.add(lib)
        except (OSError, ValueError):
            pass
        for n in names:
            count[n] = count.get(n, 0) + 1
    return count


def _list() -> dict:
    from .. import background, pano_depth
    background.sweep_reserved()                           # empty folders of killed style passes free their names
    used = _runs_using()
    jobs = _jobs()
    out = []
    for r in background.list_backgrounds():
        f = r["file"]
        depth = False
        if r["kind"] == "pano":
            try:
                depth = pano_depth.load_if_current(f, runner.file_sha(f)) is not None
            except OSError:
                depth = False
        out.append({k: r.get(k) for k in ("name", "kind", "scene", "prompt", "style", "seed", "created", "styled_from",
                                          "style_lora", "keep", "shot", "style_ref", "style_version")}
                   | {"builtin": bool(r.get("builtin"))}                 # ships with cinestudio: read-only
                   | {"style_stale": styles.is_stale(r)}                 # the card's style chip (Styles tab)
                   | {"url": file_url(f), "depth": depth, "runs": used.get(r["name"], 0),
                      "camera": background.camera_summary(r.get("camera")),
                      "source": r.get("source") or ("redraw" if r.get("styled_from") else "text"),
                      "set_url": file_url(Path(f).parent / "set.png") if r.get("source") == "set" else None,
                      "busy": [j for j in jobs if j.get("name") == r["name"]]})
    return {"backgrounds": out, "jobs": [j for j in jobs if j["kind"] in NEW_KINDS],
            "failed": [f for f in background.failed_jobs() if not any(j["ref"] == f["ref"] for j in jobs)]}


@routes.get("/api/backgrounds")
async def backgrounds_list(request: web.Request):
    """The saved 360 panoramas, newest first, with a picture URL, the depth state, the use count and running jobs.
    `?flat=1` gives the old flat list (the Video form's choice list)."""
    d = await _bg(_list)
    if request.query.get("flat"):
        return _ok([{k: b.get(k) for k in ("name", "kind", "scene", "seed", "created", "url", "camera")}
                    for b in d["backgrounds"]])
    return _ok(d)


def _start(kind: str, name: str | None, args: list[str], label: str) -> dict:
    """A detached `cinestudio background ...` worker with a queue ticket made now (request order)."""
    from .. import background
    ref = f"bg-{kind}-{name or secrets.token_hex(3)}"
    log = background.library_dir() / ".work" / f"{ref}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if not any(j["ref"] == ref for j in _jobs()):
        background.dismiss_job(ref)                       # an old failure record of the same job goes
    try:
        workqueue.add_new("lab", ref, label)              # checked and added under one lock (a double click)
    except workqueue.QueueBusy as exc:
        raise background.BackgroundError(f"a {KINDS[kind]} job for {name} is already in the queue") from exc
    try:
        pid, independent = spawn.detached([sys.executable, "-m", "cinestudio", "background", *args, "--ref", ref], log,
                                          cli.ROOT)
    except BaseException:
        workqueue.done("lab", ref)
        raise
    time.sleep(0.3)
    return {"ref": ref, "pid": pid, **({} if independent else {"warning": "the worker may stop when this app stops"})}


def _new(f: dict) -> dict:
    from .. import background
    scene = (f.get("scene") or "").strip()
    background.prompt_for(scene)                          # refuses an empty place
    name = (f.get("name") or "").strip() or None
    if name and (not background.NAME_RE.match(name) or background.taken(name)):
        raise background.BackgroundError(f"name {name!r}: letters, digits, '-' and '_', and not taken")
    seed = _int_or_none(f.get("seed"))
    from .styles import preset_or_words                   # a library style: its words, and the background records it
    preset, style = preset_or_words(f.get("style_preset"), (f.get("style") or "").strip())
    args = ["new", f"--scene={scene}", f"--style={style}"] + ([f"--style-preset={preset}"] if preset else [])
    if seed is not None:
        args += ["--seed", str(seed)]
    if name:
        args += ["--name", name]
    if (f.get("kind") or "pano") == "plate":           # a backplate for the still camera of a template
        shot = (f.get("shot") or "").strip()
        if not shot:
            raise background.BackgroundError("choose the template whose still camera the backplate is made for")
        from pathlib import Path
        from .. import shots as _shots
        allowed = {str(Path((r.get("files") or {}).get("blend") or "").resolve()) for r in _shots.list_shots()}
        if str(Path(shot).resolve()) not in allowed or not Path(shot).is_file():
            raise background.BackgroundError("the shot must be a template's .blend")
        args[0] = "plate"
        args.append(f"--shot={shot}")
        return _start("plate", name, args, f"new backplate: {scene[:50]}")
    if f.get("depth") is False:
        args.append("--no-depth")
    return _start("new", name, args, f"new panorama: {scene[:50]}")


def _from_set(f: dict) -> dict:
    """A backplate drawn OVER a template's own Blender set (`background from-set`): Blender renders the set alone from
    the template's still camera, Qwen-Image-Edit redraws it as the place. The place words are optional: the default
    is the template's Set layer words. A detached worker with a queue ticket, like the plate kind."""
    from .. import background
    from .. import shots as _shots
    shot = (f.get("shot") or "").strip()
    if not shot:
        raise background.BackgroundError("choose the template whose set the backplate is drawn over")
    allowed = {str(Path((r.get("files") or {}).get("blend") or "").resolve()) for r in _shots.list_shots()}
    if str(Path(shot).resolve()) not in allowed or not Path(shot).is_file():
        raise background.BackgroundError("the shot must be a template's .blend")
    scene = (f.get("scene") or "").strip()
    if not scene:                                         # refused now, not after the wait in the queue
        rec = _shots.library_record(Path(shot)) or {}
        if not (((rec.get("layers") or {}).get("descriptions") or {}).get("set") or "").strip():
            raise background.BackgroundError("say what the set stands for (the place): this template has no Set "
                                             "layer words")
    name = (f.get("name") or "").strip() or None
    if name and (not background.NAME_RE.match(name) or background.taken(name)):
        raise background.BackgroundError(f"name {name!r}: letters, digits, '-' and '_', and not taken")
    from .styles import preset_or_words                   # a library style: its words, and the plate records it
    preset, style = preset_or_words(f.get("style_preset"), (f.get("style") or "").strip())
    args = ["from-set", f"--shot={shot}", f"--scene={scene}", f"--style={style}"] +         ([f"--style-preset={preset}"] if preset else [])
    seed = _int_or_none(f.get("seed"))
    if seed is not None:
        args += ["--seed", str(seed)]
    if name:
        args += ["--name", name]
    return _start("setplate", name, args, f"backplate over the set: {(scene or Path(shot).parent.name)[:44]}")


@routes.post("/api/backgrounds/from-set")
async def background_from_set(request: web.Request):
    return _ok(await _bg(_from_set, await request.json()))


def _check(name: str) -> None:
    from .. import background
    background.load(name)                                 # validates the name, refuses a missing background


def _check_pano(name: str) -> None:
    from .. import background
    background.require_pano(name)                         # depth and style redraw: 360 panoramas only


def _style(name: str, f: dict) -> dict:
    from .. import background
    _check_pano(name)
    lora = (f.get("style_lora") or "").strip()
    if not lora:
        raise background.BackgroundError("choose the style LoRA for the redraw")
    keep = _float_or_none(f.get("keep"))
    strength = _float_or_none(f.get("strength"))
    args = ["style", name, f"--style-lora={lora}", f"--style={(f.get('style') or '').strip()}"]
    if keep is not None:
        args += ["--keep", str(keep)]
    if strength is not None:
        args += ["--strength", str(strength)]
    return _start("style", name, args, f"panorama {name}: style redraw")


def _delete(name: str) -> dict:
    from .. import background
    _check(name)
    return background.delete(name)                        # refuses a built-in, and while a job of this panorama waits


@routes.post("/api/background-jobs/{ref}/dismiss")
async def background_job_dismiss(request: web.Request):
    """Hide a failed background job's card (its log stays in <library>/.work)."""
    from .. import background
    return _ok(await _bg(background.dismiss_job, request.match_info["ref"]))


@routes.post("/api/backgrounds/new")
async def background_new(request: web.Request):
    return _ok(await _bg(_new, await request.json()))


@routes.post("/api/backgrounds/import")
async def background_import(request: web.Request):
    """Your own picture of the place as a backplate (multipart: image, optional name and scene); instant."""
    from .. import background
    fields, files = await _form(request)
    try:                                                  # every uploaded file goes, whatever happens
        if set(files) - {"image"}:
            raise background.BackgroundError(f"unexpected file fields: {', '.join(sorted(set(files) - {'image'}))}")
        if "image" not in files:
            raise background.BackgroundError("choose a picture of the place")
        name = (fields.get("name") or "").strip() or None
        rec = await _bg(background.import_plate, files["image"], name, fields.get("scene") or "")
    finally:
        for f in files.values():
            f.unlink(missing_ok=True)
    return _ok({"background": {"name": rec["name"], "kind": rec["kind"]}})


@routes.post("/api/backgrounds/{name}/{action}")
async def background_action(request: web.Request):
    name, action = request.match_info["name"], request.match_info["action"]
    f = await request.json() if request.can_read_body else {}
    if action == "depth":
        await _bg(_check_pano, name)                      # a bad or missing name, or a backplate, is refused
        return _ok(await _bg(_start, "depth", name, ["depth", name], f"panorama {name}: 360 depth (MoGe-2)"))
    if action == "style":
        return _ok(await _bg(_style, name, f))
    if action == "delete":
        return _ok(await _bg(_delete, name))
    raise web.HTTPNotFound()
