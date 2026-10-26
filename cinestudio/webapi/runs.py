"""Video runs: the gallery, new runs, run details and media, forks, cancel and delete."""
from __future__ import annotations

import json
from pathlib import Path

from aiohttp import web

from .. import character, cli, runner, workqueue
from .common import _bg, _cleanup_canceled, _float_or_none, _form, _inside, _int_or_none, _ok, file_url, routes


@routes.get("/api/runs")
async def runs(request: web.Request):
    """?offset=N&limit=K (limit 0 = all): one page of the runs, newest first, plus the total number."""
    q = request.query
    offset = max(0, int(q.get("offset") or 0))
    limit = max(0, min(500, int(q.get("limit") or 24)))
    await _bg(_cleanup_canceled)
    total = runner.count_runs()
    rows = runner.list_runs(limit or None, offset)
    for r in rows:
        r["final_url"] = file_url(r.get("final"))
        cs = runner.runs_dir() / r["id"] / "contact_sheet.jpg"
        r["contact_url"] = file_url(cs) if cs.is_file() else None
        try:                                              # a stopped run's card offers Continue (not a spinner)
            r["worker_alive"] = runner.worker_alive(r["id"])
        except runner.RunError:
            r["worker_alive"] = False
    return _ok({"items": rows, "total": total, "offset": offset, "limit": limit})


@routes.post("/api/runs")
async def run_new(request: web.Request):
    f = await request.json()
    return _ok(await _bg(_run_new, f))


def _choice(f: dict, key: str, allowed) -> str | None:
    """A select value of the New video form: empty = the run default; anything else must be one of `allowed`."""
    v = (f.get(key) or "").strip()
    if not v:
        return None
    if v not in allowed:
        raise ValueError(f"{key}: {v!r} is not one of {', '.join(allowed)}")
    return v


def _song_args(f: dict, template: str | None) -> dict:
    """The New video form's song: a library song as the video's sound from its second `song_start` (Blender shots
    only; a video template keeps its own sound). Empty = none (the template's sound or LTX's)."""
    name = (f.get("song") or "").strip()
    if not name:
        return {}
    if template:
        raise ValueError("a song from the Music tab can be the sound of a Blender template only")
    start = _float_or_none(f.get("song_start"))
    return {"song": name, "song_start": start or 0.0}


MAX_RUN_CHARACTERS = runner.MAX_CHARACTERS                # runner: --character once per slot (4)


def _slot_characters(f: dict) -> dict:
    """The New video form's character SLOTS (2026-10-25, user: "if a video only has 1 character there should only be
    one character option, if it has three then three options"): `characters` [{"slot": 1, "collection": "Mannequin",
    "character": "kitsune"}, ...] -> slot 1 -> the form's `character` field, every slot (two or more) ->
    `slot_characters` [{"name", "collection"}] for runner.new_run's `characters`. Without `characters` the old
    fields stay as sent (character, character2, spacing). The same character may fill two slots."""
    rows = f.get("characters")
    if rows is None:
        return f
    if not isinstance(rows, list) or not rows:
        raise ValueError("characters must list one character per template slot")
    try:
        rows = sorted(rows, key=lambda r: int(r.get("slot")))
    except (TypeError, ValueError, AttributeError):
        raise ValueError("each character needs its slot number (1, 2, ...)") from None
    if [int(r["slot"]) for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("the character slots must be 1, 2, ... without gaps")
    if len(rows) > MAX_RUN_CHARACTERS:
        raise ValueError(f"This template has {len(rows)} characters; videos support up to {MAX_RUN_CHARACTERS} for now")
    for r in rows:
        if not str(r.get("character") or "").strip():
            raise ValueError(f"choose a character for Character {r['slot']}")
    out = {k: v for k, v in f.items() if k not in ("characters", "character", "character2", "spacing")}
    out["character"] = str(rows[0]["character"]).strip()
    if len(rows) > 1:
        out["slot_characters"] = [{"name": str(r["character"]).strip(),
                                   **({"collection": str(r["collection"]).strip()} if str(r.get("collection") or "")
                                      .strip() else {})} for r in rows]
    return out                                             # (the slots stand at the template's places: no spacing)


def _props_args(f: dict, template: str | None) -> dict:
    """The form's "Objects in this template": {pattern: the user's words}; empty fields = the template's words. And
    each row's picture (`prop_images` {pattern: a path the prop picture upload returned}): only pictures in the
    uploads folder (the browser cannot hand the server any file of the PC)."""
    raw = f.get("props") or {}
    if not isinstance(raw, dict):
        raise ValueError("props must map a prop pattern to its words")
    props = {str(k).strip(): " ".join(str(v or "").split()) for k, v in raw.items() if str(v or "").strip()}
    if props and template:
        raise ValueError("prop words are for Blender templates (a video template has no props)")
    pics = _prop_image_paths(f.get("prop_images"))
    if pics and template:
        raise ValueError("prop pictures are for Blender templates (a video template has no props)")
    props, pics = _prop_refs(f.get("prop_refs"), template, props, pics)
    return {**({"props": props} if props else {}), **({"prop_images": pics} if pics else {})}


def _prop_refs(refs, template: str | None, props: dict, pics: dict) -> tuple[dict, dict]:
    """The object rows' LIBRARY props (`prop_refs` {pattern: prop ID}, the Cast tab's props): each fills its pattern's
    words and picture (the prop's front, from the prop library); the row's own words or own picture win."""
    if not refs:
        return props, pics
    if template:
        raise ValueError("library props are for Blender templates (a video template has no props)")
    from .. import props as _props
    return _props.merge_refs(refs, props, pics)


def prop_uploads() -> Path:
    return runner.home() / "uploads" / "props"


def _prop_image_paths(raw, allow_empty: bool = False) -> dict:
    """{pattern: picture} from the app: each picture one the upload endpoint saved (empty = none, or in a fork: the
    parent's picture removed)."""
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("prop pictures must map a prop pattern to an uploaded picture")
    out = {}
    for k, v in raw.items():
        v = str(v or "").strip()
        if not v:
            if allow_empty:
                out[str(k).strip()] = ""
            continue
        if not _inside(Path(v), prop_uploads()):
            raise ValueError(f"the picture for {k!r} is not an uploaded prop picture (upload it again)")
        out[str(k).strip()] = v
    return out


@routes.post("/api/prop-images")
async def prop_image_upload(request: web.Request):
    """A prop picture for the New video form (multipart: image): checked (a picture with a prop in it, at most 40 MB)
    and kept in <home>/uploads/props until a run copies it -> {"path", "width", "height"}."""
    fields, files = await _form(request)
    try:
        if set(files) != {"image"}:
            raise ValueError("send one picture in the field 'image'")
        return _ok(await _bg(_keep_prop_upload, files["image"]))
    finally:
        for f in files.values():
            f.unlink(missing_ok=True)


def _keep_prop_upload(src: Path) -> dict:
    import shutil
    runner._check_prop_image(src, "the picture")
    img = runner._read_prop(src)
    dst = prop_uploads() / f"prop_{src.stem.split('_')[-1]}{src.suffix.lower()}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return {"path": str(dst), "width": int(img.shape[1]), "height": int(img.shape[0])}


def _group_args(f: dict, template: str | None) -> dict:
    """The New video form's characters beyond the first (layout runs on a Blender shot): `slot_characters` (every
    template slot, from _slot_characters) -> runner's `characters`; the old field `character2` (a second character on
    the shot's second mannequin or on a copy of the first one `spacing` m beside it) still works. Empty = one."""
    rows = f.get("slot_characters")
    name = (f.get("character2") or "").strip()
    if not rows and not name:
        return {}
    if template:
        raise ValueError("several characters need a Blender shot (a video template has one stand-in)")
    spacing = _float_or_none(f.get("spacing"))
    extra = {"spacing": spacing} if spacing is not None else {}
    if rows:
        return {"characters": rows, **extra}
    c2 = {"name": name, **({"collection": f["character2_collection"]} if f.get("character2_collection") else {})}
    return {"character2": c2, **extra}


def _clip_texts(f: dict) -> dict:
    """{clip number: ACTIONS} from the form's "Text per clip" rows (empty rows are left out: such a clip takes the
    template's clip text, else only the shared text; the form leaves out a row that still holds the template's text)."""
    raw = f.get("clip_videos") or {}
    if not isinstance(raw, dict):
        raise ValueError("clip texts must map clip numbers to texts")
    out = {}
    for k, v in raw.items():
        text = str(v or "").strip()
        if not text:
            continue
        if not str(k).strip().isdigit() or int(k) < 1:
            raise ValueError(f"clip text: {k!r} is not a clip number (1, 2, 3 ...)")
        out[str(int(k))] = text[:1500]
    return {"clip_videos": out} if out else {}


def _run_new(f: dict) -> dict:
    f = _slot_characters(f)
    blend = f.get("blend")
    if not blend:
        raise ValueError("choose a shot")
    template = None
    if str(blend).startswith("template:"):                   # a video template (made from text)
        template, blend = str(blend).split(":", 1)[1], None
    # the Background select's "describe the place in the look image" (`--look-scene`): no panorama or backplate; a
    # second Qwen edit puts the place into the look image (the place words, else the video text)
    look_scene = None
    if f.get("background") == "look-scene":
        if template:
            raise ValueError("the place in the look image is for Blender shots")
        look_scene = (f.get("background_text") or "").strip() or (f.get("video") or "").strip()
        f = {**f, "background": None, "background_text": None, "background_ground": None}
    sheet, text = character.for_run(f["character"])
    ctl = (f.get("control_video") or "").strip().strip('"') or None
    if ctl and (not Path(ctl).is_file() or Path(ctl).suffix.lower() not in (".mp4", ".mov", ".webm", ".mkv")):
        raise ValueError(f"control video not found (an .mp4, .mov, .webm or .mkv file): {ctl}")
    follow = f.get("follow_render", True) not in (False, "false", "off", "0", "")
    from .styles import preset_or_words                     # the Style select fills the text; changed text = own words
    preset, style = preset_or_words(f.get("style_preset"), f.get("style", ""))
    m = runner.new_run(blend, sheet, text, f["video"], style, f.get("style_lora") or None,
                       bool(f.get("restyle")), None, quality=f.get("quality") or None, seed=_int_or_none(f.get("seed")),
                       style_lora_strength=_float_or_none(f.get("style_lora_strength")), character_name=f["character"],
                       ingredients_strength=_float_or_none(f.get("ingredients_strength")),
                       control_strength=_float_or_none(f.get("control_strength")),
                       control_video=ctl,
                       source_denoise=_float_or_none(f.get("source_denoise")) if follow else False,
                       describe_facing=bool(f.get("describe_facing")) or None,
                       body_type=f.get("body_type") or None, sound=f.get("sound") or None, template=template,
                       **_song_args(f, template),
                       background=(f.get("background") or None) if not template else None,
                       background_text=(f.get("background_text") or "").strip() or None,
                       negative=(f.get("negative") or "").strip() or None,
                       steps=_int_or_none(f.get("steps")), guidance=_float_or_none(f.get("guidance")),
                       look_text=(f.get("look_text") or "").strip() or None,
                       sound_text=(f.get("sound_text") or "").strip() or None,
                       background_ground=_choice(f, "background_ground", runner.BACKGROUND_GROUNDS),
                       proxy=_choice(f, "proxy", ("auto", "off")),
                       look_keyframes=_choice(f, "look_keyframes", ("auto", "off")), style_preset=preset,
                       video_model=_choice(f, "video_model", tuple(runner.VIDEO_MODELS)),
                       audio_relay=_choice(f, "audio_relay", ("carry", "off")), look_scene=look_scene,
                       **_group_args(f, template), **_clip_texts(f), **_props_args(f, template))
    note = (f.get("note") or "").strip()
    if note:
        runner.set_note(m["id"], note[:500])
    through = _int_or_none(f.get("through_clip"))
    if through:                                              # the clip gauntlet: stop ("paused") after clip K
        runner.set_clip_limit(m["id"], through)
    workqueue.add("video", m["id"], runner.queue_label(m))
    started = cli._start_detached(m["id"])
    return {"id": m["id"], "start": started, "prompt_warnings": m.get("prompt_warnings") or []}


@routes.get("/api/runs/{id}")
async def run_show(request: web.Request):
    rid = request.match_info["id"]
    st = runner.status(rid)
    d = runner.run_dir(rid)
    lines = cli._complete_lines(d / "events.jsonl")
    evs = [json.loads(x) for x in lines]
    prog = next((e for e in reversed(evs) if e["event"] == "progress"), None)
    st["events_tail"] = [e for e in evs if e["event"] != "progress"][-12:]
    st["progress"] = prog
    started = next((e["t"] for e in reversed(evs) if e["event"] == "step_started"), None)
    st["step_started_at"] = started
    st["now"] = __import__("time").time()
    st["final_url"] = file_url(st.get("final"))
    cs = d / "contact_sheet.jpg"
    st["contact_url"] = file_url(cs) if cs.is_file() else None
    st["manifest"] = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    st["retired"] = runner.retired(st["manifest"])          # an old LTX-2.3 run: no Continue, Retake or Build on
    try:
        det = runner.run_details(rid)
        det["reference_url"] = file_url(det["reference_image"])
        det["sheet_url"] = file_url(det["sheet_file"])
        for c in det["clips"]:
            c["first_url"] = file_url(c["first_image"])
        det["style_ref"] = st["manifest"].get("style_ref")                 # the library style, and assets that name
        det["style_warnings"] = st["manifest"].get("style_warnings") or []  # another one (warnings only)
        det["prompt_warnings"] = st["manifest"].get("prompt_warnings") or []   # one prompt per clip (AGENTS.md 7.2)
        st["details"] = det
    except (OSError, ValueError, KeyError) as exc:      # an unreadable old run still opens
        st["details"] = {"error": str(exc)}
    return _ok(st)


@routes.post("/api/runs/{id}/media")
async def run_media_api(request: web.Request):
    """Browser copies of the run's Blender render and control (made once in the background; poll until "ready")."""
    from .. import shot_preview
    r = await _bg(shot_preview.run_media, request.match_info["id"])
    for k in ("render", "control"):
        if r.get(k):
            r[k]["url"] = file_url(r[k].get("path"))
    return _ok(r)


@routes.post("/api/runs/{id}/fork")
async def run_fork(request: web.Request):
    """A new run that keeps clips 1..from_clip-1 and makes the rest again with a new text, style or facing text."""
    f = await request.json()
    return _ok(await _bg(_run_fork, request.match_info["id"], f))


def _fork_clip_texts(f: dict) -> dict | None:
    """{clip number: ACTIONS} for the clips a fork makes again (the same rules as a new run's; "" = no actions of its
    own); None when the form sends none. The CLI and MCP forward theirs too (Codex review 2026-10-25)."""
    raw = f.get("clip_videos")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("clip texts must map clip numbers to texts")
    out = {}
    for k, v in raw.items():
        if not str(k).strip().isdigit() or int(k) < 1:
            raise ValueError(f"clip text: {k!r} is not a clip number (1, 2, 3 ...)")
        if v is not None and not isinstance(v, str):
            raise ValueError(f"the text of clip {k} must be text")
        out[str(int(k))] = (v or "").strip()[:1500]
    return out or None


def _run_fork(run_id: str, f: dict) -> dict:
    pics = _prop_image_paths(f.get("prop_images"), allow_empty=True)
    m = runner.fork_run(run_id, int(f["from_clip"]), (f.get("note") or "").strip() or None,
                        video=(f.get("video") or "").strip() or None, style=f.get("style"),
                        describe_facing=bool(f["describe_facing"]) if "describe_facing" in f else None,
                        prop_images=pics or None, clip_videos=_fork_clip_texts(f))
    started = cli._start_detached(m["id"])                 # it also takes its place in the queue
    return {"id": m["id"], "start": started}


@routes.post("/api/runs/{id}/cancel")
async def run_cancel(request: web.Request):
    return _ok(await _bg(runner.request_cancel, request.match_info["id"]))


@routes.post("/api/runs/{id}/delete")
async def run_delete(request: web.Request):
    return _ok(await _bg(runner.delete_run, request.match_info["id"]))


# ------------------------------------------------------------------ the clip-by-clip loop (AGENTS.md 7.5)
# The same runner functions as the CLI (`run takes|retake|use-take|score|start --through-clip K --detach`) and the MCP
# tools: a person can judge a long video clip by clip, make a clip again, put the best take in and go on.

def _run_takes(run_id: str) -> dict:
    """Each clip of a run with its own video, its takes (with video links) and which take it uses; plus what the
    player needs to offer the actions: the worker state, the clip limit and whether this run can take retakes."""
    st = runner.status(run_id)
    d = runner.run_dir(run_id)
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    tk = runner.list_takes(run_id)
    raw_st = json.loads((d / "state.json").read_text(encoding="utf-8"))
    fmt = (st["steps"].get("controls") or {}).get("outputs") or {}
    n = int(fmt.get("clips") or 0)
    parts = (st["steps"].get("clip") or {}).get("parts") or {}
    clips = []
    for k in range(1, n + 1):
        part = parts.get(str(k)) or {}
        out = part.get("outputs") or {}
        takes = []
        for t in tk["takes"].get(str(k), []):
            t = dict(t)
            t["url"] = file_url((t.get("outputs") or {}).get("clip"))
            t.pop("outputs", None)                        # server paths: the page needs only the link
            takes.append(t)
        clips.append({"clip": k, "made": bool(out.get("clip")), "url": file_url(out.get("clip")),
                      # the clip's own actions (one prompt per clip; "context" runs keep the shared text in a retake)
                      "actions": runner.clip_actions_of(m, raw_st, k)[0],      # the take in use (Codex 10-25)
                      "context_join": m.get("clip_join") == "context",
                      "score": part.get("score"), "used": tk["used"].get(str(k)), "takes": takes})
    old = runner.retired(m)                                  # an old LTX-2.3 run: its graphs are gone
    retake = not old and m.get("render_mode") == "layout" and m.get("relay") == "latent"
    return {"id": run_id, "status": st.get("status"), "worker_alive": bool(st.get("worker_alive")),
            "clip_limit": st.get("clip_limit"), "clips_total": n, "clips": clips, "retake": retake,
            "final": bool(st.get("final")),
            "why_not": None if retake else old or "retakes need a layout run with the latent relay (a Blender shot "
                                                  "with a library character)"}


@routes.get("/api/runs/{id}/takes")
async def run_takes(request: web.Request):
    return _ok(await _bg(_run_takes, request.match_info["id"]))


def _no_live_worker(run_id: str, what: str) -> None:
    if runner.worker_alive(run_id):
        raise runner.RunError(f"{what} only while the run is stopped: it is working now (wait until it stops)")


RETAKE_KNOBS = {"seed": _int_or_none, "ingredients_strength": _float_or_none, "adain": _float_or_none}


def _run_retake(run_id: str, f: dict, files: dict) -> dict:
    """A retake of one clip with a new seed and at most the common one-change knobs (AGENTS.md 7.5 step 4), then the
    run's worker starts (it makes the take and stops again at its clip limit; the video stays until use-take)."""
    try:
        if set(files) - {"look_image"}:
            raise ValueError(f"unexpected file fields: {', '.join(sorted(set(files) - {'look_image'}))}")
        _no_live_worker(run_id, "a retake")
        clip = _int_or_none(f.get("clip"))
        if not clip:
            raise ValueError("say which clip to make again")
        changes = {k: conv(f.get(k)) for k, conv in RETAKE_KNOBS.items()}
        kf = (f.get("keyframe") or "").strip()
        if kf:
            changes["keyframe"] = kf                          # on / off (add_take checks it)
        video = (f.get("video") or "").strip()
        if video:
            changes["video"] = video[:1500]
        if files.get("look_image"):
            changes["look_image"] = str(files["look_image"])
        r = runner.add_take(run_id, clip, changes, (f.get("note") or "").strip()[:500] or None)
    except BaseException:
        for p in files.values():                              # a refused retake keeps no upload
            p.unlink(missing_ok=True)
        raise
    st = runner.status(run_id)                                # a stopped run without a clip limit would make the
    fmt = (st["steps"].get("controls") or {}).get("outputs") or {}   # take AND every clip that is left: the app
    parts = (st["steps"].get("clip") or {}).get("parts") or {}       # pauses it after the clips already made instead
    made = sum(1 for k in range(1, int(fmt.get("clips") or 0) + 1) if ((parts.get(str(k)) or {}).get("outputs") or {}).get("clip"))
    if not st.get("clip_limit") and made < int(fmt.get("clips") or 0):
        runner.set_clip_limit(run_id, made)
        r["clip_limit"] = made
    r["start"] = cli._start_detached(run_id)
    return r


@routes.post("/api/runs/{id}/retake")
async def run_retake(request: web.Request):
    """{clip, seed, keyframe on|off, ingredients_strength, adain, video, note} (+ a look_image file: multipart)."""
    from .common import _form
    fields, files = await _form(request)
    return _ok(await _bg(_run_retake, request.match_info["id"], fields, files))


def _run_use_take(run_id: str, f: dict) -> dict:
    _no_live_worker(run_id, "a take can be put in")
    return runner.use_take(run_id, int(f["clip"]), int(f["take"]))


@routes.post("/api/runs/{id}/use-take")
async def run_use_take(request: web.Request):
    """{clip, take}: put the take into the video (the original stays as take 0) and join the clips again."""
    return _ok(await _bg(_run_use_take, request.match_info["id"], await request.json()))


def _run_continue(run_id: str, f: dict) -> dict:
    """Start a stopped, paused or never-started run again (`run start RUN [--through-clip K] --detach`).
    through_clip K > 0: stop after clip K; 0: no limit (the rest and the final video); missing: keep the limit, but a
    PAUSED run goes on to the next clip (its limit + 1; the same limit would only pause it again)."""
    _no_live_worker(run_id, "continue")
    runner.check_runnable(run_id)                            # an old LTX-2.3 run: a clear refusal, no queue ticket
    st = runner.status(run_id)
    pending = any(t.get("status") not in ("done", "failed", "canceled")
                  for lst in (st.get("takes") or {}).values() for t in lst)
    if st.get("status") == "done" and not pending:
        raise runner.RunError("the run is done: make a clip again (Retake) or build on it instead")
    through = f.get("through_clip")
    if through in (None, "") and st.get("status") == "paused" and st.get("clip_limit") and not pending:
        through = int(st["clip_limit"]) + 1
    if through not in (None, ""):
        runner.set_clip_limit(run_id, int(through) or None)
    new_attempt = st.get("status") == "unresolved" and bool(f.get("new_attempt"))
    return {"id": run_id, "start": cli._start_detached(run_id, new_attempt)}


@routes.post("/api/runs/{id}/continue")
async def run_continue(request: web.Request):
    f = await request.json() if request.can_read_body else {}
    return _ok(await _bg(_run_continue, request.match_info["id"], f or {}))


def _run_score(run_id: str, f: dict) -> dict:
    """The picture numbers of one clip or take (sharpness, the join with the clip before, jumps): CPU, seconds. The
    motion score (DWPose, 1-2 minutes) stays in the CLI (`run score`)."""
    from .. import clip_score
    take = _int_or_none(f.get("take"))
    return clip_score.score(run_id, int(f["clip"]), take, with_motion=False)


@routes.post("/api/runs/{id}/score")
async def run_score(request: web.Request):
    return _ok(await _bg(_run_score, request.match_info["id"], await request.json()))
