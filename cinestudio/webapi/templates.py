"""Templates: the shot library, new templates (dance, from a video, import), the figure mask, shot previews."""
from __future__ import annotations

import json
import secrets
from pathlib import Path

from aiohttp import web

from .. import runner, shots, song
from .common import _bg, _cleanup_canceled, _float_or_none, _ok, file_url, routes


@routes.get("/api/shots")
async def shots_list(request: web.Request):
    """Shots for the Video tab: every done template and hand-made shot (<home>/shots/<name>/shot.blend), newest first."""
    out = []
    for r in await _bg(shots.list_shots):
        blend = (r.get("files") or {}).get("blend")
        if r.get("status") == "done" and blend and Path(blend).is_file():
            out.append({"name": r["name"], "blend": blend, "template": not r.get("handmade"),
                        "builtin": bool(r.get("builtin")), "preview": (r.get("files") or {}).get("preview"),
                        # the template's Set layer words (shot layers): the default place of a backplate over its set
                        "set_words": ((r.get("layers") or {}).get("descriptions") or {}).get("set") or "",
                        # its default background (shot.json "background"): the New video form preselects it
                        "background": shots.default_background(blend), "poster_url": file_url(
                            (r.get("files") or {}).get("poster")), **_slots_view(r),
                        # one prompt per clip (AGENTS.md 7.2): the template's clip texts, its clip context and the
                        # default clip split (each clip's frames, seconds and the template's text) for Text per clip
                        **shots.clip_texts_view(r),
                        # the template rules (shot check): the New video form marks a template with errors
                        "check": _check_chip(r)})
        elif r.get("status") == "done" and r.get("kind") == "video":      # made from text: no .blend
            prm = r.get("params") or {}
            # a suggested video text: the template's scene and action WITHOUT camera words (the template carries the
            # camera; camera words in the run text made a deep lunge smear, gauntlet round 11)
            suggest = ""
            if prm.get("scene") and prm.get("action"):
                act = " ".join(x.strip().rstrip(".") for x in (prm["action"], prm.get("then") or "") if x.strip())
                suggest = (f"In {prm['scene'].rstrip('.')}, the character {act}. The place is empty: the character is "
                           f"the only person. The whole body stays in the frame.")
            out.append({"name": r["name"], "blend": f"template:{r['name']}", "template": True, "video": True,
                        "preview": (r.get("files") or {}).get("preview"), "suggest": suggest,
                        "poster_url": file_url((r.get("files") or {}).get("poster")), **_slots_view(r)})
    return _ok(out)


def _slots_view(r: dict) -> dict:
    """The template's character slots and prop groups for the New video form and the template cards (from the
    recorded layer scan; never Blender here): "mannequins" [{"index", "collection", "label", "side", "place",
    "screen", ...}], "slots_scanned" (False = no scan: one slot assumed), "markers" (every slot has a poster place),
    "props" [{"pattern", "label", "count", "words", "objects"}]."""
    sl = shots.mannequin_slots(r)
    return {"mannequins": sl["slots"], "slots_scanned": sl["scanned"], "markers": sl["markers"],
            "props": shots.prop_groups(r)}
# ------------------------------------------------------------------ templates (dance shots made from a song)


def _shot_view(r: dict) -> dict:
    f = r.get("files") or {}
    plan, prm = r.get("plan") or {}, r.get("params") or {}
    return {k: r.get(k) for k in ("name", "status", "kind", "added", "started", "ended", "error", "progress")} | {
        "handmade": bool(r.get("handmade")), "builtin": bool(r.get("builtin")), "description": r.get("description") or "",
        "background_motion": r.get("background_motion"),
        "song": prm.get("song"), "routines": plan.get("routines") or prm.get("routines") or [],
        "loop": prm.get("loop"), "night": prm.get("night"), "start": r.get("song_start", prm.get("start")),
        "seconds": plan.get("seconds") or r.get("seconds"), "beats": plan.get("n_beats"), "bpm": plan.get("bpm"),
        "blend": f.get("blend"), "source": r.get("source"),
        "scene": prm.get("scene"), "action": prm.get("action"), "camera": prm.get("camera"),
        "camera_label": _camera_label(prm.get("camera")),
        "preview_url": file_url(f.get("preview")), "poster_url": file_url(f.get("poster")),
        "figure": _figure_chip(r),
        # a template from a dance video (cinestudio/video_template.py): its sound is private, its tempo measured
        "private_sound": bool(r.get("private_sound")), "music": r.get("music"), "motion": r.get("motion"),
        # several people in the video (video_template: people 2|3|4|all|pick): who is who, the choice, the mannequins
        "people": _people_view(r), "warnings": r.get("warnings") or [],
        # the layers' words (shot layers): the card's Layers button shows whether the Set and the props have words
        "layers": _layers_chip(r),
        # the character slots ("2 characters" chip) and the prop groups (shots.mannequin_slots / prop_groups)
        **(_slots_view(r) if r.get("status") == "done" else {}),
        # its default background (a built-in template's backplate): the card links to it in the Backgrounds tab
        "background": _background_chip(r),
        # the template rules (shot check): {errors, warnings, at, stale} or None (never checked)
        "check": _check_chip(r)}


def _check_chip(r: dict) -> dict | None:
    """The card's check chip (shot_check.chip: the cached result of the current .blend, else the stored summary)."""
    from .. import shot_check
    try:
        return shot_check.chip(r)
    except Exception:  # noqa: BLE001 - a card never fails over its chip
        return None


def _background_chip(r: dict) -> dict | None:
    blend = (r.get("files") or {}).get("blend")
    name = shots.default_background(blend) if blend and r.get("background") else None
    if not name:
        return None
    from .. import background
    try:
        b = background.load(name)
    except background.BackgroundError:
        return None
    return {"name": name, "kind": b["kind"], "url": file_url(b["file"]), "builtin": bool(b.get("builtin"))}


def _people_view(r: dict) -> dict | None:
    """The people scan of a template from a video with several people: the numbered sheet, the preview video, each
    person found (number, colour, visible share, place in the picture) and the chosen ones (Mannequin, Mannequin2)."""
    info = r.get("people")
    if r.get("source") != "video" or not info:
        return None
    return {"count": info.get("count"), "seen": info.get("seen"), "tracks": info.get("tracks") or [],
            "sheet_url": file_url(info.get("sheet")), "video_url": file_url(info.get("video")),
            "chosen": info.get("chosen") or [], "span_s": info.get("span_s"), "trimmed": info.get("trimmed"),
            "asked": (r.get("params") or {}).get("people")}


def _layers_chip(r: dict) -> dict | None:
    lay = r.get("layers")
    if not (r.get("files") or {}).get("blend") or r.get("kind") == "video":
        return None
    lay = lay or {}
    desc, scan = lay.get("descriptions") or {}, lay.get("scan") or {}
    return {"set": bool((desc.get("set") or "").strip()), "props": len(scan.get("props") or []),
            "described": len([v for v in (desc.get("props") or {}).values() if (v or "").strip()]),
            "warnings": len(lay.get("warnings") or []), "scanned": bool(scan)}


def _figure_chip(r: dict) -> dict | None:
    """The figure selection of a video template, for its card: done (+ flags) or busy."""
    if r.get("kind") != "video" or not r.get("name"):
        return None
    from .. import figure_mask
    try:
        st = figure_mask.status(r["name"])
        mj = figure_mask.mask_dir(r["name"]) / "mask.json"
        flags = len(json.loads(mj.read_text(encoding="utf-8")).get("flags") or []) if mj.is_file() else None
    except (OSError, ValueError, shots.ShotError):
        return None
    if flags is None and not st:
        return None
    return {"selected": flags is not None, "flags": flags, "status": st.get("status")}


# the camera of older video templates made from text (the text generator was removed on 2026-10-22)
CAMERA_LABELS = {"still": "Still", "push": "Slow push in", "pull": "Slow pull back", "pan": "Slow pan"}


def _camera_label(cam: str | None) -> str | None:
    return CAMERA_LABELS.get(cam, cam) if cam else None


@routes.get("/api/templates")
async def templates(request: web.Request):
    await _bg(_cleanup_canceled)
    songs = [{"name": s["name"], "bpm": (s.get("beats") or {}).get("bpm") or s.get("bpm"),
              "seconds": (s.get("beats") or {}).get("seconds") or s.get("seconds")}
             for s in await _bg(song.library) if s.get("status") == "done"]
    return _ok({"shots": [_shot_view(r) for r in await _bg(shots.list_shots)], "songs": songs,
                "routines": [{"id": k, "counts": c, "label": label} for k, (c, label) in shots.ROUTINES.items()]})


@routes.post("/api/templates/new")
async def template_new(request: web.Request):
    f = await request.json()
    loop = f.get("loop", True) not in (False, "false", "off", "0", "")
    return _ok(await _bg(shots.start_dance, (f.get("name") or "").strip(), (f.get("song") or "").strip(),
                         f.get("routines") or "", loop, _float_or_none(f.get("seconds")) or 10.0,
                         None if loop else _float_or_none(f.get("start")), bool(f.get("night"))))
TEMPLATE_UPLOAD_MAX = 2 * 1024 ** 3
FIELD_MAX = 8000                                          # bytes of one text field of an upload form
TEMPLATE_SUFFIXES = {".blend": "blend", ".mp4": "video", ".mov": "video", ".mkv": "video", ".webm": "video"}


async def _read_upload(request: web.Request, suffixes: dict, what: str) -> tuple[dict, Path | None, str | None]:
    """A multipart form: the text fields (at most FIELD_MAX bytes each) and ONE file (streamed to <home>/uploads, at most
    TEMPLATE_UPLOAD_MAX; a second file is refused). On an error the partial file is deleted."""
    if not (request.content_type or "").startswith("multipart/"):
        raise ValueError("send the template as a multipart form (name, file, ...)")
    fields, upload, kind = {}, None, None
    reader = await request.multipart()
    folder = runner.home() / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        async for part in reader:
            if not part.filename:                         # a text field, read with a size limit
                data = bytearray()
                while chunk := await part.read_chunk(4096):
                    data.extend(chunk)
                    if len(data) > FIELD_MAX:
                        raise ValueError(f"the field {part.name!r} is too long")
                fields[part.name] = data.decode("utf-8", "replace")
                continue
            if upload is not None:
                raise ValueError("send one file only")
            suffix = Path(part.filename).suffix.lower()
            kind = suffixes.get(suffix)
            if kind is None:
                raise ValueError(f"unsupported file type {suffix!r}: {what}")
            upload = folder / f"template_{secrets.token_hex(4)}{suffix}"
            size = 0
            with open(upload, "wb") as fh:
                while chunk := await part.read_chunk():
                    size += len(chunk)
                    if size > TEMPLATE_UPLOAD_MAX:
                        raise ValueError("the file is larger than 2 GB")
                    fh.write(chunk)
    except BaseException:
        if upload is not None:
            upload.unlink(missing_ok=True)
        raise
    return fields, upload, kind


@routes.post("/api/templates/import")
async def template_import(request: web.Request):
    """Multipart: name, description, file (a .blend shot, or a video with one stand-in figure). The file is stored in
    the user's template library like `shot add` / `shot add-video`; a .blend gets a Workbench preview (Blender)."""
    fields, upload, kind = await _read_upload(request, TEMPLATE_SUFFIXES,
                                              "a .blend shot or a video (.mp4, .mov, .mkv, .webm)")
    try:
        if upload is None:
            raise ValueError("choose a .blend or a video file")
        name, desc = (fields.get("name") or "").strip(), (fields.get("description") or "").strip()
        shots.check_new_name(name)
        if kind == "blend":
            rec = await _bg(shots.add, name, upload, desc, True)          # moved into the library, preview made
        else:
            rec = await _bg(shots.add_video, name, upload, desc, {"imported": True})
        return _ok(_shot_view(rec))
    finally:
        if upload is not None and upload.exists():
            upload.unlink(missing_ok=True)                                 # a failed import leaves nothing behind


MOTION_SUFFIXES = {s: "video" for s in (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi")}


@routes.post("/api/templates/from-video")
async def template_from_video(request: web.Request):
    """Multipart: name, file (a dance video), start, seconds, sound (video|none), people (1|2|3|4|all|pick), description. The
    mannequin copies the person's motion (SAM 3D Body -> a headless Blender template; cinestudio/video_template.py): a
    detached worker waits for its turn in the GPU queue. people 2..all|pick: SAM 3 finds every person (see /people)."""
    from .. import video_template
    fields, upload, _ = await _read_upload(request, MOTION_SUFFIXES, "a video (.mp4, .mov, .mkv, .webm, .m4v, .avi)")
    try:
        if upload is None:
            raise ValueError("choose a video file")
        r = await _bg(video_template.start, (fields.get("name") or "").strip(), upload,
                      _float_or_none(fields.get("start")) or 0.0, _float_or_none(fields.get("seconds")),
                      fields.get("sound") or "video", (fields.get("description") or "").strip(), True,
                      people=(fields.get("people") or "1").strip())
        return _ok(r)
    finally:
        if upload is not None and upload.exists():
            upload.unlink(missing_ok=True)                                 # moved into the template, or a failure


@routes.post("/api/templates/{name}/people")
async def template_people(request: web.Request):
    """{tracks: [0, 2]}: after a "pick" scan, the one or two people (SAM 3 numbers on the sheet) that become the
    mannequins; Blender builds the template in a detached worker (no second GPU job)."""
    from .. import video_template
    f = await request.json()
    tracks = f.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("tracks must be a list of person numbers")
    return _ok(await _bg(video_template.pick, request.match_info["name"], tracks))


@routes.get("/api/templates/{name}/mask")
async def template_mask(request: web.Request):
    """The figure selection of a video template: status, the last mask (flags, area per frame, points, urls) and the
    template video (url, size, frames, fps)."""
    from .. import figure_mask
    name = request.match_info["name"]
    s = await _bg(figure_mask.summary, name)
    files = (s.get("mask") or {}).pop("files", None) or {}
    rec = await _bg(shots.load, name)
    s["video"]["url"] = file_url((rec.get("files") or {}).get("video"))
    s["overlay_url"] = file_url(files.get("overlay"))
    s["mask_url"] = file_url(files.get("mask"))
    return _ok(s)


@routes.post("/api/templates/{name}/mask")
async def template_mask_start(request: web.Request):
    """Track the figure (SAM 2) with the user's clicks [{frame, x, y, positive}] in the template video's pixels; a
    detached worker, so an app restart does not stop it."""
    from .. import figure_mask
    f = await request.json()
    return _ok(await _bg(figure_mask.start, request.match_info["name"], f.get("clicks") or []))


def _layers_view(name: str, out: dict | None = None) -> dict:
    """A template's three layers (Mannequin*, Props, Set) and their words, for the Layers dialog. A user template
    without a stored scan is scanned now (Blender, a few seconds, no render); a built-in template is read-only."""
    rec = shots.load(name)
    if not (rec.get("files") or {}).get("blend"):
        raise shots.ShotError(f"template {name!r} has no .blend (a video template has no layers)")
    lay = (out or {}).get("layers") or rec.get("layers") or {}
    if out is None and not lay.get("scan") and not rec.get("builtin"):
        lay = shots.layers(name)["layers"]
    desc = lay.get("descriptions") or {}
    return {"name": name, "builtin": bool(rec.get("builtin")), "scan": lay.get("scan"),
            "set": desc.get("set") or "", "props": desc.get("props") or {}, "warnings": lay.get("warnings") or []}


@routes.get("/api/templates/{name}/layers")
async def template_layers(request: web.Request):
    return _ok(await _bg(_layers_view, request.match_info["name"]))


def _layers_save(name: str, f: dict) -> dict:
    """{set, props: {name or pattern: words; "" removes}, rescan}: `shot layers NAME --set --prop --rescan`."""
    props = f.get("props") or {}
    if not isinstance(props, dict):
        raise ValueError("props must map a prop (or a pattern such as Ball_*) to its words")
    props = {str(k).strip(): str(v or "") for k, v in props.items() if str(k).strip()}
    st = f.get("set")
    out = shots.layers(name, None if st is None else str(st)[:600], {k: v[:300] for k, v in props.items()},
                       bool(f.get("rescan")))
    return _layers_view(name, out)


@routes.post("/api/templates/{name}/layers")
async def template_layers_save(request: web.Request):
    return _ok(await _bg(_layers_save, request.match_info["name"], await request.json()))


def _clip_texts_save(name: str, f: dict) -> dict:
    """{clip_texts: {clip: ACTIONS; "" removes}, context}: `shot clip-texts NAME --clip K TEXT --context TEXT`."""
    texts = f.get("clip_texts") or {}
    if not isinstance(texts, dict):
        raise ValueError("clip texts must map clip numbers to texts")
    ctx = f.get("context")
    return shots.clip_texts(name, {str(k): str(v or "") for k, v in texts.items()}, None if ctx is None else str(ctx))


@routes.get("/api/templates/{name}/clip-texts")
async def template_clip_texts(request: web.Request):
    return _ok(await _bg(shots.clip_texts, request.match_info["name"]))


@routes.post("/api/templates/{name}/clip-texts")
async def template_clip_texts_save(request: web.Request):
    return _ok(await _bg(_clip_texts_save, request.match_info["name"], await request.json()))


@routes.post("/api/templates/{name}/check")
async def template_check(request: web.Request):
    """`shot check NAME`: the template rules (one headless Blender read, no render, no GPU ticket; instant when the
    .blend is cached). {fresh: true} reads the .blend again. Returns {errors, warnings, infos, checks, sha, at}."""
    from .. import shot_check
    f = (await request.json()) if request.can_read_body else {}
    res = await _bg(shot_check.check, request.match_info["name"], bool((f or {}).get("fresh")))
    return _ok({k: v for k, v in res.items() if k != "ok"})


@routes.post("/api/templates/{name}/delete")
async def template_delete(request: web.Request):
    return _ok(await _bg(shots.delete, request.match_info["name"]))


@routes.post("/api/shots/preview")
async def shot_preview_api(request: web.Request):
    """A small Workbench preview video of a shot (made once in the background; poll until "ready")."""
    from .. import shot_preview
    f = await request.json()
    blend = (f.get("blend") or "").strip().strip('"')
    if blend.startswith("template:"):                       # a video template: the template video is its preview
        rec = await _bg(shots.load, blend.split(":", 1)[1])
        prev = (rec.get("files") or {}).get("preview")
        if prev and Path(prev).is_file():
            return _ok({"status": "ready", "path": prev, "url": file_url(prev), "video_template": True})
        return _ok({"status": "unavailable", "error": "this template has no video"})
    try:                                                    # a template has its own preview (with its song): use it
        rec = shots.library_record(blend) or {}
        prev = (rec.get("files") or {}).get("preview")
        if rec.get("status") == "done" and prev and Path(prev).is_file() and \
                Path((rec.get("files") or {}).get("blend") or "").resolve() == Path(blend).resolve():
            return _ok({"status": "ready", "path": prev, "url": file_url(prev)})
    except (shots.ShotError, OSError, ValueError):
        pass
    r = await _bg(shot_preview.shot_preview, blend)
    r["url"] = file_url(r.get("path"))
    return _ok(r)
