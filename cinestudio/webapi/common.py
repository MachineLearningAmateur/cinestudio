"""Shared parts of the web API: the route table, the middlewares, the JSON answers, uploads and file links."""
from __future__ import annotations

import secrets
from pathlib import Path

from aiohttp import web

from .. import character, cli, jobs, models, runner, shots, song


WEB = Path(__file__).resolve().parents[1] / "web"     # cinestudio/web (this module is in cinestudio/webapi)
MAX_UPLOAD = 64 * 1024 * 1024


def _ok(data: dict | list, status: int = 200) -> web.Response:
    return web.json_response({"ok": True, "data": data}, status=status)


def _err(msg: str, status: int = 400, kind: str = "Error") -> web.Response:
    return web.json_response({"ok": False, "error": msg, "error_type": kind}, status=status)
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
APP_HEADER = "X-Cinestudio"


@web.middleware
async def guard(request: web.Request, handler):
    """The app listens on 127.0.0.1, but a web page in the user's browser can still send requests to it.
    1. Host must be this app (stops DNS rebinding: a foreign name that resolves to 127.0.0.1).
    2. Every request that changes something must carry the X-Cinestudio header (a page on another site
       cannot add it without a CORS preflight, which this app never allows) and, when the browser sends
       an Origin, it must be this app's origin."""
    port = request.app.get("port")
    allowed = {f"127.0.0.1:{port}", f"localhost:{port}"} if port else None
    if allowed and request.host not in allowed:
        return _err("unknown host", 403, "Forbidden")
    if request.method not in SAFE_METHODS:
        origin = request.headers.get("Origin")
        if origin and origin != f"http://{request.host}":
            return _err("requests from other sites are refused", 403, "Forbidden")
        if request.headers.get(APP_HEADER) != "1":
            return _err(f"missing {APP_HEADER} header (send '{APP_HEADER}: 1')", 403, "Forbidden")
    resp = await handler(request)
    if request.path == "/" or request.path.startswith("/static/"):   # the page and its scripts: the browser asks again
        resp.headers["Cache-Control"] = "no-cache"                    # each time (a 304 when unchanged), so a reload
    return resp                                                       # never shows an old script (2026-10-24)


async def _bg(fn, *args, **kwargs):
    """Run slow, blocking work (Blender, a worker start, a cancel, a delete) in a thread, so the app keeps
    answering the page (queue, progress, a second cancel) meanwhile."""
    import asyncio
    import functools
    return await asyncio.get_running_loop().run_in_executor(None, functools.partial(fn, *args, **kwargs))


def _int_or_none(v) -> int | None:
    """'' or None -> None; 0 stays 0 (it is a valid seed)."""
    return None if v in (None, "") else int(v)


def _float_or_none(v) -> float | None:
    return None if v in (None, "") else float(v)


def _inside(path: Path, *roots: Path) -> bool:
    p = Path(path).resolve()
    for root in roots:
        try:
            p.relative_to(Path(root).resolve())
            return True
        except ValueError:
            continue
    return False


@web.middleware
async def errors(request: web.Request, handler):
    try:
        resp = await handler(request)
        if request.path == "/" or request.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache"      # the app's own code: a reload always gets the new version
        return resp
    except web.HTTPException:
        raise
    except (character.CharacterError, runner.RunError, song.SongError, shots.ShotError, cli.CliInputError, ValueError,
            KeyError, models.ModelsError) as exc:
        return _err(str(exc), 400, type(exc).__name__)
    except FileNotFoundError as exc:
        return _err(str(exc), 404, "NotFound")
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc) or repr(exc), 500, type(exc).__name__)
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif",
               ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime", ".mkv": "video/x-matroska",
               ".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg"}


def _file_areas() -> dict[str, Path]:
    """The folders whose pictures, videos and sounds the app may serve (by area name)."""
    from .. import background as _background
    return {"characters": character.library(), "props": runner.home() / "props",    # the prop library (props.py)
            "characters_builtin": character.BUILTIN,                         # the built-in characters (read-only)
            "runs": runner.runs_dir(), "shots": runner.home() / "shots",
            "builtin": shots.builtin_dir(),
            "backgrounds": runner.home() / "backgrounds",
            "backgrounds_builtin": _background.BUILTIN,                      # the built-in backgrounds (read-only)
            "previews": runner.home() / "previews", "songs": song.songs_dir(),
            "mannequin": runner.home() / "cache" / "mannequin_preview",     # the Mannequin view's Blender previews
            "styles": runner.home() / "styles" / "refs",                    # the style library's reference pictures
            "style_files": runner.home() / "styles",       # reference candidates (styles/NAME/candidates; media only)
            "style_crops": runner.home() / "cache" / "style_crops",         # the crops Qwen gets as Picture 2
            "styles_builtin": Path(__file__).resolve().parents[2] / "styles" / "library"}


def file_url(path: str | Path | None) -> str | None:
    """URL of a library or run file (None for anything outside those folders)."""
    if not path:
        return None
    p = Path(path).resolve()
    if not p.is_file():
        return None
    for prefix, root in _file_areas().items():
        try:
            rel = p.relative_to(root.resolve())
        except ValueError:
            continue
        stamp = int(p.stat().st_mtime)
        return f"/files/{prefix}/{rel.as_posix()}?v={stamp}"
    return None


def _history(c: dict) -> list[dict]:
    """The changes of each image (front or a view) with the image before and the result of each.
    The result of a kept change is the "before" of the next change of the same image (or the image
    now); the result of an undone change was saved at the undo as <target>_undone_v<N>.png."""
    edits = c.get("edits", [])
    last_kept = {}                                     # target -> index of its last kept change (= the image now)
    for i, e in enumerate(edits):
        if not e.get("undone"):
            last_kept[e.get("target", "front")] = i
    out = []
    for i, e in enumerate(edits):
        t = e.get("target", "front")
        nxt = next((f for f in edits[i + 1:] if f.get("target", "front") == t), None)
        if e.get("undone"):
            result = Path(e["before"]).with_name(f"{t}_undone_v{e['version']}.png")
        elif nxt:
            result = nxt["before"]
        else:
            result = c.get("files", {}).get(t)
        out.append({**e, "target": t, "before_url": file_url(e.get("before")), "result_url": file_url(result),
                    "current": last_kept.get(t) == i})
    return out


def _char_view(c: dict) -> dict:
    d = character.char_dir(c["name"])
    out = dict(c)
    out["urls"] = {k: file_url(v) for k, v in c.get("files", {}).items()}
    out["candidates"] = [{**x, "url": file_url(x["file"])} for x in c.get("candidates", [])]
    out["edits"] = _history(c)
    out["jobs"] = jobs.list_jobs(c["name"], limit=8)
    out["contact_url"] = file_url(d / "candidates" / "candidates.jpg")
    out["missing"] = character.video_missing(c)             # what a Blender run still needs
    try:                                                     # the Style menu: the skins of this character (main first)
        out["skins"] = [{k: r.get(k) for k in ("name", "status", "style_ref", "style_version", "skin_of", "missing",
                                               "display_name")} | {"thumb_url": file_url(r.get("thumb"))}
                        for r in character.skins(c["name"])]
    except Exception:  # noqa: BLE001 - a broken sibling record must not hide this character
        out["skins"] = []
    out["skin_jobs"] = _skin_jobs({s["name"] for s in out["skins"]} | {c["name"]})
    return out


def _skin_jobs(family: set[str]) -> list[dict]:
    """The new skins being made for this family (queued or running new_skin jobs): the skin cards show a
    placeholder with the job's progress. Active jobs are recent: the newest 40 job records are enough."""
    out = []
    for row in jobs.list_jobs(None, limit=40):
        if row.get("action") != "new_skin" or row.get("status") not in jobs.ACTIVE or row.get("character") not in family:
            continue
        try:
            style = (jobs.load(row["id"]).get("args") or {}).get("style")
        except (OSError, ValueError, KeyError):
            style = None
        out.append({"id": row["id"], "status": row["status"], "character": row["character"], "style": style})
    return out


async def _save_upload(field, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(field.filename or "upload.png").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
        raise ValueError(f"unsupported image type {suffix!r} (png, jpg or webp)")
    dst = folder / f"upload_{secrets.token_hex(4)}{suffix}"
    size = 0
    try:                                                  # the partial file goes on ANY failure (a cancelled or broken
        with open(dst, "wb") as fh:                       # request, a size over the limit): Codex review 2026-10-24
            while chunk := await field.read_chunk():
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise ValueError("the image is larger than 64 MB")
                fh.write(chunk)
    except BaseException:
        dst.unlink(missing_ok=True)
        raise
    return dst


async def _form(request: web.Request) -> tuple[dict, dict[str, Path]]:
    """Fields and uploaded files of a multipart form (or a JSON body)."""
    ctype = request.content_type or ""
    if not request.can_read_body:
        return {}, {}
    if ctype.startswith("application/json"):
        return await request.json(), {}
    if ctype == "application/x-www-form-urlencoded":
        return dict(await request.post()), {}
    if not ctype.startswith("multipart/"):
        return {}, {}
    fields, files = {}, {}
    reader = await request.multipart()
    uploads = runner.home() / "uploads"
    try:                                                  # every saved file is deleted when the form fails part-way
        async for part in reader:
            if part.filename:
                if part.name in files:                    # a second file in the same field would orphan the first
                    raise ValueError(f"more than one file in the field {part.name!r}")
                files[part.name] = await _save_upload(part, uploads)
            else:
                fields[part.name] = await part.text()
    except BaseException:
        for f in files.values():
            f.unlink(missing_ok=True)
        raise
    return fields, files
# ---------------------------------------------------------------- API
routes = web.RouteTableDef()
_SAMPLERS: dict = {}


def _cleanup_canceled() -> None:
    """Canceled runs, songs and templates are not shown and their files are deleted (user, 2026-10-21); one that is
    still settling its cancel is left for the next call."""
    for f in (runner.cleanup_canceled, song.cleanup_canceled, shots.cleanup_canceled):
        try:
            f()
        except Exception:                                 # a cleanup problem must not break the list
            pass
