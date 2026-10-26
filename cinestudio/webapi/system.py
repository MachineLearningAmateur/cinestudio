"""The page itself, health, meters, the queue, the ComfyUI engine, LoRA lists and the media files."""
from __future__ import annotations

from aiohttp import web

from .. import cli, sysinfo, workqueue
from .common import MEDIA_TYPES, WEB, _bg, _file_areas, _form, _ok, routes


@routes.get("/")
async def index(request: web.Request):
    return web.FileResponse(WEB / "index.html")


@routes.get("/sw.js")
async def service_worker(request: web.Request):
    """At the root, so that it covers the whole app (needed to install cinestudio as an app)."""
    return web.FileResponse(WEB / "sw.js", headers={"Content-Type": "text/javascript", "Cache-Control": "no-cache"})


@routes.get("/api/status")
async def status(request: web.Request):
    import asyncio
    return _ok(await asyncio.get_running_loop().run_in_executor(None, cli.doctor))


@routes.get("/api/system")
async def system(request: web.Request):
    """CPU, RAM, GPU, VRAM and GPU temperature of this PC, for the meters in the side rail."""
    return _ok(sysinfo.snapshot())


@routes.get("/api/queue")
async def queue_view(request: web.Request):
    """All queued and running work (character steps and videos), first in line first."""
    import asyncio
    return _ok(await asyncio.get_running_loop().run_in_executor(None, workqueue.view))


@routes.post("/api/queue/cancel")
async def queue_cancel(request: web.Request):
    """Explicit cancel of one queued or running request: {"kind": "character"|"video", "ref": id}."""
    f, _ = await _form(request)
    return _ok(await _bg(workqueue.cancel, f.get("kind", ""), f.get("ref", "")))


def _gpu():
    import sys as _sys
    if _sys.platform != "win32":
        raise ValueError("the graphics choice per app is a Windows setting")
    from .. import gpu_focus
    return gpu_focus


@routes.get("/api/engine")
async def engine_status(request: web.Request):
    import asyncio
    from .. import comfy
    return _ok(await asyncio.get_running_loop().run_in_executor(None, comfy.status))


@routes.post("/api/engine/{action}")
async def engine_action(request: web.Request):
    """start: ComfyUI hidden in the background; stop: only the one cinestudio started, with an empty queue."""
    import asyncio
    from .. import comfy
    fn = {"start": comfy.start, "stop": comfy.stop}.get(request.match_info["action"])
    if fn is None:
        raise ValueError("unknown action")
    return _ok(await asyncio.get_running_loop().run_in_executor(None, fn))


@routes.get("/api/loras")
async def loras(request: web.Request):
    return _ok(cli.loras())


@routes.get("/files/{area}/{path:.+}")
async def files(request: web.Request):
    area, rel = request.match_info["area"], request.match_info["path"]
    root = _file_areas().get(area)
    if root is None:
        raise web.HTTPNotFound()
    p = (root / rel).resolve()
    try:
        p.relative_to(root.resolve())
    except ValueError:
        raise web.HTTPForbidden()
    if not p.is_file() or p.suffix.lower() not in MEDIA_TYPES:      # only pictures, videos and sounds, never active content
        raise web.HTTPNotFound()
    return web.FileResponse(p, headers={"Cache-Control": "no-cache", "Content-Type": MEDIA_TYPES[p.suffix.lower()],
                                        "X-Content-Type-Options": "nosniff"})
