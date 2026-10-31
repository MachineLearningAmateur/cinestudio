"""Settings: the setup checklist, model tiers and downloads, access tokens, the backend, GPU preferences."""
from __future__ import annotations

import sys
from pathlib import Path

from aiohttp import web

from .. import models, settings, sysinfo
from .system import _gpu
from .common import _bg, _form, _ok, routes


@routes.get("/api/setup")
async def setup_status(request: web.Request):
    """Quick setup: which tokens are set (never their values), the backend, each tier's download state, and the
    download worker's progress."""
    def collect():
        tiers = []
        for t in models.tier_names():
            try:
                st = models.status(t)
                tiers.append({k: st[k] for k in ("tier", "label", "vram_gb", "ram_gb", "about", "ready", "total_bytes",
                                                  "have_bytes", "missing_bytes", "disk_ok")} |
                             {"missing": [f["file"] for f in st["files"] if f["state"] != "ok"],
                              "files": [{k: f.get(k) for k in ("slot", "file", "size", "state")} for f in st["files"]]})
            except models.ModelsError as exc:
                tiers.append({"tier": t, **models.tier_info(t), "error": str(exc)})
        try:
            md = str(models.models_dir())
        except models.ModelsError:
            md = None
        free = None
        try:
            import shutil as _sh
            free = _sh.disk_usage(md).free if md and Path(md).exists() else None
        except OSError:
            pass
        snap = sysinfo.snapshot(background=True)
        g = snap.get("gpu") or {}
        vram = round((g.get("vram_total") or 0) / 1024 ** 3)
        ram = round((snap.get("ram_total") or 0) / 1024 ** 3)
        fits = [t for t in tiers if t.get("vram_gb", 99) <= max(vram, 1) + 1 and t.get("ram_gb", 99) <= ram + 4]
        try:
            from .. import comfy
            eng = comfy.status()
        except Exception as exc:  # noqa: BLE001 - shown as "not found"
            eng = {"running": False, "error": str(exc)}
        return {"settings": settings.get(), "tokens": settings.status(), "tiers": tiers, "models_dir": md,
                "disk_free": free, "download": models.download_state(),
                "machine": {"gpu": g.get("name"), "vram_gb": vram, "ram_gb": ram, "windows": sys.platform == "win32"},
                "recommended": (max(fits, key=lambda t: t["vram_gb"])["tier"] if fits else models.tier_names()[0]),
                "engine": {"running": bool(eng.get("running")), "found": bool(eng.get("source") or eng.get("running")),
                           "source": eng.get("source"), "error": eng.get("error")}}
    return _ok(await _bg(collect))


@routes.post("/api/setup/token")
async def setup_token(request: web.Request):
    """{"name": "hf"|"comfy", "value": "..."}: save (or replace) a token, then check it with its service.
    The answer says only whether it is set and whether it works."""
    f, _ = await _form(request)
    name = str(f.get("name", ""))
    st = settings.set_token(name, str(f.get("value", "")))
    return _ok({"token": name, **st, "check": await _bg(settings.check_token, name)})


@routes.post("/api/setup/token/{op}")
async def setup_token_op(request: web.Request):
    """check | clear, with {"name": ...}."""
    f, _ = await _form(request)
    name, op = str(f.get("name", "")), request.match_info["op"]
    if op == "check":
        return _ok({"token": name, "check": await _bg(settings.check_token, name)})
    if op == "clear":
        return _ok({"token": name, **settings.clear_token(name)})
    raise ValueError("unknown action")


@routes.post("/api/setup/tier")
async def setup_tier(request: web.Request):
    """{"tier": "8gb"|"large"}: the tier for new work (only a complete tier; runs keep their own models)."""
    f, _ = await _form(request)
    tier = str(f.get("tier", ""))
    st = await _bg(models.status, tier)
    if not st["ready"] and not f.get("force"):
        raise ValueError(f"tier {tier} is not downloaded yet ({st['missing_bytes'] / 1e9:.1f} GB missing)")
    return _ok(settings.set_tier(tier))


@routes.post("/api/setup/backend")
async def setup_backend(request: web.Request):
    f, _ = await _form(request)
    return _ok(settings.set_backend(str(f.get("backend", ""))))


@routes.post("/api/setup/download")
async def setup_download(request: web.Request):
    """{"tier": T}: start the background download of the tier's missing files; {"stop": true}: stop it."""
    f, _ = await _form(request)
    if f.get("stop"):
        return _ok(models.request_stop())
    return _ok(await _bg(models.start_download, str(f.get("tier", ""))))


@routes.get("/api/settings/gpu")
async def settings_gpu(request: web.Request):
    """Running apps (and apps with a setting) with their graphics choice: default, integrated or gpu."""
    import asyncio
    g = _gpu()
    return _ok(await asyncio.get_running_loop().run_in_executor(None, g.list_apps))


@routes.post("/api/settings/gpu")
async def settings_gpu_set(request: web.Request):
    """{"exe": full path, "choice": "default"|"integrated"|"gpu"}; the app must be restarted to use it."""
    f, _ = await _form(request)
    return _ok(_gpu().set_app(f.get("exe", ""), f.get("choice", "")))


@routes.post("/api/settings/gpu/{action}")
async def settings_gpu_bulk(request: web.Request):
    """all-integrated: the running everyday apps to the integrated graphics; reset: restore every app
    that cinestudio changed to what it was before."""
    import asyncio
    g = _gpu()
    fn = {"all-integrated": g.on, "reset": g.off}.get(request.match_info["action"])
    if fn is None:
        raise ValueError("unknown action")
    return _ok(await asyncio.get_running_loop().run_in_executor(None, fn))
