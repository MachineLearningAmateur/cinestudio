"""The local web app: Characters, Music, Templates, Backgrounds, Video and Settings on top of the same core as the CLI.

    python -m cinestudio app [--port 8765] [--open]

It listens on 127.0.0.1 only. Character steps run as background jobs in this process (cinestudio.jobs),
one at a time; video runs use the same detached workers as `run start --detach`. The page polls
the JSON API every few seconds. Files are served only from the character library and the runs.

The JSON API lives in cinestudio/webapi/ (one module per area); this module builds and serves the app."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

from aiohttp import web

from . import cli, jobs, spawn  # noqa: F401 - spawn: tests patch webapp.spawn.detached
from .webapi import system, setup, backgrounds, templates, characters, props, runs, songs, styles, workflows  # noqa: F401 - routes
from .webapi.characters import _edit_modes  # noqa: F401 - used by tests
from .webapi.common import MAX_UPLOAD, WEB, errors, guard, routes


def make_app(port: int | None = None) -> web.Application:
    app = web.Application(middlewares=[guard, errors], client_max_size=MAX_UPLOAD + 1024 * 1024)
    app["jobs"] = jobs.JobRunner()
    app["port"] = port
    app.add_routes(routes)
    app.router.add_static("/static/", WEB)
    return app


def open_window(url: str) -> None:
    """A desktop-like window: Edge or Chrome in app mode (no tabs, no address bar); else the browser."""
    candidates = []
    if sys.platform.startswith("win"):
        for base in (r"C:\Program Files (x86)", r"C:\Program Files"):
            candidates += [rf"{base}\Microsoft\Edge\Application\msedge.exe", rf"{base}\Google\Chrome\Application\chrome.exe"]
    else:
        candidates += [shutil.which(x) or "" for x in ("microsoft-edge", "google-chrome", "chromium", "chromium-browser")]
    for exe in candidates:
        if exe and Path(exe).is_file():
            subprocess.Popen([exe, f"--app={url}", "--window-size=1500,950"], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    webbrowser.open(url)


def _start_engine_soon() -> None:
    """Start ComfyUI in the background when the app opens (so the first job does not wait for it)."""
    import threading

    def go():
        from . import comfy
        try:
            comfy.ensure_running()
        except Exception as exc:  # noqa: BLE001 - the health dot and the Settings tab show the problem
            print(json.dumps({"event": "comfyui_not_started", "error": str(exc)}), flush=True)
        try:                                            # the SDK talks to ComfyUI through the local proxy
            r = cli._proxy_start()
            if not r.get("ok"):
                print(json.dumps({"event": "proxy_not_started", "error": r.get("error")}), flush=True)
        except Exception as exc:  # noqa: BLE001 - the health dot shows the problem
            print(json.dumps({"event": "proxy_not_started", "error": str(exc)}), flush=True)
    threading.Thread(target=go, daemon=True, name="start-comfyui").start()


def serve(port: int = 8765, open_browser: bool = False) -> None:
    url = f"http://127.0.0.1:{port}/"
    if os.environ.get("CINESTUDIO_AUTOSTART_COMFYUI", "1") != "0":
        _start_engine_soon()
    if open_browser:
        open_window(url)
    print(json.dumps({"ok": True, "url": url, "note": "Ctrl+C to stop"}), flush=True)
    web.run_app(make_app(port), host="127.0.0.1", port=port, print=None)
if __name__ == "__main__":
    serve(open_browser=True)
