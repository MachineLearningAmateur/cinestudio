"""cinestudio command line, for agents and scripts first.

Output contract: stdout carries JSON only, one object per line. Streaming commands (run start
without --detach, run events) print event lines (they have an "event" key) and then ONE final
result line; every other command prints only the result line. The result line always has "ok"
(true/false); on failure it also has "error" and "error_type". Exit code: 0 = ok, 1 = the command
ran but failed (e.g. the run failed), 2 = bad input or a refused request.

    python -m cinestudio app [--open]            # the local web app (all tabs: characters to videos and settings)
    python -m cinestudio doctor
    python -m cinestudio loras
    python -m cinestudio proxy start|stop|status
    python -m cinestudio sheet IMAGE --out DIR
    python -m cinestudio character create NAME --text "..." [--candidates 8] | pick NAME --seed N
    python -m cinestudio character import NAME (--image IMG | --sheet IMG) --text "..."
    python -m cinestudio character views NAME [--mode quality|fast] | style NAME --style-lora L | finish NAME
    python -m cinestudio character list | show NAME
    python -m cinestudio run new --blend shot.blend --character NAME --video "..." \
        [--style "..."] [--style-lora NAME] [--restyle] [--first-frame IMG] [--quality preview|final] [--seed N]
    python -m cinestudio run start RUN_ID [--detach] [--new-attempt]
    python -m cinestudio run status RUN_ID | run list | run cancel RUN_ID
    python -m cinestudio run events RUN_ID [--since N] [--follow]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import blender, character, runner, server, spawn, video
from .locks import pid_alive

ROOT = Path(__file__).resolve().parent.parent


def emit(obj) -> None:
    print(json.dumps(obj), flush=True)


class CliInputError(Exception):
    pass


class JsonArgumentParser(argparse.ArgumentParser):
    """argparse errors as a JSON result line (exit code 2), not as text on stderr."""

    def error(self, message):
        raise CliInputError(f"{self.prog}: {message}")


def _object_info(node: str) -> dict:
    with urllib.request.urlopen(f"{server.comfyui_url()}/object_info/{node}", timeout=15) as r:
        return json.loads(r.read())[node]["input"]


def _choices(node: str, field: str) -> list[str]:
    spec = _object_info(node)
    spec = {**spec.get("required", {}), **spec.get("optional", {})}[field]
    return list(spec[0]) if isinstance(spec[0], list) else list(spec[1].get("options", []))


def loras() -> list[str]:
    return _choices("LoraLoaderModelOnly", "lora_name")


def prop_args(items: list[str] | None, allow_empty: bool = False) -> dict:
    """--prop "PATTERN=WORDS" (repeatable) -> {pattern: words}."""
    out = {}
    for it in items or []:
        pat, sep, words = str(it).partition("=")
        if not sep or not pat.strip():
            raise CliInputError(f"--prop {it!r}: write PATTERN=WORDS, e.g. --prop \"Ball_*=red beach balls\"")
        if not words.strip() and not allow_empty:
            raise CliInputError(f"--prop {it!r}: give the words (the template's words need no --prop)")
        out[pat.strip()] = words.strip()
    return out


def prop_image_args(items: list[str] | None, allow_empty: bool = False) -> dict:
    """--prop-image "PATTERN=PICTURE" (repeatable) -> {pattern: picture path} (a fork: an empty PICTURE removes it)."""
    out = {}
    for it in items or []:
        pat, sep, path = str(it).partition("=")
        if not sep or not pat.strip():
            raise CliInputError(f"--prop-image {it!r}: write PATTERN=PICTURE, e.g. --prop-image \"Ball_*=ball.png\"")
        if not path.strip() and not allow_empty:
            raise CliInputError(f"--prop-image {it!r}: give the picture")
        out[pat.strip()] = path.strip().strip('"')
    return out


def check_style_lora(name: str) -> None:
    """Refuse a style LoRA that ComfyUI cannot load (run new; the MCP tool new_video_run uses the same check)."""
    try:
        from . import comfy
        comfy.ensure_running()               # the LoRA list comes from ComfyUI: start it when it is stopped
        available = loras()
    except Exception as exc:  # noqa: BLE001
        raise CliInputError(f"cannot check the style LoRA: ComfyUI does not answer ({exc})") from exc
    if name not in available:
        raise CliInputError(f"ComfyUI has no LoRA {name!r}; put the file in ComfyUI models/loras "
                            "and run: python -m cinestudio loras")


# model file -> (node, field) that lists it in ComfyUI
_MODEL_LISTS = {"checkpoints": ("CheckpointLoaderSimple", "ckpt_name"), "loras": ("LoraLoaderModelOnly", "lora_name"),
                "text_encoders": ("LTXAVTextEncoderLoader", "text_encoder"),
                "latent_upscale_models": ("LatentUpscaleModelLoader", "model_name")}


def doctor() -> dict:
    checks = {"blender": blender.check()}
    try:
        checks["ffmpeg"] = {"ok": True, "path": video.ffmpeg()}
    except Exception as exc:  # noqa: BLE001
        checks["ffmpeg"] = {"ok": False, "error": str(exc)}
    url = server.comfyui_url()
    checks["comfyui"] = {"ok": server.alive(url), "url": url}
    if not checks["comfyui"]["ok"]:
        checks["comfyui"]["error"] = "ComfyUI does not answer; start ComfyUI (with --enable-manager) or set COMFYUI_URL"
    proxy_url = os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8189")
    try:
        urllib.request.urlopen(proxy_url, timeout=5)
        checks["proxy"] = {"ok": True, "url": proxy_url}
    except urllib.error.HTTPError:
        checks["proxy"] = {"ok": True, "url": proxy_url}          # it answers (404 on / is normal)
    except Exception as exc:  # noqa: BLE001
        checks["proxy"] = {"ok": False, "url": proxy_url, "error": f"comfy-api-proxy does not answer ({exc}); start it: "
                                                                   "python -m cinestudio proxy start"}
    try:                                            # the video models of the chosen tier, by file (no ComfyUI needed)
        from . import models as _models
        vs = _models.status(None, ["video"])
        missing = [f"{f['dir']}/{f['file']}" for f in vs["files"] if f["state"] != "ok"]
        checks["models"] = {"ok": not missing, **({"missing": missing, "fix": "cinestudio models download --feature video"}
                                                  if missing else {})}
    except Exception as exc:  # noqa: BLE001 - no ComfyUI install found yet
        checks["models"] = {"ok": False, "error": str(exc)}
    try:
        from . import models, settings
        st = models.status()
        checks["model_tier"] = {"ok": True, "tier": st["tier"], "ready": st["ready"],
                                "missing_gb": round(st["missing_bytes"] / 1e9, 1), "models_dir": st["models_dir"],
                                **({} if st["ready"] else {"note": "the full-model tier is not complete: "
                                                           "cinestudio models download"})}
        checks["tokens"] = {"ok": True, **{k: {"set": v["set"], "source": v["source"]}
                                           for k, v in settings.status().items()}}
    except Exception as exc:  # noqa: BLE001 - a fresh machine without ComfyUI yet
        checks["model_tier"] = {"ok": True, "note": str(exc)}
    return {"ok": all(c.get("ok") for c in checks.values()), "checks": checks}


# ---------------------------------------------------------------- settings, tokens, models
def _read_secret(prompt: str) -> str:
    """A token from stdin: hidden typing on a terminal, else the first line of piped input."""
    if sys.stdin.isatty():
        import getpass
        return getpass.getpass(prompt, stream=sys.stderr)
    return sys.stdin.readline().strip()


def _settings(a) -> int:
    from . import models, settings
    if a.action == "show":
        emit({"ok": True, "settings": settings.get(), "tokens": settings.status(), "tiers": models.tier_names()})
        return 0
    if a.action == "tier":
        if not a.force:
            st = models.status(a.tier)
            if not st["ready"]:
                missing = [f["file"] for f in st["files"] if f["state"] != "ok"]
                raise CliInputError(f"tier {a.tier} is not downloaded yet ({len(missing)} files, "
                                    f"{st['missing_bytes'] / 1e9:.1f} GB): run `cinestudio models download --tier "
                                    f"{a.tier}` first, or use --force")
        emit({"ok": True, "settings": settings.set_tier(a.tier)})
        return 0
    if a.action == "backend":
        emit({"ok": True, "settings": settings.set_backend(a.backend)})
        return 0
    if a.action == "models-dir":
        emit({"ok": True, "settings": settings.set_models_dir(a.path)})
        return 0
    if a.action == "token":
        if a.op == "set":
            value = _read_secret(f"{settings.TOKENS[a.name]['label']}: ")
            emit({"ok": True, "token": a.name, **settings.set_token(a.name, value)})
        elif a.op == "check":
            r = settings.check_token(a.name)
            emit({"ok": r["ok"], "token": a.name, **r,
                  **({} if r["ok"] else {"error": r["detail"], "error_type": "TokenCheck"})})
            return 0 if r["ok"] else 1
        else:
            emit({"ok": True, "token": a.name, **settings.clear_token(a.name)})
        return 0
    raise CliInputError(f"unknown settings command {a.action}")


def _models(a) -> int:
    from . import models
    if a.action == "status":
        emit({"ok": True, **models.status(a.tier, a.feature)})
        return 0
    if a.action == "verify":
        r = models.verify(a.tier, a.feature, on_event=emit)
        emit({**r, **({} if r["ok"] else {"error": f"{len(r['bad'])} files do not match",
                                          "error_type": "HashMismatch"})})
        return 0 if r["ok"] else 1
    if a.action == "stop":
        emit({"ok": True, **models.request_stop()})
        return 0
    if a.action == "work-download":
        r = models.work_download(a.tier, a.feature, on_event=emit)
        emit({**r, **({} if r["ok"] else {"error": "some files did not download", "error_type": "DownloadFailed"})})
        return 0 if r["ok"] else 1
    if a.action == "download":
        from . import settings
        tier = a.tier or settings.get()["model_tier"]
        if a.detach:
            emit({"ok": True, **models.start_download(tier, a.feature)})
            return 0
        r = models.download(tier, a.feature, on_event=emit)
        if r["ok"] and a.tier and not a.feature:          # the jobs after it must ask for the files just downloaded
            settings.set_tier(tier)
            r = {**r, "tier_selected": tier}
        emit({**r, **({} if r["ok"] else {"error": "some files did not download", "error_type": "DownloadFailed"})})
        return 0 if r["ok"] else 1
    raise CliInputError(f"unknown models command {a.action}")


def _setup(a) -> int:
    """Quick setup in a terminal (a rented GPU machine): tokens (optional), tier, download. Questions go to stderr,
    the JSON lines to stdout."""
    from . import models, settings

    def say(text):
        print(text, file=sys.stderr, flush=True)
    ask = sys.stdin.isatty() and not a.yes
    if ask:
        for name, spec in settings.TOKENS.items():
            st = settings.status()[name]
            say(f"\n{spec['label']} - {'set (' + st['source'] + ')' if st['set'] else 'not set'}. {spec['help']}")
            value = _read_secret("Paste it (Enter = keep as is / skip): ").strip()
            if value:
                settings.set_token(name, value)
                r = settings.check_token(name)
                say(f"  {'OK' if r['ok'] else 'NOT OK'}: {r['detail']}")
    tier = a.tier or settings.get()["model_tier"]
    if ask and not a.tier:
        for t in models.tier_names():
            i = models.tier_info(t)
            say(f"  {t}: {i['label']}, {i['ram_gb']} GB RAM - {i['about']}")
        tier = input(f"Model tier [{tier}]: ").strip() or tier
    models.bundle(tier)
    st = models.status(tier)
    say(f"\nTier {tier}: {st['missing_bytes'] / 1e9:.1f} GB to download into {st['models_dir']}")
    r = models.download(tier, on_event=emit)
    if r["ok"]:
        settings.set_tier(tier)
    emit({**r, "settings": settings.get(), "tokens": settings.status(),
          **({} if r["ok"] else {"error": "some files did not download: run setup again to continue",
                                 "error_type": "DownloadFailed"})})
    return 0 if r["ok"] else 1


# ---------------------------------------------------------------- comfy-api-proxy service
def proxy(action: str) -> dict:
    """Start, stop or show the background comfy-api-proxy (job records in <home>/proxy_state)."""
    from comfy_api_proxy import cli as proxy_cli
    if action == "start":
        return _proxy_start()
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = proxy_cli.main([action])
    msg = (out.getvalue() + err.getvalue()).strip()
    if action == "status":
        return {"ok": True, "running": rc == 0, "message": msg}
    return {"ok": rc == 0, "message": msg}


def _proxy_start(host: str = "127.0.0.1", port: int = 8189, wait: float = 30) -> dict:
    """Start comfy-api-proxy in the background WITHOUT a console window.

    `comfy-api-proxy start` uses DETACHED_PROCESS. With a venv, python.exe is a launcher whose child
    (the real Python) then gets a new, visible console window, and closing that window stops the
    proxy. Here the proxy gets a hidden console (CREATE_NO_WINDOW) that its child shares. The proxy's
    own state file is written as `comfy-api-proxy start` writes it, so `status` and `stop` still work.
    A proxy that does not answer in time is stopped again (no orphan holding the port later)."""
    from comfy_api_proxy import service
    existing = service.read_state()
    if existing and service._pid_is_ours(service._state_pid(existing)):
        return {"ok": True, "already_running": True, "url": existing.get("url"), "pid": existing.get("pid")}
    if service._port_open(host, port):
        return {"ok": False, "error": f"port {port} is in use by another program", "error_type": "PortInUse"}
    state = runner.home() / "proxy_state"
    state.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "comfy_api_proxy.cli", "run", "--comfyui", server.comfyui_url(),
           "--host", host, "--port", str(port), "--state-dir", str(state)]
    kw = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} \
        if sys.platform.startswith("win") else {"start_new_session": True}
    log_path = service._log_file()
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"}, **kw)
    if not service._wait_port(host, port, proc, timeout=wait):
        service._terminate(proc.pid)
        return {"ok": False, "error": f"comfy-api-proxy did not start within {wait:.0f} s; see {log_path}",
                "error_type": "ProxyStartFailed"}
    url = f"http://{host}:{port}"
    service._state_file().write_text(json.dumps({"pid": proc.pid, "host": host, "port": port, "url": url}),
                                     encoding="utf-8")
    return {"ok": True, "url": url, "pid": proc.pid, "log": str(log_path)}


# ---------------------------------------------------------------- runs
def _start_detached(run_id: str, new_attempt: bool = False, wait: float = 30) -> dict:
    """Start a worker in the background and wait until it holds the run lock (or has ended)."""
    d = runner.run_dir(run_id)
    if runner.worker_alive(run_id):
        return {"ok": False, "id": run_id, "error": "the run already has a worker", "error_type": "RunBusy"}
    runner.check_runnable(run_id)                    # a retired (LTX-2.3) run: refuse before it takes a queue ticket
    from . import workqueue
    workqueue.add("video", run_id, runner.queue_label(runner._read_json(d / "manifest.json")))   # its place in line
    log_path = d / "worker.log"
    gen = runner.new_launch(run_id)        # this start is a new launch: a cancel made before it no longer applies
    cmd = [sys.executable, "-m", "cinestudio", "run", "start", run_id, "--launch", str(gen)] + \
        (["--new-attempt"] if new_attempt else [])
    # an independent process (see cinestudio.spawn): stopping the app does not stop the run
    pid, independent = spawn.detached(cmd, log_path, ROOT)
    warning = None if independent else ("Windows did not let the worker leave this process group: closing this "
                                         "terminal or app may stop the run")
    deadline = time.time() + wait
    while time.time() < deadline:
        if runner.worker_alive(run_id):
            return {"ok": True, "id": run_id, "worker": "running", "events": str(d / "events.jsonl"),
                    **({"warning": warning} if warning else {})}
        if not pid_alive(pid):                      # ended before or without taking the lock
            st = runner.status(run_id)
            tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
            ok = st["status"] == "done"
            return {"ok": ok, "id": run_id, "worker": "ended", "status": st["status"],
                    **({} if ok else {"error": st.get("error") or " | ".join(tail), "error_type": "WorkerEnded"})}
        time.sleep(0.3)
    return {"ok": False, "id": run_id, "error": f"the worker did not start within {wait:.0f} s; see {log_path}",
            "error_type": "WorkerStartTimeout"}


def _complete_lines(path: Path) -> list[str]:
    """Event lines that end with a newline (a line still being written is not returned yet)."""
    if not path.is_file():
        return []
    data = path.read_bytes()
    return [ln.decode("utf-8", errors="replace") for ln in data.split(b"\n")[:-1] if ln.strip()]


def events(run_id: str, since: int = 0, follow: bool = False) -> int:
    """Print the run's events from number `since` on, then a result line with "next" (the number to
    pass as --since next time). With follow: wait for new events until the run ends."""
    path = runner.run_dir(run_id) / "events.jsonl"
    n, quiet = since, 0
    while True:
        lines = _complete_lines(path)
        for line in lines[n:]:
            print(line, flush=True)
        n = max(n, len(lines))
        if not follow:
            break
        st = runner.status(run_id)
        last = json.loads(lines[-1])["event"] if lines else None
        if last and last.startswith("run_") and last[4:] in runner.TERMINAL_RUN and not st["worker_alive"]:
            break
        if not st["worker_alive"]:
            quiet += 1
            if st["status"] == "new" or quiet >= 3:      # never started, or the worker died without a final event
                emit({"ok": False, "id": run_id, "next": n, "status": st["status"],
                      "error": "no worker is running this run", "error_type": "NoWorker"})
                return 1
        else:
            quiet = 0
        time.sleep(1)
    emit({"ok": True, "id": run_id, "next": n, "status": runner.status(run_id)["status"]})
    return 0


# ---------------------------------------------------------------- arguments
def _parser() -> JsonArgumentParser:
    ap = JsonArgumentParser(prog="cinestudio", description="Blender shot + character sheet -> AI video (JSON output)")
    sub = ap.add_subparsers(dest="cmd", required=True, parser_class=JsonArgumentParser)
    ap_app = sub.add_parser("app", help="start the local web app (all tabs: characters, templates, backgrounds, "
                                        "music, videos, settings)")
    ap_app.add_argument("--port", type=int, default=8765)
    ap_app.add_argument("--open", action="store_true", help="open it in an app window")
    sub.add_parser("doctor", help="check Blender, ffmpeg, ComfyUI, the proxy and the models")
    sub.add_parser("mcp", help="serve cinestudio to agents as an MCP server (stdio)")
    sub.add_parser("loras", help="LoRA files that ComfyUI can load (for --style-lora)")
    su = sub.add_parser("setup", help="quick setup in the terminal: tokens, model tier, model download")
    su.add_argument("--tier", default=None, help="8gb or large (default: ask, or the saved tier)")
    su.add_argument("--yes", action="store_true", help="no questions: keep the tokens as they are, download now")
    sp = sub.add_parser("settings", help="model tier, backend and access tokens").add_subparsers(
        dest="action", required=True, parser_class=JsonArgumentParser)
    sp.add_parser("show", help="the settings and which tokens are set (never their values)")
    x = sp.add_parser("tier", help="the model tier for new work")
    x.add_argument("tier")
    x.add_argument("--force", action="store_true", help="also when files of the tier are missing")
    x = sp.add_parser("backend", help="local (this computer) or cloud (Comfy Cloud, needs the Comfy key)")
    x.add_argument("backend", choices=["local", "cloud"])
    x = sp.add_parser("models-dir", help="the ComfyUI models folder, when it cannot be found by itself")
    x.add_argument("path", nargs="?", default=None, help="empty: find it again")
    x = sp.add_parser("token", help="set, check or remove an access token (the value is read from stdin, never "
                                    "from the command line)")
    x.add_argument("op", choices=["set", "check", "clear"])
    x.add_argument("name", choices=["hf", "comfy", "civitai"])
    mp = sub.add_parser("models", help="the model files of a tier: status, download, verify").add_subparsers(
        dest="action", required=True, parser_class=JsonArgumentParser)
    for name, hlp in (("status", "what the tier needs and what is there"),
                      ("download", "download the missing files (several at once, resumable, hash-checked)"),
                      ("verify", "full SHA-256 check of the files that are there"),
                      ("work-download", "(internal) the background download worker"),
                      ("stop", "stop the background download")):
        x = mp.add_parser(name, help=hlp)
        if name == "work-download":
            x.add_argument("tier")
        elif name != "stop":
            x.add_argument("--tier", default=None, help="default: the tier in the settings")
        if name != "stop":
            x.add_argument("--feature", action="append", default=None,
                           help="only the files of a feature: video, character, music, select, pose, background, "
                                "mocap, full_stage2 (the full video model's stage-2 LoRA: optional in the 8gb tier) "
                                "(repeatable)")
        if name == "download":
            x.add_argument("--detach", action="store_true", help="run in the background (the app shows the progress)")
    cv = mp.add_parser("civitai", help="download one Civitai model version (a LoRA) with the optional Civitai key "
                       "(Settings > Access tokens or CIVITAI_TOKEN); SHA-256 checked")
    cv.add_argument("version_id", type=int, help="the number after modelVersionId= or /api/download/models/")
    cv.add_argument("--dest", default="loras", help="the folder under the models folder (default loras)")
    bg = sub.add_parser("background", help="the background library: 360 panoramas and backplates").add_subparsers(dest="action", required=True,
                                                                                     parser_class=JsonArgumentParser)
    bg.add_parser("list", help="the saved panoramas")
    bd = bg.add_parser("depth", help="the 360 depth of a saved panorama (MoGe-2), for --background-ground depth; "
                       "waits for its turn in the queue")
    bd.add_argument("name")
    bd.add_argument("--ref", default=None, help="(internal) the queue ticket made by the app")
    bn = bg.add_parser("new", help="a new 360 panorama from a place description (Qwen-Image + the 360 LoRA), then its "
                       "depth; waits for its turn in the queue")
    bn.add_argument("--scene", required=True, help="the place, for example 'a quiet harbour at dawn with fishing boats'")
    bn.add_argument("--style-preset", default=None, help="a style from the library (style list): its words, and the background records the style")
    bn.add_argument("--style", default="", help="look words (for example the video's style)")
    bn.add_argument("--seed", type=int, default=None)
    bn.add_argument("--name", default=None, help="the library name (default: from the text and the seed)")
    bn.add_argument("--no-depth", action="store_true", help="skip the 360 depth (it can be made later)")
    bn.add_argument("--ref", default=None, help="(internal) the queue ticket made by the app")
    bpl = bg.add_parser("plate", help="a new BACKPLATE: one picture of the place for the still camera of a shot "
                        "(Qwen-Image, the frame's shape, 4-9x sharper than a panorama there); waits for its turn")
    bpl.add_argument("--scene", required=True, help="the place")
    bpl.add_argument("--shot", required=True, help="the .blend whose still camera the picture is made for (its "
                     "height, look direction and horizon)")
    bpl.add_argument("--style-preset", default=None, help="a style from the library (style list): its words, and the background records the style")
    bpl.add_argument("--style", default="", help="look words (for example the video's style)")
    bpl.add_argument("--seed", type=int, default=None)
    bpl.add_argument("--name", default=None, help="the library name (default: from the text and the seed)")
    bpl.add_argument("--ref", default=None, help="(internal) the queue ticket made by the app")
    bfs = bg.add_parser("from-set", help="a backplate drawn OVER the template's own Blender set: Blender renders the "
                        "set alone from the still camera, Qwen-Image-Edit redraws it as the place, every object kept "
                        "in its place (waits for its turn)")
    bfs.add_argument("--scene", default="", help="the place the set stands for (default: the template's Set layer words, "
                                                 "shot layers NAME --set)")
    bfs.add_argument("--shot", required=True, help="the template .blend (its set and its still camera)")
    bfs.add_argument("--style-preset", default=None, help="a style from the library (style list): its words, and the background records the style")
    bfs.add_argument("--style", default="", help="look words: the characters' own style words")
    bfs.add_argument("--seed", type=int, default=None)
    bfs.add_argument("--name", default=None)
    bfs.add_argument("--ref", default=None, help="(internal) the queue ticket made by the app")
    bim = bg.add_parser("import", help="your own picture of the place as a backplate (fitted to the frame in each run)")
    bim.add_argument("file")
    bim.add_argument("--name", default=None)
    bim.add_argument("--scene", default="", help="a few words about the place (shown in the library)")
    bx = bg.add_parser("delete", help="delete a saved background (runs keep their own copy)")
    bx.add_argument("name")
    bs = bg.add_parser("style", help="redraw a saved panorama with LTX-2.5 and a style LoRA (the video's look); saved "
                       "as a new panorama, the original stays; waits for its turn in the queue")
    bs.add_argument("name")
    bs.add_argument("--style-lora", required=True)
    bs.add_argument("--keep", type=float, default=0.5, help="about this part of the panorama's structure stays (0..1; "
                    "the start noise is 1 - keep)")
    bs.add_argument("--profile", default=None, help="the LTX graphs (default the distilled LTX-2.5)")
    bs.add_argument("--strength", type=float, default=0.8, help="the LoRA strength")
    bs.add_argument("--style", default="", help="style words (for example the LoRA's trigger word and look)")
    bs.add_argument("--seed", type=int, default=333)
    bs.add_argument("--new-name", default=None)
    bs.add_argument("--ref", default=None, help="(internal) the queue ticket made by the app")
    pp = sub.add_parser("proxy", help="start, stop or show the background comfy-api-proxy")
    pp.add_argument("action", choices=["start", "stop", "status"])
    cp = sub.add_parser("character", help="the character library").add_subparsers(dest="action", required=True,
                                                                                    parser_class=JsonArgumentParser)
    c = cp.add_parser("create", help="front-view candidates from text (Z-Image base); then: pick")
    c.add_argument("--display-name", default="", help="the name the app shows (any text; NAME stays the ID)")
    c.add_argument("name")
    c.add_argument("--text", required=True, help="what the character looks like (also used in every video prompt)")
    c.add_argument("--style", default="", help="the look (default: the fantasy anime style)")
    c.add_argument("--candidates", type=int, default=8)
    c.add_argument("--seed", type=int, default=1, help="first seed; candidates use seed, seed+1, ...")
    c.add_argument("--width", type=int, default=1024)
    c.add_argument("--height", type=int, default=1536)
    c.add_argument("--style-lora", default=None, help="optional Z-Image LoRA for the look of the design")
    c.add_argument("--body", choices=character.BODY_TYPES, default="human",
                   help="human (default): `views` first poses the front like the character's own mannequin (AnyPose)")
    c = cp.add_parser("proxy", help="the character's proxy kit: simple clay parts put on the mannequin in every run "
                      "(bpy_jobs/proxy_kit.py), so the layout carries the character's outline")
    c.add_argument("name")
    c.add_argument("--spec", required=True, help='JSON, e.g. {"mask": {"eyes": true}, "hair": {"length": "short"}}; '
                   '"{}" = no kit')
    c = cp.add_parser("mannequin", help="the character's own mannequin body in every run and in `repose` (AnyPose): "
                      "its height and proportions (1.0 = the template's mannequin); given keys change, others stay")
    c.add_argument("name")
    c.add_argument("--scale", type=float, default=None, help="the whole mannequin, e.g. 0.8 for a short character")
    c.add_argument("--head", type=float, default=None, help="the head (with the mask and hair parts), e.g. 1.4 chibi")
    c.add_argument("--neck", type=float, default=None, help="the neck's length, e.g. 0.4 when the head sits on the "
                   "shoulders (the head keeps its own size)")
    c.add_argument("--torso", type=float, default=None, help="the spine and chest (shorter and narrower; the arms and "
                   "the head keep their own sizes)")
    c.add_argument("--legs", type=float, default=None, help="both legs (thigh, shin, foot), e.g. 0.75 for short legs")
    c.add_argument("--arms", type=float, default=None, help="both arms (upper arm, forearm, hand)")
    c.add_argument("--default", action="store_true", help="the template's mannequin as it is (every value 1.0); a "
                   "character needs a mannequin body before a Blender run can use it (the proxy kit is optional)")
    c = cp.add_parser("pick", help="choose the front view among the candidates")
    c.add_argument("name")
    c.add_argument("--seed", type=int, required=True)
    c = cp.add_parser("import", help="a character from one image (views are made) or from a full sheet (used as is)")
    c.add_argument("--display-name", default="", help="the name the app shows (any text; NAME stays the ID)")
    c.add_argument("name")
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="the front view (the other slots are optional)")
    g.add_argument("--sheet", help="a finished sheet, used as it is")
    for slot in ("right", "back", "left", "face"):
        c.add_argument(f"--{slot}", default=None, help=f"optional {slot} image (else made from the front view)")
    c.add_argument("--text", default="", help="optional: the costume in exact words (the video prompt and the look "
                   "images use it to keep small details)")
    c.add_argument("--style", default="", help="optional: the look in words (default none: the images give the look)")
    c.add_argument("--body", choices=character.BODY_TYPES, default="human",
                   help="human (default): `views` first poses the front like the character's own mannequin (AnyPose)")
    c = cp.add_parser("edit", help="change details of the front view or of one view by an instruction (Qwen-Image-Edit)")
    c.add_argument("name")
    c.add_argument("--target", choices=character.TARGETS, default="front", help="the image to change (default front)")
    c.add_argument("--region", default=None, help="x0,y0,x1,y1: edit only this box (crop and stitch; good for hands)")
    _quality_args(c, denoise=True)
    c.add_argument("--instruction", required=True, help="e.g. 'make the scarf like the one in Picture 2'")
    c.add_argument("--reference", default=None, help="optional second image (Picture 2: one detail, tightly cropped) "
                   "the instruction can point to")
    c.add_argument("--mode", choices=["quality", "fast"], default="quality")
    c.add_argument("--seed", type=int, default=None, help="default: a new random seed")
    c = cp.add_parser("undo", help="undo the last change (edit, redo or pick) of one image")
    c.add_argument("name")
    c.add_argument("--target", choices=character.TARGETS, default="front")
    c = cp.add_parser("redo-view", help="make one view again from the front view with another seed")
    c.add_argument("name")
    c.add_argument("view", choices=list(character.VIEWS))
    c.add_argument("--seed", type=int, default=None, help="default: a new random seed")
    c.add_argument("--mode", choices=["quality", "fast"], default="quality")
    _quality_args(c, angles=True)
    c = cp.add_parser("repose", help="pose the FRONT like the character's own mannequin (AnyPose; Blender renders "
                                      "the bare mannequin with the character's body), then make the views and the "
                                      "sheet from it (body type human does this in `views` by itself)")
    c.add_argument("name")
    c.add_argument("--shot", default=None, help="the .blend whose mannequin gives the pose (default: the built-in templates' rig, shan_shen_bu_balls)")
    c.add_argument("--seed", type=int, default=None, help="the AnyPose seed (default: the character's seed)")
    c = cp.add_parser("views", help="right side, back, left side and face views (Qwen-Image-Edit) + the sheet")
    c.add_argument("name")
    c.add_argument("--mode", choices=["quality", "fast"], default="quality")
    c.add_argument("--seed", type=int, default=None)
    c.add_argument("--distance", choices=["medium", "wide"], default="medium",
                   help="the Multiple-Angles distance of the body views (medium = the front's framing; wide drew rooms)")
    c.add_argument("--face", choices=["front", "qwen"], default="front",
                   help="the face panel: cut from the front (default) or the Qwen close-up")
    c = cp.add_parser("style", help="redraw the sheet with LTX and the scene's style LoRA; the character is then ready")
    c.add_argument("name")
    c.add_argument("--style-lora", required=True)
    c.add_argument("--keep", type=float, default=0.5, help="how much of the sheet stays (0..1)")
    cp.add_parser("finish", help="use the sheet as it is; the character is then ready").add_argument("name")
    cp.add_parser("show").add_argument("name")
    cp.add_parser("delete", help="delete a character for good").add_argument("name")
    rn = cp.add_parser("rename", help="give a character a new ID (its folder moves; stored paths follow); for the "
                       "name the app shows, use display-name")
    rn.add_argument("name")
    rn.add_argument("new_name")
    us = cp.add_parser("use-style", help="give the character a style from the library (its words go into the "
                                          "character's style; finished pictures do not change)")
    us.add_argument("name")
    us.add_argument("style")
    sk = cp.add_parser("skin", help="a new SKIN of the character in a library style (GPU job): its front redrawn in "
                                    "the style by Qwen (the pose stays), then the views and the sheet; one character, "
                                    "more than one look")
    sk.add_argument("name", help="the skin to start from (usually the main record)")
    sk.add_argument("style", help="a style from the library (style list)")
    sk.add_argument("--seed", type=int, default=None)
    x = cp.add_parser("skins", help="the character's skins (the main record first)")
    x.add_argument("name")
    x = cp.add_parser("link-skin", help="make an existing character a skin of another (instant; e.g. kitsune-3d of kitsune)")
    x.add_argument("name")
    x.add_argument("--to", required=True, help="the main record's ID")
    x = cp.add_parser("unlink-skin", help="make a skin a character of its own again (instant)")
    x.add_argument("name")
    dn = cp.add_parser("display-name", help="the name the app shows (any text, spaces allowed; the ID stays)")
    dn.add_argument("name")
    dn.add_argument("text")
    cp.add_parser("list")
    _prop_parser(sub)
    sp = sub.add_parser("sheet", help="prepare a character sheet (background to black, find the views)")
    sp.add_argument("image")
    sp.add_argument("--out", required=True)
    ep = sub.add_parser("comfy", help="the ComfyUI engine: cinestudio starts it hidden in the background").add_subparsers(
        dest="action", required=True, parser_class=JsonArgumentParser)
    ep.add_parser("detect", help="find the installed ComfyUI and save its start command")
    ep.add_parser("start", help="start ComfyUI in the background (if it does not answer)")
    ep.add_parser("stop", help="stop the ComfyUI that cinestudio started (only when no work waits or runs)")
    ep.add_parser("status")
    gp = sub.add_parser("gpu", help="Windows: which graphics adapter each app uses (integrated or the GPU)").add_subparsers(
        dest="action", required=True, parser_class=JsonArgumentParser)
    gp.add_parser("list", help="running apps and apps with a setting, with their choice")
    gs = gp.add_parser("set", help="set one app (restart the app to use it)")
    gs.add_argument("exe", help="full path of the app's .exe")
    gs.add_argument("choice", choices=["default", "integrated", "gpu"])
    gp.add_parser("all-integrated", help="move the running everyday apps to the integrated graphics")
    gp.add_parser("reset", help="restore every app that cinestudio changed")
    qp = sub.add_parser("queue", help="the shared queue of character steps and videos").add_subparsers(
        dest="action", required=True, parser_class=JsonArgumentParser)
    qp.add_parser("list", help="what runs now and what waits, first in line first")
    qc = qp.add_parser("cancel", help="cancel one queued or running request (explicit; nothing else stops work)")
    qc.add_argument("kind", choices=["character", "video", "song", "lab", "template"])
    qc.add_argument("ref", help="the job id (character), run id (video) or song name (song)")
    wfp = sub.add_parser("workflow", help="the user's copies of the built-in ComfyUI graphs (the Workflows view)"
                         ).add_subparsers(dest="action", required=True, parser_class=JsonArgumentParser)
    x = wfp.add_parser("list", help="a built-in graph's copies and which graph new jobs use")
    x.add_argument("name")
    x = wfp.add_parser("copy", help="make a copy of a built-in graph")
    x.add_argument("name")
    x.add_argument("--label", default=None)
    x = wfp.add_parser("set", help="set one literal value in a copy (a JSON value: 0.5, true, \"text\")")
    for k in ("name", "copy", "node", "input", "value"):
        x.add_argument(k)
    x = wfp.add_parser("reset", help="the built-in's values again in a copy (one node or all)")
    x.add_argument("name")
    x.add_argument("copy")
    x.add_argument("node", nargs="?")
    x = wfp.add_parser("use", help="which graph new jobs use: a copy id or 'builtin'")
    x.add_argument("name")
    x.add_argument("copy")
    sy = sub.add_parser("style", help="the style library: one set of look words for characters, templates, backgrounds "
                                      "and videos").add_subparsers(dest="action", required=True,
                                                                   parser_class=JsonArgumentParser)
    sy.add_parser("list", help="every style, which assets use it, the default")
    x = sy.add_parser("show", help="one style")
    x.add_argument("name")
    for verb in ("new", "edit"):
        x = sy.add_parser(verb, help=("make a style" if verb == "new" else "change a style (its version goes up; "
                                                                           "assets made before are marked older)"))
        x.add_argument("name")
        x.add_argument("--label", default=None)
        x.add_argument("--words", default=None, help="the look ('stylized cinematic 3D animation, ...')")
        x.add_argument("--light", default=None, help="the light ('evenly lit by soft neutral light from the front')")
        x.add_argument("--negative-image", default=None, help="excluded looks for image jobs")
        x.add_argument("--lora", default=None, help="the LTX style LoRA file (an empty text = none)")
        x.add_argument("--lora-strength", type=float, default=None)
        x.add_argument("--reference", default=None, help="a picture: a tight crop in this style (an empty text = none)")
    x = sy.add_parser("delete", help="delete one of your styles (built-in styles stay)")
    x.add_argument("name")
    x = sy.add_parser("set-default", help="the style new assets get by default")
    x.add_argument("name")
    x = sy.add_parser("make-reference", help="reference picture candidates made by the VIDEO model (LTX-2.5 + the "
                                             "style's LoRA) from text: a GPU job in the queue; then use-reference")
    x.add_argument("name")
    x.add_argument("--subject", default=None, help="what the pictures show (default: a young woman in a hooded jacket "
                                                   "standing in a city street, full body)")
    x.add_argument("--seeds", default=None, help="comma-separated seeds, one candidate each (default 202,303,404,505)")
    x.add_argument("--video-model", choices=("distilled", "full"), default=None,
                   help="the LTX-2.5 model (default distilled, the model of the layout videos)")
    x.add_argument("--ref", default=None, help=argparse.SUPPRESS)          # the app's queue ticket (sref-...)
    x = sy.add_parser("use-reference", help="make a picture the style's reference (instant); Qwen conversions get its "
                                            "crop box as Picture 2")
    x.add_argument("name")
    x.add_argument("file")
    x.add_argument("--crop", default=None, help="x0,y0,x1,y1 in the picture's pixels (default: a square on the "
                                                "clothes below the face)")
    fl = sub.add_parser("film", help="a film from several scenes (runs or videos) with ONE continuous score").add_subparsers(
        dest="action", required=True, parser_class=JsonArgumentParser)
    fa = fl.add_parser("assemble", help="join the scenes with hard cuts (each keeps its own sound: ambience, effects) "
                                        "and lay one score under the whole film (it never restarts at a cut)")
    fa.add_argument("name")
    fa.add_argument("--scene", action="append", required=True, help="a run ID (its final.mp4) or a video file; repeat, "
                                                                    "in film order")
    fa.add_argument("--score", default=None, help="a library song (song make) as the score")
    fa.add_argument("--score-file", default=None, help="or a music file as the score")
    fa.add_argument("--score-start", type=float, default=0.0, help="seconds into the score where the film starts")
    fa.add_argument("--music-db", type=float, default=-8.0, help="the score's level under the scene sound (dB)")
    fa.add_argument("--scene-db", type=float, default=0.0, help="the scene sound's level (dB)")
    fa.add_argument("--no-duck", action="store_true", help="do not lower the music under strong effects")
    fa.add_argument("--fade-out", type=float, default=2.0, help="the score's fade at the end (s)")
    fa.add_argument("--loudness", type=float, default=-16.0, help="the film's loudness (LUFS, EBU R128)")
    fa.add_argument("--replace", action="store_true", help="make an existing film of this name again")
    fl.add_parser("list", help="the films made so far")
    so = sub.add_parser("song", help="original songs for a shot (ACE-Step 1.5) and their beats").add_subparsers(
        dest="action", required=True, parser_class=JsonArgumentParser)
    sm = so.add_parser("make", help="make one song (waits for its turn in the queue; prints progress events)")
    sm.add_argument("name")
    sm.add_argument("--tags", required=True, help="style, instruments, mood (the tempo goes in --bpm)")
    sm.add_argument("--lyrics", default="[instrumental]", help="lyrics with [Verse]/[Chorus] tags, or [instrumental]")
    sm.add_argument("--bpm", type=int, default=120)
    sm.add_argument("--seconds", type=float, default=30.0)
    sm.add_argument("--key", default="E minor")
    sm.add_argument("--seed", type=int, default=31)
    sm.add_argument("--language", default="en")
    so.add_parser("list")
    so.add_parser("show").add_argument("name")
    sb = so.add_parser("beats", help="beat and downbeat times of a song (or any sound file), and as shot frames")
    sb.add_argument("name", help="a song name or a sound file")
    sb.add_argument("--bpm", type=float, default=None)
    sb.add_argument("--start", type=float, default=0.0, help="the song's second that plays on the shot's first frame")
    sb.add_argument("--fps", type=float, default=24.0)
    sp2 = so.add_parser("place", help="a copy of a .blend with the song as its sound (the run then uses it)")
    sp2.add_argument("blend")
    sp2.add_argument("name")
    sp2.add_argument("--start", type=float, default=0.0, help="the song's second that plays on the shot's first frame")
    sp2.add_argument("--out", required=True)
    sp2.add_argument("--file", default=None, help="a sound file of this song to place instead (for example a loop)")
    sl = so.add_parser("loop", help="a seamless loop of N beats from a song, stretched to exactly F frames")
    sl.add_argument("name")
    sl.add_argument("--beats", type=int, default=16)
    sl.add_argument("--frames", type=int, default=240)
    sl.add_argument("--fps", type=float, default=24.0)
    sl.add_argument("--start", type=float, default=None, help="the section's first beat (seconds); default: the best")
    tp = sub.add_parser("shot", help="templates: every Blender shot for the Video tab (the app's Templates tab)"
                        ).add_subparsers(dest="action", required=True, parser_class=JsonArgumentParser)
    tl = tp.add_parser("layers", help="the template's three layers (Mannequin* = characters, Props, Set) and their "
                                       "words: the same words go into the background, video and look texts")
    tl.add_argument("name")
    tl.add_argument("--set", dest="set_description", default=None, help="what the Set stands for ('a school gym with "
                                                                        "a ball cart at the far wall')")
    tl.add_argument("--prop", action="append", default=[], metavar="NAME=TEXT",
                    help="a prop's (or a pattern's: Ball_*) words; repeat; an empty TEXT removes it")
    tl.add_argument("--rescan", action="store_true", help="read the layers from the .blend again")
    tct = tp.add_parser("clip-texts", help="the template's clip texts: one ACTION text per clip (only what that clip's "
                                          "frames show) and a context for every clip; runs use them (AGENTS.md 7.2)")
    tct.add_argument("name")
    tct.add_argument("--clip", nargs=2, action="append", metavar=("K", "TEXT"), default=[],
                     help="clip K's actions ({character} = the run's words for its character); repeat; '' removes it")
    tct.add_argument("--context", default=None,
                     help="words for every clip: lasting states and orientation ('{character} faces the camera the "
                          "whole time'); '' removes them")
    tt = tp.add_parser("timeline", help="the template's motion events from its rig (which hand waves when, feet planted); "
                                        "stored in its shot.json; runs use them with --clip-text auto")
    tt.add_argument("name")
    tm = tp.add_parser("make-dance", help="a dance shot on a song's beats (a detached worker; poll `shot show`)")
    tm.add_argument("name")
    tm.add_argument("--song", required=True, help="a done song from the library (see: song list)")
    tm.add_argument("--routine", default="S1,S2",
                    help="routines in order: S1 (shuffle + kicks), S2 (points + step-touch), A (locking), B (groove), "
                         "F (finale, not in a loop); add ~ for the mirrored routine")
    tm.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True,
                    help="a seamless loop (the last frame repeats the first; default) or a dance from --start")
    tm.add_argument("--seconds", type=float, default=10.0, help="the loop's length (the nearest 8/16/32 beats)")
    tm.add_argument("--start", type=float, default=None, help="without --loop: the song's second on frame 1")
    tm.add_argument("--night", action="store_true", help="the night light of the template set")
    tm.add_argument("--wait", action="store_true", help="build in this process and print progress (no worker)")
    tx = tp.add_parser("from-video", help="a Blender template from a dance VIDEO: SAM 3D Body reads the person's 3D "
                                          "motion and the mannequin copies it (still front camera, plain floor: a 360 "
                                          "panorama gives the place in the run). A detached worker (poll `shot show`)")
    tx.add_argument("name")
    tx.add_argument("--video", required=True, help="the video file, best facing a still camera: ONE dancer with "
                                                   "--people 1 (the model takes the whole frame as one person)")
    tx.add_argument("--start", type=float, default=0.0, help="the video's second where the template starts")
    tx.add_argument("--seconds", type=float, default=None, help="the template's length (default 10, at most 30)")
    tx.add_argument("--sound", choices=["video", "none"], default="video",
                    help="video = the video's own sound (the template is then PRIVATE: someone else's recording is "
                         "not for the built-in library or a contest submission); none = no sound")
    tx.add_argument("--people", choices=["1", "2", "3", "4", "all", "pick"], default="1",
                    help="1 = one dancer (default); 2 / 3 / 4 = SAM 3 finds every person and that many seen longest "
                         "and largest become Mannequin (the picture's left: the run's first --character), Mannequin2, "
                         "... left to right, at their places from the video; all = every person seen long enough (up "
                         "to 4); pick = stop after the scan (status 'pick', a numbered sheet in people/sheet.jpg), then "
                         "`shot pick-people NAME --tracks 0,1,2`")
    tx.add_argument("--tracks", default=None, help="the people to copy by their SAM 3 numbers (e.g. 0,1,2; at most 4), "
                                                   "when known from an earlier scan of the same video part")
    tx.add_argument("--description", default="", help="what the template shows")
    tx.add_argument("--wait", action="store_true", help="make it in this process and print progress (no worker)")
    tpp = tp.add_parser("pick-people", help="after `from-video --people pick`: build the template from the chosen people "
                                            "(no second GPU job; Blender in a detached worker)")
    tpp.add_argument("name")
    tpp.add_argument("--tracks", required=True, help="1 to 4 SAM 3 numbers from people/sheet.jpg, e.g. 0,2,1")
    tpp.add_argument("--wait", action="store_true", help="build in this process and print progress (no worker)")
    tp.add_parser("work-mocap", help="internal: the background worker of a template from a video").add_argument("name")
    tk = tp.add_parser("mask", help="the figure to replace in a video template: SAM 2 follows it through the video "
                                    "(seeded from DWPose; add --clicks to correct). Writes mask.mp4 + overlay.mp4")
    tk.add_argument("name")
    tk.add_argument("--clicks", default=None, help='JSON list: [{"frame": 0, "x": 384, "y": 220, "positive": true}]')
    tp.add_parser("work-mask", help="internal: the background worker of the app's figure selection").add_argument("name")
    tv = tp.add_parser("add-video", help="store a template VIDEO (a stand-in figure in a scene, made from text) as a "
                                         "template; a run replaces the stand-in with the character")
    tv.add_argument("name")
    tv.add_argument("--video", required=True)
    tv.add_argument("--description", default="")
    ta = tp.add_parser("add", help="store any Blender shot as a template (copied; a Workbench preview with its sound "
                                   "is made). Agents: save every shot you build this way")
    ta.add_argument("name")
    ta.add_argument("--blend", required=True, help="the saved .blend (camera, set, mannequin rig; optional sound strips)")
    ta.add_argument("--description", default="", help="what the shot is: characters, action, camera, length")
    ta.add_argument("--move", action="store_true", help="move the .blend instead of copying it")
    ta.add_argument("--no-preview", action="store_true", help="skip the Workbench preview")
    tc = tp.add_parser("check", help="the template rules as automatic checks (bone scales, actions across clip joins, "
                                     "end hold, frame 1, camera speed, layers, figure and prop sizes, set colours, clip "
                                     "texts, sound, joint pops): one headless Blender read (no render, a few seconds), "
                                     "cached by the .blend's sha256; the summary goes into shot.json \"check\"")
    tc.add_argument("name")
    tc.add_argument("--json", action="store_true", help="also print the measured facts (per-frame joints, camera, "
                                                        "props) for tools")
    tc.add_argument("--fresh", action="store_true", help="read the .blend again (ignore the cache)")
    tp.add_parser("list")
    tp.add_parser("show").add_argument("name")
    tp.add_parser("delete").add_argument("name")
    tp.add_parser("work", help="internal: the background worker of a template").add_argument("name")
    jp = sub.add_parser("job", help="character jobs").add_subparsers(dest="action", required=True,
                                                                     parser_class=JsonArgumentParser)
    jp.add_parser("work", help="internal: the background worker of a queued job").add_argument("job_id")
    jp.add_parser("show").add_argument("job_id")
    jl = jp.add_parser("list")
    jl.add_argument("--character", default=None)
    rp = sub.add_parser("run", help="make and control runs").add_subparsers(dest="action", required=True,
                                                                             parser_class=JsonArgumentParser)
    n = rp.add_parser("new", help="freeze the inputs of a run (nothing is generated yet)")
    n.add_argument("--blend", default=None, help="saved .blend with the camera, set and placeholder character")
    n.add_argument("--template", default=None, help="OR a video template (made from text; see: shot list, kind video)")
    n.add_argument("--negative", default=None, help="the clips' negative prompt (the full model uses it; default: a short "
                                                       "quality list)")
    n.add_argument("--background", default=None,
                   help="Blender shots: 'text' = a 360 panorama of the place (from --background-text or --video), "
                        "'saved:NAME' = a panorama from the background library; 'plate' = ONE picture of the place for "
                        "a still camera (sharper; from --background-text, or your own --background-image); 'none' = "
                        "the template's own set. Default: the template's default background when it has one (a "
                        "built-in template's backplate, shot.json background) and no other background option is "
                        "given, else the template's own set")
    n.add_argument("--background-text", default=None, help="the place for the panorama or the backplate (default: the "
                   "video text)")
    n.add_argument("--background-image", default=None, help="--background plate: your own picture of the place (it is "
                   "cropped to the frame's shape); without it the backplate is made from the text")
    n.add_argument("--background-seed", type=int, default=None)
    n.add_argument("--background-ground", choices=("dome", "depth"), default=None,
                   help="the panorama's ground in Blender: depth (default; MoGe-2 360 depth: every direction at its own "
                        "distance, real parallax off the centre) or dome (one radius, the old way)")
    n.add_argument("--figure-mask", choices=("off", "control"), default=None,
                   help="with --template: off; control = the skeleton of the selected figure only (the whole frame is "
                        "still generated: no mask limits where the character is drawn). Needs `shot mask`")
    n.add_argument("--character", action="append", default=None,
                   help="a character from the library (see: character list). REPEATABLE, once per template slot in slot "
                        "order (slot 1 = collection Mannequin, 2 = Mannequin2, ...; at most 4; the same character may "
                        "fill two slots): several characters = a layout run with one group sheet and one look crop per "
                        "figure. On a one-mannequin shot the mannequin is copied for each (the same motion)")
    n.add_argument("--character2", default=None, help="the old form of a second --character (the next slot)")
    n.add_argument("--spacing", type=float, default=None, help="several characters on a one-mannequin shot: the "
                   "copies of the mannequin (the same motion) stand this far apart, in m (default 1.5)")
    n.add_argument("--prop", action="append", default=None, metavar="PATTERN=WORDS",
                   help="the run's own words for a template prop group (repeatable), e.g. --prop \"Ball_*=red and "
                        "white beach balls\"; the patterns: shot layers NAME (empty words = the template's)")
    n.add_argument("--prop-image", action="append", default=None, metavar="PATTERN=PICTURE",
                   help="a picture of a template prop (repeatable, at most 2), e.g. --prop-image \"Ball_*=ball.png\": "
                        "frozen into the run and drawn into a prop panel of the run's sheet (bottom-right), with a "
                        "'**Prop:**' line in the sheet text (the words: --prop or the template's). A prop on black "
                        "in the characters' style is best. Not A/B-tested yet")
    n.add_argument("--prop-ref", action="append", default=None, metavar="PATTERN=PROP_ID",
                   help="a LIBRARY prop for a template prop group (repeatable), e.g. --prop-ref 'Ball_*=dodgeball': "
                        "its front becomes the group's prop picture and its words the group's words (prop list); "
                        "--prop / --prop-image for the same pattern win")
    n.add_argument("--character2-collection", default=None)
    n.add_argument("--character2-rig", default=None)
    n.add_argument("--sheet", default=None, help="OR a character sheet image (face close-up + front/side/back is best)")
    n.add_argument("--character-text", default=None, help="with --sheet: description of the sheet, panel by panel")
    n.add_argument("--video", required=True, help="what happens in the shot. A shot of several clips: the SHARED "
                   "context only (who, place, light, look, lasting states), in every clip; each clip's actions come "
                   "from --clip-video K or the template's clip texts (AGENTS.md 7.2)")
    n.add_argument("--style", default="", help="the look, e.g. '2D anime cel-shaded animation'")
    n.add_argument("--style-preset", default=None, help="a style from the library (style list): its words and its "
                                                         "video LoRA; default: the character's own style")
    n.add_argument("--style-lora", default=None, help="file name of a LoRA in ComfyUI models/loras (see: loras)")
    n.add_argument("--style-lora-strength", type=float, default=None)
    n.add_argument("--restyle", action="store_true", help="redraw the sheet with the style LoRA before the video")
    n.add_argument("--first-frame", default=None, help="image that becomes frame 0 of the clip (e.g. a front view)")
    n.add_argument("--describe-facing", action="store_true", default=None,
                   help="add to each clip's prompt where every character stands and which way it faces (from Blender)")
    n.add_argument("--sound", choices=("auto", "ltx"), default=None,
                   help="auto = the shot's own sound (a song in Blender's Video Sequencer, or speakers) replaces LTX's "
                        "when the shot has one (default); ltx = keep LTX's own sound")
    n.add_argument("--song", default=None, help="a song from the library (see: song list) as the video's sound; "
                                                 "it wins over the shot's own sound")
    n.add_argument("--song-start", type=float, default=0.0, help="the song's second that plays on frame 1")
    n.add_argument("--body-type", choices=("chibi", "standard"), default=None,
                   help="skeleton drawing: chibi = thick and bright, for big-head characters (default); standard = "
                        "the DWPose node's drawing, for normal human proportions")
    n.add_argument("--ingredients-strength", type=float, default=None,
                   help="strength of the character sheet guide (Ingredients IC-LoRA; default 1.0)")
    n.add_argument("--look-extra", default=None, help="layout: words added to the look image instruction")
    n.add_argument("--target-text", choices=("full", "short"), default=None,
                   help="the clip prompt's Target part: short (default) = 'The character from the reference sheet.' "
                        "(identity by the pictures; the --video text carries the 2-3 key marks and the place; the "
                        "description's key marks are added when the text names none); full = the character's whole "
                        "description again and 'The place: ...' (the old target)")
    n.add_argument("--audio-relay", choices=("off", "carry"), default=None,
                   help="layout long shots: carry (default with the latent relay) = each clip starts with the clip "
                        "before's audio tail frozen, so the sound continues across joins; off = every clip makes its "
                        "own sound")
    n.add_argument("--relay", choices=("off", "latent"), default=None, help="layout long shots: latent = clip k > 1 "
                   "continues from clip k-1's own latents (no decode/encode round trip at the joins)")
    n.add_argument("--relay-strength", type=float, default=None, help="the latent relay guide's strength (0..1]")
    n.add_argument("--clip-text", choices=("manual", "auto"), default=None,
                   help="auto = each clip's text written from Blender (the action, feet planted, the camera's start and "
                        "end views); --video is then the place and light only")
    n.add_argument("--clip-video", nargs=2, action="append", metavar=("K", "TEXT"), default=None,
                   help="clip K's own ACTIONS (repeat per clip): only what the Blender render shows in that clip's "
                        "frames. Each clip's prompt = --video (the shared context: who, place, light, look) + the clip "
                        "context + its actions; a clip without its own text takes the template's (shot clip-texts)")
    n.add_argument("--clip-context", default=None,
                   help="words for EVERY clip (lasting states: 'faces the camera the whole time') instead of the "
                        "template's clip context ('' = none)")
    n.add_argument("--character-words", default=None,
                   help="what {character} in the template's clip texts becomes (default: 'the character')")
    n.add_argument("--look-keyframes", choices=("auto", "off"), default=None,
                   help="layout: keyframe looks per clip (auto, default) or none (off: for example when the shot holds "
                        "the real 3D character, whose render already shows every view)")
    n.add_argument("--pose-strength", type=float, default=None,
                   help="layout: + the rig's skeleton as a stage-1 guide (0 = off; default 0)")
    n.add_argument("--pose-body", choices=("standard", "chibi"), default=None, help="the pose guide's skeleton drawing")
    n.add_argument("--adain-ref", choices=("look", "clip1"), default=None,
                   help="the AdaIN reference of clips 2..n: clip1 = clip 1's latent (default for layout runs, AdaIN 0.3); "
                        "look = the clip's frame -1 image (the old default, AdaIN 0.5)")
    n.add_argument("--chain-light", choices=("off", "look", "tone"), default=None,
                   help="look = each chained look gets the run look's lightness (the scene darkened at every join); "
                        "tone = also its colourfulness and contrast (the background lost ~8-10%% colour per join)")
    n.add_argument("--chain-sharpen-target", type=float, default=None,
                   help="match mode: the chained look's sharpness target = clip 1's x this (default 1.0; a clip comes out "
                        "softer than its look)")
    n.add_argument("--chain-sharpen-mode", choices=("fixed", "match"), default=None,
                   help="match = sharpen a chained look only up to clip 1's sharpness (fixed added up clip by clip)")
    n.add_argument("--chain-colour", choices=("off", "look", "match"), default=None,
                   help="look = the chained frame's dark costume pixels get the run look's colour cast (stops drift)")
    n.add_argument("--chain-sharpen", type=float, default=None,
                   help="with --look-chain prev: an unsharp mask of the chained frame (0 = off, 1 = +100%% detail)")
    n.add_argument("--look-chain", choices=("run", "prev", "anchor"), default=None,
                   help="layout clips after the first: the frame -1 look image is the run's look (run) or the previous "
                        "clip's frame at the clip's start (prev)")
    n.add_argument("--adain", type=float, default=None, help="layout clips 2..n: AdaIN toward the look image's latent "
                   "statistics after each stage (0..1; 0 = off)")
    n.add_argument("--clip-frames", type=int, default=None, help="the longest clip, 8k+1 frames (49..257; default 121): "
                   "longer clips = fewer joins, more GPU memory")
    n.add_argument("--proxy", choices=("auto", "off"), default=None,
                   help="the character's proxy kit (clay mask, hair...) on the mannequin: auto (default, when the "
                        "character has one) or off")
    n.add_argument("--sound-text", default=None, help="layout: the sound in words for LTX's own audio (default: a "
                   "quiet ambience sentence; \"\" = none). Without one the Layout LoRA makes silence")
    n.add_argument("--look-scene", default=None, help="layout without a panorama: the place in words; a second Qwen edit "
                   "puts it into the look image (the grey Blender room -> the scene) and the keyframe looks are off")
    n.add_argument("--look-text", default=None, help="layout: the character's words in the look instruction instead of "
                   "its saved description (when the description and the sheet disagree)")
    n.add_argument("--video-model", choices=("distilled", "full"), default=None,
                   help="layout runs: distilled (default; LTX-2.5 distilled, 8 steps, CFG 1) or full (the dev model: "
                        "30 steps with guidance and the negative; about 8x slower in preview, 5x at final (estimate); "
                        "not yet A/B-tested; needs the dev transformer: models download --feature video); "
                        "--steps/--guidance apply to full")
    n.add_argument("--profile", choices=("ltx25dev", "ltx25"), default=None,
                   help="the video model: ltx25 = LTX-2.5 distilled (8 steps; the default of layout runs, a Blender "
                        "shot with a library character), ltx25dev = LTX-2.5 full (the default of video-to-video runs)")
    n.add_argument("--steps", type=int, default=None, help="full model: sampling steps of stage 1 (8..60; default 30)")
    n.add_argument("--guidance", type=float, default=None, help="full model: the video CFG (1..10; default 3)")
    n.add_argument("--note", default=None, help="what this run tests (shown with its prompt in the app)")
    n.add_argument("--control-video", default=None, help="a control video to use instead of the skeleton drawn from "
                   "the rig (e.g. a pose detector's skeletons of several characters); fitted to the clip size")
    n.add_argument("--quality", choices=sorted(runner.QUALITY), default=None,
                   help="preview = 768x448; final = 1536x896 (two-stage upscale)")
    n.add_argument("--seed", type=int, default=None)
    n.add_argument("--control-strength", type=float, default=None,
                   help="strength of the skeleton guide (default 1; 0 = no skeleton guide, the render alone)")
    n.add_argument("--no-follow-render", action="store_true",
                   help="do not start from the Blender render (control video only; not recommended)")
    n.add_argument("--source-denoise", type=float, default=None,
                   help="video-to-video: start from the Blender render at this noise level (0 < s <= 1; default 0.979, "
                        "0.985 for dance and video templates); "
                        "about 1 - s of the render is kept, e.g. 0.909 keeps about 9%%")
    n.add_argument("--character-collection", default=None, help="Blender collection of the placeholder (default Mannequin)")
    n.add_argument("--rig", default=None, help="armature of the placeholder (default MannequinRig)")
    n.add_argument("--allow-external", action="store_true",
                   help="accept a .blend that uses files which cannot be packed (linked libraries, caches, ...)")
    fk = rp.add_parser("fork", help="a new run that keeps the first clips of a run and makes the rest again with "
                                    "changes (queued and started)")
    fk.add_argument("run_id")
    fk.add_argument("--from-clip", type=int, required=True, help="the first clip to make again (2 or more)")
    fk.add_argument("--video", default=None, help="new shot text (the shared context) for the clips made again")
    fk.add_argument("--style", default=None, help="new style text for the clips made again")
    fk.add_argument("--describe-facing", action="store_true", default=None,
                    help="add where the character faces (from Blender) to each new clip's prompt")
    fk.add_argument("--look-chain", choices=("run", "prev", "anchor"), default=None)
    fk.add_argument("--chain-colour", choices=("off", "look", "match"), default=None)
    fk.add_argument("--chain-light", choices=("off", "look", "tone"), default=None)
    fk.add_argument("--chain-sharpen-mode", choices=("fixed", "match"), default=None)
    fk.add_argument("--chain-sharpen-target", type=float, default=None)
    fk.add_argument("--adain", type=float, default=None)
    fk.add_argument("--adain-ref", choices=("look", "clip1"), default=None)
    fk.add_argument("--keyframe", choices=("on", "off"), default=None)
    fk.add_argument("--relay-strength", type=float, default=None)
    fk.add_argument("--clip-text", choices=("manual", "auto"), default=None,
                    help="auto = the clips made again get texts written from Blender")
    fk.add_argument("--clip-video", nargs=2, action="append", metavar=("K", "TEXT"), default=None,
                    help="new ACTIONS for clip K (a clip the fork makes again; '' = none); --video changes the shared "
                         "text, the other clips keep their own actions")
    fk.add_argument("--prop", action="append", default=None, metavar="PATTERN=WORDS",
                    help="new words for a prop group in the clips made again (repeatable; empty words = the "
                         "template's words)")
    fk.add_argument("--prop-image", action="append", default=None, metavar="PATTERN=PICTURE",
                    help="a new prop picture for the clips made again (repeatable; an empty PICTURE removes the "
                         "parent's): the fork's sheet is drawn again")
    fk.add_argument("--note", default=None, help="what this fork tests (shown in the app)")
    s = rp.add_parser("start", help="run (or continue) a run")
    s.add_argument("run_id")
    s.add_argument("--detach", action="store_true", help="run in the background; return when the worker has started")
    s.add_argument("--launch", type=int, default=None, help=argparse.SUPPRESS)   # set by --detach
    s.add_argument("--new-attempt", action="store_true",
                   help="after an 'unresolved' stop: give up the old job intent and submit again")
    s.add_argument("--through-clip", type=int, default=None,
                   help="layout runs: stop ('paused') after clip K, so it can be judged and made again (run retake) "
                        "before the next clip continues from it; 0 = no limit (finish the run)")
    rt = rp.add_parser("retake", help="layout runs: make clip K again with changes (a new take beside the original; "
                       "it starts from clip K-1's saved latents and ends on clip K+1's); see run use-take")
    rt.add_argument("run_id")
    rt.add_argument("--clip", type=int, required=True)
    rt.add_argument("--seed", type=int, default=None)
    rt.add_argument("--video", default=None,
                    help="another text for this clip: ONLY the actions of THIS clip (what the layout shows in its "
                         "frames). A run with one prompt per clip keeps its shared text and clip context and puts this "
                         "in place of the clip's actions; an older run takes it as the clip's whole scene and action "
                         "(the run adds the style, 'The character from the reference sheet...' and 'One continuous "
                         "shot, no cuts.')")
    rt.add_argument("--style", default=None)
    rt.add_argument("--character", default=None,
                    help="the Reference-sheet part of the clip text (the costume in exact words), for this take only")
    rt.add_argument("--negative", default=None)
    rt.add_argument("--ingredients-strength", type=float, default=None)
    rt.add_argument("--style-lora", default=None)
    rt.add_argument("--style-lora-strength", type=float, default=None)
    rt.add_argument("--relay-strength", type=float, default=None, help="the start latents' guide strength (0..1]")
    rt.add_argument("--adain", type=float, default=None)
    rt.add_argument("--keyframe", choices=("on", "off"), default=None, help="the clip's keyframe look (default on)")
    rt.add_argument("--keyframe-image", default=None,
                    help="this take's own keyframe look (an image of the scene with the character), for example the "
                         "sheet's back panel pasted into the Blender frame; added to the run's keyframe (--keyframe off removes that)")
    rt.add_argument("--keyframe-frame", type=int, default=None, help="the clip frame of --keyframe-image (default middle)")
    rt.add_argument("--keyframe-at", action="append", default=None, metavar="FILE@FRAME",
                    help="one more keyframe look at a clip frame (repeat for more), for example a sheet panel pasted "
                         "into the Blender frame: --keyframe-at back.png@8 --keyframe-at front.png@116")
    rt.add_argument("--end-anchor", choices=("on", "off"), default=None,
                    help="a middle clip: end on clip K+1's start latents (default on)")
    rt.add_argument("--anchor-strength", type=float, default=None, help="the end anchor's strength (0..1]")
    rt.add_argument("--refine-sigmas", default=None)
    rt.add_argument("--pose-strength", type=float, default=None,
                    help="layout: + the rig's skeleton as a stage-1 guide (0 = off; the head direction in back views)")
    rt.add_argument("--pose-body", choices=("standard", "chibi"), default=None, help="the skeleton drawing (standard)")
    rt.add_argument("--pose-stage2", choices=("on", "off"), default=None, help="the pose guide in stage 2 too")
    rt.add_argument("--look-chain", choices=("run", "prev", "anchor"), default=None,
                    help="the frame -1 look image: the run's look, or the previous clip's frame at this clip's start")
    rt.add_argument("--chain-sharpen", type=float, default=None,
                    help="with --look-chain prev: an unsharp mask of that frame (0 = off, 1 = +100%% detail)")
    rt.add_argument("--look-image", default=None,
                    help="this take's own frame -1 look image (for example a fresh look of the clip's start frame)")
    rt.add_argument("--adain-ref", choices=("look", "clip1"), default=None)
    rt.add_argument("--chain-light", choices=("off", "look", "tone"), default=None)
    rt.add_argument("--chain-sharpen-mode", choices=("fixed", "match"), default=None)
    rt.add_argument("--chain-sharpen-target", type=float, default=None)
    rt.add_argument("--chain-colour", choices=("off", "look", "match"), default=None,
                    help="look = the chained frame's dark costume pixels get the run look's colour cast (stops drift)")
    rt.add_argument("--note", default=None)
    rt.add_argument("--no-start", action="store_true", help="only record the take; start the run later")
    rp.add_parser("takes", help="the takes of a run (per clip)").add_argument("run_id")
    ut = rp.add_parser("use-take", help="put a take into the video (the original stays as take 0; the previous final "
                       "video is kept) and join the clips again")
    ut.add_argument("run_id")
    ut.add_argument("--clip", type=int, required=True)
    ut.add_argument("--take", type=int, required=True)
    sc = rp.add_parser("score", help="layout runs: the gauntlet numbers of ONE clip (or one take): motion follow vs the "
                       "rig (DWPose, ComfyUI's Python, CPU, ~1-2 min), sharpness, saturation, flicker and the join jump; "
                       "stored with the clip or the take")
    sc.add_argument("run_id")
    sc.add_argument("--clip", type=int, required=True)
    sc.add_argument("--take", type=int, default=None)
    sc.add_argument("--no-motion", action="store_true", help="only the picture numbers (fast, no DWPose)")
    sc.add_argument("--critic", default=None, metavar="DIR",
                    help="also write the blind critic's material (strips, frames, rig angles) into DIR; "
                         "see docs/gauntlet_critic.md")
    sc.add_argument("--tag", default=None, help="the anonymous name of this clip in DIR (default c<K>[t<N>])")
    for name in ("status", "cancel"):
        rp.add_parser(name).add_argument("run_id")
    rp.add_parser("delete", help="delete a run and its video for good").add_argument("run_id")
    rp.add_parser("list").add_argument("--limit", type=int, default=20)
    e = rp.add_parser("events", help="the events of a run (JSON lines), then a result line with 'next'")
    e.add_argument("run_id")
    e.add_argument("--since", type=int, default=0, help="skip the first N events (use 'next' of the last call)")
    e.add_argument("--follow", action="store_true", help="wait for new events until the run ends")
    return ap


def _song(a) -> int:
    from . import song
    if a.action == "make":
        r = song.make(a.name, a.tags, a.lyrics, a.bpm, a.seconds, a.key, a.seed, a.language, on_event=emit)
        emit({"ok": True, "song": r})
    elif a.action == "list":
        emit({"ok": True, "songs": song.list_songs()})
    elif a.action == "show":
        emit({"ok": True, "song": song.load(a.name)})
    elif a.action == "beats":
        p = Path(a.name)
        if p.is_file():
            bpm = a.bpm or 120.0
        else:
            s = song.load(a.name)
            p, bpm = Path(s["file"]), a.bpm or s.get("bpm") or 120.0
        b = song.beats(p, bpm)
        emit({"ok": True, **b, "beat_frames": song.frames(b["beats"], a.start, a.fps),
              "downbeat_frames": song.frames(b["downbeats"], a.start, a.fps)})
    elif a.action == "loop":
        emit({"ok": True, **song.loop(a.name, a.beats, a.frames, a.fps, a.start)})
    else:
        emit({"ok": True, **song.place(a.blend, a.name, a.start, a.out, sound_file=a.file)})
    return 0


def _shot(a) -> int:
    from . import shots
    if a.action == "check":
        from . import shot_check
        emit(shot_check.check(a.name, fresh=a.fresh, with_facts=a.json))
        return 0
    if a.action == "timeline":
        emit({"ok": True, **shots.timeline(a.name)})
        return 0
    if a.action == "clip-texts":
        emit({"ok": True, **shots.clip_texts(a.name, {k: t for k, t in a.clip}, a.context)})
        return 0
    if a.action == "layers":
        props = {}
        for item in a.prop:
            if "=" not in item:
                raise shots.ShotError(f"--prop needs NAME=TEXT, not {item!r}")
            k, v = item.split("=", 1)
            props[k.strip()] = v
        emit({"ok": True, **shots.layers(a.name, a.set_description, props, a.rescan)})
        return 0
    if a.action == "make-dance":
        if a.wait:
            shots.prepare_dance(a.name, a.song, a.routine, a.loop, a.seconds, a.start, a.night)
            r = shots.make_dance(a.name, on_event=emit)
        else:
            r = shots.start_dance(a.name, a.song, a.routine, a.loop, a.seconds, a.start, a.night)
        emit({"ok": True, "shot": r})
    elif a.action == "from-video":
        from . import video_template
        if a.wait:
            video_template.prepare(a.name, a.video, a.start, a.seconds, a.sound, a.description, people=a.people,
                                   tracks=a.tracks)
            r = video_template.make(a.name, on_event=emit)
        else:
            r = video_template.start(a.name, a.video, a.start, a.seconds, a.sound, a.description, people=a.people,
                                     tracks=a.tracks)
        emit({"ok": True, "shot": r})
    elif a.action == "pick-people":
        from . import video_template
        r = video_template.pick(a.name, a.tracks, wait=a.wait, on_event=emit)
        emit({"ok": True, "shot": {k: r.get(k) for k in ("name", "status", "pid", "tracks", "people", "error", "files",
                                                          "warning") if k in r}})
    elif a.action == "work-mocap":
        from . import video_template
        r = video_template.make(a.name, on_event=emit)
        emit({"ok": r.get("status") in ("done", "pick"),
              "shot": {k: r.get(k) for k in ("name", "status", "error", "files", "people")}})
    elif a.action == "mask":
        from . import figure_mask
        r = figure_mask.track(a.name, json.loads(a.clicks) if a.clicks else None, on_event=emit)
        emit({"ok": True, "mask": {k: r[k] for k in ("template", "frames", "flags", "files", "seconds")}})
    elif a.action == "work-mask":
        from . import figure_mask
        r = figure_mask.work(a.name, on_event=emit)
        emit({"ok": True, "mask": {k: r.get(k) for k in ("template", "frames", "flags", "files")}})
    elif a.action == "add-video":
        emit({"ok": True, "shot": shots.add_video(a.name, a.video, a.description)})
    elif a.action == "add":
        emit({"ok": True, "shot": shots.add(a.name, a.blend, a.description, a.move, not a.no_preview, on_event=emit)})
    elif a.action == "list":
        emit({"ok": True, "shots": shots.list_shots()})
    elif a.action == "show":
        emit({"ok": True, "shot": shots.show(a.name)})
    elif a.action == "delete":
        emit({"ok": True, **shots.delete(a.name)})
    else:
        r = shots.make_dance(a.name, on_event=emit)
        emit({"ok": r.get("status") == "done", "shot": {k: r.get(k) for k in ("name", "status", "error", "files")}})
    return 0



def _background(a) -> int:
    """`background list|new|depth|style|delete`. A worker started by the app (`--ref bg-...`) leaves a job record when
    it fails, so the Backgrounds tab can show the error (cinestudio.background.record_job)."""
    import secrets as _s
    from . import background as bgs, lab, workqueue
    if a.action == "list":
        emit({"ok": True, "backgrounds": bgs.list_backgrounds()})
        return 0
    if a.action == "delete":
        emit({"ok": True, **bgs.delete(bgs.load(a.name)["name"])})
        return 0
    if a.action == "import":
        emit({"ok": True, **bgs.import_plate(a.file, a.name, a.scene)})
        return 0
    ref = a.ref or f"bg{a.action}_{_s.token_hex(3)}"
    if a.action == "new":
        label = f"new panorama: {(a.scene or '')[:50]}"
    elif a.action == "plate":
        label = f"new backplate: {(a.scene or '')[:50]}"
    elif a.action == "from-set":
        label = f"backplate over the set: {(a.scene or Path(a.shot).parent.name)[:44]}"
    elif a.action == "depth":
        label = f"panorama {a.name}: 360 depth (MoGe-2)"
    else:
        label = f"panorama {a.name}: style pass with {a.style_lora}"
    try:
        with lab.turn(ref, label):
            if a.action == "depth":
                r = {"name": a.name, "depth": str(bgs.depth(a.name, on_event=emit))}
            elif a.action == "plate":
                r = bgs.make_plate(a.scene, a.shot, bgs.style_words(a.style, a.style_preset), a.seed, a.name,
                                   on_event=emit)
                r = bgs.stamp_style(r["name"], a.style_preset)
            elif a.action == "from-set":
                r = bgs.make_set_plate(a.scene, a.shot, bgs.style_words(a.style, a.style_preset), a.seed, a.name,
                                       on_event=emit)
                r = bgs.stamp_style(r["name"], a.style_preset)
            elif a.action == "new":
                r = bgs.make(a.scene, bgs.style_words(a.style, a.style_preset), a.seed, a.name, on_event=emit)
                r = {**r, **bgs.stamp_style(r["name"], a.style_preset)}
                if not a.no_depth:
                    if lab.canceled(ref):                  # "Stop" in the queue: the panorama is kept, no depth
                        raise workqueue.QueueCanceled(f"{ref} was stopped after the panorama (it is saved)")
                    r["depth"] = str(bgs.depth(r["name"], on_event=emit))
            else:
                r = bgs.style_pass(a.name, a.style_lora, a.keep, a.strength, a.seed, a.style, a.new_name, on_event=emit,
                                   profile=a.profile)
    except workqueue.QueueCanceled:
        raise
    except Exception as exc:  # noqa: BLE001 - record it for the app, then the normal JSON error
        if a.ref:
            try:
                bgs.record_job(a.ref, "failed", label, f"{type(exc).__name__}: {exc}")
            except Exception:  # noqa: BLE001 - the record is a convenience; the log keeps the error
                pass
        raise
    emit({"ok": True, **r})
    return 0

def main(argv: list[str] | None = None) -> int:
    try:
        a = _parser().parse_args(argv)
    except CliInputError as exc:
        emit({"ok": False, "error": str(exc), "error_type": "UsageError"})
        return 2
    try:
        return _dispatch(a)
    except (runner.RunError, CliInputError, character.CharacterError) as exc:
        emit({"ok": False, "error": str(exc), "error_type": type(exc).__name__})
        return 2
    except Exception as exc:  # noqa: BLE001 - agents get JSON for every failure, never a traceback
        emit({"ok": False, "error": str(exc) or repr(exc), "error_type": type(exc).__name__})
        return 1


def _quality_args(p, denoise: bool = False, angles: bool = False) -> None:
    """Values that change the quality of an image step (default: the mode's values)."""
    p.add_argument("--steps", type=int, default=None, help="sampling steps (quality 40, fast 4)")
    p.add_argument("--cfg", type=float, default=None, help="how strictly the text is followed (quality 3, fast 1)")
    p.add_argument("--sampler", default=None, help="ComfyUI sampler (default euler)")
    p.add_argument("--scheduler", default=None, help="ComfyUI scheduler (default simple)")
    p.add_argument("--shift", type=float, default=None, help="shape against detail (default 3.1)")
    p.add_argument("--negative", default=None, help="what must not be drawn (needs cfg above 1)")
    if denoise:
        p.add_argument("--denoise", type=float, default=None, help="below 1 keeps more of the image (default 1)")
    if angles:
        p.add_argument("--angles-strength", type=float, default=None, help="strength of the camera-angle LoRA (default 0.9)")


def _settings_of(a) -> dict:
    return {k: getattr(a, k) for k in ("steps", "cfg", "sampler", "scheduler", "shift", "negative", "denoise", "angles_strength")
            if getattr(a, k, None) is not None}


def _job(action: str, name: str, **kwargs):
    """Run a character step as a job in this process: the same job record as in the app (so the app
    shows its progress), events printed as JSON lines. Raises CharacterError when the job failed."""
    from . import jobs
    j = jobs.run_now(action, name, echo=lambda ev: emit(ev) if ev.get("event") != "progress" or
                     int(ev.get("value") or 0) % 10 == 0 else None, **kwargs)
    if j["status"] != "done":
        raise character.CharacterError(f"{action} failed ({j['id']}): {j.get('error')}")
    return j["result"]


def _run_prop_options(a) -> dict:
    """run new: the prop words and pictures (--prop, --prop-image) with the library props of --prop-ref filled in."""
    from . import props
    words = prop_args(a.prop) if a.prop else {}
    pics = prop_image_args(a.prop_image) if a.prop_image else {}
    refs = getattr(a, "prop_ref", None)
    if refs:
        words, pics = props.merge_refs(props.parse_ref_args(refs), words, pics)
    return {**({"props": words} if words else {}), **({"prop_images": pics} if pics else {})}


def _prop_parser(sub) -> None:
    """`prop ...`: the prop library (cinestudio/props.py; AGENTS.md 4.10)."""
    from . import props
    pp = sub.add_parser("prop", help="the prop library: objects for the template's Props layer (a picture on black + "
                                     "words), used by runs with --prop-ref").add_subparsers(
        dest="action", required=True, parser_class=JsonArgumentParser)
    pp.add_parser("list", help="every prop (ID, display name, words, style, status)")
    pp.add_parser("show", help="one prop with its pictures and history").add_argument("name")
    x = pp.add_parser("create", help="prop picture candidates from text (a GPU job: Qwen-Image or Z-Image, the prop "
                                     "alone on black, flatly lit, in a library style); then: prop pick")
    x.add_argument("name", help="the ID (lowercase letters, digits, - and _)")
    x.add_argument("--words", required=True, help="the prop in a few concrete words (every video prompt uses them), "
                   "e.g. 'matte deep violet rubber dodgeball with a thin magenta seam'")
    x.add_argument("--description", default="", help="more detail for the picture prompt (default: the words)")
    x.add_argument("--style", default=None, help="a library style (default: the library's default style)")
    x.add_argument("--candidates", type=int, default=4)
    x.add_argument("--seed", type=int, default=202, help="first seed; candidates use seed, seed+1, ...")
    x.add_argument("--model", choices=sorted(props.MODELS), default=props.DEFAULT_MODEL)
    x.add_argument("--negative", default="", help="more things the picture must not show")
    x.add_argument("--display-name", default="", help="the name the app shows (any text; NAME stays the ID)")
    x = pp.add_parser("import", help="a prop from your own picture(s) (instant)")
    x.add_argument("name")
    x.add_argument("--image", required=True, help="the front: the prop alone on black, in the characters' style")
    x.add_argument("--words", required=True, help="the prop in a few concrete words")
    x.add_argument("--description", default="")
    x.add_argument("--style", default=None, help="the library style the picture is drawn in (its stamp)")
    x.add_argument("--side", default=None, help="an optional second view")
    x.add_argument("--candidate", action="append", default=None, help="more pictures to choose from (repeatable)")
    x.add_argument("--display-name", default="")
    x = pp.add_parser("pick", help="use one candidate as the prop's front (the old front stays undoable)")
    x.add_argument("name")
    x.add_argument("--seed", type=int, required=True)
    x = pp.add_parser("edit", help="change one picture by an instruction (Qwen-Image-Edit; a GPU job; undoable)")
    x.add_argument("name")
    x.add_argument("--target", choices=props.TARGETS, default="front")
    x.add_argument("--region", default=None, help="x0,y0,x1,y1: edit only this box (crop and stitch)")
    _quality_args(x, denoise=True)
    x.add_argument("--instruction", required=True, help="ONE change; end with 'Keep everything else exactly the same.'")
    x.add_argument("--reference", default=None, help="optional Picture 2 (one detail, tightly cropped)")
    x.add_argument("--mode", choices=["quality", "fast"], default="quality")
    x.add_argument("--seed", type=int, default=None)
    x = pp.add_parser("undo", help="undo the last change (edit or pick) of one picture")
    x.add_argument("name")
    x.add_argument("--target", choices=props.TARGETS, default="front")
    x = pp.add_parser("words", help="the prop's words (video prompts) and/or its picture description (instant)")
    x.add_argument("name")
    x.add_argument("--words", default=None)
    x.add_argument("--description", default=None)
    x = pp.add_parser("skin", help="a new SKIN of the prop in a library style (a GPU job: the front redrawn by Qwen)")
    x.add_argument("name")
    x.add_argument("style")
    x.add_argument("--seed", type=int, default=None)
    x = pp.add_parser("use-style", help="name the library style the prop is drawn in (instant; pictures stay)")
    x.add_argument("name")
    x.add_argument("style")
    x = pp.add_parser("rename", help="give a prop a new ID (its folder moves; its skins follow)")
    x.add_argument("name")
    x.add_argument("new_name")
    x = pp.add_parser("display-name", help="the name the app shows (any text; the ID stays)")
    x.add_argument("name")
    x.add_argument("text")
    x = pp.add_parser("delete", help="delete a prop for good (runs keep their own copy of its picture)")
    x.add_argument("name")


def _prop(a) -> int:
    from . import props
    if a.action == "list":
        emit({"ok": True, "props": props.list_props()})
    elif a.action == "show":
        emit({"ok": True, "prop": props.load(a.name)})
    elif a.action == "create":
        p = _job("prop_create", a.name, words=a.words, description=a.description, style=a.style,
                 candidates=a.candidates, seed=a.seed, model=a.model, negative=a.negative, display_name=a.display_name)
        emit({"ok": True, "prop": p, "next": f"look at {props.prop_dir(a.name) / 'candidates' / 'candidates.jpg'}, then: "
                                            f"prop pick {a.name} --seed N"})
    elif a.action == "import":
        emit({"ok": True, "prop": props.import_picture(a.name, a.image, a.words, a.description, a.style, side=a.side,
                                                       candidates=a.candidate, display_name=a.display_name)})
    elif a.action == "pick":
        emit({"ok": True, "prop": props.pick(a.name, a.seed)})
    elif a.action == "edit":
        p = _job("prop_edit", a.name, instruction=a.instruction, reference=a.reference, mode=a.mode, seed=a.seed,
                 target=a.target, settings=_settings_of(a), region=a.region)
        emit({"ok": True, "prop": p, "next": f"look at {a.target}.png; undo with: prop undo {a.name} --target {a.target}"})
    elif a.action == "undo":
        emit({"ok": True, "prop": props.undo(a.name, a.target)})
    elif a.action == "words":
        if a.words is None and a.description is None:
            raise CliInputError("give --words and/or --description")
        emit({"ok": True, "prop": props.set_words(a.name, a.words, a.description)})
    elif a.action == "skin":
        p = _job("prop_skin", a.name, style=a.style, seed=a.seed)
        emit({"ok": True, "prop": p, "next": f"look at the new skin's front ({p.get('skin', '?')})"})
    elif a.action == "use-style":
        emit({"ok": True, "prop": props.set_style(a.name, a.style)})
    elif a.action == "rename":
        emit({"ok": True, "prop": props.rename(a.name, a.new_name)})
    elif a.action == "display-name":
        emit({"ok": True, "prop": props.set_display_name(a.name, a.text)})
    elif a.action == "delete":
        emit({"ok": True, **props.delete(a.name)})
    else:
        raise CliInputError(f"unknown prop command {a.action}")
    return 0


def _character(a) -> int:
    if a.action == "list":
        emit({"ok": True, "characters": character.list_characters()})
        return 0
    if a.action == "show":
        emit({"ok": True, "character": character.load(a.name)})
        return 0
    if a.action == "delete":
        emit({"ok": True, **character.delete_character(a.name)})
        return 0
    if a.action == "rename":
        emit({"ok": True, "character": character.rename(a.name, a.new_name)})
        return 0
    if a.action == "display-name":
        emit({"ok": True, "character": character.set_display_name(a.name, a.text)})
        return 0
    if a.action == "use-style":
        emit({"ok": True, "character": character.set_style(a.name, a.style)})
        return 0
    if a.action == "skins":
        emit({"ok": True, "skins": character.skins(a.name)})
        return 0
    if a.action == "link-skin":
        emit({"ok": True, "character": character.link_skin(a.name, a.to)})
        return 0
    if a.action == "unlink-skin":
        emit({"ok": True, "character": character.unlink_skin(a.name)})
        return 0
    if a.action == "skin":
        c = _job("new_skin", a.name, style=a.style, seed=a.seed)
        emit({"ok": True, "character": c, "next": f"look at the new skin's sheet; then: character finish {c.get('skin', '?')}"})
        return 0
    if a.action == "create":
        c = _job("create", a.name, description=a.text, style=a.style, candidates=a.candidates, seed=a.seed,
                 width=a.width, height=a.height, style_lora=a.style_lora, body=a.body, display_name=a.display_name)
        emit({"ok": True, "character": c, "next": f"look at {character.char_dir(a.name) / 'candidates' / 'candidates.jpg'}, "
                                                 f"then: character pick {a.name} --seed N"})
        return 0
    if a.action == "pick":
        emit({"ok": True, "character": character.pick(a.name, a.seed), "next": f"character views {a.name}"})
        return 0
    if a.action == "import":
        if a.image:
            slots = {"front": a.image, **{s: getattr(a, s) for s in ("right", "back", "left", "face") if getattr(a, s)}}
            c = character.import_slots(a.name, slots, a.text, a.style, a.body, display_name=a.display_name)
        else:
            c = character.import_sheet(a.name, a.sheet, a.text, a.style, display_name=a.display_name)
        emit({"ok": True, "character": c, "next": f"character views {a.name}" if a.image else "ready"})
        return 0
    if a.action == "edit":
        c = _job("edit", a.name, instruction=a.instruction, reference=a.reference, mode=a.mode, seed=a.seed,
                 target=a.target, settings=_settings_of(a), region=a.region)
        nxt = f"look at front.png; then: character views {a.name}" if a.target == "front" \
            else f"look at the {a.target} view and the new sheet; undo with: character undo {a.name} --target {a.target}"
        emit({"ok": True, "character": c, "next": nxt})
        return 0
    if a.action == "undo":
        emit({"ok": True, "character": _job("undo", a.name, target=a.target)})
        return 0
    if a.action == "redo-view":
        c = _job("redo_view", a.name, view=a.view, seed=a.seed, mode=a.mode, settings=_settings_of(a))
        emit({"ok": True, "character": c, "next": f"look at the {a.view} view; undo with: character undo {a.name} --target {a.view}"})
        return 0
    if a.action == "repose":
        c = _job("repose", a.name, shot=a.shot, seed=a.seed)
        emit({"ok": True, "character": c, "next": f"look at the sheet; then: character finish {a.name}"})
        return 0
    if a.action == "views":
        c = _job("views", a.name, mode=a.mode, seed=a.seed, distance=f"{a.distance} shot", face=a.face)
        emit({"ok": True, "character": c, "next": f"look at the sheet; then: character style {a.name} --style-lora L, "
                                                 f"or character finish {a.name}"})
        return 0
    if a.action == "style":
        emit({"ok": True, "character": _job("style", a.name, style_lora=a.style_lora, keep=a.keep)})
        return 0
    if a.action == "mannequin":
        given = {k: getattr(a, k) for k in character.MANNEQUIN_KEYS if getattr(a, k) is not None}
        ch = character.set_mannequin(a.name, given, default=a.default)
        emit({"ok": True, "name": a.name, "mannequin": ch.get("mannequin"), "missing": character.video_missing(ch),
              "next": f"the sheet follows the proportions only after: character repose {a.name}"})
        return 0
    if a.action == "proxy":
        ch = character.set_proxy(a.name, json.loads(a.spec))
        emit({"ok": True, "name": a.name, "proxy": ch["proxy"], "missing": character.video_missing(ch)})
        return 0
    if a.action == "finish":
        emit({"ok": True, "character": character.finish(a.name)})
        return 0
    raise CliInputError(f"unknown character command {a.action}")


def _dispatch(a) -> int:
    if a.cmd == "app":
        from . import webapp
        webapp.serve(a.port, a.open)
        return 0
    if a.cmd == "mcp":
        from . import mcp_server
        mcp_server.serve()
        return 0
    if a.cmd == "doctor":
        r = doctor()
        emit(r)
        return 0 if r["ok"] else 1
    if a.cmd == "loras":
        emit({"ok": True, "loras": loras()})
        return 0
    if a.cmd == "setup":
        return _setup(a)
    if a.cmd == "settings":
        return _settings(a)
    if a.cmd == "background":
        return _background(a)
    if a.cmd == "models" and a.action == "civitai":
        from . import civitai, models as _m
        emit({"ok": True, **civitai.download(a.version_id, _m.models_dir() / a.dest)})
        return 0
    if a.cmd == "models":
        return _models(a)
    if a.cmd == "proxy":
        r = proxy(a.action)
        emit(r)
        return 0 if r["ok"] else 1
    if a.cmd == "character":
        return _character(a)
    if a.cmd == "prop":
        return _prop(a)
    if a.cmd == "comfy":
        from . import comfy
        emit({"ok": True, **{"detect": comfy.detect, "start": comfy.start, "stop": comfy.stop, "status": comfy.status}[a.action]()})
        return 0
    if a.cmd == "gpu":
        from . import gpu_focus
        r = {"list": gpu_focus.list_apps, "all-integrated": gpu_focus.on, "reset": gpu_focus.off}.get(a.action)
        res = r() if r else gpu_focus.set_app(a.exe, a.choice)
        emit({"ok": True, **res} if "ok" not in res else res)
        return 0 if res.get("ok", True) else 1
    if a.cmd == "queue":
        from . import workqueue
        if a.action == "list":
            emit({"ok": True, **workqueue.view()})
        else:
            emit({"ok": True, **workqueue.cancel(a.kind, a.ref)})
        return 0
    if a.cmd == "workflow":
        from . import user_workflows as uw
        if a.action == "list":
            emit({"ok": True, "copies": uw.list_copies(a.name), "active": uw.active_choice(a.name)})
        elif a.action == "copy":
            emit({"ok": True, "copy": uw.make_copy(a.name, a.label)})
        elif a.action == "set":
            try:
                value = json.loads(a.value)
            except ValueError:
                value = a.value                                # plain text
            emit({"ok": True, **uw.save_values(a.name, a.copy, {a.node: {a.input: value}})})
        elif a.action == "reset":
            emit({"ok": True, **uw.reset_values(a.name, a.copy, a.node)})
        else:
            emit({"ok": True, **uw.set_choice(a.name, None if a.copy == "builtin" else a.copy)})
        return 0
    if a.cmd == "style":
        from . import styles
        if a.action == "list":
            used = styles.used_by()
            emit({"ok": True, "default": styles.default_style(),
                  "styles": [{**x, "used_by": used.get(x["name"], {})} for x in styles.list_styles()]})
        elif a.action == "show":
            from . import style_ref
            emit({"ok": True, "style": styles.load(a.name), "used_by": styles.used_by().get(a.name, {}),
                  "reference_candidates": style_ref.candidates(a.name), "reference_job": style_ref.job_status(a.name)})
        elif a.action == "make-reference":
            from . import style_ref
            emit({"ok": True, **style_ref.run_job(a.name, a.subject, a.seeds, a.ref, on_event=emit,
                                                  video_model=a.video_model)})
        elif a.action == "use-reference":
            from . import style_ref
            emit({"ok": True, **style_ref.use_reference(a.name, a.file, a.crop)})
        elif a.action in ("new", "edit"):
            data = {k: v for k, v in (("label", a.label), ("words", a.words), ("light", a.light),
                                      ("negative_image", a.negative_image), ("reference", a.reference)) if v is not None}
            if a.lora is not None:
                strength = a.lora_strength if a.lora_strength is not None else 0.4
                data["lora"] = {"name": a.lora, "strength": strength} if a.lora else None
            elif a.lora_strength is not None and a.action == "edit":
                cur = styles.load(a.name).get("lora")
                if cur:
                    data["lora"] = {**cur, "strength": a.lora_strength}
            emit({"ok": True, "style": styles.save(a.name, data, new=a.action == "new")})
        elif a.action == "delete":
            emit({"ok": True, **styles.delete(a.name)})
        else:
            emit({"ok": True, **styles.set_default(a.name)})
        return 0
    if a.cmd == "film":
        from . import film
        if a.action == "list":
            emit({"ok": True, "films": film.list_films()})
            return 0
        emit({"ok": True, **film.assemble(a.name, a.scene, score=a.score, score_file=a.score_file,
                                          score_start=a.score_start, music_db=a.music_db, scene_db=a.scene_db,
                                          duck=not a.no_duck, fade_out=a.fade_out, loudness=a.loudness,
                                          replace=a.replace)})
        return 0
    if a.cmd == "song":
        return _song(a)
    if a.cmd == "shot":
        return _shot(a)
    if a.cmd == "job":
        from . import jobs
        if a.action == "work":
            j = jobs.work(a.job_id)
            emit({"ok": j["status"] == "done", "job": {k: j.get(k) for k in ("id", "status", "error")}})
            return 0 if j["status"] == "done" else 1
        if a.action == "show":
            emit({"ok": True, "job": jobs.load(a.job_id)})
        else:
            emit({"ok": True, "jobs": jobs.list_jobs(a.character)})
        return 0
    if a.cmd == "sheet":
        from . import refsheet
        emit({"ok": True, **refsheet.prepare(a.image, a.out)})
        return 0
    if a.action == "new":
        if a.style_lora:
            check_style_lora(a.style_lora)
        names = list(a.character or [])
        if a.character2:
            names.append(a.character2)
        if a.character2 and len(names) > 2:
            raise CliInputError("give the characters as --character once per slot (--character2 is the old form of "
                                "the second)")
        a.character = names[0] if names else None
        if a.character:
            if a.sheet or a.character_text:
                raise CliInputError("give --character (library) OR --sheet with --character-text, not both")
            sheet, text = character.for_run(a.character)
        elif a.sheet and a.character_text:
            sheet, text = a.sheet, a.character_text
        else:
            raise CliInputError("give --character NAME, or --sheet IMAGE with --character-text TEXT")
        if bool(a.blend) == bool(a.template):
            raise CliInputError("give --blend FILE or --template NAME (one of them)")
        if a.blend and not a.character:                 # Blender shots: Layout-To-Render only (user, 2026-10-22)
            raise CliInputError("a Blender shot uses Layout-To-Render and needs a library character: import the sheet "
                                "first (character import NAME --sheet FILE), then --character NAME")
        from . import shots as _shots
        a.background, bg_default = _shots.run_background(a.blend, a.background, a.background_text, a.background_image,
                                                         a.look_scene)
        if bg_default:
            emit({"event": "note", "note": f"the template's default background: saved:{bg_default} "
                                           "(--background none = the template's own set)"})
        m = runner.new_run(a.blend, sheet, text, a.video, a.style, a.style_lora, a.restyle, a.first_frame,
                           a.allow_external, quality=a.quality, seed=a.seed, style_lora_strength=a.style_lora_strength,
                           character_collection=a.character_collection, rig=a.rig, character_name=a.character,
                           style_preset=a.style_preset,
                           characters=[{"name": nm, **({"collection": a.character2_collection, "rig": a.character2_rig}
                                                       if i == 1 and (a.character2_collection or a.character2_rig)
                                                       else {})} for i, nm in enumerate(names)]
                           if len(names) > 1 else None, spacing=a.spacing,
                           source_denoise=False if a.no_follow_render else a.source_denoise,
                           control_strength=a.control_strength, control_video=a.control_video,
                           describe_facing=a.describe_facing,
                           ingredients_strength=a.ingredients_strength, body_type=a.body_type, sound=a.sound,
                           song=a.song, song_start=a.song_start, template=a.template,
                           figure_mask=a.figure_mask, background=a.background,
                           background_text=a.background_text, background_image=a.background_image, background_seed=a.background_seed, **({'background_ground': a.background_ground} if a.background_ground else {}),
                           negative=a.negative,
                           steps=a.steps, guidance=a.guidance, look_extra=a.look_extra, look_text=a.look_text, look_scene=a.look_scene, sound_text=a.sound_text, **({'proxy': a.proxy} if a.proxy else {}), clip_frames=a.clip_frames, adain=a.adain, relay=a.relay, relay_strength=a.relay_strength, **({'audio_relay': a.audio_relay} if a.audio_relay else {}), **({'target_text': a.target_text} if a.target_text else {}), **({'look_chain': a.look_chain} if a.look_chain else {}), **({'chain_sharpen': a.chain_sharpen} if a.chain_sharpen is not None else {}), **({'chain_colour': a.chain_colour} if a.chain_colour else {}), **({'chain_light': a.chain_light} if a.chain_light else {}), **({'chain_sharpen_mode': a.chain_sharpen_mode} if a.chain_sharpen_mode else {}), **({'chain_sharpen_target': a.chain_sharpen_target} if a.chain_sharpen_target is not None else {}), **({'adain_ref': a.adain_ref} if a.adain_ref else {}), **({'clip_text': a.clip_text} if a.clip_text else {}), **({'pose_strength': a.pose_strength} if a.pose_strength is not None else {}), **({'look_keyframes': a.look_keyframes} if a.look_keyframes else {}), **({'pose_body': a.pose_body} if a.pose_body else {}), **({'clip_videos': {k: t for k, t in a.clip_video}} if a.clip_video else {}), **({'clip_context': a.clip_context} if a.clip_context is not None else {}), **({'character_words': a.character_words} if a.character_words else {}),
                           video_model=a.video_model, profile=a.profile, **_run_prop_options(a))
        if a.note:
            runner.set_note(m["id"], a.note)
        emit({"ok": True, "id": m["id"], "prompt_warnings": m.get("prompt_warnings") or [], "manifest": m})
        return 0
    if a.action == "fork":
        m = runner.fork_run(a.run_id, a.from_clip, a.note, video=a.video, style=a.style, describe_facing=a.describe_facing,
                            look_chain=a.look_chain, chain_colour=a.chain_colour, adain=a.adain, adain_ref=a.adain_ref,
                            keyframe=a.keyframe, relay_strength=a.relay_strength, clip_text=a.clip_text,
                            chain_light=a.chain_light, chain_sharpen_mode=a.chain_sharpen_mode,
                            chain_sharpen_target=a.chain_sharpen_target,
                            props=prop_args(a.prop, allow_empty=True) if a.prop else None,
                            prop_images=prop_image_args(a.prop_image, allow_empty=True) if a.prop_image else None,
                            clip_videos={k: t for k, t in a.clip_video} if a.clip_video else None)
        started = _start_detached(m["id"])                 # it also takes its place in the queue
        emit({"ok": bool(started.get("ok")), "id": m["id"], "forked_from": m["forked_from"], "start": started,
              **({} if started.get("ok") else {"error": f"run {m['id']} was made but its worker did not start: "
                                                        f"{started.get('error')}", "error_type": "WorkerStartError"})})
        return 0 if started.get("ok") else 1
    if a.action == "retake":
        changes = {k: getattr(a, k) for k in runner.TAKE_FIELDS if hasattr(a, k)}
        if a.keyframe_at:
            changes["keyframe_images"] = [[x.rsplit("@", 1)[0], int(x.rsplit("@", 1)[1])] for x in a.keyframe_at]
        changes.pop("keyframe_at", None)
        if changes.get("pose_stage2") is not None:
            changes["pose_stage2"] = changes["pose_stage2"] == "on"
        r = runner.add_take(a.run_id, a.clip, changes, a.note)
        if not a.no_start:
            r["start"] = _start_detached(a.run_id)
        emit({"ok": True, **r})
        return 0
    if a.action == "takes":
        emit({"ok": True, **runner.list_takes(a.run_id)})
        return 0
    if a.action == "score":
        from . import clip_score
        emit({"ok": True, **clip_score.score(a.run_id, a.clip, a.take, with_motion=not a.no_motion,
                                                     critic_dir=a.critic, tag=a.tag)})
        return 0
    if a.action == "use-take":
        emit({"ok": True, **runner.use_take(a.run_id, a.clip, a.take)})
        return 0
    if a.action == "start":
        if a.through_clip is not None:
            runner.set_clip_limit(a.run_id, a.through_clip or None)
        if a.detach:
            r = _start_detached(a.run_id, a.new_attempt)
            emit(r)
            return 0 if r["ok"] else 1
        st = runner.start(a.run_id, new_attempt=a.new_attempt, launch=a.launch, on_event=emit)
        ok = st["status"] in ("done", "paused")                 # paused = stopped after --through-clip K, as asked
        emit({"ok": ok, "id": a.run_id, "status": st["status"],
              **({} if ok else {"error": st.get("error") or st["status"], "error_type": "Run" + st["status"].title()})})
        return 0 if ok else 1
    if a.action == "status":
        emit({"ok": True, **runner.status(a.run_id)})
        return 0
    if a.action == "list":
        emit({"ok": True, "runs": runner.list_runs(a.limit)})
        return 0
    if a.action == "cancel":
        emit({"ok": True, **runner.request_cancel(a.run_id)})
        return 0
    if a.action == "events":
        return events(a.run_id, a.since, a.follow)
    if a.action == "delete":
        emit({"ok": True, **runner.delete_run(a.run_id)})
        return 0
    raise CliInputError(f"unknown command {a}")


if __name__ == "__main__":
    sys.exit(main())
