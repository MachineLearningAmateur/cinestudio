"""The ComfyUI engine: cinestudio starts ComfyUI itself, hidden in the background, when it is needed.

ComfyUI must run for every image and video step. A user does not have to open Comfy Desktop (or any
ComfyUI window): `detect()` finds an installed ComfyUI once and saves its start command in
<home>/comfyui.json; `ensure_running()` starts it when it does not answer. With Comfy Desktop, the
command is the one that Comfy Desktop itself uses (same install, models, custom nodes and settings).

    python -m cinestudio comfy detect | start | stop | status

Safety: `stop` stops only a ComfyUI that cinestudio started (its pid is recorded), and only when no
request waits or runs in the cinestudio queue and ComfyUI has no queued prompts.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

from . import server, spawn
from .locks import FileLock, LockBusy, pid_alive

DESKTOP_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / "Comfy Desktop"
DROP_ARGS = {"--feature-flag"}                       # Comfy Desktop window options (with one value each)


class EngineError(RuntimeError):
    pass


def _home() -> Path:
    from .runner import home
    return home()


def config_path() -> Path:
    return _home() / "comfyui.json"


def load_config() -> dict | None:
    try:
        return json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save(cfg: dict) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")


def _clean_args(args: list[str]) -> list[str]:
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in DROP_ARGS:
            skip = True
            continue
        if a in ("--listen", "--port"):
            skip = True                                  # cinestudio sets these itself
            continue
        out.append(a)
    return out


def _from_desktop_log() -> dict | None:
    """The last ComfyUI start command in the Comfy Desktop log (the exact one Desktop uses)."""
    log = DESKTOP_DIR / "logs" / "app.log"
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    hits = re.findall(r"^> (.+?python(?:\.exe)? -s \S*main\.py.*)$", text, re.M)
    if not hits:
        return None
    parts = shlex.split(hits[-1].replace("\\", "/"), posix=True)
    python = parts[0]
    main_i = next(i for i, p in enumerate(parts) if p.endswith("main.py"))
    inst = _desktop_install()
    cwd = inst.get("installPath") if inst else str(Path(python).parents[3])
    args = [a.replace("/", "\\") if sys.platform.startswith("win") and (":" in a and "/" in a) else a for a in parts[main_i + 1:]]
    # the user name in the log is hidden ("[REDACTED]"): put the real profile folder back
    args = [a.replace("C:\\Users\\[REDACTED]", str(Path.home())) for a in args]
    return {"source": "Comfy Desktop", "python": python.replace("/", "\\") if sys.platform.startswith("win") else python,
            "main": parts[main_i].replace("/", "\\"), "cwd": cwd, "args": _clean_args(args)}


def _desktop_install() -> dict | None:
    try:
        items = json.loads((DESKTOP_DIR / "installations.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return items[0] if items else None


def detect(save: bool = True) -> dict:
    """Find an installed ComfyUI and its start command. Comfy Desktop first; else COMFYUI_DIR (a
    ComfyUI folder with main.py, started with the python of its .venv or the current python)."""
    cfg = _from_desktop_log()
    if cfg is None and os.environ.get("COMFYUI_DIR"):
        d = Path(os.environ["COMFYUI_DIR"])
        py = next((p for p in (d / ".venv" / "Scripts" / "python.exe", d / ".venv" / "bin" / "python") if p.is_file()), None)
        cfg = {"source": "COMFYUI_DIR", "python": str(py or sys.executable), "main": "main.py", "cwd": str(d),
               "args": ["--enable-manager"]}
    if cfg is None:
        raise EngineError("no ComfyUI found: install Comfy Desktop (and start it once), or set COMFYUI_DIR to a "
                          "ComfyUI folder")
    if not Path(cfg["python"]).is_file():
        raise EngineError(f"the ComfyUI python is missing: {cfg['python']}")
    cfg["url"] = server.comfyui_url()
    if save:
        old = load_config() or {}
        _save({**cfg, "started_pid": old.get("started_pid")})
    return cfg


def _lifecycle_lock() -> FileLock:
    """One start or stop at a time (an app start and a job must not both start a ComfyUI)."""
    return FileLock(_home() / "comfy.lock", timeout=300)


def _ctime(pid: int | None) -> float | None:
    """Creation time of a process (with its pid, the identity of that process: Windows reuses pids)."""
    if not pid:
        return None
    try:
        import psutil
        return psutil.Process(int(pid)).create_time()
    except Exception:  # noqa: BLE001 - gone, or no access
        return None


def _record(pid: int, independent: bool | None = None) -> None:
    cfg = load_config() or {}
    cfg.update(started_pid=pid, started_ctime=_ctime(pid), started_at=time.time())
    if independent is not None:
        cfg["independent"] = independent
    _save(cfg)


def owns_engine() -> bool:
    """True if the ComfyUI process that cinestudio started (same pid AND same creation time) still runs."""
    cfg = load_config() or {}
    pid, ct = cfg.get("started_pid"), cfg.get("started_ctime")
    now = _ctime(pid)
    return bool(pid and ct and now and abs(now - float(ct)) < 1.0)


def adopt_listener() -> int | None:
    """After a restart through ComfyUI-Manager the process is new: record the one that now listens on the
    port as cinestudio's (call this only when the restarted engine was cinestudio's)."""
    port = _port(server.comfyui_url())
    try:
        import psutil
        for c in psutil.net_connections(kind="tcp"):
            if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid:
                _record(c.pid)
                return c.pid
    except Exception:  # noqa: BLE001 - no access to the connection table
        return None
    return None


def _port(url: str) -> int:
    m = re.search(r":(\d+)", url.split("//", 1)[-1])
    return int(m.group(1)) if m else 8188


def status() -> dict:
    url = server.comfyui_url()
    cfg = load_config() or {}
    ours = owns_engine()
    out = {"url": url, "running": server.alive(url), "started_by_cinestudio": ours,
           "pid": cfg.get("started_pid") if ours else None, "configured": bool(cfg.get("python")),
           "source": cfg.get("source"), "log": str(_home() / "comfyui.log")}
    if ours and cfg.get("independent") is False:
        out["warning"] = "Windows did not let ComfyUI leave the app's process group"
    return out


def has_sage_attention(cfg: dict) -> bool:
    """SageAttention installed in ComfyUI's own Python environment (site-packages/sageattention)."""
    root = Path(cfg["python"]).resolve().parent.parent
    return any(root.glob("Lib/site-packages/sageattention/__init__.py")) or         any(root.glob("lib/python3*/site-packages/sageattention/__init__.py"))


NEXT_FAMILIES: list[str] | None = None   # set by server.ensure_family for the restart it is about to do
# SageAttention only where it was checked: the LTX-2.5 distilled clips were right with it (2026-10-21); Qwen-Image-Edit
# made an ALL-BLACK picture with it (the Layout-To-Render look image, 2026-10-21) - a family not listed runs without it
SAGE_FAMILIES = ("ltx25",)


def io_dir(kind: str) -> Path:
    """ComfyUI's "input" or "output" folder (the --input-directory / --output-directory of its saved command)."""
    cfg = load_config() or {}
    args = cfg.get("args") or []
    flag = f"--{kind}-directory"
    if flag in args and args.index(flag) + 1 < len(args):
        return Path(args[args.index(flag) + 1])
    main = Path(cfg.get("main") or "ComfyUI/main.py")
    return (Path(cfg.get("cwd") or ".") / main).parent / kind   # Desktop: cwd/ComfyUI/main.py; COMFYUI_DIR: cwd/main.py


def base_args(cfg: dict) -> list[str]:
    """The saved start flags without --use-sage-attention: speed_args alone decides it per model family (a saved flag
    would turn SageAttention on for Qwen-Image-Edit too, which then draws all-black pictures). Plus the models folder
    of the settings, when ComfyUI does not know it yet (a rented machine: the models on a network volume)."""
    return [a for a in cfg.get("args", []) if a != "--use-sage-attention"] + model_path_args(cfg)


MODEL_SUBDIRS = ("checkpoints", "diffusion_models", "unet", "loras", "vae", "text_encoders", "clip", "upscale_models",
                 "latent_upscale_models", "embeddings", "controlnet", "sam2", "geometry_estimation")


def model_path_args(cfg: dict) -> list[str]:
    """`--extra-model-paths-config FILE` for the settings' models folder (`settings models-dir`), else []. The downloader
    writes there; without this flag ComfyUI did not look there (Codex review 2026-10-22). Not when the saved command
    already has its own paths file (Comfy Desktop) or the folder is ComfyUI's own models folder."""
    from . import settings
    from .runner import home
    md = settings.get().get("models_dir")
    if not md or "--extra-model-paths-config" in (cfg.get("args") or []):
        return []
    own = (Path(cfg.get("cwd") or ".") / Path(cfg.get("main") or "main.py")).parent / "models"
    if Path(md).resolve() == own.resolve():
        return []
    lines = ["cinestudio:", f"    base_path: {Path(md).as_posix()}", "    is_default: true"]
    lines += [f"    {d}: {d}/" for d in MODEL_SUBDIRS]
    f = home() / "extra_model_paths.yaml"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ["--extra-model-paths-config", str(f)]


def speed_args(cfg: dict) -> list[str]:
    """Extra start flags for speed: --use-sage-attention when SageAttention is installed (the user approved it
    2026-10-21; setting "sage_attention": false turns it off), and only for a restart into model families of
    SAGE_FAMILIES (NEXT_FAMILIES; a start with unknown families gets none). Never twice."""
    return ["--use-sage-attention"] if sage_for(NEXT_FAMILIES or [], cfg) else []


def sage_for(families, cfg: dict | None = None) -> bool:
    """True when an engine started for `families` gets --use-sage-attention: SageAttention installed, not turned off
    in the settings, and EVERY family Sage-safe (Qwen-Image-Edit made all-black pictures with it)."""
    from . import settings
    if settings.get().get("sage_attention") is False:
        return False
    fams = list(families or [])
    if not fams or not all(f.split("@")[0] in SAGE_FAMILIES for f in fams):
        return False
    try:
        return has_sage_attention(cfg or load_config() or {})
    except Exception:  # noqa: BLE001 - no config yet: no flag
        return False


def start(wait: float = 240) -> dict:
    """Start ComfyUI hidden in the background (if it does not answer yet) and wait until it answers."""
    url = server.comfyui_url()
    with _lifecycle_lock():
        if server.alive(url):                              # checked again under the lock
            return {**status(), "already_running": True}
        cfg = load_config() or detect()
        cmd = [cfg["python"], "-s", cfg["main"], *base_args(cfg), *speed_args(cfg), "--listen", "127.0.0.1",
               "--port", str(_port(url))]
        pid, independent = spawn.detached(cmd, _home() / "comfyui.log", cfg["cwd"])
        _record(pid, independent)
        return _wait_started(url, pid, wait)


def _wait_started(url: str, pid: int, wait: float) -> dict:
    t0 = time.time()
    while time.time() - t0 < wait:
        if server.alive(url):
            return {**status(), "seconds": round(time.time() - t0, 1)}
        if not pid_alive(pid):
            tail = (_home() / "comfyui.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()[-5:]
            raise EngineError("ComfyUI stopped while it started: " + " | ".join(tail))
        time.sleep(2)
    raise EngineError(f"ComfyUI did not answer within {wait:.0f} s; see {_home() / 'comfyui.log'}")


def ensure_running() -> dict:
    """Start ComfyUI when it does not answer (used before every image or video step).
    CINESTUDIO_AUTOSTART_COMFYUI=0 turns this off everywhere: then a step stops with a clear error."""
    if server.alive(server.comfyui_url()):
        return {"running": True}
    if os.environ.get("CINESTUDIO_AUTOSTART_COMFYUI", "1") == "0":
        raise EngineError("ComfyUI does not answer, and CINESTUDIO_AUTOSTART_COMFYUI=0: start ComfyUI first")
    return start()


def stop() -> dict:
    """Stop the ComfyUI that cinestudio started, when no work waits or runs."""
    from . import workqueue
    with _lifecycle_lock():
        if not owns_engine():
            raise EngineError("this ComfyUI was not started by cinestudio (close it where it was started)")
        gpu = server.gpu_lock(note={"engine": "stop"})
        try:
            gpu.acquire()                                  # timeout 0: refuse while a step uses ComfyUI
        except LockBusy as exc:
            raise EngineError("a step uses ComfyUI now; stop it first, or wait until it ends") from exc
        try:
            if workqueue.tickets():
                raise EngineError("work waits or runs in the queue; stop it first, or wait until it ends")
            url = server.comfyui_url()
            if server.queue_items(url):
                raise EngineError("ComfyUI has prompts in its queue")
            return {"stopped": True, "pid": _kill_own()}
        finally:
            gpu.release()


def _kill_own() -> int:
    """Stop the process tree of cinestudio's own engine (call with the lifecycle lock held)."""
    pid = int((load_config() or {})["started_pid"])
    if not owns_engine():                              # checked right before the stop: never a reused pid
        raise EngineError("the recorded ComfyUI process changed; nothing was stopped")
    if sys.platform.startswith("win"):
        import subprocess
        r = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
        if r.returncode != 0:
            raise EngineError(f"taskkill failed: {(r.stdout + r.stderr).strip()[-200:]}")
    else:
        os.kill(pid, 15)
    t0 = time.time()
    while pid_alive(pid) and time.time() - t0 < 15:
        time.sleep(0.5)
    if pid_alive(pid):
        raise EngineError(f"ComfyUI (pid {pid}) did not stop")
    url = server.comfyui_url()
    t0 = time.time()
    while server.alive(url) and time.time() - t0 < 15:  # the port is free again
        time.sleep(0.5)
    cfg = load_config() or {}
    for k in ("started_pid", "started_ctime"):
        cfg.pop(k, None)
    _save(cfg)
    return pid


def restart_own(wait: float = 240) -> float:
    """Restart cinestudio's own engine (for a model-family change): stop its process and start it again
    with the saved command. ComfyUI-Manager's own restart is not used: its 'legacy' restart splits the
    command at spaces and broke on a path with a space. The caller holds the GPU lock."""
    t0 = time.time()
    with _lifecycle_lock():
        _kill_own()
        cfg = load_config() or detect()
        url = server.comfyui_url()
        cmd = [cfg["python"], "-s", cfg["main"], *base_args(cfg), *speed_args(cfg), "--listen", "127.0.0.1",
               "--port", str(_port(url))]
        pid, independent = spawn.detached(cmd, _home() / "comfyui.log", cfg["cwd"])
        _record(pid, independent)
        _wait_started(url, pid, wait)
    return round(time.time() - t0, 1)
