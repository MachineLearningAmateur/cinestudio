"""Model tiers: which files a tier needs, where they go, what is there, and a fast verified download.

    python -m cinestudio models status [--tier large] [--feature video]
    python -m cinestudio models download [--tier 8gb] [--feature video --feature pose] [--detach]

Tiers and files come from setup/models.json ("model_tiers"): every file is pinned to a repository commit with its
size and SHA-256. A download goes to <file>.part (curl, resumable, several files at once), and the file gets its
final name only after the size and the hash match, so ComfyUI never sees a half-written model. A per-file lock stops
two downloads of the same file. The Hugging Face token (settings.token("hf")) is sent only to huggingface.co.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from . import settings
from .locks import FileLock, LockBusy

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "setup" / "models.json"
PARALLEL = 3                      # files at the same time (one curl each); more rarely helps against one CDN
DISK_MARGIN = 5 * 1024 ** 3       # keep 5 GiB free after the download


class ModelsError(RuntimeError):
    pass


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _tiers() -> dict:
    return _manifest()["model_tiers"]


OLD_NAMES = {"24gb": "large"}      # renamed 2026-10-24 (the bf16 files need an 80 GB+ GPU); old settings and runs keep it


def canon(tier: str | None) -> str | None:
    """The current name of a tier (an old name such as "24gb" -> "large")."""
    return OLD_NAMES.get(tier, tier) if tier else tier


def tier_names() -> list[str]:
    return list(_tiers()["tiers"])


def features() -> dict[str, list[str]]:
    return dict(_tiers()["features"])


def bundle(tier: str) -> dict[str, dict]:
    """slot -> file entry (dir, file, url, size, sha256) of a tier."""
    t = _tiers()
    tier = canon(tier)
    if tier not in t["tiers"]:
        raise ModelsError(f"unknown model tier {tier!r}; use one of {list(t['tiers'])}")
    return {**t["common"], **t["tiers"][tier]["slots"]}


def graph_settings(tier: str) -> dict[str, dict]:
    """class_type -> {input: value} that the tier writes into every graph of a run (e.g. larger VAE decode tiles on a
    large GPU). Keys that start with "_" are notes."""
    t = _tiers()
    tier = canon(tier)
    if tier not in t["tiers"]:
        raise ModelsError(f"unknown model tier {tier!r}; use one of {list(t['tiers'])}")
    return {k: dict(v) for k, v in (t["tiers"][tier].get("graph_settings") or {}).items() if not k.startswith("_")}


def engine_settings(tier: str) -> dict:
    """The tier's engine policy ({} = one model family per engine: a restart at every family change)."""
    t = _tiers()
    tier = canon(tier)
    if tier not in t["tiers"]:
        raise ModelsError(f"unknown model tier {tier!r}; use one of {list(t['tiers'])}")
    return {k: v for k, v in (t["tiers"][tier].get("engine") or {}).items() if not k.startswith("_")}


_JOB_TIER: str | None = None       # the tier frozen into the job this process runs (a worker runs one job)


def use_tier(tier: str | None) -> None:
    """A job worker: use the tier that was frozen into the job when it was made, not today's setting."""
    global _JOB_TIER
    _JOB_TIER = tier


def job_tier() -> str:
    return canon(_JOB_TIER or settings.get()["model_tier"])


def job_bundle() -> dict[str, dict]:
    return bundle(job_tier())


def tier_info(tier: str) -> dict:
    t = _tiers()["tiers"][canon(tier)]
    return {k: t[k] for k in ("label", "vram_gb", "ram_gb", "about")}


# ------------------------------------------------------------------ where ComfyUI keeps things

def comfy_root() -> Path | None:
    """The ComfyUI folder (the one with main.py and custom_nodes), from the saved engine config."""
    from . import comfy
    cfg = comfy.load_config()
    if not cfg:
        try:
            cfg = comfy.detect(save=False)
        except Exception:  # noqa: BLE001 - no ComfyUI found yet (a fresh install)
            cfg = None
    if not cfg:
        d = os.environ.get("COMFYUI_DIR")
        return Path(d) if d else None
    main = Path(cfg["cwd"]) / cfg.get("main", "main.py")
    return main.parent


def _desktop_models_dir(cfg: dict) -> Path | None:
    """The default models folder of a Comfy Desktop install: base_path of the is_default entry of the
    --extra-model-paths-config file."""
    args = list(cfg.get("args") or [])
    if "--extra-model-paths-config" not in args:
        return None
    i = args.index("--extra-model-paths-config")
    try:
        text = Path(args[i + 1]).read_text(encoding="utf-8")
    except (IndexError, OSError):
        return None
    for block in re.split(r"\n(?=\S)", text):
        if re.search(r"^\s*is_default:\s*true", block, re.M):
            m = re.search(r"^\s*base_path:\s*['\"]?(.+?)['\"]?\s*$", block, re.M)
            if m:
                return Path(m.group(1))
    return None


def models_dir() -> Path:
    """The ComfyUI 'models' folder: the setting, else Comfy Desktop's shared folder, else <ComfyUI>/models."""
    s = settings.get().get("models_dir")
    if s:
        return Path(s)
    from . import comfy
    cfg = comfy.load_config()
    if cfg:
        d = _desktop_models_dir(cfg)
        if d:
            return d
    root = comfy_root()
    if root is None:
        raise ModelsError("no ComfyUI found: install it (setup/install.sh) or set the models folder "
                          "(cinestudio settings models-dir PATH)")
    return root / "models"


def aux_dir() -> Path | None:
    """The comfyui_controlnet_aux node folder (DWPose), or None."""
    env = os.environ.get("CONTROLNET_AUX")
    if env:
        return Path(env)
    root = comfy_root()
    d = root / "custom_nodes" / "comfyui_controlnet_aux" if root else None
    return d if d and d.is_dir() else None


def path_of(entry: dict) -> Path:
    d = entry["dir"]
    if d.startswith("@aux/"):
        aux = aux_dir()
        if aux is None:
            raise ModelsError("the comfyui_controlnet_aux node is not installed (DWPose)")
        return aux / d[len("@aux/"):] / entry["file"]
    return models_dir() / d / entry["file"]


# ------------------------------------------------------------------ status

def optional_features(tier: str) -> list[str]:
    """Features downloaded only when explicitly requested, rather than with a full tier download."""
    t = _tiers()
    tier = canon(tier)
    if tier not in t["tiers"]:
        raise ModelsError(f"unknown model tier {tier!r}; use one of {list(t['tiers'])}")
    return list(t["tiers"][tier].get("optional_features") or [])


def _slots(tier: str, feats: list[str] | None) -> dict[str, dict]:
    b = bundle(tier)
    f = features()
    unknown = [x for x in feats or [] if x not in f]
    if unknown:
        raise ModelsError(f"unknown feature {unknown[0]!r}; use {list(f)}")
    asked = set(feats or [])
    optional = {s for x in optional_features(tier) if x not in asked for s in f.get(x, [])}
    if not feats:
        return {k: v for k, v in b.items() if k not in optional}
    keep = {s for x in feats for s in f[x]} - optional
    return {k: v for k, v in b.items() if k in keep}


def status(tier: str | None = None, feats: list[str] | None = None) -> dict:
    """What a tier needs and what is there (by size: a full hash of 100 GB takes minutes; `verify` does that)."""
    tier = canon(tier or settings.get()["model_tier"])
    files, need, have = [], 0, 0
    for slot, e in _slots(tier, feats).items():
        rec = {"slot": slot, "file": e["file"], "dir": e["dir"], "size": e["size"]}
        try:
            p = path_of(e)
        except ModelsError as exc:
            rec.update(state="unavailable", error=str(exc))
            files.append(rec)
            need += e["size"]
            continue
        part = p.with_name(p.name + ".part")
        if p.is_file():
            rec["state"] = "ok" if p.stat().st_size == e["size"] else "wrong size"
        elif part.is_file():
            rec.update(state="partial", have=part.stat().st_size)
        else:
            rec["state"] = "missing"
        got = e["size"] if rec["state"] == "ok" else rec.get("have", 0)
        have += got
        need += e["size"] - got
        files.append(rec)
    try:
        md = models_dir()
        free = shutil.disk_usage(md if md.exists() else md.anchor).free
    except (ModelsError, OSError):
        md, free = None, None
    ready = all(f["state"] == "ok" for f in files)
    return {"tier": tier, **tier_info(tier), "features": feats or list(features()), "ready": ready,
            "total_bytes": have + need, "have_bytes": have, "missing_bytes": need, "files": files,
            "models_dir": str(md) if md else None, "disk_free": free,
            "disk_ok": free is None or free - need >= DISK_MARGIN}


def verify(tier: str | None = None, feats: list[str] | None = None, on_event=None) -> dict:
    """Full SHA-256 check of the files that are there."""
    tier = canon(tier or settings.get()["model_tier"])
    bad = []
    for slot, e in _slots(tier, feats).items():
        p = path_of(e)
        if not p.is_file():
            continue
        ok = p.stat().st_size == e["size"] and _sha256(p) == e["sha256"]
        if on_event:
            on_event({"event": "verified", "file": e["file"], "ok": ok})
        if not ok:
            bad.append(e["file"])
    return {"tier": tier, "bad": bad, "ok": not bad}


def _sha256(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ download

class _Stop(Exception):
    pass


def _curl_one(url: str, part: Path, token: str | None, stop: threading.Event) -> int:
    """One curl attempt (resumes the .part). Returns the HTTP status (0 = no answer). --fail: an error page is
    never appended to the .part file."""
    curl = shutil.which("curl")
    if not curl:
        raise ModelsError("curl was not found (it comes with Windows 10+, macOS and most Linux systems)")
    cmd = [curl, "-L", "--fail", "-sS", "-C", "-", "--retry", "5", "--retry-delay", "3", "--connect-timeout", "30",
           "-o", str(part), "-w", "%{http_code}", url]
    if token and urlsplit(url).hostname == "huggingface.co":
        # curl drops this header on the redirect to the CDN host (a signed URL needs none): the token stays on HF
        cmd[1:1] = ["-H", f"Authorization: Bearer {token}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    while proc.poll() is None:
        if stop.is_set():
            proc.terminate()
            proc.wait(timeout=10)
            raise _Stop()
        time.sleep(0.5)
    out = (proc.stdout.read() or b"").decode(errors="replace").strip()
    return int(out[-3:]) if out[-3:].isdigit() else 0


def _fetch(e: dict, token: str | None, stop: threading.Event, emit) -> dict:
    dst = path_of(e)
    rec = {"file": e["file"], "dir": e["dir"]}
    if dst.is_file() and dst.stat().st_size == e["size"]:
        return {**rec, "state": "ok", "skipped": True}
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    try:
        lock = FileLock(dst.with_name(dst.name + ".lock"), timeout=0).acquire()
    except LockBusy:
        return {**rec, "state": "busy", "error": "another download of this file is running"}
    try:
        emit({"event": "download_start", "file": e["file"], "size": e["size"]})
        for attempt in range(8):
            if part.is_file() and part.stat().st_size > e["size"]:
                part.unlink()                                   # longer than the pinned file: start again
            if part.is_file() and part.stat().st_size == e["size"]:
                break
            code = _curl_one(e["url"], part, token, stop)
            if code in (401, 403):
                return {**rec, "state": "error", "error_type": "auth",
                        "error": "access refused: this model needs a Hugging Face token with access "
                                 "(accept the model's licence on its Hugging Face page, then set the token)"}
            if code == 404:
                return {**rec, "state": "error", "error_type": "missing", "error": "the pinned file is not there (404)"}
            if code == 416 and part.is_file() and part.stat().st_size == e["size"]:
                break                                           # nothing left to fetch
            if code == 429:
                time.sleep(min(60, 10 * (attempt + 1)))
            if part.is_file() and part.stat().st_size == e["size"]:
                break
        have = part.stat().st_size if part.is_file() else 0
        if have != e["size"]:
            return {**rec, "state": "incomplete", "have": have,
                    "error": f"{have} of {e['size']} bytes: start the download again to continue"}
        emit({"event": "download_check", "file": e["file"]})
        if _sha256(part) != e["sha256"]:
            part.unlink()                                       # a full-length bad copy cannot be resumed
            return {**rec, "state": "error", "error_type": "hash",
                    "error": "the downloaded file does not match its SHA-256 (deleted: download again)"}
        os.replace(part, dst)
        emit({"event": "download_done", "file": e["file"]})
        return {**rec, "state": "ok"}
    finally:
        lock.release()
        for leftover in (lock.path, lock.path.with_name(lock.path.name + ".who")):   # keep the models folder clean
            try:
                leftover.unlink()
            except OSError:
                pass


def download(tier: str | None = None, feats: list[str] | None = None, on_event=None,
             stop: threading.Event | None = None) -> dict:
    """Download the missing files of a tier (several at once). Returns per-file results."""
    tier = canon(tier or settings.get()["model_tier"])
    emit = on_event or (lambda e: None)
    stop = stop or threading.Event()
    st = status(tier, feats)
    if not st["disk_ok"]:
        raise ModelsError(f"not enough disk space in {st['models_dir']}: {st['missing_bytes'] / 1e9:.1f} GB to "
                          f"download, {st['disk_free'] / 1e9:.1f} GB free (keep {DISK_MARGIN / 1024 ** 3:.0f} GB spare)")
    todo = [e for s, e in _slots(tier, feats).items()
            if next(f for f in st["files"] if f["slot"] == s)["state"] != "ok"]
    token = settings.token("hf")
    results = []
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futs = [pool.submit(_fetch, e, token, stop, emit) for e in todo]
        for f in futs:
            try:
                results.append(f.result())
            except _Stop:
                results.append({"state": "stopped"})
            except ModelsError as exc:
                results.append({"state": "error", "error": str(exc)})
    failed = [r for r in results if r.get("state") not in ("ok",)]
    return {"tier": tier, "files": results, "ok": not failed and not stop.is_set(), "stopped": stop.is_set(),
            "failed": failed}


# ------------------------------------------------------------------ the detached download worker

def _dl_dir() -> Path:
    from .runner import home
    d = home() / "downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_path() -> Path:
    return _dl_dir() / "status.json"


def download_state() -> dict:
    """The worker's last state plus live progress (bytes in the .part files)."""
    try:
        st = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"state": "idle"}
    if st.get("tier"):
        st["tier"] = canon(st["tier"])                  # a worker record from before the rename ("24gb" -> "large")
    from .locks import is_locked
    if st.get("state") == "running" and not is_locked(_dl_dir() / "worker.lock"):
        st["state"] = "stopped"                         # the worker died (crash, reboot)
    if st.get("state") == "running":
        try:
            s = status(st["tier"], st.get("features"))
            st["have_bytes"], st["total_bytes"] = s["have_bytes"], s["total_bytes"]
        except ModelsError:
            pass
    return st


def start_download(tier: str, feats: list[str] | None = None) -> dict:
    """Start the download worker in the background (it keeps going when the app closes)."""
    tier = canon(tier)
    from . import spawn
    from .locks import is_locked
    bundle(tier)                                        # refuse an unknown tier before anything starts
    if is_locked(_dl_dir() / "worker.lock"):
        raise ModelsError("a model download is already running")
    (_dl_dir() / "stop").unlink(missing_ok=True)
    cmd = [sys.executable, "-m", "cinestudio", "models", "work-download", tier]
    for f in feats or []:
        cmd += ["--feature", f]
    pid, _ = spawn.detached(cmd, _dl_dir() / "worker.log", ROOT)
    _write_state({"state": "starting", "tier": tier, "features": feats, "pid": pid, "started": time.time()})
    return {"started": True, "tier": tier, "pid": pid}


def request_stop() -> dict:
    (_dl_dir() / "stop").write_text(str(time.time()), encoding="utf-8")
    return {"stop_requested": True}


_STATE_LOCK = threading.Lock()


def _write_state(d: dict) -> None:
    """The worker's status.json. Several download threads report at once: one writer at a time, and each write has
    its own temporary file (two threads once shared one name, and the second replace found it gone)."""
    with _STATE_LOCK:
        tmp = _state_path().with_name(f"status.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
        os.replace(tmp, _state_path())


def work_download(tier: str, feats: list[str] | None = None, on_event=None) -> dict:
    """The worker: one at a time (worker.lock), writes status.json, stops when the stop file appears."""
    tier = canon(tier)
    lock = FileLock(_dl_dir() / "worker.lock", timeout=0).acquire()
    stop = threading.Event()
    rec = {"state": "running", "tier": tier, "features": feats, "pid": os.getpid(), "started": time.time(),
           "current": []}

    def watch():
        while not stop.wait(1.0):
            if (_dl_dir() / "stop").is_file():
                stop.set()
    threading.Thread(target=watch, daemon=True).start()

    def emit(ev):
        if ev.get("event") == "download_start":
            rec["current"].append(ev["file"])
        elif ev.get("event") == "download_done" and ev["file"] in rec["current"]:
            rec["current"].remove(ev["file"])
        _write_state(rec)
        if on_event:
            on_event(ev)
    try:
        _write_state(rec)
        res = download(tier, feats, emit, stop)
        rec.update(state="stopped" if res["stopped"] else "done" if res["ok"] else "failed",
                   failed=res["failed"], ended=time.time(), current=[])
        return res
    except Exception as exc:  # noqa: BLE001 - the app shows the reason
        rec.update(state="failed", error=str(exc), ended=time.time(), current=[])
        raise
    finally:
        stop.set()
        _write_state(rec)
        lock.release()
