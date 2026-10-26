"""One model family per ComfyUI process (best effort on a shared ComfyUI).

ComfyUI keeps loaded models in RAM between jobs. On a 32 GB machine, LTX-2.3 (~39 GB of files) and
Wan 2.2 5B (~17 GB) do not fit together: a Wan job after LTX jobs stopped ComfyUI (2026-10-18).
ComfyUI's /free endpoint is not a safe way out on such machines (a reload after /free also crashed
ComfyUI), so the only guarantee is a new ComfyUI process.

Rule: before a job, ``ensure_family(family)`` makes sure that the ComfyUI process holds no model
family other than `family`. It keeps the process only if BOTH are true:
  - the recorded family equals `family`, and
  - only this tool used the process since it was recorded: the newest ComfyUI history entry is a
    prompt this tool submitted (or the process is the fresh one this tool restarted, with an empty
    history), and the queue holds only prompts this tool submitted.
Otherwise it restarts ComfyUI, after the queue is empty: it waits for its own prompts (other runs)
and refuses (``ServerBusy``) while ComfyUI holds prompts that this tool did not submit.

Ownership is a list of prompt ids recorded at submit time (``note_owned``); the comfy-api-proxy job
id is the ComfyUI prompt id. A file-name prefix is NOT proof of ownership (anyone can run our graphs).

Limits (documented for users): this is best effort on a ComfyUI that people also use by hand. A
prompt queued in ComfyUI Desktop between the queue check and the restart can still be stopped, and
a prompt of another family queued right before ours can still run in the same process. Do not use
the same ComfyUI by hand while a cinestudio run is working. All cinestudio processes that use
one ComfyUI share one GPU lock (``gpu_lock``), so they never interleave a restart and a submit.

Restart methods (CINESTUDIO_RESTART):
  manager  POST /v2/manager/reboot (ComfyUI-Manager, local ComfyUI started with --enable-manager)
  none     never restart; raise RestartNeeded with an instruction for the user
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

from .locks import FileLock

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".cinestudio" / "server_state.json"
DEFAULT_COMFYUI = "http://127.0.0.1:8188"
MAX_OWNED = 500


class ServerBusy(RuntimeError):
    """ComfyUI has prompts that this tool did not send; a restart would stop them."""


class RestartNeeded(RuntimeError):
    """The model family changes, but this tool may not restart ComfyUI (CINESTUDIO_RESTART=none)."""


class RestartFailed(RuntimeError):
    """ComfyUI-Manager accepted the reboot, but no new ComfyUI process could be confirmed."""


def comfyui_url() -> str:
    return os.environ.get("COMFYUI_URL", DEFAULT_COMFYUI).rstrip("/")


def gpu_lock(url: str | None = None, timeout: float = 0, note: dict | None = None) -> FileLock:
    """The lock that every cinestudio process takes before it restarts or submits to this ComfyUI.
    It lives in the system temp folder, keyed by the ComfyUI URL, so it is shared even between
    different CINESTUDIO_HOME folders."""
    key = hashlib.sha256((url or comfyui_url()).encode()).hexdigest()[:12]
    return FileLock(Path(tempfile.gettempdir()) / f"cinestudio-gpu-{key}.lock", timeout=timeout, note=note)


def _get(url: str, timeout: float = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _post(url: str, timeout: float = 10) -> int:
    req = urllib.request.Request(url, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return 0   # the server went down while it answered: expected for a restart


def _load() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, STATE)


def alive(url: str | None = None) -> bool:
    try:
        _get(f"{url or comfyui_url()}/system_stats")
        return True
    except Exception:  # noqa: BLE001 - any failure means "not answering"
        return False


HISTORY_LIMIT = 200


def history_ids(url: str, n: int = HISTORY_LIMIT) -> list[str]:
    """Prompt ids in ComfyUI's history, oldest first (the last one is the newest)."""
    return list(_get(f"{url}/history?max_items={n}", timeout=15))


def queue_items(url: str) -> list[tuple[str, dict]]:
    """(prompt id, graph) of running and pending prompts."""
    q = _get(f"{url}/queue", timeout=10)
    return [(item[1], item[2]) for item in q.get("queue_running", []) + q.get("queue_pending", [])]


def prompt_status(url: str, prompt_id: str) -> str:
    """What ComfyUI itself knows about a prompt: done_ok, done_error, queued (or running), or unknown.
    'unknown' is not a failure: ComfyUI's history can be cut or cleared."""
    hist = _get(f"{url}/history/{prompt_id}", timeout=15)
    if prompt_id in hist:
        st = hist[prompt_id].get("status") or {}
        ok = st.get("status_str") == "success" or (st.get("completed") and st.get("status_str") != "error")
        return "done_ok" if ok else "done_error"
    if prompt_id in queue_ids(url):
        return "queued"
    return "unknown"


def download_output(url: str, prompt_id: str, node: str, dst: Path) -> Path:
    """The first file that output node `node` of a finished prompt wrote, from ComfyUI directly."""
    import urllib.parse
    outputs = _get(f"{url}/history/{prompt_id}", timeout=15)[prompt_id].get("outputs", {})
    entry = None
    for value in (outputs.get(str(node)) or {}).values():
        if isinstance(value, list) and value and isinstance(value[0], dict) and value[0].get("filename"):
            entry = value[0]
            break
    if entry is None:
        raise RuntimeError(f"prompt {prompt_id}: output node {node} wrote no file")
    q = urllib.parse.urlencode({"filename": entry["filename"], "subfolder": entry.get("subfolder", ""),
                                "type": entry.get("type", "output")})
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(f"{url}/view?{q}", timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dst)
    return dst


def queue_ids(url: str | None = None) -> set[str]:
    """Prompt ids that ComfyUI runs or has queued now."""
    return {pid for pid, _ in queue_items(url or comfyui_url())}


def owned_ids(url: str | None = None) -> set[str]:
    return set(_load().get(url or comfyui_url(), {}).get("owned", []))


def note_owned(prompt_id: str, url: str | None = None) -> None:
    """Record a prompt id that this tool submitted (call it as soon as the submit returns the id)."""
    url = url or comfyui_url()
    state = _load()
    rec = state.setdefault(url, {})
    owned = [p for p in rec.get("owned", []) if p != prompt_id] + [prompt_id]
    rec["owned"] = owned[-MAX_OWNED:]
    rec["last_prompt"] = prompt_id
    _save(state)


def foreign_work(url: str, owned: set[str] | None = None) -> list[str]:
    """Prompt ids in the queue that this tool did not submit."""
    owned = owned_ids(url) if owned is None else owned
    return [pid for pid, _ in queue_items(url) if pid not in owned]


def restart(url: str, timeout: float = 240) -> float:
    """Restart ComfyUI and wait until it answers again. Returns the seconds it took. Raises
    RestartFailed unless a new process is confirmed: ComfyUI was seen down, or its history went
    from non-empty to empty (a new process starts with an empty history)."""
    method = os.environ.get("CINESTUDIO_RESTART", "manager")
    if method == "none":
        raise RestartNeeded("restart ComfyUI by hand (the model family changes), then run the job again")
    if method not in ("manager", "manager-shared"):
        raise ValueError(f"unknown CINESTUDIO_RESTART={method!r}")
    from . import comfy
    own = comfy.owns_engine()
    if method == "manager" and not own:
        # another program started this ComfyUI: someone else may use it, so it is never restarted automatically
        raise RestartNeeded("this ComfyUI was started outside cinestudio and the model family must change: "
                            "close it (cinestudio then starts its own), or restart it by hand, then run the job "
                            "again (or set CINESTUDIO_RESTART=manager-shared)")
    if own:
        return comfy.restart_own(timeout)             # cinestudio restarts its own engine itself
    try:
        before = history_ids(url, 1)
    except Exception:  # noqa: BLE001
        before = []
    t0 = time.time()
    status = _post(f"{url}/v2/manager/reboot")
    if status not in (0, 200):
        raise RuntimeError(f"ComfyUI-Manager refused the restart (HTTP {status}); start ComfyUI with "
                           f"--enable-manager, or set CINESTUDIO_RESTART=none and restart it by hand")
    seen_down = False
    down_by = time.time() + 60
    while time.time() < down_by:                       # wait until the old process is gone
        if not alive(url):
            seen_down = True
            break
        time.sleep(0.5)
    while not alive(url):
        if time.time() - t0 > timeout:
            raise RestartFailed(f"ComfyUI did not come back within {timeout:.0f} s after the restart")
        time.sleep(2)
    if not seen_down:
        try:
            fresh = bool(before) and not history_ids(url, 1)
        except Exception:  # noqa: BLE001
            fresh = False
        if not fresh:
            raise RestartFailed("ComfyUI-Manager accepted the reboot, but ComfyUI never went down and its history "
                                "is unchanged: the old process (and its models) may still be loaded")
    return round(time.time() - t0, 1)


def _only_us(url: str, rec: dict, owned: set[str]) -> bool:
    """True if only this tool used the current process since the family was recorded."""
    try:
        hist = history_ids(url)
        queue = [pid for pid, _ in queue_items(url)]
    except Exception:  # noqa: BLE001 - not answering: unknown
        return False
    if any(pid not in owned for pid in queue):
        return False
    if len(hist) >= HISTORY_LIMIT:
        return False                                   # history may be cut: it cannot prove that only we used it
    if hist:
        return all(h in owned for h in hist)           # every prompt since the restart was ours
    return bool(rec.get("restarted_at"))       # the fresh process this tool started, nothing run yet


def engine_sage(url: str) -> bool | None:
    """Whether the RUNNING engine was started with --use-sage-attention, from its own start flags (/system_stats
    system.argv); None when they cannot be read. A state file cannot prove this: another program or a restart outside
    cinestudio can replace the process (Codex review 2026-10-24)."""
    try:
        argv = (_get(f"{url}/system_stats", timeout=5).get("system") or {}).get("argv")
    except Exception:  # noqa: BLE001 - not answering: unknown
        return None
    if not isinstance(argv, list):
        return None
    return "--use-sage-attention" in argv


def shared_engine() -> bool:
    """True when the model tier lets several model families stay loaded in one engine ("engine": {"share_families":
    true, "min_ram_gb": N} in setup/models.json; the large tier) AND this machine has that much RAM. The 8gb tier keeps
    one family per engine (a 32 GB PC cannot hold Qwen + LTX + ACE-Step weights). 2026-10-24: on a pod every family
    change restarted the engine and reloaded tens of GB from the network volume (docs/large_gpu_speed_research.md)."""
    try:
        from . import models
        eng = models.engine_settings(models.job_tier())
    except Exception:  # noqa: BLE001 - no tier list: the safe default
        return False
    if not eng.get("share_families"):
        return False
    try:
        import psutil
        return psutil.virtual_memory().total >= float(eng.get("min_ram_gb", 96)) * 1024 ** 3
    except Exception:  # noqa: BLE001
        return False


def ensure_family(family: str | Iterable[str], url: str | None = None, wait_own: float = 3600,
                  poll: float = 5) -> dict:
    """Make sure that ComfyUI holds no model family other than `family`. Returns what was done.
    The caller must hold ``gpu_lock(url)`` from this call until its job has ended."""
    url = url or comfyui_url()
    wanted = sorted({family} if isinstance(family, str) else set(family))
    state = _load()
    rec = state.get(url, {})
    owned = set(rec.get("owned", []))
    known = rec.get("families")
    from . import comfy
    shared = shared_engine()
    real_sage = engine_sage(url) if shared else None
    if shared and known and rec.get("shared") and real_sage is not None and _only_us(url, rec, owned):
        new = sorted(set(known) | set(wanted))      # the new family joins the running engine
        if not real_sage or comfy.sage_for(new):
            # an engine that was started with SageAttention (LTX only) must not take a family that Sage breaks
            # (Qwen: black pictures): then one restart without Sage for all of them, below
            if new != known:
                state[url] = {**rec, "families": new}
                _save(state)
            return {"restarted": False, "families": new, "shared": True}
        wanted = new
    elif known == wanted and _only_us(url, rec, owned):
        return {"restarted": False, "families": wanted}
    if not alive(url):
        raise RuntimeError(f"ComfyUI does not answer at {url}")
    # never restart under a running prompt: wait for our own (other runs), refuse for anyone else's
    deadline = time.time() + wait_own
    while True:
        queue = [pid for pid, _ in queue_items(url)]
        foreign = [pid for pid in queue if pid not in owned]
        if foreign:
            raise ServerBusy(f"ComfyUI has {len(foreign)} prompt(s) that cinestudio did not send; "
                             f"let them finish (or stop them) before the model family changes")
        if not queue:
            break
        if time.time() > deadline:
            raise ServerBusy(f"cinestudio prompts are still running after {wait_own:.0f} s; try again later")
        time.sleep(poll)
    comfy.NEXT_FAMILIES = wanted                       # the restart chooses its speed flags for these families
    try:
        seconds = restart(url)
    finally:
        comfy.NEXT_FAMILIES = None
    state = _load()
    after = engine_sage(url)
    state[url] = {"families": wanted, "owned": [], "last_prompt": None, "restarted_at": time.time(),
                  "shared": shared, "sage": after if after is not None else comfy.sage_for(wanted)}
    _save(state)
    reason = "unknown state" if not known else f"{known} -> {wanted}" if known != wanted else "ComfyUI was used by others"
    return {"restarted": True, "seconds": seconds, "families": wanted, "reason": reason}


def _prefixes(graph: dict) -> set[str]:
    return {n["inputs"]["filename_prefix"] for n in (graph or {}).values()
            if isinstance(n, dict) and isinstance(n.get("inputs"), dict)
            and isinstance(n["inputs"].get("filename_prefix"), str)}


def find_prompt(prefix: str, url: str | None = None, look_back: int = 200) -> str | None:
    """A prompt in ComfyUI's own queue or history whose graph has an output node with exactly this
    filename_prefix (for a submit whose answer was lost, also when the proxy has no record of it).
    The proxy job id is the ComfyUI prompt id, so the result can be watched through the proxy."""
    url = url or comfyui_url()
    try:
        for pid, graph in queue_items(url):
            if prefix in _prefixes(graph):
                return pid
        hist = _get(f"{url}/history?max_items={look_back}", timeout=15)
    except Exception:  # noqa: BLE001 - ComfyUI not answering: unknown
        return None
    for pid, entry in reversed(list(hist.items())):
        prompt = entry.get("prompt") or []
        if len(prompt) > 2 and prefix in _prefixes(prompt[2]):
            return pid
    return None


def cancel_prompt(prompt_id: str, url: str | None = None) -> bool:
    """Cancel ONE prompt in ComfyUI (running: interrupted only if it is still the running one;
    pending: removed from the queue). Never a global interrupt. True if a cancel was dispatched."""
    url = url or comfyui_url()
    req = urllib.request.Request(f"{url}/api/jobs/{prompt_id}/cancel", data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return bool(json.loads(r.read() or b"{}").get("cancelled"))
    except Exception:  # noqa: BLE001
        return False
