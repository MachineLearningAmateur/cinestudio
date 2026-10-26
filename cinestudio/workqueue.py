"""One first-in, first-out queue for all GPU work: character steps and video runs.

Every request gets a ticket file, <home>/queue/<seq>-<kind>-<ref>.json, where <seq> is a sequence number
taken under the queue lock (not the clock), so the order is the order of the requests. The worker
process writes its identity (pid + process creation time) into the ticket, waits until no earlier ticket
is live, does the work, and removes the ticket.

All decisions (add, claim, removal of dead tickets, admission) happen under one short queue lock that is
never held while work waits. A ticket is live while its worker process runs (the same pid AND the same
creation time: Windows reuses pids). A ticket without a pid (the worker is still starting) is live for
START_GRACE seconds.

Nothing here stops work: a request leaves the queue early only by an explicit cancel (`cancel.json`
of a run, `<job>.cancel` of a character job), which the worker checks while it waits.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Callable

from .locks import FileLock, LockBusy, pid_alive, process_ctime

START_GRACE = 90.0
KINDS = ("character", "video", "song", "lab", "template")


class QueueCanceled(RuntimeError):
    """The request was canceled while it waited for its turn."""


def qdir() -> Path:
    from .runner import home
    d = home() / "queue"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _qlock() -> FileLock:
    return FileLock(qdir() / "queue.lock", timeout=30)


def _write(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + f".{secrets.token_hex(3)}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find(kind: str, ref: str) -> Path | None:
    return next(iter(qdir().glob(f"*-{kind}-{ref}.json")), None)


def _next_seq() -> int:
    """The next sequence number (call under the queue lock). It starts after every existing ticket."""
    f = qdir() / "seq"
    try:
        n = int(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        nums = [int(m.group(1)) for p in qdir().glob("*.json") if (m := re.match(r"(\d+)-", p.name))]
        n = max(nums, default=0)
    n += 1
    f.write_text(str(n), encoding="utf-8")
    return n


def _add(kind: str, ref: str, label: str, extra: dict) -> Path:
    old = find(kind, ref)
    if old:
        return old
    p = qdir() / f"{_next_seq():020d}-{kind}-{ref}.json"
    _write(p, {"kind": kind, "ref": ref, "label": label, "added": time.time(), "pid": None, **extra})
    return p


def add(kind: str, ref: str, label: str = "", **extra) -> dict:
    """Put a request at the end of the queue (or return its ticket if it has one)."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    with _qlock():
        return _read(_add(kind, ref, label, extra)) or {}


class QueueBusy(ValueError):
    """add_new: the request is in the queue already."""


def add_new(kind: str, ref: str, label: str = "", **extra) -> dict:
    """Like add(), but refuses a ref that already has a ticket - checked and added under ONE lock, so two requests at
    the same moment (a double click) cannot both start a worker for the same ticket."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    with _qlock():
        if find(kind, ref):
            raise QueueBusy(f"{kind} {ref} is in the queue already")
        return _read(_add(kind, ref, label, extra)) or {}


def _claim(kind: str, ref: str, label: str) -> Path:
    """Call under the queue lock: this worker takes its ticket (joins the end if it has none)."""
    p = _add(kind, ref, label, {})
    t = _read(p) or {"kind": kind, "ref": ref, "label": label, "added": time.time()}
    if t.get("pid") != os.getpid():
        t.update(pid=os.getpid(), ctime=process_ctime(os.getpid()))
        _write(p, t)
    return p


def claim(kind: str, ref: str, label: str = "") -> Path:
    """The worker takes its ticket (it writes its identity). A worker started without a ticket (for
    example `run start` from a shell) joins the end of the queue."""
    with _qlock():
        return _claim(kind, ref, label)


def _live(t: dict) -> bool:
    if t.get("pid"):
        if not pid_alive(int(t["pid"])):
            return False
        ct = t.get("ctime")
        now = process_ctime(int(t["pid"])) if ct else None
        return ct is None or now is None or abs(now - float(ct)) < 1.0
    return time.time() - float(t.get("added") or 0) < START_GRACE


def _live_tickets(clean: bool) -> list[dict]:
    out = []
    for p in sorted(qdir().glob("*.json")):
        t = _read(p)
        if t is None:
            continue
        if not _live(t):
            if clean:
                again = _read(p)                       # read again: a worker may have claimed it meanwhile
                if again is not None and not _live(again):
                    p.unlink(missing_ok=True)
                    continue
                if again is None:
                    continue
                t = again
            else:
                continue
        out.append({**t, "ticket": p.name})
    return out


def tickets() -> list[dict]:
    """Live tickets, first in line first. Tickets of dead workers are removed (under the queue lock)."""
    with _qlock():
        return _live_tickets(clean=True)


def ahead_of(kind: str, ref: str) -> list[dict]:
    with _qlock():
        me = find(kind, ref)
        if me is None:
            return []
        return [t for t in _live_tickets(clean=True) if t["ticket"] < me.name]


def wait_turn(kind: str, ref: str, label: str = "", on_wait: Callable[[int], None] | None = None,
              should_stop: Callable[[], bool] | None = None, poll: float = 2.0) -> None:
    """Block until this request is first in line. on_wait(position) is called when the position
    changes (1 = next). Raises QueueCanceled when should_stop() says so. Each check happens under the
    queue lock: the worker's own ticket must still exist and be its own (else it joins again)."""
    last = None
    while True:
        if should_stop and should_stop():
            done(kind, ref)
            raise QueueCanceled(f"{kind} {ref} was canceled while it waited in the queue")
        try:
            with _qlock():
                me = _claim(kind, ref, label)          # also puts back a ticket that was lost
                n = len([t for t in _live_tickets(clean=True) if t["ticket"] < me.name])
        except LockBusy:                               # a slow moment (2026-10-21: low RAM, the lock was held over
            time.sleep(poll)                           # 30 s and two waiting runs died): a waiter only tries again
            continue
        if n == 0:
            return
        if n != last and on_wait:
            on_wait(n)
        last = n
        time.sleep(poll)


def done(kind: str, ref: str) -> None:
    with _qlock():
        p = find(kind, ref)
        if p is not None:
            p.unlink(missing_ok=True)


# ---------------------------------------------------------------- the view for the app, the CLI and agents
def _run_progress(run_id: str) -> dict | None:
    from .runner import run_dir
    p = run_dir(run_id) / "events.jsonl"
    try:
        lines = p.read_bytes().split(b"\n")[:-1]
    except OSError:
        return None
    started = None
    prog = None
    note = None
    step = None
    for ln in reversed(lines):
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if prog is None and e.get("event") == "progress":
            prog = e
        if note is None and e.get("event") == "blender":       # a Blender step sends no progress: say what it does
            note = e.get("note")
        if e.get("event") in ("step_started", "take_started"):     # a retake starts like a step (else the time
            started = e.get("t")                                    # came from an old step: "8 h left")
            step = e.get("step") or (f"retake of clip {e.get('clip')} (take {e.get('take')})"
                                     if e.get("event") == "take_started" else None)
            break
    if prog is None:
        return {"frac": 0, "text": note or step, "step_started": started} if started else None
    return {"frac": (prog.get("value") or 0) / (prog.get("max") or 1), "text": prog.get("step"), "step_started": started}


def view() -> dict:
    """The queue, first in line first: what runs now, what waits, and how many ComfyUI jobs of other
    programs share the GPU."""
    from . import jobs, runner, server
    items = []
    for i, t in enumerate(tickets()):
        kind, ref = t["kind"], t["ref"]
        item = {"kind": kind, "ref": ref, "label": t.get("label") or ref, "position": i, "added": t.get("added")}
        try:
            if kind == "character":
                j = jobs.load(ref)
                p = j.get("progress") or {}
                frac = ((p.get("item") or 1) - 1 + (p.get("value") or 0) / (p.get("max") or 1)) / (p.get("items") or 1) if p else 0
                item.update(status=j["status"], character=j.get("character"), prop=j.get("prop"), action=j.get("action"),
                            started=j.get("started"), frac=frac,
                            text=f"{p['item']} of {p['items']}" if p.get("items", 1) > 1 else None)
            elif kind == "template" and ref.startswith("mask-"):   # a figure selection (cinestudio/figure_mask.py)
                from . import figure_mask
                r = figure_mask.status(ref[len("mask-"):])
                item.update(status=r.get("status"), started=r.get("started"), frac=0, text=r.get("text"))
            elif kind == "template":                   # a template made from a video (cinestudio/video_template.py)
                from . import shots
                r = shots.load(ref)
                p = r.get("progress") or {}
                item.update(status=r.get("status"), started=r.get("started"),
                            frac=((p.get("i") or 1) - 1) / (p.get("n") or 1) if p else 0, text=p.get("text"))
            elif kind == "lab":                        # an experiment: running when it is first in line
                from . import lab
                stop = lab.canceled(ref)                   # Stop pressed: the worker ends after its current step
                item.update(status="running" if i == 0 else "queued",
                            started=(lab.started(ref) or t.get("added")) if i == 0 else t.get("added"),
                            frac=lab.progress(ref) if i == 0 else 0,
                            text="stopping after this step" if stop else None, stopping=stop or None)
            elif kind == "song":
                from . import song
                s = song.load(ref)
                p = s.get("progress") or {}
                item.update(status=s.get("status"), started=s.get("started"),
                            frac=(p.get("value") or 0) / (p.get("max") or 1) if p else 0, text=None)
            else:
                st = runner.status(ref)
                prog = _run_progress(ref) if st["status"] == "running" else None
                item.update(status=st["status"], started=(st.get("worker") or {}).get("started"),
                            frac=prog["frac"] if prog else 0, text=prog["text"] if prog else None,
                            step_started=prog["step_started"] if prog else None)
                if st["status"] == "running" and st.get("cancel_requested"):
                    item.update(text="stopping after this step", stopping=True)
        except Exception as exc:  # noqa: BLE001 - one unreadable record must not hide the queue
            item.update(status="unknown", error=str(exc))
        items.append(item)
    others = None
    try:
        owned = server.owned_ids()
        others = sum(1 for pid, _ in server.queue_items(server.comfyui_url()) if pid not in owned)
    except Exception:  # noqa: BLE001 - ComfyUI down: no count
        pass
    return {"items": items, "other_comfy_jobs": others, "now": time.time()}


def cancel(kind: str, ref: str) -> dict:
    """Explicit cancel of one request (queued or running)."""
    if kind == "character":
        from . import jobs
        return jobs.cancel(ref)
    if kind == "video":
        from . import runner
        return runner.request_cancel(ref)
    if kind == "song":
        from . import song
        return song.request_cancel(ref)
    if kind == "lab":
        from . import lab
        return lab.request_cancel(ref)
    if kind == "template" and ref.startswith("mask-"):
        from . import figure_mask
        return figure_mask.request_cancel(ref[len("mask-"):])
    if kind == "template":
        from . import video_template
        return video_template.request_cancel(ref)
    raise ValueError(f"unknown kind {kind!r}")
