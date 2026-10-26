"""Background jobs for character steps (create, edit, undo, views, style), for the web app and agents.

Every job is a JSON file in <home>/jobs/<id>.json with its status and events. Every job also has a
ticket in the shared work queue (cinestudio.workqueue), so character steps and video runs take turns
in the order of the requests.

App jobs run in their own background process (`python -m cinestudio job work ID`), not in the app
process: a restart or a closed app window does not stop them. CLI jobs run in the CLI process, after
they wait for their turn. A job stops early only by an explicit `cancel` (a <id>.cancel file).
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from . import character, props, spawn, workqueue
from . import models as _models
from .locks import FileLock, LockBusy, pid_alive

ROOT = Path(__file__).resolve().parent.parent
ACTIONS: dict[str, Callable[..., dict]] = {
    "create": character.create,
    "edit": character.edit_image,
    "redo_view": character.redo_view,
    "undo": character.undo_edit,
    "views": character.make_views,
    "repose": character.repose,
    "style": character.style_pass,
    "pick": character.pick,
    "finish": character.finish,
    "new_skin": character.new_skin,
    **props.ACTIONS,                                     # prop_create, prop_edit, prop_skin (cinestudio/props.py)
}
LONG = {"create", "edit", "redo_view", "views", "style", "repose", "new_skin", *props.ACTIONS}   # the others are instant
PROP_ACTIONS = set(props.ACTIONS)                        # a prop's job: its ID in "prop" (not "character")
ACTIVE = ("queued", "running")


class JobCanceled(character.StepCanceled):
    """The user canceled the job."""


def jobs_dir() -> Path:
    from .runner import home
    d = home() / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(job_id: str) -> Path:
    if not job_id.replace("-", "").isalnum():
        raise KeyError(job_id)
    return jobs_dir() / f"{job_id}.json"


def _write(job: dict) -> None:
    p = jobs_dir() / f"{job['id']}.json"
    tmp = p.with_name(p.name + f".{secrets.token_hex(3)}.tmp")
    tmp.write_text(json.dumps(job, indent=1), encoding="utf-8")
    tmp.replace(p)


def _read(job_id: str) -> dict:
    return json.loads(_path(job_id).read_text(encoding="utf-8"))


def load(job_id: str) -> dict:
    j = _read(job_id)
    j["status"] = live_status(j)
    return j


def subject(job: dict) -> str | None:
    """The ID a job works on: a character's, or a prop's (prop jobs)."""
    return job.get("character") or job.get("prop")


def label(job: dict) -> str:
    """For example "kitsune · edit left" or "kitsune · redo_view back"."""
    args = job.get("args") or {}
    what = args.get("target") or args.get("view")
    return f"{subject(job)} · {job.get('action')}" + (f" {what}" if what else "")


def list_jobs(character_name: str | None = None, limit: int = 30, prop_name: str | None = None) -> list[dict]:
    """The newest jobs: all, one character's (`character_name`) or one prop's (`prop_name`)."""
    out = []
    for p in sorted(jobs_dir().glob("j*.json"), reverse=True):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if prop_name is not None:
            hit = j.get("prop") == prop_name
        else:
            hit = character_name is None or j.get("character") == character_name
        if hit:
            row = {k: j.get(k) for k in ("id", "action", "character", "status", "created", "ended", "error")}
            if j.get("prop"):
                row["prop"] = j["prop"]
            row["status"] = live_status(j)
            row["via"] = (j.get("owner") or {}).get("via")
            out.append(row)
        if len(out) >= limit:
            break
    return out


def _plain(v: Any) -> Any:
    return str(v) if isinstance(v, Path) else v


def _new_job(action: str, name: str, kwargs: dict, via: str) -> dict:
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    args = {k: _plain(v) for k, v in kwargs.items()}
    who = {"character": None, "prop": name} if action in PROP_ACTIONS else {"character": name}
    return {"id": time.strftime("j%Y%m%d-%H%M%S-") + secrets.token_hex(2), "action": action, **who,
            "args": {k: v for k, v in args.items() if isinstance(v, (str, int, float, bool, type(None)))
                     or (isinstance(v, dict) and all(isinstance(x, (str, int, float, bool, type(None))) for x in v.values()))},
            "status": "queued", "created": time.time(), "events": [],
            "owner": {"pid": None, "via": via}, "model_tier": _models.job_tier()}


def cancel_requested(job_id: str) -> bool:
    return (jobs_dir() / f"{job_id}.cancel").is_file()


def execute(job: dict, kwargs: dict, echo: Callable[[dict], None] | None = None) -> dict:
    """Run one job in this process and keep its record up to date (latest progress at most every 2 s).
    `echo` also receives every event (the CLI prints them). A cancel request stops the job at its
    next event (the ComfyUI job itself is canceled by `cancel`)."""
    _models.use_tier(job.get("model_tier"))           # the models of the job's tier, frozen when it was made
    job.update(status="running", started=time.time(), owner={**job.get("owner", {}), "pid": os.getpid()})
    _write(job)
    last_write = [0.0]

    def on_event(ev: dict) -> None:
        if cancel_requested(job["id"]):
            raise JobCanceled("canceled by the user")
        ev = {"t": round(time.time(), 1), **ev}
        if echo:
            echo(ev)
        if ev.get("event") == "progress":
            job["progress"] = ev
            if time.time() - last_write[0] < 2.0:
                return
        else:
            if ev.get("event") == "comfy_job":
                job["comfy_job"] = ev.get("job")
            job["events"].append(ev)
        last_write[0] = time.time()
        _write(job)
    try:
        fn = ACTIONS[job["action"]]
        if job["action"] in LONG:
            kwargs = dict(kwargs, on_event=on_event)
        lock = props.prop_lock if job["action"] in PROP_ACTIONS else character.char_lock
        with lock(subject(job), timeout=600):           # waits for a short change (a pick) to end
            job.update(status="done", result=fn(subject(job), **kwargs))
    except Exception as exc:  # noqa: BLE001 - recorded in the job
        if isinstance(exc, JobCanceled) or cancel_requested(job["id"]):
            job.update(status="canceled", error="canceled by the user")
        else:
            job.update(status="failed", error=str(exc) or repr(exc), error_type=type(exc).__name__,
                       trace=traceback.format_exc(limit=4))
    job["ended"] = time.time()
    _write(job)
    return job


def _queued_then_execute(job: dict, kwargs: dict, echo: Callable[[dict], None] | None = None) -> dict:
    """Wait for the job's turn in the shared queue, then run it. The ticket is removed at the end.
    An INSTANT action (not in LONG: undo, pick, finish, rename ...) uses no GPU: it runs at once, without a queue
    ticket (2026-10-22, user: "why does the undo have to wait in the gpu queue?" - an undo waited behind a panorama).
    The character lock in execute() still keeps changes to one character in order."""
    job["owner"] = {**job.get("owner", {}), "pid": os.getpid()}
    _write(job)
    if job["action"] not in LONG:
        workqueue.done("character", job["id"])          # a ticket added before this rule (or by an old caller)
        return execute(job, kwargs, echo)

    def waiting(position: int) -> None:
        job["queue_position"] = position
        _write(job)
        if echo:
            echo({"event": "queued", "position": position})
    try:
        workqueue.wait_turn("character", job["id"], label(job), on_wait=waiting,
                            should_stop=lambda: cancel_requested(job["id"]))
    except workqueue.QueueCanceled:
        job.update(status="canceled", error="canceled by the user before it started", ended=time.time())
        _write(job)
        return job
    try:
        job.pop("queue_position", None)
        return execute(job, kwargs, echo)
    finally:
        workqueue.done("character", job["id"])


def _job_lock(job_id: str) -> FileLock:
    """Held by the one process that runs a job, from before it reads the status to the end."""
    return FileLock(jobs_dir() / f"{job_id}.lock", timeout=0)


def run_now(action: str, name: str, echo: Callable[[dict], None] | None = None, **kwargs: Any) -> dict:
    """Run a job in this process (the CLI), after its turn in the queue. The record is the same as
    for app jobs, so the app shows it."""
    job = _new_job(action, name, kwargs, "cli")
    with _job_lock(job["id"]):
        _write(job)
        if action in LONG:
            workqueue.add("character", job["id"], label(job))
        return _queued_then_execute(job, kwargs, echo)


def work(job_id: str) -> dict:
    """The body of a background worker (`job work ID`): run a queued job from its record."""
    lock = _job_lock(job_id)
    try:
        lock.acquire()
    except LockBusy:
        return {**_read(job_id), "busy": "another worker runs this job"}
    try:
        job = _read(job_id)                            # read under the lock: one worker per job, ever
        if job.get("status") != "queued":
            return job
        return _queued_then_execute(job, dict(job.get("args", {})))
    finally:
        lock.release()


def submit(action: str, name: str, via: str = "app", **kwargs: Any) -> dict:
    """Queue a job and start its background worker; return at once with the job record."""
    job = _new_job(action, name, kwargs, via)
    _write(job)
    if action in LONG:                                   # instant actions take no queue ticket
        workqueue.add("character", job["id"], label(job))
    try:
        # an independent process (see cinestudio.spawn): stopping the app does not stop the job
        _, independent = spawn.detached([sys.executable, "-m", "cinestudio", "job", "work", job["id"]],
                                        jobs_dir() / f"{job['id']}.log", ROOT)
        if not independent:
            job["warning"] = ("Windows did not let the worker leave the app's process group: closing the app "
                              "window or terminal may stop this job")
    except (OSError, subprocess.SubprocessError) as exc:
        workqueue.done("character", job["id"])
        job.update(status="failed", error=f"the worker did not start: {exc}", ended=time.time())
        _write(job)
    return job


def cancel(job_id: str) -> dict:
    """Explicit cancel. A queued job leaves the queue; a running job stops, and its ComfyUI job is
    canceled (job-scoped: other ComfyUI work is not touched)."""
    j = _read(job_id)                                  # the recorded status: a slow-starting worker still obeys
    if j.get("status") not in ACTIVE:
        return {"id": job_id, "status": j.get("status"), "canceled": False}
    (jobs_dir() / f"{job_id}.cancel").write_text(json.dumps({"requested_at": time.time()}), encoding="utf-8")
    out = {"id": job_id, "status": "cancel_requested", "canceled": True}
    if j["status"] == "running" and j.get("comfy_job"):
        from . import server
        try:
            from .backend import SdkBackend
            SdkBackend().cancel(j["comfy_job"])
        except Exception:  # noqa: BLE001 - not in the proxy: try ComfyUI's own queue
            server.cancel_prompt(j["comfy_job"])
        out["comfy_job"] = j["comfy_job"]
    return out


def live_status(job: dict) -> str:
    """The status, with 'interrupted' for a queued or running job whose worker process is gone."""
    st = job.get("status")
    if st not in ACTIVE:
        return st
    pid = (job.get("owner") or {}).get("pid")
    if pid:
        return st if pid_alive(int(pid)) else "interrupted"
    # no worker pid yet (it is starting), or an older record without one: live while it changes
    times = [job.get("created") or 0, job.get("started") or 0, (job.get("progress") or {}).get("t") or 0,
             *(e.get("t") or 0 for e in job.get("events", []))]
    grace = workqueue.START_GRACE if job.get("owner") else 600
    return st if time.time() - max(times) < grace else "interrupted"


class JobRunner:
    """The app's handle on jobs. Jobs run in their own processes, so this keeps no state of its own."""

    def __init__(self) -> None:
        for p in jobs_dir().glob("j*.json"):            # jobs whose worker is gone (crash, power off)
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            # only a job whose recorded worker is dead (a job still waiting for its worker is left alone:
            # a slow start must not be turned into "interrupted" for ever)
            pid = (j.get("owner") or {}).get("pid")
            if j.get("status") in ACTIVE and pid and not pid_alive(int(pid)):
                j.update(status="interrupted", error="the process that ran this job stopped")
                _write(j)

    def submit(self, action: str, name: str, **kwargs: Any) -> dict:
        return submit(action, name, "app", **kwargs)
