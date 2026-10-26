"""Experiments (gauntlets) that use the GPU: they wait for their turn in the shared queue like every other job.

    with lab.turn("tpl-r1", "templates round 1"):
        character.image_job(...)        # each job still takes the GPU lock itself

The queue shows them as kind "lab". `queue cancel lab REF` (or the app's stop button) writes a flag that the
experiment checks between its jobs (`lab.canceled(ref)`); a running ComfyUI job is not interrupted.
"""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

from . import workqueue


def lab_dir() -> Path:
    from .runner import home
    d = home() / "lab"
    d.mkdir(parents=True, exist_ok=True)
    return d


_CURRENT: list[str | None] = [None]       # the lab ticket this process works on now (lab.turn)


def note_progress(value, total) -> None:
    """Record the sampling progress of the current lab job (character._watch_image calls it), so the queue shows a
    moving bar for experiments and background jobs (before: always 0%)."""
    ref = _CURRENT[0]
    if not ref or not total:
        return
    try:
        import json
        now, t0 = time.time(), None
        try:                                       # the CURRENT job's start: a turn can run several jobs one after
            old = json.loads((lab_dir() / f"{ref}.progress.json").read_text(encoding="utf-8"))   # the other, each with
            if int(old.get("value", 0)) <= int(value or 0) and int(old.get("max", 0)) == int(total):   # its own bar
                t0 = old.get("t0")
        except (OSError, ValueError, TypeError):
            pass
        tmp = lab_dir() / f".{ref}.progress.tmp"
        tmp.write_text(json.dumps({"value": int(value or 0), "max": int(total), "t": now, "t0": t0 or now}),
                       encoding="utf-8")
        tmp.replace(lab_dir() / f"{ref}.progress.json")
    except OSError:
        pass


def progress(ref: str) -> float:
    """The current lab job's fraction 0..1 (0 when unknown)."""
    try:
        import json
        p = json.loads((lab_dir() / f"{ref}.progress.json").read_text(encoding="utf-8"))
        return max(0.0, min(1.0, p["value"] / p["max"]))
    except (OSError, ValueError, KeyError, ZeroDivisionError):
        return 0.0


def started(ref: str) -> float | None:
    """When the lab job `ref` got its turn (the queue's elapsed time and "left" estimate). Before this, the queue used
    the time the ticket was TAKEN, so a job that waited 10 min behind another one showed "1 h left" at 22%. Fallback
    for workers that started with older code: the GPU lock's holder record, when it names this job."""
    try:                                           # the current job of the turn (its bar restarts at 0 per job)
        return float(json.loads((lab_dir() / f"{ref}.progress.json").read_text(encoding="utf-8"))["t0"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        return float((lab_dir() / f"{ref}.started").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    try:
        from . import server
        who = server.gpu_lock().holder() or {}
        if ref in (who.get("character"), who.get("lab")) and who.get("since"):
            return time.mktime(time.strptime(who["since"], "%Y-%m-%dT%H:%M:%S"))
    except Exception:  # noqa: BLE001 - only an estimate
        pass
    return None


def canceled(ref: str) -> bool:
    return (lab_dir() / f"{ref}.cancel").is_file()


def request_cancel(ref: str) -> dict:
    (lab_dir() / f"{ref}.cancel").write_text(str(time.time()), encoding="utf-8")
    return {"lab": ref, "cancel_requested": True}


@contextlib.contextmanager
def turn(ref: str, label: str, on_wait=None):
    (lab_dir() / f"{ref}.cancel").unlink(missing_ok=True)
    workqueue.claim("lab", ref, label)
    try:
        workqueue.wait_turn("lab", ref, label, on_wait=on_wait, should_stop=lambda: canceled(ref))
        try:
            (lab_dir() / f"{ref}.started").write_text(str(time.time()), encoding="utf-8")
        except OSError:
            pass
        _CURRENT[0] = ref
        yield
    finally:
        _CURRENT[0] = None
        (lab_dir() / f"{ref}.progress.json").unlink(missing_ok=True)
        (lab_dir() / f"{ref}.started").unlink(missing_ok=True)
        workqueue.done("lab", ref)
