"""The shared work queue: order of requests, live tickets, explicit cancel only."""

import json
import os
import time

import pytest

from cinestudio import jobs, workqueue


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(jobs.spawn, "detached", lambda *a, **k: (0, True))
    return tmp_path


def test_order_is_the_order_of_requests(home):
    workqueue.add("character", "j1", "a")
    workqueue.add("video", "r1", "b")
    workqueue.add("character", "j2", "c")
    assert [t["ref"] for t in workqueue.tickets()] == ["j1", "r1", "j2"]
    assert [t["ref"] for t in workqueue.ahead_of("character", "j2")] == ["j1", "r1"]
    assert workqueue.add("video", "r1")["label"] == "b"           # a second add keeps the first place


def test_dead_worker_tickets_are_dropped(home):
    workqueue.add("character", "j1")
    p = workqueue.find("character", "j1")
    t = json.loads(p.read_text())
    t["pid"] = 999999                                              # no such process
    p.write_text(json.dumps(t))
    workqueue.add("character", "j2")
    assert [t["ref"] for t in workqueue.tickets()] == ["j2"]


def test_ticket_without_pid_expires(home, monkeypatch):
    workqueue.add("character", "j1")
    assert workqueue.tickets()
    monkeypatch.setattr(workqueue.time, "time", lambda: time.time_ns() / 1e9 + workqueue.START_GRACE + 5)
    assert workqueue.tickets() == []


def test_wait_turn_waits_for_earlier_live_tickets(home):
    workqueue.add("video", "r1")
    workqueue.claim("video", "r1")                                 # held by this (live) process
    workqueue.add("character", "j1")
    seen = []
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        if calls["n"] == 2:
            workqueue.done("video", "r1")                          # the run before it ends
        return False
    workqueue.wait_turn("character", "j1", on_wait=seen.append, should_stop=stop, poll=0.01)
    assert seen == [1]


def test_wait_turn_can_be_canceled(home):
    workqueue.add("video", "r1")
    workqueue.claim("video", "r1")
    workqueue.add("character", "j1")
    with pytest.raises(workqueue.QueueCanceled):
        workqueue.wait_turn("character", "j1", should_stop=lambda: True, poll=0.01)
    assert workqueue.find("character", "j1") is None


def test_submitted_job_is_queued_and_cancel_is_explicit(home):
    j = jobs.submit("views", "fox")
    assert j["status"] == "queued" and workqueue.find("character", j["id"])
    assert jobs.load(j["id"])["status"] == "queued"                # the worker is starting: still live
    v = workqueue.view()
    assert v["items"][0]["ref"] == j["id"] and v["items"][0]["label"] == "fox · views"
    r = workqueue.cancel("character", j["id"])
    assert r["canceled"] and jobs.cancel_requested(j["id"])
    # the worker sees the request before its turn and leaves the queue
    job = jobs.work(j["id"])
    assert job["status"] == "canceled" and workqueue.find("character", j["id"]) is None


def test_work_runs_a_queued_job(home, monkeypatch):
    monkeypatch.setitem(jobs.ACTIONS, "finish", lambda name, **kw: {"name": name, "status": "ready"})
    j = jobs.submit("finish", "fox")
    job = jobs.work(j["id"])
    assert job["status"] == "done" and job["owner"]["pid"] == os.getpid()
    assert workqueue.find("character", j["id"]) is None


def test_order_uses_sequence_numbers_not_the_clock(home, monkeypatch):
    workqueue.add("character", "j1")
    monkeypatch.setattr(workqueue.time, "time_ns", lambda: 1)                 # a clock that went back
    workqueue.add("character", "j2")
    names = sorted(p.name for p in workqueue.qdir().glob("*.json"))
    assert names[0].endswith("-character-j1.json") and names[1].endswith("-character-j2.json")


def test_a_lost_ticket_is_put_back_at_the_end(home):
    workqueue.add("video", "r1")
    workqueue.claim("video", "r1")
    workqueue.add("character", "j1")
    workqueue.claim("character", "j1")
    workqueue.find("character", "j1").unlink()                                  # lost
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        if calls["n"] == 2:
            workqueue.done("video", "r1")
        return False
    workqueue.wait_turn("character", "j1", should_stop=stop, poll=0.01)
    assert workqueue.find("character", "j1") is not None                         # it waited with a ticket


def test_a_second_worker_for_one_job_is_refused(home):
    j = jobs.submit("finish", "fox")
    lock = jobs._job_lock(j["id"])
    lock.acquire()                                                              # the first worker
    try:
        import threading
        out = {}
        t = threading.Thread(target=lambda: out.update(r=jobs.work(j["id"])))
        t.start(); t.join()
        assert out["r"].get("busy")
    finally:
        lock.release()


def test_cancel_works_before_the_worker_starts(home, monkeypatch):
    j = jobs.submit("views", "fox")
    monkeypatch.setattr(workqueue, "START_GRACE", 0)                            # it looks stale now
    assert jobs.cancel(j["id"])["canceled"]


def test_pick_is_refused_while_another_change_runs(home):
    import threading
    from cinestudio import character
    character._new("fox", "a fox", "", "text")
    started, release = threading.Event(), threading.Event()

    def long_change():
        with character.char_lock("fox"):
            started.set()
            release.wait(5)
    t = threading.Thread(target=long_change)
    t.start()
    started.wait(5)
    try:
        with pytest.raises(character.CharacterError, match="busy"):
            character.pick("fox", 1)
    finally:
        release.set()
        t.join()


def test_a_waiter_survives_a_busy_queue_lock(tmp_path, monkeypatch):
    from cinestudio import workqueue
    from cinestudio.locks import LockBusy
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    real = workqueue._qlock
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 2:                      # the first check inside wait_turn meets a busy lock
            raise LockBusy(tmp_path / "queue.lock", None)
        return real()
    monkeypatch.setattr(workqueue, "_qlock", flaky)
    workqueue.claim("video", "r1", "first")
    workqueue.wait_turn("video", "r1", "first", poll=0.01)   # must not raise
    assert calls["n"] >= 3


def test_run_progress_counts_from_the_take_start(tmp_path, monkeypatch):
    """A retake writes take_started, not step_started: the time must come from it, not from an old step ("8 h left")."""
    import json
    from cinestudio import runner, workqueue
    monkeypatch.setattr(runner, "run_dir", lambda rid: tmp_path)
    ev = [{"t": 100.0, "event": "step_started", "step": "final"},
          {"t": 9000.0, "event": "take_started", "clip": 3, "take": 2},
          {"t": 9100.0, "event": "progress", "step": "take3_2", "value": 4, "max": 8}]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    p = workqueue._run_progress("r1")
    assert p["step_started"] == 9000.0 and p["frac"] == 0.5 and p["text"] == "take3_2"
    assert p["progress_at"] == 9100.0


def test_unresolved_hold_survives_ticket_cleanup_and_blocks_admission(home, monkeypatch):
    import threading
    workqueue.hold("song", "ambiguous", prefix="cinestudio/ambiguous", job="p1")
    workqueue.add("character", "next")
    stopped = threading.Event()
    seen = []
    monkeypatch.setattr(workqueue, "resolve_holds", lambda: None)
    monkeypatch.setattr(workqueue.time, "sleep", lambda _: stopped.wait(0.01))

    def wait():
        try:
            workqueue.wait_turn("character", "next", on_wait=seen.append, should_stop=stopped.is_set, poll=0.01)
        except workqueue.QueueCanceled:
            pass
    thread = threading.Thread(target=wait)
    thread.start()
    try:
        assert stopped.wait(0.05) is False
        assert thread.is_alive() and seen == [1]
        assert workqueue.holds()[0]["job"] == "p1"
    finally:
        stopped.set()
        thread.join(2)
    assert not thread.is_alive()


def test_hold_is_removed_only_after_exact_job_is_terminal(home, monkeypatch):
    from cinestudio import server
    from cinestudio import backend
    workqueue.hold("song", "uncertain", prefix="cinestudio/uncertain", job="p1")
    monkeypatch.setattr(server, "prompt_status", lambda *_: "queued")
    monkeypatch.setattr(backend, "SdkBackend", lambda: type("B", (), {"state": lambda self, _: backend.JobState("p1", "running")})())
    workqueue.resolve_holds()
    assert workqueue.holds()
    # A stale proxy terminal state must not override ComfyUI reporting the job still queued.
    monkeypatch.setattr(backend, "SdkBackend", lambda: type("B", (), {"state": lambda self, _: backend.JobState("p1", "succeeded")})())
    workqueue.resolve_holds()
    assert workqueue.holds()
    monkeypatch.setattr(server, "prompt_status", lambda *_: "done_ok")
    workqueue.resolve_holds()
    assert workqueue.holds() == []


def test_unresolved_hold_can_be_released_only_after_checked_empty_services(home, monkeypatch):
    from types import SimpleNamespace
    from cinestudio import server, backend
    workqueue.hold("song", "uncertain", prefix="cinestudio/uncertain", job=None)
    with pytest.raises(ValueError, match="confirm"):
        workqueue.release_hold("song", "uncertain")
    monkeypatch.setattr(server, "alive", lambda: True)
    monkeypatch.setattr(server, "queue_items", lambda _: [("active", {})])
    with pytest.raises(ValueError, match="empty queue"):
        workqueue.release_hold("song", "uncertain", confirm=True)
    monkeypatch.setattr(server, "queue_items", lambda _: [])
    monkeypatch.setattr(backend, "SdkBackend", lambda: SimpleNamespace(client=SimpleNamespace(list_jobs=lambda **_: [])))
    assert workqueue.release_hold("song", "uncertain", confirm=True)["released"]
    assert workqueue.holds() == []


def test_unconfirmed_submission_can_be_found_and_resolved_later(home, monkeypatch):
    from cinestudio import server
    workqueue.hold("song", "uncertain", prefix="cinestudio/uncertain", job=None)
    monkeypatch.setattr(server, "find_prompt", lambda prefix, url: "p1")
    monkeypatch.setattr(server, "prompt_status", lambda url, jid: "done_ok")
    workqueue.resolve_holds()
    assert workqueue.holds() == []


def test_old_progress_is_an_informational_notice_not_a_cancel(home, monkeypatch):
    from cinestudio import lab, server
    monkeypatch.setattr(workqueue, "tickets", lambda: [{"kind": "lab", "ref": "slow", "added": 1}])
    monkeypatch.setattr(lab, "started", lambda _: workqueue.time.time() - 700)
    monkeypatch.setattr(lab, "progress", lambda _: 0.5)
    monkeypatch.setattr(server, "queue_items", lambda _: [])
    item = workqueue.view()["items"][0]
    assert item["status"] == "running" and item["no_recent_progress"] is True
    assert not lab.canceled("slow")
