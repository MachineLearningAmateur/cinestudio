"""One model family per ComfyUI process (fake ComfyUI, no real restart)."""

import pytest

from cinestudio import server as s

URL = "http://comfy:8188"


class FakeComfy:
    def __init__(self):
        self.history, self.queue, self.restarts, self.up = [], [], 0, True
        self.argv = ["main.py"]                    # the engine's start flags (/system_stats)

    def get(self, url, timeout=5):
        if not self.up:
            raise ConnectionRefusedError("down")
        if url.endswith("/system_stats"):
            return {"system": {"argv": list(self.argv)}}
        if "/history" in url:
            n = int(url.split("max_items=")[1])
            return {i: {} for i in self.history[-n:]}
        if url.endswith("/queue"):
            return {"queue_running": [[0, pid, {}] for pid in self.queue[:1]],
                    "queue_pending": [[0, pid, {}] for pid in self.queue[1:]]}
        raise AssertionError(url)

    def run_job(self, pid):
        self.history.append(pid)


@pytest.fixture
def comfy(tmp_path, monkeypatch):
    c = FakeComfy()
    monkeypatch.setattr(s, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(s, "_get", c.get)

    def fake_restart(url, timeout=240):
        from cinestudio import comfy as cf
        c.restarts += 1
        c.history.clear()          # a new process has an empty history
        c.argv = ["main.py"] + (["--use-sage-attention"] if cf.sage_for(cf.NEXT_FAMILIES or []) else [])
        return 1.0
    monkeypatch.setattr(s, "restart", fake_restart)
    return c


def our_job(c, pid):
    s.note_owned(pid, URL)
    c.run_job(pid)


def test_first_job_restarts_then_same_family_does_not(comfy):
    assert s.ensure_family("ltx23", URL)["restarted"]
    assert not s.ensure_family("ltx23", URL)["restarted"]          # fresh process, empty history
    our_job(comfy, "p1")
    assert not s.ensure_family("ltx23", URL)["restarted"]
    assert comfy.restarts == 1


def test_family_change_restarts(comfy):
    s.ensure_family("ltx23", URL)
    our_job(comfy, "p1")
    r = s.ensure_family("qwen_edit", URL)
    assert r["restarted"] and "ltx23" in r["reason"]


def test_someone_else_ran_after_us(comfy):
    """Our old prompt is still in the history, but a newer prompt is not ours: the family is unknown."""
    s.ensure_family("ltx23", URL)
    our_job(comfy, "p1")
    comfy.run_job("user-wan-job")
    r = s.ensure_family("ltx23", URL)
    assert r["restarted"] and r["reason"] == "ComfyUI was used by others"


def test_outside_restart_is_detected(comfy):
    s.ensure_family("ltx23", URL)
    our_job(comfy, "p1")
    comfy.history.clear()          # restarted by someone else: no restarted_at of ours for this process
    st = s._load(); st[URL]["restarted_at"] = None; s._save(st)
    assert s.ensure_family("ltx23", URL)["restarted"]


def test_foreign_work_blocks_restart(comfy):
    comfy.queue = ["user-prompt"]
    with pytest.raises(s.ServerBusy):
        s.ensure_family("ltx23", URL)
    assert comfy.restarts == 0


def test_prefix_is_not_ownership(comfy):
    """A user prompt that uses our output prefix is still the user's prompt."""
    comfy.queue = ["user-ran-our-graph"]
    with pytest.raises(s.ServerBusy):
        s.ensure_family("ltx23", URL)


def test_own_running_job_is_waited_for(comfy, monkeypatch):
    s.ensure_family("ltx23", URL)
    s.note_owned("other-run", URL)
    comfy.queue = ["other-run"]
    ticks = []

    def sleep(_):
        ticks.append(1)
        comfy.queue = []           # the other run's job ends
    monkeypatch.setattr(s.time, "sleep", sleep)
    r = s.ensure_family("qwen_edit", URL, poll=0)
    assert r["restarted"] and ticks                                 # waited, then restarted


def test_same_family_with_own_job_queued_does_not_restart(comfy):
    s.ensure_family("ltx23", URL)
    our_job(comfy, "p1")
    s.note_owned("p2", URL)
    comfy.queue = ["p2"]
    assert not s.ensure_family("ltx23", URL)["restarted"]


def test_restart_none_asks_the_user(monkeypatch):
    monkeypatch.setenv("CINESTUDIO_RESTART", "none")
    with pytest.raises(s.RestartNeeded):
        s.restart(URL)


def test_restart_must_be_confirmed(monkeypatch):
    """Manager says OK, but ComfyUI never goes down and keeps its history: not a new process."""
    from cinestudio import comfy as engine
    monkeypatch.setattr(engine, "owns_engine", lambda: False)
    monkeypatch.setenv("CINESTUDIO_RESTART", "manager-shared")         # the ComfyUI-Manager path
    monkeypatch.setattr(s, "_post", lambda url, timeout=10: 200)
    monkeypatch.setattr(s, "alive", lambda url=None: True)
    monkeypatch.setattr(s, "history_ids", lambda url, n=200: ["old"])
    clock = iter(range(0, 10_000, 30))
    monkeypatch.setattr(s.time, "time", lambda: next(clock))
    monkeypatch.setattr(s.time, "sleep", lambda _: None)
    with pytest.raises(s.RestartFailed):
        s.restart(URL)


def test_gpu_lock_is_shared_between_homes():
    assert s.gpu_lock("http://a:1").path == s.gpu_lock("http://a:1").path
    assert s.gpu_lock("http://a:1").path != s.gpu_lock("http://b:2").path


def test_a_comfyui_started_elsewhere_is_never_restarted(monkeypatch):
    from cinestudio import comfy as engine
    monkeypatch.setattr(engine, "owns_engine", lambda: False)
    monkeypatch.setattr(s, "_post", lambda url, timeout=10: pytest.fail("must not reboot"))
    with pytest.raises(s.RestartNeeded):
        s.restart(URL)


def test_only_us_needs_every_history_entry(monkeypatch):
    monkeypatch.setattr(s, "queue_items", lambda url: [])
    monkeypatch.setattr(s, "history_ids", lambda url, n=200: ["a", "foreign", "b"])
    assert not s._only_us(URL, {}, {"a", "b"})
    monkeypatch.setattr(s, "history_ids", lambda url, n=200: ["a", "b"])
    assert s._only_us(URL, {}, {"a", "b"})


def test_own_engine_is_restarted_by_cinestudio(monkeypatch):
    from cinestudio import comfy as engine
    monkeypatch.setattr(engine, "owns_engine", lambda: True)
    monkeypatch.setattr(engine, "restart_own", lambda timeout: 12.5)
    monkeypatch.setattr(s, "_post", lambda url, timeout=10: pytest.fail("ComfyUI-Manager must not be used"))
    assert s.restart(URL) == 12.5


def test_a_shared_engine_lets_a_new_family_join_without_a_restart(comfy, monkeypatch):
    """The large tier on a machine with enough RAM: Qwen (look images), then LTX (clips), then ACE-Step (a song) stay
    in ONE engine (2026-10-24: on a pod every family change reloaded tens of GB from the network volume)."""
    from cinestudio import comfy as cf
    monkeypatch.setattr(s, "shared_engine", lambda: True)
    monkeypatch.setattr(cf, "sage_for", lambda fams, cfg=None: bool(fams) and all(f.startswith("ltx25") for f in fams))
    assert s.ensure_family("qwen_edit@a", URL)["restarted"]           # the first start
    our_job(comfy, "p1")
    r = s.ensure_family("ltx25@b", URL)
    assert not r["restarted"] and r["families"] == ["ltx25@b", "qwen_edit@a"] and r["shared"]
    our_job(comfy, "p2")
    assert not s.ensure_family("acestep@c", URL)["restarted"]
    assert not s.ensure_family("qwen_edit@a", URL)["restarted"]       # already in it
    assert comfy.restarts == 1


def test_a_sage_engine_restarts_once_for_a_family_that_sage_breaks(comfy, monkeypatch):
    """An engine started for LTX alone has SageAttention (a start flag for the whole engine); Qwen drew black pictures
    with it. So Qwen makes ONE restart into an engine without Sage for both; after that they share."""
    from cinestudio import comfy as cf
    monkeypatch.setattr(s, "shared_engine", lambda: True)
    monkeypatch.setattr(cf, "sage_for", lambda fams, cfg=None: bool(fams) and all(f.startswith("ltx25") for f in fams))
    s.ensure_family("ltx25@b", URL)
    assert comfy.argv == ["main.py", "--use-sage-attention"]          # LTX alone: the engine runs with Sage
    our_job(comfy, "p1")
    r = s.ensure_family("qwen_edit@a", URL)
    assert r["restarted"] and r["families"] == ["ltx25@b", "qwen_edit@a"]
    our_job(comfy, "p2")
    assert not s.ensure_family("ltx25@b", URL)["restarted"] and comfy.restarts == 2


def test_the_8gb_tier_keeps_one_family_per_engine(comfy, monkeypatch):
    monkeypatch.setattr(s, "shared_engine", lambda: False)
    s.ensure_family("qwen_edit@a", URL)
    our_job(comfy, "p1")
    assert s.ensure_family("ltx25@b", URL)["restarted"] and comfy.restarts == 2


def test_shared_engine_needs_the_tier_and_the_ram(monkeypatch):
    import psutil
    from cinestudio import models
    monkeypatch.setattr(models, "job_tier", lambda: "large")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("M", (), {"total": 188 * 1024 ** 3})())
    assert s.shared_engine()
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("M", (), {"total": 125 * 1024 ** 3})())
    assert not s.shared_engine()                                      # an H100 SXM pod: one family per engine
    monkeypatch.setattr(models, "job_tier", lambda: "8gb")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("M", (), {"total": 512 * 1024 ** 3})())
    assert not s.shared_engine()


def test_a_shared_engine_trusts_the_real_start_flags_not_the_state_file(comfy, monkeypatch):
    """Codex review: a replacement engine with Sage could inherit a stored sage=False and take Qwen (black pictures).
    The shared branch reads the running engine's own flags; unknown flags mean a restart."""
    from cinestudio import comfy as cf
    monkeypatch.setattr(s, "shared_engine", lambda: True)
    monkeypatch.setattr(cf, "sage_for", lambda fams, cfg=None: bool(fams) and all(f.startswith("ltx25") for f in fams))
    s.ensure_family("qwen_edit@a", URL)                               # an engine without Sage, recorded so
    our_job(comfy, "p1")
    comfy.argv = ["main.py", "--use-sage-attention"]                  # replaced outside cinestudio, with Sage
    assert s.ensure_family("qwen_edit@a", URL)["restarted"]           # Qwen not admitted into a Sage engine
    our_job(comfy, "p2")
    real_get = comfy.get

    def no_stats(url, timeout=5):
        if url.endswith("/system_stats"):
            return {}                                                 # alive, but no start flags reported
        return real_get(url, timeout)
    monkeypatch.setattr(s, "_get", no_stats)
    assert s.ensure_family("acestep@c", URL)["restarted"]             # flags unknown: restart, never guess
