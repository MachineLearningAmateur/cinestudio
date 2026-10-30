"""SdkBackend logic with a fake SDK client (no server needed)."""

from types import SimpleNamespace

import pytest
from comfy_sdk import exceptions as sdk_errors

from cinestudio import backend as b


def graph(prefix):
    return {"save": {"class_type": "SaveVideo", "inputs": {"filename_prefix": prefix}},
            "ks": {"class_type": "KSampler", "inputs": {"seed": 1}}}


class FakeClient:
    def __init__(self, jobs, submit_error=None):
        self._jobs = jobs                      # newest first: [(id, graph)]
        self.submit_error = submit_error
        self.workflows = SimpleNamespace(from_json=lambda g: SimpleNamespace(set_input=lambda *a: None))
        self.assets = SimpleNamespace(from_file=lambda p: p)
        self.jobs = SimpleNamespace(get=self._get)

    def _get(self, job_id):
        g = dict(self._jobs)[job_id]
        return SimpleNamespace(get_workflow=lambda: SimpleNamespace(graph=g, format="api"))

    def list_jobs(self, limit=50):
        return [SimpleNamespace(id=i) for i, _ in self._jobs[:limit]]

    def submit(self, wf, idempotency_key):
        if self.submit_error:
            raise self.submit_error
        return SimpleNamespace(id="new")


def backend(client):
    be = object.__new__(b.SdkBackend)
    be.client = client
    return be


def test_find_is_exact():
    be = backend(FakeClient([("j10", graph("cinestudio/run_c01_a10")), ("j1", graph("cinestudio/run_c01_a1"))]))
    assert be.find("cinestudio/run_c01_a1") == "j1"
    assert be.find("cinestudio/run_c01_a10") == "j10"
    assert be.find("cinestudio/run_c01_a") is None


def test_ambiguous_submit_finds_existing_job():
    be = backend(FakeClient([("j1", graph("cinestudio/r_c01_a1"))], submit_error=ConnectionError("reset")))
    assert be.submit(graph("cinestudio/r_c01_a1"), {}, key="k", prefix="cinestudio/r_c01_a1") == "j1"


def test_ambiguous_submit_without_job_is_unresolved():
    be = backend(FakeClient([], submit_error=TimeoutError("timeout")))
    with pytest.raises(b.UnresolvedSubmission) as e:
        be.submit(graph("cinestudio/r_c01_a1"), {}, key="k", prefix="cinestudio/r_c01_a1")
    assert e.value.prefix == "cinestudio/r_c01_a1"


def test_rejection_is_not_ambiguous():
    err = sdk_errors.InvalidWorkflow.__new__(sdk_errors.InvalidWorkflow)
    be = backend(FakeClient([("j1", graph("cinestudio/r_c01_a1"))], submit_error=err))
    with pytest.raises(sdk_errors.InvalidWorkflow):
        be.submit(graph("cinestudio/r_c01_a1"), {}, key="k", prefix="cinestudio/r_c01_a1")


def test_explicit_base_url_wins(monkeypatch):
    monkeypatch.setenv("COMFY_BASE_URL", "http://env:1")
    monkeypatch.setattr(b, "Comfy", lambda **kw: object())
    assert b.SdkBackend(base_url="http://explicit:2").base_url == "http://explicit:2"
    monkeypatch.setenv("COMFY_BASE_URL", "http://env:1")
    assert b.SdkBackend().base_url == "http://env:1"


@pytest.mark.parametrize("status,done,ok", [("succeeded", True, True), ("expired", True, False),
                                            ("canceled", True, False), ("running", False, False)])
def test_terminal_states(status, done, ok):
    s = b.JobState("j", status)
    assert (s.done, s.ok) == (done, ok)


def test_watch_polls_if_event_stream_is_unavailable(monkeypatch):
    states = iter([b.JobState("j", "running"), b.JobState("j", "succeeded")])
    be = backend(SimpleNamespace(jobs=SimpleNamespace(get=lambda _: SimpleNamespace(events=lambda: iter(())))))
    monkeypatch.setattr(be, "state", lambda _: next(states))
    monkeypatch.setattr(b.time, "sleep", lambda _: None)
    assert be.watch("j", lambda *_: None).status == "succeeded"
