"""Driver calls must not freeze the web app or accumulate meter threads."""

import threading
import types

from cinestudio import sysinfo


def test_gpu_sampler_reuses_recent_reading(monkeypatch):
    calls = []
    monkeypatch.setattr(sysinfo, "gpu", lambda: calls.append(1) or {"util": 50.0})
    sampler = sysinfo._GpuSampler()
    assert sampler.read() == {"util": 50.0}
    assert sampler.read() == {"util": 50.0}
    assert len(calls) == 1


def test_gpu_sampler_hides_stale_reading_and_keeps_one_pending_scan(monkeypatch):
    clock = [0.0]
    release = threading.Event()
    calls = []

    def read_gpu():
        calls.append(1)
        if len(calls) > 1:
            release.wait()
        return {"util": 50.0}

    monkeypatch.setattr(sysinfo, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(sysinfo, "gpu", read_gpu)
    sampler = sysinfo._GpuSampler()
    assert sampler.read() == {"util": 50.0}
    clock[0] = 6.0
    pending = None
    try:
        assert sampler.read() is None
        pending = sampler._pending
        for _ in range(3):
            assert sampler.read() is None
        assert len(calls) == 2
    finally:
        release.set()
        if pending is not None:
            assert pending.wait(2)
