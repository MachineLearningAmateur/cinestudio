"""Linux pod behaviour (RunPod audit 2026-10-26), tested on any OS by faking the platform and its files."""

from __future__ import annotations

import errno
import sys
import types

import pytest

from cinestudio import comfy, locks, server, spawn, sysinfo


def _cgroup_v2(tmp_path, limit: str, current: str = "0", inactive: str = "0"):
    (tmp_path / "memory.max").write_text(limit + "\n")
    (tmp_path / "memory.current").write_text(current + "\n")
    (tmp_path / "memory.stat").write_text(f"anon 1\ninactive_file {inactive}\nactive_file 5\n")
    return ((str(tmp_path / "memory.max"), str(tmp_path / "memory.current"), str(tmp_path / "memory.stat"),
             "inactive_file"),)


def test_a_pod_reads_its_cgroup_limit_not_the_hosts_ram(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sysinfo, "CGROUP_FILES", _cgroup_v2(tmp_path, str(125 * 1024 ** 3), str(40 * 1024 ** 3),
                                                            str(10 * 1024 ** 3)))
    assert sysinfo.cgroup_memory() == (125 * 1024 ** 3, 30 * 1024 ** 3)       # the inactive page cache left out
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("M", (), {"total": 2048 * 1024 ** 3,
                                                                         "available": 1500 * 1024 ** 3})())
    assert sysinfo.total_ram() == 125 * 1024 ** 3
    snap = sysinfo.snapshot()
    assert snap["ram_total"] == 125 * 1024 ** 3 and snap["ram_used"] == 30 * 1024 ** 3


def test_no_cgroup_limit_keeps_the_machines_ram(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sysinfo, "CGROUP_FILES", _cgroup_v2(tmp_path, "max"))
    assert sysinfo.cgroup_memory() == (None, None)
    monkeypatch.setattr(sysinfo, "CGROUP_FILES", (("/no/such/file", "", "", ""),))
    assert sysinfo.cgroup_memory() == (None, None)
    monkeypatch.setattr(sys, "platform", "win32")
    assert sysinfo.cgroup_memory() == (None, None)


def test_the_shared_engine_uses_the_pods_limit(tmp_path, monkeypatch):
    from cinestudio import models
    monkeypatch.setattr(models, "job_tier", lambda: "large")
    monkeypatch.setattr(models, "engine_settings", lambda tier: {"share_families": True, "min_ram_gb": 150})
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("M", (), {"total": 2048 * 1024 ** 3})())
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sysinfo, "CGROUP_FILES", _cgroup_v2(tmp_path, str(125 * 1024 ** 3)))
    assert not server.shared_engine()                  # a 125 GB pod on a 2 TB host: one family per engine
    monkeypatch.setattr(sysinfo, "CGROUP_FILES", _cgroup_v2(tmp_path, str(188 * 1024 ** 3)))
    assert server.shared_engine()


def test_sage_attention_is_found_in_a_linux_venv(tmp_path):
    venv = tmp_path / "ComfyUI" / ".venv"
    (venv / "bin").mkdir(parents=True)
    py = venv / "bin" / "python"
    py.write_text("")                                 # on the pod a link to /usr/bin/python3.x; resolve() leaves the venv
    assert not comfy.has_sage_attention({"python": str(py)})
    pkg = venv / "lib" / "python3.12" / "site-packages" / "sageattention"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    assert comfy.has_sage_attention({"python": str(py)})


def _fake_fcntl(monkeypatch, err: int):
    def flock(fd, op):
        raise OSError(err, "fake")
    monkeypatch.setitem(sys.modules, "fcntl", types.SimpleNamespace(flock=flock, LOCK_EX=2, LOCK_NB=4))
    monkeypatch.setattr(sys, "platform", "linux")


def test_a_held_flock_is_busy(tmp_path, monkeypatch):
    _fake_fcntl(monkeypatch, errno.EWOULDBLOCK)
    with pytest.raises(locks.LockBusy):
        locks.FileLock(tmp_path / "q.lock", timeout=0).acquire()


def test_a_filesystem_without_flock_is_an_error_not_busy(tmp_path, monkeypatch):
    _fake_fcntl(monkeypatch, errno.ENOLCK)
    with pytest.raises(OSError, match="file locks do not work"):
        locks.FileLock(tmp_path / "q.lock", timeout=0).acquire()


def test_a_posix_worker_counts_as_independent(monkeypatch):
    seen = {}

    class P:
        def __init__(self, cmd, **kw):
            seen.update(kw)
    monkeypatch.setattr(spawn, "WIN", False)
    monkeypatch.setattr(spawn.subprocess, "Popen", P)
    _, independent = spawn._popen(["x"])
    assert independent and seen.get("start_new_session") is True
