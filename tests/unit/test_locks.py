"""OS file locks: exclusive across processes, released when the holder dies."""

import subprocess
import sys

import pytest

from cinestudio.locks import FileLock, LockBusy, is_locked

HOLD = ("import sys, time; sys.path.insert(0, sys.argv[2]); from cinestudio.locks import FileLock; "
        "l = FileLock(sys.argv[1]).acquire(); print('held', flush=True); time.sleep(60)")


def _holder(path, root):
    # the real interpreter, not the venv launcher (the launcher has another pid than the lock holder)
    exe = getattr(sys, "_base_executable", sys.executable)
    p = subprocess.Popen([exe, "-c", HOLD, str(path), str(root)], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    return p


def test_second_process_cannot_take_it_and_death_releases_it(tmp_path, request):
    root = request.config.rootpath
    lock = tmp_path / "run.lock"
    p = _holder(lock, root)
    try:
        assert is_locked(lock)
        with pytest.raises(LockBusy) as e:
            FileLock(lock, timeout=0).acquire()
        assert str(p.pid) in str(e.value)
    finally:
        p.kill()
        p.wait()
    assert not is_locked(lock)                         # the OS released it with the dead process
    with FileLock(lock, timeout=0) as held:
        assert held.held


def test_timeout_waits_then_fails(tmp_path, request):
    lock = tmp_path / "gpu.lock"
    p = _holder(lock, request.config.rootpath)
    try:
        with pytest.raises(LockBusy):
            FileLock(lock, timeout=0.5, poll=0.1).acquire()
    finally:
        p.kill()
        p.wait()
