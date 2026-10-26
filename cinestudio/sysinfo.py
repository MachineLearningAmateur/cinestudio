"""Live resource use of this PC for the app's meters: CPU, RAM, GPU, VRAM and GPU temperature.

CPU and RAM come from psutil. GPU values come from NVIDIA's NVML library (nvidia-ml-py); a PC
without an NVIDIA GPU or without the library gets `gpu: None`, and the page hides those meters.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover - listed in setup/requirements.txt
    psutil = None

_nvml = None
_nvml_lock = threading.Lock()
_retry_at = 0.0          # after a failure: no new NVML start before this time
RETRY_S = 30.0


def _gpu_handle():
    """The NVML handle of GPU 0, or None (no NVIDIA GPU, no driver or no library). A failed start is tried again after
    RETRY_S seconds (2026-10-22: after a GPU driver reset the meters stayed off until the app restarted)."""
    global _nvml, _retry_at
    with _nvml_lock:
        if _nvml is None and time.monotonic() >= _retry_at:
            started = None
            try:
                import pynvml
                pynvml.nvmlInit()
                started = pynvml
                _nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
            except Exception:  # noqa: BLE001 - any failure means "no GPU meters" for now
                if started is not None:          # init worked, the handle did not: balance the init (Codex 10-22)
                    try:
                        started.nvmlShutdown()
                    except Exception:  # noqa: BLE001
                        pass
                _retry_at = time.monotonic() + RETRY_S
        return _nvml


def _drop_handle() -> None:
    """A call on the handle failed (a driver reset makes the old handle invalid): start NVML again later."""
    global _nvml, _retry_at
    with _nvml_lock:
        if _nvml is not None:
            try:
                _nvml[0].nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
        _nvml = None
        _retry_at = time.monotonic() + 2.0


def gpu() -> dict | None:
    h = _gpu_handle()
    if h is None:
        return None
    nv, dev = h
    try:
        util = nv.nvmlDeviceGetUtilizationRates(dev)
        mem = nv.nvmlDeviceGetMemoryInfo(dev)
        out = {"util": float(util.gpu), "vram_used": mem.used, "vram_total": mem.total}
        try:
            out["temp"] = float(nv.nvmlDeviceGetTemperature(dev, nv.NVML_TEMPERATURE_GPU))
        except Exception:  # noqa: BLE001
            out["temp"] = None
        name = nv.nvmlDeviceGetName(dev)
        out["name"] = name.decode() if isinstance(name, bytes) else name
        return out
    except Exception:  # noqa: BLE001
        _drop_handle()
        return None


CGROUP_FILES = (   # (limit, usage, stat, inactive-file key): cgroup v2, then v1
    ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.stat", "inactive_file"),
    ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes",
     "/sys/fs/cgroup/memory/memory.stat", "total_inactive_file"),
)


def cgroup_memory() -> tuple[int | None, int | None]:
    """(limit, used) of this container's memory cgroup in bytes, or (None, None) without a limit. A container (a RunPod
    pod) sees the HOST's RAM in /proc/meminfo; its real limit is in the cgroup (RunPod check, 2026-10-26). `used`
    leaves out the inactive page cache, as `docker stats` does (ComfyUI's memory-mapped weights fill the cache)."""
    if not sys.platform.startswith("linux"):
        return None, None
    for limit_f, usage_f, stat_f, key in CGROUP_FILES:
        try:
            raw = Path(limit_f).read_text().strip()
        except OSError:
            continue
        if not raw.isdigit() or int(raw) >= 1 << 60:      # "max" (v2) or a huge number (v1) = no limit
            return None, None
        used = None
        try:
            used = int(Path(usage_f).read_text().strip())
            for line in Path(stat_f).read_text().splitlines():
                name, _, value = line.partition(" ")
                if name == key:
                    used = max(0, used - int(value))
                    break
        except (OSError, ValueError):
            pass
        return int(raw), used
    return None, None


def total_ram() -> int | None:
    """The RAM this process may use, in bytes: the machine's total, or the container's limit when that is smaller."""
    if psutil is None:
        return None
    total = psutil.virtual_memory().total
    limit, _ = cgroup_memory()
    return min(total, limit) if limit else total


def snapshot() -> dict:
    """One reading. CPU % is measured since the last call (the first call after start reads 0)."""
    out: dict = {"cpu": None, "ram_used": None, "ram_total": None, "gpu": gpu()}
    if psutil is not None:
        vm = psutil.virtual_memory()
        used, total = vm.total - vm.available, vm.total
        limit, cg_used = cgroup_memory()
        if limit and limit < total:
            total, used = limit, (cg_used if cg_used is not None else min(used, limit))
        out.update(cpu=psutil.cpu_percent(interval=None), ram_used=used, ram_total=total)
    return out


if psutil is not None:
    psutil.cpu_percent(interval=None)       # start the CPU counter, so the first page read is not 0
