"""Installer gates. No services are started and no generation jobs are submitted.

Run CUDA with ComfyUI's Python; run locks and doctor with the app's Python.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_pins(python: str, cuda: str) -> dict:
    manifest = json.loads((ROOT / "setup/linux/manifest.json").read_text(encoding="utf-8"))
    files = ["setup/requirements.txt", "setup/runtime.lock.json", f"setup/linux/app-py{python}.txt",
             f"setup/linux/comfy-py{python}-{cuda}.txt", "setup/linux/sageattention.txt"]
    for name in files:
        digest = hashlib.sha256((ROOT / name).read_text(encoding="utf-8").encode()).hexdigest()
        if manifest.get(name) != digest:
            raise RuntimeError(f"dependency lock is stale or modified: {name}; regenerate with "
                               "python setup/lock_linux.py and validate on a pod")
    return {"python": python, "cuda": cuda}


def check_cuda(expected_version: str, cuda: str) -> dict:
    import torch
    import torchvision  # noqa: F401 - catch incompatible binary wheels before downloading models
    import torchaudio  # noqa: F401

    expected_cuda = {"cu130": "13.0", "cu128": "12.8"}[cuda]
    if torch.__version__.split("+")[0] != expected_version:
        raise RuntimeError(f"expected torch {expected_version}, found {torch.__version__}")
    if torch.version.cuda != expected_cuda:
        raise RuntimeError(f"expected CUDA {expected_cuda} wheels, found {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot use CUDA. Check the pod's NVIDIA driver and selected CUDA image; "
                           "installation stopped before model downloads.")
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}


def probe_lock(path: Path) -> int:
    from cinestudio.locks import FileLock, LockBusy
    try:
        with FileLock(path):
            return 0
    except LockBusy:
        return 3


def check_locks(home: Path) -> dict:
    """Check exclusion AND release across processes, using only a new temporary directory."""
    from cinestudio.locks import FileLock
    home.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lock-check-", dir=home) as tmp:
        path = Path(tmp) / "probe.lock"
        cmd = [sys.executable, str(Path(__file__).resolve()), "probe-lock", str(path)]
        with FileLock(path):
            held = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        released = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if held.returncode != 3 or released.returncode != 0:
            raise RuntimeError(f"file locks do not exclude other processes or release correctly in {home}. "
                               "Set CINESTUDIO_HOME to a disk supporting file locks and rerun installation. "
                               f"{held.stderr.strip()} {released.stderr.strip()}".strip())
    return {"home": str(home), "exclusion": True, "release": True}


def assess_doctor(report: dict, skip_models: bool = False) -> dict:
    """Only absent services and explicitly deferred models are allowed during installation."""
    checks = {name: dict(value) for name, value in report["checks"].items()}
    for name in ("comfyui", "proxy"):
        if name in checks:
            checks[name] = {**checks[name], "skipped": "service startup is checked when the app starts"}
    if skip_models:
        for name in ("models", "model_tier"):
            if name in checks:
                checks[name] = {**checks[name], "skipped": "model checks deferred by --skip-models"}
    elif "model_tier" in checks and checks["model_tier"].get("ready") is not True:
        checks["model_tier"]["ok"] = False
    required = {"blender", "ffmpeg", "models", "model_tier", "comfyui", "proxy"}
    failed = sorted((required - checks.keys()) | {
        name for name, value in checks.items() if not value.get("skipped") and not value.get("ok")
    })
    return {"ok": not failed, "checks": checks, "failed": failed}


def check_render() -> dict:
    """A manual pod smoke test; refuse while work waits or runs, and retain its images/log."""
    from cinestudio import blender, runner, server, workqueue
    with server.gpu_lock(note={"check": "headless Blender"}):
        if workqueue.tickets() or (server.alive() and server.queue_items(server.comfyui_url())):
            raise RuntimeError("work is queued or running; run the Blender check when the GPU is idle")
        base = runner.home() / "cache" / "setup_checks"
        base.mkdir(parents=True, exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="render-", dir=base))
        blender.run_job("check_runtime.py", None, ["--out", str(out)], out / "blender.log", timeout=120)
        import cv2
        for name in ("workbench", "eevee"):
            img = cv2.imread(str(out / f"{name}.png"))
            if img is None or img.shape[:2] != (64, 64) or float(img.std()) < 1:
                raise RuntimeError(f"{name} did not produce a usable render; see {out / 'blender.log'}")
    return {"outputs": str(out), "engines": ["workbench", "eevee"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="check", required=True)
    p = commands.add_parser("pins")
    p.add_argument("--python-version", choices=("3.12", "3.13"), required=True)
    p.add_argument("--cuda", choices=("cu130", "cu128"), required=True)
    p = commands.add_parser("cuda")
    p.add_argument("--torch-version", required=True)
    p.add_argument("--cuda", choices=("cu130", "cu128"), required=True)
    p = commands.add_parser("locks")
    p.add_argument("--home", type=Path, required=True)
    p = commands.add_parser("probe-lock")
    p.add_argument("path", type=Path)
    p = commands.add_parser("doctor")
    p.add_argument("--skip-models", action="store_true")
    commands.add_parser("render", help="manually render one 64px frame with Workbench and EEVEE on an idle GPU")
    args = parser.parse_args(argv)
    try:
        if args.check == "probe-lock":
            return probe_lock(args.path)
        if args.check == "pins":
            result = {"ok": True, **check_pins(args.python_version, args.cuda)}
        elif args.check == "cuda":
            result = {"ok": True, **check_cuda(args.torch_version, args.cuda)}
        elif args.check == "locks":
            result = {"ok": True, **check_locks(args.home)}
        elif args.check == "render":
            result = {"ok": True, **check_render()}
        else:
            from cinestudio.cli import doctor
            result = assess_doctor(doctor(), args.skip_models)
        print(json.dumps({"check": args.check, **result}))
        return 0 if result["ok"] else 1
    except Exception as exc:  # the installer must stop, with a concrete reason
        print(json.dumps({"ok": False, "check": args.check, "error": str(exc), "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
