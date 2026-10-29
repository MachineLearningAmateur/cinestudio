"""Resolve Linux dependencies without installing packages or downloading model weights.

Requires uv on PATH. Run after changing runtime.lock.json or requirements.txt, then validate on a pod.
The checked-in locks cover Python 3.12/3.13, x86-64 Ubuntu 24.04, and both supported CUDA builds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def requirement_urls(runtime: dict) -> list[str]:
    # SAM2 has no requirements.txt at the pinned revision (its pyproject dependencies are empty).
    packs = [runtime["comfyui"], *(p for p in runtime["node_packs"]
                                 if p["name"] != "ComfyUI-segment-anything-2")]
    return [f"https://raw.githubusercontent.com/{p['repo'].removesuffix('.git').split('github.com/')[1]}/"
            f"{p['commit']}/requirements.txt" for p in packs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraints", type=Path, help="optional initial version constraints")
    args = parser.parse_args()
    runtime = json.loads((ROOT / "setup/runtime.lock.json").read_text(encoding="utf-8"))
    out = ROOT / "setup/linux"
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        requirements = []
        for url in requirement_urls(runtime):
            with urllib.request.urlopen(url, timeout=30) as response:
                requirements.extend(response.read().decode().splitlines())
        constraints = Path(tmp) / "constraints.txt"
        if args.constraints:
            constraints.write_bytes(args.constraints.read_bytes())
        for python in ("3.12", "3.13"):
            common = ["uv", "--no-config", "pip", "compile", "--quiet", "--python-version", python,
                      "--python-platform", "x86_64-manylinux_2_39", "--no-python-downloads",
                      "--generate-hashes", "--no-annotate", "--custom-compile-command",
                      "python setup/lock_linux.py"]
            app = out / f"app-py{python}.txt"
            subprocess.run([*common, str(ROOT / "setup/requirements.txt"), "-o", str(app)], check=True)
            for cuda, version in runtime["torch"]["linux_versions"].items():
                source = Path(tmp) / "comfy.in"
                source.write_text("\n".join([
                    *requirements,
                    f"torch=={version}+{cuda}",
                    "torchaudio",  # imported by ComfyUI's audio nodes
                ]) + "\n", encoding="utf-8")
                cmd = [*common, str(source), "--torch-backend", cuda,
                       "-o", str(out / f"comfy-py{python}-{cuda}.txt")]
                if args.constraints:
                    cmd += ["-c", str(constraints)]
                subprocess.run(cmd, check=True)
        sage = Path(tmp) / "sage.in"
        sage.write_text(f"sageattention=={runtime['sageattention']['tested']}\n", encoding="utf-8")
        subprocess.run([*common, str(sage), "--no-deps", "-o", str(out / "sageattention.txt")], check=True)
    paths = [ROOT / "setup/requirements.txt", ROOT / "setup/runtime.lock.json", *sorted(out.glob("*.txt"))]
    manifest = {str(p.relative_to(ROOT).as_posix()): hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
                for p in paths}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
