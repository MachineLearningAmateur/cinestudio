"""Make pinned manifest entries (commit URL, size, sha256) for setup/models.json from Hugging Face.

    python setup/pin_models.py <profile> <dir>:<repo>:<path> [...]   # prints the JSON entries

<dir> is the ComfyUI models sub-folder; the sha256 and size come from the file's LFS pointer at the
repo's current commit, and the URL is pinned to that commit.
"""

from __future__ import annotations

import json
import sys
import urllib.request


def _get(url: str, data: bytes | None = None) -> object:
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def pin(dir_: str, repo: str, path: str) -> dict:
    sha = _get(f"https://huggingface.co/api/models/{repo}")["sha"]
    info = _get(f"https://huggingface.co/api/models/{repo}/paths-info/{sha}", json.dumps({"paths": [path]}).encode())
    if not info or "lfs" not in info[0]:
        raise SystemExit(f"{repo}/{path}: not found or not an LFS file")
    lfs = info[0]["lfs"]
    return {"dir": dir_, "file": path.rsplit("/", 1)[-1],
            "url": f"https://huggingface.co/{repo}/resolve/{sha}/{path}",
            "size": lfs["size"], "sha256": lfs["oid"]}


if __name__ == "__main__":
    profile, specs = sys.argv[1], sys.argv[2:]
    print(json.dumps({profile: [pin(*s.split(":", 2)) for s in specs]}, indent=1))
