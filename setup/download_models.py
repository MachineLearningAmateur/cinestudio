"""Download the model files of one profile into a ComfyUI models folder.

    python setup/download_models.py --profile ltx23 --models-dir "D:/ComfyUI/models"
    python setup/download_models.py --profile zimage --models-dir ... --check     # verify only

- Resumable: data goes to <file>.part and is renamed only after the size and SHA-256 match,
  so ComfyUI never lists a half-written model. Run it again to continue.
- Integrity: models.json pins each URL to a repository commit and stores the expected size and
  SHA-256. A file without both values is refused; a file is never used without a hash match.
- --check-remote also compares the pins with the Hugging Face LFS metadata (needs the network).
- Uses curl (built into Windows 10+, macOS and most Linux systems) for robust resuming.
Prints one JSON line per file.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def remote_meta(url):
    """(size, sha256) from the Hugging Face resolve URL: HEAD without redirects gives the LFS metadata."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "cinestudio-setup"})
    try:
        r = opener.open(req, timeout=60)
        headers = r.headers
    except urllib.error.HTTPError as e:  # 302 to the CDN carries the headers we need
        headers = e.headers
    sha = (headers.get("X-Linked-Etag") or headers.get("ETag") or "").strip('"')
    size = headers.get("X-Linked-Size") or headers.get("Content-Length")
    return (int(size) if size else None), (sha if len(sha) == 64 else None)


def sha256(path, chunk=16 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--models-dir", required=True, help="the ComfyUI 'models' folder")
    ap.add_argument("--check", action="store_true", help="only verify files that exist")
    ap.add_argument("--check-remote", action="store_true", help="also compare the pinned size/hash with Hugging Face")
    a = ap.parse_args()
    with open(os.path.join(HERE, "models.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    if a.profile not in manifest:
        print(json.dumps({"error": f"unknown profile {a.profile}", "profiles": [k for k in manifest if not k.startswith("_")]}))
        return 1
    curl = shutil.which("curl")
    failed = 0
    for m in manifest[a.profile]:
        dst = os.path.join(a.models_dir, m["dir"], m["file"])
        size, sha = m.get("size"), m.get("sha256")
        rec = {"file": m["file"], "dir": m["dir"], "size": size, "sha256": sha}
        if not (isinstance(size, int) and size > 0 and isinstance(sha, str) and len(sha) == 64):
            rec["status"] = "REFUSED: models.json has no pinned size and sha256 for this file"
            failed += 1
            print(json.dumps(rec), flush=True)
            continue
        if a.check_remote:
            r_size, r_sha = remote_meta(m["url"])
            if (r_size, r_sha) != (size, sha):
                rec["remote"] = {"size": r_size, "sha256": r_sha}
                rec["status"] = "PIN MISMATCH with Hugging Face"
                failed += 1
                print(json.dumps(rec), flush=True)
                continue
        if os.path.exists(dst):
            ok = os.path.getsize(dst) == size and sha256(dst) == sha
            rec["status"] = "ok" if ok else "MISMATCH (delete the file and run again)"
            failed += not ok
            print(json.dumps(rec), flush=True)
            continue
        if a.check:
            rec["status"] = "missing"
            failed += 1
            print(json.dumps(rec), flush=True)
            continue
        if not curl:
            rec["status"] = "curl not found"
            failed += 1
            print(json.dumps(rec), flush=True)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        part = dst + ".part"
        for attempt in range(20):
            if os.path.exists(part) and os.path.getsize(part) == size:
                break
            if os.path.exists(part) and os.path.getsize(part) > size:
                os.remove(part)                 # longer than the pinned file: cannot be resumed
            r = subprocess.run([curl, "-L", "--fail", "--retry", "10", "--retry-delay", "5", "-C", "-",
                                "-sS", "-o", part, m["url"]])
            if r.returncode == 0:
                break
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have != size:
            rec["status"] = f"incomplete: {have} of {size} bytes (run again to resume)"
            failed += 1
        elif sha256(part) != sha:
            rec["status"] = "HASH MISMATCH (delete the .part file and run again)"
            failed += 1
        else:
            os.replace(part, dst)
            rec["status"] = "downloaded"
        print(json.dumps(rec), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
