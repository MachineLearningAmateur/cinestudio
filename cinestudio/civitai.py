"""Downloads from Civitai (LoRAs whose creator requires a login): the optional Civitai token from settings.

    python -m cinestudio models civitai VERSION_ID [--dest loras]

The file name and SHA-256 come from Civitai's public model-version API. The token goes only to civitai.com: the first
request is made without following redirects, and the signed download address it points to (a CDN) is fetched without
the token. The file is written as NAME.part, checked against the SHA-256, and only then renamed. MiniMax files are
refused (user rule: MiniMax H3 must not be used).
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import settings

API = "https://civitai.com/api/v1"
CHUNK = 1 << 20


class CivitaiError(RuntimeError):
    pass


TOKEN_HOSTS = {"civitai.com"}


def _token_allowed(url: str) -> bool:
    """The token goes only to Civitai itself over https (Codex review 2026-10-22: the API's downloadUrl was trusted)."""
    u = urllib.parse.urlsplit(url)
    return u.scheme == "https" and (u.hostname or "").lower() in TOKEN_HOSTS and u.port in (None, 443)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):   # noqa: D401 - urllib hook
        return None


def version_info(version_id: int, timeout: float = 30) -> dict:
    req = urllib.request.Request(f"{API}/model-versions/{int(version_id)}", headers={"User-Agent": "cinestudio"})
    with urllib.request.urlopen(req, timeout=timeout, context=settings.ssl_context()) as r:
        return json.loads(r.read())


def _pick_file(info: dict) -> dict:
    files = [f for f in info.get("files") or [] if (f.get("type") or "Model") == "Model"] or (info.get("files") or [])
    if not files:
        raise CivitaiError("this model version has no files")
    return next((f for f in files if f.get("primary")), files[0])


def download(version_id: int, dest_dir: Path, timeout: float = 60) -> dict:
    info = version_info(version_id)
    if "minimax" in str(info.get("baseModel", "")).lower():
        raise CivitaiError(f"version {version_id} is for {info.get('baseModel')}: MiniMax models are not used")
    f = _pick_file(info)
    name = Path(f["name"]).name
    want = ((f.get("hashes") or {}).get("SHA256") or "").lower()
    if len(want) != 64:
        raise CivitaiError(f"version {version_id} gives no SHA-256 for {name}: the file cannot be checked, not downloaded")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / name
    if out.is_file():
        if _sha256(out) == want:
            return {"file": str(out), "sha256": want, "already": True, "base_model": info.get("baseModel")}
        raise CivitaiError(f"{out} exists and is another file (another version or your own): not replaced. "
                           "Move or delete it first.")
    url = f.get("downloadUrl") or f"{API.replace('/v1', '')}/download/models/{int(version_id)}"
    headers = {"User-Agent": "cinestudio"}
    tok = settings.token("civitai")
    if tok and _token_allowed(url):
        headers["Authorization"] = f"Bearer {tok}"
    opener = urllib.request.build_opener(_NoRedirect, urllib.request.HTTPSHandler(context=settings.ssl_context()))
    try:
        r = opener.open(urllib.request.Request(url, headers=headers), timeout=timeout)
        target = None                                     # no redirect: civitai sends the file itself
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
            target, r = e.headers["Location"], None
        elif e.code in (401, 403):
            raise CivitaiError("Civitai needs a login for this file: add a Civitai token in Settings > Access tokens "
                               "(or set CIVITAI_TOKEN)" if not tok else
                               "Civitai rejected the token (wrong, deleted, or no access to this file)") from e
        else:
            raise CivitaiError(f"Civitai answered HTTP {e.code}") from e
    if target is not None:                                # the signed CDN address: fetched WITHOUT the token
        r = urllib.request.urlopen(urllib.request.Request(target, headers={"User-Agent": "cinestudio"}),
                                   timeout=timeout, context=settings.ssl_context())
    part = out.with_name(f"{out.name}.{secrets.token_hex(4)}.part")   # own temp file: parallel downloads
    h = hashlib.sha256()
    try:                                                  # the temp file never stays behind (Codex review 2026-10-22)
        with r, open(part, "wb") as fh:
            while True:
                b = r.read(CHUNK)
                if not b:
                    break
                fh.write(b)
                h.update(b)
        got = h.hexdigest()
        if got != want:
            raise CivitaiError(f"SHA-256 mismatch for {name}: expected {want[:12]}..., got {got[:12]}...")
        _publish(part, out, want)
    finally:
        part.unlink(missing_ok=True)
    return {"file": str(out), "sha256": got, "already": False, "base_model": info.get("baseModel"),
            "trained_words": info.get("trainedWords")}


def _publish(part: Path, out: Path, want: str) -> None:
    """Move the checked file into place WITHOUT replacing a file that appeared meanwhile (a hard link fails when the
    name exists; Codex review 2026-10-22)."""
    try:
        try:
            os.link(part, out)
        except FileExistsError:
            if _sha256(out) != want:
                raise CivitaiError(f"{out} appeared meanwhile and is another file: not replaced") from None
        except OSError:                                    # no hard links here (another file system): rename instead
            if out.exists():
                if _sha256(out) != want:
                    raise CivitaiError(f"{out} appeared meanwhile and is another file: not replaced") from None
            else:
                os.rename(part, out)
                return
    finally:
        part.unlink(missing_ok=True)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()
