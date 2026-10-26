"""User settings and access tokens (Settings > Quick setup, `cinestudio settings ...`).

    settings.json  {"model_tier": "8gb", "backend": "local"}       in the cinestudio home (.cinestudio/, git-ignored)
    secrets.json   {"hf": "...", "comfy": "..."}                    same folder, owner-only on POSIX

Tokens:
    hf     Hugging Face access token: optional for the public models, needed for gated ones (LTX-2.5) and gives
           faster, less throttled downloads. Env HF_TOKEN wins over the file.
    comfy  Comfy Cloud API key: only for the Comfy Cloud backend (a paid Comfy plan). Env COMFY_API_KEY wins.

A token value never leaves this module except as an HTTP header to its own service: status() reports only whether
a token is set and where it comes from. Settings are defaults for NEW work; a run keeps the models it was made with.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TOKENS = {
    "hf": {"env": "HF_TOKEN", "label": "Hugging Face token", "required": False,
           "help": "Faster model downloads. Needed only for gated models.",
           "url": "https://huggingface.co/settings/tokens"},
    "comfy": {"env": "COMFY_API_KEY", "label": "Comfy Cloud API key", "required": False,
              "help": "Only to run on Comfy Cloud instead of this computer (a paid Comfy plan).",
              "url": "https://platform.comfy.org"},
    "civitai": {"env": "CIVITAI_TOKEN", "label": "Civitai API key", "required": False,
                "help": "Only to download LoRAs from Civitai whose creator requires a login.",
                "url": "https://civitai.com/user/account"},
}
BACKENDS = ("local", "cloud")
DEFAULTS = {"model_tier": "8gb", "backend": "local"}


class SettingsError(ValueError):
    pass


def _home() -> Path:
    from .runner import home
    d = home()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _settings_path() -> Path:
    return _home() / "settings.json"


def _secrets_path() -> Path:
    return _home() / "secrets.json"


def _read(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(p: Path, data: dict, private: bool = False) -> None:
    tmp = p.with_name(p.name + ".tmp")
    if private and not sys.platform.startswith("win"):
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)     # owner-only from the first byte
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
    else:
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, p)


# ------------------------------------------------------------------ settings

def get() -> dict:
    d = {**DEFAULTS, **{k: v for k, v in _read(_settings_path()).items()
                        if k in DEFAULTS or k in ("models_dir", "sage_attention", "default_style", "advanced")}}
    from .models import canon                      # an old tier name ("24gb" -> "large", renamed 2026-10-24)
    d["model_tier"] = canon(d["model_tier"])
    return d


def _set(key: str, value) -> dict:
    d = _read(_settings_path())
    d[key] = value
    d["updated"] = time.time()
    _write(_settings_path(), d)
    return get()


def set_tier(tier: str) -> dict:
    from . import models
    tier = models.canon(tier)
    if tier not in models.tier_names():
        raise SettingsError(f"model tier must be one of {models.tier_names()}")
    return _set("model_tier", tier)


def set_backend(backend: str) -> dict:
    if backend not in BACKENDS:
        raise SettingsError(f"backend must be one of {list(BACKENDS)}")
    if backend == "cloud" and not token("comfy"):
        raise SettingsError("Comfy Cloud needs a Comfy Cloud API key first (Quick setup)")
    return _set("backend", backend)


def set_advanced(on: bool) -> dict:
    """The app's advanced mode (the Workflows view: the ComfyUI graphs behind each feature). Off by default: the
    normal screens stay simple (user, 2026-10-25)."""
    return _set("advanced", bool(on))


def set_models_dir(path: str | None) -> dict:
    """The ComfyUI models folder, when it cannot be found from the ComfyUI install (None = find it again)."""
    if path and not Path(path).is_dir():
        raise SettingsError(f"not a folder: {path}")
    return _set("models_dir", str(Path(path).resolve()) if path else None)


# ------------------------------------------------------------------ tokens

def token(name: str) -> str | None:
    """The token value (environment first, then the saved file), or None."""
    spec = TOKENS[name]
    v = (os.environ.get(spec["env"]) or "").strip()
    if v:
        return v
    v = (_read(_secrets_path()).get(name) or "").strip()
    return v or None


def status() -> dict:
    """Whether each token is set and where it comes from. NEVER the value."""
    saved = _read(_secrets_path())
    out = {}
    for name, spec in TOKENS.items():
        env = bool((os.environ.get(spec["env"]) or "").strip())
        file = bool((saved.get(name) or "").strip())
        value = token(name) or ""
        out[name] = {"label": spec["label"], "help": spec["help"], "required": spec["required"], "env": spec["env"],
                     "url": spec["url"], "set": env or file, "source": "environment" if env else "saved" if file else None,
                     "saved": file, "updated": (saved.get("_updated") or {}).get(name), "hint": _hint(value)}
    return out


def _hint(value: str) -> str | None:
    """A way to tell tokens apart without showing them (the pattern of API-key pages): the prefix and the last 4
    characters, for example 'hf_…a4f2'. None for a short or missing value."""
    if len(value) < 12:
        return None
    import re
    m = re.match(r"[A-Za-z]{1,8}[_-]", value)            # a type prefix such as 'hf_' or 'comfyui-' is not secret
    return f"{m.group(0) if m else ''}…{value[-4:]}"


def set_token(name: str, value: str) -> dict:
    if name not in TOKENS:
        raise SettingsError(f"unknown token {name!r}; use one of {list(TOKENS)}")
    value = (value or "").strip()
    if not value:
        raise SettingsError("the token is empty")
    if any(c.isspace() for c in value) or len(value) > 512:
        raise SettingsError("that does not look like a token (spaces or too long)")
    if name == "hf" and not value.startswith("hf_"):
        raise SettingsError("a Hugging Face token starts with 'hf_'")
    d = _read(_secrets_path())
    d[name] = value
    d.setdefault("_updated", {})[name] = time.time()
    _write(_secrets_path(), d, private=True)
    return status()[name]


def clear_token(name: str) -> dict:
    if name not in TOKENS:
        raise SettingsError(f"unknown token {name!r}")
    d = _read(_secrets_path())
    d.pop(name, None)
    (d.get("_updated") or {}).pop(name, None)
    _write(_secrets_path(), d, private=True)
    if name == "comfy" and get()["backend"] == "cloud" and not token("comfy"):
        _set("backend", "local")                      # no key, no cloud
    return status()[name]


def ssl_context():
    """TLS check with certifi's current root list when it is installed. The Windows store on this PC held an expired
    certificate for Let's Encrypt's new chain (civitai.com: "certificate has expired"; certifi 2026.10 is fine)."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """urllib keeps the Authorization header on a redirect to another host: a token check never follows one
    (Codex review 2026-10-22)."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):   # noqa: D401 - urllib hook
        return None


def check_token(name: str, timeout: float = 15) -> dict:
    """Ask the token's own service whether it works: {"ok": bool, "detail": ...}. Sends the token only there."""
    t = token(name)
    if not t:
        return {"ok": False, "detail": "not set"}
    if name == "hf":
        req = urllib.request.Request("https://huggingface.co/api/whoami-v2",
                                     headers={"Authorization": f"Bearer {t}", "User-Agent": "cinestudio"})
    elif name == "comfy":
        req = urllib.request.Request("https://cloud.comfy.org/api/user",
                                     headers={"X-API-Key": t, "User-Agent": "cinestudio"})
    elif name == "civitai":
        req = urllib.request.Request("https://civitai.com/api/v1/me",
                                     headers={"Authorization": f"Bearer {t}", "User-Agent": "cinestudio"})
    else:
        raise SettingsError(f"unknown token {name!r}")
    opener = urllib.request.build_opener(_NoRedirect, urllib.request.HTTPSHandler(context=ssl_context()))
    try:
        with opener.open(req, timeout=timeout) as r:
            body = json.loads(r.read() or b"{}")
        who = body.get("name") or body.get("email") or body.get("username") or ""
        return {"ok": True, "detail": f"works{(' (' + str(who) + ')') if who and name in ('hf', 'civitai') else ''}"}
    except urllib.error.HTTPError as e:
        if 300 <= e.code < 400:
            return {"ok": False, "detail": f"the service answered with a redirect (HTTP {e.code}); not followed"}
        detail = {401: "rejected: the token is wrong or was deleted",
                  403: "rejected: the token has no access",
                  429: "the key works but the Comfy plan is not active (API access needs a paid plan)"
                  if name == "comfy" else "too many requests: try again later"}.get(e.code, f"HTTP {e.code}")
        return {"ok": False, "detail": detail}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"ok": False, "detail": f"no connection: {getattr(e, 'reason', e)}"}
