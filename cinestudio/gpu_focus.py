"""GPU focus mode (Windows, ADVANCED, EXPERIMENTAL, opt-in; not exposed through the MCP server): run everyday apps on the second (integrated) GPU,
so the dedicated GPU's memory stays free for ComfyUI.

    python -m cinestudio.gpu_focus status
    python -m cinestudio.gpu_focus on      # then restart the listed apps that are running
    python -m cinestudio.gpu_focus off     # restores exactly what was there before

How: Windows keeps a per-app graphics preference in
HKCU\\Software\\Microsoft\\DirectX\\UserGpuPreferences (value name = full .exe path,
data "GpuPreference=1;" = power saving GPU). This is the same setting as
Settings > System > Display > Graphics. No admin rights. CUDA work (ComfyUI) is not affected:
the preference only steers which GPU an app draws with.

Safety: `on` writes the backup (the previous values) to disk BEFORE it changes the registry, so an
interrupted `on` can always be undone with `off`. A second `on` keeps the first backup. `off`
changes only entries that still hold the value this tool wrote; an entry that the user changed or
deleted since is left alone. Paths are compared without case (Windows paths are case-insensitive).
Python, Blender, ComfyUI's compute process and games are never listed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

KEY = r"Software\Microsoft\DirectX\UserGpuPreferences"
POWER_SAVING = "GpuPreference=1;"
APP_NAMES = ["firefox", "chrome", "msedge", "brave", "opera", "vlc", "mpc-hc64", "Discord", "slack", "Spotify",
             "Teams", "ms-teams", "Code", "claude", "Comfy Desktop", "steamwebhelper", "obs64-preview"]
NEVER = {"python", "pythonw", "blender", "comfyui"}
STATE_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / "cinestudio"
BACKUP = STATE_DIR / "gpu_focus_backup.json"


def _reg():
    import winreg
    return winreg


def adapters() -> list[str]:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_VideoController | Where-Object { $_.AdapterRAM -gt 0 }).Name"],
                         capture_output=True, text=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def running_apps(names=APP_NAMES) -> dict[str, str]:
    """{exe path: process name} for running processes on the list."""
    ps = ("Get-Process | Where-Object { $_.Path } | Select-Object Name, Path | ConvertTo-Json -Compress")
    raw = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    procs = json.loads(raw) if raw.strip() else []
    wanted = {n.lower() for n in names}
    found = {}
    for p in procs if isinstance(procs, list) else [procs]:
        name = (p.get("Name") or "").lower()
        path = (p.get("Path") or "").lower()
        if r"\.local\bin" in path:          # command-line tools (e.g. the Claude Code CLI) draw nothing
            continue
        if name in wanted and name not in NEVER:
            found[p["Path"]] = p["Name"]
    return found


def read_prefs() -> dict[str, tuple[str, str]]:
    """{normalized exe path: (exact value name, data)} of the per-app GPU preferences."""
    winreg = _reg()
    prefs = {}
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, KEY) as k:
            i = 0
            while True:
                try:
                    name, value, _ = winreg.EnumValue(k, i)
                except OSError:
                    break
                prefs[_norm(name)] = (name, value)
                i += 1
    except FileNotFoundError:
        pass
    return prefs


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def _value(prefs: dict[str, tuple[str, str]], exe: str) -> str | None:
    hit = prefs.get(_norm(exe))
    return hit[1] if hit else None


def _write(prefs: dict[str, tuple[str, str]], exe: str, value: str | None) -> None:
    """Set or delete the preference of `exe`, reusing the existing value name's spelling."""
    winreg = _reg()
    name = prefs[_norm(exe)][0] if _norm(exe) in prefs else exe
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, KEY) as k:
        if value is None:
            try:
                winreg.DeleteValue(k, name)
            except FileNotFoundError:
                pass
        else:
            winreg.SetValueEx(k, name, 0, winreg.REG_SZ, value)


def _load_backup() -> dict | None:
    if not BACKUP.exists():
        return None
    data = json.loads(BACKUP.read_text(encoding="utf-8"))
    # older backups used exact paths as keys: normalize them
    raw = data.get("entries", {})
    data.setdefault("paths", {_norm(k): k for k in raw})
    data["entries"] = {_norm(k): v for k, v in raw.items()}
    return data


def _lock():
    """One change of the per-app preferences at a time (the app, the CLI and agents share them)."""
    from .locks import FileLock
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return FileLock(STATE_DIR / "gpu_focus.lock", timeout=30)


def _save_backup(backup: dict) -> None:
    import secrets
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = BACKUP.with_name(BACKUP.name + f".{secrets.token_hex(3)}.tmp")
    tmp.write_text(json.dumps(backup, indent=1), encoding="utf-8")
    os.replace(tmp, BACKUP)


def status() -> dict:
    prefs = read_prefs()
    apps = running_apps()
    backup = _load_backup()
    return {"adapters": adapters(), "focus_on": backup is not None,
            "apps": [{"exe": p, "name": n, "preference": _value(prefs, p), "running": True} for p, n in apps.items()],
            "managed": sorted(backup["paths"].values()) if backup else []}


def on() -> dict:
    with _lock():
        return _on()


def _on() -> dict:
    if len(adapters()) < 2:
        return {"ok": False, "error": "only one GPU found; nothing to move apps to"}
    prefs = read_prefs()
    backup = _load_backup() or {"entries": {}, "paths": {}}
    todo = []
    for exe in running_apps():
        key = _norm(exe)
        if key not in backup["entries"]:
            backup["entries"][key] = _value(prefs, exe)     # None = there was no entry
            backup["paths"][key] = exe
        if choice_of(_value(prefs, exe)) != "integrated":
            todo.append(exe)
    _save_backup(backup)                                    # journal first, then change the registry
    changed = []
    for exe in todo:
        _set_app(exe, "integrated")
        changed.append(exe)
    return {"ok": True, "changed": changed, "restart_these_apps": changed}


def off() -> dict:
    with _lock():
        return _off()


def _off() -> dict:
    backup = _load_backup()
    if backup is None:
        return {"ok": True, "restored": [], "note": "focus mode was not on"}
    prefs = read_prefs()
    restored, kept = [], []
    for key, previous in backup["entries"].items():
        exe = backup["paths"].get(key, key)
        current = _value(prefs, exe)
        if current == previous:
            continue                                        # nothing to undo
        ours = {backup.get("written", {}).get(key, POWER_SAVING), backup.get("pending", {}).get(key, "<none>")}
        if current not in ours:
            kept.append(exe)                                # the user changed or deleted it since: leave it
            continue
        _write(prefs, exe, previous)
        restored.append(exe)
    BACKUP.unlink()
    return {"ok": True, "restored": restored, "left_unchanged_user_edits": kept, "restart_these_apps": restored}


# ---------------------------------------------------------------- per-app choice (the Settings tab)
CHOICES = {"default": None, "integrated": "1", "gpu": "2"}     # GpuPreference: 0/none = Windows decides
SYSTEM_DIRS = (r"c:\windows", r"c:\program files\windowsapps\microsoft.windows")


def _parse(value) -> dict[str, str]:
    """'A=1;GpuPreference=2;' -> {'A': '1', 'GpuPreference': '2'} (other settings in the value are kept)."""
    out = {}
    for part in str(value or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _format(parts: dict[str, str]) -> str | None:
    return "".join(f"{k}={v};" for k, v in parts.items()) or None


def choice_of(value) -> str:
    gp = _parse(value).get("GpuPreference")
    return {"1": "integrated", "2": "gpu"}.get(gp, "default")


def adapter_roles(names: list[str] | None = None) -> dict[str, str | None]:
    """{'gpu': dedicated adapter, 'integrated': the other one} (names as Windows reports them)."""
    names = adapters() if names is None else names
    import re
    dedicated = next((n for n in names if re.search(r"NVIDIA|Radeon RX|Radeon Pro|Arc A", n, re.I)), None)
    other = next((n for n in names if n != dedicated), None)
    return {"gpu": dedicated or (names[0] if names else None), "integrated": other if dedicated else (names[1] if len(names) > 1 else None)}


def windowed_apps() -> dict[str, str]:
    """{exe path: process name} of running apps that have a window (no system parts, no tools of ours)."""
    ps = ("Get-Process | Where-Object { $_.MainWindowHandle -ne 0 -and $_.Path } | Select-Object Name, Path "
          "| ConvertTo-Json -Compress")
    raw = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    procs = json.loads(raw) if raw.strip() else []
    out = {}
    for p in procs if isinstance(procs, list) else [procs]:
        path, name = p.get("Path") or "", p.get("Name") or ""
        if not path or path.lower().startswith(SYSTEM_DIRS) or name.lower() in NEVER:
            continue
        out[path] = name
    return out


def list_apps() -> dict:
    """Running apps with a window, plus apps that already have a preference: each with its choice."""
    prefs = read_prefs()
    backup = _load_backup() or {"entries": {}, "paths": {}}
    running = windowed_apps()
    rows = {}
    for exe, name in running.items():
        rows[_norm(exe)] = {"exe": exe, "name": name, "running": True}
    for key, (exe, value) in prefs.items():
        if not isinstance(value, str) or not exe.lower().endswith(".exe") or Path(exe).stem.lower() in NEVER:
            continue
        rows.setdefault(key, {"exe": exe, "name": Path(exe).stem, "running": False})
    for key, r in rows.items():
        r["choice"] = choice_of(_value(prefs, r["exe"]))
        r["managed"] = key in backup["entries"]
    roles = adapter_roles()
    return {"adapters": roles, "can_switch": bool(roles.get("integrated")),
            "apps": sorted(rows.values(), key=lambda r: (not r["running"], r["name"].lower())),
            "managed": bool(backup["entries"])}


def set_app(exe: str, choice: str) -> dict:
    """Set the graphics choice of one app ('default', 'integrated' or 'gpu'). The value before the first
    change is kept in the backup, so `off` (Reset) restores it."""
    with _lock():
        return _set_app(exe, choice)


def _set_app(exe: str, choice: str) -> dict:
    if choice not in CHOICES:
        raise ValueError(f"choice must be one of {sorted(CHOICES)}")
    if Path(exe).stem.lower() in NEVER:
        raise ValueError(f"{Path(exe).name} must stay on the dedicated GPU (cinestudio needs it)")
    if not exe.lower().endswith(".exe"):
        raise ValueError("give the full path of an .exe file")
    if choice != "default" and not adapter_roles().get("integrated"):
        raise ValueError("only one graphics adapter found; there is nothing to switch to")
    prefs = read_prefs()
    backup = _load_backup() or {"entries": {}, "paths": {}}
    key = _norm(exe)
    current = _value(prefs, exe)
    if key not in backup["entries"]:
        backup["entries"][key] = current
        backup["paths"][key] = exe
    parts = _parse(current)
    parts.pop("GpuPreference", None)
    if CHOICES[choice]:
        parts["GpuPreference"] = CHOICES[choice]
    new = _format(parts)
    # journal first (the value about to be written is "pending"), then the registry, then "applied":
    # if the write fails or the process stops in between, Reset still knows both values as its own
    backup.setdefault("pending", {})[key] = new
    _save_backup(backup)
    _write(prefs, exe, new)
    backup.setdefault("written", {})[key] = new
    backup["pending"].pop(key, None)
    _save_backup(backup)
    return {"ok": True, "exe": exe, "choice": choice, "restart_needed": True}


def main(argv: list[str]) -> int:
    if sys.platform != "win32":
        print(json.dumps({"ok": False, "error": "GPU focus mode is Windows-only"}))
        return 1
    cmd = argv[0] if argv else "status"
    result = {"status": status, "on": on, "off": off, "list": list_apps}[cmd]()
    print(json.dumps(result, indent=1))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
