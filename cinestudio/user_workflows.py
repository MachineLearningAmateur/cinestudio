"""The user's own copies of the built-in graphs (the Workflows view's advanced mode, user decision 2026-10-25).

The built-in graphs in workflows/ are read-only. "Make a copy" writes a USER graph:

    <home>/workflows/<builtin>/<copy_id>.json         the API graph (the same node ids; the user changes literal values)
    <home>/workflows/<builtin>/<copy_id>.meta.json    {id, label, made_from, made_from_sha1, created, updated, version}
    <home>/workflows/<builtin>/history/<copy_id>/     the previous versions (newest 20: "Undo last save")
    <home>/workflows/choices.json                     {builtin: copy_id}: which graph NEW jobs use (absent = built-in)

ProfileWorkflow loads the chosen copy instead of the built-in for every new job (character, background, song and
template jobs, and the graphs a run copies at `run new`); the binding map of the built-in still applies, so the job's
values (prompt, seed, pictures, model files) win. A copy may change only literal values, never a bound input or a link,
so it cannot break the binding. A copy that no longer fits the built-in (a bound node or the output node missing, a
node of another class: an app update changed the built-in) is refused with a clear error.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

from .workflow import WORKFLOWS

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
HISTORY_KEEP = 20
LABEL_MAX = 80


class UserWorkflowError(ValueError):
    pass


# ---------------------------------------------------------------- places
def root() -> Path:
    """<home>/workflows (CINESTUDIO_USER_WORKFLOWS wins: the tests use their own folder)."""
    env = os.environ.get("CINESTUDIO_USER_WORKFLOWS")
    if env:
        return Path(env)
    from .runner import home
    return home() / "workflows"


def _check_builtin(builtin: str, workflows_dir: Path = WORKFLOWS) -> str:
    if not isinstance(builtin, str) or not NAME_RE.match(builtin) or ".." in builtin:
        raise UserWorkflowError(f"bad workflow name {builtin!r}")
    if not (workflows_dir / f"{builtin}_api.json").is_file() or not (workflows_dir / f"{builtin}.bind.json").is_file():
        raise UserWorkflowError(f"no built-in workflow {builtin!r}")
    return builtin


def _check_id(copy_id: str) -> str:
    if not isinstance(copy_id, str) or not ID_RE.match(copy_id):
        raise UserWorkflowError(f"bad copy id {copy_id!r}")
    return copy_id


def _dir(builtin: str) -> Path:
    return root() / _check_builtin(builtin)


def _paths(builtin: str, copy_id: str) -> tuple[Path, Path]:
    d = _dir(builtin)
    _check_id(copy_id)
    return d / f"{copy_id}.json", d / f"{copy_id}.meta.json"


def _read(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_atomic(p: Path, data: Any, indent: int | None = 1) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{secrets.token_hex(3)}.tmp")
    tmp.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    os.replace(tmp, p)


def _indent_of(path: Path) -> int | None:
    """The indent style of a JSON file (so a saved copy keeps the built-in's formatting)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 1
    m = re.search(r"\n( +)\"", text)
    return len(m.group(1)) if m else None


def sha1_of(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def builtin_graph(builtin: str) -> dict:
    return _read(WORKFLOWS / f"{_check_builtin(builtin)}_api.json")


def builtin_bind(builtin: str) -> dict:
    return _read(WORKFLOWS / f"{_check_builtin(builtin)}.bind.json")


# ---------------------------------------------------------------- what the job sets (read-only in the editor)
def bound_inputs(bind: dict) -> set[tuple[str, str]]:
    """(node, input) pairs the job fills: tool-level values, model slots, optional pictures' inputs."""
    out: set[tuple[str, str]] = set()
    for targets in [*(bind.get("inputs") or {}).values(), *(bind.get("models") or {}).values()]:
        for t in targets:
            if isinstance(t, list) and len(t) >= 2:
                out.add((str(t[0]), str(t[1])))
    for opt in (bind.get("optional_images") or {}).values():
        for t in opt.get("inputs") or []:
            out.add((str(t[0]), str(t[1])))
    return out


def is_link(v: Any) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int)) and isinstance(v[1], int)


def editable(v: Any) -> bool:
    """Only literal widget values are editable: numbers, text, booleans (a combo choice is text)."""
    return isinstance(v, (bool, int, float, str)) and not is_link(v)


def problems(copy_graph: dict, builtin: str) -> list[str]:
    """Why a copy cannot be used with the CURRENT built-in's binding (empty = it fits)."""
    bind, base = builtin_bind(builtin), builtin_graph(builtin)
    out = []
    for node, field in sorted(bound_inputs(bind)):
        if node not in copy_graph:
            if any(node in (o.get("nodes") or []) for o in (bind.get("optional_images") or {}).values()):
                continue
            out.append(f"node {node!r} (set by the job) is missing")
        elif node in base and copy_graph[node].get("class_type") != base[node].get("class_type"):
            out.append(f"node {node!r} is a {copy_graph[node].get('class_type')}, the built-in has a "
                       f"{base[node].get('class_type')}")
        elif field not in (copy_graph[node].get("inputs") or {}) and field in (base.get(node, {}).get("inputs") or {}):
            out.append(f"input {node}.{field} (set by the job) is missing")
    for node in {*(bind.get("outputs") or {}).values(), *(bind.get("optional_nodes") or {})}:
        if node not in copy_graph:
            out.append(f"node {node!r} is missing")
    return out


# ---------------------------------------------------------------- copies
def _meta(builtin: str, copy_id: str) -> dict:
    g, mp = _paths(builtin, copy_id)
    if not g.is_file():
        raise UserWorkflowError(f"no copy {copy_id!r} of {builtin}")
    try:
        meta = _read(mp) if mp.is_file() else {}
    except (OSError, ValueError):
        meta = {}
    return {"id": copy_id, "label": copy_id, "made_from": builtin, **meta}


def list_copies(builtin: str) -> list[dict]:
    d = _dir(builtin)
    if not d.is_dir():
        return []
    cur = sha1_of(WORKFLOWS / f"{builtin}_api.json")
    rows = []
    for p in d.glob("*.json"):
        if p.name.endswith(".meta.json") or not ID_RE.match(p.stem):
            continue
        meta = _meta(builtin, p.stem)
        meta["stale"] = bool(meta.get("made_from_sha1")) and meta.get("made_from_sha1") != cur
        meta["history"] = len(list((d / "history" / p.stem).glob("*.json"))) if (d / "history" / p.stem).is_dir() else 0
        rows.append(meta)
    return sorted(rows, key=lambda r: r.get("created") or 0)


def _new_id() -> str:
    return time.strftime("c%Y%m%d-%H%M%S-") + secrets.token_hex(2)


def make_copy(builtin: str, label: str | None = None, from_copy: str | None = None) -> dict:
    """A new copy of the built-in (or a duplicate of another copy). The built-in file is only read."""
    _check_builtin(builtin)
    src = _paths(builtin, from_copy)[0] if from_copy else WORKFLOWS / f"{builtin}_api.json"
    if not src.is_file():
        raise UserWorkflowError(f"no copy {from_copy!r} of {builtin}")
    graph = _read(src)
    n = len(list_copies(builtin)) + 1
    base_label = _meta(builtin, from_copy)["label"] if from_copy else None
    label = _clean_label(label or (f"{base_label} (copy)" if base_label else f"My copy {n}"))
    copy_id = _new_id()
    while _paths(builtin, copy_id)[0].exists():
        copy_id = _new_id()
    g, mp = _paths(builtin, copy_id)
    now = time.time()
    _write_atomic(g, graph, _indent_of(WORKFLOWS / f"{builtin}_api.json"))
    origin = _meta(builtin, from_copy) if from_copy else {}
    meta = {"id": copy_id, "label": label, "made_from": builtin,
            "made_from_sha1": origin.get("made_from_sha1") or sha1_of(WORKFLOWS / f"{builtin}_api.json"),
            "created": now, "updated": now, "version": 1}
    if from_copy:
        meta["duplicated_from"] = from_copy
    _write_atomic(mp, meta)
    return meta


def _clean_label(label: str) -> str:
    label = " ".join(str(label or "").split())[:LABEL_MAX]
    if not label:
        raise UserWorkflowError("the name is empty")
    return label


def load_copy(builtin: str, copy_id: str) -> tuple[dict, dict]:
    g, _ = _paths(builtin, copy_id)
    meta = _meta(builtin, copy_id)
    return _read(g), meta


def rename(builtin: str, copy_id: str, label: str) -> dict:
    _, mp = _paths(builtin, copy_id)
    meta = _meta(builtin, copy_id)
    meta.update(label=_clean_label(label), updated=time.time())
    _write_atomic(mp, {k: v for k, v in meta.items() if k not in ("stale", "history")})
    return meta


def delete(builtin: str, copy_id: str) -> dict:
    """Delete a copy (its history too). A copy that new jobs use: they use the built-in again."""
    g, mp = _paths(builtin, copy_id)
    if not g.is_file():
        raise UserWorkflowError(f"no copy {copy_id!r} of {builtin}")
    was_active = active_choice(builtin) == copy_id
    if was_active:
        set_choice(builtin, None)
    g.unlink()
    mp.unlink(missing_ok=True)
    h = _dir(builtin) / "history" / copy_id
    if h.is_dir():
        for p in h.glob("*.json"):
            p.unlink(missing_ok=True)
        try:
            h.rmdir()
        except OSError:
            pass
    return {"deleted": copy_id, "was_active": was_active}


def _coerce(old: Any, new: Any, where: str) -> Any:
    """The new value with the type of the old one (int, float, bool, text)."""
    if isinstance(old, bool):
        if not isinstance(new, bool):
            raise UserWorkflowError(f"{where}: true or false")
        return new
    if isinstance(old, int):
        if isinstance(new, bool) or not isinstance(new, (int, float)) or not math.isfinite(new) or int(new) != new:
            raise UserWorkflowError(f"{where}: a whole number")
        return int(new)
    if isinstance(old, float):
        if isinstance(new, bool) or not isinstance(new, (int, float)) or not math.isfinite(float(new)):
            raise UserWorkflowError(f"{where}: a number")
        return float(new)
    if isinstance(old, str):
        if not isinstance(new, str):
            raise UserWorkflowError(f"{where}: text")
        return new
    raise UserWorkflowError(f"{where}: this input cannot be changed")


def _big_change(old: Any, new: Any) -> bool:
    if isinstance(old, bool) or not isinstance(old, (int, float)):
        return False
    if old == 0:
        return abs(new) > 1
    return new / old > 10 or new / old < 0.1


def save_values(builtin: str, copy_id: str, values: dict, version: int | None = None,
                choices: dict | None = None) -> dict:
    """Write {node: {input: value}} into the copy's JSON (atomic; the previous version goes to the history). Refused:
    an unknown node or input, a link, an input the job sets, a value of another type. Warnings: a big change of a number,
    a choice ComfyUI does not list (`choices`: {class: {input: [choices]}} from /object_info, when known)."""
    g_path, mp = _paths(builtin, copy_id)
    graph, meta = load_copy(builtin, copy_id)
    if version is not None and int(version) != int(meta.get("version") or 1):
        raise UserWorkflowError("the copy was saved elsewhere since you opened it: open it again")
    if not isinstance(values, dict) or not values:
        raise UserWorkflowError("nothing to save")
    bound = bound_inputs(builtin_bind(builtin))
    warnings, changed = [], 0
    new = json.loads(json.dumps(graph))
    for node, fields in values.items():
        if node not in new:
            raise UserWorkflowError(f"node {node!r} is not in this graph (it changed: open it again)")
        if not isinstance(fields, dict):
            raise UserWorkflowError(f"node {node!r}: {{input: value}} expected")
        inputs = new[node].setdefault("inputs", {})
        for field, value in fields.items():
            where = f"{node}.{field}"
            if field not in inputs:
                raise UserWorkflowError(f"{where} is not an input of this node")
            if (node, field) in bound:
                raise UserWorkflowError(f"{where} is set by the job (its prompt, seed, picture or model): not editable")
            old = inputs[field]
            if not editable(old):
                raise UserWorkflowError(f"{where} is a link or a list: not editable (no rewiring)")
            v = _coerce(old, value, where)
            if v == old and type(v) is type(old):
                continue
            if _big_change(old, v):
                warnings.append(f"{where}: {old} -> {v} is a big change")
            opts = ((choices or {}).get(new[node].get("class_type")) or {}).get(field)
            if isinstance(v, str) and opts and v not in opts:
                warnings.append(f"{where}: ComfyUI does not list {v!r} as a choice")
            inputs[field] = v
            changed += 1
    if not changed:
        return {"saved": False, "changed": 0, "warnings": warnings, "meta": meta}
    hist = _dir(builtin) / "history" / copy_id
    hist.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S-") + f"{int(time.time() * 1000) % 1000:03d}"
    (hist / f"{stamp}.json").write_bytes(g_path.read_bytes())
    for old_p in sorted(hist.glob("*.json"))[:-HISTORY_KEEP]:
        old_p.unlink(missing_ok=True)
    _write_atomic(g_path, new, _indent_of(g_path))
    meta.update(updated=time.time(), version=int(meta.get("version") or 1) + 1)
    _write_atomic(mp, {k: v for k, v in meta.items() if k not in ("stale", "history")})
    return {"saved": True, "changed": changed, "warnings": warnings, "meta": meta}


def reset_values(builtin: str, copy_id: str, node: str | None = None) -> dict:
    """Put the built-in's literal values back (one node, or every node the built-in also has)."""
    graph, meta = load_copy(builtin, copy_id)
    base, bound = builtin_graph(builtin), bound_inputs(builtin_bind(builtin))
    nodes = [node] if node else list(graph)
    if node and node not in graph:
        raise UserWorkflowError(f"node {node!r} is not in this graph")
    values = {}
    for n in nodes:
        if n not in base:
            continue
        for k, v in (graph[n].get("inputs") or {}).items():
            bv = (base[n].get("inputs") or {}).get(k)
            if (n, k) in bound or not editable(v) or not editable(bv) or bv == v:
                continue
            values.setdefault(n, {})[k] = bv
    if not values:
        return {"saved": False, "changed": 0, "warnings": [], "meta": meta}
    return save_values(builtin, copy_id, values)


def undo(builtin: str, copy_id: str) -> dict:
    """Go back to the version before the last save."""
    g_path, mp = _paths(builtin, copy_id)
    meta = _meta(builtin, copy_id)
    hist = _dir(builtin) / "history" / copy_id
    versions = sorted(hist.glob("*.json")) if hist.is_dir() else []
    if not versions:
        raise UserWorkflowError("nothing to undo")
    last = versions[-1]
    _write_atomic(g_path, _read(last), _indent_of(g_path))
    last.unlink()
    meta.update(updated=time.time(), version=int(meta.get("version") or 1) + 1)
    _write_atomic(mp, {k: v for k, v in meta.items() if k not in ("stale", "history")})
    return {"undone": True, "left": len(versions) - 1, "meta": meta}


def custom_values(graph: dict, builtin: str) -> dict[str, list[str]]:
    """{node: [inputs]} whose literal value differs from the built-in's."""
    base = builtin_graph(builtin)
    out: dict[str, list[str]] = {}
    for n, node in graph.items():
        if n not in base or not isinstance(node, dict):
            continue
        b_in = base[n].get("inputs") or {}
        diff = [k for k, v in (node.get("inputs") or {}).items() if editable(v) and k in b_in and b_in[k] != v]
        if diff:
            out[n] = sorted(diff)
    return out


# ---------------------------------------------------------------- which graph new jobs use
def _choices_path() -> Path:
    return root() / "choices.json"


def choices() -> dict[str, str]:
    try:
        d = _read(_choices_path())
        return {k: v for k, v in d.items() if isinstance(v, str)} if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def active_choice(builtin: str) -> str | None:
    return choices().get(builtin)


def set_choice(builtin: str, copy_id: str | None) -> dict:
    """New jobs use this copy (None = the built-in). A copy that does not fit the built-in is refused."""
    _check_builtin(builtin)
    d = choices()
    if copy_id:
        graph, meta = load_copy(builtin, copy_id)
        bad = problems(graph, builtin)
        if bad:
            raise UserWorkflowError(f"the copy {meta['label']!r} does not fit the built-in {builtin} any more "
                                    f"({'; '.join(bad[:3])}): make a new copy")
        d[builtin] = copy_id
    else:
        d.pop(builtin, None)
    _write_atomic(_choices_path(), d)
    return {"builtin": builtin, "active": d.get(builtin)}


def chosen(builtin: str) -> tuple[Path, dict] | None:
    """The chosen copy's file and its source record for a NEW job, or None (the built-in). Raises when the chosen copy
    is gone or no longer fits (the job must not run silently with another graph)."""
    copy_id = active_choice(builtin)
    if not copy_id:
        return None
    try:
        g, _ = _paths(builtin, copy_id)
        graph, meta = load_copy(builtin, copy_id)
    except (UserWorkflowError, OSError, ValueError) as exc:
        raise UserWorkflowError(f"{builtin}: the chosen graph copy {copy_id!r} cannot be read ({exc}); choose "
                                "Built-in or another copy in the Workflows view") from exc
    bad = problems(graph, builtin)
    if bad:
        raise UserWorkflowError(f"{builtin}: the chosen graph copy {meta['label']!r} was made from an older built-in "
                                f"and does not fit it ({'; '.join(bad[:3])}); choose Built-in or make a new copy in "
                                "the Workflows view")
    return g, {"copy": copy_id, "label": meta.get("label"), "sha1": sha1_of(g),
               "made_from_sha1": meta.get("made_from_sha1")}
