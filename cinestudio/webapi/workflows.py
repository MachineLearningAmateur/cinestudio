"""The Workflows view (an advanced mode of the app): the ComfyUI graphs behind each feature, read-only.

- the base graphs in workflows/ (with their binding maps and which features use them);
- a run's frozen graphs (<run>/inputs/workflows) and the exact graphs its jobs submitted (<run>/work/graphs, written by
  Run._comfy after every run-time change);
- the latest character, background, song and template job graphs (<home>/cache/graphs, written by character.image_job
  and the other job paths).
A submitted graph comes with its base graph and the difference: nodes added at run time, nodes whose inputs changed
(outside the binding), nodes only filled with the tool-level values (prompt, seed, model files) or bridged past an
optional node the build removed, nodes removed."""
from __future__ import annotations

import json
import re
from pathlib import Path

from aiohttp import web

from .. import runner, user_workflows
from ..workflow import WORKFLOWS, graph_cache
from .common import _bg, _ok, routes

NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,159}$")

# which features use each graph (from the code: runner.py, character.py, background.py, song.py, video_template.py).
# Retired graphs live in .cinestudio/backup/workflows_retired_<date>/ (git-ignored); runs keep their own frozen copies.
GROUPS = ("Videos", "Characters", "Backgrounds", "Styles", "Music", "Templates", "Other graphs")
_V2V = "video templates and Blender shots without a library character"
FEATURES: dict[str, tuple[str, list[str]]] = {
    "ltx25_layouting": ("Videos", ["Blender video clips with the sheet guide (Layout-To-Render + Ingredients; the default)",
                                   "Clip retakes"]),
    "ltx25_layout": ("Videos", ["Blender video clips without the sheet guide (sheet guide 0)"]),
    "ltx25dev_v2vref": ("Videos", [f"Video-to-video clips, preview (LTX-2.5 full; {_V2V})"]),
    "ltx25dev_v2v": ("Videos", [f"Video-to-video clips, final with the x2 refine (LTX-2.5 full; {_V2V})"]),
    "ltx25dev_ref": ("Videos", ["Control-only clips, preview (follow the render off; LTX-2.5 full)"]),
    "ltx25dev_refup": ("Videos", ["Control-only clips, final (follow the render off; LTX-2.5 full)"]),
    "ltx25dev_restyle": ("Videos", ["Restyle a video run's sheet with its style LoRA (--restyle; LTX-2.5 full)"]),
    "ltx25_v2vref": ("Videos", ["Video-to-video clips, preview, distilled (run new --profile ltx25)"]),
    "ltx25_v2v": ("Videos", ["Video-to-video clips, final, distilled (run new --profile ltx25)"]),
    "ltx25_ref": ("Videos", ["Control-only clips, preview, distilled (run new --profile ltx25)",
                             "Copied into every layout run with the other distilled modes"]),
    "ltx25_refup": ("Videos", ["Control-only clips, final, distilled (run new --profile ltx25)"]),
    "ltx25_restyle": ("Videos", ["Restyle a layout run's sheet with its style LoRA (--restyle; distilled)"]),
    "ltx25dev_layouting": ("Videos", ["Blender video clips with the sheet guide, full model (run new --profile ltx25dev)"]),
    "ltx25dev_layout": ("Videos", ["Blender video clips without the sheet guide, full model (run new --profile ltx25dev)"]),
    "qwenedit_edit": ("Characters", ["Character edits and region fixes", "Look images of a video run",
                                     "Backplates over a template's set"]),
    "qwenedit_angle": ("Characters", ["Character views (right, back, left) and view redos"]),
    "qwenedit_anypose": ("Characters", ["Repose: the front posed like the character's mannequin"]),
    "zimage_base": ("Characters", ["Character fronts from text (the candidates)"]),
    "ltx25_redraw": ("Backgrounds", ["Style redraw of a panorama", "Style redraw of a character sheet (Style pass)"]),
    "ltx25dev_redraw": ("Backgrounds", ["Style redraw of a panorama, full model (background style --profile ltx25dev)"]),
    "ltx25_txt2img": ("Styles", ["Style reference pictures from text with the video model and the style's LoRA "
                                 "(style make-reference)"]),
    "ltx25dev_txt2img": ("Styles", ["Style reference pictures from text, full model (style make-reference "
                                    "--video-model full)"]),
    "qwenimage_pano": ("Backgrounds", ["360 panoramas from text", "Backplates from text"]),
    "moge_pano": ("Backgrounds", ["360 depth of a panorama (the depth ground)"]),
    "acestep_base": ("Music", ["Songs from tags and lyrics"]),
    "sam3d_body": ("Templates", ["Dance templates from a video (3D body motion)"]),
    "sam3d_body_multi": ("Templates", ["Templates from a video with two people (SAM 3 finds and follows each person)"]),
}


def _check(name: str) -> str:
    if not isinstance(name, str) or not NAME.match(name) or ".." in name:
        raise web.HTTPBadRequest(text=json.dumps({"ok": False, "error": f"bad name {name!r}", "error_type": "ValueError"}),
                                 content_type="application/json")
    return name


def _not_found(what: str):
    return web.HTTPNotFound(text=json.dumps({"ok": False, "error": f"no {what}", "error_type": "NotFound"}),
                            content_type="application/json")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _features(name: str) -> tuple[str, list[str]]:
    return FEATURES.get(name) or (GROUPS[-1], ["Not used by a feature of the app"])


def _summary(graph_path: Path, name: str) -> dict:
    bind_path = graph_path.with_name(graph_path.name.replace("_api.json", ".bind.json"))
    graph = _read(graph_path)
    bind = _read(bind_path) if bind_path.is_file() else {}
    profile, _, variant = name.partition("_")
    group, uses = _features(name)
    return {"name": name, "profile": bind.get("profile") or profile, "variant": variant, "nodes": len(graph),
            "model_family": (bind.get("capabilities") or {}).get("model_family"),
            "capabilities": bind.get("capabilities") or {}, "about": bind.get("_about") or "",
            "group": group, "features": uses}


def _graph_names(folder: Path) -> list[str]:
    return sorted(p.name[:-len("_api.json")] for p in folder.glob("*_api.json"))


def _bound_fields(bind: dict) -> set[tuple[str, str]]:
    """(node, input) pairs that the binding fills (tool-level values and model files)."""
    out = set()
    for targets in list((bind.get("inputs") or {}).values()) + list((bind.get("models") or {}).values()):
        for t in targets:
            if isinstance(t, list) and len(t) >= 2:
                out.add((str(t[0]), str(t[1])))
    return out


def diff(graph: dict, base: dict | None, bind: dict | None) -> dict:
    """How a submitted graph differs from its base graph. added / removed: node ids; changed: nodes with a run-time
    change (a new class, a new link, an input outside the binding); filled: nodes that only took bound values;
    fields: node -> the inputs that differ."""
    if not base:
        return {"added": [], "removed": [], "changed": [], "filled": [], "fields": {}}
    bound = _bound_fields(bind or {})
    added = [n for n in graph if n not in base]
    removed = [n for n in base if n not in graph]
    optional = set((bind or {}).get("optional_nodes") or {})        # build removes these when their value is empty
    for opt in ((bind or {}).get("optional_images") or {}).values():     # and these when the picture is not given
        optional |= set(opt.get("nodes") or [])
        bound |= {(str(t[0]), str(t[1])) for t in opt.get("inputs") or []}

    def by_build(n: str, k: str, old) -> bool:                    # filled by the binding, or bridged past a removed
        return (n, k) in bound or (isinstance(old, list) and len(old) == 2 and old[0] in optional
                                   and old[0] not in graph)
    changed, filled, fields = [], [], {}
    for n, node in graph.items():
        if n not in base or not isinstance(node, dict):
            continue
        b = base[n] if isinstance(base[n], dict) else {}
        a_in, b_in = node.get("inputs") or {}, b.get("inputs") or {}
        diffs = sorted(k for k in set(a_in) | set(b_in) if a_in.get(k, "\0missing") != b_in.get(k, "\0missing"))
        if node.get("class_type") != b.get("class_type"):
            diffs = diffs or ["class_type"]
            changed.append(n)
        elif diffs:
            (filled if all(by_build(n, k, b_in.get(k)) for k in diffs) else changed).append(n)
        if diffs:
            fields[n] = diffs
    return {"added": added, "removed": removed, "changed": changed, "filled": filled, "fields": fields}


def _titles(*graphs: dict | None) -> dict:
    """Node titles (_meta.title) from the base graphs (a built graph has none)."""
    out = {}
    for g in graphs:
        for n, node in (g or {}).items():
            t = (node.get("_meta") or {}).get("title") if isinstance(node, dict) else None
            if t and n not in out:
                out[n] = t
    return out


def _submitted(folder: Path, job: str, frozen: Path | None) -> dict:
    """A saved submitted graph with its notes, its base graph (the run's frozen copy first, else workflows/) and the
    difference."""
    path = folder / f"{job}.json"
    if not path.is_file() or job.endswith(".meta"):
        raise _not_found(f"graph {job!r}")
    graph = _read(path)
    meta_p = folder / f"{job}.meta.json"
    meta = _read(meta_p) if meta_p.is_file() else {}
    base = bind = None
    base_name = f"{meta['profile']}_{meta['variant']}" if meta.get("profile") and meta.get("variant") else None
    base_from = None
    if base_name and NAME.match(base_name):
        for where, d in (("run", frozen), ("repository", WORKFLOWS)):
            if d is not None and (d / f"{base_name}_api.json").is_file():
                base = _read(d / f"{base_name}_api.json")
                bp = d / f"{base_name}.bind.json"
                bind = _read(bp) if bp.is_file() else None
                base_from = where
                break
    return {"job": job, "meta": meta, "graph": graph, "base": base, "bind": bind, "base_name": base_name,
            "base_from": base_from, "titles": _titles(base), **diff(graph, base, bind)}


def _saved_list(folder: Path) -> list[dict]:
    rows = []
    if not folder.is_dir():
        return rows
    for p in folder.glob("*.json"):
        if p.name.endswith(".meta.json"):
            continue
        job = p.name[:-5]
        mp = folder / f"{job}.meta.json"
        try:
            meta = _read(mp) if mp.is_file() else {}
        except (OSError, ValueError):
            meta = {}
        rows.append({"job": job, "time": meta.get("time") or p.stat().st_mtime, "nodes": meta.get("nodes"),
                     "profile": meta.get("profile"), "variant": meta.get("variant"), "attempt": meta.get("attempt"),
                     "tag": meta.get("tag")})
    return sorted(rows, key=lambda r: r["time"], reverse=True)


# ---------------------------------------------------------------- routes
@routes.get("/api/workflows")
async def workflows_list(request: web.Request):
    """The base graphs in workflows/, with the features that use each (grouped)."""
    rows = [_summary(WORKFLOWS / f"{n}_api.json", n) for n in _graph_names(WORKFLOWS) if not n.startswith("ltx25dev_")]
    for r in rows:                                      # the user's copies and which graph new jobs use
        r.update(_copies_info(r["name"]))
    order = {g: i for i, g in enumerate(GROUPS)}
    rows.sort(key=lambda r: (order.get(r["group"], 99), r["name"]))
    return _ok({"groups": list(GROUPS), "items": rows})


@routes.get("/api/workflows/{name}")
async def workflow_show(request: web.Request):
    name = _check(request.match_info["name"])
    path = WORKFLOWS / f"{name}_api.json"
    if not path.is_file():
        raise _not_found(f"workflow {name!r}")
    bp = WORKFLOWS / f"{name}.bind.json"
    bind = _read(bp) if bp.is_file() else None
    out = {**_summary(path, name), **_copies_info(name), "graph": _read(path), "bind": bind,
           "bound": sorted(user_workflows.bound_inputs(bind or {})), "tiers": _tier_names()}
    tier = request.query.get("tier")
    if tier:                                            # the graph as a run on that tier would freeze it
        out.update(_tier_view(out["graph"], bind or {}, tier))
    return _ok(out)


@routes.get("/api/runs/{id}/graphs")
async def run_graphs(request: web.Request):
    """A run's submitted graphs (newest first) and its frozen base graphs."""
    rid = _check(request.match_info["id"])
    d = runner.run_dir(rid)
    frozen = d / "inputs" / "workflows"
    return _ok({"run": rid, "submitted": _saved_list(d / "work" / "graphs"),
                "frozen": [{"name": n, "nodes": len(_read(frozen / f"{n}_api.json"))} for n in _graph_names(frozen)]
                if frozen.is_dir() else []})


@routes.get("/api/runs/{id}/graphs/{job}")
async def run_graph(request: web.Request):
    """One submitted graph of a run (?frozen=1: one of the run's frozen base graphs, by its name)."""
    rid = _check(request.match_info["id"])
    job = _check(request.match_info["job"])
    d = runner.run_dir(rid)
    frozen = d / "inputs" / "workflows"
    if request.query.get("frozen"):
        path = frozen / f"{job}_api.json"
        if not path.is_file():
            raise _not_found(f"frozen graph {job!r}")
        bp = frozen / f"{job}.bind.json"
        return _ok({"job": job, "graph": _read(path), "bind": _read(bp) if bp.is_file() else None, "base": None,
                    "titles": _titles(_read(path)), **diff({}, None, None)})
    return _ok(_submitted(d / "work" / "graphs", job, frozen if frozen.is_dir() else None))


@routes.get("/api/graphs/recent")
async def recent_graphs(request: web.Request):
    """The latest character, background, song and template job graphs (newest first)."""
    return _ok({"items": _saved_list(graph_cache())})


@routes.get("/api/graphs/recent/{name}")
async def recent_graph(request: web.Request):
    name = _check(request.match_info["name"])
    return _ok(_submitted(graph_cache(), name, None))


@routes.get("/api/settings/advanced")
async def advanced_get(request: web.Request):
    from .. import settings
    return _ok({"advanced": bool(settings.get().get("advanced"))})


@routes.post("/api/settings/advanced")
async def advanced_set(request: web.Request):
    """{"advanced": true|false}: the app's advanced mode (the Workflows view and the graph links)."""
    from .. import settings
    f = await request.json() if request.can_read_body else {}
    v = f.get("advanced")
    if not isinstance(v, bool):
        raise ValueError("advanced must be true or false")
    return _ok({"advanced": bool(settings.set_advanced(v).get("advanced"))})


# ---------------------------------------------------------------- the user's graph copies (user_workflows.py)
# a graph listed under more groups than its own: the label it has there (the user looked for the backplate graphs)
ALSO: dict[str, list[tuple[str, str]]] = {
    "qwenimage_pano": [("Backgrounds", "Backplates from text (the panorama graph, its 360 LoRA at 0)")],
    "qwenedit_edit": [("Backgrounds", "Backplates over a template's set (Qwen edit of the set render)"),
                      ("Videos", "Look images of a video run (Qwen edit of Blender frame 1)")],
}


def _copies_info(name: str) -> dict:
    try:
        copies = user_workflows.list_copies(name)
        active = user_workflows.active_choice(name)
    except (user_workflows.UserWorkflowError, OSError, ValueError):
        copies, active = [], None
    return {"copies": copies, "active": active if any(c["id"] == active for c in copies) else None,
            "also": [{"group": g, "label": lbl} for g, lbl in ALSO.get(name, [])]}


def _builtin(request: web.Request) -> str:
    name = _check(request.match_info["name"])
    if not (WORKFLOWS / f"{name}_api.json").is_file():
        raise _not_found(f"workflow {name!r}")
    return name


async def _body(request: web.Request) -> dict:
    if not request.can_read_body:
        return {}
    f = await request.json()
    if not isinstance(f, dict):
        raise ValueError("a JSON object expected")
    return f


def _copy_view(name: str, copy_id: str) -> dict:
    graph, meta = user_workflows.load_copy(name, copy_id)
    base = user_workflows.builtin_graph(name)
    bind = user_workflows.builtin_bind(name)
    row = next((c for c in user_workflows.list_copies(name) if c["id"] == copy_id), meta)
    return {"builtin": name, "id": copy_id, "meta": row, "graph": graph, "bind": bind, "builtin_graph": base,
            "titles": _titles(graph, base), "custom": user_workflows.custom_values(graph, name),
            "bound": sorted(user_workflows.bound_inputs(bind)), "problems": user_workflows.problems(graph, name),
            "active": user_workflows.active_choice(name) == copy_id, "features": _features(name)[1],
            "types": _types(graph)}


def _types(graph: dict) -> dict:
    """{node: {input: int|float|bool|str}} of the literal inputs (JSON 1.0 reaches the page as 1: the editor needs
    the real type to keep a float a float)."""
    names = {bool: "bool", int: "int", float: "float", str: "str"}
    return {n: {k: names[type(v)] for k, v in (node.get("inputs") or {}).items() if type(v) in names}
            for n, node in graph.items() if isinstance(node, dict)}


@routes.post("/api/workflows/{name}/copies")
async def copy_new(request: web.Request):
    """{label?, from?}: a new copy of the built-in (or a duplicate of the copy `from`). The built-in is only read."""
    name, f = _builtin(request), await _body(request)
    meta = user_workflows.make_copy(name, f.get("label"), f.get("from") or None)
    return _ok(_copy_view(name, meta["id"]))


@routes.get("/api/workflows/{name}/copies/{copy}")
async def copy_show(request: web.Request):
    name = _builtin(request)
    return _ok(_copy_view(name, request.match_info["copy"]))


@routes.post("/api/workflows/{name}/copies/{copy}")
async def copy_save(request: web.Request):
    """{values: {node: {input: value}}, version?}: write the literal values into the copy's JSON (inputs the job sets
    and links are refused)."""
    name, f = _builtin(request), await _body(request)
    copy_id = request.match_info["copy"]
    graph, _ = user_workflows.load_copy(name, copy_id)
    classes = {n.get("class_type") for n in graph.values() if isinstance(n, dict)}
    combos = (await _bg(object_info, classes)).get("combos") or {}
    r = user_workflows.save_values(name, copy_id, f.get("values") or {}, f.get("version"), combos)
    return _ok({**r, "copy": _copy_view(name, copy_id)})


@routes.post("/api/workflows/{name}/copies/{copy}/reset")
async def copy_reset(request: web.Request):
    """{node?}: the built-in's values again (one node, or all)."""
    name, f = _builtin(request), await _body(request)
    copy_id = request.match_info["copy"]
    r = user_workflows.reset_values(name, copy_id, f.get("node") or None)
    return _ok({**r, "copy": _copy_view(name, copy_id)})


@routes.post("/api/workflows/{name}/copies/{copy}/undo")
async def copy_undo(request: web.Request):
    name = _builtin(request)
    copy_id = request.match_info["copy"]
    r = user_workflows.undo(name, copy_id)
    return _ok({**r, "copy": _copy_view(name, copy_id)})


@routes.post("/api/workflows/{name}/copies/{copy}/rename")
async def copy_rename(request: web.Request):
    name, f = _builtin(request), await _body(request)
    return _ok(user_workflows.rename(name, request.match_info["copy"], str(f.get("label") or "")))


@routes.post("/api/workflows/{name}/copies/{copy}/delete")
async def copy_delete(request: web.Request):
    name = _builtin(request)
    return _ok(user_workflows.delete(name, request.match_info["copy"]))


@routes.post("/api/workflows/{name}/use")
async def copy_use(request: web.Request):
    """{copy: id | null}: which graph NEW jobs use (null = the built-in)."""
    name, f = _builtin(request), await _body(request)
    return _ok(user_workflows.set_choice(name, f.get("copy") or None))


# ComfyUI's choices for combo inputs (sampler, scheduler, model files), when the engine is up; cached 10 minutes
_OBJECT_INFO: dict[str, tuple[float, dict]] = {}


def _combos(spec: dict) -> dict:
    out = {}
    for section in ("required", "optional"):
        for k, v in ((spec.get("input") or {}).get(section) or {}).items():
            if isinstance(v, list) and v and isinstance(v[0], list):
                out[k] = [str(x) for x in v[0]]
            elif isinstance(v, list) and len(v) > 1 and v[0] == "COMBO" and isinstance(v[1], dict):
                out[k] = [str(x) for x in v[1].get("options") or []]
    return out


def object_info(classes: set) -> dict:
    """{available, combos: {class: {input: [choices]}}}: from ComfyUI's /object_info, or available False when it is
    not running (the editor then shows a text field)."""
    import time
    import urllib.parse
    import urllib.request
    from .. import server
    now, combos, reachable = time.time(), {}, True
    for c in sorted(x for x in classes if isinstance(x, str) and x):
        hit = _OBJECT_INFO.get(c)
        if hit and now - hit[0] < 600:
            combos[c] = hit[1]
            continue
        if not reachable:
            continue
        try:
            with urllib.request.urlopen(f"{server.comfyui_url()}/object_info/{urllib.parse.quote(c)}", timeout=3) as r:
                spec = json.loads(r.read()).get(c) or {}
        except Exception:  # noqa: BLE001 - ComfyUI is down or does not answer: no choices
            reachable = False
            continue
        combos[c] = _combos(spec)
        _OBJECT_INFO[c] = (now, combos[c])
    return {"available": bool(combos), "combos": combos}


@routes.get("/api/object_info")
async def object_info_route(request: web.Request):
    """?classes=A,B: the combo choices of these node classes (for the graph editor)."""
    classes = {c for c in (request.query.get("classes") or "").split(",") if re.match(r"^[A-Za-z0-9_.+ -]{1,120}$", c)}
    return _ok(await _bg(object_info, classes))


# ---------------------------------------------------------------- a graph as a model tier freezes it (8gb / large)
def _tier_names() -> list[str]:
    try:
        from .. import models
        return list(models.tier_names())
    except Exception:  # noqa: BLE001 - no tier list: no tier preview
        return []


def tier_graph(graph: dict, bind: dict, tier: str) -> dict:
    """The graph with the tier's model files and node settings written in (as runner._freeze_tier does for a run)."""
    from .. import models
    from ..workflow import resolved_slots
    g = json.loads(json.dumps(graph))
    files = models.bundle(tier)
    for slot, targets in resolved_slots(bind, files, "tier preview"):
        for node, field in targets:
            if node in g and field in (g[node].get("inputs") or {}):
                g[node]["inputs"][field] = files[slot]["file"]
    settings = models.graph_settings(tier) or {}
    for node in g.values():
        for field, value in (settings.get(node.get("class_type")) or {}).items():
            if field in (node.get("inputs") or {}) and not isinstance(node["inputs"][field], list):
                node["inputs"][field] = value
    return g


def _tier_view(graph: dict, bind: dict, tier: str) -> dict:
    """{graph (frozen for `tier`), base (frozen for the first tier, 8gb), changed, fields, diff_label}."""
    from .. import models
    names = _tier_names()
    tier = models.canon(tier)
    if tier not in names:
        raise ValueError(f"tier must be one of {names}")
    first = names[0]
    shown = tier_graph(graph, bind, tier)
    base = tier_graph(graph, bind, first) if tier != first else graph
    fields = {}
    for n, node in shown.items():
        b = (base.get(n) or {}).get("inputs") or {}
        d = sorted(k for k, v in (node.get("inputs") or {}).items() if b.get(k) != v)
        if d:
            fields[n] = d
    label = f"Differs from the {first} tier" if tier != first else "Set by the tier (differs from the file)"
    return {"graph": shown, "base": base, "changed": sorted(fields), "fields": fields, "added": [], "filled": [],
            "removed": [], "tier": tier, "diff_label": label}
