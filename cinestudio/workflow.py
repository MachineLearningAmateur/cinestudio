"""Profile workflows: a tested API-format graph plus a binding map.

The binding map (``workflows/<profile>.bind.json``) names, for each tool-level value (prompt,
size, frames, control video, ...), the node inputs that carry it. The tool never edits nodes by
guessing; it only sets the bound inputs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / "workflows"
LOADER_FIELDS = {"ckpt_name", "unet_name", "clip_name", "clip_name1", "clip_name2", "text_encoder", "vae_name"}


class ProfileWorkflow:
    def __init__(self, profile: str, variant: str = "clip", workflows_dir: Path = WORKFLOWS,
                 models: dict[str, dict] | None = None, use_copy: bool = True) -> None:
        """variant 'clip' = first clip (optional single first frame); 'cont' = continuation clip
        guided by the previous clip's last frames (if the profile has one). `models` (slot -> file entry, a tier's
        bundle) is written into the loaders of a full-recipe graph (jobs; a run's graphs are frozen at `run new`).
        A built-in graph (workflows/) is replaced by the user's chosen copy (cinestudio/user_workflows.py) unless
        `use_copy` is False; a run's frozen folder is used as it is. `source` records which graph was loaded."""
        self.profile, self.variant = profile, variant
        graph_path = workflows_dir / f"{profile}_{variant}_api.json"
        self.source: dict[str, Any] = {"builtin": True}
        if use_copy and Path(workflows_dir).resolve() == WORKFLOWS.resolve():
            from . import user_workflows
            picked = user_workflows.chosen(f"{profile}_{variant}")
            if picked:
                graph_path, self.source = picked
        self.graph: dict[str, Any] = json.loads(graph_path.read_text(encoding="utf-8"))
        self.bind: dict[str, Any] = json.loads((workflows_dir / f"{profile}_{variant}.bind.json").read_text(encoding="utf-8"))
        if models:
            for slot, targets in resolved_slots(self.bind, models, f"{profile}_{variant}"):
                for node, field in targets:
                    self.graph[node]["inputs"][field] = models[slot]["file"]
        self._check()

    def family_key(self) -> str:
        """The model family plus the model files this graph loads: a change of files (fp8 -> bf16, distilled -> full)
        inside one family also counts as a change for server.ensure_family (a clean engine, no two models in RAM)."""
        import hashlib
        files = sorted({str(v) for n in self.graph.values() for k, v in n.get("inputs", {}).items()
                        if k in LOADER_FIELDS and isinstance(v, str)})
        fam = self.capabilities.get("model_family", self.profile)
        return f"{fam}@{hashlib.sha1('|'.join(files).encode()).hexdigest()[:10]}" if files else fam

    @staticmethod
    def has_variant(profile: str, variant: str, workflows_dir: Path = WORKFLOWS) -> bool:
        return (workflows_dir / f"{profile}_{variant}_api.json").exists()

    def _check(self) -> None:
        for name, targets in self.bind["inputs"].items():
            for node, field, *_ in targets:
                if node not in self.graph:
                    raise ValueError(f"{self.profile}: binding {name!r} points to missing node {node!r}")
                if field not in self.graph[node]["inputs"]:
                    raise ValueError(f"{self.profile}: binding {name!r} points to missing input {node}.{field}")
        if self.bind["outputs"]["video"] not in self.graph:
            raise ValueError(f"{self.profile}: output node missing")
        for node, opt in self.bind.get("optional_nodes", {}).items():
            if node not in self.graph or opt["when"] not in self.bind["inputs"]:
                raise ValueError(f"{self.profile}: optional node {node!r} or its value {opt['when']!r} is missing")
            for field in opt["bridge"].values():
                if field not in self.graph[node]["inputs"]:
                    raise ValueError(f"{self.profile}: optional node {node!r} has no input {field!r}")

        for name, opt in self.bind.get("optional_images", {}).items():
            if name not in self.bind["inputs"]:
                raise ValueError(f"{self.profile}: optional image {name!r} has no binding")
            for node in opt["nodes"]:
                if node not in self.graph:
                    raise ValueError(f"{self.profile}: optional image {name!r} names missing node {node!r}")
            for node, field in opt["inputs"]:
                if field not in self.graph.get(node, {}).get("inputs", {}):
                    raise ValueError(f"{self.profile}: optional image {name!r} names missing input {node}.{field}")

    @property
    def capabilities(self) -> dict[str, Any]:
        return self.bind.get("capabilities", {})

    def targets(self, name: str) -> list[tuple[str, str]]:
        """(node, input) pairs bound to `name` (without the optional 'not' flag)."""
        return [(t[0], t[1]) for t in self.bind["inputs"][name]]

    def check_format(self, values: dict[str, Any]) -> None:
        """Refuse a size or frame count the profile cannot make. LTX with the union IC-LoRA works on
        a half-size control latent, so its sizes are multiples of 64, not 32; frames follow the
        profile rule (LTX 8n+1, Wan 4n+1)."""
        cap = self.capabilities
        step = cap.get("size_multiple")
        for name in ("width", "height"):
            if step and name in values and int(values[name]) % step:
                v = int(values[name])
                raise ValueError(f"{self.profile}: {name} {v} is not a multiple of {step} "
                                 f"(nearest: {max(step, round(v / step) * step)})")
        rule = cap.get("frames_rule")
        if rule and "frames" in values:
            n = int(rule.split("n")[0])
            if (int(values["frames"]) - 1) % n:
                raise ValueError(f"{self.profile}: frames {values['frames']} does not follow {rule}")

    def build(self, values: dict[str, Any], files: dict | None = None) -> dict[str, Any]:
        """A fresh API graph with the bound inputs set. Unknown names are an error. A target with a
        third element "not" receives the inverted boolean (e.g. 'no_first_frame' -> a 'use' switch).

        Optional nodes (bind "optional_nodes", e.g. the style LoRA slot) stay in the graph only when
        their value ("when") is given and not empty. Otherwise the node is removed and each link to
        one of its outputs is replaced by the link on the input that "bridge" names for that output
        (a LoRA loader without a LoRA = its input model).

        Optional images (bind "optional_images", e.g. Qwen's Picture 2): when `files` (the uploads) is given and has
        no file for that image, its nodes and the inputs that take it are removed. Before this, an edit without a
        reference sent the same picture again as Picture 2: Qwen then worked on 1.5x the tokens, about 1.7x the time
        for the same picture (bench 2026-10-24, .cinestudio/gauntlet/qwen_speed)."""
        self.check_format(values)
        g = copy.deepcopy(self.graph)
        for name in values:
            if name not in self.bind["inputs"]:
                raise KeyError(f"{self.profile} has no binding for {name!r}")
        dropped = {node: opt for node, opt in self.bind.get("optional_nodes", {}).items() if not values.get(opt["when"])}
        for name, value in values.items():
            for node, field, *flags in self.bind["inputs"][name]:
                if node not in dropped:
                    g[node]["inputs"][field] = (not value) if "not" in flags else value
        for node, opt in dropped.items():
            bridge = {int(out): g[node]["inputs"][field] for out, field in opt["bridge"].items()}
            del g[node]
            for other in g.values():
                for field, link in other["inputs"].items():
                    if isinstance(link, list) and len(link) == 2 and link[0] == node:
                        if link[1] not in bridge:
                            raise ValueError(f"{self.profile}: output {link[1]} of optional node {node!r} has no bridge")
                        other["inputs"][field] = bridge[link[1]]
        if files is not None:
            for name, opt in self.bind.get("optional_images", {}).items():
                if not any(t in files for t in self.targets(name)):
                    for node in opt["nodes"]:
                        g.pop(node, None)
                    for node, field in opt["inputs"]:
                        g[node]["inputs"].pop(field, None)
        for node in g.values():
            node.pop("_meta", None)
        return g

    @property
    def output_node(self) -> str:
        return self.bind["outputs"]["video"]


def resolved_slots(bind: dict, files: dict[str, dict], name: str) -> list[tuple[str, list]]:
    """The binding's model slots that take a file from `files` (a tier bundle). The RECIPE (full or distilled) chooses
    the sampling; the tier chooses the precision of every file, so distilled graphs take it too (2026-10-22: the 24gb
    tier downloads Gemma bf16 only, and the distilled layout graphs kept asking for the int8 file). A full graph must
    find every slot in the tier; another graph keeps its own file for a slot the tier does not have (old runs)."""
    out = []
    for slot, targets in (bind.get("models") or {}).items():
        if slot in files:
            out.append((slot, targets))
        elif bind.get("capabilities", {}).get("recipe") == "full":
            raise ValueError(f"{name}: the model tier has no file for slot {slot!r}")
    return out


def freeze_models(workflows_dir: Path, files: dict[str, dict]) -> dict[str, dict]:
    """Write the model files of a tier into the graphs of a run's own copy (`workflows_dir`), so a settings change
    later cannot change the run. Every graph whose binding names model slots takes the tier's file per slot ("models":
    {"ltx_model": [["ckpt", "ckpt_name"], ...]}; see resolved_slots). Returns {slot: {dir, file, size, sha256}}."""
    used: dict[str, dict] = {}
    for bind_path in sorted(Path(workflows_dir).glob("*.bind.json")):
        bind = json.loads(bind_path.read_text(encoding="utf-8"))
        graph_path = bind_path.with_name(bind_path.name.replace(".bind.json", "_api.json"))
        slots = resolved_slots(bind, files, graph_path.name)
        if not slots:
            continue
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        for slot, targets in slots:
            e = files[slot]
            for node, field in targets:
                if node not in graph or field not in graph[node]["inputs"]:
                    raise ValueError(f"{bind_path.name}: model slot {slot!r} points to missing input {node}.{field}")
                graph[node]["inputs"][field] = e["file"]
            used[slot] = {k: e[k] for k in ("dir", "file", "size", "sha256")}
        graph_path.write_text(json.dumps(graph, indent=1), encoding="utf-8")
    return used


def freeze_settings(workflows_dir: Path, settings: dict[str, dict]) -> dict[str, dict]:
    """Write a tier's node settings ({class_type: {input: value}}) into every graph of a run's own copy: each node of
    that class that has the input takes the value. Returns {class_type: {input: value}} of what was written."""
    used: dict[str, dict] = {}
    if not settings:
        return used
    for graph_path in sorted(Path(workflows_dir).glob("*_api.json")):
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        changed = False
        for node in graph.values():
            want = settings.get(node.get("class_type")) if isinstance(node, dict) else None
            for field, value in (want or {}).items():
                if field in node["inputs"] and not isinstance(node["inputs"][field], list):
                    node["inputs"][field] = value
                    used.setdefault(node["class_type"], {})[field] = value
                    changed = True
        if changed:
            graph_path.write_text(json.dumps(graph, indent=1), encoding="utf-8")
    return used


# ---------------------------------------------------------------- the submitted graphs (the app's Workflows view)
GRAPH_KEEP = 200                                   # character / background / song job graphs kept in <home>/cache/graphs


def safe_graph_name(name: str) -> str:
    """A file stem from a job name or tag: only [A-Za-z0-9_.-] (anything else -> "-"), no leading dot."""
    import re
    s = re.sub(r"[^A-Za-z0-9_.-]", "-", str(name or "")).lstrip(".-")[:120]
    return s or "job"


def save_graph(folder: Path, name: str, graph: dict, meta: dict | None = None, files: dict | None = None,
               keep: int | None = None) -> Path | None:
    """Write the exact API graph a job submits to <folder>/<name>.json (a file ComfyUI can open) and its notes
    (profile, variant, time, the uploaded files per node input) to <name>.meta.json. The newest per name is kept;
    `keep` = how many graphs the folder keeps (the oldest go). Never raises: a failed write must not stop a job."""
    import logging
    import os
    import time
    try:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = safe_graph_name(name)
        out = folder / f"{stem}.json"
        tmp = folder / f".{stem}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(graph, indent=1), encoding="utf-8")
        os.replace(tmp, out)
        note = {**(meta or {}), "name": stem, "time": time.time(), "nodes": len(graph),
                "files": {f"{k[0]}.{k[1]}" if isinstance(k, tuple) else str(k): str(v) for k, v in (files or {}).items()}}
        (folder / f"{stem}.meta.json").write_text(json.dumps(note, indent=1), encoding="utf-8")
        if keep:
            graphs = sorted((p for p in folder.glob("*.json") if not p.name.endswith(".meta.json")),
                            key=lambda p: p.stat().st_mtime, reverse=True)
            for old in graphs[keep:]:
                old.unlink(missing_ok=True)
                old.with_name(old.name[:-5] + ".meta.json").unlink(missing_ok=True)
        return out
    except Exception as exc:  # noqa: BLE001 - only a note in the log
        logging.getLogger(__name__).warning("could not save the graph of %s: %s", name, exc)
        return None


def graph_cache() -> Path:
    """<home>/cache/graphs: the graphs of the latest character, background, song and template jobs."""
    from .runner import home
    return home() / "cache" / "graphs"
