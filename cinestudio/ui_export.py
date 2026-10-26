"""Turn a profile's API graph into an arranged ComfyUI UI workflow (for people to open and edit).

    python -m cinestudio.ui_export ltx25 layouting [--push]

Node slots and widget order come from the running ComfyUI (/object_info), so the file matches the
installed node versions. Nodes are placed in columns by data-flow depth. Titles come from the
API graph's _meta.title. --push copies the file to ComfyUI's workflow list (cinema-studio/).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from .workflow import WORKFLOWS

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_USER_WORKFLOWS = os.environ.get(
    "COMFY_WORKFLOWS", r"D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\workflows")


def object_info(classes: set[str]) -> dict:
    info = {}
    for c in sorted(classes):
        with urllib.request.urlopen(f"{COMFY_URL}/object_info/{urllib.parse.quote(c)}", timeout=30) as r:
            info.update(json.loads(r.read()))
    return info


def _is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


def to_ui(api: dict, info: dict) -> dict:
    ids = {name: i + 1 for i, name in enumerate(api)}
    # depth = longest path from a source node, for columns
    depth: dict[str, int] = {}

    def d(name: str) -> int:
        if name not in depth:
            ups = [v[0] for v in api[name]["inputs"].values() if _is_link(v)]
            depth[name] = 1 + max((d(u) for u in ups), default=-1)
        return depth[name]

    for name in api:
        d(name)
    columns: dict[int, list[str]] = {}
    for name in api:
        columns.setdefault(depth[name], []).append(name)

    nodes, links, link_id = [], [], 0
    out_links: dict[tuple[str, int], list[int]] = {}
    pending = []
    for name, node in api.items():
        spec = info[node["class_type"]]
        req, opt = spec["input"].get("required", {}), spec["input"].get("optional", {})
        order = spec.get("input_order", {})
        names = list(order.get("required", req)) + list(order.get("optional", opt))
        defs = {**req, **opt}
        inputs, widgets = [], []
        for inp in names:
            typ = defs[inp][0]
            is_widget = (isinstance(typ, list) or typ in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")
                         or (isinstance(typ, str) and typ.startswith("COMFY_DYNAMICCOMBO")))
            value = node["inputs"].get(inp)
            if _is_link(value):
                entry = {"name": inp, "type": typ if isinstance(typ, str) else "COMBO", "link": None}
                if is_widget:
                    entry["widget"] = {"name": inp}
                inputs.append(entry)
                pending.append((name, len(inputs) - 1, value))
                if is_widget:                                   # linked widget keeps its value slot
                    dflt = defs[inp][1].get("default") if len(defs[inp]) > 1 and isinstance(defs[inp][1], dict) else None
                    widgets.append(dflt if dflt is not None else 0)
            elif is_widget:
                widgets.append(value)
                if inp in ("seed", "noise_seed"):
                    widgets.append("fixed")                     # control_after_generate
            elif value is None and inp in opt:
                inputs.append({"name": inp, "type": typ, "link": None, "shape": 7})
        if node["class_type"] in ("LoadVideo", "LoadImage"):
            widgets.append("image")                              # the upload widget
        outputs = [{"name": n, "type": t, "links": []} for n, t in zip(spec.get("output_name", spec["output"]), spec["output"])]
        col, row = depth[name], columns[depth[name]].index(name)
        nodes.append({"id": ids[name], "type": node["class_type"], "pos": [40 + col * 420, 40 + row * 260],
                      "size": [380, 200], "flags": {}, "order": ids[name], "mode": 0,
                      "title": node.get("_meta", {}).get("title", node["class_type"]),
                      "inputs": inputs, "outputs": outputs, "properties": {"Node name for S&R": node["class_type"]},
                      "widgets_values": widgets})
    by_name = {n["id"]: n for n in nodes}
    for name, slot, (src, out_slot) in pending:
        link_id += 1
        dst = by_name[ids[name]]
        src_node = by_name[ids[src]]
        typ = src_node["outputs"][out_slot]["type"]
        dst["inputs"][slot]["link"] = link_id
        src_node["outputs"][out_slot]["links"].append(link_id)
        links.append([link_id, ids[src], out_slot, ids[name], slot, typ])
    return {"id": "cinema-studio", "revision": 0, "last_node_id": len(nodes), "last_link_id": link_id,
            "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {"ds": {"scale": 0.6, "offset": [0, 0]}},
            "version": 0.4}


def main(argv: list[str]) -> int:
    profile, variant = argv[0], (argv[1] if len(argv) > 1 and not argv[1].startswith("--") else "clip")
    api = json.loads((WORKFLOWS / f"{profile}_{variant}_api.json").read_text(encoding="utf-8"))
    ui = to_ui(api, object_info({n["class_type"] for n in api.values()}))
    dst = WORKFLOWS / f"{profile}_{variant}_ui.json"
    dst.write_text(json.dumps(ui, indent=1), encoding="utf-8")
    out = {"ui": str(dst), "nodes": len(ui["nodes"]), "links": len(ui["links"])}
    if "--push" in argv:
        target = Path(COMFY_USER_WORKFLOWS) / "cinema-studio" / f"{profile}_{variant}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(ui, indent=1), encoding="utf-8")
        out["pushed"] = str(target)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
