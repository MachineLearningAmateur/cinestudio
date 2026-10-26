"""The distilled / full (dev) model pairs and the run-time graph features: the parity framework.

Every LTX-2.5 graph exists twice: the DISTILLED model (profile ltx25: CFG 1, 8 steps, euler_ancestral) and the FULL
model (profile ltx25dev: MultimodalGuider CFG 3 video / 7 audio + STG + rescale 0.7, the 30-step dev schedule, a real
negative). A run chooses one (run new --video-model distilled|full). The two must stay the same graph apart from what
the MODEL needs, so a fix or a feature made on one reaches the other (user, 2026-10-25: "make sure changes we make to
distilled and dev versions dont drift apart").

Three parts, all checked by tests/unit/test_model_parity.py:
- PAIRS: every graph pair, and the ALLOWED differences (the guider, the sampler, the schedule, the transformer file,
  the full model's guidance bindings). Anything else that differs is drift (pair_diff).
- FEATURES: every run-time change that cinestudio/runner.py makes to a clip graph registers itself here (the
  @graph_feature decorator on its helper, or register() for a bound value): the node ids and class types it adds, the
  inputs it edits, the values it binds. The parity test builds the clip job graphs of BOTH models with stand-ins and
  checks each feature is wired the same way in both (or declared model-specific here, with a reason).
- The generators: setup/make_layout_graphs.py, setup/make_redraw_graphs.py and setup/make_txt2img_graphs.py build both
  members of their pairs; the
  test regenerates them and compares with workflows/.
"""
from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / "workflows"

# the run option "video model" -> the graph profile
VIDEO_MODELS = {"distilled": "ltx25", "full": "ltx25dev"}
FULL_MODEL_SLOT = "ltx25_model"           # setup/models.json: the dev transformer (feature "video", both tiers)


def video_model_of(profile: str | None) -> str:
    """"full" for a dev profile, else "distilled" (old manifests have only the profile)."""
    return "full" if str(profile or "").endswith("dev") else "distilled"


# ------------------------------------------------------------------ the pairs and what may differ

# (distilled class, full class) -> why: the same node id may have another class in the two members
CLASS_SWAPS = {
    ("CFGGuider", "MultimodalGuider"): "the guider: CFG 1 (distilled) vs CFG 3 video / 7 audio + STG + rescale (full)",
    ("KSampler", "SamplerCustomAdvanced"): "a one-node KSampler (distilled) vs the custom sampler with the dev schedule",
}
# (class, input) -> why: values that may differ in nodes that are the same in both
VALUE_DIFFS = {
    ("UNETLoader", "unet_name"): "the transformer file (distilled vs dev)",
    ("KSamplerSelect", "sampler_name"): "euler_ancestral (the distilled recipe) vs euler (the dev recipe; also the "
                                        "full model's stage 2: ltx-pipelines' plain Euler, no noise between steps)",
    ("ManualSigmas", "sigmas"): "each model's own schedule (stage 1 is bound by the run; the layout pairs' stage 2 is "
                                "the same 0.909375..0 in both since 2026-10-25; v2v graphs keep their own)",
}
# binding names only the full model has (the guider's parameters, the stage-2 distilled LoRA's strength)
FULL_ONLY_BINDS = {"video_cfg", "audio_cfg", "rescale", "stg_scale", "modality_scale", "stage2_lora_strength"}
MODEL_SLOTS = {"ltx25_distilled": "ltx25_model"}       # the transformer slot of each member
CAPS_DIFFS = {"recipe", "stage2"}                       # capabilities.recipe: distilled / full; .stage2: its stage 2
# node ids only the full member has that PASS the model through (a LoRA on the stage-2 model): a link to one counts as
# a link to its "model" input, so the nodes after it compare equal with the distilled member's
FULL_ONLY_NODES = {
    "lora_dist": "Lightricks' two-stage recipe (ltx-pipelines TI2VidTwoStagesPipeline, 2026-10-25): stage 2 of the "
                 "full model = the dev transformer + the distilled LoRA, CFG 1, the distilled stage-2 sigmas; the "
                 "distilled member has the distilled transformer itself",
}
FULL_ONLY_SLOTS = {"ltx25_distilled_lora": "the distilled LoRA of the full model's stage 2 (lora_dist)"}


def through(graph: dict, value):
    """A link with the full-only pass-through nodes (FULL_ONLY_NODES) skipped: [lora_dist, 0] -> its model input."""
    seen = 0
    while isinstance(value, list) and len(value) == 2 and value[0] in FULL_ONLY_NODES and value[0] in graph \
            and seen < 8:
        value = graph[value[0]]["inputs"].get("model")
        seen += 1
    return value


@dataclass(frozen=True)
class Pair:
    variant: str
    used_by: str
    generator: str | None = None          # the script that builds BOTH members (None = hand-made graphs)
    full_only_classes: frozenset = frozenset({"GuiderParameters"})   # nodes only the full member has
    retargeted_binds: frozenset = frozenset()   # bindings that point at other nodes (KSampler vs custom sampler)

    @property
    def distilled(self) -> str:
        return f"{VIDEO_MODELS['distilled']}_{self.variant}"

    @property
    def full(self) -> str:
        return f"{VIDEO_MODELS['full']}_{self.variant}"


_KS = frozenset({"GuiderParameters", "MultimodalGuider", "RandomNoise", "KSamplerSelect", "LTXVScheduler"})
PAIRS = {p.variant: p for p in (
    Pair("layout", "Blender clips without the sheet guide (runner._layout_clip_job)", "setup/make_layout_graphs.py"),
    Pair("layouting", "Blender clips with the Ingredients sheet (runner._layout_clip_job)", "setup/make_layout_graphs.py"),
    Pair("redraw", "background style redraw (background style --profile)", "setup/make_redraw_graphs.py"),
    Pair("txt2img", "style reference pictures from text (style make-reference: cinestudio/style_ref.py)",
         "setup/make_txt2img_graphs.py"),
    Pair("v2v", "video-to-video clips, final (video templates)"),
    Pair("v2vref", "video-to-video clips, preview (video templates)"),
    Pair("ref", "control-only clips, preview", full_only_classes=_KS, retargeted_binds=frozenset({"steps", "seed"})),
    Pair("refup", "control-only clips, final", full_only_classes=_KS, retargeted_binds=frozenset({"steps", "seed"})),
    Pair("restyle", "restyle a run's sheet (--restyle)", full_only_classes=_KS,
         retargeted_binds=frozenset({"steps", "seed"})),
)}


def swap_class(cls: str) -> str:
    """One name for the two classes of an allowed swap (CFGGuider and MultimodalGuider are both 'the guider')."""
    for a, b in CLASS_SWAPS:
        if cls in (a, b):
            return f"{a}|{b}"
    return cls


def graph_diff(dist: dict, full: dict, full_only_classes=frozenset({"GuiderParameters"}), where: str = "") -> list[str]:
    """The differences between two API graphs (distilled, full) that are NOT allowed: node ids, class types (except an
    allowed swap), links and values (except VALUE_DIFFS). Inputs of a swapped node are each model's own."""
    out = []
    for k in sorted(set(dist) - set(full)):
        out.append(f"{where}node {k!r} ({dist[k]['class_type']}) is only in the distilled graph")
    for k in sorted(set(full) - set(dist)):
        if full[k]["class_type"] not in full_only_classes and k not in FULL_ONLY_NODES:
            out.append(f"{where}node {k!r} ({full[k]['class_type']}) is only in the full graph")
    for k in sorted(set(dist) & set(full)):
        a, b = dist[k], full[k]
        if a["class_type"] != b["class_type"]:
            if (a["class_type"], b["class_type"]) not in CLASS_SWAPS:
                out.append(f"{where}node {k!r} is {a['class_type']} (distilled) but {b['class_type']} (full)")
            continue                                            # a guider/sampler: each model's own inputs
        ia, ib = a.get("inputs", {}), b.get("inputs", {})
        for name in sorted(set(ia) | set(ib)):
            va, vb = ia.get(name), through(full, ib.get(name))
            if va == vb or (a["class_type"], name) in VALUE_DIFFS:
                continue
            out.append(f"{where}{k}.{name} ({a['class_type']}): {json.dumps(va)[:80]} (distilled) vs "
                       f"{json.dumps(vb)[:80]} (full)")
    return out


def bind_diff(dist: dict, full: dict, pair: Pair) -> list[str]:
    """Binding-map differences that are not allowed (inputs, model slots, capabilities, optional nodes, outputs)."""
    out = []
    di, fi = dist.get("inputs", {}), full.get("inputs", {})
    for k in sorted(set(di) ^ set(fi)):
        if k in fi and k not in di and k in FULL_ONLY_BINDS:
            continue
        out.append(f"binding {k!r} is only in the {'distilled' if k in di else 'full'} graph")
    for k in sorted(set(di) & set(fi)):
        if di[k] != fi[k] and k not in pair.retargeted_binds:
            out.append(f"binding {k!r} points at {di[k]} (distilled) but {fi[k]} (full)")
    dm = {MODEL_SLOTS.get(k, k): v for k, v in (dist.get("models") or {}).items()}
    fm = dict(full.get("models") or {})
    for k in sorted(set(dm) | set(fm)):
        if k in FULL_ONLY_SLOTS and k not in dm:
            continue
        if dm.get(k) != fm.get(k):
            out.append(f"model slot {k!r}: {dm.get(k)} (distilled) vs {fm.get(k)} (full)")
    dc, fc = dist.get("capabilities", {}), full.get("capabilities", {})
    for k in sorted(set(dc) | set(fc)):
        if dc.get(k) != fc.get(k) and k not in CAPS_DIFFS:
            out.append(f"capability {k!r}: {dc.get(k)} (distilled) vs {fc.get(k)} (full)")
    for key in ("optional_nodes", "optional_images", "outputs"):
        if dist.get(key) != full.get(key):
            out.append(f"{key}: {dist.get(key)} (distilled) vs {full.get(key)} (full)")
    return out


def load_pair(pair: Pair, workflows_dir: Path = WORKFLOWS) -> dict:
    def rd(name, suffix):
        return json.loads((Path(workflows_dir) / f"{name}{suffix}").read_text(encoding="utf-8"))
    return {"dist": rd(pair.distilled, "_api.json"), "full": rd(pair.full, "_api.json"),
            "dist_bind": rd(pair.distilled, ".bind.json"), "full_bind": rd(pair.full, ".bind.json")}


def pair_diff(pair: Pair, workflows_dir: Path = WORKFLOWS) -> list[str]:
    """Every difference of a graph pair outside its allowed differences ([] = in parity)."""
    g = load_pair(pair, workflows_dir)
    return graph_diff(g["dist"], g["full"], pair.full_only_classes) + bind_diff(g["dist_bind"], g["full_bind"], pair)


# ------------------------------------------------------------------ the run-time graph features

@dataclass(frozen=True)
class Feature:
    name: str
    where: str                            # the function (or place) in runner.py that wires it
    nodes: tuple = ()                     # node id patterns it adds (fnmatch), e.g. "relay_*"
    adds: tuple = ()                      # the class types of those nodes
    edits: tuple = ()                     # (class_type, input) of graph nodes it rewires or sets
    binds: tuple = ()                     # binding names it sets (values)
    only: str | None = None               # "full" / "distilled": wired into one model only ...
    reason: str = ""                      # ... and why (required with `only`)
    graphless: bool = False               # changes only files or text (the look image, the prompt): no node
    note: str = ""


FEATURES: dict[str, Feature] = {}


def register(name: str, where: str, **kw) -> Feature:
    f = Feature(name, where, **{k: tuple(v) if isinstance(v, (list, tuple)) else v for k, v in kw.items()})
    if f.only and not f.reason:
        raise ValueError(f"feature {name!r} is wired into the {f.only} model only: give the reason")
    old = FEATURES.get(name)
    if old is not None and old != f:
        raise ValueError(f"feature {name!r} is registered twice ({old.where}, {where})")
    FEATURES[name] = f
    return f


def graph_feature(name: str, **kw):
    """Decorator for a runner helper that changes a clip graph: registers the feature with the helper's name."""
    def deco(fn):
        register(name, fn.__name__, **kw)
        fn.graph_feature = name
        return fn
    return deco


def feature_nodes(f: Feature, graph: dict, base: dict) -> dict:
    """The nodes of `graph` that feature `f` added (ids that match its patterns and are not in `base`)."""
    return {k: n for k, n in graph.items() if k not in base and any(fnmatch.fnmatchcase(k, p) for p in f.nodes)}


def declared_classes() -> set[str]:
    return {c for f in FEATURES.values() for c in f.adds}


def declared_edits() -> set[tuple[str, str]]:
    return {(swap_class(c), i) for f in FEATURES.values() for c, i in f.edits}


def declared_binds() -> set[str]:
    return {b for f in FEATURES.values() for b in f.binds}


def job_changes(job: dict, base: dict) -> dict:
    """What a job graph adds and edits against the graph built with the same values and no feature (`base`):
    {"added": {node: class}, "edited": {(swapped class, input)}}."""
    added = {k: n["class_type"] for k, n in job.items() if k not in base}
    edited = set()
    for k in set(job) & set(base):
        ia, ib = job[k].get("inputs", {}), base[k].get("inputs", {})
        for name in set(ia) | set(ib):
            if ia.get(name) != ib.get(name):
                edited.add((swap_class(job[k]["class_type"]), name))
    return {"added": added, "edited": edited}


def undeclared(changes: dict, values: dict) -> list[str]:
    """The changes of a job graph that no registered feature declares (with what to do about each)."""
    out = []
    for node, cls in sorted(changes["added"].items()):
        owners = [f.name for f in FEATURES.values() if any(fnmatch.fnmatchcase(node, p) for p in f.nodes)]
        if not owners or cls not in declared_classes():
            out.append(f"node {node!r} ({cls}) is added by no registered feature: register it in runner.py "
                       "(@graph_feature on the helper that adds it: its node ids and class types)")
    for cls, name in sorted(changes["edited"]):
        if (cls, name) not in declared_edits():
            out.append(f"input {cls}.{name} is changed by no registered feature: declare it in the feature's edits")
    for name in sorted(set(values) - declared_binds()):
        out.append(f"value {name!r} is bound by no registered feature: register() it (runner.py, the clip bindings)")
    return out

