"""The distilled / full (dev) parity framework (cinestudio/model_pairs.py): the graph pairs differ only where the model
needs it, the generators reproduce workflows/, and every run-time clip feature is wired into BOTH models.

When a test here fails, the message says what to do. In short: change both members of a pair through their generator
(setup/make_layout_graphs.py, setup/make_redraw_graphs.py,
setup/make_txt2img_graphs.py); register every run-time graph change in runner.py
(@graph_feature on the helper, or _pairs.register for a bound value); a difference that the MODEL needs goes into
model_pairs.py (CLASS_SWAPS, VALUE_DIFFS, FULL_ONLY_BINDS, or a feature's only= with a reason).
"""
import ast
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from cinestudio import model_pairs as mp
from cinestudio import runner
from cinestudio.workflow import ProfileWorkflow

ROOT = Path(__file__).resolve().parents[2]
PROFILES = (mp.VIDEO_MODELS["distilled"], mp.VIDEO_MODELS["full"])
FIX = ("Change both graphs through their generator, or declare a difference the model needs in "
       "cinestudio/model_pairs.py (CLASS_SWAPS, VALUE_DIFFS, FULL_ONLY_BINDS, the pair's full_only_classes) with a reason.")


@pytest.fixture(scope="module", autouse=True)
def legacy_graph_metadata():
    """Exercise frozen historical dev graphs with synthetic metadata, without supporting their weights."""
    from cinestudio import models
    original = models.bundle
    with pytest.MonkeyPatch.context() as patch:
        def bundle(tier):
            result = original(tier)
            result["ltx25_model"] = {"dir": "diffusion_models", "file": "historical-dev.safetensors", "size": 1, "sha256": "0" * 64}
            result["ltx25_distilled_lora"] = {"dir": "loras", "file": "ltx-2.5-22b-distilled-lora-450-bf16.safetensors", "size": 1, "sha256": "0" * 64}
            return result
        patch.setattr(models, "bundle", bundle)
        yield


# ------------------------------------------------------------------ the graph pairs

@pytest.mark.parametrize("variant", sorted(mp.PAIRS))
def test_each_graph_pair_differs_only_where_the_model_needs_it(variant):
    pair = mp.PAIRS[variant]
    diff = mp.pair_diff(pair)
    assert not diff, f"{pair.distilled} and {pair.full} drifted apart:\n  " + "\n  ".join(diff) + "\n" + FIX


def test_every_ltx25_graph_has_its_pair():
    """A new LTX-2.5 graph needs its other member and a PAIRS entry (or the parity test cannot see it)."""
    names = {p.name[:-len("_api.json")] for p in mp.WORKFLOWS.glob("ltx25*_api.json")}
    variants = {n.split("_", 1)[1] for n in names}
    for v in sorted(variants):
        assert v in mp.PAIRS, f"workflows has ltx25*_{v} but model_pairs.PAIRS has no pair {v!r}: add it"
        for prof in PROFILES:
            assert f"{prof}_{v}" in names, f"{prof}_{v} is missing: the pair {v!r} needs both models"


def _generated(pair):
    sys.path.insert(0, str(ROOT / "setup"))
    try:
        if pair.generator == "setup/make_layout_graphs.py":
            import make_layout_graphs as gen
            return {prof: gen.build(pair.variant == "layouting", f"{prof}_v2v") for prof in PROFILES}
        if pair.generator == "setup/make_txt2img_graphs.py":
            import make_txt2img_graphs as gen
        else:
            import make_redraw_graphs as gen
        return {prof: gen.build(prof) for prof in PROFILES}
    finally:
        sys.path.remove(str(ROOT / "setup"))


@pytest.mark.parametrize("variant", sorted(v for v, p in mp.PAIRS.items() if p.generator))
def test_the_generator_reproduces_both_members(variant):
    """workflows/ is the generator's output: regenerating gives the committed files (both models), bind maps too."""
    pair = mp.PAIRS[variant]
    for prof, (graph, bind) in _generated(pair).items():
        name = f"{prof}_{variant}"
        on_disk = json.loads((mp.WORKFLOWS / f"{name}_api.json").read_text(encoding="utf-8"))
        bind_disk = json.loads((mp.WORKFLOWS / f"{name}.bind.json").read_text(encoding="utf-8"))
        changed = sorted(k for k in set(graph) | set(on_disk) if graph.get(k) != on_disk.get(k))
        assert graph == on_disk, (f"workflows/{name}_api.json is not what {pair.generator} makes (nodes {changed}): "
                                  f"edit the generator, not the file, then run `python {pair.generator}`")
        assert bind == bind_disk, (f"workflows/{name}.bind.json is not what {pair.generator} makes: edit the "
                                   f"generator, then run `python {pair.generator}`")


@pytest.mark.parametrize("tier", ["8gb", "large"])
@pytest.mark.parametrize("variant", ["layout", "layouting"])
def test_a_run_freezes_the_tier_into_both_members_the_same_way(tmp_path, tier, variant):
    """The tier's model files and node settings (large: VAEDecodeTiled 1024/256/128/48) reach both models."""
    from cinestudio import models
    pair = mp.PAIRS[variant]
    for prof in PROFILES:
        for suffix in ("_api.json", ".bind.json"):
            shutil.copy2(mp.WORKFLOWS / f"{prof}_{variant}{suffix}", tmp_path / f"{prof}_{variant}{suffix}")
    frozen = runner._freeze_tier(tmp_path, tier)
    diff = mp.pair_diff(pair, tmp_path)
    assert not diff, f"after the {tier} tier freeze:\n  " + "\n  ".join(diff)
    want = models.graph_settings(tier)
    for prof in PROFILES:
        g = json.loads((tmp_path / f"{prof}_{variant}_api.json").read_text(encoding="utf-8"))
        for cls, vals in want.items():
            nodes = [n for n in g.values() if n["class_type"] == cls]
            assert nodes and all(n["inputs"][k] == v for n in nodes for k, v in vals.items()), (prof, cls)
    assert mp.FULL_MODEL_SLOT in frozen["files"] and "ltx25_distilled" in frozen["files"]


# ------------------------------------------------------------------ the clip job graphs of both models

SCENARIOS = {
    # clip 1, final, the sheet guide, a style LoRA, a keyframe look, the large tier (decode tiles)
    "clip1-final": dict(variant="layouting", clip=1, quality="final", tier="large"),
    # clip 2, final: relay + audio carry + AdaIN (clip 1's latent) + look chain + keyframe + pose guide in both
    # stages + own refine sigmas + guidance (the full model only)
    "clip2-final": dict(variant="layouting", clip=2, quality="final",
                        m=dict(pose_strength=0.5, pose_stage2=True, refine_sigmas="0.6, 0.3, 0.0", guidance=4.0,
                               chain_colour="off")),
    # clip 2, preview (stage 1 only), no sheet guide (the layout graph), AdaIN toward the look image
    "clip2-preview-layout": dict(variant="layout", clip=2, quality="preview",
                                 m=dict(ingredients_strength=0.0, adain_ref="look", adain=0.5)),
    # clip 1, preview
    "clip1-preview": dict(variant="layouting", clip=1, quality="preview"),
    # a retake of clip 1 that ends where clip 2 begins (end anchor), another seed and its own keyframe image
    "take1-final": dict(variant="layouting", clip=1, quality="final", take=True, m=dict(seed=7)),
    # a retake of clip 2 in preview, without a style LoRA and keyframes off
    "take2-preview": dict(variant="layouting", clip=2, quality="preview", take=True,
                          m=dict(style_lora=None, keyframe="off")),
}


def _png(path, value=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((8, 8, 3), value, np.uint8))
    return path


def _stub(monkeypatch, tmp_path):
    frame = _png(tmp_path / "frame.png")
    v = runner.video
    monkeypatch.setattr(v, "frame_png", lambda src, dst, index, *a, **k: _png(Path(dst)))
    monkeypatch.setattr(v, "match_lightness", lambda *a, **k: {"gain": 1.0})
    monkeypatch.setattr(v, "background_lightness", lambda *a, **k: (50.0, None))
    monkeypatch.setattr(v, "sharpness", lambda *a, **k: 10.0)
    monkeypatch.setattr(v, "sharpen_to", lambda *a, **k: {"amount": 0.5})
    monkeypatch.setattr(v, "unsharp_png", lambda path, *a, **k: Path(path))
    monkeypatch.setattr(v, "image_video", lambda src, dst, fps: Path(dst).write_bytes(b"v"))
    monkeypatch.setattr(v, "last_frames", lambda src, dst, n: Path(dst).write_bytes(b"v"))
    monkeypatch.setattr(v, "check_clip", lambda *a, **k: [])
    monkeypatch.setattr(runner.passes, "load", lambda d: {"rgb": [str(frame)] * 400, "mask": [str(frame)] * 400})
    monkeypatch.setattr(runner, "relay_latent", lambda prefix, keep_dir=None: Path(prefix).name + ".latent")
    monkeypatch.setattr(runner.Run, "_relay_inputs", lambda self, files: None)
    monkeypatch.setattr(runner.Run, "_audio_tail", lambda self, name, fmt: Path(name).stem + "_tail.latent")
    monkeypatch.setattr(runner.Run, "_source_slice", lambda self, k, fmt: self.work / f"source_{k}.mp4")
    monkeypatch.setattr(runner.Run, "_pose_slice", lambda self, k, fmt, body, out: self.work / f"pose_{k}.mp4")
    monkeypatch.setattr(runner.Run, "_video_prompt", lambda self, m=None: "PROMPT")

    def fake_comfy(self, step, name, wf, values, files, outputs, extra_prefixes=None):
        vals = dict(values, output_prefix=f"cinestudio/{self.id}_{name}_a1")      # as Run._comfy sends it
        base = ProfileWorkflow(wf.profile, wf.variant, workflows_dir=self.d / "inputs" / "workflows")
        self.jobs.append({"name": name, "values": vals, "graph": wf.build(vals, files=files),
                          "base": base.build(vals, files=files)})
    monkeypatch.setattr(runner.Run, "_comfy", fake_comfy)


def _job(tmp_path, prof, sc):
    """The clip job graph that Run._layout_clip_job sends for scenario `sc` with the profile `prof`."""
    d = tmp_path / prof
    wfd = d / "inputs" / "workflows"
    wfd.mkdir(parents=True)
    for v in ("layout", "layouting"):
        for suffix in ("_api.json", ".bind.json"):
            shutil.copy2(mp.WORKFLOWS / f"{prof}_{v}{suffix}", wfd / f"{prof}_{v}{suffix}")
    runner._freeze_tier(wfd, sc.get("tier", "8gb"))
    work = d / "work"
    look, kf, sheet = _png(work / "look.png", 1), _png(work / "kf.png", 2), _png(work / "sheet.png", 3)
    _png(work / "reference.png", 3)
    for name in ("source.mp4", "clip.mp4", "clip_2.mp4"):
        (work / name).write_bytes(b"v")
    full = prof.endswith("dev")
    m = {**runner.DEFAULTS, "profile": prof, "render_mode": "layout", "quality": sc["quality"], "relay": "latent",
         "audio_relay": "carry", "adain_ref": "clip1", "adain": 0.3, "overlap_strength": 0.5, "overlap": 25,
         "style_lora": "style.safetensors", "style_lora_strength": 0.8, "character_name": "hero",
         "video_model": mp.video_model_of(prof), "pass_key": "parity", "prompt": {"character": "c", "video": "v", "style": "s"}}
    for k, val in (sc.get("m") or {}).items():
        if k in ("guidance", "steps") and not full:
            continue                                     # new_run refuses them for the distilled model
        m[k] = val
    relay = {k: {"s1": f"r{k}_s1.latent", "s2": f"r{k}_s2.latent", "a": f"r{k}_a.latent"} for k in ("1", "2")}
    st = {"steps": {
        "controls": {"outputs": {"width": 768, "height": 448, "frames": 121, "fps": 24.0, "overlap": 25, "clips": 2,
                                 "source": str(work / "source.mp4")}},
        "sheet": {"outputs": {"sheet": str(sheet)}},
        "passes": {"outputs": {"passes": str(d / "passes")}},
        "look": {"outputs": {"look": str(look), "keyframes": [{"clip": 1, "frame": 60, "file": str(kf)},
                                                               {"clip": 2, "frame": 50, "file": str(kf)}]}},
        "clip": {"parts": {"1": {"outputs": {"clip": str(work / "clip.mp4")}, "relay": relay["1"]},
                           "2": {"outputs": {"clip": str(work / "clip_2.mp4")}, "relay": relay["2"]}}}}}
    r = runner.Run.__new__(runner.Run)
    r.d, r.id, r.m, r.st, r.work, r.jobs = d, "rpar", m, st, work, []
    r.on_event, r._backend, r._backend_factory = None, None, None
    r.canceled_jobs, r.unresolved_jobs, r.new_attempt = [], [], False
    k, parts = sc["clip"], st["steps"]["clip"]["parts"]
    if sc.get("take"):
        (work / "takes").mkdir()                         # Run._make_take makes the take's folder
        mt = dict(m, keyframe_image=str(kf), keyframe_frame=30)
        end = parts["2"]["relay"] if k < 2 else None
        res = r._layout_clip_job({}, k, mt, parts, work / "takes" / "clip.mp4", f"take{k}_1",
                                 f"cinestudio/relay_rpar_{k}_t1", end_anchor=end)
    else:
        res = r._layout_clip_job({}, k, m, parts, work / f"clip_out_{k}.mp4", f"clip{k}", f"cinestudio/relay_rpar_{k}")
    assert isinstance(res, dict) and len(r.jobs) == 1, res
    return r.jobs[0]


@pytest.fixture(scope="module")
def jobs(tmp_path_factory):
    """{scenario: {profile: job}} for every scenario and both models (built once)."""
    out = {}
    with pytest.MonkeyPatch.context() as mpatch:
        root = tmp_path_factory.mktemp("parity")
        mpatch.setenv("CINESTUDIO_USER_WORKFLOWS", str(root / "user_workflows"))
        mpatch.setenv("CINESTUDIO_HOME", str(root / "home"))
        _stub(mpatch, root)
        for name, sc in SCENARIOS.items():
            out[name] = {prof: _job(root / name, prof, sc) for prof in PROFILES}
    return out


def _canon(graph: dict, nodes: dict) -> dict:
    """Nodes with swap-equal class names, without the values that each model has its own of, and with links through
    the full model's pass-through nodes (mp.FULL_ONLY_NODES: the stage-2 distilled LoRA) skipped."""
    out = {}
    for k, n in nodes.items():
        out[k] = {"class": mp.swap_class(n["class_type"]),
                  "inputs": {i: mp.through(graph, v) for i, v in n["inputs"].items()
                             if (n["class_type"], i) not in mp.VALUE_DIFFS}}
    return out


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_the_clip_job_graphs_of_both_models_are_in_parity(jobs, scenario):
    """The whole job graph (after every run-time feature) differs only where the model needs it."""
    d, f = jobs[scenario][PROFILES[0]], jobs[scenario][PROFILES[1]]
    diff = mp.graph_diff(d["graph"], f["graph"])
    assert not diff, (f"scenario {scenario}: the clip job graphs drifted apart:\n  " + "\n  ".join(diff) +
                      "\nA run-time feature is wired differently into one model: wire it into both "
                      "(runner._layout_clip_job and its helpers), or declare it model-specific in model_pairs.py.")
    only_full = {b for ft in mp.FEATURES.values() if ft.only == "full" for b in ft.binds}
    only_dist = {b for ft in mp.FEATURES.values() if ft.only == "distilled" for b in ft.binds}
    assert set(d["values"]) - set(f["values"]) <= only_dist, scenario
    assert set(f["values"]) - set(d["values"]) <= only_full, scenario


FEATURE_SCENARIOS = [(ft, sc) for ft in sorted(n for n, x in mp.FEATURES.items() if not x.graphless)
                     for sc in sorted(SCENARIOS)]


@pytest.mark.parametrize("feature,scenario", FEATURE_SCENARIOS)
def test_each_feature_is_wired_into_both_models(jobs, feature, scenario):
    """Feature x scenario (pair, stage, clip, take): the nodes it adds, the inputs it edits and the values it binds are
    the same in the distilled and the full job, unless the feature is declared model-specific (only=, a reason)."""
    ft = mp.FEATURES[feature]
    got = {}
    for prof in PROFILES:
        j = jobs[scenario][prof]
        ch = mp.job_changes(j["graph"], j["base"])
        got[prof] = {"nodes": _canon(j["graph"], mp.feature_nodes(ft, j["graph"], j["base"])),
                     "edits": {e for e in ch["edited"] if e in {(mp.swap_class(c), i) for c, i in ft.edits}},
                     "binds": {b for b in ft.binds if b in j["values"]}}
    d, f = got[PROFILES[0]], got[PROFILES[1]]
    if ft.only:
        on, off = (f, d) if ft.only == "full" else (d, f)
        assert not (off["nodes"] or off["binds"]), (f"feature {feature!r} is declared {ft.only}-only ({ft.reason}) but "
                                                    f"is wired into the other model too in {scenario}")
        return
    for part in ("nodes", "edits", "binds"):
        assert d[part] == f[part], (
            f"feature {feature!r} ({ft.where}) is wired differently in {scenario} ({part}):\n"
            f"  distilled: {d[part]}\n  full: {f[part]}\n"
            f"Wire it into both {PROFILES[0]}_{SCENARIOS[scenario]['variant']} and "
            f"{PROFILES[1]}_{SCENARIOS[scenario]['variant']}, or declare it model-specific in its registration "
            "(only='full'|'distilled', reason=...).")


@pytest.mark.parametrize("prof", PROFILES)
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_no_clip_graph_change_is_unregistered(jobs, scenario, prof):
    """Every node added, input changed and value bound by a clip job belongs to a registered feature."""
    j = jobs[scenario][prof]
    bad = mp.undeclared(mp.job_changes(j["graph"], j["base"]), j["values"])
    assert not bad, f"{prof}, scenario {scenario}:\n  " + "\n  ".join(bad)


def test_every_registered_feature_is_exercised_by_a_scenario(jobs):
    for name, ft in mp.FEATURES.items():
        if ft.graphless:
            continue
        seen = False
        for sc in SCENARIOS:
            for prof in PROFILES:
                j = jobs[sc][prof]
                ch = mp.job_changes(j["graph"], j["base"])
                if mp.feature_nodes(ft, j["graph"], j["base"]) or any(b in j["values"] for b in ft.binds) or \
                        ch["edited"] & {(mp.swap_class(c), i) for c, i in ft.edits}:
                    seen = True
        assert seen, (f"feature {name!r} ({ft.where}) is not exercised by any parity scenario: add one to SCENARIOS "
                      "in tests/unit/test_model_parity.py that switches it on")


def test_model_specific_values_are_each_models_own(jobs):
    """The stage-1 schedule: distilled 8 steps from 1.0; full = runner.dev_sigmas (30 steps); guidance on full only;
    the negative is bound in both (only the full model's guider uses it)."""
    d, f = jobs["clip2-final"][PROFILES[0]], jobs["clip2-final"][PROFILES[1]]
    assert d["values"]["sigmas"].startswith("1, 0.99375") and len(d["values"]["sigmas"].split(",")) == 9
    assert len(f["values"]["sigmas"].split(",")) == runner.DEV_STEPS + 1
    assert f["values"]["video_cfg"] == 4.0 and "video_cfg" not in d["values"]
    assert f["graph"]["gpv1"]["inputs"]["cfg"] == 4.0 and d["graph"]["guider1"]["inputs"]["cfg"] == 1.0
    assert d["values"]["negative"] == f["values"]["negative"] == runner.NEGATIVE
    assert f["graph"]["neg"]["inputs"]["text"] == runner.NEGATIVE


def _reachable(graph: dict, start: str) -> set:
    seen, todo = set(), [start]
    while todo:
        k = todo.pop()
        if k in seen or k not in graph:
            continue
        seen.add(k)
        todo += [v[0] for v in graph[k]["inputs"].values() if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)]
    return seen


@pytest.mark.parametrize("variant", ["layout", "layouting"])
def test_the_full_models_stage2_is_lightricks_two_stage_recipe(variant):
    """2026-10-25 (user-approved): stage 2 of the full model = the dev transformer + the distilled LoRA, no guidance
    (CFG 1), the distilled stage-2 sigmas, plain Euler (ltx-pipelines TI2VidTwoStagesPipeline); stage 1 keeps the dev
    guider. Both tiers take the same graph (the LoRA is one bf16 file, setup/models.json ltx25_distilled_lora)."""
    from cinestudio import models
    full, dist = (json.loads((mp.WORKFLOWS / f"{p}_{variant}_api.json").read_text(encoding="utf-8")) for p in PROFILES[::-1])
    bind = json.loads((mp.WORKFLOWS / f"{PROFILES[1]}_{variant}.bind.json").read_text(encoding="utf-8"))
    assert full["guider1"]["class_type"] == "MultimodalGuider"                      # stage 1: the dev recipe
    assert full["guider2"] == {"class_type": "CFGGuider", "inputs": {"model": ["lora_dist", 0], "positive": ["g_look2", 0],
                                                                     "negative": ["g_look2", 1], "cfg": 1.0}}
    assert full["lora_dist"]["inputs"] == {"model": ["lora_style2", 0], "strength_model": 1.0,
                                           "lora_name": "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"}
    assert full["sigmas2"] == dist["sigmas2"] and full["sigmas2"]["inputs"]["sigmas"] == "0.909375, 0.725, 0.421875, 0.0"
    assert full["sampler2"]["inputs"]["sampler_name"] == "euler"                   # no noise between the steps
    assert not {"gpv2", "gpa2"} & set(full)
    assert "lora_dist" in _reachable(full, "save") and "lora_dist" not in _reachable(full, "guider1")
    assert bind["models"]["ltx25_distilled_lora"] == [["lora_dist", "lora_name"]]
    assert bind["inputs"]["stage2_lora_strength"] == [["lora_dist", "strength_model"]]
    assert bind["capabilities"]["stage2"] == "distilled-lora"
    for tier in ("8gb", "large"):
        assert models.bundle(tier)["ltx25_distilled_lora"]["file"] == full["lora_dist"]["inputs"]["lora_name"]


def test_the_stage2_lora_reaches_every_job_and_preview_never_loads_it(jobs):
    """Every full FINAL job runs its stage 2 through the LoRA (also under the pose guide's LoRA, without a style LoRA);
    a preview job's outputs do not reach it (ComfyUI never runs or loads it)."""
    for name, sc in SCENARIOS.items():
        g = jobs[name][PROFILES[1]]["graph"]
        chain = _reachable(g, "guider2")
        if sc["quality"] == "final":
            assert "lora_dist" in chain and "lora_dist" in _reachable(g, "save"), name
            assert mp.through(g, g["lora_dist"]["inputs"]["model"]) == g["lora_dist"]["inputs"]["model"], name
        else:
            assert "lora_dist" not in _reachable(g, "save") and "ks2" not in _reachable(g, "save"), name
        assert "lora_dist" not in jobs[name][PROFILES[0]]["graph"], name


def test_the_pass_through_rule_still_sees_drift_behind_the_lora(tmp_path):
    """Links through lora_dist compare as links to its model input, so a stage-2 model chain that drifts behind it is
    still reported, and lora_dist is the only node allowed in the full member alone."""
    pair = mp.PAIRS["layout"]
    for prof in PROFILES:
        for suffix in ("_api.json", ".bind.json"):
            shutil.copy2(mp.WORKFLOWS / f"{prof}_layout{suffix}", tmp_path / f"{prof}_layout{suffix}")
    assert not mp.pair_diff(pair, tmp_path)
    g = json.loads((tmp_path / f"{pair.full}_api.json").read_text(encoding="utf-8"))
    g["lora_dist"]["inputs"]["model"] = ["lora_lay", 0]                         # stage 2 without the style LoRA
    g["lora_x"] = dict(g["lora_dist"])
    (tmp_path / f"{pair.full}_api.json").write_text(json.dumps(g), encoding="utf-8")
    text = "\n".join(mp.pair_diff(pair, tmp_path))
    assert "guider2.model" in text and "'lora_x'" in text


# ------------------------------------------------------------------ the source guard (code paths no scenario reaches)

def _runner_tree():
    return ast.parse((ROOT / "cinestudio" / "runner.py").read_text(encoding="utf-8"))


def _is_wf_graph(node) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "graph" and isinstance(node.value, ast.Name) \
        and node.value.id == "wf"


def test_every_helper_that_changes_a_clip_graph_is_registered():
    """A function called with `wf.graph` in runner.py changes a job graph: it must carry @graph_feature."""
    called = set()
    for node in ast.walk(_runner_tree()):
        if isinstance(node, ast.Call) and any(_is_wf_graph(a) for a in node.args) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
    assert called, "found no graph helper calls (the guard would be blind)"
    for name in sorted(called):
        fn = getattr(runner, name, None)
        assert fn is not None and getattr(fn, "graph_feature", None) in mp.FEATURES, (
            f"runner.{name}(wf.graph, ...) changes the clip graph but is not registered: put @graph_feature(...) on it "
            "with its node ids, class types and edited inputs (cinestudio/model_pairs.py)")


def test_runner_never_edits_a_job_graph_inline():
    """Graph edits live in registered helpers, not inline (`wf.graph[...] = ...` in a method would skip the registry)."""
    for node in ast.walk(_runner_tree()):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AugAssign) else []
        for t in targets:
            while isinstance(t, ast.Subscript):
                t = t.value
                if _is_wf_graph(t):
                    pytest.fail(f"runner.py line {node.lineno} edits wf.graph inline: move it into a helper with "
                                "@graph_feature (cinestudio/model_pairs.py)")


def test_every_node_class_that_runner_adds_is_declared():
    """Each "class_type": "X" literal in runner.py is a node a registered feature declares."""
    lits = set()
    for node in ast.walk(_runner_tree()):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "class_type" and isinstance(v, ast.Constant):
                    lits.add(v.value)
    missing = sorted(lits - mp.declared_classes())
    assert not missing, (f"runner.py adds nodes of {missing} that no registered feature declares: add them to the "
                         "adds= of the helper's @graph_feature")


def test_a_model_specific_feature_needs_a_reason():
    with pytest.raises(ValueError, match="reason"):
        mp.register("test-only feature", "nowhere", binds=("x",), only="full")
    assert "test-only feature" not in mp.FEATURES


def test_the_pair_diff_sees_drift(tmp_path):
    """The check is not blind: a changed value, an extra node and a lost binding in one member are all reported."""
    pair = mp.PAIRS["layouting"]
    for prof in PROFILES:
        for suffix in ("_api.json", ".bind.json"):
            shutil.copy2(mp.WORKFLOWS / f"{prof}_layouting{suffix}", tmp_path / f"{prof}_layouting{suffix}")
    g = json.loads((tmp_path / f"{pair.full}_api.json").read_text(encoding="utf-8"))
    g["g_lay"]["inputs"]["strength"] = 0.5
    g["extra"] = {"class_type": "LTXVAdainLatent", "inputs": {}}
    (tmp_path / f"{pair.full}_api.json").write_text(json.dumps(g), encoding="utf-8")
    b = json.loads((tmp_path / f"{pair.distilled}.bind.json").read_text(encoding="utf-8"))
    del b["inputs"]["refine_sigmas"]
    (tmp_path / f"{pair.distilled}.bind.json").write_text(json.dumps(b), encoding="utf-8")
    text = "\n".join(mp.pair_diff(pair, tmp_path))
    assert "g_lay.strength" in text and "'extra'" in text and "'refine_sigmas' is only in the full" in text
