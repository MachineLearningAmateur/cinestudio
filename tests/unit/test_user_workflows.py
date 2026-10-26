"""The user's own graph copies (cinestudio/user_workflows.py): copy, edit, undo, the toggle that switches what
ProfileWorkflow loads, the refusals, runs that record and freeze the copy, forks that keep it, and the routes.
The built-in graphs in workflows/ are never written. No GPU, no ComfyUI (its /object_info is stubbed)."""
import asyncio
import hashlib
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from cinestudio import runner, user_workflows as uw, webapp, workflow
from cinestudio.webapi import workflows as api

EDIT = "qwenedit_edit"


def sha(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(api, "object_info", lambda classes: {"available": False, "combos": {}})
    return tmp_path / "home"


def free_input(builtin):
    """A literal input of the built-in that the job does not set (node, input, value)."""
    graph, bound = uw.builtin_graph(builtin), uw.bound_inputs(uw.builtin_bind(builtin))
    for n, node in graph.items():
        for k, v in node["inputs"].items():
            if (n, k) not in bound and isinstance(v, float) and not isinstance(v, bool):
                return n, k, v
    raise AssertionError("no free float input")


def test_a_copy_is_edited_saved_and_undone_and_the_builtin_never_changes(home):
    before = sha(workflow.WORKFLOWS / f"{EDIT}_api.json")
    meta = uw.make_copy(EDIT)
    assert meta["label"] == "My copy 1" and meta["made_from"] == EDIT and meta["version"] == 1
    n, k, v = free_input(EDIT)
    r = uw.save_values(EDIT, meta["id"], {n: {k: 2}}, version=1)               # an int for a float input: kept a float
    graph, meta2 = uw.load_copy(EDIT, meta["id"])
    assert r["saved"] and graph[n]["inputs"][k] == 2.0 and isinstance(graph[n]["inputs"][k], float)
    assert meta2["version"] == 2 and uw.custom_values(graph, EDIT) == {n: [k]}
    assert sha(workflow.WORKFLOWS / f"{EDIT}_api.json") == before                # never the built-in
    with pytest.raises(uw.UserWorkflowError, match="saved elsewhere"):
        uw.save_values(EDIT, meta["id"], {n: {k: 3}}, version=1)                # an old version
    uw.save_values(EDIT, meta["id"], {n: {k: 30.0}})
    assert any("big change" in w for w in uw.save_values(EDIT, meta["id"], {n: {k: 3000.0}})["warnings"])
    uw.undo(EDIT, meta["id"])
    assert uw.load_copy(EDIT, meta["id"])[0][n]["inputs"][k] == 30.0
    uw.reset_values(EDIT, meta["id"])
    assert uw.load_copy(EDIT, meta["id"])[0][n]["inputs"][k] == v                # the built-in's value again
    dup = uw.make_copy(EDIT, from_copy=meta["id"])
    assert dup["label"] == "My copy 1 (copy)" and len(uw.list_copies(EDIT)) == 2
    assert uw.rename(EDIT, dup["id"], "  Softer   tail ")["label"] == "Softer tail"


def test_wrong_types_inputs_the_job_sets_links_and_unknown_nodes_are_refused(home):
    meta = uw.make_copy(EDIT)
    graph = uw.builtin_graph(EDIT)
    bound = sorted(uw.bound_inputs(uw.builtin_bind(EDIT)))
    n, k, _ = free_input(EDIT)
    cases = [({n: {k: "x"}}, "a number"), ({n: {k: True}}, "a number"), ({"nope": {"a": 1}}, "not in this graph"),
             ({n: {"nope": 1}}, "not an input"), ({bound[0][0]: {bound[0][1]: 1}}, "set by the job")]
    link = next((nn, kk) for nn, node in graph.items() for kk, vv in node["inputs"].items()
                if isinstance(vv, list) and (nn, kk) not in set(bound))
    cases.append(({link[0]: {link[1]: "x"}}, "not editable"))
    ints = next(((nn, kk) for nn, node in graph.items() for kk, vv in node["inputs"].items()
                 if type(vv) is int and (nn, kk) not in set(bound)), None)
    if ints:
        cases.append(({ints[0]: {ints[1]: 1.5}}, "whole number"))
    for values, msg in cases:
        with pytest.raises(uw.UserWorkflowError, match=msg):
            uw.save_values(EDIT, meta["id"], values)
    with pytest.raises(uw.UserWorkflowError):
        uw.make_copy("../secrets")
    with pytest.raises(uw.UserWorkflowError):
        uw.load_copy(EDIT, "../x")


def test_the_toggle_switches_what_new_jobs_load_and_the_job_values_still_win(home):
    meta = uw.make_copy(EDIT)
    n, k, _ = free_input(EDIT)
    uw.save_values(EDIT, meta["id"], {n: {k: 0.42}})
    assert workflow.ProfileWorkflow("qwenedit", "edit").graph[n]["inputs"][k] != 0.42      # not chosen yet
    uw.set_choice(EDIT, meta["id"])
    wf = workflow.ProfileWorkflow("qwenedit", "edit")
    assert wf.graph[n]["inputs"][k] == 0.42 and wf.source["copy"] == meta["id"]
    g = wf.build({"instruction": "make it red", "seed": 77})
    seed_node, seed_field = wf.targets("seed")[0]
    assert g[n]["inputs"][k] == 0.42 and g[seed_node]["inputs"][seed_field] == 77      # the job's value wins
    assert workflow.ProfileWorkflow("qwenedit", "edit", use_copy=False).source == {"builtin": True}
    assert uw.delete(EDIT, meta["id"])["was_active"] and uw.active_choice(EDIT) is None   # back to the built-in
    assert workflow.ProfileWorkflow("qwenedit", "edit").source == {"builtin": True}


def test_a_copy_that_no_longer_fits_the_builtin_is_refused(home):
    meta = uw.make_copy(EDIT)
    uw.set_choice(EDIT, meta["id"])
    path, _ = uw._paths(EDIT, meta["id"])
    graph = json.loads(path.read_text())
    seed_node = uw.builtin_bind(EDIT)["inputs"]["seed"][0][0]
    del graph[seed_node]                                                     # an app update renamed a bound node
    path.write_text(json.dumps(graph))
    with pytest.raises(ValueError, match="older built-in"):
        workflow.ProfileWorkflow("qwenedit", "edit")
    with pytest.raises(uw.UserWorkflowError, match="does not fit"):
        uw.set_choice(EDIT, meta["id"])
    assert uw.problems(graph, EDIT)


def test_a_run_records_and_freezes_the_chosen_copy_and_a_fork_keeps_it(home, tmp_path, monkeypatch):
    prof = runner.PROFILE
    name = f"{prof}_v2vref"
    meta = uw.make_copy(name, "Run test")
    n, k, _ = free_input(name)
    uw.save_values(name, meta["id"], {n: {k: 0.123}})
    uw.set_choice(name, meta["id"])
    def fake_snapshot(src, dst, report, allow_external=False):
        dst.write_bytes(src.read_bytes())
        r = {"ok": True, "packed": [], "missing": [], "unsupported": []}
        report.write_text(json.dumps(r))
        return r
    monkeypatch.setattr(runner, "snapshot_blend", fake_snapshot)
    monkeypatch.setattr(runner.blender, "check", lambda: {"ok": True, "version": "5.2.0"})
    blend, sheet = tmp_path / "shot.blend", tmp_path / "sheet.png"
    blend.write_bytes(b"BLENDER-fake")
    import cv2
    import numpy as np
    cv2.imwrite(str(sheet), np.zeros((64, 128, 3), np.uint8))
    m = runner.new_run(blend, sheet, "a figure", "it waves")
    assert m["graph_sources"][name]["copy"] == meta["id"] and m["graph_sources"][name]["label"] == "Run test"
    frozen = runner.run_dir(m["id"]) / "inputs" / "workflows" / f"{name}_api.json"
    assert json.loads(frozen.read_text())[n]["inputs"][k] == 0.123
    chips = {c["label"]: c for c in runner.run_details(m["id"])["chips"]}
    assert chips["Custom graph"]["value"] == "Run test" and chips["Custom graph"]["changed"]
    # a fork keeps the parent's frozen copy, even after the user changed the copy and went back to the built-in
    uw.save_values(name, meta["id"], {n: {k: 0.9}})
    uw.set_choice(name, None)
    d = runner.run_dir(m["id"])
    w = str(d / "work")
    (d / "work").mkdir(exist_ok=True)
    for f in ("clip.mp4", "clip_2.mp4"):
        (d / "work" / f).write_bytes(b"x")
    st = {"status": "done", "steps": {"controls": {"status": "done", "outputs": {"clips": 2}},
          "clip": {"status": "done", "parts": {"1": {"outputs": {"clip": w + "/clip.mp4"}},
                                               "2": {"outputs": {"clip": w + "/clip_2.mp4"}}}},
          "final": {"status": "done"}}}
    (d / "state.json").write_text(json.dumps(st))
    f = runner.fork_run(m["id"], 2)
    forked = runner.run_dir(f["id"]) / "inputs" / "workflows" / f"{name}_api.json"
    assert json.loads(forked.read_text())[n]["inputs"][k] == 0.123 and f["graph_sources"][name]["copy"] == meta["id"]



def test_the_copy_values_win_over_the_tiers_node_settings_in_a_run(home, tmp_path, monkeypatch):
    """The large tier writes its decode tiles into every graph of a run; a value the user set in the copy stays."""
    from cinestudio import models
    name = f"{runner.PROFILE}_v2vref"
    graph = uw.builtin_graph(name)
    vdec = next(n for n, node in graph.items() if node["class_type"] == "VAEDecodeTiled")
    meta = uw.make_copy(name)
    uw.save_values(name, meta["id"], {vdec: {"tile_size": 384}})
    uw.set_choice(name, meta["id"])
    monkeypatch.setattr(models, "graph_settings", lambda tier: {"VAEDecodeTiled": {"tile_size": 1024, "overlap": 256}})

    def fake_snapshot(src, dst, report, allow_external=False):
        dst.write_bytes(src.read_bytes())
        r = {"ok": True, "packed": [], "missing": [], "unsupported": []}
        report.write_text(json.dumps(r))
        return r
    monkeypatch.setattr(runner, "snapshot_blend", fake_snapshot)
    monkeypatch.setattr(runner.blender, "check", lambda: {"ok": True, "version": "5.2.0"})
    blend, sheet = tmp_path / "shot.blend", tmp_path / "sheet.png"
    blend.write_bytes(b"BLENDER-fake")
    import cv2
    import numpy as np
    cv2.imwrite(str(sheet), np.zeros((64, 128, 3), np.uint8))
    m = runner.new_run(blend, sheet, "a figure", "it waves")
    frozen = json.loads((runner.run_dir(m["id"]) / "inputs" / "workflows" / f"{name}_api.json").read_text())
    assert frozen[vdec]["inputs"]["tile_size"] == 384 and frozen[vdec]["inputs"]["overlap"] == 256

def call(fn):
    async def run():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as client:
            return await fn(client)
    return asyncio.run(run())


def test_the_copy_routes(home):
    n, k, v = free_input(EDIT)
    before = sha(workflow.WORKFLOWS / f"{EDIT}_api.json")

    async def go(c):
        r = await c.post(f"/api/workflows/{EDIT}/copies", json={"label": "Mine"})
        d = (await r.json())["data"]
        cid = d["id"]
        assert d["meta"]["label"] == "Mine" and d["custom"] == {} and d["problems"] == [] and d["bound"]
        assert d["builtin_graph"][n]["inputs"][k] == v
        lst = (await (await c.get("/api/workflows")).json())["data"]
        row = next(x for x in lst["items"] if x["name"] == EDIT)
        assert [x["id"] for x in row["copies"]] == [cid] and row["active"] is None
        assert any(a["group"] == "Backgrounds" for a in row["also"])                # the set backplates are listed
        pano = next(x for x in lst["items"] if x["name"] == "qwenimage_pano")
        assert any("Backplates" in a["label"] for a in pano["also"])
        url = f"/api/workflows/{EDIT}/copies/{cid}"
        r = await c.post(url, json={"values": {n: {k: 0.5}}, "version": 1})
        d = (await r.json())["data"]
        assert d["saved"] and d["copy"]["custom"] == {n: [k]}
        assert (await c.post(url, json={"values": {"zz": {"a": 1}}})).status == 400
        assert (await c.post(f"/api/workflows/{EDIT}/use", json={"copy": cid})).status == 200
        assert (await (await c.get(f"/api/workflows/{EDIT}")).json())["data"]["active"] == cid
        d = (await (await c.post(f"{url}/undo")).json())["data"]
        assert d["copy"]["custom"] == {}
        assert (await c.post(f"{url}/undo")).status == 400                       # nothing left to undo
        assert (await (await c.post(f"{url}/rename", json={"label": "Other"})).json())["data"]["label"] == "Other"
        d = (await (await c.post(f"{url}/delete")).json())["data"]
        assert d["was_active"] is True
        assert (await c.get(url)).status in (400, 404)
        assert (await c.post("/api/workflows/missing_x/copies")).status == 404
        t = (await (await c.get(f"/api/workflows/{EDIT}?tier=large")).json())["data"]
        assert t["tier"] == "large" and t["diff_label"].startswith("Differs from the")
        assert (await c.get(f"/api/workflows/{EDIT}?tier=huge")).status == 400
    call(go)
    assert sha(workflow.WORKFLOWS / f"{EDIT}_api.json") == before
