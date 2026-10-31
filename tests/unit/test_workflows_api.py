"""The Workflows view (an advanced mode): the saved submitted graphs (character.image_job, workflow.save_graph) and
the read-only API in cinestudio/webapi/workflows.py. No GPU: jobs use a fake backend, runs are folders on disk."""
import asyncio
import json
import shutil

import pytest
from aiohttp.test_utils import TestClient, TestServer

from cinestudio import character, runner, webapp, workflow
from cinestudio.backend import JobState
from cinestudio.locks import FileLock

RID = "r20261025-120000-wfvw"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def call(fn):
    async def run():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as client:
            return await fn(client)
    return asyncio.run(run())


# ---------------------------------------------------------------- saving
def test_save_graph_keeps_the_newest_and_never_raises(home, tmp_path):
    folder = tmp_path / "g"
    for i in range(5):
        assert workflow.save_graph(folder, f"job{i}", {"a": {"class_type": "X", "inputs": {"i": i}}}, keep=3)
        import os
        os.utime(folder / f"job{i}.json", (1000 + i, 1000 + i))
    names = sorted(p.name for p in folder.glob("*.json"))
    assert names == ["job2.json", "job2.meta.json", "job3.json", "job3.meta.json", "job4.json", "job4.meta.json"]
    assert workflow.save_graph(folder, "../evil name", {}) == folder / "evil-name.json"     # a safe file name
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert workflow.save_graph(blocker / "sub", "job", {"a": {}}) is None                # a write failure: no error


class FakeBackend:
    def __init__(self):
        self.submits = []

    def submit(self, graph, files, key, prefix=None):
        self.submits.append(graph)
        return "job1"

    def watch(self, jid, on_event):
        return JobState(jid, "succeeded")

    def download(self, jid, node, dst):
        dst.write_bytes(b"png")
        return dst


def _image_job_env(monkeypatch, tmp_path):
    monkeypatch.setattr(character.server, "alive", lambda url=None: True)
    monkeypatch.setattr(character.server, "ensure_family", lambda fam, url=None: {"restarted": False})
    monkeypatch.setattr(character.server, "note_owned", lambda pid, url=None: None)
    monkeypatch.setattr(character.server, "gpu_lock", lambda url=None, timeout=0, note=None:
                        FileLock(tmp_path / "gpu.lock", timeout=timeout, note=note))
    from cinestudio import models
    monkeypatch.setattr(models, "job_bundle", lambda: {})


def test_image_job_saves_the_exact_graph_it_submits(home, tmp_path, monkeypatch):
    _image_job_env(monkeypatch, tmp_path)
    be = FakeBackend()

    def edit(g):
        g["extra"] = {"class_type": "Note", "inputs": {}}
    character.image_job("qwenedit", "edit", {"instruction": "make it red", "seed": 3}, {("img", "image"): "x.png"},
                        tmp_path / "out.png", "hero_edit", backend_factory=lambda: be, graph_edit=edit)
    saved = workflow.graph_cache() / "hero_edit.json"
    assert json.loads(saved.read_text()) == be.submits[0]                  # after the caller's change
    meta = json.loads((workflow.graph_cache() / "hero_edit.meta.json").read_text())
    assert meta["profile"] == "qwenedit" and meta["variant"] == "edit" and meta["edited"] is True
    assert meta["files"] == {"img.image": "x.png"} and meta["prefix"].startswith("cinestudio/char_hero_edit_")


def test_image_job_runs_when_the_graph_cannot_be_saved(home, tmp_path, monkeypatch):
    _image_job_env(monkeypatch, tmp_path)
    home.mkdir(parents=True, exist_ok=True)
    (home / "cache").write_text("not a folder")                           # every write fails
    be = FakeBackend()
    r = character.image_job("qwenedit", "edit", {"instruction": "x", "seed": 1}, {("img", "image"): "x.png"},
                            tmp_path / "out.png", "hero_edit", backend_factory=lambda: be)
    assert r["job"] == "job1" and len(be.submits) == 1


# ---------------------------------------------------------------- the API
def _run(home):
    """A run folder with the frozen layouting graph and one submitted clip graph changed at run time."""
    d = home / "runs" / RID
    (d / "inputs" / "workflows").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"id": RID}))
    for f in ("ltx25_layouting_api.json", "ltx25_layouting.bind.json"):
        shutil.copy(workflow.WORKFLOWS / f, d / "inputs" / "workflows" / f)
    wf = workflow.ProfileWorkflow("ltx25", "layouting", workflows_dir=d / "inputs" / "workflows")
    target = next(n for n, node in wf.graph.items() if n not in wf.bind.get("optional_nodes", {})
                  and not any(isinstance(v, list) for v in node["inputs"].values()) and node["inputs"]
                  and all((n, k) not in {(t[0], t[1]) for ts in [*wf.bind["inputs"].values(), *wf.bind["models"].values()] for t in ts}
                          for k in node["inputs"]))
    first_key = next(iter(wf.graph[target]["inputs"]))
    wf.graph[target]["inputs"][first_key] = "changed at run time"          # a run-time change outside the binding
    wf.graph["relay_save"] = {"class_type": "SaveLatent", "inputs": {"samples": [target, 0], "filename_prefix": "r"}}
    graph = wf.build({"prompt": "a fox waves", "seed": 5})
    workflow.save_graph(d / "work" / "graphs", "clip2", graph, files={("look", "image"): "look.png"},
                        meta={"run": RID, "job": "clip2", "profile": "ltx25", "variant": "layouting", "attempt": 1})
    prompt_node = wf.targets("prompt")[0][0]
    return {"target": target, "prompt_node": prompt_node, "graph": graph}


def test_the_workflow_list_groups_the_graphs_by_feature(home):
    async def go(c):
        d = (await (await c.get("/api/workflows")).json())["data"]
        rows = {r["name"]: r for r in d["items"]}
        assert rows["ltx25_layouting"]["group"] == "Videos" and rows["ltx25_layouting"]["nodes"] > 10
        assert rows["qwenedit_edit"]["model_family"] == "qwen_edit"
        assert any("Look images" in f for f in rows["qwenedit_edit"]["features"])
        assert rows["ltx25_v2vref"]["group"] == "Videos"                    # distilled video-to-video
        assert not [n for n in rows if n.startswith("ltx25dev_")]       # retired full-model graphs
        assert not [n for n in rows if n.startswith("ltx23")]             # the retired LTX-2.3 graphs are gone
        assert not [r["name"] for r in d["items"] if r["group"] == d["groups"][-1]]   # every graph serves a feature
        one = (await (await c.get("/api/workflows/qwenedit_angle")).json())["data"]
        assert one["graph"] and one["bind"]["profile"] == "qwenedit"
        assert (await c.get("/api/workflows/missing_graph")).status == 404
        for bad in ("..secrets", "a%20b", "..%5Cx"):
            assert (await c.get(f"/api/workflows/{bad}")).status in (400, 404)
    call(go)


def test_a_runs_submitted_graph_shows_what_changed_at_run_time(home):
    info = _run(home)

    async def go(c):
        d = (await (await c.get(f"/api/runs/{RID}/graphs")).json())["data"]
        assert [s["job"] for s in d["submitted"]] == ["clip2"] and d["submitted"][0]["profile"] == "ltx25"
        assert [f["name"] for f in d["frozen"]] == ["ltx25_layouting"]
        g = (await (await c.get(f"/api/runs/{RID}/graphs/clip2")).json())["data"]
        assert g["graph"] == info["graph"] and g["base_from"] == "run" and g["base_name"] == "ltx25_layouting"
        assert g["added"] == ["relay_save"]
        assert g["changed"] == [info["target"]]                              # only the run-time change
        assert info["prompt_node"] in g["filled"] and info["prompt_node"] not in g["changed"]
        assert g["meta"]["files"] == {"look.image": "look.png"}
        assert g["titles"]                                                   # titles from the base graph
        fr = (await (await c.get(f"/api/runs/{RID}/graphs/ltx25_layouting?frozen=1")).json())["data"]
        assert fr["graph"] and fr["added"] == []
        assert (await c.get(f"/api/runs/{RID}/graphs/clip9")).status == 404
        assert (await c.get(f"/api/runs/{RID}/graphs/..%5Cmanifest")).status in (400, 404)
        assert (await c.get("/api/runs/nope/graphs")).status == 400            # runner.RunError
    call(go)


def test_recent_job_graphs_and_the_advanced_setting(home):
    workflow.save_graph(workflow.graph_cache(), "hero_s1", {"a": {"class_type": "X", "inputs": {}}},
                        meta={"tag": "hero_s1", "profile": "zimage", "variant": "base"})

    async def go(c):
        d = (await (await c.get("/api/graphs/recent")).json())["data"]
        assert [x["job"] for x in d["items"]] == ["hero_s1"]
        g = (await (await c.get("/api/graphs/recent/hero_s1")).json())["data"]
        assert g["base_from"] == "repository" and g["added"] == ["a"]
        assert (await (await c.get("/api/settings/advanced")).json())["data"] == {"advanced": False}
        r = await c.post("/api/settings/advanced", json={"advanced": True})
        assert (await r.json())["data"] == {"advanced": True}
        assert (await c.post("/api/settings/advanced", json={"advanced": "yes"})).status == 400
        assert (await (await c.get("/api/settings/advanced")).json())["data"] == {"advanced": True}
    call(go)
