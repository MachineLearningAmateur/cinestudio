"""Runner: frozen inputs, locks, recovery without duplicate jobs, cancel (fake backend, no ComfyUI or Blender)."""

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from cinestudio import runner
from cinestudio.backend import JobState, UnresolvedSubmission
from cinestudio.locks import FileLock


class KeyReuse(Exception):
    """Stands in for comfy-sdk's IdempotencyKeyReuse (matched through is_key_reuse)."""


class FakeBackend:
    def __init__(self):
        self.submits, self.jobs, self.canceled = [], {}, []
        self.final = "succeeded"           # status that watch() ends with
        self.fail_submit = None            # exception for the next submit
        self.found = {}                    # prefix -> job id for find()
        self.used_keys = set()
        self.on_submit = None              # hook called inside submit
        self.watch_error = None

    def submit(self, graph, files, key, prefix=None):
        if self.fail_submit:
            exc, self.fail_submit = self.fail_submit, None
            raise exc
        if key in self.used_keys:
            raise KeyReuse(key)
        self.used_keys.add(key)
        jid = f"job{len(self.submits) + 1}"
        self.submits.append({"key": key, "prefix": prefix, "graph": graph})
        self.jobs[jid] = "running"
        if self.on_submit:
            self.on_submit()
        return jid

    def find(self, prefix):
        return self.found.get(prefix)

    def state(self, jid):
        return JobState(jid, self.jobs.get(jid, "running"))

    def watch(self, jid, on_event):
        if self.watch_error:
            exc, self.watch_error = self.watch_error, None
            raise exc
        on_event("progress", {"step": 1, "steps": 8, "node": "ks"})
        if self.jobs.get(jid) == "canceled":
            return JobState(jid, "canceled")
        self.jobs[jid] = self.final
        return JobState(jid, self.final, "boom" if self.final == "failed" else None)

    def download(self, jid, node, dst):
        dst.write_bytes(b"clip")
        return dst

    def cancel(self, jid):
        self.canceled.append(jid)
        self.jobs[jid] = "canceled"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    blend = tmp_path / "shot.blend"
    blend.write_bytes(b"BLENDER-fake")
    sheet = tmp_path / "sheet.png"
    cv2.imwrite(str(sheet), np.zeros((64, 128, 3), np.uint8))

    def fake_snapshot(src, dst, report, allow_external=False):
        dst.write_bytes(src.read_bytes())
        r = {"ok": True, "packed": [], "missing": [], "unsupported": []}
        report.write_text(json.dumps(r))
        return r
    monkeypatch.setattr(runner, "snapshot_blend", fake_snapshot)
    monkeypatch.setattr(runner.blender, "check", lambda: {"ok": True, "version": "5.2.0"})
    monkeypatch.setattr(runner.server, "ensure_family", lambda fam, url=None: {"restarted": False})
    owned = []
    monkeypatch.setattr(runner.server, "note_owned", lambda pid, url=None: owned.append(pid))
    comfy_found = {}
    monkeypatch.setattr(runner.server, "find_prompt", lambda prefix, url=None: comfy_found.get(prefix))
    monkeypatch.setattr(runner.server, "cancel_prompt", lambda pid, url=None: False)
    monkeypatch.setattr(runner.server, "gpu_lock", lambda url=None, timeout=0, note=None:
                        FileLock(tmp_path / "gpu.lock", timeout=timeout, note=note))
    monkeypatch.setattr(runner.video, "check_clip", lambda *a, **k: [])
    monkeypatch.setattr(runner.video, "contact_sheet", lambda *a, **k: None)
    import cinestudio.backend as backend_mod
    monkeypatch.setattr(backend_mod, "is_key_reuse", lambda exc: isinstance(exc, KeyReuse))

    def fake_passes(self, step):
        step["outputs"] = {"passes": "fake"}

    def fake_controls(self, step):
        ctl = self.work / "control.mp4"
        ctl.write_bytes(b"ctl")
        src = self.work / "source.mp4"                      # the render (video-to-video is the default)
        src.write_bytes(b"src")
        step["outputs"] = {"control": str(ctl), "source": str(src), "frames": 121, "fps": 24.0, "width": 768,
                           "height": 448, "shot_frames": 121}
    monkeypatch.setattr(runner.Run, "step_passes", fake_passes)
    monkeypatch.setattr(runner.Run, "step_controls", fake_controls)
    monkeypatch.setattr(runner.Run, "_blend_sound", lambda self: None)   # the fake shot has no sound of its own
    from cinestudio import character as _chars                # most tests use fake characters: the readiness rule
    monkeypatch.setattr(_chars, "require_video_ready", lambda name: {})   # has its own tests below
    monkeypatch.setattr(_chars, "measure_height", lambda body, shot=None:     # no Blender in the tests
                        {"m": 1.8 * float((body or {}).get("scale", 1.0)), "ref_m": 1.8,
                         "ratio": float((body or {}).get("scale", 1.0)), "body": dict(body or {})})
    return {"blend": blend, "sheet": sheet, "backend": FakeBackend(), "tmp": tmp_path, "owned": owned,
            "comfy_found": comfy_found, "mp": monkeypatch}


def new(env, **kw):
    return runner.new_run(env["blend"], env["sheet"], "a hooded figure", "it waves", **kw)["id"]


def start(env, rid, **kw):
    return runner.start(rid, backend_factory=lambda: env["backend"], **kw)


def set_state(rid, fn):
    d = runner.run_dir(rid)
    st = json.loads((d / "state.json").read_text())
    fn(st, d)
    (d / "state.json").write_text(json.dumps(st))


def ready_for_clip(st, d):
    for s in ("passes", "controls", "sheet"):
        st["steps"][s]["status"] = "done"
    st["steps"]["controls"]["outputs"] = {"control": str(d / "c.mp4"), "source": str(d / "src.mp4"), "frames": 121,
                                          "fps": 24.0, "width": 768, "height": 448, "shot_frames": 121}
    cv2.imwrite(str(d / "s.png"), np.zeros((64, 128, 3), np.uint8))
    st["steps"]["sheet"]["outputs"] = {"sheet": str(d / "s.png")}
    (d / "work").mkdir(exist_ok=True)


# ---------------------------------------------------------------- creation
def test_new_run_freezes_inputs_and_workflows(env):
    rid = new(env, quality="final", seed=7)
    env["blend"].write_bytes(b"changed later")
    d = runner.run_dir(rid)
    m = json.loads((d / "manifest.json").read_text())
    assert (d / "inputs" / "shot.blend").read_bytes() == b"BLENDER-fake"
    assert (d / "inputs" / "workflows" / f"{runner.PROFILE}_refup_api.json").is_file()
    assert set(m["workflows"]) >= {f"{runner.PROFILE}_ref_api.json", f"{runner.PROFILE}_ref.bind.json"}
    assert m["quality"] == "final" and m["seed"] == 7 and m["ingredients_strength"] == 1.0
    assert not any(p.name.startswith(".staging") for p in runner.runs_dir().iterdir())
    assert runner.status(rid)["status"] == "new"


def test_pass_key_depends_on_rig_and_blender(env):
    a = runner.new_run(env["blend"], env["sheet"], "c", "v")["pass_key"]
    b = runner.new_run(env["blend"], env["sheet"], "c", "v", rig="OtherRig")["pass_key"]
    env["mp"].setattr(runner.blender, "check", lambda: {"ok": True, "version": "5.3.0"})
    c = runner.new_run(env["blend"], env["sheet"], "c", "v")["pass_key"]
    assert len({a, b, c}) == 3


def test_bad_input_is_refused_and_leaves_nothing(env):
    with pytest.raises(runner.RunError):
        runner.new_run(env["sheet"], env["sheet"], "x", "y")             # not a .blend
    with pytest.raises(runner.RunError):
        new(env, restyle=True)                                           # restyle without a LoRA
    with pytest.raises(runner.RunError):
        new(env, quality="ultra")
    assert runner.list_runs() == []


def test_failed_snapshot_leaves_no_run(env):
    def broken(src, dst, report, allow_external=False):
        raise runner.RunError("the .blend uses files that cannot be frozen")
    env["mp"].setattr(runner, "snapshot_blend", broken)
    with pytest.raises(runner.RunError):
        new(env)
    assert not runner.runs_dir().exists() or not list(runner.runs_dir().iterdir())


def test_list_reports_a_broken_run_and_continues(env):
    rid = new(env)
    bad = runner.runs_dir() / "r-broken"
    bad.mkdir()
    (bad / "manifest.json").write_text("{not json")
    rows = {r["id"]: r for r in runner.list_runs()}
    assert rows[rid]["status"] == "new" and rows["r-broken"]["status"] == "unreadable"


# ---------------------------------------------------------------- running
def test_full_run_and_no_second_submit(env):
    rid = new(env, style_lora="style.safetensors")
    assert start(env, rid)["status"] == "done"
    assert (runner.run_dir(rid) / "final.mp4").read_bytes() == b"clip"
    g = env["backend"].submits[0]["graph"]
    assert g["lora_style"]["inputs"]["lora_name"] == "style.safetensors"
    assert env["owned"] == ["job1"]                                      # ownership recorded at submit
    assert start(env, rid)["status"] == "done"
    assert len(env["backend"].submits) == 1
    events = [json.loads(x)["event"] for x in (runner.run_dir(rid) / "events.jsonl").read_text().splitlines()]
    assert events[0] == "run_started" and "progress" in events and events[-1] == "run_done"


def test_frozen_workflow_is_used(env):
    rid = new(env)
    wf = runner.run_dir(rid) / "inputs" / "workflows" / f"{runner.PROFILE}_v2vref_api.json"     # video-to-video: the default
    g = json.loads(wf.read_text())
    g["sampler1"]["inputs"]["sampler_name"] = "frozen_test"               # the run's own copy, not the repo's
    wf.write_text(json.dumps(g))
    start(env, rid)
    assert env["backend"].submits[0]["graph"]["sampler1"]["inputs"]["sampler_name"] == "frozen_test"


def test_without_style_lora_the_slot_is_removed(env):
    rid = new(env)
    start(env, rid)
    assert "lora_style" not in env["backend"].submits[0]["graph"]


def test_first_frame_turns_the_guide_on(env):
    rid = runner.new_run(env["blend"], env["sheet"], "a hooded figure", "it turns", first_frame=env["sheet"])["id"]
    start(env, rid)
    assert env["backend"].submits[0]["graph"]["i2v"]["inputs"]["bypass"] is False
    start(env, new(env))
    assert env["backend"].submits[1]["graph"]["i2v"]["inputs"]["bypass"] is True


def test_second_worker_is_refused(env, request):
    rid = new(env)
    lock = runner.run_dir(rid) / "run.lock"
    hold = ("import sys, time; sys.path.insert(0, sys.argv[2]); from cinestudio.locks import FileLock; "
            "l = FileLock(sys.argv[1]).acquire(); print('held', flush=True); time.sleep(60)")
    exe = getattr(sys, "_base_executable", sys.executable)
    p = subprocess.Popen([exe, "-c", hold, str(lock), str(request.config.rootpath)], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "held"
        assert runner.worker_alive(rid)
        with pytest.raises(runner.RunError, match="already has a worker"):
            start(env, rid)
    finally:
        p.kill()
        p.wait()
    assert not runner.worker_alive(rid)
    assert start(env, rid)["status"] == "done"


# ---------------------------------------------------------------- recovery
def test_resume_finds_a_job_through_the_proxy(env):
    rid = new(env)

    def fn(st, d):
        ready_for_clip(st, d)
        st["steps"]["clip"] = {"status": "running", "attempts": 1,
                               "job": {"key": f"{rid}-clip-a1", "prefix": f"cinestudio/{rid}_clip_a1"}}
    set_state(rid, fn)
    env["backend"].found[f"cinestudio/{rid}_clip_a1"] = "job-before-stop"
    assert start(env, rid)["status"] == "done"
    assert env["backend"].submits == [] and env["owned"] == ["job-before-stop"]


def test_resume_finds_a_job_only_comfyui_knows(env):
    """The proxy claimed the key but lost its record: ComfyUI's own history still has the prompt."""
    rid = new(env)

    def fn(st, d):
        ready_for_clip(st, d)
        st["steps"]["clip"] = {"status": "unresolved", "attempts": 1,
                               "job": {"key": f"{rid}-clip-a1", "prefix": f"cinestudio/{rid}_clip_a1"}}
    set_state(rid, fn)
    env["comfy_found"][f"cinestudio/{rid}_clip_a1"] = "prompt-in-comfyui"
    assert start(env, rid)["status"] == "done"
    assert env["backend"].submits == []


def test_failed_job_then_new_attempt(env):
    rid = new(env)
    env["backend"].final = "failed"
    st = start(env, rid)
    assert st["status"] == "failed" and "boom" in st["error"]
    env["backend"].final = "succeeded"
    assert start(env, rid)["status"] == "done"
    assert [s["key"] for s in env["backend"].submits] == [f"{rid}-clip-a1", f"{rid}-clip-a2"]


def test_unresolved_submit_stops_without_a_new_attempt(env):
    rid = new(env)
    env["backend"].fail_submit = UnresolvedSubmission("p", "k", OSError("reset"))
    assert start(env, rid)["status"] == "unresolved"
    env["backend"].found[f"cinestudio/{rid}_clip_a1"] = "job-late"      # the job did reach the server
    assert start(env, rid)["status"] == "done"
    assert env["backend"].submits == []


def test_key_held_by_proxy_is_unresolved_not_failed(env):
    rid = new(env)
    env["backend"].used_keys.add(f"{rid}-clip-a1")                      # claimed, but no job anywhere
    st = start(env, rid)
    assert st["status"] == "unresolved" and "--new-attempt" in st["error"]
    assert start(env, rid)["status"] == "unresolved"                    # stays put, never 'failed'
    assert start(env, rid, new_attempt=True)["status"] == "done"
    assert env["backend"].submits[0]["key"] == f"{rid}-clip-a2"


def test_lost_connection_keeps_the_job_and_resumes(env):
    rid = new(env)
    env["backend"].watch_error = ConnectionRefusedError("[WinError 10061] refused")
    st = start(env, rid)
    assert st["status"] == "interrupted" and "job is kept" in st["error"]
    assert start(env, rid)["status"] == "done"
    assert len(env["backend"].submits) == 1                             # watched again, not submitted again


# ---------------------------------------------------------------- cancel
def test_cancel_of_a_canceled_job(env):
    rid = new(env)
    env["backend"].final = "canceled"
    assert start(env, rid)["status"] == "canceled"


def test_cancel_without_worker_settles_the_run(env):
    rid = new(env)
    r = runner.request_cancel(rid, backend_factory=lambda: env["backend"])
    assert r["worker"] is False
    assert runner.status(rid)["status"] == "canceled"


def test_cancel_finds_an_unconfirmed_job_first(env):
    rid = new(env)

    def fn(st, d):
        ready_for_clip(st, d)
        st["steps"]["clip"] = {"status": "unresolved", "attempts": 1,
                               "job": {"key": f"{rid}-clip-a1", "prefix": f"cinestudio/{rid}_clip_a1"}}
        st["status"] = "unresolved"
    set_state(rid, fn)
    env["backend"].found[f"cinestudio/{rid}_clip_a1"] = "job-x"
    env["backend"].jobs["job-x"] = "running"
    runner.request_cancel(rid, backend_factory=lambda: env["backend"])
    assert "job-x" in env["backend"].canceled
    assert runner.status(rid)["status"] == "canceled"


def test_cancel_during_submit_is_not_lost(env):
    rid = new(env)
    d = runner.run_dir(rid)
    env["backend"].on_submit = lambda: (d / "cancel.json").write_text("{}")   # arrives while submit runs
    assert start(env, rid)["status"] == "canceled"
    assert env["backend"].canceled == ["job1"]


def test_cancel_request_survives_worker_saves(env):
    rid = new(env)
    d = runner.run_dir(rid)

    def passes_then_cancel(self, step):
        (d / "cancel.json").write_text("{}")                           # requested during a Blender step
        self.save()                                                     # the worker writes its state after it
        step["outputs"] = {}
    env["mp"].setattr(runner.Run, "step_passes", passes_then_cancel)
    assert start(env, rid)["status"] == "canceled"
    assert env["backend"].submits == []


def test_start_after_a_settled_cancel_resumes(env):
    rid = new(env)
    runner.request_cancel(rid, backend_factory=lambda: env["backend"])
    assert runner.status(rid)["status"] == "canceled"
    assert start(env, rid)["status"] == "done"


def test_delete_run(env):
    rid = new(env)
    start(env, rid)
    assert runner.delete_run(rid)["deleted"]
    assert rid not in [r["id"] for r in runner.list_runs()]
    with pytest.raises(runner.RunError):
        runner.delete_run(rid)


def test_a_late_worker_obeys_a_cancel_made_after_its_launch(env):
    rid = new(env)
    gen = runner.new_launch(rid)                         # the requester launches a worker ...
    runner.request_cancel(rid, backend_factory=lambda: env["backend"])   # ... the user cancels before it starts
    st = start(env, rid, launch=gen)                     # the late worker must not run
    assert st["status"] == "canceled" and not env["backend"].submits
    assert (runner.run_dir(rid) / "cancel.json").is_file()


def test_a_new_explicit_launch_resumes_after_a_cancel(env):
    rid = new(env)
    runner.request_cancel(rid, backend_factory=lambda: env["backend"])
    assert start(env, rid, launch=runner.new_launch(rid))["status"] == "done"


def test_request_cancel_returns_a_status_string(env):
    rid = new(env)
    r = runner.request_cancel(rid, backend_factory=lambda: env["backend"])
    assert r["status"] == "canceled"


def _proxy_forgot(env, rid):
    def fn(st, d):
        ready_for_clip(st, d)
        st["steps"]["clip"] = {"status": "interrupted", "attempts": 1,
                               "job": {"id": "lost-1", "key": f"{rid}-clip-a1", "prefix": f"cinestudio/{rid}_clip_a1"}}
    set_state(rid, fn)

    def state(jid):
        if jid == "lost-1":
            raise RuntimeError("job not found")
        return runner_state_orig(jid)
    runner_state_orig = env["backend"].state
    env["backend"].state = state


def test_proxy_forgot_but_comfyui_finished_it(env, monkeypatch):
    rid = new(env)
    _proxy_forgot(env, rid)
    monkeypatch.setattr(runner.server, "prompt_status", lambda url, pid: "done_ok")
    monkeypatch.setattr(runner.server, "download_output", lambda url, pid, node, dst: Path(dst).write_bytes(b"clip"))
    assert start(env, rid)["status"] == "done"
    assert env["backend"].submits == []                         # no second GPU job


def test_proxy_and_comfyui_forgot_it_is_unresolved(env, monkeypatch):
    rid = new(env)
    _proxy_forgot(env, rid)
    monkeypatch.setattr(runner.server, "prompt_status", lambda url, pid: "unknown")
    assert start(env, rid)["status"] == "unresolved"
    assert env["backend"].submits == []                         # never resubmitted by itself


def test_source_sigmas_have_eight_steps_from_the_start_noise():
    s = [float(x) for x in runner.source_sigmas(0.979).split(",")]
    assert len(s) == 9 and s[0] == 0.979 and s[-1] == 0 and all(a > b for a, b in zip(s, s[1:]))
    assert runner.source_sigmas(1.0).startswith("1, 0.99375")           # the full distilled schedule
    with pytest.raises(runner.RunError):
        runner.source_sigmas(0)


def test_video_to_video_is_the_default_and_can_be_turned_off(env):
    rid = new(env)
    start(env, rid)
    g = env["backend"].submits[0]["graph"]
    assert "src_enc" in g and g["sigmas1"]["inputs"]["sigmas"].startswith("0.979")   # follows the render
    rid2 = new(env, source_denoise=False)
    start(env, rid2)
    assert "src_enc" not in env["backend"].submits[1]["graph"]                          # control only


def test_control_video_is_frozen_with_the_run(env, tmp_path):
    ctl = tmp_path / "skeletons.mp4"
    ctl.write_bytes(b"not really a video")
    m = runner.new_run(env["blend"], env["sheet"], "three figures", "they stand", control_video=ctl)
    assert m["control_video"] == {"source": str(ctl.resolve()), "file": "inputs/control.mp4"}
    assert (runner.run_dir(m["id"]) / "inputs" / "control.mp4").read_bytes() == b"not really a video"
    assert runner.new_run(env["blend"], env["sheet"], "a", "b")["control_video"] is None
    with pytest.raises(runner.RunError, match="control video not found"):
        runner.new_run(env["blend"], env["sheet"], "a", "b", control_video=tmp_path / "missing.mp4")


def _two_clip_controls(env, calls, overlap=None):
    def fake_controls(self, step):
        for name in ("control.mp4", "source.mp4", "full.mp4"):
            (self.work / name).write_bytes(b"x")
        step["outputs"] = {"control": str(self.work / "control.mp4"), "control_full": str(self.work / "full.mp4"),
                           "source": str(self.work / "source.mp4"), "frames": 121, "fps": 24.0, "width": 768,
                           "height": 448, "shot_frames": 200, "clips": 2,
                           **({"overlap": overlap} if overlap else {})}
    mp = env["mp"]
    mp.setattr(runner.Run, "step_controls", fake_controls)
    mp.setattr(runner.Run, "_source_slice", lambda self, k, fmt: calls.setdefault("source", []).append(k) or fmt["source"])
    mp.setattr(runner.video, "slice_control", lambda src, dst, k, *a, **kw: calls.setdefault("slice", []).append(k)
               or Path(dst).write_bytes(b"c"))
    mp.setattr(runner.video, "last_frame", lambda src, dst: cv2.imwrite(str(dst), np.zeros((448, 768, 3), np.uint8)))
    mp.setattr(runner.video, "last_frames", lambda src, dst, n: calls.setdefault("last_frames", []).append(n)
               or Path(dst).write_bytes(b"frames"))
    mp.setattr(runner.video, "stitch", lambda clips, dst, fps, total_frames=None, overlap=1, **kw:
               calls.setdefault("stitch", []).append((list(clips), total_frames))
               or calls.setdefault("overlap", []).append(overlap) or Path(dst).write_bytes(b"joined"))


def test_a_long_shot_is_made_of_clips_that_continue_each_other(env):
    calls = {}
    _two_clip_controls(env, calls)
    rid = new(env)
    st = start(env, rid)
    assert st["status"] == "done"
    subs = env["backend"].submits
    assert len(subs) == 2
    assert subs[0]["graph"]["i2v"]["inputs"]["bypass"] is True      # clip 1: no first frame given
    assert subs[1]["graph"]["i2v"]["inputs"]["bypass"] is False     # clip 2 starts on clip 1's last frame
    assert calls["slice"] == [2] and calls["source"] == [2]
    clips, total = calls["stitch"][0]
    assert [Path(c).name for c in clips] == ["clip.mp4", "clip_2.mp4"] and total == 200
    assert (runner.run_dir(rid) / "final.mp4").read_bytes() == b"joined"


def test_a_finished_clip_is_not_made_again(env):
    calls = {}
    _two_clip_controls(env, calls)
    rid = new(env)

    def clip1_done(st, d):
        st["steps"]["clip"]["parts"] = {"1": {"outputs": {"clip": str(d / "work" / "clip.mp4")}}}
    set_state(rid, clip1_done)
    (runner.run_dir(rid) / "work").mkdir(exist_ok=True)
    (runner.run_dir(rid) / "work" / "clip.mp4").write_bytes(b"clip")
    st = start(env, rid)
    assert st["status"] == "done"
    assert len(env["backend"].submits) == 1                        # only clip 2
    assert env["backend"].submits[0]["graph"]["i2v"]["inputs"]["bypass"] is False


def _clip2_interrupted(rid, job2):
    def fn(st, d):
        ready_for_clip(st, d)
        st["steps"]["clip"] = {"status": "interrupted", "attempts": 1, "job": {"id": "job1", "key": "k1"},
                               "parts": {"1": {"outputs": {"clip": str(d / "work" / "clip.mp4")}},
                                         "2": {"attempts": 1, "job": job2}}}
        st["status"] = "interrupted"
    set_state(rid, fn)


def test_cancel_of_an_interrupted_run_stops_the_job_of_clip_2(env):
    rid = new(env)
    _clip2_interrupted(rid, {"id": "job2", "key": f"{rid}-clip2-a1", "prefix": f"cinestudio/{rid}_clip2_a1"})
    env["backend"].jobs.update({"job1": "succeeded", "job2": "running"})
    r = runner.request_cancel(rid, backend_factory=lambda: env["backend"])
    assert "job2" in env["backend"].canceled and "job1" not in env["backend"].canceled
    assert runner.status(rid)["status"] == "canceled" and "job2" in r["jobs_canceled"]


def test_cancel_finds_an_unconfirmed_job_of_clip_2(env):
    rid = new(env)
    _clip2_interrupted(rid, {"key": f"{rid}-clip2-a1", "prefix": f"cinestudio/{rid}_clip2_a1"})   # id never saved
    env["backend"].found[f"cinestudio/{rid}_clip2_a1"] = "job-y"
    env["backend"].jobs["job-y"] = "running"
    runner.request_cancel(rid, backend_factory=lambda: env["backend"])
    assert "job-y" in env["backend"].canceled
    assert runner.status(rid)["status"] == "canceled"


def test_job_records_skip_finished_clips():
    step = {"job": {"id": "a"}, "parts": {"1": {"outputs": {"clip": "x"}}, "2": {"job": {"id": "b"}},
                                         "3": {"job": {"id": "c"}, "outputs": {"clip": "y"}}}}
    assert [j["id"] for j in runner.job_records(step)] == ["b"]
    assert [j["id"] for j in runner.job_records({"job": {"id": "a"}})] == ["a"]


def test_run_details_rebuild_the_full_prompt_of_an_older_run(tmp_path, monkeypatch):
    import json
    from cinestudio import runner
    monkeypatch.setattr(runner, "runs_dir", lambda: tmp_path)
    d = tmp_path / "r1"
    (d / "work").mkdir(parents=True)
    (d / "work" / "reference.png").write_bytes(b"x")
    (d / "work" / "first_2.png").write_bytes(b"x")
    m = {"id": "r1", "prompt": {"character": "C" * 900, "video": "V", "style": "S"}, "sheet": {"file": "inputs/sheet.png",
         "source": "src.png"}, "blend": {"source": "shot.blend"}, "style_lora": None, "style_lora_strength": 0.8}
    (d / "manifest.json").write_text(json.dumps(m))
    st = {"steps": {"controls": {"outputs": {"clips": 2}}, "clip": {"parts": {"2": {}}}}}
    (d / "state.json").write_text(json.dumps(st))
    det = runner.run_details("r1")
    assert det["clips"][0]["prompt"].count("C") == 900 and det["clips"][0]["prompt_source"] == "rebuilt"   # not cut
    assert det["clips"][0]["first_image"] is None and det["clips"][1]["first_image"].endswith("first_2.png")
    assert det["settings"]["style_lora"] == "none" and det["settings"]["style_lora_strength"] is None
    st["steps"]["clip"]["prompt"] = "exact text"
    (d / "state.json").write_text(json.dumps(st))
    assert runner.run_details("r1")["clips"][0] == {**runner.run_details("r1")["clips"][0], "prompt": "exact text",
                                                    "prompt_source": "recorded"}


def test_run_details_show_the_note_and_what_differs_from_a_default_run(tmp_path, monkeypatch):
    import json
    from cinestudio import runner
    monkeypatch.setattr(runner, "runs_dir", lambda: tmp_path)
    d = tmp_path / "r2"
    d.mkdir()
    m = {"id": "r2", "prompt": {"character": "C", "video": "V", "style": ""}, "sheet": {"file": "inputs/sheet.png",
         "source": "s.png"}, "blend": {"source": "b.blend"}, "style_lora": None, "style_lora_strength": 0.8,
         "source_denoise": 0.99, "control_video": {"source": "sd.mp4", "file": "inputs/control.mp4"}}
    (d / "manifest.json").write_text(json.dumps(m))
    (d / "state.json").write_text(json.dumps({"steps": {}}))
    runner.set_note("r2", "  tests SDPose  ")
    det = runner.run_details("r2")
    names = {c["name"]: c["value"] for c in det["changes"]}
    assert det["note"] == "tests SDPose"
    assert names["source_denoise"] == 0.99 and names["style_lora"] == "none" and names["control video"] == "sd.mp4"
    assert "style_lora_strength" not in names                 # meaningless without a LoRA


def test_fork_keeps_the_first_clips_and_makes_the_rest_again(tmp_path, monkeypatch):
    import json
    import pytest
    from cinestudio import runner
    monkeypatch.setattr(runner, "runs_dir", lambda: tmp_path)
    src = tmp_path / "r1"
    (src / "inputs").mkdir(parents=True)
    (src / "work").mkdir()
    for f in ("clip.mp4", "clip_2.mp4", "clip_3.mp4", "first_2.png", "first_3.png", "last_1.png", "last_2.png", "control.mp4"):
        (src / "work" / f).write_bytes(b"x")
    (src / "manifest.json").write_text(json.dumps({"id": "r1", "prompt": {"character": "C", "video": "old", "style": "S"},
                                                    "sheet": {"file": "inputs/sheet.png", "source": "s"}, "blend": {"source": "b"}}))
    w = str(src / "work")
    st = {"status": "done", "steps": {
        "controls": {"status": "done", "outputs": {"clips": 3, "control": w + "/control.mp4"}},
        "clip": {"status": "done", "job": {"id": "j1"}, "outputs": {"clip": w + "/clip.mp4"},
                 "parts": {"1": {"outputs": {"clip": w + "/clip.mp4"}}, "2": {"job": {"id": "j2"}, "outputs": {"clip": w + "/clip_2.mp4"}},
                           "3": {"job": {"id": "j3"}, "outputs": {"clip": w + "/clip_3.mp4"}}}},
        "final": {"status": "done", "outputs": {"final": str(src / "final.mp4")}}}}
    (src / "state.json").write_text(json.dumps(st))
    m = runner.fork_run("r1", 2, "test note", video="new text", describe_facing=True)
    d = tmp_path / m["id"]
    st2 = json.loads((d / "state.json").read_text())
    assert m["prompt"]["video"] == "new text" and m["describe_facing"] and m["forked_from"]["run"] == "r1"
    assert set(st2["steps"]["clip"]["parts"]) == {"1"} and st2["steps"]["clip"]["status"] == "pending"
    assert st2["steps"]["final"] == {"status": "pending"} and st2["status"] == "new"
    assert (d / "work" / "clip.mp4").is_file() and not (d / "work" / "clip_2.mp4").exists() and not (d / "work" / "last_1.png").exists()
    assert str(d) in st2["steps"]["controls"]["outputs"]["control"]          # paths point into the new run
    assert runner.job_records(st2["steps"]["clip"]) == []                    # clip 1's old job is not live
    with pytest.raises(runner.RunError):
        runner.fork_run("r1", 2, seed=5)                                      # only prompt-side changes
    with pytest.raises(runner.RunError, match="no saved latent"):          # B needs clip 1's relay latent
        runner.fork_run("r1", 2, adain=0.3, adain_ref="clip1")
    st["steps"]["clip"]["parts"]["1"]["relay"] = {"s1": "relay_r1_1_s1.latent"}
    (src / "state.json").write_text(json.dumps(st))
    m3 = runner.fork_run("r1", 2, look_chain="anchor", adain=0.3, adain_ref="clip1", chain_colour="match")
    assert (m3["look_chain"], m3["adain"], m3["adain_ref"], m3["chain_colour"]) == ("anchor", 0.3, "clip1", "match")
    assert sorted(m3["forked_from"]["changes"]) == ["adain", "adain_ref", "chain_colour", "look_chain"]
    m4 = runner.fork_run("r1", 2, chain_light="look", chain_sharpen_mode="match")   # 2026-10-23: they were dropped
    assert (m4["chain_light"], m4["chain_sharpen_mode"]) == ("look", "match")
    with pytest.raises(runner.RunError, match="chain light"):
        runner.fork_run("r1", 2, chain_light="bright")
    m5 = runner.fork_run("r1", 2, chain_sharpen_mode="match", chain_sharpen_target=1.5)
    assert m5["chain_sharpen_target"] == 1.5
    with pytest.raises(runner.RunError, match="sharpen target"):
        runner.fork_run("r1", 2, chain_sharpen_target=9)


def test_fork_relocates_only_path_fields_and_refuses_a_working_source(tmp_path, monkeypatch):
    import json
    import pytest
    from cinestudio import runner
    monkeypatch.setattr(runner, "runs_dir", lambda: tmp_path)
    src = tmp_path / "r1"
    (src / "inputs").mkdir(parents=True)
    (src / "work").mkdir()
    for f in ("clip.mp4", "clip_2.mp4", "control.mp4"):
        (src / "work" / f).write_bytes(b"x")
    (src / "manifest.json").write_text(json.dumps({"id": "r1", "prompt": {"character": "C", "video": "v", "style": ""},
                                                    "sheet": {"file": "inputs/sheet.png", "source": "s"}, "blend": {"source": "b"}}))
    fwd = str(src / "work" / "control.mp4").replace("\\", "/")              # another separator
    text = f"see {src}"                                                     # a path inside a prompt: kept verbatim
    st = {"status": "done", "steps": {
        "controls": {"status": "done", "outputs": {"clips": 2, "control": fwd, "control_full": str(tmp_path / "r1-x" / "c.mp4")}},
        "clip": {"status": "done", "prompt": text, "job": {"prefix": str(src / "work")},
                 "parts": {"1": {"outputs": {"clip": str(src / "work" / "clip.mp4")}}, "2": {"outputs": {"clip": "x"}}}},
        "final": {"status": "done"}}}
    (src / "state.json").write_text(json.dumps(st))
    monkeypatch.setattr(runner, "worker_alive", lambda rid: True)
    with pytest.raises(runner.RunError, match="working"):
        runner.fork_run("r1", 2)
    monkeypatch.setattr(runner, "worker_alive", lambda rid: False)
    m = runner.fork_run("r1", 2)
    st2 = json.loads((tmp_path / m["id"] / "state.json").read_text())
    outs = st2["steps"]["controls"]["outputs"]
    assert outs["control"].startswith(str(tmp_path / m["id"]))             # forward slashes relocated too
    assert outs["control_full"] == str(tmp_path / "r1-x" / "c.mp4")          # a look-alike prefix is not touched
    assert st2["steps"]["clip"]["prompt"] == text and st2["steps"]["clip"]["job"] == {"prefix": str(src / "work")}
    assert st2["steps"]["clip"]["parts"]["1"]["outputs"]["clip"] == str(tmp_path / m["id"] / "work" / "clip.mp4")


def test_a_control_video_over_2_gib_is_refused(tmp_path, monkeypatch):
    import pytest
    from cinestudio import runner
    (tmp_path / "x.blend").write_bytes(b"BLENDER")
    (tmp_path / "s.png").write_bytes(b"png")
    big = tmp_path / "c.mp4"                       # the limit is lowered for the test: on Windows the "sparse" seek
    big.write_bytes(b"\0" * 2048)                  # wrote a real 2 GB file in every test run
    monkeypatch.setattr(runner, "MAX_CONTROL_BYTES", 1024)
    with pytest.raises(runner.RunError, match="2 GiB"):
        runner.new_run(tmp_path / "x.blend", tmp_path / "s.png", "c", "v", control_video=big)


def test_the_shots_own_sound_replaces_ltxs(env):
    calls = []
    wav = env["tmp"] / "song.wav"
    wav.write_bytes(b"RIFF")
    env["mp"].setattr(runner.Run, "_blend_sound", lambda self: wav)
    env["mp"].setattr(runner.video, "replace_audio", lambda v, a, dst: (calls.append(a), Path(dst).write_bytes(Path(v).read_bytes())))
    rid = new(env)                                              # sound defaults to "auto"
    assert start(env, rid)["status"] == "done"
    st = json.loads((runner.run_dir(rid) / "state.json").read_text())
    assert calls == [wav] and st["steps"]["final"]["outputs"]["sound"] == "blender"
    rid2 = new(env, sound="ltx")                                # keep LTX's own sound
    assert start(env, rid2)["status"] == "done"
    st2 = json.loads((runner.run_dir(rid2) / "state.json").read_text())
    assert len(calls) == 1 and st2["steps"]["final"]["outputs"]["sound"] == "ltx"


def test_a_library_song_wins_over_the_shots_sound(env):
    from cinestudio import song as songs
    calls, cuts = [], []
    d = songs.song_dir("groove")
    d.mkdir(parents=True)
    (d / "song.wav").write_bytes(b"RIFF")
    songs._write(d / "song.json", {"name": "groove", "status": "done", "file": str(d / "song.wav"),
                                   "beats": {"seconds": 30.0}})
    env["mp"].setattr(runner.Run, "_blend_sound", lambda self: env["tmp"] / "never.wav")
    env["mp"].setattr(runner.video, "cut_audio", lambda src, s, dst: (cuts.append((Path(src).name, s)),
                                                                      Path(dst).write_bytes(b"RIFF")))
    env["mp"].setattr(runner.video, "replace_audio", lambda v, a, dst: (calls.append(Path(a).name),
                                                                        Path(dst).write_bytes(Path(v).read_bytes())))
    rid = new(env, song="groove", song_start=12.5)
    m = json.loads((runner.run_dir(rid) / "manifest.json").read_text())
    assert m["song"]["file"] == "inputs/song.wav" and (runner.run_dir(rid) / "inputs" / "song.wav").is_file()
    assert start(env, rid)["status"] == "done"
    st = json.loads((runner.run_dir(rid) / "state.json").read_text())
    assert cuts == [("song.wav", 12.5)] and calls == ["song_cut.wav"]
    assert st["steps"]["final"]["outputs"]["sound"] == "song"
    with pytest.raises(songs.SongError):
        new(env, song="missing")


def test_clips_continue_from_the_last_frames_of_the_clip_before(env):
    calls = {}
    _two_clip_controls(env, calls, overlap=25)
    rid = new(env)
    assert start(env, rid)["status"] == "done"
    g1, g2 = (x["graph"] for x in env["backend"].submits)
    assert g1["i2v"]["inputs"]["strength"] == 1.0
    assert g2["i2v"]["inputs"]["bypass"] is False and g2["i2v"]["inputs"]["strength"] == runner.OVERLAP_STRENGTH
    assert g2["first_load"]["class_type"] == "LoadVideo"
    assert calls["last_frames"] == [25] and calls["overlap"] == [25]
    m = json.loads((runner.run_dir(rid) / "manifest.json").read_text())
    assert m["overlap"] == runner.OVERLAP
    with pytest.raises(runner.RunError, match="overlap"):
        new(env, overlap=20)


REAL_STEP_PASSES = runner.Run.step_passes          # the env fixture fakes it; this test needs the real one


def test_a_run_from_a_video_template_reads_passes_from_the_video(env):
    import subprocess
    from cinestudio import shots, video
    vid = env["tmp"] / "tpl.mp4"
    subprocess.run([video.ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x32:r=24:d=1",
                    "-frames:v", "24", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vid)], check=True)
    shots.add_video("beach", vid, "a stand-in dances")
    built = []

    def fake_video_passes(self, stage):
        built.append(Path(self.d / self.m["template"]["file"]).name)
        stage.mkdir(parents=True)
        (stage / "passes.json").write_text("{}")
    env["mp"].setattr(runner.Run, "_video_passes", fake_video_passes)
    env["mp"].setattr(runner.passes, "load", lambda d: {})
    env["mp"].setattr(runner.passes, "make_controls", lambda *a, **k: {})
    env["mp"].setattr(runner, "control_size", lambda info: (64, 32))
    m = runner.new_run(None, env["sheet"], "a hooded figure", "it dances", template="beach", describe_facing=True)
    assert m["blend"] is None and m["template"]["name"] == "beach" and m["pass_key"].startswith("vt_")
    assert m["describe_facing"] is False                     # the facing words need Blender's angles
    assert m["body_type"] == "standard" and m["source_denoise"] == runner.TEMPLATE_SOURCE_DENOISE
    assert (runner.run_dir(m["id"]) / "inputs" / "template.mp4").is_file()
    run = runner.Run(m["id"])
    step = {}
    REAL_STEP_PASSES(run, step)
    assert built == ["template.mp4"] and step["outputs"]["cached"] is False
    assert runner.shot_source(m).startswith("template beach")
    shots._write(shots.shot_dir("orbit") / "shot.json", {"name": "orbit", "kind": "blender", "status": "done",
                                                         "files": {"blend": "x.blend"}})
    with pytest.raises(runner.RunError, match="not a ready video template"):
        runner.new_run(None, env["sheet"], "c", "v", template="orbit")
    with pytest.raises(shots.ShotError, match="no template"):
        runner.new_run(None, env["sheet"], "c", "v", template="missing")


def test_background_options_freeze_the_panorama_graph_or_the_saved_panorama(env):
    from cinestudio import background
    rid = new(env, background="text", background_text="a neon rooftop at night")
    d = runner.run_dir(rid)
    m = json.loads((d / "manifest.json").read_text())
    st = json.loads((d / "state.json").read_text())
    assert list(st["steps"])[0] == "background" and (d / "inputs" / "workflows" / "qwenimage_pano_api.json").is_file()
    g = json.loads((d / "inputs" / "workflows" / "qwenimage_pano_api.json").read_text())
    assert g["unet"]["inputs"]["unet_name"].endswith(".gguf")                  # the tier's panorama model, frozen
    assert "background_hold" not in m and "pano_model" in m["models"]["files"]
    png = env["sheet"]
    rec = background.save(png, "p", 1, name="rooftop")
    rid2 = new(env, background="saved:rooftop")
    assert (runner.run_dir(rid2) / "inputs" / "background.png").is_file() and rec["name"] == "rooftop"
    with pytest.raises(runner.RunError, match="background must be"):
        new(env, background="sky")
    with pytest.raises(runner.RunError, match="no background"):
        new(env, background="saved:nothing")
    with pytest.raises(runner.RunError, match="unknown options"):        # the hold (a mask) is gone
        new(env, background="text", background_hold=0.9)


def test_panorama_seam_is_closed(tmp_path):
    from cinestudio import background
    img = np.zeros((64, 128, 3), np.uint8)
    img[:, :64] = 200                                       # left half bright, right half dark: a hard seam at the wrap
    p = tmp_path / "pano.png"
    cv2.imwrite(str(p), img)
    background.fix_seam(p, width=8)
    out = cv2.imread(str(p)).astype(int)
    assert abs(out[10, 0, 0] - out[10, -1, 0]) < 20          # the two edges now meet
    assert out[10, 30, 0] == 200 and out[10, 100, 0] == 0    # the inside is untouched
    assert "equirectangular" in background.prompt_for("a quiet harbour")


def test_the_clip_prompt_names_the_panorama_place():
    m = {"prompt": {"character": "c", "style": "anime", "video": "It waves."}, "background": "text",
         "background_text": "A lantern garden at dusk"}
    assert "The place: a lantern garden at dusk." in runner.video_prompt(m)
    m["prompt"]["video"] = "It waves in a lantern garden at dusk."
    assert "The place:" not in runner.video_prompt(m)                      # already in the text
    assert "The place:" not in runner.video_prompt({**m, "background": None})


def test_layout_mode_makes_the_look_then_a_layout_clip(env):
    from cinestudio import character as chars
    mp = env["mp"]
    look_frame = env["tmp"] / "rgb_0001.png"
    cv2.imwrite(str(look_frame), np.zeros((448, 768, 3), np.uint8))
    mp.setattr(runner.passes, "load", lambda d: {"rgb": [str(look_frame)]})
    mp.setattr(chars, "load", lambda name: {"description": "A sheet on black: a slim hooded figure.",
                                            "files": {"front": str(env["sheet"])}})
    with pytest.raises(runner.RunError, match="character from the library"):
        new(env, profile="ltx25", render_mode="layout")
    mp.setattr(runner.video, "image_video", lambda src, dst, fps: Path(dst).write_bytes(b"first"))
    rid = new(env, profile="ltx25", render_mode="layout", character_name="cloaked", look_extra="A LONG poncho.")
    m = json.loads((runner.run_dir(rid) / "manifest.json").read_text())
    assert runner.steps_of(m)[-3:] == ["look", "clip", "final"]

    def gone(name):                                    # the run has its own frozen views: no library needed (Codex)
        raise chars.CharacterError("deleted")
    mp.setattr(chars, "load", gone)
    assert (runner.run_dir(rid) / "inputs" / "workflows" / "ltx25_layouting_api.json").is_file()
    st = start(env, rid)
    assert st["status"] == "done"
    look_g, clip_g = env["backend"].submits[0]["graph"], env["backend"].submits[1]["graph"]
    who = m["prompt"]["character"].split("on black: ", 1)[-1].strip()
    assert "LONG poncho" in json.dumps(look_g) and who[:20] in json.dumps(look_g)       # the FROZEN description
    assert "g_lay" in clip_g and "g_look" in clip_g and "lora_ing" in clip_g      # Ingredients 1.0 > 0: layouting
    assert clip_g["sigmas1"]["inputs"]["sigmas"].startswith("1, 0.99375")       # the distilled grid from pure noise


def test_layout_is_the_default_for_a_blender_shot_with_a_library_character(env):
    m1 = runner.new_run(env["blend"], env["sheet"], "a hooded figure", "it waves", character_name="cloaked")
    assert m1["render_mode"] == "layout" and m1["profile"] == runner.LAYOUT_PROFILE
    m2 = runner.new_run(env["blend"], env["sheet"], "a hooded figure", "it waves")           # a sheet file only
    assert m2["render_mode"] == "v2v" and m2["profile"] == runner.PROFILE
    m3 = runner.new_run(env["blend"], env["sheet"], "c", "v", character_name="cloaked", render_mode="v2v")
    assert m3["render_mode"] == "v2v"


def test_layout_makes_a_view_matched_keyframe_look_for_a_turning_body(env):
    from cinestudio import character as chars
    mp = env["mp"]
    frame = env["tmp"] / "rgb.png"
    cv2.imwrite(str(frame), np.zeros((448, 768, 3), np.uint8))
    mp.setattr(runner.passes, "load", lambda d: {"rgb": [str(frame)] * 121, "mask": [str(frame)] * 121})
    from cinestudio import sheet_keyframe
    made = []
    mp.setattr(sheet_keyframe, "make", lambda rgb, mask, sheet, view, out, **kw: (made.append((view, out, kw)),
                                                                                  Path(out).write_bytes(b"kf"))[1])
    side = env["tmp"] / "right.png"
    cv2.imwrite(str(side), np.full((64, 64, 3), 9, np.uint8))
    mp.setattr(chars, "load", lambda name: {"description": "on black: a figure",
                                            "files": {"front": str(env["sheet"]), "right": str(side)}})
    mp.setattr(runner.Run, "_body_yaws", lambda self: [[float(i), "left"] for i in range(121)])   # turns 0 -> 120 deg
    mp.setattr(runner.video, "image_video", lambda src, dst, fps: Path(dst).write_bytes(b"first"))
    rid = new(env, profile="ltx25", render_mode="layout", character_name="cloaked")
    m = json.loads((runner.run_dir(rid) / "manifest.json").read_text())
    assert set(m["views"]) == {"front", "right"} and (runner.run_dir(rid) / m["views"]["right"]).is_file()   # frozen
    st = start(env, rid)
    assert st["status"] == "done"
    graphs = [s["graph"] for s in env["backend"].submits]
    assert len(graphs) == 3                                   # the look, the keyframe look (frame 60: 60 deg), the clip
    kf = json.loads((runner.run_dir(rid) / "state.json").read_text())["steps"]["look"]["outputs"]["keyframes"]
    assert kf[0] == {"clip": 1, "frame": 60, "file": kf[0]["file"], "view": "right", "body_deg": 60.0}
    # + the sheet's side panel at the purest profile (90 deg = frame 90), made without a GPU job
    assert kf[1] == {"clip": 1, "frame": 90, "file": kf[1]["file"], "view": "right", "body_deg": 90.0,
                     "source": "sheet"} and made[0][0] == "right"
    clip = graphs[2]
    assert clip["g_kf0_s1"]["inputs"]["frame_idx"] == 60 and clip["g_kf1_s1"]["inputs"]["frame_idx"] == 90
    assert clip["guider1"]["inputs"]["positive"] == ["g_kf1_s1", 0] and clip["av2"]["inputs"]["video_latent"] == ["g_kf1_s2", 2]


def test_through_clip_retake_use_take_then_continue(env):
    """The gate workflow: make clip 1 only, retake it with another seed, put the take in, then clip 2 continues from
    the TAKE's latents."""
    from cinestudio import character as chars
    calls = {}
    _two_clip_controls(env, calls, overlap=25)
    mp = env["mp"]
    frame = env["tmp"] / "rgb.png"
    cv2.imwrite(str(frame), np.zeros((448, 768, 3), np.uint8))
    mp.setattr(runner.passes, "load", lambda d: {"rgb": [str(frame)] * 200})
    mp.setattr(chars, "load", lambda name: {"description": "on black: a figure", "files": {"front": str(env["sheet"])}})
    mp.setattr(runner.Run, "_body_yaws", lambda self: None)
    mp.setattr(runner.Run, "_relay_inputs", lambda self, files: None)
    mp.setattr(runner, "relay_latent", lambda prefix, keep_dir=None: Path(prefix).name + ".latent")
    mp.setattr(runner.video, "image_video", lambda src, dst, fps: Path(dst).write_bytes(b"first"))
    mp.setattr(runner.Run, "step_final", lambda self, step: (self.d / "final.mp4").write_bytes(b"joined"))
    mp.setattr(runner.Run, "_audio_tail", lambda self, name, fmt: Path(name).stem + "_tail.latent")   # no real file
    chained = []                                                     # look_chain "prev" (default): clip k-1's frame
    mp.setattr(runner.video, "frame_png", lambda src, dst, index: (chained.append((Path(src).name, index)),
                                                                   cv2.imwrite(str(dst), np.zeros((8, 8, 3), np.uint8))))
    rid = new(env, profile="ltx25", character_name="cloaked")
    mf = json.loads((runner.run_dir(rid) / "manifest.json").read_text())
    assert mf["relay"] == "latent"
    assert (mf["adain_ref"], mf["adain"]) == ("clip1", 0.3)          # option B: the default of layout runs (2026-10-22)
    runner.set_clip_limit(rid, 1)
    assert start(env, rid)["status"] == "paused"
    subs = env["backend"].submits
    assert len(subs) == 2                                            # the look and clip 1 only
    r = runner.add_take(rid, 1, {"seed": 7}, note="another seed")
    assert r["take"] == 1
    with pytest.raises(runner.RunError, match="cannot change"):
        runner.add_take(rid, 1, {"overlap": 9})
    with pytest.raises(runner.RunError, match="look image .* does not exist"):   # a take's own frame -1 look
        runner.add_take(rid, 1, {"look_image": str(env["tmp"] / "missing.png")})
    with pytest.raises(runner.RunError, match="chain colour"):
        runner.add_take(rid, 1, {"chain_colour": "rainbow"})
    assert start(env, rid)["status"] == "paused"                     # the take is made, the run stops again
    take_graph = subs[2]["graph"]
    assert any(n["class_type"] == "RandomNoise" and n["inputs"].get("noise_seed") == 7 for n in take_graph.values())
    t = runner.list_takes(rid)["takes"]["1"][0]
    assert t["status"] == "done" and t["outputs"]["relay"]["s1"].endswith("_1_t1_s1.latent")
    runner.use_take(rid, 1, 1)
    assert runner.list_takes(rid)["takes"]["1"][0]["original"]       # the original kept as take 0
    runner.set_clip_limit(rid, None)
    assert start(env, rid)["status"] == "done"
    assert chained and chained[-1][1] == 121 - 25                    # clip 2's look = clip 1's frame 96 (shared start)
    st = json.loads((runner.run_dir(rid) / "state.json").read_text())
    assert st["steps"]["clip"]["parts"]["2"]["first_image"].endswith("look_chain_2.png")
    clip2 = subs[-1]["graph"]
    assert clip2["relay_load_s1"]["inputs"]["latent"].endswith("_1_t1_s1.latent")   # clip 2 continues from the take
    assert clip2["atail_load"]["inputs"]["latent"].endswith("_1_t1_a_tail.latent")  # and its sound (the audio carry)
    assert clip2["ks"]["inputs"]["latent_image"] == ["av_carry", 0] and "audio_save" in clip2


def test_adain_ref_clip1_needs_the_latent_relay(env):
    """Codex review 2026-10-22: B reads clip 1's saved latent, which only the latent relay saves."""
    with pytest.raises(runner.RunError, match="adain ref clip1"):
        new(env, adain=0.5, adain_ref="clip1")                             # v2v: no latent relay
    v2v = json.loads((runner.run_dir(new(env)) / "manifest.json").read_text())
    assert (v2v["adain_ref"], v2v["adain"]) == ("look", 0.5)          # without the relay: the old reference
    with pytest.raises(runner.RunError, match="adain ref"):
        new(env, adain_ref="clip9")


def test_a_character_brings_its_own_mannequin_height_to_the_run(env):
    """2026-10-23 ("swap the model and apply the motion from the video"): the template keeps its mannequin and motion;
    the run scales the mannequin in its own copy to the character's mannequin.scale."""
    from cinestudio import character as chars
    calls = []

    def fake_job(script, blend, args, log=None):
        calls.append((script, list(map(str, args))))
        out = Path(args[args.index("--out") + 1])
        out.write_bytes(Path(blend).read_bytes() + b"+" + script.encode())
        if log:
            Path(log).write_text('DUPLICATED {"made": 0, "rigs": [["MannequinRig", [0, 0, 0], 0.75]]}\n')
    env["mp"].setattr(runner.blender, "run_job", fake_job)
    env["mp"].setattr(chars, "load", lambda name: {"name": name, "status": "ready", "mannequin": {"scale": 0.75}})
    m = runner.new_run(env["blend"], env["sheet"], "a fox", "it dances", character_name="kitsune", proxy="off")
    assert [c[0] for c in calls] == ["duplicate_mannequin.py"]
    args = calls[0][1]
    assert args[args.index("--count") + 1] == "1" and json.loads(args[args.index("--bodies") + 1]) == [{"scale": 0.75}]
    assert m["mannequins"]["bodies"] == [{"scale": 0.75}] and m["mannequins"]["made"] == 0
    assert (runner.run_dir(m["id"]) / "inputs" / "shot.blend").read_bytes().endswith(b"+duplicate_mannequin.py")
    # a character at the template's own height: no Blender job, the shot as it is
    calls.clear()
    env["mp"].setattr(chars, "load", lambda name: {"name": name, "status": "ready"})
    m = runner.new_run(env["blend"], env["sheet"], "a fox", "it dances", character_name="kitsune", proxy="off")
    assert calls == [] and m["mannequins"] is None


def test_the_mannequin_scale_is_checked(env, tmp_path):
    from cinestudio import character as chars
    env["mp"].setattr(chars, "load", lambda name: {"name": name, "mannequin": {"scale": 5}})
    with pytest.raises(chars.CharacterError):
        chars.mannequin_scale("giant")
    assert chars.mannequin_scale(None) == 1.0
    env["mp"].setattr(chars, "load", lambda name: {"name": name, "mannequin": {"head": 1.4, "legs": 0.75, "arms": 1}})
    assert chars.mannequin_body("chibi") == {"head": 1.4, "legs": 0.75}       # 1.0 = unchanged: left out
    env["mp"].setattr(chars, "load", lambda name: {"name": name, "mannequin": {"tail": 2}})
    with pytest.raises(chars.CharacterError):
        chars.mannequin_body("odd")


def test_a_second_character_refuses_keyframe_looks_and_a_bare_extra_mannequin(env):
    from cinestudio import character as chars
    env["mp"].setattr(chars, "load", lambda name: {"name": name, "status": "ready"})
    with pytest.raises(runner.RunError, match="keyframe looks"):
        runner.new_run(env["blend"], env["sheet"], "a", "b", character_name="masked",
                       character2={"name": "kitsune"}, look_keyframes="auto")

    def fake_job(script, blend, args, log=None):
        Path(args[args.index("--out") + 1]).write_bytes(b"x")
        Path(log).write_text('DUPLICATED {"made": 0, "unused": ["MannequinRig3"], "rigs": []}\n')
    env["mp"].setattr(runner.blender, "run_job", fake_job)
    env["mp"].setattr(chars, "load", lambda name: {"name": name, "status": "ready", "mannequin": {"scale": 0.8}})
    with pytest.raises(runner.RunError, match="more mannequins than characters"):
        runner.new_run(env["blend"], env["sheet"], "a", "b", character_name="masked", proxy="off")


def test_a_blender_run_needs_the_characters_mannequin_but_no_proxy_kit(monkeypatch):
    """User, 2026-10-24: a character needs its own mannequin before a Blender video; the proxy kit is optional ("by
    default models shouldnt have a proxy kit, it should only be assigned thru the ai mcp option")."""
    from cinestudio import character as chars
    rec = {"name": "kit", "status": "ready"}
    monkeypatch.setattr(chars, "load", lambda name: dict(rec))
    with pytest.raises(chars.CharacterError, match="mannequin body"):
        chars.require_video_ready("kit")
    rec["mannequin"] = {}                                   # {} = the template's body, SET; no kit is needed
    assert chars.require_video_ready("kit")["name"] == "kit"
    assert chars.video_missing({"status": "sheet", "mannequin": {}}) == ["sheet"]


def test_new_run_refuses_a_character_without_a_mannequin(env):
    from cinestudio import character as chars

    def strict(name):
        raise chars.CharacterError(f"character {name!r} is not ready for a video: set its mannequin body")
    env["mp"].setattr(chars, "require_video_ready", strict)
    with pytest.raises(runner.RunError, match="not ready for a video"):
        new(env, character_name="kit")


def test_set_mannequin_and_proxy_store_empty_values_as_set(tmp_path, monkeypatch):
    from cinestudio import character as chars
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    d = chars.library() / "kit"
    d.mkdir(parents=True)
    (d / "character.json").write_text(json.dumps({"name": "kit", "status": "ready", "files": {}}))
    c = chars.set_mannequin("kit", default=True)
    assert c["mannequin"] == {} and chars.video_missing(c) == []
    c = chars.set_mannequin("kit", {"scale": 0.8, "head": 1.0})
    assert c["mannequin"] == {"scale": 0.8}
    with pytest.raises(chars.CharacterError):
        chars.set_mannequin("kit", {"tail": 2})
    with pytest.raises(chars.CharacterError):
        chars.set_proxy("kit", {"wings": {}})
    c = chars.set_proxy("kit", {})
    assert c["proxy"] == {} and chars.video_missing(c) == []


def test_a_saved_backplate_runs_as_a_plate_with_its_picture(env):
    """`--background saved:NAME` with a library BACKPLATE: the plate path with that picture (2026-10-24)."""
    from cinestudio import background
    src = env["tmp"] / "p.png"
    cv2.imwrite(str(src), np.full((560, 960, 3), 99, np.uint8))
    background.save_plate(src, "a forest", 3, scene="a forest", name="forest", camera={"horizon_from_top": 0.31})
    rid = new(env, background="saved:forest")
    m = json.loads((runner.run_dir(rid) / "manifest.json").read_text(encoding="utf-8"))
    assert m["background"] == "plate" and m["background_library"] == "forest"
    assert m["background_image"].endswith("plate.png")
    assert list((runner.run_dir(rid) / "inputs").glob("plate.*"))
    with pytest.raises(runner.RunError, match="picture already"):
        new(env, background="saved:forest", background_image=str(src))


def test_a_builtin_backplate_runs_by_its_name(env):
    """`saved:NAME` with a BUILT-IN backplate (backgrounds/library, read-only): the run copies its picture."""
    from cinestudio import background
    built = env["tmp"] / "builtin_bg"
    env["mp"].setattr(background, "BUILTIN", built)
    (built / "gym").mkdir(parents=True)
    cv2.imwrite(str(built / "gym" / "plate.png"), np.full((560, 960, 3), 77, np.uint8))
    (built / "gym" / "bg.json").write_text(json.dumps({"kind": "plate", "scene": "a gym", "source": "set"}))
    rid = new(env, background="saved:gym")
    m = json.loads((runner.run_dir(rid) / "manifest.json").read_text(encoding="utf-8"))
    assert m["background"] == "plate" and m["background_library"] == "gym"
    assert Path(m["background_image"]) == built / "gym" / "plate.png"
    assert list((runner.run_dir(rid) / "inputs").glob("plate.*"))
    assert sorted(p.name for p in (built / "gym").iterdir()) == ["bg.json", "plate.png"]


def test_sheets_share_one_scale_short_characters_are_drawn_short():
    """User, 2026-10-24: "standardize the sheets so they can properly reflect the difference in heights"."""
    from cinestudio import charsheet

    def fig(h=300, w=100):                                   # a white figure on black, `h` px tall
        im = np.zeros((h + 20, w + 20, 3), np.uint8)
        im[10:10 + h, 10:10 + w] = 255
        return charsheet.cutout(im)

    def heights(sheet):
        cols = (sheet.max(2) > 128).any(0)
        out, x = [], 0
        while x < sheet.shape[1]:
            if cols[x]:
                x1 = x
                while x1 < sheet.shape[1] and cols[x1]:
                    x1 += 1
                rows = np.where((sheet[:, x:x1].max(2) > 128).any(1))[0]
                out.append((rows.min(), rows.max()))
                x = x1
            else:
                x += 1
        return out
    face = fig(200, 150)
    full = heights(charsheet.assemble_panels(face, [fig()] * 4, panel_bg=(0, 0, 0), face_fill=0.6, fill=0.78))
    short = heights(charsheet.assemble_panels(face, [fig()] * 4, panel_bg=(0, 0, 0), face_fill=0.6, fill=0.78,
                                              body_ratio=0.72))
    hf, hs = full[1][1] - full[1][0], short[1][1] - short[1][0]
    assert abs(hs / hf - 0.72) < 0.02                         # the bodies at 72% of the template's height
    assert abs(full[1][1] - short[1][1]) <= 1                 # on the same feet line
    assert short[0][1] - short[0][0] >= full[0][1] - full[0][0]   # the face close-up is not made smaller
    wide = heights(charsheet.assemble_panels(face, [fig(300, 110)] * 4, panel_bg=(0, 0, 0), face_fill=0.6, fill=0.78))
    assert wide[1][1] - wide[1][0] == hf                      # a wide costume: the bodies keep the scale ...
    assert wide[0][1] - wide[0][0] < full[0][1] - full[0][0]  # ... and the face close-up gives way
    tall = heights(charsheet.assemble_panels(face, [fig()] * 4, panel_bg=(0, 0, 0), face_fill=0.6, fill=0.78,
                                             body_ratio=1.4))
    lim = charsheet.max_ratio(fill=0.78)                      # a taller character: at most the space above the feet
    assert abs((tall[1][1] - tall[1][0]) / hf - lim) < 0.03
    duo = heights(charsheet.assemble_groups([[fig()] * 3, [fig()] * 3], ratios=[1.0, 0.72]))
    h1, h2 = duo[0][1] - duo[0][0], duo[3][1] - duo[3][0]
    assert abs(h2 / h1 - 0.72) < 0.02 and abs(duo[0][1] - duo[3][1]) <= 1
    with pytest.raises(ValueError):
        charsheet.assemble_groups([[fig()]], ratios=[1.0, 0.5])


def test_the_height_counts_only_for_the_body_it_was_measured_for(monkeypatch):
    from cinestudio import character as chars
    calls = []
    monkeypatch.setattr(chars, "measure_height", lambda body, shot=None: calls.append(dict(body)) or
                        {"m": 1.3, "ref_m": 1.8, "ratio": 0.72, "body": dict(body)})
    c = {"mannequin": {"scale": 0.86}}
    assert chars.height_ratio(c) == 1.0                       # not measured yet
    chars.ensure_height(c)
    assert chars.height_ratio(c) == 0.72 and calls == [{"scale": 0.86}]
    chars.ensure_height(c)
    assert len(calls) == 1                                    # measured once per body
    c["mannequin"] = {"scale": 0.9}
    assert chars.height_ratio(c) == 1.0                       # stale: another body
    c2 = chars.ensure_height({"mannequin": {}})
    assert c2["height"] == {"ratio": 1.0, "body": {}} and len(calls) == 1       # the template body: no Blender
    monkeypatch.setattr(chars, "measure_height", lambda body, shot=None: (_ for _ in ()).throw(OSError("no blender")))
    c3 = chars.ensure_height({"mannequin": {"scale": 0.5}})
    assert "height" not in c3 and "no blender" in c3["height_error"] and chars.height_ratio(c3) == 1.0


def test_a_new_body_draws_the_sheet_again_and_keeps_the_character_ready(tmp_path, monkeypatch):
    from cinestudio import character as chars
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(chars, "measure_height", lambda body, shot=None:
                        {"m": 1.3, "ref_m": 1.8, "ratio": 0.72, "body": dict(body)})
    d = chars.library() / "kit"
    d.mkdir(parents=True)
    files = {}
    for k in ("front", "right", "back", "left", "face"):
        im = np.zeros((400, 200, 3), np.uint8)
        im[40:360, 60:140] = (40, 200, 90)
        cv2.imwrite(str(d / f"{k}.png"), im)
        files[k] = str(d / f"{k}.png")
    (d / "character.json").write_text(json.dumps({"name": "kit", "status": "ready", "source": "text", "files": files}))
    c = chars.build_sheet("kit")
    files["sheet"] = c["files"]["sheet"]
    chars.finish("kit")
    before = cv2.imread(files["sheet"])
    c = chars.set_mannequin("kit", {"scale": 0.86})
    assert c["status"] == "ready" and c["height"]["ratio"] == 0.72
    after = cv2.imread(c["files"]["sheet"])
    tall = lambda im: int(((im[:, 400:].max(2) > 60).any(1)).sum())
    assert tall(after) < tall(before) * 0.8                   # the bodies are drawn shorter now



def _kit_character(tmp_path, monkeypatch, ratio=0.72, styled=False):
    from cinestudio import character as chars
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(chars, "measure_height", lambda body, shot=None:
                        {"m": 1.8 * ratio, "ref_m": 1.8, "ratio": ratio, "body": dict(body)})
    d = chars.library() / "kit"
    d.mkdir(parents=True)
    files = {}
    for k in ("front", "right", "back", "left", "face"):
        im = np.zeros((400, 200, 3), np.uint8)
        im[40:360, 60:140] = (40, 200, 90)
        cv2.imwrite(str(d / f"{k}.png"), im)
        files[k] = str(d / f"{k}.png")
    (d / "character.json").write_text(json.dumps({"name": "kit", "status": "ready", "source": "text", "files": files}))
    c = chars.build_sheet("kit")
    if styled:
        cv2.imwrite(str(d / "sheet_styled.png"), np.zeros((10, 10, 3), np.uint8))
        c["files"]["sheet_styled"] = str(d / "sheet_styled.png")
        chars.save(c)
    chars.finish("kit")
    return chars


def test_a_failed_redraw_never_leaves_a_ready_character_with_the_wrong_sheet(tmp_path, monkeypatch):
    """Codex review 2026-10-24: the body change and the redraw are one locked step; the sheet is marked out of date
    BEFORE the redraw, and finish refuses until a redraw succeeds."""
    chars = _kit_character(tmp_path, monkeypatch)
    real = chars.build_sheet
    monkeypatch.setattr(chars, "build_sheet", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        chars.set_mannequin("kit", {"scale": 0.86})
    c = chars.load("kit")
    assert c["mannequin"] == {"scale": 0.86} and c["sheet_stale"] and c["status"] == "sheet"
    assert "sheet" in chars.video_missing(c)
    with pytest.raises(chars.CharacterError, match="height"):
        chars.finish("kit")
    monkeypatch.setattr(chars, "build_sheet", real)
    c = chars.rescale_sheet("kit")
    assert not c.get("sheet_stale") and c["sheet_scale"] == 0.72
    chars.finish("kit")
    assert chars.video_missing(chars.load("kit")) == []


def test_the_same_height_does_not_redraw_and_a_styled_sheet_is_kept_but_not_used(tmp_path, monkeypatch):
    chars = _kit_character(tmp_path, monkeypatch, styled=True)
    calls = []
    real = chars.build_sheet
    monkeypatch.setattr(chars, "build_sheet", lambda *a, **k: calls.append(1) or real(*a, **k))
    chars.set_mannequin("kit", default=True)                 # the template body: ratio 1.0, the sheet was drawn at 1.0
    assert calls == [] and chars.load("kit")["files"].get("sheet_styled")
    c = chars.set_mannequin("kit", {"scale": 0.86})          # a new height: redrawn
    assert calls == [1] and c["status"] == "ready"
    assert "sheet_styled" not in c["files"] and c["files"]["sheet_styled_old"].endswith("sheet_styled.png")
    assert chars.for_run("kit")[0] == c["files"]["sheet"]    # runs use the base sheet now


def test_the_sheet_height_limit_fits_above_the_feet_line():
    from cinestudio import charsheet
    lim = charsheet.max_ratio(fill=0.78)
    assert 1.13 < lim < 1.15
    face = charsheet.cutout(np.pad(np.full((200, 150, 3), 255, np.uint8), ((10, 10), (10, 10), (0, 0))))
    tall = charsheet.cutout(np.pad(np.full((300, 60, 3), 255, np.uint8), ((10, 10), (10, 10), (0, 0))))
    for r in (lim, 1.3, 2.4):                                 # no broadcast error, the figure stays on the canvas
        out = charsheet.assemble_panels(face, [tall] * 4, panel_bg=(0, 0, 0), face_fill=0.6, fill=0.78, body_ratio=r)
        assert out.shape == (896, 1536, 3)


def test_the_proxy_spec_is_checked_in_full():
    from cinestudio import character as chars
    ok = {"mask": {"eyes": False, "ears": True, "shape": "round", "marks": "kitsune", "ear_inner": [0.7, 0.2, 0.3]},
          "hair": {"length": "medium", "back": "shoulder", "color": [0.05, 0.02, 0.08]},
          "cape": {"length": "knee", "color": [0.02, 0.2, 0.17]},
          "bands": [{"bone": "forearm.L", "from": 0.1, "to": 0.9, "color": [1, 1, 1]}]}
    assert chars.validate_proxy(ok) is ok
    bad = [{"mask": True}, {"mask": {"marks": "dragon"}}, {"mask": {"eye_size": float("nan")}},
           {"hair": {"color": [1, 2, 0]}}, {"cape": {"length": "floor"}}, {"bands": [{"bone": "tail"}]},
           {"bands": [{"bone": "shin.L", "from": 0.8, "to": 0.2}]}, {"mask": {"marks": [{"x": 0.8, "z": 0.6}]}},
           {"mask": {"marks": [{"x": 0.0, "z": -0.6, "w": 0.1, "h": 0.05}]}},          # below a pointed mask's chin
           {"mask": {"top": 0.35, "marks": [{"x": 0.0, "z": 0.5, "w": 0.1, "h": 0.1}]}},   # above the cut top
           {"suit": {"color": "red"}}, {"wings": {}}]
    for spec in bad:
        with pytest.raises(chars.CharacterError):
            chars.validate_proxy(spec)



def test_a_failed_sheet_write_is_not_a_redraw(tmp_path, monkeypatch):
    """Codex review round 3: cv2.imwrite returning False counted as a successful redraw."""
    chars = _kit_character(tmp_path, monkeypatch)
    real = chars.cv2.imwrite
    monkeypatch.setattr(chars.cv2, "imwrite", lambda path, img, *a: False if "sheet" in str(path) else real(path, img))
    with pytest.raises(chars.CharacterError, match="cannot write the sheet"):
        chars.set_mannequin("kit", {"scale": 0.86})
    c = chars.load("kit")
    assert c["sheet_stale"] and c["status"] == "sheet"
    with pytest.raises(chars.CharacterError):
        chars.for_run("kit")
    assert not list(chars.char_dir("kit").glob("sheet.*.tmp.png"))


def test_presets_get_the_placement_checks_and_mark_colours_are_rgb():
    from cinestudio import character as chars
    assert "kitsune" in chars.mark_presets()
    chars.validate_proxy({"mask": {"shape": "round", "marks": "kitsune"}})
    with pytest.raises(chars.CharacterError, match="cut top"):
        chars.validate_proxy({"mask": {"shape": "round", "top": 0.0, "marks": "kitsune"}})
    for spec in ({"mask": {"marks": [{"color": 0.5}]}}, {"jacket": {"fit": "bad"}}, {"cape": {"shade": "bad"}},
                 {"bands": [{"bone": "shin.L", "shade": "bad"}]}):
        with pytest.raises(chars.CharacterError):
            chars.validate_proxy(spec)


def test_each_submitted_graph_is_saved_for_the_workflows_view(env):
    """Run._comfy writes the exact graph it sends (after the run-time changes) to work/graphs/<job>.json, before the
    submit; a failed write does not stop the run."""
    rid = new(env)
    assert start(env, rid)["status"] == "done"
    g = runner.run_dir(rid) / "work" / "graphs"
    assert json.loads((g / "clip.json").read_text()) == env["backend"].submits[0]["graph"]
    meta = json.loads((g / "clip.meta.json").read_text())
    assert meta["run"] == rid and meta["job"] == "clip" and meta["profile"] == runner.PROFILE and meta["attempt"] == 1
    rid2 = new(env, seed=9)
    (runner.run_dir(rid2) / "work").mkdir(exist_ok=True)
    (runner.run_dir(rid2) / "work" / "graphs").write_text("not a folder")      # every write fails
    assert start(env, rid2)["status"] == "done"


def test_a_run_on_the_retired_ltx23_graphs_is_listed_but_refused_with_a_clear_message(tmp_path, monkeypatch):
    """The LTX-2.3 graphs were retired (2026-10-25): an old run stays in the list with its video, and a start, retake
    or fork says why it cannot run instead of crashing on a missing graph or model."""
    monkeypatch.setattr(runner, "runs_dir", lambda: tmp_path)
    d = tmp_path / "rold"
    (d / "work").mkdir(parents=True)
    (d / "final.mp4").write_bytes(b"x")
    state = {"status": "stopped", "steps": {"controls": {"outputs": {"clips": 2}}}}
    (d / "manifest.json").write_text(json.dumps({"id": "rold", "created": "2026-10-18T23:04:15", "quality": "preview",
                                                 "profile": "ltx23", "render_mode": None}))
    (d / "state.json").write_text(json.dumps(state))
    row = next(r for r in runner.list_runs() if r["id"] == "rold")
    assert row["status"] == "stopped" and row["final"]
    for call in (lambda: runner.check_runnable("rold"), lambda: runner.start("rold"),
                 lambda: runner.add_take("rold", 1, {"seed": 2}), lambda: runner.fork_run("rold", 2)):
        with pytest.raises(runner.RunError, match="retired LTX-2.3 graphs"):
            call()
    assert json.loads((d / "state.json").read_text()) == state             # nothing written into the old run
    with pytest.raises(runner.RunError, match="no run"):
        runner.check_runnable("missing")


def test_new_runs_use_distilled_and_refuse_full_model(env):
    from cinestudio import character as chars
    env["mp"].setattr(chars, "load", lambda name: {"description": "on black: a figure", "files": {"front": str(env["sheet"])}})
    m = runner.new_run(env["blend"], env["sheet"], "a figure", "it waves", character_name="hero")
    assert (m["profile"], m["video_model"]) == ("ltx25", "distilled")
    assert m["stage2"]["recipe"] == "distilled"
    with pytest.raises(runner.RunError, match="video model must be"):
        runner.new_run(env["blend"], env["sheet"], "a", "b", character_name="hero", video_model="full")
    with pytest.raises(runner.RunError, match="distilled video profile"):
        runner.new_run(env["blend"], env["sheet"], "a", "b", character_name="hero", profile="ltx25dev")
    assert "retired LTX-2.5 full model" in runner.retired({"profile": "ltx25dev"})


def test_a_fork_and_a_take_keep_the_video_model(env):
    """The model is not a fork field: a fork continues from the parent's clips (their latents), so it keeps it."""
    assert "video_model" not in runner.FORK_FIELDS and "profile" not in runner.FORK_FIELDS
    assert "video_model" not in runner.TAKE_FIELDS
    with pytest.raises(runner.RunError, match="a fork can change only"):
        runner.fork_run(new(env), 2, video_model="full")


def _props_template(env):
    """A library template record with a Props layer (balls + seams) and one mannequin, for the fake .blend."""
    from cinestudio import character as chars, shots
    rec = {"name": "gym", "kind": "blender", "layers": {
        "scan": {"characters": [{"collection": "Mannequin", "rig": "MannequinRig",
                                 "screen": {"head": [0.5, 0.2], "feet": [0.5, 0.8]}}],
                 "props": [{"name": "Ball_1", "parent": None}, {"name": "Ball_1_Seam", "parent": "Ball_1"},
                           {"name": "Ball_2", "parent": None}]},
        "descriptions": {"set": "a gym", "props": {"Ball_*": "matte violet dodgeballs"}}}}
    env["mp"].setattr(shots, "library_record", lambda blend: rec)
    env["mp"].setattr(chars, "load", lambda name: {"name": name, "status": "ready"})
    return rec


def test_a_runs_prop_words_are_checked_frozen_and_shown(env):
    """2026-10-25 (user: "map our own props to the object (for example the ball)"): `props` replaces the template's
    words of a prop group in the clip texts; the template's frozen words stay; the run details name the change."""
    _props_template(env)
    with pytest.raises(runner.RunError, match="no prop matches 'Hoop'"):
        runner.new_run(env["blend"], env["sheet"], "a fox", "it dances", character_name="kitsune", proxy="off",
                       props={"Hoop": "a hoop"})
    m = runner.new_run(env["blend"], env["sheet"], "a fox", "it dances", character_name="kitsune", proxy="off",
                       props={"Ball_*": " red and white  beach balls "})
    assert m["props"] == {"Ball_*": "red and white beach balls"}
    assert m["layers"]["descriptions"]["props"] == {"Ball_*": "matte violet dodgeballs"}
    assert m["slots"] == [{"index": 1, "collection": "Mannequin", "side": "centre", "character": "kitsune"}]
    assert "In the scene: red and white beach balls." in runner.video_prompt(m)
    det = runner.run_details(m["id"])
    ch = {c["name"]: c for c in det["changes"]}
    assert ch["props"]["value"] == "Balls -> red and white beach balls"
    plain = runner.new_run(env["blend"], env["sheet"], "a fox", "it dances", character_name="kitsune", proxy="off")
    assert plain["props"] is None and "matte violet dodgeballs" in runner.video_prompt(plain)


def test_a_fork_keeps_or_changes_the_prop_words(env):
    _props_template(env)
    rid = runner.new_run(env["blend"], env["sheet"], "a fox", "it dances", character_name="kitsune", proxy="off",
                         props={"Ball_*": "red beach balls"})["id"]
    d = runner.run_dir(rid)
    (d / "work").mkdir(exist_ok=True)
    for f in ("clip.mp4", "clip_2.mp4"):
        (d / "work" / f).write_bytes(b"x")
    w = str(d / "work")
    st = json.loads((d / "state.json").read_text())
    st["steps"]["controls"] = {"status": "done", "outputs": {"clips": 2}}
    st["steps"]["clip"] = {"status": "done", "parts": {"1": {"outputs": {"clip": w + "/clip.mp4"},
                                                             "relay": {"s1": "relay_1_s1.latent"}},
                                                       "2": {"outputs": {"clip": w + "/clip_2.mp4"}}}}
    (d / "state.json").write_text(json.dumps(st))
    assert "props" in runner.FORK_FIELDS
    assert runner.fork_run(rid, 2)["props"] == {"Ball_*": "red beach balls"}          # kept
    assert runner.fork_run(rid, 2, props={"Ball_*": "gold balls"})["props"] == {"Ball_*": "gold balls"}
    assert runner.fork_run(rid, 2, props={"Ball_*": ""})["props"] is None             # the template's words again
    with pytest.raises(runner.RunError, match="no prop matches"):
        runner.fork_run(rid, 2, props={"Hoop": "a hoop"})


# ---------------------------------------------------------------- one prompt per clip (2026-10-25, AGENTS.md 7.2)
def _clip_template(env, name="balls", frames=276, texts=None, context=None):
    """A user template in the library with a length and clip texts (no Blender: the record says its frames)."""
    d = runner.home() / "shots" / name
    d.mkdir(parents=True)
    (d / "shot.blend").write_bytes(b"BLENDER-fake")
    rec = {"name": name, "kind": "blender", "status": "done", "frames": frames,
           "files": {"blend": str(d / "shot.blend"), "preview": None}}
    if texts is not None:
        rec["clip_texts"] = texts
    if context is not None:
        rec["clip_context"] = context
    (d / "shot.json").write_text(json.dumps(rec), encoding="utf-8")
    return d / "shot.blend"


def test_a_run_takes_each_clips_actions_from_the_template(env):
    blend = _clip_template(env, texts={"1": "{character} dodges three balls.", "2": "Then {character} kicks a ball.",
                                       "3": "{character} catches the ball."},
                           context="{character} faces the camera the whole time")
    m = runner.new_run(blend, env["sheet"], "a hooded figure", "A dim gym at night.", clip_videos={"3": "it bows"})
    assert m["clip_join"] == "context"
    assert m["clip_sources"] == {"1": "template", "2": "template", "3": "run"}
    assert m["clip_videos"] == {"1": "The character dodges three balls.", "2": "Then the character kicks a ball.",
                                "3": "it bows"}
    assert m["clip_context"] == "The character faces the camera the whole time"
    assert m["prompt_warnings"] is None
    t1, t3 = runner.clip_video_text(m, 1), runner.clip_video_text(m, 3)
    assert t1 == "A dim gym at night. The character faces the camera the whole time. The character dodges three balls."
    assert t3.endswith("the whole time. It bows.") and "dodges" not in t3 and "kicks" not in t3
    p2 = runner.video_prompt(runner.clip_manifest(m, 2))
    assert "kicks a ball" in p2 and "dodges" not in p2 and "catches" not in p2 and "A dim gym at night." in p2
    det = runner.run_details(m["id"])                             # the rebuilt prompts are per clip too
    assert det["prompt_warnings"] == [] and det["clip_context"] == m["clip_context"]


def test_the_run_warns_when_clips_share_one_action_text(env):
    blend = _clip_template(env)                                   # 3 clips, no clip texts
    m = runner.new_run(blend, env["sheet"], "c", "it dodges, kicks and catches")
    assert m["clip_sources"] == {"1": "none", "2": "none", "3": "none"}
    assert m["prompt_warnings"] and "all 3 clips share one action text" in m["prompt_warnings"][0]
    m2 = runner.new_run(blend, env["sheet"], "c", "a gym", clip_videos={"1": "it waves", "2": "it bows"})
    assert m2["prompt_warnings"] == ["clip 3 has no action text of its own: only the shared text"]
    assert runner.clip_video_text(m2, 3) == "a gym"                # no actions: the run text as it was written


def test_the_run_warns_about_single_clip_actions_in_the_shared_text(env):
    blend = _clip_template(env, texts={"1": "{character} dances and dodges.", "2": "{character} kicks a ball away.",
                                       "3": "{character} catches the ball and holds it."})
    m = runner.new_run(blend, env["sheet"], "c", "A dancer in a gym kicks two balls and catches one.")
    assert len(m["prompt_warnings"]) == 1 and "(kicks, catches)" in m["prompt_warnings"][0]
    ok = runner.new_run(blend, env["sheet"], "c", "A masked dancer in a dim gym, lit by violet light.")
    assert ok["prompt_warnings"] is None


def test_template_clip_texts_need_the_templates_clip_split(env):
    blend = _clip_template(env, texts={"1": "a", "2": "b", "3": "c"})
    m = runner.new_run(blend, env["sheet"], "c", "a gym", clip_frames=193)     # 2 clips of 193 frames
    assert "cuts its clips differently" in m["prompt_warnings"][0]
    assert m["clip_sources"] == {"1": "none", "2": "none"}
    m2 = runner.new_run(blend, env["sheet"], "c", "a gym", clip_videos={"5": "x"}, clip_context="")
    assert any("the text for clip 5 is not used" in w for w in m2["prompt_warnings"])
    assert m2["clip_context"] == "" and m2["clip_sources"]["1"] == "template"


def test_a_single_clip_run_and_an_unknown_length_keep_the_old_text(env):
    m = new(env)                                                   # a .blend outside the library: length unknown
    assert m and runner.clip_video_text(runner._read_json(runner.run_dir(m) / "manifest.json"), 1) == "it waves"
    old = {"prompt": {"video": "the whole shot"}, "clip_videos": {"2": "it bows"}}   # an old run: the clip's text
    assert runner.clip_video_text(old, 2) == "it bows" and runner.clip_video_text(old, 1) == "the whole shot"


def test_fill_character_capitalises_sentence_starts():
    t = "{character} spins. Then {character} bows. A ball hits {character}."
    assert runner.fill_character(t) == "The character spins. Then the character bows. A ball hits the character."
    assert runner.fill_character(t, "the masked dancer").startswith("The masked dancer spins. Then the masked dancer")
    assert runner.fill_character("no slot", "x") == "no slot"


def test_the_clips_are_made_with_their_own_text(env):
    calls = {}
    _two_clip_controls(env, calls)
    rid = new(env, clip_videos={"2": "it bows"})
    st = start(env, rid)
    assert st["status"] == "done"
    parts = runner.status(rid)["steps"]["clip"]
    p1, p2 = parts["prompt"], parts["parts"]["2"]["prompt"]
    assert "It waves. It bows." in p2 and "bows" not in p1


def test_a_fork_keeps_and_changes_the_clip_texts(env):
    blend = _clip_template(env, frames=200, texts={"1": "{character} waves.", "2": "{character} bows."})
    m = runner.new_run(blend, env["sheet"], "c", "a gym")
    rid = m["id"]
    d = runner.run_dir(rid)

    def two_done(st, d):
        (d / "work").mkdir(exist_ok=True)
        for f in ("clip.mp4", "clip_2.mp4"):
            (d / "work" / f).write_bytes(b"x")
        st["status"] = "done"
        st["steps"]["controls"] = {"status": "done", "outputs": {"clips": 2}}
        st["steps"]["clip"] = {"status": "done", "parts": {"1": {"outputs": {"clip": str(d / "work" / "clip.mp4")}},
                                                           "2": {"outputs": {"clip": str(d / "work" / "clip_2.mp4")}}}}
    set_state(rid, two_done)
    f = runner.fork_run(rid, 2, video="a dark gym")
    assert f["clip_videos"] == m["clip_videos"] and f["clip_join"] == "context"
    assert runner.clip_video_text(f, 2) == "A dark gym. The character bows."
    f2 = runner.fork_run(rid, 2, clip_videos={"2": "{character} jumps"})
    assert f2["clip_videos"]["2"] == "The character jumps" and f2["clip_sources"]["2"] == "run"
    assert f2["clip_videos"]["1"] == "The character waves."
    with pytest.raises(runner.RunError, match="kept by the fork"):
        runner.fork_run(rid, 2, clip_videos={"1": "x"})
    f3 = runner.fork_run(rid, 2, clip_videos={"2": ""})
    assert "2" not in f3["clip_videos"] and f3["prompt_warnings"] and "clip 2 has no action text" in f3["prompt_warnings"][0]
    assert d.is_dir()


# ---------------------------------------------------------------- Codex review 2026-10-25 (round 1)
def test_the_runs_own_shot_length_beats_an_older_template_record(env):
    """P1 (dance loops): the template record said 208 frames (its seconds without the loop tail), the run's own
    frozen shot has 233: the run must cut 3 clips and give clip 3 its template text."""
    blend = _clip_template(env, frames=208, texts={"1": "a.", "2": "b.", "3": "c."})

    def snap(src, dst, report, allow_external=False):
        dst.write_bytes(src.read_bytes())
        r = {"ok": True, "packed": [], "missing": [], "unsupported": [], "frame_start": 1, "frame_end": 233}
        report.write_text(json.dumps(r))
        return r
    env["mp"].setattr(runner, "snapshot_blend", snap)
    m = runner.new_run(blend, env["sheet"], "c", "a gym")
    assert m["clip_sources"] == {"1": "template", "2": "template", "3": "template"} and m["clip_videos"]["3"] == "c."


def test_the_template_text_beats_the_auto_text(env, monkeypatch):
    """P2: with clip_text auto the rig's texts were merged in before the template's, so auto won; the order is the
    run's own text, then the template's, then auto."""
    from cinestudio import timeline
    monkeypatch.setattr(timeline, "blend_timeline", lambda blend, out, rig: {"frames": []})
    monkeypatch.setattr(timeline, "run_clip_texts", lambda raw, place, frames, overlap:
                        {"1": "auto one.", "2": "auto two.", "3": "auto three."})
    monkeypatch.setattr(timeline, "events", lambda raw: [])
    blend = _clip_template(env, texts={"1": "{character} waves."})
    m = runner.new_run(blend, env["sheet"], "c", "a gym", clip_text="auto", clip_videos={"2": "it bows"})
    assert m["clip_videos"] == {"1": "The character waves.", "2": "it bows", "3": "auto three."}
    assert m["clip_sources"] == {"1": "template", "2": "run", "3": "auto"}


def test_clip_zero_is_refused_like_the_api_and_mcp(env):
    with pytest.raises(runner.RunError, match="clip numbers"):
        new(env, clip_videos={"0": "x"})


def test_fill_character_keeps_a_names_case():
    assert runner.fill_character("Then {character} bows to {character}.", "Alice") == "Then Alice bows to Alice."
    assert runner.fill_character("{character} waves.", "alice") == "Alice waves."
    assert runner.fill_character("Then {character} bows.", "The masked dancer") == "Then the masked dancer bows."
    assert runner.fill_character("Then {character} bows.", "A fox") == "Then a fox bows."
    assert runner.fill_character("Then {character} bows.", "UFO pilot") == "Then UFO pilot bows."


def test_the_action_word_heuristic_knows_nouns_and_plural_subjects():
    acts = {"1": "{character} catches a ball.", "2": "{character} bows."}
    assert runner.clip_actions_in_text("A gym whose safety catches glint on the doors.", acts, 2) == []
    assert runner.clip_actions_in_text("A dancer catches a ball in a gym.", acts, 2) == ["catches"]
    both = {"1": "They catch a ball.", "2": "They bow."}
    assert runner.clip_actions_in_text("Two dancers catch balls in a gym.", both, 2) == ["catch"]
    assert runner.clip_actions_in_text("A dancer catching balls.", acts, 2) == ["catching"]


def _context_two_clip_run(env, words=None):
    """A finished two-clip layout run (latent relay, one prompt per clip) without a GPU: the records only."""
    blend = _clip_template(env, frames=200, texts={"1": "{character} waves.", "2": "{character} kicks."})
    m = runner.new_run(blend, env["sheet"], "c", "a gym", **({"character_words": words} if words else {}))
    rid = m["id"]
    d = runner.run_dir(rid)
    mf = json.loads((d / "manifest.json").read_text())
    mf.update(render_mode="layout", relay="latent")
    (d / "manifest.json").write_text(json.dumps(mf))

    def two_done(st, d):
        (d / "work").mkdir(exist_ok=True)
        for f in ("clip.mp4", "clip_2.mp4"):
            (d / "work" / f).write_bytes(b"x")
        st["status"] = "done"
        st["steps"]["controls"] = {"status": "done", "outputs": {"clips": 2}}
        st["steps"]["clip"] = {"status": "done", "prompt": "ORIGINAL ONE", "negative": "n",
                               "parts": {"1": {"outputs": {"clip": str(d / "work" / "clip.mp4")},
                                               "relay": {"s1": "r1_s1.latent"}},
                                         "2": {"outputs": {"clip": str(d / "work" / "clip_2.mp4")},
                                               "relay": {"s1": "r2_s1.latent"}, "prompt": "ORIGINAL TWO"}}}
    set_state(rid, two_done)
    return rid


def test_a_retake_fills_the_character_slot(env):
    """P2: a retake's text went into the clip verbatim ("{character}" reached the model)."""
    rid = _context_two_clip_run(env, words="Alice")
    runner.add_take(rid, 2, {"video": "{character} bows low."})
    t = runner.list_takes(rid)["takes"]["2"][0]
    assert t["overrides"]["video"] == "Alice bows low."
    # a take queued before the fill (an unfilled override in the state) is filled when it is made
    set_state(rid, lambda st, d: st["takes"]["2"][0].update(overrides={"video": "Then {character} spins."}))
    seen = {}

    def fake_job(self, t, k, m, parts, dst, job, prefix, end_anchor=None):
        seen["text"] = runner.clip_video_text(m, k)
        Path(dst).write_bytes(b"take")
        return {"relay": {"s1": "t_s1.latent"}}
    env["mp"].setattr(runner.Run, "_layout_clip_job", fake_job)
    run = runner.Run(rid)
    run.load()
    take = run.st["takes"]["2"][0]
    run._make_take(2, take)
    assert "{character}" not in seen["text"] and seen["text"].endswith("Then Alice spins.")


def _done_take(rid, clip, n, prompt, video):
    def add(st, d):
        f = d / "takes" / f"clip{clip}" / f"take{n}" / "clip.mp4"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"take")
        st.setdefault("takes", {}).setdefault(str(clip), []).append(
            {"n": n, "status": "done", "overrides": {"video": video}, "prompt": prompt, "negative": "n",
             "outputs": {"clip": str(f), "relay": {"s1": f"t{clip}_{n}_s1.latent"}}})
    set_state(rid, add)


def test_the_details_and_forks_follow_the_take_in_use(env):
    """Verified (round 1, item 7): after use_take the details showed clip 1's ORIGINAL prompt as the exact text sent
    and every clip's original actions; a fork regenerated the original actions."""
    rid = _context_two_clip_run(env)
    _done_take(rid, 1, 1, "TAKE ONE PROMPT", "{character} jumps.")
    _done_take(rid, 2, 1, "TAKE TWO PROMPT", "{character} spins.")
    runner.use_take(rid, 1, 1)
    runner.use_take(rid, 2, 1)
    det = runner.run_details(rid)
    c1, c2 = det["clips"]
    assert c1["prompt"] == "TAKE ONE PROMPT" and c1["prompt_source"] == "recorded"
    assert (c1["actions"], c1["actions_source"]) == ("The character jumps.", "take")
    assert (c2["prompt"], c2["actions"], c2["actions_source"]) == ("TAKE TWO PROMPT", "The character spins.", "take")
    t0 = runner.list_takes(rid)["takes"]["1"][0]                     # the original kept as take 0 has its prompt
    assert t0["original"] and runner._read_json(runner.run_dir(rid) / "state.json")["takes"]["1"][0]["prompt"] \
        == "ORIGINAL ONE"
    f = runner.fork_run(rid, 2, video="a dark gym")
    assert f["clip_videos"] == {"1": "The character jumps.", "2": "The character spins."}
    assert f["clip_sources"] == {"1": "take", "2": "take"}
    assert runner.clip_video_text(f, 2) == "A dark gym. The character spins."
    runner.use_take(rid, 2, 0)                                       # the original back: its own actions again
    assert runner.run_details(rid)["clips"][1]["actions"] == "The character kicks."


def test_the_apps_fork_forwards_the_clip_texts(env):
    """P2: the app's "Build on this video" dropped per-clip actions (only the CLI and MCP forwarded clip_videos)."""
    from cinestudio import cli
    from cinestudio.webapi import runs as runs_api
    rid = _context_two_clip_run(env)
    env["mp"].setattr(cli, "_start_detached", lambda run_id: {"ok": True})
    r = runs_api._run_fork(rid, {"from_clip": 2, "video": "a gym", "clip_videos": {"2": "{character} jumps"}})
    f = runner._read_json(runner.run_dir(r["id"]) / "manifest.json")
    assert f["clip_videos"]["2"] == "The character jumps" and f["clip_sources"]["2"] == "run"
    r = runs_api._run_fork(rid, {"from_clip": 2, "video": "a gym", "clip_videos": {"2": ""}})
    assert "2" not in runner._read_json(runner.run_dir(r["id"]) / "manifest.json")["clip_videos"]
    for bad in ({"0": "x"}, {"two": "x"}, ["x"], {"2": 5}):
        with pytest.raises(ValueError):
            runs_api._run_fork(rid, {"from_clip": 2, "video": "a gym", "clip_videos": bad})
    with pytest.raises(runner.RunError, match="kept by the fork"):
        runs_api._run_fork(rid, {"from_clip": 2, "video": "a gym", "clip_videos": {"1": "x"}})


def test_new_runs_hold_the_chained_light_to_clip_1():
    """2026-10-27: chain_light held each chained frame to the run's LOOK image, which the video never reaches (clip 1
    came out ~8 L darker); every join was pushed up by the full clamp and the scene got brighter. New runs hold it to
    clip 1's own first frame; old manifests (no key) keep the look."""
    assert runner.DEFAULTS["chain_light_ref"] == "clip1"


def test_new_runs_grade_the_clip_joins():
    """2026-10-27: each LTX clip came out 0.3-0.6 L lighter than the still it was given (the steps sit at the joins);
    new runs grade every clip after the first to the clip before (chain_grade "match"); old manifests keep "off"."""
    assert runner.DEFAULTS["chain_grade"] == "local"
    assert "chain_grade" in runner.FORK_FIELDS


def test_the_local_gain_map_undoes_a_regional_change():
    """video.gain_map: a background region 1.5x brighter and a red cast elsewhere are found per channel, the
    character (outside the background mask) takes the global ratio, and the map stays smooth and limited."""
    import numpy as np
    from cinestudio import video
    ref = np.full((90, 160, 3), 60, np.uint8)
    this = ref.copy()
    this[:, :60] = 90                                    # the left part: 1.5x brighter in every channel
    this[:, 100:, 0] = 80                                # the right part: a red cast
    bg = np.ones((90, 160), bool)
    bg[30:60, 70:90] = False                             # the character
    m = video.gain_map([(ref, this, bg)], sigma=4)
    assert m.shape == (90, 160, 3)
    assert abs(float(m[45, 20].mean()) - 60 / 90) < 0.03
    assert abs(float(m[45, 140, 0]) - 60 / 80) < 0.03 and abs(float(m[45, 140, 1]) - 1.0) < 0.03
    assert 0.5 <= float(m.min()) and float(m.max()) <= 2.0


def test_the_gain_map_limits_a_colour_shift_per_pixel():
    """video.gain_map (2026-10-28, the yellow bandages): a strong blue cut of a violet floor is kept within +-tint of
    the pixel's mean gain, so a white part over that floor keeps most of its blue; tint None keeps the old map."""
    import numpy as np
    from cinestudio import video
    ref = np.full((60, 100, 3), 60, np.uint8)
    this = ref.copy()
    this[..., 2] = 105                                   # the floor came out much bluer (blue ratio 0.57)
    m = video.gain_map([(ref, this, None)], sigma=4)
    free = video.gain_map([(ref, this, None)], sigma=4, tint=None)
    mean = free.mean(axis=-1)                            # the limit is taken from the free map's mean gain
    assert float((m[..., 2] / mean).min()) >= 0.8 - 1e-3
    assert float((m[..., 0] / mean).max()) <= 1.2 + 1e-3
    assert abs(float(free[30, 50, 2]) - 60 / 105) < 0.03


def test_the_gain_map_writer_gives_the_character_its_own_gain(tmp_path):
    """video.apply_gain_map with fig_masks + fig_gain: inside the frame's mask the character's gain, outside the map."""
    import cv2
    import numpy as np
    from cinestudio import video
    w, h = 64, 48
    src = tmp_path / "src.mp4"
    frames = tmp_path / "f"
    frames.mkdir()
    for i in range(3):
        cv2.imwrite(str(frames / f"f_{i:04d}.png"), np.full((h, w, 3), 100, np.uint8))
    video._run(["-framerate", "24", "-i", str(frames / "f_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-color_range", "tv", "-colorspace", "bt709", str(src)])
    m = np.zeros((h, w), np.uint8)
    m[10:38, 20:44] = 255
    cv2.imwrite(str(tmp_path / "m.png"), m)
    dst = tmp_path / "dst.mp4"
    n = video.apply_gain_map(src, dst, np.ones((h, w, 3), np.float32), w, h, 24.0,
                             fig_masks=[str(tmp_path / "m.png")] * 3, fig_gain=[1.5, 1.5, 1.5], feather=1.0)
    assert n == 3
    fr = next(video.read_rgb_frames(dst, w, h)).astype(float)
    assert fr[24, 32].mean() > 135 and abs(fr[2, 2].mean() - 100) < 4
