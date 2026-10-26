"""Templates from a dance video (cinestudio/video_template.py + cinestudio/mocap.py): the GLB reader, the motion
conversion, the record checks, the worker with a fake SAM 3D Body job and a fake Blender build, cancel, the queue kind
and the web endpoint. No GPU, no ComfyUI, no Blender."""
import asyncio
import json
import math
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cinestudio import mocap, shots, video, video_template as vt, workqueue

POSE = {"Nose": (0, 1.60, 0.08), "Neck": (0, 1.45, 0), "RShoulder": (-0.18, 1.42, 0), "RElbow": (-0.22, 1.15, 0),
        "RWrist": (-0.24, 0.90, 0.02), "LShoulder": (0.18, 1.42, 0), "LElbow": (0.22, 1.15, 0),
        "LWrist": (0.24, 0.90, 0.02), "RHip": (-0.09, 0.92, 0), "RKnee": (-0.10, 0.50, 0.02),
        "RAnkle": (-0.10, 0.09, 0), "LHip": (0.09, 0.92, 0), "LKnee": (0.10, 0.50, 0.02), "LAnkle": (0.10, 0.09, 0),
        "REye": (-0.03, 1.62, 0.07), "LEye": (0.03, 1.62, 0.07), "REar": (-0.07, 1.60, 0), "LEar": (0.07, 1.60, 0)}


def _person(frames: int, lift: bool, pitch_deg: float = 0.0, steps: bool = False) -> tuple[dict, np.ndarray]:
    """GLB-axis joint tracks of a standing person (y up, facing +z); `lift` = the right knee comes up toward the
    camera in the middle third and the person sways sideways. `pitch_deg` tilts everything (a camera looking down)."""
    joints = {n: np.tile(np.array(p, float), (frames, 1)) for n, p in POSE.items()}
    root = np.zeros((frames, 3))
    if lift:
        root[:, 0] = 0.05 * np.sin(np.linspace(0, 4 * math.pi, frames))
        for t in range(frames // 3, 2 * frames // 3):
            joints["RKnee"][t] = (-0.10, 0.80, 0.35)                   # thigh forward (toward the viewer: +z)
            joints["RAnkle"][t] = (-0.10, 0.45, 0.30)
    if steps:                                                          # the feet step forward and back (and apart)
        for t in range(frames):
            joints["RAnkle"][t] += (-0.1 * math.cos(t / 3), 0, 0.3 * math.sin(t / 5))
            joints["LAnkle"][t] += (0.1 * math.cos(t / 4), 0, -0.3 * math.sin(t / 7))
    for s in ("R", "L"):                                               # 21 hand points below each wrist
        for k, n in enumerate(mocap.HAND[:21] if s == "R" else mocap.HAND[21:]):
            joints[n] = joints[f"{s}Wrist"] + np.array([0, -0.01 * min(k, 9), 0.005])
    if pitch_deg:
        a = math.radians(pitch_deg)
        R = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        joints = {n: v @ R.T for n, v in joints.items()}
        root = root @ R.T
    return joints, root


def write_glb(path: Path, people: list, fps: float = 24.0) -> Path:
    """A minimal animated GLB in BuildPoseFile's layout: trackNN nodes with openpose_<joint> children."""
    frames = len(people[0][1])
    nodes, chans, samplers, accessors, views, blob = [], [], [], [], [], bytearray()

    def add(arr: np.ndarray, typ: str) -> int:
        data = arr.astype(np.float32).tobytes()
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        blob.extend(data)
        accessors.append({"bufferView": len(views) - 1, "componentType": 5126, "count": len(arr), "type": typ})
        return len(accessors) - 1

    t_acc = add(np.arange(frames) / fps, "SCALAR")
    for i, (joints, root) in enumerate(people):
        ti = len(nodes)
        nodes.append({"name": f"track{i:02d}", "children": []})
        samplers.append({"input": t_acc, "output": add(root, "VEC3"), "interpolation": "LINEAR"})
        chans.append({"sampler": len(samplers) - 1, "target": {"node": ti, "path": "translation"}})
        for n, v in joints.items():
            nodes[ti]["children"].append(len(nodes))
            nodes.append({"name": f"openpose_{n}"})
            samplers.append({"input": t_acc, "output": add(v, "VEC3"), "interpolation": "LINEAR"})
            chans.append({"sampler": len(samplers) - 1, "target": {"node": len(nodes) - 1, "path": "translation"}})
    while len(blob) % 4:
        blob.append(0)
    j = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [k for k, nd in enumerate(nodes)
                                                                       if nd["name"].startswith("track")]}],
         "nodes": nodes, "animations": [{"name": "all_tracks", "channels": chans, "samplers": samplers}],
         "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    js = json.dumps(j).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
    return path


def tiny_video(dst: Path, seconds: float = 3.0, sound: bool = True) -> Path:
    cmd = [video.ffmpeg(), "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=gray:s=64x96:r=30:d={seconds}"]
    if sound:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac", "-shortest"]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)], check=True, stdin=subprocess.DEVNULL)
    return dst


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    return tmp_path


# ---------------------------------------------------------------- mocap
def test_motion_takes_the_busiest_person_and_reads_its_bones(tmp_path):
    glb = write_glb(tmp_path / "p.glb", [_person(72, lift=False), _person(72, lift=True)])
    m = mocap.motion_from_glb(glb, tmp_path / "m.json", fps=24.0)
    assert m["tracks"] == 2 and m["track"] == 1 and len(m["frames"]) == 72
    assert json.loads((tmp_path / "m.json").read_text())["detector"] == "sam3d_body"
    f0, mid = m["frames"][0]["dirs"], m["frames"][36]["dirs"]
    assert f0["spine"][2] > 0.99                                        # upright
    assert f0["shoulder.R"][0] < -0.9 and f0["shoulder.L"][0] > 0.9      # the person's right = the rig's .R = -X
    assert f0["thigh.R"][2] < -0.99                                     # standing leg points down
    assert mid["thigh.R"][1] < -0.5                                     # the lifted thigh points at the camera (-Y)
    for d in (f0, mid):
        assert all(abs(np.linalg.norm(v) - 1) < 1e-3 for v in d.values())
    assert max(abs(f["hips"][0]) for f in m["frames"]) > 0.03           # the sideways sway is kept
    assert all(f["contact"][".L"] for f in m["frames"][3:-3])            # the still left foot stands


def test_floor_tilt_is_removed(tmp_path):
    glb = write_glb(tmp_path / "t.glb", [_person(48, lift=False, pitch_deg=12.0, steps=True)])
    m = mocap.motion_from_glb(glb, fps=24.0)
    assert abs(m["scale"]["floor_tilt_deg"] - 12.0) < 1.0
    assert m["frames"][10]["dirs"]["spine"][2] > 0.995


def test_window_and_errors(tmp_path):
    glb = write_glb(tmp_path / "w.glb", [_person(48, lift=False)])
    assert len(mocap.motion_from_glb(glb, start=0.5, seconds=1.0)["frames"]) == 24
    with pytest.raises(mocap.MocapError, match="no frames"):
        mocap.motion_from_glb(glb, start=5.0, seconds=2.0)
    with pytest.raises(mocap.MocapError, match="positive"):
        mocap.motion_from_glb(glb, fps=0)
    assert len(mocap.motion_from_glb(glb, smooth=4, legs_smooth=8)["frames"]) == 48   # even windows are made odd
    with pytest.raises(mocap.MocapError, match="track 3"):
        mocap.motion_from_glb(glb, track=3)
    (tmp_path / "x.glb").write_bytes(b"nope")
    with pytest.raises(mocap.MocapError, match="not a GLB"):
        mocap.read_glb(tmp_path / "x.glb")


# ---------------------------------------------------------------- the record and the worker
def test_prepare_checks(home):
    clip = tiny_video(home / "dance.mp4", 3.0)
    (home / "notes.txt").write_text("x")
    with pytest.raises(vt.TemplateError, match="video file"):
        vt.prepare("a", home / "notes.txt")
    with pytest.raises(vt.TemplateError, match="sound"):
        vt.prepare("a", clip, sound="song")
    with pytest.raises(vt.TemplateError, match="after the end"):
        vt.prepare("a", clip, start=2.5)
    r = vt.prepare("a", clip, start=0.5)
    assert r["kind"] == "blender" and r["source"] == "video" and r["private_sound"] is True
    assert r["params"]["seconds"] == pytest.approx(2.5, abs=0.05)        # clipped to the video's end
    assert (shots.shot_dir("a") / "source.mp4").is_file() and clip.is_file()   # copied, the original kept
    with pytest.raises(vt.TemplateError, match="exists|being made"):
        vt.prepare("a", clip)


def _fake_sam(people):
    def sam(cut, glb, tag, emit, canceled):
        assert video.probe(cut).fps == pytest.approx(24.0, abs=0.1)
        write_glb(glb, people)
    return sam


def _fake_build(calls):
    def build(motion, out, audio, log):
        calls.append({"motion": json.loads(Path(motion).read_text()), "audio": audio})
        Path(out).write_bytes(b"BLENDER-fake")
    return build


def test_make_builds_the_template(home, monkeypatch):
    clip = tiny_video(home / "dance.mp4", 3.0, sound=True)
    vt.prepare("folk", clip, seconds=2.0)
    calls = []
    monkeypatch.setattr(shots, "make_preview", lambda d, blend, log: {
        "preview": str(d / "preview.mp4"), "poster": str(d / "poster.jpg"), "frames": 48})
    r = vt.make("folk", sam_job=_fake_sam([_person(48, lift=True)]), build=_fake_build(calls))
    d = shots.shot_dir("folk")
    assert r["status"] == "done" and r["frames"] == 48 and r["files"]["blend"] == str(d / "shot.blend")
    assert len(calls[0]["motion"]["frames"]) == 48 and calls[0]["audio"] == d / "sound_source.wav"
    assert r["private_sound"] is True and "music" in r and r["motion"]["tracks"] == 1
    assert (d / "motion.json").is_file() and not (d / "work").exists() and "progress" not in r
    assert workqueue.find("template", "folk") is None


def test_make_without_sound(home, monkeypatch):
    clip = tiny_video(home / "quiet.mp4", 3.0, sound=False)
    vt.prepare("quiet", clip, seconds=2.0)
    calls = []
    monkeypatch.setattr(shots, "make_preview", lambda d, blend, log: {"frames": 48})
    r = vt.make("quiet", sam_job=_fake_sam([_person(48, lift=False)]), build=_fake_build(calls))
    assert calls[0]["audio"] is None and r["private_sound"] is False


def test_cancel_before_the_job(home):
    clip = tiny_video(home / "d.mp4", 3.0, sound=False)
    vt.prepare("stop", clip)
    vt.request_cancel("stop")
    seen = []
    with pytest.raises(workqueue.QueueCanceled):
        vt.make("stop", sam_job=lambda *a: seen.append(a), build=lambda *a: seen.append(a))
    assert not seen and shots.load("stop")["status"] == "canceled"


def test_start_reserves_the_queue_place(home):
    clip = tiny_video(home / "d.mp4", 3.0, sound=False)
    seen = []
    r = vt.start("one", clip, spawner=lambda cmd, log, cwd: (seen.append(cmd) or (4321, True)))
    assert seen[0][-2:] == ["work-mocap", "one"] and r["pid"] == 4321
    item = [i for i in workqueue.view()["items"] if i["kind"] == "template"][0]
    assert item["ref"] == "one" and item["status"] == "queued"
    assert workqueue.cancel("template", "one")["cancel_requested"] is True
    workqueue.done("template", "one")

    def boom(cmd, log, cwd):
        raise OSError("no python")
    with pytest.raises(OSError):
        vt.start("two", clip, spawner=boom)
    assert workqueue.find("template", "two") is None and shots.load("two")["status"] == "failed"


def test_web_endpoint_starts_a_template_from_a_video(home, monkeypatch):
    from aiohttp import FormData
    from aiohttp.test_utils import TestClient, TestServer
    from cinestudio import webapp
    clip = tiny_video(home / "d.mp4", 3.0, sound=False)
    calls = []
    monkeypatch.setattr(vt, "start", lambda *a, **k: (calls.append(a) or {"name": a[0], "pid": 1, "status": "queued"}))

    async def go():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as c:
            fd = FormData()
            fd.add_field("name", "folk")
            fd.add_field("start", "1.5")
            fd.add_field("seconds", "8")
            fd.add_field("sound", "none")
            fd.add_field("file", clip.read_bytes(), filename="dance.mp4", content_type="video/mp4")
            r = await c.post("/api/templates/from-video", data=fd)
            assert r.status == 200 and (await r.json())["data"]["name"] == "folk"
            fd = FormData()
            fd.add_field("name", "bad")
            fd.add_field("file", b"x", filename="notes.txt", content_type="text/plain")
            r = await c.post("/api/templates/from-video", data=fd)
            assert r.status >= 400
    asyncio.run(go())
    name, upload, start, seconds, sound = calls[0][:5]
    assert (name, start, seconds, sound) == ("folk", 1.5, 8.0, "none") and Path(upload).suffix == ".mp4"


# ---------------------------------------------------------------- the Codex review's cases (2026-10-22)
def _rewrite_json(path: Path, change) -> None:
    """Change the JSON chunk of a GLB (the binary chunk stays)."""
    d = path.read_bytes()
    n = struct.unpack_from("<I", d, 12)[0]
    j = json.loads(d[20:20 + n])
    change(j)
    js = json.dumps(j).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + d[20 + n:]
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)


def test_a_joint_without_a_channel_takes_its_node_translation(tmp_path):
    glb = write_glb(tmp_path / "s.glb", [_person(48, lift=True)])

    def still_neck(j):                                     # as BuildPoseFile writes a joint that never moves
        k = next(i for i, nd in enumerate(j["nodes"]) if nd["name"] == "openpose_Neck")
        j["nodes"][k]["translation"] = [0.0, 1.45, 0.0]
        an = j["animations"][0]
        an["channels"] = [c for c in an["channels"] if c["target"]["node"] != k]
    _rewrite_json(glb, still_neck)
    tr = mocap.read_glb(glb)["tracks"][0]
    assert np.allclose(tr["joints"]["Neck"], [0.0, 1.45, 0.0]) and len(tr["t"]) == 48
    assert len(mocap.motion_from_glb(glb)["frames"]) == 48


def test_each_track_keeps_its_own_times(tmp_path):
    """A person who appears 1 s late: the window starts where the person is."""
    glb = write_glb(tmp_path / "late.glb", [_person(24, lift=False)], fps=24.0)
    d = glb.read_bytes()
    n = struct.unpack_from("<I", d, 12)[0]
    off = 20 + n + 8                                       # the time accessor is the first 24 floats of the blob
    t = np.frombuffer(d, np.float32, 24, off).copy() + 1.0
    glb.write_bytes(d[:off] + t.tobytes() + d[off + 96:])
    tr = mocap.read_glb(glb)["tracks"][0]
    assert tr["t"][0] == pytest.approx(1.0)
    m = mocap.motion_from_glb(glb, start=0.0, seconds=2.0)
    assert m["start_seconds"] == pytest.approx(1.0) and len(m["frames"]) == 24


def test_floor_falls_back_to_the_body_when_the_feet_give_no_plane(tmp_path):
    glb = write_glb(tmp_path / "still.glb", [_person(48, lift=False, pitch_deg=12.0)])   # the feet never move
    m = mocap.motion_from_glb(glb)
    assert m["scale"]["floor_from"] == "body" and abs(m["scale"]["floor_tilt_deg"] - 12.0) < 1.0
    assert m["frames"][10]["dirs"]["spine"][2] > 0.999
    glb = write_glb(tmp_path / "steps.glb", [_person(48, lift=False, pitch_deg=12.0, steps=True)])
    assert mocap.motion_from_glb(glb)["scale"]["floor_from"] == "floor"


def test_pelvis_roll_has_no_jump_when_the_dancer_turns():
    def hl(yaw_deg, tilt_deg=0.0):
        a, b = math.radians(yaw_deg), math.radians(tilt_deg)
        return np.array([math.cos(a) * math.cos(b), math.sin(a) * math.cos(b), math.sin(b)]) * 0.18
    assert abs(mocap.pelvis_roll(hl(89))) < 0.5 and abs(mocap.pelvis_roll(hl(91))) < 0.5   # level: no 0 -> 180 jump
    assert mocap.pelvis_roll(hl(0, 10)) == pytest.approx(10.0, abs=0.01)                 # front: the plain tilt
    assert mocap.pelvis_roll(hl(180, 10)) == pytest.approx(-10.0, abs=0.01)              # back: the picture's tilt
    assert abs(mocap.pelvis_roll(hl(90, 10))) < 0.01                                     # side: not visible


def test_scale_factor_keeps_the_decoded_frames_in_budget():
    f = vt.scale_factor(3840, 2160, 30.0)
    assert (3840 * f) * (2160 * f) * 12 * 30 * vt.FPS <= vt.DECODE_BUDGET * 1.001
    assert vt.scale_factor(360, 640, 10.0) == 1.0                  # small videos stay as they are
    assert 1920 * vt.scale_factor(1920, 1080, 2.0) <= vt.LONG_SIDE_MAX + 1e-6


def test_a_cancel_during_the_build_never_publishes_done(home, monkeypatch):
    clip = tiny_video(home / "d.mp4", 3.0, sound=False)
    vt.prepare("late", clip, seconds=2.0)
    monkeypatch.setattr(shots, "make_preview", lambda d, blend, log: {"frames": 48})

    def build(motion, out, audio, log):
        Path(out).write_bytes(b"x")
        vt.request_cancel("late")                          # the user cancels while Blender builds
    with pytest.raises(workqueue.QueueCanceled):
        vt.make("late", sam_job=_fake_sam([_person(48, lift=False)]), build=build)
    assert shots.load("late")["status"] == "canceled" and workqueue.find("template", "late") is None


def test_a_retry_may_use_the_failed_templates_own_source(home):
    clip = tiny_video(home / "d.mp4", 3.0, sound=False)
    r = vt.prepare("again", clip)
    d = shots.shot_dir("again")
    shots._write(d / "shot.json", {**r, "status": "failed"})
    r2 = vt.prepare("again", d / "source.mp4")             # the input lives in the folder that is replaced
    assert r2["status"] == "queued" and (d / "source.mp4").is_file()
    assert not list(shots.shots_dir().glob(".incoming_*"))


def test_names_for_the_figure_selection_are_refused(home):
    clip = tiny_video(home / "d.mp4", 3.0, sound=False)
    with pytest.raises(shots.ShotError, match="mask-"):
        vt.prepare("mask-foo", clip)


def test_shot_show_reports_a_dead_worker(home):
    clip = tiny_video(home / "d.mp4", 3.0, sound=False)
    r = vt.prepare("dead", clip)
    shots._write(shots.shot_dir("dead") / "shot.json", {**r, "status": "running", "pid": 999999, "pid_start": 1.0})
    assert shots.show("dead")["status"] == "interrupted"


class _Lock:
    def __init__(self):
        self.held = False

    def acquire(self):
        self.held = True

    def release(self):
        self.held = False

    def holder(self):
        return None


def test_the_sam_job_keeps_the_gpu_lock_until_comfyui_confirms_the_end(home, monkeypatch, tmp_path):
    from cinestudio import song
    lock = _Lock()
    monkeypatch.setattr(vt.server, "gpu_lock", lambda note=None: lock)
    monkeypatch.setattr(vt.server, "ensure_family", lambda key: {})
    monkeypatch.setattr(vt.server, "note_owned", lambda jid: None)
    drained = []

    def drain(prefix, jid, backend=None, **k):
        drained.append((jid, lock.held))                   # still holding the lock while it drains
        return "unknown"
    monkeypatch.setattr(song, "_drain", drain)

    class Lost:
        def submit(self, *a, **k):
            return "job-1"

        def watch(self, jid, progress):
            raise ConnectionError("lost")

        def cancel(self, jid):
            pass
    with pytest.raises(vt.TemplateError, match="could not be confirmed"):
        vt._sam(tmp_path / "cut.mp4", tmp_path / "p.glb", "t", lambda e: None, lambda: False, backend_factory=Lost)
    assert drained == [("job-1", True)] and lock.held is False


def test_upload_form_refuses_a_second_file_and_input_errors_are_400(home):
    from aiohttp import FormData
    from aiohttp.test_utils import TestClient, TestServer
    from cinestudio import webapp
    clip = tiny_video(home / "d.mp4", 3.0, sound=False)

    async def post(fields, files):
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as c:
            fd = FormData()
            for k, v in fields.items():
                fd.add_field(k, v)
            for k, name in files:
                fd.add_field(k, clip.read_bytes(), filename=name, content_type="video/mp4")
            r = await c.post("/api/templates/from-video", data=fd)
            return r.status, (await r.json()).get("error") or ""
    st, err = asyncio.run(post({"name": "two"}, [("file", "a.mp4"), ("file2", "b.mp4")]))
    assert st == 400 and "one file" in err
    st, err = asyncio.run(post({"name": "badsound", "sound": "song"}, [("file", "a.mp4")]))
    assert st == 400 and "sound" in err
    st, err = asyncio.run(post({"name": "longtext", "description": "y" * 20000}, [("file", "a.mp4")]))
    assert st == 400 and "too long" in err
    assert not list((home / "home" / "uploads").glob("template_*"))


# ---------------------------------------------------------------- the Codex review, round 2 (2026-10-22)
def _stance(frames: int, lean_deg: float = 0.0) -> np.ndarray:
    """Blender-axis body points of a person (P[frames, 18, 3]); the torso leans forward by lean_deg."""
    P = mocap.to_blender(np.stack([np.tile(np.array(POSE[n], float), (frames, 1)) for n in mocap.BODY], 1))
    if lean_deg:
        a = math.radians(lean_deg)
        hip = (P[:, mocap.BODY.index("RHip")] + P[:, mocap.BODY.index("LHip")]) / 2
        for n in ("Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "REye", "LEye",
                  "REar", "LEar"):
            k = mocap.BODY.index(n)
            v = P[:, k] - hip
            P[:, k] = hip + np.stack([v[:, 0], v[:, 1] * math.cos(a) - v[:, 2] * math.sin(a),
                                      v[:, 1] * math.sin(a) + v[:, 2] * math.cos(a)], 1)
    return P


def test_floor_fit_rechecks_the_spread_after_dropping_outliers():
    P = _stance(100)
    ra, la = mocap.BODY.index("RAnkle"), mocap.BODY.index("LAnkle")
    rng = np.random.default_rng(1)
    for t in range(100):                                   # 80 level points on one line, 20 high points all round
        if t < 80:
            P[t, ra] = (-0.1 + 0.01 * t, 0.0, 0.08)
        else:
            P[t, ra] = (rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0.30)
        P[t, la] = P[t, ra] + (0.2, 0.0, 0.2)              # the other foot is higher: the right ankle is the low one
    R, how = mocap.floor_rotation(P)
    assert how != "floor" and np.degrees(np.arccos(R[2, 2])) < 1.0


def test_a_held_bow_is_not_turned_upright():
    R, how = mocap.floor_rotation(_stance(60, lean_deg=30.0))     # still feet, a 30 deg bow the whole clip
    assert how == "none" and np.allclose(R, np.eye(3))
    R, how = mocap.floor_rotation(_stance(60, lean_deg=10.0))     # a small mean lean: a camera tilt
    assert how == "body" and abs(np.degrees(np.arccos(R[2, 2])) - 10.0) < 0.5


def test_a_wholly_still_person_is_held_for_the_asked_length(tmp_path):
    glb = write_glb(tmp_path / "still.glb", [_person(48, lift=False)])

    def no_channels(j):                                    # every joint and the root constant: no channel at all
        for c in j["animations"][0]["channels"]:
            nd = j["nodes"][c["target"]["node"]]
            nd["translation"] = [0.0, 0.0, 0.0] if nd["name"].startswith("track") else \
                list(POSE.get(nd["name"][len("openpose_"):], (0.0, 0.9, 0.0)))
        j["animations"][0]["channels"] = []
    _rewrite_json(glb, no_channels)
    assert len(mocap.read_glb(glb)["tracks"][0]["t"]) == 1
    assert len(mocap.motion_from_glb(glb, seconds=2.0)["frames"]) == 48


def test_numbers_must_be_finite_and_a_zero_length_is_refused(home, tmp_path):
    clip = tiny_video(home / "d.mp4", 3.0, sound=False)
    for kw in ({"start": float("nan")}, {"seconds": float("nan")}, {"seconds": float("inf")}):
        with pytest.raises(vt.TemplateError, match="numbers"):
            vt.prepare("n", clip, **kw)
    with pytest.raises(vt.TemplateError, match="at least 2 s"):
        vt.prepare("n", clip, seconds=0)
    glb = write_glb(tmp_path / "w.glb", [_person(48, lift=False)])
    with pytest.raises(mocap.MocapError, match="positive"):
        mocap.motion_from_glb(glb, seconds=float("nan"))


def test_the_cancel_request_is_tried_again_until_comfyui_takes_it():
    calls = []

    class Flaky:
        def cancel(self, jid):
            calls.append(jid)
            if len(calls) < 3:
                raise ConnectionError("no answer")
    import time as _t
    with vt._cancel_watch(Flaky(), "job-9", lambda: True, every=0.01):
        for _ in range(200):
            if len(calls) >= 3:
                break
            _t.sleep(0.01)
    assert calls[:3] == ["job-9"] * 3


def test_a_failed_retry_brings_the_old_try_back(home, monkeypatch):
    clip = tiny_video(home / "d.mp4", 3.0, sound=False)
    r = vt.prepare("keep", clip)
    d = shots.shot_dir("keep")
    shots._write(d / "shot.json", {**r, "status": "failed"})
    real = shots._write

    def broken(path, data):
        if data.get("status") == "queued":
            raise OSError("disk full")
        return real(path, data)
    monkeypatch.setattr(shots, "_write", broken)
    with pytest.raises(OSError):
        vt.prepare("keep", d / "source.mp4")               # the retry fails after the old folder was moved aside
    assert (d / "source.mp4").is_file() and shots.load("keep")["status"] == "failed"   # the old try is back
    assert not list(shots.shots_dir().glob(".incoming_*")) and not list(shots.shots_dir().glob(".replaced_*"))


def test_accent_turns_a_hidden_back_flick_to_the_side_and_lifts_it():
    frames = [{"dirs": {"thigh.L": [0.0, -0.65, -0.76], "shin.L": [0.21, 0.94, -0.28],
                        "thigh.R": [0.0, 0.0, -1.0], "shin.R": [0.0, 0.0, -1.0]}} for _ in range(20)]
    n = mocap.accent(frames, ".L", 5, 15, lift=1.4, thigh_lift=1.25, turn_deg=50, ease=0.4)
    assert n == 11
    mid = frames[9]["dirs"]                                            # frame 10 = the middle: full weight
    assert mid["shin.L"][0] > 0.7 and mid["shin.L"][2] > -0.1          # out to the side (+X), lifted to ~horizontal
    assert math.degrees(math.acos(-mid["thigh.L"][2])) > 45            # the thigh lifted more (was ~40 deg)
    assert frames[0]["dirs"]["shin.L"] == [0.21, 0.94, -0.28] and frames[9]["dirs"]["shin.R"] == [0.0, 0.0, -1.0]
    with pytest.raises(mocap.MocapError):
        mocap.accent(frames, ".X", 5, 15)


def test_mirror_legs_copies_a_kick_to_the_other_side():
    def fr(kick_l):
        return {"dirs": {"thigh.L": [0.3, -0.4, -0.86] if kick_l else [0.0, 0.0, -1.0], "shin.L": [0.69, 0.71, 0.14]
                         if kick_l else [0.0, 0.0, -1.0], "thigh.R": [0.0, 0.0, -1.0], "shin.R": [0.0, 0.0, -1.0],
                         "spine": [0.0, 0.0, 1.0]},
                "contact": {".L": not kick_l, ".R": True}, "kick": {".L": 1.0 if kick_l else 0.0, ".R": 0.0},
                "feet": {".L": [0.1, 0.0, 0.6 if kick_l else 0.09], ".R": [-0.1, 0.0, 0.09]}, "pelvis_roll": 8.0 if kick_l else 0.0,
                **({"kick_dir": {".L": [0.7, 0.0, -0.7]}} if kick_l else {})}
    frames = [fr(10 <= i < 20) for i in range(60)]
    assert mocap.mirror_legs(frames, 9, 22, 39, blend=2) == 14
    mid = frames[38 + 6]                                               # the copy of frame 16 (a kick frame)
    assert mid["dirs"]["shin.R"][0] == pytest.approx(-0.69, abs=0.01) and mid["dirs"]["shin.L"] == [0.0, 0.0, -1.0]
    assert mid["contact"] == {".R": False, ".L": True} and mid["kick"][".R"] == 1.0
    assert mid["kick_dir"] == {".R": [-0.7, 0.0, -0.7]} and mid["pelvis_roll"] == pytest.approx(-8.0)
    assert mid["dirs"]["spine"] == [0.0, 0.0, 1.0]                     # the torso is not copied
    with pytest.raises(mocap.MocapError, match="overlap"):
        mocap.mirror_legs(frames, 9, 22, 15)


# ---------------------------------------------------------------- the Codex review, round 3 (2026-10-23)
def test_a_failed_retry_with_move_keeps_the_source(home, monkeypatch):
    clip = tiny_video(home / "d.mp4", 3.0, sound=False)
    r = vt.prepare("mv", clip)
    d = shots.shot_dir("mv")
    shots._write(d / "shot.json", {**r, "status": "failed"})
    real = shots._write

    def broken(path, data):
        if data.get("status") == "queued":
            raise OSError("disk full")
        return real(path, data)
    monkeypatch.setattr(shots, "_write", broken)
    with pytest.raises(OSError):
        vt.prepare("mv", d / "source.mp4", move=True)     # its own source, moved: it must survive the rollback
    assert (d / "source.mp4").is_file() and shots.load("mv")["status"] == "failed"
    other = tiny_video(home / "up.mp4", 3.0, sound=False)
    with pytest.raises(OSError):
        vt.prepare("mv", other, move=True)                # an upload that is moved goes back to its place
    assert other.is_file()


def _sam_fakes(monkeypatch, drain_result, canceled_during_drain=False):
    from cinestudio import song
    lock = _Lock()
    monkeypatch.setattr(vt.server, "gpu_lock", lambda note=None: lock)
    monkeypatch.setattr(vt.server, "ensure_family", lambda key: {})
    monkeypatch.setattr(vt.server, "note_owned", lambda jid: None)
    flag = {"on": False}

    def drain(prefix, jid, backend=None, **k):
        if canceled_during_drain:
            flag["on"] = True
        return drain_result
    monkeypatch.setattr(song, "_drain", drain)
    return lock, flag


class _LostWatch:
    def submit(self, *a, **k):
        return "job-1"

    def watch(self, jid, progress):
        raise ConnectionError("lost")

    def cancel(self, jid):
        pass


def test_a_cancel_during_the_drain_ends_as_canceled(home, monkeypatch, tmp_path):
    lock, flag = _sam_fakes(monkeypatch, "ended", canceled_during_drain=True)
    with pytest.raises(workqueue.QueueCanceled):
        vt._sam(tmp_path / "cut.mp4", tmp_path / "p.glb", "t", lambda e: None, lambda: flag["on"],
                backend_factory=_LostWatch)
    assert lock.held is False


def test_an_unresolved_job_keeps_its_record_and_refuses_a_plain_retry(home, monkeypatch):
    clip = tiny_video(home / "d.mp4", 3.0, sound=False)
    vt.prepare("unres", clip, seconds=2.0)

    def sam(cut, glb, tag, emit, canceled):
        raise vt.MocapUnresolved("the ComfyUI job could not be confirmed as ended")
    with pytest.raises(vt.MocapUnresolved):
        vt.make("unres", sam_job=sam, build=lambda *a: None)
    assert shots.load("unres")["status"] == "unresolved"
    with pytest.raises(vt.TemplateError, match="unresolved"):
        vt.prepare("unres", clip)


def test_the_cancel_watch_finds_a_job_by_its_prefix(monkeypatch):
    calls = []

    class NoId:
        def find(self, prefix):
            return "job-7" if prefix == "cinestudio/mocap_x" else None

        def cancel(self, jid):
            calls.append(jid)
    import time as _t
    with vt._cancel_watch(NoId(), None, lambda: True, "cinestudio/mocap_x", every=0.01):
        for _ in range(200):
            if calls:
                break
            _t.sleep(0.01)
    assert calls == ["job-7"]


def test_settle_eases_the_end_into_the_first_pose():
    from cinestudio import mocap

    def frame(x, foot_z, contact_l):
        return {"dirs": {"thigh.L": [x, 0.0, -1.0], "upper_arm.R": [0.0, 0.0, -1.0]}, "hips": [x, 0.0],
                "pelvis_roll": 2.0 * x, "feet": {".R": [0.0, 0.0, 0.09], ".L": [x, 0.0, foot_z]},
                "contact": {".R": True, ".L": contact_l}, "kick": {".R": 0.0, ".L": x}}
    frames = [frame(0.0, 0.09, True)] + [frame(1.0, 0.40, False) for _ in range(39)]
    assert mocap.settle(frames, 16) == 16
    assert frames[23] == frame(1.0, 0.40, False)                   # before the window: unchanged
    last = frames[-1]
    assert last["hips"] == [0.0, 0.0] and last["kick"][".L"] == 0.0 and last["pelvis_roll"] == 0.0
    assert last["dirs"]["thigh.L"] == [0.0, 0.0, -1.0] and last["contact"] == {".R": True, ".L": True}
    # the lifted foot pins only when it is down (within 3 cm), never while it is still in the air
    for f in frames[24:]:
        if f["contact"][".L"]:
            assert f["feet"][".L"][2] - 0.09 <= 0.03 + 1e-9
    xs = [f["hips"][0] for f in frames[24:]]
    assert all(a >= b for a, b in zip(xs, xs[1:]))                 # a smooth, one-way ease
    assert mocap.settle(frames, 0) == 0


def test_the_tada_pose_lifts_the_popped_foot_and_the_hold_keeps_it_off_the_floor():
    """The single shan shen bu's quirky end (2026-10-24): one arm up, one hand on the hip, a head tilt and a knee pop."""
    like = mocap.rest_frame({"dirs": dict(mocap.REST_DIRS)})
    t = mocap.pose_frame(like, arms="tada", head_tilt_deg=16, knee_pop=".L", knee_pop_deg=30)
    assert t["dirs"]["upper_arm.R"][2] > 0.7                      # the right arm up
    assert t["dirs"]["thigh.L"][0] < 0 and t["dirs"]["thigh.L"][1] < 0   # the left knee forward and inward (-X)
    assert t["contact"] == {".R": True, ".L": False}
    assert t["feet"][".L"][2] > like["feet"][".L"][2] + 0.05      # the popped foot is off the floor
    assert all(isinstance(v, float) for v in t["feet"][".L"])
    frames = [mocap.rest_frame({"dirs": dict(mocap.REST_DIRS)}) for _ in range(40)]
    mocap.settle(frames, 12, t, hold=6, free_feet=True)
    assert frames[-1]["contact"] == {".R": True, ".L": False}     # the hold does not pin the lifted foot
    with pytest.raises(mocap.MocapError):
        mocap.pose_frame(like, knee_pop=".L", knee_deg=10)
