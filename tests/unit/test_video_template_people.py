"""Templates from a video with SEVERAL people (workflows/sam3d_body_multi, mocap.people_motions, video_template.pick):
the graph binding, the label video, presence / trim / placement / order, the two-step "pick" worker with fake jobs, the
run's mannequin check and the web route. No GPU, no ComfyUI, no Blender."""
import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cinestudio import mocap, runner, shots, video_template as vt, workqueue
from cinestudio.workflow import ProfileWorkflow

from test_video_template import _person, tiny_video, write_glb

FPS = 24


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    return tmp_path


def _two(frames: int, late: int = 0) -> list:
    """Two people with ABSOLUTE roots (GLB axes): A 0.8 m left at 4 m, B 0.8 m right at 5 m (farther); B sways.
    Before frame `late` B's samples are garbage (a full-frame prediction while SAM 3 had not found B yet)."""
    ja, ra = _person(frames, lift=False)
    jb, rb = _person(frames, lift=True)
    ra = ra + np.array([-0.8, 0.0, -4.0])
    rb = rb + np.array([0.8, 0.0, -5.0])
    rb[:late] = [5.0, 2.0, -1.0]
    return [(ja, ra), (jb, rb)]


def _labels(frames: int, late: int = 0, objects: int = 2) -> dict:
    area = [[0.05] * frames, [0.0] * late + [0.04] * (frames - late)]
    box = [[[0.2, 0.2, 0.35, 0.9]] * frames, [None] * late + [[0.6, 0.25, 0.72, 0.85]] * (frames - late)]
    for _ in range(objects - 2):
        area.append([0.0] * frames)
        box.append([None] * frames)
    return {"frames": frames, "size": [640, 360], "area": area, "box": box}


def _label_video(path: Path, frames: int, late: int = 0) -> Path:
    """A label video like SAM3_TrackPreview without images: person 0 blue on the left, person 1 orange on the right
    from frame `late`."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (320, 180))
    for t in range(frames):
        img = np.zeros((180, 320, 3), np.uint8)
        for k, (x0, x1) in enumerate(((60, 110), (200, 240))):
            if k == 1 and t < late:
                continue
            r, g, b = (round(c * 255) for c in mocap.TRACK_COLORS[k])
            cv2.rectangle(img, (x0, 30), (x1, 160), (b, g, r), -1)
        w.write(img)
    w.release()
    return path


# ---------------------------------------------------------------- the graph
def test_the_multi_graph_tracks_every_person_with_absolute_places():
    from cinestudio import models
    wf = ProfileWorkflow("sam3d", "body_multi", models=models.job_bundle())
    g = wf.build({"fps": 24.0, "output_prefix": "cinestudio/x", "camera_translation": "absolute",
                  "person_text": "person", "max_people": 4})
    assert g["sam3"]["inputs"]["ckpt_name"] == models.job_bundle()["sam3"]["file"]
    assert g["track"]["inputs"]["conditioning"] == ["who", 0] and g["track"]["inputs"]["max_objects"] == 4
    assert g["pred"]["inputs"]["track_data"] == ["track", 0]
    assert "smooth" not in g and g["build"]["inputs"]["pose_data"] == ["pred", 0]        # optional: off by default
    assert g["build"]["inputs"]["camera_translation"] == "absolute" and g["build"]["inputs"]["track_index"] == -1
    assert "images" not in g["labels"]["inputs"] and g["labels"]["inputs"]["opacity"] == 1.0   # flat colours on black
    assert g["preview"]["inputs"]["images"] == ["comp", 0]
    assert {wf.bind["outputs"][k] for k in ("image", "labels", "preview")} == {"save", "labels", "preview"}
    g = wf.build({"smooth_strength": 0.5})
    assert g["build"]["inputs"]["pose_data"] == ["smooth", 0]
    one = ProfileWorkflow("sam3d", "body")                                              # the default stays as it was
    assert "track_data" not in one.graph["pred"]["inputs"]


def test_sam3_is_pinned_in_the_mocap_feature():
    from cinestudio import models
    e = models.bundle("8gb")["sam3"]
    assert e == models.bundle("large")["sam3"] and e["dir"] == "checkpoints" and len(e["sha256"]) == 64
    assert "/resolve/7bb8374780a725b4353ed31f3a9395c9742b5621/" in e["url"] and e["size"] == 1745546848
    assert "sam3" in models.features()["mocap"]


# ---------------------------------------------------------------- the label video and the choice
def test_the_label_video_gives_presence_and_boxes(tmp_path):
    lab = mocap.read_labels(_label_video(tmp_path / "l.avi", 30, late=10), 2)
    assert lab["frames"] == 30 and lab["size"] == [320, 180]
    pres = mocap.presence(lab)
    assert pres[0].all() and not pres[1][:10].any() and pres[1][10:].all()
    x0, y0, x1, y1 = lab["box"][0][5]
    assert abs(x0 - 60 / 320) < 0.03 and abs(x1 - 111 / 320) < 0.03 and abs(y1 - 161 / 180) < 0.03
    s = mocap.people_summary(lab)
    assert [p["track"] for p in s] == [0, 1] and s[1]["first_s"] == round(10 / 24, 2) and s[0]["x"] < s[1]["x"]
    assert mocap.choose_people(s, 2) == [0, 1] and mocap.choose_people(s, 1) == [0]


def test_presence_drops_flicker_and_closes_short_holes():
    a = [0.05] * 10 + [0.0] * 2 + [0.05] * 10 + [0.0] * 10 + [0.05] * 2 + [0.0] * 6
    p = mocap.presence({"area": [a], "box": [[None] * len(a)]})[0]
    assert p[:22].all() and not p[22:].any()             # the 2-frame hole closed, the 2-frame blip dropped


def test_tracks_are_checked():
    assert mocap.check_tracks(["1", 0], 3) == [1, 0]
    for bad, msg in (([], "at least one"), ([0, 0], "once"), ([5], "no person 5"), ([0, 1, 2, 3, 4], "at most 4"),
                     (["x"], "numbers")):
        with pytest.raises(mocap.MocapError, match=msg):
            mocap.check_tracks(bad, 6 if len(bad) > 3 else 3)


# ---------------------------------------------------------------- trim, places, order
def test_people_motions_trim_to_the_shared_time_and_keep_the_places(tmp_path):
    glb = write_glb(tmp_path / "two.glb", _two(96, late=24))
    r = mocap.people_motions(glb, _labels(96, late=24), [1, 0])
    assert r["span"] == [24, 95] and r["start_s"] == 1.0 and r["seconds"] == 3.0
    assert [p["track"] for p in r["people"]] == [0, 1]                 # left to right, whatever order was asked
    (xa, ya), (xb, yb) = r["people"][0]["place"], r["people"][1]["place"]
    s = r["motions"][0]["scale"]["scale_dancer_to_mannequin"]
    assert abs((xb - xa) - 1.6 * s) < 0.05 and abs((yb - ya) - 1.0 * s) < 0.05   # distance and depth order kept
    assert abs(xa + xb) < 1e-6 and abs(ya + yb) < 1e-6                  # centred on the middle of the two
    assert all(len(m["frames"]) == 72 for m in r["motions"])
    hips = [f["hips"][0] for f in r["motions"][1]["frames"]]
    assert max(abs(h) for h in hips) < 0.2                              # B's garbage samples (x = 5 m) were not used
    assert r["camera_distance"] >= 4.0


def _tilted_pair(frames: int, pitch_deg: float, seed: int = 3) -> list:
    """Two upright people seen by a camera pitched down by `pitch_deg` (GLB camera space, ABSOLUTE roots), with SAM 3D
    Body's depth jitter: each root slides +-10% along its camera ray (x and height are pixel x depth)."""
    a = np.radians(pitch_deg)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    rng = np.random.default_rng(seed)
    out = []
    for dx, dz, lift in ((-0.9, -4.0, False), (0.9, -5.0, True)):
        j, r = _person(frames, lift=lift, steps=True)
        r = (r + np.array([dx, -1.0, dz])) @ R.T                          # the camera 1 m above the floor
        r = r * (1 + 0.1 * rng.standard_normal(frames))[:, None]          # the depth jitter, along the ray
        out.append(({n: v @ R.T for n, v in j.items()}, r))
    return out


def test_a_tilted_camera_is_leveled_once_for_both_people(tmp_path):
    """User, 2026-10-25: "for the multi test duo why are the mannequins slanted": the per-person floor fit read the
    depth jitter along the slanted rays as a tilted floor. One shared level: both stand upright."""
    glb = write_glb(tmp_path / "tilt.glb", _tilted_pair(96, 10.0))
    r = mocap.people_motions(glb, _labels(96), [0, 1])
    for m in r["motions"]:
        assert m["scale"]["floor_from"].startswith("shared:") and m["scale"]["torso_tilt_deg"] < 2.0
        assert abs(m["scale"]["floor_tilt_deg"] - 10.0) < 2.5
    assert r["motions"][0]["scale"]["floor_tilt_deg"] == r["motions"][1]["scale"]["floor_tilt_deg"]
    (xa, ya), (xb, yb) = r["people"][0]["place"], r["people"][1]["place"]
    assert 1.5 < xb - xa < 2.2 and ya < yb                              # distance and depth order kept
    old = [mocap.torso_tilt(mocap.motion_from_glb(glb, None, track=k)["frames"]) for k in (0, 1)]
    assert max(old) > 3.0                                               # the old per-person path leaned


def test_ray_root_keeps_the_picture_place():
    root = np.array([[0.5, -0.4, -4.0], [0.55, -0.44, -4.4], [0.45, -0.36, -3.6]])
    on = mocap.ray_root(root)
    assert np.allclose(on[:, 2], -4.0) and np.allclose(on[:, 0] / on[:, 2], root[:, 0] / root[:, 2])
    centred = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.1]])
    assert np.array_equal(mocap.ray_root(centred), centred)            # a "centered" GLB stays as it is


def test_one_person_out_of_two_stands_in_the_middle(tmp_path):
    glb = write_glb(tmp_path / "two.glb", _two(72))
    r = mocap.people_motions(glb, _labels(72), [1])
    assert r["people"][0]["place"] == [0.0, 0.0] and r["camera_distance"] == 4.0


def test_too_little_shared_time_is_refused(tmp_path):
    glb = write_glb(tmp_path / "two.glb", _two(72, late=40))
    with pytest.raises(mocap.MocapError, match="together for 1.3 s"):
        mocap.people_motions(glb, _labels(72, late=40), [0, 1])


def test_merge_places_survive_argparse(tmp_path, monkeypatch):
    import argparse
    seen = []
    monkeypatch.setattr(shots, "_run_blender", lambda script, blend, args, log: seen.append(list(map(str, args))))
    vt._merge([tmp_path / "a.blend", tmp_path / "b.blend", tmp_path / "c.blend"], tmp_path / "o.blend",
              [[-0.951, 0.784], [0.0, 0.0], [0.951, -0.784]], tmp_path / "log")
    args = [x for x in seen[0] if x.startswith("--place")]
    assert args == ["--places=-0.951,0.784;0.0,0.0;0.951,-0.784"]
    assert [x for x in seen[0] if x.endswith(".blend") and "o.blend" not in x] ==         [str(tmp_path / "b.blend"), str(tmp_path / "c.blend")]          # every other shot after its --other
    ap = argparse.ArgumentParser()
    ap.add_argument("--places")
    a = ap.parse_args(args)                                             # the first try failed: "expected one argument"
    assert a.places == "-0.951,0.784;0.0,0.0;0.951,-0.784"


def test_places_three_people_keep_depth_order_and_spacing():
    p = mocap.place_people([[-1.0, 2.0], [0.0, 5.0], [1.0, 9.0]], [1.0, 1.0, 1.0])
    ys = [y for _, y in p]
    assert ys[0] < ys[1] < ys[2]                                         # the order kept (a clamp made two equal)
    assert ys[2] - ys[0] == pytest.approx(mocap.DEPTH_MAX_M, abs=0.002)
    q = mocap.place_people([[0.0, 4.0], [0.1, 4.0], [0.2, 4.0]], [1.0, 1.0, 1.0])
    xs = sorted(x for x, _ in q)
    assert xs[1] - xs[0] >= mocap.MIN_APART_M - 1e-3 and xs[2] - xs[1] >= mocap.MIN_APART_M - 1e-3
    assert sum(xs) / 3 == pytest.approx(0.0, abs=0.01)


def test_choose_people_up_to_four():
    s = [{"track": k, "visible": 0.9, "height": 0.5, "x": x} for k, x in enumerate((0.8, 0.2, 0.5, 0.1, 0.6))]
    assert mocap.choose_people(s, mocap.PEOPLE_MAX) == [3, 1, 2, 0]      # four, left to right
    assert mocap.PEOPLE_MAX == runner.MAX_CHARACTERS == 4
    with pytest.raises(mocap.MocapError):
        mocap.check_tracks([0, 1, 2, 3, 4], 6)
    assert mocap.check_tracks([2, 0, 1], 4) == [2, 0, 1]


def test_places_keep_two_mannequins_apart_and_clamp_the_depth():
    assert mocap.place_people([[0.1, 4.0], [0.2, 4.0]], [1.0, 1.0]) == [[-0.25, 0.0], [0.25, 0.0]]
    p = mocap.place_people([[-1.0, 2.0], [1.0, 9.0]], [1.0, 1.0])
    assert p[0][1] == -0.75 and p[1][1] == 0.75
    far = mocap.camera_distance([[-3.0, 0.0], [3.0, 0.0]], [{"frames": [{"hips": [0.0, 0.0]}]}] * 2)
    assert far > 6.0                                                    # both stay in the frame


# ---------------------------------------------------------------- the worker: scan, pick, build
def _fake_multi_sam(people, frames, late=0):
    def job(cut, glb, tag, emit, canceled, multi=False):
        assert multi
        write_glb(glb, people)
        _label_video(Path(glb).parent / "labels.mp4", frames, late)
        Path(glb).parent.joinpath("track_preview.mp4").write_bytes(b"x")
        return glb
    return job


def _fakes(calls):
    def build(motion, out, audio, log, camera_distance=None):
        calls.append(("build", json.loads(Path(motion).read_text())["person"], camera_distance, audio))
        Path(out).write_bytes(b"BLENDER-fake")

    def merge(parts, out, places, log):
        calls.append(("merge", *[Path(x).name for x in parts], places))
        Path(out).write_bytes(f"BLENDER-merged-{len(parts)}".encode())

    def scan(blend):
        raw = Path(blend).read_bytes()
        n = int(raw.rsplit(b"-", 1)[1]) if raw.startswith(b"BLENDER-merged-") else 1
        return {"characters": [{"collection": "Mannequin" + ("" if i == 0 else str(i + 1)), "rig": "R"} for i in range(n)],
                "props": [], "set": ["Floor"], "loose": [], "camera_moves": False}
    return {"build": build, "merge": merge, "scan_layers": scan}


def test_pick_scans_once_then_builds_two_mannequins_without_the_gpu(home, monkeypatch):
    clip = tiny_video(home / "two.mp4", 3.0, sound=False)
    rec = vt.prepare("duo", clip, seconds=3.0, people="pick")
    assert rec["params"]["people"] == "pick" and "tracks" not in rec["params"]
    monkeypatch.setattr(shots, "make_preview", lambda d, blend, log: {"frames": 72})
    calls = []
    r = vt.make("duo", sam_job=_fake_multi_sam(_two(72), 72), **_fakes(calls))
    d = shots.shot_dir("duo")
    assert r["status"] == "pick" and not calls and workqueue.find("template", "duo") is None
    assert r["people"]["count"] == 2 and r["people"]["seen"] == 2 and Path(r["people"]["sheet"]).is_file()
    assert (d / "people" / "pose.glb").is_file() and (d / "people" / "labels.json").is_file()
    with pytest.raises(vt.TemplateError, match="once"):
        vt.pick("duo", [0, 1, 1])
    with pytest.raises(vt.TemplateError, match="no person 3"):
        vt.pick("duo", "3")
    assert shots.load("duo")["status"] == "pick"                         # a refused choice changes nothing

    def no_gpu(*a, **k):
        raise AssertionError("the pick step must not run the GPU job again")
    r = vt.pick("duo", "1,0", wait=True, sam_job=no_gpu, **_fakes(calls))
    assert r["status"] == "done" and r["params"]["tracks"] == [1, 0]
    assert [c[0] for c in calls] == ["build", "build", "merge"]
    assert calls[0][1]["track"] == 0 and calls[1][1]["track"] == 1          # the left one first (Mannequin)
    assert calls[0][2] == calls[1][2] == r["people"]["camera_distance"]     # one camera for both
    places = calls[2][3]
    assert places[0][0] < 0 < places[1][0] and places[0][1] < places[1][1]  # B farther, as in the video
    assert [c["collection"] for c in r["people"]["chosen"]] == ["Mannequin", "Mannequin2"]
    assert [c["collection"] for c in r["layers"]["scan"]["characters"]] == ["Mannequin", "Mannequin2"]
    assert (d / "motion.json").is_file() and (d / "motion2.json").is_file() and r["motion"]["people"] == 2
    calls.clear()                                                       # a done template can be chosen again
    r = vt.pick("duo", [1], wait=True, sam_job=no_gpu, **_fakes(calls))
    assert r["status"] == "done" and [c[0] for c in calls] == ["build"] and len(r["layers"]["scan"]["characters"]) == 1


def test_two_people_build_at_once_and_the_sound_follows_the_trim(home, monkeypatch):
    clip = tiny_video(home / "two.mp4", 4.0, sound=True)
    vt.prepare("pair", clip, seconds=4.0, people="2")
    monkeypatch.setattr(shots, "make_preview", lambda d, blend, log: {"frames": 72})
    cuts = []
    real = vt._sound

    def sound(src, start, seconds, dst):
        cuts.append((round(start, 3), round(seconds, 3)))
        return real(src, start, seconds, dst)
    monkeypatch.setattr(vt, "_sound", sound)
    calls = []
    r = vt.make("pair", sam_job=_fake_multi_sam(_two(96, late=24), 96, late=24), **_fakes(calls))
    assert r["status"] == "done" and [c[0] for c in calls] == ["build", "build", "merge"]
    assert r["people"]["trimmed"] and r["people"]["span_s"] == [1.0, 4.0]
    assert cuts == [(0.0, 4.0), (1.0, 3.0)] and calls[0][3] is not None and calls[1][3] is None   # sound on A only


def test_one_person_found_when_two_were_asked_builds_one_mannequin(home, monkeypatch):
    clip = tiny_video(home / "one.mp4", 3.0, sound=False)
    vt.prepare("solo", clip, seconds=3.0, people="2")
    monkeypatch.setattr(shots, "make_preview", lambda d, blend, log: {"frames": 72})
    calls = []
    r = vt.make("solo", sam_job=_fake_multi_sam(_two(72, late=72), 72, late=72), **_fakes(calls))
    assert r["status"] == "done" and [c[0] for c in calls] == ["build"] and "one mannequin" in r["warnings"][0]


def test_people_arguments_are_checked(home):
    clip = tiny_video(home / "d.mp4", 3.0, sound=False)
    with pytest.raises(vt.TemplateError, match="people must be"):
        vt.prepare("x", clip, people="5")
    with pytest.raises(vt.TemplateError, match="at most 4"):
        vt.prepare("x", clip, tracks="0,1,2,3,4")
    assert vt.prepare("x3", clip, people="3")["params"]["people"] == "3"
    assert vt.prepare("xa", clip, people="all")["params"]["people"] == "all"
    assert vt.prepare("y", clip, tracks="2,0")["params"] == {**vt.prepare("y2", clip, tracks=[2, 0])["params"],
                                                             "original": "d.mp4"}
    assert shots.load("y")["params"]["people"] == "pick" and shots.load("y")["params"]["tracks"] == [2, 0]
    assert vt.prepare("z", clip)["params"]["people"] == "1"


# ---------------------------------------------------------------- runs and the app
def test_a_two_mannequin_template_needs_a_second_character(monkeypatch):
    monkeypatch.setattr(shots, "library_record", lambda blend: {
        "layers": {"scan": {"characters": [{"collection": "Mannequin"}, {"collection": "Mannequin2"}]}}})
    assert runner.template_mannequins("x.blend") == ["Mannequin", "Mannequin2"]
    monkeypatch.setattr(shots, "library_record", lambda blend: None)
    assert runner.template_mannequins("x.blend") == []


def test_new_run_refuses_a_bare_second_mannequin(tmp_path, monkeypatch):
    from cinestudio import character as chars
    blend = tmp_path / "shot.blend"
    blend.write_bytes(b"x")
    sheet = tmp_path / "sheet.png"
    cv2.imwrite(str(sheet), np.zeros((8, 8, 3), np.uint8))
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(chars, "require_video_ready", lambda name: {})
    monkeypatch.setattr(runner, "template_mannequins", lambda b: ["Mannequin", "Mannequin2"])
    with pytest.raises(runner.RunError, match=r"2 characters \(Character 1 \(Mannequin\) = --character #1; "
                                              r"Character 2 \(Mannequin2\) = --character #2\).*1 given"):
        runner.new_run(blend, sheet, "a", "b", character_name="masked", proxy="off")
    monkeypatch.setattr(runner, "template_mannequins", lambda b: ["Mannequin", "Mannequin2"])
    with pytest.raises(runner.RunError, match=r"3 given"):              # more characters than a 2-slot template
        runner.new_run(blend, sheet, "a", "b", characters=["masked", "fox", "kit"], proxy="off")
    monkeypatch.setattr(runner, "template_mannequins", lambda b: ["Mannequin", "Mannequin2", "Mannequin3"])
    with pytest.raises(runner.RunError, match=r"3 characters .*Mannequin3\) = --character #3.*2 given"):
        runner.new_run(blend, sheet, "a", "b", characters=["masked", "fox"], proxy="off")


def test_web_people_route_and_view(home, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer
    from cinestudio import webapp
    from cinestudio.webapi import templates as wt
    calls = []
    monkeypatch.setattr(vt, "pick", lambda name, tracks: (calls.append((name, tracks)) or
                                                          {"name": name, "status": "queued", "tracks": tracks}))

    async def go():
        async with TestClient(TestServer(webapp.make_app()), headers={"X-Cinestudio": "1"}) as c:
            r = await c.post("/api/templates/duo/people", json={"tracks": [0, 2]})
            assert r.status == 200 and (await r.json())["data"]["tracks"] == [0, 2]
            r = await c.post("/api/templates/duo/people", json={"tracks": "0"})
            assert r.status >= 400
    asyncio.run(go())
    assert calls == [("duo", [0, 2])]
    v = wt._people_view({"source": "video", "params": {"people": "pick"},
                         "people": {"count": 3, "seen": 2, "tracks": [{"track": 0}], "sheet": None}})
    assert v["count"] == 3 and v["asked"] == "pick" and v["sheet_url"] is None and v["chosen"] == []
    assert wt._people_view({"source": "song"}) is None


def _three(frames: int) -> list:
    """Three people with ABSOLUTE roots: left at 4 m, middle at 4.5 m, right at 5 m."""
    out = []
    for x, z, lift in ((-1.2, -4.0, False), (0.0, -4.5, True), (1.2, -5.0, False)):
        j, r = _person(frames, lift=lift)
        out.append((j, r + np.array([x, 0.0, z])))
    return out


def _label_video3(path: Path, frames: int) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (320, 180))
    for _ in range(frames):
        img = np.zeros((180, 320, 3), np.uint8)
        for k, (x0, x1) in enumerate(((30, 70), (140, 180), (250, 290))):
            r, g, b = (round(c * 255) for c in mocap.TRACK_COLORS[k])
            cv2.rectangle(img, (x0, 30), (x1, 160), (b, g, r), -1)
        w.write(img)
    w.release()
    return path


def test_three_people_become_three_mannequins_left_to_right(home, monkeypatch):
    clip = tiny_video(home / "three.mp4", 3.0, sound=False)
    vt.prepare("trio", clip, seconds=3.0, people="3")
    monkeypatch.setattr(shots, "make_preview", lambda d, blend, log: {"frames": 72})

    def sam(cut, glb, tag, emit, canceled, multi=False):
        write_glb(glb, _three(72))
        _label_video3(Path(glb).parent / "labels.mp4", 72)
        Path(glb).parent.joinpath("track_preview.mp4").write_bytes(b"x")
        return glb
    calls = []
    r = vt.make("trio", sam_job=sam, **_fakes(calls))
    assert r["status"] == "done" and [c[0] for c in calls] == ["build", "build", "build", "merge"]
    assert [c[1]["track"] for c in calls[:3]] == [0, 1, 2]                 # left to right
    places = calls[3][4]
    assert places[0][0] < places[1][0] < places[2][0] and places[0][1] < places[1][1] < places[2][1]
    assert [c["collection"] for c in r["people"]["chosen"]] == ["Mannequin", "Mannequin2", "Mannequin3"]
    assert [c["rig"] for c in r["people"]["chosen"]] == ["MannequinRig", "MannequinRig2", "MannequinRig3"]
    assert len(r["layers"]["scan"]["characters"]) == 3 and r["motion"]["people"] == 3
    assert (shots.shot_dir("trio") / "motion3.json").is_file()
    assert vt.mannequin_names(3) == ("Mannequin4", "MannequinRig4")
