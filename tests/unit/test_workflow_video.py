"""Workflow bindings, format rules and video helpers (ffmpeg from imageio-ffmpeg, no ComfyUI)."""

import json
import subprocess

import cv2
import numpy as np
import pytest

from cinestudio import video
from cinestudio.workflow import ProfileWorkflow


@pytest.mark.parametrize("profile,variant", [("ltx25", "layouting"), ("ltx25", "layout"), ("ltx25", "ref"), ("ltx25", "refup"),
                                             ("ltx25", "v2v"), ("ltx25", "v2vref"), ("ltx25", "restyle"), ("ltx25", "redraw"),
                                             ("ltx25dev", "ref"), ("ltx25dev", "refup"), ("ltx25dev", "v2v"),
                                             ("ltx25dev", "v2vref"), ("ltx25dev", "restyle"), ("zimage", "base"),
                                             ("qwenedit", "angle"), ("qwenedit", "edit"), ("qwenedit", "anypose")])
def test_bindings_load(profile, variant):
    ProfileWorkflow(profile, variant)


def test_ltx_needs_multiples_of_64():
    w = ProfileWorkflow("ltx25", "ref")
    w.check_format({"width": 1280, "height": 704, "frames": 121})
    with pytest.raises(ValueError, match="multiple of 64"):
        w.check_format({"width": 960, "height": 544})
    with pytest.raises(ValueError, match="8n\\+1"):
        w.check_format({"frames": 120})


@pytest.mark.parametrize("profile,variant", [("ltx25", "ref"), ("ltx25", "refup"), ("ltx25dev", "v2vref"),
                                             ("ltx25", "layouting"), ("zimage", "base")])
def test_style_lora_slot_is_removed_when_empty(profile, variant):
    w = ProfileWorkflow(profile, variant)
    before = w.graph["lora_style"]["inputs"]["model"]
    users = [(k, f) for k, n in w.graph.items() for f, v in n["inputs"].items() if v == ["lora_style", 0]]
    assert users
    for values in ({}, {"style_lora": None}, {"style_lora": "", "style_lora_strength": 0.5}):
        g = w.build(values)
        assert "lora_style" not in g
        assert all(g[k]["inputs"][f] == before for k, f in users)          # bridged to the slot's own input
        assert not any(isinstance(v, list) and v[:1] == ["lora_style"] for n in g.values() for v in n["inputs"].values())


def test_style_lora_slot_is_chained_when_given():
    w = ProfileWorkflow("ltx25", "ref")
    g = w.build({"style_lora": "my_style.safetensors", "style_lora_strength": 0.6})
    assert g["ks"]["inputs"]["model"] == ["lora_style", 0]
    assert g["lora_style"]["inputs"] == {"model": ["lora_ing", 0], "lora_name": "my_style.safetensors", "strength_model": 0.6}


def test_not_flag_inverts(tmp_path):
    """A binding target with a third element "not" receives the inverted boolean."""
    (tmp_path / "t_x_api.json").write_text(json.dumps({"n": {"class_type": "X", "inputs": {"use": True}},
                                                        "save": {"class_type": "S", "inputs": {}}}))
    (tmp_path / "t_x.bind.json").write_text(json.dumps({"inputs": {"skip": [["n", "use", "not"]]},
                                                         "outputs": {"video": "save"}}))
    w = ProfileWorkflow("t", "x", workflows_dir=tmp_path)
    assert w.build({"skip": True})["n"]["inputs"]["use"] is False
    assert w.build({"skip": False})["n"]["inputs"]["use"] is True


def test_overlap_arithmetic():
    assert video.clip_start(1, 121) == 0
    assert video.clip_start(2, 121) == 120
    assert video.clip_start(2, 121, overlap=9) == 112
    assert video.shot_length(3, 121) == 361
    assert video.n_clips(361, 121) == 3


def make(path, color, frames, fps=24, audio_s=0):
    ff = video.ffmpeg()
    args = [ff, "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=128x64:r={fps}"]
    if audio_s:
        args += ["-f", "lavfi", "-i", f"sine=f=440:r=48000:d={audio_s}", "-c:a", "aac"]
    subprocess.run(args + ["-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)


def test_stitch_mixed_audio(tmp_path):
    make(tmp_path / "a.mp4", "red", 25, audio_s=3)       # audio longer than video
    make(tmp_path / "b.mp4", "green", 25)                # no audio
    make(tmp_path / "c.mp4", "blue", 25, audio_s=0.5)    # audio shorter than video
    dst = tmp_path / "shot.mp4"
    video.stitch([tmp_path / n for n in ("a.mp4", "b.mp4", "c.mp4")], dst, 24.0, total_frames=73)
    assert video.probe(dst).frames == 73
    assert video.has_audio(dst)
    cap = cv2.VideoCapture(str(dst))
    colors = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        colors.append(int(f[32, 64].argmax()))           # 2 = red, 1 = green, 0 = blue (BGR)
    assert colors[24] == 2 and colors[25] == 1 and colors[48] == 1 and colors[49] == 0


def test_stitch_refuses_other_fps(tmp_path):
    make(tmp_path / "a.mp4", "red", 25)
    make(tmp_path / "b.mp4", "red", 25, fps=30)
    with pytest.raises(ValueError, match="fps"):
        video.stitch([tmp_path / "a.mp4", tmp_path / "b.mp4"], tmp_path / "x.mp4", 24.0)


def test_fit_image_keeps_aspect(tmp_path):
    src = tmp_path / "wide.png"
    subprocess.run([video.ffmpeg(), "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=white:s=400x100",
                    "-frames:v", "1", str(src)], check=True)
    video.fit_image(src, tmp_path / "pad.png", 200, 200, mode="pad")
    pad = cv2.imread(str(tmp_path / "pad.png"))
    assert pad.shape[:2] == (200, 200)
    assert pad[100, 0].sum() > 700 and pad[100, 199].sum() > 700     # full width kept (no crop)
    assert pad[0, 100].sum() == 0                                     # black border top
    video.fit_image(src, tmp_path / "cover.png", 200, 200, mode="cover")
    assert cv2.imread(str(tmp_path / "cover.png")).shape[:2] == (200, 200)


def test_controls_fit_the_generation_size_without_stretch():
    from cinestudio import passes
    t = passes.fit_transform(1536, 896, (768, 448))              # same shape: half size, no cut
    assert t == (0.5, 0.0, 0.0, 768, 448)
    t = passes.fit_transform(1280, 704, (768, 448))              # wider render: cover, cut both sides
    s, ox, oy, W, H = t
    assert abs(s - 448 / 704) < 1e-9 and oy == 0 and ox > 0 and (W, H) == (768, 448)
    img = np.zeros((704, 1280, 3), np.uint8)
    assert passes.fit_image(img, t).shape == (448, 768, 3)
    pts = passes.fit_points([[640, 352, True], [2, 352, True], None], t)
    assert abs(pts[0][0] - 384) < 1 and abs(pts[0][1] - 224) < 1 and pts[0][2]
    assert pts[1][2] is True and pts[1][0] < 0 and pts[2] is None   # cut away: still visible, drawn to the edge
    img = passes.draw_pose([[384, 224, True]] + [[-50, 224, True]] * 17, 768, 448)
    assert img[224, 0].any()                                     # the limb reaches the left edge


def test_depth_frames_read_from_disk_and_skeleton_over_a_base(tmp_path):
    import numpy as np
    from cinestudio import passes
    z = np.full((3, 8, 10), 5.0, np.float32)
    z[:, 2:6, 3:7] = 2.0                                          # a near box
    z[0, 0, 0] = np.inf                                           # no surface
    np.save(tmp_path / "depth_m.npy", z)
    d = passes.depth_frames(tmp_path)
    assert len(d) == 3 and d[0].shape == (8, 10, 3)
    assert d[0][3, 4, 0] > d[0][7, 9, 0] and d[0][0, 0, 0] == 0  # near = bright; no surface = black
    sk = np.zeros((8, 10, 3), np.uint8)
    sk[1, 1] = (0, 0, 255)
    out = passes.over(sk, d[1])
    assert tuple(out[1, 1]) == (0, 0, 255) and out[3, 4, 0] == d[1][3, 4, 0]   # the skeleton on top of a base


def test_a_run_refuses_an_unknown_control_kind(tmp_path):
    import pytest
    from cinestudio import runner
    (tmp_path / "x.blend").write_bytes(b"BLENDER")
    (tmp_path / "s.png").write_bytes(b"png")
    with pytest.raises(runner.RunError, match="control kind"):
        runner.new_run(tmp_path / "x.blend", tmp_path / "s.png", "c", "v", control_kind="canny")


def test_body_types_draw_chibi_thick_and_standard_as_the_dwpose_node():
    import numpy as np
    from cinestudio import passes
    pts = [[384, 60 + 20 * j, 1] for j in range(18)]                     # at the real control size, 768 x 448
    pts[2], pts[3], pts[4] = [340, 100, 1], [300, 160, 1], [280, 220, 1]   # a right arm
    chibi = passes.draw_pose(pts, 768, 448)
    std = passes.draw_pose_standard(pts, 768, 448)
    assert (chibi.max(axis=2) > 0).sum() > (std.max(axis=2) > 0).sum()   # bigger joints, full-width limbs
    assert chibi.max() == 255 and std[std.max(axis=2) > 0].max() == 255     # limbs in 60% colour, joints full
    assert passes.control_name("pose_env") == "pose_env.mp4"
    assert passes.control_name("pose_env", "standard") == "pose_env_standard.mp4"


def test_people_env_fits_alone_renders_of_another_shape(tmp_path):
    import json
    import cv2
    import numpy as np
    from cinestudio import passes
    (tmp_path / "rgb").mkdir()
    cv2.imwrite(str(tmp_path / "rgb" / "rgb_0001.png"), np.zeros((1080, 1920, 3), np.uint8))
    (tmp_path / "passes.json").write_text(json.dumps({"frames": 1, "width": 1920, "height": 1080, "fps": 24, "frame_start": 1}))
    people = tmp_path / "people"
    for c in range(2):
        (people / "iso" / str(c)).mkdir(parents=True)
        cv2.imwrite(str(people / "iso" / str(c) / "rgb_0001.png"), np.zeros((432, 768, 3), np.uint8))   # 40% of 1920x1080
    (people / "people.json").write_text(json.dumps({"characters": [{"name": "a"}, {"name": "b"}], "frame_start": 1}))
    frames = list(passes.people_env(tmp_path, people, (768, 448)))
    assert frames[0].shape == (448, 768, 3)


def test_stitch_blends_an_overlap_of_several_frames(tmp_path):
    import subprocess
    import numpy as np
    from cinestudio import video
    ff = video.ffmpeg()
    clips = []
    for i, colour in enumerate(("red", "blue", "lime")):
        p = tmp_path / f"c{i}.mp4"
        subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:s=64x32:r=24:d=5.0417",
                        "-f", "lavfi", "-i", "sine=f=440:r=48000:d=5.0417", "-frames:v", "121",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p)], check=True)
        clips.append(p)
    out = tmp_path / "joined.mp4"
    video.stitch(clips, out, 24.0, total_frames=None, overlap=25)
    assert video.probe(out).frames == 121 * 3 - 2 * 25
    raw = subprocess.run([ff, "-v", "error", "-i", str(out), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    v = np.frombuffer(raw, np.uint8).reshape(-1, 32, 64, 3).astype(int)
    mid = v[96 + 12, 16, 32]                          # the middle of the first overlap (frames 96..120)
    assert v[50, 16, 32][0] > 200 and v[50, 16, 32][2] < 60          # red before
    assert 80 < mid[0] < 180 and 80 < mid[2] < 180                    # half red, half blue
    assert v[150, 16, 32][2] > 200 and v[150, 16, 32][0] < 60        # blue after
    assert video.has_audio(out)


def test_loop_close_fades_the_tail_into_the_head(tmp_path):
    import subprocess
    import numpy as np
    from cinestudio import video
    frames = [np.full((32, 64, 3), (i % 48) * 4, np.uint8) for i in range(48)] + \
             [np.full((32, 64, 3), 200, np.uint8) for _ in range(8)]      # loop 48 frames + a tail of 8
    src = tmp_path / "s.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 32))
    for f in frames:
        w.write(f)
    w.release()
    out = tmp_path / "loop.mp4"
    video.loop_close(src, out, 48, 8, 24.0)
    raw = subprocess.run([video.ffmpeg(), "-v", "error", "-i", str(out), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    v = np.frombuffer(raw, np.uint8).reshape(-1, 32, 64)[:, 16, 32].astype(int)
    assert len(v) == 48                                   # cut to the loop
    assert abs(v[0] - (200 * 8 / 9)) < 12                 # starts as the tail (what followed the last frame)
    assert abs(v[7] - (200 / 9 + 28 * 8 / 9)) < 12        # ends as the head
    assert abs(v[20] - 80) < 8                            # the middle is untouched


_RNG = np.random.default_rng(1)
_BG = cv2.GaussianBlur(_RNG.integers(0, 120, (96, 160), dtype=np.uint8), (5, 5), 0)
_TEX = cv2.GaussianBlur(_RNG.integers(150, 255, (30, 24), dtype=np.uint8), (3, 3), 0)


def _square(x: int) -> np.ndarray:
    """A textured shape at column x on a textured background (the flow needs texture, as in a real picture)."""
    f = _BG.copy()
    f[40:70, x:x + 24] = _TEX
    return cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)


def test_morph_moves_a_shape_instead_of_doubling_it():
    a, b = [_square(40)] * 3, [_square(52)] * 3                         # the same moment, the shape 12 px apart
    mid = video.morph(a, b)[1].astype(int)                              # t = 0.5
    true = _square(46).astype(int)                                      # the shape half way
    fade = (a[1].astype(int) + b[1].astype(int)) // 2
    assert np.abs(mid - true).mean() < 0.4 * np.abs(fade - true).mean()


def test_stitch_flow_blend_keeps_frames_and_sound(tmp_path):
    ff = video.ffmpeg()
    clips = []
    for i, x in enumerate((40, 52)):
        p = tmp_path / f"c{i}.mp4"
        w = cv2.VideoWriter(str(tmp_path / f"c{i}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 24, (160, 96))
        for _ in range(41):
            w.write(_square(x))
        w.release()
        subprocess.run([ff, "-v", "error", "-y", "-i", str(tmp_path / f"c{i}.avi"), "-f", "lavfi", "-i",
                        "sine=f=440:r=48000:d=1.7083", "-frames:v", "41", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", str(p)], check=True)
        clips.append(p)
    out = tmp_path / "j.mp4"
    video.stitch(clips, out, 24.0, overlap=9, blend="flow")
    assert video.probe(out).frames == 41 * 2 - 9
    assert video.has_audio(out)
    assert not list(tmp_path.glob(".stitch_*"))                         # the temporary copies are gone
    cap = cv2.VideoCapture(str(out))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    mid = frames[32 + 4].astype(int)                                    # the middle of the join (frames 32..40)
    fade = (_square(40).astype(int) + _square(52).astype(int)) // 2
    true = _square(46).astype(int)
    assert np.abs(mid - true).mean() < np.abs(fade - true).mean()       # nearer one moved shape than two faded ones


def test_loop_close_flow_blend(tmp_path):
    src = tmp_path / "s.avi"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), 24, (160, 96))
    for i in range(56):
        w.write(_square(40 if i < 48 else 52))                          # the tail has the shape in another place
    w.release()
    out = tmp_path / "loop.mp4"
    video.loop_close(src, out, 48, 8, 24.0, blend="flow")
    assert video.probe(out).frames == 48
