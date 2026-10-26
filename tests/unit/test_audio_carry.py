"""The audio carry (2026-10-24, docs/audio_consistency_research.md): clip k-1's stage-1 audio latent cut to the shared
frames and frozen at the start of clip k, the stage-2 audio frozen, and a join that does not add +3 dB. No GPU."""
import json
import subprocess
import types

import numpy as np

from cinestudio import latent_file, runner, video
from cinestudio.workflow import ProfileWorkflow


def test_the_audio_tail_is_cut_from_a_saved_latent(tmp_path):
    t = np.arange(126, dtype=np.float32)[None, None, :, None] * np.ones((1, 8, 126, 16), np.float32)
    src, dst = tmp_path / "a.latent", tmp_path / "a_tail.latent"
    latent_file.write(src, {"latent_format_version_0": ("F32", np.zeros((0,), np.float32)),
                            "latent_tensor": ("F32", t)}, {"prompt": "{}"})
    assert latent_file.audio_tail_start(121, 25, 24.0) == 100          # 96 frames = 4.0 s = 100 audio latents
    assert latent_file.cut_tail(src, dst, 100) == 26
    tensors, meta = latent_file.read(dst)
    dtype, arr = tensors["latent_tensor"]
    assert dtype == "F32" and arr.shape == (1, 8, 26, 16) and arr[0, 0, 0, 0] == 100 and arr[0, 7, -1, 15] == 125
    assert "latent_format_version_0" in tensors and meta["prompt"] == "{}"   # LoadLatent must not rescale it
    raw = dst.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    assert n % 8 == 0 and json.loads(raw[8:8 + n])["latent_tensor"]["shape"] == [1, 8, 26, 16]


def test_the_layout_graph_saves_freezes_and_pins_the_audio():
    wf = ProfileWorkflow("ltx25", "layout")
    g = json.loads(json.dumps(wf.graph))
    runner.add_audio_carry(g, "cinestudio/relay_x_1_a", None)          # clip 1: save + stage-2 freeze, no tail
    assert g["audio_save"]["inputs"]["samples"] == ["sep", 1]
    assert g["av2"]["inputs"]["audio_latent"] == ["audio_freeze2", 0] and "av_carry" not in g
    g = json.loads(json.dumps(wf.graph))
    runner.add_audio_carry(g, "cinestudio/relay_x_2_a", "relay_x_1_a_tail.latent")
    assert g["ks"]["inputs"]["latent_image"] == ["av_carry", 0]
    assert g["av_carry"]["inputs"] == {"video_latent": ["av", 0], "audio_latent": ["atail_frz", 0]}
    assert g["atail_frz"]["class_type"] == "LTXVFreezeLatent" and g["atail_load"]["inputs"]["latent"].endswith("_tail.latent")
    runner.to_stage1_only(g)                                            # preview: the stage-1 audio (with the head)
    assert g["adec"]["inputs"]["samples"] == ["sep", 1]


def _clips(tmp_path):
    """Three 121-frame clips whose sound is ONE continuous tone cut at 0, 4.0 and 8.0 s: like carried clips, the shared
    second has the same signal in both clips."""
    ff = video.ffmpeg()
    tone = tmp_path / "tone.wav"
    subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=220:r=48000:d=14", str(tone)], check=True)
    clips = []
    for i, colour in enumerate(("red", "blue", "lime")):
        p = tmp_path / f"c{i}.mp4"
        subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:s=64x32:r=24:d=5.0417",
                        "-ss", f"{4.0 * i:.3f}", "-t", "5.0417", "-i", str(tone), "-frames:v", "121",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", "-shortest",
                        str(p.with_suffix(".mov"))], check=True)
        clips.append(p.with_suffix(".mov"))
    return clips


def _rms_db(path, t0, t1):
    ff = video.ffmpeg()
    raw = subprocess.run([ff, "-v", "error", "-i", str(path), "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-vn",
                          "-f", "f32le", "-ac", "1", "-ar", "48000", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


def test_a_carried_join_keeps_the_length_and_has_no_loudness_bump(tmp_path):
    clips = _clips(tmp_path)
    fade, carry = tmp_path / "fade.mp4", tmp_path / "carry.mp4"
    video.stitch(clips, fade, 24.0, overlap=25, blend="fade")
    video.stitch(clips, carry, 24.0, overlap=25, blend="fade", audio_join="carry")
    frames = 121 * 3 - 2 * 25
    for out in (fade, carry):
        assert video.probe(out).frames == frames
    ref = _rms_db(carry, 2.0, 3.0)                                       # inside clip 1
    mid = 4.0 + 25 / 24 / 2                                              # the middle of the first shared second
    assert _rms_db(fade, mid - 0.05, mid + 0.05) - ref > 2.0             # equal-power fade of one signal: about +3 dB
    assert abs(_rms_db(carry, mid - 0.05, mid + 0.05) - ref) < 0.5       # the carry join: no bump
    sw = 4.0 + 25 / 24 - video.AUDIO_SWITCH_BEFORE                       # and no dip at the switch
    assert abs(_rms_db(carry, sw - 0.03, sw + 0.03) - ref) < 1.0


def _fake_run(parts, relay="carry"):
    return types.SimpleNamespace(m={"audio_relay": relay}, st={"steps": {"clip": {"parts": parts}}})


def test_the_join_uses_carry_only_when_every_tail_came_from_the_clip_in_use():
    ok = {"1": {"relay": {"a": "c1_a.latent"}},
          "2": {"relay": {"a": "c2_a.latent"}, "audio_carry": {"source": "c1_a.latent"}},
          "3": {"relay": {"a": "c3_a.latent"}, "audio_carry": {"source": "c2_a.latent"}}}
    assert runner.Run._audio_join(_fake_run(ok)) == ["carry", "carry"]
    swapped = json.loads(json.dumps(ok))
    swapped["2"]["relay"]["a"] = "c2_take3_a.latent"                    # a take put in for clip 2: clip 3's tail is old
    assert runner.Run._audio_join(_fake_run(swapped)) == ["carry", "fade"]   # only ITS join changes (per boundary)
    assert runner.Run._audio_join(_fake_run(ok, relay="off")) == ["fade", "fade"]
    old = {"1": {"relay": {}}, "2": {"relay": {}}}                       # a run made before the carry
    assert runner.Run._audio_join(_fake_run(old)) == ["fade"]


def test_mixed_join_modes_keep_the_length_and_each_join_its_own_mode(tmp_path):
    clips = _clips(tmp_path)
    out = tmp_path / "mixed.mp4"
    video.stitch(clips, out, 24.0, overlap=25, blend="fade", audio_join=["carry", "fade"])
    assert video.probe(out).frames == 121 * 3 - 2 * 25
    ref = _rms_db(out, 2.0, 3.0)
    mid1, mid2 = 4.0 + 25 / 24 / 2, 8.0 + 25 / 24 / 2
    assert abs(_rms_db(out, mid1 - 0.05, mid1 + 0.05) - ref) < 0.5      # carried join: no bump
    assert _rms_db(out, mid2 - 0.05, mid2 + 0.05) - ref > 2.0            # faded join (equal power on one signal)
    import pytest
    with pytest.raises(ValueError):
        video.stitch(clips, out, 24.0, overlap=25, blend="fade", audio_join=["carry"])
