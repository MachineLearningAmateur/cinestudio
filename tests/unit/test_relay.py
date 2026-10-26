"""Latent relay repairs from the Codex review (2026-10-21)."""
import pytest

from cinestudio import comfy, runner


def test_relay_latent_recovers_from_the_runs_own_copy(tmp_path, monkeypatch):
    out, inp, keep = tmp_path / "out", tmp_path / "in", tmp_path / "keep"
    monkeypatch.setattr(comfy, "io_dir", lambda kind: out if kind == "output" else inp)
    (out / "cinestudio").mkdir(parents=True)
    (out / "cinestudio" / "relay_r1_1_s1_00001_.latent").write_bytes(b"L1")
    assert runner.relay_latent("cinestudio/relay_r1_1_s1", keep_dir=keep) == "relay_r1_1_s1.latent"
    assert not list((out / "cinestudio").iterdir())            # ComfyUI's copy is removed
    (inp / "relay_r1_1_s1.latent").unlink()                     # a resume: only the run's copy is left
    assert runner.relay_latent("cinestudio/relay_r1_1_s1", keep_dir=keep) == "relay_r1_1_s1.latent"
    assert (inp / "relay_r1_1_s1.latent").read_bytes() == b"L1"
    assert runner.relay_latent("cinestudio/relay_r1_2_s1", keep_dir=keep) is None


def test_adain_feeds_the_relay_save_too():
    g = {"crop": {"class_type": "LTXVCropGuides", "inputs": {}},
         "crop2": {"class_type": "LTXVCropGuides", "inputs": {}},
         "up": {"class_type": "X", "inputs": {"samples": ["crop", 2]}},
         "vdec": {"class_type": "X", "inputs": {"samples": ["crop2", 2]}}}
    runner.add_relay_save(g, "a", "b")
    runner.add_adain(g, 1.0, 768, 448)
    assert g["relay_save_crop"]["inputs"]["samples"] == ["adain_crop", 0]
    assert g["relay_save_crop2"]["inputs"]["samples"] == ["adain_crop2", 0]
    assert g["vdec"]["inputs"]["samples"] == ["adain_crop2", 0]


def test_the_clip_text_takes_the_look_text_costume():
    lt = ("a figure in a poncho; its hem ends at mid-thigh. a white mask. "
          "The arms are exactly in the pose of the mannequin in image 1.")
    t = runner.sheet_text("a short poncho that reaches below the chest", lt)
    assert t.startswith(runner.SHEET_PANELS) and "mid-thigh" in t
    assert "image 1" not in t and "below the chest" not in t
    assert runner.sheet_text("OLD text", None) == "OLD text"


def test_canceled_runs_are_hidden_and_deleted(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(comfy, "io_dir", lambda kind: tmp_path / kind)
    (tmp_path / "input").mkdir()
    for rid, status in (("r20261021-020000-aaaa", "canceled"), ("r20261021-030000-bbbb", "done")):
        d = runner.runs_dir() / rid
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"id": rid, "created": "x", "quality": "final"}))
        (d / "state.json").write_text(json.dumps({"status": status, "steps": {}}))
    (tmp_path / "input" / "relay_r20261021-020000-aaaa_1_s1.latent").write_bytes(b"L")
    assert [r["id"] for r in runner.list_runs(0)] == ["r20261021-030000-bbbb"] and runner.count_runs() == 1
    assert runner.cleanup_canceled() == ["r20261021-020000-aaaa"]
    assert not (runner.runs_dir() / "r20261021-020000-aaaa").exists()
    assert not list((tmp_path / "input").iterdir())
    assert (runner.runs_dir() / "r20261021-030000-bbbb").is_dir()


def test_pose_guide_goes_after_the_last_guide_of_each_stage_with_its_own_lora():
    import json
    g = json.loads((runner.WORKFLOWS / "ltx25_layouting_api.json").read_text())
    runner.add_keyframe_guides(g, [(60, "kf.png")])
    m1, m2 = g["guider1"]["inputs"]["model"], g["guider2"]["inputs"]["model"]
    runner.add_pose_guide(g, 0.7, stages=(1, 2))
    assert g["g_pose"]["inputs"]["positive"] == ["g_kf0_s1", 0] and g["g_pose2"]["inputs"]["positive"] == ["g_kf0_s2", 0]
    assert g["guider1"]["inputs"]["positive"] == ["g_pose", 0] and g["crop"]["inputs"]["latent"] != ["g_pose", 2]
    assert g["av"]["inputs"]["video_latent"] == ["g_pose", 2] and g["av2"]["inputs"]["video_latent"] == ["g_pose2", 2]
    assert g["pose_lora"]["inputs"]["model"] == m1 and g["pose_lora2"]["inputs"]["model"] == m2
    assert g["guider1"]["inputs"]["model"] == ["pose_lora", 0] and g["g_pose"]["inputs"]["strength"] == 0.7
    g1 = json.loads((runner.WORKFLOWS / "ltx25_layouting_api.json").read_text())
    runner.add_pose_guide(g1, 1.0)                              # stage 1 only: stage 2 untouched
    assert "g_pose2" not in g1 and g1["guider2"]["inputs"]["positive"] == ["g_look2", 0]


def test_a_canceled_run_with_a_finished_video_is_never_deleted(tmp_path, monkeypatch):
    """A cancel of a RETAKE on a finished run must not let the cleanup delete the run and its video."""
    import json
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(comfy, "io_dir", lambda kind: tmp_path / kind)
    d = runner.runs_dir() / "r20261022-010000-cccc"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"id": d.name, "created": "x", "quality": "final"}))
    (d / "state.json").write_text(json.dumps({"status": "canceled", "steps": {}}))
    (d / "final.mp4").write_bytes(b"video")
    assert runner.cleanup_canceled() == [] and (d / "final.mp4").is_file()


def test_preview_layout_graph_does_not_reach_stage_2():
    import json
    g = json.loads((runner.WORKFLOWS / "ltx25_layouting_api.json").read_text())
    runner.add_relay_save(g, "p_s1", None)
    runner.add_adain(g, 0.5, 768, 448)
    runner.to_stage1_only(g)
    assert "relay_save_crop2" not in g and "relay_save_crop" in g
    outputs = [k for k, v in g.items() if v["class_type"] in ("SaveVideo", "SaveLatent", "SaveImage")]
    seen, todo = set(), list(outputs)
    while todo:                                                   # every node an output needs
        n = todo.pop()
        if n in seen or n not in g:
            continue
        seen.add(n)
        todo += [v[0] for v in g[n]["inputs"].values() if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)]
    assert "ks" in seen and not {"ks2", "up", "guider2"} & seen        # stage 1 runs, stage 2 does not


def test_source_sigmas_always_descend():
    """Codex review 2026-10-22: a low start (0.05) once went UP to 0.066 at the next step."""
    for s in (0.02, 0.05, 0.2, 0.5, 0.8, 0.979, 0.985, 0.99, 1.0):
        sig = [float(x) for x in runner.source_sigmas(s).split(", ")]
        assert sig[0] == pytest.approx(s) and sig[-1] == 0 and all(a > b for a, b in zip(sig, sig[1:]))


def test_a_side_score_goes_stale_when_a_compared_clip_changes(tmp_path, monkeypatch):
    d = tmp_path / "run"
    (d / "work").mkdir(parents=True)
    c1, c2 = d / "work" / "clip.mp4", d / "work" / "clip_2.mp4"
    c1.write_bytes(b"one")
    c2.write_bytes(b"two")
    st = {"steps": {"clip": {"parts": {"1": {"outputs": {"clip": str(c1)}}, "2": {"outputs": {"clip": str(c2)}}}}}}
    runner._write_json(d / "state.json", st)
    runner.save_score(d, 2, None, {"video": str(c2), "deps": {"clip": runner.file_sha(c2), "prev": runner.file_sha(c1),
                                                               "clip1": runner.file_sha(c1)}, "picture": {}})
    assert "stale" not in runner.stored_score(d, 2)
    c1.write_bytes(b"one, made again")                             # clip 1 replaced: the overlap numbers are old
    assert "prev changed" in runner.stored_score(d, 2)["stale"]


def test_an_explicit_sound_text_always_wins():
    m = {"render_mode": "layout", "sound": "auto", "prompt": {"character": "c", "video": "A soundless dancer moves.",
                                                               "style": ""}}
    assert runner.sound_sentence(m) == " " + runner.SOUND_TEXT      # "soundless" is not the whole word "sound"
    m2 = dict(m, prompt=dict(m["prompt"], video="The dancer sings a song."))
    assert runner.sound_sentence(m2) == ""                          # the text already has sound: no default
    assert runner.sound_sentence(dict(m2, sound_text="Footsteps on stone")) == " Footsteps on stone."
    assert runner.sound_sentence(dict(m, sound_text="")) == ""


def test_new_runs_use_the_ingredients_25_headings():
    m = {"render_mode": "layout", "sound": "auto", "prompt_format": "ing25",
         "prompt": {"character": "**Left Panel (Character):** a face. **Panels 2-4 (Character):** the same character "
                                 "on black: a slim masked figure in a black bodysuit.", "video": "It waves.",
                    "style": "3dsrx, premium 3D"}}
    t = runner.video_prompt(m)
    assert t.startswith("### Reference Sheet Description **Left Panel (Character):**")
    assert "\n### Target Description 3dsrx, premium 3D. The character from the reference sheet, a slim masked figure" in t
    old = dict(m)
    old.pop("prompt_format")                                         # old runs keep the 2.3 format
    assert runner.video_prompt(old).startswith("Reference sheet: ")
