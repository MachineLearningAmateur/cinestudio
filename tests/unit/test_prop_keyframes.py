"""Prop keyframes (2026-10-27): per clip, the frame where a prop is largest and fully in the picture is redrawn with
the prop (runner.pick_prop_frame chooses it from bpy_jobs/prop_track.py's boxes)."""
from cinestudio import runner


def ball(box, front=True, name="Ball_Dodge1"):
    return {"name": name, "box": box, "front": front}


def track_with(frames: dict, n=60):
    t = [[] for _ in range(n)]
    for i, props in frames.items():
        t[i] = props
    return t


def test_the_largest_prop_inside_the_picture_wins():
    t = track_with({10: [ball([0.4, 0.1, 0.5, 0.2])],        # 0.10 high
                    15: [ball([0.4, 0.1, 0.55, 0.3])],       # 0.20 high: the largest
                    20: [ball([0.4, 0.3, 0.45, 0.37])]})
    p = runner.pick_prop_frame(t, 0, 0, 59)
    assert p["local"] == 15 and p["shot"] == 15 and p["names"] == ["Ball_Dodge1"] and p["height"] == 0.2


def test_a_prop_cut_by_the_edge_or_behind_the_camera_does_not_count():
    t = track_with({14: [ball([0.5, -0.03, 0.65, 0.24])],     # cut by the top edge
                    16: [ball([0.4, 0.1, 0.6, 0.4], front=False)],
                    20: [ball([0.4, 0.3, 0.45, 0.37])]})
    assert runner.pick_prop_frame(t, 0, 0, 59)["local"] == 20


def test_too_small_props_and_frames_near_other_keyframes_are_skipped():
    t = track_with({10: [ball([0.4, 0.4, 0.42, 0.43])],      # 0.03 high: too small
                    30: [ball([0.4, 0.1, 0.5, 0.3])]})
    assert runner.pick_prop_frame(t, 0, 0, 59) ["local"] == 30
    assert runner.pick_prop_frame(t, 0, 0, 59, avoid=[25]) is None      # 5 frames from another keyframe
    assert runner.pick_prop_frame(t, 0, 0, 29) is None                  # outside lo..hi


def test_the_clip_start_offsets_the_shot_frame():
    t = track_with({110: [ball([0.4, 0.1, 0.5, 0.3])]}, n=200)
    p = runner.pick_prop_frame(t, 96, 10, 90)
    assert p["local"] == 14 and p["shot"] == 110
    assert runner.pick_prop_frame(t, 96, 27, 90) is None             # inside the shared frames: skipped


def test_prop_words_come_from_the_layers_and_the_runs_own_words():
    m = {"layers": {"descriptions": {"props": {"Ball_*": "plain dodgeballs"}}}, "props": {"Ball_*": "demon eyeballs"},
         "prompt": {"video": ""}}
    assert runner.prop_words(m) == "demon eyeballs"
    assert runner.prop_words({"prompt": {"video": ""}}) == ""


def test_the_option_is_checked_and_old_runs_read_off():
    assert runner.DEFAULTS["prop_keyframes"] == "auto"
