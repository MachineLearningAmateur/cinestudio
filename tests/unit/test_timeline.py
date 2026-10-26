"""cinestudio.timeline: clip texts written from the rig (2026-10-22)."""
import math

from cinestudio import timeline


def _raw(n=200, wave=(20, 80), step_at=None):
    frames = []
    for i in range(n):
        ang = math.radians(i * 360 / n)                   # the camera orbits once, starting in front (-Y)
        cam = [4 * math.sin(ang), -4 * math.cos(ang), 1.45]
        up = wave[0] <= i <= wave[1]
        hx = -0.28 + (0.08 * math.sin(i / 3) if up else 0)
        foot_r = [-0.11 + (0.3 if step_at and i >= step_at else 0), 0.01, 0.09]
        frames.append({"frame": i + 1, "camera": cam, "bones": {
            "head": [0, 0, 1.55], "hips": [0, 0, 0.95], "shoulder.L": [0.03, -0.01, 1.42], "shoulder.R": [-0.03, -0.01, 1.42],
            "hand.L": [0.28, -0.03, 0.8], "hand.R": [hx, -0.1, 1.88 if up else 0.8],
            "foot.L": [0.11, 0.01, 0.09], "foot.R": foot_r}})
    return {"frame_start": 1, "frame_end": n, "frames": frames}


def test_events_find_the_wave_and_planted_feet():
    ev = timeline.events(_raw())
    assert {"what": "waves its right hand", "frames": [21, 81]} in ev
    assert {"what": "feet planted", "frames": [1, 200]} in ev


def test_camera_angle_names_the_side_the_camera_sees():
    raw = _raw()
    assert timeline.view_words(timeline.camera_angle(raw["frames"][0])[0]) == "from the front"
    assert "left side" in timeline.view_words(timeline.camera_angle(raw["frames"][50])[0])   # +X = the rig's left
    assert timeline.view_words(timeline.camera_angle(raw["frames"][100])[0]) == "from behind"


def test_clip_texts_follow_the_clips():
    t = timeline.clip_texts(_raw(), "a rooftop at sunset", 200, 121, 25, 2)
    assert t["1"].startswith("A full-body shot on a rooftop at sunset.")
    assert "both feet planted" in t["1"] and "right hand" in t["1"] and "lowers it" in t["1"]
    assert "right hand" not in t["2"] and "relaxed at its sides" in t["2"]
    assert "holds this last pose" in t["2"]                  # clip 2 runs past the shot's end
    assert "by the end from the front" in t["2"]


def test_moving_feet_are_not_called_planted():
    t = timeline.clip_texts(_raw(step_at=60), "a park", 200, 121, 25, 2)
    assert "planted" not in t["1"] and "shifts its feet" in t["1"]
