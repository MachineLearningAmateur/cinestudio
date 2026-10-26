import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SETUP = Path(__file__).resolve().parents[2] / "setup"
sys.path.insert(0, str(SETUP))
spec = importlib.util.spec_from_file_location("pose_fusion", SETUP / "pose_fusion.py")
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)

RIG = np.array([[100 + 30 * (j % 5), 50 + 25 * j, 1] for j in range(18)], float)


def person(shift=0.0, face=True, hands=True):
    body = RIG.copy()
    body[:, 0] += shift
    body[:, 2] = 0.9
    hand = lambda w: np.array([[w[0] + k, w[1] + k, 0.9] for k in range(21)], float)   # noqa: E731
    return {"body": body, "L": hand(body[7]) if hands else None, "R": hand(body[4]) if hands else None,
            "face": np.array([[200 + k, 60, 0.9] for k in range(70)], float) if face else None}


def test_owner_takes_each_part_whole_from_its_detector():
    a, b = person(0.0), person(8.0)
    fused, src = pf.fuse(a, b, 450.0, RIG, "owner", dict(torso="b", arms="a", legs="b", head="a", hands="a", face="a"))
    assert src["torso"] == "b" and src["arm_r"] == "a" and src["leg_l"] == "b"
    assert fused["body"][3, 0] == a["body"][3, 0] and fused["body"][2, 0] == b["body"][2, 0]


def test_hands_go_one_to_one_to_the_wrists():
    a = person(0.0)
    a["L"], a["R"] = a["R"], a["L"]                 # swapped hand labels (as SDPose on a back view)
    out = pf.assign_hands(a, RIG, 450.0)
    assert out["L"] is not None and out["R"] is not None
    assert np.allclose(out["L"][0, :2], RIG[7, :2]) and np.allclose(out["R"][0, :2], RIG[4, :2])


def test_the_face_needs_both_detectors():
    fused, _ = pf.fuse(person(0.0), person(0.0, face=False), 450.0, RIG, "owner",
                       dict(torso="a", arms="a", legs="a", head="a", hands="a", face="a"))
    assert fused["face"] is None
    single, src = pf.fuse(person(0.0), None, 450.0, RIG, "owner", {})
    assert single["face"] is None and np.isnan(single["body"][0, 0]) and src == {"whole": "a"}


def test_dets_without_blender_boxes_are_refused(tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"detector": "dwpose", "box": "yolox", "frames": {}}))
    with pytest.raises(SystemExit):
        pf.load(f)


def test_a_part_that_would_not_join_the_torso_comes_from_the_torso_detector():
    a, b = person(0.0), person(200.0)              # b sees the whole body 200 px to the right
    rig = a["body"].copy()
    rig[[3, 4]] = b["body"][[3, 4]]                 # the rig agrees with b's right arm -> "rig" wants b for it
    fused, src = pf.fuse(a, b, 450.0, rig, "rig", dict(torso="a", arms="a", legs="a", head="a", hands="a", face="a"))
    assert src["torso"] == "a"
    # a's shoulder -> b's elbow would be a long sideways bone: the arm comes from a (the torso's detector)
    assert src["arm_r"] == "a" and src.get("relinked_arm_r") is True
    assert np.allclose(fused["body"][[3, 4], :2], a["body"][[3, 4], :2])


def test_linked_compares_length_and_direction():
    j, c = np.array([0.0, 0.0]), np.array([0.0, 100.0])
    assert pf.linked(j, c, np.array([5.0, 0.0]), np.array([5.0, 110.0]))
    assert not pf.linked(j, c, np.array([0.0, 0.0]), np.array([0.0, 40.0]))       # too short in the detector
    assert not pf.linked(j, c, np.array([0.0, 0.0]), np.array([100.0, 0.0]))      # 90 degrees apart


OWN_A = dict(torso="a", arms="a", legs="a", head="a", hands="a", face="a")


def test_rig_takes_the_detector_whose_arm_is_closer():
    a, b = person(0.0), person(0.0)
    a["body"][[3, 4], 0] += 30                      # a's right arm is off by 30 px, b's is on the rig
    fused, src = pf.fuse(a, b, 450.0, RIG, "rig", OWN_A)
    assert src["arm_r"] == "b" and np.allclose(fused["body"][[3, 4], :2], RIG[[3, 4], :2])


def test_a_mirrored_detector_cannot_flip_a_correct_one():
    a, b = pf.swap_sides(person(0.0)), person(0.0)  # a is mirrored, b is right
    fused, src = pf.fuse(a, b, 450.0, RIG, "rig", OWN_A)
    assert src["relabelled"] is True
    assert np.allclose(fused["body"][[2, 3, 4, 5, 6, 7], :2], RIG[[2, 3, 4, 5, 6, 7], :2])


def test_a_limb_without_its_shoulder_is_left_out():
    a, b = person(0.0), person(0.0)
    a["body"][2] = np.nan
    b["body"][2] = np.nan                           # no right shoulder anywhere
    fused, src = pf.fuse(a, b, 450.0, RIG, "rig", OWN_A)
    assert np.isnan(fused["body"][[3, 4], 0]).all() and src.get("dropped_arm_r") is True


def test_no_face_where_the_rigs_head_faces_away():
    rig = RIG.copy()
    rig[[0, 14, 15], 2] = 0                         # a back view: nose and eyes hidden, ears shown
    fused, _ = pf.fuse(person(0.0), person(0.0), 450.0, rig, "rig", OWN_A)
    assert fused["face"] is None and np.isnan(fused["body"][[0, 14, 15], 0]).all()
    assert not np.isnan(fused["body"][[16, 17], 0]).any()      # the ears may stay


def test_the_face_comes_from_one_detector_not_an_average():
    a, b = person(0.0), person(0.0)
    b["face"][:, 0] += 4                            # both see the face, 4 px apart
    fused, src = pf.fuse(a, b, 450.0, RIG, "owner", OWN_A)
    assert src["face"] == "a" and np.allclose(fused["face"][:, 0], a["face"][:, 0])


def test_points_with_a_missing_coordinate_are_missing():
    out = pf.arr([[1.0, None, 0.9]] + [[2.0, 3.0, 0.9]] * 20, 21)
    assert np.isnan(out[0]).all() and not np.isnan(out[1]).any()


def test_a_hand_far_from_its_wrist_is_not_attached():
    a = person(0.0)
    a["L"] = a["L"].copy()
    a["L"][:, 1] -= 100                             # 100 px above the left wrist (22% of 450)
    out = pf.assign_hands(a, a["body"], 450.0)
    assert out["L"] is None and out["R"] is not None


def test_a_lone_wrist_without_its_elbow_is_not_drawn():
    a, b = person(0.0), person(0.0)
    a["body"][3] = np.nan
    b["body"][3] = np.nan                           # no right elbow anywhere, the wrists are there
    fused, src = pf.fuse(a, b, 450.0, RIG, "rig", OWN_A)
    assert np.isnan(fused["body"][4, 0]) and src.get("dropped_arm_r") is True


def test_an_all_invalid_body_does_not_crash(tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"box": "blender_iso", "frames": {"1": {"c": {"body": [[None, None, None]] * 18,
                                                                            "L": None, "R": None, "face": None}}}}))
    d = pf.load(f)
    assert np.isnan(d["1"]["c"]["body"]).all()


def test_a_front_view_without_a_nose_has_no_lone_eyes():
    a, b = person(0.0), person(0.0)
    b["body"][0, 0] += 60                           # the detectors disagree about the nose: it is removed
    fused, _ = pf.fuse(a, b, 450.0, RIG, "rig", OWN_A)
    assert np.isnan(fused["body"][[0, 14, 15, 16, 17], 0]).all()
