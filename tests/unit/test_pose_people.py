from collections import namedtuple

import numpy as np

from cinestudio import pose_people as pp

K = namedtuple("K", "x y score")
RB = np.array([[100, 30 + 25 * j, 1] for j in range(18)], float)     # a rig 425 px tall
RB[4, :2], RB[7, :2] = (60, 300), (140, 300)                        # right wrist left of image, left wrist right


def hand(x, y):
    return [K(x + i, y + i, 0.9) for i in range(21)]


def test_two_hands_that_prefer_one_wrist_are_both_kept():
    rb = RB.copy()
    rb[7, :2], rb[4, :2] = (140, 300), (120, 300)                      # crossed arms: both wrists on one side
    body = [K(*rb[i, :2], 0.9) for i in range(18)]
    stats = {}
    out = pp.assign_hands([hand(141, 300), hand(150, 300)], body, rb, stats)   # both are nearest the left wrist
    assert out["L"] is not None and out["R"] is not None and not stats.get("hands_dropped")
    assert {out["L"][0].x, out["R"][0].x} == {141, 150}               # the old greedy rule lost one of them


def test_swapped_hand_labels_go_to_the_right_wrists():
    body = [K(*RB[i, :2], 0.9) for i in range(18)]
    out = pp.assign_hands([hand(60, 300), hand(140, 300)], body, RB, {})     # given as [L, R] but the other way round
    assert out["R"][0].x == 60 and out["L"][0].x == 140


def test_a_hand_far_from_its_body_wrist_is_dropped():
    body = [K(*RB[i, :2], 0.9) for i in range(18)]
    stats = {}
    out = pp.assign_hands([hand(60, 150)], body, RB, stats)                  # 150 px above the wrist (35% of h)
    assert out["L"] is None and out["R"] is None and stats["hands_dropped"] == 1


def test_non_finite_points_count_as_missing():
    assert pp._xy(K(1.0, float("nan"), 0.9)) is None and pp._xy(None) is None
    assert pp._xy(K(1.0, 2.0, 0.9)).tolist() == [1.0, 2.0]
