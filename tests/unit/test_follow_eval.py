import numpy as np

from cinestudio import follow_eval as fe

# a standing figure in OpenPose-18 order (not on a line): x right, y down, box 400 px high
RIG = np.array([
    [200, 60], [200, 110], [160, 112], [140, 180], [130, 245], [240, 112], [262, 180], [272, 245],
    [175, 240], [172, 330], [170, 420], [225, 240], [228, 330], [230, 420], [192, 52], [208, 52], [180, 56], [220, 56],
], float)
BOX = [100, 40, 300, 440]


def row(det, rig=RIG, name="a", frame=0):
    return {"frame": frame, "name": name, "box": BOX, "rig": [[x, y, 1] for x, y in rig],
            "det": [None if p is None else [float(p[0]), float(p[1]), 0.9] for p in det]}


def test_the_same_pose_scores_one():
    r = fe.evaluate([row(RIG)])["all"]
    assert r["bone"] == 1.0 and r["bone_coverage"] == 1.0 and r["pck20"] == 1.0 and r["lr_disagree"] == 0


def test_other_proportions_moving_and_zoom_keep_the_bone_direction():
    other = RIG.copy()
    other[[9, 10, 12, 13], 1] = RIG[[8, 8, 11, 11], 1] + (RIG[[9, 10, 12, 13], 1] - RIG[[8, 8, 11, 11], 1]) * 0.6   # short legs
    moved = other * 1.3 + np.array([40.0, -20.0])                                                                   # zoom + move
    r = fe.evaluate([row(moved)])["all"]
    assert r["bone"] > 0.99                      # every bone points the same way
    assert r["pck20"] < 0.6                      # positions do not match (proportions, framing)


def test_a_mirrored_figure_loses_the_bone_score_and_shows_left_right_disagreement():
    mirrored = RIG.copy()
    mirrored[:, 0] = 400 - mirrored[:, 0]        # mirrored about x = 200 (the figure seen from the back)
    r = fe.evaluate([row(mirrored)])["all"]
    # the shoulder and hip lines reverse; limbs that hang almost straight down keep their direction - so a mirror
    # shows in the torso part and in lr_disagree, not fully in the total
    assert r["bone_parts"]["torso"] == 0.0 and r["lr_disagree"] == 1 and r["bone"] < 0.75


def test_a_missing_arm_counts_against_the_score():
    det = [None if j in (3, 4) else p for j, p in enumerate(RIG)]
    r = fe.evaluate([row(det)])["all"]
    assert r["bone_coverage"] < 1.0 and r["bone"] < 1.0 and r["bone_parts"]["arms"] == 0.5


def test_a_bone_pointing_at_the_camera_is_not_expected():
    rig = RIG.copy()
    rig[4] = rig[3] + [1.0, 1.0]                 # the forearm points at the camera: 1.4 px long
    det = RIG.copy()                             # the video draws it long and downwards: not punished
    parts = fe.bone_scores(row(det, rig))
    assert sum(1 for p, s, _ in parts if p == "arms" and s is None) == 1


def test_nothing_detected_gives_ap_zero_and_unknown_left_right():
    r = fe.evaluate([row([None] * 18)])["all"]
    assert r["ap"] == 0.0 and r["bone"] == 0.0 and r["lr_unknown"] == 1 and r["facing_unknown"] == 1


def test_the_torso_fit_does_not_mirror():
    mirrored = RIG.copy()
    mirrored[:, 0] = 400 - mirrored[:, 0]
    fit = fe.st_fit(row(mirrored))
    assert fe.pck(row(mirrored), 0.2, det=fit) < 0.8      # a fit with a reflection would give 1.0


def test_fail_spans():
    scores = {f: (0.2 if 10 <= f <= 20 else 0.9) for f in range(40)}
    assert fe.fail_spans(scores) == [(10, 20)]
    assert fe.fail_spans({f: 0.2 for f in range(3)}) == []
