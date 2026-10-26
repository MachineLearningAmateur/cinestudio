from cinestudio import facing as fc


def pts(nose=True, r_eye=True, l_eye=True, nose_x=100.0, neck_x=100.0):
    p = [[0.0, 0.0, 0]] * 18
    p = [list(x) for x in p]
    p[0] = [nose_x, 50.0, int(nose)]
    p[1] = [neck_x, 80.0, 1]
    p[14] = [95.0, 45.0, int(r_eye)]
    p[15] = [105.0, 45.0, int(l_eye)]
    return p


def test_facing_from_the_face_points():
    assert fc.facing(pts()) == "front"
    assert fc.facing(pts(l_eye=False, nose_x=90)) == "profile_left"
    assert fc.facing(pts(r_eye=False, nose_x=110)) == "profile_right"
    assert fc.facing(pts(nose=False, r_eye=False, l_eye=False)) == "back"


def test_clip_text_names_place_and_facing():
    a = [pts()] * 10                                      # in the centre, facing the camera
    b = [pts(nose=False, r_eye=False, l_eye=False)] * 10  # on the left, from behind
    t = fc.clip_text([a, b], [[400] * 10, [100] * 10], 768, 0, 9)
    assert t.startswith("At the start:") and "the character in the centre faces the camera" in t
    assert "the character on the left has its back to the camera" in t
    assert t.count("At the") == 1                         # the same at the middle and the end: said once


def test_the_head_angle_decides_over_the_eye_flags():
    both_eyes = pts()                                     # the flags say "front" (OpenPose keeps an eye to ~75 deg)
    assert fc.facing(both_eyes, [30.0, "left"]) == "front"
    assert fc.facing(both_eyes, [70.0, "left"]) == "profile_left"
    assert fc.facing(both_eyes, [100.0, "right"]) == "profile_right"
    assert fc.facing(both_eyes, [150.0, "right"]) == "back"


def test_clip_text_uses_the_head_angle():
    a = [pts()] * 10
    t = fc.clip_text([a], [[400] * 10], 768, 0, 9, [[[80.0, "right"]] * 10])
    assert "seen in profile, facing right" in t


def test_a_back_with_the_head_turned_to_the_camera():
    a = [pts()] * 10
    t = fc.clip_text([a], [[400] * 10], 768, 0, 9, [[[100.0, "left"]] * 10], [[[160.0, "left"]] * 10])
    assert "has its back to the camera and looks back over its shoulder toward the camera" in t
    t = fc.clip_text([a], [[400] * 10], 768, 0, 9, [[[30.0, "left"]] * 10], [[[90.0, "left"]] * 10])
    assert "seen in profile, facing left and turns its head to look at the camera" in t
    t = fc.clip_text([a], [[400] * 10], 768, 0, 9, [[[80.0, "left"]] * 10], [[[90.0, "left"]] * 10])
    assert "facing left." in t                            # the head follows the body: no note


def test_two_characters_in_the_same_place_get_front_and_back():
    a = [pts()] * 10
    t = fc.clip_text([a, a], [[380] * 10, [390] * 10], 768, 0, 9, depths=[[3.0] * 10, [5.0] * 10])
    assert "the character in the centre in front faces the camera" in t
    assert "the character in the centre at the back faces the camera" in t
