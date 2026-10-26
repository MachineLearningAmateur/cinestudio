"""The short Target Description (target_text "short", the default since 2026-10-26: prompt A/B "C"): identity by the
pictures, no second copy of the costume, no place sentence; the key marks come back when the clip text names none."""
from cinestudio import runner

SHEET = ("**Left Panel (Character):** a close-up of the character's face. **Panels 2-5 (Character):** the same "
         "character in full body shown from multiple angles (front, right side, back, left side) on black: a slim "
         "masked figure with a smooth white oval mask that covers the whole face; short dark plum-purple hair. A "
         "fitted black long-sleeved bodysuit from the neck to the ankles. Plain dark charcoal shoes.")


def _m(video, target_text=None):
    m = {"prompt": {"character": SHEET, "video": video, "style": "stylized 3D anime noir"}, "prompt_format": "ing25",
         "sound_text": "", "background": "plate", "background_text": "an indoor school gymnasium"}
    if target_text is not None:
        m["target_text"] = target_text
    return m


def _target(m):
    return runner.video_prompt(m).split("### Target Description", 1)[1]


def test_the_default_is_short():
    assert runner.DEFAULTS["target_text"] == "short" and "short" in runner.TARGET_TEXTS


def test_short_target_drops_the_costume_and_the_place_when_the_text_names_the_marks():
    t = _target(_m("A wide shot, front-facing: a masked dancer with a white oval mask in a dim gym.", "short"))
    assert "The character from the reference sheet. A wide shot" in t
    assert "bodysuit" not in t and "The place:" not in t
    assert "### Reference Sheet Description" in runner.video_prompt(_m("x", "short"))   # the sheet text stays


def test_short_target_adds_the_key_marks_to_a_text_with_only_actions():
    t = _target(_m("The dancer slides toward the right of the picture, then kicks.", "short"))
    assert "The character from the reference sheet, a slim masked figure with a smooth white oval mask" in t
    assert "bodysuit" not in t and "The place:" not in t


def test_old_manifests_and_full_keep_the_whole_target():
    for m in (_m("The dancer kicks."), _m("The dancer kicks.", "full")):
        t = _target(m)
        assert "bodysuit from the neck to the ankles" in t and "The place: an indoor school gymnasium." in t


def test_names_marks_counts_whole_words():
    d = "a slim masked figure with a smooth white oval mask that covers the whole face"
    assert runner.names_marks("a masked dancer with a white oval mask", d)
    assert not runner.names_marks("the dancer unmasked herself", d)
    assert runner.names_marks("anything", "")
