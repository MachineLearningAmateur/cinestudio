"""The style library (2026-10-25): one set of look words per style, referred to by name from characters, templates,
backgrounds and runs."""
import json

import pytest

from cinestudio import background, character, runner, shots, styles


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    return tmp_path


def test_builtin_styles_and_a_user_copy_that_wins(home):
    names = {s["name"]: s for s in styles.list_styles()}
    assert {"3d_animated", "anime_cel"} <= set(names) and names["3d_animated"]["builtin"]
    assert styles.default_style() == "3d_animated"
    s = styles.save("3d_animated", {"lora": {"name": "x.safetensors", "strength": 0.4}})   # a user copy
    assert not s["builtin"] and s["version"] == 2 and s["lora"]["name"] == "x.safetensors"
    assert styles.text("3d_animated").startswith("stylized cinematic 3D animation")
    assert "evenly lit by soft neutral light" in styles.text("3d_animated")
    styles.delete("3d_animated")                                      # back to the built-in
    assert styles.load("3d_animated")["builtin"]
    with pytest.raises(styles.StyleError, match="built-in style"):
        styles.delete("anime_cel")


def test_new_style_rules(home):
    with pytest.raises(styles.StyleError, match="needs its words"):
        styles.save("clay", {"label": "Clay"}, new=True)
    styles.save("clay", {"label": "Clay", "words": "claymation, stop-motion"}, new=True)
    with pytest.raises(styles.StyleError, match="exists already"):
        styles.save("clay", {"words": "x"}, new=True)
    with pytest.raises(styles.StyleError, match="lora strength"):
        styles.save("clay", {"lora": {"name": "a", "strength": 5}})
    assert styles.set_default("clay") == {"default_style": "clay"} and styles.default_style() == "clay"


def _char(home, name="hero", style_ref=None):
    d = character.library() / name
    d.mkdir(parents=True)
    rec = {"name": name, "status": "ready", "source": "text", "files": {}, "style": "old words"}
    if style_ref:
        rec["style_ref"] = style_ref
    (d / "character.json").write_text(json.dumps(rec), encoding="utf-8")


def test_a_character_takes_a_styles_words_and_version(home):
    _char(home)
    c = character.set_style("hero", "anime_cel")
    assert c["style"] == styles.text("anime_cel", light=False) and c["style_ref"] == "anime_cel" and c["style_version"] == 1
    assert not styles.is_stale(c)
    styles.save("anime_cel", {"light": "soft rim light"})             # the style changes: the asset is older now
    assert styles.is_stale(character.load("hero"))
    assert styles.used_by()["anime_cel"]["characters"] == ["hero"]
    with pytest.raises(character.CharacterError, match="no style"):
        character.set_style("hero", "nope")


def test_background_style_words_and_mismatch_warnings(home):
    assert background.style_words("", "3d_animated") == styles.text("3d_animated")
    assert background.style_words("own words", "3d_animated") == "own words"
    _char(home, "a", "3d_animated")
    _char(home, "b", "anime_cel")
    w = runner._style_mismatch("3d_animated", {"character_name": "a", "character2": {"name": "b"}}, None)
    assert w == ["character b uses the style 'anime_cel', the run uses '3d_animated'"]
    assert runner._style_mismatch(None, {"character_name": "a"}, None) == []
