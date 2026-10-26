"""Character skins (2026-10-25): one character in more than one style; each skin is its own record with skin_of."""
import json

import cv2
import numpy as np
import pytest

from cinestudio import character, jobs, styles


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    return tmp_path


def _char(name, style_ref=None, skin_of=None, front=False, **extra):
    d = character.library() / name
    d.mkdir(parents=True)
    rec = {"name": name, "status": "ready", "source": "text", "files": {}, "description": "a fox-masked wanderer.",
           "style": "anime cel-shaded illustration, hand-drawn linework", "mannequin": {"scale": 0.9},
           "proxy": {"mask": {"shape": "round"}}, **extra}
    if front:
        cv2.imwrite(str(d / "front.png"), np.zeros((64, 48, 3), np.uint8))
        rec["files"]["front"] = str(d / "front.png")
    if style_ref:
        rec["style_ref"] = style_ref
    if skin_of:
        rec["skin_of"] = skin_of
    (d / "character.json").write_text(json.dumps(rec), encoding="utf-8")


def test_link_list_and_unlink(home):
    _char("kitsune", "anime_cel")
    _char("kitsune-3d", "3d_animated")
    _char("other")
    character.link_skin("kitsune-3d", "kitsune")
    assert [r["name"] for r in character.skins("kitsune")] == ["kitsune", "kitsune-3d"]
    assert [r["name"] for r in character.skins("kitsune-3d")] == ["kitsune", "kitsune-3d"]   # from any skin
    assert [r["name"] for r in character.skins("other")] == ["other"]
    with pytest.raises(character.CharacterError, match="itself a skin"):
        character.link_skin("other", "kitsune-3d")                    # link to the main record, not to a skin
    with pytest.raises(character.CharacterError, match="skins of its own"):
        character.link_skin("kitsune", "other")                       # a main record with skins cannot become one
    with pytest.raises(character.CharacterError, match="itself"):
        character.link_skin("other", "other")
    character.unlink_skin("kitsune-3d")
    assert [r["name"] for r in character.skins("kitsune")] == ["kitsune"]


def test_delete_and_rename_respect_the_family(home):
    _char("kitsune", "anime_cel")
    _char("kitsune-3d", "3d_animated", skin_of="kitsune")
    with pytest.raises(character.CharacterError, match="other skin"):
        character.delete_character("kitsune")
    character.rename("kitsune", "fox")
    assert character.load("kitsune-3d")["skin_of"] == "fox"
    character.delete_character("kitsune-3d")                          # a skin can go; then the main record can go
    character.delete_character("fox")
    assert character.list_characters() == []


def test_the_conversion_negative_drops_shared_words():
    src = {"style": "cinematic anime cel-shaded illustration, hand-drawn linework, painted shading"}
    neg = character.skin_negative(src, {"words": "stylized 3D animation, soft painted materials", "negative_image": "toy"})
    assert "hand-drawn linework" in neg and "cinematic anime cel-shaded illustration" in neg
    assert "painted shading" not in neg                               # shares "painted" with the target look
    assert neg.startswith(character.SKIN_NEGATIVE) and neg.endswith("hand-drawn linework") and ", toy," in neg
    ins = character.skin_instruction({"description": "a fox-masked wanderer."}, {"words": "stylized 3D."}, True)
    assert "rendering style of Picture 2: stylized 3D." in ins and "stays exactly the same: a fox-masked wanderer, and" in ins
    assert "this rendering style" in character.skin_instruction({}, {"words": "x"}, False)


def test_new_skin_copies_the_character_and_redraws_the_front(home, monkeypatch):
    _char("kitsune", "anime_cel", front=True, display_name="Kitsune")
    calls = []

    def fake_edit(name, instruction, target="front", reference=None, mode="quality", seed=None, settings=None, **kw):
        calls.append(("edit", name, target, reference, settings["negative"]))
        return character.load(name)

    def fake_views(name, mode="quality", seed=None, pose=None, **kw):
        calls.append(("views", name, pose))
        c = character.load(name)
        c["status"] = "views"
        character.save(c)
        return c
    monkeypatch.setattr(character, "edit_image", fake_edit)
    monkeypatch.setattr(character, "make_views", fake_views)
    r = character.new_skin("kitsune", "3d_animated", seed=5)
    new = r["skin"]
    assert new == "kitsune-3d_animated"
    c = character.load(new)
    assert c["skin_of"] == "kitsune" and c["copied_from"] == "kitsune" and c["style_ref"] == "3d_animated"
    assert c["mannequin"] == {"scale": 0.9} and c["proxy"] == {"mask": {"shape": "round"}} and c["display_name"] == "Kitsune"
    assert c["style"] == styles.text("3d_animated", light=False)
    assert (character.library() / new / "front.png").is_file()
    assert calls[0][:4] == ("edit", new, "front", None) and calls[1] == ("views", new, None)
    with pytest.raises(character.CharacterError, match="has a skin in"):
        character.new_skin(new, "3d_animated")                         # one skin per style in a family
    assert "new_skin" in jobs.LONG and jobs.ACTIONS["new_skin"] is character.new_skin


def test_the_character_view_lists_the_new_skins_being_made(home):
    """The skin cards show a card per new skin being made: the family's queued or running new_skin jobs."""
    import os
    import time
    from cinestudio.webapi.common import _char_view
    _char("kitsune", "anime_cel")
    _char("kitsune-3d", "3d_animated", skin_of="kitsune")
    _char("other")
    me = {"pid": os.getpid(), "via": "app"}
    for jid, action, name, status, style in [("j20261025-100000-aaaa", "new_skin", "kitsune-3d", "running", "clay"),
                                             ("j20261025-100001-bbbb", "new_skin", "kitsune", "done", "noir"),
                                             ("j20261025-100002-cccc", "new_skin", "other", "queued", "noir"),
                                             ("j20261025-100003-dddd", "edit", "kitsune", "running", None)]:
        jobs._write({"id": jid, "action": action, "character": name, "status": status, "created": time.time(),
                     "args": {"style": style} if style else {}, "events": [], "owner": me})
    v = _char_view(character.load("kitsune"))
    assert v["skin_jobs"] == [{"id": "j20261025-100000-aaaa", "status": "running", "character": "kitsune-3d", "style": "clay"}]
    assert _char_view(character.load("other"))["skin_jobs"][0]["style"] == "noir"
