"""Built-in characters (characters/library, 2026-10-26): found when the user has no character of that ID, paths
resolved, read-only, listed, hidden by a user character of the same ID, their IDs not reused."""
import json
from pathlib import Path

import pytest

from cinestudio import character


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    b = tmp_path / "builtin"
    d = b / "hero"
    (d / "views").mkdir(parents=True)
    for f in ("front.png", "sheet.png", "views/back.png"):
        (d / f).write_bytes(b"png")
    (d / "character.json").write_text(json.dumps({
        "name": "hero", "status": "ready", "description": "a hero", "mannequin": {},
        "files": {"front": "./front.png", "sheet": "./sheet.png", "back": "./views/back.png"},
        "proxy": {"mask": {"eyes": True}}}), encoding="utf-8")
    monkeypatch.setattr(character, "BUILTIN", b)
    return b


def test_a_builtin_loads_with_absolute_paths(lib):
    c = character.load("hero")
    assert c["builtin"] is True
    assert Path(c["files"]["front"]) == lib / "hero" / "front.png" and Path(c["files"]["back"]).is_file()
    assert c["proxy"] == {"mask": {"eyes": True}}            # the kit ships with it
    assert character.is_builtin("hero") and not character.is_builtin("nobody")


def test_a_builtin_is_read_only(lib):
    for call in (lambda: character.set_proxy("hero", {}), lambda: character.undo_edit("hero"),
                 lambda: character.delete_character("hero"), lambda: character.set_display_name("hero", "X"),
                 lambda: character.save(character.load("hero"))):
        with pytest.raises(character.CharacterError, match="built-in"):
            call()
    assert json.loads((lib / "hero" / "character.json").read_text())["files"]["front"] == "./front.png"


def test_listed_and_hidden_by_a_user_character(lib):
    rows = {r["name"]: r for r in character.list_characters()}
    assert rows["hero"]["builtin"] is True and rows["hero"]["thumb"].endswith("front.png")
    mine = character.library() / "hero"
    mine.mkdir(parents=True)
    (mine / "character.json").write_text(json.dumps({"name": "hero", "status": "draft"}), encoding="utf-8")
    assert not character.is_builtin("hero") and character.load("hero").get("builtin") is None
    assert [r["builtin"] for r in character.list_characters() if r["name"] == "hero"] == [False]


def test_new_ids_skip_builtin_ids(lib):
    character.library().mkdir(parents=True, exist_ok=True)
    assert character.new_id("Hero") == "hero-2"
