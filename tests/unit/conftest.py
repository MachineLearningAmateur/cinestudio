"""Shared test setup: tests never see the built-in templates and backgrounds of the repository (templates/library,
backgrounds/library), unless a test points shots.BUILTIN or background.BUILTIN somewhere itself."""
import pytest


@pytest.fixture(autouse=True)
def _no_builtin_templates(tmp_path_factory, monkeypatch):
    from cinestudio import shots
    monkeypatch.setattr(shots, "BUILTIN", tmp_path_factory.mktemp("builtin_empty"))


@pytest.fixture(autouse=True)
def _no_builtin_backgrounds(tmp_path_factory, monkeypatch):
    from cinestudio import background
    monkeypatch.setattr(background, "BUILTIN", tmp_path_factory.mktemp("builtin_bg_empty"))


@pytest.fixture(autouse=True)
def _no_user_workflow_copies(tmp_path_factory, monkeypatch):
    """Tests never load the user's own graph copies (cinestudio/user_workflows.py): their own empty folder."""
    monkeypatch.setenv("CINESTUDIO_USER_WORKFLOWS", str(tmp_path_factory.mktemp("user_workflows")))


@pytest.fixture(autouse=True)
def _no_blender_template_check(monkeypatch):
    """The template check (cinestudio/shot_check.py) never starts Blender in unit tests: its scan is refused (the
    builds go on without a check summary); tests that check rules feed facts or a `measurer` themselves."""
    from cinestudio import shot_check

    def refuse(blend):
        raise shot_check.CheckError("no Blender in unit tests")
    monkeypatch.setattr(shot_check, "measure", refuse)
