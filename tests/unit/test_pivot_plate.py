"""A backplate takes a PIVOT camera (one place, it only turns or zooms: bpy_jobs/plate_world.py --info reports `pivot`
and its cover camera) and still refuses a camera that changes its place (2026-10-27)."""
import json
from pathlib import Path

import pytest

from cinestudio import background, blender


def fake_info(monkeypatch, info):
    def run_job(script, blend, args, log, timeout=None):
        assert script == "plate_world.py" and args[0] == "--info"
        Path(args[1]).write_text(json.dumps(info), encoding="utf-8")
    monkeypatch.setattr(blender, "run_job", run_job)


def test_still_camera_is_taken(monkeypatch, tmp_path):
    shot = tmp_path / "shot.blend"
    shot.write_bytes(b"x")
    fake_info(monkeypatch, {"moves": False, "horizon_from_top": 0.4})
    assert background.camera_info(shot)["moves"] is False


def test_pivot_camera_is_taken_with_its_cover(monkeypatch, tmp_path):
    shot = tmp_path / "shot.blend"
    shot.write_bytes(b"x")
    cover = {"location": [0.0, -4.2, 1.3], "rotation_quat": [1, 0, 0, 0], "lens_mm": 31.2, "sensor_mm": 36.0}
    fake_info(monkeypatch, {"moves": True, "pivot": True, "cover": cover, "lens_mm": 31.2})
    cam = background.camera_info(shot)
    assert cam["pivot"] and cam["cover"] == cover


def test_camera_that_changes_its_place_is_refused(monkeypatch, tmp_path):
    shot = tmp_path / "shot.blend"
    shot.write_bytes(b"x")
    fake_info(monkeypatch, {"moves": True})
    with pytest.raises(background.BackgroundError, match="stays in one place"):
        background.camera_info(shot)
