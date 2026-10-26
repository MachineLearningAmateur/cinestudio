"""The per-clip gauntlet numbers (cinestudio/clip_score.py): shared-frame agreement and picture numbers."""
from pathlib import Path

import cv2
import numpy as np

from cinestudio import clip_score


def _video(path: Path, frames: list) -> Path:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return path


def _scene(t: int, shift: int = 0, tint: int = 0, blur: float = 3) -> np.ndarray:
    """A textured picture that moves 2 px per frame (a normal frame change)."""
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, (128, 512, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), blur)
    f = np.roll(base, 2 * t + shift, axis=1)[:, :256]
    return np.clip(f.astype(int) + tint, 0, 255).astype(np.uint8)


def test_overlap_of_the_same_picture_is_small_and_of_another_picture_is_large(tmp_path):
    frames, overlap = 41, 9
    a = _video(tmp_path / "a.mp4", [_scene(t) for t in range(frames)])
    start = frames - overlap                                 # clip 2 starts on clip 1's frame `start`
    same = _video(tmp_path / "same.mp4", [_scene(start + t) for t in range(frames)])
    other = _video(tmp_path / "other.mp4", [_scene(start + t, shift=150, tint=40) for t in range(frames)])
    good = clip_score.overlap_diff(a, same, frames, overlap)
    bad = clip_score.overlap_diff(a, other, frames, overlap)
    assert good["x_median"] < 1.5
    assert bad["x_median"] > 4 * max(good["x_median"], 0.5)
    assert clip_score.overlap_diff(a, same, frames, 1) is None     # one shared frame: nothing to compare


def test_picture_numbers_see_a_soft_end(tmp_path):
    fs = [_scene(t, blur=0.7) for t in range(60)]
    fs[-25:] = [cv2.GaussianBlur(f, (0, 0), 4) for f in fs[-25:]]
    p = clip_score.picture(_video(tmp_path / "c.mp4", fs))
    assert p["frames"] == 60
    assert p["sharpness_last"] < p["sharpness_first"] / 2


def test_frame_png_and_last_frame_keep_one_signature(tmp_path):
    """Regression (Codex review 2026-10-21): a second frame_png with another argument order hid the first one and
    broke video.last_frame (every v2v clip after the first)."""
    import inspect

    from cinestudio import video
    assert list(inspect.signature(video.frame_png).parameters)[:3] == ["src", "dst", "index"]
    src = _video(tmp_path / "v.mp4", [_scene(t) for t in range(10)])
    video.last_frame(src, tmp_path / "last.png")
    video.frame_png(src, tmp_path / "f3.png", 3)
    assert (tmp_path / "last.png").is_file() and (tmp_path / "f3.png").is_file()
