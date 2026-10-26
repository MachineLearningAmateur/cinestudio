"""The figure mask: DWPose seed points, merging the user's clicks, the SAM 2 graph and the mask stats (no GPU)."""
import cv2
import numpy as np

from cinestudio import figure_mask as fm


def kp(frames=90):
    body = [[100.0 + j, 50.0 + 10 * j, 1] for j in range(18)]
    return {"frames": [body for _ in range(frames)]}


def test_seed_points_on_keyframes_from_shown_joints():
    pts = fm.seed_points(kp(90), 90, every=40)
    assert sorted({p["frame"] for p in pts}) == [0, 40, 80]
    assert len([p for p in pts if p["frame"] == 0]) == len(fm.SEED_JOINTS) and all(p["positive"] for p in pts)
    k = kp(10)
    k["frames"][0] = [[0.0, 0.0, 0]] * 18                      # the figure is not seen on frame 0: no seed there
    assert fm.seed_points(k, 10, every=40) == []


def test_clicks_join_the_seed_and_negative_only_frames_are_skipped():
    seed = [{"frame": 0, "x": 1, "y": 2, "positive": True}]
    clicks = [{"frame": 0, "x": 5, "y": 6, "positive": False}, {"frame": 30, "x": 7, "y": 8, "positive": False},
              {"frame": 60, "x": 9, "y": 9, "positive": True}]
    got = fm.merge_points(seed, clicks)
    assert list(got) == [0, 60] and got[0] == {"pos": [(1.0, 2.0)], "neg": [(5.0, 6.0)]}


def test_graph_chains_one_add_points_node_per_frame():
    g = fm.graph({0: {"pos": [(1, 2)], "neg": []}, 40: {"pos": [(3, 4)], "neg": [(5, 6)]}}, "cinestudio/mask_x")
    assert g["pts0"]["inputs"]["image"] == ["frames", 0] and "prev_inference_state" not in g["pts0"]["inputs"]
    assert g["pts1"]["inputs"]["prev_inference_state"] == ["pts0", 1] and g["pts1"]["inputs"]["frame_index"] == 40
    assert "coordinates_negative" in g["pts1"]["inputs"] and g["seg"]["inputs"]["inference_state"] == ["pts1", 1]
    assert g["save"]["inputs"]["filename_prefix"] == "cinestudio/mask_x"


def test_mask_stats_flag_jumps_and_empty_frames(tmp_path):
    p = tmp_path / "m.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 32))
    for size in (10, 10, 20, 0):
        f = np.zeros((32, 64, 3), np.uint8)
        f[:size, :size] = 255
        w.write(f)
    w.release()
    s = fm.mask_stats(p)
    assert s["frames"] == 4 and s["boxes"][3] is None
    assert [x["frame"] for x in s["flags"]] == [2, 3]


def _video_template(home, name="tpl"):
    """A tiny video template (24 frames, 64x32) stored like shots.add_video does."""
    import subprocess
    from cinestudio import shots, video
    src = home / "src.mp4"
    subprocess.run([video.ffmpeg(), "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x32:r=24:d=1",
                    "-frames:v", "24", "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True,
                   stdin=subprocess.DEVNULL)
    shots.add_video(name, src, "a stand-in")
    return name


def test_start_saves_clicks_reserves_the_queue_and_spawns(tmp_path, monkeypatch):
    import pytest
    from cinestudio import workqueue
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    name = _video_template(tmp_path)
    spawned = []
    r = fm.start(name, [{"frame": 3, "x": 10, "y": 12, "positive": False}],
                 spawner=lambda cmd, log, cwd: (spawned.append(cmd) or __import__("os").getpid(), True))   # a live pid
    assert r["status"] == "queued" and spawned[0][-2:] == ["work-mask", name]
    assert workqueue.find("template", f"mask-{name}") is not None              # the place in line, before the worker
    import json
    assert json.loads((fm.mask_dir(name) / "clicks.json").read_text())[0] == {"frame": 3, "x": 10.0, "y": 12.0,
                                                                              "positive": False}
    with pytest.raises(fm.MaskError, match="already"):                         # one tracking job per template
        fm.start(name, [], spawner=lambda *a: (1, True))
    with pytest.raises(fm.MaskError, match="frame, x and y"):
        fm.clean_clicks([{"frame": "a", "x": 1, "y": 1}])
    from cinestudio import shots
    with pytest.raises(shots.ShotError, match="no template"):
        fm.start("missing_template", [], spawner=lambda *a: (1, True))


def test_work_writes_the_status_and_the_summary(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    name = _video_template(tmp_path)
    d = fm.mask_dir(name)
    d.mkdir(parents=True)
    (d / "clicks.json").write_text(json.dumps([{"frame": 0, "x": 5, "y": 5, "positive": True}]))

    def fake_track(n, clicks, on_event=None):
        on_event({"event": "progress", "text": "SAM 2 follows the figure"})
        assert fm.status(n)["status"] == "running" and clicks[0]["x"] == 5
        rec = {"template": n, "frames": 24, "flags": [{"frame": 7, "why": "no mask"}], "area": [0.1] * 24,
               "points": [{"frame": 0, "x": 1, "y": 2, "positive": True, "seed": True}], "files": {}}
        (d / "mask.json").write_text(json.dumps(rec))
        return rec
    monkeypatch.setattr(fm, "track", fake_track)
    fm.work(name)
    s = fm.summary(name)
    assert s["status"]["status"] == "done" and s["status"]["flags"] == 1
    assert s["video"]["width"] == 64 and s["video"]["frames"] == 24 and s["mask"]["flags"][0]["frame"] == 7
    assert s["clicks"][0]["frame"] == 0

    def broken(n, clicks, on_event=None):
        raise fm.MaskError("SAM 2 failed")
    monkeypatch.setattr(fm, "track", broken)
    import pytest
    with pytest.raises(fm.MaskError):
        fm.work(name)
    assert fm.status(name)["status"] == "failed" and "SAM 2" in fm.status(name)["error"]


def test_queue_cancel_reaches_a_mask_job(tmp_path, monkeypatch):
    from cinestudio import workqueue
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    r = workqueue.cancel("template", "mask-snow")
    assert r["mask_cancel_requested"] and (fm.mask_dir("snow") / "cancel").is_file()
