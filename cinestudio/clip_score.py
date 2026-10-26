"""The per-clip gauntlet numbers of a long run: one clip (or one retake) of a layout run, scored on its own.

    python -m cinestudio run score RUN --clip K [--take N]

motion   cinestudio/motion_follow.py with ComfyUI's Python (DWPose in the character's Blender box, the rig's frames
         from the clip's start in the shot) + cinestudio.follow_eval: bone (direction score vs the rig), torso / arms /
         legs, PCK@0.2, mirrored rows, facing errors. Its judge is DWPose: a mannequin-like body can score well;
         judge the character by eye (or a blind critic), never by this alone.
picture  saturation (HSV S mean), centre sharpness (setup/join_check.py's measure: Laplacian variance of the middle
         half at 768 px; median over the clip, its first and its last 25 frames), median frame change and the largest
         jump / median (flicker, smear).
overlap  for clip k > 1: clip k's first `overlap` frames and clip k-1's last ones show the SAME shot frames; the join
         cross-fades them. Their mean difference / the clip's median frame change: about 1-2 = the same picture;
         5+ = two different pictures (another scene or pose), which the cross-fade turns into a morph
         (R2b 2026-10-21: 7.6, a wall in clip 2 where clip 1 had water tanks).
critic   with `critic_dir`: the material for a blind critic (docs/gauntlet_critic.md): strip_SOURCE.jpg (the Blender
         layout of the clip span), strip_<TAG>.jpg and big_<TAG>_<frame>.jpg (the clip), and rig.txt (the rig's
         body and head angles at the strip frames: a blank mannequin head cannot show where it looks).
The result is stored as a side file run/scores/clip<K>[_take<N>].json (never in state.json: a worker's saves would race
with it; Codex review 2026-10-22), with the hashes of every video it compared ("deps"), and returned.
runner.stored_score() gives it back only while those videos are unchanged.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import runner, video

ROOT = Path(__file__).resolve().parent.parent


def _frames(path: Path) -> list:
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def _sharp(f) -> float:
    """setup/join_check.py's sharpness: Laplacian variance of the middle half, at 768 px wide."""
    import cv2
    g = cv2.cvtColor(cv2.resize(f, (768, int(768 * f.shape[0] / f.shape[1]))), cv2.COLOR_BGR2GRAY)
    w = g.shape[1]
    return float(cv2.Laplacian(g[:, w // 4: 3 * w // 4], cv2.CV_64F).var())


def _changes(fs: list) -> np.ndarray:
    return np.array([np.abs(fs[i].astype(np.int16) - fs[i - 1].astype(np.int16)).mean() for i in range(1, len(fs))])


def picture(path: Path, edge: int = 25) -> dict:
    import cv2
    fs = _frames(path)
    if not fs:
        return {"error": f"no frames in {path}"}
    sat = float(np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 1].mean() for f in fs[::4]]))
    sh = [_sharp(f) for f in fs]
    d = _changes(fs)
    med = float(np.median(d)) if len(d) else 0.0
    return {"frames": len(fs), "saturation": round(sat, 1), "sharpness": round(float(np.median(sh)), 1),
            "sharpness_first": round(float(np.median(sh[:edge])), 1),
            "sharpness_last": round(float(np.median(sh[-edge:])), 1),
            "change_median": round(med, 2), "max_jump_x_median": round(float(d.max() / max(med, 1e-6)), 2) if len(d) else 0}


def overlap_diff(prev_clip: Path, clip: Path, frames: int, overlap: int) -> dict | None:
    """How far the two versions of the shared frames differ: clip k-1's frames frames-overlap.. and clip k's frames
    0..overlap-1 (mean absolute difference per frame), as a multiple of clip k's median frame change."""
    a, b = _frames(prev_clip), _frames(clip)
    if not a or not b or overlap < 2:
        return None
    s0 = frames - overlap
    n = min(overlap, len(b), len(a) - s0)
    if n < 1:
        return None
    diff = [float(np.abs(a[s0 + j].astype(np.int16) - b[j].astype(np.int16)).mean()) for j in range(n)]
    med = max(float(np.median(_changes(b))), 1e-6)
    return {"mean": round(float(np.mean(diff)), 2), "x_median": round(float(np.mean(diff)) / med, 2),
            "first_x_median": round(diff[0] / med, 2), "last_x_median": round(diff[-1] / med, 2)}


def strip_frames(n: int, count: int = 7) -> list[int]:
    """`count` frames spread over the WHOLE clip (0, ..., n-1): a 193-frame clip is seen to its end (Codex 10-21)."""
    return sorted({round(i * (n - 1) / (count - 1)) for i in range(count)}) if n > 1 else [0]


def _strip(frames: list, idx, height: int = 220):
    import cv2
    tiles = []
    for i in idx:
        f = frames[min(i, len(frames) - 1)]
        t = cv2.resize(f, (int(f.shape[1] * height / f.shape[0]), height))
        cv2.putText(t, str(i), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(t)
    return np.hstack(tiles)


def critic_pack(run_id: str, clip: int, clip_file: Path, start: int, out_dir: Path, tag: str) -> dict:
    """Strips, full-size frames and the rig angles of one clip for a blind critic (anonymous TAG)."""
    import cv2
    run = runner.Run(run_id)
    run.load()
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = _frames(clip_file)
    idx = strip_frames(len(fs))
    cv2.imwrite(str(out_dir / f"strip_{tag}.jpg"), _strip(fs, idx))
    for i in (idx[0], idx[len(idx) // 2], idx[-1]):
        if i < len(fs):
            cv2.imwrite(str(out_dir / f"big_{tag}_{i:03d}.jpg"), fs[i])
    src = run.work / ("source.mp4" if int(clip) == 1 else f"source_{int(clip)}.mp4")
    if src.is_file() and not (out_dir / "strip_SOURCE.jpg").is_file():
        cv2.imwrite(str(out_dir / "strip_SOURCE.jpg"), _strip(_frames(src), idx))
    lines = []
    try:
        kp = json.loads((run.pass_dir / "keypoints.json").read_text(encoding="utf-8"))
        for i in idx:
            b = (kp.get("body_yaw") or [])[start + i] if start + i < len(kp.get("body_yaw") or []) else None
            h = (kp.get("head_yaw") or [])[start + i] if start + i < len(kp.get("head_yaw") or []) else None
            if b:
                lines.append(f"clip frame {i} (shot frame {start + i}): body {b[0]:.0f} deg toward the picture's {b[1]}"
                             + (f", head {h[0]:.0f} deg toward the picture's {h[1]}" if h else ""))
    except (OSError, ValueError, IndexError, TypeError):
        pass
    (out_dir / "rig.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"dir": str(out_dir), "tag": tag, "strip": str(out_dir / f"strip_{tag}.jpg"), "rig": lines}


def motion(run_id: str, clip_file: Path, start: int) -> dict:
    from . import comfy
    run = runner.run_dir(run_id)
    py = ((comfy.load_config() or comfy.detect()) or {}).get("python")
    if not py:
        return {"error": "ComfyUI's Python is unknown (comfy status)"}
    dets = clip_file.with_suffix(".follow_dets.json")
    r = subprocess.run([py, "-B", str(ROOT / "cinestudio" / "motion_follow.py"), str(run), "--video", str(clip_file),
                        "--start", str(start), "--out", str(dets)], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        return {"error": "motion follow failed", "tail": r.stderr[-600:]}
    r = subprocess.run([sys.executable, "-m", "cinestudio.follow_eval", str(dets)], capture_output=True, text=True,
                       cwd=ROOT)
    if r.returncode != 0:
        return {"error": "follow_eval failed", "tail": r.stderr[-600:]}
    ev = json.loads(r.stdout.strip().splitlines()[-1])
    al = (ev.get("vs_rig") or {}).get("all") or {}
    parts = al.get("bone_parts") or {}
    return {"bone": al.get("bone"), "torso": parts.get("torso"), "arms": parts.get("arms"), "legs": parts.get("legs"),
            "pck20": al.get("pck20"), "mirrored": al.get("lr_disagree"), "facing_errors": al.get("facing"),
            "rows": al.get("rows")}


def score(run_id: str, clip: int, take: int | None = None, with_motion: bool = True,
          critic_dir: str | Path | None = None, tag: str | None = None) -> dict:
    d = runner.run_dir(run_id)
    st = json.loads((d / "state.json").read_text(encoding="utf-8"))
    fmt = st["steps"]["controls"]["outputs"]
    parts = st["steps"]["clip"].get("parts") or {}
    key = str(int(clip))
    if take is None:
        rec = parts.get(key) or {}
        path = Path((rec.get("outputs") or {}).get("clip") or "")
    else:
        rec = next((t for t in (st.get("takes") or {}).get(key, []) if t["n"] == int(take)), {})
        path = Path((rec.get("outputs") or {}).get("clip") or "")
    if not path.is_file():
        raise runner.RunError(f"clip {clip}" + (f" take {take}" if take is not None else "") + " has no video")
    ov = int(fmt.get("overlap", 1))
    start = video.clip_start(int(clip), fmt["frames"], ov)
    def deps_now():
        files = {"clip": path}
        for name, k in (("prev", int(clip) - 1), ("next", int(clip) + 1), ("clip1", 1)):
            f = Path(((parts.get(str(k)) or {}).get("outputs") or {}).get("clip") or "")
            if k >= 1 and f.is_file() and (name != "clip1" or int(clip) > 1):
                files[name] = f
        return {k: runner.file_sha(f) for k, f in files.items()}
    before = deps_now()                                  # hashed BEFORE and AFTER the measurement (Codex 10-22)
    pic = picture(path)
    if "error" in pic:                                    # an empty or broken clip: stop with a clear error (Codex 10-22)
        raise runner.RunError(f"clip {clip}" + (f" take {take}" if take is not None else "") + f": {pic['error']}")
    out = {"run": run_id, "clip": int(clip), "take": take, "video": str(path), "start_frame": start, "picture": pic}
    if int(clip) > 1:
        prev = Path(((parts.get(str(int(clip) - 1)) or {}).get("outputs") or {}).get("clip") or "")
        if prev.is_file():
            out["overlap"] = overlap_diff(prev, path, fmt["frames"], ov)
    nxt = Path(((parts.get(str(int(clip) + 1)) or {}).get("outputs") or {}).get("clip") or "")
    if not nxt.is_file():
        nxt = None
    if nxt is not None:                                     # a middle clip: its END must match the clip after it too
        out["overlap_out"] = overlap_diff(path, nxt, fmt["frames"], ov)
    first = Path(((parts.get("1") or {}).get("outputs") or {}).get("clip") or "")
    if first.is_file() and int(clip) > 1:
        p1 = picture(first)
        if "sharpness" in p1:
            out["sharpness_vs_clip1"] = round(pic["sharpness"] / max(p1["sharpness"], 1e-6), 2)
    if with_motion:
        out["motion"] = motion(run_id, path, start)
    if critic_dir:
        out["critic"] = critic_pack(run_id, int(clip), path, start, Path(critic_dir),
                                    tag or f"c{int(clip)}" + (f"t{take}" if take is not None else ""))
        prev = Path(((parts.get(str(int(clip) - 1)) or {}).get("outputs") or {}).get("clip") or "")
        if int(clip) > 1 and prev.is_file():               # the frames that the candidate's start must continue
            import cv2
            pf = _frames(prev)
            if not pf:
                raise runner.RunError(f"clip {int(clip) - 1} has no readable frames: cannot show its end to the critic")
            s0 = fmt["frames"] - ov
            cv2.imwrite(str(Path(critic_dir) / "strip_PREV.jpg"),
                        _strip(pf, [i for i in range(s0, len(pf), max(1, ov // 4))]))
    after = deps_now()
    if after != before:
        raise runner.RunError("a compared clip changed while it was scored (a take was put in?): score it again")
    out["deps"] = before                                  # what the numbers were measured against
    runner.save_score(d, int(clip), take, out)
    return out
