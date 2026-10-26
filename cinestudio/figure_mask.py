"""The figure to replace in a video template: a per-frame mask that follows it (SAM 2.1 video tracking in ComfyUI,
kijai/ComfyUI-segment-anything-2, Apache-2.0; weights in setup/models.json profile "sam2").

    seed_points(name)                 -> positive points on keyframes from DWPose (no clicks needed)
    track(name, clicks=None)          -> <template>/mask/mask.mp4 + overlay.mp4 + mask.json (stats and flags)

Clicks: [{"frame": 0, "x": 384, "y": 220, "positive": true}, ...] in the template video's pixels; they are added to
the DWPose seed (a negative click wins over the seed on its frame). SAM 2 propagates forward from the first prompted
frame; points on later frames correct the track there (re-anchor after a turn, an occlusion or a re-entry).
The job waits in the shared queue (kind "template", ref "mask-<name>") and holds the GPU lock with the model family
"sam2" (ComfyUI restarts for it: SAM 2 is kept apart from LTX in 8 GB of VRAM).
"""
from __future__ import annotations

import json
import secrets
import subprocess
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import server, shots, video, workqueue

ROOT = Path(__file__).resolve().parent.parent
MODEL = "sam2.1_hiera_large.safetensors"
KEYFRAME_EVERY = 40               # DWPose re-anchors the track every N frames (Wan2.2-Animate seeds SAM2 the same way)
SEED_JOINTS = (1, 2, 5, 8, 11, 9, 12)   # neck, shoulders, hips, knees (COCO-18): inside the body, stable
JUMP = 0.45                       # flag a frame whose mask area changes by more than this share
FAMILY = "sam2"


class MaskError(shots.ShotError):
    pass


def mask_dir(name: str) -> Path:
    return shots.shot_dir(name) / "mask"


def _video_of(name: str) -> Path:
    rec = shots.load(name)
    v = Path((rec.get("files") or {}).get("video") or "")
    if rec.get("kind") != "video" or not v.is_file():
        raise MaskError(f"{name!r} is not a video template")
    return v


def _passes(name: str, log: Path) -> Path:
    """The template's DWPose passes (cinestudio/video_passes.py, ComfyUI's Python, CPU), made once."""
    from . import comfy
    out = shots.shot_dir(name) / "passes"
    if (out / "keypoints.json").is_file():
        return out
    cfg = comfy.load_config() or comfy.detect()
    stage = out.with_name("passes.staging")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run([cfg["python"], "-B", str(ROOT / "cinestudio" / "video_passes.py"), str(_video_of(name)),
                               str(stage)], stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if proc.returncode != 0:
        raise MaskError(f"DWPose on the template failed; log: {log}")
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    stage.rename(out)
    return out


def seed_points(keypoints: dict, frames: int, every: int = KEYFRAME_EVERY) -> list[dict]:
    """Positive points on keyframes (0, every, 2*every, ...) from the figure's DWPose joints that are shown."""
    pts = []
    for f in range(0, frames, every):
        body = keypoints["frames"][f]
        for j in SEED_JOINTS:
            x, y, shown = body[j]
            if shown:
                pts.append({"frame": f, "x": round(float(x), 1), "y": round(float(y), 1), "positive": True,
                            "seed": True})
    return pts


def merge_points(seed: list[dict], clicks: list[dict] | None) -> dict[int, dict]:
    """{frame: {"pos": [(x, y)], "neg": [(x, y)]}}; the user's clicks are added to the seed on their frame."""
    by: dict[int, dict] = {}
    for p in list(seed) + list(clicks or []):
        d = by.setdefault(int(p["frame"]), {"pos": [], "neg": []})
        d["pos" if p.get("positive", True) else "neg"].append((float(p["x"]), float(p["y"])))
    # a frame with only negative clicks needs a positive point for the node: keep the seed's or skip the frame
    return {f: d for f, d in sorted(by.items()) if d["pos"]}


def graph(points: dict[int, dict], prefix: str, model: str = MODEL) -> dict:
    """LoadVideo -> SAM2 (video) -> chained AddPoints per frame -> propagate -> mask video (white figure on black)."""
    g = {"vid": {"class_type": "LoadVideo", "inputs": {"file": "TEMPLATE.mp4"}},
         "frames": {"class_type": "GetVideoComponents", "inputs": {"video": ["vid", 0]}},
         "sam": {"class_type": "DownloadAndLoadSAM2Model",
                 "inputs": {"model": model, "segmentor": "video", "device": "cuda", "precision": "fp16"}}}
    prev, model_ref = None, ["sam", 0]
    for i, (f, d) in enumerate(points.items()):
        node = f"pts{i}"
        inp = {"sam2_model": model_ref, "coordinates_positive": json.dumps([{"x": x, "y": y} for x, y in d["pos"]]),
               "frame_index": int(f), "object_index": 0}
        if d["neg"]:
            inp["coordinates_negative"] = json.dumps([{"x": x, "y": y} for x, y in d["neg"]])
        if prev is None:
            inp["image"] = ["frames", 0]
        else:
            inp["prev_inference_state"] = [prev, 1]
        g[node] = {"class_type": "Sam2VideoSegmentationAddPoints", "inputs": inp}
        prev, model_ref = node, [node, 0]
    g["seg"] = {"class_type": "Sam2VideoSegmentation",
                "inputs": {"sam2_model": model_ref, "inference_state": [prev, 1], "keep_model_loaded": False}}
    g["img"] = {"class_type": "MaskToImage", "inputs": {"mask": ["seg", 0]}}
    g["mk"] = {"class_type": "CreateVideo", "inputs": {"images": ["img", 0], "fps": 24.0}}
    g["save"] = {"class_type": "SaveVideo", "inputs": {"video": ["mk", 0], "filename_prefix": prefix,
                                                         "format": "auto", "codec": "auto"}}
    return g


def mask_stats(mask_mp4: Path) -> dict:
    """Per-frame mask area (share of the frame) and flags: frames where the area jumps or the mask is empty."""
    cap = cv2.VideoCapture(str(mask_mp4))
    areas, boxes = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        m = f[:, :, 0] > 127
        areas.append(round(float(m.mean()), 4))
        ys, xs = np.nonzero(m)
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None)
    flags = []
    for i, a in enumerate(areas):
        if a == 0:
            flags.append({"frame": i, "why": "no mask"})
        elif i and areas[i - 1] and abs(a - areas[i - 1]) / areas[i - 1] > JUMP:
            flags.append({"frame": i, "why": f"area jumps {areas[i - 1]:.3f} -> {a:.3f}"})
    return {"frames": len(areas), "area": areas, "boxes": boxes, "flags": flags}


def overlay(template: Path, mask_mp4: Path, dst: Path) -> None:
    """The template with the mask as a red tint and a yellow outline (for the user to check the selection)."""
    tv, mv = cv2.VideoCapture(str(template)), cv2.VideoCapture(str(mask_mp4))
    fps = tv.get(cv2.CAP_PROP_FPS) or 24.0
    tmp = dst.with_suffix(".raw.mp4")
    writer = None
    while True:
        ok1, f = tv.read()
        ok2, m = mv.read()
        if not (ok1 and ok2):
            break
        if m.shape[:2] != f.shape[:2]:
            m = cv2.resize(m, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST)
        sel = m[:, :, 0] > 127
        out = f.copy()
        out[sel] = (0.55 * out[sel] + 0.45 * np.array([60, 60, 255])).astype(np.uint8)
        cnts, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 255), 2)
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (f.shape[1], f.shape[0]))
        writer.write(out)
    if writer is None:
        raise MaskError("the mask video is empty")
    writer.release()
    video._run(["-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(dst)])
    tmp.unlink(missing_ok=True)


def track(name: str, clicks: list[dict] | None = None, on_event: Callable[[dict], None] | None = None,
          backend_factory: Callable | None = None) -> dict:
    """Make (or remake) the template's figure mask. Waits for its turn in the shared queue."""
    from .backend import SdkBackend
    emit = on_event or (lambda e: None)
    tpl = _video_of(name)
    d = mask_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    log = d / "track.log"
    ref = f"mask-{name}"
    flag = d / "cancel"
    flag.unlink(missing_ok=True)
    workqueue.claim("template", ref, f"{name} · select figure")
    try:
        workqueue.wait_turn("template", ref, f"{name} · select figure", should_stop=flag.is_file,
                            on_wait=lambda n: emit({"event": "queued", "ahead": n}))
        emit({"event": "progress", "text": "DWPose finds the stand-in (CPU)"})
        kp = json.loads((_passes(name, log) / "keypoints.json").read_text(encoding="utf-8"))
        n = video.probe(tpl).frames
        seed = seed_points(kp, n)
        points = merge_points(seed, clicks)
        if not points:
            raise MaskError("no point to start from: click on the figure")
        prefix = f"cinestudio/mask_{name}_{secrets.token_hex(3)}"
        g = graph(points, prefix)
        from .workflow import graph_cache, save_graph      # the exact graph sent (the app's Workflows view)
        save_graph(graph_cache(), f"mask_{name}", g, files={("vid", "file"): tpl}, keep=200,
                   meta={"tag": f"mask_{name}", "profile": "sam2", "variant": "video", "prefix": prefix})
        if flag.is_file():                               # an explicit cancel before the GPU job
            raise workqueue.QueueCanceled(f"the figure selection of {name} was canceled")
        lock = server.gpu_lock(note={"mask": name}, timeout=6 * 3600)
        lock.acquire()
        try:
            switch = server.ensure_family(FAMILY)
            if switch.get("restarted"):
                emit({"event": "comfyui_restarted", **switch})
            emit({"event": "progress", "text": "SAM 2 follows the figure through the video"})
            be = (backend_factory or SdkBackend)()
            t0 = time.time()
            jid = be.submit(g, {("vid", "file"): tpl}, key=f"mask-{name}-{secrets.token_hex(3)}", prefix=prefix)
            server.note_owned(jid)
            st = be.watch(jid, lambda k, dd: None)
            if not st.ok:
                raise MaskError(f"ComfyUI job {jid} {st.status}: {st.error}")
            raw = d / "mask.tmp.mp4"
            server.download_output(server.comfyui_url(), jid, "save", raw)
        finally:
            lock.release()
        raw.replace(d / "mask.mp4")
        stats = mask_stats(d / "mask.mp4")
        overlay(tpl, d / "mask.mp4", d / "overlay.mp4")
        rec = {"template": name, "model": MODEL, "made": time.time(), "seconds": round(time.time() - t0, 1),
               "points": [dict(p) for p in seed] + [dict(p, seed=False) for p in (clicks or [])],
               "frames": stats["frames"], "flags": stats["flags"], "area": stats["area"], "boxes": stats["boxes"],
               "files": {"mask": str(d / "mask.mp4"), "overlay": str(d / "overlay.mp4")}}
        (d / "mask.json").write_text(json.dumps(rec), encoding="utf-8")
        emit({"event": "done", "flags": len(stats["flags"])})
        return rec
    finally:
        workqueue.done("template", ref)


# ---------------------------------------------------------------- the app's worker (Templates tab "Select figure")
BUSY = ("queued", "running")


def _status_file(name: str) -> Path:
    return mask_dir(name) / "status.json"


def _write_status(name: str, rec: dict) -> None:
    p = _status_file(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{secrets.token_hex(3)}.tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    tmp.replace(p)


def status(name: str) -> dict:
    """The mask job's status record ({} when none); a queued/running record whose worker is gone is "interrupted"."""
    from .locks import pid_alive
    p = _status_file(name)
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if rec.get("status") in BUSY:
        pid = rec.get("pid")
        young = time.time() - float(rec.get("added") or 0) < workqueue.START_GRACE
        if (pid and not pid_alive(int(pid))) or (not pid and not young):
            rec["status"] = "interrupted"
    return rec


def clean_clicks(clicks) -> list[dict]:
    """[{frame, x, y, positive}] with numbers checked; anything else is an error."""
    out = []
    for c in clicks or []:
        try:
            out.append({"frame": int(c["frame"]), "x": round(float(c["x"]), 1), "y": round(float(c["y"]), 1),
                        "positive": bool(c.get("positive", True))})
        except (KeyError, TypeError, ValueError) as exc:
            raise MaskError(f"a click needs frame, x and y numbers: {c!r}") from exc
        if out[-1]["frame"] < 0 or out[-1]["x"] < 0 or out[-1]["y"] < 0:
            raise MaskError(f"a click is outside the video: {c!r}")
    return out


def start(name: str, clicks=None, spawner: Callable | None = None) -> dict:
    """Save the clicks, reserve the queue place (request order) and start the detached worker
    (`python -m cinestudio shot work-mask NAME`): an app restart does not stop it."""
    import sys
    from . import spawn
    _video_of(name)                                              # only video templates
    old = status(name)
    if old.get("status") in BUSY:
        raise MaskError(f"the figure of {name!r} is already being tracked")
    cl = clean_clicks(clicks)
    d = mask_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "clicks.json").write_text(json.dumps(cl), encoding="utf-8")
    _write_status(name, {"status": "queued", "added": time.time(), "pid": None, "clicks": len(cl),
                         "text": "Waiting for its turn on the GPU"})
    ref = f"mask-{name}"
    workqueue.add("template", ref, f"{name} · select figure")
    try:
        pid, independent = (spawner or spawn.detached)([sys.executable, "-m", "cinestudio", "shot", "work-mask", name],
                                                       d / "worker.log", ROOT)
    except BaseException as exc:
        workqueue.done("template", ref)
        _write_status(name, {"status": "failed", "added": time.time(), "error": f"the worker did not start: {exc}"})
        raise
    rec = status(name)
    if rec.get("status") == "queued" and not rec.get("pid"):
        rec["pid"] = pid
        _write_status(name, rec)
    return {"name": name, "pid": pid, "status": rec.get("status", "queued"),
            **({} if independent else {"warning": "the worker may stop when this app stops"})}


def work(name: str, on_event: Callable[[dict], None] | None = None) -> dict:
    """The worker: track() with the saved clicks; the status record follows its progress."""
    import os
    emit = on_event or (lambda e: None)
    d = mask_dir(name)
    try:
        clicks = json.loads((d / "clicks.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        clicks = []
    rec = {"status": "queued", "added": time.time(), "pid": os.getpid(), "clicks": len(clicks),
           "text": "Waiting for its turn on the GPU"}
    _write_status(name, rec)

    def ev(e: dict) -> None:
        if e.get("event") == "queued":
            rec.update(text=f"Waiting for its turn on the GPU ({e.get('ahead')} ahead)")
        elif e.get("event") == "progress":
            rec.update(status="running", started=rec.get("started") or time.time(), text=e.get("text"))
        elif e.get("event") == "comfyui_restarted":
            rec.update(text="ComfyUI restarts for SAM 2")
        _write_status(name, rec)
        emit(e)
    try:
        out = track(name, clicks, on_event=ev)
    except Exception as exc:
        rec.update(status="failed", ended=time.time(), error=str(exc))
        _write_status(name, rec)
        raise
    rec.update(status="done", ended=time.time(), text=None, flags=len(out.get("flags") or []))
    _write_status(name, rec)
    return out


def summary(name: str) -> dict:
    """What the app shows: the status, the last mask (flags, area per frame, points) and the template video's size."""
    d = mask_dir(name)
    info = {}
    cap = cv2.VideoCapture(str(_video_of(name)))
    info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": round(cap.get(cv2.CAP_PROP_FPS) or 24.0, 3)}
    cap.release()
    mask = None
    try:
        m = json.loads((d / "mask.json").read_text(encoding="utf-8"))
        mask = {k: m.get(k) for k in ("frames", "flags", "area", "points", "made", "seconds", "files")}
    except (OSError, ValueError):
        pass
    try:
        clicks = json.loads((d / "clicks.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        clicks = []
    return {"status": status(name), "mask": mask, "clicks": clicks, "video": info}


def request_cancel(name: str) -> dict:
    """Explicit cancel of a queued figure selection (the worker stops before its GPU job)."""
    d = mask_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "cancel").write_text(str(time.time()), encoding="utf-8")
    return {"template": name, "mask_cancel_requested": True}
