"""Templates from a dance VIDEO: the mannequin copies the motion of the person in the video.

    start(name, video, start=0, seconds=None, sound="video")  -> the record ("queued") + a detached worker
    make(name)                                                 -> the worker (`python -m cinestudio shot work-mocap NAME`)

The worker (queue kind "template", one GPU job):
  1. cut the chosen part of the video and change it to 24 fps (ffmpeg), small enough for the decoded-frame budget;
     keep its sound as a WAV (`sound="video"`) and measure the tempo and the beats of that sound (cinestudio/song.py:
     an original song at the same tempo can replace it later);
  2. SAM 3D Body reads the person's 3D joints in every frame (graph workflows/sam3d_body -> a GLB). The graph gives the
     model no person boxes, so it uses one full-frame box: the video must show ONE dancer;
  3. cinestudio/mocap.py turns them into a motion file;
  4. Blender (templates/flash_step/build.py, headless) keys the mannequin with it: a still front camera, a plain floor
     (a 360 panorama gives the place in the run), the sound as a strip;
  5. the template is stored like `shot add` (a Workbench preview with the sound).

Several people (2026-10-25, `people="2"|"3"|"4"|"all"|"pick"` or `tracks=[...]`): the graph workflows/sam3d_body_multi lets SAM 3.1
find and follow every person (numbered 0, 1, ... in the order found) and reads each one's 3D joints inside its own mask,
with each person's place in camera space. The scan is kept in <template>/people/ (pose.glb, the label video, the
numbered preview video and a contact sheet "sheet.jpg"). "2" / "3" / "4" copy that many people seen longest and
largest, "all" every person seen long enough (up to mocap.PEOPLE_MAX = 4, the most characters a run takes); "pick"
stops after the scan with status "pick" until `pick(name, tracks)` (the app's "who is who" choice; no second GPU pass)
builds the template from the chosen people. The people become Mannequin / MannequinRig (the picture's LEFT one: the
run's first --character), Mannequin2 / MannequinRig2, ... left to right, each at its place from the video (distance
and depth order kept, the depth spread limited: mocap.place_people; templates/flash_step/build.py per person, then
bpy_jobs/merge_duo.py merges them all), the still camera far enough for all. The template is TRIMMED to the time all chosen people are in the video (mocap.shared_span).

Sound and licences: the video's own sound is usually someone else's recording, so a template with it is marked
`private_sound` (never for the built-in library or a contest submission). `sound="none"` keeps no sound."""
from __future__ import annotations

import json
import math
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from . import server, shots, video, workqueue
from .locks import LockBusy

FPS = 24
SECONDS_MAX = 30.0
SECONDS_DEFAULT = 10.0
SETTLE_FRAMES = 16      # the last frames ease into the first (standing) pose: a cut window ends on a stand, not in a
#                         step (2026-10-23, shan shen bu ended on one leg; mocap.settle)
SOUNDS = ("video", "none")
PEOPLE = ("1", "2", "3", "4", "all", "pick")   # one person (the full-frame graph, the default) | the N most visible |
#                                               every person seen long enough (up to mocap.PEOPLE_MAX) | the user picks
MULTI_MAX_PEOPLE = 4              # SAM 3 follows at most this many people (SAM3_VideoTrack max_objects)
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi")
BUILD = shots.blender.ROOT / "templates" / "flash_step" / "build.py"
MERGE = shots.blender.BPY_JOBS / "merge_duo.py"
# ComfyUI's LoadVideo -> GetVideoComponents decodes EVERY frame into one float32 tensor (12 bytes per pixel; the frame
# list and the stacked tensor exist together for a moment, so twice that): the cut is scaled so the stacked tensor stays
# under this budget (30 s at 24 fps -> ~0.5 Mpx per frame, ~940x530). The model crops one full-frame box to its own
# input size, so a smaller frame loses little.
DECODE_BUDGET = 4 * 1024 ** 3
LONG_SIDE_MAX = 1280


class TemplateError(shots.ShotError):
    pass


class MocapUnresolved(TemplateError):
    """The SAM 3D job could not be confirmed as ended (song._drain gave "unknown"): the record keeps "unresolved" and a
    normal retry with the same name is refused until the user checks ComfyUI and deletes the template (as songs do)."""


def cancel_flag(name: str) -> Path:
    return shots.shot_dir(name) / "cancel"


def request_cancel(name: str) -> dict:
    """Explicit cancel of one template that is queued or being made: the worker stops between its steps, and a cancel
    during the SAM 3D job cancels that ComfyUI job (job-scoped)."""
    d = shots.shot_dir(name)
    if not (d / "shot.json").is_file():
        raise TemplateError(f"no template {name!r}")
    cancel_flag(name).write_text(str(time.time()), encoding="utf-8")
    return {"template": name, "cancel_requested": True}


def _video_facts(path: Path) -> tuple[float, int, int]:
    """(seconds, width, height) from the container (fast: no full decode); a full count only when the container
    gives no frame count."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise TemplateError("cannot read the video")
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    secs = n / fps if n > 0 and fps and fps > 0 and math.isfinite(n / fps) else 0.0
    if secs <= 0:
        p = video.probe(path)
        secs = p.frames / p.fps if p.fps else 0.0
    if secs <= 0 or w <= 0 or h <= 0:
        raise TemplateError("the video has no frames")
    return secs, w, h


def scale_factor(width: int, height: int, seconds: float) -> float:
    """The factor for both sides of the cut: the long side at most LONG_SIDE_MAX, the decoded frames within
    DECODE_BUDGET. Width x height and the long side do not change with a rotation tag, so the factor is right for a
    phone video too."""
    frames = max(1.0, seconds * FPS)
    budget_px = DECODE_BUDGET / (frames * 12.0)
    return min(1.0, LONG_SIDE_MAX / max(width, height), math.sqrt(budget_px / (width * height)))


# ---------------------------------------------------------------- start (the app / CLI)
def _people_args(people: str | int | None, tracks) -> tuple[str, list[int] | None]:
    """(one of PEOPLE, tracks or None). Given tracks always use the several-people graph."""
    from . import mocap
    people = "1" if people in (None, "") else str(people).strip().lower()
    if people not in PEOPLE:
        raise TemplateError(f"people must be one of {list(PEOPLE)}")
    if isinstance(tracks, str):
        tracks = [t for t in tracks.replace(" ", "").split(",") if t]
    if tracks:
        try:
            tracks = mocap.check_tracks(tracks, MULTI_MAX_PEOPLE)
        except mocap.MocapError as exc:
            raise TemplateError(str(exc)) from None
        if people == "1":
            people = "pick"
    return people, (tracks or None)


def prepare(name: str, video_file: str | Path, start: float = 0.0, seconds: float | None = None,
            sound: str = "video", description: str = "", move: bool = False, people: str = "1",
            tracks: list[int] | None = None) -> dict:
    """Check what can be checked at once, put the video into the template's folder and write the record ("queued"),
    under the template's lifecycle lock. The video is staged OUTSIDE the template's folder first: a retry with a
    failed template's own source.mp4 must not delete its input. `people` / `tracks`: see the module text."""
    src = Path(video_file)
    if not src.is_file() or src.suffix.lower() not in VIDEO_SUFFIXES:
        raise TemplateError(f"choose a video file ({', '.join(VIDEO_SUFFIXES)})")
    if sound not in SOUNDS:
        raise TemplateError(f"sound must be one of {list(SOUNDS)}")
    people, tracks = _people_args(people, tracks)
    total, w, h = _video_facts(src)
    start = 0.0 if start is None else float(start)
    seconds = SECONDS_DEFAULT if seconds is None else float(seconds)
    if not (math.isfinite(start) and math.isfinite(seconds)):
        raise TemplateError("the start and the length must be numbers")
    if start < 0:
        raise TemplateError("the start must not be negative")
    if start >= total - 1.0:
        raise TemplateError(f"the start ({start:g} s) is at or after the end of the video ({total:.1f} s)")
    seconds = min(seconds, total - start, SECONDS_MAX)
    if seconds < 2.0:
        raise TemplateError("the part of the video must be at least 2 s long")
    shots.check_new_name(name)
    d = shots.shot_dir(name)
    with shots._Life(name, f"template {name!r} is already being made"):
        if d.exists():
            try:
                old = shots.load(name)
            except shots.ShotError:
                old = {}
            if old.get("handmade") or old.get("status") == "done" or shots._busy(old):
                raise TemplateError(f"a template named {name!r} exists: choose another name or delete it first")
            if old.get("status") == "unresolved":
                raise TemplateError(f"template {name!r} is unresolved (its SAM 3D job was not confirmed as ended): "
                                    "check ComfyUI's queue, then delete the template before you make it again")
            if not old and any(d.iterdir()):
                raise TemplateError(f"the folder of {name!r} is used by something else: choose another name")
        token = secrets.token_hex(3)
        staged = shots.shots_dir() / f".incoming_{name}_{token}{src.suffix.lower()}"
        old_dir = shots.shots_dir() / f".replaced_{name}_{token}"
        staged.parent.mkdir(parents=True, exist_ok=True)
        inside = d.exists() and d.resolve() in src.resolve().parents    # the input belongs to the folder replaced:
        move = move and not inside                                       # copy it, so a rollback still has it
        try:
            (shutil.move if move else shutil.copy2)(str(src), str(staged))
            if d.exists():                                    # a failed / canceled / interrupted try: kept aside until
                os.replace(d, old_dir)                        # the new record is written (it may hold the input)
            d.mkdir(parents=True)
            dst = d / f"source{src.suffix.lower()}"
            os.replace(staged, dst)
            params = {"video": dst.name, "original": src.name, "start": round(start, 3), "seconds": round(seconds, 3),
                      "sound": sound, "video_seconds": round(total, 3), "size": [w, h],
                      "scale": round(scale_factor(w, h, seconds), 4), "people": people,
                      **({"tracks": tracks} if tracks else {})}
            who = "The mannequin copies the motion" if people == "1" else "The mannequins copy the people"
            rec = {"name": name, "kind": "blender", "source": "video", "status": "queued", "added": time.time(),
                   "pid": None, "params": params, "private_sound": sound == "video",
                   "description": (description.strip() or
                                   f"{who} of {src.name} ({start:g}-{start + seconds:g} s, SAM 3D Body); still front "
                                   f"camera, a plain floor (a 360 panorama gives the place).")}
            shots._write(d / "shot.json", rec)               # atomic: the record exists completely or not at all
        except BaseException:
            if move and not src.exists():                     # a moved input goes back to the caller first
                for here in (staged, d / f"source{src.suffix.lower()}"):
                    if here.exists():
                        shutil.move(str(here), str(src))
                        break
            if d.exists() and not (d / "shot.json").is_file():
                shutil.rmtree(d, ignore_errors=True)          # no half-made folder that blocks the name
            if old_dir.exists() and not d.exists():
                os.replace(old_dir, d)                        # the old try comes back as it was (with its input)
            staged.unlink(missing_ok=True)                    # else a copy (the original is kept or restored above)
            raise
        shutil.rmtree(old_dir, ignore_errors=True)
        return rec


def start(name: str, video_file: str | Path, start: float = 0.0, seconds: float | None = None, sound: str = "video",
          description: str = "", move: bool = False, spawner: Callable | None = None, people: str = "1",
          tracks: list[int] | None = None) -> dict:
    """prepare(), reserve the place in the shared queue, then start the detached worker (`shot work-mocap NAME`): an
    app restart does not stop it."""
    from . import spawn
    rec = prepare(name, video_file, start, seconds, sound, description, move, people, tracks)
    label = f"{name} · template"
    workqueue.add("template", name, label)
    d = shots.shot_dir(name)
    try:
        pid, independent = (spawner or spawn.detached)([sys.executable, "-m", "cinestudio", "shot", "work-mocap", name],
                                                       d / "build.log", shots.blender.ROOT)
    except Exception as exc:
        workqueue.done("template", name)
        rec.update(status="failed", ended=time.time(), error=f"the worker did not start: {exc}")
        shots._write(d / "shot.json", rec)
        raise
    return {"name": name, "pid": pid, "status": "queued", "params": rec["params"],
            **({} if independent else {"warning": "the worker may stop when this app stops"})}


# ---------------------------------------------------------------- the worker
def make(name: str, on_event: Callable[[dict], None] | None = None, sam_job: Callable | None = None,
         build: Callable | None = None, merge: Callable | None = None, scan_layers: Callable | None = None) -> dict:
    """The worker: make the template of a queued record. It holds the template's lifecycle lock for the whole job and
    waits for its turn in the shared queue (one GPU job at a time). After a "pick" scan (several people, the choice
    made: pick()) the GPU part is done already: no queue, only Blender."""
    with shots._Life(name, f"template {name!r} is already being made by another worker"):
        return _make_locked(name, on_event or (lambda e: None), sam_job, build, merge, scan_layers)


def _cut(src: Path, start: float, seconds: float, factor: float, dst: Path) -> None:
    """The chosen part at 24 fps, both sides scaled by `factor` (even sizes for H.264)."""
    vf = f"fps={FPS},scale='trunc(iw*{factor:.5f}/2)*2':'trunc(ih*{factor:.5f}/2)*2'"
    subprocess.run([video.ffmpeg(), "-nostdin", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}",
                    "-i", str(src), "-an", "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(dst)],
                   check=True, stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _sound(src: Path, start: float, seconds: float, dst: Path) -> bool:
    if not video.has_audio(src):
        return False
    subprocess.run([video.ffmpeg(), "-nostdin", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}",
                    "-i", str(src), "-vn", "-ac", "2", "-ar", "48000", "-t", f"{seconds:.3f}", str(dst)], check=True,
                   stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return dst.is_file() and dst.stat().st_size > 1000


def _sam(cut: Path, glb: Path, tag: str, emit: Callable[[dict], None], canceled: Callable[[], bool],
         backend_factory: Callable[[], Any] | None = None, multi: bool = False) -> Path:
    """One SAM 3D Body job with the rules of every GPU job: the GPU lock from the model-family check until the job
    ended, the prompt id recorded as ours, a job-scoped cancel that does not wait for a progress event; on any other
    error the lock is kept until ComfyUI confirms that the job ended (song._drain), so no other GPU work starts next to
    an orphaned job (Codex review 2026-10-22: character.image_job releases the lock on a lost connection).
    `multi` = the several-people graph (workflows/sam3d_body_multi): besides the GLB it downloads the label video
    (labels.mp4) and the numbered preview (track_preview.mp4) beside `glb`."""
    from . import comfy, song
    from . import models as _models
    from .backend import SdkBackend
    from .workflow import ProfileWorkflow
    variant = "body_multi" if multi else "body"
    wf = ProfileWorkflow("sam3d", variant, models=_models.job_bundle())
    token = secrets.token_hex(3)
    prefix = f"cinestudio/mocap_{tag}_{token}"
    values = {"fps": float(FPS), "include_hands": True, "output_prefix": prefix,
              "camera_translation": "absolute" if multi else "centered"}
    if multi:
        values.update(person_text="person", max_people=MULTI_MAX_PEOPLE)
    graph = wf.build(values)
    from .workflow import graph_cache, save_graph          # the exact graph sent (the app's Workflows view)
    save_graph(graph_cache(), f"mocap_{tag}", graph, files={wf.targets("video")[0]: cut}, keep=200,
               meta={"tag": f"mocap_{tag}", "profile": "sam3d", "variant": variant, "prefix": prefix})
    if backend_factory is None and not server.alive():
        emit({"event": "starting_comfyui"})
        comfy.ensure_running()
    lock = server.gpu_lock(note={"template": tag})
    try:
        lock.acquire()
    except LockBusy:
        emit({"event": "waiting_for_gpu", "holder": lock.holder()})
        lock.timeout = 6 * 3600
        lock.acquire()
    try:
        switch = server.ensure_family(wf.family_key())
        if switch.get("restarted"):
            emit({"event": "comfyui_restarted", **switch})
        if canceled():
            raise workqueue.QueueCanceled(f"template {tag} was canceled")
        backend = (backend_factory or SdkBackend)()
        jid = None
        try:
            jid = backend.submit(graph, {wf.targets("video")[0]: cut}, key=f"mocap-{tag}-{token}", prefix=prefix)
            server.note_owned(jid)
            emit({"event": "comfy_job", "job": jid})

            def progress(kind: str, data: dict) -> None:
                if kind == "progress" and data.get("steps"):
                    emit({"event": "progress_job", "value": data.get("step"), "max": data.get("steps")})
            with _cancel_watch(backend, jid, canceled, prefix):
                st = backend.watch(jid, progress)
            if canceled():
                raise workqueue.QueueCanceled(f"template {tag} was canceled")
            if not st.ok:
                raise TemplateError(f"ComfyUI job {jid} {st.status}: {st.error}")
            glb.parent.mkdir(parents=True, exist_ok=True)
            backend.download(jid, wf.bind["outputs"]["image"], glb)    # a GLB: read from ComfyUI's history
            if multi:                                     # the previews are temp files (SAM3_TrackPreview)
                backend.download(jid, wf.bind["outputs"]["labels"], glb.parent / "labels.mp4")
                backend.download(jid, wf.bind["outputs"]["preview"], glb.parent / "track_preview.mp4")
            return glb
        except BaseException:
            # the job may still run: keep the GPU lock until ComfyUI confirms its end (song._drain waits up to
            # song.DRAIN_SECONDS, the rule of every GPU worker), and keep passing a cancel on while it drains
            with _cancel_watch(backend, jid, canceled, prefix):
                ended = song._drain(prefix, jid, backend)
            if ended != "ended":
                raise MocapUnresolved(f"template {tag}: the ComfyUI job ({jid or prefix}) could not be confirmed as "
                                      "ended; check ComfyUI's queue before starting more GPU work")
            if canceled():                                    # the user canceled while it drained: a cancel, not a failure
                raise workqueue.QueueCanceled(f"template {tag} was canceled")
            raise
    finally:
        lock.release()


class _cancel_watch:
    """While open: when the cancel flag is set, ask ComfyUI to cancel job `jid` (every 2 s until a request succeeds;
    a failed request is tried again). A cancel then does not wait for a progress event. Without a job id (the submit
    lost its answer) the job is looked up by its output prefix, as song._drain does."""

    def __init__(self, backend: Any, jid: str | None, canceled: Callable[[], bool], prefix: str | None = None,
                 every: float = 2.0):
        self.backend, self.jid, self.canceled, self.prefix, self.every = backend, jid, canceled, prefix, every
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"mocap-cancel-{jid}", daemon=True)

    def _find(self) -> str | None:
        if self.jid or not self.prefix:
            return self.jid
        try:
            self.jid = self.backend.find(self.prefix) or server.find_prompt(self.prefix, server.comfyui_url())
        except Exception:  # noqa: BLE001 - not found yet: try again on the next tick
            self.jid = None
        return self.jid

    def _run(self) -> None:
        while not self.stop.wait(self.every):
            if self.canceled() and self._find():
                try:
                    self.backend.cancel(self.jid)
                    return
                except Exception:  # noqa: BLE001 - ComfyUI did not answer: try again on the next tick
                    continue

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=5)
        return False


def _build(motion: Path, out: Path, audio: Path | None, log: Path, camera_distance: float | None = None) -> None:
    shots._run_blender(BUILD, None, ["--motion", motion, "--out", out, *(["--audio", audio] if audio else []),
                                     *(["--camera-distance", camera_distance] if camera_distance else [])], log)


def mannequin_names(i: int) -> tuple[str, str]:
    """The collection and rig of the i-th mannequin (0-based): Mannequin / MannequinRig, Mannequin2 / MannequinRig2, ..."""
    suffix = "" if i == 0 else str(i + 1)
    return f"Mannequin{suffix}", f"MannequinRig{suffix}"


def _merge(parts: list[Path], out: Path, places: list, log: Path) -> None:
    """One-mannequin shots -> one shot with Mannequin (parts[0], at places[0]), Mannequin2 (parts[1], at places[1]),
    ... (bpy_jobs/merge_duo.py, any number of others)."""
    # "--places=X,Y;X,Y" in ONE argument: a negative X ("-0.951,0.784") alone looks like an option to argparse
    # ("argument --place-a: expected one argument", user report 2026-10-25)
    args = []
    for other in parts[1:]:
        args += ["--other", other]
    shots._run_blender(MERGE, parts[0], [*args, "--out", out,
                                         "--places=" + ";".join(",".join(map(str, pl)) for pl in places)], log)


def _build_people(motions: list[Path], places: list, distance: float, out: Path, audio: Path | None, log: Path,
                  build: Callable | None = None, merge: Callable | None = None) -> None:
    """One person: one build (centred). Several: each person built alone with the same camera, then merged at their
    places (the first = the picture's left = Mannequin, then Mannequin2, ...)."""
    if len(motions) == 1:
        (build or _build)(motions[0], out, audio, log, camera_distance=distance)
        return
    parts = []
    for i, m in enumerate(motions):
        part = out.parent / "work" / f"person{i + 1}.blend"
        part.parent.mkdir(parents=True, exist_ok=True)
        (build or _build)(m, part, audio if i == 0 else None, log, camera_distance=distance)
        parts.append(part)
    (merge or _merge)(parts, out, places, log)


def _people_sheet(cut: Path, labels: dict, summary: list[dict], out: Path, n: int = 4) -> Path:
    """"Who is who": `n` frames of the cut side by side (the frames where the most people are visible, spread over the
    clip), each person's box in its colour with its number, for the app's choice."""
    from . import mocap
    pres = mocap.presence(labels)
    frames = labels["frames"]
    if frames == 0:
        raise TemplateError("the label video has no frames")
    count = pres.sum(0) if len(pres) else [0] * frames
    best = max(count) if len(count) else 0
    good = [i for i in range(frames) if count[i] == best] or list(range(frames))
    picks = sorted({good[round(k * (len(good) - 1) / max(1, n - 1))] for k in range(n)})
    cap = cv2.VideoCapture(str(cut))
    tiles = []
    try:
        for i in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = cap.read()
            if not ok:
                continue
            h, w = img.shape[:2]
            s = 360.0 / h
            img = cv2.resize(img, (max(2, round(w * s)), 360), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            for k, row in enumerate(labels["box"]):
                b = row[i] if i < len(row) else None
                if not b or not pres[k][i]:
                    continue
                r, g, bl = (int(round(c * 255)) for c in mocap.TRACK_COLORS[k % len(mocap.TRACK_COLORS)])
                x0, y0, x1, y1 = round(b[0] * w), round(b[1] * h), round(b[2] * w), round(b[3] * h)
                cv2.rectangle(img, (x0, y0), (x1, y1), (bl, g, r), 3)
                cv2.rectangle(img, (x0, max(0, y0 - 34)), (x0 + 34, max(34, y0)), (bl, g, r), -1)
                cv2.putText(img, str(k), (x0 + 7, max(27, y0 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2,
                            cv2.LINE_AA)
            cv2.putText(img, f"{i / FPS:.1f} s", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
            tiles.append(img)
    finally:
        cap.release()
    if not tiles:
        raise TemplateError("cannot read frames of the cut for the preview")
    row = tiles[0]
    for t in tiles[1:]:
        row = np.hstack([row, np.zeros((row.shape[0], 6, 3), np.uint8), t])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), row, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def _sound_and_beats(rec: dict, src: Path, start: float, seconds: float, wav: Path) -> Path | None:
    """The video's sound of the window as a WAV and its measured tempo (rec["music"]); None without sound."""
    from . import song
    if not _sound(src, start, seconds, wav):
        rec["private_sound"] = False
        rec["music"] = {"note": "the video has no sound"}
        return None
    try:
        b = song.beats(wav)
        rec["music"] = {"bpm": b.get("bpm"), "beats": len(b.get("beats") or []),
                        "first_downbeat": (b.get("downbeats") or [None])[0]}
    except Exception as exc:                              # noqa: BLE001 - the beats are only advice
        rec["music"] = {"error": str(exc)}
    return wav


def _scan_people(rec: dict, pdir: Path, cut: Path, glb: Path) -> None:
    """After the several-people job: read the label video (who is where in each frame), keep it as labels.json, make
    the "who is who" contact sheet and write the people found into the record (rec["people"])."""
    from . import mocap
    objects = len(mocap.read_glb(glb)["tracks"])
    labels = mocap.read_labels(pdir / "labels.mp4", objects)
    (pdir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    summary = mocap.people_summary(labels, float(FPS))
    sheet = None
    try:
        sheet = str(_people_sheet(cut, labels, summary, pdir / "sheet.jpg"))
    except Exception as exc:                              # noqa: BLE001 - the sheet is only a help for the choice
        rec.setdefault("warnings", []).append(f"no who-is-who sheet: {exc}")
    seen = [s for s in summary if s["visible"] > 0]
    rec["people"] = {"scanned": True, "count": objects, "seen": len(seen), "tracks": summary, "sheet": sheet,
                     "video": str(pdir / "track_preview.mp4") if (pdir / "track_preview.mp4").is_file() else None}
    if not seen:
        raise TemplateError("SAM 3 found no person in the video")


def _people_motion_files(rec: dict, d: Path, glb: Path, tracks: list[int], src: Path,
                         audio: Path | None) -> tuple[list[Path], Path | None]:
    """The chosen people's motion files (motion.json = the left one, motion2.json, ... to the right), each eased into
    its first pose at the end; the sound cut to the people's shared time when the template was trimmed."""
    from . import mocap
    p = rec["params"]
    labels = json.loads((d / "people" / "labels.json").read_text(encoding="utf-8"))
    res = mocap.people_motions(glb, labels, tracks, float(FPS))
    paths = []
    for i, m in enumerate(res["motions"]):
        if len(m["frames"]) > 2 * SETTLE_FRAMES:
            m["settle"] = mocap.settle(m["frames"], SETTLE_FRAMES)
        path = d / ("motion.json" if i == 0 else f"motion{i + 1}.json")
        path.write_text(json.dumps(m), encoding="utf-8")
        paths.append(path)
    first = res["motions"][0]
    trimmed = res["start_s"] > 0.02 or res["seconds"] < float(p["seconds"]) - 0.06
    if audio is not None:                                 # the sound follows the (trimmed) window; cut again each time:
        audio = _sound_and_beats(rec, src, float(p["start"]) + res["start_s"], res["seconds"],   # an earlier choice
                                 d / "sound_source.wav")                                         # may have trimmed it
    rec["people"].update(
        chosen=[{**pp, "collection": mannequin_names(i)[0], "rig": mannequin_names(i)[1]}
                for i, pp in enumerate(res["people"])],
        span_s=[res["start_s"], round(res["start_s"] + res["seconds"], 3)], trimmed=trimmed,
        camera_distance=res["camera_distance"])
    rec["motion"] = {"frames": len(first["frames"]), "tracks": first["tracks"], "track": first["track"],
                     "people": len(paths), "floor_tilt_deg": first["scale"]["floor_tilt_deg"],
                     "floor_from": first["scale"]["floor_from"],
                     "kicks": sum(len(m["scale"]["kicks"]) for m in res["motions"])}
    return paths, audio


def pick(name: str, tracks, spawner: Callable | None = None, wait: bool = False,
         on_event: Callable[[dict], None] | None = None, **make_kw) -> dict:
    """The user's choice after a "pick" scan: which people (1 to mocap.PEOPLE_MAX SAM 3 numbers, as in the sheet) become the
    mannequins. Instant: no GPU job (the scan is kept); Blender builds the template in a detached worker (`wait`: in
    this process). A done template with a kept scan can be chosen again (it is rebuilt from the scan; runs keep their
    own frozen copy of the old .blend)."""
    from . import mocap, spawn
    d = shots.shot_dir(name)
    with shots._Life(name, f"template {name!r} is being made now: wait until it ends"):
        rec = shots.load(name)
        info = rec.get("people") or {}
        if rec.get("source") != "video" or not info.get("scanned") or not (d / "people" / "pose.glb").is_file():
            raise TemplateError(f"{name!r} has no people scan (make it with people 'pick' or '2')")
        if rec.get("status") not in ("pick", "failed", "done") or rec.get("builtin"):
            raise TemplateError(f"template {name!r} is {rec.get('status')}: people can be chosen while it waits for "
                                "the choice (status pick), after a failure or when it is done")
        if isinstance(tracks, str):
            tracks = [t for t in tracks.replace(" ", "").split(",") if t]
        try:
            tracks = mocap.check_tracks(tracks, int(info.get("count") or 0))
        except mocap.MocapError as exc:
            raise TemplateError(str(exc)) from None
        seen = {s["track"] for s in info.get("tracks") or [] if s.get("visible")}
        unseen = [t for t in tracks if t not in seen]
        if unseen:
            raise TemplateError(f"person {unseen[0]} is not visible in the video")
        rec["params"]["tracks"] = tracks
        rec.update(status="queued", pid=None, requeued=time.time(), error=None)
        shots._write(d / "shot.json", rec)
    if wait:
        return make(name, on_event=on_event, **make_kw)
    try:
        pid, independent = (spawner or spawn.detached)([sys.executable, "-m", "cinestudio", "shot", "work-mocap", name],
                                                       d / "build.log", shots.blender.ROOT)
    except Exception as exc:
        rec.update(status="failed", ended=time.time(), error=f"the worker did not start: {exc}")
        shots._write(d / "shot.json", rec)
        raise
    return {"name": name, "pid": pid, "status": "queued", "tracks": tracks,
            **({} if independent else {"warning": "the worker may stop when this app stops"})}


def _make_locked(name: str, emit: Callable[[dict], None], sam_job: Callable | None, build: Callable | None,
                 merge: Callable | None = None, scan_layers: Callable | None = None) -> dict:
    from . import mocap
    d = shots.shot_dir(name)
    rec = shots.load(name)
    if rec.get("source") != "video" or rec.get("kind") != "blender":
        raise TemplateError(f"{name!r} is not a template made from a video")
    if rec.get("status") == "done":
        raise TemplateError(f"template {name!r} is done already")
    p = rec["params"]
    label = f"{name} · template"
    canceled = lambda: cancel_flag(name).is_file()        # noqa: E731
    steps = ["cut", "motion", "convert", "blender", "preview"]

    def step(key: str, text: str) -> None:
        i = steps.index(key)
        rec["progress"] = {"step": key, "i": i + 1, "n": len(steps), "text": text}
        shots._write(d / "shot.json", rec)
        emit({"event": "progress", "step": key, "value": i, "max": len(steps), "text": text})

    def stop_if_canceled() -> None:
        if canceled():
            raise workqueue.QueueCanceled(f"template {name} was canceled")

    people = str(p.get("people") or "1")
    multi = people != "1"
    pdir = d / "people"
    scanned = multi and bool((rec.get("people") or {}).get("scanned")) and (pdir / "pose.glb").is_file() \
        and (pdir / "labels.json").is_file()
    rec.update(pid=os.getpid(), status="queued")
    shots._write(d / "shot.json", rec)
    work = d / "work"
    src = d / p["video"]
    try:
        if not scanned:                                   # the GPU part: the queue first
            workqueue.claim("template", name, label)
            workqueue.wait_turn("template", name, label, should_stop=canceled,
                                on_wait=lambda n: emit({"event": "queued", "ahead": n}))
        rec.update(status="running", started=time.time())
        work.mkdir(parents=True, exist_ok=True)
        audio = d / "sound_source.wav" if p["sound"] == "video" and rec.get("private_sound") else None
        if not scanned:
            # 1. the part of the video, its sound and the sound's beats
            step("cut", "The chosen part of the video (24 fps) and its sound")
            cut = work / "cut.mp4"
            factor = p.get("scale") or scale_factor(*(p.get("size") or _video_facts(src)[1:]), p["seconds"])
            _cut(src, p["start"], p["seconds"], factor, cut)
            audio = _sound_and_beats(rec, src, p["start"], p["seconds"], d / "sound_source.wav") \
                if p["sound"] == "video" else None
            stop_if_canceled()
            # 2. SAM 3D Body (one GPU job; a cancel stops it)
            step("motion", "SAM 3D Body reads the person's 3D motion" if not multi else
                 "SAM 3 finds the people, SAM 3D Body reads their 3D motion")
            if multi:
                glb = pdir / "pose.glb"
                pdir.mkdir(parents=True, exist_ok=True)
                (sam_job or _sam)(cut, glb, name, emit, canceled, multi=True)
                stop_if_canceled()
                _scan_people(rec, pdir, cut, glb)
                shots._write(d / "shot.json", rec)
            else:
                glb = work / "pose.glb"
                (sam_job or _sam)(cut, glb, name, emit, canceled)
            stop_if_canceled()
        # 3. the motion file(s)
        step("convert", "The 3D joints become the mannequin's motion" if not multi else
             "The 3D joints become the mannequins' motions")
        if multi:
            if p.get("tracks"):
                tracks = p["tracks"]
            elif people in ("2", "3", "4", "all"):        # the N people seen longest and largest
                want = mocap.PEOPLE_MAX if people == "all" else int(people)
                tracks = mocap.choose_people(rec["people"]["tracks"], want)
                if people != "all" and 0 < len(tracks) < want:
                    words = {1: "one", 2: "two", 3: "three", 4: "four"}
                    rec.setdefault("warnings", []).append(
                        f"{words[want]} people were asked for, {len(tracks)} {'was' if len(tracks) == 1 else 'were'} "
                        f"seen long enough: the template has {words[len(tracks)]} "
                        f"mannequin{'s' if len(tracks) > 1 else ''}")
            else:
                tracks = None
            if not tracks:
                if people == "pick":                      # the user chooses who becomes a mannequin (pick())
                    rec.pop("progress", None)
                    rec.update(status="pick", ended=time.time(), pid=None)
                    shots._write(d / "shot.json", rec)
                    shutil.rmtree(work, ignore_errors=True)
                    emit({"event": "pick", "name": name, "people": rec["people"]["tracks"]})
                    return rec
                raise TemplateError("SAM 3 found no person seen long enough (a quarter of the clip) in the video")
            motions, audio = _people_motion_files(rec, d, pdir / "pose.glb", tracks, src, audio)
        else:
            motion = d / "motion.json"
            m = mocap.motion_from_glb(glb, motion, seconds=float(p["seconds"]), fps=float(FPS))
            if len(m["frames"]) > 2 * SETTLE_FRAMES:
                m["settle"] = mocap.settle(m["frames"], SETTLE_FRAMES)
                motion.write_text(json.dumps(m), encoding="utf-8")
            rec["motion"] = {"frames": len(m["frames"]), "tracks": m["tracks"], "track": m["track"],
                             "floor_tilt_deg": m["scale"]["floor_tilt_deg"], "floor_from": m["scale"]["floor_from"],
                             "kicks": len(m["scale"]["kicks"])}
        shots._write(d / "shot.json", rec)
        stop_if_canceled()
        # 4. Blender keys the mannequin(s) (headless)
        step("blender", "Blender keys the mannequin with the motion" if not multi else
             "Blender keys the mannequins with the motions")
        blend = d / "shot.blend"
        if multi:
            ch = rec["people"]["chosen"]
            _build_people(motions, [c["place"] for c in ch], rec["people"]["camera_distance"], blend, audio,
                          d / "build.log", build=build, merge=merge)
        else:
            (build or _build)(motion, blend, audio, d / "build.log")
        stop_if_canceled()
        # 5. the preview (Workbench, with the sound)
        step("preview", "Blender makes a preview (Workbench)")
        files = shots.make_preview(d, blend, d / "build.log")
        if multi:                                         # the Layers dialog and the runs see every Mannequin layer
            scan = (scan_layers or shots.scan_layers)(blend)
            rec["layers"] = {"scan": scan, "descriptions": {"set": "", "props": {}},
                             "warnings": shots.layer_warnings(scan)}
            found = [c["collection"] for c in scan.get("characters") or []]
            if len(found) != len(motions):
                raise TemplateError(f"the built shot has the mannequin layers {found}, expected {len(motions)}")
        stop_if_canceled()                                     # a cancel during the build never publishes "done"
        rec.pop("progress", None)
        rec.update(status="done", ended=time.time(), frames=files["frames"],
                   seconds=round(files["frames"] / shots.FPS, 3), files={"blend": str(blend), **files})
        shots._store_check(blend, rec, emit)               # the template rules (shot check; a headless read)
        shots._write(d / "shot.json", rec)
        shutil.rmtree(work, ignore_errors=True)
        emit({"event": "done", "name": name})
        return rec
    except workqueue.QueueCanceled:
        rec.pop("progress", None)
        rec.update(status="canceled", ended=time.time())
        shots._write(d / "shot.json", rec)
        raise
    except MocapUnresolved as exc:
        rec.pop("progress", None)
        rec.update(status="unresolved", ended=time.time(), error=str(exc))
        shots._write(d / "shot.json", rec)
        raise
    except Exception as exc:
        rec.pop("progress", None)
        rec.update(status="failed", ended=time.time(), error=str(exc))
        shots._write(d / "shot.json", rec)
        raise
    finally:
        workqueue.done("template", name)
