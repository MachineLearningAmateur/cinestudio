"""Style reference pictures made by the VIDEO model (2026-10-25, user: "why cant we have the lora generate first using
prompt and then qwen uses that as a reference img").

    python -m cinestudio style make-reference NAME [--subject TEXT] [--seeds 202,303,404,505]     # a GPU job (queue)
    python -m cinestudio style use-reference NAME FILE [--crop x0,y0,x1,y1]                        # instant

The images of a shot are made by Qwen from the style's WORDS; the video model (LTX-2.5) draws the same words with the
style's LoRA in its own look, so the pictures and the video can disagree. Here LTX-2.5 + the style's LoRA makes a short
still clip from TEXT ONLY (workflows/ltx25_txt2img: no layout, no look image, no IC-LoRA; setup/make_txt2img_graphs.py);
the sharpest saved frame of each seed is a candidate (<home>/styles/NAME/candidates/). `use_reference` makes one the
style's reference picture with a crop box (default: CROP_SIDE of the picture's width, a square on the clothes below
the face), and Qwen style conversions (character.new_skin) take that CROP as Picture 2 (AGENTS.md 4.3: one detail,
tightly cropped; a whole picture is copied too literally). A style without a LoRA still works: the picture then shows
LTX's own look for those words.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

from . import styles

VARIANT = "txt2img"
SUBJECT = "a young woman in a hooded jacket standing in a city street, full body"
SEEDS = (202, 303, 404, 505)
WIDTH, HEIGHT, FRAMES = 768, 1152, 17          # portrait for a full-body subject; 64-multiples, 8k+1 frames
NEGATIVE = "worst quality, blurry, distorted, motion blur, text, watermark, logo, extra limbs"
CROP_SIDE = 0.5                                # the default crop: a square of half the picture's width ...
CROP_CENTRE = (0.5, 0.58)                      # ... on the clothes below the face (LTX framed "full body" as a
                                               # medium shot in the first test: centre 0.45 cut through the mouth)
REF_PREFIX = "sref-"                           # queue refs of these jobs: sref-<style>-<hex>
MAX_SEEDS = 8


class StyleRefError(styles.StyleError):
    pass


# ---------------------------------------------------------------- prompt
def trigger_word(st: dict) -> str | None:
    """The LoRA's trigger word: the style record's lora.trigger, else the file name's stem when it is "<word>_<steps>"
    (a training checkpoint such as 3dsrx_1250 -> "3dsrx"); None without a LoRA or a recognizable word."""
    lora = st.get("lora") or {}
    if not lora.get("name"):
        return None
    if str(lora.get("trigger") or "").strip():
        return str(lora["trigger"]).strip()
    m = re.match(r"^((?=[a-z0-9]*[a-z])[a-z0-9]{3,})_\d{3,}$", Path(str(lora["name"])).stem, re.I)
    return m.group(1) if m else None


def prompt_for(st: dict, subject: str | None = None) -> str:
    """The clip's text: the trigger word, the style's words, the subject in the style's light, a still camera."""
    subject = " ".join((subject or SUBJECT).split()).rstrip(".")
    if not subject:
        raise StyleRefError("say what the picture shows (the subject)")
    if len(subject) > 400:
        raise StyleRefError("the subject is too long (at most 400 characters)")
    words = (st.get("words") or "").strip().rstrip(".")
    light = (st.get("light") or "").strip().rstrip(".")
    trig = trigger_word(st)
    head = f"{trig}, {words}" if trig else words
    scene = f"{subject}, {light}" if light else subject
    return (f"{head}. A still shot of {scene}. Nothing moves and the camera does not move; the whole figure is in "
            f"focus.")


# ---------------------------------------------------------------- frames and crops
def sharpness(path: str | Path) -> float:
    """The variance of the Laplacian of the grey picture (higher = sharper)."""
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise StyleRefError(f"cannot read {path}")
    return float(cv2.Laplacian(im, cv2.CV_64F).var())


def sharpest(files: list[str | Path]) -> tuple[Path, float]:
    """The sharpest of the frames (the first one on a tie) and its score."""
    if not files:
        raise StyleRefError("no frames to choose from")
    scored = [(sharpness(f), i, Path(f)) for i, f in enumerate(files)]
    best = max(scored, key=lambda t: (t[0], -t[1]))
    return best[2], best[0]


def default_crop(width: int, height: int, side: float = CROP_SIDE, centre: tuple[float, float] = CROP_CENTRE) -> list[int]:
    """A square of `side` x the picture's shorter side around `centre` (fractions), kept inside the picture."""
    s = max(32, min(int(round(side * min(width, height))), width, height))
    x0 = min(max(0, int(round(centre[0] * width - s / 2))), width - s)
    y0 = min(max(0, int(round(centre[1] * height - s / 2))), height - s)
    return [x0, y0, x0 + s, y0 + s]


def crop_file(src: str | Path, box: list[int], dst: str | Path) -> Path:
    import cv2
    im = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if im is None:
        raise StyleRefError(f"cannot read {src}")
    x0, y0, x1, y1 = styles.check_crop(box, src)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), im[y0:y1, x0:x1])
    return Path(dst)


def reference_for_qwen(name: str) -> Path | None:
    """The picture a Qwen style conversion gets as Picture 2: the style's reference CROPPED to its crop box (cached
    in <home>/cache/style_crops), or the whole reference when it has no box (an uploaded tight crop); None without
    a reference."""
    st = styles.load(name)
    ref = st.get("reference")
    if not ref or not Path(ref).is_file():
        return None
    box = st.get("reference_crop")
    if not box:
        return Path(ref)
    from .runner import home
    key = hashlib.sha1(f"{Path(ref).resolve()}|{Path(ref).stat().st_mtime_ns}|{box}".encode()).hexdigest()[:12]
    out = home() / "cache" / "style_crops" / f"{name}_{key}.png"
    if not out.is_file():
        tmp = out.with_name(f".{out.stem}.{secrets.token_hex(3)}.png")
        crop_file(ref, box, tmp)
        tmp.replace(out)
    return out


# ---------------------------------------------------------------- candidates
def style_dir(name: str) -> Path:
    styles.load(name)                                  # a known style (refuses a bad name)
    d = styles.user_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def candidates_dir(name: str) -> Path:
    d = style_dir(name) / "candidates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_path(name: str) -> Path:
    return candidates_dir(name) / "index.json"


def candidates(name: str) -> list[dict]:
    """The candidates made for this style, newest first (files that were removed are left out)."""
    try:
        rows = json.loads(_index_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rows = []
    rows = [r for r in rows if isinstance(r, dict) and Path(str(r.get("file") or "")).is_file()]
    return sorted(rows, key=lambda r: -float(r.get("made") or 0))


def _add_candidates(name: str, new: list[dict]) -> None:
    p = _index_path(name)
    rows = candidates(name) + new
    tmp = p.with_name(f".index.{secrets.token_hex(3)}.tmp")
    tmp.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    tmp.replace(p)


def parse_seeds(seeds) -> list[int]:
    if seeds is None or seeds == "":
        return list(SEEDS)
    if isinstance(seeds, str):
        seeds = [s for s in seeds.replace(" ", "").split(",") if s]
    try:
        out = [int(s) for s in seeds]
    except (TypeError, ValueError) as exc:
        raise StyleRefError("seeds are whole numbers, for example 202,303,404,505") from exc
    if not 1 <= len(out) <= MAX_SEEDS:
        raise StyleRefError(f"give 1 to {MAX_SEEDS} seeds")
    if len(set(out)) != len(out):
        raise StyleRefError("each seed once")
    return out


def job_values(st: dict, subject: str | None, seed: int, recipe: str, width: int = WIDTH, height: int = HEIGHT,
               frames: int = FRAMES) -> dict:
    """The graph values of one candidate (the txt2img binding)."""
    from . import runner
    sig = runner.dev_sigmas(width, height, frames) if recipe == "full" else runner.DISTILLED_SIGMAS
    lora = st.get("lora") or {}
    v = {"prompt": prompt_for(st, subject), "negative": NEGATIVE, "width": width, "height": height, "frames": frames,
         "seed": int(seed), "sigmas": ", ".join(f"{x:.6g}" for x in sig)}
    if lora.get("name"):
        v |= {"style_lora": lora["name"], "style_lora_strength": float(lora.get("strength", 0.4))}
    return v


def make_reference(name: str, subject: str | None = None, seeds=None, profile: str | None = None,
                   on_event: Callable[[dict], None] | None = None, backend_factory: Callable[[], Any] | None = None,
                   should_stop: Callable[[], bool] | None = None) -> dict:
    """Candidates for the style's reference picture (GPU jobs: the caller holds a queue turn, cli `style
    make-reference`): one still clip per seed with LTX-2.5 (default the distilled model) + the style's LoRA, from
    text; the sharpest saved frame of each is kept in <home>/styles/NAME/candidates/. Returns the new candidates."""
    from . import character
    from . import models as _models
    from . import runner
    from .workflow import ProfileWorkflow
    st = styles.load(name)
    seeds = parse_seeds(seeds)
    prompt_for(st, subject)                            # refuse a bad subject before any GPU work
    prof = profile or runner.LAYOUT_PROFILE
    wf = ProfileWorkflow(prof, VARIANT, models=_models.job_bundle())
    recipe = wf.capabilities.get("recipe") or "distilled"
    frames_out = wf.bind["outputs"]["frames"]
    main = wf.bind["outputs"]["image"]
    cdir = candidates_dir(name)
    batch = time.strftime("%Y%m%d-%H%M%S")
    made = []
    for i, seed in enumerate(seeds, 1):
        if should_stop and should_stop():
            break
        values = job_values(st, subject, seed, recipe)
        work = cdir / ".work" / f"{batch}_s{seed}"
        files = {node: work / f"{node}.png" for node in frames_out}
        r = character.image_job(prof, VARIANT, values, {}, files[main], f"style_{name}_s{seed}",
                                backend_factory=backend_factory, on_event=on_event, item=(i, len(seeds)),
                                extra_outputs={n: f for n, f in files.items() if n != main})
        best, score = sharpest([files[n] for n in frames_out])
        frame = frames_out.index(best.stem)
        dst = cdir / f"ref_{batch}_s{seed}.png"
        shutil.copyfile(best, dst)
        shutil.rmtree(work, ignore_errors=True)
        row = {"file": str(dst), "seed": int(seed), "frame": frame, "sharpness": round(score, 1),
               "subject": " ".join((subject or SUBJECT).split()), "prompt": values["prompt"],
               "lora": st.get("lora"), "profile": prof, "style_version": int(st.get("version", 1)),
               "made": time.time(), "seconds": r.get("seconds")}
        made.append(row)
        _add_candidates(name, [row])
        if on_event:
            on_event({"event": "candidate", **{k: row[k] for k in ("file", "seed", "frame", "sharpness")}})
    shutil.rmtree(cdir / ".work", ignore_errors=True)
    return {"style": name, "candidates": made, "folder": str(cdir),
            "next": f"look at the pictures, then: style use-reference {name} FILE [--crop x0,y0,x1,y1]"}


def use_reference(name: str, file: str | Path, crop=None) -> dict:
    """Make `file` the style's reference picture (instant; a built-in style gets the user's copy). The whole picture
    is kept and a crop box is stored with it: `crop` ([x0, y0, x1, y1] or "x0,y0,x1,y1", pixels), default
    default_crop (a square on the clothes below the face). Qwen conversions get the crop (reference_for_qwen)."""
    import cv2
    src = Path(file)
    if not src.is_file():
        raise StyleRefError(f"no picture {src}")
    im = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if im is None:
        raise StyleRefError(f"{src} is not a picture that can be read")
    h, w = im.shape[:2]
    box = styles.check_crop(crop, src) if crop not in (None, "", []) else default_crop(w, h)
    styles.load(name)
    stored = styles.store_reference(name, src)
    try:
        s = styles.save(name, {"reference": str(stored), "reference_crop": box, "reference_source": str(src.resolve())})
    except BaseException:
        stored.unlink(missing_ok=True)
        raise
    styles.drop_references(name, keep=s.get("reference"))
    return {"style": s, "reference": s["reference"], "crop": box, "crop_file": str(reference_for_qwen(name))}


# ---------------------------------------------------------------- the queue (app and MCP)
def job_record_path(name: str) -> Path:
    return style_dir(name) / "reference_job.json"


def record_job(name: str, **fields) -> dict:
    p = job_record_path(name)
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rec = {}
    if fields.get("ref") and rec.get("ref") != fields["ref"]:
        rec = {}
    rec.update(fields, t=time.time())
    tmp = p.with_name(f".reference_job.{secrets.token_hex(3)}.tmp")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    tmp.replace(p)
    return rec


def job_status(name: str) -> dict | None:
    """The style's reference job: "queued" / "running" (its queue ticket, with the position), "stopping", or the last
    record ("done", "failed", "canceled"); None when none was made."""
    from . import lab, workqueue
    prefix = f"{REF_PREFIX}{name}-"
    for i, t in enumerate(workqueue.tickets()):
        ref = str(t.get("ref") or "")
        if t.get("kind") == "lab" and ref.startswith(prefix) and re.fullmatch(r"[0-9a-f]+", ref[len(prefix):]):
            status = "running" if i == 0 and t.get("pid") else "queued"
            if lab.canceled(ref):
                status = "stopping"
            return {"ref": ref, "status": status, "position": i, "progress": lab.progress(ref)}
    try:
        rec = json.loads(job_record_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if rec.get("status") in ("queued", "running"):    # the worker is gone without its last word
        rec["status"] = "interrupted"
    return rec


def profile_of(video_model: str | None) -> str | None:
    """"distilled" / "full" -> the graph profile (None = the default, the distilled model)."""
    if not video_model:
        return None
    from .model_pairs import VIDEO_MODELS
    if video_model not in VIDEO_MODELS:
        raise StyleRefError(f"video model must be one of {sorted(VIDEO_MODELS)}")
    return VIDEO_MODELS[video_model]


def submit(name: str, subject: str | None = None, seeds=None, via: str = "app", video_model: str | None = None) -> dict:
    """Queue a make_reference job (a detached `cinestudio style make-reference ... --ref sref-NAME-hex` worker; its
    ticket is added now, in request order). Returns at once with the ref."""
    from . import lab, spawn, workqueue
    st = styles.load(name)
    seeds = parse_seeds(seeds)
    prompt_for(st, subject)
    profile_of(video_model)
    cur = job_status(name)
    if cur and cur.get("status") in ("queued", "running", "stopping"):
        raise StyleRefError(f"a reference job for {name} is in the queue already")
    ref = f"{REF_PREFIX}{name}-{secrets.token_hex(3)}"
    label = f"style {st.get('label') or name}: reference pictures with the video model ({len(seeds)} seeds)"
    log = style_dir(name) / f"{ref}.log"
    args = [sys.executable, "-m", "cinestudio", "style", "make-reference", name, "--seeds",
            ",".join(map(str, seeds)), "--ref", ref]
    if subject and subject.strip():
        args += [f"--subject={subject.strip()}"]
    if video_model:
        args += ["--video-model", video_model]
    (lab.lab_dir() / f"{ref}.cancel").unlink(missing_ok=True)
    workqueue.add_new("lab", ref, label)
    record_job(name, ref=ref, status="queued", via=via, seeds=seeds, subject=(subject or SUBJECT).strip(),
               created=time.time())
    try:
        from .cli import ROOT
        pid, independent = spawn.detached(args, log, ROOT)
    except BaseException as exc:
        workqueue.done("lab", ref)
        record_job(name, ref=ref, status="failed", error=f"the worker did not start: {exc}")
        raise
    return {"ref": ref, "pid": pid, "style": name, "seeds": seeds,
            **({} if independent else {"warning": "the worker may stop when this app stops"})}


def run_job(name: str, subject: str | None, seeds, ref: str | None,
            on_event: Callable[[dict], None] | None = None, video_model: str | None = None) -> dict:
    """The body of the CLI command: wait for the queue turn (kind "lab"), make the candidates, keep a record."""
    from . import lab, workqueue
    st = styles.load(name)
    seeds = parse_seeds(seeds)
    ref = ref or f"{REF_PREFIX}{name}-{secrets.token_hex(3)}"
    label = f"style {st.get('label') or name}: reference pictures with the video model ({len(seeds)} seeds)"
    record_job(name, ref=ref, status="queued", seeds=seeds, subject=(subject or SUBJECT).strip())
    try:
        with lab.turn(ref, label):
            record_job(name, ref=ref, status="running", started=time.time())
            r = make_reference(name, subject, seeds, profile=profile_of(video_model), on_event=on_event,
                               should_stop=lambda: lab.canceled(ref))
    except workqueue.QueueCanceled:
        record_job(name, ref=ref, status="canceled", ended=time.time())
        raise
    except BaseException as exc:
        record_job(name, ref=ref, status="failed", error=f"{type(exc).__name__}: {exc}", ended=time.time())
        raise
    stopped = lab.canceled(ref) and len(r["candidates"]) < len(seeds)
    record_job(name, ref=ref, status="canceled" if stopped else "done", ended=time.time(),
               files=[c["file"] for c in r["candidates"]])
    return r
