"""Video helpers: slice control videos into clips, take the last frame, probe, stitch.

All encoding uses the ffmpeg binary bundled with imageio-ffmpeg (has libx264rgb), so no system
ffmpeg is needed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg


def ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


BLEND = "flow"    # cross-fades of joins and loops: "flow" (the pictures move toward each other, then mix) or "fade"
FFMPEG_TIMEOUT = 3 * 3600         # no ffmpeg job of ours runs this long; a stuck one must not fill the disk


def _run(args: list[str]) -> None:
    # -nostdin + stdin=DEVNULL: ffmpeg reads stdin for key commands; on an open pipe (a background shell or worker) it
    # can block forever (a test hung on a two-input mix)
    subprocess.run([ffmpeg(), "-nostdin", "-loglevel", "error", "-y", *args], check=True, stdin=subprocess.DEVNULL,
                   timeout=FFMPEG_TIMEOUT)


@dataclass
class Probe:
    frames: int
    width: int
    height: int
    fps: float


def probe(path: str | Path) -> Probe:
    """Count decoded frames (exact) and read size and frame rate."""
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    p = Probe(n, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              round(cap.get(cv2.CAP_PROP_FPS), 3))
    cap.release()
    return p


def check_clip(path: str | Path, frames: int, width: int, height: int, fps: float,
               audio: bool | None = None) -> list[str]:
    """Problems of a generated clip against the requested format (empty list = all correct)."""
    p, problems = probe(path), []
    if p.frames != frames:
        problems.append(f"{p.frames} frames, expected {frames}")
    if (p.width, p.height) != (width, height):
        problems.append(f"{p.width}x{p.height}, expected {width}x{height}")
    if abs(p.fps - fps) > 0.01:
        problems.append(f"{p.fps} fps, expected {fps}")
    if audio is not None and has_audio(path) != audio:
        problems.append("audio missing" if audio else "unexpected audio")
    return problems


def contact_sheet(src: str | Path, dst: str | Path, picks: int = 4, extra: str | Path | None = None,
                  height: int = 288) -> None:
    """One row of `picks` evenly spaced frames (plus an optional reference image first) as JPEG."""
    import numpy as np
    cap = cv2.VideoCapture(str(src))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    idx = [round(i * (len(frames) - 1) / max(1, picks - 1)) for i in range(picks)]
    tiles = [cv2.imread(str(extra))] if extra else []
    tiles += [frames[i] for i in idx]
    tiles = [cv2.resize(t, (round(t.shape[1] * height / t.shape[0]), height), interpolation=cv2.INTER_AREA)
             for t in tiles]
    cv2.imwrite(str(dst), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])


def clip_start(k: int, clip_frames: int, overlap: int = 1) -> int:
    """0-based first shot frame of clip k (1-based). Consecutive clips share `overlap` frames:
    clip k+1 starts on the first of clip k's last `overlap` frames."""
    return (clip_frames - overlap) * (k - 1)


def n_clips(shot_frames: int, clip_frames: int, overlap: int = 1) -> int:
    return max(1, -(-(shot_frames - overlap) // (clip_frames - overlap)))


def shot_length(n: int, clip_frames: int, overlap: int = 1) -> int:
    return (clip_frames - overlap) * n + overlap


def slice_control(src: str | Path, dst: str | Path, k: int, clip_frames: int, width: int, height: int,
                  fps: float, overlap: int = 1) -> None:
    """Clip k of a lossless control video, scaled to fill the generation size (the overflow is cut
    equally on both sides; never stretched, so a pose keeps its proportions), lossless. Holds the
    last frame when the shot is shorter than the clip. The source must have the clip frame rate:
    the start is counted in source frames, so another rate would put the clip at the wrong time."""
    src_fps = probe(src).fps
    if abs(src_fps - fps) > 0.01:
        raise ValueError(f"{src}: {src_fps} fps, the clips are {fps} fps (render the passes at the clip rate)")
    start = clip_start(k, clip_frames, overlap)
    # fps= before tpad: after trim the stream has no fixed rate, and tpad then makes clones with
    # broken timestamps that the encoder drops (a short last clip came out with 8 frames, not 121).
    _run(["-i", str(src), "-vf",
          f"trim=start_frame={start},setpts=PTS-STARTPTS,fps={fps},"
          f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=area,crop={width}:{height},"
          f"tpad=stop_mode=clone:stop={clip_frames}",
          "-frames:v", str(clip_frames), "-c:v", "libx264rgb", "-crf", "0", "-preset", "veryfast", str(dst)])


def frame_png(src: str | Path, dst: str | Path, index: int, width: int | None = None, height: int | None = None) -> None:
    """Frame `index` (0-based) of a video as PNG, optionally scaled. The scaler's accurate flags matter: the default
    fast YUV->RGB path read a ComfyUI clip (BT.709, limited range) 0.5 L darker than the original picture, and every
    chained look took that loss (2026-10-23); with them a round trip returns the original (48.34 vs 48.33 L). The
    file's own colour tags are used (no forced matrix)."""
    if (width is None) != (height is None):
        raise ValueError("frame_png: give both width and height, or neither")
    flags = "accurate_rnd+full_chroma_int+bitexact"
    vf = f"select='eq(n\\,{index})'," + (f"scale={width}:{height}:flags=area+{flags}" if width else f"scale=flags={flags}")
    _run(["-i", str(src), "-vf", vf + ",format=rgb24", "-frames:v", "1", str(dst)])


def sharpness(img) -> float:
    """Laplacian variance of a picture (BGR array or a file): the sharpness measure of `run score`."""
    import cv2
    f = cv2.imread(str(img)) if not hasattr(img, "shape") else img
    return float(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def sharpen_to(path: str | Path, target: float, max_amount: float, sigma: float = 2.0) -> dict:
    """An unsharp mask in place with about the smallest amount (0..max_amount; bisection, 12 steps) whose sharpness
    REACHES `target` (it may end slightly above it); max_amount when even that stays below; none when the picture is
    already as sharp. A fixed amount on every chained frame added up from clip to clip (2026-10-23: the chained look
    2-4x sharper than its source frame, each clip 1.5-2x sharper than the one before)."""
    import cv2
    f = cv2.imread(str(path))
    blur = cv2.GaussianBlur(f, (0, 0), sigma)
    before = sharpness(f)
    lo, hi, best = 0.0, float(max_amount), 0.0
    if before < target and max_amount > 0:
        out = cv2.addWeighted(f, 1 + hi, blur, -hi, 0)
        if sharpness(out) < target:
            best = hi
        else:
            for _ in range(12):
                mid = (lo + hi) / 2
                if sharpness(cv2.addWeighted(f, 1 + mid, blur, -mid, 0)) < target:
                    lo = mid
                else:
                    hi = mid
            best = hi
    if best > 0 and not cv2.imwrite(str(path), cv2.addWeighted(f, 1 + best, blur, -best, 0)):
        raise OSError(f"cannot write {path}")
    return {"amount": round(best, 3), "before": round(before, 1), "target": round(float(target), 1),
            "after": round(sharpness(cv2.imread(str(path))), 1)}


def _lab_float(img):
    import cv2
    import numpy as np
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)      # L 0..100, a/b in their units


def _region(mask_path, shape, grow: int = 12) -> tuple:
    """The background as a boolean array: everything outside the character's mask grown by `grow` px on each side
    (a (2*grow+1) kernel). -> (array, reason): reason None = a real background; "no mask" / "mask unreadable" /
    "too little background" = the whole picture (a fallback the caller must see)."""
    import cv2
    import numpy as np
    whole = np.ones(shape[:2], bool)
    if not mask_path:
        return whole, "no mask"
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return whole, "mask unreadable"
    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 127
    k = 2 * int(grow) + 1
    m = cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    bg = ~m
    return (bg, None) if bg.mean() > 0.2 else (whole, "too little background")


def background_lightness(img, mask_path=None) -> tuple:
    """The mean Lab L (0..100) of a picture's background (outside the grown character mask). -> (L, fallback reason)."""
    import cv2
    f = cv2.imread(str(img)) if not hasattr(img, "shape") else img
    if f is None:
        raise ValueError(f"cannot read {img}")
    bg, why = _region(mask_path, f.shape)
    return float(_lab_float(f)[..., 0][bg].mean()), why


def match_lightness(path: str | Path, ref: str | Path, max_gain: float = 0.10, mask: str | Path | None = None,
                    ref_mask: str | Path | None = None, ratio: float = 1.0, tone: bool = False) -> dict:
    """Scale the picture's lightness (Lab L, colours kept) in place toward the reference's BACKGROUND mean x `ratio`
    (2026-10-23, user: "the scene slowly gets darker"): each clip came out ~1.3 L darker than its frame -1 look and the
    next clip took its look from it, so the scene lost ~1.5-2 L at every join. `mask`/`ref_mask` = the character's
    masks (the background is measured outside them, grown); `ratio` = the layout render's own background change between
    the two moments (a scene that should get darker, a moving camera). The gain is solved WITH the soft roll-off above
    L 75 (brightening fades toward white, monotonic for gains up to 1.125) and limited to +-max_gain; the result is
    measured after the colour conversion (residual_L, target_met). Nothing changes when the background regions fall
    back to whole pictures in different ways, or the ratio is not usable.
    `tone` (chain_light "tone", 2026-10-23: the background lost ~8-10% of its colourfulness and some contrast at every
    join in every run, read as "hazy, bluer" by a blind critic): also the background's mean chroma (a*, b* scaled
    together: no hue change) and its lightness spread (around its mean) toward the reference's, within +-15%."""
    import cv2
    import numpy as np
    f = cv2.imread(str(path))
    r = cv2.imread(str(ref))
    if f is None or r is None:
        raise ValueError(f"cannot read {path if f is None else ref}")
    if not (np.isfinite(max_gain) and max_gain >= 0):
        raise ValueError("max_gain must be a number >= 0")
    max_gain = min(float(max_gain), 0.125)
    lab = _lab_float(f)
    bg, why = _region(mask, f.shape)
    rbg, rwhy = _region(ref_mask, r.shape)
    before = float(lab[..., 0][bg].mean())
    base = {"before_L": round(before, 2), "background_share": round(float(bg.mean()), 3),
            "fallback": why or rwhy, "ratio": round(float(ratio), 4)}
    if why != rwhy or not (np.isfinite(ratio) and 0.5 < ratio < 2.0):
        return {**base, "gain": 1.0, "skipped": "background regions differ or the layout ratio is not usable"}
    target = float(_lab_float(r)[..., 0][rbg].mean()) * float(ratio)
    L = lab[..., 0]
    soft = 1.0 - np.clip((L - 75.0) / 25.0, 0.0, 1.0) ** 2
    if target >= before:                                   # brightening: the roll-off is part of the equation
        denom = float((L * soft)[bg].mean())
        raw = 1.0 + (target - before) / denom if denom > 1e-6 else 1.0
    else:                                                  # darkening: a plain gain
        raw = target / max(1e-3, before)
        soft = np.ones_like(L)
    gain = float(np.clip(raw, 1 - max_gain, 1 + max_gain))
    lab[..., 0] = np.clip(L + (L * gain - L) * soft, 0.0, 100.0)
    toned = {}
    if tone:
        rl = _lab_float(r)
        def chroma(x, m):
            return float(np.hypot(x[..., 1], x[..., 2])[m].mean())
        c_gain = float(np.clip(chroma(rl, rbg) / max(1e-3, chroma(lab, bg)), 0.85, 1.15))
        lab[..., 1] *= c_gain
        lab[..., 2] *= c_gain
        Lb = lab[..., 0]
        mean_bg, sd_bg = float(Lb[bg].mean()), float(Lb[bg].std())
        s_gain = float(np.clip(float(rl[..., 0][rbg].std()) / max(1e-3, sd_bg), 0.85, 1.15))
        lab[..., 0] = np.clip(mean_bg + (Lb - mean_bg) * s_gain, 0.0, 100.0)
        toned = {"chroma_gain": round(c_gain, 4), "contrast_gain": round(s_gain, 4)}
    out = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) * 255.0, 0, 255).round().astype(np.uint8)
    if not cv2.imwrite(str(path), out):
        raise OSError(f"cannot write {path}")
    after = float(_lab_float(cv2.imread(str(path)))[..., 0][bg].mean())
    return {**base, "gain": round(gain, 4), "clamped": bool(abs(raw - gain) > 1e-6), "target_L": round(target, 2),
            "after_L": round(after, 2), "residual_L": round(after - target, 2), "target_met": abs(after - target) < 0.3,
            **toned}


def unsharp_png(path: str | Path, amount: float, sigma: float = 2.0) -> Path:
    """An unsharp mask in place: picture + amount * (picture - blurred)."""
    import cv2
    f = cv2.imread(str(path))
    blur = cv2.GaussianBlur(f, (0, 0), sigma)
    cv2.imwrite(str(path), cv2.addWeighted(f, 1 + amount, blur, -amount, 0))
    return Path(path)


def last_frames(src: str | Path, dst: str | Path, n: int) -> None:
    """The last n frames of a video as a lossless video (a continuation guide)."""
    total = probe(src).frames
    _run(["-i", str(src), "-vf", f"trim=start_frame={total - n},setpts=PTS-STARTPTS,"   # accurate YUV->RGB (frame_png)
          "scale=flags=accurate_rnd+full_chroma_int+bitexact,format=rgb24", "-an",
          "-frames:v", str(n), "-c:v", "libx264rgb", "-crf", "0", "-preset", "veryfast", str(dst)])


def image_video(src: str | Path, dst: str | Path, fps: float) -> None:
    """A one-frame lossless video of a picture (a single first frame for a graph that loads first frames as video)."""
    _run(["-loop", "1", "-t", f"{2 / fps:.4f}", "-i", str(src), "-frames:v", "1", "-r", str(fps), "-c:v", "libx264rgb",
          "-crf", "0", "-preset", "veryfast", str(dst)])          # -t: a looped input must end (an unreadable picture)


def last_frame(src: str | Path, dst: str | Path) -> None:
    """The exact last decoded frame of a video, as PNG."""
    n = probe(src).frames
    frame_png(src, dst, n - 1)


def has_audio(path: str | Path) -> bool:
    r = subprocess.run([ffmpeg(), "-nostdin", "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    return "Audio:" in r.stderr


def _flow_mix(a, b, t: float, dis, width: int = 768):
    """The picture between frames `a` (t=0) and `b` (t=1) of the SAME moment from two generations: optical flow (DIS)
    between them, both warped toward time t (the intermediate flows of Super SloMo, Jiang et al. 2018), then mixed.
    A plain cross-fade of two pictures whose arms are in other places shows two pairs of arms (a soft, doubled look);
    here the arms move from one place to the other."""
    import numpy as np
    h, w = a.shape[:2]
    s = min(1.0, width / w)
    small = (max(8, int(w * s)), max(8, int(h * s)))
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), small, interpolation=cv2.INTER_AREA)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), small, interpolation=cv2.INTER_AREA)
    f01 = cv2.resize(dis.calc(ga, gb, None), (w, h)) / s
    f10 = cv2.resize(dis.calc(gb, ga, None), (w, h)) / s
    ft0 = -(1 - t) * t * f01 + t * t * f10
    ft1 = (1 - t) ** 2 * f01 - t * (1 - t) * f10
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    wa = cv2.remap(a, gx + ft0[..., 0], gy + ft0[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wb = cv2.remap(b, gx + ft1[..., 0], gy + ft1[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(wa, 1 - t, wb, t, 0)


def morph(a_frames: list, b_frames: list) -> list:
    """Frame j of a cross-fade from `a_frames` to `b_frames` (equal lengths k, the same moments): the flow mix at
    t = (j+1)/(k+1), the same weights as the plain fade."""
    k = len(a_frames)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return [_flow_mix(a, b, (j + 1) / (k + 1), dis) for j, (a, b) in enumerate(zip(a_frames, b_frames))]


def _frames(path: str | Path, keep) -> dict:
    """The decoded frames whose index is in `keep` (a set), by index."""
    cap, out, i = cv2.VideoCapture(str(path)), {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in keep:
            out[i] = f
        i += 1
    cap.release()
    return out


def _write_frames(frames, dst: str | Path, fps: float, size: tuple[int, int], audio: str | Path | None = None,
                  seconds: float | None = None, lossless: bool = False) -> None:
    """Encode BGR frames (an iterable) to `dst`; the sound of `audio` (a video or sound file) is muxed when given."""
    w, h = size
    cmd = [ffmpeg(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-"]
    if audio is not None:
        cmd += ["-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
    if seconds is not None:
        cmd += ["-t", f"{seconds:.6f}"]
    cmd += (["-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast"] if lossless else
            ["-c:v", "libx264", "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p"])
    proc = subprocess.Popen(cmd + ["-r", str(fps), str(dst)], stdin=subprocess.PIPE)
    try:
        for f in frames:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        if proc.wait(timeout=FFMPEG_TIMEOUT):
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    finally:
        if proc.poll() is None:
            proc.kill()


def _morphed_tails(clips: list, probes: list, overlap: int, tmp: Path) -> list[Path]:
    """Copies of the clips (lossless, no sound) in which the last `overlap` frames of each clip but the last are the
    flow cross-fade into the next clip's first `overlap` frames. Joined with the next clip's first `overlap` frames
    cut, they give the same frame count and timing as the plain cross-fade."""
    out = []
    for i, c in enumerate(clips):
        if i == len(clips) - 1:
            out.append(Path(c))
            continue
        n = probes[i].frames
        tail = _frames(c, set(range(n - overlap, n)))
        head = _frames(clips[i + 1], set(range(overlap)))
        mixed = morph([tail[j] for j in range(n - overlap, n)], [head[j] for j in range(overlap)])

        def gen(c=c, n=n, mixed=mixed):
            cap = cv2.VideoCapture(str(c))
            for _ in range(n - overlap):
                ok, f = cap.read()
                if not ok:
                    break
                yield f
            cap.release()
            yield from mixed
        dst = tmp / f"join{i}.mkv"
        _write_frames(gen(), dst, probes[i].fps, (probes[i].width, probes[i].height), lossless=True)
        out.append(dst)
    return out


AUDIO_SWITCH_BEFORE = 0.1      # audio_join "carry": seconds before the end of the shared frames where clip k takes over
AUDIO_SWITCH_FADE = 0.04       # and the fade there (Lightricks' AUDIO_SEAM_CROSSFADE_MS, ltx-pipelines 1.4.0)


def stitch(clips: list[str | Path], dst: str | Path, fps: float, total_frames: int | None = None,
           overlap: int = 1, blend: str | None = None, audio_join: str | list[str] = "fade") -> None:
    """Join clips that share `overlap` frames: each clip after the first starts on frames the
    previous clip already has, so its first `overlap` frames are dropped (and the same time from
    its audio). Works with and without audio.

    Each audio part is cut or padded with silence to the exact length of its video part, so a clip
    whose audio is longer or shorter than its video cannot shift the later clips. If only some
    clips have audio, the others get silence. All clips must have the same size and frame rate.

    With an overlap of several frames the pictures cross-fade over it: `blend` "flow" (default BLEND) warps both
    pictures toward each other first (no doubled arms), "fade" is a plain cross-fade.

    `audio_join` "fade" (independent clips): an equal-power cross-fade over the shared time. "carry" (the audio carry:
    clip k's sound STARTS with clip k-1's frozen tail, so both play the same sound there; an equal-power fade of two
    equal signals is up to 3 dB too loud): clip k-1's sound until AUDIO_SWITCH_BEFORE s before the end of the shared
    time, then clip k (its causal decoder has had ~1 s of the tail as context), with an AUDIO_SWITCH_FADE s fade.
    A list gives one mode PER JOIN (len(clips) - 1): a take put in for one clip changes only its own joins."""
    modes = [audio_join] * max(0, len(clips) - 1) if isinstance(audio_join, str) else list(audio_join)
    if len(modes) != max(0, len(clips) - 1) or any(x not in ("fade", "carry") for x in modes):
        raise ValueError(f"audio_join must be 'fade', 'carry' or one of them per join, not {audio_join!r}")
    audio_join = modes
    probes = [probe(c) for c in clips]
    for c, p in zip(clips, probes):
        if abs(p.fps - fps) > 0.01:
            raise ValueError(f"{c}: {p.fps} fps, the shot is {fps} fps")
        if (p.width, p.height) != (probes[0].width, probes[0].height):
            raise ValueError(f"{c}: {p.width}x{p.height}, the first clip is {probes[0].width}x{probes[0].height}")
        if p.frames <= overlap:
            raise ValueError(f"{c}: {p.frames} frames, not more than the overlap {overlap}")
    mode = blend or BLEND
    if mode not in ("flow", "fade"):
        raise ValueError(f"blend must be 'flow' or 'fade', not {mode!r}")
    if overlap > 1 and mode == "flow" and len(clips) > 1:
        import shutil
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix=".stitch_", dir=Path(dst).parent))
        try:
            # the cross-fades are made in the frames (morphed tails); ffmpeg then cuts and joins, and mixes the sound
            _stitch(clips, dst, fps, total_frames, overlap, probes, _morphed_tails(clips, probes, overlap, tmp), False,
                    audio_join)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return
    _stitch(clips, dst, fps, total_frames, overlap, probes, list(clips), overlap > 1, audio_join)


def _stitch(clips: list, dst, fps: float, total_frames: int | None, overlap: int, probes: list, pictures: list,
            blend: bool, audio_join: list[str] | None = None) -> None:
    """ffmpeg part of stitch: the pictures of `pictures` (cut by `overlap`, or cross-faded when `blend`), the sound
    of `clips` (always cross-faded over the shared time)."""
    with_audio = [has_audio(c) for c in clips]
    audio = any(with_audio)
    inputs, parts = [], []
    xfade = overlap / fps                       # the shared frames' time: there the two soundtracks cross-fade
    modes = list(audio_join or ["fade"] * (len(clips) - 1))
    ok = xfade > AUDIO_SWITCH_BEFORE + AUDIO_SWITCH_FADE        # the shared time must hold the switch
    carry = [ok and x == "carry" for x in modes]               # per join i (between clip i and clip i + 1)
    sw = xfade - AUDIO_SWITCH_BEFORE            # carry: the switch, in seconds from the start of the shared time
    half = AUDIO_SWITCH_FADE / 2
    for i, c in enumerate(pictures):
        inputs += ["-i", str(c)]
        if blend:
            parts.append(f"[{i}:v]setpts=PTS-STARTPTS,fps={fps},format=yuv444p,settb=AVTB[v{i}]")
        else:
            skip = overlap if i else 0
            trim_v = f"trim=start_frame={skip}," if skip else ""
            parts.append(f"[{i}:v]{trim_v}setpts=PTS-STARTPTS[v{i}]")
    base = len(pictures) if any(str(a) != str(b) for a, b in zip(pictures, clips)) else 0
    for i, c in enumerate(clips):
        if base and with_audio[i]:
            inputs += ["-i", str(c)]
        if audio:
            # each part keeps its WHOLE audio (also the shared frames); the cross-fade below removes exactly the
            # shared time again, so the total length and the sync are the same as a cut, without the click
            length = probes[i].frames / fps
            # carry: each part only from its switch (sw - half into its start) to the next switch (+ half), so the
            # short fades below give the same total length and sync as the cut
            a0 = (sw - half) if i and carry[i - 1] else 0.0
            a1 = (length - xfade + sw + half) if i < len(clips) - 1 and carry[i] else length
            trim = f"atrim=start={a0:.6f}:end={a1:.6f},asetpts=PTS-STARTPTS"
            if with_audio[i]:
                ai = base + sum(with_audio[:i]) if base else i
                parts.append(f"[{ai}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                             f"asetpts=PTS-STARTPTS,apad,{trim}[a{i}]")
            else:
                parts.append(f"anullsrc=r=48000:cl=stereo,aformat=sample_fmts=fltp,{trim}[a{i}]")
    if blend:                                   # linear cross-fade over the `overlap` frames of each join
        prev, done = "v0", probes[0].frames
        for i in range(1, len(clips)):
            out = "vc" if i == len(clips) - 1 else f"y{i}"
            parts.append(f"[{prev}][v{i}]xfade=transition=fade:duration={xfade:.6f}:"
                         f"offset={(done - overlap) / fps:.6f}[{out}]")
            prev, done = out, done + probes[i].frames - overlap
        if len(clips) == 1:
            parts.append("[v0]null[vc]")
    else:
        vstreams = "".join(f"[v{i}]" for i in range(len(clips)))
        parts.append(f"{vstreams}concat=n={len(clips)}:v=1:a=0[vc]")
    if audio:
        prev = "a0"
        for i in range(1, len(clips)):
            out = "ac" if i == len(clips) - 1 else f"x{i}"
            if carry[i - 1]:                    # the same sound on both sides: a short equal-GAIN fade (no +3 dB)
                parts.append(f"[{prev}][a{i}]acrossfade=d={AUDIO_SWITCH_FADE:.6f}:c1=tri:c2=tri[{out}]")
            else:
                parts.append(f"[{prev}][a{i}]acrossfade=d={xfade:.6f}:c1=qsin:c2=qsin[{out}]")
            prev = out
        if len(clips) == 1:
            parts.append("[a0]anull[ac]")
    args = [*inputs, "-filter_complex", ";".join(parts), "-map", "[vc]"]
    if audio:
        args += ["-map", "[ac]", "-c:a", "aac", "-b:a", "192k"]
    if total_frames:
        args += ["-frames:v", str(total_frames)]
    joined = total_frames or (sum(q.frames for q in probes) - (len(clips) - 1) * overlap)
    args += ["-t", f"{joined / fps:.6f}"]                 # a hard end (the padded sound parts must not run on)
    args += ["-c:v", "libx264", "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p", "-r", str(fps), str(dst)]
    _run(args)


def mix_strips(info: dict, dst: str | Path) -> Path | None:
    """The shot's sound strips (bpy_jobs/export_sound.py's sound.json) mixed into one WAV for the shot's frame range
    (frame_start..frame_end): each strip plays its part (frame_final_start..frame_final_end) at its place, with its
    volume, as Blender plays it. None when no strip plays inside the range."""
    fps = float(info["fps"])
    first, end = int(info["frame_start"]), int(info["frame_end"]) + 1           # end: the frame after the last
    length = (end - first) / fps
    ins, chains = [], []
    for s in info.get("strips") or []:
        a, b = max(int(s["frame_final_start"]), first), min(int(s["frame_final_end"]), end)
        if b <= a:
            continue
        src0 = (a - float(s["frame_start"])) / fps                                 # seconds into the sound file
        delay = int(round((a - first) / fps * 1000))
        k = len(chains)                                 # the input number (each input is 2 argv items)
        ins += ["-i", str(s["file"])]
        chains.append(f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo,"
                      f"atrim=start={src0:.6f}:duration={(b - a) / fps:.6f},asetpts=PTS-STARTPTS,"
                      f"volume={float(s.get('volume', 1.0)):.4f},adelay={delay}:all=1[s{k}]")
    if not chains:
        return None
    n = len(chains)
    mix = "".join(f"[s{k}]" for k in range(n)) + (f"amix=inputs={n}:normalize=0:duration=longest," if n > 1 else "")
    graph = ";".join(chains) + f";{mix}apad,atrim=end={length:.6f}[a]"
    # -t: a hard end for the output (apad pads forever; atrim alone did not always stop ffmpeg: a test run wrote an
    # 8 GB WAV until the disk was full)
    _run([*ins, "-filter_complex", graph, "-map", "[a]", "-t", f"{length:.6f}", "-c:a", "pcm_s16le", str(dst)])
    return Path(dst)


def loop_close(src: str | Path, dst: str | Path, loop_frames: int, tail: int, fps: float,
               blend: str | None = None) -> None:
    """A seamless loop from a video whose frames past `loop_frames` (the tail) repeat its first frames: the head
    (frames 0..tail-1) becomes a cross-fade from the tail to the head, then the video is cut to `loop_frames`.
    Played again and again, frame loop_frames-1 is followed by a frame that is mostly the tail's first frame, the
    frame that really followed it. The sound is cut to the loop (a loop sound is periodic already).
    `blend` "flow" (default BLEND) warps the tail and the head toward each other before they mix; "fade" mixes only."""
    k, n = int(tail), int(loop_frames)
    if (blend or BLEND) == "flow" and k > 0:
        got = _frames(src, set(range(k)) | set(range(n, n + k)))
        if len(got) < 2 * k:
            raise ValueError(f"{src}: fewer than {n + k} frames (loop {n} + tail {k})")
        mixed = morph([got[n + j] for j in range(k)], [got[j] for j in range(k)])
        h, w = mixed[0].shape[:2]

        def gen():
            yield from mixed
            cap = cv2.VideoCapture(str(src))
            for i in range(n):
                ok, f = cap.read()
                if not ok:
                    break
                if i >= k:
                    yield f
            cap.release()
        _write_frames(gen(), dst, fps, (w, h), audio=src if has_audio(src) else None, seconds=n / fps)
        return
    w = f"N/{k + 1}"                                   # the blend filter counts N from 1
    graph = (f"[0:v]split=3[a][b][c];[a]trim=start_frame={n}:end_frame={n + k},setpts=PTS-STARTPTS,format=yuv444p[t];"
             f"[b]trim=end_frame={k},setpts=PTS-STARTPTS,format=yuv444p[h];"
             f"[t][h]blend=all_expr='A*(1-{w})+B*{w}',format=yuv420p[x];"
             f"[c]trim=start_frame={k}:end_frame={n},setpts=PTS-STARTPTS,format=yuv420p[m];[x][m]concat=n=2:v=1:a=0[v]")
    args = ["-i", str(src), "-filter_complex", graph, "-map", "[v]"]
    if has_audio(src):
        args += ["-map", "0:a", "-c:a", "aac", "-b:a", "192k"]
    _run(args + ["-t", f"{n / fps:.6f}", "-frames:v", str(n), "-c:v", "libx264", "-crf", "12", "-preset", "slow",
                 "-pix_fmt", "yuv420p", "-r", str(fps), str(dst)])


def cut_audio(src: str | Path, start: float, dst: str | Path) -> None:
    """The sound file from second `start` to its end, as 48 kHz stereo PCM (replace_audio then fits it to the video)."""
    _run(["-ss", f"{max(0.0, start):.6f}", "-i", str(src), "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(dst)])


def replace_audio(video_in: str | Path, audio: str | Path, dst: str | Path) -> None:
    """The video with `audio` as its only sound: the picture is copied as it is, the sound is padded with silence
    or cut to the video's exact length (so it starts on frame 1 and keeps the shot's sync)."""
    p = probe(video_in)
    length = p.frames / p.fps
    _run(["-i", str(video_in), "-i", str(audio), "-filter_complex",
          f"[1:a]aresample=48000,aformat=channel_layouts=stereo,apad,atrim=end={length:.6f}[a]",
          "-map", "0:v", "-map", "[a]", "-t", f"{length:.6f}", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(dst)])


def image_resize(src: str | Path, dst: str | Path, width: int, height: int) -> None:
    """Exact resize (stretches if the aspect differs). Use only for images already at the shot's aspect."""
    _run(["-i", str(src), "-vf", f"scale={width}:{height}:flags=area", "-frames:v", "1", str(dst)])


def fit_image(src: str | Path, dst: str | Path, width: int, height: int, mode: str = "pad",
              color: str = "black") -> None:
    """Resize without distortion.
    mode 'pad':   the whole image fits inside, borders in `color` (reference sheets: a model that
                  center-crops its reference, such as Wan 2.2 Fun, then crops nothing).
    mode 'cover': the image fills the frame, the overflow is cut equally on both sides (first frames)."""
    if mode == "pad":
        vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
              f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={color}")
    elif mode == "cover":
        vf = (f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
              f"crop={width}:{height}")
    else:
        raise ValueError(f"unknown fit mode {mode!r}")
    _run(["-i", str(src), "-vf", vf, "-frames:v", "1", str(dst)])
