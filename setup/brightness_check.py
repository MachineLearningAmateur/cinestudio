"""Brightness across the clips of runs (2026-10-23, user: "maintain consistent color brightness throughout the clips").

    python setup/brightness_check.py RUN_ID [RUN_ID ...]

Per run and clip: the mean Lab L of the BACKGROUND (outside the grown Blender character mask of the same shot frame)
at the clip's frames 0, 30, 60, 90 and its last frame, read with video.frame_png (the exact decode), then the step at
each join (clip k's first frames against the same shot frames at the end of clip k-1) and the drift (the last clip
against clip 1). The goal: steps under 0.5 L and the last clip within 0.5 L of clip 1."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cinestudio import passes, runner, video  # noqa: E402


def measure(run_id: str) -> dict:
    d = runner.run_dir(run_id)
    st = json.loads((d / "state.json").read_text(encoding="utf-8"))
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    key = m.get("pass_key") or m["blend"]["sha256"][:16]           # as runner.Run.shot_dir
    out = st.get("steps", {}).get("background", {}).get("outputs") or {}
    bg = out.get("sha256")
    ground = "" if out.get("ground", "plane") == "plane" else "-" + out["ground"]
    pass_dir = runner.home() / "shots" / (f"{key}-bg{bg[:10]}{ground}" if bg else key) / "passes"
    info = passes.load(pass_dir) if (pass_dir / "passes.json").is_file() else {"mask": []}
    masks = info.get("mask") or []
    fmt = st["steps"]["controls"]["outputs"]
    frames, ov = int(fmt["frames"]), int(fmt.get("overlap", 1))
    parts = st["steps"]["clip"].get("parts") or {}
    clips = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in sorted(int(x) for x in parts):
            clip = Path((parts[str(k)].get("outputs") or {}).get("clip") or "")
            if not clip.is_file():
                continue
            n = video.probe(clip).frames
            start = video.clip_start(k, frames, ov)
            vals = {}
            for i in sorted({0, 30, 60, 90, n - 1, ov - 1}):
                if i >= n:
                    continue
                png = Path(tmp) / f"c{k}_{i}.png"
                video.frame_png(clip, png, i)
                mk = masks[start + i] if start + i < len(masks) else None
                vals[i] = round(video.background_lightness(png, mk)[0], 2)
            clips.append({"clip": k, "L": vals})
    steps = []
    for a, b in zip(clips, clips[1:]):
        # the end of clip k-1 (its frame frames-ov .. last) is the same moment as clip k's frames 0 .. ov-1
        end_prev = a["L"].get(frames - 1) or a["L"][max(a["L"])]
        first = b["L"].get(ov - 1) or b["L"][0]
        steps.append(round(first - end_prev, 2))
    mids = [c["L"].get(60) for c in clips if c["L"].get(60) is not None]
    return {"run": run_id, "note": (json.loads((d / "note.json").read_text(encoding="utf-8")).get("text", "")[:90]
                                    if (d / "note.json").is_file() else ""),
            "chain_light": m.get("chain_light"), "background": m.get("background"),
            "clip_mid_L": mids, "join_steps": steps, "drift_last_vs_first": round(mids[-1] - mids[0], 2) if mids else None,
            "clips": clips}


if __name__ == "__main__":
    for rid in sys.argv[1:]:
        print(json.dumps(measure(rid)))
