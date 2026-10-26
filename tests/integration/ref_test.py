"""One clip with a reference image: does the AI show MY character with the placeholder's pose and camera?

Passes only if the job succeeds AND the clip has the requested frames, size and frame rate (and audio
for profiles that make audio). Fidelity to the reference cannot be measured by a script: the test
writes contact_sheet.jpg (reference + 4 frames) for a person to judge colours, garments and face.

    python tests/integration/ref_test.py --profile ltx25 --variant ref --control <controls/pose_env.mp4> \
        --reference <sheet.png> --prompt-file <prompt.json> --width 1280 --height 704 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cinestudio import server, video  # noqa: E402
from cinestudio.backend import SdkBackend, marker_for  # noqa: E402
from cinestudio.workflow import ProfileWorkflow  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--variant", default="ref")
    ap.add_argument("--control", default=None, help="control video (needed if the variant binds control_video)")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--first", default=None,
                    help="optional first frame (pins the look of frame 1); cropped to fill, never stretched")
    ap.add_argument("--ref-bg", default="black", help="border colour when the reference sheet is fitted to the clip size")
    ap.add_argument("--prompt-file", required=True, help="JSON {style, prompt, negative}")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--frames", type=int, default=121)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=333)
    ap.add_argument("--set", action="append", default=[], metavar="name=value", help="extra bound values, e.g. reference_strength=0.8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = f"ref-{a.profile}-{a.variant}-{int(time.time())}"
    wf = ProfileWorkflow(a.profile, a.variant)
    p = json.loads(Path(a.prompt_file).read_text(encoding="utf-8"))
    has_control = "control_video" in wf.bind["inputs"]
    if has_control and not a.control:
        ap.error(f"{a.profile}/{a.variant} needs --control")
    ctl = out / "control.mp4"
    if has_control:
        video.slice_control(a.control, ctl, 1, a.frames, a.width, a.height, a.fps)
    # The reference keeps its whole content (borders, no crop): a crop to the clip size would cut the side
    # the clip size, which would cut the side panels of a wide reference sheet.
    ref = out / "reference.png"
    video.fit_image(a.reference, ref, a.width, a.height, mode="pad", color=a.ref_bg)
    first = out / "first.png"
    video.fit_image(a.first or a.reference, first, a.width, a.height,
                    mode="cover" if a.first else "pad", color=a.ref_bg)
    prefix = f"cinestudio/{marker_for(run, 1, 1)}"
    values = {"prompt": "\n".join(x for x in [p.get("style"), p["prompt"]] if x), "negative": p.get("negative", ""),
              "width": a.width, "height": a.height, "frames": a.frames, "fps": a.fps, "seed": a.seed,
              "output_prefix": prefix}
    if "no_first_frame" in wf.bind["inputs"]:
        values["no_first_frame"] = a.first is None
    for item in a.set:
        k, v = item.split("=", 1)
        values[k] = json.loads(v)
    graph = wf.build(values)
    files = {wf.targets("reference")[0]: ref}
    if has_control:
        files[wf.targets("control_video")[0]] = ctl
    if "first_frame" in wf.bind["inputs"]:
        files[wf.targets("first_frame")[0]] = first
    switch = server.ensure_family(wf.capabilities["model_family"])     # one model family per ComfyUI process
    print(json.dumps({"event": "model_family", **switch}), flush=True)
    backend = SdkBackend()
    t0 = time.time()
    job = backend.submit(graph, files, key=run, prefix=prefix)
    server.note_owned(job)
    print(json.dumps({"event": "submitted", "job": job}), flush=True)
    st = backend.watch(job, lambda kind, d: print(json.dumps({"event": kind, **d}), flush=True) if kind == "progress" and d.get("step") else None)
    rec = {"event": "done", "status": st.status, "error": st.error, "seconds": round(time.time() - t0, 1)}
    passed = False
    if st.ok:
        dst = out / "clip.mp4"
        backend.download(job, wf.output_node, dst)
        rec["probe"] = video.probe(dst).__dict__
        scale = wf.capabilities.get("output_scale", 1)     # two-stage graphs make a larger clip
        rec["problems"] = video.check_clip(dst, a.frames, a.width * scale, a.height * scale, a.fps,
                                           audio=wf.capabilities.get("audio"))
        video.contact_sheet(dst, out / "contact_sheet.jpg", extra=ref)
        passed = not rec["problems"]
    rec["passed_format_check"] = passed
    rec["fidelity"] = "judge contact_sheet.jpg by eye" if passed else None
    (out / "report.json").write_text(json.dumps(rec | {"values": {k: v for k, v in values.items() if k not in ("prompt", "negative")}}, indent=1))
    print(json.dumps(rec), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
