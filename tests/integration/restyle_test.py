"""Restyle a reference sheet with the user's style LoRA (workflows/ltx25_restyle, LTX-2.5 distilled).

The styled sheet can then be the reference of a video clip (ref_test.py --reference), so that the
Ingredients IC-LoRA copies the character in the new style. Passes only if the job succeeds and both
images (frame 0 and the last frame of the still clip) have the requested size. Whether the design is
kept cannot be measured by a script: compare.png shows the original and both results for a person.

    python tests/integration/restyle_test.py --reference <sheet.png> --prompt-file <prompt.json> \
        --style-lora Fantasy_Anime.safetensors --keep 0.5 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cinestudio import server, video  # noqa: E402
from cinestudio.backend import SdkBackend, marker_for  # noqa: E402
from cinestudio.workflow import ProfileWorkflow  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--prompt-file", required=True, help="JSON {style, prompt, negative}")
    ap.add_argument("--style-lora", required=True, help="file name in ComfyUI models/loras")
    ap.add_argument("--style-strength", type=float, default=0.8)
    ap.add_argument("--keep", type=float, default=0.5, help="how much of the original sheet stays (0..1)")
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=896)
    ap.add_argument("--seed", type=int, default=333)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = f"restyle-{int(time.time())}"
    wf = ProfileWorkflow("ltx25", "restyle")
    p = json.loads(Path(a.prompt_file).read_text(encoding="utf-8"))
    ref = out / "reference.png"
    video.fit_image(a.reference, ref, a.width, a.height, mode="pad", color="black")
    prefix = f"cinestudio/{marker_for(run, 1, 1)}"
    values = {"prompt": "\n".join(x for x in [p.get("style"), p["prompt"]] if x), "negative": p.get("negative", ""),
              "width": a.width, "height": a.height, "seed": a.seed, "keep_strength": a.keep,
              "style_lora": a.style_lora, "style_lora_strength": a.style_strength,
              "output_prefix": prefix, "output_prefix_last": prefix + "_last"}
    graph = wf.build(values)
    files = {wf.targets("reference")[0]: ref}
    switch = server.ensure_family(wf.capabilities["model_family"])
    print(json.dumps({"event": "model_family", **switch}), flush=True)
    backend = SdkBackend()
    t0 = time.time()
    job = backend.submit(graph, files, key=run, prefix=prefix)
    server.note_owned(job)
    print(json.dumps({"event": "submitted", "job": job}), flush=True)
    st = backend.watch(job, lambda kind, d: None)
    rec = {"event": "done", "status": st.status, "error": st.error, "seconds": round(time.time() - t0, 1), "problems": []}
    if st.ok:
        imgs = []
        for key, name in (("image", "sheet_first.png"), ("image_last", "sheet_last.png")):
            dst = out / name
            backend.download(job, wf.bind["outputs"][key], dst)
            im = cv2.imread(str(dst))
            if im is None or im.shape[:2] != (a.height, a.width):
                rec["problems"].append(f"{name}: size {None if im is None else im.shape[1::-1]}, want {(a.width, a.height)}")
            imgs.append(im)
        if not rec["problems"]:
            cv2.imwrite(str(out / "compare.png"), np.vstack([cv2.imread(str(ref))] + imgs))
    rec["passed"] = st.ok and not rec["problems"]
    (out / "report.json").write_text(json.dumps(rec | {"values": {k: v for k, v in values.items() if k not in ("prompt", "negative")}}, indent=1))
    print(json.dumps(rec), flush=True)
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
