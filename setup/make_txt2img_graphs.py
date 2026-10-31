"""Make the text-to-picture graphs (workflows/<profile>_txt2img) from the redraw graphs.

    python setup/make_txt2img_graphs.py            # ltx25 distilled

The source of the supported txt2img graph: edit this script (or the redraw graph), never the txt2img files;
tests/unit/test_model_parity.py checks that it reproduces workflows/.

Why (2026-10-25, user: "why cant we have the lora generate first using prompt and then qwen uses that as a reference img"):
the images of a shot (sheets, backplates, look images) are made by Qwen from style WORDS, and the video model draws
the same words with its own style LoRA in another look. This graph lets the VIDEO model itself (LTX-2.5 + the style's
LoRA) make a short still clip FROM TEXT ONLY: no layout guide, no look image, no IC-LoRA. The sharpest of its saved
frames becomes the style's reference picture (cinestudio/style_ref.py), and Qwen conversions take a tight crop of it as
Picture 2 (AGENTS.md 4.3: one detail).

A still clip of FRAMES frames (8k+1) at WIDTH x HEIGHT (multiples of 64; portrait for a full-body subject). Five frames
are saved (PICKS); the caller keeps the sharpest. The sampler runs the model's whole schedule (bound `sigmas`:
runner.DISTILLED_SIGMAS for the distilled model, runner.dev_sigmas for the full one). The style LoRA is optional (a style
without a LoRA shows LTX's own look for its words).
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "workflows"
WIDTH, HEIGHT, FRAMES = 768, 1152, 17
PICKS = (0, 4, 8, 12, 16)
DROP = ("sheet_load", "sheet_fit", "rep", "enc", "pick_first", "pick_last", "save_first", "save_last")


def _schedule(full: bool) -> str:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from cinestudio import runner
    vals = runner.dev_sigmas(WIDTH, HEIGHT, FRAMES) if full else runner.DISTILLED_SIGMAS
    return ", ".join(f"{x:.6g}" for x in vals)


def build(profile: str) -> tuple[dict, dict]:
    """The txt2img graph and its binding of `profile`, built from its redraw graph (nothing is written)."""
    g = copy.deepcopy(json.loads((WF / f"{profile}_redraw_api.json").read_text(encoding="utf-8")))
    b = json.loads((WF / f"{profile}_redraw.bind.json").read_text(encoding="utf-8"))
    full = b.get("capabilities", {}).get("recipe") == "full"
    for k in DROP:
        g.pop(k)
    g["lat"] = {"class_type": "EmptyLTXVLatentVideo",
                "inputs": {"width": WIDTH, "height": HEIGHT, "length": FRAMES, "batch_size": 1},
                "_meta": {"title": "Empty video latent (text only)"}}
    g["av"]["inputs"]["video_latent"] = ["lat", 0]
    g["alat"]["inputs"]["frames_number"] = FRAMES
    g["sigmas1"]["inputs"]["sigmas"] = _schedule(full)
    g["lora_style"]["_meta"] = {"title": "Style LoRA (optional)"}
    for i in PICKS:
        g[f"pick_{i}"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["vdec", 0], "batch_index": i, "length": 1},
                          "_meta": {"title": f"Frame {i}"}}
        g[f"save_{i}"] = {"class_type": "SaveImage",
                          "inputs": {"images": [f"pick_{i}", 0], "filename_prefix": "cinestudio/style_ref"},
                          "_meta": {"title": f"Save frame {i}"}}
    keep = ("prompt", "negative", "seed", "style_lora", "style_lora_strength",
            "video_cfg", "audio_cfg", "rescale", "stg_scale", "modality_scale")
    inputs = {k: b["inputs"][k] for k in keep if k in b["inputs"]}
    inputs.update(width=[["lat", "width"]], height=[["lat", "height"]],
                  frames=[["lat", "length"], ["alat", "frames_number"]], sigmas=[["sigmas1", "sigmas"]],
                  output_prefix=[[f"save_{i}", "filename_prefix"] for i in PICKS])
    b2 = {"_about": (f"{profile} text-to-picture (setup/make_txt2img_graphs.py): a {FRAMES}-frame still clip from the "
                     "prompt alone (no layout, no look image, no IC-LoRA), the optional style LoRA, frames "
                     f"{', '.join(map(str, PICKS))} saved; cinestudio/style_ref.py keeps the sharpest as a style "
                     "reference picture."),
          "profile": b["profile"], "capabilities": b["capabilities"], "inputs": inputs,
          "optional_nodes": {"lora_style": {"when": "style_lora", "bridge": {"0": "model"}}},
          "outputs": {"video": f"save_{PICKS[0]}", "image": f"save_{PICKS[0]}",
                      "frames": [f"save_{i}" for i in PICKS]},
          "models": b["models"]}
    return g, b2


def make(profile: str) -> None:
    g, b2 = build(profile)
    (WF / f"{profile}_txt2img_api.json").write_text(json.dumps(g, indent=1), encoding="utf-8")
    (WF / f"{profile}_txt2img.bind.json").write_text(json.dumps(b2, indent=1), encoding="utf-8")
    print("wrote", profile, "txt2img")


if __name__ == "__main__":
    for p in ("ltx25",):
        make(p)
