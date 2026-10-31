"""Make the image-to-image "redraw" graphs (workflows/<profile>_redraw) from the restyle graphs.

    python setup/make_redraw_graphs.py            # ltx25 distilled

The source of the supported redraw graph: edit this script (or the restyle graph), never the redraw files;
tests/unit/test_model_parity.py checks that it reproduces workflows/.

Why (2026-10-22): the restyle graph puts the picture in as frame 0 of a 9-frame clip with a noise mask
(LTXVImgToVideoInplace, strength = "keep"; the sampler pulls that frame back toward the picture at every step) and
returns frame 0. The result stayed nearly a copy of the photo: a rooftop panorama "restyled" with the 3D LoRA at keep 0.5 and at keep 0.2 stayed the same photo.
The redraw graph is real image-to-image, as the video-to-video runs: the picture is VAE-encoded into every frame and the
sampler starts at noise `source_sigmas[0]` (LTX: x = sigma * noise + (1 - sigma) * picture), so about 1 - S of the
picture's structure stays and the model (and the LoRA) redraws the rest.
"""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "workflows"


def build(profile: str) -> tuple[dict, dict]:
    """The redraw graph and its binding of `profile`, built from its restyle graph (nothing is written)."""
    g = json.loads((WF / f"{profile}_restyle_api.json").read_text(encoding="utf-8"))
    b = json.loads((WF / f"{profile}_restyle.bind.json").read_text(encoding="utf-8"))
    g = copy.deepcopy(g)
    frames = g["lat"]["inputs"]["length"]
    g.pop("i2v")
    g["rep"] = {"class_type": "RepeatImageBatch", "inputs": {"image": ["sheet_fit", 0], "amount": frames}}
    g["enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["rep", 0], "vae": ["vvae", 0]}}
    g["av"]["inputs"]["video_latent"] = ["enc", 0]
    g.pop("lat")
    distilled = "ks" in g and g["ks"]["class_type"] == "KSampler"
    if distilled:                                       # KSampler -> custom sampler with explicit sigmas
        ks = g.pop("ks")["inputs"]
        g["noise1"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": ks["seed"]}}
        g["sampler1"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": ks["sampler_name"]}}
        g["guider1"] = {"class_type": "CFGGuider", "inputs": {"model": ks["model"], "positive": ks["positive"],
                                                              "negative": ks["negative"], "cfg": ks["cfg"]}}
        g["ks"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise1", 0], "guider": ["guider1", 0], "sampler": ["sampler1", 0], "sigmas": ["sigmas1", 0],
            "latent_image": ["av", 0]}}
    g["sigmas1"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": "0.9, 0.725, 0.421875, 0.0"}}
    inputs = {k: v for k, v in b["inputs"].items() if k not in ("keep_strength", "steps")}
    inputs["width"] = [["sheet_fit", "target_width"]]
    inputs["height"] = [["sheet_fit", "target_height"]]
    inputs["seed"] = [["noise1", "noise_seed"]]
    inputs["source_sigmas"] = [["sigmas1", "sigmas"]]
    b2 = dict(b, inputs=inputs)
    b2["_about"] = (f"{profile} image-to-image redraw (setup/make_redraw_graphs.py): the picture VAE-encoded into all "
                    f"{frames} frames, sampled from source_sigmas[0] (about 1 - S of the picture stays); the style "
                    "LoRA and the prompt redraw the rest. Replaces the restyle graph's first-frame hold, which copied "
                    "the picture.")
    return g, b2


def make(profile: str) -> None:
    g, b2 = build(profile)
    distilled = g["guider1"]["class_type"] == "CFGGuider"
    (WF / f"{profile}_redraw_api.json").write_text(json.dumps(g, indent=1), encoding="utf-8")
    (WF / f"{profile}_redraw.bind.json").write_text(json.dumps(b2, indent=1), encoding="utf-8")
    print("wrote", profile, "redraw", "(distilled sampler)" if distilled else "")


if __name__ == "__main__":
    for p in ("ltx25",):
        make(p)
