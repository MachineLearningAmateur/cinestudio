"""LTX-2.5 Layout-To-Render graphs of BOTH models: workflows/ltx25dev_layout[ing] (full) and ltx25_layout[ing] (distilled).

    python setup/make_layout_graphs.py

The single source of these four graphs and their bind maps: edit this script, never the files.
tests/unit/test_model_parity.py checks that it reproduces workflows/ and that the pair differs only where the model needs
it (cinestudio/model_pairs.py).

Lightricks' recipe (model card of Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render and its ComfyUI graph
LTX-2.5_ICLoRA_Layout_To_Render_Two_Stage_Distilled.json): the layout IC-LoRA at 1.0; the LAYOUT video (a clay or
blocking render from Blender: the camera, the motion, where things stand) as an IC-LoRA guide at frame 0; the LOOK image
(the first Blender frame redrawn by an image model: palette, light, materials; never the grey clay frame) as a second
guide at frame -1 (outside the clip, attention only); both guides again in stage 2 (x2 upscale + refine); the guides
are cropped before decoding. Width and height must divide by 64; frames 8k+1.

What we keep from our graphs (ltx25dev_v2v): stage 1 of the FULL (dev) model with MultimodalGuider + GuiderParameters
(CFG rescale 0.7, STG block 28) and the 30-step dev schedule (bound `sigmas`, from runner.dev_sigmas), audio.
`layouting` adds the Ingredients IC-LoRA (2.5) with the character sheet held on every frame, as in ltx25dev_v2v.

Stage 2 of the FULL model (2026-10-25, user-approved): Lightricks' two-stage recipe (ltx-pipelines
TI2VidTwoStagesPipeline at LTX-2 e98f6d9, Diffusers' LTX-2 docs): the dev transformer + the distilled LoRA
ltx-2.5-22b-distilled-lora-450-bf16 (strength 1.0 = DEFAULT_LORA_STRENGTH and Diffusers; the CLI example uses 0.8;
bound as `stage2_lora_strength`), no guidance (SimpleDenoiser: CFGGuider cfg 1), STAGE_2_DISTILLED_SIGMA_VALUES
0.909375, 0.725, 0.421875, 0 with plain Euler (no noise between steps). The user's LoRAs stay on the stage-2 model (the
pipeline fuses `*loras, *distilled_lora`). Before: MultimodalGuider CFG 3 (no STG), 6 steps 0.6..0 (runs before
2026-10-25 keep that in their frozen graphs: runner.stage2_recipe calls it "guided").
"""
import copy
import json
from pathlib import Path

W = Path(__file__).resolve().parent.parent / "workflows"
BASE = "ltx25dev_v2v"
LAYOUT_LORA = "ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors"
DISTILLED_LORA = "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"   # setup/models.json slot ltx25_distilled_lora
STAGE2_SIGMAS = "0.909375, 0.725, 0.421875, 0.0"      # ltx-pipelines STAGE_2_DISTILLED_SIGMA_VALUES (both models)
STAGE2_LORA_STRENGTH = 1.0
KEEP = ("ckpt", "vvae", "audio_vae", "te", "pos", "neg", "cond", "alat", "noise1", "sampler1", "sigmas1", "gpv1", "gpa1",
        "noise2", "sampler2", "sigmas2", "up_model", "vdec", "adec", "mk", "save")


def guide(pos, neg, lat, image, frame_idx, crop, strength=1.0):
    return {"class_type": "LTXAddVideoICLoRAGuide",
            "inputs": {"positive": pos, "negative": neg, "vae": ["vvae", 0], "latent": lat, "image": image,
                       "frame_idx": frame_idx, "strength": strength, "latent_downscale_factor": ["lora_lay", 1],
                       "crop": crop, "use_tiled_encode": False, "tile_size": 256, "tile_overlap": 64}}


def build(ingredients: bool, base_name: str = BASE) -> tuple[dict, dict]:
    """`base_name` ltx25dev_v2v = the full model (our guider); ltx25_v2v = the DISTILLED model (CFG 1, 8 steps,
    euler_ancestral: Lightricks' own Layout-To-Render recipe, with its stage-2 sigmas 0.909375 .. 0)."""
    base = json.loads((W / f"{base_name}_api.json").read_text(encoding="utf-8"))
    bb = json.loads((W / f"{base_name}.bind.json").read_text(encoding="utf-8"))
    distilled = not base_name.startswith("ltx25dev")
    g = {k: copy.deepcopy(base[k]) for k in KEEP if k in base}
    g["sigmas2"]["inputs"]["sigmas"] = STAGE2_SIGMAS             # the distilled stage 2 in both models
    g["lora_lay"] = {"class_type": "LTXICLoRALoaderModelOnly",
                     "inputs": {"model": ["ckpt", 0], "lora_name": LAYOUT_LORA, "strength_model": 1.0},
                     "_meta": {"title": "Layout-To-Render IC-LoRA (2.5)"}}
    top = "lora_lay"
    if ingredients:
        g["lora_ing"] = copy.deepcopy(base["lora_ing"])
        g["lora_ing"]["inputs"]["model"] = ["lora_lay", 0]
        g["icp_ing"] = copy.deepcopy(base["icp_ing"])
        top = "lora_ing"
    g["lora_style"] = copy.deepcopy(base["lora_style"])
    g["lora_style"]["inputs"]["model"] = [top, 0]
    g["lora_style2"] = copy.deepcopy(base["lora_style2"])        # stage 2: the same IC-LoRAs (the guides are there too)
    g["lora_style2"]["inputs"]["model"] = [top, 0]
    # inputs: the layout video (the Blender render), the look image
    g["lay_load"] = {"class_type": "LoadVideo", "inputs": {"file": "LAYOUT.mp4"}}
    g["lay_frames"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["lay_load", 0]}}
    g["look_load"] = {"class_type": "LoadImage", "inputs": {"image": "LOOK.png"}}
    g["lat"] = copy.deepcopy(base["lat"])
    # the continuation of a long shot (clip k > 1): the previous clip's last frames in place at the start of the latent,
    # as in our v2v graphs (bypassed for clip 1); the look image stays the run's own look for every clip
    for k in ("first_load", "first"):
        g[k] = copy.deepcopy(base[k])
    g["i2v"] = copy.deepcopy(base["i2v"])
    g["i2v"]["inputs"]["latent"] = ["lat", 0]
    pos, neg = ["cond", 0], ["cond", 1]
    lat = ["i2v", 0]
    if ingredients:                                              # the sheet on every frame (Ingredients), as in v2v
        for k in ("ref_load", "ref_fit", "ref_video"):
            g[k] = copy.deepcopy(base[k])
        g["ref_guide"] = copy.deepcopy(base["ref_guide"])
        g["ref_guide"]["inputs"].update({"positive": pos, "negative": neg, "latent": lat})
        pos, neg, lat = ["ref_guide", 0], ["ref_guide", 1], ["ref_guide", 2]
    g["g_lay"] = guide(pos, neg, lat, ["lay_frames", 0], 0, "disabled")
    g["g_look"] = guide(["g_lay", 0], ["g_lay", 1], ["g_lay", 2], ["look_load", 0], -1, "center")
    g["av"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["g_look", 2], "audio_latent": ["alat", 0]}}
    g["guider1"] = copy.deepcopy(base["guider1"])
    g["guider1"]["inputs"].update({"model": ["lora_style", 0], "positive": ["g_look", 0], "negative": ["g_look", 1]})
    g["ks"] = copy.deepcopy(base["ks"])
    g["ks"]["inputs"]["latent_image"] = ["av", 0]
    g["sep"] = copy.deepcopy(base["sep"])
    g["crop"] = {"class_type": "LTXVCropGuides", "inputs": {"positive": ["g_look", 0], "negative": ["g_look", 1],
                                                           "latent": ["sep", 0]}}
    # stage 2: x2 upscale, the guides again at the new size, refine, crop, decode
    g["up"] = copy.deepcopy(base["up"])
    g["up"]["inputs"]["samples"] = ["crop", 2]
    g["i2v2"] = copy.deepcopy(base["i2v2"])
    g["i2v2"]["inputs"]["latent"] = ["up", 0]
    pos2, neg2, lat2 = ["crop", 0], ["crop", 1], ["i2v2", 0]
    g["g_lay2"] = guide(pos2, neg2, lat2, ["lay_frames", 0], 0, "disabled")
    g["g_look2"] = guide(["g_lay2", 0], ["g_lay2", 1], ["g_lay2", 2], ["look_load", 0], -1, "center")
    g["av2"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["g_look2", 2], "audio_latent": ["sep", 1]}}
    model2 = ["lora_style2", 0]
    if not distilled:                                            # the full model: + the distilled LoRA, no guidance
        g["lora_dist"] = {"class_type": "LoraLoaderModelOnly",
                          "inputs": {"model": model2, "lora_name": DISTILLED_LORA,
                                     "strength_model": STAGE2_LORA_STRENGTH},
                          "_meta": {"title": "Distilled LoRA (stage 2 of the dev model)"}}
        model2 = ["lora_dist", 0]
        g["guider2"] = {"class_type": "CFGGuider", "inputs": {"model": model2, "positive": ["g_look2", 0],
                                                              "negative": ["g_look2", 1], "cfg": 1.0}}
    else:
        g["guider2"] = copy.deepcopy(base["guider2"])
        g["guider2"]["inputs"].update({"model": model2, "positive": ["g_look2", 0], "negative": ["g_look2", 1]})
    g["ks2"] = copy.deepcopy(base["ks2"])
    g["ks2"]["inputs"]["latent_image"] = ["av2", 0]
    g["sep2"] = copy.deepcopy(base["sep2"])
    g["crop2"] = {"class_type": "LTXVCropGuides", "inputs": {"positive": ["g_look2", 0], "negative": ["g_look2", 1],
                                                            "latent": ["sep2", 0]}}
    g["vdec"]["inputs"]["samples"] = ["crop2", 2]
    g["adec"]["inputs"]["samples"] = ["sep2", 1]

    keep_inputs = ("prompt", "negative", "width", "height", "frames", "fps", "seed", "style_lora", "style_lora_strength",
                   "first_frame", "no_first_frame", "first_strength",
                   "refine_sigmas", "output_prefix", "video_cfg", "audio_cfg", "rescale", "stg_scale", "modality_scale")
    if ingredients:
        keep_inputs += ("reference", "ingredients_strength")
    inputs = {k: [t for t in bb["inputs"][k] if t[0] in g] for k in keep_inputs if k in bb["inputs"]}
    inputs["sigmas"] = [["sigmas1", "sigmas"]]
    inputs["layout_video"] = [["lay_load", "file"]]
    inputs["look_image"] = [["look_load", "image"]]
    inputs["layout_strength"] = [["lora_lay", "strength_model"]]
    inputs["guide_strength"] = [["g_lay", "strength"], ["g_lay2", "strength"]]
    models = {k: [t for t in v if t[0] in g] for k, v in (bb.get("models") or {}).items()}
    models = {k: v for k, v in models.items() if v and (ingredients or k != "ic_ingredients25") and k != "ic_union"}
    models["ic_layout"] = [["lora_lay", "lora_name"]]
    if not distilled:
        models["ltx25_distilled_lora"] = [["lora_dist", "lora_name"]]
        inputs["stage2_lora_strength"] = [["lora_dist", "strength_model"]]
    b = {"_about": "LTX-2.5 full model, Layout-To-Render IC-LoRA (setup/make_layout_graphs.py): the Blender render as the "
                   "layout guide + a look image at frame -1" + (" + the Ingredients sheet" if ingredients else ""),
         "profile": base_name.split("_")[0], "capabilities": {**bb.get("capabilities", {}), "layout": True}, "inputs": inputs,
         "models": models, "outputs": bb["outputs"],
         "optional_nodes": {k: v for k, v in bb.get("optional_nodes", {}).items() if k in ("lora_style", "lora_style2")}}
    b["capabilities"]["first_frame"] = True                      # the continuation frames (clip k > 1); the look is -1
    b["capabilities"]["stage2"] = "distilled" if distilled else "distilled-lora"   # runner.stage2_recipe
    for name, targets in list(inputs.items()) + list(models.items()):
        for t in targets:
            assert t[0] in g and t[1] in g[t[0]]["inputs"], (name, t)
    for k, n in g.items():                                       # every link points at a node of this graph
        for v in n["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g, (k, v)
    return g, b


if __name__ == "__main__":
    for base_name in ("ltx25dev_v2v", "ltx25_v2v"):
        prof = base_name.split("_")[0]
        for name, ing in (("layout", False), ("layouting", True)):
            g, b = build(ing, base_name)
            (W / f"{prof}_{name}_api.json").write_text(json.dumps(g, indent=1), encoding="utf-8")
            (W / f"{prof}_{name}.bind.json").write_text(json.dumps(b, indent=1), encoding="utf-8")
    print("ok 4")
