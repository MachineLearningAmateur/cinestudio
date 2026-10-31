# Models

cinestudio uses **open-weights models only**, run by your own ComfyUI. No partner nodes,
no closed APIs. Download the files of a model tier with:

```bash
python -m cinestudio models status --tier 8gb              # what the tier needs and what is there
python -m cinestudio models download --tier 8gb --detach   # resumable; several files at once
python -m cinestudio models verify                         # full SHA-256 check of the files that are there
```

Or in the app: **Settings > Quick setup**. `setup/models.json` ("model_tiers") pins every URL to a repository commit
and stores the size and SHA-256 of every file; a downloaded file is used only after both match. `--feature NAME`
limits a download to one feature (video, character, music, select, pose, background, mocap).

## Tiers

| Tier | GPU / RAM | Precision | Files (every feature) |
|---|---|---|---|
| `8gb` | 8 GB VRAM, 32 GB RAM (tested: RTX 3070) | int8 (ConvRot) models, fp8 Qwen text encoder | about 134 GB |
| `large` | 80 GB+ VRAM, about 120 GB RAM (RunPod: RTX PRO 6000 96 GB, H100 SXM) | INT8 video and Qwen edit; larger ACE-Step model | about 144 GB |

`large` was called `24gb` before 2026-10-24; the old name still works. The tier is frozen into each run: a run keeps the
models it was made with. The large tier also writes larger VAE decode tiles into a run's graphs.

Both Blender and video-to-video workflows use the **distilled INT8 LTX-2.5** transformer in both tiers. Existing full-model runs remain viewable, but cannot be resumed or retaken.

## What each feature needs

| Feature | Use | Files |
|---|---|---|
| video | LTX-2.5 videos | LTX-2.5 distilled INT8 transformer, Gemma 4 text encoder, video and audio VAE, spatial upscaler, Layout-To-Render and Ingredients 2.5 IC-LoRAs, the LTX-2.3 union control IC-LoRA (pose guide, video-to-video), the Qwen edit files (the look image) |
| character | front views, edits, views, repose | Z-Image (base), its text encoder and VAE; Qwen-Image-Edit-2511, its text encoder and VAE; Multiple-Angles LoRA; AnyPose base + helper LoRAs |
| music | songs | ACE-Step 1.5 (base in 8gb, XL sft in large), its two text encoders and VAE |
| background | 360 panoramas, backplates, 360 depth | Qwen-Image-2512 GGUF Q8, the qwen-360-diffusion LoRA, MoGe-2 |
| select | the figure mask of a video template (`shot mask`) | SAM 2.1 |
| pose | the motion score (`run score`), pose drawings | DWPose + YOLOX (inside the comfyui_controlnet_aux node folder) |
| mocap | templates from a dance video (`shot from-video`) | SAM 3D Body, SAM 3.1 (several people) |

## Files, sizes, sources and licences

Common to both tiers:

| File | Folder | Size | Source | Licence |
|---|---|---|---|---|
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` (gated) | `diffusion_models` | 21.5 GB | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | LTX-2.x Community License |
| `ltx-2.5-video-vae-bf16.safetensors` (gated) | `vae` | 1.47 GB | Lightricks/LTX-2.5 (the diffusion decoder: CausalDiffusionVAE, one step, ComfyUI core) | LTX-2.x Community License |
| `ltx-2.5-audio-vae-bf16.safetensors` (gated) | `vae` | 0.36 GB | Lightricks/LTX-2.5 | LTX-2.x Community License |
| `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` (gated) | `latent_upscale_models` | 1.0 GB | Lightricks/LTX-2.5 | LTX-2.x Community License |
| `ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors` (gated) | `loras` | 1.31 GB | [Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render) | LTX-2.x Community License |
| `ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors` (gated) | `loras` | 1.31 GB | [Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients) | LTX-2.x Community License |
| `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` | `loras` | 0.65 GB | [Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control) | LTX-2 Community License |
| `z_image_bf16.safetensors` | `diffusion_models` | 12.3 GB | [Comfy-Org/z_image](https://huggingface.co/Comfy-Org/z_image) | Apache-2.0 |
| `qwen_3_4b.safetensors` | `text_encoders` | 8.0 GB | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) | Apache-2.0 |
| `ae.safetensors` | `vae` | 0.34 GB | Comfy-Org/z_image_turbo | Apache-2.0 |
| `qwen_image_vae.safetensors` | `vae` | 0.25 GB | [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) | Apache-2.0 |
| `qwen-image-edit-2511-multiple-angles-lora.safetensors` | `loras` | 0.30 GB | [fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA) | Apache-2.0 |
| `2511-AnyPose-base-000006250.safetensors`, `2511-AnyPose-helper-00006000.safetensors` | `loras` | 0.30 GB each | [serotoninboi/AnyPose](https://huggingface.co/serotoninboi/AnyPose) | Apache-2.0 |
| `qwen-image-edit-2511-style-transfer-1_20.safetensors` | `loras` | 0.24 GB | [dx8152/Qwen-Image-Edit-2511-Style-Transfer](https://huggingface.co/dx8152/Qwen-Image-Edit-2511-Style-Transfer) | Apache-2.0 (tested 2026-10-24: it changed almost nothing; no graph of the app uses it) |
| `qwen-image-2512-Q8_0.gguf` | `diffusion_models` | 21.8 GB | [unsloth/Qwen-Image-2512-GGUF](https://huggingface.co/unsloth/Qwen-Image-2512-GGUF) | Apache-2.0 |
| `qwen-360-diffusion-2512-int8-bf16-v2.safetensors` | `loras` | 0.76 GB | [ProGamerGov/qwen-360-diffusion](https://huggingface.co/ProGamerGov/qwen-360-diffusion) | MIT |
| `moge_2_vitl_normal_fp16.safetensors` | `geometry_estimation` | 0.66 GB | [Comfy-Org/MoGe](https://huggingface.co/Comfy-Org/MoGe) | MIT |
| `qwen_0.6b_ace15.safetensors` | `text_encoders` | 1.19 GB | [Comfy-Org/ace_step_1.5_ComfyUI_files](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files) | MIT |
| `ace_1.5_vae.safetensors` | `vae` | 0.34 GB | Comfy-Org/ace_step_1.5_ComfyUI_files | MIT |
| `sam2.1_hiera_large.safetensors`, `sam2.1_hiera_tiny.safetensors` | `sam2` | 0.90 + 0.16 GB | [Kijai/sam2-safetensors](https://huggingface.co/Kijai/sam2-safetensors) | Apache-2.0 |
| `dw-ll_ucoco_384_bs5.torchscript.pt`, `yolox_l.torchscript.pt` | comfyui_controlnet_aux `ckpts/hr16/...` | 0.14 + 0.22 GB | [hr16](https://huggingface.co/hr16) | Apache-2.0 |
| `sam_3d_body_dinov3_bf16.safetensors` | `detection` | 2.83 GB | [Comfy-Org/sam-3d-body](https://huggingface.co/Comfy-Org/sam-3d-body) | SAM License |
| `sam3.1_multiplex_fp16.safetensors` | `checkpoints` | 1.75 GB | [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) | SAM License |

Per tier:

| Slot | `8gb` | `large` |
|---|---|---|
| Qwen-Image-Edit-2511 (`diffusion_models`) | `qwen_image_edit_2511_int8_convrot.safetensors` 20.5 GB | `qwen_image_edit_2511_int8_convrot.safetensors` 20.5 GB |
| Qwen edit text encoder (`text_encoders`) | `qwen_2.5_vl_7b_fp8_scaled.safetensors` 9.4 GB | `qwen_2.5_vl_7b_fp8_scaled.safetensors` 9.4 GB |
| LTX-2.5 text encoder, Gemma 4 12B (`text_encoders`, gated) | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` 15.4 GB | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` 15.4 GB |
| ACE-Step model (`diffusion_models`) | `acestep_v1.5_base.safetensors` 4.8 GB | `acestep_v1.5_xl_sft_bf16.safetensors` 10.0 GB |
| ACE-Step second text encoder (`text_encoders`) | `qwen_1.7b_ace15.safetensors` 3.7 GB | `qwen_4b_ace15.safetensors` 8.4 GB |

Sources: Comfy-Org/Qwen-Image-Edit_ComfyUI and Comfy-Org/Qwen-Image_ComfyUI (Apache-2.0), Lightricks/LTX-2.5
(LTX-2.x Community License; the Gemma 4 encoder is repackaged there), Comfy-Org/ace_step_1.5_ComfyUI_files (MIT).
Full hashes and pinned commits: `setup/models.json`.

## Your own style LoRA (optional)

The video graphs have an optional style LoRA slot (nodes `lora_style` / `lora_style2`, before the sampler); the Z-Image
graph has one too. cinestudio ships no style LoRA. To use one, put the file in ComfyUI's `models/loras` folder (or
`python -m cinestudio models civitai VERSION_ID`) and give its file name as `--style-lora` (strength:
`--style-lora-strength`, default 0.8). Without a style LoRA the node is removed and the graph is the same as without
the slot.

- The LoRA must be trained for the graph's base model (LTX-2.5 22B for the videos, Z-Image for the front views). An
  LTX-2.3 LoRA made LTX-2.5 worse in our test; an LTX LoRA cannot load into Qwen-Image-Edit.
- Add the LoRA's trigger words to your prompt if it has any.
- You are responsible for the LoRA's licence and training data. Do not use LoRAs trained on
  copyrighted films or series, or on real people, for work you publish.
- A style LoRA can pull the character away from the reference sheet. Lower the strength if the
  character changes.

## Licence notes (read the original texts; this is a summary, not legal advice)

- **LTX-2.x Community License** (LTX-2.5: [LICENSE-2_x](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x);
  LTX-2.3 files: [LICENSE](https://github.com/Lightricks/LTX-2/blob/main/LICENSE)): open weights. Use is free for
  individuals and for entities with less than USD 10 million annual revenue; larger entities need a paid commercial
  licence from Lightricks. Read the acceptable-use terms.
- **Gemma** (the LTX-2.5 text encoder is Gemma 4 12B): Google's Gemma 4 base model is Apache-2.0
  ([licence](https://ai.google.dev/gemma/apache_2)); the LTX-2.5 file is published under the LTX-2.x Community License.
- **Apache-2.0** (Z-Image, Qwen-Image-Edit-2511, Qwen-Image-2512, the fal Multiple-Angles LoRA, AnyPose, SAM 2.1,
  DWPose; repackaged files from the Comfy-Org repos): permissive. **MIT**: ACE-Step 1.5, qwen-360-diffusion, MoGe-2.
- **SAM License** (SAM 3D Body, SAM 3.1): royalty-free; no military, weapons or ITAR uses; pass the licence on.
- cinestudio does not ship any model file. It downloads them from the original sources.
- MiniMax models are **not** supported (project rule). MiniMax H3's open-weight licence excludes the United States,
  the European Union, the United Kingdom and South Korea.
