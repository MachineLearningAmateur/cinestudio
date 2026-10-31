# Credits

cinestudio stands on open-source software, open-weight models, published research and the work of many people.
Thank you to all of them. Licences were checked against each model card, licence file or package page on
2026-10-25; the model files, sizes and pinned versions are in [MODELS.md](MODELS.md). cinestudio ships no model
file: the installer downloads each one from its original source. Check the licence of anything you add yourself (for
example your own style LoRAs: they are yours, not part of cinestudio).

## Models (open weights)

### Video

| Model | What cinestudio uses it for | Author | Licence |
|---|---|---|---|
| [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) distilled transformer (int8 ConvRot) | every Blender-shot video (Layout-To-Render), retakes, style redraws | Lightricks | LTX-2.x Community License |
| [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) video VAE, audio VAE, x2 latent spatial upscaler | decode picture and sound; the final-quality refine pass | Lightricks | LTX-2.x Community License |
| Gemma 4 12B text encoder (with LTX projection, in [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5)) | LTX-2.5's text encoder | Google (base model), Lightricks (LTX version) | LTX-2.x Community License (as published in the LTX-2.5 repo); Gemma 4: [Apache-2.0](https://ai.google.dev/gemma/apache_2) |
| [LTX-2.5 IC-LoRA Layout-To-Render](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render) | the Blender render as the layout guide | Lightricks | LTX-2.x Community License |
| [LTX-2.5 IC-LoRA Ingredients](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients) | the character sheet as the identity guide | Lightricks | LTX-2.x Community License |
| [LTX-2.3 IC-LoRA Union Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control) (ref0.5) | optional: video-to-video control and the pose guide (off by default) | Lightricks | LTX-2 Community License |

### Images and characters

| Model | What cinestudio uses it for | Author | Licence |
|---|---|---|---|
| [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (base; [Comfy-Org repack](https://huggingface.co/Comfy-Org/z_image)), its Qwen3-4B text encoder and VAE ([Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo)) | character front candidates from text | Tongyi-MAI (Alibaba) | Apache-2.0 |
| [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) ([Comfy-Org repack](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI), int8 ConvRot / bf16), Qwen2.5-VL 7B encoder and VAE ([Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI)) | character edits, views, repose, look images, backplates over a set | Qwen (Alibaba) | Apache-2.0 |
| [Multiple-Angles LoRA](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA) | the right, back and left views | Lovis Odin, fal | Apache-2.0 |
| [AnyPose](https://huggingface.co/serotoninboi/AnyPose) (base + helper LoRAs) | repose: the front posed like the character's mannequin | serotoninboi | Apache-2.0 |
| [SAM 2.1](https://github.com/facebookresearch/sam2) ([Kijai/sam2-safetensors](https://huggingface.co/Kijai/sam2-safetensors)) | the figure mask of a video template (`shot mask`) | Meta; safetensors by Kijai | Apache-2.0 |

### Backgrounds

| Model | What cinestudio uses it for | Author | Licence |
|---|---|---|---|
| [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) ([GGUF Q8 by Unsloth](https://huggingface.co/unsloth/Qwen-Image-2512-GGUF)) | 360 panoramas and backplates from text | Qwen (Alibaba); GGUF by Unsloth | Apache-2.0 |
| [qwen-360-diffusion](https://huggingface.co/ProGamerGov/qwen-360-diffusion) LoRA | equirectangular 360 panoramas | ProGamerGov | MIT |
| [MoGe-2](https://github.com/microsoft/moge) ([Comfy-Org/MoGe](https://huggingface.co/Comfy-Org/MoGe)) | 360 depth of a panorama (the depth ground) | Microsoft | MIT |

### Music

| Model | What cinestudio uses it for | Author | Licence |
|---|---|---|---|
| [ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) (base in 8gb, XL sft in large; [Comfy-Org repack](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files)) and its text encoders and VAE | songs and film scores from tags and lyrics | ACE-Step (ACE Studio and StepFun) | MIT (the Comfy-Org repack is tagged Apache-2.0) |

### Templates and motion

| Model | What cinestudio uses it for | Author | Licence |
|---|---|---|---|
| [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3) ([code](https://github.com/facebookresearch/sam-3d-body); [Comfy-Org repack](https://huggingface.co/Comfy-Org/sam-3d-body)) and ComfyUI's native SAM3DBody nodes | 3D body and hand motion from a reference dance video (`shot from-video`) | Meta | [SAM License](https://huggingface.co/facebook/sam-3d-body-dinov3/blob/main/LICENSE) |
| [SAM 3.1](https://huggingface.co/facebook/sam3.1) ([Comfy-Org repack](https://huggingface.co/Comfy-Org/sam3.1)) and ComfyUI's native SAM3 nodes | finds and follows every person in a reference video (`shot from-video --people 2\|pick`) | Meta | [SAM License](https://huggingface.co/Comfy-Org/sam3.1/blob/main/LICENSE) |
| [DWPose](https://github.com/IDEA-Research/DWPose) (Yang et al., ICCVW 2023; [TorchScript files by hr16](https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5)) | the motion follow score (`run score`), the people check | IDEA Research | Apache-2.0 |
| [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) ([hr16/yolox-onnx](https://huggingface.co/hr16/yolox-onnx)) | DWPose's person finder | Megvii | Apache-2.0 |
| [SDPose](https://github.com/T-S-Liang/SDPose-OOD) (Liang et al., 2025) with [RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4), ComfyUI's native SDPose nodes | optional second pose detector (`pose_people.py --detector sdpose`, the detector bench) | Liang et al.; RT-DETRs | MIT; Apache-2.0 |

## Template media

| Template | What | Use |
|---|---|---|
| `shan_shen_bu_balls` (templates/library; its source `templates/shan_shen_bu_props/base_shot.blend`) | the mannequin's dance is read (SAM 3D Body) from a public clip of the viral Chinese meme dance "shan shen bu" (flash step) | the built-in flash-step dodgeball template |

The dance: "shan shen bu" (闪身步, the dodge step) is a move of the Anhui Flower Drum Lantern dance (Huagudeng), a folk
art of the Huai River region; it went viral in 2026 with a teaching clip by dance teacher Zhao Tiechun
([CGTN, 2026-09-29](https://news.cgtn.com/news/2026-09-29/Traditional-Chinese-dance-moves-spark-viral-imitation-trend-1QPGRm62xVu/p.html)).
The reference clip itself is not part of the repo. The other templates (the mannequin, the camera moves, the
dances made from code) are cinestudio's own. `shan_shen_bu_balls` is the same dance with dodgeballs and a blocked-out
gym set built in Blender for cinestudio.

## Background media

| Background | What | Use |
|---|---|---|
| `gym_noir_comfyui` (backgrounds/library) | a backplate made with Qwen-Image-Edit-2511 from the Blender set of the template `shan_shen_bu_balls` (the clay render redrawn as an anime-noir gym), then two signs painted in code and refined by Qwen region edits | the built-in backplate of `shan_shen_bu_balls` |

The Comfy wordmark in the gym backplate is Comfy Org's; used per comfy.org/brand.

## Software

| Project | What cinestudio uses it for | Author | Licence |
|---|---|---|---|
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | runs every model (cinestudio starts it in the background) | Comfy Org | GPL-3.0 |
| [comfy-sdk](https://pypi.org/project/comfy-sdk/), [comfy-api-proxy](https://pypi.org/project/comfy-api-proxy/) | submit, watch and cancel ComfyUI jobs | Comfy Org | MIT |
| [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) | LTX-2.5 guiders, IC-LoRA loader and guides, AdaIN, latent select | Lightricks | LTX-2 Community License |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | loads the Qwen-Image-2512 GGUF model | city96 | Apache-2.0 |
| [ComfyUI-segment-anything-2](https://github.com/kijai/ComfyUI-segment-anything-2) | SAM 2.1 figure selection | Kijai | Apache-2.0 |
| [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) | DWPose code and the OpenPose drawing | Fannovel16 | Apache-2.0 |
| [SageAttention](https://github.com/thu-ml/SageAttention) (optional) | faster attention for the LTX-2.5 models only | Tsinghua University (thu-ml) | Apache-2.0 |
| [PyTorch](https://pytorch.org) | ComfyUI's runtime | PyTorch Foundation | BSD-3-Clause |
| [Blender](https://www.blender.org) | the shots: mannequin, camera, renders and passes (background mode) | Blender Foundation | GPL-2.0-or-later (binaries: GPL-3.0-or-later) |
| [FFmpeg](https://ffmpeg.org) via [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | video and sound encode, slice, stitch, loudness | FFmpeg developers; imageio | the FFmpeg build: LGPL/GPL; imageio-ffmpeg: BSD-2-Clause |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | the MCP server for agents | Model Context Protocol project | MIT |
| [OpenCV](https://github.com/opencv/opencv-python) | images, edges, drawing, ECC alignment | OpenCV team | Apache-2.0 |
| [NumPy](https://numpy.org) | arrays | NumPy developers | BSD-3-Clause |
| [aiohttp](https://github.com/aio-libs/aiohttp) | the web app | aio-libs | Apache-2.0 AND MIT |
| [psutil](https://github.com/giampaolo/psutil), [nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/) | the resource meters | Giampaolo Rodola; NVIDIA | BSD-3-Clause; BSD |
| [pycocotools](https://github.com/ppwwyyxx/cocoapi) | the official COCO keypoint evaluation in the motion follow score | COCO consortium | BSD-2-Clause (FreeBSD) |

The app uses the system's fonts and its own icons; it ships no font or third-party logo (the one exception is
the Comfy wordmark painted into the built-in gym backplate: Background media above).

## Licence obligations (a summary, not legal advice; read the original texts)

- **LTX-2.x Community License** ([LICENSE-2_x](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x); the LTX-2.3
  LoRA and ComfyUI-LTXVideo: [LTX-2 Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE)): free
  for individuals and for entities with less than USD 10 million annual revenue (measured across the whole entity
  with its affiliates); larger entities need a paid licence from Lightricks for anything but non-commercial use.
  Anyone you pass the weights or derivatives to gets a copy of the licence and its use restrictions; keep the
  notices; keep the safety, watermark and provenance features. The acceptable-use policy forbids, among others,
  impersonating people without their consent (deepfakes) and publishing generated content without a clear notice
  where that would mislead. Lightricks claims no rights in your outputs.
- **SAM License** (SAM 3D Body, SAM 3.1): royalty-free; no military, weapons or ITAR uses; pass the licence on when you
  redistribute the weights.
- **Gemma 4** (inside the LTX-2.5 text encoder): Apache-2.0 for Google's base model; the LTX-2.5 file is published
  under the LTX-2.x Community License.
- **Apache-2.0 / MIT** (Z-Image, Qwen-Image-Edit-2511, Qwen-Image-2512, Multiple-Angles, AnyPose, SAM 2.1, DWPose,
  YOLOX; ACE-Step 1.5, qwen-360-diffusion, MoGe-2): keep the licence and notices when you redistribute.
- **ComfyUI** is GPL-3.0: cinestudio talks to it as a separate program through its API and does not include its code.

## Research and techniques

Papers, model cards, repositories and articles our recipes and code are built on. Our own measurements decided
the settings; these sources gave the methods.

### Video quality and long shots

| Source | What we took from it | Author | Link |
|---|---|---|---|
| LTX-2 technical report (arXiv 2601.03233) | separate audio and video latents; long single passes | Lightricks | [arxiv.org/abs/2601.03233](https://arxiv.org/abs/2601.03233) |
| Layout-To-Render model card | the look image at frame -1 sets palette, light and materials; keyframe looks | Lightricks | [model card](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render) |
| Ingredients model card | the sheet on black with one clean panel per element; strength 1.0; the 768x448 / 121-frame size; the prompt headings | Lightricks | [model card](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients) |
| ComfyUI-LTXVideo looping sampler and `LTXVAdainLatent` | AdaIN of later clips toward clip 1's latent; the previous clip's tail latents as a guide (latent relay) | Lightricks | [repo](https://github.com/Lightricks/ComfyUI-LTXVideo) |
| LTX-2 `ltx_pipelines` chunks | a 25-frame carry at the clip start and cross-faded joins | Lightricks | [repo](https://github.com/Lightricks/LTX-2) |
| diffusers PR #12614 (long multi-prompt LTX pipeline) | cross-window AdaIN at about 0.25 | Hugging Face diffusers | [PR](https://github.com/huggingface/diffusers/pull/12614) |
| Rolling Forcing (arXiv 2509.25161), ID-Forcing (arXiv 2610.03120), StreamingT2V (arXiv 2403.14773) | the anchor of a long video is the FIRST chunk, kept fixed (clip-1 AdaIN and tone match) | the papers' authors | [2509.25161](https://arxiv.org/abs/2509.25161), [2610.03120](https://arxiv.org/abs/2610.03120), [2403.14773](https://arxiv.org/abs/2403.14773) |
| Super SloMo (Jiang et al., CVPR 2018) | the intermediate-flow warp of the flow cross-fade at joins | Jiang et al. | [arXiv 1712.00080](https://arxiv.org/abs/1712.00080) |
| FFmpeg scaler flags (ticket #979) | exact frame decode (`accurate_rnd+full_chroma_int+bitexact`) | FFmpeg | [ffmpeg-trac](https://ffmpeg.org/pipermail/ffmpeg-trac/2024-August/070619.html) |
| "How To Generate 2D Animation With AI Video Models" | the images carry the style; LoRA and text only agree with it | LTX Team | [ltx.io blog](https://ltx.io/blog/how-to-generate-2d-animation-with-ai-video-models) |

### Images and characters

| Source | What we took from it | Author | Link |
|---|---|---|---|
| Z-Image paper (arXiv 2511.22699) and model card | the base model's settings (cfg, a real negative) | Tongyi-MAI | [paper](https://arxiv.org/abs/2511.22699), [model card](https://huggingface.co/Tongyi-MAI/Z-Image) |
| Z-Image prompt enhancer `pe.py` and the team's prompt advice | prompt order (core with count and colours, composition, light, material); no meta tags; long and detailed | Tongyi-MAI | [pe.py](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py), [discussion #8](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/8) |
| Qwen-Image technical report (arXiv 2508.02324) | the Qwen-Image model family | Qwen | [paper](https://arxiv.org/abs/2508.02324) |
| Qwen-Image edit prompt rewriter `prompt_utils.py` | the instruction form ("Replace Y with X", name what to keep, text in quotes); no negations in the positive text | QwenLM | [prompt_utils.py](https://github.com/QwenLM/Qwen-Image/blob/main/src/examples/tools/prompt_utils.py) |
| Qwen-Image-Edit-2511 model card and the official ComfyUI 2511 template | 40 steps, true CFG; the edit graph (cfg 3) | Qwen; Comfy Org | [model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511), [template](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_qwen_image_edit_2511.json) |
| ComfyUI `TextEncodeQwenImageEditPlus` | the inputs are called "Picture 1 / Picture 2" in prompts | Comfy Org | [nodes_qwen.py](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_qwen.py) |
| Multiple-Angles LoRA card | view prompts `<sks> ... view eye-level shot medium shot`, strength 0.9 | fal | [model card](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA) |
| SDEdit (Meng et al., ICLR 2022) | paint in code, then refine at low denoise | Stanford | [arXiv 2108.01073](https://arxiv.org/abs/2108.01073) |
| Attend-and-Excite (Chefer et al.) and SynGen (Rassin et al.) | name each garment with its side, count and extent | the papers' authors | [2301.13827](https://arxiv.org/abs/2301.13827), [2306.08877](https://arxiv.org/abs/2306.08877) |

### Backgrounds

| Source | What we took from it | Author | Link |
|---|---|---|---|
| "Qwen-Image-Edit 2511 pixel shift" | native-size input with sides in multiples of 16, no encoder rescale (the set render's edit size) | lilting.ch | [article](https://lilting.ch/en/articles/qwen-image-edit-pixel-perfect-referencelatent) |
| OpenCV `findTransformECC` (Evangelidis and Psarakis, TPAMI 2008) | fits a backplate back onto the set render | OpenCV | [docs](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html) |
| qwen-360-diffusion card | the panorama trigger and recipe | ProGamerGov | [model card](https://huggingface.co/ProGamerGov/qwen-360-diffusion) |
| MoGe-2 | 360 depth for the depth ground | Microsoft | [repo](https://github.com/microsoft/moge) |

### Audio and music

| Source | What we took from it | Author | Link |
|---|---|---|---|
| LTX-2 `ltx-pipelines` 1.4.0 (chunks, operators) | audio carry: the previous clip's audio-latent tail frozen in the next clip; a short seam fade | Lightricks | [repo](https://github.com/Lightricks/LTX-2) |
| ComfyUI-LTXVideo LTX-2.5 A2V two-stage example | stage 2 keeps the stage-1 audio frozen | Lightricks | [repo](https://github.com/Lightricks/ComfyUI-LTXVideo) |
| Fink, Holters, Zölzer, "Signal-matched power-complementary cross-fading" (DAFx 2016) | equal-gain fades for correlated sound, equal-power for independent clips | HSU Hamburg | [paper (PDF)](https://www.hsu-hh.de/ant/wp-content/uploads/sites/699/2017/10/Fink-Holters-Zölzer-2016-Signal-matched-power-complementary-cross-fading-and-dry-wet-mixing.pdf) |
| EBU R128 and ITU-R BS.1770-5 | one loudness normalization of a film (-16 LUFS) | EBU; ITU | [R128](https://tech.ebu.ch/publications/r128), [BS.1770-5](https://www.itu.int/rec/R-REC-BS.1770-5-202311-I) |
| FFmpeg filters (`loudnorm`, `acrossfade`, `sidechaincompress`) | two-pass loudness, fade curves, ducking the score | FFmpeg | [docs](https://ffmpeg.org/ffmpeg-filters.html) |
| ACE-Step 1.5 (arXiv 2602.00744) | one score of the film's exact length (10-600 s) | ACE-Step | [paper](https://arxiv.org/abs/2602.00744), [repo](https://github.com/ace-step/ACE-Step-1.5) |

### Speed

| Source | What we took from it | Author | Link |
|---|---|---|---|
| ComfyUI PR #14636 (int8 + ConvRot), issues #14642/#14824 | native int8 on RTX 30/40/50 cards (fp8 is emulated on Ampere); int8 with LoRAs; not A100 | Comfy Org | [PR #14636](https://github.com/Comfy-Org/ComfyUI/pull/14636), [#14642](https://github.com/Comfy-Org/ComfyUI/issues/14642), [#14824](https://github.com/Comfy-Org/ComfyUI/issues/14824) |
| ComfyUI-Native-INT8_ConvRot | int8 ConvRot is closer to bf16 than fp8 | obsxrver | [model card](https://huggingface.co/obsxrver/ComfyUI-Native-INT8_ConvRot) |
| LTX-2.5 8-bit quant comparison | ConvRot is the best 8-bit format for LTX | Unsloth | [model card](https://huggingface.co/unsloth/LTX-2.5-FP8) |
| Qwen-Image-Edit-Causal | each extra reference picture costs time: send Picture 1 only | ModelTC | [repo](https://github.com/ModelTC/Qwen-Image-Edit-Causal) |
| "Applying guidance in a limited interval" (Kynkäänniemi et al., NeurIPS 2024), "Adaptive Guidance" (Castillo et al.) | the CFG tail: no negative pass in the last steps (core `CFGOverride`) | the papers' authors | [2404.07724](https://arxiv.org/abs/2404.07724), [2312.12487](https://arxiv.org/abs/2312.12487) |
| SageAttention (arXiv 2410.02367); ComfyUI issue #9773; woct0rdho's fork notes | Sage for LTX only (black Qwen pictures) | thu-ml; Comfy Org; woct0rdho | [paper](https://arxiv.org/abs/2410.02367), [#9773](https://github.com/Comfy-Org/ComfyUI/issues/9773), [fork](https://github.com/woct0rdho/SageAttention) |
| "Dynamic VRAM in ComfyUI"; PR #15861 | keep the default memory flags (no `--highvram` / `--gpu-only`) | Comfy Org | [blog](https://blog.comfy.org/p/dynamic-vram-in-comfyui-saving-local), [PR #15861](https://github.com/Comfy-Org/ComfyUI/pull/15861) |
| ComfyUI-LTXVideo LTX-2.5 example workflows | large-GPU VAE decode tiles | Lightricks | [repo](https://github.com/Lightricks/ComfyUI-LTXVideo) |
| RunPod network volume docs, GPU guide and pricing | the GPU choice (RTX PRO 6000, H100 SXM) and the shared engine | RunPod | [volumes](https://docs.runpod.io/storage/network-volumes), [guide](https://www.runpod.io/articles/guides/best-gpu-for-ai-video), [pricing](https://www.runpod.io/pricing) |

### Text in images

| Source | What we took from it | Author | Link |
|---|---|---|---|
| Qwen-Image-2512 prompt rewriter `prompt_utils_2512.py` | "The image contains no recognizable text." in background prompts | QwenLM | [prompt_utils_2512.py](https://github.com/QwenLM/Qwen-Image/blob/main/src/examples/tools/prompt_utils_2512.py) |

### Motion score and testing

| Source | What we took from it | Author | Link |
|---|---|---|---|
| OpenPose (Cao et al., TPAMI 2019) | the skeleton format, colours and face-point rule of the pose pass | CMU | [arXiv 1812.08008](https://arxiv.org/abs/1812.08008) |
| Microsoft COCO (Lin et al., ECCV 2014) | COCO keypoints and OKS in the motion follow score | Lin et al. | [arXiv 1405.0312](https://arxiv.org/abs/1405.0312) |
| PCK (Johnson and Everingham, BMVC 2010); PoseGen (arXiv 2508.05091), HyperMotionX (arXiv 2505.22977) | PCK@0.2 and the protocol of running a standard detector on the generated video | the papers' authors | [2508.05091](https://arxiv.org/abs/2508.05091), [2505.22977](https://arxiv.org/abs/2505.22977) |
| Top-down pose estimation with given boxes (DWPose / RTMPose) | Blender's boxes replace the person finder, one box per character | IDEA Research; OpenMMLab | [DWPose](https://github.com/IDEA-Research/DWPose) |
| Build-and-judge loop with blind critics | the clip gauntlet and the blind critic | somethingbig.ai | [gauntlet-loop](https://somethingbig.ai/gauntlet-loop) |

### App and UI

| Source | What we took from it | Author | Link |
|---|---|---|---|
| "Response Times: The 3 Important Limits"; "Slider Design: Rules of Thumb" | the Mannequin view's delayed preview; a slider plus a number field per row | Nielsen Norman Group | [response times](https://www.nngroup.com/articles/response-times-3-important-limits/), [sliders](https://www.nngroup.com/articles/gui-slider-controls/) |
| WAI-ARIA APG slider pattern | native, labelled range inputs | W3C | [pattern](https://w3c.github.io/wai-website/ARIA/apg/patterns/slider/) |
| MetaHuman Creator body tools | presets first, then fine sliders; a height readout; a big preview beside the controls | Epic Games | [Body Params](https://dev.epicgames.com/documentation/metahuman/metahuman-creator-body-params-tool-in-unreal-engine), [navigation](https://dev.epicgames.com/documentation/metahuman/navigating-metahuman-creator-in-unreal-engine) |

Our research notes, with every source we read: `docs/*_research.md`.
