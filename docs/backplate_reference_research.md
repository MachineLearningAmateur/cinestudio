# Backplates that follow the Blender set: reference -> image options (research 2026-10-24)

User: "right now it feels like we are praying that the background would fit". Goal: a backplate (one picture for a
still camera) whose objects, floor and horizon sit exactly where the template's Blender set has them.

What this is: web research plus a read of our repo and of the local ComfyUI core (v0.42.1, the Comfy Desktop install
at `D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI`). No code was changed. Nothing was downloaded. No GPU job ran.
Marks: **[SRC]** = from a cited page; **[CODE]** = read in ComfyUI or cinestudio source; **[CALC]** = my arithmetic
from that code; **[UNVERIFIED]** = a claim or a guess that nobody has tested for our case.

## 0. Summary

1. **Our own pipeline bends the geometry before any model runs [CODE][CALC].** `workflows/qwenedit_edit_api.json`
   sends the set render through `FluxKontextImageScale`. That node picks the nearest of 17 fixed "Kontext" sizes and
   **center-crops** to that shape. `make_set_plate` then stretches the result back to the plate size with
   `cv2.resize`. For a 16:9 camera (plate 1920x1088) the bucket is 1392x752: 25 px are cut from the top and from the
   bottom, then the picture is stretched 4.7% taller. For a 768x448 camera (plate 1920x1120) the bucket is 1328x800:
   30 px are cut from each side, then the picture is stretched 3.3% wider. Every object is off by up to ~25-30 px at
   the frame edges. The text encoder also rescales the reference latent to ~1 MP, which adds a further ~1% scale
   mismatch for 1328x800 (none for 1392x752). Fix this first. It costs no download. See 2.4.
2. **The best new option is a real ControlNet on Qwen-Image-2512.** `alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union`
   (version 2602, Apache-2.0, 3.51 GB) is trained FOR 2512. It takes canny, HED, depth, pose, MLSD, scribble and gray
   images. ComfyUI core loads it with `ControlNetLoader` and applies it with `ControlNetApplyAdvanced` (core support
   since PR #12359, merged 2026-02-14) [SRC][CODE]. Blender gives us an EXACT depth map and EXACT line map of the set,
   so the control image has no estimation error.
3. **Older Qwen-Image ControlNets also load in core:** InstantX Union (canny, soft edge, depth, pose; Apache-2.0;
   3.54 GB) through `ControlNetLoader`; DiffSynth "Blockwise" canny / depth / inpaint patches (Apache-2.0; 2.27 GB
   each) through `ModelPatchLoader` + `QwenImageDiffsynthControlnet` [CODE]. They were trained on the first Qwen-Image,
   not on 2512 [SRC]. Use them only if the 2512 model fails.
4. **Qwen-Image-Edit-2511 has no stated depth/edge input.** The 2509 card says "Native Support for ControlNet:
   Including depth maps, edge maps, keypoint maps" [SRC]. The 2511 card does not mention it [SRC]. 2511 probably keeps
   it (it is a 2509 successor) [UNVERIFIED]. The edit route keeps our clay render's colours as hints for what each
   shape is. That is its strength. It has no strength knob for geometry. That is its weakness.
5. **Style:** there is no IP-Adapter for Qwen-Image [SRC: none found]. Style comes from text, from Qwen-Image-Edit's
   Picture 2, or from a LoRA. Our test of the dx8152 Style-Transfer LoRA changed almost nothing (docs/MODELS.md).
   Z-Image (our character model) has its own Apache-2.0 ControlNet for the BASE model
   (`Z-Image-Fun-Controlnet-Union-2.1`, 6.71 GB, or a 2.02 GB lite) that core loads (`ModelPatchLoader` +
   `ZImageFunControlnet`) [SRC][CODE]. It is the one route where the background and the characters come from the same
   image model.
6. **Other model families are not clearly better.** SDXL ControlNets are mature but the base looks worse. SD3.5
   ControlNets carry the Stability Community License (free only under USD 1M yearly revenue). FLUX.1-dev, FLUX.2-dev,
   FLUX.2 klein 9B, Hunyuan and Qwen-Image 2.1 are excluded by licence. Wan 2.x control is for video. Skip them.
7. **Recommendation.** (a) On the 3070: fix the crop/stretch bug, then A/B "Edit-2511 from-set (fixed)" against
   "Qwen-Image-2512 Q8 + Fun ControlNet Union 2602 with Blender depth + lines". (b) On a RunPod 80-96 GB card: the same
   two arms at bf16 and higher resolution, 4-8 seeds each, and pick the seed by a measured edge-fit score. A test plan
   with metrics is in section 7.

## 1. What we have today (repo)

- `background plate` = `make_plate`: Qwen-Image-2512 GGUF Q8 (21.8 GB) through `qwenimage_pano_api.json`
  (`UnetLoaderGGUF`, 360 LoRA at strength 0, 50 steps, cfg 4, shift 3.1, `EmptySD3LatentImage`). Text only. The text
  names the camera height, the look direction and the horizon row (`plate_prompt`). Nothing ties objects to the set.
- `background from-set` = `make_set_plate`: `bpy_jobs/render_set.py` hides Mannequin*/Props/rigs and renders the set
  with Workbench (material colours, studio light, cavity) at frame 1, at the plate size (1920 wide). Qwen-Image-Edit-2511
  int8 ConvRot (`qwenedit_edit_api.json`, 40 steps, cfg 3, `CFGOverride` to cfg 1 after 65%) redraws it with
  `SET_INSTRUCTION` ("Every object in Picture 1 keeps exactly its place, size, shape and outline ...").
- `bpy_jobs/plate_world.py` then hides the set in a run and shows the plate on a large floor (z = 0) and a far wall.
- Template layers (`shots.py`): shot.json `layers.descriptions` has a text for the set and for each prop. These words
  can feed any of the options below (a ControlNet route needs them more, because it sees no colours).
- Machine: RTX 3070 8 GB, 32 GB RAM, Windows; ComfyUI 0.42.1; Blender 5.2 (`C:\Program Files\Blender Foundation`).

## 2. Qwen-Image-Edit-2511 (the route we have)

### 2.1 Control-style inputs
- 2509 card (Sept 2025): "Native Support for ControlNet: Including depth maps, edge maps, keypoint maps, and more"; also
  "keypoint control and sketches" [SRC: huggingface.co/Qwen/Qwen-Image-Edit-2509, read 2026-10-24]. No prompt form is
  given.
- 2511 card: lists less drift, character consistency, built-in LoRAs (lighting, new viewpoints), industrial design and
  "geometric reasoning" (it draws construction lines). It does NOT mention ControlNet, depth or edge inputs
  [SRC: huggingface.co/Qwen/Qwen-Image-Edit-2511, read 2026-10-24]. Example settings: 40 steps, true_cfg 4.
- A community 2509 workflow (dx8152, Apache-2.0) uses the prompt "Generate a modern living room using the line/depth map
  in the image" for depth/line redraws and says the control is strong [SRC: huggingface.co/dx8152/Qwen-Edit-2509-workflow].
- So: Picture 1 = a Blender depth map or line map, with "Generate <the place> using the depth map in Picture 1", is a
  cheap extra arm for 2511 [UNVERIFIED for 2511]. It loses the clay colours, which today tell the model what each shape
  is (a brown box = a table). The clay render (today's input) is probably the better Picture 1.

### 2.2 Multi-image (Picture 2 = a style reference)
- 2511 takes two or more images; ComfyUI names them Picture 1 / Picture 2 (AGENTS.md 4.3). Picture 1 = the clay set,
  Picture 2 = a style crop is possible today with our graph (the `ref` slot).
- Risk (AGENTS.md 4.3): a whole image as Picture 2 is copied too much. A crop of a character would put a character or
  its parts into the empty place. Use a crop of a SURFACE in the target style (a wall, a floor, a prop of an earlier
  approved backplate), not a character crop [UNVERIFIED].

### 2.3 LoRAs for "render/clay -> finished picture" on 2511
- None found with a clear commercial licence and a geometry claim. The dx8152 Style-Transfer LoRA (Apache-2.0) changed
  almost nothing in our test (docs/MODELS.md). Zaytron40k's 2511 StyleTransfer LoRA converts to ONE fixed illustrated
  style; licence not shown [SRC: huggingface.co/Zaytron40k/Qwen-Image-Edit-2511-StyleTransfer-LoRA]. Skip.

### 2.4 The geometry bug in our edit graph (fix before any A/B)
- `FluxKontextImageScale` (comfy_extras/nodes_flux.py) chooses the bucket with the closest aspect from
  `PREFERRED_KONTEXT_RESOLUTIONS` and calls `common_upscale(..., "lanczos", "center")`: the "center" crop cuts the
  image to the bucket's shape [CODE].
- `make_set_plate` resizes the output back to (w, h) with `cv2.resize` (a stretch, no un-crop) [CODE].

| camera | plate | bucket | crop | stretch back |
|---|---|---|---|---|
| 1920x1080, 1280x720, 1024x576 | 1920x1088 (1.765) | 1392x752 (1.851) | 25 px top + 25 px bottom | 4.7% taller |
| 768x448 | 1920x1120 (1.714) | 1328x800 (1.660) | 30 px left + 30 px right | 3.3% wider |

[CALC]. The crop also drops set content at the frame edges, and the model then invents nothing there because the
stretch fills it. Every object that touches the frame edge moves.

- Second, smaller error: `TextEncodeQwenImageEditPlus` with a `vae` input rescales each reference image to ~1 MP
  (rounded to 8) before it encodes the reference latent [CODE]. The sampled latent comes from `VAEEncode` of the
  bucket image. For 1328x800 the reference is 1320x792 (-0.6% / -1.0%); for 1392x752 they match [CALC]. A test on 2511
  measured this kind of mismatch as a scale error (+3 px at the left edge to +14 px at the right; bodies 2% taller) and
  fixed it by leaving the encoder's `vae` input unconnected and passing a native-size `VAEEncode` through
  `ReferenceLatent` (residual ~1 px), with sizes snapped to multiples of 16 [SRC: lilting.ch, "Qwen-Image-Edit 2511
  pixel shift", 2026-07-06, updated 2026-09-22].
- Fix (for a backplate graph only; the character graphs can keep theirs until tested):
  1. Render the set at a ~1 MP size with the camera's exact aspect, both sides multiples of 16 (e.g. 1360x768 for
     16:9). No Kontext bucket, no crop.
  2. `VAEEncode` that image at native size -> `ReferenceLatent` (or the encoder without `vae`) and the same latent as
     the sampler's start latent. Same size in = same size out.
  3. Upscale the result to 1920 wide with the SAME aspect (or render the set larger on RunPod).
  4. Measure: phase correlation / ECC between the Blender edge map and the plate's edges must give scale 1.000 and
     shift < 2 px (section 7).

## 3. ControlNets for Qwen-Image (text -> image with a structure image)

All three below are Apache-2.0 and run in ComfyUI CORE (no custom pack) [CODE: comfy/controlnet.py,
comfy_extras/nodes_model_patch.py]. Sizes from the Hugging Face API (2026-10-24).

| option | trained on | control types | files | ComfyUI core nodes |
|---|---|---|---|---|
| **Qwen-Image-2512-Fun-Controlnet-Union** (alibaba-pai) | Qwen-Image-**2512** | canny, HED, depth, pose, MLSD, scribble; 2602 adds gray; inpaint mode | `Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors` 3.51 GB (repo created 2026-01-12, updated 2026-02-13) | `ControlNetLoader` (detected by `control_img_in.weight` -> `load_controlnet_qwen_fun`) + `ControlNetApplyAdvanced` (needs the `vae` input) |
| **InstantX Qwen-Image-ControlNet-Union** | Qwen-Image (Aug 2025) | canny, soft edge, depth, pose | 3.54 GB (`Comfy-Org/Qwen-Image-InstantX-ControlNets` repack, same size) | `ControlNetLoader` (`load_controlnet_qwen_instantx`) + `ControlNetApplyAdvanced` |
| InstantX Qwen-Image-ControlNet-Inpainting | Qwen-Image | inpaint (68-channel input) | 4.23 GB | same |
| **DiffSynth Blockwise ControlNet** canny / depth / inpaint | Qwen-Image (BLIP3o-60k data) | one file per type | 2.27 GB each (`Comfy-Org/Qwen-Image-DiffSynth-ControlNets`, `model_patches`) | `ModelPatchLoader` ("Load Model Patch") + `QwenImageDiffsynthControlnet` (strength, optional mask, start/end percent) |
| DiffSynth In-Context Control Union LoRA | Qwen-Image | canny, depth, lineart, softedge, normal, openpose | `qwen_image_union_diffsynth_lora.safetensors` 0.94 GB (Comfy-Org repack, tagged Apache-2.0; the DiffSynth repo page did not open for me) | `LoraLoaderModelOnly`; the control image goes in as a reference (in-context) [UNVERIFIED wiring] |

Details:
- **Fun Union 2512.** Model card: control on 5 layer blocks, "can be used like a standard ControlNet",
  `control_context_scale` best 0.70-0.95 (higher = stronger control, more detail kept), detailed prompts recommended,
  train control images at several resolutions [SRC: huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union].
  ComfyUI's own template `image_qwen_Image_2512_controlnet` uses strength 1.0, start 0, end 1, 50 steps, cfg 4,
  euler/simple, Canny preprocessing, optional Lightning 4-step LoRA (off) [SRC: Comfy-Org/workflow_templates].
  ComfyUI applies Fun strength through a square-root curve ("strength=1.0 stays unchanged while >1 grows more
  gently") [CODE: `QwenFunControlNet.get_control`]. So ComfyUI 0.8 acts like ~0.89 in the raw formula; the card's
  0.70-0.95 may not map 1:1 onto ComfyUI's slider [CALC/UNVERIFIED]. One PR comment said it "doesn't seem to be
  working", another that a colleague tested it fine [SRC: ComfyUI PR #12359]. Test it before relying on it.
- **InstantX Union.** 5 double blocks copied from the base, ~10M images, 1328x1328, 50K steps; conditioning scale
  0.8-1.0; example 30 steps, true_cfg 4 [SRC: huggingface.co/InstantX/Qwen-Image-ControlNet-Union]. ComfyUI's template
  makes the depth image with Lotus and then INVERTS it (near = white) [SRC: workflow template
  `image_qwen_image_instantx_controlnet`]. A community workflow used it with Qwen-Image-Edit-**2509** for pose at
  strength 1.8 [SRC: nextdiffusion.ai, 2025-10-01]: so these ControlNets also attach to the edit model.
- **DiffSynth patches.** A model patch, not a ControlNet: it adds a small block output after every double block
  [CODE: `DiffSynthCnetPatch`]. ComfyUI's template: strength 1.0, 20 steps, cfg 2.5, Canny (0.1/0.2), 1.68 MP
  [SRC: workflow template `image_qwen_image_controlnet_patch`]. A side-by-side review (Diffusion Doodles, 2025-09-16):
  InstantX depth kept pose and face best; DiffSynth depth was "similar" with DepthAnythingV2 but gave a "somewhat
  surreal blended image" with Lotus depth; the In-Context Union LoRA kept structure "very effectively" and was the
  author's all-rounder; canny carried unwanted outlines (lights) into the result [SRC: diffusiondoodles.substack.com].
- **Excluded:** Qwen-Image 2.1 and its Fun ControlNet Union (2026-09) [SRC: comfyui-wiki 2026-09-24]: the base model's
  `qwen-research` licence is non-commercial (AGENTS.md 2). ComfyUI code for it exists (`QwenImage21FunControl`) but do
  not use it.

**With Qwen-Image-2512?** The Fun Union is built for 2512 [SRC]. InstantX and DiffSynth load (same architecture) but
were trained on the first Qwen-Image; quality on 2512 is [UNVERIFIED]. **With the GGUF loader?** ControlNets work on the
model's output path and the DiffSynth patch hooks the double blocks, so both should work with `UnetLoaderGGUF`
[UNVERIFIED; check once].

**With Qwen-Image-Edit-2511?** Same transformer layout, so they attach. Only the InstantX + 2509 pose workflow is
reported [SRC]. A "hybrid" (Edit-2511 with the clay set as Picture 1 AND a depth ControlNet) is a candidate arm
[UNVERIFIED].

**8 GB feasibility and speed [UNVERIFIED estimates].** The 2512 Q8 transformer streams from RAM today (21.8 GB). The
Fun ControlNet adds 3.51 GB (bf16) and ~5 extra blocks of compute (the base has 60), so maybe +10-20% time per step.
RAM: AGENTS.md says ~4 GB stay free during Qwen jobs; +3.5 GB is close to the limit. Options: a lower GGUF (Q6_K), or
an fp8 copy of the ControlNet (ComfyUI picks the weight dtype from the file [CODE]). DiffSynth patches are lighter
(2.27 GB, small per-block adds). Watch RAM in the first run and keep other heavy jobs off.

**Keeping geometry.** A ControlNet ties the picture to the control image at every step (start 0, end 1). Depth pins
the layout and perspective; lines pin the exact outlines. Settings to test: strength 0.8 and 1.0; end_percent 1.0
(exact) and 0.8 (the last steps free for texture). The text must name what each shape is (shot.json set and prop
descriptions), because a depth map does not say "table". Canny/lines also copy every edge, including ones we do not
want (shadow edges, Workbench cavity lines): give the ControlNet a CLEAN line map from geometry (section 6), not a
Canny of the shaded render.

## 4. Image-to-image from the clay render (denoise < 1)

- What it does: encode the clay render, add noise to a level set by denoise, denoise with the scene text. Low denoise
  keeps the clay colours and the flat Workbench light; high denoise loses the layout. Generic guides: 0.3-0.5 keeps
  composition, 0.6-0.8 keeps only the general structure [SRC: ComfyUI docs on mintlify, morphic.com glossary].
- **Qwen's shift makes it harsher [CALC].** Our graphs use `ModelSamplingAuraFlow` shift 3.1:
  sigma(t) = 3.1 t / (1 + 2.1 t). With denoise d the run starts at sigma(d): d 0.3 -> 0.57, d 0.4 -> 0.67,
  d 0.5 -> 0.76, d 0.6 -> 0.82, d 0.7 -> 0.88. So "denoise 0.6" already starts at 82% noise. Our own Qwen-EDIT note
  (AGENTS.md 4.3: 1.0 rebuilds, 0.8 moves a hand, 0.6 changes nothing) is for the edit model, which also sees the
  reference latent; it does not carry over to plain 2512 img2img.
- Verdict: img2img alone cannot keep geometry exact AND replace clay materials: the two needs fight. Use it only
  together with a ControlNet (start from the clay latent at denoise ~0.9-1.0 to inherit the rough colour layout, with
  depth + lines pinning the shapes) [UNVERIFIED], or as a cheap control arm in the A/B.

## 5. Style references (the look of our 3D characters)

- **Qwen-Image (2512):** no IP-Adapter or image-prompt adapter found [SRC: searches 2026-10-24]. Style = text (the
  character's `style` field words, AGENTS.md 4.9) or a trained LoRA (training a 2512 LoRA needs a big GPU; RunPod).
- **Qwen-Image-Edit-2511:** Picture 2 as a style reference works in our graph (section 2.2). Untested for backplates.
  Or a two-pass: ControlNet 2512 makes the geometry-exact plate, then Edit-2511 restyles it with Picture 2 at denoise
  ~0.5-0.7 [UNVERIFIED; the edit may move edges: measure the fit after each pass].
- **Z-Image (base, Apache-2.0):** `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1` (Apache-2.0; created 2026-02-02):
  canny, depth, pose, MLSD, scribble, HED, gray, plus inpaint; full = 15 control layers + 2 refiner layers (6.71 GB);
  lite = 3 layers (2.02 GB, weaker, "more natural" at higher scale, for low-spec machines); scale 0.65-1.00
  [SRC: huggingface.co/alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1]. The Turbo variants are separate files (8 steps).
  Core loads both: `ModelPatchLoader` detects `control_all_x_embedder` and the 15-layer/refiner layout;
  `ZImageFunControlnet` ("Apply Fun ControlNet") applies it [CODE]. This keeps characters and background in the same
  image model and the same style words. Z-Image's skill at empty interiors and landscapes vs 2512 is [UNVERIFIED].
- **Excluded:** USO style reference (core node `USOStyleReference`) is built on FLUX.1-dev; IP-Adapters for FLUX.1-dev
  (InstantX) likewise. Our rule bans FLUX.1-dev bases.
- AGENTS.md 6 (docs/style_consistency_research.md): put ONE style into every image before LTX sees it. So whichever
  route wins, the backplate's style words must equal the sheet's style words.

## 6. Which Blender pass for which option, and how to export it

| option | best condition | second | notes |
|---|---|---|---|
| Fun Union 2512 | depth (exact) | lines (as canny or MLSD), chained as a second `ControlNetApplyAdvanced`; or gray (2602) from the clay render | MLSD = straight lines: fits boxy sets [UNVERIFIED] |
| InstantX Union | depth | canny / soft edge | depth near = white (ComfyUI inverts Lotus) |
| DiffSynth patch | depth (DepthAnything-like) | canny | one patch file per type; two patches = two loads |
| Edit-2511 | the clay render (colours = object hints) | depth or line map as Picture 1 (2509-style) | fix the crop bug first |
| Z-Image Fun 2.1 | depth | canny / MLSD | |
| segmentation (object IDs) | no Qwen/Z-Image ControlNet takes it | | use it to MEASURE the fit and for per-object masks (inpaint one object, EliGen-style regions) |
| normals | only the DiffSynth union LoRA lists "normal" | | mainly a source for crease lines |

**Depth (all ControlNets).** The models were trained on Depth-Anything-style maps: relative INVERSE depth, near =
bright, far = dark, no metric scale [SRC: InstantX card names depth-anything; ComfyUI template inverts Lotus]. From
Blender: d = 1/z; normalise with the 1st and 99th percentile of the set's pixels; pixels with no geometry (sky) = 0;
save 8-bit (or 16-bit then 8-bit) grey, repeated in R, G, B. A smooth floor ramp is what tells the model the
perspective.

**Lines.** Build them from geometry, not from shading: depth jumps (|grad log z| above a threshold) + normal creases
(neighbour normals > ~30 deg apart) + object-ID borders. 1-2 px white lines on black at ~1 MP. This gives clean
outlines without the Workbench cavity and shadow edges that a Canny of the clay render picks up. Optionally run
`cv2.Canny` on that line image once so it looks like the training data [UNVERIFIED benefit].

**Gray (Fun 2602 only).** The Workbench render in grey: the model colours it in. Keeps geometry and the Workbench
shading; may keep the clay light. An extra arm [UNVERIFIED].

**Export in `bpy_jobs/render_set.py` [UNVERIFIED in Blender 5.2; check once].** Keep the current Workbench colour
render (the Edit-2511 input), and add a second render of the same frame:
- engine Cycles on the CPU (AGENTS.md 2: no EEVEE while a GPU job runs; Cycles CPU is allowed), 1 sample, a small
  pixel filter so edges stay hard;
- `view_layer.use_pass_z`, `use_pass_normal`, `use_pass_object_index` (give each kept object `pass_index = i+1`, and
  write the name -> index map into the SET json line);
- `render.image_settings.file_format = "OPEN_EXR_MULTILAYER"` (float), `write_still`; read it in Python (Blender ships
  OpenImageIO; or OpenCV with `OPENCV_IO_ENABLE_OPENEXR=1` on the cinestudio side) and write `depth.png`,
  `lines.png`, `ids.png` (+ ids.json) next to set.png.
- Fallback with no render engine: cast one ray per pixel from the camera (`scene.ray_cast`) for depth, normal and
  object; exact, ~0.5-1 M rays at ~1 MP, maybe 10-30 s in Python [UNVERIFIED timing].
- Check whether Cycles' Z pass is planar depth or distance to the camera in 5.2; for 1/z either works, but the
  normalisation must be the same everywhere.
- Render at the SAME ~1 MP size as the generation (multiple of 16, exact camera aspect) so no resize sits between
  the control image and the latent.

**A complementary fix outside the image model.** `plate_world.py` puts the plate on a flat floor and a far wall. Since a
set plate is drawn for THIS camera, it can be projected from the camera onto the real set meshes instead (UV project
from view). Then the set stays in the layout render at its exact place, the character can walk behind a table, and the
plate's pixels land on the objects they show [UNVERIFIED; a separate task].

## 7. Ranking and test plan

### 7.1 Other model families (question 5): not clearly better
- SDXL + xinsir ControlNet Union ProMax: mature, light, core `ControlNetLoader`. SDXL base is CreativeML
  OpenRAIL++-M (commercial use allowed with use rules) [licence from memory, UNVERIFIED]. The base picture is weaker
  than Qwen-Image. Only a fallback.
- SD3.5 Large ControlNets (canny, depth, blur; ~8B each): Stability AI Community License: free commercial use only
  under USD 1M yearly revenue [SRC: stability.ai/license, stability.ai/news/sd3-5-large-controlnets]. Avoid for a
  shipped tool.
- FLUX.2 klein 4B: Apache-2.0, multi-reference editing; the 9B is non-commercial [SRC: docs.bfl.ai/flux_2]. No ControlNet
  found. FLUX.2-dev Fun ControlNet: dev licence is non-commercial. Excluded.
- HiDream-I1 (MIT), Chroma (Apache-2.0), Lumina (Apache-2.0): few or weak ControlNets [UNVERIFIED]. HunyuanImage:
  excluded (EU/UK/KR).
- Wan 2.2 Fun Control / VACE (Apache-2.0): depth-controlled VIDEO; single-frame use not documented [SRC: runcomfy,
  stablediffusiontutorials]. Heavy. Skip for plates.

### 7.2 Ranked recommendation
**(a) RTX 3070 now**
1. Fix the edit graph's geometry (2.4): no download, removes a 3-5% stretch and a 25-30 px edge crop. Keep from-set
   as the baseline.
2. Add Qwen-Image-2512 Q8 + Fun ControlNet Union 2602 (one 3.51 GB download, Apache-2.0, core nodes) with Blender
   depth + lines. It is the only option trained for 2512 and it gives a strength knob for geometry.
3. Hybrid: Edit-2511 (clay set as Picture 1) + InstantX Union depth at 0.8-1.0 (one 3.54 GB download) if arm 2 loses
   the object identities.
4. Z-Image base + Fun ControlNet Union 2.1 lite (2.02 GB) if style match to the characters wins over geometry.

**(b) RunPod 80-96 GB**
- The same arms at bf16 (2512 bf16 / Edit-2511 bf16 40.9 GB), generated nearer 2 MP (or at 1328-1664 wide and
  upscaled), Z-Image full ControlNet (6.71 GB). Make 4-8 seeds per arm and keep the seed with the best edge fit (7.3)
  automatically; then the critic picks among the top 2. Optional second pass: Edit-2511 restyle with Picture 2, with
  the fit measured again.

### 7.3 Test plan
- **Set:** a template with at least 5 set objects (table, sofa or bench, door frame, shelf, window wall). The
  built-ins have plain floors (the `set_sparse` case), so build or pick a set template first.
- **Same inputs for every arm:** one scene text (shot.json set + prop descriptions), the character's style words,
  3 seeds, the same ~1 MP generation size with the exact camera aspect.
- **Arms:** A0 today's from-set (unfixed: to measure the bug); A1 from-set fixed; B 2512 Q8 + Fun Union 2602, depth
  only, strength 1.0, end 1.0; B2 = B + lines (second Apply); B3 = B at strength 0.8 / end 0.8; C hybrid Edit-2511 +
  InstantX depth (only if B is weak on object identity); optional D Z-Image + Fun 2.1.
- **Geometry fit (measured):** (1) global alignment: ECC or phase correlation between the Blender line map and a Canny
  of the plate: scale must be 1.000 +- 0.005, shift < 2 px; (2) edge F-score: precision and recall of plate edges
  within 3 px (at 1920 wide) of the Blender line map (distance transform); (3) per-object IoU: SAM 2.1 (already in our
  models) prompted with each object's box from `ids.png`, against the Blender object mask; report the mean and the
  worst object; (4) horizon / floor-line row error in px.
- **Style match:** a fresh blind critic (docs/gauntlet_critic.md style) gets the plate next to the character sheet and
  scores "same rendering style" 1-10; plus lightness and chroma statistics of the plate vs the sheet.
- **Picture quality:** the blind critic (1-10: clean, believable, no clay look, no invented people or text) and a
  sharpness number (variance of the Laplacian at 1920 wide).
- **Pass bar (proposal):** scale error < 0.5%, edge F-score >= 0.7, mean object IoU >= 0.85, critic style >= 7,
  picture >= 7. Then one layout run in PREVIEW with the winning plate to see that the feet meet the floor.
- Record every arm in `.cinestudio/gauntlet/backplate_ref/RESULTS.md`; add the winning trick to AGENTS.md section 6.

## 8. Sources (all read 2026-10-24)

- Qwen-Image-Edit-2511 card: https://huggingface.co/Qwen/Qwen-Image-Edit-2511 ; docs: https://docs.qwenlm.ai/qwen-image-edit-2511/index.html
- Qwen-Image-Edit-2509 card (native ControlNet inputs): https://huggingface.co/Qwen/Qwen-Image-Edit-2509
- dx8152 2509 workflow (depth/line redraw prompt, Apache-2.0): https://huggingface.co/dx8152/Qwen-Edit-2509-workflow
- Fun ControlNet Union for 2512 (Apache-2.0, 2602): https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union
- ComfyUI issue #12100 (2026-01-26) and PR #12359 (merged 2026-02-14): https://github.com/Comfy-Org/ComfyUI/issues/12100 , https://github.com/Comfy-Org/ComfyUI/pull/12359
- ComfyUI workflow templates (2512 Fun ControlNet, InstantX, DiffSynth patch): https://github.com/Comfy-Org/workflow_templates (templates/image_qwen_Image_2512_controlnet.json, image_qwen_image_instantx_controlnet.json, image_qwen_image_controlnet_patch.json)
- InstantX Union (Apache-2.0): https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union ; Inpainting: https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting ; repack: https://huggingface.co/Comfy-Org/Qwen-Image-InstantX-ControlNets
- DiffSynth Blockwise depth / canny (Apache-2.0): https://huggingface.co/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth , https://huggingface.co/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny ; repack incl. union LoRA: https://huggingface.co/Comfy-Org/Qwen-Image-DiffSynth-ControlNets
- DiffSynth EliGen-V2 (entity masks, Apache-2.0, DiffSynth pipeline; ComfyUI only via a community pack): https://huggingface.co/DiffSynth-Studio/Qwen-Image-EliGen-V2
- Comparison of Qwen ControlNets (Diffusion Doodles, 2025-09-16): https://diffusiondoodles.substack.com/p/qwen-image-controlnets
- Edit-2509 + InstantX Union pose workflow (2025-10-01): https://www.nextdiffusion.ai/tutorials/consistent-poses-qwen-image-edit-2509-controlnet-union-comfyui
- Qwen-Image-Edit 2511 pixel shift (2026-07-06, upd. 2026-09-22): https://lilting.ch/en/articles/qwen-image-edit-pixel-perfect-referencelatent
- Z-Image Fun ControlNet Union 2.1 (base, Apache-2.0): https://huggingface.co/alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1 ; Turbo: https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1
- Qwen-Image 2.1 Fun ControlNet news (2026-09-24): https://comfyui-wiki.com/en/news/2026-09-24-qwen-image-2-1-fun-controlnet-union
- Stability licence and SD3.5 ControlNets: https://stability.ai/license , https://stability.ai/news/sd3-5-large-controlnets
- FLUX.2 licences: https://docs.bfl.ai/flux_2
- Generic img2img denoise bands: https://www.mintlify.com/Comfy-Org/ComfyUI/guides/image-to-image , https://morphic.com/ai-glossary/image-to-image
- Local code read: ComfyUI 0.42.1 `comfy/controlnet.py` (Qwen InstantX/Fun loaders, `QwenFunControlNet`),
  `comfy_extras/nodes_model_patch.py` (`ModelPatchLoader`, `QwenImageDiffsynthControlnet`, `ZImageFunControlnet`),
  `comfy_extras/nodes_flux.py` (`FluxKontextImageScale`), `comfy_extras/nodes_qwen.py` (`TextEncodeQwenImageEditPlus`),
  `comfy/utils.py` (`common_upscale`); cinestudio `background.py`, `bpy_jobs/render_set.py`, `bpy_jobs/plate_world.py`,
  `workflows/qwenedit_edit_api.json`, `workflows/qwenimage_pano_api.json`.
