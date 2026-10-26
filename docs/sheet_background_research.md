# Is black the best background for character sheets (for AI video)? Research 2026-10-24

User: "is black the best color for character sheets? for ai video consumption?" Research only. No code was changed and
no GPU job was run.

Tags: [SRC] = checked in a source, [OURS] = our own code or notes, [INF] = inference, [UNVERIFIED] = third-party or
not checked.

## Short answer

Black is right for the sheet's **canvas and gutters**. It is probably not the best colour for the **panels** behind
a character in a black bodysuit. Our grey-panel try failed because of how we cut the figure out, not because of the
model.

## 1. What Lightricks uses

- **Card text:** the Ingredients 2.5 card asks for "a single composite image with one clean panel per distinct visual
  element ... laid out on a black background with no text". It also says "The more space an element takes up ... the
  more faithfully it carries over". It says nothing about panel colour or cut-outs, or about the sheet background
  leaking into the video. [SRC: HF card]
- **Card example prompts:** one prop panel is described "against a white background". [SRC]
- **The pack's example sheet** (`ComfyUI-LTXVideo/example_workflows/assets/ingredients_input.jpg`, pixel samples):
  - The black is the canvas and the gutters only (RGB 0-4).
  - Every character, prop and animal panel is a **light-grey studio backdrop**: RGB 173-196, about L* 71-79, with a
    soft falloff and contact shadows.
  - The location panels are photos. [SRC]
- **So "black background" means the canvas.** The panels are light grey. [INF]
- **Example workflows:** the 2.3 and 2.5 example workflows never name a panel colour. [SRC]
- **Layout-To-Render card:** nothing on backgrounds. [SRC]

## 2. Other reference-to-video systems

- **Wan VACE:** "Salient object segmentation can be performed to keep the background white" for references. VACE also
  uses grey to mark the area to be generated. [SRC]
- **SkyReels-A2:** each subject is segmented "with white grounding ... to avoid additional noise". [SRC]
- **MAGREF:** subjects are cut out with SAM2 and placed on a blank canvas; the colour is not stated. [SRC]
- **Phantom, OpenS2V-Nexus, Kaleido:**
  - When references come from the target clip, models copy their background, pose and light ("copy-paste").
  - The fixes are cross-video pairs and segmentation of the reference.
  - None of these papers ablates the background *colour*. [SRC]
- **HunyuanCustom:** it fights copy-paste with a latent shift, not with a background step. [SRC]
- **Kling Elements:** a front image plus 1-3 extra angles; no colour advice. [SRC] Runway (via third parties) says
  "clean, uncluttered / neutral backdrop". [UNVERIFIED]
- **Vidu:** no background colour advice. [SRC]
- **Consensus:** a plain, uniform, segmented background, usually white or light. No paper prefers black, and none
  tests the colour. [INF]

## 3. The image models and the VAE

- **No published comparison** of white, grey and black backgrounds exists for Qwen-Image-Edit-2511 or Z-Image.
  [UNVERIFIED]
- **Our own failures on black** [OURS, AGENTS.md]:
  - A chain of edits on near-black grew pixel bars.
  - Black gloves on black came out as "thin dark claws".
  - Black on black hides the silhouette from the model. [INF]
- **Light imported backgrounds** leave halos; `clean_edges` treats them. [OURS]
- **Why the grey panels failed:** `charsheet.assemble_panels` finds the figure with `~background_mask(it, 2)`, a flood
  fill from the border with a tolerance of 2 (refsheet.py:29, charsheet.py:125). Near-black suit pixels that touch the
  black background count as background, so on grey "arms and thighs vanished". This is a MATTING bug, not evidence
  against grey. [OURS]
- **Resolution matters more than colour here** [INF]:
  - At 768x448, LTX's VAE (32x spatial) makes a ~150 px body panel ~5 latent cells wide.
  - The outline of the gloves and the suit needs luminance contrast to survive that compression.
- **Animation practice** [UNVERIFIED, forums]: model sheets are traditionally on white, flat-lit; 3D turnarounds are on
  neutral grey.

## 4. Dark versus light costumes

Outline contrast (black suit about L* 10-15, white mask and bandages about L* 90) [INF]:

| Panel | Black suit | White parts |
|---|---|---|
| Black | ~0 (the outline is gone) | ~90 |
| Light grey (Lightricks, L* ~77) | ~65 | ~15 |
| Mid-grey (L* ~50-55) | ~40 | ~40 |

- **A strongly coloured background** risks a colour cast and leaking into the video. [INF]
- **A background in the scene's palette** goes against "clean panel". The place belongs in its own location panel.
  [SRC/INF]
- **Leaking of the sheet background into the LTX video:** no source covers it. In layout runs the layout and the look
  image set the background, and we have seen no black void. [UNVERIFIED/OURS]

## 5. Recommendation

1. Keep the black canvas and gutters.
2. Cut figures out with a real matte instead of the colour flood fill: SAM2 with a box prompt (installed), or BiRefNet
   (MIT). Keep the alpha.
3. A/B neutral-grey panels: RGB ~190 (Lightricks-like) against ~128 (mid-grey), against black. A possible later
   default: choose the panel lightness per character that maximises the weakest outline contrast. Never use a strong
   hue.
4. Separately, for MAKING the views, test a mid-grey edit background against black for dark characters.

## A/B test design

**Stage 1: images only, no GPU**
- Sheets for masked3d_rest and kitsune_rest, each with black, RGB 190 and RGB 128 panels, all from one alpha matte.
- Measure:
  - silhouette recall (should be about 100% on every panel);
  - outline ΔL* per part (gloves, suit legs, mask, bandages), reporting the minimum;
  - halo;
  - the same numbers after downscaling to 768x448 (optionally after an LTX VAE round trip).

**Stage 1b: one Qwen view edit**
- Front -> right view, on black against grey 128, same seed.
- Measure: fingers, glove width, background uniformity, suit colour drift.

**Stage 2: one preview clip pair (later two seeds)**
- Only the panel colour changes.
- Measure: a blind critic's CHARACTER score, void patches, `run score`, `brightness_check`.

## Sources

- https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients ;
  https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render
- Local: D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI-LTXVideo\example_workflows\assets\ingredients_input.jpg
- VACE UserGuide: https://raw.githubusercontent.com/ali-vilab/VACE/main/UserGuide.md
- SkyReels-A2 https://arxiv.org/html/2504.02436 ; MAGREF https://arxiv.org/html/2505.23742 ; Phantom
  https://arxiv.org/abs/2502.11079 ; Phantom-Data https://arxiv.org/html/2506.18851v1 ; OpenS2V-Nexus
  https://arxiv.org/html/2505.20292 ; Kaleido https://arxiv.org/html/2510.18573 ; HunyuanCustom
  https://arxiv.org/html/2505.04512
- Kling Element Library https://kling.ai/quickstart/klingai-element-library-3-user-guide ; Runway (third party)
  https://morphic.com/ai-glossary/runway-gen-4 ; Vidu https://www.vidu.com/en/blog/how-to-maintain-character-consistency-in-ai-videos
- Lin et al. 2023 (noise schedules) https://arxiv.org/abs/2305.08891 ; forums: https://polycount.com/discussion/comment/743287 ,
  https://forum.svslearn.com/post/130628 ; BiRefNet https://github.com/ZhengPeng7/BiRefNet
- Our code: cinestudio/charsheet.py (assemble_panels), cinestudio/refsheet.py (background_mask), AGENTS.md
