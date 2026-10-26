# One style for the character and the place: techniques, evidence and what to test in cinestudio

Date 2026-10-24. Research only: no code changed, no GPU job run. This report follows docs/image_prompt_research.md,
docs/ltx_long_drift_research.md and docs/color_consistency_research.md and does not repeat them. Those reports cover
colour and brightness drift across chained clips. This one covers **style**: line work, shading model, texture,
materials, palette and light, and whether the character and the place look as if one renderer drew them.

Tags:
- [SRC] = what a source says (URL in Sources)
- [CODE] = read in our repo or in an installed node pack
- [OURS] = our own runs and images, which I looked at for this report
- [INF] = my inference
- [UNVERIFIED] = I could not check it

## 0. Short answer

The technique we are not using properly is simple. **One style is decided once, in the images, and every image the
video model sees is in that style: the world (panorama or backplate), the character sheet, and above all the look image
and keyframes.** An image model draws them, and each is checked against one style reference. The video model's style
LoRA and the style text then only *agree* with those images; they cannot repair a mixed set.

Every source that is specific on this point agrees:

- **Lightricks.** The Layout-To-Render card says the look image "sets palette, lighting, and materials", and: "If the
  video ignores the art direction, the first frame is not the picture you want" [SRC]. Lightricks' 2D-animation guide
  says "Generate or create your first frame in the target style ... the image handles the visual style", and keep clips
  short "to prevent style drift toward photorealism" [SRC].
- **The research.** SDEdit, StyleAligned, InstantStyle and StyleMaster all get harmony by letting ONE model pass draw
  all the content, or by sharing one style signal across all images. Pasting parts made by different generators and
  fixing them afterwards is not how they do it.
- **The commercial tools.** Runway References, Kling Elements / O1 and Runway Aleph all generate the character and the
  scene in one pass from references; none of them composites.

[OURS] Our own runs show it. We have one clear success:
- **DL0b** (2026-10-21) had a drawn-looking rooftop panorama and the anime character (cloaked). The look edit was
  told to "keep the place exactly the same", but Qwen redrew the WHOLE frame in the character's anime style.
- The video came out clean anime in every frame, including set pieces that were never in the look image (water tanks,
  a roof hut drawn with outlines). The blind critic put it first by far.

Our recent runs break the rule. The duo look image of r20261023-170716-f5b7 holds three styles in one frame:
- a photographic forest backplate;
- a semi-realistic 3D masked3d;
- a flat 2D sticker-like kitsune.

The video then splits the difference: a photoreal forest with 3D-ish characters.

**What we do wrong now:**
- LOOK_INSTRUCTION ends "Keep the camera, the place, the floor, the sky and the light exactly the same" [CODE
  runner.py:2175]. This freezes a photographic background, forbids relighting the character, and asks Qwen to paste a
  flat-lit sheet figure into it.
- The backgrounds come from a self-described "realism finetune" (the 360 LoRA). The backplate also comes from
  Qwen-Image-2512, which ignored style words before.
- Nothing ties the three image sources (sheet, world, look) to one style reference.
- The LTX LoRA and the style text are asked to restyle after the fact, which the card says will not work.

**The top 3 to try** (details in §5):
1. **Style the world.** Make the backplate or panorama in the target style before the layout render. Use Qwen-Image-Edit-2511,
   either with the dx8152 2511 Style-Transfer LoRA (Picture 2 = a style reference) or a plain text restyle (already
   proven on the forest panorama). Keep a clay-layout control arm in the same A/B.
2. **A one-renderer look image and keyframes.** Make the look and every keyframe in the target style over the whole
   frame:
   - change the look instruction so it no longer freezes the place and the light;
   - add a second "unify" edit (Style-Transfer LoRA, Picture 2 = the style reference);
   - give the same style reference to every image step (a "style bible").
3. **One style source in the video model.** Drop or lower the LTX style LoRA when the images already carry the
   style, and fix the style sentence: the card says to avoid "3D" and "clay" and to "match the first frame". Later,
   train our own LTX-2.5 style LoRA on our unified look images on a rented 32 GB+ GPU (still images are supported;
   the 3dsrx LoRA was 54 images).

## 1. What we have now (the facts this report builds on)

**Look image.** [CODE runner.py:2175-2200] Qwen-Image-Edit-2511 edits Blender frame 1, which already contains the
panorama or backplate. Picture 2 is the character's front view. The instruction:

> "Replace the white wooden mannequin with the character from image 2 ... The character keeps every detail from image 2:
> {who} {extra} Keep the camera, the place, the floor, the sky and the light exactly the same."

- There are no style words (AGENTS.md: "the Qwen look instruction has no style").
- For duo runs, each figure is a crop-and-stitch edit of its own; there is no whole-frame pass afterwards. So each
  figure keeps its own sheet's style, and the soft paste edge is the only blending.

**Scene option.** `look_scene` [CODE runner.py:2399] makes a second edit that swaps the grey room for a place described
in words. This is the closest thing we have to "one renderer", because the same model draws the place after the
character.

**Backplate.** Qwen-Image-2512 makes it through the panorama graph, with the 360 LoRA at 0. PLATE_PROMPT does pass the
run's style text [CODE runner.py:2109-2166], but the forest plates in the recent runs are photographic.

**Panorama.** Qwen-Image-2512 + the qwen-360 LoRA, which is "a realism finetune" (AGENTS.md).
- The LTX `redraw` of the panorama with the 3D LoRA did not restyle it: at keep 0.15 it changed the content and stayed
  photographic.
- A Qwen-Image-Edit TEXT restyle ("Redraw this whole picture as a hand-painted 2D anime film background ...",
  `.cinestudio/gauntlet/forest_2d/qwen_style_pano.py`) **worked**. [OURS] I looked at forest_front_2d_out.png: a clean
  2D anime forest in the same layout.
- Its cost: the edit graph scales to ~1 MP, so the 2048-wide panorama came back at 1504x688.

**Video.** Layout-To-Render distilled LTX-2.5:
- the layout render is the guide at frame 0;
- the look is at frame -1;
- the Ingredients sheet (5 or 6 character panels on black, no location panel);
- optionally the 3dsrx LoRA at 0.4, with a style sentence that starts "3dsrx, premium stylized cinematic 3D animation ...".

**Evidence in our own runs.**

| Run | World | Look | Result |
|---|---|---|---|
| DL0b / DL1b (ltx25_ab) | drawn-looking rooftop panorama | whole frame redrawn anime by Qwen (look_r20261021-030309-5a37.png) | clean anime everywhere; critic first by far |
| S2 / S5 / C1 (v2v, ltx25_ab) | same panorama | none (v2v) / anime words | "fully photographic" / "clean 3D look, NOT anime: the prompt alone is not enough" |
| F1 (v2v + anime frame 0) | same | anime frame pinned | frame 0 anime, then photographic: one frame does not hold style in v2v |
| f5b7 duo (2026-10-23) | photographic forest plate | photo background + 3D-ish masked3d + 2D kitsune | photoreal forest, characters pulled to semi-3D by the LoRA |
| a498 vs b2c2 (style text A/B) | 2D forest panorama | same look | anime text gave better legs; style text matters a little but does not decide |

[INF] The pattern: when the look image (and the world, in DL0b) is in one style, LTX follows it. When the inputs
disagree, LTX averages them, leaning toward photoreal (its training prior, per Lightricks).

## 2. Techniques, one by one

For each technique: what it does, the evidence, the licence, whether it fits 8 GB, and where it would go in our pipeline.

### 2.1 One image pass draws the character and the place together (a unified look image)

**What.** The look image and the keyframes are drawn by ONE image model over the whole frame, so the line work,
shading and light are shared. This is the opposite of "keep the place exactly the same and paste the character in".

**Evidence.**
- Layout-To-Render card: the first frame "sets palette, lighting, and materials"; "If the video ignores the art
  direction, the first frame is not the picture you want"; add a mid frame and a last frame "when the picture should
  hold through the shot" [SRC].
- LTX blog (2026-04-26): "Image-to-video conditioning solves this. By providing a reference image in the style you
  want, you lock the visual language from the first frame" [SRC].
- SDEdit (Meng et al., ICLR 2022) shows that noising a composite and denoising it with one model harmonizes it ("image
  compositing" is one of its three named uses) [SRC].
- Commercial tools generate the subject and the scene in one model call from references: Runway Gen-4 References takes
  up to 3 references named image_1..3 in the prompt; Kling Elements / O1 take separate style or background references
  [SRC]. They do not composite.

**Licence and VRAM.** Qwen-Image-Edit-2511 is Apache-2.0 and already runs here (fp8, ~12 min per 40-step edit).

**Where it goes.** step_look (and the keyframe looks, the duo crop-and-stitch, the look_scene edit). Three variants:
- (a) Change LOOK_INSTRUCTION: drop "and the light", and add a closing sentence that draws the whole picture in one
  style: "Draw the whole picture, the character and the place, in one consistent style: <style words>; the
  character is lit by the same light as the place." Keep "one change per edit" in mind: (b) is the cleaner form.
- (b) A second **unify edit** of the finished look, done by the same model: "Redraw the whole picture in the style of
  Picture 2 ... Keep the composition, the character's pose, position, size and costume, and the camera exactly the
  same." Use it with the 2511 Style-Transfer LoRA (§2.3) or with text only. For duo runs this pass removes the paste
  edges and the 2D/3D clash.
- (c) The "clay -> render" route: start from the GREY Blender frame and let two edits draw character and place (our
  `look_scene`). The same model draws both, so they share a renderer.

**Risk.** The unify edit can change identity details: the mask shape, garment lengths (AGENTS.md: the look decides the
costume), or the pose. Two guards help:
- check the look against the mask as we already do;
- [INF] if the costume is still lost, run the pass with a lower denoise, or with a soft mask that keeps the figure
  (our edit graph has a denoise input; ComfyUI core's DifferentialDiffusion node takes a grey-level mask: figure
  ~0.4, place ~0.8).

**A/B.** See E2 in §4.

### 2.2 Style the world before the layout render (the world in the target style)

**What.** Restyle the panorama or backplate itself in the target style, so the LAYOUT video, which is a guide on every
frame, is already in the right style. Today it carries photographic texture into every frame.

**Evidence.**
- Lightricks: the layout is meant to be "a detailed clay render, or a blocking animation built from simple shapes";
  materials come from the look [SRC]. A photographic textured layout is outside that description [INF].
- [OURS] DL0b worked on a drawn-looking rooftop, while the forest runs on a photographic plate stayed photoreal behind
  the characters. The text restyle of the forest panorama with Qwen worked by eye.
- Kling's docs suggest a separate style image beside the background image [SRC], which is the same idea: the place has
  its own style source.

**Licence and VRAM.** Qwen-Image-Edit-2511 (Apache-2.0, fits); the dx8152 2511 Style-Transfer LoRA (Apache-2.0).

**Where it goes.** The `background` step:
- Backplate: after `_plate_background` makes or loads the plate, run one Qwen edit (text restyle or Style-Transfer
  with the style reference) before plate_world.py. The plate is frame-shaped (~2 MP); at ~1 MP it is still ~1340 px
  wide, enough for preview (768) and close to final (1536).
- Panorama: the forest_2d script exists. To avoid the 1504x688 loss [INF], restyle overlapping tiles (for example 4
  tiles of 1024x1024 with a 25% wrap overlap, same seed and same reference) and feather the joins, or upscale the
  result before saving. The depth (MoGe) is reused: the layout does not change.
- Save it as a new library background (as `background style` already does) so it can be reused.

**Control arm.** The research ranked a CLAY layout A/B as open item (3) in "Background control: research". Do it in
the same experiment: grey depth-mesh world plus a styled look, against a styled world plus the same look.

### 2.3 Style references in the image model (Picture 2 = the style)

There are several ways to bring a style reference into our image steps.

**dx8152 / Qwen-Image-Edit-2511-Style-Transfer** (Apache-2.0, updated 2026-03-15)
- Trigger "style transfer"; prompt "Change the style of Figure 1 to the style of Figure 2." [SRC]
- Built for exactly our 2511 base.
- [UNVERIFIED] Strength, steps, and whether it needs the Lightning LoRA. dx8152's 2509 LoRAs say to use Lightning, and
  we refuse fast modes, so test at 40 steps / cfg 3 first and at the card's setting second.
- Slot: the unify edit (2.1b), the world restyle (2.2), and converting one character's sheet to the other's style (2.8).

**peteromallet / Qwen-Image-Edit-InStyle** (Apache-2.0)
- "Make an image in this style of ..." makes a NEW image from a style reference and reduces content leakage [SRC].
- It is for the ORIGINAL Qwen-Image-Edit, not 2511. [UNVERIFIED] whether it loads on 2511.
- It was trained on Midjourney --sref outputs (a provenance question to note, not a licence term).
- Slot: making a backplate or a location panel FROM the style reference.

**Plain Qwen-Image-Edit-2511 with two pictures, no LoRA**
- Multi-image input is native: ComfyUI labels them "Picture 1/2" (image_prompt_research.md).
- Our own note: a whole image as Picture 2 "was copied too much" (extra arm). For a style pass this content leakage
  is less harmful when Picture 2 is the same character (the leakage then reinforces identity).
- Slot: the cheapest test of 2.1b.

**InstantStyle / IP-Adapter style blocks (SDXL), StyleAligned (shared attention, SDXL; code Apache-2.0)**
- The research form of "one style signal for a whole set" [SRC].
- They need an SDXL-family model; good commercial anime SDXL checkpoints usually carry restrictive licences
  [UNVERIFIED per checkpoint]. comfyui_ipadapter_plus is installed [CODE], but our characters are Qwen/Z-Image made.
- Rank low: a second model family for little gain over 2511 multi-image.

**FLUX.2 [klein] 4B** (Apache-2.0; the 9B is non-commercial)
- One model for text-to-image, editing and multi-reference; consumer-GPU sized [SRC].
- A possible faster style-reference editor if Qwen is too slow. [UNVERIFIED] Its quality on stylized characters.
  dx8152's Klein "Migration" / "Consistency" LoRAs are for the 9B, which we cannot use.

**Ruled out:** FLUX.1 Kontext dev / Redux dev ("other": the FLUX non-commercial licence).

### 2.4 Relighting and harmonization (the light, not the line work)

These fix "the character is lit differently". They do not fix "it is drawn differently".

**Qwen-Image-Edit-2511 built-in lighting**
- 2511 "integrates popular LoRAs" including lighting [SRC]. A text instruction "relight the character to match the
  scene's light" is free to test.

**dx8152 Light-Migration (2509, Apache-2.0)**
- Picture 1 = the image to relight, Picture 2 = the light reference.
- Prompt (Chinese): "参考色调，移除图1原有的光照并参考图2的光照和色调对图1重新照明" [SRC].
- [UNVERIFIED] on 2511 and on illustrations.

**dx8152 Fusion (2509, Apache-2.0)**
- Blends a pasted product into a background, "correcting perspective and lighting" [SRC]. It is a 2509 LoRA.
- A community ComfyUI workflow, "Relight and Environment Merge" (araminta-k), stacks an illustration LoRA with a relight
  LoRA on Qwen-Image-Edit to "properly light illustrated characters in a keyframe" [SRC]. That is our exact case,
  though the LoRA names are not on the page.

**IC-Light v1** (SD1.5)
- Repo Apache-2.0; the weights' HF repo carries no licence tag in the API (the card says creativeml-openrail-m).
- It is photo-oriented and works at 512-768 px.
- Light-A-Video (ICCV 2025, Apache-2.0) uses it for training-free video relighting [SRC].
- Low value for a stylized look [INF].

**Excluded or out of reach:**
- LBM relighting (Jasper AI): CC BY-NC 4.0.
- GenCompositor (ICLR 2026): generative video compositing with shadows and luminance harmonization [SRC]; repo licence
  NOASSERTION; photoreal; not our pipeline.
- TC-Light (NeurIPS 2025): licence not found.

**LTX-2.3 IC-LoRA Relight** (LTX-2.x community licence)
- Relights a whole exterior clip to a sun direction given by a light ball composited into the frame, plus 12 trained
  looks in text.
- Single-stage only; exteriors; trained on 2.3 [SRC].
- It relights character and place together (one renderer), but it is photo-trained [INF]. Keep it in mind for scene
  light changes, not for style.

**Colour-statistics harmonization** (Reinhard / MKL of the figure toward the place)
- Fixes the colour cast only. Our earlier colour work shows that hue moves are risky (the blue jacket).
- Use only as a light finishing touch on the look's figure.

### 2.5 The style LoRA and the style text in the video model

**Evidence.**
- [OURS] The 2.3 Fantasy_Anime LoRA made 2.5 worse (S1, take 13). The 3dsrx 2.5 LoRA was trained on 54 images at rank 32
  (Civitai card, via search; the page itself returned 403 to the fetch), with suggested strengths 0.8 t2v / 0.4 i2v.
- Layout-To-Render card: write "one or two sentences describing the finished shot", "Match the first frame", and avoid
  "clay", "3D" or "Unreal" [SRC]. [INF] Our style sentence leads with "3dsrx, premium stylized cinematic 3D animation,
  polished ..." and so asks for the words the card warns against. It also carries the "premium / polished /
  cinematic" meta words that image_prompt_research.md already flagged for the image models.
- LTX blog: style vocabulary first; image conditioning locks the style; IC-LoRA "trained on paired photorealistic and
  stylized versions of the same footage" for systematic style transfer; keep clips short (2-4 s) against drift to
  photorealism; IC-LoRA only on the distilled model [SRC].

**Where it goes.**
- Clip prompt (`prompt.style`): when the look carries the style, make the sentence describe the look ("the hand-painted
  2D anime look of the first frame ..."), drop "3D" and the meta words, and A/B the LoRA at 0 / 0.2 / 0.4.
- The LoRA must match the images: a 3D LoRA on a 2D look fights it [INF].

### 2.6 Train our own LTX-2.5 style LoRA (or a style IC-LoRA)

**What.** A LoRA trained on ~50-100 of our own unified look images and keyframes in the chosen style, characters AND
scenes together, so the video model's prior learns our exact style.

**Evidence.**
- ltx-trainer accepts still images (bucket `WxHx1`, e.g. 960x544x1; mixed image and video datasets with batch size 1;
  the trigger is added by `--lora-trigger`) [SRC].
- Hardware: 80 GB recommended; a 32 GB low-VRAM INT8 config exists (t2v_lora_low_vram.yaml) [SRC]. A rental is needed.
- The 3dsrx precedent: 54 images, 1250 steps, rank 32 [SRC via search].
- A style IC-LoRA (paired "Blender layout render -> unified look" frames) is Lightricks' own suggestion for systematic
  style transfer [SRC]. It is a bigger project and overlaps with what Layout-To-Render already does.

**Licence.** Our own LoRA on the LTX-2.x community licence. The training images are ours (Qwen/Z-Image outputs,
Apache-2.0 models).

**Where it goes.** The clip graphs (`style_lora`), only after 2.1-2.2 have produced a set of approved unified images.

### 2.7 A location panel in the Ingredients sheet

**What.** Add one clean location panel (the styled backplate or a styled view of the panorama) to the Ingredients
sheet, drawn in the same style as the character panels.

**Evidence.**
- The Ingredients card recommends "one clean location panel per sheet". Its example sheets have a setting in the
  bottom row, and the model reads the sheet "to match characters, props, and setting" [SRC].
- The card's own examples are 3D animated in every panel; the style cues are in the prompts [SRC].
- We have only character panels on black (AGENTS.md "Sheet layout ...").

**Cost.** Panel area: "bigger panels carry over better". In the duo sheet, 6 panels already share 768x448 [INF].

**Where it goes.** character.build_sheet / charsheet (a run-level sheet variant), plus a "**Location panel:** ..."
line in the Reference Sheet Description.

### 2.8 One style across two characters (duo runs)

**What.** When two characters come from different styles (masked3d is semi-3D, kitsune is flat 2D), convert one
character's sheet to the other's style, or both to the run's style, BEFORE the run. The 2511 Style-Transfer LoRA does
this per view with Picture 2 = the other character's front. Check the views as usual (hands, colours, sides).

**Evidence.** [OURS] The f5b7 look image shows the clash. LTX + 3dsrx pulled kitsune to 3D, but only partly.

**Where it goes.** `character style` (it exists, but uses the LTX redraw, which barely restyles). Add a Qwen
Style-Transfer mode.

### 2.9 Video-level restyle of the whole clip (one renderer at the end)

**What.** A second low-noise video-to-video pass over the finished clip with the style LoRA, so every pixel passes
through the same renderer (SDEdit applied to video).

**Evidence and cost.**
- [OURS] The LTX redraw barely restyled at keep 0.3 and changed content at 0.15, and pinning an anime frame in v2v did
  not hold style. Low expected value.
- Wan2.1 VACE (Apache-2.0; 1.3B fits ~8 GB) and Wan2.2 (Apache-2.0) do reference-image + depth/pose-controlled video
  restyle. The community recipe is "restyle the first frame (Kontext or a reference), propagate with depth/canny/pose
  control" (RunComfy, ComfyUI.org, Stable-Diffusion-Art, 2025) [SRC].
- That is the recipe we already run with LTX Layout-To-Render. A second video model would lose LTX's audio and
  detail and add another style source.
- StyleMaster (CVPR 2025), OmniTransfer (2026) and Runway Aleph are research or closed equivalents.
- LTX-2.5 Refine-Details explicitly does not change look or colour [SRC], so it is not a restyler.

**Verdict.** Not recommended now.

### 2.10 What professional animation does (an analogy)

[SRC, a single secondary source; UNVERIFIED in detail]
- In anime production the art director's boards are the reference for every background painter.
- Colour design (iro-shitei) is "tightly linked" to the art so the palette stays balanced.
- Compositing / photography (satsuei) layers backgrounds and characters and adds shared optical and colour filters and
  light.

[INF] The lesson matches §2.1-2.2: one style bible decided before production; per-scene colour models for the
characters (a night palette for a night scene); and a shared final filter pass. Our "style bible" image and the
unify edit are the AI form of the art board and the satsuei pass.

## 3. Where each technique goes in our pipeline

| Pipeline step | Today | Proposed |
|---|---|---|
| Style choice | style text per run + a LoRA | a **style bible**: one reference image (e.g. the approved front view, or a styled frame) + one style sentence, stored per run (or per character) |
| Character sheet | Z-Image / Qwen style text | views checked against the bible; Qwen Style-Transfer to convert a character to the run's style (duo) |
| Background (plate / panorama) | Qwen-Image-2512 (realism 360 LoRA) | + one Qwen-Image-Edit restyle (text or Style-Transfer with the bible), tiles for panoramas, depth reused |
| Layout render | textured photo world | the styled world, OR clay (A/B) |
| Look image | character pasted, "keep the place and the light" | instruction without the freeze + a whole-frame **unify edit** with the bible |
| Keyframe looks | Qwen edits of photo frames, sheet pastes | the same unify edit (sheet pastes too: a pasted sheet panel is flat-lit) |
| Ingredients sheet | character panels on black | + one location panel in the same style (A/B) |
| Clip prompt | "3dsrx, premium stylized cinematic 3D animation ..." | describe the look ("the ... look of the first frame"), no "3D"/meta words; LoRA only when it matches the bible |
| Video model LoRA | 3dsrx 0.4 / none | 0 when images carry the style (A/B), later our own trained LoRA |
| Chained clips | chained look + AdaIN clip 1 | unchanged; style drift is checked with the harmony measure (below) |

## 4. Experiments (one change each, preview, same seed)

**Common settings.**
- Shot: `shan_shen_bu` (still camera, so the backplate route applies).
- Character: masked3d_rest + its proxy kit; seed 202; current defaults (P4: chain_light tone etc.).
- `--quality preview --through-clip 1` unless the test is about drift.
- Image-only stages first (each Qwen edit ~12 min); a video only for the image winner.

**A new critic line for docs/gauntlet_critic.md.** STYLE HARMONY 1-10: "Do the character and the place look drawn by
the same artist or renderer, under the same light? Line work, shading, texture detail, saturation." Also: style drift
between frame 0 and the last frame.

**A numeric proxy (a new script, not built).** Per frame, use the Blender mask (eroded for the figure; a ring 10-40 px
outside it for the place) and compare:
1. CSD style-embedding cosine (Contrastive Style Descriptors; code MIT, arXiv 2404.01292; weights licence UNVERIFIED,
   evaluation use only);
2. edge density (Canny edges per pixel);
3. high-frequency energy (Laplacian variance);
4. Lab chroma;
5. local contrast.

Also compute the CSD drift of each region against frame 0. As with the motion score: the numbers filter, the critic
decides.

**E1. Style the world (backplate).**
- Arms (one look procedure in every arm, the current one):
  - A = the current photographic plate;
  - B = the plate restyled by a Qwen text instruction (the forest_2d wording adapted to the run style);
  - C = the plate restyled with the 2511 Style-Transfer LoRA, Picture 2 = the character's front view.
- Pass: the critic ranks B or C over A on STYLE HARMONY with CHARACTER not lower; the harmony proxy moves toward 1.

**E2. A unified look image (images only, then one video).**
- Arms:
  - L0 = the current look;
  - L1 = the look instruction without "and the light" + the one-style sentence;
  - L2 = L0 + a unify edit, text only;
  - L3 = L0 + a unify edit with the Style-Transfer LoRA (Picture 2 = the front view);
  - L4 = the grey-frame route (look_scene).
- Check every look against the Blender mask (size, place) and the costume.
- The best one or two get a clip 1 video against L0.
- Repeat for the duo shot, where the paste edges and the 2D/3D clash are worst.

**E3. Layout texture: photo vs styled vs clay.**
- Same unified look (the E2 winner) in each arm. World arms:
  - the photo plate;
  - the styled plate (the E1 winner);
  - clay (the world hidden, a grey floor and wall; plate_world can show a grey material).
- This settles whether the layout's texture or the look decides the background style. AGENTS.md lists it as an open
  A/B.

**E4. The video LoRA when the images carry the style.** The E2/E3 winner with 3dsrx at 0 / 0.2 / 0.4. With the 2D
style, the 3D LoRA is expected to hurt [INF].

**E5. The style sentence.** The current sentence vs "Style: <plain description of the look>, matching the first frame"
(no "3D", no "premium / polished / cinematic").

**E6. A location panel in the Ingredients sheet.** The current sheet vs + one styled location panel (bottom-right, as
large as the side panels).

**E7. Duo: one style for both characters.** kitsune_rest's views converted to masked3d_rest's style (or both to the run
style) with Style-Transfer, then the duo run against r20261023-170716-f5b7's settings.

**E8 (rental, later). Our own LTX-2.5 style LoRA.**
- Data: 60-100 approved unified images (looks, keyframes, styled plates, sheet views) in one style.
- Training: 960x544x1 buckets, rank 32, ~1250-2000 steps, the low-VRAM config on a 32 GB card.
- A/B against no LoRA on E2's winner, clips 1-3 (style drift over the shot).

**Relighting side test (cheap, images only).** The look's figure relit to the place, three arms:
- 2511 text "relight the character to match the light and colour of the place. Keep everything else exactly the same.";
- Light-Migration (2509 LoRA on 2511, UNVERIFIED);
- none.

## 5. Ranked recommendation

**1. Style the world AND unify the look with one style bible (E1 + E2 together, then E3).**

Why first:
- It matches Lightricks' cards and blog exactly.
- It matches our one clean success (DL0b) and explains our failure (f5b7).
- It uses only tools we already run (Qwen-Image-Edit-2511) plus one Apache-2.0 LoRA (dx8152 2511 Style-Transfer).
- The text restyle of the panorama is already proven by eye.

Product shape [INF]:
- `run new --style-ref FILE` (default: the character's front view) feeds the backplate or panorama restyle, the
  unify edit of the look and every keyframe;
- LOOK_INSTRUCTION stops freezing "the light";
- in duo runs the unify edit runs over the stitched frame.

Cost: +1-2 Qwen edits per run (~12-25 min at 40 steps).

**2. One style source in the video model (E4 + E5).** It is free: drop or lower the LTX LoRA when the images carry the
style, and rewrite the style sentence per the Layout card ("match the first frame", no "3D"/"clay", no meta words).
Then decide whether a trained LoRA (E8) is worth a rental.

**3. Clay vs styled layout (E3), then the location panel (E6).** E3 answers the structural question: does the textured
photo world in the layout guide drag every frame toward photoreal? If clay plus a unified look is as good or better,
the panorama's job shrinks to geometry and depth (the research's "clay A/B"), and styling the world becomes optional.
E6 is cheap and is the card's own advice.

Not now:
- a video-level restyle pass (2.9);
- a second video model (Wan VACE);
- SDXL style adapters (2.3);
- photo relighting models (IC-Light, LBM, which is non-commercial);
- generative compositing (GenCompositor).

## 6. What we are doing wrong or missing (summary)

1. **The look image pastes instead of renders.** "Keep the place ... and the light exactly the same" freezes a
   photographic background and the sheet's flat light; nothing unifies the style afterwards. Lightricks: the look
   decides the art direction.
2. **Three style sources and no shared reference.** The sheet (Z-Image/Qwen text), the world (a realism-finetuned
   panorama model or Qwen-Image-2512) and LTX (LoRA + text) are styled separately. Duo runs add a fourth: each
   character's own style.
3. **The video LoRA is asked to fix it afterwards.** The card and blog say the image decides; a LoRA that disagrees
   with the look fights it (the 2.3 anime LoRA on 2.5 was worse; 3dsrx on a 2D character only half-converts it).
4. **The style sentence works against the card.** It contains "3D" and "premium / polished / cinematic", which the
   Layout card and our image-prompt research both advise against.
5. **The world is styled, if at all, by a model that cannot style** (the LTX redraw), not by the one that can (a Qwen
   edit, proven on forest_front_2d), and only at ~1 MP for panoramas.
6. **Keyframe looks bring the problem back mid-clip.** They are made from photographic Blender frames, or are flat-lit
   sheet pastes, so each keyframe re-injects the mismatch.
7. **No location panel in the Ingredients sheet**, against the card's advice.
8. **Nothing measures style harmony.** The critic has CHARACTER / MOTION / BACKGROUND / PICTURE but no "does it belong
   together".

## 7. Unverified points

- The dx8152 2511 Style-Transfer LoRA: its strength, steps, whether it needs Lightning, and how well it handles a 3D or
  anime reference. The card shows images with no captions.
- Light-Migration and Fusion (2509 LoRAs) on 2511; InStyle (original Qwen-Image-Edit) on 2511.
- The 3dsrx Civitai card details (the page returned 403 to the fetch; numbers from search snippets).
- InstantStyle's licence (no licence file in the GitHub API); TC-Light's licence; the CSD weights' licence.
- The "Relight and Environment Merge" workflow's LoRA names and strengths (the page does not list them).
- The anime production details (one secondary Italian source).
- Runway / Kling internals: product descriptions only.

## Sources

Lightricks / LTX:
- Layout-To-Render IC-LoRA card: https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render
- Ingredients 2.5 card: https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients
- Relight 2.3 IC-LoRA card: https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Relight
- Refine-Details 2.5 card: https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details
- Lightricks model list (HF API): https://huggingface.co/api/models?author=Lightricks
- "How To Generate 2D Animation With AI Video Models" (LTX Team, 2026-04-26): https://ltx.io/blog/how-to-generate-2d-animation-with-ai-video-models
- ltx-trainer: https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer ; dataset preparation: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/dataset-preparation.md
- LTX "how to use a community LoRA": https://ltx.io/blog/how-to-use-a-community-lora
- 3D Animation Style LoRA (3dsrx), Civitai 2895989: https://civitai.com/models/2895989 (page not fetched: 403)

Qwen-Image-Edit LoRAs and tools:
- Qwen-Image-Edit-2511 docs: https://docs.qwenlm.ai/qwen-image-edit-2511/index.html ; ComfyUI: https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511
- dx8152 Style-Transfer (2511): https://huggingface.co/dx8152/Qwen-Image-Edit-2511-Style-Transfer
- dx8152 Light-Migration: https://huggingface.co/dx8152/Qwen-Edit-2509-Light-Migration ; demo: https://huggingface.co/spaces/multimodalart/Qwen-Image-Light-Migration
- dx8152 Fusion: https://huggingface.co/dx8152/Qwen-Image-Edit-2509-Fusion ; White_film_to_rendering: https://huggingface.co/dx8152/Qwen-Image-Edit-2509-White_film_to_rendering
- dx8152 model list (HF API, licences): https://huggingface.co/api/models?author=dx8152
- peteromallet InStyle: https://huggingface.co/peteromallet/Qwen-Image-Edit-InStyle
- Zaytron40k 2511 StyleTransfer (fixed style): https://huggingface.co/Zaytron40k/Qwen-Image-Edit-2511-StyleTransfer-LoRA
- "Relight and Environment Merge" workflow: https://comfy.org/workflows/50199f76fe74-50199f76fe74/
- FLUX.2 klein (Comfy blog): https://blog.comfy.org/p/flux2-klein-4b-fast-local-image-editing ; HF: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B

Relighting / harmonization / compositing:
- IC-Light: https://github.com/lllyasviel/IC-Light ; weights: https://huggingface.co/lllyasviel/ic-light
- LBM relighting (CC BY-NC 4.0): https://huggingface.co/jasperai/LBM_relighting
- Light-A-Video (ICCV 2025): https://mlanthology.org/iccv/2025/zhou2025iccv-lightavideo ; code: https://github.com/bcmi/Light-A-Video
- TC-Light (NeurIPS 2025): https://arxiv.org/abs/2506.18904
- GenCompositor (ICLR 2026): https://arxiv.org/abs/2509.02460

Style research:
- SDEdit: https://arxiv.org/abs/2108.01073
- StyleAligned: https://arxiv.org/abs/2312.02133 ; code: https://github.com/google/style-aligned
- InstantStyle: https://arxiv.org/abs/2404.02733
- StyleMaster (CVPR 2025): https://arxiv.org/abs/2412.07744
- OmniTransfer: https://arxiv.org/abs/2601.14250
- CSD style descriptors: https://arxiv.org/abs/2404.01292 ; https://github.com/learn2phoenix/CSD
- Differential Diffusion: https://arxiv.org/abs/2306.00950

Video restyle (community):
- Wan2.2 first-frame restyle: https://www.runcomfy.com/comfyui-workflows/wan2-2-first-frame-restyle-comfyui-video-generation
- VACE V2V style: https://www.runcomfy.com/comfyui-workflows/vace-wan2-1-video-to-video-workflow
- VACE tutorial: https://stable-diffusion-art.com/wan-vace-ref/
- VACE: https://github.com/ali-vilab/VACE ; Wan2.2: https://github.com/Wan-Video/Wan2.2

Commercial tools:
- Runway Gen-4 (TechCrunch, 2025-03-31): https://techcrunch.com/2025/03/31/runway-releases-an-impressive-new-video-generating-ai-model
- Gen-4 References guide: https://help.scenario.com/en/articles/runway-gen-4-the-essentials/
- Kling character consistency: https://kling.ai/quickstart/ai-video-character-consistency ; Element Library: https://kling.ai/quickstart/klingai-element-library-3-user-guide
- Kling O1 (Scenario): https://help.scenario.com/en/articles/kling-o1-family-the-essentials

Anime production (secondary): https://www.animeclick.it/news/84205-anime-no-bijutsu-13-gallery-di-sfondi-mozzafiato-dalle-serie-in-corso

Our files used:
- AGENTS.md
- cinestudio/runner.py (LOOK_INSTRUCTION 2175, SCENE_INSTRUCTION 2399, PLATE_PROMPT 2109)
- .cinestudio/gauntlet/ltx25_ab/RESULTS.md and look_r20261021-030309-5a37.png / big_DL0b_060.jpg / strip_SOURCE.jpg
- .cinestudio/gauntlet/forest_2d/qwen_style_pano.py and forest_front_2d_out.png
- .cinestudio/runs/r20261023-170716-f5b7/work/look.png and final.mp4 (frame at 4 s)
- ComfyUI-LTXVideo example_workflows notes (pack 220e1d4)
