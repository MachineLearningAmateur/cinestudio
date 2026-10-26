# Qwen-Image-Edit negatives, output control, and the sheet background: research 2026-10-24

User: "this looks awful", "we need to better control our output", "research what negatives to put if qwen gives that
option", "did we verify whether or not black background is the best color for sheets consumed by video models".

Research only. No project code was changed and no GPU job was run. The CPU checks are described in each section.

This report builds on two earlier reports and does not repeat them:
- docs/image_prompt_research.md (prompt form, "Picture N", the duplicate Picture 2, Z-Image negatives);
- docs/sheet_background_research.md (Lightricks' example sheet, VACE / SkyReels-A2 white references, the matting bug).

Tags:
- [CODE]: checked in ComfyUI's code (v0.42.1, 63be2f9, our pinned install) or in our repo;
- [SRC]: checked in an official source;
- [OURS]: our own runs and notes;
- [MEAS]: measured today on the CPU;
- [INF]: my inference;
- [UNVERIFIED]: a third-party claim, or not checked.

## 0. What went wrong today (E7a_T, kitsune -> 3D)

- **Input and output.** The input `kit_front.png` had a pure black background (border RGB ~1.5). The reference
  `ref_m3d_crop.png` was also on black. The output `E7a_T_kit_to_m3d_text.png` has a border of RGB ~250: a white
  seamless studio backdrop, a grey floor shadow under the boots, and a toy-like "Pixar" render. [MEAS]
- **The instruction** ended with a 12-item list of features to keep: "...the dark trousers and the brown boots, the
  pose and the black background." The black background was the LAST item of that list. Attribute neglect is
  documented for long lists (Attend-and-Excite; see the earlier report). [OURS/INF]
- **The style words** ("a semi-realistic stylized 3D look with soft painted shading, smooth gradients of light on
  rounded forms") match the look of studio product and turntable renders. Those are typically shot on white or grey
  seamless with a soft floor shadow. The model's prior for "3D character render" won over one buried phrase. [INF]
- **Why the white mask was eaten.** Our `black_background` (refsheet.background_mask, a fixed-range flood fill from
  the border with tolerance 18) started on the white backdrop and flowed into the white mask, which touches the
  backdrop. Colour flood fills cannot tell "white backdrop" from "white mask". [CODE]
- **Other drift:** the cape went teal -> green, the trousers went navy, and the skin got darker. The 2D->3D direction
  pulls Qwen's own palette. The colours of the masked3d crop may also leak in. [OURS/INF]

## 1. Negative prompts for Qwen-Image-Edit-2511

### 1.1 Is the negative applied in our graph? Yes, at cfg 3.

- **The official ComfyUI template** (Comfy-Org/workflow_templates `image_qwen_image_edit_2511.json`, downloaded
  today):
  - the negative `TextEncodeQwenImageEditPlus` has prompt `''` and the SAME image1/image2/image3 and VAE as the
    positive;
  - both go through `FluxKontextMultiReferenceLatentMethod index_timestep_zero`;
  - `CFGNorm` strength 1, pre_cfg false;
  - `KSampler` 40 steps, cfg 3, euler/simple, denoise 1.
  - Our `qwenedit_edit_api.json` matches this exactly (plus the optional Lightning LoRA). [CODE]
- **ComfyUI computes the negative branch whenever cfg is not 1.0.** `comfy/samplers.py` `sampling_function` skips the
  uncond only when `math.isclose(cond_scale, 1.0)`. At cfg 3 every step runs the positive and the negative. So
  whatever we put in `neg_txt.prompt` IS used. [CODE]
- **The negative is encoded WITH the images.**
  - `nodes_qwen.py` builds `"Picture 1: <vision> Picture 2: <vision>" + prompt` under the system template ("Describe
    the key features of the input image ... then explain how the user's text instruction should alter or modify the
    image").
  - It then attaches the VAE reference latents.
  - With an empty negative, the uncond means "these pictures, no instruction". So cfg 3 amplifies "follow the
    instruction" over "reproduce the input". The negative is the EDIT-STRENGTH baseline. [CODE/INF]
- **Diffusers does the same** (`pipeline_qwenimage_edit_plus.py`):
  - the negative is encoded with `image=condition_images`;
  - `do_true_cfg = true_cfg_scale > 1 and has_neg_prompt`, and it warns when no negative is given;
  - the default `true_cfg_scale` is 4.0;
  - the result is rescaled to the conditional prediction's norm: `comb_pred * (cond_norm / noise_norm)`. [SRC]
  - ComfyUI's CFGNorm in its default mode (post-cfg) does the same rescale, but only DOWN (clamped to <= 1).
    `pre_cfg=True` would match diffusers exactly. [CODE]
  - "true_cfg" (diffusers) and "cfg" (ComfyUI KSampler) are the same two-pass CFG. "guidance_scale" is an unused
    distilled-guidance input for Qwen (None/1.0). [SRC/INF]
- **DiffSynth-Studio:**
  - `QwenImagePipeline.__call__` defaults to `negative_prompt=""` and `cfg_scale=4.0`;
  - its negative branch also gets the edit images;
  - its 2511 example passes no negative and uses `zero_cond_t=True`. That flag is the counterpart of our
    `index_timestep_zero`. [SRC]
- **Qwen's own examples:** the 2511 and 2509 cards use `negative_prompt=" "` with true_cfg 4.0 and 40 steps.
  Qwen gives NO edit negative list. The only official Qwen negative list is for text-to-image (Qwen-Image-2512,
  Chinese; English in the earlier report §1.3): "low resolution, low quality, deformed limbs, deformed fingers,
  oversaturated, waxy look, faces without detail, overly smooth, AI look; chaotic composition; blurry, distorted
  text". [SRC]
- **The community has little to offer:**
  - Hosted 2509 wrappers (Replicate) default to "blurry, low quality, distorted". [UNVERIFIED]
  - One blog (dreampixelforge) says negatives work for "broad failure modes, not scene content". [UNVERIFIED]
  - One grid test on Qwen-Image text-to-image (promptmaster) saw NO effect of the negative "child" at cfg 1-7.
    [UNVERIFIED; t2i, not edit]
  - I found no Reddit or Civitai A/B of edit negatives.
- **Verdict:** the mechanism is real and active in our graph. Whether a CONTENT negative ("white backdrop") moves
  Qwen-Edit is unproven. Artifact negatives probably help a little. Test it (§1.5).

### 1.2 Side effects to expect

1. **Weaker edits.** A text negative replaces the "no instruction" baseline. The guidance direction becomes
   (instruction - negative words) instead of (instruction - keep the picture). Both branches now ask for a change, so
   part of the "change it" push can cancel. Expect a weaker style conversion or a weaker view turn. [INF]
   - Remedy: ComfyUI's `PerpNegGuider` (comfy_extras/nodes_perpneg.py, used with SamplerCustomAdvanced).
     - It keeps the empty-instruction cond as the baseline and removes only the part of the negative that is
       perpendicular to the positive direction.
     - It runs the post-cfg hooks, so CFGNorm still applies.
     - Cost: 3 model passes per step, about +50% time (12 -> ~18 min per edit on the 3070).
     - It is marked experimental. [CODE]
2. **Washed-out colour or contrast.** Long negatives at cfg 3-4 can flatten colour (Lightricks and our Z-Image notes).
   CFGNorm caps the result's norm at the conditional's. That limits oversaturation but can also mute contrast.
   Measure the chroma and the L* spread of the figure against the input. [INF/OURS]
3. **Drift away from the input.** The negative is read as an INSTRUCTION about the input pictures. Words that
   describe something the input HAS and must keep push the result away from the input. [INF]
   - Never put the character's own colours or parts into the negative: "black" for masked's suit, "mask", "purple".
   - In a look image (§1.4c), never put the scene's own things in the negative ("grey floor", "wall") unless the edit
     must replace them.
4. **cfg 4 with a second image gave pixel errors here** (AGENTS.md). Keep cfg 3. Do not raise cfg to "make the
   negative stronger". [OURS]

### 1.3 Negations in the positive text ("NOT visible")

- **Official guidance:** Qwen's Chinese text-to-image rewriter removes every negation word (rule 8). The EDIT rewriter
  has no negation rule. It only resolves contradictions and names what to keep ("keeping picture 2's background
  unchanged"). [SRC: prompt_utils.py]
- **The encoder:** an LLM (Qwen2.5-VL 7B). It parses "not" far better than CLIP. But the embedding of "the mask is not
  visible" still contains "mask". [INF]
- **Our evidence is mixed:**
  - the cloaked3d back view needed "the mask and the face are NOT visible" + denoise 0.7 (no LoRA). That is
    confounded with the denoise change;
  - the reposed masked3d_rest back got a MASK on the back of its head when the text named the mask, and positive
    words fixed it ("the back of the head with its hair"). [OURS]
- **Rule:**
  1. Describe what IS seen ("the back of the head, covered with short dark plum hair; the back of the jacket").
  2. Put the forbidden things in the NEGATIVE (back views: "face, mask, eye holes"). That is exactly where a
     negative is meant to push away from the input.
  3. Keep a "NOT" sentence only where an A/B shows it helps.
  - Test: one back view, the same seed, "NOT visible" positive vs positive words + the back-view negative.

### 1.4 Proposed default negatives (comma lists of plain nouns; CFGNorm stays at 1.0)

**Base (every Qwen edit):**
```
extra arms, extra hands, extra fingers, missing fingers, fused fingers, duplicate character, second figure,
cropped feet, cropped head, text, watermark, logo, blurry, low quality
```
- Sources: Qwen-2512's official t2i negative (deformed limbs and fingers, blurry, low quality) and our own failures
  (a third arm, fingers, duplicate figures).

**(a) Sheet views and edits on black** = base +
```
white background, grey background, gradient background, studio backdrop, seamless paper backdrop, room, wall,
floor, ground shadow, cast shadow, floor reflection, vignette, spotlight, rim light, dramatic side lighting,
harsh shadows, strong highlights, {character exclude list}
```
- Back views only, add: `face, mask, eye holes` (character-specific words; only for the back).
- Sources: today's white backdrop + floor shadow; the rooms behind views (AGENTS.md); "strong side light"
  (cloaked3d); the textured brown backdrop (masked3d seed 8).

**(b) Style conversion** = (a) +
- the SOURCE style, which strengthens the edit instead of weakening it:
  - 2D -> 3D: `flat 2D illustration, line art, cel shading, dark outlines`;
  - 3D -> 2D: `3D render, CGI, glossy plastic, subsurface skin`;
- the drift seen in E7: `plastic toy, vinyl figure, product photo`;
- the character's known colour drift as concrete wrong colours (kitsune: `green cape, navy trousers`).
- Do NOT add the turntable wording of the earlier report ("like a neutral turntable reference render") to the
  positive for conversions: turntable renders sit on grey or white seamless. [INF]

**(c) Look images (Blender frame -> character)** = base +
```
white wooden mannequin, mannequin joints, grey clay figure, flat 2D sticker, cut-out outline, white outline around
the figure, extra person, crowd
```
- NO background words here: the background is the Blender place, which must stay.
- For `look_scene` (grey room -> place) add: `plain grey room, grey floor, empty grey studio`.
- Sources: the duo look's flat 2D kitsune "sticker", mannequin parts left in look images, the floating mask.

The exact numbers of terms are a judgement call (20-35 terms). Shorter is safer under side effect 2.

### 1.5 A/B for the negative (GPU; queue it like any job)

- **Replay E7a_T** (kitsune front -> masked3d style, seed 202, the same instruction), 4 arms:
  1. empty negative (as before);
  2. negative (b);
  3. negative (b) through PerpNegGuider, neg_scale 1;
  4. empty negative + the latent noise mask of §2.
- **Then repeat** with a second seed.
- **Measure:**
  - the border mean and the share of non-black pixels outside the dilated input figure;
  - floor-shadow pixels below the feet line;
  - the silhouette IoU against the input;
  - the figure's chroma and the L* spread against the input;
  - is the white mask present (a white blob in the head box);
  - a blind critic on "3D-ness" and design fidelity (did the negative weaken the conversion?).
- **Cost:** about 12-18 min per arm.

## 2. Output control beyond prompts

### 2.1 Ranked: what most reliably gives the same pose, a chosen background, flat light, no floor shadow and no invented details

**1. Composite the result over the wanted background through a LEARNED matte. This is the guarantee.**
- **What:** after every Qwen edit, view, repose or style job, run a background-removal model on the OUTPUT, intersect
  its alpha with a generously dilated figure region, and composite onto pure black (or the panel colour).
- **What it fixes:** the backdrop, the room, the floor shadow and the vignette go away, whatever Qwen painted.
- **It replaces the colour flood fills** that ate today's white mask and the black suit's edges
  (`black_background`, `charsheet.assemble_panels`' `~background_mask(it, 2)`, the run's `refsheet.prepare`
  tolerance 14). [MEAS: on masked's sheet the run step blacks another 2.2% of the figure pixels, mean RGB 9/7/9 = the
  suit's edge]
- **BiRefNet is native in our ComfyUI** [CODE]:
  - core nodes `LoadBackgroundRemovalModel` + `RemoveBackground` (comfy_extras/nodes_bg_removal.py,
    comfy/bg_removal_model.py; a 1024 px input; returns a soft mask; added 2026-05);
  - weights `Comfy-Org/BiRefNet background_removal/birefnet.safetensors` (444 MB, MIT; docs.comfy.org says "available
    under the MIT License") into `models/background_removal/` (empty today);
  - no custom pack needed.
- **Licences of the candidates:**

| model | licence | notes |
|---|---|---|
| BiRefNet (ZhengPeng7) | MIT [SRC] | general model trained on DIS-TR. The DIS5K README asks commercial users to sign separate dataset terms [UNVERIFIED wording; ask before shipping]. Native in ComfyUI. |
| Lucida (egeorcun; also in Comfy-Org/BiRefNet as lucida.safetensors, 885 MB) | MIT weights [SRC] | BiRefNet_HR fine-tune for illustrations, glows and transparent parts; uses ToonOut (CC-BY) AND research-only sets (P3M-10k, COD10K, DIS5K), as its card warns. Whether the native loader reads it is not checked (both are BiRefNet). |
| ToonOut (joelseytre/toonout) | MIT weights, data CC-BY 4.0 [SRC] | BiRefNet fine-tuned on 1,228 anime images (95.3% -> 99.5% pixel accuracy). Best fit for our 2D characters. The format for the native loader is not checked (.pth -> safetensors?). |
| BEN2 base (PramaLLC) | MIT [SRC] | own architecture, not loadable by the native node. The better "full" model is API or commercial only. |
| InSPyReNet / transparent-background | repo MIT; the weights' licence is not stated [SRC] | a pip package, CPU possible. |
| RMBG-2.0 (BRIA) | CC BY-NC 4.0, commercial use only by agreement [SRC] | do not use. RMBG-1.4: also non-commercial. |
| SAM 2.1 | Apache-2.0, installed [OURS] | needs a box or points; binary mask, coarser edges. Good as a second opinion (the box from the input figure) and for videos. |
| MatAnyone | S-Lab non-commercial (AGENTS.md) | do not use. |

- **Order of the composite:**
  1. Find the input's figure (BiRefNet on the input; the input sits on black).
  2. Dilate it to a "may change" region: about 4-6% of the figure height for edits and style; the figure's column
     band +-15% for views.
  3. After the edit: alpha = BiRefNet(output) x region.
  4. Composite onto the chosen colour.
  5. Keep the alpha as `<view>_alpha.png`.
- **Caveat:** a white mask on a white backdrop is also hard for BiRefNet (low contrast), although it is semantic, not
  colour-based [UNVERIFIED]. That is why #2 matters: it keeps the backdrop from being painted at all.

**2. Latent noise mask: only the figure region can change.**
- **What:** add `SetLatentNoiseMask` between `VAEEncode` and `KSampler` (core node).
- **How it works:** KSamplerX0Inpaint replaces the masked-out latent with the re-noised original at every step and
  returns the original latent there at the end (`out * mask + latent_image * (1 - mask)`). So the background outside
  the mask stays the input's black: the VAE round trip of black is still black. [CODE]
- **The model still sees the whole picture** (the reference latents and the VL branch are unchanged).
- **Mask size:** latents are 8 px (packed 2x2 = 16 px). Dilate by >= 24 px and feather a little. #1 then cleans the
  band.
- **Per use:**
  - (a) edits and style: the dilated input figure;
  - views: a column band (a side view changes width);
  - (c) look images: the Blender mannequin mask from the passes (exact), grown generously for capes and hair, as in
    the duo crop boxes (+20% width, +8% height). Then the camera, the place and the light outside it CANNOT change.
    That is today's "Keep the camera, the place ... exactly the same" made physical.
  - Earlier lesson: in VIDEO the masks cut the cape. For a still look image the dilation can be large, and #1 is not
    used for look images (the scene stays). Check the cape length against the sheet.
- **Cost:** none (no extra pass).

**3. Generate and verify (seeds + automatic gates), then a blind look.**
- Make 2-3 seeds per important edit (quality over speed is the user's rule).
- Reject automatically:
  - border or background non-uniformity outside the matte;
  - floor-shadow pixels under the feet;
  - silhouette IoU < ~0.9 against the input (edits and style);
  - per-part colour shift in Lab (hair, cape, suit, mask: a cluster of the input's colours, matched by place)
    over a limit;
  - a wrong facing side for views (already planned in the earlier report).
- Fix small colour drift in code (`touch_up`), as for cloaked3d. Flatten light with `delight.py` (prompts did not
  remove Qwen's side light).

**Then, in order of expected gain:**

4. **The negative of §1.4**, after the A/B. Use PerpNegGuider if edits get weaker.
5. **Prompt structure.**
   - Put the background as its OWN early sentence, not the last item of a list: "The background stays solid pure
     black everywhere, exactly as in Picture 1."
   - Keep "studio", "turntable", "product", "Pixar" and "toy" out of the positive.
   - Name the inputs "Picture 1/2". `runner.LOOK_INSTRUCTION` still says "image 2" (runner.py:2202). The official
     template's example says "image 1/2" too, so this is consistency, not a known bug.
   - For a style reference, say "only the rendering style of Picture 2; every colour stays as in Picture 1".
6. **A one-image graph** for edits without a reference (the earlier report: our graph passes the same picture twice).
7. **Denoise 0.85-0.9 for style conversions** keeps more of the input latent (its layout and black) than 1.0.
   - Our denoise notes are for local fixes (0.6 changed nothing, 0.8 moved a hand). A conversion needs a test:
     0.85 / 0.9 / 1.0 on seed 202.
8. **A mid-grey working background for DARK characters, only with #1 and #2.**
   - Black gloves on black came out as "thin dark claws". A grey around the figure lets the model see the silhouette.
   - Without #1/#2, grey invites a studio floor and shadow.
   - Keep black for light characters.

**Not recommended:**
- trusting prompts alone for the background (today);
- colour-based keying of a model output;
- non-commercial mattes (RMBG-2.0, MatAnyone).

Qwen-Image-Layered (RGBA layer decomposition) exists, but I did not check its licence or fit. [UNVERIFIED]

### 2.2 Light

- No mask controls light inside the figure. The levers are:
  - the negative (`rim light, dramatic side lighting, harsh shadows, strong highlights`);
  - the dx8152 relight sentence (the earlier report, not tested on 2511);
  - `delight.py` / `match.py` in code (proven on cloaked3d).
- A lower denoise keeps the input's flat light better. [OURS/INF]

## 3. Sheet background colour for video models

### 3.1 What the makers say (new since sheet_background_research.md)

- **Ingredients 2.5 card** (re-read today):
  - "laid out on a black background with no text";
  - "one clean panel per distinct visual element";
  - "Author the reference sheet with the element-driven reference-sheet generator";
  - trained on "a proprietary dataset of video clips paired with generated reference sheets", at 768x448 with the
    reference at the output size.
  - No panel colour is given. [SRC]
  - So the training sheets came from Lightricks' own generator, which is not public. Its one public output (the pack's
    `ingredients_input.jpg`) has a black canvas and gutters, with LIGHT-GREY panels (RGB 173-196, L* 71-79) for
    characters and props. In distribution = black canvas + light-grey panels, as far as one example shows. [INF]
- **ComfyUI-LTXVideo 2.5 example workflow notes:** "laid out side by side"; refer to elements "by their position".
  Nothing on colour. [CODE]
- **Other reference-to-video systems** (from the earlier report):
  - VACE: references segmented "to keep the background white";
  - SkyReels-A2: subjects cut out "with white grounding ... to avoid additional noise";
  - MAGREF: SAM2 cut-outs on a blank canvas (the colour is not stated).
  - Re-checked today: the Phantom paper and README say NOTHING about reference backgrounds. Its fixes for
    copy-paste are cross-video pairs.
  - No paper ablates the reference background COLOUR. Kling Elements and Runway give only "clean / neutral" advice.
    [SRC/UNVERIFIED]
- **The consensus** is a uniform, segmented background, light or white in the systems that state one. Nobody
  documents black PANELS behind a character. [INF]

### 3.2 Our measurement (CPU, today)

Method:
- Outline contrast of the CURRENT sheets at the Ingredients input size (768x448).
- The pixels on a ring 2-4 px inside the visible figure outline.
- The share of them within 10 L* of a hypothetical panel lightness. Lower is better.

| sheet | black L0 | L20 | L35 | L50 | L60 | L70 | L77 (Lightricks) | L85 | white L95 |
|---|---|---|---|---|---|---|---|---|---|
| masked (black suit) | **25%** | 45% | 16% | 9% | 9% | 7% | **5%** | 4% | 3% |
| kitsune-3d (light costume, white mask) | 6% | 38% | 30% | 19% | **6%** | 11% | 12% | 10% | **14%** |
| kitsune (2D) | 4% | 42% | 27% | 21% | **3%** | 7% | 10% | 11% | **18%** |

[MEAS; script in the session scratchpad]

How to read the table:
- The masked number on black UNDER-states the problem. The figure was cut by the flood fill, so the darkest suit
  pixels that touched the background are already gone from the "figure".
- On the outermost ring, 62% of masked's outline pixels are within 10 L* of black.
- **Black is the worst panel for a dark costume.** A quarter or more of the outline disappears at 768x448.
- **White is the worst panel for a white mask** (kitsune: 14-18%).
- **Lightness 60-77 is best for both kinds.** masked is best at 77 or above. kitsune is best around 60.
- A common panel of L* about 65-70 (RGB about 160-170), Lightricks-like, is the compromise.
- These are proxies: the real test is the video (§3.4).

### 3.3 Trade-offs

**Black panels**
- For: the canvas matches the card's wording; no colour cast; no "studio" feel to leak.
- Against: dark costumes vanish (gloves, suit legs, the black hood); the matting bug eats edges (2.2% more at run
  time).

**Light grey (L* 70-77)**
- For: the in-distribution guess (Lightricks' example); strong contrast for dark parts; a fair contrast for white
  parts (masks, bandages: ~15-20 L*).
- Against: it could leak a grey studio haze into the video [UNVERIFIED]. In layout runs the layout guide and the
  frame -1 look image set the scene, so leakage is less likely. [OURS/INF]

**Mid grey (L* ~50-55)**
- For: equal contrast (~40 L*) for black and white parts.
- Against: weak for mid-tone costumes (teal, grey trousers: 13-21% for kitsune).

**White**
- For: VACE / SkyReels-style grounding.
- Against: white masks and bandages vanish; it invites "white studio" (today's Qwen failure is the image-model
  version of this).

**Practical points**
- **Panels need a real alpha** (§2.1 #1). The views are made on black, so any panel colour is a composite at sheet
  time: no new Qwen job.
- **Keep the black canvas and gutters** (the card's wording; the panels stay clearly separated).
- **Never a strong hue.**
- **The face panel** (a crop with a hard bottom edge on black) has the same issue. It also shows a black blob eaten
  inside the suit in masked's current sheet (visible today).

### 3.4 A/B plan (preview, one seed, then a second seed)

**Stage 0: sheets (CPU, after #1 is installed)**
- Make ONE alpha per view (BiRefNet, checked by eye).
- Build masked3d_rest sheets that differ ONLY in the panel fill:
  - A = black (0);
  - B = mid grey (128, L* ~54);
  - C = light grey (~185, L* ~75, Lightricks-like);
  - D = white (255), optional and lowest priority: the table predicts it loses on white parts.
- Same layout, same scale, same alpha, black canvas and gutters in all.
- Images-only metrics per arm:
  - inner-ring ΔL* per part (gloves, suit legs, mask, bandages), the minimum reported;
  - the same after a 768x448 downscale.

**Stage 1: video**
- **Setup:** three (or four) layout runs, identical except the sheet.
  - Copies of masked3d_rest whose `sheet.png` is the arm's sheet (rewrite the copy's paths on the parsed record, as
    in AGENTS.md).
  - orbit_wave_360 through clip 2 (front -> side -> back), `--quality preview`, one seed, the same look text.
  - `--look-keyframes off`: sheet keyframes paste sheet panels, and a grey panel would paste a grey box.
  - The Qwen look image uses the views, not the sheet, so it should be identical across arms. Check its hash. If it
    differs, give arm A's look to the others (`run retake --look-image`).
  - The run's `refsheet.prepare` re-blacks only border-connected near-black: grey panels pass unchanged, and the
    black arm stays "production black".
- **Judge:**
  - a blind critic (docs/gauntlet_critic.md) on CHARACTER: suit silhouette, glove shape (no "claws"), arm and leg
    width, the mask, the white bandages, side and back views;
  - BACKGROUND: grey or white haze, void patches, a panel colour in the scene;
  - numbers: `run score` (motion bone, mirrored, overlap), `setup/brightness_check.py` (does a light sheet raise the
    scene's L*?), and the suit drift metric (mean Lab b* of the dark pixels in the Blender mask, as in masked3d_e2e).
- **Decision:** a grey arm wins only if the critic ranks it above black on CHARACTER without a BACKGROUND loss, on
  both seeds.
- **Round 2:** the winner against black on kitsune_rest (a light costume), to choose one default or a per-character
  rule (the panel lightness that maximises the weakest outline contrast).
- **Cost:** per arm, the look image (~12 min Qwen) + 2 preview clips. Queue them; one GPU job at a time.

## 4. Summary of recommendations

1. **Install native BiRefNet** (core ComfyUI, MIT, 444 MB) and composite every Qwen output through its matte onto the
   chosen background. Replace the colour flood fills (black_background, assemble_panels, refsheet.prepare) with that
   alpha.
2. **Add `SetLatentNoiseMask`** (a dilated figure region; for look images the Blender mannequin mask) to the Qwen edit,
   view and style graphs, so the background cannot be repainted.
3. **Generate 2-3 seeds and gate them automatically** (background, floor shadow, silhouette IoU, colour drift, facing).
4. **A/B the negatives of §1.4** (empty vs list vs PerpNeg) on the E7a_T replay before making them defaults.
   Background and light words go to the negative for sheet work, never for look images.
5. **Sheet panels:** black is not verified as best, and our measurement says it is the worst for the black suit.
   Keep the black canvas and gutters, and A/B black vs mid grey vs light grey (vs white) panels per §3.4.

## Not verified

- Whether content negatives (backdrop, shadow) move Qwen-Edit-2511 at all. There is no published A/B. One t2i test
  showed none.
- Whether PerpNegGuider behaves well with Qwen's reference latents. The code path supports it; it was not run.
- Whether BiRefNet keeps a white mask against a white backdrop, and whether it drops soft contact shadows.
- The DIS5K commercial terms (the README asks commercial users to sign separate terms). The ToonOut and Lucida
  checkpoint formats for the native loader.
- Whether a grey or white sheet panel leaks into LTX layout videos. No source covers it.
- The outline-contrast table is a proxy on mattes that are already biased (the flood-fill cut).

## Sources

- ComfyUI (local, v0.42.1 63be2f9):
  - comfy_extras/nodes_qwen.py (TextEncodeQwenImageEditPlus, TextEncodeQwenImage21);
  - comfy/samplers.py (sampling_function cfg==1 skip, KSamplerX0Inpaint);
  - comfy_extras/nodes_cfg.py (CFGNorm); comfy_extras/nodes_perpneg.py (PerpNegGuider);
  - comfy_extras/nodes_bg_removal.py, comfy/bg_removal_model.py (native BiRefNet); nodes.py SetLatentNoiseMask.
- Official ComfyUI 2511 template: https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_qwen_image_edit_2511.json
- Diffusers Qwen edit plus pipeline: https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py
- DiffSynth-Studio:
  - https://raw.githubusercontent.com/modelscope/DiffSynth-Studio/main/examples/qwen_image/model_inference/Qwen-Image-Edit-2511.py
  - https://raw.githubusercontent.com/modelscope/DiffSynth-Studio/main/diffsynth/pipelines/qwen_image.py
- Qwen:
  - https://huggingface.co/Qwen/Qwen-Image-Edit-2511
  - https://github.com/QwenLM/Qwen-Image
  - https://raw.githubusercontent.com/QwenLM/Qwen-Image/main/src/examples/tools/prompt_utils.py
- Community [UNVERIFIED]:
  - https://blog.promptmaster.pro/posts/qwen-image-negative-prompts/
  - https://www.dreampixelforge.com/blog/qwen-image-prompts
  - https://replicate.com/jackieyaung/qwen-image-edit-2509-gguf/api
- Background removal:
  - https://docs.comfy.org/tutorials/utility/remove-background-birefnet
  - https://huggingface.co/Comfy-Org/BiRefNet
  - https://huggingface.co/ZhengPeng7/BiRefNet
  - https://huggingface.co/egeorcun/lucida
  - https://huggingface.co/joelseytre/toonout ; https://arxiv.org/abs/2509.06839
  - https://huggingface.co/PramaLLC/BEN2
  - https://huggingface.co/briaai/RMBG-2.0
  - https://github.com/plemeri/transparent-background
  - https://github.com/xuebinqin/DIS (DIS5K terms)
- LTX:
  - https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients
  - local ComfyUI-LTXVideo example_workflows/2.5/README.md and LTX-2.5_ICLoRA_Ingredients_Single_Stage_Distilled.json
- Reference-to-video (see docs/sheet_background_research.md for VACE, SkyReels-A2, MAGREF, OpenS2V, Kling):
  - https://github.com/Phantom-video/Phantom ; https://arxiv.org/html/2502.11079
- Ours:
  - .cinestudio/gauntlet/style/RESULTS.md (E7), jobs2.json (the E7a_T instruction);
  - cinestudio/character.py (GRAPH_DEFAULTS, black_background);
  - cinestudio/refsheet.py, cinestudio/charsheet.py, cinestudio/runner.py (LOOK_INSTRUCTION, step_sheet);
  - workflows/qwenedit_*_api.json.
