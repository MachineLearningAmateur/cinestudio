# The look image keeps its background (research, 2026-10-26)

The question: the look image (Qwen-Image-Edit-2511 turns the clay mannequin in layout frame 1 into the character)
redraws the WHOLE frame and sometimes drops background objects (final run r20261026-035007-8d6b, seed 303: the right
banner "Open Source FTW", both EXIT signs and the hoop were lost; the video kept the losses). How do we keep every
background pixel without a pixel crop-and-paste, and still put the right character in?

Tags: [OFFICIAL] model card or vendor docs; [CODE] read in our code, the installed ComfyUI 0.42.1 or the installed
ComfyUI-LTXVideo pack (commit 220e1d4, 2026-10-20), or diffusers; [MEASURED] measured for this report on existing run
files (CPU only, no GPU job); [COMMUNITY] blogs, workflow pages; [RESEARCH] papers; [INFERENCE] my reasoning, untested.

## Short answer

1. **Fix the geometry first (cheap, core nodes, no new model).** Today the look graph centre-crops the 768x448 frame to
   1328x800 (FluxKontextImageScale), while the text encoder rescales the same picture to 1320x792 for the reference
   latent. The output is drawn at the reference's grid: every past look is 0.6% narrower and 1.0% shorter than the
   layout [MEASURED: ECC scale 0.9939 x 0.9902 on 20 of 22 looks = 1320/1328 x 792/800]. LTX then centre-crops the
   look again. Send the frame at 1344x784 (exactly 1.75x the frame, a size the encoder keeps: background.edit_layout)
   without the crop node, as the set redraws already do. A community test found that matching grids cut changed pixels
   outside an edit from 19% to 0.8% [COMMUNITY].
2. **Pin the background in latent space during the Qwen sampling: SetLatentNoiseMask (+ DifferentialDiffusion for a
   soft edge).** Core nodes. At every step the sampler puts the input's own latent back outside the figure mask, so
   only the figure area is drawn. The model still sees the whole frame on every step, so light and contact shadow are
   drawn in context. Nothing is pasted in pixels. The background comes out as the VAE round trip of the input
   (about 0.01-0.02% of pixels change) [CODE][COMMUNITY]. This is the same per-step blend that diffusers'
   QwenImageEditInpaintPipeline uses [CODE].
3. **Gate every look automatically (CPU, seconds).** Align the look to the layout frame (ECC), then measure outside the
   dilated figure mask: SSIM, edge recall and "lost tiles". Our good looks score SSIM 0.97-0.99 and edge recall
   0.93-0.98; the bad final look scored 0.83 and 0.68 with 24 lost tiles and a 14.6 px shift [MEASURED]. On a fail,
   make the look again with the next seed. This works with or without option 2.

A fallback on the video side, with no change to Qwen: **mask the look guide in LTX** (LTXAddVideoICLoRAGuideAdvanced,
`attention_mask` = the figure). The background tokens of the look are then ignored and the background comes from the
layout render. It is in the installed pack, but untested, it turns SageAttention off for the masked attention, and the
Layout-To-Render card says the look image "sets palette, lighting, and materials" [OFFICIAL], so the background may
lose its look.

Side finding [MEASURED, by eye]: the looks do not keep the layout's POSE. In every look I opened (gym seed 202/303,
forest seed 202), the mannequin's hand-on-hip pose became the front view's straight arms-down pose. Option 2 limits
where the figure can be drawn, so it may help the pose too [INFERENCE]. Add a pose check to the A/B.

## What the look step does today [CODE]

- runner.Run.step_look -> _qwen_look: ProfileWorkflow("qwenedit", "edit") = workflows/qwenedit_edit_api.json.
  Picture 1 = passes rgb_0001.png (768x448: the plate, the mannequin and its proxy kit). Picture 2 = the run's frozen
  front view. Text = LOOK_INSTRUCTION. Quality values: 40 steps, cfg 3, euler/simple, shift 3.1, denoise 1.0,
  CFGOverride cfg 1 from 65%. The seed = the run's seed. The same graph makes the keyframe looks (look_k1.png ...).
- The graph: `img` LoadImage -> `scale` FluxKontextImageScale (centre crop to one of the fixed Kontext sizes: 1328x800
  for a 768x448 frame) -> `enc` VAEEncode -> KSampler latent at denoise 1.0. So the start latent only gives the size;
  the content comes from the reference latents (`pos`/`neg` FluxKontextMultiReferenceLatentMethod,
  `index_timestep_zero`).
- TextEncodeQwenImageEditPlus (comfy_extras/nodes_qwen.py) scales every picture to 384x384 area for the vision
  tokens, and, when `vae` is connected, to ~1024x1024 area rounded to multiples of 8 for the reference latent. 1328x800
  becomes 1320x792. 1344x784 stays 1344x784 (background._stable).
- The look (1328x800) goes to `g_look` LTXAddVideoICLoRAGuide, frame_idx -1, crop "center": cut to 1.714:1 and
  scaled to the 768x448 latent. Net effect: the look shows the central ~96.9% x 96.9% of the layout, about 1.032x
  zoomed, plus Qwen's own 0.6%/1.0% shrink. At 768 px that is up to ~12 px of mismatch at the frame edges against
  the layout guide [CODE + arithmetic].
- LOOK_INSTRUCTION says "image 2" and "the sky". AGENTS 4.3 says name the inputs "Picture 1"/"Picture 2" (the
  encoder labels them so), and 7.3 found "sky" in a crop instruction drew a blue sky. Both are small prompt fixes.

## Measured: how much do past looks change the background? [MEASURED]

Method (script kept outside the repo; CPU): layout frame 1 put through the same Kontext crop to 1328x800; the figure
mask dilated by 30 px (61x61 kernel) and excluded; ECC affine fit of the look to the layout outside the mask; then on
the aligned look: SSIM (Gaussian 11 px), mean absolute difference (MAD, 0-255), Canny edge recall (layout edges found
within 3 px in the look), and "lost tiles" (64 px tiles with >= 60 layout edge pixels where < 40% are kept).

| run | seed | quality | ECC shift x,y px | SSIM out | MAD | edge recall | lost tiles |
|---|---|---|---|---|---|---|---|
| r20261026-035007-8d6b (the failure) | 303 | final | -14.6, 0.6 | 0.83 | 14.8 | 0.68 | 24 |
| r20261026-035011-a791 | 303 | final | -0.4, 0.0 | 0.977 | 2.0 | 0.969 | 0 |
| r20261026-022958-2a09 (+4 more previews) | 303 | preview | -0.4, 0.0 | 0.972-0.978 | 2.0-2.4 | 0.964-0.976 | 0 |
| r20261025-233013-0242 | 404 | preview | -0.3, -0.1 | 0.972 | 2.3 | 0.967 | 0 |
| r20261025-130232-5475, -190638-f318, -2057/2101 | 202 | both | -0.5, 0.0 | 0.978 | 1.6-1.7 | 0.931-0.976 | 0-3 |
| r20261023-023440-3932 (forest panorama) | 202 | preview | -4.1, -0.7 | 0.881 | 5.0 | 0.756 | 7 |
| r20261022-* (older backgrounds) | 202 | preview | -0.3, 0.1 | 0.988-0.990 | 1.4-1.5 | 0.84 | 1-4 |

- Before alignment, even the good gym looks scored SSIM 0.81 and edge recall 0.64: the court lines are a few px off
  because of the 0.6%/1.0% shrink. A gate must align first, or the shrink alone fails it.
- The failure is clear on every number. The forest look (no lost object, but the foliage texture redrawn) sits between.
  So the thresholds below are per kind of background, and lost tiles plus the shift are the most direct signals.
- Every look shares the same scale (0.9939 x 0.9902, or 1.0003 x 0.9908 for one other bucket): this is the
  grid mismatch, not chance.

## Option 1: fix the geometry (do this in every case)

How: the same change background._no_kontext_crop makes for the set redraws. Resize the frame to edit_layout(768, 448)
= 1344x784 in code (exactly 1.75x, the same 1.714:1 shape), load it, and connect `img` straight to `pos_txt.image1`,
`neg_txt.image1` and `enc.pixels`; remove `scale`. The encoder keeps 1344x784, so the reference latent and the
sampling latent have the same grid. The look comes out at 1344x784, the frame's exact shape, and LTX's centre crop
then cuts nothing.

A stricter form [COMMUNITY]: leave `vae` unconnected on the text encoders and add the reference latents with core
`ReferenceLatent` from the SAME VAEEncode the KSampler uses (one for Picture 1, one for Picture 2). Sizes in multiples
of 16 (8 px latent grid x 2x2 patches). The tester measured whole-image drift 8 px -> ~1 px, and changed pixels
outside an expression edit 19.0% -> 0.8%; a "keep everything else" clause changed almost nothing (0.8% vs 0.7%) once
the grids matched (lilting.ch). Our 1344x784 route gives the same grid match with fewer changes; ReferenceLatent is
the fallback if a shift remains.

Evidence: [MEASURED] the 0.6%/1.0% shrink above; [CODE] nodes_qwen.py rescale; AGENTS 6 geometry lessons (set redraw
edge fit 0.26 -> 0.52 after the same fix). [COMMUNITY] lilting.ch numbers.
Risks: none known for the background. Qwen still redraws the whole frame, so it does not stop a dropped banner by
itself (the 14.6 px shift and the lost objects came on top of the shrink).
Cost: none (1344x784 = 1.05 MP, about the same tokens as 1328x800). The LTX graph needs no change.

## Option 2: latent noise mask inside the Qwen edit (recommended main fix)

How it works [CODE: comfy/samplers.py KSamplerX0Inpaint]: with a `noise_mask` on the latent, before each model call
the sampler sets `x = x*mask + noised(input latent)*(1-mask)`, and after it `out = out*mask + input latent*(1-mask)`.
So outside the mask the result is exactly the input latent. The model attends to the whole frame at every step, so
the new figure gets light, colour and contact shadow from the real surroundings. This is "blended latent diffusion"
[RESEARCH: Avrahami et al. 2023], and diffusers' official QwenImageEditInpaintPipeline does the same per step
(`latents = (1 - init_mask) * init_latents_proper + init_mask * latents`) [CODE].

Is there an official Qwen inpaint mode? Qwen's 2511 pages mention no mask or inpaint input [OFFICIAL]. The model has
no mask channel: ComfyUI's QwenImage `extra_conds` reads only `ref_latents` (no concat latent), so
InpaintModelConditioning adds nothing for Qwen except the same noise mask [CODE]. Use SetLatentNoiseMask.

Nodes (all core) added to qwenedit_edit (or a new `qwenedit_look` graph, so character edits stay as they are):
- `mask_load` LoadImageMask (or LoadImage + ImageToMask) = the run's figure mask, made in code (below), 1344x784.
- `noise_mask` SetLatentNoiseMask(samples = `enc`, mask = `mask_load`) -> `ks.latent_image`.
- `diffdiff` DifferentialDiffusion(model = `cfg_tail`, strength 1.0) -> `ks.model` (only for the soft edge).
- Keep denoise 1.0 (inside the mask the figure is drawn from noise; the mannequin is still visible to the model
  through Picture 1, and the instruction says to replace it).
- Bind `look_mask` in the .bind.json; add the mask file to the run's frozen inputs.

The mask (in code, from passes mask_0001.png, scaled to 1344x784):
- Core = the mannequin + proxy kit silhouette, dilated so the character's costume fits: ~3% of the frame height
  (~24 px at 784) by default, more for wide costumes (cape, long hair: read the proxy kit's cape/hair lengths).
  Value 1.0. Every mannequin pixel MUST be inside the 1.0 core, or white clay edges stay (they would be the halo).
- Soft ring outside the core, ~24-32 px, falling 1.0 -> 0.3 -> 0. With DifferentialDiffusion a pixel of value v is
  held to the input until the last ~v of the schedule [CODE: nodes_differential_diffusion.py threshold], so the ring
  can change tone and light late but not structure.
- A contact-shadow ellipse under the feet (the feet's x span x ~1.4, ~6% of the frame height below them) at 0.3-0.4,
  for the same reason. Pure SetLatentNoiseMask treats the mask as 0/1 per latent pixel; the soft values only mean
  something with DifferentialDiffusion.
- The latent grid is 8 px and the transformer works on 16 px patches: keep rings >= 16 px.

Expected result [INFERENCE from CODE + COMMUNITY]: outside the mask SSIM ~0.99 (the VAE round trip; lilting.ch measured
0.01-0.02% changed pixels and a ~1 px contour fringe for the round trip). No lost banners, signs or hoop, whatever the
seed. The character is drawn only where the layout puts the body, which should also hold the pose and size closer to
the layout [INFERENCE].

Risks:
- Seams: latent blending is not pixel-exact; the VAE decoder mixes neighbouring latents, so a hard latent mask can
  leave a faint halo or colour step at the edge [RESEARCH: arXiv 2512.05198 measured halo L1 0.141 for binary masks
  vs 0.050 soft, on the FLUX VAE]. The soft ring + DifferentialDiffusion is the cheap answer; the paper's learned fix
  (DecFormer) is trained per VAE and not available for Qwen.
- A costume wider than the mask is cut at the mask edge (a cape tip, hair). Use the proxy kit to size the dilation and
  check the look (the gate's "figure touches the mask edge" test below).
- Qwen may draw a smaller character to fit, or leave part of the region as background: fine, Qwen inpaints the
  background behind (it sees the whole room), but check the size.
- The mannequin's own floor shadow outside the ring stays as it was rendered (it has the same pose, so it should
  fit) [INFERENCE].
- Not for `look_scene` (that edit must change the background) and not for character edits.

Cost: no new model, no new download. Same 40 steps; the noise mask and DifferentialDiffusion add almost nothing.
About the same ~7 min per look on the 3070 as today (look event to look_k1 event: 420 s in r20261026-035007-8d6b).

Also apply to the keyframe looks (look_k*.png come from the same graph) and, for several characters, as a
replacement for the paste: one whole-frame edit per figure in turn, each with its own figure's mask (the other figures
and the room pinned). Trade-off: the crop gave each figure ~1024 px; whole-frame figures are ~300-350 px tall at 784.
LTX sees the look at 768x448 anyway, and face detail comes from the Ingredients sheet, so this is probably enough
[INFERENCE]. If the crop is kept for small figures, pin the crop's background with the same latent mask: then the
crop box pasted back differs from the frame only by the VAE round trip and the resampling, so a feathered box edge
carries no halo of a redrawn background [INFERENCE].

## Option 3: an automatic gate (with any option)

Where: right after each Qwen look (and keyframe look), before the clips. CPU only (OpenCV), well under a second.

Steps:
1. Load layout frame 1 and the look at the same size (after option 1: both 1344x784 / 768x448 shape, no crop).
2. Exclude the figure: the mask dilated by ~4% of the frame height (the look's character can be wider).
3. ECC affine fit of look -> layout outside the mask (background.align_to_set already does this on edge maps).
   Record shift and scale.
4. On the aligned look outside the mask: SSIM (Gaussian, sigma 1.5), MAD, edge recall within 3 px, and lost tiles
   (64 px tiles of the 1328-wide frame with >= 60 layout edge pixels and < 40% kept).
5. Pose and size (optional, recommended): the character's silhouette must lie inside the figure mask (pixels changed
   by > 25/255 outside the mask core and ring: < 1% of the ring) and DWPose keypoints of the look vs the rig's
   keypoints (passes keypoints.json): the project's dwpose_score tool.

Thresholds (from the 22 looks above; recheck after the A/B):
- Without option 2: fail when |shift| > 3 px or scale off 1.0 by > 1.5% (after option 1 it should be ~1.0), or
  lost tiles >= 4, or edge recall < 0.90, or aligned SSIM < 0.95. Textured scenes (foliage) need SSIM >= 0.85 and
  edge recall >= 0.75 instead: decide the class from the layout's edge density, or keep only lost tiles and shift as
  hard gates and the rest as warnings.
- With option 2 the background is pinned: expect SSIM >= 0.98 and no lost tile. Then the gate guards the mask edge
  (a gradient/colour step in a 4 px ring around the core: compare it with a ring 20 px further out; flag a ratio
  > 1.5) and the pose.
- On a fail: make the look again with seed+1 (record both), at most 3 tries; keep the best by lost tiles then SSIM;
  write the numbers into the run's look step (`look_check`) and show them in run details.

Cost: ~7 min of Qwen per extra try (only on a fail). The check itself is free.

## Option 4: prompt-only control (small lever)

- Qwen says 2511 "mitigates image drift" and improves character and group consistency; it gives no numbers and no
  advice on keeping unedited areas [OFFICIAL].
- With matched grids, "keep everything else" made almost no difference (0.8% vs 0.7%) [COMMUNITY]. Our own look
  failure came WITH "Keep the camera, the place, the floor, the sky and the light exactly the same."
- AGENTS 4.4 (negatives, background sentence first) is not A/B-tested.
- Worth doing anyway (free): "Picture 1"/"Picture 2" instead of "image 1/2"; no "sky" for indoor places; name the
  background's key objects in the keep sentence ("the two banners, the hoop, the EXIT signs and the doors"), using
  the template's Set words or the backplate's scene text. Not reliable on its own: the result is still chance per seed.

## Option 5: LTX-side alternatives

5a. **Masked look guide (attention mask)** [CODE]. The installed ComfyUI-LTXVideo has
`LTXAddVideoICLoRAGuideAdvanced` (same inputs as LTXAddVideoICLoRAGuide + `attention_strength` and an optional
pixel-space `attention_mask`). Core `LTXVAddGuide` also has `attention_mask` now. The model turns the mask into a
log-weight bias on attention to that guide's tokens (comfy/ldm/lightricks/model.py GuideAttentionMask: weight 0 =
-inf, the token is ignored). So `g_look`/`g_look2` with mask = the figure (1.0) and the background (0.0, or 0.2-0.4
to keep some palette) makes LTX take the character from the look and the background from the layout render.
- Pros: no extra Qwen time; works on today's looks; also usable for keyframe looks.
- Risks: the Layout-To-Render card says the look "sets palette, lighting, and materials" and trained the layout as a
  clay "blocky playblast" [OFFICIAL]; with the look's background masked out, LTX may render our photographic plate
  less faithfully than the look (untested). The guide grid is coarse: LTX's VAE is 32x spatially, so at 768x448 the
  look is 24x14 tokens of 32 px; the mask edge is a 32 px step (blurred by area averaging). Masked attention runs with
  `low_precision_attention=False`, i.e. without SageAttention, and builds a (guide tokens x all tokens) bias; the
  slowdown and the VRAM on 8 GB are unmeasured [CODE + INFERENCE]. Retakes and forks would need the mask too.
- A variant [INFERENCE, untested]: a second frame -1 guide with the clean layout frame 1 and the inverse mask, so each
  token position has one active look token (the figure from Qwen, the room from the render): a composite in
  attention, not in pixels.

5b. **No look, Ingredients alone.** Not advised: the card calls the first-frame look "the normal recipe" and warns
that a grey first frame gives a grey result [OFFICIAL]; our measured frame-0 match (12.8/255 to the look vs 19.7 to
the layout) shows how much the look steers the start.

5c. **Lower look strength overall** weakens the character as much as the background. Not a fix.

## Option 6: another model for the look (not advised now)

- InstantX Qwen-Image-ControlNet-Inpainting (Apache-2.0, needs ComfyUI >= 0.3.59) and DiffSynth
  Qwen-Image-Blockwise-ControlNet-Inpaint (ComfyUI core `QwenImageDiffsynthControlnet` with a mask; licence of the
  inpaint file not confirmed) are true mask inpainters, but for the Qwen-Image TEXT-to-image base: no reference
  picture, so the character's design would come from words only [OFFICIAL card / COMMUNITY]. Our core node adds the
  control only to the target tokens (`img[:, :encoded.shape[1]]`), so it would technically run on the Edit model
  with reference latents, but nobody reports that it works [CODE + INFERENCE]. Extra download ~1-3 GB.
- Z-Image Fun ControlNet Union (inpaint mode in core `ZImageControlPatch`): also no reference picture.
- FLUX.2 [klein] 4B (Apache-2.0; 9B is non-commercial) does multi-reference editing in ComfyUI [COMMUNITY]; no mask,
  so the same whole-frame drift question, a new model family and download. Check the licence on BFL's card first.
- Excluded by our rules: Qwen-Image 2.1 inpainting (qwen-research licence), FLUX Kontext dev, LanPaint (GPL-3.0),
  ComfyUI-Inpaint-CropAndStitch (GPL), Bria/Recraft nodes.

## Halo and seam risks, compared honestly

| method | where the seam is made | typical artefact | fixes |
|---|---|---|---|
| Pixel crop + paste (today's group look) | pixels, after generation | colour/light step, a ring of the crop's redrawn background, a shadow cut at the mask | feathered mask (done), colour transfer in a ring, Laplacian/Poisson blending (cv2.seamlessClone) |
| Latent noise mask (option 2) | latents, every step | faint halo/colour step from the nonlinear VAE decoder; ~1 px fringe | soft ring + DifferentialDiffusion; mask core outside the mannequin; ring >= 16 px |
| LTX attention mask (5a) | attention, at 32 px tokens | coarse boundary; background may drift from the look's palette | soft weights (0.2-0.4) for the background |

Notes: Poisson blending matches gradients, so it removes colour steps but drags a light/dark background colour into the
figure edge (it bleeds when the two sides differ strongly); Laplacian pyramid blending is softer [RESEARCH, standard
image editing]. Option 2 needs none of them because the figure is drawn while the model sees the real surroundings.

## A/B test plan (preview, GPU queue, after code changes are approved)

Shot: shan_shen_bu_balls with the run's gym plate (the failing setup), character `masked`. A second background with
texture (the forest panorama) for the gate thresholds. One fixed note per run.

Stage 1, looks only (Qwen, ~7 min each; no clips): seeds 202, 303, 404, 505, 606 x arms:
- A0 today's graph (control; reuse the existing looks for 202/303/404 where the inputs match).
- A1 option 1 (1344x784, no Kontext crop).
- A2 A1 + SetLatentNoiseMask, hard mask (core only).
- A3 A1 + soft mask + DifferentialDiffusion + contact-shadow ellipse.
- A4 A3 + the prompt fixes (Picture 1/2, no sky, named background objects).
Measure per look:
- Background: the gate numbers (aligned SSIM, MAD, edge recall, lost tiles, ECC shift/scale) outside the dilated mask;
  a named-object check by eye (two banners, hoop, both EXIT signs, doors).
- Character: head crop vs the front view (mask shape, eye holes, hair colour), jacket/suit colour (mean Lab in the
  garment regions vs the front view, delta E), finger count; a blind critic (docs/gauntlet_critic.md) on anonymous tags.
- Pose: DWPose of the look vs the rig keypoints (setup dwpose_score), silhouette inside the mask.
- Seams: 4 px ring gradient ratio and a colour step across the mask edge; zoomed crops at the shoulders and feet.
- Time per look.

Stage 2, clip 1 only (preview, `--through-clip 1`), seeds 303 and 202, with the winner of stage 1 vs A0, plus 5a
(A0 look + attention-masked look guide, background weight 0 and 0.3): the same background numbers on video frames
0, 60 and 120 against the layout frames; `run score` (character, motion, picture); clip time and peak VRAM for 5a.

Pass: no lost object on any seed; aligned SSIM outside >= 0.98 for option 2 arms; character scores not lower than A0
(blind critic); no visible seam at 2x zoom; time within +10%. Write the rounds to
.cinestudio/gauntlet/look_bg/RESULTS.md and the lasting rules to AGENTS.md 7.1.

## Finding after the build: an exactly aligned look darkens the video [MEASURED, 2026-10-26]

Options 1-3 were built (cinestudio/look_pin.py) and the A/B above was not run (user: "no need for a/b"). The first
final with the pinned look (r20261026-053113-1f16) kept every background object in the look image (gate: SSIM 0.995,
edge recall 0.95) but the VIDEO came out ~30% darker than the unpinned final with the same seed and lost the banner
text and the EXIT signs. Diagnosis with fixed look files (preview, clip 1, seed 303, no keyframe looks; the look file
put into work/look.png before the start; .cinestudio/gauntlet/look_pin_ab/RESULTS.md):

| Look given to LTX | Mean grey of clip 1 |
|---|---|
| pinned look (background = the layout frame after a VAE round trip, pixel-aligned) | 27.9 |
| the same file again | 27.0 (LTX repeats itself within 1.6/255) |
| the old unpinned look (Qwen redrew the room ~0.7% smaller and shifted) | 42.7 |
| pinned look resized 1344x784 -> 1328x800 | 39.5 |
| pinned figure pasted on the old background | 39.8 |
| pinned look moved 6 px right, 4 px down (no resampling) | 39.9 |
| pinned look blurred 1.2 px (alignment kept, VAE texture removed) | 30.5 |

- Cause: the exact alignment of the look's background with the layout guide, not the size and not the VAE texture.
  LTX-2's keyframe_cond.py moves `frame_idx` on the time axis only, so the -1 look and the frame-0 layout share every
  (x, y) position; when their pixels also match, the model gets near-duplicate tokens at the same places. Untested
  explanations: the aligned look is read as layout (the Layout-To-Render card: "A grey clay frame used as the first
  frame produces a grey result"), or the duplicated background tokens are weighted twice.
- Fix (built): look_pin.offset moves the kept look 6/4 px at 1344x784 after the gate. The repaired final
  (r20261026-110407-838f) is about as bright as the old one (29.3 -> 36.3 vs 31.8 -> 36.9) and keeps every object.
- Novelty (docs/novelty_research.md, 2026-10-26): NOT FOUND in Lightricks' cards and code, the ComfyUI-LTXVideo and
  LTX-2 issues, papers or the web. Lightricks' recipe (a free redraw of the first frame "by any image model") implies a
  pixel-exact look is outside the training data.
- Honest outlook (user and agent, 2026-10-26): worth recording and worth a GitHub issue to Lightricks with these
  numbers; at most a short note or workshop paper, and only if more data holds. Weak points: one seed, one scene, one
  model and LoRA; an input almost nobody else makes; it may be specific to the Layout-To-Render LoRA. It becomes a
  general result only if other models that share guide and output positions (Wan VACE, FLUX Kontext) show it too.
  The related panel copy (a sheet panel appearing at its own place in the frame) is NOT worth a paper: copy-paste and
  the shared-position effect are already known for reference conditioning in general (OminiControl, UNO, Phantom,
  MAGREF).
- Next data, when there is GPU time after the contest: 10 seeds x aligned / moved looks (~1 h of previews); then a
  move sweep (0, 1, 2, 4, 8, 16 px and scale), partial alignment (background only, figure only), the dev model, and
  one other model family.

## Sources

- Qwen-Image-Edit-2511 card and docs [OFFICIAL]: https://huggingface.co/Qwen/Qwen-Image-Edit-2511 ,
  https://docs.qwenlm.ai/qwen-image-edit-2511/index.html
- LTX-2.5 Layout-To-Render IC-LoRA card [OFFICIAL]: https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render
- LTX-2.5 card (licence, image conditioning) [OFFICIAL]: https://huggingface.co/Lightricks/LTX-2.5
- InstantX Qwen-Image-ControlNet-Inpainting [OFFICIAL card]: https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting
- ComfyUI QwenImageDiffsynthControlnet docs: https://docs.comfy.org/built-in-nodes/QwenImageDiffsynthControlnet.md ;
  DiffSynth Qwen-Image docs: https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Qwen-Image.html
- diffusers QwenImageEditInpaintPipeline [CODE]: https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage ,
  src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_inpaint.py (per-step blend, line ~1031)
- Qwen-Edit grid mismatch and ReferenceLatent fix [COMMUNITY]:
  https://lilting.ch/en/articles/qwen-image-edit-pixel-perfect-referencelatent
- Qwen inpaint workflow with SetLatentNoiseMask [COMMUNITY]:
  https://www.runcomfy.com/comfyui-workflows/qwen-image-2-1-inpainting-in-comfyui-no-offset-edits
- Differential Diffusion [RESEARCH]: https://arxiv.org/abs/2306.00950
- Blended Latent Diffusion [RESEARCH]: https://arxiv.org/abs/2206.02779
- Latent compositing is not pixel compositing (PELC/DecFormer) [RESEARCH]: https://arxiv.org/abs/2512.05198
- FLUX.2 klein licences [COMMUNITY mirror]: https://huggingface.co/YuCollection/FLUX.2-klein-base-4B-Diffusers/blob/main/README.md
- LanPaint licence [COMMUNITY]: https://dev.co/ai/frameworks/lanpaint
- Local code [CODE]: cinestudio/runner.py (step_look, _qwen_look, LOOK_INSTRUCTION, _layout_clip_job),
  cinestudio/background.py (edit_layout, _no_kontext_crop, align_to_set, edge_fit), workflows/qwenedit_edit_api.json,
  workflows/ltx25_layouting_api.json; ComfyUI 0.42.1 comfy_extras/nodes_qwen.py, nodes_differential_diffusion.py,
  nodes_model_patch.py, nodes_lt.py (LTXVAddGuide attention_mask), comfy/samplers.py (KSamplerX0Inpaint),
  comfy/model_base.py (QwenImage extra_conds), comfy/ldm/lightricks/model.py (GuideAttentionMask);
  ComfyUI-LTXVideo iclora.py (LTXAddVideoICLoRAGuideAdvanced), iclora_attention.py.
- Measurements [MEASURED]: 22 look.png files in .cinestudio/runs (r20261022-* to r20261026-*) against their passes
  rgb_0001.png / mask_0001.png; script in the system temp folder (claude/look_research/bgcheck*.py), not in the repo.
