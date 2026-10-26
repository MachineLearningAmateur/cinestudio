# Readable text and logos in generated pictures (research, 2026-10-25)

The case: the gym backplate `gym_noir_comfyui` (1920x1120, stylized 3D anime noir, made by Qwen-Image-Edit-2511 from a
Blender set render). It needs a framed cloth banner with the official Comfy wordmark and a dark scoreboard that reads
"Open Source FTW" in the wordmark's style. The user's goal: "ideally i want the text comfy to feel like it is actually
part of the scene and same with open source FTW". So the text must look painted in the same look, lit by the room and
made of the sign's material (cloth or panel), and the letter shapes must stay the brand's.

Evidence labels: **[measured here]** = a CPU experiment in `.cinestudio/gauntlet/text_research/`; **[repo]** = measured
earlier in this project (AGENTS.md); **[claimed]** = a vendor or paper statement we did not check; **[inference]** =
reasoning, not tested. No GPU job was run for this report.

## 1. How Qwen-Image / Qwen-Image-Edit render text

- **Training:** the tech report ([arXiv 2508.02324](https://arxiv.org/html/2508.02324), sec. 3.4) builds text data three
  ways: pure rendering on clean backgrounds, text **composited onto paper and wooden boards in scenes**, and template
  edits (slides, UI). Its curriculum goes from simple text to paragraphs. Pictures with dense or very small text were
  **removed** ("difficult to render legibly"), and it notes that font coverage limits the synthesis. Training ran at
  256, then 640 and 1328 px, with aspect ratios from 1:3 to 3:1. [claimed]
- **Scores:** CVTG-2K English word accuracy 0.829 (GPT Image 1: 0.857); LongText-Bench EN 0.943. Text editing (sec.
  5.3.3): Qwen changed 'H' to 'Q' "while maintaining style consistency" and was the only model to make an enamel
  material on text. [claimed]
- **Model cards:** Qwen-Image-Edit "preserv[es] the original font, size, and style" when it adds, deletes or changes text
  ([card](https://huggingface.co/Qwen/Qwen-Image-Edit)). It suggests fixing errors in steps by drawing boxes on the
  image. 2509 adds "editing text fonts, colors, and materials", native depth/edge/keypoint conditions as input
  pictures, and "1 to 3 input images" ([card](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)). 2511 claims less
  drift in the unedited parts. All Apache-2.0. [claimed]
- **Official prompt rules** (the edit prompt rewriter,
  [prompt_utils.py](https://raw.githubusercontent.com/QwenLM/Qwen-Image/main/src/examples/tools/prompt_utils.py)):
  - put all text in English double quotes; keep its language and capital letters;
  - adding text and changing text are both written as replacements: `Replace "xx" to "yy"`, `Replace the mask /
    bounding box to "yy"`;
  - give position, colour and layout only when the user asks for them.
  - For text-to-image: quote the text, give its position and style, and add no other text. Sizes for 2512: 1328², 1664x928
    (16:9), 1584x1056 (3:2) ([GitHub](https://github.com/QwenLM/Qwen-Image)).
- **No official guidance for a font or logo reference picture.** "Write X in the font of Picture 2" is not documented,
  and we found no test of it. The documented ability is narrower: text that is already in Picture 1 can be replaced in
  its own font. Community reports: small text breaks up when you zoom in and varies between seeds
  ([wavespeed](https://wavespeed.ai/blog/posts/fix-qwen-image-2512-text/)). The InstantX card says small-font text is
  lost unless the prompt says "TEXT".
- **What we saw:**
  - [repo] A region edit with the text in quotes spelled "ComfyUI" correctly, but in a generic bold sans font.
  - [repo] The from-set edit drew the blank banner with an invented crest and **unreadable small text**
    (`gym_noir_set`). That is the known small-text weakness.
- **Why size matters** [inference]: the Qwen VAE compresses 8x and the transformer packs 2x2 latents into one token, so
  one token covers 16x16 px. On the 1920 px plate the banner is about 258x140 px (bg.json region), so a 40 px letter is
  about 2.5 tokens tall. The wordmark's ~0.35-unit notches cannot survive that, and a whole-frame edit will redraw them.
  Crop and enlarge first (crop-and-stitch, ~4x): the letters become ~160 px, about 10 tokens.

## 2. Glyph-first methods (exact shapes first, then the model adds the look)

- **SDEdit** ([Meng et al. 2021](https://arxiv.org/abs/2108.01073)): starting from the noised picture, a lower start noise
  keeps more of it. Faithfulness and realism trade against each other, roughly over t0 0.3-0.6. Community numbers are for
  colour and composition (0.2-0.4 keeps colours), not for letters. In this repo **[repo]**, with the Qwen-Edit
  crop-and-stitch at shift 3.1:
  - 0.85 does not remove a part;
  - 0.8 changes the pose of a hand;
  - 0.6 "changes nothing";
  - 0.5 turned a thumb painted in code into the drawn style (the "paint, then refine" rule, AGENTS 4.3).
  For letters, sweep **0.45 / 0.55 / 0.65**. Below 1.0 the edit keeps full CFG (character.edit_image).
- **Masks:** our crop-and-stitch already limits the change to a box with soft edges. Inside the box, `SetLatentNoiseMask`
  (a core node) could limit the change to the sign area, so the wall and frame stay as they are. Untested with Qwen-Edit.
- **ControlNets for Qwen-Image** (the text-to-image base, not the edit model). The licences are commercial-safe:
  - [InstantX Qwen-Image-ControlNet-Union](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union): Apache-2.0,
    canny / soft edge / depth / pose, strength 0.8-1.0, trained at 1328, 2B parameters. Loaded with core "Load ControlNet
    Model".
  - DiffSynth Blockwise ControlNet Canny / Depth / **Inpaint**: Apache-2.0 per the
    [Comfy-Org repack](https://huggingface.co/Comfy-Org/Qwen-Image-DiffSynth-ControlNets). The original ModelScope card
    needs a login, so check it before shipping. Core nodes `ModelPatchLoader` + `QwenImageDiffsynthControlnet` (image +
    optional mask) ([docs](https://docs.comfy.org/built-in-nodes/QwenImageDiffsynthControlnet)).
  - How it would work: inpaint mask = the sign, canny = the edges of the exact glyphs. Untested here. They need extra
    VRAM on 8 GB, they target the base Qwen-Image (our 2512 is GGUF, so check compatibility), and edges carry outline
    only, not the yellow fill.
- **Text-specific models:**
  - RepText ([2504.19724](https://arxiv.org/abs/2504.19724)) uses a glyph latent plus canny and regional masking. Its
    authors list "disharmony with the scene", weak small fonts and extra text as limits.
  - **Licence: RepText, Calligrapher, FLUX-Text and TextFlux are built on FLUX.1-dev, which is excluded.**
  - AnyText / AnyText2 and GlyphControl are SD1.5-era research code with a fixed standard font
    ([ControlText](https://arxiv.org/html/2502.10999v1)). They are not worth adding.
- **How others keep text exact:** they keep the glyph in the latent (start from it or replicate it, as RepText does),
  limit the change to the text region, and check the result with OCR. The same ideas map to our tools: an exact
  composite, then a low-denoise region refine, then a shape check.

## 3. "Open Source FTW" in the wordmark's style

The wordmark (comfy.org `logo.svg`, `#F2FF59`) is drawn, not a font. Rendered and un-sheared **[measured here]**, it is
built from rounded rectangular blocks on an **11-row grid**:
- 2 ascender rows, 7 x-height rows, 2 descender rows;
- stems 3 units wide, counters ~0.8 unit;
- a **17.0° slant** (measured);
- small ~0.35-unit steps where a stem meets a bar, which make the notches;
- letters that touch.

| Option | Result |
|---|---|
| (a) closest OFL fonts | 31 fonts rendered (all from [google/fonts `ofl/`](https://github.com/google/fonts/tree/main/ofl), SIL OFL 1.1: commercial use OK, rendered pictures carry no licence duty). Upright fonts sheared to 17°. Silhouette IoU of "Comfy" against the wordmark in brackets. None has the stepped-block notches. |
| | **[Tektur](https://fonts.google.com/specimen/Tektur) Black** (0.67): heavy, square counters, closest overall |
| | **[Exo 2](https://fonts.google.com/specimen/Exo+2) Black Italic** (0.64): heavy, but rounder |
| | **[Saira Stencil One](https://fonts.google.com/specimen/Saira+Stencil+One)** (0.60): stencil cuts look like notches |
| | **[Russo One](https://fonts.google.com/specimen/Russo+One)** (0.57): blocky with rounded corners |
| | **[Jersey 10](https://fonts.google.com/specimen/Jersey+10)** (0.53) and **[Pixelify Sans](https://fonts.google.com/specimen/Pixelify+Sans) Bold** (0.45): the pixel steps match the notches, but the strokes are too thin |
| (b) build the glyphs from the wordmark's own system | **Feasible, and the most faithful.** A code prototype (`tools/blockfont.py`: cells on the grid, 3-unit stems, outer corners rounded by morphological opening at r 0.55 u, fillets at r 0.18 u, the 0.35 u steps, 17° shear) rebuilds "Comfy" at IoU 0.66 against the real SVG and writes "Open Source FTW" with the same notches. It still has rough parts (the y, the o's tabs, the bridges): a designer should polish it in Inkscape with the SVG's own blocks as the parts kit (1-3 h). Other letters come from existing ones: n from m, u = n turned 180°, W = m turned 180°, c from C. |
| (c) wordmark as Picture 2 | Undocumented (section 1). Qwen copies a reference picture too literally (AGENTS 4.3) or falls back to a generic italic. The documented variant is to put "Comfy" in Picture 1 on the panel and `Replace "Comfy" to "Open Source FTW"`, which the card says keeps the font. Worth one cheap seed, but with only 5 letters of evidence the brand style will likely drift. |

**Recommendation:** use (b), polished and exported as SVG; Tektur Black sheared 17° is the fallback. Samples:
`.cinestudio/gauntlet/text_research/font_candidates_top.png` (the top 7), `font_candidates_all.png` (all 31),
`blockfont_prototype.png`.

## 4. Compositing so it is exact but not pasted on

- **Geometry:**
  - Take the sign's corners from the Blender set: project the object's corners with the template camera
    (`world_to_camera_view`), then apply the plate's `align_to_set` transform. Detecting them in the picture is the
    fallback.
  - Warp the glyph with a homography at 4x supersampling, then INTER_AREA down.
  - On cloth, add a small displacement along the fold normals: take the cloth's low-pass luminance gradient, ≤1-2 px.
- **Light and material** (work in linear light):
  - **Printed cloth:** colour = ink albedo × the room light's colour (the mean of the cloth or wall around it, normalised)
    × exposure, so the print sits below the light sources (exit signs). Multiply by the cloth's shading ratio:
    blur(L, 1 px) / blur(L, 12 px) restores folds and weave. Keep the cloth's specular highlights by adding them on top
    with a screen blend. The ink should be a little less saturated than the brand colour where the cloth is dark.
  - **Lit LED panel:** emissive colour, not multiplied by the room. Grade it 10-20% toward the plate's colour cast. Add a
    two-scale bloom (σ 4 and 14 px) and a faint spill on the frame; an optional dot-matrix mask needs a pitch of 3 px or
    more, or it aliases.
  - Clear the old content first: cv2.inpaint of the paper squares.
- **Optics:**
  - Match the blur to the plate's **measured** edge width. A blurred step has σ ≈ contrast / (max gradient·√(2π)): **σ ≈
    1.1 px** on this plate's scoreboard frame [measured here].
  - Match the grain: the high-pass std of a flat wall is **1.2/255** here [measured here]. Add the grain after the blur.
  - Also match the plate's colour cast and black level.
- **Demo [measured here]:** `composite_demo.png` shows the scoreboard before and after (`tools/composite_demo.py`: board
  found by its dark outline, squares inpainted, block-grid text, LED bloom, matched blur and grain). An emissive panel
  reads as part of the scene this way. A printed banner needs the model refine (section 6) for material and light
  response. Re-lighting the already pasted banner from a colour mask smeared it, so use the exact SVG alpha instead.
- **Brand note:** comfy.org/brand: "Use as provided. Do not distort, recolor, or outline." and "don't modify the logo".
  Scene light, a perspective view and print texture are normal product placement [inference], but a refine that
  redraws the letters is a modification. Keep the shapes locked (section 6) and ask the organisers or Comfy Org if unsure.

## 5. The Blender route

- **Before the edit** (text as an image texture or emissive mesh on the Set's banner and scoreboard, rendered in the set
  render that `background from-set` sends as Picture 1):
  - Gains: perspective, folds and glow are geometrically exact, and the model sees the sign while it lights the whole
    room, so the glow on the wall and the reflections agree.
  - Risks: the set render is Workbench clay (switch to `color_type = 'TEXTURE'` so the yellow shows), and the full-frame
    edit runs at ~1 MP with ~2.5-token letters.
  - Whether Qwen keeps existing text exactly has no evidence beyond the "preserves font" claims. Our blank banner was
    redrawn with invented text [repo]. Expect the letters to drift, so follow it with the crop refine and the shape check.
- **After the edit:** render a text-only pass from the same camera, with the sign objects holding only the text and all
  else transparent (CPU Cycles is allowed next to a GPU job). Use it as an exact, perspective-correct alpha plus a shading
  hint for section 4. Blender's light is not the plate's light (Qwen invented it), so take the light from the plate, not
  from Blender.

## 6. Recommended pipeline and the ranking

**Ranked for our case** (integration first, exact brand shapes required):
1. **Exact glyph composite (section 4), then a Qwen-Edit crop-and-stitch refine at low denoise, then a shape check or
   lock.** The model adds material and light; the low denoise bounds the shape drift; this is the proven "paint, then
   refine" pattern [repo].
2. **Blender text on the Set before the from-set edit, then the same crop refine and check.** The best chance for
   scene-wide coherence (glow, light on the sign), but untested and it needs the refine anyway.
3. **Code composite only.** Exact and deterministic. Good for the emissive scoreboard; a printed banner can look pasted.
   It is also the fallback and the shape lock.
4. **Qwen-Image 2512 + DiffSynth Inpaint/Canny or InstantX Union with glyph edges.** Licence OK and core nodes, but a
   second model family, VRAM and downloads; untested.
5. **Qwen-Edit `Replace "Comfy" to "..."` with the wordmark already in Picture 1.** Font keeping is claimed; it likely
   drifts on a custom drawn mark.
6. **Text in quotes only.** Spells right in a generic font [repo]; useless for a brand. FLUX-based text models: licence
   excluded.

**Steps:**
1. **Glyphs:** the official logo SVG from comfy.org/brand (the logo zip), and "Open Source FTW" as a polished block-grid
   SVG. Rasterize them with `tools/svgraster.py` at 4x.
2. **Placement:** the sign quads from the Blender set via `align_to_set`.
3. **Paint in code:** a future `background.paint_text(name, svg, quad, material=print|emissive)` does section 4 and saves
   a NEW plate (never overwrite).
4. **Refine:** a background crop-and-stitch edit of each sign box. Use the character region-edit math (`_parse_region`,
   ~1 MP encoder-stable size) and 40 steps, cfg 3. The instruction describes the crop, puts the text in double quotes
   and ends with the keep line. Example: *'This is a close-up of a framed dark plum cloth banner on a gym wall, stylized
   3D anime noir. The yellow "Comfy" logo is screen-printed on the cloth: it follows the folds, the cloth's weave shows
   through the ink, lit by the same dim violet room light. Keep the letter shapes, their size and their position exactly
   the same. Keep everything else exactly the same.'* For the scoreboard: *'...the yellow text "Open Source FTW" is lit
   LED lettering glowing softly on the dark panel, a faint glow on the frame...'* Negative: misspelled text, extra
   letters, extra text, blurry letters, plastic sticker, flat overlay. Run 2 seeds × denoise 0.45 / 0.55 / 0.65.
5. **Check, then lock:** warp the exact alpha to the plate, binarize the refined crop (colour distance to the text colour
   plus Otsu), and allow ±2 px by ECC translation. Then measure:
   - glyph IoU;
   - edge F-score: Canny edges vs the glyph's edges within 1.5 px (sensitive to the notches);
   - OCR as a sanity check (Tesseract or PaddleOCR, both Apache-2.0; a stylized font can fail OCR).
   If IoU < 0.90 or edge F < 0.85, take the next lower denoise. If none passes, **shape-lock**: keep the refined
   background outside the exact alpha, and inside it fill with the refined letter colour (median of the overlapping letter
   pixels, extended by inpaint). The shapes are then exact and the light is the model's.

**A/B to decide between #1 and #2** (plus #3 as the control):
- One gym plate and both signs. Arm A = composite + refine (sweep above); Arm B = Blender text on the Set → from-set
  edit (same seed as `gym_noir_set`) → refine at 0.45 only when needed; Arm C = composite only.
- Score each candidate by:
  - shape: glyph IoU and edge F against the official SVG / prototype alpha;
  - a blind critic (a fresh agent; anonymous tags; the crops at 2x and the full frame; never told the method), scoring
    1-10 "looks part of the scene" (light, material, sharpness, perspective, colour cast) and brand fidelity (side by
    side with the SVG).
- **Win** = the best integration score among candidates with IoU ≥ 0.90 and edge F ≥ 0.85.
- Cost: about 4 min per region edit on the 3070 [repo]. A is ~12 edits, B is one set edit plus ≤4 edits.
- Write the results to `.cinestudio/gauntlet/text_research/RESULTS.md`.
