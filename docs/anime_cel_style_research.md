# A 2D anime cel style for the style library (research, 2026-10-27)

Goal: a library style `anime_cel_2d` that makes the dance demo (the masked dancer in the dim violet school gym)
read as 2D TV or film anime, not as a 3D render with outlines. No 2D-anime LTX-2.5 LoRA is available, so the look
must come from the pictures (sheet, look image, backplate redrawn in the style) and the words. Nothing here is
GPU-tested yet. Tags: [SRC] = a source says it; [OURS] = our own runs; [INF] = my inference, to be A/B-tested.

## Deliverable

### Recommended style JSON

```json
{
  "label": "Anime cel 2D",
  "version": 1,
  "words": "2D hand-drawn anime, digital cel animation frame, clean thin dark ink outlines, flat solid colour fills, one hard-edged shadow tone per colour, small sharp highlight shapes, painted 2D background art behind flat cel characters, crisp edges",
  "light": "flat cel lighting from one clear direction, hard-edged shadow shapes, a thin bright rim light along the figure's edges, a soft glow only around lamps and lit signs",
  "negative_image": "photo, photorealistic, 3D render, CGI, 3D animation, toon-shaded 3D model, video game screenshot, plastic, glossy specular highlights, subsurface skin, soft airbrushed gradients, smooth gradient shading, oil painting, watercolour bleed, rough pencil sketch, heavy film grain, depth of field, bokeh, lens flare, volumetric light rays, white outline around the figure, text, watermark",
  "lora": null,
  "reference": null,
  "_about": "docs/anime_cel_style_research.md (2026-10-27). Not A/B-tested yet."
}
```

`words` (36 words). Why each phrase is there:
- **2D hand-drawn anime**: the style comes first, and "2D" is said outright. LTX's 2D guide says to put the style first
  because the model falls back to photorealism [SRC 1]. Its domain guide counts "hand-drawn linework" as 2D vocabulary
  [SRC 2].
- **digital cel animation frame**: training data calls this look "cel animation" or "anime frame" [SRC 3]. "Digital"
  matches how anime has been made since the industry left painted cels [SRC 5, SRC 9]. "Frame" asks for a still from a
  moving picture, not an illustration poster [INF].
- **clean thin dark ink outlines**: anime has "very defined lines" [SRC 5]. "Clean" and "thin" keep out sketchy lines and
  thick comic-book lines [INF].
- **flat solid colour fills**: cel painting uses a paint bucket inside closed lines, one colour per segment [SRC 6]. The
  name "cel shading" comes from the fact that cels could not hold gradients [SRC 7]. Qwen-Image's own prompt examples say
  "flat design, large color blocks" [SRC 8].
- **one hard-edged shadow tone per colour**: an anime colour design sheet gives each element three colours: highlight,
  normal and shadow [SRC 6]. Cel shading uses only 2-3 shades [SRC 10]. This phrase is the main thing that tells the
  look apart from a soft 3D render [INF].
- **small sharp highlight shapes**: the third colour on the design sheet [SRC 6], drawn as hard shapes, not as specular
  shine.
- **painted 2D background art behind flat cel characters**: anime backgrounds are painted separately, by hand,
  traditionally in watercolour or gouache. They carry brush texture and more detail than the flat characters
  [SRC 11, SRC 12]. This also lets our backplate keep painted shading while the figure stays flat [INF].
- **crisp edges**: works against blur. A director of photography, interviewed by Anime News Network, says modern anime
  overuses the diffusion filter, which "just makes the picture look blurred". He limits it to cuts with an artistic
  reason [SRC 13].

`light` (one direction, a hard shadow, a rim light, glow only at the light sources): LTX's anime guide names "dramatic
lighting with rim highlights" [SRC 1]. Cel light is drawn as a few hard bands, not as smooth gradients [SRC 14]. The
traditional glow effect (transmitted light) is for light sources: sun, flames, beams, sparkle [SRC 15]. The violet
colour belongs in the shot's own text (AGENTS.md 7.2: name the same place and light as the background), not in the style.

`negative_image` (image jobs only; the video runs at CFG 1): the 3D and photo looks [SRC 3], plus the material words our
3D style uses on purpose (`3d_animated` asks for "soft global illumination" and "detailed fabric materials"). "Soft
airbrushed gradients" and "smooth gradient shading" catch the soft 3D look. "White outline around the figure" catches
the sticker halo seen on our earlier 2D kitsune (AGENTS.md 4.4 and 4.8). Leave out "chibi": kitsune is a chibi character.
Leave out "dark outlines" and "line art": those words are the look we want.

### Two `words` variants for an A/B test

- **A, TV anime** (the recommendation above, close to the shot's dim gym): flat, few tones, uniform lines.
- **B, theatrical anime film** (45 words):
  `2D hand-drawn anime feature film frame, finely drawn tapered ink outlines, flat cel colours with a hard-edged shadow tone and a second deeper shadow tone in folds, crisp highlight shapes, richly hand-painted detailed background art with visible brush texture, gentle light diffusion around bright areas`
  Film anime adds a second shadow tone, more detailed painted backgrounds and diffusion and glow in compositing
  [SRC 12, SRC 13]. Risk: "detailed" and "diffusion" may let LTX drift back toward a soft render [INF].
- A third control: the current `anime_cel` words, which the user finds not curated enough. "painted shading", "vivid
  magical atmosphere" and "cinematic" invite gradients and a lit set [INF; AGENTS.md 4.2 lists "cinematic" among the
  words that invite a lit set].

Test plan: one seed, the same template clip 1 in preview. Make the backplate, sheet and look image with each variant's
words (the pictures decide the style; see below), then compare. The judge is a blind critic asked: "2D anime frame, or
a 3D render with outlines?"

### What to check in the output pictures (proposed measurements; thresholds untested)

Inside the figure mask, and separately in the background:
1. **Flatness.** Smooth the frame with a mean-shift or bilateral filter, then segment it (SLIC or a flood fill at
   ΔE < 3). Measure the share of pixels within ΔE 3 of their segment's mode colour. A cel figure should be high (start
   at >= 0.8). A 3D render is low, because its shading is continuous.
2. **Shadow tones.** For each main hue cluster in the figure (k-means in Lab), the L* histogram should show 2-3 clear
   peaks: highlight, normal, shadow [SRC 6, SRC 10], not one wide hump. Count the peaks with a smoothed histogram.
3. **Soft-gradient share.** The share of figure pixels with a small but non-zero luminance gradient (about 1-6 L* per
   pixel at 768 px). A 3D shading ramp is high. A cel figure is low: flat areas give zero, lines give large values.
4. **Line presence.** The share of the figure's outline, and of internal segment borders, that has a thin dark line:
   a pixel darker than both sides by >= 15 L* within 2 px. Cel is high; "3D with outlines" has outlines but no interior
   cel borders.
5. **Highlight shape.** Small bright blobs should have hard edges (a steep edge profile), not a Gaussian falloff.
6. **Drift over the clip.** Run 1-4 on the first and last 25 frames of each clip. LTX's 2D guide warns that long clips
   drift toward photorealism [SRC 1], so a falling flatness score across clips or joins is the number to watch.
7. **Blind critic.** Ask "TV/film 2D anime or 3D render with outlines?" and "Which parts look 3D?" (docs/gauntlet_critic.md).

## Research body

### 1. What makes a picture read as anime cel

- **Production pipeline.** Key animators draw the line art. Finishing (shiage) fills each line-enclosed segment from the
  colour design sheet, which gives a highlight, a normal and a shadow colour per element. Animators mark the shadow and
  highlight regions with coloured lines [SRC 6]. Digital key-animation tools ship 3 colour-trace colours (red, green,
  blue) next to the black line for this purpose [SRC 16]. In the finished frame those boundaries are hard-edged colour
  changes, sometimes drawn as coloured lines instead of black ones [SRC 6, SRC 16]. Result: **solid black outlines on
  the silhouette, colour-to-colour hard edges inside.**
- **Shading.** It has no gradients, because the cel medium could not hold them [SRC 7]. Cel shading uses 2-3 shades
  [SRC 10]. Light shows as a few hard bands [SRC 14].
- **Line and colour.** Anime is described by "very defined lines, bright colors", large oval eyes and flat expression
  [SRC 5]. The eyes get a light shade, the base colour and a dark shade [SRC 5]. Hair is coloured and its movement is
  exaggerated [SRC 5]. Our dancer is masked, so face conventions matter less. The mask and hair are carried by the
  sheet.
- **Backgrounds.** They are painted separately, traditionally in watercolour or gouache from the layout [SRC 11], with
  brush texture and atmospheric colour [SRC 12]. Backgrounds carry the atmosphere [SRC 5]. The look is flat characters
  over a painted world. On a "book" layer, a painted foreground piece sits in front of the cels; I found no good source
  for it. It is not needed here.
- **Compositing (satsuei, "photography").** This step joins the finished paint data with the art team's backgrounds
  [SRC 13]. Diffusion (DF) filters, glow and blur are common, and the director of photography quoted above calls their
  overuse meaningless blur [SRC 13]. Glow for light sources comes from the cel-era transmitted-light technique [SRC 15].
- **Timing.** TV anime is often shot "on threes" (each drawing held three frames, about 8 drawings a second); American
  TV animation is usually on twos [SRC 17]. LTX makes continuous motion. Its 2D guide says the held-frame look needs
  post-processing or pacing cues [SRC 1]. Holding frames in ffmpeg after the stitch is possible but untested here [INF].
  It would also interact with our join cross-fades.

**What words can steer, and what they cannot.** Words can move line presence, flatness, shadow hardness, the background
type and rim light, all of which have stable visual names [SRC 1, SRC 2]. Words cannot hold the look through a clip.
LTX's guide says the strongest lever is a strongly stylized conditioning image, more than the text [SRC 1]. Our runs
agree:
- DL0b: a whole frame redrawn in anime by Qwen gave "clean anime everywhere".
- Anime words alone on a non-anime picture gave "clean 3D look, NOT anime".
- One anime frame pinned in video-to-video went photographic later.
These runs are in docs/style_consistency_research.md [OURS]. So the style's words matter most in the image jobs (the
backplate, the sheet, the look image). In the video they are a secondary hint.

### 2. Words that open models respond to

- **LTX-2 / 2.3 (Lightricks).** Put the style declaration first. The guide's anime vocabulary: "anime style animation",
  "Japanese animation aesthetic", "dramatic lighting with rim highlights", "speed lines". Its cartoon vocabulary:
  "cel-shaded rendering", "bold outlines with cell shading", "limited animation framerate" [SRC 1]. The domain guide
  says 2D words ("hand-drawn linework", "cel-shaded flat color", "limited animation with held frames", "ink-and-paint
  style") and 3D words ("rigged character", "sub-surface skin scatter") give visibly different results. Mixing them
  gives mixed results: "Pick one and stick with it" [SRC 2]. When several style vocabularies are combined, rank them
  (the primary first) [SRC 2]. Prompts are one paragraph of fewer than 200 words [SRC 2]. These guides are for LTX-2 and
  2.3. That they carry over to 2.5 is [INF], but likely: 2.5 is the same model family and uses the same Gemma text
  encoder named in [SRC 2].
- **Community LTX-2 test (not official, no data shown).** It suggests "cel shaded", "cell animation", "2D animation
  look", "anime screenshot" and "anime frame". It warns against "cartoon" (too broad) and against mixing realistic and
  anime terms, and it rates LTX-2's native anime ability as weak [SRC 3].
- **Explicit negation in the video text.** LTX's guide recommends "not photorealistic, no realistic textures, 2D only"
  when the model reverts to realism [SRC 1]. Our distilled video ignores negative prompts (CFG 1), so this is the only
  way to say "not 3D" to the video. But the style `words` also go into Qwen and Z-Image prompts, where AGENTS.md 4.2
  forbids negations. So keep them out of `words`. If the A/B shows drift, test "2D only, no 3D shading" in the run's
  video text or in a template's `clip_context` [INF].
- **Qwen-Image / Qwen-Image-Edit.** The official prompt rewriter asks for "a precise, niche style" described "concisely
  using key visual features". Its example of a flat drawn style uses "outlined lines, large color blocks, flat design,
  emphasized linework". For style transfer it names the target style's features in plain words [SRC 8]. Our working
  2D restyle instruction followed that pattern: "Redraw this whole picture as a hand-painted 2D anime film background:
  flat cel-shaded colours with soft painted gradients, clean hand-drawn outlines ..." (forest_2d, worked [OURS]). For
  the backplate and the look image, use the style words inside such an instruction: "Redraw this whole picture as
  <words> ... Keep the composition exactly the same".
- **Z-Image.** Its prompt rewriter wants objective, concrete descriptions without metaphors and without meta tags such
  as "8K" or "masterpiece" [SRC 18]. Every recommended phrase above describes something you can see.

### 3. Words that pull toward 3D, CG, painterly or photo (keep them out of the positive text)

- **3D and render words:** "3D", "render", "CGI", "C4D", "ultra-detailed CG". Qwen's own example uses "ultra-detailed
  CG, 32K resolution, C4D rendering" for a CG look [SRC 8]. Also "subsurface", "rigged" [SRC 2], "global
  illumination", "volumetric", "materials", "fabric detail", "realistic textures" [SRC 1]. These are exactly our
  `3d_animated` words [INF].
- **"cel-shaded" alone.** In games, "cel-shaded" means a toon-shaded 3D model. LTX lists "cel-shaded" under cartoons,
  which "range from classic hand-drawn animation to modern cel-shaded 3D" [SRC 1]. Always pair it with "2D" and
  "hand-drawn", or use "cel animation" [INF].
- **Painterly words on the character:** "painted shading", "soft painterly rendering", "watercolor". These are
  storybook words in LTX's guide [SRC 1]. Allow "painted" only for the background.
- **Lit-set words:** "cinematic", "premium", "polished", "masterpiece", "8K" (AGENTS.md 4.2 [OURS]; [SRC 18]).
- **Lens words:** "bokeh", "depth of field", "lens flare", "film grain". They read as camera footage. Anime does use
  them in compositing, but they are not what identifies it as anime [INF].
- **Studio and artist names.** Some guides tell users to name a studio or an era [SRC 4]. We do not, for licensing and
  style-mimicry reasons. The traits above describe the look instead.

### 4. Light for a 2D anime night interior (the violet gym)

- Night palettes in anime are led by deep blues and purples, with small warm or saturated accents [SRC 14]. These are
  secondary sources; I found no studio source on night interiors.
- In cel terms, a night scene is still flat fills. It gets darker base colours, a cool shadow tone and a hard shadow
  edge [SRC 6, SRC 10]. A thin coloured rim light separates the figure from a dark background [SRC 1]. Glow is used only
  on light sources, such as the scoreboard, lamps and EXIT signs [SRC 15], and kept restrained [SRC 13].
- Suggested shot words (for the run's `--video` text or `clip_context`, not the style): "lit by cool violet ambient
  light with flat dark violet shadows, a thin magenta rim light on the dancer's edges, soft glow around the scoreboard
  and the green EXIT signs" [INF]. The backplate must already be redrawn in the 2D style with the same words. A
  photographic or 3D plate pulls the whole video back to 3D [OURS, style_consistency_research.md].
- Risk: our dark-costume character on a dark background loses its black outlines. Then the rim light and the flat
  shadow tones carry the silhouette [INF]. AGENTS.md 4.5 has the same problem for sheets.

## Sources

1. Lightricks, "How To Generate 2D Animation With AI Video Models", https://ltx.io/blog/how-to-generate-2d-animation-with-ai-video-models
2. Lightricks, "Domain-Specific Prompting for LTX-2.3", https://ltx.io/blog/domain-specific-prompting
3. Apatero, "LTX-2 Anime Video Generation Guide" (community, no test data), https://www.apatero.com/blog/ltx-2-anime-video-generation-complete-guide-2025
4. CapCut, "AI Image for Japanese Anime Style", https://www.capcut.com/ideas/ai-image/ai-image-for-japanese-anime-style
5. Wikipedia, "Anime" (attributes, eyes, hair, backgrounds), https://en.wikipedia.org/wiki/Anime
6. Dai et al., "Paint Bucket Colorization Using Anime Character Color Design Sheets", https://arxiv.org/html/2410.19424v1
7. Graphixly, "Tips for Cel Shading", https://graphixly.com/blogs/news/tips-for-cel-shading
8. QwenLM, Qwen-Image prompt rewriter (prompt_utils.py), https://github.com/QwenLM/Qwen-Image/blob/main/src/examples/tools/prompt_utils.py
9. GarageFarm, "Traditional animation: the art of cel animation explained", https://garagefarm.net/blog/traditional-animation-the-art-of-cel-animation-explained
10. Saarland University CG1 course page, "Cel shading", https://graphics.cg.uni-saarland.de/courses/cg1-2018/RC/groupT_web/celshading.html
11. "Image Essay: Mobile Worldviews" (Mechademia, Project MUSE), https://muse.jhu.edu/article/488600
12. Krea, "Anime backgrounds with Krea 2", https://www.krea.ai/blog/anime-backgrounds-with-krea-2
13. Anime News Network, interview with director of photography Kentaro Waki (2019), https://www.animenewsnetwork.com/interview/2019-01-09/sword-art-online-alicization-director-of-photography-kentaro-waki/.140449
14. Morphic, "Anime color palettes: the colors behind the style", https://morphic.com/resources/how-to/anime-color-palettes
15. Japan Ministry of Internal Affairs and Communications, white paper chapter on video display methods (透過光), https://www.soumu.go.jp/main_sosiki/joho_tsusin/policyreports/japanese/papers/press/Japanese/Housou/0626J14.html
16. CLIP STUDIO TIPS, "Fundamentals of Digital Key Animation (Japanese Animation Style) #2", https://tips.clip-studio.com/en-us/articles/1356
17. Anime News Network lexicon, "Shot on threes (ones, twos, etc.)", https://www.animenewsnetwork.com/encyclopedia/lexicon.php?id=61
18. Tongyi-MAI, Z-Image prompt enhancer (pe.py), https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py
