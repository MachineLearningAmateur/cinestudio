# Prompting Qwen-Image-Edit-2511, Qwen-Image-2512 and Z-Image: research and rewrites (2026-10-22)

Every claim below is tagged **[OFFICIAL]** (the model makers' cards, repos and code), **[CODE]** (what ComfyUI or our repo actually does), **[RESEARCH]** (papers; tested on other models) or **[COMMUNITY]** (guides and forums). Each proposal ends with **A/B:** for the part that still needs a test.

## 1. Official guidance

### 1.1 Qwen-Image-Edit: the rewriter's rules (`polish_edit_prompt`)
QwenLM/Qwen-Image `src/examples/tools/prompt_utils.py` (commit 3453042c, 2025-12-23). The README says: "editing results may become unstable if prompt rewriting is not used. Therefore, we strongly recommend applying prompt rewriting". The rewriter runs on Qwen-VL-Max and sees the image. These are its rules, quoted:
- "Keep the enhanced prompt **direct and specific**." "Keep the core intention ... only enhancing its clarity, rationality, and visual feasibility." "All added objects or modifications must align with the logic and style of the edited input image's overall scene."
- Add / delete / replace: "If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.)". Their example: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera". For a replacement: "specify 'Replace Y with X' and briefly describe the key visual features of X."
- Text: "All text content must be enclosed in English double quotes", "keep the capitalization".
- Person (ID) edits: "Emphasize maintaining the person's core visual consistency (ethnicity, gender, age, hairstyle, expression, outfit, etc.)". Their example: "Replace the man's hat with a dark brown beret; keep smile, short hair, and gray jacket unchanged". So the rewriter **names the features to keep**. It does not use a general "everything else" clause.
- Style: "describe it concisely using key visual features" ("1970s disco style: flashing lights, disco ball, mirrored walls, colorful tones"). "If there are other changes, place the style description at the end." "Clearly specify the object to be modified": "Change the girl in Picture 1 to the ink-wash style of Picture 2 — rendered in black-and-white watercolor with soft color transitions."
- Fixed templates: "Perform inpainting on this image. The original caption is: " and "Extend the image beyond its boundaries using outpainting. The original caption is: ".
- Several images: "must clearly point out which image's element is being modified": "Replace the girl of picture 1 with the boy of picture 2, keeping picture 2's background unchanged".
- "Resolve contradictory instructions"; "if position is unspecified, choose a reasonable area".

### 1.2 How the images are named
- **[CODE]** ComfyUI's `TextEncodeQwenImageEditPlus` (comfy_extras/nodes_qwen.py, commit 6bfaacc) puts `"Picture {i}: <vision tokens>"` before the instruction for each input image. It also uses this system prompt: "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image ... maintaining consistency with the original input where appropriate." The model therefore sees the images labelled **"Picture 1" and "Picture 2"**, which are also the words of the official rewriter. We currently write "image 2". I found no official statement that requires "Image N" wording; one tutorial claims it, unverified.
- **[OFFICIAL]** 2509 card: "Optimal performance is currently achieved with 1 to 3 input images." The 2509 model also takes keypoint, depth and edge maps as input images.
- **[CODE] Finding in our repo:** `qwenedit_edit_api.json` always wires `image2`. When the user gives no reference, `edit_image` passes **the same picture as Picture 2** (`ref ... else clean_in`). The model then sees two identical pictures for a single-image edit. The official single-image use passes one image (2511 card, "Turn this cat into a dog"). **A/B:** a one-image edit graph against the current one, for drift and extra limbs at denoise 1.0.

### 1.3 Settings, language, negative
- **[OFFICIAL]** The 2511 and 2509 cards both use `true_cfg_scale 4.0`, `negative_prompt " "` (one space), 40 steps and `guidance_scale 1.0`. Our edits use cfg 3 with an empty negative, which is close (cfg 4 with a second image gave errors here; AGENTS.md). Qwen gives no negative-prompt advice for edits.
- **[OFFICIAL]** Qwen-Image-2512's README gives a Chinese negative prompt. In English: "low resolution, low quality, deformed limbs, deformed fingers, oversaturated, waxy look, faces without detail, overly smooth, AI look; chaotic composition; blurry, distorted text."
- **[OFFICIAL]** Language: English and Chinese both work, with a separate rewriter for each. Chinese matters mainly for **rendering** Chinese text. The 2509-era Chinese text-to-image rewriter (`polish_prompt_zh`, rule 8) says: "改写之后的prompt中不应该出现任何否定词" ("the rewritten prompt must contain no negation words"). Its example: "不要有筷子" ("no chopsticks") → chopsticks are simply not mentioned. The English rewriter keeps prompts "less than 200 words".
- **[OFFICIAL]** Qwen-Image-2512 rewriter (`prompt_utils_2512.py`, commit d755887d): one "fluent, natural descriptive language" paragraph with no lists; "Clearly specify the overall artistic style"; "Exercise restraint in simple scenes". When there is no text: "explicitly state: 'The image contains no recognizable text.'" This is a POSITIVE way to forbid text. For people, the order is identity → clothing → face → pose → background, and "around 200 words".
- **[OFFICIAL] 2511 release notes:** "mitigate image drift, improved character consistency, integrated LoRA capabilities", "further enhances consistency in multi-person group photos", "integrates selected popular LoRAs directly into the base model". Its demos show lighting control and new viewpoints "directly with the base model", but the blog gives no prompts.

### 1.4 fal Multiple-Angles LoRA (Qwen-Image-Edit-2511)
- **[OFFICIAL]** The card gives `<sks> [azimuth] [elevation] [distance]`: "`<sks>` trigger is essential", "Respect the order".
  - Azimuths: front view, front-right quarter view, right side view, back-right quarter view, back view, back-left quarter view, left side view, front-left quarter view.
  - Elevations: low-angle shot (-30°), eye-level shot, elevated shot (30°), high-angle shot (60°).
  - Distances: close-up (×0.6, "Details, textures"), medium shot (×1.0, "Balanced, standard"), **wide shot (×1.8, "Context, environment")**.
- Recommended strength is 0.8–1.0 ("Start with 0.9"). The LoRA was trained on "3000+ Gaussian Splatting renders", 96 positions. The card's tip: "Clear subjects with good lighting work best."
- The card does NOT say whether "right side view" means the camera on the subject's right or on the viewer's right. Our "redo gave the LEFT side" is therefore an ambiguity the card does not resolve.

### 1.5 dx8152 Light-restoration LoRA (Qwen-Edit-2509)
- **[OFFICIAL]** The card says its prompt is fixed: "移除光影,使用柔和光线（无明显光斑和阴影）对图片进行重新照明" ("remove the light and shadow and relight the picture with soft light, with no obvious light spots and shadows"). 2511 says it absorbed "popular LoRAs". Whether this one is among them is NOT stated.

### 1.6 Z-Image (base)
- **[OFFICIAL]** HF card for the base model: guidance 3.0–5.0, **28–50 steps**, negative prompt "Optional, but would be powerful when you want to remove some unwanted content". The base model supports negatives and has "high diversity"; Turbo has neither. Our graph follows the ComfyUI template: 25 steps, cfg 4, res_multistep.
- **[OFFICIAL]** The team, on Turbo (discussion #8): "works best with long and detailed prompts. You may consider first manually writing the prompt and then feeding it to an LLM to enhance it". The demo caps prompts at 512 tokens; locally they recommend `max_sequence_length` 1024.
- **[OFFICIAL]** Prompt-enhancer template (Space `Tongyi-MAI/Z-Image-Turbo` pe.py, rev 768cb50d; Chinese, summarised):
  1. First lock the unchangeable core: subject, **quantity**, action, state, named colours and text.
  2. Then add composition, light and atmosphere, material, colour scheme and spatial layering.
  3. Put text in English double quotes.
  4. The final description must be "客观、具象" (objective, concrete). No metaphors or emotional rhetoric, and **no meta tags such as "8K" or "杰作" (masterpiece)**.

## 2. Community and research findings (not official)
- **[RESEARCH]** Attend-and-Excite (arXiv 2301.13827) documents "catastrophic neglect": subjects or attributes in a prompt get dropped. SynGen (arXiv 2306.08877) shows attributes leaking to the wrong noun or binding to only part of it. Both were tested on Stable Diffusion, not on Z-Image or Qwen, but they match "bandages on 3 of 4 limbs".
- **[COMMUNITY]** Z-Image prompting gist (illuminatianon):
  - 80–250 words; "Long and precise = good", "poetic = often worse".
  - Order: shot → subject → clothing → environment → lighting → style → constraints, with "no X" phrases at the END.
  - Simple backgrounds ("plain studio background", "nothing distracting behind the subject").
  - The model "responds very well to lighting keywords".
  - Its CFG and negative advice is for Turbo and is disputed in the comments.
- **[COMMUNITY]** Qwen-Edit zoom and crop drift is blamed on the encoder's image rescaling. Phr00t's patched encoder scales the inputs to the target latent size (Rapid-AIO commit a16b312). We already make the output size match the input via FluxKontextImageScale. No action unless drift is seen.
- **[COMMUNITY]** Multi-image tutorials (ThinkDiffusion, stable-diffusion-art) name each image and its element: "A woman in image 1 is riding a vintage [bike] in image 2"; "Change ... the girl's top in image 2 from image 1".
- **[COMMUNITY]** Turnaround and character-sheet helpers (civitai, wikiprompt) use plain white or light-grey studio backgrounds plus colour swatches. I found no tested recipe for symmetric accessories. Everyone recommends naming each limb and fixing the odd limb by inpainting.
- **[OURS, AGENTS.md]** These already agree with the above: one change per edit; a tight crop as the reference; `--region` for small details; "cinematic studio light" gave strong light; invented details ("brown straps") appeared; a back view needs "the face is NOT visible".

## 3. Our texts and what to change

### 3.1 Current FRONT_PROMPT, style and description (masked3d)
`FRONT_PROMPT` = "A full-body character design illustration of ONE original character: {description} The character stands straight ... neutral A-pose, arms about 35 degrees ... exactly five clearly separated fingers each, legs slightly apart. The whole body is visible ... on a plain flat black background. No text, no scenery, no floor shadow, no other characters. Style: {style}."

`DEFAULT_STYLE` = "cinematic fantasy anime cel-shaded illustration with hand-drawn linework, painted shading and stylized fantasy character design".

masked3d's style text = "premium stylized cinematic 3D animation, polished animated-film quality, soft stylized proportions, detailed fabric materials, soft global illumination, evenly lit by soft neutral light from the front".

Problems, each with its source:
1. **The description has no closing period and repeats the pose.** The masked3d description ends "...exactly five clearly separated fingers on each hand The character stands straight...", so the pose and fingers are said twice and the sentence boundary is lost. Source: our stored `front_prompt`.
2. **Negations in the positive text** ("no cape, no poncho, no hood, no scarf", "No text, no scenery, no floor shadow"). Each one names the very object it forbids. Qwen's rewriter removes all negation words; Z-Image base has a working negative prompt, which is where these belong. [OFFICIAL zh rule 8 + Z-Image card]
3. **Light words that fight each other.** "Cinematic", "soft global illumination" and "premium / polished" invite dramatic or studio light and a backdrop. Z-Image's enhancer bans meta-quality tags. AGENTS.md already saw "cinematic studio light" give strong light. [OFFICIAL pe.py + OURS]
4. **"Plain flat black background" is said once, in the middle.** "3D animation" plus "global illumination" suggests a lit set, which gave the brown backdrop. There is no negative against a backdrop: the graph's negative only has "background scenery". [COMMUNITY + inference]
5. **Repeated items are written as one plural phrase** ("white bandages wrapped on the forearms and the shins"). There is no count, no side and no extent, so the model binds the attribute to part of the set. [RESEARCH, inference]
6. **25 steps is below the card's 28–50.** [OFFICIAL]

### 3.2 Proposed front template (Z-Image base)
One paragraph and ordered slots. This keeps the Qwen and Z-Image rewriters' order (subject → clothing → face → pose → background → light → style), states the medium first and keeps negations out:

```
{MEDIUM} of ONE original character for a character model sheet: a single full-body front view, centered, isolated on a solid uniform pure black background.
{IDENTITY}.                       # build / age read / proportions, one sentence
{COSTUME}                         # head-to-feet clauses, grammar 3.3
The character stands straight and faces the viewer in a symmetrical A-pose: both arms held about 35 degrees away from the body, both hands open and relaxed with five clearly separated fingers each, feet a little apart, the whole body visible from the top of the head to the soles of the feet with empty black space all around.
The light is flat and even from the front, like a neutral turntable reference render: soft shadowless illumination, every colour at its true value, matte surfaces with gentle shading only.
The background is the same solid pure black everywhere, empty and seamless, without floor or horizon.
Style: {STYLE}. The image contains no recognizable text.
```

- **MEDIUM** for masked3d: "A stylized 3D animated-film character render". **STYLE**: "stylized 3D animation, soft stylized proportions, detailed fabric materials". Drop "premium", "polished ... quality", "cinematic" and "global illumination".
- **Sources:** the slot order and one paragraph follow the 2512 rewriter and pe.py; the explicit style follows the 2512 rule 5; "contains no recognizable text" is the 2512 rewriter's own wording; banning meta tags follows pe.py; "objective, concrete" follows pe.py; light keywords follow [COMMUNITY].
- **A/B:** (a) new template vs current, same seeds 1–8, scored on bandage count, backdrop and light balance; (b) 25 vs 40 steps; (c) with vs without "turntable reference render" (it could pull a 3D-viewport look).

### 3.3 Description grammar (for users and agents; `character create --text`)
Rules:
- Describe **head to feet**, one garment or part per clause.
- Each clause gives colour + material + **extent** (from where to where) + **side and count**.
- Repeated items get BOTH, left and right, a count, and "identical".
- Only what IS there. Excluded items go to a separate exclude list, which feeds the negative.
- No pose, fingers or light in the description: the template says those.
- Keep the description to about 120–180 words; the template adds about 120.

The masked3d description rewritten:
> a slim figure with a smooth white oval mask covering the whole face, two black almond-shaped eye holes angled upward like snake eyes, no mouth opening; short dark plum-purple hair. A fitted black long-sleeved bodysuit from neck to ankles. A short fitted plum-purple jacket with a thin magenta trim along every edge, ending exactly at the waist. Black gloves on both hands. White cloth bandages wrapped around BOTH forearms, left and right, from the wrist to just below the elbow, and around BOTH shins, left and right, from the ankle to just below the knee: four identical bandage wraps in total. Plain dark charcoal shoes on both feet.
> exclude: cape, poncho, hood, scarf, cloak, loose hanging cloth

- **Sources:** naming the count and the sides follows the pe.py core rule (quantity) and the Qwen edit rewriter (category, colour, size, position); the bound attributes follow [RESEARCH]. "four identical ... in total" is my addition. **A/B:** the "both / left and right / four in total" wording vs the current phrase on 8 seeds; count the limbs with bandages.

### 3.4 Negative prompt for Z-Image base (cfg 4 uses it)
Current graph negative: "blurry, low quality, deformed hands, extra fingers, missing fingers, extra limbs, cropped body, text, watermark, signature, background scenery". Proposed:
```
textured backdrop, gradient background, vignette, studio backdrop, wall, floor, ground shadow, scenery, room, props,
dramatic lighting, rim light, spotlight, harsh shadows, strong highlights, glossy specular glare, high contrast,
asymmetrical outfit, mismatched accessories, extra arms, extra legs, extra fingers, missing fingers, fused fingers,
deformed hands, cropped head, cropped feet, multiple characters, duplicate figure, text, watermark, signature, logo,
blurry, low quality, oversaturated, {EXCLUDE}
```
- **Sources:** the card ("powerful when you want to remove some unwanted content"); Qwen-2512's official negative contributes "oversaturated, deformed limbs/fingers". Leave out its "waxy / overly smooth" terms for a stylized 3D look.
- **A/B:** the full list vs the current one. A long negative at cfg 4 can flatten or desaturate the picture, so watch the colours.

### 3.5 Edit templates (Qwen-Image-Edit-2511)
General form, after the official rewriter: one verb, the target with its location, the new look with 2–4 key visual features, then named features to keep, then our general clause.
- **Add:** "Add white cloth bandages around the character's LEFT shin, from the ankle to just below the knee, identical to the bandages on the right shin. Keep the mask, the jacket, the pose and the black background unchanged. Keep everything else exactly the same."
  - Say "the character's left", not "the left of the picture". If the side matters, add the picture side too: "the leg on the right side of the picture".
- **Replace:** "Replace the brown textured backdrop with a solid uniform pure black background. Keep the character, its pose, size and position exactly the same." (official "Replace Y with X")
- **Remove:** "Remove the hood from the character's head, showing the short dark plum-purple hair underneath. Keep the mask and the jacket unchanged. Keep everything else exactly the same."
- **Recolour:** "Change the colour of the shoes to dark charcoal grey. Keep their shape and everything else exactly the same."
- **Region (`--region`):** the model sees ONLY the crop, so describe the crop: "This is a close-up of a black-gloved hand hanging at the side. Redraw the hand with five clearly separated fingers, flat cel-shaded, without fingernails. Keep the glove colour, the bandage and the background exactly the same."
- **Reference (Picture 2):** "Change the bandages on the character's forearms in Picture 1 to match the bandages in Picture 2: white cloth strips wrapped in a spiral. Keep Picture 1's character, pose and background unchanged."
  - Rename "image 2" to **Picture 2** in the UI hint and in runner.LOOK_INSTRUCTION (the encoder's own label).
  - Keep the tight-crop rule from AGENTS.md.
- **Light flattening:** first try the dx8152 wording in English and Chinese on 2511 without a LoRA: "Remove the light and shadow and relight the picture with soft even light, with no obvious light spots and no shadows. Keep the character and the background exactly the same." / "移除光影,使用柔和光线（无明显光斑和阴影）对图片进行重新照明".
  - **A/B:** this against our OpenCV delight.py; with the 2509 LoRA only if 2511 needs it (compatibility unknown).
- **Style:** the key visual features, at the end of the instruction (official).
- **Infill:** "Perform inpainting on this image. The original caption is: <caption>" (official fixed template) for painted-out areas.
- **A/B:** named features to keep + the general clause vs the general clause alone (5 edits, same seeds); "Picture 2" vs "image 2".

### 3.6 Views (fal Multiple-Angles)
Current: "<sks> right side view eye-level shot wide shot" (also back and left); face "<sks> front view eye-level shot close-up". These follow the card's grammar exactly.
- **Rooms behind the character:** the card defines **wide shot as ×1.8 distance, "Context, environment"**, and the LoRA was trained on splat renders of full scenes. "Wide shot" probably invites the room. Our front already shows the whole body, so **medium shot** (×1.0, "Balanced, standard") keeps the input's framing.
  - **A/B:** medium vs wide, 3 seeds per view. Watch for cropped feet with medium, and a room or a smaller figure with wide.
- **Wrong side:** the card does not fix the convention, so add an automatic check instead of a wording change. Compare the view with the mirrored front (the facing direction of the mask or nose), or test whether "right side view" really shows the character's right side on 4 seeds, then pin the mapping in code.
- **Extra words after the LoRA prompt** ("on a solid black background", "the face is not visible" for the back): the card says to use the exact format. Our cloaked3d back view needed the words (Qwen without the LoRA).
  - **A/B:** the bare LoRA prompt vs the prompt + ". Solid pure black background." on 3 seeds.
- **Quarter views** (front-right quarter view etc.) exist in the card's list. A 3/4 panel is in Lightricks' sheet example (AGENTS.md); add it as an optional slot.

### 3.7 Panoramas and text templates
- **Panorama** (`PANO_PROMPT`): replace "No people, no animals, no text." with the 2512 rewriter's "The image contains no recognizable text" and a positive "the place is deserted and still"; people and animals stay in `NEGATIVE` (cfg 4 uses it). Add Qwen-2512's official "oversaturated, chaotic composition" to `NEGATIVE`. **A/B:** person count on 3 seeds.
- **Text template** (`IMAGE`): "no other people anywhere, no pedestrians, no crowd, no animals" names the people it forbids, and the park run got small people by the fountain. Use "the place is deserted: the stand-in is alone in the whole scene"; these words are already in `NEGATIVE`, which the base model uses. Replace "no text" with the 2512 sentence. **A/B:** large and small person counts by DWPose on 6 seeds.

## 4. What is unproven
None of the rewrites has been tested in our pipeline. The most likely wins, in my order:
1. the per-limb grammar with an exclude list routed to the negative;
2. a backdrop and light negative plus no "cinematic / global illumination";
3. medium shot for the views;
4. one-image edits without the duplicate Picture 2.

Language (Chinese prompts) is unlikely to matter for us except for the dx8152 light prompt. Keep everything in English.

## Sources
- QwenLM/Qwen-Image README and 2509 notes (prompt rewriting "strongly recommend"): https://github.com/QwenLM/Qwen-Image , https://github.com/QwenLM/Qwen-Image/blob/main/Qwen-Image-Edit-2509.md
- Edit rewriter + T2I rewriters, commit 3453042c: https://github.com/QwenLM/Qwen-Image/blob/main/src/examples/tools/prompt_utils.py
- 2512 rewriter, commit d755887d: https://github.com/QwenLM/Qwen-Image/blob/main/src/examples/tools/prompt_utils_2512.py
- Qwen-Image-Edit-2511 card: https://huggingface.co/Qwen/Qwen-Image-Edit-2511 ; blog: https://docs.qwenlm.ai/qwen-image-edit-2511/index.html
- Qwen-Image-Edit-2509 card: https://huggingface.co/Qwen/Qwen-Image-Edit-2509
- ComfyUI TextEncodeQwenImageEditPlus ("Picture N", system prompt), commit 6bfaacc: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_qwen.py
- ComfyUI 2511 tutorial: https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511
- fal Multiple-Angles LoRA: https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
- dx8152 Light restoration: https://huggingface.co/dx8152/Qwen-Image-Edit-2509-Light_restoration
- Z-Image base card: https://huggingface.co/Tongyi-MAI/Z-Image ; Turbo card: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo ; team on prompts: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/8
- Z-Image PE template (Space rev 768cb50d): https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py
- [COMMUNITY] Z-Image prompting gist: https://gist.github.com/illuminatianon/c42f8e57f1e3ebf037dd58043da9de32
- [COMMUNITY] Qwen-Edit scaling/zoom fix: https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/commit/a16b312300e0a9acb1a8de78b5a48c4df11b7613
- [COMMUNITY] multi-image prompting: https://learn.thinkdiffusion.com/qwen-image-edit-2509-combine-multiple-images-into-one-scene-for-fashion-products-poses-more/ , https://stable-diffusion-art.com/qwen-image-edit-multiple-images/
- [COMMUNITY] sheet helpers: https://civitai.com/models/100435 , https://www.wikiprompt.org/character-turnaround-sheet-prompt
- [RESEARCH] Attend-and-Excite: https://arxiv.org/abs/2301.13827 ; SynGen (linguistic binding): https://arxiv.org/abs/2306.08877
