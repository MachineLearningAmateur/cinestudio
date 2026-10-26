# Prompting a long shot clip by clip (LTX-2.5 Layout-To-Render + Ingredients): what Lightricks, the research and the community do

Date 2026-10-25. Tags: **[OFFICIAL]** Lightricks docs, model cards, shipped code and system prompts; **[RESEARCH]** papers;
**[VENDOR]** other model makers' official guides (OpenAI, Google, Runway); **[COMMUNITY]** third-party guides and
nodes; **[INFERENCE]** my reasoning. No GPU jobs were run for this report and no code was changed.

## 0. Short answer

Lightricks gives **two different prompt styles** for the two LoRAs we stack. Layout-To-Render says the text should be
**"One or two sentences describing the finished shot. Match the first frame"**, and the LoRA was trained "with short
simple prompts and a layout video as the control signal". Ingredients wants a long two-part text whose Target part
"drives the action". Lightricks' control guidance says **motion comes from the reference video and the prompt should
describe the look ("prompt the style ... not the motion")**. Its own chunked long-video pipeline sends **one prompt to
every chunk**. The ComfyUI looping sampler also supports **one full prompt per tile** (a hard switch, no blending).

The research agrees with our per-clip rule. One merged text gives consistency but poor event adherence (TALC: text
adherence 33 vs 70). Separate texts give adherence but drift. Our relay, look chain and sheet already supply the
consistency, so per-clip texts get both. No source supports long event narratives on top of a control video. Several
say that **text which contradicts or over-explains the control hurts** (Lightricks: "Inaccurate descriptions may cause
scene cuts").

What I recommend: keep the per-clip structure, but make the clip text **short and in the training-caption
vocabulary**:
- Shared: style, shot type + viewpoint + "the camera remains static", the character's marks, place and light, facing,
  and a pacing/motion-style sentence.
- Per clip: one or two sentences of body mechanics with picture directions.
- **No cause-and-threat events** ("a ball flies at the head").
- Each later clip **opens on the state** that the shared frames show, not on a repeat of the action.

Test all of this with a 4-arm clip-1 A/B (section 6).

## 1. Official Lightricks guidance

### 1.1 General LTX-2.5 prompting (docs.ltx.io Prompting Guide)
Source: https://docs.ltx.io/open-source-model/usage-guides/prompting-guide.md [OFFICIAL]
- Six elements: establish the shot, set the scene, describe the action, define the characters, camera movement, audio.
- Action: "Write the core action as a natural sequence that flows from beginning to end. Give each sentence a verb that
  does something — *walks, turns, exhales, reaches*". Camera: "Specify how and when the camera moves. Describing how
  subjects appear *after* the movement helps the model complete the motion accurately."
- Single shot: "a **single flowing paragraph**", "**present tense**", "roughly **4–8 descriptive sentences**", "Keep
  lighting consistent — use one coherent light logic per shot", "Start simple and layer".
- Pacing: "It won't stretch a moment or add a pause you didn't prompt for. Write the beats you want into the prompt".
- Multishot (2.5): "**Keep identity consistent** — reuse the same visual identifiers for recurring people or objects";
  "**Keep action chronological** e.g. 'Initially…', 'A moment later…'"; "Avoid conflicting geography or unexplained
  costume changes". For a first-frame start, "prefer a **single continuous take**".
- "Complex physics — highly chaotic motion can introduce artifacts; simpler, more plausible motion is more reliable."
- Pacing words are a real lever. LTX's 2D-animation blog says: "The phrase 'gentle, calming motion' is a pacing cue
  ... prevents the model from generating the kind of energetic motion that works for cartoons"
  (https://ltx.io/blog/how-to-generate-2d-animation-with-ai-video-models) [OFFICIAL blog].

### 1.2 Layout-To-Render (the LoRA that carries our motion)
Sources: https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render (gated card, read with our HF token),
https://docs.ltx.io/open-source-model/vfx-post-production/layout-to-render.md [OFFICIAL]
- "**Prompting:** One or two sentences describing the finished shot. Match the first frame. Do not write 'clay,' '3D,'
  or 'Unreal.'"; ComfyUI step 7: "Write a short prompt that describes the first frame, and run."
- Training: "IC-LoRA trained on LTX-2.5 dev, **with short simple prompts** and a layout video as the control signal."
- "The layout clip sets the camera, the motion, and the framing, and it keeps objects in place. The image sets palette,
  lighting, and materials."
- The three card examples are look-and-place sentences. Even the blocking example ("Two men in dark coats stand with
  their backs to the camera in a night apartment ...") has no motion narrative. The well example adds "Match the first
  frame: high summer ... Keep the same well, bucket, posts, and camera as the driver."
- The docs say: "Do not describe the layout source — avoid words like 'clay,' '3D,' or 'Unreal,' which push the output
  toward the blockout look".

### 1.3 Ingredients 2.5 (the sheet LoRA)
Source: https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients [OFFICIAL]
- "The prompt is split into two labeled parts, matching how the model was trained": `Reference sheet: ...` /
  `Generated video: ...`. The sheet "supplies the *what things look like*, and the `Generated video:` portion of the
  prompt supplies the *what happens*." The card's six widget examples use the headings `### Reference Sheet
  Description` / `### Target Description` (what we use). The card is inconsistent here, so both forms are official.
- The Target examples are long (8-12 sentences). They re-describe the character with the sheet's words and narrate one
  or two actions in time order with connectors ("then", "pauses for a beat"). They are dialogue scenes without a
  control video.

### 1.4 Control IC-LoRAs: text vs control
- IC-LoRA guide (https://docs.ltx.io/open-source-model/usage-guides/ic-lo-ra.md) [OFFICIAL]: "describe visual style,
  not control type"; "**Align prompt with the reference input's motion and composition**"; "**Avoid contradicting the
  structure or content of the reference**".
- Union Control guide (https://docs.ltx.io/open-source-model/feature-guides/structural-control/union-control.md)
  [OFFICIAL]: "Describe the visual content you want, not the structure, which the reference already provides. If your
  reference shows a person walking and you're using pose, **prompt the style** ('a knight in silver armor walking
  through a misty forest'), **not the motion**." Note that the example still contains the verb "walking": one plain
  verb, no choreography.
- LTX blog on IC-LoRA (https://ltx.io/blog/how-to-use-ic-lora-in-ltx-2) [OFFICIAL blog]: "Motion comes from reference
  video; prompts control visual style only."
- [COMMUNITY] ltxworkflow.com (third-party despite "Official" in its title,
  https://ltxworkflow.com/resources/tutorials/ic-lora-ltx-2-3-complete-guide): avoid "Camera pans left", "Character
  walks forward"; "Describing it in the prompt creates conflicting instructions and degrades output quality."

### 1.5 The prompt enhancer = the training-caption style
- The LTX-2.5 (Gemma 4) enhancer system prompts say they write "in the EXACT style of the training captions"
  (https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma4_i2v_system_prompt.txt,
  last change 2026-08-11) [OFFICIAL]. For every shot the caption MUST include:
  - the shot type ("exactly one: ... wide shot / medium shot ...");
  - the camera motion ("always stated; if none, explicitly say the camera remains static");
  - the "Camera viewpoint relative to subject (**front-facing / back-facing / side view** ...)".
  It also asks for "Strict chronological, real-time flow using transitions like 'Initially…', 'A moment later…'" and
  "Objective, observable description only". For I2V: "the opening of your caption must match the reference image
  exactly ... then narrate chronologically". Length 150-220 words.
- The older (LTX-2/2.3, Gemma 3) I2V prompt, still shipped in ComfyUI-LTXVideo
  (https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/system_prompts/gemma_i2v_system_prompt.txt) [OFFICIAL]:
  "**Describe only changes from the image: Don't reiterate established visual details. Inaccurate descriptions may
  cause scene cuts.**"; "No timestamps or cuts"; "DO NOT invent camera motion"; "Restrained language".
- [INFERENCE] "front-facing" and "the camera remains static" are in-distribution words for the base model. Our facing
  words and a static-camera statement are therefore not foreign "camera words".

### 1.6 Long video: what Lightricks' own tools do with prompts
- The chunked pipeline (`ltx_pipelines/chunks/planner.py`, `generate_uniform_chunks`, https://github.com/Lightricks/LTX-2)
  [OFFICIAL code] stamps **the same `video_context`/`audio_context` (one prompt) on every chunk** (97-frame chunks,
  25-frame carry). That is one shot, one description.
- `LTXVLoopingSampler` + `MultiPromptProvider` (ComfyUI-LTXVideo `looping_sampler.md`, installed commit 220e1d4)
  [OFFICIAL]: "Consider using different prompts per tile via MultiPromptProvider"; the recommended "Long-Form Content"
  preset is "+ MultiPromptProvider for narrative progression". Prompts are "pipe-separated" and "Each prompt is used in
  one temporal_tile". Each tile gets a complete, separately encoded prompt. There is no interpolation and no shared
  part, so the user repeats whatever must persist.
- Extend API (https://docs.ltx.io/api-documentation/api-reference/async-video-generation/submit-extend.md) [OFFICIAL]:
  `prompt` = "Description of what should happen in the extended portion of the video"; the example prompt is just
  "Continue the motion smoothly".
- "How to make longer AI videos" (https://ltx.io/blog/how-to-make-longer-ai-videos) [OFFICIAL blog]: "Subject or
  colour identity was never pinned. **Supply a keyframe image at the join rather than re-describing the subject in
  prompt text.**"; "**Put joins at rest points where you can**, so the bridge has less to reconcile."

## 2. Community practice for segment prompts

- Scenario's LTX-2 Extend help (https://help.scenario.com/articles/8433569940-ltx-2-19b-extend-video-seamless-video-continuity)
  [COMMUNITY/vendor]: "describe only the new action for the added segment, not the whole scene", because the model
  already has the context.
- Prompt Relay (Kijai's ComfyUI node for LTX-2.3 and Wan, https://github.com/kijai/ComfyUI-PromptRelay; paper
  https://arxiv.org/abs/2604.10030) [COMMUNITY + RESEARCH]:
  - a `global_prompt` "anchors persistent details across the entire video" (style, character, lighting); local prompts
    hold each period's content;
  - README tips: keep segment 1 "to the static visible state", and later segments "describe only what changes, without
    repeating segment 1";
  - the paper suppresses cross-attention to a prompt outside its time window, with a smooth decay at the edges, and
    credits better quality to "reduced competition between prompts in cross-attention".
  - This is the same shared-context + per-segment shape as our rule, inside one clip.
- Wan context windows: WanVideoTextEncode spreads `|`-separated prompts evenly over the video (aggregator pages, e.g.
  https://comfy.icu/node/WanVideoTextEncode) [COMMUNITY].
- I found no community thread with measured evidence on how long a per-segment LTX prompt should be.

## 3. Research and other model makers

- **TALC** (https://arxiv.org/abs/2405.04682) [RESEARCH]:
  - merging all scene captions into one prompt gave the highest visual consistency (96.5) but weak adherence (33.0):
    the model "does not understand the temporal boundaries between the two events";
  - generating the scenes separately gave adherence (70.0) but poor consistency (55.0);
  - time-aligned captions per segment won (+29% overall).
- **Mind the Time / MinT** (https://arxiv.org/abs/2412.05263) [RESEARCH]: single-prompt models "often ignore some of the
  events" and "fail to arrange them in the correct order"; MinT binds each event to its time span.
- **DiTCtrl** (https://arxiv.org/abs/2412.18597) [RESEARCH]: a merged long prompt "fail[s] to capture semantic
  transitions". Smooth transitions come from KV-sharing with the previous segment plus latent blending over the
  overlap, i.e. from the latents, not from the words.
- **FreeNoise** (https://arxiv.org/abs/2310.15169) [RESEARCH]: when the prompt changes, the first prompt keeps the early
  (layout) steps and the new prompt enters in the shape/motion steps, interpolated over a few frames "to achieve smooth
  transition". Changing everything at once lost a leg and the layout.
- **Presto** (https://arxiv.org/abs/2412.01316): one overall caption + "five progressive sub-captions", one per temporal
  segment.
- **LongLive** (https://arxiv.org/abs/2509.22622): prompt switches need cache refresh for "smooth, adherent switches".
- **Gen-L-Video** (https://arxiv.org/abs/2305.18264): multi-text long video by co-denoising overlapping clips.
- **VideoDirectorGPT** (https://arxiv.org/abs/2309.15091): "consistency groupings of the entities", i.e. the same entity
  words across scenes.
- **MEVG** (https://arxiv.org/abs/2312.04086): an LLM splits the user's text into one prompt per event.
- **OpenAI Sora 2 guide** (https://developers.openai.com/cookbook/examples/sora/sora2_prompting_guide) [VENDOR]:
  - "Each shot should have one clear camera move and one clear subject action";
  - "Actions work best when described in beats or counts"; "Movement is often the hardest part to get right, so keep it
    simple";
  - "reuse phrasing for continuity, and avoid mixing traits that may compete";
  - "When a result is close, pin it as a reference and describe only the tweak".
- **Google Veo 3.1 guide** (https://cloud.google.com/blog/products/ai-machine-learning/ultimate-prompting-guide-for-veo-3-1)
  [VENDOR]: timestamped segments `[00:00-00:02] ...` inside one generation; consistency comes from reference images
  ("ingredients"), not from repeating text.
- **Runway Gen-4** (https://help.runwayml.com/hc/en-us/articles/37327109429011) [VENDOR]: "Since the image conveys key
  visual information about subjects, composition, colors, lighting, and style, your text prompt should be almost
  entirely focused on describing the desired motion."
- **Kling 3.0 via fal** (https://blog.fal.ai/kling-3-0-prompting-guide/) [COMMUNITY]: "Character labels must be unique
  and consistent. Avoid pronouns or synonyms."
- What the research says makes joins smooth: shared latents/attention at the overlap (we have this: relay + flow
  cross-fade), a fixed anchor, and texts that change only the event part while the subject phrase stays word-for-word
  the same. The common phrasing advice: one action (or beat) per segment, in time order, with connectors; positive,
  observable verbs.

## 4. Motion fidelity with a control video: what text helps and what hurts

- **Helps (official):**
  - text that agrees with the layout's motion and composition (1.4);
  - the facing/viewpoint triple of the training captions (1.5);
  - pacing words that set the motion energy (1.1);
  - one plain verb for the overall action (Union Control's "walking").
- **Hurts (official):**
  - text that contradicts the control (1.4), or "inaccurate descriptions" of what is visible (cuts, 1.5);
  - invented camera moves (1.5);
  - "clay/3D" words (1.2);
  - "highly chaotic motion" (1.1).
- **Event words with a cause** ("a ball flies straight at the head; the dancer dodges") [INFERENCE + our evidence]: the
  model has a strong prior for what dodging a thrown object looks like (flinch, turn away, lean, step out of line).
  When the layout's sideways slide is small, that prior can win. That fits the profile walk we measured.
  - Neither "describe what the control shows" nor "describe less" is tested anywhere for LTX. Lightricks' own examples
    and advice lean to describe less, but every example that does name motion names it in agreement with the control.
  - So the safe middle is: body mechanics that match the rig (direction, stop, which foot), the pacing word ("fast",
    "hard stops"), and no causal story.
- **Objects coming at a person:** Layout-To-Render "keeps the placement of objects from the layout". The props
  sentence names the balls' look, so the threat story adds nothing the layout lacks [INFERENCE].

## 5. Recommendation for cinestudio

### 5.1 Top rules (source in brackets)
1. **Keep one prompt per clip with a shared context**, word-for-word identical across clips (TALC, MinT, DiTCtrl,
   Prompt Relay; LTX multishot "reuse the same visual identifiers"; Sora "reuse phrasing").
2. **Shared context = training-caption vocabulary:**
   - style first (LTX enhancer "Style: <style>, ...");
   - one shot type + "front-facing" viewpoint + "the camera remains static" when the template's camera is still (LTX-2.5
     enhancer: always stated);
   - the character's 2-3 marks, the place and the one light logic;
   - the facing state and one pacing/motion-style sentence (LTX pacing cue).
3. **Clip actions: one or two sentences, about 15-40 words**, body mechanics only (verb + direction + stop), in time
   order with "first ... then ...". No timestamps (Layout-To-Render "one or two sentences"; Sora "one clear subject
   action", "keep it simple"; LTX enhancer "No timestamps").
4. **Describe what the frames show, never more** (LTX: "Avoid contradicting the structure"; "Inaccurate descriptions may
   cause scene cuts").
5. **No cause-and-threat stories** for objects the layout already moves. Name props by their look in the shared
   context ("violet rubber dodgeballs"), and at most "a ball passes beside the head" (Union Control "not the motion";
   our 2026-10-25 runs).
6. **Clip k ≥ 2 opens on the state the shared frames show** ("Still facing the camera with the left foot lowering after
   the kick, ...") and then gives only its new actions. A move that completes inside the 25 shared frames is a verb in
   clip k-1 and a state in clip k (Gemma 4 I2V "opening must match the reference image"; Layout-To-Render "Match the
   first frame", where clip k's first frame is the chained look; Extend API "what should happen in the extended
   portion"; Prompt Relay "only what changes").
7. **Pin identity with images, not words:** the sheet, the look and the keyframe looks carry the look. Do not grow the
   Target part with more costume text (LTX blog: "Supply a keyframe image at the join rather than re-describing the
   subject").
8. **Plan joins at held or rest poses** (LTX blog "Put joins at rest points"; AGENTS 5 template rules).
9. **Restrained, observable words**: no "explosive", "dramatic" or intent words; picture directions for sideways moves
   (LTX enhancer "Restrained language", "Objective only").
10. **Check the assembled prompt's length.** The whole Target part (style + marks + place + props + context + actions +
    sound) should stay near the Ingredients examples (~120-180 words), with the action itself short.

### 5.2 Where this goes against AGENTS.md 7.2 today
- **Length.** 7.2 says "~4-8 sentences" for the whole text and asks for the direction of every move. The
  Layout-To-Render card says "one or two sentences" and was trained on short prompts. That is a real conflict for the
  ACTION part. Keep the action part short and let the layout carry the detail (A/B arm B vs D below).
- **"Name the DIRECTION of every move."** Lightricks' control guidance says prompt the look, not the motion. Directions
  that agree with the layout are allowed ("align with the reference input's motion"), but "every move" over-specifies.
  Name the facing and the main slides, not every flick.
- **"Do not add camera words for a template."** Training captions always state the shot type, the viewpoint and "the
  camera remains static". Recommendation: state the camera exactly as the layout shows it, never invent a move.
- **"An action inside the 25 shared frames may be named in both clips."** Change to: an action in clip k-1, a state in
  clip k (risk: a second kick or catch).
- **Clip 1 of `shan_shen_bu_balls`** ("Three times a dodgeball flies straight at the head; each time the dancer slides
  sideways at the last moment ...") is the causal event style that rule 5 advises against.
- Agrees with 7.2: shared vs clip split, the motion-style sentence (an official pacing lever), facing words, picture
  directions, the "state not action" rule for later clips, and no camera moves for a template.

### 5.3 Template
```
--video (shared, every clip; no single-clip actions):
  "<Style words>. A <shot type>, front-facing, the camera remains static: <character's 2-3 marks> in <place>,
   <one light logic>. {character} faces the camera the whole time and <motion style: how the body moves, with
   pacing words>."
--clip-video 1  "<first beat with direction and stop>, then <second beat>, then <third beat>."
--clip-video k  "Still <state visible in the shared frames: pose, held prop, facing>, {character} <new beat>, then <next beat>."
(sound sentence and props sentence: added by the code as today)
```

Worked example (`shan_shen_bu_balls`, masked3d; clips = frames 1-121, 97-217, 193-276):
- Shared: "3D animated style. A wide shot, front-facing, the camera remains static: a masked dancer with a white oval
  mask, a short dark purple jacket and a black bodysuit in a dim school gym at night, lit by dim violet light.
  {character} faces the camera the whole time and dances the flash step: quick sideways slides with hard stops, small
  fast steps, knee lifts and foot flicks, loose arm swings."
- Clip 1: "{character} slides toward the right of the picture, stops hard, slides back toward the left, stops, slides
  right again, then flicks the left foot out to the side." (The three passing balls are left to the layout and the
  props sentence.)
- Clip 2: "Still facing the camera, {character} flicks the right foot out to the side, then claps both hands high above
  the head twice with small bouncy steps in place." If the left-foot flick ends inside frames 97-121, start with "Still
  facing the camera, with the left foot coming back down, ...".
- Clip 3: "Still facing the camera with the hands coming down from the clap, {character} raises both hands once more,
  then settles into a ta-da pose and holds it: the right arm up on the left side of the picture, the left hand on the
  hip, the head tilted; the right hand closes around a violet ball." (The catch stays: it is the end state the critic
  checks.)

### 5.4 What to avoid
- Event stories with causes or threats.
- Words for the previous clip's actions.
- Timestamps, shot lists, numbered beats.
- Camera moves the layout lacks.
- "clay", "mannequin", "3D render of".
- Pronoun or synonym swaps for the character (keep `{character}` words fixed).
- Intent or emotion words.
- More than about three beats per clip.
- A different place or light sentence per clip.

## 6. A/B plan (cheap, preview, clip 1 only, then one join)

Setup:
- `run new --blend templates/library/shan_shen_bu_balls/shot.blend --character masked3d --quality preview --seed N
  --note "prompt A/B arm X"` with `--video`, `--clip-context ""` and `--clip-video 1` set per arm;
- `run start RUN --through-clip 1 --detach`;
- 2 seeds x 4 arms = 8 preview clips through the queue.

| Arm | Shared `--video` | Clip 1 text |
|---|---|---|
| A (today) | marks + place + light | template clip_context (motion style) + today's clip-1 events incl. balls at the head |
| B (official minimum) | style + shot/viewpoint/static camera + marks + place + light + "faces the camera the whole time" | none |
| C (B + pacing) | B + the motion-style sentence | none |
| D (proposal) | C | the 5.3 clip-1 body-mechanics sentence, no balls |

Measure each clip:
- `run score RUN --clip 1 --critic DIR --tag <anonymous>`;
- DWPose motion (arm score, `motion.mirrored`, `motion.rows`), `overlap.x_median`, `picture.max_jump_x_median`;
- a fresh blind critic with docs/gauntlet_critic.md, asked specifically "does the body stay frontal during the
  sideways moves; do the stops read; any profile walk";
- keep the tag map outside DIR.

How to decide:
- **The winner** must lead on critic MOTION on both seeds without losing CHARACTER.
- **If B ≈ D:** the action text adds nothing, so shorten 7.2.
- **If A < D:** the threat story hurts, so drop it from the template.

Then one join test from the winning arm (`run fork`, clip 2):
- E = clip 2 opens with the shared-frame state;
- F = clip 2 names the clip-1 action that overlaps as a verb;
- the critic looks for a repeated kick at frames 97-121, and `setup/join_check.py` runs on the join.

Record everything in `.cinestudio/gauntlet/prompt_ab/RESULTS.md`.
Untested proposals in this report: every recommendation in 5.1-5.3.
