# Prop control in layout runs (research, 2026-10-26)

How to make props (thrown, kicked and caught balls; held objects) come out right in our LTX-2.5 layout runs, and how to
stop the giant-ball copy. No GPU jobs were run for this report. It reads our code, our frozen graphs, our finished
runs, the installed ComfyUI and Lightricks sources, and papers.

Tags:
- [OFFICIAL]: Lightricks model cards, docs.ltx.io, the LTX-2 and ComfyUI-LTXVideo repos.
- [CODE]: our code, the frozen graphs in our runs, the installed ComfyUI code, and frames of our own finished runs
  (named by run ID; I looked at the frames).
- [COMMUNITY]: third-party guides and posts.
- [RESEARCH]: papers.
- [INFERENCE]: my reasoning; not tested.

The look image is covered in docs/look_image_research.md (another report). This report touches it only where props
need it.

## Short answer: the 3 best levers for our props

1. **Make the props readable in the LAYOUT, not in the sheet.** LTX draws what the layout shows, at the layout's size.
   One latent token is 32x32 px and 8 frames [RESEARCH LTX-Video; CODE]. Our dodge balls are about 25 px at preview
   (0.8 of a token) and move about 14 px a frame [CODE shot_check facts]. Aim at 48-64 px for a prop that matters
   (1.5-2 tokens), keep it in view for at least 16-24 frames (2-3 latent frames), and give it a clear value contrast
   against the background behind its path. Cheat the size up (1.3-1.5x) or bring the path nearer the camera. Evidence:
   in our runs the 50-60 px kicked and caught balls were drawn as balls; the 25 px dodge balls came out as small dots;
   the fast bounce in clip 3 was lost (run r20261026-035007-8d6b) [CODE].
2. **Small props get WORDS only, and short words.** Keep the prop panel off for props under ~2 tokens on screen (today's
   default). Name them with 2-4 concrete words that can show at that size ("violet rubber dodgeballs"). Drop sub-token
   details ("a thin magenta seam", "matte"). Lightricks: "Match the level of detail to the shot scale" [OFFICIAL].
   Never tell a threat story for objects the layout already moves (docs/multiclip_prompting_research.md) [CODE].
3. **If a prop panel is used (large or held props): weaken it where it can be copied.** The giant ball is a POSITION
   effect. The sheet enters as a 121-frame guide whose tokens sit on the SAME (time, x, y) positions as the video
   [OFFICIAL LTX-2 code; CODE ComfyUI]. So the bottom-right panel pulls a ball into the bottom-right of the frame. The
   core `LTXVAddGuide` node already has an `attention_mask` input: scale the prop panel to ~0.3-0.5 (or switch it on only
   in the frames where the layout shows the prop). Also draw the prop smaller and from 2-3 angles, as Lightricks'
   example sheets do. All of this is untested: A/B it first (section 8).

## 1. How the sheet and the layout reach the model (the mechanism)

**Our graph** (frozen graph of run r20261026-035007-8d6b, work/graphs/clip3.json; workflows/ltx25_layouting_api.json)
[CODE]:
- The sheet (`ref_load`) is fitted to 768x448 with black padding (`ResizeAndPadImage`), repeated 121 times
  (`RepeatImageBatch`) and added by `LTXVAddGuide` at `frame_idx` 0 with the Ingredients IC-LoRA's parameters.
- The layout render is added by `LTXAddVideoICLoRAGuide` at frame 0; the look image at frame -1.
- Stage 2 (final only) gets the layout and the look again, but NOT the sheet: `LTXVCropGuides` drops the stage-1
  guides. So the sheet acts in stage 1 only, and a preview run shows any sheet copy. [CODE]
- The same pattern is the official one: "The sheet is resized once and then held on every frame as the IC-LoRA guide"
  (ComfyUI-LTXVideo, LTX-2.5_ICLoRA_Ingredients_Single_Stage_Distilled.json, Inputs note) [OFFICIAL].

**Where the sheet's tokens sit.**
- ComfyUI `LTXVAddGuide.add_keyframe_index` gives each guide token the pixel coordinates of its own latent grid, plus
  `frame_idx` on the time axis. The guide was resized to the target's size, so its grid is the target's grid
  (comfy_extras/nodes_lt.py, lines 303-376) [CODE].
- Lightricks' own code does the same: `reference_video_cond.py` will "Append reference video tokens with positions
  translated into the target's frame", and `downscale_factor` "scales reference positions to match target coordinates,
  preserving the learned positional relationships" (LTX-2 repo, packages/ltx-core, main branch, checked 2026-10-26;
  an earlier quote here, "no spatial offset is added", is not in the current file: docs/novelty_research.md)
  [OFFICIAL]. `keyframe_cond.py` moves only the time axis by `frame_idx`.
- So the sheet's bottom-right panel at time t has the same rotary position as the video's bottom-right at time t
  [CODE]. Attention between tokens at the same position is favoured [INFERENCE from how rotary position encodings
  work].

**What research says about shared positions.**
- OminiControl: shared position indices "work well for spatially aligned tasks like edge-guided generation"; for
  subject-driven (non-aligned) tasks "shared position indexing can lead to suboptimal performance due to spatial
  overlap"; an offset made training converge faster and the result better [RESEARCH].
- UNO: reference positions are offset from the noisy image ("UnoPE"), which keeps the model from copying the
  reference's spatial layout; removing it dropped DINO from 0.542 to 0.386 on its multi-subject test [RESEARCH].
- Layout-To-Render NEEDS shared positions (it copies placement on purpose). Ingredients was trained with them too, so
  it learned to not copy most of the time [INFERENCE from the OFFICIAL code above].

**What our runs show** [CODE]:
- The run's sheet (r20261026-035007-8d6b/inputs/sheet_props.png, 1536x896) has the ball at x 1145-1512, y 484-853:
  the right quarter, lower half, 41% of the height. At 768x448 that is ~185 px, about 6x6 tokens.
- In clip 3 of that run (distilled, final, seed 303) a large static violet ball sits at the right edge at ~70% of the
  height, from the first frame of clip 3 to the end, never moving. The layout has no ball there (source_3.mp4). It is
  ~70 px wide at 768 px, about 3x the layout's dodge balls. Same quarter as the panel.
- With the dev model the panel was copied "as a giant ball in the bottom-right corner in clips 1 and 3" (AGENTS.md 5,
  run r20261025-060521-d2d6, no longer on disk). Always the panel's corner.
- The two no-panel finals with seed 202 (r20261025-130232-5475, r20261025-190638-f318) have no giant ball. The paired
  no-panel run with seed 303 (r20261026-035011-a791) was still running clip 2 when I wrote this: its clip 3 is a free
  control. Look at it when it is done.
- [INFERENCE] The panel tells the model "a big violet ball is part of this scene". The layout shows only tiny balls.
  At the panel's position nothing in the layout contradicts a ball strongly (floor and wall), so in some seeds the
  model "satisfies" the sheet there. In training a prop that is large in the sheet was probably also visible in the
  video ("video clips paired with generated reference sheets" [OFFICIAL]); a 7x size mismatch is out of distribution.

## 2. What Lightricks documents (question 1)

**Ingredients 2.5 card** (huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients) [OFFICIAL]:
- The sheet is "a single composite image inventorying the characters, props, and location of a scene".
- One clean panel per element, on black, no text. Characters: face close-up + turnaround. Props: "product-style"
  renders. One clean location panel.
- "The more space an element takes up in the reference image, the more faithfully it carries over." Add a dedicated
  panel for any prop that must persist and describe it in the prompt.
- Input: the sheet as a static video at the output resolution (downscale factor 1), at least 121 frames.
- Prompt: "Reference sheet: ..." then "Generated video: ...". The examples use the headings "Reference Sheet
  Description" and "Target Description".
- Trained at 768x448, 121 frames, 24 fps; other sizes are out of distribution. LoRA strength 1.0 (weights
  pre-scaled). Rank 128, 12,000 steps. Data: "a proprietary dataset of video clips paired with generated reference
  sheets".
- Nothing about panels being copied into the frame. Nothing about a separate prop reference: everything is one sheet.

**The card's six example sheets** (as the card describes them) [OFFICIAL]:
- Sheets are GRIDS of rows (Top / Middle / Bottom Row, Left / Right). Props get their own cells, often from several
  angles: "a soda can shown from three angles", "a black game controller from top, front, and angled views", "a white
  hatchback from several angles".
- Labels in the text: "Top Row Right ... Prop - Backpack", "Middle Row Left ... Prop - Walking Stick".
- In the target the props appear by ACTION and place: "grips the staff in his right hand, holds the soda can in his
  left", "grips the controller with both hands", "the backpack leans against the stone shrine". Sheet items not
  needed are simply not mentioned in the target.
- The example props are hand-held or large (controller, can, staff, backpack, hose reel). None is a small fast object.

**ComfyUI-LTXVideo example workflow and README** [OFFICIAL]:
- "Describe them in the prompt and refer to them by their position on the sheet, so the model knows which element you
  mean."
- The Ingredients example builds the empty latent "at the sheet's resized size": sheet and video share one grid.

**Layout-To-Render card and docs** [OFFICIAL]:
- It "keeps the camera path and the placement of objects from the layout", "including the movement of simple blocking
  shapes".
- A layout can be "a grey clay viewport, or a blocky playblast of simple shapes"; "a detailed render is not required".
- The art-directed first frame "sets palette, lighting, and materials".
- Prompt: "one or two sentences describing the finished shot", "match the first frame"; avoid "clay", "3D", "Unreal".
- Nothing on prop size, colour, texture, motion blur or small objects.

**docs.ltx.io IC-LoRA guide** [OFFICIAL]:
- `attention_strength` (0-1) and an optional `attention_mask`, spatial (HxW) or spatiotemporal (TxHxW), values 0 to
  1: attention scores "are scaled by attention_strength x attention_mask".
- In ComfyUI core, `LTXVAddGuide` has the `attention_mask` input; ComfyUI-LTXVideo has
  `LTXAddVideoICLoRAGuideAdvanced` with `attention_strength` + `attention_mask` [CODE]. The model applies it as a log
  weight on the guide tokens (comfy/ldm/lightricks/model.py, `GuideAttentionMask`); a weight of 0 blocks those tokens
  [CODE].

**LTX prompting guide** [OFFICIAL]: "Match the level of detail to the shot scale (close-ups need more detail than wide
shots)"; "reuse the same visual identifiers for recurring people or objects"; "simpler, more plausible motion is more
reliable".

## 3. What research and the community do (question 2)

Ranked by use to us.

1. **Copy-paste is a known failure of reference conditioning.** Phantom-Data: the "copy-paste problem ... arises from
   the widely used in-pair training paradigm" (reference and target from the same scene) [RESEARCH]. UNO: training on
   crops "may introduce copy-paste issues"; the model then "pastes" the reference [RESEARCH]. MAGREF names
   "copy-paste artifacts" and uses region-aware masking [RESEARCH]. Fixes in papers: cross-pair data, position
   offsets, region masks. We cannot retrain; region masks (the attention mask) and our sheet design are what we can
   use [INFERENCE].
2. **Position offsets against copying** (OminiControl, UNO, above) [RESEARCH]. For us this would need a custom node that
   shifts the sheet's positions, and the Ingredients LoRA was trained WITHOUT a shift [OFFICIAL]. Out of distribution;
   research only (section 7).
3. **Box and trajectory control** for object motion: Boximator (hard and soft boxes as a plug-in; better box
   alignment) [RESEARCH]; Lightricks' own LTX-2.5 Motion Track IC-LoRA (sparse tracks drawn on the first frame)
   [OFFICIAL]. Our layout already carries the path of every ball, so a second motion control adds little and stacking
   two IC-LoRAs with Ingredients is untested [INFERENCE].
4. **Multi-subject references:** SkyReels-A2 (characters, objects and backgrounds composed into one video) and MAGREF
   aim at "balancing element-specific consistency with global coherence" [RESEARCH]. They are other models (not LTX);
   check licences before any use. Wan VACE (Apache-2.0) does reference + control video, but would replace LTX and its
   audio (docs/style_consistency_research.md 2.9) [CODE].
5. **Small objects:** I found no paper that measures small-object failure in video diffusion. The cause is clear from
   the latent: LTX-Video compresses "32 x 32 x 8 pixels per token" [RESEARCH]; our stage 1 is 768x448 = a 24x14 token
   grid [CODE runner.SIZES]. A 25 px ball is less than one token; at 14 px a frame it moves ~112 px (3.5 tokens) within
   one 8-frame latent step. The model sees a faint smear, not a ball [INFERENCE]. The decoder adds fine detail, but only
   what the latent implies [RESEARCH LTX-Video].
6. **Community practice for Ingredients** (aistudynow guide for 2.3): describe every panel in the sheet text; repeat the
   key outfit words in the "Generated video" part [COMMUNITY]. Nothing on props or copying. I found no public report of
   a sheet panel copied into the frame [COMMUNITY search, none found].

## 4. Layout-side levers (question 3)

**Our template's numbers** (shan_shen_bu_balls, from the cached shot_check facts, at preview 768x448) [CODE]:

| Prop | Frames in view | Size (median, max) | Speed px/frame (median, p90) | Seen in run 8d6b |
| --- | --- | --- | --- | --- |
| Ball_Dodge1-3 | 29-32 | 25 px (up to 124 near the camera) | 14-15, 24-29 | small dots near the head |
| Ball_Kick1-2 | 12 | 63 px | 42, 82 | drawn at the foot (clip 2) |
| Ball_Catch | 25 | 51 px | 0.3 (held), 68 (bounce) | caught ball drawn in the hand; the fast bounce lost |

The template check warns only below 12 px (shot_check.PROP_MIN_PX) [CODE], which is a third of a token.

Ranked levers:

1. **Size on screen.** A prop needs about 1 token (32 px) to exist and 1.5-2 tokens (48-64 px) to keep its shape and
   colour [INFERENCE from the token size + the table]. How: scale the prop up 1.3-1.5x (a 22 cm ball as 30-33 cm still
   reads as a dodgeball), or move its path nearer the camera, or a tighter lens. At 4 m, 1 px ~ 6 mm at preview
   (AGENTS.md 5) [CODE]. Spend the size on the HERO props (kicked, caught, held); a passing ball may stay small.
   Risk: a ball that looks too big next to the head. Cost: Blender only.
   Proposed rule change: `prop_size` warning below 32 px, info below 48 px.
2. **Time in view.** The kicked balls are visible 12 frames: 1.5 latent frames [CODE]. Keep a hero prop in view at least
   16-24 frames (a flatter or slower path, a bounce before it leaves) [INFERENCE]. Fast moves become one smear inside
   one latent step.
3. **Value and colour contrast against what is behind it.** The layout render includes the backplate. A deep violet
   ball over a dim violet wall has little contrast [CODE: the gym_noir plate; INFERENCE]. Give the Blender prop a
   lighter value than the wall behind its path (the source render already shows the balls lighter, lavender), or put the
   path in front of a lighter area. Keep the main colour (a grey prop merges with the body; TEMPLATE_GUIDE 5) [CODE].
4. **Shape, not texture.** A seam or pebble texture is far below one token at preview and does not survive [INFERENCE].
   Only for a prop of 64 px or more, a large two-tone pattern (half and half) can show its spin. Layout-To-Render
   accepts "a blocky playblast of simple shapes" [OFFICIAL]: the outline matters, not surface detail.
5. **No motion blur in the layout.** Workbench renders none, and the training layouts were viewport playblasts
   [OFFICIAL wording]. A blurred streak would lower contrast and widen the outline; LTX adds its own blur
   [INFERENCE].
6. **Contact and ground.** For a bounce or a catch, a contact shadow helps the floor contact read (Workbench has a
   shadow option) [INFERENCE]. Keep loose copies of the prop out of the play area (AGENTS.md 6) [CODE].
7. **The look image and keyframe looks** carry the prop's palette only where the prop is in that frame. Frame 1 of our
   template has no ball. See docs/look_image_research.md.

When to make a prop bigger than in reality: when it is under ~1.5 tokens at preview and the shot needs it to read
(it is touched, caught, or the gag depends on it). When its real size is part of the joke, move the camera or the
path instead [INFERENCE].

## 5. Text (question 4)

1. **Short, visible words, the same everywhere.** "violet rubber dodgeballs". Drop details that cannot show at that
   size ("thin magenta seam", "matte", "pebbled") [OFFICIAL "match the level of detail to the shot scale";
   INFERENCE]. Use the same words in the background, look and video texts (AGENTS.md 6) [CODE] and for every recurring
   object [OFFICIAL multishot guidance].
2. **Props by their look in the shared context; by an ACTION only where a character touches them, in that clip.** The
   Ingredients examples write "grips the controller", "holds the soda can" [OFFICIAL]. Our clip 3 text "catches and
   holds it" fits. Clip 2's "kicks a second dodgeball away with the right foot" fits [CODE shot.json].
3. **No cause or threat stories** for objects the layout already moves ("a ball flies at the head; the dancer dodges").
   They pulled the dancer into a profile walk (docs/multiclip_prompting_research.md) [CODE]. At most "a ball passes
   beside the head" [CODE AGENTS.md 7.2].
4. **No size-loaded words**: "giant", "big", "close-up of the ball" [INFERENCE]. A relative size can help: "small violet
   balls about the size of the dancer's head" (untested) [INFERENCE].
5. **Counts:** say a count only when every clip shows it; the layout carries the count [INFERENCE].
6. **Avoid sheet words in the target:** "panel", "reference", "prop". Keep "shown in the bottom-right panel" in the
   Reference Sheet Description only (the official way to point at a panel) [OFFICIAL]. Whether the words
   "bottom-right" also pull the copy to that corner is an open question: test it (arm T2 below) [INFERENCE].
7. **Never** "clay", "3D", "Unreal", "mannequin" [OFFICIAL Layout-To-Render; CODE AGENTS.md 7.2].

## 6. Fixes for the giant-ball copy (question 5), ranked

| # | Fix | How | Evidence | Risks | Cost |
| --- | --- | --- | --- | --- | --- |
| 1 | No panel for small props | `run new` without `--prop-image` (today's default); props under ~2 tokens on screen | No giant ball in any no-panel run we have; panel vs words gave the same 20-35 px balls (AGENTS.md 5 A/B) [CODE] | No look transfer for the prop (none visible at that size anyway) | 0 |
| 2 | Weaken the prop panel by an attention mask | Write a mask the size of the run's sheet: 1.0 everywhere, 0.3-0.5 in `prop_layout.prop_boxes` (padded by one token); load it (`LoadImageMask`) into `ref_guide.attention_mask` (core `LTXVAddGuide`). Register it as a graph feature for both model members (AGENTS.md 11) | Official mechanism for per-region guide strength [OFFICIAL docs.ltx.io; CODE nodes_lt.py]; region masks against copy-paste in MAGREF [RESEARCH] | Weaker prop look too; masked attention calls run without SageAttention (`low_precision_attention=False`) and keep a (guide tokens x all tokens) mask, ~0.3-0.4 GB at preview [CODE model.py; estimate]; untested | small code change + A/B |
| 2b | Same, spatiotemporal | A 121-frame mask: the prop panel at 1.0 only in frames where the layout shows that prop (boxes from shot_check facts), else 0-0.3 | As 2. Our copy was static from the first frame of clip 3 [CODE] | A copy may still appear while the prop flies | medium |
| 3 | Smaller panel, several views | `PROP_CELL_FILL` 0.86 -> ~0.45, and 2-3 small views (front, side, 3/4) in the cell, like the official "three angles" sheets | Official example sheets [OFFICIAL]; a smaller single blob is a weaker target for a copy [INFERENCE] | "Bigger panels carry over better" [OFFICIAL]: less look transfer | small (charsheet.PropColumn) |
| 4 | Move the panel (diagnostic first) | Put the prop cell top-right or left; if the giant ball follows the panel, the position mechanism is confirmed | Section 1 [CODE + INFERENCE] | A copy elsewhere is still a copy; a per-clip panel place (where the layout's prop really is) is possible later but complex | small |
| 5 | Sheet sentence without the position, or with a size cue | "**Prop:** violet rubber dodgeball (small in the video)" | Official advice is to name the position [OFFICIAL]; size cue untested [INFERENCE] | May confuse which panel is the prop | 0 |
| 6 | Automatic check + retake | After each clip: find ball-coloured blobs outside the layout's prop boxes (dilated) and outside the figure mask, lasting >= 8 frames; on a hit, retake with the next seed | Seed decides a lot (seed 202 had no copy, 303 had one) [CODE]; the gauntlet already retakes by seed (AGENTS.md 7.5) | False hits on the violet jacket (use the figure mask); costs a retake | medium |
| 7 | Lower the Ingredients strength | LoRA or guide strength < 1 | None for props | Hurts the character's identity for every panel | 0, not advised |
| 8 | A separate prop guide | A second `LTXVAddGuide` with the prop alone | None; the LoRA was trained on one sheet [OFFICIAL] | Out of distribution; adds as many tokens as the video (121 frames) on 8 GB [CODE] | high, not advised |

Recommended order: keep 1 as the default; build 2 and test it with 3 and 4 for LARGE props; add 6 as a safety net
for any run that uses a panel.

## 7. Options considered and set aside

- **Shift the sheet's positions** (OminiControl / UNO style) with a custom node. Strong evidence against copying in
  those models [RESEARCH], but Ingredients learned with shared positions [OFFICIAL]; identity transfer may collapse
  (UNO's ablation shows how much positions matter) [RESEARCH]. A custom node is also a maintenance cost. Research only.
- **Put the sheet before the clip in time** (like the look image at frame -1): also out of distribution for
  Ingredients [INFERENCE].
- **Motion Track IC-LoRA** for the balls' paths: the layout already gives the paths; three stacked IC-LoRAs on 8 GB is
  untested [OFFICIAL exists; INFERENCE].
- **Another video model** (Wan VACE, SkyReels-A2, MAGREF): loses LTX's audio and our pipeline; check licences
  [CODE; RESEARCH].

## 8. A/B plan

Template `shan_shen_bu_balls`, character `masked`, background `gym_noir_comfyui`, distilled model, PREVIEW (stage 1
is where the sheet acts: section 1). Paired seeds 202 / 303 / 404 (303 copied the ball in clip 3). Same look image per
seed. One change per arm. Notes on every run (`--note`).

**Round 1: the copy (sheet arms).** Clip 3 shows the copy; clip 1 did with the dev model.
- Clip 3 cheaply: make one no-panel base run per seed through all clips, then `run fork BASE --from-clip 3` per arm
  (keeps clips 1-2, changes only the sheet) [CODE AGENTS.md 7.5]. Clip 1: `--through-clip 1` per arm.
- Arms: P0 words only (control); P1 panel as today; P2 panel + attention mask 0.4; P3 panel moved to top-right
  (diagnostic); P4 small 3-view panel. Optional T2: P1 without "bottom-right" in the sheet sentence.
- Cost: about 5 arms x 3 seeds x 2 clips ~ 30 preview clips at ~2-3 min each, plus the look images of the base runs
  (~4 min each) [CODE AGENTS.md 7.1/7.3 timings]; about 1.5-2 GPU hours.

**Round 2: the layout (props readable).** Words only. Full 3-clip preview runs.
- L0 today's balls; L1 hero balls (kick, catch) x1.4 and kept in view >= 20 frames; L2 = L1 + a lighter ball value
  against the wall. 3 seeds each: 9 runs.
- Cost: Blender edits of the template copy (add_balls.py options), then ~9 x 3 clips.

**What to measure** (per clip, per seed):
- **Phantom props (the giant ball):** ball-coloured blobs (hue/value from the layout's prop colour, measured on the
  run's frames with `video.frame_png`) outside the dilated layout prop boxes (shot_check facts, per frame) and outside
  the figure mask (cinestudio/figure_mask.py); count blobs >= 32 px lasting >= 8 frames. Report the frames and the
  place. Goal: 0.
- **Prop recall:** share of frames where a layout ball is in view AND the output has ball-coloured pixels in its box
  (dilated by one ball size). Split by hero balls and passing balls.
- **Prop count and place:** blobs per frame vs the layout's count; centre distance to the layout box in px.
- **Prop look:** crops of the caught and kicked balls at their largest frame; a blind critic (docs/gauntlet_critic.md)
  scores "reads as a violet rubber dodgeball" 0-10. A/B tags hidden.
- **The dancer's motion:** `run score` (DWPose bone score, mirrored, overlap), the turn-away frames (back to camera),
  and the critic's MOTION and CHARACTER scores. A prop arm must not cost motion or identity.
- Look at every strip yourself before any claim (AGENTS.md 2).

**Decision rules.**
- P1 vs P0: keep the panel off unless it raises hero-prop recall by >= 10 points with 0 phantoms in all 3 seeds.
- P2/P4: adopt for large props only if phantoms are 0 in all seeds AND the prop look score is >= P0's + 1.
- P3: if the giant ball moves with the panel, record the position mechanism in AGENTS.md as measured.
- L1/L2: adopt if hero-prop recall rises and the critic's MOTION score does not fall; then raise the `prop_size` rule.

## 9. Sources

- Lightricks, LTX-2.5 Ingredients card: https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients
- Lightricks, LTX-2.3 Ingredients card: https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients
- Lightricks, Layout-To-Render card: https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render
- docs.ltx.io Layout to Render: https://docs.ltx.io/open-source-model/vfx-post-production/layout-to-render.md
- docs.ltx.io IC-LoRA guide (attention strength and mask): https://docs.ltx.io/open-source-model/usage-guides/ic-lo-ra
- docs.ltx.io IC-LoRA adapters: https://docs.ltx.io/open-source-model/integration-tools/ic-lo-ra-adapters
- docs.ltx.io prompting guide: https://docs.ltx.io/open-source-model/usage-guides/prompting-guide
- LTX-2 repo, reference video conditioning:
  https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/conditioning/types/reference_video_cond.py
- ComfyUI-LTXVideo example workflows (installed): custom_nodes/ComfyUI-LTXVideo/example_workflows/2.5/README.md and
  LTX-2.5_ICLoRA_Ingredients_Single_Stage_Distilled.json; iclora.py (`LTXAddVideoICLoRAGuideAdvanced`)
- ComfyUI core (installed): comfy_extras/nodes_lt.py (`LTXVAddGuide`), comfy/ldm/lightricks/model.py
  (`GuideAttentionMask`)
- LTX-Video paper (VAE 32x32x8): https://arxiv.org/abs/2501.00103
- OminiControl (position offset for non-aligned conditions): https://arxiv.org/html/2411.15098
- UNO / UnoPE (copy-paste, multi-subject): https://arxiv.org/html/2504.02160
- Phantom and Phantom-Data (copy-paste from in-pair data): https://arxiv.org/abs/2502.11079,
  https://arxiv.org/html/2506.18851v1
- MAGREF (region-aware masking, copy-paste artifacts): https://arxiv.org/abs/2505.23742
- SkyReels-A2 (elements to video): https://arxiv.org/abs/2504.02436
- OpenS2V-Nexus (subject-to-video metrics): https://arxiv.org/abs/2505.20292
- Boximator (box control): https://arxiv.org/abs/2402.01566
- aistudynow, LTX 2.3 Ingredients guide: https://aistudynow.com/how-to-keep-characters-consistent-in-ltx-2-3-ic-lora-ingredients/
- Ours: cinestudio/runner.py (`prop_sheet`, `prop_sheet_sentence`, `props_sentence`, `SIZES`), cinestudio/charsheet.py
  (`PropColumn`, `PROP_CELL_FILL`), cinestudio/props.py, cinestudio/shot_check.py (`rule_prop_size`),
  templates/library/shan_shen_bu_balls/shot.json, runs r20261026-035007-8d6b (panel), r20261026-035011-a791 (no panel,
  running), r20261025-130232-5475 and r20261025-190638-f318 (no panel), AGENTS.md 5 and 7,
  docs/multiclip_prompting_research.md, docs/style_consistency_research.md 2.7.
