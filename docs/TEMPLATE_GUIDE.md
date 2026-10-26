# TEMPLATE_GUIDE.md — authoring a Blender template that makes good cinestudio videos

For a person in Blender or an agent writing bpy code. The rules in brief are in AGENTS.md section 5; this is the full
checklist with the numbers behind it. A built example to copy: `shan_shen_bu_balls` (section 10).

## 1. What a template is and how a run uses it

A template is a .blend with grey clay geometry and motion: one or more mannequins, a camera, the props, a static set
and (optionally) sound. It has no style (user rule): the characters, the background and the video text bring the look.
A run renders the shot (EEVEE) as the LAYOUT guide for LTX-2.5: LTX keeps the placement of objects and follows their
OUTLINES; text alone steers little. Frame 1 becomes the LOOK image (a Qwen edit: mannequin -> the character) that sets
palette and materials; the character sheet rides along through the Ingredients IC-LoRA. The Set layer is hidden and
replaced by the background picture (a backplate projected on a floor + far wall, or a 360 panorama); Props stay in the
layout. A shot longer than one clip is cut into 121-frame clips that share 25 frames, each with its own prompt.

| Item | Value |
|---|---|
| Frame rate | 24 fps; the template's own resolution 1536x896 (16:9-ish, what the run renders) |
| Preview / final video | 768x448 (stage 1 only, for tests) / 1536x896 |
| Clip k | frames 96(k-1)+1 .. 96(k-1)+121; clip 1 = 1-121, clip 2 = 97-217, clip 3 = 193-313 |
| Shared frames (join windows) | 97-121, 193-217, 289-313, ... (25 frames each); the last clip is clamped to the shot |
| Lengths that fill whole clips | 121, 217, 313 frames (5.0, 9.0, 13.0 s); shan_shen_bu_balls: 276 frames, clip 3 = 193-276 |
| What a run takes from shot.json | layer words (`--set`, `--prop`), `clip_texts`, `clip_context`, `background`, `timeline` |

## 2. Scene setup checklist

- **Units:** metric, scale 1.0, metres. The mannequin faces -Y, its right side is -X, the floor is z = 0. A still front
  camera stands at -Y looking along +Y (screen right = +X).
- **Frames:** `frame_start = 1`, `fps = 24`, resolution 1536x896 at 100%. Plan the length in clips (section 1).
- **Collections (the layers; `bpy_jobs/scan_layers.py` reads them):**
  - `Mannequin`, `Mannequin2`, ... `Mannequin4` with the rigs `MannequinRig`, `MannequinRig2`, ... (at most 4 characters
    per run). Slot order is the run's `--character` order; templates from video number them LEFT TO RIGHT.
  - `Props`: every object that moves or that a character touches. NEVER put a prop in `Mannequin` (the character mask
    and the proxy kit read that collection).
  - `Set`: the static place (floor, walls, furniture). The template's floor plane belongs in it.
  - Nothing else at the top level but the camera, its target empty and the lights (`scan` lists anything else as
    "loose" and warns).
- **Rig:** build it with `blender_lib/mannequin.py` (`build(materials)`, then `key_pose(rig, frame, dirs, yaws, palms,
  hands)`, `set_bone_dir`, `curl_fingers`) or start from a generator (`templates/flash_step/build.py` for motion files,
  `templates/funk_dance/build.py` for beat dances). The cinestudio bone names are required (hips, spine, chest, neck,
  head; shoulder, upper_arm, forearm, hand, thigh, shin, foot, toe .L/.R; fingers thumb..pinky_01..03): the run scales
  the rig to each character's body (bpy_jobs/mannequin_body.py) and reads keypoints from it. Rotations only:
  **every pose bone's scale must stay 1** (a leak once made blade-like hands; the character body sets its own scales).
- **A real 3D character instead of the mannequin:** `bpy_jobs/swap_character.py` bakes the mannequin's motion onto a
  rig with the cinestudio names (meshes into `Mannequin`, the mannequin's into the hidden `MannequinHidden`).
- **Lights:** the three area lights of `templates/character_showcase/build.py` (Key (-2.5, -3, 3.2) 450 W from the front
  left, Fill 150 W, Rim 300 W), neutral colours. Light the figure from the front: a backlit place gave hazy silhouettes.
- **Materials:** set both the Principled Base Color (the EEVEE layout render) and `material.diffuse_color` (the Workbench
  preview), as gym_set.py does. In bpy, find shader nodes by type, never by name.

## 3. Camera

- **Still camera = the default.** A backplate (4-9x sharper than a panorama: one picture at the frame's shape) and
  `background from-set` need a still camera; a single key on a still camera is not a move. A moving camera needs a 360
  panorama.
- **Framing (measured on the built-ins):** 30 mm lens, 36 mm sensor, camera at (0, -4, 1.45) with a Track-To on an empty
  (pitch -7.5 deg; horizon at 0.31 of the frame from the top). shan_shen_bu_balls' scan at frame 1: head top at 0.20,
  feet at 0.81 of the frame height (a 1.75 m mannequin ~ 61% of the height: ~275 px at preview). Keep the whole body,
  feet included, inside the frame for every frame of the motion (sideways dashes too).
- **Speed:** keep the camera under ~0.6 deg/frame about the subject (a 90 deg arc needs >= 150 frames); an orbit backup
  at 1.22 deg/frame failed. Start an orbit at the FRONT. In orbits the body turns ~30 deg ahead of the camera (a known
  limit); more clips are better than a faster move.
- Camera words in the clip context say only what the template's camera does (a still camera: "the camera remains
  static"); never a move the layout lacks (AGENTS.md 7.2).

## 4. Motion rules (per clip)

- **Frame 1 = a clean front pose:** the body faces the camera, the whole figure visible. The
  look image is a Qwen edit of frame 1 (Qwen cannot draw a back view).
- **Each action inside ONE clip and outside the 25 shared frames.** A move across a join must agree in two clips; put a
  held pose or a freeze across each join instead. (`shot check` flags the built-in's clap above the head at 209-222
  across the window 193-217: rule `shared_frames`, for hand raises, kicks, jumps, fast turns and hips dashes.)
- **The head follows the body.** Start a wave or a look at the camera only while the body faces the camera within ~30
  deg: an early head turn made LTX turn the body early and step.
- **End on a held stance:** ease into the end pose over >= 24 frames, then hold (shan_shen_bu_balls holds 0.5 s; `shot check`
  rule `end_hold`: < 5 mm/frame in the last 12 frames). Without it LTX carries fast steps on to the last frame.
- **Moves must read from the front camera:** moves toward the camera do not read. A kick must go sideways; a hidden kick
  is made readable with `mocap.accent` (lift and turn the shin outward; check the ankle acceleration); a knee pop must go
  INWARD. A visible bounce needs ~8 cm of hip drop. End poses: `mocap.settle` + `pose_frame(arms="rest"|"crossed"|
  "tada", yaw_deg, ...)`; keep end turns small (an arms-crossed end turned 30 deg outward made LTX turn the characters
  to profile and back).
- **Fast sideways dashes (open problem, 2026-10-25):** at the dodges LTX turned the body to profile and walked. Keep the
  body frontal (shoulders square to the camera), avoid a big hip translation in few frames where you can, and say
  "faces the camera, slides sideways without turning" in the shared clip context (section 8). Not solved yet.
- **No pops, no foot slide:** look at joint acceleration peaks (screen px/frame^2; `shot check` warns over 40 at preview
  768 wide: the built-in's hard flash-step stops peak at 31) and at feet that move on the floor (`shot check` counts
  > 1.8 cm/frame within 3 cm of the floor as an info; deliberate slides are fine). Judge motion with a
  blind critic on Workbench strips, not by 2D angle scores.
- Motion sources: `shot from-video NAME --video F [--people N|pick]` (SAM 3D Body; the reference must face a still
  camera), `shot make-dance NAME --song S --routine S1,S2`, `mocap.py` keyframes (`pose_frame`, `settle`,
  `mirror_legs`, `free_feet`), or your own `key_pose` calls.

## 5. Props

- In `Props`. A shape with the right SILHOUETTE and size (a photo = a thin flat card; a blocky shape is fine if its outline
  reads), its MAIN COLOUR (a grey prop merges with the body), and a readable size on screen: at preview and 4 m, 1 px is
  about 6 mm (a 10 cm card ~16 px, a 22 cm ball ~35 px; at final 3 mm/px).
- A held prop is parented to the hand bone (it then counts as animated). Detail that does not change the outline
  belongs in the words or a prop picture, not in the mesh.
- **Thrown or bounced props (add_balls.py):** find the moments from the RIG (the dash = the fastest hips sideways move;
  the kick = the foot farthest out and high; the catch = the raised hand's visible stop), never type frame numbers.
  Ballistic paths with g = 9.81 and floor/wall bounces, keyed on EVERY frame (linear). A prop out of play is PARKED far
  away under the floor with constant keys (no blend into or out of the park), never hidden by animating hide_render.
  Measure every frame: the gap to every mannequin vertex, ball to ball, ball to set; refuse to save on a touch.
- **Timing lessons (preview px/frame at 768 wide):** a dodge ball passes the head 2.7-3 cm clear at the dash's peak
  sideways speed (17.5 m/s flat throws from beside the camera); a ball that rises 1.6 m slows on screen from 51 to 28
  px/frame (keep rises short, arrive far from the apex); a ball travelling AWAY from the camera shrinks and slows (keep a
  slight toward-the-camera direction for a constant screen speed); time a catch to the hand's VISIBLE stop (< 1
  cm/frame), not its settle (a wait reads as a pause).
- **Loose copies** (a cart of balls) belong in the Set, neutral grey, away from the play area: they look like the moving
  props otherwise.
- **Words:** `shot layers NAME --prop "Ball_*=matte deep violet rubber dodgeballs with a thin magenta seam"` (a name
  pattern; the same concrete words in the background, video and look texts). A run can re-word them with `--prop`.
- **Prop pictures** (`run new --prop-image`, `--prop-ref`): only for LARGE detailed props. Small props get words only:
  the 2026-10-25 A/B showed no visible gain for 20-35 px balls, and with the dev model the sheet's prop panel was copied
  into the frame.

## 6. Set

- Neutral grey clay only (R = G = B, value steps so parts read apart; user: "let the AI dictate the colours"). Plain
  blocks, no textures, no animation. The props bring the colour.
- Boards and screens stay EMPTY (a blank scoreboard, a plain banner): digit boxes made the backplate draw paper squares;
  the background text decides what they show.
- Set objects stand BEHIND the characters: the run projects the plate onto a floor + far wall, so a set object nearer
  than a character is painted behind it. Nothing in the characters' space or in the props' paths.
- Things at the frame edge: fully in or fully out (a sliver of bleachers at the edge read as a mistake). Side walls just
  outside the frame.
- Words: `shot layers NAME --set "an indoor school gymnasium: a wooden basketball court ..."`; then `background
  from-set --shot TEMPLATE.blend --style-preset STYLE` (the Set words are the default scene) draws the place over the
  clay render and keeps each object's place (edge fit ~0.5 is good). A template with only a floor gains nothing from
  it: use `background plate`. Ship the good one as the template's default background (section 9).

## 7. Sound

- Sound strips in the .blend (`song place SHOT.blend SONG --start S --out OUT.blend`); the export packs them. The run
  mixes the strips with ffmpeg (Blender's mixdown does not work in background mode).
- Dances: time the moves on the song's MEASURED beats (`song beats NAME`), not the asked bpm; `song loop` for seamless
  loops; put a held freeze across a clip join.
- The template's sound is the run's music unless the run names `--song`. Someone else's recording keeps a template
  PRIVATE (never built-in or in a contest entry): `shot from-video --sound video` makes such a template.

## 8. Clip texts (shot.json `clip_texts` and `clip_context`)

Follow AGENTS.md 7.2. Every clip's prompt = the SHARED CONTEXT + that clip's ACTIONS.
- `clip_context`: what holds in EVERY clip, in the words of LTX's training captions (AGENTS.md 7.2, "text B"): the
  shot type and the camera as the template has them, the facing and the motion style with pacing words, e.g. the
  built-in's "A wide shot, front-facing, the camera remains static. {character} faces the camera the whole time and
  dances the flash step: quick sideways slides with hard stops, small fast steps, knee lifts and foot flicks, loose arm
  swings." (`{character}` becomes the run's words for its character; at most 600 characters). Place, light, look and
  the character's marks come from the run's main `--video` text; the props from the layer words (by their look, never
  by what they do).
- `clip_texts` {"1": ..., "2": ...}: only what THAT clip's frames show, as BODY MECHANICS: one or two sentences, about
  15-40 words, verb + direction + stop, in time order with "then", at most about three beats (at most 1500 characters
  each). Sideways moves in picture directions, limbs by the character's own side. No cause or threat stories for objects
  the layout moves ("a ball flies at the head ... dodges" -> "slides toward the right of the picture, stops hard ...").
  Read `shot timeline NAME` (events with frames) and map each event to the clips by section 1's table. Never name an
  earlier or later clip's action. Clip k >= 2 opens on the state its shared frames show ("Still facing the camera, with
  the left foot coming back down, ..."); a move that ends inside the shared window is a verb in clip k-1 and a state in
  clip k ("holds the ball in the raised hand", not "catches the ball"; research, not A/B-tested yet).
- Set them: `shot clip-texts NAME --clip 1 "..." --clip 2 "..." --context "..."` (MCP `template_clip_texts`; `''`
  removes one; the output lists each clip's frame span). Built-in templates are read-only: edit
  `templates/library/NAME/shot.json` (the export keeps both fields). A run's `--clip-video K "..."` and
  `--clip-context` win over them; `--character-words` fills `{character}`.
- Check: print each clip's final prompt and compare it with that clip's frames of the layout.

## 9. Checks before you ship

- [ ] **`shot check NAME`** (MCP `check_template`; app: the template card's **Check** button, its chip shows "✓ checks",
      "N warnings" or "N errors"). One headless Blender read (no render, a few seconds; cached by the .blend's sha256);
      each finding has its frames and a one-line fix. `shot add`, `from-video` and `make-dance` run it at the end and
      store the summary in shot.json `check`; the export refuses on errors (also those of the shipped .blend) unless
      `--force`. The rules:
      | rule | level | what it checks |
      |---|---|---|
      | `check_incomplete` | error | every character layer's rig was measured with its tracked bones on every frame, and no rule crashed (an incomplete check is not a pass) |
      | `bone_scale` | error (> 1 %), warning (0.1-1 %) | every pose bone's scale is 1 on every frame |
      | `shared_frames` | warning | no hand raise, kick, jump, fast turn (45 deg in 8 frames) or hips dash (> 6 cm/frame) crosses a join's 25 shared frames (section 1); a raised pose held still through them is info |
      | `end_hold` | warning | < 5 mm/frame in the last 12 frames (a loop is exempt) |
      | `first_frame` | warning | frame 1 is still (< 1.5 cm/frame) and faces the camera within 30 deg |
      | `camera_speed` | warning > 0.6, error > 1.0 deg/frame | the camera's turn and orbit about the subject |
      | `camera_still` | info | still camera: a backplate is possible; moving: use a panorama |
      | `layers` | error / warning | a prop or loose object in Mannequin, no character layer or rig (errors); objects in no layer, in two layers, no Set (warnings) |
      | `figure_size` | warning | the figure fills 35-85 % of the frame height at frame 1; the head, hands and feet stay inside the frame on every frame |
      | `prop_size` | warning | each prop's median on-screen size is >= 12 px at the run's preview size (768x448; portrait 448x768) |
      | `set_neutral` | info | Set materials are neutral grey (saturation <= 0.25, no textures) |
      | `clip_texts` | warning | a multi-clip template has a clip text for every clip and a `clip_context` (section 8) |
      | `sound` | info | the packed sound (or none) |
      | `joint_accel` | warning | no joint pops over 40 px/frame^2 at preview |
      | `foot_slide` | info | feet sliding on the floor (> 1.8 cm/frame within 3 cm of it) |
- [ ] `shot add NAME --blend F --description "..."` (copies it, scans the layers, makes the Workbench preview with
      sound): WATCH the preview. Props readable at that size? Feet in frame? Frame 1 a clean front pose?
- [ ] `shot layers NAME --rescan`: `warnings` empty, `loose` [], `camera_moves` false for a backplate, each mannequin's
      `screen` head/feet inside 0..1; then the `--set` and `--prop` words.
- [ ] `shot timeline NAME`; write `clip_texts` and `clip_context` (section 8).
- [ ] A preview run with a ready character: `run new --blend ... --character C --quality preview --seed N --note "..."`,
      `run start RUN --through-clip 1 --detach`, `run score` + a FRESH blind critic per AGENTS.md 7.5 (pass: CHARACTER
      >= 8, MOTION >= 8, BACKGROUND >= 7, PICTURE >= 7), then the rest and each join.
- [ ] Export: `python setup/export_template.py NAME --order N [--background BUILTIN_BG]` (the default background must be
      a built-in in backgrounds/library; shot.json keeps the layer words, `clip_texts` and `clip_context`).
- [ ] Stored paths: `blender -b templates/library/NAME/shot.blend --python-expr "import bpy;
      print(bpy.utils.blend_paths(absolute=True))"`: no personal paths. Sizes: shan_shen_bu_balls = 2.9 MB .blend +
      0.7 MB preview.
- [ ] Licences: sound cleared or your own (docs/CREDITS.md); set assets CC0 (Poly Haven, Kenney, ambientCG,
      Quaternius) or made in code; never BlenderKit "Royalty Free" or paid generated models in a shipped template.

## 10. Worked example: shan_shen_bu_balls

1. Base dance: the flash-step moves read with SAM 3D Body (`shot from-video`), cleaned in mocap.py (`accent` for the
   side flicks, `settle` into a `pose_frame(arms="tada", ...)` end with a 0.5 s hold) ->
   `templates/shan_shen_bu_props/base_shot.blend` (still camera at 4 m, 30 mm, 276 frames = 3 clips).
2. Set + props in one script: `blender -b templates/shan_shen_bu_props/base_shot.blend --python
   templates/shan_shen_bu_props/add_balls.py -- --out balls.blend --report balls.json` (gym_set.py builds the grey gym
   in `Set`; add_balls finds 3 dashes, 2 kicks and the raised hand from the rig, keys 6 violet balls in `Props`, refuses
   to save on any touch).
3. `python -m cinestudio shot add shan_shen_bu_balls --blend balls.blend --description "..."`; watch the preview.
4. `shot layers shan_shen_bu_balls --set "an indoor school gymnasium: ..." --prop "Ball_*=matte deep violet rubber
   dodgeballs with a thin magenta seam"`.
5. `shot timeline shan_shen_bu_balls` -> `shot clip-texts` per clip (dodges / kicks and claps / the catch and the end
   pose) + `--context` ("faces the camera ...").
6. `background from-set --shot <its shot.blend> --style-preset anime_noir --seed 202` -> look -> saved as
   `gym_noir_comfyui` (edge fit 0.43 -> 0.58 after align_to_set).
7. `shot check shan_shen_bu_balls` (2026-10-25: 0 errors, 3 warnings: the clap at 209-222 crosses the shared frames
   193-217, and frame 1 already moves: the hands swing 6-16 cm/frame from the first frame).
8. Preview runs + gauntlet (AGENTS.md 7.5).
9. `python setup/export_template.py shan_shen_bu_balls --order 4 --background gym_noir_comfyui`.

## 11. Common mistakes

| Symptom in the video | Cause in the template | Fix |
|---|---|---|
| Body turns early and steps during a wave | head turned before the body faced the camera | start waves/looks within ~30 deg of the camera; head follows body |
| Steps run on to the last frame | no end hold | ease >= 24 frames into the end pose, then hold |
| Jump or morph at a clip join | an action inside the 25 shared frames | move it into one clip; hold a pose across the join |
| Blade-like hands | a pose bone scale leaked (not 1) | scales 1 on every bone and frame |
| Body turns to profile and walks at the dodges | fast sideways dash; one action text for all clips | keep the body frontal, less hip travel; per-clip texts; "faces the camera" in `clip_context` (open) |
| Kick or knee pop does not show | the move goes toward the camera / outward | sideways kicks (`mocap.accent`); knee pops inward |
| Prop ignored or merged with the body | too small, grey, or the wrong outline | readable size (px rule), main colour, the right silhouette |
| The character mask or kit includes a prop | prop in `Mannequin` | move it to `Props` |
| Extra copies of the prop fly around | loose copies near the play area | copies in the Set, grey, at the far wall |
| A set object painted behind the character | set object nearer than the character | keep set objects behind the characters |
| Paper squares or invented text on boards | digit boxes or marks on the set's boards | empty boards; the background text decides |
| A sliver at the frame edge looks like a bug | object half in frame | fully in or fully out |
| Ball looks slow-motion | long rise near the apex; flies away from the camera | short rises, slight toward-camera direction |
| Pause before a catch | catch timed to the hand's exact settle | time it to the visible stop (< 1 cm/frame) |
| Background smeared or horizon off the floor | moving camera with a backplate; plate for another camera | still camera + a plate made for it; panorama for moves |
| Hazy silhouette | the place lit from behind | light from the front (Key front-left) |
| Look image wrong from the start | frame 1 not a clean front pose | start frontal, the whole body in frame |
| Characters turn far beyond the layout at the end | end pose turned ~30 deg outward | keep end turns small |
