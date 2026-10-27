# AGENTS.md — the manual for driving cinestudio (and getting good output)

cinestudio turns a Blender shot (a mannequin, a camera, a motion) into an AI video of the user's own character, with
open-weight models in ComfyUI: Z-Image makes character fronts, Qwen-Image-Edit-2511 edits them, poses them and makes
the views, LTX-2.5 (Layout-To-Render) makes the video. Everything runs on the user's PC (RTX 3070, 8 GB VRAM, 32 GB RAM)
or on a rented GPU (docs/RUNPOD.md).

This file is an instruction manual: how to use the tool, the rules, and the tips that our tests proved. Research
reports with sources are in docs/ (named below where they matter). A full copy of the older research log is in
.cinestudio/backup/AGENTS_full_*.md (git-ignored). When you learn a lasting trick, add it to the right section here
(short, with the evidence); when a feature changes, change its section (do not append history).

## 1. How to drive it

Three interfaces, the same functions:
- **JSON CLI:** `python -m cinestudio <command> ...` (`--help` on every command and subcommand).
- **Web app:** `python -m cinestudio app` (the user's window; `--open` only on the first start).
- **MCP server:** `python -m cinestudio mcp` (stdio; 99 tools; this file is the resource `cinestudio://agents`;
  guide below).

**The CLI contract:** every command prints JSON lines: zero or more event lines (with an `"event"` key), then ONE result
line with `"ok": true|false` (on an error also `"error"` and `"error_type"`). Long steps (character images, videos,
songs, templates, backgrounds) go into the shared GPU queue and return at once with a job or run id: poll them
(`run status`, `queue list`, the job record), not faster than once a minute.

Typical flow:

```
python -m cinestudio doctor                                    # Blender, ffmpeg, ComfyUI, models
python -m cinestudio character create my-hero --display-name "My Hero" --text "what it looks like" --candidates 8
python -m cinestudio character pick my-hero --seed N
python -m cinestudio character edit my-hero --target front --instruction "..."      # optional, repeatable, undoable
python -m cinestudio character mannequin my-hero --scale 0.86 --head 1.55 ...       # its own mannequin body (or --default)
python -m cinestudio character repose my-hero                  # AnyPose: the front posed like its mannequin, then the views
python -m cinestudio character finish my-hero                  # ready (needs a body; the proxy kit is optional)
python -m cinestudio run new --blend SHOT.blend --character my-hero --video "scene and action" --quality preview --note "what this tests"
python -m cinestudio run start RUN_ID --detach                 # or --through-clip 1 for the clip gauntlet (section 7.5)
python -m cinestudio queue list                                # what runs, what waits
```

### Driving cinestudio through MCP

- **Setup:** from the repo root, `claude mcp add cinestudio -- "<repo>/.venv/Scripts/python.exe" -m cinestudio mcp`
  (Claude Code), or in any MCP client's JSON config `{"mcpServers": {"cinestudio": {"command":
  "<repo>/.venv/Scripts/python.exe", "args": ["-m", "cinestudio", "mcp"], "cwd": "<repo>"}}}`. Read the resource
  `cinestudio://agents` (this file) first; every tool's description says whether it is instant or a GPU job.
- **Tools** (the same functions as the CLI; tests/unit/test_mcp_server.py fails when a CLI command has no tool and no
  reason in its CLI-only list):
  - setup: `doctor`, `settings_status`, `models_status`, `download_models`, `stop_model_download`, `engine_status`, `list_loras`;
  - queue: `queue_list`, `cancel_queued`, `cancel_job`, `cancel_run`;
  - styles: `list_styles`, `show_style`, `new_style`, `edit_style`, `delete_style`, `set_default_style`, `set_character_style`,
    `make_style_reference`, `use_style_reference`;
  - characters: `create_character`, `import_character`, `pick_candidate`, `edit_image`, `undo_edit`, `make_views`,
    `redo_view`, `repose_character`, `restyle_sheet`, `redraw_sheet`, `finish_character`, `preview_mannequin`,
    `set_mannequin_body`, `set_proxy_kit`, `character_skins`, `rename_character`, `set_display_name`, `delete_character`,
    `list_characters`, `show_character`, `job_status`, `list_jobs`;
  - props: `list_props`, `show_prop`, `create_prop`, `import_prop`, `pick_prop_candidate`, `edit_prop`,
    `undo_prop_edit`, `set_prop_words`, `prop_skins`, `rename_prop`, `delete_prop` (and `new_video_run(prop_refs=...)`);
  - backgrounds: `list_backgrounds`, `show_background`, `new_backplate`, `backplate_from_set`, `new_panorama`,
    `panorama_depth`, `restyle_panorama`, `import_background`, `delete_background`;
  - templates: `list_templates`, `show_template`, `add_template`, `check_template` (the template rules: `shot check`),
    `template_layers`, `template_timeline`, `template_clip_texts` (one action text per clip + the clip context),
    `template_from_video` (people 1|2|pick), `pick_template_people`, `make_dance_template`, `delete_template`;
  - music and films: `make_song`, `list_songs`, `show_song`, `song_beats`, `song_loop`, `place_song`, `assemble_film`, `list_films`;
  - videos: `new_video_run` (every `run new` option for a layout run; rare ones in `advanced`), `run_status`,
    `run_details`, `run_events`, `list_runs`, `continue_run`, `fork_run`, `retake_clip`, `list_takes`, `use_take`,
    `score_clip`, `delete_run`;
  - workflows: `list_workflows`, `show_workflow`, `copy_workflow`, `set_workflow_value`, `reset_workflow`, `use_workflow`.
- **Typical flow:** `create_character` -> `job_status` -> look at the candidates -> `pick_candidate` -> (`edit_image`)
  -> `set_mannequin_body` + `set_proxy_kit` (draft both with `preview_mannequin` and LOOK at the PNGs) ->
  `repose_character` -> check the views -> `finish_character` -> `new_video_run` (quality preview, a `note`) ->
  `run_status` -> look at final.mp4. Long shots: `new_video_run(through_clip=1)` -> `score_clip` -> `retake_clip` /
  `use_take` -> `continue_run` (section 7.5). Backgrounds: `backplate_from_set` or `new_backplate` with the
  characters' style, then `background="saved:NAME"`. Templates: `template_from_video` or `make_dance_template`, then
  `template_layers` words. Films: one `make_song` score, then `assemble_film`.
- **Agent rules:** poll long work (`job_status`, `run_status`, `show_song`, `show_template`, `list_backgrounds`) at most
  once a minute; new work waits at the end of the shared queue; cancel, stop and delete tools only when the user asked
  in this conversation (deletes also need `confirm=true`); look at the output files before you claim a result; tokens
  are never set or shown through MCP (the user sets them in the app or with `settings token set`); engine start/stop,
  tier/backend changes and the Windows GPU settings stay in the CLI. A refusal comes back as a tool error with its
  reason (the same text as the CLI's `error`).

### Repository layout (where things live; keep this list current when you add a folder or a main module)

```
AGENTS.md, CLAUDE.md        this manual (CLAUDE.md only includes it); README.md: the user-facing intro
cinestudio/                 the Python package (python -m cinestudio ...)
  cli.py                      the JSON CLI (every command); mcp_server.py: the MCP tools; webapp.py: the app server
  webapi/                     the app's JSON API, one module per area (characters, props, runs, backgrounds, songs, templates,
                              styles, workflows, setup, system; common.py: shared views and file areas)
  web/                        the page: index.html, app.css, js/ (classic scripts in load order: core, characters, video,
                              music, templates, figure, backgrounds, settings, styles, props, workflows, main last)
  runner.py                   video runs: new_run, clips, chain options, takes/retakes/forks, stitching, run details
  character.py, charsheet.py, refsheet.py   characters: create/edit/views/repose/sheet/skins/mannequin body/proxy kit
  props.py                    the prop library (a picture on black + words; create/edit/undo/skins; runs' prop_refs)
  background.py, pano_depth.py              panoramas, backplates (incl. over the template's set), built-in backgrounds
  shots.py, video_template.py, mocap.py, timeline.py   templates: the library, layers, from-video (SAM 3D Body), motion
  shot_check.py               `shot check`: the template rules on one headless Blender read (bpy_jobs/check_shot.py)
  song.py, film.py            music (ACE-Step) and multi-scene films with one score
  styles.py, style_ref.py     the style library; style reference pictures made by the video model (LTX + the LoRA)
  user_workflows.py           the user's copies of the built-in graphs
  workflow.py, model_pairs.py ComfyUI graph loading/binding; the distilled/full graph pairs + the feature registry
  models.py, settings.py      model tiers and downloads; app settings and tokens
  comfy.py, server.py, backend.py, workqueue.py, jobs.py, lab.py, locks.py   the engine, the GPU queue, jobs, locks
  video.py, latent_file.py    ffmpeg helpers, stitching, audio joins; safetensors latents without torch
blender_lib/                mannequin.py (the rig, bone names, REST pose, key_pose): imported by Blender scripts
bpy_jobs/                   scripts that run INSIDE Blender (headless): renders (Workbench), scans, body/proxy kit,
                            set render, the template check's read (check_shot.py), merging N one-mannequin
                            shots (merge_duo.py), copying mannequins
                            (duplicate_mannequin.py), sound packing; mannequin_rules.json, mask_marks.json
workflows/                  the BUILT-IN ComfyUI graphs: <name>_api.json + <name>.bind.json (read-only in the app;
                            users edit copies in <home>/workflows/)
templates/                  library/<name>/ = BUILT-IN templates (shot.blend with packed sound, preview.mp4, shot.json);
                            the other folders are generator scripts (flash_step, funk_dance, shan_shen_bu_props, ...)
backgrounds/library/        BUILT-IN backgrounds (plate.png or pano.png, bg.json, set.png), read-only
characters/library/         BUILT-IN characters (character.json with ./ paths + front, views, sheet; kit inside), read-only
styles/library/             BUILT-IN styles (<name>.json)
setup/                      install (install_linux.sh, requirements, runtime.lock.json), models.json (pinned model
                            files per tier), graph generators (make_layout_graphs.py, make_redraw_graphs.py,
                            make_txt2img_graphs.py),
                            export_template.py, export_character.py (a user character -> built-in), measuring tools (brightness_check, join_check, dwpose_score, ...)
docs/                       research reports (*_research.md, with sources), CREDITS.md, MODELS.md, RUNPOD.md,
                            gauntlet_critic.md, IDEAS.md (untested ideas), TEMPLATE_GUIDE.md (authoring templates)
tests/unit/                 pytest (no GPU, no Blender); tests/integration/: manual scripts, not collected
.cinestudio/  (git-ignored) <home>: the user's PRIVATE data and runtime state: characters/, props/, runs/, shots/ (user
                            templates), backgrounds/, songs/, films/, styles/, workflows/ (graph copies), jobs/, queue/,
                            cache/, gauntlet/ (experiments + RESULTS.md), backup/, uploads, settings.json, secrets.json
```

A built-in and a user item with the same name: the USER's copy wins (templates, backgrounds, styles, characters). New built-in
assets go into the repo folders above; private results stay in .cinestudio (.gitignore lists what never ships).

## 2. Hard rules (do not break these)

- **Never stop, cancel, delete or overwrite the user's work without an explicit request.** New work goes to the end of
  the queue. Cancel only with `queue cancel` / `run cancel` when the user asks. Deletes in the app are deliberate: do
  not recreate what the user deleted.
- **Never call ComfyUI's `/free`** and never send a global interrupt; cancels are job-scoped.
- **One GPU job at a time.** The queue and the GPU lock enforce it; never start ComfyUI work around them.
- **No Blender EEVEE render while a GPU job runs** (8 GB VRAM). Workbench and CPU Cycles are fine.
- **Do not touch the user's other ComfyUI outputs, apps or PC settings** without asking (`gpu set`, Windows TDR: only on
  request). Ask before anything that costs money (renting, paid APIs).
- **Private assets stay out of the repo.** `.cinestudio/` (characters, runs, uploads, tokens) is git-ignored. Someone
  else's recording (a dance video's sound) keeps a template private; never ship it built-in or in a contest entry.
- **Commercial-safe licences only.** Never MiniMax; no non-commercial weights (RMBG-2.0/1.4, MatAnyone, Qwen-Image-2.1's
  `qwen-research`, FLUX.1-dev bases); avoid Hunyuan models (EU/UK/KR excluded) and GPL ComfyUI packs.
- **Do not claim a result you did not check:** look at the output image or video (frames, strips) before you report it.
- **Tokens are never shown** (only "set" and the source). They live in `.cinestudio/secrets.json`; environment variables
  win; each token goes only to its own service; `settings token set NAME` reads the value from stdin.
- **Do not download from YouTube;** the user supplies reference videos.
- Commit only when the user asks. Restart the app server without `--open` (the user keeps one window and reloads it).

## 3. The engine, the queue and the machine

- cinestudio starts ComfyUI itself, hidden (`comfy status|start|stop`), with Comfy Desktop's command; do not ask the user
  to open Comfy Desktop (both want port 8188). A model-family change (Z-Image / Qwen-Edit / LTX) restarts the engine
  that cinestudio started.
- Only GPU work takes a queue ticket (character create/edit/redo-view/views/style/repose, runs, songs, templates,
  backgrounds). Instant actions (undo, pick, finish, rename, touch-ups, mannequin body, proxy kit) run at once; the
  character lock keeps changes to one character in order.
- A detached worker loads the code when it is SUBMITTED: after a code change, submit again. Use detached jobs
  (`jobs.submit`, the CLI's long commands) for long work: Claude Code stops background shells on low memory; if that
  happens, do not restart them on your own.
- RAM: ComfyUI streams weights from RAM; about 4 GB stay free during Qwen jobs. Never run two heavy CPU jobs next to an
  LTX run.
- "Could not reach ComfyUI" during a FINAL-quality stage 2 can be an NVIDIA driver reset: check
  `Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'}` before blaming the code.
- SageAttention is on only for LTX-2.5 families (`comfy.SAGE_FAMILIES`): Qwen-Image-Edit made all-black pictures with it.
- **Shared engine (large tier, 2026-10-24):** `setup/models.json` tier `engine` {share_families, min_ram_gb 150}: on a
  machine with that much RAM a new model family JOINS the running engine (server.shared_engine/ensure_family): no
  restart and no reload from the network volume per family change. An engine started with SageAttention (LTX only)
  restarts once without Sage when a Sage-unsafe family (Qwen, ACE) arrives. The 8gb tier keeps one family per
  engine. Not yet tested on a pod (RAM use of Qwen bf16 + LTX + ACE together: check it).
- Tests that hang: check the free disk (`Get-PSDrive C`, `%TEMP%/pytest-of-*`) and stop leftover ffmpeg processes.

## 4. Characters

### 4.1 IDs, names, readiness
- `name` in character.json is the **ID**: the folder, every file path, jobs, locks, API URLs, the CLI and the runs. The
  **display name** (any text, at most 80 characters) is only shown. In the app the form's name is the display name and
  the ID is made from it (`character.new_id`: lowercase, other characters -> "-", -2/-3 when taken). The app's rename
  changes only the display name; `character rename` changes the ID (the folder moves). Never use the display name in a
  path, a prompt or a job.
- **Built-in characters (2026-10-26, user: "move it in as a built in character ... ship it in with proxy kits
  too"):** `characters/library/<id>/` in the repo (character.json with RELATIVE paths `./front.png`, the front,
  views, face and sheet; the body and the proxy kit inside the record), read-only like the built-in templates,
  backgrounds and styles: character.load resolves the paths and marks `builtin`; every change (edits, kit, body,
  rename, delete, style) is refused (`refuse_builtin`); a user character of the same ID wins; new IDs skip built-in
  IDs (`masked` -> `masked-2`); skins of a built-in are allowed (a new record). Built-ins: `masked` ("Masked",
  anime_noir, the fitted-head kit), `kitsune` ("Kitsune", anime_cel, chibi body, the ring-loft kitsune kit) and its skin
  `kitsune-3d` ("Kitsune 3D", 3d_animated, the v3 modelled head `./head_asset.blend`; 2026-10-27, user: "make kitsune
  a built in char"). To change a built-in: edit its folder in the repo (or make a copy as a user character). Export:
  `python setup/export_character.py NAME [--display-name T] [--note T]` (copies only the record's files and a kit's
  head asset, writes `./` paths, drops candidates/edits/checks and personal paths, refuses on a path left); then move
  the user's own folder to .cinestudio/backup (while it exists it hides the built-in).
- **Ready for a Blender video** = status `ready` (views + sheet finished), not `sheet_stale`, and a **mannequin body**
  (`character.video_missing`; {} = the template's body counts as set). The proxy kit is optional. `run new` refuses a
  Blender shot with any character that is not ready; the app lists such characters disabled with the reason.

### 4.2 The front view (Z-Image)
- Make 8 candidates and let the user pick. Z-Image base: 25 steps, cfg 4, res_multistep, a real negative.
- Prompt (Z-Image's own rewriter, pe.py): objective and concrete; first the core (subject, COUNT, colours), then
  composition, light, material; no metaphors or meta tags ("8K", "masterpiece", "premium"); long and detailed works best.
- Repeated details: one garment per clause, head to feet, each with colour, material, EXTENT (from where to where), SIDE
  and COUNT ("white bandages around BOTH forearms, left and right, from the wrist to just below the elbow ... four
  wraps in total"). End the description with a period; keep pose, fingers and light out of the description.
- Say "exactly five clearly separated fingers". No negations in the positive text: excluded items go into the negative.
- The background as its own sentence ("The background is the same solid pure black everywhere, empty and seamless,
  without floor or horizon"); light "flat and even from the front, like a neutral turntable reference render". Avoid
  "cinematic", "global illumination", "premium", "polished" (they invite a lit set).

### 4.3 Edits (Qwen-Image-Edit-2511)
- **One change per edit.** End with "Keep everything else exactly the same." Quality mode: 40 steps, cfg 3 (cfg 4 with a
  second image gave errors); about 4 minutes per edit on the 3070 (int8 file, one picture, CFG tail: below). The input
  background is set to pure black first (automatic).
- **Speed (bench 2026-10-24, .cinestudio/gauntlet/qwen_speed, same edit and seed; docs/qwen_speed_research.md):** the edit
  is COMPUTE-bound (each picture adds ~4k tokens). 18.2 s/step before -> int8 ConvRot file 11.6 (fp8 is emulated as bf16
  on Ampere; int8 is native) -> + Picture 1 only when there is no reference 6.3 (bind `optional_images`: the graph sent
  the same picture again as Picture 2) -> + core `CFGOverride` cfg 1 for the last 35% of the steps 5.2 (3.3x in all).
  All results were the same picture (mean difference <= 0.5/255). The CFG tail is only in the edit graph so far (views
  and AnyPose untested). A real reference still costs its tokens: a smaller Picture 2 is an untested proposal.
- **Name the inputs "Picture 1" / "Picture 2"** (ComfyUI labels them so). Official instruction form: the change with its
  place; "Replace Y with X" + 2-4 key features of X; then NAME what to keep; style words at the end; text in quotes.
- **A reference image (Picture 2) must show ONE detail, tightly cropped** (a whole image is copied too much: a third arm,
  other gloves). Without a reference the graph sends Picture 1 only (bind `optional_images`).
- **Small fixes: always `--region x0,y0,x1,y1`** (the box is cut out, enlarged to ~1024 px, edited and pasted back with
  soft edges: nothing outside can change). Check the image size first (1024x1536 = width x height). Describe the CROP,
  not the character ("This is a close-up of a gloved hand ...").
- `--denoise`: 1.0 rebuilds (extra limbs), 0.8 changes the pose of a hand, 0.6 changes nothing. Region edits at 0.85 do
  NOT remove a part (a strap, a skin patch): do removals in code (4.8).
- **Paint, then refine:** draw the part in code (`character.touch_up`, colours sampled from the image), then a region
  refine at denoise 0.5. This fixed a missing thumb after five failed text edits. Say "flat cel-shaded, no fingernails,
  no wrinkles" for drawn styles.
- Every edit, redo, pick and touch-up can be undone per image: `character undo NAME --target T`.
- Look at a region edit's result BEFORE you queue another edit with the same method.
- The CFG tail (no negative pass in the last 35%) is an ABSOLUTE noise range: edits with `--denoise` < 1 keep full
  CFG (character.edit_image), or most of their steps would run without the negative.

### 4.4 Prompting and controlling Qwen output (docs/qwen_negatives_output_control_research.md; NOT yet A/B-tested)
- Say the background as its OWN sentence near the start ("The background is pure black everywhere, empty, without
  floor, shadow or horizon"). A conversion that listed "the black background" as the last of 12 kept items came out on
  a WHITE studio backdrop with a floor shadow. Leave out "turntable", "studio", "toy", "Pixar", "product".
- Negatives ARE applied at cfg 3 (our graph = the official 2511 template, negative "" today). Proposed lists:
  - base: extra arms, extra hands, extra fingers, missing fingers, fused fingers, duplicate character, second figure,
    cropped feet, cropped head, text, watermark, logo, blurry, low quality;
  - views/edits on black: + white background, grey background, gradient background, studio backdrop, seamless paper
    backdrop, room, wall, floor, ground shadow, cast shadow, floor reflection, vignette, spotlight, rim light, dramatic
    side lighting, harsh shadows, strong highlights + the character's exclude list; a BACK view also: face, mask, eye
    holes (and positive words for what IS visible, not "NOT visible");
  - style conversion: + the source style (2D -> 3D: flat 2D illustration, line art, cel shading, dark outlines), plastic
    toy, vinyl figure, product photo, known colour drift;
  - look images: base + white wooden mannequin, mannequin joints, grey clay figure, flat 2D sticker, cut-out outline,
    white outline around the figure, extra person, crowd (no background words: the Blender scene must stay).
  - Side effects: weaker edits (PerpNegGuider keeps the baseline, ~+50% time), washed-out colour (measure chroma); never
    negate what must stay ("black" for a black suit).
- Control that does not depend on the text (ranked): (1) a learned figure matte on every result, then composite onto
  black (ComfyUI core `LoadBackgroundRemovalModel` / `RemoveBackground`; BiRefNet MIT weights not downloaded yet; check
  the DIS5K data's commercial terms first; never the paid Bria/Recraft nodes); (2) `SetLatentNoiseMask` so only the
  figure region can change; (3) 2-3 seeds gated in code (background uniformity, floor-shadow pixels, silhouette IoU,
  colour drift, facing side).

### 4.5 Views, repose and the sheet
- **Order A (the default):** `character repose NAME` = AnyPose poses ONLY THE FRONT like the character's own mannequin
  (Blender renders the bare mannequin with the character's body: bpy_jobs/render_turnaround.py), then the views are made
  from the posed front with the Multiple-Angles LoRA (0.9). For body "human" `views` poses the front first unless it is
  already posed or the user gave side/back images. Repose from the ORIGINAL front (undo an old posed front first).
- **The slot rule (AnyPose):** Picture 1 = the CHARACTER (Qwen keeps its identity), Picture 2 = the POSE SOURCE. The
  other way round copies the wrong pose and draws the mannequin's parts.
- AnyPose pulls proportions toward a normal figure (the head ~5-10% smaller): give the mannequin a body ~10% more
  extreme than the target. It can also redraw small design parts (a taller mask, a head strap): compare with the
  original and paste the original part back in code.
- Views prompt: `<sks> right side view eye-level shot medium shot` (also back, left). "medium shot" is the default ("wide
  shot" invites a room). The card does not say whose right is meant: check the side in code. **Sheet slots go by PICTURE
  direction:** views/right.png faces the picture's LEFT.
- Check every view: the side (both sides once faced the same way; a redo can give the wrong side and a room), the finger
  count (a hanging hand may hide its thumb), colours that went missing (the colour check notes), and on a BACK view no
  mask or face on the back of the head (the back gets no description; repair by pasting the input back's head).
- The FACE panel is cut from the front (`character.face_from_front`); a Qwen close-up drew a wrong small full body.
- **The sheet** (`character.build_sheet`, layout `panels5`): face close-up + front, right, back, left on one feet line,
  black panels and gutters. **One scale for every sheet:** the bodies are drawn at `SHEET_FILL` 0.78 x the character's
  height ratio (its mannequin height / the template's 1.806 m, measured by bpy_jobs/mannequin_height.py, cached), so a
  shorter character is drawn shorter; a wide costume makes the face panel give way first; the limit is 1.14x the
  template (taller characters are drawn at the limit). A run's group sheet (several characters, 7.3) draws every
  character at its real ratio.
  The app's Sheet view crops the picture to its content FOR DISPLAY only (characters.js fitSheetView); the file keeps
  its margins.
- **Sheet background colour is NOT verified.** Black panels lose a black costume's edges (masked: >= 25% of outline
  pixels within 10 L*); white loses a white mask; light-grey panels (RGB ~160-170, Lightricks' example) measured best.
  Do not change the default before the A/B in the research report (black vs grey 128 vs grey ~185 panels, one seed,
  identical layout runs, a blind critic).
- Imported art on a light background: run `refsheet.clean_edges` (white specks between fingers otherwise).

### 4.6 The character's own mannequin (body) and the Mannequin view
- Each character has its own mannequin body: `character.json` `mannequin` {scale, head, neck, torso, legs, arms} (1.0 =
  the template's; bpy_jobs/mannequin_body.py applies uniform pose-bone scales from bpy_jobs/mannequin_rules.json, the one
  source of the body math; shorter legs lower the rig so the feet stay on the floor). The same body is used by the
  repose (the AnyPose pose source) and by every Blender run (the run's copy of the template's mannequin).
- **The app's Mannequin view** (Cast tab, Characters): the character's front beside its mannequin (front, side, back) rendered
  by Blender Workbench at ONE scale (bpy_jobs/mannequin_preview.py, ~3 s, cached in <home>/cache/mannequin_preview; a
  teal ring = the template's 1.81 m); presets Adult / Teen / Child / Chibi (Teen, Child: untested proposals); six body
  sliders; one Save (redraws the sheet at the new height; the pose changes only after a repose). Saving a body after a
  preview starts no Blender (the height is cached).
- Measured: kitsune (chibi) body {scale 0.86, head 1.55, neck 0.35, torso 0.8, legs 0.7, arms 0.85} = 1.30 m (72%).
- `character mannequin NAME --default` replaces a body: ask before using it on a character that has one. The app's
  Mannequin view also has "Pose the front like this body" (the repose: Qwen jobs, undoable).

### 4.7 The proxy kit (stand-in parts) — set by an AI agent through MCP
- Simple clay parts on the mannequin in every Blender run (bpy_jobs/proxy_kit.py `build`), so the layout carries the
  character's outline: a mask (shape round|pointed, eyes, eye_size, ears + ear_inner, top, `marks`), hair (length,
  front, locks, back, colour), cape (length hips|thigh|knee, width, flare, wrap), jacket, bands, suit and shoe colours.
  Colours are linear RGB 0..1. `character.validate_proxy` checks the whole kit before it is saved.
- **The app has no kit editor (user decision):** agents set it with the MCP tools `preview_mannequin` (render a draft
  body/kit; LOOK at the PNGs), `set_proxy_kit`, `set_mannequin_body` (CLI: `character proxy NAME --spec JSON`).
- Make the kit from the sheet: the character's own colours (a light-grey hair cap read as part of a white mask: use
  the hair's real colour), the mask's shape and size, the garment lengths. **Give a mask its marks** (`marks: "kitsune"`
  or a list of {x, z, w, h, roll, color} stickers; presets in bpy_jobs/mask_marks.json): a plain white mask on a white
  head reads the same from the front and the back, and the video turned characters away. A short hair cap that leaves
  the back of the head white looks like a second blank face: use hair "medium" or a `back`.
- Check the kit from front, side and back (`preview_mannequin`) before a run.
- **Kit gauntlet (2026-10-26, user: "make sure it looks very similar to our reference images"):**
  `.cinestudio/gauntlet/kit_masked/tools/kit_eval.py CHARACTER CANDIDATES.json OUT` renders each candidate kit with the
  REPOSE camera and the character's body (proxy_kit.py + render_turnaround.py, Cycles CPU), so the character's own
  views line up, and scores colour classes (white/purple/hair/dark) per view and for the head; `calibrate.py` compares
  rendered with reference colours; `eyes.py` measures the eye holes inside the mask. Change ONE thing per candidate,
  then a blind critic (anonymous tags). RESULTS.md there. Lessons, for every kit:
  - **Kit only what the video draws wrong (user, 2026-10-26: "it's mainly the mask"; "i dont really like the straps"):**
    the mask with its eye holes, the hair that frames it, the head and neck. The clothes render well from the sheet and
    the look image; bandage tubes made earlier renders worse. (A full costume kit scored higher on colour classes,
    0.37 vs 0.09, but that is not the goal.)
  - The kit material has NO specular shine and the eye holes are pure black (proxy_kit.clay, since 2026-10-26): the
    old 0.5 shine turned the dark plum hair grey-lavender, and the 0.03 eye material rendered mid-grey.
  - The mask: `mask.top` must not cut through the eye holes (`mask.eye_height`, default 0.12 head radii; masked
    -0.04); a pointed chin needs `mask.chin_back` (~0.6), or it is a forward blade from the side (the "beak" the
    video drew); keep `mask.depth` 1 (0.8 let the head poke through the mask); `mask.width`/`length` shape it.
  - `hair.cover: true` paints the mannequin's own head in the hair colour (only the mask stays light): a white head
    beside or behind the hair reads as a second face. `head.neck` = a dark collar colour (a white neck merges with a
    white mask's chin). `head.width/length/depth` reshape the ball head (masked: a smaller head was worse).
  - **The kit's OWN fitted head (`head.fit`, 2026-10-26, user: "making the headshape part of the proxy kit"):**
    `.cinestudio/gauntlet/head_masked/fit_head.py CHARACTER OUT.json` measures the character's views (front, right)
    in the frame of its repose mannequin render: the head outline, the mask region (front width, how far back it
    wraps from the side) and the eye-hole outline, in head units; the kit then builds a hair loft + a separate mask
    loft with black eye faces and hides the ball head (proxy_kit.fitted_head; `fit.eye_scale` 1.3 so the eyes read
    at the ~37 px video size). Mask options (10 blind rounds on masked, .cinestudio/gauntlet/kit_masked/RESULTS.md):
    `round_top` (a broad arch under a fringe, not a spike), `ridge` ~0.12 + `shade` ~0.83 / `shade_side` (the
    reference's centre crease with one darker half: a flat white mask reads as a card), `proud` ~0.15 (the mask stands
    out of the hair), `chin_drop` ~0.12, the eye a clean ALMOND fitted to the measured eye (fit_head.py) at
    `eye_scale` ~1.25 (1.15 read as slits, 1.35 as alien eyes); keep `asymmetric` false (the fringe's line looked
    torn). masked (H14): critic 6-7/10 vs 4/10 for the ellipsoid mask. Open: the side profile (a short shell; the
    chin not long enough), bob volume and a fringe.
  - **A MODELLED head (`head.asset`, 2026-10-26, user: "construct the head seperately and then throw it onto the
    head"):** `{"head": {"asset": {"file": "<path>.blend"}}}` appends every mesh of a .blend made in HEAD UNITS (head
    centre at the origin, radius 1, face toward -Y), scales it to the mannequin head and parents it to the head bone
    (proxy_kit.head_asset; the ball head is hidden; mask/hair parts are not built). For shapes the ring loft cannot
    make (a fox mask: ears with depth, a snout, cords, hair clumps): kitsune-3d's head is built by
    `.cinestudio/gauntlet/kit_kitsune3d/tools/build_kitsune_head.py FIT PARAMS OUT` from its measured fit
    (fit_head_kitsune.py: the dome split, ears, paint polygons front and side, the band colour) as overlapping
    primitives joined by a VOXEL REMESH + smooth (no ledges between ears and dome). Blind critics in both column
    orders: the modelled head (A2) 6-6.5/10 beat the best ring kit 4-5/10; the ring method had plateaued (~6/10, its
    winners flipped with the column order); the old kit 3/10 (RESULTS.md there). A face mask covers only the FACE:
    the back view must show hair (kitsune-3d's back view had a white shell over the head: corrected by a region
    edit, 2026-10-26, user), and the head then follows it (A8: a hair dome, the ear tips and the cord on the hair).
    Since 2026-10-27 kitsune-3d's head is the v3 MODEL (tools/v3_mask.py, v3_split.py, v3_marks.py, v3_cords.py,
    v3_hair.py: a mask SHELL with its ears and traced decals, worn on a separate head, the hair growing from the
    head; modelled in the live Blender through the Blender MCP against reference planes, judged in a Cycles look
    render beside the sheet; the kit copy matte with the measured hair colour). The Mannequin view's cache key
    includes the head file (mannequin_preview._asset_sig). Not yet tried in a video run.
    masked's head is a v3 model too (2026-10-27, user: "run the same gaunlet for masked 3d's head";
    .cinestudio/gauntlet/kit_masked3d: tools/fit_masked_v3.py, m3_head.py, m3_hair.py, RESULTS.md): a mask shell lofted
    over the measured mask region (thickness grown BACKWARD, the inner shell dark), a head volume carved behind it, a
    bob of clumps from a crown whorl and ONE fringe sheet ending on the measured diagonal edge. Lessons: draw a mask's
    eye holes as ALMOND decals between the measured corners (outlines measured from shaded pixels came out blobby; cut
    holes tore after the rounding); keep a decal on the mask (a point past its edge stood out as a fin from the side);
    a grey half in the sheet is drawn LIGHT (the side view shows it white): paint only a strip by the ridge; Cycles
    look renders on the CPU next to a FINAL LTX run crashed ComfyUI: render after the run. Blind critics: 7 and 6/10
    vs 4 and 3.5/10 for the ring-loft kit.
  - The ring-loft fitted head also gained `paint` / `paint_side` (mark polygons), `ears`, `cap`, `helmet_v`, `band`,
    `hair_below` / `hair_drop`, `snout`, `chin_cap` (proxy_kit.fitted_head); masked keeps its ring head.
  - Other kit parts stay available when a garment does render wrong: `jacket.open` (0..0.9) opens the front;
    `bands` as 4-5 thin wraps on one bone read as wrapped (one tube reads as a sleeve or a boot).

### 4.8 Fixes in code (touch_up) and pitfalls
- Exact small fixes are better in code (OpenCV) than by the model; save with `character.touch_up(name, image, note,
  target=...)` (undoable).
- Remove a part (a strap) by its colour components that reach where they should not (not the mask's own marks), then
  fill along the real edge line (interpolated from clean rows above and below): mask side -> the mask's colour, hair
  side -> a column fill between clean rows. Re-add a part (knots) by mirroring the clean side view's part.
- **A colour flood fill is dangerous on a light backdrop:** `black_background` flowed into a white mask. Protect the
  figure with the input's own outline (same pose): keep everything inside the eroded outline, black outside a dilated
  band, only light colourless pixels connected to the outside in the band; colourless grey-white pixels low in the
  figure are the floor shadow. Then remove the HALO (the outline's pixels are mixed with the old white backdrop and show
  as a white outline on black: the user found it distracting and it can reach the video through the sheet): in a ~4 px
  ring inside the outline, light colourless pixels -> background, the rest un-mixed from white; bluish-grey shadow
  wedges at the feet (hue 95-130, low saturation) -> background. Measure such regions on the CLEANED picture (on the
  raw output the white backdrop starts at row 0). Script: scratchpad k3d_front_fix.py (kitsune-3d front).
- **A code change of the FRONT drops the views from the record** (`touch_up(target="front")`): put them back when the
  views stay valid (a style conversion).
- Light that Qwen left uneven: flatten per material (`.cinestudio/gauntlet/cloaked3d/tools/delight.py`, then match the
  other views' brightness: `match.py`). Colour fixes by hue in the masked region (brown -> charcoal etc.).
- A converted character's back view: dark gloves on black can come out as thin claws: paint them from the front gloves
  (flipped, fitted to the forearm end).

### 4.9 A character in another style (2D -> 3D and back)
- **Skins (2026-10-25, user: "think of it as like switching skins for a character"):** one character, more than one
  style. Each skin is its own record (its ID, pictures, sheet, body, kit; runs and jobs use the skin's ID) with
  `skin_of` = the main record's ID. The app shows one tile per character and a floating skin menu on the picture's
  top-left corner (no layout room; a row per skin, a skin being made with its progress, "+ New skin…" = a dialog of the
  library styles, delete / name-the-style actions; hidden on Mannequin and Candidates) (docs/skin_picker_research.md).
  A new skin (`character skin NAME STYLE`, MCP `character_skins`): the front redrawn by the instruction below,
  Picture 2 = the style's reference picture if it has one, the pose kept, then the views and the sheet; finish it
  after a look. `character link-skin NAME --to MAIN` joins an existing copy
  (kitsune-3d is a skin of kitsune); a main record with skins cannot be deleted first; a rename moves its skins along.
  Not yet run on the GPU end to end: check the first new skin's front and views (4.5 checks).
- Make a COPY (a new ID), then convert the front and each view with the same Qwen instruction: "Redraw the character of
  Picture 1 in the rendering style of Picture 2: <the look in plain words>. The character's design stays exactly the
  same: <every part with its colour>, the pose." + the background sentence (4.4); Picture 2 = a tight crop of a character
  in the target style. A BACK view: describe the back, denoise 0.7. The text method converted kitsune to a real 3D look
  and kept the design; the dx8152 Style-Transfer LoRA changed almost nothing.
- Use the target style's own words in the character's `style` field (an LTX LoRA cannot load into Qwen; its trigger
  word does nothing there). Do not invent design details in the instruction.
- Compare every converted view with the original (marks, straps, colours, hands) and with the other views.

### 4.10 Props (objects for the template's Props layer; 2026-10-25, NOT yet tested in a GPU run)
- **A prop** = one picture (the object alone on pure black, flatly lit, in the shot's style; optionally a side view) +
  its `words` (a few concrete words every clip prompt and the run sheet's "**Prop:**" line use) in <home>/props/<id>/
  (cinestudio/props.py). It shares the character tools (IDs and display names, the image job, the Qwen edit with
  region and undo: character.edit_picture, the style stamp, skins) but has no views, sheet, mannequin or kit: the
  template's Blender object is its stand-in in the layout (section 5 props).
- **Make one:** `prop create NAME --words "..." [--description "..."] --style STYLE [--candidates 4]` (a GPU job:
  Qwen-Image-2512 through the backplate graph, 360 LoRA at 0, ~7 min per 1024x1024 picture on the 3070; `--model
  z-image` = the fronts' model), LOOK at candidates/candidates.jpg, `prop pick NAME --seed N`; or `prop import NAME
  --image FILE --words "..." [--style S] [--candidate F ...]`. Then `prop edit` (one change, `--region` for details;
  `prop undo`), `prop words`, `prop skin NAME STYLE` (the same prop in another library style: one Qwen job, the shape
  kept), `prop display-name|rename|delete`. MCP: the props group of tools; app: the Cast tab's Props switch.
- **The picture prompt** (props.PROP_PROMPT, from the dodgeball that worked): "One single object shown alone as a prop
  reference picture: <description>, seen at eye level, centred ... filling about half of its height. <style look>.
  The background is the same solid pure black everywhere ... The light is flat and even from the front"; negative
  props.PROP_NEGATIVE + the style's excluded looks. Use the characters' style (5. "every asset in one style").
- **Use one in a video:** `run new ... --prop-ref "Ball_*=dodgeball"` (MCP `new_video_run(prop_refs=...)`; app: the
  object row's prop select in "Objects in this template"): the prop's front becomes the group's prop picture (the
  prop panel of the run's sheet, at most 2) and its words the group's words; `--prop` / `--prop-image` for the same
  pattern win. Give the Blender prop the picture's main colour (the look image still shows the Blender colour).
- First library prop: `dodgeball` (anime_noir; front = Qwen-Image seed 505, one thin magenta seam, closest to a real
  dodgeball and to the template's Blender balls; seeds 202/303/404 are its candidates).

## 5. Templates (Blender shots)

**Authoring guide:** docs/TEMPLATE_GUIDE.md (the full checklist); the bullets below keep the rules in brief.

- **Every Blender shot is a template** (the Templates tab is the one library; the Video tab takes shots from it). Built-in
  templates ship in `templates/library/<name>/` (shot.blend with packed sounds, preview.mp4, shot.json with relative
  paths); the user's templates come first (a user template of the same name hides the built-in in the list). Export
  with `python setup/export_template.py NAME --order N [--background BUILTIN_BG]`, then list the
  .blend's stored paths and remove personal ones.
- **Rules for a template that does not mislead the video:** each action inside ONE clip (clip k = frames 96(k-1)+1 ..
  +121) and outside the 25 shared frames; start a wave or a look at the camera only while the body is within ~30 deg of
  the camera (an early head turn made LTX turn the body early and step); the head follows the body; start an orbit at
  the front; keep the camera under ~0.6 deg/frame (or use more clips); end on a held stance (LTX carries fast steps on to
  the last frame otherwise).
- Built-in: `shan_shen_bu_balls` (the flash-step meme, the moves rebuilt in Blender from SAM 3D, 11.5 s / 3 clips, as a
  dodgeball gag: three balls pass the head at the flash steps, two kicked away, one caught in the quirky ta-da end pose;
  Props = 6 violet balls, Set = a grey gym with empty boards; its default background is the built-in backplate
  `gym_noir_comfyui`). It is also the default pose source (character.default_pose_shot: only its Mannequin collection is
  rendered). The plain dance without props left the built-ins on 2026-10-25; its .blend stays as the generator's source
  (templates/shan_shen_bu_props/base_shot.blend, read by add_balls.py).
- **From a dance video:** `shot from-video NAME --video F [--start S] [--seconds N] [--sound video|none]` = SAM 3D Body
  (one GPU job) -> cinestudio/mocap.py -> templates/flash_step/build.py (still front camera at 4 m, plain floor). The
  reference must face a still camera. Tools in mocap.py: `accent` (make a hidden kick read from the front: lift and turn
  the shin outward; check the ankle acceleration), `mirror_legs` (copy a kick to the other side; ask which side first),
  `settle` (ease into an end pose: `rest_frame()` or `pose_frame(arms="rest"|"crossed"|"tada", yaw_deg, lean_deg,
  back_deg, chin_deg, knee_deg, hips_fwd, head_tilt_deg, knee_pop)`; ease >= 24 frames + a hold; `free_feet`). A kick
  toward the camera does not read; a knee pop must go INWARD to show from the front.
- **Several people from one video (2026-10-25):** `shot from-video NAME --video F --people 2|3|4|all|pick` (app:
  People: One / Two / Three / All (up to 4) / Pick) runs workflows/sam3d_body_multi: SAM 3.1
  (`checkpoints/sam3.1_multiplex_fp16`, SAM License) finds and numbers every person (at most 4 followed), SAM 3D Body
  reads each inside its mask (absolute camera-space roots). `2`/`3`/`4` take that many seen longest and largest, `all`
  every one seen in a quarter of the clip (up to mocap.PEOPLE_MAX 4 = the most characters a run takes). `pick` stops at
  status "pick" with a numbered sheet (people/sheet.jpg, the preview video): LOOK at it, then `shot pick-people NAME
  --tracks 0,2,1` (1-4 people; no second GPU job). The people become Mannequin, Mannequin2, ... LEFT TO RIGHT (each
  built alone, then bpy_jobs/merge_duo.py merges any number), at their places from the video; the template is trimmed
  to the time all are in it; a run on it needs one --character per slot. Depth for 3+: the depth SPREAD is scaled into
  1.5 m (the order kept; a clamp per person made two equal), neighbours closer than 0.5 m are moved apart sideways.
  Measured (trio_test 2026-10-25, a composed 5 s clip of three run videos side by side: .cinestudio/refs/multi_test/
  three_dances.mp4): all three found and followed (one hidden in 11 of 120 frames, interpolated), three separate
  mannequins, each with its own dance; the larger middle dancer was read nearer (depth -0.7 m vs +0.6/+0.7 m).
  Measured (two test clips, ~2.5 min GPU for 10 s): both people found and followed 100% of frames, no identity swap,
  each mannequin copies its own dance; lateral distance about right (1.78 m read for a 1.5 m spacing); DEPTH is a guess
  (a short chibi character was put 1.9 m behind its partner at the same depth): clamped to 1.5 m in total. ONE level
  for all people (mocap.shared_floor: each root slid along its camera ray, the feet's floor only when it agrees with
  the mean body up within 5 deg, else the body up): per-person floor fits leaned the duo's mannequins 7.5 and 18.5 deg
  (the user saw them "slanted"); now 1.6 deg each.
- **Check every template: `shot check NAME`** (MCP `check_template`; app: the card's Check button and chip;
  cinestudio/shot_check.py + bpy_jobs/check_shot.py: one headless read, no render, a few seconds, cached by the .blend's
  sha256). Rules: `bone_scale` (every pose bone 1 on every frame; > 1 % = error: a leak once made blade-like hands),
  `check_incomplete` (error: a rig not measured or without the tracked bones, or a rule that crashed: unknown is not a
  pass), `shared_frames` (a hand raise, kick, jump, fast turn or hips dash across a join's 25 shared frames; a raised
  pose HELD still through them is only info), `end_hold`
  (< 5 mm/frame in the last 12 frames; loops exempt), `first_frame` (still, within 30 deg of the camera), `camera_speed`
  (> 0.6 deg/frame warning, > 1.0 error) / `camera_still`, `layers` (a prop in Mannequin = error; loose, no Set),
  `figure_size` (35-85 % of the frame height at frame 1; the body inside the frame on every frame), `prop_size`
  (>= 12 px at preview), `set_neutral`, `clip_texts` (a text per clip + the context), `sound`, `joint_accel` (pops
  > 40 px/frame^2 at preview; the built-in's hard flash-step stops peak at 31), `foot_slide` (info). "At preview" =
  the run's preview size from the render's shape (768x448; portrait 448x768). `shot add`, from-video and make-dance
  store the summary in shot.json `check` (with a fingerprint of the clip texts / loop: the card marks it stale when
  they change); `setup/export_template.py` refuses on errors of the template AND of the shipped .blend unless
  `--force`. It caught the r18 scale bug (hand.L 1.35) and the 1.22 deg/frame orbit. Judge motion by a blind
  critic, not by 2D angle scores.
- **Dance from a song:** `shot make-dance NAME --song S --routine S1,S2 [--loop] [--seconds 10]` (routines in
  templates/funk_dance/build.py: Locking, shuffles S1/S2, horse-party H1/H2). Moves toward the camera do not read; a
  visible bounce needs ~8 cm of hip drop; put a held freeze across a clip join.
- **Several characters:** `run new ... --character A --character B [--character C ...]` (one per slot in slot order,
  at most 4; `--character2` = the old form of the second) on a one-mannequin template copies the mannequin (the same
  motion) for each, `--spacing` m apart (bpy_jobs/duplicate_mannequin.py `--names`); a template with several mannequins
  needs exactly one character per slot (fewer or more are refused, naming each slot). For different motions build a
  template with several mannequins: from a video (above) or bpy_jobs/merge_duo.py (`--other B.blend --other C.blend
  [--places "x,y;x,y;x,y"]`: Mannequin2, Mannequin3, ...). `shan_shen_bu_duo` (private) = arms-crossed b-boy end.
- **Template layers (2026-10-24):** every template has three layers by Blender collection: `Mannequin`, `Mannequin2`,
  ... (the characters' rigs), `Props` (objects that move or that a character touches) and `Set` (the static place).
  Nothing else except the camera and lights. `shot add` scans them (bpy_jobs/scan_layers.py) and warns (objects in no
  layer, no Set, a moving camera for a backplate); `shot layers NAME --set "what the set stands for" --prop
  "Ball_*=plain red rubber dodgeballs" [--rescan]` stores their words in shot.json. A run freezes the words: the clip
  prompt gets "In the scene: <props>." and `background from-set` takes the Set words when `--scene` is not given. The
  Set is the source of the backplate and is hidden in the run (the picture replaces it); Props stay in the layout.
  The scan also records each mannequin's first-frame place (hips [x, y], head/feet projected by the camera: the form's
  numbered markers); templates scanned before 2026-10-25 lack it (`shot layers NAME --rescan`). A run can re-word a
  prop group: `run new --prop "Ball_*=red and white beach balls"` (MCP `props`; app: "Objects in this template";
  clip prompts only, not the look image; a fork keeps them unless `--prop` changes them, empty words = the template's).
- **Props in a template (2026-10-24; not A/B-tested yet):** Layout-To-Render keeps "the placement of objects from the
  layout" and LTX follows the OUTLINES in it (the bare mannequin's outline turned a poncho into a capelet), while text
  alone steers little. So a prop needs a Blender shape: the right silhouette and size (a photo = a thin flat card; a
  blocky shape is fine if its outline reads), the right place (a held prop parented to the hand bone), its main colour
  (a grey prop merges with the body), and a visible size on screen (at preview 768x448 and 4 m, 1 px is about 6 mm: a
  10 cm card is ~16 px; a 22 cm ball ~35 px). Detail that does not change the outline belongs in the reference images.
  Put props in a `Props` collection, never in `Mannequin` (the character mask and the proxy kit read that collection).
- **A prop picture (2026-10-25; OFF by default after the A/B below):** `run new --prop-image "Ball_*=ball.png"` (at most 2; MCP
  `prop_images`; app: "Picture…" in the object row; `run fork --prop-image "Ball_*="` removes it). The picture (the prop
  alone on black, in the characters' style) is frozen into the run and drawn into the bottom-right cell of a prop column
  in the RUN's copy of the sheet (inputs/sheet_props.png, same size; the bodies keep their scale, the face panel gives
  way first; the character's sheet file is not changed), and the sheet text gets "**Prop:** <words>, shown in the
  bottom-right panel." The look image still shows the Blender prop's colour: give the Blender prop the picture's main
  colour (or say it with `--look-extra`).
  - **A/B (2026-10-25, masked on shan_shen_bu_balls, gym_noir_comfyui, seed 202, final):** with the FULL (dev) model the
    panel was copied into the frame as a giant ball in the bottom-right corner in clips 1 and 3 (run
    r20261025-060521-d2d6). With the distilled model, panel (r20261025-130229-e61e) vs words only
    (r20261025-130232-5475): no giant ball; the 20-35 px balls looked the same (no seam detail at that size); joins
    1.7/2.7x vs 1.8/2.8x. So: small props get WORDS only (the Blender shape and colour carry them); a prop picture only
    for a large, detailed prop, and never with the dev model until tested again.
  - **Confirmed 2026-10-26 (masked, built-in kit, seed 303, final, distilled):** with the panel clip 3 had a still
    giant ball in the frame's bottom-right corner (r20261026-035007-8d6b); without it none (r20261026-035011-a791).
    Cause (docs/prop_control_research.md): the sheet guide shares the video's (t, x, y) positions, so the sheet's
    bottom-right panel lines up with the frame's bottom-right. Levers: props readable in the LAYOUT (48-64 px on
    screen, 16-24 frames in view, clear value contrast; 25 px balls came out as dots), short prop words, no panel for
    small props; a panel for a large prop only with an attention mask (untested).
- Import a .blend (collection `Mannequin`, rig `MannequinRig`, a camera) or a stand-in video in the Templates tab.
- A real 3D character instead of the mannequin: bpy_jobs/swap_character.py bakes the mannequin's motion onto a rigged
  character with the cinestudio bone names.

## 6. Backgrounds (Blender shots)

- One library (`<home>/backgrounds`, the Backgrounds tab), two kinds:
  - **360 panorama** (pano.png): any camera, also orbits. Qwen-Image-2512 GGUF Q8 + the qwen-360-diffusion LoRA (MIT;
    trigger "360 degree panorama with equirectangular projection"; a photographic realism finetune). Ground default
    `depth` (MoGe-2 360 depth -> a mesh of everything nearer than a dome wall); `--background-ground dome` = the old dome.
    `background new|depth|style`; a still camera looks along +Y = the panorama's u 0.25 (roll the panorama, and its depth
    .npy by the same fraction, to put the sun behind the camera).
  - **Backplate** (plate.png + the camera it was made for): ONE picture at the frame's shape for a STILL camera, 4-9x
    sharper than the part of a panorama that camera sees. `background plate --scene S --shot SHOT.blend`, `background
    import FILE` (your own picture); a run that makes a plate from text saves it in the library. A plate made for another
    camera warns (the horizon may not meet the floor).
  - **Backplate over the template's own set (2026-10-24):** `background from-set --scene "the place" --shot
    TEMPLATE.blend --style "<the characters' style words>"`: Blender renders the static set alone (bpy_jobs/render_set.py:
    mannequins, `Props` and rigs hidden; Workbench clay from the still camera), then Qwen-Image-Edit redraws it as the
    place, keeping every object's place and outline; stored as a plate (source "set", set.png beside it). Use it when
    the template has its own set: the picture then agrees with the layout. A template with only a floor gains nothing
    (use `background plate`). Option B (the picture projected onto the set meshes) is not built yet: today the run
    hides the Set and projects the plate onto a floor + far wall, so a set object nearer than the character is painted
    behind it (keep set objects behind the characters).
    - Geometry lessons (measured as the edge fit to the set render, background.edge_fit; ~0.5 is good): the edit graph's
      FluxKontextImageScale CENTRE-CROPS to 17 fixed sizes and the text encoder rescales to ~1 MP in multiples of 8, so
      the set is rendered at background.edit_layout (an encoder-stable size, the frame's exact shape, other shapes
      padded) and sent without the crop node (fit 0.26 -> 0.52). Qwen can still draw the whole picture a few % smaller
      or shifted (v3: x0.96 and 44 px lower): background.align_to_set fits it back by ECC (fit 0.26 -> 0.46).
- **Built-in backgrounds (2026-10-25):** `backgrounds/library/<name>/` (plate.png or pano.png, bg.json without personal
  paths, set.png for a plate drawn over a set) ship read-only like the built-in styles and templates; the user's
  library (<home>/backgrounds) comes first and a user background of the same name wins. Delete, style stamps and saves
  under a built-in's name are refused; a built-in panorama's depth goes to <home>/cache/pano_depth; a redraw is a new
  user background. `saved:NAME` works with a built-in. A template's shot.json `"background": NAME` is its DEFAULT: the
  Video tab preselects it while the Background choice is at its default, `run new` / MCP `new_video_run` use it when no
  background option is given (`--background none` = the template's own set); templates without one are unchanged.
  The cards link both ways (the backplate's "drawn over the set of T" opens T; T's "Backplate: NAME" opens it).
- Use: `run new --background saved:NAME` (a panorama or a backplate), `--background text --background-text "..."` (a new
  panorama), `--background plate --background-text|--background-image`. Or no panorama: `--look-scene "the place"`
  (a second Qwen edit puts the place into the look image).
- The sun BEHIND the character gives a hazy silhouette: choose the light direction for the shot.
- A panorama redraw with an LTX style LoRA (`background style`, keep 0.15-0.3) changes little and invents details.
- Style harmony (docs/style_consistency_research.md): put ONE style into the images (backplate/panorama, sheet, look
  image) before LTX sees them; the LTX LoRA and the style text cannot repair a pasted-in look afterwards.
- **Make every asset of a shot in the same style (user rule, 2026-10-24): background, characters and props.**
  - **Use the STYLE LIBRARY (2026-10-25):** `style list|show|new|edit|delete|set-default` (cinestudio/styles.py; MCP
    `list_styles`, `set_character_style`; built-ins in styles/library: `3d_animated` (default), `anime_cel`; the user's
    copies in <home>/styles win, e.g. 3d_animated + the 3dsrx LoRA at 0.4). Give assets a style by NAME, not by copied
    words: `character use-style NAME STYLE`, `background new|plate|from-set
    --style-preset STYLE`, `run new --style-preset STYLE` (words + the style's LoRA; without words or preset a run takes
    its character's style). A run records `style_warnings` when its characters or background name another style. A
    TEMPLATE has no style (user, 2026-10-25): it is grey clay geometry and motion, reusable in any look; keep its Set
    neutral grey and let the background and the video set the colours. Editing a style raises its version: assets made before are reported as older (pictures never change alone).
  - **A style reference made by the video model (2026-10-25):** `style make-reference NAME [--subject T] [--seeds ...]`
    (GPU queue; LTX-2.5 + the style's LoRA draw a still from text, workflows/ltx25_txt2img; ~30 s each on the 3070), look,
    then `style use-reference NAME FILE [--crop x0,y0,x1,y1]`: Qwen conversions (skins) get only the crop as Picture 2.
    One A/B (Kitsune -> 3d_animated, one seed, .cinestudio/gauntlet/lora_ref): with the crop the conversion was more
    saturated and contrasty with clearer fabric, closer to the LoRA's look than without a reference; moderate, not
    blind-tested. LTX framed "full body" as a medium shot: check the crop box.
  - Style words: take the shot's characters' own `style` field (character.json) for the backplate/panorama `--style`,
    and the same words in the run's style; drop meta words ("premium", "polished", "8K": they invite a lit set).
    Characters in one shot must share one style (convert a copy first: section 4.9).
  - Props: name each prop with the same concrete words everywhere (the background text, the video text, the look text):
    "plain red rubber dodgeballs". Show the same prop design in the background where it fits the place (a cart of the
    same red balls at the far wall of a dodgeball gym): it tells LTX what the props look like, in the same style.
    Keep loose copies out of the play area (they look like the moving props). In the Blender layout give each prop its
    main colour and a readable size (section 5 props).
  - Light: the same light logic in the background text, the style words and the video text ("evenly lit by soft
    neutral light from the front" when the characters' style says so); a backlit place gave hazy silhouettes.
  - Check: compare the finished backplate with the character sheet side by side (palette, shading, line style) before
    a run; make it again with the shared words when it looks photographic or flat 2D.
  Not A/B-tested yet: the effect of the prop cart in the background on the props' look in the video.

## 7. Videos (Layout-To-Render, LTX-2.5 distilled)

### 7.1 The recipe (the defaults of `run new` for a Blender shot with a library character)
- The Blender render of the shot (with the background) is the LAYOUT guide; a LOOK image (Qwen edit of Blender frame 1:
  mannequin -> the character, its description and `--look-text`) at frame -1 sets the palette and materials; the
  Ingredients 2.5 IC-LoRA at 1.0 carries the sheet; no skeleton. Keyframe looks `auto` (one per clip at its middle when
  the body turns >= 30 deg; a back-view clip uses the sheet's back panel; side views add the sheet's side panel).
- **The look image keeps the room (`look_pin`, default for new runs since 2026-10-26; cinestudio/look_pin.py,
  docs/look_image_research.md):** the plain look edit redrew the WHOLE frame and, by chance per run (Qwen is not
  reproducible on this PC even with identical inputs and seed), lost the hoop, a printed banner and the EXIT signs;
  the video followed it (frame 0 is 12.8/255 from the look, 19.7/255 from the layout). Now: the frame goes to Qwen at
  1344x784 without the Kontext centre crop (every old look was ~0.7% smaller and shifted); the figure mask (dilated
  3% + a soft ring + a contact-shadow ellipse) is a latent noise mask (SetLatentNoiseMask + DifferentialDiffusion, no
  pixel paste); a CPU gate (ECC fit, SSIM, edge recall, lost tiles outside the figure, at 768 px) makes the look again
  with seed+1 (at most 3 tries, the best kept; state `look_checks`, events `look_check`). Gate limits from 4 old looks:
  kept rooms SSIM 0.957-0.966 / recall 0.83-0.86 / 15-21 lost tiles; the failed one 0.770 / 0.64 / 41. Old manifests
  keep the plain edit. Several characters still use crop and stitch.
  **Never give LTX a look that lines up with the layout pixel for pixel:** the first pinned final came out ~35%
  darker (banners blank at final). Diagnosis (2026-10-26, preview clip 1, seed 303, fixed look files,
  .cinestudio/gauntlet/look_pin_ab/RESULTS.md): aligned look 27.9 / 27.0 mean brightness (the same file twice: LTX
  repeats itself within 1.6/255), the old unaligned look 42.7, aligned + moved 6/4 px 39.9, aligned + resized 39.5,
  new figure on the old background 39.8, aligned + blurred 30.5. So the cause is the exact alignment, not the VAE
  texture. The kept look is now moved 6 px right and 4 px down at 1344x784 (look_pin.offset, state `look_offsets`,
  event `look_offset`); the gate's tries keep the unmoved picture.
- Long shots: 121-frame clips with 25 shared frames; latent relay (clip k-1's last latents as a guide), `look_chain prev`
  (the previous clip's frame at this clip's start as its look, sharpened to clip 1's sharpness x1.5 preview / x1.0
  final), `chain_light tone` (lightness, chroma and contrast of the background held to clip 1's own first frame:
  `chain_light_ref clip1`, default since 2026-10-27; old runs held it to the LOOK image, which the video never reaches
  (clip 1 ~8 L darker), so every join was pushed up by the full +10% clamp and the scene got brighter: 308e grey 26.5 ->
  28.0 -> 32.3, the floor 25.5 -> 36), AdaIN 0.3 toward clip 1's latent. **The stitch grades each join (`chain_grade match`, default since
  2026-10-27; runner._grade_clips):** LTX makes every clip ~0.3-0.6 L lighter or darker than the still it is given,
  each clip flat inside, so the steps sit at the joins and no input guard removes them (r20261027-005603-a08c:
  background L 10.53 -> 10.80 -> 11.40). Clip k's luma x ONE gain so that the background mean of its 25 shared frames
  matches clip k-1 as graded (cumulative, +-15%; clip 1 unchanged; graded copies in work/graded/, the clips kept).
  Measured on a08c (mean grey clip 1/2/3): 27.5/27.9/29.3 -> 27.5/26.4/26.6; floor 25.4/26.9/28.9 -> 25.4/25.6/26.4.
  Pitfalls: grade in the clips' own YUV form (an RGB copy next to YUV clips read ~4% brighter in the stitch); match
  the plain MEANS (8-bit percentiles are too coarse; dark/light halves followed lamps and signs: a 0.83 gain; halves
  split at the median are biased). Limit: one gain cannot undo a changed spread (the later clips have a lighter floor
  and a darker wall: after the grade the wall reads ~10% darker by clip 3). Joins are flow cross-faded.
- **Test runs in PREVIEW** (stage 1 only, 768x448, ~half the time, below the 8 GB peak); render one chosen setting at
  final with the same seed for a demo. 1024x576 runs out of memory on 8 GB.
- Write the costume's garment lengths into `--look-text` when the description and the sheet disagree (the look image
  decides the costume: a poncho became a capelet).
- A `--note "what this run tests"` is shown in the app with the settings that differ from a default run.
- **Two video models** (`run new --video-model distilled|full`, the app's Video model select, MCP `video_model`):
  `distilled` (default, ltx25_layout[ing]: 8 steps, CFG 1) or `full` (ltx25dev_layout[ing]: the dev model, 30 steps,
  MultimodalGuider CFG 3 + STG, the negative; `--steps`/`--guidance` apply). Full is about 8x slower in preview (~20 min
  a clip on the 3070: 4 guider passes x 30 steps vs 1 x 8) and ~5x at final (estimate from measured s/step). It is
  kept in parity by the generator and tests/unit/test_model_parity.py, but NOT A/B-tested against distilled yet (plan:
  one blind A/B in the first RunPod session, same shot and seed). It needs the dev transformer (`models download
  --feature video`); forks and retakes keep the run's model.
- **Final output = the distilled model (user, 2026-10-25);** the full (dev) model is the optional "higher quality"
  path, kept up to date and in parity.
- **Stage 2 of the full model (2026-10-25):** Lightricks' two-stage recipe (ltx-pipelines TI2VidTwoStagesPipeline):
  the dev transformer + `loras/ltx-2.5-22b-distilled-lora-450-bf16` (node `lora_dist`, strength 1.0 = their default;
  the CLI example uses 0.8; binding `stage2_lora_strength`), CFGGuider cfg 1, the distilled stage-2 sigmas 0.909375,
  0.725, 0.421875, 0 with plain euler (3 model calls instead of the old 6 steps x 2 CFG passes). Stage 1 keeps the dev
  guider. One graph for both tiers: on 8 GB ComfyUI's dynamic VRAM memory-maps every safetensors file (fast_disk) and
  streams the weights per layer, so the 8.9 GB LoRA needs no second resident copy (the dev run already streamed 20.5 GB
  + 480 LoRA patches with ~3.5 GB RAM free). The manifest's `stage2` and each clip's `stage2` say which stage 2 ran
  (`distilled`, `distilled-lora`, or `guided` = the old MultimodalGuider stage 2 frozen in older runs); a preview has none.

### 7.2 Prompts (LTX)
- New runs use the Ingredients 2.5 headings ("### Reference Sheet Description ... ### Target Description ...") with
  the SHORT target (default since 2026-10-26, user: "let's go with C's format"): the style, "The character from the
  reference sheet.", then the clip text, the props, the sound. The character is named in the clip text by 2-3 key
  visible marks (the same words every time); the pictures (sheet, look image) carry the rest of the identity.
- One paragraph of present-tense prose, ~4-8 sentences: style, shot + place + light, the character, the action in time
  order, the camera, the sound; restrained words. Name the SAME place and light as the background. Never name the
  control type ("depth map", "skeleton"). Text that contradicts the layout can cause cuts.
- **Text in the set: quote it in the prompt, every sign (2026-10-26, user: "call out the text in prompting").** LTX's
  VAE packs 32x32 px into one latent cell, so letters ~20-40 px tall at final are about one cell: without the words
  the video guessed them again ("Pomfy", "Ppen Source", EXIT signs as "SAIT"), although the look image and the layout
  had clean text. Final runs, seed 303, the same look files (r20261026-110407-838f / -165726-39b8 / -191306-6c13):
  no text named -> both banners wrong; the banners quoted -> both right in all 3 clips, but the unquoted EXIT signs
  copied the banner words ("COMY", "ETW") and the video was ~15% darker; the banners AND `small green signs above both
  doors read "EXIT"` quoted, light "evenly lit by soft violet light" -> every sign right in all clips, ~5-9% darker
  than without text. Quote EVERY visible sign (a sign left out takes another sign's words). A built-in template
  keeps its sign text in `clip_context` (shan_shen_bu_balls since 2026-10-26).
- **Every segmented shot gets ONE PROMPT PER CLIP (the default; user rule, 2026-10-25).** A shot longer than one clip is
  cut into clips (clip k = frames 96(k-1)+1 .. +121, 25 frames shared). Each clip's action text describes ONLY what that
  clip's frames of the Blender render show, in time order: read the rig timeline (`shot timeline NAME`: events with
  frames) and the template's events, and map them to the clips. Never describe an earlier or later clip's action in a
  clip ("whatever happened in clip 1 should not be described in clip 3"). Measured: one shared action text for all 3
  clips (dodges + kicks + catch) went with a profile walk at the dodges and a drifting clip 3 (runs of 2026-10-25),
  where an earlier run with a dance-only text followed the dodges.
  - **Write it in the style of LTX's training captions (prompt A/B 2026-10-25/26, "text B" + the short target = "C";
    docs/multiclip_prompting_research.md).** Every clip's prompt = the SHARED CONTEXT (word for word the same in every
    clip) + that clip's ACTIONS (user: "context that is needed throughout each clip should still be present but actions
    specific to that scene should not be used in other clips").
    - Shared context, in this order: one shot type + the viewpoint + the camera EXACTLY as the layout shows it ("A wide
      shot, front-facing, the camera remains static:" for a still template camera; never a move the layout lacks); the
      character's 2-3 key marks; the place; ONE light logic; the facing that holds ("faces the camera the whole time");
      the MOTION STYLE of the whole shot = the moves it uses, with pacing words ("dances the flash step: quick sideways
      slides with hard stops, small fast steps, knee lifts and foot flicks, loose arm swings"). Props are named by
      their look by the props sentence (the template's layer words), not by what they do.
    - Clip actions: one or two sentences, about 15-40 words, BODY MECHANICS only (verb + direction + stop), in time
      order with "then"; no timestamps, no numbered beats, at most about three beats. Directions: sideways moves in
      PICTURE directions (read from the rig: hips x velocity; world +X = the picture's right for a camera looking along
      +Y), limbs by the character's own side, with the picture side where it helps (user: "model's direction is
      important to call out, not just events"). Name the facing and the main moves, not every flick.
    - **Name the moves whose DEPTH a front view cannot show (user rule, 2026-10-26).** The layout is a flat picture:
      overlapping limbs and moves toward or away from the camera are ambiguous, and LTX can read them as a body turn
      (the turn-away) or as legs that swap sides (mirrored motion). Call them out in that clip's actions: limbs that
      CROSS (which limb, in front of or behind the other, toward which picture side: "steps the left foot across in
      front of the right foot toward the left of the picture, then steps back out"), a knee, foot or hand that moves
      toward or away from the camera ("the knee lifts toward the camera"), a step or lean forward or back, a hand that
      goes behind the body or the head. Read the depth from the rig, not from the render: the template's timeline.json
      has every frame's head, hips, hands and feet; for a camera looking along +Y a smaller y is NEARER the camera, and
      crossed feet = foot.L x < foot.R x while the body faces the camera (`shot timeline` has no cross event).
      Example: shan_shen_bu_balls crosses the feet at frames 95-102 (the clip 1/2 join), 129-136 (clip 2, 26 cm) and
      213-223 (the clip 2/3 join: the left foot in front, toward the picture's left); no clip text named them. Keep
      about three beats: drop a less important beat (a prop story) for it. Not A/B-tested yet.
    - **No cause or threat stories for objects the layout already moves.** Bad: "Three times a dodgeball flies straight
      at the head; each time the dancer slides sideways at the last moment ...". Good: "The masked dancer slides toward
      the right of the picture, stops hard, slides back toward the left, stops, slides right again, then flicks the
      left foot out to the side." (The balls come from the layout and the props sentence; at most "a ball passes beside
      the head".)
    - Clip k >= 2 opens on the STATE that its shared frames show ("Still facing the camera, with the left foot coming
      back down, ..."), then gives only its new actions. A move that ends inside the 25 shared frames is a verb in clip
      k-1 and a state in clip k (a state left by an earlier clip: "holds the ball in the raised hand", not "catches the
      ball"). From the research; not A/B-tested yet.
    - Restrained, observable words: no intent or emotion words, no "explosive"/"dramatic", never "clay", "mannequin" or
      "3D render of"; keep the character's words fixed (`{character}`), no pronoun or synonym swaps.
    - Evidence (.cinestudio/gauntlet/prompt_ab, masked on shan_shen_bu_balls, preview clip 1, paired seeds 202/303/404,
      the same look image per seed): text B (above) vs text A (motion style + the ball-threat clip text): DWPose bone
      0.779/0.798/0.786 vs 0.761/0.793/0.753 (B higher on every seed, mean +0.019; arms +0.021); the frames look almost
      the same per seed. The seed decides most of the motion: the turn-away (the back to the camera at frame ~30 and
      ~78-92, where the layout faces the camera) came with seed 202 under every text (P, A, B) and not at frame 30 with
      seed 303. So the text is a small lever: try 2-3 seeds per clip (7.5) for a turn-away. C (B + the short target,
      174 instead of 355 words in the Target part): bone 0.785/0.799/0.787, the same frames and the same character
      look as B; the user chose C's format (seed 303 "seems pretty solid"). Not yet tested on a whole 3-clip run
      (colour drift and back views without the costume words).
  - Sources, in order: the run's `--clip-video K "..."`, else the template's `clip_texts` (shot.json, written once by the
    template author from the timeline), else `--clip-text auto` (the rig's texts fill only clips still empty), else
    write them yourself from the timeline before `run new`. The main
    `--video` text carries the shared context (place, light, look, lasting props and states); the actions live in the
    clip texts.
  - **Write the main `--video` text WITHOUT single-clip actions** (it goes into every clip). Good (text B, for a
    template without such a `clip_context`): "A wide shot, front-facing, the camera remains static: a masked dancer with a white oval mask, a short dark purple jacket
    and a black bodysuit in a dim school gym at night, lit by dim violet light, with a glowing LED scoreboard and a
    framed banner on the back wall. The masked dancer faces the camera the whole time and dances the flash step: quick
    sideways slides with hard stops, small fast steps, knee lifts and foot flicks, loose arm swings." Bad (it was used
    on 2026-10-25 and leaked into every clip): "... dodges each ball with a sudden sideways step, kicks two balls away,
    and at the end catches one ball ...". The motion-style sentence is shared context: `run new`'s `prompt_warnings`
    names its move words ("slides", "flicks") anyway; that warning is expected for it. Move every other single-clip
    action it names into the clip texts before you start the run. (A template's `clip_context` carries the shot type,
    the camera words, the facing and the motion style: then `--video` needs only the marks, place and light.)
  - How the code builds it (runner.clip_video_text, manifest `clip_join: "context"`): clip k's video text = `--video` +
    the clip context + clip k's actions, then the usual target part (style, the character's marks, place, props,
    sound). Template authors: `shot clip-texts NAME --clip K "ACTIONS" ... --context "{character} faces the camera the
    whole time"` (MCP `template_clip_texts`; built-ins only through the repo's shot.json; `setup/export_template.py`
    keeps them); `{character}` becomes `--character-words` (default "the character"; a name keeps its case inside a
    sentence; runs, retakes and forks all fill it). Template texts apply only to the
    default split (121-frame clips, 25 shared); `--clip-context ""` drops the template's context. The run records
    `clip_sources` (run|template|auto|none) and `prompt_warnings` (all clips share one action text; clips without
    actions; single-clip action verbs such as "kicks", "catches" found in `--video`), shown by `run new`, `run_details`
    and the app. A retake's `--video` replaces only that clip's actions; a fork keeps every clip's actions, those of
    the take in use after `run use-take` (`run fork --clip-video K`, or the app's per-clip fields under "Build on this
    video", changes those of the clips it makes again). Old runs (no `clip_join`) keep the old rule
    (a clip's text replaced the run text). The app's Text per clip opens for multi-clip templates, prefilled with the
    template's texts, and asks before Generate when a clip has no action text.
  - Facing words are recommended where the layout keeps the body frontal ("faces the camera the whole time"). Camera
    words say only what the template's camera does ("the camera remains static"); never add a camera move.
  - Check before a run: print each clip's final prompt and compare it with that clip's frames of the layout.
  - `run new --target-text short|full` (MCP `advanced.target_text`; default `short`): short = the Target part keeps
    only "The character from the reference sheet." (no second copy of the whole costume, no "The place: ..."
    sentence), so the `--video` text (or the template's clip context) carries the 2-3 key marks, the place and the
    light; when the clip text names none of the description's key marks, runner.names_marks fails and the key marks
    (runner.target_marks) are added after "The character from the reference sheet". full = the old target. Old
    manifests without the key build the full target. Aim at ~120-180 words in the Target part
    (docs/multiclip_prompting_research.md; .cinestudio/gauntlet/prompt_ab/RESULTS.md).
- **Sound:** the Layout LoRA pushes the audio to silence; every clip text gets an audible sound sentence (`--sound-text`,
  concrete: a breeze, cloth, footsteps; never "quiet"). Music by text is made per clip and changes at the joins: use a
  library song (`--song NAME --song-start S`, or the template's own sound) for music, or `film assemble` (section 8).
- **Audio carry (default for layout runs since 2026-10-24; `--audio-relay off|carry`):** before, every clip started from
  an empty audio latent (loudness steps of 1-20 LU and timbre jumps at the joins). Now each clip saves its stage-1 audio
  latent; clip k freezes clip k-1's tail (the 25 shared frames = 26 audio latents, cut by cinestudio/latent_file.py) at
  its start; stage 2 keeps the stage-1 audio frozen; the stitch switches from clip k-1 to clip k 0.1 s before the end of
  the shared frames with a 40 ms fade (an equal-power fade of the same sound added up to +3 dB). A take put in for clip
  k-1 makes the joins fall back to the old fade. Not yet A/B-tested on a real run. Open: clip 1 and the last clip fade
  ~30 dB inside themselves (cause unknown; docs/audio_consistency_research.md 9.1).
- The distilled model runs at CFG 1: the negative prompt does nothing; steps and guidance are refused.
- Style text and LoRA: the style sentence changes the result (an anime sentence beat a 3D sentence on motion in one
  seed); keep one style source (images first, 6.).

### 7.3 Several characters in one video (2 to 4; 2026-10-25)
- The manifest keeps `characters` [{name, collection, rig}] in slot order (character_name / character_collection / rig
  mirror slot 1; old runs with `character2` are read by runner.run_characters / run_views / run_group). CLI: repeat
  `--character`; MCP `new_video_run(characters=[...])`; the app's New video form shows ONE character select per
  template slot (its Mannequin* layers, /api/shots "mannequins", numbered markers) and sends them all; more than 4 slots
  disable Generate. The same character may fill two slots (twins: a note, not a refusal). The one-mannequin copy
  (`--spacing`) is CLI/MCP only. Forks keep the characters (not a fork field).
- Layout only: every character's body (one duplicate_mannequin job, which also reports each rig's picture x at frame
  1: `mannequins.screen_x`) and proxy kit (ONE proxy_kit.py job: `--kits`; a single kit keeps `--spec`), ONE group sheet
  (runner.group_views: 2 characters front/right/back, 3 front/back, 4 front only, so at most six full-body panels; the
  groups LEFT TO RIGHT as they stand in the picture, each at its real height ratio; a prop column gives way first,
  then the bodies: 3 characters + a prop measured body scale ~0.78).
- The look image by CROP AND STITCH (runner.plan_figure_crops): the figures are the runs of occupied mask columns,
  each slot assigned by its screen x; each figure cropped, enlarged, edited with its own single-character look (its
  front view, its description) and pasted back: one Qwen edit (~4 min) per figure. Whole-frame edits drew a second
  character at the wrong size and dropped the first one's hair. FALLBACK for figures that touch (no gap in the mask):
  one shared crop edited once per figure in turn, each instruction naming its place ("Replace the white wooden
  mannequin on the left ..."); the run's look step records `look_fallback: "shared crop"` (not GPU-tested yet).
  The paste covers only the figure (runner.stitch_look: its mask grown by 3.5% of the box, feathered) and the crop
  instruction names "the background behind the figure ... as in image 1", never "the sky": in trio_test run
  r20261025-061453-0da4 a crop of the gym wall came back as a blue SKY, the whole-box paste put a sky patch behind the
  left figure and LTX kept it for the whole clip. A changed background INSIDE the mannequin's silhouette (between the
  kit's ears) still shows: check the look image.
  Measured (that run, preview, 3 Qwen edits ~7 min each + a 2 min clip): every figure kept its identity for the whole
  clip (kitsune / masked / kitsune, no swap, each dances its own motion); the masked figure lost its plum hair in its
  crop edit (as in the duo runs: a whole-head mask hides it). The same run with the new instruction and paste
  (r20261025-084344-7e2a, same seed): the crops kept the gym wall, no sky patch, identities held again; the masked
  figure's jacket came out sleeveless and still without hair (fix by --look-text or a retake look).
- The clip text (runner.group_sheet_text / _group_target): "**Panels 1-2 (Character 1):** ..." left to right, and in
  the target "Three characters from the reference sheet side by side: on the left, the first character, <key marks>;
  in the middle ...": two characters keep their full descriptions (exactly the old duo text), three or more get key
  marks (runner.target_marks: the description's first clause, cut at a comma to <= 24 words). Keyframe looks are off
  (they know one character). Check the look image against the mask before the clips: a figure of the wrong size or
  place makes the video morph or float a mask.
- Open problem: at an arms-crossed end turned 30 deg outward, LTX turned both characters much further (profile, back):
  give the masks marks (4.7), keep end turns small.

### 7.4 Known limits
- Colour drift through the chain (a black suit to bronze in later clips): the fixes in 7.1 reduce it; a FRESH look of a
  clip's start frame (`run retake --look-image FILE --keyframe off`) repairs one clip (Qwen cannot draw a back view).
- In orbits the body turns ~30 deg ahead of the camera; LTX may invent props the layout does not show.
- The pose guide (`--pose-strength`) gave no gain and hurt joins: off. Also harmful in blind tests: `--chain-colour
  look|match` (a blue jacket, seams) and `--look-chain anchor` (the front view's background pasted into back views);
  AdaIN 1.0 in every clip (a black veil): use 1.0 only as a one-clip retake. `--clip-text auto` gave no visible gain.
- Video-to-video (start noise, skeleton drawing) remains for video templates and Blender shots without a library
  character: new runs use the `ltx25dev_*` graphs (the LTX-2.5 dev model). The LTX-2.3 graphs are retired: old runs
  stay listed and playable, and a start, retake or fork is refused (`runner.retired`).

### 7.5 The clip gauntlet (a long shot clip by clip until each clip passes)
- In the app: New video > "Stop after each clip"; the run player's Clips section shows each clip's takes, "Make this
  clip again" (one change), "Measure" (the picture numbers), "Use this take" and "Make clip K+1" / "Make the rest";
  the run card's "Continue" restarts a paused or stopped run. The CLI and MCP do the same (below).
1. `run new ... --seed N --note "..."`, then `run start RUN --through-clip 1 --detach`; wait for status "paused".
2. Measure: `run score RUN --clip K --critic DIR --tag A` (one anonymous tag per candidate; keep the tag -> take map
   OUTSIDE DIR). Look at the strip yourself.
3. Judge with a FRESH agent and docs/gauntlet_critic.md (never tell it the settings or which tag is the original).
   **Pass:** CHARACTER >= 8, MOTION >= 8, BACKGROUND >= 7, PICTURE >= 7, and no gross number: `overlap.x_median` <= 2.5
   (read `overlap.mean` too), `sharpness_vs_clip1` >= 0.5, `picture.max_jump_x_median` <= 2.5, `motion.mirrored` ~0 with
   enough `motion.rows`. The motion score's judge is DWPose: a mannequin-like body scores well; the critic decides.
4. Failed: `run retake RUN --clip K --seed N2 [ONE change] --note "why"` (first 2-3 seeds, then one knob:
   `--keyframe on|off`, `--ingredients-strength`, `--adain`, `--video`, `--look-image`, `--refine-sigmas`).
5. Winner: `run use-take RUN --clip K --take N` (only while the worker is stopped); next clip `--through-clip K+1`; the
   end `--through-clip 0`, then judge the whole final.mp4 and each join.
6. After ~4 takes without a pass, STOP: record every take, keep the run paused, report the best take and the suspected
   cause. Write each round to `.cinestudio/gauntlet/<name>/RESULTS.md`.
- `run fork RUN --from-clip N` = a new run that keeps clips 1..N-1 (the app's "Build on this video"). Forks copy only the
  fork fields: add every new chain option to `FORK_FIELDS` and `_fork_locked`.
- Tools: `python setup/brightness_check.py RUN` (background L per clip; goal < 0.5 L drift), `setup/join_check.py`,
  `.cinestudio/gauntlet/ltx25_ab/tools/measure.py` (saturation, frame change). Read video frames with
  `video.frame_png` (exact decode; OpenCV reads are 0.5 L bright).

## 8. Music

- **A film of several scenes with ONE score (2026-10-24):** never let LTX make the music (each clip, and each scene,
  invents new music). Render each scene with a sound text for its ambience and visible actions only ("no music, no one
  speaks"); make ONE score for the whole film (`song make ... --seconds <film length>`, instrumental); then
  `film assemble NAME --scene RUN --scene RUN ... --score SONG [--music-db -8] [--no-duck]` (cinestudio/film.py): hard
  cuts, each scene keeps its own sound, the score runs under the whole film from 0 (it never restarts at a cut), ducked
  under strong effects (sidechain), faded at the end, one loudness normalization (-16 LUFS). <home>/films/NAME/film.json
  is the cue sheet (each scene's start). Fix the scene lengths first, then fit the score (beats at the cuts and gags).
  Accents on exact frames (stingers at rig-timeline events) are not built yet.
- `song make NAME --tags "..." --bpm B --seconds N` (ACE-Step, MIT): describe traits, never song or artist names.
- `song beats NAME`: measured beats (time a dance on the MEASURED beats, not the asked bpm); look at a spectrogram (the
  song's stops and end). `song loop NAME --beats 16 --frames 240` cuts a seamless loop (16 beats at 96 bpm = 10 s).
- `song place SHOT.blend NAME --start S --out OUT.blend` puts it into a shot; the run packs it.
- Blender's mixdown does not work in background mode: the run mixes the sound strips with ffmpeg.

## 9. The app (for agents that change it)

- Server: cinestudio/webapp.py builds the app; the JSON API is the package cinestudio/webapi/ (one module per area,
  `@routes.get/post`). Page: cinestudio/web/index.html loads classic scripts in order from web/js/ (core, characters,
  video, music, templates, figure, backgrounds, settings, styles, workflows, main last); top-level names must be unique; `node --check`
  each changed file. The page and scripts are served with Cache-Control no-cache.
- Do not test the live app with headless Edge `--virtual-time-budget` (it flooded the server); a plain screenshot is
  taken before the data arrives. Render a static copy of a dialog to check its layout.
- Canceled runs, songs and templates are deleted on the next list, but never a run with a final.mp4.
- **Workflows view (advanced mode, off by default; Settings > Advanced or Ctrl+Shift+G):** web/js/workflows.js +
  webapi/workflows.py draw the ComfyUI graphs. Every submit saves its exact graph: runs in <run>/work/graphs/<job>.json
  (Run._comfy, after the run-time changes), other jobs in <home>/cache/graphs/<tag>.json (newest 200; + .meta.json).
  A new graph or feature: add it to `FEATURES` in webapi/workflows.py. Keep the normal screens simple (user).
- **User graph copies (cinestudio/user_workflows.py):** the built-in graphs in workflows/ are read-only; "Make a
  copy" writes <home>/workflows/<builtin>/<id>.json (+ .meta.json, history/ newest 20) and choices.json says which
  graph NEW jobs use. ProfileWorkflow loads the chosen copy for workflows/ only (never for a run's frozen folder); the
  bind map still applies, so bound inputs and links are read-only in a copy, and a copy that no longer fits the
  built-in is refused. `run new` copies the chosen copy (runner._copy_graph), records it in manifest `graph_sources`
  (chip "Custom graph"), keeps its values over the tier's node settings; a fork keeps the parent's frozen copy.

## 10. Setup, models, tokens, RunPod

- Model tiers (`setup/models.json`, `models download --tier 8gb|large`, Settings > Quick setup): every file pinned
  (commit, size, SHA-256); the tier sets the precision; the tier is frozen into each run. The layout video uses the
  distilled LTX-2.5 int8 transformer in both tiers; the 8gb tier's Qwen edit file is int8 ConvRot. Tiers: `8gb` (tested:
  RTX 3070 + 32 GB RAM) and `large` (bf16; an 80 GB+ GPU and ~120 GB RAM; was `24gb`, the old name still works). A tier's
  `graph_settings` are written into a run's graphs (the large tier: VAE decode tiles 1024/256/128/48).
  `models civitai VERSION_ID` downloads a Civitai file (token optional). Sizes with every feature: 8gb ~156 GB,
  large ~234 GB. A tier's `optional_features` are downloaded only when named: the 8gb tier's `full_stage2` (the 8.9 GB
  distilled LoRA that only FINAL runs of the full video model load; part of `video` in the large tier). The LTX-2.5 dev transformer and the LTX-2.3 union LoRA serve only video-to-video and the (off) pose
  guide. The SAM 2 node reads only <ComfyUI>/models/sam2 and, at fp16, only `sam2.1_hiera_large-fp16.safetensors`.
- RunPod GPU (docs/large_gpu_speed_research.md): RTX PRO 6000 96 GB (best price per clip), H100 SXM (~6% faster, 60%
  more per hour). Not A100 (no fp8; int8 is slow there), not 4090/5090 pods (35-41 GB RAM). Do not add `--highvram` or
  `--gpu-only` (they turn off dynamic VRAM). int8 vs bf16 on the large tier: blind A/B in the first pod session.
- Tokens: `hf` (gated LTX files), `comfy` (Comfy Cloud, optional), `civitai` (optional): `settings token set NAME`.
  The app HIDES the Comfy Cloud key and the Run on choice (user, 2026-10-25; web/js/settings.js `SHOW_COMFY_CLOUD`):
  Cloud's support of the 4 LTXVideo pack nodes and the gated IC-LoRAs is unverified (docs/comfy_cloud_compat.md).
- RunPod: docs/RUNPOD.md (`bash setup/install_linux.sh --tier T` inside tmux on a stock PyTorch CUDA 13.0 pod with a
  network volume; not yet tested on a pod: its section 6 lists the first-session checks). The code reaches the pod by
  `git clone` of the private GitHub repo (MachineLearningAmateur/cinestudio, branch `main`, a read-only token). The
  script pins torch (setup/runtime.lock.json `torch.linux_versions`; default cu130 = 2.12.1: the cu128 index ends at 2.11).
  Pod facts (static audit 2026-10-26): a container sees the HOST's RAM in psutil, so RAM checks read the cgroup limit
  (sysinfo.total_ram); the locks need flock on the volume (the install checks it; locks.py raises other lock errors
  instead of reporting "busy"); the tunnel's local port must be 8765 (the app's Host check).

## 11. Development

- Tests: `.venv/Scripts/python.exe -m pytest -q` (no GPU, no Blender). Keep the JSON CLI contract stable.
- Distilled and full (dev) graphs are a pair (cinestudio/model_pairs.py PAIRS): change both through the generator
  (setup/make_layout_graphs.py, make_redraw_graphs.py, make_txt2img_graphs.py); register every run-time clip-graph
  change in runner.py (`@graph_feature` on its helper, `_pairs.register` for a bound value; a model-specific one with
  `only=` and a reason);
  tests/unit/test_model_parity.py fails on drift and on unregistered changes. A node only the full member has must be
  a model pass-through listed in model_pairs.FULL_ONLY_NODES (today `lora_dist`, the stage-2 distilled LoRA): links
  through it compare as links to its model input.
- Retired graphs live in .cinestudio/backup/workflows_retired_<date>/ (with the setup scripts that made them); a later
  release can add a formal retire/preserve step.
- Commit only when the user asks.
