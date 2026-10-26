# App feature parity review (2026-10-25)

Question: which features can an AI agent use through MCP or the CLI that a person cannot do by clicking in the web app?

How this was checked: a read-only pass over `cinestudio/mcp_server.py` (26 tools), the argument parser in
`cinestudio/cli.py`, every route in `cinestudio/webapi/*.py`, and the page (`cinestudio/web/index.html` and
`cinestudio/web/js/*.js`). Nothing was run. The Styles library (`cinestudio/styles.py`) is being built and was not
reviewed.

Short answer: the app covers the basic path well (character, mannequin body, template, background, video, song,
settings). The biggest gaps are the clip-by-clip quality loop (continue, retake, takes, use-take), a song as a video's
sound, region edits, repose, and the newest features (background over the set, template layers, film assembly, two
characters).

Every route that the page calls exists. No dead routes were found.

## 1. Feature table

"—" = not available there. App places name the rail tab first.

### Characters

| Feature | MCP tool | CLI command | App UI | Notes |
|---|---|---|---|---|
| Health check | `doctor` | `doctor` | Status bar dot (click for the report) | |
| LoRA list | `list_loras` | `loras` | Filled into the LoRA selects | |
| Queue view | `queue_list` | `queue list` | Status bar > Queue popover | |
| Cancel queued or running work | `cancel_run`, `cancel_job` | `queue cancel KIND REF`, `run cancel` | Queue popover stop button; run player Stop/Remove | MCP can cancel only videos and character jobs, not songs, templates or backgrounds. |
| Character list and details | `list_characters`, `show_character` | `character list`, `character show` | Characters shelf and view | |
| New character from text | `create_character` | `character create` | Characters > + > Text | MCP has no display name, body type or seed. CLI-only: `--seed`, `--width`, `--height`, `--style-lora`. The API accepts `seed`, but the form has no field. |
| New character from images or a sheet | — | `character import` | Characters > + > My images / Sheet | |
| Pick a candidate | `pick_candidate` | `character pick` | Candidates grid (click) | |
| Edit one image | `edit_image` | `character edit` | Click an image > Apply | |
| Edit only a box (`region`) | `edit_image(region=)` | `--region` | **MISSING** | The API already accepts `region`; the form has no way to set it. |
| Edit settings (denoise, steps, cfg, sampler, scheduler, shift, negative) | denoise only | `--denoise` + quality flags | Edit panel > Settings | |
| Reference image (Picture 2) | — | `--reference` | Edit panel > Reference image | |
| Undo one image | `undo_edit` | `character undo` | Undo icon in the edit panel | |
| Edit history | `show_character` (raw) | `character show` | History thumbnails | |
| Make the views and sheet | `make_views` | `character views` (+ `--distance`, `--face`, `--mode`, `--seed`) | Front view > "Make the views" | The button shows only before any view exists. |
| Redo one view | `redo_view` | `character redo-view` | Select a view > Redo icon | |
| Repose (AnyPose front, then views) | — | `character repose` | **MISSING** | A job action exists (`jobs.ACTIONS["repose"]`). The Mannequin view even says "the pose changes only after a repose". |
| Style pass of the sheet | — | `character style` | Finish panel > Style pass | |
| Finish (ready) | `finish_character` | `character finish` | Finish panel > Use as is | |
| Display name | — | `character display-name` | Click the name in the header | |
| Rename the ID | — | `character rename` | — | By design: the app changes only the display name. |
| Delete a character | — | `character delete` | Trash on the tile | |
| Mannequin body | `set_mannequin_body` | `character mannequin` | Characters > Mannequin view (presets, sliders, Save) | |
| Mannequin preview | `preview_mannequin` (draft body and draft kit) | — | Mannequin view (draft body, saved kit) | |
| Proxy kit (stand-in parts) | `set_proxy_kit` | `character proxy` | Read-only summary in the Mannequin view | **Intentional:** the user decided the proxy kit is AI-only. Not a gap. |
| Touch-ups (fixes in code) | — | — (Python `character.touch_up`) | History and undo only | Code-only by nature. |
| Job status | `job_status` | `job show`, `job list` | Job bar under the header | |
| Prepare a sheet image | — | `sheet` | Covered by the Sheet import | |

### Videos

| Feature | MCP tool | CLI command | App UI | Notes |
|---|---|---|---|---|
| New run from a Blender shot or template | `new_video_run` | `run new` + `run start` | Video > + New video | MCP takes far fewer options (see below). |
| Background in the run (text, saved, backplate) | — | `--background`, `--background-text`, `--background-image` | New video > 4 Background | MCP cannot set a background. |
| Look text, sound words, note, stand-in parts on/off, keyframe looks, sheet guide strength | — | yes | New video form | MCP has none of these, not even `note`. |
| A library song as the video's sound | `song`, `song_start` | `--song`, `--song-start` | **MISSING** | `runs._run_new` does not pass `song`. Songs made in the Music tab can reach a video only through a dance template. |
| Second character | — | `--character2`, `--spacing` | **MISSING** | |
| Per-clip text | — | `--clip-video K TEXT` | **MISSING** | AGENTS.md 7.2 recommends it for long shots. |
| Look scene (no panorama) | — | `--look-scene` | **MISSING** | |
| Audio relay | — | `--audio-relay off\|carry` | **MISSING** | The default (carry) is good, so this matters little. |
| Stop after clip K (clip gauntlet) | `new_video_run(through_clip=)`, `continue_run` | `run start --through-clip K` | **MISSING** | |
| Continue a paused or stopped run | `continue_run` | `run start` | **MISSING** | A paused run shows only the trash button. A run whose worker did not start stays "new" with no Start button. |
| Retake one clip | `retake_clip` (subset) | `run retake` (many knobs) | **MISSING** | |
| List takes | `list_takes` | `run takes` | **MISSING** | The player shows only the final video. |
| Use a take | `use_take` | `run use-take` | **MISSING** | |
| Score a clip | `score_clip` | `run score` | **MISSING** | |
| Build on this video (fork) | — | `run fork` (+ chain, adain, keyframe, relay, clip-text knobs) | Run player > Build on this video | The app sends only from-clip, text, style, facing and note. |
| Run status, list, events | `run_status`, `list_runs` | `run status`, `run list`, `run events` | Video gallery and player | |
| Cancel a run | `cancel_run` | `run cancel` | Player Stop/Remove | |
| Delete a run | — | `run delete` | Trash | |
| See the Blender render and control a run used | — | — | Player > Show the Blender render | App-only. |
| Shot preview | — | — | New video form | App-only. |
| Expert knobs (adain, look-chain, chain-light, relay, clip-frames, pose guide, refine sigmas, profile) | some in `retake_clip` | yes | — (v2v-only fields for video templates) | Keep these in the CLI. |

### Music and films

| Feature | MCP tool | CLI command | App UI | Notes |
|---|---|---|---|---|
| Make a song | — | `song make` | Music > + New song | |
| List and play songs | — | `song list`, `song show` | Music grid | |
| Delete a song | — | — | Trash on the song card | App-only (the CLI has no `song delete`). |
| Beats | — | `song beats` | Shown on the card (measured when the song is made) | Other start points, fps or plain sound files: CLI only. |
| Seamless loop | — | `song loop` | **MISSING** | |
| Put a song into a .blend | — | `song place` | **MISSING** | A song choice in the New video form would replace most uses. |
| Assemble a film with one score | — | `film assemble`, `film list` | **MISSING** | The films folder is not even served by `/files/`. |

### Templates

| Feature | MCP tool | CLI command | App UI | Notes |
|---|---|---|---|---|
| Dance on a song | — | `shot make-dance` | Templates > + > Dance on a song | |
| Template from a dance video | — | `shot from-video` | Templates > + > From a video | |
| Import a .blend or a stand-in video | — | `shot add`, `shot add-video` | Templates > + > Import | |
| Figure mask (video templates) | — | `shot mask` | Card > Select figure | |
| Layers and their words (Set, Props) | — | `shot layers` | **MISSING** | These words go into the background, video and look texts. |
| Motion timeline | — | `shot timeline` | **MISSING** | Used only by `--clip-text auto` (under test). |
| List, show, delete | — | `shot list`, `shot show`, `shot delete` | Grid and trash | |

### Backgrounds

| Feature | MCP tool | CLI command | App UI | Notes |
|---|---|---|---|---|
| 360 panorama | — | `background new` | Backgrounds > + > 360° panorama | |
| Backplate from text | — | `background plate` | Backgrounds > + > Backplate | |
| Backplate drawn over the template's set | — | `background from-set` | **MISSING** | |
| Your own picture | — | `background import` | Backgrounds > + > My picture | |
| 360 depth | — | `background depth` | Card > Make depth | |
| Style redraw | — | `background style` | Card > Redraw in a style | The app and CLI defaults differ (section 4). |
| Delete | — | `background delete` | Trash | |
| Dismiss a failed job | — | — | Failed card > Dismiss | App-only. |

### Settings and machine

| Feature | MCP tool | CLI command | App UI | Notes |
|---|---|---|---|---|
| Quick setup | — | `setup` | Settings > Setup checklist | |
| Model tier | — | `settings tier` | Settings > Models | |
| Download and stop | — | `models download`, `models stop` | Settings > Models | |
| Model status | — | `models status` | Settings > Models (file table) | |
| Full hash check | — | `models verify` | **MISSING** | |
| Civitai download | — | `models civitai ID` | **MISSING** | The Civitai token can be saved in the app, but nothing in the app uses it. |
| Models folder | — | `settings models-dir` | Shown, read-only | When it says "not found", the app offers no fix. |
| Access tokens | — | `settings token set\|check\|clear` | Settings > Access tokens | |
| Backend local or cloud | — | `settings backend` | Shown as "Coming soon" | Intentional. |
| ComfyUI start, stop, status | — | `comfy start\|stop\|status` | Settings > Engine | |
| Find ComfyUI | — | `comfy detect` | **MISSING** | |
| comfy-api-proxy | — | `proxy start\|stop\|status` | **MISSING** | |
| Graphics per app | — | `gpu list\|set\|all-integrated\|reset` | Settings > Graphics per app | |
| CPU, RAM, GPU meters | — | — | Status bar | App-only. |

## 2. Gaps ranked by value for a person

### High

**H1. The clip-by-clip loop: continue, retake, takes, use-take.** This is the main way to make a long video good
(AGENTS.md 7.5). Today it is agent-only.
- Video > New video > 6 Output: add a checkbox "Stop after each clip so I can check it". It sets `through_clip=1`.
  In `webapi/runs._run_new`, call `runner.set_clip_limit(id, 1)` (as `mcp_server.new_video_run` does).
- Run player: add a "Clips" section. Each clip shows its takes (thumbnail or strip, seed, note, status) and which take
  the video uses.
  - New `GET /api/runs/{id}/takes` → `runner.list_takes` plus `file_url` for each take's files (they sit in the runs
    folder, which is already served).
  - New `POST /api/runs/{id}/retake` {clip, seed, video, keyframe, adain, ingredients_strength, note} →
    `runner.add_take` + `cli._start_detached`. Offer "new seed" first and one knob at a time, as AGENTS.md says.
  - New `POST /api/runs/{id}/use-take` {clip, take} → `runner.use_take`. Turn the button off while the worker runs.
  - New `POST /api/runs/{id}/continue` {through_clip} → `runner.set_clip_limit` + `cli._start_detached`. Buttons:
    "Next clip" (K+1) and "Finish the video" (0).

**H2. Continue or start a stopped run.** A paused, stopped or interrupted run, or a "new" run whose worker did not start,
cannot be restarted in the app.
- Run player bar: a "Continue" button when the worker is not alive and the status is paused, stopped, interrupted or
  new. It uses the same `/continue` route as H1.

**H3. A library song as the video's sound.** The Music tab makes songs, but the Video form cannot use them.
- Video > New video > 5 Sound: add "A song from the Music tab" to the select, then a song list and a "start second"
  field. Fill the list from `GET /api/songs` (done songs only) and show the measured BPM.
- `webapi/runs._run_new`: pass `song` and `song_start` to `runner.new_run` (it already takes them).

**H4. Region edits.** AGENTS.md says small fixes should always use a region box. The app cannot do this.
- Characters > select an image > edit panel: add a "Mark a box" toggle. Drag a rectangle on the large image, turn it
  into the image's real pixels, and send it as `region=x0,y0,x1,y1`. `/api/characters/{name}/edit` already accepts
  it. Show the image size and the box. Suggest denoise 0.85 for region edits.

**H5. Repose.** After a body change, the sheet's pose stays old until a repose. The app says so but has no button.
- Characters > Mannequin view footer (after Save), and the Views view: a "Pose the front like this body" button.
- `webapi/characters._character_action`: a new `repose` branch → `j.submit("repose", name, seed=...)`. The job
  action already exists.
- Ask first. Say that it makes the views again (undo brings them back) and that a repose should start from the
  original front (AGENTS.md 4.5).

**H6. A backplate over the template's set.** This was made because a picture from text often did not fit the set.
- Backgrounds > + New background: a fourth card, "Over the template's set". Reuse the template select from the
  Backplate card. The place text is optional (the default is the template's Set words).
- `webapi/backgrounds._new`: a new kind that starts `background from-set --scene --shot --style`. Add it to `KINDS`.
  Use a kind name without a hyphen (for example `setplate`), because `_jobs()` splits the ticket name on "-".
- `backgrounds.js` `bgFacts`: label `source == "set"` as "over the set" (see section 4).

### Medium

**M1. Template layers (Set and Props words).**
- Templates > card: a "Layers" button opens a dialog with the Set words, one line per prop, and a "Read the .blend
  again" button.
- New `GET` and `POST /api/templates/{name}/layers` → `shots.layers(name, set_description, props, rescan)`. Show the
  warnings from `shots.layer_warnings`.

**M2. Two characters.**
- New video > 2 Character: an optional "Second character" select (ready characters only) and a spacing field.
  Explain that the left figure is the first character and that keyframe looks are turned off.
- `runs._run_new`: pass `character2={"name": ...}` and `spacing`.

**M3. Per-clip text.**
- New video > 3 Scene and action: a "Text per clip" section with one box for each clip. The page needs the clip count.
  Add it to the `/api/shots/preview` answer, or add `GET /api/shots/clips?blend=`.
- `runs._run_new`: pass `clip_videos` {clip number: text}.

**M4. Film assembly.**
- Video tab toolbar: a "New film" button. Pick finished runs in order, a score song and its start second, and the music
  level. Keep the rest at the CLI defaults.
- New `POST /api/films` (a detached worker or a background thread: ffmpeg takes time) and `GET /api/films`. Add a
  `films` area to `webapi/common._file_areas` so the app can play the result.

**M5. Clip scores.** Add a "Measure" button for each take in the H1 Clips section. New `POST /api/runs/{id}/score` runs
`clip_score.score` in the background (CPU, 1-2 minutes). Show the pass limits from AGENTS.md 7.5.

**M6. Setup repairs.**
- Settings > Engine: a "Find ComfyUI again" button → new `POST /api/engine/detect`.
- Settings > Models > Models folder: a "Change…" path box → new `POST /api/setup/models-dir`.

**M7. Civitai LoRA download.** Settings > Models: an "Add a LoRA from Civitai (version ID)" box → new
`POST /api/setup/civitai` (background, SHA-256 checked, as the CLI does).

### Low

- **Song loop:** a "Make a loop" action on the song card → new `POST /api/songs/{name}/loop` {beats, frames}.
  `song place` is not needed once H3 exists.
- **Look scene:** a choice in the Background block: "No background: describe the place in the look image".
- **Audio relay:** an on/off switch in New video > Advanced (default on).
- **Motion timeline:** a read-only line on the template card ("waves right hand in clip 1 ...").
- **Hash check:** a "Check files" link in Settings > Models → `models verify` in the background.
- **More options:** views `--distance` and `--face`; create `--seed`, `--width`, `--height`, `--style-lora`; more
  fork knobs (look chain, adain, keyframe).
- **comfy-api-proxy** start and stop in Settings > Engine.

### Intentional (not gaps)

- The proxy kit (`set_proxy_kit`, `character proxy`): AI-only by the user's decision. The app shows a read-only summary.
- Touch-ups in code (`character.touch_up`): agent work by nature. The app shows them in the history and can undo them.
- Changing a character's ID: the app renames only the display name, by design.
- The cloud backend: shown as "Coming soon".

## 3. In the app but not in MCP

An agent working through MCP cannot:
- make or delete songs, templates (dance, from a video, import, figure mask) or backgrounds (panorama, backplate,
  import, depth, style redraw);
- import a character from images or a sheet, set a display name, do a style pass, or repose;
- give an edit a reference image (Picture 2), or change its steps, cfg, sampler, scheduler, shift or negative;
- set a new run's background, look text, sound words, note, stand-in parts on/off, keyframe looks or sheet guide
  strength;
- build on a video (fork) or delete runs or characters;
- cancel song, template or background jobs (only videos and character jobs);
- start or stop ComfyUI, or manage tiers, downloads, tokens or graphics per app.

Most of these exist in the CLI, so an agent with a shell can still do them. Deletes are safer left out of MCP. The
missing `note` and `background` in `new_video_run` matter most, because AGENTS.md asks for a note on every test run.

## 4. Behaviour problems found while reading

1. **MCP test runs default to final quality.** `mcp_server.py:177` has `quality: str = "final"`. The app, the runner
   (`runner.py:86`) and AGENTS.md 7.1 all use preview for tests. A quick MCP test takes about twice as long.
2. **The app always sends `body_type=chibi`.** The Body type select (`index.html:351-355`) has no empty option. Hidden
   fields are still sent, so every run gets "chibi". For video templates this overrides the runner's "standard"
   default (`runner.py:642-643`), which was chosen because chibi drew the character too large (gauntlet round 12).
   Fix: add an empty "Automatic" option.
3. **A stale sheet is a dead end in the Mannequin view.** When a sheet redraw fails, `sheet_stale` stays set and the
   status goes to "sheet" (`character.py:990-997`). The panel says "Save draws it again" (`characters.js:302`), but
   Save is off unless the body changed or is missing (`characters.js:339`). "Use as is" is refused too
   (`character.py:829`). Fix: also turn Save on when `c.sheet_stale` is set (or add a "Draw the sheet again" button
   that calls `character.rescale_sheet`).
4. **CLI help shows wrong defaults.**
   - `cli.py:703-705` says the panorama ground default is "dome". It is "depth" (`runner.py:187`). The app is right.
   - `cli.py:736-737` says the Ingredients default is 1.4. It is 1.0 (`runner.py:86`). The app is right.
   - `cli.py:789-790` says the default profile is "ltx25dev" (full). Layout runs use "ltx25" (distilled)
     (`runner.py:478`, `692-693`).
   - `cli.py:357`: `app` help says "(Characters and Video tabs)". The app has six tabs.
5. **The Models section says every tier uses the full models** (`index.html:234-235`). The layout video uses the
   distilled LTX-2.5 in both tiers (AGENTS.md 10).
6. **"Template mannequin" texts after the per-character body change.** The new character form's Body type hint
   (`index.html:71`) and the CLI help (`cli.py:468`, `502`, `521-523`) say the front is posed like the template
   mannequin. Repose now uses the character's own mannequin body (AGENTS.md 4.5-4.6). The `character.repose` docstring
   (`character.py:730`) says the same old thing.
7. **The proxy kit is described as required in two places.** `cli.py:485-486` (`--default` help) says a character needs
   a proxy kit before a Blender run. `characters.js:38-40` keeps a "proxy" entry and a comment that say the same. The
   kit is optional and `video_missing` never returns "proxy" (`character.py:888-900`). This is only a stale text and a
   dead entry.
8. **A backplate made over the set shows "from text".** `backgrounds.js:13` knows only "import" and "run". A from-set
   plate has `source: "set"` (`background.py:471`).
9. **Background jobs started from the CLI do not show as cards** in the Backgrounds tab. `_jobs()` keeps only tickets
   that start with `bg-` (`webapi/backgrounds.py:28-31`). The CLI makes `bg<action>_xxxx` (`cli.py:1014`). They are
   shown only in the queue popover.
10. **Style redraw defaults differ.** The app uses keep 0.3 and strength 1.0 (`index.html:188-191`). The CLI uses keep
    0.5 and strength 0.8 (`cli.py:445-448`). AGENTS.md 6 suggests keep 0.15-0.3.
11. **"image 2" instead of "Picture 2".** The edit panel's reference tip (`characters.js:219`) and the CLI example
    (`cli.py:508`) say "image 2". AGENTS.md 4.3 says to name the inputs "Picture 1" and "Picture 2" (ComfyUI's labels).
12. **"Z-Image-Turbo" in texts.** `mcp_server.py:119`, `cli.py:457` and `character.py:12` say candidates come from
    Z-Image-Turbo. The full recipe uses Z-Image base (`recipe.py:14`, AGENTS.md 4.2).
13. **The MCP `new_video_run` docstring describes old video-to-video options** (`mcp_server.py:183-184`: `body_type`
    skeleton, `control_video`). Layout runs, the default for a library character, draw no skeleton.
14. **The token note leaves out Civitai.** `settings.js:189` lists HF_TOKEN and COMFY_API_KEY as the overriding
    variables. CIVITAI_TOKEN works too (`cli.py:397-398`).
15. **Tab texts are out of date.**
    - The Templates tooltip (`index.html:20`) says "Blender shots from a song and a dance". Templates also come from a
      video or an import.
    - The file header of `templates.js:1` still names a "text" template kind; that generator was removed
      (`webapi/templates.py:144`).
    - The Music tooltip (`index.html:28`) says "the score (BGM) for the films". The app has no films and cannot put a
      song under a video (H3, M4).
16. **AGENTS.md describes a button that does not exist.** `AGENTS.md:197` names a "Template body" button. The
    Mannequin view has presets (Adult, Teen, Child, Chibi) and a reset arrow on each slider, but no button with that
    name.
17. **Routes the page never calls** (not bugs): `POST /api/characters/{name}/mannequin` and `/proxy` (the Mannequin
    view saves through `/stand-in`; the kit is AI-only), and `POST /api/setup/backend` (cloud is "Coming soon").
    `GET /api/runs/{id}` returns `events_tail`, which the page does not show.
