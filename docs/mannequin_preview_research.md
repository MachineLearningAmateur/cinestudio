# Mannequin editor: live preview and a redesign of the side panel (research, 2026-10-24)

User: "some kind of rendered live image of what the mannequin would look like in the mannequin config", then "please
research on how to properly implement that into our ui"; scope addition: "this side bar is no where near intuitive to
use" (redesign the whole Mannequin panel: a structured parts editor, body sliders, where the editor lives, plain status).

Nothing in the app was changed for this report. Measurements were made with background Blender 5.2 (`-b`, Workbench
only, a few short runs; scripts and outputs in the session scratchpad), while a Qwen job was running on the GPU.

---------------------------------------------------------------------------------------------------------------------

## 0. Summary

- **Recommendation: C, the hybrid, built in two phases.**
  - **Phase 1 (exact, small): Blender Workbench previews on the server.** ONE background Blender launch does body +
    proxy kit + front / side / back Workbench renders + the height: **about 3 s wall time on this PC** (measured:
    start + file load ~1.9 s, kit 0.23 s, the first three renders 0.86 s, each further render ~0.10 s; VRAM +25-40 MB
    for a moment). Cached by a hash of (body, parts, the kit scripts, the template file). This is exactly what the runs
    put in the layout (same scripts), so it is the "truth" view.
  - **Phase 2 (live): a three.js turntable in the canvas,** fed by a GLB of the kitted mannequin that Blender exports
    in the same launch (+0.6-2.0 s, only when the PARTS change). The body is applied in the browser as 8 node scales
    + the rig's scale and floor drop. **I checked this math against Blender: the glTF node tree with the same scales
    gives the same heights to 0.1 mm** (template 1.8057 m, kitsune_rest 1.3019 m, a large test body 2.2315 m; the
    floor lines match too). Slider drags then update at screen rate (< 0.1 s), and the Workbench strip refreshes about
    1 s after the user stops.
- **Redesign:** a new **"Mannequin" view in the canvas** (the wide middle area: the 3D preview + the character's own
  front / sheet beside it at the same scale + a template ghost), and the **right panel becomes a structured editor**:
  presets, six labelled sliders with numbers and per-control reset, collapsible part cards (Mask, Hair, Cape, Jacket,
  Bands, Suit, Shoes) with toggles, segmented choices, colour swatches (sRGB in the UI, linear RGB stored) and an
  eyedropper on the front image, "Advanced: JSON" kept as an escape hatch, ONE Save with an unsaved-changes state, and
  a plain-language status (ready / what Save does to the sheet / when a repose is needed).
- **Main risks:** two body implementations (Python and JS) drifting apart (mitigated by one rules file + golden tests);
  the cape length depends on the legs at kit build time (approximate while dragging; exact after the Workbench check);
  three.js has had no non-module build since r161 (needs an import map + `import()` from our classic scripts, and the
  module files must be served as JavaScript: on this PC Python maps `.mjs` to `text/plain`); WebGL on the same 8 GB GPU
  as ComfyUI (render on demand only, dispose on leave); Blender start time dominates every server preview.

---------------------------------------------------------------------------------------------------------------------

## 1. What exists now (read from the code)

- `cinestudio/web/js/characters.js` `mannequinBlock(c)` / `bindMannequin(c)`: shown in the right panel (`#char-panel`,
  `--panel-w: 320px`, 280 px under 1000 px wide; app.css) only when the character is `ready` and nothing is selected.
  Two status lines, the height line (`c.height`), warnings (taller than 1.14x, sheet stale, old styled sheet), six
  number inputs (`MQ_KEYS`: scale, head, neck, torso, legs, arms; 0.3..2, step 0.01, placeholder 1.0), "Save body" /
  "Template body" (confirm), a JSON textarea "Stand-in parts (JSON)" with "Save parts" / "No parts" (confirm), and a
  note about `character repose`.
- API `POST /api/characters/{name}/mannequin` (`body` JSON or `default=1`) -> `character.set_mannequin`: merges the
  keys, **measures the height synchronously in the request** (`measure_height` = one Blender launch of
  `bpy_jobs/mannequin_height.py`, timeout 180 s) and redraws the sheet at the new height (`_rescale_locked`).
  `POST /api/characters/{name}/proxy` (`spec` JSON) -> `validate_proxy` + save.
- Body math: `bpy_jobs/mannequin_body.py apply()`: rig object scale = `scale`; pose-bone uniform scales spine = torso,
  neck = neck/torso, head = head/neck, thigh.L/R = legs, upper_arm.L/R = arms/torso; the shoulder BALLS (near-spherical
  meshes on the upper arms, aspect < 1.3) get object scale x torso; the rig drops by (1 - legs) x thigh head height x
  scale so the feet stay on the floor.
- Template mannequin (`templates/library/orbit_wave_360/shot.blend`): rig `MannequinRig`, 51 bones, **every part is a
  RIGID mesh parented to a bone** (no skinning, no vertex groups, no modifiers), 110 objects, 15 395 vertices.
- Proxy kit: `bpy_jobs/proxy_kit.py` builds clay parts in the REST pose, sized from the head mesh, parented to bones
  (head, chest, limb bones), then SAVES a .blend (`save_as_mainfile`). It runs `main()` at import (no
  `__name__` guard). **The only body-dependent part is the cape**: its "knee"/"thigh" end is computed from the
  `thigh` pose-bone scale (the legs) at build time. Mask marks presets: `bpy_jobs/mask_marks.json` (only `kitsune`).
- Schema: `character.validate_proxy` (parts mask, hair, suit, shoes, bands, jacket, cape; ranges and choices listed
  in section 6).
- Page: classic scripts loaded in order by `index.html` (core, characters, ..., main), no modules, no build step; the
  service worker caches nothing; no CSP header; the app listens on 127.0.0.1 only and opens in Edge/Chrome app mode
  (`webapp.open_window`), so Chromium features are available.

---------------------------------------------------------------------------------------------------------------------

## 2. UX research: how editors present a live preview of a body and its parts

### 2.1 What the references do

| Product | What it shows | Source / verified? |
|---|---|---|
| **Unreal MetaHuman Creator** | Big live viewport + parameter panel. Body measurements are **sliders that also take a typed value** ("drag the slider or click to type"); global controls (incl. Height) and grouped regions (Upper torso, Lower torso, Neck/Arms, Legs, Hands); **read-only diagnostics** (shoulder height, rise); **presets** as starting points (Fixed body types; Height then scales the preset); **Reset Body** returns to the shape when the tool was opened; **camera presets** Face / Body / Far and an **Auto** mode that frames the part of the tool you are using; "Show Measurement" draws debug lines in the viewport; editing a value **pins** it. | Epic docs (fetched): Body Params tool; Navigating MetaHuman Creator. Verified. |
| **VRoid Studio** | Real-time 3D preview of every edit; a Body tab with sliders (head size etc.). Third-party tools (VRM Body Tuner, V-Shape Studio) exist because its body controls are limited; they offer per-region sliders, presets and a 3D preview. | Steam store page + community + Booth listings. The exact panel layout was NOT verified from an official manual. |
| **Ready Player Me** | Avatar editor = category navigation + scrollable asset panels beside a renderer widget; body type is the first step (full/half body). | RPM docs search snippets; their docs site did not resolve from here, so the desktop/mobile layout rules were NOT verified. |
| **Mixamo** | A large 3D viewport with orbit; parameters as sliders in a side panel; changes preview immediately. | From general knowledge, NOT verified in this session. |
| **Blender** | Number fields that you can drag to scrub or click to type; colour picker values 0..1 with the **hex field gamma-corrected to sRGB** while the stored values are linear; an eyedropper. | Blender manual 5.2, Color Picker (fetched via search). Number-field drag behaviour from general knowledge. |
| **Game character creators** (The Sims, Baldur's Gate 3, Cyberpunk 2077 ...) | The figure fills the centre, controls on one side, drag to turn, the camera moves to the body part being edited, presets/"randomise" first then fine-tune, undo/reset. | General knowledge, NOT verified with a source in this session. |
| **3D product configurators** (e-commerce) | Instant visual update per option, reset/undo, simplified UI on small screens, touch gestures, fast load with a poster image first. Practitioners: the hard part is the interface around the 3D (users must predict what an option does). | Vendor/blog sources only (zolak.tech, thedigitalbunch). **No Baymard or NN/g study of configurators was found.** |
| **Google `<model-viewer>`** | Poster image until the model is ready (`poster` + `reveal`), camera orbit with limits, alt text for screen readers. | web.dev article + modelviewer.dev examples. |

### 2.2 Guidance that applies

- **Response times (NN/g, Nielsen 1993/2010):** < 0.1 s feels like direct manipulation; < 1 s keeps the flow of
  thought but the user "loses the feeling of operating directly on the data"; > 10 s needs progress feedback. A 3 s
  server render after each slider move is in the "noticeable, flow broken" band: fine for a check after release, not
  for dragging. This is the main argument for the in-browser view during drags.
- **Sliders (NN/g "Slider Design: Rules of Thumb"):** sliders are for approximate values; for exact values let people
  type; labels above or beside the slider, not below (the thumb hides them); avoid tiny increments on wide ranges.
  -> Slider **plus** a number field per row; a sensible slider range narrower than the allowed range (the number field
  accepts the full 0.3..2.0).
- **Immediate feedback, no Apply per control** (uxcel slider lessons; Microsoft slider guidelines) - but our values
  have side effects on Save (the sheet is redrawn, the run uses them), so: live PREVIEW per control, ONE explicit Save
  for the whole draft (common in settings forms with consequences).
- **Accessibility (WAI-ARIA APG slider pattern):** prefer the native `<input type="range">` (arrow keys, Page Up/Down,
  Home/End for free); `aria-valuetext` for a meaningful value ("1.55 x the template head"); test touch AT. Do not make
  orbit drag-only: offer Front / Side / Back / turn buttons; respect `prefers-reduced-motion` (no auto-spin).

### 2.3 Patterns to adopt (synthesis)

1. **Preview big, controls on the side** (MetaHuman, game creators, configurators). For us: the canvas area (wide)
   holds the preview; the 320 px right panel holds the controls.
2. **Presets first, then fine-tune** (MetaHuman fixed bodies; V-Shape Studio presets).
3. **Slider + typed value, grouped, with reset** (MetaHuman; NN/g).
4. **Read-only measurements next to the controls** (MetaHuman diagnostics) -> our height in metres, % of the
   template, and "heads tall".
5. **Camera follows the edited part** (MetaHuman Auto camera; game creators): editing Head/Mask/Hair frames the head;
   Legs frames the full body; Cape turns to the back.
6. **A reference for scale:** a ghost of the template mannequin (1.806 m) + a ruler; and the character's own front
   image / sheet beside the mannequin at the same pixel scale (our sheets now share one scale: AGENTS.md "One scale
   for every sheet").
7. **Poster / placeholder while loading** (model-viewer): show the last Workbench render until the GLB is ready.
8. **Unsaved state + Revert + per-control reset + confirm for destructive resets** (configurators; MetaHuman Reset).

---------------------------------------------------------------------------------------------------------------------

## 3. Technical options

### 3.1 Measurements on this PC (Blender 5.2, background, 2026-10-24)

| Step (one launch, `orbit_wave_360/shot.blend`, 5.3 MB) | Time |
|---|---|
| Blender start + open the file + a trivial script | 1.9 s wall |
| `mannequin_body.apply` | 0.007 s |
| proxy kit (kitsune_rest spec: mask, ears, 14 marks, hair, locks, back, cape) incl. its .blend save | 0.23 s |
| 3 Workbench renders 512x768 (first one compiles shaders) | 0.86 s |
| each further Workbench render | ~0.10 s |
| GLB export of mannequin + kit (182 nodes, 130 meshes, 1.1 MB) | 2.0 s (bare mannequin: 0.56 s) |
| **Whole launch: body + kit + 3 renders + GLB** | **5.0 s wall** (without the GLB ~3 s) |
| VRAM while rendering (nvidia-smi total, a Qwen job running) | +25-40 MB for ~1 s |

The Workbench renders (ortho camera, studio light, material colours, transparent film) look clean and readable; the
kit's colours and the mask marks show (front and side checked by eye). Note: `RenderSettings.engine`'s enum listed
only `BLENDER_EEVEE` in background mode, but assigning `BLENDER_WORKBENCH` worked (the enum under-reports).

### 3.2 Option A - server-side Workbench renders

- One new job script `bpy_jobs/mannequin_preview.py`: rest pose (as `mannequin_height.rest`), body, kit, renders,
  height, (Phase 2: GLB). Output into a cache folder keyed by a hash.
- **Fidelity: exact** (the same `mannequin_body.apply` and `proxy_kit` code as the runs; Workbench instead of the
  run's EEVEE layout light, but the shapes, places and colours are the same).
- **Latency: ~3 s** per new draft (cache hits instant). Blender start is 2/3 of it. A persistent warm Blender
  (a background process reading jobs from stdin) would cut it to ~1 s, at the cost of a long-lived ~300 MB process,
  state leaks between jobs and more code: not worth it at first.
- **Effort: small** (one script, one endpoint, an `<img>` strip). **Maintenance: none extra** (no second body
  implementation).
- **GPU:** Workbench uses the GPU through OpenGL, but AGENTS.md allows Workbench next to GPU jobs and the measured cost
  is tiny. Not a ComfyUI job: no queue ticket (AGENTS.md: only GPU/ComfyUI work queues). One preview process at a
  time (a lock), newest draft wins.
- Weakness: not live while dragging (see 2.2), and only fixed angles.

### 3.3 Option B - real-time three.js view

- **Model:** Blender exports the kitted mannequin as GLB. Checked: the Khronos exporter (v5.2.39) writes the rig as a
  node tree (joint nodes = bones, `skins` 1) and puts **every bone-parented rigid mesh as a child node of its joint
  node** (e.g. `head` -> `Sphere.004`, `ProxyMask`, `ProxyEarL`...); joint TRS are the REST pose (pose-bone scales are
  NOT exported). The rig object's own scale/location IS exported on the `MannequinRig` node. So export with the rig at
  scale 1, location 0, identity pose, and let the page apply the body.
- **Do three.js and Blender agree on bone scale for rigid children?** In three.js a `Bone` is an `Object3D`; scaling it
  scales its subtree about the joint origin (the bone head) - its child bones and its child meshes alike. Blender's
  pose-bone uniform scale with full inheritance does the same; bone-parented objects sit at the bone's tail, which the
  exporter bakes into the child node's translation, and a uniform scale about the head moves and scales them the same
  way. **Verified numerically:** a Python re-implementation of the Object3D TRS hierarchy on `bare.glb`, with the
  table below, reproduced Blender's measured height and floor for three bodies to 0.1 mm:

  | Body | Blender (`mannequin_body.apply`) | glTF tree + JS rules |
  |---|---|---|
  | template `{}` | 1.8057 m, floor -0.0100 | 1.8057, -0.0100 |
  | kitsune_rest `{scale .86, head 1.55, neck .35, torso .8, legs .7, arms .85}` | 1.3019, -0.0060 | 1.3019, -0.0060 |
  | `{scale 1.1, head .9, legs 1.2, torso 1.15, arms 1.1}` | 2.2315, -0.0131 | 2.2315, -0.0131 |

  (Skinned meshes would also follow joint scales in three.js; ours are rigid, which is the simpler case.)
- **The rules in JS** (what must be reproduced):
  - `rigNode.scale = s`; node scales spine = torso, neck = neck/torso, head = head/neck, thigh.L/R = legs,
    upper_arm.L/R = arms/torso;
  - shoulder balls: the mesh children of `upper_arm.L/R` whose bounding box aspect (largest/smallest, after the node's
    own scale) is < 1.3 get node scale x torso;
  - floor: `rigNode.position.y = -(1 - legs) * thighHeadY * s`, with `thighHeadY` = the rest world height of the
    `thigh.L` node (0.95 m) - read from the GLB once.
  - **Not reproducible exactly in JS: the cape.** proxy_kit ends the cape at the knee/thigh of the SCALED leg at build
    time. With a GLB built on legs = 1, the page can only stretch the cape node vertically by
    (top - knee(legs)) / (top - knee(1)) about its top (store top/knee heights as custom properties -> glTF `extras`
    with `export_extras=True`). Approximate while dragging Legs; the Workbench check after release is exact.
- **Alternative: bake the body into the export** (`export_rest_position_armature=False` after applying the body) - then
  every body change needs a new export (2-5 s): no live drag. Use the live rules; bake nothing.
- **Loading three.js in our classic-script page:** the non-module builds (`build/three.js`, `three.min.js`) were
  removed in **r161**, and since ~r167 `three.module.js` imports `three.core.js`. Addons (GLTFLoader, OrbitControls)
  import the bare specifier `three`. So: vendor the files locally (the app is local and should work offline; cdnjs hosts
  the core builds - 0.185.0 is listed - but I could not confirm it hosts the addons), declare an **import map** in
  `index.html` `<head>` (`"three": "/static/vendor/three/three.module.min.js"`), and load the viewer lazily from
  `characters.js` with **dynamic `import()`** (allowed in classic scripts; it uses the import map) only when the
  Mannequin view opens. three.js is MIT: keep its LICENSE file next to the vendored copy.
- **Effort: medium** (~400 lines of JS: viewer, camera presets, ghost, ruler, on-demand rendering, disposal).
  **Maintenance:** a second implementation of the body rules (mitigated in 4.4) and a pinned three.js copy.
- **GPU:** WebGL in the Edge GPU process on the same card; the scene is small (15k vertices + kit), but render **on
  demand** (only after a change or during orbit), never a continuous loop, and dispose the renderer when the view
  closes (browsers cap live WebGL contexts per page).

### 3.4 Option C - hybrid (recommended)

B for the drag (< 0.1 s), A as the exact "what Blender renders" check after release (~1 s debounce + ~3 s render), and
A's GLB as B's model (re-exported only when the PARTS change). If WebGL fails or three.js does not load, the page
still works with A alone (the strip becomes the main preview). Fidelity, latency and robustness all better than A or B
alone; the cost is B's effort.

| | A Workbench | B three.js | C hybrid |
|---|---|---|---|
| Matches the runs | exact | shapes exact (cape approx.), light differs | exact check + live view |
| Latency while dragging | ~3 s | < 0.1 s | < 0.1 s |
| Effort | small | medium | medium (A first) |
| Second body implementation | no | yes | yes (tested) |
| Works without WebGL/vendor files | yes | no | yes (falls back to A) |
| GPU use | +~30 MB, ~1 s per check | small, on demand | both, small |

---------------------------------------------------------------------------------------------------------------------

## 4. Implementation plan (C, in two phases)

### 4.1 Phase 1 - the redesigned editor with exact Workbench previews (no 3D yet)

1. **`bpy_jobs/proxy_kit.py`:** split `main()` into `build(spec, rig, coll) -> parts` and keep the CLI behind
   `if __name__ == "__main__":` (today `main()` runs on import; `runpy` with a patched `sys.argv` works - tested - but
   it also writes a 3.6 MB .blend per call). No behaviour change for runs.
2. **`bpy_jobs/mannequin_rules.json` (new, the ONE source of the body math):**
   ```json
   {"keys": ["scale", "head", "neck", "torso", "legs", "arms"], "limits": [0.3, 2.0],
    "bones": {"spine": {"mul": ["torso"]}, "neck": {"mul": ["neck"], "div": ["torso"]},
              "head": {"mul": ["head"], "div": ["neck"]}, "thigh.L": {"mul": ["legs"]}, "thigh.R": {"mul": ["legs"]},
              "upper_arm.L": {"mul": ["arms"], "div": ["torso"]}, "upper_arm.R": {"mul": ["arms"], "div": ["torso"]}},
    "balls": {"bones": ["upper_arm.L", "upper_arm.R"], "max_aspect": 1.3, "factor": "torso"},
    "drop": {"bone": "thigh.L", "key": "legs"}}
   ```
   `mannequin_body.py` reads it (same results as today; a unit test proves it). The UI labels, slider ranges and presets
   go in a separate UI table (`cinestudio/mannequin_ui.py` or JSON), served with the schema (step 5).
3. **`bpy_jobs/mannequin_preview.py` (new):** args `--body FILE --spec FILE --out DIR [--views front,right,back]
   [--size 512x768] [--glb]`. Rest pose (copy `mannequin_height.rest`), measure the template height, apply the body,
   build the kit (`proxy_kit.build`, no save), measure the height, Workbench renders with an **orthographic camera at
   a FIXED scale** (e.g. 2.1 m tall frame, feet line at the same pixel row for every character: renders of two
   characters are comparable, and a template-height line can be drawn at a known pixel row), transparent film,
   studio light, material colours; print `PREVIEW {"height": h, "ref": h0, "views": [...]}`. Pass JSON as FILES, not
   argv (marks lists can be long; Windows argv limits and quoting).
4. **Cache:** `<home>/cache/mannequin_preview/<key>/` with `front.png right.png back.png info.json` (+ `kit.glb` in
   Phase 2). `key` = sha256 of canonical JSON of (body, spec) + the sha256 of `mannequin_body.py`,
   `mannequin_rules.json`, `proxy_kit.py`, `mask_marks.json`, `mannequin_preview.py` and the template .blend (size +
   mtime is enough). Keep ~200 entries / LRU by atime. `measure_height` checks this cache first, so a Save right after
   a preview does not launch Blender again (today Save always launches Blender inside the request).
5. **Endpoints** (`cinestudio/webapi/characters.py`, executor via `_bg`):
   - `GET /api/mannequin/schema` -> the proxy field schema + body UI table + presets + mark preset names (one place
     the UI reads; `validate_proxy` ranges come from the same table, see 4.4).
   - `POST /api/characters/{name}/mannequin-preview` `{body, spec}` (the DRAFT, unsaved) -> validates
     (`validate_proxy`, body limits), returns `{key, height_m, ratio, urls: {front, right, back}}` from the cache or
     after one Blender launch. A module-level lock runs one Blender preview at a time; a newer request for the same
     character supersedes a waiting older one (do not kill a running process: just drop its result).
   - `POST /api/characters/{name}/stand-in` `{body | default, spec}` -> ONE save for the whole panel: validate both,
     then under `char_lock`: set mannequin + proxy, height (from the cache), sheet redraw - the same steps as today's two
     endpoints, atomically. Keep the old endpoints for the CLI/MCP.
6. **UI (`characters.js`):** a new canvas view `"mannequin"` in `charViews()` (always available once there is a front;
   the panel shows the editor whenever this view is active, not only when nothing is selected). Canvas: the Workbench
   strip (front / side / back) beside the character's front image at the same scale; panel: the editor (section 5).
   A draft object `state.mq = {saved, draft, history}`; any change marks it dirty and schedules a preview
   (debounce ~600 ms after the last input; while waiting, dim the old images and show "Updating Blender preview...").
7. **Tests (no GPU, no Blender unless marked):** rules JSON <-> `mannequin_body.apply` (fake rig objects); `stand-in`
   endpoint validation (bad body, bad spec, atomic save); cache key changes when a script changes; preview endpoint with
   `blender.run_job` patched; one Blender-marked test (skipped without Blender) that runs `mannequin_preview.py` and
   checks the height of kitsune_rest = 1.302 +- 0.002 m and three PNGs.

### 4.2 Phase 2 - the live three.js view

8. **Vendor three.js** (pin one release, e.g. the 0.18x the import map names) into `cinestudio/web/vendor/three/`:
   `three.module.min.js`, `three.core.min.js`, `addons/loaders/GLTFLoader.js`, `addons/controls/OrbitControls.js`,
   `addons/utils/BufferGeometryUtils.js`, `LICENSE`. Import map in `index.html` `<head>` BEFORE any script:
   `{"imports": {"three": "/static/vendor/three/three.module.min.js", "three/addons/": "/static/vendor/three/addons/"}}`.
9. **`cinestudio/web/js/mannequin3d.js`** (an ES module; keep the `.js` extension) exporting `createViewer(el, opts)`
   -> `{load(glbUrl), setBody(body), frame(part), view("front"|"right"|"back"), setGhost(on), height(), dispose()}`;
   `characters.js` does `const m = await import("/static/js/mannequin3d.js")` on first open. It loads the template GLB
   once more as a translucent, unlit **ghost** (body `{}`) beside/behind the figure, draws a floor ring and a metre
   ruler, frames by part (head / full body / back), renders only on change, and disposes the renderer on view leave.
10. **GLB:** `mannequin_preview.py --glb` exports `kit.glb` with the rig at identity (scale 1, location 0, no pose
    scales) after building the kit on the template body (so the page can apply any body), with
    `export_animations=False, export_extras=True`; the cape gets custom props `cs_top_y`, `cs_end_y`, `cs_end`
    ("knee"/"thigh"/"hips") so the page can stretch it for other legs. If the exact renders need the cape built for the
    real legs, the script deletes the kit after the export, applies the body and builds the kit again (+0.2 s).
    The GLB key = hash of the spec only (the body is live), so parts changes re-export and body changes do not.
11. **Body rules in JS** read `mannequin_rules.json` (served by the schema endpoint) - table-driven, ~30 lines, no
    hard-coded formulas.
12. **Fallback:** WebGL unavailable or import fails -> keep the Phase 1 strip as the main preview, one quiet note.

### 4.3 What the user sees while work runs

- Dragging a slider: the 3D figure and the height readout change at once (Phase 2); the Workbench strip shows a small
  "out of date" chip.
- ~0.6 s after the last change: the strip dims, a thin progress bar and "Checking in Blender..." appear; the old
  images stay (no empty boxes); typically ~3 s later the new renders swap in. Errors: the old images stay, a red line
  with the Blender error tail and a Retry link.
- Parts change (Phase 2): the 3D view keeps the old model with a spinner badge until the new GLB arrives (~5 s), then
  swaps without moving the camera.
- Save: the button shows "Saving and redrawing the sheet..." (it is a few seconds: height from the cache + sheet
  build); on success a toast "Saved. The sheet now shows 1.30 m."

### 4.4 Keeping JS and Blender in sync

- One rules file (`bpy_jobs/mannequin_rules.json`) read by `mannequin_body.py`, the JS viewer and a small Python
  evaluator used in tests.
- A **golden fixture** `tests/fixtures/mannequin_tree.json` (node names, parents, rest TRS, mesh bounds extracted from
  the template GLB, ~30 KB) + `tests/fixtures/mannequin_golden.json` (Blender heights/floors for 5 bodies).
  Unit test: the Python evaluator on the fixture == the golden values (1 mm). Blender-marked test: regenerate both
  from the template and compare (catches a changed template or rule). Optional: `node tests/js/mannequin_rules.mjs`
  runs the JS evaluator on the same fixture (node 22 is installed) and pytest calls it when `node` exists.
- One proxy schema table (parts, fields, types, ranges, choices, defaults, labels) used by the UI and by
  `validate_proxy` for the plain range/choice checks (the geometric mark checks stay in code); a test walks the table
  and checks each boundary is accepted and each value beyond it refused.

---------------------------------------------------------------------------------------------------------------------

## 5. Redesign of the Mannequin editor (the coordinator's scope addition)

### 5.1 Where it lives: canvas view + side panel (recommended)

| Option | For | Against |
|---|---|---|
| Keep everything in the 320 px panel (preview on top) | no new view | a 320 px preview is ~280x420: too small to judge proportions; the panel scrolls a lot; no room for the comparison with the sheet |
| Modal dialog / drawer | big | hides the character's sheet and front (the thing we compare with); modal state on a long edit; another layout system |
| **"Mannequin" view in the canvas + editor in the panel** | the canvas (~1100 px at the default 1500 px window) holds the 3D view, the template ghost and the character's own front/sheet side by side; the panel keeps its familiar role (controls); matches MetaHuman / game creators (big preview, side controls) | needs a new canvas view (small: `charViews` + `renderCanvas`) |

The panel can stay 320 px: a slider row = label 84 px + range + number 56 px + reset 24 px fits in 288 px of content.
Under 1000 px the app already narrows the panel to 280 px; there the canvas stacks the 3D view above the comparison.

### 5.2 Wireframes

Desktop (Characters tab, "Mannequin" view):

```
+------+---------+--------------------------------------------------------------+-------------------------------+
| rail | shelf   | kitsune_rest [ready]   [Front][Views][Sheet][*Mannequin*]   | MANNEQUIN            * unsaved|
|      |         +--------------------------------------------------------------+ (o) Ready for videos.         |
|      | [tile]  |  +------------------------------+  +-----------------------+ |  Saving redraws the sheet at  |
|      | [tile]  |  |   3D view (drag to turn)     |  | Its front    | Sheet  | |  1.30 m. The sheet's pose     |
|      | [tile]  |  |                              |  | (same scale) |  ...   | |  changes only after a repose. |
|      |  [+]    |  |   . . ghost 1.81 m . .  -2.0m|  |   [image]    |        | +-------------------------------+
|      |         |  |   :   .---.            -1.5m |  |              |        | | PRESET [Adult][Teen][Child]   |
|      |         |  |   :  ( o o )  <- kit   -1.0m |  |  [eyedropper | active]| |        [Chibi][Custom]        |
|      |         |  |   :   /| |\            -0.5m |  +-----------------------+ +-------------------------------+
|      |         |  |   :____/ \_____ floor ring   |                            | BODY             1.30 m · 72% |
|      |         |  |  [Front][Side][Back][<][>] [ghost] [ruler]             | |  Overall height  ===o== 0.86 ↺|
|      |         |  +------------------------------+                            |  Head size       =====o 1.55 ↺|
|      |         |  Blender check (exact):  [front] [side] [back]  ✓ up to date | |  Neck length     =o==== 0.35 ↺|
|      |         |                                                              |  Torso length    ==o=== 0.80 ↺|
|      |         |                                                              |  Leg length      ==o=== 0.70 ↺|
|      |         |                                                              |  Arm length      ===o== 0.85 ↺|
|      |         |                                                              |  3.4 heads tall               |
|      |         |                                                              +-------------------------------+
|      |         |                                                              | STAND-IN PARTS                |
|      |         |                                                              | v Mask      [on ]  ■ ivory    |
|      |         |                                                              |    Shape  [Pointed|*Round*]   |
|      |         |                                                              |    Eyes [off] size ==o== 1.0  |
|      |         |                                                              |    Ears [on ] h ==o== w =o==  |
|      |         |                                                              |    Marks [Kitsune v]          |
|      |         |                                                              | > Hair      [on ]  ■ plum     |
|      |         |                                                              | > Cape      [on ]  ■ teal     |
|      |         |                                                              | > Jacket    [off]             |
|      |         |                                                              | > Bands     (0)  [+ add]      |
|      |         |                                                              | > Suit      [clay]  ■         |
|      |         |                                                              | > Shoes     [clay]  ■         |
|      |         |                                                              | > Advanced: JSON              |
|      |         |                                                              +-------------------------------+
|      |         |                                                              | [Revert]   [ Save changes ]   |
|      |         |                                                              |  (sticky footer)              |
+------+---------+--------------------------------------------------------------+-------------------------------+
```

A part card open (Hair):

```
v Hair                       [on ]   ■ plum  [pick from front]
   Length       [Short|*Medium*|Long]
   Hairline     high ===o======== low       0.95 ↺
   Side locks   [None|Chin|*Shoulder*|Chest]
   Back         [None|Chin|*Shoulder*|Chest]
```

A band row (Bands card):

```
v Bands (2)                                          [+ Add band]
   [Forearm v] [x] both sides   from =o====== to   ■ white   thick 1.3  [🗑]
   [Shin    v] [x] both sides   from ==o===== to   ■ white   thick 1.3  [🗑]
```

Narrow (< 1000 px): the canvas shows the 3D view on top and the comparison below; the panel is unchanged (280 px).

### 5.3 Body controls

| Key | Label | Slider range (typed: 0.3..2.0) | Step | Help text |
|---|---|---|---|---|
| scale | Overall height | 0.50 .. 1.30 (shows metres too) | 0.01 | the whole figure; 1 = the template (1.81 m) |
| head | Head size | 0.70 .. 2.00 | 0.01 | with the mask and hair parts; chibi ~1.5 |
| neck | Neck length | 0.20 .. 1.50 | 0.01 | the head keeps its size |
| torso | Torso length | 0.60 .. 1.40 | 0.01 | spine and chest; arms and head keep their sizes |
| legs | Leg length | 0.50 .. 1.40 | 0.01 | the feet stay on the floor |
| arms | Arm length | 0.60 .. 1.40 | 0.01 | both arms |

- Each row: label (above/beside, not below: NN/g), native range input, number input (full allowed range), ↺ reset to
  the template value (1.0; long-press/Alt = reset to the saved value). Double-click on the label also resets.
- Readouts at the top of the card: **height in metres and % of the template** (live from the 3D view's bounding box
  in Phase 2 - verified exact above; from the Blender check in Phase 1), **heads tall** (figure height / head-part
  height), and the existing warning when > 1.14x ("taller than a sheet can show").
- Optional: type a height in metres ("1.30 m") and let the page solve `scale` = target / height-at-scale-1.
- **Presets** (chips; selecting one fills the draft, nothing is saved). Heights computed with the verified rules:

  | Preset | Body | Height | Note |
  |---|---|---|---|
  | Adult (template) | `{}` | 1.806 m, 100% | the template mannequin |
  | Teen | `{scale 0.93, head 1.08, legs 0.97}` | 1.671 m, 93% | proposal, not tested in a run |
  | Child | `{scale 0.72, head 1.3, torso 0.92, legs 0.85, arms 0.92}` | 1.228 m, 68% | proposal, not tested in a run |
  | Chibi (kitsune_rest) | `{scale 0.86, head 1.55, neck 0.35, torso 0.8, legs 0.7, arms 0.85}` | 1.302 m, 72% | measured and used (AGENTS.md: AnyPose needs a body ~10% more extreme than the target) |
  | Custom | whatever the sliders say | | shown when the draft matches no preset |

### 5.4 Stand-in parts: a structured editor (schema from `validate_proxy` + `proxy_kit.py`)

Every part = a collapsible card with an on/off toggle in its header (off = the key is absent from the spec), a colour
swatch in the header, and its fields. Defaults below are proxy_kit's own (from its code/docstring).

| Part | Fields (control, range/choices, default) |
|---|---|
| **Mask** | colour/shade (swatch; default shade 0.92); Shape: segmented Pointed / Round (`shape`, default pointed); Eyes: toggle (`eyes`) + Eye size slider 0.3..3.0 (`eye_size`, 1.0); Covers up to: slider -1.0..1.5 (`top`, absent = full shell; label "whole head <-> brow"); Ears: toggle (`ears`) + Ear height 0.05..3.0 (`ear_height`, 0.95) + Ear width 0.05..2.0 (`ear_width`, 0.30) + Inner ear colour (`ear_inner`); Marks: picker None / Kitsune / Custom (custom = JSON list in Advanced; presets from `mask_marks.json`). Mark checks (inside the mask, above a pointed chin, below a cut top) stay server-side; show the server's message under the card. |
| **Hair** | colour/shade (default shade 0.22); Length: Short / Medium / Long (`length`); Hairline: slider -1.5..1.5 (`front`, 0.45); Side locks: None / Chin / Shoulder / Chest (`locks`); Back: None / Chin / Shoulder / Chest (`back`) |
| **Cape** | colour (default [0.02, 0.18, 0.15]); Length: Hips / Thigh / Knee (`length`, thigh); Width 0.2..4.0 (1.25); Flare 0.2..4.0 (1.7); Wrap 0..1 (0.25) |
| **Jacket** | colour; Sleeve length 0..1 (`sleeve`, 0.55; "none <-> whole upper arm"); Bottom 0..1 (`bottom`, 0.75; "chest <-> hips"); Advanced: fit 0.5..3 (1.04), sleeve fit 0.5..3 (1.15) |
| **Bands** | a list; each row: limb select (upper arm, forearm, hand, thigh, shin, foot, + spine, chest, neck, hips), "both sides" checkbox (writes `.L` and `.R` entries), From/To 0..1 (from < to; "joint at the top <-> joint at the bottom"), colour, Thickness 0.5..3 (`scale`, 1.3 in the docstring example) |
| **Suit** | colour (the mannequin's own body parts); off = clay |
| **Shoes** | colour (feet and toes); off = clay |
| **Advanced: JSON** | the current textarea, two-way: edits in the cards update it; valid JSON typed here updates the cards; invalid JSON shows an error and blocks Save (the JSON is the escape hatch for custom marks and new keys) |

**Colours.** The kit stores linear RGB 0..1 (or one grey). `<input type="color">` and the EyeDropper API give sRGB hex.
Convert with the sRGB transfer function (IEC 61966-2-1): `lin = c <= 0.04045 ? c/12.92 : ((c+0.055)/1.055)^2.4` and
back `c = lin <= 0.0031308 ? 12.92*lin : 1.055*lin^(1/2.4) - 0.055`; round stored values to 3 decimals. This is the same
convention as Blender's colour picker (hex = gamma-corrected sRGB, values linear). A stored grey (`shade`) shows as a
grey swatch; picking a colour writes `color` and removes `shade`.

**Pick colours from the character (eyedropper).** In the Mannequin view the character's front image sits beside the
3D view; a "pick" button on a swatch arms the eyedropper, the cursor becomes a crosshair over the front image, a click
samples the **median of a 7x7 patch** (avoids line art and noise) from a hidden canvas copy of the image (same origin:
no taint) and writes the linear value. This works in every browser; the EyeDropper API (Chromium 95+, secure context -
`127.0.0.1` counts) can be offered as "pick anywhere on screen" but it returns one pixel only. Note: the sampled colour
is lit; the clay is lit again in the render, so prefer a mid-tone patch (the UI can say so in the tooltip).

### 5.5 One Save, unsaved changes, undo, resets

- One draft for body + parts; the panel header shows "● Unsaved changes" when draft != saved; the sticky footer has
  **Revert** (to saved) and **Save changes** (primary; disabled when clean or invalid).
- **Ctrl+Z / Ctrl+Shift+Z** inside the editor: a local history of draft states (one entry per released slider or
  changed field).
- Per-control ↺ (template value); "Template body" becomes the Adult preset (no confirm needed: it only changes the
  draft); "No parts" becomes a "Remove all parts" link in the parts header that turns every card off in the draft. The
  destructive moment is **Save**, so the confirmation moves there when the change is big: Save asks only when the body
  changes the height by more than ~5% or parts are removed ("Save? The sheet is redrawn at 1.30 m (was 1.81 m) and the
  mask, hair and cape are removed from the videos.").
- Leaving the view, switching character or closing the tab with unsaved changes: a confirm ("Discard the unsaved
  mannequin changes?") and `beforeunload`.

### 5.6 Status and consequences in plain words

At the top of the panel, one line + one sentence:

| State | Text |
|---|---|
| ready, clean | "✓ Ready for videos. Every video puts this body and these parts on the dancer." |
| missing body or parts | "Not ready for videos yet: choose a body (Adult is fine) and the stand-in parts (or none), then Save." |
| draft changes the body | "Saving redraws the sheet at 1.30 m (72% of the template). The sheet's pose stays as it is until you repose it." + a "Repose..." link (GPU, ~15 min; queued) |
| draft changes only parts | "Saving changes the stand-in parts of every new video. The sheet does not change." |
| sheet stale | "The sheet still shows another height: Save draws it again." |
| style-pass sheet outdated | "The style-pass sheet was made at the old height: run the style pass again." |

---------------------------------------------------------------------------------------------------------------------

## 6. Pitfalls

- **Blender start time** (1.9 s of every ~3 s preview): cache aggressively, debounce, one process at a time, newest
  wins. A warm Blender server is an optimisation for later.
- **`proxy_kit.py` runs `main()` on import and saves a .blend:** refactor before reusing it (step 1).
- **Windows:** pass file paths in the argument list (`blender.run_job` already does), JSON as files (argv length and
  quoting), `CREATE_NO_WINDOW` (already), cache paths under `<home>` (git-ignored `.cinestudio/`).
- **MIME types:** on this PC Python's `mimetypes` maps `.js` -> `application/javascript` (fine) but **`.mjs` ->
  `text/plain`** and `.glb` -> nothing. Module scripts are refused with a non-JS MIME type: keep `.js`, and register
  `mimetypes.add_type("text/javascript", ".mjs")` and `("model/gltf-binary", ".glb")` at app start.
- **Classic scripts + modules:** an import map must come before the first module load; dynamic `import()` from a
  classic script uses it; module code runs deferred, so no top-level dependency from classic scripts on the viewer;
  top-level names in classic scripts stay unique (the viewer adds none: it is a module).
- **Offline / CSP:** vendor three.js (no CDN at runtime). No CSP exists today; if one is added, the inline import map
  needs a hash or must move to a file-backed approach.
- **GPU (8 GB, often busy):** Workbench is allowed next to GPU jobs (AGENTS.md) and measured tiny; never EEVEE for
  previews; no ComfyUI queue ticket (not ComfyUI work). WebGL: render on demand, dispose on leave, low pixel ratio
  (devicePixelRatio capped at 1.5). After an NVIDIA driver reset (AGENTS.md: nvlddmkm event 153) both Workbench and
  WebGL can fail: show the error, keep the last images, offer Retry.
- **Memory:** a Blender preview process ~300-500 MB for ~3 s (not measured precisely); the GLB is 1.1 MB; three.js
  scene a few tens of MB. Fine with 32 GB, but never run previews in parallel.
- **Cape and legs:** the GLB cape is approximate under a leg change (5.4/3.3); the Workbench check is exact.
- **The template changes:** include the template file in the cache key; the golden test catches rig changes.
- **Request threads:** today `set_mannequin` launches Blender inside the HTTP request (up to 180 s timeout); the new
  Save should reuse the preview cache's height so it normally returns fast.
- **Two mannequins in a run** (duo shots): the preview shows one character's own mannequin, which is what each run
  mannequin gets (`duplicate_mannequin --bodies`); no change needed.

---------------------------------------------------------------------------------------------------------------------

## 7. Not verified / open

- VRoid Studio's, Ready Player Me's and Mixamo's exact panel layouts (official docs not reachable or not found);
  game character creator patterns are from general knowledge.
- No Baymard or NN/g research specifically on 3D configurators or live previews was found; the configurator points
  come from vendor blogs.
- Whether cdnjs hosts the three.js addons (GLTFLoader/OrbitControls) - irrelevant if vendored locally.
- The Blender preview process' RAM was not measured; the WebGL VRAM was not measured.
- The Teen and Child presets are proposals (heights computed, never used in a run).
- The three.js side was verified by re-implementing the Object3D transform math in Python on the exported GLB, not by
  running three.js in a browser.

---------------------------------------------------------------------------------------------------------------------

## Sources

- Nielsen Norman Group, Response Times: The 3 Important Limits - https://www.nngroup.com/articles/response-times-3-important-limits/
- Nielsen Norman Group, Slider Design: Rules of Thumb - https://www.nngroup.com/articles/gui-slider-controls/
- uxcel, Sliders best practices (immediate changes) - https://app.uxcel.com/lessons/sliders-best-practices-918/immediate-changes-8900
- Microsoft, Slider guidelines - https://learn.microsoft.com/en-ie/Windows/Win32/uxguide/ctrl-sliders
- W3C WAI-ARIA APG, Slider pattern - https://w3c.github.io/wai-website/ARIA/apg/patterns/slider/ ; rating example - https://www.w3.org/WAI/ARIA/apg/patterns/slider/examples/slider-rating
- Evinced, Slider (native range vs ARIA) - https://knowledge.evinced.com/components/Slider
- Epic Games, MetaHuman Creator Body Params tool - https://dev.epicgames.com/documentation/metahuman/metahuman-creator-body-params-tool-in-unreal-engine
- Epic Games, Navigating MetaHuman Creator - https://dev.epicgames.com/documentation/metahuman/navigating-metahuman-creator-in-unreal-engine
- Epic Games, MetaHuman Blend tool - https://dev.epicgames.com/documentation/metahuman/metahuman-creator-blend-tool-in-unreal-engine
- VRoid Studio on Steam - https://store.steampowered.com/app/1486350/VRoid_Studio ; community thread - https://steamcommunity.com/app/1486350/discussions/0/3110269850326447017
- VRM Body Tuner - https://booth.pm/ko/items/8606214 ; V-Shape Studio - https://booth.pm/ja/items/8506445
- Ready Player Me docs (snippets only): avatar creator integration - https://docs.readyplayer.me/ready-player-me/integration-guides/web-and-native-integration/avatar-creator-integration ; UX/UI guidelines - https://docs.readyplayer.me/ready-player-me/integration-guides/ux-ui-guidelines/ux-and-ui-guidelines-and-essentials
- Configurator UX (vendor/blog): https://zolak.tech/blog/how-to-build-product-configurator ; https://www.thedigitalbunch.com/blog/how-to-uxui-design-a-3d-configurator-that-makes-sense ; https://gokickflip.com/blog/what-are-major-features-of-product-configurator
- web.dev, The <model-viewer> web component - https://web.dev/articles/model-viewer ; loading examples - https://modelviewer.dev/examples/loading/
- three.js forum, UMD version removed in r161 - https://discourse.threejs.org/t/umd-version-of-three-js-v-161/60912 ; GitHub issue - https://github.com/mrdoob/three.js/issues/27763 ; three.core.js split - https://discourse.threejs.org/t/three-js-now-imports-three-core-js/75187 ; build deprecation - https://discourse.threejs.org/t/build-three-js-depreciation/53915
- three.js manual, Installation (import maps, addons) - https://threejs.org/manual/en/installation (404 from the fetcher; content via search snippets)
- cdnjs three.js - https://cdnjs.com/libraries/three.js
- Blender glTF exporter commits on bone-parented objects - https://developer.blender.org/rBAeb07144ea7e8ace18efd2aefe358c2f3a6db4120 ; https://developer.blender.org/rBA4cb6ebd8747e15b417e565afb5b1c4623a66d381
- Blender manual 5.2, Color Picker (hex is gamma-corrected sRGB) - https://docs.blender.org/manual/en/5.2/interface/controls/templates/color_picker.html
- MDN, EyeDropper API - https://developer.mozilla.org/docs/Web/API/EyeDropper_API ; caniuse - https://caniuse.com/mdn-api_eyedropper
- sRGB transfer function (IEC 61966-2-1) - https://en.wikipedia.org/wiki/SRGB
- Local measurements (this PC, 2026-10-24): scripts `inspect.py`, `preview_test.py`, `export_clean.py`, `sim.py`,
  `wb_loop.py` in a session scratchpad (not in the repository).
