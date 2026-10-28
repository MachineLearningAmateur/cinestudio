# Docs and downloads review (2026-10-24)

A review of the project documents and of the model files. Nothing was deleted, moved or downloaded. No GPU work was
run. The model folders were only listed (and one 0.45 GB file was hashed). Code, JSON, AGENTS.md and CREDITS.md were
not edited: their changes are proposals below.

Short version:
- The 8gb tier is complete on disk: 31 files, 153.9 GB (`models status`: ready).
- About **23 GB** on disk belongs to old runs or unused features. The biggest file is the old fp8mixed Qwen edit model
  (20.5 GB). About **2 GB** are sam2 copies that the node never loads. About **20 GB** is not used by cinestudio at all:
  it is probably the user's other ComfyUI work.
- The setup list has one real bug. The pinned SHA-256 of the distilled LTX-2.5 file is a placeholder (64 `*`). A new
  download of that file fails its hash check.
- The setup list downloads the SAM 2.1 files that the node does NOT load. It misses the fp16 file that it does load.
- README.md and docs/MODELS.md were badly out of date (LTX-2.3 profiles, Z-Image-Turbo, Lightning, wrong tier sizes,
  18 MCP tools). They are corrected.

---------------------------------------------------------------------------------------------------------------------

## Part 1: documentation

### 1.1 Edits I made

**README.md**
- Intro note: "Setup steps, the CLI, the MCP server and the Blender add-on are not finished yet" -> the CLI, the web
  app and the MCP server work; AGENTS.md is the manual. (There is no Blender add-on.)
- Layout block: added the web app, MCP, blender_lib/, the template library, the Linux installer, the unit tests.
- Local files line: lists what `.cinestudio/` holds now (characters, runs, templates, songs, backgrounds, tokens).
- Quick setup: "tokens (both optional)" -> "tokens". The HF token is needed for a new install, because the LTX-2.5
  files are gated. Added the Civitai key and `CIVITAI_TOKEN`.
- Tier sizes: 104 GB / 174 GB -> about 154 GB / 223 GB (measured from `setup/models.json` with every feature).
- "Every tier uses the full (not distilled) models" -> the default layout video uses the DISTILLED LTX-2.5 int8 model
  in both tiers; the dev model is for video-to-video.
- Models section: `setup/download_models.py --profile ltx23|zimage|qwen_edit` -> `cinestudio models status|download|verify`
  and `settings models-dir`.
- "Using a shared ComfyUI": restarts "through ComfyUI-Manager" because of "LTX-2.3" -> cinestudio restarts its OWN
  engine itself; a ComfyUI started elsewhere is never restarted automatically (`CINESTUDIO_RESTART=manager-shared`
  allows it through ComfyUI-Manager). This is what `server.restart` and `comfy.restart_own` do.
- Queue: GPU work also covers songs, templates and backgrounds; `queue cancel` kinds are character, video, song,
  template, lab.
- MCP: the 18-tool list -> the 26 tools of `mcp_server.py`. The proxy kit is set only through MCP.

**docs/MODELS.md** (rewritten in the same style)
- Removed: the `ltx23` / `zimage` / `qwenedit` profile tables, LTX-2.3 distilled fp8, Gemma 3 fp4, the 2.3
  Ingredients LoRA, the 2.3 upscalers, Z-Image-Turbo, the Lightning LoRA, `download_models.py`.
- Added: the two tiers and their sizes, which LTX model is used where, a feature -> files table, every file of
  `model_tiers` with folder, size, source and licence, and the per-tier slot table.
- Style LoRA section: LTX-2.3 / Z-Image-Turbo -> LTX-2.5 / Z-Image. Added the note that a 2.3 LoRA made 2.5 worse.
- Licence notes: LTX-2.5 uses the LTX-2.x Community License (LICENSE-2_x, read from the HF card today). Gemma 4
  replaces Gemma 3. Added the MIT, Apache and SAM licences of the newer files. MiniMax: worded as a project rule.

**docs/RUNPOD.md**
- Network volume: 150 GB / 220 GB -> about 200 GB / 275 GB. The models alone are 154 GB / 223 GB.
- `HF_TOKEN`: "optional" -> needed (the LTX-2.5 files in every tier are gated).

Not edited, on purpose:
- docs/IDEAS.md: a dated ideas log (2026-10-19). Its LTX-2.3 wording is history.
- docs/gauntlet_critic.md: a test instrument. Changing it between gauntlet rounds would break comparisons. Note: it
  says "anime video clips" and "the mannequin's head is a blank sphere". The main character is now 3D, and the
  proxy kit puts a mask on the mannequin. Update it before the next new gauntlet.
- The research reports and CREDITS.md (as instructed).

### 1.2 Proposed AGENTS.md changes (exact old text -> new text)

1. Section 4.3. Line 116 contradicts line 109.
   - Old: `other gloves). Without a reference our graph sends the same picture as Picture 2.`
   - New: `other gloves). Without a reference the graph sends Picture 1 only (bind `optional_images`).`

2. Section 7.4. New video-template runs use the `ltx25dev_*` graphs, not `ltx23_*` (runner.py: a run with a template
   or without a library character gets `render_mode` v2v and `PROFILE` = recipe.LTX_PROFILE = "ltx25dev").
   - Old: `- Video-to-video (`ltx23_*` graphs, start noise, skeleton drawing) remains only for old runs and video templates.`
   - New: `- Video-to-video (start noise, skeleton drawing) remains for video templates and Blender shots without a
     library character: new runs use the `ltx25dev_*` graphs (the LTX-2.5 dev model). The `ltx23_*` graphs serve only
     old runs, whose LTX-2.3 model files are no longer on this PC (they cannot be resumed, retaken or forked).`

3. Section 10. Add after `` `models civitai VERSION_ID` downloads a Civitai file (token optional). ``:
   - New: `  Sizes with every feature: 8gb ~154 GB, large ~223 GB. The LTX-2.5 dev transformer (21.5 / 42 GB) and the
     LTX-2.3 union LoRA serve only video-to-video and the (off) pose guide. The SAM 2 node reads only
     <ComfyUI>/models/sam2 and, at fp16, only `sam2.1_hiera_large-fp16.safetensors`.`

4. Section 10, tokens. AGENTS.md says `settings token set NAME` works for `civitai`. The CLI refuses it:
   `cli.py:379 choices=["hf", "comfy"]`. Fix the code (item C8 below), not AGENTS.md.

### 1.3 Proposed CREDITS.md changes (report only)

- Models table:
  - "LTX-2.3 ... and its IC-LoRAs (union control, ingredients) and upscalers" -> "LTX-2.5 (Lightricks) and its IC-LoRAs
    (Layout-To-Render, Ingredients 2.5) and spatial upscaler; the LTX-2.3 union control IC-LoRA (pose guide,
    video-to-video)". Licence: LTX-2.x Community License (LICENSE-2_x) for the 2.5 files.
  - "Gemma 3 12B (repackaged by Comfy-Org)" -> "Gemma 4 12B (the LTX-2.5 text encoder, repackaged by Lightricks)".
    Check its licence text.
  - "Z-Image-Turbo" -> "Z-Image (base)" (Comfy-Org/z_image).
  - Remove the "Qwen-Image-Edit-2511-Lightning ... fast edit mode" row (fast mode is off; the file is deleted).
  - Add: AnyPose LoRAs (serotoninboi, Apache-2.0: the repose); Qwen-Image-2512 (Qwen, GGUF by unsloth, Apache-2.0:
    panoramas and backplates); qwen-360-diffusion LoRA (ProGamerGov, MIT); MoGe-2 (Microsoft, MIT: 360 depth);
    ACE-Step 1.5 (MIT: songs); SAM 2.1 (Meta, Apache-2.0: the video-template figure mask). Add the dx8152
    Style-Transfer LoRA only if it stays in the download list.
- Engine table: add the ComfyUI node packs that the installer pins (ComfyUI-LTXVideo, ComfyUI-GGUF,
  ComfyUI-segment-anything-2, comfyui_controlnet_aux is already there). Check each licence.
- Pose table: DWPose "control of shots with several characters" is no longer the default (the pose guide is off). It is
  still the judge of the motion score.
- The header date "2026-10-19" should be updated after the re-check.

### 1.4 Code help texts and comments that are out of date (report only; another session edits code)

| # | Where | Now | Should say |
|---|---|---|---|
| C1 | cli.py:357 `app` help | "(Characters and Video tabs)" | Characters, Video, Templates, Backgrounds, Music, Settings |
| C2 | cli.py:447 `character create` help | "(Z-Image-Turbo)" | "(Z-Image)" |
| C3 | cli.py `run new --template` help | "a video template (made from text ...)" | "an imported template video (a stand-in figure in a scene)" |
| C4 | cli.py `shot add-video` help | "(... made from text)" | drop "made from text" (the text template generator was removed) |
| C5 | cli.py:750 `--profile` help | "ltx25dev = LTX-2.5 full (default)" | default: ltx25 (distilled) for a Blender shot with a library character; ltx25dev for video-to-video |
| C6 | cli.py `--ingredients-strength` help | "default 1.4" | "default 1.0" (runner.DEFAULTS) |
| C7 | cli.py `--sound-text` help | "default: a quiet ambience sentence" | "default: a breeze and ambient-sound sentence" (runner.SOUND_TEXT; AGENTS says never "quiet") |
| C8 | cli.py:379 token names | `["hf", "comfy"]` | `["hf", "comfy", "civitai"]` (settings.TOKENS has civitai) |
| C9 | cli.py `models status/download --feature` help | "video, character, music, select, pose" | + background, mocap |
| C10 | cli.py `--adain` help | "toward the look image's latent statistics" | toward clip 1's latent by default (`--adain-ref`) |
| C11 | character.py:1308 `redo_view` | `c.get("view_distance") or "wide shot"` | `or VIEW_DISTANCE` ("medium shot", AGENTS 4.5): older characters get "wide shot" on a redo |
| C12 | character.py:16 docstring | "redraws the sheet with LTX-2.3" | LTX-2.5 distilled (`runner.LAYOUT_PROFILE`, graph `ltx25_redraw`) |
| C13 | recipe.py docstring, runner.py:48 comment | "every tier uses the full models"; `"ltx23dev": the full model` | layout runs use the distilled LTX-2.5; PROFILE is ltx25dev (video-to-video only) |
| C14 | setup/models.json `model_tiers._about` | "Every tier uses the FULL (non-distilled) models" | the layout video uses the distilled LTX-2.5 int8 in both tiers |
| C15 | setup/models.json tier `8gb.about` | "int8 video model (LTX-2.5 dev)" | "int8 LTX-2.5 (distilled for layout videos, dev for video-to-video)" |
| C16 | setup/runtime.lock.json `pod.volume` | "8gb tier ~150 GB, large tier ~220 GB" | ~200 GB / ~275 GB |
| C17 | figure_mask.py docstring | "weights in setup/models.json profile sam2" | the fp16 file in <ComfyUI>/models/sam2 (see 2.4) |

Also noticed: MCP has no tool for `character repose` or `run fork`, although AGENTS.md's flow uses repose. That is
not an error, only a gap.

---------------------------------------------------------------------------------------------------------------------

## Part 2: downloads

### 2.1 What the current pipeline needs

From `setup/models.json` `model_tiers` and the graphs the code reaches (`ProfileWorkflow` / `image_job` calls,
`runner.VARIANTS` and the layout graph copies).

| Feature | Slots (8gb file / large file when different) | GB 8gb / large |
|---|---|---|
| video (layout, default) | ltx25_distilled (int8, both tiers), ltx25_text (Gemma 4 int8 / bf16), ltx25_vae, ltx25_audio_vae, ltx25_upscale, ic_layout, ic_ingredients25, qwen_edit_model + qwen_edit_text + qwen_vae (the look image) | |
| video (video-to-video only) | ltx25_model (dev int8 21.5 / bf16 42.0), ic_union (LTX-2.3 union LoRA, 0.65; also the off-by-default pose guide) | |
| video total | | 94.6 / 153.6 |
| character | zimage_model (z_image_bf16), zimage_text, zimage_vae, qwen_edit_model, qwen_edit_text, qwen_vae, qwen_angles, anypose_base, anypose_helper, **qwen_style_transfer (not used)** | 51.9 / 79.5 |
| music | ace_model (base / XL sft), ace_text1, ace_text2 (1.7b / 4b), ace_vae | 10.0 / 19.9 |
| background | pano_model (Qwen-Image-2512 Q8 GGUF), pano_lora, pano_depth (MoGe-2), qwen_edit_text, qwen_vae | 32.8 / 40.0 |
| select | sam2_large, sam2_tiny (**wrong files: see 2.4**) | 1.1 |
| pose | dwpose, yolox (in comfyui_controlnet_aux/ckpts) | 0.4 |
| mocap | sam3d_body | 2.8 |
| **all (unique files)** | 31 files | **153.9 / 222.7** |

The older flat profiles at the top of models.json (`ltx23`, `zimage`, `qwen_edit`, `sam2`) are read only by
`setup/download_models.py`. Nothing else uses them. `ltx23` still downloads the LTX-2.3 Ingredients LoRA (1.31 GB),
which no current graph uses. `zimage` has no model file (the turbo file was removed from it).

### 2.2 Which graphs are still reachable

| Graph(s) | Reachable? | Used by |
|---|---|---|
| `ltx25_layout`, `ltx25_layouting` | yes (default) | every Blender shot with a library character |
| `ltx25_redraw` | yes | `character style`, `background style` |
| `ltx25_ref/refup/restyle/v2v/v2vref` | yes | copied into every run; `--profile ltx25` video-to-video; `--restyle` |
| `ltx25dev_*` (all 8) | yes | video templates and character-less shots (v2v default `PROFILE`), `--profile ltx25dev` |
| `qwenedit_edit`, `qwenedit_angle`, `qwenedit_anypose` | yes | edits and look images, views, repose |
| `qwenedit_style` | **no** | no caller in cinestudio/ (only the dx8152 experiments of 2026-10-24) |
| `zimage_base` | yes | character fronts |
| `zimage_create` | **no** | Z-Image-Turbo (recipe FULL=True; the turbo file is deleted) |
| `acestep_base` | yes | songs |
| `acestep_song` | **no** | ACE turbo AIO (recipe FULL=True; the file is not on disk) |
| `qwenimage_pano`, `moge_pano`, `sam3d_body` | yes | backgrounds, 360 depth, templates from video |
| `ltx23_ref/refup/restyle/v2v/v2vref`, `ltx23dev_*` | **no** for new runs | `run new --profile` allows only ltx25dev / ltx25. A fork of an old ltx23 run would copy them, but their model files are gone. |
| `ltx23_clip`, `ltx23_cont`, `ltx23_create`, `ltx23_ing`, `ltx23_turntable` | **no** | no variant name in the code |

The qwenedit graphs still carry an optional Lightning LoRA node. It is removed at build time (fast mode is off). That
is harmless. If the unreachable graphs are removed later, update tests/unit/test_server.py and
tests/unit/test_workflow_video.py (they read ltx23 graphs) and the integration scripts.

### 2.3 What is on disk, classified

Folders: `D:\Comfy-Desktop\ComfyUI-Shared\models` (Comfy Desktop's paths file; cinestudio `models_dir()` resolves
here; 197.3 GB in all) and ComfyUI's own `D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\models` (only `sam2/`
has files) plus `custom_nodes\comfyui_controlnet_aux\ckpts`. cinestudio's settings have no `models_dir`. No
`.part` or other partial files were found. The HF cache (%USERPROFILE%\.cache\huggingface, 57 GB) holds no
cinestudio files.

**(a) Needed by the 8gb tier now: 153.0 GB (keep)**

| File | GB | Note |
|---|---|---|
| diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot | 21.50 | default video |
| diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot | 21.50 | video-to-video only (video templates) |
| text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot | 15.37 | |
| vae/ltx-2.5-video-vae-bf16, vae/ltx-2.5-audio-vae-bf16 | 1.47 + 0.36 | |
| latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0 | 1.00 | |
| loras/ltx-2.5-22b-ic-lora-layout-to-render-1.0, loras/ltx-2.5-22b-ic-lora-ingredients-0.9 | 1.31 + 1.31 | |
| loras/ltx-2.3-22b-ic-lora-union-control-ref0.5 | 0.65 | v2v graphs and the pose guide |
| diffusion_models/qwen_image_edit_2511_int8_convrot | 20.50 | |
| text_encoders/qwen_2.5_vl_7b_fp8_scaled, vae/qwen_image_vae | 9.38 + 0.25 | |
| loras/qwen-image-edit-2511-multiple-angles-lora, loras/2511-AnyPose-base, loras/2511-AnyPose-helper | 0.30 x 3 | |
| diffusion_models/z_image_bf16, text_encoders/qwen_3_4b, vae/ae | 12.31 + 8.04 + 0.34 | |
| diffusion_models/qwen-image-2512-Q8_0.gguf, loras/qwen-360-diffusion-2512-int8-bf16-v2, geometry_estimation/moge_2_vitl_normal_fp16 | 21.76 + 0.76 + 0.66 | |
| diffusion_models/acestep_v1.5_base, text_encoders/qwen_0.6b_ace15, text_encoders/qwen_1.7b_ace15, vae/ace_1.5_vae | 4.79 + 1.19 + 3.71 + 0.34 | |
| detection/sam_3d_body_dinov3_bf16 | 2.83 | |
| controlnet_aux ckpts: dw-ll_ucoco_384_bs5.torchscript.pt, yolox_l.torchscript.pt | 0.14 + 0.22 | |
| ComfyUI's own models/sam2/sam2.1_hiera_large-fp16 | 0.45 | the file the SAM 2 node really loads (not in the setup list) |

**(b) Needed by the large tier only:** none on disk. (The large files are for RunPod.)

**(c) Only old runs or old / unused features: 23.2 GB**

| File | GB | Why |
|---|---|---|
| diffusion_models/qwen_image_edit_2511_fp8mixed | 20.53 | replaced by int8 ConvRot today (same picture, 1.57x faster). Frozen into all 11 LTX-2.5 layout runs (see 2.5). |
| loras/qwen-image-edit-2511-style-transfer-1_20 (dx8152) | 0.24 | tested today: it changed almost nothing (AGENTS 4.9). No reachable graph. Still in the setup list. |
| loras/Fantasy_Anime | 0.35 | DELETED 2026-10-27 (user): the LTX-2.3 style LoRA of the old ltx23 runs; it made LTX-2.5 worse. |
| checkpoints/sdpose_wholebody_fp16 + diffusion_models/rt_detr_v4-x-hgnet_fp16 | 1.92 + 0.12 | SDPose, the second pose detector of the bench (`pose_people.py --detector sdpose|fusion`; default dwpose). Not in the setup list. |

**(d) Not referenced by cinestudio: 20.2 GB (probably the user's other ComfyUI work; dated before this project. Do not
delete without the user's check)**

| File | GB | Date |
|---|---|---|
| diffusion_models/acestep_v1.5_turbo | 4.79 | 2026-09-14 |
| checkpoints/stable_audio_3_small_sfx | 2.27 | 2026-09-14 |
| text_encoders/t5gemma_b_b_ul2 | 1.19 | 2026-09-14 |
| diffusion_models/minimax_music3_dit_int8_convrot | 2.50 | 2026-08-29 |
| text_encoders/minimax_music3_text_encoder_pruned_int8_convrot | 9.20 | 2026-08-29 |
| vae/minimax_music3_dav | 0.22 | 2026-08-29 |

The MiniMax files must never be used by cinestudio (project rule). They are not touched by it.

Also on disk and used: loras/3dsrx_1250 (0.67 GB), the user's own LTX-2.5 3D style LoRA. All 11 layout runs used it.
Keep.

**(e) Duplicates and copies that are never loaded: 2.1 GB**

The SAM 2 node (ComfyUI-segment-anything-2, `DownloadAndLoadSAM2Model`) reads only `folder_paths.models_dir/sam2` =
ComfyUI's OWN models folder, not the shared folder. With `precision: "fp16"` (what figure_mask.py sends) it loads
`sam2.1_hiera_large-fp16.safetensors` only.

| File | GB | Why |
|---|---|---|
| ComfyUI-Shared/models/sam2/sam2.1_hiera_large | 0.90 | the setup list downloads it, but the node never reads this folder |
| ComfyUI-Shared/models/sam2/sam2.1_hiera_tiny | 0.16 | same; also no code uses "tiny" |
| ComfyUI's own models/sam2/sam2.1_hiera_large | 0.90 | fp32 copy; fp16 is loaded |
| ComfyUI's own models/sam2/sam2.1_hiera_tiny | 0.16 | not used |

Risk: `models status` checks the shared sam2 files. If they go before the list is fixed (2.4), the "select" feature
shows as not ready and a download fetches them again.

Also absent already (nothing to do): the LTX-2.3 distilled fp8 checkpoint, Gemma 3 fp4, the 2.3 Ingredients LoRA,
the 2.3 upscalers, Z-Image-Turbo, the Lightning LoRA, the ACE turbo AIO file.

### 2.4 Setup-list problems and proposed changes

1. **Fix the distilled LTX-2.5 hash (bug).** `model_tiers.common.ltx25_distilled.sha256` is
   `"****************************************************************"` (a placeholder or a masked value).
   `models.py:317` checks it after a download, so a new install fails on this file. `models verify` reports the local
   file as bad. Compute the real value when the GPU is idle (a 21.5 GB read; I did not run it because a Qwen job was
   running): `certutil -hashfile "D:\Comfy-Desktop\ComfyUI-Shared\models\diffusion_models\ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" SHA256`,
   or read the LFS oid on the HF file page with a token. The same placeholder is frozen in the 11 layout runs'
   manifests (only a record).
2. **SAM 2.1: list the file the node loads.** Replace `sam2_large` and `sam2_tiny` (and the flat `sam2` profile) with:
   `"sam2_large_fp16": {"dir": "sam2", "file": "sam2.1_hiera_large-fp16.safetensors", "url":
   "https://huggingface.co/Kijai/sam2-safetensors/resolve/f885607d88bb3f9145efa49c3e3c50a9e5bf13eb/sam2.1_hiera_large-fp16.safetensors",
   "size": 448999828, "sha256": "278af5296fcc153aa94645a4981e93fe8b7a936a9fa27dbd71e8c850df6d7038"}`
   (hash checked against the local file and the HF page today). The `select` feature then lists only this slot.
   Saves 0.6 GB per install. On Windows/Desktop, `models_dir()` points at the shared folder, but the node reads
   ComfyUI's own `models/sam2`: give this slot a special dir (like `@aux/`, e.g. `@comfy/sam2` = `<ComfyUI>/models/sam2`).
   The Linux installer links ComfyUI's models folder to the volume, so it is right there already. Without this fix the
   node downloads the fp16 file itself at the first `shot mask` (a network call at run time).
3. **Remove `qwen_style_transfer`** from `common` and `features.character` (0.24 GB). No graph of the app uses it,
   and it changed almost nothing in the test. Remove or keep `workflows/qwenedit_style*` as an experiment.
4. **Optional: split video-to-video out of "video".** Make a feature `video_v2v` = `ltx25_model`, `ic_union`. A user
   who only makes Blender-shot layout videos then saves 22.2 GB (8gb) or 42.7 GB (large). This needs a code check:
   `run new` should refuse v2v when that feature is missing.
5. **Retire the flat profiles** `ltx23`, `zimage`, `qwen_edit`, `sam2` and `setup/download_models.py` (or make it read
   the tiers). `ltx23` downloads an unused 1.31 GB LoRA; `zimage` is incomplete. README and MODELS.md no longer point
   to it.
6. Update the `_about` texts (C14, C15) and `runtime.lock.json pod.volume` (C16).
7. Nothing else is missing. Every model file named in the reachable graphs is in the tier lists (the Lightning LoRA
   node is optional and off).

Optional code change that lowers the cost of deleting fp8mixed: in `runner._freeze_tier`, map a frozen
`qwen_image_edit_2511_fp8mixed.safetensors` that is missing on disk to the tier's current `qwen_edit_model` (the
int8 file gives the same picture). Then old layout runs keep working without the 20.5 GB file.

### 2.5 Old runs that reference files proposed for removal

58 runs in `.cinestudio/runs`. All have status `done` and a `final.mp4` (the videos are safe; only a resume, retake
or fork needs model files).

- **47 runs r20261018-230415 ... r20261020-205032 (profile ltx23):** they need LTX-2.3 distilled fp8, Gemma 3 fp4
  and the 2.3 LoRAs, which are ALREADY gone. They cannot be resumed or forked now. Fantasy_Anime is their style LoRA.
  Deleting it changes nothing for them.
- **11 runs r20261022-091953 ... r20261024-164309 (profile ltx25, layout, tier 8gb):** their manifests freeze
  `qwen_edit_model = qwen_image_edit_2511_fp8mixed.safetensors`. A fork keeps the parent's frozen models
  (`_freeze_tier(..., parent.files)`). Any retake or fork that makes a new look or keyframe look (a Qwen edit) would
  need fp8mixed again, unless the mapping above is added. They also use 3dsrx_1250 (keep) and freeze the union LoRA
  name (kept).

---------------------------------------------------------------------------------------------------------------------

## Proposed deletions (the user decides; nothing was deleted)

| File | Size | Class | Why | Risk |
|---|---|---|---|---|
| ComfyUI-Shared/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors | 20.53 GB | c | replaced by int8 ConvRot (same picture, faster) | retake/fork of the 11 layout runs that redoes a look needs it (unless the `_freeze_tier` mapping is added); re-download 20.5 GB |
| ComfyUI-Shared/models/loras/qwen-image-edit-2511-style-transfer-1_20.safetensors | 0.24 GB | c | tested useless; no graph of the app uses it | `models status` shows "character" not ready until the list drops it (2.4 item 3) |
| ComfyUI-Shared/models/sam2/sam2.1_hiera_large.safetensors | 0.90 GB | e | the node never reads the shared folder | `models status` shows "select" not ready until the list is fixed (2.4 item 2) |
| ComfyUI-Shared/models/sam2/sam2.1_hiera_tiny.safetensors | 0.16 GB | e | never used | same |
| ComfyUI/models/sam2/sam2.1_hiera_large.safetensors (ComfyUI's own folder) | 0.90 GB | e | fp32 copy; the fp16 file is loaded | none for cinestudio; another ComfyUI workflow at fp32 would download it again |
| ComfyUI/models/sam2/sam2.1_hiera_tiny.safetensors (ComfyUI's own folder) | 0.16 GB | e | never used | as above |
| ComfyUI-Shared/models/checkpoints/sdpose_wholebody_fp16.safetensors | 1.92 GB | c | SDPose bench only (default detector is DWPose) | `--detector sdpose|fusion` stops working |
| ComfyUI-Shared/models/diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors | 0.12 GB | c | SDPose's person finder | as above |
| ComfyUI-Shared/models/loras/Fantasy_Anime.safetensors | 0.35 GB | c | DELETED 2026-10-27 (user) | - |
| **Total (classes c + e)** | **25.3 GB** | | | |

Not proposed (class d, 20.2 GB: acestep_v1.5_turbo, stable_audio_3_small_sfx, t5gemma_b_b_ul2, the three
minimax_music3 files): cinestudio does not use them, but they predate the project and may be the user's other work.
Ask the user first. Keep everything in class (a) and 3dsrx_1250.

## Setup-list changes (summary)

1. `ltx25_distilled.sha256`: replace the 64-`*` placeholder with the real hash (bug: new downloads fail).
2. `sam2_large` + `sam2_tiny` -> `sam2_large_fp16` (`sam2.1_hiera_large-fp16.safetensors`, 448999828 bytes, sha256
   278af529...6d7038), in ComfyUI's own `models/sam2`; drop the flat `sam2` profile.
3. Remove `qwen_style_transfer` from `common` and `features.character`.
4. Optional: a `video_v2v` feature for `ltx25_model` + `ic_union` (saves 22-43 GB for layout-only users).
5. Remove the flat profiles `ltx23`, `zimage`, `qwen_edit` (and retire `setup/download_models.py`).
6. Fix the `_about` texts (models.json `model_tiers._about`, `tiers.8gb.about`) and `runtime.lock.json pod.volume`.

## Doc edits made

- README.md: intro note, layout block, local files line, tokens (HF needed, Civitai, `CIVITAI_TOKEN`), tier sizes
  (154 / 223 GB), distilled vs dev, the models commands, the shared-ComfyUI restart section, queue kinds, the 26 MCP
  tools.
- docs/MODELS.md: rewritten for the tiers (files, sizes, sources, licences, features, style LoRA, licence notes).
- docs/RUNPOD.md: volume sizes (200 / 275 GB), `HF_TOKEN` needed.
- docs/docs_and_downloads_review.md: this report (new).
