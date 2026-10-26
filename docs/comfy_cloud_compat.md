# Comfy Cloud and cinestudio: does it work? (research, 2026-10-24)

Scope: web research, a read of our repo, and read-only, free GET requests to cloud.comfy.org. No job was submitted,
nothing was uploaded, no code was changed. Marks: **[checked]** = we read it from the live API or our code today.
**[docs]** = Comfy's documentation says so (source at the end). **[est.]** = our own estimate. **unverified** = nobody
has confirmed it yet.

## 0. Summary

- **No Comfy key is saved yet** [checked]. `.cinestudio/secrets.json` has only `hf` and `civitai`, and
  `COMFY_API_KEY` is not set. So the two checks that need a key (`GET /api/user`, `GET /api/object_info`) answered
  401. Every other check below used public, free endpoints.
- **Our client already speaks Comfy Cloud.** comfy-sdk 0.5.0 targets `https://cloud.comfy.org` by default (Comfy API
  v2, `/api/v2/jobs`, `/api/v2/assets`, Bearer key) [checked: `comfy_sdk/client.py`]. comfy-api-proxy is only for a
  local ComfyUI; Cloud does not need it.
- **Cloud runs ComfyUI v0.42.5** [checked: `/api/system_stats`]. We run v0.42.1. 60 of our 68 node types are core
  ComfyUI nodes, so they should exist on Cloud (unverified until `/api/object_info` is read with a key).
- **8 node types come from custom packs:** 4 from ComfyUI-LTXVideo, 1 from ComfyUI-GGUF, 3 from kijai's
  ComfyUI-segment-anything-2. Users cannot install custom nodes on Cloud [docs]. Whether these 8 exist there is
  **unverified**. The 4 LTXVideo nodes are a blocker for the video if they are missing.
- **Models: 22 of our 27 model files are already in Cloud's library, with the exact file names** [checked: public
  `/api/experiment/models`]. That includes the LTX-2.5 distilled int8 ConvRot transformer, the Gemma 4 int8 ConvRot
  encoder, Qwen-Image-Edit-2511 int8 ConvRot, AnyPose, Multiple-Angles, ACE-Step 1.5 base, MoGe-2, SAM 3D Body, SAM2.1
  and the LTX-2.3 union control LoRA.
- **Missing models:** the two LTX-2.5 IC-LoRAs (Layout-To-Render, Ingredients; both gated on Hugging Face), the
  Qwen-Image-2512 Q8 GGUF, the Qwen style-transfer LoRA, and the user's own LTX style LoRA (3dsrx). Users can import
  `.safetensors` files from Hugging Face or Civitai, but only on the **Creator plan or higher** [docs]. GGUF cannot be
  imported; the panorama graph would use Cloud's `qwen_image_2512_bf16` instead.
- **Verdict: works with changes** (section 6), about 1-1.5 weeks of work, IF the 4 LTXVideo nodes exist on Cloud and
  the gated IC-LoRAs import. Plan: Creator ($35/month, 7,400 credits). Cost per run [est.]: about $0.20-0.85 for a
  3-clip preview run and $0.30-1.10 for a 3-clip final run.

## 1. What Comfy Cloud offers through its API

| Item | Answer | Mark |
|---|---|---|
| API versions | Comfy API v2 (current, what comfy-sdk uses) and the v1 Cloud API (deprecated, still runs). | [docs] |
| Base URL, auth | `https://cloud.comfy.org`. v2: `Authorization: Bearer <key>`. v1: header `X-API-Key`. | [docs], [checked] |
| Plan needed | API access needs a paid plan. A key on an inactive plan gets HTTP 429 (our `check_token` already says so). | [docs] |
| Submit | v2 `POST /api/v2/jobs` with the API-format graph. Optional `Idempotency-Key` (single use, 24 h). UI-format graphs are refused. Cloud refuses non-empty `metadata` (we send none). | [docs] |
| Inputs | v2 `POST /api/v2/assets` (multipart). Cloud: 200 MB per upload, images max 50 MiB, 10 min per upload. Content type is a free string; no allow-list is documented. Uploads are de-duplicated by blake3 hash. Graphs reference uploads as `core/ASSET` objects (comfy-sdk does this). | [docs] |
| Outputs | Job outputs are a flat list with `type` = `image`, `video`, `audio`, `text`, `file` or `latent`, each with an asset id and a signed URL (~6 h). Job records are kept ~30 days (a placeholder, not a promise). | [docs] |
| Status, progress | `GET /api/v2/jobs/{id}` (queued, running, succeeded, canceling, canceled, failed, expired) and an SSE event stream. | [docs] |
| Cancel | Job-scoped `POST /api/v2/jobs/{id}/cancel`. No global interrupt needed. | [docs] |
| Concurrency | Standard 1, Creator 3, Pro 5 jobs at once; up to 100 queued. A full queue gives 429 `queue_full`. | [docs] |
| Time limit per job | 30 min (Standard, Creator), 1 h (Pro). Longer jobs are canceled. | [docs] |
| GPU | RTX PRO 6000 Blackwell, 96 GB VRAM, ~180 GB RAM. | [docs] |
| Price | Credits pay for active GPU time only (not idle, not editing). The Cloud rate was 0.266 credits per GPU second after the 2026-01-23 price cut (blog). The pricing page's "Comfy API" table lists the RTX PRO 6000 at 736.39 credits/h ($3.49/h), i.e. 0.205 credits/s. Plans: Standard $20 = 4,200 credits/month, Creator $35 = 7,400, Pro $100 = 21,100 (about 210 credits per dollar). Which rate applies to API jobs today is unverified; we use 0.266 as the upper bound. | [docs] |
| Model import | Creator plan or higher. Hugging Face or Civitai links only, `.safetensors` only, up to 100 GB, private to the user. Gated or private repos: save an HF / Civitai token in Cloud's Secrets (the docs' FAQ contradicts this; unverified). | [docs] |
| Custom nodes | Only the packs Comfy pre-installs. No user installs. A request form exists. | [docs] |
| Free, public reads | `/api/system_stats`, `/api/features`, `/api/experiment/models`, `/api/experiment/models/{folder}` answered without a key. | [checked] |

## 2. Node coverage

Method: a script collected every `class_type` from `workflows/*_api.json` for ltx25_layout, ltx25_layouting,
qwenedit_edit, qwenedit_angle, qwenedit_anypose, qwenedit_style, zimage_base, acestep_base, qwenimage_pano, moge_pano and
sam3d_body, plus the nodes our code adds at run time (`runner.py`: relay, audio carry, AdaIN, pose guide; `figure_mask.py`:
SAM2 shot mask; `character.py` adds none). Result: **68 node types**. Their source pack was read from our local
ComfyUI's `/object_info/<node>` (`python_module`) [checked].

- **60 core nodes** (`nodes`, `comfy_extras`), including all LTXV core nodes, LoadLatent / SaveLatent, LoadVideo,
  SaveGLB, the MoGe nodes (`nodes_moge`), SAM 3D Body (`nodes_sam3d_body`) and the ACE-Step 1.5 nodes. Cloud runs a
  newer core (v0.42.5), so these should all exist (unverified until `/api/object_info` is read).
- **8 custom-pack nodes:**

| Pack | Nodes | Used by | If missing on Cloud |
|---|---|---|---|
| ComfyUI-LTXVideo (Lightricks) | `LTXICLoRALoaderModelOnly`, `LTXAddVideoICLoRAGuide` | every layout video clip | **Blocker.** The Layout-To-Render IC-LoRA cannot be loaded or guided. A rewrite with core nodes is not known to be possible. |
| ComfyUI-LTXVideo | `LTXVAdainLatent`, `LTXVSelectLatents` | AdaIN 0.3 and the latent relay of long shots | The chain fixes of section 7.1 of AGENTS.md stop working. |
| ComfyUI-GGUF | `UnetLoaderGGUF` | qwenimage_pano | Not a problem: use core `UNETLoader` with Cloud's `qwen_image_2512_bf16` (96 GB fits it). |
| ComfyUI-segment-anything-2 (kijai) | `DownloadAndLoadSAM2Model`, `Sam2VideoSegmentation`, `Sam2VideoSegmentationAddPoints` | the shot mask (`figure_mask.py`) | Keep that small job local, or move to core SAM 3 (Cloud has a `sam3` folder). |

Hints, not proof: Cloud's model library holds Lightricks' LTX-2.x IC-LoRAs and LTXVideo-pack prompt-enhancer files
(`LLM/Florence-2-large-PromptGen-v2.0`, `ltxv_*` names), and its `sam2` folder holds kijai's file names
(`sam2.1_hiera_*-fp16.safetensors`). A cached search snippet of `comfy.org/cloud/supported-nodes/ComfyUI-LTXVideo`
said the pack is "preinstalled" with "19 nodes"; that page now redirects, and 19 nodes may not include ours
(unverified). No GGUF diffusion model is in the library (only SeedVR2 GGUFs), so ComfyUI-GGUF may be absent.

**To finish this check** (free, read-only), save the key (`python -m cinestudio settings token set comfy`), then run
this from the repo root. It prints only counts and the missing node types, never the key. It was tested against our
local ComfyUI (`--url http://127.0.0.1:8188/object_info`: 68 types, 0 missing).

```python
"""Compare cinestudio's node types with Comfy Cloud's /api/object_info (read-only GET). Run from the repo root."""
import json, re, sys, pathlib, urllib.request
sys.path.insert(0, ".")
from cinestudio import settings
GRAPHS = ["ltx25_layout", "ltx25_layouting", "qwenedit_edit", "qwenedit_angle", "qwenedit_anypose", "zimage_base",
          "acestep_base", "qwenimage_pano", "moge_pano", "sam3d_body"]
ours = {}
for g in GRAPHS:
    for n in json.loads(pathlib.Path(f"workflows/{g}_api.json").read_text(encoding="utf-8")).values():
        if isinstance(n, dict) and "class_type" in n:
            ours.setdefault(n["class_type"], set()).add(g)
for f in ("cinestudio/runner.py", "cinestudio/figure_mask.py", "cinestudio/character.py"):
    for c in re.findall(r'"class_type":\s*"([A-Za-z0-9_.]+)"', pathlib.Path(f).read_text(encoding="utf-8")):
        ours.setdefault(c, set()).add(pathlib.Path(f).stem)
url = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else "https://cloud.comfy.org/api/object_info"
h = {"User-Agent": "cinestudio"}
if url.startswith("https://cloud.comfy.org/"):
    key = settings.token("comfy")
    if not key:
        sys.exit("no Comfy key set (settings token set comfy)")
    h["X-API-Key"] = key
info = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120))
missing = {c: sorted(u) for c, u in ours.items() if c not in info}
print(json.dumps({"node_types": len(ours), "cloud_nodes": len(info), "missing": missing}, indent=1))
```

Also check, in the same `object_info`, that the LoRA and model combo lists accept our file names (section 3).

## 3. Model coverage

Read from the public `GET /api/experiment/models` (140 folders) and `/api/experiment/models/{folder}` (2,133 files)
on 2026-10-24 [checked]. File names are matched exactly.

| Our file | On Cloud? | Cloud folder |
|---|---|---|
| ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors | yes | diffusion_models |
| gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors | yes | text_encoders |
| ltx-2.5-video-vae-bf16 / ltx-2.5-audio-vae-bf16 | yes | vae |
| ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0 | yes | latent_upscale_models |
| **ltx-2.5-22b-ic-lora-layout-to-render-1.0** | **no** (gated HF repo) | import into loras |
| **ltx-2.5-22b-ic-lora-ingredients-0.9** | **no** (gated HF repo; Cloud has only the 2.3 version) | import into loras |
| ltx-2.3-22b-ic-lora-union-control-ref0.5 (pose guide, off by default) | yes | loras |
| qwen_image_edit_2511_int8_convrot | yes | diffusion_models |
| qwen_2.5_vl_7b_fp8_scaled, qwen_image_vae | yes | text_encoders, vae |
| Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16 | yes | loras |
| 2511-AnyPose-base-000006250, 2511-AnyPose-helper-00006000 | yes | loras |
| qwen-image-edit-2511-multiple-angles-lora | yes | loras |
| qwen-image-edit-2511-style-transfer-1_20 (qwenedit_style; "changed almost nothing", AGENTS 4.9) | no | import, or drop |
| z_image_bf16, qwen_3_4b, ae | yes | diffusion_models, text_encoders, vae |
| acestep_v1.5_base, qwen_0.6b_ace15, qwen_1.7b_ace15, ace_1.5_vae | yes | diffusion_models, text_encoders, vae |
| **qwen-image-2512-Q8_0.gguf** | **no**; GGUF cannot be imported | use `qwen_image_2512_bf16` (or fp8) |
| qwen-360-diffusion-2512-int8-bf16-v2 | yes | loras |
| moge_2_vitl_normal_fp16 | yes | geometry_estimation |
| sam_3d_body_dinov3_bf16 | yes | detection |
| sam2.1_hiera_large / base_plus | yes | sam2 |
| STYLE_LORA (the user's 3dsrx_1250 LTX-2.5 style LoRA, and any Civitai LoRA) | no | import (Creator plan) |

Notes:
- Cloud also has the bf16 versions (LTX-2.5 distilled and dev, Gemma, Qwen-Edit-2511). On an RTX PRO 6000 int8 is
  the faster choice (docs/large_gpu_speed_research.md), so the 8gb tier's files are a good default there too.
- Import needs the files on Hugging Face or Civitai. The two IC-LoRAs are gated: the user's HF token must be saved in
  Cloud's Secrets and must have accepted the LTX licence (unverified that gated imports work). The 3dsrx LoRA must
  first be uploaded to a (public or private) HF or Civitai repo.
- Whether Cloud keeps the original file name of an imported model (our graphs name the files exactly) is unverified.
  The public listing `private_models_enabled: false` is the anonymous default; the per-user value is unknown.

## 4. Our special patterns on Cloud

| Pattern | Today (local) | On Cloud |
|---|---|---|
| Submit, watch, cancel | comfy-sdk -> comfy-api-proxy -> ComfyUI | comfy-sdk -> Cloud v2 directly. Idempotency keys, `find(prefix)` and job-scoped cancel map 1:1 (the `GET /jobs/{id}/workflow` route used by `find` is unverified on Cloud). |
| Input images, videos | uploaded as SDK assets (`files` of `SdkBackend.submit`) | same. Limits 200 MB / 50 MiB per image are far above our inputs. LoadVideo with an uploaded mp4 is unverified. |
| Latent relay, audio carry (SaveLatent -> LoadLatent) | `runner.relay_latent` and the audio tail COPY files from ComfyUI's output folder into its input folder (`comfy.io_dir`) | **Must change.** No filesystem on Cloud. Download the SaveLatent output (v2 output type `latent`) and upload it as an asset bound to LoadLatent. Our latents are small (~3 MB [est.]). LoadLatent with a `core/ASSET` input is unverified. |
| GLB outputs (MoGe, SAM 3D Body) and some audio/video downloads | `server.download_output` reads ComfyUI's own `/history` and `/view` (backend fallback, song.py, figure_mask.py, recovery) | **Must change** to `backend.download` (v2 type `file` / `audio` / `video`). GLB as `file` is unverified. |
| Our GPU lock and queue | one GPU job at a time | The lock is not needed for Cloud jobs. Keep our queue (order, tickets, user cancel). Later allow 3 parallel jobs on Creator (5 on Pro). Blender and ffmpeg still run locally. |
| `ensure_family` restarts, `/free` | restart the local engine at each model-family change | Skip on Cloud: Comfy manages memory. Never needed there. |
| `workqueue` counts "other" ComfyUI jobs | `server.queue_items(comfyui_url())` | Skip on Cloud (no local engine). |
| Long jobs | final clip ~6 min, Qwen edit ~4-8 min on the 3070 | Well under 30 min on an RTX PRO 6000 [est.: a final clip ~1-2 min, a Qwen edit ~1-2 min]. Cold model loads on Cloud may count as billed GPU time (unverified). |
| SageAttention per family | our start flag | Cloud's own setting; we cannot change it. Qwen gave black images with Sage on our side; Cloud's Qwen templates suggest it is fine there (unverified). |
| comfy-api-proxy 0.1.6 | needed locally | Not used for Cloud. |
| comfy-sdk 0.5.0 | `COMFY_BASE_URL` set in `os.environ` by `SdkBackend` | Works for Cloud with `base_url="https://cloud.comfy.org"` and the key. The env variable is process-wide: one process should not mix local and Cloud jobs. |

## 5. Licence and terms

- **Outputs belong to the user.** Terms of Service, effective 2026-05-13, section 2: the customer "retains all right,
  title, and interest" in inputs and outputs. Comfy "will not use Input or Output to train generative AI or diffusion
  models". Inputs, outputs and workflows are private by default (support article).
- **Commercial use** of outputs is not named in the ToS. The pricing page says models offered through Comfy can be
  used commercially. The ToS grants access "for your internal business purposes": fine for one user's own account;
  each cinestudio user needs their own Comfy account and key (we never share one key).
- **Uploaded models:** the ToS forbids "submitting models, images, or other materials without the right to do so".
  Our models are under commercial-safe licences (AGENTS.md section 2). Check that the LTX-2.5 licence allows running
  the weights on a third-party host for the licensee (likely, unverified). The 3dsrx LoRA is the user's own.
- **Impersonation:** the ToS forbids synthetic media that impersonates a real person without consent. Our use
  (invented characters) does not conflict.
- **Privacy note:** Cloud mode sends the user's character images, sheets and shot renders to Comfy. The app should say
  so in the Cloud switch.

## 6. Verdict, changes, effort and cost

**Verdict: works with changes**, under two conditions that only a key can confirm: (1) the 4 ComfyUI-LTXVideo nodes
exist on Cloud, (2) the gated LTX-2.5 IC-LoRAs import. If (1) fails, the video pipeline does NOT work on Cloud;
characters (Z-Image, Qwen-Edit, AnyPose, views), songs, panoramas and depth would still work.

Changes (proposals only, no code was changed):

| # | Change | Effort [est.] |
|---|---|---|
| 1 | Backend switch: build `SdkBackend(base_url="https://cloud.comfy.org", api_key=settings.token("comfy"))` when `settings.backend == "cloud"`; pass the key, never log it. | 0.5 day |
| 2 | Bypass every local-engine call in Cloud mode: `comfy start`, `server.ensure_family`, `note_owned`, `queue_items`, `prompt_status`, the GPU lock for Cloud jobs; `doctor` checks Cloud (key, plan, models, nodes) instead of local files. | 1-2 days |
| 3 | Files through the API only: relay latents and the audio tail via v2 `latent` outputs and asset uploads; `server.download_output` fallbacks (GLB, song audio, figure mask, recovery) via `backend.download`. | 2-3 days |
| 4 | Models: import the two LTX-2.5 IC-LoRAs and the user's style LoRAs (Creator plan); a Cloud variant of qwenimage_pano with `UNETLoader` + `qwen_image_2512_bf16`; drop or import the style-transfer LoRA; a name map if Cloud renames imports. | 1 day + tests |
| 5 | Shot mask: if kijai's SAM2 is missing, keep it local or port to core SAM 3. | 0-1 day |
| 6 | App: enable the Cloud switch with a cost and privacy note; optional credit display (`/api/billing/usage/timeseries`). Later: parallel jobs up to the plan's limit. | 1 day |
| 7 | Tests: unit tests with a fake backend; one paid smoke test (a preview clip) only with the user's approval. | 1 day |

Total: about 1-1.5 weeks, plus a few dollars of credits for the smoke test.

**Cost [est.]** at 0.266 credits per GPU second (~958 credits/h, ~$4.56 per GPU hour at ~210 credits/$; the lower
0.205 credits/s rate makes everything ~25% cheaper). Job times on the RTX PRO 6000 are estimates from our 3070 log
(stage 1 at 11.2 s/step, stage 2 at 41.6 s/step, Qwen int8 at 11.4 s/step) and a ~6-8x faster GPU without RAM
streaming; model loads are included roughly.

| Work | GPU time [est.] | Credits | Dollars |
|---|---|---|---|
| Preview run, 3 clips (1 look image + 0-3 keyframe looks + 3 stage-1 clips) | 2.5-11 min | 40-175 | $0.19-0.84 |
| Final run, 3 clips (same looks + 3 two-stage clips) | 4-14 min | 65-225 | $0.30-1.06 |
| New character (8 Z-Image fronts + repose + 3 views) | 6-12 min | 95-190 | $0.45-0.90 |
| One Qwen edit | 1-2 min | 15-30 | $0.07-0.15 |

A Creator plan (7,400 credits/month) would cover about 40-180 preview runs or 30-110 final runs per month [est.].
Compare: a RunPod RTX PRO 6000 pod costs ~$2.49/h but bills idle time and needs a network volume
(docs/large_gpu_speed_research.md); Cloud bills only active GPU time and has the models already.

## 7. Requests made (all GET, all read-only and free)

| Request | Answer |
|---|---|
| `GET /api/user` (no key available) | 401 |
| `GET /api/object_info` (no key available) | 401 |
| `GET /api/system_stats` | 200: `comfyui_version` v0.42.5, `cloud_version` v0.494.0 |
| `GET /api/features` | 200: e.g. `max_upload_size` 104857600, `max_concurrent_jobs` 1, `user_secrets_enabled` true (anonymous defaults) |
| `GET /api/experiment/models` and 140 x `/api/experiment/models/{folder}` | 200: 2,133 model files |
| `GET /api/workflow_templates`, `/api/global_subgraphs` | 200: no match for our custom nodes |
| local `GET http://127.0.0.1:8188/object_info/<node>` (68 x) | node -> pack map (section 2) |

## Sources (all read 2026-10-24)

- Comfy Cloud v1 API overview (deprecated): https://docs.comfy.org/development/cloud/overview
- Cloud API reference: https://docs.comfy.org/development/cloud/api-reference
- Cloud OpenAPI spec: https://docs.comfy.org/openapi-cloud.yaml
- Usage and limits (concurrency, 30 min / 1 h): https://docs.comfy.org/development/deploy/cloud-usage
- Cloud quickstart with comfy-sdk: https://docs.comfy.org/development/deploy/cloud
- Comfy API v2 overview: https://docs.comfy.org/api-reference/v2/overview
- v2 submit: https://docs.comfy.org/api-reference/v2/jobs/submit-a-workflow-for-execution
- v2 job status and outputs: https://docs.comfy.org/api-reference/v2/jobs/job-status-the-polling-workhorse
- v2 asset upload: https://docs.comfy.org/api-reference/v2/assets/upload-an-asset-single-call-multipart
- Import models (Creator+, HF/Civitai, safetensors): https://docs.comfy.org/cloud/import-models
- Custom nodes on Cloud (no user installs): https://support.comfy.org/articles/6791455062-custom-nodes-on-cloud
- Data and storage on Cloud: https://support.comfy.org/articles/6342729551-data-storage-on-cloud
- Pricing (plans, credits, GPU table, import on Creator+): https://comfy.org/pricing
- Blog "Comfy Cloud: new features and pricing changes" (2025-11-25, updated after the 2026-01-23 price cut; 0.266
  credits/s): https://blog.comfy.org/p/comfy-cloud-new-features-and-pricing
- Terms of Service (effective 2026-05-13): https://comfy.org/terms-of-service
- Our code: cinestudio/backend.py, settings.py, runner.py (`relay_latent`, audio tail), server.py, figure_mask.py,
  song.py, workqueue.py; comfy-sdk 0.5.0 `comfy_sdk/client.py`; setup/models.json; setup/runtime.lock.json.
