# Making ACE-Step 1.5 base (and XL sft) faster (research, 2026-10-24)

Scope: web research, a read of our repo, our ComfyUI 0.42.1 install and our ComfyUI log. No downloads of models, no GPU
or ComfyUI jobs, no code changes. Numbers from other people are cited with source, date and version. Our own
estimates are marked **estimate**. Things nobody has confirmed are marked **unverified**. "[local]" = read in our
repo, our ComfyUI install (`D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI`, v0.42.1, commit 63be2f9, torch
2.12.1+cu130, comfy-kitchen 0.2.39) or our log `.cinestudio/comfyui.log`.

## 0. Summary

- **The restart costs more than the song.** Our last 73 s score took 73.7 s from start to end. 45.1 s of it was the
  ComfyUI restart (qwen_edit -> acestep). ComfyUI itself needed 27.8 s [local]. Earlier ACE restarts took 16-45 s, and
  the next job of another family pays one more restart (13-25 s). Avoiding the restart is the biggest win, with no
  quality risk.
- **Inside ComfyUI, the DiT dominates and is compute-bound.** 73 s song: LM 4.4 s, DiT 15.9 s (50 steps, 0.318 s per
  step), loads + VAE about 7.5 s [local]. A 30 s song runs at 0.164 s per step. The step time grows with the length
  (about 82% of the step scales with the sequence), so streaming weights is not the limit [local, estimate].
- **SageAttention does nothing for ACE-Step 1.5 in ComfyUI.** ComfyUI turned it off for this model on 2026-02-04
  (PR #12297). The DiT calls attention with `low_precision_attention=False`, so Sage falls back to PyTorch [local].
  The LM uses PyTorch attention too. So the Sage flag is NOT a reason to restart for ACE.
- **CFG for part of the steps is the cheapest sampling win.** Core `CFGOverride` (cfg 1 over a step range) skips the
  negative pass. With the `simple` scheduler its percent equals the step fraction [local]. cfg 1 on the last 35%:
  about -2.3 s of 15.9 s (-14%) **estimate**. ACE-Step v1 used guidance only in the middle 50% of the steps by default
  (`guidance_interval=0.5`). ACE-Step 1.5 keeps the knobs (`cfg_interval_start/end`) but defaults to the whole range.
- **int8 DiT:** Comfy-Org ships no fp8 or int8 ACE file. A community int8 file set exists (hrktxz, 2026-07-07, 0
  downloads, no card). Its header shows plain `int8_tensorwise` without the ConvRot flag, despite its name [header
  read]. Making our own int8 ConvRot file is the clean path. Expected DiT gain on the 3070: about 1.3-1.5x
  (**estimate**, from our Qwen result of 1.5x). No audio quality data exists (**unverified**).
- **Do not skip the LM.** It costs only 4.4 s of the 73 s song. ComfyUI's own tooltip says it raises quality.
- **Caching (EasyCache) and torch.compile:** no ACE-Step reports exist. torch.compile would recompile in every new
  engine, and every song starts a new engine today. Test them last, if at all.
- **Large GPU (estimate):** base 73 s song about 5-9 s of compute, XL sft + 4B LM about 8-15 s, without restart and
  cold loads. On a pod, the cold load from the network volume will cost more than the generation.
- **Turbo** (distilled, 8 steps, cfg 1) would cut the DiT from 15.9 s to about 1.5 s. The user refused distilled
  models. Report it only as an option (section 12).

## 1. What we run today [local]

- Graph: `workflows/acestep_base_api.json` + `workflows/acestep_base.bind.json` (`recipe.ACE_VARIANT = "base"`,
  `song.py` builds `ProfileWorkflow("acestep", "base")`). Nodes: `UNETLoader` -> `ModelSamplingAuraFlow` (shift 3) ->
  `KSampler` (50 steps, euler, simple, denoise 1), `TextEncodeAceStepAudio1.5` (generate_audio_codes true, LM cfg 2,
  temperature 0.85, top_p 0.9), negative = `ConditioningZeroOut`, `EmptyAceStep1.5LatentAudio`, `VAEDecodeAudio`.
  All are core nodes.
- cfg: `recipe.ace_cfg` = 6 for base, 7 when the file name has "sft" (XL sft).
- Files (`setup/models.json`): 8gb tier `acestep_v1.5_base.safetensors` (4.79 GB bf16) + `qwen_0.6b_ace15` (1.19 GB,
  text encoder) + `qwen_1.7b_ace15` (3.71 GB, the 5 Hz LM) + `ace_1.5_vae` (0.34 GB). Large tier:
  `acestep_v1.5_xl_sft_bf16` (9.97 GB) + `qwen_4b_ace15` (8.38 GB LM).
- Minor: the `_about` text in `acestep_base.bind.json` still describes the turbo template (8 steps, cfg 1). The graph
  itself is the 50-step base graph.

These settings match ComfyUI's official XL templates in our install (`comfyui_workflow_templates_json` 0.1.115):
`audio_ace_step1_5_xl_base.json` = 50 steps, cfg 6, euler, simple, shift 3; `audio_ace_step1_5_xl_sft.json` = 50 steps,
cfg 7, euler, simple, shift 3 (LM top_p 1.0 there; ours 0.9). There is no official 2B-base ComfyUI template; the
2B templates are turbo only (8 steps, cfg 1) [local].

## 2. Where the time goes (question 2)

### 2.1 Measured on the RTX 3070 [local: comfyui.log, songs/*/song.json, song logs]

| Job | Restart before | LM | DiT | Rest (loads + VAE) | ComfyUI total |
|---|---|---|---|---|---|
| 73 s base score (noir_score_v1, 2026-10-24) | 45.1 s | 365 tokens, 4.4 s (83 tok/s) | 50 steps, 15.9 s (3.14 it/s) | about 7.5 s | 27.79 s |
| 30 s base song (two runs) | not logged | 150 tokens, about 1.9 s (76-79 tok/s) | 50 steps, 8.2 s (6.1 it/s) | about 6.7 s | 16.93 / 17.51 s |
| 30 s turbo song (two runs, 2026-10-20) | 40.7 s (zimage -> acestep) | 150 tokens, about 1.9 s | 8 steps, under 1 s | about 6.7 s | 9.59 / 8.89 s |

- The 73 s job ran 73.7 s from `started` to `ended` in its song record: 45.1 s restart + 28.2 s ComfyUI + about
  0.4 s for ffmpeg and beat detection.
- All ACE restarts in our logs: 16.4, 37.1, 40.7, 45.1 s. Restarts of other families: mostly 13-25 s.
- The log shows dynamic VRAM loading: the text encoder (LM + embedder) "4672MB Staged", the DiT "4564MB Staged", the
  VAE "loaded completely; 321.70 MB". The LM ran through the Comfy model compiler ("graph breaks: 0").

### 2.2 Sizes and token counts

- VAE: 48 kHz stereo, 64 latent channels at 25 Hz (1920x compression). An FSQ tokenizer makes 5 Hz codes. The DiT
  patchifies to 12.5 Hz (tech report, arXiv 2602.00744v1). ComfyUI's empty latent uses `seconds * 48000 / 1920`
  frames [local].
- So a 73 s song = 1825 latent frames = about 913 DiT tokens per sample. With CFG the batch is 2, so 1826 tokens.
  The LM writes 5 codes per second = 365 tokens (matches the log).
- DiT base: about 2B (README, tech report). Config: 24 layers, hidden 2048, MLP 6144, 16 query heads / 8 KV heads of
  128, sliding window 128 on every other layer (HF `ACE-Step/acestep-v15-base` config.json). The decoder linears
  hold about 1.5B parameters **estimate**.
- DiT XL: 4B DiT, about 9 GB bf16 (README, 2026-04-02 release). HF lists 4,987,310,726 parameters for xl-sft. Config:
  hidden 2560, MLP 9728, 32 query heads / 8 KV heads, 32 layers. About 2.5x the base work per token **estimate**.
- LM: Qwen3-based, 0.6B / 1.7B / 4B. README rates composition "Weak / Medium / Strong". Text encoder: Qwen3
  0.6B embeddings (tech report).

### 2.3 Compute-bound or load-bound?

- **Compute-bound** (estimate from our two lengths). Step time 0.164 s at 375 tokens per sample and 0.318 s at 913.
  A line through both: 0.057 s fixed + 0.000286 s per token. At 73 s, about 82% of the step grows with the length.
  If streaming 4.6 GB of weights per step were the limit, the step time would barely change with the length.
- Work per step at 73 s: about 2 x 1.5B x 1826 tokens + cross-attention + attention = about 5.3 TFLOP
  **estimate**. At 0.318 s that is about 17 TFLOPS, about 40% of the 3070's roughly 40 TFLOPS bf16 peak with fp32
  accumulation (NVIDIA Ampere whitepaper, as cited in docs/qwen_speed_research.md; the 40 TFLOPS figure is
  **unverified** here). Short sequences use the GPU less well than Qwen's 12k tokens (80% of peak there).
- Attention is small: about 3% of the FLOPs at 913 tokens **estimate**.
- The LM is decode-bound: 83 tokens/s for 1.7B in fp16 (log: text encoder dtype float16). At 448 GB/s and about
  3.4 GB of weights per token, the bandwidth limit is about 130 tokens/s **estimate**.
- Loads and VAE: about 6-7 s, mostly fixed (30 s song: 6.7 s, 73 s song: 7.5 s). So VAE decode is about 1-2 s and
  loading after a fresh start about 5-6 s **estimate**.

**Order of the parts for a 73 s song:** restart (16-45 s) > DiT (15.9 s) > loads (about 5-6 s) > LM (4.4 s) >
VAE (about 1-2 s).

## 3. Official recommendations (question 1)

- **Steps:** base and sft 50 steps with CFG; turbo 8 steps without CFG (README model table). The inference guide:
  "Base: 1-200 (recommended 32-64)"; for best quality "inference_steps=64 or higher" (docs/en/INFERENCE.md, main,
  read 2026-10-24). The minimum that keeps quality is not stated anywhere (**unverified**).
- **CFG:** default `guidance_scale` 7.0, typical 5-9; best quality 7-9 (INFERENCE.md). ComfyUI templates: 6 (XL
  base), 7 (XL sft) [local].
- **Guidance type:** the official code does NOT use plain CFG. For base/sft it calls `apg_forward` (APG) by default,
  or `adg_forward` with `use_adg=True` ("Adaptive Dual Guidance (base model only)") (`acestep/models/base/
  modeling_acestep_v15_base.py`, main, read 2026-10-24). ComfyUI's templates use plain CFG. ComfyUI has an `APG` core
  node [local]. This is a quality question, not a speed one: APG costs the same as CFG.
- **Shift:** default 1.0; "Recommended 3.0 for turbo"; for best base quality the guide also suggests shift 3.0
  (INFERENCE.md). The Gradio UI defaults to 3.0 (GRADIO_GUIDE.md). ComfyUI templates: 3 [local].
- **Sampler:** `infer_method` "ode" (Euler, deterministic) default; "sde" optional (INFERENCE.md). ComfyUI: euler,
  simple [local].
- **CFG interval:** ACE-Step 1.5 has `cfg_interval_start` 0.0 and `cfg_interval_end` 1.0 (whole range) by default
  (INFERENCE.md, GRADIO_GUIDE.md). No recommended narrower values were found (**unverified**). In the official code
  the negative pass still runs outside the interval: it computes both halves and keeps only the conditional one
  (modeling_acestep_v15_base.py). So the official app saves no time with it.
- **ACE-Step v1** (for comparison): `guidance_interval=0.5` (CFG only from 25% to 75% of the steps),
  `guidance_interval_decay=0.0`, `min_guidance_scale=3.0`, `guidance_scale=15`, `cfg_type="apg"`, 60 steps. Outside
  the interval v1 ran ONE forward pass (conditional only) (`acestep/pipeline_ace_step.py`, ACE-Step v1 main, read
  2026-10-24). The idea matches "Applying guidance in a limited interval improves sample and distribution quality"
  (Kynkäänniemi et al., NeurIPS 2024, arXiv 2404.07724): guidance at high noise hurts, at low noise it adds little.
- DCW (a wavelet sampler correction): "adds negligible compute", off by default for non-turbo models (docs/en/DCW.md).
  Not a speed item.

## 4. CFG for part of the steps (our CFGOverride equivalent)

- ComfyUI 0.42.1 has core `CFGOverride` (model, cfg, start_percent, end_percent) [local,
  `comfy_extras/nodes_custom_sampler.py`]. At cfg 1.0 the sampler's cfg1 optimization skips the negative pass
  (`samplers.py`, `math.isclose(cond_scale, 1.0)`) [local].
- With ModelSamplingAuraFlow and the `simple` scheduler, percent maps linearly to steps: `percent_to_sigma(p) =
  shift(1 - p)` and `simple` takes evenly spaced entries of the same shifted table [local, `model_sampling.py`,
  `samplers.py`]. So start_percent 0.65 = step 33 of 50.
- Cost model (estimate, from 2.3): a cfg-1 step = 0.057 + 0.000143 x 913 = about 0.19 s instead of 0.318 s.
  - cfg 1 on the last 35% (17-18 steps): about -2.3 s, DiT 15.9 -> 13.6 s (-14%).
  - v1-style (cfg only 25%-75%, cfg 1 on 25 steps): about -3.2 s (-20%). Riskier: the early steps set the structure.
- Quality: no ACE-1.5 test exists (**unverified**). Plain CFG at 6 differs from v1's APG at 15, so v1's values are a
  hint, not a rule.

## 5. Quantized files (question 3)

- **Comfy-Org repo** (`Comfy-Org/ace_step_1.5_ComfyUI_files`, now `Comfy-Org/ACE-Step-1.5`, commit 6707deb,
  2026-09-23): only bf16 files (base, turbo, XL base/sft/turbo, 3 LMs, VAE, turbo AIO). No fp8, no int8, no 2B sft
  file (HF API, read 2026-10-24).
- **hrktxz/ACE_Step_1.5_ComfyUI_int8_convrot** (2026-07-07, no licence tag, 0 downloads, README only says "Converted
  using convert_to_quant"): base 2.40 GB, XL base / XL sft 4.99 GB, also turbo files. We read the base file's header
  (194 KB, not the weights): 390 layers in `int8_tensorwise`, including the lyric/timbre encoders and the tokenizer.
  Each layer's config is 29 bytes = `{"format": "int8_tensorwise"}`. ComfyUI turns on ConvRot only when a layer config
  has `"convrot": true` (`comfy/ops.py` [local]). So this looks like **plain int8, not ConvRot** (check before use).
- **Make our own:** silveroxides `convert_to_quant` (`--int8 --convrot --comfy_quant`, as used for the Qwen int8
  ConvRot files, see docs/qwen_speed_research.md). GitHub shows no licence file (API, read 2026-10-24). The weights
  stay MIT. Keep only the DiT decoder in int8 at first; the official app also quantizes only `decoder.*` linears
  and excludes the tokenizer and detokenizer (`init_service_loader.py`, main).
- **Native on our card:** ComfyUI 0.42.1 lists `int8_tensorwise` as a native op on the 3070; fp8 is emulated on Ampere
  (our log, quoted in docs/qwen_speed_research.md). comfyanonymous: int8 tensor ops are "4x faster" than fp16/bf16
  on 30/40/50-series, "only 2x" on A100 (ComfyUI issue #14824, 2026-07-08, via docs/qwen_speed_research.md).
- **Our own int8 evidence:** Qwen-Image-Edit int8 ConvRot 11.4 s/it vs 17.4 s/it for emulated fp8 (1.5x)
  [local, 2026-10-24, docs/large_gpu_speed_research.md].
- **Expected for ACE on the 3070 (estimate):** about 80% of a step is linears; at 913 tokens the activation quantizing
  costs relatively more than at Qwen's 12k tokens. DiT 15.9 s -> about 11-12 s (1.3-1.5x).
- **Official app:** torchao `int8_weight_only` (also `fp8_weight_only`, `w8a8_dynamic`), on by default below 20 GB
  VRAM, "required" with torch.compile (GPU_COMPATIBILITY.md, `init_service_loader.py`). It is meant for VRAM, and its
  speed and quality effects are not stated (**unverified**).
- **GGUF** (CC-TM/ACE-Step-1.5-GGUF, Hagwell/..., MIT, 2026-07): Q4_K_M-Q8_0 and BF16 for every DiT and LM. They are
  for acestep.cpp (GGML), not for ComfyUI-GGUF (README). Not usable in our graph (**unverified** whether ComfyUI-GGUF
  could load them).
- **Side note (quality, same speed):** the official table rates 2B base "Medium" quality and 2B sft "High"; XL sft
  "Very High" (README). An int8 XL sft (4.99 GB) would fit the 3070's VRAM. It would cost about 2.5x the base DiT
  time before the int8 gain **estimate**.

## 6. Attention (question 4)

- ComfyUI disabled SageAttention for ACE-Step 1.5 on 2026-02-04 (PR #12297, comfyanonymous, "Disable sage attention
  on ace step 1.5", no reason given). The DiT passes `low_precision_attention=False`; `attention_sage` then falls back
  to `attention_pytorch` [local, `comfy/ldm/ace/ace_step15.py`, `comfy/ldm/modules/attention.py`]. No public
  artifact report was found (**unverified** why).
- The LM uses `optimized_attention_for_device(..., small_input=True)`, which picks PyTorch attention [local,
  `comfy/text_encoders/llama.py`]. The VAE has no attention that Sage touches **estimate**.
- So `--use-sage-attention` changes nothing for ACE. Attention is about 3% of the DiT FLOPs anyway.
- FlashAttention: the official app uses `flash_attention_2` when installed (`init_service_loader.py`), so exact flash
  attention is fine for this model. In ComfyUI the sliding-window layers pass a dense bias mask, which flash kernels
  do not take **estimate**. Gain: near zero.
- Warning: `--use-ck-attention` (Comfy Kitchen int8 attention) only falls back for fp32 inputs, not for bf16 [local].
  It would quantize ACE attention, which ComfyUI chose to avoid for Sage. Do not use it for ACE.

## 7. Caching and step skipping (question 5)

- ComfyUI has core `EasyCache` (reuse_threshold 0.2, start 0.15, end 0.95) and LazyCache [local,
  `comfy_extras/nodes_easycache.py`]. They work on the model output tensor, so they should run on ACE's
  [batch, 64, frames] latent **estimate**.
- No report of EasyCache, TeaCache or MagCache on ACE-Step or any music model was found. The caching papers test
  video and image models only (MagCache arXiv 2506.09045; EasyCache arXiv 2507.02860; DiCache arXiv 2508.17356).
- Risk: reusing a step can smear transients (drum hits) **unverified**. Test last, with a spectrogram.

## 8. The LM "thinking" part (question 6)

- Our graph uses it: `generate_audio_codes` true. ComfyUI's tooltip: "Enable the LLM that generates audio codes. This
  can be slow but will increase the quality of the generated audio" [local, `comfy_extras/nodes_ace.py`]. ComfyUI
  made ACE 1.5 work without the LLM on 2026-02-05 (commit 6555dc65) [local].
- Cost: 4.4 s of 27.8 s for 73 s (16%); 1.9 s for 30 s [local]. The official benchmark example shows think mode
  adding 2.45 s to a 30 s turbo run (BENCHMARK.md, example output, GPU not stated).
- Skipping it: -4.4 s, with a quality cost the docs warn about. The LM plans structure ("Composer Agent", tech report).
  The 0.6B LM is rated "Weak" for composition (README). **Keep the 1.7B LM** (4B on the large tier).
- Faster LM without a quality loss: int8 LM weights would halve the bytes per token (decode-bound), maybe 1.5x
  **estimate**. No ACE LM int8 file for ComfyUI was found, and whether ComfyUI's LM sampler runs int8 is
  **unverified**. Gain at most about 1.5-2 s. Low priority.

## 9. The engine restart (question 7)

- What forces it [local]: `server.ensure_family(wf.family_key())` keeps the engine only if the recorded family list
  equals the wanted one. `family_key()` = `"acestep@" + sha1(loader file names)[:10]`, so any other family, or the 8gb
  vs large files, means a restart (`cinestudio/workflow.py`, `cinestudio/server.py`). The reason is RAM:
  "one model family per ComfyUI process" after LTX + Wan crashed a 32 GB machine, and `/free` was not safe
  (server.py docstring).
- SageAttention is not the reason: `comfy.speed_args` adds `--use-sage-attention` only when all next families are in
  `SAGE_FAMILIES = ("ltx25",)`. ACE ignores Sage anyway (section 6). So ACE could run in an LTX engine with Sage on
  and give the same audio **estimate** (the attention path is the same code).
- RAM math on 32 GB (8gb tier files, **estimate**): ACE = 4.79 + 1.19 + 3.71 + 0.34 = about 10 GB.
  - Qwen edit (int8 20.5 GB + fp8 encoder 9.4 GB) + ACE = about 40 GB: does not fit; relies on eviction.
  - LTX distilled int8 + Gemma int8 (about 37 GB) already exceed RAM: no room.
  - Z-Image (about 20 GB with its encoder) + ACE = about 30 GB: tight.
  ComfyUI 0.42 has a RAM-pressure cache and disk-backed dynamic loading (`--fast-disk`, on in our args). Whether it
  now evicts safely enough is **unverified**.
- Options (no code written):
  1. **Group song jobs** in the queue so one restart serves several songs.
  2. **Join instead of restart** when free RAM is above about ACE's 10 GB + a margin: record the families as a set
     (e.g. `{zimage@..., acestep@...}`) and keep the engine when the wanted family is in it. On the large tier
     (120 GB+ RAM) this is safe for every family.
  3. If a restart stays: an ACE-only engine can start with `--disable-all-custom-nodes` (our ACE graph uses only core
     nodes). The custom-node imports took about 4.7 s in our log. Small gain; it changes the flags per family.
- Gain: -16 to -45 s per song, and often -13 to -25 s for the next job of another family. No quality risk.

## 10. torch.compile (question 8)

- Core `TorchCompileModel` exists; triton-windows 3.7.1 is installed [local]. ComfyUI 0.42 already runs its own
  "Comfy model compiler" (CUDA-graph malloc) by default; the LM showed "graph breaks: 0" [local,
  `comfy/model_prefetch.py`].
- The official app turns on `torch.compile` for the DiT and VAE on all tiers (GPU_COMPATIBILITY.md,
  `init_service_loader_components.py`). Its speed effect is not stated (**unverified**).
- For us: every song starts a fresh engine today, so the compile cost comes back each time, and a new duration may
  recompile. Gain on a 16 s DiT is likely smaller than the compile time **estimate**. Only worth a test on a warm,
  long-lived engine (a pod, or after option 9.2).

## 11. Large GPU (question 9)

Rates: H100 SXM about 989 TFLOPS bf16 dense; RTX PRO 6000 about 500 (dense figure unverified); int8 = 4x bf16 on
the PRO 6000, 2x on H100 (docs/large_gpu_speed_research.md, NVIDIA datasheets). No ACE-Step 1.5 timing on H100,
PRO 6000 or 4090 was found. The only official claims: "under 2 seconds per full song on an A100 and under 10 seconds
on an RTX 3090" (turbo, tech report and README).

Estimate for a 73 s song (no restart, models already in VRAM):

| Part | 3070 measured | H100 / PRO 6000, base 2B + 1.7B LM | H100 / PRO 6000, XL sft + 4B LM |
|---|---|---|---|
| DiT 50 steps, CFG | 15.9 s | 1.5-3.5 s (small batch, overhead-bound) | 3-7 s |
| LM | 4.4 s | 2-4 s (decode, Python loop) | 3-6 s |
| VAE | about 1-2 s | under 1 s | under 1 s |
| Total compute | about 22 s | about 5-9 s | about 8-15 s |

- The first load from a network volume (10-20 GB for ACE) will cost more than this **estimate**. Keep the engine
  warm and do not restart for ACE on a pod (RAM 188 GB on a PRO 6000 pod, per docs/large_gpu_speed_research.md).
- int8 on a large GPU: little gain at these short sequences (the step is mostly overhead) **estimate**. Use bf16 there.

## 12. Turbo (option only; the user refused distilled models)

- 8 steps, shift 3, no CFG (README, tech report: DMD2 distillation, "50 to 8 steps"). On our 3070: 8 steps took
  under 1 s for 30 s [local]; a 73 s turbo song would take about 12 s in ComfyUI instead of 28 s **estimate**.
- Quality cost: the official table rates turbo "Very High" quality but "Medium" diversity and harder to fine-tune
  (README). That is the vendor's own rating; no independent A/B was found. The cost the user cares about (fidelity to
  the full model) is **unverified**.
- Community merges exist (e.g. "sftturbo50" GGUFs, XL base/turbo merges). They are partly distilled too.

## 13. Ranked table

Gains are for one 73 s song. 3070 baseline: 73.7 s job (45.1 s restart + 27.8 s ComfyUI).

| # | Technique | Gain on the 3070 | Gain on a large GPU | Quality risk | Effort | Licence |
|---|---|---|---|---|---|---|
| 1 | No restart for ACE (group song jobs; join the running engine when RAM allows) | -16 to -45 s per song, plus often -13 to -25 s on the next job | avoids cold loads from the volume (larger than the compute) | none (same graph) | medium (server.py policy, RAM check) | n/a |
| 2 | `CFGOverride` cfg 1 on the last 30-35% of steps | about -2 s (DiT -14%) | similar share, smaller in seconds | low-medium (A/B) | tiny (one core node) | ComfyUI core (GPL-3, already used) |
| 3 | int8 ConvRot DiT (our own file) | about -4 to -5 s (DiT 1.3-1.5x) | small (overhead-bound) | low-medium, no audio data | low-medium (convert, pin hash, A/B) | MIT weights; converter has no licence file |
| 4 | Fewer steps (40, then 32; official range 32-64) | -3 s (40) / -6 s (32) | same share | medium | setting | n/a |
| 5 | v1-style CFG interval (cfg only 25-75%) instead of #2 | about -3 s | same share | medium | tiny | core |
| 6 | `--disable-all-custom-nodes` for an ACE-only restart | about -5 s per restart | small | none | low | n/a |
| 7 | EasyCache | maybe -4 to -8 s **unverified** | similar | high (untested on audio) | tiny | core |
| 8 | torch.compile | likely a net loss while we restart per song | maybe 10-20% on a warm engine **unverified** | low | low | core / PyTorch |
| 9 | int8 LM | about -1.5 to -2 s **unverified** | small | low-medium | high (no file, support unclear) | MIT |
| 10 | Skip the LM / use the 0.6B LM | -4.4 s / about -2.5 s | -2 to -4 s | high (docs say quality drops) | setting | n/a |
| 11 | SageAttention / FlashAttention | none (ComfyUI forces PyTorch attention for ACE 1.5) | none | n/a | n/a | n/a |
| 12 | Turbo (distilled) | about -16 s in ComfyUI | -1 to -5 s | refused by the user (less diversity, vendor-rated quality) | setting + file | MIT |

Avoid: `--use-ck-attention` for ACE (int8 attention on bf16, section 6).

## 14. Test order (one change at a time, same seed, timed)

Common rules: the 73 s noir score (seed 202, D minor, 104 bpm) plus two more seeds. Read the times from
`comfyui.log` (LM tok/s, DiT it/s, "Prompt executed") and the song record. Euler + simple is deterministic, so a
change that should not change the result (int8, attention) can be checked with a difference signal and a log-mel
distance against the baseline. Changes that alter sampling (CFG range, steps) need a blind listen: the user hears A/B
pairs without labels (3 seeds). Also compare spectrograms (transients, energy above 10 kHz, silence at the end) and
`song.beats` (tempo stability).

1. **Baseline.** Run the 73 s score twice in a row without other jobs between. The first run shows the restart cost,
   the second the warm-engine time (no code needed). This measures what option 9.2 would save.
2. **Restart policy** (#1): group songs or join the engine. Watch RAM (keep about 4 GB free) and ComfyUI stability.
3. **CFGOverride** cfg 1.0, start 0.65, end 1.0 (#2). Same seeds. Spectrogram + blind listen. If it passes, try 0.60.
4. **int8 ConvRot DiT** (#3), made by us. Check the "Using mixed precision operations" log line. Time, difference
   signal, blind listen.
5. **Steps 40** (#4), only after 3 and 4 pass. Then 32 if 40 is blind-equal.
6. **EasyCache** default values (#7), last; reject at the first audible smear.
7. On a pod: the same order; add torch.compile (#8) only on a warm engine.

Write each round to `.cinestudio/gauntlet/acestep_speed/RESULTS.md`.

## Sources

- ACE-Step 1.5 repo README (main; newest entry 2026-04-02, XL release): https://github.com/ace-step/ACE-Step-1.5
- INFERENCE.md (main, read 2026-10-24): https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md
- GRADIO_GUIDE.md: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GRADIO_GUIDE.md
- GPU_COMPATIBILITY.md: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GPU_COMPATIBILITY.md
- BENCHMARK.md (example output only): https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/BENCHMARK.md
- DCW.md: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/DCW.md
- Base model code (APG/ADG, cfg interval): https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/models/base/modeling_acestep_v15_base.py
- Official quantization/attention loader: https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/core/generation/handler/init_service_loader.py
- Tech report, ACE-Step 1.5 (arXiv 2602.00744v1): https://arxiv.org/html/2602.00744v1
- HF configs: https://huggingface.co/ACE-Step/acestep-v15-base (commit e432212, 2026-02-06),
  https://huggingface.co/ACE-Step/acestep-v15-xl-sft (commit d06de46, 2026-04-07)
- Comfy-Org files (commit 6707deb, 2026-09-23): https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files
- ComfyUI ACE-Step 1.5 tutorial (turbo only, undated): https://docs.comfy.org/tutorials/audio/ace-step/ace-step-v1-5
- ComfyUI PR #12297 (2026-02-04): https://github.com/Comfy-Org/ComfyUI/pull/12297
- Community int8 files (2026-07-07): https://huggingface.co/hrktxz/ACE_Step_1.5_ComfyUI_int8_convrot
- convert_to_quant: https://github.com/silveroxides/convert_to_quant
- GGUF for acestep.cpp (MIT, 2026-07-13): https://huggingface.co/CC-TM/ACE-Step-1.5-GGUF
- ACE-Step v1 pipeline (guidance interval defaults): https://github.com/ace-step/ACE-Step/blob/main/acestep/pipeline_ace_step.py
- ACE-Step v1 README (RTF table): https://github.com/ace-step/ACE-Step
- Guidance in a limited interval (arXiv 2404.07724): https://arxiv.org/abs/2404.07724
- MagCache (arXiv 2506.09045), EasyCache (arXiv 2507.02860), DiCache (arXiv 2508.17356)
- Our reports: docs/qwen_speed_research.md, docs/large_gpu_speed_research.md (GPU rates, int8 results, ComfyUI
  issue #14824)
- Local: ComfyUI v0.42.1 source (`comfy/ldm/ace/ace_step15.py`, `comfy/ldm/modules/attention.py`,
  `comfy_extras/nodes_ace.py`, `nodes_custom_sampler.py`, `nodes_easycache.py`, `comfy/ops.py`,
  `comfy/model_sampling.py`), templates package 0.1.115, `.cinestudio/comfyui.log`, `.cinestudio/songs/*/song.json`,
  `.cinestudio/song_horse.log`, `.cinestudio/songs/funk_lock.log`.
