# Speed on large GPUs: LTX-2.5 and Qwen-Image-Edit-2511 (research, 2026-10-24)

Scope: web research, a read of our repo, our ComfyUI 0.42.1 install (`D:\Comfy-Desktop\...\ComfyUI`, comfy-kitchen
0.2.39, comfy-aimdo 0.5.6, torch 2.12.1+cu130) and our ComfyUI log (`.cinestudio/comfyui.log`). No downloads, no GPU
runs, no code changes. This report extends docs/qwen_speed_research.md (Qwen on the 3070); it does not repeat it.

Marks: **[local]** = read in our code, install or log. **[est.]** = our own estimate. **unverified** = nobody has
confirmed it. Every outside number has its source in brackets; the full list is at the end.

## 0. Summary

- **Most of our 8 GB choices do not slow a big GPU.** ComfyUI's dynamic VRAM keeps a model resident when it fits
  [local, code; Comfy blog 2026-03-25]. Stage 1 and stage 2 already run in one prompt without unloading. The real
  8 GB leftovers are the small VAE decode tiles and the engine restart at every model-family change.
- **Do NOT add `--highvram` or `--gpu-only` on a big GPU.** In ComfyUI 0.42.1 both flags turn dynamic VRAM OFF
  (`enables_dynamic_vram()` in `comfy/cli_args.py`) [local]. That also turns off the "Comfy model compiler" (the aimdo
  memory compiler, PR #15861). The default start flags are already right.
- **The biggest structural cost on RunPod is cold model loads.** Every family change (Qwen look image -> LTX clips)
  restarts ComfyUI [local, server.ensure_family]. A network volume reads at 200-400 MB/s (RunPod docs). The LTX int8
  set (~40 GB) then takes about 2-3 min to load, the bf16 set (~70 GB) about 3-6 min [est.]. On a big pod, skip the
  restart and keep the models on the local container disk or in the RAM page cache.
- **Precision:** Lightricks publishes bf16, int8 ConvRot (ComfyUI only) and NVFP4 (distilled) for LTX-2.5; no fp8
  (HF Lightricks/LTX-2.5). int8 ConvRot is the closest 8-bit format to bf16 (unsloth: LPIPS 0.265 vs fp8 0.366). On
  GeForce and RTX PRO cards int8 math runs at 4x the bf16 rate; on L40S and H100 at 2x (NVIDIA datasheets). So our
  `24gb` tier, which switches the dev model and Qwen to bf16, is the SLOW choice on every large GPU. Keep bf16 only if
  a blind A/B shows a visible gain.
- **Avoid int8 on A100 and B300.** int8 ConvRot ran at 33-50% of fp8 speed on an A100 (ComfyUI #14824). The B300 has
  almost no int8 tensor throughput (~190 TOPS dense per GPU, NVIDIA HGX page).
- **Attention is about half the work in our stage 2** [est.: about 44k tokens with the layout guide]. SageAttention
  2.2 has kernels for sm89 (4090, L40S), sm90 (H100) and sm120 (5090, RTX PRO 6000), but none for sm100 (B200): there
  ComfyUI falls back to PyTorch attention silently (thu-ml code). SageAttention 2.2 has no Linux wheel for torch 2.12:
  build it from source. Core ComfyUI now also has its own Apache-2.0 int8 attention (`--use-ck-attention`, v0.32.0).
- **Install script fix needed:** torch 2.12 has no cu128 build (download.pytorch.org index, read 2026-10-24). Our
  installer's default `--cuda cu128` would install torch 2.11, not the tested 2.12. The comfy-kitchen CUDA backend
  (native fp8/NVFP4 kernels) needs a CUDA 13 torch build. Use `--cuda cu130` on pods.
- **Best speed per dollar for us: RTX PRO 6000 96 GB** ($2.49/h Secure, 188 GB RAM). It led RunPod's own measured
  video benchmark per dollar (RunPod, 2026-10-12). H100 SXM is faster per clip but costs 60% more per hour. RTX
  4090/5090 pods have only 35-41 GB RAM, too little for our bf16 models. Network volumes force Secure Cloud prices.
- **Caching (EasyCache) makes sense only for the step-heavy runs** (Qwen 40 steps, LTX dev 30 steps), not for the
  8+3-step distilled clips. NVFP4, SageAttention3 (FP4) and the conv VAE are fast but cost quality: options only.

## A. Where the time goes today (RTX 3070, our log)

Measured in `.cinestudio/comfyui.log` [local], ComfyUI 0.42.1, SageAttention 1.0.6 for LTX:

| Job | Steps | Median s/step | Prompt total |
|---|---|---|---|
| LTX-2.5 distilled int8, layouting, stage 1, 768x448x121 | 8 | 11.2 (234 runs) | preview clip 121-160 s |
| same, stage 2, 1536x896 | 3 | 41.6 (128 runs) | final clip about 285-360 s; outliers 8-13 min (memory pressure) |
| LTX-2.5 dev int8, stage 1 (MultimodalGuider) | 28-30 | 33-40 (few runs) | about 17-20 min per stage 1 |
| Qwen-Image-Edit-2511 fp8mixed (emulated), 40 steps | 40 | 17.2-18.8 | 11:38-12:29 |
| Qwen-Image-Edit-2511 int8 ConvRot (run of 2026-10-24) | 40 | 11.4 | about 1.5x faster; quality not yet checked |

- In a final distilled clip about 60-90 s are outside sampling: the Gemma encode when the prompt changes (prompts with
  a text encode take about 15-25 s longer), the guide VAE encodes, and the tiled decode [local, log].
- Token counts in our layout graph [est., read from `workflows/ltx25_layouting_api.json`; LTX video tokens =
  ((F-1)/8+1)·(H/32)·(W/32), docs.ltx.io]:
  - Stage 1: video 5,376 + layout guide 5,376 (full size: the LoRA has no `reference_downscale_factor`, our log
    warns "using 1.0") + the Ingredients sheet repeated on 121 frames 5,376 + look image 336 = **about 16.5k tokens**.
  - Stage 2: video 21,504 + layout guide 21,504 + look 672 = **about 43.7k tokens**.
  - At 43.7k tokens, attention and the linear layers cost about the same [est.: 4·L²·4096·48 ≈ 1.5 PFLOP vs
    2·~19B·L ≈ 1.7 PFLOP per pass]. So attention kernels matter a lot in stage 2.
- LTX keeps PyTorch attention for its masked guide attention even when SageAttention is on
  (`low_precision_attention=False`, "sageattn mask support is unreliable", `comfy/ldm/lightricks/model.py`). So
  SageAttention speeds up only part of our guided runs.

## B. Question 1 — our 8 GB choices on a big GPU

| 8 GB choice (where) | Does it slow a big GPU? | Large-GPU setting |
|---|---|---|
| `VAEDecodeTiled` 512 / 64 / temporal 64 / 16 (`make_25_graphs.py`, all ltx25 graphs) | Yes. Lightricks: "fewer, larger tiles run faster" (2.5 example README). Tiled decode measured 1.6x slower than untiled on an LTX graph (diffuseR PR #38, RTX 5060 Ti). Lightricks: the diffusion VAE handles "roughly 500x more tokens" than the conv VAE (HF LTX-2.5 discussion #15, 2026-08-14). | Lightricks' table (in every 2.5 example graph): `1024/256/128/48`, `1536/384/192/48`, or `768/192` spatial only. Use 1024/256/128/48 on 48 GB, 1536/384/192/48 on 80-96 GB. Keep the 8 GB values in the 8gb tier. |
| Weights streamed from RAM (dynamic VRAM) | No, when the model fits. A weight that was loaded once stays in VRAM until memory pressure (Comfy blog "Dynamic VRAM", 2026-03-25; `free_memory(for_dynamic=True)` does not unload dynamic models [local]). | Keep the defaults. Never `--highvram`/`--gpu-only` (they turn dynamic VRAM off [local]). |
| Model unload between stage 1 and stage 2 | No. Both stages are one ComfyUI prompt with the same patched model. Lightricks only notes that skipping memory cleanup between stages saves time when VRAM allows (ltx-pipelines docs). | Nothing to change. |
| Gemma 4 12B load per clip | Only when it does not fit next to the transformer. Each clip has its own text, so it is encoded again every clip. The compute is tiny (12B params x ~1k tokens ≈ 24 TFLOP: well under 1 s on an H100 [est.]). The cost is the reload. | 80-96 GB: LTX bf16 42 GB + Gemma bf16 26 GB stay resident together. 48 GB: they evict each other every clip; use the int8 Gemma (15.4 GB) there. |
| Engine restart at every family change (`server.ensure_family`, `comfy.restart_own`) | **Yes, the largest leftover.** Needed on 8 GB/32 GB RAM and for the per-family SageAttention start flag. On a pod each restart reloads tens of GB from a 200-400 MB/s network volume (RunPod docs). | Large tier: no restart between Qwen and LTX when RAM ≥ 128 GB; choose attention per model instead of by start flag (section D). |
| 768x448 generation, x2 to 1536x896 | Not a speed choice. It is the Ingredients IC-LoRA's only training size (its card: "768x448, 121 frames"). The Layout-To-Render card lists 960x544 -> 1920x1088 as "known good". | Keep 768x448. 960x544 is a quality test, about 1.5x the tokens per frame and about 2.3x the attention [est.]. |
| Preview = stage 1 only (`runner.to_stage1_only`) | No. | Keep (user rule: tests in preview). |
| Pose guide only in stage 1; 1024x576 out of memory | Memory limits only. The pose guide gave no gain (AGENTS.md 7.4). | Nothing to gain. |
| Comfy Desktop flags `--vram-headroom 1 --fast-disk` | Windows only (our saved Desktop command). On Linux the command is `--enable-manager` (`comfy.detect`). `--fast-disk` forces disk-backed loading: bad on a network volume. | Do not copy them to the pod. |

## C. Question 2 — precision of the LTX-2.5 transformer

- **Official files** (HF Lightricks/LTX-2.5, collection updated Sep 9 2026): distilled in bf16, comfy-int8-convrot
  ("ComfyUI only") and NVFP4; dev in bf16 and comfy-int8-convrot; Gemma in bf16 and int8-convrot. **No fp8, no GGUF.**
- **Lightricks' recommendations** (docs.ltx.io system-requirements, comfy-ui, workflow-asset-compatibility, read
  2026-10-24): minimum "32GB+ VRAM"; recommended "A100 (80GB) or H100". Their "lower-VRAM" set is int8 transformer +
  int8 Gemma + conv VAE. NVFP4 is "for FP4-capable (NVIDIA Blackwell) GPUs". No per-GPU timing tables are published
  (the optimization doc gives "no absolute timings or VRAM figures").
- **Quality** (unsloth/LTX-2.5-FP8 card, undated, their own quants, distilled, same seed, vs bf16): int8 ConvRot LPIPS
  0.265 / PSNR 20.2 dB; fp8 per-row 0.366 / 17.6 dB; plain int8 0.385 / 17.4 dB. ConvRot is the best 8-bit format.
  These are seed-level differences in a video; a blind A/B decides whether they show.
- **Tensor rates (dense)** (NVIDIA datasheets/whitepapers, read 2026-10-24):

  | GPU | bf16 (fp32 acc.) | fp8 | int8 | Note |
  |---|---|---|---|---|
  | RTX 4090 | 165 | 330 (fp32 acc.) | 661 | int8 = 4x bf16 |
  | RTX 5090 | 210 | 419 (fp32 acc.) | 838 | int8 = 4x bf16; FP4 1676 |
  | RTX PRO 6000 | ~500 | ~1000 | unverified | dense figures unverified on NVIDIA pages |
  | L40S | 362 | 733 | 733 | int8 = fp8 = 2x bf16 |
  | A100 80 GB | 312 | none | 624 | int8 slow in ComfyUI (#14824) |
  | H100 SXM | 990 | 1979 | 1979 | int8 = fp8 = 2x bf16 |
  | B200 | 2250 | 4500 | 4500 (derived, unverified) | FP4 9000 |
  | B300 | 2250 | 4500 | ~190 | int8 nearly gone |

- **What this means for us** [est.]: the linear layers are about half of a stage-2 step and more of a stage-1 step.
  int8 instead of bf16 should give about **1.3-1.8x per clip on 4090/5090/PRO 6000** and **1.2-1.5x on L40S/H100**.
  There is no published Hopper benchmark of int8 ConvRot vs bf16 for LTX (unverified). On an RTX A6000 (sm86
  workstation, full-rate bf16) LTX-2.3 bf16, fp8 and int8 all ran at about 22 s/it (ComfyUI #14824).
- **NVFP4 (Blackwell only):** on an RTX 5090, distilled NVFP4 sampling was only 5.3% faster than int8 (14.31 s vs
  13.55 s; end to end 28.67 s vs 28.82 s) (ai-muninn, 2026-09-11). NVIDIA claims 2.5x over bf16 for LTX-2 on RTX 50
  (NVIDIA blog, 2026-03-10). It needs a cu130 torch, else "up to 2x slower than fp8" (Comfy blog, 2026-01-09). One
  shape error in an official workflow (TrueNorthAI, 2026-08-12). **Verdict: not worth its quality risk for us.**
- **Q8 kernels** (Lightricks/LTX-Video-Q8-Kernels): legacy, for LTXV 13B; no LTX-2.x support. Not relevant.

## D. Question 3 — attention

- **SageAttention** (thu-ml/SageAttention, main branch, read 2026-10-24): PyPI has only 1.0.6 (2024). The 2.2.0
  install (`pip install sageattention==2.2.0 --no-build-isolation`) builds from source. Apache-2.0 repo licence.
  Auto kernel choice per GPU (core.py):

  | GPU | Kernel | P·V accumulation |
  |---|---|---|
  | A100 (sm80) | qk_int8_pv_fp16_cuda | fp32 |
  | 4090, L40S (sm89) | qk_int8_pv_fp8_cuda | fp32+fp16 |
  | H100 (sm90) | qk_int8_pv_fp8_cuda_sm90 | fp32+fp32 |
  | 5090, RTX PRO 6000 (sm120) | qk_int8_pv_fp8_cuda | fp32+fp16 |
  | B200 (sm100) | none: error, ComfyUI falls back to PyTorch | - |

- Our installer installs **1.0.6** (Triton only, fp16 P·V). It works for LTX on our 3070 [local]. On the pod, build
  2.2.0 for its better kernels. HF `kernels-community/sage-attention` has builds only up to torch 2.10.
- **Measured gains:** LTX-2.3 on an RTX 5090 laptop, 121 frames, 8 steps: fp8 7.08 -> 5.95 s/it with Sage (1.19x)
  (HF LTX-2.3-nvfp4 discussion #2, 2026-03-18, a forum post: unverified). SageAttention3 paper (RTX 5090,
  HunyuanVideo): FlashAttention 489 s, Sage2 240 s, Sage3 164 s. Low-bit attention vs BF16 FlashAttention-4 end to end:
  1.13-1.19x on B200/B300/H200, 1.36-1.70x on RTX PRO 6000/5090 (VC-Attention, arXiv 2609.15810, 2026-09-14).
- **Quality:** Sage2 kept video temporal quality (HunyuanVideo FScore 1.474 vs 1.479 full precision); **Sage3 FP4
  lost some** (1.232) (SageAttention3 paper, Table 2). Sage3 is source-only and does not run on B200 (issue #237).
  Treat Sage3 as a high-risk option.
- **Known fault on sm120:** MiniMax H3 with the fp8-PV kernel gave noise above ~160k tokens (ComfyUI #15263,
  2026-08). Our stage 2 has ~44k tokens; longer clips get closer.
- **FlashAttention 3/4:** `--use-flash-attention` is FA2 (`flash_attn_func`). Core ComfyUI has no FA3/FA4 path. torch
  2.11+ has a prototype `torch.nn.attention.activate_flash_attention_impl("FA3")` (FA4 from 2.10). ComfyUI puts
  Flash first in the SDPA order (PR #15146, 2026-07-29), so activating FA3 at start should route ComfyUI's PyTorch
  attention to FA3 (inference, untested; FA3 is a separate install). FA3 kernel: 1.5-2.0x over FA2 on H100
  (tridao.me, 2024). Lightricks' own ltx-pipelines uses FA3 on Hopper and FA4 4.0.0b9 on B200 (LTX-2 README).
- **Comfy Kitchen attention** (core, v0.32.0; `--use-ck-attention` or the `ModelAttentionBackend` node): pure int8
  attention, NVIDIA code, Apache-2.0, sm75+. Z-Image Turbo 2048²: CK 14.55 s, Sage 14.24 s, PyTorch 23.16 s
  (comfyui-wiki, 2026-08-11; GPU not named). Bugs: TRELLIS.2 corruption (#16027), an RTX PRO 6000 hard reset with Flux3
  (#17332, 2026-10-24). No LTX or Qwen numbers.
- **Per-model attention without GPL packs** (core API, `comfy/ldm/modules/attention.py` [local]): a tiny node of ours
  can call `register_attention_function(...)` and `model.set_model_optimized_attention(fn)`. That gives Sage auto
  for LTX and `sageattn_qk_int8_pv_fp16_cuda(pv_accum_dtype="fp32")` for Qwen in ONE ComfyUI process. This removes
  the reason for the per-family start flag, and with it the restart (section B).
- **Expected gain for us** [est.]: stage 2 is about half attention. A 1.5-2x faster kernel gives **1.2-1.4x on a
  final clip**, less in stage 1 and less where the masked guide attention stays on PyTorch.

## E. Question 4 — torch.compile and the "Comfy model compiler"

- The log line "Comfy model compiler graph breaks: N, rogues: M" [local] is NOT torch.compile. It is the comfy-aimdo
  memory compiler (record the allocation pattern once, then allocate VRAM once) plus per-block CUDA graph replay
  (PR #15861, merged 2026-09-04, v0.35.0; `comfy/model_prefetch.py` [local]). "Diffusion models are just memory
  compiler for now" (the PR). It is on by default with dynamic VRAM; `--disable-comfy-compiler` turns it off.
- Core `TorchCompileModel` (inductor or cudagraphs; no mode option) clones the model with `disable_dynamic=True`, so
  it **turns dynamic VRAM off for that model** (PR #12612, 2026-02-25). EasyCache + compile is slow or EasyCache stops
  working (#14795). KJNodes compile (GPL): a LoRA strength change recompiles; about 4 min per recompile on a 4090
  (WanVideoWrapper #1736, 2025-12-09).
- **Our risk:** the token count changes from clip to clip (relay guides, keyframe guides, look-chain), so each new
  shape recompiles [est.]. The LTX layout graph also swaps the style LoRA per run.
- **Gains published:** none for LTX or Qwen in ComfyUI (unverified). Diffusers Flux on H100: about 2.5x with compile +
  FA3-fp8 + fp8 + CUDA graphs together (PyTorch blog, 2025-07-09); compile alone about 1.23x on FLUX.1-dev (Pruna).
  Lightricks' `--compile` in ltx-pipelines is opt-in with no numbers.
- **Verdict:** test last, on stage 1 of one clip, and only if the first compile is under a minute.

## F. Question 5 — caching and step skipping for LTX

- Core EasyCache supports LTX-2 audio-video (PR #12231, kijai, merged 2026-02-06; decisions from the video stream).
  Defaults: threshold 0.2, start 0.15, end 0.95 [local, `nodes_easycache.py`]. The original PR called LTXV "not very
  good, but does not crash" (#9496).
- No published speed or quality numbers for EasyCache, TeaCache, MagCache or FBCache on LTX-2.x. TeaCache supports
  only the old LTX-Video; MagCache has no LTX; FBCache lists LTXV 0.9.1; Cache-DiT lists LTX2/2.5 but is
  diffusers-based (each repo's README, read 2026-10-24).
- Audio risk: EasyCache on MiniMax H3 skipped 5-6 of 20 steps (1.38-1.40x), video SSIM 0.945-0.963, but the audio was
  damaged (#15326, open). Our clips carry sound.
- **Distilled 8+3 steps:** about 6 steps sit in the default window. Kijai on distilled Wan: EasyCache gave "mostly
  pretty much same results as you'd get setting low steps in the first place" (WanVideoWrapper #811). **Skip it for
  the distilled clips.** Worth one A/B on the dev model (30 steps), threshold 0.05-0.1, start 0.2-0.3.

## G. Question 6 — Lightricks' own speed tips (what applies to us)

- **Decode:** larger tiles (table in section B). The conv VAE (`ltx-2.5-video-vae-conv-bf16`, "faster, lighter") is
  much faster than the diffusion VAE ("higher quality, heavier") (HF Lightricks/LTX-2.5). **A quality cost by
  Lightricks' own words: option only.**
- **MultimodalGuider (dev):** up to four model calls per step (positive, negative, STG, modality). They are NOT batched
  (ComfyUI-LTXVideo code, commit 220e1d4). Lightricks suggests audio `skip_step=1`: "2x faster generation with minimal
  audio quality impact" (docs.ltx.io nodes page; the 2x is their claim, unverified). Our dev graphs use `skip_step 0`
  for audio and video [local, `ltx25dev_layouting_api.json`]. The distilled default runs at CFG 1: one call per step.
- **Fewer dev steps:** gradient-estimation Euler, "20-30 instead of 40" steps at similar quality (ltx-pipelines docs).
  We already run 30.
- **Text encoder:** `LTXVSaveConditioning` / `LTXVLoadConditioning` avoid Gemma reloads (docs.ltx.io). Not needed on
  80-96 GB, where Gemma stays resident. The LTX API encoder costs money: not for us.
- **Multi-GPU:** ltx-pipelines has sequence parallelism ("numerically equivalent"), distributed VAE and Gemma; Linux,
  one node, ≥2 GPUs with P2P; "a latency tool, not a memory tool"; no scaling numbers (ltx-pipelines
  docs/multigpu). Not in ComfyUI. xDiT/xfuser: no LTX-2.5 support found.
- **Timings:** the only official claim is a 10 s 720p clip in 6.8 s on 2x GB200 (VentureBeat, 2026-08-11). The LTX-2
  paper: 1.22 s per step on an H100 at 720p, 121 frames (arXiv 2601.03233, Table 1).
- **Third-party:** LTX-2.5 distilled int8 + conv VAE on an RTX 5090, 1280x704x121, 8+3 steps: sampling 14.31 s,
  whole flow 28.67 s, peak 31.7 GiB (ai-muninn, 2026-09-11). LTX-2.3 distilled 768x512x97, 8 steps (RunPod,
  2026-10-12): RTX PRO 6000 7.5 s, H200 8.1 s, B300 8.7 s, H100 SXM 11.3 s, 5090 15.3 s, 4090 15.8 s, A100 SXM 16.1 s,
  L40S 16.8 s. No LTX-2.5 timing on H100, A100, L40S or B200 was found.

## H. Question 7 — structural changes a big GPU allows

1. **No restart between Qwen and LTX** (section B, D). Saves one restart plus one cold load per run: about 2-6 min on
   a network volume [est.]. No quality risk. Needs per-model attention (our own small node) and a "large" check
   (RAM ≥ 128 GB) in `server.ensure_family`.
2. **Models on fast local storage.** Copy the tier's files from the network volume to the container disk at pod start,
   or rely on the page cache (188 GB RAM on a PRO 6000 or H100 PCIe pod). No quality risk.
3. **Larger decode tiles** (section B). No quality risk (fewer seams if anything).
4. **Stage 1 and 2 without unloading:** already the case in one prompt. Nothing to do.
5. **Batch the look images:** little gain. Qwen is compute-bound; batch 4 took 3.6x the time of batch 1 on a B200
   (PyTorch blog 2026-04-08, from the earlier report). Better: two jobs on two GPUs.
6. **Longer clips (fewer joins):** our code allows `clip_frames` up to 257. A 360-frame shot as 2 clips of 193 frames
   instead of 4 of 121 costs about the same compute [est.: attention 2·25² vs 4·16² latent-frame units, linears
   less], but needs about 1.6x the activation memory per clip. Both IC-LoRAs were trained at 121 frames, so this is a
   quality test, not a speed gain.
7. **1024x576 or 960x544 directly:** a quality choice that costs time (about 1.5x tokens, more attention). The
   Ingredients LoRA was trained only at 768x448.
8. **Pose guide in stage 2:** fits on a big GPU, but the pose guide gave no gain. Skip.
9. **Two jobs at once on one GPU:** little gain for compute-bound diffusion. Two ComfyUI instances on one laptop GPU:
   0.40 -> 0.48 jobs/s with 5 instances, each job much slower (dev.to, SD1.5). MPS: about 1.2x for two Amber jobs on
   Hopper (TACC docs). MIG halves the compute per job; L40S has no MIG. **One job per GPU; use more GPUs (or more
   pods) for parallel seeds.**

## I. Questions 8-10 — Qwen-Image-Edit-2511 on large GPUs

- **Fit:** bf16 (40.9 GB) fits a 48 GB card only with the text encoder evicted; dynamic VRAM does that on demand.
  80-96 GB hold model + encoder + activations. int8/fp8 (~20 GB) fit any 24 GB+ card.
- **How ComfyUI chooses the kernel** (`comfy/ops.py`, `quant_ops.py` [local] and the agent's read of v0.42.1): a file
  with a `quant_config` uses mixed-precision ops automatically ("Using mixed precision operations"); no `--fast` flag.
  fp8 is native at sm89+ (Ada, Hopper, Blackwell) and emulated on A100 and the 3070. int8 runs through `torch._int_mm`
  on all NVIDIA cards. NVFP4 needs sm100+. The comfy-kitchen CUDA backend needs a CUDA 13 torch build. Check the
  `Native ops:` log line on the pod.
- **Speed by format** (rates in section C): on 4090/5090/PRO 6000 int8 is 4x bf16; on L40S/H100 int8 = fp8 = 2x bf16;
  on A100 int8 was 33-50% of fp8 (#14824). Locally int8 ConvRot gave 11.4 s/it vs 17.4 s/it for the emulated fp8 file
  (1.5x) [local, 2026-10-24]. No measured 40-step 2511 s/it exists for H100, L40S, 4090 or 5090 (unverified). H100
  reference: Qwen-Image (not Edit) bf16, 1328², 50 steps, about 34 s (Replicate readme, ~Nov 2025); 2511 with two
  references, bf16, 40 steps: 63.3 s (ModelTC, from the earlier report).
- **Quality by format:** BF16 > INT8 ConvRot > MXFP8 > FP8 (obsxrver latent-divergence ranking, unverified).
  Qwen-Image is "more sensitive to quantization than Flux" (PyTorch blog, B200: MXFP8 LPIPS 0.34, NVFP4 0.41). So:
  int8 ConvRot first; fp8 second; NVFP4 only as an option (5090: 50 steps cfg 4, batch 4: fp8 194 s vs NVFP4 76 s,
  "a bit of quality loss", ANXS1 card, unverified).
- **`--fast fp8_matrix_mult`:** only for old fp8 files without `quant_config`; it clamps activations to ±448 with
  scale 1 (ops.py). Our files do not need it. **`--fast fp16_accumulation`** does nothing for Qwen and LTX: both run
  only in bf16/fp32 (`supported_inference_dtypes`, v0.42.1).
- **Attention for Qwen:** on sm89 and sm120 Sage 2.2's auto choice is the fp8-PV kernel with fp32+fp16 accumulation,
  close to the fp16 path that turned Qwen black; on sm90 it is fp32+fp32 (core.py). For Qwen call
  `sageattn_qk_int8_pv_fp16_cuda(pv_accum_dtype="fp32")` explicitly through our per-model node, and stop on any black
  or NaN image. CK attention is a second candidate (Apache-2.0). Expected 1.2-1.4x [est.].
- **torch.compile for Qwen:** diffusers measured 2.4x on an A100 (earlier report). In ComfyUI it disables dynamic
  VRAM for the model and recompiles on LoRA swaps (AnyPose, Multiple-Angles) and size changes. Test last.
- **CFG batching:** ComfyUI batches cond and uncond in one pass only if `memory_required·1.5 < free memory` AND the
  two token lengths are compatible (lcm ≤ 4x the shorter one, `conds.py`). Our negative "" against a long instruction
  usually runs as two passes. It saves weight reads, not FLOPs: little gain when compute-bound (unverified). The
  `CFGOverride` cut from the earlier report saves real FLOPs.
- **Text encoder (Qwen2.5-VL 7B):** two encodes per edit (positive, negative), with the images at ~384² for the VL part.
  No measured seconds found (unverified); locally the whole prompt is only 10-30 s longer than sampling, with
  streaming. With dynamic VRAM the encoder runs on the GPU and stays resident until evicted; the node cache skips the
  encode when prompt and images are unchanged [local code]. On 80-96 GB: nothing to do. bf16 vs fp8 encoder: a quality
  choice with almost no speed cost once resident.

## J. Question 11 — which RunPod GPU (prices read 2026-10-24)

Prices: runpod.io/pricing and gpu-instance/pricing (read 2026-10-24); getdeploying.com/runpod (updated 2026-10-25)
agrees. **Network volumes work only in Secure Cloud** (docs.runpod.io network-volumes), so Secure prices apply.

| GPU | Secure $/h | Community $/h | VRAM | Pod RAM | Fit for our tiers |
|---|---|---|---|---|---|
| RTX 4090 | 0.89 | 0.34 | 24 GB | 41 GB | 8gb files only; RAM too small for bf16 |
| RTX 5090 | 1.19 | 0.69 | 32 GB | 35 GB | same |
| RTX 6000 Ada | 0.99 | 0.74 | 48 GB | 167 GB | int8 everywhere; bf16 streams |
| L40S | 1.09 | 0.79 | 48 GB | 94 GB | int8; bf16 streams; RAM just under 96 GB |
| A100 SXM / PCIe 80 GB | 1.79 | 1.39 / 1.19 | 80 GB | 117-125 GB | bf16 OK; no fp8; int8 slow (#14824) |
| **RTX PRO 6000** | **2.49** | 1.69 | 96 GB | 188 GB | everything resident; int8/fp8/NVFP4 native |
| H100 PCIe | 2.89 | 1.99 | 80 GB | 188 GB | everything resident |
| H100 SXM | 3.99 | 2.69 | 80 GB | 125 GB | fastest of the 80 GB class |
| H200 | 5.29 | 3.59 | 141 GB | 276 GB | |
| B200 | 7.99 | 5.98 | 180 GB | 283 GB | no SageAttention kernel |
| B300 | 8.99 | 6.94 | 288 GB | 251 GB | almost no int8 |

- **Measured per clip** (RunPod "Best GPU for AI video", 2026-10-12, ComfyUI templates):
  - Wan 2.2 14B fp8, 1280x704x81, 20 steps: B300 482 s, B200 507 s, H100 SXM 616 s, H200 642 s, RTX PRO 6000 656 s,
    H100 PCIe 978 s, A100 SXM 1246 s, L40S 1520 s, RTX 4090 1546 s, RTX 6000 Ada 1693 s.
  - RunPod: below 80 GB the Wan weights streamed from RAM, 3-4x longer on 48 GB cards.
- **Cost per clip at today's Secure prices** [our arithmetic from the Wan fp8 row]: RTX PRO 6000 $0.45, H100 SXM
  $0.68, H100 PCIe $0.79, B200 $1.13, B300 $1.20, L40S $0.46, RTX 6000 Ada $0.47, A100 SXM $0.62.
  - L40S and RTX 6000 Ada cost about the same per clip as the PRO 6000 but take 2.3-2.6x longer per clip, and they
    must stream our bf16 models.
  - **Choice: RTX PRO 6000** for the first session (best measured $/clip, 96 GB holds every model of a family, 188
    GB RAM, int8 and fp8 native). **H100 SXM** when wall time matters more than money. B200/B300 are about 2x the
    price per clip and lose SageAttention or int8.
- **Two jobs at once:** one job per GPU (section H.9). For parallel seeds, rent a 2x pod or two pods (multi-GPU pods
  bill per GPU; PRO 6000 offered as 1/2/4/8x, getdeploying).
- **Other costs:** network volume $0.07/GB-month (220 GB 24gb tier ≈ $15/month); container disk $0.10/GB-month
  (RunPod pricing docs). No spot tier is listed.
- **Our tier names do not match the pods.** The `24gb` tier needs about 96 GB RAM (models.json), but RunPod's 24-32 GB
  cards come with 35-41 GB RAM. The large tier really means 80-96 GB VRAM pods.

## K. Question 12 — ComfyUI start flags on a big dedicated Linux GPU (v0.42.1)

| Flag | What it does in 0.42.1 [local code + agent read] | Use on a big GPU? | Risk |
|---|---|---|---|
| (defaults) dynamic VRAM, async offload 2 streams, pinned memory, cudaMallocAsync, Comfy compiler | Resident when it fits; on-demand eviction | **Yes, keep** | #15255 dynamic VRAM crashes (open since 2026-08-03, mostly multi-GPU, also single Blackwell); workaround `--disable-dynamic-vram` or `--disable-pinned-memory` |
| `--highvram` | HIGH_VRAM AND dynamic VRAM off | **No** | Back to the old estimate loader; loses the Comfy compiler |
| `--gpu-only` | Same + text encoders and VAE on the GPU | **No** | Same; OOM when two families meet |
| `--disable-smart-memory` | Unloads whole models after use | No | Slower; reloads every prompt |
| `--reserve-vram GB` | Default reserve 400 MB on Linux | Only if another process uses VRAM | None |
| `--vram-headroom GB` | Extra free VRAM for dynamic VRAM | No (a dedicated GPU) | None |
| `--fast` (fp16_accumulation, fp8_matrix_mult, cublas_ops, autotune) | Only these four exist | `autotune` maybe (cudnn.benchmark for the VAE convs); the rest do nothing for our bf16/quant files | `--fast` with no list turns on all four; first run slower with autotune |
| `--use-sage-attention` | Global; auto kernel; `smooth_k=False` | Only for an LTX-only process; better per model | Qwen black output |
| `--use-flash-attention` | FA2 package | No (PyTorch SDPA already tries flash first on Linux) | Extra install |
| `--use-ck-attention` | Core int8 attention (Apache-2.0) | Test | New; two bug reports |
| `--fast-disk` | Disk-backed loading | No on a network volume | Slow loads |
| `--cache-ram` (default) / `--cache-none` | Node output cache | Keep default | `--cache-none` re-encodes every prompt |
| `--disable-cuda-malloc` | Off switch | Only on a CUDA error | Slower allocations |

## Tables

### (a) Tier-aware settings to add

| Setting | 8gb value | large (80-96 GB) value | Expected gain | Risk |
|---|---|---|---|---|
| LTX `VAEDecodeTiled` | 512 / 64 / 64 / 16 | 1536 / 384 / 192 / 48 (48 GB: 1024 / 256 / 128 / 48) | decode 1.5-2x [est.]; maybe 20-40 s per final clip on the 3070 scale | OOM -> ComfyUI retries tiled |
| torch build in `install_linux.sh` | cu128 (gets torch 2.11) | cu130, torch 2.12.1 pinned | comfy-kitchen CUDA kernels, the tested torch | needs driver ≥ 580 (filter CUDA 13.0) |
| SageAttention | 1.0.6, LTX only, start flag | 2.2.0 built from source; per-model node: auto for LTX, fp16_cuda + fp32 PV for Qwen; none on B200 | LTX 1.2-1.4x on final clips, Qwen 1.2-1.4x [est.] | Qwen black images (check every image); build time |
| Family restart | every family change | none when RAM ≥ 128 GB and attention is per model | 2-6 min per run on a pod [est.] | RAM pressure if both families are cached |
| Model storage | local disk | copy to container disk at pod start (or page cache) | 2-5x faster cold loads [est.] | container disk is lost with the pod |
| LTX dev transformer | int8 ConvRot | int8 ConvRot by default; bf16 only after a blind A/B win | 1.3-1.8x (GeForce/PRO), 1.2-1.5x (H100) [est.] | small detail loss |
| Gemma 4 text encoder | int8 | bf16 (resident: no speed cost) | none | none |
| Qwen edit model | int8 ConvRot | int8 ConvRot on PRO 6000/4090/5090; int8 or fp8mixed on H100/L40S; never int8 on A100 | 1.5-2x vs bf16 [est.] | small detail loss vs bf16 |
| Dev guider audio `skip_step` | 0 | 1 (Lightricks baseline) | dev only; Lightricks claims up to 2x (unverified) | audio quality |
| EasyCache | off | Qwen (40 steps) and LTX dev (30 steps) only, after an A/B; never distilled | 1.3-1.5x [est.] | fine detail, audio |
| ComfyUI flags | Desktop's (`--vram-headroom 1 --fast-disk`) | defaults only; never `--highvram`/`--gpu-only` | keeps dynamic VRAM + Comfy compiler | none |
| `clip_frames` | 121 | 121 (193 as a quality test) | fewer joins, same compute | out of the LoRAs' training length |

### (b) Ranked optimizations for the large tier

LTX-2.5 (default = distilled layout clips; dev noted where it differs):

| # | Optimization | Speedup | Quality risk | Effort | Licence |
|---|---|---|---|---|---|
| 1 | No family restart + models on local disk/page cache | 2-6 min per run [est.] | none | medium (ensure_family, per-model attention node, pod start script) | ours |
| 2 | cu130 torch 2.12.1 in the installer | enables native kernels | none | tiny | BSD (torch) |
| 3 | Larger decode tiles | decode 1.5-2x [est.] | none | tiny (graph values per tier) | - |
| 4 | int8 ConvRot instead of bf16 for dev (and Qwen) | 1.2-1.8x by GPU [est.] | low-medium (A/B) | tiny (tier files) | LTX-2 Community Licence / Apache-2.0 |
| 5 | SageAttention 2.2 (or FA3 on H100, or CK attention) | 1.2-1.4x per final clip [est.] | low (8-bit); high for Sage3 FP4 | medium (source build, node) | Apache-2.0 / BSD |
| 6 | Dev: audio `skip_step` 1 | up to 2x claimed (unverified) | audio | tiny | - |
| 7 | Dev: EasyCache | 1.2-1.5x (unverified) | medium (detail, audio) | tiny (core node) | GPL-3.0 core |
| 8 | torch.compile | 1.1-1.3x (unverified) | none, but dynamic VRAM off, recompiles | medium | core |
| 9 | Conv VAE | big decode gain | **yes** (Lightricks: DiffVAE is "higher quality") | tiny | LTX licence |
| 10 | NVFP4 distilled (Blackwell) | sampling only 1.05x vs int8 on a 5090 | medium-high | small | LTX licence |
| 11 | Multi-GPU sequence parallel | latency only | none | large (ltx-pipelines, not ComfyUI) | LTX licence |

Qwen-Image-Edit-2511 (add the earlier report's graph wins: no duplicate Picture 2 1.7x, CFGOverride 1.2x, 30 steps):

| # | Optimization | Speedup | Quality risk | Effort | Licence |
|---|---|---|---|---|---|
| 1 | Drop the duplicate Picture 2 (earlier report) | 1.7x | low-medium | small | ours |
| 2 | int8 ConvRot on PRO 6000/4090/5090 (fp8mixed or int8 on H100/L40S) | 1.5-2x vs bf16 [est.]; 1.5x measured locally vs emulated fp8 | low | tiny | Apache-2.0 |
| 3 | SageAttention 2.2 fp16_cuda + fp32 PV per model (or CK attention) | 1.2-1.4x [est.] | medium (black output: check) | medium | Apache-2.0 |
| 4 | CFGOverride / 30 steps / EasyCache (earlier report) | 1.2-1.5x each | low-medium | tiny | core |
| 5 | Keep Qwen2.5-VL resident | small (automatic with dynamic VRAM) | none | none | - |
| 6 | torch.compile | 1.1-1.3x Linux (unverified) | none; recompiles on LoRA swaps | medium | core |
| 7 | NVFP4 (Blackwell) | about 2.5x vs fp8 (5090 card) | high (LPIPS 0.41 on B200) | small | Apache-2.0 (community files) |

### (c) Test plan for the first RunPod session

Pod: **RTX PRO 6000, Secure Cloud**, with the network volume in the same data center, CUDA filter 13.0. Fill the
volume first (about 150-220 GB at ~100 MB/s or more, 30-40 min); a cheaper GPU pod in that data center can do the
download. Every test: one change, same seed, timed (s/step per stage, decode, prompt total, `nvidia-smi dmon`),
quality checked by eye side by side with the baseline (AGENTS.md: side, fingers, colours, background; for clips the
strip and the sound). Record each result in AGENTS.md section 3 or 7 when it wins.

Fixed test set: one layout run (masked3d, a built-in template, `--quality final`, seed 333, clip 1 only via
`--through-clip 1`), the same run in preview, and the Qwen set from the earlier report (2 edits without reference, 1
with, 1 repose, 1 angle view).

| # | Test | One change | Expect | GPU time |
|---|---|---|---|---|
| 0 | Install, `doctor`, read `Native ops:`, torch/CUDA versions, record the image tag | `--cuda cu130` | native int8 + fp8 + nvfp4 listed | 0.5 h |
| 1 | LTX baseline: today's 8gb graphs (int8, tiles 512/64/64/16, Sage 1.0.6), cold then warm | - | reference times; cold vs warm = load cost | 0.25 h |
| 2 | Decode tiles 1024/256/128/48, then 1536/384/192/48, then spatial-only 768/192 | tiles | decode 1.5-2x; same picture | 0.2 h |
| 3 | Attention: PyTorch vs Sage 1.0.6 vs Sage 2.2 (source build) vs CK attention | attention | 1.2-1.4x on stage 2 | 0.5 h (+ build) |
| 4 | Load path: restart + network volume vs container disk vs no restart (Qwen look then LTX clip) | storage / restart | minutes per run | 0.3 h |
| 5 | Dev model: int8 vs bf16, 30 steps; then audio `skip_step` 1 | precision, then skip_step | 1.2-1.8x for int8; blind A/B | 1.0 h |
| 6 | Qwen: bf16 vs fp8mixed vs int8 on the fixed set | precision | int8 ≈ 1.5-2x vs bf16 | 0.7 h |
| 7 | Qwen attention: Sage 2.2 fp16_cuda + fp32 PV (per-model node), then CK attention | attention | 1.2-1.4x; abort on any black image | 0.4 h |
| 8 | EasyCache conservative on Qwen (40 steps) and LTX dev (30 steps) | cache | 1.3-1.5x; check detail and sound | 0.4 h |
| 9 | torch.compile (core `TorchCompileModel`) on distilled stage 1 | compile | record first compile time and s/step | 0.3 h |
| 10 | Optional: repeat tests 1 and 6 on an H100 SXM | GPU | $/clip comparison | 0.5 h on H100 |

Total about **4.5-5 GPU-hours on the RTX PRO 6000 (about $11-13 at $2.49/h)** plus about 0.5 h on an H100 SXM
(about $2) and the volume ($15/month). Add a buffer of 1-2 h for the SageAttention build and surprises.

## Sources

Repo and local (read 2026-10-24): `setup/make_25_graphs.py`, `setup/make_layout_graphs.py`,
`workflows/ltx25_layouting_api.json`, `workflows/ltx25dev_layouting_api.json`, `cinestudio/comfy.py`,
`cinestudio/server.py` (`ensure_family`), `cinestudio/runner.py` (`to_stage1_only`, `SIZES`, relay), `cinestudio/recipe.py`,
`setup/models.json`, `setup/install_linux.sh`, `setup/runtime.lock.json`, `docs/RUNPOD.md`, `.cinestudio/comfyui.log`;
ComfyUI 0.42.1: `comfy/cli_args.py` (`enables_dynamic_vram`, `PerformanceFeature`), `comfy/model_management.py`,
`comfy/model_prefetch.py`, `comfy/model_base.py`, `main.py`; ComfyUI-LTXVideo 220e1d4: README "Low VRAM",
`example_workflows/2.5/README.md` and the Decode notes in its graphs.

LTX:
- HF Lightricks/LTX-2.5 (files, VAEs): https://huggingface.co/Lightricks/LTX-2.5 ; discussion #15 (2026-08-14): https://huggingface.co/Lightricks/LTX-2.5/discussions/15
- Layout-To-Render card v1.0: https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render ; Ingredients card v0.9: https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients
- LTX-2 repo and ltx-pipelines docs (optimization.md, pipelines.md, multigpu/README.md): https://github.com/Lightricks/LTX-2
- docs.ltx.io (system requirements, comfy-ui, nodes, workflow-asset-compatibility): https://docs.ltx.io/llms.txt
- unsloth LTX-2.5 quant comparison: https://huggingface.co/unsloth/LTX-2.5-FP8
- ai-muninn RTX 5090 LTX-2.5 timings (2026-09-11): https://ai-muninn.com/en/blog/ltx25-vs-minimax-h3-rtx5090
- TrueNorthAI (2026-08-12): https://note.com/truenorthai/n/nf600b6190507
- LTX-2 paper: https://arxiv.org/html/2601.03233 ; VentureBeat (2026-08-11): https://venturebeat.com/technology/ltx-2-5-can-generate-a-10-second-ai-video-from-an-image-in-just-6-8-seconds-on-nvidia-superchips-and-its-open-weights
- Comfy blog LTX-2.5 day 0: https://blog.comfy.org/p/ltx-25-day-0-support-in-comfyui ; NVIDIA RTX AI Garage (2026-03-10): https://blogs.nvidia.com/blog/rtx-ai-garage-flux-ltx-video-comfyui-gdc/
- LTX-Video-Q8-Kernels: https://github.com/Lightricks/LTX-Video-Q8-Kernels
- LTX-2.3 Sage forum timing: https://huggingface.co/Lightricks/LTX-2.3-nvfp4/discussions/2

ComfyUI:
- Dynamic VRAM blog (2026-03-25): https://blog.comfy.org/p/dynamic-vram-in-comfyui-saving-local ; NVIDIA optimizations blog (2026-01-09): https://blog.comfy.org/p/new-comfyui-optimizations-for-nvidia
- Issues/PRs: #15255 https://github.com/Comfy-Org/ComfyUI/issues/15255 ; #14824 https://github.com/Comfy-Org/ComfyUI/issues/14824 ; #12231 https://github.com/Comfy-Org/ComfyUI/pull/12231 ; #9496 https://github.com/Comfy-Org/ComfyUI/pull/9496 ; #15326 https://github.com/Comfy-Org/ComfyUI/issues/15326 ; #15861 (Comfy Compiler) https://github.com/Comfy-Org/ComfyUI/pull/15861 ; #12612 (compile disables dynamic VRAM) https://github.com/Comfy-Org/ComfyUI/pull/12612 ; #14795 ; #15146 (SDPA order) https://github.com/Comfy-Org/ComfyUI/pull/15146 ; #15263 (sm120 noise) ; #16027, #17332 (CK attention bugs)
- v0.32.0 news (CK attention benchmark): https://comfyui-wiki.com/en/news/2026-08-11-comfyui-v0-32-0 ; changelog: https://docs.comfy.org/changelog
- comfy-kitchen: https://github.com/Comfy-Org/comfy-kitchen

Attention, compile, cache:
- SageAttention: https://github.com/thu-ml/SageAttention ; SageAttention3 paper: https://arxiv.org/abs/2505.11594 ; HF kernels: https://huggingface.co/kernels-community/sage-attention
- FA3: https://tridao.me/blog/2024/flash3/ ; FA4: https://arxiv.org/abs/2603.05451 ; VC-Attention: https://arxiv.org/abs/2609.15810
- PyTorch Flux-fast (2025-07-09): https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/ ; PyTorch MXFP8/NVFP4 B200 (2026-04-08): https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/
- WanVideoWrapper #1736 (recompiles), #811 (EasyCache on distilled): https://github.com/kijai/ComfyUI-WanVideoWrapper
- TeaCache: https://github.com/welltop-cn/ComfyUI-TeaCache ; MagCache: https://github.com/Zehong-Ma/ComfyUI-MagCache ; WaveSpeed: https://github.com/chengzeyi/Comfy-WaveSpeed ; Cache-DiT: https://github.com/vipshop/cache-dit
- torch wheel index (no cu128 for 2.12): https://download.pytorch.org/whl/cu128

Qwen:
- obsxrver INT8 ConvRot ranking: https://huggingface.co/obsxrver/ComfyUI-Native-INT8_ConvRot ; ANXS1 NVFP4: https://huggingface.co/ANXS1/Qwen-Image-Edit-2511-NVFP4 ; Replicate Qwen-Image H100: https://replicate.com/smkrv/qwen-image/readme ; Qwen-Image-Edit-Causal (H100 1/2-ref): https://github.com/ModelTC/Qwen-Image-Edit-Causal

GPUs and RunPod:
- RunPod pricing (read 2026-10-24): https://www.runpod.io/pricing , https://www.runpod.io/gpu-instance/pricing , https://docs.runpod.io/pods/pricing ; getdeploying (2026-10-25): https://getdeploying.com/runpod
- RunPod network volumes (Secure only, 200-400 MB/s): https://docs.runpod.io/storage/network-volumes ; pod API CUDA filter: https://docs.runpod.io/api-reference/pods/POST/pods.md
- RunPod "Best GPU for AI video" (2026-10-12): https://www.runpod.io/articles/guides/best-gpu-for-ai-video
- NVIDIA: RTX Blackwell whitepaper https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf ; Ada whitepaper https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf ; L40S https://www.nvidia.com/en-us/data-center/l40s/ ; A100 https://www.nvidia.com/en-us/data-center/a100/ ; H100 https://www.nvidia.com/en-us/data-center/h100/ ; HGX (B200/B300) https://www.nvidia.com/en-us/data-center/hgx/ ; RTX PRO 6000 https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/
- MPS (TACC): https://docs.tacc.utexas.edu/hpc/vista/mps ; MIG user guide: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/
