# Making Qwen-Image-Edit-2511 faster (research, 2026-10-24)

Scope: web research plus a read of our own ComfyUI install and log. No downloads, no GPU runs, no code changes.
Numbers from other people are quoted with who measured them and on what GPU. Our own estimates are marked
**estimate**. Things nobody has confirmed are marked **unverified**.

## 0. Summary

- **Our edits are compute-bound, not transfer-bound.** The coordinator's reasoning holds (section A). The RTX 3070 runs
  at about 80% of its bf16 tensor peak during sampling. Streaming the 19.6 GB from RAM costs 10% or less.
- **The biggest free win is the token count.** Our graph always sends two images. When there is no reference, it sends
  the same picture twice. An H100 test by ModelTC shows that a second reference image makes a 40-step 2511 edit 1.71x
  slower (37.0 s with one image, 63.3 s with two). Our FLOP model predicts the same 1.7x for our graph.
- **The second-best win is a core ComfyUI format we already have the kernels for:** Comfy-Org publishes an official
  `qwen_image_edit_2511_int8_convrot.safetensors` (20.5 GB). Our ComfyUI 0.42.1 lists `int8_tensorwise` as a
  NATIVE op on the 3070. Today our fp8mixed file is EMULATED (upcast to bf16 and run as bf16). On GeForce cards int8
  tensor math runs about 4x faster than bf16 with fp32 accumulation. Quality reports put INT8 ConvRot close to bf16,
  closer than fp8.
- **SageAttention went black because we have SageAttention 1.0.6** (Triton only). It casts V to fp16 and accumulates
  P·V in fp16. ComfyUI also calls it with `smooth_k=False`. Qwen-Image is a model that ComfyUI never runs in fp16,
  because its values overflow fp16. The known fix is SageAttention 2.2 with the `sageattn_qk_int8_pv_fp16_cuda`
  kernel and `pv_accum_dtype="fp32"`, used only for Qwen.
- **CFG:** core ComfyUI 0.42.1 has a `CFGOverride` node. Set to cfg 1.0 over a step range, it really skips the
  negative pass (we read the code). This saves up to half the work in that range.
- Combined, without distillation, a rough **estimate** for the 3070 is 12 min -> about 3-4 min per edit. Each step needs its
  own A/B test.
- On RunPod: an H100 already runs our kind of edit about 11x faster than the 3070 (63 s with 2 images, bf16,
  diffusers, measured by ModelTC). bf16 fits only on 48 GB+ cards (L40S tight, A100/H100 80 GB comfortable).

## A. Is our edit compute-bound? (checking the coordinator's estimate)

What our graph sends (read from `workflows/qwenedit_edit_api.json` and ComfyUI's `comfy_extras/nodes_qwen.py`):

- The output latent is `VAEEncode(FluxKontextImageScale(Picture 1))`: about 1 MP (for 2:3, 832x1248) -> about 4,050
  image tokens (latent /8, patch 2x2).
- `TextEncodeQwenImageEditPlus` resizes EVERY input image again to about 1024x1024 total pixels (multiple of 8) for the
  VAE reference latents, whatever size it gets. So `FluxKontextImageScale` on Picture 2 does not change its token
  count. Each reference adds about 4,100 tokens.
- The VL encoder sees each image at about 384x384 total pixels (about 190 vision tokens each, in the text stream).
- Total: about 4,050 + 2 x 4,100 = **about 12,200 image tokens** plus about 500 text tokens. This matches the coordinator's
  "about 12k".
- Both the positive and the negative conditioning carry both reference latents. So the negative pass costs as much as
  the positive pass.

FLOP model (**estimate**). Qwen-Image has 60 dual-stream MMDiT blocks, hidden size 3072, MLP 12288. Each image token
passes through about 6.8 B weights per forward (the text tokens use the other stream). That gives about 13.6 GFLOP per
token for the linears.
- Linears: 12.2k tokens x 13.6 GFLOP = about 170 TFLOP per pass.
- Joint attention: 4 x L² x 3072 x 60 layers with L about 12.7k = about 120 TFLOP per pass (about 41% of the total).
- Total per pass: about 290 TFLOP. At cfg 3 (2 passes): about 580 TFLOP per step.
- Measured: 17.5-18 s per step (our log: 40 steps, "Prompt executed in" 11:32-12:29, so text encoding and VAE add only
  10-30 s). That is about 33 TFLOPS sustained.
- The RTX 3070 peak for bf16 tensor math with fp32 accumulation is about 40 TFLOPS dense (NVIDIA Ampere GA10x
  whitepaper figure, from memory). ComfyUI runs Qwen in bf16 (`supported_inference_dtypes = [bf16, fp32]` in our
  `comfy/supported_models.py`). So we run at about 80% of peak. **The GPU is the limit.**
- Transfer: 19.6 GB per forward pass over PCIe 4.0 x16 (about 25 GB/s) = about 0.8 s, or 1.6 s if cond and uncond run as
  separate passes. That is 5-10% of a step. ComfyUI's async offload (2 streams, on in our log) overlaps it with
  compute.
- Cross-check: the same model predicts 1.71x between one and two 1 MP references. ModelTC measured 63.3 s / 37.0 s =
  1.71x on an H100 (section 8). The model holds.

How to confirm locally: `nvidia-smi dmon` during a step should show the GPU near 100% busy and the power near its limit.

## 1. Nunchaku / SVDQuant (int4, NVFP4)

- **Official support for 2511: none.** The nunchaku-ai Hugging Face org has Qwen-Image, Qwen-Image-Edit and
  Qwen-Image-Edit-2509 only (HF API, checked 2026-10-24). The request issue for 2511 (nunchaku #858, 2025-12-31) was
  closed as inactive with no official release. The last Nunchaku release is v1.2.0 (2026-01-12).
- **Community builds:** QuantFunc/Nunchaku-Qwen-Image-EDIT-2511 (INT4 and FP4, ranks 32/128/256, Apache-2.0, not
  affiliated with Nunchaku). The README pushes its own engine and ComfyUI plugin, and its "2x-11x" claim has no
  benchmark. In #858 one user ran a script-made 2511 int4 and got lower quality and skipped LoRAs.
  LAXMAYDAY/wazimondo `Qwen_image_edit_2511_int4` publishes metrics against a bf16 50-step reference: PSNR about 19-21 dB,
  LPIPS about 0.17-0.21. That is a clearly visible difference.
- Hardware: INT4 runs on Turing/Ampere/Ada. NVFP4 needs Blackwell. **Hopper (H100) is not supported** (HF blog
  "Bringing Nunchaku 4-bit Diffusion Inference to Diffusers", 2026-07-23).
- Speed: Nunchaku claims 2-3x over bf16 for Qwen-Image on an RTX 4090 (secondary summary, **unverified**). No 2511
  numbers. A user in #858 saw about 12 min per run with a script-made 2511 int4, against about 19 s with the official
  Qwen-Image int4.
- Windows: ComfyUI-nunchaku has a wheel-installer node. Windows wheels exist for older torch versions. Our torch
  2.12+cu130 is **unverified**.
- Licence: Nunchaku and ComfyUI-nunchaku are Apache-2.0. The weights are derived from Apache-2.0.
- Verdict: **skip.** There is no official 2511 build, the quality is measurably lower, LoRA support is shaky, and core
  ComfyUI now has its own 4-bit and 8-bit formats.

## 1b. Core ComfyUI quant formats (the coordinator's extra question)

Our log: `Native ops: convrot_w4a4, int8_tensorwise, w6a8_int8, asym_w4a8_int8, emulated ops: float8_e5m2,
float8_e4m3fn, mxfp8, nvfp4`. ComfyUI 0.42.1, comfy-kitchen 0.2.39, torch 2.12.1+cu130.

What they are (from our `comfy/quant_ops.py` and the PRs):
- `int8_tensorwise`: W8A8 (int8 weights, int8 activations quantized on the fly), optional ConvRot (Hadamard rotation,
  for example group 256) and per-row scales. Added by comfyanonymous in PR #14636, merged 2026-06-25 (v0.27.0). The
  int8+LoRA requantization bug (#14642: LoRAs dropped the rotation) was fixed in PR #14650 (2026-06-26).
- `convrot_w4a4`: 4-bit weights and 4-bit activations, int4 tensor-core matmul with an int8 fallback (PR #14859, merged
  2026-07-09). The author's own test model "will not be very high quality".
- `asym_w4a8_int8`: 4-bit weights (codebook, Hadamard) with int8 activations (Kijai, comfy-kitchen PR #90).
- `w6a8_int8`: the same layout with 6-bit uniform weights (comment in our quant_ops.py). No public write-up found.
- Ampere has int8 and int4 tensor cores but no fp8. That is why fp8 is emulated on the 3070 and int8 is native.

Ready-made 2511 files (all transformer only; use the usual Qwen2.5-VL encoder and VAE):
- **Comfy-Org/Qwen-Image-Edit_ComfyUI `split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors`**,
  20,499,083,824 bytes. Official, loads with the stock UNETLoader. Licence: the Apache-2.0 base. Comfy-Org gives no
  speed or quality numbers. **This is the one to test.**
- wazimondo or LAXMAYDAY `Qwen-Image-Edit-2511-int8-tensorwise`: int8 + ConvRot g256 on all 839 linears. Five
  embed/out layers stay bf16. It has the `__index_timestep_zero__` sentinel that 2511 needs. Weight SQNR about 40.6 dB.
  Made with Comfy-Org/comfy-quants. Apache-2.0.
- cardamonnl `Qwen-Image-Edit-2511-int8-convrot` (made with silveroxides `convert_to_quant`, `--int8 --scaling_mode row
  --simple --convrot --comfy_quant`, plus a "learned rounding" variant). Apache-2.0.
- LAXMAYDAY `Qwen-Image-Edit-2511-int4-tensorwise-mixed`: W4A4 on 480 layers, int8 on 359 sensitive ones. It needs an
  unreleased comfy-kitchen PR #63 runtime. Measured by the author on an NVIDIA L4 (sm_89), 20-step edit: bf16 114 s,
  int8_tensorwise 93.1 s, int4-mixed 87.1 s. PSNR vs bf16: int8 30.05 dB, int4-mixed 26.58 dB.
- No 2511 file in `asym_w4a8_int8` or `w6a8_int8` was found.

Speed evidence for int8 (no Qwen-2511 measurement on a 3070 exists):
- comfyanonymous (issue #14824, 2026-07-08): "int8 tensor ops are only 2x faster than fp16/bf16 [on A100] while on 30,
  40, 50 series it's 4x faster." On GeForce cards bf16 with fp32 accumulation runs at half rate; int8 does not.
- bertbobson (Z-Image int8 tensorwise card): about 2x over fp8 and bf16 on an RTX 3090, 1.3x over fp8 on an RTX 5060.
  Self-reported.
- Another 3090 card: bf16 1.76 s/it -> INT8 (QuIP) 1.09 s/it (1.6x).
- Against that: an RTX A6000 (sm_86 workstation card, where bf16 runs at full rate) saw no gain. An A100 went from 2-3x
  SLOWER to 1.4x faster after a clean reinstall (#14824, closed "not planned").
- comfy-quant-bench (JoaoZaokk, MIT, RTX 3090): Qwen-Image-2512 W4A8 3.65x per step vs bf16 (the bf16 run must have
  offloaded on 24 GB, so this inflates the ratio). On Qwen-Image-Edit-2511, **W4A4 gave bad static output, W4A8
  worked**. For Qwen-Image-2.1 in a blind review, "full INT8 ConvRot looked like BF16", while plain W4A4 looked poor in 18
  of 24 scenes.
- SparknightLLC (RTX 3090, Krea2): W4A8 was 12.5-15% SLOWER than INT8 ConvRot (smaller, not faster).
- Quality ranking by latent divergence (obsxrver card): BF16 > INT8 ConvRot > MXFP8 > FP8 >= INT8 row > INT8 tensorwise
  (plain). INT8 ConvRot beat GGUF Q8 on Qwen-Image-2512.

Expected on our 3070 (**estimate**): the linears are about 59% of the time. If int8 GEMMs give 2-3x in practice,
0.41 + 0.59/2.5 = 0.65, so **about 1.3-1.6x**. On a 4090 or 5090 int8 is also about 4x bf16 (GeForce). On L40S and
H100 int8 runs at the same rate as fp8 (2x bf16).

Making our own (only if we want W4A8/W6A8 or another recipe; for int8 the official file is enough):
- Comfy-Org/comfy-quants (GPL-3.0 tool; the output weights keep the model's licence): `export-model-int8-tensorwise`,
  `export-model-int4-tensorwise` (on a branch).
- silveroxides `convert_to_quant` (`pip install convert-to-quant`; `ctq ... --int8 --convrot --comfy_quant
  --save-quant-metadata`; has a Qwen preset). Licence **unverified**.
- comfy-quant-bench `tools/quant_w4a8.py` (MIT) for `asym_w4a8_int8`.
- SparknightLLC ComfyUI-QuantizationToolkit can quantize at load time and save. It is AGPL-3.0, so avoid it.
- Core ComfyUI has no node that quantizes to int8 at load time (UNETLoader offers only fp8 options).
- Keep the `__index_timestep_zero__` sentinel and the bf16 embedding/output layers like the published files do.

## 2. GGUF (Q4_K_M .. Q8_0)

- Available: unsloth/Qwen-Image-Edit-2511-GGUF (Dynamic 2.0, important layers upcast). Its licence follows the base,
  Apache-2.0.
- Speed: ComfyUI-GGUF (city96, Apache-2.0) dequantizes each layer to bf16 and then runs a bf16 matmul. Our edit is
  compute-bound, so GGUF cannot be faster than today. It adds dequant work and is usually a little slower than fp8.
  Its benefit is RAM and VRAM size only. No 2511 timing was found.
- Quality (MyAIForce, 2026-01-02, 2511): Q6 (16.8 GB) looked close to bf16. "Q4 basically unusable". bf16 vs fp8: faces
  identical, bf16 kept a little more dress detail.
- Verdict: **no speed gain.** Q8_0 is a quality option if RAM is tight, not a speed option.

## 3. SageAttention and FlashAttention

Why it is black for us (checked in our install):
- Our venv has **sageattention 1.0.6** (Triton kernels only; triton-windows 3.7.1). In
  `sageattention/core.py` it does `v = v.to(torch.float16)` for bf16 inputs. The kernel accumulates P·V with
  `tl.dot(p, v, out_dtype=tl.float16)`.
- ComfyUI's `attention_sage` (our `comfy/ldm/modules/attention.py`) calls `sageattn(..., smooth_k=False)`.
- Qwen-Image activations overflow fp16. That is why ComfyUI only runs Qwen in bf16 or fp32. An fp16 accumulator, or V
  values above 65504, give inf, then NaN, then a black image. This matches the public reports: ComfyUI #9773
  (Triton kernel gives NaN on sm80/sm86 for Qwen), #9184, #9223. The woct0rdho fork README says "some intermediate
  values overflow SageAttention's quantization" for Qwen-Image and Wan.

What works (community reports, not a guarantee):
- SageAttention **2.2** with `sageattn_qk_int8_pv_fp16_cuda(..., pv_accum_dtype="fp32")`: P·V accumulates in fp32,
  and it is "least likely to overflow" (woct0rdho README, KJNodes docs). Confirmed for Qwen on an RTX 3xxx and an RTX 4090
  (#9773, 2025-09-30). One RTX 4090 user still got black with every setting (2025-10-07). One RTX 3070 user said it
  "did not work for Qwen Image Edit 2511" (2026-03-26), without saying which kernel. So **test it, do not assume it.**
- Do not use `pv_accum_dtype="fp16"` or the Triton kernel for Qwen. On sm86 SageAttention 2.2's automatic choice is
  the Triton kernel (#9773), so `--use-sage-attention` alone is not enough.
- How to do it for Qwen only: (a) KJNodes "Patch Sage Attention KJ" with `sageattn_qk_int8_pv_fp16_cuda` (KJNodes is
  GPL-3.0); or (b) better for us, our own tiny node or a launch hook that sets core ComfyUI's
  `transformer_options["optimized_attention_override"]` (it exists in our attention.py) to call the fp16_cuda kernel
  with fp32 accumulation and `smooth_k=True`, on the Qwen model only. LTX keeps its current setting.
- Install: woct0rdho/SageAttention Windows wheels (abi3, "torch2.10.0andhigher"). The CUDA kernels are built for
  sm80/89/90/120, and sm80 code runs on our sm86. A cu130 wheel for torch 2.12 is **unverified**. Licence: SageAttention is
  Apache-2.0 (thu-ml upstream; **verify** the fork).

Speed:
- Kernel claims: SageAttention 1 is about 2.1x FlashAttention2 (paper, RTX 4090). SageAttention 2 is faster still.
- Qwen-Image-Edit: one RTX 5090 user, 40 steps, 14m30s -> 9m30s (about 1.5x total, with the fp16_cuda kernel)
  (ComfyUI discussion #11583). The absolute times look very slow for a 5090 (maybe offloading or a large size); the
  ratio is the useful part.
- Our 3070 (**estimate**): attention is about 41% of a step. A 1.6-2x faster attention gives **about 1.2-1.3x** total.
  After int8 linears the attention share grows, so the gain is larger then.
- SageAttention 3 (FP4 attention) is Blackwell only (RTX 5090, B200) and has a larger quality risk. It is in ComfyUI
  core as `sageattn3_blackwell`.
- FlashAttention 3 (Hopper): 1.5-2.0x over FA2 for the bf16 forward (FA3 paper, kernel level). Qwen-Image training on
  4x H100: SDPA 9 s/it, xformers 5.6 s/it, FA3 4.6 s/it (musubi-tuner #874). In ComfyUI, `--use-flash-attention`
  uses the flash_attn package (FA2). FA3 in ComfyUI is **unverified**. On H100 SageAttention 2 also has an sm90 kernel.

## 4. Cache-based step skipping

- **EasyCache / LazyCache** (core ComfyUI, PR #9496, merged 2025-08-23, by Kosinkadink). The author measured "Qwen
  Image Edit" (2509-era) at 20 steps: 9 of 20 steps skipped, **1.82x** (EasyCache 0.1/0.2/0.95; LazyCache
  0.2/0.2/0.95). Flux Kontext at 40 steps: 2.11x and 2.22x. The quality cost is "some" (the author's words). Not
  independently measured for Qwen. A MiniMax-H3 user reports EasyCache is lossy at about 1.8x and that `end_percent`
  helps. At 40 steps the skip rate will be higher than at 20. For character sheets, where fine detail matters, start
  conservative: threshold 0.05-0.1, start 0.2-0.3, end 0.9. That probably gives **about 1.3-1.5x (estimate)**. No install
  needed. It works with CFG.
- **MagCache** (Apache-2.0): the official repo reports 1.5x "without visual quality degradation" for Qwen-Image-Edit in
  diffusers on an A800 (about 120 s -> 80 s, version not stated). The ComfyUI-MagCache node lists **Qwen-Image only**
  (1.75x, thresh 0.10, K 2, retention 0.2). Whether it works on Edit-2511 with reference latents is **unverified**.
- **TeaCache:** the ComfyUI-TeaCache node has no Qwen-Image entry. **FBCache (WaveSpeed):** no Qwen support found.
  **Cache-DiT** (in SGLang, claims up to 7.4x) is outside ComfyUI.
- Qwen-Image-Edit-Causal (lightx2v, Apache-2.0, 2026-02-15): block-causal attention so the reference tokens can be
  cached, but it is a **4-step distilled** model and diffusers only. H100, 1024², 3 references: 93.6 s (2511, 80 NFE)
  -> 2.1 s. Its own examples show 5 of 6 edits passing. It falls under the "distilled" refusal.

## 5. CFG cost

- The official setting is `true_cfg_scale=4.0`, negative `" "`, 40 steps (model card). Qwen-Image-Edit is NOT
  guidance-distilled, so cfg 1 everywhere loses prompt adherence. The negative pass is needed, but not on every step.
- **Core `CFGOverride` node** (our ComfyUI 0.42.1, `comfy_extras/nodes_custom_sampler.py`): it sets the guider's cfg
  over a [start, end] percent range. Inside that range `sampling_function` sees `cond_scale == 1.0` and sets
  `uncond_ = None`, so **the negative pass is skipped**. CFGNorm and euler do not disable this. Example: cfg 1.0 from
  0.65 to 1.0 removes 14 of 80 passes, **about 1.2x**. Insert it between the model chain and the KSampler.
- Research basis: "Applying guidance in a limited interval" (Kynkäänniemi et al., NeurIPS 2024, arXiv 2404.07724):
  guidance mostly matters in the middle of the noise range. "Adaptive Guidance" (Castillo et al., arXiv 2312.12487)
  drops CFG in later steps with a small quality loss. Neither tests Qwen. Only the late-step cut is safe to try first.
  An early cut may weaken the edit.
- Custom alternatives: KJNodes ScheduledCFGGuidance (GPL-3.0), Extraltodeus pre-CFG and AutomaticCFG (skip uncond).
  We do not need them.
- Batching cond+uncond: ComfyUI batches them when memory allows. It saves weight transfers but not FLOPs, so there is
  no gain on our compute-bound run. **APG** and **CFGZeroStar** (core) change quality, not cost.

## 6. Fewer steps without distillation

- Official: 40 steps, euler-type flow sampler. The ComfyUI template uses euler/simple, shift 3.1 and CFGNorm (ours too).
- Community: one Civitai author ("Qwen Edit 2511: max quality outputs") found no sampler/scheduler that beat
  euler/simple. They used 20 steps and said 30 steps were sharper. MyAIForce saw composition and lighting changes with
  er_sde + ddim_uniform and euler + FlowMatchEulerDiscrete, with no clear winner. res_multistep with sgm_uniform/beta
  was "working well" on Qwen-Image (anecdote).
- No controlled 20/30/40-step A/B for 2511 was found. Test 30 steps euler/simple (1.33x) first, then 25.

## 7. Lightning / distilled LoRAs (reported honestly)

- lightx2v/Qwen-Image-Edit-2511-Lightning (Apache-2.0). Files: 4-step and **8-step** LoRAs (bf16 and fp32, about 850 MB
  bf16), plus fp8 models with the LoRA fused (4-step, 8-step, and a ComfyUI 4-step). Use cfg 1.
- Cost: 4 steps x 1 pass = 4 passes against 80 today (20x fewer). "About 10x faster" (model card). On an H100, 1024², 1 ref:
  37.0 s -> 1.85 s (ModelTC). On our 3070 an 8-step edit would take **about 1.5-2.5 min (estimate)**, a 4-step edit
  about 1 min.
- Quality: the card says quality is kept but publishes no numbers. A community config rates 8-step as having "better
  detail" than 4-step. MyAIForce: bf16+LoRA vs fp8-Lightning were close; Lightning needs the reference-method nodes
  (smooth skin without them). Lightning breaks with some samplers (res_2s/beta57 with the 8-step LoRA). Nunchaku int4 +
  Lightning-8 metrics (PSNR about 20-21 dB vs a bf16 50-step reference) show it is clearly not the same image.
- Our graph already has the 4-step LoRA node. The user refused it for quality. If ever reconsidered: the 8-step LoRA
  on the int8 file, blind A/B against 40 steps.

## 8. Sequence length (the important one for our graph)

- **Yes, every image adds tokens and costs time.** Each reference becomes about 4,100 tokens at about 1 MP (the node
  resizes it to about 1 MP regardless of its input size), plus about 190 VL tokens. Attention grows with the square of the
  sequence.
- **Measured** (ModelTC/Qwen-Image-Edit-Causal README, H100, SDPA, 1024², 2511, 80 NFE): 1 ref 37.0 s, **2 refs 63.3 s
  (+71%)**, 3 refs 93.6 s.
- **Our graph:** with no reference we still send Picture 1 twice. Dropping `image2` (and the "Picture 2" wording) when
  there is no reference should give **about 1.7x** (estimate, matches the H100 ratio). The official examples send only the
  images you have. The output will differ from today's, so the quality must be A/B'd. It may even improve, because the
  model no longer gets a duplicate.
- **When there IS a reference** (a tight crop of one detail): its 1 MP latent is mostly upscaled pixels. To control
  its size, take the VAE off `TextEncodeQwenImageEditPlus` (it then adds no reference latents; the VL part and "Picture N"
  labels stay) and add core `ReferenceLatent` nodes yourself: `VAEEncode(Picture 1 at about 1 MP)` and
  `VAEEncode(Picture 2 at about 0.25-0.4 MP)`, on BOTH the positive and the negative, then the reference-method node as
  today. **Estimate:** 0.3 MP for Picture 2 gives about 9.3k tokens instead of 12.2k, **about 1.4x**. Different reference sizes
  per image are supported by the index method, but the quality at a small reference size is **unverified**.
- `ReferenceLatent` itself costs nothing; the cost is the tokens it adds. The VAE encodes and the VL encode are
  seconds (our log: the whole prompt takes only 10-30 s longer than the sampler).
- The output size counts the same way: 832x1248 (about 1 MP) is what we render now, because the latent comes from
  `FluxKontextImageScale`. Going to 1024x1536 would add about 50% tokens.

## 9. torch.compile

- The diffusers QwenImage docs: about 2.4x for the transformer on an A100 80 GB (4.70 s -> 1.93 s) with torch.compile
  (with about 7 s of compile time). That is diffusers, not ComfyUI.
- ComfyUI: core `TorchCompileModel`; KJNodes `TorchCompileModelQwenImage` (GPL-3.0). No Qwen measurement in ComfyUI was
  found. Known problems: recompiles with LoRAs (ComfyUI #9289, recompile limit 64 with a Qwen Lightning LoRA) and with
  changing image sizes (diffusers #15027). Our sizes change per character, and we swap LoRAs (AnyPose, angles).
- We already have triton-windows 3.7.1 in the venv. Expected on the 3070: **1.05-1.2x, unverified**. Whether compile
  works together with the int8 and convrot layouts is **unverified** (comfy-quant-bench needed "compile repairs" for W4A4).
- Linux pods: compile is more reliable. Try it last, as a separate test.

## 10. fp8, int8 and NVFP4 on RunPod GPUs

- Our fp8mixed file is EMULATED on the 3070. On Ada (sm_89), Hopper and Blackwell, ComfyUI's mixed-precision ops should
  list float8 as native and use fp8 matmul automatically. Check the "Native ops" log line on the pod. The old
  `--fast fp8_matrix_mult` flag is for plain fp8 files. No Qwen measurement of it was found.
- Measured, RTX 5090, ComfyUI, 832x1280, 4 steps, cfg 1 (ANXS1 NVFP4 card): **bf16 1.62 s/it, fp8mixed 1.02 s/it
  (1.6x), NVFP4 0.40 s/it (2.5x more)**. Calibration is "rough", quality "seems okay".
- Measured, B200, diffusers + torchao + compile, Qwen-Image 1024², 50 steps, cfg 4: bf16 7.45 s, MXFP8 1.16x, NVFP4
  1.39x. **LPIPS vs bf16: MXFP8 0.34, NVFP4 0.41.** The authors say QwenImage is "more sensitive to quantization than
  Flux" (Flux MXFP8: 0.11). PyTorch blog, 2026-04-08. This argues for bf16 or int8-ConvRot when quality matters.
- Comfy blog (2026-01-09): NVFP4 about 2x over fp8/bf16 on RTX 50-series. It needs torch cu130, otherwise it can be 2x
  SLOWER.
- int8 vs fp8: on a 4090 or 5090 int8 is about 4x bf16 (per comfyanonymous). On L40S and H100 int8 and fp8 run at the
  same rate. The quality ranking favours INT8 ConvRot over fp8.

## 11. Memory and offload

- Our log: "Using async weight offloading with 2 streams", "QwenImage prepared for dynamic VRAM loading. 19582MB
  Staged", `--vram-headroom 1 --fast-disk`. Dynamic VRAM became the default around March 2026 (Comfy blog
  2026-03-25). Async offload and pinned memory have been on by default since December 2025. The Comfy blog
  (2026-01-09) says they gave 10-50% on low-VRAM cards including the RTX 3070, on PCIe 4.0 x16.
- Our estimate (section A): transfer is 5-10% of a step and mostly overlapped. **A 24 GB+ card removes it, but the gain
  from that alone is small.** The big gain on RunPod is the faster GPU.
- Fit: bf16 2511 = 40.9 GB, so it needs a 48 GB card (L40S, tight with activations and the text encoder unloaded) or
  80 GB (A100/H100). A 4090 (24 GB) holds the 20.5 GB fp8 or int8 file only barely (expect partial offload with two
  references). A 5090 (32 GB) holds fp8 or int8 comfortably.
- Known risk: Dynamic VRAM crash reports after the Aug 2026 update (#15255; workarounds `--disable-dynamic-vram`,
  `--disable-pinned-memory`).

## 12. Multi-GPU and batching on RunPod

- Per-image parallelism: xDiT (Ulysses/Ring/PipeFusion/CFG-parallel) has no ComfyUI integration that we would use. Its
  paper shows weak scaling at 1024 px. CFG-parallel is about 1.8x on 2 GPUs (vLLM-Omni docs, Qwen-Image-Edit example).
  SGLang supports 2511 with `--cfg-parallel-size` and sequence parallelism. All of these are outside ComfyUI.
- Better for us: **one edit per GPU** (several pods or several ComfyUI instances on a multi-GPU pod), fed by our queue.
  The 8 front candidates and per-view edits are independent jobs.
- Batching several seeds in one run gives little on a compute-bound GPU (B200 Qwen-Image: batch 4 took 3.6x the time
  of batch 1).
- Reference speeds for planning (other people's measurements; scale to our token count):
  - H100, diffusers bf16, 40 steps cfg 4, 1024²: 1 ref 37.0 s, 2 refs 63.3 s (ModelTC). About 11x our 3070.
  - B200, SGLang, single 2511 edit: 35.3 s mean, about 48 GB peak (steps and size not stated: **unclear**).
  - RTX 5090 (ComfyUI, cfg 1, 1 ref, 832x1280): bf16 1.62 s/it, fp8 1.02 s/it. With our 2 refs and cfg 3, about 3.5x
    more work per step, so **estimate** about 3.6 s/step fp8, about 2.4 min per 40-step edit.
  - L40S (bf16 362 TFLOPS dense peak vs H100 989): **estimate** 2-3x slower than an H100. A100: similar to L40S for bf16,
    no fp8.

## Ranked table

Expected speedups are per edit, relative to today's graph on the same GPU. "Est." = our estimate from the FLOP model
or other people's measurements; none was measured on our setup yet.

| # | Method | 3070 (est.) | RunPod 4090 / L40S / H100 (est.) | Quality risk | Effort | Licence |
|---|---|---|---|---|---|---|
| 1 | Drop the duplicate Picture 2 when there is no reference | **1.7x** (H100 measured 1.71x) | 1.7x | Low-medium (output changes; official usage) | Small (graph/code) | ours |
| 2 | Comfy-Org `qwen_image_edit_2511_int8_convrot` | 1.3-1.6x | 4090 1.3-2x; L40S/H100 about the same as fp8, 1.3-1.6x over bf16 | Low (ranked closer to bf16 than fp8) | Small (20.5 GB download, file name) | Apache-2.0 (weights); ComfyUI core GPL-3.0 |
| 3 | CFGOverride cfg 1.0 for the last ~35% of steps | 1.2x | 1.2x | Low-medium | Tiny (core node) | core |
| 4 | Smaller Picture 2 reference (own ReferenceLatent, about 0.3 MP) | 1.3-1.4x (ref edits only) | same | Medium (reference detail) | Small-medium | core |
| 5 | SageAttention 2.2 `qk_int8_pv_fp16_cuda`, fp32 accum, Qwen only | 1.2-1.3x | 4090 1.2-1.4x; H100 sm90 kernel 1.2-1.4x | Medium (black output risk; check every image) | Medium (wheel + small node) | Apache-2.0 (verify fork) |
| 6 | 30 steps instead of 40 (euler/simple) | 1.33x | 1.33x | Low-medium | Tiny | - |
| 7 | EasyCache (conservative) | 1.3-1.5x | same | Medium (fine detail) | Tiny (core) | core |
| 8 | torch.compile | 1.05-1.2x (unverified) | 1.1-1.3x on Linux (unverified) | Low | Medium (recompiles, LoRA) | core / triton MIT |
| 9 | fp8 native on Ada/Hopper (pod only) | n/a (emulated) | 1.3-1.6x over bf16 | Medium for Qwen (MXFP8 LPIPS 0.34 on B200) | Tiny | Apache-2.0 |
| 10 | NVFP4 (Blackwell only) | n/a | 5090: about 2.5x over fp8 | High (LPIPS 0.41) | Small | Apache-2.0 (community files) |
| 11 | Lightning 8-step LoRA (distilled) | 5-8x | 5-10x | High (user refused) | Tiny (node exists) | Apache-2.0 |
| 12 | GGUF Q8/Q6 | 0.9-1.0x (no gain) | no gain | Low (Q8) | Small | Apache-2.0 |
| 13 | Nunchaku int4/FP4 (community 2511) | uncertain | not on H100 | High (PSNR ~19-21 dB) | Medium | Apache-2.0 |
| 14 | W4A4 convrot / W4A8 | W4A8 ≈ int8 speed | - | W4A4 broke 2511; W4A8 ok but less detail | Medium (make our own) | MIT tools |

Rough combined estimate for the 3070 without distillation: #1 x #2 x #3 x #5 ≈ 1.7 x 1.4 x 1.2 x 1.2 ≈ 3.4x, so
12 min -> about 3.5 min for an edit without a reference. With #6 as well, about 2.7 min. These factors do not multiply
exactly, because they shrink different parts of a step.

## Recommended test order (one change each, same seed, timed, checked by eye)

Use one fixed test set: 2 edits without a reference, 1 with a reference, 1 AnyPose repose, 1 angle view. Record the
s/it and the total, and keep the images side by side with the baseline. Judge blind where possible. Check side,
fingers, colours and the black background as in AGENTS.md 4.3-4.5.

1. **Baseline**: today's graph, 40 steps cfg 3, fp8mixed. Record s/it and `nvidia-smi` utilisation (confirm
   compute-bound).
2. **No duplicate Picture 2** (edits without a reference only). Expect about 1.7x. Quality A/B against the baseline.
3. **int8_convrot file** (Comfy-Org). Expect 1.3-1.6x. A/B, including one run with each LoRA we use (AnyPose,
   Multiple-Angles). The int8 LoRA path was fixed on 2026-06-26, but check it.
4. **CFGOverride** cfg 1.0 at 0.65-1.0. Expect about 1.2x. If clean, try 0.5-1.0.
5. **SageAttention 2.2 fp16_cuda + fp32 accumulation, Qwen only.** Abort on any black or NaN image. Expect 1.2-1.3x.
6. **Smaller Picture 2 reference** (0.3-0.4 MP via ReferenceLatent) for reference edits.
7. **30 steps** (then 25) with euler/simple.
8. **EasyCache** conservative (0.05-0.1 / 0.25 / 0.9), only if more speed is still wanted.
9. On the first RunPod pod: repeat the baseline in bf16, then fp8 native and int8 on the same seeds, and read the
   "Native ops" line. Then torch.compile on Linux.

Each winning change goes into AGENTS.md section 4.3 with its measured numbers (short, with the evidence).

## Sources

- Qwen-Image-Edit-2511 model card (40 steps, true_cfg 4, Apache-2.0): https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- Comfy-Org 2511 files (int8_convrot, fp8mixed, bf16 sizes): https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI (API tree `split_files/diffusion_models`)
- ComfyUI PR #14636 int8 (2026-06-25): https://github.com/Comfy-Org/ComfyUI/pull/14636
- ComfyUI issue #14642 / PR #14650 int8 LoRA fix: https://github.com/Comfy-Org/ComfyUI/issues/14642
- ComfyUI PR #14859 convrot int4 (2026-07-09): https://github.com/Comfy-Org/ComfyUI/pull/14859
- ComfyUI issue #14824 int8 on A100/A6000, the "4x on 30/40/50 series" comment: https://github.com/Comfy-Org/ComfyUI/issues/14824
- wazimondo int8 tensorwise 2511: https://huggingface.co/wazimondo/Qwen-Image-Edit-2511-int8-tensorwise
- cardamonnl int8 convrot 2511: https://huggingface.co/cardamonnl/Qwen-Image-Edit-2511-int8-convrot
- LAXMAYDAY int4 mixed 2511 (L4 timings, PSNR): https://huggingface.co/LAXMAYDAY/Qwen-Image-Edit-2511-int4-tensorwise-mixed
- obsxrver INT8 ConvRot (quality ranking, ctq command): https://huggingface.co/obsxrver/ComfyUI-Native-INT8_ConvRot
- bertbobson Z-Image INT8 (3090 claim): https://huggingface.co/bertbobson/Z-Image-Turbo-INT8-Tensorwise
- comfy-quant-bench (MIT; Qwen 2511 W4A4 vs W4A8): https://github.com/JoaoZaokk/comfy-quant-bench
- SparknightLLC QuantizationToolkit (AGPL-3.0): https://github.com/SparknightLLC/ComfyUI-QuantizationToolkit
- Comfy-Org comfy-quants (GPL-3.0): https://github.com/Comfy-Org/comfy-quants
- Nunchaku: https://github.com/nunchaku-ai/nunchaku ; ComfyUI-nunchaku: https://github.com/nunchaku-ai/ComfyUI-nunchaku ; issue #858: https://github.com/nunchaku-ai/nunchaku/issues/858
- QuantFunc 2511: https://huggingface.co/QuantFunc/Nunchaku-Qwen-Image-EDIT-2511 ; int4 metrics: https://huggingface.co/wazimondo/Qwen_image_edit_2511_int4
- HF blog Nunchaku in diffusers (2026-07-23): https://huggingface.co/blog/nunchaku-diffusers
- Unsloth GGUF 2511: https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF
- MyAIForce 2511 deep dive (2026-01-02): https://myaiforce.com/qie-2511/
- Civitai "Qwen Edit 2511: max quality outputs" (via search summary; page blocked): https://civitai.com/models/2659067
- ComfyUI #9773 (Sage black on sm86, fp16_cuda fix): https://github.com/Comfy-Org/ComfyUI/issues/9773 ; #9184, #9223
- woct0rdho SageAttention fork: https://github.com/woct0rdho/SageAttention
- KJNodes Patch Sage Attention docs: https://comfyui-wiki.com/en/custom-nodes/ComfyUI-KJNodes/nodes/pathch-sage-attention-kj
- ComfyUI discussion #11583 (5090 Qwen Edit 14m30 -> 9m30): https://github.com/Comfy-Org/ComfyUI/discussions/11583
- SageAttention papers: https://arxiv.org/abs/2410.02367 , https://arxiv.org/abs/2505.11594 ; FA3: https://arxiv.org/abs/2407.08608
- musubi-tuner #874 (FA3 timings): https://github.com/kohya-ss/musubi-tuner/issues/874
- EasyCache/LazyCache PR #9496: https://github.com/Comfy-Org/ComfyUI/pull/9496
- MagCache: https://github.com/Zehong-Ma/MagCache ; ComfyUI-MagCache: https://github.com/Zehong-Ma/ComfyUI-MagCache
- ComfyUI-TeaCache: https://github.com/welltop-cn/ComfyUI-TeaCache
- Qwen-Image-Edit-Causal (H100 1/2/3-ref timings): https://github.com/ModelTC/Qwen-Image-Edit-Causal
- Lightning 2511 (file list via HF API): https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning
- ANXS1 NVFP4 2511 (5090 timings): https://huggingface.co/ANXS1/Qwen-Image-Edit-2511-NVFP4
- PyTorch blog MXFP8/NVFP4 on B200 (2026-04-08): https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/
- Comfy blog NVFP4/async offload/pinned memory (2026-01-09): https://blog.comfy.org/p/new-comfyui-optimizations-for-nvidia
- Comfy blog Dynamic VRAM (2026-03-25): https://blog.comfy.org/p/dynamic-vram-in-comfyui-saving-local ; crash issue #15255
- Diffusers QwenImage docs (torch.compile 2.4x A100): https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage
- ComfyUI #9289 (compile recompile limit with Qwen LoRA): https://github.com/comfyanonymous/ComfyUI/issues/9289
- SGLang 2511 cookbook (B200, parallel options): https://lmsysorg.mintlify.app/cookbook/diffusion/Qwen-Image/Qwen-Image-Edit
- vLLM-Omni CFG parallel: https://docs.vllm.ai/projects/vllm-omni/en/stable/user_guide/diffusion/parallelism/cfg_parallel/
- xDiT paper: https://arxiv.org/abs/2411.01738
- Guidance interval: https://arxiv.org/abs/2404.07724 ; Adaptive Guidance: https://arxiv.org/abs/2312.12487
- Local evidence: `D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI` (v0.42.1): `comfy/quant_ops.py`, `comfy/ops.py`,
  `comfy/ldm/modules/attention.py`, `comfy/samplers.py`, `comfy_extras/nodes_custom_sampler.py` (CFGOverride),
  `comfy_extras/nodes_qwen.py`, `.venv/Lib/site-packages/sageattention` (1.0.6); `.cinestudio/comfyui.log`.
