# Colour consistency across chained clips: methods, evidence and what to test in cinestudio

Date 2026-10-23. A follow-up to docs/ltx_long_drift_research.md (read that first; it is not repeated here). No code was
changed. Tags: [SRC] = what a source claims, [CODE] = read in installed source, [OURS] = our own measurements
(AGENTS.md, .cinestudio/gauntlet/masked3d_e2e/RESULTS.md), [INF] = my inference.

## 0. Short answer

Our drift happens in two steps. First an **event**: from frame ~180 the warm back-light turns the black suit into
"reflective bronze" ([OURS] critic 2: "a latex sheen" where the sunset rim light hits it). Then **persistence**: the
chained frame -1 look and the relay latents carry that colour into every later clip, even when the light no longer
explains it (frame 0 vs frame 345: same angle and layout light, b* -14 vs +0.6). The published fixes fall into
three groups:

1. Latent statistics toward a **fixed first-chunk** reference (Lightricks' AdaIN, the diffusers LTX long pipeline,
   LAMP, FreqForcing).
2. A **fixed anchor** in attention (StreamingT2V, Rolling Forcing, ID-Forcing).
3. Pixel colour matching after decoding (KJNodes ColorMatch / MKL: the community's default).

Each of our failed tries broke one of two rules. Rule 1: the correction must be *local to the character*, because the
background changes with the view in an orbit. Our whole-frame Reinhard and the global per-channel AdaIN fail here.
Rule 2: frame -1 and the relay must *agree*. chain_colour look corrected only the look, and gave the two-tone seam. The
most promising next tests are: (1) **masked AdaIN** on the character tokens toward clip 1's character statistics,
ideally in the early steps; (2) a check of the **VAE round-trip bias** of the chained look (seconds of GPU); (3) an
**MKL transfer on the character only**, applied to the chained look AND, as a latent delta, to the relay latents;
(4) a weaker chained conditioning (look and relay strength); (5) light the layout so the event never happens.

## 1. Latent-space methods

**AdaIN / latent normalisation.**
- [CODE] `LTXVAdainLatent` (ComfyUI-LTXVideo `latent_norm.py`): a per-channel mean/std over the WHOLE C×F×H×W tensor
  (or per frame), `lerp` by factor. It has no mask, and it also touches appended guide frames when it runs before the
  crop. We call it after each stage's guide crop, so the relay saves the corrected latent ([CODE] runner.add_adain).
- [SRC] Lightricks' looping sampler: AdaIN of each new tile toward the first tile's output, 0.1-0.3, "Color Shifts: use
  optional_normalizing_latents and/or increase adain_factor".
- [SRC] The merged diffusers `LTXI2VLongMultiPromptPipeline` (PR #12614, LTX-Video 0.9.8, Jan 2026) uses "AdaIN
  cross-window normalization" at `adain_factor=0.25` in the base pass and 0 in the refine pass.
- [SRC] LAMP (arXiv 2310.10769): AdaIN toward the first frame only in the early denoising steps ("applying it
  throughout produced artifacts").
- [OURS] Global AdaIN 0.3 toward clip 1's latent gave the best joins (x 0.93-1.61) and the lowest background hue drift.
  The suit got only a little darker (b* end -2.4 vs +0.6). At 1.0 toward the look, saturation rose clip by clip and the
  mask turned into a black veil.
- [INF] Why it is weak: the per-channel statistics are dominated by the background (the character covers maybe
  15-25% of the tokens). Pulling the whole frame toward clip 1 also fights legitimate view changes: in an orbit the
  back view shows different buildings and sky.

**Early-step low-frequency anchoring.**
- [SRC] FreqForcing (arXiv 2607.27110, training-free, on Self-Forcing): it treats drift as "low-frequency energy
  drift". The low spatio-temporal frequencies of the attention output are pulled toward an anchor branch built from
  frames inside the trained horizon (λ 0.6, σ 0.125), in the first two denoising steps only.
- [SRC] Ablation at 60 s: without this anchoring, overall consistency 20.94 -> 20.26 and background consistency
  96.47 -> 96.19. The paper gives no colour-drift number. Code at github.com/jiatongli2024/FreqForcing, licence not
  stated.
- [INF] The transferable idea: correct **low frequencies, early**, and let the later steps redraw the detail. A
  post-hoc shift like ours instead flattens detail, which is our "veil" and "soft" result.
- [CODE] The pack's `LTXVNormalizingSampler` (easy_samplers.py l.916) already splits the sigmas, runs
  SamplerCustomAdvanced per chunk and changes the latent between chunks. It is a working pattern for "intervene after
  step k" on LTXAV models.

**Noise reuse / noise initialisation.**
- [SRC] FreeNoise (arXiv 2310.15169, ICLR 2024): it reschedules (shuffles) one noise sequence across windows and
  fuses window attention. About 17% extra time, against 255% for the previous best.
- [SRC] ConsistI2V (arXiv 2402.04324): noise initialisation "from the low-frequency band of the first frame" for
  layout consistency.
- [INF] For us: start the clip k latent from noise whose low frequencies come from clip 1's character-region latent
  (or from the relay). This is a palette prior at sigma 1. It is untested on 8-step distilled schedules, where the
  first step decides almost everything, so the effect could be large or ruin the layout.

**Overlapping-window co-denoising.**
- [SRC] MultiDiffusion (arXiv 2302.08113), Gen-L-Video (arXiv 2305.18264, "without additional training", temporal
  co-denoising: overlapping windows averaged at every step), FIFO-Diffusion (arXiv 2405.11473, NeurIPS 2024, diagonal
  denoising queue), Ouroboros-Diffusion (arXiv 2501.09019, AAAI 2025; FIFO + subject-aware cross-frame attention, VBench
  subject-consistency gains).
- [SRC] ComfyUI core has generic Context Windows nodes (`ContextWindowsManual`, `dim`, fuse PYRAMID; a WAN variant
  with pyramid/gaussian/average/overlap).
- [INF] This is structurally the right answer: no chain, so there is nothing to accumulate. But our graphs append
  IC-LoRA guide tokens (layout, look, sheet, keyframes) along the frame axis, and the window slicing would cut them
  wrongly. The cost per step equals sequential generation. The engineering risk is high; it is not a near-term test.

**Latent blending at joins.**
- [CODE] Lightricks' LinearOverlapLatentTransition is a linear latent cross-fade.
- [SRC] LTX-2's chunk pipeline uses a 25-frame carry plus a decoded cross-fade, the same as our relay plus flow blend.
- [INF] Join blending hides a step at the seam. It does not remove a slow drift.

**The VAE round trip.**
- [SRC, single community source] H3 Delta Color Carry (ComfyUI-MAINodes, GPL-3.0, alpha, for the MiniMax H3 VAE)
  claims "each round trip darkens the picture by about 2.4%" and that the error accumulates over links.
- [SRC] The same node corrects in latent space: it encodes the original and the corrected frames and adds only
  E(corrected) − E(original), low-passed and tapered over the carried frames, so the encoder bias cancels.
- [SRC] LatentTailContinuation-style nodes avoid the pixel→PNG→VAE path for the same reason.
- [INF] Our chained look takes exactly this path: decoded frame 96 -> unsharp mask -> PNG -> VAEEncode. Nobody has
  measured the bias of the LTX-2.5 VAE. This is the cheapest thing to check.

## 2. Conditioning methods

- [SRC] **The fixed first-chunk anchor wins everywhere in the literature.**
  - StreamingT2V (arXiv 2403.14773, CVPR 2025): the Appearance Preservation Module takes "a fixed anchor frame" from
    the first chunk, because conditioning only on the last frames "loses early object and scene details".
  - Rolling Forcing (2509.25161): without its attention sink, drift 0.01 -> 4.63.
  - ID-Forcing (2610.03120): training-free; the first chunk is a permanent KV sink, with a hue-histogram colour-drift
    metric.
  - LongLive (2509.22622) and Self-Forcing (2506.08009) are trained.
- [SRC] SVI (arXiv 2510.09212, ICLR 2026 oral) says that anchoring and noise-schedule tricks only *reduce* error and
  tend to give homogeneous scenes and repetitive motion. Its own fix (error-recycling fine-tuning) needs training.
- [OURS] A FIXED anchor in our only available slot (frame -1, a full image) brought the wrong background: the front
  skyline in back views. LTX reads the frame -1 image as "palette, lighting, AND materials" of the whole scene (the
  Layout-To-Render card). [INF] We have no attention-sink slot that carries *only* the character. The Ingredients sheet
  is the nearest one, and 1.3 moved b* by ~1 point.
- [SRC] SkyReels-V2 README: `--addnoise_condition 20` (≤ 50) "adding some noise to the clean condition" improves
  long-video consistency; too much noise causes inconsistency. [SRC] Lightricks' default pin strength for the carried
  latents is 0.5. We use 1.0.
- [OURS] Keyframes are weak against frame -1. The fresh Qwen look per clip fixed the colour (b* -8/-9) but gave a colour
  jump at the join and a step, at ~11 min of GPU per clip.

## 3. Pixel-space post-processing

- **Global transfer.** Reinhard (mean/std in Lab), histogram matching, MKL (Pitié: a linear map that matches the full
  3×3 colour covariance) and MVGD.
  - [SRC] KJNodes `ColorMatch` uses the `color-matcher` library. **Both are GPL-3.0** (checked on GitHub).
  - [SRC] The community uses MKL to the first frame for Wan-VACE extensions, "not always applicable if the scene
    changes a lot". VACE workflow authors report that colour match "helps"; others spread the shift by frame
    interpolation, or use a white instead of a grey mask.
  - [INF] MKL and Reinhard are short formulas. Write them in numpy (BSD) or use scikit-image `match_histograms`
    (BSD-3) and OpenCV (Apache-2.0). Do not import the GPL code.
- **Why our whole-frame match hurt** [INF]: in an orbit the frame's content changes with the view. A whole-frame match
  to clip 1 frame 0 recolours the sky and buildings to front-view statistics, which is the higher frame hue drift we
  measured.
- **Why the dark-pixel a*/b* shift hurt** [INF]: one constant offset, applied to every dark pixel, also moved the
  plum jacket's dark folds, hence the blue jacket. A covariance (MKL) map on the character region maps bronze→black and
  magenta→plum *jointly*.
- **Masked / per-region matching** [INF]: our run has per-frame Blender character masks, and SAM 2.1 (Apache-2.0) is
  installed. Blender could also render a per-part ID pass of the mannequin (torso/legs = suit, upper body = jacket,
  feet = shoes), so each part can be matched on its own; the layout keeps the parts roughly in place.
- **Ramps** [INF]: match clip k to clip k-1 on the 25 shared frames (character mask, MKL fitted on all 25 pairs), then
  apply it at full strength at frame 0 and keep it, or fade it out. This removes the colour part of the join step.
  It is cosmetic only: the chain does not see it.
- **3D LUT from the overlap** [INF]: a 17³ LUT fitted by least squares on the shared frames' pixel pairs. It is more
  flexible than MKL, but it overfits where geometry differs (the shared frames differ ~4x the normal frame change:
  they are not pixel-aligned). Use MKL first.
- **Deflicker.**
  - [SRC] All-In-One-Deflicker (Lei et al., arXiv 2303.08120, CVPR 2023): **Apache-2.0**, about 3 GB VRAM. It
    optimises a neural atlas per video (10k iterations for 80 frames), with a recommended maximum of 200 frames.
  - [SRC] Learning Blind Video Temporal Consistency (Lai et al., arXiv 1808.00449): MIT, but the repo also bundles a
    CC BY-NC part (EdgeConnect). Check which files are used.
  - [SRC] Deep Video Prior (arXiv 2010.11838): per-video training, licence not found.
  - [INF] These remove frame-to-frame flicker, not a slow drift over 10 s. They would also average away intended
    light changes. Not a fit.

## 4. What fits our limits

| Method | 8 GB / distilled 8-step / IC-LoRA | Licence | Fit |
|---|---|---|---|
| Global AdaIN to clip 1 (now) | yes | LTX-2 Community (pack) | keep; weak on the costume |
| Masked AdaIN, post or early-step | yes; a small custom node (no stock mask input) | own code | **high** |
| VAE round-trip check / latent-delta correction | yes, seconds | own code (H3 idea, GPL code not used) | **high, cheap** |
| Character-only MKL on chained look + relay delta | yes, CPU + 2 encodes | numpy/OpenCV/SAM2 | **high** |
| Lower look / relay strength, noised condition | yes, existing knobs | — | medium, zero code |
| Low-frequency noise init (FrameInit-like) | yes, custom noise node | own code | medium-low, unpredictable on 8 steps |
| Context-window co-denoising | conflicts with appended guides | ComfyUI GPL-3.0 core | low now |
| Trained fixes (Self/Rolling Forcing, SVI, FramePack) | other backbones, training | various | no |
| KJNodes ColorMatch / color-matcher | yes | **GPL-3.0** | reimplement instead |
| Deflicker nets | 3 GB, per-video | Apache-2.0 / MIT(+NC part) | no (wrong problem) |

Measure every A/B with our four numbers: dark-costume Lab b* (late / end), the shared-frame difference x median frame
change (plus the absolute value), sharpness vs clip 1, and a blind critic. Add ID-Forcing's colour drift (L1 between
180-bin hue histograms) on the **character mask only**, clip k's last frame vs clip 1, as a fifth number. Every
experiment below forks the e2e run 1 (r20261022-080844-d158) from clip 2 in preview, with the same seed, and compares
against the B-only fork (r20261022-144214-58fa, the current default).

## 5. Ranked shortlist

**1. Masked AdaIN on the character tokens (latent, toward clip 1's character statistics).**
- Change: a custom node "LTXVMaskedAdain". It takes the latent plus a mask from the Blender character mask, dilated
  ~10% and resized to the latent grid (1/32 spatial, 1+8k temporal). It computes the per-channel mean/std over the
  masked tokens of clip 1's saved latent (the reference) and over the masked tokens of clip k, applies the AdaIN inside
  the mask only, and feathers the edge.
- Variants:
  - (a) after the stage, like now, factor 0.5 and 1.0;
  - (b) early-step (LAMP/FreqForcing): split the sigmas after step 2 of 8, as LTXVNormalizingSampler does, apply the
    AdaIN to the intermediate latent's real frames only (not the appended guides), then resume.
- Reason: B already showed that AdaIN toward clip 1 helps joins and background. Its weakness is that the background
  dominates the statistics, and the mask removes exactly that.
- Risk: latent channels are not separable by colour; a 24×14 mask is coarse and can give a halo; a mean shift in a
  small region can give a veil again. Variant (b) needs care with the guide tokens.
- Pass: b* end ≤ -8, character hue drift below B's 1.61, joins ≤ B's, no veil for the critic.

**2. VAE round-trip bias of the chained look (diagnostic, then the fix).**
- Change: take clip 1 frames 0 and 96. VAEEncode->decode them 1-5 times with the LTX-2.5 video VAE, with and without
  `chain_sharpen`, and measure the costume b*, L and saturation per cycle. Seconds of GPU, but run it only when the
  queue is empty.
- If there is a bias, give the chained look as the saved **latent** frame of clip k-1 (frame 96 = latent index 12; a
  LoadLatent + LTXVSelectLatents guide at -1), or add the H3-style delta E(sharpened) − E(decoded) so only the
  sharpening survives.
- Reason: each clip adds one round trip. This is a drift source that exists independently of the model, and nobody
  has measured it for LTX.
- Risk: there may be no measurable bias. In that case it costs minutes.

**3. Character-only MKL on the chained look, plus the same correction on the relay latents.**
- Change: a numpy MKL in Lab, fitted from the chained frame's character pixels (SAM 2.1 mask seeded by the Blender
  mask, or the dilated Blender mask) to clip 1's character pixels. Pool all of clip 1, or take the frame with the
  nearest body yaw; for back views, the sheet's back panel at a lower strength. Strength 0.6-0.8, feathered mask.
- Then decode clip k-1's last 25 frames, apply the same map, and add E(corrected) − E(original) to the saved relay
  latents, low-passed. Frame -1 and the relay then agree.
- Optional variant: per-part matching from a Blender part-ID pass (suit / jacket / shoes).
- Reason: it repairs both causes of chain_colour look's failure (a constant offset on dark pixels; look and relay that
  disagree), and it keeps the background (the reason chain_colour match failed). Pose and place stay as they are, so
  no step like the fresh Qwen look.
- Risk: the garment mix changes with the view and biases the fit; lit vs flat-lit colour; the delta adds 2-3 encodes
  per clip.

**4. Weaker chained conditioning (zero code).**
- Change: `--relay-strength 0.5` (Lightricks' default pin; SkyReels' noised condition), and the chained frame -1
  look at 0.7 (expose the strength if it is not already a knob). The Ingredients sheet and the clip 1 AdaIN then weigh
  more.
- Reason: the chained frames decide the colour ([OURS]: the sheet at 1.3 did not), so a weaker carrier lets less of the
  error through.
- Risk: worse joins (watch `overlap`), and the background may wander.

**5. Remove the trigger: light the layout (zero code, panorama/Blender side).**
- Change: turn the panorama so the sun is behind the camera for the frames 160-260 part of the orbit, or give the
  layout render a neutral front fill; same seed, whole run.
- Reason: the drift starts exactly where the warm back-light hits the suit, and AGENTS.md already notes that "sun
  behind" gives a hazy, 3D look.
- Risk: it changes the scene, not the method. Other shots will have other triggers, so combine it with 1 or 3.

Cosmetic fallback for deliveries (not the chain): the per-join character-masked MKL ramp from §3 on the decoded
clips.

## 6. Brightness across chained clips (added 2026-10-23)

**Our data.** The coordinator measured 3 runs: still camera, 3 clips, look_chain prev, relay on, AdaIN 0.3 to clip 1.
- The mean L is flat inside each clip and steps down at each join: run 1 55.7 -> 54.1 -> 52.6, run 2 46.9 -> 44.7
  -> 43.0, run 3 36.6 -> 35.7 -> 34.9.
- Each clip comes out 1.2-1.4 L below its frame -1 look.
- The ffmpeg PNG extraction reads 0.9 L below OpenCV. On a known PNG: ffmpeg -0.5 L, OpenCV +0.5 L.

**6.1 Causes**
- **The VAE round trip.**
  - [SRC] The H3 "2.4% darkening per round trip" is **unverified**. Its own page gives no VAE, metric or frame count,
    says the strong-drift bench is "still owed", and concerns the MiniMax H3 VAE, not LTX. Treat it as a hypothesis.
  - [CODE] The LTX VAE normalises the latent per channel with `mean-of-means` / `std-of-means`
    (causal_video_autoencoder.py l.1143). Brightness is spread over 128 channels: no single channel is "brightness".
  - No LTX-specific darkening note was found. [SRC] Community reports of LTX-2.3 I2V "losing some colour in the first
    0.5 s" blame the VAE re-encode of the start image (see the earlier report).
- **Our own extraction.** [CODE] `video.frame_png` uses plain ffmpeg, no colour flags. [INF] In run 2 the chained look
  (45.9) sits 1.0 L below the clip it comes from (46.9). That is about the ffmpeg read bias, so it is not "a small part"
  but **~40% of that 2.2 L step**; the model's look -> output offset (-1.2 L) is the rest. H.264 itself adds a little
  (4:2:0, quantisation).
- **Conditioning vs output.**
  - [SRC] The Layout-To-Render card: the look image "sets palette, lighting, and materials". It is an attention guide,
    not a pixel copy.
  - [INF] The output tone is a mix of the look and the layout render at frame 0. A constant offset is expected.
    It only *compounds* because the next look is cut from the output.
  - [INF] The darkening has the same sign at L 36 and at L 56, so it is a bias, not a regression to mid-grey.
- **CFG / distillation.**
  - [SRC] Over-exposure from high CFG is fixed by CFG rescale, and the mid-grey bias by zero-terminal-SNR schedules
    (Lin et al., arXiv 2305.08891).
  - [INF] Neither applies to us: distilled LTX-2.5 runs at cfg 1, and rectified flow starts from pure noise.

**6.2 What tools and papers do**
- **Gain vs CDF.**
  - [SRC] The FFmpeg `deflicker` filter (built in; window 2-129 frames, arithmetic/geometric/median mean) normalises the
    mean luminance of each frame to a moving average.
  - [SRC] H3 Scene Color Stats uses centre-weighted luma and saturation percentiles. H3 Delta Color Carry applies a weak
    exposure/saturation correction with clamps (max 6 code values luma, 0.06 saturation, strength 0.5).
  - [INF] Exposure is a gain in linear light. A gain keeps contrast and is safe. CDF matching of L also forces the
    contrast and the content mix to match, which is only valid when the content is the same.
- **Gamma vs linear.**
  - [INF] For L* > 8, L* ≈ 116·Y^(1/3) − 16, so a gain on L is close to a linear-light gain of about gain³ on Y, but not
    equal (the −16 offset). A Y gain (or the same gain on linear RGB) is the physically right "exposure".
  - [INF] For ±5% the two differ by < 0.3 L, so this is not urgent.
- **Anchor.** [SRC] Every drift paper anchors to the first chunk (§2). [INF] Anchoring to the neighbour only smooths
  the step; it keeps the drift.
- **Latent vs pixel.**
  - [SRC] H3's latent delta: E(corrected) − E(original) is added to the carried latents, so the encoder bias cancels.
  - [INF] A per-channel latent mean shift (AdaIN) moves colour and brightness together. A pixel gain, carried into the
    latent by the delta, is the clean "brightness only" lever.
- **Both carriers.** [INF] A look corrected to the anchor and relay frames from an already-anchored clip k-1 differ only
  by the model's constant look->output offset. Clip 1 has the same gap (look vs layout), so the relay needs no
  correction once every look is anchored from clip 2 on. Check it: the L of clip k's first 25 frames vs clip k-1's
  last 25.
- **Stable region.** [INF] With a still camera, the background outside the dilated character mask is the same scene
  in every clip: measure the gain there and apply it to the whole frame. The character's light can legitimately change
  (it turns), and it covers a different area each clip.
- **Post-hoc smoothing.**
  - [SRC] ffmpeg `deflicker=size=49:mode=pm`, or a gain fitted on the shared 25 frames and ramped in, turns each step
    into a ramp. It cannot undo a global darkening. FFmpeg is LGPL/GPL by build; we already call it as a CLI.
  - [SRC] All-In-One-Deflicker (Apache-2.0; per-video atlas, ≤ 200 frames) is overkill for 1-2 L steps.

**6.3 Decoding BT.709 limited-range H.264 to sRGB PNG**
- [CODE] Local ffmpeg 8.1.1: `scale` has `in_color_matrix`, `in_range` and `in_transfer` all defaulting to `auto`
  (from the frame tags). Swscale's own flags (`accurate_rnd`, `full_chroma_int`) are set inside the filter with
  `flags=`, or with `-sws_flags` after `-i`. [SRC] FFmpeg ticket #979: `accurate_rnd` "only works if you use it after
  -i".
- Explicit and reproducible:

  ```
  ffmpeg -i clip.mp4 -vf "select='eq(n\,96)',scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int+bitexact,format=rgb24" -frames:v 1 frame.png
  ```

  Do not convert the transfer (no `out_transfer`/zscale `transfer=iec61966-2-1`). [INF] By convention, BT.709 video code
  values ARE the display values; converting the OETF to sRGB shifts the midtones.
- zscale alternative: `zscale=matrixin=709:rangein=limited:range=full,format=gbrp`, then `format=rgb24`.
- [INF] Check whether the PNG gets cICP/gAMA chunks from the bt709 tag (browsers apply gAMA; OpenCV/PIL ignore it).
- **PyAV.** [SRC] In `reformat`/`to_ndarray`, `src_colorspace=None` means the frame's own colorspace (bt709 for our
  files), but `src_color_range=None` means UNSPECIFIED. Pass the range explicitly
  (`src_color_range` MPEG, `dst_color_range` JPEG).
- **OpenCV.** [INF, not confirmed by a source] Its FFmpeg backend converts with swscale and historically ignores the
  stream's matrix (BT.601 default). The ±0.5 L pair fits a matrix/rounding difference with the truth in between.
  [SRC] OpenCV PR #27652 (2025) moved NV12->RGB to cvtColor, so results depend on the version.
- **Best option.** [INF] Skip the decode: save frame 96 from ComfyUI's decoded IMAGE tensor (an ImageFromBatch slice
  -> SaveImage) next to the clip. The chained look is then exactly what the VAE produced, with no H.264 or reader
  differences.

**6.4 The planned fix (`chain_light look`, video.match_lightness)**

Sound in principle. [INF] Matching input to input (each chained look to the run's first look) makes every clip sit
where clip 1 sits (~1.3 L below its look), so the offset stops compounding. Weak points:
- **Whole-frame mean.** The character's area and pose change the mean: use the background (dilated Blender mask
  inverted) for the gain.
- **A scene that should get darker, and a moving camera.** A fixed target fights a sunset or a pan into shadow.
  Scale the target by the layout: target = run-look L × (layout-render background L at clip k's start ÷ at frame 0).
  The ratio is 1 for a still camera, and Blender already knows the intended exposure change.
- **Highlights.** `L*gain` clipped at 255 in 8-bit OpenCV Lab, then Lab->BGR, clips the brights and can shift the hue of
  saturated brights. Work in float Lab (float32 input, L 0-100), and use a soft knee above L ≈ 85 or a linear-Y gain.
- **8-bit quantisation.** The 8-bit Lab round trip quantises by itself. Float avoids it.
- **±10% clamp.** Fine: 3 joins lost 5.6% in run 1. Log the clamp hits.
- **Relay disagreement.** Probably small (6.2), but measure it.

**6.5 Ranked tests** (fork run 2 from clip 2, same seed, preview)
1. **A lossless chained frame.** Take frame 96 from ComfyUI's image output, or use the explicit ffmpeg command above,
   instead of plain `frame_png`. Measure: chained-look L vs the clip's own frame 96 (target |Δ| < 0.2 L; now ~1.0) and
   the per-clip mean-L step. Zero risk; it removes ~40% of the step.
2. **`chain_light` with the background region, float Lab / linear-Y gain, the layout-ratio target and a soft knee**,
   on top of 1. Measure: per-clip background L (target: steps < 0.5 L, clip 3 within 0.5 L of clip 1), L of the first
   25 frames vs the end of clip k-1 (relay agreement), highlight clipping (% of pixels at 255), join x median, b*,
   blind critic.
3. **Attribution.** Run the run look through the LTX-2.5 VAE 1-5× (no sampling), and compare the clip 1 output L with
   the look L and the layout L. If the VAE alone loses about 1 L per pass, add the H3-style latent delta to the
   relay instead of only gaining the PNG. Seconds of GPU; run it only when the queue is empty.

Delivery fallback: `deflicker=size=49` on the final video smooths the remaining steps into ramps.

## Sources

- ComfyUI-LTXVideo (local, LTX-2 Community License): `latent_norm.py`, `easy_samplers.py` (LTXVNormalizingSampler,
  LinearOverlapLatentTransition), `looping_sampler.md`; https://github.com/Lightricks/ComfyUI-LTXVideo
- diffusers PR #12614 (LTXI2VLongMultiPromptPipeline, AdaIN 0.25): https://github.com/huggingface/diffusers/pull/12614
- Layout-To-Render card: https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render
- Kijai LTX-2.3 "Extend Any Video" discussion: https://huggingface.co/Kijai/LTX2.3_comfy/discussions/32
- LAMP https://arxiv.org/abs/2310.10769 · FreqForcing https://arxiv.org/abs/2607.27110 (html v2) · FreeNoise
  https://arxiv.org/abs/2310.15169 · ConsistI2V https://arxiv.org/abs/2402.04324 · MultiDiffusion
  https://arxiv.org/abs/2302.08113 · Gen-L-Video https://arxiv.org/abs/2305.18264 · FIFO-Diffusion
  https://arxiv.org/abs/2405.11473 · Ouroboros-Diffusion https://arxiv.org/abs/2501.09019
- StreamingT2V https://arxiv.org/abs/2403.14773 · Rolling Forcing https://arxiv.org/abs/2509.25161 · LongLive
  https://arxiv.org/abs/2509.22622 · Self Forcing https://arxiv.org/abs/2506.08009 · ID-Forcing
  https://arxiv.org/abs/2610.03120 · SVI https://arxiv.org/abs/2510.09212 · Recency Forcing
  https://arxiv.org/abs/2609.19729 · FramePack https://arxiv.org/abs/2504.12626
- SkyReels-V2 README (addnoise_condition): https://github.com/SkyworkAI/SkyReels-V2
- ComfyUI Context Windows: https://docs.comfy.org/built-in-nodes/ContextWindowsManual
- H3 Delta Color Carry (ComfyUI-MAINodes, GPL-3.0): https://comfy.icu/node/H3DeltaColorCarry ;
  https://github.com/matlowai/ComfyUI-MAINodes
- color-matcher (GPL-3.0): https://github.com/hahnec/color-matcher · KJNodes (GPL-3.0):
  https://github.com/kijai/ComfyUI-KJNodes
- Wan VACE colour-shift notes: https://civitai.com/models/2306202 ;
  https://civarchive.com/models/1536883
- §6: H3 nodes page https://github.com/matlowai/ComfyUI-MAINodes · Lin et al. "Common Diffusion Noise Schedules and
  Sample Steps are Flawed" https://arxiv.org/abs/2305.08891 · FFmpeg ticket #979
  https://ffmpeg.org/pipermail/ffmpeg-trac/2024-August/070619.html · OpenCV PR #27652
  https://github.com/opencv/opencv/pull/27652 · PyAV video API https://pyav.basswood.io/docs/stable/api/video.html ·
  local `ffmpeg -h filter=scale` / `filter=deflicker` (8.1.1) · ComfyUI `comfy/ldm/lightricks/vae/causal_video_autoencoder.py`
- All-In-One-Deflicker (Apache-2.0): https://github.com/ChenyangLEI/All-In-One-Deflicker , arXiv 2303.08120 ·
  fast_blind_video_consistency (MIT + CC BY-NC part): https://github.com/phoenix104104/fast_blind_video_consistency ·
  Deep Video Prior: https://github.com/ChenyangLEI/deep-video-prior , arXiv 2010.11838
