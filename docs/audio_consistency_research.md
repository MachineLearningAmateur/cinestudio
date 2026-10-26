# Audio consistency across the clips of one long video

Research report, 2026-10-24. Web research plus our repo and the installed ComfyUI. No code was changed, nothing was
downloaded into the repo, no GPU job was started. One CPU measurement was made on finished runs (section 1.4).

Tags used: **[REPO]** = read in our code or the installed ComfyUI (path given). **[MEAS]** = measured by me today.
**[WEB]** = a web source (URL, date or version). **[INF]** = my inference. **UNVERIFIED** = could not be confirmed.

Short answer:
- Today **no audio crosses a join**. Each clip starts from an empty audio latent. Only the video latent is relayed.
- Lightricks' own long-video code (ltx-pipelines 1.4.0, 2026-09-29) **carries the audio latent tail into the next
  window, frozen**, exactly like the video tail. We can rebuild that with ComfyUI core nodes and our existing relay.
  This is the first thing to build.
- A seam fix alone is not enough. Our clips also **fade inside themselves** (clip 1 drops 30-40 dB in 4 s in all three
  measured runs). That needs its own test.
- No open video-to-audio model is both strong and commercial-safe. The only clean route to one sound pass over the
  whole shot is LTX itself (frozen video, generated audio) or a Blender-driven hybrid (one bed + Foley at the contacts).

---

## 1. What we do today

### 1.1 The graph [REPO]

`workflows/ltx25_layout_api.json` (the default Blender-shot graph):
- `alat` = `LTXVEmptyLatentAudio` (121 frames, 24 fps): an all-zero audio latent, new in every clip.
- Stage 1: `av` = `LTXVConcatAVLatent(video = g_look[2], audio = alat)`, then one `SamplerCustomAdvanced`, CFG 1
  (`guider1`), sigmas from 0.979 (layout runs start below 1.0).
- Stage 2: `av2` = `LTXVConcatAVLatent(video = g_look2[2], audio = sep[1])`: the stage-1 audio is **re-noised** to
  sigma 0.909 and sampled again. Preview (`runner.to_stage1_only`) decodes the stage-1 audio (`adec` <- `sep[1]`).
- `adec` = `LTXVAudioVAEDecode`; `CreateVideo` muxes it.

`cinestudio/runner.py`:
- `add_relay_save` saves `[crop, 2]` = the **video** latent only (`LTXVCropGuides` output 2). `add_relay_guides` loads
  it into the next clip with `LTXVSelectLatents` + `LTXVAddLatentGuide`. **The audio latent is not saved and not
  relayed.**
- Every clip uses the **same seed** (`values["seed"] = m["seed"]`, runner.py ~line 2600) and, with a run-level
  `sound_text`, the **same sound sentence**. So "same seed, same sound text" is already what we do (section 5).
- `step_final`: `video.stitch` joins the clips; the audio parts are joined with
  `acrossfade=d=overlap/fps:c1=qsin:c2=qsin` (cinestudio/video.py `_stitch`): an **equal-power** fade over the 25 shared
  frames (1.042 s). Then a library song or the shot's own sound strips (`video.mix_strips`) can replace LTX's sound
  (`replace_audio`).

### 1.2 The audio latent [REPO]

- Audio VAE config in `ltx-2.5-audio-vae-bf16.safetensors` (header metadata): 16 kHz input, hop 160, 64 mel bins,
  stereo, 8 latent channels; output 48 kHz (vocoder `output_sampling_rate 48000`).
- ComfyUI `comfy/ldm/lightricks/vae/audio_vae.py`: latents per second = 16000 / 160 / 4 = **25**. The latent is 4-D
  (batch, 8, time, 16). The VAE is causal.
- So one 121-frame clip = 5.042 s = **126 audio latents**; the 25 shared frames = 1.042 s = **26 audio latents**; a
  408-frame shot = 17 s = 425 audio latents. The audio latent is tiny next to the video latent.
- Lightricks' code agrees: 25 latents/s (`ltx_core/types.py AudioLatentShape.from_duration`, LTX-2 repo) [WEB].

### 1.3 Nodes that touch audio latents (installed versions)

ComfyUI core (commit 63be2f9b, 2026-10-18) [REPO]:

| Node | File | What it does for us |
|---|---|---|
| `LTXVEmptyLatentAudio` | comfy_extras/nodes_lt_audio.py | zero audio latent of N frames |
| `LTXVAudioVAEEncode` / `LTXVAudioVAEDecode` | nodes_lt_audio.py | waveform <-> audio latent |
| `LTXVConcatAVLatent` | comfy_extras/nodes_lt.py | joins video + audio, keeps both noise masks. **If the first input is already an AV latent, it swaps the audio and `fit_audio` pads a shorter audio latent with zeros whose mask is 1 (generated)**: a short frozen head + a generated rest in one node |
| `LTXVSeparateAVLatent` | nodes_lt.py | splits, keeps the masks |
| `LTXVFreezeLatent` | comfy_extras/nodes_lt_keyframes.py | noise mask 0 on a video (5-D) or audio (4-D) latent; its own help text: "freeze audio before Concat AV" |
| `LTXVReferenceAudio` | nodes_lt.py | ID-LoRA speaker identity: reference audio tokens in the conditioning + an extra guidance pass. Made for voices with the ID-LoRA; for ambience UNVERIFIED |
| `LTXVModalityGuidance` | nodes_lt.py | extra pass with the audio<->video cross-attention cut, to strengthen sync; costs one pass per step |
| `LTXVDualCFGGuider` | nodes_lt.py | separate video and audio CFG (the distilled model runs at CFG 1) |
| `SetLatentNoiseMask`, `SolidMask` | core | used by Lightricks' A2V graph to freeze the audio |

The sampler keeps masked latents: `comfy/samplers.py` line ~642 `out = out * denoise_mask + latent_image * (1 - mask)`.

Lightricks ComfyUI-LTXVideo (v1.1.0, commit 220e1d4, 2026-10-20; LTX-2 Community License) [REPO]:
- `LTXVSetAudioVideoMaskByTime` (latents.py): time-range masks on the joint latent. It **overwrites** the video mask
  (it keeps a base mask only when that mask is per frame), so it may break our IC-LoRA guide masks [INF]. A community
  report says it cannot pad the latent (ComfyUI-LTXVideo#377, 2026-01-16) [WEB].
- `LTXVAudioOnlyModel` + `LTXVAudioOnlyEmptyVideoLatent` (audio_only.py): text-to-audio with the same model; it turns
  off the video stream and both cross-attentions.
- `LTXVLoopingSampler` refuses AV latents (looping_sampler.py line 728). PR #472 (open since 2026-04-25) adds audio [WEB].
- Example `example_workflows/2.5/LTX-2.5_A2V_Two_Stage_Distilled.json`: `LTXVAudioVAEEncode` -> `SetLatentNoiseMask`
  (SolidMask 0) -> `LTXVConcatAVLatent`; its notes: the audio tokens are "frozen (noise mask = 0) ... in either
  generation stage", and stage 2 concatenates "the original frozen audio tokens (not the stage-1 audio output)".

comfyui-kjnodes (installed) has `LTXVAudioVideoMask` (time masks, 25 latents/s hard-coded), but its licence is
**GPL-3.0** (its LICENSE file) [REPO]. Not needed: the core nodes above are enough.

### 1.4 Measured baseline [MEAS]

Script: `join_audio.py` in my scratchpad (CPU: ffmpeg decode + numpy RMS, ffmpeg `ebur128`, a 40-band log-mel
spectrum). It compares, at every join, clip k-1's version of the shared 1.042 s with clip k's version, and the second
before the shared time with the second after it. Runs: three 4-clip preview runs with LTX's own sound (same seed, the
same "upbeat light cinematic music ... evening breeze ... rustle of the jacket" sound sentence in every clip).

Loudness per 0.5 s of the finished final.mp4 (dBFS RMS; joins start at 4.0, 8.0, 12.0 s):

```
ce96  -16 -21 -25 -28 -30 -35 -37 -42 | -30 -22 -20 -20 -20 -20 -20 -20 | -25 -28 -29 -31 -30 -29 -27 -21 | -19 -23 -26 -30 -35
9377  -15 -21 -25 -29 -31 -36 -38 -44 | -19 -16 -19 -21 -24 -30 -37 -46 | -35 -28 -26 -24 -23 -23 -22 -20 | -21 -24 -31 -34 -41 -48
76bd  -14 -20 -24 -28 -31 -37 -39 -46 | -30 -21 -21 -21 -21 -21 -22 -24 | -25 -27 -25 -25 -27 -26 -24 -23 | -18 -18 -22 -25 -29 -35
```

| Run, join | Loudness 1 s before -> 1 s after (LUFS) | Step | Shared time: clip k-1 tail vs clip k head (dB RMS) |
|---|---|---|---|
| ce96 1->2 | -38.4 -> -20.8 | +17.6 LU | -50.7 vs -21.5 |
| ce96 2->3 | -20.1 -> -28.3 | -8.2 | -19.2 vs -30.4 |
| ce96 3->4 | -27.6 -> -25.6 | +2.0 | -27.2 vs -14.2 |
| 9377 1->2 | -40.2 -> -22.0 | +18.2 | -53.1 vs -11.5 |
| 9377 2->3 | -42.7 -> -25.2 | +17.5 | -58.5 vs -26.6 |
| 9377 3->4 | -21.8 -> -32.1 | -10.3 | -19.0 vs -29.1 |
| 76bd 1->2 | -41.7 -> -21.7 | +20.0 | -56.6 vs -21.3 |
| 76bd 2->3 | -23.4 -> -24.5 | -1.1 | -26.0 vs -27.2 |
| 76bd 3->4 | -23.4 -> -24.6 | -1.2 | -23.1 vs -12.6 |

Spectral distance (level-normalised log-mel, mean |dB|): across a join 3.3-8.0 dB; between two neighbouring seconds
inside one clip 0.9-3.4 dB. So a join changes the timbre about 2-3x more than the natural change inside a clip.

What this shows:
1. **Big loudness steps at most joins (up to 20 LU)**, with the same seed and the same sound sentence.
2. **Clips fade inside themselves.** Clip 1 falls from about -15 to -45 dB over 4 s in all three runs; the last clip
   and some middle clips fall the same way. The "music starts at 4.0 s" impression is clip 2 entering after clip 1 has
   faded. The cause is unknown (section 9, open question 1). A tail carry alone would carry a faded tail on.
3. The two versions of the shared second disagree by 10-40 dB: an equal-power cross-fade between them cannot hide
   that. The fade curve is right (qsin for uncorrelated audio, section 6), the inputs are the problem.
4. Shan shen bu runs (3 clips) replace LTX's sound with the template's music. Their LTX clip audio was near silence
   (-55 to -105 dB), so they are not a test of LTX's sound unless re-made with `--sound ltx` and a sound sentence.

---

## 2. Question 1: carrying the audio latent across a join inside LTX

### 2.1 What Lightricks does [WEB, checked in source]

ltx-pipelines **1.4.0 (2026-09-29)** added `ltx_pipelines.chunks` (CHANGELOG.md, github.com/Lightricks/LTX-2):
- `--chunked`, `--chunk-pixel-frames` (default 97), `--chunk-carry-frames` (default 25), `--chunk-blend-frames`, on
  Distilled, TI2Vid two-stage, A2Vid, Dub-It and IC-LoRA pipelines.
- `chunks/operators.py`: after each window, `carry_video = _slice_latent_tail(video_latent, ...)` and
  `carry_audio = _slice_latent_tail(audio_latent, ...)`, the audio tail length = `AudioLatentShape.from_duration(
  carry_frames / fps)`. The next window gets them at latent index 0 as `VideoConditionByLatentIndex` and
  **`AudioConditionByLatentIndex(strength 1.0)`** (also new in 1.4.0: "pins an audio latent at a given latent index").
- When a modality is frozen, the code keeps the latent that went in, not the sampler's copy (bf16 drift is "inaudible in
  a latent, but the vocoder turns it into a visibly different waveform").
- Audio seams are decoded with a **40 ms equal-power cross-fade** (`AUDIO_SEAM_CROSSFADE_MS = 40.0`); video uses a
  linear fade over the carried frames.
- `chunks/layout.py audio_latent_for_layout`: chunked A2Vid gives each window "the matching slice" of ONE full audio
  latent, frozen.

This is our layout: 25 carried frames, windows of about 100-121 frames. It is the official answer to Q1.

Other official pieces [WEB]:
- `RetakePipeline` (`retake.py`): `start_time`/`end_time`, `regenerate_video` / `regenerate_audio`; the other modality is
  frozen with a `TemporalRegionMask`. Without a source audio track it makes audio for the whole canvas: in effect V2A
  when `regenerate_video=False`. Distilled: 8 fixed steps, no guidance; frames must be 8k+1.
- `A2VidPipelineTwoStage`: audio frozen in both stages; the original waveform is muxed back.
- `T2AOneStagePipeline`: audio only.
- No dedicated V2A pipeline and no "extend" pipeline in the open code. The LTX API has `submit-extend` ("preserving
  audio and visual continuity", ltx-2-3-pro, 2026-02-18) and `submit-retake` (at least 2 s), docs.ltx.io.
- The LTX-2 report (arXiv 2601.03233, v1, Jan 2026): separate audio and video streams joined by bidirectional
  cross-attention in every layer; separate latents "natively support ... synchronized audio for an existing video
  (V2A) or ... video driven by a specific audio track (A2V)"; "up to 20 seconds" in one pass; drift beyond ~20 s.
  No training clip lengths, nothing on multi-segment audio.

Community [WEB]: users continue videos by encoding the old clip's audio, freezing it and generating the extension
(ComfyUI-LTXVideo#377, 2026-01-16; RuneXX "Extend Any Video", HF Kijai/LTX2.3_comfy discussions #31/#32/#39, March
2026, about 3 s overlap; chaining degrades after several extensions). PR #472 says adding audio afterwards in a low-sigma
pass gave worse lip-sync than carrying it. Encore (arXiv 2609.04249, 2026-08-28, built on LTX-2.3, LTX-2 Community
License) carries "last frame plus audio tail" and adds a frozen audio reference at negative time positions; that
reference trick needs its trained LoRA.

### 2.2 Can we do it in our ComfyUI graph? Yes, with core nodes only [REPO + INF]

1. **Save** clip k's stage-1 audio latent: `SaveLatent(sep[1])` next to `relay_save_crop` (audio is not cropped).
2. **Cut** the last 26 latents in Python (runner, at relay time): load the safetensors file, slice
   `latent_tensor[:, :, -26:, :]`, write a new `.latent` **with the `latent_format_version_0` key** (otherwise
   `LoadLatent` multiplies by 1/0.18215, nodes.py `LoadLatent.load`). No custom node is needed. (Whether
   `LTXVSelectLatents` slices 4-D audio is UNVERIFIED; the Python cut avoids the question.)
3. **Pin** it in clip k: `LoadLatent` -> `LTXVFreezeLatent` -> a second `LTXVConcatAVLatent(av, frozen_tail)`. The
   first input is already an AV latent, so `fit_audio` pads the 26 frozen latents to 126 with mask-1 zeros: the first
   1.042 s are fixed, the rest is generated. Index 0 of clip k's audio = clip k-1's 4.0 s = the start of the shared
   frames, the same timing as the video relay.
4. **Stage 2:** freeze the whole stage-1 audio (`LTXVFreezeLatent(sep[1])` into `av2`) instead of re-noising it. This
   is Lightricks' own pattern (A2V notes; the Dub-It README: "Stage 2 upscales while freezing the audio"). Then preview
   and final have the same audio, and only the stage-1 audio latent needs to be relayed.
5. **Join:** the shared second is now (almost) the same signal in both clips. An equal-power fade of two **correlated**
   signals rises by up to +3 dB in the middle (section 6). So for carried clips: take clip k-1's audio through the
   shared time and switch to clip k near its end (where clip k's causal decoder has 1 s of context), with a short fade
   (40 ms, Lightricks' value). Better still: **decode once** (section 2.3).

Unknowns: whether a pinned tail at sigma-start 0.979 (our layout schedule) behaves like Lightricks' carry (UNVERIFIED);
whether the joint attention lets the frozen audio pull the video (it should be harmless or helpful: PR #472 [WEB]);
whether 1 s of carry holds the timbre over 3-4 clips (Lightricks uses the same 25 frames; the community reports drift
over many extensions [WEB]).

### 2.3 One decode for the whole shot [INF]

With every carried tail frozen, the full shot's audio latent is: clip 1's 126 latents + clip k's latents 27..126 for
k > 1 (the frozen heads are dropped; take the tails from the saved inputs, not the sampler copies, as Lightricks does).
One `LTXVAudioVAEDecode` over that latent (425 latents for 17 s) gives one waveform with **no seams at all**. The
decode is small. Cost: one more tiny ComfyUI job (a queue ticket) or an extra branch in the last clip's graph. Retakes
must rebuild the concatenated latent.

---

## 3. Question 2: one audio pass for the whole shot

### 3.1 LTX itself, video frozen, audio generated (V2A) [WEB + INF]

- Supported by design (LTX-2 report) and by `RetakePipeline` with `regenerate_video=False`; not as a ComfyUI example.
  In ComfyUI: VAE-encode final.mp4 -> `LTXVFreezeLatent` -> `LTXVConcatAVLatent` with an empty audio latent of the full
  length -> plain distilled model (no Layout LoRA: the LoRA is what pushes the audio to silence, runner.py SOUND_TEXT
  note) -> decode -> `video.replace_audio`. All core nodes.
- Length: 17 s is inside the "up to 20 s" of one pass. Longer shots: windows with the audio carry (2.2).
- Tokens [INF]: at 768x448 a 409-frame latent is 52 x 14 x 24 = 17,472 video tokens; a preview clip is 16 x 14 x 24 =
  5,376 (plus its guide tokens). At 384x224 the whole shot is 4,368 tokens, less than one clip. Whether V2A works well
  from a half-size video is UNVERIFIED.
- 8 GB: a 20 s LTX-2.3 distilled GGUF T2V at 900x1600 on an RTX 3070 took 21 min (Reddit r/StableDiffusion, March 2026,
  UNVERIFIED single report). For LTX-2.5 on 8 GB: no report found.
- Risk: Lightricks' **Foley V2A LoRA** (HF Lightricks/LTX-2.3-22b-LoRA-Foley-V2A, LTX-2 Community License; docs.ltx.io
  feature guide, fetched 2026-10-24) needs guidance 6-8 ("around 4 collapses to near-silence"), is trained on 89
  frames, runs at most 169 frames (~7 s) per pass, is validated on LTX-2.3 only ("not validated on LTX-2.5 yet"), and
  quiet sounds (footsteps) depend strongly on the seed. Our distilled model refuses guidance. So plain V2A on 2.5
  distilled at CFG 1 may give weak or silent Foley: UNVERIFIED, test first.

### 3.2 Separate open V2A models [WEB, licences spot-checked on the model cards]

| Model | Weights licence | Verdict |
|---|---|---|
| MMAudio (code MIT) | CC-BY-NC 4.0 (HF hkchengrex/MMAudio) | no |
| HunyuanVideo-Foley | tencent-hunyuan-community, excludes EU/UK/KR | no (our rule) |
| ThinkSound / PrismAudio | README: "Commercial use is NOT permitted" | no |
| AudioX | CC-BY-NC 4.0 | no |
| FoleyCrafter | adapter Apache-2.0, but needs Auffusion (CC-BY-NC-SA) | no |
| Kling-Foley, SoundReactor, Tri-Ergon | weights not released | no |
| Ovi, MOVA, JavisDiT, MAGI | pull in NC or community audio parts; none does V2A on an existing video | no |
| CAFA (MIT, ICCV 2025) | built on Stable Audio Open (Stability Community) | maybe; ≤10 s; UNVERIFIED bundle |
| Diff-Foley | MIT, weak quality (FoleyBench FAD 31.9 vs MMAudio 8.76) | licence ok, quality no |
| LTX-2.3 Foley V2A LoRA | LTX-2 Community License (free under USD 10M revenue; LICENSE-2_x, 2026-08-11) | conditional; 2.3 only |

All are trained on 8-10 s clips. No benchmark covers stylised or animated video; the one real-vs-generated test
(Movie Gen Audio Bench, arXiv 2410.13720) does not cover these models. **Conclusion: no third-party V2A model now.**

---

## 4. Question 3: sound from the Blender scene

- Events we already have [REPO]: `bpy_jobs/rig_timeline.py` exports `foot.L`, `foot.R`, hips, hands, head and the camera
  per frame for every Blender shot; `cinestudio/timeline.py` uses it (PLANTED = 0.06 m). `cinestudio/mocap.py` keeps a
  per-frame `contact` for SAM 3D templates. A foot contact = foot near the floor and slow, from the same data: no GPU,
  no extra Blender pass.
- Mixing [REPO]: `video.mix_strips` already mixes timed sound strips with ffmpeg (`adelay`, `amix`, `volume`, hard `-t`).
  Foley one-shots fit the same pattern.
- Game practice [WEB]: footsteps fire on animation events at the foot plant, the surface picks the sample set (Unreal
  "Add footstep sounds", dev.epicgames.com; Wwise switch, audiokinetic blog). Avoid repeats with 4-8 samples per surface,
  shuffle, small random pitch (about x0.9-1.1) and volume (Unity Audio Random Container docs, Unity 6; FMOD 2.03 docs).
  Distance: about -6 dB per doubling for a point source (SFU Handbook for Acoustic Ecology); we know the
  camera-to-foot distance per frame.
- Film practice [WEB, trade sources]: one continuous background ("BG") ambience under a scene ties the shots together
  and hides seams (irpr glossary; StudioBinder on room tone).
- Commercial-safe samples [WEB]:

| Source | Use in a video | Commit to the public repo |
|---|---|---|
| Freesound CC0 (filter `f=license:"Creative Commons 0"`; check the "No generative AI" flag and keep the URL) | yes | yes |
| Kenney audio packs (CC0, kenney.nl/support) | yes | yes |
| OpenGameArt, CC0 items only | yes | yes |
| Sonniss GDC bundle (licence v2.0, 2026-08-27) | yes | **no** ("may not distribute ... as sound effects"; ML training forbidden) |
| Pixabay (terms 2026-10-20), Mixkit | yes | no (no standalone redistribution) |
| Zapsplat | free tier needs credit (secondary sources, UNVERIFIED) | no |
| BBC Sound Effects (RemArc) | no (non-commercial; UNVERIFIED primary text) | no |

- Bed sources: an ACE-Step song (MIT; we have it), LTX text-to-audio (`LTXVAudioOnlyModel`, our LTX licence, ≤20 s),
  CC0 room tone, or Stable Audio Open / Stable Audio 3 (Stability AI Community License: free under USD 1M revenue,
  registration, "Powered by Stability AI" notice: conditional, not plain permissive). MOSS-SoundEffect is tagged
  Apache-2.0 but its size (8B per the card) and data are UNVERIFIED.

Is the hybrid better than model sound? For **consistency and sync**, yes by construction: one bed has no joins; a step
on a contact frame is exact to the layout [INF]. LTX follows the layout's timing closely, but in orbits the body runs
~30 deg ahead (AGENTS.md 7.4), so a contact may be a few frames off the picture: check. For **richness**, model sound
can be better (cloth, the place); one study found procedural SFX "less believable" for everyday actions in cartoons
(Garcia & Reiss, arXiv 2605.31082, 2026-05-29). No direct test of our case exists: needs our own blind test.

---

## 5. Question 4: text-only fixes

- Same sound sentence + same seed in every clip: **already done** and measured: steps up to 20 LU (1.4). Not enough.
- Lightricks' prompting guide: describe the sound concretely; our A/B (runner.py note, 2026-10-22): a concrete sound
  sentence -34 dB vs a "quiet" one -55.6 dB. Keep it, but it does not tie clips together.
- Audio CFG: the LTX-2 report uses audio text scale 7 for the full model; the distilled model runs at CFG 1 and
  refuses guidance (AGENTS.md 7.2). Not available.
- `LTXVModalityGuidance`: deformed faces at every strength in a community LTX-2.5 A2V test (HF Lightricks/LTX-2.5
  discussion #44, Aug 2026) [WEB]. Not first.
- No evidence was found that ending every clip on the same audio words helps. UNVERIFIED; low value.
- Music by text will always change per clip: use a library song for music (AGENTS.md 7.2 already says so).

---

## 6. Question 5: post-processing at joins

- **Fade curve** [WEB]: equal power (qsin, -3 dB per side in the middle) for uncorrelated signals; equal gain (tri) for
  correlated ones, else +3 dB bump (QSC crossfader docs; Fink, Holters, Zölzer, DAFx 2016). ffmpeg `acrossfade`
  defaults to `tri`; we set `qsin`, which is right for today's independent clips and wrong for carried (identical)
  audio. Keep both, chosen by the carry.
- **Loudness match** [WEB]: EBU R128 v5.0 (Nov 2023), ITU-R BS.1770-5: momentary 400 ms, short-term 3 s. ffmpeg
  `ebur128` gives `lavfi.r128.M/S` per frame; `loudnorm` linear mode needs two passes and falls back to dynamic
  compression in some cases (ffmpeg-filters docs); for clip matching use a plain `volume` gain ramp. pyloudnorm is MIT
  (v0.2.0, 2026-01-04). Our measured steps are 1-20 LU; a gain ramp over the shared second can remove the step, but not
  a fade inside a clip, and over-gaining a faded tail lifts noise [INF].
- **EQ match** [WEB]: smoothed spectrum ratio -> `firequalizer gain_entry` (Logic Match EQ, Ozone docs: smooth it, use
  under ~50%). matchering does this but is **GPL-3.0**: use the idea only.
- **Longer fades**: no standard found (UNVERIFIED). Rule of thumb: seconds for steady ambience, beat-aligned short
  fades for music. Our fade is bounded by the shared 1.042 s of real overlap; longer means fading over mismatched time.
- **Join point**: for independent clips, cut inside the shared second where the two versions are most alike (lowest
  spectral distance, lowest level) with a short fade [INF; splice-detection literature uses spectral flux and
  noise-floor changes: tembrica.com; Zhao et al. arXiv 1411.7084].
- Post-processing is a safety net. It cannot make two different sound scenes one scene.

---

## 7. Question 6: what other systems do

- **Lightricks:** carry the audio tail frozen, slice one audio track for A2V, 40 ms seam fade (2.1). LTX API extend
  "preserving audio and visual continuity" [WEB, docs.ltx.io].
- **Veo 3.1:** an extension is "generated based on the final second of the previous clip ... ideal for extending shots
  with background audio" (Google developers blog, Oct 2025) [WEB]: the same 1 s carry idea.
- **Sora 2:** extensions use "the full source clip as context" (developers.openai.com video guide); audio not
  documented; API shut down 2026-09-24 [WEB].
- **Kling 3.0, Runway:** no documented audio-aware extend (UNVERIFIED).
- **Research:** frozen clean prefixes / audio tails with recurrence (Vorch-Human arXiv 2609.26117; Encore 2609.04249;
  JoyAI-Echo-1.5 2608.23383 uses full-shot audio cues for identity). All agree: condition the next segment on the
  previous segment's clean audio [WEB].
- **ACE-Step 1.5** (MIT, github.com/ace-step/ACE-Step-1.5, released 2026-01-28) [WEB]: `duration` 10-600 s, so **a bed of
  the shot's exact length is direct** (our `song make --seconds`; `EmptyAceStep1.5LatentAudio seconds` in
  workflows/acestep_base_api.json [REPO]; shorter shots: make 10 s and cut). `repaint` regenerates a time range of a
  source audio; `extend` at the end is UNVERIFIED in 1.5 (v1, Apache-2.0, had it). ComfyUI core has no ACE repaint node
  (nodes_ace.py only has `ReferenceTimbreAudio`) [REPO]. ACE-Step is a music model; ambience from it is UNVERIFIED.
- **Stable Audio 3** (arXiv 2605.17991, 2026-05) has SFX "outpainting" (causal continuation): a "continue this ambience"
  tool, but under the Stability Community License + Gemma terms [WEB].

---

## 8. Question 7: the approaches side by side

| Approach | Consistency | Sync with motion | Effort in our code | VRAM 8 GB / large GPU | Licence |
|---|---|---|---|---|---|
| A. Audio tail carry + stage-2 audio freeze + carry-aware join (2.2) | good at joins; drift over many clips possible | as today (LTX per clip) | small: runner relay + 3 nodes + stitch mode | +26 audio latents: none / none | LTX (already used) |
| A+. One audio decode for the shot (2.3) | no seams | as A | small-medium | tiny job | LTX |
| B. Join post-processing (gain ramp, EQ match, join point) | reduces steps; not scene changes or in-clip fades | unchanged | small, CPU | none | ffmpeg/numpy (our code) |
| C. Hybrid: one bed + Foley at rig contacts, LTX sound off or ducked | best (bed has no joins) | exact to the layout; check orbits | medium, CPU | none | CC0 / ACE-Step MIT / LTX T2A |
| D. LTX V2A over the stitched final (3.1) | one pass, no joins (≤20 s) | follows the final picture | medium: new graph + step + queue | UNVERIFIED on 8 GB; easy on 48-80 GB | LTX |
| E. Third-party V2A | one pass per 8-10 s | good on real footage | medium | MMAudio ~6 GB | **none clean** |
| F. Text only | no gain (measured) | as today | none | none | - |

---

## 9. Open questions found on the way

1. **Why do clips fade?** Clip 1 drops ~30 dB in all three runs; the last clip too. Ideas to test (UNVERIFIED): the
   frame -1 look image as an end guide (our 2026-10-22 A/B: "no look image -91 dB", so the look strongly changes the
   audio), the zero audio init at sigma 0.979, the music text ("score ... plays throughout"). Test on existing seeds
   with ambience-only text, and with the look guide moved. This matters for A: a faded tail would be carried on.
2. Does a frozen audio head change the video (cross-attention)? Measure the picture with `run score` too.
3. Does LTX-2.5 distilled make useful V2A at CFG 1, at half size? (D)

---

## 10. Recommendation

1. **First: A + A+ (audio carry, frozen stage-2 audio, one decode or a carry-aware join), plus the fade test of 9.1.**
   It is the official Lightricks recipe, needs no new model, no new licence and no VRAM, and reuses our relay.
2. **Second: C (hybrid bed + Foley at contacts)** as a sound mode for shots that need clean, steady sound (contest
   demo, music shots). CPU only; ACE-Step or CC0 bed; CC0 steps. It also gives a safe fallback when LTX's sound is weak.
3. **Third: D (LTX V2A over the whole final)** as an experiment on preview runs; and B (gain ramp at joins) as a cheap
   safety net for old runs and independent clips. Watch for an LTX-2.5 Foley LoRA (docs: "in development").
   Do not adopt any third-party V2A model now (licences).

### 10.1 Design of the first step (no code written; for the implementer)

Files: `cinestudio/runner.py`, `cinestudio/video.py`, `setup/make_layout_graphs.py` (regenerate
`workflows/ltx25*_layout*_api.json` + `.bind.json`), tests in `tests/`.

1. Manifest option `audio_relay`: `"carry"` (new default for layout runs with LTX sound) | `"off"` (old runs: no field =
   off). Add it to `FORK_FIELDS`, `_fork_locked` and `TAKE_FIELDS` where relevant (AGENTS.md 7.5).
2. Graph (all clips): `audio_save` = `SaveLatent(samples=["sep", 1], filename_prefix=<relay prefix>_a)`.
   Stage 2: `audio_freeze2 = LTXVFreezeLatent(["sep", 1])`; `av2.audio_latent = ["audio_freeze2", 0]`. Preview keeps
   `adec <- sep[1]`.
3. Clip k > 1: runner cuts the previous clip's saved audio latent to its last `round(overlap / fps * 25)` latents (26)
   in Python (safetensors, keep `latent_format_version_0`), copies it to ComfyUI input (like `_relay_inputs`), and adds
   `atail_load = LoadLatent`, `atail_frz = LTXVFreezeLatent`, `av_carry = LTXVConcatAVLatent(video_latent=["av", 0],
   audio_latent=["atail_frz", 0])`; `ks.latent_image = ["av_carry", 0]`. Keep the run's own copy in `work/latents`.
4. Retakes: a take of clip k uses clip k-1's audio tail; the take after it (clip k+1) must be re-made or its head
   re-pinned, the same rule as the video relay's end anchor. Record the audio file in `parts[k]["relay"]["a1"]`.
5. Join: `video.stitch(..., audio_join="carry")`: take clip k-1's audio until `shared - 0.1 s`, then clip k, 40 ms
   equal-gain fade at the switch. (A+ later: concatenate the saved latents and decode once in a small graph.)
6. Events + note: show "audio carry on" in the run's settings list.
7. Unit tests (no GPU): the tail cut length and file format, the graph wiring (`av_carry` only for k > 1, stage-2
   freeze), the stitch mode keeps the total length and sync.

### 10.2 A/B test plan (preview, one existing template)

Shot: `shan_shen_bu` (11.5 s, 3 clips, 276 frames) **with `--sound ltx`** and an ambience-and-steps sound text
without music (for example "The sound: a steady breeze over the place, the soft rustle of cloth and quick light
footsteps on the floor as the figure moves; no music, no one speaks."). The template's own music must not replace
LTX's sound. Same seed (two seeds if time allows), same character, preview.

Arms:
- **A0** today (independent clips, qsin fade).
- **A1** audio carry + stage-2 freeze + carry join.
- **B on A0** gain-ramp + join point (CPU, no GPU: also run on the three old 4-clip runs above).
- **C** hybrid: CC0 steps at rig contacts + one bed (CPU; on A1's video).
Optional later: **D** LTX V2A over A0's final.

Measures (scratchpad `join_audio.py` can be extended into `setup/audio_join_check.py`):
1. Loudness step at each join: LUFS of the 1 s before the shared time vs the 1 s after, and short-term (3 s) S across
   the join. Goal (proposed, no standard found): |step| <= 2 LU. Baseline 1-20 LU.
2. Spectral distance across the join (level-normalised log-mel, mean |dB|) divided by the median inside-clip distance.
   Goal: <= 1.5. Baseline ~2-3x.
3. In-clip fade: max drop of the 0.5 s RMS inside each clip. Goal: < 10 dB unless the action ends. Baseline 30-45 dB.
4. Sync: onsets (song.py `_onsets`) vs rig foot contacts; share of contacts with an onset within ±2 frames (83 ms).
5. Picture unchanged: `run score` per clip (A1 vs A0), the gauntlet numbers must not get worse.
6. Blind listening by the user: anonymous file names, pairwise preference per join and for the whole clip; an ABX of
   16 trials needs 13/16 correct for p < 0.05 (binomial). A fresh agent cannot listen, so this is a human check.

Pass for A1: goals 1 and 2 met at every join, 5 not worse, and the user prefers A1 over A0 in the blind check.

---

## Sources

Repo and installed software (read 2026-10-24): `cinestudio/runner.py`, `cinestudio/video.py`, `cinestudio/song.py`,
`cinestudio/timeline.py`, `bpy_jobs/rig_timeline.py`, `workflows/ltx25_layout_api.json`,
`workflows/acestep_base_api.json`; ComfyUI core 63be2f9b (2026-10-18): `comfy_extras/nodes_lt.py`,
`nodes_lt_audio.py`, `nodes_lt_keyframes.py`, `nodes_ace.py`, `comfy/ldm/lightricks/vae/audio_vae.py`,
`comfy/samplers.py`, `nodes.py`; ComfyUI-LTXVideo v1.1.0 220e1d4 (2026-10-20): `latents.py`, `audio_only.py`,
`looping_sampler.py`, `README.md`, `example_workflows/2.5/LTX-2.5_A2V_Two_Stage_Distilled.json`;
comfyui-kjnodes `nodes/ltxv_nodes.py`, `LICENSE`; audio VAE header of `ltx-2.5-audio-vae-bf16.safetensors`.
Copies of some upstream source files fetched by a helper are in the session scratchpad (`ltx2src`, `encore`, `ace`),
not in the repo.

Web (accessed 2026-10-24 unless a date is given):
- Lightricks LTX-2 repo, CHANGELOG 1.4.0 (2026-09-29), `packages/ltx-pipelines` chunks/operators.py, chunks/layout.py,
  retake.py, docs/pipelines.md: https://github.com/Lightricks/LTX-2
- LTX-2 technical report, arXiv 2601.03233 v1 (Jan 2026): https://arxiv.org/abs/2601.03233
- LTX docs (API endpoints, extend, retake, Foley V2A guide): https://docs.ltx.io/welcome ,
  https://docs.ltx.io/models/ltx-2-3 , https://docs.ltx.io/open-source-model/feature-guides/audio/video-to-audio-foley
- LTX-2.3 Foley V2A LoRA card: https://huggingface.co/Lightricks/LTX-2.3-22b-LoRA-Foley-V2A
- LTX Community licences: https://raw.githubusercontent.com/Lightricks/LTX-2/main/LICENSE-2 (2026-01-05),
  https://raw.githubusercontent.com/Lightricks/LTX-2/main/LICENSE-2_x (2026-08-11)
- ComfyUI-LTXVideo issues/PRs: #369 (2026-01-13), #377 (2026-01-16), PR #472 (open, 2026-04-25):
  https://github.com/Lightricks/ComfyUI-LTXVideo
- HF Lightricks/LTX-2.5 discussion #44 (Aug 2026); HF Kijai/LTX2.3_comfy discussions #31, #32, #39 (March 2026)
- Encore, arXiv 2609.04249 (2026-08-28): https://github.com/shaohua-pan/Encore ; Vorch-Human arXiv 2609.26117;
  JoyAI-Echo-1.5 arXiv 2608.23383
- Veo 3.1 developers blog (Oct 2025): https://developers.googleblog.com (post "Introducing Veo 3.1"); Sora video guide:
  https://developers.openai.com/docs/guides/video-generation
- ACE-Step 1.5 (MIT): https://github.com/ace-step/ACE-Step-1.5 ; ACE-Step v1 (Apache-2.0): https://github.com/ace-step/ACE-Step
- Stable Audio Open 1.0: https://huggingface.co/stabilityai/stable-audio-open-1.0 ; Stable Audio 3 small-sfx:
  https://huggingface.co/stabilityai/stable-audio-3-small-sfx (arXiv 2605.17991); licence:
  https://stability.ai/community-license-agreement (2024-07-05)
- V2A licences: https://huggingface.co/hkchengrex/MMAudio ; https://github.com/FunAudioLLM/ThinkSound ;
  https://huggingface.co/tencent/HunyuanVideo-Foley ; https://raw.githubusercontent.com/ZeyueT/AudioX/main/LICENSE ;
  https://github.com/finmickey/CAFA ; https://huggingface.co/klingfoley ; FoleyBench-Long arXiv 2511.13219 ;
  MMAudio paper arXiv 2412.15322 ; Movie Gen arXiv 2410.13720 ; HunyuanVideo-Foley arXiv 2508.16930 ;
  Garcia & Reiss arXiv 2605.31082
- Sample libraries: https://freesound.org/help/faq/ ; https://sonniss.com/gdc-bundle-license/ (v2.0, 2026-08-27);
  https://kenney.nl/support ; https://opengameart.org/content/faq ; https://pixabay.com/service/terms/ (2026-10-20);
  https://mixkit.co/llm-info/
- Game Foley: https://dev.epicgames.com/documentation/unreal-engine/artist-11-add-footstep-sounds-to-a-character ;
  https://docs.unity3d.com/Manual/AudioRandomContainer-UI.html ; https://www.fmod.com/docs/2.03/studio/working-with-instruments.html ;
  https://www.sfu.ca/sonic-studio-webdav/cmns/Handbook/Inverse-Square_Law.html
- Loudness and fades: https://www.itu.int/rec/R-REC-BS.1770-5-202311-I ; https://tech.ebu.ch/publications/r128 ;
  https://ffmpeg.org/ffmpeg-filters.html ; https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/libavfilter/af_afade.c ;
  https://q-syshelp.qsc.com/q-sys_5.1/Content/Schematic%20Library/crossfader.htm ; Fink, Holters, Zölzer, DAFx 2016
  (https://www.hsu-hh.de/ant/wp-content/uploads/sites/699/2017/10/Fink-Holters-Zölzer-2016-Signal-matched-power-complementary-cross-fading-and-dry-wet-mixing.pdf);
  https://pypi.org/project/pyloudnorm/ ; https://github.com/sergree/matchering (GPL-3.0)
- Seams and listening tests: https://en.wikipedia.org/wiki/Log-spectral_distance ; https://en.wikipedia.org/wiki/MUSHRA ;
  https://arxiv.org/pdf/1411.7084 ; https://tembrica.com/en/audio-splice-detector
