# Generating Blender shots (templates) from text (research, 2026-10-25)

The question (user): "is there any opensource model we can use to drive the generation of the blender views?", meaning
an open model that builds a TEMPLATE from a text prompt: the mannequin's motion, the camera, the set and the props.

Scope: web research, a read of our template code, and one small CPU experiment (a rule checker,
`.cinestudio/gauntlet/shot_gen_research/check_rules.py`). No GPU jobs, no downloads, no project code changed.
Markers: **[V]** = checked on the primary source (licence file, model card, repo README). **[M]** = measured by us.
**[C]** = claimed by the authors, not checked by us. **[I]** = our inference. This is a licence summary, not legal advice.

## 0. Summary and ranking

- **No single open model makes a whole shot.** Motion, camera and set come from different places. Our own code joins
  them. The best split: an LLM writes a small JSON shot spec, and our code builds the .blend from it.
- **Almost every academic text-to-motion model is blocked by its training data.** They train on HumanML3D, KIT-ML,
  BABEL or Motion-X, which come from AMASS/SMPL. The AMASS licence "prohibits the use of the Dataset to train
  methods/algorithms/neural networks/etc. for commercial use of any kind" [V]. MIT or Apache code does not help.
- **One text-to-motion family is commercial-safe: NVIDIA Kimodo SOMA** (released 2026-03-16, SOMA-RP v1.2 on
  2026-10-15). Its model card says "This model is ready for commercial use" and names the NVIDIA Open Model License. It
  was trained on 700 h of Bones Rigplay mocap, not on AMASS [V]. It fits our 3070 when the text encoder runs on the CPU.
- **Video-to-motion with a generated video works with what we already have** (LTX-2.5 or Wan 2.2 -> SAM 3D Body ->
  mocap.py). A 2025-26 paper used it at scale: about 14k of 20k Wan2.1 clips kept, with somewhat more foot sliding [C].
  We rank it second, as a fallback for actions that Kimodo does not know.

| # | Option | Licence | Doable by 10-30? | One line |
|---|---|---|---|---|
| 1 | **Kimodo-SOMA-RP v1.2** text-to-motion -> our motion format -> flash_step builder | **OK** (NVIDIA Open Model License + Llama 3 text encoder, both with obligations) | yes, 1-2 days | Real text-to-motion, 10 s max, keyframe/path constraints, 30 fps |
| 2 | **LLM planner -> JSON shot spec -> our own code** (pose_frame, routines, settle, rule-based camera, set builders) | **OK** (our code; Qwen3.5/3.6, gpt-oss, Gemma 4, Mistral Small: Apache-2.0) | yes, 1-3 days (small vocabulary) | Controllable and checkable; motion only as rich as our vocabulary |
| 3 | **Video-to-motion generator:** text -> LTX-2.5 (or Wan 2.2) video of a person, front camera -> SAM 3D Body -> mocap.py | **OK** (LTX-2.x Community License: free under USD 10M revenue; Wan 2.2 Apache-2.0; SAM License) | yes, 1-2 days to wire | Reuses the proven path; GPU minutes per take; jitter and foot-contact risks |
| 4 | **CMU mocap library + retrieval** (LLM picks clips by CMU's own descriptions) | **OK with a limit**: no resale of the data "even in converted form" | 2-3 days (retargeting) | Real mocap; small labels; do not ship the library itself |
| 5 | Academic T2M: MDM, MotionDiffuse, T2M-GPT, MoMask, MotionGPT/3, MotionLCM, LaMP, Light-T2M, MARDM, BAMM, STMC, MotionStreamer | **BLOCKED** (AMASS/SMPL data; some also have NC code) | - | Good benchmarks, unusable for us |
| 6 | HY-Motion 1.0 (Tencent) | **EXCLUDED** (project rule; licence does not apply in the EU/UK/KR) | - | Strong, but a Hunyuan model |
| 7 | Learned camera paths (E.T./DIRECTOR, GenDoP, Director3D) | **BLOCKED/UNCLEAR** (film or YouTube training clips) | - | Use rule-based camera moves instead |
| 8 | Sets: our builders (gym_set.py) + CC0 libraries (Poly Haven, Kenney, ambientCG, Quaternius); Infinigen (BSD) later | **OK** | builders yes; Infinigen later | Holodeck/Objaverse: BLOCKED in practice (NC objects, GPT-4o) |

**Recommended MVP (section 6):** text -> LLM shot spec -> Kimodo beats (+ our pose_frame end pose) -> the existing
flash_step builder -> rule-based camera + set from a small vocabulary -> automatic template-rule check -> Workbench
preview -> blind critic. The planner can be the agent that already drives cinestudio through MCP. A local Apache LLM
can come later for the app's own text box.

## 1. Where a generator plugs in (our code)

- `templates/flash_step/build.py --motion M.json` builds a whole template (rig animation, leg pins or IK, kicks, still
  front camera at 4 m, plain floor) from a **motion file**. That file has, per frame: unit bone directions, the hip
  move, ankle positions, foot contacts, the pelvis roll and the kick weights [V, code].
- `cinestudio/mocap.py` makes that file from 3D joint positions (`_frames(P[frames,18,3], H[frames,42,3])`,
  OpenPose-18 + hands, metres). It also edits it: `accent`, `mirror_legs`, `settle`, `pose_frame`, `rest_frame` [V].
- So **any source that gives 3D joint positions per frame can become a template.** That covers a text-to-motion model,
  SAM 3D Body on a generated video, a mocap BVH, and LLM keyframes made with `pose_frame`. For each one we need only an
  adapter to the 18 joints. The template rules, the leg pins and the end hold then come for free.
- Existing helpers: `bpy_jobs/rig_timeline.py` (per-frame bone and camera positions), `cinestudio/timeline.py` (events
  and clip texts), `bpy_jobs/scan_layers.py` (layer check), `bpy_jobs/preview_shot.py` (Workbench preview), and
  `templates/shan_shen_bu_props/add_balls.py` (ballistic props found from the rig, with contact checks).

## 2. Text-to-motion models

**The licence wall** [V]:
- AMASS (amass.is.tue.mpg.de/license.html) allows only "non-commercial scientific research" and "prohibits the use of
  the Dataset to train methods/algorithms/neural networks/etc. for commercial use of any kind".
- SMPL (smpl.is.tue.mpg.de/modellicense.html) and SMPL-X have the same training ban. They also forbid "production of
  other artefacts for commercial purposes". A commercial licence is sold via Meshcapade.
- HumanML3D is built from AMASS + HumanAct12 (github.com/EricGuo5513/HumanML3D). BABEL's labels are under a
  "non-commercial scientific research" licence. Motion-X is "for non-commercial purposes" and includes AMASS.
- **Joint-only output does not clean the weights [I]:** the model does not run SMPL when it generates, but its weights
  were trained under both bans. The 22-joint skeleton layout itself is free to reuse.

| Model | Code | Weights / data | Verdict |
|---|---|---|---|
| **Kimodo-SOMA-RP / -SEED (NVIDIA)** | Apache-2.0 [V] | NVIDIA Open Model License, "ready for commercial use"; Bones Rigplay 700 h (RP) / BONES-SEED 288 h [V] | **OK** |
| Kimodo-SMPLX-RP | Apache-2.0 | NVIDIA R&D licence, non-commercial [V] | BLOCKED |
| HY-Motion 1.0 / Lite (Tencent, 2025-12) | Tencent community licence: "DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA" [V] | 3,000 h incl. web video -> SMPL-X (GVHMR) | EXCLUDED |
| MDM, T2M-GPT, MoMask, MotionGPT, MotionGPT3, LaMP, Light-T2M, MARDM | MIT / Apache [V] | HumanML3D/KIT | BLOCKED (data) |
| MotionDiffuse | S-Lab 1.0: commercial use means "contact the contributor(s)" [V] | HumanML3D/KIT | BLOCKED |
| MotionLCM / V2 | licence "does not allow commercial use" [V] | HumanML3D | BLOCKED |
| STMC (NVIDIA) | non-commercial licence [V] | AMASS | BLOCKED |
| BAMM | CC BY-NC-ND 4.0 [V] | HumanML3D | BLOCKED |
| MotionStreamer | data "solely for academic purposes" [V] | HumanML3D/BABEL | BLOCKED |
| Go to Zero / MotionMillion | Apache code; weights and data with no licence [V] | web video -> SMPL | UNCLEAR -> treat as blocked |
| Meta Motivo | CC BY-NC 4.0; a physics control policy without text [V] | - | BLOCKED / not T2M |
| ViMoGen (ICLR 2026) | release promised; data "research-friendly licenses" [C] | 30 mocap sets + Wan2.1 clips | UNCLEAR |

Other vendors: we found no open text-to-motion model from Lightricks, Stability, ByteDance or Alibaba (their releases
animate images or videos). Motorica is a commercial product.

**Kimodo in detail** (github.com/nv-tlabs/kimodo; huggingface.co/nvidia/Kimodo-SOMA-RP-v1):
- **Output:** root translation + joint rotations, 30 fps, at most 10 s (300 frames) [V]. The NPZ has `posed_joints
  [T,J,3]`, `local_rot_mats`, and `foot_contacts [T,4]` (heels and toes) [V]. Positions + contacts map straight onto our
  motion format (feet pins).
- **Control:** multi-prompt sequences, full-body pose keyframes, end-effector positions, 2D root paths and waypoints;
  `--duration`, `--seed`, `--num_samples`; a foot-skate cleanup post-process [V].
- **Size and memory:**
  - 282M parameters [V].
  - About 17 GB VRAM fully on the GPU; with `TEXT_ENCODER_DEVICE=cpu` it is "below 3 GB" [V].
  - The text encoder is LLM2Vec on the gated Meta-Llama-3-8B-Instruct [V, ComfyUI-Kimodo README]. That is about 16 GB
    of RAM in bf16 [I]: run it as a queue job while ComfyUI is idle (AGENTS.md section 3, RAM).
- **Limits** (card) [V]: "may include artifacts like foot skating", does not always follow the text, best at
  locomotion, gestures, combat, dancing and everyday actions, realistic motion only (no cartoon squash), one character,
  no awareness of scene objects.
- **Integrations:** ComfyUI-Kimodo (jtydhr88, Apache-2.0 [V]) has sampler, foot-skate post-process, NPZ and BVH (SOMA)
  export. Its FBX export needs a Mixamo character (Adobe terms) and `fbxsdkpy`. Skip the FBX export and use our own
  adapter.
- **Licence obligations:**
  - NVIDIA Open Model License: "Models are commercially usable"; NVIDIA claims no rights in the outputs. Keep the
    NVIDIA notice when redistributing. Rights end if you bypass the guardrails or start patent litigation [V via the
    survey agent].
  - Llama 3 licence: "prominently display 'Built with Meta Llama 3'" (CREDITS and the About page). A separate licence
    is needed above 700M MAU. No EU restriction for the text model [V, HF LICENSE].
  - Open risk [I]: the Bones-NVIDIA data terms are not public. We rely on NVIDIA's grant. We cannot train our own model
    on BONES-SEED (its licence §6 excludes motion-generating models) [V via the survey agent].
- **Quality:** we found no HumanML3D FID (Kimodo was not trained on it). NVIDIA reports its own benchmark [C]. Our
  quality is **unmeasured**: the first step is a blind A/B against a SAM 3D template.

## 3. Video-to-motion as a generator (our proven path)

Text -> a short video of ONE person doing the action, full body, still front camera at about 4 m, plain backdrop ->
`shot from-video` (SAM 3D Body -> mocap.py -> flash_step). Everything after the video exists and passed blind critics
(shan shen bu round 14: pose 7 vs 5.5 over DWPose [M, mocap.py docstring]).

- **Evidence that it works:**
  - ViMoGen (arXiv 2510.26794, ICLR 2026) built part of its 228k training set this way. It wrote over 20,000
    "MoCap-friendly" prompts (stable camera, full body, one person), generated 81-frame Wan2.1 clips, then filtered and
    fitted them. About 14,000 clips were kept, so roughly 70% survived [C].
  - Adding them raised generalizability (0.50 -> 0.55) but also foot sliding (0.0042 -> 0.0051) [C].
  - An earlier ECCV-W 2024 study (arXiv 2411.12831) found video models give plausible motion for common actions and
    fail on rare ones [C].
- **Quality of open video models on human motion:** HumanScore (arXiv 2604.20157, 13 generators) gives real video 94.3.
  Wan 2.2 scores 86.1, with good anatomy (94.0) but weaker kinematics (78.8) [C]. LTX was not tested. The best open
  model, HunyuanVideo 1.5 (91.1), is excluded by our Hunyuan rule. Known failure modes: unstable bone lengths, jitter,
  "unstable foot contacts", and over- or under-acting [C]. A NeurIPS 2025 paper names foot sliding, hyperextension and
  desynced limbs [C].
- **Our risks [I]:**
  1. The video model moves the camera. Write "locked-off tripod, static camera", and reject takes whose background
     moves (optical flow on the backdrop).
  2. Spins and crossed limbs swap sides: mocap.py already flags mirrored frames and kicks. Prefer ~30 deg of turning.
  3. Feet slide: mocap.py's contact pins and `settle` hide part of it.
  4. Clips are 5-10 s, so long shots need several takes or a loop.
  5. Cost: one LTX preview clip plus one SAM 3D job per take, several takes per beat.
- **Make it easy for SAM 3D [I]:**
  - A real-looking adult in fitted, plain, contrasting clothes; a light grey backdrop; the whole body and the feet in
    frame. Our mannequin as the actor (image-to-video from a Blender still) is untested: SAM 3D was trained on people.
  - Wan 2.2 TI2V-5B (Apache-2.0 [V], "a 5-second 720P video in under 9 minutes" on a 4090 [C]) would be a new download
    of about 10+ GB.
  - LTX-2.5 distilled is already installed. A plain text-to-video graph is new but small.
- **Licences:** LTX-2.x Community License (free under USD 10M annual revenue [V, docs/MODELS.md]), Wan 2.2 Apache-2.0,
  SAM 3D Body under the SAM License (royalty-free; no ITAR or military use) [V]. Its body model MHR is Apache-2.0 [V].
  Do not use MHR's SMPL conversion tools (SMPL licence) [I].

## 4. An open LLM as the shot planner

- **Nothing ready-made fits** [V, survey]:
  - BlenderMCP (MIT; anonymous telemetry ON by default, turn it off with `DISABLE_TELEMETRY=true`) and BlenderGPT (MIT,
    OpenAI API only) run free-form bpy.
  - SceneCraft (ICML 2024, no code licence), 3D-GPT (no licence file) and LayoutGPT (MIT code, 3D-FRONT data
    non-commercial) make static scenes.
  - Holodeck (Apache code) needs GPT-4o and Objaverse objects. Objaverse 1.0 has 25k CC-BY-NC and 52k CC-BY-NC-SA
    objects [V].
  - Not usable: LL3M (academic licence), MeshCoder (CC-BY-NC-SA), BlenderLLM (Apache, but its data is mostly
    GPT-4o-annotated; static CAD only).
- **LLM keyframe animation is unproven.** No benchmark measures LLM-written rig keyframes. BlenderGym (CVPR 2025)
  found even the best VLM systems "struggle with tasks relatively easy for human Blender users" [V, abstract].
- **What does work in research is splitting the job:**
  - The LLM writes high-level operators or a few symbolic keyframes, and a solver or motion model fills the motion.
  - Iterative Motion Editing (Goel et al., SIGGRAPH 2024): the LLM writes a program of editing operators that become
    keyframe constraints [C].
  - Re²MoGen (2026): coarse direction words beat raw coordinates [C].
  - FreeMotion: a GPT-4V keyframe designer plus interpolation [C].
  - **So the LLM should write our spec, not bpy.**
- **Local models** (all Apache-2.0 [V]): Qwen3.6-35B-A3B / Qwen3.5-35B-A3B, Qwen3-Coder-30B-A3B, gpt-oss-20b, Gemma 4
  26B-A4B, Mistral Small 3.2 24B.
  - On 8 GB VRAM + 32 GB RAM, the MoE models at Q4 run with llama.cpp `--n-cpu-moe`. Users report about 30-40 tok/s
    for the 30B-A3B class on a 3070 Ti [C]. gpt-oss-20b has an official 8 GB recipe [V, llama.cpp discussion 15396].
  - A 1-2k token spec takes about a minute [I]. The planner competes for RAM with ComfyUI: run it before or between
    GPU jobs.
  - Llama 4 is usable but needs attribution. DeepSeek-V3 (MIT) and GLM-4.6 (MIT) need a pod.
- **JSON output:** llama.cpp `json_schema` / `response_format`, Ollama `format` [V]. llama.cpp silently skips
  unsupported schema parts (`not`, if/then, nested `$ref`; min/max on integers only) and does not add the schema to the
  prompt [V]. So keep the schema flat with enums, repeat it in the prompt, and validate with Pydantic afterwards.
- **For the contest:** the agent that drives cinestudio (MCP/CLI) can be the planner. A `shot plan --spec JSON` command
  needs no local LLM. The local model is needed only for the app's text box.

## 5. Camera and set from text

- **Camera: rule-based** [I]. Learned text-to-camera models train on film or YouTube clips: E.T./DIRECTOR (MIT code,
  CondensedMovies data with no licence [V]) and GenDoP (DataDoP "intended exclusively for research purposes" [V]).
  CameraCtrl (Apache) and CamI2V (MIT) condition video models; they do not generate paths [V].
  - The spec picks from a vocabulary: `still`, `arc` (degrees, start at the front), `orbit`, `dolly_in/out`, `track`,
    `crane`, plus a framing (full/medium) and the distance. Our code solves the path and clamps it to < 0.6 deg/frame.
    If the clamp is hit, the shot needs more clips.
  - Two-character framing can follow toric space (Lino & Christie, TOG 2015 [V]), reimplemented from the paper.
- **Set** [V, survey]:
  - Our own builders (gym_set.py style: neutral grey blocks in `Set`) for a few named places (empty floor, gym, room,
    street block).
  - CC0 assets: Poly Haven (CC0; its ToS forbids scraping), Kenney, ambientCG, Quaternius, Blender's CC0 bundles.
  - Infinigen / Infinigen Indoors (BSD-3) is procedural with no learned weights, but slow and heavy: later, for rich
    sets.
  - BlenderKit "Royalty Free" assets must not be shipped inside templates.
  - The Blender MCP `generate_3d` tool may cost the user credits: do not use it here (no paid APIs).
- **Props:** the add_balls.py pattern (find the moment from the rig, ballistic path, contact check, `Props`
  collection). The spec names the prop, its event (dodge, kick, catch) and its colour.

## 6. Recommended architecture for cinestudio

```
text --(agent or local LLM, JSON schema)--> shot spec --validate (Pydantic + rules)-->
  per beat: motion source  [kimodo | pose (pose_frame/settle) | routine (funk_dance) | video (LTX->SAM 3D) | library]
  -> joints -> mocap motion file (adapter per source) -> stitch beats (settle/ease, hold)
  -> Blender build (flash_step animate + camera rules + set builder + props) -> scan_layers + rig_timeline
  -> automatic rule check (below) -> Workbench preview strip -> blind critic -> shot add (template)
```

**Spec sketch** (flat, enums only; frames at 24 fps):

```json
{"name": "dodge_and_bow", "seconds": 9.0, "fps": 24,
 "beats": [
  {"start": 0.0, "end": 4.5, "source": "kimodo", "text": "a person sidesteps left fast, then right, arms up to guard"},
  {"start": 4.5, "end": 7.5, "source": "kimodo", "text": "a person bows deeply and stands back up"},
  {"start": 7.5, "end": 9.0, "source": "pose", "pose": {"arms": "tada", "yaw_deg": 10, "head_tilt_deg": 8}, "hold": 0.5}],
 "camera": {"move": "arc", "degrees": 20, "distance_m": 4.0, "height_m": 1.45, "framing": "full"},
 "set": {"kind": "gym", "words": "a school gym with a polished wooden floor"},
 "props": [{"kind": "ball", "event": "dodge", "beat": 0, "words": "plain red rubber dodgeballs"}],
 "seed": 7}
```

**Have:** flash_step builder, mocap.py tools, pose_frame vocabulary, funk_dance routines, add_balls, gym_set,
scan_layers, rig_timeline, timeline events, preview_shot, `shot from-video`, and the critic prompt.
**New:**
1. The Kimodo runner (a separate venv and its CLI, as a queue job).
2. The SOMA-joints -> OpenPose-18 adapter. The nose, eyes and ears come from the head joint, so read the head yaw from
   the shoulders.
3. The beat stitcher.
4. The spec schema and validator.
5. The camera solver.
6. `templates/rule_check`.

**Automatic rule checks (AGENTS.md section 5).** We prototyped them on `rig_timeline` JSON
(`.cinestudio/gauntlet/shot_gen_research/check_rules.py`) [M]:
- The built-in `shan_shen_bu` passes the camera rule (still camera: 0.00 deg/frame) and the end hold (0 mm/frame in the
  last 12 frames). Facing stays within -15..+4 deg of the camera.
- The checker flags a raised-hands event at frames 209-222 that crosses the shared window 193-217 (the clap above the
  head). It also counts 20/46 "foot moves on the floor" frames, which in this dance are the deliberate slides.
- An old orbit_wave_360 backup fails the camera rule (1.22 deg/frame peak) and the end hold (a hand moving in the last
  12 frames).
- So the checks are cheap and they catch real issues.
- Still to add: per-bone scale = 1, joint acceleration peaks in screen px/frame² (from the camera), the head-vs-body
  yaw lag, and "turn starts within 30 deg of the camera".

**1-3 day MVP (before 10-30):**
- Day 1:
  - Install Kimodo in its own venv. This needs the hf token for Llama-3-8B, about 16 GB.
  - Generate 3 prompts on the 3070 with the text encoder on the CPU, and LOOK at the NPZ joints (a matplotlib strip).
  - Write the adapter -> motion file -> flash_step build -> Workbench preview.
- Day 2:
  - Spec + validator + beat stitcher (Kimodo multi-prompt or per-beat generations eased with `settle`; the end on
    `pose_frame` + hold).
  - Camera vocabulary (still/arc) and the set kinds we already have (floor, gym).
  - The rule checker as a gate. A `shot plan` CLI/MCP command.
- Day 3:
  - A blind critic on Workbench strips: a Kimodo template vs. a SAM 3D template vs. a code routine.
  - Then ONE preview LTX run of the best generated template, judged with the gauntlet critic (character >= 8, motion
    >= 8).
- Fallback if Kimodo disappoints on a beat: option 3 (video -> SAM 3D) for that beat.

**Later:** a local Qwen3.6-35B-A3B planner in the app; video-to-motion beats with take selection; a CMU library;
Infinigen sets; two-character specs (Kimodo makes one skeleton: generate each person and place them); prop events from
the spec.

**Risks:**
1. Kimodo's motion realism vs. readability from a front camera. Moves toward the camera do not read (AGENTS.md section
   5), so add a spec rule: actions sideways or in the picture plane.
2. Foot skating (Kimodo admits it): use our pins plus its post-process.
3. Beat joins: ease them over 12-24 frames, and keep joins out of the shared clip windows.
4. LLM specs that break the rules: the validator rejects them and asks again with the error.
5. Licence drift: re-read the Kimodo card at each update.

**Judging generated templates (gauntlet style):**
- A fresh critic sees anonymous Workbench strips (7 frames per clip plus rig.txt angles). It scores
  READABILITY (does the action read from the camera?), NATURALNESS (no pops, no skating), PROMPT MATCH and RULES
  (from the checker).
- Pass for a template: rule checker clean, READABILITY >= 8, NATURALNESS >= 7, PROMPT MATCH >= 8.
- Then the video judgment as in AGENTS.md 7.5.

## Sources

- Licences: AMASS https://amass.is.tue.mpg.de/license.html · SMPL https://smpl.is.tue.mpg.de/modellicense.html · SMPL-X
  https://smpl-x.is.tue.mpg.de/modellicense.html · HumanML3D https://github.com/EricGuo5513/HumanML3D · BABEL
  https://github.com/abhinanda-punnakkal/BABEL · Motion-X https://github.com/IDEA-Research/Motion-X · CMU
  https://mocap.cs.cmu.edu/faqs.php ("You may include this data in commercially-sold products, but you may not resell
  this data directly, even in converted form") · 100STYLE (CC BY 4.0) https://www.ianxmason.com/100style/ · LAFAN1
  (CC BY-NC-ND) https://github.com/ubisoft/ubisoft-laforge-animation-dataset · Bandai Namco (CC BY-NC)
  https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset
- Kimodo: https://github.com/nv-tlabs/kimodo · https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1 ·
  https://huggingface.co/nvidia/Kimodo-SOMA-SEED-v1 · https://github.com/jtydhr88/ComfyUI-Kimodo · NVIDIA Open Model
  License https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/ · Llama 3 licence
  https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/LICENSE
- HY-Motion: https://github.com/Tencent-Hunyuan/HY-Motion-1.0 · https://arxiv.org/abs/2512.23464
- Academic T2M repos: MDM https://github.com/GuyTevet/motion-diffusion-model · MotionDiffuse
  https://github.com/mingyuan-zhang/MotionDiffuse · T2M-GPT https://github.com/Mael-zys/T2M-GPT · MoMask
  https://github.com/EricGuo5513/momask-codes · MotionGPT https://github.com/OpenMotionLab/MotionGPT · MotionLCM
  https://github.com/Dai-Wenxun/MotionLCM · STMC https://github.com/nv-tlabs/stmc · BAMM https://github.com/exitudio/BAMM
  · MotionStreamer https://github.com/zju3dv/MotionStreamer · MotionMillion https://github.com/VankouF/MotionMillion-Codes
  · Meta Motivo https://github.com/facebookresearch/metamotivo
- Video to motion: SAM 3D Body https://github.com/facebookresearch/sam-3d-body, licence
  https://huggingface.co/facebook/sam-3d-body-dinov3/blob/main/LICENSE · MHR https://github.com/facebookresearch/MHR ·
  ViMoGen https://arxiv.org/abs/2510.26794 · Towards motion from video diffusion models https://arxiv.org/abs/2411.12831
  · HumanScore https://arxiv.org/abs/2604.20157 · Responsible realistic human motion (NeurIPS 2025)
  https://neurips.cc/virtual/2025/133230 · Wan2.2 https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B · LTX-2.x licence
  https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x
- LLM and Blender: BlenderMCP https://github.com/ahujasid/blender-mcp · BlenderGPT https://github.com/gd3kr/BlenderGPT ·
  SceneCraft https://arxiv.org/abs/2403.01248 · 3D-GPT https://github.com/Chuny1/3DGPT · LayoutGPT
  https://github.com/weixi-feng/LayoutGPT · Holodeck https://github.com/allenai/Holodeck · Objaverse
  https://huggingface.co/datasets/allenai/objaverse · Infinigen https://github.com/princeton-vl/infinigen · BlenderLLM
  https://github.com/FreedomIntelligence/BlenderLLM · LL3M https://github.com/threedle/ll3m · MeshCoder
  https://huggingface.co/InternRobotics/MeshCoder · BlenderGym https://arxiv.org/abs/2504.01786 · Iterative Motion
  Editing https://arxiv.org/abs/2312.11538 · FreeMotion https://arxiv.org/abs/2406.10740 · Re²MoGen
  https://arxiv.org/abs/2604.17807 · llama.cpp grammars https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
  · gpt-oss on 8 GB https://github.com/ggml-org/llama.cpp/discussions/15396 · Ollama structured outputs
  https://docs.ollama.com/capabilities/structured-outputs
- Camera and assets: DIRECTOR https://github.com/robincourant/DIRECTOR · GenDoP https://github.com/3DTopia/GenDoP ·
  DataDoP https://huggingface.co/datasets/Dubhe-zmc/DataDoP · CameraCtrl https://github.com/hehao13/CameraCtrl · CamI2V
  https://github.com/ZGCTroy/CamI2V · Toric space https://hal.archives-ouvertes.fr/hal-01142876 · Poly Haven
  https://polyhaven.com/license · Kenney https://kenney.nl/support · ambientCG https://docs.ambientcg.com/license/ ·
  Quaternius https://quaternius.com/faq.html · BlenderKit https://www.blenderkit.com/docs/licenses/
