"""Runs: one saved Blender shot + one character sheet -> one video.

A run lives in <home>/runs/<run_id>/ (created in a staging folder and renamed when complete):
  manifest.json   the frozen inputs (written once by new_run, never changed)
  inputs/         shot.blend (a Blender snapshot: external images/sounds/fonts packed), the sheet,
                  the optional first frame, and copies of the workflow graphs and bindings
  state.json      progress of each step; written ONLY by the worker that holds run.lock
  run.lock        OS lock held by the worker for its whole life (released if it dies)
  cancel.json     a cancel request; written only by `cancel` (never by the worker)
  events.jsonl    one JSON line per event (agents and the web app read it)
  logs/           Blender output
  work/           intermediate files (control clip, fitted sheet, styled sheet, ...)
  final.mp4       the result
Raw passes and control videos are cached in <home>/shots/<key>/passes, where <key> is a hash of the
snapshot .blend, the character collection, the rig, the Blender version and the pass code. A cache
is built in a staging folder under a per-shot lock and published by a rename.

Steps: passes -> controls -> sheet -> [restyle] -> clip -> final. A step that is done is never done
again. ComfyUI jobs: the job intent (key + output prefix) is saved BEFORE the submit; the GPU lock
(server.gpu_lock) is held from the model-family check until the job has ended; a lost job is looked
for in the proxy AND in ComfyUI's own queue and history (the proxy job id is the ComfyUI prompt id).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import blender, passes, refsheet, server, video, workqueue
import cv2
import numpy as np

from .locks import FileLock, LockBusy, is_locked
from .workflow import WORKFLOWS, ProfileWorkflow

ROOT = Path(__file__).resolve().parent.parent
from . import recipe as _recipe
from . import model_pairs as _pairs
from .model_pairs import graph_feature
PROFILE = _recipe.LTX_PROFILE      # video-to-video: LTX-2.5 full (dev); old runs keep their own
MAX_CLIP_FRAMES = 121            # one LTX clip; a longer shot is made of more clips that share one frame
CONTROL_KIND = "pose_env"
MAX_CONTROL_BYTES = 2 ** 31
SOUND_MODES = ("auto", "ltx")
# The Layout-To-Render LoRA makes SILENT audio unless the text describes the sound (A/B 2026-10-22, R4 clip 1, stage 1,
# seed 333; mean loudness: as is -75 dB, no look image -91 dB, a sound sentence -34 dB, the layout LoRA off -42 dB).
# Lightricks' prompt guide: describe the sound concretely and say what is absent.
SOUND_WORDS = re.compile(r"\b(sound|sounds|music|song|sings?|singing|speaks?|says|voice|footsteps|silence|silent)\b",
                         re.IGNORECASE)
SOUND_TEXT = ("The sound: a steady breeze and the clear ambient sounds of the place, with the rustle of cloth and "
              "light footsteps as the figure moves; no music, no one speaks.")
# (2026-10-22: "the QUIET natural ambience ..." gave -55.6 dB in the depth-ground test run; a concrete, audible sentence
# gave -34 dB in the A/B: say what is heard, not that it is quiet)
# Clip joins (2026-10-20): a clip that continues the one before starts OVERLAP frames earlier and is conditioned on the
# previous clip's last OVERLAP frames (latent in place, OVERLAP_STRENGTH); the stitch blends the overlap. One shared
# frame gave LTX the place but not the motion: the motion jumped at every join. 25 frames = 4 latent frames (the
# LTXVLoopingSampler default overlap is 24). Runs made before keep overlap 1 (their manifests have no field).
OVERLAP = 25
# the selected figure (a video template's SAM 2 mask) only chooses WHICH figure gets the skeleton ("control"); no mask
# ever limits where the character is drawn (user, 2026-10-21: "since mask is actually cutting parts of the character
# we should just delete it" - the soft/full figure masks, the scene pass and the panorama hold are gone)
FIGURE_MASK_MODES = ("off", "control")
REMOVED_OPTIONS = ("mask_background", "scene_pass", "scene_text", "background_hold")   # old manifests may have them
DANCE_SOURCE_DENOISE = 0.985     # dance templates (shot make-dance): 0.979 kept the white mannequin's arms and lost the
#                                  cape (r20261020-054713-b12b); 0.985 brought the cape, scarf and skin back with the same
#                                  moves (r20261020-205032-25d6). Orbits and other shots keep 0.979 (the most exact turn).
TEMPLATE_SOURCE_DENOISE = 0.985  # runs from a video template (gauntlet: 0.99 beat 0.979 in round 4, 0.985 beat 0.99 in round 8)
OVERLAP_STRENGTH = 0.8
LAYOUT_OVERLAP_STRENGTH = 0.5   # layout long shots (360 deg, 4 clips): 0.8 = smooth joins but later clips soft (sharpness
#                                 92 -> 31 -> 19 -> 14); 0 = sharp (100 -> 43 -> 55 -> 83) but the scene changed at the
#                                 joins (ghost tanks, a double leg); 0.5 = sharp enough (100 -> 49 -> 38 -> 61), smooth,
#                                 the scene continuous (runs P4b / P5 / P6, 2026-10-21)
CONTROL_KINDS = ("pose_env",)   # the AI always animates over the Blender reference: skeleton + set outlines (user, 2026-10-19)
SIZES = {"landscape": (768, 448), "portrait": (448, 768)}    # the Ingredients IC-LoRA training size
# variant: control only; v2v: video-to-video from the Blender render (the default: the render decides the motion)
QUALITY = {"preview": {"variant": "ref", "v2v": "v2vref"}, "final": {"variant": "refup", "v2v": "v2v"}}
VARIANTS = ("ref", "refup", "restyle", "v2v", "v2vref")                       # graphs a run may use (copied into the run)
DEFAULTS = {"quality": "preview", "seed": 333, "ingredients_strength": 1.0, "control_strength": 1.0,
            "style_lora_strength": 0.8, "restyle_keep": 0.5, "character_collection": "Mannequin",
            "rig": "MannequinRig", "character_name": None,
            "source_denoise": 0.979,    # video-to-video from the render: the start noise (0 < s <= 1); False = off
            "describe_facing": False,   # add to each clip's prompt where every character stands and faces (Blender)
            "control_kind": CONTROL_KIND,
            "body_type": "chibi",       # skeleton drawing: chibi (thick, ours) or standard (the DWPose node's)
            "sound": "auto",            # auto: the shot's own sound (Blender) when it has one, else LTX's; ltx: LTX's
            "overlap": OVERLAP,         # frames shared by two clips (1 = the old single-frame join)
            "negative": None,           # the clips' negative prompt (None = NEGATIVE, or TEMPLATE_NEGATIVE for a template)
            "steps": None,              # full model: sampling steps of stage 1 (None = DEV_STEPS)
            "refine_sigmas": None,      # stage 2 (refine) sigmas, e.g. "0.909375, 0.725, 0.421875, 0.0" (Lightricks' 2.5
            #                             distilled); None = the graph's
            "render_mode": None,        # None = "layout" for a Blender shot with a library character, else "v2v" (user,
            #                             2026-10-21: "yeah let's make it default"); Blender shots: "v2v" = video-to-video
            #                             from the render + skeleton and set edges
            #                             (union control); "layout" = Layout-To-Render: the render is the layout guide
            #                             and a LOOK image (the first frame with the character, Qwen-Image-Edit) sets the
            #                             look (LTX-2.5; the distilled gauntlet's winner 2026-10-21: anime, best motion)
            "look_extra": None,         # layout: words added to the look image's instruction (e.g. a garment length)
            "look_pin": True,           # layout: the look image keeps the room (cinestudio/look_pin.py, 2026-10-26):
            #                             no centre crop, the figure mask as a latent noise mask, a background gate
            #                             with up to 3 seeds. Old manifests without the key: the plain whole-frame edit
            "clip_frames": None,        # the longest clip (8k+1, 49..257; None = MAX_CLIP_FRAMES 121): longer clips =
            #                             fewer joins (a 360-frame shot: 4 clips at 121, 2 at 193)
            "relay": None,              # None = "latent" for layout runs, "off" otherwise. Layout long shots: "latent" = clip k > 1 continues from clip k-1's own LATENTS
            #                             (SaveLatent -> LoadLatent -> LTXVSelectLatents -> LTXVAddLatentGuide, per stage):
            #                             no decode / re-encode round trip at the joins; "off" = the pixel frames in place
            "audio_relay": None,        # None = "carry" with the latent relay: clip k starts with clip k-1's frozen audio tail
            "target_text": "short",     # the Target Description. DEFAULT "short" since 2026-10-26 (user: "let's go
            #                             with C's format"; docs/multiclip_prompting_research.md: identity by the
            #                             pictures, ~120-180 words) = "The character from the reference sheet." without
            #                             a second copy of the whole costume and without "The place: <set words>.": the
            #                             --video text (or the template's clip context) carries the 2-3 key marks and
            #                             the place; when the clip text names none of the marks, the description's key
            #                             marks (target_marks) are added. "full" = the old target (the costume again +
            #                             the place sentence). Manifests without the key are "full" (old runs).
            "relay_strength": 1.0,      # the latent guide's strength (0..1)
            "look_chain": "prev",       # layout clips k > 1: the frame -1 look image. "prev" (default since 2026-10-21,
            #                             user: "why is the clip start of the subsequent clips always the front facing
            #                             view") = clip k-1's frame at clip k's start: the right angle and background
            #                             (R3 clip 2: join 7.6 -> 1.8, the back kept, 3 seeds); "run" = the run's front
            #                             look (frame 1, the START angle: clip 2 copied its wall and turned to the front)
            "pose_strength": 0.0,       # layout clips: + the rig's skeleton as a guide in STAGE 1 (union control
            #                             IC-LoRA; 0 = off). The layout render cannot say where a blank mannequin head
            #                             faces: in back views the face turned to the camera (user, 2026-10-21)
            "pose_body": "standard",    # the skeleton drawing of the pose guide: standard (DWPose node) or chibi
            "clip_videos": None,        # {clip number: that clip's own ACTION text} (run new --clip-video K; frozen
            #                             with the template's clip_texts filled in). New runs (clip_join "context"):
            #                             clip k's text = the run text (SHARED CONTEXT: place, light, look) + the
            #                             clip context + clip k's actions; old runs: the clip's text replaced the run text
            "clip_context": None,       # words for EVERY clip (lasting states: "faces the camera the whole time");
            #                             None = the template's clip_context, "" = none
            "character_words": None,    # what "{character}" in the template's clip texts becomes (None = "the
            #                             character": the target part names the character's marks already)
            "clip_join": None,          # recorded: "context" (new runs, see clip_videos); None = an old run
            "clip_sources": None,       # recorded: {clip: "run" | "template" | "auto" | "none"} where its actions came from
            "prompt_warnings": None,    # recorded: clips without their own action text, actions in the shared text
            "clip_text": "manual",      # "auto" = each clip's text written from Blender (cinestudio/timeline.py): the
            #                             run's --video text is then the PLACE and light; the action, "feet planted"
            #                             and the camera's start/end views come from the rig per clip. Under test.
            "pose_stage2": False,       # the pose guide in stage 2 too (half-size guide: may fit in 8 GB; to measure)
            "adain_ref": None,          # the AdaIN reference: "look" = the clip's frame -1 image (the CHAINED, drifted
            #                             frame in look_chain prev); "clip1" = clip 1's saved stage-1 latent (Lightricks'
            #                             LTXVLoopingSampler normalizes every later tile toward the FIRST tile's output;
            #                             docs/ltx_long_drift_research.md, test B). None = "clip1" when the run has the
            #                             latent relay (layout runs: the DEFAULT since 2026-10-22, user: "let's go with
            #                             option b as default"), else "look"
            "chain_colour": "off",      # look_chain prev: "look" = the chained frame's dark costume pixels get the
            #                             colour cast of the run's look image (cinestudio/chain_colour.py; the masked3d
            #                             orbit drifted black -> brown clip by clip, 2026-10-22). Under test.
            "chain_light": "tone",      # DEFAULT "tone" since 2026-10-23 (P4: colourfulness 14.6/14.5/14.2 vs "look"'s
            #                             14.6/13.2/12.2; contrast held 22.3/22.5/22.1; drift +0.09 L) = "look" + the
            #                             background chroma (a*, b* scaled together) and contrast toward the first look's.
            #                             look_chain prev: "look" = the chained frame's background lightness (Lab L)
            #                             anchored to the run look's (a FIXED anchor; each clip came out ~1.3 L darker
            #                             than its look, so the scene darkened at every join). DEFAULT since 2026-10-23
            #                             (user: "a must fix"): with the exact frame_png, drift clip 3 - clip 1 -0.28
            #                             (F1b) and -0.29 (P1) against -1.18 (run A) and -3.39 (anime); "off" = as made
            "chain_grade": "local",     # the stitch: each clip after the first is graded so that the background of
            #                             its shared frames matches the clip before: "local" (DEFAULT) = a smooth
            #                             per-channel gain MAP (LTX re-lights a still scene region by region: a08c's
            #                             wall 24.6 -> 13.0 L, a banner 13.8 -> 19.0; with the map 24.3 and 15.2; a
            #                             moving camera falls back to "match"), "match" = one gain, "off" = as made
            #                             (old manifests without the key). LTX makes every clip ~0.3-0.6 L lighter or
            #                             darker than the still it is given (r20261027-005603-a08c: background L 10.53
            #                             -> 10.80 -> 11.40, each clip flat inside): the steps sit at the joins and the
            #                             input guards (chain_light) cannot remove them (user, 2026-10-27: "why does it
            #                             keep getting brighter or darker between clips")
            "chain_light_ref": "clip1",  # chain_light: the picture that the chained frame is held to. "clip1" (DEFAULT
            #                             since 2026-10-27) = clip 1's own first frame; "look" (old manifests without
            #                             the key) = the run's first look image. With "look" the scene got BRIGHTER at
            #                             every join (r20261026-202634-308e, 838f, a791): clip 1 came out ~8 L darker than
            #                             its look (L 11.6 vs 20.0), so every chained frame was pushed up by the full
            #                             +10% clamp, contrast and chroma +15%, and the video followed (grey 26.5 ->
            #                             28.0 -> 32.3; the floor 25.5 -> 29.4 -> 36)
            "chain_sharpen_target": None,  # chain_sharpen_mode match: the target = clip 1's median sharpness x this.
            #                             None = by quality: preview 1.5 (a stage-1 clip comes out at ~0.65x its look's
            #                             sharpness: P2 at 1.0 -> clips at 0.7x clip 1, P3 at 1.5 -> clip 1's level),
            #                             final 1.0 (the stage-2 refine keeps more: the final render at 1.5 gave clips 2-3
            #                             at 1.5x clip 1, r20261023-033850-7ab5, 2026-10-23)
            "chain_sharpen_mode": "match",   # "match" (DEFAULT since 2026-10-23) = about the smallest amount up to
            #                             chain_sharpen that brings the frame to clip 1's sharpness x the target; "fixed"
            #                             = the amount on every chained frame (it added up: clip 3 was 3-4x as sharp as
            #                             clip 1 in the forest runs; P3 kept 320-386 in all three clips)
            "chain_sharpen": 1.0,       # look_chain prev: an unsharp mask of that frame (0 = off; 1 = +100% detail):
            #                             the chained frame is a soft video frame and the clip copied its softness
            #                             (sharpness 35 without, 85 with)
            "adain": None,              # None = 0.3 with adain_ref clip1 (the tested B), 0.5 with "look" (the old default)
            #                             layout clips k > 1: LTXVAdainLatent toward the reference's latent statistics
            #                             after each stage (0 = off). 0.5 (2026-10-21): R4 clips 2/3/4 retaken at 0.5
            #                             were sharper (26 -> 65, 27 -> 71, 16 -> 147) at clip 1's saturation (73-77 vs
            #                             70) with the mask kept; 1.0 in every clip (R5) rose to 80 and turned clip 4's mask
            #                             into a black veil; 1.0 is a retake tool for one clip that faded badly
            "spacing": 1.5,             # several characters on a one-mannequin shot: the copies stand this far (m) apart
            "style_ref": None,          # the style library entry the run's words came from (styles.py)
            "style_warnings": None,     # assets that name another style (warnings only)
            "characters": None,         # EVERY library character of the run in slot order (2026-10-25, user: "if it
            #                             has three then three options"): [{"name", "collection", "rig"}], slot 1 =
            #                             Mannequin / MannequinRig, slot 2 = Mannequin2 / MannequinRig2, ... (at most
            #                             MAX_CHARACTERS); character_name / character_collection / rig mirror slot 1.
            #                             Several: layout runs only. Old manifests: character_name + "character2"
            #                             (run_characters reads both)
            "look_keyframes": "auto",   # layout: "auto" = one more look image at the middle of each clip whose body turns
            #                             (>= 30 deg from the camera there), made with the sheet view that matches the
            #                             body direction (the gauntlet's first pass, DL8: the costume holds through the
            #                             turn); "off" = the first look only
            "proxy": "auto",            # the character's proxy kit on the mannequin ("off" = the bare mannequin)
            "props": None,              # {prop pattern: words} over the template's prop words in the clip prompts
            #                             (2026-10-25, user: "map our own props to the object (for example the ball)");
            #                             only the given patterns (shots.check_prop_words); None = the template's words
            "prop_images": None,        # {prop pattern: picture} (2026-10-25, user: "generate a ball as a prop ... and
            #                             swap it out with the ball in the blender reference video"): each picture is
            #                             frozen (inputs/props/) and drawn into a PROP PANEL of the run's own copy of the
            #                             sheet (inputs/sheet_props.png, prop_sheet) + a "**Prop:**" line in the Reference
            #                             Sheet Description; recorded as {pattern: {"file", "source"}}. Not A/B-tested
            "prompt_format": "ing25",   # the clip text: "ing25" = the Ingredients 2.5 card's headings (### Reference
                                        # Sheet Description / ### Target Description); old runs (no field) keep 2.3
            "sound_text": None,         # layout: the clip text's sound sentence (None = SOUND_TEXT; "" = none)
            "look_scene": None,         # layout without a panorama: the place in words; a second Qwen edit puts it
            #                             into the look image (the grey Blender room -> the scene), keyframe looks off
            "look_text": None,          # layout: the character words of the look instruction instead of its saved
            #                             description (a description that contradicts the sheet wins over the sheet:
            #                             cloaked's "hem ... below the chest" gave a capelet; an exact text fixed it)
            "source_schedule": "lq",    # distilled v2v stage 1: "lq" = a linear-quadratic tail (our 2.3 tuning);
            #                             "grid" = the start noise, then Lightricks' distilled grid below it
            "guidance": None,           # full model: the video CFG of the guider (None = the graph's 3.0)
            "video_model": None,        # "distilled" (layout default: LTX-2.5 distilled, 8 steps, CFG 1) or "full" (the
            #                             dev model: 30 steps, MultimodalGuider CFG 3; slower, NOT yet A/B-tested
            #                             against distilled); video-to-video runs use the full model. Recorded resolved.
            "background": None,         # Blender shots: None = the template's own set; "text" = a 360 panorama made from
            #                             background_text (or the video text); "saved:NAME" = a panorama from the library
            "background_text": None,    # the place for the panorama (default: the video text)
            "background_image": None,   # background "plate": your own picture of the place (else made from the text)
            "background_seed": None,
            "background_ground": "depth",  # the panorama's ground in Blender: "depth" (default since 2026-10-22, user:
                                          # "new depth looks better") or "dome" (one radius); depth = MoGe-2 360
                                          # depth: every direction at its own distance; research 2026-10-22)    # the panorama's seed (default: the run's seed)
            "figure_mask": "off",       # video templates: off / control (the skeleton of the selected figure only)
            "overlap_strength": OVERLAP_STRENGTH}
# (0.979 + style LoRA 0.8: the best character and the full turn of a fast orbit, 2026-10-19)
# the official LTX-2.3 distilled schedule (8 steps); video-to-video starts part of the way down it
DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
RESTYLE_LORA_STRENGTH = 0.8     # tested: keeps the design of a drawn sheet (2026-10-18)
PASS_CODE = (ROOT / "bpy_jobs" / "render_passes.py", ROOT / "cinestudio" / "passes.py",
             ROOT / "bpy_jobs" / "render_people.py", ROOT / "cinestudio" / "pose_people.py")
PASS_FORMAT = 4                 # change when the pass files change meaning (2: hand points, camera angles;
                                # 3: controls drawn at the generation size; 4: hidden face points)
TERMINAL_RUN = {"done", "failed", "canceled", "unresolved", "interrupted", "cancel_unresolved", "paused"}
# what a retake of one clip may change (run retake RUN --clip K ...); the rest is the run's own manifest
LOOK_CHAINS = ("run", "prev", "anchor")   # anchor = the run look at frame -1 + the chained frame at frame 0
CHAIN_COLOURS = ("off", "look", "match")
TAKE_FIELDS = ("seed", "video", "style", "negative", "ingredients_strength", "style_lora", "style_lora_strength",
               "relay_strength", "adain", "keyframe", "end_anchor", "anchor_strength", "refine_sigmas", "look_chain",
               "chain_sharpen", "chain_sharpen_mode", "chain_sharpen_target", "chain_light", "chain_colour", "adain_ref", "look_image", "character", "keyframe_image",
               "keyframe_frame",
               "pose_strength", "pose_body", "pose_stage2", "keyframe_images")
GPU_WAIT = 6 * 3600             # how long a worker waits for another run's ComfyUI job


def _linear_quadratic(steps: int, threshold_noise: float = 0.025) -> list[float]:
    """ComfyUI's linear_quadratic schedule (comfy/samplers.py), the one LTX-2.3 distilled is tuned for."""
    linear_steps = steps // 2
    lin = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    diff = linear_steps - threshold_noise * steps
    q_steps = steps - linear_steps
    q_coef = diff / (linear_steps * q_steps ** 2)
    l_coef = threshold_noise / linear_steps - 2 * diff / (q_steps ** 2)
    const = q_coef * (linear_steps ** 2)
    quad = [q_coef * (i ** 2) + l_coef * i + const for i in range(linear_steps, steps)]
    return [1.0 - x for x in lin + quad + [1.0]]


def _copy_graph(name: str, dst: Path, sources: dict) -> None:
    """Copy a graph file (<graph>_api.json or <graph>.bind.json) from workflows/ into a run's folder. A graph the user
    chose a copy of (the Workflows view, cinestudio/user_workflows.py) comes from that copy; `sources` records each
    copy used ({graph: {copy, label, sha1, made_from_sha1}}: the manifest's graph_sources)."""
    src = WORKFLOWS / name
    if name.endswith("_api.json"):
        from . import user_workflows
        graph = name[:-len("_api.json")]
        try:
            picked = user_workflows.chosen(graph)
        except user_workflows.UserWorkflowError as exc:
            raise RunError(str(exc)) from exc
        if picked:
            src, sources[graph] = picked
    shutil.copy2(src, dst / name)


def _keep_copy_values(workflows_dir: Path, sources: dict) -> None:
    """After the tier freeze: the user's own values in a chosen graph copy win over the tier's node settings (the large
    tier's decode tiles); the model files stay the tier's (a copy cannot change an input the job sets)."""
    from . import user_workflows
    for graph, src in (sources or {}).items():
        p = workflows_dir / f"{graph}_api.json"
        try:
            copy_graph, _ = user_workflows.load_copy(graph, src["copy"])
            custom = user_workflows.custom_values(copy_graph, graph)
        except (user_workflows.UserWorkflowError, OSError, ValueError, KeyError):
            continue
        if not custom or not p.is_file():
            continue
        g = json.loads(p.read_text(encoding="utf-8"))
        for n, fields in custom.items():
            for k in fields:
                if n in g and k in (g[n].get("inputs") or {}):
                    g[n]["inputs"][k] = copy_graph[n]["inputs"][k]
        p.write_text(json.dumps(g, indent=1), encoding="utf-8")


def _freeze_tier(workflows_dir: Path, tier: str | None, files: dict | None = None,
                 settings: dict | None = None) -> dict:
    """Write the tier's model files into the run's copied graphs (only graphs with the full recipe take a tier).
    `files` (a parent run's frozen slots) wins over the tier's current list. Returns the manifest's "models"."""
    from . import models
    from . import settings as _settings
    from .workflow import freeze_models, freeze_settings
    if not files:                                  # no frozen files (a new run, or a parent from before the tiers)
        files = None
        tier = tier or _settings.get()["model_tier"]
    if files is None and tier:
        try:
            files = models.bundle(tier)
        except models.ModelsError as exc:
            raise RunError(str(exc)) from exc
    elif files and tier:                           # a fork: today's graphs may name a slot the parent never had
        try:                                       # (ltx25_distilled_lora, 2026-10-25): the tier's file fills it
            files = {**models.bundle(tier), **files}
        except models.ModelsError:
            pass
    try:
        used = freeze_models(workflows_dir, files or {})
    except ValueError as exc:
        raise RunError(str(exc)) from exc
    out = {"tier": models.canon(tier), "files": used}
    if settings is not None:                       # a fork: the parent's node settings ({} = a run from before them)
        written = freeze_settings(workflows_dir, settings)
    elif tier:                                     # the tier's node settings (larger decode tiles on a large GPU)
        try:
            written = freeze_settings(workflows_dir, models.graph_settings(tier))
        except models.ModelsError:
            written = {}
    else:
        written = {}
    if written:
        out["settings"] = written
    return out


def _style_mismatch(style_ref: str | None, opts: dict, blend) -> list[str]:
    """The run's assets that name ANOTHER style than the run (user rule 2026-10-24/25: one style for characters,
    background, template and props). Warnings only: the run still starts."""
    refs = {}
    try:
        from . import character as _chs
        for row in run_characters(opts):
            n = row["name"]
            r = _chs.load(n).get("style_ref")
            if r:
                refs[f"character {n}"] = r
    except Exception:  # noqa: BLE001
        pass
    bg = str(opts.get("background") or "")
    if bg.startswith("saved:") or opts.get("background_library"):
        try:
            from . import background as _bgs
            r = _bgs.load(opts.get("background_library") or bg.split(":", 1)[1]).get("style_ref")
            if r:
                refs["background"] = r
        except Exception:  # noqa: BLE001
            pass
    main = style_ref or next(iter(refs.values()), None)
    return [f"{what} uses the style {r!r}, the run uses {main!r}" for what, r in refs.items() if main and r != main]


def _library_kind(blend: Path) -> str | None:
    """The template kind ("dance", "blender") of a .blend in the template library, else None."""
    try:
        from . import shots
        rec = shots.library_record(blend)
        if rec:
            return rec.get("kind")
    except Exception:  # noqa: BLE001 - not a library template
        return None
    return None


def source_sigmas(start: float, steps: int = 8) -> str:
    """Sigmas for a video-to-video stage 1 that starts at noise `start` with `steps` steps: the tail of a longer
    linear_quadratic schedule whose first value is closest to `start` (the value itself is then used). About
    (1 - start) of the render is kept (LTX: x = sigma * noise + (1 - sigma) * source). Tested 2026-10-19 on a
    fast orbit: 0.975-0.99 keeps the camera's turn; 0.909 already kept the mannequin's look."""
    if not 0.0 < start <= 1.0:
        raise RunError("source denoise must be above 0 and at most 1 (the start noise; 0.98 keeps about 2% of the render)")
    best = min((_linear_quadratic(n)[-(steps + 1):] for n in range(steps, 400)), key=lambda t: abs(t[0] - start))
    tail = [x for x in best[1:] if x < start]          # strictly descending (a low start once went UP: Codex 10-22)
    if len(tail) < 2:                                  # a very low start: an even descent to 0
        tail = [start * (steps - i) / steps for i in range(1, steps)] + [0.0]
    sig = [start, *tail]
    assert all(a > b for a, b in zip(sig, sig[1:])), sig
    return ", ".join(f"{x:.5g}" for x in sig)


@graph_feature("latent relay: save", nodes=("relay_save_*",), adds=("SaveLatent",))
def add_relay_save(graph: dict, prefix1: str, prefix2: str) -> None:
    """Layout graphs: save the clip's video latent after each stage's guide crop (stage 1 at the generation size, stage 2
    at the output size) for the next clip's latent relay."""
    for crop, prefix in (("crop", prefix1), ("crop2", prefix2)):
        if crop in graph and prefix:
            graph[f"relay_save_{crop}"] = {"class_type": "SaveLatent",
                                           "inputs": {"samples": [crop, 2], "filename_prefix": prefix}}


def relay_latent(prefix: str, keep_dir: Path | None = None) -> str | None:
    """The latent file SaveLatent wrote under `prefix` (ComfyUI's output folder), copied into ComfyUI's input folder
    (where LoadLatent reads) and into `keep_dir` (the run's own copy: retakes need it later); returns the file name for
    LoadLatent, or None when no file was written."""
    from . import comfy
    out, inp = comfy.io_dir("output"), comfy.io_dir("input")
    found = sorted(out.glob(prefix.replace("/", os.sep) + "_*_.latent"), key=lambda q: q.stat().st_mtime) \
        if out.is_dir() else []
    name = Path(prefix).name + ".latent"
    if not found:                                      # a resumed run: the first pass copied and removed it already
        kept = keep_dir / name if keep_dir is not None else None
        if kept is None or not kept.is_file():
            return None
        inp.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(kept, inp / name)
        return name
    inp.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(found[-1], inp / name)
    if keep_dir is not None:
        keep_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(found[-1], keep_dir / name)
    for q in found:                                    # our own files only (the unique prefix): no leak in output/
        q.unlink(missing_ok=True)
    return name


@graph_feature("latent guides (relay, end anchor)", nodes=("relay_load_*", "relay_sel_*", "relay_guide_*", "end_load_*",
                                                     "end_sel_*", "end_guide_*"),
               adds=("LoadLatent", "LTXVSelectLatents", "LTXVAddLatentGuide"),
               edits=(("LTXVAddGuide", "positive"), ("LTXVAddGuide", "negative"), ("LTXVAddGuide", "latent"),
                      ("LTXAddVideoICLoRAGuide", "positive"), ("LTXAddVideoICLoRAGuide", "negative"),
                      ("LTXAddVideoICLoRAGuide", "latent")))
def add_latent_guides(graph: dict, tag: str, file1: str, file2: str, sel_start: int, sel_end: int, latent_idx: int,
                      strength: float) -> None:
    """Layout graphs: saved latent frames [sel_start..sel_end] (LTXVSelectLatents; negative = from the end) of a saved
    clip latent (file1: stage 1, file2: stage 2, ComfyUI input files) as an LTXVAddLatentGuide at `latent_idx` (in LATENT
    frames: index i >= 1 = picture frame 1 + 8 (i - 1)), in front of the first guide of each stage. Lightricks' node:
    the same as LTXV Add Guide "without the VAE decode/encode round trip"; the guide is cropped by LTXVCropGuides."""
    for st_, file, firsts in (("s1", file1, ("ref_guide", "g_lay")), ("s2", file2, ("g_lay2",))):
        first = next((f for f in firsts if f in graph), None)
        if first is None or not file:                   # preview clips have no stage-2 latent
            continue
        ld, sel, gd = f"{tag}_load_{st_}", f"{tag}_sel_{st_}", f"{tag}_guide_{st_}"
        graph[ld] = {"class_type": "LoadLatent", "inputs": {"latent": file}}
        graph[sel] = {"class_type": "LTXVSelectLatents",
                      "inputs": {"samples": [ld, 0], "start_index": sel_start, "end_index": sel_end}}
        pos, neg = graph[first]["inputs"]["positive"], graph[first]["inputs"]["negative"]
        graph[gd] = {"class_type": "LTXVAddLatentGuide",
                     "inputs": {"positive": pos, "negative": neg, "vae": ["vvae", 0],
                                "latent": graph[first]["inputs"]["latent"], "guiding_latent": [sel, 0],
                                "latent_idx": latent_idx, "strength": strength}}
        graph[first]["inputs"].update(positive=[gd, 0], negative=[gd, 1], latent=[gd, 2])


@graph_feature("preview: stage 1 only", edits=(("VAEDecodeTiled", "samples"), ("LTXVAudioVAEDecode", "samples")))
def to_stage1_only(graph: dict) -> None:
    """Preview quality for the two-stage layout graphs: the video and the audio are decoded from STAGE 1 (768x448);
    stage 2 is then not needed by any output, so ComfyUI does not run it; the stage-2 relay save is removed (it would
    make stage 2 run). Codex review 2026-10-21: layout runs ignored the preview choice."""
    graph["vdec"]["inputs"]["samples"] = graph["up"]["inputs"]["samples"]   # the stage-1 latent (after crop / AdaIN)
    if "adec" in graph and "sep" in graph:
        graph["adec"]["inputs"]["samples"] = ["sep", 1]
    graph.pop("relay_save_crop2", None)


@graph_feature("latent relay: guides", nodes=("relay_load_*", "relay_sel_*", "relay_guide_*"),
               adds=("LoadLatent", "LTXVSelectLatents", "LTXVAddLatentGuide"))
def add_relay_guides(graph: dict, file1: str, file2: str, strength: float, n_latent: int) -> None:
    """Clip k > 1: the last `n_latent` latent frames of clip k-1 at LATENT FRAME 1. Timing: clip k starts on clip k-1's
    frame N-overlap; LTX latent frame 0 holds 1 picture frame and every later one 8, so clip k-1's last (overlap-1)/8
    latent frames are clip k's latent frames 1..(overlap-1)/8 exactly (25 frames: 3 latents at index 1)."""
    add_latent_guides(graph, "relay", file1, file2, -n_latent, -1, 1, strength)


@graph_feature("audio carry", nodes=("audio_save", "audio_freeze2", "atail_load", "atail_frz", "av_carry"),
               adds=("SaveLatent", "LTXVFreezeLatent", "LoadLatent", "LTXVConcatAVLatent"),
               edits=(("LTXVConcatAVLatent", "audio_latent"), ("SamplerCustomAdvanced", "latent_image")))
def add_audio_carry(graph: dict, save_prefix: str | None, tail_file: str | None) -> None:
    """Layout graphs, the audio carry (Lightricks' ltx-pipelines 1.4.0 chunks: the audio tail is pinned like the video
    tail; docs/audio_consistency_research.md). Every clip: save its stage-1 AUDIO latent (`save_prefix`) and freeze the
    stage-1 audio in stage 2 (no re-noise: preview and final get the same sound, and only the stage-1 audio is relayed).
    Clip k > 1: `tail_file` (clip k-1's audio cut to the shared frames, a ComfyUI input file) is frozen at the start of
    this clip's audio; LTXVConcatAVLatent pads it to the clip's length with zeros that the model generates."""
    if save_prefix and "sep" in graph:
        graph["audio_save"] = {"class_type": "SaveLatent", "inputs": {"samples": ["sep", 1], "filename_prefix": save_prefix}}
    if "av2" in graph and "sep" in graph:
        graph["audio_freeze2"] = {"class_type": "LTXVFreezeLatent", "inputs": {"latent": ["sep", 1]}}
        graph["av2"]["inputs"]["audio_latent"] = ["audio_freeze2", 0]
    if tail_file and "av" in graph and "ks" in graph:
        graph["atail_load"] = {"class_type": "LoadLatent", "inputs": {"latent": tail_file}}
        graph["atail_frz"] = {"class_type": "LTXVFreezeLatent", "inputs": {"latent": ["atail_load", 0]}}
        graph["av_carry"] = {"class_type": "LTXVConcatAVLatent",
                             "inputs": {"video_latent": ["av", 0], "audio_latent": ["atail_frz", 0]}}
        graph["ks"]["inputs"]["latent_image"] = ["av_carry", 0]


@graph_feature("AdaIN", nodes=("adain_*",), adds=("LoadLatent", "ImageScale", "VAEEncode", "LTXVAdainLatent"),
               edits=(("LTXVLatentUpsampler", "samples"), ("VAEDecodeTiled", "samples"), ("SaveLatent", "samples")))
def add_adain(graph: dict, factor: float, width: int, height: int, ref_latent: str | None = None) -> None:
    """Layout graphs: after each stage's guide crop, LTXVAdainLatent moves the clip's video latent toward the latent
    statistics (per channel) of its reference: by default the clip's frame -1 look image (VAE-encoded; with look_chain
    "prev" that is the CHAINED frame of the clip before, which can itself have drifted: Codex review 2026-10-22), or
    `ref_latent` = clip 1's saved latent (adain_ref "clip1", Lightricks' looping-sampler reference). Changes `graph`."""
    if ref_latent:                                   # clip 1's saved latent (a whole video latent, per channel)
        graph["adain_ref"] = {"class_type": "LoadLatent", "inputs": {"latent": ref_latent}}
    else:
        graph["adain_img"] = {"class_type": "ImageScale",        # the VAE needs the clip's size (a 1328x800 look failed)
                              "inputs": {"image": ["look_load", 0], "upscale_method": "lanczos", "width": width,
                                         "height": height, "crop": "center"}}
        graph["adain_ref"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["adain_img", 0], "vae": ["vvae", 0]}}
    for crop, users in (("crop", ("up",)), ("crop2", ("vdec",))):
        if crop not in graph:
            continue
        node = f"adain_{crop}"
        graph[node] = {"class_type": "LTXVAdainLatent",
                       "inputs": {"latents": [crop, 2], "reference": ["adain_ref", 0], "factor": factor,
                                  "per_frame": False}}
        for u in (*users, f"relay_save_{crop}"):        # the relay must save the latent that is decoded (Codex 10-21)
            if u not in graph:
                continue
            for key, v in graph[u]["inputs"].items():
                if v == [crop, 2]:
                    graph[u]["inputs"][key] = [node, 0]


POSE_LORA = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"   # no 2.5 version; Lightricks' 2.5 example loads it


@graph_feature("pose guide", nodes=("pose_*", "g_pose*"),
               adds=("LoadVideo", "GetVideoComponents", "LoraLoaderModelOnly", "GetICLoRAParameters", "LTXVAddGuide"),
               edits=(("LTXVConcatAVLatent", "video_latent"), ("CFGGuider", "positive"), ("CFGGuider", "negative"),
                      ("CFGGuider", "model"), ("LTXVCropGuides", "positive"), ("LTXVCropGuides", "negative"),
                      ("LTXVCropGuides", "latent")))
def add_pose_guide(graph: dict, strength: float, lora_strength: float = 1.0, stages: tuple = (1,)) -> dict:
    """Layout graphs: the rig's skeleton video as one more guide (the union control IC-LoRA, a "ref0.5" model: its guide
    is encoded at half size, about a quarter of a full video guide's tokens) in the given stages (1 = generation size,
    2 = the upscale pass). It goes after the last guide of the stage (look, keyframes), and the LoRA goes on that stage's
    model only. Changes `graph` in place; returns {(node, input): "POSE"} for the upload."""
    graph["pose_load"] = {"class_type": "LoadVideo", "inputs": {"file": "POSE.mp4"}}
    graph["pose_frames"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["pose_load", 0]}}
    for st in stages:
        sfx = "" if st == 1 else "2"
        guider = f"guider{st}"
        if guider not in graph:
            continue
        gi = graph[guider]["inputs"]
        last = gi["positive"][0]
        lora, icp, guide = f"pose_lora{sfx}", f"pose_icp{sfx}", f"g_pose{sfx}"
        graph[lora] = {"class_type": "LoraLoaderModelOnly",
                       "inputs": {"model": gi["model"], "lora_name": POSE_LORA, "strength_model": float(lora_strength)}}
        graph[icp] = {"class_type": "GetICLoRAParameters", "inputs": {"iclora_model": [lora, 0]}}
        graph[guide] = {"class_type": "LTXVAddGuide",
                        "inputs": {"positive": [last, 0], "negative": [last, 1], "vae": graph["g_lay"]["inputs"]["vae"],
                                   "latent": [last, 2], "image": ["pose_frames", 0], "frame_idx": 0,
                                   "strength": float(strength), "iclora_parameters": [icp, 0]}}
        for c in (f"av{sfx}", guider, f"crop{sfx}"):
            for key, v in graph[c]["inputs"].items():
                if isinstance(v, list) and len(v) == 2 and v[0] == last:
                    graph[c]["inputs"][key] = [guide, v[1]]
        gi["model"] = [lora, 0]
    return {("pose_load", "file"): "POSE"}


@graph_feature("keyframe looks", nodes=("kf*_load", "g_kf*"), adds=("LoadImage", "LTXAddVideoICLoRAGuide"),
               edits=(("LTXVConcatAVLatent", "video_latent"), ("CFGGuider", "positive"), ("CFGGuider", "negative"),
                      ("LTXVCropGuides", "positive"), ("LTXVCropGuides", "negative"), ("LTXVCropGuides", "latent")))
def add_keyframe_guides(graph: dict, keys: list[tuple[int, str]]) -> dict:
    """Layout graphs: one more look guide per (frame, image) after the frame -1 look guide, in stage 1 and stage 2
    (Lightricks' Layout-To-Render card: "add a mid frame ... when the picture should hold through the shot"). Changes
    `graph` in place; returns the image files to upload {(node, input): path}."""
    import copy
    files = {}
    for look_node, consumers in (("g_look", ("av", "guider1", "crop")), ("g_look2", ("av2", "guider2", "crop2"))):
        if look_node not in graph:
            continue
        prev = look_node
        for i, (frame, image) in enumerate(keys):
            ld, gd = f"kf{i}_load", f"g_kf{i}_s{2 if look_node.endswith('2') else 1}"   # no kf1+"2" vs kf12 clash
            graph.setdefault(ld, {"class_type": "LoadImage", "inputs": {"image": "KEYFRAME.png"}})
            node = copy.deepcopy(graph[look_node])
            node["inputs"].update(positive=[prev, 0], negative=[prev, 1], latent=[prev, 2], image=[ld, 0],
                                  frame_idx=int(frame))
            graph[gd] = node
            files[(ld, "image")] = Path(image)
            prev = gd
        for c in consumers:
            for key, v in graph[c]["inputs"].items():
                if isinstance(v, list) and len(v) == 2 and v[0] == look_node:
                    graph[c]["inputs"][key] = [prev, v[1]]
    return files


def grid_source_sigmas(start: float) -> str:
    """Distilled video-to-video on Lightricks' own 8-step grid (DISTILLED_SIGMAS): `start`, then the grid values below
    it (fewer steps for a high start: 0.99 -> 0.99, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0)."""
    if not 0.0 < start <= 1.0:
        raise RunError("source denoise must be above 0 and at most 1")
    return ", ".join(f"{x:.6g}" for x in [start, *[x for x in DISTILLED_SIGMAS if x < start]])


# The values a layout clip binds (Run._layout_clip_job) and the features that change only files or text: registered
# for the distilled / full parity test (cinestudio/model_pairs.py; tests/unit/test_model_parity.py)
_pairs.register("clip format", "Run._layout_clip_job", binds=("width", "height", "frames", "fps", "seed"))
_pairs.register("prompt and negative", "Run._layout_clip_job", binds=("prompt", "negative"),
                note="the negative is bound in both; only the full model uses it (the distilled runs at CFG 1)")
_pairs.register("stage-1 schedule", "Run._layout_clip_job", binds=("sigmas",),
                note="each model's own values: DISTILLED_SIGMAS (8 steps) vs dev_sigmas (30 steps, LTXVScheduler shift)")
_pairs.register("style LoRA", "Run._layout_clip_job", binds=("style_lora", "style_lora_strength"),
                note="the optional lora_style / lora_style2 nodes (dropped without a LoRA)")
_pairs.register("continuation frames", "Run._layout_clip_job", binds=("no_first_frame", "first_strength"))
_pairs.register("ingredients sheet", "Run._layout_clip_job", binds=("ingredients_strength",))
_pairs.register("refine sigmas", "Run._layout_clip_job", binds=("refine_sigmas",),
                note="a run's own stage-2 sigmas; without it each graph keeps its model's own")
_pairs.register("guidance", "Run._layout_clip_job", binds=("video_cfg",), only="full",
                reason="the distilled model runs at CFG 1: new_run refuses guidance (and steps) for it")
_pairs.register("job output prefix", "Run._comfy", binds=("output_prefix",))
_pairs.register("look chain", "Run._layout_clip_job", graphless=True,
                note="look_chain prev, chain_light, chain_sharpen, chain_colour, a take's look image: the frame -1 look "
                     "FILE only (the same LoadImage in both graphs)")
_pairs.register("several characters", "new_run / Run._group_look", graphless=True,
                note="the group sheet, the N bodies and kits in the shot and the stitched look image: files only "
                     "(the same LoadImage / prompt inputs in both graphs)")
_pairs.register("sound", "video_prompt / Run.step_final", graphless=True,
                note="the sound sentence is prompt text; a song or the shot's sound replaces the audio with ffmpeg")
_pairs.register("tier graph settings", "_freeze_tier", graphless=True,
                note="written into the run's frozen graph files (both members) before any job: tested on the frozen pair")


# the panorama's ground in Blender (bpy_jobs/pano_world.py): "dome" = a floor disc to where the panorama's floor ends +
# a wall there up to 30 degrees (user, 2026-10-21: "the background looks distorted ... especially near the floor":
# the flat 200 m ground laid the roof's low walls and the tank bases flat; the dome stands them up)
PANO_GROUND = "dome"
LAYOUT_PROFILE = "ltx25"        # layout runs: LTX-2.5 DISTILLED (the IC-LoRAs are trained for it; ~5 min per final clip)
VIDEO_MODELS = _pairs.VIDEO_MODELS   # the run option "video model": distilled (LAYOUT_PROFILE) or full (ltx25dev)
PANO_ARGS = ("--ground", PANO_GROUND, "--wall-top", "30")
PANO_DEPTH_ARGS = ("--ground", "depth", "--depth-cols", "512", "--edge-ratio", "1.35")
BACKGROUND_GROUNDS = ("dome", "depth")


def pano_args(m: dict | None = None) -> tuple:
    return PANO_DEPTH_ARGS if (m or {}).get("background_ground") == "depth" else PANO_ARGS


def plate_world_tag() -> str:
    """The pass cache key's part for a backplate: a hash of bpy_jobs/plate_world.py (its geometry and material)."""
    return "plate" + hashlib.sha256((ROOT / "bpy_jobs" / "plate_world.py").read_bytes()).hexdigest()[:6]


def pano_world_tag(m: dict | None = None) -> str:
    """The pass cache key's part for the Blender world: the ground mode + a hash of pano_world.py and its arguments
    (a change of the dome code or its radius / wall height must not reuse passes rendered with the old geometry)."""
    args = pano_args(m)
    h = hashlib.sha256((ROOT / "bpy_jobs" / "pano_world.py").read_bytes() + " ".join(args).encode()).hexdigest()
    return f"{args[1]}{h[:6]}"
DEV_STEPS = 30                  # the full model's stage-1 steps (Lightricks' template; user: "let's do 30 steps")


def dev_sigmas(width: int, height: int, frames: int, steps: int = DEV_STEPS, max_shift: float = 2.05, base_shift: float = 0.95,
               terminal: float = 0.1) -> list[float]:
    """The LTX-2 schedule of the FULL (dev) model, as ComfyUI's LTXVScheduler makes it: the shift grows with the number
    of latent tokens (LTX-2 VAE: 32x32 pixels, 8 frames per token), stretched so the last non-zero sigma is `terminal`."""
    import math
    tokens = (width // 32) * (height // 32) * ((frames - 1) // 8 + 1)
    mm = (max_shift - base_shift) / (4096 - 1024)
    shift = tokens * mm + base_shift - mm * 1024
    lin = [1 - i / steps for i in range(steps + 1)]
    sig = [math.exp(shift) / (math.exp(shift) + (1 / t - 1)) if t > 0 else 0.0 for t in lin]
    nz = [x for x in sig if x > 0]
    scale = (1 - nz[-1]) / (1 - terminal)
    return [1 - (1 - x) / scale if x > 0 else 0.0 for x in sig]


def dev_source_sigmas(start: float, width: int, height: int, frames: int, steps: int = DEV_STEPS) -> str:
    """Video-to-video sigmas for the full model: `start`, then the rest of the dev schedule below it. About (1 - start)
    of the source is kept, as with the distilled model (x = sigma * noise + (1 - sigma) * source)."""
    if not 0.0 < start <= 1.0:
        raise RunError("source denoise must be above 0 and at most 1 (the start noise; 0.98 keeps about 2% of the render)")
    tail = [x for x in dev_sigmas(width, height, frames, steps) if x < start]
    return ", ".join(f"{x:.5g}" for x in [start, *tail])


class RunError(RuntimeError):
    pass


STAGE2_LORA_SLOT = "ltx25_distilled_lora"   # setup/models.json: the full model's stage-2 LoRA (feature full_stage2)


def full_model_missing(tier: str | None, quality: str = "final") -> str | None:
    """Why the FULL (dev) video model cannot run with `tier` (a file is not downloaded), or None. The dev transformer is
    the slot ltx25_model of setup/models.json, in the "video" feature of every tier; a FINAL run also needs the
    distilled LoRA of its stage 2 (slot ltx25_distilled_lora: feature video of the large tier, the optional feature
    full_stage2 of the 8gb tier). A preview runs stage 1 only (ComfyUI never loads the stage-2 LoRA)."""
    from . import models
    tier = models.canon(tier) or "8gb"
    slots = [(_pairs.FULL_MODEL_SLOT, "the full video model", "video")]
    if quality != "preview":
        feat = "full_stage2" if "full_stage2" in models.optional_features(tier) else "video"
        slots.append((STAGE2_LORA_SLOT, "the full video model's stage-2 LoRA (final quality)", feat))
    for slot, what, feat in slots:
        try:
            e = models.bundle(tier)[slot]
            p = models.path_of(e)
        except (models.ModelsError, KeyError) as exc:
            return f"{what} cannot be found: {exc}"
        if p.is_file() and p.stat().st_size == e["size"]:
            continue
        return (f"{what} ({e['file']}, {e['size'] / 1e9:.1f} GB) is not downloaded; get it with "
                f"`python -m cinestudio models download --tier {tier} --feature {feat}` (Settings > Quick setup), or "
                "use the distilled model" + (" or quality preview" if slot == STAGE2_LORA_SLOT else ""))
    return None


def stage2_recipe(workflows_dir: Path, profile: str | None, quality: str | None) -> dict | None:
    """Which stage 2 (the x2 upscale refine) a layout run's FROZEN graphs make: "distilled" (the distilled transformer,
    CFG 1), "distilled-lora" (the full model: the dev transformer + Lightricks' distilled LoRA, CFG 1, the distilled
    stage-2 sigmas; since 2026-10-25) or "guided" (the full model's graphs before that: MultimodalGuider CFG 3,
    6 steps 0.6..0). Preview runs make no stage 2 ("runs": False). None = not a layout run."""
    wd = Path(workflows_dir)
    for v in ("layouting", "layout"):
        api, bind = wd / f"{profile}_{v}_api.json", wd / f"{profile}_{v}.bind.json"
        if api.is_file() and bind.is_file():
            break
    else:
        return None
    g = _read_json(api)
    caps = _read_json(bind).get("capabilities") or {}
    recipe = caps.get("stage2") or ("guided" if (g.get("guider2") or {}).get("class_type") == "MultimodalGuider"
                                    else "distilled")
    out = {"recipe": recipe, "runs": quality != "preview",
           "sigmas": ((g.get("sigmas2") or {}).get("inputs") or {}).get("sigmas"),
           "sampler": ((g.get("sampler2") or {}).get("inputs") or {}).get("sampler_name"),
           "guidance": "CFG 1" if (g.get("guider2") or {}).get("class_type") == "CFGGuider" else "guided"}
    lora = (g.get("lora_dist") or {}).get("inputs")
    if lora:
        out["lora"] = {"file": lora.get("lora_name"), "strength": lora.get("strength_model")}
    return out


def retired(m: dict) -> str | None:
    """Why a run cannot run any more (its graphs were retired), or None. Old runs stay listed and their videos play."""
    if m.get("profile") in _recipe.RETIRED_PROFILES:
        return (f"this run used the retired LTX-2.3 graphs ({m['profile']}): it can be watched, but not resumed, "
                "retaken or forked; make a new run")
    return None


def check_runnable(run_id: str) -> None:
    """Refuse a start, retake or fork of a run whose graphs were retired (a clear message instead of a crash)."""
    why = retired(_read_json(run_dir(run_id) / "manifest.json"))     # run_dir refuses a missing run
    if why:
        raise RunError(why)


def home() -> Path:
    return Path(os.environ.get("CINESTUDIO_HOME", ROOT / ".cinestudio")).resolve()


def runs_dir() -> Path:
    return home() / "runs"


def run_dir(run_id: str) -> Path:
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        raise RunError(f"bad run id {run_id!r}")
    d = runs_dir() / run_id
    if not (d / "manifest.json").is_file():
        raise RunError(f"no run {run_id!r} in {runs_dir()}")
    return d


# ---------------------------------------------------------------- small file helpers

def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _is_connection_error(exc: BaseException) -> bool:
    """A transport error (server down or connection lost), not a failed job."""
    names = {"ConnectError", "ConnectTimeout", "ReadError", "ReadTimeout", "RemoteProtocolError", "ConnectionError",
             "ConnectionRefusedError", "ConnectionResetError", "URLError"}
    while exc is not None:
        if type(exc).__name__ in names:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def control_size(info: dict) -> tuple[int, int]:
    """The generation size for a shot's passes: the control video is drawn at this size."""
    return SIZES["portrait" if info["height"] > info["width"] else "landscape"]


def queue_label(m: dict) -> str:
    text = " ".join((m.get("prompt") or {}).get("video", "").split())
    who = " + ".join(dict.fromkeys(r["name"] for r in run_characters(m))) or m.get("character_name")
    return (f"{who} · " if who else "") + (text[:48] + "…" if len(text) > 48 else text or m.get("id", "video"))


def worker_alive(run_id: str) -> bool:
    return is_locked(run_dir(run_id) / "run.lock")


# ---------------------------------------------------------------- create, read

def snapshot_blend(src: Path, dst: Path, report: Path, allow_external: bool = False) -> dict:
    """Freeze the .blend through Blender (external images/sounds/fonts packed). Raises RunError with
    the list of files that cannot be frozen (linked libraries, movie clips, caches, missing files)."""
    args = ["--out", str(dst), "--report", str(report)] + (["--allow-external"] if allow_external else [])
    try:
        blender.run_job("snapshot.py", src, args, log=report.with_suffix(".log"))
    except blender.BlenderError as exc:
        if report.is_file():
            r = _read_json(report)
            if not r.get("ok", True):
                items = [f"{i['kind']} {i['name']} ({i['path']})" for i in r["missing"] + r["unsupported"]]
                raise RunError("the .blend uses files that cannot be frozen into the run: " + "; ".join(items[:12]) +
                               (" ..." if len(items) > 12 else "") + ". Pack or remove them, or use --allow-external "
                               "(then later changes to those files change the run)") from exc
        raise RunError(f"Blender could not snapshot the .blend: {exc}") from exc
    return _read_json(report)


def shot_source(m: dict) -> str:
    """What the run animates over: the .blend, or the video template (a template made from text)."""
    if m.get("template"):
        return f"template {m['template']['name']} ({m['template']['source']})"
    return m["blend"]["source"]


def template_mannequins(blend: str | Path) -> list[str]:
    """The Mannequin* layers a library template recorded (shot.json layers.scan: `shot layers`, or a template from a
    video with two people); [] when the .blend is no template or was never scanned (no Blender here)."""
    try:
        from . import shots
        rec = shots.library_record(blend) or {}
    except Exception:  # noqa: BLE001 - not a template
        return []
    return [c.get("collection") for c in ((rec.get("layers") or {}).get("scan") or {}).get("characters") or []
            if c.get("collection")]


def template_slots(blend: str | Path) -> list[dict]:
    """The character slots of a library template (shots.mannequin_slots: index, collection, side ...); [] for a
    .blend outside the library."""
    try:
        from . import shots
        rec = shots.library_record(blend)
    except Exception:  # noqa: BLE001 - not a template
        return []
    return shots.mannequin_slots(rec)["slots"] if rec else []


MAX_CHARACTERS = 4              # characters per run (2026-10-25): the group sheet's panels get small beyond it (GROUP_VIEWS)


def slot_names(i: int, collection: str = "Mannequin", rig: str = "MannequinRig") -> tuple[str, str]:
    """The collection and rig of slot i (0-based): the first slot's own names, then + "2", "3", ... (the names of
    bpy_jobs/merge_duo.py, duplicate_mannequin.py and the templates from a video)."""
    return (collection, rig) if i == 0 else (f"{collection}{i + 1}", f"{rig}{i + 1}")


def run_characters(m: dict) -> list[dict]:
    """The run's library characters in slot order: [{"name", "collection", "rig"}] ([] without a library character).
    New manifests keep the list ("characters"); old ones have character_name (+ character_collection, rig) and a
    second character in "character2" (2026-10-23 .. 10-25)."""
    rows = m.get("characters")
    if rows:
        return [{"name": r["name"], "collection": r.get("collection") or slot_names(i)[0],
                 "rig": r.get("rig") or slot_names(i)[1]} for i, r in enumerate(rows)]
    if not m.get("character_name"):
        return []
    out = [{"name": m["character_name"], "collection": m.get("character_collection") or "Mannequin",
            "rig": m.get("rig") or "MannequinRig"}]
    c2 = m.get("character2")
    if c2:
        out.append({"name": c2["name"], "collection": c2.get("collection") or "Mannequin2",
                    "rig": c2.get("rig") or "MannequinRig2"})
    return out


def run_views(m: dict) -> list[dict]:
    """Each character's frozen view files ({"front": "inputs/views/front.png", ...}) in slot order: the manifest's
    "views_all" (several characters), else "views" (+ the old "views2")."""
    if m.get("views_all"):
        return [dict(v or {}) for v in m["views_all"]]
    out = [dict(m.get("views") or {})]
    if m.get("views2"):
        out.append(dict(m["views2"]))
    return out


def run_group(m: dict) -> dict | None:
    """The several-character text of a run's prompt: {"figures": [{"name", "description", "marks", "height"}] in slot
    order, "order": [slot indices, left to right in the picture at frame 1], "views": [the sheet's views per
    character]}; an old two-character run ("duo": character 1 on the left) is read as the same. None for one."""
    p = m.get("prompt") or {}
    if p.get("group"):
        return p["group"]
    duo = p.get("duo")
    if not duo:
        return None
    names = duo.get("names") or [None, None]
    hs = duo.get("heights") or [1.0, 1.0]
    return {"figures": [{"name": names[i], "description": duo[k], "marks": None, "height": hs[i]}
                        for i, k in enumerate(("left", "right"))], "order": [0, 1], "views": list(DUO_VIEWS)}


def _run_slots(blend: str | Path, opts: dict) -> list[dict]:
    """Which character stands in which slot of the shot (the run details' "Character 1 (left)"): the template's own
    slots, or for several characters on copies of one mannequin the copies' places in the picture (duplicate_mannequin
    reports them; left to right in slot order before that)."""
    rows = run_characters(opts)
    tpl = {x["collection"]: x for x in template_slots(blend)}
    xs = ((opts.get("mannequins") or {}).get("screen_x") or []) if isinstance(opts.get("mannequins"), dict) else []
    out = []
    for i, row in enumerate(rows, 1):
        coll = row["collection"]
        if len(rows) == 1:
            side = (tpl.get(coll) or {}).get("side") if len(tpl) >= 1 else "centre"
        elif len(xs) == len(rows) and all(x is not None for x in xs):
            from . import shots
            side = shots._sides(xs)[i - 1]
        elif len(tpl) >= len(rows):
            side = (tpl.get(coll) or {}).get("side")
        else:
            side = "left" if i == 1 else "right" if i == len(rows) else "centre"
        out.append({"index": i, "collection": coll, "side": side, "character": row["name"]})
    return out


def _slots_refusal(blend: str | Path, collections: list[str], given: int) -> str:
    """The refusal of a run whose characters do not fill the template's slots, naming each slot."""
    from . import shots
    known = {s["collection"]: s for s in template_slots(blend)}
    slots = [{**(known.get(c) or {"side": None}), "index": i, "label": f"Character {i}", "collection": c}
             for i, c in enumerate(collections, 1)]
    names = "; ".join(f"{shots.slot_name(x)} = --character #{x['index']}" for x in slots)
    if len(collections) > MAX_CHARACTERS:
        return (f"the template has {len(collections)} characters ({names}); a video takes at most {MAX_CHARACTERS} "
                "characters for now")
    return (f"the template has {len(collections)} characters ({names}): give a character for each slot, in slot "
            f"order (--character once per slot; {given} given)")


def proxy_kit_args(inputs: Path, todo: list[dict], first: dict | None, out: Path) -> list[str]:
    """The arguments of bpy_jobs/proxy_kit.py for a run's kits (writes their JSON files into `inputs`): ONE kit =
    the classic --spec (inputs/proxy.json) with its --rig / --collection; several = --kits (inputs/proxy_kits.json,
    one Blender job for all). `todo` = [{"spec", "rig", "collection"}] of the characters that have a kit."""
    if first:
        _write_json(inputs / "proxy.json", first)
    if len(todo) == 1 and first:                       # the first character's kit alone
        return ["--spec", str(inputs / "proxy.json"), "--out", str(out), "--rig", todo[0]["rig"],
                "--collection", todo[0]["collection"]]
    _write_json(inputs / "proxy_kits.json", todo)
    return ["--kits", str(inputs / "proxy_kits.json"), "--out", str(out)]


def _characters_option(options: dict) -> list[dict]:
    """The run's characters from new_run's options, in slot order: `characters` (the list), else character_name (+ the
    old `character2`). Pops both list forms and sets character_name / character_collection / rig from slot 1.
    Collections and rigs left out are filled in later (slot_names)."""
    rows = options.pop("characters", None)
    c2 = options.pop("character2", None)
    if rows:
        if c2:
            raise RunError("give the characters as one list (--character repeated), not with character2")
        if not isinstance(rows, (list, tuple)):
            raise RunError("characters must be a list of characters in slot order")
        out = []
        for r in rows:
            r = {"name": r} if isinstance(r, str) else dict(r or {})
            nm = str(r.get("name") or "").strip()
            if not nm:
                raise RunError("every character slot needs a character")
            out.append({"name": nm, "collection": (str(r.get("collection") or "").strip() or None),
                        "rig": (str(r.get("rig") or "").strip() or None)})
        first = options.get("character_name")
        if first and first != out[0]["name"]:
            raise RunError(f"character_name {first!r} is not the first of the characters ({out[0]['name']!r})")
        options["character_name"] = out[0]["name"]
        if out[0]["collection"]:
            options["character_collection"] = out[0]["collection"]
        if out[0]["rig"]:
            options["rig"] = out[0]["rig"]
        return out
    if not options.get("character_name"):
        if c2:
            raise RunError("a second character needs a first one (--character)")
        return []
    out = [{"name": options["character_name"], "collection": None, "rig": None}]
    if c2:
        c2 = {"name": c2} if isinstance(c2, str) else dict(c2)
        out.append({"name": c2["name"], "collection": c2.get("collection") or None, "rig": c2.get("rig") or None})
    return out


def new_run(blend: str | Path | None, sheet: str | Path, character: str, video_text: str, style: str = "",
            style_lora: str | None = None, restyle: bool = False, first_frame: str | Path | None = None,
            allow_external: bool = False, control_video: str | Path | None = None, song: str | None = None,
            song_start: float = 0.0, template: str | None = None, model_tier: str | None = None,
            profile: str | None = None, **options: Any) -> dict:
    """Freeze the inputs of a run. `character` describes the sheet (panel by panel is best),
    `video_text` what happens in the shot, `style` the look. `first_frame` (optional image) becomes
    frame 0 of the clip, e.g. the front view of a character for a turntable run. `control_video` (optional video)
    replaces the control that the run draws from the rig (for example a pose detector's skeleton of several
    characters); it is fitted to the clip size like the drawn control. `song` (a song from the library, see
    cinestudio.song) becomes the video's sound from its second `song_start` on frame 1; it wins over the shot's own
    sound. Returns the manifest. Several library characters: `characters=[{"name", "collection"?, "rig"?}, ...]` in
    slot order (the first one is `character_name`); `character2={"name", ...}` is the old two-character form."""
    sheet = Path(sheet).resolve()
    rows_in = _characters_option(options)
    tpl_video = None
    tpl_rec = None                                     # the template's record (clip texts, length)
    if template:                                       # a video template (made from text): no .blend
        from . import shots
        rec = shots.load(template)
        tpl_rec = rec
        tpl_video = Path((rec.get("files") or {}).get("video") or "")
        if rec.get("kind") != "video" or rec.get("status") != "done" or not tpl_video.is_file():
            raise RunError(f"{template!r} is not a ready video template")
        blend = None
        options["describe_facing"] = False             # the facing words need Blender's angles
        if options.get("body_type") is None:           # the stand-in has adult proportions: the chibi drawing drew
            options["body_type"] = "standard"           # the character too large, legs cut off (gauntlet round 12)
        if options.get("source_denoise") is None:      # a photographic text template: keep less of it (blind A/B,
            options["source_denoise"] = TEMPLATE_SOURCE_DENOISE   # 0.99 beat 0.979: more of the scene redrawn)
    else:
        blend = Path(blend).resolve()
        if not blend.is_file() or blend.suffix.lower() != ".blend":
            raise RunError(f"not a saved .blend file: {blend}")
        if options.get("source_denoise") is None and _library_kind(blend) == "dance":
            options["source_denoise"] = DANCE_SOURCE_DENOISE
    if not sheet.is_file():
        raise RunError(f"character sheet not found: {sheet}")
    first = Path(first_frame).resolve() if first_frame else None
    if first and not first.is_file():
        raise RunError(f"first frame not found: {first}")
    ctl = Path(control_video).resolve() if control_video else None
    if ctl and not ctl.is_file():
        raise RunError(f"control video not found: {ctl}")
    if ctl and ctl.stat().st_size > MAX_CONTROL_BYTES:  # a guard against a wrong file, not a resource limit
        raise RunError(f"control video is larger than 2 GiB: {ctl}")
    song_src = None
    if song:
        from . import song as songs
        rec = songs.load(song)
        if rec.get("status") != "done" or not Path(rec.get("file") or "").is_file():
            raise RunError(f"song {song!r} is not ready ({rec.get('status')})")
        song_src = Path(rec["file"])
        if not 0 <= float(song_start) < float((rec.get("beats") or {}).get("seconds") or 1e9):
            raise RunError(f"song start {song_start} s is outside the song")
    from . import styles as _styles                    # the style library (2026-10-25): one set of words per look
    preset = options.pop("style_preset", None) or None
    style_ref = None
    if preset:
        if style.strip():
            raise RunError("give a style preset (--style-preset) or style words (--style), not both")
        try:
            st = _styles.load(preset)
        except _styles.StyleError as exc:
            raise RunError(str(exc)) from exc
        style, style_ref = _styles.text(preset), preset
        if style_lora is None and st.get("lora"):           # the style's own video LoRA
            style_lora = st["lora"]["name"]
            if options.get("style_lora_strength") is None:
                options["style_lora_strength"] = st["lora"]["strength"]
    elif not style.strip() and options.get("character_name"):
        try:                                              # no words given: the character's own style
            from . import character as _chs
            ref = _chs.load(options["character_name"]).get("style_ref")
            if ref:
                style, style_ref = _styles.text(ref), ref
        except Exception:  # noqa: BLE001 - an old character without a style: as before
            pass
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise RunError(f"unknown options: {sorted(unknown)}")
    opts = {**DEFAULTS, **{k: v for k, v in options.items() if v is not None}}
    opts["style_ref"] = style_ref
    opts["style_warnings"] = _style_mismatch(style_ref, opts, blend)
    if opts.get("source_denoise"):
        source_sigmas(float(opts["source_denoise"]))          # refuse a bad value before anything is copied
    if opts["quality"] not in QUALITY:
        raise RunError(f"quality must be one of {sorted(QUALITY)}")
    if opts["control_kind"] not in CONTROL_KINDS:
        raise RunError(f"control kind must be one of {list(CONTROL_KINDS)}")
    ov = int(opts["overlap"])
    if not (ov == 1 or (9 <= ov <= 57 and (ov - 1) % 8 == 0)):
        raise RunError("overlap must be 1 or 8k+1 frames between 9 and 57 (whole latent frames)")
    if not 0.0 <= float(opts["overlap_strength"]) <= 1.0:
        raise RunError("overlap strength must be 0..1")
    if opts["figure_mask"] not in FIGURE_MASK_MODES:
        raise RunError(f"figure mask must be one of {list(FIGURE_MASK_MODES)}")
    if opts.get("render_mode") is None:                 # the default: Layout-To-Render where it can run
        opts["render_mode"] = "layout" if (not template and opts.get("character_name")) else "v2v"
    if opts["render_mode"] == "layout" and options.get("overlap_strength") is None:
        opts["overlap_strength"] = LAYOUT_OVERLAP_STRENGTH
    if opts.get("video_model") is not None:              # the quality option: distilled (fast) or full (dev)
        if opts["video_model"] not in _pairs.VIDEO_MODELS:
            raise RunError(f"video model must be one of {list(_pairs.VIDEO_MODELS)}")
        want = _pairs.VIDEO_MODELS[opts["video_model"]]
        if profile is not None and profile != want:
            raise RunError(f"--video-model {opts['video_model']} means the profile {want}, not {profile}: give one")
        profile = want
    if opts["render_mode"] == "layout" and profile is None:
        profile = LAYOUT_PROFILE                         # Lightricks' recipe and our tests: the distilled model
    if opts.get("render_mode") not in ("v2v", "layout"):
        raise RunError("render mode must be v2v or layout")
    if opts["render_mode"] == "layout":
        if template:
            raise RunError("layout mode is for Blender shots (a video template has no layout render)")
        if not opts.get("character_name"):
            raise RunError("layout mode needs a character from the library (its front view makes the look image)")
        if not (profile or PROFILE).startswith("ltx25"):
            raise RunError("layout mode needs an LTX-2.5 profile (the Layout-To-Render IC-LoRA is 2.5 only)")
    cf = opts.get("clip_frames")
    if cf is not None and not (49 <= int(cf) <= 257 and (int(cf) - 1) % 8 == 0):
        raise RunError("clip frames must be 8k+1 between 49 and 257")
    if ov >= int(cf or MAX_CLIP_FRAMES):
        raise RunError("overlap must be shorter than a clip")
    if opts.get("relay") is None:                       # R1 (2026-10-21, seed 101): later clips 96 -> 55 -> 71 -> 89 with
        opts["relay"] = "latent" if opts["render_mode"] == "layout" else "off"   # the relay, 96 -> 19 -> 9 -> 9 without
    if opts.get("relay") not in ("off", "latent"):
        raise RunError("relay must be off or latent")
    if opts["relay"] == "latent" and ov < 9:
        raise RunError("the latent relay needs an overlap of at least 9 frames (one whole latent frame)")
    if opts.get("audio_relay") is None:                 # the sound continues across joins (2026-10-24): clip k-1's audio
        opts["audio_relay"] = "carry" if opts["relay"] == "latent" else "off"   # tail frozen at clip k's start
    if opts["audio_relay"] not in ("off", "carry"):
        raise RunError("audio relay must be off or carry")
    if opts["audio_relay"] == "carry" and opts["relay"] != "latent":
        raise RunError("the audio carry needs the latent relay (relay latent)")
    if opts.get("target_text", "full") not in TARGET_TEXTS:
        raise RunError("target text must be full or short")
    if not 0.0 < float(opts.get("relay_strength") or 0) <= 1.0:
        raise RunError("relay strength must be above 0 and at most 1")
    if opts.get("adain_ref") is None:                   # option B by default where clip 1's latent is saved
        opts["adain_ref"] = "clip1" if opts["relay"] == "latent" else "look"
    if opts.get("adain") is None:
        opts["adain"] = 0.3 if opts["adain_ref"] == "clip1" else 0.5
    if not 0.0 <= float(opts.get("adain") or 0) <= 1.0:
        raise RunError("adain must be 0..1")
    if opts.get("look_keyframes") not in ("auto", "off"):
        raise RunError("look keyframes must be auto or off")
    rows = []
    for i, r in enumerate(rows_in):                     # every slot's collection and rig (slot 1 = the options')
        coll, rig = slot_names(i, opts["character_collection"], opts["rig"])
        rows.append({"name": r["name"], "collection": coll if i == 0 else (r["collection"] or coll),
                     "rig": rig if i == 0 else (r["rig"] or rig)})
    if len(rows) > MAX_CHARACTERS:
        raise RunError(f"a video takes at most {MAX_CHARACTERS} characters, not {len(rows)}")
    if len({r["collection"] for r in rows}) != len(rows) or len({r["rig"] for r in rows}) != len(rows):
        raise RunError("each character needs its own mannequin collection and rig")
    opts["characters"] = rows or None
    if len(rows) > 1:
        if opts["render_mode"] != "layout" or template:
            raise RunError("several characters need a Blender shot in layout mode")
        if not 0.5 <= float(opts.get("spacing") or 1.5) <= 5.0:
            raise RunError("spacing must be 0.5..5 m")
        if options.get("look_keyframes") == "auto":     # the sheet keyframes paste ONE character's panel into the mask
            raise RunError("keyframe looks work with one character: leave them off with several characters")  # box
        opts["look_keyframes"] = "off"                  # of every figure, and the Qwen keyframes know only character 1
    if not template and blend is not None:              # a Blender shot: every library character brings its own
        from . import character as _chars              # mannequin body and proxy kit (user, 2026-10-24: "a character
        for nm in dict.fromkeys(r["name"] for r in rows):   # isnt ready to be used until a mannequin is set ... with
            try:                                            # its respective proxy kit"); the same character may stand
                _chars.require_video_ready(nm)              # in two slots (twins, a clone)
            except _chars.CharacterError as exc:
                raise RunError(str(exc)) from None
        layers_found = template_mannequins(blend)        # a template with several mannequins (people from a video):
        if rows and layers_found and (len(layers_found) > len(rows) or       # each must get a character (a bare one
                                      (len(layers_found) > 1 and len(rows) > len(layers_found))):   # stayed before);
            raise RunError(_slots_refusal(blend, layers_found, len(rows)))   # copies only of a ONE-mannequin shot
        from . import shots as _shots_p                   # mannequin before. The run's own words for the props:
        try:
            opts["props"] = _shots_p.check_prop_words(_shots_p.library_record(blend), opts.get("props")) or None
        except _shots_p.ShotError as exc:
            raise RunError(str(exc)) from None
        prop_pics = check_prop_images(_shots_p.library_record(blend), opts.get("prop_images"))   # frozen below
    elif opts.get("props") or opts.get("prop_images"):
        raise RunError("prop words and pictures need a Blender template with a Props layer")
    else:
        prop_pics = {}
    opts["prop_images"] = None
    if opts.get("clip_text", "manual") not in ("manual", "auto"):
        raise RunError("clip text must be manual or auto")
    if opts.get("chain_colour", "off") not in CHAIN_COLOURS:
        raise RunError(f"chain colour must be one of {CHAIN_COLOURS}")
    if opts.get("adain_ref", "look") not in ("look", "clip1"):
        raise RunError("adain ref must be look or clip1")
    if opts.get("adain_ref") == "clip1" and float(opts.get("adain") or 0) > 0 and opts.get("relay") != "latent":
        raise RunError("adain ref clip1 needs clip 1's saved latent: use it with the latent relay (layout runs)")
    if opts.get("look_chain", "run") not in LOOK_CHAINS:
        raise RunError("look chain must be run or prev")
    if not 0.0 <= float(opts.get("chain_sharpen") or 0) <= 3.0:
        raise RunError("chain sharpen must be 0..3")
    if opts.get("chain_sharpen_mode", "fixed") not in ("fixed", "match"):
        raise RunError("chain sharpen mode must be fixed or match")
    if not 0.5 <= float(opts.get("chain_sharpen_target") or 1.0) <= 3.0:
        raise RunError("chain sharpen target must be 0.5..3 (x clip 1's sharpness)")
    if opts.get("chain_light", "off") not in ("off", "look", "tone"):
        raise RunError("chain light must be off, look or tone")
    if not 0.0 <= float(opts.get("pose_strength") or 0) <= 1.0:
        raise RunError("pose strength must be 0..1")
    cvs = opts.get("clip_videos") or {}
    if not isinstance(cvs, dict) or not all(str(k).strip().isdigit() and int(k) >= 1 and isinstance(v, str)
                                            for k, v in cvs.items()):
        raise RunError("clip videos must map clip numbers (1, 2, 3 ...) to texts")
    opts["clip_videos"] = {str(int(k)): v.strip() for k, v in cvs.items() if v.strip()}
    run_clip_keys = set(opts["clip_videos"])           # the run's own clip texts (resolve_clip_texts)
    if opts.get("clip_context") is not None and not isinstance(opts["clip_context"], str):
        raise RunError("the clip context must be a text")
    if opts.get("pose_body", "standard") not in ("standard", "chibi"):
        raise RunError("pose body must be standard or chibi")
    if opts.get("source_schedule") not in ("lq", "grid"):
        raise RunError("source schedule must be lq or grid")
    prof_full = (profile or PROFILE).endswith("dev")
    if not prof_full and (opts.get("steps") is not None or opts.get("guidance") is not None):
        raise RunError("steps and guidance are for the full model; the distilled model has a fixed 8-step schedule at CFG 1")
    if opts.get("steps") is not None and not 8 <= int(opts["steps"]) <= 60:
        raise RunError("steps must be 8..60 (the full model's default is 30)")
    if opts.get("guidance") is not None and not 1.0 <= float(opts["guidance"]) <= 10.0:
        raise RunError("guidance must be 1..10 (the video CFG; default 3)")
    if opts["figure_mask"] != "off" and not template:
        raise RunError("a figure mask needs a video template (shot mask NAME makes it)")
    bg_mode = opts.get("background")
    bg_saved = None
    if bg_mode:
        if template:
            raise RunError("a background panorama is for Blender shots (a video template has its own scenery)")
        if bg_mode in ("text", "plate"):
            pass
        elif isinstance(bg_mode, str) and bg_mode.startswith("saved:"):
            from . import background as _background
            try:
                rec_bg = _background.load(bg_mode.split(":", 1)[1])
                if rec_bg["kind"] == "plate":              # a saved BACKPLATE: the plate path with its picture
                    if opts.get("background_image"):
                        raise RunError("a saved backplate is the picture already: leave out --background-image")
                    bg_mode = opts["background"] = "plate"
                    opts["background_image"] = rec_bg["file"]
                    opts["background_library"] = rec_bg["name"]
                    if not opts.get("background_text") and rec_bg.get("scene"):
                        opts["background_text"] = rec_bg["scene"]     # the clip text names the same place
                else:
                    bg_saved = _background.path(rec_bg["name"])
            except _background.BackgroundError as exc:
                raise RunError(str(exc)) from exc
        else:
            raise RunError("background must be 'text', 'saved:NAME' or 'plate' (or empty: the template's own set)")
        if not opts.get("source_denoise"):
            raise RunError("a background panorama needs video-to-video (follow the render)")
    if opts.get("background_image") and bg_mode != "plate":
        raise RunError("a background image is a backplate: use it with --background plate")
    if opts.get("background_image") and not Path(opts["background_image"]).is_file():
        raise RunError(f"no background image {opts['background_image']}")
    if opts.get("background_ground", "dome") not in BACKGROUND_GROUNDS:
        raise RunError(f"background ground must be one of {list(BACKGROUND_GROUNDS)}")
    if opts["sound"] not in SOUND_MODES:
        raise RunError(f"sound must be one of {list(SOUND_MODES)}")
    if opts["body_type"] not in passes.BODY_TYPES:
        raise RunError(f"body type must be one of {list(passes.BODY_TYPES)}")
    if restyle and not style_lora:
        raise RunError("restyle needs a style LoRA")
    if not character.strip() or not video_text.strip():
        raise RunError("character and video text are both needed")
    from . import settings as _settings
    options_tier = model_tier or _settings.get()["model_tier"]
    prof = profile or PROFILE
    if not (WORKFLOWS / f"{prof}_{VARIANTS[0]}_api.json").is_file():
        raise RunError(f"unknown graph profile {prof!r}")
    opts["video_model"] = _pairs.video_model_of(prof)
    if opts["render_mode"] == "layout" and opts["video_model"] == "full":
        why = full_model_missing(options_tier, opts.get("quality"))   # refuse now, not after the look image
        if why:
            raise RunError(why)
    run_id = time.strftime("r%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    stage = runs_dir() / f".staging-{run_id}"
    (stage / "inputs" / "workflows").mkdir(parents=True)
    try:
        loop_info = None                                   # a dance loop template: close the loop in the video
        layers = None                                      # the template's layer words (frozen into the run)
        if blend is not None:
            try:
                from . import shots
                rec = shots.library_record(blend)
                if rec:
                    tpl_rec = rec
                    plan = rec.get("plan") or {}
                    if plan.get("loop") and plan.get("loop_tail"):
                        loop_info = {"frames": int(plan["loop_frames"]), "tail": int(plan["loop_tail"])}
                    if (rec.get("layers") or {}).get("descriptions"):
                        layers = {"descriptions": rec["layers"]["descriptions"]}
            except Exception:  # noqa: BLE001 - not a template: no loop
                loop_info = None
        if tpl_video:
            shutil.copy2(tpl_video, stage / "inputs" / "template.mp4")
            if opts["figure_mask"] != "off":
                from . import shots as _shots
                fm = _shots.shot_dir(template) / "mask" / "mask.mp4"
                if not fm.is_file():
                    raise RunError(f"template {template!r} has no figure mask yet: select the figure first "
                                   f"(shot mask {template})")
                shutil.copy2(fm, stage / "inputs" / "figure_mask.mp4")
            snap = {"missing": [], "unsupported": [], "packed": []}
        else:
            snap = snapshot_blend(blend, stage / "inputs" / "shot.blend", stage / "inputs" / "snapshot.json", allow_external)
        mannequins = None                               # each character's OWN mannequin (its height) with the
        rows = opts.get("characters") or []             # template's motion, in the run's own copy of the shot
        several = len(rows) > 1
        if not tpl_video and rows:
            from . import character as _chars
            bodies = [_chars.mannequin_body(r["name"]) for r in rows]
            if several or any(bodies):                  # several characters, or a body of its own (proportions)
                placed = stage / "inputs" / "shot_mannequins.blend"
                log = stage / "inputs" / "mannequins.log"
                blender.run_job("duplicate_mannequin.py", stage / "inputs" / "shot.blend",
                                ["--out", str(placed), "--count", str(len(bodies)), "--bodies",
                                 json.dumps(bodies), "--collection", opts["character_collection"],
                                 "--rig", opts["rig"], "--names", json.dumps([[r["collection"], r["rig"]] for r in rows]),
                                 "--spacing", str(float(opts.get("spacing") or 1.5))], log=log)
                line = next((x for x in log.read_text(encoding="utf-8", errors="replace").splitlines()
                             if x.startswith("DUPLICATED ")), None)
                if line is None:
                    raise RunError(f"duplicate_mannequin.py gave no result; see {log}")
                info = json.loads(line[len("DUPLICATED "):])
                if info.get("unused"):
                    raise RunError(f"the shot has more mannequins than characters ({', '.join(info['unused'])} would "
                                   "stay a bare mannequin in the video): give a character for each (--character "
                                   "repeated)")
                os.replace(placed, stage / "inputs" / "shot.blend")
                mannequins = {"bodies": bodies, **info}
        kits = []                                       # each character's PROXY KIT on its own mannequin in the run's
        if not tpl_video and rows and opts.get("proxy", "auto") != "off":   # copy of the shot (bpy_jobs/proxy_kit.py:
            from . import character as _chars           # clay mask, hair... on the mannequin's bones), ONE Blender job
            for r in rows:
                try:
                    kits.append(_chars.load(r["name"]).get("proxy") or None)
                except Exception:  # noqa: BLE001 - no library character: no kit
                    kits.append(None)
            todo = [{"spec": k, "rig": r["rig"], "collection": r["collection"]} for k, r in zip(kits, rows) if k]
            if todo:
                kitted = stage / "inputs" / "shot_proxy.blend"
                blender.run_job("proxy_kit.py", stage / "inputs" / "shot.blend",
                                proxy_kit_args(stage / "inputs", todo, kits[0], kitted),
                                log=stage / "inputs" / "proxy_kit.log")
                os.replace(kitted, stage / "inputs" / "shot.blend")     # the pass key below hashes the kitted shot
        proxy = kits[0] if kits else None
        auto_texts = None
        if opts.get("clip_text") == "auto" and not tpl_video:   # clip texts from the rig (timeline.py)
            from . import timeline as _tl
            raw = _tl.blend_timeline(stage / "inputs" / "shot.blend", stage / "inputs" / "timeline.json", opts["rig"])
            auto_texts = _tl.run_clip_texts(raw, "", int(opts.get("clip_frames") or MAX_CLIP_FRAMES),   # the place
                                            int(opts.get("overlap", 1)))    # is the run's main text (clip_join context)
            opts["timeline_events"] = _tl.events(raw)   # kept apart: run -> template -> auto (resolve_clip_texts)
        from . import shots as _shots_c                 # one prompt per clip: the clip texts, frozen (AGENTS.md 7.2)
        n_frames = None                                 # the run's OWN copy of the shot decides its length (the
        if snap.get("frame_end") is not None and snap.get("frame_start") is not None:   # template's record may be
            n_frames = int(snap["frame_end"]) - int(snap["frame_start"]) + 1               # older: Codex 10-25)
        if not n_frames and tpl_rec:
            n_frames = _shots_c.shot_frames(tpl_rec)
        resolve_clip_texts(opts, tpl_rec, len(_shots_c.clip_plan(n_frames, opts.get("clip_frames"), opts.get("overlap"))),
                           run_clip_keys, video_text, auto=auto_texts)
        shutil.copy2(sheet, stage / "inputs" / ("sheet" + sheet.suffix.lower()))
        sheet_info = {}                                 # how the sheet was drawn (a prop sheet draws it again)
        if opts.get("character_name") and not several:
            try:
                from . import character as _chars
                crec = _chars.load(opts["character_name"])
                if crec.get("sheet_layout") == _chars.SHEET_LAYOUT and \
                        Path((crec.get("files") or {}).get("sheet") or "").resolve() == sheet:
                    sheet_info = {"layout": _chars.SHEET_LAYOUT, "scale": float(crec.get("sheet_scale") or 1.0)}
            except Exception:  # noqa: BLE001 - no library character: a sheet file of its own layout
                sheet_info = {}
        views_all = []                                  # each character's own view images, frozen with the run (sheet
        for i, r in enumerate(rows or [{"name": None}]):   # keyframes and look images; Codex review 2026-10-22)
            vs = {}
            try:
                from . import character as _chars
                crec = _chars.load(r["name"]) if r["name"] else {}
            except Exception:  # noqa: BLE001 - no library character: the sheet alone
                crec = {}
            if several and crec.get("status") != "ready":
                raise RunError(f"character {r['name']!r} is {crec.get('status')}, not ready")
            folder = "views" if i == 0 else f"views{i + 1}"
            for v in ("front", "right", "back", "left", "face"):
                src_v = Path((crec.get("files") or {}).get(v) or "")
                img_v = cv2.imread(str(src_v)) if src_v.is_file() else None
                if img_v is not None:                   # on pure black (an import on white would paste a white box
                    (stage / "inputs" / folder).mkdir(exist_ok=True)        # into a keyframe: Codex 10-22)
                    cv2.imwrite(str(stage / "inputs" / folder / f"{v}.png"), _chars.black_background(img_v))
                    vs[v] = f"inputs/{folder}/{v}.png"
            views_all.append(vs)
        views = views_all[0]
        group = None
        if several:                                     # several characters: ONE sheet with all of them (the views of
            from . import character as _chars           # group_views each, left to right as they stand in the picture
            from . import charsheet                     # at frame 1); the text names each by its place
            gviews = group_views(len(rows))
            missing = [r["name"] for r, vs in zip(rows, views_all) if not all(v in vs for v in gviews)]
            if missing:
                raise RunError(f"{len(rows)} characters need the {', '.join(gviews)} views of each "
                               f"({', '.join(missing)} {'has' if len(missing) == 1 else 'have'} not all)")
            order = picture_order((mannequins or {}).get("screen_x"), len(rows))
            # one scale for all: each character at its real height against the others (user, 2026-10-24: "standardize
            # the sheets so they can properly reflect the difference in heights")
            ratios = [_chars.height_ratio(_chars.ensure_height(_chars.load(r["name"]))) for r in rows]
            for f in (stage / "inputs").glob("sheet.*"):
                f.unlink()
            sheet = stage / "inputs" / "sheet.png"
            groups = [[charsheet.cutout(cv2.imread(str(stage / views_all[i][v]))) for v in gviews] for i in order]
            cv2.imwrite(str(sheet), charsheet.assemble_groups(groups, ratios=[ratios[i] for i in order]))
            d1 = character.split("on black: ", 1)[1].strip() if "on black: " in character else character.strip()
            descs = [sheet_text(d1, opts.get("look_text")).replace(SHEET_PANELS, "")] + \
                [(_chars.load(r["name"]).get("description") or "").strip() for r in rows[1:]]
            group = {"figures": [{"name": r["name"], "description": dsc, "marks": target_marks(dsc), "height": h}
                                 for r, dsc, h in zip(rows, descs, ratios)],
                     "order": order, "views": list(gviews)}
        if first:
            shutil.copy2(first, stage / "inputs" / ("first" + first.suffix.lower()))
        if ctl:
            shutil.copy2(ctl, stage / "inputs" / ("control" + ctl.suffix.lower()))
        if song_src:
            shutil.copy2(song_src, stage / "inputs" / ("song" + song_src.suffix.lower()))
        graph_sources: dict = {}                         # the user's graph copies this run uses (Workflows view)
        for v in VARIANTS:
            for suffix in ("_api.json", ".bind.json"):
                name = f"{prof}_{v}{suffix}"
                _copy_graph(name, stage / "inputs" / "workflows", graph_sources)
        if bg_mode == "text" or (bg_mode == "plate" and not opts.get("background_image")):
            for f in ("_api.json", ".bind.json"):                # the plate: the same model, its 360 LoRA at 0
                _copy_graph(f"qwenimage_pano{f}", stage / "inputs" / "workflows", graph_sources)
        if bg_mode == "plate" and opts.get("background_image"):
            img = Path(opts["background_image"])
            shutil.copy2(img, stage / "inputs" / ("plate" + img.suffix.lower()))
        if bg_mode and bg_mode != "plate" and opts.get("background_ground") == "depth":
            for f in ("_api.json", ".bind.json"):
                _copy_graph(f"moge_pano{f}", stage / "inputs" / "workflows", graph_sources)
        if opts["render_mode"] == "layout":                # the look image (Qwen edit) + the layout graphs
            for name in ("qwenedit_edit", f"{prof}_layout", f"{prof}_layouting"):
                for f in ("_api.json", ".bind.json"):
                    _copy_graph(f"{name}{f}", stage / "inputs" / "workflows", graph_sources)
        if bg_saved is not None:
            shutil.copy2(bg_saved, stage / "inputs" / "background.png")
        frozen = _freeze_tier(stage / "inputs" / "workflows", options_tier)
        _keep_copy_values(stage / "inputs" / "workflows", graph_sources)      # the user's copy values win
        graphs = {p.name: _sha256(p) for p in sorted((stage / "inputs" / "workflows").iterdir())}   # what runs
        bver = blender.check() if blend is not None else {"ok": True, "version": None}   # a video template: no Blender
        if not bver.get("ok"):
            raise RunError(f"Blender: {bver.get('error')}")
        if tpl_video:
            tpl_sha = _sha256(stage / "inputs" / "template.mp4")
            key_src = {"template": tpl_sha, "format": PASS_FORMAT,
                       "figure_mask": _sha256(stage / "inputs" / "figure_mask.mp4")
                       if opts["figure_mask"] != "off" else None,
                       "code": {p.name: _sha256(p) for p in (ROOT / "cinestudio" / "video_passes.py",
                                                             ROOT / "cinestudio" / "passes.py")}}
        else:
            blend_sha = _sha256(stage / "inputs" / "shot.blend")
            key_src = {"blend": blend_sha, "collection": opts["character_collection"], "rig": opts["rig"],
                       **({"collection2": rows[1]["collection"]} if len(rows) == 2 else {}),
                       **({"collections": [r["collection"] for r in rows]} if len(rows) > 2 else {}),
                       "blender": bver["version"], "format": PASS_FORMAT,
                       "code": {p.name: _sha256(p) for p in PASS_CODE}}
        manifest = {
            "models": frozen,
            **({"stage2": stage2_recipe(stage / "inputs" / "workflows", prof, opts.get("quality"))}
               if opts["render_mode"] == "layout" else {}),
            **({"layers": layers} if layers else {}),
            "version": 2, "id": run_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "profile": prof,
            "blend": None if tpl_video else {
                "source": str(blend), "file": "inputs/shot.blend", "sha256": blend_sha,
                "external_kept": len(snap["missing"]) + len(snap["unsupported"]), "packed": len(snap["packed"])},
            "template": {"name": template, "source": str(tpl_video), "file": "inputs/template.mp4",
                         "sha256": tpl_sha} if tpl_video else None,
            "pass_key": ("vt_" if tpl_video else "") + hashlib.sha256(json.dumps(key_src, sort_keys=True).encode()
                                                                      ).hexdigest()[:20],
            "pass_key_source": key_src,
            "sheet": {"source": f"{len(rows)} characters: {' + '.join(r['name'] for r in rows)}" if group
                      else str(sheet), "file": "inputs/sheet" + sheet.suffix.lower(),
                      **({"layout": "group"} if group else sheet_info)},
            "views": views, "sheet_fill_gaps": False, "proxy_kit": proxy, "mannequins": mannequins,
            **({"slots": _run_slots(blend, {**opts, "mannequins": mannequins})}
               if blend is not None and opts.get("character_name") else {}),
            **({"views_all": views_all, "proxy_kits": kits} if several else {}),
            "first_frame": {"source": str(first), "file": "inputs/first" + first.suffix.lower()} if first else None,
            "control_video": {"source": str(ctl), "file": "inputs/control" + ctl.suffix.lower()} if ctl else None,
            "loop": loop_info,
            "song": {"name": song, "source": str(song_src), "file": "inputs/song" + song_src.suffix.lower(),
                     "start": float(song_start)} if song_src else None,
            "workflows": graphs,
            "graph_sources": graph_sources,     # {graph: the user's copy} ({} = every graph built-in)
            "prompt": {"character": group_sheet_text(group) if group else sheet_text(character, opts.get("look_text")),
                       "video": video_text.strip(), "style": style.strip(), **({"group": group} if group else {})},
            "style_lora": style_lora or None, "restyle": bool(restyle), **opts,
        }
        if prop_pics:                                   # the prop pictures: frozen, drawn into the run's own sheet
            manifest["prop_images"] = _freeze_prop_images(stage, prop_pics)
            manifest["sheet"].update(prop_sheet(stage, manifest))
        _write_json(stage / "manifest.json", manifest)
        _write_json(stage / "state.json", {"status": "new", "steps": {s: {"status": "pending"} for s in steps_of(manifest)},
                                           "updated": time.time()})
        os.replace(stage, runs_dir() / run_id)          # the run appears complete or not at all
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def steps_of(manifest: dict) -> list[str]:
    return (["background"] if manifest.get("background") else []) + ["passes", "controls", "sheet"] + (["restyle"] if manifest["restyle"] else []) + \
        (["look"] if manifest.get("render_mode") == "layout" else []) + ["clip", "final"]


def status(run_id: str) -> dict:
    d = run_dir(run_id)
    st = _read_json(d / "state.json")
    st["worker_alive"] = worker_alive(run_id)
    st["id"] = run_id
    st["cancel_requested"] = (d / "cancel.json").is_file()
    if (d / "final.mp4").is_file():
        st["final"] = str(d / "final.mp4")
    if (d / "scores").is_dir():                         # run score's side files, checked (not written back)
        raw = _read_json(d / "state.json")
        for k, rec in ((st["steps"].get("clip") or {}).get("parts") or {}).items():
            sc = stored_score(d, int(k), None, raw)
            if sc is not None:
                rec["score"] = sc
    return st


NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted"
# runs from a video template (an EMPTY place with one stand-in): a freely regenerated background invents villagers and
# runners (blind A/B 2026-10-20); a background noise mask stopped them only by freezing the background
TEMPLATE_NEGATIVE = NEGATIVE + ", other people, background figures, pedestrians, villagers, crowd, extra person"


SHEET_PANELS = ("Left panel: a close-up of the character's face. Next four panels: the same character in full body "
                "(front, right side, back, left side) on black: ")


def sheet_text(character: str, look_text: str | None) -> str:
    """The Reference-sheet part of the clip text. With a look text (the costume in exact words, given because the
    library description and the sheet disagree), the clip text takes it too: R3 (2026-10-21) had a mid-thigh poncho in
    the look but "reaches directly below the chest ... no cape" in the clip text, and every take drew a one-sided
    capelet (blind critic CHARACTER 5). The look text's sentences about image 1 / image 2 (Qwen's inputs) are left out."""
    if not (look_text or "").strip():
        return character.strip()
    keep = [x.strip() for x in re.split(r"(?<=[.;])\s+", look_text.strip())
            if x.strip() and "image 1" not in x and "image 2" not in x]
    return SHEET_PANELS + " ".join(keep)


DUO_VIEWS = ("front", "right", "back")
# The group sheet's views per character (2026-10-25, N characters): the panels share one row, so 4 views x N groups get
# small (at 1536 x 896 and three characters each body would be ~0.4 of the panel height). At most six full-body
# panels in all: two characters keep front, right side and back; three get front and back (the back has the most that
# the front cannot show); four get the front only. Not A/B-tested.
GROUP_VIEWS = {2: DUO_VIEWS, 3: ("front", "back"), 4: ("front",)}
VIEW_WORDS = {"front": "front", "right": "right side", "back": "back", "left": "left side"}


def group_views(n: int) -> tuple:
    """The sheet's views per character for n characters (GROUP_VIEWS)."""
    if not 2 <= n <= MAX_CHARACTERS:
        raise RunError(f"a group sheet is for 2 to {MAX_CHARACTERS} characters, not {n}")
    return GROUP_VIEWS[n]


def picture_order(xs, n: int) -> list[int]:
    """The slots left to right in the picture at frame 1 from each slot's screen x (duplicate_mannequin's report);
    slot order when any x is unknown (the copies are placed left to right in slot order)."""
    xs = list(xs or [])
    if len(xs) != n or any(x is None for x in xs):
        return list(range(n))
    return sorted(range(n), key=lambda i: (float(xs[i]), i))


def place_word(rank: int, n: int) -> str:
    """Where the rank-th figure from the left (0-based) of n stands: "on the left", "in the middle", "second from the
    left" ..."""
    if n == 1:
        return "in the middle"
    if rank == 0:
        return "on the left"
    if rank == n - 1:
        return "on the right"
    if n == 3:
        return "in the middle"
    return "second from the left" if rank == 1 else "second from the right"


ORDINALS = ("first", "second", "third", "fourth")
TARGET_MARK_WORDS = 24
TARGET_TEXTS = ("full", "short")
_MARK_STOP = {"with", "that", "which", "from", "over", "under", "above", "below", "beside", "their", "there", "these",
              "those", "covers", "whole", "shown", "wears", "worn", "like", "into", "onto", "both", "each", "every"}


def names_marks(text: str, description: str, need: int = 2) -> bool:
    """True when `text` already names the character by its marks: at least `need` of the content words of the
    description's key marks (target_marks) appear in it. A short Target (target_text "short") then adds no costume
    words; otherwise it adds the key marks, so a clip text with only actions still says who the character is."""
    marks = target_marks(description)
    if not marks:
        return True
    words = {w for w in re.findall(r"[a-z][a-z-]{3,}", marks.lower()) if w not in _MARK_STOP}
    low = (text or "").lower()
    return sum(1 for w in words if re.search(rf"\b{re.escape(w)}\b", low)) >= min(need, len(words))


def target_marks(description: str) -> str:
    """The few key visible marks of a character for the Target Description of a group of three or more (AGENTS.md
    7.2: repeat the character by 2-3 key marks, the same words every time): the description's first clause (up to
    the first ':' or ';' or sentence end), at most TARGET_MARK_WORDS words."""
    t = " ".join((description or "").split())
    if not t:
        return ""
    for sep in (": ", "; ", ". "):
        if sep in t:
            t = t.split(sep, 1)[0]
    t = t.rstrip(".")
    if len(t.split(" ")) > TARGET_MARK_WORDS:          # too long: up to the last comma that fits, else a hard cut
        cut = ""
        for part in t.split(", "):
            nxt = f"{cut}, {part}" if cut else part
            if len(nxt.split(" ")) > TARGET_MARK_WORDS:
                break
            cut = nxt
        if not cut:
            words = t.split(" ")[:TARGET_MARK_WORDS]
            while words and words[-1].lower() in ("a", "an", "the", "and", "with", "of", "in", "on", "to", "behind",
                                                  "like", "that"):
                words.pop()
            cut = " ".join(words)
        t = cut
    return t.rstrip(",")


def group_sheet_text(group: dict) -> str:
    """The Reference-sheet part for several characters: the sheet's groups of panels left to right (as they stand in
    the picture at frame 1), each with its description. Two characters with three views read exactly like the
    duo text before (2026-10-23)."""
    views = list(group.get("views") or DUO_VIEWS)
    k = len(views)
    vw = ", ".join(VIEW_WORDS.get(v, v) for v in views)
    parts = []
    for rank, slot in enumerate(group["order"]):
        f = group["figures"][slot]
        first, last = rank * k + 1, rank * k + k
        panels = f"Panels {first}-{last}" if k > 1 else f"Panel {first}"
        body = "in full body" if k > 1 else "in full body, from the front,"
        view_txt = f"({vw}) " if k > 1 else ""
        parts.append(f"**{panels} (Character {rank + 1}):** the {ORDINALS[rank]} character {body} {view_txt}on black: "
                     f"{(f.get('description') or '').strip().rstrip('.')}.")
    return " ".join(parts)


def _group_target(group: dict) -> str:
    """The Target Description's sentence naming each character by its place: two characters by their full
    descriptions (as the duo text), three or more by their key marks (target_marks)."""
    def low(t):
        t = (t or "").strip().rstrip(".")
        return t[:1].lower() + t[1:]
    n = len(group["order"])
    count = {2: "Two", 3: "Three", 4: "Four"}.get(n, str(n))
    bits = []
    for rank, slot in enumerate(group["order"]):
        f = group["figures"][slot]
        words = f.get("description") if n == 2 else (f.get("marks") or target_marks(f.get("description") or ""))
        bits.append(f"{place_word(rank, n)}, the {ORDINALS[rank]} character, {low(words)}")
    return f"{count} characters from the reference sheet side by side: " + "; ".join(bits) + ". "


def duo_sheet_text(duo: dict) -> str:
    """The Reference-sheet part of an old two-character run (prompt "duo"): the group text of its two characters."""
    return group_sheet_text(run_group({"prompt": {"duo": duo}}))


def figure_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """The runs of occupied columns of a figure mask (bool, H x W), left to right: [(x0, x1 exclusive)]."""
    occ = mask.any(0)
    out, x0 = [], None
    for x, v in enumerate(occ):
        if v and x0 is None:
            x0 = x
        elif not v and x0 is not None:
            out.append((x0, x))
            x0 = None
    if x0 is not None:
        out.append((x0, len(occ)))
    return out


def plan_figure_crops(mask: np.ndarray, n: int, xs: list | None = None) -> list[dict]:
    """The crop-and-stitch plan of an n-character look image (2026-10-25, N characters). `mask` = the first layout
    frame's mask of ALL characters (bool, H x W); `xs` = each slot's figure x in pixels (None: unknown, the figures
    are taken left to right in slot order). The figures are found as the runs of occupied columns: one run per figure
    when every figure stands apart. A run with a figure assigned by its x gets it; runs without one (a stray part)
    join the nearest group. Without xs: too many runs are merged at their narrowest gaps; too few: the widest runs
    take the extra figures. -> [{"box": (x0, y0, x1, y1), "slots": [slot, ...] left to right, "shared": bool}] left
    to right. A box = the group's extent + 20% of its width + 24 px at the sides, 8% of its height + 24 px above and
    below, never past the middle of the gap to a neighbouring group. A SHARED box (figures that overlap or touch, no
    gap between them) is the fallback: its figures are edited one after the other in the same crop, each edit naming
    its figure's place (Run._group_look)."""
    H, W = mask.shape
    segs = figure_segments(mask)
    if not segs:
        raise RunError("the first layout frame has no character in its mask")
    known = xs is not None and len(xs) == n and all(x is not None for x in xs)
    groups: list[list] = []                         # [[seg, ...], slots]
    if known:
        owner: dict[int, list] = {}
        for slot in range(n):
            x = float(xs[slot])
            inside = [i for i, (a, b) in enumerate(segs) if a <= x < b]
            i = inside[0] if inside else min(range(len(segs)), key=lambda j: min(abs(x - segs[j][0]),
                                                                                 abs(x - (segs[j][1] - 1))))
            owner.setdefault(i, []).append(slot)
        taken = sorted(owner)
        members = {i: [i] for i in taken}
        for i in range(len(segs)):                  # a run without a figure: to the nearest run with one
            if i not in owner:
                near = min(taken, key=lambda j: abs((segs[i][0] + segs[i][1]) / 2 - (segs[j][0] + segs[j][1]) / 2))
                members[near].append(i)
        for i in taken:
            groups.append([[segs[j] for j in sorted(members[i])],
                           sorted(owner[i], key=lambda sl: (float(xs[sl]), sl))])
    else:
        runs = [[sg] for sg in segs]
        while len(runs) > n:                        # merge at the narrowest gap
            gaps = [runs[i + 1][0][0] - runs[i][-1][1] for i in range(len(runs) - 1)]
            i = gaps.index(min(gaps))
            runs[i:i + 2] = [runs[i] + runs[i + 1]]
        counts = [1] * len(runs)
        while sum(counts) < n:                      # too few runs: the widest (per figure) takes one more
            widths = [(r[-1][1] - r[0][0]) / c for r, c in zip(runs, counts)]
            counts[widths.index(max(widths))] += 1
        slot = 0
        for r, c in zip(runs, counts):
            groups.append([r, list(range(slot, slot + c))])
            slot += c
    groups.sort(key=lambda g: g[0][0][0])
    ext = [(min(sg[0] for sg in g[0]), max(sg[1] for sg in g[0])) for g in groups]
    out = []
    for gi, (g, (a, b)) in enumerate(zip(groups, ext)):
        lo = 0 if gi == 0 else (ext[gi - 1][1] + a) // 2
        hi = W if gi == len(groups) - 1 else (b + ext[gi + 1][0]) // 2
        ys, xs_ = np.where(mask[:, a:b])
        x0, x1, y0, y1 = a, b, int(ys.min()), int(ys.max()) + 1
        mx, my = int(0.2 * (x1 - x0)) + 24, int(0.08 * (y1 - y0)) + 24
        out.append({"box": (max(lo, x0 - mx), max(0, y0 - my), min(hi, x1 + mx), min(H, y1 + my)),
                    "slots": g[1], "shared": len(g[1]) > 1, "columns": (a, b)})
    return out


PASTE_GROW = 0.035             # the figure paste: the group's mask grown by this part of the box's longer side


def figure_paste_alpha(mask: np.ndarray, box: tuple, columns: tuple) -> np.ndarray:
    """Where a crop's look edit is pasted back (0..1, the box's size): the figure mask of THIS group (its own columns
    only, not a neighbour's arm in the margin) grown by PASTE_GROW of the box and feathered, times a soft box edge.
    The rest of the box keeps the original frame: in trio_test (2026-10-25) Qwen redrew a crop's gym wall as a blue
    sky; a whole-box paste would have put a sky patch into the gym."""
    x0, y0, x1, y1 = box
    a, b = columns
    m = np.zeros(mask.shape, np.uint8)
    m[:, a:b] = mask[:, a:b]
    m = m[y0:y1, x0:x1]
    r = max(4, int(PASTE_GROW * max(x1 - x0, y1 - y0)))
    grown = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    soft = cv2.GaussianBlur(grown.astype(np.float32), (0, 0), max(1.0, r / 3))
    f = max(3, min(x1 - x0, y1 - y0) // 10)
    ramp = lambda q: np.minimum(np.minimum(np.arange(q), np.arange(q)[::-1]) / f, 1.0)  # noqa: E731
    edge = ramp(y1 - y0)[:, None] * ramp(x1 - x0)[None, :]
    return np.clip(np.minimum(soft, 1.0) * edge, 0.0, 1.0)


def stitch_look(full: np.ndarray, mask: np.ndarray, plan: list, crops: list) -> np.ndarray:
    """The look image: each crop's edit (`crops`, one picture per plan entry, any size) pasted back into the frame
    (figure_paste_alpha). `mask` = the characters' mask at the frame's size (bool)."""
    out = full.copy()
    for c, img in zip(plan, crops):
        x0, y0, x1, y1 = c["box"]
        res = cv2.resize(img, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32)
        al = figure_paste_alpha(mask, c["box"], c.get("columns") or (x0, x1))[..., None]
        part = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = np.clip(part * (1 - al) + res * al, 0, 255).astype(np.uint8)
    return out


def duo_boxes(mask: np.ndarray) -> tuple[dict, int]:
    """The crop box of each figure in a two-character mask (the old two-character form of plan_figure_crops):
    -> ({"left": (x0, y0, x1, y1), "right": ...}, split = the middle of the gap). Refuses touching figures."""
    if not mask.any():
        raise RunError("the first layout frame has no character in its mask")
    plan = plan_figure_crops(mask, 2)
    if len(plan) != 2:
        raise RunError("the two characters touch in the first layout frame: no gap to split the look image")
    return {"left": plan[0]["box"], "right": plan[1]["box"]}, plan[0]["box"][2]


# ---------------------------------------------------------------- one prompt per clip (2026-10-25)
# User: "each clip should have its own set of prompts ... whatever happened in clip 1 should not be described in clip
# 3"; "context that is needed throughout each clip should still be present but actions specific to that scene should
# not be used in other clips". Clip k's video text = the run's main text (the SHARED CONTEXT: who, place, light, look)
# + the clip context (lasting states for every clip, e.g. the template's "faces the camera the whole time") + clip k's
# ACTIONS (--clip-video K, else the template's clip_texts, else clip_text auto). AGENTS.md 7.2.
CHARACTER_SLOT = "{character}"
_SUBJECTS = {"character", "dancer", "it", "he", "she", "they", "then", "and", "which", "who", "ball", "dodgeball",
             "figure", "hero", "now", "again", "finally"}
_NOT_VERBS = {"this", "its", "is", "was", "has", "does", "always", "sideways", "towards", "afterwards", "times", "yes",
              "less", "unless", "across", "plus", "thus"}


_DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "our", "my", "your", "his", "her", "their", "its"}


def fill_character(text: str, words: str | None = None) -> str:
    """`text` with every "{character}" replaced by the run's words for its character (default "the character"), with
    a capital at the start of a sentence. Inside a sentence the words keep their own case (a name stays "Alice"); only
    a leading article or determiner gets a small letter ("The masked dancer" -> "the masked dancer")."""
    words = " ".join((words or "").split()) or "the character"
    if CHARACTER_SLOT not in (text or ""):
        return text or ""
    first = words.split(" ", 1)[0]
    lower = first.lower() in _DETERMINERS and (len(first) == 1 or not first.isupper())
    inner = words[0].lower() + words[1:] if lower else words
    out = re.sub(r"(^\s*|[.!?]\s+)\{character\}", lambda mo: mo.group(1) + words[0].upper() + words[1:], text)
    return out.replace(CHARACTER_SLOT, inner)


def _sentence(text: str) -> str:
    t = " ".join((text or "").split())
    if not t:
        return ""
    t = t[0].upper() + t[1:]
    return t if t[-1] in ".!?\"'" else t + "."


def clip_video_text(m: dict, k: int) -> str:
    """The video text of clip k. New runs (clip_join "context"): the run's main text + the clip context + clip k's
    actions. Old runs: clip k's own text replaced the run's text."""
    main = m["prompt"].get("video") or ""
    cv = (m.get("clip_videos") or {}).get(str(k)) or ""
    if m.get("clip_join") != "context":
        return cv or main
    ctx = m.get("clip_context") or ""
    if not cv.strip() and not ctx.strip():
        return main
    return " ".join(x for x in (_sentence(main), _sentence(ctx), _sentence(cv)) if x)


def clip_manifest(m: dict, k: int) -> dict:
    """The manifest as clip k sees it: its own video text in prompt.video (video_prompt builds the clip's prompt)."""
    return dict(m, prompt={**m["prompt"], "video": clip_video_text(m, k)})


# plural subjects take the base form ("they catch", "both kick"): a base form after one counts as a verb
_PLURAL_SUBJECTS = {"they", "both", "characters", "dancers", "figures", "heroes"}
_NOT_BASE_VERBS = {"are", "were", "have", "had", "do", "did", "can", "will", "would", "could", "should", "may",
                   "might", "must", "the", "a", "an", "and", "then", "now", "again", "also", "still", "not", "never",
                   "always", "together", "each", "other", "of", "in", "on", "at", "to", "with", "both", "stand", "stay",
                   "look", "face", "keep"}
# words before a verb-looking word that make it a noun ("safety catches", "the kicks"): such an occurrence in the
# main text is not an action (an advisory heuristic; Codex review 2026-10-25)
_NOUN_BEFORE = _DETERMINERS | {"safety", "door", "window", "latch", "lock", "belt", "seat", "box", "case", "wall",
                               "three", "four", "many", "some", "several", "few", "no", "of", "metal", "steel",
                               "wooden", "iron", "brass", "small", "big", "large", "little"}


def _stem(w: str) -> str:
    """The base of a verb form: kicks / kicking -> kick, catches -> catch, running -> run."""
    w = w.lower()
    if len(w) > 5 and w.endswith("ing"):
        b = w[:-3]
        return b[:-1] if len(b) >= 3 and b[-1] == b[-2] and b[-1] not in "lsz" else b
    if re.search(r"(ch|sh|ss|x|z)es$", w):
        return w[:-2]
    return w[:-1] if w.endswith("s") and not w.endswith("ss") else w


def _clip_verbs(text: str) -> set[str]:
    """The present-tense verbs of a clip text ("kicks", "catches"): a word ending in s after a subject-like word or an
    adverb, or a base form after a plural subject ("they catch", "both kick")."""
    toks = re.findall(r"[a-z][a-z'-]*", text.lower().replace(CHARACTER_SLOT, "character"))
    out = set()
    for prev, w in zip(["."] + toks, toks):
        if len(w) >= 4 and w.endswith("s") and not w.endswith("ss") and w not in _NOT_VERBS \
                and (prev in _SUBJECTS or prev.endswith("ly")):
            out.add(_stem(w))
        elif prev in _PLURAL_SUBJECTS and len(w) >= 3 and not w.endswith("s") and not w.endswith("ly") \
                and w not in _NOT_BASE_VERBS:
            out.add(_stem(w))
    return out


def clip_actions_in_text(main: str, actions: dict, n: int) -> list[str]:
    """The words of the main (shared) text that name an action the clip texts give to SOME clips only ("kicks",
    "catches"): those words reach every clip. [] when fewer than two clips have their own actions."""
    texts = [actions.get(str(k)) or "" for k in range(1, n + 1)]
    if sum(1 for t in texts if t.strip()) < 2:
        return []
    verbs = [_clip_verbs(t) for t in texts]
    specific = set().union(*verbs) - set.intersection(*verbs)
    seen, out = set(), []
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", main or "")
    for prev, w in zip([""] + words, words):
        s = _stem(w)
        if prev.lower() in _NOUN_BEFORE:                # "safety catches", "the kicks": a noun here
            continue
        if s in specific and s not in seen:
            seen.add(s)
            out.append(w.lower())
    return out


def prompt_warnings(n: int, sources: dict, main: str, actions: dict) -> list[str]:
    """What is wrong with a multi-clip run's texts (AGENTS.md 7.2): clips without their own actions, and actions of
    single clips in the main text (it goes into every clip)."""
    if n < 2:
        return []
    w = []
    none = [k for k in range(1, n + 1) if sources.get(str(k), "none") == "none"]
    if len(none) == n:
        w.append(f"all {n} clips share one action text: write each clip's text (--clip-video K, or the app's Text per "
                 "clip) or use a template with clip texts (shot clip-texts)")
    elif none:
        w.append(f"clip{'s' if len(none) > 1 else ''} {', '.join(map(str, none))} "
                 f"{'have' if len(none) > 1 else 'has'} no action text of its own: only the shared text")
    leaked = clip_actions_in_text(main, actions, n)
    if leaked:
        w.append(f"the main text names actions of single clips ({', '.join(leaked)}): it goes into every clip; keep it "
                 "to the shared context (who, place, light, look, lasting states) and put the actions into the clip "
                 "texts")
    return w


def resolve_clip_texts(opts: dict, rec: dict | None, n: int, run_keys: set, main: str,
                       auto: dict | None = None) -> None:
    """Freeze a new run's clip texts into `opts`: each clip's actions (the run's own, `run_keys`; else the template's
    clip_texts when the run keeps the template's clip split; else the `auto` texts written from the rig, clip_text
    auto), the clip context, where each clip's actions came from (clip_sources) and the prompt warnings. `n` = the
    run's clips (0 = unknown: no warnings). Clip texts in clip_videos that are not in `run_keys` count as auto texts
    (older callers merged them there)."""
    who = opts.get("character_words")
    cv = {k: fill_character(v, who) for k, v in (opts.get("clip_videos") or {}).items() if k in run_keys}
    src = {k: "run" for k in cv}
    auto = {**{k: v for k, v in (opts.get("clip_videos") or {}).items() if k not in run_keys},
            **{str(k): v for k, v in (auto or {}).items() if isinstance(v, str) and v.strip()}}
    warn = []
    texts = {str(k): v for k, v in ((rec or {}).get("clip_texts") or {}).items() if isinstance(v, str) and v.strip()}
    if texts and n:
        if int(opts.get("clip_frames") or MAX_CLIP_FRAMES) == MAX_CLIP_FRAMES and int(opts.get("overlap", OVERLAP)) == OVERLAP:
            for k in range(1, n + 1):
                if str(k) not in cv and texts.get(str(k)):
                    cv[str(k)], src[str(k)] = fill_character(texts[str(k)], who), "template"
        else:
            warn.append(f"the template's clip texts are written for {MAX_CLIP_FRAMES}-frame clips with {OVERLAP} shared "
                        "frames; this run cuts its clips differently: give each clip its own text (--clip-video K)")
    for k, v in auto.items():                           # the rig's texts fill only what is still empty
        if k not in cv:
            cv[k], src[k] = fill_character(v, who), "auto"
    ctx = opts.get("clip_context")
    if ctx is None:
        ctx = (rec or {}).get("clip_context") or ""
    opts["clip_context"] = fill_character(" ".join(str(ctx).split()), who)
    if n:
        extra = sorted((int(k) for k in cv if int(k) > n))
        if extra:
            warn.append(f"the shot has {n} clip{'s' if n != 1 else ''}: the text for clip "
                        f"{', '.join(map(str, extra))} is not used")
        for k in range(1, n + 1):
            src.setdefault(str(k), "none")
    opts["clip_join"] = "context"
    opts["clip_videos"] = cv
    opts["clip_sources"] = dict(sorted(src.items(), key=lambda kv: int(kv[0]))) or None
    opts["prompt_warnings"] = (warn + prompt_warnings(n, src, main, cv)) or None


def video_prompt(m: dict) -> str:
    """The text prompt of a clip (without the optional facing text): the sheet's description, the style and the
    shot's text."""
    p = m["prompt"]
    style = (p["style"].rstrip(".") + ". ") if p["style"] else ""
    short = m.get("target_text") == "short"             # identity by the pictures: no repeated costume, no place line
    place = "" if short else background_sentence(m)
    group = run_group(m)
    if group:                                           # several characters (an old run: "duo")
        head = "### Reference Sheet Description " if m.get("prompt_format") == "ing25" else "Reference sheet: "
        mid = "### Target Description " if m.get("prompt_format") == "ing25" else "Generated video: "
        return (f"{head}{p['character']}{prop_sheet_sentence(m)}\n{mid}{style}{_group_target(group)}{p['video']}"
                f"{place}"
                f"{props_sentence(m)}{sound_sentence(m)} One continuous shot, no cuts.")
    if m.get("prompt_format") == "ing25":               # the Ingredients 2.5 card's format; the target repeats the
        who = p["character"].split("on black: ", 1)[1].strip() if "on black: " in p["character"] else ""   # character
        if short:                                       # the pictures carry the identity; the clip text its marks
            who = "" if names_marks(p["video"], who) else target_marks(who)
        target = (f"{style}The character from the reference sheet"
                  + (f", {who[0].lower()}{who[1:]}" if who else "")).rstrip(".") + ". "
        return (f"### Reference Sheet Description {p['character']}{prop_sheet_sentence(m)}\n"
                f"### Target Description {target}{p['video']}{place}{props_sentence(m)}"
                f"{sound_sentence(m)} "
                "One continuous shot, no cuts.")
    return (f"Reference sheet: {p['character']}{prop_sheet_sentence(m)}\n"
            f"Generated video: {style}The character from the reference sheet, with the same design and colours. "
            f"{p['video']}{background_sentence(m)}{props_sentence(m)}{sound_sentence(m)} One continuous shot, no cuts.")


def sound_sentence(m: dict) -> str:
    """Layout runs that may keep LTX's own sound get a sound sentence (see SOUND_TEXT): without one the Layout LoRA
    makes silence. Not for a run with a library song (its sound replaces LTX's) or with sound_text ""."""
    if m.get("render_mode") != "layout" or m.get("song") or m.get("sound", "auto") not in SOUND_MODES:
        return ""
    txt = m.get("sound_text")
    if txt is not None:                                 # the user's own words always win ("" = no sentence)
        txt = txt.strip()
        return (" " + (txt if txt.endswith(".") else txt + ".")) if txt else ""
    if SOUND_WORDS.search(m["prompt"].get("video") or ""):   # the video text already says the sound: no default
        return ""
    txt = SOUND_TEXT
    return " " + (txt if txt.endswith(".") else txt + ".")


def props_sentence(m: dict) -> str:
    """The template's props in the words of its layers (shot.json layers.descriptions.props, frozen into the run),
    with the run's own prop words (`props`) over them: " In the scene: plain red rubber dodgeballs." The same words as
    the background and the look texts (AGENTS.md 6, style consistency). Nothing when the video text already contains
    them."""
    from . import shots
    s = shots.props_sentence(shots.effective_layers(m.get("layers"), m.get("props")))
    if not s:
        return ""
    words = s[len("In the scene: "):-1].lower()
    if all(w.strip() in (m["prompt"].get("video") or "").lower() for w in words.split(";")):
        return ""
    return " " + s


def background_sentence(m: dict) -> str:
    """With a background made from words (a panorama, a backplate from text, or a saved backplate with its scene
    words: Fable review 2026-10-24), the clip's prompt names the same place, so the text and the picture agree
    (unless the video text already contains the place)."""
    place = (m.get("background_text") or "").strip().rstrip(".")
    if m.get("background") not in ("text", "plate") or not place:
        return ""
    if place.lower() in (m["prompt"].get("video") or "").lower():
        return ""
    return f" The place: {place[:1].lower()}{place[1:]}."


def _control_label(m: dict, control_full: str | None) -> str:
    """What the control video of a run really was, read from the file it used (older runs have no control_kind)."""
    if m.get("control_video"):
        return "a control video made outside the run (see Control video)"
    name = Path(control_full).name if control_full else ""
    several = Path(control_full).parent.name == "people" if control_full else False
    if several:
        return {"control.mp4": "several characters: detector skeletons on black",
                "control_pose_env.mp4": "several characters: detector skeletons + set edges in grey",
                "control_depth.mp4": "several characters: detector skeletons + depth"}.get(name, name)
    if not name:                       # older runs kept only the clip's slice; their control was always pose_env
        return "skeleton from the Blender rig + set edges in grey"
    return {"pose_env.mp4": "skeleton from the Blender rig (chibi drawing) + set edges in grey",
            "pose_env_standard.mp4": "skeleton from the Blender rig (standard drawing) + set edges in grey",
            "pose_depth.mp4": "skeleton from the Blender rig + depth (an old test option)"}.get(name, name)


def props_label(props: dict) -> str:
    """A run's own prop words for the run details: "Balls -> red and white beach balls; Photo -> ..."."""
    from . import shots
    return "; ".join(f"{shots.prop_label(pat, 2 if any(ch in pat for ch in '*?[') else 1)} -> {words}"
                     for pat, words in (props or {}).items())


# ---------------------------------------------------------------- prop pictures (2026-10-25)
# A run's own picture of a template prop (a ball made with Qwen in the characters' style) goes into a PROP PANEL of the
# run's copy of the sheet: the Ingredients 2.5 card reads the sheet "to match characters, props, and setting"
# (docs/style_consistency_research.md 2.7). The character's own sheet file is never changed.
PROP_IMAGE_MAX_BYTES = 40 * 1024 * 1024
PROP_IMAGE_SIDES = (32, 8192)                       # the shortest and the longest side a prop picture may have


def _read_prop(path: Path):
    """A prop picture as OpenCV reads it (alpha kept; any path, also one with non-ASCII letters), or None."""
    try:
        return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    except (OSError, ValueError):
        return None


def _check_prop_image(path: Path, pat: str) -> Path:
    from . import charsheet
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise RunError(f"no prop picture {p} (for {pat!r})")
    if p.stat().st_size > PROP_IMAGE_MAX_BYTES:
        raise RunError(f"the prop picture {p.name} is larger than {PROP_IMAGE_MAX_BYTES // 2**20} MB")
    img = _read_prop(p)
    if img is None:
        raise RunError(f"{p.name} is not a picture (png, jpg or webp)")
    lo, hi = PROP_IMAGE_SIDES
    if min(img.shape[:2]) < lo or max(img.shape[:2]) > hi:
        raise RunError(f"the prop picture {p.name} is {img.shape[1]}x{img.shape[0]}: {lo}..{hi} px a side")
    try:
        charsheet.prop_cutout(img)
    except ValueError as exc:
        raise RunError(f"{p.name}: {exc}") from None
    return p


def check_prop_images(rec: dict | None, images: dict | None) -> dict:
    """A run's prop pictures {pattern: picture path}, checked: each pattern is one of the template's prop groups or
    props (shots.check_prop_pattern), each file a readable picture with a prop in it (at most 40 MB, 32..8192 px a
    side), at most charsheet.MAX_PROP_PANELS pictures. Empty paths are left out. -> {pattern: resolved Path}."""
    from . import charsheet, shots
    if not images:
        return {}
    if not isinstance(images, dict):
        raise RunError('prop images must map a prop pattern to a picture, e.g. {"Ball_*": "ball.png"}')
    out = {}
    for pat, path in images.items():
        pat, path = str(pat or "").strip(), str(path or "").strip().strip('"')
        if not path:
            continue
        try:
            shots.check_prop_pattern(rec, pat)
        except shots.ShotError as exc:
            raise RunError(str(exc)) from None
        out[pat] = _check_prop_image(Path(path), pat)
    if len(out) > charsheet.MAX_PROP_PANELS:
        raise RunError(f"a run takes at most {charsheet.MAX_PROP_PANELS} prop pictures (the sheet's prop column has "
                       f"{charsheet.MAX_PROP_PANELS} cells), not {len(out)}")
    return out


def _freeze_prop_images(d: Path, images: dict) -> dict:
    """Copy each checked prop picture into the run as a PNG (inputs/props/prop_<hash>.png, alpha kept) ->
    {pattern: {"file": the run's path, "source": the picture given}}."""
    (d / "inputs" / "props").mkdir(parents=True, exist_ok=True)
    out = {}
    for pat, p in images.items():
        ok, buf = cv2.imencode(".png", _read_prop(p))
        if not ok:
            raise RunError(f"cannot copy the prop picture {p}")
        name = f"prop_{hashlib.sha1(buf.tobytes()).hexdigest()[:10]}.png"
        (d / "inputs" / "props" / name).write_bytes(buf.tobytes())
        out[pat] = {"file": f"inputs/props/{name}", "source": str(p)}
    return out


def prop_sheet(d: Path, m: dict) -> dict:
    """The run's sheet with its prop pictures, drawn from the run's FROZEN inputs (d = the run or its staging folder):
    inputs/sheet_props.png, the size of the run's sheet. A library character's panel sheet is drawn again from its
    frozen views with a prop column (charsheet.assemble_panels: the bodies keep the one scale of every sheet, the face
    panel gives way first); a group sheet (several characters) the same way (assemble_groups: the props give way
    first); any other sheet (an imported or restyled one)
    moves left for the column (charsheet.add_prop_column). Returns the manifest's sheet fields ({"props_file",
    "prop_places" {pattern: "bottom-right"...}, "prop_layout"}); {} (and no file) without prop pictures."""
    from . import character as _chars
    from . import charsheet
    out = d / "inputs" / "sheet_props.png"
    out.unlink(missing_ok=True)
    imgs = m.get("prop_images") or {}
    if not imgs:
        return {}
    try:
        props = [charsheet.prop_cutout(_read_prop(d / r["file"])) for r in imgs.values()]
    except ValueError as exc:
        raise RunError(f"a prop picture: {exc}") from None
    sh, views, lay = m["sheet"], m.get("views") or {}, {}
    base = cv2.imread(str(d / sh["file"]))
    if base is None:
        raise RunError(f"cannot read the run's sheet {sh['file']}")
    size = (base.shape[1], base.shape[0])

    def cut(rel):
        return charsheet.cutout(cv2.imread(str(d / rel)))
    group, vall = run_group(m), run_views(m)
    gv = list((group or {}).get("views") or DUO_VIEWS)
    if group and len(vall) == len(group["figures"]) and all(v in vs for vs in vall for v in gv):
        order = group["order"]
        img = charsheet.assemble_groups([[cut(vall[i][v]) for v in gv] for i in order], size=size,
                                        ratios=[group["figures"][i].get("height") or 1.0 for i in order], props=props,
                                        layout=lay)
        lay["drawn"] = "group"
    elif sh.get("layout") == _chars.SHEET_LAYOUT and all(v in views for v in ("face", *_chars.SHEET_PANELS)):
        img = _chars.sheet_image(cut(views["face"]), [cut(views[v]) for v in _chars.SHEET_PANELS],
                                 float(sh.get("scale") or 1.0), size=size, props=props, layout=lay)
        lay["drawn"] = _chars.SHEET_LAYOUT
    else:
        img = charsheet.add_prop_column(base, props, layout=lay)
        lay["drawn"] = "column"
    if not cv2.imwrite(str(out), img):
        raise RunError("cannot write the run's prop sheet")
    return {"props_file": "inputs/sheet_props.png", "prop_places": dict(zip(imgs, lay.get("prop_places") or [])),
            "prop_layout": lay}


def prop_sheet_sentence(m: dict) -> str:
    """The Reference Sheet Description's line for each prop panel: " **Prop:** matte violet dodgeball, shown in the
    bottom-right panel." The words: the run's own prop words, else the template's, else the group's name."""
    imgs = m.get("prop_images") or {}
    if not imgs:
        return ""
    from . import shots
    places = (m.get("sheet") or {}).get("prop_places") or {}
    words = (((shots.effective_layers(m.get("layers"), m.get("props")) or {}).get("descriptions") or {})
             .get("props") or {})
    lines = []
    for pat in imgs:
        w = (words.get(pat) or "").strip().rstrip(".") or shots.prop_label(pat, 1).lower()
        lines.append(f"**Prop:** {w}, shown in the {places.get(pat) or 'bottom-right'} panel.")
    return " " + " ".join(lines)


def prop_images_label(m: dict) -> str:
    """"Balls -> ball_s202.png" for the run details."""
    from . import shots
    return "; ".join(f"{shots.prop_label(pat, 2 if any(ch in pat for ch in '*?[') else 1)} -> "
                     f"{Path(r.get('source') or r.get('file') or '').name}"
                     for pat, r in (m.get("prop_images") or {}).items())


def _character_rows(m: dict) -> list:
    """The run details' character rows: "Character" alone, or one row per template slot ("Character 1 (left)")."""
    rows = run_characters(m)
    if len(rows) < 2:
        return [("Character", m.get("character_name") or "(a sheet file)")]
    sides = {x.get("index"): x.get("side") for x in m.get("slots") or []}
    out = []
    for i, r in enumerate(rows, 1):                     # older runs: the left figure was character 1
        sd = sides.get(i, "left" if i == 1 else "right" if i == len(rows) else None)
        out.append((f"Character {i}{f' ({sd})' if sd else ''}", r["name"]))
    return out


def video_model_label(m: dict) -> str:
    """The run's video model in words: "full (dev), 30 steps, guidance 3" or "distilled (8 steps, CFG 1)"."""
    if _pairs.video_model_of(m.get("profile")) == "full":
        return f"full (dev), {m.get('steps') or DEV_STEPS} steps, guidance {m.get('guidance') or 3.0:g}"
    return "distilled (8 steps, CFG 1)"


def stage2_label(m: dict) -> str | None:
    """The run's stage 2 in words (layout runs at final quality), e.g. "dev + distilled LoRA 0.8, CFG 1, 3 steps"."""
    s2 = m.get("stage2") or {}
    if not s2 or not s2.get("runs"):
        return None
    steps = max(0, len([x for x in str(s2.get("sigmas") or "").split(",") if x.strip()]) - 1)
    if s2.get("recipe") == "distilled-lora":
        lora = s2.get("lora") or {}
        head = f"dev + distilled LoRA {float(lora.get('strength') or 0):g}"
    else:
        head = {"guided": "dev, guided", "distilled": "distilled"}.get(s2.get("recipe"), str(s2.get("recipe")))
    return f"{head}, {s2.get('guidance')}, {steps} steps"


def take_actions(m: dict, k: int, ov: dict) -> dict:
    """The actions a take of clip k is made with and where they came from: its own text ("take") when the retake
    changes the video text, else the run's actions for that clip (with their source)."""
    if isinstance(ov.get("video"), str):
        return {"actions": fill_character(ov["video"], m.get("character_words")), "actions_source": "take"}
    cv = (m.get("clip_videos") or {}).get(str(k)) or ""
    return {"actions": cv, "actions_source": (m.get("clip_sources") or {}).get(str(k)) or ("run" if cv else "none")}


def clip_actions_of(m: dict, st: dict, k: int) -> tuple[str, str]:
    """(actions, source) of clip k as the video now has it: the run's actions, or those of the take put in for the
    clip (run use-take; Codex review 2026-10-25: the details and forks showed the original's)."""
    used = (st.get("takes_used") or {}).get(str(k))
    if used:                                            # take 0 = the original
        t = next((x for x in (st.get("takes") or {}).get(str(k), []) if x.get("n") == int(used)), None)
        if t is not None and not t.get("original"):
            a = t if "actions" in t else take_actions(m, k, t.get("overrides") or {})
            return a.get("actions") or "", a.get("actions_source") or "none"
    a = take_actions(m, k, {})
    return a["actions"], a["actions_source"]


def _clip_record(clip: dict, st: dict, k: int) -> dict:
    """The record of what clip k was given (prompt, negative, first image ...): clip 1's lives on the clip step; a take
    put in for clip 1 by an older use_take wrote its own into parts["1"], which then wins."""
    parts = clip.get("parts", {})
    if k != 1:
        return parts.get(str(k), {})
    p1 = parts.get("1") or {}
    if (st.get("takes_used") or {}).get("1") and "prompt" in p1:
        return {**clip, **p1}
    return clip


def run_details(run_id: str) -> dict:
    """What each clip of a run was given: the full prompt (recorded when the clip ran; for older runs rebuilt from
    the manifest with the same code), the negative prompt, the reference sheet image, the first-frame image (None
    when the clip had no first-frame guide), and the main settings."""
    d = run_dir(run_id)
    m, st = _read_json(d / "manifest.json"), _read_json(d / "state.json")
    clip = st.get("steps", {}).get("clip", {})
    fmt = st.get("steps", {}).get("controls", {}).get("outputs") or {}
    n = fmt.get("clips", 1)
    work = d / "work"
    clips = []
    for k in range(1, n + 1):
        rec = _clip_record(clip, st, k)
        actions, actions_source = clip_actions_of(m, st, k)
        if "prompt" in rec:
            prompt, how = rec["prompt"], "recorded"
        else:
            prompt, how = video_prompt(clip_manifest(m, k)), "rebuilt"
            if m.get("describe_facing"):
                how = "rebuilt (the facing text of this clip was not recorded)"
        if "first_image" in rec:
            first = rec["first_image"]
        elif k == 1:
            first = str(work / "first.png") if m.get("first_frame") else None
        else:
            first = str(work / f"first_{k}.png")
        clips.append({"clip": k, "prompt": prompt, "prompt_source": how, "negative": rec.get("negative", NEGATIVE),
                      "first_image": first if first and Path(first).is_file() else None,
                      # its own actions and where they came from (run / template / auto / none: AGENTS.md 7.2)
                      "actions": actions, "actions_source": actions_source})
    note = _read_json(d / "note.json") if (d / "note.json").is_file() else {}
    changes = []                                   # what this run does differently from a default run
    for k, v in DEFAULTS.items():
        if k in ("quality", "character_name", "style_ref", "style_warnings", "video_model", "props",   # shown
                 "prop_images", "characters", "clip_videos", "clip_context", "clip_join", "clip_sources",
                 "prompt_warnings", "character_words"):
            continue                                                                       # elsewhere; not a test change
        if k in m and m[k] != v and not (k == "style_lora_strength" and not m.get("style_lora")):
            changes.append({"name": k, "value": m[k], "default": v})
    if m.get("props"):
        changes.append({"name": "props", "value": props_label(m["props"]), "default": "the template's words"})
    if m.get("prop_images"):
        changes.append({"name": "prop_images", "value": prop_images_label(m), "default": "none (the sheet alone)"})
    if not m.get("style_lora"):
        changes.append({"name": "style_lora", "value": "none", "default": "a style LoRA"})
    if m.get("render_mode") == "layout" and _pairs.video_model_of(m.get("profile")) == "full":
        changes.append({"name": "video_model", "value": "full", "default": "distilled"})
    for k, label in (("control_video", "control video"), ("first_frame", "first frame")):
        if m.get(k):
            changes.append({"name": label, "value": m[k]["source"], "default": "none"})
    if m.get("restyle"):
        changes.append({"name": "restyle", "value": True, "default": False})
    ref = work / "reference.png"
    keys = ("character_name", "quality", "seed", "style_lora", "style_lora_strength", "ingredients_strength",
            "control_strength", "control_kind", "source_denoise", "describe_facing")
    p = m["prompt"]
    # the values in the order and with the names of the app's Video form ("More" holds style, LoRA and seed)
    form = [("Shot", shot_source(m)), *_character_rows(m),
            *([("Props", props_label(m["props"]))] if m.get("props") else []),
            *([("Prop pictures", prop_images_label(m))] if m.get("prop_images") else []),
            ("Quality", m.get("quality")), ("Style", p.get("style") or ""),
            ("Style LoRA", m.get("style_lora") or "none"),
            ("LoRA strength", m.get("style_lora_strength") if m.get("style_lora") else "-"), ("Seed", m.get("seed"))]
    sd = m.get("source_denoise")
    layout = m.get("render_mode") == "layout"           # Layout-To-Render: no skeleton, no start noise, no body type
    other = [("Start from the Blender render", f"yes, start noise {sd} (about {round((1 - sd) * 100)}% of the render kept)"
              if sd else "no (control only)"),
             ("Ingredients strength (character sheet)", m.get("ingredients_strength")),
             ("Control strength (skeleton)", m.get("control_strength")),
             ("Control", _control_label(m, fmt.get("control_full"))),
             ("Body type (skeleton drawing)", m.get("body_type") or "chibi"),
             ("Control video", (m.get("control_video") or {}).get("source") or "drawn from the shot"),
             ("First frame", (m.get("first_frame") or {}).get("source") or "none"),
             ("Facing text in the prompt", "yes" if m.get("describe_facing") else "no"),
             ("Steps (full model)", m.get("steps") or (DEV_STEPS if str(m.get("profile", "")).endswith("dev") else "-")),
             ("Guidance (video CFG)", m.get("guidance") or (3.0 if str(m.get("profile", "")).endswith("dev") else "-")),
             ("Negative prompt", m.get("negative") or "default"),
             ("Restyle the sheet first", "yes" if m.get("restyle") else "no")]
    if layout:                                          # the rows that a layout run really uses (user, 2026-10-22: "why
        bg = m.get("background")                        # is the body type still a thing here?")
        other = [("Video model", video_model_label(m)),
                 *([("Stage 2 (x2 refine)", stage2_label(m))] if stage2_label(m) else []),
                 ("Ingredients strength (character sheet)", m.get("ingredients_strength")),
                 ("Keyframe looks", m.get("look_keyframes") or "auto"),
                 ("Stand-in parts (proxy kit)", "on" if m.get("proxy_kit") else "off"),
                 ("Background", ("one picture of the place (backplate, still camera)" if bg == "plate" else
                                 f"{bg} ({m.get('background_ground') or 'dome'} ground)" if bg else "the template's set"))]
        # 2026-10-24 (user: "clean up this side section with any details that are no longer relevant"): rows of
        # options that are off by default show only when a run used them
        if float(m.get("pose_strength") or 0) > 0:
            other.append(("Pose guide (skeleton)", f"{m.get('pose_strength')} ({m.get('pose_body') or 'standard'})"))
        if m.get("first_frame"):
            other.append(("First frame", m["first_frame"].get("source")))
        if _pairs.video_model_of(m.get("profile")) == "full":   # only the full model uses the negative
            other.append(("Negative prompt", m.get("negative") or "default"))
        if m.get("restyle"):
            other.append(("Restyle the sheet first", "yes"))
    changed = {c["name"] for c in changes}
    lora = m.get("style_lora")
    ctl_src = (m.get("control_video") or {}).get("source")
    chips = [{"label": "Quality", "value": m.get("quality"), "title": "final = 1536x896 with a second refine stage"},
             {"label": "Seed", "value": m.get("seed"), "title": "the random seed"},
             {"label": "LoRA", "value": f"{Path(lora).stem} x{m.get('style_lora_strength')}" if lora else "none",
              "title": "style LoRA and its strength", "changed": not lora or "style_lora_strength" in changed},
             {"label": "Sheet guide", "value": m.get("ingredients_strength"), "title": "Ingredients IC-LoRA strength",
              "changed": "ingredients_strength" in changed}]
    if layout:                     # layout is the default pipeline: its chips show only what differs from a default run
        if lora:                   # (user, 2026-10-24); the full list stays under Details
            chips[2]["changed"] = False
        else:
            chips.pop(2)
        if "ingredients_strength" not in changed:
            chips = [c for c in chips if c["label"] != "Sheet guide"]
        if _pairs.video_model_of(m.get("profile")) == "full":
            chips.append({"label": "Video model", "value": "full", "changed": True, "title": video_model_label(m)})
        if (m.get("look_keyframes") or "auto") != "auto":
            chips.append({"label": "Keyframes", "value": m.get("look_keyframes"), "title": "look images inside the clips",
                          "changed": True})
        if not m.get("proxy_kit"):
            chips.append({"label": "Proxy kit", "value": "off", "changed": True,
                          "title": "the character's stand-in parts (mask, hair) on the mannequin"})
        if m.get("background") == "plate":
            chips.append({"label": "Background", "value": "one picture",
                          "title": "a backplate: one picture of the place for a still camera (sharper than a panorama)"})
        elif m.get("background"):
            chips.append({"label": "Background", "value": str(m["background"]).replace("saved:", ""),
                          "title": f"360 panorama on the {m.get('background_ground') or 'dome'} ground"})
        if float(m.get("pose_strength") or 0) > 0:
            chips.append({"label": "Pose guide", "value": m["pose_strength"], "changed": True,
                          "title": f"the rig's skeleton as an extra guide ({m.get('pose_body') or 'standard'} drawing)"})
    else:
        chips.insert(3, {"label": "Render kept", "value": f"{round((1 - sd) * 100)}%" if sd else "none",
                         "title": f"video-to-video from the Blender render, start noise {sd}" if sd else
                         "control only (no render)", "changed": "source_denoise" in changed})
        chips.append({"label": "Control", "value": Path(ctl_src).parent.name if ctl_src else
                      ("rig + outlines" if (m.get("control_kind") or "pose_env") == "pose_env" else m.get("control_kind")),
                      "title": _control_label(m, fmt.get("control_full")),
                      "changed": bool(ctl_src) or "control_kind" in changed})
        if not ctl_src:
            chips.append({"label": "Body type", "value": m.get("body_type") or "chibi",
                          "title": "skeleton drawing: chibi = thick (ours), standard = the DWPose node's",
                          "changed": (m.get("body_type") or "chibi") != "chibi"})
    if not layout and m.get("control_strength") not in (None, 1, 1.0):
        chips.append({"label": "Control strength", "value": m["control_strength"], "title": "skeleton guide strength",
                      "changed": True})
    if m.get("first_frame"):
        chips.append({"label": "First frame", "value": "yes", "title": m["first_frame"]["source"], "changed": True})
    if m.get("describe_facing"):
        chips.append({"label": "Facing text", "value": "on", "title": "where each character faces, in the prompt",
                      "changed": True})
    if m.get("graph_sources"):                     # the user's own graph copies (the Workflows view)
        chips.append({"label": "Custom graph", "value": ", ".join(str(v.get("label") or v.get("copy"))
                                                                   for v in m["graph_sources"].values()),
                      "title": "; ".join(f"{k}: {v.get('label')} ({v.get('copy')}, sha1 {str(v.get('sha1'))[:10]})"
                                         for k, v in m["graph_sources"].items()), "changed": True})
    used = (st.get("steps", {}).get("final", {}).get("outputs") or {}).get("sound")
    if used or m.get("sound"):
        song_txt = f"{m['song']['name']} from {float(m['song'].get('start') or 0):g} s" if m.get("song") else None
        chips.append({"label": "Sound", "value": song_txt or {"blender": "Blender song", "template": "template's sound", "ltx": "LTX"}.get(used, m.get("sound")),
                      "title": "blender = the shot's own sound replaced LTX's; ltx = LTX's own sound",
                      "changed": used in ("blender", "song", "template") or bool(m.get("song"))})
    return {"clips": clips, "note": note.get("text"), "changes": changes, "video_text": p.get("video", ""),
            "clip_context": m.get("clip_context") or "", "clip_join": m.get("clip_join"),
            "prompt_warnings": m.get("prompt_warnings") or [],
            "layout": layout, "negative_used": str(m.get("profile", "")).endswith("dev"),
            "chips": chips, "style_text": p.get("style", ""),
            "character_text": p.get("character", ""), "form": form, "other": other,
            "reference_image": str(ref) if ref.is_file() else None,
            "sheet_file": str(d / (m["sheet"].get("props_file") or m["sheet"]["file"])),   # what the clips got
            "sheet_source": m["sheet"]["source"],
            "first_frame_source": (m.get("first_frame") or {}).get("source"),
            "control_video_source": (m.get("control_video") or {}).get("source"),
            "blend_source": shot_source(m),
            "settings": {k: ("none" if k == "style_lora" else None) if k.startswith("style_lora") and not m.get("style_lora")
                         else m.get(k) for k in keys}}


# what a fork may change (they act on the clips it makes). NOT the video model (profile, video_model, steps, guidance):
# a fork keeps the parent's clips and continues from their latents, so it keeps the parent's model (copied manifest);
# a retake keeps it too (TAKE_FIELDS)
FORK_FIELDS = ("video", "style", "describe_facing",
               "look_chain", "chain_colour", "adain", "adain_ref", "keyframe", "relay_strength", "clip_text",
               "chain_light", "chain_light_ref", "chain_grade", "chain_sharpen_mode", "chain_sharpen_target", "audio_relay", "props",
               "prop_images",
               "clip_videos")                          # {clip: ACTIONS} for the clips the fork makes ("" = none)
PATH_FIELDS = {"control", "control_full", "source", "sheet", "clip", "clips", "final", "contact_sheet", "first_image"}


def _inside(path: str, root: Path) -> Path | None:
    """The path relative to root when `path` is an absolute path inside root (any separator or case), else None."""
    try:
        p = Path(path)
        if not p.is_absolute():
            return None
        rel = os.path.relpath(os.path.normcase(os.path.normpath(p)), os.path.normcase(os.path.normpath(root)))
    except (ValueError, OSError):                      # another drive, or not a path at all
        return None
    return None if rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel) else Path(rel)


def _relocate(obj: Any, src: Path, dst: Path) -> Any:
    """A copy of a state tree where the known artifact-path fields that point into src point into dst. Job records,
    prompts and every other text stay verbatim."""
    if isinstance(obj, list):
        return [_relocate(v, src, dst) for v in obj]
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k, v in obj.items():
        if k == "job":
            out[k] = v
        elif k in PATH_FIELDS and isinstance(v, str):
            rel = _inside(v, src)
            out[k] = str(dst / rel) if rel is not None else v
        elif k in PATH_FIELDS and isinstance(v, list):
            out[k] = [str(dst / r) if isinstance(x, str) and (r := _inside(x, src)) is not None else x for x in v]
        else:
            out[k] = _relocate(v, src, dst)
    return out


def fork_run(run_id: str, from_clip: int, note: str | None = None, **changes: Any) -> dict:
    """A new run that keeps the finished clips 1..from_clip-1 of `run_id` as they are and makes the clips from
    `from_clip` on again, with `changes` (video text, style, facing text). Clip from_clip starts on the last frame of
    the clip before it, as in any run. The shot, sheet, control and seed stay the same, so the cached passes are
    reused. Returns the new manifest."""
    src = run_dir(run_id)
    changes = {k: v for k, v in changes.items() if v is not None}      # only what really changes
    unknown = set(changes) - set(FORK_FIELDS)
    if unknown:
        raise RunError(f"a fork can change only {list(FORK_FIELDS)}, not {sorted(unknown)}")
    for key, allowed in (("look_chain", LOOK_CHAINS), ("chain_colour", CHAIN_COLOURS), ("adain_ref", ("look", "clip1")),
                         ("keyframe", ("on", "off")), ("chain_light", ("off", "look", "tone")),
                         ("chain_light_ref", ("look", "clip1")), ("chain_grade", ("off", "match", "local")),
                         ("chain_sharpen_mode", ("fixed", "match")), ("audio_relay", ("off", "carry"))):
        if key in changes and changes[key] not in allowed:
            raise RunError(f"{key.replace('_', ' ')} must be one of {list(allowed)}")
    if "clip_text" in changes and changes["clip_text"] not in ("manual", "auto"):
        raise RunError("clip text must be manual or auto")
    if "adain" in changes and not 0.0 <= float(changes["adain"]) <= 1.0:
        raise RunError("adain must be 0..1")
    if "chain_sharpen_target" in changes and not 0.5 <= float(changes["chain_sharpen_target"]) <= 3.0:
        raise RunError("chain sharpen target must be 0.5..3 (x clip 1's sharpness)")
    if "relay_strength" in changes and not 0 < float(changes["relay_strength"]) <= 1:
        raise RunError("relay strength must be above 0 and at most 1")
    if "props" in changes and not isinstance(changes["props"], dict):
        raise RunError("props must map a prop pattern to its words")
    if "prop_images" in changes and not isinstance(changes["prop_images"], dict):
        raise RunError("prop images must map a prop pattern to a picture")
    if "clip_videos" in changes and not (isinstance(changes["clip_videos"], dict) and all(
            str(k).strip().isdigit() and int(k) >= 1 and isinstance(v, str) for k, v in changes["clip_videos"].items())):
        raise RunError("clip videos must map clip numbers to texts")
    with _life_lock(run_id):                           # no delete and no worker start during the copy
        if not (src / "manifest.json").is_file():
            raise RunError(f"run {run_id} was deleted")
        if worker_alive(run_id):
            raise RunError(f"run {run_id} is working; fork it when it has stopped")
        return _fork_locked(run_id, src, from_clip, note, changes)


def _fork_props(m: dict, change: dict) -> dict:
    """A fork's prop words: the parent's own words with `change` over them; empty words drop the parent's (the
    template's words again). A pattern must be one of the template's (its record, else the run's frozen words)."""
    from . import shots
    out = dict(m.get("props") or {})
    given = {}
    for pat, words in change.items():
        if " ".join(str(words or "").split()):
            given[str(pat).strip()] = words
        else:
            out.pop(str(pat).strip(), None)
    if not given:
        return out
    try:
        out.update(shots.check_prop_words(_fork_prop_record(m, set(out)), given))
    except shots.ShotError as exc:
        raise RunError(str(exc)) from None
    return out


def _fork_prop_record(m: dict, known: set) -> dict | None:
    """The template record a fork checks prop patterns against: the library's, else (the template is gone or was
    never scanned) one made of the run's frozen prop words and the patterns it already uses (`known`)."""
    from . import shots
    rec = shots.library_record(m["blend"]["source"]) if (m.get("blend") or {}).get("source") else None
    if not ((rec or {}).get("layers") or {}).get("scan"):
        known = set(((m.get("layers") or {}).get("descriptions") or {}).get("props") or {}) | set(known)
        rec = {"layers": {"scan": {"props": [{"name": k} for k in known]},
                          "descriptions": {"props": {k: "" for k in known}}}}
    return rec


def _fork_prop_images(m: dict, change: dict, stage: Path) -> dict:
    """A fork's prop pictures: the parent's with `change` {pattern: picture} over them (an empty picture drops the
    parent's); the new ones are checked and frozen into the fork's inputs."""
    from . import charsheet
    out = dict(m.get("prop_images") or {})
    given = {}
    for pat, path in change.items():
        if str(path or "").strip():
            given[str(pat).strip()] = path
        else:
            out.pop(str(pat).strip(), None)
    if given:
        checked = check_prop_images(_fork_prop_record(m, set(out) | set(m.get("props") or {})), given)
        out.update(_freeze_prop_images(stage, checked))
    if len(out) > charsheet.MAX_PROP_PANELS:
        raise RunError(f"a run takes at most {charsheet.MAX_PROP_PANELS} prop pictures, not {len(out)}")
    return out


def _fork_locked(run_id: str, src: Path, from_clip: int, note: str | None, changes: dict) -> dict:
    m, st = _read_json(src / "manifest.json"), _read_json(src / "state.json")     # read under the lock
    if retired(m):
        raise RunError(retired(m))
    clip = st.get("steps", {}).get("clip", {})
    n = (st.get("steps", {}).get("controls", {}).get("outputs") or {}).get("clips", 1)
    if not 2 <= from_clip <= n:
        raise RunError(f"from_clip must be between 2 and {n} (the run has {n} clips)")
    parts = clip.get("parts") or {}
    kept = [str(k) for k in range(1, from_clip)]
    if not all((parts.get(k) or {}).get("outputs") for k in kept):
        raise RunError(f"clips {', '.join(kept)} of {run_id} are not all finished")
    ref = changes.get("adain_ref", m.get("adain_ref"))  # B reads clip 1's saved latent: refuse a fork without it
    if ref == "clip1" and float(changes.get("adain", m.get("adain")) or 0) > 0 \
            and not ((parts.get("1") or {}).get("relay") or {}).get("s1"):
        raise RunError(f"clip 1 of {run_id} has no saved latent: adain ref clip1 cannot be used in this fork")
    new_id = time.strftime("r%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    stage = runs_dir() / f".staging-{new_id}"
    try:
        shutil.copytree(src / "inputs", stage / "inputs")
        graphs = {}                                     # the new clips use today's graphs (repairs reach a fork)
        (stage / "inputs" / "workflows").mkdir(parents=True, exist_ok=True)
        prof = m.get('profile', PROFILE)
        extra = ("layout", "layouting") if m.get("render_mode") == "layout" else ()   # layout runs (Codex 10-21)
        for v in (*VARIANTS, *extra):
            for suffix in ("_api.json", ".bind.json"):
                gname = f"{prof}_{v}{suffix}"
                if (WORKFLOWS / gname).is_file() and f"{prof}_{v}" not in (m.get("graph_sources") or {}):
                    shutil.copy2(WORKFLOWS / gname, stage / "inputs" / "workflows" / gname)   # a user's graph copy:
                    #                                              the parent's frozen copy stays (reproducible fork)
        parent = m.get("models") or {}
        kept_copies = {g: (stage / "inputs" / "workflows" / f"{g}_api.json").read_bytes()     # the parent's frozen
                       for g in (m.get("graph_sources") or {})                                  # graph copies stay
                       if (stage / "inputs" / "workflows" / f"{g}_api.json").is_file()}         # exactly as they ran
        _freeze_tier(stage / "inputs" / "workflows", parent.get("tier"), parent.get("files"),   # the parent's models
                     settings=parent.get("settings") or {})                                       # and node settings
        for g, data in kept_copies.items():
            (stage / "inputs" / "workflows" / f"{g}_api.json").write_bytes(data)
        graphs = {p.name: _sha256(p) for p in sorted((stage / "inputs" / "workflows").iterdir())}
        keep_names = {"clip.mp4" if k == "1" else f"clip_{k}.mp4" for k in kept}
        drop_names = ({f"clip_{k}.mp4" for k in range(from_clip, n + 1)} | {f"first_{k}.png" for k in range(from_clip, n + 1)}
                      | {f"last_{k}.png" for k in range(from_clip - 1, n + 1)} | {"web_control.mp4"})
        if from_clip == 1:
            drop_names.add("clip.mp4")
        shutil.copytree(src / "work", stage / "work", ignore=lambda d, names: [x for x in names if x in drop_names])
        assert all((stage / "work" / x).is_file() for x in keep_names)
        m2 = json.loads(json.dumps(m))
        m2.update(id=new_id, created=time.strftime("%Y-%m-%dT%H:%M:%S"), workflows=graphs,
                  forked_from={"run": run_id, "from_clip": from_clip, "changes": sorted(changes)})
        for key, used in (st.get("takes_used") or {}).items():     # a take put in for a clip: the fork goes on with
            if used and str(key).isdigit() and 1 <= int(key) <= n:  # ITS actions (the takes are not copied; Codex
                act, srck = clip_actions_of(m, st, int(key))         # review 2026-10-25)
                if srck == "take":
                    cv = dict(m2.get("clip_videos") or {})
                    if act:
                        cv[str(int(key))] = act
                    else:
                        cv.pop(str(int(key)), None)
                    m2["clip_videos"] = cv
                    m2["clip_sources"] = dict(sorted({**(m2.get("clip_sources") or {}),
                                                      str(int(key)): "take" if act else "none"}.items(),
                                                     key=lambda kv: int(kv[0])))
        if m.get("render_mode") == "layout":            # the new clips take today's stage 2; the kept ones ran the
            s2 = stage2_recipe(stage / "inputs" / "workflows", prof, m.get("quality"))   # parent's
            parent_s2 = m.get("stage2") or stage2_recipe(src / "inputs" / "workflows", prof, m.get("quality"))
            if s2 and parent_s2 and parent_s2.get("recipe") != s2.get("recipe"):
                s2["kept_clips"] = parent_s2.get("recipe")
            m2["stage2"] = s2
        for k in ("video", "style"):
            if changes.get(k) is not None:
                m2["prompt"][k] = changes[k]
        if changes.get("describe_facing") is not None:
            m2["describe_facing"] = bool(changes["describe_facing"])
        for k in ("look_chain", "chain_colour", "adain", "adain_ref", "keyframe", "relay_strength", "chain_light",
                  "chain_light_ref", "chain_grade", "chain_sharpen_mode", "chain_sharpen_target", "audio_relay"):              # (chain_light/chain_sharpen_mode were missing: a fork kept the old
            if changes.get(k) is not None:              # values, 2026-10-23) the chain options of the drift ablation
                m2[k] = changes[k]
        if m2.get("audio_relay") == "carry" and m2.get("relay") != "latent":
            raise RunError("the audio carry needs the latent relay; the parent run has none")
        if changes.get("props") is not None:            # the parent's prop words stay unless changed ("" = back to
            m2["props"] = _fork_props(m, changes["props"]) or None   # the template's words for that pattern)
        new_sheet = None
        if changes.get("prop_images") is not None:      # new prop pictures: the run's sheet is drawn again for the
            if m2.get("restyle"):                       # clips made again (the kept clips keep the old sheet)
                raise RunError("a run with a restyled sheet cannot change its prop pictures in a fork")
            m2["prop_images"] = _fork_prop_images(m, changes["prop_images"], stage) or None
            m2["sheet"] = {k: v for k, v in m2["sheet"].items() if k not in ("props_file", "prop_places", "prop_layout")}
            m2["sheet"].update(prop_sheet(stage, m2))
            new_sheet = refsheet.prepare(stage / (m2["sheet"].get("props_file") or m2["sheet"]["file"]),
                                         stage / "work" / "sheet")
        if changes.get("clip_text") == "auto":           # the clips made again get texts written from the rig
            from . import timeline as _tl
            raw = _tl.blend_timeline(stage / "inputs" / "shot.blend", stage / "inputs" / "timeline.json",
                                     m2.get("rig", "MannequinRig"))
            fmt = (st.get("steps", {}).get("controls", {}).get("outputs") or {})
            place = "" if m2.get("clip_join") == "context" else m2["prompt"].get("video", "")   # context: the main
            auto = _tl.run_clip_texts(raw, place, int(fmt.get("frames") or MAX_CLIP_FRAMES),      # text is joined
                                      int(fmt.get("overlap") or m2.get("overlap", 1)))
            cv = dict(m2.get("clip_videos") or {})
            cv.update({k: v for k, v in auto.items() if int(k) >= from_clip})
            m2["clip_videos"], m2["clip_text"], m2["timeline_events"] = cv, "auto", _tl.events(raw)
            if m2.get("clip_sources"):
                m2["clip_sources"] = {**m2["clip_sources"], **{k: "auto" for k in auto if int(k) >= from_clip}}
        if changes.get("clip_videos") is not None:      # the clips made again get their own actions (the kept clips
            cv = dict(m2.get("clip_videos") or {})      # keep theirs); "" = no actions of its own
            srcs = dict(m2.get("clip_sources") or {})
            for k, text in changes["clip_videos"].items():
                k = str(int(k))
                if not 1 <= int(k) <= n:
                    raise RunError(f"clip {k}: the run has {n} clips")
                if int(k) < from_clip:
                    raise RunError(f"clip {k} is kept by the fork (it makes clips {from_clip}..{n}): change its text "
                                   "with a retake")
                text = fill_character(" ".join(text.split()), m2.get("character_words"))
                if text:
                    cv[k], srcs[k] = text, "run"
                else:
                    cv.pop(k, None)
                    srcs[k] = "none"
            m2["clip_videos"], m2["clip_sources"] = cv, dict(sorted(srcs.items(), key=lambda kv: int(kv[0])))
        if m2.get("clip_join") == "context" and (changes.get("clip_videos") is not None or changes.get("video") is not None
                                                 or changes.get("clip_text") == "auto"):
            m2["prompt_warnings"] = prompt_warnings(n, m2.get("clip_sources") or {}, m2["prompt"].get("video", ""),
                                                    m2.get("clip_videos") or {}) or None
        for k in REMOVED_OPTIONS:                      # the masks are gone: a fork of an old run is drawn without them
            m2.pop(k, None)
        if m2.get("figure_mask") not in FIGURE_MASK_MODES:     # soft/full: keep only the choice of the figure
            m2["figure_mask"] = "control" if (src / "inputs" / "figure_mask.mp4").is_file() else "off"
        st2 = _relocate(st, src, runs_dir() / new_id)
        st2["steps"].pop("scene", None)
        if new_sheet is not None:                       # the sheet step's output: the new prop sheet on black
            st2["steps"]["sheet"] = {**(st2["steps"].get("sheet") or {}), "status": "done",
                                     "outputs": {"sheet": str(runs_dir() / new_id / "work" / "sheet" / "sheet_black.png"),
                                                 "views": len(new_sheet["views"])}}
        c2 = st2["steps"]["clip"]
        for k in [k for k in c2.get("parts", {}) if int(k) >= from_clip]:
            del c2["parts"][k]
        for k in ("outputs", "ended"):
            c2.pop(k, None)
        c2["status"] = "pending"
        st2["steps"]["final"] = {"status": "pending"}
        for k in ("takes", "takes_used", "clip_limit"):     # the takes folder is not copied: a fork starts without them
            st2.pop(k, None)
        st2["status"] = "new"
        st2.pop("updated", None)
        _write_json(stage / "manifest.json", m2)
        _write_json(stage / "state.json", st2)
        if note:
            (stage / "note.json").write_text(json.dumps({"text": note.strip(), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}),
                                             encoding="utf-8")
        os.replace(stage, runs_dir() / new_id)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return m2


def set_note(run_id: str, text: str) -> dict:
    """A short note on what the run tests (shown with its prompt in the app). Not an input: the manifest stays frozen."""
    d = run_dir(run_id)
    note = {"text": text.strip(), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _write_json(d / "note.json", note)               # atomic: the app may read it at the same moment
    return {"id": run_id, **note}


def _run_dirs() -> list[Path]:
    """Every run folder, newest first (run ids start with their time)."""
    if not runs_dir().is_dir():
        return []
    return sorted((p for p in runs_dir().iterdir() if p.is_dir() and not p.name.startswith(".")), reverse=True)


def _status_of(d: Path) -> str | None:
    try:
        return _read_json(d / "state.json").get("status")
    except (OSError, ValueError):
        return None


def _listed_dirs() -> list[Path]:
    """The run folders the app lists: a CANCELED run is not shown (user, 2026-10-21: "canceled tasks shouldnt show and
    should be properly cleaned up"; cleanup_canceled deletes them)."""
    return [d for d in _run_dirs() if _status_of(d) != "canceled" or (d / "final.mp4").is_file()]   # kept videos show


def cleanup_canceled() -> list[str]:
    """Delete every canceled run whose worker has stopped (its folder and its latent copies in ComfyUI's input folder).
    A run that is still settling its cancel (live worker) is left for the next call. Returns the deleted ids."""
    gone = []
    for d in _run_dirs():
        if _status_of(d) != "canceled" or (d / "final.mp4").is_file():   # never a run with a finished video
            continue
        try:
            delete_run(d.name, only_if_canceled=True)          # checked again inside the lock (Codex 10-22)
        except (RunError, LockBusy, OSError):
            continue
        gone.append(d.name)
        try:
            from . import comfy
            for q in comfy.io_dir("input").glob(f"relay_{d.name}_*.latent"):
                q.unlink(missing_ok=True)
        except Exception:                               # the cleanup of ComfyUI's input folder is best effort
            pass
    return gone


def count_runs() -> int:
    return len(_listed_dirs())


def list_runs(limit: int | None = 20, offset: int = 0) -> list[dict]:
    """Runs, newest first: `limit` of them from `offset` (limit None or 0 = all)."""
    dirs = _listed_dirs()[max(0, int(offset)):]
    if limit:
        dirs = dirs[:int(limit)]
    out = []
    for d in dirs:
        try:
            m, s = _read_json(d / "manifest.json"), _read_json(d / "state.json")
            fin = d / "final.mp4"
            out.append({"id": m["id"], "created": m["created"], "status": s["status"], "quality": m["quality"],
                        "final": str(fin) if fin.is_file() else None,
                        "retired": retired(m),          # an old LTX-2.3 run: watched, not continued (the card says so)
                        # when the video last changed (a retake put in later joins the clips again)
                        "updated": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(fin.stat().st_mtime))
                        if fin.is_file() else None})
        except (OSError, ValueError, KeyError) as exc:
            out.append({"id": d.name, "status": "unreadable", "error": str(exc)})
    return out


def _state_lock(d: Path) -> FileLock:
    """A short lock for the launch counter and the cancel request (never held while work waits)."""
    return FileLock(d / "state.lock", timeout=30)


def launch_gen(run_id: str) -> int:
    try:
        return int(_read_json(run_dir(run_id) / "launch.json").get("gen", 0))
    except (OSError, ValueError):
        return 0


def new_launch(run_id: str) -> int:
    """A new explicit start of the run: the next launch number. A cancel made before it no longer
    applies; a cancel made after it (or for it) stops it."""
    d = run_dir(run_id)
    with _state_lock(d):
        gen = launch_gen(run_id) + 1
        _write_json(d / "launch.json", {"gen": gen, "at": time.time()})
    return gen


def job_records(step: dict) -> list[dict]:
    """Every ComfyUI job record of a step that may still run: the step's own record (clip 1 and every other
    step) and the record of each unfinished clip part (clips 2..n live in step["parts"])."""
    parts = step.get("parts") or {}
    recs = []
    if step.get("job") and not (parts.get("1") or {}).get("outputs"):
        recs.append(step["job"])
    for k, part in parts.items():
        if k != "1" and part.get("job") and not part.get("outputs"):
            recs.append(part["job"])
    return recs


def request_cancel(run_id: str, backend_factory: Callable[[], Any] | None = None) -> dict:
    """Ask for a cancel. The request is a file of its own (the worker never writes it, so it cannot be
    lost). A known ComfyUI job of the run is canceled at once (job-scoped, never a global interrupt).
    Without a live worker, the run is settled here: its jobs are found and canceled, and the run is
    marked canceled (under the run lock, so no worker can start meanwhile)."""
    d = run_dir(run_id)
    with _state_lock(d):                       # the cancel belongs to the newest launch (it stops that launch)
        _write_json(d / "cancel.json", {"requested_at": time.time(), "gen": launch_gen(run_id)})
    st = _read_json(d / "state.json")
    if st["status"] == "done":
        return {"id": run_id, "status": "done", "canceled": False}
    backend = None
    canceled = []
    takes = [t for lst in (st.get("takes") or {}).values() for t in lst]       # a retake's job lives in its record
    for name, step in [*st["steps"].items(), *((f"take{i}", t) for i, t in enumerate(takes))]:
        if step.get("status") not in ("running", "interrupted", "unresolved"):
            continue
        for job in job_records(step):
            if not job.get("id"):
                continue
            backend = backend or (backend_factory or _default_backend)()
            try:
                backend.cancel(job["id"])
                canceled.append(job["id"])
            except Exception:  # noqa: BLE001 - the job may be gone already
                pass
    try:
        with FileLock(d / "run.lock", timeout=0):
            run = Run(run_id, backend_factory=backend_factory)
            outcome = run.settle_cancel()["status"]
            return {"id": run_id, "status": outcome, "jobs_canceled": canceled + run.canceled_jobs,
                    "unresolved_jobs": run.unresolved_jobs, "worker": False}
    except LockBusy:
        return {"id": run_id, "status": "cancel_requested", "jobs_canceled": canceled, "worker": True}


def _default_backend():
    from .backend import SdkBackend
    return SdkBackend()


# ---------------------------------------------------------------- the worker

class Run:
    def __init__(self, run_id: str, backend_factory: Callable[[], Any] | None = None,
                 on_event: Callable[[dict], None] | None = None) -> None:
        self.d = run_dir(run_id)
        self.id = run_id
        self.m = _read_json(self.d / "manifest.json")
        self.st: dict = {}
        self.work = self.d / "work"
        self._backend = None
        self._backend_factory = backend_factory or _default_backend
        self.on_event = on_event
        self.canceled_jobs: list[str] = []
        self.unresolved_jobs: list[str] = []
        self.new_attempt = False

    # state and events (only the lock holder calls these)
    def load(self) -> None:
        self.st = _read_json(self.d / "state.json")

    def save(self) -> None:
        self.st["updated"] = time.time()
        _write_json(self.d / "state.json", self.st)

    def removed_options(self) -> list[str]:
        """Options of an old run that no longer exist (the masks, 2026-10-21): such a run cannot be resumed as made."""
        bad = [k for k in REMOVED_OPTIONS if self.m.get(k) not in (None, False)]
        if self.m.get("figure_mask", "off") not in FIGURE_MASK_MODES:
            bad.append(f"figure_mask={self.m['figure_mask']}")
        return bad

    def cancel_requested(self) -> bool:
        return (self.d / "cancel.json").is_file()

    def event(self, kind: str, **data: Any) -> None:
        ev = {"t": round(time.time(), 2), "run": self.id, "event": kind, **data}
        with open(self.d / "events.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev) + "\n")
        if self.on_event:
            self.on_event(ev)

    @property
    def backend(self):
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    def workflow(self, variant: str) -> ProfileWorkflow:
        """The graph frozen into the run at `run new` (not the one in the repository now)."""
        frozen = self.d / "inputs" / "workflows"
        return ProfileWorkflow(self.m.get("profile", PROFILE), variant,
                               workflows_dir=frozen if frozen.is_dir() else WORKFLOWS)  # v1 runs: the repository

    # paths
    @property
    def shot_dir(self) -> Path:
        key = self.m.get("pass_key") or self.m["blend"]["sha256"][:16]                          # v1 runs: blend only
        out = self.st.get("steps", {}).get("background", {}).get("outputs") or {}
        bg = out.get("sha256")
        ground = "" if out.get("ground", "plane") == "plane" else "-" + out["ground"]          # old runs: the flat ground
        return home() / "shots" / (f"{key}-bg{bg[:10]}{ground}" if bg else key)                 # the panorama's passes

    def shot_blend(self) -> Path:
        """The .blend the passes are rendered from: the run's copy with its panorama world, else the frozen shot."""
        bg = (self.st.get("steps", {}).get("background", {}).get("outputs") or {}).get("blend")
        return Path(bg) if bg and Path(bg).is_file() else self.d / self.m["blend"]["file"]

    @property
    def pass_dir(self) -> Path:
        return self.shot_dir / "passes"

    def clip_format(self) -> dict:
        info = _read_json(self.pass_dir / "passes.json")
        most = int(self.m.get("clip_frames") or MAX_CLIP_FRAMES)
        frames = min(info["frames"], most)
        frames = (frames - 1) // 8 * 8 + 1
        if frames < 9:
            raise RunError(f"the shot has {info['frames']} frames; at least 9 are needed")
        w, h = SIZES["portrait" if info["height"] > info["width"] else "landscape"]
        az = info.get("camera_azimuth_deg")
        # a shot up to one clip long: one clip (a few last frames may be cut to the 8k+1 rule); a longer shot:
        # full clips that share one frame (clip k+1 starts on the last frame of clip k)
        ov = int(self.m.get("overlap", 1))             # runs made before the overlap joins: one shared frame
        clips = 1 if info["frames"] <= most else video.n_clips(info["frames"], frames, ov)
        return {"frames": frames, "fps": float(info["fps"]), "width": w, "height": h, "shot_frames": info["frames"],
                "clips": clips, "overlap": ov, "camera_azimuth_deg": az[:frames] if az else None}

    # main loop
    def start(self, new_attempt: bool = False, launch: int | None = None) -> dict:
        """Run (or continue) the run. new_attempt: after an 'unresolved' stop, give up on the old job
        intent and submit a new attempt (only when the user has checked that no job is running)."""
        lock = FileLock(self.d / "run.lock", timeout=0, note={"run": self.id})
        with _life_lock(self.id):                     # not while a delete removes the folder
            if not (self.d / "manifest.json").is_file():
                raise RunError(f"run {self.id} was deleted")
            try:
                lock.acquire()
            except LockBusy as exc:
                raise RunError(f"run {self.id} already has a worker: {exc}") from exc
        try:
            self.load()                          # reread under the lock
            self.new_attempt = new_attempt
            if self.st["status"] == "done" and not self._pending_takes():
                self.event("run_done", final=str(self.d / "final.mp4"), note="already done")
                return self.st
            if retired(self.m):                  # its graphs and models are gone: refuse before the queue
                raise RunError(retired(self.m))
            launch = new_launch(self.id) if launch is None else launch   # a start from a shell is a new launch
            with _state_lock(self.d):
                cancel = _read_json(self.d / "cancel.json") if self.cancel_requested() else None
                if cancel is not None and int(cancel.get("gen", 0)) >= launch:
                    obey = True                      # canceled after (or for) this launch: do not run
                else:
                    obey = False
                    if cancel is not None:           # a cancel of an earlier launch: this explicit start resumes
                        (self.d / "cancel.json").unlink()
            if obey:
                self.event("cancel_before_start", launch=launch)
                return self.settle_cancel()
            # wait for this run's turn in the shared queue (character steps and other videos)
            self.st.update(status="queued", worker={"pid": os.getpid(), "started": time.time()})
            self.save()

            def waiting(position: int) -> None:
                self.st["queue_position"] = position
                self.save()
                self.event("queued", position=position)
            try:
                workqueue.wait_turn("video", self.id, queue_label(self.m), on_wait=waiting,
                                    should_stop=self.cancel_requested)
            except workqueue.QueueCanceled:
                self.st.pop("queue_position", None)
                return self.settle_cancel()
            try:
                self.st.pop("queue_position", None)
                self.work.mkdir(exist_ok=True)
                bad = self.removed_options()
                if bad:
                    raise RunError(f"this run uses options that were removed (the masks, 2026-10-21): {bad}; "
                                   "fork it (run fork) to make it again without them")
                self.st.update(status="running", worker={"pid": os.getpid(), "started": time.time()})
                self.st.pop("error", None)
                self.save()
                self.event("run_started", steps=steps_of(self.m))
                return self._loop()
            finally:
                workqueue.done("video", self.id)
        finally:
            lock.release()

    def _loop(self) -> dict:
        try:
            outcome = self._run_takes()                       # retakes first: a paused run makes its takes, then
            if outcome == "canceled":                         # stops again at its clip limit
                return self.settle_cancel()
            if outcome == "unresolved":
                return self._finish(outcome)
            for name in steps_of(self.m):
                step = self.st["steps"][name]
                if step["status"] == "done":
                    continue
                if self.cancel_requested():
                    return self.settle_cancel()
                step.update(status="running", started=time.time())
                step.pop("error", None)
                self.save()
                self.event("step_started", step=name)
                outcome = getattr(self, "step_" + name)(step)
                if outcome == "canceled":
                    return self.settle_cancel()        # cancels the step's ComfyUI job, if it has one
                if outcome == "unresolved":
                    step["status"] = outcome
                    self.save()
                    return self._finish(outcome)
                if outcome == "paused":                     # run start --through-clip K: stop after clip K
                    step["status"] = "paused"
                    self.save()
                    self.event("paused", after_clip=self.st.get("clip_limit"))
                    return self._finish("paused")
                step.update(status="done", ended=time.time())
                self.save()
                self.event("step_done", step=name, outputs=step.get("outputs", {}))
            return self._finish("done")
        except Exception as exc:  # noqa: BLE001 - every failure is recorded in the state
            lost = _is_connection_error(exc)
            status = "interrupted" if lost else "failed"
            message = (f"lost the connection to ComfyUI or comfy-api-proxy ({exc}). The job is kept: check "
                       "`python -m cinestudio doctor`, start what is down, then start this run again") if lost else str(exc)
            for step in self.st["steps"].values():
                if step["status"] == "running":
                    step.update(status=status, error=message, ended=time.time())
            self.st["error"] = message
            self.save()
            self.event("error", error=message, error_type=type(exc).__name__)
            return self._finish(status)

    def settle_cancel(self) -> dict:
        """Cancel requested: find and cancel every outstanding job of this run first (also a job whose
        submit was never confirmed), then mark the run canceled."""
        self.load() if not self.st else None
        takes = [t for lst in (self.st.get("takes") or {}).values() for t in lst
                 if t.get("status") not in ("done", "failed", "canceled")]
        for name, step in [*self.st["steps"].items(), *((f"take{i}", t) for i, t in enumerate(takes))]:
            records = job_records(step)
            if step.get("status") == "done" or not records:
                if name.startswith("take") and step.get("status") not in ("done", "failed"):
                    step["status"] = "canceled"
                continue
            for job in records:                   # clip 1 and every unfinished later clip
                jid = job.get("id") or self._find_job(job)
                if not jid:
                    continue
                try:
                    st = self.backend.state(jid)
                    if not st.done:
                        self.backend.cancel(jid)
                        self.canceled_jobs.append(jid)
                except Exception:  # noqa: BLE001 - not in the proxy: try ComfyUI's own queue
                    if server.cancel_prompt(jid):
                        self.canceled_jobs.append(jid)
                    elif server.alive() and jid not in server.queue_ids():
                        pass                          # ComfyUI answers and does not have it: nothing runs
                    else:
                        self.unresolved_jobs.append(jid)
            if step.get("status") in ("running", "pending", "interrupted", "unresolved"):
                step["status"] = "canceled"
        for t in takes:                                     # queued takes that never started
            if t.get("status") in ("pending", "running"):
                t["status"] = "canceled"
        if self.canceled_jobs:
            self.event("jobs_canceled", jobs=self.canceled_jobs)
        if self.unresolved_jobs:
            # the cancel could not be confirmed (ComfyUI or the proxy did not answer): say so, keep the job
            self.st["error"] = (f"cancel requested, but ComfyUI did not confirm that job(s) {self.unresolved_jobs} "
                                "stopped; cancel again when ComfyUI answers")
            self.event("cancel_unresolved", jobs=self.unresolved_jobs)
            return self._finish("cancel_unresolved")
        if self.st["steps"].get("final", {}).get("status") == "done" and (self.d / "final.mp4").is_file():
            self.event("takes_canceled", note="only the retakes were stopped: the finished video stays")
            return self._finish("done")                     # a finished run whose RETAKES were canceled stays done
        return self._finish("canceled")

    def _finish(self, status: str) -> dict:
        self.st["status"] = status
        self.st.pop("worker", None)
        self.save()
        self.event("run_" + status, final=str(self.d / "final.mp4") if status == "done" else None,
                   error=self.st.get("error"))
        return self.st

    # steps
    def step_passes(self, step: dict) -> None:
        """Render the raw passes once per pass key: under a per-shot lock, into a staging folder that
        is checked and then renamed, so no run ever sees half-written or mixed passes."""
        if (self.pass_dir / "passes.json").is_file():
            step["outputs"] = {"passes": str(self.pass_dir), "cached": True}
            return
        self.shot_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(self.shot_dir / "build.lock", timeout=3600):
            if (self.pass_dir / "passes.json").is_file():             # another run built it meanwhile
                step["outputs"] = {"passes": str(self.pass_dir), "cached": True}
                return
            stage = self.shot_dir / f"passes.staging-{os.getpid()}"
            shutil.rmtree(stage, ignore_errors=True)
            if self.m.get("template"):                 # a video template: its frames + the stand-in's pose (DWPose)
                self.event("passes", job="video_passes", note="DWPose reads the template's stand-in (CPU)")
                self._video_passes(stage)
            else:
                self.event("blender", job="render_passes", note="Blender renders the shot in the background")
                blender.run_job("render_passes.py", self.shot_blend(),
                                ["--out", str(stage), "--character", ",".join(
                                    [r["collection"] for r in run_characters(self.m)]
                                    or [self.m["character_collection"]]), "--rig",
                                 self.m["rig"]], log=self.d / "logs" / "render_passes.log")
            try:
                passes.load(stage)                                        # frame counts agree
            except (OSError, ValueError, KeyError) as exc:
                raise RunError(f"Blender wrote incomplete passes ({exc}); see {self.d / 'logs' / 'render_passes.log'}") from exc
            passes.make_controls(stage, [CONTROL_KIND], control_size(passes.load(stage)))
            if self.pass_dir.exists():
                shutil.rmtree(self.pass_dir)
            os.replace(stage, self.pass_dir)
        step["outputs"] = {"passes": str(self.pass_dir), "cached": False}

    def _video_passes(self, stage: Path) -> None:
        """cinestudio/video_passes.py in ComfyUI's Python (it has DWPose), on the CPU."""
        import subprocess
        from . import comfy
        cfg = comfy.load_config() or comfy.detect()
        log = self.d / "logs" / "video_passes.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            fig = ["--mask", str(self.d / "inputs" / "figure_mask.mp4")] \
                if self.m.get("figure_mask", "off") != "off" else []
            proc = subprocess.run([cfg["python"], "-B", str(ROOT / "cinestudio" / "video_passes.py"),
                                   str(self.d / self.m["template"]["file"]), str(stage), *fig], stdout=fh,
                                  stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if proc.returncode != 0:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
            raise RunError(f"the template passes failed (exit {proc.returncode}); log: {log}\n" + "\n".join(tail))

    def people_control(self) -> Path | None:
        """Shots with two or more mannequins: the control comes from DWPose on each character's own render in
        its Blender box, checked by Blender (bpy_jobs/render_people.py, cinestudio/pose_people.py). Built once
        per pass key; None for a shot with one mannequin (the control is then drawn from the rig)."""
        pdir = self.pass_dir / "people"
        ctl = pdir / "control.mp4"
        if ctl.is_file():
            return ctl
        if (pdir / "people.json").is_file() and not _read_json(pdir / "people.json")["characters"]:
            return None
        from . import comfy
        with FileLock(self.shot_dir / "build.lock", timeout=3600):
            if ctl.is_file():
                return ctl
            info = passes.load(self.pass_dir)
            pct = max(1, round(100 * control_size(info)[0] / info["width"]))
            stage = self.shot_dir / f"people.staging-{os.getpid()}"
            shutil.rmtree(stage, ignore_errors=True)
            self.event("blender", job="render_people", note="Blender renders each character alone")
            blender.run_job("render_people.py", self.shot_blend(), ["--out", stage, "--percent", pct],
                            self.d / "logs" / "render_people.log")
            if not _read_json(stage / "people.json")["characters"]:
                pdir.mkdir(parents=True, exist_ok=True)
                os.replace(stage / "people.json", pdir / "people.json")
                shutil.rmtree(stage, ignore_errors=True)
                return None
            cfg = comfy.load_config() or comfy.detect()
            root = (Path(cfg["cwd"]) / cfg["main"]).parent
            aux = root / "custom_nodes" / "comfyui_controlnet_aux"
            if not (aux / "src").is_dir():
                raise RunError(f"several characters need DWPose (comfyui_controlnet_aux), not found in {aux}")
            self.event("pose", note="DWPose reads each character (CPU)")
            log = self.d / "logs" / "pose_people.log"
            with open(log, "w", encoding="utf-8", errors="replace") as fh:
                # the face rule is pinned (not taken from the environment): the cache key does not hold it
                r = subprocess.run([cfg["python"], "-B", str(ROOT / "cinestudio" / "pose_people.py"), str(stage),
                                    str(stage / "frames"), "--aux", str(aux), "--face-rule", "nose"],
                                   stdout=fh, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if r.returncode != 0:
                raise RunError(f"pose_people failed (exit {r.returncode}); log: {log}")
            frames = sorted((stage / "frames").glob("f_*.png"))
            passes.write_video([cv2.imread(str(f)) for f in frames], stage / "control.mp4", float(info["fps"]))
            if pdir.exists():
                shutil.rmtree(pdir)
            os.replace(stage, pdir)
            self.event("people_control", stats=_read_json(pdir / "frames" / "stats.json"))
        return ctl

    def people_under(self, skeleton: Path, kind: str) -> Path:
        """The several-character skeleton over the set edges without the characters (the camera's turn).
        Built once per pass key."""
        dst = skeleton.with_name(f"control_{kind}.mp4")
        if dst.is_file():
            return dst
        with FileLock(self.shot_dir / "build.lock", timeout=3600):
            if not dst.is_file():
                info = passes.load(self.pass_dir)
                size = control_size(info)
                t = passes.fit_transform(info["width"], info["height"], size)
                frames = sorted((skeleton.parent / "frames").glob("f_*.png"))
                if len(frames) != info["frames"]:
                    raise RunError(f"{len(frames)} skeleton frames but the shot has {info['frames']}")
                bases = passes.people_env(self.pass_dir, skeleton.parent, size)
                out = []
                for f, base in zip(frames, bases):
                    sk = cv2.imread(str(f))
                    if sk.shape != base.shape:            # alone-renders at a rounded percent: fit like the render
                        tf = passes.fit_transform(sk.shape[1], sk.shape[0], size)
                        sk = passes.fit_image(sk, tf, cv2.INTER_NEAREST)
                    out.append(passes.over(sk, base))
                passes.write_video(out, dst, float(info["fps"]))
        return dst

    def step_controls(self, step: dict) -> None:
        kind = self.m.get("control_kind") or CONTROL_KIND
        body = self.m.get("body_type") or "chibi"
        ctl = self.pass_dir / "controls" / passes.control_name(kind, body)
        people = None if (self.m.get("control_video") or self.m.get("render_mode") == "layout") \
            else self.people_control()                     # layout clips use no control video (the rig is the layout)
        if self.m.get("control_video"):                    # a control made outside the run
            ctl = self.d / self.m["control_video"]["file"]
        elif people:                                       # several characters
            ctl = self.people_under(people, kind)
        elif not ctl.is_file():
            with FileLock(self.shot_dir / "build.lock", timeout=3600):
                if not ctl.is_file():
                    passes.make_controls(self.pass_dir, [kind], control_size(passes.load(self.pass_dir)), body)
        fmt = self.clip_format()
        if fmt["clips"] == 1 and fmt["shot_frames"] > fmt["frames"]:
            self.event("warning", message=f"the shot has {fmt['shot_frames']} frames; the clip has {fmt['frames']} "
                                          f"(LTX clips have 8k+1 frames)")
        if fmt["clips"] > 1:
            self.event("clips", clips=fmt["clips"], frames=fmt["frames"], shot_frames=fmt["shot_frames"])
        clip_ctl = self.work / "control.mp4"
        video.slice_control(ctl, clip_ctl, 1, fmt["frames"], fmt["width"], fmt["height"], fmt["fps"])
        step["outputs"] = {"control": str(clip_ctl), "control_full": str(ctl), **fmt}
        if self.m.get("source_denoise"):                  # video-to-video: the render itself, at the clip size
            step["outputs"]["source"] = str(self._source_slice(1, fmt))

    def step_sheet(self, step: dict) -> None:
        sh = self.m["sheet"]                              # the run's copy with its prop panels (prop_images), if any;
        res = refsheet.prepare(self.d / (sh.get("props_file") or sh["file"]), self.work / "sheet")   # the keyframes
        #                                                 cut their panels from the character's sheet ("file")
        step["outputs"] = {"sheet": res["sheet_black"], "views": len(res["views"])}

    def step_restyle(self, step: dict) -> str | None:
        sheet = self.st["steps"]["sheet"]["outputs"]["sheet"]
        wf = self.workflow("restyle")
        fitted = self.work / "restyle_in.png"
        video.fit_image(sheet, fitted, 1536, 896, mode="pad", color="black")
        values = {"prompt": self._sheet_prompt(), "negative": "worst quality, blurry, distorted, text, watermark",
                  "width": 1536, "height": 896, "seed": self.m["seed"], "keep_strength": self.m["restyle_keep"],
                  "style_lora": self.m["style_lora"], "style_lora_strength": RESTYLE_LORA_STRENGTH}
        dst = self.work / "sheet_styled.png"
        outcome = self._comfy(step, "restyle", wf, values, {wf.targets("reference")[0]: fitted},
                              {wf.bind["outputs"]["image"]: dst}, extra_prefixes={"output_prefix_last": "_last"})
        if outcome:
            return outcome
        step["outputs"] = {"sheet": str(dst)}
        return None

    def _pose_slice(self, k: int, fmt: dict, body: str, out_dir: Path) -> Path:
        """Clip k's skeleton video (the rig's points, face points by OpenPose's visibility rules) at the generation
        size, drawn standard (DWPose node) or chibi; the last frame is held when the shot ends inside the clip."""
        info = passes.load(self.pass_dir)
        t = passes.fit_transform(info["width"], info["height"], (fmt["width"], fmt["height"]))
        kp = info["keypoints"]
        draw = passes.draw_pose_standard if body == "standard" else passes.draw_pose
        start = video.clip_start(k, fmt["frames"], fmt.get("overlap", 1))
        frames = []
        for i in range(fmt["frames"]):
            j = min(start + i, len(kp["frames"]) - 1)
            hands = [passes.fit_points(hp, t) for hp in passes.hands_at(kp, j)]
            if body == "standard":
                hands = hands[::-1]
            frames.append(draw(passes.fit_points(kp["frames"][j], t), t[3], t[4], hands))
        dst = out_dir / f"pose_{k}.mp4"
        passes.write_video(frames, dst, fmt["fps"])
        return dst

    def _source_slice(self, k: int, fmt: dict) -> Path:
        """Video-to-video source of clip k: the render's frames of that clip at the clip size (the last frame
        is held when the shot ends inside the clip)."""
        info = passes.load(self.pass_dir)
        t = passes.fit_transform(info["width"], info["height"], (fmt["width"], fmt["height"]))
        start = video.clip_start(k, fmt["frames"], fmt.get("overlap", 1))
        rgb = info["rgb"]
        pick = [rgb[min(start + i, len(rgb) - 1)] for i in range(fmt["frames"])]
        src = self.work / ("source.mp4" if k == 1 else f"source_{k}.mp4")
        passes.write_video([passes.fit_image(cv2.imread(str(p)), t) for p in pick], src, fmt["fps"])
        return src

    def step_background(self, step: dict) -> str | None:
        """The run's panorama world: made from the background text with Qwen-Image-2512 + the 360 LoRA (or the saved
        panorama frozen into the run), its seam closed, saved in the background library, then put into the run's copy
        of the shot (bpy_jobs/pano_world.py: the world + a ground from the panorama; the template's set hidden)."""
        from . import background as bgs
        out = step.get("outputs") or {}
        built = (out.get("ground") or "").split("-d")[0]
        wanted = pano_world_tag({} if step.get("depth_error") else self.m)   # a depth that failed was built as dome
        if out.get("blend") and Path(out["blend"]).is_file() and built == wanted:
            return None                                   # made with this pano_world code and these arguments
            # (a depth ground's tag also carries the depth file's hash: the passes' folder follows a new depth)
        if self.m["background"] == "plate":
            return self._plate_background(step)
        pano = self.work / "pano.png"
        mode = self.m["background"]
        if mode == "text" and not pano.is_file():
            scene = self.m.get("background_text") or self.m["prompt"]["video"]
            prompt = bgs.prompt_for(scene, self.m["prompt"].get("style") or "")
            seed = int(self.m.get("background_seed") or self.m["seed"])
            wf = ProfileWorkflow("qwenimage", "pano", workflows_dir=self.d / "inputs" / "workflows")
            values = {"prompt": prompt, "negative": bgs.NEGATIVE, "seed": seed, "width": bgs.WIDTH, "height": bgs.HEIGHT}
            self.event("background", note="Qwen-Image makes the 360 panorama of the place")
            outcome = self._comfy(step, "background", wf, values, {}, {wf.output_node: pano})
            if outcome:
                return outcome
            bgs.fix_seam(pano)
            rec = bgs.save(pano, prompt, seed, scene=scene, style=self.m["prompt"].get("style") or "")
            step["library"] = rec["name"]
        elif mode != "text":
            shutil.copyfile(self.d / "inputs" / "background.png", pano)
        blend = self.work / "shot_bg.blend"
        extra = []
        world_m = self.m                                    # the ground mode actually built (dome after a depth error)
        if self.m.get("background_ground") == "depth":      # the panorama's 360 depth (MoGe-2), reused when the
            from . import pano_depth                        # library panorama already has it for these pixels
            sha = _sha256(pano)
            npy = pano_depth.load_if_current(pano, sha)
            lib = step.get("library") or (mode.split(":", 1)[1] if mode.startswith("saved:") else None)
            if npy is None and lib:
                try:
                    have = pano_depth.load_if_current(bgs.path(lib), sha)
                except bgs.BackgroundError:
                    have = None
                if have:
                    npy = pano_depth.save(np.load(have), pano, sha)
            if npy is None:
                glb = self.work / "pano_depth.glb"
                wf = ProfileWorkflow("moge", "pano", workflows_dir=self.d / "inputs" / "workflows")
                self.event("background", note="MoGe-2 measures the distance in every direction of the panorama")
                part = step.setdefault("parts", {}).setdefault("depth", {})   # its OWN job record (Codex: the step's
                outcome = self._comfy(part, "depth", wf, {}, {wf.targets("pano")[0]: pano},   # record is the panorama
                                      {wf.output_node: glb})                                  # job of a text background
                if outcome:
                    return outcome
                try:
                    npy = pano_depth.save(pano_depth.distance_map(glb), pano, sha)
                except pano_depth.DepthError as e:          # an unusable depth (all sky, an unexpected layout): the
                    npy = None                              # dome, and the run says so (Codex review 2026-10-22)
                    step["depth_error"] = str(e)
                    self.event("background", note=f"the panorama's depth cannot be used ({e}): the dome ground instead")
            if npy is not None:
                extra = ["--depth", str(npy)]
                step["depth_sha"] = _sha256(Path(npy))
            else:
                world_m = {}
        self.event("blender", job="pano_world", note="Blender puts the panorama round the shot")
        blender.run_job("pano_world.py", self.d / self.m["blend"]["file"],
                        ["--pano", str(pano), "--out", str(blend), "--character", self.character_collections(),
                         *pano_args(world_m), *extra],
                        log=self.d / "logs" / "pano_world.log")
        tag = pano_world_tag(world_m) + (f"-d{step['depth_sha'][:8]}" if extra else "")
        step["outputs"] = {"pano": str(pano), "blend": str(blend), "sha256": _sha256(pano), "ground": tag,
                           "library": step.get("library") or (mode.split(":", 1)[1] if mode.startswith("saved:") else None)}
        return None

    def _plate_background(self, step: dict) -> str | None:
        """Background "plate" (still cameras): ONE picture of the place at the frame's shape instead of a 360 panorama
        (the template camera sees ~350 px of a 2048 px panorama), put into the run's copy of the shot by
        bpy_jobs/plate_world.py (a camera projection onto a floor and a far wall). Made from the background text with
        Qwen-Image-2512 (the panorama graph, its 360 LoRA at 0), or the user's own picture (inputs/plate.*)."""
        from . import background as bgs
        tag = plate_world_tag()
        out = step.get("outputs") or {}
        if out.get("blend") and Path(out["blend"]).is_file() and out.get("ground") == tag:
            return None
        shot = self.d / self.m["blend"]["file"]
        info_f = self.work / "plate_camera.json"
        if not info_f.is_file():
            blender.run_job("plate_world.py", shot, ["--info", str(info_f)], log=self.d / "logs" / "plate_world.log")
        cam = _read_json(info_f)
        if cam.get("moves"):
            raise RunError("a backplate needs a still camera: this shot's camera moves (use --background text, a 360 "
                           "panorama)")
        w, h = bgs.plate_size(cam)
        plate = self.work / "plate.png"
        own = sorted((self.d / "inputs").glob("plate.*"))
        lib = self.m.get("background_library")
        if lib:                                           # a saved backplate: was it made for this camera?
            try:
                made = bgs.load(lib).get("camera") or {}
            except bgs.BackgroundError:
                made = {}
            if made and abs(float(made.get("horizon_from_top", 0.5)) - float(cam.get("horizon_from_top", 0.5))) > 0.08:
                self.event("warning", note=f"the backplate {lib} was made for another camera ({bgs.camera_summary(made)}; "
                           f"this shot: {bgs.camera_summary(cam)}): its horizon may not meet the floor")
        if not plate.is_file():
            if own:                                           # the user's picture, cropped to the frame's shape
                video.fit_image(own[0], plate, w, h, mode="cover")
            else:
                scene = (self.m.get("background_text") or self.m["prompt"]["video"]).strip().rstrip(".")
                style = (self.m["prompt"].get("style") or "").strip()
                prompt = bgs.plate_prompt(scene, style, cam)
                seed = int(self.m.get("background_seed") or self.m["seed"])
                wf = ProfileWorkflow("qwenimage", "pano", workflows_dir=self.d / "inputs" / "workflows")
                values = {"prompt": prompt, "negative": bgs.NEGATIVE, "seed": seed, "width": w, "height": h,
                          "lora_strength": 0.0}
                step["prompt"] = prompt
                self.event("background", note="Qwen-Image makes the backplate of the place (one picture, still camera)")
                outcome = self._comfy(step, "background", wf, values, {}, {wf.output_node: plate})
                if outcome:
                    return outcome
                try:                                      # kept in the library for the next video (Backgrounds tab)
                    rec = bgs.save_plate(plate, prompt, seed, scene=scene, style=style, camera=cam, source="run",
                                         run=self.id)
                    step["library"] = rec["name"]
                except Exception as exc:  # noqa: BLE001 - the run goes on; only the library copy is missing
                    self.event("warning", note=f"the backplate was not saved in the library: {exc}")
        blend = self.work / "shot_plate.blend"
        self.event("blender", job="plate_world", note="Blender puts the backplate behind the shot")
        blender.run_job("plate_world.py", shot, ["--plate", str(plate), "--out", str(blend),
                                                 "--character", self.character_collections()],
                        log=self.d / "logs" / "plate_world.log")
        step["outputs"] = {"plate": str(plate), "pano": str(plate), "blend": str(blend), "sha256": _sha256(plate),
                           "ground": tag, "camera": cam, "own_image": bool(own),
                           "library": step.get("library") or self.m.get("background_library")}
        return None

    # "Picture 2" = ComfyUI's own label of the second image (AGENTS 4.3); no "sky" (an indoor set has none:
    # docs/look_image_research.md)
    LOOK_INSTRUCTION = ("Replace the white wooden mannequin with the character from Picture 2: the same pose, the same "
                        "body direction, the same position and size in the picture, the same arms and legs. The "
                        "character keeps every detail from Picture 2: {who} {extra} Keep the camera, the place, the "
                        "walls, the floor and the light exactly the same.")

    def step_look(self, step: dict) -> str | None:
        """Layout mode: the LOOK image = the first Blender frame (with its panorama) where Qwen-Image-Edit replaced the
        mannequin by the character (its front view as image 2, its description in the text). Lightricks' Layout-To-Render
        recipe: an image model redraws the first layout frame; never the grey frame itself (a grey look gives a grey
        video). A first frame given at `run new` is used as the look instead."""
        look = self.work / "look.png"
        if self.m.get("first_frame"):
            if not look.is_file():
                shutil.copyfile(self.d / self.m["first_frame"]["file"], look)
            step["outputs"] = {"look": str(look), "made": False}
            return None
        rgb = passes.load(self.pass_dir)["rgb"]
        scene = (self.m.get("look_scene") or "").strip()
        person = self.work / "look_person.png" if scene else look
        if not look.is_file():
            if len(run_characters(self.m)) > 1 and not person.is_file():   # several characters: one crop per
                outcome = self._group_look(step, Path(rgb[0]), passes.load(self.pass_dir)["mask"][0], person)   # figure
                if outcome:
                    return outcome
            if not person.is_file():
                outcome = self._pinned_look(step, "look", Path(rgb[0]), self._figure_mask(0), "front", person)
                if outcome:
                    return outcome
            if scene:                                   # the place from the text (user, 2026-10-23: "use prompt to
                outcome = self._qwen_scene(step, person, scene, look)   # create the background rather than having the
                if outcome:                                             # panorama determine what it looks like")
                    return outcome
        keys = []
        if scene and self.m.get("look_keyframes", "auto") == "auto":
            self.event("keyframes_off", note="look_scene: the keyframe looks come from the grey Blender frames and "
                                             "would pull the video back to the grey room")
        if not scene and self.m.get("look_keyframes", "auto") == "auto":
            fmt = self.st["steps"]["controls"]["outputs"]
            yaws = self._body_yaws()
            n, ov, frames = fmt.get("clips", 1), int(fmt.get("overlap", 1)), fmt["frames"]
            for k in range(1, n + 1):
                local = frames // 2
                shot = video.clip_start(k, frames, ov) + local
                if shot >= len(rgb) or yaws is None or shot >= len(yaws) or yaws[shot] is None:
                    continue
                deg, side = yaws[shot]
                if deg < 30:                                # the body still faces the camera: the first look holds
                    continue
                if deg > 135:                               # R3 (2026-10-21): with the BACK view as image 2, Qwen drew
                    #                                         the FRONT (mask to the camera; clip 2 copied it: 71/121
                    #                                         mirrored frames). The sheet's own back panel pasted into
                    #                                         the Blender frame gave a full poncho (take 11)
                    out = self.work / f"look_k{k}_sheet.png"
                    if self._sheet_keyframe(rgb[shot], passes.load(self.pass_dir)["mask"][shot], "back", out):
                        keys.append({"clip": k, "frame": local, "file": str(out), "view": "back", "body_deg": deg,
                                 "source": "sheet"})
                    continue
                view = "front" if deg < 45 else ("right" if side == "left" else "left")
                out = self.work / f"look_k{k}.png"
                if not out.is_file():
                    outcome = self._pinned_look(step, f"look_k{k}", Path(rgb[shot]), self._figure_mask(shot), view,
                                                out)
                    if outcome:
                        return outcome
                keys.append({"clip": k, "frame": local, "file": str(out), "view": view, "body_deg": deg})
                if view in ("right", "left"):              # + the sheet's side panel at the clip's purest profile:
                    span = range(video.clip_start(k, frames, ov), min(video.clip_start(k, frames, ov) + frames,
                                                                     len(rgb), len(yaws)))
                    prof = [(abs(yaws[i][0] - 90), i) for i in span if yaws[i] is not None and yaws[i][1] == side]
                    if prof:                                # R3 clip 1 (2026-10-21): kept the poncho in the side views
                        _, ps = min(prof)
                        pl = ps - video.clip_start(k, frames, ov)
                        if abs(pl - local) >= 16 and 0 < pl < frames - 1:
                            pout = self.work / f"look_k{k}_profile.png"
                            if self._sheet_keyframe(rgb[ps], passes.load(self.pass_dir)["mask"][ps], view, pout):
                                keys.append({"clip": k, "frame": pl, "file": str(pout), "view": view,
                                         "body_deg": yaws[ps][0], "source": "sheet"})
            heads = self._head_yaws()
            for k in range(2, n + 1):                       # the MASK: later clips do not see the run's front look;
                s0 = video.clip_start(k, frames, ov)        # R3 clip 3 (2026-10-21) turned the mask into a face (nose,
                span = [i for i in range(s0 + 1, min(s0 + frames - 1, len(rgb), len(heads or [])))   # not the clip's
                        if heads[i] is not None]            # first or last frame: the best frame inside (R4 clip 3)
                if not span:                                # lips) until the sheet's FRONT panel was a keyframe where
                    continue                                # the head faces the camera
                hd, hi = min((heads[i][0], i) for i in span)
                fl = hi - s0
                near = [kf["frame"] for kf in keys if kf["clip"] == k]
                if hd >= 25 or not 0 < fl < frames - 1 or any(abs(fl - x) < 16 for x in near):
                    continue
                fout = self.work / f"look_k{k}_face.png"
                if self._sheet_keyframe(rgb[hi], passes.load(self.pass_dir)["mask"][hi], "front", fout):
                    keys.append({"clip": k, "frame": fl, "file": str(fout), "view": "front", "head_deg": hd,
                             "source": "sheet"})
            last = min(len(rgb), len(heads or [])) - 1      # the shot ends inside the last clip: its tail holds the last
            for k in range(2, n + 1):                       # layout frame, and LTX drifts there (R4 clip 4: the mask
                s0 = video.clip_start(k, frames, ov)        # became a face; a sheet front at frame 100 kept it)
                held = s0 + frames - 1 - last
                if held < 24 or heads is None or heads[last] is None or heads[last][0] >= 25:
                    continue
                tl = frames - 21
                if any(abs(tl - kf["frame"]) < 16 for kf in keys if kf["clip"] == k):
                    continue
                tout = self.work / f"look_k{k}_tail.png"
                if self._sheet_keyframe(rgb[last], passes.load(self.pass_dir)["mask"][last], "front", tout):
                    keys.append({"clip": k, "frame": tl, "file": str(tout), "view": "front", "head_deg": heads[last][0],
                             "source": "sheet", "held": held})
        step["outputs"] = {"look": str(look), "made": True, "keyframes": keys}
        return None

    def _head_yaws(self) -> list | None:
        """Per shot frame the rig's head direction [degrees from the camera, picture side], or None."""
        try:
            return _read_json(self.pass_dir / "keypoints.json").get("head_yaw")
        except (OSError, ValueError):
            return None

    def _body_yaws(self) -> list | None:
        """Per shot frame the rig's body direction [degrees from the camera, picture side] (render_passes), or None."""
        try:
            return _read_json(self.pass_dir / "keypoints.json").get("body_yaw")
        except (OSError, ValueError):
            return None

    def _sheet_keyframe(self, rgb, mask, view: str, out: Path) -> bool:
        """A keyframe made from the character's own view (the run's copy) or the sheet's panel; False (and an event)
        when it cannot be made - the run goes on without it (an imported sheet of another layout: Codex 10-22)."""
        from . import sheet_keyframe
        if out.is_file():
            return True
        vfile = (self.m.get("views") or {}).get(view)
        try:
            sheet_keyframe.make(rgb, mask, self.d / self.m["sheet"]["file"], view, out,
                                view_image=self.d / vfile if vfile else None,
                                fill_gaps=bool(self.m.get("sheet_fill_gaps", "views" not in self.m)))   # old runs: on
            return True
        except (sheet_keyframe.LayoutError, RuntimeError, OSError, ValueError, IndexError) as e:
            self.event("keyframe_skipped", file=out.name, view=view, reason=str(e))
            return False

    # a crop of the frame (several characters): no "sky" (a crop of a gym wall came back with a blue sky, trio_test
    # 2026-10-25); the background is named as it is in image 1
    CROP_LOOK_INSTRUCTION = ("Replace the white wooden mannequin with the character from Picture 2: the same pose, the "
                             "same body direction, the same position and size in the picture, the same arms and legs. "
                             "The character keeps every detail from Picture 2: {who} {extra} Keep the camera, the "
                             "background behind the figure, the floor and the light exactly as they are in Picture 1.")
    GROUP_LOOK_INSTRUCTION = ("Replace the white wooden mannequin {place} with the character from Picture 2: the same "
                              "pose, the same body direction, the same position and size in the picture, the same arms "
                              "and legs. The character keeps every detail from Picture 2: {who} {extra} Keep the other "
                              "figures, the camera, the background behind the figures, the floor and the light exactly "
                              "as they are in Picture 1.")

    def character_collections(self) -> str:
        """Every character collection of the run, comma-separated (bpy_jobs/plate_world.py, pano_world.py keep them)."""
        return ",".join([r["collection"] for r in run_characters(self.m)] or [self.m["character_collection"]])

    def _figure_xs(self, width: int) -> list | None:
        """Each slot's figure x in pixels of a `width` frame at frame 1 (duplicate_mannequin's screen_x, 0..1), else
        None (old runs: slot order = left to right)."""
        xs = (self.m.get("mannequins") or {}).get("screen_x") or []
        n = len(run_characters(self.m))
        if len(xs) != n or any(x is None for x in xs):
            return None
        return [float(x) * width for x in xs]

    def _group_look(self, step: dict, frame: Path, mask_file, out: Path) -> str | None:
        """The look image of a several-character run by CROP AND STITCH (2026-10-23 for two, 2026-10-25 for N): the
        first layout frame's figures are found in the characters' mask (plan_figure_crops: one crop per figure that
        stands apart); each crop (+ a margin) is cut out, made ~1024 px large, gets the single-character look edit with
        its own character (its frozen front view, its own description) and is pasted back with soft edges. Whole-frame
        edits (one per side, duo runs 1-4) drew the second character at its reference's size and dropped masked3d's
        hair - the single-character look kept both. FALLBACK for figures that overlap or touch (no gap between them):
        one shared crop, edited once per figure in turn (left to right, each edit naming its figure's place among the
        figures of the crop: "on the left", "in the middle" ...); an event "group_look" records it as "shared"."""
        full = cv2.imread(str(frame))
        if full is None:
            raise RunError(f"cannot read the first layout frame {frame}")
        H, W = full.shape[:2]
        m = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RunError(f"cannot read the first mask {mask_file}")
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 127
        n = len(run_characters(self.m))
        plan = plan_figure_crops(m, n, self._figure_xs(W))
        first = True
        for gi, crop_plan in enumerate(plan):
            x0, y0, x1, y1 = crop_plan["box"]
            res = self.work / f"look_g{gi + 1}_crop_out.png"
            if res.is_file():
                first = False
                continue
            src = self.work / f"look_g{gi + 1}_crop.png"
            k = 1024 / max(x1 - x0, y1 - y0)
            cv2.imwrite(str(src), cv2.resize(full[y0:y1, x0:x1], (round((x1 - x0) * k), round((y1 - y0) * k)),
                                             interpolation=cv2.INTER_CUBIC))
            slots = crop_plan["slots"]
            for j, slot in enumerate(slots):            # one edit per figure; a shared crop: in turn, on the last result
                dst = res if j == len(slots) - 1 else self.work / f"look_g{gi + 1}_f{j + 1}.png"
                if dst.is_file():
                    src = dst
                    first = False
                    continue
                place = place_word(j, len(slots)) if len(slots) > 1 else None
                outcome = self._figure_look(step, "look" if first else f"look_fig{slot + 1}", src, slot, dst, place)
                first = False
                if outcome:
                    return outcome
                src = dst
        full = stitch_look(full, m, plan, [cv2.imread(str(self.work / f"look_g{gi + 1}_crop_out.png"))
                                           for gi in range(len(plan))])   # the figures only: the frame stays around
        if not cv2.imwrite(str(out), full):
            raise OSError(f"cannot write {out}")
        shared = [c for c in plan if c["shared"]]
        self.event("group_look", crops=[{"box": list(c["box"]), "slots": [sl + 1 for sl in c["slots"]],
                                         "shared": c["shared"]} for c in plan],
                   note="the look image: one single-character edit per figure, cropped and pasted back"
                        + (f"; {len(shared)} crop(s) with figures that touch: edited once per figure in the same "
                           "crop (fallback)" if shared else ""))
        if shared:
            step["look_fallback"] = "shared crop"
        return None

    def _figure_look(self, step: dict, name: str, frame: Path, slot: int, out: Path,
                     place: str | None = None) -> str | None:
        """One figure of a several-character look: Qwen-Image-Edit puts the character of `slot` (0-based; its frozen
        front view as image 2, its own description) into `frame` (a crop of the Blender frame). `place` ("on the
        left", ...) names the figure among several in a shared crop (the overlap fallback)."""
        from . import character as chars
        group = run_group(self.m) or {}
        figs = group.get("figures") or []
        who = (figs[slot].get("description") if slot < len(figs) else "") or ""
        vs = run_views(self.m)[slot] if slot < len(run_views(self.m)) else {}
        wf = ProfileWorkflow("qwenedit", "edit", workflows_dir=self.d / "inputs" / "workflows")
        values, _ = chars.quality_values("quality", {}, chars.EDIT_SETTINGS)
        extra = (self.m.get("look_extra") or "") if slot == 0 else ""
        text = (self.GROUP_LOOK_INSTRUCTION.format(place=place, who=who, extra=extra) if place
                else self.CROP_LOOK_INSTRUCTION.format(who=who, extra=extra))
        values.update(instruction=text.replace("  ", " "), seed=int(self.m["seed"]))
        ref = self.d / (vs.get("front") or next(iter(vs.values())))
        files = {wf.targets("image")[0]: frame, wf.targets("reference")[0]: ref}
        self.event("look", image=name, view="front", character=slot + 1, place=place,
                   note=f"Qwen-Image-Edit puts character {slot + 1} into its crop of the Blender frame")
        return self._comfy(step if name == "look" else step.setdefault("parts", {}).setdefault(name, {}), name, wf,
                           values, files, {wf.output_node: out})

    def _qwen_look(self, step: dict, name: str, frame: Path, view: str, out: Path, mask: Path | None = None,
                   seed: int | None = None) -> str | None:
        """One look image: Qwen-Image-Edit puts the character (its `view` from the library as image 2) into `frame`.
        Several characters: Run._figure_look per figure."""
        from . import character as chars
        # frozen inputs first (Codex review 2026-10-22: a deleted library character must not break a run that has its
        # own copies): the look text, else the description frozen in the run's prompt; the library only as a fallback
        frozen = self.m.get("views") or {}
        rv = frozen.get(view) or frozen.get("front")
        frozen_desc = ((self.m.get("prompt") or {}).get("character") or "")
        c = None if rv else chars.load(self.m["character_name"])   # old runs (no frozen views): the library
        if c is not None:
            desc = (c.get("description") or "").split("on black: ", 1)[-1].strip()
        elif "on black: " in frozen_desc:               # views: "... on black: <description>"
            desc = frozen_desc.split("on black: ", 1)[1].strip()
        elif frozen_desc.rstrip().endswith("on black."):  # a textless import (character.for_run): no description
            desc = ""
        else:                                           # a sheet import: the description itself
            desc = frozen_desc.strip()
        who = self.m.get("look_text") or desc
        wf = ProfileWorkflow("qwenedit", "edit", workflows_dir=self.d / "inputs" / "workflows")
        values, _ = chars.quality_values("quality", {}, chars.EDIT_SETTINGS)
        values.update(instruction=self.LOOK_INSTRUCTION.format(who=who, extra=self.m.get("look_extra") or "").replace("  ", " "),
                      seed=int(self.m["seed"] if seed is None else seed))
        ref = self.d / rv if rv else Path((c.get("files") or {}).get(view) or c["files"]["front"])
        files = {wf.targets("image")[0]: frame, wf.targets("reference")[0]: ref}
        edit = None
        if mask is not None and self.m.get("look_pin"):          # the room is pinned (look_pin.py): the frame at an
            from . import look_pin                               # encoder-stable size, no centre crop, the figure's
            prep = look_pin.prepare(frame, mask, self.work / "look_in", name)   # soft mask as a latent noise mask
            files[wf.targets("image")[0]] = Path(prep["frame"])
            files[("look_mask", "image")] = Path(prep["mask"])
            edit = look_pin.graph_edit("look_mask")
        self.event("look", image=name, view=view, pinned=edit is not None,
                   note="Qwen-Image-Edit puts the character into a Blender frame (a look image)")
        return self._comfy(step if name == "look" else step.setdefault("parts", {}).setdefault(name, {}), name, wf,
                           values, files, {wf.output_node: out}, graph_edit=edit)

    def _figure_mask(self, i: int) -> Path | None:
        """The figure mask of shot frame `i` from the passes, or None (no mask pass: the plain look)."""
        masks = passes.load(self.pass_dir).get("mask") or []
        if i >= len(masks) or not Path(masks[i]).is_file():
            return None
        m = cv2.imread(str(masks[i]), cv2.IMREAD_UNCHANGED)
        if m is None or (m[..., :3] if m.ndim == 3 else m).max() <= 127:   # no figure in the frame: nothing to pin
            return None
        return Path(masks[i])

    def _offset_look(self, step: dict, name: str, out: Path) -> None:
        """The kept pinned look moved a few pixels (look_pin.offset): a look that lines up with the layout pixel for
        pixel made the video ~35% darker. The gate's tries (look_tryN.png) keep the unmoved picture."""
        from . import look_pin
        dx, dy = look_pin.offset(out)
        step.setdefault("look_offsets", {})[name] = [dx, dy]
        self.event("look_offset", image=name, dx=dx, dy=dy)
        self.save()
        return None

    def _pinned_look(self, step: dict, name: str, frame: Path, mask: Path | None, view: str, out: Path) -> str | None:
        """A look image with the room pinned and GATED (look_pin.check): when the look lost the background (a banner,
        the hoop, a shift), it is made again with the next seed, at most look_pin.MAX_TRIES times; the best try is
        kept. Old runs (no look_pin) and frames without a mask: one plain look, as before."""
        if mask is None or not self.m.get("look_pin"):
            return self._qwen_look(step, name, frame, view, out)
        from . import look_pin
        checks = step.setdefault("look_checks", {}).setdefault(name, [])
        target = step if name == "look" else step.setdefault("parts", {}).setdefault(name, {})
        while True:
            seed = int(self.m["seed"]) + len(checks)
            outcome = self._qwen_look(step, name, frame, view, out, mask=mask, seed=seed)
            if outcome:
                return outcome
            try:
                res = dict(look_pin.check(out, frame, mask), seed=seed)
            except (ValueError, cv2.error, TypeError, AttributeError) as exc:   # not a readable picture: keep it, say so
                self.event("look_check_skipped", image=name, reason=f"{type(exc).__name__}: {exc}"[:200])
                return None
            keep = out.with_name(f"{out.stem}_try{len(checks) + 1}{out.suffix}")
            shutil.copyfile(out, keep)
            res["file"] = keep.name
            checks.append(res)
            self.event("look_check", image=name, **{k: v for k, v in res.items() if k != "file"})
            self.save()
            if res["ok"]:
                return self._offset_look(step, name, out)
            if len(checks) >= look_pin.MAX_TRIES:                  # none passed: the best try (fewest lost tiles,
                best = min(checks, key=lambda r: (r["lost_tiles"], -r["ssim"]))   # then the highest SSIM)
                shutil.copyfile(out.with_name(best["file"]), out)
                self.event("look_check_failed", image=name, kept=best["file"],
                           note="no try kept the whole background; the best one is used")
                return self._offset_look(step, name, out)
            target.pop("job", None)                                # a new job for the next seed
            out.unlink()

    SCENE_INSTRUCTION = ("Replace the plain grey background and the grey floor with this place: {scene}. The ground is "
                         "where the grey floor is, at the same height and angle, and the horizon stays where it is. "
                         "Keep the character, its pose, its position and size in the picture, and the camera exactly "
                         "the same. Keep everything else exactly the same.")

    def _qwen_scene(self, step: dict, image: Path, scene: str, out: Path) -> str | None:
        """The second look edit of `look_scene`: Qwen-Image-Edit replaces the grey Blender room with the place in words
        (one change per edit: the character came in the edit before)."""
        from . import character as chars
        wf = ProfileWorkflow("qwenedit", "edit", workflows_dir=self.d / "inputs" / "workflows")
        values, _ = chars.quality_values("quality", {}, chars.EDIT_SETTINGS)
        values.update(instruction=self.SCENE_INSTRUCTION.format(scene=scene.rstrip(". ")), seed=int(self.m["seed"]))
        files = {wf.targets("image")[0]: image}
        if "reference" not in wf.bind.get("optional_images", {}):          # an old run's graph: the same picture twice
            files[wf.targets("reference")[0]] = image
        self.event("look", image="look_scene", note="Qwen-Image-Edit puts the place from the text into the look image")
        return self._comfy(step.setdefault("parts", {}).setdefault("look_scene", {}), "look_scene", wf, values, files,
                           {wf.output_node: out})

    def step_clip(self, step: dict) -> str | None:
        if self.m.get("render_mode") == "layout":
            return self._layout_clips(step)
        return self._clips(step)

    def _layout_clips(self, step: dict) -> str | None:
        """Layout-To-Render clips (see _layout_clip_job). Each finished clip is kept in step["parts"] with its latents
        (relay), so a stopped run continues with the next clip and a clip can be made again later (run retake)."""
        fmt = self.st["steps"]["controls"]["outputs"]
        parts = step.setdefault("parts", {})
        n = fmt.get("clips", 1)
        ov = int(fmt.get("overlap", 1))
        scale = 1
        limit = self.st.get("clip_limit")
        for k in range(1, n + 1):
            if parts.get(str(k), {}).get("outputs"):
                continue
            if limit and k > int(limit):                     # made through clip `limit`: judge it, retake it, go on
                return "paused"
            sub = step if k == 1 else parts.setdefault(str(k), {})
            dst = self.work / ("clip.mp4" if k == 1 else f"clip_{k}.mp4")
            if n > 1:
                self.event("clip_part", clip=k, of=n)
            res = self._layout_clip_job(sub, k, self.m, parts, dst, "clip" + ("" if k == 1 else str(k)),
                                        f"cinestudio/relay_{self.id}_{k}")
            if isinstance(res, str):
                return res
            scale = res["scale"]
            parts.setdefault(str(k), {})["outputs"] = {"clip": str(dst)}
            if res.get("relay"):
                parts[str(k)]["relay"] = res["relay"]
            self.save()
        scale = 1 if self.m.get("quality") == "preview" else \
            self.workflow("layouting" if float(self.m.get("ingredients_strength") or 0) > 0 else "layout") \
            .capabilities.get("output_scale", 1)                # a preview is stage 1 only (Codex: 1536 recorded)
        clips = [parts[str(k)]["outputs"]["clip"] for k in range(1, n + 1)]
        step["outputs"] = {"clip": clips[0], "clips": clips, "width": fmt["width"] * scale,
                           "height": fmt["height"] * scale, "frames": fmt["frames"], "fps": fmt["fps"],
                           "shot_frames": fmt.get("shot_frames", fmt["frames"]), "overlap": ov,
                           "camera_azimuth_deg": fmt.get("camera_azimuth_deg")}
        return None

    def _layout_clip_job(self, sub: dict, k: int, m: dict, parts: dict, dst: Path, job_name: str,
                         relay_prefix: str, end_anchor: dict | None = None) -> dict | str:
        """One Layout-To-Render clip with the settings `m` (the run's manifest, or it with a retake's changes):
        the Blender render is the layout guide at frame 0, the run's look image a guide at frame -1, the character
        sheet the Ingredients guide when its strength is above 0, the clip's keyframe look (look_keyframes), and for
        clip k > 1 the START of clip k-1's own latents (relay "latent") or its last pixel frames in place. `end_anchor`
        ({"s1", "s2"}: the latents of clip k+1) pins the END of the clip to the start of the clip after it (a retake
        between two clips). The clip's own latents are saved (relay) for the next clip and for retakes.
        Returns {"scale", "relay"} or an outcome string (canceled ...)."""
        fmt = self.st["steps"]["controls"]["outputs"]
        sheet = (self.st["steps"].get("restyle", {}).get("outputs") or self.st["steps"]["sheet"]["outputs"])["sheet"]
        ing = float(m.get("ingredients_strength") or 0) > 0
        wf = self.workflow("layouting" if ing else "layout")
        preview = m.get("quality") == "preview"             # stage 1 only (768x448): to_stage1_only below
        scale = 1 if preview else wf.capabilities.get("output_scale", 1)
        ref = self.work / "reference.png"
        if not ref.is_file():
            video.fit_image(sheet, ref, fmt["width"], fmt["height"], mode="pad", color="black")
        ov = int(fmt.get("overlap", 1))
        full = wf.capabilities.get("recipe") == "full"
        sig = dev_sigmas(fmt["width"], fmt["height"], fmt["frames"], int(m.get("steps") or DEV_STEPS)) if full \
            else DISTILLED_SIGMAS
        keys = {}
        for kf in (self.st["steps"]["look"].get("outputs") or {}).get("keyframes") or []:
            keys.setdefault(kf["clip"], []).append(kf)
        look = Path(self.st["steps"]["look"]["outputs"]["look"])
        chain_key = None                                    # look_chain "anchor": the chained frame as a frame-0 guide
        if k > 1 and m.get("look_chain", "run") in ("prev", "anchor"):   # clip k-1's picture where clip k starts
            prev_clip = Path(parts[str(k - 1)]["outputs"]["clip"])
            chained = dst.parent / f"look_chain_{k}.png"
            chained.parent.mkdir(parents=True, exist_ok=True)
            video.frame_png(prev_clip, chained, fmt["frames"] - int(fmt.get("overlap", 1)))
            if m.get("chain_light") in ("look", "tone"):    # the lightness (and with "tone" the colour and contrast)
                #                                                 of the run's first look, every clip
                info = passes.load(self.pass_dir)
                at = video.clip_start(k, fmt["frames"], int(fmt.get("overlap", 1)))
                rgb, masks = info.get("rgb") or [], info.get("mask") or []
                if at >= len(rgb) or at >= len(masks) or not masks:
                    sub["chain_light"] = {"skipped": f"the passes have no frame {at} (rgb {len(rgb)}, masks {len(masks)})"}
                else:
                    # the layout render's own BACKGROUND change between frame 0 and here (Codex review: a whole-frame
                    # ratio moved with the character's size); a darker scene or a moving camera keep their change
                    l0, w0 = video.background_lightness(rgb[0], masks[0])
                    lk, wk = video.background_lightness(rgb[at], masks[at])
                    ratio = lk / l0 if l0 > 1.0 and w0 == wk else 1.0
                    ref = look
                    if m.get("chain_light_ref", "look") == "clip1":   # clip 1's own start, not the look (the
                        #                                               video never reaches the look's lightness)
                        ref = dst.parent / f"clip1_light_{k}.png"     # read again each time: a clip 1 retake
                        video.frame_png(Path(parts["1"]["outputs"]["clip"]), ref, 0)
                    sub["chain_light"] = dict(video.match_lightness(chained, ref, mask=masks[at], ref_mask=masks[0],
                                                                    ratio=ratio, tone=m.get("chain_light") == "tone"),
                                              ref=m.get("chain_light_ref", "look"))
            if float(m.get("chain_sharpen") or 0) > 0:
                if m.get("chain_sharpen_mode") == "match":    # up to clip 1's own sharpness (about the smallest
                    #                                              amount that reaches it), not a fixed amount
                    clip1 = Path(parts["1"]["outputs"]["clip"])
                    probe = dst.parent / f"clip1_sharp_{k}.png"
                    vals = []
                    for i in (0, fmt["frames"] // 3, 2 * fmt["frames"] // 3, fmt["frames"] - int(fmt.get("overlap", 1))):
                        video.frame_png(clip1, probe, i)
                        vals.append(video.sharpness(probe))
                    probe.unlink(missing_ok=True)
                    factor = m.get("chain_sharpen_target") or (1.0 if m.get("quality") == "final" else 1.5)
                    target = float(np.median(vals)) * float(factor)
                    sub["chain_sharpen"] = video.sharpen_to(chained, target, float(m["chain_sharpen"]))
                else:
                    video.unsharp_png(chained, float(m["chain_sharpen"]))
            if m.get("chain_light") in ("look", "tone") or m.get("chain_sharpen_mode") == "match":   # clip k's look
                fin_mask = None                             # the background of the final look, as measured above
                try:
                    fin_info = passes.load(self.pass_dir)
                    at_f = video.clip_start(k, fmt["frames"], int(fmt.get("overlap", 1)))
                    fin_mask = (fin_info.get("mask") or [])[at_f]
                except (IndexError, KeyError, OSError, ValueError):
                    pass
                fin_l, fin_why = video.background_lightness(chained, fin_mask)
                sub["chain_final"] = {"sharpness": round(video.sharpness(chained), 1), "lightness": round(fin_l, 2),
                                      "lightness_region": fin_why or "background"}
            if m.get("chain_colour") == "match":           # whole-frame colour match to clip 1's first frame (test C)
                from . import chain_colour
                ref1 = dst.parent / f"clip1_frame0_{k}.png"   # read again each time: a clip 1 retake changes it
                video.frame_png(Path(parts["1"]["outputs"]["clip"]), ref1, 0)
                r = chain_colour.match(chained, ref1, 0.8)
                sub["chain_colour"] = r
                self.event("chain_colour", clip=k, method=r["method"], strength=r["strength"])
            if m.get("chain_colour") == "look":            # stop the costume colour drift being passed on
                from . import chain_colour
                masks = Path(self.st["steps"]["passes"]["outputs"]["passes"]) / "mask"
                start = (k - 1) * (fmt["frames"] - int(fmt.get("overlap", 1)))
                r = chain_colour.lock(chained, look, masks / f"mask_{start + 1:04d}.png", masks / "mask_0001.png")
                sub["chain_colour"] = r
                self.event("chain_colour", clip=k, **r)
            if m.get("look_chain") == "anchor":            # test A: the run look stays the fixed frame -1 anchor (the
                chain_key = {"frame": 0, "file": str(chained), "view": "chain"}   # first chunk as a sink); the join
            else:                                          # comes from the chained frame as a guide at frame 0
                look = chained
        if m.get("look_image"):                         # a take's own frame -1 look (for example a fresh Qwen look of
            look = Path(m["look_image"])                # this clip's start frame: the chained frame had drifted)
        first_in = dst.parent / (f"first_{k}.mp4" if k > 1 else "first.mp4")
        relay = m.get("relay") == "latent"
        if k > 1 and not relay:                         # continue from the last frames of the clip before
            video.last_frames(parts[str(k - 1)]["outputs"]["clip"], first_in, max(ov, 1))
        else:                                           # loaded but bypassed
            video.image_video(look, first_in, fmt["fps"])
        m = clip_manifest(m, k)                             # this clip's own text: the shared context + ITS actions
        prompt = self._video_prompt(m)                      # (the shot's text is for the whole shot: AGENTS.md 7.2)
        negative = m.get("negative") or NEGATIVE
        sub["prompt"], sub["negative"], sub["first_image"] = prompt, negative, str(look)
        sub["stage2"] = "off (preview: stage 1 only)" if preview else (    # which stage 2 made this clip
            wf.capabilities.get("stage2") or ("guided" if wf.graph.get("guider2", {}).get("class_type") ==
                                              "MultimodalGuider" else "distilled"))
        values = {"prompt": prompt, "negative": negative, "width": fmt["width"], "height": fmt["height"],
                  "frames": fmt["frames"], "fps": fmt["fps"], "seed": m["seed"],
                  "style_lora": m["style_lora"], "style_lora_strength": m["style_lora_strength"],
                  "sigmas": ", ".join(f"{x:.6g}" for x in sig), "no_first_frame": k == 1 or relay,
                  "first_strength": float(m.get("overlap_strength", 1.0)) if k > 1 else 1.0}
        if ing:
            values["ingredients_strength"] = m["ingredients_strength"]
        if m.get("refine_sigmas") and "refine_sigmas" in wf.bind["inputs"]:
            values["refine_sigmas"] = m["refine_sigmas"]
        if full and m.get("guidance") and "video_cfg" in wf.bind["inputs"]:
            values["video_cfg"] = float(m["guidance"])
        files = {wf.targets("layout_video")[0]: fmt["source"] if k == 1 else self._source_slice(k, fmt),
                 wf.targets("look_image")[0]: look, wf.targets("first_frame")[0]: first_in}
        if ing:
            files[wf.targets("reference")[0]] = ref
        rp = {s_: f"{relay_prefix}_{s_}" for s_ in (("s1",) if preview else ("s1", "s2"))}
        carry = relay and m.get("audio_relay") == "carry" and wf.capabilities.get("audio") is not False
        if carry:
            rp["a"] = f"{relay_prefix}_a"                 # the clip's stage-1 audio latent (the audio carry)
        n_lat = max(1, (ov - 1) // 8)
        if relay:
            add_relay_save(wf.graph, rp["s1"], rp.get("s2"))  # every clip: the next clip and retakes use them
            if k > 1:
                prev = parts[str(k - 1)].get("relay") or {}
                if not prev:
                    raise RunError(f"clip {k - 1} has no saved latents for the latent relay; make it again")
                self._relay_inputs(prev)
                add_relay_guides(wf.graph, prev["s1"], None if preview else prev.get("s2"),
                                 float(m.get("relay_strength") or 1.0), n_lat)
        if carry:
            tail = None
            if k > 1:
                prev = parts[str(k - 1)].get("relay") or {}
                if prev.get("a"):
                    tail = self._audio_tail(prev["a"], fmt)
                    sub["audio_carry"] = {"from_clip": k - 1, "tail": tail, "source": prev["a"]}
                else:                                    # a clip made before the carry (a resumed run): no tail
                    self.event("audio_carry_skipped", clip=k, reason=f"clip {k - 1} has no saved audio latent")
            add_audio_carry(wf.graph, rp["a"], tail)
        if end_anchor:                                   # the retake ends where the clip after it starts
            self._relay_inputs(end_anchor)
            total = (fmt["frames"] - 1) // 8 + 1
            add_latent_guides(wf.graph, "end", end_anchor["s1"], None if preview else end_anchor.get("s2"), 1, n_lat,
                              total - n_lat,
                              float(m.get("anchor_strength") or m.get("relay_strength") or 1.0))
        if k > 1 and float(m.get("adain") or 0) > 0:
            ref_lat = None
            if m.get("adain_ref") == "clip1":               # test B: clip 1's own latent (Lightricks' reference)
                first = parts["1"].get("relay") or {}
                if not first.get("s1"):
                    raise RunError("clip 1 has no saved latent for adain_ref clip1; make it again")
                self._relay_inputs(first)
                ref_lat = first["s1"]
            add_adain(wf.graph, float(m["adain"]), fmt["width"], fmt["height"], ref_latent=ref_lat)
        kfs = []
        if k in keys and m.get("keyframe", "on") != "off":   # the clip's keyframe looks: guides at their frames
            kfs.extend(keys[k])
        if chain_key:                                       # look_chain anchor: always, also with keyframes off
            kfs.append(chain_key)
        if m.get("keyframe_image"):                         # a take's own extra keyframe look (for example a sheet
            kf = m.get("keyframe_frame")                    # panel pasted into the frame); 0 is a real frame
            kfs.append({"frame": int(kf) if kf is not None else fmt["frames"] // 2,
                        "file": str(m["keyframe_image"]), "view": "given"})
        for f, fr in m.get("keyframe_images") or []:       # a take's extra keyframe looks at their own frames
            kfs.append({"frame": int(fr), "file": str(f), "view": "given"})
        if kfs:
            kfs.sort(key=lambda x: x["frame"])
            files.update(add_keyframe_guides(wf.graph, [(x["frame"], x["file"]) for x in kfs]))
            sub["keyframe"] = kfs[0] if len(kfs) == 1 else kfs
        if float(m.get("pose_strength") or 0) > 0:          # the skeleton: where the head faces in back views
            add_pose_guide(wf.graph, float(m["pose_strength"]), stages=(1, 2) if m.get("pose_stage2") else (1,))
            files[("pose_load", "file")] = self._pose_slice(k, fmt, m.get("pose_body") or "standard", dst.parent)
            sub["pose"] = {"strength": float(m["pose_strength"]), "body": m.get("pose_body") or "standard",
                           "stage2": bool(m.get("pose_stage2"))}
        if preview:
            to_stage1_only(wf.graph)
        outcome = self._comfy(sub, job_name, wf, values, files, {wf.output_node: dst})
        if outcome:
            return outcome
        problems = video.check_clip(dst, fmt["frames"], fmt["width"] * scale, fmt["height"] * scale, fmt["fps"],
                                    audio=wf.capabilities.get("audio"))
        if problems:
            raise RunError(f"clip {k} does not have the requested format: " + "; ".join(problems))
        kept = None
        if relay:
            kept = {s_: relay_latent(p_, keep_dir=self.work / "latents") for s_, p_ in rp.items()}
            kept = kept if all(kept.values()) else None
            if kept is not None and preview:
                kept["s2"] = None
            if kept is None and k < self.st["steps"]["controls"]["outputs"].get("clips", 1):
                raise RunError(f"clip {k} was made, but its latents were not saved: the next clip cannot continue "
                               "from it (make the clip again)")
        return {"scale": scale, "relay": kept}

    def _pending_takes(self) -> bool:
        return any(t.get("status") not in ("done", "failed", "canceled")
                   for lst in (self.st.get("takes") or {}).values() for t in lst)

    def _run_takes(self) -> str | None:
        """Make the pending retakes (run retake), each with its own job record (inside the state: crash recovery),
        own job name and own latent names; the run's clips and final video are not touched (run use-take)."""
        for key in sorted((self.st.get("takes") or {}), key=int):
            for t in self.st["takes"][key]:
                if t.get("status") in ("done", "failed", "canceled"):
                    continue
                if self.cancel_requested():
                    return "canceled"
                outcome = self._make_take(int(key), t)
                if outcome:
                    return outcome
        return None

    def _make_take(self, k: int, t: dict) -> str | None:
        fmt = self.st["steps"]["controls"]["outputs"]
        parts = self.st["steps"]["clip"].setdefault("parts", {})
        n = fmt.get("clips", 1)
        ov = dict(t.get("overrides") or {})
        if isinstance(ov.get("video"), str):                # a take asked before the fill (Codex review 2026-10-25)
            ov["video"] = fill_character(ov["video"], self.m.get("character_words"))
        if "actions" not in t:
            t.update(take_actions(self.m, k, ov))
        m = json.loads(json.dumps(self.m))
        m["prompt"] = dict(m.get("prompt") or {})
        context = m.get("clip_join") == "context"
        if "video" in ov and context:                       # one prompt per clip: the retake's text is this clip's
            m["clip_videos"] = {**(m.get("clip_videos") or {}), str(k): ov["video"]}   # ACTIONS (the shared text stays)
        elif "video" in ov:                                 # an old run: the retake's own text wins over the clip's text
            m["clip_videos"] = {kk: vv for kk, vv in (m.get("clip_videos") or {}).items() if kk != str(k)}
        for key, v in ov.items():
            if key == "video" and context:
                continue
            if key in ("video", "style", "character"):
                m["prompt"][key] = v
            else:
                m[key] = v
        tdir = self.d / "takes" / f"clip{k}" / f"take{t['n']}"
        tdir.mkdir(parents=True, exist_ok=True)
        end = None
        if k < n and ov.get("end_anchor", "on") != "off" and m.get("relay") == "latent":
            end = parts.get(str(k + 1), {}).get("relay")         # the take ends where clip k+1 begins
        t.update(status="running", started=time.time())
        self.save()
        self.event("take_started", clip=k, take=t["n"], changes=ov)
        try:
            res = self._layout_clip_job(t, k, m, parts, tdir / "clip.mp4", f"take{k}_{t['n']}",
                                        f"cinestudio/relay_{self.id}_{k}_t{t['n']}", end_anchor=end)
        except RunError as exc:
            t.update(status="failed", error=str(exc), ended=time.time())
            self.save()
            self.event("take_failed", clip=k, take=t["n"], error=str(exc))
            return None
        if isinstance(res, str):
            return res
        t.update(status="done", ended=time.time(), end_anchor=bool(end),
                 outputs={"clip": str(tdir / "clip.mp4"), "relay": res.get("relay")})
        self.save()
        self.event("take_done", clip=k, take=t["n"], clip_file=str(tdir / "clip.mp4"))
        return None

    def _audio_join(self) -> list[str]:
        """One mode PER JOIN: "carry" where clip k started with the audio tail of the clip k-1 IN USE (a take put in
        for clip k-1 changes its sound), else "fade" (independent sounds: the equal-power cross-fade). Per join, so one
        replaced clip does not give the +3 dB fade back to the other carried joins (Codex review 2026-10-24)."""
        parts = (self.st["steps"].get("clip") or {}).get("parts") or {}
        n = len(parts)
        modes = []
        for k in range(2, n + 1):
            carried = (parts.get(str(k)) or {}).get("audio_carry") or {}
            prev_a = ((parts.get(str(k - 1)) or {}).get("relay") or {}).get("a")
            ok = self.m.get("audio_relay") == "carry" and carried.get("source") and carried["source"] == prev_a
            modes.append("carry" if ok else "fade")
        return modes

    def _audio_tail(self, audio_file: str, fmt: dict) -> str:
        """Clip k-1's saved stage-1 audio latent cut to the shared frames (the tail that clip k starts with), written
        next to the run's latents and into ComfyUI's input folder; returns the LoadLatent file name."""
        from . import comfy, latent_file
        src = self.work / "latents" / audio_file
        if not src.is_file():
            src = comfy.io_dir("input") / audio_file
        if not src.is_file():
            raise RunError(f"the saved audio latent {audio_file} is missing: make the clip before again")
        name = Path(audio_file).stem + "_tail.latent"
        dst = self.work / "latents" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        start = latent_file.audio_tail_start(int(fmt["frames"]), int(fmt.get("overlap", 1)), float(fmt["fps"]))
        latent_file.cut_tail(src, dst, start)
        inp = comfy.io_dir("input")
        inp.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dst, inp / name)
        return name

    def _relay_inputs(self, files: dict) -> None:
        """Put saved relay latents back into ComfyUI's input folder (from the run's own copies) when they are gone."""
        from . import comfy
        inp = comfy.io_dir("input")
        for name in files.values():
            if name and not (inp / name).is_file() and (self.work / "latents" / name).is_file():
                shutil.copyfile(self.work / "latents" / name, inp / name)

    def _clips(self, step: dict) -> str | None:
        """All clips of the shot. Clip 1 as before; clip k > 1 continues from
        the last frames of clip k-1 with its own slice of the control and of the source. Each finished clip is kept
        in step["parts"], so a stopped run continues with the next clip."""
        fmt = self.st["steps"]["controls"]["outputs"]
        sheet = (self.st["steps"].get("restyle", {}).get("outputs") or self.st["steps"]["sheet"]["outputs"])["sheet"]
        v2v = bool(self.m.get("source_denoise"))
        wf = self.workflow(QUALITY[self.m["quality"]]["v2v" if v2v else "variant"])
        scale = wf.capabilities.get("output_scale", 1)
        ref = self.work / "reference.png"
        video.fit_image(sheet, ref, fmt["width"], fmt["height"], mode="pad", color="black")
        parts = step.setdefault("parts", {})
        n = fmt.get("clips", 1)
        ov = int(fmt.get("overlap", 1))
        for k in range(1, n + 1):
            if parts.get(str(k), {}).get("outputs"):
                continue
            sub = step if k == 1 else parts.setdefault(str(k), {})     # the job record of the clip
            first = ref                                     # loaded even when the first-frame guide is off
            use_first = bool(self.m.get("first_frame")) if k == 1 else True
            # first frames at the clip's OUTPUT size: the LTX node resizes them to each stage itself, so the
            # refine stage gets the full-size picture (a 2x blow-up of a 768 px copy made every join soft)
            fw, fh = fmt["width"] * scale, fmt["height"] * scale
            if k == 1 and self.m.get("first_frame"):
                first = self.work / "first.png"
                video.fit_image(self.d / self.m["first_frame"]["file"], first, fw, fh, mode="pad", color="black")
            elif k > 1:                                     # continue from the last frame(s) of the clip before
                last = self.work / f"last_{k - 1}.png"
                video.last_frame(parts[str(k - 1)]["outputs"]["clip"], last)
                first = self.work / f"first_{k}.png"
                video.fit_image(last, first, fw, fh, mode="pad", color="black")
            first_in = first                                # what the graph loads: a picture (old graphs) or a video
            if wf.graph["first"]["class_type"] != "LoadImage":
                first_in = self.work / (f"first_{k}.mp4" if k > 1 else "first.mp4")
                if k > 1 and ov > 1:                        # the previous clip's last `ov` frames (motion, not only place)
                    video.last_frames(parts[str(k - 1)]["outputs"]["clip"], first_in, ov)
                else:
                    video.image_video(first, first_in, fmt["fps"])
            if k == 1:
                control = fmt["control"]
            else:
                control = self.work / f"control_{k}.mp4"
                video.slice_control(fmt["control_full"], control, k, fmt["frames"], fmt["width"], fmt["height"],
                                    fmt["fps"], overlap=ov)
            prompt = self._video_prompt(clip_manifest(self.m, k))   # the shared text + this clip's actions
            if self.m.get("describe_facing"):
                text = self._facing_text(k, fmt)
                if text:
                    prompt += " " + text
                    self.event("facing_prompt", clip=k, text=text)
            negative = self.m.get("negative") or (TEMPLATE_NEGATIVE if self.m.get("template") else NEGATIVE)
            sub["prompt"], sub["negative"] = prompt, negative               # what this clip got, for the app
            sub["first_image"] = str(first) if use_first else None
            values = {"prompt": prompt, "negative": negative,
                      "width": fmt["width"], "height": fmt["height"], "frames": fmt["frames"], "fps": fmt["fps"],
                      "seed": self.m["seed"], "no_first_frame": not use_first,
                      "ingredients_strength": self.m["ingredients_strength"],
                      "control_strength": self.m["control_strength"],
                      "style_lora": self.m["style_lora"], "style_lora_strength": self.m["style_lora_strength"]}
            steps = int(self.m.get("steps") or DEV_STEPS)
            if self.m.get("steps") and "steps" in wf.bind["inputs"]:
                values["steps"] = steps
            if self.m.get("guidance") and "video_cfg" in wf.bind["inputs"]:
                values["video_cfg"] = float(self.m["guidance"])
            if "first_strength" in wf.bind["inputs"]:
                values["first_strength"] = float(self.m.get("overlap_strength", 1.0)) if k > 1 and ov > 1 else 1.0
            files = {wf.targets("reference")[0]: ref, wf.targets("control_video")[0]: control,
                     wf.targets("first_frame")[0]: first_in}
            if v2v:
                sd = float(self.m["source_denoise"])
                values["source_sigmas"] = (dev_source_sigmas(sd, fmt["width"], fmt["height"], fmt["frames"], steps)
                                           if wf.capabilities.get("recipe") == "full"
                                           else grid_source_sigmas(sd) if self.m.get("source_schedule") == "grid"
                                           else source_sigmas(sd))
                files[wf.targets("source_video")[0]] = fmt["source"] if k == 1 else self._source_slice(k, fmt)
            if self.m.get("refine_sigmas") and "refine_sigmas" in wf.bind["inputs"]:
                values["refine_sigmas"] = self.m["refine_sigmas"]
            dst = self.work / ("clip.mp4" if k == 1 else f"clip_{k}.mp4")
            if n > 1:
                self.event("clip_part", clip=k, of=n)
            name = "clip" + ("" if k == 1 else str(k))
            outcome = self._comfy(sub, name, wf, values, files, {wf.output_node: dst})
            if outcome:
                return outcome
            problems = video.check_clip(dst, fmt["frames"], fmt["width"] * scale, fmt["height"] * scale, fmt["fps"],
                                        audio=wf.capabilities.get("audio"))
            if problems:
                raise RunError(f"clip {k} does not have the requested format: " + "; ".join(problems))
            parts.setdefault(str(k), {})["outputs"] = {"clip": str(dst)}
            self.save()
        clips = [parts[str(k)]["outputs"]["clip"] for k in range(1, n + 1)]
        step["outputs"] = {"clip": clips[0], "clips": clips, "width": fmt["width"] * scale,
                           "height": fmt["height"] * scale, "frames": fmt["frames"], "fps": fmt["fps"],
                           "shot_frames": fmt.get("shot_frames", fmt["frames"]), "overlap": ov,
                           "camera_azimuth_deg": fmt.get("camera_azimuth_deg")}
        return None

    def _grade_local(self, clips: list, out: dict, step: dict) -> list:
        """chain_grade "local" (user, 2026-10-27: "the video still got brighter over time"): LTX re-lights a still
        scene region by region from clip to clip (a08c, lightness frame 30 -> 255: the upper wall 24.6 -> 13.0, a
        banner 13.8 -> 19.0, a door window 9.0 -> 12.6), which one gain or a contrast match cannot undo. Clip k
        (k >= 2) is multiplied by a smooth GAIN MAP (video.gain_map: the ratio of clip k-1 as graded to clip k over
        the background of their shared frames); the maps add up from clip to clip. A moving camera falls back to
        "match". Clip 1 is never changed; the originals stay, the graded copies go to work/graded/."""
        bg_out = (self.st["steps"].get("background") or {}).get("outputs") or {}
        if (bg_out.get("camera") or {}).get("moves"):
            step["grade_mode"] = "match (the camera moves)"
            return self._grade_clips(clips, out, step)
        ov, frames, fps = int(out.get("overlap", 1)), int(out["frames"]), float(out["fps"])
        w, h = int(out["width"]), int(out["height"])
        if ov < 5:
            step["grade"] = {"skipped": f"overlap {ov}: too few shared frames to measure"}
            return clips
        try:
            masks = passes.load(self.pass_dir).get("mask") or []
        except (OSError, ValueError, KeyError):
            masks = []
        gdir = self.work / "graded"
        gdir.mkdir(parents=True, exist_ok=True)
        sample = set(range(2, ov - 1, 4))
        total = np.ones((h, w, 3), np.float32)
        graded, rec = [clips[0]], [{"clip": 1, "map": None}]
        for k in range(2, len(clips) + 1):
            s0 = video.clip_start(k, frames, ov)
            prev = [f for i, f in enumerate(video.read_rgb_frames(clips[k - 2], w, h)) if i - (frames - ov) in sample]
            this = [f for i, f in enumerate(video.read_rgb_frames(clips[k - 1], w, h)) if i in sample]
            pairs = []
            for j, i in enumerate(sorted(sample)):
                mk = masks[s0 + i] if s0 + i < len(masks) else None
                bg, _ = video._region(mk, (h, w))
                ref = np.clip(prev[j].astype(np.float32) * total, 0, 255).astype(np.uint8)
                pairs.append((ref, this[j], bg))
            total = (total * 0 + video.gain_map(pairs)).astype(np.float32)
            dst = gdir / f"clip_{k}.mp4"
            video.apply_gain_map(clips[k - 1], dst, total, w, h, fps)
            graded.append(str(dst))
            rec.append({"clip": k, "map": {"min": round(float(total.min()), 3), "max": round(float(total.max()), 3),
                                           "mean": round(float(total.mean()), 3)}})
            np.save(gdir / f"map_{k}.npy", total)
        step["grade"] = rec
        step["grade_mode"] = "local"
        self.event("chain_grade", mode="local", maps=[r["map"] for r in rec])
        return graded

    def _grade_clips(self, clips: list, out: dict, step: dict) -> list:
        """chain_grade "match": the luma of clip k (k >= 2) is multiplied by one gain so that the mean grey of the
        background (outside the character's grown mask) of its shared frames matches the same frames of clip k-1 as
        graded (the gains add up, each limited to +-15%). The clips stay in their own YUV form (an RGB copy read ~4%
        brighter in the stitch); clip 1 is never changed; the originals stay, the graded copies go to work/graded/.
        A gain cannot fix a changed spread of the light (in r20261027-005603-a08c the later clips have a lighter floor
        and a darker wall). -> the clip files to stitch."""
        ov, frames = int(out.get("overlap", 1)), int(out["frames"])
        if ov < 5:
            step["grade"] = {"skipped": f"overlap {ov}: too few shared frames to measure"}
            return clips
        try:
            masks = passes.load(self.pass_dir).get("mask") or []
        except (OSError, ValueError, KeyError):
            masks = []
        gdir = self.work / "graded"
        gdir.mkdir(parents=True, exist_ok=True)
        probe = gdir / "probe.png"
        sample = list(range(2, ov - 1, 4))
        maps, rec, graded = [(1.0, 0.0)], [{"clip": 1, "a": 1.0, "b": 0.0}], [clips[0]]
        for k in range(2, len(clips) + 1):
            s0 = video.clip_start(k, frames, ov)
            prev, this, why = [], [], None
            for i in sample:
                mk = masks[s0 + i] if s0 + i < len(masks) else None
                video.frame_png(clips[k - 2], probe, frames - ov + i)
                lp, w1 = video.background_levels(probe, mk)
                video.frame_png(clips[k - 1], probe, i)
                lt, w2 = video.background_levels(probe, mk)
                prev.append(lp)
                this.append(lt)
                why = why or w1 or w2
            p_lo, p_hi, p_m, p_s = (float(np.median([x[j] for x in prev])) for j in (0, 1, 2, 3))
            t_lo, t_hi, t_m, t_s = (float(np.median([x[j] for x in this])) for j in (0, 1, 2, 3))
            pa, pb = maps[-1]                               # clip k-1 as graded
            g_lo, g_hi = pa * p_lo + pb, pa * p_hi + pb
            # the MEAN and the SPREAD of the background matched: grey' = a * grey + b (user, 2026-10-27: "the video
            # still got brighter over time": with one gain the mean held, but LTX raised the CONTRAST clip by clip:
            # brighter door windows, banner letters, court lines and floor light pools on a darker wall). The plain
            # mean and standard deviation (8-bit percentiles were too coarse; the halves split at the median were
            # biased and followed the lamps)
            g_m, g_s = pa * p_m + pb, pa * p_s
            a = float(min(1.25, max(0.8, g_s / max(0.5, t_s))))
            b = float(min(15.0, max(-15.0, g_m - a * t_m)))
            if abs(a - 1.0) < 0.003 and abs(b) < 0.3:
                a, b = 1.0, 0.0
                graded.append(clips[k - 1])
            else:
                dst = gdir / f"clip_{k}.mp4"
                video.levels_yuv(clips[k - 1], dst, a, b)
                graded.append(str(dst))
            maps.append((a, b))
            rec.append({"clip": k, "a": round(a, 4), "b": round(b, 2), "prev_mean": round(g_m, 2),
                        "mean": round(t_m, 2), "prev_spread": round(g_s, 2), "spread": round(t_s, 2), "prev_levels": [round(g_lo, 1), round(g_hi, 1)],
                        "levels": [round(t_lo, 1), round(t_hi, 1)], "region": why or "background"})
        probe.unlink(missing_ok=True)
        step["grade"] = rec
        self.event("chain_grade", maps=[[r["a"], r["b"]] for r in rec])
        return graded

    def step_final(self, step: dict) -> None:
        out = self.st["steps"]["clip"]["outputs"]
        clips = out.get("clips") or [out["clip"]]
        tmp = self.d / "final.tmp.mp4"
        if len(clips) > 1 and self.m.get("chain_grade", "off") in ("match", "local"):
            try:
                clips = (self._grade_local(clips, out, step) if self.m.get("chain_grade") == "local"
                         else self._grade_clips(clips, out, step))
            except Exception as exc:  # noqa: BLE001 - never fail a run over its grade: the clips as made
                step["grade"] = {"skipped": f"the grade could not run: {exc}"[:300]}
                self.event("chain_grade", skipped=step["grade"]["skipped"])
        if len(clips) > 1:          # join on the shared frame; cut the held frames after the shot's end
            video.stitch(clips, tmp, out["fps"], total_frames=out["shot_frames"], overlap=int(out.get("overlap", 1)),
                         audio_join=(lambda j: j if len(j) == len(clips) - 1 else "fade")(self._audio_join()))
        else:
            shutil.copyfile(clips[0], tmp)
        lp = self.m.get("loop")
        if lp and video.probe(tmp).frames >= lp["frames"] + lp["tail"]:   # a loop: the tail fades into the start
            closed = self.d / "final.loop.tmp.mp4"
            video.loop_close(tmp, closed, lp["frames"], lp["tail"], float(out["fps"]))
            os.replace(closed, tmp)
            self.event("loop_closed", frames=lp["frames"], tail=lp["tail"])
        sound = "ltx"
        if self.m.get("song"):                          # a song from the library wins over the shot's own sound
            s = self.m["song"]
            cut = self.work / "sound" / "song_cut.wav"
            cut.parent.mkdir(parents=True, exist_ok=True)
            video.cut_audio(self.d / s["file"], float(s.get("start") or 0.0), cut)
            with_song = self.d / "final.song.tmp.mp4"
            video.replace_audio(tmp, cut, with_song)
            os.replace(with_song, tmp)
            sound = "song"
            self.event("sound", source="song", song=s.get("name"), start=s.get("start"))
        elif self.m.get("sound", "ltx") == "auto":     # older runs (no field) keep LTX's sound, as they did
            wav = self._blend_sound()
            if wav:
                with_song = self.d / "final.song.tmp.mp4"
                video.replace_audio(tmp, wav, with_song)
                os.replace(with_song, tmp)
                sound = "template" if self.m.get("template") else "blender"
                self.event("sound", source=sound, file=str(wav))
        os.replace(tmp, self.d / "final.mp4")
        video.contact_sheet(self.d / "final.mp4", self.d / "contact_sheet.jpg",
                            extra=self.st["steps"].get("restyle", {}).get("outputs", {}).get("sheet")
                            or self.st["steps"]["sheet"]["outputs"]["sheet"])
        step["outputs"] = {"final": str(self.d / "final.mp4"), "contact_sheet": str(self.d / "contact_sheet.jpg"),
                           "sound": sound}

    def _blend_sound(self) -> Path | None:
        """The frozen shot's own sound for its frame range (its sound strips, mixed by ffmpeg), or None."""
        out = self.work / "sound"
        if self.m.get("template"):                      # a video template: its own sound, when it has one
            src = self.d / self.m["template"]["file"]
            if not video.has_audio(src):
                return None
            wav = out / "sound.wav"
            if not wav.is_file():
                out.mkdir(parents=True, exist_ok=True)
                video.cut_audio(src, 0.0, wav)
            return wav
        if (out / "sound.json").is_file():
            info = _read_json(out / "sound.json")
        else:
            self.event("blender", job="export_sound", note="Blender reads the shot's own sound")
            blender.run_job("export_sound.py", self.d / self.m["blend"]["file"], ["--out", out],
                            self.d / "logs" / "export_sound.log")
            info = _read_json(out / "sound.json")
        if not info.get("sound"):
            return None
        wav = out / "sound.wav"
        if not wav.is_file():
            tmp = out / "sound.tmp.wav"
            if video.mix_strips(info, tmp) is None:
                return None
            os.replace(tmp, wav)
        return wav

    def _recover_from_comfyui(self, name: str, job: dict, outputs: dict[str, Path]) -> str | None:
        """A job the proxy does not know: None = its outputs were downloaded from ComfyUI; "new" = it ended
        with an error (a new attempt is safe); "unresolved" = nobody knows it (the user decides);
        "canceled" = a cancel came while it still ran."""
        url = server.comfyui_url()
        jid = job["id"]
        while True:
            st = server.prompt_status(url, jid)
            if st == "done_ok":
                for node, dst in outputs.items():
                    server.download_output(url, jid, node, dst)
                self.event("recovered_from_comfyui", step=name, job=jid)
                return None
            if st == "done_error":
                return "new"
            if st == "unknown":
                self.st["error"] = (f"job {jid} is unknown to the proxy and to ComfyUI: its outcome cannot be found. "
                                    "Start the run again with --new-attempt only when no job of it runs")
                self.event("unresolved", step=name, message=self.st["error"])
                return "unresolved"
            if self.cancel_requested():
                return "canceled"
            self.event("job_status", step=name, status="waiting_in_comfyui", job=jid)
            time.sleep(5)

    # prompts
    def _sheet_prompt(self) -> str:
        p = self.m["prompt"]
        look = p["style"] or "the same drawing style"
        return (f"A character reference sheet in {look}, on a plain black background. {p['character']} "
                "A still image: nothing moves, the camera does not move.")

    def _video_prompt(self, m: dict | None = None) -> str:
        return video_prompt(m or self.m)

    def _facing_text(self, k: int, fmt: dict) -> str:
        """Where each character stands and which way it faces at the start, middle and end of clip k (facing.py)."""
        from . import facing
        first = video.clip_start(k, fmt["frames"], fmt.get("overlap", 1))
        last = min(first + fmt["frames"] - 1, fmt.get("shot_frames", fmt["frames"]) - 1)
        people = self.pass_dir / "people" / "people.json"
        info = _read_json(people) if people.is_file() else None
        if info and info["characters"]:
            n = len(info["characters"])
            fr = range(info["frame_start"], info["frame_end"] + 1)
            frames = [info["keypoints"][c]["frames"] for c in range(n)]
            xs = [[(b[0] + b[2]) / 2 if (b := info["frames"][str(f)][c]["box"]) else None for f in fr] for c in range(n)]
            yaws = [info["keypoints"][c].get("head_yaw") for c in range(n)]
            bodies = [info["keypoints"][c].get("body_yaw") for c in range(n)]
            depths = [[info["frames"][str(f)][c].get("depth") for f in fr] for c in range(n)]
            return facing.clip_text(frames, xs, info["width"], first, last, yaws, bodies, depths)
        kp_file = self.pass_dir / "keypoints.json"
        if not kp_file.is_file():
            return ""
        kp = _read_json(kp_file)
        pinfo = passes.load(self.pass_dir)
        xs = [[p[1][0] if p[1][2] else None for p in kp["frames"]]]
        return facing.clip_text([kp["frames"]], xs, pinfo["width"], first, last, [kp.get("head_yaw")], [kp.get("body_yaw")])

    # ComfyUI jobs
    def _find_job(self, job: dict) -> str | None:
        """A job whose submit was not confirmed: the proxy's list, then ComfyUI's own queue/history."""
        prefix = job.get("prefix")
        if not prefix:
            return None
        found = None
        try:
            found = self.backend.find(prefix)
        except Exception:  # noqa: BLE001 - the proxy may be down; ComfyUI may still know it
            pass
        return found or server.find_prompt(prefix)

    def _comfy(self, step: dict, name: str, wf: ProfileWorkflow, values: dict, files: dict,
               outputs: dict[str, Path], extra_prefixes: dict[str, str] | None = None,
               graph_edit=None) -> str | None:
        """One ComfyUI job with recovery. Holds the GPU lock from the family check to the download."""
        from .backend import UnresolvedSubmission, is_key_reuse
        from . import comfy
        if self._backend_factory is _default_backend and not server.alive():   # ComfyUI not open: start it
            self.event("starting_comfyui", step=name)
            comfy.ensure_running()
        lock = server.gpu_lock(note={"run": self.id, "step": name})
        try:
            lock.acquire()
        except LockBusy:
            self.event("waiting_for_gpu", step=name, holder=lock.holder())
            lock.timeout = GPU_WAIT
            lock.acquire()
        try:
            job = step.get("job") or {}
            if self.new_attempt and job and not job.get("id"):
                self.event("new_attempt", step=name, abandoned=job)
                job = {}
            if job.get("id"):
                try:
                    state = self.backend.state(job["id"])
                    if state.done and not state.ok:
                        job = {}                                    # the old attempt ended badly: a new attempt
                except Exception as exc:  # noqa: BLE001
                    if _is_connection_error(exc):
                        raise
                    # unknown to the proxy (for example after a proxy restart): ask ComfyUI itself before
                    # anything else. Never a new attempt while the first one may still run or has a result.
                    outcome = self._recover_from_comfyui(name, job, outputs)
                    if outcome != "new":
                        return outcome
                    job = {}
            if job and not job.get("id"):
                found = self._find_job(job)                         # submitted before a stop, id not saved?
                if found:
                    job["id"] = found
                    server.note_owned(found)
            if not job:
                attempt = step.get("attempts", 0) + 1
                step["attempts"] = attempt
                job = {"key": f"{self.id}-{name}-a{attempt}", "prefix": f"cinestudio/{self.id}_{name}_a{attempt}"}
            step["job"] = job
            self.save()                                             # the intent is on disk before the submit
            if self.cancel_requested():
                return "canceled"
            if not job.get("id"):
                graph_values = dict(values, output_prefix=job["prefix"])
                for key, suffix in (extra_prefixes or {}).items():
                    graph_values[key] = job["prefix"] + suffix
                graph = wf.build(graph_values, files=files)
                if graph_edit is not None:                        # a step's own change (look_pin: the pinned look)
                    graph_edit(graph)
                from .workflow import save_graph                  # the exact graph sent (the app's Workflows view)
                save_graph(self.work / "graphs", name, graph, files=files,
                           meta={"run": self.id, "job": name, "profile": wf.profile, "variant": wf.variant,
                                 "prefix": job["prefix"], "attempt": step.get("attempts")})
                switch = server.ensure_family(wf.family_key())
                if switch.get("restarted"):
                    self.event("comfyui_restarted", **switch)
                try:
                    job["id"] = self.backend.submit(graph, files, key=job["key"], prefix=job["prefix"])
                except UnresolvedSubmission as exc:
                    self.event("unresolved", step=name, message=str(exc))
                    self.st["error"] = (f"the submit of {job['prefix']} has an unknown outcome; start the run again "
                                        "later to look for it, or use --new-attempt if no job is running")
                    return "unresolved"
                except Exception as exc:
                    if is_key_reuse(exc):                           # the proxy holds the key: never 'failed'
                        self.event("unresolved", step=name, message=f"key {job['key']} was used, no job found")
                        self.st["error"] = (f"the proxy already used key {job['key']} but neither it nor ComfyUI has "
                                            "the job; if nothing runs, start again with --new-attempt")
                        return "unresolved"
                    raise
                server.note_owned(job["id"])
                self.save()
                self.event("submitted", step=name, job=job["id"])
                if self.cancel_requested():                         # requested while the submit ran
                    self.backend.cancel(job["id"])
            return self._watch(step, name, job, outputs)
        finally:
            lock.release()

    def _watch(self, step: dict, name: str, job: dict, outputs: dict[str, Path]) -> str | None:
        stop = threading.Event()

        def cancel_watch() -> None:                                 # a cancel file during a silent node
            while not stop.wait(2.0):
                if self.cancel_requested():
                    try:
                        self.backend.cancel(job["id"])
                    except Exception:  # noqa: BLE001 - tried again on the next tick
                        continue
                    return
        watcher = threading.Thread(target=cancel_watch, daemon=True)
        watcher.start()
        last = [0.0]

        def progress(kind: str, data: dict) -> None:
            if kind == "progress" and data.get("steps") and time.time() - last[0] > 1:
                last[0] = time.time()
                self.event("progress", step=name, value=data.get("step"), max=data.get("steps"), node=data.get("node"))
            elif kind == "status":
                self.event("job_status", step=name, **data)
        try:
            result = self.backend.watch(job["id"], progress)
        finally:
            stop.set()
        if result.status in ("canceled", "cancelled"):
            return "canceled"
        if not result.ok:
            hint = " (out of GPU memory: close other GPU programs or use quality 'preview')" if result.oom else ""
            raise RunError(f"ComfyUI job {job['id']} {result.status}: {result.error}{hint}")
        for node, dst in outputs.items():
            self.backend.download(job["id"], node, dst)
        return None


def start(run_id: str, new_attempt: bool = False, launch: int | None = None, **kw: Any) -> dict:
    return Run(run_id, **kw).start(new_attempt=new_attempt, launch=launch)


def _no_worker(run_id: str) -> Path:
    d = run_dir(run_id)
    if is_locked(d / "run.lock"):
        raise RunError(f"run {run_id} is working; wait until it stops (or cancel it)")
    return d


@contextlib.contextmanager
def _hold_run(run_id: str, note: dict):
    """Hold the run's worker lock for a whole read-change-write of state.json (Codex review 2026-10-22: a check of
    run.lock, then _state_lock, let a worker start in between; the worker's saves do not take _state_lock). Taken
    under the life lock, in the same order as a worker start, so it also excludes a delete."""
    d = run_dir(run_id)
    lock = FileLock(d / "run.lock", timeout=0, note=note)
    with _life_lock(run_id, timeout=10):
        try:
            lock.acquire()
        except LockBusy as exc:
            raise RunError(f"run {run_id} is working; wait until it stops (or cancel it)") from exc
    try:
        yield d
    finally:
        lock.release()


def set_clip_limit(run_id: str, through_clip: int | None) -> dict:
    """`run start --through-clip K`: the run stops ("paused") after clip K, so clip K can be judged and made again
    (run retake ... / run use-take) before the next clip continues from it. None = no limit (finish the run)."""
    with _hold_run(run_id, {"run": run_id, "clip_limit": through_clip}) as d, _state_lock(d):
        st = _read_json(d / "state.json")
        if through_clip:
            st["clip_limit"] = int(through_clip)
        else:
            st.pop("clip_limit", None)
        _write_json(d / "state.json", st)
    return {"id": run_id, "clip_limit": st.get("clip_limit")}


def add_take(run_id: str, clip: int, changes: dict | None = None, note: str | None = None) -> dict:
    """Ask for a retake of clip `clip` with `changes` (TAKE_FIELDS). The run's worker makes it (start the run): the
    take starts from clip-1's saved latents and, for a middle clip, ends on clip+1's (end anchor). The run's own clip
    and final video stay until `use_take`."""
    changes = {k: v for k, v in (changes or {}).items() if v is not None}
    bad = set(changes) - set(TAKE_FIELDS)
    if bad:
        raise RunError(f"a retake cannot change {sorted(bad)}; it can change {list(TAKE_FIELDS)}")
    for key in ("anchor_strength", "relay_strength"):
        if key in changes and not 0 < float(changes[key]) <= 1:
            raise RunError(f"{key.replace('_', ' ')} must be above 0 and at most 1 (LTXVAddLatentGuide caps it at 1)")
    for key, lo, hi in (("adain", 0.0, 1.0), ("chain_sharpen", 0.0, 3.0), ("ingredients_strength", 0.0, 3.0),
                        ("pose_strength", 0.0, 1.0),
                        ("style_lora_strength", 0.0, 3.0)):
        if key in changes and not lo <= float(changes[key]) <= hi:
            raise RunError(f"{key.replace('_', ' ')} must be {lo:g}..{hi:g}")
    if changes.get("look_image"):
        li = Path(changes["look_image"]).resolve()
        if not li.is_file():
            raise RunError(f"look image {li} does not exist")
        changes["look_image"] = str(li)
    if changes.get("keyframe_image"):
        kfi = Path(changes["keyframe_image"]).resolve()
        if not kfi.is_file():
            raise RunError(f"keyframe image {kfi} does not exist")
        changes["keyframe_image"] = str(kfi)
    if changes.get("keyframe_images"):                    # [[file, clip frame], ...]
        fixed = []
        for f, fr in changes["keyframe_images"]:
            kfi = Path(f).resolve()
            if not kfi.is_file():
                raise RunError(f"keyframe image {kfi} does not exist")
            fixed.append([str(kfi), int(fr)])
        changes["keyframe_images"] = fixed
    for key, allowed in (("keyframe", ("on", "off")), ("end_anchor", ("on", "off")), ("look_chain", LOOK_CHAINS),
                         ("pose_body", ("standard", "chibi")), ("chain_colour", CHAIN_COLOURS),
                         ("adain_ref", ("look", "clip1"))):
        if key in changes and changes[key] not in allowed:
            raise RunError(f"{key.replace('_', ' ')} must be one of {list(allowed)}")
    m = _read_json(run_dir(run_id) / "manifest.json")
    if retired(m):
        raise RunError(retired(m))
    if m.get("render_mode") != "layout" or m.get("relay") != "latent":
        raise RunError("retakes need a layout run with the latent relay (the default for Blender shots)")
    with _hold_run(run_id, {"run": run_id, "add_take": clip}) as d, _state_lock(d):
        st = _read_json(d / "state.json")
        fmt = (st["steps"].get("controls") or {}).get("outputs") or {}
        n = fmt.get("clips", 1)
        if not 1 <= int(clip) <= n:
            raise RunError(f"clip must be 1..{n}")
        parts = (st["steps"].get("clip") or {}).get("parts") or {}
        if not (parts.get(str(clip)) or {}).get("outputs"):
            raise RunError(f"clip {clip} is not made yet: start the run (with --through-clip {clip})")
        if int(clip) > 1 and not (parts.get(str(int(clip) - 1)) or {}).get("relay"):
            raise RunError(f"clip {int(clip) - 1} has no saved latents: a retake of clip {clip} cannot continue from it")
        takes = st.setdefault("takes", {}).setdefault(str(int(clip)), [])
        if isinstance(changes.get("video"), str):         # "{character}" filled as in new runs and forks (Codex 10-25)
            changes["video"] = fill_character(changes["video"], m.get("character_words"))
        t = {"n": max([x["n"] for x in takes] + [0]) + 1, "status": "pending", "overrides": changes,
             "note": note, "asked": time.time(), **take_actions(m, int(clip), changes)}
        takes.append(t)
        _write_json(d / "state.json", st)
    return {"id": run_id, "clip": int(clip), "take": t["n"], "changes": changes}


_SHA_CACHE: dict = {}


def file_sha(path: Path) -> str:
    """SHA-256 of a file, cached by (path, size, change time): status() checks the side scores often."""
    p = Path(path)
    stt = p.stat()
    key = (str(p.resolve()), stt.st_size, stt.st_mtime_ns)
    if key not in _SHA_CACHE:
        if len(_SHA_CACHE) > 512:
            _SHA_CACHE.clear()
        _SHA_CACHE[key] = _sha256(p)
    return _SHA_CACHE[key]


def _score_file(d: Path, clip: int, take: int | None) -> Path:
    return d / "scores" / (f"clip{int(clip)}" + (f"_take{int(take)}" if take is not None else "") + ".json")


def save_score(d: Path, clip: int, take: int | None, score: dict) -> Path:
    """`run score`'s result as a side file (atomic; never state.json, which the worker rewrites: Codex 10-22)."""
    f = _score_file(d, clip, take)
    f.parent.mkdir(exist_ok=True)
    _write_json(f, score)
    return f


def stored_score(d: Path, clip: int, take: int | None = None, st: dict | None = None) -> dict | None:
    """The saved score of a clip (or a take) while every video it measured is unchanged (the clip, the clips before
    and after it, clip 1); None when there is none or it is out of date. Old runs: the score in the state record."""
    f = _score_file(d, clip, take)
    st = st or _read_json(d / "state.json")
    parts = (st["steps"].get("clip") or {}).get("parts") or {}
    if not f.is_file():
        rec = parts.get(str(int(clip))) if take is None else next(
            (t for t in (st.get("takes") or {}).get(str(int(clip)), []) if t["n"] == int(take)), None)
        return (rec or {}).get("score")
    try:
        sc = _read_json(f)
    except (OSError, ValueError):
        return None
    deps = sc.get("deps") or {}

    def clip_file(k):
        return Path(((parts.get(str(k)) or {}).get("outputs") or {}).get("clip") or "")
    own = Path(sc.get("video") or "")
    want = {"clip": own, "prev": clip_file(int(clip) - 1), "next": clip_file(int(clip) + 1), "clip1": clip_file(1)}
    for name, sha in deps.items():
        p = want.get(name)
        if p is None or not p.is_file() or file_sha(p) != sha:
            return {**sc, "stale": f"{name} changed since it was scored"}
    return sc


def list_takes(run_id: str) -> dict:
    d = run_dir(run_id)
    st = _read_json(d / "state.json")
    out = {}
    for key, lst in (st.get("takes") or {}).items():
        out[key] = [{**{k: t.get(k) for k in ("n", "status", "overrides", "note", "outputs", "error", "end_anchor",
                                              "used", "original")},
                     "score": stored_score(d, int(key), t["n"], st)} for t in lst]
    return {"id": run_id, "takes": out, "used": st.get("takes_used") or {}}


def use_take(run_id: str, clip: int, take: int) -> dict:
    """Put take `take` of clip `clip` into the video: its clip file goes to the clip's own file name (the original is
    kept as take 0), its latents become the clip's relay latents, the previous final video is kept as
    final.<time>.mp4, and the clips are joined again."""
    with _hold_run(run_id, {"run": run_id, "use_take": [clip, take]}):
        run = Run(run_id)
        run.load()
        key = str(int(clip))
        takes = (run.st.get("takes") or {}).get(key) or []
        t = next((x for x in takes if x["n"] == int(take)), None)
        if t is None or t.get("status") != "done":
            raise RunError(f"take {take} of clip {clip} is not done")
        parts = run.st["steps"]["clip"]["parts"]
        cur = parts[key]
        # what the clip was given (prompt, first image ...): clip 1's record is the clip step itself (_layout_clips),
        # so a take for clip 1 goes there (Codex review 2026-10-25: the details kept clip 1's original prompt)
        given = run.st["steps"]["clip"] if key == "1" else cur
        canonical = Path(cur["outputs"]["clip"])
        n = run.st["steps"]["controls"]["outputs"].get("clips", 1)
        if int(clip) < n and not (t.get("outputs") or {}).get("relay"):
            raise RunError(f"take {take} has no saved latents: clip {int(clip) + 1} could not continue from it")
        if not any(x.get("original") for x in takes):        # keep the original as take 0
            t0 = run.d / "takes" / f"clip{key}" / "take0"
            t0.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(canonical, t0 / "clip.mp4")
            takes.insert(0, {"n": 0, "status": "done", "original": True, "overrides": {},
                             "outputs": {"clip": str(t0 / "clip.mp4"), "relay": cur.get("relay")},
                             **({"score": cur["score"]} if cur.get("score") else {}),
                             **take_actions(run.m, int(clip), {}),
                             **{f: v[f] for v in (given, cur) for f in ("audio_carry", "prompt", "negative",
                                                                        "first_image", "keyframe", "pose", "scale")
                                if f in v}})   # (Codex: take 0 lost them)
            own = _score_file(run.d, int(clip), None)       # the original's side score goes with it (same bytes)
            if own.is_file():
                save_score(run.d, int(clip), 0, {**_read_json(own), "video": str(t0 / "clip.mp4")})
        shutil.copyfile(t["outputs"]["clip"], canonical)
        cur["relay"] = (t.get("outputs") or {}).get("relay")
        cur.pop("score", None)                                  # the clip's score belongs to the take it came from
        cur.pop("audio_carry", None)                            # the take's own audio tail record (or none)
        for field in ("prompt", "negative", "first_image", "keyframe", "pose", "scale", "audio_carry"):
            for rec in {id(given): given, id(cur): cur}.values():
                if field in t:                              # (own name: `key` is the clip number; Codex 10-22)
                    rec[field] = t[field]
                elif field != "scale" or rec is cur:        # the clip step's own "scale" is not the take's
                    rec.pop(field, None)
        if t.get("score"):
            cur["score"] = t["score"]
        tf = _score_file(run.d, int(clip), int(take))
        own = _score_file(run.d, int(clip), None)
        if tf.is_file():                                    # the take's side score is now the clip's (same bytes)
            save_score(run.d, int(clip), None, {**_read_json(tf), "video": str(canonical), "take": None})
        elif own.is_file():
            own.unlink()
        run.st["takes_used"] = {**(run.st.get("takes_used") or {}), key: int(take)}
        for x in takes:
            x["used"] = x["n"] == int(take)
        final = run.d / "final.mp4"
        kept = None
        if final.is_file():
            kept = run.d / f"final.{time.strftime('%Y%m%d-%H%M%S')}.mp4"
            shutil.copyfile(final, kept)
        run.save()
        run.event("take_used", clip=int(clip), take=int(take), previous_final=str(kept) if kept else None)
        if run.st["steps"].get("final", {}).get("status") == "done" or final.is_file():
            fin = run.st["steps"]["final"]
            fin["status"] = "pending"                           # until the new join is written (a failed stitch
            run.save()                                          # does not leave "done" over the old video)
            run.step_final(fin)
            fin.update(status="done", ended=time.time())
            run.save()
        return {"id": run_id, "clip": int(clip), "take": int(take), "final": str(final),
                "previous_final": str(kept) if kept else None}


def _life_lock(run_id: str, timeout: float = 30) -> FileLock:
    """Outside the run folder: a delete holds it until the folder is gone, and a start takes it before it
    takes the run lock, so no worker can start in the middle of a delete. The id is checked first (as run_dir)."""
    if not run_id or not isinstance(run_id, str) or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        raise RunError(f"bad run id {run_id!r}")
    return FileLock(runs_dir() / ".locks" / f"{run_id}.lock", timeout=timeout)


def delete_run(run_id: str, only_if_canceled: bool = False) -> dict:
    """Delete a run and its video for good. A run with a live worker is refused (cancel it first). `only_if_canceled`
    (the automatic cleanup): the status and the finished video are checked again INSIDE the lock, so a run that was
    resumed or finished after the caller's check is kept."""
    d = run_dir(run_id)
    with _life_lock(run_id, timeout=0):
        if is_locked(d / "run.lock"):
            raise RunError(f"run {run_id} is working; cancel it first")
        if only_if_canceled and (_status_of(d) != "canceled" or (d / "final.mp4").is_file()):
            raise RunError(f"run {run_id} is no longer a canceled run without a video: kept")
        shutil.rmtree(d)
    return {"id": run_id, "deleted": True}
