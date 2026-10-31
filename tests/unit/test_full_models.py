"""The full-model (non-distilled) graphs: conversion, schedule, model slots, family key, no distilled shortcuts."""

import json

import pytest

from cinestudio import character, models, recipe, runner
from cinestudio.workflow import ProfileWorkflow

DEV_VARIANTS = ("ref", "refup", "restyle", "v2v", "v2vref")


def test_the_app_uses_the_full_graphs():
    assert recipe.FULL and runner.PROFILE == "ltx25"
    assert "ltx25dev" in recipe.RETIRED_PROFILES
    assert recipe.ZIMAGE_VARIANT == "base" and recipe.ACE_VARIANT == "base" and not recipe.FAST_EDITS


@pytest.mark.parametrize("tier", ["8gb", "large"])
@pytest.mark.parametrize("variant", ["layout", "layouting", "v2v", "v2vref", "ref", "refup", "restyle", "redraw", "txt2img"])
def test_supported_ltx_workflows_load_the_distilled_int8_model(tier, variant):
    wf = ProfileWorkflow("ltx25", variant, models=models.bundle(tier), use_copy=False)
    files = [n["inputs"]["unet_name"] for n in wf.graph.values()
             if "unet_name" in n.get("inputs", {}) and "ltx-2.5" in n["inputs"]["unet_name"]]
    assert files and all("distilled" in name and "int8" in name for name in files)


@pytest.mark.parametrize("profile", ["ltx25dev"])
@pytest.mark.parametrize("variant", DEV_VARIANTS)
def test_dev_graphs_have_no_distilled_parts(variant, profile):
    wf = ProfileWorkflow(profile, variant)
    assert wf.capabilities["recipe"] == "full"
    text = json.dumps(wf.graph)
    assert "distilled" not in text and "linear_quadratic" not in text
    guiders = [n for n in wf.graph.values() if n["class_type"].endswith("Guider")]
    assert guiders and all(n["class_type"] == "MultimodalGuider" for n in guiders)      # with CFG rescale
    params = [n["inputs"] for n in wf.graph.values() if n["class_type"] == "GuiderParameters"]
    assert params and all(p["cfg"] > 1 and p["rescale"] == 0.7 for p in params)        # cfg > 1, rescaled
    assert "ltx25_model" not in models.bundle("8gb")  # legacy graphs are not downloaded or run


@pytest.mark.parametrize("profile,variant", [("zimage", "base"), ("acestep", "base"), ("qwenedit", "edit"),
                                             ("qwenedit", "angle")])
def test_image_and_music_graphs_take_the_tier(profile, variant):
    for tier in ("8gb", "large"):
        b = models.bundle(tier)
        wf = ProfileWorkflow(profile, variant, models=b)
        used = {wf.graph[n]["inputs"][f] for targets in wf.bind["models"].values() for n, f in targets}
        assert used == {b[s]["file"] for s in wf.bind["models"]}
        assert not any("turbo" in u or "distilled" in u for u in used)


def test_distilled_family_key_is_shared_across_tiers():
    a = ProfileWorkflow("ltx25", "v2v", models=models.bundle("8gb")).family_key()
    b = ProfileWorkflow("ltx25", "v2v", models=models.bundle("large")).family_key()
    assert a.startswith("ltx25@") and a == b
    assert a == ProfileWorkflow("ltx25", "v2vref", models=models.bundle("8gb")).family_key()


def test_dev_schedule_matches_ltxvscheduler_and_keeps_the_start():
    sig = runner.dev_sigmas(768, 448, 121)
    assert len(sig) == 31 and sig[0] == 1.0 and sig[-1] == 0.0
    assert abs(sig[-2] - 0.1) < 1e-9                                      # stretched to the terminal value
    assert all(a > b for a, b in zip(sig, sig[1:]))
    tail = [float(x) for x in runner.dev_source_sigmas(0.985, 768, 448, 121).split(",")]
    assert tail[0] == 0.985 and all(x < 0.985 for x in tail[1:]) and tail[-1] == 0.0
    big = runner.dev_sigmas(1536, 896, 121)                               # more tokens: a larger shift
    assert big[15] > sig[15]


def test_fast_edit_mode_is_off():
    with pytest.raises(character.CharacterError, match="distilled"):
        character.quality_values("fast", None, character.EDIT_SETTINGS)


def test_ace_cfg_follows_the_tier():
    assert recipe.ace_cfg(models.bundle("8gb")) == 6.0
    assert recipe.ace_cfg(models.bundle("large")) == 7.0


def test_the_app_offers_no_distilled_edit_mode():
    from cinestudio import webapp
    assert "fast" not in webapp._edit_modes() and "quality" in webapp._edit_modes()


def test_run_steps_and_guidance_options():
    assert runner.DEFAULTS["steps"] is None and runner.DEFAULTS["guidance"] is None      # None = 30 steps, cfg 3
    few = runner.dev_source_sigmas(0.985, 768, 448, 121, steps=20).split(",")
    many = runner.dev_source_sigmas(0.985, 768, 448, 121).split(",")
    assert len(few) < len(many) and float(few[0]) == 0.985


def test_grid_source_sigmas_use_the_distilled_grid():
    s = [float(x) for x in runner.grid_source_sigmas(0.99).split(",")]
    assert s[0] == 0.99 and s[1:] == [x for x in runner.DISTILLED_SIGMAS if x < 0.99]


def test_pano_world_tag_follows_the_code_and_arguments():
    tag = runner.pano_world_tag()
    assert tag.startswith(runner.PANO_GROUND) and len(tag) == len(runner.PANO_GROUND) + 6
    assert runner.DEFAULTS["ingredients_strength"] == 1.0          # the 2.5 Ingredients file is pre-scaled


def test_latent_relay_guides_sit_at_latent_frame_1_in_both_stages():
    import json as _j
    g = _j.loads((runner.WORKFLOWS / "ltx25_layouting_api.json").read_text(encoding="utf-8"))
    runner.add_relay_save(g, "cinestudio/relay_x_2_s1", "cinestudio/relay_x_2_s2")
    runner.add_relay_guides(g, "a.latent", "b.latent", 0.8, 3)
    assert g["relay_save_crop"]["inputs"]["samples"] == ["crop", 2] and g["relay_save_crop2"]["inputs"]["samples"] == ["crop2", 2]
    for tag, first in (("s1", "ref_guide"), ("s2", "g_lay2")):
        gd = g[f"relay_guide_{tag}"]["inputs"]
        assert gd["latent_idx"] == 1 and gd["strength"] == 0.8
        assert g[f"relay_sel_{tag}"]["inputs"]["start_index"] == -3 and g[f"relay_sel_{tag}"]["inputs"]["end_index"] == -1
        assert g[first]["inputs"]["latent"] == [f"relay_guide_{tag}", 2]
