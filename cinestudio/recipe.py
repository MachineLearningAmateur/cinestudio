"""Supported generation recipes: distilled INT8 LTX video and the existing image/music graphs.

Full/dev LTX video profiles are retired; old runs remain viewable.
"""
from __future__ import annotations

FULL = True

# All new LTX video-to-video and Blender layout runs use distilled INT8.
LTX_PROFILE = "ltx25"
RETIRED_PROFILES = ("ltx23", "ltx23dev", "ltx25dev")              # old runs keep their frozen graphs: listed and watched, not run
ZIMAGE_VARIANT = "base"                               # character front views: Z-Image base
ACE_VARIANT = "base"                                  # songs: ACE-Step 1.5 base / XL sft
FAST_EDITS = not FULL                                 # the Qwen Lightning LoRA (a distilled shortcut): off


def ace_cfg(bundle: dict) -> float:
    """The official ACE-Step templates: cfg 6 for the base model, 7 for XL sft."""
    return 7.0 if "sft" in bundle.get("ace_model", {}).get("file", "") else 6.0
