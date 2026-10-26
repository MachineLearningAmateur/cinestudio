"""Which graphs the app uses: the FULL (non-distilled) models or the old distilled ones.

User decision 2026-10-20: "i dont want distilled versions" - every tier uses the full models; the tier sets only the
precision (setup/models.json "model_tiers"). The old graphs (LTX-2.3, Z-Image-Turbo "create", ACE-Step turbo "song")
were retired on 2026-10-25 (.cinestudio/backup/workflows_retired_20261025/).
"""
from __future__ import annotations

FULL = True

# video-to-video: LTX-2.5 dev (layout runs use the distilled LTX-2.5: runner.LAYOUT_PROFILE)
LTX_PROFILE = "ltx25dev"
RETIRED_PROFILES = ("ltx23", "ltx23dev")              # old runs keep their frozen graphs: listed and watched, not run
ZIMAGE_VARIANT = "base"                               # character front views: Z-Image base
ACE_VARIANT = "base"                                  # songs: ACE-Step 1.5 base / XL sft
FAST_EDITS = not FULL                                 # the Qwen Lightning LoRA (a distilled shortcut): off


def ace_cfg(bundle: dict) -> float:
    """The official ACE-Step templates: cfg 6 for the base model, 7 for XL sft."""
    return 7.0 if "sft" in bundle.get("ace_model", {}).get("file", "") else 6.0
