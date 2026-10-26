# Ideas to try later

## Two-pass video: a motion pass, then LTX for the final picture (2026-10-19)

**Idea (from the user):** a model that is strong at motion makes a rough reference video; LTX-2.3 then makes
the final video from it (the character from the sheet, the style LoRA, the upscale).

**How LTX would use it:** extract the skeleton (pose, with hands) and the depth from every frame of the
motion video, and feed them as the control video, the same way the Blender passes are fed today
(`pose_env` / depth through the union control LoRA). The character still comes from the Ingredients sheet.

**Not MiniMax H3:** its open-weight licence excludes users in the US, EU, UK and Korea, and the project rule
is that H3 is not used. Allowed motion sources for the same idea:

1. **A video from the user** (for example filmed on a phone): pose and depth estimation per frame, then LTX.
   No extra generator model; the most natural motion. Needs a pose estimator node (e.g. DWPose) and a depth
   estimator (e.g. Depth Anything) — check their licences.
2. **Motion capture** (CMU BVH library) retargeted onto the mannequin in Blender (planned for dances).
3. **Wan 2.2 (Apache-2.0) as the motion pass only** — it was dropped for picture quality, not for motion.

**When to try it:** if the Blender mannequin motion is not natural enough (complex actions, dances), or if LTX
alone does not keep the motion.

## Help the standard detector with a detector-friendly Blender pass (2026-10-19)

**Idea (from the user):** keep the standard detector (DWPose) as the control, but improve its precision with what
Blender knows about the mannequin. The detector stays the proven method; only its input gets better.

1. **A face on the mannequin:** simple eye and nose marks on the head, so DWPose can tell front from back (a blank
   sphere head gives no face points, the same front/back ambiguity that broke the control in back views).
2. **Left/right tints** on the limbs, so the detector does not swap them when the camera is behind the figure.
3. **A clean detector pass:** the mannequin alone on a plain background (the mask pass already exists), so props
   cannot make false detections.
4. **A higher detection resolution:** DWPose on the full 1536x896 render (not 512), for hands and fingers.
5. **A check, not a replacement:** compare DWPose's points with the known bone positions from Blender and flag
   frames with a missing or swapped joint (the detector's output stays the control).

**First:** compare the DWPose run with the old bone-drawn control; start with 1 and 4 if DWPose has trouble.
