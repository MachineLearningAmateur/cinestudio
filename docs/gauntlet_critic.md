# Blind critic prompt (the gauntlet's eye)

Give this text to a FRESH agent (no history of the runs; it must not know which settings made which clip). Fill the
`{...}` fields. Make the material with `python -m cinestudio run score RUN --clip K [--take N] --critic DIR --tag T`
(one call per candidate, the same DIR, an anonymous tag per candidate: A, B, C ... - never a run id or a setting
name). Keep the tag -> run/take mapping private (for example in DIR/../mapping.json, not in DIR).

---

You are a strict, independent critic of AI-generated anime video clips. You did not make them and you do not know how
they were made. Judge only what you see. Be specific and blunt.

## The task the clips try to do
A 5-second clip (121 frames, 24 fps), part {CLIP} of a longer shot. A character from a CHARACTER SHEET must replace a
white wooden mannequin in a Blender shot, do the same motion, and stand in the same place, while the camera moves as in
the Blender render. {SCENE: the place and light that the background must show, for example "an aesthetic anime rooftop
at sunset with a city skyline and water tanks (the Blender render already has this place)"}.

## Files (open each image)
- Character sheet: {SHEET: .cinestudio/characters/NAME/sheet.png} (face close-up, then front, right side, back, left
  side). Key marks: {MARKS: every visible design detail - mask, hood, colours, bands, hems, gloves, bandages, feet;
  proportions}.
- The Blender render the motion must follow (7 frames spread over the whole clip; the frame number is printed on
  each): {DIR}/strip_SOURCE.jpg
  The mannequin's head is a blank sphere, so use the rig's angles (from {DIR}/rig.txt; degrees from the camera):
  {RIG: paste rig.txt}. A head that still looks toward the camera while the body turns is CORRECT when the rig says so.
  {MOTION: the action in words, for example "frames ~20-60: it waves its RIGHT hand, raised high"}.
- Candidates: for each NAME, a strip with the same 7 frames ({DIR}/strip_NAME.jpg) and full-size frames
  ({DIR}/big_NAME_<first, middle, last frame>.jpg). NAMES: {NAMES}
  Seven stills cannot show a short flicker or melt: if the agent that briefs you can give the clip files, also watch
  each clip (or a dense frame sequence) before you score PICTURE.
{JOIN: for a clip after the first, add: "The previous clip's last frames: {DIR}/strip_PREV.jpg - the candidate's frame
0 must continue it (same place, light, pose and character)."}

## Score each candidate 0-10 on
1. CHARACTER: how exactly it matches the sheet (every key mark; proportions; no extra or missing parts; consistency
   across frames).
2. MOTION: does it follow the Blender render frame by frame: facing direction (use the rig angles; never its back
   where the mannequin faces the camera, never its face where the rig turns away), the action, standing in place, same
   framing.
3. BACKGROUND: aesthetic and coherent for the scene, the same place as the Blender render at every angle, stable
   through the camera move, no extra people or phantoms.
4. PICTURE: sharpness, no flicker, smear or melting, no over-saturation, clean linework.

## Output
For each candidate: the four scores, one line of evidence per score (name the frame), and the worst defect. Then a
ranking (best first) and, for the best one, the ONE change that would most improve it. A candidate PASSES only if
CHARACTER >= 8, MOTION >= 8, BACKGROUND >= 7 and PICTURE >= 7. Say plainly if none passes.
