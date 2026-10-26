"""The shot's own sound (the sound strips of the Video Sequencer) as a list for cinestudio to mix:

    blender -b shot.blend --python bpy_jobs/export_sound.py -- --out DIR

Writes DIR/sound.json {"sound": true|false, "strips": [...], "speakers": n, "frame_start", "frame_end", "fps"}. Each strip
has its file (a packed sound is written to DIR), where its sound starts (frame_start), the part that plays
(frame_final_start..frame_final_end) and its volume. cinestudio mixes them with ffmpeg (video.mix_strips).
Blender's own mixdown is not used: in background mode (blender -b) Blender 5 quits before the mixdown job ends, and
the file is empty or missing. Speakers (3D sound) are counted but not mixed. Nothing in the .blend changes.
"""
import argparse
import json
import os
import re
import sys

import bpy


def sound_strips(sc):
    se = sc.sequence_editor
    if se is None:
        return []
    every = getattr(se, "strips_all", None)          # Blender 4.4+; older versions call it sequences_all
    if every is None:
        every = getattr(se, "sequences_all", [])
    return [s for s in every if s.type == "SOUND" and not s.mute and s.sound is not None]


def strip_file(s, out, i):
    snd = s.sound
    if snd.packed_file is not None:
        ext = os.path.splitext(snd.filepath)[1] or ".wav"
        stem = os.path.splitext(snd.name)[0] if snd.name.lower().endswith(ext.lower()) else snd.name
        path = os.path.join(out, f"packed_{i}_{re.sub(r'[^A-Za-z0-9_.-]', '_', stem)}{ext}")
        with open(path, "wb") as fh:
            fh.write(snd.packed_file.data)
        return path
    return os.path.abspath(bpy.path.abspath(snd.filepath, library=snd.library))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    sc = bpy.context.scene
    strips = []
    for i, s in enumerate(sound_strips(sc)):
        f = strip_file(s, out, i)
        if not os.path.isfile(f):
            print("MISSING SOUND FILE", f)
            continue
        strips.append({"name": s.name, "file": f, "frame_start": s.frame_start, "frame_final_start": s.frame_final_start,
                       "frame_final_end": s.frame_final_end, "volume": s.volume, "channel": s.channel})
    speakers = [o for o in sc.objects if o.type == "SPEAKER" and o.data.sound and not o.data.muted]
    info = {"sound": bool(strips), "strips": strips, "speakers": len(speakers),
            "frame_start": sc.frame_start, "frame_end": sc.frame_end, "fps": sc.render.fps / sc.render.fps_base}
    with open(os.path.join(out, "sound.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print("SOUND DONE", json.dumps({k: v for k, v in info.items() if k != "strips"}), len(strips), "strips")


if __name__ == "__main__":
    main()
