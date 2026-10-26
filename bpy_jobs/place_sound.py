"""Put a sound file into a shot as a Video Sequencer sound strip on the shot's first frame and save a copy:

    blender -b shot.blend --python bpy_jobs/place_sound.py -- --sound song.wav --out shot_song.blend

Other sound strips are muted in the copy (the new song is the shot's only sound); the new sound is PACKED into the copy,
so the shot keeps its sound when the song is deleted. The original .blend does not change.
"""
import argparse
import os
import sys

import bpy


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--sound", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", type=int, default=1)
    a = ap.parse_args(argv)
    sc = bpy.context.scene
    se = sc.sequence_editor or sc.sequence_editor_create()
    every = getattr(se, "strips_all", None)
    if every is None:
        every = getattr(se, "sequences_all", [])
    for s in every:
        if s.type == "SOUND":
            s.mute = True
    strips = getattr(se, "strips", None)
    if strips is None:
        strips = se.sequences
    strip = strips.new_sound("song", os.path.abspath(a.sound), a.channel, sc.frame_start)
    strip.mute = False
    strip.sound.pack()              # the sound lives IN the .blend: deleting or moving the song keeps the shot's sound
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out), copy=True)
    print("PLACED", strip.name, strip.frame_start, strip.frame_final_duration)


if __name__ == "__main__":
    main()
