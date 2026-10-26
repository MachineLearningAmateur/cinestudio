"""Pack every external sound of a shot into the .blend and save it in place (one-off repair for shots whose song
strip still points to a file in the song library):

    blender -b shot.blend --python bpy_jobs/pack_sounds.py
"""
import bpy


def main():
    packed = 0
    for snd in bpy.data.sounds:
        if snd.packed_file is None:
            snd.pack()
            packed += 1
    bpy.ops.wm.save_mainfile()
    print("PACKED", packed, "of", len(bpy.data.sounds))


if __name__ == "__main__":
    main()
