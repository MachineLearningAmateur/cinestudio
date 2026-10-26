"""A quick preview of a shot for the app (Workbench: light enough to run next to a GPU job):

    blender -b shot.blend --python bpy_jobs/preview_shot.py -- --out <dir> --percent 33

Writes <dir>/f_####.png for every frame of the shot and <dir>/preview.json (frame range, fps, size).
Nothing in the .blend changes (the file is not saved).
"""
import argparse
import json
import os
import sys

import bpy


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--percent", type=int, default=33)
    a = ap.parse_args(argv)
    sc = bpy.context.scene
    if sc.camera is None:
        sys.exit("preview_shot: the scene has no camera")
    try:
        sc.render.engine = "BLENDER_WORKBENCH"
    except TypeError as exc:
        sys.exit(f"preview_shot: no Workbench engine ({exc})")
    sh = sc.display.shading
    sh.light, sh.color_type = "STUDIO", "MATERIAL"
    sh.show_shadows = True
    sc.render.resolution_percentage = a.percent
    sc.render.image_settings.file_format = "PNG"
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    sc.render.filepath = os.path.join(out, "f_####")
    bpy.ops.render.render(animation=True)
    fps = sc.render.fps / sc.render.fps_base
    w = sc.render.resolution_x * a.percent // 100
    h = sc.render.resolution_y * a.percent // 100
    with open(os.path.join(out, "preview.json"), "w") as fh:
        json.dump({"frame_start": sc.frame_start, "frame_end": sc.frame_end, "fps": fps, "width": w, "height": h}, fh)
    print("PREVIEW DONE", sc.frame_start, sc.frame_end)


if __name__ == "__main__":
    main()
