"""One frame per engine from Blender's factory scene, for the manual Linux pod smoke test."""
import argparse
from pathlib import Path
import sys

import bpy


parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
args.out.mkdir(parents=True, exist_ok=True)
scene = bpy.context.scene
scene.render.resolution_x = scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
for name, engine in (("workbench", "BLENDER_WORKBENCH"), ("eevee", "BLENDER_EEVEE")):
    scene.render.engine = engine
    scene.render.filepath = str(args.out / f"{name}.png")
    bpy.ops.render.render(write_still=True)
