"""The rig's joints seen through the shot camera, per frame, in the coco18 order of DWPose (for comparing a template's
motion with a reference video: setup/motion_match.py).

blender -b SHOT.blend --python bpy_jobs/rig_project.py -- OUT.json [--rig MannequinRig]

Per frame: coco18 [x, y] in pixels of the render size (y down) and the world positions of the ankles (foot slide).
No render: only the pose is evaluated (safe next to a GPU job)."""
import argparse
import json
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view

# coco18 joint -> (bone, "head"|"tail")
JOINTS = [("nose", "head", "mid"), ("neck", "neck", "head"),
          ("r_shoulder", "upper_arm.R", "head"), ("r_elbow", "forearm.R", "head"), ("r_wrist", "hand.R", "head"),
          ("l_shoulder", "upper_arm.L", "head"), ("l_elbow", "forearm.L", "head"), ("l_wrist", "hand.L", "head"),
          ("r_hip", "thigh.R", "head"), ("r_knee", "shin.R", "head"), ("r_ankle", "foot.R", "head"),
          ("l_hip", "thigh.L", "head"), ("l_knee", "shin.L", "head"), ("l_ankle", "foot.L", "head")]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--rig", default="MannequinRig")
    a = ap.parse_args(argv)
    sc = bpy.context.scene
    rig = bpy.data.objects[a.rig]
    cam = sc.camera
    W, H = sc.render.resolution_x, sc.render.resolution_y
    frames = []
    for f in range(sc.frame_start, sc.frame_end + 1):
        sc.frame_set(f)
        pts, ankles = [], {}
        for name, bone, end in JOINTS:
            pb = rig.pose.bones[bone]
            p = pb.head if end == "head" else (pb.tail if end == "tail" else (pb.head + pb.tail) / 2)
            w = rig.matrix_world @ p
            v = world_to_camera_view(sc, cam, w)
            pts.append([round(v.x * W, 2), round((1 - v.y) * H, 2)])
            if name.endswith("ankle"):
                ankles[name] = [round(c, 4) for c in w]
        frames.append({"frame": f, "coco": pts, "ankles": ankles})
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"names": [j[0] for j in JOINTS], "width": W, "height": H, "fps": sc.render.fps, "frames": frames}, fh)
    print("RIG_PROJECT", a.out, len(frames))


main()
