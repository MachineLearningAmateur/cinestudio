"""SDPose (ComfyUI's native nodes) for setup/dwpose_score.py --detector sdpose. Run with ComfyUI's Python.

It calls the code of ComfyUI's own nodes (comfy_extras/nodes_sdpose.py, nodes_rtdetr.py) in this process, inside
torch.inference_mode() as ComfyUI runs nodes. Resources (measured 2026-10-19): on the CPU about 10 GB of RAM per
process; on the GPU much less RAM, but the 8 GB card must not hold other models (stop the idle ComfyUI first;
never /free). So: at most ONE SDPose process (a lock), a RAM check at the start, and the device is fixed per
process ("cpu" or "cuda"). Boxes are widened to the model's 3:4 shape (the native node stretches each crop); a box
past the image edge is padded with black so the shape survives ("SDPose with expanded boxes", not the bare node).
Models (Comfy-Org/SDPose, MIT):
  checkpoints/sdpose_wholebody_fp16.safetensors          the pose model (Stable Diffusion 2 U-Net + heatmap head)
  diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors   the person finder (RT-DETRv4), the standard multi-person path
"""
import os
import sys
from pathlib import Path

import numpy as np

COMFY = Path(os.environ.get("COMFYUI_DIR", "D:/Comfy-Desktop/ComfyUI-Installs/ComfyUI/ComfyUI"))
MODELS = Path(os.environ.get("COMFYUI_MODELS", "D:/Comfy-Desktop/ComfyUI-Shared/models"))
SDPOSE = MODELS / "checkpoints/sdpose_wholebody_fp16.safetensors"
RTDETR = MODELS / "diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors"


LOCK = Path(os.environ.get("CINESTUDIO_HOME", Path(__file__).resolve().parent.parent / ".cinestudio")) / "sdpose.lock"
MIN_FREE_GB = {"cpu": 12.0, "cuda": 4.0}


class SDPose:
    def __init__(self, device="cpu", threshold=0.5):
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda' (fixed for the whole process)")
        import psutil
        free = psutil.virtual_memory().available / 1e9
        if free < MIN_FREE_GB[device]:
            raise RuntimeError(f"SDPose on {device} needs about {MIN_FREE_GB[device]:.0f} GB of free RAM; "
                               f"{free:.1f} GB is free")
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        try:                                        # at most one SDPose process (the lock goes with the process)
            self._lock = open(LOCK, "x")
            self._lock.write(str(os.getpid()))
            self._lock.flush()
        except FileExistsError:
            pid = LOCK.read_text().strip() or "?"
            if pid.isdigit() and psutil.pid_exists(int(pid)):
                raise RuntimeError(f"another SDPose process runs (pid {pid})")
            LOCK.unlink()                           # left by a process that has ended
            self._lock = open(LOCK, "x")
            self._lock.write(str(os.getpid()))
            self._lock.flush()
        import atexit
        atexit.register(lambda: (self._lock.close(), LOCK.unlink(missing_ok=True)))
        # ComfyUI's "comfy" is a namespace package (no __init__.py): a plain comfy.py anywhere on the path (for example
        # cinestudio/comfy.py when cinestudio/pose_people.py runs as a script) would win. Take such folders off the path.
        sys.path[:] = [d for d in sys.path if not (Path(d or ".").resolve() / "comfy.py").is_file()]
        sys.path.insert(0, str(COMFY))
        import comfy.options
        comfy.options.enable_args_parsing()
        sys.argv = [sys.argv[0]] + (["--cpu"] if device == "cpu" else [])
        import comfy.sd
        import comfy.sample  # noqa: F401  (loaded by ComfyUI at start; the SDPose node needs it)
        import comfy.samplers  # noqa: F401
        import torch  # noqa: F401
        from comfy_extras import nodes_rtdetr, nodes_sdpose
        self.nodes_sdpose, self.nodes_rtdetr = nodes_sdpose, nodes_rtdetr
        out = comfy.sd.load_checkpoint_guess_config(str(SDPOSE), output_vae=True, output_clip=False)
        self.model, self.vae = out[0], out[2]
        self.rtdetr = comfy.sd.load_diffusion_model(str(RTDETR))
        self.threshold = threshold

    @staticmethod
    def _pad(img, boxes):
        """Pad the image with black so every box lies inside it; returns (image, shifted boxes)."""
        h, w = img.shape[:2]
        import math                                 # round UP: a box 0.9 px past the edge still gets padding
        left = math.ceil(max(0, -min(b[0] for b in boxes)))
        top = math.ceil(max(0, -min(b[1] for b in boxes)))
        right = math.ceil(max(0, max(b[2] for b in boxes) - w))
        bottom = math.ceil(max(0, max(b[3] for b in boxes) - h))
        if not (left or top or right or bottom):
            return img, boxes, (0, 0)
        out = np.zeros((h + top + bottom, w + left + right, 3), img.dtype)
        out[top:top + h, left:left + w] = img
        return out, [[b[0] + left, b[1] + top, b[2] + left, b[3] + top] for b in boxes], (left, top)

    @staticmethod
    def _tensor(img):
        import torch
        return torch.from_numpy(np.ascontiguousarray(img)).float().div(255.0)[None]

    def boxes_standard(self, img, threshold=None):
        """RT-DETR person boxes (the standard multi-person path), as SDPose bbox dicts."""
        import torch
        with torch.inference_mode():
            r = self.nodes_rtdetr.RTDETR_detect.execute(self.rtdetr, self._tensor(img), self.threshold if threshold
                                                         is None else threshold, "person", 20)
        return r.result[0][0]

    def person_score(self, img):
        b = self.boxes_standard(img, threshold=0.0)
        return round(max((x["score"] for x in b), default=0.0), 4)

    @staticmethod
    def model_box(b, scale=1.25, aspect=0.75):
        """Widen a person box to the model's shape (3:4) round its centre, with DWPose's top-down margin (1.25):
        the native node stretches each crop to 768x1024, so a tall box would be squeezed."""
        x0, y0, x1, y1 = b
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * scale, (y1 - y0) * scale
        if w / h < aspect:
            w = h * aspect
        else:
            h = w / aspect
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def detect(self, img, boxes=None):
        """boxes None = RT-DETR finds the people; else [[x0, y0, x1, y1], ...] in pixels (Blender boxes).
        Returns (people as the scorer's dicts, the OpenPose frame for drawing)."""
        import torch
        H, W = img.shape[:2]
        if boxes is None:
            boxes = [[x["x"], x["y"], x["x"] + x["width"], x["y"] + x["height"]] for x in self.boxes_standard(img)]
        if not boxes:
            return [], {"canvas_width": W, "canvas_height": H, "people": []}
        boxes = [self.model_box(b) for b in boxes]
        work, boxes, (ox, oy) = self._pad(img, boxes)     # the 3:4 boxes stay whole at the image edges
        bb = [{"x": float(b[0]), "y": float(b[1]), "width": float(b[2] - b[0]), "height": float(b[3] - b[1])}
              for b in boxes]
        with torch.inference_mode():
            r = self.nodes_sdpose.SDPoseKeypointExtractor.execute(self.model, self.vae, self._tensor(work), 1,
                                                                   bboxes=[bb])
        frame = r.result[0][0]
        if ox or oy:                                       # back to the coordinates of the original image
            for person in frame["people"]:
                for key, v in person.items():
                    if isinstance(v, list) and v:
                        for k in range(0, len(v), 3):
                            v[k] -= ox
                            v[k + 1] -= oy
        frame["canvas_width"], frame["canvas_height"] = W, H

        def arr(flat, n):
            a = np.array(flat, float).reshape(n, 3) if flat else np.full((n, 3), np.nan)
            bad = ~(a[:, 2] >= 0.3) | ~np.isfinite(a).all(axis=1)   # score < 0.3, or any NaN / infinite value = absent
            a[bad, :2] = np.nan
            return None if np.all(np.isnan(a[:, 0])) else a
        people = []
        for p in frame["people"]:
            body = arr(p["pose_keypoints_2d"], 18)
            people.append({"body": body if body is not None else np.full((18, 3), np.nan),
                           "L": arr(p.get("hand_left_keypoints_2d"), 21), "R": arr(p.get("hand_right_keypoints_2d"), 21),
                           "face": arr(p.get("face_keypoints_2d"), 70)})
        return people, frame

    def draw(self, frame):
        """The control picture as SDPose Draw Keypoints makes it (node defaults: body, hands, face; no feet)."""
        import torch
        with torch.inference_mode():
            r = self.nodes_sdpose.SDPoseDrawKeypoints.execute([frame], True, True, True, False, 4, 3, 0.3, True)
        return (r.result[0][0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
