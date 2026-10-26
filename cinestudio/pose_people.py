"""The control of a shot with several characters (run with ComfyUI's Python: it has torch and comfyui_controlnet_aux):

    <ComfyUI python> cinestudio/pose_people.py PEOPLE_DIR OUT_DIR [--aux DIR] [--detector dwpose|sdpose --device cuda]

PEOPLE_DIR is made by bpy_jobs/render_people.py. For every frame and character, the standard DWPose pose model
reads the character's own render (iso/<n>) inside its Blender box (YOLOX does not find the mannequin, and a shared
render mixes the people). --detector fusion runs BOTH and takes each body part from the detector that agrees more with
the rig, keeping the parts connected (setup/pose_fusion.py fuse(), choose "rig"). --detector sdpose uses ComfyUI's native SDPose instead (setup/sdpose_backend.py: the box is
widened to its 3:4 shape; one SDPose process at a time). Blender then checks the result; no point is moved or invented:
  - left/right: when DWPose's limbs fit the rig better with left and right swapped, its labels are swapped
  - hands: each detected hand goes to the nearer wrist of the rig
  - hidden points: a point that ANOTHER character hides (Blender ray test) is removed; so is the dense face when
    the nose is hidden
  - face: points that the face rule hides (the head turned away; render_passes.FACE_VISIBLE_DOT) are removed, and
    the dense face goes when the nose is hidden by that rule (no face on the back of a head)
The skeletons are drawn far to near, as the DWPose node draws them. Writes OUT_DIR/f_####.png and stats.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _find_aux() -> str:
    """The comfyui_controlnet_aux folder: CONTROLNET_AUX, else next to the ComfyUI whose python runs this script
    (<ComfyUI>/.venv/{Scripts,bin}/python -> <ComfyUI>/custom_nodes/comfyui_controlnet_aux)."""
    import os as _os
    import sys as _sys
    env = _os.environ.get("CONTROLNET_AUX")
    if env:
        return env
    for parent in Path(_sys.executable).resolve().parents:
        cand = parent / "custom_nodes" / "comfyui_controlnet_aux"
        if cand.is_dir():
            return str(cand)
    return ""


AUX_DEFAULT = _find_aux()
PAIRS = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (14, 15), (16, 17)]
LR = {a: b for a, b in PAIRS} | {b: a for a, b in PAIRS}


def load_dwpose(aux: Path):
    sys.path.insert(0, str(aux / "src"))
    from custom_controlnet_aux.dwpose import DwposeDetector
    return DwposeDetector.from_pretrained("hr16/DWPose-TorchScript-BatchSize5", "hr16/yolox-onnx",
                                          det_filename="yolox_l.torchscript.pt",
                                          pose_filename="dw-ll_ucoco_384_bs5.torchscript.pt", torchscript_device="cpu")


def pose_in_box(det, img, box):
    """DWPose's pose model in one box (the same steps as Wholebody.__call__ after its detector)."""
    import torch
    from custom_controlnet_aux.dwpose.dw_torchscript.jit_pose import inference_pose
    from custom_controlnet_aux.dwpose.wholebody import Wholebody
    w = det.dw_pose_estimation
    with torch.inference_mode():
        keypoints, scores = inference_pose(w.pose, np.asarray([box], float), img, w.pose_input_size)
    info = np.concatenate((keypoints, scores[..., None]), axis=-1)
    neck = np.mean(info[:, [5, 6]], axis=1)
    neck[:, 2:4] = np.logical_and(info[:, 5, 2:4] > 0.3, info[:, 6, 2:4] > 0.3).astype(int)
    new = np.insert(info, 17, neck, axis=1)
    new[:, [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]] = new[:, [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]]
    return Wholebody.format_result(new)[0]


def sdpose_pose(det, img, box):
    """SDPose in one box -> a controlnet_aux PoseResult (like pose_in_box), or None."""
    from custom_controlnet_aux.dwpose.types import BodyResult, Keypoint, PoseResult
    people, _ = det.detect(img, [box])
    if not people:
        return None
    p = people[0]

    def kps(a):
        if a is None:
            return None
        out = [Keypoint(float(x), float(y), float(sc)) if np.isfinite(x) and np.isfinite(y) else None for x, y, sc in a]
        return None if all(k is None for k in out) else out
    return PoseResult(BodyResult(kps(p["body"]) or [None] * 18, 0.0, 18), kps(p["L"]), kps(p["R"]), kps(p["face"]))


def _as_arrays(pose):
    """A PoseResult -> the dict of (n, 3) arrays that setup/pose_fusion.py uses (NaN = missing)."""
    if pose is None:
        return None

    def arr(kps, n):
        if kps is None:
            return None
        a = np.array([[k.x, k.y, k.score] if k is not None else [np.nan] * 3 for k in kps], float)
        a[~np.isfinite(a).all(axis=1)] = np.nan
        return None if np.isnan(a[:, 0]).all() else a
    body = arr(pose.body.keypoints, 18)
    return {"body": body if body is not None else np.full((18, 3), np.nan), "L": arr(pose.left_hand, 21),
            "R": arr(pose.right_hand, 21), "face": arr(pose.face, 70)}


def fused_pose(dw, sdp, kp, i, box, stats):
    """DWPose and SDPose merged part by part (chosen by the rig, joins checked) -> a PoseResult."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "setup"))
    import pose_fusion
    from custom_controlnet_aux.dwpose.types import BodyResult, Keypoint, PoseResult
    rig = np.array(kp["frames"][i], float)
    rig_hands = {s: np.array(kp["hands"][s][i], float) for s in ("L", "R")} if "hands" in kp else None
    h = max(box[3] - box[1], 1.0)
    fused, src = pose_fusion.fuse(_as_arrays(dw), _as_arrays(sdp), h, rig, "rig", {}, rig_hands)
    for k, v in src.items():
        if k in ("relabelled", "absent", "whole"):
            continue
        key = k.split("_")[0] if k.startswith(("relinked_", "dropped_")) else f"{k}={v}"
        stats[key] = stats.get(key, 0) + 1
    if fused is None:
        return None

    def kps(a):
        if a is None:
            return None
        out = [Keypoint(float(x), float(y), float(sc)) if np.isfinite([x, y]).all() else None for x, y, sc in a]
        return None if all(k is None for k in out) else out
    return PoseResult(BodyResult(kps(fused["body"]) or [None] * 18, 0.0, 18), kps(fused.get("L")), kps(fused.get("R")),
                      kps(fused.get("face")))


def _xy(k):
    """A point's (x, y), or None when it is missing or not a finite number."""
    if k is None or not (np.isfinite(k.x) and np.isfinite(k.y)):
        return None
    return np.array([k.x, k.y])


HAND_GAP = 0.08          # a hand must start within this share of the figure height from its body wrist


def assign_hands(hands, body, rb, stats):
    """The detected hands one-to-one to the left and right wrist (the pair of sides with the smaller distance to the
    RIG's wrists); then each hand must start near the FINAL body wrist (or the rig's wrist when the body has none),
    else it is dropped. Nothing is lost by two hands that prefer the same wrist."""
    shown = rb[rb[:, 2] > 0, 1]
    h = float(np.ptp(shown)) if len(shown) > 1 else 1.0
    cand = [x for x in hands if x is not None and _xy(x[0]) is not None]
    sides = {"L": 7, "R": 4}
    best = None
    from itertools import permutations
    for perm in permutations(("L", "R"), len(cand)):
        cost = sum(np.linalg.norm(_xy(hand[0]) - rb[sides[sd], :2]) for hand, sd in zip(cand, perm))
        if best is None or cost < best[0]:
            best = (cost, perm)
    out = {"L": None, "R": None}
    for hand, sd in zip(cand, best[1] if best else ()):
        anchor = _xy(body[sides[sd]])              # a hand needs its detected body wrist (no floating hand)
        if anchor is not None and np.linalg.norm(_xy(hand[0]) - anchor) <= HAND_GAP * h:
            out[sd] = hand
        else:
            stats["hands_dropped"] = stats.get("hands_dropped", 0) + 1
    return out


FACE_RULE = os.environ.get("CINESTUDIO_FACE_RULE", "nose")   # "nose" (default), "both_eyes", "head" or "off"
FACE_MAX_DEG = 60.0     # "head" rule: the dense face only when the face turns less than this from the camera


def checked(pose, rig_body, rig_hands, occ, stats, face_rule=None, head_yaw=None):
    """The Blender checks on one DWPose result. rig_body: 18 [x, y, shown]; rig_hands: {"L"/"R": 21 points};
    head_yaw: [degrees, side] from render_passes (the "head" face rule needs it)."""
    from custom_controlnet_aux.dwpose.types import BodyResult, PoseResult
    body = list(pose.body.keypoints)
    rb = np.array(rig_body, float)
    st = sw = 0.0
    n = 0
    for a, b in PAIRS[:6]:
        pa, pb = _xy(body[a]), _xy(body[b])
        if pa is not None and pb is not None:
            n += 1
            st += np.linalg.norm(pa - rb[a, :2]) + np.linalg.norm(pb - rb[b, :2])
            sw += np.linalg.norm(pa - rb[b, :2]) + np.linalg.norm(pb - rb[a, :2])
    hands = [pose.left_hand, pose.right_hand]
    if n >= 2 and sw < 0.85 * st:                        # DWPose has left and right the wrong way round
        body = [body[LR.get(i, i)] for i in range(18)]
        hands = hands[::-1]
        stats["relabelled"] += 1
    face_hidden_by_rule = rb[0, 2] == 0                  # the head turned away (or the nose out of the frame)
    body = [None if _xy(k) is None else k for k in body]             # not finite = missing
    for i in range(18):
        if body[i] is None:
            continue
        if occ["body"][i] or (i in (0, 14, 15, 16, 17) and rb[i, 2] == 0):
            body[i] = None
            stats["removed"] += 1
    # hands: one-to-one to the wrists, attached to the final body wrist; then hide what another character hides
    out = assign_hands(hands, body, rb, stats)
    for side in ("L", "R"):
        if out[side] is not None:
            flags = occ.get(side) or [False] * 21
            kept = [None if (k is None or flags[j] or _xy(k) is None) else k for j, k in enumerate(out[side])]
            out[side] = None if all(k is None for k in kept) else kept
    face = None if (occ["body"][0] or face_hidden_by_rule) else pose.face
    # "both_eyes": the dense face only when the rig shows BOTH eyes (the head faces the camera); a profile or a back view
    # keeps only the OpenPose eye / ear / nose points - a face blob on a side view pulls the model to turn it frontal
    # "head": the dense face only when the real head angle (Blender) is below FACE_MAX_DEG - the eye flags keep both
    # eyes up to about 75 degrees, so "both_eyes" still draws a face on a three-quarter view
    rule = face_rule or FACE_RULE
    if face is not None and rule == "both_eyes" and not (rb[14, 2] > 0 and rb[15, 2] > 0):
        face = None
        stats["profile_faces_removed"] = stats.get("profile_faces_removed", 0) + 1
    if face is not None and rule == "off":       # no dense face at all (the 5 OpenPose face points stay)
        face = None
    if face is not None and rule == "head" and head_yaw is not None and head_yaw[0] >= FACE_MAX_DEG:
        face = None
        stats["profile_faces_removed"] = stats.get("profile_faces_removed", 0) + 1
    return PoseResult(BodyResult(body, pose.body.total_score, pose.body.total_parts), out["L"], out["R"], face)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("people_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--aux", default=AUX_DEFAULT)
    ap.add_argument("--detector", choices=("dwpose", "sdpose", "fusion"), default="dwpose")
    ap.add_argument("--device", default="cpu", help="sdpose only: cpu or cuda")
    ap.add_argument("--face-rule", choices=("nose", "both_eyes", "head", "off"), default=None,
                    help="when to draw the dense face: nose = whenever the rig shows the nose (default); both_eyes = "
                         "only when the rig shows both eyes; head = only when the head turns less than 60 degrees "
                         "from the camera (Blender head angle) (env CINESTUDIO_FACE_RULE)")
    a = ap.parse_args(argv)
    import cv2
    src, out = Path(a.people_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((src / "people.json").read_text())
    sd = det = None
    if a.detector in ("dwpose", "fusion"):
        det = load_dwpose(Path(a.aux))
    if a.detector in ("sdpose", "fusion"):               # (the drawing and the types of controlnet_aux too)
        sys.path.insert(0, str(Path(a.aux) / "src"))
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "setup"))
        from sdpose_backend import SDPose
        sd = SDPose(a.device)
    from custom_controlnet_aux.dwpose import draw_poses
    stats = {"detector": a.detector, "frames": 0, "people": 0, "relabelled": 0, "removed": 0}
    W, H = info["width"], info["height"]
    control_points = {}
    for f in range(info["frame_start"], info["frame_end"] + 1):
        i = f - info["frame_start"]
        people = []
        for n, ch in enumerate(info["frames"][str(f)]):
            if not ch["box"]:
                continue
            img = cv2.imread(str(src / "iso" / str(n) / f"rgb_{f:04d}.png"))
            if img is None:
                sys.exit(f"missing render iso/{n}/rgb_{f:04d}.png")
            rgb = np.ascontiguousarray(img[:, :, ::-1])
            kp = info["keypoints"][n]
            if a.detector == "fusion":
                pose = fused_pose(pose_in_box(det, rgb, ch["box"]), sdpose_pose(sd, rgb, ch["box"]), kp, i, ch["box"], stats)
            elif sd:
                pose = sdpose_pose(sd, rgb, ch["box"])
            else:
                pose = pose_in_box(det, rgb, ch["box"])
            if pose is None:
                stats["missing"] = stats.get("missing", 0) + 1
                continue
            rig_hands = {s: kp["hands"][s][i] for s in ("L", "R")} if "hands" in kp else {}
            yaw = kp["head_yaw"][i] if "head_yaw" in kp else None
            final = checked(pose, kp["frames"][i], rig_hands, ch, stats, a.face_rule, yaw)
            people.append((ch["depth"], final))
            control_points.setdefault(str(f), {})[info["characters"][n]["name"]] = [
                [k.x, k.y, 1] if k is not None else [0.0, 0.0, 0] for k in final.body.keypoints]
            stats["people"] += 1
        people.sort(key=lambda t: -t[0])                 # far first: the near character is drawn on top
        canvas = draw_poses([p for _, p in people], H, W, draw_body=True, draw_hand=True, draw_face=True)
        cv2.imwrite(str(out / f"f_{f:04d}.png"), canvas[:, :, ::-1])
        stats["frames"] += 1
    (out / "stats.json").write_text(json.dumps(stats))
    # the points that were really drawn (after every check): the second reference of the motion follow score
    (out / "control_points.json").write_text(json.dumps({"detector": a.detector, "frames": control_points}))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
