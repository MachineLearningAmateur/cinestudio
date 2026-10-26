"""Score how well DWPose reads a Blender render: DWPose points vs the true points from the rig's bones.

Run it with ComfyUI's Python (it has torch and comfyui_controlnet_aux); the CPU is the default, so it can run
while a GPU job runs:

    <ComfyUI python> setup/dwpose_score.py --passes DIR --box blender --out score.json [--overlay DIR]

DIR is made by setup/dwpose_bench_render.py (rgb/rgb_<frame>.png, boxes.json, keypoints.json, passes.json) or by
bpy_jobs/render_passes.py (no boxes.json: use --box yolox / full / mask). The frame number comes from the file
name. The score is the base for making a mannequin that DWPose reads well (docs/IDEAS.md).

The scale of a character is the height of its Blender box (the projected mesh, head to shoes); without
boxes.json, the height of its true points. Per character and frame (a "row"):

  found        DWPose gave this character (yolox: a matched person; blender: a pose in its own box)
  body         share of the true body points (COCO-18 without the 5 face points, in the frame) that DWPose put
               within 5% of the height; the neck is the middle of the shoulders (as DWPose makes it)
  body_lr      the same, but with left and right swapped if that fits better (the position quality alone)
  swap         left/right fit better swapped (1), not (0), or cannot tell (None: fewer than 2 pairs, or the two
               fits are within 15%)
  face5_ok     face points (nose, eyes, ears) that should show, that DWPose put within 5% of the height
  face5_miss   face points that should show but DWPose did not give
  face5_extra  face points that the face rule hides but DWPose gave (the rule: render_passes.FACE_VISIBLE_DOT,
               a cinestudio control rule, not part of DWPose)
  face70_back  dense face points (DWPose's 68 + 2) that DWPose gave when the true nose and both eyes are hidden
               (a back view: these draw a face on the back of the head in the control)
  hand_pts     per eligible hand (all 21 true points in the frame): share of the 21 points within
               max(20% of the hand length, 2% of the height); hand_px = median error in pixels
  yolox_person the highest YOLOX person score in the frame (the node needs 0.3): how near YOLOX is to a find
Per frame: people (DWPose people) and expected (characters in the frame) -> people_wrong counts frames.
Sides (front/left/back/right) use each character's own camera angle (passes.json character_azimuth_deg).
Undefined values (nothing eligible) are null.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

AUX = Path(os.environ.get("CONTROLNET_AUX",
                          "D:/Comfy-Desktop/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/comfyui_controlnet_aux"))
PAIRS = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13)]      # right/left limbs in OpenPose-18
FACE_PAIRS = [(14, 15), (16, 17)]
FACE = (0, 14, 15, 16, 17)
BODY = tuple(i for i in range(18) if i not in FACE)


def load_detector(device):
    sys.path.insert(0, str(AUX / "src"))
    from custom_controlnet_aux.dwpose import DwposeDetector
    return DwposeDetector.from_pretrained("hr16/DWPose-TorchScript-BatchSize5", "hr16/yolox-onnx",
                                          det_filename="yolox_l.torchscript.pt",
                                          pose_filename="dw-ll_ucoco_384_bs5.torchscript.pt",
                                          torchscript_device=device)


def pose_in_boxes(det, img, boxes):
    """DWPose's pose model on given person boxes (x0, y0, x1, y1), skipping its YOLOX person finder.
    The same steps as Wholebody.__call__ after the detector: the neck, then the OpenPose order."""
    import torch
    from custom_controlnet_aux.dwpose.dw_torchscript.jit_pose import inference_pose
    from custom_controlnet_aux.dwpose.wholebody import Wholebody
    w = det.dw_pose_estimation
    with torch.no_grad():
        keypoints, scores = inference_pose(w.pose, np.asarray(boxes, float), img, w.pose_input_size)
    info = np.concatenate((keypoints, scores[..., None]), axis=-1)
    neck = np.mean(info[:, [5, 6]], axis=1)
    neck[:, 2:4] = np.logical_and(info[:, 5, 2:4] > 0.3, info[:, 6, 2:4] > 0.3).astype(int)
    new = np.insert(info, 17, neck, axis=1)
    new[:, [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]] = new[:, [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]]
    return Wholebody.format_result(new)


def mask_box(mask):
    """ONE character's box from its Blender mask (white = the character; all its parts together)."""
    ys, xs = np.nonzero(mask > 127)
    return [[xs.min(), ys.min(), xs.max(), ys.max()]] if len(xs) else []


def detect(det, img, boxes=None, yolox_bgr=False):
    """DWPose on one RGB uint8 image -> list of people {"body": (18,3), "L", "R": (21,3)|None, "face": (70,3)|None}.
    boxes None = the standard way (YOLOX finds the people); else the pose model runs once per box, in order."""
    if boxes is None:
        if yolox_bgr:        # ablation: YOLOX gets BGR (as upstream YOLOX), the pose model still gets RGB
            w = det.dw_pose_estimation
            from custom_controlnet_aux.dwpose.dw_torchscript.jit_det import inference_detector
            b = inference_detector(w.det, np.ascontiguousarray(img[:, :, ::-1]), detect_classes=[0])
            poses = pose_in_boxes(det, img, b) if b is not None and len(b) else []
        else:
            poses = det.detect_poses(img)
    else:
        poses = pose_in_boxes(det, img, boxes) if len(boxes) else []

    def arr(kps):
        if kps is None:
            return None
        return np.array([[k.x, k.y, k.score] if k is not None else [np.nan, np.nan, 0.0] for k in kps])
    return [{"body": arr(p.body.keypoints), "L": arr(p.left_hand), "R": arr(p.right_hand), "face": arr(p.face)}
            for p in poses], poses


def truth(kp, i, boxes_frame, az_chars=None):
    """The true people of frame i: per character {"name", "body": (18,3) [x, y, shown], "L", "R", "h"}."""
    people = kp["characters"] if "characters" in kp else [kp]
    out = []
    for n, c in enumerate(people):
        b = np.array(c["frames"][i], float)
        b[1, :2] = (b[2, :2] + b[5, :2]) / 2       # DWPose's neck: the middle of the shoulders
        b[1, 2] = float(b[2, 2] > 0 and b[5, 2] > 0)
        t = {"name": c.get("name", f"character{n + 1}"), "body": b}
        for s in ("L", "R"):
            t[s] = np.array(c["hands"][s][i], float) if "hands" in c and s in c["hands"] else None
        box = boxes_frame[n] if boxes_frame and n < len(boxes_frame) else None
        if box and (box[2] - box[0] < 4 or box[3] - box[1] < 4):
            box = None                              # a degenerate box: the character is not in the frame
        t["box"] = box
        t["az"] = az_chars[n][i] if az_chars and n < len(az_chars) and az_chars[n] else None
        if box:
            t["h"] = float(box[3] - box[1])
        else:
            s = b[b[:, 2] > 0, 1]
            t["h"] = float(np.ptp(s)) if len(s) > 1 else 1.0
        out.append(t)
    return out


def _dist(a, b):
    return np.linalg.norm(a[:2] - b[:2])


def match(truths, dets):
    """yolox mode: pair true people with DWPose people. Cost = mean body distance / height over the true body
    points, a missing point costs 0.5; pairs above 0.25 or with fewer than 4 common points stay unmatched."""
    from scipy.optimize import linear_sum_assignment
    if not truths or not dets:
        return [-1] * len(truths)
    big = 1e6
    cost = np.full((len(truths), len(dets)), big)
    for i, t in enumerate(truths):
        shown = [k for k in BODY if t["body"][k, 2] > 0]
        for j, d in enumerate(dets):
            sw = {x: y for x, y in PAIRS} | {y: x for x, y in PAIRS}
            h = max(t["h"], 1.0)
            fits = []
            for mp in (lambda k: k, lambda k: sw.get(k, k)):      # a left/right swap is still this person
                common = [k for k in shown if not np.isnan(d["body"][mp(k), 0])]
                if len(common) >= 4:
                    fits.append(sum(min(_dist(t["body"][k], d["body"][mp(k)]) / h, 0.5) for k in common)
                                + 0.5 * (len(shown) - len(common)))
            if not fits:
                continue
            c = min(fits) / max(len(shown), 1)
            if c <= 0.25:
                cost[i, j] = c
    rows, cols = linear_sum_assignment(cost)
    out = [-1] * len(truths)
    for r, c in zip(rows, cols):
        if cost[r, c] < big:
            out[r] = int(c)
    return out


def _pck(tb, db, idx, h, swap=False):
    sw = {a: b for a, b in PAIRS + FACE_PAIRS} | {b: a for a, b in PAIRS + FACE_PAIRS}
    ok = 0
    for i in idx:
        j = sw.get(i, i) if swap else i
        if not np.isnan(db[j, 0]) and _dist(tb[i], db[j]) < 0.05 * h:
            ok += 1
    return ok / len(idx) if idx else np.nan


def score_person(t, d):
    h = max(t["h"], 1.0)
    tb = t["body"]
    shown = tb[:, 2] > 0
    body = [i for i in BODY if shown[i]]
    face_shown = [i for i in FACE if shown[i]]
    face_hidden = [i for i in FACE if not shown[i]]
    # a back view: the character's own camera angle (Blender) is behind it and its head is in the frame
    head_in = bool(t["body"][16, 2] > 0 or t["body"][17, 2] > 0)
    back_view = t["az"] is not None and 135 <= t["az"] % 360 < 225 and head_in
    hands = [s for s in ("L", "R") if t[s] is not None and t[s][:, 2].sum() == 21]      # eligibility: truth only
    r = {"found": int(d is not None), "eligible_hands": len(hands), "back_view": int(back_view),
         "eligible_body": len(body)}
    if d is None:
        z = 0.0 if body else np.nan
        r.update(body=z, body_lr=z, swap=None, face5_ok=0, face5_miss=len(face_shown), face5_extra=0,
                 face5_shown=len(face_shown), face70_back=0, hand_pts=[0.0] * len(hands), hand_px=[None] * len(hands))
        return r
    db = d["body"]
    found = ~np.isnan(db[:, 0])
    r["body"] = _pck(tb, db, body, h)
    r["body_lr"] = max(r["body"], _pck(tb, db, body, h, swap=True))
    straight = swapped = 0.0
    pairs = 0
    for a, b in PAIRS:
        if shown[a] and shown[b] and found[a] and found[b]:
            pairs += 1
            straight += _dist(tb[a], db[a]) + _dist(tb[b], db[b])
            swapped += _dist(tb[a], db[b]) + _dist(tb[b], db[a])
    if pairs < 2 or abs(straight - swapped) < 0.15 * max(straight, swapped, 1e-6):
        r["swap"] = None
    else:
        r["swap"] = int(swapped < straight)
    r["face5_shown"] = len(face_shown)
    r["face5_ok"] = int(sum(1 for i in face_shown if found[i] and _dist(tb[i], db[i]) < 0.05 * h))
    r["face5_miss"] = int(sum(1 for i in face_shown if not found[i]))
    r["face5_extra"] = int(sum(1 for i in face_hidden if found[i]))
    r["face70_back"] = int(np.sum(~np.isnan(d["face"][:, 0]))) if back_view and d["face"] is not None else 0
    r["hand_pts"], r["hand_px"] = [], []
    for s in hands:
        th, dh = t[s], d[s]
        if dh is None:
            r["hand_pts"].append(0.0)
            r["hand_px"].append(None)
            continue
        size = max(_dist(th[0], th[12]), 1.0)
        tol = max(0.2 * size, 0.02 * h)
        e = np.linalg.norm(th[:, :2] - dh[:, :2], axis=1)
        r["hand_pts"].append(float(np.sum(np.nan_to_num(e, nan=1e9) < tol) / 21))
        r["hand_px"].append(float(np.nanmedian(e)) if np.any(~np.isnan(e)) else None)
    return r


def yolox_person(det, img):
    """The highest YOLOX person score in the frame (objectness x person class; the node keeps boxes above 0.3),
    on RGB as the node gives it. A measure of how near the figure is to being found."""
    import torch
    from custom_controlnet_aux.dwpose.dw_torchscript import jit_det as J
    m = det.dw_pose_estimation.det
    x, _ = J.preprocess(img, (640, 640))
    dev, dt = next(m.parameters()).device, next(m.parameters()).dtype
    with torch.no_grad():
        out = m(torch.from_numpy(x[None]).to(dev, dt)).float().cpu().numpy()
    pr = J.demo_postprocess(out[0], (640, 640))
    return round(float((pr[:, 4] * pr[:, 5]).max()), 4)


def side(az):
    if az is None:
        return "unknown"
    a = az % 360
    return "front" if a < 45 or a >= 315 else "left" if a < 135 else "back" if a < 225 else "right"


def _mean(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return round(float(np.mean(v)), 3) if v else None


def summary(rows, frames):
    if not rows:
        return {}
    swaps = [r["swap"] for r in rows if r["swap"] is not None]
    hp = [x for r in rows for x in r["hand_pts"]]
    hpx = [x for r in rows for x in r["hand_px"] if x is not None]
    shown5 = sum(r["face5_shown"] for r in rows)
    return {
        "rows": len(rows), "frames": len(frames),
        "found": _mean([r["found"] for r in rows]),
        "body": _mean([r["body"] for r in rows]),
        "body_lr": _mean([r["body_lr"] for r in rows]),
        "yolox_person": _mean([f["yolox_person"] for f in frames.values()]),
        "swap": f"{sum(swaps)}/{len(swaps)}", "swap_unclear": sum(r["swap"] is None for r in rows),
        "face5_ok": round(sum(r["face5_ok"] for r in rows) / shown5, 3) if shown5 else None,
        "face5_miss": sum(r["face5_miss"] for r in rows), "face5_extra": sum(r["face5_extra"] for r in rows),
        "face70_back": sum(r["face70_back"] for r in rows),
        "back_rows": sum(r["back_view"] for r in rows),
        "hand_pts": round(float(np.mean(hp)), 3) if hp else None,
        "hand_px": round(float(np.median(hpx)), 1) if hpx else None,
        "hands_eligible": len(hp),
        "people_wrong": sum(1 for f in frames.values() if f["people"] != f["expected"]),
    }


def draw(img, truths, dets, path):
    import cv2
    v = img[:, :, ::-1].copy()
    for t in truths:
        for x, y, s in t["body"]:
            if s > 0:
                cv2.circle(v, (int(x), int(y)), 6, (0, 255, 0), 2)
    for d in dets:
        for x, y, _ in d["body"]:
            if not np.isnan(x):
                cv2.circle(v, (int(x), int(y)), 3, (0, 0, 255), -1)
        if d["face"] is not None:
            for x, y, _ in d["face"]:
                if not np.isnan(x):
                    cv2.circle(v, (int(x), int(y)), 1, (255, 255, 255), -1)
        for s in ("L", "R"):
            if d[s] is not None:
                for x, y, _ in d[s]:
                    if not np.isnan(x):
                        cv2.circle(v, (int(x), int(y)), 2, (255, 0, 255) if s == "L" else (255, 255, 0), -1)
    cv2.imwrite(str(path), v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--passes", required=True, help="pass folder (rgb/, keypoints.json, passes.json[, boxes.json])")
    ap.add_argument("--every", type=int, default=1, help="score every Nth rendered frame")
    ap.add_argument("--width", type=int, default=768, help="resize the frame to this width first (the pipeline "
                    "runs DWPose on the source video at the generation size); 0 = full size")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--detector", choices=("dwpose", "sdpose"), default="dwpose",
                    help="dwpose = comfyui_controlnet_aux DWPose (YOLOX boxes); sdpose = ComfyUI's native SDPose "
                         "(RT-DETRv4 boxes; setup/sdpose_backend.py). --box yolox means the detector's own person finder")
    ap.add_argument("--box", choices=("yolox", "yolox_bgr", "full", "mask", "blender", "blender_iso"), default="yolox",
                    help="yolox = standard DWPose; yolox_bgr = YOLOX gets BGR (ablation); full = the whole frame; "
                         "mask = one box from the mask pass (one character only); blender = boxes.json (one box "
                         "per character, the pose stays with its character); blender_iso = like blender, but each "
                         "character's pose comes from its render alone (iso/<n>/, bench --isolate)")
    ap.add_argument("--pad", type=float, default=0.0, help="extra box margin per side, as a share of the box "
                    "size (DWPose adds its own 25%%)")
    ap.add_argument("--out", help="write the full result as JSON here")
    ap.add_argument("--overlay", help="folder for frames with the true (green) and DWPose (red) points")
    ap.add_argument("--control", help="folder for the control pictures as the DWPose node draws them")
    ap.add_argument("--save-dets", help="write every detected person per character and frame here (JSON), for "
                    "setup/pose_fusion.py")
    a = ap.parse_args(argv)

    import cv2
    p = Path(a.passes)
    kp = json.loads((p / "keypoints.json").read_text())
    info = json.loads((p / "passes.json").read_text())
    boxes_json = json.loads((p / "boxes.json").read_text()) if (p / "boxes.json").exists() else None
    if a.box in ("blender", "blender_iso") and boxes_json is None:
        sys.exit("--box blender needs boxes.json (setup/dwpose_bench_render.py)")
    az = info.get("camera_azimuth_deg") or [0.0] * info["frames"]
    az_chars = info.get("character_azimuth_deg")         # per character (bench); else the scene camera angle
    if boxes_json:
        names = [c.get("name") for c in kp.get("characters", [kp])]
        assert boxes_json["characters"] == names, (boxes_json["characters"], names)
    frames = sorted((p / "rgb").glob("*.png"))
    if a.detector == "sdpose":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from sdpose_backend import SDPose
        sdp = SDPose(a.device)
        det = None
    else:
        det = load_detector(a.device)

    def run(image, bxs, bgr=False):
        if a.detector == "sdpose":
            ppl, frame = sdp.detect(image, bxs)
            return ppl, frame
        return detect(det, image, bxs, yolox_bgr=bgr)
    if a.overlay:
        Path(a.overlay).mkdir(parents=True, exist_ok=True)
    rows, per_frame, saved = [], {}, {}
    for n in range(0, len(frames), a.every):
        fnum = int(frames[n].stem.split("_")[-1])
        i = fnum - info["frame_start"]
        full = cv2.imread(str(frames[n]))[:, :, ::-1]
        img, k = full, 1.0
        if a.width and img.shape[1] != a.width:
            k = a.width / img.shape[1]
            img = cv2.resize(img, (a.width, round(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
        img = np.ascontiguousarray(img)
        bframe = boxes_json["frames"].get(str(fnum)) if boxes_json else None
        boxes = None
        if a.box == "full":
            boxes = [[0, 0, img.shape[1] - 1, img.shape[0] - 1]]
        elif a.box == "mask":
            mk = cv2.imread(str(p / "mask" / frames[n].name.replace("rgb_", "mask_")), cv2.IMREAD_GRAYSCALE)
            boxes = [[v * k for v in b] for b in mask_box(mk)]
        elif a.box in ("blender", "blender_iso"):
            bframe = [b if b and b[2] - b[0] >= 4 and b[3] - b[1] >= 4 else None for b in bframe]
            boxes = [b for b in bframe if b]
            boxes = [[(b[0] - (b[2] - b[0]) * a.pad) * k, (b[1] - (b[3] - b[1]) * a.pad) * k,
                      (b[2] + (b[2] - b[0]) * a.pad) * k, (b[3] + (b[3] - b[1]) * a.pad) * k] for b in boxes]
        if a.box == "blender_iso":            # each character's box on its own render (no other person in it)
            dets, poses = [], []
            for n2, b in enumerate(bframe):
                if not b:
                    continue
                iso_path = p / "iso" / str(n2) / frames[n].name
                iso = cv2.imread(str(iso_path))
                if iso is None or iso.shape[:2] != full.shape[:2]:
                    sys.exit(f"missing or wrong-size isolated render {iso_path} (bench --isolate)")
                iso = iso[:, :, ::-1]
                if k != 1.0:
                    iso = cv2.resize(iso, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
                bb = [[(b[0] - (b[2] - b[0]) * a.pad) * k, (b[1] - (b[3] - b[1]) * a.pad) * k,
                       (b[2] + (b[2] - b[0]) * a.pad) * k, (b[3] + (b[3] - b[1]) * a.pad) * k]]
                d1, p1 = run(np.ascontiguousarray(iso), bb)
                dets += d1
                poses += p1["people"] if a.detector == "sdpose" else p1
        else:
            dets, poses = run(img, boxes, a.box == "yolox_bgr")
            if a.detector == "sdpose":
                poses = poses["people"]
        if a.control and a.detector == "sdpose":   # the control picture as SDPose Draw Keypoints makes it
            canvas = sdp.draw({"canvas_width": img.shape[1], "canvas_height": img.shape[0], "people": poses})
        if a.control and a.detector == "dwpose":   # the control picture as the DWPose node draws it
            from custom_controlnet_aux.dwpose import draw_poses
            canvas = draw_poses(poses, img.shape[0], img.shape[1], draw_body=True, draw_hand=True, draw_face=True)
        if a.control:
            Path(a.control).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(a.control) / f"{frames[n].stem}.png"), canvas[:, :, ::-1])
        for d in dets:                        # back to the pass size, where the true points are
            for s in ("body", "L", "R", "face"):
                if d[s] is not None:
                    d[s][:, :2] /= k
        truths = truth(kp, i, bframe, az_chars)
        if a.box in ("blender", "blender_iso"):   # the pose of box n belongs to character n (no guessing)
            assert len(dets) == len(boxes), (len(dets), len(boxes))
            it = iter(range(len(dets)))
            m = [next(it) if bframe[n2] else -1 for n2 in range(len(truths))]
        else:
            m = match(truths, dets)
        per_frame[fnum] = {"people": len(dets), "expected": sum(1 for t in truths if t["box"] or not boxes_json),
                           "side": side(az[i]),
                           "yolox_person": sdp.person_score(img) if a.detector == "sdpose" else yolox_person(det, img)}
        for t, j in zip(truths, m):
            if a.save_dets:
                saved.setdefault(str(fnum), {})[t["name"]] = (
                    {k: (None if v is None else np.where(np.isnan(v), None, v).tolist()) for k, v in dets[j].items()}
                    if j >= 0 else None)
            r = score_person(t, dets[j] if j >= 0 else None)
            caz = t["az"] if t["az"] is not None else az[i]
            r.update(frame=fnum, az=caz, side=side(caz), name=t["name"])
            rows.append(r)
        if a.overlay:
            draw(full, truths, dets, Path(a.overlay) / f"{frames[n].stem}.png")
    result = {"all": summary(rows, per_frame)}
    for s in ("front", "left", "back", "right"):       # per character side (its own camera angle)
        sel = [r for r in rows if r["side"] == s]
        result[s] = summary(sel, {f: per_frame[f] for f in sorted({r["frame"] for r in sel})})
    if a.save_dets:
        Path(a.save_dets).write_text(json.dumps({"detector": a.detector, "box": a.box, "frames": saved}))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": result, "rows": rows, "frames": per_frame,
                                           "settings": vars(a)}, indent=1, default=float))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
