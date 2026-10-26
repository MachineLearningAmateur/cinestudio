"""Step 2 of the motion follow score: how well a generated video follows the reference poses (protocol version 2).

    python -m cinestudio.follow_eval RUN_DIR/follow_dets.json [--out RUN_DIR/follow.json]

The joints come from cinestudio/motion_follow.py (DWPose on the video inside each character's Blender box). The
reference is the Blender rig ("rig": what the shot asks for); a joint is EXPECTED only when it is in the frame and
observable (not hidden by another character; face joints only when the face rule shows them). When the dets also
carry the drawn control ("ctrl": what the model was told), every score is also given against it.

Main score - BONE DIRECTION (proportion-free; a big head or short legs do not matter):
  For each directed bone (shoulder line, hip line, upper arms, forearms, thighs, shins) that is expected, the angle e
  between its 2D direction in the video and in the reference; score g = max(0, 1 - e / 45 deg). A missing or collapsed
  bone scores 0, and the denominator is fixed by the reference (hiding a hard limb cannot raise the score). A
  reference bone shorter than 4% of the box height (pointing at the camera) is not expected. Reported: the score
  (mean over expected bones), coverage (bones measured / expected), the median angle of the measured bones, per body
  part (torso / arms / legs) and the time spans where the frame score stays below 0.5. A MIRRORED character shows in
  the torso part (the shoulder and hip lines reverse) and in lr_disagree; limbs that hang straight down keep their
  direction in a mirror, so the total alone does not reveal it.
Other outputs (diagnostics, not the headline):
  pck05/10/20   joints within 5/10/20% of the box height (an LSP-style threshold; normalised by the box HEIGHT, not
                the standard protocol). Penalises other proportions.
  st_pck20      PCK@0.2 after a scale + translation fit on the torso (no rotation, no mirror): body shape only; it
                hides framing and position errors.
  ap/ap50/ap75  official pycocotools OKS AP with the given person correspondence (each character-frame is its own
                image: not the standard COCO multi-person protocol); 0 when nothing is detected.
  lr_disagree   left/right correspondence fits better swapped (a mirrored character OR a detector label error);
  facing        face shown vs the rig's face rule. Both count only where there is evidence; "unknown" rows are
                counted apart.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

# OpenPose-18 index for each COCO-17 joint (COCO order: nose, l_eye, r_eye, l_ear, r_ear, l_sho, r_sho, l_elb, r_elb,
# l_wri, r_wri, l_hip, r_hip, l_knee, r_knee, l_ank, r_ank)
COCO_FROM_OP = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
LR = {2: 5, 3: 6, 4: 7, 8: 11, 9: 12, 10: 13, 14: 15, 16: 17}
LR |= {b: a for a, b in LR.items()}
BONES = {"torso": [(2, 5), (8, 11)],                                  # shoulder line, hip line (right -> left)
         "arms": [(2, 3), (3, 4), (5, 6), (6, 7)],
         "legs": [(8, 9), (9, 10), (11, 12), (12, 13)]}
TORSO = (1, 2, 5, 8, 11)
MAX_DEG = 45.0
MIN_BONE = 0.04          # a reference bone shorter than this share of the box height is not expected
FAR = -1e4               # where a missing predicted joint goes for COCO (it never matches by chance)


def _pt(p):
    """(x, y) of a point or None (missing, not finite)."""
    if p is None or len(p) < 2 or p[0] is None or p[1] is None or not np.isfinite([p[0], p[1]]).all():
        return None
    return np.array(p[:2], float)


def expected(ref, j):
    return ref[j] is not None and len(ref[j]) > 2 and bool(ref[j][2])


def _height(d):
    return max(d["box"][3] - d["box"][1], 1.0)


# ---------------------------------------------------------------- bone direction
def bone_scores(d, ref_key="rig"):
    """[(part, score or None if not expected, angle or None)] for one character-frame."""
    ref, det, h = d[ref_key], d["det"], _height(d)
    out = []
    for part, bones in BONES.items():
        for a, b in bones:
            if not (expected(ref, a) and expected(ref, b)):
                out.append((part, None, None))
                continue
            r = np.array(ref[b][:2], float) - np.array(ref[a][:2], float)
            if np.linalg.norm(r) < MIN_BONE * h:     # foreshortened: the direction is not defined well enough
                out.append((part, None, None))
                continue
            pa, pb = _pt(det[a]), _pt(det[b])
            if pa is None or pb is None or np.linalg.norm(pb - pa) < 1e-6:
                out.append((part, 0.0, None))         # expected but missing / collapsed: no credit
                continue
            v = pb - pa
            e = np.degrees(np.arctan2(abs(r[0] * v[1] - r[1] * v[0]), float(np.dot(r, v))))
            out.append((part, max(0.0, 1.0 - e / MAX_DEG), float(e)))
    return out


# ---------------------------------------------------------------- PCK and the torso-fit diagnostic
def pck(d, thr=0.05, ref_key="rig", det=None):
    ref, h = d[ref_key], _height(d)
    det = d["det"] if det is None else det
    shown = [j for j in range(18) if expected(ref, j)]
    if not shown:
        return None
    ok = sum(1 for j in shown if _pt(det[j]) is not None and np.linalg.norm(_pt(det[j]) - np.array(ref[j][:2])) < thr * h)
    return ok / len(shown)


def st_fit(d, ref_key="rig"):
    """The detection moved and scaled (no rotation, no mirror) to fit the reference torso; None with < 3 points."""
    ref, det = d[ref_key], d["det"]
    js = [j for j in TORSO if expected(ref, j) and _pt(det[j]) is not None]
    if len(js) < 3:
        return None
    P = np.array([_pt(det[j]) for j in js])
    Q = np.array([ref[j][:2] for j in js], float)
    pc, qc = P.mean(0), Q.mean(0)
    sp = np.sqrt(((P - pc) ** 2).sum())
    if sp < 1e-6:
        return None
    s = np.sqrt(((Q - qc) ** 2).sum()) / sp
    return [None if _pt(p) is None else list((_pt(p) - pc) * s + qc) for p in det]


# ---------------------------------------------------------------- COCO AP (diagnostic)
def _coco(dets, ref_key="rig"):
    images, anns, results = [], [], []
    for n, d in enumerate(dets):
        x0, y0, x1, y1 = d["box"]
        images.append({"id": n})
        gt = []
        for j in COCO_FROM_OP:
            x, y = d[ref_key][j][:2]
            gt += [x, y, 2 if expected(d[ref_key], j) else 0]
        anns.append({"id": n + 1, "image_id": n, "category_id": 1, "iscrowd": 0, "bbox": [x0, y0, x1 - x0, y1 - y0],
                     "area": (x1 - x0) * (y1 - y0), "keypoints": gt, "num_keypoints": sum(1 for v in gt[2::3] if v)})
        pk, scores = [], []
        for j in COCO_FROM_OP:
            p = _pt(d["det"][j])
            pk += [float(p[0]), float(p[1]), 1] if p is not None else [FAR, FAR, 0]
            if p is not None:
                sc = d["det"][j][2] if len(d["det"][j]) > 2 and d["det"][j][2] is not None else 1.0
                scores.append(float(sc))
        if scores:
            results.append({"image_id": n, "category_id": 1, "keypoints": pk, "score": float(np.mean(scores))})
    return images, anns, results


def coco_ap(dets, ref_key="rig") -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images, anns, results = _coco(dets, ref_key)
    if not any(a["num_keypoints"] for a in anns):
        return {"ap": None, "ap50": None, "ap75": None}          # nothing to evaluate
    if not results:
        return {"ap": 0.0, "ap50": 0.0, "ap75": 0.0}             # expected people, nothing detected
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, "keypoints")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {"ap": round(float(ev.stats[0]), 3), "ap50": round(float(ev.stats[1]), 3), "ap75": round(float(ev.stats[2]), 3)}


# ---------------------------------------------------------------- left/right and facing (with "unknown")
def lr_disagree(d, ref_key="rig"):
    """True / False, or None when there is no evidence (fewer than 2 limb pairs on both sides)."""
    ref, det = d[ref_key], d["det"]
    st = sw = 0.0
    n = 0
    for a in (2, 3, 4, 8, 9, 10):
        b = LR[a]
        pa, pb = _pt(det[a]), _pt(det[b])
        if pa is not None and pb is not None and expected(ref, a) and expected(ref, b):
            ra, rb = np.array(ref[a][:2], float), np.array(ref[b][:2], float)
            n += 1
            st += np.linalg.norm(pa - ra) + np.linalg.norm(pb - rb)
            sw += np.linalg.norm(pa - rb) + np.linalg.norm(pb - ra)
    if n < 2:
        return None
    return bool(sw < 0.7 * st)


def facing_wrong(d, ref_key="rig"):
    """True / False, or None without evidence (the body was not found in the box)."""
    det = d["det"]
    if sum(1 for p in det if _pt(p) is not None) < 4:
        return None
    ref_face = expected(d[ref_key], 0)
    det_face = _pt(det[0]) is not None and (_pt(det[14]) is not None or _pt(det[15]) is not None)
    return ref_face != det_face


# ---------------------------------------------------------------- the report
def fail_spans(frame_scores, below=0.5, min_len=6):
    """[(first, last)] frames where the score stays below `below` for at least `min_len` frames."""
    spans, start, last = [], None, None
    for f, s in sorted(frame_scores.items()):
        if s is not None and s < below:
            start = f if start is None else start
            last = f
        elif start is not None:
            if last - start + 1 >= min_len:
                spans.append((start, last))
            start = None
    if start is not None and last - start + 1 >= min_len:
        spans.append((start, last))
    return spans


def evaluate(dets, ref_key="rig") -> dict:
    def block(sel):
        if not sel:
            return {}
        allb = [b for d in sel for b in bone_scores(d, ref_key)]
        exp = [b for b in allb if b[1] is not None]
        meas = [b for b in exp if b[2] is not None]
        per_part = {}
        for part in BONES:
            pe = [b for b in exp if b[0] == part]
            per_part[part] = round(float(np.mean([b[1] for b in pe])), 3) if pe else None
        frames = {}
        for d in sel:
            frames.setdefault(d["frame"], []).extend(b[1] for b in bone_scores(d, ref_key) if b[1] is not None)
        frame_scores = {f: (float(np.mean(v)) if v else None) for f, v in frames.items()}

        def m(thr):
            p = [x for x in (pck(d, thr, ref_key) for d in sel) if x is not None]
            return round(float(np.mean(p)), 3) if p else None
        stp = []
        for d in sel:
            fit = st_fit(d, ref_key)
            if fit is not None:
                v = pck(d, 0.2, ref_key, fit)
                if v is not None:
                    stp.append(v)
        lr = [lr_disagree(d, ref_key) for d in sel]
        fw = [facing_wrong(d, ref_key) for d in sel]
        return {"bone": round(float(np.mean([b[1] for b in exp])), 3) if exp else None,
                "bone_coverage": round(len(meas) / len(exp), 3) if exp else None,
                "bone_median_deg": round(float(np.median([b[2] for b in meas])), 1) if meas else None,
                "bone_parts": per_part, "fail_spans": fail_spans(frame_scores),
                "pck05": m(0.05), "pck10": m(0.10), "pck20": m(0.20),
                "st_pck20": round(float(np.mean(stp)), 3) if stp else None, **coco_ap(sel, ref_key),
                "lr_disagree": sum(1 for x in lr if x), "lr_unknown": sum(1 for x in lr if x is None),
                "facing": sum(1 for x in fw if x), "facing_unknown": sum(1 for x in fw if x is None), "rows": len(sel)}
    names = sorted({d["name"] for d in dets})
    return {"all": block(dets), "per_character": {n: block([d for d in dets if d["name"] == n]) for n in names}}


def swapped(d):
    """The first version's name for lr_disagree (False without evidence)."""
    return bool(lr_disagree(d))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dets")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    src = Path(a.dets)
    data = json.loads(src.read_text())
    res = {"protocol": 2, "vs_rig": evaluate(data["dets"], "rig"), "frames": data.get("frames")}
    if data["dets"] and all("ctrl" in d for d in data["dets"]):
        res["vs_control"] = evaluate(data["dets"], "ctrl")
    (Path(a.out) if a.out else src.with_name("follow.json")).write_text(json.dumps(res, indent=1))
    print(json.dumps({"ok": True, **res}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
