"""Integration check: video and image in/out through comfy-sdk -> comfy-api-proxy -> ComfyUI.

No GPU work: LoadVideo -> GetVideoComponents -> CreateVideo -> SaveVideo, and LoadImage -> SaveImage.
Needs a running ComfyUI and comfy-api-proxy (COMFY_BASE_URL, default http://127.0.0.1:8189).

    python tests/integration/sdk_video_check.py <control_video.mp4> <image.png> [--frames 121]

Prints one JSON report; exit code 0 only when every check passes.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import cv2
import imageio_ffmpeg
from comfy_sdk import Comfy

os.environ.setdefault("COMFY_BASE_URL", "http://127.0.0.1:8189")


def probe(path):
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.read()[0]:
        n += 1
    info = {"frames": n, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "fps": round(cap.get(cv2.CAP_PROP_FPS), 3)}
    cap.release()
    return info


def make_slice(src, dst, frames):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", src,
                    "-frames:v", str(frames), "-c:v", "libx264rgb", "-crf", "0", "-preset", "veryfast", dst], check=True)


def graph(prefix):
    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": "placeholder.mp4"}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "CreateVideo", "inputs": {"images": ["2", 0], "fps": 24.0}},
        "4": {"class_type": "SaveVideo", "inputs": {"video": ["3", 0], "filename_prefix": f"{prefix}/video",
                                                    "format": "auto", "format.codec": "auto", "codec": "auto"}},
        "5": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": f"{prefix}/image"}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("image")
    ap.add_argument("--frames", type=int, default=121)
    a = ap.parse_args()
    report, ok = {"base_url": os.environ["COMFY_BASE_URL"]}, True
    tmp = tempfile.mkdtemp(prefix="cinestudio-check-")
    clip = os.path.join(tmp, "control.mp4")
    make_slice(a.video, clip, a.frames)
    report["input"] = probe(clip) | {"bytes": os.path.getsize(clip)}

    client = Comfy()
    t0 = time.time()
    v_asset = client.assets.from_file(clip)
    i_asset = client.assets.from_file(a.image)
    report["upload"] = {"video_asset": v_asset.id, "image_asset": i_asset.id, "seconds": round(time.time() - t0, 2)}

    wf = client.workflows.from_json(graph("cinestudio_check"))
    wf.set_input("1", "file", v_asset)
    wf.set_input("5", "image", i_asset)
    key = f"cinestudio-check-{int(time.time())}"
    job = client.submit(wf, idempotency_key=key)
    report["job"] = {"id": job.id, "status_after_submit": job.status}
    events = []
    for ev in job.events():
        events.append(type(ev).__name__)
        if len(events) > 500:
            break
    job = job.refresh()
    report["job"] |= {"status": job.status, "events": sorted(set(events)), "n_events": len(events),
                      "seconds": round(time.time() - t0, 2)}

    outs = {}
    for node in ("4", "6"):
        items = job.get_outputs(node)
        outs[node] = [{"name": o.name, "type": o.type, "content_type": o.content_type} for o in items]
        if items:
            dst = os.path.join(tmp, f"out_{node}_{items[0].name}")
            items[0].to_file(dst)
            outs[node][0]["local"] = dst
            if node == "4":
                outs[node][0]["probe"] = probe(dst)
    report["outputs"] = outs
    vid = (outs.get("4") or [{}])[0].get("probe", {})
    checks = {
        "video_frames": vid.get("frames") == a.frames,
        "video_size": (vid.get("width"), vid.get("height")) == (report["input"]["width"], report["input"]["height"]),
        "video_fps": abs((vid.get("fps") or 0) - 24) < 0.01,
        "image_out": bool(outs.get("6")),
        "job_completed": job.status in ("completed", "succeeded", "success"),
    }

    # same idempotency key again: must not create a second job
    try:
        again = client.submit(wf, idempotency_key=key)
        report["idempotency_resend"] = {"returned_job": again.id, "same_job": again.id == job.id}
    except Exception as e:  # noqa: BLE001 - we want the exact type for the report
        report["idempotency_resend"] = {"raised": type(e).__name__, "message": str(e)[:300]}

    # forced failure: an image file that does not exist
    bad = client.workflows.from_json({"1": {"class_type": "LoadImage", "inputs": {"image": "does_not_exist_cinestudio.png"}},
                                      "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}}})
    try:
        bj = client.submit(bad)
        bj = bj.wait(timeout=60)
        report["forced_failure"] = {"status": bj.status, "error": str(bj.error)[:300]}
    except Exception as e:  # noqa: BLE001
        report["forced_failure"] = {"raised": type(e).__name__, "message": str(e)[:300]}

    report["checks"] = checks
    ok = all(checks.values())
    report["ok"] = ok
    print(json.dumps(report, indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
