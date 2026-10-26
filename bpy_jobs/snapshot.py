"""Freeze a .blend for a run (run inside Blender, headless).

    blender -b original.blend --python bpy_jobs/snapshot.py -- --out <copy.blend> --report <report.json>

The copy must render the same later, even if files next to the original change or move:
  - images, sounds and fonts that live in external files are PACKED into the copy;
  - linked libraries, movie clips, external caches (Alembic/USD/point caches, volumes) cannot be
    packed: they are listed as "unsupported" and the snapshot fails (exit code 2) unless
    --allow-external is given (then they stay external, as absolute paths, and are listed);
  - missing files are listed and fail the snapshot the same way.
The original file is never changed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import bpy


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--allow-external", action="store_true")
    return ap.parse_args(argv)


def external(datablocks, kind):
    out = []
    for d in datablocks:
        if getattr(d, "library", None) is not None:
            continue                       # comes with a linked library (reported there)
        path = getattr(d, "filepath", "") or ""
        if not path or getattr(d, "packed_file", None) is not None:
            continue
        if kind == "image" and d.source not in {"FILE", "SEQUENCE", "MOVIE", "TILED"}:
            continue
        absp = bpy.path.abspath(path)
        out.append({"kind": kind, "name": d.name, "path": absp, "exists": os.path.exists(absp)})
    return out


def main():
    a = args()
    report = {"original": bpy.data.filepath, "packed": [], "missing": [], "unsupported": []}
    sc = bpy.context.scene                 # the shot's frame range: the run knows its clips before the passes
    report.update(frame_start=int(sc.frame_start), frame_end=int(sc.frame_end), fps=sc.render.fps / sc.render.fps_base)
    for lib in bpy.data.libraries:
        report["unsupported"].append({"kind": "linked library", "name": lib.name, "path": bpy.path.abspath(lib.filepath)})
    for clip in bpy.data.movieclips:
        report["unsupported"].append({"kind": "movie clip", "name": clip.name, "path": bpy.path.abspath(clip.filepath)})
    for cache in getattr(bpy.data, "cache_files", []):
        report["unsupported"].append({"kind": "cache file", "name": cache.name, "path": bpy.path.abspath(cache.filepath)})
    for vol in getattr(bpy.data, "volumes", []):
        if vol.filepath and vol.packed_file is None:
            report["unsupported"].append({"kind": "volume", "name": vol.name, "path": bpy.path.abspath(vol.filepath)})
    for ob in bpy.data.objects:
        for mod in ob.modifiers:
            cache = getattr(mod, "point_cache", None)
            if cache is not None and getattr(cache, "use_external", False):
                report["unsupported"].append({"kind": "external point cache", "name": f"{ob.name}/{mod.name}",
                                              "path": bpy.path.abspath(cache.filepath)})
    files = external(bpy.data.images, "image") + external(bpy.data.sounds, "sound") + \
        [f for f in external(bpy.data.fonts, "font") if f["name"] != "Bfont Regular"]
    for f in files:
        (report["packed"] if f["exists"] else report["missing"]).append(f)
    # image sequences and movie-file images cannot be packed into a .blend
    seq = [f for f in report["packed"] if f["kind"] == "image" and bpy.data.images[f["name"]].source in {"SEQUENCE", "MOVIE", "TILED"}]
    for f in seq:
        report["packed"].remove(f)
        report["unsupported"].append({**f, "kind": "image sequence/movie/UDIM"})
    blocked = report["missing"] or report["unsupported"]
    if blocked and not a.allow_external:
        report["ok"] = False
        with open(a.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print("SNAPSHOT BLOCKED", json.dumps({k: len(v) for k, v in report.items() if isinstance(v, list)}))
        sys.exit(2)
    pools = {"image": bpy.data.images, "sound": bpy.data.sounds, "font": bpy.data.fonts}
    for f in report["packed"]:              # one by one: pack_all stops at the first missing file
        pools[f["kind"]][f["name"]].pack()
    if blocked:                                 # --allow-external: keep them, but as absolute paths
        bpy.ops.file.make_paths_absolute()
    bpy.ops.wm.save_as_mainfile(filepath=a.out, copy=True, relative_remap=False, compress=False)
    report["ok"] = True
    with open(a.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("SNAPSHOT OK", json.dumps({k: len(v) for k, v in report.items() if isinstance(v, list)}))


if __name__ == "__main__":
    main()
