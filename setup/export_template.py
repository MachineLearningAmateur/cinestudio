"""Copy a finished template from the user's library into the built-in library that ships with the repository.

    python setup/export_template.py NAME [--order N] [--description "..."] [--background BUILTIN_BACKGROUND] [--force]

templates/library/NAME/ gets shot.blend (frozen by Blender: images, sounds and fonts PACKED, so the song travels with it),
preview.mp4 (the Workbench preview with sound) and shot.json with file paths RELATIVE to the folder and no personal
paths (no source path, pid, logs). Only Blender templates (kind "blender" or "dance"). Needs Blender (no GPU, no EEVEE).
The template rules run first (`shot check NAME`, cinestudio/shot_check.py): an ERROR refuses the export unless
`--force`; warnings are printed. The exported .blend is checked again (its errors refuse too, unless `--force`) and
its summary stored in shot.json "check".
`--background NAME`: the template's default background (shot.json "background"), a BUILT-IN background
(backgrounds/library/NAME: a new user must have it too); the Video tab preselects it, `run new` without --background uses it.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cinestudio import runner, shot_check, shots  # noqa: E402

KEEP = ("name", "kind", "description", "seconds", "frames", "plan", "loop", "song_start", "night", "added", "ended",
        "layers", "clip_texts", "clip_context")       # clip texts: one prompt per clip (AGENTS.md 7.2)


def check_before_export(name: str, force: bool = False, checker=None) -> dict:
    """The template rules on the user's template: SystemExit on errors unless `force` (the errors are listed)."""
    try:
        res = (checker or shot_check.check)(name)
    except Exception as exc:  # noqa: BLE001 - Blender missing or the scan failed: unknown is not "no errors"
        if force:
            print(json.dumps({"event": "check", "text": f"the template check could not run: {exc}"}))
            return {}
        raise SystemExit(f"{name}: the template check could not run ({exc}); fix it or export with --force")
    errs = [c for c in res["checks"] if c["severity"] == "error"]
    if errs and not force:
        lines = "; ".join(f"{c['id']}: {c['message']} (fix: {c['fix']})" for c in errs)
        raise SystemExit(f"{name}: {len(errs)} template check error{'s' if len(errs) > 1 else ''} (shot check {name}); "
                         f"fix them or export with --force. {lines}")
    return res


def check_shipped(blend: Path, rec: dict, name: str, force: bool = False) -> dict | None:
    """The check of the exported .blend itself: SystemExit when it has errors (an incomplete check included) or cannot
    run, unless `force` (then None when it could not run)."""
    try:
        res = shot_check.check_blend(blend, rec, name)
    except Exception as exc:  # noqa: BLE001 - reported, and refused unless forced
        if force:
            print(json.dumps({"event": "check", "text": f"the check of the exported .blend could not run: {exc}"}))
            return None
        raise SystemExit(f"{name}: the check of the exported .blend could not run ({exc}); fix it or export with "
                         "--force")
    errs = [c for c in res["checks"] if c["severity"] == "error"]
    if errs and not force:
        lines = "; ".join(f"{c['id']}: {c['message']}" for c in errs[:6])
        raise SystemExit(f"{name}: the exported .blend has {len(errs)} template check error"
                         f"{'s' if len(errs) > 1 else ''}; fix them or export with --force. {lines}")
    return res


def export(name: str, order: int = 99, description: str | None = None, background: str | None = None,
           force: bool = False, checker=None) -> Path:
    rec = shots.load(name)
    if background is not None:
        from cinestudio import background as bgs
        if not bgs.has_builtin(background):
            raise SystemExit(f"{background}: the default background of a built-in template must be a built-in "
                             f"background (backgrounds/library/{background})")
    if rec.get("builtin"):
        raise SystemExit(f"{name} is already a built-in template")
    if rec.get("kind") not in ("blender", "dance") or rec.get("status") != "done":
        raise SystemExit(f"{name}: only a finished Blender template can be built in (kind {rec.get('kind')}, "
                         f"status {rec.get('status')})")
    check_before_export(name, force, checker)
    src = shots.shot_dir(name)
    dst = shots.builtin_dir() / name
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    runner.snapshot_blend(src / "shot.blend", tmp / "shot.blend", tmp / "snapshot.json")   # sounds packed
    (tmp / "snapshot.json").unlink(missing_ok=True)
    (tmp / "snapshot.log").unlink(missing_ok=True)
    files = {"blend": "shot.blend"}
    prev = (rec.get("files") or {}).get("preview")
    if prev and Path(prev).is_file():
        shutil.copy2(prev, tmp / "preview.mp4")
        files["preview"] = "preview.mp4"
    out = {k: rec[k] for k in KEEP if k in rec}
    prm = rec.get("params") or {}
    out["params"] = {k: prm[k] for k in ("song", "routines", "loop", "seconds", "night", "start") if k in prm}
    if description is not None:
        out["description"] = description
    out.update(name=name, order=order, files=files, builtin=True)
    if background is not None:
        out["background"] = background
    try:                                                   # the SHIPPED (packed) .blend is checked too, with the
        shipped = check_shipped(tmp / "shot.blend", out, name, force)   # same rule: errors refuse unless --force
    except BaseException:                                  # (Codex review 2026-10-25: only printed before)
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if shipped:
        out["check"] = shot_check.summary(shipped)
    (tmp / "shot.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--order", type=int, default=99, help="place in the list (small = first)")
    ap.add_argument("--description", default=None)
    ap.add_argument("--background", default=None, help="its default background: a built-in background's name")
    ap.add_argument("--force", action="store_true", help="export although `shot check NAME` finds errors")
    a = ap.parse_args()
    d = export(a.name, a.order, a.description, a.background, a.force)
    print(json.dumps({"ok": True, "built_in": str(d), "files": sorted(p.name for p in d.iterdir())}))
