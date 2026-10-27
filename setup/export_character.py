"""Export a user character as a BUILT-IN character (characters/library/<id>/, AGENTS.md 4.1).

    python setup/export_character.py NAME [--display-name TEXT] [--note TEXT] [--force]

Copies only what a run needs: the record's `files` (front, views, face, sheet) and a proxy kit's head asset
(proxy.head.asset.file -> ./head_asset.blend); writes character.json with ./ paths and without the private working
history (candidates, edits, checks, copied_from, the repose shot's path). The user's own folder in <home>/characters is
NOT changed: while it exists it hides the built-in (a user character of the same ID wins), so move it away after a look
(the caller's choice). Refuses a character that is not ready, and an existing built-in folder unless --force.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DROP = {"candidates", "edits", "checks", "copied_from", "front_posed", "front_size", "builtin"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--display-name", default=None)
    ap.add_argument("--note", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    from cinestudio import character
    src = character.library() / a.name
    if not (src / "character.json").is_file():
        raise SystemExit(f"export_character: no user character {a.name!r} in {character.library()}")
    c = json.loads((src / "character.json").read_text(encoding="utf-8"))
    if c.get("status") != "ready":
        raise SystemExit(f"export_character: {a.name!r} is not ready ({c.get('status')})")
    out = character.BUILTIN / a.name
    if out.exists() and not a.force:
        raise SystemExit(f"export_character: {out} exists (use --force)")
    tmp = out.with_name(out.name + ".export-tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "views").mkdir(parents=True)
    rec = {k: v for k, v in c.items() if k not in DROP}
    files = {}
    for key, f in (c.get("files") or {}).items():
        p = Path(f)
        if not p.is_file():
            continue
        rel = p.name if p.parent == src else f"views/{p.name}" if p.parent.name == "views" else p.name
        shutil.copy2(p, tmp / rel)
        files[key] = "./" + rel
    rec["files"] = files
    proxy = json.loads(json.dumps(c.get("proxy") or {}))
    asset = ((proxy.get("head") or {}).get("asset"))
    if asset:
        f = Path(asset["file"] if isinstance(asset, dict) else asset)
        shutil.copy2(f, tmp / "head_asset.blend")
        if isinstance(asset, dict):
            asset["file"] = "./head_asset.blend"
        else:
            proxy["head"]["asset"] = "./head_asset.blend"
    rec["proxy"] = proxy
    if isinstance(rec.get("reposed"), dict):
        rec["reposed"] = {k: v for k, v in rec["reposed"].items() if k != "shot"}      # a personal path
    if a.display_name:
        rec["display_name"] = a.display_name
    if a.note:
        rec["builtin_note"] = a.note
    left = [s for s in json.dumps(rec).split('"') if ":\\\\" in s or ":/" in s]
    if left:
        raise SystemExit(f"export_character: personal paths left in the record: {left[:3]}")
    (tmp / "character.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if not any((tmp / "views").iterdir()):
        (tmp / "views").rmdir()
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    print(json.dumps({"ok": True, "name": a.name, "dir": str(out), "files": sorted(files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
