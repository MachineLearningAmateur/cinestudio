"""The 360 depth of a panorama, for the Blender world (bpy_jobs/pano_world.py --ground depth).

Research 2026-10-22 (AGENTS.md "Background control"): the one-radius dome gave a wrong parallax (up to ~12 degrees with
the camera 4 m off centre) and laid things below the horizon flat. MoGe-2 (Microsoft, MIT; ComfyUI's native
MoGePanoramaInference, graph workflows/moge_pano) estimates the distance in every direction of the panorama. ComfyUI
gives it as a mesh (GLB): each vertex is direction * distance, and its UV is the panorama pixel it came from
(u = column / (W - 1), v = row / (H - 1), row 0 = the top; comfy/ldm/moge/geometry.py triangulate_grid_mesh). Only the
DISTANCE (the vertex length) is read here, so the axis conventions of the mesh do not matter: pano_world.py places each
pixel in Blender's own panorama convention.

The distances are RELATIVE (MoGe's panorama mode has no metric scale); pano_world.py scales them with the eye height.
The sky has no geometry: the node puts it at 5x the 95th percentile of the rest ("far").
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

COMPONENT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


class DepthError(RuntimeError):
    pass


def read_glb(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(positions (N, 3), uvs (N, 2)) of the first primitive of a binary glTF."""
    data = Path(path).read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise DepthError(f"{path} is not a binary glTF file")
    off, js, binc = 12, None, None
    while off + 8 <= len(data):
        ln, kind = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + ln]
        if kind == 0x4E4F534A:                    # JSON
            js = json.loads(chunk)
        elif kind == 0x004E4942:                  # BIN
            binc = chunk
        off += 8 + ln
    if js is None or binc is None:
        raise DepthError(f"{path}: no JSON or BIN chunk")
    attrs = js["meshes"][0]["primitives"][0]["attributes"]

    def accessor(i):
        a = js["accessors"][i]
        v = js["bufferViews"][a["bufferView"]]
        dt = np.dtype(COMPONENT[a["componentType"]])
        n, w = a["count"], WIDTH[a["type"]]
        start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = v.get("byteStride") or dt.itemsize * w
        if stride == dt.itemsize * w:
            return np.frombuffer(binc, dt, n * w, start).reshape(n, w).astype(np.float64)
        raw = np.frombuffer(binc, np.uint8, stride * (n - 1) + dt.itemsize * w, start)
        return np.stack([np.frombuffer(raw[k * stride: k * stride + dt.itemsize * w].tobytes(), dt)
                         for k in range(n)]).astype(np.float64)
    if "POSITION" not in attrs or "TEXCOORD_0" not in attrs:
        raise DepthError(f"{path}: the mesh has no positions or no texture coordinates")
    return accessor(attrs["POSITION"]), accessor(attrs["TEXCOORD_0"])


def distance_map(glb: str | Path) -> np.ndarray:
    """The distance of every panorama pixel of the (decimated) grid, row 0 = the top; NaN where the mesh has no vertex."""
    pos, uv = read_glb(glb)
    cols = len(np.unique(np.round(uv[:, 0] * 1e5)))      # u = column / (W - 1): one value per grid column
    rows = len(np.unique(np.round(uv[:, 1] * 1e5)))
    if cols < 8 or rows < 4:
        raise DepthError(f"{glb}: a grid of {cols}x{rows} is too small")
    d = np.full((rows, cols), np.nan, np.float32)
    c = np.clip(np.round(uv[:, 0] * (cols - 1)).astype(int), 0, cols - 1)
    r = np.clip(np.round(uv[:, 1] * (rows - 1)).astype(int), 0, rows - 1)
    d[r, c] = np.linalg.norm(pos, axis=1)
    top, bottom = np.nanmedian(d[: max(1, rows // 10)]), np.nanmedian(d[-max(1, rows // 10):])
    if not top > bottom:                                  # the sky (far) is at the top, the floor (near) at the bottom
        raise DepthError(f"{glb}: the top of the panorama is nearer ({top:.3g}) than its bottom ({bottom:.3g}): "
                         "not the expected layout")
    return d


def clean_sky(d: np.ndarray, pano: str | Path, grad_max: float = 12.0) -> np.ndarray:
    """NOT USED in the pipeline (Codex review 2026-10-22: smoothness + a link to the top edge also takes a plain facade
    or a ceiling; and the two-layer ground needs no clean sky: the near layer ignores everything beyond the dome wall).
    Kept as a tool. MoGe can give parts of a plain sky a finite distance (2026-10-22, rooftop: the sky at the sides about 1.4x the
    skyline's distance: a curved wall just behind the city). Smooth, untextured pixels above the horizon that are
    connected to the top edge are sky: they get the far value (pano_world.py then leaves them to the world sky at
    infinity). Textured things (tanks, domes, skyline, clouds with detail) keep their depth."""
    import cv2
    img = cv2.imread(str(pano), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return d
    R, C = d.shape
    g = cv2.resize(img, (C, R), interpolation=cv2.INTER_AREA).astype(np.float32)
    pad = 16                                              # the panorama wraps round: no false edge at the seam
    gw = np.concatenate([g[:, -pad:], g, g[:, :pad]], axis=1)
    grad = cv2.GaussianBlur(np.hypot(cv2.Sobel(gw, cv2.CV_32F, 1, 0), cv2.Sobel(gw, cv2.CV_32F, 0, 1)),
                            (0, 0), 1.5)[:, pad:pad + C]
    above = (np.arange(R) < int(R * 0.49))[:, None] & np.ones((1, C), bool)
    smooth = (above & (grad < grad_max)).astype(np.uint8)
    # a real sky is FARTHER than what stands at the horizon; a plain ceiling (an indoor panorama) is nearer: kept
    hz = d[max(0, int(R * (0.5 - 3.0 / 180))): int(R * (0.5 + 3.0 / 180)) + 1]
    horizon = float(np.nanmedian(hz)) if np.isfinite(hz).any() else 0.0
    smooth &= (np.nan_to_num(d, nan=np.inf) >= horizon).astype(np.uint8)
    n, lab = cv2.connectedComponents(smooth, connectivity=4)
    top = set(np.unique(lab[0])) - {0}
    sky = np.isin(lab, list(top))
    far = float(np.nanmax(d))
    out = d.copy()
    out[sky] = far
    # islands: things above the horizon stand on something (tanks, domes, poles, the skyline all touch the horizon);
    # a finite patch in the sky that does not reach the horizon band (the last ~3 degrees above it) is a cloud: far too
    solid = (above & (out < 0.97 * far)).astype(np.uint8)
    n, lab = cv2.connectedComponents(solid, connectivity=8)
    last = int(R * 0.49)                                  # `above` ends at row last - 1
    band = lab[max(0, last - max(2, int(R * 3.0 / 180))): last]
    keep = set(np.unique(band)) - {0}
    out[(lab > 0) & ~np.isin(lab, list(keep))] = far
    return out


def cached(pano: str | Path) -> Path:
    """Where the depth of `pano` is kept: next to it (a library panorama keeps its depth for every run). A BUILT-IN
    panorama (backgrounds/library, read-only) keeps it in the user's cache: <home>/cache/pano_depth/<name>/."""
    p = Path(pano)
    from .background import BUILTIN
    try:
        rel = p.resolve().relative_to(BUILTIN.resolve())
    except ValueError:
        return p.with_name(p.stem + "_depth.npy")
    from .runner import home
    d = home() / "cache" / "pano_depth" / rel.parent
    d.mkdir(parents=True, exist_ok=True)
    return d / (p.stem + "_depth.npy")


RECIPE = 2      # the depth's recipe: change it when the graph, distance_map or the cleaning changes (old caches go)


def save(d: np.ndarray, pano: str | Path, sha256: str) -> Path:
    out = cached(pano)
    tmp = out.with_name(out.name + ".tmp.npy")
    np.save(tmp, d.astype(np.float32))
    tmp.replace(out)
    out.with_suffix(".json").write_text(json.dumps({"pano_sha256": sha256, "shape": list(d.shape), "recipe": RECIPE,
                                                    "model": "moge_2_vitl_normal_fp16"}), encoding="utf-8")
    return out


def load_if_current(pano: str | Path, sha256: str) -> Path | None:
    """The cached depth of `pano` when it was made from these exact pixels."""
    out = cached(pano)
    try:
        meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not out.is_file() or meta.get("pano_sha256") != sha256 or meta.get("recipe") != RECIPE:
        return None
    try:                                                  # a broken file is not a cache (Codex review 2026-10-22)
        d = np.load(out, mmap_mode="r")
        if d.ndim != 2 or list(d.shape) != list(meta.get("shape") or []) or not np.isfinite(d).any():
            return None
    except (OSError, ValueError):
        return None
    return out
