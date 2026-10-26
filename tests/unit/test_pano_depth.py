"""The panorama depth reader (cinestudio/pano_depth.py): a GLB grid mesh -> the distance of every panorama pixel."""
import json
import struct

import numpy as np
import pytest

from cinestudio import pano_depth


def glb(path, pos, uv):
    """A minimal binary glTF with POSITION and TEXCOORD_0 (as ComfyUI's SaveGLB writes them)."""
    pos, uv = pos.astype(np.float32), uv.astype(np.float32)
    binc = pos.tobytes() + uv.tobytes()
    js = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binc)}],
          "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": pos.nbytes},
                          {"buffer": 0, "byteOffset": pos.nbytes, "byteLength": uv.nbytes}],
          "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3"},
                        {"bufferView": 1, "componentType": 5126, "count": len(uv), "type": "VEC2"}],
          "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}}]}]}
    j = json.dumps(js).encode()
    j += b" " * (-len(j) % 4)
    binc += b"\0" * (-len(binc) % 4)
    total = 12 + 8 + len(j) + 8 + len(binc)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<II", len(j), 0x4E4F534A) + j
                     + struct.pack("<II", len(binc), 0x004E4942) + binc)


def test_the_distance_of_each_pixel_comes_back_in_place(tmp_path):
    rows, cols = 8, 16
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    dist = np.where(yy < rows // 2, 50.0, 2.0 + yy)                 # sky far at the top, floor near at the bottom
    dist[3, 5] = 7.0                                                # one marked pixel
    rng = np.random.default_rng(1)
    dirs = rng.normal(size=(rows * cols, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)              # any direction: only the length is read
    pos = dirs * dist.reshape(-1, 1)
    uv = np.stack([xx.ravel() / (cols - 1), yy.ravel() / (rows - 1)], axis=1)
    keep = np.ones(rows * cols, bool)
    keep[2 * cols + 7] = False                                      # a vertex the mesh left out
    glb(tmp_path / "d.glb", pos[keep], uv[keep])
    d = pano_depth.distance_map(tmp_path / "d.glb")
    assert d.shape == (rows, cols)
    assert d[3, 5] == pytest.approx(7.0, rel=1e-5) and d[7, 0] == pytest.approx(9.0, rel=1e-5)
    assert np.isnan(d[2, 7])


def test_an_upside_down_map_is_refused(tmp_path):
    rows, cols = 8, 16
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    dist = np.where(yy >= rows // 2, 50.0, 2.0)                     # far at the BOTTOM: not a panorama layout
    pos = np.stack([dist.ravel(), np.zeros(rows * cols), np.zeros(rows * cols)], axis=1)
    uv = np.stack([xx.ravel() / (cols - 1), yy.ravel() / (rows - 1)], axis=1)
    glb(tmp_path / "d.glb", pos, uv)
    with pytest.raises(pano_depth.DepthError, match="nearer"):
        pano_depth.distance_map(tmp_path / "d.glb")


def test_the_cache_follows_the_pixels(tmp_path):
    pano = tmp_path / "pano.png"
    pano.write_bytes(b"x")
    pano_depth.save(np.ones((4, 8), np.float32), pano, "sha-a")
    assert pano_depth.load_if_current(pano, "sha-a") == pano_depth.cached(pano)
    assert pano_depth.load_if_current(pano, "sha-b") is None


def test_a_cache_of_another_recipe_or_a_broken_file_is_not_used(tmp_path):
    pano = tmp_path / "pano.png"
    pano.write_bytes(b"x")
    f = pano_depth.save(np.ones((4, 8), np.float32), pano, "sha")
    meta = json.loads(f.with_suffix(".json").read_text())
    f.with_suffix(".json").write_text(json.dumps({**meta, "recipe": 1}))
    assert pano_depth.load_if_current(pano, "sha") is None          # an old recipe
    f.with_suffix(".json").write_text(json.dumps(meta))
    f.write_bytes(b"not an array")
    assert pano_depth.load_if_current(pano, "sha") is None          # a broken file


def test_a_job_client_without_the_file_falls_back_to_comfyui_history(tmp_path, monkeypatch):
    from cinestudio import backend, server

    class Job:
        def get_outputs(self, node):
            return []

    class Jobs:
        def get(self, jid):
            return Job()

    b = backend.SdkBackend.__new__(backend.SdkBackend)
    b.client = type("C", (), {"jobs": Jobs()})()
    got = {}
    monkeypatch.setattr(server, "comfyui_url", lambda: "http://x")
    monkeypatch.setattr(server, "download_output", lambda url, jid, node, dst: got.update(jid=jid, node=node) or dst)
    assert b.download("j1", "save", tmp_path / "m.glb") == tmp_path / "m.glb" and got == {"jid": "j1", "node": "save"}
