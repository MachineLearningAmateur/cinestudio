"""ComfyUI latent files (.latent = safetensors) read and cut without torch: the app's Python has no torch or
safetensors package. The format: 8 bytes little-endian header length, a JSON header {name: {dtype, shape,
data_offsets}, "__metadata__": {...}}, then the raw tensor bytes.

Used by the audio carry (2026-10-24, docs/audio_consistency_research.md): clip k-1's saved stage-1 AUDIO latent is cut
to its tail (the shared frames) and frozen at the start of clip k, so the sound continues across the join.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "F64": np.float64,
           "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}

AUDIO_LATENTS_PER_SECOND = 25        # LTX-2 audio VAE: 16 kHz / hop 160 / 4 (comfy ldm/lightricks/vae/audio_vae.py)


def read(path: str | Path) -> tuple[dict[str, tuple[str, np.ndarray]], dict]:
    """{name: (dtype name, array)} and the metadata of a safetensors file. BF16 arrays come back as raw uint16."""
    raw = Path(path).read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n].decode("utf-8"))
    meta = header.pop("__metadata__", {}) or {}
    body = raw[8 + n:]
    out = {}
    for name, e in header.items():
        if e["dtype"] not in _DTYPES:
            raise ValueError(f"{path}: tensor {name!r} has an unknown dtype {e['dtype']}")
        a, b = e["data_offsets"]
        arr = np.frombuffer(body[a:b], dtype=_DTYPES[e["dtype"]]).reshape(e["shape"]) if e["shape"] else \
            np.frombuffer(body[a:b], dtype=_DTYPES[e["dtype"]])
        out[name] = (e["dtype"], arr)
    return out, meta


def write(path: str | Path, tensors: dict[str, tuple[str, np.ndarray]], meta: dict | None = None) -> None:
    """Write {name: (dtype name, array)} as a safetensors file (atomic: a temporary file, then a rename)."""
    header, chunks, off = {}, [], 0
    for name, (dtype, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        chunks.append(b)
        off += len(b)
    if meta:
        header["__metadata__"] = {str(k): str(v) for k, v in meta.items()}
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    h += b" " * ((8 - len(h) % 8) % 8)                 # the header is padded to 8 bytes (safetensors convention)
    dst = Path(path)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(chunks))
    tmp.replace(dst)


def cut_tail(src: str | Path, dst: str | Path, start: int, axis: int = 2) -> int:
    """Keep `latent_tensor`[..., start:, ...] along `axis` (time: axis 2 for an LTX audio latent (B, C, T, F)) and every
    other key as it is (ComfyUI's `latent_format_version_0` marker: without it LoadLatent multiplies by 1/0.18215).
    Returns the number of kept latent frames."""
    tensors, meta = read(src)
    if "latent_tensor" not in tensors:
        raise ValueError(f"{src}: no latent_tensor")
    dtype, arr = tensors["latent_tensor"]
    if not 0 <= start < arr.shape[axis]:
        raise ValueError(f"{src}: tail start {start} outside 0..{arr.shape[axis] - 1}")
    idx = [slice(None)] * arr.ndim
    idx[axis] = slice(start, None)
    tensors["latent_tensor"] = (dtype, arr[tuple(idx)])
    write(dst, tensors, meta)
    return arr.shape[axis] - start


def audio_tail_start(frames: int, overlap: int, fps: float) -> int:
    """The first audio latent of a clip's shared tail: the next clip starts on picture frame frames - overlap
    (0-based), which is (frames - overlap) / fps seconds into this clip (121 frames, 25 shared, 24 fps: 4.0 s = 100)."""
    return int(round((frames - overlap) / float(fps) * AUDIO_LATENTS_PER_SECOND))
