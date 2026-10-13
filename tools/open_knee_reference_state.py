"""Read a hash-admitted Open Knee reference; never infer admission from metadata.

The expected NPZ digest must come from an independently frozen extraction
record. It binds this derived artifact, while the recorded source XPLT identity
and field hashes constrain its shape and lineage. This is software admission,
not solver/source equivalence or a cryptographic attestation of extraction.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

SOURCE_XPLT_SHA256 = "c370ae9f94e9faee2d7060bf2a6819e03be1312e82ca79bd2e9cebf8b34398de"

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def load_displacement_reference(path: Path, expected_sha256: str):
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("reference digest must be 64 lowercase hexadecimal characters")
    if digest(path) != expected_sha256:
        raise ValueError("reference NPZ differs from the independently frozen digest")
    with np.load(path, allow_pickle=False) as archive:
        meta = archive["metadata_json"]
        if meta.shape != () or meta.dtype.kind != "U":
            raise ValueError("reference metadata must be a Unicode scalar")
        metadata = json.loads(str(meta))
        if not isinstance(metadata, dict) or metadata.get("schema") != "numi.matter.open-knee-xplt-state.v1":
            raise ValueError("unsupported reference metadata schema")
        if metadata.get("source_xplt_sha256") != SOURCE_XPLT_SHA256:
            raise ValueError("reference state does not identify the pinned source XPLT")
        time = metadata.get("continuation_time")
        if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time) or time < 0:
            raise ValueError("reference continuation time must be finite and nonnegative")
        state = metadata.get("state_index")
        if isinstance(state, bool) or not isinstance(state, int) or state < 0:
            raise ValueError("reference state index must be a nonnegative integer")
        array = archive["node_01"]
        if array.dtype != np.dtype("<f4") or array.ndim != 2 or array.shape[1] != 3 or not array.shape[0]:
            raise ValueError("reference displacement must be a nonempty float32 Nx3 array")
        if not np.isfinite(array).all():
            raise ValueError("reference displacement contains nonfinite values")
        for label in ["node_count", "mesh_node_count"]:
            count = metadata.get(label)
            if isinstance(count, bool) or not isinstance(count, int) or count != array.shape[0]:
                raise ValueError("reference node counts differ from the displacement array")
        fields = metadata.get("fields")
        if not isinstance(fields, list):
            raise ValueError("reference field manifest is missing")
        matching = [f for f in fields if isinstance(f, dict) and f.get("class") == "node"
                    and type(f.get("dictionary_id")) is int and f["dictionary_id"] == 1]
        if len(matching) != 1 or matching[0].get("shape") != list(array.shape):
            raise ValueError("reference displacement field manifest is ambiguous or malformed")
        if hashlib.sha256(array.tobytes(order="C")).hexdigest() != matching[0].get("float32_array_sha256"):
            raise ValueError("reference displacement bytes differ from their field manifest")
        result = array.astype(np.float64, copy=True)
    # Detect a reference replacement during the read before returning evidence.
    if digest(path) != expected_sha256:
        raise ValueError("reference NPZ changed while reading")
    return metadata, result
