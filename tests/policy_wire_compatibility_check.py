#!/usr/bin/env python3
"""CPU-only native producer/MLX reader and C ABI wire compatibility check."""
from pathlib import Path
import struct
import sys
import tempfile
import numpy as np

from metalrobo.mlx_policy_learning import read_policy_pack
from metalrobo.native import PolicyDenseLayerArtifact, learning_pack_content_hash, write_policy_pack

root = Path(sys.argv[1])
library = Path(sys.argv[2])
fixtures = root / "docs/evidence/window-integration-20261107/historical"
v4 = read_policy_pack(fixtures / "native-policy-v4.pack", library_path=library)
v5 = read_policy_pack(fixtures / "native-policy-v5-multitask.pack", library_path=library)
assert v4.format_version == 4 and v4.contract_version == 1
assert v4.compatible_task_bindings == ()
assert v5.format_version == 5 and v5.contract_version == 2
assert v5.compatible_task_bindings == ((11, 22), (11, 23))
assert v4.id == v5.id == "wire-fixture"
assert np.array_equal(v4.layers[0].weights, v5.layers[0].weights)

def must_fail(path: Path) -> None:
    try:
        read_policy_pack(path, library_path=library)
    except ValueError:
        return
    raise AssertionError(f"invalid policy unexpectedly accepted: {path}")

with tempfile.TemporaryDirectory() as td:
    directory = Path(td)
    layer = PolicyDenseLayerArtifact(
        weights=np.array([[0.5, -0.5]], dtype=np.float32),
        bias=np.array([0.125], dtype=np.float32), activation=0)
    common = dict(policy_id="python-wire", revision=2,
                  world_fingerprint=11, task_fingerprint=22,
                  observation_fingerprint=33, action_fingerprint=44,
                  layers=[layer], library_path=library)
    for version, bindings in ((1, ()), (2, ((11, 22), (11, 23)))):
        path = directory / f"roundtrip-{version}.pack"
        write_policy_pack(path, contract_version=version,
                          compatible_task_bindings=bindings, **common)
        loaded = read_policy_pack(path, library_path=library)
        assert loaded.contract_version == version
        assert loaded.compatible_task_bindings == bindings
    try:
        write_policy_pack(directory / "invalid-writer.pack", contract_version=2,
                          compatible_task_bindings=((11, 22), (11, 22)), **common)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate writer bindings accepted")
    data = bytearray((fixtures / "native-policy-v5-multitask.pack").read_bytes())
    data[-16:] = data[-32:-16]  # duplicate a binding while preserving wire size
    payload_hash = learning_pack_content_hash(
        np.frombuffer(data, dtype=np.uint8, offset=32), path=library)
    struct.pack_into("<Q", data, 24, payload_hash)
    invalid = directory / "duplicate.pack"
    invalid.write_bytes(data)
    must_fail(invalid)
    invalid.write_bytes(data[:-1])
    must_fail(invalid)
print("v4/v5 native-producer, Python C ABI round trip, and malformed checks passed")
