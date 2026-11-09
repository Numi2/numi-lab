#!/usr/bin/env python3
"""Inspect selected, read-only midpoint Newton snapshots; not an accuracy gate."""
import hashlib
import json
import math
import struct
import sys

raw = open(sys.argv[1], "rb").read()
records = {}
for line in raw.splitlines():
    if line.startswith(b"fem_newton_iterate="):
        entry = json.loads(line.split(b"=", 1)[1])
        arenas = entry["state_arenas"]
        if arenas:
            assert entry["stage"] == "final_line_search"
            assert entry["fem_node_count"] == 729
            assert entry["alpha"] == [1]
            i = entry["iteration"]
            assert i not in records
            records[i] = {key: bytes.fromhex(value) for key, value in arenas.items()}
assert set(records) == {3, 4}

def f32(data, offset):
    return struct.unpack_from("<f", data, offset)[0]

def round32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]

result = {"trace_sha256": hashlib.sha256(raw).hexdigest(),
          "description": "FP32 velocity plus unit-alpha proposed correction at the captured boundary; not independent b-Ax",
          "iterations": {}}
for iteration, arenas in sorted(records.items()):
    nodes, correction = arenas["working_nodes"], arenas["correction"]
    assert len(nodes) == 729 * 96 and len(correction) == 1458 * 16
    axes = {}
    for axis, name in enumerate("xyz"):
        values = [(f32(nodes, n * 96 + 16 + axis * 4),
                   f32(correction, n * 16 + axis * 4)) for n in range(729)]
        assert all(math.isfinite(v) and math.isfinite(d) for v, d in values)
        axes[name] = {"nonzero_proposals": sum(d != 0 for v, d in values),
                      "rounded_unchanged": sum(round32(v + d) == v for v, d in values),
                      "maximum_absolute_proposal_m_per_s": max(abs(d) for v, d in values)}
    result["iterations"][str(iteration)] = axes
print(json.dumps(result, indent=2, sort_keys=True))
