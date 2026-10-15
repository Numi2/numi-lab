#!/usr/bin/env python3
"""Pack exact captured EE endpoints from native private-trial cases 333/334."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import numpy as np

BASE = Path("/Users/home/cardboard-implementation-20261014")
READER_DIR = BASE / "analysis/private-trial-reader-001"
sys.path.insert(0, str(READER_DIR))
from trial_capture_reader import Capture  # noqa: E402

EXPECTED = {
    333: {
        "payload": "8fe7d2943880bb9962b1745569fa4120d88a317fee43b00fbbe9d609cc5d07eb",
        "manifest": "c62cc99acfb766413d604250586704b63e9cd6e5930064329dfb57a277815e74",
        "trial": 20,
    },
    334: {
        "payload": "bb645f01802e2d0cfcec1a15ec0769ff988e8f06b1ce4a58610bc82c8c7f6e38",
        "manifest": "eeff0101144e3e6e4fffbd46a5b4b9642b2e5bc6ab071ff3969ef4d9ecdeec00",
        "trial": 21,
    },
}
CASES_SCHEMA = "numi.matter.private_ee_ccd_native_cases.v1"
CONTACT_SOURCE_SHA256 = "3cef6ae3dbd384d7b26ff6b0eb4ecda6da7a32be74cdea0badc5975eef692132"
FROZEN_MANIFEST_SHA256 = "5fb22c61c280f576f15677c8cf6dbbc174beceaa225973242f6d53f9c7957446"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def records(raw: bytes, stride: int) -> np.ndarray:
    if len(raw) % stride:
        raise ValueError("unaligned record payload")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, stride)


def source_case(manifest_path: Path, case_id: int):
    cap = Capture(manifest_path)
    expected = EXPECTED[case_id]
    manifest_bytes = manifest_path.read_bytes()
    if sha(manifest_bytes) != expected["manifest"]:
        raise ValueError(f"case {case_id}: manifest SHA changed")
    if sha(cap.payload) != expected["payload"]:
        raise ValueError(f"case {case_id}: payload SHA changed")
    m = cap.manifest
    if (m.get("root"), m.get("iteration"), m.get("trial"), m.get("selected_phase")) != (
        1201, 12, expected["trial"], "both"
    ):
        raise ValueError(f"case {case_id}: unexpected captured selector identity")
    phase = "private_contact_rebuild_before_free_body_terms"
    candidate_records = records(cap.raw("deformable_candidates_private_trial", phase), 16)
    candidate = np.frombuffer(candidate_records[331], dtype="<u4")
    if tuple(int(v) for v in candidate) != (88, 127, 331, 1):
        raise ValueError(f"case {case_id}: active candidate identity changed")
    primitive_records = records(cap.raw("surface_primitives_private_trial", phase), 64)
    primitive_ids = [88, 127]
    primitive_nodes = [
        np.frombuffer(primitive_records[i], dtype="<u4")[:3].astype(int).tolist()
        for i in primitive_ids
    ]
    if primitive_nodes != [[13, 14, 17], [28, 32, 31]]:
        raise ValueError(f"case {case_id}: primitive topology changed")
    node_records = records(cap.raw("fem_armijo_candidate_nodes", phase), 80)
    dispatch = np.frombuffer(cap.raw("dispatch_buffer", "candidate_before_private_contact_rebuild"), dtype="<f4")
    scheduler_words = np.frombuffer(cap.raw("schedulers", "candidate_before_private_contact_rebuild"), dtype="<u4")
    base_dt = np.float32(dispatch[43])
    exponent = int(scheduler_words[1])
    object_dt = np.float32(base_dt / np.float32(2 ** exponent))
    thickness = np.float32(dispatch[44])
    if not (np.isfinite(object_dt) and object_dt > 0 and np.isfinite(thickness) and thickness > 0):
        raise ValueError(f"case {case_id}: invalid exact captured dt/thickness")

    node_ids = [13, 14, 28, 32]
    endpoints = {}
    for node_id in node_ids:
        state = np.frombuffer(node_records[node_id], dtype="<f4")
        finish = np.asarray(state[0:3], dtype="<f4").copy()
        velocity = np.asarray(state[4:7], dtype="<f4").copy()
        start = np.asarray(
            [np.float32(finish[k] - np.float32(object_dt * velocity[k])) for k in range(3)],
            dtype="<f4",
        )
        endpoints[node_id] = {"start": start, "finish": finish, "velocity": velocity}

    packed = []
    rows = []
    for budget in (16, 17):
        case_index = len(rows)
        # Nine records follow the exact layout consumed by the existing
        # production regression kernel: kind/thickness, four starts, four ends.
        points = [
            np.asarray([1.0, thickness, float(budget), float(case_id)], dtype="<f4"),
            np.r_[endpoints[13]["start"], np.float32(0)],
            np.r_[endpoints[14]["start"], np.float32(0)],
            np.r_[endpoints[28]["start"], np.float32(0)],
            np.r_[endpoints[32]["start"], np.float32(0)],
            np.r_[endpoints[13]["finish"], np.float32(0)],
            np.r_[endpoints[14]["finish"], np.float32(0)],
            np.r_[endpoints[28]["finish"], np.float32(0)],
            np.r_[endpoints[32]["finish"], np.float32(0)],
        ]
        case_bytes = b"".join(np.asarray(p, dtype="<f4").tobytes() for p in points)
        if len(case_bytes) != 9 * 16:
            raise AssertionError("bad case record length")
        packed.append(case_bytes)
        rows.append({
            "index": case_index,
            "capture_case": case_id,
            "capture_trial": expected["trial"],
            "budget": budget,
            "kind": "edge_edge",
            "thickness_f32_hex": np.asarray([thickness], dtype="<f4").tobytes().hex(),
            "edge_node_ids": [[13, 14], [28, 32]],
            "record_layout": "9 little-endian float4: header; p00,p10,q00,q10; p01,p11,q01,q11",
            "records_f32_hex": [np.asarray(p, dtype="<f4").tobytes().hex() for p in points],
        })
    return cap, rows, b"".join(packed), {
        "case": case_id,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha(manifest_bytes),
        "payload_sha256": sha(cap.payload),
        "candidate_identity": [int(v) for v in candidate],
        "primitive_nodes": primitive_nodes,
        "dispatch_sha256": sha(cap.raw("dispatch_buffer", "candidate_before_private_contact_rebuild")),
        "base_dt_f32_hex": np.asarray([base_dt], dtype="<f4").tobytes().hex(),
        "scheduler_exponent": exponent,
        "object_dt_f32_hex": np.asarray([object_dt], dtype="<f4").tobytes().hex(),
        "thickness_f32_hex": np.asarray([thickness], dtype="<f4").tobytes().hex(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-333", type=Path, required=True)
    parser.add_argument("--capture-334", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    captures = []
    cases = []
    chunks = []
    for case_id, path in ((333, args.capture_333), (334, args.capture_334)):
        cap, rows, blob, identity = source_case(path, case_id)
        captures.append(identity)
        cases.extend(rows)
        chunks.append(blob)
    payload = b"".join(chunks)
    if len(cases) != 4 or len(payload) != 4 * 9 * 16:
        raise AssertionError("unexpected four-case packed input")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "captured-ee-cases.float4.bin"
    data_path.write_bytes(payload)
    manifest = {
        "schema": CASES_SCHEMA,
        "contact_source_sha256": CONTACT_SOURCE_SHA256,
        "frozen_source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "captures": captures,
        "case_count": len(cases),
        "case_stride_float4": 9,
        "case_stride_bytes": 144,
        "payload_file": data_path.name,
        "payload_bytes": len(payload),
        "payload_sha256": sha(payload),
        "cases": cases,
        "classification": "exact captured FP32 endpoints packed for native CCD diagnostics; no GPU result is asserted here",
    }
    manifest_path = args.output_dir / "captured-ee-cases.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "payload_bytes": len(payload),
                      "payload_sha256": manifest["payload_sha256"], "case_count": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
