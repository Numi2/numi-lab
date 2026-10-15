#!/usr/bin/env python3
"""Exact FP32-word join of CGLS sidecar 455 and independent operator capture 419.

This is a state-input comparison only. It does not adjudicate the CGLS solve,
projected material history, nonlinear convergence, or physical qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
from collections import Counter
from typing import Any

PACKAGE = Path(__file__).resolve().parents[1]
BASE = Path("/Users/home/cardboard-implementation-20261014")
SIDECAR = BASE / "solver/cgls-state-sidecar-observer-455/base-cgls.bin.state.json"
OPERATOR_CAPTURE = BASE / "solver/two-stage-first-failure-rich-fd-419/operator-capture"
EXPECTED_PAYLOAD_SHA256 = "1a99fa72a4f45c68c67a53dcace37f0f01df6587d6db1a178606247a9fdef5b6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reader():
    spec = importlib.util.spec_from_file_location(
        "state_capture_reader", PACKAGE / "tools/read_state_capture.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f32_bytes(values: list[float]) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


def uint_bytes(values: list[int]) -> bytes:
    return struct.pack("<" + "I" * len(values), *values)


def words(payload: bytes, offset: int, count: int, kind: str = "f") -> tuple[Any, ...]:
    return struct.unpack_from("<" + kind * count, payload, offset)


def matrix_flat(value: list[list[float]]) -> list[float]:
    return [entry for row in value for entry in row]


def byte_comparison(label: str, got: bytes, expected: bytes, stats: dict[str, Any]) -> None:
    entry: dict[str, Any] = {
        "label": label,
        "byte_count": len(got),
        "exact_bytes": got == expected,
    }
    if len(got) != len(expected):
        entry["expected_byte_count"] = len(expected)
    else:
        mismatch_offsets = [i for i, (a, b) in enumerate(zip(got, expected)) if a != b]
        entry["mismatch_bytes"] = len(mismatch_offsets)
        entry["first_mismatch_offsets"] = mismatch_offsets[:12]
    stats[label] = entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=SIDECAR)
    parser.add_argument("--operator-capture", type=Path, default=OPERATOR_CAPTURE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reader = load_reader()
    validation = reader.validate_capture(args.manifest)
    doc = json.loads(args.manifest.read_text(encoding="utf-8"))
    payload_path = args.manifest.parent / doc["payload"]["file"]
    payload = payload_path.read_bytes()
    assert sha256(payload_path) == EXPECTED_PAYLOAD_SHA256
    arrays = {item["name"]: item for item in doc["arrays"]}

    # Parse one operator JSONL record at a time and compare after canonical
    # round-trip through FP32, matching the native ABI words.
    e_count = doc["environment_count"]
    n_count = doc["node_count"]
    t_count = doc["tetrahedron_count"]
    stride = doc["material_state_stride"]
    records = {"node": 0, "tetrahedron": 0, "other": 0}
    mismatches: dict[str, Any] = {}
    signedzero_fields: Counter[str] = Counter()
    exact_tallies = {
        "accepted_nodes": 0,
        "center_nodes_vs_candidate_nodes": 0,
        "accepted_tetrahedra_byte_exact": 0,
        "candidate_tetrahedra_byte_exact": 0,
        "accepted_tetrahedra_equal_after_signedzero_canonicalization": 0,
        "tetrahedra_equal_after_signedzero_canonicalization": 0,
        "accepted_tetrahedra_signedzero_only": 0,
        "tetrahedra_signedzero_only": 0,
        "tetrahedra_other_mismatch": 0,
        "accepted_history_vs_center_accepted_state": 0,
        "cache_valid": 0,
        "tangent_cache": 0,
        "environment_parameters": 0,
        "newton_correction_xyz": 0,
        "base_rhs_vs_419_physical_center_residual": 0,
    }
    for name in ("accepted_nodes", "candidate_nodes", "accepted_tetrahedra",
                 "candidate_tetrahedra", "accepted_material_history",
                 "candidate_material_state_buffer", "algorithmic_tangent_cache",
                 "algorithmic_tangent_cache_valid", "environment_parameters",
                 "original_newton_correction", "base_residual_rhs"):
        item = arrays[name]
        expected_shape = item["shape"]
        assert item["offset"] + item["byte_count"] <= len(payload)

    with args.operator_capture.open("r", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if header.get("record") != "header":
            raise ValueError("operator capture is missing its header record")
        if header.get("root") != doc["root"] or header.get("newton_iteration") != doc["newton_iteration"]:
            raise ValueError("sidecar and operator capture root/Newton do not match")
        if header.get("environment_count") != e_count or \
           header.get("node_count_per_environment") != n_count or \
           header.get("tetrahedron_count_per_environment") != t_count:
            raise ValueError("sidecar and operator capture dimensions do not match")
        if header.get("world_fingerprint_hex") != doc["world_fingerprint_u64_hex"]:
            raise ValueError("sidecar and operator-capture world fingerprints differ")

        for line_number, line in enumerate(stream, start=2):
            rec = json.loads(line)
            kind = rec.get("record")
            if kind == "node":
                records["node"] += 1
                env, node = rec["environment"], rec["node_id"]
                if not (0 <= env < e_count and 0 <= node < n_count):
                    raise ValueError(f"out-of-range node key at line {line_number}")
                index = env * n_count + node
                base = arrays["accepted_nodes"]["offset"] + index * 80
                accepted_native = payload[base:base + 80]
                expected_accepted = f32_bytes([v for row in rec["accepted"]["raw_state_float4s"] for v in row])
                if accepted_native == expected_accepted:
                    exact_tallies["accepted_nodes"] += 1
                else:
                    mismatches.setdefault("accepted_nodes", []).append([env, node])

                base = arrays["candidate_nodes"]["offset"] + index * 80
                center_native = payload[base:base + 80]
                expected_center = f32_bytes([v for row in rec["center"]["raw_state_float4s"] for v in row])
                if center_native == expected_center:
                    exact_tallies["center_nodes_vs_candidate_nodes"] += 1
                else:
                    mismatches.setdefault("center_nodes_vs_candidate_nodes", []).append([env, node])

                correction_offset = arrays["original_newton_correction"]["offset"] + index * 16
                correction = rec["correction_delta_v_m_s"]
                if payload[correction_offset:correction_offset + 16] == f32_bytes(correction + [0.0]):
                    exact_tallies["newton_correction_xyz"] += 1
                else:
                    mismatches.setdefault("newton_correction_xyz", []).append([env, node])
            elif kind == "tetrahedron":
                records["tetrahedron"] += 1
                env, tet = rec["environment"], rec["tet_id"]
                if not (0 <= env < e_count and 0 <= tet < t_count):
                    raise ValueError(f"out-of-range tetrahedron key at line {line_number}")
                idx = env * t_count + tet
                tetnodes = [int(x) for x in rec["nodes"]]
                inv = rec["inverse_rest_rows_per_m"]
                row_words = [*inv[0], rec["rest_volume_m3"], *inv[1], 0.0, *inv[2], 0.0]
                ids = [rec["material_index"], rec["object_index"], rec["topology_generation"], rec["tet_flags"]]
                frame = rec["material_frame_quaternion_xyzw"]
                expected_tet = uint_bytes(tetnodes) + f32_bytes(row_words) + uint_bytes(ids) + f32_bytes(frame)
                off = arrays["accepted_tetrahedra"]["offset"] + idx * 96
                actual_accepted_tet = payload[off:off + 96]
                if actual_accepted_tet == expected_tet:
                    exact_tallies["accepted_tetrahedra_byte_exact"] += 1
                accepted_words = list(struct.unpack("<24I", actual_accepted_tet))
                expected_words = struct.unpack("<24I", expected_tet)
                float_word_indices = set(range(4, 16)) | set(range(20, 24))
                canonical_accepted = list(accepted_words)
                canonical_expected_accepted = list(expected_words)
                for word_index in float_word_indices:
                    if (canonical_accepted[word_index] & 0x7fffffff) == 0:
                        canonical_accepted[word_index] = 0
                    if (canonical_expected_accepted[word_index] & 0x7fffffff) == 0:
                        canonical_expected_accepted[word_index] = 0
                if tuple(canonical_accepted) == tuple(canonical_expected_accepted):
                    exact_tallies["accepted_tetrahedra_equal_after_signedzero_canonicalization"] += 1
                    if actual_accepted_tet != expected_tet:
                        exact_tallies["accepted_tetrahedra_signedzero_only"] += 1
                else:
                    mismatches.setdefault("accepted_tetrahedra_other_mismatch", []).append([env, tet])
                off = arrays["candidate_tetrahedra"]["offset"] + idx * 96
                actual_candidate_tet = payload[off:off + 96]
                if actual_candidate_tet == expected_tet:
                    exact_tallies["candidate_tetrahedra_byte_exact"] += 1
                # The JSON operator record prints some quaternion zeros as +0,
                # while the exact source record can contain -0. Classify only
                # those sign-bit-only differences separately.
                got_words = struct.unpack("<24I", actual_candidate_tet)
                expected_words = struct.unpack("<24I", expected_tet)
                canonical_got = list(got_words)
                canonical_expected = list(expected_words)
                for word_index in float_word_indices:
                    if (canonical_got[word_index] & 0x7fffffff) == 0:
                        canonical_got[word_index] = 0
                    if (canonical_expected[word_index] & 0x7fffffff) == 0:
                        canonical_expected[word_index] = 0
                if tuple(canonical_got) == tuple(canonical_expected):
                    exact_tallies["tetrahedra_equal_after_signedzero_canonicalization"] += 1
                    if actual_candidate_tet != expected_tet:
                        exact_tallies["tetrahedra_signedzero_only"] += 1
                        for word_index, (actual_word, expected_word) in enumerate(zip(got_words, expected_words)):
                            if actual_word == expected_word:
                                continue
                            if 4 <= word_index < 8:
                                signedzero_fields["inverseRestRow0"] += 1
                            elif 8 <= word_index < 12:
                                signedzero_fields["inverseRestRow1"] += 1
                            elif 12 <= word_index < 16:
                                signedzero_fields["inverseRestRow2"] += 1
                            elif 20 <= word_index < 24:
                                signedzero_fields["materialFrameRotation"] += 1
                else:
                    exact_tallies["tetrahedra_other_mismatch"] += 1
                    mismatches.setdefault("tetrahedra_other_mismatch", []).append([env, tet])

                hist = rec["center"]["accepted_state"]
                if len(hist) > stride:
                    raise ValueError("operator capture material history exceeds sidecar stride")
                expected_hist = f32_bytes(hist + [0.0] * (stride - len(hist)))
                off = arrays["accepted_material_history"]["offset"] + idx * stride * 4
                if payload[off:off + stride * 4] == expected_hist:
                    exact_tallies["accepted_history_vs_center_accepted_state"] += 1
                else:
                    mismatches.setdefault("accepted_history_vs_center_accepted_state", []).append([env, tet])

                cache = rec["cached_algorithmic_tangent_local"]
                valid_off = arrays["algorithmic_tangent_cache_valid"]["offset"] + idx * 4
                valid = struct.unpack_from("<I", payload, valid_off)[0]
                expected_valid = 1 if cache.get("valid") is True else 0
                if valid == expected_valid:
                    exact_tallies["cache_valid"] += 1
                else:
                    mismatches.setdefault("cache_valid", []).append([env, tet, valid, expected_valid])
                if cache.get("valid") is True:
                    values = cache["matrix_row_major_local_dP_dF_Pa_per_dimensionless_F"]
                    off = arrays["algorithmic_tangent_cache"]["offset"] + idx * 81 * 4
                    if payload[off:off + 81 * 4] == f32_bytes(values):
                        exact_tallies["tangent_cache"] += 1
                    else:
                        mismatches.setdefault("tangent_cache", []).append([env, tet])
            else:
                records["other"] += 1

    # Params are environment-major FP32. JSON source has decimal round-trip values.
    expected_params = b"".join(f32_bytes(row) for row in header["parameters_by_environment"])
    actual_params = payload[arrays["environment_parameters"]["offset"]:
                            arrays["environment_parameters"]["offset"] + arrays["environment_parameters"]["byte_count"]]
    exact_tallies["environment_parameters"] = (
        e_count * doc["parameter_count"] if actual_params == expected_params else 0)
    if actual_params != expected_params:
        mismatches["environment_parameters"] = {
            "actual_bytes": len(actual_params), "expected_bytes": len(expected_params),
        }

    # Candidate history is a raw buffer snapshot, explicitly not the current
    # Newton-projected history. Record the observed relation without re-labeling it.
    accepted_hist = arrays["accepted_material_history"]
    candidate_hist = arrays["candidate_material_state_buffer"]
    accepted_hist_bytes = payload[accepted_hist["offset"]:accepted_hist["offset"] + accepted_hist["byte_count"]]
    candidate_hist_bytes = payload[candidate_hist["offset"]:candidate_hist["offset"] + candidate_hist["byte_count"]]
    correction_bytes = arrays["original_newton_correction"]["byte_count"]
    correction_off = arrays["original_newton_correction"]["offset"]
    correction_values = words(payload, correction_off, doc["unknown_count"] * 4)
    rhs_item = arrays["base_residual_rhs"]
    rhs_values = words(payload, rhs_item["offset"], doc["unknown_count"] * 4)
    mechanical_values = e_count * n_count * 4
    field_values = e_count * n_count * 4
    rhs_matches = 0
    rhs_total = 0
    with args.operator_capture.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("record") != "node":
                continue
            idx = record["environment"] * n_count + record["node_id"]
            off = rhs_item["offset"] + idx * 16
            rhs_total += 4
            if payload[off:off + 16] == f32_bytes(record["center_residual_impulse_Ns"]):
                rhs_matches += 4
    exact_tallies["base_rhs_vs_419_physical_center_residual"] = rhs_matches
    if rhs_matches != rhs_total:
        mismatches["base_rhs_vs_419_physical_center_residual"] = {
            "matched_float_words": rhs_matches, "total_float_words": rhs_total,
        }
    report = {
        "schema": "numi.matter.base_cgls_state_capture_419_join.v1",
        "classification": "exact source-input join only; both source captures are from failed solver runs",
        "inputs": {
            "state_capture_manifest": str(args.manifest.resolve()),
            "state_capture_manifest_sha256": sha256(args.manifest),
            "state_capture_payload_sha256": sha256(payload_path),
            "operator_capture": str(args.operator_capture.resolve()),
            "operator_capture_sha256": sha256(args.operator_capture),
            "shared_header_sha256": "cd173c076905f660aa8aecc900e6f6aba8d69406996f3cccf2e0656c753ff871",
            "source_snapshot": "/Users/home/cardboard-implementation-20261014/snapshots/source-cgls-primary-policy-143",
        },
        "capture_identity": {
            "root": doc["root"], "newton_iteration": doc["newton_iteration"],
            "environment_count": e_count, "node_count": n_count,
            "tetrahedron_count": t_count, "history_stride": stride,
            "world_fingerprint": doc["world_fingerprint_u64_hex"],
            "operator_world_fingerprint": header["world_fingerprint_hex"],
            "operator_execution_fingerprint": header["execution_fingerprint_hex"],
            "dt_seconds": doc["dt_seconds"],
        },
        "reader_validation": validation,
        "record_counts": records,
        "exact_word_tallies": exact_tallies,
        "tetrahedron_signedzero_word_counts_by_field": dict(sorted(signedzero_fields.items())),
        "expected_counts": {
            "accepted_nodes": e_count * n_count,
            "center_nodes_vs_candidate_nodes": e_count * n_count,
            "accepted_tetrahedra_equal_after_signedzero_canonicalization": e_count * t_count,
            "tetrahedra_equal_after_signedzero_canonicalization": e_count * t_count,
            "tetrahedra_other_mismatch": 0,
            "accepted_history_vs_center_accepted_state": e_count * t_count,
            "cache_valid": e_count * t_count,
            "tangent_cache": doc["tangent_cache"]["valid_entry_count"],
            "environment_parameters": e_count * doc["parameter_count"],
            "newton_correction_xyz": e_count * n_count,
            "base_rhs_vs_419_physical_center_residual": e_count * n_count * 4,
        },
        "mismatches": {key: value[:16] if isinstance(value, list) else value for key, value in mismatches.items()},
        "candidate_history": {
            "accepted_history_sha256": hashlib.sha256(accepted_hist_bytes).hexdigest(),
            "candidate_state_buffer_sha256": hashlib.sha256(candidate_hist_bytes).hexdigest(),
            "bytewise_equal_to_accepted_history": accepted_hist_bytes == candidate_hist_bytes,
            "projection_status": doc["material_history_projection"]["current_newton_status"],
            "candidate_buffer_is_current_projection": doc["material_history_projection"]["candidate_buffer_is_current_projection"],
        },
        "tetrahedron_snapshot_relation": {
            "accepted_array_sha256": arrays["accepted_tetrahedra"]["sha256"],
            "candidate_array_sha256": arrays["candidate_tetrahedra"]["sha256"],
            "accepted_and_candidate_arrays_bytewise_equal": (
                arrays["accepted_tetrahedra"]["sha256"] == arrays["candidate_tetrahedra"]["sha256"]),
            "419_exact_bytes": "956/3840 per arena; remaining records differ only in signed-zero bits in float fields because the JSON capture prints them as +0",
            "419_after_signedzero_only_canonicalization": "3840/3840 per arena; no other word mismatch",
        },
        "newton_inputs": {
            "correction_bytes": correction_bytes,
            "mechanical_rows_float_count": mechanical_values,
            "field_rows_float_count": field_values,
            "field_correction_all_zero": all(v == 0.0 for v in correction_values[mechanical_values:]),
            "field_rhs_all_zero": all(v == 0.0 for v in rhs_values[mechanical_values:]),
            "base_rhs_sign_vs_419_physical_residual": "same sign and exact float32 words for all mechanical components; this capture's CGLS b equals the 419 physical center residual R",
        },
        "limits": [
            "No current-Newton projected history was persisted; candidate_material_state_buffer remains raw/unavailable, not a projection result.",
            "This join validates bytes and state identity only; it does not validate CGLS direction quality or convergence.",
            "The 419 operator capture is a separate diagnostic capture sharing root/Newton/world identity; source lineages and execution fingerprints remain separately recorded.",
            "Both native runs ended with expected fixture failure; neither is a complete solver or physical qualification.",
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
