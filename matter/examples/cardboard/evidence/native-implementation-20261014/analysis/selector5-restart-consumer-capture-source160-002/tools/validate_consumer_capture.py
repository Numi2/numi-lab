#!/usr/bin/env python3
"""Validate an exact selector-5 restart-consumer capture and rebuild raw RAP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

CAP_BYTES = 64 * 1024 * 1024
SCHEMA = "numi.matter.selector5_restart_consumer_capture.v2"


class CaptureError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise CaptureError(message)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_map(manifest: dict[str, Any], payload: bytes) -> dict[str, memoryview]:
    arrays = manifest.get("arrays")
    _need(isinstance(arrays, list) and arrays, "arrays_missing")
    mapped: dict[str, memoryview] = {}
    expected_offset = 0
    for array in arrays:
        name = array.get("name")
        _need(isinstance(name, str) and name and name not in mapped,
              "array_name_missing_or_duplicate")
        shape = array.get("shape")
        _need(isinstance(shape, list) and shape and
              all(type(value) is int and value > 0 for value in shape),
              f"{name}:invalid_shape")
        element_bytes = array.get("element_bytes")
        byte_count = array.get("byte_count")
        offset = array.get("offset")
        _need(type(element_bytes) is int and element_bytes > 0 and
              type(byte_count) is int and byte_count > 0 and
              type(offset) is int and offset >= 0,
              f"{name}:invalid_extent")
        count = math.prod(shape)
        _need(count <= (1 << 64) - 1 and
              count * element_bytes == byte_count,
              f"{name}:shape_byte_count_mismatch")
        _need(offset == expected_offset, f"{name}:noncontiguous_or_overlapping_offset")
        end = offset + byte_count
        _need(end <= len(payload), f"{name}:payload_range_out_of_bounds")
        view = memoryview(payload)[offset:end]
        _need(array.get("sha256") == _sha(view), f"{name}:sha256_mismatch")
        mapped[name] = view
        expected_offset = end
    _need(expected_offset == len(payload), "payload_has_unindexed_bytes")
    return mapped


def _u32(view: memoryview, index: int = 0) -> int:
    offset = index * 4
    _need(offset + 4 <= len(view), "u32_array_too_short")
    return struct.unpack_from("<I", view, offset)[0]


def _f32(view: memoryview, index: int = 0) -> float:
    offset = index * 4
    _need(offset + 4 <= len(view), "f32_array_too_short")
    return struct.unpack_from("<f", view, offset)[0]


def reconstruct_raw_rap(manifest: dict[str, Any], arrays: dict[str, memoryview]) -> list[list[float]]:
    p = arrays["augmented_effective_P"]
    ap = arrays["augmented_AP"]
    p_meta = next(item for item in manifest["arrays"]
                  if item["name"] == "augmented_effective_P")
    ap_meta = next(item for item in manifest["arrays"]
                   if item["name"] == "augmented_AP")
    _need(p_meta["shape"] == ap_meta["shape"] and
          len(p_meta["shape"]) == 2 and p_meta["element_bytes"] == 16 and
          ap_meta["element_bytes"] == 16,
          "effective_P_AP_layout_mismatch")
    modes, nodes = p_meta["shape"]
    rap = [[0.0 for _ in range(modes)] for _ in range(modes)]
    for i in range(modes):
        for j in range(modes):
            total = 0.0
            for node in range(nodes):
                pi = (i * nodes + node) * 16
                aj = (j * nodes + node) * 16
                pxyz = struct.unpack_from("<3f", p, pi)
                axyz = struct.unpack_from("<3f", ap, aj)
                total += sum(float(x) * float(y) for x, y in zip(pxyz, axyz))
            rap[i][j] = total
    return rap


def validate_capture(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    _need(manifest.get("schema") == SCHEMA, "schema_mismatch")
    _need(manifest.get("capture_point") ==
          "after_graph_apply_barrier_before_fem_operator_consumes_z",
          "capture_point_mismatch")
    payload_info = manifest.get("payload", {})
    payload_name = payload_info.get("file")
    _need(isinstance(payload_name, str) and payload_name,
          "payload_filename_missing")
    payload_path = Path(payload_name)
    if not payload_path.is_absolute():
        payload_path = manifest_path.parent / payload_path
    payload = payload_path.read_bytes()
    _need(len(payload) <= CAP_BYTES, "aggregate_64MiB_payload_cap_exceeded")
    _need(payload_info.get("byte_count") == len(payload), "payload_byte_count_mismatch")
    _need(payload_info.get("aggregate_cap_bytes") == CAP_BYTES,
          "capture_cap_binding_mismatch")
    _need(type(manifest.get("maximum_rate_exponent")) is int and
          manifest["maximum_rate_exponent"] == 0 and
          type(manifest.get("microtick")) is int and manifest["microtick"] == 0 and
          type(manifest.get("microtick_count")) is int and
          manifest["microtick_count"] == 1,
          "capture_outside_zero_rate_single_microtick_scope")
    _need(manifest.get("selector_attempted") is True and
          type(manifest.get("selector_attempt_count")) is int and
          manifest["selector_attempt_count"] == 1,
          "capture_not_unique_one_shot_event")
    for field in ("control_step", "physics_substep", "physics_substep_count",
                  "micro_generation"):
        _need(type(manifest.get(field)) is int and
              0 <= manifest[field] <= 0xFFFFFFFF,
              f"{field}:invalid_uint32")
    physics_count = manifest["physics_substep_count"]
    micro_generation = manifest["micro_generation"]
    _need(physics_count > 0 and manifest["physics_substep"] < physics_count,
          "physics_substep_count_or_index_invalid")
    expected_generation = (
        (manifest["control_step"] * physics_count + manifest["physics_substep"])
        + 1
    ) & 0xFFFFFFFF
    if expected_generation == 0:
        expected_generation = 1
    _need(micro_generation == expected_generation,
          "micro_generation_does_not_match_selected_single_tick")
    _need(payload_info.get("sha256") == _sha(payload), "payload_sha256_mismatch")
    arrays = _array_map(manifest, payload)
    required = {
        "v", "z", "augmented_effective_P", "augmented_AP",
        "augmented_coarse_lu", "augmented_coarse_pivots",
        "augmented_coarse_valid", "augmented_coarse_rhs",
        "augmented_coarse_coefficients", "graph_patch_lu",
        "graph_patch_original_matrices", "graph_patch_pivots",
        "graph_patch_factor_valid", "graph_patch_corrections",
        "graph_patches", "graph_patch_nodes", "graph_rows",
        "graph_node_patch_offsets", "graph_node_patch_values",
        "graph_node_coverage", "candidate_nodes", "linearization_nodes",
        "accepted_material_history", "scheduler_exponents",
        "deformation_high", "deformation_low", "fgmres_state",
        "production_status", "restart_generation",
        "restart_raw_norm_squared", "restart_target_scale",
        "restart_mode_scale", "restart_mode_valid", "restart_tail_preserved",
    }
    _need(required.issubset(arrays), "required_consumer_arrays_missing")
    mode_count = manifest.get("mode_count")
    base_mode_count = manifest.get("base_mode_count")
    mode_stride = manifest.get("mode_stride")
    _need(manifest.get("preconditioner_selector") == 5 and
          manifest.get("restart_cycle", 0) > 0 and
          type(mode_count) is int and mode_count > 0 and
          type(base_mode_count) is int and base_mode_count > 0 and
          type(mode_stride) is int and mode_stride >= mode_count,
          "selector_or_mode_layout_mismatch")
    _need(next(item for item in manifest["arrays"]
               if item["name"] == "augmented_effective_P")["shape"][0] == mode_count,
          "P_mode_count_mismatch")
    _need(next(item for item in manifest["arrays"]
               if item["name"] == "augmented_coarse_lu")["shape"] ==
          [mode_count, mode_count], "coarse_LU_shape_mismatch")
    rap_values = reconstruct_raw_rap(manifest, arrays)
    rap_nonfinite = sum(not math.isfinite(value)
                        for row in rap_values for value in row)
    rap = [[value if math.isfinite(value) else None for value in row]
           for row in rap_values]
    generation = [_u32(arrays["restart_generation"], i) for i in range(4)]
    expected = manifest.get("expected_generator_tag")
    _need(isinstance(expected, list) and len(expected) == 4 and
          all(type(v) is int and 0 <= v <= 0xFFFFFFFF for v in expected),
          "expected_generator_tag_invalid")
    mode_valid = _u32(arrays["restart_mode_valid"])
    tail = _u32(arrays["restart_tail_preserved"])
    coarse_valid = _u32(arrays["augmented_coarse_valid"])
    graph_valid_array = arrays["graph_patch_factor_valid"]
    graph_patch_valid = [
        _u32(graph_valid_array, index)
        for index in range(len(graph_valid_array) // 4)
    ]
    fgmres_view = arrays["fgmres_state"]
    _need(len(fgmres_view) >= 32, "fgmres_state_truncated")
    latch = struct.unpack_from("<f", fgmres_view, 8)[0]
    status_code = _u32(arrays["production_status"])
    generation_matches = generation == expected
    valid_augmented_consumer = (
        manifest.get("augmentation_enabled") is True and
        generation_matches and mode_valid == 1 and coarse_valid == 1 and
        tail == 0 and all(value == 1 for value in graph_patch_valid) and
        math.isfinite(latch) and latch <= 0.5 and status_code == 0
    )
    return {
        "integrity": "pass",
        "consumer_state": "valid_augmented_map_inputs" if valid_augmented_consumer
            else "captured_invalid_or_fallback_state",
        "generation_matches_expected": generation_matches,
        "actual_restart_generation": generation,
        "expected_generator_tag": expected,
        "restart_mode_valid": mode_valid,
        "augmented_coarse_valid": coarse_valid,
        "restart_tail_preserved": tail,
        "graph_patch_valid_count": sum(value == 1 for value in graph_patch_valid),
        "graph_patch_count": len(graph_patch_valid),
        "fgmres_completed_latch": latch if math.isfinite(latch) else None,
        "production_status_code": status_code,
        "raw_rap_reconstruction": {
            "formula": "augmented_effective_P^T @ augmented_AP; consumer LU is not raw RAP",
            "matrix": rap,
            "nonfinite_entry_count": rap_nonfinite,
        },
        "payload_sha256": payload_info["sha256"],
        "payload_bytes": len(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate_capture(args.manifest)
    except (CaptureError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        report = {"integrity": "fail", "reason": str(exc)}
        code = 2
    else:
        code = 0
    text = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
