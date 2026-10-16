#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).parents[1] / "tools" / "validate_consumer_capture.py"
SPEC = importlib.util.spec_from_file_location("consumer_capture", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def packed_u32(*values: int) -> bytes:
    return struct.pack("<" + "I" * len(values), *values)


def packed_f32(*values: float) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


def packed_vec4(*rows: tuple[float, float, float, float]) -> bytes:
    return b"".join(struct.pack("<4f", *row) for row in rows)


def capture_arrays(generation=(5, 2, 3, 0), mode_valid=1, tail=0):
    zero_vec = (0.0, 0.0, 0.0, 0.0)
    return [
        ("v", "<f4x4", [2], 16, packed_vec4(zero_vec, zero_vec)),
        ("z", "<f4x4", [2], 16, packed_vec4(zero_vec, zero_vec)),
        ("augmented_effective_P", "<f4x4", [2, 2], 16,
         packed_vec4((1, 0, 0, 0), (0, 0, 0, 0),
                     (0, 0, 0, 0), (1, 0, 0, 0))),
        ("augmented_AP", "<f4x4", [2, 2], 16,
         packed_vec4((2, 0, 0, 0), (4, 0, 0, 0),
                     (1, 0, 0, 0), (3, 0, 0, 0))),
        ("augmented_coarse_lu", "<f4", [2, 2], 4, packed_f32(2, 1, 2, 1)),
        ("augmented_coarse_pivots", "<u4", [2], 4, packed_u32(0, 1)),
        ("augmented_coarse_valid", "<u4", [1], 4, packed_u32(1)),
        ("augmented_coarse_rhs", "<f4", [2], 4, packed_f32(1, 2)),
        ("augmented_coarse_coefficients", "<f4", [2], 4, packed_f32(0.5, 1)),
        ("graph_patch_lu", "<f4", [1], 4, packed_f32(1)),
        ("graph_patch_original_matrices", "<f4", [1], 4, packed_f32(1)),
        ("graph_patch_pivots", "<u4", [1], 4, packed_u32(0)),
        ("graph_patch_factor_valid", "<u4", [1], 4, packed_u32(1)),
        ("graph_patch_corrections", "<f4x4", [1], 16, packed_vec4(zero_vec)),
        ("graph_patches", "<u4x4", [1], 16, packed_u32(0, 1, 0, 0)),
        ("graph_patch_nodes", "<u4", [1], 4, packed_u32(0)),
        ("graph_rows", "<u4x4", [1], 16, packed_u32(0, 1, 0, 0)),
        ("graph_node_patch_offsets", "<u4", [3], 4, packed_u32(0, 1, 1)),
        ("graph_node_patch_values", "<u4", [1], 4, packed_u32(0)),
        ("graph_node_coverage", "<u4", [2], 4, packed_u32(1, 0)),
        ("candidate_nodes", "struct:NMFEMNodeStateGPU", [2], 16,
         packed_vec4((0, 0, 0, 0), (0, 0, 0, 1))),
        ("linearization_nodes", "struct:NMFEMNodeStateGPU", [2], 16,
         packed_vec4((0, 0, 0, 0), (0, 0, 0, 1))),
        ("accepted_material_history", "<f4", [1, 1], 4, packed_f32(0)),
        ("scheduler_exponents", "<u4", [1], 4, packed_u32(0)),
        ("deformation_high", "<f4x4", [1, 3], 16,
         packed_vec4(zero_vec, zero_vec, zero_vec)),
        ("deformation_low", "<f4x4", [1, 3], 16,
         packed_vec4(zero_vec, zero_vec, zero_vec)),
        ("fgmres_state", "struct:NMFGMRESStateGPU", [1], 32,
         packed_f32(1, 1, 0, 4, 0, 0, 0, 0)),
        ("production_status", "struct:NMMatterStatusGPU", [1], 48,
         packed_u32(*([0] * 12))),
        ("restart_generation", "<u4", [4], 4, packed_u32(*generation)),
        ("restart_raw_norm_squared", "<f4", [1], 4, packed_f32(1)),
        ("restart_target_scale", "<f4", [1], 4, packed_f32(1)),
        ("restart_mode_scale", "<f4", [1], 4, packed_f32(1)),
        ("restart_mode_valid", "<u4", [1], 4, packed_u32(mode_valid)),
        ("restart_tail_preserved", "<u4", [1], 4, packed_u32(tail)),
    ]


def write_capture(directory: Path, values=None) -> Path:
    values = values or capture_arrays()
    payload = bytearray()
    arrays = []
    for name, dtype, shape, elem, raw in values:
        offset = len(payload)
        payload.extend(raw)
        arrays.append({"name": name, "dtype": dtype, "shape": shape,
                       "offset": offset, "byte_count": len(raw),
                       "element_bytes": elem,
                       "sha256": hashlib.sha256(raw).hexdigest()})
    binary = directory / "sample.consumer.bin"
    binary.write_bytes(payload)
    metadata = {
        "schema": MODULE.SCHEMA,
        "capture_point": "after_graph_apply_barrier_before_fem_operator_consumes_z",
        "preconditioner_selector": 5,
        "augmentation_enabled": True,
        "control_step": 5, "physics_substep": 2,
        "physics_substep_count": 4,
        "microtick": 0, "microtick_count": 1,
        "micro_generation": 23, "maximum_rate_exponent": 0,
        "selector_attempted": True, "selector_attempt_count": 1,
        "newton_iteration": 3,
        "restart_cycle": 1, "arnoldi_column_in_cycle": 0,
        "global_arnoldi_column": 16, "environment": 0,
        "environment_count": 1, "base_mode_count": 1,
        "mode_count": 2, "mode_stride": 2,
        "expected_generator_tag": [5, 2, 3, 0],
        "payload": {"file": binary.name, "byte_count": len(payload),
                    "aggregate_cap_bytes": MODULE.CAP_BYTES,
                    "sha256": hashlib.sha256(payload).hexdigest()},
        "arrays": arrays,
    }
    manifest = directory / "sample.consumer.json"
    manifest.write_text(json.dumps(metadata, sort_keys=True))
    return manifest


class CaptureValidatorTests(unittest.TestCase):
    def test_valid_capture_reconstructs_raw_nonsymmetric_rap(self):
        with tempfile.TemporaryDirectory() as temp:
            result = MODULE.validate_capture(write_capture(Path(temp)))
        self.assertEqual(result["integrity"], "pass")
        self.assertEqual(result["consumer_state"], "valid_augmented_map_inputs")
        self.assertEqual(result["raw_rap_reconstruction"]["matrix"],
                         [[2.0, 1.0], [4.0, 3.0]])

    def test_invalid_generation_is_captured_not_promoted(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = write_capture(Path(temp), capture_arrays(generation=(5, 2, 3, 9)))
            result = MODULE.validate_capture(manifest)
        self.assertEqual(result["integrity"], "pass")
        self.assertEqual(result["consumer_state"], "captured_invalid_or_fallback_state")
        self.assertFalse(result["generation_matches_expected"])

    def test_invalid_mode_and_preserved_tail_remain_observable(self):
        for values, expected_field, expected_value in (
            (capture_arrays(mode_valid=0), "restart_mode_valid", 0),
            (capture_arrays(tail=1), "restart_tail_preserved", 1),
        ):
            with tempfile.TemporaryDirectory() as temp:
                result = MODULE.validate_capture(write_capture(Path(temp), values))
            self.assertEqual(result["consumer_state"], "captured_invalid_or_fallback_state")
            self.assertEqual(result[expected_field], expected_value)

    def test_payload_corruption_fails_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = write_capture(Path(temp))
            binary = manifest.parent / "sample.consumer.bin"
            changed = bytearray(binary.read_bytes())
            changed[-1] ^= 1
            binary.write_bytes(changed)
            with self.assertRaises(MODULE.CaptureError):
                MODULE.validate_capture(manifest)

    def test_multitick_or_duplicate_event_metadata_rejects(self):
        for changes, reason in (
            ({"maximum_rate_exponent": 1, "microtick_count": 2},
             "capture_outside_zero_rate_single_microtick_scope"),
            ({"microtick": 1},
             "capture_outside_zero_rate_single_microtick_scope"),
            ({"selector_attempt_count": 2},
             "capture_not_unique_one_shot_event"),
            ({"selector_attempted": False},
             "capture_not_unique_one_shot_event"),
            ({"micro_generation": 24},
             "micro_generation_does_not_match_selected_single_tick"),
        ):
            with tempfile.TemporaryDirectory() as temp:
                path = write_capture(Path(temp))
                metadata = json.loads(path.read_text())
                metadata.update(changes)
                path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(MODULE.CaptureError, reason):
                    MODULE.validate_capture(path)

    def test_v2_capture_bound_is_64_mib(self):
        self.assertEqual(MODULE.CAP_BYTES, 64 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
