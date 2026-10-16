#!/usr/bin/env python3
"""CPU contract tests for the v1 regional sidecar reader.

The fixtures are synthetic parser inputs, not native solver evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import read_regional_capture as reader  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pack_array(dtype: str, shape: list[int], element_bytes: int,
               data: bytes) -> dict:
    return {
        "name": "",
        "dtype": dtype,
        "semantic": "synthetic parser fixture only",
        "shape": shape,
        "offset": 0,
        "byte_count": len(data),
        "element_bytes": element_bytes,
        "sha256": sha(data),
    }


def payload_and_entries(specs: list[tuple[str, str, list[int], int, bytes]]):
    payload = bytearray()
    entries = []
    for name, dtype, shape, element_bytes, data in specs:
        payload.extend(b"\0" * ((-len(payload)) % 16))
        entry = pack_array(dtype, shape, element_bytes, data)
        entry["name"] = name
        entry["offset"] = len(payload)
        payload.extend(data)
        entries.append(entry)
    return bytes(payload), entries


def make_capture(directory: Path, *, augmented: bool = False,
                 graph: bool = False, selector: int | None = None,
                 block_inverse: bool = False,
                 primary_selected: bool = False) -> Path:
    root, substep, microtick, microcount, newton = 7, 2, 1, 3, 4
    source_fp = "00000000000000a1"
    world_fp = "00000000000000b2"
    program_fp = "00000000000000c3"
    device = {"name": "synthetic-fixture", "registry_id": 17}

    state_specs = [
        ("accepted_nodes", "record:NMFEMNodeStateGPU", [1, 1], 80, bytes(80)),
        ("candidate_nodes", "record:NMFEMNodeStateGPU", [1, 1], 80, bytes(80)),
        ("accepted_tetrahedra", "record:NMTetrahedronGPU", [1, 1], 96, bytes(96)),
        ("candidate_tetrahedra", "record:NMTetrahedronGPU", [1, 1], 96, bytes(96)),
        ("accepted_material_history", "float32", [1, 1, 1], 4, bytes(4)),
        ("candidate_material_state_buffer", "float32", [1, 1, 1], 4, bytes(4)),
        ("original_newton_correction", "float32x4", [2], 16, bytes(32)),
        ("base_residual_rhs", "float32x4", [2], 16, bytes(32)),
        ("algorithmic_tangent_cache", "float32", [1, 1, 9, 9], 4, bytes(324)),
        ("algorithmic_tangent_cache_valid", "uint32", [1, 1], 4, bytes(4)),
        ("fgmres_state", "record:NMFGMRESStateGPU", [1], 32, bytes(32)),
        ("armijo_metrics", "float32x4", [1], 16, bytes(16)),
    ]
    state_payload, state_entries = payload_and_entries(state_specs)
    state_name = "case.state.bin"
    state_manifest_name = "case.state.json"
    state_manifest = {
        "schema": reader.STATE_SCHEMA,
        "complete": True,
        "array_count": len(state_entries),
        "control_step": root,
        "physics_substep": substep,
        "microtick": microtick,
        "microtick_count": microcount,
        "micro_generation": 5,
        "newton_iteration": newton,
        "environment_count": 1,
        "node_count": 1,
        "tetrahedron_count": 1,
        "material_state_stride": 1,
        "unknown_count": 2,
        "source_physics_fingerprint_u64_hex": source_fp,
        "world_fingerprint_u64_hex": world_fp,
        "device_program_fingerprint_u64_hex": program_fp,
        "device": device,
        "dt_seconds": 0.01,
        "material_history_projection": {
            "current_newton_status": "unavailable_not_persisted",
        },
        "payload": {"file": state_name, "byte_count": len(state_payload),
                    "sha256": sha(state_payload)},
        "arrays": state_entries,
    }

    f = struct.pack("<f", 0.0)
    u = struct.pack("<I", 0)
    mode_stride = 2 if augmented else 1
    augmented_count = 2 if augmented else 0
    effective_words = mode_stride * 4
    actual_selector = selector if selector is not None else (5 if graph else 2)
    specs = [
        ("P", "float32x4", [1, 1], 16, struct.pack("<4f", 1, 0, 0, 0)),
        ("effective_P", "float32x4", [1, mode_stride, 1], 16,
         struct.pack("<" + "f" * effective_words,
                     *([1, 0, 0, 0] * mode_stride))),
        ("AP", "float32x4", [1, mode_stride, 1], 16,
         struct.pack("<" + "f" * effective_words,
                     *([1, 0, 0, 0] * mode_stride))),
        ("base_RAP_lu", "float32", [1, 1, 1], 4, f),
        ("base_RAP_pivots", "uint32", [1, 1], 4, u),
        ("base_RAP_valid", "uint32", [1], 4, u),
        ("base_coarse_rhs", "float32", [1, 1], 4, f),
        ("base_coarse_coefficients", "float32", [1, 1], 4, f),
    ]
    if actual_selector in (2, 3, 4):
        specs.extend([
            ("inverse_diagonal", "float32x4", [1, 1], 16,
             struct.pack("<4f", 1.0, 1.0, 1.0, 0.0)),
            ("node_ranges", "record:NMIncidenceRangeGPU", [1, 1], 16,
             struct.pack("<4I", 0, 1, 0, 0)),
        ])
    if actual_selector in (3, 4) and block_inverse:
        specs.append(("node_block_inverse", "float32x4", [1, 1, 3], 16,
                      struct.pack("<12f", 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0)))
    if primary_selected:
        specs.append(("primary_policy_selected_environment", "uint32", [1], 4,
                      struct.pack("<I", 1)))
    if augmented:
        specs.extend([
            ("augmented_RAP_lu", "float32", [1, 2, 2], 4,
             struct.pack("<4f", 1, 0, 0, 1)),
            ("augmented_RAP_pivots", "uint32", [1, 2], 4,
             struct.pack("<2I", 0, 1)),
            ("augmented_RAP_valid", "uint32", [1], 4, struct.pack("<I", 1)),
            ("augmented_coarse_rhs", "float32", [1, 2], 4,
             struct.pack("<2f", 0, 0)),
            ("augmented_coarse_coefficients", "float32", [1, 2], 4,
             struct.pack("<2f", 0, 0)),
            ("restart_generation", "uint32x4", [1], 16,
             struct.pack("<4I", root, substep, newton, 3)),
            ("restart_mode_valid", "uint32", [1], 4, struct.pack("<I", 1)),
        ])
    if graph:
        specs.extend([
            ("graph_patches", "uint32x4", [1], 16,
             struct.pack("<4I", 0, 1, 0, 0)),
            ("graph_patch_nodes", "uint32", [1], 4, struct.pack("<I", 0)),
            ("graph_rows", "uint32x4", [3], 16,
             struct.pack("<12I", 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0)),
            ("graph_node_patch_offsets", "uint32", [2], 4,
             struct.pack("<2I", 0, 1)),
            ("graph_node_patch_values", "uint32", [1], 4, struct.pack("<I", 0)),
            ("graph_node_coverage", "uint32", [1], 4, struct.pack("<I", 1)),
            ("graph_original_matrices", "float32", [1, 9], 4,
             struct.pack("<9f", *([0.0] * 9))),
            ("graph_patch_lu", "float32", [1, 9], 4,
             struct.pack("<9f", *([0.0] * 9))),
            ("graph_pivots", "uint32", [1, 3], 4,
             struct.pack("<3I", 0, 1, 2)),
            ("graph_factor_valid", "uint32", [1, 1], 4,
             struct.pack("<I", 1)),
        ])
    regional_payload, regional_entries = payload_and_entries(specs)
    regional_name = "case.regional.bin"
    regional_manifest_name = "case.regional.json"
    regional_manifest = {
        "schema": reader.SCHEMA,
        "complete": True,
        "byte_order": "little_endian",
        "capture_point": reader.EXPECTED_CAPTURE_POINT,
        "physical_state_changed": False,
        "array_count": len(regional_entries),
        "root": root,
        "control_step": root,
        "physics_substep": substep,
        "microtick": microtick,
        "microtick_count": microcount,
        "micro_generation": 5,
        "linearization_generation": f"{((5 << 32) | newton):016x}",
        "newton_iteration": newton,
        "environment_count": 1,
        "object_count": 1,
        "node_count": 1,
        "tetrahedron_count": 1,
        "material_state_stride": 1,
        "unknown_count": 2,
        "preconditioner_selector": actual_selector,
        "base_mode_count": 1,
        "mode_stride": mode_stride,
        "augmented_mode_count": augmented_count,
        "same_newton_restart_augmentation_enabled": augmented,
        "cgls_iteration_budget": 32,
        "dt_seconds": 0.01,
        "candidate_state_binding": {
            "schema": reader.STATE_SCHEMA,
            "manifest_file": state_manifest_name,
            "payload_file": state_name,
            "payload_sha256": sha(state_payload),
        },
        "source_physics_fingerprint_u64_hex": source_fp,
        "world_fingerprint_u64_hex": world_fp,
        "device_program_fingerprint_u64_hex": program_fp,
        "device": device,
        "graph_schwarz": {
            "enabled": graph,
            "patch_count": 1 if graph else 0,
            "patch_node_count": 1 if graph else 0,
            "dof_count": 3 if graph else 0,
            "matrix_float_count": 9 if graph else 0,
            "pivot_count": 3 if graph else 0,
            "layout_checksum_fnv1a32": 123 if graph else 0,
            "source_topology_sha256": "1" * 64 if graph else "",
            "source_free_mask_sha256": "2" * 64 if graph else "",
            "potential_node_mask_sha256": "3" * 64 if graph else "",
        },
        "primary_policy_selected_newton": primary_selected,
        "regional_apply_replay": {
            "row_mapping": "candidate_nodes are environment-major; restAndFixed.w is the raw fixed-node tag; node_ranges is included for selectors 2-4; selector 5 carries graph_rows",
            "fgmres_health": "read fgmres_state.diagnostics from the joined state payload",
            "fine_diagonal": "inverse_diagonal is captured for selectors 2-4",
            "fine_block": "node_block_inverse contains three float4 rows per node, including row0.w validity/fallback marker, for selectors 3-4",
            "native_arbitrary_vector_action": "not captured; coarse RHS and coefficients only describe the last recorded coarse input",
        },
        "payload": {"file": regional_name,
                    "byte_count": len(regional_payload),
                    "sha256": sha(regional_payload)},
        "arrays": regional_entries,
    }
    (directory / state_name).write_bytes(state_payload)
    (directory / state_manifest_name).write_text(
        json.dumps(state_manifest, separators=(",", ":")) + "\n",
        encoding="utf-8")
    (directory / regional_name).write_bytes(regional_payload)
    manifest_path = directory / regional_manifest_name
    manifest_path.write_text(
        json.dumps(regional_manifest, separators=(",", ":")) + "\n",
        encoding="utf-8")
    return manifest_path


class RegionalCaptureReaderTests(unittest.TestCase):
    def test_valid_minimal_join(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = make_capture(Path(temp))
            result = reader.read_capture(manifest)
            self.assertTrue(result["identity_joined"])
            self.assertFalse(result["graph_schwarz_captured"])
            self.assertFalse(result["native_or_physical_qualification"])

    def test_augmented_generation_is_explicitly_joined(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = make_capture(Path(temp), augmented=True)
            result = reader.read_capture(manifest)
            state = result["augmented_generation"][0]
            self.assertTrue(state["generation_matches_selected_newton"])
            self.assertTrue(state["augmented_data_admissible_for_same_newton"])

    def test_block_regional_apply_requires_diagonal_ranges_and_inverse_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = make_capture(Path(temp), selector=3, block_inverse=True)
            reader.read_capture(manifest)
            names = {item["name"] for item in json.loads(
                manifest.read_text(encoding="utf-8"))["arrays"]}
            self.assertTrue({"inverse_diagonal", "node_ranges",
                             "node_block_inverse"}.issubset(names))

    def test_block_regional_apply_missing_inverse_rows_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = make_capture(Path(temp), selector=3, block_inverse=True)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for entry in data["arrays"]:
                if entry["name"] == "node_block_inverse":
                    entry["name"] = "unbound_block_rows"
                    break
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(reader.CaptureError,
                                        "selector-3/4 block apply"):
                reader.read_capture(manifest)

    def test_primary_selection_words_are_required_only_when_selected(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = make_capture(Path(temp), primary_selected=True)
            self.assertTrue(reader.read_capture(manifest)["identity_joined"])
        with tempfile.TemporaryDirectory() as temp:
            manifest = make_capture(Path(temp), primary_selected=True)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for entry in data["arrays"]:
                if entry["name"] == "primary_policy_selected_environment":
                    entry["name"] = "unbound_selection"
                    break
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(reader.CaptureError,
                                        "per-environment selection words"):
                reader.read_capture(manifest)

    def test_nonstandard_json_constant_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = make_capture(Path(temp))
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace('"dt_seconds":0.01',
                                         '"dt_seconds":NaN'),
                            encoding="utf-8")
            with self.assertRaises(reader.CaptureError):
                reader.read_capture(path)

    def test_duplicate_json_key_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = make_capture(Path(temp))
            path.write_text('{"schema":"bad","schema":"bad"}',
                            encoding="utf-8")
            with self.assertRaises(reader.CaptureError):
                reader.read_capture(path)

    def test_absolute_payload_path_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = make_capture(Path(temp))
            data = json.loads(path.read_text(encoding="utf-8"))
            data["payload"]["file"] = str(path.parent / "case.regional.bin")
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(reader.CaptureError):
                reader.read_capture(path)

    def test_payload_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = make_capture(Path(temp))
            payload = path.parent / "case.regional.bin"
            payload.write_bytes(payload.read_bytes() + b"x")
            with self.assertRaises(reader.CaptureError):
                reader.read_capture(path)

    def test_state_identity_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = make_capture(Path(temp))
            data = json.loads(path.read_text(encoding="utf-8"))
            data["world_fingerprint_u64_hex"] = "00000000000000ff"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(reader.CaptureError):
                reader.read_capture(path)

    def test_linearization_generation_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = make_capture(Path(temp))
            data = json.loads(path.read_text(encoding="utf-8"))
            data["linearization_generation"] = "0000000500000005"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(reader.CaptureError):
                reader.read_capture(path)

    def test_missing_full_state_identity_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = make_capture(Path(temp))
            state_path = path.parent / "case.state.json"
            data = json.loads(state_path.read_text(encoding="utf-8"))
            next(entry for entry in data["arrays"]
                 if entry["name"] == "algorithmic_tangent_cache")["name"] = "other"
            state_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(reader.CaptureError):
                reader.read_capture(path)

    def test_node_record_abi_width_is_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            path = make_capture(Path(temp))
            state_path = path.parent / "case.state.json"
            data = json.loads(state_path.read_text(encoding="utf-8"))
            next(entry for entry in data["arrays"]
                 if entry["name"] == "candidate_nodes")["element_bytes"] = 96
            state_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(reader.CaptureError):
                reader.read_capture(path)

    def test_overlapping_array_extent_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = make_capture(Path(temp))
            data = json.loads(path.read_text(encoding="utf-8"))
            data["arrays"][1]["offset"] = data["arrays"][0]["offset"]
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(reader.CaptureError):
                reader.read_capture(path)

    def test_augmented_wrong_generation_not_admitted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = make_capture(Path(temp), augmented=True)
            data = json.loads(path.read_text(encoding="utf-8"))
            payload_path = path.parent / data["payload"]["file"]
            payload = bytearray(payload_path.read_bytes())
            entry = next(a for a in data["arrays"]
                         if a["name"] == "restart_generation")
            struct.pack_into("<I", payload, entry["offset"], 99)
            entry["sha256"] = sha(payload[entry["offset"]:
                                      entry["offset"] + entry["byte_count"]])
            payload_path.write_bytes(payload)
            data["payload"]["sha256"] = sha(payload)
            path.write_text(json.dumps(data), encoding="utf-8")
            result = reader.read_capture(path)
            self.assertFalse(result["augmented_generation"][0][
                "augmented_data_admissible_for_same_newton"])

    def test_selector5_graph_structure_is_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            path = make_capture(Path(temp), graph=True)
            result = reader.read_capture(path)
            self.assertTrue(result["graph_schwarz_captured"])

    def test_selector5_inconsistent_row_map_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = make_capture(Path(temp), graph=True)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            payload_path = path.parent / manifest["payload"]["file"]
            payload = bytearray(payload_path.read_bytes())
            row_entry = next(a for a in manifest["arrays"]
                             if a["name"] == "graph_rows")
            struct.pack_into("<I", payload, row_entry["offset"] + 12, 7)
            row_entry["sha256"] = sha(payload[row_entry["offset"]:
                                                row_entry["offset"] + row_entry["byte_count"]])
            payload_path.write_bytes(payload)
            manifest["payload"]["sha256"] = sha(payload)
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(reader.CaptureError):
                reader.read_capture(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
