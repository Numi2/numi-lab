from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "state_capture_reader", ROOT / "tools/read_state_capture.py")
assert SPEC is not None and SPEC.loader is not None
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def valid_pair(directory: Path) -> Path:
    payload = bytearray()
    arrays: list[dict[str, object]] = []

    def add(name: str, dtype: str, shape: list[int], element_bytes: int) -> None:
        while len(payload) % 16:
            payload.append(0)
        data = bytes(element_bytes * __import__("math").prod(shape))
        offset = len(payload)
        payload.extend(data)
        arrays.append({
            "name": name,
            "dtype": dtype,
            "semantic": "test fixture only; no native evidence",
            "shape": shape,
            "offset": offset,
            "byte_count": len(data),
            "element_bytes": element_bytes,
            "sha256": hashlib.sha256(data).hexdigest(),
        })

    add("accepted_nodes", "record:NMFEMNodeStateGPU", [1, 1], 80)
    add("candidate_nodes", "record:NMFEMNodeStateGPU", [1, 1], 80)
    add("accepted_tetrahedra", "record:NMTetrahedronGPU", [1, 1], 112)
    add("candidate_tetrahedra", "record:NMTetrahedronGPU", [1, 1], 112)
    add("accepted_material_history", "float32", [1, 1, 1], 4)
    add("candidate_material_state_buffer", "float32", [1, 1, 1], 4)
    add("objects", "record:NMContinuumObjectGPU", [1], 4)
    add("materials", "record:NMMaterialGPU", [1], 4)
    add("scalar_programs", "record:NMScalarProgramGPU", [1], 4)
    add("expression_instructions", "record:NMExpressionInstructionGPU", [1], 4)
    add("environment_parameters", "float32", [1, 1], 4)
    add("schedulers", "record:NMSchedulerStateGPU", [1, 1], 4)
    add("adaptive_state", "record:NMAdaptiveStateGPU", [1, 1], 4)
    add("original_newton_correction", "float32x4", [2], 16)
    add("base_residual_rhs", "float32x4", [2], 16)
    add("algorithmic_tangent_cache", "float32", [1, 1, 9, 9], 4)
    add("algorithmic_tangent_cache_valid", "uint32", [1, 1], 4)
    add("fgmres_state", "record:NMFGMRESStateGPU", [1], 4)
    add("mixed_solver", "record:NMMixedSolverGPU", [1], 4)
    add("armijo_metrics", "float32x4", [1], 16)
    add("armijo_statuses", "record:NMMatterStatusGPU", [1], 4)
    add("production_statuses", "record:NMMatterStatusGPU", [1], 4)
    add("fem_line_search", "float32x4", [1, 1], 16)
    add("environment_line_search", "float32", [1], 4)

    binary = directory / "input.state.bin"
    binary.write_bytes(payload)
    manifest = {
        "schema": reader.SCHEMA,
        "complete": True,
        "byte_order": "little_endian",
        "capture_point": reader.CAPTURE_POINT,
        "physical_state_changed": False,
        "root": 7,
        "control_step": 7,
        "physics_substep": 0,
        "microtick": 0,
        "microtick_count": 1,
        "newton_iteration": 2,
        "environment_count": 1,
        "object_count": 1,
        "material_count": 1,
        "parameter_count": 1,
        "node_count": 1,
        "tetrahedron_count": 1,
        "material_state_stride": 1,
        "unknown_count": 2,
        "cgls_iterations": 32,
        "array_count": 24,
        "mixed_solver_preconditioner": 0,
        "cgls_capture_file": "input.cgls",
        "microstep_flags": 0,
        "dispatch_abi_version": 1,
        "dispatch_flags": 0,
        "rigid_generalized_capacity": 0,
        "layout": {
            "unknown_count": 2,
            "mechanical_rows": "first E*N float4 entries",
            "field_backing_rows": "second E*N float4 entries; CGLS requires exact pure-FEM layout and preserves captured bits",
        },
        "dt_seconds": 0.01,
        "inverse_dt_seconds": 100.0,
        "source_physics_fingerprint_u64_hex": "0123456789abcdef",
        "world_fingerprint_u64_hex": "0123456789abcdef",
        "device_program_fingerprint_u64_hex": "0123456789abcdef",
        "device": {"name": "CPU schema fixture", "registry_id": 1},
        "tangent_cache": {
            "enabled_by_solver_policy": True,
            "force_rebuild_requested": False,
            "valid_entry_count": 1,
            "generation_claim": "raw bytes; not independently adjudicated",
        },
        "material_history_projection": {
            "current_newton_status": "unavailable_not_persisted",
            "candidate_buffer_name": "candidate_material_state_buffer",
            "candidate_buffer_is_current_projection": False,
            "reason": "current projection not persisted",
        },
        "payload": {
            "file": binary.name,
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "arrays": arrays,
    }
    manifest_path = directory / "input.state.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class StateCaptureReaderTests(unittest.TestCase):
    def test_valid_schema_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = valid_pair(Path(directory))
            result = reader.validate_capture(manifest)
            self.assertEqual(result["array_count"], 24)
            self.assertEqual(result["projected_history_status"], "unavailable_not_persisted")

    def test_absolute_payload_path_is_rejected_even_if_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = valid_pair(Path(directory))
            doc = json.loads(manifest.read_text())
            doc["payload"]["file"] = str(manifest.parent / "input.state.bin")
            manifest.write_text(json.dumps(doc))
            with self.assertRaisesRegex(reader.CaptureError, "adjacent basename"):
                reader.validate_capture(manifest)

    def test_missing_adjacent_payload_has_no_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = valid_pair(Path(directory))
            doc = json.loads(manifest.read_text())
            original = manifest.parent / doc["payload"]["file"]
            doc["payload"]["file"] = "missing.bin"
            manifest.write_text(json.dumps(doc))
            original.rename(manifest.parent / "elsewhere.bin")
            with self.assertRaisesRegex(reader.CaptureError, "no path fallback"):
                reader.validate_capture(manifest)

    def test_duplicate_json_keys_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = valid_pair(Path(directory))
            manifest.write_text('{"schema":"x","schema":"y"}')
            with self.assertRaisesRegex(reader.CaptureError, "duplicate JSON key"):
                reader.validate_capture(manifest)

    def test_nonstandard_json_number_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = valid_pair(Path(directory))
            text = manifest.read_text().replace('"dt_seconds": 0.01', '"dt_seconds": NaN')
            manifest.write_text(text)
            with self.assertRaisesRegex(reader.CaptureError, "nonstandard JSON"):
                reader.validate_capture(manifest)

    def test_candidate_state_cannot_claim_current_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = valid_pair(Path(directory))
            doc = json.loads(manifest.read_text())
            doc["material_history_projection"]["candidate_buffer_is_current_projection"] = True
            manifest.write_text(json.dumps(doc))
            with self.assertRaisesRegex(reader.CaptureError, "must not be misidentified"):
                reader.validate_capture(manifest)

    def test_payload_and_array_hashes_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = valid_pair(Path(directory))
            payload = manifest.parent / "input.state.bin"
            payload.write_bytes(payload.read_bytes() + b"x")
            with self.assertRaisesRegex(reader.CaptureError, "byte count mismatch"):
                reader.validate_capture(manifest)

    def test_wrong_fixed_record_width_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = valid_pair(Path(directory))
            doc = json.loads(manifest.read_text())
            doc["arrays"][0]["element_bytes"] = 72
            doc["arrays"][0]["byte_count"] = 72
            manifest.write_text(json.dumps(doc))
            with self.assertRaisesRegex(reader.CaptureError, "unexpected element width"):
                reader.validate_capture(manifest)


if __name__ == "__main__":
    unittest.main()
