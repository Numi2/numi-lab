#!/usr/bin/env python3
"""Synthetic algebra/admission tests for cached-operator reconstruction.

These cases test equations and fail-closed parsing only. They are not native
Matter measurements or cardboard validation.
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("cardboard_cached_operator_audit.py")
SPEC = importlib.util.spec_from_file_location("cached_operator_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def quat_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2.0)
    return [*(axis * s), float(np.cos(angle / 2.0))]


def direct_tet_force_derivative(inv, volume, tangent, quat, active, dv):
    rotation = audit.rotation_from_xyzw(quat) if active else np.eye(3)
    edges = np.column_stack((dv[1]-dv[0], dv[2]-dv[0], dv[3]-dv[0]))
    dF = edges @ inv
    local = dF @ rotation
    dP = (tangent @ local.reshape(9)).reshape(3, 3) @ rotation.T
    h = -volume * dP @ inv.T
    derivative = np.vstack((-h.sum(axis=1), h.T))
    return derivative


class LocalOperatorTests(unittest.TestCase):
    def test_captured_f32_roundtrip_is_distinct_from_generic_fp64_input(self):
        # This is the decimal spelling produced by a C++ stream at
        # max_digits10 for an FP32 value. JSON parses it as FP64, but the
        # original Metal operand was the nearest float32.
        wire_value = float(format(float(np.float32(0.2)), ".9g"))
        recovered = audit.captured_f32_array([wire_value], (1,), "captured float")[0]
        self.assertEqual(recovered, float(np.float32(0.2)))
        self.assertNotEqual(recovered, wire_value)
        # Authored/external FP64 references retain their original precision.
        generic = audit.finite_array([wire_value], (1,), "reference float")[0]
        self.assertEqual(generic, wire_value)
        self.assertNotEqual(generic, recovered)

    def test_captured_f32_rejects_overflow_after_json_parse(self):
        with self.assertRaisesRegex(ValueError, "finite FP32 range"):
            audit.captured_f32_array([1.0e300], (1,), "captured float")

    def test_unit_tangent_element_force_sign(self):
        block = audit.local_tet_stiffness(np.eye(3), 1.0/6.0, np.eye(9),
                                          [0, 0, 0, 1], False)
        self.assertAlmostEqual(block[3, 3], 1.0/6.0, places=14)
        self.assertAlmostEqual(block[0, 3], -1.0/6.0, places=14)
        self.assertAlmostEqual(block[3, 0], -1.0/6.0, places=14)

    def test_local_block_preserves_unsymmetric_tangent_and_frame(self):
        inv = np.array([[1.7, 0.2, -0.1], [0.1, 1.2, 0.3], [0.0, -0.2, 0.9]])
        volume = 0.37
        rng = np.random.default_rng(20261015)
        tangent = rng.normal(size=(9, 9))
        quat = quat_axis_angle([1.0, -2.0, 0.4], 0.71)
        block = audit.local_tet_stiffness(inv, volume, tangent, quat, True)
        self.assertGreater(np.linalg.norm(block - block.T), 1e-4)
        dv = rng.normal(size=(4, 3))
        expected = -direct_tet_force_derivative(inv, volume, tangent,
                                                quat, True, dv).reshape(12)
        actual = block @ dv.reshape(12)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)

    def test_inactive_frame_uses_world_coordinates(self):
        inv = np.array([[1.0, 0.0, 0.0], [0.1, 1.0, 0.0], [0.0, 0.2, 1.0]])
        tangent = np.arange(81, dtype=float).reshape(9, 9) / 31.0
        quat = [0.0, 0.0, 0.0, 0.0]  # canonical absent-frame payload
        no_frame = audit.local_tet_stiffness(inv, 0.5, tangent, quat, False)
        identity = audit.local_tet_stiffness(inv, 0.5, tangent, [0, 0, 0, 1], False)
        np.testing.assert_array_equal(no_frame, identity)

    def test_rejects_invalid_quaternion_and_nonfinite_cache(self):
        with self.assertRaisesRegex(ValueError, "quaternion"):
            audit.rotation_from_xyzw([0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "cached local tangent"):
            audit.local_tet_stiffness(np.eye(3), 1.0,
                                      np.full((9, 9), np.nan), [0, 0, 0, 1], False)


def synthetic_capture():
    nodes = []
    directions = [[0, 0, 0], [0.2, -0.1, 0.3], [-0.4, 0.2, 0.1], [0.1, 0.3, -0.2]]
    for i in range(4):
        mass = 2.0 + i
        tag = 1 if i == 0 else 0
        position = [float(i == 1), float(i == 2), float(i == 3)]
        raw_state = [position + [mass], [0.0, 0.0, 0.0, 0.0 if tag else 1.0/mass],
                     [0.0, 0.0, 0.0, float(tag)], [0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0]]
        nodes.append({"record": "node", "environment": 0, "node_id": i,
                      "center": {"mass_kg": mass, "constraint_tag": tag,
                                 "x_m": position, "raw_state_float4s": raw_state},
                      "correction_delta_v_m_s": directions[i],
                      "center_residual_impulse_Ns": [0.01*(i+1), -0.02*(i+1),
                                                       0.03*(i+1), 0.0],
                      "production_A_delta_v_impulse_Ns": [0.0, 0.0, 0.0, 0.0]})
    nodes[0]["production_A_delta_v_impulse_Ns"][:3] = [0, 0, 0]
    tangent = np.diag(np.arange(1, 10, dtype=float) * 10.0)
    q = [0, 0, 0, 1]
    inv = np.eye(3)
    local_k = audit.local_tet_stiffness(inv, 1/6, tangent, q, False)
    free_map = [3*i+a for i in range(1, 4) for a in range(3)]
    direction = np.asarray(directions, dtype=float)[1:].reshape(-1)
    dt = 0.25
    assembled = np.repeat([3.0, 4.0, 5.0], 3) * direction + dt*dt*(local_k[np.ix_(free_map, free_map)] @ direction)
    for i in range(1, 4):
        nodes[i]["production_A_delta_v_impulse_Ns"][:3] = assembled[3*(i-1):3*i].tolist()
    tet = {"record": "tetrahedron", "environment": 0, "tet_id": 0,
           "material_index": 0, "material_name": "synthetic_linear",
           "nodes": [0, 1, 2, 3], "rest_volume_m3": 1/6,
           "object_index": 0, "tet_flags": 1,
           "inverse_rest_rows_per_m": np.eye(3).tolist(),
           "material_frame_quaternion_xyzw": q,
           "center": {"dt_seconds_bulk_modulus_kind_frame_active": [dt, 0, 0, 0],
                      "projection": [0, 0, 1, 1],
                      "flags_terminal_status_parameter_count": [0, 0, 0, 0]},
           "cached_algorithmic_tangent_local": {
               "valid": True,
               "matrix_row_major_local_dP_dF_Pa_per_dimensionless_F": tangent.reshape(-1).tolist()}}
    header = {"record": "header", "schema": audit.CAPTURE_SCHEMA,
              "production_inputs_bitwise_unchanged": True,
              "environment_count": 1, "node_count_per_environment": 4,
              "tetrahedron_count_per_environment": 1,
              "dt_seconds_by_environment": [dt],
              "materials": [{"index": 0, "name": "synthetic_linear",
                             "parameter_offset": 0, "parameter_names": []}],
              "parameters_by_environment": [[]],
              "cached_algorithmic_tangent_capture": {
                  "policy_enabled": True, "available": True,
                  "snapshot_bitwise_guard_included": True,
                  "matrix_layout": "row-major 9x9, local material frame",
                  "derivative_contract": "dP_local/dF_local at captured linearization"}}
    integrity = {"record": "integrity", "production_inputs_bitwise_unchanged": True,
                 "cached_tangent_snapshot_bitwise_guard_included": True}
    return header, nodes, [tet], integrity


class AssemblyAdmissionTests(unittest.TestCase):
    def test_operator_assembly_uses_canonical_native_f32_operands(self):
        def wire_f32(value):
            return float(format(float(np.float32(value)), ".9g"))

        header, nodes, tets, integrity = synthetic_capture()
        mass32 = np.asarray([np.float32(wire_f32(2.1 + i * 0.7))
                             for i in range(4)], dtype=np.float64)
        for node_id, node in enumerate(nodes):
            node["center"]["mass_kg"] = wire_f32(mass32[node_id])
            node["center"]["raw_state_float4s"][0][3] = node["center"]["mass_kg"]
            position = [wire_f32(0.13 + 0.11 * node_id),
                        wire_f32(-0.27 + 0.09 * node_id),
                        wire_f32(0.31 - 0.07 * node_id)]
            node["center"]["x_m"] = position
            node["center"]["raw_state_float4s"][0][:3] = position

        inverse_f32 = np.asarray([
            [1.1, 0.04, -0.03], [0.02, 1.2, 0.06], [-0.01, 0.03, 0.9]
        ], dtype=np.float32)
        inverse_wire = [[wire_f32(x) for x in row] for row in inverse_f32]
        inverse64 = np.asarray(inverse_wire, dtype=np.float32).astype(np.float64)
        volume32 = np.float32(1.0 / (6.0 * abs(np.linalg.det(inverse64))))
        volume_wire = wire_f32(volume32)
        dt_wire = wire_f32(0.1234567)
        tangent32 = np.asarray(
            np.arange(1, 82, dtype=np.float32).reshape(9, 9) * 0.03125,
            dtype=np.float32)
        tangent_wire = [wire_f32(x) for x in tangent32.reshape(-1)]
        quaternion32 = np.asarray(quat_axis_angle([1.0, -2.0, 0.4], 0.71),
                                  dtype=np.float32)
        quaternion_wire = [wire_f32(x) for x in quaternion32]
        tet = tets[0]
        tet["inverse_rest_rows_per_m"] = inverse_wire
        tet["rest_volume_m3"] = volume_wire
        tet["material_frame_quaternion_xyzw"] = quaternion_wire
        tet["center"]["dt_seconds_bulk_modulus_kind_frame_active"] = [
            dt_wire, wire_f32(0.0), wire_f32(0.0), wire_f32(1.0)]
        tet["center"]["flags_terminal_status_parameter_count"] = [0, 0, 0, 0]
        tet["cached_algorithmic_tangent_local"][
            "matrix_row_major_local_dP_dF_Pa_per_dimensionless_F"] = tangent_wire
        header["dt_seconds_by_environment"] = [dt_wire]

        direction32 = np.asarray([
            [0.0, 0.0, 0.0], [0.13, -0.17, 0.21],
            [-0.31, 0.27, 0.19], [0.23, 0.11, -0.29]
        ], dtype=np.float32)
        for node_id, node in enumerate(nodes):
            node["correction_delta_v_m_s"] = [wire_f32(x)
                                                for x in direction32[node_id]]
        canonical_direction = direction32[1:].astype(np.float64).reshape(-1)
        canonical_inverse = np.asarray(inverse_wire, dtype=np.float32).astype(np.float64)
        canonical_tangent = np.asarray(tangent_wire, dtype=np.float32).astype(np.float64).reshape(9, 9)
        canonical_volume = float(np.float32(volume_wire))
        canonical_dt = float(np.float32(dt_wire))
        canonical_quaternion = np.asarray(quaternion_wire, dtype=np.float32).astype(np.float64)
        element = audit.local_tet_stiffness(canonical_inverse, canonical_volume,
                                            canonical_tangent, canonical_quaternion, True)
        free_map = np.arange(3, 12, dtype=np.int64)
        expected = np.diag(np.repeat(mass32[1:], 3)) + canonical_dt ** 2 * \
            element[np.ix_(free_map, free_map)]
        expected_action = expected @ canonical_direction
        for node_id in range(1, 4):
            nodes[node_id]["production_A_delta_v_impulse_Ns"] = [
                wire_f32(x) for x in expected_action[3 * (node_id - 1):3 * node_id]
            ] + [0.0]

        mass, stiffness, operator, direction, metadata = audit.validate_and_assemble(
            header, nodes, tets, integrity, 0)
        np.testing.assert_array_equal(mass.diagonal(), np.repeat(mass32[1:], 3))
        np.testing.assert_array_equal(operator.toarray(), expected)
        np.testing.assert_array_equal(direction, canonical_direction)
        np.testing.assert_array_equal(metadata["node_positions_m"][1],
                                      np.asarray(nodes[1]["center"]["x_m"], dtype=np.float32)
                                      .astype(np.float64))
        self.assertLess(metadata["action_parity"]["relative_l2_error"], 5e-8)

    def test_synthetic_one_tet_action_parity_and_fixed_elimination(self):
        header, nodes, tets, integrity = synthetic_capture()
        mass, stiffness, operator, direction, metadata = audit.validate_and_assemble(
            header, nodes, tets, integrity, 0)
        self.assertEqual(operator.shape, (9, 9))
        self.assertEqual(metadata["fixed_node_count"], 1)
        # The captured production action is itself an FP32 Metal output.
        self.assertLess(metadata["action_parity"]["relative_l2_error"], 1e-7)
        self.assertGreater(float(np.linalg.norm(stiffness @ direction)), 0.0)
        self.assertEqual(float(metadata["action_parity"]["fixed_rows_max_abs_native_action_Ns"]), 0.0)
        spectral = audit.spectral_diagnostics(operator, metadata, requested_modes=3)
        self.assertEqual(spectral["condition_estimate_status"], "iterative_estimate")
        self.assertEqual(len(spectral["small_right_singular_modes"]), 3)

    def test_refuses_incomplete_or_mutated_tangent_cache(self):
        header, nodes, tets, integrity = synthetic_capture()
        tets[0]["cached_algorithmic_tangent_local"]["valid"] = False
        with self.assertRaisesRegex(ValueError, "valid captured production tangent"):
            audit.validate_and_assemble(header, nodes, tets, integrity, 0)
        header, nodes, tets, integrity = synthetic_capture()
        integrity["cached_tangent_snapshot_bitwise_guard_included"] = False
        with self.assertRaisesRegex(ValueError, "mutation guard"):
            audit.validate_and_assemble(header, nodes, tets, integrity, 0)

    def test_rejects_bad_topology_and_fixed_direction(self):
        header, nodes, tets, integrity = synthetic_capture()
        tets[0]["nodes"] = [0, 1, 1, 3]
        with self.assertRaisesRegex(ValueError, "connectivity"):
            audit.validate_and_assemble(header, nodes, tets, integrity, 0)
        header, nodes, tets, integrity = synthetic_capture()
        nodes[0]["correction_delta_v_m_s"] = [1e-2, 0, 0]
        with self.assertRaisesRegex(ValueError, "fixed/gripped"):
            audit.validate_and_assemble(header, nodes, tets, integrity, 0)

    def test_binding_requires_sealed_source_selector_and_cache_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "source"
            prediction_dir = root / "run"
            snapshot.mkdir()
            prediction_dir.mkdir()
            source_hashes = {}
            for relative in audit.REQUIRED_BOUND_SOURCES:
                path = snapshot / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic source bytes for binding check\n")
                source_hashes[relative] = audit.sha256(path)
            capture = prediction_dir / "capture.jsonl"
            capture.write_text("synthetic capture bytes\n")
            prediction = prediction_dir / "prediction.json"
            prediction.write_text(json.dumps({
                "schema": "numi.cardboard.native-prediction.v1",
                "environment": {"NM_FEM_OPERATOR_FD_SELECTOR": "2:5"},
                "source_snapshot": str(snapshot),
                "inputs_sha256": source_hashes,
            }))
            stderr = prediction_dir / "stderr.log"
            stderr.write_text(
                'fem_operator_fd_enabled={"root":2,"iteration":5}\n'
                'fem_operator_fd_iteration={"root":2,"newton_iteration":5,'
                '"world_fingerprint":17,"execution_fingerprint":23,'
                '"cached_tangent_snapshot_available":true,'
                '"cached_tangent_snapshot_bitwise_guard_included":true,'
                '"production_inputs_bitwise_unchanged":true,'
                '"environments":[{"epsilon_pairs":[{"input_integrity_flag":1},'
                '{"input_integrity_flag":1}]}]}\n')
            receipt = prediction_dir / "receipt.json"
            receipt.write_text(json.dumps({
                "exit_code": 2, "timed_out": False,
                "prediction_sha256": audit.sha256(prediction),
                "capture_sha256": audit.sha256(capture),
            }))
            binding = audit.bind_inputs(capture, prediction, receipt, stderr, 0)
            self.assertEqual(binding["prediction_selector"], "2:5")
            self.assertEqual(len(binding["source_hashes_verified_against_prediction"]), 6)
            self.assertEqual(binding["receipt_exit_code"], 2)  # diagnostics can accompany failed runs

            bad = json.loads(prediction.read_text())
            bad["environment"]["NM_FEM_OPERATOR_FD_SELECTOR"] = "2:6"
            prediction.write_text(json.dumps(bad))
            receipt.write_text(json.dumps({"exit_code": 2, "timed_out": False,
                                           "prediction_sha256": audit.sha256(prediction),
                                           "capture_sha256": audit.sha256(capture)}))
            with self.assertRaisesRegex(ValueError, "scope admission"):
                audit.bind_inputs(capture, prediction, receipt, stderr, 0)

    def test_binding_resolves_current_source_runtime_and_external_namespaces(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            runtime = root / "runtime"
            run = root / "run"
            snapshot.mkdir(); runtime.mkdir(); run.mkdir()
            hashes = {}
            for relative in audit.REQUIRED_BOUND_SOURCES:
                path = snapshot / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("source " + relative)
                hashes["source/" + relative] = audit.sha256(path)
            runtime_file = runtime / "matter" / "probe"
            runtime_file.parent.mkdir(parents=True)
            runtime_file.write_text("runtime executable")
            hashes["runtime/matter/probe"] = audit.sha256(runtime_file)
            external = run / "external" / "source-build.json"
            external.parent.mkdir()
            external.write_text("external build receipt")
            hashes["external/source-build.json"] = audit.sha256(external)

            capture = run / "capture.jsonl"
            capture.write_text("capture")
            stderr = run / "stderr.log"
            stderr.write_text(
                'fem_operator_fd_enabled={"root":3,"iteration":23}\n'
                'fem_operator_fd_iteration={"root":3,"newton_iteration":23,'
                '"world_fingerprint":17,"execution_fingerprint":23,'
                '"cached_tangent_snapshot_available":true,'
                '"cached_tangent_snapshot_bitwise_guard_included":true,'
                '"production_inputs_bitwise_unchanged":true,'
                '"environments":[{"epsilon_pairs":[{"input_integrity_flag":1},'
                '{"input_integrity_flag":1}]}]}\n')
            prediction = run / "prediction.json"
            prediction.write_text(json.dumps({
                "schema": "numi.cardboard.native-prediction.v1",
                "environment": {"NM_FEM_OPERATOR_FD_SELECTOR": "3:23"},
                "source_snapshot": str(snapshot),
                "runtime_dir": str(runtime),
                "inputs_sha256": hashes,
                "external_inputs": [str(external)],
            }))
            receipt = run / "receipt.json"
            receipt.write_text(json.dumps({"exit_code": 2, "timed_out": False}))
            binding = audit.bind_inputs(capture, prediction, receipt, stderr, 0)
            self.assertEqual(len(binding["source_hashes_verified_against_prediction"]), 6)
            self.assertEqual(binding["runtime_and_external_input_hashes_verified_against_prediction"][
                "runtime/matter/probe"]["location"], "runtime_dir")
            self.assertEqual(binding["runtime_and_external_input_hashes_verified_against_prediction"][
                "external/source-build.json"]["location"], "external_input")


if __name__ == "__main__":
    unittest.main()
