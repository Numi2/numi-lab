#!/usr/bin/env python3
"""Manufactured algebra/format tests; never native solver qualification."""
import importlib.util
import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from scipy import sparse


MODULE = Path(__file__).with_name("cardboard_offline_conditioning_study.py")
SPEC = importlib.util.spec_from_file_location("offline_conditioning", MODULE)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


class SolverAndPatchTests(unittest.TestCase):
    def test_rowwise_fsum_matches_finite_matrix_product(self):
        rng = np.random.default_rng(8)
        matrix = rng.normal(size=(257, 16)) * 100.0
        vector = rng.normal(size=16) * 1.0e-4
        actual = study.rowwise_fsum_matvec(matrix, vector)
        expected = np.einsum("ij,j->i", matrix, vector, optimize=False)
        np.testing.assert_allclose(actual, expected, rtol=5e-15, atol=5e-16)

    def test_graph_patches_cover_only_free_nodes_and_are_deterministic(self):
        topology = {"nodes": np.asarray([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]),
                    "node_count": np.asarray(6)}
        free = np.asarray([1, 2, 3, 4])
        p0, m0 = study.build_graph_patches(topology, free, 3, 2)
        p1, m1 = study.build_graph_patches(topology, free, 3, 2)
        self.assertEqual([p.tolist() for p in p0], [p.tolist() for p in p1])
        np.testing.assert_array_equal(m0, m1)
        self.assertTrue(np.all(m0[free] >= 1))
        self.assertEqual(int(m0[0]), 0)
        self.assertEqual(int(m0[5]), 0)
        self.assertLessEqual(max(map(len, p0)), 3)

    def test_schwarz_and_restarted_gmres_report_true_residual(self):
        # Six-node graph, node-major xyz with a stable positive mass diagonal.
        scalar = sparse.diags([2.0, 2.5, 3.0, 3.5, 4.0, 4.5]).toarray()
        scalar += np.diag([-0.2] * 5, 1) + np.diag([-0.1] * 5, -1)
        A = sparse.kron(sparse.csr_matrix(scalar), sparse.eye(3), format="csr")
        topology = {"nodes": np.asarray([[0, 1, 2, 3], [1, 2, 3, 4],
                                         [2, 3, 4, 5]]), "node_count": np.asarray(6)}
        free = np.arange(6)
        patches, coverage = study.build_graph_patches(topology, free, 3, 2)
        apply, details = study.schwarz_preconditioner(A, free, patches, coverage)
        b = np.arange(1, 19, dtype=float)
        result = study.restarted_right_gmres(A, b, apply, restart=16,
                                              max_iterations=64,
                                              relative_tolerance=1e-10)
        self.assertLess(result["final_true_relative_residual"], 1e-9)
        self.assertEqual(result["true_residual_history"][0]["iteration"], 0)
        self.assertGreater(details["estimated_factorization_flops"], 0)
        self.assertLess(details["dense_factor_bytes_estimate"], study.MAX_HOST_BYTES)

    def test_balanced_rap_application_uses_exact_native_lu_and_block(self):
        # A=diag(2,3,4), with one node and one coarse basis vector.
        A = sparse.diags([2.0, 3.0, 4.0], format="csr")
        P = np.asarray([[1.0], [0.0], [0.0]])
        rap = {"P_free": P, "AP_free": np.asarray([[2.0], [0.0], [0.0]]),
               "pivoted_lu": np.asarray([[2.0]]), "pivots": np.asarray([0]),
               "inverse_blocks": np.asarray([np.diag([0.5, 1/3, 0.25])]),
               "mode_count": 1}
        apply = study.native_balanced_rap_preconditioner(A, rap)
        r = np.asarray([4.0, 6.0, 8.0])
        # Coarse direction is removed exactly; fine node inverse handles y/z.
        np.testing.assert_allclose(apply(r), [2.0, 2.0, 2.0], atol=1e-14)


class NativeRAPFormatTests(unittest.TestCase):
    def test_same_candidate_native_export_and_ap_lu_parity(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            state = np.zeros((2, 20), dtype="<f4")
            state[0, 3] = 1.0
            state[0, 11] = 1.0  # static node
            state[1, 3] = 2.0
            state[1, 7] = 0.5
            P = np.zeros((1, 2, 4), dtype="<f4")
            P[0, 1, 0] = 1.0
            effective = P[None].copy()
            AP = np.zeros_like(effective)
            AP[0, 0, 1, 0] = 2.0
            lu = np.asarray([[[2.0]]], dtype="<f4")
            piv = np.asarray([[0]], dtype="<u4")
            valid = np.asarray([1], dtype="<u4")
            blocks = np.zeros((1, 2, 3, 4), dtype="<f4")
            blocks[0, 1, 0] = [0.5, 0, 0, 1]
            blocks[0, 1, 1] = [0, 1/3, 0, 0]
            blocks[0, 1, 2] = [0, 0, 0.25, 0]
            arrays = {"candidate_nodes": state, "P": P,
                      "effective_P": effective, "AP": AP,
                      "pivoted_RAP": lu, "pivots": piv,
                      "factor_valid": valid, "node_block_inverse": blocks}
            inventory = []
            for name, array in arrays.items():
                path = folder / f"{name}.bin"
                payload = array.tobytes(order="C")
                path.write_bytes(payload)
                import hashlib
                stride = {"candidate_nodes": 80, "pivots": 4,
                          "factor_valid": 4, "pivoted_RAP": 4}.get(name, 16)
                inventory.append({"name": path.name,
                                  "count": len(payload) // stride,
                                  "stride_bytes": stride,
                                  "bytes": len(payload), "sha256": study.sha256(path)})
            manifest = {"schema": "numi.cardboard.regional-conditioning.v1",
                        "accepted_physical_state": False, "control_step": 2,
                        "source_physics_fingerprint": "17",
                        "device_program_fingerprint": "23",
                        "environment_count": 1, "fem_node_count": 2,
                        "mode_count": 1, "byte_order": "native_little_endian",
                        "files": inventory}
            (folder / "manifest.json").write_text(json.dumps(manifest))
            vectors = {"captured_node_state_f32": state,
                       "free_node_ids": np.asarray([1])}
            report = {"capture_selection": {"environment": 0},
                      "binding": {"operator_fd_iteration":
                                  {"world_fingerprint": 17,
                                   "execution_fingerprint": 23}}}
            A = sparse.diags([2.0, 3.0, 4.0], format="csr")
            loaded, hashes = study.load_native_rap(folder, report, A, vectors)
            self.assertAlmostEqual(loaded["native_ap_relative_error"], 0.0)
            self.assertAlmostEqual(loaded["native_lu_reconstruction_relative_error"], 0.0)
            self.assertIn("manifest.json", hashes)

    def test_wrong_candidate_state_is_rejected(self):
        # Pairing is tested directly here; the native reader test covers bytes.
        report = {"capture_selection": {"root": 2, "newton_iteration": 7,
                                         "environment": 0},
                  "binding": {"operator_fd_iteration":
                              {"world_fingerprint": 17,
                               "execution_fingerprint": 23},
                              "capture_run_fingerprints_match": True},
                  "operator": {"node_count": 2,
                               "action_parity": {"relative_l2_error": 0.0}}}
        matrix = sparse.eye(3, format="csr")
        vectors = {"free_node_ids": np.asarray([1]),
                   "native_rhs_free_impulse_Ns": np.ones(3),
                   "captured_node_state_f32": np.zeros((2, 20), dtype=np.float32)}
        topology = {"nodes": np.asarray([[0, 1, 1, 0]])}
        rap = {"control_step": 2, "environment_count": 1, "fem_node_count": 2,
               "source_physics_fingerprint": "17",
               "device_program_fingerprint": "23"}
        result = study.validate_pairing(report, matrix, vectors, topology, rap)
        self.assertEqual(result["newton_iteration"], 7)
        report["binding"]["capture_run_fingerprints_match"] = False
        with self.assertRaisesRegex(ValueError, "source-bound stderr"):
            study.validate_pairing(report, matrix, vectors, topology, rap)
        report["binding"]["capture_run_fingerprints_match"] = True
        rap["control_step"] = 3
        with self.assertRaisesRegex(ValueError, "control step"):
            study.validate_pairing(report, matrix, vectors, topology, rap)

    def test_complete_synthetic_same_matrix_comparison_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audit_path = root / "audit.json"
            rap_dir = root / "rejected-conditioning-000003"
            rap_dir.mkdir()
            node_count, modes = 5, 1
            state = np.zeros((node_count, 20), dtype=np.float32)
            state[0, 3] = 1.0
            state[0, 11] = 1.0
            for node in range(1, node_count):
                state[node, 3] = 1.0
                state[node, 7] = 1.0
            A = sparse.eye(12, format="csr", dtype=np.float64) * 2.0
            vectors = {
                "direction": np.ones(12), "native_action_Ns": np.ones(12),
                "assembled_action_Ns": np.ones(12), "action_difference_Ns": np.zeros(12),
                "native_rhs_free_impulse_Ns": np.arange(1, 13, dtype=np.float64),
                "free_node_ids": np.asarray([1, 2, 3, 4], dtype=np.int64),
                "node_positions_m": np.zeros((node_count, 3)),
                "captured_node_state_f32": state,
            }
            topology = {"nodes": np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64),
                        "material_indices": np.zeros(2, dtype=np.int64),
                        "rest_volumes_m3": np.ones(2),
                        "inverse_rest_rows_per_m": np.tile(np.eye(3), (2, 1, 1)),
                        "node_count": np.asarray(node_count, dtype=np.int64)}
            sparse.save_npz(root / "audit.operator.npz", A)
            np.savez_compressed(root / "audit.vectors.npz", **vectors)
            np.savez_compressed(root / "audit.topology.npz", **topology)
            operator_files = {name: study.sha256(root / name)
                              for name in ("audit.operator.npz", "audit.vectors.npz",
                                           "audit.topology.npz")}
            report = {
                "schema": "numi.cardboard.cached-operator-audit.v1",
                "capture_selection": {"root": 3, "newton_iteration": 23, "environment": 0},
                "binding": {"operator_fd_iteration": {"world_fingerprint": 17,
                                                         "execution_fingerprint": 23},
                            "capture_run_fingerprints_match": True},
                "operator": {"node_count": node_count,
                             "action_parity": {"relative_l2_error": 0.0}},
                "matrix_artifacts": {"files_sha256": operator_files},
            }
            audit_path.write_text(json.dumps(report))
            P = np.zeros((modes, node_count, 4), dtype="<f4")
            P[0, 1:, 0] = 1.0
            effective = P[None].copy()
            AP = np.zeros_like(effective)
            AP[0, 0, 1:, 0] = 2.0
            lu = np.asarray([[[8.0]]], dtype="<f4")
            piv = np.asarray([[0]], dtype="<u4")
            valid = np.asarray([1], dtype="<u4")
            blocks = np.zeros((1, node_count, 3, 4), dtype="<f4")
            for node in range(1, node_count):
                blocks[0, node, 0] = [0.5, 0.0, 0.0, 1.0]
                blocks[0, node, 1] = [0.0, 0.5, 0.0, 0.0]
                blocks[0, node, 2] = [0.0, 0.0, 0.5, 0.0]
            arrays = {"candidate_nodes": state.astype("<f4"), "P": P,
                      "effective_P": effective, "AP": AP,
                      "pivoted_RAP": lu, "pivots": piv,
                      "factor_valid": valid, "node_block_inverse": blocks}
            inventory = []
            for name, array in arrays.items():
                path = rap_dir / f"{name}.bin"
                path.write_bytes(array.tobytes())
                stride = {"candidate_nodes": 80, "pivoted_RAP": 4,
                          "pivots": 4, "factor_valid": 4}.get(name, 16)
                inventory.append({"name": path.name, "count": path.stat().st_size // stride,
                                  "stride_bytes": stride, "bytes": path.stat().st_size,
                                  "sha256": study.sha256(path)})
            manifest = {"schema": "numi.cardboard.regional-conditioning.v1",
                        "accepted_physical_state": False, "control_step": 3,
                        "source_physics_fingerprint": "17",
                        "device_program_fingerprint": "23",
                        "environment_count": 1, "fem_node_count": node_count,
                        "mode_count": modes, "byte_order": "native_little_endian",
                        "files": inventory}
            (rap_dir / "manifest.json").write_text(json.dumps(manifest))
            args = Namespace(operator_report=str(audit_path), rap_export=str(rap_dir),
                             restart=16, max_iterations=16, relative_tolerance=1e-10,
                             schwarz_patch_nodes=4, schwarz_anchor_stride=2)
            result = study.run(args)
            self.assertTrue(result["fixed_physical_operator"]["same_A_and_b_for_both_methods"])
            self.assertLess(result["gmres_restart16_balanced_native_rap"][
                "final_true_relative_residual"], 1e-10)
            self.assertLess(result["gmres_restart16_bounded_graph_schwarz"][
                "final_true_relative_residual"], 1e-10)


class SelectedNewtonRegionalSidecarTests(unittest.TestCase):
    def _fixture(self, folder: Path, *, iteration=23, capture_point=None,
                 corrupt_payload=False, wrong_low=False, wrong_candidate=False, node_words=20):
        folder.mkdir(parents=True, exist_ok=True)
        envs, nodes, tets, modes, stride = 1, 5, 2, 1, 13
        state = np.zeros((envs, nodes, node_words), dtype="<f4")
        state[0, :, 3] = 1.0
        state[0, 0, 11] = 1.0
        if node_words == 24:
            state[0, :, 20] = 2.0**-40
        captured_state = state.copy()
        if wrong_candidate:
            state[0, 1, 0] = 1.0
        states = np.zeros((envs, tets, stride), dtype="<f4")
        states[0, 0, 1] = 0.25
        states[0, 1, 1] = 0.5
        high = np.zeros((envs, tets, 3, 4), dtype="<f4")
        for tet in range(tets):
            high[0, tet, 0, 0] = 1.0
            high[0, tet, 1, 1] = 1.0
            high[0, tet, 2, 2] = 1.0
        low = np.zeros_like(high)
        captured_low = low.copy()
        if wrong_low:
            low[0, 0, 0, 0] = 1.0e-5
        P = np.zeros((modes, nodes, 4), dtype="<f4")
        P[0, 1:, 0] = 1.0
        effective = P[None].copy()
        effective[0, :, 0, :] = 0.0
        AP = np.zeros_like(effective)
        AP[0, 0, 1:, 0] = 2.0
        RAP = np.asarray([[[8.0]]], dtype="<f4")
        pivots = np.asarray([[0]], dtype="<u4")
        factor_valid = np.asarray([1], dtype="<u4")
        blocks = np.zeros((envs, nodes, 3, 4), dtype="<f4")
        for node in range(1, nodes):
            blocks[0, node, 0] = [0.5, 0.0, 0.0, 1.0]
            blocks[0, node, 1] = [0.0, 0.5, 0.0, 0.0]
            blocks[0, node, 2] = [0.0, 0.0, 0.5, 0.0]
        arrays = [
            ("P", "<f4x4", [modes, nodes], P),
            ("effective_P", "<f4x4", [envs, modes, nodes], effective),
            ("AP", "<f4x4", [envs, modes, nodes], AP),
            ("pivoted_RAP", "<f4", [envs, modes, modes], RAP),
            ("pivots", "<u4", [envs, modes], pivots),
            ("factor_valid", "<u4", [envs], factor_valid),
            ("node_block_inverse", "<f4x4", [envs, nodes, 3], blocks),
            ("candidate_nodes", "struct:NMFEMNodeStateGPU", [envs, nodes], state),
            ("material_history", "<f4", [envs, tets, stride], states),
            ("scheduler_exponents", "<u4", [envs, tets], np.zeros((envs, tets), dtype="<u4")),
            ("deformation_high", "<f4x4", [envs, tets, 3], high),
            ("deformation_low", "<f4x4", [envs, tets, 3], low),
            ("production_status", "struct:NMMatterStatusGPU", [envs],
             np.zeros((envs, study.NMMATTER_STATUS_GPU_BYTES), dtype="|u1")),
        ]
        capture = folder / "operator-capture"
        header = {"record": "header", "schema": "numi.cardboard.fem-newton-capture.v1",
                  "root": 3, "newton_iteration": 23,
                  "world_fingerprint_hex": f"{17:016x}",
                  "execution_fingerprint_hex": f"{23:016x}",
                  "environment_count": envs, "node_count_per_environment": nodes,
                  "tetrahedron_count_per_environment": tets,
                  "dt_seconds": 0.1, "dt_seconds_by_environment": [0.1],
                  "production_inputs_bitwise_unchanged": True}
        if node_words == 24:
            header["schema"] = "numi.cardboard.fem-newton-capture.v2"
        rows = [header]
        for tet in range(tets):
            fmat = high[0, tet, :, :3].astype(float).tolist()
            rows.append({"record": "tetrahedron", "environment": 0, "tet_id": tet,
                         "material_state_count": stride,
                         "center": {"F_local": fmat,
                                    "F_local_low": captured_low[0, tet, :, :3].astype(float).tolist(),
                                    "accepted_state": states[0, tet].astype(float).tolist()}})
        rows.append({"record": "integrity", "production_inputs_bitwise_unchanged": True})
        capture.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))

        binary = folder / "operator-capture.regional.bin"
        inventory = []
        payload = bytearray()
        array_raw = {}
        for name, dtype, shape, array in arrays:
            element_bytes = {"<f4x4": 16, "<f4": 4, "<u4": 4,
                             "struct:NMFEMNodeStateGPU": node_words * 4,
                             "struct:NMMatterStatusGPU": study.NMMATTER_STATUS_GPU_BYTES}[dtype]
            if name == "production_status":
                raw = array.tobytes(order="C")
            else:
                raw = array.tobytes(order="C")
            array_raw[name] = raw
            inventory.append({"name": name, "dtype": dtype, "shape": shape,
                              "offset": len(payload), "byte_count": len(raw),
                              "element_bytes": element_bytes,
                              "sha256": hashlib.sha256(raw).hexdigest()})
            payload.extend(raw)
        binary.write_bytes(payload)
        canonical = ["candidate_nodes", "material_history", "scheduler_exponents",
                     "deformation_high", "deformation_low"]
        candidate_digest = hashlib.sha256(b"".join(array_raw[name] for name in canonical)).hexdigest()
        node_digest = hashlib.sha256(array_raw["candidate_nodes"]).hexdigest()
        manifest = {"schema": study.REGIONAL_RAP_SCHEMA,
                    "byte_order": "little_endian",
                    "capture_point": capture_point or study.REGIONAL_CAPTURE_POINT,
                    "source_physics_fingerprint": f"{17:016x}",
                    "device_program_fingerprint": f"{23:016x}",
                    "root": 3, "iteration": iteration,
                    "environment_count": envs, "node_count": nodes,
                    "tetrahedron_count": tets, "mode_count": modes,
                    "material_state_stride": stride, "device": {"name": "synthetic", "registry_id": 1},
                    "substep": 0, "microtick": 1, "microtick_count": 2,
                    "linearization_generation": (5 << 32) | iteration,
                    "linearization_generation_encoding": "(micro_generation << 32) | newton_iteration",
                    "micro_generation": 5, "dt_seconds": 0.1,
                    "operator_generation": study.REGIONAL_OPERATOR_GENERATION,
                    "candidate_linearization_digest_algorithm": study.REGIONAL_DIGEST_ALGORITHM,
                    "tangent_cache_used": False,
                    "candidate_linearization_sha256": [{"environment": 0,
                        "sha256": candidate_digest,
                        "candidate_nodes_sha256": node_digest}],
                    "payload": {"file": str(binary), "byte_count": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest()},
                    "arrays": inventory}
        manifest_path = folder / "operator-capture.regional.json"
        manifest_path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
        if corrupt_payload:
            binary.write_bytes(bytes([binary.read_bytes()[0] ^ 1]) + binary.read_bytes()[1:])

        A = sparse.eye(12, format="csr", dtype=np.float64) * 2.0
        vectors = {"direction": np.ones(12), "native_action_Ns": np.ones(12),
                   "assembled_action_Ns": np.ones(12), "action_difference_Ns": np.zeros(12),
                   "native_rhs_free_impulse_Ns": np.arange(1, 13, dtype=np.float64),
                   "free_node_ids": np.asarray([1, 2, 3, 4], dtype=np.int64),
                   "node_positions_m": np.zeros((nodes, 3)),
                   "captured_node_state_f32": captured_state[0].copy()}
        topology = {"nodes": np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64),
                    "material_indices": np.zeros(tets, dtype=np.int64),
                    "rest_volumes_m3": np.ones(tets),
                    "inverse_rest_rows_per_m": np.tile(np.eye(3), (tets, 1, 1)),
                    "node_count": np.asarray(nodes, dtype=np.int64)}
        sparse.save_npz(folder / "audit.operator.npz", A)
        np.savez_compressed(folder / "audit.vectors.npz", **vectors)
        np.savez_compressed(folder / "audit.topology.npz", **topology)
        operator_files = {name: study.sha256(folder / name)
                          for name in ("audit.operator.npz", "audit.vectors.npz", "audit.topology.npz")}
        report = {"schema": "numi.cardboard.cached-operator-audit.v1",
                  "capture_selection": {"root": 3, "newton_iteration": 23, "environment": 0},
                  "capture_integrity": {"production_inputs_bitwise_unchanged": True},
                  "binding": {"operator_fd_iteration": {"world_fingerprint": 17,
                       "execution_fingerprint": 23, "production_inputs_bitwise_unchanged": True},
                       "capture_run_fingerprints_match": True,
                       "capture_sha256": study.sha256(capture),
                       "prediction_environment": {"NM_FEM_OPERATOR_FD_CAPTURE_PATH": str(capture)},
                       "source_hashes_verified_against_prediction": {"source/runtime.mm": "a" * 64}},
                  "operator": {"node_count": nodes, "tetrahedron_count": tets,
                       "action_parity": {"relative_l2_error": 0.0},
                       "dt_seconds": 0.1},
                  "matrix_artifacts": {"files_sha256": operator_files}}
        audit_path = folder / "audit.json"
        audit_path.write_text(json.dumps(report))
        return audit_path, manifest_path

    def _run_sidecar(self, audit_path, manifest_path):
        args = Namespace(operator_report=str(audit_path), rap_export=None,
                         regional_sidecar=str(manifest_path), restart=16,
                         max_iterations=16, relative_tolerance=1e-10,
                         schwarz_patch_nodes=4, schwarz_anchor_stride=2)
        return study.run(args)

    def test_selected_newton_v2_preserves_96_byte_node_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            audit, sidecar = self._fixture(Path(temp) / "paired", node_words=24)
            result = self._run_sidecar(audit, sidecar)
            self.assertTrue(result["fixed_physical_operator"]["same_A_and_b_for_both_methods"])
            manifest, arrays, _, _ = study._read_regional_payload(sidecar)
            self.assertEqual(arrays["candidate_nodes"].shape, (1, 5, 24))
            np.testing.assert_array_equal(arrays["candidate_nodes"][0, :, 20], np.float32(2.0**-40))

    def test_selected_newton_sidecar_is_admitted_and_runs_same_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            audit, sidecar = self._fixture(Path(temp) / "positive")
            result = self._run_sidecar(audit, sidecar)
            self.assertEqual(result["native_rap_source_kind"], "selected_newton_factor_barrier_sidecar")
            self.assertEqual(result["native_rap_selected_generation"]["iteration"], 23)
            self.assertEqual(result["native_rap_selected_generation"]["capture_point"], study.REGIONAL_CAPTURE_POINT)
            self.assertLess(result["native_rap_validation"]["native_ap_relative_error"], 1e-12)
            self.assertTrue(result["fixed_physical_operator"]["same_A_and_b_for_both_methods"])

    def test_sidecar_rejects_wrong_iteration_capture_point_payload_and_low_F(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audit, sidecar = self._fixture(root / "wrong-iteration", iteration=24)
            with self.assertRaisesRegex(ValueError, "root/Newton iteration"):
                self._run_sidecar(audit, sidecar)
            audit, sidecar = self._fixture(root / "wrong-capture-point", capture_point="after_correction")
            with self.assertRaisesRegex(ValueError, "pre-FGMRES factor barrier"):
                self._run_sidecar(audit, sidecar)
            audit, sidecar = self._fixture(root / "bad-payload", corrupt_payload=True)
            with self.assertRaisesRegex(ValueError, "payload SHA-256"):
                self._run_sidecar(audit, sidecar)
            audit, sidecar = self._fixture(root / "wrong-low-F", wrong_low=True)
            with self.assertRaisesRegex(ValueError, "deformation low words"):
                self._run_sidecar(audit, sidecar)

    def test_sidecar_rejects_candidate_mismatch_even_if_manifest_hashes_are_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            audit, sidecar = self._fixture(Path(temp) / "wrong-candidate", wrong_candidate=True)
            with self.assertRaisesRegex(ValueError, "candidate nodes differ bitwise"):
                self._run_sidecar(audit, sidecar)

    def test_sidecar_rejects_candidate_digest_and_unverified_source_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audit, sidecar = self._fixture(root / "bad-digest")
            manifest = json.loads(sidecar.read_text())
            manifest["candidate_linearization_sha256"][0]["sha256"] = "0" * 64
            sidecar.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "candidate linearization digest mismatch"):
                self._run_sidecar(audit, sidecar)

            audit, sidecar = self._fixture(root / "missing-source-binding")
            report = json.loads(audit.read_text())
            report["binding"]["source_hashes_verified_against_prediction"] = {}
            audit.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "source files verified"):
                self._run_sidecar(audit, sidecar)

    def test_sidecar_rejects_stale_32_byte_production_status_abi(self):
        with tempfile.TemporaryDirectory() as temp:
            audit, sidecar = self._fixture(Path(temp) / "stale-status-abi")
            manifest = json.loads(sidecar.read_text())
            status = next(row for row in manifest["arrays"]
                          if row["name"] == "production_status")
            status["element_bytes"] = 32
            sidecar.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "incorrect element_bytes"):
                self._run_sidecar(audit, sidecar)


if __name__ == "__main__":
    unittest.main()
