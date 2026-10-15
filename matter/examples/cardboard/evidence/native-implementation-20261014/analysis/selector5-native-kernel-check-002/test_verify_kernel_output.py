#!/usr/bin/env python3
"""Failure-injection tests for the complete selector-5 CPU audit."""
from __future__ import annotations

import importlib.util
import struct
import unittest
import warnings
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("verify_kernel_output.py")
spec = importlib.util.spec_from_file_location("verify_kernel_output_002", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def _make_two_patch_case() -> tuple[bytes, bytes]:
    h = (1, 2, 2, 1, 2, 2, 6, 18, 6, 2, 0)
    matrices = np.asarray([
        2, 0, 0, 0, 3, 0, 0, 0, 4,
        5, 0, 0, 0, 6, 0, 0, 0, 7,
    ], dtype="<f4")
    input_arrays = [
        np.asarray([[2.0]], dtype="<f4"),
        np.asarray([0], dtype="<u4"),
        np.asarray([1], dtype="<u4"),
        np.asarray([[2, 4, 6, 0], [5, 12, 14, 0]], dtype="<f4"),
        np.asarray([[[1, 0, 0, 0], [0, 0, 0, 0]]], dtype="<f4"),
        np.asarray([[[2, 0, 0, 0], [0, 0, 0, 0]]], dtype="<f4"),
        np.asarray([[0, 1, 0, 0], [1, 1, 9, 3]], dtype="<u4"),
        np.asarray([0, 1], dtype="<u4"),
        np.asarray([0, 1, 2], dtype="<u4"),
        np.asarray([0, 1], dtype="<u4"),
        np.asarray([1, 1], dtype="<u4"),
        matrices,
    ]
    raw_in = bytearray(module.INPUT_MAGIC + struct.pack(module.HEADER_FMT, *h))
    for array in input_arrays:
        raw_in.extend(np.ascontiguousarray(array).tobytes())
    raw_out = bytearray(module.OUTPUT_MAGIC + struct.pack(module.HEADER_FMT, *h))
    output_arrays = [
        np.asarray([2], dtype="<f4"),
        np.asarray([1], dtype="<f4"),
        matrices.copy(),
        np.asarray([0, 1, 2, 0, 1, 2], dtype="<u4"),
        np.asarray([1, 1], dtype="<u4"),
        np.asarray([[0, 4 / 3, 1.5, 0], [1, 2, 2, 0]], dtype="<f4").reshape(-1),
        np.asarray([[1, 4 / 3, 1.5, 0], [1, 2, 2, 0]], dtype="<f4").reshape(-1),
        np.zeros(12, dtype="<u4"),
    ]
    for array in output_arrays:
        raw_out.extend(np.ascontiguousarray(array).tobytes())
    return bytes(raw_in), bytes(raw_out)


class CompleteKernelOutputAuditTests(unittest.TestCase):
    def test_two_patch_fixture_passes_without_numpy_runtime_warnings(self):
        raw_in, raw_out = _make_two_patch_case()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = module.verify(raw_in, raw_out)
        self.assertEqual(result["status"], "arithmetic_checks_passed")
        self.assertEqual(len(result["patch_results"]), 2)
        self.assertTrue(result["diagnostics"]["no_numpy_matmul_or_dot_used"])

    def test_failure_in_first_patch_does_not_hide_later_patch(self):
        raw_in, raw_out = _make_two_patch_case()
        damaged = bytearray(raw_out)
        header_bytes = struct.calcsize(module.HEADER_FMT)
        correction_offset = (8 + header_bytes + 2 * 4 + 2 * 4 + 18 * 4 +
                             6 * 4 + 2 * 4)
        # Corrupt patch 0's x correction. The verifier must still report patch 1.
        struct.pack_into("<f", damaged, correction_offset, 0.25)
        result = module.verify(raw_in, bytes(damaged))
        self.assertEqual(result["status"], "arithmetic_checks_failed")
        self.assertEqual([p["patch_index"] for p in result["patch_results"]], [0, 1])
        self.assertGreater(result["patch_results"][0]["native_solve_residual_relative"], 1e-3)
        self.assertLess(result["patch_results"][1]["native_solve_residual_relative"], 1e-3)
        self.assertIn("patch_0_native_solve_residual", [x["name"] for x in result["failed_checks"]])

    def test_patch_lu_error_and_native_solve_error_are_reported_separately(self):
        raw_in, raw_out = _make_two_patch_case()
        damaged = bytearray(raw_out)
        header_bytes = struct.calcsize(module.HEADER_FMT)
        lu_offset = 8 + header_bytes + 2 * 4 + 2 * 4
        struct.pack_into("<f", damaged, lu_offset, 2.5)
        result = module.verify(raw_in, bytes(damaged))
        self.assertEqual(len(result["patch_results"]), 2)
        self.assertGreater(result["patch_results"][0]["factor_reconstruction_relative"], 1e-3)
        self.assertIn("patch_0_factor_reconstruction", [x["name"] for x in result["failed_checks"]])

    def test_rejects_malformed_dimensions_before_numerical_audit(self):
        raw_in, raw_out = _make_two_patch_case()
        damaged = bytearray(raw_out)
        struct.pack_into("<I", damaged, 8 + 4, 3)
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            module.verify(raw_in, bytes(damaged))

    def test_bundle_manifest_binds_input_bytes_and_shapes(self):
        raw_in, _ = _make_two_patch_case()
        h = struct.unpack_from(module.HEADER_FMT, raw_in, 8)
        _, nodes, tets, modes, patches, patch_nodes, dofs, matrix_floats, pivots, envs, selected = h
        shapes = {
            "rap_lu": [modes, modes], "rap_pivots": [modes], "rap_valid": [1],
            "basis": [nodes, 4], "regional_basis": [modes, nodes, 4],
            "regional_ap": [modes, nodes, 4], "patches": [patches, 4],
            "patch_nodes": [patch_nodes], "node_patch_offsets": [nodes + 1],
            "node_patch_values": [patch_nodes], "node_coverage": [nodes],
            "patch_matrices": [matrix_floats],
        }
        dims = {"rap_lu": modes * modes, "rap_pivots": modes, "rap_valid": 1,
                "basis": nodes * 4, "regional_basis": modes * nodes * 4,
                "regional_ap": modes * nodes * 4, "patches": patches * 4,
                "patch_nodes": patch_nodes, "node_patch_offsets": nodes + 1,
                "node_patch_values": patch_nodes, "node_coverage": nodes,
                "patch_matrices": matrix_floats}
        desc = []
        cur = 8 + struct.calcsize(module.HEADER_FMT)
        for name, dtype, _ in module.ARRAY_SPECS:
            byte_count = dims[name] * np.dtype(dtype).itemsize
            part = raw_in[cur:cur + byte_count]
            desc.append({"name": name, "dtype": dtype, "shape": shapes[name],
                         "offset": cur, "byte_count": byte_count,
                         "sha256": module.sha256(part)})
            cur += byte_count
        bundle = {
            "schema": "numi.matter.selector5-kernel-check-input.v1",
            "source_capture_selector": 3, "native_selector5_capture_present": False,
            "bundle_sha256": module.sha256(raw_in), "arrays": desc,
            "dimensions": {"node_count": nodes, "tetrahedron_count": tets,
                           "mode_count": modes, "patch_count": patches,
                           "patch_node_count": patch_nodes, "dof_count": dofs,
                           "matrix_float_count": matrix_floats, "pivot_count": pivots,
                           "source_environment_count": envs, "selected_environment": selected},
        }
        self.assertEqual(module.validate_bundle_manifest(raw_in, bundle)["node_count"], 2)
        bundle["bundle_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "do not match"):
            module.validate_bundle_manifest(raw_in, bundle)


if __name__ == "__main__":
    unittest.main()
