#!/usr/bin/env python3
import importlib.util
import struct
import unittest
from pathlib import Path
import numpy as np

MODULE_PATH = Path(__file__).with_name("verify_kernel_output.py")
spec = importlib.util.spec_from_file_location("verify_kernel_output", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_case():
    # One free node, one coarse mode, one 3-DOF patch. This checks the complete
    # projection -> RAP solve -> local solve -> overlap gather ABI with a
    # hand-solvable positive diagonal example.
    h = (1, 1, 1, 1, 1, 1, 3, 9, 3, 2, 0)
    input_arrays = [
        np.array([[2.0]], dtype="<f4"),                   # RAP LU
        np.array([0], dtype="<u4"),                     # RAP pivots
        np.array([1], dtype="<u4"),                     # RAP valid
        np.array([[2., 4., 6., 0.]], dtype="<f4"),      # RHS basis
        np.array([[[1., 0., 0., 0.]]], dtype="<f4"),    # P
        np.array([[[2., 0., 0., 0.]]], dtype="<f4"),    # AP
        np.array([[0, 1, 0, 0]], dtype="<u4"),
        np.array([0], dtype="<u4"),
        np.array([0, 1], dtype="<u4"),
        np.array([0], dtype="<u4"),
        np.array([1], dtype="<u4"),
        np.diag([4., 5., 6.]).astype("<f4").reshape(-1),
    ]
    raw_in = bytearray(module.INPUT_MAGIC + struct.pack(module.HEADER_FMT, *h))
    for array in input_arrays:
        raw_in.extend(np.ascontiguousarray(array).tobytes())
    output_arrays = [
        np.array([2.], dtype="<f4"),
        np.array([1.], dtype="<f4"),
        np.diag([4., 5., 6.]).astype("<f4").reshape(-1),
        np.array([0, 1, 2], dtype="<u4"),
        np.array([1], dtype="<u4"),
        np.array([[0., .8, 1., 0.]], dtype="<f4").reshape(-1),
        np.array([[1., .8, 1., 0.]], dtype="<f4").reshape(-1),
        np.zeros(12, dtype="<u4"),
    ]
    raw_out = bytearray(module.OUTPUT_MAGIC + struct.pack(module.HEADER_FMT, *h))
    for array in output_arrays:
        raw_out.extend(np.ascontiguousarray(array).tobytes())
    return bytes(raw_in), bytes(raw_out)


def make_bundle_manifest(raw):
    h = struct.unpack_from(module.HEADER_FMT, raw, 8)
    _, nodes, tets, modes, patches, patch_nodes, dofs, matrix_count, pivots, envs, selected = h
    counts = {
        "rap_lu": modes * modes, "rap_pivots": modes, "rap_valid": 1,
        "basis": nodes * 4, "regional_basis": modes * nodes * 4,
        "regional_ap": modes * nodes * 4, "patches": patches * 4,
        "patch_nodes": patch_nodes, "node_patch_offsets": nodes + 1,
        "node_patch_values": patch_nodes, "node_coverage": nodes,
        "patch_matrices": matrix_count,
    }
    shapes = {
        "rap_lu": [modes, modes], "rap_pivots": [modes], "rap_valid": [1],
        "basis": [nodes, 4], "regional_basis": [modes, nodes, 4],
        "regional_ap": [modes, nodes, 4], "patches": [patches, 4],
        "patch_nodes": [patch_nodes], "node_patch_offsets": [nodes + 1],
        "node_patch_values": [patch_nodes], "node_coverage": [nodes],
        "patch_matrices": [matrix_count],
    }
    descriptors, cursor = [], 8 + struct.calcsize(module.HEADER_FMT)
    for name, dtype, _ in module.ARRAY_SPECS:
        size = counts[name] * np.dtype(dtype).itemsize
        blob = raw[cursor:cursor + size]
        descriptors.append({"name": name, "dtype": dtype, "shape": shapes[name],
                            "offset": cursor, "byte_count": size,
                            "sha256": module.sha256(blob)})
        cursor += size
    return {"schema": "numi.matter.selector5-kernel-check-input.v1",
            "source_capture_selector": 3, "native_selector5_capture_present": False,
            "bundle_sha256": module.sha256(raw), "arrays": descriptors,
            "dimensions": {"node_count": nodes, "tetrahedron_count": tets,
                "mode_count": modes, "patch_count": patches, "patch_node_count": patch_nodes,
                "dof_count": dofs, "matrix_float_count": matrix_count, "pivot_count": pivots,
                "source_environment_count": envs, "selected_environment": selected}}


class KernelOutputAuditTests(unittest.TestCase):
    def test_hand_solvable_end_to_end_bundle(self):
        raw_in, raw_out = make_case()
        report = module.verify(raw_in, raw_out)
        self.assertEqual(report["status"], "arithmetic_checks_passed")
        self.assertLess(report["relative_errors"]["RAP_factor_vs_PtAP"], 1e-12)
        self.assertLess(report["relative_errors"]["maximum_patch_solve_residual"], 1e-7)
        self.assertLess(report["relative_errors"]["global_gather"], 1e-7)

    def test_rejects_invalid_patch_factor(self):
        raw_in, raw_out = make_case()
        damaged = bytearray(raw_out)
        # Output offset: magic + header + coarse rhs + coefficient + patch LU + pivots.
        hbytes = struct.calcsize(module.HEADER_FMT)
        valid_offset = 8 + hbytes + 4 + 4 + 9 * 4 + 3 * 4
        struct.pack_into("<I", damaged, valid_offset, 0)
        with self.assertRaisesRegex(ValueError, "patch factors are invalid"):
            module.verify(raw_in, bytes(damaged))

    def test_rejects_corrupt_gather_even_when_factors_are_valid(self):
        raw_in, raw_out = make_case()
        damaged = bytearray(raw_out)
        # Final preconditioned x component at the end of output arrays before status.
        # Its location follows header, coarse(1), coeff(1), LU(9), pivots(3),
        # factorValid(1), correction(4), then output x/y/z/w.
        hbytes = struct.calcsize(module.HEADER_FMT)
        preconditioned_offset = 8 + hbytes + 4 + 4 + 9 * 4 + 3 * 4 + 4 + 4 * 4
        struct.pack_into("<f", damaged, preconditioned_offset, 9.0)
        with self.assertRaisesRegex(ValueError, "gather mismatch"):
            module.verify(raw_in, bytes(damaged))

    def test_rejects_dimension_or_trailing_byte_corruption(self):
        raw_in, raw_out = make_case()
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            bad_header = bytearray(raw_out)
            struct.pack_into("<I", bad_header, 8 + 4, 2)
            module.verify(raw_in, bytes(bad_header))
        with self.assertRaisesRegex(ValueError, "trailing bytes"):
            module.verify(raw_in + b"x", raw_out)

    def test_bundle_manifest_binds_exact_binary_arrays_and_dimensions(self):
        raw_in, _ = make_case()
        manifest = make_bundle_manifest(raw_in)
        self.assertEqual(module.validate_bundle_manifest(raw_in, manifest)["node_count"], 1)
        bad = dict(manifest)
        bad["dimensions"] = dict(manifest["dimensions"], node_count=2)
        with self.assertRaisesRegex(ValueError, "binary dimensions"):
            module.validate_bundle_manifest(raw_in, bad)
        bad = make_bundle_manifest(raw_in)
        bad["arrays"] = [dict(item) for item in bad["arrays"]]
        bad["arrays"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "descriptor SHA-256"):
            module.validate_bundle_manifest(raw_in, bad)


if __name__ == "__main__":
    unittest.main()
