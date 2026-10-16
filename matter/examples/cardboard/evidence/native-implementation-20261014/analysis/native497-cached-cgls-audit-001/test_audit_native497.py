#!/usr/bin/env python3
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("audit_native497.py")
SPEC = importlib.util.spec_from_file_location("audit_native497", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def make_v3_capture(path: Path) -> None:
    envs, nodes, unknowns = 1, 2, 4
    header = struct.pack("<8I", audit.MAGIC, 3, envs, nodes, unknowns, 64, 13, 0)
    segments = [
        ("x", "<f4", unknowns * 4), ("r", "<f4", unknowns * 4),
        ("s", "<f4", unknowns * 4), ("p", "<f4", unknowns * 4),
        ("q", "<f4", unknowns * 4), ("b", "<f4", unknowns * 4),
        ("mask", "<f4", unknowns * 4), ("row_environment", "<u4", unknowns),
        ("scalars", "<f4", envs * 4), ("control", "<u4", envs * 4),
        ("errors", "<u4", envs), ("selected", "<u4", envs),
        ("right_scale_rows", "<f4", envs * nodes * 3 * 4),
    ]
    body = bytearray(header)
    for _, dtype, count in segments:
        aligned = audit._align16(len(body))
        body.extend(b"\0" * (aligned - len(body)))
        body.extend(np.zeros(count, dtype=np.dtype(dtype)).tobytes())
    path.write_bytes(body)


class BinaryReaderTests(unittest.TestCase):
    def test_version3_warm_capture_layout_and_exact_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cgls.bin"
            make_v3_capture(path)
            header, arrays = audit.parse_cgls_capture(path)
            self.assertEqual(header["version"], 3)
            self.assertEqual(header["unknown_count"], 4)
            self.assertEqual(header["segment_count"], 13)
            self.assertEqual(arrays["x"].shape, (4, 4))
            self.assertEqual(arrays["right_scale_rows"].shape, (1, 2, 3, 4))
            self.assertEqual(len(header["segments"]), 13)

    def test_truncation_and_wrong_v3_segment_count_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cgls.bin"
            make_v3_capture(path)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "truncated CGLS segment|trailing bytes"):
                audit.parse_cgls_capture(path)
            raw = bytearray(path.read_bytes())
            # Restore complete payload then corrupt only the versioned segment count.
            make_v3_capture(path)
            raw = bytearray(path.read_bytes())
            struct.pack_into("<I", raw, 24, 12)
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "segment count"):
                audit.parse_cgls_capture(path)

    def test_nonzero_alignment_padding_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cgls.bin"
            make_v3_capture(path)
            header, _ = audit.parse_cgls_capture(path)
            # The 4-byte errors segment is followed by padding before selected.
            errors = next(x for x in header["segments"] if x["name"] == "errors")
            pad = errors["offset"] + errors["byte_count"]
            path.write_bytes(path.read_bytes()[:pad] + b"\x01" + path.read_bytes()[pad + 1:])
            with self.assertRaisesRegex(ValueError, "alignment padding"):
                audit.parse_cgls_capture(path)


class WarmResidualSemanticsTests(unittest.TestCase):
    def test_total_candidate_reconstructs_residual_from_x0_plus_delta(self):
        M = np.diag([2.0, 3.0, 4.0])
        K = np.array([[4.0, -1.0, 0.0], [-1.0, 3.0, -0.5], [0.0, -0.5, 2.0]])
        dt2 = 0.125
        A = M + dt2 * K
        b = np.array([1.0, -2.0, 4.0])
        x0 = np.array([0.25, -0.75, 0.125])
        delta = np.array([-0.1, 0.05, 0.3])
        X = x0 + delta
        result = audit.warm_residual_decomposition(A, b, x0, X)
        np.testing.assert_allclose(result["delta"], delta, rtol=0, atol=1e-15)
        np.testing.assert_allclose(result["initial_residual"], b - A @ x0, rtol=0, atol=1e-15)
        np.testing.assert_allclose(result["final_residual"], b - A @ X, rtol=0, atol=1e-15)
        np.testing.assert_allclose(result["residual_from_delta"], result["final_residual"], rtol=0, atol=1e-15)
        self.assertLess(np.linalg.norm(result["identity_error"]), 1e-14)
        # The fresh B is b-A*x0, not the original right-hand side b.
        self.assertGreater(np.linalg.norm(result["initial_residual"] - b), 1e-3)

    def test_mismatched_vector_extents_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "inconsistent extents"):
            audit.warm_residual_decomposition(np.eye(2), np.ones(2), np.ones(2), np.ones(3))


class PredictionInputMirrorTests(unittest.TestCase):
    def test_explicit_checkpoint_case_alias_verifies_exact_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit.BASE = root
            rel = Path("solver/macmini-warm-cgls-control19-checkpoint-480/output/checkpoints/coupon-control-000019.nmcb")
            target = root / rel
            target.parent.mkdir(parents=True)
            target.write_bytes(b"checkpoint bytes")
            remote = "/Users/n/cardboard-implementation-20261014-host-macmini/solver/warm-cgls-control19-checkpoint-480/output/checkpoints/coupon-control-000019.nmcb"
            result = audit.resolve_prediction_inputs({
                "external_inputs": [remote],
                "inputs_sha256": {"external/coupon-control-000019.nmcb": audit.sha256(target)},
            }, root / "source", root / "artifacts", root)
            self.assertTrue(result["all_declared_prediction_inputs_verified"])
            self.assertEqual(result["verified_count"], 1)
            self.assertEqual(result["explicit_local_path_aliases"][0]["local"], str(target))

    def test_alias_is_not_generalized_to_unlisted_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit.BASE = root
            target = root / "solver/macmini-other-case/payload"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"do not alias")
            remote = "/Users/n/cardboard-implementation-20261014-host-macmini/solver/other-case/payload"
            result = audit.resolve_prediction_inputs({
                "external_inputs": [remote],
                "inputs_sha256": {"external/payload": audit.sha256(target)},
            }, root / "source", root / "artifacts", root)
            self.assertFalse(result["all_declared_prediction_inputs_verified"])
            self.assertEqual(result["missing_count"], 1)


if __name__ == "__main__":
    unittest.main()
