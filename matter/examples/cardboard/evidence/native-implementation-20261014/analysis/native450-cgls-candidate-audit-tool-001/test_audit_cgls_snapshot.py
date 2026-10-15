import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("audit_cgls_snapshot.py")
spec = importlib.util.spec_from_file_location("audit_cgls_snapshot", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def make_snapshot(path, *, magic=audit.MAGIC, bad_padding=False):
    envs, nodes, unknown, cap = 2, 3, 12, 64
    data = bytearray(struct.pack("<8I", magic, 1, envs, nodes, unknown, cap, 12, 0))
    specs = [(name, 4 * unknown) for name in audit.VECTOR_NAMES]
    specs += [("row_environment", unknown), ("scalars", 4 * envs),
              ("control", 4 * envs), ("errors", envs), ("selected", envs)]
    for _, count in specs:
        target = audit.align16(len(data))
        pad_start = len(data)
        data.extend(b"\0" * (target - pad_start))
        if bad_padding and target > pad_start:
            data[pad_start] = 1
            bad_padding = False
        data.extend(b"\0" * (count * 4))
    path.write_bytes(data)


class SnapshotTests(unittest.TestCase):
    def test_exact_layout(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "base-cgls.bin"
            make_snapshot(p)
            h, arrays = audit.parse_snapshot(p)
            self.assertEqual(h["environment_count"], 2)
            self.assertEqual(h["nodes_per_environment"], 3)
            self.assertEqual(h["unknown_count"], 12)
            self.assertEqual(h["iteration_cap"], 64)
            self.assertEqual(arrays["x"].shape, (12, 4))
            self.assertEqual(arrays["control"].shape, (2, 4))

    def test_reject_bad_magic_and_padding(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.bin"
            make_snapshot(bad, magic=0)
            with self.assertRaisesRegex(ValueError, "unsupported CGLS header"):
                audit.parse_snapshot(bad)
            make_snapshot(bad, bad_padding=True)
            with self.assertRaisesRegex(ValueError, "alignment padding"):
                audit.parse_snapshot(bad)

    def test_reject_trailing_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trailing.bin"
            make_snapshot(p)
            p.write_bytes(p.read_bytes() + b"\0")
            with self.assertRaisesRegex(ValueError, "trailing bytes"):
                audit.parse_snapshot(p)


class GateAndTraceTests(unittest.TestCase):
    def test_fp32_word_comparison_includes_signed_zero(self):
        exact = audit.compare_float32_words([1.25, -0.0], [1.25, -0.0])
        self.assertTrue(exact["bitwise_word_match"])
        signed_zero = audit.compare_float32_words([-0.0], [0.0])
        self.assertFalse(signed_zero["bitwise_word_match"])
        self.assertEqual(signed_zero["mismatch_count"], 1)
        self.assertEqual(signed_zero["first_left_word_hex"], "0x80000000")
        self.assertEqual(signed_zero["first_right_word_hex"], "0x00000000")

    def test_existing_one_percent_action_gate(self):
        within = audit.existing_action_gate({"relative_l2": 0.009,
                                             "difference_l2": 0.009,
                                             "reference_l2": 1.0})
        outside = audit.existing_action_gate({"relative_l2": 0.01001,
                                              "difference_l2": 0.01001,
                                              "reference_l2": 1.0})
        zero = audit.existing_action_gate({"relative_l2": None,
                                           "difference_l2": 1e-12,
                                           "reference_l2": 0.0})
        self.assertTrue(within["applicable_for_nonzero_finite_reference_action"])
        self.assertTrue(within["within_existing_1_percent_gate"])
        self.assertFalse(outside["within_existing_1_percent_gate"])
        self.assertFalse(zero["applicable_for_nonzero_finite_reference_action"])
        self.assertFalse(zero["within_existing_1_percent_gate"])

    def test_exact_pretarget_trace_and_mismatch(self):
        rows = [{"root": 1, "newton_iteration": i,
                 "environments": [{"accepted": True}, {"accepted": True}]}
                for i in range(3)]
        same = audit.summarize_pretarget_trace(rows, list(rows), 1, 3)
        self.assertTrue(same["exact_pretarget_trace_parity"])
        changed = [dict(r) for r in rows]
        changed[1] = {**rows[1], "environments": [{"accepted": False}, {"accepted": True}]}
        mismatch = audit.summarize_pretarget_trace(rows, changed, 1, 3)
        self.assertFalse(mismatch["exact_pretarget_trace_parity"])
        self.assertEqual(mismatch["mismatch_iterations"], [1])

    def test_capture_skip_is_run_specific(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "stderr.log"
            p.write_text('fem_coupled_capture_skipped={"schema":1,"root":1,"iteration":48,"reason":"environment_count_out_of_scope","observed":{"environment_count":2}}\n')
            got = audit.summarize_coupled_capture_scope(p, 1, 48)
            self.assertEqual(got["classification"], "runtime emitted a bounded skip marker")
            self.assertEqual(got["skipped_markers"][0]["reason"], "environment_count_out_of_scope")
            other = audit.summarize_coupled_capture_scope(p, 0, 48)
            self.assertIn("absence is not interpreted as success", other["classification"])


if __name__ == "__main__":
    unittest.main()
