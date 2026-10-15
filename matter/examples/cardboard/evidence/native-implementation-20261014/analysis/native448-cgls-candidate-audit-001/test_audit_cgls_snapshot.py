import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np

SCRIPT = Path(__file__).with_name("audit_cgls_snapshot.py")
spec = importlib.util.spec_from_file_location("audit_cgls_snapshot", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def make_snapshot(path, *, magic=audit.MAGIC, bad_padding=False):
    E, N, U, cap = 2, 3, 12, 32
    data = bytearray(struct.pack("<8I", magic, 1, E, N, U, cap, 12, 0))
    specs = [(name, 4*U) for name in audit.VECTOR_NAMES]
    specs += [("row_environment", U), ("scalars", 4*E), ("control", 4*E),
              ("errors", E), ("selected", E)]
    for i, (_, count) in enumerate(specs):
        target = audit.align16(len(data))
        pad_start = len(data)
        data.extend(b"\0" * (target-pad_start))
        if bad_padding and target > pad_start:
            data[pad_start] = 1
            bad_padding = False
        data.extend(b"\0" * (count*4))
    path.write_bytes(data)


class SnapshotParserTests(unittest.TestCase):
    def test_valid_layout_and_segments(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "base-cgls.bin"
            make_snapshot(p)
            h, a = audit.parse_snapshot(p)
            self.assertEqual((h["environment_count"], h["nodes_per_environment"], h["unknown_count"]), (2,3,12))
            self.assertEqual(h["iteration_cap"], 32)
            self.assertEqual(a["x"].shape, (12,4))
            self.assertEqual(a["control"].shape, (2,4))
            self.assertEqual(a["selected"].shape, (2,))
            self.assertEqual(h["segment_count"], 12)

    def test_reject_bad_magic(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.bin"
            make_snapshot(p, magic=0)
            with self.assertRaisesRegex(ValueError, "unsupported CGLS header"):
                audit.parse_snapshot(p)

    def test_reject_nonzero_alignment_padding(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad-padding.bin"
            make_snapshot(p, bad_padding=True)
            with self.assertRaisesRegex(ValueError, "alignment padding"):
                audit.parse_snapshot(p)

    def test_reject_trailing_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trailing.bin"
            make_snapshot(p)
            p.write_bytes(p.read_bytes() + b"\0")
            with self.assertRaisesRegex(ValueError, "trailing bytes"):
                audit.parse_snapshot(p)


if __name__ == "__main__":
    unittest.main()
