import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from import_equine_suture_ramp import HEADERS, convert

class ImportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.source = self.base / "input.txt"
        self.output = self.base / "output.csv"
    def tearDown(self):
        self.tmp.cleanup()
    def write(self, body):
        data = body.encode("utf-16")
        self.source.write_bytes(data)
        return hashlib.sha256(data).hexdigest()
    def test_si_and_signed_force(self):
        digest = self.write("\t".join(HEADERS) + "\n0\t0\t-3\t-2\t5\n.01\t.01\t-2\t7\t6\n")
        meta = convert(self.source, digest, self.output)
        with self.output.open(newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual([float(x) for x in rows[1]], [0, 0, -.003, -2, .005])
        self.assertEqual(meta["rows"], 2)
        self.assertEqual(meta["output_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            convert(self.source, digest, self.output)
    def test_reject_bad_integrity_and_nonmonotonic_time(self):
        digest = self.write("\t".join(HEADERS) + "\n0\t0\t0\t0\t0\n0\t.01\t0\t0\t0\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            convert(self.source, "0"*64, self.output)
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            convert(self.source, digest, self.output)
        self.assertFalse(self.output.exists())
    def test_reject_nonfinite_and_changed_schema(self):
        digest = self.write("\t".join(HEADERS) + "\n0\t0\t0\t0\t0\n.01\t.01\t0\tnan\t0\n")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            convert(self.source, digest, self.output)
        digest = self.write("changed header\n")
        with self.assertRaisesRegex(ValueError, "column names"):
            convert(self.source, digest, self.output)
        self.assertFalse(self.output.exists())

if __name__ == "__main__":
    unittest.main()
