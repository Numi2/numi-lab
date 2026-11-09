#!/usr/bin/env python3
"""Check that a qualified trace cannot be attached to another native build."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from cost_provenance import verify_accuracy_provenance


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build = self.root / "build"
        self.build.mkdir()
        self.names = ["numi_plane_contact_probe_scaled", "NumiMatter.metallib",
                      "metalrobo_plane_contact_benchmark"]
        for name in self.names:
            (self.build / name).write_bytes(name.encode())
        self.manifest = self.root / "inputs.sha256"
        self.hashes = {str(self.build / name): hashlib.sha256(name.encode()).hexdigest()
                       for name in self.names}
        self.manifest.write_text("".join(h + "  " + n + "\n" for n, h in self.hashes.items()))
        self.record = {"command": [str(self.build / self.names[0]), str(self.build / self.names[1])],
                       "input_sha256": dict(self.hashes)}

    def test_both_retained_record_formats(self):
        verify_accuracy_provenance(self.record, self.manifest, self.build)
        self.record.pop("input_sha256")
        self.record["input_hashes"] = self.manifest.name
        verify_accuracy_provenance(self.record, self.manifest, self.build)

    def test_changed_accuracy_hash_is_rejected(self):
        self.record["input_sha256"][str(self.build / self.names[0])] = "0" * 64
        with self.assertRaises(ValueError):
            verify_accuracy_provenance(self.record, self.manifest, self.build)

    def test_changed_benchmark_or_shader_is_rejected(self):
        for name in self.names[1:]:
            path = self.build / name
            original = path.read_bytes()
            path.write_bytes(b"different")
            with self.assertRaises(ValueError):
                verify_accuracy_provenance(self.record, self.manifest, self.build)
            path.write_bytes(original)

    def test_other_producer_or_manifest_is_rejected(self):
        self.record["command"][0] = "/different/probe"
        with self.assertRaises(ValueError):
            verify_accuracy_provenance(self.record, self.manifest, self.build)
        self.record["command"][0] = str(self.build / self.names[0])
        self.record.pop("input_sha256")
        self.record["input_hashes"] = "other.sha256"
        with self.assertRaises(ValueError):
            verify_accuracy_provenance(self.record, self.manifest, self.build)


if __name__ == "__main__":
    unittest.main()
