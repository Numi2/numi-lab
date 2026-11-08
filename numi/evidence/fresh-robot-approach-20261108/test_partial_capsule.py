#!/usr/bin/env python3
import tempfile
from pathlib import Path
import unittest
from verify_partial_capsule import verify, COMPONENTS

def fnv(data):
    h = 1469598103934665603
    for b in data: h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    return h

class CapsuleValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stage = Path(self.tmp.name)
        self.capsule = self.stage / "partial-owner-capsule"
        self.capsule.mkdir()
        self.rows = ["schema\tnumi.surgical-partial-owner-capsule.v1", "complete\t0", "restorable\t0", "lineage\tsynthetic_same_resident_input_bound_by_hash", "compiled_world_fingerprint\t123", "accepted_state_step\t1400", "missing\tconvex_cache"]
        self.input = self.stage / "input.tsv"
        self.input.write_bytes(b"input checkpoint")
        self.rows += ["input_checkpoint_name\tinput.tsv", f"input_checkpoint_size\t{len(self.input.read_bytes())}", f"input_checkpoint_fnv64\t{fnv(self.input.read_bytes())}", "input_state_step\t1389"]
        for kind, name, data in [("checkpoint", "state.tsv", b"matter_snapshot\tmatter.bin\t1\n"), ("matter", "matter.bin", b"matter"), ("rigid_cache", "cache.bin", b"cache")]:
            (self.stage / name).write_bytes(data)
            self.rows.append(f"binding\t{kind}\t{name}\t{len(data)}\t{fnv(data)}")
        for name in sorted(COMPONENTS):
            data = b"abcd"
            (self.capsule / name).write_bytes(data)
            self.rows.append(f"component\t{name}\t1\t4\t4\t{fnv(data)}")
        self.save()
    def tearDown(self): self.tmp.cleanup()
    def save(self): (self.capsule / "manifest.tsv").write_text("\n".join(self.rows) + "\n")
    def test_valid_partial(self): self.assertEqual(verify(self.capsule, self.input)[0]["restorable"], "0")
    def test_corrupt_component(self):
        (self.capsule / "q.bin").write_bytes(b"abce")
        with self.assertRaises(ValueError): verify(self.capsule, self.input)
    def test_truncated_component(self):
        (self.capsule / "q.bin").write_bytes(b"abc")
        with self.assertRaises(ValueError): verify(self.capsule, self.input)
    def test_missing_component(self):
        (self.capsule / "q.bin").unlink()
        with self.assertRaises(ValueError): verify(self.capsule, self.input)
    def test_false_restorable(self):
        self.rows[self.rows.index("restorable\t0")] = "restorable\t1"
        self.save()
        with self.assertRaises(ValueError): verify(self.capsule, self.input)
    def test_bad_binding(self):
        (self.stage / "matter.bin").write_bytes(b"xatter")
        with self.assertRaises(ValueError): verify(self.capsule, self.input)
    def test_changed_input(self):
        self.input.write_bytes(b"Input checkpoint")
        with self.assertRaises(ValueError): verify(self.capsule, self.input)
    def test_missing_fingerprint(self):
        self.rows.remove("compiled_world_fingerprint\t123")
        self.save()
        with self.assertRaises(ValueError): verify(self.capsule, self.input)

if __name__ == "__main__": unittest.main()
