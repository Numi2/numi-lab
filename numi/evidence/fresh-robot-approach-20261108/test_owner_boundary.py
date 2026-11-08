#!/usr/bin/env python3
import unittest
from test_partial_capsule import CapsuleValidation, fnv
from verify_owner_boundary import verify

class BoundaryValidation(CapsuleValidation):
    def setUp(self):
        super().setUp()
        self.boundary = self.stage / "owner-boundary-evidence"
        self.boundary.mkdir()
        prior = (self.capsule / "manifest.tsv").read_bytes()
        cache = bytes(range(80))
        (self.boundary / "convex-caches.bin").write_bytes(cache)
        self.extra = ["schema\tnumi.synthetic-owner-boundary-evidence.v1", "complete\t0", "restorable\t0", "capture_exclusion\tsingle_thread_private_fixture_after_submission_wait", "environment_count\t1", "supports_reset\t0", "task_fingerprint\t0", "millard_fingerprint\t0", "world_fingerprint\t123", "device_physics_fingerprint\t1", "matter_source_fingerprint\t2", "matter_device_fingerprint\t3", "selectors\t1,2,3,4,5,6,7,8,9", "missing\towner_transaction_state", f"prior_capsule_manifest_size\t{len(prior)}", f"prior_capsule_manifest_fnv64\t{fnv(prior)}", f"component\tconvex-caches.bin\t1\t80\t80\t{fnv(cache)}"]
        self.save_boundary()
    def save_boundary(self): (self.boundary / "manifest.tsv").write_text("\n".join(self.extra) + "\n")
    def test_boundary_valid(self): self.assertEqual(verify(self.stage, self.input)[0]["restorable"], "0")
    def test_wrong_world(self):
        self.extra[self.extra.index("world_fingerprint\t123")] = "world_fingerprint\t124"; self.save_boundary()
        with self.assertRaises(ValueError): verify(self.stage, self.input)
    def test_wrong_reset(self):
        self.extra[self.extra.index("supports_reset\t0")] = "supports_reset\t1"; self.save_boundary()
        with self.assertRaises(ValueError): verify(self.stage, self.input)
    def test_corrupt_cache(self):
        (self.boundary / "convex-caches.bin").write_bytes(b"x" * 80)
        with self.assertRaises(ValueError): verify(self.stage, self.input)
    def test_bad_cache_size(self):
        (self.boundary / "convex-caches.bin").write_bytes(bytes(79))
        with self.assertRaises(ValueError): verify(self.stage, self.input)
    def test_false_restorable_boundary(self):
        self.extra[self.extra.index("restorable\t0")] = "restorable\t1"; self.save_boundary()
        with self.assertRaises(ValueError): verify(self.stage, self.input)

if __name__ == "__main__": unittest.main()
