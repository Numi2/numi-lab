#!/usr/bin/env python3
import unittest
from unittest.mock import patch
import audit_spatial_law_provenance as subject

class AuditTests(unittest.TestCase):
    def test_retained_revisions_and_missing_binary_binding(self):
        result = subject.audit()
        self.assertTrue(result["source_checks_pass"])
        records = {r["name"]: r for r in result["cases"]}
        self.assertFalse(records["n64_parallel"]["execution_record_contains_binary_sha256"])
        self.assertTrue(records["n128_parallel"]["execution_record_contains_binary_sha256"])

    def test_mutated_source_cannot_match_manifest(self):
        original = subject.subprocess.check_output
        def corrupted(*args, **kwargs):
            result = original(*args, **kwargs)
            return result.replace(b"k=1e16", b"k=2e16", 1)
        with patch.object(subject.subprocess, "check_output", side_effect=corrupted):
            with self.assertRaisesRegex(ValueError, "source hash"):
                subject.audit()

    def test_function_extraction_rejects_missing_close(self):
        with self.assertRaisesRegex(ValueError, "unclosed"):
            subject.body("double pressure(double g)const { if(g){return 1;}",
                         "double pressure(double g)const")

if __name__ == "__main__":
    unittest.main()
