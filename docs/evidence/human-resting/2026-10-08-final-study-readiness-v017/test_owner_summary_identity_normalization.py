#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("readiness_p15", ROOT / "prepare_final_plan.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class OwnerSummaryIdentityNormalizationTests(unittest.TestCase):
    def test_large_native_decimal_strings_normalize_losslessly(self):
        source = {
            "body_source_fingerprint": "1437929698787792806",
            "coupled_program_fingerprint": "13712686933603975462",
            "world_fingerprint": "14697719457569737910",
            "device": "Apple M4 Pro",
        }
        normalized = prepare.normalize_owner_summary_identities(source)
        self.assertEqual(normalized["body_source_fingerprint"], 1437929698787792806)
        self.assertEqual(normalized["coupled_program_fingerprint"], 13712686933603975462)
        self.assertEqual(normalized["world_fingerprint"], source["world_fingerprint"])
        self.assertIs(type(normalized["body_source_fingerprint"]), int)
        self.assertIs(type(normalized["coupled_program_fingerprint"]), int)
        self.assertIs(type(normalized["world_fingerprint"]), str)

    def test_uint64_normalization_rejects_noncanonical_or_unrepresentable_values(self):
        self.assertEqual(prepare.normalize_identity_uint64("1437929698787792806", "fixture"),
                         1437929698787792806)
        self.assertEqual(prepare.normalize_identity_uint64(13712686933603975462, "fixture"),
                         13712686933603975462)
        for value in (True, 1.0, -1, "", "01", "+1", " 1", "١",
                      "18446744073709551616", "999999999999999999999"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare.normalize_identity_uint64(value, "fixture")

    def test_owner_summary_rejects_invalid_uint64_string(self):
        with self.assertRaisesRegex(ValueError, "leading zeroes"):
            prepare.normalize_owner_summary_identities({
                "body_source_fingerprint": "01",
                "coupled_program_fingerprint": "3",
            })


if __name__ == "__main__":
    unittest.main()
