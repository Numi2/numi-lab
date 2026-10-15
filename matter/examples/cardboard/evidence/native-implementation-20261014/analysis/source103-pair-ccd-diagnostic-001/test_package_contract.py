#!/usr/bin/env python3
"""Small fail-closed tests for package admission and exact input packing."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('pair_package_verifier', HERE/'verify_pair_package.py')
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)

class PackageContractTests(unittest.TestCase):
    def test_current_prepared_package_and_speed_bounds(self):
        d, summary = VERIFY.validate_prepared(HERE/'prepared/prepared.json')
        self.assertEqual(summary['case_count'], 30)
        self.assertEqual(summary['case_counts'], {'vertex_triangle': 12, 'edge_edge': 18})
        self.assertGreater(summary['speed_bound_ratio_ranges']['vertex_triangle']['max'], 6.0)
        self.assertGreater(summary['speed_bound_ratio_ranges']['edge_edge']['max'], 2.0)
        self.assertFalse(summary['native_result_admitted'])

    def test_source_binding_tamper_rejected(self):
        original=VERIFY.strict_json(HERE/'prepared/prepared.json')
        original['source_contact_sha256']='0'*64
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'prepared.json'
            path.write_text(json.dumps(original))
            (Path(temp)/'cases.bin').write_bytes((HERE/'prepared/cases.bin').read_bytes())
            with self.assertRaisesRegex(ValueError, 'frozen source hash mismatch'):
                VERIFY.validate_prepared(path)

    def test_binary_tamper_rejected(self):
        original=VERIFY.strict_json(HERE/'prepared/prepared.json')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'cases.bin').write_bytes((HERE/'prepared/cases.bin').read_bytes())
            # Also copy the manifest's receipt path is immutable and validated;
            # this test reaches the binary gate using a local temporary copy of
            # the prepared record after rebinding only that byte count/hash.
            # Avoid modifying a receipt; instead test the isolated exact-packing
            # invariant directly by changing one byte after the canonical hash.
            raw=bytearray((HERE/'prepared/cases.bin').read_bytes())
            raw[0] ^= 1
            (root/'cases.bin').write_bytes(raw)
            dpath=root/'prepared.json'
            dpath.write_text(json.dumps(VERIFY.strict_json(HERE/'prepared/prepared.json')))
            with self.assertRaisesRegex(ValueError, 'input binary hash mismatch'):
                VERIFY.validate_prepared(dpath)

    def test_exact_formula_rejects_nonconservative_claim_fields(self):
        original=VERIFY.strict_json(HERE/'prepared/prepared.json')
        original['cases'][0]['cpu_fp64_reference']['speed_bound_comparison']['galilean_relative_lipschitz_bound_m'] *= 0.5
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'cases.bin').write_bytes((HERE/'prepared/cases.bin').read_bytes())
            (root/'prepared.json').write_text(json.dumps(original))
            # Keep the genuine source receipt reachable via the absolute path.
            with self.assertRaisesRegex(ValueError, 'relative bound mismatch'):
                VERIFY.validate_prepared(root/'prepared.json')

if __name__ == '__main__':
    unittest.main(verbosity=2)
