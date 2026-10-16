"""Compare the FP64 reference with a portable independent 90-digit witness."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

import cardboard_bilinear_fp64_reference as reference


class ReferenceFischerBurmeisterJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parent / 'test-data/cardboard/reference-fb-tet846.json'
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == '391c170098bf0e76583c772c6508969e80ebdf1695982ddf721b466afe28b514'
        cls.fixture = json.loads(raw)

    def test_elastic_seed_and_resolved_root_match_independent_mp_witness(self):
        fixture = self.fixture
        p = np.asarray(fixture['parameters'], dtype=np.float64)[None, :]
        f = np.asarray(fixture['F'], dtype=np.float64)[None, :, :]
        old = np.asarray(fixture['accepted_history'], dtype=np.float64)[None, :]
        for row in fixture['rows']:
            with self.subTest(state=row['name']):
                state = np.asarray(row['state'], dtype=np.float64)[None, :]
                actual = reference.jacobian(p, f, old, state)[0, 12]
                expected = np.asarray(row['fb_jacobian_90_digit'], dtype=np.float64)
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                if row['name'] == 'elastic_trial':
                    self.assertEqual(actual[12], -1.0)

    def test_projection_acceptance_limits_remain_strict(self):
        self.assertEqual(reference.TIGHT_RESIDUAL, 2e-12)
        self.assertEqual(reference.TIGHT_DYNAMIC_CORRECTION, 2e-11)


if __name__ == '__main__':
    unittest.main()
