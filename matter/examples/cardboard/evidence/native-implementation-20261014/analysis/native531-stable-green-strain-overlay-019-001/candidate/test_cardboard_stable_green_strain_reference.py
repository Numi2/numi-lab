"""Focused regression for stable Green-strain formation at the retained 531 row."""
import json
import pathlib
import unittest

import numpy as np
import cardboard_bilinear_fp64_reference as base
import cardboard_compensated_fp64_reference as strict


FIXTURE = pathlib.Path(__file__).with_name("cardboard_stable_green_strain_tet222_col6.json")


def branch_signature(p, old, z):
    fields = base.paper_branch(base, p, old, z)
    dl = fields["plastic_multiplier_increment"]
    y = fields["yield_function_residual_value"]
    den = np.sqrt(dl * dl + y * y + p[:, 26] ** 2) + dl + y
    return np.stack((np.sign(dl), np.sign(y), den > 1.0e-14), axis=1)


class StableGreenStrainRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())
        inp = cls.fixture["input"]
        cls.p = np.asarray(inp["parameters"], dtype=np.float64)[None, :]
        cls.f = np.asarray(inp["F_center"], dtype=np.float64)[None, :, :]
        cls.old = np.asarray(inp["accepted_history"], dtype=np.float64)[None, :]
        cls.h = np.asarray(inp["H"], dtype=np.float64)[None, :, :]
        cls.engine = strict.Reference(base)

    def test_original_implementation_misses_retained_case(self):
        # This is a recorded, source-hash-bound old result, not a recomputed
        # result from the patched module.
        old = self.fixture["expected"]["legacy_relative_errors"]
        self.assertEqual([33, 34], self.fixture["input"]["epsilon_powers"])
        self.assertTrue(all(value > 1.0e-6 for value in old), old)

    def test_center_and_both_fine_centered_differences_pass_strict_gates(self):
        exp = self.fixture["expected"]
        z, iterations, residual, correction, valid, reasons, _ = \
            self.engine.project_with_metrics(self.p, self.f, self.old)
        self.assertTrue(valid[0], reasons)
        self.assertLessEqual(float(residual[0]), strict.STRICT_RESIDUAL)
        self.assertLessEqual(float(correction[0]), strict.STRICT_DYNAMIC_CORRECTION)
        self.assertEqual(int(iterations[0]), exp["center_iterations"])
        self.assertTrue(np.array_equal(branch_signature(self.p, self.old, z)[0],
                                       np.asarray(exp["center_branch"])))
        np.testing.assert_allclose(z[0], exp["center_state"], rtol=1e-9, atol=1e-15)

        tangent = strict.tangent(base, self.p, self.f, self.old, z, self.h)[0]
        np.testing.assert_allclose(tangent, exp["tangent"], rtol=2e-10, atol=2e-8)
        center_stress = base.stress(self.p, self.f, z)[0]
        for item in exp["directions"]:
            eps = 2.0 ** (-int(item["power"]))
            endpoints = []
            for sign in (1.0, -1.0):
                f1 = self.f + sign * eps * self.h
                z1, it1, rn1, cn1, ok1, why1, _ = \
                    self.engine.project_with_metrics(self.p, f1, self.old)
                self.assertTrue(ok1[0], (item["power"], sign, why1))
                self.assertLessEqual(float(rn1[0]), strict.STRICT_RESIDUAL)
                self.assertLessEqual(float(cn1[0]), strict.STRICT_DYNAMIC_CORRECTION)
                self.assertEqual(int(it1[0]), 2)
                signature = branch_signature(self.p, self.old, z1)[0]
                self.assertTrue(np.array_equal(signature, np.asarray(exp["center_branch"])))
                endpoints.append(base.stress(self.p, f1, z1)[0])
            fd = (endpoints[0] - endpoints[1]) / (2.0 * eps)
            relative = float(np.linalg.norm(fd - tangent) / np.linalg.norm(tangent))
            self.assertLessEqual(relative, 1.0e-6)
            self.assertTrue(item["same_branch_both"])
            self.assertAlmostEqual(relative, item["centered_relative_error"], delta=2.0e-10)


if __name__ == "__main__":
    unittest.main()
