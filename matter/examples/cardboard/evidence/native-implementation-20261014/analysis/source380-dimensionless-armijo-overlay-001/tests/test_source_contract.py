from __future__ import annotations

import unittest
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "source/matter/src/metal/common.metalinc").read_text()


class SourceContractTests(unittest.TestCase):
    def test_existing_material_policy_is_the_only_selector(self):
        self.assertIn("NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION", SOURCE)
        self.assertIn("const bool scaledCorrectionPolicy = usesScaledCorrectionProjection(material);", SOURCE)

    def test_raw_residual_and_scaled_correction_convergence_gates_remain(self):
        self.assertIn("residualNorm <= projectionTolerance * (1.0f + stateNorm)", SOURCE)
        self.assertIn("scaledCorrectionNorm <= 2.0e-8f", SOURCE)

    def test_armijo_is_strict_and_legacy_path_keeps_raw_merit(self):
        self.assertIn("trialScaledNorm < scaledResidualNorm", SOURCE)
        self.assertIn("(1.0f - 1.0e-4f * step) * scaledResidualNorm", SOURCE)
        self.assertIn("trialNorm <= (1.0f - 1.0e-4f * step) * residualNorm", SOURCE)

    def test_15_state_trace_padding_fits_16_trials_independent_of_trace_depth(self):
        self.assertIn("kMaterialProjectionArmijoTrialCapacity = 16u", SOURCE)
        self.assertIn("material.stateCount * material.stateCount +\n                material.stateCount +\n                kMaterialProjectionArmijoTrialCapacity <=", SOURCE)


if __name__ == "__main__":
    unittest.main()
