#!/usr/bin/env python3
"""Source contract for fresh Newton-direction common-alpha initialization.

This checks dispatch placement and scratch ownership; the native Metal fixture
separately exercises the kernel writes and unhealthy-environment behavior.
"""
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
MATTER = HERE.parent
RUNTIME = (MATTER / "src/runtime.mm").read_text()
FEM = (MATTER / "src/metal/fem.metalinc").read_text()
NATIVE = (HERE / "fgmres_true_residual_native_check.mm").read_text()


class FreshDirectionAlphaResetContract(unittest.TestCase):
    def test_kernel_resets_only_alpha_for_healthy_environments(self):
        kernel = FEM[
            FEM.index("kernel void nm_fem_reset_fresh_direction_common_alpha("):
            FEM.index("// The Newton/KKT unknown is environment-wide:",
                      FEM.index("kernel void nm_fem_reset_fresh_direction_common_alpha("))
        ]
        self.assertIn("environmentHealthy(statuses, environment)", kernel)
        self.assertIn("environmentLineSearch[environment] = 1.0f", kernel)
        self.assertIn("state.x = 1.0f", kernel)
        self.assertNotRegex(kernel, r"state\.[yzw]\s*=")
        self.assertIn('"nm_fem_reset_fresh_direction_common_alpha"', RUNTIME)
        self.assertIn(
            '"numi_matter_metal::nm_fem_reset_fresh_direction_common_alpha"',
            NATIVE)

    def test_reset_is_strict_default_off_and_fingerprinted_only_when_enabled(self):
        self.assertRegex(
            RUNTIME,
            r'std::getenv\(\s*"NM_FEM_FRESH_DIRECTION_ALPHA_RESET"\s*\)')
        self.assertIn('std::strcmp(freshDirectionAlphaReset, "0") != 0', RUNTIME)
        self.assertIn('std::strcmp(freshDirectionAlphaReset, "1") != 0', RUNTIME)
        self.assertIn('freshDirectionAlphaReset != nullptr &&', RUNTIME)
        self.assertIn('std::strcmp(freshDirectionAlphaReset, "1") == 0', RUNTIME)
        self.assertIn(
            'if (candidate->femFreshDirectionAlphaResetEnabled) {', RUNTIME)
        self.assertIn('0x4644414c50484131ull', RUNTIME)
        self.assertIn('if (!state.femFreshDirectionAlphaResetEnabled) return;', RUNTIME)
        self.assertRegex(
            RUNTIME,
            r'bool femFreshDirectionAlphaResetEnabled = false;')

    def test_reset_occurs_once_before_each_new_direction_search(self):
        self.assertEqual(RUNTIME.count("resetFreshDirectionCommonAlpha();"), 2)
        passes = RUNTIME[
            RUNTIME.index("for (std::uint32_t directionPass = 0u;"):
            RUNTIME.index("} // Primary Newton and optional same-history Cauchy pass.")
        ]
        reset = passes.index("resetFreshDirectionCommonAlpha();")
        trials = passes.index("for (std::uint32_t trialIndex = 0u;")
        self.assertLess(reset, trials)
        # The optional second pass saves the primary result and constructs its
        # Cauchy direction before resetting for that distinct direction.
        second = passes[passes.index("if (directionPass == 1u)"):reset]
        self.assertIn("nm_fgmres_base_cgls_best_of_save", second)
        self.assertIn("encodeBaseCauchyDirection();", second)
        self.assertNotIn("resetFreshDirectionCommonAlpha();", passes[trials:])

        standard = RUNTIME[
            RUNTIME.index("} else {\n                resetFreshDirectionCommonAlpha();",
                          RUNTIME.index("} // Primary Newton and optional same-history Cauchy pass.")):
            RUNTIME.index('"nm_fem_parallel_backtracking_initialize"',
                          RUNTIME.index("} else {\n                resetFreshDirectionCommonAlpha();",
                                        RUNTIME.index("} // Primary Newton and optional same-history Cauchy pass.")))
        ]
        self.assertLess(standard.index("resetFreshDirectionCommonAlpha();"),
                        standard.index("collapseCompletedBacktracking"))

    def test_native_fixture_covers_preserved_diagnostics_and_failed_env(self):
        self.assertIn("--fresh-direction-common-alpha-reset-only", NATIVE)
        self.assertIn("unhealthy environment object line-search state was modified", NATIVE)
        self.assertIn("fresh direction reset overwrote object search diagnostics", NATIVE)


if __name__ == "__main__":
    unittest.main()
