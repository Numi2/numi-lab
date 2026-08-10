import unittest

import numpy as np

from metalrobo.upper_body_composition import STANDING_JOINTS, compose_upper_body


class UpperBodyCompositionTest(unittest.TestCase):
    def test_lower_body_is_fixed_and_generated_delta_is_limit_qualified(self):
        source = np.tile(np.linspace(-0.2, 0.2, 29, dtype=np.float32), (4, 1))
        source[1:, 22] -= np.asarray((0.1, 0.2, 0.3), dtype=np.float32)
        source[1:, 23] += np.asarray((0.05, 0.1, 0.15), dtype=np.float32)
        result = compose_upper_body(source)
        np.testing.assert_array_equal(result[:, :15], np.tile(STANDING_JOINTS[:15], (4, 1)))
        self.assertAlmostEqual(float(result[-1, 15]), -0.65)
        self.assertAlmostEqual(float(result[-1, 22]), -0.65)
        self.assertAlmostEqual(
            float(result[-1, 16] - STANDING_JOINTS[16]),
            -float(result[-1, 23] - STANDING_JOINTS[23]),
        )

    def test_other_arm_joint_limits_reduce_uniform_scale(self):
        source = np.zeros((4, 29), dtype=np.float32)
        source[-1, 22] = -0.5
        source[-1, 25] = -2.0
        result = compose_upper_body(source, active_arms=("left",))
        self.assertGreater(float(result[-1, 15]), -0.65)
        self.assertGreaterEqual(float(result[-1, 18]), -1.0472)

    def test_invalid_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            compose_upper_body(np.zeros((4, 28), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
