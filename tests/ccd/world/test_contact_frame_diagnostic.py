#!/usr/bin/env python3
import math
import unittest

from contact_frame_diagnostic import contact_frame, rotate


class ContactFrameTests(unittest.TestCase):
    def test_two_moving_rotating_bodies(self):
        # A's local +x rotates to world +y; B's +y rotates to world -x.
        # Point velocities are A=(0,3,0), B=(0,-2,0); gap along +y is 1.
        s = math.sqrt(.5)
        f = contact_frame((0, 0, 0), (0, 0, s, s),
                          (2, 3, 0), (0, 0, 2), (1, 0, 0),
                          (1, 2, 0), (0, 0, s, s),
                          (0, 1, 0), (0, 0, 3), (0, 1, 0),
                          (0, 1, 0))
        self.assertAlmostEqual(f["point_a"][0], 0)
        self.assertAlmostEqual(f["point_a"][1], 1)
        self.assertAlmostEqual(f["point_b"][0], 0)
        self.assertAlmostEqual(f["point_b"][1], 2)
        self.assertAlmostEqual(f["geometric_gap"], 1)
        self.assertAlmostEqual(f["relative_vn"], -5)
        self.assertAlmostEqual(f["midpoint"][1], 1.5)

    def test_scaled_quaternion_and_invalid(self):
        self.assertAlmostEqual(rotate((0, 0, 3, 3), (1, 0, 0))[1], 1)
        with self.assertRaises(ValueError):
            rotate((0, 0, 0, 0), (1, 0, 0))


if __name__ == "__main__":
    unittest.main()
