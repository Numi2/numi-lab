#!/usr/bin/env python3
from decimal import Decimal, localcontext
import unittest
import analyze_slider_split_representability as subject

class SplitTests(unittest.TestCase):
    def test_nearest_word_handles_midpoint_and_tiny_offset(self):
        with localcontext() as context:
            context.prec = 100
            tie = Decimal(1) + Decimal(2) ** -24
            self.assertEqual(subject.nearest_fp32(tie), 1.0)
            self.assertEqual(subject.nearest_fp32(tie + Decimal(2) ** -90),
                             1.0 + 2.0 ** -23)
            self.assertEqual(subject.nearest_fp32(-tie), -1.0)
            self.assertEqual(subject.nearest_fp32(-tie - Decimal(2) ** -90),
                             -1.0 - 2.0 ** -23)

    def test_split_improves_exact_first_retained_step(self):
        with localcontext() as context:
            context.prec = 70
            result = subject.analyze(subject.E / (subject.NAMES[0] + ".json"))
        self.assertGreater(result["single_exceeds_gate"], 0)
        self.assertEqual(result["split_exceeds_gate"], 0)
        self.assertLess(result["split_max_residual_Ns"], result["single_max_residual_Ns"])

if __name__ == "__main__":
    unittest.main()
