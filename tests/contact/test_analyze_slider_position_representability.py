#!/usr/bin/env python3
from decimal import localcontext
import json
from pathlib import Path
import tempfile
import unittest
import analyze_slider_position_representability as subject

class PositionTests(unittest.TestCase):
    def test_wrong_published_endpoint_cannot_pass(self):
        data = {"h": .001, "initial_slider_velocity_mps": 0.,
                "steps": [{"accepted": True, "q": .01, "v": 0.},
                          {"accepted": False}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(data))
            with localcontext() as context:
                context.prec = 70
                result = subject.analyze(path)
        self.assertEqual(result["accepted_steps"], 1)
        self.assertEqual(result["published_exceeds_bound"], 1)
        self.assertEqual(result["single_exceeds_bound"], 0)
        self.assertEqual(result["split_exceeds_bound"], 0)

if __name__ == "__main__":
    unittest.main()
