import tempfile
import unittest
from pathlib import Path

import numpy as np

from metalrobo.foundation_policy import (
    FoundationInferenceResult,
    _array_fingerprint,
    _provider_order,
)


class FoundationPolicyTest(unittest.TestCase):
    def test_provider_selection_preserves_correctness_fallback(self) -> None:
        available = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self.assertEqual(
            _provider_order("auto", available),
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        self.assertEqual(_provider_order("cpu", available), ["CPUExecutionProvider"])
        with self.assertRaises(RuntimeError):
            _provider_order("coreml", ["CPUExecutionProvider"])

    def test_action_chunk_is_fingerprinted_and_reproducible(self) -> None:
        actions = {
            "left_arm": np.zeros((1, 16, 7), dtype=np.float32),
            "base_height_command": np.ones((1, 16, 1), dtype=np.float32),
        }
        fingerprint = _array_fingerprint(actions)
        self.assertEqual(fingerprint, _array_fingerprint(dict(reversed(list(actions.items())))))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            FoundationInferenceResult(actions, {"format": "test"}).write(output)
            self.assertTrue((output / "action_chunk.npz").is_file())
            evidence = (output / "evidence.json").read_text(encoding="utf-8")
            self.assertIn(fingerprint, evidence)


if __name__ == "__main__":
    unittest.main()
