import tempfile
import unittest
from pathlib import Path

import numpy as np

from metalrobo.g1_legs_policy import expand_leg_actions, inspect_onnx


class G1LegsPolicyTest(unittest.TestCase):
    def test_expand_leg_actions_preserves_legs_and_holds_upper_body(
        self,
    ) -> None:
        legs = np.arange(12, dtype=np.float32)[None, :]
        full = expand_leg_actions(legs)
        self.assertEqual(full.shape, (1, 29))
        np.testing.assert_array_equal(full[:, :12], legs)
        np.testing.assert_array_equal(full[:, 12:], 0.0)

    def test_inspect_onnx_rejects_unknown_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            unknown = Path(directory) / "unknown.onnx"
            unknown.write_bytes(b"not an onnx model")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                inspect_onnx(unknown)


if __name__ == "__main__":
    unittest.main()
