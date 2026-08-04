import tempfile
import unittest
from pathlib import Path

import numpy as np

from metalrobo.foundation_policy import (
    FOUNDATION_ADAPTER_FORMAT,
    FoundationInferenceResult,
    _array_fingerprint,
    _provider_order,
    _validate_foundation_adapter,
)


class FoundationPolicyTest(unittest.TestCase):
    def test_robot_authored_adapter_contract_is_generic_and_validated(self) -> None:
        adapter = {
            "format": FOUNDATION_ADAPTER_FORMAT,
            "id": "test-arm-provider",
            "provider": "test/provider",
            "robot": "test_arm",
            "observation": {
                "root_archive_key": "root",
                "root_q_offset": 0,
                "root_q_count": 7,
                "joint_q_offset": 7,
                "state_groups": [
                    {"name": "arm", "joints": ["joint_a"], "placeholder_count": 0},
                ],
            },
            "action_outputs": [{"name": "arm", "joints": ["joint_a"]}],
            "controller": {
                "joint_order": ["joint_a"],
                "default_pose": [0.0],
                "task_action_scale": [0.5],
                "velocity_limits": [1.0],
                "position_limits": [[-1.0, 1.0]],
                "policy_timestep_seconds": 0.02,
            },
            "interaction": {
                "contact_tracks": [{
                    "id": "tool",
                    "task_contact_group": "tool_contact",
                    "counterpart": "workpiece",
                    "mode": 1,
                    "confidence": 0.75,
                }],
            },
        }
        _validate_foundation_adapter(adapter)
        adapter["interaction"]["contact_tracks"] = []
        _validate_foundation_adapter(adapter)
        adapter["action_outputs"][0]["joints"] = ["unknown_joint"]
        with self.assertRaises(ValueError):
            _validate_foundation_adapter(adapter)

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
