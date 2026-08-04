import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from metalrobo.ardy_onnx import (
    ARDY_MOTION_FORMAT,
    ARDYMotionResult,
    _cosine_diffusion,
    _output_parity,
    _physical_tracking_outcome,
    _provider_attempts,
    _validate_contract,
)


class ARDYONNXTest(unittest.TestCase):
    def test_diffusion_schedule_is_finite_and_monotonic(self) -> None:
        cumulative, previous = _cosine_diffusion()
        self.assertEqual(cumulative.shape, (10,))
        self.assertTrue(np.isfinite(cumulative).all())
        self.assertTrue(np.all(np.diff(cumulative) < 0.0))
        self.assertEqual(float(previous[0]), 1.0)

    def test_contract_rejects_wrong_horizon(self) -> None:
        contract = {
            "model": "ARDY-Core-RP-20FPS-Horizon40", "skeleton": "cskel27",
            "fps": 20, "horizon_frames": 8, "frames_per_token": 4,
            "max_tokens": 64, "max_frames": 256, "diffusion_steps": 10,
            "hybrid_dim": 148, "root_dim": 5, "latent_dim": 128,
            "motion_dim": 330, "body_dim": 325, "local_root_dim": 4,
            "text_dim": 4096, "joint_count": 27, "joint_names": [""] * 27,
            "joint_parents": [-1] * 27, "motion_mean": [0.0] * 330,
            "motion_std": [1.0] * 330, "local_root_mean": [0.0] * 4,
            "local_root_std": [1.0] * 4,
        }
        with self.assertRaises(ValueError):
            _validate_contract(contract)

    def test_motion_result_writes_fingerprinted_artifact(self) -> None:
        arrays = {"root_positions": np.zeros((40, 3), dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            ARDYMotionResult(arrays, {"format": ARDY_MOTION_FORMAT}).write(target)
            evidence = json.loads((target / "evidence.json").read_text())
            self.assertEqual(evidence["format"], ARDY_MOTION_FORMAT)
            self.assertEqual(evidence["motion_proposal"]["shapes"]["root_positions"], [40, 3])

    def test_provider_attempts_are_stage_local(self) -> None:
        available = ("CoreMLExecutionProvider", "CPUExecutionProvider")
        self.assertEqual(
            _provider_attempts("auto", available),
            (
                (
                    "coreml",
                    ("CoreMLExecutionProvider", "CPUExecutionProvider"),
                ),
                ("cpu", ("CPUExecutionProvider",)),
            ),
        )
        self.assertEqual(
            _provider_attempts("cpu", available),
            (("cpu", ("CPUExecutionProvider",)),),
        )

    def test_coreml_candidate_requires_cpu_parity(self) -> None:
        reference = np.asarray((1.0, 2.0, 3.0), dtype=np.float32)
        close = reference + np.asarray((1.0e-5, -1.0e-5, 0.0), dtype=np.float32)
        wrong = reference + np.asarray((0.0, 0.0, 0.1), dtype=np.float32)
        self.assertTrue(_output_parity((close,), (reference,))["passed"])
        self.assertFalse(_output_parity((wrong,), (reference,))["passed"])

    def test_physical_tracking_reports_progress_without_gating(self) -> None:
        reference = {
            "root_position_quaternion_xyzw": np.asarray(
                ((0, 0, 1, 0, 0, 0, 1), (1, 0, 1, 0, 0, 0, 1)),
                dtype=np.float32,
            ),
            "joint_positions": np.zeros((2, 2), dtype=np.float32),
        }
        solver = {
            "root_position_quaternion_xyzw": np.asarray(
                ((0.25, 0, 1, 0, 0, 0, 1), (0.5, 0, 1, 0, 0, 0, 1)),
                dtype=np.float32,
            ),
            "joint_positions": np.zeros((2, 2), dtype=np.float32),
        }
        outcome = _physical_tracking_outcome(reference, solver, 1.0, 2.0)
        self.assertEqual(outcome["projected_progress_ratio"], 0.25)
        self.assertIn("not a promotion", outcome["semantics"])


if __name__ == "__main__":
    unittest.main()
