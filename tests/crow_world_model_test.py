from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from metalrobo.crow_world_model import (
    V10_TASK,
    load_replays,
    require_navigation_flight_data,
)


class CrowWorldModelContractTest(unittest.TestCase):
    def _replay(self, path: Path, *, task: str = V10_TASK, world: str = "11") -> None:
        frames = []
        for step in range(3):
            frames.append({
                "step": step,
                "actor_observation": [float(step), float(step + 1), 0.5, -0.5],
                "accepted_actions": [0.1 * step, -0.1 * step],
                "reward": 0.25 * step,
                "root_height": 0.2 + 0.4 * step,
                "done": step == 2,
            })
        payload = {
            "task": task,
            "actor_observation_count": 4,
            "action_count": 2,
            "world_fingerprint": world,
            "task_fingerprint": "12",
            "observation_fingerprint": "13",
            "action_fingerprint": "14",
            "navigation_course": "training",
            "scheduled_resets": False,
            "frames": frames,
        }
        path.write_text(json.dumps({
            "schema": "numi.crow-replay.v1",
            "payload_sha256": "synthetic",
            "payload": payload,
        }))

    def test_loads_contiguous_accepted_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            self._replay(path)
            dataset = load_replays([path])
            self.assertEqual(dataset.observations.shape, (2, 4))
            self.assertEqual(dataset.actions.shape, (2, 2))
            self.assertEqual(dataset.next_observations.shape, (2, 4))
            np.testing.assert_allclose(dataset.rewards[:, 0], [0.25, 0.5])
            np.testing.assert_allclose(dataset.dones[:, 0], [0.0, 1.0])
            self.assertEqual(dataset.fingerprints["world_fingerprint"], "11")
            self.assertAlmostEqual(dataset.maximum_root_height, 1.0)
            self.assertAlmostEqual(dataset.airborne_frame_fraction, 2.0 / 3.0)
            require_navigation_flight_data(
                dataset,
                minimum_maximum_root_height=0.5,
                minimum_airborne_fraction=0.05,
            )

    def test_rejects_ground_only_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            self._replay(path)
            value = json.loads(path.read_text())
            for frame in value["payload"]["frames"]:
                frame["root_height"] = 0.19
            path.write_text(json.dumps(value))
            dataset = load_replays([path])
            with self.assertRaisesRegex(ValueError, "ground-only"):
                require_navigation_flight_data(
                    dataset,
                    minimum_maximum_root_height=0.5,
                    minimum_airborne_fraction=0.05,
                )

    def test_rejects_non_v10_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            self._replay(path, task="birdflow_american_crow_journey_v9_visual_neural")
            with self.assertRaisesRegex(ValueError, "v10 navigation"):
                load_replays([path])

    def test_rejects_cross_contract_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            self._replay(first, world="11")
            self._replay(second, world="99")
            with self.assertRaisesRegex(ValueError, "fingerprints"):
                load_replays([first, second])


if __name__ == "__main__":
    unittest.main()
