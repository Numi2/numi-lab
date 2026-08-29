from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from metalrobo.crow_world_model import (
    V10_TASK,
    extract_accepted_demonstrations,
    load_replays,
    require_navigation_flight_data,
    route_heading_residual_targets,
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

    def test_extracts_only_native_five_waypoint_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay = Path(directory) / "replay.json"
            output = Path(directory) / "demonstrations.json"
            self._replay(replay)
            value = json.loads(replay.read_text())
            for index, frame in enumerate(value["payload"]["frames"]):
                frame["outcomes"] = {
                    "navigation_waypoints_reached": [0, 1, 5][index],
                    "navigation_completion": float(index == 2),
                    "physics_error": 0.0,
                }
            replay.write_text(json.dumps(value))
            payload = extract_accepted_demonstrations(SimpleNamespace(
                replay=[str(replay)],
                relabel_replay=None,
                relabel_start_step=1,
                minimum_waypoints=5,
                stride=1,
                horizon=1,
                output=str(output),
            ))
            self.assertEqual(
                payload["classification"],
                "accepted native five-waypoint demonstrations",
            )
            self.assertEqual(len(payload["demonstrations"]), 3)
            self.assertEqual(
                payload["sources"][0]["completed_episode_count"], 1
            )
            self.assertEqual(
                payload["demonstrations"][-1]["demonstration_kind"],
                "accepted_completion",
            )

    def test_extracts_explicit_partial_route_as_non_promotable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay = Path(directory) / "replay.json"
            output = Path(directory) / "demonstrations.json"
            self._replay(replay)
            value = json.loads(replay.read_text())
            for index, frame in enumerate(value["payload"]["frames"]):
                frame["outcomes"] = {
                    "navigation_waypoints_reached": [0, 2, 3][index],
                    "navigation_completion": 0.0,
                    "physics_error": 0.0,
                }
            replay.write_text(json.dumps(value))
            payload = extract_accepted_demonstrations(SimpleNamespace(
                replay=[str(replay)],
                relabel_replay=None,
                relabel_start_step=1,
                minimum_waypoints=3,
                development_partial=True,
                stride=1,
                horizon=1,
                output=str(output),
            ))
            self.assertEqual(
                payload["classification"],
                "accepted native partial-route development demonstrations; "
                "non-promotable",
            )
            self.assertEqual(len(payload["demonstrations"]), 3)
            self.assertEqual(
                payload["sources"][0]["completed_episode_count"], 0
            )
            self.assertEqual(
                payload["sources"][0]["development_prefix_episode_count"], 1
            )
            self.assertEqual(
                payload["demonstrations"][-1]["demonstration_kind"],
                "accepted_development_prefix",
            )

    def test_route_heading_residual_changes_only_steering_actions(self) -> None:
        observations = np.zeros((3, 100), dtype=np.float32)
        observations[0, 84:86] = [1.0, 1.0]
        observations[0, 90] = 0.4
        observations[1, 84:86] = [1.0, -1.0]
        observations[1, 90] = 0.4
        observations[2, 84:86] = [1.0, 1.0]
        observations[2, 90] = 0.4
        observations[2, 91] = 1.0
        actions = np.zeros((3, 15), dtype=np.float32)
        targets, active = route_heading_residual_targets(
            observations,
            actions,
            observation_offset=84,
            yaw_gain=0.2,
            sweep_gain=0.1,
            minimum_waypoint_fraction=0.39,
            maximum_waypoint_fraction=0.41,
        )
        self.assertEqual(active.tolist(), [True, True, False])
        self.assertLess(targets[0, 2], 0.0)
        self.assertGreater(targets[0, 3], 0.0)
        self.assertLess(targets[0, 13], 0.0)
        self.assertGreater(targets[1, 2], 0.0)
        self.assertLess(targets[1, 3], 0.0)
        self.assertGreater(targets[1, 13], 0.0)
        np.testing.assert_allclose(targets[2], actions[2])
        np.testing.assert_allclose(targets[:, :2], actions[:, :2])

        reversed_targets, _ = route_heading_residual_targets(
            observations,
            actions,
            observation_offset=84,
            yaw_gain=0.2,
            sweep_gain=0.1,
            yaw_direction=-1,
            sweep_direction=-1,
            minimum_waypoint_fraction=0.39,
            maximum_waypoint_fraction=0.41,
        )
        np.testing.assert_allclose(reversed_targets, -targets)


if __name__ == "__main__":
    unittest.main()
