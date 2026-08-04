import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from metalrobo.ardy_interaction_convert import (
    read_interaction_pack,
    write_interaction_pack,
)
from metalrobo.foundation_policy import (
    FOUNDATION_ADAPTER_FORMAT,
    FoundationInferenceResult,
    G1_GROUP_JOINTS,
    G1_HAND_ACTION_JOINTS,
    G1_HAND_STATE_JOINTS,
    _array_fingerprint,
    compile_numi_observation,
    _compose_joint_proposal,
    _g1_foundation_adapter,
    _provider_order,
    _validate_foundation_adapter,
)


class FoundationPolicyTest(unittest.TestCase):
    def test_observation_requires_explicit_model_constant_authority(self) -> None:
        adapter = {
            "format": FOUNDATION_ADAPTER_FORMAT,
            "id": "test-cross-embodiment-provider",
            "provider": "test/provider",
            "robot": "test_arm",
            "observation": {
                "root_archive_key": "root",
                "root_q_offset": 0,
                "root_q_count": 7,
                "joint_q_offset": 7,
                "state_groups": [
                    {
                        "name": "arm",
                        "source": {
                            "kind": "joint_positions",
                            "joints": ["joint_a"],
                        },
                    },
                    {
                        "name": "model_hand",
                        "source": {
                            "kind": "model_constant",
                            "values": [0.0, 0.0],
                            "semantics": "provider hand absent from the robot",
                        },
                    },
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
            "interaction": {"contact_tracks": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = root / "camera.ppm"
            trace = root / "state.tsv"
            output = root / "observation.npz"
            evidence = root / "evidence.json"
            camera.write_bytes(
                b"P6\n640 480\n255\n" + bytes(640 * 480 * 3)
            )
            trace.write_text(
                "# step nq=8\n0\t0\t0\t0\t0\t0\t0\t1\t0.25\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "explicit model-only constants"):
                compile_numi_observation(
                    camera, trace, output, evidence, None, adapter
                )
            result = compile_numi_observation(
                camera,
                trace,
                output,
                evidence,
                None,
                adapter,
                allow_model_constants=True,
            )
            self.assertEqual(
                list(result["model_constant_state_groups"]), ["model_hand"]
            )
            with np.load(output, allow_pickle=False) as archive:
                self.assertTrue(
                    np.array_equal(
                        archive["model_hand"],
                        np.zeros((1, 2), dtype=np.float32),
                    )
                )

    def test_g1_dex3_contract_maps_exact_hand_state_and_action_orders(self) -> None:
        joints = tuple(
            joint
            for group in (
                *G1_GROUP_JOINTS.values(),
                *G1_HAND_STATE_JOINTS.values(),
            )
            for joint in group
        )
        contract = {
            "robot": "unitree_g1_dex3",
            "joint_order": list(joints),
            "default_pose": [0.0] * len(joints),
            "task_action_scale": [0.25] * len(joints),
            "velocity_limits": [1.0] * len(joints),
            "position_limits": [[-1.0, 1.0]] * len(joints),
            "policy_timestep_seconds": 0.02,
            "solver_root_frame": "center_of_mass",
            "root_center_of_mass_local_xyz": [0.0, 0.0, 0.0],
        }
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "libmetalrobo.dylib"
            library.write_bytes(b"test-library")
            with patch(
                "metalrobo.foundation_policy._native_g1_contract",
                return_value=contract,
            ):
                adapter = _g1_foundation_adapter(library)
        state_groups = {
            group["name"]: group
            for group in adapter["observation"]["state_groups"]
        }
        outputs = {
            group["name"]: group
            for group in adapter["action_outputs"]
        }
        for side in ("left", "right"):
            name = f"{side}_hand"
            self.assertEqual(
                state_groups[name]["source"]["joints"],
                list(G1_HAND_STATE_JOINTS[name]),
            )
            self.assertEqual(
                state_groups[name]["source"]["kind"],
                "joint_positions",
            )
            self.assertEqual(
                outputs[name]["joints"],
                list(G1_HAND_ACTION_JOINTS[side]),
            )
            self.assertNotIn(name, adapter["unmapped_output_semantics"])

    def test_g1_29dof_contract_exposes_model_constants_without_fake_hands(self) -> None:
        joints = tuple(
            joint for group in G1_GROUP_JOINTS.values() for joint in group
        )
        contract = {
            "robot": "unitree_g1",
            "joint_order": list(joints),
            "default_pose": [0.0] * len(joints),
            "task_action_scale": [0.25] * len(joints),
            "velocity_limits": [1.0] * len(joints),
            "position_limits": [[-1.0, 1.0]] * len(joints),
            "policy_timestep_seconds": 0.02,
            "solver_root_frame": "center_of_mass",
            "root_center_of_mass_local_xyz": [0.0, 0.0, 0.0],
        }
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "libmetalrobo.dylib"
            library.write_bytes(b"test-library")
            with patch(
                "metalrobo.foundation_policy._native_g1_contract",
                return_value=contract,
            ):
                adapter = _g1_foundation_adapter(library)
        state_groups = {
            group["name"]: group
            for group in adapter["observation"]["state_groups"]
        }
        outputs = {output["name"] for output in adapter["action_outputs"]}
        for side in ("left", "right"):
            name = f"{side}_hand"
            source = state_groups[name]["source"]
            self.assertEqual(source["kind"], "model_constant")
            self.assertEqual(source["values"], [0.0] * 7)
            self.assertIn("no corresponding mechanics", source["semantics"])
            self.assertNotIn(name, outputs)
            self.assertIn(name, adapter["unmapped_output_semantics"])

    def test_interaction_pack_reader_authenticates_and_preserves_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.interactionpack"
            roots = np.asarray(
                ((0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0),
                 (0.1, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0)),
                dtype=np.float32,
            )
            joints = np.asarray(((0.0,), (0.1,)), dtype=np.float32)
            write_interaction_pack(
                output=path,
                pack_id="base",
                clip_id="walk",
                desired_outcome="walk forward",
                source_repository="test/provider",
                source_revision="revision",
                license_name="test",
                frames_per_second=10.0,
                root_targets=roots,
                joint_targets=joints,
                tracks=(),
                contact_modes=np.empty((2, 0), dtype=np.uint32),
                contact_confidence=np.empty((2, 0), dtype=np.float32),
                joint_names=("joint",),
                joint_lower=np.asarray((-1.0,), dtype=np.float32),
                joint_upper=np.asarray((1.0,), dtype=np.float32),
                joint_velocity=np.asarray((2.0,), dtype=np.float32),
            )
            decoded = read_interaction_pack(path)
            self.assertEqual(decoded.id, "base")
            self.assertEqual(decoded.joint_names, ("joint",))
            self.assertEqual(decoded.clips[0].id, "walk")
            self.assertTrue(np.array_equal(decoded.clips[0].root_targets, roots))
            self.assertTrue(np.array_equal(decoded.clips[0].joint_targets, joints))

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
                    {
                        "name": "arm",
                        "source": {
                            "kind": "joint_positions",
                            "joints": ["joint_a"],
                        },
                    },
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

    def test_motion_base_composition_preserves_unmapped_joints(self) -> None:
        base = np.asarray(
            ((0.0, 0.5, -0.5), (0.1, 0.6, -0.4), (0.2, 0.7, -0.3)),
            dtype=np.float32,
        )
        proposal = np.asarray(((1.0, 9.0, 8.0), (2.0, 9.0, 8.0)), dtype=np.float32)
        composed, evidence = _compose_joint_proposal(
            base,
            ("arm", "left_leg", "right_leg"),
            proposal,
            ("arm", "left_leg", "right_leg"),
            ("arm",),
            0.25,
        )
        self.assertTrue(np.array_equal(composed[:, 1:], base[:, 1:]))
        self.assertTrue(np.allclose(composed[:, 0], (0.25, 0.45, 0.65)))
        self.assertEqual(evidence["mapped_joint_names"], ["arm"])
        self.assertEqual(evidence["base_frames"], 3)
        self.assertEqual(evidence["proposal_frames"], 2)

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
