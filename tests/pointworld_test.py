import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from metalrobo.pointworld import (
    CHECKPOINT_CONTRACT_SHA256,
    DINO_REVISION,
    DROID_BEHAVIOR_STATS_SHA256,
    POINTWORLD_CHECKPOINT,
    POINTWORLD_CHECKPOINT_SHA256,
    POINTWORLD_MODEL_REPOSITORY,
    POINTWORLD_MODEL_REVISION,
    POINTWORLD_SOURCE_REVISION,
    PTV3_BLUEPRINT_SHA256,
    _fingerprint_arrays,
    _release_confidence,
    compile_observation,
    compile_robot_flow_candidates,
    evaluate_forecast,
    rank_forecast,
    seal_reference_forecast,
    validate_model_pack,
)


class PointWorldTest(unittest.TestCase):
    def _model_pack(self, root: Path):
        manifest = {
            "format": "numi.pointworld-model-pack.v1",
            "id": "pointworld-large-droid-behavior-v1",
            "source": {"repository": "NVlabs/PointWorld", "revision": POINTWORLD_SOURCE_REVISION, "license": "Apache-2.0"},
            "checkpoint": {
                "repository": POINTWORLD_MODEL_REPOSITORY, "revision": POINTWORLD_MODEL_REVISION,
                "path": POINTWORLD_CHECKPOINT, "sha256": POINTWORLD_CHECKPOINT_SHA256,
                "license": "nvidia-open-model-license",
            },
            "dino": {"revision": DINO_REVISION, "weights_sha256": "d" * 64, "license": "DINOv3 License"},
            "release_assets": {
                "normalization_path": "stats/droid_behavior/norm_stats.json", "normalization_sha256": DROID_BEHAVIOR_STATS_SHA256,
                "ptv3_blueprint_path": "ptv3/ptv3_arch.yaml", "ptv3_blueprint_sha256": PTV3_BLUEPRINT_SHA256,
                "checkpoint_contract_path": "pointworld/checkpoint_contract.py", "checkpoint_contract_sha256": CHECKPOINT_CONTRACT_SHA256,
            },
            "architecture": {
                "scene_encoder": "dinov3_vitl16", "scene_encoder_layers": [4, 11, 17, 23],
                "ptv3_size": "large", "ptv3_patch_size": 256, "predictor_dim": 256,
                "grid_size_m": 0.015, "depth_threshold_m": 0.003,
                "robot_features": ["robot_flows", "robot_colors", "robot_normals", "gripper_open", "robot_velocity", "robot_acceleration"],
                "scene_features": ["scene_flows", "scene_colors", "scene_normals", "gripper_open", "dist2robot"],
            },
            "license_receipts": {
                "nvidia_open_model_license_accepted": True,
                "dinov3_access_granted": True,
                "dinov3_license_accepted": True,
            },
            "contract": {
                "image_width": 320, "image_height": 180, "context_frames": 1,
                "prediction_frames": 10, "max_scene_points": 12000,
                "max_robot_points": 500, "domains": ["droid", "behavior"],
            },
        }
        path = root / "pointworld.modelpack.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return validate_model_pack(manifest), path

    def _observation(self, root: Path, pack):
        rgb = np.zeros((1, 180, 320, 3), dtype=np.uint8)
        depth = np.ones((1, 180, 320), dtype=np.float32)
        source = root / "capture.npz"
        np.savez_compressed(
            source, rgb=rgb, depth=depth,
            depth_validity=np.ones(depth.shape, dtype=np.bool_),
            intrinsic=np.asarray([[[200, 0, 160], [0, 200, 90], [0, 0, 1]]], dtype=np.float32),
            camera_to_world=np.eye(4, dtype=np.float32)[None],
            timestamp_ns=np.asarray([9], dtype=np.uint64),
            frame_fingerprint=np.asarray(["a" * 64]),
        )
        observation = root / "observation.npz"
        result = compile_observation(model_pack=pack, source=source, output=observation)
        return observation, result

    def test_release_shaped_artifacts_evaluate_and_plan_with_bound_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, pack_path = self._model_pack(root)
            observation, observation_result = self._observation(root, pack)
            self.assertLessEqual(observation_result["scene_point_count"], 12000)
            self.assertEqual(observation_result["coordinate_frame"], "world")

            transforms = np.broadcast_to(np.eye(4, dtype=np.float64), (2, 11, 1, 4, 4)).copy()
            transforms[1, :, 0, 0, 3] = np.linspace(0.0, 0.25, 11)
            candidate_source = root / "candidate-source.npz"
            candidate_arrays = {
                "candidate_names": np.asarray(["hold", "push-right"]),
                "link_transforms": transforms,
                "surface_points_local": np.asarray([[0, 0, 0], [0, 0.01, 0], [0, -0.01, 0]], dtype=np.float64),
                "surface_normals_local": np.asarray([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64),
                "surface_link_indices": np.zeros(3, dtype=np.uint32),
                "gripper_open": np.ones((2, 11, 1), dtype=np.float32),
            }
            candidate_arrays["visual_geometry_fingerprint"] = np.asarray([_fingerprint_arrays({
                "surface_points_local": candidate_arrays["surface_points_local"],
                "surface_normals_local": candidate_arrays["surface_normals_local"],
                "surface_link_indices": candidate_arrays["surface_link_indices"],
            })])
            candidate_arrays["robot_topology_fingerprint"] = np.asarray([_fingerprint_arrays({
                "surface_link_indices": candidate_arrays["surface_link_indices"],
                "link_count": np.asarray([1], dtype=np.uint32),
            })])
            candidate_arrays["source_fingerprint"] = np.asarray([_fingerprint_arrays(candidate_arrays)])
            np.savez_compressed(candidate_source, **candidate_arrays)
            candidates = root / "candidates.npz"
            candidate_result = compile_robot_flow_candidates(
                source=candidate_source, observation=observation,
                observation_manifest=observation.with_suffix(".json"), output=candidates,
            )
            self.assertEqual(candidate_result["candidate_count"], 2)
            with np.load(candidates, allow_pickle=False) as archive:
                self.assertEqual(archive["scene_dist2robot"].shape[:2], (2, 11))
                self.assertEqual(archive["robot_velocity"].shape, (2, 11, 3, 3))

            point_count = observation_result["scene_point_count"]
            flow = np.zeros((2, 10, point_count, 3), dtype=np.float32)
            flow[1, -1, :, 0] = 0.25
            log_var = np.zeros((2, 10, point_count, 1), dtype=np.float32)
            domains = np.asarray(["droid", "droid"])
            reference_source = root / "cuda.npz"
            np.savez_compressed(
                reference_source, scene_relative=flow, log_var=log_var,
                confidence=_release_confidence(log_var, domains), domains=domains,
                provider_timing_names=np.asarray(["dino", "ptv3", "heads"]),
                provider_timings_ms=np.asarray([2.0, 4.0, 1.0], dtype=np.float64),
                stage__dino=np.zeros((2, 4), dtype=np.float32),
            )
            forecast = root / "forecast.npz"
            sealed = seal_reference_forecast(
                model_pack=pack, observation=observation,
                observation_manifest=observation.with_suffix(".json"),
                candidates=candidates, candidates_manifest=candidates.with_suffix(".json"),
                source=reference_source, output=forecast,
            )
            with np.load(observation, allow_pickle=False) as archive:
                scene_points = archive["scene_points"]
            target = root / "target.npz"
            np.savez_compressed(
                target, target_mask=np.ones((point_count,), dtype=np.bool_),
                target_points=scene_points[::64] + np.asarray([0.25, 0, 0], dtype=np.float32),
            )
            planned = rank_forecast(
                observation_manifest=observation.with_suffix(".json"), observation=observation,
                forecast_manifest=forecast.with_suffix(".json"), forecast=forecast,
                target=target, uncertainty_penalty_m=0.05, output=root / "plan.json",
            )
            self.assertEqual(planned["selected_candidate"], 1)
            self.assertTrue(planned["advisory_only"])
            self.assertEqual(planned["forecast_sha256"], sealed["sha256"])

            mask = np.ones(flow.shape[:-1], dtype=np.bool_)
            ground_truth = root / "ground-truth.npz"
            np.savez_compressed(
                ground_truth, scene_relative=flow, scene_exists=mask,
                scene_supervised_mask=mask, scene_moved_mask=mask,
                scene_filter_mask=mask,
            )
            metrics = evaluate_forecast(
                forecast_manifest=forecast.with_suffix(".json"), forecast=forecast,
                ground_truth=ground_truth, output=root / "metrics.json",
            )
            self.assertEqual(metrics["full_eval/test/filtered_l2_moved/mean"], 0.0)
            self.assertTrue(pack_path.is_file())

    def test_raw_actions_and_tampered_observation_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack, _ = self._model_pack(root)
            observation, _ = self._observation(root, pack)
            source = root / "raw-actions.npz"
            np.savez_compressed(source, joint_actions=np.zeros((1, 11, 7), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "raw joint actions"):
                compile_robot_flow_candidates(
                    source=source, observation=observation,
                    observation_manifest=observation.with_suffix(".json"), output=root / "out.npz",
                )
            with observation.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                compile_robot_flow_candidates(
                    source=source, observation=observation,
                    observation_manifest=observation.with_suffix(".json"), output=root / "out.npz",
                )

    def test_model_pack_rejects_nonrelease_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            pack, _ = self._model_pack(Path(directory))
            manifest = dict(pack.manifest)
            manifest["checkpoint"] = dict(manifest["checkpoint"])
            manifest["checkpoint"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "released large DROID"):
                validate_model_pack(manifest)


if __name__ == "__main__":
    unittest.main()
