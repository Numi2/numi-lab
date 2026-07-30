#!/usr/bin/env python3
"""Focused native-GPU check for tactile imitation ingestion and training."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from metalrobo.lerobot_dataset import (
    LeRobotNumericReader,
    LeRobotVideoReader,
    OrigamiWindowSampler,
    RobotDatasetManifest,
    build_origami_manifest,
)
from metalrobo.mlx_tactile_policy import (
    TactileTubePolicyConfig,
    TactileTubeRuntime,
    TactileTubeTrainer,
)
from metalrobo.tactile_stream import (
    TactileStreamContract,
    decode_sharpa_tacmap_uint8,
    make_origami_stream_contract,
)


REVISION = "8194af6b9341dac7686c2f29704ff893e6f2f95e"
VIDEO_KEY = "observation.images.head_left"


def _fixed_list(values: np.ndarray):
    import pyarrow as pa

    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float32()),
        values.shape[1],
    )


def _statistics(value: np.ndarray) -> dict[str, object]:
    return {
        "min": value.min(axis=0).tolist(),
        "max": value.max(axis=0).tolist(),
        "mean": value.mean(axis=0).tolist(),
        "std": value.std(axis=0).tolist(),
        "count": [len(value)],
    }


def _write_season(
    root: Path,
    *,
    index: int,
    frames: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    season_root = (
        root / f"season_fixture_{index:02d}" / "lerobot3.0"
    )
    metadata = season_root / "meta"
    data = season_root / "data" / "chunk-000"
    video_path = (
        season_root
        / "videos"
        / VIDEO_KEY
        / "chunk-000"
        / "file-000.mp4"
    )
    metadata.mkdir(parents=True)
    data.mkdir(parents=True)
    video_path.parent.mkdir(parents=True)
    rng = np.random.default_rng(100 + index)
    time = np.arange(frames, dtype=np.float32)[:, None]
    state = (
        0.04 * rng.normal(size=(frames, 65))
        + 0.02 * time
        + index
    ).astype(np.float32)
    wrench = (
        0.03 * rng.normal(size=(frames, 60))
        + np.sin(time * 0.2)
        + 0.25 * index
    ).astype(np.float32)
    action = (
        0.7 * state
        + 0.1 * np.pad(wrench, ((0, 0), (0, 5)))
    ).astype(np.float32)
    table = pa.table(
        {
            "observation.state": _fixed_list(state),
            "action": _fixed_list(action),
            "observation.tactile": _fixed_list(wrench),
            "episode_index": pa.array(
                np.zeros(frames, dtype=np.int64)
            ),
            "frame_index": pa.array(
                np.arange(frames, dtype=np.int64)
            ),
            "timestamp": pa.array(
                np.arange(frames, dtype=np.float32) / 30.0
            ),
        }
    )
    pq.write_table(table, data / "file-000.parquet")
    import av

    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 30)
        for frame_index in range(frames):
            image = np.empty((32, 32, 3), dtype=np.uint8)
            image[..., 0] = 20 * index
            image[..., 1] = 10 * frame_index
            image[..., 2] = np.arange(32, dtype=np.uint8)[None, :]
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = frame_index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    episode_metadata = pa.table(
        {
            "episode_index": pa.array([0], type=pa.int64()),
            f"videos/{VIDEO_KEY}/chunk_index": pa.array(
                [0],
                type=pa.int64(),
            ),
            f"videos/{VIDEO_KEY}/file_index": pa.array(
                [0],
                type=pa.int64(),
            ),
            f"videos/{VIDEO_KEY}/from_timestamp": pa.array(
                [0.0],
                type=pa.float64(),
            ),
            f"videos/{VIDEO_KEY}/to_timestamp": pa.array(
                [frames / 30.0],
                type=pa.float64(),
            ),
        }
    )
    episode_path = metadata / "episodes" / "chunk-000"
    episode_path.mkdir(parents=True)
    pq.write_table(
        episode_metadata,
        episode_path / "file-000.parquet",
    )
    info = {
        "codebase_version": "v3.0",
        "robot_type": "SharpaWave-fixture",
        "total_episodes": 1,
        "total_frames": frames,
        "total_tasks": 1,
        "fps": 30,
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/"
            "file-{file_index:03d}.mp4"
        ),
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [65],
            },
            "action": {
                "dtype": "float32",
                "shape": [65],
            },
            "observation.tactile": {
                "dtype": "float32",
                "shape": [60],
            },
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [3, 32, 32],
            },
        },
    }
    stats = {
        "observation.state": _statistics(state),
        "action": _statistics(action),
        "observation.tactile": _statistics(wrench),
    }
    (metadata / "info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (metadata / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return state, action, wrench


def _first_parameter(trainer: TactileTubeTrainer) -> np.ndarray:
    parameters = dict(tree_flatten(trainer.model.trainable_parameters()))
    first = parameters[sorted(parameters)[0]]
    return np.array(first, dtype=np.float32, copy=True)


def main() -> int:
    if "gpu" not in str(mx.default_device()).lower():
        raise RuntimeError(
            "tactile imitation check requires the MLX Metal device"
        )
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "install the tactile-dataset extra to run this probe"
        ) from error

    with tempfile.TemporaryDirectory(
        prefix="metalrobo-tactile-imitation-"
    ) as temporary:
        root = Path(temporary)
        source_arrays = [
            _write_season(root, index=index)
            for index in range(3)
        ]
        contract = make_origami_stream_contract(REVISION)
        contract_path = root / "tactile-stream.json"
        contract.to_json(contract_path)
        if (
            TactileStreamContract.from_json(contract_path).fingerprint
            != contract.fingerprint
        ):
            raise RuntimeError(
                "physical tactile stream did not round trip"
            )

        manifest = build_origami_manifest(
            root,
            stream_contract=contract,
            validation_fraction=0.2,
            test_fraction=0.2,
        )
        splits = {
            season.split for season in manifest.seasons
        }
        if splits != {"train", "validation", "test"}:
            raise RuntimeError(
                "fixture was not isolated by collection season"
            )
        training_season = manifest.seasons_for("train")[0]
        source_index = int(training_season.name.rsplit("_", 1)[-1])
        expected_training_mean = source_arrays[source_index][0].mean(
            axis=0
        )
        if not np.allclose(
            manifest.state_normalization.mean,
            expected_training_mean,
            atol=1.0e-6,
        ):
            raise RuntimeError(
                "normalization leaked held-out collection seasons"
            )
        manifest_path = root / "dataset-manifest.json"
        manifest.to_json(manifest_path)
        loaded_manifest = RobotDatasetManifest.from_json(
            manifest_path
        )
        loaded_manifest.validate(dataset_root=root)

        reader = LeRobotNumericReader(root, loaded_manifest)
        sampler = OrigamiWindowSampler(
            reader,
            split="train",
            history=2,
            horizon=4,
            seed=7,
        )
        batch = sampler.sample(2)
        if (
            batch.state_history.shape != (2, 2, 65)
            or batch.wrench_history.shape != (2, 2, 60)
            or batch.action_chunk.shape != (2, 4, 65)
            or not batch.action_mask[:, 0].all()
            or np.any(
                (~batch.action_mask[:, :-1])
                & batch.action_mask[:, 1:]
            )
        ):
            raise RuntimeError("sampled imitation batch is malformed")
        video_reader = LeRobotVideoReader(root, loaded_manifest)
        vision = video_reader.batch_histories(
            batch.references,
            keys=(VIDEO_KEY,),
            history=2,
            width=16,
            height=16,
        )
        stream_vision = video_reader.batch_histories(
            batch.references,
            keys=(VIDEO_KEY,),
            history=2,
            width=16,
            height=16,
            streaming=True,
        )
        if (
            vision.shape != (2, 2, 1, 16, 16, 3)
            or stream_vision.shape != vision.shape
        ):
            raise RuntimeError("aligned video batch is malformed")

        config = TactileTubePolicyConfig(
            history=2,
            horizon=4,
            model_dimensions=32,
            transformer_layers=1,
            attention_heads=4,
            mlp_dimensions=64,
            vision_views=1,
            vision_keys=(VIDEO_KEY,),
            vision_height=16,
            vision_width=16,
            diffusion_steps=2,
            seed=11,
        )
        trainer = TactileTubeTrainer(
            config,
            manifest=loaded_manifest,
            stream_contract=contract,
        )
        before = _first_parameter(trainer)
        first_metrics = trainer.train_batch(
            batch,
            vision=vision,
            stream_vision=stream_vision,
        )
        second_metrics = trainer.train_batch(
            batch,
            vision=vision,
            stream_vision=stream_vision,
        )
        after = _first_parameter(trainer)
        metric_values = np.asarray(
            [
                value
                for key, value in second_metrics.items()
                if key not in {"step_seconds", "examples"}
            ]
        )
        if (
            not np.isfinite(metric_values).all()
            or np.array_equal(before, after)
            or trainer.step != 2
        ):
            raise RuntimeError(
                "compiled tactile optimization did not update weights"
            )

        checkpoint = trainer.save_checkpoint(root / "checkpoint")
        runtime = TactileTubeRuntime.from_checkpoint(
            checkpoint,
            manifest=loaded_manifest,
            stream_contract=contract,
        )
        tube = runtime.initial_tube(
            batch.state_history,
            batch.wrench_history,
            vision=vision,
            seed=19,
        )
        tactile_ablated = runtime.initial_tube(
            batch.state_history,
            np.zeros_like(batch.wrench_history),
            vision=vision,
            seed=19,
        )
        decision = runtime.stream_decision(
            tube,
            batch.stream_state_history,
            batch.stream_wrench_history,
            elapsed_seconds=batch.stream_time,
            vision=stream_vision,
        )
        streamed = decision.action_tube
        provenance = json.loads(
            (checkpoint / "provenance.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            tube.shape != (2, 4, 65)
            or streamed.shape != tube.shape
            or decision.reconstructed_wrench.shape != (2, 10, 6)
            or decision.predicted_next_wrench.shape != (2, 10, 6)
            or decision.correction_rms.shape != (2,)
            or not np.isfinite(tube).all()
            or not np.isfinite(streamed).all()
            or not np.isfinite(
                decision.predicted_next_wrench
            ).all()
            or provenance.get("hardware_deployable") is not False
            or set(provenance.get("promotion_blockers", ()))
            != {
                "action_semantics_unverified",
                "wrench_units_or_frame_unverified",
            }
            or set(provenance.get("artifact_sha256", ()))
            != {
                "model.safetensors",
                "training_model.safetensors",
                "optimizer.safetensors",
                "config.json",
            }
            or float(np.max(np.abs(tube - tactile_ablated)))
            <= 1.0e-7
        ):
            raise RuntimeError(
                "tactile policy inference is not finite or touch-aware"
            )
        try:
            runtime.assert_hardware_ready()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "unverified dataset was incorrectly hardware-promoted"
            )

        restored = TactileTubeTrainer(
            config,
            manifest=loaded_manifest,
            stream_contract=contract,
        )
        restored.load_checkpoint(checkpoint)
        if restored.step != trainer.step:
            raise RuntimeError("checkpoint training state was not restored")

        incompatible = replace(
            contract,
            action=replace(
                contract.action,
                semantics="different_unverified_command",
            ),
        )
        incompatible.validate()
        try:
            TactileTubeRuntime.from_checkpoint(
                checkpoint,
                manifest=loaded_manifest,
                stream_contract=incompatible,
            )
        except ValueError:
            pass
        else:
            raise RuntimeError(
                "checkpoint accepted mismatched physical provenance"
            )

        decoded, valid = decode_sharpa_tacmap_uint8(
            np.asarray([0, 100, 101, 255], dtype=np.uint8)
        )
        expected = np.asarray(
            [0.0, 5.0e-4, 5.3e-4, 5.15e-3],
            dtype=np.float32,
        )
        if (
            not np.allclose(decoded, expected, atol=1.0e-8)
            or valid.tolist() != [True, True, True, False]
        ):
            raise RuntimeError("Sharpa Tacmap decoding changed")

        print(
            json.dumps(
                {
                    "status": "ok",
                    "device": str(mx.default_device()),
                    "manifest_fingerprint": manifest.fingerprint,
                    "stream_fingerprint": contract.fingerprint,
                    "first_step_seconds": first_metrics[
                        "step_seconds"
                    ],
                    "cached_step_seconds": second_metrics[
                        "step_seconds"
                    ],
                    "loss": second_metrics["loss"],
                    "tactile_ablation_max_delta": float(
                        np.max(np.abs(tube - tactile_ablated))
                    ),
                    "hardware_deployable": False,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
