"""Command-line entrypoint for pinned tactile imitation training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
import numpy as np

from .lerobot_dataset import (
    ORIGAMI_PINNED_REVISION,
    LeRobotNumericReader,
    LeRobotVideoReader,
    OrigamiWindowSampler,
    RobotDatasetManifest,
    build_origami_manifest,
    download_origami_seasons,
    prepare_origami_metadata,
)
from .mlx_tactile_policy import (
    TactileTubePolicyConfig,
    TactileTubeRuntime,
    TactileTubeTrainer,
)
from .tactile_stream import (
    TactileStreamContract,
    make_origami_stream_contract,
)


def _emit(value: Any) -> None:
    print(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _load_contracts(
    args: argparse.Namespace,
) -> tuple[TactileStreamContract, RobotDatasetManifest]:
    stream = TactileStreamContract.from_json(args.stream_contract)
    manifest = RobotDatasetManifest.from_json(args.manifest)
    manifest.validate(dataset_root=args.dataset_root)
    if manifest.stream_fingerprint != stream.fingerprint:
        raise ValueError(
            "manifest and physical tactile stream do not match"
        )
    return stream, manifest


def _prepare(args: argparse.Namespace) -> int:
    root = prepare_origami_metadata(
        args.dataset_root,
        revision=args.revision,
    )
    contract = make_origami_stream_contract(args.revision)
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    stream_path = contract.to_json(
        output / "tactile-stream.json"
    )
    manifest = build_origami_manifest(
        root,
        stream_contract=contract,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    manifest_path = manifest.to_json(
        output / "dataset-manifest.json"
    )
    _emit(
        {
            "status": "prepared",
            "dataset_root": str(root),
            "stream_contract": str(stream_path),
            "stream_fingerprint": contract.fingerprint,
            "manifest": str(manifest_path),
            "manifest_fingerprint": manifest.fingerprint,
            "seasons": len(manifest.seasons),
            "episodes": manifest.total_episodes,
            "frames": manifest.total_frames,
            "hardware_deployable": (
                contract.action.verified
                and contract.wrench_verified
            ),
        }
    )
    return 0


def _fetch(args: argparse.Namespace) -> int:
    root = download_origami_seasons(
        args.dataset_root,
        revision=args.revision,
        seasons=args.season,
        video_keys=args.video,
    )
    _emit(
        {
            "status": "downloaded",
            "dataset_root": str(root),
            "revision": args.revision,
            "seasons": args.season,
            "videos": args.video,
        }
    )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    stream, manifest = _load_contracts(args)
    _emit(
        {
            "status": "valid",
            "device": str(mx.default_device()),
            "source": {
                "repository": manifest.source_repository,
                "revision": manifest.source_revision,
            },
            "manifest_fingerprint": manifest.fingerprint,
            "stream_fingerprint": stream.fingerprint,
            "canonical_target_fingerprint": (
                stream.canonical_target_fingerprint
            ),
            "action_semantics_verified": stream.action.verified,
            "wrench_contract_verified": stream.wrench_verified,
            "splits": {
                split: {
                    "seasons": len(manifest.seasons_for(split)),
                    "episodes": sum(
                        season.episodes
                        for season in manifest.seasons_for(split)
                    ),
                    "frames": sum(
                        season.frames
                        for season in manifest.seasons_for(split)
                    ),
                }
                for split in ("train", "validation", "test")
            },
        }
    )
    return 0


def _policy_config(
    args: argparse.Namespace,
    manifest: RobotDatasetManifest,
    stream: TactileStreamContract,
) -> TactileTubePolicyConfig:
    return TactileTubePolicyConfig(
        state_dimensions=manifest.state.dimensions,
        action_dimensions=manifest.action.dimensions,
        sensor_count=10,
        wrench_dimensions=6,
        history=args.history,
        horizon=args.horizon,
        model_dimensions=args.model_dimensions,
        transformer_layers=args.layers,
        attention_heads=args.heads,
        mlp_dimensions=args.mlp_dimensions,
        dropout=args.dropout,
        vision_views=len(args.video_key),
        vision_keys=tuple(args.video_key),
        vision_height=args.vision_height,
        vision_width=args.vision_width,
        diffusion_steps=args.diffusion_steps,
        stream_rate_hz=stream.rate_hz,
        stream_attraction=args.stream_attraction,
        stream_noise=args.stream_noise,
        stream_noise_decay=args.stream_noise_decay,
        tactile_mask_probability=args.tactile_mask_probability,
        diffusion_loss_weight=args.diffusion_loss_weight,
        stream_loss_weight=args.stream_loss_weight,
        tactile_reconstruction_weight=(
            args.tactile_reconstruction_weight
        ),
        next_tactile_weight=args.next_tactile_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_gradient_norm=args.max_gradient_norm,
        ema_decay=args.ema_decay,
        seed=args.seed,
    )


def _train(args: argparse.Namespace) -> int:
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("training steps and batch size must be positive")
    stream, manifest = _load_contracts(args)
    config = _policy_config(args, manifest, stream)
    reader = LeRobotNumericReader(args.dataset_root, manifest)
    sampler = OrigamiWindowSampler(
        reader,
        split="train",
        history=config.history,
        horizon=config.horizon,
        seed=config.seed,
    )
    video_reader = (
        LeRobotVideoReader(args.dataset_root, manifest)
        if config.vision_views
        else None
    )
    trainer = TactileTubeTrainer(
        config,
        manifest=manifest,
        stream_contract=stream,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint: Path | None = None
    for _ in range(args.steps):
        batch = sampler.sample(args.batch_size)
        vision = (
            video_reader.batch_histories(
                batch.references,
                keys=config.vision_keys,
                history=config.history,
                width=config.vision_width,
                height=config.vision_height,
            )
            if video_reader is not None
            else None
        )
        stream_vision = (
            video_reader.batch_histories(
                batch.references,
                keys=config.vision_keys,
                history=config.history,
                width=config.vision_width,
                height=config.vision_height,
                streaming=True,
            )
            if video_reader is not None
            else None
        )
        metrics = trainer.train_batch(
            batch,
            vision=vision,
            stream_vision=stream_vision,
        )
        _emit(
            {
                "status": "training",
                "step": trainer.step,
                "device": str(mx.default_device()),
                **metrics,
            }
        )
        if (
            args.checkpoint_interval > 0
            and trainer.step % args.checkpoint_interval == 0
        ):
            checkpoint = trainer.save_checkpoint(
                output / f"checkpoint-{trainer.step:06d}"
            )
    if (
        checkpoint is None
        or checkpoint.name
        != f"checkpoint-{trainer.step:06d}"
    ):
        checkpoint = trainer.save_checkpoint(
            output / f"checkpoint-{trainer.step:06d}"
        )
    _emit(
        {
            "status": "complete",
            "step": trainer.step,
            "examples": trainer.examples,
            "checkpoint": str(checkpoint),
            "hardware_deployable": (
                stream.action.verified
                and stream.wrench_verified
            ),
        }
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    if args.batches <= 0 or args.batch_size <= 0:
        raise ValueError(
            "evaluation batches and batch size must be positive"
        )
    stream, manifest = _load_contracts(args)
    runtime = TactileTubeRuntime.from_checkpoint(
        args.checkpoint,
        manifest=manifest,
        stream_contract=stream,
    )
    reader = LeRobotNumericReader(args.dataset_root, manifest)
    sampler = OrigamiWindowSampler(
        reader,
        split=args.split,
        history=runtime.config.history,
        horizon=runtime.config.horizon,
        seed=args.seed,
    )
    video_reader = (
        LeRobotVideoReader(args.dataset_root, manifest)
        if runtime.config.vision_views
        else None
    )
    absolute_error = 0.0
    squared_error = 0.0
    sample_count = 0
    tactile_sensitivity = 0.0
    sensitivity_count = 0
    stream_absolute_error = 0.0
    stream_squared_error = 0.0
    stream_sample_count = 0
    stream_tactile_sensitivity = 0.0
    next_wrench_squared_error = 0.0
    next_wrench_sample_count = 0
    correction_rms = 0.0
    for index in range(args.batches):
        batch = sampler.sample(args.batch_size)
        vision = (
            video_reader.batch_histories(
                batch.references,
                keys=runtime.config.vision_keys,
                history=runtime.config.history,
                width=runtime.config.vision_width,
                height=runtime.config.vision_height,
            )
            if video_reader is not None
            else None
        )
        stream_vision = (
            video_reader.batch_histories(
                batch.references,
                keys=runtime.config.vision_keys,
                history=runtime.config.history,
                width=runtime.config.vision_width,
                height=runtime.config.vision_height,
                streaming=True,
            )
            if video_reader is not None
            else None
        )
        prediction = runtime.initial_tube(
            batch.state_history,
            batch.wrench_history,
            vision=vision,
            seed=args.seed + index,
        )
        mask = batch.action_mask[..., None]
        difference = prediction - batch.action_chunk
        absolute_error += float(
            np.abs(difference * mask).sum()
        )
        squared_error += float(
            np.square(difference * mask).sum()
        )
        sample_count += int(mask.sum()) * difference.shape[-1]
        ablated = runtime.initial_tube(
            batch.state_history,
            np.zeros_like(batch.wrench_history),
            vision=vision,
            seed=args.seed + index,
        )
        tactile_sensitivity += float(
            np.abs(prediction - ablated).mean()
        )
        sensitivity_count += 1
        decision = runtime.stream_decision(
            prediction,
            batch.stream_state_history,
            batch.stream_wrench_history,
            elapsed_seconds=batch.stream_time,
            vision=stream_vision,
        )
        stream_mask = batch.stream_action_mask[..., None]
        stream_difference = (
            decision.action_tube - batch.stream_action_chunk
        )
        stream_absolute_error += float(
            np.abs(stream_difference * stream_mask).sum()
        )
        stream_squared_error += float(
            np.square(stream_difference * stream_mask).sum()
        )
        stream_sample_count += (
            int(stream_mask.sum()) * stream_difference.shape[-1]
        )
        stream_ablated = runtime.stream_decision(
            prediction,
            batch.stream_state_history,
            np.zeros_like(batch.stream_wrench_history),
            elapsed_seconds=batch.stream_time,
            vision=stream_vision,
        )
        stream_tactile_sensitivity += float(
            np.abs(
                decision.action_tube
                - stream_ablated.action_tube
            ).mean()
        )
        target_next_wrench = batch.next_wrench.reshape(
            (
                args.batch_size,
                runtime.config.sensor_count,
                runtime.config.wrench_dimensions,
            )
        )
        next_wrench_squared_error += float(
            np.square(
                decision.predicted_next_wrench
                - target_next_wrench
            ).sum()
        )
        next_wrench_sample_count += int(target_next_wrench.size)
        correction_rms += float(decision.correction_rms.mean())
    _emit(
        {
            "status": "evaluated",
            "split": args.split,
            "device": str(mx.default_device()),
            "mean_absolute_error": (
                absolute_error / max(sample_count, 1)
            ),
            "root_mean_squared_error": math.sqrt(
                squared_error / max(sample_count, 1)
            ),
            "mean_tactile_ablation_delta": (
                tactile_sensitivity
                / max(sensitivity_count, 1)
            ),
            "stream_mean_absolute_error": (
                stream_absolute_error
                / max(stream_sample_count, 1)
            ),
            "stream_root_mean_squared_error": math.sqrt(
                stream_squared_error
                / max(stream_sample_count, 1)
            ),
            "stream_mean_tactile_ablation_delta": (
                stream_tactile_sensitivity
                / max(sensitivity_count, 1)
            ),
            "next_wrench_root_mean_squared_error": math.sqrt(
                next_wrench_squared_error
                / max(next_wrench_sample_count, 1)
            ),
            "mean_stream_correction_rms": (
                correction_rms / max(sensitivity_count, 1)
            ),
            "hardware_deployable": (
                stream.action.verified
                and stream.wrench_verified
            ),
        }
    )
    return 0


def _shared_contract_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--stream-contract", required=True)
    parser.add_argument("--manifest", required=True)


def _model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--model-dimensions", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-dimensions", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--video-key",
        action="append",
        default=[],
        help=(
            "aligned LeRobot video key; repeat for multiple views "
            "(omitting all keys trains numeric tactile only)"
        ),
    )
    parser.add_argument("--vision-height", type=int, default=128)
    parser.add_argument("--vision-width", type=int, default=128)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--stream-attraction", type=float, default=8.0)
    parser.add_argument("--stream-noise", type=float, default=0.2)
    parser.add_argument("--stream-noise-decay", type=float, default=4.0)
    parser.add_argument(
        "--tactile-mask-probability",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--diffusion-loss-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--stream-loss-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--tactile-reconstruction-weight",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--next-tactile-weight",
        type=float,
        default=0.2,
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metalrobo-tactile",
        description=(
            "Apple-native tactile imitation learning for pinned "
            "LeRobot datasets"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset-root", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument(
        "--revision",
        default=ORIGAMI_PINNED_REVISION,
    )
    prepare.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
    )
    prepare.add_argument("--test-fraction", type=float, default=0.1)
    prepare.set_defaults(handler=_prepare)

    fetch = commands.add_parser("fetch")
    fetch.add_argument("--dataset-root", required=True)
    fetch.add_argument("--season", action="append", required=True)
    fetch.add_argument("--video", action="append", default=[])
    fetch.add_argument(
        "--revision",
        default=ORIGAMI_PINNED_REVISION,
    )
    fetch.set_defaults(handler=_fetch)

    inspect = commands.add_parser("inspect")
    _shared_contract_arguments(inspect)
    inspect.set_defaults(handler=_inspect)

    train = commands.add_parser("train")
    _shared_contract_arguments(train)
    _model_arguments(train)
    train.add_argument("--steps", type=int, required=True)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--checkpoint-interval", type=int, default=1000)
    train.add_argument("--output", required=True)
    train.add_argument("--resume")
    train.set_defaults(handler=_train)

    evaluate = commands.add_parser("evaluate")
    _shared_contract_arguments(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )
    evaluate.add_argument("--batches", type=int, default=8)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.set_defaults(handler=_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
