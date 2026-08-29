#!/usr/bin/env python3
"""Create a fingerprint-preserving Crow actor weight interpolation candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from metalrobo.mlx_policy_learning import (
    MLXPolicyLearner,
    read_policy_pack,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_matching_contract(source: object, target: object) -> None:
    scalar_fields = (
        "format_version",
        "contract_version",
        "world_fingerprint",
        "task_fingerprint",
        "observation_fingerprint",
        "action_fingerprint",
        "actor_observation_count",
        "critic_observation_count",
        "action_count",
        "observation_clip",
        "action_clip",
    )
    for field in scalar_fields:
        if getattr(source, field) != getattr(target, field):
            raise ValueError(f"policy interpolation disagrees on {field}")
    array_fields = (
        "effective_observation_mean",
        "effective_observation_inverse_standard_deviation",
        "effective_action_bias",
        "effective_action_scale",
    )
    for field in array_fields:
        if not np.array_equal(getattr(source, field), getattr(target, field)):
            raise ValueError(f"policy interpolation disagrees on {field}")
    source_shape = [
        (layer.input_count, layer.output_count, layer.activation)
        for layer in source.layers
    ]
    target_shape = [
        (layer.input_count, layer.output_count, layer.activation)
        for layer in target.layers
    ]
    if source_shape != target_shape:
        raise ValueError("policy interpolation actor architectures disagree")


def _interpolate_changed(
    source: np.ndarray,
    target: np.ndarray,
    alpha: np.float32,
) -> np.ndarray:
    """Blend changed elements without rounding source-equal parameters."""
    result = source.copy()
    changed = source != target
    result[changed] = (
        (np.float32(1.0) - alpha) * source[changed]
        + alpha * target[changed]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument(
        "--allow-extrapolation",
        action="store_true",
        help=(
            "permit a bounded alpha above one for isolated residual dose "
            "screens; the hard maximum is eight"
        ),
    )
    parser.add_argument("--revision", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--deployment-output", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--native-library", required=True, type=Path)
    parser.add_argument("--learner-state-output", type=Path)
    parser.add_argument(
        "--output-actions",
        help="comma-separated output rows; when set, preserve every hidden layer",
    )
    arguments = parser.parse_args()

    maximum_alpha = 8.0 if arguments.allow_extrapolation else 1.0
    if (
        not np.isfinite(arguments.alpha)
        or not 0.0 <= arguments.alpha <= maximum_alpha
    ):
        raise ValueError(
            f"policy interpolation alpha must be in [0, {maximum_alpha:g}]"
        )
    if arguments.revision <= 0:
        raise ValueError("policy interpolation revision must be positive")

    source_path = arguments.source.expanduser().resolve()
    target_path = arguments.target.expanduser().resolve()
    native_library = arguments.native_library.expanduser().resolve()
    source_pack = read_policy_pack(source_path, library_path=native_library)
    target_pack = read_policy_pack(target_path, library_path=native_library)
    _require_matching_contract(source_pack, target_pack)
    output_actions: tuple[int, ...] | None = None
    if arguments.output_actions is not None:
        output_actions = tuple(
            int(value) for value in arguments.output_actions.split(",")
        )
        if (
            not output_actions
            or len(set(output_actions)) != len(output_actions)
            or min(output_actions) < 0
            or max(output_actions) >= source_pack.action_count
        ):
            raise ValueError("policy interpolation output actions are invalid")

    learner = MLXPolicyLearner.from_policy_pack(
        source_path,
        library_path=native_library,
    )
    actor_layers = [
        layer for layer in learner.model.actor.layers
        if isinstance(layer, nn.Linear)
    ]
    if len(actor_layers) != len(target_pack.layers):
        raise RuntimeError("restored actor disagrees with target PolicyPack")
    alpha = np.float32(arguments.alpha)
    for layer_index, (destination, source_layer, target_layer) in enumerate(zip(
        actor_layers,
        source_pack.layers,
        target_pack.layers,
        strict=True,
    )):
        source_weight = np.asarray(source_layer.weights, dtype=np.float32)
        source_bias = np.asarray(source_layer.bias, dtype=np.float32)
        if output_actions is not None:
            if layer_index + 1 != len(actor_layers):
                continue
            weight = source_weight.copy()
            bias = source_bias.copy()
            rows = np.asarray(output_actions, dtype=np.int64)
            weight[rows] = _interpolate_changed(
                source_weight[rows],
                np.asarray(target_layer.weights, dtype=np.float32)[rows],
                alpha,
            )
            bias[rows] = _interpolate_changed(
                source_bias[rows],
                np.asarray(target_layer.bias, dtype=np.float32)[rows],
                alpha,
            )
        else:
            weight = _interpolate_changed(
                source_weight,
                np.asarray(target_layer.weights, dtype=np.float32),
                alpha,
            )
            bias = _interpolate_changed(
                source_bias,
                np.asarray(target_layer.bias, dtype=np.float32),
                alpha,
            )
        destination.weight = mx.array(weight, dtype=mx.float32)
        destination.bias = mx.array(bias, dtype=mx.float32)
    learner.revision = arguments.revision
    mx.eval(learner.model.parameters())

    output = arguments.output.expanduser().resolve()
    deployment_output = arguments.deployment_output.expanduser().resolve()
    record = arguments.record.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    deployment_output.parent.mkdir(parents=True, exist_ok=True)
    record.parent.mkdir(parents=True, exist_ok=True)
    learner.write_policy_pack(output, library_path=native_library)
    learner.write_policy_pack(
        deployment_output,
        stochastic=False,
        library_path=native_library,
    )
    learner_state_output: Path | None = None
    if arguments.learner_state_output is not None:
        from metalrobo.mlx_policy_worker import _write_learner_state

        learner_state_output = (
            arguments.learner_state_output.expanduser().resolve()
        )
        _write_learner_state(learner, learner_state_output)
    record.write_text(
        json.dumps(
            {
                "schema": "numi.crow-policy-interpolation.v1",
                "classification": "unqualified actor weight interpolation candidate",
                "source": str(source_path),
                "source_sha256": _sha256(source_path),
                "source_revision": source_pack.revision,
                "target": str(target_path),
                "target_sha256": _sha256(target_path),
                "target_revision": target_pack.revision,
                "alpha": arguments.alpha,
                "extrapolated": arguments.alpha > 1.0,
                "mode": (
                    "selected-output-actions"
                    if output_actions is not None
                    else "all-actor-parameters"
                ),
                "output_actions": output_actions,
                "revision": arguments.revision,
                "output": str(output),
                "output_sha256": _sha256(output),
                "deployment_output": str(deployment_output),
                "deployment_output_sha256": _sha256(deployment_output),
                "learner_state_output": (
                    str(learner_state_output)
                    if learner_state_output is not None
                    else None
                ),
                "learner_state_output_sha256": (
                    _sha256(learner_state_output)
                    if learner_state_output is not None
                    else None
                ),
                "world_fingerprint": source_pack.world_fingerprint,
                "task_fingerprint": source_pack.task_fingerprint,
                "observation_fingerprint": source_pack.observation_fingerprint,
                "action_fingerprint": source_pack.action_fingerprint,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
