#!/usr/bin/env python3
"""Compile the pinned G1 legs ONNX actor into a native PolicyPack.

The imported actor remains a 240 -> 12 ELU MLP.  This bridge only widens its
last layer to Numi Lab's 29 G1 action lanes: the first twelve legs pass
through unmodified and the seventeen waist/arm residual lanes are held at
zero by the source deployment pose authored in the legs-only task.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from metalrobo.native import PolicyDenseLayerArtifact, write_policy_pack
from metalrobo.g1_legs_policy import (
    G1_LEGS_ACTION_SCALE,
    G1_LEGS_ACTIONS,
    G1_LEGS_OBSERVATION_SIZE,
    inspect_onnx,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    for name in (
        "world-fingerprint",
        "task-fingerprint",
        "observation-fingerprint",
        "action-fingerprint",
    ):
        parser.add_argument(f"--{name}", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    artifact = inspect_onnx(args.onnx)
    graph = onnx.load(artifact.path).graph
    tensors = {
        initializer.name: numpy_helper.to_array(initializer).astype(np.float32)
        for initializer in graph.initializer
    }

    # The published graph owns the same normalization as the real deployment:
    # clip((observation - mean) / std, -5, +5).
    mean = tensors[graph.node[0].input[1]].reshape(-1)
    standard_deviation = tensors[graph.node[1].input[1]].reshape(-1)
    if mean.shape != (G1_LEGS_OBSERVATION_SIZE,) or np.any(
        ~np.isfinite(standard_deviation)
    ) or np.any(standard_deviation <= 0.0):
        raise ValueError("G1 legs observation normalization is malformed")

    layers: list[PolicyDenseLayerArtifact] = []
    for index in range(4):
        weights = tensors[f"fcs.{index}.weight"]
        bias = tensors[f"fcs.{index}.bias"].reshape(-1)
        if index == 3:
            widened_weights = np.zeros((29, weights.shape[1]), dtype=np.float32)
            widened_bias = np.zeros(29, dtype=np.float32)
            widened_weights[:G1_LEGS_ACTIONS] = weights
            widened_bias[:G1_LEGS_ACTIONS] = bias
            weights, bias = widened_weights, widened_bias
        layers.append(
            PolicyDenseLayerArtifact(
                weights=weights,
                bias=bias,
                activation=3 if index < 3 else 0,
            )
        )

    # Native task action bindings apply 0.5 rad to leg residuals, exactly as
    # The deployed target is default_joint_angles + 0.5 * action. The
    # upper lanes retain their authored hold targets and receive no residual.
    action_scale = np.ones(29, dtype=np.float32)
    action_scale[:G1_LEGS_ACTIONS] = 1.0
    action_scale[G1_LEGS_ACTIONS:] = 0.0
    output = write_policy_pack(
        args.output,
        policy_id="g1_legs_locomotion_v26",
        revision=42_290,
        contract_version=1,
        world_fingerprint=args.world_fingerprint,
        task_fingerprint=args.task_fingerprint,
        observation_fingerprint=args.observation_fingerprint,
        action_fingerprint=args.action_fingerprint,
        layers=layers,
        observation_mean=mean,
        observation_inverse_standard_deviation=1.0 / standard_deviation,
        action_scale=action_scale,
        observation_clip=5.0,
        library_path=args.library,
    )
    print(
        f"wrote {output} from {artifact.path.name} sha256={artifact.sha256} "
        f"leg_action_scale={G1_LEGS_ACTION_SCALE}"
    )


if __name__ == "__main__":
    main()
