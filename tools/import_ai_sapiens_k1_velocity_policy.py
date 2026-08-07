#!/usr/bin/env python3
"""Compile ROBOTIS AI Sapiens K1's official ONNX actors for Numi Lab.

The actor is imported only after the native K1 owner packs have compiled a
matching run.  That binds the PolicyPack to the exact URDF, action ordering,
five-frame observation program, and source asset digest instead of treating
the ONNX file as a free-standing controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

from metalrobo.native import PolicyDenseLayerArtifact, write_policy_pack


SOURCE_REVISION = "c2880e89fb3451a07b6d2600e274224ffcf912e4"
ACTIONS = 23
VELOCITY_BIAS = np.array(
    [
        -0.205, -0.205, 0.0, -0.0106, 0.0106, 0.218, 0.218,
        0.00225, -0.00225, 0.315, -0.315, 0.517, 0.517,
        -0.0695, 0.0695, -0.307, -0.307, 1.08, 1.08,
        0.0108, -0.0108, 0.00186, -0.00186,
    ],
    dtype=np.float32,
)
MIMIC_SCALE = np.array(
    [0.317, 0.317, 0.317, 0.317, 0.317, 0.53, 0.53, 0.317, 0.317,
     0.53, 0.53, 0.317, 0.317, 0.53, 0.53, 0.265, 0.265, 0.53, 0.53,
     0.265, 0.265, 0.53, 0.53], dtype=np.float32,
)
MIMIC_BIAS = np.array(
    [-0.29, -0.307, 0.00124, 0.000442, 0.00489, 0.202, 0.209,
     0.00796, 0.00546, 0.203, -0.199, 0.63, 0.632, -0.00489, 0.00793,
     -0.336, -0.326, 0.594, 0.61, 0.00258, -0.00029, 0.00475,
     -0.00448], dtype=np.float32,
)
POLICIES = {
    "velocity": {
        "relative_path": Path("ai_sapiens_sim2real/assets/k1/locomotion/velocity/walk_default/exported/policy.onnx"),
        "sha256": "be72f4c1ecc00a48be79a4dfbd80c50743b2bb25ebbb1342539235fe29e71c34",
        "observations": 390,
        "scale": np.full(ACTIONS, 0.25, dtype=np.float32),
        "bias": VELOCITY_BIAS,
    },
    "squat": {
        "relative_path": Path("ai_sapiens_sim2real/assets/k1/mimic/squat/exported/policy.onnx"),
        "sha256": "2e3ac6a34b054291deff984d91ae9ed45daa0b7409564903840e2dff9808daf7",
        "observations": 124, "scale": MIMIC_SCALE, "bias": MIMIC_BIAS,
    },
    "dance1": {
        "relative_path": Path("ai_sapiens_sim2real/assets/k1/mimic/dance1/exported/policy.onnx"),
        "sha256": "2e6cac6ba6d5ea5702cee3f47617618b0fb8951783ebcc323b948ccf1c4eb98b",
        "observations": 124, "scale": MIMIC_SCALE, "bias": MIMIC_BIAS,
    },
    "dance2": {
        "relative_path": Path("ai_sapiens_sim2real/assets/k1/mimic/dance2/exported/policy.onnx"),
        "sha256": "9be08b1009ab834e8cdead4fe54df80e8a728f947541b653c6396b2071c54c49",
        "observations": 124, "scale": MIMIC_SCALE, "bias": MIMIC_BIAS,
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--policy", choices=tuple(POLICIES), default="velocity")
    parser.add_argument("--run-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    return parser.parse_args()


def clean_source(source: Path) -> Path:
    source = source.expanduser().resolve()
    try:
        revision = subprocess.run(
            ("git", "-C", str(source), "rev-parse", "HEAD"),
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ("git", "-C", str(source), "status", "--porcelain"),
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("AI Sapiens source must be a readable git checkout") from error
    if revision != SOURCE_REVISION or dirty:
        raise ValueError("AI Sapiens source is not the clean pinned revision")
    return source


def layout(evidence_path: Path, observations: int) -> dict[str, int]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    expected = {
        "actor_observation_count": observations,
        "action_count": ACTIONS,
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise ValueError(f"K1 compiled run has incompatible {key}")
    required = (
        "world_fingerprint",
        "task_fingerprint",
        "observation_fingerprint",
        "action_fingerprint",
    )
    try:
        return {key: int(evidence[key]) for key in required}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("K1 run evidence lacks native policy fingerprints") from error


def main() -> None:
    args = parse_arguments()
    contract = POLICIES[args.policy]
    observations = int(contract["observations"])
    source = clean_source(args.source)
    evidence = layout(args.run_evidence, observations)
    policy = source / contract["relative_path"]
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    if digest != contract["sha256"]:
        raise ValueError(f"official K1 {args.policy} ONNX digest changed")

    model = onnx.load(policy)
    onnx.checker.check_model(model)
    if (
        len(model.graph.input) != 1
        or len(model.graph.output) != 1
        or model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value != observations
        or model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value != ACTIONS
    ):
        raise ValueError(f"official K1 {args.policy} ONNX ABI changed")
    expected_ops = ("Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm")
    if tuple(node.op_type for node in model.graph.node) != expected_ops:
        raise ValueError("official K1 velocity ONNX graph is not the supported ELU actor")
    tensors = {
        initializer.name: np.asarray(numpy_helper.to_array(initializer), dtype=np.float32)
        for initializer in model.graph.initializer
    }
    layers: list[PolicyDenseLayerArtifact] = []
    layer_shapes = ((512, observations), (256, 512), (128, 256), (23, 128))
    for index, shape in zip((0, 2, 4, 6), layer_shapes, strict=True):
        weight = np.ascontiguousarray(tensors[f"actor.{index}.weight"])
        bias = np.ascontiguousarray(tensors[f"actor.{index}.bias"].reshape(-1))
        if weight.shape != shape or bias.shape != (shape[0],):
            raise ValueError(f"official K1 {args.policy} actor topology changed")
        layers.append(
            PolicyDenseLayerArtifact(
                weights=weight,
                bias=bias,
                activation=3 if index != 6 else 0,
            )
        )
    output = write_policy_pack(
        args.output,
        policy_id=f"robotis_ai_sapiens_k1_{args.policy}_c2880e8",
        revision=1,
        contract_version=1,
        layers=layers,
        action_bias=contract["bias"],
        action_scale=contract["scale"],
        library_path=args.library,
        **evidence,
    )
    samples = (0.4 * np.sin(np.arange(3 * observations, dtype=np.float32) * 0.017)).reshape(3, observations)
    expected = np.concatenate(
        [
            np.asarray(
                ReferenceEvaluator(model).run(
                    None, {model.graph.input[0].name: sample[None, :]}
                )[0],
                dtype=np.float32,
            )
            for sample in samples
        ],
        axis=0,
    )
    actual = samples.copy()
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for index, layer in enumerate(layers):
            actual = actual @ layer.weights.T + layer.bias
            if index + 1 != len(layers):
                actual = np.where(actual >= 0.0, actual, np.expm1(actual))
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise RuntimeError(f"K1 {args.policy} ONNX actor parity probe produced non-finite actions")
    maximum_error = float(np.max(np.abs(expected - actual)))
    if maximum_error > 5.0e-5:
        raise RuntimeError(f"K1 {args.policy} PolicyPack parity failed: {maximum_error:.9g}")
    print(json.dumps({
        "policy_pack": str(output),
        "policy": args.policy,
        "source_revision": SOURCE_REVISION,
        "source_policy_sha256": digest,
        "actor_observation_count": observations,
        "action_count": ACTIONS,
        "reference_maximum_absolute_error": maximum_error,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
