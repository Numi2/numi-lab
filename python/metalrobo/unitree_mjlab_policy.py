"""Official Unitree MuJoCo-Lab G1 policy import."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .mlx_policy_learning import read_policy_pack
from .native import PolicyDenseLayerArtifact, write_policy_pack

MJLAB_REVISION = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
POLICY_PATH = Path(
    "deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
)
DEPLOY_PATH = Path(
    "deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml"
)
POLICY_SHA256 = (
    "2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28"
)
ACTOR_OBSERVATIONS = 98
ACTIONS = 29


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    return result.stdout.strip()


def _official_source(
    repository: str | Path,
) -> tuple[Path, dict[str, str]]:
    root = Path(repository).expanduser().resolve()
    policy = root / POLICY_PATH
    try:
        revision = _git(root, "rev-parse", "HEAD")
        dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
        _git(root, "ls-files", "--error-unmatch", POLICY_PATH.as_posix())
        _git(root, "ls-files", "--error-unmatch", DEPLOY_PATH.as_posix())
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(
            "could not verify the official Unitree MuJoCo-Lab checkout"
        ) from error
    if revision != MJLAB_REVISION or dirty:
        raise ValueError(
            "Unitree MuJoCo-Lab checkout is not the clean pinned revision"
        )
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    if digest != POLICY_SHA256:
        raise ValueError("official Unitree MuJoCo-Lab policy digest changed")
    deploy = (root / DEPLOY_PATH).read_text(encoding="utf-8")
    if "joint_ids_map:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]" not in "".join(deploy.split()):
        raise ValueError("Unitree MuJoCo-Lab joint order changed")
    return policy, {
        "source_repository": (
            "https://github.com/unitreerobotics/unitree_rl_mjlab"
        ),
        "source_revision": revision,
        "source_policy_path": POLICY_PATH.as_posix(),
        "source_policy_sha256": digest,
        "source_deploy_path": DEPLOY_PATH.as_posix(),
    }


def import_unitree_g1_mjlab_policy(
    official_repository: str | Path,
    output: str | Path,
    *,
    world_fingerprint: int,
    task_fingerprint: int,
    observation_fingerprint: int,
    action_fingerprint: int,
    library_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write the current official MuJoCo-Lab G1 actor as PolicyPack."""

    try:
        import onnx
        from onnx import numpy_helper
        from onnx.reference import ReferenceEvaluator
    except ImportError as error:
        raise RuntimeError(
            "Unitree policy import requires the optional 'onnx' package"
        ) from error

    source, provenance = _official_source(official_repository)
    model = onnx.load(source)
    onnx.checker.check_model(model)
    tensors = {
        tensor.name: np.asarray(
            numpy_helper.to_array(tensor), dtype=np.float32
        )
        for tensor in model.graph.initializer
    }
    mean = tensors["obs_normalizer._mean"].reshape(-1)
    standard_deviation = tensors["onnx::Div_24"].reshape(-1)
    if (
        mean.shape != (ACTOR_OBSERVATIONS,)
        or standard_deviation.shape != (ACTOR_OBSERVATIONS,)
        or np.any(standard_deviation <= 0.0)
    ):
        raise ValueError("official MuJoCo-Lab observation normalizer changed")

    names = tuple(
        (f"mlp.{index}.weight", f"mlp.{index}.bias")
        for index in (0, 2, 4, 6)
    )
    shapes = (
        (512, ACTOR_OBSERVATIONS),
        (256, 512),
        (128, 256),
        (ACTIONS, 128),
    )
    layers = []
    for index, ((weight_name, bias_name), shape) in enumerate(
        zip(names, shapes, strict=True)
    ):
        weight = np.ascontiguousarray(tensors[weight_name])
        bias = np.ascontiguousarray(tensors[bias_name])
        if weight.shape != shape or bias.shape != (shape[0],):
            raise ValueError("official MuJoCo-Lab actor topology changed")
        layers.append(
            PolicyDenseLayerArtifact(
                weights=weight,
                bias=bias,
                activation=3 if index < 3 else 0,
            )
        )

    target = write_policy_pack(
        output,
        policy_id="unitree_g1_mjlab_velocity_1425b15",
        revision=1,
        contract_version=1,
        world_fingerprint=world_fingerprint,
        task_fingerprint=task_fingerprint,
        observation_fingerprint=observation_fingerprint,
        action_fingerprint=action_fingerprint,
        observation_mean=mean,
        observation_inverse_standard_deviation=(
            1.0 / standard_deviation
        ),
        layers=tuple(layers),
        library_path=library_path,
    )
    converted = read_policy_pack(target, library_path=library_path)
    samples = (
        0.35
        * np.sin(
            0.013
            * np.arange(
                3 * ACTOR_OBSERVATIONS, dtype=np.float32
            ).reshape(3, ACTOR_OBSERVATIONS)
        )
    ).astype(np.float32)
    reference = ReferenceEvaluator(model)
    expected = np.concatenate(
        tuple(
            np.asarray(
                reference.run(
                    None,
                    {model.graph.input[0].name: sample[None, :]},
                )[0],
                dtype=np.float32,
            )
            for sample in samples
        ),
        axis=0,
    )
    actual = converted.actions(samples)
    maximum_error = float(np.max(np.abs(expected - actual)))
    if maximum_error > 5.0e-5:
        raise RuntimeError(
            "MuJoCo-Lab PolicyPack parity failed: "
            f"maximum absolute error {maximum_error:.9g}"
        )
    return {
        **provenance,
        "policy_pack": str(target),
        "policy_pack_id": converted.id,
        "policy_pack_revision": converted.revision,
        "policy_pack_content_hash": converted.content_hash,
        "actor_observation_count": converted.actor_observation_count,
        "action_count": converted.action_count,
        "reference_maximum_absolute_error": maximum_error,
    }


__all__ = ["import_unitree_g1_mjlab_policy"]
