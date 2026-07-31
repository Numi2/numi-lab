"""Canonical PolicyPack export and independent Unitree MuJoCo playback."""

from __future__ import annotations

import json
import hashlib
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .mlx_policy_learning import (
    NativePolicyPack,
    read_policy_pack,
)
from .native import unitree_g1_deployment_contract
from .native import PolicyDenseLayerArtifact, write_policy_pack

G1_ACTOR_FRAME_SIZE = 96
G1_ACTOR_HISTORY = 5
G1_ACTOR_OBSERVATION_SIZE = (
    G1_ACTOR_FRAME_SIZE * G1_ACTOR_HISTORY
)
G1_ACTION_COUNT = (G1_ACTOR_FRAME_SIZE - 9) // 3

UNITREE_RL_LAB_REVISION = (
    "4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
)
UNITREE_G1_VELOCITY_POLICY_PATH = Path(
    "deploy/robots/g1_29dof/config/policy/velocity/v0/"
    "exported/policy.onnx"
)
UNITREE_G1_VELOCITY_DEPLOY_PATH = Path(
    "deploy/robots/g1_29dof/config/policy/velocity/v0/"
    "params/deploy.yaml"
)
UNITREE_G1_VELOCITY_POLICY_SHA256 = (
    "610c27e463a8f666aa50a06346678c00b4df3859f10b54bcc1f817c28251406f"
)

# Unitree's velocity actor is trained in this interleaved order. Each value is
# the corresponding Unitree SDK / MetalRobo action index. Keeping the pinned
# table here makes the one-time conversion self-contained and reviewable; the
# source deployment YAML is still verified before conversion.
UNITREE_G1_POLICY_TO_SDK_JOINT = np.asarray(
    (
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
        16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    ),
    dtype=np.int64,
)

G1_PROMOTION_COMMANDS = (
    ("idle", (0.0, 0.0, 0.0)),
    ("forward_slow", (0.1, 0.0, 0.0)),
    ("forward", (0.5, 0.0, 0.0)),
    ("reverse", (-0.5, 0.0, 0.0)),
    ("left", (0.0, 0.3, 0.0)),
    ("right", (0.0, -0.3, 0.0)),
    ("yaw_left", (0.0, 0.0, 0.2)),
    ("yaw_right", (0.0, 0.0, -0.2)),
)


@dataclass(frozen=True, slots=True)
class G1DeployableSensors:
    """Only signals available to the deployed proprioceptive actor."""

    base_angular_velocity: np.ndarray
    projected_gravity: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray


def _g1_policy(
    path: str | Path,
    library_path: str | Path | None = None,
) -> NativePolicyPack:
    pack = read_policy_pack(
        path,
        library_path=library_path,
    )
    if (
        pack.actor_observation_count
        != G1_ACTOR_OBSERVATION_SIZE
        or pack.action_count != G1_ACTION_COUNT
    ):
        raise ValueError(
            "PolicyPack dimensions do not match the bundled G1 TaskPack"
        )
    return pack


def _g1_contract(
    library_path: str | Path | None,
) -> dict[str, Any]:
    contract = unitree_g1_deployment_contract(library_path)
    if (
        len(contract["joint_order"]) != G1_ACTION_COUNT
        or contract["actor_frame_size"] != G1_ACTOR_FRAME_SIZE
        or contract["actor_history_length"] != G1_ACTOR_HISTORY
    ):
        raise RuntimeError(
            "Native G1 deployment contract disagrees with the actor ABI"
        )
    return contract


def _run_git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    return completed.stdout.strip()


def _verify_unitree_velocity_policy_source(
    repository: str | Path,
) -> tuple[Path, dict[str, str]]:
    root = Path(repository).expanduser().resolve()
    policy = root / UNITREE_G1_VELOCITY_POLICY_PATH
    deploy = root / UNITREE_G1_VELOCITY_DEPLOY_PATH
    try:
        revision = _run_git(root, "rev-parse", "HEAD")
        dirty = _run_git(
            root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
        _run_git(
            root,
            "ls-files",
            "--error-unmatch",
            UNITREE_G1_VELOCITY_POLICY_PATH.as_posix(),
        )
        _run_git(
            root,
            "ls-files",
            "--error-unmatch",
            UNITREE_G1_VELOCITY_DEPLOY_PATH.as_posix(),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(
            "could not verify the official Unitree RL Lab checkout"
        ) from error
    if revision != UNITREE_RL_LAB_REVISION or dirty:
        raise ValueError(
            "Unitree RL Lab checkout is not the clean pinned revision"
        )
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    if digest != UNITREE_G1_VELOCITY_POLICY_SHA256:
        raise ValueError("official Unitree G1 policy digest changed")
    deploy_text = deploy.read_text(encoding="utf-8")
    compact_deploy = "".join(deploy_text.split())
    expected_mapping = "joint_ids_map:[" + ",".join(
        str(int(value)) for value in UNITREE_G1_POLICY_TO_SDK_JOINT
    ) + "]"
    if expected_mapping not in compact_deploy:
        raise ValueError(
            "official Unitree G1 deployment joint mapping changed"
        )
    return policy, {
        "source_repository": (
            "https://github.com/unitreerobotics/unitree_rl_lab"
        ),
        "source_revision": revision,
        "source_policy_path": (
            UNITREE_G1_VELOCITY_POLICY_PATH.as_posix()
        ),
        "source_policy_sha256": digest,
        "source_deploy_path": (
            UNITREE_G1_VELOCITY_DEPLOY_PATH.as_posix()
        ),
    }


def _unitree_actor_input_indices() -> np.ndarray:
    """Map one Unitree term-major actor observation onto our frame-major ABI."""

    indices: list[int] = []
    for frame_offset, width, joint_ordered in (
        (0, 3, False),
        (3, 3, False),
        (6, 3, False),
        (9, G1_ACTION_COUNT, True),
        (38, G1_ACTION_COUNT, True),
        (67, G1_ACTION_COUNT, True),
    ):
        components = (
            UNITREE_G1_POLICY_TO_SDK_JOINT
            if joint_ordered
            else np.arange(width, dtype=np.int64)
        )
        for history in range(G1_ACTOR_HISTORY):
            indices.extend(
                (
                    history * G1_ACTOR_FRAME_SIZE
                    + frame_offset
                    + components
                ).tolist()
            )
    result = np.asarray(indices, dtype=np.int64)
    if (
        result.shape != (G1_ACTOR_OBSERVATION_SIZE,)
        or np.unique(result).size != G1_ACTOR_OBSERVATION_SIZE
    ):
        raise RuntimeError("Unitree G1 actor input permutation is invalid")
    return result


def import_unitree_g1_velocity_policy(
    official_repository: str | Path,
    output: str | Path,
    *,
    library_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convert Unitree's pinned official G1 actor into one native PolicyPack."""

    try:
        import onnx
        from onnx import numpy_helper
        from onnx.reference import ReferenceEvaluator
    except ImportError as error:
        raise RuntimeError(
            "Unitree policy import requires the optional 'onnx' package"
        ) from error

    source, provenance = _verify_unitree_velocity_policy_source(
        official_repository
    )
    model = onnx.load(source)
    onnx.checker.check_model(model)
    if (
        len(model.graph.input) != 1
        or len(model.graph.output) != 1
        or [dimension.dim_value for dimension in
            model.graph.input[0].type.tensor_type.shape.dim]
            != [1, G1_ACTOR_OBSERVATION_SIZE]
        or [dimension.dim_value for dimension in
            model.graph.output[0].type.tensor_type.shape.dim]
            != [1, G1_ACTION_COUNT]
    ):
        raise ValueError("official Unitree G1 actor shape changed")
    tensors = {
        value.name: np.asarray(
            numpy_helper.to_array(value), dtype=np.float32
        )
        for value in model.graph.initializer
    }
    layer_names = tuple(
        (f"actor.{index}.weight", f"actor.{index}.bias")
        for index in (0, 2, 4, 6)
    )
    expected_shapes = (
        (512, G1_ACTOR_OBSERVATION_SIZE),
        (256, 512),
        (128, 256),
        (G1_ACTION_COUNT, 128),
    )
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for names, shape in zip(layer_names, expected_shapes, strict=True):
        try:
            weight = np.ascontiguousarray(tensors[names[0]])
            bias = np.ascontiguousarray(tensors[names[1]])
        except KeyError as error:
            raise ValueError(
                "official Unitree G1 actor layer names changed"
            ) from error
        if weight.shape != shape or bias.shape != (shape[0],):
            raise ValueError("official Unitree G1 actor topology changed")
        weights.append(weight)
        biases.append(bias)

    input_indices = _unitree_actor_input_indices()
    first_weight = np.empty_like(weights[0])
    first_weight[:, input_indices] = weights[0]
    final_weight = np.empty_like(weights[-1])
    final_bias = np.empty_like(biases[-1])
    final_weight[UNITREE_G1_POLICY_TO_SDK_JOINT] = weights[-1]
    final_bias[UNITREE_G1_POLICY_TO_SDK_JOINT] = biases[-1]
    converted_weights = (
        first_weight,
        weights[1],
        weights[2],
        final_weight,
    )
    converted_biases = (
        biases[0], biases[1], biases[2], final_bias,
    )
    target = write_policy_pack(
        output,
        policy_id="unitree_g1_velocity_v0_4960b84",
        revision=1,
        layers=tuple(
            PolicyDenseLayerArtifact(
                weights=weight,
                bias=bias,
                activation=3 if index < 3 else 0,
            )
            for index, (weight, bias) in enumerate(
                zip(
                    converted_weights,
                    converted_biases,
                    strict=True,
                )
            )
        ),
        library_path=library_path,
    )
    converted = read_policy_pack(target, library_path=library_path)

    sample_ids = np.arange(
        3 * G1_ACTOR_OBSERVATION_SIZE,
        dtype=np.float32,
    ).reshape((3, G1_ACTOR_OBSERVATION_SIZE))
    native_observations = (
        0.35 * np.sin(0.013 * sample_ids)
    ).astype(np.float32)
    reference = ReferenceEvaluator(model)
    unitree_actions = np.concatenate(
        tuple(
            np.asarray(
                reference.run(
                    None,
                    {
                        model.graph.input[0].name:
                            native_observations[index:index + 1,
                                                input_indices]
                    },
                )[0],
                dtype=np.float32,
            )
            for index in range(native_observations.shape[0])
        ),
        axis=0,
    )
    expected_native_actions = np.empty_like(unitree_actions)
    expected_native_actions[:, UNITREE_G1_POLICY_TO_SDK_JOINT] = (
        unitree_actions
    )
    converted_actions = converted.actions(native_observations)
    maximum_error = float(
        np.max(np.abs(converted_actions - expected_native_actions))
    )
    if maximum_error > 5.0e-5:
        raise RuntimeError(
            "converted PolicyPack differs from the official Unitree actor: "
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


def _verify_simulator_source(
    model_path: Path,
    contract: dict[str, Any],
) -> dict[str, str]:
    expected_commit = str(contract["simulator_commit"])
    expected_path = str(contract["simulator_model_path"])
    repository_root: Path | None = None
    for candidate in (model_path.parent, *model_path.parents):
        if (candidate / ".git").exists():
            repository_root = candidate
            break
    if repository_root is None:
        raise ValueError(
            "official sim2sim evidence requires the model inside its "
            "pinned Unitree MuJoCo Git checkout"
        )
    relative_path = model_path.relative_to(
        repository_root
    ).as_posix()
    if relative_path != expected_path:
        raise ValueError(
            "MuJoCo model path differs from the pinned native contract"
        )

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", "-C", str(repository_root), *arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        return completed.stdout.strip()

    try:
        revision = git("rev-parse", "HEAD")
        git("ls-files", "--error-unmatch", relative_path)
        dirty = git(
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(
            "could not verify the official Unitree MuJoCo source"
        ) from error
    if revision != expected_commit or dirty:
        raise ValueError(
            "Unitree MuJoCo checkout is not the clean pinned revision"
        )
    return {
        "simulator_repository":
            str(contract["simulator_repository"]),
        "simulator_commit": revision,
        "simulator_model_path": relative_path,
    }


class G1NumpyPolicy:
    """Portable deterministic actor used by sim2sim and export parity."""

    def __init__(
        self,
        policy_pack: str | Path,
        *,
        library_path: str | Path | None = None,
    ) -> None:
        self.policy_pack_path = (
            Path(policy_pack).expanduser().resolve()
        )
        self.pack = _g1_policy(
            self.policy_pack_path,
            library_path,
        )

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        return self.pack.actions(observation)


class G1ObservationHistory:
    """Exact 96-value frame assembly and five-frame deployment history."""

    def __init__(self, default_pose: np.ndarray) -> None:
        pose = np.asarray(default_pose, dtype=np.float32)
        if pose.shape != (G1_ACTION_COUNT,):
            raise ValueError(
                "G1 default pose does not match the actor action count"
            )
        self.default_pose = pose
        self.previous_action = np.zeros(
            G1_ACTION_COUNT,
            dtype=np.float32,
        )
        self.history = np.zeros(
            (G1_ACTOR_HISTORY, G1_ACTOR_FRAME_SIZE),
            dtype=np.float32,
        )
        self.initialized = False

    def reset(
        self,
        sensors: G1DeployableSensors,
        command: np.ndarray,
    ) -> np.ndarray:
        self.previous_action.fill(0.0)
        frame = self._frame(sensors, command)
        self.history[:] = frame
        self.initialized = True
        return self.history.reshape(-1).copy()

    def update(
        self,
        sensors: G1DeployableSensors,
        command: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        if not self.initialized:
            return self.reset(sensors, command)
        self.previous_action = np.asarray(
            action,
            dtype=np.float32,
        )
        self.history[:-1] = self.history[1:]
        self.history[-1] = self._frame(sensors, command)
        return self.history.reshape(-1).copy()

    def _frame(
        self,
        sensors: G1DeployableSensors,
        command: np.ndarray,
    ) -> np.ndarray:
        command_array = np.asarray(command, dtype=np.float32)
        if command_array.shape != (3,):
            raise ValueError("G1 velocity command must contain 3 values")
        gravity = np.asarray(
            sensors.projected_gravity,
            dtype=np.float32,
        )
        gravity /= max(float(np.linalg.norm(gravity)), 1.0e-6)
        frame = np.concatenate(
            (
                0.2
                * np.asarray(
                    sensors.base_angular_velocity,
                    dtype=np.float32,
                ),
                gravity,
                command_array,
                np.asarray(
                    sensors.joint_position,
                    dtype=np.float32,
                )
                - self.default_pose,
                0.05
                * np.asarray(
                    sensors.joint_velocity,
                    dtype=np.float32,
                ),
                self.previous_action,
            )
        )
        if frame.shape != (G1_ACTOR_FRAME_SIZE,):
            raise ValueError("deployable G1 sensor dimensions changed")
        return frame


def _parity_observations() -> np.ndarray:
    sample = np.arange(
        3 * G1_ACTOR_OBSERVATION_SIZE,
        dtype=np.float32,
    ).reshape((3, G1_ACTOR_OBSERVATION_SIZE))
    return (0.25 * np.sin(0.017 * sample)).astype(np.float32)


def export_g1_onnx(
    policy_pack: str | Path,
    output: str | Path,
    *,
    library_path: str | Path | None = None,
) -> Path:
    """Export the canonical G1 PolicyPack actor with parity evidence."""

    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as error:
        raise RuntimeError(
            "ONNX export requires the optional 'onnx' package"
        ) from error
    pack = _g1_policy(policy_pack, library_path)
    contract = _g1_contract(library_path)
    nodes = []
    initializers = [
        numpy_helper.from_array(
            pack.effective_observation_mean,
            "observation_mean",
        ),
        numpy_helper.from_array(
            pack.effective_observation_inverse_standard_deviation,
            "observation_inverse_standard_deviation",
        ),
        numpy_helper.from_array(
            np.asarray(-pack.observation_clip, dtype=np.float32),
            "observation_clip_minimum",
        ),
        numpy_helper.from_array(
            np.asarray(pack.observation_clip, dtype=np.float32),
            "observation_clip_maximum",
        ),
        numpy_helper.from_array(
            pack.effective_action_scale,
            "action_scale",
        ),
        numpy_helper.from_array(
            pack.effective_action_bias,
            "action_bias",
        ),
        numpy_helper.from_array(
            np.asarray(-pack.action_clip, dtype=np.float32),
            "action_clip_minimum",
        ),
        numpy_helper.from_array(
            np.asarray(pack.action_clip, dtype=np.float32),
            "action_clip_maximum",
        ),
    ]
    source = "actor_observation"
    nodes.extend(
        (
            helper.make_node(
                "Sub",
                (source, "observation_mean"),
                ("centered_observation",),
            ),
            helper.make_node(
                "Mul",
                (
                    "centered_observation",
                    "observation_inverse_standard_deviation",
                ),
                ("normalized_observation",),
            ),
            helper.make_node(
                "Clip",
                (
                    "normalized_observation",
                    "observation_clip_minimum",
                    "observation_clip_maximum",
                ),
                ("clipped_observation",),
            ),
        )
    )
    source = "clipped_observation"
    for index, layer in enumerate(pack.layers):
        weight_name = f"actor_weight_{index}"
        bias_name = f"actor_bias_{index}"
        linear_name = f"actor_linear_{index}"
        initializers.extend(
            (
                numpy_helper.from_array(layer.weights, weight_name),
                numpy_helper.from_array(layer.bias, bias_name),
            )
        )
        nodes.append(
            helper.make_node(
                "Gemm",
                (source, weight_name, bias_name),
                (linear_name,),
                transB=1,
            )
        )
        if layer.activation != 0:
            activated = f"actor_activation_{index}"
            activation = {
                1: "Relu",
                2: "Tanh",
                3: "Elu",
            }.get(layer.activation)
            if activation is None:
                raise ValueError(
                    "ONNX G1 export does not support this PolicyPack activation"
                )
            nodes.append(
                helper.make_node(
                    activation,
                    (linear_name,),
                    (activated,),
                    **(
                        {"alpha": 1.0}
                        if activation == "Elu"
                        else {}
                    ),
                )
            )
            source = activated
        else:
            source = linear_name
    if pack.format_version == 2:
        nodes.append(
            helper.make_node(
                "Tanh",
                (source,),
                ("legacy_squashed_action",),
            )
        )
        source = "legacy_squashed_action"
    nodes.extend(
        (
            helper.make_node(
                "Mul",
                (source, "action_scale"),
                ("scaled_action",),
            ),
            helper.make_node(
                "Add",
                ("scaled_action", "action_bias"),
                ("biased_action",),
            ),
            helper.make_node(
                "Clip",
                (
                    "biased_action",
                    "action_clip_minimum",
                    "action_clip_maximum",
                ),
                ("normalized_action",),
            ),
        )
    )
    graph = helper.make_graph(
        nodes,
        "MetalRoboG1Locomotion",
        (
            helper.make_tensor_value_info(
                "actor_observation",
                TensorProto.FLOAT,
                (None, G1_ACTOR_OBSERVATION_SIZE),
            ),
        ),
        (
            helper.make_tensor_value_info(
                "normalized_action",
                TensorProto.FLOAT,
                (None, G1_ACTION_COUNT),
            ),
        ),
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="MetalRobo",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    for key, value in {
        "joint_order": ",".join(contract["joint_order"]),
        "action_scale_radians": str(
            contract["action_scale_radians"]
        ),
        "policy_id": pack.id,
        "policy_revision": str(pack.revision),
        "policy_content_hash": str(pack.content_hash),
    }.items():
        metadata = model.metadata_props.add()
        metadata.key = key
        metadata.value = value
    onnx.checker.check_model(model)
    from onnx.reference import ReferenceEvaluator

    parity_input = _parity_observations()
    expected = G1NumpyPolicy(
        policy_pack,
        library_path=library_path,
    )(parity_input)
    actual = ReferenceEvaluator(model).run(
        None,
        {"actor_observation": parity_input},
    )[0]
    if not np.allclose(actual, expected, rtol=2.0e-5, atol=2.0e-5):
        raise RuntimeError("ONNX G1 policy failed export parity")
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, target)
    return target


def export_g1_mlx(
    policy_pack: str | Path,
    output_directory: str | Path,
    *,
    library_path: str | Path | None = None,
) -> Path:
    """Write MLX actor tensors plus the native deployment contract."""

    pack = _g1_policy(policy_pack, library_path)
    contract = _g1_contract(library_path)
    actor_weights: dict[str, mx.array] = {}
    for index, layer in enumerate(pack.layers):
        actor_weights[
            f"actor.layers.{index}.weight"
        ] = mx.array(layer.weights)
        actor_weights[
            f"actor.layers.{index}.bias"
        ] = mx.array(layer.bias)
    actor_weights.update(
        {
            "actor.observation_mean": mx.array(
                pack.effective_observation_mean
            ),
            "actor.observation_inverse_standard_deviation":
                mx.array(
                    pack
                        .effective_observation_inverse_standard_deviation
                ),
            "actor.action_scale": mx.array(
                pack.effective_action_scale
            ),
            "actor.action_bias": mx.array(
                pack.effective_action_bias
            ),
        }
    )
    target = Path(output_directory).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(target / "g1-locomotion.safetensors"),
        actor_weights,
        metadata={
            "format": "metalrobo.policy-pack-mlx-actor",
            "schema": "1",
            "policy_id": pack.id,
            "policy_revision": str(pack.revision),
            "policy_content_hash": str(pack.content_hash),
            "observation_clip": str(pack.observation_clip),
            "action_clip": str(pack.action_clip),
        },
    )
    deployment = {
        **contract,
        "format": "metalrobo.g1-policy-deployment",
        "schema": 1,
        "observation": {
            "history_frames": G1_ACTOR_HISTORY,
            "frame_size": G1_ACTOR_FRAME_SIZE,
            "frame_order": (
                "base_angular_velocity_x0.2",
                "projected_gravity",
                "velocity_command",
                "joint_position_minus_default",
                "joint_velocity_x0.05",
                "previous_normalized_action",
            ),
        },
        "policy": {
            "id": pack.id,
            "revision": pack.revision,
            "content_hash": pack.content_hash,
            "observation_clip": pack.observation_clip,
            "action_clip": pack.action_clip,
        },
    }
    (target / "g1-locomotion-deployment.json").write_text(
        json.dumps(
            deployment,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def export_g1_coreml(
    policy_pack: str | Path,
    output: str | Path,
    *,
    library_path: str | Path | None = None,
) -> Path:
    """Write a Core ML package from the canonical PolicyPack actor."""

    try:
        import coremltools as ct
        from coremltools.models import datatypes
        from coremltools.models.neural_network import (
            NeuralNetworkBuilder,
        )
    except ImportError as error:
        raise RuntimeError(
            "Core ML export requires the optional 'coremltools' package"
        ) from error
    pack = _g1_policy(policy_pack, library_path)
    contract = _g1_contract(library_path)
    expected_activations = [3] * (len(pack.layers) - 1) + [0]
    if [
        layer.activation for layer in pack.layers
    ] != expected_activations:
        raise ValueError(
            "Core ML G1 export requires ELU hidden layers and an identity output"
        )
    builder = NeuralNetworkBuilder(
        [
            (
                "actor_observation",
                datatypes.Array(G1_ACTOR_OBSERVATION_SIZE),
            )
        ],
        [
            (
                "normalized_action",
                datatypes.Array(G1_ACTION_COUNT),
            )
        ],
        disable_rank5_shape_mapping=True,
        use_float_arraytype=True,
    )
    observation_inverse_standard_deviation = (
        pack.effective_observation_inverse_standard_deviation
    )
    builder.add_load_constant_nd(
        name="observation_inverse_standard_deviation",
        output_name="observation_inverse_standard_deviation",
        constant_value=observation_inverse_standard_deviation,
        shape=[G1_ACTOR_OBSERVATION_SIZE],
    )
    builder.add_elementwise(
        name="scale_observation",
        input_names=[
            "actor_observation",
            "observation_inverse_standard_deviation",
        ],
        output_name="scaled_observation",
        mode="MULTIPLY",
    )
    builder.add_load_constant_nd(
        name="observation_normalization_bias",
        output_name="observation_normalization_bias",
        constant_value=(
            -pack.effective_observation_mean
            * observation_inverse_standard_deviation
        ),
        shape=[G1_ACTOR_OBSERVATION_SIZE],
    )
    builder.add_elementwise(
        name="normalize_observation",
        input_names=[
            "scaled_observation",
            "observation_normalization_bias",
        ],
        output_name="normalized_observation",
        mode="ADD",
    )
    builder.add_clip(
        name="clip_observation",
        input_name="normalized_observation",
        output_name="clipped_observation",
        min_value=-pack.observation_clip,
        max_value=pack.observation_clip,
    )
    source = "clipped_observation"
    for index, layer in enumerate(pack.layers):
        linear = f"actor_linear_{index}"
        builder.add_inner_product(
            name=linear,
            W=layer.weights,
            b=layer.bias,
            input_channels=layer.weights.shape[1],
            output_channels=layer.weights.shape[0],
            has_bias=True,
            input_name=source,
            output_name=linear,
        )
        source = linear
        if index + 1 < len(pack.layers):
            activated = f"actor_elu_{index}"
            builder.add_activation(
                name=activated,
                non_linearity="ELU",
                input_name=linear,
                output_name=activated,
                params=1.0,
            )
            source = activated
    if pack.format_version == 2:
        builder.add_activation(
            name="legacy_squashed_action",
            non_linearity="TANH",
            input_name=source,
            output_name="legacy_squashed_action",
        )
        source = "legacy_squashed_action"
    builder.add_load_constant_nd(
        name="action_scale",
        output_name="action_scale",
        constant_value=pack.effective_action_scale,
        shape=[G1_ACTION_COUNT],
    )
    builder.add_elementwise(
        name="scale_action",
        input_names=[source, "action_scale"],
        output_name="scaled_action",
        mode="MULTIPLY",
    )
    builder.add_load_constant_nd(
        name="action_bias",
        output_name="action_bias",
        constant_value=pack.effective_action_bias,
        shape=[G1_ACTION_COUNT],
    )
    builder.add_elementwise(
        name="transform_action",
        input_names=["scaled_action", "action_bias"],
        output_name="transformed_action",
        mode="ADD",
    )
    builder.add_clip(
        name="normalized_action",
        input_name="transformed_action",
        output_name="normalized_action",
        min_value=-pack.action_clip,
        max_value=pack.action_clip,
    )
    specification = builder.spec
    specification.description.metadata.shortDescription = (
        "MetalRobo G1 29-DoF proprioceptive locomotion actor"
    )
    specification.description.metadata.userDefined.update(
        {
            "joint_order": ",".join(contract["joint_order"]),
            "action_scale_radians": str(
                contract["action_scale_radians"]
            ),
            "policy_id": pack.id,
            "policy_revision": str(pack.revision),
            "policy_content_hash": str(pack.content_hash),
            "recommended_compute_units": "cpuAndGPU",
        }
    )
    parity_observations = _parity_observations()
    expected = G1NumpyPolicy(
        policy_pack,
        library_path=library_path,
    )(parity_observations)

    def predict(
        model: object,
    ) -> np.ndarray:
        return np.stack(
            [
                np.asarray(
                    model.predict(
                        {"actor_observation": observation}
                    )["normalized_action"],
                    dtype=np.float32,
                )
                for observation in parity_observations
            ]
        )

    cpu_gpu_model = ct.models.MLModel(
        specification,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    cpu_gpu_actual = predict(cpu_gpu_model)
    cpu_gpu_maximum_absolute_error = float(
        np.max(np.abs(cpu_gpu_actual - expected))
    )
    if not np.allclose(
        cpu_gpu_actual,
        expected,
        rtol=1.0e-4,
        atol=1.0e-4,
    ):
        raise RuntimeError(
            "Core ML G1 policy failed CPU/GPU export parity: "
            "maximum_absolute_error="
            f"{cpu_gpu_maximum_absolute_error:.9g}"
        )
    all_compute_units_model = ct.models.MLModel(
        specification,
        compute_units=ct.ComputeUnit.ALL,
    )
    all_compute_units_actual = predict(all_compute_units_model)
    all_compute_units_maximum_absolute_error = float(
        np.max(np.abs(all_compute_units_actual - expected))
    )
    if all_compute_units_maximum_absolute_error > 5.0e-3:
        raise RuntimeError(
            "Core ML G1 policy failed all-compute-units export parity: "
            "maximum_absolute_error="
            f"{all_compute_units_maximum_absolute_error:.9g}"
        )
    specification.description.metadata.userDefined.update(
        {
            "cpu_gpu_parity_maximum_absolute_error":
                f"{cpu_gpu_maximum_absolute_error:.9g}",
            "all_compute_units_parity_maximum_absolute_error":
                f"{all_compute_units_maximum_absolute_error:.9g}",
        }
    )
    model = ct.models.MLModel(
        specification,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(target))
    return target


class UnitreeG1MuJoCoRunner:
    """Independent-engine playback using deployable observations only."""

    def __init__(
        self,
        policy_pack: str | Path,
        model_path: str | Path,
        *,
        library_path: str | Path | None = None,
    ) -> None:
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError(
                "sim2sim requires the optional official 'mujoco' package"
            ) from error
        self.mujoco = mujoco
        self.policy = G1NumpyPolicy(
            policy_pack,
            library_path=library_path,
        )
        self.contract = _g1_contract(library_path)
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise ValueError("official Unitree MuJoCo model was not found")
        self.simulator_source = _verify_simulator_source(
            self.model_path,
            self.contract,
        )
        self.model_sha256 = hashlib.sha256(
            self.model_path.read_bytes()
        ).hexdigest()
        self.model = mujoco.MjModel.from_xml_path(
            str(self.model_path)
        )
        self.physics_timestep = float(
            self.contract["physics_timestep_seconds"]
        )
        self.policy_timestep = float(
            self.contract["policy_timestep_seconds"]
        )
        substeps = self.policy_timestep / self.physics_timestep
        self.physics_substeps = int(round(substeps))
        if (
            self.physics_timestep <= 0.0
            or self.policy_timestep <= 0.0
            or self.physics_substeps <= 0
            or not math.isclose(
                substeps,
                self.physics_substeps,
                rel_tol=0.0,
                abs_tol=1.0e-7,
            )
        ):
            raise ValueError(
                "native G1 policy and physics timesteps are incompatible"
            )
        self.model.opt.timestep = self.physics_timestep
        self.data = mujoco.MjData(self.model)
        self.pelvis_body = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "pelvis",
        )
        if self.pelvis_body < 0:
            raise ValueError("MuJoCo model has no pelvis body")
        self.qpos_addresses = []
        self.qvel_addresses = []
        self.actuator_addresses = []
        self._joint_ids = []
        for joint_name in self.contract["joint_order"]:
            joint = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint < 0:
                raise ValueError(
                    f"MuJoCo model is missing {joint_name}"
                )
            self.qpos_addresses.append(
                int(self.model.jnt_qposadr[joint])
            )
            self.qvel_addresses.append(
                int(self.model.jnt_dofadr[joint])
            )
            self._joint_ids.append(joint)
            actuator = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                joint_name,
            )
            if actuator < 0:
                matches = np.flatnonzero(
                    self.model.actuator_trnid[:, 0] == joint
                )
                if matches.size != 1:
                    raise ValueError(
                        f"MuJoCo actuator mapping is ambiguous for "
                        f"{joint_name}"
                    )
                actuator = int(matches[0])
            self.actuator_addresses.append(actuator)
        self.default_pose = np.asarray(
            self.contract["default_pose"],
            dtype=np.float32,
        )
        self.stiffness = np.asarray(
            self.contract["stiffness"],
            dtype=np.float64,
        )
        self.damping = np.asarray(
            self.contract["damping"],
            dtype=np.float64,
        )
        self.effort_limits = np.asarray(
            self.contract["effort_limits"],
            dtype=np.float64,
        )
        self.position_limits = np.asarray(
            self.contract["position_limits"],
            dtype=np.float64,
        )
        self.action_scale_radians = float(
            self.contract["action_scale_radians"]
        )
        self.drive_prediction_seconds = float(
            self.contract["drive_prediction_seconds"]
        )
        if (
            self.stiffness.shape != (G1_ACTION_COUNT,)
            or self.damping.shape != (G1_ACTION_COUNT,)
            or self.effort_limits.shape != (G1_ACTION_COUNT,)
            or self.position_limits.shape != (G1_ACTION_COUNT, 2)
        ):
            raise ValueError(
                "G1 policy pack has incomplete actuator metadata"
            )
        model_ranges = np.asarray(
            self.model.jnt_range[self._joint_ids],
            dtype=np.float64,
        )
        if not np.allclose(
            model_ranges,
            self.position_limits,
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise ValueError(
                "official MuJoCo joint limits differ from the policy pack"
            )
        self.actuator_gears = np.asarray(
            self.model.actuator_gear[
                self.actuator_addresses, 0
            ],
            dtype=np.float64,
        )
        if np.any(np.abs(self.actuator_gears) < 1.0e-8):
            raise ValueError("MuJoCo actuator gear cannot be zero")
        self.actuator_control_limited = np.asarray(
            self.model.actuator_ctrllimited[
                self.actuator_addresses
            ],
            dtype=bool,
        )
        self.actuator_control_ranges = np.asarray(
            self.model.actuator_ctrlrange[
                self.actuator_addresses
            ],
            dtype=np.float64,
        )
        self.history = G1ObservationHistory(self.default_pose)

    def _sensors(self) -> G1DeployableSensors:
        velocity = np.zeros(6, dtype=np.float64)
        self.mujoco.mj_objectVelocity(
            self.model,
            self.data,
            self.mujoco.mjtObj.mjOBJ_BODY,
            self.pelvis_body,
            velocity,
            1,
        )
        rotation = np.asarray(
            self.data.xmat[self.pelvis_body],
            dtype=np.float64,
        ).reshape(3, 3)
        projected_gravity = rotation.T @ np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float64,
        )
        return G1DeployableSensors(
            base_angular_velocity=velocity[:3].astype(np.float32),
            projected_gravity=projected_gravity.astype(np.float32),
            joint_position=np.asarray(
                self.data.qpos[self.qpos_addresses],
                dtype=np.float32,
            ),
            joint_velocity=np.asarray(
                self.data.qvel[self.qvel_addresses],
                dtype=np.float32,
            ),
        )

    def reset(self, command: np.ndarray) -> np.ndarray:
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_addresses] = self.default_pose
        self.mujoco.mj_forward(self.model, self.data)
        return self.history.reset(self._sensors(), command)

    def _apply_joint_target(self, target: np.ndarray) -> int:
        """Evaluate the joint servo at the native physics cadence."""

        joint_position = np.asarray(
            self.data.qpos[self.qpos_addresses],
            dtype=np.float64,
        )
        joint_velocity = np.asarray(
            self.data.qvel[self.qvel_addresses],
            dtype=np.float64,
        )
        unclipped_torque = (
            self.stiffness
            * (
                target
                - joint_position
                - self.drive_prediction_seconds * joint_velocity
            )
            - self.damping * joint_velocity
        )
        desired_torque = np.clip(
            unclipped_torque,
            -self.effort_limits,
            self.effort_limits,
        )
        control = desired_torque / self.actuator_gears
        limited_control = np.where(
            self.actuator_control_limited,
            np.clip(
                control,
                self.actuator_control_ranges[:, 0],
                self.actuator_control_ranges[:, 1],
            ),
            control,
        )
        self.data.ctrl[self.actuator_addresses] = limited_control
        effort_saturated = np.abs(unclipped_torque) > (
            self.effort_limits + 1.0e-9
        )
        control_saturated = self.actuator_control_limited & (
            np.abs(limited_control - control) > 1.0e-9
        )
        return int(np.count_nonzero(
            effort_saturated | control_saturated
        ))

    def run(
        self,
        command: np.ndarray,
        *,
        seconds: float = 20.0,
        zero_action: bool = False,
    ) -> dict[str, float]:
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise ValueError("sim2sim duration must be positive")
        command_array = np.asarray(command, dtype=np.float32)
        observation = self.reset(command_array)
        action = np.zeros(G1_ACTION_COUNT, dtype=np.float32)
        control_steps = int(
            math.ceil(seconds / self.policy_timestep)
        )
        survived = 0
        squared_planar_error = 0.0
        squared_yaw_error = 0.0
        summed_local_velocity = np.zeros(3, dtype=np.float64)
        maximum_absolute_yaw_velocity = 0.0
        maximum_absolute_joint_velocity = 0.0
        saturated_joint_samples = 0
        initial_pelvis_position = np.asarray(
            self.data.xpos[self.pelvis_body],
            dtype=np.float64,
        ).copy()
        termination_reason = "duration"
        for _ in range(control_steps):
            action = (
                np.zeros(G1_ACTION_COUNT, dtype=np.float32)
                if zero_action
                else self.policy(observation[None, :])[0]
            )
            target = (
                self.default_pose
                + self.action_scale_radians * action
            )
            target = np.clip(
                target,
                self.position_limits[:, 0],
                self.position_limits[:, 1],
            )
            for _ in range(self.physics_substeps):
                saturated_joint_samples += (
                    self._apply_joint_target(target)
                )
                self.mujoco.mj_step(self.model, self.data)
            sensors = self._sensors()
            observation = self.history.update(
                sensors,
                command_array,
                action,
            )
            pelvis_height = float(
                self.data.xpos[self.pelvis_body, 2]
            )
            tilt = math.acos(
                float(
                    np.clip(
                        -sensors.projected_gravity[2],
                        -1.0,
                        1.0,
                    )
                )
            )
            if pelvis_height < 0.2 or tilt > 0.8:
                termination_reason = (
                    "height" if pelvis_height < 0.2 else "tilt"
                )
                break
            survived += 1
            local_velocity = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_objectVelocity(
                self.model,
                self.data,
                self.mujoco.mjtObj.mjOBJ_BODY,
                self.pelvis_body,
                local_velocity,
                1,
            )
            summed_local_velocity += local_velocity[[3, 4, 2]]
            maximum_absolute_yaw_velocity = max(
                maximum_absolute_yaw_velocity,
                abs(float(local_velocity[2])),
            )
            maximum_absolute_joint_velocity = max(
                maximum_absolute_joint_velocity,
                float(np.max(np.abs(
                    self.data.qvel[self.qvel_addresses]
                ))),
            )
            squared_planar_error += float(
                np.sum(
                    np.square(
                        local_velocity[3:5] - command_array[:2]
                    )
                )
            )
            squared_yaw_error += float(
                (
                    local_velocity[2] - command_array[2]
                )
                ** 2
            )
        denominator = max(survived, 1)
        mean_local_velocity = summed_local_velocity / denominator
        final_pelvis_position = np.asarray(
            self.data.xpos[self.pelvis_body],
            dtype=np.float64,
        )
        displacement = final_pelvis_position - initial_pelvis_position
        planar_command_squared = float(
            np.dot(command_array[:2], command_array[:2])
        )
        planar_response_ratio = (
            float(
                np.dot(
                    mean_local_velocity[:2],
                    command_array[:2],
                ) / planar_command_squared
            )
            if planar_command_squared > 1.0e-8
            else 0.0
        )
        yaw_response_ratio = (
            float(mean_local_velocity[2] / command_array[2])
            if abs(float(command_array[2])) > 1.0e-8
            else 0.0
        )
        return {
            "policy_id": self.policy.pack.id,
            "policy_revision": self.policy.pack.revision,
            "policy_content_hash": self.policy.pack.content_hash,
            "official_model": str(self.model_path),
            "official_model_sha256": self.model_sha256,
            "official_source_verified": True,
            **self.simulator_source,
            "unitree_source_repository":
                self.contract["source_repository"],
            "unitree_source_commit": self.contract["source_commit"],
            "unitree_source_model_path":
                self.contract["source_model_path"],
            "requested_seconds": float(seconds),
            "command_vx": float(command_array[0]),
            "command_vy": float(command_array[1]),
            "command_yaw": float(command_array[2]),
            "controller": (
                "zero_action_baseline"
                if zero_action
                else "policy"
            ),
            "survival_seconds":
                survived * self.policy_timestep,
            "survival_fraction": survived / control_steps,
            "termination_reason": termination_reason,
            "planar_velocity_rmse": math.sqrt(
                squared_planar_error / denominator
            ),
            "yaw_velocity_rmse": math.sqrt(
                squared_yaw_error / denominator
            ),
            "mean_local_vx": float(mean_local_velocity[0]),
            "mean_local_vy": float(mean_local_velocity[1]),
            "mean_local_yaw_velocity": float(
                mean_local_velocity[2]
            ),
            "world_displacement_x": float(displacement[0]),
            "world_displacement_y": float(displacement[1]),
            "planar_response_ratio": planar_response_ratio,
            "yaw_response_ratio": yaw_response_ratio,
            "maximum_absolute_yaw_velocity": (
                maximum_absolute_yaw_velocity
            ),
            "maximum_absolute_joint_velocity": (
                maximum_absolute_joint_velocity
            ),
            "saturated_joint_samples": saturated_joint_samples,
        }

    def run_promotion_suite(
        self,
        *,
        seconds: float = 20.0,
        zero_action: bool = False,
        minimum_survival_fraction: float = 0.99,
        minimum_response_ratio: float = 0.25,
    ) -> dict[str, Any]:
        """Run deterministic commands that distinguish balance from gait."""

        if not 0.0 < minimum_survival_fraction <= 1.0:
            raise ValueError(
                "minimum survival fraction must be in (0, 1]"
            )
        if not math.isfinite(minimum_response_ratio) or (
            minimum_response_ratio <= 0.0
        ):
            raise ValueError("minimum response ratio must be positive")
        reports: list[dict[str, Any]] = []
        failures: list[str] = []
        for name, command in G1_PROMOTION_COMMANDS:
            report = self.run(
                np.asarray(command, dtype=np.float32),
                seconds=seconds,
                zero_action=zero_action,
            )
            report["case"] = name
            survived = (
                report["survival_fraction"]
                >= minimum_survival_fraction
            )
            response = True
            if abs(command[0]) + abs(command[1]) > 0.0:
                response = (
                    report["planar_response_ratio"]
                    >= minimum_response_ratio
                )
            elif abs(command[2]) > 0.0:
                response = (
                    report["yaw_response_ratio"]
                    >= minimum_response_ratio
                )
            else:
                response = (
                    math.hypot(
                        report["mean_local_vx"],
                        report["mean_local_vy"],
                    ) <= 0.1
                    and abs(report["mean_local_yaw_velocity"])
                    <= 0.2
                )
            report["survival_pass"] = survived
            report["response_pass"] = response
            report["case_pass"] = survived and response
            if not report["case_pass"]:
                failures.append(name)
            reports.append(report)
        return {
            "policy_id": self.policy.pack.id,
            "policy_revision": self.policy.pack.revision,
            "controller": (
                "zero_action_baseline"
                if zero_action
                else "policy"
            ),
            "requested_seconds_per_case": float(seconds),
            "minimum_survival_fraction": (
                minimum_survival_fraction
            ),
            "minimum_response_ratio": minimum_response_ratio,
            "promotion_ready": not failures,
            "failed_cases": failures,
            "cases": reports,
        }


__all__ = [
    "G1DeployableSensors",
    "G1NumpyPolicy",
    "G1ObservationHistory",
    "G1_PROMOTION_COMMANDS",
    "UnitreeG1MuJoCoRunner",
    "export_g1_coreml",
    "export_g1_mlx",
    "export_g1_onnx",
    "import_unitree_g1_velocity_policy",
]
