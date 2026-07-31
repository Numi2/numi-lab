"""Deployment artifacts and independent Unitree MuJoCo playback for G1."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

G1_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
G1_ACTOR_FRAME_SIZE = 96
G1_ACTOR_HISTORY = 5
G1_ACTOR_OBSERVATION_SIZE = (
    G1_ACTOR_FRAME_SIZE * G1_ACTOR_HISTORY
)


@dataclass(frozen=True, slots=True)
class G1DeployableSensors:
    """Only signals available to the deployed proprioceptive actor."""

    base_angular_velocity: np.ndarray
    projected_gravity: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray


def _policy_record(checkpoint: Path) -> dict[str, Any]:
    record = json.loads(
        (checkpoint / "policy.json").read_text(encoding="utf-8")
    )
    if (
        record.get("format") != "metalrobo.g1-policy-pack"
        or record.get("actor_observation_size")
        != G1_ACTOR_OBSERVATION_SIZE
        or tuple(record.get("joint_order", ())) != G1_JOINT_ORDER
    ):
        raise ValueError("policy pack does not match the G1 actor ABI")
    weight_path = checkpoint / "model.safetensors"
    if not weight_path.is_file():
        raise ValueError("G1 policy pack has no model weights")
    with weight_path.open("rb") as weight_file:
        weight_hash = hashlib.file_digest(
            weight_file,
            "sha256",
        ).hexdigest()
    if weight_hash != record.get("weights_sha256"):
        raise ValueError("G1 policy weight identity does not match its pack")
    return record


def _actor_layers(
    checkpoint: Path,
) -> list[tuple[np.ndarray, np.ndarray]]:
    weights = mx.load(str(checkpoint / "model.safetensors"))
    if not isinstance(weights, dict):
        raise ValueError("G1 policy weights are not a named tensor map")
    layers = []
    index = 0
    while True:
        weight_name = f"actor.layers.{index}.weight"
        bias_name = f"actor.layers.{index}.bias"
        if weight_name in weights:
            layers.append(
                (
                    np.asarray(weights[weight_name], dtype=np.float32),
                    np.asarray(weights[bias_name], dtype=np.float32),
                )
            )
        index += 1
        if index > 32:
            break
    if len(layers) < 2:
        raise ValueError("G1 actor does not contain a complete MLP")
    return layers


class G1NumpyPolicy:
    """Portable deterministic actor used by sim2sim and export parity."""

    def __init__(self, checkpoint: str | Path) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.record = _policy_record(self.checkpoint)
        self.layers = _actor_layers(self.checkpoint)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape[-1] != G1_ACTOR_OBSERVATION_SIZE:
            raise ValueError(
                "G1 actor observation must contain 480 values"
            )
        for index, (weight, bias) in enumerate(self.layers):
            value = value @ weight.T + bias
            if index + 1 < len(self.layers):
                value = np.where(
                    value >= 0.0,
                    value,
                    np.expm1(value),
                )
        return np.tanh(value).astype(np.float32, copy=False)


class G1ObservationHistory:
    """Exact 96-value frame assembly and five-frame deployment history."""

    def __init__(self, default_pose: np.ndarray) -> None:
        pose = np.asarray(default_pose, dtype=np.float32)
        if pose.shape != (29,):
            raise ValueError("G1 default pose must contain 29 joints")
        self.default_pose = pose
        self.previous_action = np.zeros(29, dtype=np.float32)
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
    checkpoint: str | Path,
    output: str | Path,
) -> Path:
    """Write a dependency-light ONNX graph with the packaged joint order."""

    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as error:
        raise RuntimeError(
            "ONNX export requires the optional 'onnx' package"
        ) from error
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    record = _policy_record(checkpoint_path)
    layers = _actor_layers(checkpoint_path)
    nodes = []
    initializers = []
    source = "actor_observation"
    for index, (weight, bias) in enumerate(layers):
        weight_name = f"actor_weight_{index}"
        bias_name = f"actor_bias_{index}"
        linear_name = f"actor_linear_{index}"
        initializers.extend(
            (
                numpy_helper.from_array(weight, weight_name),
                numpy_helper.from_array(bias, bias_name),
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
        if index + 1 < len(layers):
            activated = f"actor_elu_{index}"
            nodes.append(
                helper.make_node(
                    "Elu",
                    (linear_name,),
                    (activated,),
                    alpha=1.0,
                )
            )
            source = activated
        else:
            nodes.append(
                helper.make_node(
                    "Tanh",
                    (linear_name,),
                    ("normalized_action",),
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
                (None, 29),
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
        "joint_order": ",".join(G1_JOINT_ORDER),
        "action_scale_radians": str(
            record["action_scale_radians"]
        ),
        "source_revision": str(record["source_revision"]),
    }.items():
        metadata = model.metadata_props.add()
        metadata.key = key
        metadata.value = value
    onnx.checker.check_model(model)
    from onnx.reference import ReferenceEvaluator

    parity_input = _parity_observations()
    expected = G1NumpyPolicy(checkpoint_path)(parity_input)
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
    checkpoint: str | Path,
    output_directory: str | Path,
) -> Path:
    """Write compact MLX actor weights and the complete deployment contract."""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    record = _policy_record(checkpoint_path)
    weights = mx.load(str(checkpoint_path / "model.safetensors"))
    if not isinstance(weights, dict):
        raise ValueError("G1 policy weights are not a named tensor map")
    actor_weights = {
        name: value
        for name, value in weights.items()
        if name.startswith("actor.")
    }
    if not actor_weights:
        raise ValueError("G1 checkpoint has no deployable actor weights")
    target = Path(output_directory).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(target / "g1-locomotion.safetensors"),
        actor_weights,
    )
    deployment = {
        "format": "metalrobo.g1-deployment",
        "physics_timestep_seconds": 0.005,
        "policy_timestep_seconds": 0.02,
        "joint_order": record["joint_order"],
        "default_pose": record["default_pose"],
        "stiffness": record["stiffness"],
        "damping": record["damping"],
        "position_limits": record["position_limits"],
        "velocity_limits": record["velocity_limits"],
        "effort_limits": record["effort_limits"],
        "action_scale_radians": record["action_scale_radians"],
        "drive_prediction_seconds":
            record["drive_prediction_seconds"],
        "command_limits": record["command_limits"],
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
        "source_revision": record["source_revision"],
        "source_model": record["source_model"],
        "policy_fingerprint": record["fingerprint"],
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
    checkpoint: str | Path,
    output: str | Path,
) -> Path:
    """Write a Core ML neural-network package with scene-free inputs."""

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
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    record = _policy_record(checkpoint_path)
    layers = _actor_layers(checkpoint_path)
    builder = NeuralNetworkBuilder(
        [
            (
                "actor_observation",
                datatypes.Array(G1_ACTOR_OBSERVATION_SIZE),
            )
        ],
        [("normalized_action", datatypes.Array(29))],
    )
    source = "actor_observation"
    for index, (weight, bias) in enumerate(layers):
        linear = f"actor_linear_{index}"
        builder.add_inner_product(
            name=linear,
            W=weight,
            b=bias,
            input_channels=weight.shape[1],
            output_channels=weight.shape[0],
            has_bias=True,
            input_name=source,
            output_name=linear,
        )
        if index + 1 < len(layers):
            activated = f"actor_elu_{index}"
            builder.add_activation(
                name=activated,
                non_linearity="ELU",
                input_name=linear,
                output_name=activated,
                params=(1.0,),
            )
            source = activated
        else:
            builder.add_activation(
                name="normalized_action",
                non_linearity="TANH",
                input_name=linear,
                output_name="normalized_action",
            )
    specification = builder.spec
    specification.description.metadata.shortDescription = (
        "MetalRobo G1 29-DoF proprioceptive locomotion actor"
    )
    specification.description.metadata.userDefined.update(
        {
            "joint_order": ",".join(G1_JOINT_ORDER),
            "action_scale_radians": str(
                record["action_scale_radians"]
            ),
            "source_revision": str(record["source_revision"]),
        }
    )
    model = ct.models.MLModel(specification)
    parity_input = _parity_observations()[0]
    expected = G1NumpyPolicy(checkpoint_path)(
        parity_input[None, :]
    )[0]
    actual = np.asarray(
        model.predict(
            {"actor_observation": parity_input}
        )["normalized_action"],
        dtype=np.float32,
    )
    if not np.allclose(actual, expected, rtol=1.0e-4, atol=1.0e-4):
        raise RuntimeError("Core ML G1 policy failed export parity")
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(target))
    return target


class UnitreeG1MuJoCoRunner:
    """Independent-engine playback using deployable observations only."""

    def __init__(
        self,
        checkpoint: str | Path,
        model_path: str | Path,
    ) -> None:
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError(
                "sim2sim requires the optional official 'mujoco' package"
            ) from error
        self.mujoco = mujoco
        self.policy = G1NumpyPolicy(checkpoint)
        self.record = self.policy.record
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise ValueError("official Unitree MuJoCo model was not found")
        self.model = mujoco.MjModel.from_xml_path(
            str(self.model_path)
        )
        self.model.opt.timestep = 0.005
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
        for joint_name in G1_JOINT_ORDER:
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
            self.record["default_pose"],
            dtype=np.float32,
        )
        self.stiffness = np.asarray(
            self.record["stiffness"],
            dtype=np.float64,
        )
        self.damping = np.asarray(
            self.record["damping"],
            dtype=np.float64,
        )
        self.effort_limits = np.asarray(
            self.record["effort_limits"],
            dtype=np.float64,
        )
        self.position_limits = np.asarray(
            self.record["position_limits"],
            dtype=np.float64,
        )
        if (
            self.stiffness.shape != (29,)
            or self.damping.shape != (29,)
            or self.effort_limits.shape != (29,)
            or self.position_limits.shape != (29, 2)
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

    def run(
        self,
        command: np.ndarray,
        *,
        seconds: float = 20.0,
    ) -> dict[str, float]:
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise ValueError("sim2sim duration must be positive")
        command_array = np.asarray(command, dtype=np.float32)
        observation = self.reset(command_array)
        action = np.zeros(29, dtype=np.float32)
        control_steps = int(math.ceil(seconds / 0.02))
        survived = 0
        squared_planar_error = 0.0
        squared_yaw_error = 0.0
        for _ in range(control_steps):
            action = self.policy(observation[None, :])[0]
            target = (
                self.default_pose
                + float(self.record["action_scale_radians"]) * action
            )
            target = np.clip(
                target,
                self.position_limits[:, 0],
                self.position_limits[:, 1],
            )
            joint_position = np.asarray(
                self.data.qpos[self.qpos_addresses],
                dtype=np.float64,
            )
            joint_velocity = np.asarray(
                self.data.qvel[self.qvel_addresses],
                dtype=np.float64,
            )
            desired_torque = (
                self.stiffness
                * (
                    target
                    - joint_position
                    - float(
                        self.record["drive_prediction_seconds"]
                    )
                    * joint_velocity
                )
                - self.damping * joint_velocity
            )
            desired_torque = np.clip(
                desired_torque,
                -self.effort_limits,
                self.effort_limits,
            )
            control = desired_torque / self.actuator_gears
            limited = np.asarray(
                self.model.actuator_ctrllimited[
                    self.actuator_addresses
                ],
                dtype=bool,
            )
            ranges = np.asarray(
                self.model.actuator_ctrlrange[
                    self.actuator_addresses
                ],
                dtype=np.float64,
            )
            control = np.where(
                limited,
                np.clip(control, ranges[:, 0], ranges[:, 1]),
                control,
            )
            self.data.ctrl[self.actuator_addresses] = control
            for _ in range(4):
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
        return {
            "requested_seconds": float(seconds),
            "survival_seconds": survived * 0.02,
            "survival_fraction": survived / control_steps,
            "planar_velocity_rmse": math.sqrt(
                squared_planar_error / denominator
            ),
            "yaw_velocity_rmse": math.sqrt(
                squared_yaw_error / denominator
            ),
        }


__all__ = [
    "G1DeployableSensors",
    "G1NumpyPolicy",
    "G1ObservationHistory",
    "UnitreeG1MuJoCoRunner",
    "export_g1_coreml",
    "export_g1_mlx",
    "export_g1_onnx",
]
