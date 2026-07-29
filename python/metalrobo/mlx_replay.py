"""Recorded-task replay across exact, GPU-resident candidate twins.

Capture ingestion is an explicit host boundary. Once a trace is loaded, every
candidate world advances through the same command stream as MLX arrays; no
Python code inspects an individual environment or physics step result.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import mlx.core as mx
import numpy as np
import numpy.typing as npt

from .mlx_world import (
    ControllerDelayState,
    MLXCompiledWorld,
    SampledWorldFamilyState,
    sampled_state_from_world_family,
    step_sampled_world_family,
)
from .mlx_r2s2r import ReplayEvaluation


def _readonly(
    value: npt.ArrayLike,
    *,
    dtype: npt.DTypeLike,
) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _masked_values(
    values: np.ndarray,
    mask: np.ndarray,
    label: str,
) -> np.ndarray:
    if not np.all(np.isfinite(values[mask])):
        raise ValueError(f"{label} contains non-finite measured values")
    return _readonly(np.where(mask, values, 0.0), dtype=np.float32)


def _mask_for(
    arrays: dict[str, np.ndarray],
    key: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    value = arrays.get(key)
    if value is None:
        return _readonly(np.ones(shape, dtype=np.bool_), dtype=np.bool_)
    mask = np.asarray(value, dtype=np.bool_)
    if mask.shape == shape[:-1] and len(shape) > 1:
        mask = np.broadcast_to(mask[..., None], shape)
    if mask.shape != shape:
        raise ValueError(f"{key} must have shape {shape}")
    return _readonly(mask, dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class ReplayResidualScales:
    robot_q: float = 0.05
    robot_v: float = 0.20
    object_position: float = 0.02
    rod_position: float = 0.01
    contact_timing: float = 1.0
    physics_failure: float = 1.0

    def validate(self) -> None:
        for value in (
            self.robot_q,
            self.robot_v,
            self.object_position,
            self.rod_position,
            self.contact_timing,
            self.physics_failure,
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "physical replay residual scales must be positive"
                )


@dataclass(frozen=True, slots=True)
class PhysicalReplayTrace:
    """A time-aligned physical episode in the anchor-world coordinate frame.

    ``commands`` has T joint-position targets. State observations have T+1
    frames so frame zero constrains the sampled reset and frame i+1 constrains
    the world after command i. Contact observations are post-step and have T
    entries. Missing measurements are represented only through explicit masks.
    """

    commands: np.ndarray
    robot_q: np.ndarray
    robot_q_mask: np.ndarray
    robot_v: np.ndarray | None
    robot_v_mask: np.ndarray | None
    object_positions: np.ndarray | None
    object_position_mask: np.ndarray | None
    scene_body_indices: np.ndarray
    rod_positions: np.ndarray | None
    rod_position_mask: np.ndarray | None
    rod_node_indices: np.ndarray
    contact_active: np.ndarray | None
    contact_mask: np.ndarray | None
    control_period_seconds: float
    scales: ReplayResidualScales

    @property
    def step_count(self) -> int:
        return int(self.commands.shape[0])

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        control_period_seconds: float | None = None,
        scales: ReplayResidualScales = ReplayResidualScales(),
    ) -> PhysicalReplayTrace:
        trace_path = Path(path).expanduser().resolve()
        if not trace_path.is_file():
            raise FileNotFoundError(
                f"physical replay trace does not exist: {trace_path}"
            )
        with np.load(trace_path, allow_pickle=False) as payload:
            arrays = {
                name: payload[name].copy()
                for name in payload.files
            }
        if "commands" not in arrays or "robot_q" not in arrays:
            raise ValueError(
                "physical replay NPZ requires commands and robot_q"
            )
        commands = _readonly(arrays["commands"], dtype=np.float32)
        robot_q_raw = np.asarray(arrays["robot_q"], dtype=np.float32)
        if commands.ndim != 2 or commands.shape[0] == 0:
            raise ValueError("commands must have shape [step, nv]")
        expected_frames = commands.shape[0] + 1
        if (
            robot_q_raw.ndim != 2
            or robot_q_raw.shape[0] != expected_frames
        ):
            raise ValueError(
                "robot_q must have shape [step + 1, nq]"
            )
        if not np.all(np.isfinite(commands)):
            raise ValueError("commands contain non-finite values")
        robot_q_mask = _mask_for(
            arrays,
            "robot_q_mask",
            robot_q_raw.shape,
        )
        if not bool(np.any(robot_q_mask)):
            raise ValueError("robot_q_mask contains no measurements")
        robot_q = _masked_values(
            robot_q_raw,
            robot_q_mask,
            "robot_q",
        )

        robot_v: np.ndarray | None = None
        robot_v_mask: np.ndarray | None = None
        if "robot_v" in arrays:
            robot_v_raw = np.asarray(
                arrays["robot_v"],
                dtype=np.float32,
            )
            if (
                robot_v_raw.ndim != 2
                or robot_v_raw.shape[0] != expected_frames
            ):
                raise ValueError(
                    "robot_v must have shape [step + 1, nv]"
                )
            candidate_mask = _mask_for(
                arrays,
                "robot_v_mask",
                robot_v_raw.shape,
            )
            if bool(np.any(candidate_mask)):
                robot_v = _masked_values(
                    robot_v_raw,
                    candidate_mask,
                    "robot_v",
                )
                robot_v_mask = candidate_mask

        object_positions: np.ndarray | None = None
        object_mask: np.ndarray | None = None
        scene_indices = _readonly([], dtype=np.uint32)
        if "object_positions" in arrays:
            object_raw = np.asarray(
                arrays["object_positions"],
                dtype=np.float32,
            )
            if object_raw.ndim == 2 and object_raw.shape[-1] == 3:
                object_raw = object_raw[:, None, :]
            if (
                object_raw.ndim != 3
                or object_raw.shape[0] != expected_frames
                or object_raw.shape[2] != 3
            ):
                raise ValueError(
                    "object_positions must have shape "
                    "[step + 1, object, 3]"
                )
            candidate_mask = _mask_for(
                arrays,
                "object_position_mask",
                object_raw.shape,
            )
            if bool(np.any(candidate_mask)):
                object_positions = _masked_values(
                    object_raw,
                    candidate_mask,
                    "object_positions",
                )
                object_mask = candidate_mask
                scene_indices = _readonly(
                    arrays.get(
                        "scene_body_indices",
                        np.arange(object_raw.shape[1]),
                    ),
                    dtype=np.uint32,
                )
                if scene_indices.shape != (object_raw.shape[1],):
                    raise ValueError(
                        "scene_body_indices must have one entry per "
                        "tracked object"
                    )

        rod_positions: np.ndarray | None = None
        rod_mask: np.ndarray | None = None
        rod_indices = _readonly([], dtype=np.uint32)
        if "rod_positions" in arrays:
            rod_raw = np.asarray(
                arrays["rod_positions"],
                dtype=np.float32,
            )
            if (
                rod_raw.ndim != 3
                or rod_raw.shape[0] != expected_frames
                or rod_raw.shape[2] != 3
            ):
                raise ValueError(
                    "rod_positions must have shape "
                    "[step + 1, marker, 3]"
                )
            candidate_mask = _mask_for(
                arrays,
                "rod_position_mask",
                rod_raw.shape,
            )
            if bool(np.any(candidate_mask)):
                rod_positions = _masked_values(
                    rod_raw,
                    candidate_mask,
                    "rod_positions",
                )
                rod_mask = candidate_mask
                rod_indices = _readonly(
                    arrays.get(
                        "rod_node_indices",
                        np.arange(rod_raw.shape[1]),
                    ),
                    dtype=np.uint32,
                )
                if rod_indices.shape != (rod_raw.shape[1],):
                    raise ValueError(
                        "rod_node_indices must have one entry per marker"
                    )

        contact_active: np.ndarray | None = None
        contact_mask: np.ndarray | None = None
        if "contact_active" in arrays:
            contact_raw = np.asarray(
                arrays["contact_active"],
                dtype=np.float32,
            ).reshape((-1,))
            if contact_raw.shape != (commands.shape[0],):
                raise ValueError(
                    "contact_active must have one value per command"
                )
            candidate_mask = _mask_for(
                arrays,
                "contact_mask",
                contact_raw.shape,
            )
            if bool(np.any(candidate_mask)):
                measured_contact = contact_raw[candidate_mask]
                if np.any(measured_contact < 0.0) or np.any(
                    measured_contact > 1.0
                ):
                    raise ValueError(
                        "contact_active measurements must lie in [0, 1]"
                    )
                contact_active = _masked_values(
                    contact_raw,
                    candidate_mask,
                    "contact_active",
                )
                contact_mask = candidate_mask

        recorded_period = (
            None
            if "control_period_seconds" not in arrays
            else float(
                np.asarray(arrays["control_period_seconds"]).reshape(())
            )
        )
        if (
            control_period_seconds is not None
            and recorded_period is not None
            and not math.isclose(
                control_period_seconds,
                recorded_period,
                rel_tol=1.0e-6,
                abs_tol=1.0e-9,
            )
        ):
            raise ValueError(
                "manifest and replay NPZ control periods disagree"
            )
        period = (
            control_period_seconds
            if control_period_seconds is not None
            else recorded_period
        )
        if period is None or not math.isfinite(period) or period <= 0.0:
            raise ValueError(
                "control_period_seconds must be supplied by the "
                "manifest or replay NPZ"
            )
        scales.validate()
        return cls(
            commands=commands,
            robot_q=robot_q,
            robot_q_mask=robot_q_mask,
            robot_v=robot_v,
            robot_v_mask=robot_v_mask,
            object_positions=object_positions,
            object_position_mask=object_mask,
            scene_body_indices=scene_indices,
            rod_positions=rod_positions,
            rod_position_mask=rod_mask,
            rod_node_indices=rod_indices,
            contact_active=contact_active,
            contact_mask=contact_mask,
            control_period_seconds=float(period),
            scales=scales,
        )


class ReplayAccumulator(NamedTuple):
    squared_error: mx.array
    sample_count: mx.array
    valid: mx.array
    first_physics_status: mx.array


class MLXPhysicalReplayEvaluator:
    """SMC evaluator that replays one physical episode in every candidate.

    A candidate-quantile upload occurs only at an SMC round boundary. Candidate
    worlds, commands, physics state, contacts, and residual reductions remain
    on the Apple GPU for the complete episode.
    """

    def __init__(
        self,
        world: MLXCompiledWorld,
        family,
        trace: PhysicalReplayTrace,
        *,
        seed: int = 1,
    ) -> None:
        self.world = world
        self.family = family
        self.trace = trace
        self.seed = int(seed)
        self.round_index = 0
        if (
            not world.contact_supported
            or world.nq != trace.robot_q.shape[1]
            or world.nv != trace.commands.shape[1]
        ):
            raise ValueError(
                "physical replay requires a topology-compatible, "
                "contact-capable world"
            )
        if trace.robot_v is not None and (
            trace.robot_v.shape[1] != world.nv
        ):
            raise ValueError("robot_v width does not match the world")
        if not math.isclose(
            float(world.control_timestep),
            trace.control_period_seconds,
            rel_tol=1.0e-6,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "physical replay must be resampled to the compiled "
                "world control period"
            )
        if trace.scene_body_indices.size and (
            int(np.max(trace.scene_body_indices))
            >= int(world.scene_body_count)
        ):
            raise ValueError(
                "a tracked object index exceeds the scene topology"
            )
        if trace.rod_node_indices.size and (
            int(np.max(trace.rod_node_indices))
            >= int(world.rod_node_count)
        ):
            raise ValueError(
                "a tracked rod marker exceeds the rod topology"
            )

        self._commands = mx.array(trace.commands)
        self._robot_q = mx.array(trace.robot_q)
        self._robot_q_mask = mx.array(trace.robot_q_mask)
        self._robot_v = (
            None if trace.robot_v is None else mx.array(trace.robot_v)
        )
        self._robot_v_mask = (
            None
            if trace.robot_v_mask is None
            else mx.array(trace.robot_v_mask)
        )
        self._object_positions = (
            None
            if trace.object_positions is None
            else mx.array(trace.object_positions)
        )
        self._object_mask = (
            None
            if trace.object_position_mask is None
            else mx.array(trace.object_position_mask)
        )
        self._scene_indices = mx.array(
            trace.scene_body_indices,
            dtype=mx.uint32,
        )
        self._rod_positions = (
            None
            if trace.rod_positions is None
            else mx.array(trace.rod_positions)
        )
        self._rod_mask = (
            None
            if trace.rod_position_mask is None
            else mx.array(trace.rod_position_mask)
        )
        self._rod_indices = mx.array(
            trace.rod_node_indices,
            dtype=mx.uint32,
        )
        self._contact_active = (
            None
            if trace.contact_active is None
            else mx.array(trace.contact_active)
        )
        self._contact_mask = (
            None
            if trace.contact_mask is None
            else mx.array(trace.contact_mask)
        )

        state_names = ["robot_q"]
        state_scales = [trace.scales.robot_q]
        if self._robot_v is not None:
            state_names.append("robot_v")
            state_scales.append(trace.scales.robot_v)
        if self._object_positions is not None:
            state_names.append("object_position")
            state_scales.append(trace.scales.object_position)
        if self._rod_positions is not None:
            state_names.append("rod_position")
            state_scales.append(trace.scales.rod_position)
        self._state_names = tuple(state_names)
        trajectory_names = [
            f"{name}_trajectory" for name in self._state_names
        ]
        trajectory_scales = list(state_scales)
        if self._contact_active is not None:
            trajectory_names.append("contact_timing")
            trajectory_scales.append(trace.scales.contact_timing)
        trajectory_names.append("physics_failure_rate")
        trajectory_scales.append(trace.scales.physics_failure)
        terminal_names = [
            f"{name}_terminal"
            for name in self._state_names
        ]
        self.residual_names = tuple(
            trajectory_names + terminal_names
        )
        self._trajectory_scales = mx.array(
            trajectory_scales,
            dtype=mx.float32,
        )
        self._state_scales = mx.array(
            state_scales,
            dtype=mx.float32,
        )

        maximum_latency = 0.0
        for feature in family.scenario_schema.features:
            if int(feature.target) != 13:
                continue
            parameters = feature.parameters
            distribution = int(feature.distribution)
            maximum_latency = max(
                maximum_latency,
                float(
                    parameters[0]
                    if distribution == 0
                    else parameters[3]
                    if distribution == 3
                    else parameters[1]
                ),
            )
        self.maximum_delay_steps = max(
            1,
            int(
                math.ceil(
                    maximum_latency / trace.control_period_seconds
                )
            ),
        )
        self._compiled_step = mx.compile(self._step)

    @property
    def residual_count(self) -> int:
        return len(self.residual_names)

    @staticmethod
    def _component(
        simulated: mx.array,
        observed: mx.array,
        mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        mask_float = mask.astype(mx.float32)
        difference = (
            simulated - observed.astype(mx.float32)
        ) * mask_float
        reduction_axes = tuple(range(1, difference.ndim))
        squared = mx.sum(
            difference * difference,
            axis=reduction_axes,
        )
        return squared, mx.sum(mask_float)

    def _state_components(
        self,
        state: SampledWorldFamilyState,
        frame: int | mx.array,
    ) -> tuple[list[mx.array], list[mx.array]]:
        squared: list[mx.array] = []
        counts: list[mx.array] = []
        value, count = self._component(
            state.world.q,
            self._robot_q[frame],
            self._robot_q_mask[frame],
        )
        squared.append(value)
        counts.append(count)
        if self._robot_v is not None:
            assert self._robot_v_mask is not None
            value, count = self._component(
                state.world.v,
                self._robot_v[frame],
                self._robot_v_mask[frame],
            )
            squared.append(value)
            counts.append(count)
        if self._object_positions is not None:
            assert self._object_mask is not None
            value, count = self._component(
                mx.take(
                    state.world.scene_bodies.position[:, :, :3],
                    self._scene_indices,
                    axis=1,
                ),
                self._object_positions[frame],
                self._object_mask[frame],
            )
            squared.append(value)
            counts.append(count)
        if self._rod_positions is not None:
            assert self._rod_mask is not None
            value, count = self._component(
                mx.take(
                    state.world.rods.position[:, :, :3],
                    self._rod_indices,
                    axis=1,
                ),
                self._rod_positions[frame],
                self._rod_mask[frame],
            )
            squared.append(value)
            counts.append(count)
        return squared, counts

    def _step(
        self,
        state: SampledWorldFamilyState,
        delay: ControllerDelayState,
        accumulator: ReplayAccumulator,
        command: mx.array,
        frame: mx.array,
        expected_contact: mx.array,
        expected_contact_mask: mx.array,
    ) -> tuple[
        SampledWorldFamilyState,
        ControllerDelayState,
        ReplayAccumulator,
    ]:
        environment_count = int(state.world.q.shape[0])
        actions = mx.broadcast_to(
            command[None, :],
            (environment_count, self.world.nv),
        )
        stepped = step_sampled_world_family(
            self.world,
            state,
            actions,
            delay,
            control_period_seconds=self.trace.control_period_seconds,
        )
        squared, counts = self._state_components(
            stepped.next_state,
            frame,
        )
        if self._contact_active is not None:
            simulated_contact = mx.any(
                stepped.physics.contacts.mask,
                axis=-1,
            ).astype(mx.float32)
            contact_difference = (
                simulated_contact - expected_contact
            ) * expected_contact_mask
            squared.append(contact_difference * contact_difference)
            counts.append(expected_contact_mask)
        squared.append(
            stepped.physics.physics_error.astype(mx.float32)
        )
        counts.append(mx.array(1.0, dtype=mx.float32))
        return (
            stepped.next_state,
            stepped.delay_state,
            ReplayAccumulator(
                squared_error=(
                    accumulator.squared_error
                    + mx.stack(squared, axis=-1)
                ),
                sample_count=(
                    accumulator.sample_count
                    + mx.stack(counts)
                ),
                valid=(
                    accumulator.valid
                    & ~stepped.physics.physics_error
                ),
                first_physics_status=mx.where(
                    accumulator.first_physics_status != 0,
                    accumulator.first_physics_status,
                    stepped.physics.status[:, 0],
                ),
            ),
        )

    def __call__(self, quantiles: mx.array) -> ReplayEvaluation:
        if (
            quantiles.ndim != 2
            or int(quantiles.shape[1])
            != len(self.family.scenario_schema.features)
        ):
            raise ValueError(
                "candidate quantiles do not match the ScenarioSchema"
            )
        candidate_count = int(quantiles.shape[0])
        if (
            candidate_count > int(self.family.layout.capacity)
            or candidate_count > int(self.world.environment_capacity)
        ):
            raise ValueError(
                "candidate replay exceeds the compiled world capacity"
            )

        # SMC calls this once per round. This is the only candidate transfer:
        # the native sampler uploads quantiles into private Metal buffers and
        # maps candidate i to environment i exactly.
        mx.eval(quantiles)
        candidate_values = np.ascontiguousarray(
            np.asarray(quantiles),
            dtype=np.float32,
        )
        fingerprint = int.from_bytes(
            hashlib.sha256(candidate_values.tobytes()).digest()[:8],
            "little",
        ) or 1
        self.family.configure_sampling(
            alignment_fingerprint=fingerprint,
            particle_quantiles=candidate_values,
            particle_weights=np.full(
                (candidate_count,),
                1.0 / float(candidate_count),
                dtype=np.float32,
            ),
            particle_residuals=np.zeros(
                (candidate_count,),
                dtype=np.float32,
            ),
            alignment_jitter=0.0,
        )
        self.family.sample(
            candidate_count,
            seed=self.seed,
            mode="replay",
            episode_counter=self.round_index * candidate_count,
        )
        self.round_index += 1
        state = sampled_state_from_world_family(
            self.world,
            self.family,
        )
        first_command = mx.broadcast_to(
            self._commands[0][None, None, :],
            (
                candidate_count,
                self.maximum_delay_steps + 1,
                self.world.nv,
            ),
        )
        delay = ControllerDelayState(first_command)

        initial_squared, initial_counts = self._state_components(
            state,
            0,
        )
        if self._contact_active is not None:
            initial_squared.append(
                mx.zeros(
                    (candidate_count,),
                    dtype=mx.float32,
                )
            )
            initial_counts.append(mx.array(0.0, dtype=mx.float32))
        initial_squared.append(
            mx.zeros(
                (candidate_count,),
                dtype=mx.float32,
            )
        )
        initial_counts.append(mx.array(0.0, dtype=mx.float32))
        accumulator = ReplayAccumulator(
            squared_error=mx.stack(initial_squared, axis=-1),
            sample_count=mx.stack(initial_counts),
            valid=mx.ones((candidate_count,), dtype=mx.bool_),
            first_physics_status=mx.zeros(
                (candidate_count,),
                dtype=mx.uint32,
            ),
        )

        zero_contact = mx.array(0.0, dtype=mx.float32)
        for step_index in range(self.trace.step_count):
            (
                state,
                delay,
                accumulator,
            ) = self._compiled_step(
                state,
                delay,
                accumulator,
                self._commands[step_index],
                mx.array(step_index + 1, dtype=mx.uint32),
                (
                    zero_contact
                    if self._contact_active is None
                    else self._contact_active[step_index]
                ),
                (
                    zero_contact
                    if self._contact_mask is None
                    else self._contact_mask[step_index].astype(
                        mx.float32
                    )
                ),
            )

        trajectory = mx.sqrt(
            accumulator.squared_error
            / mx.maximum(accumulator.sample_count, 1.0)
        ) / self._trajectory_scales
        terminal_squared, terminal_counts = self._state_components(
            state,
            self.trace.step_count,
        )
        terminal = mx.sqrt(
            mx.stack(terminal_squared, axis=-1)
            / mx.maximum(mx.stack(terminal_counts), 1.0)
        ) / self._state_scales
        residuals = mx.concatenate((trajectory, terminal), axis=-1)
        result = ReplayEvaluation(
            residuals=residuals,
            valid=accumulator.valid,
            physics_status=accumulator.first_physics_status,
        )
        mx.async_eval(*result)
        return result


__all__ = [
    "MLXPhysicalReplayEvaluator",
    "PhysicalReplayTrace",
    "ReplayResidualScales",
]
