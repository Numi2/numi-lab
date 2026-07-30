"""Apple-native learned layers above MetalRobo's canonical tactile contract.

The physics runtime never calls this module. Simulation first publishes metric
geometry and solver evidence; these MLX models then provide optional
cross-sensor, object-centric, and predictive policy representations.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import numpy.typing as npt


TACTILE_LATENT_DIMENSIONS = 64
TACTILE_DYNAMICS_STATE_DIMENSIONS = 128

SUPPORTED_TACTILE_MODALITIES = frozenset(
    {
        "canonical",
        "rgb_camera",
        "capacitance",
        "magnetic",
        "marker_motion",
        "force_array",
    }
)


class TactileLatentState(NamedTuple):
    """Explicit recurrent sensor state carried by an MLX rollout."""

    hidden: mx.array


class TactileLatentObservation(NamedTuple):
    """One shared latent plus explicit source availability and confidence."""

    latent: mx.array
    presence: mx.array
    confidence: mx.array
    state: TactileLatentState


class TactileEncoderInput(NamedTuple):
    """Named, atlas-shaped input for a shared tactile encoder."""

    spatial: mx.array
    validity: mx.array
    summaries: mx.array
    presence: mx.array
    confidence: mx.array


class PhysicalTactileReconstruction(NamedTuple):
    """Training-only decoder outputs that ground the shared latent."""

    native_modality: mx.array
    penetration_depth_m: mx.array
    tangential_motion: mx.array
    wrench_si: mx.array
    center_of_pressure_local_m: mx.array
    friction_utilization: mx.array


class TactileDynamicsState(NamedTuple):
    """Explicit recurrent state for the compact tactile world model."""

    hidden: mx.array


class TactileDynamicsPrediction(NamedTuple):
    """Predicted tactile consequences of an action."""

    next_latent: mx.array
    contact_transition_logits: mx.array
    wrench_si: mx.array
    friction_utilization: mx.array
    contact_loss_logit: mx.array
    uncertainty: mx.array
    state: TactileDynamicsState


class TactilePolicyInput(NamedTuple):
    """Policy input that never hides whether touch was measured or predicted."""

    latent: mx.array
    presence: mx.array
    confidence: mx.array
    predicted: mx.array


class ObjectFieldState(NamedTuple):
    """Persistent estimator state, separate from the physics solver."""

    hidden: mx.array


class ObjectFieldPrediction(NamedTuple):
    """Contact field values evaluated at caller-provided object-local points."""

    contact_probability: mx.array
    penetration_depth_m: mx.array
    tangential_motion: mx.array
    uncertainty: mx.array
    state: ObjectFieldState


class ContactEventTargets(NamedTuple):
    """Metric event labels for dynamics training and attention gating."""

    onset: mx.array
    sustained: mx.array
    release: mx.array
    motion_change: mx.array


class TactileAlignmentLoss(NamedTuple):
    """Named components of shared cross-sensor representation training."""

    total: mx.array
    contrastive: mx.array
    metric_reconstruction: mx.array
    cross_reconstruction: mx.array
    self_reconstruction: mx.array


@dataclass(frozen=True, slots=True)
class TactileAtlasRange:
    """Static flat-sample range for one cooked rectangular atlas."""

    offset: int
    width: int
    height: int

    @property
    def sample_count(self) -> int:
        return self.width * self.height

    def validate(self) -> None:
        if self.offset < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("tactile atlas range is invalid")


@dataclass(frozen=True, slots=True)
class TactileAlignmentLossWeights:
    contrastive: float = 1.0
    metric_reconstruction: float = 1.0
    cross_reconstruction: float = 1.0
    self_reconstruction: float = 1.0

    def validate(self) -> None:
        values = (
            self.contrastive,
            self.metric_reconstruction,
            self.cross_reconstruction,
            self.self_reconstruction,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("tactile alignment loss weights are invalid")


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.LayerNorm(channels)

    def __call__(self, value: mx.array) -> mx.array:
        hidden = nn.silu(self.conv1(value))
        return nn.silu(self.norm(value + self.conv2(hidden)))


def _require_shape(
    value: mx.array,
    dimensions: int,
    label: str,
) -> None:
    if value.ndim != dimensions:
        raise ValueError(
            f"{label} must have {dimensions} dimensions, got {value.ndim}"
        )


class SharedTactileEncoder(nn.Module):
    """Small modality-specific stem with a shared 64-state temporal interface.

    Inputs are NHWC sensor atlases with shape
    ``[environment, sensor, height, width, channel]``. One instance represents
    one modality and channel contract. The recurrent state is explicit so the
    encoder remains suitable for ``mx.compile`` and deterministic replay.
    """

    def __init__(
        self,
        *,
        modality: str,
        input_channels: int,
        summary_channels: int,
        hidden_channels: int = 32,
        latent_dimensions: int = TACTILE_LATENT_DIMENSIONS,
    ) -> None:
        super().__init__()
        if modality not in SUPPORTED_TACTILE_MODALITIES:
            raise ValueError(f"unsupported tactile modality {modality!r}")
        if (
            input_channels <= 0
            or summary_channels < 0
            or hidden_channels <= 0
            or latent_dimensions != TACTILE_LATENT_DIMENSIONS
        ):
            raise ValueError("shared tactile encoder dimensions are invalid")
        self.modality = modality
        self.input_channels = input_channels
        self.summary_channels = summary_channels
        self.latent_dimensions = latent_dimensions
        self.stem = nn.Conv2d(
            input_channels,
            hidden_channels,
            5,
            padding=2,
        )
        self.spatial1 = _ResidualBlock(hidden_channels)
        self.spatial2 = _ResidualBlock(hidden_channels)
        self.summary = (
            nn.Linear(summary_channels, hidden_channels)
            if summary_channels
            else None
        )
        fused_channels = hidden_channels * (
            2 if summary_channels else 1
        )
        self.fuse = nn.Linear(fused_channels, latent_dimensions)
        self.recurrent = nn.GRU(
            latent_dimensions,
            latent_dimensions,
        )
        self.output_norm = nn.LayerNorm(latent_dimensions)

    def initial_state(
        self,
        environment_count: int,
        sensor_count: int,
    ) -> TactileLatentState:
        if environment_count <= 0 or sensor_count <= 0:
            raise ValueError("latent state dimensions must be positive")
        return TactileLatentState(
            mx.zeros(
                (
                    environment_count,
                    sensor_count,
                    self.latent_dimensions,
                ),
                dtype=mx.float32,
            )
        )

    def __call__(
        self,
        spatial: mx.array,
        validity: mx.array,
        summaries: mx.array,
        state: TactileLatentState,
        presence: mx.array,
        confidence: mx.array,
    ) -> TactileLatentObservation:
        _require_shape(spatial, 5, "spatial")
        _require_shape(validity, 4, "validity")
        _require_shape(summaries, 3, "summaries")
        _require_shape(state.hidden, 3, "state.hidden")
        _require_shape(presence, 2, "presence")
        _require_shape(confidence, 2, "confidence")
        environment_count, sensor_count, height, width, channels = (
            spatial.shape
        )
        if channels != self.input_channels:
            raise ValueError("spatial channel count does not match encoder")
        if validity.shape != (
            environment_count,
            sensor_count,
            height,
            width,
        ):
            raise ValueError("validity does not match the spatial atlas")
        if summaries.shape != (
            environment_count,
            sensor_count,
            self.summary_channels,
        ):
            raise ValueError("summary shape does not match encoder")
        latent_shape = (
            environment_count,
            sensor_count,
            self.latent_dimensions,
        )
        if state.hidden.shape != latent_shape:
            raise ValueError("latent state does not match the sensor batch")
        if presence.shape != latent_shape[:2]:
            raise ValueError("presence does not match the sensor batch")
        if confidence.shape != latent_shape[:2]:
            raise ValueError("confidence does not match the sensor batch")

        flattened_count = environment_count * sensor_count
        mask = validity.astype(spatial.dtype)[..., None]
        atlas = (spatial * mask).reshape(
            (
                flattened_count,
                height,
                width,
                self.input_channels,
            )
        )
        flat_mask = mask.reshape(
            (flattened_count, height, width, 1)
        )
        hidden = self.spatial2(
            self.spatial1(nn.silu(self.stem(atlas)))
        )
        numerator = mx.sum(hidden * flat_mask, axis=(1, 2))
        denominator = mx.maximum(
            mx.sum(flat_mask, axis=(1, 2)),
            mx.array(1.0, dtype=hidden.dtype),
        )
        pooled = numerator / denominator
        if self.summary is not None:
            summary_features = nn.silu(
                self.summary(
                    summaries.reshape(
                        (flattened_count, self.summary_channels)
                    )
                )
            )
            pooled = mx.concatenate(
                (pooled, summary_features),
                axis=-1,
            )
        recurrent_input = nn.silu(self.fuse(pooled))
        previous = state.hidden.reshape(
            (flattened_count, self.latent_dimensions)
        )
        candidate = self.recurrent(
            recurrent_input[:, None, :],
            previous,
        )[:, 0, :]
        candidate = self.output_norm(candidate).reshape(
            latent_shape
        )
        present = presence.astype(mx.bool_)[..., None]
        next_hidden = mx.where(present, candidate, state.hidden)
        visible_latent = mx.where(
            present,
            next_hidden,
            mx.zeros_like(next_hidden),
        )
        bounded_confidence = mx.where(
            presence.astype(mx.bool_),
            mx.clip(confidence, 0.0, 1.0),
            mx.zeros_like(confidence),
        )
        return TactileLatentObservation(
            latent=visible_latent,
            presence=presence.astype(mx.bool_),
            confidence=bounded_confidence,
            state=TactileLatentState(next_hidden),
        )


class TactileReconstructionDecoder(nn.Module):
    """Training-only native and physics reconstruction heads."""

    def __init__(
        self,
        *,
        atlas_height: int,
        atlas_width: int,
        native_channels: int,
        maximum_depth_m: float,
        hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        if (
            atlas_height <= 0
            or atlas_width <= 0
            or native_channels <= 0
            or maximum_depth_m <= 0.0
            or hidden_channels <= 0
        ):
            raise ValueError("tactile decoder dimensions are invalid")
        self.atlas_height = atlas_height
        self.atlas_width = atlas_width
        self.native_channels = native_channels
        self.maximum_depth_m = float(maximum_depth_m)
        spatial_values = (
            atlas_height * atlas_width * hidden_channels
        )
        self.spatial_seed = nn.Linear(
            TACTILE_LATENT_DIMENSIONS,
            spatial_values,
        )
        self.spatial = _ResidualBlock(hidden_channels)
        self.native_head = nn.Conv2d(
            hidden_channels,
            native_channels,
            1,
        )
        self.depth_head = nn.Conv2d(hidden_channels, 1, 1)
        self.motion_head = nn.Conv2d(hidden_channels, 4, 1)
        self.wrench_head = nn.Linear(TACTILE_LATENT_DIMENSIONS, 6)
        self.cop_head = nn.Linear(TACTILE_LATENT_DIMENSIONS, 3)
        self.friction_head = nn.Linear(
            TACTILE_LATENT_DIMENSIONS,
            2,
        )

    def __call__(
        self,
        latent: mx.array,
    ) -> PhysicalTactileReconstruction:
        _require_shape(latent, 3, "latent")
        if latent.shape[-1] != TACTILE_LATENT_DIMENSIONS:
            raise ValueError("latent must have 64 channels")
        environment_count, sensor_count, _ = latent.shape
        flattened = latent.reshape(
            (
                environment_count * sensor_count,
                TACTILE_LATENT_DIMENSIONS,
            )
        )
        hidden_channels = self.spatial.conv1.weight.shape[0]
        spatial = self.spatial(
            self.spatial_seed(flattened).reshape(
                (
                    environment_count * sensor_count,
                    self.atlas_height,
                    self.atlas_width,
                    hidden_channels,
                )
            )
        )
        atlas_prefix = (
            environment_count,
            sensor_count,
            self.atlas_height,
            self.atlas_width,
        )
        native = self.native_head(spatial).reshape(
            atlas_prefix + (self.native_channels,)
        )
        depth = (
            nn.sigmoid(self.depth_head(spatial))
            * self.maximum_depth_m
        ).reshape(atlas_prefix + (1,))
        motion = self.motion_head(spatial).reshape(
            atlas_prefix + (4,)
        )
        return PhysicalTactileReconstruction(
            native_modality=native,
            penetration_depth_m=depth,
            tangential_motion=motion,
            wrench_si=self.wrench_head(flattened).reshape(
                (environment_count, sensor_count, 6)
            ),
            center_of_pressure_local_m=self.cop_head(
                flattened
            ).reshape((environment_count, sensor_count, 3)),
            friction_utilization=nn.sigmoid(
                self.friction_head(flattened)
            ).reshape((environment_count, sensor_count, 2)),
        )


class TactileDynamicsModel(nn.Module):
    """Compact recurrent tactile action model with a 128-value state."""

    def __init__(
        self,
        *,
        sensor_count: int,
        robot_state_dimensions: int,
        action_dimensions: int,
    ) -> None:
        super().__init__()
        if (
            sensor_count <= 0
            or robot_state_dimensions < 0
            or action_dimensions <= 0
        ):
            raise ValueError("tactile dynamics dimensions are invalid")
        self.sensor_count = sensor_count
        self.robot_state_dimensions = robot_state_dimensions
        self.action_dimensions = action_dimensions
        input_dimensions = (
            sensor_count * TACTILE_LATENT_DIMENSIONS
            + sensor_count
            + robot_state_dimensions
            + action_dimensions
        )
        self.input = nn.Linear(
            input_dimensions,
            TACTILE_DYNAMICS_STATE_DIMENSIONS,
        )
        self.recurrent = nn.GRU(
            TACTILE_DYNAMICS_STATE_DIMENSIONS,
            TACTILE_DYNAMICS_STATE_DIMENSIONS,
        )
        self.next_latent = nn.Linear(
            TACTILE_DYNAMICS_STATE_DIMENSIONS,
            sensor_count * TACTILE_LATENT_DIMENSIONS,
        )
        self.transitions = nn.Linear(
            TACTILE_DYNAMICS_STATE_DIMENSIONS,
            sensor_count * 3,
        )
        self.wrench = nn.Linear(
            TACTILE_DYNAMICS_STATE_DIMENSIONS,
            sensor_count * 6,
        )
        self.friction = nn.Linear(
            TACTILE_DYNAMICS_STATE_DIMENSIONS,
            sensor_count * 2,
        )
        self.loss = nn.Linear(
            TACTILE_DYNAMICS_STATE_DIMENSIONS,
            sensor_count,
        )
        self.uncertainty = nn.Linear(
            TACTILE_DYNAMICS_STATE_DIMENSIONS,
            sensor_count,
        )

    def initial_state(
        self,
        environment_count: int,
    ) -> TactileDynamicsState:
        if environment_count <= 0:
            raise ValueError("environment_count must be positive")
        return TactileDynamicsState(
            mx.zeros(
                (
                    environment_count,
                    TACTILE_DYNAMICS_STATE_DIMENSIONS,
                ),
                dtype=mx.float32,
            )
        )

    def __call__(
        self,
        latent: mx.array,
        presence: mx.array,
        robot_state: mx.array,
        action: mx.array,
        state: TactileDynamicsState,
    ) -> TactileDynamicsPrediction:
        _require_shape(latent, 3, "latent")
        _require_shape(presence, 2, "presence")
        _require_shape(robot_state, 2, "robot_state")
        _require_shape(action, 2, "action")
        _require_shape(state.hidden, 2, "state.hidden")
        environment_count = latent.shape[0]
        if latent.shape[1:] != (
            self.sensor_count,
            TACTILE_LATENT_DIMENSIONS,
        ):
            raise ValueError("latent shape does not match dynamics model")
        if presence.shape != (
            environment_count,
            self.sensor_count,
        ):
            raise ValueError("presence shape does not match dynamics model")
        if robot_state.shape != (
            environment_count,
            self.robot_state_dimensions,
        ):
            raise ValueError(
                "robot_state shape does not match dynamics model"
            )
        if action.shape != (
            environment_count,
            self.action_dimensions,
        ):
            raise ValueError("action shape does not match dynamics model")
        if state.hidden.shape != (
            environment_count,
            TACTILE_DYNAMICS_STATE_DIMENSIONS,
        ):
            raise ValueError("state shape does not match dynamics model")
        model_input = mx.concatenate(
            (
                latent.reshape(
                    (
                        environment_count,
                        self.sensor_count
                        * TACTILE_LATENT_DIMENSIONS,
                    )
                ),
                presence.astype(latent.dtype),
                robot_state,
                action,
            ),
            axis=-1,
        )
        next_state = self.recurrent(
            nn.silu(self.input(model_input))[:, None, :],
            state.hidden,
        )[:, 0, :]
        return TactileDynamicsPrediction(
            next_latent=self.next_latent(next_state).reshape(
                (
                    environment_count,
                    self.sensor_count,
                    TACTILE_LATENT_DIMENSIONS,
                )
            ),
            contact_transition_logits=self.transitions(
                next_state
            ).reshape((environment_count, self.sensor_count, 3)),
            wrench_si=self.wrench(next_state).reshape(
                (environment_count, self.sensor_count, 6)
            ),
            friction_utilization=nn.sigmoid(
                self.friction(next_state)
            ).reshape((environment_count, self.sensor_count, 2)),
            contact_loss_logit=self.loss(next_state).reshape(
                (environment_count, self.sensor_count)
            ),
            uncertainty=nn.softplus(
                self.uncertainty(next_state)
            ).reshape((environment_count, self.sensor_count)),
            state=TactileDynamicsState(next_state),
        )


class ObjectContactFieldEstimator(nn.Module):
    """Persistent object-field decoder evaluated at sparse local queries.

    Exact simulator poses and contact labels supervise this estimator. They are
    not consumed by a deployed policy unless deliberately exposed as
    privileged critic input.
    """

    def __init__(
        self,
        *,
        sensor_count: int,
        visual_feature_dimensions: int,
        semantic_feature_dimensions: int = 0,
        maximum_depth_m: float,
    ) -> None:
        super().__init__()
        if (
            sensor_count <= 0
            or visual_feature_dimensions < 0
            or semantic_feature_dimensions < 0
            or maximum_depth_m <= 0.0
        ):
            raise ValueError("object field dimensions are invalid")
        self.sensor_count = sensor_count
        self.visual_feature_dimensions = visual_feature_dimensions
        self.semantic_feature_dimensions = semantic_feature_dimensions
        self.maximum_depth_m = float(maximum_depth_m)
        global_dimensions = (
            TACTILE_LATENT_DIMENSIONS
            + visual_feature_dimensions
            + sensor_count
        )
        self.global_input = nn.Linear(
            global_dimensions,
            TACTILE_DYNAMICS_STATE_DIMENSIONS,
        )
        self.recurrent = nn.GRU(
            TACTILE_DYNAMICS_STATE_DIMENSIONS,
            TACTILE_DYNAMICS_STATE_DIMENSIONS,
        )
        query_dimensions = (
            TACTILE_DYNAMICS_STATE_DIMENSIONS
            + 3
            + semantic_feature_dimensions
        )
        self.query1 = nn.Linear(query_dimensions, 128)
        self.query2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 7)

    def initial_state(
        self,
        environment_count: int,
    ) -> ObjectFieldState:
        if environment_count <= 0:
            raise ValueError("environment_count must be positive")
        return ObjectFieldState(
            mx.zeros(
                (
                    environment_count,
                    TACTILE_DYNAMICS_STATE_DIMENSIONS,
                ),
                dtype=mx.float32,
            )
        )

    def __call__(
        self,
        tactile_latent: mx.array,
        tactile_presence: mx.array,
        visual_features: mx.array,
        query_points_object_local: mx.array,
        state: ObjectFieldState,
        semantic_features: mx.array | None = None,
    ) -> ObjectFieldPrediction:
        _require_shape(tactile_latent, 3, "tactile_latent")
        _require_shape(tactile_presence, 2, "tactile_presence")
        _require_shape(visual_features, 2, "visual_features")
        _require_shape(
            query_points_object_local,
            3,
            "query_points_object_local",
        )
        environment_count, query_count, point_dimensions = (
            query_points_object_local.shape
        )
        if point_dimensions != 3:
            raise ValueError("object-local query points must be xyz")
        if tactile_latent.shape != (
            environment_count,
            self.sensor_count,
            TACTILE_LATENT_DIMENSIONS,
        ):
            raise ValueError("tactile latent shape does not match field model")
        if tactile_presence.shape != (
            environment_count,
            self.sensor_count,
        ):
            raise ValueError("presence shape does not match field model")
        if visual_features.shape != (
            environment_count,
            self.visual_feature_dimensions,
        ):
            raise ValueError("visual feature shape does not match field model")
        if self.semantic_feature_dimensions:
            if semantic_features is None:
                raise ValueError(
                    "authored semantic features are required by this model"
                )
            if semantic_features.shape != (
                environment_count,
                query_count,
                self.semantic_feature_dimensions,
            ):
                raise ValueError(
                    "semantic features do not match object queries"
                )
        elif semantic_features is not None:
            raise ValueError(
                "semantic features were supplied to a geometry-only model"
            )
        weights = tactile_presence.astype(tactile_latent.dtype)[..., None]
        pooled = mx.sum(tactile_latent * weights, axis=1) / mx.maximum(
            mx.sum(weights, axis=1),
            mx.array(1.0, dtype=tactile_latent.dtype),
        )
        global_input = mx.concatenate(
            (
                pooled,
                visual_features,
                tactile_presence.astype(tactile_latent.dtype),
            ),
            axis=-1,
        )
        next_state = self.recurrent(
            nn.silu(self.global_input(global_input))[:, None, :],
            state.hidden,
        )[:, 0, :]
        broadcast_state = mx.broadcast_to(
            next_state[:, None, :],
            (
                environment_count,
                query_count,
                TACTILE_DYNAMICS_STATE_DIMENSIONS,
            ),
        )
        query_parts = [broadcast_state, query_points_object_local]
        if semantic_features is not None:
            query_parts.append(semantic_features)
        hidden = nn.silu(
            self.query1(mx.concatenate(query_parts, axis=-1))
        )
        output = self.output(nn.silu(self.query2(hidden)))
        return ObjectFieldPrediction(
            contact_probability=nn.sigmoid(output[..., 0]),
            penetration_depth_m=(
                nn.sigmoid(output[..., 1]) * self.maximum_depth_m
            ),
            tangential_motion=output[..., 2:6],
            uncertainty=nn.softplus(output[..., 6]),
            state=ObjectFieldState(next_state),
        )


def paired_contrastive_loss(
    first: mx.array,
    second: mx.array,
    *,
    temperature: float = 0.1,
) -> mx.array:
    """Symmetric paired-stimulus contrastive loss for aligned modalities."""

    _require_shape(first, 2, "first")
    _require_shape(second, 2, "second")
    if (
        first.shape != second.shape
        or first.shape[-1] != TACTILE_LATENT_DIMENSIONS
    ):
        raise ValueError("paired latents must have matching [batch,64] shape")
    if first.shape[0] <= 0 or temperature <= 0.0:
        raise ValueError("contrastive batch and temperature must be positive")
    first_normalized = first / mx.sqrt(
        mx.sum(first * first, axis=-1, keepdims=True) + 1.0e-8
    )
    second_normalized = second / mx.sqrt(
        mx.sum(second * second, axis=-1, keepdims=True) + 1.0e-8
    )
    logits = (
        first_normalized @ mx.transpose(second_normalized)
    ) / temperature
    positive = mx.diag(logits)
    first_loss = mx.mean(mx.logsumexp(logits, axis=1) - positive)
    second_loss = mx.mean(
        mx.logsumexp(logits, axis=0) - positive
    )
    return 0.5 * (first_loss + second_loss)


def canonical_tactile_encoder_input(
    observation: Any,
    atlas_ranges: Sequence[TactileAtlasRange],
) -> TactileEncoderInput:
    """Pack named canonical MLX outputs into seven physical atlas channels.

    Channels are depth, depth velocity, two displacement components, two
    surface-velocity components, and an explicit contact indicator. Atlas
    validity stays a separate mask. All sensors in one encoder group must have
    equal dimensions; heterogeneous atlases use separate encoder instances.
    """

    if not atlas_ranges:
        raise ValueError("at least one tactile atlas range is required")
    for atlas in atlas_ranges:
        atlas.validate()
    first = atlas_ranges[0]
    if any(
        atlas.width != first.width or atlas.height != first.height
        for atlas in atlas_ranges
    ):
        raise ValueError(
            "one shared encoder group requires equal atlas dimensions"
        )
    environment_count = int(observation.penetration_depth_m.shape[0])
    if (
        observation.penetration_depth_m.ndim != 2
        or observation.depth_velocity_m_s.shape
        != observation.penetration_depth_m.shape
        or observation.tangential_motion.shape
        != observation.penetration_depth_m.shape + (4,)
        or observation.validity.shape
        != observation.penetration_depth_m.shape
    ):
        raise ValueError("canonical tactile dense arrays are inconsistent")
    spatial_values: list[mx.array] = []
    validity_values: list[mx.array] = []
    for atlas in atlas_ranges:
        end = atlas.offset + atlas.sample_count
        if end > int(observation.penetration_depth_m.shape[1]):
            raise ValueError("tactile atlas range exceeds dense observation")
        depth = observation.penetration_depth_m[
            :, atlas.offset:end
        ].reshape(
            (environment_count, atlas.height, atlas.width, 1)
        )
        depth_velocity = observation.depth_velocity_m_s[
            :, atlas.offset:end
        ].reshape(
            (environment_count, atlas.height, atlas.width, 1)
        )
        motion = observation.tangential_motion[
            :, atlas.offset:end, :
        ].reshape(
            (environment_count, atlas.height, atlas.width, 4)
        )
        validity_bits = observation.validity[:, atlas.offset:end]
        sample_valid = ((validity_bits & 1) != 0).reshape(
            (environment_count, atlas.height, atlas.width)
        )
        contact = ((validity_bits & 2) != 0).astype(
            depth.dtype
        ).reshape(
            (environment_count, atlas.height, atlas.width, 1)
        )
        spatial_values.append(
            mx.concatenate(
                (depth, depth_velocity, motion, contact),
                axis=-1,
            )
        )
        validity_values.append(sample_valid)
    spatial = mx.stack(spatial_values, axis=1)
    validity = mx.stack(validity_values, axis=1)
    summary = observation.summary
    summaries = mx.concatenate(
        (
            summary.net_force_and_contact_area[
                :, : len(atlas_ranges), :
            ],
            summary.net_torque_and_maximum_depth[
                :, : len(atlas_ranges), :
            ],
            summary.center_of_pressure_local_and_force_weight[
                :, : len(atlas_ranges), :
            ],
            summary.tangential_motion_and_friction[
                :, : len(atlas_ranges), :
            ],
        ),
        axis=-1,
    )
    expected_summary_shape = (
        environment_count,
        len(atlas_ranges),
        16,
    )
    if summaries.shape != expected_summary_shape:
        raise ValueError("tactile summaries do not match atlas ranges")
    availability_shape = (
        environment_count,
        len(atlas_ranges),
    )
    return TactileEncoderInput(
        spatial=spatial,
        validity=validity,
        summaries=summaries,
        presence=mx.ones(availability_shape, dtype=mx.bool_),
        confidence=mx.ones(availability_shape, dtype=mx.float32),
    )


def canonical_metric_tactile_stem(
    observation: Any,
    atlas_ranges: Sequence[TactileAtlasRange],
) -> TactilePolicyInput:
    """Publish one deterministic 64-value metric baseline per sensor.

    This uses the canonical seven-channel packing shared with learned
    encoders. Forty-eight coarse spatial values retain depth, depth velocity,
    and all four bounded rigid-body tangent-motion components; the sixteen
    physical summary values complete the policy vector. It occupies the shared
    latent boundary without claiming learned cross-sensor alignment. The
    metric path pools those named arrays directly so captured RL graphs do not
    materialize an otherwise disposable full seven-channel atlas tensor.
    """

    if not atlas_ranges:
        raise ValueError("at least one tactile atlas range is required")
    for atlas in atlas_ranges:
        atlas.validate()
    if any(
        atlas.width != 32 or atlas.height != 32
        for atlas in atlas_ranges
    ):
        raise ValueError(
            "canonical metric tactile stem requires 32x32 atlases"
        )
    depth = observation.penetration_depth_m
    if (
        depth.ndim != 2
        or observation.depth_velocity_m_s.shape != depth.shape
        or observation.tangential_motion.shape != depth.shape + (4,)
        or observation.validity.shape != depth.shape
    ):
        raise ValueError("canonical tactile dense arrays are inconsistent")
    environment_count = int(depth.shape[0])
    pooled_sensors: list[mx.array] = []
    for atlas in atlas_ranges:
        end = atlas.offset + atlas.sample_count
        if end > int(depth.shape[1]):
            raise ValueError(
                "tactile atlas range exceeds dense observation"
            )
        valid = (
            (
                observation.validity[:, atlas.offset:end]
                & 1
            )
            != 0
        ).astype(depth.dtype)
        bin_counts = mx.maximum(
            mx.sum(
                valid.reshape(
                    (environment_count, 2, 16, 4, 8, 1)
                ),
                axis=(2, 4),
            ),
            mx.array(1.0, dtype=depth.dtype),
        )
        depth_channels = mx.stack(
            (
                depth[:, atlas.offset:end],
                observation.depth_velocity_m_s[
                    :, atlas.offset:end
                ],
            ),
            axis=-1,
        )
        pooled_depth = mx.sum(
            (
                depth_channels * valid[..., None]
            ).reshape(
                (environment_count, 2, 16, 4, 8, 2)
            ),
            axis=(2, 4),
        ) / bin_counts
        pooled_motion = mx.sum(
            (
                observation.tangential_motion[
                    :, atlas.offset:end, :
                ]
                * valid[..., None]
            ).reshape(
                (environment_count, 2, 16, 4, 8, 4)
            ),
            axis=(2, 4),
        ) / bin_counts
        pooled_sensors.append(
            mx.concatenate(
                (pooled_depth, pooled_motion),
                axis=-1,
            ).reshape((environment_count, 48))
        )
    spatial = mx.stack(pooled_sensors, axis=1)
    summary = observation.summary
    summaries = mx.concatenate(
        (
            summary.net_force_and_contact_area[
                :, : len(atlas_ranges), :
            ],
            summary.net_torque_and_maximum_depth[
                :, : len(atlas_ranges), :
            ],
            summary.center_of_pressure_local_and_force_weight[
                :, : len(atlas_ranges), :
            ],
            summary.tangential_motion_and_friction[
                :, : len(atlas_ranges), :
            ],
        ),
        axis=-1,
    )
    if summaries.shape != (
        environment_count,
        len(atlas_ranges),
        16,
    ):
        raise ValueError("tactile summaries do not match atlas ranges")
    latent = mx.concatenate(
        (spatial, summaries),
        axis=-1,
    )
    availability_shape = (
        environment_count,
        len(atlas_ranges),
    )
    presence = mx.ones(availability_shape, dtype=mx.bool_)
    return TactilePolicyInput(
        latent=latent,
        presence=presence,
        confidence=mx.ones(
            availability_shape,
            dtype=mx.float32,
        ),
        predicted=mx.zeros_like(presence),
    )


def canonical_metric_tactile_policy_observation(
    observation: Any,
    atlas_ranges: Sequence[TactileAtlasRange],
) -> mx.array:
    """Flatten the shared metric stem with presence and confidence."""

    encoded = canonical_metric_tactile_stem(
        observation,
        atlas_ranges,
    )
    environment_count = int(encoded.latent.shape[0])
    return mx.concatenate(
        (
            encoded.latent.reshape((environment_count, -1)),
            encoded.presence.astype(mx.float32),
            encoded.confidence,
        ),
        axis=-1,
    )


def masked_metric_reconstruction_loss(
    prediction: mx.array,
    target: mx.array,
    validity: mx.array,
    *,
    channel_weights: mx.array | None = None,
) -> mx.array:
    """Masked weighted MSE; callers choose SI-aware channel normalization."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if validity.shape != prediction.shape[:-1]:
        raise ValueError("validity must match all non-channel dimensions")
    weights = validity.astype(prediction.dtype)[..., None]
    error = (prediction - target) ** 2
    if channel_weights is not None:
        if channel_weights.ndim != 1 or (
            channel_weights.shape[0] != prediction.shape[-1]
        ):
            raise ValueError("channel_weights must match the channel count")
        error = error * channel_weights
    denominator = mx.maximum(
        mx.sum(weights) * prediction.shape[-1],
        mx.array(1.0, dtype=prediction.dtype),
    )
    return mx.sum(error * weights) / denominator


ReconstructionPair = tuple[mx.array, mx.array, mx.array]


def shared_tactile_alignment_loss(
    simulation_latent: mx.array,
    real_latent: mx.array,
    *,
    metric_pairs: Sequence[ReconstructionPair],
    cross_pairs: Sequence[ReconstructionPair],
    self_pairs: Sequence[ReconstructionPair],
    weights: TactileAlignmentLossWeights = (
        TactileAlignmentLossWeights()
    ),
    temperature: float = 0.1,
) -> TactileAlignmentLoss:
    """Combine the four physically grounded TactSpace-style objectives.

    Every reconstruction pair is ``(prediction, target, validity)``. Callers
    normalize heterogeneous SI channels deliberately before passing them; the
    platform does not silently mix metre, newton, and torque scales.
    """

    weights.validate()
    contrastive = paired_contrastive_loss(
        simulation_latent,
        real_latent,
        temperature=temperature,
    )
    zero = mx.array(0.0, dtype=simulation_latent.dtype)

    def average(pairs: Sequence[ReconstructionPair]) -> mx.array:
        if not pairs:
            return zero
        losses = [
            masked_metric_reconstruction_loss(
                prediction,
                target,
                validity,
            )
            for prediction, target, validity in pairs
        ]
        total_loss = losses[0]
        for loss in losses[1:]:
            total_loss = total_loss + loss
        return total_loss / len(losses)

    metric = average(metric_pairs)
    cross = average(cross_pairs)
    self_reconstruction = average(self_pairs)
    total = (
        weights.contrastive * contrastive
        + weights.metric_reconstruction * metric
        + weights.cross_reconstruction * cross
        + weights.self_reconstruction * self_reconstruction
    )
    return TactileAlignmentLoss(
        total=total,
        contrastive=contrastive,
        metric_reconstruction=metric,
        cross_reconstruction=cross,
        self_reconstruction=self_reconstruction,
    )


def contact_event_targets(
    previous_depth_m: mx.array,
    current_depth_m: mx.array,
    previous_tangential_motion: mx.array,
    current_tangential_motion: mx.array,
    *,
    contact_threshold_m: float = 1.0e-6,
    motion_change_threshold: float = 1.0e-5,
) -> ContactEventTargets:
    """Create metric contact-transition labels outside the physics solver."""

    if previous_depth_m.shape != current_depth_m.shape:
        raise ValueError("depth histories must have matching shapes")
    if (
        previous_tangential_motion.shape
        != current_tangential_motion.shape
        or previous_tangential_motion.shape[:-1]
        != previous_depth_m.shape
        or previous_tangential_motion.shape[-1] != 4
    ):
        raise ValueError("motion histories must match depth samples")
    if contact_threshold_m < 0.0 or motion_change_threshold < 0.0:
        raise ValueError("event thresholds must be non-negative")
    previous_active = previous_depth_m > contact_threshold_m
    current_active = current_depth_m > contact_threshold_m
    motion_delta = current_tangential_motion - previous_tangential_motion
    changed = mx.sqrt(mx.sum(motion_delta * motion_delta, axis=-1))
    return ContactEventTargets(
        onset=(~previous_active) & current_active,
        sustained=previous_active & current_active,
        release=previous_active & (~current_active),
        motion_change=current_active
        & (changed > motion_change_threshold),
    )


def measured_policy_input(
    observation: TactileLatentObservation,
) -> TactilePolicyInput:
    """Publish measured touch without any implicit substitution."""

    return TactilePolicyInput(
        latent=observation.latent,
        presence=observation.presence,
        confidence=observation.confidence,
        predicted=mx.zeros_like(observation.presence).astype(mx.bool_),
    )


def explicitly_impute_missing_touch(
    measured: TactileLatentObservation,
    predicted_latent: mx.array,
    predicted_uncertainty: mx.array,
    *,
    enabled: bool,
) -> TactilePolicyInput:
    """Opt-in missing-sensor imputation with source and uncertainty exposed."""

    if not enabled:
        raise ValueError(
            "tactile imputation is disabled; missing touch cannot be hidden"
        )
    if predicted_latent.shape != measured.latent.shape:
        raise ValueError("predicted latent does not match measured latent")
    if predicted_uncertainty.shape != measured.presence.shape:
        raise ValueError("predicted uncertainty does not match sensor batch")
    missing = ~measured.presence.astype(mx.bool_)
    latent = mx.where(
        missing[..., None],
        predicted_latent,
        measured.latent,
    )
    predicted_confidence = mx.exp(
        -mx.maximum(predicted_uncertainty, 0.0)
    )
    confidence = mx.where(
        missing,
        predicted_confidence,
        measured.confidence,
    )
    return TactilePolicyInput(
        latent=latent,
        presence=mx.ones_like(measured.presence).astype(mx.bool_),
        confidence=mx.clip(confidence, 0.0, 1.0),
        predicted=missing,
    )


@dataclass(frozen=True, slots=True)
class TactileEncoderAsset:
    """Artifact-bound encoder identity for policy and hardware deployment."""

    sensor_key: str
    modality: str
    model_uri: str
    input_channels: int
    observation_fingerprint: str
    latent_dimensions: int = TACTILE_LATENT_DIMENSIONS
    recurrent_state_dimensions: int = TACTILE_LATENT_DIMENSIONS
    coreml_model_uri: str | None = None
    coreml_parity_max_abs_error: float | None = None

    def validate(self) -> None:
        if not self.sensor_key:
            raise ValueError("encoder sensor_key must not be empty")
        if self.modality not in SUPPORTED_TACTILE_MODALITIES:
            raise ValueError(
                f"unsupported tactile modality {self.modality!r}"
            )
        if not self.model_uri:
            raise ValueError("encoder model_uri must not be empty")
        if self.input_channels <= 0:
            raise ValueError("encoder input_channels must be positive")
        if (
            self.latent_dimensions != TACTILE_LATENT_DIMENSIONS
            or self.recurrent_state_dimensions
            != TACTILE_LATENT_DIMENSIONS
        ):
            raise ValueError("encoder must use the shared 64-state contract")
        if re.fullmatch(
            r"0x[0-9a-f]{16}",
            self.observation_fingerprint,
        ) is None:
            raise ValueError(
                "encoder must identify one canonical observation contract"
            )
        if self.coreml_parity_max_abs_error is not None and (
            not np.isfinite(self.coreml_parity_max_abs_error)
            or self.coreml_parity_max_abs_error < 0.0
            or not self.coreml_model_uri
        ):
            raise ValueError("Core ML parity metadata is invalid")

    def to_json(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TactileEncoderAsset":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        asset = cls(**payload)
        asset.validate()
        return asset


class TactileEncoderRegistry:
    """Exact sensor/modality lookup with no simulated or imagined fallback."""

    def __init__(
        self,
        assets: Sequence[TactileEncoderAsset] = (),
    ) -> None:
        self._assets: dict[tuple[str, str], TactileEncoderAsset] = {}
        for asset in assets:
            self.add(asset)

    def add(self, asset: TactileEncoderAsset) -> None:
        asset.validate()
        key = (asset.sensor_key, asset.modality)
        if key in self._assets:
            raise ValueError(
                f"duplicate tactile encoder for {key[0]!r}/{key[1]!r}"
            )
        self._assets[key] = asset

    def require(
        self,
        *,
        sensor_key: str,
        modality: str,
        observation_fingerprint: str,
        require_coreml: bool = False,
    ) -> TactileEncoderAsset:
        asset = self._assets.get((sensor_key, modality))
        if asset is None:
            raise LookupError(
                "no matching tactile encoder asset for "
                f"{sensor_key!r}/{modality!r}"
            )
        if asset.observation_fingerprint != observation_fingerprint:
            raise ValueError(
                "tactile encoder observation contract does not match"
            )
        if require_coreml and (
            not asset.coreml_model_uri
            or asset.coreml_parity_max_abs_error is None
        ):
            raise ValueError(
                "real sensor requires an exported and parity-validated "
                "stateful Core ML encoder"
            )
        return asset


@dataclass(frozen=True, slots=True)
class StatefulParityReport:
    frame_count: int
    maximum_output_error: float
    maximum_state_error: float


ArrayPairStep = Callable[
    [npt.NDArray[np.float32], npt.NDArray[np.float32]],
    tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
]


def check_stateful_encoder_parity(
    mlx_step: ArrayPairStep,
    coreml_step: ArrayPairStep,
    frames: Sequence[npt.NDArray[np.float32]],
    initial_state: npt.NDArray[np.float32],
    *,
    absolute_tolerance: float = 1.0e-4,
) -> StatefulParityReport:
    """Run sequence-level MLX/Core ML parity through caller-owned adapters.

    The Core ML adapter is deliberately injected so ``coremltools`` is not a
    simulation-runtime dependency.
    """

    if not frames:
        raise ValueError("parity sequence must contain at least one frame")
    if absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be non-negative")
    mlx_state = np.asarray(initial_state, dtype=np.float32).copy()
    coreml_state = mlx_state.copy()
    maximum_output_error = 0.0
    maximum_state_error = 0.0
    for frame in frames:
        frame_value = np.asarray(frame, dtype=np.float32)
        mlx_output, mlx_state = mlx_step(frame_value, mlx_state)
        coreml_output, coreml_state = coreml_step(
            frame_value,
            coreml_state,
        )
        output_error = float(
            np.max(
                np.abs(
                    np.asarray(mlx_output, dtype=np.float32)
                    - np.asarray(coreml_output, dtype=np.float32)
                )
            )
        )
        state_error = float(
            np.max(
                np.abs(
                    np.asarray(mlx_state, dtype=np.float32)
                    - np.asarray(coreml_state, dtype=np.float32)
                )
            )
        )
        maximum_output_error = max(
            maximum_output_error,
            output_error,
        )
        maximum_state_error = max(
            maximum_state_error,
            state_error,
        )
    if max(maximum_output_error, maximum_state_error) > absolute_tolerance:
        raise ValueError(
            "stateful MLX/Core ML parity exceeded tolerance: "
            f"output={maximum_output_error:.8g}, "
            f"state={maximum_state_error:.8g}, "
            f"tolerance={absolute_tolerance:.8g}"
        )
    return StatefulParityReport(
        frame_count=len(frames),
        maximum_output_error=maximum_output_error,
        maximum_state_error=maximum_state_error,
    )


def flatten_named_tactile_summaries(
    summaries: Mapping[str, mx.array],
) -> mx.array:
    """Flatten named physical summaries without exposing magic indices."""

    required = (
        "net_force_and_contact_area",
        "net_torque_and_maximum_depth",
        "center_of_pressure_local_and_force_weight",
        "tangential_motion_and_friction",
    )
    missing = [name for name in required if name not in summaries]
    if missing:
        raise ValueError(
            "missing tactile summaries: " + ", ".join(missing)
        )
    values = [summaries[name] for name in required]
    if any(value.ndim != 3 or value.shape[-1] != 4 for value in values):
        raise ValueError("named tactile summaries must be [env,sensor,4]")
    prefix = values[0].shape[:2]
    if any(value.shape[:2] != prefix for value in values):
        raise ValueError("named tactile summaries must share a batch")
    return mx.concatenate(values, axis=-1)


__all__ = [
    "ContactEventTargets",
    "ObjectContactFieldEstimator",
    "ObjectFieldPrediction",
    "ObjectFieldState",
    "PhysicalTactileReconstruction",
    "SUPPORTED_TACTILE_MODALITIES",
    "SharedTactileEncoder",
    "StatefulParityReport",
    "TACTILE_DYNAMICS_STATE_DIMENSIONS",
    "TACTILE_LATENT_DIMENSIONS",
    "TactileAlignmentLoss",
    "TactileAlignmentLossWeights",
    "TactileAtlasRange",
    "TactileDynamicsModel",
    "TactileDynamicsPrediction",
    "TactileDynamicsState",
    "TactileEncoderAsset",
    "TactileEncoderInput",
    "TactileEncoderRegistry",
    "TactileLatentObservation",
    "TactileLatentState",
    "TactilePolicyInput",
    "TactileReconstructionDecoder",
    "check_stateful_encoder_parity",
    "canonical_tactile_encoder_input",
    "canonical_metric_tactile_policy_observation",
    "canonical_metric_tactile_stem",
    "contact_event_targets",
    "explicitly_impute_missing_touch",
    "flatten_named_tactile_summaries",
    "masked_metric_reconstruction_loss",
    "measured_policy_input",
    "paired_contrastive_loss",
    "shared_tactile_alignment_loss",
]
