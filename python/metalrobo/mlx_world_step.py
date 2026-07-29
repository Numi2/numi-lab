"""Explicit policy-independent contracts for one compiled world-graph step."""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from .mlx_r2s2r import CompactedEpisodeRecords
from .mlx_world import SampledWorldFamilyState, ScenarioState


class WorldStepRequest(NamedTuple):
    """All mutable inputs needed to encode one world transition.

    Scenario state and episode counters are carried explicitly so reset and
    rollout graphs remain reproducible under ``mx.compile``. The recurrent
    tensor may have zero columns for stateless policies.
    """

    action: mx.array
    reset_mask: mx.array
    episode_counter: mx.array
    scenario_state: ScenarioState
    recurrent_state: mx.array
    requested_sensor_mask: mx.array


class WorldStepResult(NamedTuple):
    """Causal result shared by policies, planners, trainers, and evaluators."""

    next_state: SampledWorldFamilyState
    deployable_observations: mx.array
    privileged_observations: mx.array
    task_phase: mx.array
    task_evidence: mx.array
    reward: mx.array
    terminated: mx.array
    termination: mx.array
    scenario_state: ScenarioState
    episode_counter: mx.array
    recurrent_state: mx.array
    physics_status: mx.array
    valid: mx.array
    completed: CompactedEpisodeRecords


__all__ = [
    "WorldStepRequest",
    "WorldStepResult",
]
