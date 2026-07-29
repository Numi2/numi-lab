"""MLX-native alignment and policy-specific failure-region learning.

Only four SMC round boundaries and final artifact publication are allowed to
materialize scalar metadata on the host. Replay residuals, ensemble training,
65,536-scenario scoring, and region clustering remain MLX array programs on
the Apple GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


class AlignmentPopulationArrays(NamedTuple):
    quantiles: mx.array
    weights: mx.array
    replay_residuals: mx.array


class EpisodeAccumulator(NamedTuple):
    """Fixed-shape episode evidence retained on the Apple GPU."""

    episode_return: mx.array
    duration_seconds: mx.array
    minimum_visibility: mx.array
    integrated_contact_load: mx.array
    peak_contact_load: mx.array
    valid_contact_count: mx.array
    step_count: mx.array


class CompactedEpisodeRecords(NamedTuple):
    """Completed records sorted to the front without dynamic host shapes."""

    outcome_words: mx.array
    task_values: mx.array
    interaction_values: mx.array
    scenario_headers: mx.array
    scenario_values: mx.array
    scenario_identities: mx.array
    valid_mask: mx.array
    valid_count: mx.array


@dataclass(frozen=True, slots=True)
class SMCConfig:
    rounds: int = 4
    huber_delta: float = 1.0
    minimum_effective_sample_fraction: float = 0.5
    jitter_scale: float = 0.05
    seed: int = 1


ReplayEvaluator = Callable[[mx.array], mx.array]


def initial_episode_accumulator(
    environment_count: int,
) -> EpisodeAccumulator:
    if environment_count <= 0:
        raise ValueError("environment_count must be positive")
    zeros = mx.zeros((environment_count,), dtype=mx.float32)
    return EpisodeAccumulator(
        episode_return=zeros,
        duration_seconds=zeros,
        minimum_visibility=mx.ones(
            (environment_count,),
            dtype=mx.float32,
        ),
        integrated_contact_load=zeros,
        peak_contact_load=zeros,
        valid_contact_count=mx.zeros(
            (environment_count,),
            dtype=mx.uint32,
        ),
        step_count=mx.zeros(
            (environment_count,),
            dtype=mx.uint32,
        ),
    )


def accumulate_episode_evidence(
    accumulator: EpisodeAccumulator,
    *,
    reward: mx.array,
    visibility: mx.array,
    contact_load: mx.array,
    valid_contact_count: mx.array,
    control_period_seconds: float,
) -> EpisodeAccumulator:
    """Accumulate rollout evidence without evaluating an environment on CPU."""

    if control_period_seconds <= 0.0:
        raise ValueError("control_period_seconds must be positive")
    shape = accumulator.episode_return.shape
    if (
        reward.shape != shape
        or visibility.shape != shape
        or contact_load.shape != shape
        or valid_contact_count.shape != shape
    ):
        raise ValueError("episode evidence tensors must be environment vectors")
    return EpisodeAccumulator(
        episode_return=accumulator.episode_return + reward,
        duration_seconds=(
            accumulator.duration_seconds + control_period_seconds
        ),
        minimum_visibility=mx.minimum(
            accumulator.minimum_visibility,
            visibility,
        ),
        integrated_contact_load=(
            accumulator.integrated_contact_load
            + contact_load * control_period_seconds
        ),
        peak_contact_load=mx.maximum(
            accumulator.peak_contact_load,
            contact_load,
        ),
        valid_contact_count=(
            accumulator.valid_contact_count
            + valid_contact_count.astype(mx.uint32)
        ),
        step_count=(
            accumulator.step_count
            + mx.array(1, dtype=mx.uint32)
        ),
    )


def reset_episode_accumulator(
    accumulator: EpisodeAccumulator,
    completed: mx.array,
) -> EpisodeAccumulator:
    """Reset only completed environments as a pure MLX state transition."""

    if completed.shape != accumulator.episode_return.shape:
        raise ValueError("completed mask has the wrong shape")
    completed = completed.astype(mx.bool_)
    zeros_float = mx.zeros_like(accumulator.episode_return)
    zeros_uint = mx.zeros_like(accumulator.step_count)
    return EpisodeAccumulator(
        episode_return=mx.where(
            completed,
            zeros_float,
            accumulator.episode_return,
        ),
        duration_seconds=mx.where(
            completed,
            zeros_float,
            accumulator.duration_seconds,
        ),
        minimum_visibility=mx.where(
            completed,
            mx.ones_like(accumulator.minimum_visibility),
            accumulator.minimum_visibility,
        ),
        integrated_contact_load=mx.where(
            completed,
            zeros_float,
            accumulator.integrated_contact_load,
        ),
        peak_contact_load=mx.where(
            completed,
            zeros_float,
            accumulator.peak_contact_load,
        ),
        valid_contact_count=mx.where(
            completed,
            zeros_uint,
            accumulator.valid_contact_count,
        ),
        step_count=mx.where(
            completed,
            zeros_uint,
            accumulator.step_count,
        ),
    )


def compact_completed_episodes(
    accumulator: EpisodeAccumulator,
    *,
    scenario_headers: mx.array,
    scenario_values: mx.array,
    scenario_identities: mx.array,
    completed: mx.array,
    success: mx.array,
    termination: mx.array,
    physics_status: mx.array,
    failure_mask_low: mx.array,
    failure_mask_high: mx.array,
    task_margin: mx.array,
    safety_margin: mx.array,
    source: int = 0,
) -> CompactedEpisodeRecords:
    """GPU-compact completed episodes into the fixed R2S2R record ABI.

    The output retains environment capacity; ``valid_count`` and ``valid_mask``
    identify its dense prefix. Only a rollout-chunk drain should materialize
    that prefix on the host.
    """

    environment_count = int(accumulator.episode_return.shape[0])
    vector_shape = (environment_count,)
    if (
        scenario_headers.shape != (environment_count, 3, 4)
        or int(scenario_values.shape[0]) != environment_count
        or scenario_identities.shape != scenario_values.shape
        or any(
            value.shape != vector_shape
            for value in (
                completed,
                success,
                termination,
                physics_status,
                failure_mask_low,
                failure_mask_high,
                task_margin,
                safety_margin,
            )
        )
        or source not in {0, 1}
    ):
        raise ValueError("completed-episode record tensors are incompatible")
    completed = completed.astype(mx.bool_)
    # False sorts before true, so completed environments form a dense prefix.
    order = mx.argsort((~completed).astype(mx.uint32))
    outcome = mx.stack(
        (
            mx.full(
                vector_shape,
                source,
                dtype=mx.uint32,
            ),
            termination.astype(mx.uint32),
            success.astype(mx.uint32),
            physics_status.astype(mx.uint32),
        ),
        axis=-1,
    )
    evidence = mx.stack(
        (
            failure_mask_low.astype(mx.uint32),
            failure_mask_high.astype(mx.uint32),
            accumulator.step_count,
            mx.full(
                vector_shape,
                1,
                dtype=mx.uint32,
            ),
        ),
        axis=-1,
    )
    outcome_words = mx.concatenate(
        (
            scenario_headers[:, 0, :],
            outcome,
            evidence,
        ),
        axis=-1,
    )
    task_values = mx.stack(
        (
            accumulator.episode_return,
            task_margin.astype(mx.float32),
            safety_margin.astype(mx.float32),
            accumulator.duration_seconds,
        ),
        axis=-1,
    )
    interaction_values = mx.stack(
        (
            accumulator.minimum_visibility,
            accumulator.integrated_contact_load,
            accumulator.peak_contact_load,
            accumulator.valid_contact_count.astype(mx.float32),
        ),
        axis=-1,
    )
    valid_count = mx.sum(completed.astype(mx.uint32))
    valid_mask = (
        mx.arange(environment_count, dtype=mx.uint32)
        < valid_count
    )
    return CompactedEpisodeRecords(
        outcome_words=outcome_words[order],
        task_values=task_values[order],
        interaction_values=interaction_values[order],
        scenario_headers=scenario_headers[order],
        scenario_values=scenario_values[order],
        scenario_identities=scenario_identities[order],
        valid_mask=valid_mask,
        valid_count=valid_count,
    )


def _huber(values: mx.array, delta: float) -> mx.array:
    magnitude = mx.abs(values)
    return mx.where(
        magnitude <= delta,
        0.5 * values * values,
        delta * (magnitude - 0.5 * delta),
    )


def fit_alignment_smc(
    initial_quantiles: mx.array,
    residual_count: int,
    evaluator: ReplayEvaluator,
    *,
    config: SMCConfig = SMCConfig(),
) -> AlignmentPopulationArrays:
    """Fit a multimodal replay posterior with four-round robust SMC."""

    if initial_quantiles.ndim != 2:
        raise ValueError(
            "initial_quantiles must have shape [particle, feature]"
        )
    particle_count = int(initial_quantiles.shape[0])
    if not 1 <= particle_count <= 4096:
        raise ValueError("alignment requires between 1 and 4096 particles")
    if int(initial_quantiles.shape[1]) <= 0 or residual_count <= 0:
        raise ValueError("alignment feature/residual counts must be positive")
    if (
        config.rounds <= 0
        or config.huber_delta <= 0.0
        or not 0.0 < config.minimum_effective_sample_fraction <= 1.0
        or config.jitter_scale < 0.0
    ):
        raise ValueError("SMC configuration is invalid")

    particles = initial_quantiles.astype(mx.float32)
    weights = mx.full(
        (particle_count,),
        1.0 / float(particle_count),
        dtype=mx.float32,
    )
    key = mx.random.key(config.seed)
    losses = mx.zeros((particle_count,), dtype=mx.float32)
    for round_index in range(config.rounds):
        residuals = evaluator(particles)
        expected = (particle_count, residual_count)
        if tuple(residuals.shape) != expected:
            raise ValueError(
                f"replay evaluator returned {residuals.shape}, "
                f"expected {expected}"
            )
        losses = mx.sum(
            _huber(residuals.astype(mx.float32), config.huber_delta),
            axis=-1,
        )
        scale = mx.maximum(mx.median(losses), 1.0e-9)
        log_weights = (
            mx.log(mx.maximum(weights, 1.0e-30))
            - losses / scale / float(config.rounds)
        )
        log_weights -= mx.max(log_weights)
        weights = mx.exp(log_weights)
        weights /= mx.sum(weights)
        effective_sample_size = 1.0 / mx.sum(weights * weights)
        mx.eval(weights, effective_sample_size)
        if (
            round_index + 1 == config.rounds
            or float(effective_sample_size.item())
            >= config.minimum_effective_sample_fraction
            * particle_count
        ):
            continue

        key, offset_key, jitter_key = mx.random.split(key, num=3)
        offset = mx.random.uniform(
            (),
            low=0.0,
            high=1.0 / float(particle_count),
            key=offset_key,
        )
        selectors = offset + mx.arange(
            particle_count,
            dtype=mx.float32,
        ) / float(particle_count)
        cdf = mx.cumsum(weights)
        # The comparison implements inverse-CDF search without a host loop.
        ancestors = mx.sum(
            cdf[None, :] < selectors[:, None],
            axis=1,
        ).astype(mx.uint32)
        jitter = mx.random.normal(
            particles.shape,
            key=jitter_key,
        ) * (
            config.jitter_scale
            / ((round_index + 1.0) ** 0.5)
        )
        particles = mx.clip(particles[ancestors] + jitter, 0.0, 1.0)
        weights = mx.full(
            (particle_count,),
            1.0 / float(particle_count),
            dtype=mx.float32,
        )

    order = mx.argsort(-weights)
    result = AlignmentPopulationArrays(
        quantiles=particles[order],
        weights=weights[order] / mx.sum(weights),
        replay_residuals=losses[order],
    )
    mx.eval(*result)
    return result


class FailureTrainingBatch(NamedTuple):
    quantiles: mx.array
    success: mx.array
    task_margin: mx.array
    hardware_mask: mx.array


class FailureScores(NamedTuple):
    simulation_success: mx.array
    hardware_success: mx.array | None
    failure_score: mx.array
    uncertainty_score: mx.array


class QuantileRegion(NamedTuple):
    kind: int
    weight: float
    lower: list[float]
    upper: list[float]
    score: float


def _forward(
    parameters: dict[str, mx.array],
    quantiles: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    hidden = mx.tanh(
        mx.einsum("nf,mfh->mnh", quantiles, parameters["w1"])
        + parameters["b1"][:, None, :]
    )
    hidden = mx.tanh(
        mx.einsum("mnh,mhk->mnk", hidden, parameters["w2"])
        + parameters["b2"][:, None, :]
    )
    success = (
        mx.einsum("mnh,mh->mn", hidden, parameters["success_w"])
        + parameters["success_b"][:, None]
    )
    margin = (
        mx.einsum("mnh,mh->mn", hidden, parameters["margin_w"])
        + parameters["margin_b"][:, None]
    )
    residual = (
        mx.einsum("mnh,mh->mn", hidden, parameters["residual_w"])
        + parameters["residual_b"][:, None]
    )
    return success, margin, residual


class FiveMemberFailureEnsemble:
    """Policy/task/embodiment-specific simulation plus hardware residual model."""

    def __init__(
        self,
        feature_count: int,
        *,
        hidden_width: int = 64,
        seed: int = 1,
    ) -> None:
        if feature_count <= 0 or hidden_width <= 0:
            raise ValueError("ensemble dimensions must be positive")
        self.feature_count = int(feature_count)
        self.hidden_width = int(hidden_width)
        self.member_count = 5
        key = mx.random.key(seed)
        keys = mx.random.split(key, num=6)
        scale1 = (2.0 / float(feature_count + hidden_width)) ** 0.5
        scale2 = (1.0 / float(hidden_width)) ** 0.5
        self.parameters: dict[str, mx.array] = {
            "w1": mx.random.normal(
                (
                    self.member_count,
                    feature_count,
                    hidden_width,
                ),
                key=keys[0],
            )
            * scale1,
            "b1": mx.zeros(
                (self.member_count, hidden_width),
                dtype=mx.float32,
            ),
            "w2": mx.random.normal(
                (
                    self.member_count,
                    hidden_width,
                    hidden_width,
                ),
                key=keys[1],
            )
            * scale2,
            "b2": mx.zeros(
                (self.member_count, hidden_width),
                dtype=mx.float32,
            ),
            "success_w": mx.random.normal(
                (self.member_count, hidden_width),
                key=keys[2],
            )
            * scale2,
            "success_b": mx.zeros(
                (self.member_count,),
                dtype=mx.float32,
            ),
            "margin_w": mx.random.normal(
                (self.member_count, hidden_width),
                key=keys[3],
            )
            * scale2,
            "margin_b": mx.zeros(
                (self.member_count,),
                dtype=mx.float32,
            ),
            "residual_w": mx.random.normal(
                (self.member_count, hidden_width),
                key=keys[4],
            )
            * (0.1 * scale2),
            "residual_b": mx.zeros(
                (self.member_count,),
                dtype=mx.float32,
            ),
        }
        self.hardware_available = False

    def fit(
        self,
        batch: FailureTrainingBatch,
        *,
        steps: int = 300,
        learning_rate: float = 3.0e-3,
    ) -> list[float]:
        if (
            batch.quantiles.ndim != 2
            or int(batch.quantiles.shape[1]) != self.feature_count
            or tuple(batch.success.shape)
            != (int(batch.quantiles.shape[0]),)
            or batch.task_margin.shape != batch.success.shape
            or batch.hardware_mask.shape != batch.success.shape
            or steps <= 0
        ):
            raise ValueError("failure-training batch has invalid shapes")
        sample_count = int(batch.quantiles.shape[0])
        if sample_count == 0:
            raise ValueError("failure-training batch is empty")
        x = batch.quantiles.astype(mx.float32)
        success = batch.success.astype(mx.float32)
        margin = batch.task_margin.astype(mx.float32)
        hardware = batch.hardware_mask.astype(mx.float32)
        simulation = 1.0 - hardware
        mx.eval(hardware)
        self.hardware_available = bool(mx.sum(hardware).item() > 0)

        sample_index = mx.arange(sample_count, dtype=mx.uint32)[None, :]
        member_index = mx.arange(
            self.member_count,
            dtype=mx.uint32,
        )[:, None]
        bootstrap = (
            (
                sample_index * mx.array(2654435761, dtype=mx.uint32)
                + member_index * mx.array(2246822519, dtype=mx.uint32)
            )
            % mx.array(10, dtype=mx.uint32)
            < 8
        ).astype(mx.float32)
        bootstrap += 0.1
        simulation_weight = bootstrap * simulation[None, :]
        hardware_weight = bootstrap * hardware[None, :]
        member_success = mx.broadcast_to(
            success[None, :],
            (self.member_count, sample_count),
        )
        member_margin = mx.broadcast_to(
            margin[None, :],
            (self.member_count, sample_count),
        )

        def loss_fn(parameters: dict[str, mx.array]) -> mx.array:
            sim_logit, predicted_margin, residual_logit = _forward(
                parameters,
                x,
            )
            sim_bce = nn.losses.binary_cross_entropy(
                sim_logit,
                member_success,
                reduction="none",
            )
            sim_loss = mx.sum(sim_bce * simulation_weight) / mx.maximum(
                mx.sum(simulation_weight),
                1.0,
            )
            margin_loss = mx.sum(
                (predicted_margin - member_margin) ** 2
                * simulation_weight
            ) / mx.maximum(mx.sum(simulation_weight), 1.0)
            hardware_logit = (
                mx.stop_gradient(sim_logit) + residual_logit
            )
            hardware_bce = nn.losses.binary_cross_entropy(
                hardware_logit,
                member_success,
                reduction="none",
            )
            hardware_loss = mx.sum(
                hardware_bce * hardware_weight
            ) / mx.maximum(mx.sum(hardware_weight), 1.0)
            residual_regularizer = 1.0e-3 * mx.mean(
                residual_logit * residual_logit
            )
            return (
                sim_loss
                + 0.35 * margin_loss
                + hardware_loss
                + residual_regularizer
            )

        value_and_grad = mx.value_and_grad(loss_fn)
        optimizer = optim.Adam(learning_rate=learning_rate)
        history: list[float] = []
        for step_index in range(steps):
            loss, gradients = value_and_grad(self.parameters)
            self.parameters = optimizer.apply_gradients(
                gradients,
                self.parameters,
            )
            mx.eval(loss, self.parameters, optimizer.state)
            if step_index == 0 or (step_index + 1) % 25 == 0:
                history.append(float(loss.item()))
        return history

    def score(self, quantiles: mx.array) -> FailureScores:
        if (
            quantiles.ndim != 2
            or int(quantiles.shape[1]) != self.feature_count
        ):
            raise ValueError("score input has the wrong feature width")
        sim_logit, _, residual_logit = _forward(
            self.parameters,
            quantiles.astype(mx.float32),
        )
        simulation_members = mx.sigmoid(sim_logit)
        simulation_success = mx.mean(simulation_members, axis=0)
        if self.hardware_available:
            hardware_members = mx.sigmoid(
                sim_logit + residual_logit
            )
            hardware_success: mx.array | None = mx.mean(
                hardware_members,
                axis=0,
            )
            members = hardware_members
        else:
            hardware_success = None
            members = simulation_members
        mean = mx.mean(members, axis=0)
        uncertainty = mx.mean(
            (members - mean[None, :]) ** 2,
            axis=0,
        )
        return FailureScores(
            simulation_success=simulation_success,
            hardware_success=hardware_success,
            failure_score=1.0 - mean,
            uncertainty_score=uncertainty,
        )

    def save(self, path: str) -> None:
        mx.savez(path, **self.parameters)


def deterministic_candidate_scenarios(
    feature_count: int,
    *,
    count: int = 65536,
) -> mx.array:
    """Generate a deterministic space-filling candidate tensor on the GPU."""

    if feature_count <= 0 or count <= 0:
        raise ValueError("candidate dimensions must be positive")
    primes = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
    ]
    multipliers = mx.array(
        [primes[index % len(primes)] for index in range(feature_count)],
        dtype=mx.uint32,
    )
    offsets = mx.arange(feature_count, dtype=mx.uint32) * 131
    rows = mx.arange(count, dtype=mx.uint32)[:, None]
    return (
        (
            rows * multipliers[None, :] + offsets[None, :]
        )
        % mx.array(count, dtype=mx.uint32)
    ).astype(mx.float32) / float(count)


def _cluster_regions(
    candidates: mx.array,
    scores: mx.array,
    *,
    kind: int,
    region_count: int,
    top_count: int,
    iterations: int = 8,
) -> list[QuantileRegion]:
    selected_count = min(top_count, int(candidates.shape[0]))
    cluster_count = min(region_count, selected_count)
    if cluster_count <= 0:
        return []
    order = mx.argsort(scores)
    indices = order[-selected_count:]
    points = candidates[indices]
    selected_scores = scores[indices]
    seeds = (
        mx.arange(cluster_count, dtype=mx.uint32)
        * selected_count
        // cluster_count
    )
    centers = points[seeds]
    assignments = mx.zeros((selected_count,), dtype=mx.uint32)
    for _ in range(iterations):
        distances = mx.sum(
            (points[:, None, :] - centers[None, :, :]) ** 2,
            axis=-1,
        )
        assignments = mx.argmin(distances, axis=1)
        membership = (
            assignments[:, None]
            == mx.arange(cluster_count, dtype=mx.uint32)[None, :]
        ).astype(mx.float32)
        counts = mx.sum(membership, axis=0)
        updated = mx.einsum("nk,nf->kf", membership, points)
        updated /= mx.maximum(counts[:, None], 1.0)
        centers = mx.where(
            counts[:, None] > 0.0,
            updated,
            centers,
        )

    membership = (
        assignments[:, None]
        == mx.arange(cluster_count, dtype=mx.uint32)[None, :]
    )
    counts = mx.sum(membership.astype(mx.float32), axis=0)
    expanded = points[:, None, :]
    lower = mx.min(
        mx.where(membership[:, :, None], expanded, 1.0),
        axis=0,
    )
    upper = mx.max(
        mx.where(membership[:, :, None], expanded, 0.0),
        axis=0,
    )
    score_sums = mx.sum(
        membership.astype(mx.float32)
        * selected_scores[:, None],
        axis=0,
    )
    mean_scores = score_sums / mx.maximum(counts, 1.0)
    weights = counts * mx.maximum(mean_scores, 1.0e-6)
    weights /= mx.maximum(mx.sum(weights), 1.0e-6)
    mx.eval(lower, upper, mean_scores, weights, counts)
    lower_values = lower.tolist()
    upper_values = upper.tolist()
    score_values = mean_scores.tolist()
    weight_values = weights.tolist()
    count_values = counts.tolist()
    return [
        QuantileRegion(
            kind=kind,
            weight=float(weight_values[index]),
            lower=[float(value) for value in lower_values[index]],
            upper=[float(value) for value in upper_values[index]],
            score=float(score_values[index]),
        )
        for index in range(cluster_count)
        if float(count_values[index]) > 0.0
    ]


def compile_feedback_regions(
    ensemble: FiveMemberFailureEnsemble,
    *,
    candidate_count: int = 65536,
    maximum_regions: int = 64,
) -> tuple[list[QuantileRegion], FailureScores]:
    """Score 65,536 GPU candidates and compile <=64 quantile boxes."""

    if not 2 <= maximum_regions <= 64:
        raise ValueError("maximum_regions must be between 2 and 64")
    if candidate_count < 8:
        raise ValueError("candidate_count must be at least 8")
    candidates = deterministic_candidate_scenarios(
        ensemble.feature_count,
        count=candidate_count,
    )
    scores = ensemble.score(candidates)
    half = maximum_regions // 2
    failure = _cluster_regions(
        candidates,
        scores.failure_score,
        kind=0,
        region_count=half,
        top_count=min(candidate_count // 8, 8192),
    )
    uncertainty = _cluster_regions(
        candidates,
        scores.uncertainty_score,
        kind=1,
        region_count=maximum_regions - half,
        top_count=min(candidate_count // 8, 8192),
    )
    return failure + uncertainty, scores


__all__ = [
    "AlignmentPopulationArrays",
    "CompactedEpisodeRecords",
    "EpisodeAccumulator",
    "FailureScores",
    "FailureTrainingBatch",
    "FiveMemberFailureEnsemble",
    "QuantileRegion",
    "SMCConfig",
    "compile_feedback_regions",
    "compact_completed_episodes",
    "deterministic_candidate_scenarios",
    "fit_alignment_smc",
    "initial_episode_accumulator",
    "accumulate_episode_evidence",
    "reset_episode_accumulator",
]
