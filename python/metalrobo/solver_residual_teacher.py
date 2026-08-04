"""Turn solver-realized stochastic residual rollouts into teacher labels.

ARDY remains the nominal motion.  A native PolicyPack samples closed-loop
residuals, NumiSolver measures every candidate under gravity and contact, and
this module gives every physically valid trajectory a continuous contribution
according to reference-relative tracking and realized TaskPack reward.  The
non-dominated set remains evidence, not an admission gate.  This module never
authors a pose, root trajectory, contact, or support state.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ResidualCandidate:
    environment: int
    peak_forward_progress_m: float
    final_forward_progress_m: float
    mean_tracking_score: float
    mean_root_height: float
    minimum_root_height: float
    mean_tilt: float
    maximum_tilt: float
    termination_count: int
    physics_error_count: int
    mean_reward: float = 0.0
    pareto: bool = False
    quality: float = 0.0


_ARRAY_FIELDS = {
    "peak_forward_progress_m": "peak_forward_progress_by_environment_m",
    "final_forward_progress_m": "final_forward_progress_by_environment_m",
    "mean_tracking_score": "mean_tracking_score_by_environment",
    "mean_root_height": "mean_root_height_by_environment",
    "minimum_root_height": "minimum_root_height_by_environment",
    "mean_tilt": "mean_tilt_by_environment",
    "maximum_tilt": "maximum_tilt_by_environment",
    "termination_count": "termination_count_by_environment",
    "physics_error_count": "physics_error_count_by_environment",
}


def _finite_array(
    evidence: dict[str, Any],
    key: str,
    environment_count: int,
) -> np.ndarray:
    values = np.asarray(evidence.get(key, ()), dtype=np.float64)
    if values.shape != (environment_count,):
        raise ValueError(
            f"{key} must contain one value per environment "
            f"({environment_count})"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key} contains a non-finite value")
    return values


def _rank_quality(values: np.ndarray, *, higher: bool) -> np.ndarray:
    """Return deterministic [0, 1] mid-ranks without scale assumptions."""

    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and values[order[end]] == values[order[index]]:
            end += 1
        rank = 0.5 * (index + end - 1)
        ranks[order[index:end]] = rank
        index = end
    if values.size > 1:
        ranks /= values.size - 1
    else:
        ranks.fill(1.0)
    return ranks if higher else 1.0 - ranks


def rank_physical_realizations(
    evidence: dict[str, Any],
    *,
    environment_count: int,
    mean_rewards: np.ndarray | None = None,
) -> tuple[list[ResidualCandidate], list[int]]:
    """Continuously rank all realizations and report their Pareto set."""

    if environment_count <= 0:
        raise ValueError("environment_count must be positive")
    arrays = {
        field: _finite_array(evidence, key, environment_count)
        for field, key in _ARRAY_FIELDS.items()
    }
    rewards = (
        np.zeros(environment_count, dtype=np.float64)
        if mean_rewards is None
        else np.asarray(mean_rewards, dtype=np.float64)
    )
    if rewards.shape != (environment_count,):
        raise ValueError(
            "mean_rewards must contain one value per environment "
            f"({environment_count})"
        )
    if not np.all(np.isfinite(rewards)):
        raise ValueError("mean_rewards contains a non-finite value")

    # Reference-relative tracking and the authored TaskPack reward are valid
    # across standing, locomotion, get-up, crouching, manipulation, and
    # acrobatics.  Absolute forward distance, height, and world tilt remain in
    # evidence, but they cannot rank a generic motion: low height may be the
    # requested crouch and large tilt may be the requested rotation.
    objectives = np.column_stack(
        (
            arrays["mean_tracking_score"],
            rewards,
            -arrays["termination_count"],
            -arrays["physics_error_count"],
        )
    )
    pareto = np.ones(environment_count, dtype=bool)
    for candidate in range(environment_count):
        dominates = np.all(
            objectives >= objectives[candidate], axis=1
        ) & np.any(objectives > objectives[candidate], axis=1)
        dominates[candidate] = False
        pareto[candidate] = not bool(np.any(dominates))

    reward_weight = 0.25 if mean_rewards is not None else 0.0
    quality = (
        (0.65 + 0.25 - reward_weight)
        * _rank_quality(arrays["mean_tracking_score"], higher=True)
        + reward_weight * _rank_quality(rewards, higher=True)
        + 0.05 * _rank_quality(arrays["termination_count"], higher=False)
        + 0.05 * _rank_quality(arrays["physics_error_count"], higher=False)
    )
    ranked = np.argsort(-quality, kind="stable")
    candidates = [
        ResidualCandidate(
            environment=environment,
            **{
                field: (
                    int(values[environment])
                    if field.endswith("_count")
                    else float(values[environment])
                )
                for field, values in arrays.items()
            },
            mean_reward=float(rewards[environment]),
            pareto=bool(pareto[environment]),
            quality=float(quality[environment]),
        )
        for environment in range(environment_count)
    ]
    return candidates, [int(value) for value in ranked]


# Retain the public name for callers written against the first residual
# teacher.  Ranking now returns every realization; the Pareto flag is evidence
# rather than a filter.
def select_physical_frontier(
    evidence: dict[str, Any],
    *,
    environment_count: int,
    elite_count: int | None = None,
) -> tuple[list[ResidualCandidate], list[int]]:
    del elite_count
    return rank_physical_realizations(
        evidence,
        environment_count=environment_count,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distill_residual_frontier(
    *,
    policy_pack_path: Path,
    rollout_pack_path: Path,
    evidence_path: Path,
    output_policy_path: Path,
    output_deployment_policy_path: Path,
    output_evidence_path: Path,
    learning_rate: float,
    update_epochs: int,
    minibatch_size: int,
    seed: int,
    native_library_path: Path | None = None,
) -> dict[str, Any]:
    from metalrobo.mlx_policy_learning import (
        MLXPPOConfiguration,
        MLXPolicyLearner,
        MLXPolicyBatch,
        read_policy_rollout_pack,
    )

    rollout = read_policy_rollout_pack(
        rollout_pack_path,
        library_path=native_library_path,
    )
    evidence = json.loads(evidence_path.read_text())
    transitions = rollout.transitions.reshape(
        rollout.control_step_count, rollout.environment_count
    )
    mean_rewards = np.mean(transitions["reward"], axis=0)
    candidates, ranked = rank_physical_realizations(
        evidence,
        environment_count=rollout.environment_count,
        mean_rewards=mean_rewards,
    )
    if not ranked:
        raise ValueError("physical search did not contain a candidate")

    steps = rollout.control_step_count
    environments = rollout.environment_count
    samples = rollout.sample_count
    environment_weights = np.asarray(
        [0.10 + 0.90 * candidate.quality for candidate in candidates],
        dtype=np.float32,
    )
    step_reward_quality = _rank_quality(
        transitions["reward"].reshape(samples).astype(np.float64),
        higher=True,
    ).astype(np.float32).reshape(steps, environments)
    valid = (
        (transitions["physics_error"] == 0)
        & (transitions["done"] == 0)
    ).astype(np.float32)
    teacher_weights = (
        environment_weights[np.newaxis, :]
        * valid
        * (np.float32(0.25) + np.float32(0.75) * step_reward_quality)
    ).reshape(samples)
    if not np.any(teacher_weights > 0.0):
        raise ValueError("physical search has no valid transitions")

    configuration = MLXPPOConfiguration(
        update_epochs=update_epochs,
        minibatch_size=min(minibatch_size, samples),
        learning_rate=learning_rate,
        minimum_learning_rate=learning_rate,
        maximum_learning_rate=learning_rate,
        value_coefficient=0.0,
        entropy_coefficient=0.0,
        imagination_distillation_coefficient=1.0,
        adaptive_learning_rate=False,
        target_kl=None,
        seed=seed,
    )
    learner = MLXPolicyLearner.from_policy_pack(
        policy_pack_path,
        configuration,
        library_path=native_library_path,
    )
    batch = MLXPolicyBatch.from_numpy(
        actor_observations=rollout.actor_observations.reshape(
            samples, rollout.actor_observation_count
        ),
        critic_observations=rollout.critic_observations.reshape(
            samples, rollout.critic_observation_count
        ),
        latents=rollout.latents.reshape(samples, rollout.action_count),
        old_log_probabilities=rollout.old_log_probabilities,
        old_values=rollout.old_values,
        advantages=np.zeros(samples, dtype=np.float32),
        returns=rollout.old_values,
        # Actor means and sampled latents share raw Gaussian policy space.
        teacher_actions=rollout.latents.reshape(samples, rollout.action_count),
        teacher_weights=teacher_weights,
        policy_weights=np.zeros(samples, dtype=np.float32),
    )
    metrics = learner.update(batch)
    source_policy_id = learner.policy_id
    output_policy_path.parent.mkdir(parents=True, exist_ok=True)
    learner.write_policy_pack(
        output_policy_path,
        policy_id=f"{source_policy_id}-solver-residual-teacher",
        stochastic=True,
        library_path=native_library_path,
    )
    output_deployment_policy_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    learner.write_policy_pack(
        output_deployment_policy_path,
        policy_id=(
            f"{source_policy_id}-solver-residual-teacher-deployment"
        ),
        stochastic=False,
        library_path=native_library_path,
    )
    result = {
        "schema": "numi.solver-residual-teacher.v2",
        "principle": (
            "ARDY supplies nominal motion; sampled policy residuals are "
            "continuously weighted by NumiSolver physical outcomes without "
            "a motion-class or elite admission gate."
        ),
        "source_policy": str(policy_pack_path),
        "source_policy_sha256": _sha256(policy_pack_path),
        "source_rollout": str(rollout_pack_path),
        "source_rollout_sha256": _sha256(rollout_pack_path),
        "source_evidence": str(evidence_path),
        "source_evidence_sha256": _sha256(evidence_path),
        "output_policy": str(output_policy_path),
        "output_policy_sha256": _sha256(output_policy_path),
        "output_deployment_policy": str(output_deployment_policy_path),
        "output_deployment_policy_sha256": _sha256(
            output_deployment_policy_path
        ),
        "environment_count": environments,
        "control_step_count": steps,
        "physical_frontier_environments": [
            candidate.environment for candidate in candidates if candidate.pareto
        ],
        "ranked_environments": ranked,
        "contributing_environments": [
            candidate.environment for candidate in candidates
        ],
        "ranked_candidates": [
            asdict(candidates[environment]) for environment in ranked
        ],
        "teacher_sample_count": int(np.count_nonzero(teacher_weights)),
        "mean_teacher_weight": float(np.mean(teacher_weights)),
        "learning_rate": learning_rate,
        "update_epochs": update_epochs,
        "training_metrics": metrics,
        "limitations": [
            "The output is a candidate and requires matched physical "
            "evaluation before promotion.",
            "This is simulator evidence, not real-hardware evidence.",
        ],
    }
    output_evidence_path.parent.mkdir(parents=True, exist_ok=True)
    output_evidence_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def run_solver_search(
    *,
    evaluator: Path,
    metallib: Path,
    policy_pack_path: Path,
    interaction_pack_path: Path,
    interaction_clip: str,
    output_directory: Path,
    environment_count: int,
    control_step_count: int,
    seed: int,
) -> tuple[Path, Path]:
    if environment_count <= 0 or control_step_count <= 0:
        raise ValueError("search environments and steps must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    rollout_path = output_directory / "search.rolloutpack"
    evidence_path = output_directory / "search.evidence.json"
    trace_path = output_directory / "search-environment-0.tsv"
    command = [
        str(evaluator),
        "--metallib", str(metallib),
        "--task", "velocity",
        "--scene", "ground",
        "--interaction-pack", str(interaction_pack_path),
        "--interaction-clip", interaction_clip,
        "--interaction-student-authority", "1",
        "--interaction-reset-phase-probability", "0",
        "--interaction-reset-maximum-phase", "0",
        "--policy-pack", str(policy_pack_path),
        "--envs", str(environment_count),
        "--steps", str(control_step_count),
        "--repeats", "1",
        "--chunk", "1",
        "--seed", str(seed),
        "--rollout-pack", str(rollout_path),
        "--state-trace", str(trace_path),
        "--state-trace-environment", "0",
    ]
    (output_directory / "search.arguments.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
    )
    (output_directory / "search.stdout.log").write_text(completed.stdout)
    (output_directory / "search.stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            "NumiSolver residual search failed; see "
            f"{output_directory / 'search.stderr.log'}"
        )
    record = json.loads(completed.stdout)
    if int(record.get("failed_environment_steps", -1)) != 0:
        raise RuntimeError(
            "NumiSolver residual search reported failed environment steps"
        )
    evidence_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return rollout_path, evidence_path


def _add_distillation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy-pack", type=Path, required=True)
    parser.add_argument("--rollout-pack", type=Path, required=True)
    parser.add_argument("--rollout-evidence", type=Path, required=True)
    parser.add_argument("--output-policy", type=Path, required=True)
    parser.add_argument(
        "--output-deployment-policy", type=Path, required=True
    )
    parser.add_argument("--output-evidence", type=Path, required=True)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--learning-rate", type=float, default=2.0e-6)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2650443581)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Distil a NumiSolver-qualified stochastic residual frontier; "
            "no pose or contact state is authored."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    distill = commands.add_parser(
        "distill", help="distil an existing physical search rollout"
    )
    _add_distillation_arguments(distill)
    run = commands.add_parser(
        "run", help="run native search and distil its physical frontier"
    )
    run.add_argument("--evaluator", type=Path, required=True)
    run.add_argument("--metallib", type=Path, required=True)
    run.add_argument("--policy-pack", type=Path, required=True)
    run.add_argument("--interaction-pack", type=Path, required=True)
    run.add_argument("--interaction-clip", required=True)
    run.add_argument("--output-directory", type=Path, required=True)
    run.add_argument("--native-library", type=Path)
    run.add_argument("--envs", type=int, default=256)
    run.add_argument("--steps", type=int, default=104)
    run.add_argument("--learning-rate", type=float, default=2.0e-6)
    run.add_argument("--update-epochs", type=int, default=2)
    run.add_argument("--minibatch-size", type=int, default=4096)
    run.add_argument("--seed", type=int, default=2650443581)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    if options.command == "run":
        rollout_path, evidence_path = run_solver_search(
            evaluator=options.evaluator,
            metallib=options.metallib,
            policy_pack_path=options.policy_pack,
            interaction_pack_path=options.interaction_pack,
            interaction_clip=options.interaction_clip,
            output_directory=options.output_directory,
            environment_count=options.envs,
            control_step_count=options.steps,
            seed=options.seed,
        )
        options.rollout_pack = rollout_path
        options.rollout_evidence = evidence_path
        options.output_policy = options.output_directory / "candidate.policypack"
        options.output_deployment_policy = (
            options.output_directory / "candidate.deployment.policypack"
        )
        options.output_evidence = (
            options.output_directory / "distillation.evidence.json"
        )
    result = distill_residual_frontier(
        policy_pack_path=options.policy_pack,
        rollout_pack_path=options.rollout_pack,
        evidence_path=options.rollout_evidence,
        output_policy_path=options.output_policy,
        output_deployment_policy_path=options.output_deployment_policy,
        output_evidence_path=options.output_evidence,
        learning_rate=options.learning_rate,
        update_epochs=options.update_epochs,
        minibatch_size=options.minibatch_size,
        seed=options.seed,
        native_library_path=options.native_library,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
