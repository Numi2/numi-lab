"""Matched physical-outcome selection for Numi policy candidates.

Training always keeps the candidate.  This module only decides whether the
protected deployment artifact should advance from the pre-training incumbent.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Sequence


_VALUE_OPTIONS = frozenset(
    {
        "--envs",
        "--steps",
        "--seed",
        "--scene",
        "--task",
        "--minimum-difficulty-band",
        "--maximum-difficulty-band",
        "--interaction-student-authority",
        "--interaction-pack",
        "--interaction-clip",
        "--world-pack",
        "--task-pack",
        "--urdf",
        "--srdf",
        "--g1-visual-pack-dir",
        "--ball-visual-pack-dir",
        "--visual-observation-config",
        "--visual-environment-pack",
    }
)
_FLAG_OPTIONS = frozenset(
    {
        "--interaction-reset-only",
        "--materialize-articulated-contact-responses",
    }
)

_GENERIC_WORLD_SOURCES = frozenset({"world_pack", "urdf"})


def _option_value(arguments: Sequence[str], option: str) -> str | None:
    result = None
    for index, value in enumerate(arguments[:-1]):
        if value == option:
            result = arguments[index + 1]
    return result


def evaluation_arguments(
    training_arguments: Sequence[str],
    *,
    policy_pack: Path,
    metallib: Path,
    state_trace: Path,
    maximum_environments: int,
    held_out_seed: int,
    evaluation_steps: int | None = None,
) -> list[str]:
    """Project trainer arguments onto one deterministic rollout contract."""
    projected: list[str] = []
    index = 0
    while index < len(training_arguments):
        option = training_arguments[index]
        if option in _VALUE_OPTIONS:
            if index + 1 >= len(training_arguments):
                raise ValueError(f"{option} is missing its value")
            projected.extend((option, training_arguments[index + 1]))
            index += 2
            continue
        if option in _FLAG_OPTIONS:
            projected.append(option)
        index += 1

    training_environments = int(
        _option_value(training_arguments, "--envs") or 32
    )
    evaluation_environments = min(
        training_environments, maximum_environments
    )
    student_authority = _option_value(
        training_arguments, "--interaction-student-authority"
    )
    if student_authority is not None and float(student_authority) == 0.0:
        # A zero-authority rollout measures the teacher, not the learned actor.
        # Keep the teacher's accepted reset state, then evaluate autonomously.
        filtered: list[str] = []
        skip = False
        for value in projected:
            if skip:
                skip = False
                continue
            if value == "--interaction-student-authority":
                skip = True
                continue
            if value == "--interaction-reset-only":
                continue
            filtered.append(value)
        projected = filtered + ["--interaction-reset-only"]

    # Appended values win if a caller supplied the same option earlier.
    projected.extend(
        (
            "--envs",
            str(evaluation_environments),
            "--repeats",
            "1",
            "--chunk",
            "1",
            "--seed",
            str(held_out_seed),
            "--metallib",
            str(metallib),
            "--policy-pack",
            str(policy_pack),
            "--state-trace",
            str(state_trace),
        )
    )
    if evaluation_steps is not None:
        if evaluation_steps <= 0:
            raise ValueError("evaluation steps must be positive")
        projected.extend(("--steps", str(evaluation_steps)))
    return projected


def _rate(record: dict[str, Any], count: str, rate: str) -> float:
    if rate in record:
        return float(record[rate])
    environments = max(len(record.get("termination_count_by_environment", [])), 1)
    return float(record.get(count, 0)) / environments


def _support_rate(record: dict[str, Any]) -> float | None:
    evidence = record.get("squat_cycle_evidence_by_environment")
    if not isinstance(evidence, list):
        return None
    values = [
        float(item["bilateral_support_rate"])
        for item in evidence
        if isinstance(item, dict)
        and item.get("available")
        and "bilateral_support_rate" in item
    ]
    return sum(values) / len(values) if values else None


def _physical_failure_rate(record: dict[str, Any]) -> float:
    """Exclude ordinary horizon timeouts from physical policy failures."""

    environments = max(
        len(record.get("termination_count_by_environment", [])), 1
    )
    if (
        "height_or_tilt_termination_count" in record
        and str(record.get("world_source", ""))
        not in _GENERIC_WORLD_SOURCES
    ):
        failures = int(record["height_or_tilt_termination_count"])
    else:
        failures = max(
            int(record.get("termination_count", 0))
            - int(record.get("timeout_count", 0)),
            0,
        )
    return failures / environments


def _uses_generic_task_outcome(record: dict[str, Any]) -> bool:
    """Return whether selection must avoid bundled humanoid assumptions."""

    return str(record.get("world_source", "")) in _GENERIC_WORLD_SOURCES


def _relative_progress(old: float, new: float, floor: float) -> float:
    progress = (new - old) / max(abs(old), abs(new), floor)
    return max(-1.0, min(progress, 1.0))


def compare_evidence(
    incumbent: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compare matched physical evidence while retaining useful tradeoffs."""
    task = str(candidate.get("task", incumbent.get("task", "unknown")))
    generic_task = _uses_generic_task_outcome(candidate) or (
        _uses_generic_task_outcome(incumbent)
    )
    incumbent_termination = _physical_failure_rate(incumbent)
    candidate_termination = _physical_failure_rate(candidate)
    regressions: list[str] = []
    improvements: list[str] = []

    if int(candidate.get("failed_environment_steps", 0)) != 0:
        regressions.append("candidate has failed environment steps")
    if candidate_termination > incumbent_termination + 1.0e-12:
        regressions.append("termination rate increased")
    elif candidate_termination < incumbent_termination - 1.0e-12:
        improvements.append("termination rate decreased")

    if generic_task:
        old_task_reward = float(incumbent.get("mean_task_reward", 0))
        new_task_reward = float(candidate.get("mean_task_reward", 0))
        if new_task_reward < old_task_reward - 1.0e-12:
            regressions.append("authored task outcome decreased")
        elif new_task_reward > old_task_reward + 1.0e-12:
            improvements.append("authored task outcome increased")

        old_reward = float(incumbent.get("mean_reward", 0))
        new_reward = float(candidate.get("mean_reward", 0))
        if new_reward < old_reward - 1.0e-12:
            regressions.append("total task reward decreased")
        elif new_reward > old_reward + 1.0e-12:
            improvements.append("total task reward increased")

        old_tracking = float(incumbent.get("mean_tracking_score", 0))
        new_tracking = float(candidate.get("mean_tracking_score", 0))
        if new_tracking < old_tracking - 0.001:
            regressions.append("tracking score decreased")
        elif new_tracking > old_tracking + 0.001:
            improvements.append("tracking score increased")
    elif task == "ball-dodge":
        pairs = (
            ("any_link_dodge_rate", 1.0, "clean dodge rate"),
            ("any_link_projectile_hit_rate", -1.0, "projectile hit rate"),
        )
        for key, direction, label in pairs:
            old = float(incumbent.get(key, 0))
            new = float(candidate.get(key, 0))
            delta = direction * (new - old)
            if delta < -1.0e-12:
                regressions.append(f"{label} regressed")
            elif delta > 1.0e-12:
                improvements.append(f"{label} improved")
    elif task == "supine-get-up":
        old_completed = _rate(
            incumbent,
            "squat_cycle_completed_environment_count",
            "squat_cycle_completed_environment_rate",
        )
        new_completed = _rate(
            candidate,
            "squat_cycle_completed_environment_count",
            "squat_cycle_completed_environment_rate",
        )
        if new_completed < old_completed - 1.0e-12:
            regressions.append("completed squat-cycle rate decreased")
        elif new_completed > old_completed + 1.0e-12:
            improvements.append("completed squat-cycle rate increased")
        old_support = _support_rate(incumbent)
        new_support = _support_rate(candidate)
        if old_support is not None and new_support is not None:
            if new_support < old_support - 0.005:
                regressions.append("bilateral support rate decreased")
            elif new_support > old_support + 0.005:
                improvements.append("bilateral support rate increased")
    else:
        old_forward = float(
            incumbent.get("mean_peak_forward_progress_m", 0)
        )
        new_forward = float(
            candidate.get("mean_peak_forward_progress_m", 0)
        )
        if bool(incumbent.get("forward_progress_available")) and bool(
            candidate.get("forward_progress_available")
        ):
            if new_forward < old_forward - 0.005:
                regressions.append("mean peak forward progress decreased")
            elif new_forward > old_forward + 0.005:
                improvements.append("mean peak forward progress increased")
            old_final = float(
                incumbent.get(
                    "mean_final_forward_progress_m", old_forward
                )
            )
            new_final = float(
                candidate.get(
                    "mean_final_forward_progress_m", new_forward
                )
            )
            if new_final < old_final - 0.005:
                regressions.append("mean final forward progress decreased")
            elif new_final > old_final + 0.005:
                improvements.append("mean final forward progress increased")
        old_tracking = float(incumbent.get("mean_tracking_score", 0))
        new_tracking = float(candidate.get("mean_tracking_score", 0))
        if new_tracking < old_tracking - 0.001:
            regressions.append("tracking score decreased")
        elif new_tracking > old_tracking + 0.001:
            improvements.append("tracking score increased")

    old_tilt = float(incumbent.get("mean_tilt", 0))
    new_tilt = float(candidate.get("mean_tilt", 0))
    if not generic_task:
        if new_tilt > old_tilt + 0.005:
            regressions.append("mean tilt increased")
        elif new_tilt < old_tilt - 0.001:
            improvements.append("mean tilt decreased")

    old_height = float(incumbent.get("mean_root_height", 0))
    new_height = float(candidate.get("mean_root_height", 0))
    old_forward = float(
        incumbent.get("mean_peak_forward_progress_m", 0)
    )
    new_forward = float(
        candidate.get("mean_peak_forward_progress_m", 0)
    )
    upright_tasks = {
        "velocity",
        "disturbance-recovery",
        "ball-recovery",
        "ball-dodge",
    }
    if (
        not generic_task
        and task in upright_tasks
        and (old_height > 0.0 or new_height > 0.0)
    ):
        if new_height < old_height - 0.01:
            regressions.append("mean root height decreased")
        elif new_height > old_height + 0.005:
            improvements.append("mean root height increased")

    selection_score: float | None = None
    selection_method = "task_physical_comparison"
    if generic_task:
        old_task_reward = float(incumbent.get("mean_task_reward", 0))
        new_task_reward = float(candidate.get("mean_task_reward", 0))
        old_reward = float(incumbent.get("mean_reward", 0))
        new_reward = float(candidate.get("mean_reward", 0))
        old_tracking = float(incumbent.get("mean_tracking_score", 0))
        new_tracking = float(candidate.get("mean_tracking_score", 0))
        selection_score = (
            0.60
            * _relative_progress(
                old_task_reward, new_task_reward, 0.05
            )
            + 0.25
            * _relative_progress(old_reward, new_reward, 0.05)
            + 0.15 * (new_tracking - old_tracking)
            + incumbent_termination
            - candidate_termination
        )
        selected = (
            int(candidate.get("failed_environment_steps", 0)) == 0
            and selection_score > 1.0e-12
        )
        selection_method = "continuous_authored_task_outcome"
    elif (
        task == "velocity"
        and bool(incumbent.get("forward_progress_available"))
        and bool(candidate.get("forward_progress_available"))
    ):
        old_final = float(
            incumbent.get("mean_final_forward_progress_m", old_forward)
        )
        new_final = float(
            candidate.get("mean_final_forward_progress_m", new_forward)
        )
        peak_progress = (new_forward - old_forward) / max(
            abs(old_forward), 0.25
        )
        final_progress = (new_final - old_final) / max(
            abs(old_final), 0.25
        )
        tracking_progress = float(
            candidate.get("mean_tracking_score", 0)
        ) - float(incumbent.get("mean_tracking_score", 0))
        height_progress = (new_height - old_height) / 0.78
        tilt_progress = (old_tilt - new_tilt) / 3.141592653589793
        failure_progress = incumbent_termination - candidate_termination
        selection_score = (
            0.35 * peak_progress
            + 0.35 * final_progress
            + 0.10 * tracking_progress
            + 0.10 * height_progress
            + 0.10 * tilt_progress
            + failure_progress
        )
        # A device-rejected step is invalid evidence. Ordinary physical
        # tradeoffs remain continuous contributions rather than veto gates.
        selected = (
            int(candidate.get("failed_environment_steps", 0)) == 0
            and selection_score > 1.0e-12
        )
        selection_method = "continuous_locomotion_progress"
    else:
        selected = not regressions and bool(improvements)
    return {
        "schema": "numi.policy-selection.v1",
        "task": task,
        "selected": "candidate" if selected else "incumbent",
        "candidate_advanced_deployment": selected,
        "regressions": regressions,
        "improvements": improvements,
        "candidate_retained": True,
        "selection_score": selection_score,
        "selection_method": selection_method,
        "metrics": {
            "world_source": str(
                candidate.get(
                    "world_source", incumbent.get("world_source", "")
                )
            ),
            "incumbent_termination_rate": incumbent_termination,
            "candidate_termination_rate": candidate_termination,
            "incumbent_physical_failure_rate": incumbent_termination,
            "candidate_physical_failure_rate": candidate_termination,
            "incumbent_mean_tilt": old_tilt,
            "candidate_mean_tilt": new_tilt,
            "incumbent_mean_root_height": old_height,
            "candidate_mean_root_height": new_height,
            "incumbent_mean_peak_forward_progress_m": old_forward,
            "candidate_mean_peak_forward_progress_m": new_forward,
            "incumbent_mean_final_forward_progress_m": float(
                incumbent.get(
                    "mean_final_forward_progress_m", old_forward
                )
            ),
            "candidate_mean_final_forward_progress_m": float(
                candidate.get(
                    "mean_final_forward_progress_m", new_forward
                )
            ),
            "incumbent_mean_task_reward": float(
                incumbent.get("mean_task_reward", 0)
            ),
            "candidate_mean_task_reward": float(
                candidate.get("mean_task_reward", 0)
            ),
            "incumbent_mean_reward": float(
                incumbent.get("mean_reward", 0)
            ),
            "candidate_mean_reward": float(
                candidate.get("mean_reward", 0)
            ),
            "incumbent_mean_tracking_score": float(
                incumbent.get("mean_tracking_score", 0)
            ),
            "candidate_mean_tracking_score": float(
                candidate.get("mean_tracking_score", 0)
            ),
        },
    }


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def select_candidate_champion(
    incumbent: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Select a checkpoint without making locomotion order-dependent."""

    continuous_comparison = _uses_generic_task_outcome(incumbent) or (
        str(incumbent.get("task", "")) == "velocity"
        and bool(incumbent.get("forward_progress_available"))
        and all(
            bool(record.get("forward_progress_available"))
            for record in candidates.values()
        )
    )
    comparisons: dict[str, dict[str, Any]] = {}
    if continuous_comparison:
        for name, record in candidates.items():
            comparisons[name] = compare_evidence(incumbent, record)
        eligible = [
            name
            for name in candidates
            if comparisons[name]["selected"] == "candidate"
        ]
        champion = (
            max(
                eligible,
                key=lambda name: float(
                    comparisons[name].get("selection_score") or 0.0
                ),
            )
            if eligible
            else "incumbent"
        )
        return champion, comparisons

    champion = "incumbent"
    records = {"incumbent": incumbent, **candidates}
    for name in candidates:
        comparison = compare_evidence(records[champion], records[name])
        comparisons[name] = comparison
        if comparison["selected"] == "candidate":
            champion = name
    return champion, comparisons


def _evaluate(
    evaluator: Path, arguments: list[str], evidence_path: Path
) -> dict[str, Any]:
    completed = subprocess.run(
        [str(evaluator), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    evidence_path.write_text(completed.stdout, encoding="utf-8")
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--metallib", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument(
        "--candidate", type=Path, action="append", required=True
    )
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument("--maximum-environments", type=int, default=256)
    parser.add_argument("--held-out-seed", type=int, default=2_650_443_581)
    parser.add_argument("--evaluation-steps", type=int)
    parser.add_argument("training_arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.maximum_environments <= 0:
        parser.error("--maximum-environments must be positive")
    if options.evaluation_steps is not None and options.evaluation_steps <= 0:
        parser.error("--evaluation-steps must be positive")
    training_arguments = options.training_arguments
    if training_arguments[:1] == ["--"]:
        training_arguments = training_arguments[1:]

    options.evidence_directory.mkdir(parents=True, exist_ok=True)
    # Deployment is safe even if evaluation itself is interrupted or fails.
    _atomic_copy(options.incumbent, options.deployment)
    candidate_policies: list[Path] = []
    if options.checkpoint_directory is not None:
        candidate_policies.extend(sorted(
            options.checkpoint_directory.glob(
                "revision-*.candidate.policypack"
            )
        ))
    candidate_policies.extend(options.candidate)
    candidate_policies = list(dict.fromkeys(candidate_policies))

    records: dict[str, dict[str, Any]] = {}
    policies: dict[str, Path] = {"incumbent": options.incumbent}
    if len(candidate_policies) == 1:
        policies["candidate"] = candidate_policies[0]
    else:
        for index, policy in enumerate(candidate_policies):
            revision = policy.name.removeprefix("revision-").split(".")[0]
            label = (
                f"candidate-{revision}"
                if revision.isdigit()
                else f"candidate-{index:03d}"
            )
            policies[label] = policy
    try:
        for name, policy in policies.items():
            records[name] = _evaluate(
                options.evaluator,
                evaluation_arguments(
                    training_arguments,
                    policy_pack=policy,
                    metallib=options.metallib,
                    state_trace=(
                        options.evidence_directory / f"{name}.state.tsv"
                    ),
                    maximum_environments=options.maximum_environments,
                    held_out_seed=options.held_out_seed,
                    evaluation_steps=options.evaluation_steps,
                ),
                options.evidence_directory / f"{name}.evidence.json",
            )
    except Exception as error:
        failure = {
            "schema": "numi.policy-selection.v1",
            "selected": "incumbent",
            "candidate_advanced_deployment": False,
            "candidate_retained": True,
            "selection_error": str(error),
        }
        encoded = json.dumps(failure, indent=2, sort_keys=True) + "\n"
        (options.evidence_directory / "selection.json").write_text(
            encoded, encoding="utf-8"
        )
        print(encoded, end="")
        return 1
    candidate_names = [name for name in policies if name != "incumbent"]
    champion, comparisons = select_candidate_champion(
        records["incumbent"],
        {name: records[name] for name in candidate_names},
    )

    reported_candidate = (
        champion if champion != "incumbent" else candidate_names[-1]
    )
    decision = compare_evidence(
        records["incumbent"], records[reported_candidate]
    )
    if champion == "incumbent":
        decision["selected"] = "incumbent"
        decision["candidate_advanced_deployment"] = False
    decision.update(
        {
            "incumbent_policy_pack": str(options.incumbent),
            "candidate_policy_pack": str(policies[reported_candidate]),
            "deployment_policy_pack": str(options.deployment),
            "maximum_evaluation_environments": options.maximum_environments,
            "held_out_seed": options.held_out_seed,
            "evaluated_candidate_policy_packs": [
                str(policies[name]) for name in candidate_names
            ],
            "selected_candidate_label": champion
            if champion != "incumbent" else None,
            "checkpoint_comparisons": comparisons,
        }
    )
    if decision["candidate_advanced_deployment"]:
        _atomic_copy(policies[champion], options.deployment)
    encoded = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    (options.evidence_directory / "selection.json").write_text(
        encoded, encoding="utf-8"
    )
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
