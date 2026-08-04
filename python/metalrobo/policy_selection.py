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
        "--visual-environment-pack",
    }
)
_FLAG_OPTIONS = frozenset(
    {
        "--interaction-reset-only",
        "--materialize-articulated-contact-responses",
    }
)


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


def compare_evidence(
    incumbent: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Return a conservative Pareto selection without discarding progress."""
    task = str(candidate.get("task", incumbent.get("task", "unknown")))
    environment_count = max(
        len(candidate.get("termination_count_by_environment", [])), 1
    )
    incumbent_termination = (
        float(incumbent.get("termination_count", 0)) / environment_count
    )
    candidate_termination = (
        float(candidate.get("termination_count", 0)) / environment_count
    )
    regressions: list[str] = []
    improvements: list[str] = []

    if int(candidate.get("failed_environment_steps", 0)) != 0:
        regressions.append("candidate has failed environment steps")
    if candidate_termination > incumbent_termination + 1.0e-12:
        regressions.append("termination rate increased")
    elif candidate_termination < incumbent_termination - 1.0e-12:
        improvements.append("termination rate decreased")

    if task == "ball-dodge":
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
        old_tracking = float(incumbent.get("mean_tracking_score", 0))
        new_tracking = float(candidate.get("mean_tracking_score", 0))
        if new_tracking < old_tracking - 0.001:
            regressions.append("tracking score decreased")
        elif new_tracking > old_tracking + 0.001:
            improvements.append("tracking score increased")

    old_tilt = float(incumbent.get("mean_tilt", 0))
    new_tilt = float(candidate.get("mean_tilt", 0))
    if new_tilt > old_tilt + 0.005:
        regressions.append("mean tilt increased")
    elif new_tilt < old_tilt - 0.001:
        improvements.append("mean tilt decreased")

    selected = not regressions and bool(improvements)
    return {
        "schema": "numi.policy-selection.v1",
        "task": task,
        "selected": "candidate" if selected else "incumbent",
        "candidate_advanced_deployment": selected,
        "regressions": regressions,
        "improvements": improvements,
        "candidate_retained": True,
        "metrics": {
            "incumbent_termination_rate": incumbent_termination,
            "candidate_termination_rate": candidate_termination,
            "incumbent_mean_tilt": old_tilt,
            "candidate_mean_tilt": new_tilt,
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
    parser.add_argument("--candidate", type=Path, required=True)
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
    records: dict[str, dict[str, Any]] = {}
    try:
        for name, policy in (
            ("incumbent", options.incumbent),
            ("candidate", options.candidate),
        ):
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
    decision = compare_evidence(records["incumbent"], records["candidate"])
    decision.update(
        {
            "incumbent_policy_pack": str(options.incumbent),
            "candidate_policy_pack": str(options.candidate),
            "deployment_policy_pack": str(options.deployment),
            "maximum_evaluation_environments": options.maximum_environments,
            "held_out_seed": options.held_out_seed,
        }
    )
    if decision["candidate_advanced_deployment"]:
        _atomic_copy(options.candidate, options.deployment)
    encoded = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    (options.evidence_directory / "selection.json").write_text(
        encoded, encoding="utf-8"
    )
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
