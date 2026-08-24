"""Matched physical-outcome selection for Numi policy candidates.

Training always keeps the candidate.  This module only decides whether the
protected deployment artifact should advance from the pre-training incumbent.
"""

from __future__ import annotations

import argparse
import hashlib
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
        "--robot-actuator-pack",
        "--sensor-program-pack",
        "--reality-program-pack",
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
        "--birdflow-dove",
        "--interaction-reset-only",
        "--materialize-articulated-contact-responses",
    }
)

_GENERIC_WORLD_SOURCES = frozenset({"world_pack", "urdf"})
_EVALUATION_OVERRIDE_OPTIONS = frozenset(
    {
        "--envs",
        "--steps",
        "--seed",
    }
)


def _option_value(arguments: Sequence[str], option: str) -> str | None:
    result = None
    for index, value in enumerate(arguments[:-1]):
        if value == option:
            result = arguments[index + 1]
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evaluation_contract(evaluator: Path, arguments: Sequence[str]) -> str:
    """Fingerprint the exact native rollout contract for resumable evidence."""

    file_fingerprints: dict[str, str] = {}
    for option in ("--metallib", "--policy-pack"):
        value = _option_value(arguments, option)
        if value is None:
            continue
        path = Path(value)
        if path.is_file():
            file_fingerprints[option] = _sha256_file(path)
    contract = {
        "evaluator": str(evaluator),
        "evaluator_sha256": _sha256_file(evaluator),
        "arguments": list(arguments),
        "file_fingerprints": file_fingerprints,
    }
    encoded = json.dumps(contract, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_kind(task_id: str) -> str:
    """Map authored task IDs onto the stable promotion-policy vocabulary."""

    normalized = task_id.strip().lower().replace("_", "-")
    if "adult" in normalized and "locomotion" in normalized:
        return "adult-locomotion"
    if "developmental" in normalized and "recovery" in normalized:
        return "developmental-recovery"
    if "supine" in normalized and "get-up" in normalized:
        return "supine-get-up"
    if "ball" in normalized and "dodge" in normalized:
        return "ball-dodge"
    if "ball" in normalized and (
        "recovery" in normalized or "disturbance" in normalized
    ):
        return "ball-recovery"
    if "disturbance" in normalized or "recovery" in normalized:
        return "disturbance-recovery"
    if "velocity" in normalized or "locomotion" in normalized:
        return "velocity"
    return normalized


def _adult_evaluation_bands(
    training_arguments: Sequence[str],
) -> tuple[int | None, int | None]:
    """Return the current adult rung and its protected predecessor."""

    if _task_kind(_option_value(training_arguments, "--task") or "") != (
        "adult-locomotion"
    ):
        return None, None
    maximum_band = _option_value(
        training_arguments, "--maximum-difficulty-band"
    )
    if maximum_band is None:
        return None, None
    current_band = int(maximum_band)
    return current_band, current_band - 1 if current_band > 0 else None


def evaluation_arguments(
    training_arguments: Sequence[str],
    *,
    policy_pack: Path,
    metallib: Path,
    state_trace: Path,
    maximum_environments: int,
    held_out_seed: int,
    evaluation_steps: int | None = None,
    evaluation_minimum_band: int | None = None,
    evaluation_maximum_band: int | None = None,
) -> list[str]:
    """Project trainer arguments onto one deterministic rollout contract."""
    projected: list[str] = []
    index = 0
    while index < len(training_arguments):
        option = training_arguments[index]
        if option in _VALUE_OPTIONS:
            if index + 1 >= len(training_arguments):
                raise ValueError(f"{option} is missing its value")
            if option not in _EVALUATION_OVERRIDE_OPTIONS:
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
        # Keep the zero authority in the task contract, keep the teacher's
        # accepted reset state, then evaluate autonomously. Removing the
        # option would silently restore the default 0.1 authority and produce
        # a different task fingerprint.
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
        # Preserve the authored interaction task contract.  In particular,
        # `--interaction-reset-only` changes the compiled task fingerprint
        # for physics-gated InteractionPacks, so adding it here would make
        # candidate selection reject the very PolicyPack just trained.
        projected = filtered + [
            "--interaction-student-authority",
            "0",
        ]

    # Adult training deliberately mixes the previous and current bands so
    # the learner retains the earlier balance skill. Promotion evidence must
    # answer a stricter question: did the candidate survive the newest band
    # itself? Appending the maximum band makes the native evaluator's last
    # value authoritative without changing the training rollout contract.
    task = _task_kind(_option_value(training_arguments, "--task") or "")
    maximum_band = _option_value(
        training_arguments, "--maximum-difficulty-band"
    )
    if task == "adult-locomotion" and maximum_band is not None:
        selected_minimum_band = (
            str(evaluation_minimum_band)
            if evaluation_minimum_band is not None
            else maximum_band
        )
        selected_maximum_band = (
            str(evaluation_maximum_band)
            if evaluation_maximum_band is not None
            else maximum_band
        )
        projected.extend(
            (
                "--minimum-difficulty-band",
                selected_minimum_band,
                "--maximum-difficulty-band",
                selected_maximum_band,
            )
        )

    # Appended values win if a caller supplied the same option earlier.
    projected.extend(
        (
            "--envs",
            str(evaluation_environments),
            "--steps",
            str(evaluation_steps or int(_option_value(training_arguments, "--steps") or 1)),
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


def _recovery_phase_rate(
    record: dict[str, Any], phase: str
) -> float | None:
    rates = record.get("recovery_phase_rates")
    if not isinstance(rates, dict) or phase not in rates:
        return None
    return float(rates[phase])


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


def _authored_outcomes(record: dict[str, Any]) -> dict[str, tuple[float, int]]:
    """Return the typed task outcomes emitted by the held-out rollout."""

    encoded = record.get("outcomes")
    if not isinstance(encoded, dict):
        return {}
    outcomes: dict[str, tuple[float, int]] = {}
    for identifier, value in encoded.items():
        if not isinstance(value, dict):
            continue
        try:
            outcomes[str(identifier)] = (
                float(value["mean"]),
                int(value["direction"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return outcomes


def _compare_adult_authored_outcomes(
    incumbent: dict[str, Any], candidate: dict[str, Any]
) -> tuple[list[str], list[str], dict[str, dict[str, float | int]]]:
    """Keep adult survivability outcomes from being traded for motion."""

    old_outcomes = _authored_outcomes(incumbent)
    new_outcomes = _authored_outcomes(candidate)
    guarded = ("contact_reward", "standing_completion", "restoration")
    regressions: list[str] = []
    improvements: list[str] = []
    metrics: dict[str, dict[str, float | int]] = {}
    if not old_outcomes or not new_outcomes:
        regressions.append("adult authored outcomes unavailable")
        return regressions, improvements, metrics

    for identifier in guarded:
        old = old_outcomes.get(identifier)
        new = new_outcomes.get(identifier)
        if old is None or new is None or old[1] != new[1]:
            regressions.append(
                f"adult authored outcome schema missing {identifier}"
            )
            continue
        old_mean, direction = old
        new_mean, _ = new
        tolerance = 1.0e-4 if identifier == "contact_reward" else 1.0e-3
        metrics[f"incumbent_{identifier}"] = {
            "mean": old_mean,
            "direction": direction,
        }
        metrics[f"candidate_{identifier}"] = {
            "mean": new_mean,
            "direction": direction,
        }
        if direction == 1:
            delta = new_mean - old_mean
        elif direction == 2:
            delta = old_mean - new_mean
        else:
            continue
        if delta < -tolerance:
            regressions.append(
                f"adult authored outcome {identifier} decreased"
            )
        elif delta > tolerance:
            improvements.append(
                f"adult authored outcome {identifier} increased"
            )
    return regressions, improvements, metrics


def compare_evidence(
    incumbent: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compare matched physical evidence while retaining useful tradeoffs."""
    task_id = str(candidate.get("task", incumbent.get("task", "unknown")))
    task = _task_kind(task_id)
    generic_task = _uses_generic_task_outcome(candidate) or (
        _uses_generic_task_outcome(incumbent)
    )
    incumbent_termination = _physical_failure_rate(incumbent)
    candidate_termination = _physical_failure_rate(candidate)
    incumbent_clean_horizon = float(
        incumbent.get("clean_horizon_environment_rate", 0.0)
    )
    candidate_clean_horizon = float(
        candidate.get("clean_horizon_environment_rate", 0.0)
    )
    regressions: list[str] = []
    improvements: list[str] = []

    if int(candidate.get("failed_environment_steps", 0)) != 0:
        regressions.append("candidate has failed environment steps")
    if candidate_termination > incumbent_termination + 1.0e-12:
        regressions.append("termination rate increased")
    elif candidate_termination < incumbent_termination - 1.0e-12:
        improvements.append("termination rate decreased")

    if generic_task:
        # A velocity actor commands an ongoing balance/locomotion task.  A
        # marginally lower reset count is not deployable progress if every
        # held-out environment still collapses before its requested horizon.
        # Keep this narrowly scoped to imported velocity tasks: other generic
        # tasks can intentionally terminate on success before the horizon.
        if (
            task == "velocity"
            and "clean_horizon_environment_rate" in candidate
            and candidate_clean_horizon <= 0.0
        ):
            regressions.append("candidate completed no clean horizon")
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
    elif task in {"supine-get-up", "developmental-recovery"}:
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
        # Recovery is staged: a later foothold or rise gain cannot buy a
        # collapse in an earlier physical capability. Keep every authored
        # phase monotonic when both matched rollouts publish that phase.
        for phase in (
            "brace",
            "foot_support",
            "rise",
            "support_transfer",
            "trunk_clear",
            "quiet_stand",
        ):
            old_phase = _recovery_phase_rate(incumbent, phase)
            new_phase = _recovery_phase_rate(candidate, phase)
            if old_phase is None or new_phase is None:
                continue
            if new_phase < old_phase - 0.005:
                regressions.append(f"{phase} rate decreased")
            elif new_phase > old_phase + 0.005:
                improvements.append(f"{phase} rate increased")
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

    authored_outcome_metrics: dict[str, dict[str, float | int]] = {}
    if task == "adult-locomotion":
        authored_regressions, authored_improvements, authored_outcome_metrics = (
            _compare_adult_authored_outcomes(incumbent, candidate)
        )
        regressions.extend(authored_regressions)
        improvements.extend(authored_improvements)

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
        "adult-locomotion",
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
            and not (
                task == "velocity"
                and "clean_horizon_environment_rate" in candidate
                and candidate_clean_horizon <= 0.0
            )
            and selection_score > 1.0e-12
        )
        selection_method = "continuous_authored_task_outcome"
    elif (
        task in {"velocity", "adult-locomotion"}
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
        if task == "adult-locomotion":
            # Terminations are counts accumulated over a long held-out
            # rollout, not probabilities. Keep their influence bounded so a
            # stationary policy cannot buy promotion by ending fewer episodes.
            failure_progress = _relative_progress(
                candidate_termination, incumbent_termination, 1.0
            )
            selection_score = (
                0.32 * peak_progress
                + 0.32 * final_progress
                + 0.16 * tracking_progress
                + 0.10 * height_progress
                + 0.05 * tilt_progress
                + 0.05 * failure_progress
            )
            # Adult promotion is a monotonic curriculum gate: survivability
            # and authored standing outcomes must not be purchased by losing
            # the locomotion capability learned at the previous band.
            selected = (
                int(candidate.get("failed_environment_steps", 0)) == 0
                and not regressions
                and selection_score > 1.0e-12
            )
            selection_method = "adult_locomotion_physical_comparison"
        else:
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
            "incumbent_clean_horizon_environment_rate":
                incumbent_clean_horizon,
            "candidate_clean_horizon_environment_rate":
                candidate_clean_horizon,
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
            "adult_authored_outcomes": authored_outcome_metrics,
        },
    }


def compare_adult_bands(
    current_incumbent: dict[str, Any],
    current_candidate: dict[str, Any],
    previous_incumbent: dict[str, Any],
    previous_candidate: dict[str, Any],
) -> dict[str, Any]:
    """Require new-band progress without regressing the previous rung."""

    decision = compare_evidence(current_incumbent, current_candidate)
    previous = compare_evidence(previous_incumbent, previous_candidate)
    decision["previous_band_comparison"] = previous
    if previous["regressions"]:
        decision["regressions"].extend(
            f"previous-band: {reason}"
            for reason in previous["regressions"]
        )
        decision["selected"] = "incumbent"
        decision["candidate_advanced_deployment"] = False
    return decision


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
    *,
    comparison_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Select a checkpoint without making locomotion order-dependent."""

    incumbent_task = _task_kind(str(incumbent.get("task", "")))
    continuous_comparison = _uses_generic_task_outcome(incumbent) or (
        incumbent_task in {
            "velocity",
            "adult-locomotion",
        }
        and bool(incumbent.get("forward_progress_available"))
        and all(
            bool(record.get("forward_progress_available"))
            for record in candidates.values()
        )
    )
    comparisons: dict[str, dict[str, Any]] = {}
    if continuous_comparison:
        for name, record in candidates.items():
            comparisons[name] = (
                comparison_overrides[name]
                if comparison_overrides is not None
                and name in comparison_overrides
                else compare_evidence(incumbent, record)
            )
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
        comparison = (
            comparison_overrides[name]
            if comparison_overrides is not None
            and name in comparison_overrides
            else compare_evidence(records[champion], records[name])
        )
        comparisons[name] = comparison
        if comparison["selected"] == "candidate":
            champion = name
    return champion, comparisons


def _evaluate(
    evaluator: Path, arguments: list[str], evidence_path: Path
) -> dict[str, Any]:
    metadata_path = evidence_path.with_name(
        f"{evidence_path.name}.meta.json"
    )
    contract = _evaluation_contract(evaluator, arguments)
    state_trace_value = _option_value(arguments, "--state-trace")
    state_trace = Path(state_trace_value) if state_trace_value else None
    if evidence_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("contract_sha256") == contract
                and state_trace is not None
                and state_trace.is_file()
                and state_trace.stat().st_size > 0
            ):
                return json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            # A partial cache is treated as a miss and regenerated below.
            pass
    completed = subprocess.run(
        [str(evaluator), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    evidence_path.write_text(completed.stdout, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "numi.policy-evidence-cache.v1",
                "contract_sha256": contract,
                "state_trace": state_trace_value,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
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
    parser.add_argument(
        "--advance-candidate",
        action="store_true",
        help=(
            "deploy the latest candidate after recording held-out comparison "
            "metrics instead of using the comparison as an advancement gate"
        ),
    )
    parser.add_argument("training_arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.maximum_environments <= 0:
        parser.error("--maximum-environments must be positive")
    if options.evaluation_steps is not None and options.evaluation_steps <= 0:
        parser.error("--evaluation-steps must be positive")
    training_arguments = options.training_arguments
    if training_arguments[:1] == ["--"]:
        training_arguments = training_arguments[1:]

    adult_current_band, adult_previous_band = _adult_evaluation_bands(
        training_arguments
    )

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
    previous_band_records: dict[str, dict[str, Any]] = {}
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
            if adult_previous_band is not None:
                previous_band_records[name] = _evaluate(
                    options.evaluator,
                    evaluation_arguments(
                        training_arguments,
                        policy_pack=policy,
                        metallib=options.metallib,
                        state_trace=(
                            options.evidence_directory
                            / f"{name}.previous-band.state.tsv"
                        ),
                        maximum_environments=options.maximum_environments,
                        held_out_seed=options.held_out_seed,
                        evaluation_steps=options.evaluation_steps,
                        evaluation_minimum_band=adult_previous_band,
                        evaluation_maximum_band=adult_previous_band,
                    ),
                    options.evidence_directory
                    / f"{name}.previous-band.evidence.json",
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
    comparison_overrides: dict[str, dict[str, Any]] | None = None
    if adult_previous_band is not None:
        comparison_overrides = {}
        for name in candidate_names:
            comparison_overrides[name] = compare_adult_bands(
                records["incumbent"],
                records[name],
                previous_band_records["incumbent"],
                previous_band_records[name],
            )
    champion, comparisons = select_candidate_champion(
        records["incumbent"],
        {name: records[name] for name in candidate_names},
        comparison_overrides=comparison_overrides,
    )

    reported_candidate = (
        champion if champion != "incumbent" else candidate_names[-1]
    )
    deployment_candidate = (
        candidate_names[-1] if options.advance_candidate else champion
    )
    # Copy the selected comparison before attaching the complete comparison
    # table. Reusing the dictionary here would make checkpoint_comparisons
    # contain itself and fail JSON publication with a circular-reference
    # error after all expensive rollouts had already completed.
    decision = dict(comparisons[reported_candidate])
    if options.advance_candidate:
        # Exploration is continuous for this route: held-out rollouts remain
        # immutable diagnostics, but they do not veto the learner's next
        # physical policy.  This preserves every regression signal without
        # turning the selector into a training gate.
        decision = dict(comparisons[deployment_candidate])
        decision["selected"] = deployment_candidate
        decision["candidate_advanced_deployment"] = True
        decision["selection_method"] = "continuous_candidate"
    elif champion == "incumbent":
        decision["selected"] = "incumbent"
        decision["candidate_advanced_deployment"] = False
    decision.update(
        {
            "incumbent_policy_pack": str(options.incumbent),
            "candidate_policy_pack": str(policies[reported_candidate]),
            "deployment_policy_pack": str(options.deployment),
            "maximum_evaluation_environments": options.maximum_environments,
            "held_out_seed": options.held_out_seed,
            "adult_current_band": adult_current_band,
            "adult_previous_band": adult_previous_band,
            "evaluated_candidate_policy_packs": [
                str(policies[name]) for name in candidate_names
            ],
            "selected_candidate_label": deployment_candidate
            if deployment_candidate != "incumbent" else None,
            "comparison_champion": champion,
            "checkpoint_comparisons": comparisons,
        }
    )
    if decision["candidate_advanced_deployment"]:
        _atomic_copy(policies[deployment_candidate], options.deployment)
    encoded = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    (options.evidence_directory / "selection.json").write_text(
        encoded, encoding="utf-8"
    )
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
