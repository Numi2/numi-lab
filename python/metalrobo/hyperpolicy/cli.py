"""Production command line for ARDY motion-to-hyper-policy compilation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence

import numpy as np

from .base import HyperBasePolicy
from .bundle import MotionPolicyBundle
from .canonical_io import read_canonical_motion, write_canonical_motion
from .generated import GeneratedMotionPolicy, build_generated_motion_policy
from .motion import CanonicalARDYMotion
from .native_pack import write_native_hyper_policy_pack


def _json_file(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _word_fingerprint(path: Path) -> int:
    digest = hashlib.sha256(path.read_bytes()).digest()
    return int.from_bytes(digest[:8], "little") or 1


def _parse_indices(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or min(result) < 0:
        raise argparse.ArgumentTypeError(
            "contact group indices must be non-negative comma-separated integers"
        )
    return result


def _parse_ranks(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or min(result) <= 0:
        raise argparse.ArgumentTypeError(
            "adapter ranks must be positive comma-separated integers"
        )
    return result


def _find_proposal(directory: Path) -> Path:
    candidates = [
        path.parent
        for path in directory.rglob("motion_proposal.npz")
        if (path.parent / "evidence.json").is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one ARDY proposal below "
            f"{directory}, found {len(candidates)}"
        )
    return candidates[0]


def _imagine(
    *,
    prompt: str,
    output_directory: Path,
    seed: int,
    provider: str,
    model_family: str,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    command = [
        "numi",
        "motion",
        "imagine-g1",
        "--prompt",
        prompt,
        "--output-directory",
        str(output_directory),
        "--seed",
        str(seed),
        "--provider",
        provider,
        "--model-family",
        model_family,
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    (output_directory / "ardy.stdout.log").write_text(completed.stdout)
    (output_directory / "ardy.stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ARDY generation failed; see {output_directory / 'ardy.stderr.log'}"
        )
    return _find_proposal(output_directory)


def _canonicalize(
    *,
    proposal_directory: Path,
    g1_urdf: Path,
    output_directory: Path,
    target_fps: float,
) -> tuple[CanonicalARDYMotion, dict[str, np.ndarray], dict[str, Any]]:
    from ..ardy_g1 import native_g1_mechanism
    from ..g1_motion_retarget import retarget_g1

    source_evidence = json.loads(
        (proposal_directory / "evidence.json").read_text(encoding="utf-8")
    )
    skeleton = source_evidence.get("skeleton", {})
    skeleton_id = skeleton.get("id") if isinstance(skeleton, dict) else None
    if skeleton_id == "g1skel34":
        arrays, evidence = native_g1_mechanism(proposal_directory, g1_urdf)
    elif skeleton_id == "cskel27":
        arrays, evidence = retarget_g1(proposal_directory, g1_urdf)
    else:
        raise ValueError(
            f"ARDY proposal skeleton must be g1skel34 or cskel27, got {skeleton_id!r}"
        )
    motion = CanonicalARDYMotion.from_native_g1(
        arrays,
        evidence,
        target_frames_per_second=target_fps,
    )
    write_canonical_motion(motion, output_directory)
    return motion, arrays, evidence


def _g1_action_contract(
    *,
    motion: CanonicalARDYMotion,
    base: HyperBasePolicy,
    native_library: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from ..native import unitree_g1_deployment_contract

    contract = unitree_g1_deployment_contract(native_library)
    default = np.asarray(contract["default_pose"], dtype=np.float32)
    scale = np.asarray(contract["task_action_scale"], dtype=np.float32)
    limits = np.asarray(contract["position_limits"], dtype=np.float32)
    velocity = np.asarray(contract["velocity_limits"], dtype=np.float32)
    if (
        default.shape != (motion.joint_count,)
        or scale.shape != default.shape
        or limits.shape != (motion.joint_count, 2)
        or velocity.shape != default.shape
        or np.any(np.abs(scale) <= 1.0e-8)
    ):
        raise ValueError("native G1 action contract disagrees with ARDY motion")
    task_reference = (motion.joint_positions - default[None, :]) / scale[None, :]
    actor_reference = np.zeros_like(task_reference, dtype=np.float32)
    active = np.abs(base.action_scale) > 1.0e-12
    actor_reference[:, active] = (
        task_reference[:, active] - base.action_bias[None, active]
    ) / base.action_scale[None, active]
    task_lower = (limits[:, 0] - default) / scale
    task_upper = (limits[:, 1] - default) / scale
    lower = np.minimum(task_lower, task_upper).astype(np.float32)
    upper = np.maximum(task_lower, task_upper).astype(np.float32)
    maximum_rate = (velocity / np.abs(scale)).astype(np.float32)
    return actor_reference.astype(np.float32), lower, upper, maximum_rate


def _replace_reference(
    policy: GeneratedMotionPolicy,
    reference_actions: np.ndarray,
) -> GeneratedMotionPolicy:
    result = replace(
        policy,
        reference_actions=np.asarray(reference_actions, dtype=np.float32),
        fingerprint="",
    ).with_fingerprint()
    result.validate(require_fingerprint=True)
    return result


def _build_policy(
    *,
    policy_id: str,
    hyper_base: HyperBasePolicy,
    motion: CanonicalARDYMotion,
    coefficient_knots: np.ndarray,
    coefficient_uncertainty: np.ndarray,
    authority_knots: np.ndarray,
    phase_rate_multiplier: np.ndarray,
    reference_actions: np.ndarray,
    robot_fingerprint: int,
    action_lower: np.ndarray,
    action_upper: np.ndarray,
    maximum_action_rate: np.ndarray,
    failure_probability: float,
    ood_score: float,
) -> GeneratedMotionPolicy:
    generated = build_generated_motion_policy(
        policy_id=policy_id,
        hyper_base=hyper_base,
        motion=motion,
        coefficient_knots=coefficient_knots,
        coefficient_uncertainty=coefficient_uncertainty,
        authority_knots=authority_knots,
        phase_rate_multiplier_knots=phase_rate_multiplier,
        robot_fingerprint=robot_fingerprint,
        world_fingerprint=hyper_base.world_fingerprint,
        action_lower=action_lower,
        action_upper=action_upper,
        maximum_action_rate=maximum_action_rate,
        predicted_failure_probability=failure_probability,
        predicted_out_of_distribution_score=ood_score,
    )
    return _replace_reference(generated, reference_actions)


def _load_compiler(checkpoint: Path):
    from .checkpoint import read_compiler_checkpoint
    from .mlx_compiler import ARDYHyperPolicyCompiler

    network, base, configuration, schema, manifest = read_compiler_checkpoint(
        checkpoint
    )
    return (
        ARDYHyperPolicyCompiler(network, base),
        network,
        base,
        configuration,
        schema,
        manifest,
    )


def _bind_base_to_interaction_task(
    *,
    base: HyperBasePolicy,
    evaluator: Path,
    metallib: Path,
    interaction_pack: Path,
    interaction_clip: str,
    task: str,
    scene: str,
    seed: int,
    output_directory: Path,
) -> HyperBasePolicy:
    """Explicitly bind a reusable actor base to one compiled InteractionPack."""

    output_directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(evaluator),
        "--metallib",
        str(metallib),
        "--task",
        task,
        "--scene",
        scene,
        "--interaction-pack",
        str(interaction_pack),
        "--interaction-clip",
        interaction_clip,
        "--interaction-student-authority",
        "1",
        "--interaction-reset-phase-probability",
        "0",
        "--interaction-reset-maximum-phase",
        "0",
        "--zero-actions",
        "--envs",
        "1",
        "--steps",
        "1",
        "--repeats",
        "1",
        "--chunk",
        "1",
        "--seed",
        str(seed),
    ]
    _json_file(output_directory / "arguments.json", command)
    completed = subprocess.run(command, text=True, capture_output=True)
    (output_directory / "stdout.log").write_text(completed.stdout)
    (output_directory / "stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            "InteractionPack contract preflight failed; see "
            f"{output_directory / 'stderr.log'}"
        )
    try:
        record = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "contract preflight output was not one JSON record"
        ) from error
    _json_file(output_directory / "evidence.json", record)
    resolved = {
        "world_fingerprint": int(record["world_fingerprint"]),
        "task_fingerprint": int(record["task_fingerprint"]),
        "observation_fingerprint": int(record["observation_fingerprint"]),
        "action_fingerprint": int(record["action_fingerprint"]),
        "actor_observation_count": int(record["actor_observation_count"]),
        "action_count": int(record["action_count"]),
    }
    if (
        resolved["world_fingerprint"] != base.world_fingerprint
        or resolved["observation_fingerprint"] != base.observation_fingerprint
        or resolved["action_fingerprint"] != base.action_fingerprint
        or resolved["actor_observation_count"] != base.observation_count
        or resolved["action_count"] != base.action_count
        or resolved["task_fingerprint"] <= 0
    ):
        raise ValueError(
            "InteractionPack changes the actor's world, observation, "
            "or action semantics"
        )
    rebound = replace(
        base,
        task_fingerprint=resolved["task_fingerprint"],
        fingerprint="",
    ).with_fingerprint()
    _json_file(
        output_directory / "binding.json",
        {
            "source_task_fingerprint": base.task_fingerprint,
            "resolved_task_fingerprint": rebound.task_fingerprint,
            "source_hyper_base_fingerprint": base.fingerprint,
            "resolved_hyper_base_fingerprint": rebound.fingerprint,
        },
    )
    return rebound


def _qualify(
    *,
    evaluator: Path,
    metallib: Path,
    native_pack: Path,
    interaction_pack: Path,
    interaction_clip: str,
    output_directory: Path,
    environments: int,
    steps: int,
    chunk: int,
    seed: int,
    task: str,
    scene: str,
) -> tuple[dict[str, Any], Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    rollout = output_directory / "rollout.rolloutpack"
    command = [
        str(evaluator),
        "--metallib",
        str(metallib),
        "--task",
        task,
        "--scene",
        scene,
        "--interaction-pack",
        str(interaction_pack),
        "--interaction-clip",
        interaction_clip,
        "--interaction-student-authority",
        "1",
        "--interaction-reset-phase-probability",
        "0",
        "--interaction-reset-maximum-phase",
        "0",
        "--hyper-policy-pack",
        str(native_pack),
        "--envs",
        str(environments),
        "--steps",
        str(steps),
        "--repeats",
        "1",
        "--chunk",
        str(max(1, min(chunk, steps))),
        "--seed",
        str(seed),
        "--rollout-pack",
        str(rollout),
    ]
    _json_file(output_directory / "arguments.json", command)
    completed = subprocess.run(command, text=True, capture_output=True)
    (output_directory / "stdout.log").write_text(completed.stdout)
    (output_directory / "stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"HyperPolicy qualification failed; see {output_directory / 'stderr.log'}"
        )
    try:
        record = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("qualification output was not one JSON record") from error
    _json_file(output_directory / "evidence.json", record)
    if not rollout.is_file():
        raise RuntimeError("qualification did not publish a PolicyRolloutPack")
    return record, rollout


def _quality(
    record: dict[str, Any],
    *,
    environments: int,
    minimum_tracking: float,
    maximum_termination_rate: float,
) -> tuple[bool, float, dict[str, float]]:
    failed_steps = int(record.get("failed_environment_steps", 0))
    tracking_values = np.asarray(
        record.get("mean_tracking_score_by_environment", ()),
        dtype=np.float64,
    )
    termination_values = np.asarray(
        record.get("termination_count_by_environment", ()),
        dtype=np.float64,
    )
    physics_values = np.asarray(
        record.get("physics_error_count_by_environment", ()),
        dtype=np.float64,
    )
    if tracking_values.shape != (environments,):
        mean_tracking = float(record.get("mean_tracking_score", 0.0))
    else:
        mean_tracking = float(np.mean(tracking_values))
    termination_events = (
        float(np.sum(termination_values)) if termination_values.size else 0.0
    )
    terminated_environments = (
        float(np.count_nonzero(termination_values)) if termination_values.size else 0.0
    )
    physics_errors = float(np.sum(physics_values)) if physics_values.size else 0.0
    termination_rate = terminated_environments / max(float(environments), 1.0)
    accepted = (
        failed_steps == 0
        and physics_errors == 0.0
        and mean_tracking >= minimum_tracking
        and termination_rate <= maximum_termination_rate
    )
    score = (
        mean_tracking
        - 2.0 * termination_rate
        - 1000.0 * float(failed_steps != 0)
        - 1000.0 * float(physics_errors != 0.0)
    )
    return (
        accepted,
        score,
        {
            "mean_tracking_score": mean_tracking,
            "termination_rate": termination_rate,
            "termination_events": termination_events,
            "physics_errors": physics_errors,
            "failed_environment_steps": float(failed_steps),
        },
    )


def _publish_candidate(
    *,
    directory: Path,
    base: HyperBasePolicy,
    policy: GeneratedMotionPolicy,
    contact_group_indices: Sequence[int],
) -> tuple[Path, Path]:
    bundle_directory = directory / "bundle"
    MotionPolicyBundle(base.with_fingerprint(), policy).write(bundle_directory)
    native_pack = directory / "policy.hyperpolicypack"
    write_native_hyper_policy_pack(
        native_pack,
        hyper_base=base,
        motion_policy=policy,
        contact_group_indices=contact_group_indices,
    )
    return bundle_directory, native_pack


def _initialize(arguments: argparse.Namespace) -> int:
    from .checkpoint import write_compiler_checkpoint
    from .mlx_bridge import initialize_hyper_base_from_policy_pack
    from .mlx_production_model import ARDYHyperNetwork, ARDYHyperPolicyConfiguration

    motion = read_canonical_motion(arguments.canonical_motion)
    base = initialize_hyper_base_from_policy_pack(
        arguments.policy_pack,
        ranks=arguments.ranks,
        coefficient_limit=arguments.coefficient_limit,
        seed=arguments.seed,
    )
    configuration = ARDYHyperPolicyConfiguration(
        feature_count=motion.features.shape[1],
        action_count=base.action_count,
        coefficient_count=base.coefficient_count,
        event_feature_count=motion.knot_event_features.shape[1],
        coefficient_limits=tuple(float(value) for value in base.coefficient_limits),
        seed=arguments.seed,
    )
    network = ARDYHyperNetwork(configuration)
    write_compiler_checkpoint(
        arguments.output,
        hypernetwork=network,
        hyper_base=base,
        configuration=configuration,
        feature_schema=motion.feature_schema,
        training_updates=0,
        metrics={"initialized": 1.0},
    )
    print(json.dumps({"checkpoint": str(arguments.output), "trained": False}))
    return 0


def _canonicalize_command(arguments: argparse.Namespace) -> int:
    motion, _, evidence = _canonicalize(
        proposal_directory=arguments.proposal_directory,
        g1_urdf=arguments.g1_urdf,
        output_directory=arguments.output,
        target_fps=arguments.target_fps,
    )
    print(
        json.dumps(
            {
                "canonical_motion": str(arguments.output),
                "source_fingerprint": motion.source_fingerprint,
                "frame_count": motion.frame_count,
                "event_count": len(motion.events),
                "feature_count": motion.features.shape[1],
                "source": evidence["source_motion"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _compile_command(arguments: argparse.Namespace) -> int:
    compiler, _, base, configuration, schema, manifest = _load_compiler(
        arguments.checkpoint
    )
    motion = read_canonical_motion(arguments.canonical_motion)
    if tuple(motion.feature_schema) != tuple(schema):
        raise ValueError("canonical motion feature semantics differ from checkpoint")
    distribution = compiler.generate_distribution(motion)
    reference, lower, upper, rate = _g1_action_contract(
        motion=motion,
        base=base,
        native_library=arguments.native_library,
    )
    policy = _build_policy(
        policy_id=arguments.id,
        hyper_base=base,
        motion=motion,
        coefficient_knots=distribution.coefficient_mean,
        coefficient_uncertainty=distribution.coefficient_uncertainty,
        authority_knots=distribution.authority,
        phase_rate_multiplier=distribution.phase_rate_multiplier,
        reference_actions=reference,
        robot_fingerprint=arguments.robot_fingerprint,
        action_lower=lower,
        action_upper=upper,
        maximum_action_rate=rate,
        failure_probability=distribution.failure_probability,
        ood_score=distribution.out_of_distribution_score,
    )
    bundle, native = _publish_candidate(
        directory=arguments.output,
        base=base,
        policy=policy,
        contact_group_indices=arguments.contact_group_indices,
    )
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "native_pack": str(native),
                "training_updates": manifest["training_updates"],
                "predicted_failure_probability": policy.predicted_failure_probability,
                "predicted_out_of_distribution_score": (
                    policy.predicted_out_of_distribution_score
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _meta_update_command(arguments: argparse.Namespace) -> int:
    from .checkpoint import read_compiler_checkpoint, write_compiler_checkpoint
    from .training_pipeline import (
        MetaTrainingItem,
        meta_update,
        read_specialist_program,
    )
    from ..mlx_policy_learning import read_policy_rollout_pack

    network, base, configuration, schema, checkpoint = read_compiler_checkpoint(
        arguments.checkpoint
    )
    specification = json.loads(arguments.items.read_text())
    items: list[MetaTrainingItem] = []
    for record in specification["items"]:
        motion = read_canonical_motion(Path(record["canonical_motion"]))
        if tuple(motion.feature_schema) != tuple(schema):
            raise ValueError("meta-training motion feature schema differs")
        bundle = MotionPolicyBundle.read(Path(record["bundle"]))
        specialist, metadata = read_specialist_program(Path(record["specialist"]))
        if metadata["motion_fingerprint"] != motion.source_fingerprint:
            raise ValueError("specialist and canonical motion differ")
        rollout = read_policy_rollout_pack(
            Path(record["rollout"]),
            library_path=arguments.native_library,
        )
        items.append(
            MetaTrainingItem(
                motion=motion,
                generated_policy=bundle.motion_policy,
                specialist=specialist,
                rollout=rollout,
                qualified=bool(record.get("qualified", True)),
            )
        )
    network, base, metrics = meta_update(
        hypernetwork=network,
        hyper_base=base,
        configuration=configuration,
        items=items,
        updates=arguments.updates,
    )
    write_compiler_checkpoint(
        arguments.output,
        hypernetwork=network,
        hyper_base=base,
        configuration=configuration,
        feature_schema=schema,
        training_updates=int(checkpoint["training_updates"]) + arguments.updates,
        metrics=metrics,
    )
    print(
        json.dumps({"checkpoint": str(arguments.output), "metrics": metrics}, indent=2)
    )
    return 0


def _create(arguments: argparse.Namespace) -> int:
    from .training_pipeline import (
        generated_program,
        train_specialist_from_rollout,
        write_specialist_program,
    )
    from ..ardy_g1 import write_native_g1_interaction_pack
    from ..mlx_policy_learning import read_policy_rollout_pack

    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if arguments.repair_updates < 0:
        raise ValueError("repair update count must be non-negative")
    if arguments.repair_rollout_pack is not None and arguments.repair_updates == 0:
        raise ValueError(
            "--repair-rollout-pack requires a positive --repair-updates value"
        )
    proposal = arguments.proposal_directory
    if proposal is None:
        proposal = _imagine(
            prompt=arguments.prompt,
            output_directory=output / "ardy",
            seed=arguments.ardy_seed,
            provider=arguments.provider,
            model_family=arguments.model_family,
        )
    motion, arrays, motion_evidence = _canonicalize(
        proposal_directory=proposal,
        g1_urdf=arguments.g1_urdf,
        output_directory=output / "canonical-motion",
        target_fps=arguments.target_fps,
    )
    compiler, _, base, configuration, schema, checkpoint = _load_compiler(
        arguments.checkpoint
    )
    if tuple(motion.feature_schema) != tuple(schema):
        raise ValueError("ARDY motion feature schema differs from compiler checkpoint")
    if int(checkpoint["training_updates"]) == 0 and not arguments.allow_untrained:
        raise ValueError(
            "compiler checkpoint has no training updates; pass "
            "--allow-untrained only for integration work"
        )
    if motion.joint_count != base.action_count:
        raise ValueError("ARDY G1 and hyper-base action widths differ")
    robot_fingerprint = _word_fingerprint(arguments.g1_urdf)
    interaction_pack = output / "motion.interactionpack"
    write_native_g1_interaction_pack(
        output=interaction_pack,
        arrays=arrays,
        evidence=motion_evidence,
        desired_outcome=motion.prompt,
        clip_id=arguments.interaction_clip,
    )
    base = _bind_base_to_interaction_task(
        base=base,
        evaluator=arguments.evaluator,
        metallib=arguments.metallib,
        interaction_pack=interaction_pack,
        interaction_clip=arguments.interaction_clip,
        task=arguments.task,
        scene=arguments.scene,
        seed=arguments.qualification_seed,
        output_directory=output / "contract-preflight",
    )
    reference, lower, upper, rate = _g1_action_contract(
        motion=motion,
        base=base,
        native_library=arguments.native_library,
    )
    distribution = compiler.generate_distribution(motion)
    if (
        distribution.failure_probability > arguments.maximum_predicted_failure
        or distribution.out_of_distribution_score > arguments.maximum_ood_score
    ) and not arguments.allow_ood:
        raise RuntimeError(
            "hypernetwork rejected the ARDY motion before physical execution"
        )
    candidates = distribution.candidate_coefficients(
        count=arguments.candidates,
        seed_material=(
            motion.source_fingerprint
            + base.fingerprint
            + str(checkpoint["training_updates"])
        ),
        maximum_standard_deviations=arguments.candidate_radius,
    )
    records: list[dict[str, Any]] = []
    best: tuple[float, GeneratedMotionPolicy, Path, Path, dict[str, Any]] | None = None
    accepted: tuple[GeneratedMotionPolicy, Path, Path, dict[str, Any]] | None = None
    steps = arguments.steps or (
        int(math.ceil(motion.duration_seconds * motion.frames_per_second))
        + arguments.stabilization_steps
    )
    for index, coefficients in enumerate(candidates):
        candidate_dir = output / "candidates" / f"candidate-{index:02d}"
        policy = _build_policy(
            policy_id=f"{arguments.id}-candidate-{index:02d}",
            hyper_base=base,
            motion=motion,
            coefficient_knots=coefficients,
            coefficient_uncertainty=distribution.coefficient_uncertainty,
            authority_knots=distribution.authority,
            phase_rate_multiplier=distribution.phase_rate_multiplier,
            reference_actions=reference,
            robot_fingerprint=robot_fingerprint,
            action_lower=lower,
            action_upper=upper,
            maximum_action_rate=rate,
            failure_probability=distribution.failure_probability,
            ood_score=distribution.out_of_distribution_score,
        )
        bundle, native = _publish_candidate(
            directory=candidate_dir,
            base=base,
            policy=policy,
            contact_group_indices=arguments.contact_group_indices,
        )
        evidence, rollout = _qualify(
            evaluator=arguments.evaluator,
            metallib=arguments.metallib,
            native_pack=native,
            interaction_pack=interaction_pack,
            interaction_clip=arguments.interaction_clip,
            output_directory=candidate_dir / "qualification",
            environments=arguments.environments,
            steps=steps,
            chunk=arguments.chunk,
            seed=arguments.qualification_seed,
            task=arguments.task,
            scene=arguments.scene,
        )
        passed, score, metrics = _quality(
            evidence,
            environments=arguments.environments,
            minimum_tracking=arguments.minimum_tracking,
            maximum_termination_rate=arguments.maximum_termination_rate,
        )
        record = {
            "candidate": index,
            "score": score,
            "passed": passed,
            "metrics": metrics,
            "bundle": str(bundle),
            "native_pack": str(native),
            "rollout": str(rollout),
        }
        records.append(record)
        if best is None or score > best[0]:
            best = (score, policy, native, rollout, evidence)
        if passed:
            accepted = (policy, native, rollout, evidence)
            break
    assert best is not None

    repair_record: dict[str, Any] | None = None
    if accepted is None and arguments.repair_updates > 0:
        if arguments.repair_rollout_pack is None:
            raise ValueError(
                "specialist repair requires --repair-rollout-pack from an "
                "independent solver teacher; a candidate cannot teach itself"
            )
        _, best_policy, _, _, _ = best
        rollout = read_policy_rollout_pack(
            arguments.repair_rollout_pack,
            library_path=arguments.native_library,
        )
        if rollout.task_fingerprint != base.task_fingerprint:
            raise ValueError("repair rollout and compiled InteractionPack task differ")
        repair = train_specialist_from_rollout(
            rollout=rollout,
            motion_policy=best_policy,
            hyper_base=base,
            configuration=configuration,
            updates=arguments.repair_updates,
            initial=generated_program(best_policy),
            learning_rate=arguments.repair_learning_rate,
        )
        program = repair.program
        repaired = _build_policy(
            policy_id=f"{arguments.id}-repaired",
            hyper_base=base,
            motion=motion,
            coefficient_knots=program.coefficients[0],
            coefficient_uncertainty=distribution.coefficient_uncertainty,
            authority_knots=program.authority[0],
            phase_rate_multiplier=program.phase_rate_multiplier[0],
            reference_actions=reference,
            robot_fingerprint=robot_fingerprint,
            action_lower=lower,
            action_upper=upper,
            maximum_action_rate=rate,
            failure_probability=distribution.failure_probability,
            ood_score=distribution.out_of_distribution_score,
        )
        repair_dir = output / "repair"
        bundle, native = _publish_candidate(
            directory=repair_dir,
            base=base,
            policy=repaired,
            contact_group_indices=arguments.contact_group_indices,
        )
        evidence, rollout_path = _qualify(
            evaluator=arguments.evaluator,
            metallib=arguments.metallib,
            native_pack=native,
            interaction_pack=interaction_pack,
            interaction_clip=arguments.interaction_clip,
            output_directory=repair_dir / "qualification",
            environments=arguments.environments,
            steps=steps,
            chunk=arguments.chunk,
            seed=arguments.qualification_seed,
            task=arguments.task,
            scene=arguments.scene,
        )
        passed, score, metrics = _quality(
            evidence,
            environments=arguments.environments,
            minimum_tracking=arguments.minimum_tracking,
            maximum_termination_rate=arguments.maximum_termination_rate,
        )
        specialist_path = write_specialist_program(
            output / "specialist-program.npz",
            program=program,
            motion_fingerprint=motion.source_fingerprint,
            hyper_base_fingerprint=base.fingerprint,
            metrics=repair.metrics,
        )
        repair_record = {
            "score": score,
            "passed": passed,
            "metrics": metrics,
            "training_metrics": repair.metrics,
            "contributing_samples": repair.contributing_samples,
            "specialist": str(specialist_path),
            "teacher_rollout": str(arguments.repair_rollout_pack),
            "rollout": str(rollout_path),
        }
        if passed:
            accepted = (repaired, native, rollout_path, evidence)

    _json_file(
        output / "candidate-selection.evidence.json",
        {
            "format": "numi.ardy-hyperpolicy-selection.v1",
            "source_motion_fingerprint": motion.source_fingerprint,
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_training_updates": checkpoint["training_updates"],
            "candidates": records,
            "repair": repair_record,
        },
    )
    if accepted is None:
        raise RuntimeError(
            "no generated or repaired hyper-policy passed physical qualification"
        )
    policy, native, rollout, evidence = accepted
    final_bundle = output / "deployment.motionpolicy"
    if final_bundle.exists():
        shutil.rmtree(final_bundle)
    MotionPolicyBundle(base.with_fingerprint(), policy).write(final_bundle)
    final_native = output / "deployment.hyperpolicypack"
    shutil.copy2(native, final_native)
    _json_file(
        output / "deployment.evidence.json",
        {
            "status": "verified",
            "bundle": str(final_bundle),
            "native_pack": str(final_native),
            "rollout": str(rollout),
            "source_motion_fingerprint": motion.source_fingerprint,
            "hyper_base_fingerprint": base.fingerprint,
            "policy_fingerprint": policy.fingerprint,
            "solver_evidence": evidence,
        },
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "bundle": str(final_bundle),
                "native_pack": str(final_native),
                "policy_fingerprint": policy.fingerprint,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile ARDY output into a verified event-synchronised HyperPolicy"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    canonical = commands.add_parser("canonicalize")
    canonical.add_argument("--proposal-directory", type=Path, required=True)
    canonical.add_argument("--g1-urdf", type=Path, required=True)
    canonical.add_argument("--output", type=Path, required=True)
    canonical.add_argument("--target-fps", type=float, default=50.0)
    canonical.set_defaults(function=_canonicalize_command)

    initialize = commands.add_parser("initialize-checkpoint")
    initialize.add_argument("--policy-pack", type=Path, required=True)
    initialize.add_argument("--canonical-motion", type=Path, required=True)
    initialize.add_argument("--ranks", type=_parse_ranks, required=True)
    initialize.add_argument("--coefficient-limit", type=float, default=1.0)
    initialize.add_argument("--seed", type=int, default=1)
    initialize.add_argument("--output", type=Path, required=True)
    initialize.set_defaults(function=_initialize)

    compile_command = commands.add_parser("compile")
    compile_command.add_argument("--canonical-motion", type=Path, required=True)
    compile_command.add_argument("--checkpoint", type=Path, required=True)
    compile_command.add_argument("--output", type=Path, required=True)
    compile_command.add_argument("--id", required=True)
    compile_command.add_argument("--robot-fingerprint", type=int, required=True)
    compile_command.add_argument("--native-library", type=Path)
    compile_command.add_argument(
        "--contact-group-indices", type=_parse_indices, default=(0, 1)
    )
    compile_command.set_defaults(function=_compile_command)

    meta = commands.add_parser("meta-update")
    meta.add_argument("--checkpoint", type=Path, required=True)
    meta.add_argument("--items", type=Path, required=True)
    meta.add_argument("--updates", type=int, default=1)
    meta.add_argument("--native-library", type=Path)
    meta.add_argument("--output", type=Path, required=True)
    meta.set_defaults(function=_meta_update_command)

    create = commands.add_parser("create")
    source = create.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--proposal-directory", type=Path)
    create.add_argument("--g1-urdf", type=Path, required=True)
    create.add_argument("--checkpoint", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--id", required=True)
    create.add_argument("--evaluator", type=Path, required=True)
    create.add_argument("--metallib", type=Path, required=True)
    create.add_argument("--native-library", type=Path)
    create.add_argument("--provider", choices=("auto", "coreml", "cpu"), default="auto")
    create.add_argument("--model-family", choices=("g1", "core"), default="g1")
    create.add_argument("--ardy-seed", type=int, default=4)
    create.add_argument("--qualification-seed", type=int, default=2650443581)
    create.add_argument("--target-fps", type=float, default=50.0)
    create.add_argument("--interaction-clip", default="ardy-g1")
    create.add_argument("--task", default="velocity")
    create.add_argument("--scene", default="ground")
    create.add_argument("--environments", type=int, default=1024)
    create.add_argument("--steps", type=int)
    create.add_argument("--stabilization-steps", type=int, default=50)
    create.add_argument("--chunk", type=int, default=16)
    create.add_argument("--candidates", type=int, default=16)
    create.add_argument("--candidate-radius", type=float, default=1.5)
    create.add_argument("--repair-updates", type=int, default=0)
    create.add_argument("--repair-rollout-pack", type=Path)
    create.add_argument("--repair-learning-rate", type=float, default=2.0e-4)
    create.add_argument("--minimum-tracking", type=float, default=0.65)
    create.add_argument("--maximum-termination-rate", type=float, default=0.05)
    create.add_argument("--maximum-predicted-failure", type=float, default=0.50)
    create.add_argument("--maximum-ood-score", type=float, default=5.0)
    create.add_argument("--allow-untrained", action="store_true")
    create.add_argument("--allow-ood", action="store_true")
    create.add_argument("--contact-group-indices", type=_parse_indices, default=(0, 1))
    create.set_defaults(function=_create)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parser = _parser()
    options = parser.parse_args(arguments)
    return int(options.function(options))


if __name__ == "__main__":
    raise SystemExit(main())
