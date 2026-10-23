#!/usr/bin/env python3
"""CPU-only paired evidence audit for the native 006 elbow-flexor trial.

This reads already-captured stdout/invocation receipts, decodes the pinned NHMYO2
owner parameters, and evaluates the archived MyoSim source model. It never
launches the native binary or changes source/runtime physics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

REPO = Path("/Users/n/numi-human-aortic-contact-opt-20261019")
DEFAULT_EVIDENCE = REPO / (
    "matter/tools/evidence/human-resting-20261005/"
    "supported-continuation-20261020/metal-optimization-080/"
    "visual-verification-20261022/wrist-connected-repair-001/"
    "arm-actuation-preparation-001"
)
OWNER_HELPER = REPO / (
    "matter/tools/evidence/human-resting-20261005/"
    "supported-continuation-20261020/metal-optimization-080/"
    "visual-verification-20261022/wrist-connected-repair-001/"
    "hand-actuation-qualification-001/qualify_hand_muscle_owner_law.py"
)
SOURCE_CHECKOUT = Path(
    "/Users/numi/numi-human-retained-delivery-20261009/"
    "skin-source-binding-1202/source-inputs/myosim/checkout"
)
EXPECTED = {
    230: "BIClong", 231: "BICshort", 232: "BRA", 233: "BRD",
    293: "BIClong_l", 294: "BICshort_l", 295: "BRA_l", 296: "BRD_l",
}
ELBOWS = {"right": (39, 38), "left": (77, 76)}
TENSION_ULP_LIMIT = 4
FORCE_ULP_LIMIT = 1
RESIDUAL_LIMIT = 1.0e-4


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(), parse_constant=lambda _: float("nan"))


def stdout_json(path: Path, prefix: str):
    found = [
        line[len(prefix) + 1:]
        for line in path.read_text().splitlines()
        if line.startswith(prefix + "=")
    ]
    if len(found) != 1:
        raise ValueError(f"{path}: expected one {prefix}= JSON line, got {len(found)}")
    return json.loads(found[0], parse_constant=lambda _: float("nan"))


def finite(x):
    try:
        y = float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    return y if math.isfinite(y) else None


def load_owner_helper():
    spec = importlib.util.spec_from_file_location("hand_owner_law", OWNER_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import owner helper: {OWNER_HELPER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_source_model():
    sys.path.insert(0, str(SOURCE_CHECKOUT))
    import mujoco
    from myo_sim.build.compose import build_model
    model = build_model("myofullbody")
    if (model.nq, model.nv, model.nu) != (129, 128, 416):
        raise ValueError(f"unexpected source model shape {(model.nq, model.nv, model.nu)}")
    return mujoco, model


def sparse_actuator_moment(data, actuator: int, dof: int) -> float:
    start = int(data.moment_rowadr[actuator])
    count = int(data.moment_rownnz[actuator])
    for k in range(start, start + count):
        if int(data.moment_colind[k]) == dof:
            return float(data.actuator_moment[k])
    return 0.0


def make_source_state(mujoco, model, q, v):
    data = mujoco.MjData(model)
    data.qpos[:] = q
    data.qvel[:] = v
    mujoco.mj_forward(model, data)
    return data


def fail_if(condition: bool, errors: list[str], code: str):
    if not condition:
        errors.append(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    evidence = args.evidence.resolve()
    output = (args.output or (evidence / "arm-flexor-assessment-006.json")).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {output}")

    errors: list[str] = []
    old_inv = read_json(evidence / "results/bilateral-biceps-activation/invocation.json")
    old_state = stdout_json(
        evidence / "results/bilateral-biceps-activation/stdout.log",
        "stand_terminal_state",
    )
    old_control = stdout_json(
        evidence / "results/bilateral-biceps-activation/stdout.log",
        "selected_hand_wrist_actuation",
    )
    parity_inv = read_json(evidence / "results/bilateral-biceps-parity006/invocation.json")
    parity_state = stdout_json(
        evidence / "results/bilateral-biceps-parity006/stdout.log",
        "stand_terminal_state",
    )
    parity_control = stdout_json(
        evidence / "results/bilateral-biceps-parity006/stdout.log",
        "selected_hand_wrist_actuation",
    )
    group_inv = read_json(evidence / "results/bilateral-elbow-flexors006/invocation.json")
    group_state = stdout_json(
        evidence / "results/bilateral-elbow-flexors006/stdout.log",
        "stand_terminal_state",
    )
    group_control = stdout_json(
        evidence / "results/bilateral-elbow-flexors006/stdout.log",
        "selected_hand_wrist_actuation",
    )
    group_result = read_json(evidence / "results/bilateral-elbow-flexors006/result.json")
    parity_result = read_json(evidence / "results/bilateral-biceps-parity006/result.json")

    # All exact input hashes in the new native invocation must still resolve.
    input_hash_results = {}
    for raw_path, expected_hash in group_inv["input_hashes"].items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO / path
        actual = sha(path) if path.is_file() else None
        input_hash_results[str(path)] = {
            "expected_sha256": expected_hash,
            "actual_sha256": actual,
            "matches": actual == expected_hash,
        }
        fail_if(actual == expected_hash, errors, "pinned_input_hash_mismatch:" + str(path))
    fail_if(sha(Path(group_inv["binary_path"])) == group_inv["binary_sha256"],
            errors, "binary_hash_mismatch")
    build_manifest = Path(group_inv["build_source_manifest_path"])
    fail_if(sha(build_manifest) == group_inv["build_source_manifest_sha256"],
            errors, "build_source_manifest_hash_mismatch")

    # 004-vs-006 BIC activation comparison isolates telemetry parity.
    fail_if(old_state["q"] == parity_state["q"], errors, "004_006_biceps_q_not_bitwise_equal")
    fail_if(old_state["v"] == parity_state["v"], errors, "004_006_biceps_v_not_bitwise_equal")
    fail_if(old_control["q_rows"] == parity_control["q_rows"],
            errors, "004_006_biceps_hand_q_rows_not_bitwise_equal")
    old_muscles = {int(r["source_actuator_index"]): r for r in old_control["selected_muscles"]}
    parity_muscles = {int(r["source_actuator_index"]): r for r in parity_control["selected_muscles"]}
    common_ids = sorted(EXPECTED.keys() & old_muscles.keys() & parity_muscles.keys())
    parity_muscle_checks = {
        str(i): old_muscles[i] == parity_muscles[i] for i in common_ids
    }
    fail_if(common_ids == [230, 231, 293, 294] and all(parity_muscle_checks.values()),
            errors, "004_006_biceps_muscle_rows_not_bitwise_equal")

    # Pair baselines for BIC-only and complete flexor cohort must match on all
    # emitted hand/wrist and shoulder/elbow/forearm coordinates.
    baseline_rows = {}
    for key in ("q_rows", "arm_q_rows"):
        a = {int(r["q_index"]): r for r in parity_control[key]}
        b = {int(r["q_index"]): r for r in group_control[key]}
        shared = sorted(a.keys() & b.keys())
        baseline_rows[key] = {
            "shared_count": len(shared),
            "exact_q_count": sum(a[i]["baseline_q_rad"] == b[i]["baseline_q_rad"] for i in shared),
            "exact_v_count": sum(a[i]["baseline_v_rad_s"] == b[i]["baseline_v_rad_s"] for i in shared),
            "exact": all(
                a[i]["baseline_q_rad"] == b[i]["baseline_q_rad"]
                and a[i]["baseline_v_rad_s"] == b[i]["baseline_v_rad_s"]
                for i in shared
            ),
        }
        fail_if(
            len(shared) == len(a) == len(b)
            and baseline_rows[key]["exact"],
            errors,
            "006_paired_baseline_mismatch:" + key,
        )

    dt = float(group_inv["timestep_seconds"])
    fail_if(dt == 0.000125, errors, "unexpected_timestep")
    fail_if(group_inv["step_count"] == 80, errors, "unexpected_step_count")
    fail_if(group_result["native_exit_code"] == 0 and parity_result["native_exit_code"] == 0,
            errors, "native_exit_nonzero")
    fail_if(group_control["accepted_steps"] == 80, errors, "accepted_step_count_mismatch")
    fail_if(group_control["replay"] == "bitwise", errors, "native_replay_not_bitwise")
    fail_if(group_control["matched_baseline"] == "same_input_zero_activation_increment",
            errors, "matched_zero_activation_baseline_missing")
    fail_if(group_state["root_assistance"] is False, errors, "root_assistance_present")
    fail_if(len(group_control["q_rows"]) == 44 and len(group_control["arm_q_rows"]) == 10,
            errors, "hand_or_arm_endpoint_telemetry_missing")

    nhmyo = next(Path(p) for p in group_inv["input_hashes"] if p.endswith(".nhmyo"))
    source_manifest_path = Path(
        "/Users/n/numi-human-separation-20261018/C/publication-source-001/"
        "docs/media/numi-human-passive-fit-v4/source-manifest.json"
    )
    source_manifest = read_json(source_manifest_path)
    source_manifest_sha = sha(source_manifest_path)
    owner = load_owner_helper()
    raw, muscle_count, muscle_offset, arch_offset, archive_sha = owner.parse_model(nhmyo)
    manifest_rows = {
        int(row["source_actuator_index"]): row
        for row in source_manifest["muscles"]
    }
    fail_if(muscle_count == 416, errors, "unexpected_nhmyo_actuator_count")
    fail_if(archive_sha == "280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975",
            errors, "embedded_source_archive_hash_mismatch")
    fail_if(all(manifest_rows.get(i, {}).get("name") == n for i, n in EXPECTED.items()),
            errors, "source_manifest_flexor_identity_mismatch")

    muscles = []
    rows_by_id = {
        int(row["source_actuator_index"]): row
        for row in group_control["selected_muscles"]
    }
    observed_ids = sorted(rows_by_id)
    fail_if(observed_ids == sorted(EXPECTED), errors, "selected_actuator_ids_not_full_bilateral_flexor_group")
    for idx in sorted(EXPECTED):
        row = rows_by_id.get(idx)
        if row is None:
            continue
        rec = struct.unpack_from("<4I37f", raw, muscle_offset + idx * 164)
        arch = struct.unpack_from("<8f", raw, arch_offset + idx * 32)
        max_f = owner.max_force(rec[4:])
        predicted, candidates = owner.reconstruct(row, "accepted", dt, arch)
        accepted_tensions = [t for t, _ in candidates]
        observed_tension = finite(row.get("tendon_tension_normalized"))
        source_force = finite(row.get("source_force_n"))
        tendon_force = finite(row.get("tendon_tension_n"))
        residual = finite(row.get("normalized_equilibrium_residual"))
        baseline_residual = finite(row.get("baseline_normalized_equilibrium_residual"))
        accepted_tension_ulps = min(
            owner.ulps(observed_tension, t) for t in accepted_tensions
        ) if observed_tension is not None and accepted_tensions else None
        expected_source_force = owner.f32(-owner.f32(max_f * owner.f32(observed_tension)))
        expected_tendon_force = owner.f32(max_f * owner.f32(observed_tension))
        source_force_ulps = owner.ulps(source_force, expected_source_force)
        tendon_force_ulps = owner.ulps(tendon_force, expected_tendon_force)
        _, base_candidates = owner.reconstruct(row, "baseline", dt, arch)
        base_tensions = [t for t, _ in base_candidates]
        # Baseline force itself is not emitted as a standalone field by 006.
        # This derived value is diagnostic only; the native baseline residual
        # and owner-state reconstruction are checked directly.
        inferred_base_force = owner.f32(
            owner.f32(source_force) - owner.f32(row["source_force_increment_n"])
        )
        baseline_force_candidates = [
            owner.f32(-owner.f32(max_f * owner.f32(t))) for t in base_tensions
        ]
        closest_baseline_force = min(
            baseline_force_candidates,
            key=lambda f: abs(float(f) - float(inferred_base_force)),
        ) if baseline_force_candidates else None
        baseline_force_ulps = owner.ulps(inferred_base_force, closest_baseline_force) if closest_baseline_force is not None else None
        baseline_force_abs_error = abs(float(inferred_base_force) - float(closest_baseline_force)) if closest_baseline_force is not None else None
        accepted_activation = finite(row.get("accepted_activation"))
        baseline_activation = finite(row.get("baseline_accepted_activation"))
        commanded_activation = finite(row.get("commanded_activation"))
        commanded_increment = finite(row.get("commanded_activation_increment"))

        fail_if(
            accepted_tension_ulps is not None and accepted_tension_ulps <= TENSION_ULP_LIMIT,
            errors, f"accepted_tension_owner_law_mismatch:{idx}",
        )
        fail_if(source_force_ulps is not None and source_force_ulps <= FORCE_ULP_LIMIT,
                errors, f"accepted_source_force_owner_product_mismatch:{idx}")
        fail_if(tendon_force_ulps is not None and tendon_force_ulps <= FORCE_ULP_LIMIT,
                errors, f"accepted_tendon_force_owner_product_mismatch:{idx}")
        fail_if(residual is not None and abs(residual) <= RESIDUAL_LIMIT,
                errors, f"accepted_equilibrium_residual_exceeds_source_limit:{idx}")
        fail_if(baseline_residual is not None and abs(baseline_residual) <= RESIDUAL_LIMIT,
                errors, f"baseline_equilibrium_residual_exceeds_source_limit:{idx}")
        fail_if(
            bool(base_tensions) and all(math.isfinite(t) for t in base_tensions),
            errors, f"baseline_owner_state_reconstruction_failed:{idx}",
        )
        fail_if(
            accepted_activation == commanded_activation == 0.5
            and baseline_activation == 0.0
            and commanded_increment == 0.5,
            errors, f"activation_not_realized_as_requested:{idx}",
        )
        muscles.append({
            "source_actuator_index": idx,
            "source_name": EXPECTED[idx],
            "source_tendon_index": manifest_rows[idx].get("source_tendon_index"),
            "max_force_n": max_f,
            "activation": accepted_activation,
            "baseline_activation": baseline_activation,
            "accepted_path_length_m": row["accepted_path_length_m"],
            "accepted_path_velocity_m_s": row["accepted_path_velocity_m_s"],
            "accepted_fiber_length_m": row["accepted_fiber_length_m"],
            "accepted_fiber_velocity_m_s": row["accepted_fiber_velocity_m_s"],
            "accepted_tendon_tension_normalized": observed_tension,
            "accepted_owner_tension_candidates": accepted_tensions,
            "accepted_tension_error_float32_ulps": accepted_tension_ulps,
            "accepted_source_force_n": source_force,
            "accepted_source_force_error_float32_ulps": source_force_ulps,
            "accepted_tendon_force_n": tendon_force,
            "accepted_tendon_force_error_float32_ulps": tendon_force_ulps,
            "baseline_owner_tension_candidates": base_tensions,
            "baseline_force_inferred_from_increment_n": inferred_base_force,
            "baseline_force_owner_candidate_n": closest_baseline_force,
            "baseline_force_inference_abs_error_n_diagnostic_only": baseline_force_abs_error,
            "baseline_force_inference_error_float32_ulps_diagnostic_only": baseline_force_ulps,
            "accepted_normalized_equilibrium_residual": residual,
            "baseline_normalized_equilibrium_residual": baseline_residual,
            "source_force_increment_n": row["source_force_increment_n"],
            "baseline_source_force_directly_emitted": False,
        })

    # Q/v response and native source torque are read from source-indexed arm rows.
    elbow_response = {}
    for side, (qidx, vidx) in ELBOWS.items():
        row = next((r for r in group_control["arm_q_rows"] if int(r["q_index"]) == qidx), None)
        if row is None:
            errors.append("elbow_arm_q_row_missing:" + side)
            continue
        accepted_q = finite(row["accepted_q_rad"])
        baseline_q = finite(row["baseline_q_rad"])
        accepted_v = finite(row["accepted_v_rad_s"])
        baseline_v = finite(row["baseline_v_rad_s"])
        lower = finite(row["source_lower_limit_rad"])
        upper = finite(row["source_upper_limit_rad"])
        tau = finite(row["selected_muscle_generalized_torque_increment_nm"])
        tau_l1 = finite(row["selected_muscle_generalized_torque_increment_l1_nm"])
        fail_if(lower <= accepted_q <= upper, errors, "elbow_joint_limit_violation:" + side)
        fail_if(accepted_q > baseline_q and accepted_v > baseline_v and tau > 0 and tau_l1 > 0,
                errors, "elbow_flexor_directional_response_missing:" + side)
        fail_if(math.isclose(accepted_v, baseline_v, rel_tol=0.0, abs_tol=1e-12) is False,
                errors, "elbow_velocity_did_not_change:" + side)
        elbow_response[side] = {
            "source_q_index": qidx,
            "source_v_index": vidx,
            "lower_limit_rad": lower,
            "upper_limit_rad": upper,
            "accepted_q_rad": accepted_q,
            "baseline_q_rad": baseline_q,
            "q_increment_rad": finite(row["q_increment_rad"]),
            "accepted_v_rad_s": accepted_v,
            "baseline_v_rad_s": baseline_v,
            "v_increment_rad_s": finite(row["v_increment_rad_s"]),
            "native_selected_muscle_generalized_torque_increment_nm": tau,
            "native_selected_muscle_generalized_torque_increment_l1_nm": tau_l1,
            "within_source_joint_limits": True,
            "positive_free_flexor_response_vs_matched_zero_activation": True,
        }

    # Evaluate source path lengths, velocities, moment arms and endpoint torque
    # at both accepted and matched baseline joint states.
    mujoco, source_model = load_source_model()
    baseline_state = read_json(evidence / "results/bilateral-biceps-baseline/result.json")
    baseline_run_state = stdout_json(
        evidence / "results/bilateral-biceps-baseline/stdout.log",
        "stand_terminal_state",
    )
    q_base = list(baseline_run_state["q"])
    v_base = list(baseline_run_state["v"])
    # Replace all relevant baseline coordinates with the exact paired 006
    # baseline readback; the rest of the body remains at its recorded 004
    # zero-activation endpoint.
    for row in group_control["q_rows"] + group_control["arm_q_rows"]:
        q_base[int(row["q_index"])] = row["baseline_q_rad"]
        v_base[int(row["dof_index"])] = row["baseline_v_rad_s"]
    source_base = make_source_state(mujoco, source_model, q_base, v_base)
    source_active = make_source_state(mujoco, source_model, group_state["q"], group_state["v"])

    path_checks = []
    endpoint_torque = {}
    for side, (qidx, dof) in ELBOWS.items():
        group = [230, 231, 232, 233] if side == "right" else [293, 294, 295, 296]
        source_delta = 0.0
        base_total = 0.0
        active_total = 0.0
        native_row = next(r for r in group_control["arm_q_rows"] if int(r["q_index"]) == qidx)
        for idx in group:
            row = rows_by_id[idx]
            rec = struct.unpack_from("<4I37f", raw, muscle_offset + idx * 164)
            arch = struct.unpack_from("<8f", raw, arch_offset + idx * 32)
            max_f = owner.max_force(rec[4:])
            _, base_candidates = owner.reconstruct(row, "baseline", dt, arch)
            inferred_base = owner.f32(
                owner.f32(row["source_force_n"])
                - owner.f32(row["source_force_increment_n"])
            )
            base_force_candidates = [
                owner.f32(-owner.f32(max_f * owner.f32(t)))
                for t, _ in base_candidates
            ]
            base_force = min(
                base_force_candidates,
                key=lambda f: abs(float(f) - float(inferred_base)),
            )
            active_force = float(row["source_force_n"])
            tid = int(source_model.actuator_trnid[idx, 0])
            base_moment = sparse_actuator_moment(source_base, idx, dof)
            active_moment = sparse_actuator_moment(source_active, idx, dof)
            base_tau = base_moment * float(base_force)
            active_tau = active_moment * active_force
            base_total += base_tau
            active_total += active_tau
            source_delta += active_tau - base_tau
            path_checks.append({
                "source_actuator_index": idx,
                "source_name": EXPECTED[idx],
                "baseline": {
                    "native_path_length_m": row["baseline_path_length_m"],
                    "source_path_length_m": float(source_base.ten_length[tid]),
                    "path_length_abs_error_m": abs(float(row["baseline_path_length_m"]) - float(source_base.ten_length[tid])),
                    "native_path_velocity_m_s": row["baseline_path_velocity_m_s"],
                    "source_path_velocity_m_s": float(source_base.ten_velocity[tid]),
                    "path_velocity_abs_error_m_s": abs(float(row["baseline_path_velocity_m_s"]) - float(source_base.ten_velocity[tid])),
                    "source_actuator_moment_dL_dq": base_moment,
                    "owner_force_n": float(base_force),
                    "source_tau_nm": base_tau,
                },
                "accepted": {
                    "native_path_length_m": row["accepted_path_length_m"],
                    "source_path_length_m": float(source_active.ten_length[tid]),
                    "path_length_abs_error_m": abs(float(row["accepted_path_length_m"]) - float(source_active.ten_length[tid])),
                    "native_path_velocity_m_s": row["accepted_path_velocity_m_s"],
                    "source_path_velocity_m_s": float(source_active.ten_velocity[tid]),
                    "path_velocity_abs_error_m_s": abs(float(row["accepted_path_velocity_m_s"]) - float(source_active.ten_velocity[tid])),
                    "source_actuator_moment_dL_dq": active_moment,
                    "owner_force_n": active_force,
                    "source_tau_nm": active_tau,
                },
            })
        native_tau = float(native_row["selected_muscle_generalized_torque_increment_nm"])
        endpoint_torque[side] = {
            "source_endpoint_baseline_tau_nm": base_total,
            "source_endpoint_accepted_tau_nm": active_total,
            "source_endpoint_tau_increment_nm": source_delta,
            "native_endpoint_tau_increment_nm": native_tau,
            "absolute_difference_nm": abs(source_delta - native_tau),
            "difference_over_native_torque_l1": (
                abs(source_delta - native_tau)
                / float(native_row["selected_muscle_generalized_torque_increment_l1_nm"])
            ),
            "comparison_status": "reported_only_no_new_threshold",
        }

    source_comparison = {
        "source_builder": "MyoSim archived source build_model(myofullbody)",
        "source_checkout": str(SOURCE_CHECKOUT),
        "source_archive_sha256_embedded_in_nhmyo": archive_sha,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": source_manifest_sha,
        "model_shape": {"nq": source_model.nq, "nv": source_model.nv, "nu": source_model.nu},
        "baseline_full_body_state": "separate 004 zero-activation endpoint with all 54 emitted arm/hand coordinates overwritten by exact paired 006 baseline q/v",
        "path_checks": path_checks,
        "endpoint_torque_by_elbow": endpoint_torque,
        "path_error_maxima": {
            "baseline_path_length_abs_error_m": max(x["baseline"]["path_length_abs_error_m"] for x in path_checks),
            "accepted_path_length_abs_error_m": max(x["accepted"]["path_length_abs_error_m"] for x in path_checks),
            "baseline_path_velocity_abs_error_m_s": max(x["baseline"]["path_velocity_abs_error_m_s"] for x in path_checks),
            "accepted_path_velocity_abs_error_m_s": max(x["accepted"]["path_velocity_abs_error_m_s"] for x in path_checks),
        },
        "limits": "MyoSim source comparison is diagnostic; endpoint torque difference is reported without introducing a new acceptance threshold.",
    }

    # Verify standalone 004 zero-activation terminal state is near, not bitwise
    # equal to the paired baseline readback.
    baseline_qv_comparison = []
    for row in group_control["q_rows"] + group_control["arm_q_rows"]:
        qi = int(row["q_index"])
        vi = int(row["dof_index"])
        baseline_qv_comparison.append({
            "q_index": qi,
            "abs_q_difference_rad": abs(float(baseline_run_state["q"][qi]) - float(row["baseline_q_rad"])),
            "dof_index": vi,
            "abs_v_difference_rad_s": abs(float(baseline_run_state["v"][vi]) - float(row["baseline_v_rad_s"])),
        })
    baseline_qv_summary = {
        "compared_coordinate_count": len(baseline_qv_comparison),
        "bitwise_exact_q_count": sum(x["abs_q_difference_rad"] == 0 for x in baseline_qv_comparison),
        "bitwise_exact_v_count": sum(x["abs_v_difference_rad_s"] == 0 for x in baseline_qv_comparison),
        "max_abs_q_difference_rad": max(x["abs_q_difference_rad"] for x in baseline_qv_comparison),
        "max_abs_v_difference_rad_s": max(x["abs_v_difference_rad_s"] for x in baseline_qv_comparison),
        "comparison": "standalone old-binary run vs internal paired baseline; small floating-point differences preserved, not treated as bitwise identity",
    }

    status = "PASS_CORE_DIRECTIONAL_10MS_OWNER_LAW" if not errors else "FAIL"
    receipt = {
        "schema": "numi.human.elbow-flexor-assessment.v1",
        "derived_status": status,
        "scope": "core selected-tendon actuation and source owner-law/directional response only; not whole-body FEM/contact, grasp, sustained range, Brain control, biological, clinical, or production qualification",
        "errors": errors,
        "native": {
            "binary_path": group_inv["binary_path"],
            "binary_sha256": group_inv["binary_sha256"],
            "build_source_manifest_path": group_inv["build_source_manifest_path"],
            "build_source_manifest_sha256": group_inv["build_source_manifest_sha256"],
            "step_count": group_inv["step_count"],
            "timestep_seconds": dt,
            "simulated_seconds": group_inv["step_count"] * dt,
            "full_flexor_pair_wall_seconds": group_result["wall_elapsed_seconds"],
            "full_flexor_native_exit_code": group_result["native_exit_code"],
            "biceps_parity_wall_seconds": parity_result["wall_elapsed_seconds"],
            "biceps_parity_native_exit_code": parity_result["native_exit_code"],
            "accepted_steps": group_control["accepted_steps"],
            "replay": group_control["replay"],
            "matched_baseline": group_control["matched_baseline"],
            "root_assistance": group_state["root_assistance"],
            "muscle_ids": sorted(EXPECTED),
            "muscle_names": [EXPECTED[i] for i in sorted(EXPECTED)],
            "invocation_path": str(evidence / "results/bilateral-elbow-flexors006/invocation.json"),
            "invocation_sha256": sha(evidence / "results/bilateral-elbow-flexors006/invocation.json"),
            "stdout_path": str(evidence / "results/bilateral-elbow-flexors006/stdout.log"),
            "stdout_sha256": sha(evidence / "results/bilateral-elbow-flexors006/stdout.log"),
            "input_hashes": input_hash_results,
        },
        "telemetry_006_biceps_parity_vs_004": {
            "accepted_q_129_bitwise_equal": old_state["q"] == parity_state["q"],
            "accepted_v_128_bitwise_equal": old_state["v"] == parity_state["v"],
            "all_44_hand_wrist_rows_bitwise_equal": old_control["q_rows"] == parity_control["q_rows"],
            "common_biceps_muscle_rows_bitwise_equal": parity_muscle_checks,
            "passed": all([
                old_state["q"] == parity_state["q"],
                old_state["v"] == parity_state["v"],
                old_control["q_rows"] == parity_control["q_rows"],
                all(parity_muscle_checks.values()),
            ]),
        },
        "paired_baseline_between_006_cohorts": baseline_rows,
        "elbow_response": elbow_response,
        "muscle_owner_law": {
            "nhmyo2_path": str(nhmyo),
            "nhmyo2_sha256": sha(nhmyo),
            "embedded_source_archive_sha256": archive_sha,
            "accepted_owner_tension_ulps_max": max(m["accepted_tension_error_float32_ulps"] for m in muscles),
            "accepted_source_force_product_ulps_max": max(m["accepted_source_force_error_float32_ulps"] for m in muscles),
            "accepted_tendon_force_product_ulps_max": max(m["accepted_tendon_force_error_float32_ulps"] for m in muscles),
            "accepted_normalized_equilibrium_residual_abs_max": max(abs(m["accepted_normalized_equilibrium_residual"]) for m in muscles),
            "baseline_normalized_equilibrium_residual_abs_max": max(abs(m["baseline_normalized_equilibrium_residual"]) for m in muscles),
            "baseline_force_directly_emitted": False,
            "baseline_force_inference_note": "Baseline owner tension is reconstructed from path/fiber state and residual. The ULP comparison inferred by subtracting rounded accepted force and force increment is diagnostic only; its ULP count is not meaningful near zero/slack and is not a gate.",
            "muscles": muscles,
        },
        "source_model_endpoint_comparison": source_comparison,
        "standalone_vs_internal_baseline_qv": baseline_qv_summary,
        "assessment_script_path": str(Path(__file__).resolve()),
        "assessment_script_sha256": sha(Path(__file__).resolve()),
        "command_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "working_directory": str(Path.cwd()),
        "qualification_statement": (
            "The full bilateral source elbow-flexor group produces a positive native elbow torque increment "
            "and positive free elbow q/v response from the authored lower limit over this 10 ms trial. "
            "This is a short core directional actuation result, not a full-range or whole-body-contact qualification."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "derived_status": status,
        "errors": errors,
        "output": str(output),
        "output_sha256": sha(output),
        "elbow_response": elbow_response,
        "owner_law_maxima": receipt["muscle_owner_law"],
        "source_endpoint_torque": endpoint_torque,
        "path_error_maxima": source_comparison["path_error_maxima"],
        "parity": receipt["telemetry_006_biceps_parity_vs_004"],
        "standalone_vs_internal_baseline": baseline_qv_summary,
    }, indent=2, allow_nan=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
