#!/usr/bin/env python3
"""CPU replay of a sealed FEM-versus-prescribed-capsule contact sample.

This reads the bounded `fem_contact_iterate` JSONL arena capture and frozen
source. It does not invoke Matter, Metal, or advance a simulation. The replay
compares the captured PSD radial operator with the full smooth side-contact
derivative for the capsule fixture, then emulates binary32 trial updates for
the recorded Armijo alpha sequence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path


DEFAULT_CAPTURE = Path(
    "/Users/home/cardboard-implementation-20261014/native/"
    "full-contact-resolved-descent-037/stderr.log"
)
DEFAULT_PREDICTION = DEFAULT_CAPTURE.with_name("prediction.json")
DEFAULT_SNAPSHOT = Path(
    "/Users/home/cardboard-implementation-20261014/snapshots/strict-contact-024"
)
DEFAULT_BUILD = Path("/Users/home/numi-cardboard-20261006/build-cardboard-provenance-024")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def scale(s, a):
    return [s * x for x in a]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def rel_norm_error(actual, expected):
    den = norm(expected)
    return norm(sub(actual, expected)) / den if den else None


def strict_inertial_stiffness_branch(separation, thickness, inverse_response,
                                    timestep, closing_speed):
    """Match the capsule correction's fail-closed inertial-branch guard."""
    if not (thickness > 0.0 and inverse_response > 0.0 and timestep > 0.0):
        return False
    if not (math.isfinite(separation) and math.isfinite(closing_speed)):
        return False
    distance = max(separation, 1.0e-4 * thickness)
    remaining = max(thickness - distance, 1.0e-3 * thickness)
    h2 = thickness * thickness
    k_inertial = 10.0 / max(
        inverse_response * timestep * timestep * h2, 1.0e-20
    )
    k_impact = max(closing_speed, 0.0) / max(
        inverse_response * timestep * remaining * h2, 1.0e-20
    )
    # Keep the same 32-epsilon margin from the float32 max() branch kink.
    float32_epsilon = 2.0 ** -23
    return (math.isfinite(k_inertial) and math.isfinite(k_impact) and
            k_inertial > 0.0 and
            k_impact < k_inertial * (1.0 - 32.0 * float32_epsilon))


def capsule_side_feature_admissible(parameter, normal, capsule_axis,
                                    margin=1.0e-3, alignment_tolerance=64.0 * 2.0 ** -23):
    """Check the strict interior-side feature guard used by the native patch."""
    if not (math.isfinite(parameter) and
            all(math.isfinite(value) for value in (*normal, *capsule_axis))):
        return False
    if not (parameter > margin and parameter < 1.0 - margin):
        return False
    return abs(dot(normal, capsule_axis)) <= alignment_tolerance


def capsule_side_geometric_action(normal, capsule_axis, normal_impulse,
                                  radial_distance, timestep, direction):
    """Apply the signed capsule-distance geometric Hessian to a direction."""
    if radial_distance <= 0.0:
        raise ValueError("capsule radial distance must be positive")
    transverse = sub(direction, scale(dot(capsule_axis, direction), capsule_axis))
    tangent_to_normal = sub(transverse, scale(dot(normal, direction), normal))
    return scale(-timestep * normal_impulse / radial_distance, tangent_to_normal)


def capsule_side_operator_action(normal, capsule_axis, radial_curvature,
                                 normal_impulse, radial_distance, timestep,
                                 direction):
    """Full smooth-side action: radial barrier plus signed geometry term."""
    radial = scale(radial_curvature * dot(normal, direction), normal)
    geometric = capsule_side_geometric_action(
        normal, capsule_axis, normal_impulse, radial_distance, timestep,
        direction)
    return add(radial, geometric)


def parse_constant(source: str, name: str) -> float:
    match = re.search(
        rf"constexpr\s+double\s+{re.escape(name)}\s*=\s*([^;]+);", source
    )
    if not match:
        raise ValueError(f"fixture constant {name} was not found")
    expr = match.group(1).strip()
    # This intentionally supports only the arithmetic used by the fixture.
    if "kMm" in expr:
        mm = parse_constant(source, "kMm")
        expr = expr.replace("kMm", repr(mm))
    if not re.fullmatch(r"[0-9eE.+* /()-]+", expr):
        raise ValueError(f"unsupported expression for {name}: {expr}")
    return float(eval(expr, {"__builtins__": {}}, {}))


def parse_arena_hex(value: str, expected_bytes: int, name: str) -> bytes:
    raw = bytes.fromhex(value)
    if len(raw) != expected_bytes:
        raise ValueError(f"{name} has {len(raw)} bytes; expected {expected_bytes}")
    return raw


def unpack_node(nodes: bytes, node_id: int):
    # NMFEMNodeStateGPU: five float4 rows, sizeof=80 bytes in frozen ABI45.
    return struct.unpack_from("<20f", nodes, node_id * 80)


def unpack_sample(samples: bytes, pair_id: int):
    # NMContactSampleGPU: uint4 identity followed by nine float4 rows.
    offset = pair_id * 160
    identity = struct.unpack_from("<4I", samples, offset)
    floats = struct.unpack_from("<36f", samples, offset + 16)
    return identity, floats


def barrier(gap, thickness, inverse_response, dt, closing_speed, fixed_k=None):
    """Double-precision transcription of ipcBarrierResponse's active branch."""
    if not (thickness > 0.0 and inverse_response > 0.0 and dt > 0.0):
        return {"impulse": 0.0, "radial_velocity_hessian": 0.0,
                "k_inertial": 0.0, "k_impact": 0.0, "stiffness": 0.0,
                "force": 0.0}
    if gap >= thickness:
        return {"impulse": 0.0, "radial_velocity_hessian": 0.0,
                "k_inertial": 0.0, "k_impact": 0.0, "stiffness": 0.0,
                "force": 0.0}
    d = max(gap, 1.0e-4 * thickness)
    d2 = d * d
    h2 = thickness * thickness
    off = d2 - h2
    log_term = math.log(d2 / h2)
    k_inertial = 10.0 / max(inverse_response * dt * dt * h2, 1.0e-20)
    remaining = max(thickness - d, 1.0e-3 * thickness)
    k_impact = max(closing_speed, 0.0) / max(
        inverse_response * dt * remaining * h2, 1.0e-20
    )
    stiffness = max(k_inertial, k_impact) if fixed_k is None else fixed_k
    first = -stiffness * (2.0 * off * log_term + off * off / d2)
    second = -stiffness * (
        2.0 * log_term + 4.0 * off / d2 - off * off / (d2 * d2)
    )
    force = -2.0 * d * first
    hessian = (max(4.0 * d2 * second + 2.0 * first, 0.0)
               if gap >= 1.0e-4 * thickness else 0.0)
    return {
        "impulse": max(force, 0.0) * dt,
        "radial_velocity_hessian": hessian * dt * dt,
        "k_inertial": k_inertial,
        "k_impact": k_impact,
        "stiffness": stiffness,
        "force": force,
    }


def contact_residual(direction, epsilon, current_velocity, relative_tool_velocity,
                     gap0, normal0, radius, dt, thickness, inverse_response,
                     capsule_axis):
    """Node impulse residual for an infinitesimal side-contact perturbation."""
    # Capsule side distance is invariant along the axis. Its radial vector is
    # anchored to the captured gap and normal, avoiding re-solving the captured
    # float32 closest point from rounded witness coordinates.
    radial_distance = gap0 + radius
    displacement = scale(dt * epsilon, direction)
    projected = [
        displacement[i] - capsule_axis[i] * dot(capsule_axis, displacement)
        for i in range(3)
    ]
    radial = add(scale(radial_distance, normal0), projected)
    radial = [radial[i] - capsule_axis[i] * dot(radial, capsule_axis)
              for i in range(3)]
    rho = norm(radial)
    if rho <= 0.0:
        raise ValueError("invalid capsule radial distance")
    normal = scale(1.0 / rho, radial)
    gap = rho - radius
    velocity = add(current_velocity, scale(epsilon, direction))
    relative_velocity = sub(velocity, relative_tool_velocity)
    normal_velocity = dot(relative_velocity, normal)
    response = barrier(gap, thickness, inverse_response, dt,
                       max(-normal_velocity, 0.0))
    return (scale(response["impulse"], normal), gap, normal,
            normal_velocity, response)


def analyze_contact(sample, node, correction, dt, radius, thickness,
                    relative_tool_velocity, capsule_axis, capsule_center,
                    epsilons, captured_includes_capsule_geometry):
    f = sample[1]
    # float4 rows after identity: point/separation, normal/normalVelocity,
    # admission relative motion, rigid impulse/normal impulse, angular/tangent,
    # barrier, and three Hessian rows.
    gap = f[3]
    normal_raw = f[4:7]
    n_projected = [
        normal_raw[i] - capsule_axis[i] * dot(normal_raw, capsule_axis)
        for i in range(3)
    ]
    normal = scale(1.0 / norm(n_projected), n_projected)
    normal_velocity_captured = f[7]
    impulse_captured = f[20]
    barrier_row = f[20:24]
    hessian_rows = [f[24:27], f[28:31], f[32:35]]
    captured_operator_action = [dot(row, correction) for row in hessian_rows]
    captured_radial_curvature = f[21]
    inverse_response = node[7]
    current_velocity = list(node[4:7])
    response = barrier(gap, thickness, inverse_response, dt,
                       max(-normal_velocity_captured, 0.0))
    stiffness_branch = "inertial" if response["k_inertial"] >= response["k_impact"] else "impact"

    # The rich capture stores the candidate's authoritative reference
    # displacement separately from its rounded world position. For this
    # identity-pose, +Y capsule fixture, compare the production world-position
    # route with a locally rebased (rest - capsule_center) + u route. This is
    # an arithmetic diagnostic only; it does not replace the native contact
    # calculation or assume the two float32 evaluations are identical.
    rest = list(node[8:11])
    displacement = list(node[16:19])
    mode = node[19]
    reconstructed_position = [f32(rest[i] + displacement[i]) for i in range(3)]
    center = [f32(value) for value in capsule_center]
    relative_world_position = [f32(node[i] - center[i]) for i in range(3)]
    relative_rebased = [
        f32(f32(rest[i] - center[i]) + displacement[i]) for i in range(3)
    ]

    def side_gap_from_relative(relative):
        return f32(math.sqrt(
            f32(relative[0] * relative[0]) +
            f32(relative[2] * relative[2])
        ) - f32(radius))

    candidate_state_arithmetic = {
        "reference_displacement_mode": mode,
        "reference_displacement_u_m": displacement,
        "rest_position_m": rest,
        "captured_world_position_m": list(node[0:3]),
        "rest_plus_u_world_position_m": reconstructed_position,
        "captured_minus_rest_plus_u_m": [
            node[i] - reconstructed_position[i] for i in range(3)
        ],
        "u_minus_dt_times_candidate_velocity_m": [
            displacement[i] - f32(dt * node[4 + i]) for i in range(3)
        ],
        "capsule_relative_world_position_f32_m": relative_world_position,
        "capsule_relative_rebased_rest_plus_u_f32_m": relative_rebased,
        "capsule_side_gap_from_world_position_f32_m":
            side_gap_from_relative(relative_world_position),
        "capsule_side_gap_from_rebased_rest_plus_u_f32_m":
            side_gap_from_relative(relative_rebased),
        "capsule_gap_difference_rebased_minus_world_m":
            side_gap_from_relative(relative_rebased) -
            side_gap_from_relative(relative_world_position),
        "scope": "identity-pose capsule side arithmetic diagnostic; not a native alternate contact evaluation",
    }

    # Complete smooth Hessian of a capsule side distance. Its curvature lives
    # in the plane normal to the capsule axis; the axis is a null direction.
    radial_distance = gap + radius
    normal_projection = scale(dot(normal, correction), normal)
    captured_radial_action = scale(captured_radial_curvature, normal_projection)
    transverse = [
        correction[i] - capsule_axis[i] * dot(capsule_axis, correction)
        for i in range(3)
    ]
    geometric_direction = sub(transverse, normal_projection)
    independent_radial_action = scale(
        response["radial_velocity_hessian"], normal_projection)
    geometric_action = capsule_side_geometric_action(
        normal, capsule_axis, response["impulse"], radial_distance, dt, correction)
    full_action = add(independent_radial_action, geometric_action)
    expected_captured_action = (
        full_action if captured_includes_capsule_geometry
        else independent_radial_action
    )
    captured_vs_full_error = rel_norm_error(
        captured_operator_action, full_action)

    fd = []
    for epsilon in epsilons:
        plus = contact_residual(
            correction, epsilon, current_velocity, relative_tool_velocity,
            gap, normal, radius, dt, thickness, inverse_response, capsule_axis)
        minus = contact_residual(
            correction, -epsilon, current_velocity, relative_tool_velocity,
            gap, normal, radius, dt, thickness, inverse_response, capsule_axis)
        action = scale(-0.5 / epsilon, sub(plus[0], minus[0]))
        fd.append({
            "epsilon_velocity_scale": epsilon,
            "plus_gap_m": plus[1],
            "minus_gap_m": minus[1],
            "plus_normal_velocity_m_per_s": plus[3],
            "minus_normal_velocity_m_per_s": minus[3],
            "plus_stiffness_branch": "inertial" if plus[4]["k_inertial"] >= plus[4]["k_impact"] else "impact",
            "minus_stiffness_branch": "inertial" if minus[4]["k_inertial"] >= minus[4]["k_impact"] else "impact",
            "negative_residual_directional_derivative_Ns_per_mps": action,
            "relative_error_vs_captured_operator_action": rel_norm_error(
                action, captured_operator_action),
            "relative_error_vs_expected_captured_action": rel_norm_error(
                action, expected_captured_action),
            "relative_error_vs_full_capsule_hessian": rel_norm_error(action, full_action),
        })

    # Derivatives w.r.t. line-search alpha: x' = dt*correction, v'=correction.
    dn_dalpha = scale(dt / radial_distance, geometric_direction)
    dc_dalpha = scale(dt * dot(capsule_axis, correction), capsule_axis)
    dwitness_dalpha = add(dc_dalpha, scale(radius, dn_dalpha))
    rel_velocity = sub(current_velocity, relative_tool_velocity)
    normal_velocity_derivative = dot(correction, normal) + dot(rel_velocity, dn_dalpha)

    return {
        "node_id": sample[0][0],
        "identity": list(sample[0]),
        "candidate_position_m": list(node[0:3]),
        "candidate_velocity_m_per_s": current_velocity,
        "candidate_inverse_mass_kg_inv": inverse_response,
        "candidate_state_arithmetic": candidate_state_arithmetic,
        "correction_velocity_m_per_s_per_alpha": list(correction),
        "gap_m": gap,
        "normal": normal,
        "captured_normal_velocity_m_per_s": normal_velocity_captured,
        "captured_normal_impulse_Ns": impulse_captured,
        "captured_sample_operator_rows_Ns_per_mps": hessian_rows,
        "captured_sample_operator_action_Ns_per_mps": captured_operator_action,
        "captured_radial_action_Ns_per_mps": captured_radial_action,
        "independent_fp64_radial_action_Ns_per_mps": independent_radial_action,
        "full_capsule_geometric_action_Ns_per_mps": geometric_action,
        "full_capsule_directional_action_Ns_per_mps": full_action,
        "captured_vs_full_capsule_action_relative_error": captured_vs_full_error,
        "captured_includes_capsule_geometry": captured_includes_capsule_geometry,
        "geometric_action_over_radial_action": norm(geometric_action) / norm(independent_radial_action),
        "geometric_action_dot_correction": dot(geometric_action, correction),
        "radial_action_dot_correction": dot(independent_radial_action, correction),
        "full_action_dot_correction": dot(full_action, correction),
        "radial_distance_m": radial_distance,
        "barrier_replay": response,
        "stiffness_branch": stiffness_branch,
        "impact_to_inertial_stiffness_ratio": response["k_impact"] / response["k_inertial"],
        "dnormal_dalpha_per_alpha": dn_dalpha,
        "dwitness_dalpha_m_per_alpha": dwitness_dalpha,
        "dnormal_velocity_dalpha_per_alpha_s_inv": normal_velocity_derivative,
        "central_difference": fd,
    }


def parse_trace(path: Path, root: int, iteration: int):
    stages = {}
    armijo = None
    for line in path.read_text().splitlines():
        if line.startswith("fem_contact_iterate="):
            row = json.loads(line.split("=", 1)[1])
            if row.get("root") == root and row.get("iteration") == iteration:
                stages[row["stage"]] = row
        elif line.startswith("fem_residual_armijo_iteration="):
            row = json.loads(line.split("=", 1)[1])
            if row.get("root") == root and row.get("newton_iteration") == iteration:
                armijo = row
    expected_stages = {
        "before_contact_limits", "after_deformable_limit",
        "after_rigid_limit", "final_line_search",
    }
    if set(stages) != expected_stages:
        raise ValueError(f"expected four selected contact stages; got {sorted(stages)}")
    baseline = stages["before_contact_limits"]["arenas"]
    for stage in expected_stages - {"before_contact_limits"}:
        arenas = stages[stage]["arenas"]
        for key in ("candidate", "solution", "samples"):
            if arenas[key] != baseline[key]:
                raise ValueError(f"captured {key} changed across contact stages")
    if armijo is None:
        raise ValueError("selected Armijo record is missing")
    return stages, armijo


def f32_capsule_separation(position, radius, center_z, half_span):
    """Binary32-like centerline-side distance for the fixture's y-axis capsule."""
    # This mirrors capsuleContact's float operations for the identity pose.
    first = [f32(0.0), f32(-half_span), f32(center_z)]
    second = [f32(0.0), f32(half_span), f32(center_z)]
    edge = [f32(second[i] - first[i]) for i in range(3)]
    edge2 = f32(f32(edge[0] * edge[0]) + f32(edge[1] * edge[1]) + f32(edge[2] * edge[2]))
    point_minus_first = [f32(position[i] - first[i]) for i in range(3)]
    projection = f32(
        f32(point_minus_first[0] * edge[0]) +
        f32(point_minus_first[1] * edge[1]) +
        f32(point_minus_first[2] * edge[2])
    )
    t = max(0.0, min(1.0, f32(projection / edge2)))
    center = [f32(first[i] + f32(t * edge[i])) for i in range(3)]
    delta = [f32(position[i] - center[i]) for i in range(3)]
    d2 = f32(f32(delta[0] * delta[0]) + f32(delta[1] * delta[1]) + f32(delta[2] * delta[2]))
    distance = f32(math.sqrt(d2))
    return f32(distance - f32(radius))


def representability(node, correction, alpha_values, dt, center_z, radius, half_span):
    # NMFEMNodeStateGPU rows are position/mass, velocity/inverse-mass,
    # rest/fixed, deltaVelocity, reference-displacement/mode. Reconstruct the
    # accepted reference displacement from the captured candidate using the
    # same update relation as the frozen trial-preparation kernel.
    v = list(node[4:7])
    rest = list(node[8:11])
    mode = node[19]
    current_position = list(node[0:3])
    if mode == 1.0:
        current_u = list(node[16:19])
        accepted_u = [f32(current_u[i] - f32(f32(dt) * f32(v[i])))
                      for i in range(3)]
        accepted_position = [f32(rest[i] + accepted_u[i]) for i in range(3)]
    elif mode == 0.0:
        accepted_u = None
        accepted_position = [
            f32(current_position[i] - f32(f32(dt) * f32(v[i])))
            for i in range(3)
        ]
    else:
        raise ValueError(f"unsupported reference-displacement mode {mode}")
    rows = []
    for alpha in alpha_values:
        trial_v = [f32(f32(v[i]) + f32(f32(alpha) * f32(correction[i])))
                   for i in range(3)]
        if mode == 1.0:
            trial_u = [f32(accepted_u[i] + f32(f32(dt) * trial_v[i]))
                       for i in range(3)]
            trial_x = [f32(rest[i] + trial_u[i]) for i in range(3)]
        else:
            trial_u = None
            trial_x = [f32(accepted_position[i] + f32(f32(dt) * trial_v[i]))
                       for i in range(3)]
        gap = f32_capsule_separation(trial_x, radius, center_z, half_span)
        current_gap = f32_capsule_separation(
            current_position, radius, center_z, half_span)
        rows.append({
            "alpha": alpha,
            "velocity_bits_unchanged": all(f32(trial_v[i]) == f32(v[i]) for i in range(3)),
            "position_bits_unchanged": all(trial_x[i] == current_position[i] for i in range(3)),
            "position_delta_m": [trial_x[i] - current_position[i] for i in range(3)],
            "trial_velocity_m_per_s": trial_v,
            "trial_reference_displacement_m": trial_u,
            "trial_position_m": trial_x,
            "binary32_gap_m": gap,
            "gap_bits_unchanged": gap == current_gap,
        })
    return {
        "reference_displacement_mode": mode,
        "candidate_position_m": current_position,
        "candidate_velocity_m_per_s": v,
        "accepted_position_reconstructed_m": accepted_position,
        "accepted_displacement_reconstructed_m": accepted_u,
        "trials": rows,
    }


def decode_armijo_result(environment):
    """Decode the overloaded line-search record without treating stale trial
    status as a rejection after an alpha was already selected.
    """
    raw = environment["line_search_raw"]
    if len(raw) != 4:
        raise ValueError("Armijo line-search record must contain four values")
    accepted = bool(environment.get("accepted", False))
    certified_noop = bool(environment.get("certified_noop", False))
    trial_alpha = float(environment["trial_alpha"])
    selected_alpha = float(environment["selected_alpha"])
    diagnostics = environment.get("solver_status", {}).get("diagnostics", [])

    if accepted and certified_noop:
        return {
            "outcome": "certified_noop",
            "accepted": True,
            "certified_noop": True,
            "initial_bound": 0.0,
            "trial_count": 1,
            "selected_alpha": 0.0,
            "last_tested_alpha": 0.0,
            "alpha_schedule": [0.0],
            "trial_status_code_is_authoritative": False,
        }
    if accepted:
        count_value = float(raw[2])
        count = int(round(count_value))
        if count < 1 or not math.isclose(count_value, count, abs_tol=1.0e-5):
            raise ValueError("accepted Armijo record has an invalid trial count")
        alpha = selected_alpha
        if alpha <= 0.0 or not math.isclose(alpha, trial_alpha, rel_tol=2.0e-5,
                                             abs_tol=1.0e-15):
            raise ValueError("accepted selected alpha differs from the recorded trial")
        schedule = [alpha * math.ldexp(1.0, count - 1 - i)
                    for i in range(count)]
        initial_bound = schedule[0]
        if not math.isclose(schedule[-1], alpha, rel_tol=2.0e-5, abs_tol=1.0e-15):
            raise ValueError("accepted Armijo alpha schedule is inconsistent")
        return {
            "outcome": "accepted_trial",
            "accepted": True,
            "certified_noop": False,
            "initial_bound": initial_bound,
            "trial_count": count,
            "selected_alpha": alpha,
            "last_tested_alpha": alpha,
            "alpha_schedule": schedule,
            "trial_status_code_is_authoritative": False,
            "trial_status_interpretation": (
                "the trial status arena can contain a later no-op/sentinel status; "
                "accepted=true and the selected alpha record decide this iteration"
            ),
        }

    if len(diagnostics) < 4:
        raise ValueError("failed Armijo record lacks failure diagnostics")
    initial_bound = float(raw[2])
    count = int(round(float(diagnostics[3])))
    last_alpha = float(diagnostics[2])
    if initial_bound <= 0.0 or count < 1:
        raise ValueError("failed Armijo record has an invalid bound or trial count")
    schedule = [initial_bound * math.ldexp(1.0, -i) for i in range(count)]
    if not math.isclose(schedule[-1], last_alpha, rel_tol=3.0e-5,
                        abs_tol=1.0e-15):
        raise ValueError("failed Armijo alpha schedule does not reproduce its final trial")
    return {
        "outcome": "rejected_line_search",
        "accepted": False,
        "certified_noop": False,
        "initial_bound": initial_bound,
        "trial_count": count,
        "selected_alpha": 0.0,
        "last_tested_alpha": last_alpha,
        "alpha_schedule": schedule,
        "trial_status_code_is_authoritative": False,
    }


def build_report(capture: Path, prediction_path: Path, snapshot: Path,
                 build_dir: Path, root: int, iteration: int):
    prediction = json.loads(prediction_path.read_text())
    if Path(prediction["source_snapshot"]) != snapshot:
        raise ValueError("capture prediction and requested frozen source snapshot differ")
    stages, armijo = parse_trace(capture, root, iteration)
    arenas = stages["before_contact_limits"]["arenas"]
    candidate_bytes = bytes.fromhex(arenas["candidate"])
    solution_bytes = bytes.fromhex(arenas["solution"])
    sample_bytes = bytes.fromhex(arenas["samples"])
    node_count = len(candidate_bytes) // 80
    pair_count = len(sample_bytes) // 160
    if len(candidate_bytes) % 80 or len(solution_bytes) != node_count * 16 or len(sample_bytes) % 160:
        raise ValueError("captured arena dimensions do not match frozen ABI node/sample records")

    fixture_path = snapshot / "matter/tools/cardboard_full_contact_armijo_check.mm"
    tooling_path = snapshot / "matter/tools/cardboard_tooling.hpp"
    contact_path = snapshot / "matter/src/metal/contact.metalinc"
    fem_path = snapshot / "matter/src/metal/fem.metalinc"
    material_path = snapshot / "matter/materials/hajali2009_liner_hill_ideal.nmatter"
    source = fixture_path.read_text()
    tooling = tooling_path.read_text()
    material = material_path.read_text()
    contact_source = contact_path.read_text()
    captured_includes_capsule_geometry = (
        "ipcCapsuleSideGeometricHessian(" in contact_source
    )
    # The fixture authors the frame step as a literal in makeWorld.
    timestep_match = re.search(r"source\.frameTimestep\s*=\s*([0-9.eE+-]+)\s*;", source)
    if not timestep_match:
        raise ValueError("frozen fixture timestep spelling changed; update parser")
    dt = float(timestep_match.group(1))
    board_thickness = parse_constant(source, "kPadThickness")
    radius = parse_constant(source, "kPunchRadius")
    width = parse_constant(source, "kPadWidth")
    overhang = parse_constant(source, "kPunchEndOverhang") if "constexpr double kPunchEndOverhang" in source else 0.1 * parse_constant(source, "kMm")
    initial_clearance = parse_constant(source, "kInitialClearance")
    support_clearance = parse_constant(source, "kSupportClearance")
    tool_speed = parse_constant(source, "kToolSpeed")
    contact_slop = 10.0e-6
    if "source.contactSlop = 10.0e-6;" not in source:
        raise ValueError("frozen fixture contact thickness spelling changed")
    pose_z = f32(-support_clearance + board_thickness + radius + initial_clearance)
    endpoint_delta_z = f32(-tool_speed * dt)
    capsule_center_z = f32(pose_z + endpoint_delta_z)
    half_span = width * 0.5 + overhang - radius

    interface = re.search(r"interface\s*\{([^}]*)\}", material, re.S)
    if not interface:
        raise ValueError("material friction interface block missing")
    friction = {}
    for name in ("static_friction", "dynamic_friction"):
        match = re.search(rf"{name}\s*=\s*([0-9.eE+-]+)", interface.group(1))
        if not match:
            raise ValueError(f"material {name} missing")
        friction[name] = float(match.group(1))
    if friction != {"static_friction": 0.0, "dynamic_friction": 0.0}:
        raise ValueError("this replay only supports the captured zero-friction source")

    # The fixture has one material on both sides; its source uses a fixed
    # kinematic body moving down at 0.2 m/s.
    relative_tool_velocity = [0.0, 0.0, -tool_speed]
    capsule_axis = [0.0, 1.0, 0.0]
    selected = []
    for pair_id in range(pair_count):
        identity, _ = unpack_sample(sample_bytes, pair_id)
        if identity[0] in (22, 29) and identity[1] == 0:
            selected.append((pair_id, identity))
    if selected != [(42, (22, 0, 0, 1)), (56, (29, 0, 0, 1))]:
        raise ValueError(f"captured contact identities differ from selected source samples: {selected}")

    epsilon_values = [0.01, 0.001, 0.0001]
    contacts = []
    for pair_id, identity in selected:
        node = unpack_node(candidate_bytes, identity[0])
        correction = struct.unpack_from("<4f", solution_bytes, identity[0] * 16)[:3]
        contacts.append({
            "pair_slot": pair_id,
            **analyze_contact((identity, struct.unpack_from("<36f", sample_bytes, pair_id * 160 + 16)),
                              node, correction, dt, radius, contact_slop,
                              relative_tool_velocity, capsule_axis,
                              [0.0, 0.0, capsule_center_z], epsilon_values,
                              captured_includes_capsule_geometry),
        })
    armijo_env = armijo["environments"][0]
    armijo_result = decode_armijo_result(armijo_env)
    initial_bound = armijo_result["initial_bound"]
    alpha_values = armijo_result["alpha_schedule"]
    last_alpha = armijo_result["last_tested_alpha"]

    fp32_trials = []
    for pair_id, identity in selected:
        node = unpack_node(candidate_bytes, identity[0])
        correction = struct.unpack_from("<4f", solution_bytes, identity[0] * 16)[:3]
        fp32_trials.append({
            "node_id": identity[0],
            "accepted_displacement_assumption": "reconstructed from captured candidate u/v and the frozen trial update relation; not forced to zero",
            "trial_update": "v'=v+alpha*dV; u'=accepted_u+dt*v'; x'=rest+u' (binary32 emulation)",
            **representability(node, correction, alpha_values, dt,
                               capsule_center_z, radius, half_span),
        })

    captured_contact_action = [0.0, 0.0, 0.0]
    full_contact_action = [0.0, 0.0, 0.0]
    for contact in contacts:
        captured_contact_action = add(
            captured_contact_action,
            contact["captured_sample_operator_action_Ns_per_mps"])
        full_contact_action = add(
            full_contact_action,
            contact["full_capsule_directional_action_Ns_per_mps"])

    receipt_path = capture.with_name("receipt.json")
    native_run = None
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        native_run = {
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256(receipt_path),
            "execution_started": receipt.get("execution_started"),
            "source_inputs_unchanged": receipt.get("source_inputs_unchanged"),
            "exit_code": receipt.get("exit_code"),
            "timed_out": receipt.get("timed_out"),
            "wall_seconds": receipt.get("wall_seconds"),
            "run_interpretation": (
                "the selected Newton trial may be accepted even when the native "
                "process later exits unsuccessfully; this receipt is run-level"
            ),
        }

    build_manifest = json.loads((build_dir / "source-build.json").read_text())
    expected_hashes = prediction.get("inputs_sha256", {})
    bound_paths = {
        "contact_metal": contact_path,
        "fem_metal": fem_path,
        "fixture": fixture_path,
        "tooling": tooling_path,
        "material": material_path,
    }
    source_bindings = {}
    for key, path in bound_paths.items():
        digest = sha256(path)
        rel = str(path.relative_to(snapshot)).replace("\\", "/")
        expected = expected_hashes.get("source/" + rel)
        if expected != digest:
            raise ValueError(f"frozen source hash does not match prediction for {rel}")
        source_bindings[key] = {"path": str(path), "sha256": digest}
    artifacts = build_manifest.get("artifacts_sha256", {})
    binary_path = build_dir / "numi-matter-cardboard-full-contact-armijo-check"
    metallib_path = build_dir / "shaders/NumiMatter.metallib"
    if artifacts.get(str(binary_path)) != sha256(binary_path):
        raise ValueError("retained executable hash does not match source-build manifest")
    if artifacts.get(str(metallib_path)) != sha256(metallib_path):
        raise ValueError("retained Metal library hash does not match source-build manifest")

    return {
        "schema": "numi.cardboard.rigid-contact-derivative-replay.v2",
        "classification": "local_derivative_diagnosis_only_not_global_solver_causality_or_physical_qualification",
        "capture": {
            "path": str(capture),
            "sha256": sha256(capture),
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256(prediction_path),
            "root": root,
            "newton_iteration": iteration,
            "captured_stages": sorted(stages),
            "same_candidate_solution_samples_across_stages": True,
            "armijo_record": armijo,
            "armijo_interpretation": armijo_result,
        },
        "native_run": native_run,
        "build_binding": {
            "source_snapshot": str(snapshot),
            "source_build_manifest": str(build_dir / "source-build.json"),
            "source_build_manifest_sha256": sha256(build_dir / "source-build.json"),
            "artifact_sha256": {
                str(binary_path): sha256(binary_path),
                str(metallib_path): sha256(metallib_path),
            },
            "source_files": source_bindings,
        },
        "scope": {
            "model": "one FEM free node against one prescribed identity-pose capsule, capsule axis world +Y; local node contact sample replay only",
            "dt_s": dt,
            "capsule_radius_m": radius,
            "capsule_centerline_center_z_m": capsule_center_z,
            "capsule_centerline_halfspan_y_m": half_span,
            "capsule_velocity_world_m_per_s": relative_tool_velocity,
            "slop_m": contact_slop,
            "friction_coefficients": friction,
            "accepted_material_history_mutation": False,
            "replay_advances_simulation": False,
        },
        "derivative_interpretation": {
            "system_action": "negative derivative of the node impulse residual with respect to velocity correction",
            "captured_operator": (
                "radial barrier curvature plus signed capsule-side geometric curvature; "
                "friction is zero here"
                if captured_includes_capsule_geometry else
                "PSD radial barrier curvature; friction is zero here"
            ),
            "capsule_exact_smooth_side_hessian": "radial curvature n*n^T plus Eprime/rho*(I-axis*axis^T-n*n^T); Eprime=-normal_force, rho=gap+radius",
            "signed_geometric_curvature_status": (
                "included in the captured operator for this source snapshot"
                if captured_includes_capsule_geometry else
                "omitted by the captured radial-only operator; negative semidefinite on transverse tangent directions"
            ),
            "stiffness_branch": "inertial max branch is active in both selected samples; derivative of inactive impact stiffness is zero locally",
        },
        "contacts": contacts,
        "selected_contacts_total_action": {
            "contact_node_ids": [row["node_id"] for row in contacts],
            "captured_operator_action_sum_Ns_per_mps": captured_contact_action,
            "independent_fp64_full_capsule_action_sum_Ns_per_mps": full_contact_action,
            "relative_error": rel_norm_error(
                captured_contact_action, full_contact_action),
            "scope": "sum of the two selected local sample actions, not the complete assembled contact or FEM residual",
        },
        "fp32_armijo_representability": {
            "initial_bound": initial_bound,
            "outcome": armijo_result["outcome"],
            "accepted": armijo_result["accepted"],
            "certified_noop": armijo_result["certified_noop"],
            "recorded_trial_count": armijo_result["trial_count"],
            "recorded_last_trial_alpha": last_alpha,
            "selected_alpha": armijo_result["selected_alpha"],
            "alpha_schedule": alpha_values,
            "nodes": fp32_trials,
            "interpretation": "binary32 arithmetic emulation reconstructs the accepted state from captured candidate u/v; it is not a second Metal execution",
        },
        "limitations": [
            "This local capsule calculation excludes the assembled FEM stiffness, other contacts, cohesive terms, and global residual interactions.",
            "The finite-difference replay is continuous FP64 for the captured smooth capsule-side feature. It does not reproduce all Metal rounding or feature-switch guards.",
            "A local 1.8% operator discrepancy is not evidence by itself that it caused the global nonlinear plateau.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--root", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new evidence filename")
    report = build_report(args.capture, args.prediction, args.snapshot,
                          args.build_dir, args.root, args.iteration)
    report["analyzer"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "capture_sha256": report["capture"]["sha256"],
        "armijo_outcome": report["capture"]["armijo_interpretation"]["outcome"],
        "accepted": report["capture"]["armijo_interpretation"]["accepted"],
        "native_run_exit_code": (report["native_run"] or {}).get("exit_code"),
        "contacts": [{
            "node_id": c["node_id"],
            "stiffness_branch": c["stiffness_branch"],
            "impact_to_inertial_stiffness_ratio": c["impact_to_inertial_stiffness_ratio"],
            "geometric_action_over_radial_action": c["geometric_action_over_radial_action"],
            "fd_full_hessian_error_at_1e-3": c["central_difference"][1]["relative_error_vs_full_capsule_hessian"],
        } for c in report["contacts"]],
        "selected_or_last_alpha": report["fp32_armijo_representability"]["recorded_last_trial_alpha"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
