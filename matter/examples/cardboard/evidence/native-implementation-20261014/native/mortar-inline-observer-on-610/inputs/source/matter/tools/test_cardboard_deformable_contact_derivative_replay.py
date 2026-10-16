#!/usr/bin/env python3
"""Portable synthetic codec and derivative checks for the contact replay."""
import json
import math
import struct
import unittest

import cardboard_deformable_contact_derivative_replay as replay
from cardboard_deformable_contact_derivative_replay import (
    CONTACT_STRIDE,
    analyze_capture,
    decode_contact,
    parse_capture_line,
)


def put_u4(buf, offset, values):
    struct.pack_into("<4I", buf, offset, *values)


def put_f4(buf, offset, values):
    struct.pack_into("<4f", buf, offset, *values)


def contact_bytes(separation=0.005, friction=0.0):
    dt, thickness, stiffness = 1.0e-3, 1.0e-2, 1.0e6
    d = separation
    d2, h2 = d * d, thickness * thickness
    off = d2 - h2
    import math
    logterm = math.log(d2 / h2)
    first = -stiffness * (2.0 * off * logterm + off * off / d2)
    second = -stiffness * (2.0 * logterm + 4.0 * off / d2 -
                           off * off / (d2 * d2))
    force = -2.0 * d * first
    hessian = max(4.0 * d2 * second + 2.0 * first, 0.0)
    buf = bytearray(CONTACT_STRIDE)
    put_u4(buf, 0, [0, 1, 0xFFFFFFFF, 0xFFFFFFFF])
    put_u4(buf, 16, [0, 0, 2, 1])  # valid point-point
    put_f4(buf, 32, [1.0, -1.0, 0.0, 0.0])
    put_f4(buf, 48, [1.0, 1.0, 0.0, 0.0])
    put_f4(buf, 64, [-1.0, 0.0, 0.0, 0.0])
    put_f4(buf, 96, [0.0, 0.0, 0.0, 1.0])
    put_f4(buf, 128, [force * dt, 0.0, 0.0, friction])
    put_f4(buf, 144, [force * dt, hessian * dt * dt, thickness, stiffness])
    put_f4(buf, 160, [hessian * dt * dt, 0.0, 0.0, 0.0])
    put_f4(buf, 176, [0.0, 0.0, 0.0, 0.0])
    put_f4(buf, 192, [0.0, 0.0, 0.0, 0.0])
    put_f4(buf, 272, [1.0, 0.0, 0.0, stiffness])
    put_u4(buf, 288, [0xFFFFFFFF, 0xFFFFFFFF, 0, 0])
    put_f4(buf, 304, [0.0, 0.0, 0.0, 0.0])
    put_f4(buf, 320, [0.0, 0.0, 0.0, 0.0])
    put_u4(buf, 368, [0, 0, 0, 0])
    return bytes(buf)


def node_bytes(separation=0.005):
    rows = [
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0),
        (separation, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
         separation, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0),
    ]
    return b"".join(struct.pack("<20f", *row) for row in rows)


def solution_bytes():
    return (struct.pack("<4f", 0.5, 0.0, 0.0, 0.0) +
            struct.pack("<4f", -0.5, 0.0, 0.0, 0.0))


def capture_line(separation=0.005, friction=0.0, truncated=False,
                 exact_action_capture=False, exact_markers=None,
                 exact_actions=None):
    contact = contact_bytes(separation, friction)
    nodes, solution = node_bytes(separation), solution_bytes()
    envs, slots, object_count, record_stride = 1, 2, 1, 900
    count_pad = 16
    record_bytes = envs * slots * record_stride
    line_bytes = envs * object_count * 16
    total = record_bytes + count_pad + line_bytes + len(nodes) + len(solution)
    record = {
        "environment": 0, "active_slot": 0, "contact_slot": 3,
        "identity": list(struct.unpack_from("<4I", contact, 16)),
        "record_hex": contact.hex(),
    }
    if exact_action_capture:
        record["production_exact_action_dt_s"] = 1.0e-3
        record["production_exact_action_supported"] = (
            [1, 1, 1, 1] if exact_markers is None else exact_markers)
        record["production_exact_action"] = (
            [[float(i + 1), float(2 * i), float(-i)] for i in range(4)]
            if exact_actions is None else exact_actions)
    row = {
        "schema": 1, "root": 7, "microtick": 0, "iteration": 42,
        "stage": "after_deformable_limit", "dt_s": 1.0e-3,
        "environment_count": envs, "fem_node_count": 2,
        "grid_node_count": 0, "object_count": object_count,
        "deformable_contact_capacity": 8, "trace_slots_per_environment": slots,
        "capture_bytes": total, "record_stride_bytes": record_stride,
        "contact_stride_bytes": CONTACT_STRIDE, "node_state_stride_bytes": 80,
        "solution_stride_bytes": 16, "candidate_node_bytes": len(nodes),
        "solution_bytes": len(solution), "active_counts": [3 if truncated else 1],
        "truncated": [truncated], "candidate_nodes_hex": nodes.hex(),
        "solution_hex": solution.hex(), "contacts": [record],
        "recorded_count": 1,
    }
    if exact_action_capture:
        row["production_exact_action_capture"] = True
        row["production_exact_action_source"] = (
            "deformableExactVertexTriangleDirectionalForce")
        row["production_exact_action_direction"] = "femSolution"
        row["production_exact_action_support_order"] = "contact.nodes[0..3]"
        row["production_exact_action_quantity"] = "native_fgmres_operator_action"
    return "fem_deformable_contact_capture=" + json.dumps(row)


def fixed_feature_vt_capture_line(policy):
    contact, positions, velocities, inverse_masses, _, dt = make_vt_contact(
        [0.2, 2.0e-6])
    direction = [
        [0.4, -0.3, 0.2], [-0.2, 0.5, -0.1],
        [0.3, 0.1, 0.25], [-0.15, -0.2, -0.35],
    ]
    action, reason = replay._exact_vt_operator_action(
        contact, positions, velocities, direction, inverse_masses, dt,
        allow_fixed_feature=True)
    if reason is not None:
        raise AssertionError(f"synthetic fixed-feature action unavailable: {reason}")

    raw_contact = bytearray(CONTACT_STRIDE)
    put_u4(raw_contact, 0, [0, 1, 2, 3])
    put_u4(raw_contact, 16, contact["identity"])
    put_f4(raw_contact, 32, contact["weights"][:4])
    put_f4(raw_contact, 48, contact["time_scales"][:4])
    put_f4(raw_contact, 64, contact["normal_toi"])
    put_f4(raw_contact, 80, contact["point_and_separation"])
    put_f4(raw_contact, 96, contact["relative_velocity_response"])
    put_f4(raw_contact, 112, [0.0] * 4)
    put_f4(raw_contact, 128, contact["impulse_friction"])
    put_f4(raw_contact, 144, contact["barrier"])
    for offset in (160, 176, 192):
        put_f4(raw_contact, offset, [0.0] * 4)
    for offset in (208, 224, 240, 256, 336, 352):
        put_f4(raw_contact, offset, [0.0] * 4)
    put_f4(raw_contact, 272, contact["mollifier"])
    put_u4(raw_contact, 288, [0xFFFFFFFF, 0xFFFFFFFF, 0, 0])
    put_u4(raw_contact, 368, [0, 0, 0, 0])
    put_f4(raw_contact, 384, [0.0] * 4)
    put_f4(raw_contact, 400, [0.0] * 4)
    put_f4(raw_contact, 416, [0.0] * 4)

    node_rows = []
    for i in range(4):
        position, velocity = positions[i], velocities[i]
        node_rows.append(struct.pack(
            "<20f", *position, 1.0, *velocity, inverse_masses[i],
            *position, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    nodes = b"".join(node_rows)
    solution = b"".join(struct.pack("<4f", *value, 0.0) for value in direction)
    record_stride, slots, object_count = 448, 1, 1
    capture_bytes = (record_stride + 16 + 16 + len(nodes) + len(solution))
    markers = [1, 1, 1, 1] if policy == "fixed_candidate_feature" else [0] * 4
    actions = action[:4] if policy == "fixed_candidate_feature" else [[0.0] * 3] * 4
    raw = bytes(raw_contact)
    row = {
        "schema": 1, "root": 7, "microtick": 0, "iteration": 42,
        "stage": "after_deformable_limit", "dt_s": dt,
        "environment_count": 1, "fem_node_count": 4, "grid_node_count": 0,
        "object_count": object_count, "deformable_contact_capacity": 4,
        "trace_slots_per_environment": slots, "capture_bytes": capture_bytes,
        "record_stride_bytes": record_stride,
        "contact_stride_bytes": CONTACT_STRIDE, "node_state_stride_bytes": 80,
        "solution_stride_bytes": 16, "candidate_node_bytes": len(nodes),
        "solution_bytes": len(solution), "active_counts": [1],
        "truncated": [False], "candidate_nodes_hex": nodes.hex(),
        "solution_hex": solution.hex(), "recorded_count": 1,
        "contacts": [{"environment": 0, "active_slot": 0, "contact_slot": 3,
                      "identity": contact["identity"], "record_hex": raw.hex(),
                      "production_exact_action_dt_s": dt,
                      "production_exact_action_supported": markers,
                      "production_exact_action": actions}],
        "production_exact_action_capture": True,
        "production_exact_action_source":
            "deformableExactVertexTriangleDirectionalForce",
        "production_exact_action_direction": "femSolution",
        "production_exact_action_support_order": "contact.nodes[0..3]",
        "production_exact_action_quantity": "native_fgmres_operator_action",
        "production_exact_action_feature_policy": policy,
    }
    return "fem_deformable_contact_capture=" + json.dumps(row)


def make_vt_contact(point_xy):
    dt, thickness = 1.0e-3, 1.0e-2
    positions = [list(point_xy) + [0.004], [0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    _, bary, feature = replay._triangle_closest(*positions[:4])
    inverse_masses = [1.2, 0.7, 1.4, 0.9, 0.0, 0.0]
    velocities = [[0.0, 0.0, 0.0] for _ in range(6)]
    weights = [1.0, -bary[0], -bary[1], -bary[2], 0.0, 0.0]
    response = sum(weights[i] ** 2 * inverse_masses[i] for i in range(4))
    delta = replay.sub(positions[0], replay._triangle_closest(*positions[:4])[0])
    separation = replay.norm(delta)
    normal = replay.scale(1.0 / separation, delta)
    stiffness, branch = replay._stiffness_from_state(
        {"barrier": [0.0, 0.0, thickness, 0.0]}, normal, separation,
        response, [0.0, 0.0, 0.0], dt)
    if branch != "inertial":
        raise AssertionError("synthetic fixture did not select inertial stiffness")
    _, first, second, _ = replay._barrier_exact_terms(
        separation, thickness, stiffness)
    impulse = -dt * first
    return ({
        "identity": [0, 1, 0, replay.CONTACT_VALID],
        "weights": weights,
        "time_scales": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        "normal_toi": normal + [0.5],
        "point_and_separation": [0.0, 0.0, 0.0, separation],
        "relative_velocity_response": [0.0, 0.0, 0.0, response],
        "impulse_friction": [impulse, 0.0, 0.0, 0.0],
        "barrier": [impulse, max(dt * dt * second, 0.0), thickness, stiffness],
        "mollifier_gradients": [[0.0, 0.0, 0.0] for _ in range(6)],
        "mollifier": [1.0, 0.0, 0.0, stiffness],
    }, positions, velocities, inverse_masses, feature, dt)


class ExactVertexTriangleDerivativeTests(unittest.TestCase):
    def test_fixed_center_feature_margin_action_is_linear_and_matches_both_sides(self):
        # The point is inside a nondegenerate face but within the production
        # FP32 barycentric safety margin. The ordinary exact-action admission
        # must still reject it; this opt-in CPU diagnostic evaluates the
        # Jacobian of the center-selected face branch only.
        contact, positions, velocities, inv_mass, feature, dt = make_vt_contact(
            [0.2, 2.0e-6])
        self.assertEqual(feature, "face")
        direction_a = [
            [0.4, -0.3, 0.2], [-0.2, 0.5, -0.1],
            [0.3, 0.1, 0.25], [-0.15, -0.2, -0.35],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        ]
        direction_b = [
            [-0.1, 0.1, 0.05], [0.2, -0.15, 0.0],
            [-0.05, 0.0, 0.1], [0.0, 0.15, -0.1],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        ]
        strict, strict_reason = replay._exact_vt_operator_action(
            contact, positions, velocities, direction_a, inv_mass, dt)
        self.assertIsNone(strict)
        self.assertEqual(strict_reason, "feature_boundary_or_degenerate")

        action_a, reason_a = replay._exact_vt_operator_action(
            contact, positions, velocities, direction_a, inv_mass, dt,
            allow_fixed_feature=True)
        action_b, reason_b = replay._exact_vt_operator_action(
            contact, positions, velocities, direction_b, inv_mass, dt,
            allow_fixed_feature=True)
        action_sum, reason_sum = replay._exact_vt_operator_action(
            contact, positions, velocities,
            [replay.add(a, b) for a, b in zip(direction_a, direction_b)],
            inv_mass, dt, allow_fixed_feature=True)
        self.assertIsNone(reason_a)
        self.assertIsNone(reason_b)
        self.assertIsNone(reason_sum)
        self.assertLess(replay._relative_error(
            action_sum, [replay.add(a, b) for a, b in zip(action_a, action_b)]),
            2.0e-12)

        center, center_state = replay._residual_impulse(
            contact, positions, dt, inv_mass, velocities)
        self.assertEqual(center_state["feature_branch"], "face")
        for epsilon in (1.0e-3, 1.0e-4):
            residuals, side_states = [], []
            for sign in (+1.0, -1.0):
                side_positions = [replay.add(positions[i], replay.scale(
                    sign * dt * epsilon * contact["time_scales"][i],
                    direction_a[i])) for i in range(6)]
                side_velocities = [replay.add(velocities[i], replay.scale(
                    sign * epsilon, direction_a[i])) for i in range(6)]
                residual, state = replay._residual_impulse(
                    contact, side_positions, dt, inv_mass, side_velocities)
                residuals.append(residual)
                side_states.append(state)
            self.assertEqual([s["feature_branch"] for s in side_states],
                             ["face", "face"])
            fd = [[-(residuals[0][i][axis] - residuals[1][i][axis]) /
                   (2.0 * epsilon) for axis in range(3)] for i in range(6)]
            self.assertLess(replay._relative_error(fd, action_a), 2.0e-5)

    def test_fixed_center_feature_comparison_refuses_changed_side(self):
        positions = [[0.0, 0.0, 0.0] for _ in range(6)]
        direction = [[1.0, 0.0, 0.0] for _ in range(6)]
        residual = [[0.0, 0.0, 0.0] for _ in range(6)]
        center = {"feature_branch": "face", "mollifier_branch": "one",
                  "barrier_branch": "active"}
        side = dict(center, feature_branch="edge01")
        fixed = [[1.0, 0.0, 0.0] for _ in range(6)]
        result = replay._one_sided_description(
            {}, positions, direction, residual, residual, side, center,
            "inertial", "inertial", 1.0e-3, 1.0e-4, +1.0,
            fixed_feature_action=fixed)
        self.assertEqual(
            result["fixed_center_feature_action_comparison"]["status"],
            "branch_or_stiffness_changed")
        self.assertIsNone(
            result["fixed_center_feature_action_comparison"]["relative_error"])

    def test_fixed_feature_mode_keeps_degeneracy_and_material_branch_guards(self):
        contact, positions, velocities, inv_mass, _, dt = make_vt_contact(
            [0.2, 2.0e-6])
        direction = [
            [0.4, -0.3, 0.2], [-0.2, 0.5, -0.1],
            [0.3, 0.1, 0.25], [-0.15, -0.2, -0.35],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        ]

        degenerate = [list(row) for row in positions]
        degenerate[3] = [2.0, 0.0, 0.0]
        action, _ = replay._exact_vt_operator_action(
            contact, degenerate, velocities, direction, inv_mass, dt,
            allow_fixed_feature=True)
        self.assertIsNone(action)

        nonunit = dict(contact, mollifier=[0.5, 0.0, 0.0, contact["mollifier"][3]])
        action, reason = replay._exact_vt_operator_action(
            nonunit, positions, velocities, direction, inv_mass, dt,
            allow_fixed_feature=True)
        self.assertIsNone(action)
        self.assertEqual(reason, "nonunit_mollifier")

        impact_velocity = [list(row) for row in velocities]
        impact_velocity[0] = [0.0, 0.0, -100.0]
        action, reason = replay._exact_vt_operator_action(
            contact, positions, impact_velocity, direction, inv_mass, dt,
            allow_fixed_feature=True)
        self.assertIsNone(action)
        self.assertEqual(reason, "impact_or_stiffness_switch")

        floor_positions = [list(row) for row in positions]
        floor_positions[0][2] = 1.0e-7
        action, reason = replay._exact_vt_operator_action(
            contact, floor_positions, velocities, direction, inv_mass, dt,
            allow_fixed_feature=True)
        self.assertIsNone(action)
        self.assertEqual(reason, "barrier_floor_or_inactive")

    def test_one_sided_replay_preserves_sign_and_never_claims_unsupported_metal(self):
        positions = [[0.0, 0.0, 0.0] for _ in range(6)]
        direction = [[1.0, 0.0, 0.0] for _ in range(6)]
        center_residual = [[0.0, 0.0, 0.0] for _ in range(6)]
        center = {"feature_branch": "face", "mollifier_branch": "one",
                  "barrier_branch": "active"}
        side = dict(center)
        plus_residual = [[-2.0e-4, 0.0, 0.0] for _ in range(6)]
        result = replay._one_sided_description(
            {}, positions, direction, center_residual, plus_residual,
            side, center, "inertial", "inertial", 1.0e-3, 1.0e-4, +1.0)
        self.assertEqual(result["status"], "same_discrete_branch")
        self.assertEqual(result["classification"],
                         "descriptive_one_sided_replay_not_qualification")
        self.assertAlmostEqual(result["fp64_replayed_action_norm"],
                               math.sqrt(6.0) * 2.0)
        self.assertEqual(result["production_action_comparison"]["status"],
                         "native_support_unavailable_for_this_branch")

        changed = dict(side, feature_branch="edge01")
        rejected = replay._one_sided_description(
            {}, positions, direction, center_residual, plus_residual,
            changed, center, "inertial", "inertial", 1.0e-3, 1.0e-4, +1.0)
        self.assertEqual(rejected["status"], "branch_or_stiffness_changed")

    def test_triangle_condition_separates_area_guard_from_feature_margin(self):
        nondegenerate = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        degenerate = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        good = replay._triangle_geometry_condition(nondegenerate)
        bad = replay._triangle_geometry_condition(degenerate)
        self.assertTrue(good["above_replay_float32_degeneracy_guard"])
        self.assertFalse(bad["above_replay_float32_degeneracy_guard"])
        self.assertGreater(good["gram_determinant_over_guard"], 1.0)
        self.assertEqual(bad["gram_determinant_over_guard"], 0.0)

    def test_exact_directional_action_matches_recomputed_residual_on_all_features(self):
        cases = (
            ("face", [0.2, 0.3]),
            ("edge01", [0.2, -0.002]),
            ("edge02", [-0.002, 0.2]),
            ("edge12", [0.502, 0.502]),
            ("vertex0", [-0.002, -0.002]),
            ("vertex1", [1.002, -0.002]),
            ("vertex2", [-0.002, 1.002]),
        )
        direction = [
            [0.4, -0.3, 0.2], [-0.2, 0.5, -0.1],
            [0.3, 0.1, 0.25], [-0.15, -0.2, -0.35],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        ]
        epsilon = 1.0e-4
        for expected_feature, point_xy in cases:
            with self.subTest(feature=expected_feature):
                contact, positions, velocities, inv_mass, feature, dt = \
                    make_vt_contact(point_xy)
                # Exercise the owner-timestep chain rule as well as the
                # geometric tangent; the contact impulse response itself is
                # evaluated from unscaled physical node velocities.
                contact["time_scales"] = [1.0, 0.75, 1.1, 0.5, 0.0, 0.0]
                self.assertEqual(feature, expected_feature)
                exact, reason = replay._exact_vt_operator_action(
                    contact, positions, velocities, direction, inv_mass, dt)
                self.assertIsNone(reason)
                plus = [replay.add(positions[i], replay.scale(
                    dt * epsilon * contact["time_scales"][i], direction[i]))
                    for i in range(6)]
                minus = [replay.sub(positions[i], replay.scale(
                    dt * epsilon * contact["time_scales"][i], direction[i]))
                    for i in range(6)]
                plus_v = [replay.add(velocities[i], replay.scale(
                    epsilon, direction[i])) for i in range(6)]
                minus_v = [replay.sub(velocities[i], replay.scale(
                    epsilon, direction[i])) for i in range(6)]
                plus_r, plus_state = replay._residual_impulse(
                    contact, plus, dt, inv_mass, plus_v)
                minus_r, minus_state = replay._residual_impulse(
                    contact, minus, dt, inv_mass, minus_v)
                self.assertEqual(plus_state["feature_branch"], expected_feature)
                self.assertEqual(minus_state["feature_branch"], expected_feature)
                self.assertAlmostEqual(
                    plus_state["stiffness"] / minus_state["stiffness"],
                    1.0, delta=1.0e-3)
                fd = [[-(plus_r[i][axis] - minus_r[i][axis]) /
                       (2.0 * epsilon) for axis in range(3)]
                      for i in range(6)]
                self.assertLess(replay._relative_error(fd, exact), 1.0e-6)

    def test_exact_action_includes_inertial_stiffness_response_derivative(self):
        contact, positions, velocities, inv_mass, _, dt = make_vt_contact(
            [0.2, -0.002])
        direction = [
            [0.4, -0.3, 0.2], [-0.2, 0.5, -0.1],
            [0.3, 0.1, 0.25], [-0.15, -0.2, -0.35],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        ]
        exact, reason = replay._exact_vt_operator_action(
            contact, positions, velocities, direction, inv_mass, dt)
        self.assertIsNone(reason)
        epsilon = 1.0e-4
        sides = []
        for sign in (1.0, -1.0):
            side_positions = [replay.add(positions[i], replay.scale(
                sign * dt * epsilon, direction[i])) for i in range(6)]
            side_velocities = [replay.add(velocities[i], replay.scale(
                sign * epsilon, direction[i])) for i in range(6)]
            sides.append((side_positions, side_velocities))
        dynamic_rows = [replay._residual_impulse(
            contact, p, dt, inv_mass, v)[0] for p, v in sides]
        frozen_rows = [replay._residual_impulse(contact, p, dt)[0]
                       for p, _ in sides]
        dynamic_fd = [[-(dynamic_rows[0][i][axis] - dynamic_rows[1][i][axis]) /
                       (2.0 * epsilon) for axis in range(3)] for i in range(6)]
        frozen_fd = [[-(frozen_rows[0][i][axis] - frozen_rows[1][i][axis]) /
                      (2.0 * epsilon) for axis in range(3)] for i in range(6)]
        self.assertLess(replay._relative_error(dynamic_fd, exact), 1.0e-6)
        self.assertGreater(replay._relative_error(frozen_fd, exact), 1.0e-5)

    def test_feature_transition_is_not_admitted_to_exact_action(self):
        contact, positions, velocities, inv_mass, _, dt = make_vt_contact(
            [0.2, 0.0])
        direction = [[0.1, -0.2, 0.3] for _ in range(4)] + [
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        exact, reason = replay._exact_vt_operator_action(
            contact, positions, velocities, direction, inv_mass, dt)
        self.assertIsNone(exact)
        self.assertEqual(reason, "feature_boundary_or_degenerate")


class DeformableContactCaptureCodecTests(unittest.TestCase):
    def test_record_layout_and_total_bytes_are_verified(self):
        capture = parse_capture_line(capture_line(), root=7, iteration=42)
        self.assertEqual(capture["_complete_active_coverage"], True)
        self.assertEqual(capture["contacts"][0]["_decoded"]["nodes"][:2], (0, 1))
        report = analyze_capture(capture, epsilons=(1e-3,))
        self.assertEqual(report["classification"],
                         "diagnostic_only_not_solver_or_physical_qualification")
        self.assertEqual(report["counts"]["analyzed"], 1)
        self.assertEqual(report["contacts"][0]["production_exact_vt_action"]["status"],
                         "not_captured")

    def test_production_action_capture_compares_all_four_supports(self):
        capture = parse_capture_line(capture_line(exact_action_capture=True))
        record = capture["contacts"][0]
        exact = [record["_production_exact_action"][i] for i in range(4)] + [
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        comparison = replay._compare_production_exact_action(
            capture, record, exact, None)
        self.assertEqual(comparison["status"], "compared")
        self.assertEqual(comparison["supported_support_nodes"], 4)
        self.assertEqual(comparison["relative_error"], 0.0)
        self.assertAlmostEqual(comparison["cosine"], 1.0)

    def test_production_action_partial_support_is_not_compared_as_zero(self):
        capture = parse_capture_line(capture_line(
            exact_action_capture=True, exact_markers=[1, 1, 0, 1]))
        record = capture["contacts"][0]
        exact = [[1.0, 2.0, 3.0] for _ in range(6)]
        comparison = replay._compare_production_exact_action(
            capture, record, exact, None)
        self.assertEqual(comparison["status"], "metal_branch_unsupported")
        self.assertNotIn("relative_error", comparison)

    def test_production_action_capture_rejects_malformed_payloads(self):
        malformed_marker = capture_line(
            exact_action_capture=True, exact_markers=[1, 1, 2, 1])
        with self.assertRaises(ValueError):
            parse_capture_line(malformed_marker)
        malformed_action = capture_line(
            exact_action_capture=True,
            exact_actions=[[1.0, 2.0, 3.0], [0.0, 0.0],
                           [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError):
            parse_capture_line(malformed_action)
        malformed_dt = capture_line(exact_action_capture=True).replace(
            '"production_exact_action_dt_s": 0.001',
            '"production_exact_action_dt_s": 0.0')
        with self.assertRaises(ValueError):
            parse_capture_line(malformed_dt)
        malformed_source = capture_line(exact_action_capture=True).replace(
            '"production_exact_action_direction": "femSolution"',
            '"production_exact_action_direction": "other"')
        with self.assertRaises(ValueError):
            parse_capture_line(malformed_source)

        malformed_policy = capture_line(exact_action_capture=True).replace(
            '"production_exact_action_quantity": "native_fgmres_operator_action"',
            '"production_exact_action_quantity": "native_fgmres_operator_action", '
            '"production_exact_action_feature_policy": "direction_dependent"')
        with self.assertRaises(ValueError):
            parse_capture_line(malformed_policy)

    def test_old_exact_action_capture_defaults_to_strict_feature_policy(self):
        capture = parse_capture_line(capture_line(exact_action_capture=True))
        self.assertEqual(capture["_production_exact_action_feature_policy"],
                         "strict_smooth_feature")

    def test_explicit_fixed_candidate_feature_policy_is_preserved(self):
        encoded = capture_line(exact_action_capture=True).split("=", 1)[1]
        row = json.loads(encoded)
        row["production_exact_action_feature_policy"] = "fixed_candidate_feature"
        capture = parse_capture_line("fem_deformable_contact_capture=" +
                                    json.dumps(row))
        self.assertEqual(capture["_production_exact_action_feature_policy"],
                         "fixed_candidate_feature")

    def test_capture_policy_controls_only_feature_margin_action_admission(self):
        strict = analyze_capture(parse_capture_line(
            fixed_feature_vt_capture_line("strict_smooth_feature")),
            epsilons=(1.0e-3,))
        fixed = analyze_capture(parse_capture_line(
            fixed_feature_vt_capture_line("fixed_candidate_feature")),
            epsilons=(1.0e-3,))
        strict_row, fixed_row = strict["contacts"][0], fixed["contacts"][0]
        self.assertEqual(strict_row["strict_smooth_feature_operator_status"],
                         "unsupported")
        self.assertEqual(strict_row["production_exact_vt_action"]["status"],
                         "cpu_reference_unsupported")
        self.assertEqual(fixed_row["exact_vt_operator_status"], "supported")
        self.assertEqual(fixed_row["strict_smooth_feature_operator_status"],
                         "unsupported")
        self.assertEqual(fixed_row["production_exact_vt_action"]["status"],
                         "compared")
        self.assertLess(fixed_row["production_exact_vt_action"]["relative_error"],
                        1.0e-6)
        side = fixed_row["sweep"][0]["one_sided"]
        self.assertEqual(side["plus"]["status"], "same_discrete_branch")
        self.assertEqual(side["minus"]["status"], "same_discrete_branch")
        self.assertLess(
            side["plus"]["fixed_center_feature_action_comparison"]["relative_error"],
            5.0e-5)

    def test_inconsistent_strides_slots_truncation_and_raw_identity_fail_closed(self):
        bad_stride = capture_line().replace('"record_stride_bytes": 900',
                                             '"record_stride_bytes": 400')
        with self.assertRaises(ValueError):
            parse_capture_line(bad_stride)
        bad_truncated = capture_line(truncated=True).replace('"truncated": [true]',
                                                              '"truncated": [false]')
        with self.assertRaises(ValueError):
            parse_capture_line(bad_truncated)
        bad_identity = capture_line().replace('"identity": [0, 0, 2, 1]',
                                               '"identity": [0, 0, 1, 1]')
        with self.assertRaises(ValueError):
            parse_capture_line(bad_identity)

    def test_friction_is_labeled_unsupported_instead_of_omitted(self):
        capture = parse_capture_line(capture_line(friction=0.2))
        report = analyze_capture(capture, epsilons=(1e-3,))
        self.assertEqual(report["contacts"][0]["status"], "unsupported")
        self.assertEqual(report["contacts"][0]["unsupported_reason"],
                         "friction_or_lagged_tangent_not_replayed")

    def test_smooth_point_point_fd_matches_captured_radial_action(self):
        capture = parse_capture_line(capture_line())
        report = analyze_capture(capture, epsilons=(1e-3, 1e-4, 1e-5))
        row = report["contacts"][0]
        self.assertLess(row["center_residual_reconstruction_relative_error"], 2e-7)
        smooth = [item for item in row["sweep"] if item["status"] == "smooth_branch"]
        self.assertEqual(len(smooth), 3)
        self.assertLess(smooth[0]["captured_action_relative_error_fp64"], 1e-5)
        self.assertTrue(smooth[-1]["fp32_position_sides_distinct"])
        self.assertEqual(smooth[-1]["fp32_moving_nodes"], 2)
        self.assertEqual(smooth[-1]["fp32_realized_moving_nodes"], 2)

    def test_zero_direction_is_unresolved_not_vacuously_representable(self):
        prefix, encoded = capture_line().split("=", 1)
        row = json.loads(encoded)
        row["solution_hex"] = bytes(len(bytes.fromhex(row["solution_hex"]))).hex()
        report = analyze_capture(parse_capture_line(
            prefix + "=" + json.dumps(row)), epsilons=(1e-3,))
        sample = report["contacts"][0]["sweep"][0]
        self.assertIsNone(sample["fp32_position_sides_distinct"])
        self.assertEqual(sample["fp32_moving_nodes"], 0)

    def test_barrier_floor_is_reported_as_a_branch_boundary(self):
        capture = parse_capture_line(capture_line(separation=1e-7))
        report = analyze_capture(capture, epsilons=(1e-3,))
        row = report["contacts"][0]
        self.assertEqual(row["center_feature_state"]["barrier_branch"], "floor")
        self.assertEqual(row["sweep"][0]["status"], "branch_changed_or_boundary")


if __name__ == "__main__":
    unittest.main()
