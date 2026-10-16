#!/usr/bin/env python3
"""Independent finite-difference audit for the smooth free-BOX contact Jacobian.

This is a CPU formula check, not a native Metal or solver qualification. It
covers the frictionless, strictly inertial barrier branch used by the current
free-BOX angular slice: one smooth face/edge feature, one free six-DOF body,
and no active-set transition.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
import unittest


def add(a, b): return [x + y for x, y in zip(a, b)]
def sub(a, b): return [x - y for x, y in zip(a, b)]
def scale(s, a): return [s * x for x in a]
def dot(a, b): return sum(x * y for x, y in zip(a, b))
def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]
def norm(a): return math.sqrt(dot(a, a))
def eye(): return [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
def skew(a): return [[0., -a[2], a[1]], [a[2], 0., -a[0]], [-a[1], a[0], 0.]]
def transpose(a): return [[a[j][i] for j in range(3)] for i in range(3)]
def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]
def matvec(a, x): return [dot(row, x) for row in a]
def madd(a, b): return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]
def mscale(s, a): return [[s * x for x in row] for row in a]


def quaternion_normalize(q):
    length = math.sqrt(sum(x * x for x in q))
    return [x / length for x in q]


def quaternion_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def quaternion_rotation_matrix(q):
    x, y, z, w = quaternion_normalize(q)
    return [[1.0 - 2.0 * (y * y + z * z),
             2.0 * (x * y - z * w),
             2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w),
             1.0 - 2.0 * (x * x + z * z),
             2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w),
             2.0 * (y * z + x * w),
             1.0 - 2.0 * (x * x + y * y)]]


def scene_integrate_quaternion(source, omega, dt):
    rotation_vector = scale(dt, omega)
    angle = norm(rotation_vector)
    half_angle = 0.5 * angle
    scale_factor = (math.sin(half_angle) / angle if angle > 1.0e-6
                    else 0.5 - angle * angle / 48.0)
    delta = [*(scale(scale_factor, rotation_vector)), math.cos(half_angle)]
    return quaternion_normalize(quaternion_multiply(
        delta, quaternion_normalize(source)))


def matter_pre_slice_quaternion(source, omega, dt):
    angle = norm(omega) * dt
    if angle > 1.0e-6:
        half_angle = 0.5 * angle
        delta = [*(scale(math.sin(half_angle) / norm(omega), omega)),
                 math.cos(half_angle)]
    else:
        delta = quaternion_normalize([*(scale(0.5 * dt, omega)), 1.0])
    return quaternion_normalize(quaternion_multiply(
        delta, quaternion_normalize(source)))


def free_box_angular_slice_admits(friction_limit):
    """The current angular Jacobian is frictionless-only, independent of omega."""
    return math.isfinite(friction_limit) and friction_limit == 0.0


def alpha_path_gap(p, alpha, first_point, last_point,
                   base_velocity, trial_velocity,
                   base_omega, trial_omega):
    dt = p["dt"]
    body_velocity = add(base_velocity,
                        scale(alpha, sub(trial_velocity, base_velocity)))
    omega = add(base_omega, scale(alpha, sub(trial_omega, base_omega)))
    center = add(p["c0"], scale(dt, body_velocity))
    rotation = matmul(exp_so3(scale(dt, omega)), p["rotation0"])
    point = add(first_point, scale(alpha, sub(last_point, first_point)))
    distance, *_ = box_geometry(point, center, rotation,
                                p["proxy_offset"], p["half_extent"])
    return distance


def physical_sweep_gap(p, alpha, first_point, last_point,
                       body_velocity, angular_velocity):
    dt = p["dt"]
    center = add(p["c0"], scale(alpha * dt, body_velocity))
    rotation = matmul(exp_so3(scale(alpha * dt, angular_velocity)),
                      p["rotation0"])
    point = add(first_point, scale(alpha, sub(last_point, first_point)))
    distance, *_ = box_geometry(point, center, rotation,
                                p["proxy_offset"], p["half_extent"])
    return distance


def inverse3(a):
    d = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
         - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
         + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    return [[(a[1][1]*a[2][2]-a[1][2]*a[2][1])/d,
             (a[0][2]*a[2][1]-a[0][1]*a[2][2])/d,
             (a[0][1]*a[1][2]-a[0][2]*a[1][1])/d],
            [(a[1][2]*a[2][0]-a[1][0]*a[2][2])/d,
             (a[0][0]*a[2][2]-a[0][2]*a[2][0])/d,
             (a[0][2]*a[1][0]-a[0][0]*a[1][2])/d],
            [(a[1][0]*a[2][1]-a[1][1]*a[2][0])/d,
             (a[0][1]*a[2][0]-a[0][0]*a[2][1])/d,
             (a[0][0]*a[1][1]-a[0][1]*a[1][0])/d]]


def exp_so3(v):
    theta = norm(v)
    k = skew(v)
    if theta < 1.0e-10:
        return madd(madd(eye(), k), mscale(0.5, matmul(k, k)))
    k = mscale(1.0 / theta, k)
    return madd(madd(eye(), mscale(math.sin(theta), k)),
                mscale(1.0 - math.cos(theta), matmul(k, k)))


def so3_left_jacobian(v):
    theta = norm(v)
    k = skew(v)
    k2 = matmul(k, k)
    if theta < 1.0e-8:
        return madd(madd(eye(), mscale(0.5, k)), mscale(1.0 / 6.0, k2))
    return madd(madd(eye(), mscale((1.0 - math.cos(theta)) / theta**2, k)),
                mscale((theta - math.sin(theta)) / theta**3, k2))


def box_geometry(point, center, rotation, proxy_offset, half_extent):
    proxy_center = add(center, matvec(rotation, proxy_offset))
    local = matvec(transpose(rotation), sub(point, proxy_center))
    signed = [abs(local[i]) - half_extent[i] for i in range(3)]
    outside = [max(x, 0.0) for x in signed]
    distance = norm(outside)
    if not distance:
        raise ValueError("fixture point must lie strictly outside the box")
    local_sign = [-1.0 if value < 0.0 else 1.0 for value in local]
    local_normal = [local_sign[i] * outside[i] / distance for i in range(3)]
    normal = matvec(rotation, local_normal)
    witness_local = [max(-half_extent[i], min(half_extent[i], local[i]))
                     for i in range(3)]
    witness = add(proxy_center, matvec(rotation, witness_local))
    active = [[1.0 if i == j and signed[i] > 0.0 else 0.0
               for j in range(3)] for i in range(3)]
    local_hessian = mscale(
        1.0 / distance,
        madd(active, mscale(-1.0,
             [[local_normal[i] * local_normal[j] for j in range(3)]
              for i in range(3)])))
    world_hessian = matmul(matmul(rotation, local_hessian), transpose(rotation))
    return distance, normal, witness, local, signed, local_normal, active, local_hessian, world_hessian


def barrier(distance, inverse_response, dt, thickness):
    d = max(distance, 1.0e-4 * thickness)
    d2, h2 = d * d, thickness * thickness
    offset = d2 - h2
    k = 10.0 / max(inverse_response * dt * dt * h2, 1.0e-20)
    logarithm = math.log(d2 / h2)
    first = -k * (2.0 * offset * logarithm + offset * offset / d2)
    second = -k * (2.0 * logarithm + 4.0 * offset / d2
                   - offset * offset / (d2 * d2))
    force = -2.0 * d * first
    hessian = max(4.0 * d2 * second + 2.0 * first, 0.0)
    return dt * force, dt * dt * hessian


def fixture(local_point):
    p = {
        "dt": 0.03,
        "thickness": 0.1,
        "x0": [0.86, 0.25, 0.18],
        "c0": [0.0, 0.0, 0.0],
        "rotation0": exp_so3([0.12, -0.18, 0.08]),
        "proxy_offset": [0.08, -0.05, 0.03],
        "omega": [0.7, -0.4, 0.5],
        "source_omega": [0.25, -0.15, 0.35],
        "candidate_delta_omega": [0.45, -0.25, 0.15],
        "v": [0.2, 0.1, -0.3],
        "node_v": [-0.3, 0.2, 0.1],
        "node_inverse_mass": 0.25,
        "body_inverse_mass": 0.4,
        "body_inertia": [[0.9, 0.08, 0.02],
                          [0.08, 1.1, -0.04],
                          [0.02, -0.04, 0.7]],
        "half_extent": [0.7, 0.8, 0.9],
        "local_point": local_point,
    }
    current_rotation = matmul(exp_so3(scale(p["dt"], p["omega"])),
                              p["rotation0"])
    current_center = add(p["c0"], scale(p["dt"], p["v"]))
    current_point = add(current_center, matvec(current_rotation,
        add(p["proxy_offset"], local_point)))
    p["x0"] = sub(current_point, scale(p["dt"], p["node_v"]))
    return p


def state_at(p, direction=None, epsilon=0.0):
    dt = p["dt"]
    node_v = list(p["node_v"])
    body_v = list(p["v"])
    omega = list(p["omega"])
    if direction is not None:
        node_v = add(node_v, scale(epsilon, direction[:3]))
        body_v = add(body_v, scale(epsilon, direction[3:6]))
        omega = add(omega, scale(epsilon, direction[6:9]))
    rotation = matmul(exp_so3(scale(dt, omega)), p["rotation0"])
    center = add(p["c0"], scale(dt, body_v))
    point = add(p["x0"], scale(dt, node_v))
    return point, center, rotation, node_v, body_v, omega


def nonlinear_contact_residual(p, direction=None, epsilon=0.0):
    point, center, rotation, _, _, _ = state_at(p, direction, epsilon)
    distance, normal, witness, *_ = box_geometry(
        point, center, rotation, p["proxy_offset"], p["half_extent"])
    lever = sub(witness, center)
    inverse_inertia = matmul(matmul(rotation, inverse3(p["body_inertia"])),
                             transpose(rotation))
    angular_arm = cross(lever, normal)
    response = (p["node_inverse_mass"] + p["body_inverse_mass"]
                + dot(angular_arm, matvec(inverse_inertia, angular_arm)))
    impulse, _ = barrier(distance, response, p["dt"], p["thickness"])
    normal_impulse = scale(impulse, normal)
    return (list(normal_impulse) + list(scale(-1.0, normal_impulse))
            + list(scale(-1.0, cross(lever, normal_impulse))))


def analytic_contact_action(p, direction):
    dt = p["dt"]
    point, center, rotation, _, _, omega = state_at(p)
    distance, normal, witness, local, signed, local_normal, active, local_hessian, world_hessian = box_geometry(
        point, center, rotation, p["proxy_offset"], p["half_extent"])
    delta_theta = scale(dt, matvec(so3_left_jacobian(scale(dt, omega)),
                                   direction[6:9]))
    delta_point = scale(dt, direction[:3])
    delta_center = scale(dt, direction[3:6])
    point_from_com = sub(point, center)
    local_displacement = matvec(transpose(rotation),
        add(sub(delta_point, delta_center), cross(point_from_com, delta_theta)))
    inside = madd(eye(), mscale(-1.0, active))
    delta_lever = add(cross(delta_theta, sub(witness, center)),
                      matvec(rotation, matvec(inside, local_displacement)))
    exact_normal_direction = add(cross(delta_theta, normal),
                      matvec(rotation, matvec(local_hessian, local_displacement)))
    relative_displacement = sub(sub(delta_point, delta_center), delta_lever)
    relative_velocity = scale(1.0 / dt, relative_displacement)
    spatial_normal_direction = matvec(world_hessian, relative_displacement)
    unrepresented_normal_direction = sub(exact_normal_direction,
                                         spatial_normal_direction)

    lever = sub(witness, center)
    inverse_inertia = matmul(matmul(rotation, inverse3(p["body_inertia"])),
                             transpose(rotation))
    angular_arm = cross(lever, normal)
    inverse_inertia_arm = matvec(inverse_inertia, angular_arm)
    inverse_response = (p["node_inverse_mass"] + p["body_inverse_mass"]
                        + dot(angular_arm, inverse_inertia_arm))
    arm_direction = add(cross(delta_lever, normal),
                         cross(lever, exact_normal_direction))
    inverse_response_direction = (
        2.0 * dot(arm_direction, inverse_inertia_arm)
        + 2.0 * dot(delta_theta, cross(inverse_inertia_arm, angular_arm)))
    impulse, spatial_hessian = barrier(
        distance, inverse_response, dt, p["thickness"])
    node_action = add(
        add(scale(spatial_hessian * dot(normal, relative_velocity), normal),
            scale(-dt * impulse, matvec(world_hessian, relative_velocity))),
        add(scale(-impulse, unrepresented_normal_direction),
            scale(impulse * inverse_response_direction / inverse_response,
                  normal)))
    body_action = scale(-1.0, node_action)
    body_torque_action = add(
        cross(lever, scale(-1.0, node_action)),
        cross(delta_lever, scale(impulse, normal)))
    return list(node_action) + list(body_action) + list(body_torque_action)


class FreeBoxAngularOperatorTest(unittest.TestCase):
    def assert_operator_matches_fd(self, local_point):
        p = fixture(local_point)
        # All points are separated from box-feature transitions and inside the
        # barrier band. This test does not exercise nonsmooth feature switches.
        point, center, rotation, _, _, _ = state_at(p)
        gap, *_ = box_geometry(point, center, rotation,
                               p["proxy_offset"], p["half_extent"])
        self.assertGreater(gap, 1.0e-3)
        self.assertLess(gap, p["thickness"])
        worst_abs = 0.0
        worst_rel = 0.0
        directions = []
        for axis in range(9):
            e = [0.0] * 9
            e[axis] = 1.0
            directions.append(e)
        for seed in range(64):
            random.seed(9000 + seed)
            v = [random.gauss(0.0, 1.0) for _ in range(9)]
            directions.append(scale(1.0 / norm(v), v))
        epsilon = 1.0e-5
        for direction in directions:
            plus = nonlinear_contact_residual(p, direction, epsilon)
            minus = nonlinear_contact_residual(p, direction, -epsilon)
            finite_difference = scale(-1.0 / (2.0 * epsilon), sub(plus, minus))
            action = analytic_contact_action(p, direction)
            error = norm(sub(action, finite_difference))
            reference = max(norm(finite_difference), 1.0e-12)
            worst_abs = max(worst_abs, error)
            worst_rel = max(worst_rel, error / reference)
            self.assertLess(error, 2.0e-5)
            self.assertLess(error / reference, 2.0e-7)
        return worst_abs, worst_rel

    def test_rotated_smooth_face_includes_normal_frame_and_offset_torque(self):
        worst_abs, worst_rel = self.assert_operator_matches_fd([0.74, 0.2, 0.15])
        self.assertLess(worst_abs, 2.0e-5)
        self.assertLess(worst_rel, 2.0e-7)

    def test_shader_compares_normal_differentials_in_displacement_units(self):
        shader = (Path(__file__).resolve().parents[1] /
                  "src" / "metal" / "fgmres.metalinc").read_text()
        self.assertIn(
            "const float3 spatialNormalDirection = timestep * rotation *\n"
            "        localDistanceHessian * transpose(rotation) * relativeDirection;",
            shader)
        # The explicit timestep is required because relativeDirection is a
        # velocity perturbation while dn and H*dx are displacement differentials.
        dt = 0.03
        hessian = [[2.0, 0.2, 0.0], [0.2, 1.0, -0.1], [0.0, -0.1, 0.5]]
        relative_velocity = [0.4, -0.3, 0.2]
        relative_displacement = scale(dt, relative_velocity)
        exact_linearized = matvec(hessian, relative_displacement)
        scaled_shader_term = scale(dt, matvec(hessian, relative_velocity))
        unscaled_shader_term = matvec(hessian, relative_velocity)
        self.assertLess(norm(sub(exact_linearized, scaled_shader_term)), 1.0e-16)
        self.assertGreater(norm(sub(exact_linearized, unscaled_shader_term)), 1.0e-3)

    def test_rotated_smooth_edge_includes_inverse_response_derivative(self):
        worst_abs, worst_rel = self.assert_operator_matches_fd([0.74, 0.83, 0.15])
        self.assertLess(worst_abs, 2.0e-5)
        self.assertLess(worst_rel, 2.0e-7)

    def test_rotated_smooth_corner_includes_all_three_active_axes(self):
        worst_abs, worst_rel = self.assert_operator_matches_fd([0.74, 0.83, 0.95])
        self.assertLess(worst_abs, 2.0e-5)
        self.assertLess(worst_rel, 2.0e-7)

    def test_negative_and_mixed_sign_face_edge_corner_directions(self):
        for local_point in ([-0.74, 0.2, 0.15],
                            [-0.74, 0.83, 0.15],
                            [-0.74, -0.83, 0.95]):
            with self.subTest(local_point=local_point):
                worst_abs, worst_rel = self.assert_operator_matches_fd(local_point)
                self.assertLess(worst_abs, 2.0e-5)
                self.assertLess(worst_rel, 2.0e-7)

    def test_rotated_world_inertia_action(self):
        p = fixture([0.74, 0.2, 0.15])
        dt = p["dt"]
        omega = p["omega"]
        rotation = matmul(exp_so3(scale(dt, omega)), p["rotation0"])
        inertia_world = matmul(matmul(rotation, p["body_inertia"]),
                               transpose(rotation))
        direction = [0.4, -0.2, 0.7]
        delta_theta = scale(dt, matvec(so3_left_jacobian(scale(dt, omega)),
                                       direction))
        candidate_delta = p["candidate_delta_omega"]
        omega_term = add(matvec(inertia_world, direction),
            sub(cross(delta_theta, matvec(inertia_world, candidate_delta)),
                matvec(inertia_world, cross(delta_theta, candidate_delta))))
        eps = 2.0e-6
        def angular_momentum(sign):
            delta = add(candidate_delta,
                        scale(sign * eps, direction))
            w = add(p["source_omega"], delta)
            r = matmul(exp_so3(scale(dt, w)), p["rotation0"])
            iw = matmul(matmul(r, p["body_inertia"]), transpose(r))
            return matvec(iw, delta)
        fd = scale(1.0 / (2.0 * eps),
                   sub(angular_momentum(1.0), angular_momentum(-1.0)))
        self.assertLess(norm(sub(omega_term, fd)), 2.0e-7)

    def test_small_angle_pose_map_matches_metalworld_scene_integrator(self):
        source = quaternion_normalize([0.17, -0.11, 0.08, 0.97])
        for angle in (0.5e-6, 1.0e-6, 2.0e-6):
            omega = [angle / 0.03, -0.0, 0.0]
            exact = [math.sin(0.5 * angle), 0.0, 0.0,
                     math.cos(0.5 * angle)]
            exact = quaternion_normalize(quaternion_multiply(
                exact, source))
            self.assertLess(norm(sub(
                scene_integrate_quaternion(source, omega, 0.03), exact)),
                1.0e-15)
        # The generic Matter branch remains an FP32-near-equivalent formula;
        # the BOX-specific path uses the exact same small-angle series as the
        # body integrator so endpoint checking is representation-consistent.
        omega = [0.5e-6 / 0.03, 0.0, 0.0]
        self.assertLess(norm(sub(
            scene_integrate_quaternion(source, omega, 0.03),
            matter_pre_slice_quaternion(source, omega, 0.03))), 2.0e-16)

    def test_scene_candidate_endpoint_matches_exponential_rotation(self):
        source = quaternion_normalize([0.21, -0.13, 0.09, 0.94])
        dt = 0.017
        accepted_omega = [0.41, -0.32, 0.27]
        candidate_increment = [0.18, 0.07, -0.11]
        candidate_omega = add(accepted_omega, candidate_increment)
        for omega in (accepted_omega, candidate_omega,
                      [2.7, -1.9, 1.2], [1.0e-5, -2.0e-5, 3.0e-5]):
            scene_q = scene_integrate_quaternion(source, omega, dt)
            expected_rotation = matmul(
                exp_so3(scale(dt, omega)), quaternion_rotation_matrix(source))
            observed_rotation = quaternion_rotation_matrix(scene_q)
            max_error = max(abs(observed_rotation[i][j] - expected_rotation[i][j])
                            for i in range(3) for j in range(3))
            self.assertLess(max_error, 2.0e-14)

    def test_frictional_box_is_rejected_even_when_iterate_omega_is_zero(self):
        self.assertTrue(free_box_angular_slice_admits(0.0))
        self.assertFalse(free_box_angular_slice_admits(1.0e-30))
        self.assertFalse(free_box_angular_slice_admits(0.25))

    def test_alpha_sweep_gap_is_bounded_with_rotating_offset_proxy(self):
        p = fixture([0.74, 0.2, 0.15])
        dt = p["dt"]
        base_v = p["v"]
        trial_v = add(base_v, [0.01, -0.015, 0.006])
        base_omega = p["omega"]
        trial_omega = add(base_omega, [0.3, -0.2, 0.15])
        first_point = add(p["x0"], scale(dt, p["node_v"]))
        last_point = add(first_point, [0.002, -0.001, 0.0015])
        first_center = add(p["c0"], scale(dt, base_v))
        last_center = add(p["c0"], scale(dt, trial_v))
        relative_first = sub(first_point, first_center)
        relative_last = sub(last_point, last_center)
        offset_radius = norm(p["proxy_offset"]) + norm(p["half_extent"])
        angular_radius = max(norm(relative_first), norm(relative_last)) + offset_radius
        lipschitz = (norm(sub(relative_last, relative_first)) +
                     dt * norm(sub(trial_omega, base_omega)) * angular_radius)
        self.assertGreater(lipschitz, 0.0)
        samples = [alpha_path_gap(p, i / 256.0, first_point, last_point,
                                  base_v, trial_v, base_omega, trial_omega)
                   for i in range(257)]
        for i, (first, last) in enumerate(zip(samples, samples[1:])):
            self.assertLessEqual(abs(last - first), lipschitz / 256.0 + 2.0e-10,
                                 f"gap escaped angular sweep bound at interval {i}")

    def test_physical_rotating_sweep_gap_bound_includes_proxy_offset(self):
        p = fixture([0.74, 0.2, 0.15])
        dt = p["dt"]
        first_point = p["x0"]
        last_point = add(first_point, scale(dt, p["node_v"]))
        body_velocity = p["v"]
        angular_velocity = [0.7, -0.4, 0.5]
        center_end = add(p["c0"], scale(dt, body_velocity))
        relative_first = sub(first_point, p["c0"])
        relative_last = sub(last_point, center_end)
        offset_radius = norm(p["proxy_offset"]) + norm(p["half_extent"])
        angular_radius = max(norm(relative_first), norm(relative_last)) + offset_radius
        lipschitz = (norm(sub(relative_last, relative_first)) +
                     dt * norm(angular_velocity) * angular_radius)
        samples = [physical_sweep_gap(p, i / 256.0, first_point, last_point,
                                      body_velocity, angular_velocity)
                   for i in range(257)]
        self.assertTrue(all(math.isfinite(value) and value > 0.0
                            for value in samples))
        for i, (first, last) in enumerate(zip(samples, samples[1:])):
            self.assertLessEqual(abs(last - first), lipschitz / 256.0 + 2.0e-10,
                                 f"physical sweep escaped angular bound at interval {i}")


if __name__ == "__main__":
    unittest.main()
