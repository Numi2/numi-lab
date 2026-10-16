#include "cardboard_mixed_cohesive_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using V = std::array<double, 3>;
using Nodes = std::array<V, 6>; // side A slots 0..2, side B slots 3..5
using Force = Nodes;
using cardboard::mixed_cohesive_reference::History;
using cardboard::mixed_cohesive_reference::Law;

[[noreturn]] void fail(const std::string_view text) {
    std::cerr << "mixed cohesive frame check failed: " << text << '\n';
    std::exit(1);
}

void require(bool value, const std::string_view text) {
    if (!value) fail(text);
}

V add(V a, V b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
V sub(V a, V b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
V mul(double s, V a) { return {s * a[0], s * a[1], s * a[2]}; }
double dot(V a, V b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
V cross(V a, V b) {
    return {a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]};
}
double norm(V a) { return std::sqrt(dot(a, a)); }
V normalized(V a) { return mul(1.0 / norm(a), a); }

struct D2 {
    double v = 0.0;
    double d = 0.0;
    double q = 0.0;
    double dq = 0.0;
};

D2 operator+(D2 a, D2 b) {
    return {a.v + b.v, a.d + b.d, a.q + b.q, a.dq + b.dq};
}
D2 operator-(D2 a, D2 b) {
    return {a.v - b.v, a.d - b.d, a.q - b.q, a.dq - b.dq};
}
D2 operator*(D2 a, D2 b) {
    return {a.v * b.v, a.d * b.v + a.v * b.d,
        a.q * b.v + a.v * b.q,
        a.dq * b.v + a.d * b.q + a.q * b.d + a.v * b.dq};
}
D2 inverse(D2 a) {
    const double i = 1.0 / a.v;
    const double i2 = i * i;
    const double i3 = i2 * i;
    return {i, -a.d * i2, -a.q * i2,
        2.0 * a.d * a.q * i3 - a.dq * i2};
}
D2 operator/(D2 a, D2 b) { return a * inverse(b); }
D2 sqrtD(D2 a) {
    const double v = std::sqrt(a.v);
    const double first = 0.5 / v;
    const double second = -0.25 / (v * v * v);
    return {v, first * a.d, first * a.q,
        first * a.dq + second * a.d * a.q};
}

struct V2 { D2 x, y, z; };
V2 operator-(V2 a, V2 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V2 operator*(V2 a, D2 s) { return {a.x * s, a.y * s, a.z * s}; }
D2 dot(V2 a, V2 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V2 cross(V2 a, V2 b) {
    return {a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}
V values(V2 a) { return {a.x.v, a.y.v, a.z.v}; }
V along(V2 a) { return {a.x.d, a.y.d, a.z.d}; }
V test(V2 a) { return {a.x.q, a.y.q, a.z.q}; }
V mixed(V2 a) { return {a.x.dq, a.y.dq, a.z.dq}; }

struct Frame2 { V2 t1, t2, n; };

Frame2 frame(const Nodes& x, const Nodes& direction,
             const unsigned testSlot = 99u, const unsigned testAxis = 99u) {
    const auto variable = [&](unsigned slot) {
        V q{};
        if (slot == testSlot && testAxis < 3u) q[testAxis] = 1.0;
        return V2{{x[slot][0], direction[slot][0], q[0], 0.0},
            {x[slot][1], direction[slot][1], q[1], 0.0},
            {x[slot][2], direction[slot][2], q[2], 0.0}};
    };
    const V2 a0 = variable(0u);
    const V2 edge0 = variable(1u) - a0;
    const V2 edge1 = variable(2u) - a0;
    const V2 t1 = edge0 * (D2{1.0} / sqrtD(dot(edge0, edge0)));
    const V2 area = cross(edge0, edge1);
    const V2 n = area * (D2{1.0} / sqrtD(dot(area, area)));
    return {t1, cross(n, t1), n};
}

struct Result { Force force{}; Force tangent{}; bool valid = false; };

constexpr std::array<std::array<double, 3>, 3> kShape{{
    {{2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0}},
    {{1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0}},
    {{1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0}},
}};

std::array<double, 3> local(V d, const Frame2& f) {
    return {dot(d, values(f.n)), dot(d, values(f.t1)),
        dot(d, values(f.t2))};
}

Result evaluate(const Law& law, const Nodes& accepted, const Nodes& trial,
                const Force& direction, const std::array<History, 3>& history,
                const double referenceArea) {
    Result out{};
    const Frame2 current = frame(trial, direction);
    const Frame2 previous = frame(accepted, Force{});
    V frameMoment{};
    V frameMomentDirection{};
    for (std::size_t q = 0; q < 3; ++q) {
        V d{};
        V dd{};
        V d0{};
        for (std::size_t slot = 0; slot < 3; ++slot) {
            d = add(d, mul(kShape[q][slot],
                sub(trial[slot + 3], trial[slot])));
            dd = add(dd, mul(kShape[q][slot],
                sub(direction[slot + 3], direction[slot])));
            d0 = add(d0, mul(kShape[q][slot],
                sub(accepted[slot + 3], accepted[slot])));
        }
        const auto gap = local(d, current);
        const auto oldGap = local(d0, previous);
        std::array<double, 3> gapD{
            dot(dd, values(current.n)) + dot(d, along(current.n)),
            dot(dd, values(current.t1)) + dot(d, along(current.t1)),
            dot(dd, values(current.t2)) + dot(d, along(current.t2))};
        cardboard::mixed_cohesive_reference::Response response;
        if (!cardboard::mixed_cohesive_reference::evaluate(
                law, oldGap, gap, history[q], response) || !response.valid)
            return out;
        std::array<double, 3> tractionD{};
        for (std::size_t row = 0; row < 3; ++row)
            for (std::size_t column = 0; column < 3; ++column)
                tractionD[row] += response.tangentPaPerM[row][column] * gapD[column];
        const V t = add(add(mul(response.tractionPa[0], values(current.n)),
                            mul(response.tractionPa[1], values(current.t1))),
                       mul(response.tractionPa[2], values(current.t2)));
        const V td = add(add(add(mul(tractionD[0], values(current.n)),
                                 mul(tractionD[1], values(current.t1))),
                            mul(tractionD[2], values(current.t2))),
                         add(add(mul(response.tractionPa[0], along(current.n)),
                                 mul(response.tractionPa[1], along(current.t1))),
                             mul(response.tractionPa[2], along(current.t2))));
        const double w = referenceArea / 3.0;
        for (std::size_t slot = 0; slot < 3; ++slot) {
            out.force[slot] = add(out.force[slot], mul(w * kShape[q][slot], t));
            out.force[slot + 3] = sub(out.force[slot + 3],
                mul(w * kShape[q][slot], t));
            out.tangent[slot] = add(out.tangent[slot],
                mul(w * kShape[q][slot], td));
            out.tangent[slot + 3] = sub(out.tangent[slot + 3],
                mul(w * kShape[q][slot], td));
        }
        frameMoment = add(frameMoment, mul(w, cross(d, t)));
        frameMomentDirection = add(frameMomentDirection,
            mul(w, add(cross(dd, t), cross(d, td))));
    }
    for (unsigned slot = 0; slot < 3; ++slot) {
        for (unsigned axis = 0; axis < 3; ++axis) {
            const Frame2 differentiated = frame(trial, direction, slot, axis);
            const V omega = mul(0.5,
                add(add(cross(values(differentiated.t1), test(differentiated.t1)),
                       cross(values(differentiated.t2), test(differentiated.t2))),
                    cross(values(differentiated.n), test(differentiated.n))));
            const V omegaD = mul(0.5, add(
                add(add(cross(along(differentiated.t1), test(differentiated.t1)),
                        cross(values(differentiated.t1), mixed(differentiated.t1))),
                    add(cross(along(differentiated.t2), test(differentiated.t2)),
                        cross(values(differentiated.t2), mixed(differentiated.t2)))),
                add(cross(along(differentiated.n), test(differentiated.n)),
                    cross(values(differentiated.n), mixed(differentiated.n)))));
            out.force[slot][axis] += dot(frameMoment, omega);
            out.tangent[slot][axis] += dot(frameMomentDirection, omega) +
                dot(frameMoment, omegaD);
        }
    }
    out.valid = true;
    return out;
}

V rotate(V x) {
    constexpr double c = 0.86602540378443864676;
    constexpr double s = 0.5;
    // Fixed proper rotation around normalized (1,2,3), via Rodrigues.
    const V axis = normalized({1.0, 2.0, 3.0});
    return add(add(mul(c, x), mul(s, cross(axis, x))),
        mul((1.0 - c) * dot(axis, x), axis));
}

Nodes rotate(const Nodes& nodes) {
    Nodes result{};
    for (std::size_t i = 0; i < nodes.size(); ++i)
        result[i] = rotate(nodes[i]);
    return result;
}

double potential(const Law& law, const Nodes& accepted, const Nodes& trial,
                 const std::array<History, 3>& history, const double area) {
    const Frame2 f = frame(trial, Force{});
    const Frame2 f0 = frame(accepted, Force{});
    double value = 0.0;
    for (std::size_t q = 0; q < 3; ++q) {
        V d{}, d0{};
        for (std::size_t slot = 0; slot < 3; ++slot) {
            d = add(d, mul(kShape[q][slot], sub(trial[slot + 3], trial[slot])));
            d0 = add(d0, mul(kShape[q][slot], sub(accepted[slot + 3], accepted[slot])));
        }
        const auto gap = local(d, f);
        const auto oldGap = local(d0, f0);
        cardboard::mixed_cohesive_reference::Response response;
        if (!cardboard::mixed_cohesive_reference::evaluate(
                law, oldGap, gap, history[q], response) || !response.valid)
            fail("potential fixture left the valid law domain");
        // All work tests stay strictly below initiation, so this is the exact
        // quadratic potential whose derivative is the authored traction.
        require(!response.initiated, "work test unexpectedly entered damage");
        const double normal = std::max(gap[0], 0.0);
        value += area / 6.0 * (law.normalPenaltyPaPerM * normal * normal +
            law.shearPenaltyPaPerM * (gap[1] * gap[1] + gap[2] * gap[2]));
    }
    return value;
}

double vectorNorm(V a) { return norm(a); }

} // namespace

int main() {
    // Synthetic asymmetric penalties deliberately expose frame coupling.
    const Law law{2.0e7, 3.5e6, 1.0e7, 1.0e8, 8.0e7, 10.0, 20.0, 1.5};
    const Nodes nodes{{
        V{0.0000, 0.0000, 0.0000}, V{0.0120, 0.0010, 0.0000},
        V{0.0020, 0.0100, 0.0010}, V{0.0005, 0.0004, 0.0008},
        V{0.0117, 0.0012, 0.0014}, V{0.0024, 0.0097, 0.0011}}};
    const Force direction{{
        V{0.11, -0.03, 0.07}, V{-0.02, 0.08, 0.04},
        V{0.05, 0.03, -0.09}, V{-0.04, 0.06, 0.01},
        V{0.03, -0.07, 0.08}, V{-0.08, 0.02, 0.05}}};
    const std::array<History, 3> history{};
    constexpr double referenceArea = 5.0e-5;
    const Result center = evaluate(law, nodes, nodes, direction, history,
                                   referenceArea);
    require(center.valid, "center response invalid");

    // Internal force and moment must balance even when K_normal != K_shear
    // and the face is rotated/skewed. The frame-conjugate term carries the
    // couple that direct equal-and-opposite tractions omit.
    V forceSum{};
    V momentSum{};
    for (std::size_t i = 0; i < 6; ++i) {
        forceSum = add(forceSum, center.force[i]);
        momentSum = add(momentSum, cross(nodes[i], center.force[i]));
    }
    require(vectorNorm(forceSum) <= 1.0e-9 && vectorNorm(momentSum) <= 1.0e-9,
        "asymmetric rotated face does not conserve resultant force and torque");

    // A common rigid rotation must preserve local separation and rotate every
    // nodal force covariantly.
    const Nodes rotated = rotate(nodes);
    const Result rotatedResult = evaluate(law, rotated, rotated, Force{},
        history, referenceArea);
    require(rotatedResult.valid, "rotated response invalid");
    double rotationError = 0.0;
    for (std::size_t i = 0; i < 6; ++i)
        rotationError = std::max(rotationError,
            vectorNorm(sub(rotatedResult.force[i], rotate(center.force[i]))));
    require(rotationError < 1.0e-8,
        "rigidly rotated interface did not rotate its force response");

    // Compare the complete frame/material operator action against centered
    // finite differences of the assembled force map.
    constexpr double epsilon = 1.0e-7;
    Nodes plus = nodes;
    Nodes minus = nodes;
    for (std::size_t i = 0; i < 6; ++i) {
        plus[i] = add(plus[i], mul(epsilon, direction[i]));
        minus[i] = sub(minus[i], mul(epsilon, direction[i]));
    }
    const Result plusForce = evaluate(law, nodes, plus, Force{}, history,
                                      referenceArea);
    const Result minusForce = evaluate(law, nodes, minus, Force{}, history,
                                       referenceArea);
    double tangentError = 0.0;
    double tangentScale = 0.0;
    for (std::size_t i = 0; i < 6; ++i) {
        const V fd = mul(0.5 / epsilon,
            sub(plusForce.force[i], minusForce.force[i]));
        tangentError = std::max(tangentError,
            vectorNorm(sub(fd, center.tangent[i])));
        tangentScale = std::max(tangentScale, vectorNorm(fd));
    }
    require(tangentError <= 3.0e-5 * std::max(1.0, tangentScale),
        "frame/material operator disagrees with centered force differences");

    // For an intact face, generalized force power is minus the derivative of
    // the local cohesive potential. This checks the work sign and frame term.
    double forcePower = 0.0;
    for (std::size_t i = 0; i < 6; ++i)
        forcePower += dot(center.force[i], direction[i]);
    const double energyDirection = (potential(law, nodes, plus, history,
        referenceArea) - potential(law, nodes, minus, history, referenceArea)) /
        (2.0 * epsilon);
    require(std::abs(forcePower + energyDirection) <=
        5.0e-5 * std::max(1.0, std::abs(forcePower)),
        "interface force virtual work does not match the energy derivative");

    std::cout << "PASS cohesive frame force/operator: asymmetric penalties, "
        << "resultant torque, rigid rotation covariance, centered tangent FD, "
        << "and intact virtual work; synthetic values only\n";
}
