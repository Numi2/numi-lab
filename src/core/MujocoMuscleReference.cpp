#include "metalrobo/MujocoMuscleReference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<double, 9>;

// MuJoCo's double-precision mjMINVAL.  The functions below are a direct FP64
// adaptation of MuJoCo 3.12's Apache-2.0 engine_util_misc.c routines
// (mju_wrap and mju_muscle{Gain,Bias,Dynamics}), with Core point-Jacobian
// queries substituted for MuJoCo's tendon Jacobian scatter.
constexpr double kMinimum = 1.0e-15;
constexpr double kPi = 3.141592653589793238462643383279502884;

bool finite(const double value) { return std::isfinite(value); }
bool finite(const Vec3& value) {
    return std::all_of(value.begin(), value.end(), [](const double item) { return finite(item); });
}
bool finite(const Mat3& value) {
    return std::all_of(value.begin(), value.end(), [](const double item) { return finite(item); });
}

Vec3 add(const Vec3& left, const Vec3& right) {
    return {left[0] + right[0], left[1] + right[1], left[2] + right[2]};
}
Vec3 subtract(const Vec3& left, const Vec3& right) {
    return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}
Vec3 scale(const Vec3& value, const double scalar) {
    return {value[0] * scalar, value[1] * scalar, value[2] * scalar};
}
double dot(const Vec3& left, const Vec3& right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}
Vec3 cross(const Vec3& left, const Vec3& right) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}
double norm(const Vec3& value) { return std::sqrt(dot(value, value)); }
Vec3 normalized(const Vec3& value) {
    return scale(value, 1.0 / norm(value));
}
Mat3 transpose(const Mat3& matrix) {
    return {
        matrix[0], matrix[3], matrix[6], matrix[1], matrix[4], matrix[7], matrix[2], matrix[5], matrix[8],
    };
}
Mat3 multiply(const Mat3& left, const Mat3& right) {
    Mat3 result{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            for (std::size_t cursor = 0; cursor < 3; ++cursor) {
                result[row * 3 + column] += left[row * 3 + cursor] * right[cursor * 3 + column];
            }
        }
    }
    return result;
}
Vec3 matApply(const Mat3& matrix, const Vec3& value) {
    return {
        matrix[0] * value[0] + matrix[1] * value[1] + matrix[2] * value[2],
        matrix[3] * value[0] + matrix[4] * value[1] + matrix[5] * value[2],
        matrix[6] * value[0] + matrix[7] * value[1] + matrix[8] * value[2],
    };
}
Mat3 quaternionMatrix(const std::array<double, 4>& quaternion) {
    const double x = quaternion[0];
    const double y = quaternion[1];
    const double z = quaternion[2];
    const double w = quaternion[3];
    return {
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
        2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
        2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y),
    };
}

MujocoMuscleReferenceDiagnostics failure(
    const MujocoMuscleReferenceStatus status,
    const std::uint32_t index = MR_INVALID_INDEX
) {
    return {.status = status, .failingIndex = index};
}

bool intersect2(const std::array<double, 2>& p1, const std::array<double, 2>& p2,
                const std::array<double, 2>& p3, const std::array<double, 2>& p4) {
    const double determinant = (p4[1] - p3[1]) * (p2[0] - p1[0]) -
        (p4[0] - p3[0]) * (p2[1] - p1[1]);
    if (std::abs(determinant) < kMinimum) return false;
    const double a = ((p4[0] - p3[0]) * (p1[1] - p3[1]) -
                      (p4[1] - p3[1]) * (p1[0] - p3[0])) / determinant;
    const double b = ((p2[0] - p1[0]) * (p1[1] - p3[1]) -
                      (p2[1] - p1[1]) * (p1[0] - p3[0])) / determinant;
    return a >= 0.0 && a <= 1.0 && b >= 0.0 && b <= 1.0;
}

double norm2(const std::array<double, 2>& value) {
    return std::sqrt(value[0] * value[0] + value[1] * value[1]);
}
std::array<double, 2> normal2(const std::array<double, 2>& value) {
    const double length = norm2(value);
    return {value[0] / length, value[1] / length};
}
double dot2(const std::array<double, 2>& left, const std::array<double, 2>& right) {
    return left[0] * right[0] + left[1] * right[1];
}

double circleLength(const std::array<double, 2>& p0, const std::array<double, 2>& p1,
                    const int index, const double radius) {
    const std::array<double, 2> unit0 = normal2(p0);
    const std::array<double, 2> unit1 = normal2(p1);
    double angle = std::acos(std::clamp(dot2(unit0, unit1), -1.0, 1.0));
    const double determinant = p0[1] * p1[0] - p0[0] * p1[1];
    if ((determinant > 0.0 && index) || (determinant < 0.0 && !index)) {
        angle = 2.0 * kPi - angle;
    }
    return radius * angle;
}

std::optional<double> wrapCircle(
    std::array<double, 4>& points, const std::array<double, 4>& endpoints,
    const std::optional<std::array<double, 2>>& side, const double radius
) {
    const double squared0 = endpoints[0] * endpoints[0] + endpoints[1] * endpoints[1];
    const double squared1 = endpoints[2] * endpoints[2] + endpoints[3] * endpoints[3];
    const double squaredRadius = radius * radius;
    if (squared0 < squaredRadius || squared1 < squaredRadius || radius < kMinimum) return std::nullopt;
    const std::array<double, 2> difference{endpoints[2] - endpoints[0], endpoints[3] - endpoints[1]};
    const double differenceSquared = dot2(difference, difference);
    if (differenceSquared < kMinimum) return std::nullopt;
    double lineFraction = -(difference[0] * endpoints[0] + difference[1] * endpoints[1]) / differenceSquared;
    lineFraction = std::clamp(lineFraction, 0.0, 1.0);
    const std::array<double, 2> nearest{
        lineFraction * difference[0] + endpoints[0], lineFraction * difference[1] + endpoints[1],
    };
    if (dot2(nearest, nearest) > squaredRadius && (!side || dot2(*side, nearest) >= 0.0)) return std::nullopt;
    const double root0 = std::sqrt(squared0 - squaredRadius);
    const double root1 = std::sqrt(squared1 - squaredRadius);
    std::array<std::array<std::array<double, 2>, 2>, 2> solutions{};
    std::array<double, 2> goodness{};
    const std::array<double, 2> endpoint0{endpoints[0], endpoints[1]};
    const std::array<double, 2> endpoint1{endpoints[2], endpoints[3]};
    for (int index = 0; index < 2; ++index) {
        const double sign = index == 0 ? 1.0 : -1.0;
        solutions[index][0] = {
            (endpoints[0] * squaredRadius + sign * radius * endpoints[1] * root0) / squared0,
            (endpoints[1] * squaredRadius - sign * radius * endpoints[0] * root0) / squared0,
        };
        solutions[index][1] = {
            (endpoints[2] * squaredRadius - sign * radius * endpoints[3] * root1) / squared1,
            (endpoints[3] * squaredRadius + sign * radius * endpoints[2] * root1) / squared1,
        };
        if (side) {
            goodness[index] = dot2(normal2({
                solutions[index][0][0] + solutions[index][1][0],
                solutions[index][0][1] + solutions[index][1][1],
            }), *side);
        } else {
            const std::array<double, 2> differencePoint{
                solutions[index][0][0] - solutions[index][1][0],
                solutions[index][0][1] - solutions[index][1][1],
            };
            goodness[index] = -dot2(differencePoint, differencePoint);
        }
        if (intersect2(endpoint0, solutions[index][0], endpoint1, solutions[index][1])) {
            goodness[index] = -10000.0;
        }
    }
    const int selected = goodness[0] > goodness[1] ? 0 : 1;
    points = {
        solutions[selected][0][0], solutions[selected][0][1],
        solutions[selected][1][0], solutions[selected][1][1],
    };
    if (intersect2(endpoint0, {points[0], points[1]}, endpoint1, {points[2], points[3]})) return std::nullopt;
    return circleLength({points[0], points[1]}, {points[2], points[3]}, selected, radius);
}

std::optional<double> wrapInside(std::array<double, 4>& points,
                                 const std::array<double, 4>& endpoints,
                                 const double radius) {
    constexpr int maxIterations = 20;
    constexpr double initial = 1.0 - 1.0e-7;
    constexpr double tolerance = 1.0e-6;
    const std::array<double, 2> endpoint0{endpoints[0], endpoints[1]};
    const std::array<double, 2> endpoint1{endpoints[2], endpoints[3]};
    const double length0 = norm2(endpoint0);
    const double length1 = norm2(endpoint1);
    const std::array<double, 2> difference{endpoints[2] - endpoints[0], endpoints[3] - endpoints[1]};
    const double differenceSquared = dot2(difference, difference);
    if (length0 <= radius || length1 <= radius || radius < kMinimum || length0 < kMinimum || length1 < kMinimum) {
        return std::nullopt;
    }
    if (differenceSquared > kMinimum) {
        const double fraction = -(endpoints[0] * difference[0] + endpoints[1] * difference[1]) / differenceSquared;
        if (fraction > 0.0 && fraction < 1.0) {
            const std::array<double, 2> nearest{
                endpoints[0] + fraction * difference[0], endpoints[1] + fraction * difference[1],
            };
            if (norm2(nearest) <= radius) return std::nullopt;
        }
    }
    std::array<double, 2> midpoint{0.5 * (endpoints[0] + endpoints[2]), 0.5 * (endpoints[1] + endpoints[3])};
    midpoint = normal2(midpoint);
    midpoint[0] *= radius;
    midpoint[1] *= radius;
    points[0] = midpoint[0]; points[1] = midpoint[1]; points[2] = midpoint[0]; points[3] = midpoint[1];
    const double a = radius / length0;
    const double b = radius / length1;
    const double cosine = (length0 * length0 + length1 * length1 - differenceSquared) / (2.0 * length0 * length1);
    if (cosine < -1.0 + kMinimum) return std::nullopt;
    if (cosine > 1.0 - kMinimum) return 0.0;
    const double angle = std::acos(std::clamp(cosine, -1.0, 1.0));
    double z = initial;
    double residual = std::asin(a * z) + std::asin(b * z) - 2.0 * std::asin(z) + angle;
    if (residual > 0.0) return 0.0;
    int iteration = 0;
    for (; iteration < maxIterations && std::abs(residual) > tolerance; ++iteration) {
        const double derivative = a / std::max(kMinimum, std::sqrt(1.0 - z * z * a * a)) +
            b / std::max(kMinimum, std::sqrt(1.0 - z * z * b * b)) -
            2.0 / std::max(kMinimum, std::sqrt(1.0 - z * z));
        if (derivative > -kMinimum) return 0.0;
        const double next = z - residual / derivative;
        if (next > z) return 0.0;
        z = next;
        residual = std::asin(a * z) + std::asin(b * z) - 2.0 * std::asin(z) + angle;
        if (residual > tolerance) return 0.0;
    }
    if (iteration >= maxIterations) return 0.0;
    std::array<double, 2> vector{};
    double rotation = 0.0;
    if (endpoints[0] * endpoints[3] - endpoints[1] * endpoints[2] > 0.0) {
        vector = endpoint0;
        rotation = std::asin(z) - std::asin(a * z);
    } else {
        vector = endpoint1;
        rotation = std::asin(z) - std::asin(b * z);
    }
    vector = normal2(vector);
    points[0] = radius * (std::cos(rotation) * vector[0] - std::sin(rotation) * vector[1]);
    points[1] = radius * (std::sin(rotation) * vector[0] + std::cos(rotation) * vector[1]);
    points[2] = points[0]; points[3] = points[1];
    return 0.0;
}

struct WrapResult { Vec3 first{}; Vec3 second{}; double length = -1.0; };

std::optional<WrapResult> wrapGeometry(
    const Vec3& endpoint0, const Vec3& endpoint1, const Vec3& center, const Mat3& rotation,
    const double radius, const MujocoRouteNodeType type, const std::optional<Vec3>& side
) {
    if (type != MujocoRouteNodeType::sphere && type != MujocoRouteNodeType::cylinder) return std::nullopt;
    const Mat3 inverse = transpose(rotation);
    const Vec3 point0 = matApply(inverse, subtract(endpoint0, center));
    const Vec3 point1 = matApply(inverse, subtract(endpoint1, center));
    if (norm(point0) < kMinimum || norm(point1) < kMinimum) return std::nullopt;
    std::array<Vec3, 2> axes{};
    if (type == MujocoRouteNodeType::sphere) {
        axes[0] = normalized(point0);
        Vec3 normal = cross(point0, point1);
        if (norm(normal) < kMinimum) {
            std::size_t selected = 0;
            if (std::abs(axes[0][1]) > std::abs(axes[0][0]) && std::abs(axes[0][1]) > std::abs(axes[0][2])) selected = 1;
            if (std::abs(axes[0][2]) > std::abs(axes[0][0]) && std::abs(axes[0][2]) > std::abs(axes[0][1])) selected = 2;
            Vec3 second{1.0, 1.0, 1.0}; second[selected] = 0.0;
            normal = normalized(cross(axes[0], second));
        } else {
            normal = normalized(normal);
        }
        axes[1] = normalized(cross(normal, axes[0]));
    } else {
        axes[0] = {1.0, 0.0, 0.0};
        axes[1] = {0.0, 1.0, 0.0};
    }
    const std::array<double, 4> endpoints{
        dot(point0, axes[0]), dot(point0, axes[1]), dot(point1, axes[0]), dot(point1, axes[1]),
    };
    std::optional<std::array<double, 2>> projectedSide;
    Vec3 localSide{};
    if (side) {
        localSide = matApply(inverse, subtract(*side, center));
        std::array<double, 2> projected{dot(localSide, axes[0]), dot(localSide, axes[1])};
        const double projectedNorm = norm2(projected);
        if (projectedNorm > kMinimum) {
            projected[0] *= radius / projectedNorm;
            projected[1] *= radius / projectedNorm;
            projectedSide = projected;
        }
    }
    std::array<double, 4> contact{};
    std::optional<double> length;
    if (side && norm(localSide) < radius) {
        length = wrapInside(contact, endpoints, radius);
    } else {
        length = wrapCircle(contact, endpoints, projectedSide, radius);
    }
    if (!length || *length < 0.0) return std::nullopt;
    Vec3 first = add(scale(axes[0], contact[0]), scale(axes[1], contact[1]));
    Vec3 second = add(scale(axes[0], contact[2]), scale(axes[1], contact[3]));
    double correctedLength = *length;
    if (type == MujocoRouteNodeType::cylinder) {
        const double firstLeg = std::sqrt((point0[0] - first[0]) * (point0[0] - first[0]) +
                                          (point0[1] - first[1]) * (point0[1] - first[1]));
        const double secondLeg = std::sqrt((point1[0] - second[0]) * (point1[0] - second[0]) +
                                           (point1[1] - second[1]) * (point1[1] - second[1]));
        const double denominator = firstLeg + correctedLength + secondLeg;
        if (!(denominator > kMinimum)) return std::nullopt;
        first[2] = point0[2] + (point1[2] - point0[2]) * firstLeg / denominator;
        second[2] = point0[2] + (point1[2] - point0[2]) * (firstLeg + correctedLength) / denominator;
        correctedLength = std::sqrt(correctedLength * correctedLength + (second[2] - first[2]) * (second[2] - first[2]));
    }
    return WrapResult{add(matApply(rotation, first), center), add(matApply(rotation, second), center), correctedLength};
}

struct ResolvedPoint {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    Vec3 local{};
    Vec3 world{};
};
struct Segment { ResolvedPoint first{}; ResolvedPoint second{}; };

bool validDefinition(const MujocoMuscleDefinition& definition,
                     const std::span<const MujocoMuscleSite> sites,
                     const std::span<const MujocoWrapGeometry> wraps) {
    if (definition.route.size() < 2u || !finite(definition.lengthRange[0]) || !finite(definition.lengthRange[1]) ||
        definition.lengthRange[1] <= definition.lengthRange[0] || !finite(definition.accelerationScale) ||
        !(definition.accelerationScale > 0.0) || !finite(definition.controlRange[0]) || !finite(definition.controlRange[1]) ||
        !std::all_of(definition.gainParameters.begin(), definition.gainParameters.end(), [](const double value) { return finite(value); }) ||
        !std::all_of(definition.biasParameters.begin(), definition.biasParameters.end(), [](const double value) { return finite(value); }) ||
        !std::all_of(definition.dynamicParameters.begin(), definition.dynamicParameters.end(), [](const double value) { return finite(value); })) return false;
    for (const MujocoRouteNode& node : definition.route) {
        if (node.type == MujocoRouteNodeType::site) {
            if (node.targetIndex >= sites.size()) return false;
        } else if (node.type == MujocoRouteNodeType::sphere || node.type == MujocoRouteNodeType::cylinder) {
            if (node.targetIndex >= wraps.size() || wraps[node.targetIndex].type != node.type ||
                !(wraps[node.targetIndex].radius > 0.0) || !finite(wraps[node.targetIndex].radius) ||
                !finite(wraps[node.targetIndex].localCenter) || !finite(wraps[node.targetIndex].rotationBody)) return false;
        } else return false;
        if (node.sideSiteIndex != MR_INVALID_INDEX && node.sideSiteIndex >= sites.size()) return false;
    }
    return definition.route.front().type == MujocoRouteNodeType::site &&
        definition.route.back().type == MujocoRouteNodeType::site;
}

MujocoMuscleReferenceDiagnostics resolvePath(
    const EngineModel& model, const std::uint32_t articulationIndex, const std::span<const double> q,
    const std::span<const double> v, const std::span<const MujocoMuscleSite> sites,
    const std::span<const MujocoWrapGeometry> wraps, const MujocoMuscleDefinition& definition,
    MujocoMusclePathResult& result, const ArticulatedDynamicsConfig& config
) {
    if (!validDefinition(definition, sites, wraps)) return failure(MujocoMuscleReferenceStatus::invalidDefinition);
    const MRArticulationGPU& articulation = model.articulations.at(articulationIndex);
    if (q.size() != articulation.nq || v.size() != articulation.nv) return failure(MujocoMuscleReferenceStatus::invalidState);
    std::vector<ArticulatedBodyKinematics> bodies(articulation.bodyCount);
    const ArticulatedDynamicsDiagnostics bodyDiagnostics = computeArticulatedBodyKinematics(
        model, articulationIndex, q, v, bodies, config
    );
    if (!bodyDiagnostics.succeeded()) return failure(MujocoMuscleReferenceStatus::kinematicsFailure);
    auto body = [&bodies, &articulation](const std::uint32_t index) -> const ArticulatedBodyKinematics* {
        if (index < articulation.firstBody || index >= articulation.firstBody + bodies.size()) return nullptr;
        return &bodies[index - articulation.firstBody];
    };
    auto site = [&sites, &body](const std::uint32_t index) -> std::optional<ResolvedPoint> {
        if (index >= sites.size()) return std::nullopt;
        const MujocoMuscleSite& source = sites[index];
        const ArticulatedBodyKinematics* pose = body(source.bodyIndex);
        if (!pose) return std::nullopt;
        const Mat3 rotation = quaternionMatrix(pose->orientation);
        if (!finite(rotation) || !finite(source.localPoint)) return std::nullopt;
        return ResolvedPoint{source.bodyIndex, source.localPoint, add(pose->centerOfMassPosition, matApply(rotation, source.localPoint))};
    };
    auto wrappedPoint = [&body](const std::uint32_t bodyIndex, const Vec3& world) -> std::optional<ResolvedPoint> {
        const ArticulatedBodyKinematics* pose = body(bodyIndex);
        if (!pose) return std::nullopt;
        const Mat3 rotation = quaternionMatrix(pose->orientation);
        if (!finite(rotation) || !finite(world)) return std::nullopt;
        return ResolvedPoint{bodyIndex, matApply(transpose(rotation), subtract(world, pose->centerOfMassPosition)), world};
    };
    std::vector<Segment> segments;
    double length = 0.0;
    std::uint32_t appliedWraps = 0u;
    std::size_t cursor = 0u;
    while (cursor + 1u < definition.route.size()) {
        const MujocoRouteNode& startNode = definition.route[cursor];
        if (startNode.type != MujocoRouteNodeType::site) return failure(MujocoMuscleReferenceStatus::invalidPath, static_cast<std::uint32_t>(cursor));
        const std::optional<ResolvedPoint> first = site(startNode.targetIndex);
        if (!first) return failure(MujocoMuscleReferenceStatus::invalidPath, static_cast<std::uint32_t>(cursor));
        const MujocoRouteNode& nextNode = definition.route[cursor + 1u];
        if (nextNode.type == MujocoRouteNodeType::site) {
            const std::optional<ResolvedPoint> second = site(nextNode.targetIndex);
            if (!second) return failure(MujocoMuscleReferenceStatus::invalidPath, static_cast<std::uint32_t>(cursor + 1u));
            const double distance = norm(subtract(second->world, first->world));
            // MuJoCo permits coincident authored sites.  Its tangent
            // normalization contributes a zero Jacobian column there, so
            // retain the zero length but omit a singular native segment.
            if (distance > kMinimum) segments.push_back({*first, *second});
            length += distance; cursor += 1u; continue;
        }
        if ((nextNode.type != MujocoRouteNodeType::sphere && nextNode.type != MujocoRouteNodeType::cylinder) ||
            cursor + 2u >= definition.route.size() || definition.route[cursor + 2u].type != MujocoRouteNodeType::site) {
            return failure(MujocoMuscleReferenceStatus::invalidPath, static_cast<std::uint32_t>(cursor + 1u));
        }
        const MujocoWrapGeometry& wrap = wraps[nextNode.targetIndex];
        const std::optional<ResolvedPoint> last = site(definition.route[cursor + 2u].targetIndex);
        const ArticulatedBodyKinematics* wrapBody = body(wrap.bodyIndex);
        if (!last || !wrapBody) return failure(MujocoMuscleReferenceStatus::invalidPath, static_cast<std::uint32_t>(cursor + 2u));
        const Mat3 bodyRotation = quaternionMatrix(wrapBody->orientation);
        const Vec3 center = add(wrapBody->centerOfMassPosition, matApply(bodyRotation, wrap.localCenter));
        const Mat3 rotation = multiply(bodyRotation, wrap.rotationBody);
        std::optional<Vec3> side;
        if (nextNode.sideSiteIndex != MR_INVALID_INDEX) {
            const std::optional<ResolvedPoint> sidePoint = site(nextNode.sideSiteIndex);
            if (!sidePoint) return failure(MujocoMuscleReferenceStatus::invalidPath, static_cast<std::uint32_t>(cursor + 1u));
            side = sidePoint->world;
        }
        const std::optional<WrapResult> wrapped = wrapGeometry(
            first->world, last->world, center, rotation, wrap.radius, nextNode.type, side
        );
        if (!wrapped) {
            const double distance = norm(subtract(last->world, first->world));
            if (distance > kMinimum) segments.push_back({*first, *last});
            length += distance;
        } else {
            const std::optional<ResolvedPoint> tangent0 = wrappedPoint(wrap.bodyIndex, wrapped->first);
            const std::optional<ResolvedPoint> tangent1 = wrappedPoint(wrap.bodyIndex, wrapped->second);
            if (!tangent0 || !tangent1 || !(wrapped->length >= 0.0)) {
                return failure(MujocoMuscleReferenceStatus::invalidPath, static_cast<std::uint32_t>(cursor + 1u));
            }
            for (const Segment& segment : std::array<Segment, 3>{{{*first, *tangent0}, {*tangent0, *tangent1}, {*tangent1, *last}}}) {
                if (norm(subtract(segment.second.world, segment.first.world)) > kMinimum) segments.push_back(segment);
            }
            length += norm(subtract(tangent0->world, first->world)) + wrapped->length + norm(subtract(last->world, tangent1->world));
            ++appliedWraps;
        }
        cursor += 2u;
    }
    if (cursor != definition.route.size() - 1u || !(length > kMinimum) || !finite(length)) {
        return failure(MujocoMuscleReferenceStatus::invalidPath);
    }
    std::vector<ArticulatedPointQuery> queries;
    queries.reserve(segments.size() * 2u);
    for (const Segment& segment : segments) {
        queries.push_back({segment.first.bodyIndex, segment.first.local});
        queries.push_back({segment.second.bodyIndex, segment.second.local});
    }
    std::vector<ArticulatedPointKinematics> kinematics(queries.size());
    std::vector<double> jacobians(queries.size() * 3u * articulation.nv);
    const ArticulatedDynamicsDiagnostics pointDiagnostics = computeArticulatedPointJacobians(
        model, articulationIndex, q, v, queries, kinematics, jacobians, config
    );
    if (!pointDiagnostics.succeeded()) return failure(MujocoMuscleReferenceStatus::kinematicsFailure);
    result = {};
    result.length = length;
    result.appliedWrapCount = appliedWraps;
    result.lengthJacobian.assign(articulation.nv, 0.0);
    for (std::size_t segmentIndex = 0u; segmentIndex < segments.size(); ++segmentIndex) {
        const Segment& segment = segments[segmentIndex];
        const Vec3 direction = normalized(subtract(segment.second.world, segment.first.world));
        for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
            double gradient = 0.0;
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                gradient += direction[axis] * (
                    jacobians[((segmentIndex * 2u + 1u) * 3u + axis) * articulation.nv + dof] -
                    jacobians[((segmentIndex * 2u + 0u) * 3u + axis) * articulation.nv + dof]
                );
            }
            result.lengthJacobian[dof] += gradient;
        }
    }
    result.velocity = 0.0;
    for (std::size_t dof = 0u; dof < articulation.nv; ++dof) result.velocity += result.lengthJacobian[dof] * v[dof];
    if (!finite(result.length) || !finite(result.velocity) || !std::all_of(result.lengthJacobian.begin(), result.lengthJacobian.end(), [](const double value) { return finite(value); })) {
        return failure(MujocoMuscleReferenceStatus::nonfiniteResult);
    }
    return {};
}

double muscleGainLength(const double length, const double lower, const double upper) {
    if (lower <= length && length <= upper) {
        const double lowerMid = 0.5 * (lower + 1.0);
        const double upperMid = 0.5 * (1.0 + upper);
        if (length <= lowerMid) { const double x = (length - lower) / std::max(kMinimum, lowerMid - lower); return 0.5 * x * x; }
        if (length <= 1.0) { const double x = (1.0 - length) / std::max(kMinimum, 1.0 - lowerMid); return 1.0 - 0.5 * x * x; }
        if (length <= upperMid) { const double x = (length - 1.0) / std::max(kMinimum, upperMid - 1.0); return 1.0 - 0.5 * x * x; }
        const double x = (upper - length) / std::max(kMinimum, upper - upperMid); return 0.5 * x * x;
    }
    return 0.0;
}

double muscleGain(const double length, const double velocity, const MujocoMuscleDefinition& definition) {
    const auto& parameters = definition.gainParameters;
    double force = parameters[2];
    if (force < 0.0) force = parameters[3] / std::max(kMinimum, definition.accelerationScale);
    const double optimumLength = (definition.lengthRange[1] - definition.lengthRange[0]) /
        std::max(kMinimum, parameters[1] - parameters[0]);
    const double normalizedLength = parameters[0] + (length - definition.lengthRange[0]) / std::max(kMinimum, optimumLength);
    const double normalizedVelocity = velocity / std::max(kMinimum, optimumLength * parameters[6]);
    const double lengthGain = muscleGainLength(normalizedLength, parameters[4], parameters[5]);
    const double y = parameters[8] - 1.0;
    double velocityGain = 0.0;
    if (normalizedVelocity <= -1.0) velocityGain = 0.0;
    else if (normalizedVelocity <= 0.0) velocityGain = (normalizedVelocity + 1.0) * (normalizedVelocity + 1.0);
    else if (normalizedVelocity <= y) velocityGain = parameters[8] - (y - normalizedVelocity) * (y - normalizedVelocity) / std::max(kMinimum, y);
    else velocityGain = parameters[8];
    return -force * lengthGain * velocityGain;
}

double muscleBias(const double length, const MujocoMuscleDefinition& definition) {
    const auto& parameters = definition.biasParameters;
    double force = parameters[2];
    if (force < 0.0) force = parameters[3] / std::max(kMinimum, definition.accelerationScale);
    const double optimumLength = (definition.lengthRange[1] - definition.lengthRange[0]) /
        std::max(kMinimum, parameters[1] - parameters[0]);
    const double normalizedLength = parameters[0] + (length - definition.lengthRange[0]) / std::max(kMinimum, optimumLength);
    const double upperMid = 0.5 * (1.0 + parameters[5]);
    if (normalizedLength <= 1.0) return 0.0;
    if (normalizedLength <= upperMid) {
        const double x = (normalizedLength - 1.0) / std::max(kMinimum, upperMid - 1.0);
        return -force * parameters[7] * 0.5 * x * x;
    }
    const double x = (normalizedLength - upperMid) / std::max(kMinimum, upperMid - 1.0);
    return -force * parameters[7] * (0.5 + x);
}

double sigmoid(const double value) { return 1.0 / (1.0 + std::exp(-value)); }
double muscleDynamics(const double excitation, const double activation, const std::array<double, 10>& parameters) {
    const double control = std::clamp(excitation, 0.0, 1.0);
    const double act = std::clamp(activation, 0.0, 1.0);
    const double activationTime = parameters[0] * (0.5 + 1.5 * act);
    const double deactivationTime = parameters[1] / (0.5 + 1.5 * act);
    const double excess = control - activation;
    const double tau = parameters[2] < kMinimum
        ? (excess > 0.0 ? activationTime : deactivationTime)
        : deactivationTime + (activationTime - deactivationTime) * sigmoid(excess / parameters[2] + 0.5);
    return excess / std::max(kMinimum, tau);
}

} // namespace

MujocoMuscleReferenceDiagnostics evaluateMujocoMuscle(
    const EngineModel& model, const std::uint32_t articulationIndex, const std::span<const double> q,
    const std::span<const double> v, const std::span<const MujocoMuscleSite> sites,
    const std::span<const MujocoWrapGeometry> wraps, const MujocoMuscleDefinition& definition,
    const MujocoMuscleState& state, MujocoMuscleResult& result, const ArticulatedDynamicsConfig& config
) {
    if (!finite(state.excitation) || !finite(state.activation)) return failure(MujocoMuscleReferenceStatus::invalidState);
    MujocoMusclePathResult path;
    const MujocoMuscleReferenceDiagnostics diagnostics = resolvePath(
        model, articulationIndex, q, v, sites, wraps, definition, path, config
    );
    if (!diagnostics.succeeded()) return diagnostics;
    const double derivative = muscleDynamics(state.excitation, state.activation, definition.dynamicParameters);
    const double force = muscleGain(path.length, path.velocity, definition) * state.activation + muscleBias(path.length, definition);
    if (!finite(derivative) || !finite(force)) return failure(MujocoMuscleReferenceStatus::nonfiniteResult);
    result = {.path = std::move(path), .activationDerivative = derivative, .actuatorForce = force};
    return {};
}

MujocoMuscleReferenceDiagnostics projectMujocoMuscleForce(
    const EngineModel& model, const std::uint32_t articulationIndex, const std::span<const double> q,
    const std::span<const double> v, const std::span<const MujocoMuscleSite> sites,
    const std::span<const MujocoWrapGeometry> wraps, const MujocoMuscleDefinition& definition,
    const MujocoMuscleState& state, const std::span<double> generalizedForce,
    MujocoMuscleResult* result, const ArticulatedDynamicsConfig& config
) {
    if (articulationIndex >= model.articulations.size() || generalizedForce.size() != model.articulations[articulationIndex].nv) {
        return failure(MujocoMuscleReferenceStatus::invalidState);
    }
    MujocoMuscleResult local;
    const MujocoMuscleReferenceDiagnostics diagnostics = evaluateMujocoMuscle(
        model, articulationIndex, q, v, sites, wraps, definition, state, local, config
    );
    if (!diagnostics.succeeded()) return diagnostics;
    for (std::size_t dof = 0u; dof < generalizedForce.size(); ++dof) {
        generalizedForce[dof] += local.actuatorForce * local.path.lengthJacobian[dof];
    }
    if (!std::all_of(generalizedForce.begin(), generalizedForce.end(), [](const double value) { return finite(value); })) return failure(MujocoMuscleReferenceStatus::nonfiniteResult);
    if (result) *result = std::move(local);
    return {};
}

const char* mujocoMuscleReferenceStatusName(const MujocoMuscleReferenceStatus status) noexcept {
    switch (status) {
    case MujocoMuscleReferenceStatus::success: return "success";
    case MujocoMuscleReferenceStatus::invalidDefinition: return "invalidDefinition";
    case MujocoMuscleReferenceStatus::invalidState: return "invalidState";
    case MujocoMuscleReferenceStatus::invalidPath: return "invalidPath";
    case MujocoMuscleReferenceStatus::kinematicsFailure: return "kinematicsFailure";
    case MujocoMuscleReferenceStatus::nonfiniteResult: return "nonfiniteResult";
    }
    return "unknown";
}

} // namespace metalrobo
