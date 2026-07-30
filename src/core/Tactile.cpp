#include "metalrobo/Tactile.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numbers>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr double kTiny = 1.0e-14;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Quaternion {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 1.0;
};

struct Pose {
    Vec3 position;
    Quaternion orientation;
};

struct RayHit {
    double depth = 0.0;
    Vec3 point;
    std::uint32_t shape = MR_INVALID_INDEX;
    std::uint32_t triangleTests = 0u;
};

Vec3 operator+(const Vec3 a, const Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3 a, const Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator-(const Vec3 value) {
    return {-value.x, -value.y, -value.z};
}

Vec3 operator*(const Vec3 value, const double scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

Vec3 operator/(const Vec3 value, const double scalar) {
    return {value.x / scalar, value.y / scalar, value.z / scalar};
}

double dot(const Vec3 a, const Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3 a, const Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

double lengthSquared(const Vec3 value) {
    return dot(value, value);
}

double length(const Vec3 value) {
    return std::sqrt(lengthSquared(value));
}

Vec3 normalized(const Vec3 value) {
    const double magnitude = length(value);
    return magnitude > kTiny ? value / magnitude : Vec3{};
}

Vec3 vector(const mr_float4 value) {
    return {value.x, value.y, value.z};
}

mr_float4 packed(const Vec3 value, const float w = 0.0f) {
    return {
        static_cast<float>(value.x),
        static_cast<float>(value.y),
        static_cast<float>(value.z),
        w,
    };
}

Quaternion quaternion(const mr_float4 value) {
    return {value.x, value.y, value.z, value.w};
}

mr_float4 packed(const Quaternion value) {
    return {
        static_cast<float>(value.x),
        static_cast<float>(value.y),
        static_cast<float>(value.z),
        static_cast<float>(value.w),
    };
}

Quaternion conjugate(const Quaternion value) {
    return {-value.x, -value.y, -value.z, value.w};
}

Quaternion multiply(const Quaternion a, const Quaternion b) {
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    };
}

Vec3 rotate(const Quaternion q, const Vec3 value) {
    const Vec3 axis{q.x, q.y, q.z};
    const Vec3 twiceCross = cross(axis, value) * 2.0;
    return value + twiceCross * q.w + cross(axis, twiceCross);
}

Pose compose(const Pose parent, const Pose local) {
    return {
        parent.position + rotate(parent.orientation, local.position),
        multiply(parent.orientation, local.orientation),
    };
}

Vec3 inverseTransformPoint(const Pose pose, const Vec3 point) {
    return rotate(conjugate(pose.orientation), point - pose.position);
}

Vec3 inverseTransformVector(const Pose pose, const Vec3 direction) {
    return rotate(conjugate(pose.orientation), direction);
}

Vec3 pointVelocity(
    const MRBodyStateGPU& body,
    const Vec3 worldPoint
) {
    return
        vector(body.linearVelocityAndInverseMass) +
        cross(
            vector(body.angularVelocity),
            worldPoint - vector(body.position)
        );
}

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const float value) {
    return std::isfinite(value);
}

bool finite(const mr_float4 value) {
    return finite(value.x) && finite(value.y) &&
        finite(value.z) && finite(value.w);
}

bool unitVector(const mr_float4 value) {
    if (!finite(value)) {
        return false;
    }
    const double squared =
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z;
    return std::abs(squared - 1.0) <= 2.0e-4;
}

bool unitQuaternion(const mr_float4 value) {
    if (!finite(value)) {
        return false;
    }
    const double squared =
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z +
        static_cast<double>(value.w) * value.w;
    return std::abs(squared - 1.0) <= 2.0e-5;
}

bool supportedTarget(const std::uint32_t shapeType) {
    return shapeType == MR_SHAPE_SPHERE ||
        shapeType == MR_SHAPE_CAPSULE ||
        shapeType == MR_SHAPE_BOX ||
        shapeType == MR_SHAPE_CYLINDER ||
        shapeType == MR_SHAPE_CONVEX ||
        shapeType == MR_SHAPE_TRIANGLE_MESH;
}

bool filtersAdmit(
    const MRShapeGPU& backing,
    const MRShapeGPU& target
) {
    return
        (backing.collisionGroup & target.collisionMask) != 0u &&
        (target.collisionGroup & backing.collisionMask) != 0u;
}

std::uint64_t appendHash(
    std::uint64_t hash,
    const void* bytes,
    const std::size_t size
) {
    const auto* values =
        static_cast<const unsigned char*>(bytes);
    for (std::size_t index = 0u; index < size; ++index) {
        hash ^= values[index];
        hash *= kFnvPrime;
    }
    return hash;
}

template <typename T>
std::uint64_t appendHash(std::uint64_t hash, const T& value) {
    return appendHash(hash, &value, sizeof(value));
}

std::uint64_t appendHash(
    std::uint64_t hash,
    const std::string_view value
) {
    const std::uint64_t size = value.size();
    hash = appendHash(hash, size);
    return appendHash(hash, value.data(), value.size());
}

template <typename T>
std::uint64_t appendHash(
    std::uint64_t hash,
    const std::span<const T> values
) {
    const std::uint64_t count = values.size();
    hash = appendHash(hash, count);
    if (!values.empty()) {
        hash = appendHash(hash, values.data(), values.size_bytes());
    }
    return hash;
}

TactileCookResult cookFailure(
    const TactileCookStatus status,
    std::string message
) {
    return {status, std::move(message)};
}

TactileObserveResult observeFailure(
    const MRTactileStatusCode status,
    std::string message
) {
    return {status, std::move(message)};
}

Pose bodyPose(const MRBodyStateGPU& body) {
    return {
        vector(body.position),
        quaternion(body.orientation),
    };
}

Pose sensorPose(
    const MRTactileSensorGPU& sensor,
    const MRBodyStateGPU& body
) {
    return compose(
        bodyPose(body),
        {
            vector(sensor.localPositionAndQueryEpsilon),
            quaternion(sensor.localOrientation),
        }
    );
}

Pose shapePose(
    const MRShapeGPU& shape,
    const MRBodyStateGPU& body
) {
    return compose(
        bodyPose(body),
        {
            vector(shape.localPosition),
            quaternion(shape.localRotation),
        }
    );
}

std::optional<double> sphereExit(
    const Vec3 origin,
    const Vec3 direction,
    const double radius,
    const double tolerance
) {
    const double squared = lengthSquared(origin);
    if (!(squared < radius * radius + tolerance)) {
        return std::nullopt;
    }
    const double b = dot(origin, direction);
    const double discriminant =
        b * b - (squared - radius * radius);
    if (discriminant < -tolerance) {
        return std::nullopt;
    }
    const double exit =
        -b + std::sqrt(std::max(0.0, discriminant));
    return exit >= -tolerance
        ? std::optional<double>{std::max(0.0, exit)}
        : std::nullopt;
}

std::optional<double> boxExit(
    const Vec3 origin,
    const Vec3 direction,
    const Vec3 halfExtent,
    const double tolerance
) {
    if (std::abs(origin.x) > halfExtent.x + tolerance ||
        std::abs(origin.y) > halfExtent.y + tolerance ||
        std::abs(origin.z) > halfExtent.z + tolerance) {
        return std::nullopt;
    }
    double exit = std::numeric_limits<double>::infinity();
    const auto axis = [&](const double position,
                          const double velocity,
                          const double extent) {
        if (std::abs(velocity) <= kTiny) {
            return;
        }
        const double boundary =
            velocity > 0.0 ? extent : -extent;
        const double candidate =
            (boundary - position) / velocity;
        if (candidate >= -tolerance) {
            exit = std::min(exit, std::max(0.0, candidate));
        }
    };
    axis(origin.x, direction.x, halfExtent.x);
    axis(origin.y, direction.y, halfExtent.y);
    axis(origin.z, direction.z, halfExtent.z);
    return finite(exit)
        ? std::optional<double>{exit}
        : std::nullopt;
}

std::optional<double> cylinderExit(
    const Vec3 origin,
    const Vec3 direction,
    const double radius,
    const double halfLength,
    const double tolerance
) {
    const double radialSquared =
        origin.x * origin.x + origin.z * origin.z;
    if (radialSquared > radius * radius + tolerance ||
        std::abs(origin.y) > halfLength + tolerance) {
        return std::nullopt;
    }
    double exit = std::numeric_limits<double>::infinity();
    if (std::abs(direction.y) > kTiny) {
        const double cap =
            (direction.y > 0.0 ? halfLength : -halfLength);
        const double candidate =
            (cap - origin.y) / direction.y;
        if (candidate >= -tolerance) {
            exit = std::min(exit, std::max(0.0, candidate));
        }
    }
    const double a =
        direction.x * direction.x +
        direction.z * direction.z;
    if (a > kTiny) {
        const double b =
            origin.x * direction.x +
            origin.z * direction.z;
        const double c = radialSquared - radius * radius;
        const double discriminant = b * b - a * c;
        if (discriminant >= -tolerance) {
            const double candidate =
                (-b + std::sqrt(std::max(0.0, discriminant))) /
                a;
            if (candidate >= -tolerance) {
                exit = std::min(
                    exit,
                    std::max(0.0, candidate)
                );
            }
        }
    }
    return finite(exit)
        ? std::optional<double>{exit}
        : std::nullopt;
}

double capsuleSignedDistance(
    const Vec3 point,
    const double radius,
    const double halfLength
) {
    const double y =
        std::clamp(point.y, -halfLength, halfLength);
    return length(point - Vec3{0.0, y, 0.0}) - radius;
}

std::optional<double> capsuleExit(
    const Vec3 origin,
    const Vec3 direction,
    const double radius,
    const double halfLength,
    const double maximumQueryDepth,
    const double tolerance
) {
    if (capsuleSignedDistance(origin, radius, halfLength) >
        tolerance) {
        return std::nullopt;
    }
    if (capsuleSignedDistance(
            origin + direction * maximumQueryDepth,
            radius,
            halfLength
        ) <= 0.0) {
        return maximumQueryDepth;
    }
    double lower = 0.0;
    double upper = maximumQueryDepth;
    // A bounded bisection is deterministic and reaches sub-nanometre error
    // for millimetre-scale shells. It avoids branchy capsule feature cases.
    for (std::uint32_t iteration = 0u; iteration < 40u; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        if (capsuleSignedDistance(
                origin + direction * middle,
                radius,
                halfLength
            ) <= 0.0) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    return 0.5 * (lower + upper);
}

std::optional<double> convexExit(
    const Vec3 origin,
    const Vec3 direction,
    const MRShapeGPU& shape,
    const EngineModel& model,
    const double tolerance
) {
    if (shape.geometryOffset >= model.geometryHeaders.size()) {
        return std::nullopt;
    }
    const MRGeometryHeaderGPU& geometry =
        model.geometryHeaders[shape.geometryOffset];
    if (geometry.kind != MR_GEOMETRY_CONVEX ||
        geometry.faceOffset > model.convexFaces.size() ||
        geometry.faceCount >
            model.convexFaces.size() - geometry.faceOffset ||
        shape.dimensions.x <= 0.0f ||
        shape.dimensions.y <= 0.0f ||
        shape.dimensions.z <= 0.0f) {
        return std::nullopt;
    }
    const Vec3 scaledOrigin{
        origin.x / shape.dimensions.x,
        origin.y / shape.dimensions.y,
        origin.z / shape.dimensions.z,
    };
    const Vec3 scaledDirection{
        direction.x / shape.dimensions.x,
        direction.y / shape.dimensions.y,
        direction.z / shape.dimensions.z,
    };
    double exit = std::numeric_limits<double>::infinity();
    for (std::uint32_t index = 0u;
         index < geometry.faceCount;
         ++index) {
        const MRConvexFaceGPU& face =
            model.convexFaces[geometry.faceOffset + index];
        const Vec3 normal = vector(face.plane);
        const double signedDistance =
            dot(normal, scaledOrigin) - face.plane.w;
        if (signedDistance > tolerance) {
            return std::nullopt;
        }
        const double denominator = dot(normal, scaledDirection);
        if (denominator > kTiny) {
            const double candidate =
                -signedDistance / denominator;
            if (candidate >= -tolerance) {
                exit = std::min(
                    exit,
                    std::max(0.0, candidate)
                );
            }
        }
    }
    return finite(exit)
        ? std::optional<double>{exit}
        : std::nullopt;
}

bool rayTriangle(
    const Vec3 origin,
    const Vec3 direction,
    const Vec3 a,
    const Vec3 b,
    const Vec3 c,
    double& parameter
) {
    const Vec3 edge1 = b - a;
    const Vec3 edge2 = c - a;
    const Vec3 p = cross(direction, edge2);
    const double determinant = dot(edge1, p);
    const double scale =
        std::max({length(edge1), length(edge2), 1.0});
    const double epsilon =
        64.0 * std::numeric_limits<double>::epsilon() * scale;
    if (std::abs(determinant) <= epsilon) {
        return false;
    }
    const double inverse = 1.0 / determinant;
    const Vec3 translated = origin - a;
    const double u = dot(translated, p) * inverse;
    if (u < -epsilon || u > 1.0 + epsilon) {
        return false;
    }
    const Vec3 q = cross(translated, edge1);
    const double v = dot(direction, q) * inverse;
    if (v < -epsilon || u + v > 1.0 + epsilon) {
        return false;
    }
    const double candidate = dot(edge2, q) * inverse;
    if (candidate <= epsilon) {
        return false;
    }
    parameter = candidate;
    return true;
}

std::optional<double> meshExit(
    const Vec3 origin,
    const Vec3 direction,
    const MRShapeGPU& shape,
    const EngineModel& model,
    std::uint32_t& triangleTests
) {
    if (shape.geometryOffset >= model.geometryHeaders.size() ||
        shape.dimensions.x <= 0.0f ||
        shape.dimensions.y <= 0.0f ||
        shape.dimensions.z <= 0.0f) {
        return std::nullopt;
    }
    const MRGeometryHeaderGPU& geometry =
        model.geometryHeaders[shape.geometryOffset];
    if (geometry.kind != MR_GEOMETRY_TRIANGLE_MESH ||
        (geometry.flags & MR_GEOMETRY_FLAG_CLOSED) == 0u ||
        geometry.triangleOffset > model.meshTriangles.size() ||
        geometry.triangleCount >
            model.meshTriangles.size() - geometry.triangleOffset) {
        return std::nullopt;
    }
    const Vec3 localOrigin{
        origin.x / shape.dimensions.x,
        origin.y / shape.dimensions.y,
        origin.z / shape.dimensions.z,
    };
    const Vec3 localDirection{
        direction.x / shape.dimensions.x,
        direction.y / shape.dimensions.y,
        direction.z / shape.dimensions.z,
    };
    const Vec3 parityDirection =
        normalized(Vec3{1.0, 0.1732050807568877, 0.071});
    std::uint32_t parity = 0u;
    double exit = std::numeric_limits<double>::infinity();
    for (std::uint32_t index = 0u;
         index < geometry.triangleCount;
         ++index) {
        const MRMeshTriangleGPU& triangle =
            model.meshTriangles[geometry.triangleOffset + index];
        const std::array vertexIndices{
            triangle.verticesAndFeature.x,
            triangle.verticesAndFeature.y,
            triangle.verticesAndFeature.z,
        };
        if (std::ranges::any_of(
                vertexIndices,
                [&](const std::uint32_t vertexIndex) {
                    return vertexIndex >=
                        model.geometryVertices.size();
                }
            )) {
            return std::nullopt;
        }
        const Vec3 a = vector(
            model.geometryVertices[vertexIndices[0]]
        );
        const Vec3 b = vector(
            model.geometryVertices[vertexIndices[1]]
        );
        const Vec3 c = vector(
            model.geometryVertices[vertexIndices[2]]
        );
        double parameter = 0.0;
        ++triangleTests;
        if (rayTriangle(
                localOrigin,
                parityDirection,
                a,
                b,
                c,
                parameter
            )) {
            ++parity;
        }
        ++triangleTests;
        if (rayTriangle(
                localOrigin,
                localDirection,
                a,
                b,
                c,
                parameter
            )) {
            exit = std::min(exit, parameter);
        }
    }
    if ((parity & 1u) == 0u || !finite(exit)) {
        return std::nullopt;
    }
    return exit;
}

std::optional<double> shapeExit(
    const Vec3 worldOrigin,
    const Vec3 worldDirection,
    const double maximumDepth,
    const double queryEpsilon,
    const MRShapeGPU& shape,
    const MRBodyStateGPU& body,
    const EngineModel& model,
    std::uint32_t& triangleTests
) {
    const Pose pose = shapePose(shape, body);
    const Vec3 origin = inverseTransformPoint(pose, worldOrigin);
    const Vec3 direction =
        inverseTransformVector(pose, worldDirection);
    const double tolerance =
        std::max<double>(queryEpsilon, 32.0 *
            std::numeric_limits<float>::epsilon() *
            (maximumDepth + 1.0));
    switch (shape.shapeType) {
    case MR_SHAPE_SPHERE:
        return sphereExit(
            origin,
            direction,
            shape.dimensions.x,
            tolerance
        );
    case MR_SHAPE_BOX:
        return boxExit(
            origin,
            direction,
            vector(shape.dimensions),
            tolerance
        );
    case MR_SHAPE_CYLINDER:
        return cylinderExit(
            origin,
            direction,
            shape.dimensions.x,
            shape.dimensions.y,
            tolerance
        );
    case MR_SHAPE_CAPSULE:
        return capsuleExit(
            origin,
            direction,
            shape.dimensions.x,
            shape.dimensions.y,
            maximumDepth,
            tolerance
        );
    case MR_SHAPE_CONVEX:
        return convexExit(
            origin,
            direction,
            shape,
            model,
            tolerance
        );
    case MR_SHAPE_TRIANGLE_MESH:
        return meshExit(
            origin,
            direction,
            shape,
            model,
            triangleTests
        );
    default:
        return std::nullopt;
    }
}

std::string escapedJSON(const std::string_view input) {
    std::string result;
    result.reserve(input.size() + 8u);
    for (const char value : input) {
        switch (value) {
        case '\\':
            result += "\\\\";
            break;
        case '"':
            result += "\\\"";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            result += value;
            break;
        }
    }
    return result;
}

} // namespace

TactileSensorSpec makeFlatTactileSensor(
    std::string id,
    const std::uint32_t parentBodyIndex,
    std::vector<std::uint32_t> backingShapeIndices,
    const TactilePose localPose,
    const std::uint32_t width,
    const std::uint32_t height,
    const float physicalWidthMeters,
    const float physicalHeightMeters,
    const float maximumDepthMeters
) {
    TactileSensorSpec result;
    result.id = std::move(id);
    result.parentBodyIndex = parentBodyIndex;
    result.backingShapeIndices = std::move(backingShapeIndices);
    result.localPose = localPose;
    result.width = width;
    result.height = height;
    result.surfaceKind = MR_TACTILE_SURFACE_FLAT;
    result.maximumDepthMeters = maximumDepthMeters;
    result.maximumTangentialDisplacementMeters =
        maximumDepthMeters;
    if (width == 0u || height == 0u ||
        !(physicalWidthMeters > 0.0f) ||
        !(physicalHeightMeters > 0.0f)) {
        return result;
    }
    const double cellWidth =
        static_cast<double>(physicalWidthMeters) / width;
    const double cellHeight =
        static_cast<double>(physicalHeightMeters) / height;
    result.samples.reserve(
        static_cast<std::size_t>(width) * height
    );
    for (std::uint32_t v = 0u; v < height; ++v) {
        for (std::uint32_t u = 0u; u < width; ++u) {
            const double x =
                (static_cast<double>(u) + 0.5) * cellWidth -
                0.5 * physicalWidthMeters;
            const double y =
                (static_cast<double>(v) + 0.5) * cellHeight -
                0.5 * physicalHeightMeters;
            result.samples.push_back({
                packed(Vec3{x, y, 0.0}),
                {0.0f, 0.0f, 1.0f, 0.0f},
                {1.0f, 0.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f, 0.0f},
                static_cast<float>(cellWidth * cellHeight),
                maximumDepthMeters,
                u,
                v,
                true,
            });
        }
    }
    return result;
}

TactileSensorSpec makeSphericalTactileSensor(
    std::string id,
    const std::uint32_t parentBodyIndex,
    std::vector<std::uint32_t> backingShapeIndices,
    const TactilePose localPose,
    const std::uint32_t width,
    const std::uint32_t height,
    const mr_float4 sphereCenterLocal,
    const float undeformedRadiusMeters,
    const float horizontalSpanRadians,
    const float verticalSpanRadians,
    const float maximumDepthMeters
) {
    TactileSensorSpec result;
    result.id = std::move(id);
    result.parentBodyIndex = parentBodyIndex;
    result.backingShapeIndices = std::move(backingShapeIndices);
    result.localPose = localPose;
    result.width = width;
    result.height = height;
    result.surfaceKind = MR_TACTILE_SURFACE_CURVED;
    result.maximumDepthMeters = maximumDepthMeters;
    result.maximumTangentialDisplacementMeters =
        maximumDepthMeters;
    if (width == 0u || height == 0u ||
        !(undeformedRadiusMeters > 0.0f) ||
        !(horizontalSpanRadians > 0.0f) ||
        !(verticalSpanRadians > 0.0f) ||
        !finite(sphereCenterLocal)) {
        return result;
    }
    const double dTheta =
        static_cast<double>(horizontalSpanRadians) / width;
    const double dPhi =
        static_cast<double>(verticalSpanRadians) / height;
    const Vec3 center = vector(sphereCenterLocal);
    result.samples.reserve(
        static_cast<std::size_t>(width) * height
    );
    for (std::uint32_t v = 0u; v < height; ++v) {
        const double phi =
            (static_cast<double>(v) + 0.5) * dPhi -
            0.5 * verticalSpanRadians;
        const double sinPhi = std::sin(phi);
        const double cosPhi = std::cos(phi);
        for (std::uint32_t u = 0u; u < width; ++u) {
            const double theta =
                (static_cast<double>(u) + 0.5) * dTheta -
                0.5 * horizontalSpanRadians;
            const double sinTheta = std::sin(theta);
            const double cosTheta = std::cos(theta);
            const Vec3 normal{
                cosPhi * sinTheta,
                sinPhi,
                cosPhi * cosTheta,
            };
            const Vec3 tangentU = normalized({
                cosTheta,
                0.0,
                -sinTheta,
            });
            const Vec3 tangentV = normalized({
                -sinPhi * sinTheta,
                cosPhi,
                -sinPhi * cosTheta,
            });
            const double area =
                static_cast<double>(undeformedRadiusMeters) *
                undeformedRadiusMeters *
                std::max(0.0, cosPhi) * dTheta * dPhi;
            result.samples.push_back({
                packed(
                    center +
                    normal * undeformedRadiusMeters
                ),
                packed(normal),
                packed(tangentU),
                packed(tangentV),
                static_cast<float>(area),
                maximumDepthMeters,
                u,
                v,
                true,
            });
        }
    }
    return result;
}

bool CookedTactileSystem::valid(
    const EngineModel& model,
    std::string* reason
) const {
    const auto reject = [&](std::string message) {
        if (reason != nullptr) {
            *reason = std::move(message);
        }
        return false;
    };
    if (reason != nullptr) {
        reason->clear();
    }
    if (abiVersion != MR_TACTILE_ABI_VERSION ||
        fingerprint == 0u ||
        sensorIds.size() != sensors.size() ||
        shapeToSensor.size() != model.shapes.size()) {
        return reject("tactile system identity is invalid");
    }
    std::unordered_set<std::string> ids;
    std::vector<std::uint32_t> expectedShapeOwners(
        model.shapes.size(),
        MR_INVALID_INDEX
    );
    std::size_t expectedSample = 0u;
    std::size_t expectedBacking = 0u;
    std::size_t expectedTarget = 0u;
    for (std::uint32_t sensorIndex = 0u;
         sensorIndex < sensors.size();
         ++sensorIndex) {
        const MRTactileSensorGPU& sensor = sensors[sensorIndex];
        if (sensorIds[sensorIndex].empty() ||
            !ids.insert(sensorIds[sensorIndex]).second ||
            sensor.topology.x >= model.bodies.size() ||
            sensor.topology.y != 0u ||
            sensor.topology.z != expectedSample ||
            sensor.topology.w == 0u ||
            sensor.backingRange.x != expectedBacking ||
            sensor.backingRange.y == 0u ||
            sensor.backingRange.z != 0u ||
            sensor.backingRange.w != 0u ||
            sensor.atlasAndTargets.x == 0u ||
            sensor.atlasAndTargets.y == 0u ||
            static_cast<std::uint64_t>(
                sensor.atlasAndTargets.x
            ) * sensor.atlasAndTargets.y != sensor.topology.w ||
            sensor.atlasAndTargets.z != expectedTarget ||
            sensor.atlasAndTargets.w >
                MR_TACTILE_MAX_TARGETS_PER_SENSOR ||
            sensor.scheduleAndIdentity.x == 0u ||
            sensor.scheduleAndIdentity.y >
                MR_TACTILE_SURFACE_CUSTOM_ATLAS ||
            (sensor.scheduleAndIdentity.z &
             ~MR_TACTILE_SENSOR_COMPLIANT_SHELL) != 0u ||
            sensor.scheduleAndIdentity.w != sensorIndex ||
            !finite(sensor.localPositionAndQueryEpsilon) ||
            !unitQuaternion(sensor.localOrientation) ||
            !finite(sensor.depth) ||
            !(sensor.localPositionAndQueryEpsilon.w >= 0.0f) ||
            !(sensor.depth.x > 0.0f) ||
            !(sensor.depth.y >= 0.0f) ||
            sensor.depth.y > sensor.depth.x ||
            sensor.depth.z + 1.0e-7f < sensor.depth.x ||
            !(sensor.depth.w > 0.0f)) {
            return reject("tactile sensor descriptor is invalid");
        }

        expectedBacking += sensor.backingRange.y;
        if (expectedBacking > backingShapeIndices.size()) {
            return reject("tactile backing range is invalid");
        }
        const MRShapeGPU* filterReference = nullptr;
        for (std::uint32_t local = 0u;
             local < sensor.backingRange.y;
             ++local) {
            const std::uint32_t shapeIndex =
                backingShapeIndices[sensor.backingRange.x + local];
            if (shapeIndex >= model.shapes.size() ||
                expectedShapeOwners[shapeIndex] != MR_INVALID_INDEX ||
                shapeToSensor[shapeIndex] != sensorIndex) {
                return reject(
                    "tactile backing ownership is invalid or duplicated"
                );
            }
            const MRShapeGPU& backing = model.shapes[shapeIndex];
            if (backing.bodyIndex != sensor.topology.x ||
                (backing.flags &
                 MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u) {
                return reject(
                    "tactile backing is disabled or owned by another body"
                );
            }
            if (filterReference == nullptr) {
                filterReference = &backing;
            } else if (
                backing.collisionGroup !=
                    filterReference->collisionGroup ||
                backing.collisionMask !=
                    filterReference->collisionMask
            ) {
                return reject(
                    "compound tactile backing filters disagree"
                );
            } else if (
                std::abs(
                    backing.contactRestAndBoundingRadius.x -
                    filterReference
                        ->contactRestAndBoundingRadius.x
                ) > 1.0e-6f
            ) {
                return reject(
                    "compound tactile backing contact offsets disagree"
                );
            }
            if ((sensor.scheduleAndIdentity.z &
                 MR_TACTILE_SENSOR_COMPLIANT_SHELL) != 0u &&
                (!(backing.contactRestAndBoundingRadius.y > 0.0f) ||
                 backing.contactRestAndBoundingRadius.x <
                     backing.contactRestAndBoundingRadius.y ||
                 std::abs(
                     backing.contactRestAndBoundingRadius.y -
                     sensor.depth.z
                 ) > 1.0e-6f)) {
                return reject(
                    "tactile shell disagrees with a backing-shape rest "
                    "offset"
                );
            }
            expectedShapeOwners[shapeIndex] = sensorIndex;
        }

        expectedSample += sensor.topology.w;
        expectedTarget += sensor.atlasAndTargets.w;
        if (expectedSample > samples.size() ||
            expectedTarget > targetShapeIndices.size()) {
            return reject("tactile descriptor range is invalid");
        }
        for (std::uint32_t local = 0u;
             local < sensor.topology.w;
             ++local) {
            const MRTactileSampleGPU& sample =
                samples[sensor.topology.z + local];
            const std::uint32_t expectedU =
                local % sensor.atlasAndTargets.x;
            const std::uint32_t expectedV =
                local / sensor.atlasAndTargets.x;
            if (sample.atlasAndIdentity.x != expectedU ||
                sample.atlasAndIdentity.y != expectedV ||
                sample.atlasAndIdentity.z != sensorIndex ||
                (sample.atlasAndIdentity.w &
                 ~MR_TACTILE_SAMPLE_VALID) != 0u ||
                !finite(sample.localPositionAndArea) ||
                !finite(sample.localNormalAndMaximumDepth) ||
                !finite(sample.localTangentU) ||
                !finite(sample.localTangentV)) {
                return reject("tactile sample atlas is invalid");
            }
            if ((sample.atlasAndIdentity.w &
                 MR_TACTILE_SAMPLE_VALID) != 0u) {
                if (!(sample.localPositionAndArea.w > 0.0f) ||
                    !unitVector(
                        sample.localNormalAndMaximumDepth
                    ) ||
                    !unitVector(sample.localTangentU) ||
                    !unitVector(sample.localTangentV) ||
                    !(sample.localNormalAndMaximumDepth.w > 0.0f) ||
                    sample.localNormalAndMaximumDepth.w >
                        sensor.depth.x + 1.0e-7f ||
                    std::abs(dot(
                        vector(sample.localNormalAndMaximumDepth),
                        vector(sample.localTangentU)
                    )) > 2.0e-4 ||
                    std::abs(dot(
                        vector(sample.localNormalAndMaximumDepth),
                        vector(sample.localTangentV)
                    )) > 2.0e-4 ||
                    std::abs(dot(
                        vector(sample.localTangentU),
                        vector(sample.localTangentV)
                    )) > 2.0e-4) {
                    return reject(
                        "tactile sample frame or metric area is invalid"
                    );
                }
            }
        }
        for (std::uint32_t local = 0u;
             local < sensor.atlasAndTargets.w;
             ++local) {
            const std::uint32_t shapeIndex =
                targetShapeIndices[
                    sensor.atlasAndTargets.z + local
                ];
            if (shapeIndex >= model.shapes.size() ||
                expectedShapeOwners[shapeIndex] == sensorIndex ||
                model.shapes[shapeIndex].bodyIndex ==
                    sensor.topology.x ||
                !supportedTarget(
                    model.shapes[shapeIndex].shapeType
                )) {
                return reject("tactile target shape is invalid");
            }
            for (std::uint32_t backingLocal = 0u;
                 backingLocal < sensor.backingRange.y;
                 ++backingLocal) {
                const MRShapeGPU& backing = model.shapes[
                    backingShapeIndices[
                        sensor.backingRange.x + backingLocal
                    ]
                ];
                if (!filtersAdmit(backing, model.shapes[shapeIndex])) {
                    return reject(
                        "tactile target is rejected by a backing filter"
                    );
                }
            }
            if (model.shapes[shapeIndex].shapeType ==
                MR_SHAPE_TRIANGLE_MESH) {
                const std::uint32_t geometryIndex =
                    model.shapes[shapeIndex].geometryOffset;
                if (geometryIndex >= model.geometryHeaders.size() ||
                    (model.geometryHeaders[geometryIndex].flags &
                     MR_GEOMETRY_FLAG_CLOSED) == 0u) {
                    return reject(
                        "tactile mesh target is not a closed volume"
                    );
                }
            }
        }
    }
    if (expectedSample != samples.size() ||
        expectedBacking != backingShapeIndices.size() ||
        expectedTarget != targetShapeIndices.size()) {
        return reject("tactile arenas contain unreferenced records");
    }
    if (expectedShapeOwners != shapeToSensor) {
        return reject("tactile shape ownership lookup is inconsistent");
    }
    return true;
}

TactileObservationSchema
CookedTactileSystem::observationSchema() const {
    TactileObservationSchema result;
    result.fingerprint = fingerprint;
    result.sensorCount =
        static_cast<std::uint32_t>(sensors.size());
    result.totalSampleCount =
        static_cast<std::uint32_t>(samples.size());
    return result;
}

TactileCookResult cookTactileSystem(
    const std::span<const TactileSensorSpec> sensors,
    const EngineModel& model,
    CookedTactileSystem& output
) {
    if (sensors.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        model.shapes.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        return cookFailure(
            TactileCookStatus::capacityOverflow,
            "tactile sensor or shape count exceeds the ABI"
        );
    }
    CookedTactileSystem candidate;
    candidate.shapeToSensor.assign(
        model.shapes.size(),
        MR_INVALID_INDEX
    );
    std::unordered_set<std::string> ids;
    try {
        for (std::uint32_t sensorIndex = 0u;
             sensorIndex < sensors.size();
             ++sensorIndex) {
            const TactileSensorSpec& source = sensors[sensorIndex];
            const std::uint64_t sampleCount =
                static_cast<std::uint64_t>(source.width) *
                source.height;
            if (source.id.empty() ||
                !ids.insert(source.id).second ||
                source.parentBodyIndex >= model.bodies.size() ||
                source.backingShapeIndices.empty() ||
                source.backingShapeIndices.size() >
                    std::numeric_limits<std::uint32_t>::max() ||
                source.width == 0u || source.height == 0u ||
                sampleCount != source.samples.size() ||
                sampleCount >
                    std::numeric_limits<std::uint32_t>::max() ||
                source.surfaceKind >
                    MR_TACTILE_SURFACE_CUSTOM_ATLAS ||
                !finite(source.localPose.position) ||
                !unitQuaternion(source.localPose.orientation) ||
                !finite(source.maximumDepthMeters) ||
                !(source.maximumDepthMeters > 0.0f) ||
                !finite(
                    source.maximumTangentialDisplacementMeters
                ) ||
                !(
                    source.maximumTangentialDisplacementMeters >
                    0.0f
                ) ||
                !finite(source.activeDepthThresholdMeters) ||
                source.activeDepthThresholdMeters < 0.0f ||
                source.activeDepthThresholdMeters >
                    source.maximumDepthMeters ||
                !finite(source.queryEpsilonMeters) ||
                source.queryEpsilonMeters < 0.0f ||
                source.updatePeriodSteps == 0u ||
                (source.flags &
                 ~MR_TACTILE_SENSOR_COMPLIANT_SHELL) != 0u) {
                return cookFailure(
                    TactileCookStatus::invalidSpecification,
                    "tactile sensor authoring is invalid"
                );
            }
            std::vector<std::uint32_t> backings =
                source.backingShapeIndices;
            std::ranges::sort(backings);
            if (std::adjacent_find(
                    backings.begin(),
                    backings.end()
                ) != backings.end()) {
                return cookFailure(
                    TactileCookStatus::invalidBackingShape,
                    "tactile backing list contains a duplicate shape"
                );
            }
            const MRShapeGPU* filterReference = nullptr;
            for (const std::uint32_t shapeIndex : backings) {
                if (shapeIndex >= model.shapes.size() ||
                    candidate.shapeToSensor[shapeIndex] !=
                        MR_INVALID_INDEX) {
                    return cookFailure(
                        TactileCookStatus::invalidBackingShape,
                        "tactile backing shape is invalid or already owned"
                    );
                }
                const MRShapeGPU& backing = model.shapes[shapeIndex];
                if (backing.bodyIndex != source.parentBodyIndex ||
                    (backing.flags &
                     MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u) {
                    return cookFailure(
                        TactileCookStatus::invalidBackingShape,
                        "tactile backing must be an enabled collider on "
                        "the sensor parent body"
                    );
                }
                if (filterReference == nullptr) {
                    filterReference = &backing;
                } else if (
                    backing.collisionGroup !=
                        filterReference->collisionGroup ||
                    backing.collisionMask !=
                        filterReference->collisionMask
                ) {
                    return cookFailure(
                        TactileCookStatus::invalidBackingShape,
                        "compound tactile backing filters disagree"
                    );
                } else if (
                    std::abs(
                        backing.contactRestAndBoundingRadius.x -
                        filterReference
                            ->contactRestAndBoundingRadius.x
                    ) > 1.0e-6f
                ) {
                    return cookFailure(
                        TactileCookStatus::invalidBackingShape,
                        "compound tactile backing contact offsets "
                        "disagree"
                    );
                }
            }
            const MRShapeGPU& backing = *filterReference;
            const float shellThickness =
                backing.contactRestAndBoundingRadius.y;
            for (const std::uint32_t shapeIndex : backings) {
                const MRShapeGPU& compoundBacking =
                    model.shapes[shapeIndex];
                if ((source.flags &
                     MR_TACTILE_SENSOR_COMPLIANT_SHELL) != 0u &&
                    (!(compoundBacking
                           .contactRestAndBoundingRadius.y > 0.0f) ||
                     compoundBacking
                             .contactRestAndBoundingRadius.x <
                         compoundBacking
                             .contactRestAndBoundingRadius.y ||
                     source.maximumDepthMeters >
                         compoundBacking
                                 .contactRestAndBoundingRadius.y +
                             1.0e-7f ||
                     std::abs(
                         compoundBacking
                                 .contactRestAndBoundingRadius.y -
                             shellThickness
                     ) > 1.0e-6f)) {
                    return cookFailure(
                        TactileCookStatus::invalidBackingShape,
                        "tactile depth must fit inside every backing "
                        "collider's common positive rest-offset shell"
                    );
                }
            }
            if (candidate.samples.size() >
                    std::numeric_limits<std::uint32_t>::max() -
                        source.samples.size() ||
                candidate.backingShapeIndices.size() >
                    std::numeric_limits<std::uint32_t>::max() -
                        backings.size() ||
                candidate.targetShapeIndices.size() >
                    std::numeric_limits<std::uint32_t>::max()) {
                return cookFailure(
                    TactileCookStatus::capacityOverflow,
                    "tactile static arena exceeds the ABI"
                );
            }

            std::vector<std::uint32_t> targets =
                source.targetShapeIndices;
            if (targets.empty()) {
                for (std::uint32_t shapeIndex = 0u;
                     shapeIndex < model.shapes.size();
                     ++shapeIndex) {
                    const MRShapeGPU& target =
                        model.shapes[shapeIndex];
                    const bool isBacking =
                        std::binary_search(
                            backings.begin(),
                            backings.end(),
                            shapeIndex
                        );
                    const bool admitted =
                        std::ranges::all_of(
                            backings,
                            [&](const std::uint32_t backingIndex) {
                                return filtersAdmit(
                                    model.shapes[backingIndex],
                                    target
                                );
                            }
                        );
                    if (!isBacking &&
                        target.bodyIndex != source.parentBodyIndex &&
                        (target.flags &
                         MR_SHAPE_FLAG_SIMULATION_DISABLED) == 0u &&
                        supportedTarget(target.shapeType) &&
                        admitted) {
                        targets.push_back(shapeIndex);
                    }
                }
            }
            std::ranges::sort(targets);
            targets.erase(
                std::unique(targets.begin(), targets.end()),
                targets.end()
            );
            if (targets.size() >
                MR_TACTILE_MAX_TARGETS_PER_SENSOR) {
                return cookFailure(
                    TactileCookStatus::capacityOverflow,
                    "tactile target list exceeds its fixed capacity"
                );
            }
            for (const std::uint32_t shapeIndex : targets) {
                const bool isBacking =
                    std::binary_search(
                        backings.begin(),
                        backings.end(),
                        shapeIndex
                    );
                const bool admitted =
                    shapeIndex < model.shapes.size() &&
                    std::ranges::all_of(
                        backings,
                        [&](const std::uint32_t backingIndex) {
                            return filtersAdmit(
                                model.shapes[backingIndex],
                                model.shapes[shapeIndex]
                            );
                        }
                    );
                if (shapeIndex >= model.shapes.size() ||
                    isBacking ||
                    model.shapes[shapeIndex].bodyIndex ==
                        source.parentBodyIndex ||
                    !supportedTarget(
                        model.shapes[shapeIndex].shapeType
                    ) ||
                    !admitted) {
                    return cookFailure(
                        TactileCookStatus::unsupportedGeometry,
                        "tactile target is unsupported, self-owned, or "
                        "rejected by collision filtering"
                    );
                }
                if (model.shapes[shapeIndex].shapeType ==
                    MR_SHAPE_TRIANGLE_MESH) {
                    const std::uint32_t geometryIndex =
                        model.shapes[shapeIndex].geometryOffset;
                    if (geometryIndex >=
                            model.geometryHeaders.size() ||
                        (model.geometryHeaders[geometryIndex].flags &
                         MR_GEOMETRY_FLAG_CLOSED) == 0u) {
                        return cookFailure(
                            TactileCookStatus::unsupportedGeometry,
                            "normal penetration requires a closed mesh "
                            "target"
                        );
                    }
                }
            }

            MRTactileSensorGPU cooked{};
            cooked.topology = {
                source.parentBodyIndex,
                0u,
                static_cast<std::uint32_t>(
                    candidate.samples.size()
                ),
                static_cast<std::uint32_t>(
                    source.samples.size()
                ),
            };
            cooked.backingRange = {
                static_cast<std::uint32_t>(
                    candidate.backingShapeIndices.size()
                ),
                static_cast<std::uint32_t>(backings.size()),
                0u,
                0u,
            };
            cooked.atlasAndTargets = {
                source.width,
                source.height,
                static_cast<std::uint32_t>(
                    candidate.targetShapeIndices.size()
                ),
                static_cast<std::uint32_t>(targets.size()),
            };
            cooked.scheduleAndIdentity = {
                source.updatePeriodSteps,
                source.surfaceKind,
                source.flags,
                sensorIndex,
            };
            cooked.localPositionAndQueryEpsilon = {
                source.localPose.position.x,
                source.localPose.position.y,
                source.localPose.position.z,
                source.queryEpsilonMeters,
            };
            cooked.localOrientation =
                source.localPose.orientation;
            cooked.depth = {
                source.maximumDepthMeters,
                source.activeDepthThresholdMeters,
                shellThickness,
                source.maximumTangentialDisplacementMeters,
            };

            for (std::uint32_t local = 0u;
                 local < source.samples.size();
                 ++local) {
                const TactileSampleSpec& sample =
                    source.samples[local];
                const std::uint32_t expectedU =
                    local % source.width;
                const std::uint32_t expectedV =
                    local / source.width;
                if (sample.atlasU != expectedU ||
                    sample.atlasV != expectedV ||
                    !finite(sample.localPosition) ||
                    !finite(sample.localNormal) ||
                    !finite(sample.localTangentU) ||
                    !finite(sample.localTangentV) ||
                    !finite(sample.areaSquareMeters) ||
                    !finite(sample.maximumDepthMeters)) {
                    return cookFailure(
                        TactileCookStatus::invalidSampleAtlas,
                        "tactile samples must be dense atlas order"
                    );
                }
                if (sample.valid &&
                    (!(sample.areaSquareMeters > 0.0f) ||
                     !(sample.maximumDepthMeters > 0.0f) ||
                     sample.maximumDepthMeters >
                        source.maximumDepthMeters + 1.0e-7f ||
                     !unitVector(sample.localNormal) ||
                     !unitVector(sample.localTangentU) ||
                     !unitVector(sample.localTangentV) ||
                     std::abs(dot(
                         vector(sample.localNormal),
                         vector(sample.localTangentU)
                     )) > 2.0e-4 ||
                     std::abs(dot(
                         vector(sample.localNormal),
                         vector(sample.localTangentV)
                     )) > 2.0e-4 ||
                     std::abs(dot(
                         vector(sample.localTangentU),
                         vector(sample.localTangentV)
                     )) > 2.0e-4)) {
                    return cookFailure(
                        TactileCookStatus::invalidSampleAtlas,
                        "tactile sample frames must be orthonormal and "
                        "carry metric area/depth"
                    );
                }
                candidate.samples.push_back({
                    {
                        sample.localPosition.x,
                        sample.localPosition.y,
                        sample.localPosition.z,
                        sample.valid
                            ? sample.areaSquareMeters
                            : 0.0f,
                    },
                    {
                        sample.localNormal.x,
                        sample.localNormal.y,
                        sample.localNormal.z,
                        sample.valid
                            ? sample.maximumDepthMeters
                            : 0.0f,
                    },
                    sample.localTangentU,
                    sample.localTangentV,
                    {
                        sample.atlasU,
                        sample.atlasV,
                        sensorIndex,
                        sample.valid
                            ? MR_TACTILE_SAMPLE_VALID
                            : 0u,
                    },
                });
            }
            candidate.sensorIds.push_back(source.id);
            candidate.sensors.push_back(cooked);
            candidate.backingShapeIndices.insert(
                candidate.backingShapeIndices.end(),
                backings.begin(),
                backings.end()
            );
            for (const std::uint32_t shapeIndex : backings) {
                candidate.shapeToSensor[shapeIndex] = sensorIndex;
            }
            candidate.targetShapeIndices.insert(
                candidate.targetShapeIndices.end(),
                targets.begin(),
                targets.end()
            );
        }
    } catch (const std::bad_alloc&) {
        return cookFailure(
            TactileCookStatus::capacityOverflow,
            "host allocation failed while cooking tactile sensors"
        );
    }

    std::uint64_t fingerprint = kFnvOffset;
    fingerprint = appendHash(
        fingerprint,
        static_cast<std::uint32_t>(MR_TACTILE_ABI_VERSION)
    );
    for (const std::string& id : candidate.sensorIds) {
        fingerprint = appendHash(
            fingerprint,
            std::string_view{id}
        );
    }
    fingerprint = appendHash<MRTactileSensorGPU>(
        fingerprint,
        candidate.sensors
    );
    fingerprint = appendHash<MRTactileSampleGPU>(
        fingerprint,
        candidate.samples
    );
    fingerprint = appendHash<std::uint32_t>(
        fingerprint,
        candidate.backingShapeIndices
    );
    fingerprint = appendHash<std::uint32_t>(
        fingerprint,
        candidate.shapeToSensor
    );
    fingerprint = appendHash<std::uint32_t>(
        fingerprint,
        candidate.targetShapeIndices
    );
    candidate.fingerprint = fingerprint;
    std::string reason;
    if (!candidate.valid(model, &reason)) {
        return cookFailure(
            TactileCookStatus::invalidSpecification,
            std::move(reason)
        );
    }
    output = std::move(candidate);
    return {};
}

TactileObserveResult packTactileSolverContacts(
    const TactileSolverContactFrame& frame,
    TactileSolverContactBatch& output
) {
    const std::size_t contactArena =
        static_cast<std::size_t>(frame.environmentCount) *
        frame.contactCapacityPerEnvironment;
    const std::size_t manifoldArena =
        static_cast<std::size_t>(frame.environmentCount) *
        frame.manifoldCapacityPerEnvironment;
    if (frame.environmentCount == 0u ||
        frame.bodyCount == 0u ||
        frame.contactCapacityPerEnvironment == 0u ||
        frame.manifoldCapacityPerEnvironment == 0u ||
        frame.shapes.empty() ||
        frame.bodies.size() !=
            static_cast<std::size_t>(frame.environmentCount) *
                frame.bodyCount ||
        frame.constraints.size() != contactArena ||
        frame.metadata.size() != contactArena ||
        frame.manifoldHeaders.size() != manifoldArena ||
        frame.activeContactCounts.size() !=
            frame.environmentCount) {
        return observeFailure(
            MR_TACTILE_INVALID_CONFIGURATION,
            "solver-contact tactile adapter extents are invalid"
        );
    }
    TactileSolverContactBatch candidate;
    try {
        candidate.environmentCount = frame.environmentCount;
        candidate.capacityPerEnvironment =
            frame.contactCapacityPerEnvironment;
        candidate.contacts.assign(contactArena, {});
        candidate.counts.assign(frame.environmentCount, 0u);
    } catch (const std::bad_alloc&) {
        return observeFailure(
            MR_TACTILE_CAPACITY_OVERFLOW,
            "host allocation failed for tactile solver contacts"
        );
    }
    const auto stableTangent = [](const Vec3 normal) {
        const Vec3 absolute{
            std::abs(normal.x),
            std::abs(normal.y),
            std::abs(normal.z),
        };
        Vec3 reference;
        if (absolute.x <= absolute.y &&
            absolute.x <= absolute.z) {
            reference = {1.0, 0.0, 0.0};
        } else if (absolute.y <= absolute.z) {
            reference = {0.0, 1.0, 0.0};
        } else {
            reference = {0.0, 0.0, 1.0};
        }
        return normalized(cross(reference, normal));
    };
    for (std::uint32_t environment = 0u;
         environment < frame.environmentCount;
         ++environment) {
        const std::uint32_t required =
            frame.activeContactCounts[environment];
        if (required > frame.contactCapacityPerEnvironment) {
            return observeFailure(
                MR_TACTILE_CAPACITY_OVERFLOW,
                "solver active contact count exceeds its tactile arena"
            );
        }
        const std::size_t contactBase =
            static_cast<std::size_t>(environment) *
            frame.contactCapacityPerEnvironment;
        const std::size_t bodyBase =
            static_cast<std::size_t>(environment) *
            frame.bodyCount;
        std::uint32_t packedCount = 0u;
        for (std::uint32_t slot = 0u;
             slot < frame.contactCapacityPerEnvironment &&
             packedCount < required;
             ++slot) {
            const MRContactPointMetaGPU& metadata =
                frame.metadata[contactBase + slot];
            if (metadata.colliderA >= frame.shapes.size() ||
                metadata.colliderB >= frame.shapes.size() ||
                metadata.manifoldIndex >=
                    frame.manifoldCapacityPerEnvironment) {
                continue;
            }
            const MRShapeGPU& shapeA =
                frame.shapes[metadata.colliderA];
            const MRShapeGPU& shapeB =
                frame.shapes[metadata.colliderB];
            if (shapeA.bodyIndex >= frame.bodyCount ||
                shapeB.bodyIndex >= frame.bodyCount) {
                return observeFailure(
                    MR_TACTILE_INVALID_CONFIGURATION,
                    "solver contact references an unknown body"
                );
            }
            const MRBodyStateGPU& bodyA =
                frame.bodies[bodyBase + shapeA.bodyIndex];
            if (!unitQuaternion(bodyA.orientation)) {
                return observeFailure(
                    MR_TACTILE_NONFINITE_INPUT,
                    "solver contact body orientation is invalid"
                );
            }
            const MRContactConstraintGPU& contact =
                frame.constraints[contactBase + slot];
            const MRManifoldHeaderGPU& manifold =
                frame.manifoldHeaders[
                    static_cast<std::size_t>(environment) *
                        frame.manifoldCapacityPerEnvironment +
                    metadata.manifoldIndex
                ];
            if (!finite(contact.normal) ||
                !finite(contact.pointAndSeparation) ||
                !finite(contact.impulses) ||
                !finite(contact.friction) ||
                contact.friction.x < 0.0f ||
                contact.friction.y < 0.0f ||
                !finite(manifold.tangentAndMetric) ||
                lengthSquared(vector(contact.normal)) <= kTiny) {
                return observeFailure(
                    MR_TACTILE_NONFINITE_INPUT,
                    "solver contact evidence is non-finite or degenerate"
                );
            }
            Vec3 normal = normalized(vector(contact.normal));
            Vec3 tangent = rotate(
                quaternion(bodyA.orientation),
                vector(manifold.tangentAndMetric)
            );
            tangent = tangent - normal * dot(tangent, normal);
            tangent = lengthSquared(tangent) > 1.0e-12
                ? normalized(tangent)
                : stableTangent(normal);
            const Vec3 bitangent = cross(normal, tangent);
            // Solver rows encode relative B-A velocity along (n,u,v), so a
            // positive cone impulse applies the opposite world impulse to A.
            const Vec3 impulseOnA = -(
                normal * contact.impulses.x +
                tangent * contact.impulses.y +
                bitangent * contact.impulses.z
            );
            candidate.contacts[contactBase + packedCount] = {
                {
                    metadata.colliderA,
                    metadata.colliderB,
                    MR_TACTILE_CONTACT_SOLVER_IMPULSE,
                    0u,
                },
                contact.pointAndSeparation,
                packed(impulseOnA),
                {
                    std::abs(contact.impulses.x),
                    std::hypot(
                        contact.impulses.y,
                        contact.impulses.z
                    ),
                    contact.friction.x,
                    contact.friction.y,
                },
            };
            ++packedCount;
        }
        if (packedCount != required) {
            return observeFailure(
                MR_TACTILE_INVALID_CONFIGURATION,
                "solver evidence did not contain every active contact"
            );
        }
        candidate.counts[environment] = packedCount;
    }
    output = std::move(candidate);
    return {};
}

TactileObserveResult observeTactileCpuReference(
    const CookedTactileSystem& tactile,
    const EngineModel& model,
    const TactileCpuFrame& frame,
    TactileObservationBatch& output
) {
    std::string reason;
    if (!tactile.valid(model, &reason)) {
        return observeFailure(
            MR_TACTILE_INVALID_CONFIGURATION,
            std::move(reason)
        );
    }
    const std::size_t bodyCount = model.bodies.size();
    const std::size_t denseCount =
        static_cast<std::size_t>(frame.environmentCount) *
        tactile.samples.size();
    const std::size_t summaryCount =
        static_cast<std::size_t>(frame.environmentCount) *
        tactile.sensors.size();
    const bool contactsPresent =
        frame.contactCapacityPerEnvironment != 0u;
    if (frame.environmentCount == 0u ||
        !finite(frame.observationTimestepSeconds) ||
        !(frame.observationTimestepSeconds > 0.0f) ||
        (contactsPresent &&
         (
             !finite(frame.contactImpulseTimestepSeconds) ||
             !(frame.contactImpulseTimestepSeconds > 0.0f)
         )) ||
        !finite(frame.timestampSeconds) ||
        frame.bodies.size() !=
            static_cast<std::size_t>(frame.environmentCount) *
                bodyCount ||
        (!frame.contactCounts.empty() &&
         frame.contactCounts.size() != frame.environmentCount) ||
        (contactsPresent &&
         (frame.contacts.size() !=
              static_cast<std::size_t>(frame.environmentCount) *
                  frame.contactCapacityPerEnvironment ||
          frame.contactCounts.size() != frame.environmentCount)) ||
        (!contactsPresent &&
         (!frame.contacts.empty() ||
          !frame.contactCounts.empty())) ||
        (!frame.previousDepthMeters.empty() &&
         frame.previousDepthMeters.size() != denseCount) ||
        (!frame.previousValidity.empty() &&
         frame.previousValidity.size() != denseCount) ||
        (!frame.previousObjectShapeIds.empty() &&
         frame.previousObjectShapeIds.size() != denseCount) ||
        (!frame.previousTangentialMotion.empty() &&
         frame.previousTangentialMotion.size() != denseCount) ||
        (!frame.previousTargetLocalContactAnchors.empty() &&
         frame.previousTargetLocalContactAnchors.size() != denseCount) ||
        (!frame.previousDebugHits.empty() &&
         frame.previousDebugHits.size() != denseCount) ||
        (!frame.resetMask.empty() &&
         frame.resetMask.size() != frame.environmentCount)) {
        return observeFailure(
            MR_TACTILE_INVALID_CONFIGURATION,
            "CPU tactile frame extents or timing are invalid"
        );
    }
    for (std::uint32_t environment = 0u;
         environment < frame.environmentCount;
         ++environment) {
        if (contactsPresent &&
            frame.contactCounts[environment] >
                frame.contactCapacityPerEnvironment) {
            return observeFailure(
                MR_TACTILE_CAPACITY_OVERFLOW,
                "CPU tactile contact count exceeds fixed capacity"
            );
        }
        if (contactsPresent) {
            const std::size_t contactBase =
                static_cast<std::size_t>(environment) *
                frame.contactCapacityPerEnvironment;
            for (std::uint32_t contactIndex = 0u;
                 contactIndex < frame.contactCounts[environment];
                 ++contactIndex) {
                const MRTactileContactGPU& contact =
                    frame.contacts[contactBase + contactIndex];
                if (contact.shapesAndFlags.x >= model.shapes.size() ||
                    contact.shapesAndFlags.y >= model.shapes.size() ||
                    (contact.shapesAndFlags.z &
                     MR_TACTILE_CONTACT_SOLVER_IMPULSE) == 0u ||
                    (contact.shapesAndFlags.z &
                     ~MR_TACTILE_CONTACT_SOLVER_IMPULSE) != 0u ||
                    !finite(contact.worldPoint) ||
                    !finite(contact.worldImpulseOnA) ||
                    !finite(contact.solverImpulseAndFriction) ||
                    contact.solverImpulseAndFriction.x < 0.0f ||
                    contact.solverImpulseAndFriction.y < 0.0f ||
                    contact.solverImpulseAndFriction.z < 0.0f ||
                    contact.solverImpulseAndFriction.w < 0.0f) {
                    return observeFailure(
                        MR_TACTILE_NONFINITE_INPUT,
                        "CPU tactile solver evidence is invalid"
                    );
                }
            }
        }
        for (std::size_t body = 0u; body < bodyCount; ++body) {
            const MRBodyStateGPU& state =
                frame.bodies[environment * bodyCount + body];
            if (!finite(state.position) ||
                !unitQuaternion(state.orientation) ||
                !finite(state.linearVelocityAndInverseMass) ||
                !finite(state.angularVelocity)) {
                return observeFailure(
                    MR_TACTILE_NONFINITE_INPUT,
                    "CPU tactile body pose is non-finite or unnormalized"
                );
            }
        }
    }

    TactileObservationBatch candidate;
    try {
        candidate.environmentCount = frame.environmentCount;
        candidate.sensorCount =
            static_cast<std::uint32_t>(tactile.sensors.size());
        candidate.sampleCount =
            static_cast<std::uint32_t>(tactile.samples.size());
        candidate.frameIndex = frame.frameIndex;
        candidate.timestampSeconds = frame.timestampSeconds;
        candidate.penetrationDepthMeters.assign(denseCount, 0.0f);
        candidate.depthVelocityMetersPerSecond.assign(
            denseCount,
            0.0f
        );
        candidate.tangentialMotion.assign(denseCount, {});
        candidate.validity.assign(denseCount, 0u);
        candidate.objectShapeIds.assign(
            denseCount,
            MR_INVALID_INDEX
        );
        candidate.debugHits.assign(denseCount, {});
        candidate.targetLocalContactAnchors.assign(
            denseCount,
            {}
        );
        candidate.summaries.assign(summaryCount, {});
        candidate.statuses.assign(summaryCount, {});
    } catch (const std::bad_alloc&) {
        return observeFailure(
            MR_TACTILE_CAPACITY_OVERFLOW,
            "host allocation failed for CPU tactile observation"
        );
    }

    for (std::uint32_t environment = 0u;
         environment < frame.environmentCount;
         ++environment) {
        const bool reset =
            !frame.resetMask.empty() &&
            frame.resetMask[environment] != 0u;
        const std::span<const MRBodyStateGPU> bodies =
            frame.bodies.subspan(environment * bodyCount, bodyCount);
        for (std::uint32_t sensorIndex = 0u;
             sensorIndex < tactile.sensors.size();
             ++sensorIndex) {
            const MRTactileSensorGPU& sensor =
                tactile.sensors[sensorIndex];
            const Pose worldSensor = sensorPose(
                sensor,
                bodies[sensor.topology.x]
            );
            const bool update =
                frame.frameIndex %
                    sensor.scheduleAndIdentity.x == 0u;
            std::uint32_t visitedTargets = 0u;
            std::uint32_t triangleTests = 0u;
            for (std::uint32_t localSample = 0u;
                 localSample < sensor.topology.w;
                 ++localSample) {
                const std::uint32_t sampleIndex =
                    sensor.topology.z + localSample;
                const std::size_t outputIndex =
                    static_cast<std::size_t>(environment) *
                        tactile.samples.size() +
                    sampleIndex;
                const MRTactileSampleGPU& sample =
                    tactile.samples[sampleIndex];
                if ((sample.atlasAndIdentity.w &
                     MR_TACTILE_SAMPLE_VALID) == 0u) {
                    continue;
                }
                candidate.validity[outputIndex] =
                    MR_TACTILE_VALIDITY_SAMPLE;
                if (!update &&
                    !frame.previousDepthMeters.empty()) {
                    candidate.penetrationDepthMeters[outputIndex] =
                        frame.previousDepthMeters[outputIndex];
                    candidate.validity[outputIndex] =
                        frame.previousValidity.empty()
                        ? MR_TACTILE_VALIDITY_SAMPLE
                        : frame.previousValidity[outputIndex];
                    candidate.objectShapeIds[outputIndex] =
                        frame.previousObjectShapeIds.empty()
                        ? MR_INVALID_INDEX
                        : frame.previousObjectShapeIds[outputIndex];
                    if (!frame.previousTangentialMotion.empty()) {
                        candidate.tangentialMotion[outputIndex]
                            .displacementAndVelocity = {
                                frame.previousTangentialMotion[
                                    outputIndex
                                ].displacementAndVelocity.x,
                                frame.previousTangentialMotion[
                                    outputIndex
                                ].displacementAndVelocity.y,
                                0.0f,
                                0.0f,
                            };
                    }
                    if (!frame
                             .previousTargetLocalContactAnchors
                             .empty()) {
                        candidate.targetLocalContactAnchors[
                            outputIndex
                        ] =
                            frame
                                .previousTargetLocalContactAnchors[
                                    outputIndex
                                ];
                    }
                    if (!frame.previousDebugHits.empty()) {
                        candidate.debugHits[outputIndex] =
                            frame.previousDebugHits[outputIndex];
                    }
                    continue;
                }

                const Vec3 worldOrigin =
                    worldSensor.position +
                    rotate(
                        worldSensor.orientation,
                        vector(sample.localPositionAndArea)
                    );
                const Vec3 worldNormal = normalized(
                    rotate(
                        worldSensor.orientation,
                        vector(
                            sample.localNormalAndMaximumDepth
                        )
                    )
                );
                const Vec3 worldDirection = -worldNormal;
                const double maximumDepth = std::min<double>(
                    sensor.depth.x,
                    sample.localNormalAndMaximumDepth.w
                );
                RayHit best;
                for (std::uint32_t target = 0u;
                     target < sensor.atlasAndTargets.w;
                     ++target) {
                    ++visitedTargets;
                    const std::uint32_t shapeIndex =
                        tactile.targetShapeIndices[
                            sensor.atlasAndTargets.z + target
                        ];
                    const MRShapeGPU& shape =
                        model.shapes[shapeIndex];
                    std::uint32_t localTriangleTests = 0u;
                    const std::optional<double> exit =
                        shapeExit(
                            worldOrigin,
                            worldDirection,
                            maximumDepth,
                            sensor.localPositionAndQueryEpsilon.w,
                            shape,
                            bodies[shape.bodyIndex],
                            model,
                            localTriangleTests
                        );
                    triangleTests += localTriangleTests;
                    if (!exit.has_value() ||
                        *exit <= sensor.depth.y) {
                        continue;
                    }
                    const double depth =
                        std::clamp(*exit, 0.0, maximumDepth);
                    if (depth > best.depth ||
                        (depth == best.depth &&
                         shapeIndex < best.shape)) {
                        best.depth = depth;
                        best.point =
                            worldOrigin +
                            worldDirection * depth;
                        best.shape = shapeIndex;
                        best.triangleTests =
                            localTriangleTests;
                    }
                }
                if (best.shape != MR_INVALID_INDEX) {
                    std::uint32_t flags =
                        MR_TACTILE_VALIDITY_SAMPLE |
                        MR_TACTILE_VALIDITY_CONTACT |
                        MR_TACTILE_VALIDITY_FILTERED_TARGET;
                    if (best.depth >= maximumDepth -
                            std::max<double>(
                                sensor.localPositionAndQueryEpsilon.w,
                                1.0e-8
                            )) {
                        flags |= MR_TACTILE_VALIDITY_SATURATED;
                    }
                    candidate.penetrationDepthMeters[outputIndex] =
                        static_cast<float>(best.depth);
                    candidate.validity[outputIndex] = flags;
                    candidate.objectShapeIds[outputIndex] =
                        best.shape;
                    candidate.debugHits[outputIndex] = {
                        packed(
                            best.point,
                            static_cast<float>(best.depth)
                        ),
                        packed(
                            worldNormal,
                            static_cast<float>(best.depth)
                        ),
                        {
                            best.shape,
                            localSample,
                            flags,
                            0u,
                        },
                    };

                    const MRShapeGPU& targetShape =
                        model.shapes[best.shape];
                    const MRBodyStateGPU& targetBody =
                        bodies[targetShape.bodyIndex];
                    const Pose targetBodyPose = bodyPose(targetBody);
                    const bool continuingContact =
                        !reset &&
                        !frame.previousValidity.empty() &&
                        !frame.previousObjectShapeIds.empty() &&
                        !frame.previousTangentialMotion.empty() &&
                        !frame
                             .previousTargetLocalContactAnchors
                             .empty() &&
                        (
                            frame.previousValidity[outputIndex] &
                            MR_TACTILE_VALIDITY_CONTACT
                        ) != 0u &&
                        frame.previousObjectShapeIds[outputIndex] ==
                            best.shape;
                    const Vec3 targetLocalAnchor =
                        continuingContact
                        ? vector(
                            frame
                                .previousTargetLocalContactAnchors[
                                    outputIndex
                                ]
                        )
                        : inverseTransformPoint(
                            targetBodyPose,
                            best.point
                        );
                    candidate.targetLocalContactAnchors[
                        outputIndex
                    ] = packed(targetLocalAnchor);

                    Vec3 displacementLocal{};
                    if (continuingContact) {
                        const Vec3 anchorWorld =
                            targetBodyPose.position +
                            rotate(
                                targetBodyPose.orientation,
                                targetLocalAnchor
                            );
                        displacementLocal =
                            inverseTransformPoint(
                                worldSensor,
                                anchorWorld
                            ) -
                            vector(sample.localPositionAndArea);
                    }
                    double displacementU = dot(
                        displacementLocal,
                        vector(sample.localTangentU)
                    );
                    double displacementV = dot(
                        displacementLocal,
                        vector(sample.localTangentV)
                    );
                    const double displacementMagnitude =
                        std::hypot(displacementU, displacementV);
                    if (displacementMagnitude > sensor.depth.w) {
                        const double scale =
                            sensor.depth.w /
                            displacementMagnitude;
                        displacementU *= scale;
                        displacementV *= scale;
                    }

                    const MRBodyStateGPU& sensorBody =
                        bodies[sensor.topology.x];
                    const Vec3 relativeVelocity =
                        pointVelocity(targetBody, best.point) -
                        pointVelocity(sensorBody, best.point);
                    const Vec3 worldTangentU = rotate(
                        worldSensor.orientation,
                        vector(sample.localTangentU)
                    );
                    const Vec3 worldTangentV = rotate(
                        worldSensor.orientation,
                        vector(sample.localTangentV)
                    );
                    candidate.tangentialMotion[outputIndex]
                        .displacementAndVelocity = {
                            static_cast<float>(displacementU),
                            static_cast<float>(displacementV),
                            static_cast<float>(
                                dot(relativeVelocity, worldTangentU)
                            ),
                            static_cast<float>(
                                dot(relativeVelocity, worldTangentV)
                            ),
                        };
                }
                if (!reset &&
                    !frame.previousDepthMeters.empty()) {
                    candidate.depthVelocityMetersPerSecond[
                        outputIndex
                    ] =
                        (
                            candidate.penetrationDepthMeters[
                                outputIndex
                            ] -
                            frame.previousDepthMeters[outputIndex]
                        ) /
                        (
                            frame.observationTimestepSeconds *
                            sensor.scheduleAndIdentity.x
                        );
                }
            }

            const std::size_t summaryIndex =
                static_cast<std::size_t>(environment) *
                    tactile.sensors.size() +
                sensorIndex;
            MRTactileSummaryGPU& summary =
                candidate.summaries[summaryIndex];
            summary.posePositionAndTimestamp = packed(
                worldSensor.position,
                static_cast<float>(frame.timestampSeconds)
            );
            summary.poseOrientation =
                packed(worldSensor.orientation);
            Vec3 centroidLocal{};
            Vec3 centroidWorld{};
            double activeArea = 0.0;
            double areaDepth = 0.0;
            double maximumDepth = 0.0;
            double areaTangentialSpeedSquared = 0.0;
            double maximumTangentialDisplacement = 0.0;
            std::uint32_t activeCount = 0u;
            std::uint32_t saturatedCount = 0u;
            std::uint32_t unanimousObject = MR_INVALID_INDEX;
            bool objectDisagrees = false;
            for (std::uint32_t localSample = 0u;
                 localSample < sensor.topology.w;
                 ++localSample) {
                const std::uint32_t sampleIndex =
                    sensor.topology.z + localSample;
                const std::size_t outputIndex =
                    static_cast<std::size_t>(environment) *
                        tactile.samples.size() +
                    sampleIndex;
                const float depth =
                    candidate.penetrationDepthMeters[outputIndex];
                if (depth <= sensor.depth.y ||
                    (candidate.validity[outputIndex] &
                     MR_TACTILE_VALIDITY_CONTACT) == 0u) {
                    continue;
                }
                const MRTactileSampleGPU& sample =
                    tactile.samples[sampleIndex];
                const double area =
                    sample.localPositionAndArea.w;
                const Vec3 localPoint =
                    vector(sample.localPositionAndArea) -
                    vector(
                        sample.localNormalAndMaximumDepth
                    ) * depth;
                const Vec3 worldPoint =
                    worldSensor.position +
                    rotate(worldSensor.orientation, localPoint);
                centroidLocal =
                    centroidLocal + localPoint * area;
                centroidWorld =
                    centroidWorld + worldPoint * area;
                activeArea += area;
                areaDepth += area * depth;
                maximumDepth =
                    std::max<double>(maximumDepth, depth);
                const mr_float4 motion =
                    candidate.tangentialMotion[outputIndex]
                        .displacementAndVelocity;
                areaTangentialSpeedSquared +=
                    area *
                    (
                        static_cast<double>(motion.z) * motion.z +
                        static_cast<double>(motion.w) * motion.w
                    );
                maximumTangentialDisplacement = std::max(
                    maximumTangentialDisplacement,
                    std::hypot(
                        static_cast<double>(motion.x),
                        static_cast<double>(motion.y)
                    )
                );
                ++activeCount;
                if ((candidate.validity[outputIndex] &
                     MR_TACTILE_VALIDITY_SATURATED) != 0u) {
                    ++saturatedCount;
                }
                const std::uint32_t object =
                    candidate.objectShapeIds[outputIndex];
                if (unanimousObject == MR_INVALID_INDEX) {
                    unanimousObject = object;
                } else if (unanimousObject != object) {
                    objectDisagrees = true;
                }
            }
            if (activeArea > 0.0) {
                centroidLocal = centroidLocal / activeArea;
                centroidWorld = centroidWorld / activeArea;
            }

            Vec3 force{};
            Vec3 torque{};
            Vec3 centerOfPressureWorld{};
            double centerOfPressureForceWeight = 0.0;
            std::uint32_t contactContributors = 0u;
            std::uint32_t centerOfPressureContributors = 0u;
            double frictionUtilizationWeighted = 0.0;
            double frictionUtilizationWeight = 0.0;
            double maximumFrictionUtilization = 0.0;
            if (contactsPresent) {
                const std::size_t contactBase =
                    static_cast<std::size_t>(environment) *
                    frame.contactCapacityPerEnvironment;
                for (std::uint32_t contactIndex = 0u;
                     contactIndex <
                        frame.contactCounts[environment];
                     ++contactIndex) {
                    const MRTactileContactGPU& contact =
                        frame.contacts[contactBase + contactIndex];
                    Vec3 impulse{};
                    const std::uint32_t shapeA =
                        contact.shapesAndFlags.x;
                    const std::uint32_t shapeB =
                        contact.shapesAndFlags.y;
                    const std::uint32_t sensorA =
                        shapeA < tactile.shapeToSensor.size()
                        ? tactile.shapeToSensor[shapeA]
                        : MR_INVALID_INDEX;
                    const std::uint32_t sensorB =
                        shapeB < tactile.shapeToSensor.size()
                        ? tactile.shapeToSensor[shapeB]
                        : MR_INVALID_INDEX;
                    if (sensorA == sensorIndex) {
                        impulse = vector(contact.worldImpulseOnA);
                    } else if (sensorB == sensorIndex) {
                        impulse =
                            -vector(contact.worldImpulseOnA);
                    } else {
                        continue;
                    }
                    const Vec3 contactForce =
                        impulse /
                        frame.contactImpulseTimestepSeconds;
                    force = force + contactForce;
                    torque = torque + cross(
                        vector(contact.worldPoint) -
                            worldSensor.position,
                        contactForce
                    );
                    const double forceWeight =
                        length(contactForce);
                    if (forceWeight > kTiny) {
                        centerOfPressureWorld =
                            centerOfPressureWorld +
                            vector(contact.worldPoint) *
                                forceWeight;
                        centerOfPressureForceWeight +=
                            forceWeight;
                        ++centerOfPressureContributors;
                    }
                    const double normalImpulse =
                        contact.solverImpulseAndFriction.x;
                    const double tangentialImpulse =
                        contact.solverImpulseAndFriction.y;
                    const double staticFriction =
                        contact.solverImpulseAndFriction.z;
                    double utilization = 0.0;
                    if (normalImpulse > kTiny) {
                        const double capacity =
                            staticFriction * normalImpulse;
                        utilization = capacity > kTiny
                            ? std::clamp(
                                tangentialImpulse / capacity,
                                0.0,
                                1.0
                            )
                            : (tangentialImpulse > kTiny
                                ? 1.0
                                : 0.0);
                    }
                    const double normalForceWeight =
                        normalImpulse /
                        frame.contactImpulseTimestepSeconds;
                    if (normalForceWeight > kTiny) {
                        frictionUtilizationWeighted +=
                            normalForceWeight * utilization;
                        frictionUtilizationWeight +=
                            normalForceWeight;
                    }
                    maximumFrictionUtilization = std::max(
                        maximumFrictionUtilization,
                        utilization
                    );
                    ++contactContributors;
                }
            }
            if (centerOfPressureForceWeight > 0.0) {
                centerOfPressureWorld =
                    centerOfPressureWorld /
                    centerOfPressureForceWeight;
            }
            const Vec3 centerOfPressureLocal =
                centerOfPressureForceWeight > 0.0
                ? inverseTransformPoint(
                    worldSensor,
                    centerOfPressureWorld
                )
                : Vec3{};
            const std::uint32_t summaryFlags =
                MR_TACTILE_SUMMARY_DEPTH_VALID |
                (contactsPresent
                    ? MR_TACTILE_SUMMARY_WRENCH_VALID
                    : 0u) |
                (update
                    ? MR_TACTILE_SUMMARY_UPDATED
                    : 0u) |
                (reset
                    ? MR_TACTILE_SUMMARY_RESET
                    : 0u);
            summary.netForceAndContactArea =
                packed(force, static_cast<float>(activeArea));
            summary.netTorqueAndMaximumDepth =
                packed(torque, static_cast<float>(maximumDepth));
            summary.centroidLocalAndMeanDepth =
                packed(
                    centroidLocal,
                    activeArea > 0.0
                        ? static_cast<float>(
                            areaDepth / activeArea
                        )
                        : 0.0f
                );
            summary.centroidWorldAndActiveCount =
                packed(
                    centroidWorld,
                    static_cast<float>(activeCount)
                );
            summary.centerOfPressureLocalAndForceWeight =
                packed(
                    centerOfPressureLocal,
                    static_cast<float>(
                        centerOfPressureForceWeight
                    )
                );
            summary.centerOfPressureWorldAndContactCount =
                packed(
                    centerOfPressureWorld,
                    static_cast<float>(
                        centerOfPressureContributors
                    )
                );
            summary.tangentialMotionAndFriction = {
                activeArea > 0.0
                    ? static_cast<float>(std::sqrt(
                        areaTangentialSpeedSquared / activeArea
                    ))
                    : 0.0f,
                static_cast<float>(
                    maximumTangentialDisplacement
                ),
                frictionUtilizationWeight > 0.0
                    ? static_cast<float>(
                        frictionUtilizationWeighted /
                        frictionUtilizationWeight
                    )
                    : 0.0f,
                static_cast<float>(
                    maximumFrictionUtilization
                ),
            };
            summary.statisticsAndIdentity = {
                saturatedCount,
                contactContributors,
                summaryFlags,
                objectDisagrees
                    ? MR_INVALID_INDEX
                    : unanimousObject,
            };
            candidate.statuses[summaryIndex] = {
                MR_TACTILE_SUCCESS,
                environment,
                sensorIndex,
                MR_INVALID_INDEX,
                {
                    MR_TACTILE_QUERY_CPU_REFERENCE,
                    visitedTargets,
                    0u,
                    triangleTests,
                },
            };
        }
    }
    output = std::move(candidate);
    return {};
}

std::string tactileObservationMetadataJSON(
    const CookedTactileSystem& tactile
) {
    std::ostringstream json;
    json << std::setprecision(9);
    json << "{\"schema\":\"metalrobo.tactile_observation\","
         << "\"abi_version\":" << tactile.abiVersion << ','
         << "\"fingerprint\":\"0x" << std::hex
         << std::setw(16) << std::setfill('0')
         << tactile.fingerprint << std::dec << "\","
         << "\"primary_representation\":"
            "\"metric_normal_penetration\","
         << "\"depth_unit\":\"m\","
         << "\"velocity_unit\":\"m/s\","
         << "\"force_unit\":\"N\","
         << "\"torque_unit\":\"N*m\","
         << "\"layout\":\"environment_sensor_row_major\","
         << "\"sensors\":[";
    for (std::uint32_t index = 0u;
         index < tactile.sensors.size();
         ++index) {
        if (index != 0u) {
            json << ',';
        }
        const MRTactileSensorGPU& sensor =
            tactile.sensors[index];
        json << "{\"id\":\""
             << escapedJSON(tactile.sensorIds[index])
             << "\",\"width\":" << sensor.atlasAndTargets.x
             << ",\"height\":" << sensor.atlasAndTargets.y
             << ",\"surface_kind\":"
             << sensor.scheduleAndIdentity.y
             << ",\"maximum_depth_m\":" << sensor.depth.x
             << ",\"maximum_tangential_displacement_m\":"
             << sensor.depth.w
             << ",\"active_threshold_m\":" << sensor.depth.y
             << ",\"shell_thickness_m\":" << sensor.depth.z
             << ",\"update_period_steps\":"
             << sensor.scheduleAndIdentity.x
             << ",\"parent_body\":" << sensor.topology.x
             << ",\"backing_shapes\":[";
        for (std::uint32_t local = 0u;
             local < sensor.backingRange.y;
             ++local) {
            if (local != 0u) {
                json << ',';
            }
            json << tactile.backingShapeIndices[
                sensor.backingRange.x + local
            ];
        }
        json << "]}";
    }
    json << "],\"dense_channels\":["
         << "\"penetration_depth_m\","
         << "\"validity_bits\","
         << "\"object_shape_id\","
         << "\"depth_velocity_m_per_s\","
         << "\"tangential_displacement_u_m\","
         << "\"tangential_displacement_v_m\","
         << "\"surface_velocity_u_m_per_s\","
         << "\"surface_velocity_v_m_per_s\"],"
         << "\"summary_channels\":["
         << "\"sensor_pose\","
         << "\"timestamp_s\","
         << "\"net_force_n\","
         << "\"net_torque_nm\","
         << "\"center_of_pressure_sensor_m\","
         << "\"center_of_pressure_force_weight_n\","
         << "\"geometric_contact_centroid_m\","
         << "\"contact_area_m2\","
         << "\"maximum_depth_m\","
         << "\"mean_depth_m\","
         << "\"tangential_speed_rms_m_per_s\","
         << "\"maximum_tangential_displacement_m\","
         << "\"friction_utilization_force_weighted\","
         << "\"friction_utilization_maximum\"],"
         << "\"wrench_source\":"
            "\"physics_solver_impulse_over_explicit_interval\","
         << "\"center_of_pressure_definition\":"
            "\"solver_force_magnitude_weighted_contact_position\","
         << "\"tangential_displacement_definition\":"
            "\"bounded_kinematic_target_anchor_motion_proxy\","
         << "\"friction_utilization_definition\":"
            "\"tangential_impulse_over_static_friction_capacity\","
         << "\"simulation_translator_required\":false,"
         << "\"real_sensor_input\":"
            "\"camera_to_metric_depth_translator\"}";
    return json.str();
}

const char*
tactileCookStatusName(const TactileCookStatus status) noexcept {
    switch (status) {
    case TactileCookStatus::success:
        return "success";
    case TactileCookStatus::invalidSpecification:
        return "invalid_specification";
    case TactileCookStatus::invalidBackingShape:
        return "invalid_backing_shape";
    case TactileCookStatus::invalidSampleAtlas:
        return "invalid_sample_atlas";
    case TactileCookStatus::unsupportedGeometry:
        return "unsupported_geometry";
    case TactileCookStatus::capacityOverflow:
        return "capacity_overflow";
    }
    return "unknown";
}

} // namespace metalrobo
