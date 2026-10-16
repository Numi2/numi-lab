#include "metalrobo/Collision.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace metalrobo {

struct CollisionCacheAccess {
    static const std::vector<PersistentManifold>& read(
        const PersistentManifoldCache& cache
    ) {
        return cache.entries_;
    }

    static void commit(
        PersistentManifoldCache& cache,
        std::vector<PersistentManifold> entries
    ) {
        cache.entries_ = std::move(entries);
    }
};

void PersistentManifoldCache::clear() noexcept {
    entries_.clear();
}

std::span<const PersistentManifold>
PersistentManifoldCache::entries() const noexcept {
    return entries_;
}

std::size_t PersistentManifoldCache::size() const noexcept {
    return entries_.size();
}

namespace {

constexpr double kTiny = 1.0e-14;

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
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

Vec3 operator*(const Vec3 value, const double scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

Vec3 operator*(const double scale, const Vec3 value) {
    return value * scale;
}

Vec3 operator/(const Vec3 value, const double scale) {
    return value * (1.0 / scale);
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
    if (!(magnitude > kTiny)) {
        return {};
    }
    return value / magnitude;
}

Vec3 absolute(const Vec3 value) {
    return {
        std::abs(value.x),
        std::abs(value.y),
        std::abs(value.z),
    };
}

double component(const Vec3 value, const std::uint32_t axis) {
    if (axis == 0u) {
        return value.x;
    }
    return axis == 1u ? value.y : value.z;
}

void setComponent(
    Vec3& value,
    const std::uint32_t axis,
    const double componentValue
) {
    if (axis == 0u) {
        value.x = componentValue;
    } else if (axis == 1u) {
        value.y = componentValue;
    } else {
        value.z = componentValue;
    }
}

Vec3 axisVector(const std::uint32_t axis) {
    Vec3 result{};
    setComponent(result, axis, 1.0);
    return result;
}

bool finite(const mr_float4 value) {
    return
        std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z) &&
        std::isfinite(value.w);
}

bool canonicalFloat(const float value) {
    if (!std::isfinite(value)) {
        return false;
    }
    const std::uint32_t bits =
        std::bit_cast<std::uint32_t>(value);
    const std::uint32_t exponent = bits & 0x7f800000u;
    const std::uint32_t mantissa = bits & 0x007fffffu;
    return exponent != 0u || mantissa == 0u;
}

bool canonicalFloat4(const mr_float4 value) {
    return
        canonicalFloat(value.x) &&
        canonicalFloat(value.y) &&
        canonicalFloat(value.z) &&
        canonicalFloat(value.w);
}

bool collisionDomainXyz(const mr_float4 value) {
    return
        canonicalFloat4(value) &&
        std::abs(value.x) <= MR_MAX_COLLISION_COORDINATE &&
        std::abs(value.y) <= MR_MAX_COLLISION_COORDINATE &&
        std::abs(value.z) <= MR_MAX_COLLISION_COORDINATE;
}

bool collisionInputDomainXyz(const mr_float4 value) {
    return
        canonicalFloat4(value) &&
        std::abs(value.x) <=
            MR_MAX_COLLISION_INPUT_COORDINATE &&
        std::abs(value.y) <=
            MR_MAX_COLLISION_INPUT_COORDINATE &&
        std::abs(value.z) <=
            MR_MAX_COLLISION_INPUT_COORDINATE;
}

bool collisionDomain(const double value) {
    return
        std::isfinite(value) &&
        std::abs(value) <=
            static_cast<double>(MR_MAX_COLLISION_COORDINATE);
}

bool collisionDomain(const Vec3 value) {
    return
        collisionDomain(value.x) &&
        collisionDomain(value.y) &&
        collisionDomain(value.z);
}

Vec3 xyz(const mr_float4 value) {
    return {value.x, value.y, value.z};
}

struct Quaternion {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 1.0;
};

std::optional<mr_float4> checkedQuaternionFloat(
    const mr_float4 value
) {
    if (!canonicalFloat4(value)) {
        return std::nullopt;
    }
    const float maximumComponent = std::max({
        std::abs(value.x),
        std::abs(value.y),
        std::abs(value.z),
        std::abs(value.w),
    });
    if (maximumComponent <
            MR_MIN_QUATERNION_MAX_COMPONENT ||
        maximumComponent >
            MR_MAX_QUATERNION_MAX_COMPONENT) {
        return std::nullopt;
    }
    const float squared =
        value.x * value.x +
        value.y * value.y +
        value.z * value.z +
        value.w * value.w;
    const float inverseNorm = 1.0f / std::sqrt(squared);
    const mr_float4 result{
        value.x * inverseNorm,
        value.y * inverseNorm,
        value.z * inverseNorm,
        value.w * inverseNorm,
    };
    return finite(result)
        ? std::optional<mr_float4>{result}
        : std::nullopt;
}

Quaternion quaternionFromCanonicalFloat(
    const mr_float4 value
) {
    return {
        static_cast<double>(value.x),
        static_cast<double>(value.y),
        static_cast<double>(value.z),
        static_cast<double>(value.w),
    };
}

mr_float4 multiplyFloatQuaternion(
    const mr_float4 left,
    const mr_float4 right
) {
    return {
        left.w * right.x + right.w * left.x +
            left.y * right.z - left.z * right.y,
        left.w * right.y + right.w * left.y +
            left.z * right.x - left.x * right.z,
        left.w * right.z + right.w * left.z +
            left.x * right.y - left.y * right.x,
        left.w * right.w -
            left.x * right.x -
            left.y * right.y -
            left.z * right.z,
    };
}

mr_float4 rotateFloatVector(
    const mr_float4 quaternion,
    const mr_float4 value
) {
    const float twiceCrossX =
        2.0f *
        (quaternion.y * value.z - quaternion.z * value.y);
    const float twiceCrossY =
        2.0f *
        (quaternion.z * value.x - quaternion.x * value.z);
    const float twiceCrossZ =
        2.0f *
        (quaternion.x * value.y - quaternion.y * value.x);
    return {
        value.x + quaternion.w * twiceCrossX +
            quaternion.y * twiceCrossZ -
            quaternion.z * twiceCrossY,
        value.y + quaternion.w * twiceCrossY +
            quaternion.z * twiceCrossX -
            quaternion.x * twiceCrossZ,
        value.z + quaternion.w * twiceCrossZ +
            quaternion.x * twiceCrossY -
            quaternion.y * twiceCrossX,
        0.0f,
    };
}

Quaternion conjugate(const Quaternion value) {
    return {-value.x, -value.y, -value.z, value.w};
}

Quaternion multiply(const Quaternion left, const Quaternion right) {
    return {
        left.w * right.x + right.w * left.x +
            left.y * right.z - left.z * right.y,
        left.w * right.y + right.w * left.y +
            left.z * right.x - left.x * right.z,
        left.w * right.z + right.w * left.z +
            left.x * right.y - left.y * right.x,
        left.w * right.w -
            left.x * right.x -
            left.y * right.y -
            left.z * right.z,
    };
}

Vec3 rotate(const Quaternion q, const Vec3 value) {
    const Vec3 vector{q.x, q.y, q.z};
    const Vec3 twiceCross = 2.0 * cross(vector, value);
    return value + q.w * twiceCross + cross(vector, twiceCross);
}

Vec3 inverseRotate(const Quaternion q, const Vec3 value) {
    return rotate(conjugate(q), value);
}

mr_float4 pointFloat4(const Vec3 value) {
    return {
        static_cast<float>(value.x),
        static_cast<float>(value.y),
        static_cast<float>(value.z),
        1.0f,
    };
}

mr_float4 vectorFloat4(const Vec3 value, const float w = 0.0f) {
    return {
        static_cast<float>(value.x),
        static_cast<float>(value.y),
        static_cast<float>(value.z),
        w,
    };
}

float outwardLower(const double value) {
    float result = static_cast<float>(value);
    if (!std::isfinite(result)) {
        return result;
    }
    if (static_cast<double>(result) > value) {
        result = std::nextafter(
            result,
            -std::numeric_limits<float>::infinity()
        );
    }
    return result;
}

float outwardUpper(const double value) {
    float result = static_cast<float>(value);
    if (!std::isfinite(result)) {
        return result;
    }
    if (static_cast<double>(result) < value) {
        result = std::nextafter(
            result,
            std::numeric_limits<float>::infinity()
        );
    }
    return result;
}

MRAabbGPU makeAabb(const Vec3 lower, const Vec3 upper) {
    const Vec3 padding{
        (std::max(std::abs(lower.x), std::abs(upper.x)) + 1.0) *
            static_cast<double>(
                MR_COLLISION_AABB_RELATIVE_PAD
            ),
        (std::max(std::abs(lower.y), std::abs(upper.y)) + 1.0) *
            static_cast<double>(
                MR_COLLISION_AABB_RELATIVE_PAD
            ),
        (std::max(std::abs(lower.z), std::abs(upper.z)) + 1.0) *
            static_cast<double>(
                MR_COLLISION_AABB_RELATIVE_PAD
            ),
    };
    return {
        {
            outwardLower(lower.x - padding.x),
            outwardLower(lower.y - padding.y),
            outwardLower(lower.z - padding.z),
            0.0f,
        },
        {
            outwardUpper(upper.x + padding.x),
            outwardUpper(upper.y + padding.y),
            outwardUpper(upper.z + padding.z),
            0.0f,
        },
    };
}

MRAabbGPU planeAabb() {
    constexpr float maximum = std::numeric_limits<float>::max();
    return {
        {-maximum, -maximum, -maximum, 0.0f},
        {maximum, maximum, maximum, 0.0f},
    };
}

bool finiteAabb(const MRAabbGPU& aabb) {
    return
        collisionDomainXyz(aabb.lower) &&
        collisionDomainXyz(aabb.upper) &&
        aabb.lower.x <= aabb.upper.x &&
        aabb.lower.y <= aabb.upper.y &&
        aabb.lower.z <= aabb.upper.z;
}

double aabbLower(
    const MRAabbGPU& aabb,
    const std::uint32_t axis
) {
    if (axis == 0u) {
        return aabb.lower.x;
    }
    return axis == 1u ? aabb.lower.y : aabb.lower.z;
}

double aabbUpper(
    const MRAabbGPU& aabb,
    const std::uint32_t axis
) {
    if (axis == 0u) {
        return aabb.upper.x;
    }
    return axis == 1u ? aabb.upper.y : aabb.upper.z;
}

bool aabbOverlap(const MRAabbGPU& a, const MRAabbGPU& b) {
    return
        a.lower.x <= b.upper.x && a.upper.x >= b.lower.x &&
        a.lower.y <= b.upper.y && a.upper.y >= b.lower.y &&
        a.lower.z <= b.upper.z && a.upper.z >= b.lower.z;
}

struct WorldShape {
    std::uint32_t index = 0;
    std::uint32_t type = MR_SHAPE_SPHERE;
    std::uint32_t body = 0;
    std::uint32_t generation = 0;
    bool disabled = false;
    Vec3 center{};
    Quaternion rotation{};
    Vec3 halfExtents{};
    Vec3 capsuleEndpoint0{};
    Vec3 capsuleEndpoint1{};
    Vec3 cylinderAxis{};
    Vec3 cylinderBasisX{};
    Vec3 cylinderBasisZ{};
    Vec3 planeNormal{};
    double radius = 0.0;
    double halfLength = 0.0;
    double contactOffset = 0.0;
    double restOffset = 0.0;
    MRAabbGPU aabb{};
};

bool supportedShapeType(const std::uint32_t type) {
    return
        type == MR_SHAPE_SPHERE ||
        type == MR_SHAPE_CAPSULE ||
        type == MR_SHAPE_BOX ||
        type == MR_SHAPE_CYLINDER ||
        type == MR_SHAPE_PLANE;
}

bool validShapeRecord(
    const MRShapeGPU& shape,
    std::span<const MRBodyStateGPU> bodies
) {
    if (shape.bodyIndex >= bodies.size()) {
        return false;
    }
    constexpr std::uint32_t knownShapeFlags =
        MR_SHAPE_FLAG_SIMULATION_DISABLED |
        MR_SHAPE_FLAG_ENABLE_CCD |
        MR_SHAPE_FLAG_MESH_TWO_SIDED;
    if ((shape.flags & ~knownShapeFlags) != 0u ||
        !collisionInputDomainXyz(shape.localPosition) ||
        !finite(shape.localRotation) ||
        !collisionInputDomainXyz(shape.dimensions) ||
        !collisionInputDomainXyz(
            shape.contactRestAndBoundingRadius
        ) ||
        shape.contactRestAndBoundingRadius.x < 0.0f ||
        shape.contactRestAndBoundingRadius.x <
            shape.contactRestAndBoundingRadius.y ||
        shape.contactRestAndBoundingRadius.z < 0.0f) {
        return false;
    }
    const auto bodyRotation =
        checkedQuaternionFloat(
            bodies[shape.bodyIndex].orientation
        );
    const auto localRotation =
        checkedQuaternionFloat(shape.localRotation);
    if (!bodyRotation.has_value() ||
        !localRotation.has_value()) {
        return false;
    }
    const mr_float4 worldRotation =
        multiplyFloatQuaternion(*bodyRotation, *localRotation);
    const mr_float4 rotatedLocal =
        rotateFloatVector(*bodyRotation, shape.localPosition);
    const mr_float4 worldCenter{
        bodies[shape.bodyIndex].position.x + rotatedLocal.x,
        bodies[shape.bodyIndex].position.y + rotatedLocal.y,
        bodies[shape.bodyIndex].position.z + rotatedLocal.z,
        1.0f,
    };
    return
        finite(worldRotation) &&
        collisionDomainXyz(worldCenter);
}

std::optional<WorldShape> makeWorldShape(
    const std::uint32_t index,
    const MRShapeGPU& shape,
    std::span<const MRBodyStateGPU> bodies
) {
    if (!validShapeRecord(shape, bodies)) {
        return std::nullopt;
    }

    const MRBodyStateGPU& body = bodies[shape.bodyIndex];
    if (!finite(body.position) ||
        !finite(body.orientation) ||
        body.flagsAndIndices[0] > MR_MOTION_DYNAMIC) {
        return std::nullopt;
    }
    const auto bodyRotationFloat =
        checkedQuaternionFloat(body.orientation);
    const auto localRotationFloat =
        checkedQuaternionFloat(shape.localRotation);
    if (!bodyRotationFloat.has_value() ||
        !localRotationFloat.has_value()) {
        return std::nullopt;
    }
    const Quaternion bodyRotation =
        quaternionFromCanonicalFloat(*bodyRotationFloat);
    const Quaternion localRotation =
        quaternionFromCanonicalFloat(*localRotationFloat);
    const mr_float4 worldRotationFloat =
        multiplyFloatQuaternion(
            *bodyRotationFloat,
            *localRotationFloat
        );
    const mr_float4 rotatedLocalFloat =
        rotateFloatVector(
            *bodyRotationFloat,
            shape.localPosition
        );
    const mr_float4 worldCenterFloat{
        body.position.x + rotatedLocalFloat.x,
        body.position.y + rotatedLocalFloat.y,
        body.position.z + rotatedLocalFloat.z,
        0.0f,
    };
    if (!canonicalFloat4(worldRotationFloat) ||
        !collisionDomainXyz(worldCenterFloat)) {
        return std::nullopt;
    }

    WorldShape result;
    result.index = index;
    result.type = shape.shapeType;
    result.body = shape.bodyIndex;
    result.generation = shape.slotGeneration;
    result.disabled =
        (shape.flags & MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u;
    result.rotation = multiply(bodyRotation, localRotation);
    result.center =
        xyz(body.position) +
        rotate(bodyRotation, xyz(shape.localPosition));
    result.contactOffset = shape.contactRestAndBoundingRadius.x;
    result.restOffset = shape.contactRestAndBoundingRadius.y;

    if (result.disabled) {
        result.aabb = {};
        return result;
    }
    if (!supportedShapeType(shape.shapeType)) {
        return std::nullopt;
    }

    Vec3 lower{};
    Vec3 upper{};
    if (shape.shapeType == MR_SHAPE_SPHERE) {
        result.radius = shape.dimensions.x;
        if (result.radius <
            static_cast<double>(MR_MIN_COLLISION_EXTENT)) {
            return std::nullopt;
        }
        const Vec3 extent{
            result.radius + result.contactOffset,
            result.radius + result.contactOffset,
            result.radius + result.contactOffset,
        };
        lower = result.center - extent;
        upper = result.center + extent;
    } else if (shape.shapeType == MR_SHAPE_CAPSULE) {
        result.radius = shape.dimensions.x;
        result.halfLength = shape.dimensions.y;
        if (result.radius <
                static_cast<double>(MR_MIN_COLLISION_EXTENT) ||
            result.halfLength <
                static_cast<double>(MR_MIN_COLLISION_EXTENT)) {
            return std::nullopt;
        }
        const Vec3 axis = rotate(
            result.rotation,
            {0.0, result.halfLength, 0.0}
        );
        result.capsuleEndpoint0 = result.center - axis;
        result.capsuleEndpoint1 = result.center + axis;
        const double expansion =
            result.radius + result.contactOffset;
        const Vec3 extent{expansion, expansion, expansion};
        lower = {
            std::min(
                result.capsuleEndpoint0.x,
                result.capsuleEndpoint1.x
            ),
            std::min(
                result.capsuleEndpoint0.y,
                result.capsuleEndpoint1.y
            ),
            std::min(
                result.capsuleEndpoint0.z,
                result.capsuleEndpoint1.z
            ),
        };
        upper = {
            std::max(
                result.capsuleEndpoint0.x,
                result.capsuleEndpoint1.x
            ),
            std::max(
                result.capsuleEndpoint0.y,
                result.capsuleEndpoint1.y
            ),
            std::max(
                result.capsuleEndpoint0.z,
                result.capsuleEndpoint1.z
            ),
        };
        lower = lower - extent;
        upper = upper + extent;
    } else if (shape.shapeType == MR_SHAPE_BOX) {
        result.halfExtents = xyz(shape.dimensions);
        if (result.halfExtents.x <
                static_cast<double>(MR_MIN_COLLISION_EXTENT) ||
            result.halfExtents.y <
                static_cast<double>(MR_MIN_COLLISION_EXTENT) ||
            result.halfExtents.z <
                static_cast<double>(MR_MIN_COLLISION_EXTENT)) {
            return std::nullopt;
        }
        const Vec3 basisX =
            rotate(result.rotation, {1.0, 0.0, 0.0});
        const Vec3 basisY =
            rotate(result.rotation, {0.0, 1.0, 0.0});
        const Vec3 basisZ =
            rotate(result.rotation, {0.0, 0.0, 1.0});
        const Vec3 extent =
            absolute(basisX) * result.halfExtents.x +
            absolute(basisY) * result.halfExtents.y +
            absolute(basisZ) * result.halfExtents.z +
            Vec3{
                result.contactOffset,
                result.contactOffset,
                result.contactOffset,
            };
        lower = result.center - extent;
        upper = result.center + extent;
    } else if (shape.shapeType == MR_SHAPE_CYLINDER) {
        result.radius = shape.dimensions.x;
        result.halfLength = shape.dimensions.y;
        if (result.radius <
                static_cast<double>(MR_MIN_COLLISION_EXTENT) ||
            result.halfLength <
                static_cast<double>(MR_MIN_COLLISION_EXTENT)) {
            return std::nullopt;
        }

        result.cylinderAxis = normalized(
            rotate(result.rotation, {0.0, 1.0, 0.0})
        );
        Vec3 basisX =
            rotate(result.rotation, {1.0, 0.0, 0.0});
        basisX = basisX -
            result.cylinderAxis *
                dot(result.cylinderAxis, basisX);
        result.cylinderBasisX = normalized(basisX);
        result.cylinderBasisZ = normalized(
            cross(result.cylinderBasisX, result.cylinderAxis)
        );
        if (lengthSquared(result.cylinderAxis) <= kTiny ||
            lengthSquared(result.cylinderBasisX) <= kTiny ||
            lengthSquared(result.cylinderBasisZ) <= kTiny) {
            return std::nullopt;
        }

        const auto componentExtent =
            [&](const double axisComponent) {
                return
                    std::abs(axisComponent) *
                        result.halfLength +
                    std::sqrt(std::max(
                        0.0,
                        1.0 -
                            axisComponent * axisComponent
                    )) *
                        result.radius +
                    result.contactOffset;
            };
        const Vec3 extent{
            componentExtent(result.cylinderAxis.x),
            componentExtent(result.cylinderAxis.y),
            componentExtent(result.cylinderAxis.z),
        };
        lower = result.center - extent;
        upper = result.center + extent;
    } else {
        result.planeNormal = normalized(
            rotate(result.rotation, {0.0, 1.0, 0.0})
        );
        if (lengthSquared(result.planeNormal) <= kTiny) {
            return std::nullopt;
        }
        result.aabb = planeAabb();
        return result;
    }

    if (!collisionDomain(lower) ||
        !collisionDomain(upper)) {
        return std::nullopt;
    }
    result.aabb = makeAabb(lower, upper);
    if (!finiteAabb(result.aabb)) {
        return std::nullopt;
    }
    return result;
}

std::optional<std::uint32_t> pairClass(
    const std::uint32_t typeA,
    const std::uint32_t typeB
) {
    if (typeA == MR_SHAPE_SPHERE &&
        typeB == MR_SHAPE_SPHERE) {
        return collisionPairSphereSphere;
    }
    if ((typeA == MR_SHAPE_SPHERE &&
         typeB == MR_SHAPE_PLANE) ||
        (typeA == MR_SHAPE_PLANE &&
         typeB == MR_SHAPE_SPHERE)) {
        return collisionPairSpherePlane;
    }
    if ((typeA == MR_SHAPE_CAPSULE &&
         typeB == MR_SHAPE_PLANE) ||
        (typeA == MR_SHAPE_PLANE &&
         typeB == MR_SHAPE_CAPSULE)) {
        return collisionPairCapsulePlane;
    }
    if ((typeA == MR_SHAPE_BOX &&
         typeB == MR_SHAPE_PLANE) ||
        (typeA == MR_SHAPE_PLANE &&
         typeB == MR_SHAPE_BOX)) {
        return collisionPairBoxPlane;
    }
    if ((typeA == MR_SHAPE_CYLINDER &&
         typeB == MR_SHAPE_PLANE) ||
        (typeA == MR_SHAPE_PLANE &&
         typeB == MR_SHAPE_CYLINDER)) {
        return collisionPairCylinderPlane;
    }
    if ((typeA == MR_SHAPE_SPHERE &&
         typeB == MR_SHAPE_CAPSULE) ||
        (typeA == MR_SHAPE_CAPSULE &&
         typeB == MR_SHAPE_SPHERE)) {
        return collisionPairSphereCapsule;
    }
    if (typeA == MR_SHAPE_CAPSULE &&
        typeB == MR_SHAPE_CAPSULE) {
        return collisionPairCapsuleCapsule;
    }
    if ((typeA == MR_SHAPE_SPHERE &&
         typeB == MR_SHAPE_BOX) ||
        (typeA == MR_SHAPE_BOX &&
         typeB == MR_SHAPE_SPHERE)) {
        return collisionPairSphereBox;
    }
    if ((typeA == MR_SHAPE_CAPSULE &&
         typeB == MR_SHAPE_BOX) ||
        (typeA == MR_SHAPE_BOX &&
         typeB == MR_SHAPE_CAPSULE)) {
        return collisionPairCapsuleBox;
    }
    if (typeA == MR_SHAPE_BOX &&
        typeB == MR_SHAPE_BOX) {
        return collisionPairBoxBox;
    }
    return std::nullopt;
}

std::uint64_t pairKey(
    const std::uint32_t shapeA,
    const std::uint32_t shapeB
) {
    const std::uint32_t low = std::min(shapeA, shapeB);
    const std::uint32_t high = std::max(shapeA, shapeB);
    return
        (static_cast<std::uint64_t>(low) << 32u) |
        static_cast<std::uint64_t>(high);
}

std::vector<std::uint64_t> canonicalExclusions(
    std::span<const CollisionPairExclusion> exclusions,
    const std::size_t shapeCount
) {
    std::vector<std::uint64_t> result;
    result.reserve(exclusions.size());
    for (const CollisionPairExclusion exclusion : exclusions) {
        if (exclusion.colliderA == exclusion.colliderB ||
            exclusion.colliderA >= shapeCount ||
            exclusion.colliderB >= shapeCount) {
            continue;
        }
        result.push_back(pairKey(
            exclusion.colliderA,
            exclusion.colliderB
        ));
    }
    std::ranges::sort(result);
    result.erase(
        std::unique(result.begin(), result.end()),
        result.end()
    );
    return result;
}

bool pairPassesFilter(
    const std::uint32_t shapeIndexA,
    const std::uint32_t shapeIndexB,
    std::span<const MRShapeGPU> shapes,
    std::span<const MRBodyStateGPU> bodies,
    std::span<const std::uint64_t> exclusions
) {
    if (shapeIndexA == shapeIndexB) {
        return false;
    }
    const MRShapeGPU& shapeA = shapes[shapeIndexA];
    const MRShapeGPU& shapeB = shapes[shapeIndexB];
    if ((shapeA.flags & MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u ||
        (shapeB.flags & MR_SHAPE_FLAG_SIMULATION_DISABLED) != 0u ||
        shapeA.bodyIndex == shapeB.bodyIndex ||
        (shapeA.collisionGroup & shapeB.collisionMask) == 0u ||
        (shapeB.collisionGroup & shapeA.collisionMask) == 0u ||
        !pairClass(shapeA.shapeType, shapeB.shapeType).has_value()) {
        return false;
    }
    const std::uint32_t motionA =
        bodies[shapeA.bodyIndex].flagsAndIndices[0];
    const std::uint32_t motionB =
        bodies[shapeB.bodyIndex].flagsAndIndices[0];
    if (motionA != MR_MOTION_DYNAMIC &&
        motionB != MR_MOTION_DYNAMIC) {
        return false;
    }
    return !std::ranges::binary_search(
        exclusions,
        pairKey(shapeIndexA, shapeIndexB)
    );
}

bool planeMayOverlap(
    const WorldShape& plane,
    const WorldShape& finiteShape
) {
    const MRAabbGPU& aabb = finiteShape.aabb;
    const Vec3 center{
        0.5 * (
            static_cast<double>(aabb.lower.x) +
            aabb.upper.x
        ),
        0.5 * (
            static_cast<double>(aabb.lower.y) +
            aabb.upper.y
        ),
        0.5 * (
            static_cast<double>(aabb.lower.z) +
            aabb.upper.z
        ),
    };
    const Vec3 half{
        0.5 * (
            static_cast<double>(aabb.upper.x) -
            aabb.lower.x
        ),
        0.5 * (
            static_cast<double>(aabb.upper.y) -
            aabb.lower.y
        ),
        0.5 * (
            static_cast<double>(aabb.upper.z) -
            aabb.lower.z
        ),
    };
    const double minimumSignedDistance =
        dot(plane.planeNormal, center - plane.center) -
        dot(absolute(plane.planeNormal), half);
    return minimumSignedDistance <= plane.contactOffset;
}

std::uint32_t broadphaseAxis(
    std::span<const WorldShape> worldShapes
) {
    Vec3 lower{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    Vec3 upper{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    bool foundFiniteShape = false;
    for (const WorldShape& shape : worldShapes) {
        if (shape.disabled || shape.type == MR_SHAPE_PLANE) {
            continue;
        }
        foundFiniteShape = true;
        lower.x = std::min(
            lower.x,
            static_cast<double>(shape.aabb.lower.x)
        );
        lower.y = std::min(
            lower.y,
            static_cast<double>(shape.aabb.lower.y)
        );
        lower.z = std::min(
            lower.z,
            static_cast<double>(shape.aabb.lower.z)
        );
        upper.x = std::max(
            upper.x,
            static_cast<double>(shape.aabb.upper.x)
        );
        upper.y = std::max(
            upper.y,
            static_cast<double>(shape.aabb.upper.y)
        );
        upper.z = std::max(
            upper.z,
            static_cast<double>(shape.aabb.upper.z)
        );
    }
    if (!foundFiniteShape) {
        return 0u;
    }
    const Vec3 extent = upper - lower;
    std::uint32_t axis = 0u;
    if (extent.y > extent.x) {
        axis = 1u;
    }
    if (extent.z > component(extent, axis)) {
        axis = 2u;
    }
    return axis;
}

template <typename Callback>
void forEachBroadphasePair(
    std::span<const MRShapeGPU> shapes,
    std::span<const MRBodyStateGPU> bodies,
    std::span<const WorldShape> worldShapes,
    std::span<const std::uint64_t> exclusions,
    const std::uint32_t axis,
    Callback&& callback
) {
    struct Proxy {
        std::uint32_t shape = 0;
        double lower = 0.0;
        double upper = 0.0;
    };

    std::vector<Proxy> proxies;
    proxies.reserve(worldShapes.size());
    for (const WorldShape& shape : worldShapes) {
        if (shape.disabled || shape.type == MR_SHAPE_PLANE) {
            continue;
        }
        proxies.push_back({
            shape.index,
            aabbLower(shape.aabb, axis),
            aabbUpper(shape.aabb, axis),
        });
    }
    std::ranges::sort(
        proxies,
        [](const Proxy& left, const Proxy& right) {
            return std::tie(left.lower, left.upper, left.shape) <
                std::tie(right.lower, right.upper, right.shape);
        }
    );

    std::vector<Proxy> active;
    active.reserve(proxies.size());
    for (const Proxy proxy : proxies) {
        active.erase(
            std::remove_if(
                active.begin(),
                active.end(),
                [proxy](const Proxy candidate) {
                    return candidate.upper < proxy.lower;
                }
            ),
            active.end()
        );

        for (const Proxy candidate : active) {
            const std::uint32_t colliderA =
                std::min(candidate.shape, proxy.shape);
            const std::uint32_t colliderB =
                std::max(candidate.shape, proxy.shape);
            if (!aabbOverlap(
                    worldShapes[colliderA].aabb,
                    worldShapes[colliderB].aabb
                ) ||
                !pairPassesFilter(
                    colliderA,
                    colliderB,
                    shapes,
                    bodies,
                    exclusions
                )) {
                continue;
            }
            callback(
                colliderA,
                colliderB,
                *pairClass(
                    shapes[colliderA].shapeType,
                    shapes[colliderB].shapeType
                )
            );
        }
        active.push_back(proxy);
    }

    for (const WorldShape& plane : worldShapes) {
        if (plane.disabled || plane.type != MR_SHAPE_PLANE) {
            continue;
        }
        for (const WorldShape& finiteShape : worldShapes) {
            if (finiteShape.disabled ||
                finiteShape.type == MR_SHAPE_PLANE ||
                !planeMayOverlap(plane, finiteShape)) {
                continue;
            }
            const std::uint32_t colliderA =
                std::min(plane.index, finiteShape.index);
            const std::uint32_t colliderB =
                std::max(plane.index, finiteShape.index);
            if (!pairPassesFilter(
                    colliderA,
                    colliderB,
                    shapes,
                    bodies,
                    exclusions
                )) {
                continue;
            }
            callback(
                colliderA,
                colliderB,
                *pairClass(
                    shapes[colliderA].shapeType,
                    shapes[colliderB].shapeType
                )
            );
        }
    }
}

std::uint32_t featureKey(
    const std::uint32_t shapeType,
    const std::uint32_t localFeature
) {
    return
        ((shapeType & 0x0fu) << 28u) |
        (localFeature & 0x0fffffffu);
}

MRRawContactGPU makeRawContact(
    const Vec3 normal,
    const double separation,
    const Vec3 pointA,
    const Vec3 pointB,
    const std::uint32_t featureA,
    const std::uint32_t featureB
) {
    MRRawContactGPU result{};
    result.normalAndSeparation = vectorFloat4(
        normal,
        static_cast<float>(separation)
    );
    result.pointAWorld = pointFloat4(pointA);
    result.pointBWorld = pointFloat4(pointB);
    result.featureAndFlags[0] = featureA;
    result.featureAndFlags[1] = featureB;
    result.featureAndFlags[2] = 0u;
    result.featureAndFlags[3] = 0u;
    return result;
}

MRRawContactGPU swappedContact(const MRRawContactGPU& input) {
    MRRawContactGPU result = input;
    result.normalAndSeparation.x = -input.normalAndSeparation.x;
    result.normalAndSeparation.y = -input.normalAndSeparation.y;
    result.normalAndSeparation.z = -input.normalAndSeparation.z;
    result.pointAWorld = input.pointBWorld;
    result.pointBWorld = input.pointAWorld;
    result.featureAndFlags[0] = input.featureAndFlags[1];
    result.featureAndFlags[1] = input.featureAndFlags[0];
    return result;
}

Vec3 coincidentNormal(
    const std::uint32_t colliderA,
    const std::uint32_t colliderB
) {
    const std::uint32_t hash =
        colliderA * 73856093u ^ colliderB * 19349663u;
    Vec3 result{};
    const std::uint32_t axis = hash % 3u;
    const double sign = (hash & 4u) == 0u ? 1.0 : -1.0;
    if (axis == 0u) {
        result.x = sign;
    } else if (axis == 1u) {
        result.y = sign;
    } else {
        result.z = sign;
    }
    return result;
}

Vec3 stableSegmentNormal(
    const Vec3 directionA,
    const Vec3 directionB,
    const std::uint32_t colliderA,
    const std::uint32_t colliderB
) {
    const Vec3 crossed = cross(directionA, directionB);
    if (lengthSquared(crossed) > kTiny) {
        return normalized(crossed);
    }

    const Vec3 direction =
        lengthSquared(directionA) > kTiny
        ? normalized(directionA)
        : (
              lengthSquared(directionB) > kTiny
              ? normalized(directionB)
              : Vec3{}
          );
    if (lengthSquared(direction) <= kTiny) {
        return coincidentNormal(colliderA, colliderB);
    }

    const Vec3 absoluteDirection = absolute(direction);
    Vec3 reference;
    if (absoluteDirection.x <= absoluteDirection.y &&
        absoluteDirection.x <= absoluteDirection.z) {
        reference = {1.0, 0.0, 0.0};
    } else if (absoluteDirection.y <= absoluteDirection.z) {
        reference = {0.0, 1.0, 0.0};
    } else {
        reference = {0.0, 0.0, 1.0};
    }
    Vec3 result = normalized(cross(direction, reference));
    if (((colliderA * 73856093u) ^
         (colliderB * 19349663u)) & 4u) {
        result = -result;
    }
    return result;
}

struct SegmentClosestPoint {
    Vec3 point{};
    double parameter = 0.0;
};

SegmentClosestPoint closestPointOnSegment(
    const Vec3 point,
    const Vec3 endpoint0,
    const Vec3 endpoint1
) {
    const Vec3 segment = endpoint1 - endpoint0;
    const double squaredLength = lengthSquared(segment);
    if (!(squaredLength > kTiny)) {
        return {endpoint0, 0.0};
    }
    const double parameter = std::clamp(
        dot(point - endpoint0, segment) / squaredLength,
        0.0,
        1.0
    );
    return {
        endpoint0 + segment * parameter,
        parameter,
    };
}

struct SegmentPairClosestPoints {
    Vec3 pointA{};
    Vec3 pointB{};
    double parameterA = 0.0;
    double parameterB = 0.0;
};

SegmentPairClosestPoints closestPointsOnSegments(
    const Vec3 endpointA0,
    const Vec3 endpointA1,
    const Vec3 endpointB0,
    const Vec3 endpointB1
) {
    const Vec3 directionA = endpointA1 - endpointA0;
    const Vec3 directionB = endpointB1 - endpointB0;
    const Vec3 offset = endpointA0 - endpointB0;
    const double squaredA = lengthSquared(directionA);
    const double squaredB = lengthSquared(directionB);
    const double projectedA = dot(directionA, offset);
    const double projectedB = dot(directionB, offset);

    double parameterA = 0.0;
    double parameterB = 0.0;
    if (!(squaredA > kTiny) && !(squaredB > kTiny)) {
        return {
            endpointA0,
            endpointB0,
            0.0,
            0.0,
        };
    }
    if (!(squaredA > kTiny)) {
        parameterB = std::clamp(
            projectedB / squaredB,
            0.0,
            1.0
        );
    } else {
        const double crossProjection =
            dot(directionA, directionB);
        if (!(squaredB > kTiny)) {
            parameterA = std::clamp(
                -projectedA / squaredA,
                0.0,
                1.0
            );
        } else {
            const double denominator =
                squaredA * squaredB -
                crossProjection * crossProjection;
            if (denominator > kTiny * squaredA * squaredB) {
                parameterA = std::clamp(
                    (
                        crossProjection * projectedB -
                        projectedA * squaredB
                    ) /
                        denominator,
                    0.0,
                    1.0
                );
            }
            parameterB =
                (
                    crossProjection * parameterA +
                    projectedB
                ) /
                squaredB;
            if (parameterB < 0.0) {
                parameterB = 0.0;
                parameterA = std::clamp(
                    -projectedA / squaredA,
                    0.0,
                    1.0
                );
            } else if (parameterB > 1.0) {
                parameterB = 1.0;
                parameterA = std::clamp(
                    (
                        crossProjection - projectedA
                    ) /
                        squaredA,
                    0.0,
                    1.0
                );
            }
        }
    }
    return {
        endpointA0 + directionA * parameterA,
        endpointB0 + directionB * parameterB,
        parameterA,
        parameterB,
    };
}

std::uint32_t capsuleFeature(const double parameter) {
    if (parameter <= 0.0) {
        return 0u;
    }
    return parameter >= 1.0 ? 1u : 2u;
}

struct SphereBoxWitness {
    Vec3 normal{};
    Vec3 boxPoint{};
    double separation = 0.0;
    std::uint32_t boxFeature = 0u;
};

struct CapsuleBoxWitness {
    Vec3 normal{};
    Vec3 capsulePoint{};
    Vec3 boxPoint{};
    double separation = 0.0;
    double capsuleParameter = 0.0;
    std::uint32_t boxFeature = 0u;
};

SphereBoxWitness sphereBoxWitness(
    const WorldShape& sphere,
    const WorldShape& box
) {
    const Vec3 localCenter = inverseRotate(
        box.rotation,
        sphere.center - box.center
    );
    Vec3 closest{
        std::clamp(
            localCenter.x,
            -box.halfExtents.x,
            box.halfExtents.x
        ),
        std::clamp(
            localCenter.y,
            -box.halfExtents.y,
            box.halfExtents.y
        ),
        std::clamp(
            localCenter.z,
            -box.halfExtents.z,
            box.halfExtents.z
        ),
    };
    const Vec3 localDelta = closest - localCenter;
    const double distance = length(localDelta);

    SphereBoxWitness result;
    if (distance > kTiny) {
        const Vec3 localNormal = localDelta / distance;
        result.normal = rotate(box.rotation, localNormal);
        result.separation = distance - sphere.radius;
        const auto region = [](const double coordinate,
                               const double halfExtent) {
            if (coordinate < -halfExtent) {
                return 0u;
            }
            return coordinate > halfExtent ? 2u : 1u;
        };
        result.boxFeature =
            region(localCenter.x, box.halfExtents.x) +
            3u * region(localCenter.y, box.halfExtents.y) +
            9u * region(localCenter.z, box.halfExtents.z);
    } else {
        const std::array distances{
            box.halfExtents.x - std::abs(localCenter.x),
            box.halfExtents.y - std::abs(localCenter.y),
            box.halfExtents.z - std::abs(localCenter.z),
        };
        std::uint32_t axis = 0u;
        if (distances[1] < distances[axis]) {
            axis = 1u;
        }
        if (distances[2] < distances[axis]) {
            axis = 2u;
        }
        const double coordinate = component(localCenter, axis);
        const double sign = coordinate >= 0.0 ? 1.0 : -1.0;
        Vec3 outward{};
        if (axis == 0u) {
            outward.x = sign;
            closest.x = sign * box.halfExtents.x;
        } else if (axis == 1u) {
            outward.y = sign;
            closest.y = sign * box.halfExtents.y;
        } else {
            outward.z = sign;
            closest.z = sign * box.halfExtents.z;
        }
        result.normal = rotate(box.rotation, -outward);
        result.separation = -distances[axis] - sphere.radius;
        result.boxFeature =
            27u + 2u * axis + (sign > 0.0 ? 1u : 0u);
    }
    result.boxPoint =
        box.center + rotate(box.rotation, closest);
    return result;
}

void considerCapsuleBoxCandidate(
    const Vec3 segmentPoint,
    const Vec3 boxPoint,
    const double parameter,
    const std::uint32_t feature,
    double& bestSquared,
    Vec3& bestSegmentPoint,
    Vec3& bestBoxPoint,
    double& bestParameter,
    std::uint32_t& bestFeature
) {
    const double squared =
        lengthSquared(boxPoint - segmentPoint);
    const double tolerance =
        1.0e-12 *
        (1.0 + std::max(bestSquared, squared));
    if (!std::isfinite(bestSquared) ||
        squared < bestSquared - tolerance ||
        (std::abs(squared - bestSquared) <= tolerance &&
         feature < bestFeature)) {
        bestSquared = squared;
        bestSegmentPoint = segmentPoint;
        bestBoxPoint = boxPoint;
        bestParameter = parameter;
        bestFeature = feature;
    }
}

CapsuleBoxWitness capsuleBoxWitness(
    const WorldShape& capsule,
    const WorldShape& box
) {
    const Vec3 segment0 = inverseRotate(
        box.rotation,
        capsule.capsuleEndpoint0 - box.center
    );
    const Vec3 segment1 = inverseRotate(
        box.rotation,
        capsule.capsuleEndpoint1 - box.center
    );
    const Vec3 direction = segment1 - segment0;

    double enter = 0.0;
    double exit = 1.0;
    bool intersects = true;
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        const double directionComponent =
            component(direction, axis);
        const double startComponent =
            component(segment0, axis);
        const double halfExtent =
            component(box.halfExtents, axis);
        if (std::abs(directionComponent) <= kTiny) {
            if (startComponent < -halfExtent ||
                startComponent > halfExtent) {
                intersects = false;
            }
            continue;
        }
        double first =
            (-halfExtent - startComponent) /
            directionComponent;
        double second =
            (halfExtent - startComponent) /
            directionComponent;
        if (first > second) {
            std::swap(first, second);
        }
        enter = std::max(enter, first);
        exit = std::min(exit, second);
        if (enter > exit) {
            intersects = false;
        }
    }

    CapsuleBoxWitness result;
    if (intersects) {
        result.capsuleParameter =
            std::clamp(0.5 * (enter + exit), 0.0, 1.0);
        const Vec3 core =
            segment0 + direction * result.capsuleParameter;
        const std::array distances{
            box.halfExtents.x - std::abs(core.x),
            box.halfExtents.y - std::abs(core.y),
            box.halfExtents.z - std::abs(core.z),
        };
        std::uint32_t axis = 0u;
        if (distances[1] < distances[axis]) {
            axis = 1u;
        }
        if (distances[2] < distances[axis]) {
            axis = 2u;
        }
        const double sign =
            component(core, axis) >= 0.0 ? 1.0 : -1.0;
        Vec3 outward{};
        setComponent(outward, axis, sign);
        Vec3 localBoxPoint = core;
        setComponent(
            localBoxPoint,
            axis,
            sign * component(box.halfExtents, axis)
        );
        const Vec3 localNormal = -outward;
        result.normal =
            rotate(box.rotation, localNormal);
        result.capsulePoint =
            box.center +
            rotate(
                box.rotation,
                core + localNormal * capsule.radius
            );
        result.boxPoint =
            box.center +
            rotate(box.rotation, localBoxPoint);
        result.separation =
            -distances[axis] - capsule.radius;
        result.boxFeature =
            27u + 2u * axis + (sign > 0.0 ? 1u : 0u);
        return result;
    }

    double bestSquared =
        std::numeric_limits<double>::infinity();
    Vec3 bestSegmentPoint{};
    Vec3 bestBoxPoint{};
    double bestParameter = 0.0;
    std::uint32_t bestFeature =
        std::numeric_limits<std::uint32_t>::max();
    for (std::uint32_t endpoint = 0u;
         endpoint < 2u;
         ++endpoint) {
        const Vec3 segmentPoint =
            endpoint == 0u ? segment0 : segment1;
        const Vec3 boxPoint{
            std::clamp(
                segmentPoint.x,
                -box.halfExtents.x,
                box.halfExtents.x
            ),
            std::clamp(
                segmentPoint.y,
                -box.halfExtents.y,
                box.halfExtents.y
            ),
            std::clamp(
                segmentPoint.z,
                -box.halfExtents.z,
                box.halfExtents.z
            ),
        };
        considerCapsuleBoxCandidate(
            segmentPoint,
            boxPoint,
            static_cast<double>(endpoint),
            endpoint,
            bestSquared,
            bestSegmentPoint,
            bestBoxPoint,
            bestParameter,
            bestFeature
        );
    }
    std::uint32_t edgeFeature = 2u;
    for (std::uint32_t varyingAxis = 0u;
         varyingAxis < 3u;
         ++varyingAxis) {
        const std::uint32_t fixedAxis0 =
            (varyingAxis + 1u) % 3u;
        const std::uint32_t fixedAxis1 =
            (varyingAxis + 2u) % 3u;
        for (std::uint32_t signs = 0u;
             signs < 4u;
             ++signs) {
            Vec3 edge0{};
            Vec3 edge1{};
            setComponent(
                edge0,
                varyingAxis,
                -component(box.halfExtents, varyingAxis)
            );
            setComponent(
                edge1,
                varyingAxis,
                component(box.halfExtents, varyingAxis)
            );
            const double sign0 =
                (signs & 1u) == 0u ? -1.0 : 1.0;
            const double sign1 =
                (signs & 2u) == 0u ? -1.0 : 1.0;
            setComponent(
                edge0,
                fixedAxis0,
                sign0 * component(box.halfExtents, fixedAxis0)
            );
            setComponent(
                edge1,
                fixedAxis0,
                component(edge0, fixedAxis0)
            );
            setComponent(
                edge0,
                fixedAxis1,
                sign1 * component(box.halfExtents, fixedAxis1)
            );
            setComponent(
                edge1,
                fixedAxis1,
                component(edge0, fixedAxis1)
            );
            const SegmentPairClosestPoints closest =
                closestPointsOnSegments(
                    segment0,
                    segment1,
                    edge0,
                    edge1
                );
            considerCapsuleBoxCandidate(
                closest.pointA,
                closest.pointB,
                closest.parameterA,
                edgeFeature,
                bestSquared,
                bestSegmentPoint,
                bestBoxPoint,
                bestParameter,
                bestFeature
            );
            ++edgeFeature;
        }
    }
    const double distance =
        std::sqrt(std::max(bestSquared, 0.0));
    const Vec3 localNormal =
        distance > kTiny
        ? (bestBoxPoint - bestSegmentPoint) / distance
        : inverseRotate(
              box.rotation,
              coincidentNormal(capsule.index, box.index)
          );
    result.normal =
        rotate(box.rotation, localNormal);
    result.capsulePoint =
        box.center +
        rotate(
            box.rotation,
            bestSegmentPoint +
                localNormal * capsule.radius
        );
    result.boxPoint =
        box.center +
        rotate(box.rotation, bestBoxPoint);
    result.separation = distance - capsule.radius;
    result.capsuleParameter = bestParameter;
    result.boxFeature = 64u + bestFeature;
    return result;
}

struct ContactBatch {
    std::array<MRRawContactGPU, 8> contacts{};
    std::uint32_t count = 0;
};

void appendFinitePlaneContact(
    ContactBatch& result,
    const MRCandidatePairGPU& pair,
    const WorldShape& plane,
    const Vec3 finiteSurfacePoint,
    const double separation,
    const std::uint32_t finiteFeature
) {
    const Vec3 planePoint =
        finiteSurfacePoint - plane.planeNormal * separation;
    MRRawContactGPU contact = makeRawContact(
        -plane.planeNormal,
        separation,
        finiteSurfacePoint,
        planePoint,
        finiteFeature,
        featureKey(MR_SHAPE_PLANE, 0u)
    );
    if (pair.colliderA == plane.index) {
        contact = swappedContact(contact);
    }
    result.contacts[result.count++] = contact;
}

Vec3 boxAxis(
    const WorldShape& box,
    const std::uint32_t axis
) {
    return rotate(box.rotation, axisVector(axis));
}

double boxProjectionRadius(
    const WorldShape& box,
    const Vec3 axis
) {
    return
        std::abs(dot(axis, boxAxis(box, 0u))) *
            box.halfExtents.x +
        std::abs(dot(axis, boxAxis(box, 1u))) *
            box.halfExtents.y +
        std::abs(dot(axis, boxAxis(box, 2u))) *
            box.halfExtents.z;
}

Vec3 boxVertex(
    const WorldShape& box,
    const std::uint32_t vertexIndex
) {
    const Vec3 local{
        (vertexIndex & 1u) == 0u
            ? -box.halfExtents.x
            : box.halfExtents.x,
        (vertexIndex & 2u) == 0u
            ? -box.halfExtents.y
            : box.halfExtents.y,
        (vertexIndex & 4u) == 0u
            ? -box.halfExtents.z
            : box.halfExtents.z,
    };
    return box.center + rotate(box.rotation, local);
}

bool pointInsideInflatedBox(
    const Vec3 point,
    const WorldShape& box,
    const double inflation
) {
    const Vec3 local = inverseRotate(
        box.rotation,
        point - box.center
    );
    return
        std::abs(local.x) <= box.halfExtents.x + inflation &&
        std::abs(local.y) <= box.halfExtents.y + inflation &&
        std::abs(local.z) <= box.halfExtents.z + inflation;
}

Vec3 boxSupport(
    const WorldShape& box,
    const Vec3 direction
) {
    Vec3 result = box.center;
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        const Vec3 basis = boxAxis(box, axis);
        result = result +
            basis *
                (dot(direction, basis) >= 0.0
                     ? component(box.halfExtents, axis)
                     : -component(box.halfExtents, axis));
    }
    return result;
}

void appendBoxBoxContact(
    ContactBatch& result,
    const Vec3 normal,
    const double separation,
    const Vec3 pointA,
    const Vec3 pointB,
    const std::uint32_t featureA,
    const std::uint32_t featureB
) {
    if (result.count >= result.contacts.size()) {
        return;
    }
    result.contacts[result.count++] = makeRawContact(
        normal,
        separation,
        pointA,
        pointB,
        featureKey(MR_SHAPE_BOX, featureA),
        featureKey(MR_SHAPE_BOX, featureB)
    );
}

ContactBatch boxBoxContacts(
    const MRCandidatePairGPU& pair,
    const WorldShape& boxA,
    const WorldShape& boxB,
    const double acceptedContactDistance
) {
    ContactBatch result;
    std::array<Vec3, 15> axes{};
    std::uint32_t axisCount = 0u;
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        axes[axisCount++] = boxAxis(boxA, axis);
    }
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        axes[axisCount++] = boxAxis(boxB, axis);
    }
    for (std::uint32_t axisA = 0u;
         axisA < 3u;
         ++axisA) {
        for (std::uint32_t axisB = 0u;
             axisB < 3u;
             ++axisB) {
            const Vec3 crossed = cross(
                boxAxis(boxA, axisA),
                boxAxis(boxB, axisB)
            );
            const double squared = lengthSquared(crossed);
            if (squared > 1.0e-24) {
                axes[axisCount++] =
                    crossed / std::sqrt(squared);
            }
        }
    }

    const Vec3 centerDelta = boxB.center - boxA.center;
    double bestSeparation =
        -std::numeric_limits<double>::infinity();
    Vec3 bestNormal =
        coincidentNormal(pair.colliderA, pair.colliderB);
    std::uint32_t bestAxis = 0u;
    for (std::uint32_t axisIndex = 0u;
         axisIndex < axisCount;
         ++axisIndex) {
        Vec3 axis = axes[axisIndex];
        const double projection = dot(centerDelta, axis);
        if (projection < 0.0 ||
            (projection == 0.0 &&
             dot(axis, bestNormal) < 0.0)) {
            axis = -axis;
        }
        const double separation =
            std::abs(projection) -
            boxProjectionRadius(boxA, axis) -
            boxProjectionRadius(boxB, axis);
        if (separation > acceptedContactDistance) {
            return result;
        }
        const double tieTolerance =
            1.0e-12 *
            (1.0 +
             std::max(
                 std::abs(bestSeparation),
                 std::abs(separation)
             ));
        if (!std::isfinite(bestSeparation) ||
            separation > bestSeparation + tieTolerance ||
            (std::abs(separation - bestSeparation) <=
                 tieTolerance &&
             axisIndex < bestAxis)) {
            bestSeparation = separation;
            bestNormal = axis;
            bestAxis = axisIndex;
        }
    }

    const double radiusA =
        boxProjectionRadius(boxA, bestNormal);
    const double radiusB =
        boxProjectionRadius(boxB, bestNormal);
    const double nearPlaneB =
        dot(boxB.center, bestNormal) - radiusB;
    const double farPlaneA =
        dot(boxA.center, bestNormal) + radiusA;
    for (std::uint32_t vertexIndex = 0u;
         vertexIndex < 8u;
         ++vertexIndex) {
        const Vec3 pointA = boxVertex(boxA, vertexIndex);
        if (!pointInsideInflatedBox(
                pointA,
                boxB,
                acceptedContactDistance
            )) {
            continue;
        }
        const double separation =
            nearPlaneB - dot(pointA, bestNormal);
        if (separation <= acceptedContactDistance) {
            appendBoxBoxContact(
                result,
                bestNormal,
                separation,
                pointA,
                pointA + bestNormal * separation,
                vertexIndex,
                128u + bestAxis
            );
        }
    }
    for (std::uint32_t vertexIndex = 0u;
         vertexIndex < 8u;
         ++vertexIndex) {
        const Vec3 pointB = boxVertex(boxB, vertexIndex);
        if (!pointInsideInflatedBox(
                pointB,
                boxA,
                acceptedContactDistance
            )) {
            continue;
        }
        const double separation =
            dot(pointB, bestNormal) - farPlaneA;
        if (separation <= acceptedContactDistance) {
            appendBoxBoxContact(
                result,
                bestNormal,
                separation,
                pointB - bestNormal * separation,
                pointB,
                128u + bestAxis,
                vertexIndex
            );
        }
    }
    if (result.count == 0u) {
        const Vec3 pointA =
            boxSupport(boxA, bestNormal);
        const Vec3 pointB =
            boxSupport(boxB, -bestNormal);
        appendBoxBoxContact(
            result,
            bestNormal,
            dot(pointB - pointA, bestNormal),
            pointA,
            pointB,
            256u + bestAxis,
            256u + bestAxis
        );
    }
    return result;
}

ContactBatch generateContacts(
    const MRCandidatePairGPU& pair,
    std::span<const WorldShape> worldShapes
) {
    // Cylinder support-feature classification is deliberately geometric,
    // rather than based on the authored quaternion. A cap face contributes a
    // four-point ring, a side face contributes its two cap-rim endpoints, and
    // a general orientation has one extremal rim point. The tolerance only
    // recognizes numerically exact face alignments; it is not a contact slop.
    constexpr double cylinderAlignmentTolerance = 1.0e-6;
    constexpr double cylinderAlignmentToleranceSquared =
        cylinderAlignmentTolerance * cylinderAlignmentTolerance;
    constexpr std::uint32_t cylinderNegativeCapRingBase = 0u;
    constexpr std::uint32_t cylinderPositiveCapRingBase = 4u;
    constexpr std::uint32_t cylinderNegativeSideRim = 8u;
    constexpr std::uint32_t cylinderPositiveSideRim = 9u;
    constexpr std::uint32_t cylinderNegativeGeneralRim = 10u;
    constexpr std::uint32_t cylinderPositiveGeneralRim = 11u;

    ContactBatch result;
    const WorldShape& shapeA = worldShapes[pair.colliderA];
    const WorldShape& shapeB = worldShapes[pair.colliderB];
    const double contactDistance =
        shapeA.contactOffset + shapeB.contactOffset;

    if (pair.flags == collisionPairSphereSphere) {
        const Vec3 delta = shapeB.center - shapeA.center;
        const double centerDistance = length(delta);
        const Vec3 normal =
            centerDistance > kTiny
            ? delta / centerDistance
            : coincidentNormal(pair.colliderA, pair.colliderB);
        const double separation =
            centerDistance - shapeA.radius - shapeB.radius;
        if (separation <= contactDistance) {
            result.contacts[0] = makeRawContact(
                normal,
                separation,
                shapeA.center + normal * shapeA.radius,
                shapeB.center - normal * shapeB.radius,
                featureKey(MR_SHAPE_SPHERE, 0u),
                featureKey(MR_SHAPE_SPHERE, 0u)
            );
            result.count = 1u;
        }
        return result;
    }

    if (pair.flags == collisionPairSphereCapsule) {
        const bool sphereIsA = shapeA.type == MR_SHAPE_SPHERE;
        const WorldShape& sphere = sphereIsA ? shapeA : shapeB;
        const WorldShape& capsule = sphereIsA ? shapeB : shapeA;
        const SegmentClosestPoint closest =
            closestPointOnSegment(
                sphere.center,
                capsule.capsuleEndpoint0,
                capsule.capsuleEndpoint1
            );
        const Vec3 delta = closest.point - sphere.center;
        const double centerDistance = length(delta);
        const Vec3 normal =
            centerDistance > kTiny
            ? delta / centerDistance
            : stableSegmentNormal(
                  capsule.capsuleEndpoint1 -
                      capsule.capsuleEndpoint0,
                  {},
                  sphere.index,
                  capsule.index
              );
        const double separation =
            centerDistance - sphere.radius - capsule.radius;
        if (separation <= contactDistance) {
            MRRawContactGPU contact = makeRawContact(
                normal,
                separation,
                sphere.center + normal * sphere.radius,
                closest.point - normal * capsule.radius,
                featureKey(MR_SHAPE_SPHERE, 0u),
                featureKey(
                    MR_SHAPE_CAPSULE,
                    capsuleFeature(closest.parameter)
                )
            );
            if (!sphereIsA) {
                contact = swappedContact(contact);
            }
            result.contacts[0] = contact;
            result.count = 1u;
        }
        return result;
    }

    if (pair.flags == collisionPairCapsuleCapsule) {
        const SegmentPairClosestPoints closest =
            closestPointsOnSegments(
                shapeA.capsuleEndpoint0,
                shapeA.capsuleEndpoint1,
                shapeB.capsuleEndpoint0,
                shapeB.capsuleEndpoint1
            );
        const Vec3 delta = closest.pointB - closest.pointA;
        const double centerDistance = length(delta);
        const Vec3 normal =
            centerDistance > kTiny
            ? delta / centerDistance
            : stableSegmentNormal(
                  shapeA.capsuleEndpoint1 -
                      shapeA.capsuleEndpoint0,
                  shapeB.capsuleEndpoint1 -
                      shapeB.capsuleEndpoint0,
                  pair.colliderA,
                  pair.colliderB
              );
        const double separation =
            centerDistance - shapeA.radius - shapeB.radius;
        if (separation <= contactDistance) {
            result.contacts[0] = makeRawContact(
                normal,
                separation,
                closest.pointA + normal * shapeA.radius,
                closest.pointB - normal * shapeB.radius,
                featureKey(
                    MR_SHAPE_CAPSULE,
                    capsuleFeature(closest.parameterA)
                ),
                featureKey(
                    MR_SHAPE_CAPSULE,
                    capsuleFeature(closest.parameterB)
                )
            );
            result.count = 1u;
        }
        return result;
    }

    if (pair.flags == collisionPairSphereBox) {
        const bool sphereIsA = shapeA.type == MR_SHAPE_SPHERE;
        const WorldShape& sphere = sphereIsA ? shapeA : shapeB;
        const WorldShape& box = sphereIsA ? shapeB : shapeA;
        const SphereBoxWitness witness =
            sphereBoxWitness(sphere, box);
        if (witness.separation <= contactDistance) {
            MRRawContactGPU contact = makeRawContact(
                witness.normal,
                witness.separation,
                sphere.center +
                    witness.normal * sphere.radius,
                witness.boxPoint,
                featureKey(MR_SHAPE_SPHERE, 0u),
                featureKey(
                    MR_SHAPE_BOX,
                    witness.boxFeature
                )
            );
            if (!sphereIsA) {
                contact = swappedContact(contact);
            }
            result.contacts[0] = contact;
            result.count = 1u;
        }
        return result;
    }

    if (pair.flags == collisionPairCapsuleBox) {
        const bool capsuleIsA =
            shapeA.type == MR_SHAPE_CAPSULE;
        const WorldShape& capsule =
            capsuleIsA ? shapeA : shapeB;
        const WorldShape& box =
            capsuleIsA ? shapeB : shapeA;
        const CapsuleBoxWitness witness =
            capsuleBoxWitness(capsule, box);
        if (witness.separation <= contactDistance) {
            MRRawContactGPU contact = makeRawContact(
                witness.normal,
                witness.separation,
                witness.capsulePoint,
                witness.boxPoint,
                featureKey(
                    MR_SHAPE_CAPSULE,
                    capsuleFeature(
                        witness.capsuleParameter
                    )
                ),
                featureKey(
                    MR_SHAPE_BOX,
                    witness.boxFeature
                )
            );
            if (!capsuleIsA) {
                contact = swappedContact(contact);
            }
            result.contacts[0] = contact;
            result.count = 1u;
        }
        return result;
    }

    if (pair.flags == collisionPairBoxBox) {
        return boxBoxContacts(
            pair,
            shapeA,
            shapeB,
            contactDistance
        );
    }

    const WorldShape& plane =
        shapeA.type == MR_SHAPE_PLANE ? shapeA : shapeB;
    const WorldShape& finiteShape =
        shapeA.type == MR_SHAPE_PLANE ? shapeB : shapeA;

    if (pair.flags == collisionPairSpherePlane) {
        const Vec3 surface =
            finiteShape.center -
            plane.planeNormal * finiteShape.radius;
        const double separation =
            dot(plane.planeNormal, surface - plane.center);
        if (separation <= contactDistance) {
            appendFinitePlaneContact(
                result,
                pair,
                plane,
                surface,
                separation,
                featureKey(MR_SHAPE_SPHERE, 0u)
            );
        }
        return result;
    }

    if (pair.flags == collisionPairCapsulePlane) {
        const std::array<Vec3, 2> endpoints{
            finiteShape.capsuleEndpoint0,
            finiteShape.capsuleEndpoint1,
        };
        for (std::uint32_t endpoint = 0u;
             endpoint < endpoints.size();
             ++endpoint) {
            const Vec3 surface =
                endpoints[endpoint] -
                plane.planeNormal * finiteShape.radius;
            const double separation =
                dot(plane.planeNormal, surface - plane.center);
            if (separation <= contactDistance) {
                appendFinitePlaneContact(
                    result,
                    pair,
                    plane,
                    surface,
                    separation,
                    featureKey(MR_SHAPE_CAPSULE, endpoint)
                );
            }
        }
        return result;
    }

    if (pair.flags == collisionPairCylinderPlane) {
        const Vec3 axis = finiteShape.cylinderAxis;
        const double axialProjection =
            dot(plane.planeNormal, axis);
        // Project through the authored orthonormal disk basis. The
        // algebraically equivalent `n - axis * dot(n, axis)` loses its
        // second-order axial component when n and axis are nearly parallel.
        // Basis coefficients retain the first-order tilt, guarantee that the
        // witness lies in the represented disk plane, and avoid subtracting
        // values near one.
        const double radialX = dot(
            plane.planeNormal,
            finiteShape.cylinderBasisX
        );
        const double radialZ = dot(
            plane.planeNormal,
            finiteShape.cylinderBasisZ
        );
        const Vec3 radialProjection =
            finiteShape.cylinderBasisX * radialX +
            finiteShape.cylinderBasisZ * radialZ;
        const double radialSquared =
            std::max(0.0, radialX * radialX + radialZ * radialZ);

        if (radialSquared <=
            cylinderAlignmentToleranceSquared) {
            const bool positiveCap = axialProjection < 0.0;
            const Vec3 capCenter =
                finiteShape.center +
                axis *
                    (positiveCap
                        ? finiteShape.halfLength
                        : -finiteShape.halfLength);
            const std::array<Vec3, 4> directions{
                finiteShape.cylinderBasisX,
                -finiteShape.cylinderBasisX,
                finiteShape.cylinderBasisZ,
                -finiteShape.cylinderBasisZ,
            };
            const std::uint32_t featureBase =
                positiveCap
                ? cylinderPositiveCapRingBase
                : cylinderNegativeCapRingBase;
            for (std::uint32_t point = 0u;
                 point < directions.size();
                 ++point) {
                const Vec3 surface =
                    capCenter +
                    directions[point] * finiteShape.radius;
                const double separation = dot(
                    plane.planeNormal,
                    surface - plane.center
                );
                if (separation <= contactDistance) {
                    appendFinitePlaneContact(
                        result,
                        pair,
                        plane,
                        surface,
                        separation,
                        featureKey(
                            MR_SHAPE_CYLINDER,
                            featureBase + point
                        )
                    );
                }
            }
            // The four ring samples provide a stable cap manifold, but at a
            // small nonzero tilt none of those authored axes is guaranteed to
            // be the true support direction. Always add the exact extremal
            // rim witness when the radial direction is numerically resolved;
            // otherwise a shallow arbitrary-azimuth impact can be missed.
            // `radialSquared` is a direction measure, not a length scale.
            // Reusing the general geometry epsilon here creates a
            // non-conservative angular band that can be amplified by a large
            // (but valid) radius. Every resolved nonzero projection has an
            // exact support direction.
            if (radialSquared > 0.0) {
                const Vec3 radialDirection =
                    -radialProjection / std::sqrt(radialSquared);
                const Vec3 surface =
                    capCenter +
                    radialDirection * finiteShape.radius;
                const double separation = dot(
                    plane.planeNormal,
                    surface - plane.center
                );
                if (separation <= contactDistance) {
                    appendFinitePlaneContact(
                        result,
                        pair,
                        plane,
                        surface,
                        separation,
                        featureKey(
                            MR_SHAPE_CYLINDER,
                            positiveCap
                                ? cylinderPositiveGeneralRim
                                : cylinderNegativeGeneralRim
                        )
                    );
                }
            }
            return result;
        }

        const Vec3 radialDirection =
            -radialProjection / std::sqrt(radialSquared);
        if (std::abs(axialProjection) <=
            cylinderAlignmentTolerance) {
            const std::array<Vec3, 2> capCenters{
                finiteShape.center -
                    axis * finiteShape.halfLength,
                finiteShape.center +
                    axis * finiteShape.halfLength,
            };
            const std::array<std::uint32_t, 2> features{
                cylinderNegativeSideRim,
                cylinderPositiveSideRim,
            };
            for (std::uint32_t point = 0u;
                 point < capCenters.size();
                 ++point) {
                const Vec3 surface =
                    capCenters[point] +
                    radialDirection * finiteShape.radius;
                const double separation = dot(
                    plane.planeNormal,
                    surface - plane.center
                );
                if (separation <= contactDistance) {
                    appendFinitePlaneContact(
                        result,
                        pair,
                        plane,
                        surface,
                        separation,
                        featureKey(
                            MR_SHAPE_CYLINDER,
                            features[point]
                        )
                    );
                }
            }
            return result;
        }

        const bool positiveCap = axialProjection < 0.0;
        const Vec3 surface =
            finiteShape.center +
            axis *
                (positiveCap
                    ? finiteShape.halfLength
                    : -finiteShape.halfLength) +
            radialDirection * finiteShape.radius;
        const double separation =
            dot(plane.planeNormal, surface - plane.center);
        if (separation <= contactDistance) {
            appendFinitePlaneContact(
                result,
                pair,
                plane,
                surface,
                separation,
                featureKey(
                    MR_SHAPE_CYLINDER,
                    positiveCap
                        ? cylinderPositiveGeneralRim
                        : cylinderNegativeGeneralRim
                )
            );
        }
        return result;
    }

    for (std::uint32_t vertex = 0u; vertex < 8u; ++vertex) {
        const Vec3 local{
            (vertex & 1u) != 0u
                ? finiteShape.halfExtents.x
                : -finiteShape.halfExtents.x,
            (vertex & 2u) != 0u
                ? finiteShape.halfExtents.y
                : -finiteShape.halfExtents.y,
            (vertex & 4u) != 0u
                ? finiteShape.halfExtents.z
                : -finiteShape.halfExtents.z,
        };
        const Vec3 world =
            finiteShape.center +
            rotate(finiteShape.rotation, local);
        const double separation =
            dot(plane.planeNormal, world - plane.center);
        if (separation <= contactDistance) {
            appendFinitePlaneContact(
                result,
                pair,
                plane,
                world,
                separation,
                featureKey(MR_SHAPE_BOX, vertex)
            );
        }
    }
    return result;
}

bool rawContactFinite(const MRRawContactGPU& contact) {
    return
        finite(contact.normalAndSeparation) &&
        finite(contact.pointAWorld) &&
        finite(contact.pointBWorld) &&
        std::abs(
            length(xyz(contact.normalAndSeparation)) - 1.0
        ) <= 2.0e-5;
}

std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>
manifoldKey(const PersistentManifold& manifold) {
    return {
        manifold.header.pairAndCount[0],
        manifold.header.pairAndCount[1],
        manifold.header.pairAndCount[2],
    };
}

const PersistentManifold* findOldManifold(
    std::span<const PersistentManifold> oldEntries,
    const MRCandidatePairGPU& pair,
    const WorldShape& shapeA,
    const WorldShape& shapeB
) {
    const auto target = std::tuple{
        pair.environment,
        pair.colliderA,
        pair.colliderB,
    };
    const auto iterator = std::lower_bound(
        oldEntries.begin(),
        oldEntries.end(),
        target,
        [](const PersistentManifold& manifold, const auto& key) {
            return manifoldKey(manifold) < key;
        }
    );
    if (iterator == oldEntries.end() ||
        manifoldKey(*iterator) != target ||
        iterator->header.generationsAndFlags[0] !=
            shapeA.generation ||
        iterator->header.generationsAndFlags[1] !=
            shapeB.generation) {
        return nullptr;
    }
    return &*iterator;
}

Vec3 bodyLocalPoint(
    const MRBodyStateGPU& body,
    const Quaternion bodyRotation,
    const Vec3 worldPoint
) {
    return inverseRotate(
        bodyRotation,
        worldPoint - xyz(body.position)
    );
}

Vec3 bodyWorldPoint(
    const MRBodyStateGPU& body,
    const Quaternion bodyRotation,
    const mr_float4 localPoint
) {
    return
        xyz(body.position) +
        rotate(bodyRotation, xyz(localPoint));
}

void stableBasis(
    const Vec3 normal,
    Vec3& tangentU,
    Vec3& tangentV
) {
    const Vec3 absNormal = absolute(normal);
    Vec3 reference;
    if (absNormal.x <= absNormal.y &&
        absNormal.x <= absNormal.z) {
        reference = {1.0, 0.0, 0.0};
    } else if (absNormal.y <= absNormal.z) {
        reference = {0.0, 1.0, 0.0};
    } else {
        reference = {0.0, 0.0, 1.0};
    }
    tangentU = normalized(cross(reference, normal));
    tangentV = cross(normal, tangentU);
}

struct ManifoldCandidate {
    MRManifoldPointGPU point{};
    Vec3 worldPoint{};
    double separation = 0.0;
    double tangentialDrift = 0.0;
};

std::uint64_t candidateFeatureKey(const ManifoldCandidate& candidate) {
    return
        (static_cast<std::uint64_t>(
            candidate.point.featureAndLife[0]
        ) << 32u) |
        candidate.point.featureAndLife[1];
}

bool betterScore(
    const double score,
    const std::uint64_t feature,
    const double bestScore,
    const std::uint64_t bestFeature
) {
    constexpr double tolerance = 1.0e-18;
    return
        score > bestScore + tolerance ||
        (
            std::abs(score - bestScore) <= tolerance &&
            feature < bestFeature
        );
}

std::vector<ManifoldCandidate> reduceManifold(
    std::vector<ManifoldCandidate> candidates,
    const Vec3 normal
) {
    std::ranges::sort(
        candidates,
        [](const ManifoldCandidate& left,
           const ManifoldCandidate& right) {
            return std::tie(
                left.separation,
                left.point.featureAndLife[0],
                left.point.featureAndLife[1]
            ) <
                std::tie(
                    right.separation,
                    right.point.featureAndLife[0],
                    right.point.featureAndLife[1]
                );
        }
    );
    if (candidates.size() <= 4u) {
        std::ranges::sort(
            candidates,
            [](const ManifoldCandidate& left,
               const ManifoldCandidate& right) {
                return candidateFeatureKey(left) <
                    candidateFeatureKey(right);
            }
        );
        return candidates;
    }

    std::vector<std::size_t> selected;
    selected.reserve(4);
    selected.push_back(0u);

    const auto alreadySelected =
        [&selected](const std::size_t index) {
            return std::ranges::find(selected, index) !=
                selected.end();
        };

    auto selectMaximum =
        [&](const auto& scoreFunction) {
            std::size_t best = candidates.size();
            double bestScore = -1.0;
            std::uint64_t bestFeature =
                std::numeric_limits<std::uint64_t>::max();
            for (std::size_t index = 0;
                 index < candidates.size();
                 ++index) {
                if (alreadySelected(index)) {
                    continue;
                }
                const double score = scoreFunction(index);
                const std::uint64_t feature =
                    candidateFeatureKey(candidates[index]);
                if (betterScore(
                        score,
                        feature,
                        bestScore,
                        bestFeature
                    )) {
                    best = index;
                    bestScore = score;
                    bestFeature = feature;
                }
            }
            if (best != candidates.size()) {
                selected.push_back(best);
            }
        };

    selectMaximum([&](const std::size_t index) {
        const Vec3 delta =
            candidates[index].worldPoint -
            candidates[selected[0]].worldPoint;
        const Vec3 tangent = delta - normal * dot(delta, normal);
        return lengthSquared(tangent);
    });

    selectMaximum([&](const std::size_t index) {
        if (selected.size() < 2u) {
            return 0.0;
        }
        const Vec3 a =
            candidates[selected[1]].worldPoint -
            candidates[selected[0]].worldPoint;
        const Vec3 b =
            candidates[index].worldPoint -
            candidates[selected[0]].worldPoint;
        return std::abs(dot(cross(a, b), normal));
    });

    selectMaximum([&](const std::size_t index) {
        double minimumDistance =
            std::numeric_limits<double>::infinity();
        for (const std::size_t chosen : selected) {
            const Vec3 delta =
                candidates[index].worldPoint -
                candidates[chosen].worldPoint;
            const Vec3 tangent =
                delta - normal * dot(delta, normal);
            minimumDistance = std::min(
                minimumDistance,
                lengthSquared(tangent)
            );
        }
        return minimumDistance;
    });

    std::vector<ManifoldCandidate> reduced;
    reduced.reserve(selected.size());
    for (const std::size_t index : selected) {
        reduced.push_back(candidates[index]);
    }
    std::ranges::sort(
        reduced,
        [](const ManifoldCandidate& left,
           const ManifoldCandidate& right) {
            return candidateFeatureKey(left) <
                candidateFeatureKey(right);
        }
    );
    return reduced;
}

bool sameFeatures(
    const ManifoldCandidate& candidate,
    const MRRawContactGPU& contact
) {
    return
        candidate.point.featureAndLife[0] ==
            contact.featureAndFlags[0] &&
        candidate.point.featureAndLife[1] ==
            contact.featureAndFlags[1];
}

bool buildManifold(
    const MRCandidatePairGPU& pair,
    std::span<const MRRawContactGPU> rawContacts,
    std::span<const MRBodyStateGPU> bodies,
    std::span<const WorldShape> worldShapes,
    const CollisionConfig& config,
    std::span<const PersistentManifold> oldEntries,
    PersistentManifold& output,
    CollisionDiagnostics& diagnostics
) {
    if (rawContacts.empty()) {
        return false;
    }

    const WorldShape& shapeA = worldShapes[pair.colliderA];
    const WorldShape& shapeB = worldShapes[pair.colliderB];
    const MRBodyStateGPU& bodyA = bodies[shapeA.body];
    const MRBodyStateGPU& bodyB = bodies[shapeB.body];
    const auto bodyRotationAFloat =
        checkedQuaternionFloat(bodyA.orientation);
    const auto bodyRotationBFloat =
        checkedQuaternionFloat(bodyB.orientation);
    if (!bodyRotationAFloat.has_value() ||
        !bodyRotationBFloat.has_value()) {
        return false;
    }
    const Quaternion bodyRotationA =
        quaternionFromCanonicalFloat(*bodyRotationAFloat);
    const Quaternion bodyRotationB =
        quaternionFromCanonicalFloat(*bodyRotationBFloat);

    const auto deepest = std::min_element(
        rawContacts.begin(),
        rawContacts.end(),
        [](const MRRawContactGPU& left,
           const MRRawContactGPU& right) {
            return std::tie(
                left.normalAndSeparation.w,
                left.featureAndFlags[0],
                left.featureAndFlags[1]
            ) <
                std::tie(
                    right.normalAndSeparation.w,
                    right.featureAndFlags[0],
                    right.featureAndFlags[1]
                );
        }
    );
    const Vec3 normalWorld =
        normalized(xyz(deepest->normalAndSeparation));
    if (lengthSquared(normalWorld) <= kTiny) {
        return false;
    }

    const PersistentManifold* old = findOldManifold(
        oldEntries,
        pair,
        shapeA,
        shapeB
    );
    std::vector<ManifoldCandidate> candidates;
    candidates.reserve(12u);

    bool oldNormalCompatible = false;
    if (old != nullptr && old->header.pairAndCount[3] <= 4u) {
        const Vec3 oldNormalWorld = normalized(
            rotate(
                bodyRotationA,
                xyz(old->header.normalAndAge)
            )
        );
        oldNormalCompatible =
            dot(oldNormalWorld, normalWorld) >=
            config.manifoldNormalCosine;
        if (oldNormalCompatible) {
            for (std::uint32_t pointIndex = 0u;
                 pointIndex < old->header.pairAndCount[3];
                 ++pointIndex) {
                const MRManifoldPointGPU& oldPoint =
                    old->points[pointIndex];
                const Vec3 pointAWorld = bodyWorldPoint(
                    bodyA,
                    bodyRotationA,
                    oldPoint.localAnchorA
                );
                const Vec3 pointBWorld = bodyWorldPoint(
                    bodyB,
                    bodyRotationB,
                    oldPoint.localAnchorB
                );
                const Vec3 delta = pointBWorld - pointAWorld;
                const double separation = dot(delta, normalWorld);
                const Vec3 tangentDelta =
                    delta - normalWorld * separation;
                const double tangentialDrift =
                    length(tangentDelta);
                if (separation >
                        config.manifoldBreakingSeparation ||
                    tangentialDrift >
                        config.manifoldBreakingTangential) {
                    continue;
                }

                ManifoldCandidate candidate;
                candidate.point = oldPoint;
                candidate.point.featureAndLife[2] =
                    oldPoint.featureAndLife[2] ==
                        std::numeric_limits<std::uint32_t>::max()
                    ? oldPoint.featureAndLife[2]
                    : oldPoint.featureAndLife[2] + 1u;
                candidate.worldPoint =
                    0.5 * (pointAWorld + pointBWorld);
                candidate.separation = separation;
                candidate.tangentialDrift = tangentialDrift;
                candidates.push_back(candidate);
                ++diagnostics.refreshedPoints;
            }
        }
    }

    const double mergeDistanceSquared =
        config.manifoldMergeDistance *
        config.manifoldMergeDistance;
    for (const MRRawContactGPU& contact : rawContacts) {
        const Vec3 pointAWorld = xyz(contact.pointAWorld);
        const Vec3 pointBWorld = xyz(contact.pointBWorld);
        const Vec3 localA = bodyLocalPoint(
            bodyA,
            bodyRotationA,
            pointAWorld
        );
        const Vec3 localB = bodyLocalPoint(
            bodyB,
            bodyRotationB,
            pointBWorld
        );

        auto match = std::find_if(
            candidates.begin(),
            candidates.end(),
            [&contact](const ManifoldCandidate& candidate) {
                return sameFeatures(candidate, contact);
            }
        );
        if (match == candidates.end()) {
            match = std::find_if(
                candidates.begin(),
                candidates.end(),
                [&](const ManifoldCandidate& candidate) {
                    return
                        lengthSquared(
                            xyz(candidate.point.localAnchorA) -
                            localA
                        ) +
                        lengthSquared(
                            xyz(candidate.point.localAnchorB) -
                            localB
                        ) <= 2.0 * mergeDistanceSquared;
                }
            );
        }

        if (match == candidates.end()) {
            ManifoldCandidate candidate;
            candidate.point.localAnchorA = pointFloat4(localA);
            candidate.point.localAnchorB = pointFloat4(localB);
            candidate.point.featureAndLife[0] =
                contact.featureAndFlags[0];
            candidate.point.featureAndLife[1] =
                contact.featureAndFlags[1];
            candidate.point.featureAndLife[2] = 0u;
            candidate.point.featureAndLife[3] =
                contact.featureAndFlags[3];
            candidate.worldPoint =
                0.5 * (pointAWorld + pointBWorld);
            candidate.separation =
                contact.normalAndSeparation.w;
            candidates.push_back(candidate);
            ++diagnostics.newPoints;
        } else {
            match->point.localAnchorA = pointFloat4(localA);
            match->point.localAnchorB = pointFloat4(localB);
            match->point.featureAndLife[0] =
                contact.featureAndFlags[0];
            match->point.featureAndLife[1] =
                contact.featureAndFlags[1];
            match->point.featureAndLife[3] =
                contact.featureAndFlags[3];
            match->worldPoint =
                0.5 * (pointAWorld + pointBWorld);
            match->separation =
                contact.normalAndSeparation.w;
            match->tangentialDrift = 0.0;
        }
    }

    candidates = reduceManifold(
        std::move(candidates),
        normalWorld
    );
    if (candidates.empty()) {
        return false;
    }

    output = {};
    output.header.pairAndCount[0] = pair.environment;
    output.header.pairAndCount[1] = pair.colliderA;
    output.header.pairAndCount[2] = pair.colliderB;
    output.header.pairAndCount[3] =
        static_cast<std::uint32_t>(candidates.size());
    output.header.generationsAndFlags[0] = shapeA.generation;
    output.header.generationsAndFlags[1] = shapeB.generation;
    output.header.generationsAndFlags[2] = 0u;
    output.header.generationsAndFlags[3] = 0u;

    const Vec3 normalLocalA =
        inverseRotate(bodyRotationA, normalWorld);
    Vec3 tangentWorld;
    Vec3 bitangentWorld;
    stableBasis(normalWorld, tangentWorld, bitangentWorld);
    const Vec3 tangentLocalA =
        inverseRotate(bodyRotationA, tangentWorld);
    const float age =
        old != nullptr && oldNormalCompatible
        ? std::min(
            old->header.normalAndAge.w + 1.0f,
            static_cast<float>(
                std::numeric_limits<std::uint16_t>::max()
            )
        )
        : 0.0f;
    output.header.normalAndAge =
        vectorFloat4(normalLocalA, age);

    double breakingMetric = 0.0;
    for (const ManifoldCandidate& candidate : candidates) {
        breakingMetric = std::max(
            breakingMetric,
            std::max(
                std::max(candidate.separation, 0.0),
                candidate.tangentialDrift
            )
        );
    }
    output.header.tangentAndMetric = vectorFloat4(
        tangentLocalA,
        static_cast<float>(breakingMetric)
    );
    for (std::size_t point = 0; point < candidates.size(); ++point) {
        output.points[point] = candidates[point].point;
    }
    return true;
}

CollisionFrame failureFrame(const CollisionDiagnostics& diagnostics) {
    CollisionFrame result;
    result.diagnostics = diagnostics;
    return result;
}

bool validConfig(const CollisionConfig& config) {
    return
        std::isfinite(config.manifoldBreakingSeparation) &&
        std::isfinite(config.manifoldBreakingTangential) &&
        std::isfinite(config.manifoldMergeDistance) &&
        std::isfinite(config.manifoldNormalCosine) &&
        config.manifoldBreakingSeparation >= 0.0 &&
        config.manifoldBreakingTangential >= 0.0 &&
        config.manifoldMergeDistance >= 0.0 &&
        config.manifoldNormalCosine >= -1.0 &&
        config.manifoldNormalCosine <= 1.0;
}

} // namespace

CollisionFrame collideCpuReference(
    std::span<const MRShapeGPU> shapes,
    std::span<const MRBodyStateGPU> bodies,
    const CollisionConfig& config,
    PersistentManifoldCache& cache,
    std::span<const CollisionPairExclusion> exclusions
) {
    CollisionDiagnostics diagnostics;
    if (!validConfig(config) ||
        shapes.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        diagnostics.code = MR_STEP_NONFINITE_INPUT;
        return failureFrame(diagnostics);
    }
    for (const MRBodyStateGPU& body : bodies) {
        if (!collisionInputDomainXyz(body.position) ||
            !finite(body.orientation) ||
            body.flagsAndIndices[0] > MR_MOTION_DYNAMIC ||
            !checkedQuaternionFloat(
                body.orientation
            ).has_value()) {
            diagnostics.code = MR_STEP_NONFINITE_INPUT;
            return failureFrame(diagnostics);
        }
    }
    for (const CollisionPairExclusion exclusion : exclusions) {
        if (exclusion.colliderA >= shapes.size() ||
            exclusion.colliderB >= shapes.size()) {
            diagnostics.code = MR_STEP_NONFINITE_INPUT;
            return failureFrame(diagnostics);
        }
    }
    for (const MRShapeGPU& shape : shapes) {
        if (!validShapeRecord(shape, bodies)) {
            diagnostics.code = MR_STEP_NONFINITE_INPUT;
            return failureFrame(diagnostics);
        }
    }

    std::vector<WorldShape> worldShapes;
    worldShapes.reserve(shapes.size());
    for (std::uint32_t shapeIndex = 0u;
         shapeIndex < shapes.size();
         ++shapeIndex) {
        const auto world = makeWorldShape(
            shapeIndex,
            shapes[shapeIndex],
            bodies
        );
        if (!world.has_value()) {
            diagnostics.code =
                supportedShapeType(shapes[shapeIndex].shapeType)
                ? MR_STEP_NONFINITE_INPUT
                : MR_STEP_UNSUPPORTED;
            return failureFrame(diagnostics);
        }
        worldShapes.push_back(*world);
    }

    const std::vector<std::uint64_t> exclusionKeys =
        canonicalExclusions(exclusions, shapes.size());
    diagnostics.broadphaseAxis = broadphaseAxis(worldShapes);

    std::uint64_t pairCount = 0u;
    forEachBroadphasePair(
        shapes,
        bodies,
        worldShapes,
        exclusionKeys,
        diagnostics.broadphaseAxis,
        [&](std::uint32_t, std::uint32_t, std::uint32_t) {
            if (pairCount !=
                std::numeric_limits<std::uint64_t>::max()) {
                ++pairCount;
            }
        }
    );
    diagnostics.requiredPairs = pairCount;
    if (pairCount > config.capacities.pairCapacity) {
        diagnostics.code = MR_STEP_PAIR_CAPACITY_OVERFLOW;
        return failureFrame(diagnostics);
    }

    std::vector<MRCandidatePairGPU> pairs;
    pairs.reserve(static_cast<std::size_t>(pairCount));
    forEachBroadphasePair(
        shapes,
        bodies,
        worldShapes,
        exclusionKeys,
        diagnostics.broadphaseAxis,
        [&](const std::uint32_t colliderA,
            const std::uint32_t colliderB,
            const std::uint32_t flags) {
            pairs.push_back({
                config.environment,
                colliderA,
                colliderB,
                flags,
            });
        }
    );
    std::ranges::sort(
        pairs,
        [](const MRCandidatePairGPU& left,
           const MRCandidatePairGPU& right) {
            return std::tie(
                left.environment,
                left.colliderA,
                left.colliderB
            ) <
                std::tie(
                    right.environment,
                    right.colliderA,
                    right.colliderB
                );
        }
    );
    pairs.erase(
        std::unique(
            pairs.begin(),
            pairs.end(),
            [](const MRCandidatePairGPU& left,
               const MRCandidatePairGPU& right) {
                return
                    left.environment == right.environment &&
                    left.colliderA == right.colliderA &&
                    left.colliderB == right.colliderB;
            }
        ),
        pairs.end()
    );
    diagnostics.requiredPairs = pairs.size();

    std::vector<std::uint32_t> contactCounts(pairs.size(), 0u);
    std::uint64_t rawContactCount = 0u;
    std::uint64_t manifoldCount = 0u;
    for (std::size_t pairIndex = 0;
         pairIndex < pairs.size();
         ++pairIndex) {
        const ContactBatch batch = generateContacts(
            pairs[pairIndex],
            worldShapes
        );
        contactCounts[pairIndex] = batch.count;
        if (batch.count > 0u) {
            ++manifoldCount;
        }
        if (rawContactCount >
            std::numeric_limits<std::uint64_t>::max() -
                batch.count) {
            diagnostics.code = MR_STEP_CONTACT_CAPACITY_OVERFLOW;
            diagnostics.requiredRawContacts =
                std::numeric_limits<std::uint64_t>::max();
            return failureFrame(diagnostics);
        }
        rawContactCount += batch.count;
    }
    diagnostics.requiredRawContacts = rawContactCount;
    diagnostics.requiredManifolds = manifoldCount;
    if (rawContactCount >
        config.capacities.rawContactCapacity) {
        diagnostics.code = MR_STEP_CONTACT_CAPACITY_OVERFLOW;
        return failureFrame(diagnostics);
    }
    if (manifoldCount >
        config.capacities.manifoldCapacity) {
        diagnostics.code = MR_STEP_MANIFOLD_CAPACITY_OVERFLOW;
        return failureFrame(diagnostics);
    }

    std::vector<MRRawContactGPU> rawContacts;
    std::vector<std::uint32_t> rawContactPairIndices;
    rawContacts.reserve(static_cast<std::size_t>(rawContactCount));
    rawContactPairIndices.reserve(
        static_cast<std::size_t>(rawContactCount)
    );
    std::vector<std::size_t> pairContactOffsets(
        pairs.size() + 1u,
        0u
    );
    for (std::size_t pairIndex = 0;
         pairIndex < pairs.size();
         ++pairIndex) {
        pairContactOffsets[pairIndex] = rawContacts.size();
        const ContactBatch batch = generateContacts(
            pairs[pairIndex],
            worldShapes
        );
        if (batch.count != contactCounts[pairIndex]) {
            diagnostics.code = MR_STEP_NONFINITE_RESULT;
            return failureFrame(diagnostics);
        }
        for (std::uint32_t contactIndex = 0u;
             contactIndex < batch.count;
             ++contactIndex) {
            if (!rawContactFinite(batch.contacts[contactIndex])) {
                diagnostics.code = MR_STEP_NONFINITE_RESULT;
                return failureFrame(diagnostics);
            }
            rawContacts.push_back(batch.contacts[contactIndex]);
            rawContactPairIndices.push_back(
                static_cast<std::uint32_t>(pairIndex)
            );
        }
    }
    pairContactOffsets.back() = rawContacts.size();

    const std::span<const PersistentManifold> oldEntries =
        CollisionCacheAccess::read(cache);
    std::vector<PersistentManifold> nextEntries;
    nextEntries.reserve(static_cast<std::size_t>(manifoldCount));
    for (std::size_t pairIndex = 0;
         pairIndex < pairs.size();
         ++pairIndex) {
        const std::size_t begin = pairContactOffsets[pairIndex];
        const std::size_t end = pairContactOffsets[pairIndex + 1u];
        if (begin == end) {
            continue;
        }
        PersistentManifold manifold;
        if (!buildManifold(
                pairs[pairIndex],
                std::span(rawContacts).subspan(
                    begin,
                    end - begin
                ),
                bodies,
                worldShapes,
                config,
                oldEntries,
                manifold,
                diagnostics
            )) {
            diagnostics.code = MR_STEP_NONFINITE_RESULT;
            return failureFrame(diagnostics);
        }
        nextEntries.push_back(manifold);
    }
    if (nextEntries.size() != manifoldCount) {
        diagnostics.code = MR_STEP_NONFINITE_RESULT;
        return failureFrame(diagnostics);
    }

    std::vector<MRAabbGPU> aabbs;
    aabbs.reserve(worldShapes.size());
    for (const WorldShape& shape : worldShapes) {
        aabbs.push_back(shape.aabb);
    }

    CollisionFrame result;
    result.diagnostics = diagnostics;
    result.worldAabbs = std::move(aabbs);
    result.pairs = std::move(pairs);
    result.rawContacts = std::move(rawContacts);
    result.rawContactPairIndices =
        std::move(rawContactPairIndices);
    result.manifoldHeaders.reserve(nextEntries.size());
    result.manifoldPoints.reserve(nextEntries.size() * 4u);
    for (const PersistentManifold& manifold : nextEntries) {
        result.manifoldHeaders.push_back(manifold.header);
        result.manifoldPoints.insert(
            result.manifoldPoints.end(),
            manifold.points.begin(),
            manifold.points.end()
        );
    }
    CollisionCacheAccess::commit(cache, std::move(nextEntries));
    return result;
}

} // namespace metalrobo
