#include "metalrobo/SurgicalAssets.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace metalrobo {
namespace {

constexpr double kMinimumDimension = 1.0e-9;
constexpr std::string_view kValidationBoundary =
    "Research/training rigid geometry only; no tissue puncture, cutting, "
    "biomechanical, or clinical validation.";

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Mat3 {
    double m[3][3]{};
};

struct MassComponent {
    double volume = 0.0;
    double mass = 0.0;
    Vec3 center{};
    Mat3 inertiaAtCenter{};
};

struct MassProperties {
    double volume = 0.0;
    double mass = 0.0;
    Vec3 center{};
    Mat3 inertiaAtCenter{};
};

Vec3 operator+(const Vec3 a, const Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3 a, const Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(const Vec3 value, const double scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
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

double norm(const Vec3 value) {
    return std::sqrt(dot(value, value));
}

Vec3 normalized(const Vec3 value) {
    const double length = norm(value);
    if (!(length > kMinimumDimension) || !std::isfinite(length)) {
        throw std::invalid_argument("surgical asset has a degenerate axis");
    }
    return value / length;
}

Mat3 identity() {
    Mat3 result{};
    result.m[0][0] = 1.0;
    result.m[1][1] = 1.0;
    result.m[2][2] = 1.0;
    return result;
}

Mat3 operator+(const Mat3& a, const Mat3& b) {
    Mat3 result{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            result.m[row][column] =
                a.m[row][column] + b.m[row][column];
        }
    }
    return result;
}

Mat3& operator+=(Mat3& a, const Mat3& b) {
    a = a + b;
    return a;
}

Mat3 operator*(const Mat3& value, const double scale) {
    Mat3 result{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            result.m[row][column] = value.m[row][column] * scale;
        }
    }
    return result;
}

Mat3 operator-(const Mat3& a, const Mat3& b) {
    Mat3 result{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            result.m[row][column] =
                a.m[row][column] - b.m[row][column];
        }
    }
    return result;
}

Mat3 outer(const Vec3 value) {
    Mat3 result{};
    const double elements[3] = {value.x, value.y, value.z};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            result.m[row][column] = elements[row] * elements[column];
        }
    }
    return result;
}

Mat3 axialInertia(
    const Vec3 axis,
    const double transverse,
    const double axial
) {
    return
        identity() * transverse +
        outer(axis) * (axial - transverse);
}

Mat3 parallelAxis(const double mass, const Vec3 offset) {
    return
        identity() * (mass * dot(offset, offset)) -
        outer(offset) * mass;
}

MassComponent cylinderComponent(
    const Vec3 center,
    const Vec3 axis,
    const double radius,
    const double length,
    const double density
) {
    MassComponent result;
    result.volume =
        std::numbers::pi * radius * radius * length;
    result.mass = density * result.volume;
    result.center = center;
    const double axial = 0.5 * result.mass * radius * radius;
    const double transverse =
        result.mass *
        (3.0 * radius * radius + length * length) / 12.0;
    result.inertiaAtCenter =
        axialInertia(normalized(axis), transverse, axial);
    return result;
}

MassComponent hemisphereComponent(
    const Vec3 baseCenter,
    const Vec3 outwardAxis,
    const double radius,
    const double density
) {
    MassComponent result;
    result.volume =
        (2.0 / 3.0) * std::numbers::pi *
        radius * radius * radius;
    result.mass = density * result.volume;
    const Vec3 axis = normalized(outwardAxis);
    result.center = baseCenter + axis * (3.0 * radius / 8.0);
    const double axial =
        (2.0 / 5.0) * result.mass * radius * radius;
    const double transverse =
        (83.0 / 320.0) * result.mass * radius * radius;
    result.inertiaAtCenter =
        axialInertia(axis, transverse, axial);
    return result;
}

MassComponent boxComponent(
    const Vec3 center,
    const Vec3 size,
    const double density
) {
    MassComponent result;
    result.volume = size.x * size.y * size.z;
    result.mass = density * result.volume;
    result.center = center;
    result.inertiaAtCenter.m[0][0] =
        result.mass * (size.y * size.y + size.z * size.z) / 12.0;
    result.inertiaAtCenter.m[1][1] =
        result.mass * (size.x * size.x + size.z * size.z) / 12.0;
    result.inertiaAtCenter.m[2][2] =
        result.mass * (size.x * size.x + size.y * size.y) / 12.0;
    return result;
}

MassProperties combineMass(
    const std::vector<MassComponent>& components
) {
    if (components.empty()) {
        throw std::invalid_argument("surgical asset has no mass geometry");
    }

    MassProperties result;
    Vec3 firstMoment{};
    for (const MassComponent& component : components) {
        if (!(component.mass > 0.0) ||
            !(component.volume > 0.0) ||
            !std::isfinite(component.mass) ||
            !std::isfinite(component.volume)) {
            throw std::invalid_argument(
                "surgical mass component is invalid"
            );
        }
        result.volume += component.volume;
        result.mass += component.mass;
        firstMoment = firstMoment + component.center * component.mass;
    }
    result.center = firstMoment / result.mass;

    for (const MassComponent& component : components) {
        result.inertiaAtCenter +=
            component.inertiaAtCenter +
            parallelAxis(
                component.mass,
                component.center - result.center
            );
    }
    return result;
}

double determinant(const Mat3& value) {
    return
        value.m[0][0] *
            (value.m[1][1] * value.m[2][2] -
             value.m[1][2] * value.m[2][1]) -
        value.m[0][1] *
            (value.m[1][0] * value.m[2][2] -
             value.m[1][2] * value.m[2][0]) +
        value.m[0][2] *
            (value.m[1][0] * value.m[2][1] -
             value.m[1][1] * value.m[2][0]);
}

Mat3 inverse(const Mat3& value) {
    const double det = determinant(value);
    if (!(det > 0.0) || !std::isfinite(det)) {
        throw std::invalid_argument(
            "surgical asset inertia is not positive definite"
        );
    }

    Mat3 result{};
    result.m[0][0] =
        (value.m[1][1] * value.m[2][2] -
         value.m[1][2] * value.m[2][1]) / det;
    result.m[0][1] =
        (value.m[0][2] * value.m[2][1] -
         value.m[0][1] * value.m[2][2]) / det;
    result.m[0][2] =
        (value.m[0][1] * value.m[1][2] -
         value.m[0][2] * value.m[1][1]) / det;
    result.m[1][0] =
        (value.m[1][2] * value.m[2][0] -
         value.m[1][0] * value.m[2][2]) / det;
    result.m[1][1] =
        (value.m[0][0] * value.m[2][2] -
         value.m[0][2] * value.m[2][0]) / det;
    result.m[1][2] =
        (value.m[0][2] * value.m[1][0] -
         value.m[0][0] * value.m[1][2]) / det;
    result.m[2][0] =
        (value.m[1][0] * value.m[2][1] -
         value.m[1][1] * value.m[2][0]) / det;
    result.m[2][1] =
        (value.m[0][1] * value.m[2][0] -
         value.m[0][0] * value.m[2][1]) / det;
    result.m[2][2] =
        (value.m[0][0] * value.m[1][1] -
         value.m[0][1] * value.m[1][0]) / det;
    return result;
}

mr_float4 f4(
    const double x,
    const double y,
    const double z,
    const double w = 0.0
) {
    return {
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z),
        static_cast<float>(w),
    };
}

mr_float4 f4(const Vec3 value, const double w = 0.0) {
    return f4(value.x, value.y, value.z, w);
}

mr_float4 quaternionFromY(const Vec3 target) {
    const Vec3 direction = normalized(target);
    const Vec3 source{0.0, 1.0, 0.0};
    const double cosine = std::clamp(dot(source, direction), -1.0, 1.0);
    if (cosine < -1.0 + 1.0e-12) {
        return f4(1.0, 0.0, 0.0, 0.0);
    }
    const Vec3 xyz = cross(source, direction);
    const double scale = std::sqrt(2.0 * (1.0 + cosine));
    const Vec3 vector = xyz / scale;
    return f4(vector.x, vector.y, vector.z, 0.5 * scale);
}

Vec3 rotate(const mr_float4 quaternion, const Vec3 value) {
    const Vec3 q{quaternion.x, quaternion.y, quaternion.z};
    const Vec3 twiceCross = cross(q, value) * 2.0;
    return
        value +
        twiceCross * static_cast<double>(quaternion.w) +
        cross(q, twiceCross);
}

void validateScalar(
    const SurgicalScalar scalar,
    const std::string_view name,
    const bool allowZero = false
) {
    const auto basis = static_cast<std::uint32_t>(scalar.basis);
    if (!std::isfinite(scalar.value) ||
        (allowZero ? scalar.value < 0.0 : scalar.value <= 0.0) ||
        basis > static_cast<std::uint32_t>(
            SurgicalValueBasis::researchDefault
        )) {
        throw std::invalid_argument(
            std::string(name) + " is not a valid sourced scalar"
        );
    }
}

void validateIds(
    const SurgicalAssetIds& ids,
    const std::size_t shapeCount
) {
    if (ids.motionType > MR_MOTION_DYNAMIC ||
        shapeCount >
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max() -
                ids.slotGenerationBase
            )) {
        throw std::invalid_argument(
            "surgical asset IDs or motion type are invalid"
        );
    }
}

MRMaterialGPU steelResearchMaterial() {
    MRMaterialGPU result{};
    result.friction = f4(0.35, 0.25, 0.0, 0.0);
    result.response = f4(0.0, 0.05, 1.0e-9, 0.0);
    result.geometry = f4(0.0, 0.0, 0.0, 0.0);
    return result;
}

MRMaterialGPU polymerResearchMaterial() {
    MRMaterialGPU result{};
    result.friction = f4(0.70, 0.55, 0.0, 0.0);
    result.response = f4(0.05, 0.05, 2.0e-9, 0.0);
    result.geometry = f4(0.0, 0.0, 0.0, 0.0);
    return result;
}

MRShapeGPU shapeRecord(
    const SurgicalAssetIds& ids,
    const std::uint32_t localShape,
    const std::uint32_t type,
    const Vec3 center,
    const mr_float4 rotation,
    const mr_float4 dimensions,
    const double contactOffset,
    const double boundingRadius
) {
    MRShapeGPU result{};
    result.bodyIndex = ids.bodyIndex;
    result.shapeType = type;
    result.materialIndex = ids.materialIndex;
    result.collisionGroup = ids.collisionGroup;
    result.collisionMask = ids.collisionMask;
    result.slotGeneration = ids.slotGenerationBase + localShape;
    result.localPosition = f4(center, 1.0);
    result.localRotation = rotation;
    result.dimensions = dimensions;
    result.contactRestAndBoundingRadius =
        f4(contactOffset, 0.0, boundingRadius, 0.0);
    return result;
}

MRBodyPropertiesGPU bodyRecord(
    const SurgicalAssetIds& ids,
    const MassProperties& properties
) {
    const Mat3 inverseInertia = inverse(properties.inertiaAtCenter);
    MRBodyPropertiesGPU result{};
    result.articulationIndex = MR_INVALID_INDEX;
    result.parentBody = MR_INVALID_INDEX;
    result.inboundJoint = MR_INVALID_INDEX;
    result.motionType = ids.motionType;
    result.massAndInverseMass = f4(
        properties.mass,
        ids.motionType == MR_MOTION_DYNAMIC
            ? 1.0 / properties.mass
            : 0.0,
        0.0,
        0.0
    );
    result.centerOfMass = f4(0.0, 0.0, 0.0, 0.0);
    result.inertiaRow0 = f4(
        properties.inertiaAtCenter.m[0][0],
        properties.inertiaAtCenter.m[0][1],
        properties.inertiaAtCenter.m[0][2],
        0.0
    );
    result.inertiaRow1 = f4(
        properties.inertiaAtCenter.m[1][0],
        properties.inertiaAtCenter.m[1][1],
        properties.inertiaAtCenter.m[1][2],
        0.0
    );
    result.inertiaRow2 = f4(
        properties.inertiaAtCenter.m[2][0],
        properties.inertiaAtCenter.m[2][1],
        properties.inertiaAtCenter.m[2][2],
        0.0
    );
    if (ids.motionType == MR_MOTION_DYNAMIC) {
        result.inverseInertiaRow0 = f4(
            inverseInertia.m[0][0],
            inverseInertia.m[0][1],
            inverseInertia.m[0][2],
            0.0
        );
        result.inverseInertiaRow1 = f4(
            inverseInertia.m[1][0],
            inverseInertia.m[1][1],
            inverseInertia.m[1][2],
            0.0
        );
        result.inverseInertiaRow2 = f4(
            inverseInertia.m[2][0],
            inverseInertia.m[2][1],
            inverseInertia.m[2][2],
            0.0
        );
    }
    result.dampingAndSpeedLimits = f4(0.0, 0.0, 25.0, 1000.0);
    return result;
}

void shiftShapesToCom(
    std::vector<MRShapeGPU>& shapes,
    const Vec3 centerOfMass
) {
    for (MRShapeGPU& shape : shapes) {
        shape.localPosition.x -= static_cast<float>(centerOfMass.x);
        shape.localPosition.y -= static_cast<float>(centerOfMass.y);
        shape.localPosition.z -= static_cast<float>(centerOfMass.z);
    }
}

void computeAabb(SurgicalRigidAsset& asset) {
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

    auto include = [&](const Vec3 point) {
        lower.x = std::min(lower.x, point.x);
        lower.y = std::min(lower.y, point.y);
        lower.z = std::min(lower.z, point.z);
        upper.x = std::max(upper.x, point.x);
        upper.y = std::max(upper.y, point.y);
        upper.z = std::max(upper.z, point.z);
    };

    for (const MRShapeGPU& shape : asset.shapes) {
        const Vec3 center{
            shape.localPosition.x,
            shape.localPosition.y,
            shape.localPosition.z,
        };
        const double contact = shape.contactRestAndBoundingRadius.x;
        if (shape.shapeType == MR_SHAPE_SPHERE) {
            const double radius = shape.dimensions.x + contact;
            include(center - Vec3{radius, radius, radius});
            include(center + Vec3{radius, radius, radius});
        } else if (shape.shapeType == MR_SHAPE_CAPSULE) {
            const double radius = shape.dimensions.x + contact;
            const Vec3 axis = rotate(
                shape.localRotation,
                {0.0, shape.dimensions.y, 0.0}
            );
            const Vec3 expansion{radius, radius, radius};
            include(center - axis - expansion);
            include(center - axis + expansion);
            include(center + axis - expansion);
            include(center + axis + expansion);
        } else if (shape.shapeType == MR_SHAPE_BOX) {
            const Vec3 half{
                shape.dimensions.x,
                shape.dimensions.y,
                shape.dimensions.z,
            };
            const Vec3 basisX = rotate(
                shape.localRotation,
                {half.x, 0.0, 0.0}
            );
            const Vec3 basisY = rotate(
                shape.localRotation,
                {0.0, half.y, 0.0}
            );
            const Vec3 basisZ = rotate(
                shape.localRotation,
                {0.0, 0.0, half.z}
            );
            const Vec3 extent{
                std::abs(basisX.x) + std::abs(basisY.x) +
                    std::abs(basisZ.x) + contact,
                std::abs(basisX.y) + std::abs(basisY.y) +
                    std::abs(basisZ.y) + contact,
                std::abs(basisX.z) + std::abs(basisY.z) +
                    std::abs(basisZ.z) + contact,
            };
            include(center - extent);
            include(center + extent);
        } else {
            throw std::invalid_argument(
                "surgical asset contains an unsupported primitive"
            );
        }
    }
    asset.localAabbLowerM = {lower.x, lower.y, lower.z};
    asset.localAabbUpperM = {upper.x, upper.y, upper.z};
}

SurgicalRigidAsset finalizeAsset(
    const SurgicalAssetIds& ids,
    const MassProperties& mass,
    std::vector<MRShapeGPU> shapes,
    const MRMaterialGPU material,
    const std::string_view name
) {
    shiftShapesToCom(shapes, mass.center);

    SurgicalRigidAsset result;
    result.body = bodyRecord(ids, mass);
    result.material = material;
    result.shapes = std::move(shapes);
    result.volumeM3 = mass.volume;
    result.massKg = mass.mass;
    result.geometryCenterOfMassM = {
        mass.center.x,
        mass.center.y,
        mass.center.z,
    };
    result.name = name;
    result.validationBoundary = kValidationBoundary;
    computeAabb(result);
    return result;
}

double needleRadiusAt(
    const CurvedSutureNeedleSpec& spec,
    const double arcPosition
) {
    const double arcLength = spec.arcLengthM.value;
    const double baseRadius = spec.crossSectionRadiusM.value;
    if (arcPosition < spec.swageLengthM.value) {
        const double fraction =
            1.0 - arcPosition / spec.swageLengthM.value;
        return baseRadius *
            (1.0 + (spec.swageRadiusRatio - 1.0) * fraction);
    }
    const double tipStart = arcLength - spec.tipTaperLengthM.value;
    if (arcPosition > tipStart) {
        const double remaining =
            (arcLength - arcPosition) / spec.tipTaperLengthM.value;
        return baseRadius *
            (spec.tipRadiusRatio +
             (1.0 - spec.tipRadiusRatio) * remaining);
    }
    return baseRadius;
}

} // namespace

std::string_view surgicalValueBasisName(
    const SurgicalValueBasis basis
) noexcept {
    switch (basis) {
    case SurgicalValueBasis::gs21ProductGeometry:
        return "sourced:GS-21-product-geometry";
    case SurgicalValueBasis::orbitSurgicalOpenAsset:
        return "sourced:ORBIT-Surgical-open-asset";
    case SurgicalValueBasis::derivedGeometry:
        return "derived-from-geometry";
    case SurgicalValueBasis::researchDefault:
        return "explicit-research-default";
    }
    return "invalid";
}

std::string_view surgicalValueSourceReference(
    const SurgicalValueBasis basis
) noexcept {
    switch (basis) {
    case SurgicalValueBasis::gs21ProductGeometry:
        return "https://www.medtronic.com/content/dam/medtronic-wide/"
            "public/united-states/products/wound-closure/"
            "v-loc-wound-closure-device-robotics-guide-brochure.pdf";
    case SurgicalValueBasis::orbitSurgicalOpenAsset:
        return "https://github.com/orbit-surgical/orbit-surgical/tree/"
            "6e47534f7d412e4be523116f250c992a63146883";
    case SurgicalValueBasis::derivedGeometry:
        return "analytic primitive geometry";
    case SurgicalValueBasis::researchDefault:
        return "MetalRobo research default; measurement required";
    }
    return "invalid";
}

CurvedSutureNeedleAsset makeCurvedSutureNeedleAsset(
    const SurgicalAssetIds& ids,
    const CurvedSutureNeedleSpec& spec
) {
    validateScalar(spec.arcLengthM, "needle arc length");
    validateScalar(spec.arcAngleRad, "needle arc angle");
    validateScalar(spec.crossSectionRadiusM, "needle cross section");
    validateScalar(spec.densityKgPerM3, "needle density");
    validateScalar(spec.tipTaperLengthM, "needle tip taper");
    validateScalar(spec.swageLengthM, "needle swage length");
    validateScalar(
        spec.graspZoneStartFraction,
        "needle grasp-zone start",
        true
    );
    validateScalar(
        spec.graspZoneEndFraction,
        "needle grasp-zone end"
    );
    if (spec.arcSegments < 8u ||
        spec.arcSegments > 256u ||
        spec.arcAngleRad.value > 2.0 * std::numbers::pi ||
        spec.tipTaperLengthM.value >= spec.arcLengthM.value ||
        spec.swageLengthM.value >= spec.arcLengthM.value ||
        spec.tipTaperLengthM.value + spec.swageLengthM.value >=
            spec.arcLengthM.value ||
        !(spec.graspZoneStartFraction.value <
          spec.graspZoneEndFraction.value) ||
        spec.graspZoneEndFraction.value > 1.0 ||
        !(spec.tipRadiusRatio > 0.0) ||
        spec.tipRadiusRatio >= 1.0 ||
        spec.swageRadiusRatio < 1.0 ||
        !std::isfinite(spec.tipRadiusRatio) ||
        !std::isfinite(spec.swageRadiusRatio) ||
        !(spec.contactOffsetM >= 0.0) ||
        !std::isfinite(spec.contactOffsetM)) {
        throw std::invalid_argument("curved needle specification is invalid");
    }
    validateIds(ids, spec.arcSegments);

    const double radius =
        spec.arcLengthM.value / spec.arcAngleRad.value;
    const double angleStep =
        spec.arcAngleRad.value /
        static_cast<double>(spec.arcSegments);
    const double startAngle = -0.5 * spec.arcAngleRad.value;
    std::vector<Vec3> nodes;
    nodes.reserve(static_cast<std::size_t>(spec.arcSegments) + 1u);
    for (std::uint32_t index = 0u;
         index <= spec.arcSegments;
         ++index) {
        const double angle =
            startAngle + angleStep * static_cast<double>(index);
        nodes.push_back({
            radius * std::cos(angle),
            radius * std::sin(angle),
            0.0,
        });
    }

    std::vector<MassComponent> massComponents;
    massComponents.reserve(
        static_cast<std::size_t>(spec.arcSegments) + 2u
    );
    std::vector<MRShapeGPU> shapes;
    shapes.reserve(spec.arcSegments);
    double representedLength = 0.0;
    for (std::uint32_t index = 0u;
         index < spec.arcSegments;
         ++index) {
        const Vec3 segment = nodes[index + 1u] - nodes[index];
        const double chordLength = norm(segment);
        const Vec3 axis = segment / chordLength;
        const Vec3 center = (nodes[index] + nodes[index + 1u]) * 0.5;
        const double arcEnd =
            spec.arcLengthM.value *
            static_cast<double>(index + 1u) /
            static_cast<double>(spec.arcSegments);
        const double segmentRadius = needleRadiusAt(spec, arcEnd);
        massComponents.push_back(cylinderComponent(
            center,
            axis,
            segmentRadius,
            chordLength,
            spec.densityKgPerM3.value
        ));
        representedLength += chordLength;
        shapes.push_back(shapeRecord(
            ids,
            index,
            MR_SHAPE_CAPSULE,
            center,
            quaternionFromY(axis),
            f4(segmentRadius, 0.5 * chordLength, 0.0, 0.0),
            spec.contactOffsetM,
            segmentRadius + 0.5 * chordLength
        ));
    }

    const Vec3 firstAxis = normalized(nodes[1u] - nodes[0u]);
    const Vec3 lastAxis = normalized(
        nodes[spec.arcSegments] -
        nodes[spec.arcSegments - 1u]
    );
    massComponents.push_back(hemisphereComponent(
        nodes.front(),
        firstAxis * -1.0,
        needleRadiusAt(spec, 0.0),
        spec.densityKgPerM3.value
    ));
    massComponents.push_back(hemisphereComponent(
        nodes.back(),
        lastAxis,
        needleRadiusAt(spec, spec.arcLengthM.value),
        spec.densityKgPerM3.value
    ));
    const MassProperties mass = combineMass(massComponents);

    CurvedSutureNeedleAsset result;
    result.spec = spec;
    result.metadata.centerlineRadiusM = radius;
    result.metadata.representedArcLengthM = representedLength;
    result.metadata.maximumCenterlineErrorM =
        radius * (1.0 - std::cos(0.5 * angleStep));
    // Open ORBIT-Surgical commit 6e47534 uses scale=(0.4,0.4,0.4).
    // Its source USD extent is 50.163 x 98.948 x 4.129 mm.
    result.metadata.orbitReferenceMeshScale = 0.4;
    result.metadata.orbitReferenceScaledExtentM = {
        0.0200652,
        0.0395792,
        0.0016516,
    };
    result.metadata.tipTaperStartM =
        spec.arcLengthM.value - spec.tipTaperLengthM.value;
    result.metadata.swageEndM = spec.swageLengthM.value;
    result.metadata.graspZoneStartM =
        spec.arcLengthM.value * spec.graspZoneStartFraction.value;
    result.metadata.graspZoneEndM =
        spec.arcLengthM.value * spec.graspZoneEndFraction.value;
    const auto shapeAtArc = [&](const double arc) {
        return std::min(
            spec.arcSegments,
            static_cast<std::uint32_t>(std::ceil(
                arc / spec.arcLengthM.value *
                static_cast<double>(spec.arcSegments)
            ))
        );
    };
    result.metadata.swageShapeBegin = 0u;
    result.metadata.swageShapeEnd =
        shapeAtArc(result.metadata.swageEndM);
    result.metadata.graspShapeBegin =
        static_cast<std::uint32_t>(std::floor(
            spec.graspZoneStartFraction.value *
            static_cast<double>(spec.arcSegments)
        ));
    result.metadata.graspShapeEnd =
        shapeAtArc(result.metadata.graspZoneEndM);
    result.metadata.tipShapeBegin =
        static_cast<std::uint32_t>(std::floor(
            result.metadata.tipTaperStartM /
            spec.arcLengthM.value *
            static_cast<double>(spec.arcSegments)
        ));
    result.metadata.tipShapeEnd = spec.arcSegments;
    result.rigid = finalizeAsset(
        ids,
        mass,
        std::move(shapes),
        steelResearchMaterial(),
        "GS-21-scale curved suture needle"
    );
    return result;
}

SurgicalTrainingRingAsset makeSurgicalTrainingRingAsset(
    const SurgicalAssetIds& ids,
    const SurgicalTrainingRingSpec& spec
) {
    validateScalar(spec.majorRadiusM, "ring major radius");
    validateScalar(spec.tubeRadiusM, "ring tube radius");
    validateScalar(spec.densityKgPerM3, "ring density");
    if (spec.segments < 8u ||
        spec.segments > 256u ||
        spec.tubeRadiusM.value >= spec.majorRadiusM.value ||
        !(spec.contactOffsetM >= 0.0) ||
        !std::isfinite(spec.contactOffsetM)) {
        throw std::invalid_argument("training ring specification is invalid");
    }
    validateIds(ids, spec.segments);

    const double major = spec.majorRadiusM.value;
    const double tube = spec.tubeRadiusM.value;
    const double angleStep =
        2.0 * std::numbers::pi /
        static_cast<double>(spec.segments);
    std::vector<MRShapeGPU> shapes;
    shapes.reserve(spec.segments);
    for (std::uint32_t index = 0u; index < spec.segments; ++index) {
        const double angle0 =
            angleStep * static_cast<double>(index);
        const double angle1 =
            angleStep * static_cast<double>(index + 1u);
        const Vec3 start{
            major * std::cos(angle0),
            0.0,
            major * std::sin(angle0),
        };
        const Vec3 end{
            major * std::cos(angle1),
            0.0,
            major * std::sin(angle1),
        };
        const Vec3 segment = end - start;
        const double length = norm(segment);
        shapes.push_back(shapeRecord(
            ids,
            index,
            MR_SHAPE_CAPSULE,
            (start + end) * 0.5,
            quaternionFromY(segment),
            f4(tube, 0.5 * length, 0.0, 0.0),
            spec.contactOffsetM,
            tube + 0.5 * length
        ));
    }

    MassProperties mass;
    mass.volume =
        2.0 * std::numbers::pi * std::numbers::pi *
        major * tube * tube;
    mass.mass = mass.volume * spec.densityKgPerM3.value;
    mass.center = {};
    mass.inertiaAtCenter.m[0][0] =
        mass.mass *
        (0.5 * major * major + 0.625 * tube * tube);
    mass.inertiaAtCenter.m[1][1] =
        mass.mass *
        (major * major + 0.75 * tube * tube);
    mass.inertiaAtCenter.m[2][2] =
        mass.inertiaAtCenter.m[0][0];

    SurgicalTrainingRingAsset result;
    result.spec = spec;
    result.metadata.innerRadiusM = major - tube;
    result.metadata.outerRadiusM = major + tube;
    result.metadata.maximumCenterlineErrorM =
        major * (1.0 - std::cos(0.5 * angleStep));
    result.rigid = finalizeAsset(
        ids,
        mass,
        std::move(shapes),
        polymerResearchMaterial(),
        "surgical manipulation training ring"
    );
    return result;
}

SurgicalPegBlockAsset makeSurgicalPegBlockAsset(
    const SurgicalAssetIds& ids,
    const SurgicalPegBlockSpec& spec
) {
    validateScalar(spec.baseSizeXM, "peg-block base x");
    validateScalar(spec.baseSizeYM, "peg-block base y");
    validateScalar(spec.baseSizeZM, "peg-block base z");
    validateScalar(spec.pegRadiusM, "peg radius");
    validateScalar(spec.pegHeightM, "peg height");
    validateScalar(spec.pegSpacingXM, "peg spacing x");
    validateScalar(spec.pegSpacingZM, "peg spacing z");
    validateScalar(spec.densityKgPerM3, "peg-block density");
    const std::uint64_t pegCount64 =
        static_cast<std::uint64_t>(spec.columns) * spec.rows;
    if (spec.columns == 0u ||
        spec.rows == 0u ||
        pegCount64 > 63u ||
        spec.pegHeightM.value <=
            2.0 * spec.pegRadiusM.value + kMinimumDimension ||
        (spec.columns > 1u &&
         (static_cast<double>(spec.columns - 1u) *
              spec.pegSpacingXM.value +
          2.0 * spec.pegRadiusM.value >
          spec.baseSizeXM.value)) ||
        (spec.rows > 1u &&
         (static_cast<double>(spec.rows - 1u) *
              spec.pegSpacingZM.value +
          2.0 * spec.pegRadiusM.value >
          spec.baseSizeZM.value)) ||
        !(spec.contactOffsetM >= 0.0) ||
        !std::isfinite(spec.contactOffsetM)) {
        throw std::invalid_argument("surgical peg-block specification is invalid");
    }
    const std::uint32_t pegCount =
        static_cast<std::uint32_t>(pegCount64);
    validateIds(ids, static_cast<std::size_t>(pegCount) + 1u);

    const Vec3 baseSize{
        spec.baseSizeXM.value,
        spec.baseSizeYM.value,
        spec.baseSizeZM.value,
    };
    std::vector<MassComponent> components;
    components.reserve(
        static_cast<std::size_t>(pegCount) * 3u + 1u
    );
    components.push_back(boxComponent(
        {},
        baseSize,
        spec.densityKgPerM3.value
    ));

    std::vector<MRShapeGPU> shapes;
    shapes.reserve(static_cast<std::size_t>(pegCount) + 1u);
    shapes.push_back(shapeRecord(
        ids,
        0u,
        MR_SHAPE_BOX,
        {},
        f4(0.0, 0.0, 0.0, 1.0),
        f4(
            0.5 * baseSize.x,
            0.5 * baseSize.y,
            0.5 * baseSize.z,
            0.0
        ),
        spec.contactOffsetM,
        0.5 * norm(baseSize)
    ));

    SurgicalPegBlockMetadata metadata;
    metadata.baseShape = 0u;
    metadata.firstPegShape = 1u;
    metadata.pegCount = pegCount;
    metadata.pegCenters.reserve(pegCount);
    const double pegCylinderLength =
        spec.pegHeightM.value - 2.0 * spec.pegRadiusM.value;
    const double pegHalfLength = 0.5 * pegCylinderLength;
    for (std::uint32_t row = 0u; row < spec.rows; ++row) {
        for (std::uint32_t column = 0u;
             column < spec.columns;
             ++column) {
            const Vec3 center{
                (static_cast<double>(column) -
                 0.5 * static_cast<double>(spec.columns - 1u)) *
                    spec.pegSpacingXM.value,
                0.5 * spec.baseSizeYM.value +
                    0.5 * spec.pegHeightM.value,
                (static_cast<double>(row) -
                 0.5 * static_cast<double>(spec.rows - 1u)) *
                    spec.pegSpacingZM.value,
            };
            const Vec3 lowerCapCenter{
                center.x,
                center.y - pegHalfLength,
                center.z,
            };
            const Vec3 upperCapCenter{
                center.x,
                center.y + pegHalfLength,
                center.z,
            };
            components.push_back(cylinderComponent(
                center,
                {0.0, 1.0, 0.0},
                spec.pegRadiusM.value,
                pegCylinderLength,
                spec.densityKgPerM3.value
            ));
            components.push_back(hemisphereComponent(
                lowerCapCenter,
                {0.0, -1.0, 0.0},
                spec.pegRadiusM.value,
                spec.densityKgPerM3.value
            ));
            components.push_back(hemisphereComponent(
                upperCapCenter,
                {0.0, 1.0, 0.0},
                spec.pegRadiusM.value,
                spec.densityKgPerM3.value
            ));
            const std::uint32_t shapeIndex =
                1u + row * spec.columns + column;
            shapes.push_back(shapeRecord(
                ids,
                shapeIndex,
                MR_SHAPE_CAPSULE,
                center,
                f4(0.0, 0.0, 0.0, 1.0),
                f4(
                    spec.pegRadiusM.value,
                    pegHalfLength,
                    0.0,
                    0.0
                ),
                spec.contactOffsetM,
                spec.pegRadiusM.value + pegHalfLength
            ));
            metadata.pegCenters.push_back({
                center.x,
                center.y,
                center.z,
            });
        }
    }
    const MassProperties mass = combineMass(components);
    for (std::array<double, 3>& center : metadata.pegCenters) {
        center[0] -= mass.center.x;
        center[1] -= mass.center.y;
        center[2] -= mass.center.z;
    }

    SurgicalPegBlockAsset result;
    result.spec = spec;
    result.metadata = std::move(metadata);
    result.rigid = finalizeAsset(
        ids,
        mass,
        std::move(shapes),
        polymerResearchMaterial(),
        "surgical manipulation peg block"
    );
    return result;
}

} // namespace metalrobo
