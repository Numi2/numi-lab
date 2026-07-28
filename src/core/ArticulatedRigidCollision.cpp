#include "metalrobo/ArticulatedRigidCollision.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr double kQuaternionTolerance = 2.0e-4;
constexpr double kInverseInertiaSymmetryTolerance = 2.0e-5;

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

struct Mat3 {
    double m[3][3]{};
};

struct ShapeBinding {
    bool articulated = false;
    std::uint32_t sourceShape = 0u;
    std::uint32_t sourceBody = 0u;
};

struct SourcePoint {
    std::uint64_t collisionPairKey = 0u;
    std::uint64_t collisionFeatureKey = 0u;
    std::uint32_t articulatedShape = 0u;
    std::uint32_t rigidShape = 0u;
    std::uint32_t articulatedBody = 0u;
    std::uint32_t rigidBody = 0u;
    std::uint32_t articulatedGeneration = 0u;
    std::uint32_t rigidGeneration = 0u;
    std::uint32_t articulatedFeature = 0u;
    std::uint32_t rigidFeature = 0u;
    std::uint32_t manifold = 0u;
    std::uint32_t point = 0u;
    std::uint32_t lifetime = 0u;
    std::array<double, 3> localWitnessArticulated{};
    std::array<double, 3> localWitnessRigid{};
};

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const mr_float4 value) {
    return
        finite(value.x) &&
        finite(value.y) &&
        finite(value.z) &&
        finite(value.w);
}

template <std::size_t Size>
bool finite(const std::array<double, Size>& values) {
    return std::ranges::all_of(
        values,
        [](const double value) {
            return finite(value);
        }
    );
}

Vec3 vector(const mr_float4 value) {
    return {value.x, value.y, value.z};
}

Vec3 vector(const std::array<double, 3>& value) {
    return {value[0], value[1], value[2]};
}

std::array<double, 3> array(const Vec3 value) {
    return {value.x, value.y, value.z};
}

Vec3 operator-(const Vec3 left, const Vec3 right) {
    return {
        left.x - right.x,
        left.y - right.y,
        left.z - right.z,
    };
}

Vec3 operator*(const Mat3& matrix, const Vec3 value) {
    return {
        matrix.m[0][0] * value.x +
            matrix.m[0][1] * value.y +
            matrix.m[0][2] * value.z,
        matrix.m[1][0] * value.x +
            matrix.m[1][1] * value.y +
            matrix.m[1][2] * value.z,
        matrix.m[2][0] * value.x +
            matrix.m[2][1] * value.y +
            matrix.m[2][2] * value.z,
    };
}

Mat3 operator*(const Mat3& left, const Mat3& right) {
    Mat3 result{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            for (std::size_t inner = 0u; inner < 3u; ++inner) {
                result.m[row][column] +=
                    left.m[row][inner] * right.m[inner][column];
            }
        }
    }
    return result;
}

Mat3 transpose(const Mat3& value) {
    Mat3 result{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            result.m[row][column] = value.m[column][row];
        }
    }
    return result;
}

Mat3 matrix(
    const mr_float4 row0,
    const mr_float4 row1,
    const mr_float4 row2
) {
    return {{
        {row0.x, row0.y, row0.z},
        {row1.x, row1.y, row1.z},
        {row2.x, row2.y, row2.z},
    }};
}

Mat3 rotationMatrix(const Quaternion q) {
    const double xx = q.x * q.x;
    const double yy = q.y * q.y;
    const double zz = q.z * q.z;
    const double xy = q.x * q.y;
    const double xz = q.x * q.z;
    const double yz = q.y * q.z;
    const double xw = q.x * q.w;
    const double yw = q.y * q.w;
    const double zw = q.z * q.w;
    return {{
        {
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - zw),
            2.0 * (xz + yw),
        },
        {
            2.0 * (xy + zw),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - xw),
        },
        {
            2.0 * (xz - yw),
            2.0 * (yz + xw),
            1.0 - 2.0 * (xx + yy),
        },
    }};
}

bool checkedQuaternion(
    const mr_float4 source,
    Quaternion& result
) {
    if (!finite(source)) {
        return false;
    }
    const double squared =
        static_cast<double>(source.x) * source.x +
        static_cast<double>(source.y) * source.y +
        static_cast<double>(source.z) * source.z +
        static_cast<double>(source.w) * source.w;
    if (!(squared > 0.0) ||
        !finite(squared) ||
        std::abs(squared - 1.0) > kQuaternionTolerance) {
        return false;
    }
    const double inverseNorm = 1.0 / std::sqrt(squared);
    result = {
        source.x * inverseNorm,
        source.y * inverseNorm,
        source.z * inverseNorm,
        source.w * inverseNorm,
    };
    return true;
}

Vec3 inverseRotate(
    const Quaternion quaternion,
    const Vec3 value
) {
    return transpose(rotationMatrix(quaternion)) * value;
}

bool representableAsFloat(const double value) {
    return
        finite(value) &&
        std::abs(value) <=
            static_cast<double>(std::numeric_limits<float>::max());
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

bool positiveDefiniteInverseInertia(
    const MRBodyStateGPU& body
) {
    double values[3][3]{
        {
            body.inverseInertiaWorldRow0.x,
            body.inverseInertiaWorldRow0.y,
            body.inverseInertiaWorldRow0.z,
        },
        {
            body.inverseInertiaWorldRow1.x,
            body.inverseInertiaWorldRow1.y,
            body.inverseInertiaWorldRow1.z,
        },
        {
            body.inverseInertiaWorldRow2.x,
            body.inverseInertiaWorldRow2.y,
            body.inverseInertiaWorldRow2.z,
        },
    };
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            if (!finite(values[row][column])) {
                return false;
            }
        }
    }
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = row + 1u;
             column < 3u;
             ++column) {
            const double scale =
                1.0 + std::max(
                    std::abs(values[row][column]),
                    std::abs(values[column][row])
                );
            if (std::abs(
                    values[row][column] -
                    values[column][row]
                ) > kInverseInertiaSymmetryTolerance * scale) {
                return false;
            }
            const double average = 0.5 * (
                values[row][column] +
                values[column][row]
            );
            values[row][column] = average;
            values[column][row] = average;
        }
    }
    double lower[3][3]{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u;
             column <= row;
             ++column) {
            double value = values[row][column];
            for (std::size_t inner = 0u;
                 inner < column;
                 ++inner) {
                value -=
                    lower[row][inner] *
                    lower[column][inner];
            }
            if (row == column) {
                if (!(value > 0.0) || !finite(value)) {
                    return false;
                }
                lower[row][column] = std::sqrt(value);
            } else {
                lower[row][column] =
                    value / lower[column][column];
            }
        }
    }
    return true;
}

bool validRigidBody(const MRBodyStateGPU& body) {
    Quaternion orientation;
    return
        body.flagsAndIndices[0] == MR_MOTION_DYNAMIC &&
        body.flagsAndIndices[1] == MR_INVALID_INDEX &&
        finite(body.position) &&
        finite(body.linearVelocityAndInverseMass) &&
        finite(body.angularVelocity) &&
        body.linearVelocityAndInverseMass.w > 0.0f &&
        checkedQuaternion(body.orientation, orientation) &&
        positiveDefiniteInverseInertia(body);
}

MRStepStatusCode dynamicsCode(
    const ArticulatedDynamicsStatus status
) {
    switch (status) {
    case ArticulatedDynamicsStatus::success:
        return MR_STEP_SUCCESS;
    case ArticulatedDynamicsStatus::unsupportedTopology:
        return MR_STEP_UNSUPPORTED;
    case ArticulatedDynamicsStatus::massMatrixNotPositiveDefinite:
        return MR_STEP_FACTORIZATION_FAILED;
    case ArticulatedDynamicsStatus::nonlinearSolveFailed:
        return MR_STEP_DID_NOT_CONVERGE;
    case ArticulatedDynamicsStatus::nonfiniteResult:
        return MR_STEP_NONFINITE_RESULT;
    case ArticulatedDynamicsStatus::invalidModel:
    case ArticulatedDynamicsStatus::invalidDimensions:
    case ArticulatedDynamicsStatus::nonfiniteInput:
    case ArticulatedDynamicsStatus::invalidQuaternion:
    case ArticulatedDynamicsStatus::jointLimitViolation:
    case ArticulatedDynamicsStatus::bodySpeedLimitViolation:
        return MR_STEP_NONFINITE_INPUT;
    }
    return MR_STEP_NONFINITE_RESULT;
}

ArticulatedRigidCollisionResult fail(
    ArticulatedRigidCollisionDiagnostics diagnostics,
    const ArticulatedRigidCollisionStatus status,
    const MRStepStatusCode code,
    std::string reason
) {
    diagnostics.status = status;
    diagnostics.code = code;
    diagnostics.failure = std::move(reason);
    ArticulatedRigidCollisionResult result;
    result.diagnostics = std::move(diagnostics);
    return result;
}

bool writeArticulatedState(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const ArticulatedBodyKinematics& source,
    MRBodyStateGPU& result
) {
    if (source.bodyIndex >= model.bodies.size()) {
        return false;
    }
    const MRBodyPropertiesGPU& properties =
        model.bodies[source.bodyIndex];
    if (properties.articulationIndex != articulationIndex ||
        properties.motionType != MR_MOTION_DYNAMIC ||
        !(properties.massAndInverseMass.y > 0.0f) ||
        !finite(properties.massAndInverseMass) ||
        !std::ranges::all_of(
            source.centerOfMassPosition,
            representableAsFloat
        ) ||
        !std::ranges::all_of(
            source.orientation,
            representableAsFloat
        ) ||
        !std::ranges::all_of(
            source.linearVelocity,
            representableAsFloat
        ) ||
        !std::ranges::all_of(
            source.angularVelocity,
            representableAsFloat
        )) {
        return false;
    }

    result = {};
    result.position = f4(
        source.centerOfMassPosition[0],
        source.centerOfMassPosition[1],
        source.centerOfMassPosition[2],
        1.0
    );
    result.orientation = f4(
        source.orientation[0],
        source.orientation[1],
        source.orientation[2],
        source.orientation[3]
    );
    result.linearVelocityAndInverseMass = f4(
        source.linearVelocity[0],
        source.linearVelocity[1],
        source.linearVelocity[2],
        properties.massAndInverseMass.y
    );
    result.angularVelocity = f4(
        source.angularVelocity[0],
        source.angularVelocity[1],
        source.angularVelocity[2]
    );

    const Quaternion orientation{
        source.orientation[0],
        source.orientation[1],
        source.orientation[2],
        source.orientation[3],
    };
    const Mat3 rotation = rotationMatrix(orientation);
    const Mat3 inverseBody = matrix(
        properties.inverseInertiaRow0,
        properties.inverseInertiaRow1,
        properties.inverseInertiaRow2
    );
    const Mat3 inverseWorld =
        rotation * inverseBody * transpose(rotation);
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            if (!representableAsFloat(
                    inverseWorld.m[row][column]
                )) {
                return false;
            }
        }
    }
    result.inverseInertiaWorldRow0 = f4(
        inverseWorld.m[0][0],
        inverseWorld.m[0][1],
        inverseWorld.m[0][2]
    );
    result.inverseInertiaWorldRow1 = f4(
        inverseWorld.m[1][0],
        inverseWorld.m[1][1],
        inverseWorld.m[1][2]
    );
    result.inverseInertiaWorldRow2 = f4(
        inverseWorld.m[2][0],
        inverseWorld.m[2][1],
        inverseWorld.m[2][2]
    );
    result.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
    result.flagsAndIndices[1] = articulationIndex;
    result.flagsAndIndices[2] = source.bodyIndex;
    return
        finite(result.position) &&
        finite(result.orientation) &&
        finite(result.linearVelocityAndInverseMass) &&
        finite(result.angularVelocity) &&
        finite(result.inverseInertiaWorldRow0) &&
        finite(result.inverseInertiaWorldRow1) &&
        finite(result.inverseInertiaWorldRow2);
}

std::uint64_t packedKey(
    const std::uint32_t high,
    const std::uint32_t low
) {
    return
        (static_cast<std::uint64_t>(high) << 32u) |
        static_cast<std::uint64_t>(low);
}

bool sourcePointLess(
    const SourcePoint& left,
    const SourcePoint& right
) {
    return std::tie(
        left.collisionPairKey,
        left.collisionFeatureKey
    ) < std::tie(
        right.collisionPairKey,
        right.collisionFeatureKey
    );
}

ArticulatedRigidContactKey publicKey(
    const SourcePoint& source,
    const std::uint32_t articulationIndex
) {
    return {
        .pairKey = packedKey(
            source.articulatedShape,
            source.rigidShape
        ),
        .featureKey = packedKey(
            source.articulatedFeature,
            source.rigidFeature
        ),
        .articulationIndex = articulationIndex,
        .articulatedGeneration =
            source.articulatedGeneration,
        .rigidGeneration = source.rigidGeneration,
    };
}

} // namespace

ArticulatedRigidCollisionResult
collideArticulatedRigidContactsCpu(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const double> v,
    const std::span<const MRShapeGPU> rigidShapes,
    const std::span<const MRMaterialGPU> rigidMaterials,
    const std::span<const MRBodyStateGPU> rigidBodies,
    PersistentManifoldCache& manifoldCache,
    const ArticulatedRigidCollisionConfig& config,
    const std::span<const ArticulatedRigidContactWarmStart>
        warmStarts
) {
    ArticulatedRigidCollisionDiagnostics diagnostics;
    diagnostics.articulationIndex = articulationIndex;
    diagnostics.rigidShapeCount =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            rigidShapes.size(),
            std::numeric_limits<std::uint32_t>::max()
        ));
    diagnostics.suppliedWarmStartCount =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            warmStarts.size(),
            std::numeric_limits<std::uint32_t>::max()
        ));

    std::string modelFailure;
    if (articulationIndex >= model.articulations.size() ||
        !model.valid(&modelFailure)) {
        return fail(
            diagnostics,
            ArticulatedRigidCollisionStatus::invalidModel,
            MR_STEP_NONFINITE_INPUT,
            articulationIndex >= model.articulations.size()
                ? "articulation index is outside the compiled model"
                : "compiled engine model is invalid: " + modelFailure
        );
    }
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    const std::size_t maximumSize =
        std::numeric_limits<std::size_t>::max();
    if (q.size() != articulation.nq ||
        v.size() != articulation.nv ||
        rigidShapes.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        rigidMaterials.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        rigidBodies.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        warmStarts.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        articulation.bodyCount >
            maximumSize - rigidBodies.size()) {
        return fail(
            diagnostics,
            ArticulatedRigidCollisionStatus::invalidDimensions,
            MR_STEP_NONFINITE_INPUT,
            "articulated or rigid stream dimensions are inconsistent"
        );
    }
    for (std::size_t body = 0u;
         body < rigidBodies.size();
         ++body) {
        if (!validRigidBody(rigidBodies[body])) {
            return fail(
                diagnostics,
                ArticulatedRigidCollisionStatus::invalidRigidBody,
                MR_STEP_NONFINITE_INPUT,
                "rigid body " + std::to_string(body) +
                    " is not finite, dynamic, independent, "
                    "unit-oriented, or positive definite"
            );
        }
    }
    for (std::size_t shape = 0u;
         shape < rigidShapes.size();
         ++shape) {
        if (rigidShapes[shape].bodyIndex >= rigidBodies.size() ||
            rigidShapes[shape].materialIndex >=
                rigidMaterials.size() ||
            rigidShapes[shape].slotGeneration == 0u) {
            return fail(
                diagnostics,
                ArticulatedRigidCollisionStatus::invalidRigidShape,
                MR_STEP_NONFINITE_INPUT,
                "rigid shape " + std::to_string(shape) +
                    " has an invalid body, material, or zero generation"
            );
        }
    }
    for (std::size_t warm = 0u;
         warm < warmStarts.size();
         ++warm) {
        if (!finite(warmStarts[warm].worldImpulseOnRigid)) {
            return fail(
                diagnostics,
                ArticulatedRigidCollisionStatus::invalidWarmStart,
                MR_STEP_NONFINITE_INPUT,
                "warm-start impulse " + std::to_string(warm) +
                    " is non-finite"
            );
        }
        for (std::size_t previous = 0u;
             previous < warm;
             ++previous) {
            if (warmStarts[previous].key ==
                warmStarts[warm].key) {
                return fail(
                    diagnostics,
                    ArticulatedRigidCollisionStatus::invalidWarmStart,
                    MR_STEP_NONFINITE_INPUT,
                    "warm-start keys must be unique"
                );
            }
        }
    }

    std::vector<ArticulatedBodyKinematics> bodyKinematics(
        articulation.bodyCount
    );
    diagnostics.kinematics =
        computeArticulatedBodyKinematics(
            model,
            articulationIndex,
            q,
            v,
            bodyKinematics,
            config.dynamics
        );
    if (!diagnostics.kinematics.succeeded()) {
        return fail(
            diagnostics,
            ArticulatedRigidCollisionStatus::kinematicsFailure,
            dynamicsCode(diagnostics.kinematics.status),
            "articulated body kinematics failed"
        );
    }

    std::vector<MRBodyStateGPU> collisionBodies(
        static_cast<std::size_t>(articulation.bodyCount) +
            rigidBodies.size()
    );
    for (const ArticulatedBodyKinematics& body :
         bodyKinematics) {
        if (body.bodyIndex < articulation.firstBody ||
            body.bodyIndex >=
                articulation.firstBody +
                    articulation.bodyCount ||
            !writeArticulatedState(
                model,
                articulationIndex,
                body,
                collisionBodies[
                    body.bodyIndex - articulation.firstBody
                ]
            )) {
            return fail(
                diagnostics,
                ArticulatedRigidCollisionStatus::nonfiniteResult,
                MR_STEP_NONFINITE_RESULT,
                "articulated collision-state projection failed"
            );
        }
    }
    std::ranges::copy(
        rigidBodies,
        collisionBodies.begin() + articulation.bodyCount
    );

    std::vector<MRShapeGPU> collisionShapes;
    std::vector<ShapeBinding> shapeBindings;
    std::vector<MRMaterialGPU> collisionMaterials;
    std::vector<std::uint32_t> articulatedMaterialMap(
        model.materials.size(),
        MR_INVALID_INDEX
    );
    std::vector<std::uint32_t> rigidMaterialMap(
        rigidMaterials.size(),
        MR_INVALID_INDEX
    );
    collisionShapes.reserve(model.shapes.size() + rigidShapes.size());
    shapeBindings.reserve(model.shapes.size() + rigidShapes.size());

    const auto appendMaterial = [&collisionMaterials](
        const MRMaterialGPU& material,
        std::uint32_t& mapped
    ) -> bool {
        if (mapped != MR_INVALID_INDEX) {
            return true;
        }
        if (collisionMaterials.size() >=
            std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        mapped = static_cast<std::uint32_t>(
            collisionMaterials.size()
        );
        collisionMaterials.push_back(material);
        return true;
    };

    const std::uint64_t bodyBegin = articulation.firstBody;
    const std::uint64_t bodyEnd =
        bodyBegin + articulation.bodyCount;
    for (std::size_t shapeIndex = 0u;
         shapeIndex < model.shapes.size();
         ++shapeIndex) {
        const MRShapeGPU& source = model.shapes[shapeIndex];
        if (source.bodyIndex < bodyBegin ||
            source.bodyIndex >= bodyEnd) {
            continue;
        }
        if (source.materialIndex >= model.materials.size() ||
            source.slotGeneration == 0u ||
            !appendMaterial(
                model.materials[source.materialIndex],
                articulatedMaterialMap[source.materialIndex]
            )) {
            return fail(
                diagnostics,
                ArticulatedRigidCollisionStatus::invalidModel,
                MR_STEP_NONFINITE_INPUT,
                "articulated collision shape material or "
                "generation is invalid"
            );
        }
        MRShapeGPU rebased = source;
        rebased.bodyIndex =
            source.bodyIndex - articulation.firstBody;
        rebased.materialIndex =
            articulatedMaterialMap[source.materialIndex];
        collisionShapes.push_back(rebased);
        shapeBindings.push_back({
            .articulated = true,
            .sourceShape =
                static_cast<std::uint32_t>(shapeIndex),
            .sourceBody = source.bodyIndex,
        });
    }
    const std::size_t articulatedShapeCount =
        collisionShapes.size();
    diagnostics.articulatedShapeCount =
        static_cast<std::uint32_t>(articulatedShapeCount);

    for (std::size_t shapeIndex = 0u;
         shapeIndex < rigidShapes.size();
         ++shapeIndex) {
        const MRShapeGPU& source = rigidShapes[shapeIndex];
        if (!appendMaterial(
                rigidMaterials[source.materialIndex],
                rigidMaterialMap[source.materialIndex]
            )) {
            return fail(
                diagnostics,
                ArticulatedRigidCollisionStatus::invalidDimensions,
                MR_STEP_NONFINITE_INPUT,
                "combined material stream exceeds index capacity"
            );
        }
        MRShapeGPU rebased = source;
        const std::size_t rebasedBody =
            static_cast<std::size_t>(articulation.bodyCount) +
            source.bodyIndex;
        if (rebasedBody >
            std::numeric_limits<std::uint32_t>::max()) {
            return fail(
                diagnostics,
                ArticulatedRigidCollisionStatus::invalidDimensions,
                MR_STEP_NONFINITE_INPUT,
                "combined body stream exceeds index capacity"
            );
        }
        rebased.bodyIndex =
            static_cast<std::uint32_t>(rebasedBody);
        rebased.materialIndex =
            rigidMaterialMap[source.materialIndex];
        collisionShapes.push_back(rebased);
        shapeBindings.push_back({
            .articulated = false,
            .sourceShape =
                static_cast<std::uint32_t>(shapeIndex),
            .sourceBody = source.bodyIndex,
        });
    }
    if (collisionShapes.size() >
        std::numeric_limits<std::uint32_t>::max()) {
        return fail(
            diagnostics,
            ArticulatedRigidCollisionStatus::invalidDimensions,
            MR_STEP_NONFINITE_INPUT,
            "combined shape stream exceeds index capacity"
        );
    }

    const auto sameSetPairCount = [](const std::size_t count) {
        return count > 1u
            ? count * (count - 1u) / 2u
            : 0u;
    };
    if (
        (
            articulatedShapeCount > 1u &&
            articulatedShapeCount - 1u >
                maximumSize / articulatedShapeCount
        ) ||
        (
            rigidShapes.size() > 1u &&
            rigidShapes.size() - 1u >
                maximumSize / rigidShapes.size()
        )
    ) {
        return fail(
            diagnostics,
            ArticulatedRigidCollisionStatus::invalidDimensions,
            MR_STEP_NONFINITE_INPUT,
            "same-set exclusion count overflow"
        );
    }
    const std::size_t articulatedExclusions =
        sameSetPairCount(articulatedShapeCount);
    const std::size_t rigidExclusions =
        sameSetPairCount(rigidShapes.size());
    if (articulatedExclusions >
        maximumSize - rigidExclusions) {
        return fail(
            diagnostics,
            ArticulatedRigidCollisionStatus::invalidDimensions,
            MR_STEP_NONFINITE_INPUT,
            "same-set exclusion stream exceeds capacity"
        );
    }
    std::vector<CollisionPairExclusion> exclusions;
    exclusions.reserve(
        articulatedExclusions + rigidExclusions
    );
    for (std::uint32_t first = 0u;
         first < articulatedShapeCount;
         ++first) {
        for (std::uint32_t second = first + 1u;
             second < articulatedShapeCount;
             ++second) {
            exclusions.push_back({first, second});
        }
    }
    const std::uint32_t rigidColliderOffset =
        static_cast<std::uint32_t>(articulatedShapeCount);
    for (std::uint32_t first = 0u;
         first < rigidShapes.size();
         ++first) {
        for (std::uint32_t second = first + 1u;
             second < rigidShapes.size();
             ++second) {
            exclusions.push_back({
                rigidColliderOffset + first,
                rigidColliderOffset + second,
            });
        }
    }

    PersistentManifoldCache workingCache = manifoldCache;
    const CollisionFrame collision = collideCpuReference(
        collisionShapes,
        collisionBodies,
        config.collision,
        workingCache,
        exclusions
    );
    diagnostics.collision = collision.diagnostics;
    if (!collision.succeeded()) {
        return fail(
            diagnostics,
            ArticulatedRigidCollisionStatus::collisionFailure,
            collision.diagnostics.code,
            "cross-system collision generation failed"
        );
    }

    ContactAssemblyResult assembly = assembleContactConstraints(
        collision,
        collisionShapes,
        collisionMaterials,
        collisionBodies,
        config.contact.contactCapacity
    );
    diagnostics.assembly = assembly.diagnostics;
    diagnostics.maximumPenetration =
        assembly.diagnostics.maximumPenetration;
    if (!assembly.diagnostics.succeeded()) {
        return fail(
            diagnostics,
            ArticulatedRigidCollisionStatus::
                contactAssemblyFailure,
            assembly.diagnostics.code,
            "common contact assembly failed"
        );
    }

    // The coupled exact-cone solve exposes one coefficient. Select the
    // already-mixed dynamic coefficient explicitly; never erase rolling or
    // torsional material semantics.
    for (MRContactConstraintGPU& contact :
         assembly.constraints) {
        if (contact.friction.z != 0.0f ||
            contact.friction.w != 0.0f) {
            return fail(
                diagnostics,
                ArticulatedRigidCollisionStatus::
                    contactAdaptationFailure,
                MR_STEP_UNSUPPORTED,
                "rolling or torsional friction is unsupported by "
                "the coupled exact-cone contact"
            );
        }
        contact.friction.x = contact.friction.y;
    }

    std::vector<MRBodyStateGPU> adaptationBodies =
        collisionBodies;
    for (std::size_t body = articulation.bodyCount;
         body < adaptationBodies.size();
         ++body) {
        MRBodyStateGPU& state = adaptationBodies[body];
        state.linearVelocityAndInverseMass =
            f4(0.0, 0.0, 0.0, 0.0);
        state.angularVelocity = f4(0.0, 0.0, 0.0, 0.0);
        state.inverseInertiaWorldRow0 =
            f4(0.0, 0.0, 0.0, 0.0);
        state.inverseInertiaWorldRow1 =
            f4(0.0, 0.0, 0.0, 0.0);
        state.inverseInertiaWorldRow2 =
            f4(0.0, 0.0, 0.0, 0.0);
        state.flagsAndIndices[0] = MR_MOTION_STATIC;
        state.flagsAndIndices[1] = MR_INVALID_INDEX;
        state.flagsAndIndices[2] = MR_INVALID_INDEX;
    }
    const ArticulatedCollisionResult adaptation =
        adaptArticulatedContactConstraints(
            model,
            articulationIndex,
            assembly.constraints,
            adaptationBodies,
            config.contact
        );
    diagnostics.adaptation = adaptation.diagnostics;
    if (!adaptation.succeeded()) {
        return fail(
            diagnostics,
            ArticulatedRigidCollisionStatus::
                contactAdaptationFailure,
            adaptation.diagnostics.code,
            "common-to-articulated contact adaptation failed"
        );
    }

    std::vector<SourcePoint> sourcePoints;
    sourcePoints.reserve(assembly.constraints.size());
    if (collision.manifoldPoints.size() !=
        4u * collision.manifoldHeaders.size()) {
        return fail(
            diagnostics,
            ArticulatedRigidCollisionStatus::nonfiniteResult,
            MR_STEP_NONFINITE_RESULT,
            "collision manifold payload dimensions changed"
        );
    }
    for (std::size_t manifold = 0u;
         manifold < collision.manifoldHeaders.size();
         ++manifold) {
        const MRManifoldHeaderGPU& header =
            collision.manifoldHeaders[manifold];
        const std::uint32_t colliderA =
            header.pairAndCount[1];
        const std::uint32_t colliderB =
            header.pairAndCount[2];
        const std::uint32_t pointCount =
            header.pairAndCount[3];
        if (colliderA >= shapeBindings.size() ||
            colliderB >= shapeBindings.size() ||
            pointCount == 0u ||
            pointCount > 4u ||
            shapeBindings[colliderA].articulated ==
                shapeBindings[colliderB].articulated) {
            return fail(
                diagnostics,
                ArticulatedRigidCollisionStatus::nonfiniteResult,
                MR_STEP_NONFINITE_RESULT,
                "collision emitted a malformed or non-cross manifold"
            );
        }
        const bool articulatedIsA =
            shapeBindings[colliderA].articulated;
        const std::uint32_t articulatedCollider =
            articulatedIsA ? colliderA : colliderB;
        const std::uint32_t rigidCollider =
            articulatedIsA ? colliderB : colliderA;
        const ShapeBinding& articulatedBinding =
            shapeBindings[articulatedCollider];
        const ShapeBinding& rigidBinding =
            shapeBindings[rigidCollider];
        const MRShapeGPU& articulatedShape =
            collisionShapes[articulatedCollider];
        const MRShapeGPU& rigidShape =
            collisionShapes[rigidCollider];
        for (std::uint32_t pointIndex = 0u;
             pointIndex < pointCount;
             ++pointIndex) {
            const MRManifoldPointGPU& point =
                collision.manifoldPoints[
                    manifold * 4u + pointIndex
                ];
            const std::uint32_t featureA =
                point.featureAndLife[0];
            const std::uint32_t featureB =
                point.featureAndLife[1];
            SourcePoint source;
            source.collisionPairKey =
                packedKey(colliderA, colliderB);
            source.collisionFeatureKey =
                packedKey(featureA, featureB);
            source.articulatedShape =
                articulatedBinding.sourceShape;
            source.rigidShape = rigidBinding.sourceShape;
            source.articulatedBody =
                articulatedBinding.sourceBody;
            source.rigidBody = rigidBinding.sourceBody;
            source.articulatedGeneration =
                articulatedShape.slotGeneration;
            source.rigidGeneration =
                rigidShape.slotGeneration;
            source.articulatedFeature =
                articulatedIsA ? featureA : featureB;
            source.rigidFeature =
                articulatedIsA ? featureB : featureA;
            source.manifold =
                static_cast<std::uint32_t>(manifold);
            source.point = pointIndex;
            source.lifetime = point.featureAndLife[2];
            source.localWitnessArticulated = {
                articulatedIsA
                    ? point.localAnchorA.x
                    : point.localAnchorB.x,
                articulatedIsA
                    ? point.localAnchorA.y
                    : point.localAnchorB.y,
                articulatedIsA
                    ? point.localAnchorA.z
                    : point.localAnchorB.z,
            };
            source.localWitnessRigid = {
                articulatedIsA
                    ? point.localAnchorB.x
                    : point.localAnchorA.x,
                articulatedIsA
                    ? point.localAnchorB.y
                    : point.localAnchorA.y,
                articulatedIsA
                    ? point.localAnchorB.z
                    : point.localAnchorA.z,
            };
            sourcePoints.push_back(source);
        }
    }
    std::ranges::sort(sourcePoints, sourcePointLess);
    if (sourcePoints.size() != assembly.constraints.size()) {
        return fail(
            diagnostics,
            ArticulatedRigidCollisionStatus::nonfiniteResult,
            MR_STEP_NONFINITE_RESULT,
            "manifold/source contact count mismatch"
        );
    }
    for (std::size_t contactIndex = 0u;
         contactIndex < sourcePoints.size();
         ++contactIndex) {
        const SourcePoint& source = sourcePoints[contactIndex];
        const MRContactConstraintGPU& contact =
            assembly.constraints[contactIndex];
        if (source.collisionPairKey != contact.pairKey ||
            source.collisionFeatureKey != contact.featureKey ||
            (
                contactIndex > 0u &&
                source.collisionPairKey ==
                    sourcePoints[contactIndex - 1u].
                        collisionPairKey &&
                source.collisionFeatureKey ==
                    sourcePoints[contactIndex - 1u].
                        collisionFeatureKey
            )) {
            return fail(
                diagnostics,
                ArticulatedRigidCollisionStatus::nonfiniteResult,
                MR_STEP_NONFINITE_RESULT,
                "manifold keys are ambiguous or disagree with "
                "common contact assembly"
            );
        }
    }
    if (adaptation.contacts.size() !=
            assembly.constraints.size() ||
        adaptation.sourceConstraintIndices.size() !=
            adaptation.contacts.size()) {
        return fail(
            diagnostics,
            ArticulatedRigidCollisionStatus::nonfiniteResult,
            MR_STEP_NONFINITE_RESULT,
            "articulated adaptation/source dimensions disagree"
        );
    }

    std::vector<CoupledArticulatedRigidContact> contacts;
    std::vector<ArticulatedRigidContactMetadata> metadata;
    contacts.reserve(adaptation.contacts.size());
    metadata.reserve(adaptation.contacts.size());
    for (std::size_t adaptedIndex = 0u;
         adaptedIndex < adaptation.contacts.size();
         ++adaptedIndex) {
        const std::uint32_t sourceIndex =
            adaptation.sourceConstraintIndices[adaptedIndex];
        if (sourceIndex >= sourcePoints.size()) {
            return fail(
                diagnostics,
                ArticulatedRigidCollisionStatus::nonfiniteResult,
                MR_STEP_NONFINITE_RESULT,
                "adapted source index is out of range"
            );
        }
        const ArticulatedContact& articulatedContact =
            adaptation.contacts[adaptedIndex];
        const MRContactConstraintGPU& commonContact =
            assembly.constraints[sourceIndex];
        const SourcePoint& source = sourcePoints[sourceIndex];
        if (source.rigidBody >= rigidBodies.size()) {
            return fail(
                diagnostics,
                ArticulatedRigidCollisionStatus::nonfiniteResult,
                MR_STEP_NONFINITE_RESULT,
                "rigid source body is out of range"
            );
        }
        Quaternion rigidOrientation;
        if (!checkedQuaternion(
                rigidBodies[source.rigidBody].orientation,
                rigidOrientation
            )) {
            return fail(
                diagnostics,
                ArticulatedRigidCollisionStatus::nonfiniteResult,
                MR_STEP_NONFINITE_RESULT,
                "rigid orientation changed after validation"
            );
        }
        const Vec3 worldPoint =
            vector(commonContact.pointAndSeparation);
        const Vec3 rigidLocalPoint = inverseRotate(
            rigidOrientation,
            worldPoint -
                vector(rigidBodies[source.rigidBody].position)
        );

        CoupledArticulatedRigidContact contact;
        contact.articulatedBody = articulatedContact.bodyA;
        contact.rigidBody = source.rigidBody;
        contact.localPointArticulated =
            articulatedContact.localPointA;
        contact.localPointRigid = array(rigidLocalPoint);
        contact.normal = articulatedContact.normal;
        contact.tangentU = articulatedContact.tangentU;
        contact.tangentV = articulatedContact.tangentV;
        contact.targetVelocity =
            articulatedContact.targetVelocity;
        contact.regularization =
            articulatedContact.regularization;
        contact.warmImpulse = {};
        contact.friction = articulatedContact.friction;

        const ArticulatedRigidContactKey key =
            publicKey(source, articulationIndex);
        for (const ArticulatedRigidContactWarmStart& warm :
             warmStarts) {
            if (!(warm.key == key)) {
                continue;
            }
            const Vec3 impulse =
                vector(warm.worldImpulseOnRigid);
            const Vec3 normal = vector(contact.normal);
            const Vec3 tangentU = vector(contact.tangentU);
            const Vec3 tangentV = vector(contact.tangentV);
            contact.warmImpulse = {
                impulse.x * normal.x +
                    impulse.y * normal.y +
                    impulse.z * normal.z,
                impulse.x * tangentU.x +
                    impulse.y * tangentU.y +
                    impulse.z * tangentU.z,
                impulse.x * tangentV.x +
                    impulse.y * tangentV.y +
                    impulse.z * tangentV.z,
            };
            ++diagnostics.matchedWarmStartCount;
            break;
        }

        ArticulatedRigidContactMetadata item;
        item.key = key;
        item.articulatedShapeIndex =
            source.articulatedShape;
        item.rigidShapeIndex = source.rigidShape;
        item.articulatedBodyIndex =
            source.articulatedBody;
        item.rigidBodyIndex = source.rigidBody;
        item.manifoldIndex = source.manifold;
        item.manifoldPointIndex = source.point;
        item.lifetime = source.lifetime;
        item.collisionPairKey = source.collisionPairKey;
        item.collisionFeatureKey =
            source.collisionFeatureKey;
        item.contactPointWorld = array(worldPoint);
        item.localWitnessArticulated =
            source.localWitnessArticulated;
        item.localWitnessRigid = source.localWitnessRigid;
        item.effectiveSeparation =
            commonContact.pointAndSeparation.w;

        if (!finite(contact.localPointArticulated) ||
            !finite(contact.localPointRigid) ||
            !finite(contact.normal) ||
            !finite(contact.tangentU) ||
            !finite(contact.tangentV) ||
            !finite(contact.targetVelocity) ||
            !finite(contact.regularization) ||
            !finite(contact.warmImpulse) ||
            !finite(contact.friction) ||
            !finite(item.contactPointWorld) ||
            !finite(item.localWitnessArticulated) ||
            !finite(item.localWitnessRigid) ||
            !finite(item.effectiveSeparation)) {
            return fail(
                diagnostics,
                ArticulatedRigidCollisionStatus::nonfiniteResult,
                MR_STEP_NONFINITE_RESULT,
                "contact or contact metadata is non-finite"
            );
        }
        contacts.push_back(contact);
        metadata.push_back(item);
    }

    diagnostics.contactCount =
        static_cast<std::uint32_t>(contacts.size());
    diagnostics.status =
        ArticulatedRigidCollisionStatus::success;
    diagnostics.code = MR_STEP_SUCCESS;
    manifoldCache = std::move(workingCache);

    ArticulatedRigidCollisionResult result;
    result.diagnostics = std::move(diagnostics);
    result.contacts = std::move(contacts);
    result.metadata = std::move(metadata);
    return result;
}

} // namespace metalrobo
