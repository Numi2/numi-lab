#include "metalrobo/CoupledArticulatedRigidContact.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {
namespace {

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

struct ContactFrame {
    Vec3 normal;
    Vec3 tangentU;
    Vec3 tangentV;
};

struct CholeskyFactor {
    std::vector<double> lower;
    std::size_t dimension = 0u;
    double minimumPivot = 0.0;
    double maximumPivot = 0.0;
    bool valid = false;
};

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const Vec3 value) {
    return finite(value.x) &&
        finite(value.y) &&
        finite(value.z);
}

template <std::size_t Size>
bool finite(const std::array<double, Size>& values) {
    return std::ranges::all_of(values, [](const double value) {
        return finite(value);
    });
}

bool finite(const std::span<const double> values) {
    return std::ranges::all_of(values, [](const double value) {
        return finite(value);
    });
}

bool finite(const std::vector<double>& values) {
    return finite(std::span<const double>(values));
}

Vec3 vector(const std::array<double, 3>& value) {
    return {value[0], value[1], value[2]};
}

Vec3 vector(const mr_float4 value) {
    return {
        static_cast<double>(value.x),
        static_cast<double>(value.y),
        static_cast<double>(value.z),
    };
}

Vec3 operator+(const Vec3 left, const Vec3 right) {
    return {
        left.x + right.x,
        left.y + right.y,
        left.z + right.z,
    };
}

Vec3 operator*(const double scale, const Vec3 value) {
    return {
        scale * value.x,
        scale * value.y,
        scale * value.z,
    };
}

double dot(const Vec3 left, const Vec3 right) {
    return
        left.x * right.x +
        left.y * right.y +
        left.z * right.z;
}

Vec3 cross(const Vec3 left, const Vec3 right) {
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    };
}

double norm(const Vec3 value) {
    return std::sqrt(dot(value, value));
}

bool makeFrame(
    const CoupledArticulatedRigidIslandContact& contact,
    ContactFrame& frame
) {
    const Vec3 normal = vector(contact.normal);
    const Vec3 tangentU = vector(contact.tangentU);
    const Vec3 tangentV = vector(contact.tangentV);
    if (!finite(normal) || !finite(tangentU) || !finite(tangentV)) {
        return false;
    }
    const double normalLength = norm(normal);
    const double tangentULength = norm(tangentU);
    const double tangentVLength = norm(tangentV);
    constexpr double directionTolerance = 2.0e-4;
    constexpr double orthogonalityTolerance = 4.0e-4;
    constexpr double handednessTolerance = 6.0e-4;
    if (!finite(normalLength) ||
        !finite(tangentULength) ||
        !finite(tangentVLength) ||
        std::abs(normalLength - 1.0) > directionTolerance ||
        std::abs(tangentULength - 1.0) > directionTolerance ||
        std::abs(tangentVLength - 1.0) > directionTolerance ||
        std::abs(dot(normal, tangentU)) >
            orthogonalityTolerance ||
        std::abs(dot(normal, tangentV)) >
            orthogonalityTolerance ||
        std::abs(dot(tangentU, tangentV)) >
            orthogonalityTolerance ||
        std::abs(dot(cross(normal, tangentU), tangentV) - 1.0) >
            handednessTolerance) {
        return false;
    }
    frame = {normal, tangentU, tangentV};
    return true;
}

bool quaternion(
    const mr_float4 value,
    Quaternion& result
) {
    result = {
        static_cast<double>(value.x),
        static_cast<double>(value.y),
        static_cast<double>(value.z),
        static_cast<double>(value.w),
    };
    if (!finite(result.x) ||
        !finite(result.y) ||
        !finite(result.z) ||
        !finite(result.w)) {
        return false;
    }
    const double squaredNorm =
        result.x * result.x +
        result.y * result.y +
        result.z * result.z +
        result.w * result.w;
    if (!finite(squaredNorm) ||
        std::abs(squaredNorm - 1.0) > 4.0e-4) {
        return false;
    }
    const double inverseNorm = 1.0 / std::sqrt(squaredNorm);
    result.x *= inverseNorm;
    result.y *= inverseNorm;
    result.z *= inverseNorm;
    result.w *= inverseNorm;
    return true;
}

Vec3 rotate(const Quaternion q, const Vec3 value) {
    const Vec3 imaginary{q.x, q.y, q.z};
    const Vec3 doubled = 2.0 * cross(imaginary, value);
    return value + q.w * doubled + cross(imaginary, doubled);
}

Vec3 scenePointVelocity(
    const MRBodyStateGPU& body,
    const Quaternion orientation,
    const std::array<double, 3>& localPoint
) {
    if (body.flagsAndIndices[0] == MR_MOTION_STATIC) {
        return {};
    }
    const Vec3 arm = rotate(orientation, vector(localPoint));
    return
        vector(body.linearVelocityAndInverseMass) +
        cross(vector(body.angularVelocity), arm);
}

CholeskyFactor factorize(
    const std::span<const double> matrix,
    const std::size_t dimension
) {
    CholeskyFactor result;
    result.dimension = dimension;
    if (dimension == 0u ||
        matrix.size() != dimension * dimension ||
        !finite(matrix)) {
        return result;
    }
    double matrixScale = 0.0;
    for (const double value : matrix) {
        matrixScale = std::max(matrixScale, std::abs(value));
    }
    if (!(matrixScale > 0.0) || !finite(matrixScale)) {
        return result;
    }

    result.lower.assign(dimension * dimension, 0.0);
    result.minimumPivot = std::numeric_limits<double>::infinity();
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u; column <= row; ++column) {
            double value = matrix[row * dimension + column];
            for (std::size_t inner = 0u;
                 inner < column;
                 ++inner) {
                value -=
                    result.lower[row * dimension + inner] *
                    result.lower[column * dimension + inner];
            }
            if (row == column) {
                const double threshold =
                    64.0 * std::numeric_limits<double>::epsilon() *
                    matrixScale *
                    static_cast<double>(dimension);
                if (!(value > threshold) || !finite(value)) {
                    return result;
                }
                const double pivot = std::sqrt(value);
                result.lower[row * dimension + row] = pivot;
                result.minimumPivot =
                    std::min(result.minimumPivot, pivot);
                result.maximumPivot =
                    std::max(result.maximumPivot, pivot);
            } else {
                result.lower[row * dimension + column] =
                    value /
                    result.lower[column * dimension + column];
            }
        }
    }
    result.valid =
        finite(result.lower) &&
        finite(result.minimumPivot) &&
        finite(result.maximumPivot);
    return result;
}

bool solve(
    const CholeskyFactor& factor,
    const std::span<const double> right,
    std::vector<double>& solution
) {
    if (!factor.valid ||
        right.size() != factor.dimension ||
        !finite(right)) {
        return false;
    }
    std::vector<double> intermediate(factor.dimension, 0.0);
    for (std::size_t row = 0u;
         row < factor.dimension;
         ++row) {
        double value = right[row];
        for (std::size_t column = 0u;
             column < row;
             ++column) {
            value -=
                factor.lower[row * factor.dimension + column] *
                intermediate[column];
        }
        intermediate[row] =
            value /
            factor.lower[row * factor.dimension + row];
    }
    solution.assign(factor.dimension, 0.0);
    for (std::size_t reverse = 0u;
         reverse < factor.dimension;
         ++reverse) {
        const std::size_t row =
            factor.dimension - 1u - reverse;
        double value = intermediate[row];
        for (std::size_t column = row + 1u;
             column < factor.dimension;
             ++column) {
            value -=
                factor.lower[column * factor.dimension + row] *
                solution[column];
        }
        solution[row] =
            value /
            factor.lower[row * factor.dimension + row];
    }
    return finite(solution);
}

bool validInverseInertia(
    const MRBodyStateGPU& state,
    std::array<double, 9>& inverseInertia
) {
    inverseInertia = {
        state.inverseInertiaWorldRow0.x,
        state.inverseInertiaWorldRow0.y,
        state.inverseInertiaWorldRow0.z,
        state.inverseInertiaWorldRow1.x,
        state.inverseInertiaWorldRow1.y,
        state.inverseInertiaWorldRow1.z,
        state.inverseInertiaWorldRow2.x,
        state.inverseInertiaWorldRow2.y,
        state.inverseInertiaWorldRow2.z,
    };
    if (!finite(std::span<const double>(
            inverseInertia.data(),
            inverseInertia.size()
        ))) {
        return false;
    }
    constexpr double symmetryTolerance = 2.0e-5;
    const auto symmetric = [&inverseInertia](
        const std::size_t row,
        const std::size_t column
    ) {
        const double left = inverseInertia[row * 3u + column];
        const double right = inverseInertia[column * 3u + row];
        const double scale =
            1.0 + std::max(std::abs(left), std::abs(right));
        return std::abs(left - right) <=
            symmetryTolerance * scale;
    };
    if (!symmetric(0u, 1u) ||
        !symmetric(0u, 2u) ||
        !symmetric(1u, 2u)) {
        return false;
    }
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = row + 1u;
             column < 3u;
             ++column) {
            const double average = 0.5 * (
                inverseInertia[row * 3u + column] +
                inverseInertia[column * 3u + row]
            );
            inverseInertia[row * 3u + column] = average;
            inverseInertia[column * 3u + row] = average;
        }
    }
    return factorize(inverseInertia, 3u).valid;
}

CoupledArticulatedRigidContactDiagnostics diagnosticsFor(
    const std::uint32_t articulationIndex,
    const std::size_t nv,
    const std::size_t rigidBodyCount,
    const std::size_t contactCount,
    const std::size_t jointLimitCount
) {
    CoupledArticulatedRigidContactDiagnostics result;
    result.articulationIndex = articulationIndex;
    result.articulationNv =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            nv,
            std::numeric_limits<std::uint32_t>::max()
        ));
    result.rigidBodyCount =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            rigidBodyCount,
            std::numeric_limits<std::uint32_t>::max()
        ));
    result.contactCount =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            contactCount,
            std::numeric_limits<std::uint32_t>::max()
        ));
    result.jointLimitCount =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            jointLimitCount,
            std::numeric_limits<std::uint32_t>::max()
        ));
    return result;
}

void fail(
    CoupledArticulatedRigidContactDiagnostics& diagnostics,
    const CoupledArticulatedRigidContactStatus status,
    std::string failure
) {
    diagnostics.status = status;
    diagnostics.failure = std::move(failure);
}

} // namespace

CoupledArticulatedRigidContactDiagnostics
solveCoupledArticulatedRigidIslandCpu(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const double> freeArticulationVelocity,
    const std::span<const MRBodyStateGPU> sceneBodies,
    const std::span<const CoupledArticulatedRigidIslandContact> contacts,
    const std::span<double> postArticulationVelocity,
    const std::span<CoupledRigidBodyVelocity> postSceneBodyVelocities,
    const ArticulatedDynamicsConfig& dynamicsConfig,
    const QualityContactSolverConfig& solverConfig,
    const std::span<const ArticulatedJointLimitRow> jointLimitRows,
    const std::span<const double> jointLimitWarmImpulses
) {
    const std::size_t nq =
        articulationIndex < model.articulations.size()
        ? model.articulations[articulationIndex].nq
        : 0u;
    const std::size_t nv =
        articulationIndex < model.articulations.size()
        ? model.articulations[articulationIndex].nv
        : 0u;
    CoupledArticulatedRigidContactDiagnostics diagnostics =
        diagnosticsFor(
            articulationIndex,
            nv,
            sceneBodies.size(),
            contacts.size(),
            jointLimitRows.size()
        );

    if (articulationIndex >= model.articulations.size()) {
        fail(
            diagnostics,
            CoupledArticulatedRigidContactStatus::invalidModel,
            "articulation index is outside the compiled model"
        );
        return diagnostics;
    }
    const std::size_t maximumSize =
        std::numeric_limits<std::size_t>::max();
    const std::size_t dynamicSceneBodyCount =
        static_cast<std::size_t>(std::ranges::count_if(
            sceneBodies,
            [](const MRBodyStateGPU& body) {
                return body.flagsAndIndices[0] == MR_MOTION_DYNAMIC;
            }
        ));
    diagnostics.dynamicRigidBodyCount =
        static_cast<std::uint32_t>(
            std::min<std::size_t>(
                dynamicSceneBodyCount,
                std::numeric_limits<std::uint32_t>::max()
            )
        );
    diagnostics.prescribedRigidBodyCount =
        static_cast<std::uint32_t>(
            std::min<std::size_t>(
                sceneBodies.size() - dynamicSceneBodyCount,
                std::numeric_limits<std::uint32_t>::max()
            )
        );
    const bool combinedDimensionFits =
        dynamicSceneBodyCount <= (maximumSize - nv) / 6u;
    const bool blockCountFits =
        contacts.size() <=
            maximumSize - jointLimitRows.size();
    const std::size_t blockCount = blockCountFits
        ? contacts.size() + jointLimitRows.size()
        : 0u;
    const bool constraintRowsFit =
        blockCountFits && blockCount <= maximumSize / 3u;
    const std::size_t combinedNv = combinedDimensionFits
        ? nv + 6u * dynamicSceneBodyCount
        : 0u;
    const std::size_t constraintRows = constraintRowsFit
        ? 3u * blockCount
        : 0u;
    const std::size_t contactRows =
        contacts.size() <= maximumSize / 3u
        ? 3u * contacts.size()
        : 0u;
    const bool matrixDimensionsFit =
        combinedDimensionFits &&
        constraintRowsFit &&
        combinedNv > 0u &&
        combinedNv <= maximumSize / combinedNv &&
        constraintRows > 0u &&
        constraintRows <= maximumSize / combinedNv &&
        constraintRows <= maximumSize / constraintRows &&
        (nv == 0u || contactRows <= maximumSize / nv);
    std::string modelFailure;
    if (!model.valid(&modelFailure)) {
        fail(
            diagnostics,
            CoupledArticulatedRigidContactStatus::invalidModel,
            "compiled engine model is invalid: " + modelFailure
        );
        return diagnostics;
    }
    if (nv == 0u ||
        blockCount == 0u ||
        !matrixDimensionsFit ||
        sceneBodies.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        contacts.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        jointLimitRows.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        q.size() != nq ||
        freeArticulationVelocity.size() != nv ||
        (!jointLimitWarmImpulses.empty() &&
         jointLimitWarmImpulses.size() != jointLimitRows.size()) ||
        postArticulationVelocity.size() != nv ||
        postSceneBodyVelocities.size() != sceneBodies.size() ||
        combinedNv >
            std::numeric_limits<std::uint32_t>::max()) {
        fail(
            diagnostics,
            CoupledArticulatedRigidContactStatus::invalidDimensions,
            "state, contact, or output dimensions are inconsistent"
        );
        return diagnostics;
    }
    if (!finite(q) ||
        !finite(freeArticulationVelocity) ||
        !finite(jointLimitWarmImpulses) ||
        std::ranges::any_of(
            jointLimitWarmImpulses,
            [](const double impulse) {
                return impulse < 0.0;
            }
        )) {
        fail(
            diagnostics,
            CoupledArticulatedRigidContactStatus::nonfiniteInput,
            "articulated configuration or free velocity is non-finite"
        );
        return diagnostics;
    }

    std::vector<Quaternion> sceneOrientations(sceneBodies.size());
    std::vector<std::array<double, 9>> sceneInverseInertias(
        sceneBodies.size()
    );
    std::vector<std::uint32_t> sceneToDynamic(
        sceneBodies.size(),
        MR_INVALID_INDEX
    );
    std::vector<std::uint32_t> dynamicToScene;
    dynamicToScene.reserve(dynamicSceneBodyCount);
    for (std::size_t bodyIndex = 0u;
         bodyIndex < sceneBodies.size();
         ++bodyIndex) {
        const MRBodyStateGPU& body = sceneBodies[bodyIndex];
        const std::uint32_t motion = body.flagsAndIndices[0];
        const double inverseMass =
            body.linearVelocityAndInverseMass.w;
        if (motion > MR_MOTION_DYNAMIC ||
            body.flagsAndIndices[1] != MR_INVALID_INDEX ||
            !finite(vector(body.position)) ||
            !finite(vector(body.linearVelocityAndInverseMass)) ||
            !finite(vector(body.angularVelocity)) ||
            !finite(vector(body.inverseInertiaWorldRow0)) ||
            !finite(vector(body.inverseInertiaWorldRow1)) ||
            !finite(vector(body.inverseInertiaWorldRow2)) ||
            !finite(inverseMass) ||
            !quaternion(
                body.orientation,
                sceneOrientations[bodyIndex]
            ) ||
            (
                motion == MR_MOTION_DYNAMIC &&
                (
                    !(inverseMass > 0.0) ||
                    !validInverseInertia(
                        body,
                        sceneInverseInertias[bodyIndex]
                    )
                )
            ) ||
            (
                motion != MR_MOTION_DYNAMIC &&
                inverseMass != 0.0
            )) {
            fail(
                diagnostics,
                CoupledArticulatedRigidContactStatus::invalidRigidBody,
                "scene body must be finite, independent, unit-oriented, "
                "and carry mass only when dynamic"
            );
            return diagnostics;
        }
        if (motion == MR_MOTION_DYNAMIC) {
            sceneToDynamic[bodyIndex] =
                static_cast<std::uint32_t>(dynamicToScene.size());
            dynamicToScene.push_back(
                static_cast<std::uint32_t>(bodyIndex)
            );
        }
    }

    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    const std::uint64_t firstBody = articulation.firstBody;
    const std::uint64_t bodyEnd =
        firstBody + articulation.bodyCount;
    std::vector<ContactFrame> frames(contacts.size());
    std::vector<ArticulatedPointQuery> articulatedQueries;
    articulatedQueries.reserve(contacts.size());
    std::vector<std::uint32_t> articulatedQueryForContact(
        contacts.size(),
        MR_INVALID_INDEX
    );
    for (std::size_t contactIndex = 0u;
         contactIndex < contacts.size();
         ++contactIndex) {
        const CoupledArticulatedRigidIslandContact& contact =
            contacts[contactIndex];
        const auto validEndpoint =
            [&](const CoupledContactEndpoint& endpoint) {
                if (!finite(endpoint.localPoint)) {
                    return false;
                }
                if (endpoint.kind ==
                    CoupledContactEndpointKind::articulated) {
                    return endpoint.body >= firstBody &&
                        endpoint.body < bodyEnd;
                }
                if (endpoint.kind ==
                    CoupledContactEndpointKind::sceneBody) {
                    return endpoint.body < sceneBodies.size();
                }
                return false;
            };
        const bool articulatedA =
            contact.endpointA.kind ==
                CoupledContactEndpointKind::articulated;
        const bool articulatedB =
            contact.endpointB.kind ==
                CoupledContactEndpointKind::articulated;
        const bool dynamicA =
            articulatedA ||
            (
                validEndpoint(contact.endpointA) &&
                sceneBodies[contact.endpointA.body].
                    flagsAndIndices[0] == MR_MOTION_DYNAMIC
            );
        const bool dynamicB =
            articulatedB ||
            (
                validEndpoint(contact.endpointB) &&
                sceneBodies[contact.endpointB.body].
                    flagsAndIndices[0] == MR_MOTION_DYNAMIC
            );
        if (!validEndpoint(contact.endpointA) ||
            !validEndpoint(contact.endpointB) ||
            (articulatedA && articulatedB) ||
            (
                contact.endpointA.kind == contact.endpointB.kind &&
                contact.endpointA.body == contact.endpointB.body
            ) ||
            (!dynamicA && !dynamicB) ||
            !finite(contact.targetVelocity) ||
            !finite(contact.regularization) ||
            !finite(contact.warmImpulse) ||
            !finite(contact.friction) ||
            !(contact.friction >= 0.0) ||
            !std::ranges::all_of(
                contact.regularization,
                [](const double value) {
                    return finite(value) && value > 0.0;
                }
            ) ||
            !makeFrame(contact, frames[contactIndex])) {
            fail(
                diagnostics,
                CoupledArticulatedRigidContactStatus::invalidContact,
                "contact ownership, frame, material, or point is invalid"
            );
            return diagnostics;
        }
        if (articulatedA || articulatedB) {
            const CoupledContactEndpoint& endpoint =
                articulatedA
                ? contact.endpointA
                : contact.endpointB;
            articulatedQueryForContact[contactIndex] =
                static_cast<std::uint32_t>(
                    articulatedQueries.size()
                );
            articulatedQueries.push_back({
                endpoint.body,
                endpoint.localPoint,
            });
        }
    }

    for (std::size_t limitIndex = 0u;
         limitIndex < jointLimitRows.size();
         ++limitIndex) {
        const ArticulatedJointLimitRow& row =
            jointLimitRows[limitIndex];
        const bool lower =
            row.side == ArticulatedJointLimitSide::lower;
        const bool upper =
            row.side == ArticulatedJointLimitSide::upper;
        const std::uint64_t expectedGlobalV =
            static_cast<std::uint64_t>(articulation.vOffset) +
            row.localVIndex;
        const std::uint64_t expectedGlobalQ =
            static_cast<std::uint64_t>(articulation.qOffset) +
            row.localQIndex;
        const bool metadataIndexValid =
            row.localVIndex < nv &&
            row.localQIndex < nq &&
            expectedGlobalV < model.dofs.size() &&
            expectedGlobalV <=
                std::numeric_limits<std::uint32_t>::max() &&
            expectedGlobalQ <=
                std::numeric_limits<std::uint32_t>::max();
        const MRDofPropertiesGPU* dof = metadataIndexValid
            ? &model.dofs[
                static_cast<std::size_t>(expectedGlobalV)
            ]
            : nullptr;
        const double expectedFreeVelocity =
            row.direction *
            freeArticulationVelocity[
                std::min<std::size_t>(row.localVIndex, nv - 1u)
            ];
        const double expectedPositionLimit = dof == nullptr
            ? 0.0
            : static_cast<double>(
                lower ? dof->limits.x : dof->limits.y
            );
        const double expectedGap =
            dof == nullptr || row.localQIndex >= q.size()
            ? 0.0
            : (
                lower
                ? q[row.localQIndex] - expectedPositionLimit
                : expectedPositionLimit - q[row.localQIndex]
            );
        const double velocityScale =
            1.0 + std::abs(expectedFreeVelocity);
        if (row.localVIndex >= nv ||
            row.localQIndex >= nq ||
            (!lower && !upper) ||
            (lower && row.direction != 1.0) ||
            (upper && row.direction != -1.0) ||
            dof == nullptr ||
            dof->articulationIndex != articulationIndex ||
            dof->vIndex != row.globalVIndex ||
            dof->qIndex != row.globalQIndex ||
            row.globalVIndex != expectedGlobalV ||
            row.globalQIndex != expectedGlobalQ ||
            (dof->flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u ||
            row.stableKey !=
                2u * expectedGlobalV + (lower ? 0u : 1u) ||
            row.positionLimit != expectedPositionLimit ||
            !finite(row.positionLimit) ||
            !finite(row.gap) ||
            !finite(row.freeNormalVelocity) ||
            !finite(row.targetVelocity) ||
            !finite(row.regularization) ||
            !(row.regularization > 0.0) ||
            std::abs(
                row.freeNormalVelocity - expectedFreeVelocity
            ) > 1.0e-12 * velocityScale ||
            std::abs(row.gap - expectedGap) >
                1.0e-12 *
                (1.0 + std::abs(expectedGap)) ||
            (limitIndex > 0u &&
             jointLimitRows[limitIndex - 1u].stableKey >=
                 row.stableKey)) {
            fail(
                diagnostics,
                CoupledArticulatedRigidContactStatus::invalidJointLimit,
                "joint-limit row is non-canonical or inconsistent "
                "with the free articulation velocity"
            );
            return diagnostics;
        }
    }

    std::vector<ArticulatedPointKinematics> articulatedPoints(
        articulatedQueries.size()
    );
    std::vector<double> articulatedJacobians(
        articulatedQueries.size() * 3u * nv,
        0.0
    );
    if (!articulatedQueries.empty()) {
        const ArticulatedDynamicsDiagnostics kinematicsDiagnostics =
            computeArticulatedPointJacobians(
                model,
                articulationIndex,
                q,
                freeArticulationVelocity,
                articulatedQueries,
                articulatedPoints,
                articulatedJacobians,
                dynamicsConfig
            );
        if (!kinematicsDiagnostics.succeeded()) {
            diagnostics.dynamicsStatus =
                kinematicsDiagnostics.status;
            fail(
                diagnostics,
                CoupledArticulatedRigidContactStatus::dynamicsFailure,
                "articulated point kinematics failed"
            );
            return diagnostics;
        }
    }

    std::vector<double> articulationMass(nv * nv, 0.0);
    const ArticulatedDynamicsDiagnostics massDiagnostics =
        computeArticulatedMassMatrix(
            model,
            articulationIndex,
            q,
            articulationMass,
            dynamicsConfig
        );
    if (!massDiagnostics.succeeded()) {
        diagnostics.dynamicsStatus = massDiagnostics.status;
        fail(
            diagnostics,
            CoupledArticulatedRigidContactStatus::dynamicsFailure,
            "articulation CRBA mass operator failed"
        );
        return diagnostics;
    }
    const CholeskyFactor articulationFactor =
        factorize(articulationMass, nv);
    if (!articulationFactor.valid) {
        fail(
            diagnostics,
            CoupledArticulatedRigidContactStatus::factorizationFailure,
            "articulation mass operator is not positive definite"
        );
        return diagnostics;
    }
    diagnostics.minimumArticulationCholeskyPivot =
        articulationFactor.minimumPivot;
    diagnostics.maximumArticulationCholeskyPivot =
        articulationFactor.maximumPivot;

    std::vector<double> inverseMass(
        combinedNv * combinedNv,
        0.0
    );
    std::vector<double> right(nv, 0.0);
    std::vector<double> inverseColumn;
    for (std::size_t column = 0u; column < nv; ++column) {
        std::ranges::fill(right, 0.0);
        right[column] = 1.0;
        if (!solve(articulationFactor, right, inverseColumn)) {
            fail(
                diagnostics,
                CoupledArticulatedRigidContactStatus::factorizationFailure,
                "articulation inverse-mass action failed"
            );
            return diagnostics;
        }
        for (std::size_t row = 0u; row < nv; ++row) {
            inverseMass[row * combinedNv + column] =
                inverseColumn[row];
        }
    }
    for (std::size_t row = 0u; row < nv; ++row) {
        for (std::size_t column = row + 1u;
             column < nv;
             ++column) {
            const double symmetric = 0.5 * (
                inverseMass[row * combinedNv + column] +
                inverseMass[column * combinedNv + row]
            );
            inverseMass[row * combinedNv + column] = symmetric;
            inverseMass[column * combinedNv + row] = symmetric;
        }
    }
    for (std::size_t row = 0u; row < nv; ++row) {
        for (std::size_t column = 0u;
             column < nv;
             ++column) {
            double value = 0.0;
            for (std::size_t inner = 0u; inner < nv; ++inner) {
                value +=
                    articulationMass[row * nv + inner] *
                    inverseMass[inner * combinedNv + column];
            }
            diagnostics.maximumArticulationInverseResidual = std::max(
                diagnostics.maximumArticulationInverseResidual,
                std::abs(
                    value -
                    (row == column ? 1.0 : 0.0)
                )
            );
        }
    }
    if (!finite(diagnostics.maximumArticulationInverseResidual) ||
        diagnostics.maximumArticulationInverseResidual > 1.0e-9) {
        fail(
            diagnostics,
            CoupledArticulatedRigidContactStatus::factorizationFailure,
            "articulation inverse-mass residual exceeds FP64 gate"
        );
        return diagnostics;
    }

    for (std::size_t dynamicIndex = 0u;
         dynamicIndex < dynamicToScene.size();
         ++dynamicIndex) {
        const std::uint32_t bodyIndex =
            dynamicToScene[dynamicIndex];
        const std::size_t offset = nv + 6u * dynamicIndex;
        const double inverseMassValue =
            sceneBodies[bodyIndex].linearVelocityAndInverseMass.w;
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            inverseMass[
                (offset + axis) * combinedNv + offset + axis
            ] = inverseMassValue;
        }
        for (std::size_t row = 0u; row < 3u; ++row) {
            for (std::size_t column = 0u;
                 column < 3u;
                 ++column) {
                inverseMass[
                    (offset + 3u + row) * combinedNv +
                    offset + 3u + column
                ] = sceneInverseInertias[bodyIndex][
                    row * 3u + column
                ];
            }
        }
    }
    for (std::size_t row = 0u; row < combinedNv; ++row) {
        for (std::size_t column = row + 1u;
             column < combinedNv;
             ++column) {
            diagnostics.maximumInverseMassAsymmetry = std::max(
                diagnostics.maximumInverseMassAsymmetry,
                std::abs(
                    inverseMass[row * combinedNv + column] -
                    inverseMass[column * combinedNv + row]
                )
            );
        }
    }
    if (!finite(inverseMass)) {
        fail(
            diagnostics,
            CoupledArticulatedRigidContactStatus::nonfiniteResult,
            "combined block inverse mass is non-finite"
        );
        return diagnostics;
    }

    DenseConicProblem conic;
    conic.nv = static_cast<std::uint32_t>(combinedNv);
    conic.inverseMass = std::move(inverseMass);
    conic.freeVelocity.assign(
        freeArticulationVelocity.begin(),
        freeArticulationVelocity.end()
    );
    for (const std::uint32_t bodyIndex : dynamicToScene) {
        const MRBodyStateGPU& body = sceneBodies[bodyIndex];
        conic.freeVelocity.push_back(
            body.linearVelocityAndInverseMass.x
        );
        conic.freeVelocity.push_back(
            body.linearVelocityAndInverseMass.y
        );
        conic.freeVelocity.push_back(
            body.linearVelocityAndInverseMass.z
        );
        conic.freeVelocity.push_back(body.angularVelocity.x);
        conic.freeVelocity.push_back(body.angularVelocity.y);
        conic.freeVelocity.push_back(body.angularVelocity.z);
    }
    conic.contacts.resize(blockCount);

    std::vector<double> constraintJacobian(
        constraintRows * combinedNv,
        0.0
    );
    std::vector<double> prescribedContactVelocity(
        contactRows,
        0.0
    );
    diagnostics.freeContactVelocity.assign(contactRows, 0.0);
    for (std::size_t contactIndex = 0u;
         contactIndex < contacts.size();
         ++contactIndex) {
        const CoupledArticulatedRigidIslandContact& contact =
            contacts[contactIndex];
        const std::array<Vec3, 3> axes{
            frames[contactIndex].normal,
            frames[contactIndex].tangentU,
            frames[contactIndex].tangentV,
        };

        DenseContactBlock& block = conic.contacts[contactIndex];
        block.normalJacobian.assign(combinedNv, 0.0);
        block.tangentUJacobian.assign(combinedNv, 0.0);
        block.tangentVJacobian.assign(combinedNv, 0.0);
        const std::array<std::span<double>, 3> blockRows{
            block.normalJacobian,
            block.tangentUJacobian,
            block.tangentVJacobian,
        };
        for (std::size_t axisIndex = 0u;
             axisIndex < 3u;
             ++axisIndex) {
            const Vec3 axis = axes[axisIndex];
            const std::size_t row =
                3u * contactIndex + axisIndex;
            std::span<double> jacobianRow(
                constraintJacobian.data() + row * combinedNv,
                combinedNv
            );
            const std::array<const CoupledContactEndpoint*, 2>
                endpoints{
                    &contact.endpointA,
                    &contact.endpointB,
                };
            const std::array<double, 2> signs{-1.0, 1.0};
            Vec3 prescribedRelative{};
            for (std::size_t endpointIndex = 0u;
                 endpointIndex < endpoints.size();
                 ++endpointIndex) {
                const CoupledContactEndpoint& endpoint =
                    *endpoints[endpointIndex];
                const double sign = signs[endpointIndex];
                if (endpoint.kind ==
                    CoupledContactEndpointKind::articulated) {
                    const std::uint32_t queryIndex =
                        articulatedQueryForContact[contactIndex];
                    if (queryIndex == MR_INVALID_INDEX ||
                        queryIndex >= articulatedQueries.size()) {
                        fail(
                            diagnostics,
                            CoupledArticulatedRigidContactStatus::
                                nonfiniteResult,
                            "articulated endpoint lost its point query"
                        );
                        return diagnostics;
                    }
                    const std::size_t pointBase =
                        static_cast<std::size_t>(queryIndex) *
                        3u * nv;
                    for (std::size_t dof = 0u;
                         dof < nv;
                         ++dof) {
                        const Vec3 articulatedColumn{
                            articulatedJacobians[
                                pointBase + 0u * nv + dof
                            ],
                            articulatedJacobians[
                                pointBase + 1u * nv + dof
                            ],
                            articulatedJacobians[
                                pointBase + 2u * nv + dof
                            ],
                        };
                        jacobianRow[dof] +=
                            sign * dot(axis, articulatedColumn);
                    }
                    continue;
                }

                const std::uint32_t bodyIndex = endpoint.body;
                const MRBodyStateGPU& body =
                    sceneBodies[bodyIndex];
                const std::uint32_t dynamicIndex =
                    sceneToDynamic[bodyIndex];
                if (dynamicIndex == MR_INVALID_INDEX) {
                    prescribedRelative =
                        prescribedRelative +
                        sign * scenePointVelocity(
                            body,
                            sceneOrientations[bodyIndex],
                            endpoint.localPoint
                        );
                    continue;
                }

                const std::size_t bodyOffset =
                    nv + 6u * dynamicIndex;
                jacobianRow[bodyOffset + 0u] +=
                    sign * axis.x;
                jacobianRow[bodyOffset + 1u] +=
                    sign * axis.y;
                jacobianRow[bodyOffset + 2u] +=
                    sign * axis.z;
                const Vec3 arm = rotate(
                    sceneOrientations[bodyIndex],
                    vector(endpoint.localPoint)
                );
                const Vec3 angularColumn =
                    sign * cross(arm, axis);
                jacobianRow[bodyOffset + 3u] +=
                    angularColumn.x;
                jacobianRow[bodyOffset + 4u] +=
                    angularColumn.y;
                jacobianRow[bodyOffset + 5u] +=
                    angularColumn.z;
            }
            std::ranges::copy(jacobianRow, blockRows[axisIndex].begin());
            prescribedContactVelocity[row] =
                dot(prescribedRelative, axis);
            for (std::size_t dof = 0u;
                 dof < combinedNv;
                 ++dof) {
                diagnostics.freeContactVelocity[row] +=
                    jacobianRow[dof] * conic.freeVelocity[dof];
            }
            diagnostics.freeContactVelocity[row] +=
                prescribedContactVelocity[row];
            block.targetVelocity[axisIndex] =
                contact.targetVelocity[axisIndex] -
                prescribedContactVelocity[row];
        }
        block.regularization = contact.regularization;
        block.warmImpulse = contact.warmImpulse;
        block.friction = contact.friction;
    }

    diagnostics.freeJointLimitVelocity.assign(
        jointLimitRows.size(),
        0.0
    );
    for (std::size_t limitIndex = 0u;
         limitIndex < jointLimitRows.size();
         ++limitIndex) {
        const ArticulatedJointLimitRow& limit =
            jointLimitRows[limitIndex];
        const std::size_t blockIndex =
            contacts.size() + limitIndex;
        const std::size_t row = 3u * blockIndex;
        DenseContactBlock& block = conic.contacts[blockIndex];
        block.normalJacobian.assign(combinedNv, 0.0);
        block.tangentUJacobian.assign(combinedNv, 0.0);
        block.tangentVJacobian.assign(combinedNv, 0.0);
        block.normalJacobian[limit.localVIndex] =
            limit.direction;
        constraintJacobian[
            row * combinedNv + limit.localVIndex
        ] = limit.direction;
        block.targetVelocity = {
            limit.targetVelocity,
            0.0,
            0.0,
        };
        block.regularization = {
            limit.regularization,
            limit.regularization,
            limit.regularization,
        };
        block.warmImpulse = {
            jointLimitWarmImpulses.empty()
                ? 0.0
                : jointLimitWarmImpulses[limitIndex],
            0.0,
            0.0,
        };
        block.friction = 0.0;
        diagnostics.freeJointLimitVelocity[limitIndex] =
            limit.direction *
            conic.freeVelocity[limit.localVIndex];
    }

    diagnostics.quality =
        solveQualityContactProblem(conic, solverConfig);
    if (!diagnostics.quality.converged()) {
        fail(
            diagnostics,
            CoupledArticulatedRigidContactStatus::solverFailure,
            "quality exact-cone solve failed: " +
                diagnostics.quality.failure
        );
        return diagnostics;
    }
    if (diagnostics.quality.velocity.size() != combinedNv ||
        diagnostics.quality.impulses.size() != constraintRows ||
        !finite(diagnostics.quality.velocity) ||
        !finite(diagnostics.quality.impulses)) {
        fail(
            diagnostics,
            CoupledArticulatedRigidContactStatus::nonfiniteResult,
            "quality solve returned inconsistent or non-finite output"
        );
        return diagnostics;
    }

    std::vector<double> generalizedImpulse(combinedNv, 0.0);
    for (std::size_t row = 0u; row < constraintRows; ++row) {
        for (std::size_t dof = 0u; dof < combinedNv; ++dof) {
            generalizedImpulse[dof] +=
                constraintJacobian[row * combinedNv + dof] *
                diagnostics.quality.impulses[row];
        }
    }
    std::vector<double> reconstructedVelocity = conic.freeVelocity;
    for (std::size_t row = 0u; row < combinedNv; ++row) {
        double delta = 0.0;
        for (std::size_t column = 0u;
             column < combinedNv;
             ++column) {
            delta +=
                conic.inverseMass[row * combinedNv + column] *
                generalizedImpulse[column];
        }
        reconstructedVelocity[row] += delta;
        diagnostics.maximumVelocityReconstructionError = std::max(
            diagnostics.maximumVelocityReconstructionError,
            std::abs(
                reconstructedVelocity[row] -
                diagnostics.quality.velocity[row]
            )
        );
    }

    diagnostics.postContactVelocity.assign(contactRows, 0.0);
    for (std::size_t row = 0u; row < contactRows; ++row) {
        diagnostics.postContactVelocity[row] =
            prescribedContactVelocity[row];
        for (std::size_t dof = 0u; dof < combinedNv; ++dof) {
            diagnostics.postContactVelocity[row] +=
                constraintJacobian[row * combinedNv + dof] *
                diagnostics.quality.velocity[dof];
        }
        double predicted =
            diagnostics.freeContactVelocity[row];
        for (std::size_t dof = 0u; dof < combinedNv; ++dof) {
            predicted +=
                constraintJacobian[row * combinedNv + dof] *
                (
                    reconstructedVelocity[dof] -
                    conic.freeVelocity[dof]
                );
        }
        diagnostics.maximumContactVelocityConsistencyError = std::max(
            diagnostics.maximumContactVelocityConsistencyError,
            std::abs(
                predicted -
                diagnostics.postContactVelocity[row]
            )
        );
    }
    diagnostics.postJointLimitVelocity.assign(
        jointLimitRows.size(),
        0.0
    );
    for (std::size_t limitIndex = 0u;
         limitIndex < jointLimitRows.size();
         ++limitIndex) {
        const std::size_t row =
            3u * (contacts.size() + limitIndex);
        for (std::size_t dof = 0u; dof < combinedNv; ++dof) {
            diagnostics.postJointLimitVelocity[limitIndex] +=
                constraintJacobian[row * combinedNv + dof] *
                diagnostics.quality.velocity[dof];
        }
        double predicted =
            diagnostics.freeJointLimitVelocity[limitIndex];
        for (std::size_t dof = 0u; dof < combinedNv; ++dof) {
            predicted +=
                constraintJacobian[row * combinedNv + dof] *
                (
                    reconstructedVelocity[dof] -
                    conic.freeVelocity[dof]
                );
        }
        diagnostics.maximumContactVelocityConsistencyError =
            std::max(
                diagnostics.maximumContactVelocityConsistencyError,
                std::abs(
                    predicted -
                    diagnostics.postJointLimitVelocity[limitIndex]
                )
            );
    }
    if (!finite(diagnostics.maximumVelocityReconstructionError) ||
        !finite(
            diagnostics.maximumContactVelocityConsistencyError
        ) ||
        diagnostics.maximumVelocityReconstructionError > 1.0e-10 ||
        diagnostics.maximumContactVelocityConsistencyError > 1.0e-10) {
        fail(
            diagnostics,
            CoupledArticulatedRigidContactStatus::nonfiniteResult,
            "post-impulse velocity reconstruction failed"
        );
        return diagnostics;
    }

    std::vector<double> candidateArticulation(
        diagnostics.quality.velocity.begin(),
        diagnostics.quality.velocity.begin() +
            static_cast<std::ptrdiff_t>(nv)
    );
    std::vector<CoupledRigidBodyVelocity> candidateRigid(
        sceneBodies.size()
    );
    for (std::size_t bodyIndex = 0u;
         bodyIndex < sceneBodies.size();
         ++bodyIndex) {
        candidateRigid[bodyIndex].linear = {
            sceneBodies[bodyIndex].linearVelocityAndInverseMass.x,
            sceneBodies[bodyIndex].linearVelocityAndInverseMass.y,
            sceneBodies[bodyIndex].linearVelocityAndInverseMass.z,
        };
        candidateRigid[bodyIndex].angular = {
            sceneBodies[bodyIndex].angularVelocity.x,
            sceneBodies[bodyIndex].angularVelocity.y,
            sceneBodies[bodyIndex].angularVelocity.z,
        };
        const std::uint32_t dynamicIndex =
            sceneToDynamic[bodyIndex];
        if (dynamicIndex == MR_INVALID_INDEX) {
            continue;
        }
        const std::size_t offset = nv + 6u * dynamicIndex;
        candidateRigid[bodyIndex].linear = {
            diagnostics.quality.velocity[offset + 0u],
            diagnostics.quality.velocity[offset + 1u],
            diagnostics.quality.velocity[offset + 2u],
        };
        candidateRigid[bodyIndex].angular = {
            diagnostics.quality.velocity[offset + 3u],
            diagnostics.quality.velocity[offset + 4u],
            diagnostics.quality.velocity[offset + 5u],
        };
    }
    if (!finite(candidateArticulation) ||
        !std::ranges::all_of(
            candidateRigid,
            [](const CoupledRigidBodyVelocity& velocity) {
                return finite(velocity.linear) &&
                    finite(velocity.angular);
            }
        )) {
        fail(
            diagnostics,
            CoupledArticulatedRigidContactStatus::nonfiniteResult,
            "post-impulse output velocity is non-finite"
        );
        return diagnostics;
    }

    diagnostics.contactImpulses.assign(
        diagnostics.quality.impulses.begin(),
        diagnostics.quality.impulses.begin() +
            static_cast<std::ptrdiff_t>(contactRows)
    );
    diagnostics.impulses = diagnostics.contactImpulses;
    diagnostics.jointLimitImpulses.assign(
        jointLimitRows.size(),
        0.0
    );
    for (std::size_t limitIndex = 0u;
         limitIndex < jointLimitRows.size();
         ++limitIndex) {
        const std::size_t offset =
            3u * (contacts.size() + limitIndex);
        diagnostics.jointLimitImpulses[limitIndex] =
            diagnostics.quality.impulses[offset];
        if (std::abs(
                diagnostics.quality.impulses[offset + 1u]
            ) > 1.0e-12 ||
            std::abs(
                diagnostics.quality.impulses[offset + 2u]
            ) > 1.0e-12) {
            fail(
                diagnostics,
                CoupledArticulatedRigidContactStatus::nonfiniteResult,
                "frictionless joint-limit block produced a "
                "tangential impulse"
            );
            return diagnostics;
        }
    }
    diagnostics.failure.clear();
    diagnostics.status =
        CoupledArticulatedRigidContactStatus::success;
    std::ranges::copy(
        candidateArticulation,
        postArticulationVelocity.begin()
    );
    std::ranges::copy(
        candidateRigid,
        postSceneBodyVelocities.begin()
    );
    return diagnostics;
}

CoupledArticulatedRigidContactDiagnostics
solveCoupledArticulatedRigidContactsCpu(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const double> freeArticulationVelocity,
    const std::span<const MRBodyStateGPU> rigidBodies,
    const std::span<const CoupledArticulatedRigidContact> contacts,
    const std::span<double> postArticulationVelocity,
    const std::span<CoupledRigidBodyVelocity> postRigidVelocities,
    const ArticulatedDynamicsConfig& dynamicsConfig,
    const QualityContactSolverConfig& solverConfig,
    const std::span<const ArticulatedJointLimitRow> jointLimitRows,
    const std::span<const double> jointLimitWarmImpulses
) {
    std::vector<CoupledArticulatedRigidIslandContact> islandContacts;
    islandContacts.reserve(contacts.size());
    for (const CoupledArticulatedRigidContact& contact : contacts) {
        islandContacts.push_back({
            .endpointA = {
                .kind = CoupledContactEndpointKind::articulated,
                .body = contact.articulatedBody,
                .localPoint = contact.localPointArticulated,
            },
            .endpointB = {
                .kind = CoupledContactEndpointKind::sceneBody,
                .body = contact.rigidBody,
                .localPoint = contact.localPointRigid,
            },
            .normal = contact.normal,
            .tangentU = contact.tangentU,
            .tangentV = contact.tangentV,
            .targetVelocity = contact.targetVelocity,
            .regularization = contact.regularization,
            .warmImpulse = contact.warmImpulse,
            .friction = contact.friction,
        });
    }
    return solveCoupledArticulatedRigidIslandCpu(
        model,
        articulationIndex,
        q,
        freeArticulationVelocity,
        rigidBodies,
        islandContacts,
        postArticulationVelocity,
        postRigidVelocities,
        dynamicsConfig,
        solverConfig,
        jointLimitRows,
        jointLimitWarmImpulses
    );
}

} // namespace metalrobo
