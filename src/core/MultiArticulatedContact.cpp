#include "metalrobo/MultiArticulatedContact.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct ContactFrame {
    Vec3 normal;
    Vec3 tangentU;
    Vec3 tangentV;
};

struct EndpointQuery {
    std::uint32_t articulation = MR_INVALID_INDEX;
    std::size_t query = std::numeric_limits<std::size_t>::max();
    bool isStatic = false;
};

struct ArticulationQueries {
    std::vector<ArticulatedPointQuery> queries;
    std::vector<ArticulatedPointKinematics> kinematics;
    std::vector<double> jacobians;
};

Vec3 vector(const std::array<double, 3>& value) {
    return {value[0], value[1], value[2]};
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

Vec3 rotateVector(
    const mr_float4 quaternion,
    const std::array<double, 3>& value
) {
    const Vec3 q{
        quaternion.x,
        quaternion.y,
        quaternion.z,
    };
    const Vec3 input = vector(value);
    const Vec3 doubledCross = cross(q, input);
    const Vec3 doubled{
        2.0 * doubledCross.x,
        2.0 * doubledCross.y,
        2.0 * doubledCross.z,
    };
    const Vec3 second = cross(q, doubled);
    return {
        input.x + quaternion.w * doubled.x + second.x,
        input.y + quaternion.w * doubled.y + second.y,
        input.z + quaternion.w * doubled.z + second.z,
    };
}

double norm(const Vec3 value) {
    return std::sqrt(dot(value, value));
}

bool finite(const double value) {
    return std::isfinite(value);
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

bool finite(const std::span<const double> values) {
    return std::ranges::all_of(
        values,
        [](const double value) {
            return finite(value);
        }
    );
}

template <typename Contact>
bool makeFrame(
    const Contact& contact,
    ContactFrame& frame
) {
    const Vec3 normal = vector(contact.normal);
    const Vec3 tangentU = vector(contact.tangentU);
    const Vec3 tangentV = vector(contact.tangentV);
    const double normalLength = norm(normal);
    const double tangentULength = norm(tangentU);
    const double tangentVLength = norm(tangentV);
    constexpr double directionTolerance = 2.0e-4;
    constexpr double orthogonalityTolerance = 4.0e-4;
    constexpr double handednessTolerance = 6.0e-4;
    if (!finite(normalLength) ||
        !finite(tangentULength) ||
        !finite(tangentVLength) ||
        std::abs(normalLength - 1.0) >
            directionTolerance ||
        std::abs(tangentULength - 1.0) >
            directionTolerance ||
        std::abs(tangentVLength - 1.0) >
            directionTolerance ||
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

MultiArticulatedContactDiagnostics diagnosticsFor(
    const EngineModel& model,
    const std::size_t contactCount
) {
    MultiArticulatedContactDiagnostics diagnostics;
    diagnostics.articulationCount =
        static_cast<std::uint32_t>(
            std::min<std::size_t>(
                model.articulations.size(),
                std::numeric_limits<std::uint32_t>::max()
            )
        );
    diagnostics.contactCount =
        static_cast<std::uint32_t>(
            std::min<std::size_t>(
                contactCount,
                std::numeric_limits<std::uint32_t>::max()
            )
        );
    diagnostics.rowCount =
        contactCount <=
            std::numeric_limits<std::uint32_t>::max() / 3u
        ? static_cast<std::uint32_t>(3u * contactCount)
        : std::numeric_limits<std::uint32_t>::max();
    return diagnostics;
}

MultiArticulatedContactDiagnostics fail(
    MultiArticulatedContactDiagnostics diagnostics,
    const MultiArticulatedContactStatus status,
    const std::uint32_t contact = MR_INVALID_INDEX,
    const std::uint32_t articulation = MR_INVALID_INDEX
) {
    diagnostics.status = status;
    diagnostics.firstFailingContact = contact;
    diagnostics.firstFailingArticulation = articulation;
    return diagnostics;
}

bool structurallyValid(
    const MultiArticulatedContactProblem& problem
) {
    const std::size_t rowCount =
        3u * static_cast<std::size_t>(problem.contactCount);
    const std::size_t equalityRows =
        problem.generalizedConstraintRowCount;
    const bool equalityLayoutValid =
        equalityRows == 0u
        ? problem.generalizedConstraintJacobian.empty() &&
            problem.generalizedConstraintResponseColumns.empty() &&
            problem.generalizedConstraintFreeImpulses.empty() &&
            problem.generalizedConstraintContactCoupling.empty() &&
            problem.generalizedConstraintTargets.empty() &&
            problem.generalizedConstraintRegularization.empty()
        : problem.generalizedConstraintJacobian.size() ==
                equalityRows * problem.nv &&
            problem.generalizedConstraintResponseColumns.size() ==
                equalityRows * problem.nv &&
            problem.generalizedConstraintFreeImpulses.size() ==
                equalityRows &&
            problem.generalizedConstraintContactCoupling.size() ==
                equalityRows * rowCount &&
            problem.generalizedConstraintTargets.size() ==
                equalityRows &&
            problem.generalizedConstraintRegularization.size() ==
                equalityRows;
    return
        problem.freeVelocity.size() == problem.nv &&
        problem.contactJacobian.size() ==
            rowCount * problem.nv &&
        problem.responseColumns.size() ==
            rowCount * problem.nv &&
        problem.prescribedContactVelocity.size() ==
            rowCount &&
        problem.conic.delassus.size() ==
            rowCount * rowCount &&
        problem.conic.freeContactVelocity.size() ==
            rowCount &&
        problem.conic.contacts.size() ==
            problem.contactCount &&
        problem.pointA.size() == problem.contactCount &&
        problem.pointB.size() == problem.contactCount &&
        problem.sceneBodyFreeVelocities.size() ==
            problem.sceneBodyVelocityOffsets.size() &&
        problem.sceneBodyStates.size() ==
            problem.sceneBodyVelocityOffsets.size() &&
        equalityLayoutValid;
}

bool buildGeneralizedEqualityJacobian(
    const ConstraintIR& program,
    const std::size_t width,
    std::vector<double>& jacobian
) {
    const std::size_t rowCount = program.rows.size();
    if (rowCount == 0u || width == 0u ||
        rowCount >
            std::numeric_limits<std::size_t>::max() / width ||
        !program.cones.empty()) {
        return false;
    }
    jacobian.assign(rowCount * width, 0.0);
    for (const ConstraintIRBlock& block : program.blocks) {
        if (block.type == MR_CONSTRAINT_CONTACT ||
            block.coneIndex != kConstraintIRInvalidIndex ||
            (block.flags & constraintIRBlockDisabled) != 0u) {
            return false;
        }
        for (std::uint32_t local = 0u;
             local < block.endpointCount;
             ++local) {
            const ConstraintIREndpoint& endpoint =
                program.endpoints[
                    block.endpointOffset + local
                ];
            const std::uint32_t row =
                endpoint.flags &
                constraintIREndpointRowMask;
            if (endpoint.jacobianKind !=
                    constraintIRJacobianGeneralized ||
                endpoint.objectIndex >= width ||
                row >= block.dimension) {
                return false;
            }
            double& coefficient = jacobian[
                (block.rowOffset + row) * width +
                endpoint.objectIndex
            ];
            coefficient += endpoint.axis.x;
            if (!finite(coefficient)) {
                return false;
            }
        }
    }
    for (std::size_t row = 0u; row < rowCount; ++row) {
        if (program.rows[row].impulseLower >
                -0.5f * kConstraintIRUnbounded ||
            program.rows[row].impulseUpper <
                0.5f * kConstraintIRUnbounded) {
            return false;
        }
        bool nonzero = false;
        for (std::size_t dof = 0u;
             dof < width;
             ++dof) {
            nonzero =
                nonzero ||
                jacobian[row * width + dof] != 0.0;
        }
        if (!nonzero) {
            return false;
        }
    }
    return true;
}

bool factorDenseSPD(
    std::vector<double>& matrix,
    const std::size_t dimension
) {
    if (matrix.size() != dimension * dimension ||
        dimension == 0u) {
        return false;
    }
    double scale = 1.0;
    for (std::size_t row = 0u; row < dimension; ++row) {
        scale = std::max(
            scale,
            std::abs(matrix[row * dimension + row])
        );
    }
    const double pivotFloor =
        4096.0 * std::numeric_limits<double>::epsilon() *
        scale * static_cast<double>(dimension);
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u;
             column <= row;
             ++column) {
            double value =
                matrix[row * dimension + column];
            for (std::size_t inner = 0u;
                 inner < column;
                 ++inner) {
                value -=
                    matrix[row * dimension + inner] *
                    matrix[column * dimension + inner];
            }
            if (row == column) {
                if (!(value > pivotFloor) || !finite(value)) {
                    return false;
                }
                matrix[row * dimension + column] =
                    std::sqrt(value);
            } else {
                value /=
                    matrix[column * dimension + column];
                if (!finite(value)) {
                    return false;
                }
                matrix[row * dimension + column] = value;
            }
        }
        for (std::size_t column = row + 1u;
             column < dimension;
             ++column) {
            matrix[row * dimension + column] = 0.0;
        }
    }
    return true;
}

bool solveDenseSPD(
    const std::span<const double> lower,
    const std::span<const double> right,
    const std::span<double> solution
) {
    const std::size_t dimension = right.size();
    if (dimension == 0u ||
        solution.size() != dimension ||
        lower.size() != dimension * dimension) {
        return false;
    }
    std::vector<double> intermediate(dimension, 0.0);
    for (std::size_t row = 0u; row < dimension; ++row) {
        double value = right[row];
        for (std::size_t column = 0u;
             column < row;
             ++column) {
            value -=
                lower[row * dimension + column] *
                intermediate[column];
        }
        value /= lower[row * dimension + row];
        if (!finite(value)) {
            return false;
        }
        intermediate[row] = value;
    }
    for (std::size_t reverse = 0u;
         reverse < dimension;
         ++reverse) {
        const std::size_t row =
            dimension - 1u - reverse;
        double value = intermediate[row];
        for (std::size_t column = row + 1u;
             column < dimension;
             ++column) {
            value -=
                lower[column * dimension + row] *
                solution[column];
        }
        value /= lower[row * dimension + row];
        if (!finite(value)) {
            return false;
        }
        solution[row] = value;
    }
    return true;
}

} // namespace

MultiArticulatedContactDiagnostics
buildMultiArticulatedContactProblem(
    const EngineModel& model,
    const std::span<const double> q,
    const std::span<const double> freeVelocity,
    const std::span<const ArticulatedContact> contacts,
    MultiArticulatedContactProblem& output,
    const ArticulatedDynamicsConfig& config
) {
    MultiArticulatedContactDiagnostics diagnostics =
        diagnosticsFor(model, contacts.size());
    std::string reason;
    if (!model.valid(&reason)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::invalidModel
        );
    }
    if (contacts.size() >
            std::numeric_limits<std::uint32_t>::max() / 3u ||
        q.size() != model.world.nq ||
        freeVelocity.size() != model.world.nv) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::invalidDimensions
        );
    }
    if (!finite(q) || !finite(freeVelocity)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::nonfiniteInput
        );
    }

    const std::size_t contactCount = contacts.size();
    const std::size_t rowCount = 3u * contactCount;
    const std::size_t nv = model.world.nv;
    std::vector<ContactFrame> frames(contactCount);
    std::vector<EndpointQuery> endpointsA(contactCount);
    std::vector<EndpointQuery> endpointsB(contactCount);
    std::vector<ArticulationQueries> grouped(
        model.articulations.size()
    );

    const auto addEndpoint = [&](
        const std::uint32_t bodyIndex,
        const std::array<double, 3>& localPoint,
        EndpointQuery& endpoint
    ) -> bool {
        if (bodyIndex >= model.bodies.size()) {
            return false;
        }
        const std::uint32_t articulation =
            model.bodies[bodyIndex].articulationIndex;
        if (articulation == MR_INVALID_INDEX ||
            articulation >= model.articulations.size()) {
            return false;
        }
        endpoint.articulation = articulation;
        endpoint.query = grouped[articulation].queries.size();
        grouped[articulation].queries.push_back({
            bodyIndex,
            localPoint,
        });
        return true;
    };

    for (std::size_t index = 0u;
         index < contactCount;
         ++index) {
        const ArticulatedContact& contact = contacts[index];
        const bool finiteContact =
            finite(contact.localPointA) &&
            finite(contact.localPointB) &&
            finite(contact.targetVelocity) &&
            finite(contact.regularization) &&
            finite(contact.warmImpulse) &&
            finite(contact.friction);
        const bool validParameters =
            finiteContact &&
            contact.friction >= 0.0 &&
            std::ranges::all_of(
                contact.regularization,
                [](const double value) {
                    return finite(value) && value > 0.0;
                }
            );
        if (!validParameters ||
            !makeFrame(contact, frames[index]) ||
            !addEndpoint(
                contact.bodyA,
                contact.localPointA,
                endpointsA[index]
            )) {
            return fail(
                std::move(diagnostics),
                finiteContact
                    ? MultiArticulatedContactStatus::
                          invalidContact
                    : MultiArticulatedContactStatus::
                          nonfiniteInput,
                static_cast<std::uint32_t>(index)
            );
        }
        if (contact.bodyB == kArticulatedStaticWorld) {
            endpointsB[index].isStatic = true;
        } else if (contact.bodyB == contact.bodyA ||
                   !addEndpoint(
                       contact.bodyB,
                       contact.localPointB,
                       endpointsB[index]
                   )) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedContactStatus::invalidContact,
                static_cast<std::uint32_t>(index)
            );
        }
    }

    for (std::uint32_t articulationIndex = 0u;
         articulationIndex < grouped.size();
         ++articulationIndex) {
        ArticulationQueries& work = grouped[articulationIndex];
        if (work.queries.empty()) {
            continue;
        }
        const MRArticulationGPU& articulation =
            model.articulations[articulationIndex];
        work.kinematics.resize(work.queries.size());
        work.jacobians.assign(
            work.queries.size() * 3u * articulation.nv,
            0.0
        );
        const ArticulatedDynamicsDiagnostics result =
            computeArticulatedPointJacobians(
                model,
                articulationIndex,
                q.subspan(
                    articulation.qOffset,
                    articulation.nq
                ),
                freeVelocity.subspan(
                    articulation.vOffset,
                    articulation.nv
                ),
                work.queries,
                work.kinematics,
                work.jacobians,
                config
            );
        if (!result.succeeded()) {
            diagnostics.dynamicsStatus = result.status;
            return fail(
                std::move(diagnostics),
                MultiArticulatedContactStatus::
                    kinematicsFailure,
                MR_INVALID_INDEX,
                articulationIndex
            );
        }
    }

    MultiArticulationFactorCache factors;
    const MultiArticulatedWorldDiagnostics factorDiagnostics =
        buildMultiArticulationFactorCache(
            model,
            q,
            freeVelocity,
            factors,
            config
        );
    if (!factorDiagnostics.succeeded()) {
        diagnostics.maximumFactorResidual =
            factorDiagnostics.maximumFactorResidual;
        return fail(
            std::move(diagnostics),
            factorDiagnostics.status ==
                MultiArticulatedWorldStatus::
                    factorizationFailure
                ? MultiArticulatedContactStatus::
                      factorizationFailure
                : MultiArticulatedContactStatus::
                      kinematicsFailure,
            MR_INVALID_INDEX,
            factorDiagnostics.firstFailingArticulation
        );
    }

    MultiArticulatedContactProblem staged;
    staged.nv = model.world.nv;
    staged.articulatedNv = model.world.nv;
    staged.contactCount =
        static_cast<std::uint32_t>(contactCount);
    staged.factors = std::move(factors);
    staged.freeVelocity.assign(
        freeVelocity.begin(),
        freeVelocity.end()
    );
    staged.contactJacobian.assign(rowCount * nv, 0.0);
    staged.responseColumns.assign(rowCount * nv, 0.0);
    staged.prescribedContactVelocity.assign(
        rowCount,
        0.0
    );
    staged.conic.delassus.assign(
        rowCount * rowCount,
        0.0
    );
    staged.conic.freeContactVelocity.assign(
        rowCount,
        0.0
    );
    staged.conic.contacts.resize(contactCount);
    staged.pointA.resize(contactCount);
    staged.pointB.resize(contactCount);

    const auto point = [&](
        const EndpointQuery endpoint
    ) -> const ArticulatedPointKinematics& {
        return grouped[endpoint.articulation]
            .kinematics[endpoint.query];
    };
    const auto addPointJacobian = [&](
        const EndpointQuery endpoint,
        const double sign,
        const Vec3 axis,
        const std::size_t row
    ) {
        if (endpoint.isStatic) {
            return;
        }
        const MRArticulationGPU& articulation =
            model.articulations[endpoint.articulation];
        const ArticulationQueries& work =
            grouped[endpoint.articulation];
        const std::size_t base =
            endpoint.query * 3u * articulation.nv;
        for (std::size_t localDof = 0u;
             localDof < articulation.nv;
             ++localDof) {
            staged.contactJacobian[
                row * nv + articulation.vOffset + localDof
            ] += sign * (
                axis.x * work.jacobians[
                    base + 0u * articulation.nv + localDof
                ] +
                axis.y * work.jacobians[
                    base + 1u * articulation.nv + localDof
                ] +
                axis.z * work.jacobians[
                    base + 2u * articulation.nv + localDof
                ]
            );
        }
    };

    for (std::size_t contactIndex = 0u;
         contactIndex < contactCount;
         ++contactIndex) {
        const ArticulatedContact& contact =
            contacts[contactIndex];
        staged.pointA[contactIndex] =
            point(endpointsA[contactIndex]);
        if (endpointsB[contactIndex].isStatic) {
            staged.pointB[contactIndex].position =
                contact.localPointB;
            staged.pointB[contactIndex].linearVelocity =
                {0.0, 0.0, 0.0};
        } else {
            staged.pointB[contactIndex] =
                point(endpointsB[contactIndex]);
        }
        const std::array<Vec3, 3> axes{
            frames[contactIndex].normal,
            frames[contactIndex].tangentU,
            frames[contactIndex].tangentV,
        };
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            const std::size_t row =
                3u * contactIndex + axis;
            addPointJacobian(
                endpointsA[contactIndex],
                -1.0,
                axes[axis],
                row
            );
            addPointJacobian(
                endpointsB[contactIndex],
                1.0,
                axes[axis],
                row
            );
            double contactVelocity = 0.0;
            for (std::size_t dof = 0u;
                 dof < nv;
                 ++dof) {
                contactVelocity +=
                    staged.contactJacobian[row * nv + dof] *
                    freeVelocity[dof];
            }
            staged.conic.freeContactVelocity[row] =
                contactVelocity;
        }
        staged.conic.contacts[contactIndex] = {
            contact.targetVelocity,
            contact.regularization,
            contact.warmImpulse,
            contact.friction,
        };
    }

    std::vector<double> generalizedImpulse(nv, 0.0);
    std::vector<double> velocityDelta(nv, 0.0);
    for (std::size_t row = 0u; row < rowCount; ++row) {
        std::ranges::copy(
            std::span(
                staged.contactJacobian.data() + row * nv,
                nv
            ),
            generalizedImpulse.begin()
        );
        const MultiArticulatedWorldDiagnostics response =
            applyMultiArticulationInverseMass(
                model,
                staged.factors,
                generalizedImpulse,
                velocityDelta
            );
        if (!response.succeeded()) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedContactStatus::
                    factorizationFailure,
                MR_INVALID_INDEX,
                response.firstFailingArticulation
            );
        }
        diagnostics.maximumFactorResidual = std::max(
            diagnostics.maximumFactorResidual,
            response.maximumFactorResidual
        );
        std::ranges::copy(
            velocityDelta,
            staged.responseColumns.begin() + row * nv
        );
    }

    double scale = 1.0;
    diagnostics.minimumDelassusDiagonal =
        rowCount == 0u
        ? 0.0
        : std::numeric_limits<double>::infinity();
    for (std::size_t row = 0u; row < rowCount; ++row) {
        for (std::size_t column = 0u;
             column < rowCount;
             ++column) {
            double value = 0.0;
            for (std::size_t dof = 0u;
                 dof < nv;
                 ++dof) {
                value +=
                    staged.contactJacobian[row * nv + dof] *
                    staged.responseColumns[
                        column * nv + dof
                    ];
            }
            staged.conic.delassus[
                row * rowCount + column
            ] = value;
            scale = std::max(scale, std::abs(value));
        }
        diagnostics.minimumDelassusDiagonal = std::min(
            diagnostics.minimumDelassusDiagonal,
            staged.conic.delassus[
                row * rowCount + row
            ]
        );
    }
    for (std::size_t row = 0u; row < rowCount; ++row) {
        for (std::size_t column = row + 1u;
             column < rowCount;
             ++column) {
            const std::size_t left =
                row * rowCount + column;
            const std::size_t right =
                column * rowCount + row;
            diagnostics.maximumDelassusAsymmetry =
                std::max(
                    diagnostics.maximumDelassusAsymmetry,
                    std::abs(
                        staged.conic.delassus[left] -
                        staged.conic.delassus[right]
                    )
                );
            const double symmetric = 0.5 * (
                staged.conic.delassus[left] +
                staged.conic.delassus[right]
            );
            staged.conic.delassus[left] = symmetric;
            staged.conic.delassus[right] = symmetric;
        }
    }
    const double tolerance =
        4096.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(std::max(rowCount, nv)) *
        scale;
    if (!finite(
            std::span<const double>(
                staged.contactJacobian
            )
        ) ||
        !finite(
            std::span<const double>(
                staged.responseColumns
            )
        ) ||
        !finite(
            std::span<const double>(
                staged.conic.delassus
            )
        ) ||
        !finite(
            std::span<const double>(
                staged.conic.freeContactVelocity
            )
        )) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::nonfiniteResult
        );
    }
    if (diagnostics.maximumDelassusAsymmetry >
            tolerance ||
        diagnostics.minimumDelassusDiagonal < -tolerance) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::
                factorizationFailure
        );
    }

    output = std::move(staged);
    return diagnostics;
}

MultiArticulatedContactDiagnostics
buildMultiArticulatedIslandContactProblem(
    const EngineModel& model,
    const std::span<const double> q,
    const std::span<const double> freeArticulationVelocity,
    const std::span<const MRBodyStateGPU> sceneBodies,
    const std::span<const MultiArticulatedIslandContact> contacts,
    MultiArticulatedContactProblem& output,
    const ArticulatedDynamicsConfig& config
) {
    MultiArticulatedContactDiagnostics diagnostics =
        diagnosticsFor(model, contacts.size());
    std::string reason;
    if (!model.valid(&reason)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::invalidModel
        );
    }
    if (contacts.size() >
            std::numeric_limits<std::uint32_t>::max() / 3u ||
        q.size() != model.world.nq ||
        freeArticulationVelocity.size() != model.world.nv) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::invalidDimensions
        );
    }
    if (!finite(q) || !finite(freeArticulationVelocity)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::nonfiniteInput
        );
    }

    const auto finite4 = [](const mr_float4 value) {
        return
            std::isfinite(value.x) &&
            std::isfinite(value.y) &&
            std::isfinite(value.z) &&
            std::isfinite(value.w);
    };
    const auto rotate = [](
        const mr_float4 quaternion,
        const std::array<double, 3>& value
    ) {
        const Vec3 q{
            quaternion.x,
            quaternion.y,
            quaternion.z,
        };
        const Vec3 input = vector(value);
        const Vec3 first = cross(q, input);
        const Vec3 doubled{
            2.0 * first.x,
            2.0 * first.y,
            2.0 * first.z,
        };
        const Vec3 second = cross(q, doubled);
        return Vec3{
            input.x + quaternion.w * doubled.x + second.x,
            input.y + quaternion.w * doubled.y + second.y,
            input.z + quaternion.w * doubled.z + second.z,
        };
    };
    const auto add = [](const Vec3 left, const Vec3 right) {
        return Vec3{
            left.x + right.x,
            left.y + right.y,
            left.z + right.z,
        };
    };
    const auto array = [](const Vec3 value) {
        return std::array<double, 3>{
            value.x,
            value.y,
            value.z,
        };
    };
    const auto scenePoint = [&](
        const MRBodyStateGPU& body,
        const std::array<double, 3>& localPoint
    ) {
        ArticulatedPointKinematics result;
        const Vec3 offset = rotate(
            body.orientation,
            localPoint
        );
        const Vec3 position = add(
            Vec3{
                body.position.x,
                body.position.y,
                body.position.z,
            },
            offset
        );
        const Vec3 pointVelocity = add(
            Vec3{
                body.linearVelocityAndInverseMass.x,
                body.linearVelocityAndInverseMass.y,
                body.linearVelocityAndInverseMass.z,
            },
            cross(
                Vec3{
                    body.angularVelocity.x,
                    body.angularVelocity.y,
                    body.angularVelocity.z,
                },
                offset
            )
        );
        result.position = array(position);
        result.linearVelocity = array(pointVelocity);
        return result;
    };
    const auto validInverseInertia = [&](
        const MRBodyStateGPU& body
    ) {
        const std::array<double, 9> inverse{
            body.inverseInertiaWorldRow0.x,
            body.inverseInertiaWorldRow0.y,
            body.inverseInertiaWorldRow0.z,
            body.inverseInertiaWorldRow1.x,
            body.inverseInertiaWorldRow1.y,
            body.inverseInertiaWorldRow1.z,
            body.inverseInertiaWorldRow2.x,
            body.inverseInertiaWorldRow2.y,
            body.inverseInertiaWorldRow2.z,
        };
        if (!finite(std::span<const double>(inverse))) {
            return false;
        }
        double scale = 1.0;
        for (const double value : inverse) {
            scale = std::max(scale, std::abs(value));
        }
        const double symmetryTolerance = 2.0e-5 * scale;
        if (std::abs(inverse[1] - inverse[3]) >
                symmetryTolerance ||
            std::abs(inverse[2] - inverse[6]) >
                symmetryTolerance ||
            std::abs(inverse[5] - inverse[7]) >
                symmetryTolerance) {
            return false;
        }
        const double a00 = inverse[0];
        const double a01 = 0.5 * (inverse[1] + inverse[3]);
        const double a02 = 0.5 * (inverse[2] + inverse[6]);
        const double a11 = inverse[4];
        const double a12 = 0.5 * (inverse[5] + inverse[7]);
        const double a22 = inverse[8];
        const double determinant =
            a00 * (a11 * a22 - a12 * a12) -
            a01 * (a01 * a22 - a12 * a02) +
            a02 * (a01 * a12 - a11 * a02);
        return a00 > 0.0 &&
            a00 * a11 - a01 * a01 > 0.0 &&
            determinant > 0.0 &&
            finite(determinant);
    };

    std::vector<std::uint32_t> sceneOffsets(
        sceneBodies.size(),
        MR_INVALID_INDEX
    );
    std::size_t totalNv = model.world.nv;
    for (std::size_t bodyIndex = 0u;
         bodyIndex < sceneBodies.size();
         ++bodyIndex) {
        const MRBodyStateGPU& body = sceneBodies[bodyIndex];
        const std::uint32_t motion = body.flagsAndIndices[0];
        const bool basicFinite =
            finite4(body.position) &&
            finite4(body.orientation) &&
            finite4(body.linearVelocityAndInverseMass) &&
            finite4(body.angularVelocity) &&
            finite4(body.inverseInertiaWorldRow0) &&
            finite4(body.inverseInertiaWorldRow1) &&
            finite4(body.inverseInertiaWorldRow2);
        const double quaternionNorm = std::sqrt(
            double(body.orientation.x) * body.orientation.x +
            double(body.orientation.y) * body.orientation.y +
            double(body.orientation.z) * body.orientation.z +
            double(body.orientation.w) * body.orientation.w
        );
        if (!basicFinite ||
            !finite(quaternionNorm) ||
            std::abs(quaternionNorm - 1.0) > 2.0e-4 ||
            body.flagsAndIndices[1] != MR_INVALID_INDEX ||
            (motion != MR_MOTION_DYNAMIC &&
             motion != MR_MOTION_KINEMATIC &&
             motion != MR_MOTION_STATIC)) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedContactStatus::invalidContact
            );
        }
        if (motion == MR_MOTION_DYNAMIC) {
            if (!(body.linearVelocityAndInverseMass.w > 0.0f) ||
                !validInverseInertia(body) ||
                totalNv >
                    std::numeric_limits<std::uint32_t>::max() -
                        6u) {
                return fail(
                    std::move(diagnostics),
                    MultiArticulatedContactStatus::
                        invalidContact
                );
            }
            sceneOffsets[bodyIndex] =
                static_cast<std::uint32_t>(totalNv);
            totalNv += 6u;
        }
    }

    struct IslandEndpointQuery {
        MultiContactEndpointKind kind =
            MultiContactEndpointKind::staticWorld;
        std::uint32_t articulation = MR_INVALID_INDEX;
        std::size_t query =
            std::numeric_limits<std::size_t>::max();
        std::uint32_t sceneBody = MR_INVALID_INDEX;
        std::uint32_t velocityOffset = MR_INVALID_INDEX;
        std::array<double, 3> worldPoint{};
    };
    const std::size_t contactCount = contacts.size();
    const std::size_t rowCount = 3u * contactCount;
    std::vector<ContactFrame> frames(contactCount);
    std::vector<IslandEndpointQuery> endpointsA(contactCount);
    std::vector<IslandEndpointQuery> endpointsB(contactCount);
    std::vector<ArticulationQueries> grouped(
        model.articulations.size()
    );
    const auto addEndpoint = [&](
        const MultiContactEndpoint& source,
        IslandEndpointQuery& endpoint
    ) {
        endpoint.kind = source.kind;
        if (source.kind ==
            MultiContactEndpointKind::staticWorld) {
            endpoint.worldPoint = source.localPoint;
            return finite(source.localPoint);
        }
        if (source.kind ==
            MultiContactEndpointKind::sceneBody) {
            if (source.body >= sceneBodies.size() ||
                !finite(source.localPoint)) {
                return false;
            }
            endpoint.sceneBody = source.body;
            endpoint.velocityOffset =
                sceneOffsets[source.body];
            return true;
        }
        if (source.kind !=
                MultiContactEndpointKind::articulatedBody ||
            source.body >= model.bodies.size() ||
            !finite(source.localPoint)) {
            return false;
        }
        const std::uint32_t articulation =
            model.bodies[source.body].articulationIndex;
        if (articulation == MR_INVALID_INDEX ||
            articulation >= model.articulations.size()) {
            return false;
        }
        endpoint.articulation = articulation;
        endpoint.query = grouped[articulation].queries.size();
        grouped[articulation].queries.push_back({
            source.body,
            source.localPoint,
        });
        return true;
    };
    const auto dynamicEndpoint = [&](
        const IslandEndpointQuery& endpoint
    ) {
        return endpoint.kind ==
                MultiContactEndpointKind::articulatedBody ||
            (endpoint.kind ==
                 MultiContactEndpointKind::sceneBody &&
             endpoint.velocityOffset != MR_INVALID_INDEX);
    };

    for (std::size_t index = 0u;
         index < contactCount;
         ++index) {
        const MultiArticulatedIslandContact& contact =
            contacts[index];
        const bool finiteContact =
            finite(contact.endpointA.localPoint) &&
            finite(contact.endpointB.localPoint) &&
            finite(contact.targetVelocity) &&
            finite(contact.regularization) &&
            finite(contact.warmImpulse) &&
            finite(contact.friction);
        if (!finiteContact ||
            contact.friction < 0.0 ||
            !std::ranges::all_of(
                contact.regularization,
                [](const double value) {
                    return finite(value) && value > 0.0;
                }
            ) ||
            !makeFrame(contact, frames[index]) ||
            !addEndpoint(
                contact.endpointA,
                endpointsA[index]
            ) ||
            !addEndpoint(
                contact.endpointB,
                endpointsB[index]
            ) ||
            (contact.endpointA.kind ==
                 contact.endpointB.kind &&
             contact.endpointA.kind !=
                 MultiContactEndpointKind::staticWorld &&
             contact.endpointA.body ==
                 contact.endpointB.body) ||
            (!dynamicEndpoint(endpointsA[index]) &&
             !dynamicEndpoint(endpointsB[index]))) {
            return fail(
                std::move(diagnostics),
                finiteContact
                    ? MultiArticulatedContactStatus::
                          invalidContact
                    : MultiArticulatedContactStatus::
                          nonfiniteInput,
                static_cast<std::uint32_t>(index)
            );
        }
    }

    for (std::uint32_t articulationIndex = 0u;
         articulationIndex < grouped.size();
         ++articulationIndex) {
        ArticulationQueries& work = grouped[articulationIndex];
        if (work.queries.empty()) {
            continue;
        }
        const MRArticulationGPU& articulation =
            model.articulations[articulationIndex];
        work.kinematics.resize(work.queries.size());
        work.jacobians.assign(
            work.queries.size() * 3u * articulation.nv,
            0.0
        );
        const ArticulatedDynamicsDiagnostics result =
            computeArticulatedPointJacobians(
                model,
                articulationIndex,
                q.subspan(
                    articulation.qOffset,
                    articulation.nq
                ),
                freeArticulationVelocity.subspan(
                    articulation.vOffset,
                    articulation.nv
                ),
                work.queries,
                work.kinematics,
                work.jacobians,
                config
            );
        if (!result.succeeded()) {
            diagnostics.dynamicsStatus = result.status;
            return fail(
                std::move(diagnostics),
                MultiArticulatedContactStatus::
                    kinematicsFailure,
                MR_INVALID_INDEX,
                articulationIndex
            );
        }
    }

    MultiArticulationFactorCache factors;
    const MultiArticulatedWorldDiagnostics factorDiagnostics =
        buildMultiArticulationFactorCache(
            model,
            q,
            freeArticulationVelocity,
            factors,
            config
        );
    if (!factorDiagnostics.succeeded()) {
        return fail(
            std::move(diagnostics),
            factorDiagnostics.status ==
                MultiArticulatedWorldStatus::factorizationFailure
                ? MultiArticulatedContactStatus::
                      factorizationFailure
                : MultiArticulatedContactStatus::
                      kinematicsFailure,
            MR_INVALID_INDEX,
            factorDiagnostics.firstFailingArticulation
        );
    }

    MultiArticulatedContactProblem staged;
    staged.nv = static_cast<std::uint32_t>(totalNv);
    staged.articulatedNv = model.world.nv;
    staged.contactCount =
        static_cast<std::uint32_t>(contactCount);
    staged.factors = std::move(factors);
    staged.sceneBodyVelocityOffsets = sceneOffsets;
    staged.sceneBodyFreeVelocities.resize(
        sceneBodies.size()
    );
    staged.sceneBodyStates.assign(
        sceneBodies.begin(),
        sceneBodies.end()
    );
    staged.freeVelocity.assign(totalNv, 0.0);
    std::ranges::copy(
        freeArticulationVelocity,
        staged.freeVelocity.begin()
    );
    for (std::size_t bodyIndex = 0u;
         bodyIndex < sceneBodies.size();
         ++bodyIndex) {
        const MRBodyStateGPU& body = sceneBodies[bodyIndex];
        staged.sceneBodyFreeVelocities[bodyIndex] = {
            {
                body.linearVelocityAndInverseMass.x,
                body.linearVelocityAndInverseMass.y,
                body.linearVelocityAndInverseMass.z,
            },
            {
                body.angularVelocity.x,
                body.angularVelocity.y,
                body.angularVelocity.z,
            },
        };
        const std::uint32_t offset = sceneOffsets[bodyIndex];
        if (offset == MR_INVALID_INDEX) {
            continue;
        }
        staged.freeVelocity[offset + 0u] =
            body.linearVelocityAndInverseMass.x;
        staged.freeVelocity[offset + 1u] =
            body.linearVelocityAndInverseMass.y;
        staged.freeVelocity[offset + 2u] =
            body.linearVelocityAndInverseMass.z;
        staged.freeVelocity[offset + 3u] =
            body.angularVelocity.x;
        staged.freeVelocity[offset + 4u] =
            body.angularVelocity.y;
        staged.freeVelocity[offset + 5u] =
            body.angularVelocity.z;
    }
    staged.contactJacobian.assign(
        rowCount * totalNv,
        0.0
    );
    staged.responseColumns.assign(
        rowCount * totalNv,
        0.0
    );
    staged.prescribedContactVelocity.assign(
        rowCount,
        0.0
    );
    staged.conic.delassus.assign(
        rowCount * rowCount,
        0.0
    );
    staged.conic.freeContactVelocity.assign(
        rowCount,
        0.0
    );
    staged.conic.contacts.resize(contactCount);
    staged.pointA.resize(contactCount);
    staged.pointB.resize(contactCount);

    const auto pointFor = [&](
        const IslandEndpointQuery& endpoint,
        const MultiContactEndpoint& source
    ) {
        if (endpoint.kind ==
            MultiContactEndpointKind::articulatedBody) {
            return grouped[endpoint.articulation]
                .kinematics[endpoint.query];
        }
        if (endpoint.kind ==
            MultiContactEndpointKind::sceneBody) {
            return scenePoint(
                sceneBodies[endpoint.sceneBody],
                source.localPoint
            );
        }
        ArticulatedPointKinematics result;
        result.position = endpoint.worldPoint;
        return result;
    };
    const auto addEndpointJacobian = [&](
        const IslandEndpointQuery& endpoint,
        const MultiContactEndpoint& source,
        const double sign,
        const Vec3 axis,
        const std::size_t row
    ) {
        if (endpoint.kind ==
            MultiContactEndpointKind::articulatedBody) {
            const MRArticulationGPU& articulation =
                model.articulations[endpoint.articulation];
            const ArticulationQueries& work =
                grouped[endpoint.articulation];
            const std::size_t base =
                endpoint.query * 3u * articulation.nv;
            for (std::size_t localDof = 0u;
                 localDof < articulation.nv;
                 ++localDof) {
                staged.contactJacobian[
                    row * totalNv +
                    articulation.vOffset + localDof
                ] += sign * (
                    axis.x * work.jacobians[
                        base + 0u * articulation.nv +
                        localDof
                    ] +
                    axis.y * work.jacobians[
                        base + 1u * articulation.nv +
                        localDof
                    ] +
                    axis.z * work.jacobians[
                        base + 2u * articulation.nv +
                        localDof
                    ]
                );
            }
            return;
        }
        if (endpoint.kind !=
            MultiContactEndpointKind::sceneBody) {
            return;
        }
        const MRBodyStateGPU& body =
            sceneBodies[endpoint.sceneBody];
        const Vec3 offset = rotate(
            body.orientation,
            source.localPoint
        );
        if (endpoint.velocityOffset == MR_INVALID_INDEX) {
            const ArticulatedPointKinematics prescribed =
                scenePoint(body, source.localPoint);
            staged.prescribedContactVelocity[row] +=
                sign * dot(
                    axis,
                    vector(prescribed.linearVelocity)
                );
            return;
        }
        const std::size_t base = endpoint.velocityOffset;
        staged.contactJacobian[
            row * totalNv + base + 0u
        ] += sign * axis.x;
        staged.contactJacobian[
            row * totalNv + base + 1u
        ] += sign * axis.y;
        staged.contactJacobian[
            row * totalNv + base + 2u
        ] += sign * axis.z;
        const Vec3 angular = cross(offset, axis);
        staged.contactJacobian[
            row * totalNv + base + 3u
        ] += sign * angular.x;
        staged.contactJacobian[
            row * totalNv + base + 4u
        ] += sign * angular.y;
        staged.contactJacobian[
            row * totalNv + base + 5u
        ] += sign * angular.z;
    };

    for (std::size_t contactIndex = 0u;
         contactIndex < contactCount;
         ++contactIndex) {
        const MultiArticulatedIslandContact& contact =
            contacts[contactIndex];
        staged.pointA[contactIndex] = pointFor(
            endpointsA[contactIndex],
            contact.endpointA
        );
        staged.pointB[contactIndex] = pointFor(
            endpointsB[contactIndex],
            contact.endpointB
        );
        const std::array<Vec3, 3> axes{
            frames[contactIndex].normal,
            frames[contactIndex].tangentU,
            frames[contactIndex].tangentV,
        };
        for (std::size_t axisIndex = 0u;
             axisIndex < 3u;
             ++axisIndex) {
            const std::size_t row =
                3u * contactIndex + axisIndex;
            addEndpointJacobian(
                endpointsA[contactIndex],
                contact.endpointA,
                -1.0,
                axes[axisIndex],
                row
            );
            addEndpointJacobian(
                endpointsB[contactIndex],
                contact.endpointB,
                1.0,
                axes[axisIndex],
                row
            );
            double velocity =
                staged.prescribedContactVelocity[row];
            for (std::size_t dof = 0u;
                 dof < totalNv;
                 ++dof) {
                velocity +=
                    staged.contactJacobian[
                        row * totalNv + dof
                    ] * staged.freeVelocity[dof];
            }
            staged.conic.freeContactVelocity[row] =
                velocity;
        }
        staged.conic.contacts[contactIndex] = {
            contact.targetVelocity,
            contact.regularization,
            contact.warmImpulse,
            contact.friction,
        };
    }

    std::vector<double> articulatedImpulse(
        model.world.nv,
        0.0
    );
    std::vector<double> articulatedResponse(
        model.world.nv,
        0.0
    );
    for (std::size_t row = 0u;
         row < rowCount;
         ++row) {
        std::ranges::copy_n(
            staged.contactJacobian.begin() +
                row * totalNv,
            model.world.nv,
            articulatedImpulse.begin()
        );
        const MultiArticulatedWorldDiagnostics response =
            applyMultiArticulationInverseMass(
                model,
                staged.factors,
                articulatedImpulse,
                articulatedResponse
            );
        if (!response.succeeded()) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedContactStatus::
                    factorizationFailure,
                MR_INVALID_INDEX,
                response.firstFailingArticulation
            );
        }
        diagnostics.maximumFactorResidual = std::max(
            diagnostics.maximumFactorResidual,
            response.maximumFactorResidual
        );
        std::ranges::copy(
            articulatedResponse,
            staged.responseColumns.begin() +
                row * totalNv
        );
        for (std::size_t bodyIndex = 0u;
             bodyIndex < sceneBodies.size();
             ++bodyIndex) {
            const std::uint32_t offset =
                sceneOffsets[bodyIndex];
            if (offset == MR_INVALID_INDEX) {
                continue;
            }
            const MRBodyStateGPU& body =
                sceneBodies[bodyIndex];
            const double inverseMass =
                body.linearVelocityAndInverseMass.w;
            for (std::size_t axis = 0u;
                 axis < 3u;
                 ++axis) {
                staged.responseColumns[
                    row * totalNv + offset + axis
                ] =
                    inverseMass *
                    staged.contactJacobian[
                        row * totalNv + offset + axis
                    ];
            }
            const std::array<double, 3> angularImpulse{
                staged.contactJacobian[
                    row * totalNv + offset + 3u
                ],
                staged.contactJacobian[
                    row * totalNv + offset + 4u
                ],
                staged.contactJacobian[
                    row * totalNv + offset + 5u
                ],
            };
            const std::array<mr_float4, 3> inertiaRows{
                body.inverseInertiaWorldRow0,
                body.inverseInertiaWorldRow1,
                body.inverseInertiaWorldRow2,
            };
            for (std::size_t axis = 0u;
                 axis < 3u;
                 ++axis) {
                staged.responseColumns[
                    row * totalNv + offset + 3u + axis
                ] =
                    inertiaRows[axis].x *
                        angularImpulse[0] +
                    inertiaRows[axis].y *
                        angularImpulse[1] +
                    inertiaRows[axis].z *
                        angularImpulse[2];
            }
        }
    }

    double delassusScale = 1.0;
    diagnostics.minimumDelassusDiagonal =
        rowCount == 0u
        ? 0.0
        : std::numeric_limits<double>::infinity();
    for (std::size_t row = 0u;
         row < rowCount;
         ++row) {
        for (std::size_t column = 0u;
             column < rowCount;
             ++column) {
            double value = 0.0;
            for (std::size_t dof = 0u;
                 dof < totalNv;
                 ++dof) {
                value +=
                    staged.contactJacobian[
                        row * totalNv + dof
                    ] *
                    staged.responseColumns[
                        column * totalNv + dof
                    ];
            }
            staged.conic.delassus[
                row * rowCount + column
            ] = value;
            delassusScale = std::max(
                delassusScale,
                std::abs(value)
            );
        }
        diagnostics.minimumDelassusDiagonal = std::min(
            diagnostics.minimumDelassusDiagonal,
            staged.conic.delassus[
                row * rowCount + row
            ]
        );
    }
    for (std::size_t row = 0u;
         row < rowCount;
         ++row) {
        for (std::size_t column = row + 1u;
             column < rowCount;
             ++column) {
            const std::size_t left =
                row * rowCount + column;
            const std::size_t right =
                column * rowCount + row;
            diagnostics.maximumDelassusAsymmetry =
                std::max(
                    diagnostics.maximumDelassusAsymmetry,
                    std::abs(
                        staged.conic.delassus[left] -
                        staged.conic.delassus[right]
                    )
                );
            const double symmetric = 0.5 * (
                staged.conic.delassus[left] +
                staged.conic.delassus[right]
            );
            staged.conic.delassus[left] = symmetric;
            staged.conic.delassus[right] = symmetric;
        }
    }
    const double tolerance =
        4096.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(
            std::max(rowCount, totalNv)
        ) * delassusScale;
    if (!finite(
            std::span<const double>(
                staged.contactJacobian
            )
        ) ||
        !finite(
            std::span<const double>(
                staged.responseColumns
            )
        ) ||
        !finite(
            std::span<const double>(
                staged.prescribedContactVelocity
            )
        ) ||
        !finite(
            std::span<const double>(
                staged.conic.delassus
            )
        ) ||
        !finite(
            std::span<const double>(
                staged.conic.freeContactVelocity
            )
        )) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::nonfiniteResult
        );
    }
    if (diagnostics.maximumDelassusAsymmetry >
            tolerance ||
        diagnostics.minimumDelassusDiagonal < -tolerance) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::
                factorizationFailure
        );
    }
    output = std::move(staged);
    return diagnostics;
}

MultiArticulatedContactDiagnostics
projectMultiArticulatedContactThroughGeneralizedEqualities(
    const EngineModel& model,
    MultiArticulatedContactProblem& problem,
    const ConstraintIREvaluationConfig& config
) {
    MultiArticulatedContactDiagnostics diagnostics =
        diagnosticsFor(model, problem.contactCount);
    std::string reason;
    EngineModel topologyModel = model;
    topologyModel.constraintProgram = {};
    if (!topologyModel.valid(&reason)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::invalidModel
        );
    }
    if (!structurallyValid(problem) ||
        problem.articulatedNv != model.world.nv ||
        problem.generalizedConstraintRowCount != 0u) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::invalidDimensions
        );
    }
    if (model.constraintProgram.rows.empty()) {
        return diagnostics;
    }
    std::vector<double> equalityJacobian;
    if (!buildGeneralizedEqualityJacobian(
            model.constraintProgram,
            problem.nv,
            equalityJacobian
        )) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::
                unsupportedConstraint
        );
    }
    const std::size_t equalityRows =
        model.constraintProgram.rows.size();
    const std::size_t contactRows =
        3u * static_cast<std::size_t>(problem.contactCount);
    std::vector<float> relative(equalityRows, 0.0f);
    for (std::size_t row = 0u;
         row < equalityRows;
         ++row) {
        double value = 0.0;
        for (std::uint32_t dof = 0u;
             dof < problem.nv;
             ++dof) {
            value +=
                equalityJacobian[
                    row * problem.nv + dof
                ] * problem.freeVelocity[dof];
        }
        if (!finite(value) ||
            std::abs(value) >
                std::numeric_limits<float>::max()) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedContactStatus::
                    nonfiniteInput
            );
        }
        relative[row] = static_cast<float>(value);
    }
    const ConstraintIREvaluationResult evaluation =
        evaluateConstraintIR(
            model.constraintProgram,
            {relative, {}},
            config
        );
    if (!evaluation.succeeded() ||
        evaluation.evaluated.rows.size() != equalityRows) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::
                constraintEvaluationFailure
        );
    }

    MultiArticulatedContactProblem staged = problem;
    staged.generalizedConstraintRowCount =
        static_cast<std::uint32_t>(equalityRows);
    staged.generalizedConstraintJacobian.assign(
        equalityRows * staged.nv,
        0.0
    );
    staged.generalizedConstraintResponseColumns.assign(
        equalityRows * staged.nv,
        0.0
    );
    staged.generalizedConstraintFreeImpulses.assign(
        equalityRows,
        0.0
    );
    staged.generalizedConstraintContactCoupling.assign(
        equalityRows * contactRows,
        0.0
    );
    staged.generalizedConstraintTargets.resize(equalityRows);
    staged.generalizedConstraintRegularization.resize(
        equalityRows
    );
    for (std::size_t row = 0u;
         row < equalityRows;
         ++row) {
        const EvaluatedConstraintIRRow& semantics =
            evaluation.evaluated.rows[row];
        if (semantics.impulseLower >
                -0.5f * kConstraintIRUnbounded ||
            semantics.impulseUpper <
                0.5f * kConstraintIRUnbounded ||
            !finite(semantics.targetVelocity) ||
            !finite(semantics.regularization) ||
            semantics.regularization < 0.0f) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedContactStatus::
                    unsupportedConstraint
            );
        }
        staged.generalizedConstraintTargets[row] =
            semantics.targetVelocity;
        staged.generalizedConstraintRegularization[row] =
            semantics.regularization;
        std::ranges::copy_n(
            equalityJacobian.begin() +
                static_cast<std::ptrdiff_t>(
                    row * problem.nv
                ),
            problem.nv,
            staged.generalizedConstraintJacobian.begin() +
                static_cast<std::ptrdiff_t>(
                    row * staged.nv
                )
        );
    }

    std::vector<double> rhs(problem.nv, 0.0);
    std::vector<double> response(problem.nv, 0.0);
    std::vector<double> articulatedRhs(
        model.world.nv,
        0.0
    );
    std::vector<double> articulatedResponse(
        model.world.nv,
        0.0
    );
    for (std::size_t row = 0u;
         row < equalityRows;
         ++row) {
        std::ranges::copy_n(
            equalityJacobian.begin() +
                static_cast<std::ptrdiff_t>(
                    row * problem.nv
                ),
            problem.nv,
            rhs.begin()
        );
        std::ranges::copy_n(
            rhs.begin(),
            model.world.nv,
            articulatedRhs.begin()
        );
        const MultiArticulatedWorldDiagnostics applied =
            applyMultiArticulationInverseMass(
                model,
                staged.factors,
                articulatedRhs,
                articulatedResponse
            );
        if (!applied.succeeded()) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedContactStatus::
                    factorizationFailure,
                MR_INVALID_INDEX,
                applied.firstFailingArticulation
            );
        }
        std::ranges::fill(response, 0.0);
        std::ranges::copy_n(
            articulatedResponse.begin(),
            model.world.nv,
            response.begin()
        );
        for (std::size_t bodyIndex = 0u;
             bodyIndex <
                 staged.sceneBodyVelocityOffsets.size();
             ++bodyIndex) {
            const std::uint32_t offset =
                staged.sceneBodyVelocityOffsets[bodyIndex];
            if (offset == MR_INVALID_INDEX) {
                continue;
            }
            const MRBodyStateGPU& body =
                staged.sceneBodyStates[bodyIndex];
            for (std::size_t axis = 0u;
                 axis < 3u;
                 ++axis) {
                response[offset + axis] =
                    body.linearVelocityAndInverseMass.w *
                    rhs[offset + axis];
            }
            const std::array<mr_float4, 3> inertiaRows{
                body.inverseInertiaWorldRow0,
                body.inverseInertiaWorldRow1,
                body.inverseInertiaWorldRow2,
            };
            for (std::size_t axis = 0u;
                 axis < 3u;
                 ++axis) {
                response[offset + 3u + axis] =
                    inertiaRows[axis].x *
                        rhs[offset + 3u] +
                    inertiaRows[axis].y *
                        rhs[offset + 4u] +
                    inertiaRows[axis].z *
                        rhs[offset + 5u];
            }
        }
        std::ranges::copy(
            response,
            staged.generalizedConstraintResponseColumns
                .begin() +
                static_cast<std::ptrdiff_t>(
                    row * staged.nv
                )
        );
    }

    std::vector<double> equalityOperator(
        equalityRows * equalityRows,
        0.0
    );
    double maximumAsymmetry = 0.0;
    for (std::size_t row = 0u;
         row < equalityRows;
         ++row) {
        for (std::size_t column = 0u;
             column < equalityRows;
             ++column) {
            double value = 0.0;
            for (std::uint32_t dof = 0u;
                 dof < problem.nv;
                 ++dof) {
                value +=
                    equalityJacobian[
                        row * problem.nv + dof
                    ] *
                    staged
                        .generalizedConstraintResponseColumns[
                            column * staged.nv + dof
                        ];
            }
            equalityOperator[
                row * equalityRows + column
            ] = value;
        }
    }
    for (std::size_t row = 0u;
         row < equalityRows;
         ++row) {
        for (std::size_t column = row + 1u;
             column < equalityRows;
             ++column) {
            const double left =
                equalityOperator[
                    row * equalityRows + column
                ];
            const double right =
                equalityOperator[
                    column * equalityRows + row
                ];
            maximumAsymmetry = std::max(
                maximumAsymmetry,
                std::abs(left - right)
            );
            const double symmetric = 0.5 * (left + right);
            equalityOperator[
                row * equalityRows + column
            ] = symmetric;
            equalityOperator[
                column * equalityRows + row
            ] = symmetric;
        }
        equalityOperator[row * equalityRows + row] +=
            staged.generalizedConstraintRegularization[row];
    }
    diagnostics.maximumDelassusAsymmetry =
        maximumAsymmetry;
    std::vector<double> equalityFactor = equalityOperator;
    if (!factorDenseSPD(equalityFactor, equalityRows)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::
                constraintFactorizationFailure
        );
    }

    std::vector<double> equalityRight(equalityRows, 0.0);
    for (std::size_t row = 0u;
         row < equalityRows;
         ++row) {
        equalityRight[row] =
            staged.generalizedConstraintTargets[row] -
            relative[row];
    }
    if (!solveDenseSPD(
            equalityFactor,
            equalityRight,
            staged.generalizedConstraintFreeImpulses
        )) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::
                constraintFactorizationFailure
        );
    }

    std::vector<double> cross(equalityRows, 0.0);
    std::vector<double> coupling(equalityRows, 0.0);
    for (std::size_t contactRow = 0u;
         contactRow < contactRows;
         ++contactRow) {
        for (std::size_t equalityRow = 0u;
             equalityRow < equalityRows;
             ++equalityRow) {
            double value = 0.0;
            for (std::uint32_t dof = 0u;
                 dof < problem.nv;
                 ++dof) {
                value +=
                    equalityJacobian[
                        equalityRow * problem.nv + dof
                    ] *
                    staged.responseColumns[
                        contactRow * staged.nv + dof
                    ];
            }
            cross[equalityRow] = value;
        }
        std::ranges::fill(coupling, 0.0);
        if (!solveDenseSPD(
                equalityFactor,
                cross,
                coupling
            )) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedContactStatus::
                    constraintFactorizationFailure
            );
        }
        for (std::size_t equalityRow = 0u;
             equalityRow < equalityRows;
             ++equalityRow) {
            staged.generalizedConstraintContactCoupling[
                equalityRow * contactRows + contactRow
            ] = coupling[equalityRow];
        }
        for (std::size_t dof = 0u;
             dof < staged.nv;
             ++dof) {
            double correction = 0.0;
            for (std::size_t equalityRow = 0u;
                 equalityRow < equalityRows;
                 ++equalityRow) {
                correction +=
                    staged
                        .generalizedConstraintResponseColumns[
                            equalityRow * staged.nv + dof
                        ] * coupling[equalityRow];
            }
            staged.responseColumns[
                contactRow * staged.nv + dof
            ] -= correction;
        }
    }
    for (std::size_t dof = 0u;
         dof < staged.nv;
         ++dof) {
        double correction = 0.0;
        for (std::size_t row = 0u;
             row < equalityRows;
             ++row) {
            correction +=
                staged.generalizedConstraintResponseColumns[
                    row * staged.nv + dof
                ] *
                staged.generalizedConstraintFreeImpulses[row];
        }
        staged.freeVelocity[dof] += correction;
    }

    for (std::size_t row = 0u;
         row < contactRows;
         ++row) {
        double freeValue =
            staged.prescribedContactVelocity[row];
        for (std::size_t dof = 0u;
             dof < staged.nv;
             ++dof) {
            freeValue +=
                staged.contactJacobian[
                    row * staged.nv + dof
                ] * staged.freeVelocity[dof];
        }
        staged.conic.freeContactVelocity[row] = freeValue;
        for (std::size_t column = 0u;
             column < contactRows;
             ++column) {
            double value = 0.0;
            for (std::size_t dof = 0u;
                 dof < staged.nv;
                 ++dof) {
                value +=
                    staged.contactJacobian[
                        row * staged.nv + dof
                    ] *
                    staged.responseColumns[
                        column * staged.nv + dof
                    ];
            }
            staged.conic.delassus[
                row * contactRows + column
            ] = value;
        }
    }
    if (!finite(staged.freeVelocity) ||
        !finite(staged.responseColumns) ||
        !finite(staged.conic.freeContactVelocity) ||
        !finite(staged.conic.delassus)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::nonfiniteResult
        );
    }
    diagnostics.minimumDelassusDiagonal =
        std::numeric_limits<double>::infinity();
    diagnostics.maximumDelassusAsymmetry = 0.0;
    for (std::size_t row = 0u;
         row < contactRows;
         ++row) {
        diagnostics.minimumDelassusDiagonal = std::min(
            diagnostics.minimumDelassusDiagonal,
            staged.conic.delassus[
                row * contactRows + row
            ]
        );
        for (std::size_t column = row + 1u;
             column < contactRows;
             ++column) {
            diagnostics.maximumDelassusAsymmetry = std::max(
                diagnostics.maximumDelassusAsymmetry,
                std::abs(
                    staged.conic.delassus[
                        row * contactRows + column
                    ] -
                    staged.conic.delassus[
                        column * contactRows + row
                    ]
                )
            );
        }
    }
    problem = std::move(staged);
    return diagnostics;
}

MultiArticulatedContactDiagnostics
projectMultiArticulatedContactThroughPointEqualities(
    const EngineModel& model,
    const std::span<const double> q,
    MultiArticulatedContactProblem& problem,
    const std::span<const MultiArticulatedPointEquality>
        equalities,
    const ConstraintIREvaluationConfig& config,
    const ArticulatedDynamicsConfig& dynamicsConfig
) {
    MultiArticulatedContactDiagnostics diagnostics =
        diagnosticsFor(model, problem.contactCount);
    std::string reason;
    if (!model.valid(&reason)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::invalidModel
        );
    }
    if (!structurallyValid(problem) ||
        problem.articulatedNv != model.world.nv ||
        problem.generalizedConstraintRowCount != 0u ||
        q.size() != model.world.nq ||
        equalities.size() >
            (
                std::numeric_limits<std::uint32_t>::max() -
                model.constraintProgram.rows.size()
            ) / 3u) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::invalidDimensions
        );
    }
    if (!finite(q)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::nonfiniteInput
        );
    }
    if (equalities.empty()) {
        return projectMultiArticulatedContactThroughGeneralizedEqualities(
            model,
            problem,
            config
        );
    }

    struct OwnedPointBlock {
        std::vector<ConstraintIREndpoint> endpoints;
        std::array<ConstraintIRRow, 3> rows{};
        std::array<float, 3> warm{};
        std::array<double, 3> prescribedVelocity{};
    };
    std::vector<OwnedPointBlock> owned(equalities.size());
    std::vector<ConstraintIRSourceBlock> sources;
    sources.reserve(
        model.constraintProgram.blocks.size() +
        equalities.size()
    );
    for (const ConstraintIRBlock& block :
         model.constraintProgram.blocks) {
        ConstraintIRSourceBlock source;
        source.key = block.key;
        source.type = block.type;
        source.flags = block.flags;
        source.islandIndex = block.islandIndex;
        source.eventSlot = block.eventSlot;
        source.endpoints = std::span{
            model.constraintProgram.endpoints
        }.subspan(
            block.endpointOffset,
            block.endpointCount
        );
        source.rows = std::span{
            model.constraintProgram.rows
        }.subspan(
            block.rowOffset,
            block.dimension
        );
        if (block.coneIndex != kConstraintIRInvalidIndex) {
            source.cone =
                model.constraintProgram.cones[
                    block.coneIndex
                ];
        }
        source.warmImpulses = std::span{
            model.constraintProgram.warmImpulses
        }.subspan(
            block.impulseOffset,
            block.dimension
        );
        sources.push_back(source);
    }

    const auto appendEndpointJacobian = [&](
        const MultiContactEndpoint& endpoint,
        const std::array<double, 3>& axis,
        const double sign,
        const std::uint32_t localRow,
        std::vector<ConstraintIREndpoint>& endpoints,
        double& prescribedVelocity
    ) -> bool {
        if (endpoint.kind ==
            MultiContactEndpointKind::staticWorld) {
            return true;
        }
        if (endpoint.kind ==
            MultiContactEndpointKind::sceneBody) {
            if (endpoint.body >=
                    problem.sceneBodyVelocityOffsets.size() ||
                endpoint.body >= problem.sceneBodyStates.size() ||
                !finite(endpoint.localPoint) ||
                !finite(axis)) {
                return false;
            }
            const std::uint32_t velocityOffset =
                problem.sceneBodyVelocityOffsets[endpoint.body];
            const Vec3 pointOffset = rotateVector(
                problem.sceneBodyStates[endpoint.body]
                    .orientation,
                endpoint.localPoint
            );
            if (velocityOffset == MR_INVALID_INDEX) {
                const MRBodyStateGPU& state =
                    problem.sceneBodyStates[endpoint.body];
                const Vec3 linear{
                    state.linearVelocityAndInverseMass.x,
                    state.linearVelocityAndInverseMass.y,
                    state.linearVelocityAndInverseMass.z,
                };
                const Vec3 angularVelocity{
                    state.angularVelocity.x,
                    state.angularVelocity.y,
                    state.angularVelocity.z,
                };
                const Vec3 rotationalVelocity =
                    cross(angularVelocity, pointOffset);
                const Vec3 velocity{
                    linear.x + rotationalVelocity.x,
                    linear.y + rotationalVelocity.y,
                    linear.z + rotationalVelocity.z,
                };
                prescribedVelocity +=
                    sign * dot(vector(axis), velocity);
                return finite(prescribedVelocity);
            }
            const Vec3 angular = cross(
                pointOffset,
                vector(axis)
            );
            const std::array<double, 6> coefficients{
                axis[0],
                axis[1],
                axis[2],
                angular.x,
                angular.y,
                angular.z,
            };
            if (model.articulations.size() >
                std::numeric_limits<std::uint32_t>::max() -
                    endpoint.body) {
                return false;
            }
            const std::uint32_t owner =
                static_cast<std::uint32_t>(
                    model.articulations.size() +
                    endpoint.body
                );
            for (std::uint32_t localDof = 0u;
                 localDof < coefficients.size();
                 ++localDof) {
                const double coefficient =
                    sign * coefficients[localDof];
                if (!finite(coefficient) ||
                    std::abs(coefficient) >
                        std::numeric_limits<float>::max()) {
                    return false;
                }
                if (coefficient == 0.0) {
                    continue;
                }
                endpoints.push_back(
                    makeConstraintIRGeneralizedEndpoint(
                        owner,
                        kConstraintIRInvalidIndex,
                        velocityOffset + localDof,
                        localRow,
                        static_cast<float>(coefficient)
                    )
                );
            }
            return true;
        }
        if (endpoint.kind !=
                MultiContactEndpointKind::articulatedBody ||
            endpoint.body >= model.bodies.size() ||
            !finite(endpoint.localPoint) ||
            !finite(axis)) {
            return false;
        }
        const std::uint32_t articulationIndex =
            model.bodies[endpoint.body].articulationIndex;
        if (articulationIndex == MR_INVALID_INDEX ||
            articulationIndex >= model.articulations.size()) {
            return false;
        }
        const MRArticulationGPU& articulation =
            model.articulations[articulationIndex];
        const std::array query{
            ArticulatedPointQuery{
                endpoint.body,
                endpoint.localPoint,
            },
        };
        std::array<ArticulatedPointKinematics, 1>
            kinematics{};
        std::vector<double> jacobian(
            3u * articulation.nv,
            0.0
        );
        const ArticulatedDynamicsDiagnostics point =
            computeArticulatedPointJacobians(
                model,
                articulationIndex,
                q.subspan(
                    articulation.qOffset,
                    articulation.nq
                ),
                std::span<const double>(
                    problem.freeVelocity
                ).subspan(
                    articulation.vOffset,
                    articulation.nv
                ),
                query,
                kinematics,
                jacobian,
                dynamicsConfig
            );
        if (!point.succeeded()) {
            diagnostics.firstFailingArticulation =
                articulationIndex;
            return false;
        }
        for (std::uint32_t localDof = 0u;
             localDof < articulation.nv;
             ++localDof) {
            const double coefficient = sign * (
                axis[0] * jacobian[localDof] +
                axis[1] *
                    jacobian[
                        articulation.nv + localDof
                    ] +
                axis[2] *
                    jacobian[
                        2u * articulation.nv + localDof
                    ]
            );
            if (!finite(coefficient) ||
                std::abs(coefficient) >
                    std::numeric_limits<float>::max()) {
                return false;
            }
            if (coefficient == 0.0) {
                continue;
            }
            endpoints.push_back(
                makeConstraintIRGeneralizedEndpoint(
                    articulationIndex,
                    kConstraintIRInvalidIndex,
                    articulation.vOffset + localDof,
                    localRow,
                    static_cast<float>(coefficient)
                )
            );
        }
        return true;
    };

    for (std::size_t equalityIndex = 0u;
         equalityIndex < equalities.size();
         ++equalityIndex) {
        const MultiArticulatedPointEquality& equality =
            equalities[equalityIndex];
        const std::array axes{
            equality.axisX,
            equality.axisY,
            equality.axisZ,
        };
        const ContactFrame frame{
            vector(equality.axisX),
            vector(equality.axisY),
            vector(equality.axisZ),
        };
        const auto responds = [&](
            const MultiContactEndpoint& endpoint
        ) {
            return endpoint.kind ==
                    MultiContactEndpointKind::articulatedBody ||
                (
                    endpoint.kind ==
                        MultiContactEndpointKind::sceneBody &&
                    endpoint.body <
                        problem.sceneBodyVelocityOffsets.size() &&
                    problem.sceneBodyVelocityOffsets[
                        endpoint.body
                    ] != MR_INVALID_INDEX
                );
        };
        const bool frameValid =
            finite(equality.endpointA.localPoint) &&
            finite(equality.endpointB.localPoint) &&
            finite(equality.positionError) &&
            finite(equality.targetVelocity) &&
            finite(equality.compliance) &&
            finite(equality.dissipation) &&
            finite(equality.warmImpulse) &&
            finite(equality.timeConstant) &&
            finite(equality.dampingRatio) &&
            equality.timeConstant >= 0.0 &&
            equality.dampingRatio >= 0.0 &&
            std::ranges::all_of(
                equality.compliance,
                [](const double value) {
                    return value >= 0.0;
                }
            ) &&
            std::ranges::all_of(
                equality.dissipation,
                [](const double value) {
                    return value >= 0.0;
                }
            ) &&
            std::abs(norm(frame.normal) - 1.0) <=
                2.0e-4 &&
            std::abs(norm(frame.tangentU) - 1.0) <=
                2.0e-4 &&
            std::abs(norm(frame.tangentV) - 1.0) <=
                2.0e-4 &&
            std::abs(dot(frame.normal, frame.tangentU)) <=
                4.0e-4 &&
            std::abs(dot(frame.normal, frame.tangentV)) <=
                4.0e-4 &&
            std::abs(dot(frame.tangentU, frame.tangentV)) <=
                4.0e-4 &&
            std::abs(
                dot(
                    cross(frame.normal, frame.tangentU),
                    frame.tangentV
                ) - 1.0
            ) <= 6.0e-4;
        if (!frameValid ||
            (
                !responds(equality.endpointA) &&
                !responds(equality.endpointB)
            ) ||
            (
                equality.endpointA.kind ==
                    MultiContactEndpointKind::staticWorld &&
                equality.endpointB.kind ==
                    MultiContactEndpointKind::staticWorld
            ) ||
            (
                equality.endpointA.kind ==
                    equality.endpointB.kind &&
                equality.endpointA.kind !=
                    MultiContactEndpointKind::staticWorld &&
                equality.endpointA.body ==
                    equality.endpointB.body
            )) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedContactStatus::invalidContact,
                static_cast<std::uint32_t>(equalityIndex)
            );
        }
        OwnedPointBlock& block = owned[equalityIndex];
        block.endpoints.reserve(
            6u * model.world.nv
        );
        for (std::uint32_t row = 0u; row < 3u; ++row) {
            if (!appendEndpointJacobian(
                    equality.endpointA,
                    axes[row],
                    -1.0,
                    row,
                    block.endpoints,
                    block.prescribedVelocity[row]
                ) ||
                !appendEndpointJacobian(
                    equality.endpointB,
                    axes[row],
                    1.0,
                    row,
                    block.endpoints,
                    block.prescribedVelocity[row]
                )) {
                const std::uint32_t failingArticulation =
                    diagnostics.firstFailingArticulation;
                return fail(
                    std::move(diagnostics),
                    MultiArticulatedContactStatus::
                        kinematicsFailure,
                    static_cast<std::uint32_t>(
                        equalityIndex
                    ),
                    failingArticulation
                );
            }
            ConstraintIRRow& semantics = block.rows[row];
            semantics.direction = {
                static_cast<float>(axes[row][0]),
                static_cast<float>(axes[row][1]),
                static_cast<float>(axes[row][2]),
                0.0f,
            };
            semantics.positionError =
                static_cast<float>(
                    equality.positionError[row]
                );
            semantics.targetVelocity =
                static_cast<float>(
                    equality.targetVelocity[row] -
                    block.prescribedVelocity[row]
                );
            semantics.compliance =
                static_cast<float>(
                    equality.compliance[row]
                );
            semantics.dissipation =
                static_cast<float>(
                    equality.dissipation[row]
                );
            semantics.timeConstant =
                static_cast<float>(equality.timeConstant);
            semantics.dampingRatio =
                static_cast<float>(equality.dampingRatio);
            semantics.impulseLower =
                -kConstraintIRUnbounded;
            semantics.impulseUpper =
                kConstraintIRUnbounded;
            semantics.flags =
                equality.positionStabilized
                ? constraintIRRowPositionStabilized
                : 0u;
            block.warm[row] =
                static_cast<float>(equality.warmImpulse[row]);
        }
        sources.push_back({
            .key = equality.key,
            .type = MR_CONSTRAINT_BILATERAL,
            .flags = 0u,
            .islandIndex = 0u,
            .eventSlot = kConstraintIRInvalidIndex,
            .endpoints = block.endpoints,
            .rows = block.rows,
            .cone = std::nullopt,
            .warmImpulses = block.warm,
        });
    }

    ConstraintIRCompilationResult compiled =
        compileConstraintIR(sources);
    if (!compiled.succeeded()) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::
                unsupportedConstraint
        );
    }
    EngineModel stagedModel = model;
    stagedModel.constraintProgram = std::move(compiled.ir);
    return projectMultiArticulatedContactThroughGeneralizedEqualities(
        stagedModel,
        problem,
        config
    );
}

MultiArticulatedContactDiagnostics
solveMultiArticulatedContactProblem(
    const MultiArticulatedContactProblem& problem,
    MultiArticulatedContactSolution& output,
    const QualityContactSolverConfig& config
) {
    EngineModel diagnosticModel;
    diagnosticModel.world.nv = problem.nv;
    diagnosticModel.articulations.resize(
        problem.factors.factors.size()
    );
    MultiArticulatedContactDiagnostics diagnostics =
        diagnosticsFor(
            diagnosticModel,
            problem.contactCount
        );
    if (!structurallyValid(problem)) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::invalidDimensions
        );
    }
    if (!finite(
            std::span<const double>(problem.freeVelocity)
        ) ||
        !finite(
            std::span<const double>(
                problem.contactJacobian
            )
        ) ||
        !finite(
            std::span<const double>(
                problem.responseColumns
            )
        )) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::nonfiniteInput
        );
    }

    MultiArticulatedContactSolution staged;
    staged.quality =
        solveQualityContactSpaceProblem(
            problem.conic,
            config
        );
    if (!staged.quality.converged()) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::solverFailure
        );
    }
    const std::size_t rowCount =
        3u * static_cast<std::size_t>(
            problem.contactCount
        );
    if (staged.quality.impulses.size() != rowCount ||
        staged.quality.velocity.size() != rowCount) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::solverFailure
        );
    }
    staged.generalizedVelocity = problem.freeVelocity;
    staged.impulses = staged.quality.impulses;
    for (std::size_t row = 0u;
         row < rowCount;
         ++row) {
        const double impulse = staged.impulses[row];
        for (std::size_t dof = 0u;
             dof < problem.nv;
             ++dof) {
            staged.generalizedVelocity[dof] +=
                impulse *
                problem.responseColumns[
                    row * problem.nv + dof
                ];
        }
    }
    if (!finite(
            std::span<const double>(
                staged.generalizedVelocity
            )
        )) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::nonfiniteResult
        );
    }
    const std::size_t equalityRows =
        problem.generalizedConstraintRowCount;
    if (equalityRows != 0u) {
        staged.generalizedConstraintImpulses =
            problem.generalizedConstraintFreeImpulses;
        for (std::size_t row = 0u;
             row < equalityRows;
             ++row) {
            for (std::size_t contactRow = 0u;
                 contactRow < rowCount;
                 ++contactRow) {
                staged.generalizedConstraintImpulses[row] -=
                    problem
                        .generalizedConstraintContactCoupling[
                            row * rowCount + contactRow
                        ] * staged.impulses[contactRow];
            }
            double residual =
                -problem.generalizedConstraintTargets[row] +
                problem.generalizedConstraintRegularization[
                    row
                ] *
                    staged
                        .generalizedConstraintImpulses[row];
            for (std::size_t dof = 0u;
                 dof < problem.nv;
                 ++dof) {
                residual +=
                    problem.generalizedConstraintJacobian[
                        row * problem.nv + dof
                    ] *
                    staged.generalizedVelocity[dof];
            }
            diagnostics.maximumGeneralizedConstraintResidual =
                std::max(
                    diagnostics
                        .maximumGeneralizedConstraintResidual,
                    std::abs(residual)
                );
        }
        if (!finite(
                std::span<const double>(
                    staged.generalizedConstraintImpulses
                )
            ) ||
            !finite(
                diagnostics
                    .maximumGeneralizedConstraintResidual
            )) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedContactStatus::
                    nonfiniteResult
            );
        }
    }
    if (problem.articulatedNv > problem.nv ||
        problem.sceneBodyVelocityOffsets.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::invalidDimensions
        );
    }
    staged.articulatedVelocity.assign(
        staged.generalizedVelocity.begin(),
        staged.generalizedVelocity.begin() +
            problem.articulatedNv
    );
    staged.sceneBodyVelocities =
        problem.sceneBodyFreeVelocities;
    for (std::size_t body = 0u;
         body < problem.sceneBodyVelocityOffsets.size();
         ++body) {
        const std::uint32_t offset =
            problem.sceneBodyVelocityOffsets[body];
        if (offset == MR_INVALID_INDEX) {
            continue;
        }
        if (offset < problem.articulatedNv ||
            static_cast<std::uint64_t>(offset) + 6u >
                problem.nv) {
            return fail(
                std::move(diagnostics),
                MultiArticulatedContactStatus::
                    invalidDimensions
            );
        }
        MultiContactSceneBodyVelocity& velocity =
            staged.sceneBodyVelocities[body];
        std::ranges::copy_n(
            staged.generalizedVelocity.begin() + offset,
            3u,
            velocity.linear.begin()
        );
        std::ranges::copy_n(
            staged.generalizedVelocity.begin() + offset + 3u,
            3u,
            velocity.angular.begin()
        );
    }
    for (std::size_t row = 0u;
         row < rowCount;
         ++row) {
        double velocity =
            problem.prescribedContactVelocity[row];
        for (std::size_t dof = 0u;
             dof < problem.nv;
             ++dof) {
            velocity +=
                problem.contactJacobian[
                    row * problem.nv + dof
                ] *
                staged.generalizedVelocity[dof];
        }
        diagnostics.maximumContactVelocityResidual =
            std::max(
                diagnostics.maximumContactVelocityResidual,
                std::abs(
                    velocity -
                    staged.quality.velocity[row]
                )
            );
    }
    double velocityScale = 1.0;
    for (const double value : staged.quality.velocity) {
        velocityScale = std::max(
            velocityScale,
            std::abs(value)
        );
    }
    const double velocityTolerance =
        8192.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(
            std::max<std::size_t>(
                1u,
                std::max(
                    rowCount,
                    static_cast<std::size_t>(
                        problem.nv
                    )
                )
            )
        ) *
        velocityScale;
    if (!finite(
            diagnostics.maximumContactVelocityResidual
        ) ||
        diagnostics.maximumContactVelocityResidual >
            velocityTolerance) {
        return fail(
            std::move(diagnostics),
            MultiArticulatedContactStatus::nonfiniteResult
        );
    }
    output = std::move(staged);
    return diagnostics;
}

const char* multiArticulatedContactStatusName(
    const MultiArticulatedContactStatus status
) noexcept {
    switch (status) {
    case MultiArticulatedContactStatus::success:
        return "success";
    case MultiArticulatedContactStatus::invalidModel:
        return "invalid_model";
    case MultiArticulatedContactStatus::invalidDimensions:
        return "invalid_dimensions";
    case MultiArticulatedContactStatus::invalidContact:
        return "invalid_contact";
    case MultiArticulatedContactStatus::nonfiniteInput:
        return "nonfinite_input";
    case MultiArticulatedContactStatus::kinematicsFailure:
        return "kinematics_failure";
    case MultiArticulatedContactStatus::factorizationFailure:
        return "factorization_failure";
    case MultiArticulatedContactStatus::unsupportedConstraint:
        return "unsupported_constraint";
    case MultiArticulatedContactStatus::
        constraintEvaluationFailure:
        return "constraint_evaluation_failure";
    case MultiArticulatedContactStatus::
        constraintFactorizationFailure:
        return "constraint_factorization_failure";
    case MultiArticulatedContactStatus::solverFailure:
        return "solver_failure";
    case MultiArticulatedContactStatus::nonfiniteResult:
        return "nonfinite_result";
    }
    return "unknown";
}

} // namespace metalrobo
