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

bool makeFrame(
    const ArticulatedContact& contact,
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
    return
        problem.freeVelocity.size() == problem.nv &&
        problem.contactJacobian.size() ==
            rowCount * problem.nv &&
        problem.responseColumns.size() ==
            rowCount * problem.nv &&
        problem.conic.delassus.size() ==
            rowCount * rowCount &&
        problem.conic.freeContactVelocity.size() ==
            rowCount &&
        problem.conic.contacts.size() ==
            problem.contactCount &&
        problem.pointA.size() == problem.contactCount &&
        problem.pointB.size() == problem.contactCount;
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
    staged.contactCount =
        static_cast<std::uint32_t>(contactCount);
    staged.factors = std::move(factors);
    staged.freeVelocity.assign(
        freeVelocity.begin(),
        freeVelocity.end()
    );
    staged.contactJacobian.assign(rowCount * nv, 0.0);
    staged.responseColumns.assign(rowCount * nv, 0.0);
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
    for (std::size_t row = 0u;
         row < rowCount;
         ++row) {
        double velocity = 0.0;
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
    case MultiArticulatedContactStatus::solverFailure:
        return "solver_failure";
    case MultiArticulatedContactStatus::nonfiniteResult:
        return "nonfinite_result";
    }
    return "unknown";
}

} // namespace metalrobo
