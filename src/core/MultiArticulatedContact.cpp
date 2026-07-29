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
            problem.sceneBodyVelocityOffsets.size();
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
    case MultiArticulatedContactStatus::solverFailure:
        return "solver_failure";
    case MultiArticulatedContactStatus::nonfiniteResult:
        return "nonfinite_result";
    }
    return "unknown";
}

} // namespace metalrobo
