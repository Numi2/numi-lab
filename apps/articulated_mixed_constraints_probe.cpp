#include "metalrobo/ArticulatedMixedConstraints.hpp"
#include "metalrobo/G1.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<double> asDouble(const std::vector<float>& values) {
    return {values.begin(), values.end()};
}

double maximumError(
    const std::span<const double> left,
    const std::span<const double> right
) {
    if (left.size() != right.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double result = 0.0;
    for (std::size_t index = 0u; index < left.size(); ++index) {
        result = std::max(
            result,
            std::abs(left[index] - right[index])
        );
    }
    return result;
}

bool emptyPayload(
    const metalrobo::ArticulatedMixedConstraintSolution& solution
) {
    return
        solution.contactImpulses.empty() &&
        solution.limitImpulses.empty() &&
        solution.contactVelocity.empty() &&
        solution.limitVelocity.empty() &&
        solution.generalizedVelocity.empty();
}

bool bitwiseEqual(
    const std::vector<double>& left,
    const std::vector<double>& right
) {
    return
        left.size() == right.size() &&
        (left.empty() ||
         std::memcmp(
             left.data(),
             right.data(),
             left.size() * sizeof(double)
         ) == 0);
}

metalrobo::ArticulatedContactProblem factorOnly(
    const metalrobo::ArticulatedContactProblem& source
) {
    metalrobo::ArticulatedContactProblem result = source;
    result.contactCount = 0u;
    result.conic.contacts.clear();
    result.contactJacobian.clear();
    result.delassus.clear();
    result.pointA.clear();
    result.pointB.clear();
    return result;
}

} // namespace

int main() {
    try {
        const metalrobo::EngineModel model =
            metalrobo::makeUnitreeG1EngineModel();
        std::string modelFailure;
        require(
            model.valid(&modelFailure),
            "canonical G1 is invalid: " + modelFailure
        );
        const MRArticulationGPU& articulation =
            model.articulations.at(0u);
        require(
            articulation.nq == 36u && articulation.nv == 35u,
            "canonical G1 dimensions changed"
        );

        metalrobo::ArticulatedDynamicsConfig dynamics;
        dynamics.gravity = {0.0, 0.0, 0.0};
        dynamics.applyBodyDamping = false;
        std::vector<double> q = asDouble(model.defaultQ);
        std::vector<double> freeVelocity(
            articulation.nv,
            0.0
        );
        freeVelocity[0] = 0.35;
        freeVelocity[2] = -0.8;

        // The first actuated DoF is an ancestor of the left foot, which makes
        // its lower stop physically cross-coupled to the foot contact.
        const MRDofPropertiesGPU& stoppedDof = model.dofs.at(6u);
        const std::uint32_t stoppedLocalQ =
            stoppedDof.qIndex - articulation.qOffset;
        const std::uint32_t stoppedLocalV =
            stoppedDof.vIndex - articulation.vOffset;
        q[stoppedLocalQ] =
            static_cast<double>(stoppedDof.limits.x) - 7.5e-4;
        freeVelocity[stoppedLocalV] = -1.1;

        metalrobo::ArticulatedJointLimitConfig limitConfig;
        limitConfig.timestep = 1.0e-3;
        limitConfig.activationDistance = 2.0e-3;
        limitConfig.positionSlop = 1.0e-8;
        limitConfig.recoveryFraction = 0.2;
        limitConfig.maximumRecoverySpeed = 2.0;
        limitConfig.regularization = 1.0e-8;
        limitConfig.quality.kktTolerance = 1.0e-12;
        std::vector<metalrobo::ArticulatedJointLimitRow>
            limitRows;
        const auto limitBuild =
            metalrobo::compileArticulatedJointLimitRows(
                model,
                0u,
                q,
                freeVelocity,
                limitRows,
                limitConfig
            );
        require(
            limitBuild.succeeded() &&
            limitRows.size() == 1u &&
            limitRows[0].side ==
                metalrobo::ArticulatedJointLimitSide::lower,
            "actual G1 state did not compile exactly one lower stop"
        );

        const auto& foot =
            metalrobo::unitreeG1Metadata().feet.at(0u);
        const metalrobo::ArticulatedPointQuery footQuery{
            foot.bodyIndex,
            {
                foot.solePosition.x,
                foot.solePosition.y,
                foot.solePosition.z,
            },
        };
        metalrobo::ArticulatedPointKinematics footPoint;
        std::vector<double> footPointJacobian(
            3u * articulation.nv,
            0.0
        );
        const auto pointDiagnostics =
            metalrobo::computeArticulatedPointJacobians(
                model,
                0u,
                q,
                freeVelocity,
                std::span(&footQuery, 1u),
                std::span(&footPoint, 1u),
                footPointJacobian,
                dynamics
            );
        require(
            pointDiagnostics.succeeded(),
            "G1 left-foot point kinematics failed"
        );

        metalrobo::ArticulatedContact footContact;
        footContact.bodyA = foot.bodyIndex;
        footContact.localPointA = footQuery.localPoint;
        footContact.localPointB = footPoint.position;
        footContact.normal = {0.0, 0.0, -1.0};
        footContact.tangentU = {1.0, 0.0, 0.0};
        footContact.tangentV = {0.0, -1.0, 0.0};
        footContact.regularization = {
            1.0e-8,
            1.0e-8,
            1.0e-8,
        };
        footContact.friction = 0.8;
        metalrobo::ArticulatedContactProblem contactProblem;
        const auto contactBuild =
            metalrobo::buildArticulatedContactProblem(
                model,
                0u,
                q,
                freeVelocity,
                std::span(&footContact, 1u),
                contactProblem,
                dynamics,
                false
            );
        require(
            contactBuild.succeeded(),
            "actual G1 foot contact construction failed"
        );

        metalrobo::QualityContactSolverConfig quality;
        quality.maximumIterations = 400u;
        quality.kktTolerance = 1.0e-11;

        const auto contactOnly =
            metalrobo::solveArticulatedMixedConstraints(
                contactProblem,
                freeVelocity,
                {},
                quality
            );
        require(
            contactOnly.converged() &&
            contactOnly.contactImpulses.size() == 3u &&
            contactOnly.limitImpulses.empty() &&
            contactOnly.contactImpulses[0] > 0.0,
            "contact-only mixed path failed"
        );

        const auto limitOnly =
            metalrobo::solveArticulatedMixedConstraints(
                factorOnly(contactProblem),
                freeVelocity,
                limitRows,
                quality
            );
        require(
            limitOnly.converged() &&
            limitOnly.contactImpulses.empty() &&
            limitOnly.limitImpulses.size() == 1u &&
            limitOnly.limitImpulses[0] > 0.0,
            "limit-only mixed path failed"
        );

        const auto mixed =
            metalrobo::solveArticulatedMixedConstraints(
                contactProblem,
                freeVelocity,
                limitRows,
                quality
            );
        require(
            mixed.converged(),
            std::string("monolithic G1 solve failed: ") +
                metalrobo::articulatedMixedConstraintStatusName(
                    mixed.diagnostics.status
                )
        );
        require(
            mixed.contactImpulses.size() == 3u &&
            mixed.limitImpulses.size() == 1u &&
            mixed.contactImpulses[0] > 0.0 &&
            mixed.limitImpulses[0] > 0.0 &&
            mixed.diagnostics.maximumCrossDelassusMagnitude >
                1.0e-6 &&
            mixed.diagnostics.finalFactorApplications == 1u,
            "monolithic solve did not retain active cross coupling"
        );
        require(
            mixed.diagnostics.scaledKktCertificate < 2.0e-10 &&
            mixed.diagnostics.maximumFactorSolveResidual <
                2.0e-12 &&
            mixed.diagnostics
                    .maximumContactVelocityConsistencyError <
                2.0e-11 &&
            mixed.diagnostics
                    .maximumLimitVelocityConsistencyError <
                2.0e-11,
            "monolithic KKT or factor/velocity certificate failed"
        );
        const double limitDual =
            mixed.limitVelocity[0] -
            limitRows[0].targetVelocity +
            limitRows[0].regularization *
                mixed.limitImpulses[0];
        const double limitComplementarity =
            std::abs(mixed.limitImpulses[0] * limitDual);
        require(
            limitDual >= -2.0e-10 &&
            limitComplementarity < 2.0e-9,
            "monolithic scalar limit KKT failed"
        );
        std::array<double, 3> directContactVelocity{};
        const auto contactAction =
            metalrobo::applyArticulatedContactJacobian(
                contactProblem,
                mixed.generalizedVelocity,
                directContactVelocity
            );
        require(
            contactAction.succeeded() &&
            maximumError(
                directContactVelocity,
                mixed.contactVelocity
            ) < 2.0e-14 &&
            std::abs(
                mixed.limitVelocity[0] -
                limitRows[0].direction *
                    mixed.generalizedVelocity[
                        limitRows[0].localVIndex
                    ]
            ) < 2.0e-14,
            "returned physical constraint velocities are inconsistent"
        );

        // A contact-then-limit split uses a different free velocity for its
        // second solve and therefore omits simultaneous cross-KKT coupling.
        metalrobo::ArticulatedJointLimitProblem sequentialLimitProblem;
        const auto sequentialLimitBuild =
            metalrobo::buildArticulatedJointLimitProblem(
                model,
                0u,
                q,
                contactOnly.generalizedVelocity,
                sequentialLimitProblem,
                limitConfig,
                dynamics
            );
        require(
            sequentialLimitBuild.succeeded() &&
            sequentialLimitProblem.rows.size() == 1u,
            "sequential limit stage did not retain active stop"
        );
        const auto sequentialLimit =
            metalrobo::solveArticulatedJointLimits(
                sequentialLimitProblem,
                limitConfig
            );
        require(
            sequentialLimit.converged(),
            "sequential contact-then-limit solve failed"
        );
        const double sequentialDifference = maximumError(
            sequentialLimit.generalizedVelocity,
            mixed.generalizedVelocity
        );
        require(
            sequentialDifference > 1.0e-5,
            "sequential split unexpectedly matched monolithic coupling"
        );

        // Two identical G1 foot rows make the physical W singular. Positive
        // row regularization still gives a unique monolithic solution, and
        // exercises the quality solver's rank-deficient PSD gate.
        const std::array<metalrobo::ArticulatedContact, 2>
            duplicateContacts{footContact, footContact};
        metalrobo::ArticulatedContactProblem rankDeficientProblem;
        const auto rankDeficientBuild =
            metalrobo::buildArticulatedContactProblem(
                model,
                0u,
                q,
                freeVelocity,
                duplicateContacts,
                rankDeficientProblem,
                dynamics,
                false
            );
        require(
            rankDeficientBuild.succeeded(),
            "rank-deficient G1 contact operator construction failed"
        );
        double duplicateRowError = 0.0;
        for (std::size_t row = 0u; row < 3u; ++row) {
            for (std::size_t dof = 0u;
                 dof < articulation.nv;
                 ++dof) {
                duplicateRowError = std::max(
                    duplicateRowError,
                    std::abs(
                        rankDeficientProblem.contactJacobian[
                            row * articulation.nv + dof
                        ] -
                        rankDeficientProblem.contactJacobian[
                            (row + 3u) * articulation.nv + dof
                        ]
                    )
                );
            }
        }
        const auto rankDeficient =
            metalrobo::solveArticulatedMixedConstraints(
                rankDeficientProblem,
                freeVelocity,
                limitRows,
                quality
            );
        require(
            duplicateRowError == 0.0 &&
            rankDeficient.converged() &&
            rankDeficient.diagnostics.scaledKktCertificate <
                2.0e-10 &&
            rankDeficient.contactImpulses.size() == 6u &&
            rankDeficient.limitImpulses.size() == 1u,
            "rank-deficient monolithic solve failed"
        );

        // Exercise the full off-diagonal limit-limit block on actual G1 and
        // compare it to the dedicated factor-backed limit solver.
        std::vector<double> twoLimitQ = q;
        std::vector<double> twoLimitFreeVelocity = freeVelocity;
        const MRDofPropertiesGPU& secondStoppedDof =
            model.dofs.at(7u);
        const std::uint32_t secondLocalQ =
            secondStoppedDof.qIndex - articulation.qOffset;
        const std::uint32_t secondLocalV =
            secondStoppedDof.vIndex - articulation.vOffset;
        twoLimitQ[secondLocalQ] =
            static_cast<double>(secondStoppedDof.limits.y) +
            6.0e-4;
        twoLimitFreeVelocity[secondLocalV] = 0.9;
        std::vector<metalrobo::ArticulatedJointLimitRow>
            twoLimitRows;
        const auto twoLimitCompile =
            metalrobo::compileArticulatedJointLimitRows(
                model,
                0u,
                twoLimitQ,
                twoLimitFreeVelocity,
                twoLimitRows,
                limitConfig
            );
        require(
            twoLimitCompile.succeeded() &&
            twoLimitRows.size() == 2u,
            "two active G1 stops did not compile deterministically"
        );
        metalrobo::ArticulatedContactProblem twoLimitContactProblem;
        const auto twoLimitContactBuild =
            metalrobo::buildArticulatedContactProblem(
                model,
                0u,
                twoLimitQ,
                twoLimitFreeVelocity,
                std::span(&footContact, 1u),
                twoLimitContactProblem,
                dynamics,
                false
            );
        require(
            twoLimitContactBuild.succeeded(),
            "two-limit G1 retained contact factor failed"
        );
        const auto twoLimitMixed =
            metalrobo::solveArticulatedMixedConstraints(
                twoLimitContactProblem,
                twoLimitFreeVelocity,
                twoLimitRows,
                quality
            );
        require(
            twoLimitMixed.converged() &&
            twoLimitMixed.limitImpulses.size() == 2u &&
            twoLimitMixed.limitImpulses[0] > 0.0 &&
            twoLimitMixed.limitImpulses[1] > 0.0 &&
            twoLimitMixed.diagnostics.scaledKktCertificate <
                2.0e-10,
            "two-limit monolithic G1 solve failed"
        );
        const auto twoLimitOnly =
            metalrobo::solveArticulatedMixedConstraints(
                factorOnly(twoLimitContactProblem),
                twoLimitFreeVelocity,
                twoLimitRows,
                quality
            );
        metalrobo::ArticulatedJointLimitConfig parityLimitConfig =
            limitConfig;
        parityLimitConfig.quality = quality;
        metalrobo::ArticulatedJointLimitProblem
            dedicatedTwoLimitProblem;
        const auto dedicatedTwoLimitBuild =
            metalrobo::buildArticulatedJointLimitProblem(
                model,
                0u,
                twoLimitQ,
                twoLimitFreeVelocity,
                dedicatedTwoLimitProblem,
                parityLimitConfig,
                dynamics
            );
        const auto dedicatedTwoLimit =
            metalrobo::solveArticulatedJointLimits(
                dedicatedTwoLimitProblem,
                parityLimitConfig
            );
        require(
            dedicatedTwoLimitBuild.succeeded() &&
            dedicatedTwoLimit.converged() &&
            dedicatedTwoLimitProblem.rows.size() == 2u &&
            std::abs(dedicatedTwoLimitProblem.delassus[1]) >
                1.0e-8 &&
            twoLimitOnly.converged(),
            "dedicated two-limit G1 parity setup failed"
        );
        const double twoLimitImpulseParity = maximumError(
            twoLimitOnly.limitImpulses,
            dedicatedTwoLimit.impulses
        );
        const double twoLimitVelocityParity = maximumError(
            twoLimitOnly.generalizedVelocity,
            dedicatedTwoLimit.generalizedVelocity
        );
        require(
            twoLimitImpulseParity < 2.0e-10 &&
            twoLimitVelocityParity < 2.0e-11 &&
            twoLimitOnly.diagnostics
                    .maximumLimitVelocityConsistencyError <
                2.0e-11,
            "mixed off-diagonal limit-limit action lost parity"
        );

        const auto replay =
            metalrobo::solveArticulatedMixedConstraints(
                contactProblem,
                freeVelocity,
                limitRows,
                quality
            );
        require(
            replay.converged() &&
            bitwiseEqual(
                replay.contactImpulses,
                mixed.contactImpulses
            ) &&
            bitwiseEqual(
                replay.limitImpulses,
                mixed.limitImpulses
            ) &&
            bitwiseEqual(
                replay.generalizedVelocity,
                mixed.generalizedVelocity
            ),
            "monolithic replay is not bitwise deterministic"
        );

        std::vector<double> shortVelocity = freeVelocity;
        shortVelocity.pop_back();
        const auto invalidDimensions =
            metalrobo::solveArticulatedMixedConstraints(
                contactProblem,
                shortVelocity,
                limitRows,
                quality
            );
        require(
            invalidDimensions.diagnostics.status ==
                metalrobo::ArticulatedMixedConstraintStatus::
                    invalidDimensions &&
            emptyPayload(invalidDimensions),
            "invalid dimensions published a partial solution"
        );
        auto invalidRows = limitRows;
        invalidRows[0].freeNormalVelocity += 1.0;
        const auto invalidLimit =
            metalrobo::solveArticulatedMixedConstraints(
                contactProblem,
                freeVelocity,
                invalidRows,
                quality
            );
        require(
            invalidLimit.diagnostics.status ==
                metalrobo::ArticulatedMixedConstraintStatus::
                    invalidLimitRow &&
            emptyPayload(invalidLimit),
            "invalid limit row published a partial solution"
        );
        std::cout
            << std::setprecision(10)
            << "articulated_mixed_constraints=cpu_fp64"
            << " contacts=" << contactProblem.contactCount
            << " limits=" << limitRows.size()
            << " cross="
            << mixed.diagnostics.maximumCrossDelassusMagnitude
            << " contact_impulse=" << mixed.contactImpulses[0]
            << " limit_impulse=" << mixed.limitImpulses[0]
            << " kkt="
            << mixed.diagnostics.scaledKktCertificate
            << " factor_residual="
            << mixed.diagnostics.maximumFactorSolveResidual
            << " velocity_consistency="
            << std::max(
                mixed.diagnostics
                    .maximumContactVelocityConsistencyError,
                mixed.diagnostics
                    .maximumLimitVelocityConsistencyError
            )
            << " sequential_difference="
            << sequentialDifference
            << " two_limit_impulse_parity="
            << twoLimitImpulseParity
            << " two_limit_velocity_parity="
            << twoLimitVelocityParity
            << " rank_deficient=pass"
            << " contact_only=pass"
            << " limit_only=pass"
            << " factor_applications="
            << mixed.diagnostics.finalFactorApplications
            << " replay=bitwise"
            << " transaction=pass"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "articulated_mixed_constraints_probe failed: "
            << exception.what() << '\n';
        return 1;
    }
}
