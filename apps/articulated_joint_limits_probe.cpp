#include "metalrobo/ArticulatedJointLimits.hpp"
#include "metalrobo/G1.hpp"

#include <algorithm>
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

bool sameProblem(
    const metalrobo::ArticulatedJointLimitProblem& left,
    const metalrobo::ArticulatedJointLimitProblem& right
) {
    return
        left.articulationIndex == right.articulationIndex &&
        left.nv == right.nv &&
        left.rows.size() == right.rows.size() &&
        bitwiseEqual(left.freeVelocity, right.freeVelocity) &&
        bitwiseEqual(
            left.massCholeskyLower,
            right.massCholeskyLower
        ) &&
        bitwiseEqual(left.delassus, right.delassus) &&
        (left.rows.empty() ||
         std::memcmp(
             left.rows.data(),
             right.rows.data(),
             left.rows.size() *
                 sizeof(metalrobo::ArticulatedJointLimitRow)
         ) == 0);
}

metalrobo::ArticulatedJointLimitProblem sentinelProblem() {
    metalrobo::ArticulatedJointLimitProblem result;
    result.articulationIndex = 91u;
    result.nv = 2u;
    result.rows.resize(1u);
    result.rows[0].stableKey = 77u;
    result.freeVelocity = {3.0, -4.0};
    result.massCholeskyLower = {1.0, 0.0, 0.5, 2.0};
    result.delassus = {6.0};
    return result;
}

double maximumOffDiagonal(
    const metalrobo::ArticulatedJointLimitProblem& problem
) {
    double result = 0.0;
    const std::size_t count = problem.rows.size();
    for (std::size_t row = 0u; row < count; ++row) {
        for (std::size_t column = 0u;
             column < count;
             ++column) {
            if (row != column) {
                result = std::max(
                    result,
                    std::abs(
                        problem.delassus[row * count + column]
                    )
                );
            }
        }
    }
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

        metalrobo::ArticulatedJointLimitConfig config;
        config.timestep = 1.0e-3;
        config.activationDistance = 2.0e-3;
        config.positionSlop = 1.0e-8;
        config.recoveryFraction = 0.2;
        config.maximumRecoverySpeed = 2.0;
        config.regularization = 1.0e-10;
        config.quality.kktTolerance = 1.0e-12;

        const std::vector<double> defaultQ =
            asDouble(model.defaultQ);
        const std::vector<double> zeroVelocity(
            articulation.nv,
            0.0
        );
        metalrobo::ArticulatedJointLimitProblem inactiveProblem;
        const auto inactiveDiagnostics =
            metalrobo::buildArticulatedJointLimitProblem(
                model,
                0u,
                defaultQ,
                zeroVelocity,
                inactiveProblem,
                config
            );
        require(
            inactiveDiagnostics.succeeded(),
            std::string("default G1 limit build failed: ") +
                metalrobo::articulatedJointLimitStatusName(
                    inactiveDiagnostics.status
                )
        );
        require(
            inactiveProblem.rows.empty() &&
                inactiveProblem.massCholeskyLower.empty() &&
                inactiveProblem.delassus.empty(),
            "inactive G1 limits performed unnecessary factor work"
        );
        const auto inactiveSolution =
            metalrobo::solveArticulatedJointLimits(
                inactiveProblem,
                config
            );
        require(
            inactiveSolution.converged() &&
            bitwiseEqual(
                inactiveSolution.generalizedVelocity,
                zeroVelocity
            ),
            "empty limit solve did not preserve free velocity"
        );

        std::vector<double> q = defaultQ;
        std::vector<double> freeVelocity(
            articulation.nv,
            0.0
        );
        const MRDofPropertiesGPU& lowerDof = model.dofs.at(6u);
        const MRDofPropertiesGPU& upperDof = model.dofs.at(7u);
        const std::uint32_t lowerLocalQ =
            lowerDof.qIndex - articulation.qOffset;
        const std::uint32_t upperLocalQ =
            upperDof.qIndex - articulation.qOffset;
        const std::uint32_t lowerLocalV =
            lowerDof.vIndex - articulation.vOffset;
        const std::uint32_t upperLocalV =
            upperDof.vIndex - articulation.vOffset;
        q[lowerLocalQ] =
            static_cast<double>(lowerDof.limits.x) - 2.0e-3;
        q[upperLocalQ] =
            static_cast<double>(upperDof.limits.y) + 1.5e-3;
        freeVelocity[lowerLocalV] = -0.7;
        freeVelocity[upperLocalV] = 0.6;
        const std::vector<double> qBefore = q;

        metalrobo::ArticulatedJointLimitProblem problem;
        const auto buildDiagnostics =
            metalrobo::buildArticulatedJointLimitProblem(
                model,
                0u,
                q,
                freeVelocity,
                problem,
                config
            );
        require(
            buildDiagnostics.succeeded(),
            std::string("penetrated G1 limit build failed: ") +
                metalrobo::articulatedJointLimitStatusName(
                    buildDiagnostics.status
                )
        );
        require(
            problem.rows.size() == 2u &&
            problem.rows[0].side ==
                metalrobo::ArticulatedJointLimitSide::lower &&
            problem.rows[1].side ==
                metalrobo::ArticulatedJointLimitSide::upper &&
            problem.rows[0].stableKey < problem.rows[1].stableKey,
            "joint-limit rows are not canonically lower/upper ordered"
        );
        require(
            buildDiagnostics.penetratingRowCount == 2u &&
            buildDiagnostics.lowerRowCount == 1u &&
            buildDiagnostics.upperRowCount == 1u &&
            buildDiagnostics.maximumPenetration > 1.9e-3,
            "joint-limit activation diagnostics are incorrect"
        );
        const double coupling = maximumOffDiagonal(problem);
        require(
            coupling > 1.0e-8,
            "G1 two-limit Delassus operator lost articulation coupling"
        );

        const auto solution =
            metalrobo::solveArticulatedJointLimits(
                problem,
                config
            );
        require(
            solution.converged(),
            std::string("G1 joint-limit solve failed: ") +
                metalrobo::articulatedJointLimitStatusName(
                    solution.diagnostics.status
                )
        );
        require(
            solution.impulses.size() == 2u &&
            solution.impulses[0] > 0.0 &&
            solution.impulses[1] > 0.0,
            "penetrated lower/upper limits did not produce impulses"
        );
        require(
            solution.diagnostics.scaledKktCertificate <= 2.0e-12 &&
            solution.diagnostics.maximumDualViolation <= 2.0e-11 &&
            solution.diagnostics
                    .maximumComplementarityResidual <=
                2.0e-10 &&
            solution.diagnostics.maximumFactorSolveResidual <=
                2.0e-13,
            "joint-limit solve certificate exceeded tolerance"
        );
        for (std::size_t row = 0u;
             row < problem.rows.size();
             ++row) {
            require(
                solution.constraintVelocity[row] >=
                    problem.rows[row].targetVelocity - 2.0e-8,
                "post-limit velocity violates recovery target"
            );
        }
        require(
            bitwiseEqual(q, qBefore) &&
            q[lowerLocalQ] <
                static_cast<double>(lowerDof.limits.x) &&
            q[upperLocalQ] >
                static_cast<double>(upperDof.limits.y),
            "limit solve clamped or otherwise mutated q"
        );

        std::vector<double> appliedVelocity = freeVelocity;
        const auto applyDiagnostics =
            metalrobo::applyArticulatedJointLimitImpulses(
                problem,
                solution.impulses,
                appliedVelocity
            );
        require(
            applyDiagnostics.succeeded() &&
            bitwiseEqual(
                appliedVelocity,
                solution.generalizedVelocity
            ),
            "factor-backed limit impulse application disagrees with solve"
        );
        const auto replay =
            metalrobo::solveArticulatedJointLimits(
                problem,
                config
            );
        require(
            replay.converged() &&
            bitwiseEqual(replay.impulses, solution.impulses) &&
            bitwiseEqual(
                replay.constraintVelocity,
                solution.constraintVelocity
            ) &&
            bitwiseEqual(
                replay.generalizedVelocity,
                solution.generalizedVelocity
            ),
            "joint-limit replay is not bitwise deterministic"
        );

        // Stress the actual floating-base G1 with every bounded joint active
        // at once. This exercises the fully coupled 29x29 Delassus solve,
        // rather than treating joint stops as independent scalar clamps.
        std::vector<double> allLimitQ = defaultQ;
        std::vector<double> allLimitVelocity(
            articulation.nv,
            0.0
        );
        for (std::size_t joint = 0u;
             joint < metalrobo::kUnitreeG1JointCount;
             ++joint) {
            const MRDofPropertiesGPU& dof =
                model.dofs.at(6u + joint);
            const std::uint32_t localQ =
                dof.qIndex - articulation.qOffset;
            const std::uint32_t localV =
                dof.vIndex - articulation.vOffset;
            const double penetration =
                1.0e-4 + 2.0e-6 * static_cast<double>(joint);
            if ((joint & 1u) == 0u) {
                allLimitQ[localQ] =
                    static_cast<double>(dof.limits.x) -
                    penetration;
                allLimitVelocity[localV] = -0.25;
            } else {
                allLimitQ[localQ] =
                    static_cast<double>(dof.limits.y) +
                    penetration;
                allLimitVelocity[localV] = 0.25;
            }
        }
        metalrobo::ArticulatedJointLimitProblem allLimitProblem;
        const auto allLimitBuild =
            metalrobo::buildArticulatedJointLimitProblem(
                model,
                0u,
                allLimitQ,
                allLimitVelocity,
                allLimitProblem,
                config
            );
        require(
            allLimitBuild.succeeded() &&
            allLimitProblem.rows.size() ==
                metalrobo::kUnitreeG1JointCount,
            "full G1 joint-limit operator did not compile 29 rows"
        );
        const auto allLimitSolution =
            metalrobo::solveArticulatedJointLimits(
                allLimitProblem,
                config
            );
        require(
            allLimitSolution.converged() &&
            allLimitSolution.diagnostics.scaledKktCertificate <=
                2.0e-12 &&
            allLimitSolution.diagnostics
                    .maximumPhysicalVelocityViolation <=
                2.0e-8,
            "fully coupled 29-row G1 limit solve failed"
        );

        std::vector<double> predictiveQ = defaultQ;
        std::vector<double> predictiveVelocity(
            articulation.nv,
            0.0
        );
        predictiveQ[lowerLocalQ] =
            static_cast<double>(lowerDof.limits.x) + 1.0e-2;
        predictiveVelocity[lowerLocalV] = -20.0;
        metalrobo::ArticulatedJointLimitProblem predictiveProblem;
        const auto predictiveBuild =
            metalrobo::buildArticulatedJointLimitProblem(
                model,
                0u,
                predictiveQ,
                predictiveVelocity,
                predictiveProblem,
                config
            );
        require(
            predictiveBuild.succeeded() &&
            predictiveProblem.rows.size() == 1u &&
            predictiveProblem.rows[0].gap >
                config.activationDistance,
            "one-step crossing did not predictively activate lower stop"
        );
        const auto predictiveSolution =
            metalrobo::solveArticulatedJointLimits(
                predictiveProblem,
                config
            );
        require(
            predictiveSolution.converged() &&
            predictiveSolution.constraintVelocity[0] >=
                predictiveProblem.rows[0].targetVelocity - 2.0e-8 &&
            predictiveSolution.generalizedVelocity[lowerLocalV] >
                predictiveVelocity[lowerLocalV],
            "predictive stop failed to prevent the one-step crossing"
        );

        const auto sentinel = sentinelProblem();
        auto rollbackProblem = sentinel;
        std::vector<double> invalidQ = q;
        invalidQ[lowerLocalQ] =
            std::numeric_limits<double>::quiet_NaN();
        const auto invalidBuild =
            metalrobo::buildArticulatedJointLimitProblem(
                model,
                0u,
                invalidQ,
                freeVelocity,
                rollbackProblem,
                config
            );
        require(
            !invalidBuild.succeeded() &&
            sameProblem(rollbackProblem, sentinel),
            "non-finite build failure published a partial problem"
        );

        metalrobo::ArticulatedJointLimitConfig capacityConfig =
            config;
        capacityConfig.maximumRows = 1u;
        const auto capacityBuild =
            metalrobo::buildArticulatedJointLimitProblem(
                model,
                0u,
                q,
                freeVelocity,
                rollbackProblem,
                capacityConfig
            );
        require(
            capacityBuild.status ==
                metalrobo::ArticulatedJointLimitStatus::
                    capacityExceeded &&
            sameProblem(rollbackProblem, sentinel),
            "capacity failure published a partial problem"
        );

        std::vector<double> rollbackVelocity = freeVelocity;
        const std::vector<double> rollbackVelocityBefore =
            rollbackVelocity;
        std::vector<double> negativeImpulse(
            problem.rows.size(),
            0.0
        );
        negativeImpulse[0] = -1.0;
        const auto invalidApply =
            metalrobo::applyArticulatedJointLimitImpulses(
                problem,
                negativeImpulse,
                rollbackVelocity
            );
        require(
            invalidApply.status ==
                metalrobo::ArticulatedJointLimitStatus::
                    invalidImpulse &&
            bitwiseEqual(
                rollbackVelocity,
                rollbackVelocityBefore
            ),
            "invalid impulse failure mutated velocity"
        );

        std::cout
            << std::setprecision(10)
            << "articulated_joint_limits=cpu_fp64"
            << " rows=" << problem.rows.size()
            << " lower=" << buildDiagnostics.lowerRowCount
            << " upper=" << buildDiagnostics.upperRowCount
            << " penetration="
            << buildDiagnostics.maximumPenetration
            << " coupling=" << coupling
            << " max_impulse="
            << solution.diagnostics.maximumImpulse
            << " kkt="
            << solution.diagnostics.scaledKktCertificate
            << " factor_residual="
            << solution.diagnostics.maximumFactorSolveResidual
            << " full_g1_rows=" << allLimitProblem.rows.size()
            << " predictive=pass"
            << " q_clamp=none"
            << " replay=bitwise"
            << " transaction=pass"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "articulated_joint_limits_probe failed: "
            << exception.what() << '\n';
        return 1;
    }
}
