#include "metalrobo/QualityContactSolver.hpp"
#include "metalrobo/ReferenceConicSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

metalrobo::DenseConicProblem makeCoupledProblem() {
    metalrobo::DenseConicProblem problem;
    problem.nv = 6u;
    problem.inverseMass = {
        1.20,  0.08, -0.03,  0.04,  0.02, -0.01,
        0.08,  0.95,  0.05, -0.02,  0.03,  0.01,
       -0.03,  0.05,  1.35,  0.01, -0.04,  0.03,
        0.04, -0.02,  0.01,  0.72,  0.06, -0.02,
        0.02,  0.03, -0.04,  0.06,  0.81,  0.05,
       -0.01,  0.01,  0.03, -0.02,  0.05,  0.68,
    };
    problem.freeVelocity = {2.1, -1.4, -0.8, 0.7, -0.3, 0.4};

    const std::array<std::array<double, 3>, 4> points{{
        { 0.30, -0.50,  0.20},
        {-0.30, -0.50,  0.20},
        { 0.25, -0.50, -0.25},
        {-0.25, -0.50, -0.25},
    }};
    const std::array<double, 4> frictions{0.72, 0.48, 0.91, 0.63};
    for (std::size_t index = 0; index < points.size(); ++index) {
        const double x = points[index][0];
        const double y = points[index][1];
        const double z = points[index][2];
        metalrobo::DenseContactBlock contact;
        // Translational + angular point-velocity Jacobian rows for a shared
        // six-DOF body. Every block therefore couples to every other block.
        contact.normalJacobian = {0.0, 1.0, 0.0, -z, 0.0, x};
        contact.tangentUJacobian = {1.0, 0.0, 0.0, 0.0, z, -y};
        contact.tangentVJacobian = {0.0, 0.0, 1.0, y, -x, 0.0};
        contact.targetVelocity = {
            0.03 + 0.01 * static_cast<double>(index),
            0.0,
            0.0,
        };
        contact.regularization = {
            2.0e-4,
            1.0e-4,
            1.0e-4,
        };
        contact.friction = frictions[index];
        problem.contacts.push_back(std::move(contact));
    }
    return problem;
}

double distance(
    const std::vector<double>& left,
    const std::vector<double>& right
) {
    if (left.size() != right.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double squared = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const double difference = left[index] - right[index];
        squared += difference * difference;
    }
    return std::sqrt(squared);
}

double relativeDistance(
    const std::vector<double>& left,
    const std::vector<double>& right
) {
    double rightSquared = 0.0;
    for (const double value : right) {
        rightSquared += value * value;
    }
    return distance(left, right) / (1.0 + std::sqrt(rightSquared));
}

metalrobo::DenseConicProblem permuteProblem(
    const metalrobo::DenseConicProblem& original,
    const std::array<std::size_t, 4>& permutation
) {
    metalrobo::DenseConicProblem result = original;
    result.contacts.clear();
    for (const std::size_t index : permutation) {
        result.contacts.push_back(original.contacts[index]);
    }
    return result;
}

std::vector<double> undoImpulsePermutation(
    const std::vector<double>& permuted,
    const std::array<std::size_t, 4>& permutation
) {
    std::vector<double> result(permuted.size(), 0.0);
    for (std::size_t newIndex = 0;
         newIndex < permutation.size();
         ++newIndex) {
        const std::size_t oldIndex = permutation[newIndex];
        for (std::size_t axis = 0; axis < 3u; ++axis) {
            result[3u * oldIndex + axis] =
                permuted[3u * newIndex + axis];
        }
    }
    return result;
}

} // namespace

int main() {
    try {
        const metalrobo::DenseConicProblem problem =
            makeCoupledProblem();

        metalrobo::QualityContactSolverConfig qualityConfig;
        qualityConfig.maximumIterations = 200u;
        qualityConfig.kktTolerance = 1.0e-11;
        qualityConfig.enableWarmStart = false;
        const auto quality =
            metalrobo::solveQualityContactProblem(
                problem,
                qualityConfig
            );
        if (!quality.converged()) {
            throw std::runtime_error(
                "quality solve failed: " + quality.failure
            );
        }

        metalrobo::ReferenceConicSolverConfig referenceConfig;
        referenceConfig.maximumIterations = 200000u;
        referenceConfig.optimalityTolerance = 2.0e-11;
        referenceConfig.enableWarmStart = false;
        const auto reference =
            metalrobo::solveReferenceConicProblem(
                problem,
                referenceConfig
            );
        if (!reference.converged()) {
            throw std::runtime_error(
                "projected-gradient reference failed to converge"
            );
        }

        const double referenceImpulseError =
            relativeDistance(
                quality.impulses,
                reference.impulses
            );
        const double referenceVelocityError =
            relativeDistance(
                quality.velocity,
                reference.velocity
            );
        const double objectiveError = std::abs(
            quality.objective - reference.objective
        ) / (1.0 + std::abs(reference.objective));
        if (referenceImpulseError > 2.0e-8 ||
            referenceVelocityError > 2.0e-8 ||
            objectiveError > 2.0e-10) {
            throw std::runtime_error(
                "quality/reference solutions disagree"
            );
        }
        if (quality.iterations >= reference.iterations) {
            throw std::runtime_error(
                "semismooth Newton did not improve iteration count"
            );
        }
        if (quality.maximumPrimalConeViolation > 2.0e-13 ||
            quality.maximumDualConeViolation > 2.0e-9 ||
            quality.maximumNormalImpulseViolation > 2.0e-13 ||
            quality.scaledKktCertificate >
                1.1 * qualityConfig.kktTolerance) {
            throw std::runtime_error(
                "quality solution failed exact-cone/KKT certificate"
            );
        }

        constexpr std::array<std::size_t, 4> permutation{
            2u, 0u, 3u, 1u,
        };
        const auto permuted =
            metalrobo::solveQualityContactProblem(
                permuteProblem(problem, permutation),
                qualityConfig
            );
        if (!permuted.converged()) {
            throw std::runtime_error(
                "permuted coupled solve failed"
            );
        }
        const std::vector<double> unpermutedImpulses =
            undoImpulsePermutation(
                permuted.impulses,
                permutation
            );
        const double permutationImpulseError =
            relativeDistance(
                quality.impulses,
                unpermutedImpulses
            );
        const double permutationVelocityError =
            relativeDistance(
                quality.velocity,
                permuted.velocity
            );
        if (permutationImpulseError > 5.0e-10 ||
            permutationVelocityError > 5.0e-10) {
            throw std::runtime_error(
                "contact permutation changed the coupled solution"
            );
        }

        // Exercise the degenerate mu=0 cone as an exact nonnegative normal
        // ray. Here conic complementarity reduces to classical
        // lambda_n * w_n = 0 for that contact.
        metalrobo::DenseConicProblem frictionlessProblem = problem;
        frictionlessProblem.contacts[1].friction = 0.0;
        const auto frictionless =
            metalrobo::solveQualityContactProblem(
                frictionlessProblem,
                qualityConfig
            );
        const auto frictionlessReference =
            metalrobo::solveReferenceConicProblem(
                frictionlessProblem,
                referenceConfig
            );
        if (!frictionless.converged() ||
            !frictionlessReference.converged() ||
            frictionless
                .maximumFrictionlessNormalComplementarityResidual >
                2.0e-9 ||
            frictionless.maximumPrimalConeViolation > 2.0e-13 ||
            relativeDistance(
                frictionless.impulses,
                frictionlessReference.impulses
            ) > 2.0e-8) {
            throw std::runtime_error(
                "frictionless normal complementarity failed"
            );
        }

        metalrobo::DenseConicProblem warmProblem = problem;
        for (std::size_t contact = 0;
             contact < warmProblem.contacts.size();
             ++contact) {
            for (std::size_t axis = 0; axis < 3u; ++axis) {
                warmProblem.contacts[contact].warmImpulse[axis] =
                    quality.impulses[3u * contact + axis];
            }
        }
        qualityConfig.enableWarmStart = true;
        const auto warm =
            metalrobo::solveQualityContactProblem(
                warmProblem,
                qualityConfig
            );
        if (!warm.converged() ||
            !warm.usedWarmStart ||
            warm.iterations > quality.iterations ||
            relativeDistance(warm.impulses, quality.impulses) >
                2.0e-10) {
            throw std::runtime_error(
                "cost-aware warm start regressed convergence"
            );
        }

        metalrobo::QualityContactSolverConfig failureConfig =
            qualityConfig;
        failureConfig.enableWarmStart = false;
        failureConfig.maximumIterations = 1u;
        failureConfig.kktTolerance = 1.0e-15;
        const auto expectedFailure =
            metalrobo::solveQualityContactProblem(
                problem,
                failureConfig
            );
        if (expectedFailure.code != MR_STEP_DID_NOT_CONVERGE ||
            expectedFailure.failure.empty()) {
            throw std::runtime_error(
                "iteration exhaustion was not reported explicitly"
            );
        }

        std::cout << std::scientific << std::setprecision(6)
                  << "solver=quality_fp64_semismooth_newton"
                  << " contacts=" << problem.contacts.size()
                  << " iterations=" << quality.iterations
                  << " reference_iterations="
                  << reference.iterations
                  << " newton_steps="
                  << quality.semismoothNewtonSteps
                  << " gauss_newton_fallbacks="
                  << quality.gaussNewtonFallbackSteps
                  << " projected_gradient_fallbacks="
                  << quality.projectedGradientFallbackSteps
                  << " kkt=" << quality.scaledKktCertificate
                  << " primal_cone_violation="
                  << quality.maximumPrimalConeViolation
                  << " dual_cone_violation="
                  << quality.maximumDualConeViolation
                  << " contact_complementarity="
                  << quality.maximumConicComplementarityResidual
                  << " frictionless_normal_complementarity="
                  << frictionless
                      .maximumFrictionlessNormalComplementarityResidual
                  << " reference_impulse_error="
                  << referenceImpulseError
                  << " permutation_impulse_error="
                  << permutationImpulseError
                  << " warm_iterations=" << warm.iterations
                  << " explicit_failure=yes"
                  << " finite=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_quality_contact_probe: "
                  << error.what() << '\n';
        return 1;
    }
}
