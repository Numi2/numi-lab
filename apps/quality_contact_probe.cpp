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
#include <string>
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

metalrobo::ContactSpaceConicProblem makeContactSpaceProblem(
    const metalrobo::DenseConicProblem& dense
) {
    const std::size_t dimension = 3u * dense.contacts.size();
    metalrobo::ContactSpaceConicProblem result;
    result.delassus.assign(dimension * dimension, 0.0);
    result.freeContactVelocity.assign(dimension, 0.0);
    result.contacts.reserve(dense.contacts.size());

    std::vector<const std::vector<double>*> rows;
    rows.reserve(dimension);
    for (const auto& contact : dense.contacts) {
        rows.push_back(&contact.normalJacobian);
        rows.push_back(&contact.tangentUJacobian);
        rows.push_back(&contact.tangentVJacobian);
        result.contacts.push_back({
            contact.targetVelocity,
            contact.regularization,
            contact.warmImpulse,
            contact.friction,
        });
    }
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t dof = 0u; dof < dense.nv; ++dof) {
            result.freeContactVelocity[row] +=
                (*rows[row])[dof] * dense.freeVelocity[dof];
        }
        for (std::size_t column = 0u;
             column < dimension;
             ++column) {
            for (std::size_t leftDof = 0u;
                 leftDof < dense.nv;
                 ++leftDof) {
                for (std::size_t rightDof = 0u;
                     rightDof < dense.nv;
                     ++rightDof) {
                    result.delassus[row * dimension + column] +=
                        (*rows[row])[leftDof] *
                        dense.inverseMass[
                            leftDof * dense.nv + rightDof
                        ] *
                        (*rows[column])[rightDof];
                }
            }
        }
    }
    return result;
}

std::vector<double> contactVelocity(
    const metalrobo::DenseConicProblem& problem,
    const std::vector<double>& generalizedVelocity
) {
    std::vector<double> result(3u * problem.contacts.size(), 0.0);
    for (std::size_t contact = 0u;
         contact < problem.contacts.size();
         ++contact) {
        const auto& block = problem.contacts[contact];
        const std::array<const std::vector<double>*, 3u> rows{
            &block.normalJacobian,
            &block.tangentUJacobian,
            &block.tangentVJacobian,
        };
        for (std::size_t axis = 0u; axis < rows.size(); ++axis) {
            for (std::size_t dof = 0u;
                 dof < generalizedVelocity.size();
                 ++dof) {
                result[3u * contact + axis] +=
                    (*rows[axis])[dof] * generalizedVelocity[dof];
            }
        }
    }
    return result;
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
        const auto contactSpace =
            metalrobo::solveQualityContactSpaceProblem(
                makeContactSpaceProblem(problem),
                qualityConfig
            );
        const double contactSpaceImpulseError = relativeDistance(
            contactSpace.impulses,
            quality.impulses
        );
        const double contactSpaceVelocityError = relativeDistance(
            contactSpace.velocity,
            contactVelocity(problem, quality.velocity)
        );
        if (!contactSpace.converged() ||
            contactSpaceImpulseError > 2.0e-12 ||
            contactSpaceVelocityError > 2.0e-12 ||
            std::abs(contactSpace.objective - quality.objective) >
                2.0e-12 * (1.0 + std::abs(quality.objective))) {
            throw std::runtime_error(
                "contact-space and dense quality paths disagree"
            );
        }
        metalrobo::ContactSpaceConicProblem indefinitePhysicalOperator =
            makeContactSpaceProblem(problem);
        std::ranges::fill(
            indefinitePhysicalOperator.delassus,
            0.0
        );
        // Regularization keeps the total Hessian positive, so only an
        // independent physical-W PSD gate catches this malformed operator.
        indefinitePhysicalOperator.delassus[0] = -1.0e-4;
        const auto indefinitePhysicalResult =
            metalrobo::solveQualityContactSpaceProblem(
                indefinitePhysicalOperator,
                qualityConfig
            );
        if (indefinitePhysicalResult.code !=
                MR_STEP_NONFINITE_INPUT ||
            indefinitePhysicalResult.failure.find(
                "positive semidefinite"
            ) == std::string::npos) {
            throw std::runtime_error(
                "indefinite physical Delassus operator was accepted"
            );
        }
        metalrobo::ContactSpaceConicProblem zeroPhysicalOperator =
            makeContactSpaceProblem(problem);
        std::ranges::fill(zeroPhysicalOperator.delassus, 0.0);
        const auto zeroPhysicalResult =
            metalrobo::solveQualityContactSpaceProblem(
                zeroPhysicalOperator,
                qualityConfig
            );
        if (!zeroPhysicalResult.converged()) {
            throw std::runtime_error(
                "rank-deficient positive-semidefinite Delassus was rejected"
            );
        }

        const std::size_t contactDimension =
            zeroPhysicalOperator.freeContactVelocity.size();
        metalrobo::ContactSpaceConicProblem rankOnePhysicalOperator =
            zeroPhysicalOperator;
        std::vector<double> rankOneVector(contactDimension, 0.0);
        for (std::size_t contact = 0u;
             contact < rankOnePhysicalOperator.contacts.size();
             ++contact) {
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                rankOnePhysicalOperator.freeContactVelocity[
                    3u * contact + axis
                ] = rankOnePhysicalOperator.contacts[contact]
                        .targetVelocity[axis];
            }
        }
        for (std::size_t index = 0u;
             index < contactDimension;
             ++index) {
            rankOneVector[index] =
                (index % 2u == 0u ? 1.0 : -1.0) *
                (0.25 + 0.07 * static_cast<double>(index));
        }
        for (std::size_t row = 0u;
             row < contactDimension;
             ++row) {
            for (std::size_t column = 0u;
                 column < contactDimension;
                 ++column) {
                rankOnePhysicalOperator.delassus[
                    row * contactDimension + column
                ] = rankOneVector[row] * rankOneVector[column];
            }
        }
        const auto rankOnePhysicalResult =
            metalrobo::solveQualityContactSpaceProblem(
                rankOnePhysicalOperator,
                qualityConfig
            );
        metalrobo::ContactSpaceConicProblem
            permutedRankOnePhysicalOperator =
                rankOnePhysicalOperator;
        for (std::size_t row = 0u;
             row < contactDimension;
             ++row) {
            for (std::size_t column = 0u;
                 column < contactDimension;
                 ++column) {
                permutedRankOnePhysicalOperator.delassus[
                    row * contactDimension + column
                ] = rankOnePhysicalOperator.delassus[
                    (contactDimension - 1u - row) *
                            contactDimension +
                        (contactDimension - 1u - column)
                ];
            }
        }
        const auto permutedRankOnePhysicalResult =
            metalrobo::solveQualityContactSpaceProblem(
                permutedRankOnePhysicalOperator,
                qualityConfig
            );
        if (!rankOnePhysicalResult.converged() ||
            !permutedRankOnePhysicalResult.converged()) {
            throw std::runtime_error(
                "nonzero rank-deficient PSD Delassus was rejected: " +
                rankOnePhysicalResult.failure + " / " +
                permutedRankOnePhysicalResult.failure
            );
        }

        const double psdTolerance =
            256.0 * std::numeric_limits<double>::epsilon() *
            static_cast<double>(contactDimension);
        metalrobo::ContactSpaceConicProblem
            withinPsdTolerance = rankOnePhysicalOperator;
        std::ranges::fill(withinPsdTolerance.delassus, 0.0);
        withinPsdTolerance.delassus[0] = 1.0;
        withinPsdTolerance.delassus[
            (contactDimension - 1u) * contactDimension +
            (contactDimension - 1u)
        ] = -0.5 * psdTolerance;
        const auto withinPsdToleranceResult =
            metalrobo::solveQualityContactSpaceProblem(
                withinPsdTolerance,
                qualityConfig
            );
        if (!withinPsdToleranceResult.converged()) {
            throw std::runtime_error(
                "Delassus eigenvalue inside PSD tolerance was rejected"
            );
        }
        metalrobo::ContactSpaceConicProblem
            outsidePsdTolerance = withinPsdTolerance;
        outsidePsdTolerance.delassus[
            (contactDimension - 1u) * contactDimension +
            (contactDimension - 1u)
        ] = -2.0 * psdTolerance;
        const auto outsidePsdToleranceResult =
            metalrobo::solveQualityContactSpaceProblem(
                outsidePsdTolerance,
                qualityConfig
            );
        if (outsidePsdToleranceResult.code !=
                MR_STEP_NONFINITE_INPUT ||
            outsidePsdToleranceResult.failure.find(
                "positive semidefinite"
            ) == std::string::npos) {
            throw std::runtime_error(
                "Delassus eigenvalue outside PSD tolerance was accepted"
            );
        }

        metalrobo::ContactSpaceConicProblem denseTinyIndefinite =
            zeroPhysicalOperator;
        denseTinyIndefinite.delassus[0] = 1.0;
        for (std::size_t row = 1u;
             row < contactDimension;
             ++row) {
            denseTinyIndefinite.delassus[
                row * contactDimension + row
            ] = 0.5 * psdTolerance;
            for (std::size_t column = 1u;
                 column < row;
                 ++column) {
                denseTinyIndefinite.delassus[
                    row * contactDimension + column
                ] = -7.0 * psdTolerance;
                denseTinyIndefinite.delassus[
                    column * contactDimension + row
                ] = -7.0 * psdTolerance;
            }
        }
        const auto denseTinyIndefiniteResult =
            metalrobo::solveQualityContactSpaceProblem(
                denseTinyIndefinite,
                qualityConfig
            );
        if (denseTinyIndefiniteResult.code !=
                MR_STEP_NONFINITE_INPUT ||
            denseTinyIndefiniteResult.failure.find(
                "positive semidefinite"
            ) == std::string::npos) {
            throw std::runtime_error(
                "dense tiny-indefinite Delassus tail was accepted"
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
                  << " contact_space_impulse_error="
                  << contactSpaceImpulseError
                  << " contact_space_velocity_error="
                  << contactSpaceVelocityError
                  << " delassus_psd_gate=yes"
                  << " rank_deficient_psd=yes"
                  << " nonzero_rank_deficient_psd=yes"
                  << " psd_tolerance_contract=yes"
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
