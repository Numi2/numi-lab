#include "metalrobo/ReferenceConicSolver.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

metalrobo::DenseConicProblem makeProblem(const double angle) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    metalrobo::DenseConicProblem problem;
    problem.nv = 3u;
    problem.inverseMass = {
        1.3, 0.2, 0.1,
        0.2, 0.8, 0.05,
        0.1, 0.05, 1.7,
    };
    problem.freeVelocity = {2.0, -1.0, 0.5};

    metalrobo::DenseContactBlock contact;
    contact.normalJacobian = {0.0, 1.0, 0.0};
    contact.tangentUJacobian = {cosine, 0.0, sine};
    contact.tangentVJacobian = {-sine, 0.0, cosine};
    contact.regularization = {1.0e-5, 1.0e-5, 1.0e-5};
    contact.friction = 0.7;
    problem.contacts.push_back(contact);
    return problem;
}

double distance(
    const std::vector<double>& left,
    const std::vector<double>& right
) {
    double squared = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const double difference = left[index] - right[index];
        squared += difference * difference;
    }
    return std::sqrt(squared);
}

} // namespace

int main() {
    try {
        metalrobo::ReferenceConicSolverConfig config;
        config.maximumIterations = 50000u;
        config.optimalityTolerance = 1.0e-11;

        const auto baseline = metalrobo::solveReferenceConicProblem(
            makeProblem(0.0),
            config
        );
        const auto rotated = metalrobo::solveReferenceConicProblem(
            makeProblem(0.731),
            config
        );
        if (!baseline.converged() || !rotated.converged()) {
            throw std::runtime_error("exact-cone reference did not converge");
        }
        const double basisError =
            distance(baseline.velocity, rotated.velocity);
        if (baseline.scaledOptimalityResidual > config.optimalityTolerance ||
            rotated.scaledOptimalityResidual > config.optimalityTolerance ||
            baseline.maximumConeViolation > 1.0e-12 ||
            rotated.maximumConeViolation > 1.0e-12 ||
            basisError > 1.0e-9) {
            throw std::runtime_error(
                "reference conic accuracy or basis invariance failed"
            );
        }

        metalrobo::DenseConicProblem warmProblem = makeProblem(0.0);
        warmProblem.contacts[0].warmImpulse = {
            baseline.impulses[0],
            baseline.impulses[1],
            baseline.impulses[2],
        };
        const auto warm = metalrobo::solveReferenceConicProblem(
            warmProblem,
            config
        );
        if (!warm.converged() || !warm.usedWarmStart ||
            distance(warm.velocity, baseline.velocity) > 1.0e-10 ||
            warm.iterations > baseline.iterations) {
            throw std::runtime_error("cost-aware warm start regressed");
        }

        std::cout << std::scientific << std::setprecision(6)
                  << "solver=reference_fp64_exact_cone"
                  << " iterations_zero=" << baseline.iterations
                  << " iterations_warm=" << warm.iterations
                  << " optimality=" << baseline.scaledOptimalityResidual
                  << " cone_violation=" << baseline.maximumConeViolation
                  << " tangent_basis_velocity_error=" << basisError
                  << " warm_same_solution=yes"
                  << " finite=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_reference_conic_probe: "
                  << error.what() << '\n';
        return 1;
    }
}
