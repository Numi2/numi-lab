#include "metalrobo/MetalQualityContactSolver.hpp"
#include "metalrobo/QualityContactSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double maximumError(
    const std::vector<double>& left,
    const std::vector<double>& right
) {
    require(
        left.size() == right.size(),
        "quality result dimensions differ"
    );
    double error = 0.0;
    for (std::size_t index = 0u;
         index < left.size();
         ++index) {
        error = std::max(
            error,
            std::abs(left[index] - right[index])
        );
    }
    return error;
}

} // namespace

int main() {
    try {
        metalrobo::ContactSpaceConicProblem problem;
        constexpr std::size_t contacts = 2u;
        constexpr std::size_t dimension = 3u * contacts;
        problem.delassus.assign(
            dimension * dimension,
            0.0
        );
        const std::array<double, dimension> diagonal{
            1.2, 0.9, 0.85,
            1.0, 0.75, 0.8,
        };
        for (std::size_t row = 0u;
             row < dimension;
             ++row) {
            problem.delassus[row * dimension + row] =
                diagonal[row];
        }
        const auto couple = [&](const std::size_t a,
                                const std::size_t b,
                                const double value) {
            problem.delassus[a * dimension + b] = value;
            problem.delassus[b * dimension + a] = value;
        };
        couple(0u, 3u, 0.18);
        couple(1u, 4u, 0.06);
        couple(2u, 5u, -0.04);
        problem.freeContactVelocity = {
            -1.2, 0.45, -0.15,
            -0.8, -0.25, 0.35,
        };
        problem.contacts.resize(contacts);
        problem.contacts[0].friction = 0.65;
        problem.contacts[1].friction = 0.80;
        for (std::size_t contact = 0u;
             contact < contacts;
             ++contact) {
            problem.contacts[contact].regularization = {
                0.04,
                0.03,
                0.03,
            };
        }

        metalrobo::QualityContactSolverConfig cpuConfig;
        cpuConfig.maximumIterations = 400u;
        cpuConfig.kktTolerance = 1.0e-11;
        const metalrobo::QualityContactSolution cpu =
            metalrobo::solveQualityContactSpaceProblem(
                problem,
                cpuConfig
            );
        require(
            cpu.converged(),
            "FP64 quality oracle did not converge"
        );

        metalrobo::MetalQualityContactSolution gpu;
        metalrobo::MetalQualityContactSolverConfig gpuConfig;
        gpuConfig.maximumNewtonIterations = 64u;
        gpuConfig.maximumCGIterations = 128u;
        gpuConfig.convergenceTolerance = 3.0e-5f;
        const auto diagnostics =
            metalrobo::solveMetalQualityContactSpace(
                problem,
                gpu,
                gpuConfig
            );
        require(
            diagnostics.succeeded(),
            "Metal quality solve failed: " +
                diagnostics.message
        );
        const double impulseError =
            maximumError(gpu.impulses, cpu.impulses);
        const double velocityError =
            maximumError(gpu.velocity, cpu.velocity);
        require(
            impulseError < 2.5e-3 &&
                velocityError < 2.5e-3 &&
                gpu.gpuStatus.diagnostics.x <=
                    1.1f *
                    gpuConfig.convergenceTolerance &&
                gpu.gpuStatus.diagnostics.z < 2.0e-5f,
            "Metal semismooth quality result disagrees with FP64 oracle"
        );

        metalrobo::MetalQualityContactSolution sentinel = gpu;
        const auto sentinelImpulses = sentinel.impulses;
        metalrobo::ContactSpaceConicProblem unsupported =
            problem;
        unsupported.contacts[0].friction = 0.0;
        const auto rejected =
            metalrobo::solveMetalQualityContactSpace(
                unsupported,
                sentinel,
                gpuConfig
            );
        require(
            !rejected.succeeded() &&
                sentinel.impulses == sentinelImpulses,
            "unsupported Metal quality problem changed output"
        );

        std::cout
            << "metal_quality_contact=ok"
            << " device=\"" << diagnostics.deviceName << "\""
            << " contacts=" << contacts
            << " newton="
            << gpu.gpuStatus.newtonIterations
            << " cg=" << gpu.gpuStatus.cgIterations
            << " backtracks="
            << gpu.gpuStatus.lineSearchBacktracks
            << " projected_fallbacks="
            << gpu.gpuStatus.projectedGradientFallbacks
            << " residual="
            << gpu.gpuStatus.diagnostics.x
            << " cone_violation="
            << gpu.gpuStatus.diagnostics.z
            << " impulse_error=" << impulseError
            << " velocity_error=" << velocityError
            << " transactional=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "metal_quality_contact=failed reason=\""
            << error.what()
            << "\"\n";
        return 1;
    }
}
