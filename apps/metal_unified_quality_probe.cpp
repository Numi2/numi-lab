#include "metalrobo/MetalUnifiedQualitySolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

MRUnifiedQualityBlockGPU scalarBlock(
    const std::uint32_t row,
    const float regularization,
    const float lower,
    const float upper,
    const bool equality,
    const std::uint32_t key
) {
    MRUnifiedQualityBlockGPU block{};
    block.layout = {
        row,
        1u,
        MR_UNIFIED_QUALITY_SCALAR_INTERVAL,
        equality
            ? MR_UNIFIED_QUALITY_BLOCK_HARD_EQUALITY
            : 0u,
    };
    block.stableKey = {key, 0u, 0u, 0u};
    block.scale0 = {1.0f, 1.0f, 1.0f, 1.0f};
    block.scale1 = {1.0f, 1.0f, 1.0f, 1.0f};
    block.regularization0 = {
        regularization,
        1.0f,
        1.0f,
        1.0f,
    };
    block.regularization1 = {1.0f, 1.0f, 1.0f, 1.0f};
    block.boundsAndShift = {lower, upper, 0.0f, 0.0f};
    return block;
}

MRUnifiedQualityBlockGPU coneBlock(
    const std::uint32_t row,
    const std::span<const float> scales,
    const float regularization,
    const float adhesion,
    const float maximumNormal,
    const std::uint32_t key
) {
    MRUnifiedQualityBlockGPU block{};
    block.layout = {
        row,
        static_cast<std::uint32_t>(scales.size()),
        MR_UNIFIED_QUALITY_ELLIPTIC_CONE,
        0u,
    };
    block.stableKey = {key, 1u, 0u, 0u};
    block.scale0 = {1.0f, 1.0f, 1.0f, 1.0f};
    block.scale1 = {1.0f, 1.0f, 1.0f, 1.0f};
    block.regularization0 = {
        regularization,
        regularization,
        regularization,
        regularization,
    };
    block.regularization1 = {
        regularization,
        regularization,
        regularization,
        regularization,
    };
    for (std::size_t index = 0u;
         index < scales.size();
         ++index) {
        if (index < 4u) {
            (&block.scale0.x)[index] = scales[index];
        } else {
            (&block.scale1.x)[index - 4u] = scales[index];
        }
    }
    block.boundsAndShift = {
        adhesion,
        maximumNormal,
        0.0f,
        0.0f,
    };
    return block;
}

} // namespace

int main() {
    try {
        constexpr std::size_t environments = 2u;
        constexpr std::size_t nv = 12u;
        constexpr std::size_t rows = 15u;
        const float infinity =
            std::numeric_limits<float>::max();
        const std::array<float, 3> contactScale{
            1.0f,
            0.72f,
            0.31f,
        };
        const std::array<float, 4> torsionScale{
            1.0f,
            0.61f,
            0.38f,
            0.12f,
        };
        const std::array<float, 6> patchScale{
            1.0f,
            0.81f,
            0.44f,
            0.09f,
            0.035f,
            0.027f,
        };
        const std::array blocks{
            scalarBlock(
                0u,
                2.0e-2f,
                -infinity,
                infinity,
                true,
                11u
            ),
            scalarBlock(
                1u,
                3.0e-2f,
                0.0f,
                infinity,
                false,
                12u
            ),
            coneBlock(
                2u,
                contactScale,
                1.8e-2f,
                0.0f,
                0.0f,
                21u
            ),
            coneBlock(
                5u,
                torsionScale,
                2.4e-2f,
                0.015f,
                0.75f,
                22u
            ),
            coneBlock(
                9u,
                patchScale,
                3.1e-2f,
                0.0f,
                0.0f,
                23u
            ),
        };

        std::vector<float> dynamics(
            environments * nv * nv,
            0.0f
        );
        std::vector<float> jacobian(
            environments * rows * nv,
            0.0f
        );
        std::vector<float> bias(
            environments * rows,
            0.0f
        );
        std::vector<float> freeVelocity(
            environments * nv,
            0.0f
        );
        std::vector<float> warmVelocity(
            environments * nv,
            0.0f
        );
        std::vector<float> warmImpulses(
            environments * rows,
            0.0f
        );
        for (std::size_t environment = 0u;
             environment < environments;
             ++environment) {
            for (std::size_t row = 0u; row < nv; ++row) {
                for (std::size_t column = 0u;
                     column < nv;
                     ++column) {
                    dynamics[
                        environment * nv * nv +
                        row * nv +
                        column
                    ] =
                        row == column
                        ? 1.25f +
                            0.07f *
                                static_cast<float>(row)
                        : 0.015f /
                            (
                                1.0f +
                                static_cast<float>(
                                    row > column
                                    ? row - column
                                    : column - row
                                )
                            );
                }
                freeVelocity[environment * nv + row] =
                    0.04f *
                    std::sin(
                        0.37f *
                        static_cast<float>(
                            1u + row + 3u * environment
                        )
                    );
                warmVelocity[environment * nv + row] =
                    0.5f *
                    freeVelocity[environment * nv + row];
            }
            for (std::size_t row = 0u; row < rows; ++row) {
                bias[environment * rows + row] =
                    row == 0u
                    ? 0.08f
                    : row == 1u
                    ? -0.11f
                    : -0.045f +
                        0.012f *
                            static_cast<float>(row % 4u);
                for (std::size_t dof = 0u;
                     dof < nv;
                     ++dof) {
                    jacobian[
                        environment * rows * nv +
                        row * nv +
                        dof
                    ] =
                        0.16f *
                        std::sin(
                            0.19f *
                            static_cast<float>(
                                1u +
                                row * 7u +
                                dof * 3u +
                                environment
                            )
                        );
                }
            }
        }

        metalrobo::MetalUnifiedQualityProblem problem;
        problem.problemCount = environments;
        problem.generalizedVelocityCount = nv;
        problem.rowCount = rows;
        problem.blocks = blocks;
        problem.dynamics = dynamics;
        problem.jacobian = jacobian;
        problem.bias = bias;
        problem.freeVelocity = freeVelocity;
        problem.warmVelocity = warmVelocity;
        problem.warmImpulses = warmImpulses;

        metalrobo::MetalUnifiedQualityConfig directConfig;
        directConfig.maximumNewtonIterations = 40u;
        directConfig.maximumPCGIterations = 128u;
        directConfig.optimalityTolerance = 1.0e-5f;
        directConfig.feasibilityTolerance = 2.0e-5f;
        directConfig.directMaximumGeneralizedVelocities = 64u;
        metalrobo::MetalUnifiedQualityResult direct;
        const auto directDiagnostics =
            metalrobo::solveMetalUnifiedQuality(
                problem,
                direct,
                directConfig
            );
        require(
            directDiagnostics.succeeded(),
            "direct unified quality failed: " +
                directDiagnostics.message
        );

        auto pcgConfig = directConfig;
        pcgConfig.directMaximumGeneralizedVelocities = 1u;
        pcgConfig.maximumPCGIterations = 128u;
        metalrobo::MetalUnifiedQualityResult pcg;
        const auto pcgDiagnostics =
            metalrobo::solveMetalUnifiedQuality(
                problem,
                pcg,
                pcgConfig
            );
        require(
            pcgDiagnostics.succeeded(),
            "matrix-free unified quality failed: " +
                pcgDiagnostics.message
        );

        float maximumAgreement = 0.0f;
        for (std::size_t index = 0u;
             index < direct.velocity.size();
             ++index) {
            maximumAgreement = std::max(
                maximumAgreement,
                std::abs(
                    direct.velocity[index] -
                    pcg.velocity[index]
                )
            );
        }
        for (std::size_t index = 0u;
             index < direct.impulses.size();
             ++index) {
            maximumAgreement = std::max(
                maximumAgreement,
                std::abs(
                    direct.impulses[index] -
                    pcg.impulses[index]
                )
            );
        }
        float maximumOptimality = 0.0f;
        float maximumFeasibility = 0.0f;
        std::uint32_t maximumNewton = 0u;
        std::uint32_t maximumPCG = 0u;
        for (const auto& status : direct.statuses) {
            require(
                status.solvePath ==
                    MR_UNIFIED_QUALITY_PATH_DIRECT,
                "direct solve selected the wrong path"
            );
            maximumOptimality = std::max(
                maximumOptimality,
                status.certificates0.x
            );
            maximumFeasibility = std::max(
                maximumFeasibility,
                status.certificates0.y
            );
            maximumNewton = std::max(
                maximumNewton,
                status.newtonIterations
            );
        }
        for (const auto& status : pcg.statuses) {
            require(
                status.solvePath ==
                    MR_UNIFIED_QUALITY_PATH_PCG,
                "PCG solve selected the wrong path"
            );
            maximumOptimality = std::max(
                maximumOptimality,
                status.certificates0.x
            );
            maximumFeasibility = std::max(
                maximumFeasibility,
                status.certificates0.y
            );
            maximumNewton = std::max(
                maximumNewton,
                status.newtonIterations
            );
            maximumPCG = std::max(
                maximumPCG,
                status.pcgIterations
            );
        }
        require(
            maximumAgreement < 8.0e-4f &&
                maximumOptimality <=
                    directConfig.optimalityTolerance &&
                maximumFeasibility <=
                    directConfig.feasibilityTolerance,
            "direct/PCG quality closure failed: agreement=" +
                std::to_string(maximumAgreement) +
                " optimality=" +
                std::to_string(maximumOptimality) +
                " feasibility=" +
                std::to_string(maximumFeasibility)
        );

        // Cross the direct-path row threshold with a sparse but fully
        // coupled product-cone problem. This exercises the 768-row
        // threadgroup workspace and proves that PCG does not require an
        // environment-major dense Hessian allocation.
        constexpr std::size_t largeNv = 96u;
        constexpr std::size_t largeRows = 300u;
        constexpr std::size_t largeBlockCount = 50u;
        std::vector<MRUnifiedQualityBlockGPU> largeBlocks;
        largeBlocks.reserve(largeBlockCount);
        for (std::size_t block = 0u;
             block < largeBlockCount;
             ++block) {
            largeBlocks.push_back(coneBlock(
                static_cast<std::uint32_t>(6u * block),
                patchScale,
                5.0e-1f,
                0.0f,
                0.0f,
                static_cast<std::uint32_t>(1000u + block)
            ));
        }
        std::vector<float> largeDynamics(
            largeNv * largeNv,
            0.0f
        );
        std::vector<float> largeJacobian(
            largeRows * largeNv,
            0.0f
        );
        std::vector<float> largeBias(largeRows, 0.0f);
        std::vector<float> largeVelocity(largeNv, 0.0f);
        std::vector<float> largeImpulses(largeRows, 0.0f);
        for (std::size_t dof = 0u; dof < largeNv; ++dof) {
            largeDynamics[dof * largeNv + dof] =
                1.0f + 0.002f * static_cast<float>(dof);
        }
        for (std::size_t row = 0u; row < largeRows; ++row) {
            largeJacobian[
                row * largeNv + row % largeNv
            ] = 0.05f;
            largeBias[row] =
                row % 6u == 0u
                ? -0.005f
                : 2.0e-4f *
                    static_cast<float>(
                        static_cast<int>(row % 3u) - 1
                    );
        }
        metalrobo::MetalUnifiedQualityProblem largeProblem;
        largeProblem.problemCount = 1u;
        largeProblem.generalizedVelocityCount = largeNv;
        largeProblem.rowCount = largeRows;
        largeProblem.blocks = largeBlocks;
        largeProblem.dynamics = largeDynamics;
        largeProblem.jacobian = largeJacobian;
        largeProblem.bias = largeBias;
        largeProblem.freeVelocity = largeVelocity;
        largeProblem.warmVelocity = largeVelocity;
        largeProblem.warmImpulses = largeImpulses;

        auto largeConfig = directConfig;
        largeConfig.maximumNewtonIterations = 40u;
        metalrobo::MetalUnifiedQualityResult largeResult;
        const auto largeDiagnostics =
            metalrobo::solveMetalUnifiedQuality(
                largeProblem,
                largeResult,
                largeConfig
            );
        require(
            largeDiagnostics.succeeded() &&
                largeResult.statuses.size() == 1u &&
                largeResult.statuses[0].solvePath ==
                    MR_UNIFIED_QUALITY_PATH_PCG,
            "large-row quality problem did not use matrix-free PCG: " +
                largeDiagnostics.message
        );

        std::cout
            << "metal_unified_quality=ok"
            << " device=\"" << directDiagnostics.deviceName
            << "\""
            << " environments=" << environments
            << " nv=" << nv
            << " rows=" << rows
            << " blocks=" << blocks.size()
            << " direct_pcg_error=" << maximumAgreement
            << " optimality=" << maximumOptimality
            << " feasibility=" << maximumFeasibility
            << " newton=" << maximumNewton
            << " pcg=" << maximumPCG
            << " allocated_bytes="
            << directDiagnostics.allocatedBytes
            << " large_rows=" << largeRows
            << " large_pcg="
            << largeResult.statuses[0].pcgIterations
            << " large_allocated_bytes="
            << largeDiagnostics.allocatedBytes
            << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "metal_unified_quality=failed reason=\""
            << exception.what() << "\"\n";
        return 1;
    }
}
