#include "metalrobo/ArticulatedContact.hpp"
#include "metalrobo/G1.hpp"
#include "metalrobo/MetalArticulatedInverseMass.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef METALROBO_INVERSE_MASS_TEST_METALLIB
#define METALROBO_INVERSE_MASS_TEST_METALLIB ""
#endif

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

metalrobo::MetalArticulatedInverseMassConfig testConfig() {
    return {
        .metallibPath = METALROBO_INVERSE_MASS_TEST_METALLIB,
    };
}

struct Batch {
    std::uint32_t articulationIndex = 0u;
    std::size_t environmentCount = 0u;
    std::size_t rhsCount = 0u;
    std::vector<float> q;
    std::vector<float> rhs;

    [[nodiscard]]
    metalrobo::MetalArticulatedInverseMassInput input() const {
        return {
            .articulationIndex = articulationIndex,
            .environmentCount = environmentCount,
            .rhsCount = rhsCount,
            .q = q,
            .rightHandSides = rhs,
        };
    }
};

Batch makeBatch(
    const metalrobo::EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::size_t environmentCount,
    const std::size_t rhsCount
) {
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    Batch batch{
        .articulationIndex = articulationIndex,
        .environmentCount = environmentCount,
        .rhsCount = rhsCount,
    };
    batch.q.resize(environmentCount * articulation.nq);
    batch.rhs.resize(
        environmentCount * rhsCount * articulation.nv
    );
    for (std::size_t environment = 0u;
         environment < environmentCount;
         ++environment) {
        const std::size_t qBase =
            environment * articulation.nq;
        std::copy_n(
            model.defaultQ.begin() + articulation.qOffset,
            articulation.nq,
            batch.q.begin() +
                static_cast<std::ptrdiff_t>(qBase)
        );
        const std::size_t variant = environment % 7u;
        if (articulation.rootType == MR_ROOT_FLOATING) {
            batch.q[qBase] +=
                0.015f * static_cast<float>(variant);
            const float x =
                0.006f * static_cast<float>(variant);
            const float y =
                -0.004f * static_cast<float>(variant);
            const float z =
                0.003f * static_cast<float>(variant);
            batch.q[qBase + 3u] = x;
            batch.q[qBase + 4u] = y;
            batch.q[qBase + 5u] = z;
            batch.q[qBase + 6u] =
                std::sqrt(1.0f - x * x - y * y - z * z);
        }
        for (std::size_t rhsIndex = 0u;
             rhsIndex < rhsCount;
             ++rhsIndex) {
            const std::size_t rhsBase =
                (
                    environment * rhsCount + rhsIndex
                ) *
                articulation.nv;
            for (std::size_t dof = 0u;
                 dof < articulation.nv;
                 ++dof) {
                batch.rhs[rhsBase + dof] =
                    0.7f *
                        std::sin(
                            0.17f *
                            static_cast<float>(
                                (environment + 1u) *
                                (dof + 1u)
                            ) +
                            0.31f *
                            static_cast<float>(rhsIndex + 1u)
                        ) +
                    0.2f *
                        std::cos(
                            0.11f *
                            static_cast<float>(
                                (rhsIndex + 1u) *
                                (dof + 2u)
                            )
                        );
            }
        }
    }
    return batch;
}

void requireSuccess(
    const metalrobo::
        MetalArticulatedInverseMassDiagnostics& diagnostics,
    const std::string& operation
) {
    require(
        diagnostics.succeeded() &&
            diagnostics.dispatched &&
            diagnostics.published,
        operation + " failed: " +
            metalrobo::
                metalArticulatedInverseMassHostStatusName(
                    diagnostics.status
                ) +
            " " + diagnostics.message
    );
}

template <typename T>
bool byteEqual(
    const std::vector<T>& left,
    const std::vector<T>& right
) {
    return left.size() == right.size() &&
        (left.empty() ||
         std::memcmp(
             left.data(),
             right.data(),
             left.size() * sizeof(T)
         ) == 0);
}

bool samePayload(
    const metalrobo::MetalArticulatedInverseMassResult& left,
    const metalrobo::MetalArticulatedInverseMassResult& right
) {
    return byteEqual(left.output, right.output) &&
        byteEqual(left.statuses, right.statuses);
}

std::vector<double> solveRetainedCholesky(
    const std::span<const double> lower,
    const std::span<const float> rhs
) {
    const std::size_t dimension = rhs.size();
    require(
        lower.size() == dimension * dimension,
        "retained factor dimensions are wrong"
    );
    std::vector<double> intermediate(dimension);
    for (std::size_t row = 0u; row < dimension; ++row) {
        double value = rhs[row];
        for (std::size_t column = 0u;
             column < row;
             ++column) {
            value -=
                lower[row * dimension + column] *
                intermediate[column];
        }
        intermediate[row] =
            value / lower[row * dimension + row];
    }
    std::vector<double> output(dimension);
    for (std::size_t reverse = 0u;
         reverse < dimension;
         ++reverse) {
        const std::size_t row = dimension - 1u - reverse;
        double value = intermediate[row];
        for (std::size_t column = row + 1u;
             column < dimension;
             ++column) {
            value -=
                lower[column * dimension + row] *
                output[column];
        }
        output[row] =
            value / lower[row * dimension + row];
    }
    return output;
}

struct Parity {
    double scaledError = 0.0;
};

Parity compareCPU(
    const metalrobo::EngineModel& model,
    const Batch& batch,
    const metalrobo::MetalArticulatedInverseMassResult& gpu
) {
    const MRArticulationGPU& articulation =
        model.articulations[batch.articulationIndex];
    Parity parity{};
    for (std::size_t environment = 0u;
         environment < batch.environmentCount;
         ++environment) {
        const std::size_t qBase =
            environment * articulation.nq;
        std::vector<double> q(
            batch.q.begin() +
                static_cast<std::ptrdiff_t>(qBase),
            batch.q.begin() +
                static_cast<std::ptrdiff_t>(
                    qBase + articulation.nq
                )
        );
        std::vector<double> zeroVelocity(articulation.nv);
        metalrobo::ArticulatedContactProblem factor;
        const auto diagnostics =
            metalrobo::buildArticulatedContactProblem(
                model,
                batch.articulationIndex,
                q,
                zeroVelocity,
                std::span<
                    const metalrobo::ArticulatedContact
                >{},
                factor
            );
        require(
            diagnostics.succeeded(),
            "CPU retained-factor oracle failed"
        );
        for (std::size_t rhsIndex = 0u;
             rhsIndex < batch.rhsCount;
             ++rhsIndex) {
            const std::size_t base =
                (
                    environment * batch.rhsCount + rhsIndex
                ) *
                articulation.nv;
            const std::span<const float> rhs{
                batch.rhs.data() + base,
                articulation.nv,
            };
            const auto expected = solveRetainedCholesky(
                factor.massCholeskyLower,
                rhs
            );
            for (std::size_t dof = 0u;
                 dof < articulation.nv;
                 ++dof) {
                const double error = std::abs(
                    gpu.output[base + dof] - expected[dof]
                );
                parity.scaledError = std::max(
                    parity.scaledError,
                    error / (1.0 + std::abs(expected[dof]))
                );
            }
        }
    }
    return parity;
}

} // namespace

int main() {
    try {
        const auto config = testConfig();
        const metalrobo::EngineModel g1 =
            metalrobo::makeUnitreeG1EngineModel();
        const Batch small = makeBatch(g1, 0u, 3u, 3u);
        const Batch large = makeBatch(g1, 0u, 17u, 3u);
        metalrobo::MetalArticulatedInverseMassContext context{
            config
        };
        require(
            context.stats().pipelineCreationCount == 0u &&
                context.stats().bufferAllocationCount == 0u,
            "inverse-mass context performed eager Metal work"
        );

        metalrobo::MetalArticulatedInverseMassResult first;
        const auto firstDiagnostics =
            context.run(g1, small.input(), first);
        requireSuccess(
            firstDiagnostics,
            "first persistent inverse-mass run"
        );
        const Parity parity = compareCPU(g1, small, first);
        require(
            parity.scaledError < 5.0e-5,
            "public G1 inverse-mass parity failed"
        );
        const auto coldStats = context.stats();
        require(
            coldStats.pipelineCreationCount == 1u &&
                coldStats.bufferAllocationCount == 10u &&
                coldStats.bufferGrowthCount == 0u &&
                coldStats.submissionCount == 1u &&
                coldStats.completedSubmissionCount == 1u &&
                !coldStats.hasInFlightSubmission,
            "cold inverse-mass context counters are inconsistent"
        );

        metalrobo::MetalArticulatedInverseMassResult replay;
        requireSuccess(
            context.run(g1, small.input(), replay),
            "same-capacity inverse-mass replay"
        );
        require(
            context.stats().pipelineCreationCount == 1u &&
                context.stats().bufferAllocationCount ==
                    coldStats.bufferAllocationCount &&
                samePayload(first, replay),
            "inverse-mass replay rebuilt resources or changed output"
        );

        metalrobo::MetalArticulatedInverseMassResult oneRhs;
        metalrobo::MetalArticulatedInverseMassResult twoRhs;
        const Batch one = makeBatch(g1, 0u, 2u, 1u);
        const Batch two = makeBatch(g1, 0u, 2u, 2u);
        requireSuccess(
            context.run(g1, one.input(), oneRhs),
            "one-RHS inverse-mass run"
        );
        requireSuccess(
            context.run(g1, two.input(), twoRhs),
            "two-RHS inverse-mass run"
        );

        metalrobo::EngineModel asyncModel = g1;
        Batch asyncBatch = small;
        metalrobo::MetalArticulatedInverseMassSubmission pending;
        const auto submitted = context.submit(
            asyncModel,
            asyncBatch.input(),
            pending
        );
        require(
            submitted.succeeded() &&
                submitted.dispatched &&
                !submitted.published &&
                pending.valid() &&
                context.stats().hasInFlightSubmission,
            "asynchronous inverse-mass submit did not return "
            "a live ticket"
        );
        metalrobo::MetalArticulatedInverseMassSubmission rejected;
        const auto busy =
            context.submit(g1, small.input(), rejected);
        require(
            busy.status ==
                    metalrobo::
                        MetalArticulatedInverseMassHostStatus::
                            contextBusy &&
                !busy.dispatched &&
                !rejected.valid(),
            "shared inverse-mass arena admitted overlap"
        );
        std::fill(
            asyncBatch.q.begin(),
            asyncBatch.q.end(),
            std::numeric_limits<float>::quiet_NaN()
        );
        asyncBatch.rhs.clear();
        asyncModel.world.abiVersion = 0u;
        asyncModel.bodies.clear();
        metalrobo::MetalArticulatedInverseMassSubmission moved =
            std::move(pending);
        require(
            !pending.valid() && moved.valid(),
            "inverse-mass ticket move did not transfer ownership"
        );
        metalrobo::MetalArticulatedInverseMassResult asynchronous;
        requireSuccess(
            moved.wait(asynchronous),
            "asynchronous inverse-mass wait"
        );
        require(
            !moved.valid() && samePayload(first, asynchronous),
            "inverse-mass submission did not snapshot all inputs"
        );

        const auto completedBeforeDiscard =
            context.stats().completedSubmissionCount;
        {
            metalrobo::MetalArticulatedInverseMassSubmission
                discarded;
            const auto diagnostics = context.submit(
                g1,
                small.input(),
                discarded
            );
            require(
                diagnostics.succeeded() && discarded.valid(),
                "discarded inverse-mass ticket was not committed"
            );
        }
        require(
            !context.stats().hasInFlightSubmission &&
                context.stats().completedSubmissionCount ==
                    completedBeforeDiscard + 1u,
            "inverse-mass ticket destruction did not drain GPU"
        );

        metalrobo::MetalArticulatedInverseMassSubmission orphaned;
        {
            metalrobo::MetalArticulatedInverseMassContext transient{
                config
            };
            const auto diagnostics = transient.submit(
                g1,
                small.input(),
                orphaned
            );
            require(
                diagnostics.succeeded() && orphaned.valid(),
                "inverse-mass ticket did not retain its context"
            );
        }
        metalrobo::MetalArticulatedInverseMassResult orphanResult;
        requireSuccess(
            orphaned.wait(orphanResult),
            "inverse-mass wait after context destruction"
        );
        require(
            samePayload(first, orphanResult),
            "context-before-ticket destruction changed output"
        );

        metalrobo::MetalArticulatedInverseMassResult grown;
        requireSuccess(
            context.run(g1, large.input(), grown),
            "grown inverse-mass batch"
        );
        const auto grownStats = context.stats();
        require(
            grownStats.bufferGrowthCount > 0u &&
                grownStats.retainedBufferBytes >
                    coldStats.retainedBufferBytes,
            "larger inverse-mass batch did not grow arena"
        );
        const auto allocationsAfterGrowth =
            grownStats.bufferAllocationCount;
        metalrobo::MetalArticulatedInverseMassResult shrunk;
        requireSuccess(
            context.run(g1, small.input(), shrunk),
            "post-growth inverse-mass small batch"
        );
        require(
            context.stats().bufferAllocationCount ==
                allocationsAfterGrowth,
            "inverse-mass arena shrank or rebuilt"
        );

        metalrobo::MetalArticulatedInverseMassResult sentinel =
            first;
        Batch invalidDimensions = small;
        invalidDimensions.q.pop_back();
        const auto badDimensions = context.run(
            g1,
            invalidDimensions.input(),
            sentinel
        );
        require(
            badDimensions.status ==
                    metalrobo::
                        MetalArticulatedInverseMassHostStatus::
                            invalidDimensions &&
                samePayload(first, sentinel),
            "dimension rejection was not transactional"
        );
        Batch invalidCount = small;
        invalidCount.rhsCount = 4u;
        const auto badCount = context.run(
            g1,
            invalidCount.input(),
            sentinel
        );
        require(
            badCount.status ==
                    metalrobo::
                        MetalArticulatedInverseMassHostStatus::
                            invalidDimensions &&
                samePayload(first, sentinel),
            "four-RHS input escaped the host gate"
        );
        Batch nonfinite = small;
        nonfinite.rhs[7] =
            std::numeric_limits<float>::quiet_NaN();
        const auto badFinite = context.run(
            g1,
            nonfinite.input(),
            sentinel
        );
        require(
            badFinite.status ==
                    metalrobo::
                        MetalArticulatedInverseMassHostStatus::
                            nonfiniteInput &&
                samePayload(first, sentinel),
            "nonfinite rejection was not transactional"
        );

        auto unsupported = g1;
        unsupported.joints[0].jointType = MR_JOINT_PRISMATIC;
        std::string reason;
        require(
            unsupported.valid(&reason),
            "unsupported topology canary is invalid: " + reason
        );
        const auto unsupportedDiagnostics = context.run(
            unsupported,
            small.input(),
            sentinel
        );
        require(
            unsupportedDiagnostics.status ==
                    metalrobo::
                        MetalArticulatedInverseMassHostStatus::
                            unsupportedTopology &&
                samePayload(first, sentinel),
            "unsupported topology escaped host validation"
        );

        auto tinyPivot =
            metalrobo::makeFreeSphereEngineModel();
        MRBodyPropertiesGPU& tinyBody = tinyPivot.bodies[
            tinyPivot.articulations[0].rootBody
        ];
        tinyBody.massAndInverseMass =
            {1.0e-20f, 1.0e20f, 0.0f, 0.0f};
        tinyBody.inertiaRow0 =
            {1.0e-20f, 0.0f, 0.0f, 0.0f};
        tinyBody.inertiaRow1 =
            {0.0f, 1.0e-20f, 0.0f, 0.0f};
        tinyBody.inertiaRow2 =
            {0.0f, 0.0f, 1.0e-20f, 0.0f};
        tinyBody.inverseInertiaRow0 =
            {1.0e20f, 0.0f, 0.0f, 0.0f};
        tinyBody.inverseInertiaRow1 =
            {0.0f, 1.0e20f, 0.0f, 0.0f};
        tinyBody.inverseInertiaRow2 =
            {0.0f, 0.0f, 1.0e20f, 0.0f};
        require(
            tinyPivot.valid(&reason),
            "tiny-pivot model is invalid: " + reason
        );
        const Batch tinyBatch =
            makeBatch(tinyPivot, 0u, 1u, 3u);
        metalrobo::MetalArticulatedInverseMassResult failedGPU;
        const auto gpuFailure = context.run(
            tinyPivot,
            tinyBatch.input(),
            failedGPU
        );
        require(
            gpuFailure.status ==
                    metalrobo::
                        MetalArticulatedInverseMassHostStatus::
                            gpuEnvironmentFailure &&
                gpuFailure.published &&
                failedGPU.statuses.size() == 1u &&
                failedGPU.statuses[0].code ==
                    MR_INVERSE_MASS_FACTORIZATION_FAILED &&
                std::all_of(
                    failedGPU.output.begin(),
                    failedGPU.output.end(),
                    [](const float value) {
                        return value == 0.0f;
                    }
                ),
            "GPU failure did not publish one atomic zero result"
        );

        constexpr std::size_t kThroughputEnvironments = 1024u;
        const Batch throughput = makeBatch(
            g1,
            0u,
            kThroughputEnvironments,
            3u
        );
        metalrobo::MetalArticulatedInverseMassResult warmup;
        requireSuccess(
            context.run(g1, throughput.input(), warmup),
            "inverse-mass throughput warmup"
        );
        metalrobo::MetalArticulatedInverseMassResult measured;
        const auto wallStart = std::chrono::steady_clock::now();
        const auto throughputDiagnostics = context.run(
            g1,
            throughput.input(),
            measured
        );
        const auto wallEnd = std::chrono::steady_clock::now();
        requireSuccess(
            throughputDiagnostics,
            "warm inverse-mass throughput"
        );
        const double wallMilliseconds =
            std::chrono::duration<double, std::milli>(
                wallEnd - wallStart
            ).count();
        const double gpuEnvironmentActionsPerSecond =
            1000.0 *
            static_cast<double>(kThroughputEnvironments) /
            throughputDiagnostics.elapsedMilliseconds;
        const double wallEnvironmentActionsPerSecond =
            1000.0 *
            static_cast<double>(kThroughputEnvironments) /
            wallMilliseconds;

        std::cout
            << "articulated_inverse_mass_host=metal"
            << " device=\"" << firstDiagnostics.deviceName << "\""
            << " environments=" << small.environmentCount
            << " rhs=3"
            << " dofs=" << g1.articulations[0].nv
            << " scaled_error=" << parity.scaledError
            << " pipeline_creations="
            << context.stats().pipelineCreationCount
            << " buffer_growths="
            << context.stats().bufferGrowthCount
            << " retained_bytes="
            << context.stats().retainedBufferBytes
            << " warm_batch=" << kThroughputEnvironments
            << " gpu_environment_actions_per_s="
            << gpuEnvironmentActionsPerSecond
            << " gpu_rhs_actions_per_s="
            << 3.0 * gpuEnvironmentActionsPerSecond
            << " wall_environment_actions_per_s="
            << wallEnvironmentActionsPerSecond
            << " replay=bitwise"
            << " rhs_1_to_3=pass"
            << " async=pass"
            << " busy_gate=pass"
            << " input_snapshot=pass"
            << " discard_drain=pass"
            << " orphan_lifetime=pass"
            << " grow_only=pass"
            << " host_failures=pass"
            << " gpu_failure_transaction=pass"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "articulated_inverse_mass_host=failed reason=\""
            << exception.what() << "\"\n";
        return 1;
    }
}
