#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/EngineModelComposer.hpp"
#include "metalrobo/G1.hpp"
#include "metalrobo/MetalArticulatedABA.hpp"

#include <algorithm>
#include <array>
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

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

metalrobo::EngineModel duplicateModel(
    const metalrobo::EngineModel& source
) {
    const std::array<metalrobo::EngineModelComponent, 2u> components{{
        {&source, "first"},
        {&source, "second"},
    }};
    metalrobo::EngineModelComposeConfig config;
    config.name = source.name + "_duplicated";
    config.gravityAndTimestep = source.world.gravityAndTimestep;
    config.solverScales = source.world.solverScales;
    config.solverType = source.world.solverType;
    config.frictionConeType = source.world.frictionConeType;
    metalrobo::EngineModel combined;
    const auto diagnostics = metalrobo::composeEngineModels(
        components,
        combined,
        config
    );
    require(
        diagnostics.succeeded(),
        "offset model composition failed: " + diagnostics.message
    );
    return combined;
}

struct Batch {
    std::uint32_t articulationIndex = 0u;
    std::size_t environmentCount = 0u;
    std::vector<float> q;
    std::vector<float> v;
    std::vector<float> effort;
    std::vector<MRABABodyWrenchGPU> wrenches;
    bool applyBodyDamping = true;

    [[nodiscard]] metalrobo::MetalArticulatedABAInput input()
        const {
        return {
            .articulationIndex = articulationIndex,
            .environmentCount = environmentCount,
            .q = q,
            .v = v,
            .effort = effort,
            .bodyWrenches = wrenches,
            .applyBodyDamping = applyBodyDamping,
        };
    }
};

Batch makeBatch(
    const metalrobo::EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::size_t environmentCount,
    const bool withWrenches
) {
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    Batch batch{
        .articulationIndex = articulationIndex,
        .environmentCount = environmentCount,
    };
    batch.q.resize(environmentCount * articulation.nq);
    batch.v.resize(environmentCount * articulation.nv);
    batch.effort.resize(environmentCount * articulation.nv);
    if (withWrenches) {
        batch.wrenches.resize(
            environmentCount * articulation.bodyCount
        );
    }

    for (std::size_t environment = 0u;
         environment < environmentCount;
         ++environment) {
        std::copy_n(
            model.defaultQ.begin() + articulation.qOffset,
            articulation.nq,
            batch.q.begin() + environment * articulation.nq
        );
        std::copy_n(
            model.defaultV.begin() + articulation.vOffset,
            articulation.nv,
            batch.v.begin() + environment * articulation.nv
        );
        const std::size_t qBase =
            environment * articulation.nq;
        const std::size_t vBase =
            environment * articulation.nv;
        const std::size_t stateVariant = environment % 7u;
        if (articulation.rootType == MR_ROOT_FLOATING) {
            batch.q[qBase + 0u] +=
                0.015f * static_cast<float>(stateVariant);
            const float x =
                0.006f * static_cast<float>(stateVariant);
            const float y =
                -0.004f * static_cast<float>(stateVariant);
            const float z =
                0.003f * static_cast<float>(stateVariant);
            const float w =
                std::sqrt(1.0f - x * x - y * y - z * z);
            batch.q[qBase + 3u] = x;
            batch.q[qBase + 4u] = y;
            batch.q[qBase + 5u] = z;
            batch.q[qBase + 6u] = w;
        }
        for (std::size_t dof = 0u;
             dof < articulation.nv;
             ++dof) {
            batch.v[vBase + dof] =
                0.025f * std::sin(
                    static_cast<float>(
                        1u + dof + 3u * environment
                    ) *
                    0.17f
                );
            batch.effort[vBase + dof] =
                0.35f * std::cos(
                    static_cast<float>(
                        2u + dof + 5u * environment
                    ) *
                    0.11f
                );
        }
        if (withWrenches) {
            MRABABodyWrenchGPU& rootWrench =
                batch.wrenches[
                    environment * articulation.bodyCount
                ];
            rootWrench.force = {
                0.2f,
                -0.1f,
                0.35f +
                    0.05f * static_cast<float>(environment),
                0.0f,
            };
            rootWrench.torque = {
                0.01f,
                -0.015f,
                0.02f,
                0.0f,
            };
        }
    }
    return batch;
}

void requireSuccess(
    const metalrobo::MetalArticulatedABADiagnostics& diagnostics,
    const std::string& operation
) {
    require(
        diagnostics.succeeded() &&
            diagnostics.dispatched &&
            diagnostics.published,
        operation + " failed: " +
            metalrobo::metalArticulatedABAHostStatusName(
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
    const metalrobo::MetalArticulatedABAResult& left,
    const metalrobo::MetalArticulatedABAResult& right
) {
    return byteEqual(left.acceleration, right.acceleration) &&
        byteEqual(left.nextV, right.nextV) &&
        byteEqual(left.nextQ, right.nextQ) &&
        byteEqual(left.statuses, right.statuses);
}

struct Parity {
    double accelerationScaled = 0.0;
    double nextV = 0.0;
    double nextQ = 0.0;
};

Parity compareCPU(
    const metalrobo::EngineModel& model,
    const Batch& batch,
    const metalrobo::MetalArticulatedABAResult& gpu
) {
    const MRArticulationGPU& articulation =
        model.articulations[batch.articulationIndex];
    Parity parity{};
    for (std::size_t environment = 0u;
         environment < batch.environmentCount;
         ++environment) {
        const std::size_t qBase =
            environment * articulation.nq;
        const std::size_t vBase =
            environment * articulation.nv;
        const std::size_t wrenchBase =
            environment * articulation.bodyCount;
        std::vector<double> q(
            batch.q.begin() +
                static_cast<std::ptrdiff_t>(qBase),
            batch.q.begin() +
                static_cast<std::ptrdiff_t>(
                    qBase + articulation.nq
                )
        );
        std::vector<double> v(
            batch.v.begin() +
                static_cast<std::ptrdiff_t>(vBase),
            batch.v.begin() +
                static_cast<std::ptrdiff_t>(
                    vBase + articulation.nv
                )
        );
        std::vector<double> effort(
            batch.effort.begin() +
                static_cast<std::ptrdiff_t>(vBase),
            batch.effort.begin() +
                static_cast<std::ptrdiff_t>(
                    vBase + articulation.nv
                )
        );
        std::vector<metalrobo::ArticulatedBodyWrench>
            globalWrenches;
        if (!batch.wrenches.empty()) {
            globalWrenches.resize(model.bodies.size());
            for (std::size_t localBody = 0u;
                 localBody < articulation.bodyCount;
                 ++localBody) {
                const MRABABodyWrenchGPU& input =
                    batch.wrenches[wrenchBase + localBody];
                auto& output = globalWrenches[
                    articulation.firstBody + localBody
                ];
                output.force = {
                    input.force.x,
                    input.force.y,
                    input.force.z,
                };
                output.torque = {
                    input.torque.x,
                    input.torque.y,
                    input.torque.z,
                };
            }
        }

        metalrobo::ArticulatedDynamicsConfig config;
        config.gravity = {
            model.world.gravityAndTimestep.x,
            model.world.gravityAndTimestep.y,
            model.world.gravityAndTimestep.z,
        };
        config.timestep = model.world.gravityAndTimestep.w;
        config.applyBodyDamping = batch.applyBodyDamping;

        std::vector<double> acceleration(articulation.nv);
        auto diagnostics =
            metalrobo::computeArticulatedForwardDynamics(
                model,
                batch.articulationIndex,
                q,
                v,
                effort,
                globalWrenches,
                acceleration,
                config
            );
        require(
            diagnostics.succeeded(),
            "CPU forward-dynamics oracle failed"
        );
        for (std::size_t dof = 0u;
             dof < articulation.nv;
             ++dof) {
            const double error = std::abs(
                gpu.acceleration[vBase + dof] -
                acceleration[dof]
            );
            parity.accelerationScaled = std::max(
                parity.accelerationScaled,
                error / (1.0 + std::abs(acceleration[dof]))
            );
        }

        diagnostics = metalrobo::integrateArticulatedState(
            model,
            batch.articulationIndex,
            q,
            v,
            effort,
            globalWrenches,
            config
        );
        require(
            diagnostics.succeeded(),
            "CPU symplectic integration oracle failed"
        );
        for (std::size_t dof = 0u;
             dof < articulation.nv;
             ++dof) {
            parity.nextV = std::max(
                parity.nextV,
                std::abs(gpu.nextV[vBase + dof] - v[dof])
            );
        }
        for (std::size_t coordinate = 0u;
             coordinate < articulation.nq;
             ++coordinate) {
            parity.nextQ = std::max(
                parity.nextQ,
                std::abs(
                    gpu.nextQ[qBase + coordinate] -
                    q[coordinate]
                )
            );
        }
    }
    return parity;
}

void requireParity(const Parity& parity) {
    require(
        parity.accelerationScaled < 5.0e-5 &&
            parity.nextV < 1.0e-4 &&
            parity.nextQ < 5.0e-5,
        "G1 CPU parity failed"
    );
}

} // namespace

int main() {
    try {
        const metalrobo::EngineModel g1 =
            metalrobo::makeUnitreeG1EngineModel();
        const Batch small = makeBatch(g1, 0u, 3u, true);
        const Batch large = makeBatch(g1, 0u, 17u, true);
        metalrobo::MetalArticulatedABAContext context;
        require(
            context.stats().pipelineCreationCount == 0u &&
                context.stats().bufferAllocationCount == 0u,
            "ABA context performed eager Metal work"
        );

        metalrobo::MetalArticulatedABAResult first;
        const auto firstDiagnostics =
            context.run(g1, small.input(), first);
        requireSuccess(firstDiagnostics, "first persistent ABA run");
        const Parity parity = compareCPU(g1, small, first);
        requireParity(parity);
        const auto warmStats = context.stats();
        require(
            warmStats.pipelineCreationCount == 1u &&
                warmStats.bufferAllocationCount == 14u &&
                warmStats.bufferGrowthCount == 0u &&
                warmStats.submissionCount == 1u &&
                warmStats.completedSubmissionCount == 1u &&
                !warmStats.hasInFlightSubmission,
            "cold ABA context counters are inconsistent"
        );

        metalrobo::MetalArticulatedABAResult replay;
        requireSuccess(
            context.run(g1, small.input(), replay),
            "same-capacity ABA replay"
        );
        const auto replayStats = context.stats();
        require(
            replayStats.pipelineCreationCount == 1u &&
                replayStats.bufferAllocationCount ==
                    warmStats.bufferAllocationCount &&
                samePayload(first, replay),
            "ABA replay rebuilt resources or changed payload"
        );

        Batch asyncBatch = small;
        metalrobo::MetalArticulatedABASubmission pending;
        const auto submitted =
            context.submit(g1, asyncBatch.input(), pending);
        require(
            submitted.succeeded() &&
                submitted.dispatched &&
                !submitted.published &&
                pending.valid(),
            "asynchronous ABA submit did not return a live ticket"
        );
        const auto inFlightStats = context.stats();
        require(
            inFlightStats.hasInFlightSubmission &&
                inFlightStats.submissionCount == 3u,
            "ABA in-flight state was not published"
        );
        metalrobo::MetalArticulatedABASubmission rejected;
        const auto busy =
            context.submit(g1, small.input(), rejected);
        require(
            busy.status ==
                    metalrobo::MetalArticulatedABAHostStatus::
                        contextBusy &&
                !busy.dispatched &&
                !rejected.valid(),
            "shared ABA arena admitted overlapping submissions"
        );
        std::fill(
            asyncBatch.q.begin(),
            asyncBatch.q.end(),
            std::numeric_limits<float>::quiet_NaN()
        );
        std::fill(
            asyncBatch.v.begin(),
            asyncBatch.v.end(),
            std::numeric_limits<float>::quiet_NaN()
        );
        asyncBatch.effort.clear();
        asyncBatch.wrenches.clear();
        metalrobo::MetalArticulatedABASubmission moved =
            std::move(pending);
        require(
            !pending.valid() && moved.valid(),
            "ABA ticket move did not transfer ownership"
        );
        metalrobo::MetalArticulatedABAResult asynchronous;
        requireSuccess(
            moved.wait(asynchronous),
            "asynchronous ABA wait"
        );
        require(
            !moved.valid() && samePayload(first, asynchronous),
            "asynchronous ABA did not own its submitted input snapshot"
        );
        const auto completedAsyncStats = context.stats();
        require(
            !completedAsyncStats.hasInFlightSubmission &&
                completedAsyncStats.completedSubmissionCount == 3u,
            "ABA wait did not release the shared arena"
        );

        const auto completedBeforeDiscard =
            context.stats().completedSubmissionCount;
        {
            metalrobo::MetalArticulatedABASubmission discarded;
            const auto discardSubmit =
                context.submit(g1, small.input(), discarded);
            require(
                discardSubmit.succeeded() && discarded.valid(),
                "discarded ABA submission was not committed"
            );
        }
        const auto discardedStats = context.stats();
        require(
            !discardedStats.hasInFlightSubmission &&
                discardedStats.completedSubmissionCount ==
                    completedBeforeDiscard + 1u,
            "ABA ticket destruction did not drain the GPU"
        );

        metalrobo::MetalArticulatedABASubmission orphaned;
        {
            metalrobo::MetalArticulatedABAContext transient;
            const auto orphanSubmit =
                transient.submit(g1, small.input(), orphaned);
            require(
                orphanSubmit.succeeded() && orphaned.valid(),
                "ABA ticket did not retain its context"
            );
        }
        metalrobo::MetalArticulatedABAResult orphanResult;
        requireSuccess(
            orphaned.wait(orphanResult),
            "ABA wait after context destruction"
        );
        require(
            samePayload(first, orphanResult),
            "context-before-ticket destruction changed ABA output"
        );

        metalrobo::MetalArticulatedABAResult grown;
        requireSuccess(
            context.run(g1, large.input(), grown),
            "grown ABA batch"
        );
        const auto grownStats = context.stats();
        require(
            grownStats.bufferGrowthCount > 0u &&
                grownStats.bufferAllocationCount >
                    warmStats.bufferAllocationCount &&
                grownStats.retainedBufferBytes >
                    warmStats.retainedBufferBytes,
            "larger ABA batch did not grow the persistent arena"
        );
        const auto allocationsAfterGrowth =
            grownStats.bufferAllocationCount;
        metalrobo::MetalArticulatedABAResult shrunk;
        requireSuccess(
            context.run(g1, small.input(), shrunk),
            "post-growth ABA small batch"
        );
        require(
            context.stats().bufferAllocationCount ==
                allocationsAfterGrowth,
            "smaller ABA batch shrank or rebuilt the arena"
        );

        const metalrobo::EngineModel duplicated =
            duplicateModel(g1);
        std::string reason;
        require(
            duplicated.valid(&reason),
            "offset model is invalid: " + reason
        );
        const Batch offsetBatch =
            makeBatch(duplicated, 1u, 3u, true);
        metalrobo::MetalArticulatedABAResult offsetResult;
        requireSuccess(
            context.run(
                duplicated,
                offsetBatch.input(),
                offsetResult
            ),
            "nonzero-offset ABA run"
        );
        require(
            byteEqual(first.acceleration, offsetResult.acceleration) &&
                byteEqual(first.nextV, offsetResult.nextV) &&
                byteEqual(first.nextQ, offsetResult.nextQ),
            "nonzero articulation/body/joint/q/v offsets changed ABA"
        );

        metalrobo::MetalArticulatedABAResult sentinel = first;
        Batch invalidDimensions = small;
        invalidDimensions.q.pop_back();
        const auto badDimensions = context.run(
            g1,
            invalidDimensions.input(),
            sentinel
        );
        require(
            badDimensions.status ==
                    metalrobo::MetalArticulatedABAHostStatus::
                        invalidDimensions &&
                samePayload(first, sentinel),
            "ABA dimension rejection was not transactional"
        );
        Batch nonfinite = small;
        nonfinite.q[0] =
            std::numeric_limits<float>::quiet_NaN();
        const auto badFinite =
            context.run(g1, nonfinite.input(), sentinel);
        require(
            badFinite.status ==
                    metalrobo::MetalArticulatedABAHostStatus::
                        nonfiniteInput &&
                samePayload(first, sentinel),
            "ABA nonfinite rejection was not transactional"
        );
        Batch badWrench = small;
        badWrench.wrenches[0].force.w = 1.0f;
        const auto badWrenchDiagnostics =
            context.run(g1, badWrench.input(), sentinel);
        require(
            badWrenchDiagnostics.status ==
                    metalrobo::MetalArticulatedABAHostStatus::
                        invalidBodyWrench &&
                samePayload(first, sentinel),
            "ABA wrench rejection was not transactional"
        );

        auto prismatic = g1;
        prismatic.joints[0].jointType = MR_JOINT_PRISMATIC;
        require(
            prismatic.valid(&reason),
            "prismatic topology canary is not a valid model: " +
                reason
        );
        metalrobo::MetalArticulatedABAResult prismaticResult;
        const auto prismaticDiagnostics =
            context.run(prismatic, small.input(), prismaticResult);
        require(
            prismaticDiagnostics.succeeded() &&
                prismaticResult.acceleration.size() ==
                    small.environmentCount * g1.world.nv,
            "prismatic ABA topology did not execute"
        );

        auto tinyPivot =
            metalrobo::makeFreeSphereEngineModel();
        MRBodyPropertiesGPU& tinyBody = tinyPivot.bodies[
            tinyPivot.articulations[0].rootBody
        ];
        tinyBody.massAndInverseMass = {
            1.0e-20f,
            1.0e20f,
            0.0f,
            0.0f,
        };
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
            "tiny-pivot canary model is invalid: " + reason
        );
        const Batch tinyBatch =
            makeBatch(tinyPivot, 0u, 1u, false);
        metalrobo::MetalArticulatedABAResult failedGPU;
        const auto gpuFailure =
            context.run(tinyPivot, tinyBatch.input(), failedGPU);
        require(
            gpuFailure.status ==
                    metalrobo::MetalArticulatedABAHostStatus::
                        gpuEnvironmentFailure &&
                gpuFailure.published &&
                failedGPU.statuses.size() == 1u &&
                failedGPU.statuses[0].code ==
                    MR_ABA_FACTORIZATION_FAILED &&
                std::all_of(
                    failedGPU.acceleration.begin(),
                    failedGPU.acceleration.end(),
                    [](const float value) {
                        return value == 0.0f;
                    }
                ) &&
                std::all_of(
                    failedGPU.nextV.begin(),
                    failedGPU.nextV.end(),
                    [](const float value) {
                        return value == 0.0f;
                    }
                ) &&
                std::all_of(
                    failedGPU.nextQ.begin(),
                    failedGPU.nextQ.end(),
                    [](const float value) {
                        return value == 0.0f;
                    }
                ),
            "GPU ABA failure did not publish one atomic zero state"
        );

        constexpr std::size_t throughputEnvironmentCount = 1024u;
        const Batch throughputBatch = makeBatch(
            g1,
            0u,
            throughputEnvironmentCount,
            false
        );
        metalrobo::MetalArticulatedABAResult throughputWarmup;
        requireSuccess(
            context.run(
                g1,
                throughputBatch.input(),
                throughputWarmup
            ),
            "ABA throughput warmup"
        );
        metalrobo::MetalArticulatedABAResult throughputResult;
        const auto wallStart = std::chrono::steady_clock::now();
        const auto throughputDiagnostics = context.run(
            g1,
            throughputBatch.input(),
            throughputResult
        );
        const auto wallEnd = std::chrono::steady_clock::now();
        requireSuccess(
            throughputDiagnostics,
            "warm ABA throughput run"
        );
        const double wallMilliseconds =
            std::chrono::duration<double, std::milli>(
                wallEnd - wallStart
            ).count();
        const double gpuEnvironmentsPerSecond =
            1000.0 *
            static_cast<double>(throughputEnvironmentCount) /
            throughputDiagnostics.elapsedMilliseconds;
        const double wallEnvironmentsPerSecond =
            1000.0 *
            static_cast<double>(throughputEnvironmentCount) /
            wallMilliseconds;

        std::cout
            << "articulated_aba_host=metal"
            << " device=\"" << firstDiagnostics.deviceName << "\""
            << " environments=" << small.environmentCount
            << " dofs=" << g1.articulations[0].nv
            << " acceleration_scaled_error="
            << parity.accelerationScaled
            << " next_v_error=" << parity.nextV
            << " next_q_error=" << parity.nextQ
            << " pipeline_creations="
            << context.stats().pipelineCreationCount
            << " buffer_growths="
            << context.stats().bufferGrowthCount
            << " retained_bytes="
            << context.stats().retainedBufferBytes
            << " warm_batch=" << throughputEnvironmentCount
            << " gpu_environments_per_s="
            << gpuEnvironmentsPerSecond
            << " wall_environments_per_s="
            << wallEnvironmentsPerSecond
            << " replay=bitwise"
            << " async=pass"
            << " busy_gate=pass"
            << " input_snapshot=pass"
            << " discard_drain=pass"
            << " orphan_lifetime=pass"
            << " offsets=pass"
            << " host_failures=pass"
            << " gpu_failure_transaction=pass"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "articulated_aba_host=failed reason=\""
            << exception.what() << "\"\n";
        return 1;
    }
}
