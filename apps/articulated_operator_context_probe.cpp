#include "metalrobo/G1.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

MRArticulatedPointImpulseGPU makePoint(
    const std::uint32_t bodyIndex
) {
    MRArticulatedPointImpulseGPU point{};
    point.bodyIndex = bodyIndex;
    point.localPoint = {0.01f, -0.02f, 0.03f, 0.0f};
    point.worldImpulse = {0.4f, -0.1f, 0.7f, 0.0f};
    return point;
}

struct Batch {
    std::size_t environmentCount = 0u;
    std::size_t pointCount = 0u;
    std::vector<float> q;
    std::vector<MRArticulatedPointImpulseGPU> points;

    [[nodiscard]] metalrobo::MetalArticulatedOperatorInput input()
        const {
        return {
            .articulationIndex = 0u,
            .environmentCount = environmentCount,
            .pointCount = pointCount,
            .q = q,
            .points = points,
        };
    }
};

Batch makeBatch(
    const metalrobo::EngineModel& model,
    const std::size_t environmentCount,
    const std::size_t pointCount
) {
    const MRArticulationGPU& articulation =
        model.articulations[0];
    Batch batch{
        .environmentCount = environmentCount,
        .pointCount = pointCount,
    };
    batch.q.resize(environmentCount * articulation.nq);
    batch.points.reserve(environmentCount * pointCount);
    for (std::size_t environment = 0u;
         environment < environmentCount;
         ++environment) {
        std::copy_n(
            model.defaultQ.begin() + articulation.qOffset,
            articulation.nq,
            batch.q.begin() + environment * articulation.nq
        );
        if (articulation.nq > 7u) {
            batch.q[environment * articulation.nq + 7u] +=
                static_cast<float>(environment) * 0.001f;
        }
        for (std::size_t pointIndex = 0u;
             pointIndex < pointCount;
             ++pointIndex) {
            batch.points.push_back(makePoint(
                articulation.firstBody +
                1u +
                static_cast<std::uint32_t>(
                    pointIndex %
                    (articulation.bodyCount - 1u)
                )
            ));
        }
    }
    return batch;
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
    const metalrobo::MetalArticulatedOperatorResult& left,
    const metalrobo::MetalArticulatedOperatorResult& right
) {
    return byteEqual(left.bodyPoses, right.bodyPoses) &&
        byteEqual(left.pointWorld, right.pointWorld) &&
        byteEqual(left.diagnosticMassMatrix,
                  right.diagnosticMassMatrix) &&
        byteEqual(left.pointJacobians, right.pointJacobians) &&
        byteEqual(left.generalizedImpulse,
                  right.generalizedImpulse) &&
        byteEqual(left.deltaVelocity, right.deltaVelocity) &&
        byteEqual(left.statuses, right.statuses) &&
        byteEqual(left.millardResults, right.millardResults) &&
        byteEqual(left.millardGeneralizedForces,
                  right.millardGeneralizedForces) &&
        byteEqual(left.mujocoResults, right.mujocoResults) &&
        byteEqual(left.mujocoActivationStates,
                  right.mujocoActivationStates) &&
        byteEqual(left.mujocoMuscleGeneralizedForces,
                  right.mujocoMuscleGeneralizedForces) &&
        byteEqual(left.mujocoGeneralizedForces,
                  right.mujocoGeneralizedForces) &&
        byteEqual(left.standQ, right.standQ) &&
        byteEqual(left.standV, right.standV) &&
        byteEqual(left.standStatuses, right.standStatuses) &&
        byteEqual(left.standTendonTransfers, right.standTendonTransfers) &&
        byteEqual(
            left.standTendonGeneralizedCorrections,
            right.standTendonGeneralizedCorrections
        );
}

void requireSuccess(
    const metalrobo::MetalArticulatedOperatorDiagnostics& diagnostics,
    const std::string& operation
) {
    require(
        diagnostics.succeeded() &&
            diagnostics.dispatched &&
            diagnostics.published,
        operation + " failed: " +
            metalrobo::metalArticulatedOperatorHostStatusName(
                diagnostics.status
            ) +
            " " + diagnostics.message
    );
}

} // namespace

int main() {
    try {
        const metalrobo::EngineModel model =
            metalrobo::makeUnitreeG1EngineModel();
        const Batch small = makeBatch(model, 1u, 1u);
        const Batch large = makeBatch(model, 8u, 3u);
        metalrobo::MetalArticulatedOperatorContext context;

        const auto coldStats = context.stats();
        require(
            coldStats.pipelineCreationCount == 0u &&
                coldStats.bufferAllocationCount == 0u,
            "context performed eager Metal work"
        );

        metalrobo::MetalArticulatedOperatorResult first;
        requireSuccess(
            context.run(model, small.input(), first),
            "first persistent run"
        );
        const auto warmStats = context.stats();
        require(
            warmStats.pipelineCreationCount == 1u &&
                // The cold fixed binding arena retains the 16 generic slots,
                // eight Millard slots, seven MyoSim route-force slots, and
                // one stand-velocity placeholder. The remaining stand arena
                // is allocated only when a persistent-Human horizon exists.
                // Empty source programs deliberately retain one typed
                // placeholder apiece so every dispatched ABI slot is bound.
                warmStats.bufferAllocationCount == 32u &&
                warmStats.bufferGrowthCount == 0u &&
                warmStats.submissionCount == 1u &&
                warmStats.completedSubmissionCount == 1u &&
                !warmStats.hasInFlightSubmission,
            "cold context counters are inconsistent: pipelines=" +
                std::to_string(warmStats.pipelineCreationCount) +
                " allocations=" +
                std::to_string(warmStats.bufferAllocationCount) +
                " growths=" +
                std::to_string(warmStats.bufferGrowthCount) +
                " submissions=" +
                std::to_string(warmStats.submissionCount) +
                " completions=" +
                std::to_string(warmStats.completedSubmissionCount) +
                " in_flight=" +
                std::to_string(warmStats.hasInFlightSubmission)
        );

        metalrobo::MetalArticulatedOperatorResult replay;
        requireSuccess(
            context.run(model, small.input(), replay),
            "same-capacity persistent replay"
        );
        const auto replayStats = context.stats();
        require(
            replayStats.pipelineCreationCount == 1u &&
                replayStats.bufferAllocationCount ==
                    warmStats.bufferAllocationCount &&
                replayStats.bufferGrowthCount ==
                    warmStats.bufferGrowthCount &&
                samePayload(first, replay),
            "same-capacity run rebuilt resources or changed payload"
        );

        metalrobo::MetalArticulatedOperatorSubmission pending;
        const auto submitted =
            context.submit(model, large.input(), pending);
        require(
            submitted.succeeded() &&
                submitted.dispatched &&
                !submitted.published &&
                pending.valid(),
            "asynchronous submit did not return a live ticket"
        );
        const auto inFlightStats = context.stats();
        require(
            inFlightStats.hasInFlightSubmission &&
                inFlightStats.submissionCount == 3u,
            "in-flight context state was not published"
        );

        metalrobo::MetalArticulatedOperatorSubmission rejected;
        const auto busy =
            context.submit(model, small.input(), rejected);
        require(
            busy.status ==
                metalrobo::MetalArticulatedOperatorHostStatus::
                    contextBusy &&
                !busy.dispatched &&
                !rejected.valid(),
            "shared arena admitted overlapping submissions"
        );

        metalrobo::MetalArticulatedOperatorResult largeResult;
        requireSuccess(
            pending.wait(largeResult),
            "asynchronous wait"
        );
        require(!pending.valid(), "wait did not consume ticket");
        const auto grownStats = context.stats();
        require(
            !grownStats.hasInFlightSubmission &&
                grownStats.pipelineCreationCount == 1u &&
                grownStats.bufferGrowthCount > 0u &&
                grownStats.bufferAllocationCount > 15u &&
                grownStats.completedSubmissionCount == 3u &&
                grownStats.retainedBufferBytes >
                    warmStats.retainedBufferBytes,
            "larger batch did not grow and retain the arena"
        );

        const auto allocationsBeforeShrink =
            grownStats.bufferAllocationCount;
        metalrobo::MetalArticulatedOperatorResult shrunk;
        requireSuccess(
            context.run(model, small.input(), shrunk),
            "post-growth small run"
        );
        require(
            context.stats().bufferAllocationCount ==
                allocationsBeforeShrink,
            "smaller batch shrank or rebuilt the arena"
        );

        const auto completedBeforeDiscard =
            context.stats().completedSubmissionCount;
        {
            metalrobo::MetalArticulatedOperatorSubmission discarded;
            const auto discardSubmit =
                context.submit(model, small.input(), discarded);
            require(
                discardSubmit.succeeded() && discarded.valid(),
                "discarded submission was not committed"
            );
        }
        const auto discardedStats = context.stats();
        require(
            !discardedStats.hasInFlightSubmission &&
                discardedStats.completedSubmissionCount ==
                    completedBeforeDiscard + 1u,
            "ticket destruction did not safely drain the GPU"
        );

        metalrobo::MetalArticulatedOperatorResult finalResult;
        const auto finalDiagnostics =
            context.run(model, small.input(), finalResult);
        requireSuccess(finalDiagnostics, "post-discard run");

        metalrobo::MetalArticulatedOperatorSubmission orphaned;
        {
            metalrobo::MetalArticulatedOperatorContext transient;
            const auto orphanSubmit =
                transient.submit(model, small.input(), orphaned);
            require(
                orphanSubmit.succeeded() && orphaned.valid(),
                "submission did not retain its execution context"
            );
        }
        metalrobo::MetalArticulatedOperatorResult orphanResult;
        requireSuccess(
            orphaned.wait(orphanResult),
            "wait after context destruction"
        );

        std::cout
            << "articulated_operator_context=metal"
            << " device=\""
            << finalDiagnostics.deviceName
            << "\""
            << " pipeline_creations="
            << context.stats().pipelineCreationCount
            << " buffer_allocations="
            << context.stats().bufferAllocationCount
            << " buffer_growths="
            << context.stats().bufferGrowthCount
            << " retained_bytes="
            << context.stats().retainedBufferBytes
            << " async=pass"
            << " busy_gate=pass"
            << " discard_drain=pass"
            << " orphan_lifetime=pass"
            << " replay=bitwise"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "articulated_operator_context status=failed error=\""
            << exception.what() << "\"\n";
        return 1;
    }
}
