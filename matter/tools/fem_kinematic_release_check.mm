#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace numi::matter;

namespace {

constexpr std::size_t kEnvironmentCount = 2u;
constexpr std::size_t kFEMNodeCount = 4u;
constexpr std::size_t kTargetCount = kEnvironmentCount * kFEMNodeCount;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
id<MTLBuffer> makeSharedBuffer(
    id<MTLDevice> device,
    const std::vector<T>& values
) {
    require(!values.empty(), "cannot create an empty Metal buffer");
    return [device newBufferWithBytes:values.data()
                               length:values.size() * sizeof(T)
                              options:MTLResourceStorageModeShared];
}

struct StepResult {
    RuntimeStateSnapshot snapshot;
    std::vector<nm_float4> femMechanicalResidual;
};

struct Fixture {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = nil;
    Runtime runtime;
    CompiledWorld world;

    explicit Fixture(const bool originallyFixed = true) {
        const bool isAppleDevice =
            device != nil &&
            [[device name] rangeOfString:@"Apple"].location != NSNotFound &&
            [[device name] rangeOfString:@"Paravirtual"].location == NSNotFound;
        require(isAppleDevice, "physical Apple device required");

        queue = [device newCommandQueue];
        require(queue != nil, "failed to create Metal command queue");

        const auto parsed = parseMatter(R"(
            material release_fixture {
                parameter density : kg/m^3 = 1000;
                parameter mu : Pa = 1000;
                parameter lambda : Pa = 1000;
                model neo_hookean;
                energy = neo_hookean(mu, lambda);
                valid = J() - 0.2;
                supports fem;
            }
        )");
        require(parsed.succeeded(), "release fixture material parse failed");

        WorldSource source;
        source.environmentCount = static_cast<std::uint32_t>(kEnvironmentCount);
        source.gravity = {0.0f, 0.0f, 0.0f};
        source.frameTimestep = 1.0e-4;
        source.materials = {parsed.material};

        ObjectSource object;
        object.name = "release_fixture";
        object.representation = Representation::fem;
        object.mixedFEM = false;
        object.characteristicLength = 0.01;
        object.deformableContact = false;
        object.deformableSelfContact = false;
        object.femNodes = {
            {{0.00, 0.00, 0.00}},
            {{0.01, 0.00, 0.00}},
            {{0.00, 0.01, 0.00}},
            {{0.00, 0.00, 0.01}},
        };
        object.tetrahedra = {{{0u, 1u, 2u, 3u}}};
        object.femFixedNodes = originallyFixed
            ? std::vector<std::uint32_t>{0u, 1u, 2u, 3u}
            : std::vector<std::uint32_t>{1u, 2u, 3u};
        // Use the uniform-material FEM path here so grip snapshot admission is
        // exercised even when the world has no regional material layout.
        source.objects = {object};

        auto compiled = compileWorld(
            source,
            {.maximumRateExponent = 0u, .emitSpecializedMetal = false}
        );
        for (const auto& diagnostic : compiled.diagnostics) {
            std::cerr << diagnostic.message << '\n';
        }
        require(compiled.succeeded(), "release fixture world compile failed");
        world = std::move(compiled.world);

        RuntimeConfiguration configuration;
        configuration.metallib = NUMI_MATTER_METALLIB;
        configuration.environmentCount =
            static_cast<std::uint32_t>(kEnvironmentCount);
        configuration.adaptiveTransfer = false;
        configuration.captureDiagnostics = true;

        const auto initialized = runtime.initialize(world, configuration);
        require(initialized.encoded, initialized.message);
    }

    RuntimeStateSnapshot state() {
        auto snapshot = runtime.snapshot();
        require(snapshot.available, snapshot.message);
        return snapshot;
    }

    StepResult step(
        unsigned controlStep,
        const std::vector<nm_float4>& targets,
        int environmentToReject = -1,
        const std::vector<std::uint32_t>& resetMasks = {},
        std::uint32_t resetMaskStepStride = 0u
    ) {
        require(targets.size() == kTargetCount,
                "kinematic target array has the wrong size");

        std::vector<MRMetalWorldStatusGPU> environmentStatuses(
            kEnvironmentCount
        );
        environmentStatuses[1].environment = 1u;
        id<MTLBuffer> statusBuffer =
            makeSharedBuffer(device, environmentStatuses);
        id<MTLBuffer> targetBuffer = makeSharedBuffer(device, targets);
        id<MTLBuffer> resetMaskBuffer = nil;
        if (resetMaskStepStride != 0u) {
            require(resetMaskStepStride >= kEnvironmentCount &&
                        !resetMasks.empty(),
                    "reset-mask input has an invalid stride or no values");
            resetMaskBuffer = makeSharedBuffer(device, resetMasks);
        } else {
            require(resetMasks.empty(),
                    "reset-mask values were supplied without a stride");
        }

        auto commandBuffer = [queue commandBuffer];
        require(commandBuffer != nil, "failed to create Metal command buffer");

        EncodeRequest request;
        request.commandBuffer = (__bridge void*)commandBuffer;
        request.environmentStatuses = (__bridge void*)statusBuffer;
        request.femKinematicTargets = (__bridge void*)targetBuffer;
        request.femKinematicTargetCount =
            static_cast<std::uint32_t>(targets.size());
        request.controlStep = controlStep;
        request.physicsSubsteps = 1u;
        request.timestepSeconds = runtime.timestepSeconds();
        request.runAdaptiveTransfer = false;
        request.resetMasks = resetMaskBuffer == nil
            ? nullptr : (__bridge void*)resetMaskBuffer;
        request.resetMaskStepStride = resetMaskStepStride;

        request.phase = EncodePhase::preDynamics;
        const auto preDynamics = runtime.encode(request);
        require(preDynamics.encoded, preDynamics.message);

        // The residual arena is private device memory. Copy it into shared
        // staging after the accepted pre-dynamics solve and before post-commit,
        // keeping the diagnostic readback ordered in this same command buffer.
        id<MTLBuffer> residualBuffer = (__bridge id<MTLBuffer>)
            runtime.femMechanicalResidualBuffer();
        require(residualBuffer != nil,
                "runtime did not expose the FEM mechanical residual buffer");
        require(residualBuffer.length % sizeof(nm_float4) == 0u,
                "FEM mechanical residual buffer has an invalid byte length");

        id<MTLBuffer> residualReadback = [device
            newBufferWithLength:residualBuffer.length
                        options:MTLResourceStorageModeShared];
        require(residualReadback != nil,
                "failed to allocate FEM residual readback buffer");

        id<MTLBlitCommandEncoder> residualCopy =
            [commandBuffer blitCommandEncoder];
        require(residualCopy != nil,
                "failed to create residual readback blit encoder");
        [residualCopy copyFromBuffer:residualBuffer
                        sourceOffset:0u
                            toBuffer:residualReadback
                   destinationOffset:0u
                                size:residualBuffer.length];
        [residualCopy endEncoding];

        // Inject a failure only after pre-dynamics so the runtime must reject
        // the candidate transaction independently for the selected region.
        id<MTLBuffer> injectedFailureBuffer = nil;
        if (environmentToReject >= 0) {
            require(static_cast<std::size_t>(environmentToReject) <
                        kEnvironmentCount,
                    "rejected environment index is out of range");

            MRMetalWorldStatusGPU failure{};
            failure.environment =
                static_cast<std::uint32_t>(environmentToReject);
            failure.code = MR_STEP_DID_NOT_CONVERGE;
            injectedFailureBuffer = makeSharedBuffer(
                device, std::vector<MRMetalWorldStatusGPU>{failure}
            );

            id<MTLBlitCommandEncoder> failureInjection =
                [commandBuffer blitCommandEncoder];
            require(failureInjection != nil,
                    "failed to create failure injection blit encoder");
            [failureInjection copyFromBuffer:injectedFailureBuffer
                                sourceOffset:0u
                                    toBuffer:statusBuffer
                           destinationOffset:
                               static_cast<NSUInteger>(environmentToReject) *
                               sizeof(failure)
                                        size:sizeof(failure)];
            [failureInjection endEncoding];
        }

        request.phase = EncodePhase::postCommit;
        const auto postCommit = runtime.encode(request);
        require(postCommit.encoded, postCommit.message);

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        require(commandBuffer.status == MTLCommandBufferStatusCompleted,
                "Metal command buffer failed");
        (void)injectedFailureBuffer; // Keep it alive through command completion.

        const auto* residualRows = static_cast<const nm_float4*>(
            residualReadback.contents
        );
        const auto residualRowCount =
            residualReadback.length / sizeof(nm_float4);
        require(residualRows != nullptr && residualRowCount > 0u,
                "FEM residual readback returned no rows");

        std::vector<nm_float4> copiedResidual(
            residualRows, residualRows + residualRowCount
        );
        return {state(), std::move(copiedResidual)};
    }
};

const nm_float4& residualRow(
    const StepResult& result,
    std::size_t environment,
    std::size_t node
) {
    const std::size_t row = environment * kFEMNodeCount + node;
    require(row < result.femMechanicalResidual.size(),
            "FEM residual readback is missing a node row");
    return result.femMechanicalResidual[row];
}

void requireFiniteResidual(const nm_float4& value, const std::string& label) {
    require(std::isfinite(value.x) && std::isfinite(value.y) &&
                std::isfinite(value.z) && std::isfinite(value.w),
            "non-finite FEM residual row: " + label);
}

void check() {
    Fixture fixture;
    const auto initial = fixture.state();
    require(initial.femNodes.size() == kTargetCount,
            "fixture did not create two four-node FEM environments");

    std::vector<nm_float4> targets(kTargetCount);
    for (std::size_t node = 0; node < targets.size(); ++node) {
        const auto& position = initial.femNodes[node].positionAndMass;
        targets[node] = {position.x, position.y, position.z + 1.0e-5f, 1.0f};
    }

    const auto driven = fixture.step(0u, targets);
    for (const auto& status : driven.snapshot.statuses) {
        require(status.code == NM_STATUS_SUCCESS, "drive step was rejected");
    }

    // Release all four nodes in environment 0. Environment 1 remains a fixed
    // kinematic control at the same translated position.
    for (std::size_t node = 0; node < kFEMNodeCount; ++node) {
        targets[node] = {0.0f, 0.0f, 0.0f, 2.0f};
    }

    const auto released = fixture.step(1u, targets);
    for (const auto& status : released.snapshot.statuses) {
        require(status.code == NM_STATUS_SUCCESS, "release step was rejected");
    }

    for (std::size_t node = 0; node < kFEMNodeCount; ++node) {
        const auto& freeNode = released.snapshot.femNodes[node];
        require(freeNode.positionAndMass.w ==
                    initial.femNodes[node].positionAndMass.w,
                "release changed node mass");
        require(freeNode.restAndFixed.w == 0.0f &&
                    freeNode.velocityAndInverseMass.w ==
                        1.0f / freeNode.positionAndMass.w,
                "released node did not regain inverse mass");

        const float velocityDifference = std::abs(
            freeNode.velocityAndInverseMass.z -
            driven.snapshot.femNodes[node].velocityAndInverseMass.z
        );
        require(velocityDifference < 2.0e-5f,
                "release lost z momentum at node " + std::to_string(node) +
                    "; residual=" + std::to_string(
                        released.snapshot.solverCertificates[0].nonlinear.x
                    ));

        const float zDisplacement =
            freeNode.positionAndMass.z -
            driven.snapshot.femNodes[node].positionAndMass.z;
        require(std::abs(zDisplacement - 1.0e-5f) < 2.0e-8f,
                "released node did not continue its rigid translation");

        // A released free row must be available and finite after the accepted
        // pre-dynamics solve that produced its state.
        requireFiniteResidual(
            residualRow(released, 0u, node),
            "free environment 0, node " + std::to_string(node)
        );

        const auto& fixedControl = released.snapshot.femNodes[
            kFEMNodeCount + node
        ];
        require(fixedControl.restAndFixed.w == 1.0f &&
                    fixedControl.positionAndMass.z ==
                        driven.snapshot.femNodes[kFEMNodeCount + node]
                            .positionAndMass.z,
                "fixed environment 1 control moved");

        // This fixture uses non-mixed FEM. Its fixed mechanical rows, including
        // the unused fourth component, are specified to be exactly zero.
        const auto& fixedResidual = residualRow(released, 1u, node);
        require(fixedResidual.x == 0.0f && fixedResidual.y == 0.0f &&
                    fixedResidual.z == 0.0f && fixedResidual.w == 0.0f,
                "fixed environment 1 residual row is nonzero");
    }

    for (auto& target : targets) {
        target = {0.0f, 0.0f, 0.0f, 2.0f};
    }
    const auto rejected = fixture.step(2u, targets, 1);
    require(rejected.snapshot.statuses[0].code == NM_STATUS_SUCCESS &&
                rejected.snapshot.statuses[1].code != NM_STATUS_SUCCESS,
            "isolated environment rejection was not observed");
    require(std::memcmp(
                rejected.snapshot.femNodes.data() + kFEMNodeCount,
                released.snapshot.femNodes.data() + kFEMNodeCount,
                kFEMNodeCount * sizeof(NMFEMNodeStateGPU)
            ) == 0,
            "failed release did not roll back environment 1 state");

    const auto restored = fixture.runtime.restore(released.snapshot);
    require(restored.encoded,
            "released snapshot restore failed: " + restored.message);

    auto corrupted = released.snapshot;
    corrupted.femNodes[0].positionAndMass.w *= 2.0f;
    require(!fixture.runtime.restore(corrupted).encoded,
            "snapshot with altered regional mass was restored");

    corrupted = released.snapshot;
    corrupted.femNodes[0].restAndFixed.x += 0.01f;
    require(!fixture.runtime.restore(corrupted).encoded,
            "snapshot with altered rest coordinates was restored");

    // The authored x coordinate is +0.0f. Released-snapshot validation must
    // preserve the exact authored rest-coordinate bits, so a signed-zero
    // mutation is corruption even though IEEE float equality treats it equal.
    const float authoredRestX = initial.femNodes[0].restAndFixed.x;
    require(authoredRestX == 0.0f && !std::signbit(authoredRestX),
            "fixture rest x coordinate is not authored positive zero");
    corrupted = released.snapshot;
    corrupted.femNodes[0].restAndFixed.x = -0.0f;
    require(std::memcmp(&corrupted.femNodes[0].restAndFixed.x,
                        &released.snapshot.femNodes[0].restAndFixed.x,
                        sizeof(float)) != 0,
            "negative-zero fixture mutation did not change rest-coordinate bits");
    require(!fixture.runtime.restore(corrupted).encoded,
            "snapshot with signed-zero rest-coordinate mutation was restored");

    // Acquire a temporary grip on an accepted free row, move it through a
    // prescribed endpoint, persist the constraint tag, then release it again.
    // Environment 1 is left as an untouched free-state control for this step.
    std::vector<nm_float4> gripTargets(kTargetCount);
    const auto& freeNodeBeforeGrip = released.snapshot.femNodes[0];
    gripTargets[0] = {
        freeNodeBeforeGrip.positionAndMass.x + 1.0e-5f,
        freeNodeBeforeGrip.positionAndMass.y,
        freeNodeBeforeGrip.positionAndMass.z,
        3.0f,
    };
    const auto acquired = fixture.step(3u, gripTargets);
    require(acquired.snapshot.statuses[0].code == NM_STATUS_SUCCESS &&
                acquired.snapshot.statuses[1].code == NM_STATUS_SUCCESS,
            "temporary grip acquisition was rejected");
    const auto& grippedNode = acquired.snapshot.femNodes[0];
    require(grippedNode.restAndFixed.w == 3.0f &&
                grippedNode.positionAndMass.w == freeNodeBeforeGrip.positionAndMass.w &&
                grippedNode.velocityAndInverseMass.w == 0.0f,
            "temporary grip did not persist its accepted constraint and mass");
    require(std::abs(grippedNode.positionAndMass.x -
                         gripTargets[0].x) < 2.0e-7f,
            "temporary grip did not reach its prescribed endpoint");

    const auto temporaryRestore = fixture.runtime.restore(acquired.snapshot);
    require(temporaryRestore.encoded,
            "temporary-grip snapshot restore failed: " + temporaryRestore.message);
    auto corruptedGrip = acquired.snapshot;
    corruptedGrip.femNodes[0].restAndFixed.w = 1.0f;
    const auto authoredStaticRestore = fixture.runtime.restore(corruptedGrip);
    require(authoredStaticRestore.encoded,
            "snapshot could not restore its legitimately authored static tag: " +
                authoredStaticRestore.message);
    require(fixture.runtime.restore(acquired.snapshot).encoded,
            "could not restore the temporary-grip snapshot after static-state check");
    corruptedGrip = acquired.snapshot;
    corruptedGrip.femNodes[0].velocityAndInverseMass.w = 1.0f /
        corruptedGrip.femNodes[0].positionAndMass.w;
    require(!fixture.runtime.restore(corruptedGrip).encoded,
            "snapshot with a temporary grip but free inverse mass was restored");
    corruptedGrip = acquired.snapshot;
    corruptedGrip.femNodes[0].restAndFixed.z += 0.01f;
    require(!fixture.runtime.restore(corruptedGrip).encoded,
            "temporary-grip restore accepted changed authored rest coordinates");

    // A second world distinguishes a real forged-static-tag rejection from
    // restoring the original fixture's legitimate authored static state.
    // Node 0 is authored free here while the other three tetrahedron nodes
    // remain fixed to remove rigid-body modes during the grip solve.
    Fixture freeNodeFixture(false);
    const auto freeNodeInitial = freeNodeFixture.state();
    require(freeNodeInitial.femNodes[0].restAndFixed.w == 0.0f &&
                freeNodeInitial.femNodes[0].velocityAndInverseMass.w > 0.0f,
            "forged-tag control node was not authored free");
    std::vector<nm_float4> freeNodeGripTargets(kTargetCount);
    const auto& freeBeforeGrip = freeNodeInitial.femNodes[0];
    freeNodeGripTargets[0] = {
        freeBeforeGrip.positionAndMass.x + 1.0e-5f,
        freeBeforeGrip.positionAndMass.y,
        freeBeforeGrip.positionAndMass.z,
        3.0f,
    };
    const auto freeNodeAcquired = freeNodeFixture.step(
        0u, freeNodeGripTargets);
    require(freeNodeAcquired.snapshot.statuses[0].code == NM_STATUS_SUCCESS &&
                freeNodeAcquired.snapshot.statuses[1].code == NM_STATUS_SUCCESS &&
                freeNodeAcquired.snapshot.femNodes[0].restAndFixed.w == 3.0f,
            "temporary grip on an authored-free node was rejected");
    require(freeNodeFixture.runtime.restore(freeNodeAcquired.snapshot).encoded,
            "authored-free temporary-grip snapshot did not restore");
    auto forgedFreeGrip = freeNodeAcquired.snapshot;
    forgedFreeGrip.femNodes[0].restAndFixed.w = 1.0f;
    require(!freeNodeFixture.runtime.restore(forgedFreeGrip).encoded,
            "snapshot with a forged static tag replaced a grip on an authored-free node");

    // An authored episode reset must discard the accepted temporary grip and
    // restore the exact compiled FEM defaults in only the selected region.
    std::vector<std::uint32_t> resetMasks(5u * kEnvironmentCount, 0u);
    resetMasks[4u * kEnvironmentCount] = 1u;
    const auto resetWhileGripped = fixture.step(
        4u, std::vector<nm_float4>(kTargetCount), -1, resetMasks,
        static_cast<std::uint32_t>(kEnvironmentCount));
    require(resetWhileGripped.snapshot.statuses[0].code == NM_STATUS_SUCCESS &&
                resetWhileGripped.snapshot.statuses[1].code == NM_STATUS_SUCCESS,
            "episode reset of a temporary grip was rejected");
    require(std::memcmp(resetWhileGripped.snapshot.femNodes.data(),
                        initial.femNodes.data(),
                        kFEMNodeCount * sizeof(NMFEMNodeStateGPU)) == 0,
            "episode reset did not restore default FEM constraint tags/poses");
    require(std::memcmp(resetWhileGripped.snapshot.femNodes.data() + kFEMNodeCount,
                        acquired.snapshot.femNodes.data() + kFEMNodeCount,
                        kFEMNodeCount * sizeof(NMFEMNodeStateGPU)) == 0,
            "episode reset changed an unselected environment");
    require(fixture.runtime.restore(acquired.snapshot).encoded,
            "could not restore the accepted temporary-grip state after reset check");

    // Inject rejection after candidate acquisition/motion. The accepted node
    // and its temporary tag must roll back bit-for-bit.
    gripTargets[0].x += 1.0e-5f;
    const auto gripRollback = fixture.step(5u, gripTargets, 0);
    require(gripRollback.snapshot.statuses[0].code != NM_STATUS_SUCCESS &&
                gripRollback.snapshot.statuses[1].code == NM_STATUS_SUCCESS,
            "temporary-grip failure injection did not isolate environment 0");
    require(std::memcmp(gripRollback.snapshot.femNodes.data(),
                        acquired.snapshot.femNodes.data(),
                        kFEMNodeCount * sizeof(NMFEMNodeStateGPU)) == 0,
            "failed temporary-grip command did not restore the accepted tag/pose");

    gripTargets[0] = {0.0f, 0.0f, 0.0f, 2.0f};
    const auto releasedGrip = fixture.step(6u, gripTargets);
    require(releasedGrip.snapshot.statuses[0].code == NM_STATUS_SUCCESS &&
                releasedGrip.snapshot.statuses[1].code == NM_STATUS_SUCCESS,
            "temporary-grip release was rejected");
    const auto& releasedGripNode = releasedGrip.snapshot.femNodes[0];
    require(releasedGripNode.restAndFixed.w == 0.0f &&
                releasedGripNode.positionAndMass.w == grippedNode.positionAndMass.w &&
                releasedGripNode.velocityAndInverseMass.w ==
                    1.0f / releasedGripNode.positionAndMass.w,
            "temporary grip release did not restore the free row/mass");

    // Unknown command tags remain fail-closed and cannot alter accepted state.
    gripTargets[0] = {0.0f, 0.0f, 0.0f, 4.0f};
    const auto invalidTarget = fixture.step(7u, gripTargets);
    require(invalidTarget.snapshot.statuses[0].code != NM_STATUS_SUCCESS &&
                invalidTarget.snapshot.statuses[1].code == NM_STATUS_SUCCESS,
            "invalid target tag did not remain isolated to its environment");
    require(std::memcmp(invalidTarget.snapshot.femNodes.data(),
                        releasedGrip.snapshot.femNodes.data(),
                        kFEMNodeCount * sizeof(NMFEMNodeStateGPU)) == 0,
            "invalid target tag mutated accepted environment 0 state");

    std::cout
        << "native_release_passed"
        << " momentum_preserved=true"
        << " free_residual_finite=true"
        << " fixed_residual_rows_zero=true"
        << " rejected_release_rollback=true"
        << " released_snapshot_restore=true"
        << " corrupt_mass_rejected=true"
        << " signed_zero_rest_coordinate_rejected=true"
        << " temporary_grip_acquire_release=true"
        << " temporary_grip_snapshot_restore=true"
        << " authored_static_snapshot_restore=true"
        << " forged_static_tag_on_authored_free_rejected=true"
        << " temporary_grip_episode_reset=true"
        << " temporary_grip_rejection_rollback=true"
        << " temporary_grip_invalid_tag_isolated=true"
        << " physical_validation=false\n";
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            check();
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
