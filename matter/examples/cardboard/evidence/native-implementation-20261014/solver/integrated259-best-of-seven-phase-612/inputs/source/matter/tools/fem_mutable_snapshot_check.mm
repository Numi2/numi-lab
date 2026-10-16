#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace numi::matter;

namespace {

constexpr std::uint32_t kEnvironmentCount = 2u;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void appendTetrahedron(ObjectSource& object, const double xOffset) {
    object.femNodes = {
        {{xOffset + 0.000, 0.000, 0.000}},
        {{xOffset + 0.010, 0.000, 0.000}},
        {{xOffset + 0.000, 0.010, 0.000}},
        {{xOffset + 0.000, 0.000, 0.010}},
    };
    object.tetrahedra = {{{0u, 1u, 2u, 3u}}};
    object.femFixedNodes = {0u, 1u, 2u, 3u};
    object.mixedFEM = false;
    object.deformableContact = false;
    object.deformableSelfContact = false;
    object.characteristicLength = 0.01;
}

void runAcceptedControlStep(Runtime& runtime,
                           id<MTLDevice> device,
                           id<MTLCommandQueue> queue,
                           const std::uint32_t controlStep,
                           const std::vector<nm_float4>& femTargets = {}) {
    std::vector<MRMetalWorldStatusGPU> statuses(kEnvironmentCount);
    for (std::uint32_t environment = 0u;
         environment < kEnvironmentCount; ++environment) {
        statuses[environment].environment = environment;
    }
    id<MTLBuffer> statusBuffer = [device newBufferWithBytes:statuses.data()
        length:statuses.size() * sizeof(MRMetalWorldStatusGPU)
        options:MTLResourceStorageModeShared];
    require(statusBuffer != nil, "cannot allocate environment status buffer");

    id<MTLBuffer> targetsBuffer = nil;
    if (!femTargets.empty()) {
        targetsBuffer = [device newBufferWithBytes:femTargets.data()
            length:femTargets.size() * sizeof(nm_float4)
            options:MTLResourceStorageModeShared];
        require(targetsBuffer != nil, "cannot allocate FEM target buffer");
    }
    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(command != nil, "cannot allocate mutation command buffer");

    EncodeRequest request;
    request.commandBuffer = (__bridge void*)command;
    request.environmentStatuses = (__bridge void*)statusBuffer;
    request.femKinematicTargets = (__bridge void*)targetsBuffer;
    request.femKinematicTargetCount =
        static_cast<std::uint32_t>(femTargets.size());
    request.controlStep = controlStep;
    request.physicsSubsteps = 1u;
    request.timestepSeconds = runtime.timestepSeconds();
    request.runAdaptiveTransfer = false;
    request.phase = EncodePhase::preDynamics;
    RuntimeDiagnostics encoded = runtime.encode(request);
    require(encoded.encoded, "mutation preDynamics encode failed: " + encoded.message);
    request.phase = EncodePhase::postCommit;
    encoded = runtime.encode(request);
    require(encoded.encoded, "mutation postCommit encode failed: " + encoded.message);
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
            "mutation command buffer did not complete");
}

void requireAcceptedSplit(const RuntimeStateSnapshot& snapshot,
                         const CompiledWorld& world,
                         const std::uint32_t environment) {
    const std::uint32_t nodeCount = world.dispatch.femNodeCount;
    const std::uint32_t tetrahedronCount = world.dispatch.tetrahedronCount;
    const auto& mutableLayout = world.objects[0];
    const std::uint32_t activatedNode = mutableLayout.stateOffset + 4u;
    const std::size_t nodeBase =
        static_cast<std::size_t>(environment) * nodeCount;
    const std::size_t tetrahedronBase =
        static_cast<std::size_t>(environment) * tetrahedronCount;
    std::uint32_t activeTetrahedra = 0u;
    bool splitUsesReservedNode = false;
    for (std::uint32_t local = 0u; local < mutableLayout.elementCount; ++local) {
        const auto& tetrahedron = snapshot.femTopologyTetrahedra[
            tetrahedronBase + mutableLayout.elementOffset + local];
        if ((tetrahedron.identity.w & NM_OBJECT_ACTIVE) == 0u) continue;
        ++activeTetrahedra;
        splitUsesReservedNode = splitUsesReservedNode ||
            tetrahedron.nodes.x == activatedNode ||
            tetrahedron.nodes.y == activatedNode ||
            tetrahedron.nodes.z == activatedNode ||
            tetrahedron.nodes.w == activatedNode;
    }
    require(activeTetrahedra == 2u && splitUsesReservedNode,
            "environment snapshot does not contain its accepted edge split");
    require((snapshot.femTopologyNodes[nodeBase + activatedNode].identity.w &
                NM_TOPOLOGY_ACTIVE) != 0u &&
                snapshot.femNodes[nodeBase + activatedNode].positionAndMass.w > 0.0f,
            "environment snapshot lost its activated split node");
}

void check() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil, "Apple Metal device is unavailable");
    const std::string deviceName = [[device name] UTF8String];
    require(deviceName.find("Apple") != std::string::npos &&
                deviceName.find("Paravirtual") == std::string::npos,
            "a physical Apple Metal device is required");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, "cannot create Metal command queue");

    const auto parsed = parseMatter(R"(
        material topology_snapshot_fixture {
            parameter density : kg/m^3 = 1000;
            parameter mu : Pa = 1000;
            parameter lambda : Pa = 1000;
            model neo_hookean;
            energy = neo_hookean(mu, lambda);
            valid = J() - 0.2;
            supports fem;
        }
    )");
    require(parsed.succeeded(), "fixture material parse failed");

    WorldSource source;
    source.environmentCount = kEnvironmentCount;
    source.gravity = {0.0f, 0.0f, 0.0f};
    source.frameTimestep = 1.0e-4;
    source.materials = {parsed.material};

    ObjectSource mutableObject;
    mutableObject.name = "mutable_split_target";
    mutableObject.representation = Representation::fem;
    appendTetrahedron(mutableObject, 0.0);
    mutableObject.mutationPolicy.enabled = true;
    mutableObject.femCapacity.nodes = 5u;
    mutableObject.femCapacity.tetrahedra = 2u;
    mutableObject.femCapacity.mutationCommands = 1u;
    MutationCommandSource split;
    split.kind = NM_MUTATION_EDGE_SPLIT;
    split.stableIdentifier = 41u;
    split.controlStep = 0u;
    split.target = 0u;
    mutableObject.mutationCommands.push_back(split);

    ObjectSource immutableObject;
    immutableObject.name = "immutable_snapshot_control";
    immutableObject.representation = Representation::fem;
    appendTetrahedron(immutableObject, 0.1);
    source.objects = {mutableObject, immutableObject};

    CompileResult compiled = compileWorld(source,
        {.maximumRateExponent = 0u, .emitSpecializedMetal = false});
    for (const Diagnostic& diagnostic : compiled.diagnostics)
        std::cerr << diagnostic.message << '\n';
    require(compiled.succeeded(), "mixed topology fixture did not compile");
    require(compiled.world.objects.size() == 2u &&
                (compiled.world.objects[0].flags & NM_OBJECT_MUTABLE_TOPOLOGY) != 0u &&
                (compiled.world.objects[1].flags & NM_OBJECT_MUTABLE_TOPOLOGY) == 0u,
            "compiled fixture lost its mutable/immutable ownership split");

    Runtime runtime;
    RuntimeConfiguration configuration;
    configuration.metallib = NUMI_MATTER_METALLIB;
    configuration.environmentCount = kEnvironmentCount;
    configuration.adaptiveTransfer = false;
    configuration.captureDiagnostics = true;
    const RuntimeDiagnostics initialized = runtime.initialize(compiled.world, configuration);
    require(initialized.encoded, "runtime initialization failed: " + initialized.message);

    const RuntimeStateSnapshot before = runtime.snapshot();
    require(before.available, "initial snapshot unavailable: " + before.message);
    const auto& mutableLayout = compiled.world.objects[0];
    const auto& immutableLayout = compiled.world.objects[1];
    const std::uint32_t nodeCount = compiled.world.dispatch.femNodeCount;
    const std::uint32_t activatedNode = mutableLayout.stateOffset + 4u;
    require(before.femNodes.size() ==
                static_cast<std::size_t>(kEnvironmentCount) * nodeCount,
            "initial snapshot does not contain both environments");
    for (std::uint32_t environment = 0u;
         environment < kEnvironmentCount; ++environment) {
        const std::size_t nodeBase = static_cast<std::size_t>(environment) * nodeCount;
        require(before.femNodes[nodeBase + activatedNode].positionAndMass.w == 0.0f,
                "split reserve node was not initially dormant in both environments");
    }

    runAcceptedControlStep(runtime, device, queue, 0u);
    const RuntimeStateSnapshot splitSnapshot = runtime.snapshot();
    require(splitSnapshot.available,
            "post-split snapshot unavailable: " + splitSnapshot.message);
    require(splitSnapshot.statuses.size() == kEnvironmentCount,
            "post-split status snapshot does not cover both environments");
    for (std::uint32_t environment = 0u;
         environment < kEnvironmentCount; ++environment) {
        require(splitSnapshot.statuses[environment].code == NM_STATUS_SUCCESS,
                "scheduled edge split was rejected in environment " +
                    std::to_string(environment));
        const std::size_t topologyBase =
            static_cast<std::size_t>(environment) * compiled.world.dispatch.objectCount;
        require(splitSnapshot.topologyStates[topologyBase].roles.w >
                    compiled.world.objects[0].topologyGeneration,
                "edge split did not advance topology in both environments");
        requireAcceptedSplit(splitSnapshot, compiled.world, environment);

        const std::size_t nodeBase = static_cast<std::size_t>(environment) * nodeCount;
        require(std::memcmp(&splitSnapshot.femNodes[nodeBase + immutableLayout.stateOffset],
                            &before.femNodes[nodeBase + immutableLayout.stateOffset],
                            sizeof(NMFEMNodeStateGPU)) == 0,
                "mutable topology transaction changed an immutable object's node");
    }

    const RuntimeDiagnostics validRestore = runtime.restore(splitSnapshot);
    require(validRestore.encoded,
            "restore rejected an actual accepted two-environment mutable-topology snapshot: " +
                validRestore.message);

    RuntimeStateSnapshot corruptImmutable = splitSnapshot;
    const std::size_t corruptNode =
        static_cast<std::size_t>(1u) * nodeCount + immutableLayout.stateOffset;
    corruptImmutable.femNodes[corruptNode].restAndFixed.x += 0.001f;
    const RuntimeDiagnostics rejectedRestore = runtime.restore(corruptImmutable);
    require(!rejectedRestore.encoded,
            "mixed-world restore admitted changed immutable rest coordinates");
    require(runtime.restore(splitSnapshot).encoded,
            "immutable corruption rejection prevented restoring the actual split snapshot");

    std::vector<nm_float4> targets(
        static_cast<std::size_t>(kEnvironmentCount) * nodeCount);
    std::array<nm_float4, kEnvironmentCount> expectedTargets{};
    for (std::uint32_t environment = 0u;
         environment < kEnvironmentCount; ++environment) {
        const std::size_t nodeBase = static_cast<std::size_t>(environment) * nodeCount;
        for (std::uint32_t node = 0u; node < nodeCount; ++node) {
            const auto& position = splitSnapshot.femNodes[nodeBase + node].positionAndMass;
            targets[nodeBase + node] = {position.x, position.y, position.z, 0.0f};
        }
        const std::uint32_t movedNode = mutableLayout.stateOffset + 1u;
        const auto& original = splitSnapshot.femNodes[nodeBase + movedNode].positionAndMass;
        const float dx = environment == 0u ? 1.0e-5f : 2.0e-5f;
        expectedTargets[environment] = {original.x + dx, original.y, original.z, 1.0f};
        targets[nodeBase + movedNode] = expectedTargets[environment];
    }
    runAcceptedControlStep(runtime, device, queue, 1u, targets);
    const RuntimeStateSnapshot postRestoreStep = runtime.snapshot();
    require(postRestoreStep.available,
            "post-restore solver-step snapshot unavailable: " + postRestoreStep.message);
    require(postRestoreStep.statuses.size() == kEnvironmentCount,
            "post-restore step did not publish both environment statuses");
    for (std::uint32_t environment = 0u;
         environment < kEnvironmentCount; ++environment) {
        require(postRestoreStep.statuses[environment].code == NM_STATUS_SUCCESS,
                "post-restore FEM step was rejected in environment " +
                    std::to_string(environment));
        requireAcceptedSplit(postRestoreStep, compiled.world, environment);
        const std::size_t nodeBase = static_cast<std::size_t>(environment) * nodeCount;
        const auto& actual = postRestoreStep.femNodes[
            nodeBase + mutableLayout.stateOffset + 1u].positionAndMass;
        require(std::fabs(actual.x - expectedTargets[environment].x) < 1.0e-7f &&
                    std::fabs(actual.y - expectedTargets[environment].y) < 1.0e-7f &&
                    std::fabs(actual.z - expectedTargets[environment].z) < 1.0e-7f,
                "post-restore kinematic target did not reach its environment endpoint");
        require(std::memcmp(
                    &postRestoreStep.femNodes[nodeBase + immutableLayout.stateOffset],
                    &splitSnapshot.femNodes[nodeBase + immutableLayout.stateOffset],
                    sizeof(NMFEMNodeStateGPU)) == 0,
                "post-restore step leaked FEM state across object/environment isolation");
    }
    const RuntimeDiagnostics postStepRestore = runtime.restore(postRestoreStep);
    require(postStepRestore.encoded,
            "restore rejected the accepted two-environment post-split FEM step: " +
                postStepRestore.message);

}

} // namespace

int main() {
    @autoreleasepool {
        try {
            check();
            std::cout << "fem_mutable_snapshot_check=passed"
                      << " mutation=edge_split"
                      << " actual_accepted_snapshot=1"
                      << " mixed_mutable_immutable=1"
                      << " environments=2"
                      << " split_restore_postrestore_step=1"
                      << " immutable_corruption_rejected=1\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "fem_mutable_snapshot_check=failed reason="
                      << error.what() << '\n';
            return 1;
        }
    }
}
