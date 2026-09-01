#include "numi/matter/matter.hpp"
#include "numi/matter/numi_human.hpp"

#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            numi::matter::WorldSource source;
            source.environmentCount = 1u;
            source.frameTimestep = 1.0e-4;
            source.gravity = {0.0, 0.0, 0.0};
            source.mixedSolver.newtonIterations = 8u;
            source.mixedSolver.fgmresIterations = 20u;
            auto material = numi::matter::parseMatterFile(
                NUMI_HUMAN_TENDON_FEM_PROBE_MATERIAL
            );
            require(material.succeeded(), "probe material did not parse");
            source.materials.push_back(std::move(material.material));
            numi::matter::ObjectSource object;
            object.name = "numi_human_tendon_fem_load_probe";
            object.materialIndex = 0u;
            object.representation = numi::matter::Representation::fem;
            object.mixedFEM = false;
            object.deformableContact = false;
            object.deformableSelfContact = false;
            object.characteristicLength = 0.01;
            object.femNodes = {
                {0.0, 0.0, 0.0}, {0.01, 0.0, 0.0},
                {0.0, 0.01, 0.0}, {0.0, 0.0, 0.01},
            };
            object.femFixedNodes = {0u, 1u, 2u};
            object.femContactNodes = {0u};
            object.tetrahedra.push_back({{0u, 1u, 2u, 3u}});
            source.objects.push_back(std::move(object));
            numi::matter::CompileOptions compileOptions;
            compileOptions.maximumRateExponent = 0u;
            auto compiled = numi::matter::compileWorld(source, compileOptions);
            require(compiled.succeeded(), "probe world did not compile");
            require(compiled.world.objects.size() == 1u &&
                        (compiled.world.objects.front().flags &
                            NM_OBJECT_DISABLE_DEFORMABLE_CONTACT) != 0u,
                    "deformable-contact opt-out was not compiled");

            numi::matter::Runtime runtime;
            const auto initialized = runtime.initialize(compiled.world, {
                .metallib = NUMI_MATTER_METALLIB,
                .environmentCount = 1u,
                .captureEvents = true,
                .captureDiagnostics = true,
                .automaticIdentification = false,
                .adaptiveTransfer = false,
            });
            require(initialized.encoded && runtime.valid(),
                    "probe runtime did not initialize");
            const auto initial = runtime.snapshot();
            require(initial.available && initial.femNodes.size() == 4u,
                    "probe initial snapshot is unavailable");

            std::vector<NMNumiHumanTendonFEMNodeLoadGPU> nodeLoads(4u);
            for (auto& load : nodeLoads) {
                std::fill_n(load.endpointIndex, 4u, NM_INVALID_INDEX);
            }
            nodeLoads[3u].endpointIndex[0u] = 0u;
            nodeLoads[3u].scale.x = 0.1f;
            std::vector<NMNumiHumanTendonFEMNodeAnchorGPU> nodeAnchors(4u);
            for (auto& anchor : nodeAnchors) anchor.bodyIndex = NM_INVALID_INDEX;
            constexpr float anchorPoints[3u][3u]{
                {0.0f, 0.0f, 0.0f},
                {0.01f, 0.0f, 0.0f},
                {0.0f, 0.01f, 0.0f},
            };
            for (std::uint32_t node = 0u; node < 3u; ++node) {
                nodeAnchors[node].bodyIndex = 0u;
                nodeAnchors[node].flags =
                    NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE;
                nodeAnchors[node].localPoint = {
                    anchorPoints[node][0], anchorPoints[node][1],
                    anchorPoints[node][2], 0.0f};
            }
            NMNumiHumanTendonFEMEndpointReplacementGPU replacement{};
            replacement.loadEndpointIndex = 0u;
            replacement.anchorEndpointIndex = 1u;
            replacement.flags =
                NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_ACTIVE |
                NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_FULL_MUSCLE_ROW;
            replacement.forceOwnerFraction.x = 0.1f;
            NMNumiHumanFEMContactSampleGPU contactSample{};
            contactSample.slaveNode = 3u;
            contactSample.masterNode0 = 0u;
            contactSample.masterNode1 = 1u;
            contactSample.masterNode2 = 2u;
            contactSample.barycentricAndReferenceSeparation = {
                1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f, 0.0102f};
            contactSample.normalAndArea = {
                0.0f, 0.0f, 1.0f, 1.0e-5f};
            contactSample.stiffness = {1.0e8f, 0.0f, 0.0f, 0.0f};
            NMNumiHumanArticularContactSampleGPU articularContactSample{};
            articularContactSample.slaveBodyIndex = 0u;
            articularContactSample.masterBodyIndex = 1u;
            articularContactSample.flags =
                NM_NUMI_HUMAN_ARTICULAR_CONTACT_ACTIVE;
            articularContactSample.slaveLocalPointAndArea = {
                0.0f, 0.015f, 0.015f, 1.0e-4f};
            articularContactSample.masterLocalTriangle0AndReferenceSeparation = {
                0.0f, 0.0f, 0.0f, 0.002f};
            articularContactSample.masterLocalTriangle1AndStiffness = {
                0.0f, 0.02f, 0.0f, 1.0e7f};
            articularContactSample
                .masterLocalTriangle2AndNormalStrainPerPressure = {
                    0.0f, 0.0f, 0.02f, 1.0e-6f};
            articularContactSample.masterLocalReferenceNormalAndReserved = {
                1.0f, 0.0f, 0.0f, 0.0f};
            articularContactSample
                .masterLocalAdjacentOpposite1AndActive = {
                    0.0f, 0.02f, 0.02f, 1.0f};
            const std::array<NMNumiHumanFEMContactContributionGPU, 4u>
                contactContributions{{
                    {.sampleIndex = 0u, .role = 1u},
                    {.sampleIndex = 0u, .role = 2u},
                    {.sampleIndex = 0u, .role = 3u},
                    {.sampleIndex = 0u, .role = 0u},
                }};
            const std::array<NMIncidenceRangeGPU, 4u> contactRanges{{
                {.first = 0u, .count = 1u},
                {.first = 1u, .count = 1u},
                {.first = 2u, .count = 1u},
                {.first = 3u, .count = 1u},
            }};
            auto invalidContactContributions = contactContributions;
            invalidContactContributions[3u].role = 1u;
            numi::matter::NumiHumanTendonFEMLoadAdapter rejectedAdapter;
            require(!rejectedAdapter.initialize(runtime, {
                        .nodeLoads = nodeLoads,
                        .nodeAnchors = nodeAnchors,
                        .endpointReplacements = std::span(&replacement, 1u),
                        .contactSamples = std::span(&contactSample, 1u),
                        .contactContributions = invalidContactContributions,
                        .contactRanges = contactRanges,
                        .articularContactSamples =
                            std::span(&articularContactSample, 1u),
                        .endpointCount = 2u,
                        .environmentCount = 1u,
                        .productionForceOwnerFraction = 0.1f,
                    }, {
                        .metallib = NUMI_MATTER_METALLIB,
                    }),
                    "probe malformed contact incidence did not fail closed");
            auto invalidArticularContact = articularContactSample;
            invalidArticularContact.masterBodyIndex =
                invalidArticularContact.slaveBodyIndex;
            numi::matter::NumiHumanTendonFEMLoadAdapter
                rejectedArticularAdapter;
            require(!rejectedArticularAdapter.initialize(runtime, {
                        .nodeLoads = nodeLoads,
                        .nodeAnchors = nodeAnchors,
                        .endpointReplacements = std::span(&replacement, 1u),
                        .contactSamples = std::span(&contactSample, 1u),
                        .contactContributions = contactContributions,
                        .contactRanges = contactRanges,
                        .articularContactSamples =
                            std::span(&invalidArticularContact, 1u),
                        .endpointCount = 2u,
                        .environmentCount = 1u,
                        .productionForceOwnerFraction = 0.1f,
                    }, {
                        .metallib = NUMI_MATTER_METALLIB,
                    }),
                    "probe malformed articular contact did not fail closed");
            auto invalidReferenceNormal = articularContactSample;
            invalidReferenceNormal.masterLocalReferenceNormalAndReserved.x =
                2.0f;
            numi::matter::NumiHumanTendonFEMLoadAdapter
                rejectedReferenceNormalAdapter;
            require(!rejectedReferenceNormalAdapter.initialize(runtime, {
                        .nodeLoads = nodeLoads,
                        .nodeAnchors = nodeAnchors,
                        .endpointReplacements = std::span(&replacement, 1u),
                        .contactSamples = std::span(&contactSample, 1u),
                        .contactContributions = contactContributions,
                        .contactRanges = contactRanges,
                        .articularContactSamples =
                            std::span(&invalidReferenceNormal, 1u),
                        .endpointCount = 2u,
                        .environmentCount = 1u,
                        .productionForceOwnerFraction = 0.1f,
                    }, {
                        .metallib = NUMI_MATTER_METALLIB,
                    }),
                    "probe non-unit articular reference normal did not fail closed");
            auto invalidAdjacentContact = articularContactSample;
            invalidAdjacentContact
                .masterLocalAdjacentOpposite1AndActive.w = 2.0f;
            numi::matter::NumiHumanTendonFEMLoadAdapter
                rejectedAdjacentAdapter;
            require(!rejectedAdjacentAdapter.initialize(runtime, {
                        .nodeLoads = nodeLoads,
                        .nodeAnchors = nodeAnchors,
                        .endpointReplacements = std::span(&replacement, 1u),
                        .contactSamples = std::span(&contactSample, 1u),
                        .contactContributions = contactContributions,
                        .contactRanges = contactRanges,
                        .articularContactSamples =
                            std::span(&invalidAdjacentContact, 1u),
                        .endpointCount = 2u,
                        .environmentCount = 1u,
                        .productionForceOwnerFraction = 0.1f,
                    }, {
                        .metallib = NUMI_MATTER_METALLIB,
                    }),
                    "probe malformed articular adjacency did not fail closed");
            auto internalArticularContact = articularContactSample;
            internalArticularContact.masterBodyIndex =
                internalArticularContact.slaveBodyIndex;
            internalArticularContact.flags =
                NM_NUMI_HUMAN_ARTICULAR_CONTACT_INTERNAL_SAME_BODY;
            const std::array<NMNumiHumanArticularContactSampleGPU, 2u>
                articularContactSamples{{
                    articularContactSample, internalArticularContact}};
            NMNumiHumanFEMBodyContactSampleGPU femBodyContactSample{};
            femBodyContactSample.slaveNode = 3u;
            femBodyContactSample.bodyIndex = 0u;
            femBodyContactSample.flags =
                NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE;
            // body 0 is translated +1 mm in x below. This local plane is at
            // world x=0.5 mm, so the initial x=0 FEM node closes it by
            // 0.4999 mm after the common 0.1 um roundoff guard.
            femBodyContactSample.bodyLocalPointAndArea = {
                -0.0005f, 0.0f, 0.0f, 1.0e-4f};
            femBodyContactSample
                .bodyLocalNormalAndReferenceSeparation = {
                    1.0f, 0.0f, 0.0f, 0.0f};
            femBodyContactSample
                .stiffnessAndNormalStrainPerPressure = {
                    1.0e7f, 1.0e-6f, 0.0f, 0.0f};
            auto invalidFEMBodyContact = femBodyContactSample;
            invalidFEMBodyContact
                .bodyLocalNormalAndReferenceSeparation.x = 2.0f;
            numi::matter::NumiHumanTendonFEMLoadAdapter
                rejectedFEMBodyContactAdapter;
            require(!rejectedFEMBodyContactAdapter.initialize(runtime, {
                        .nodeLoads = nodeLoads,
                        .nodeAnchors = nodeAnchors,
                        .endpointReplacements = std::span(&replacement, 1u),
                        .contactSamples = std::span(&contactSample, 1u),
                        .contactContributions = contactContributions,
                        .contactRanges = contactRanges,
                        .femBodyContactSamples =
                            std::span(&invalidFEMBodyContact, 1u),
                        .articularContactSamples = articularContactSamples,
                        .endpointCount = 2u,
                        .environmentCount = 1u,
                        .productionForceOwnerFraction = 0.1f,
                    }, {
                        .metallib = NUMI_MATTER_METALLIB,
                    }),
                    "probe malformed FEM/body contact did not fail closed");
            NMNumiHumanPassiveRoutedBandGPU routedBand{};
            routedBand.originBodyIndex = 0u;
            routedBand.pulleyBodyIndex = 0u;
            routedBand.insertionBodyIndex = 1u;
            routedBand.flags = NM_NUMI_HUMAN_PASSIVE_ROUTED_BAND_ACTIVE;
            routedBand.originLocalPoint = {-0.04f, 0.03f, 0.0f, 0.0f};
            routedBand.pulleyLocalPoint = {0.0f, 0.03f, 0.0f, 0.0f};
            routedBand.insertionLocalPoint = {0.05f, 0.03f, 0.0f, 0.0f};
            routedBand.pulleyLocalAxisAndRadius = {
                0.0f, 0.0f, 1.0f, 0.005f};
            routedBand.neutralRelativeOrientation = {
                0.0f, 0.0f, -std::sin(0.05f), std::cos(0.05f)};
            routedBand.material = {
                14.449e6f, 254.02e6f, 10.397f, 0.4f};
            routedBand.reference = {0.086f, 14.0e-6f, 0.15f, 0.0f};
            numi::matter::NumiHumanPassiveRoutedBandEvaluation routedOracle;
            require(numi::matter::evaluateNumiHumanPassiveRoutedBand(
                        routedBand, 0.089, 0.1, routedOracle) &&
                        routedOracle.strain > 0.0 &&
                        routedOracle.tensionNewtons > 0.0 &&
                        routedOracle.storedEnergyJoules > 0.0,
                    "probe routed-band CPU oracle failed");
            numi::matter::NumiHumanPassiveRoutedBandEvaluation slackOracle;
            require(numi::matter::evaluateNumiHumanPassiveRoutedBand(
                        routedBand, 0.080, 0.0, slackOracle) &&
                        slackOracle.strain < 0.0 &&
                        slackOracle.tensionNewtons == 0.0 &&
                        slackOracle.storedEnergyJoules == 0.0,
                    "probe routed band carried compression");
            auto invalidRoutedQuaternion = routedBand;
            invalidRoutedQuaternion.neutralRelativeOrientation.w = 2.0f;
            require(!numi::matter::evaluateNumiHumanPassiveRoutedBand(
                        invalidRoutedQuaternion, 0.089, 0.1, slackOracle),
                    "probe non-unit routed-band quaternion was accepted");
            auto invalidRoutedMaterial = routedBand;
            invalidRoutedMaterial.material.z = 0.0f;
            require(!numi::matter::evaluateNumiHumanPassiveRoutedBand(
                        invalidRoutedMaterial, 0.089, 0.1, slackOracle),
                    "probe malformed routed-band material was accepted");
            require(!numi::matter::evaluateNumiHumanPassiveRoutedBand(
                        routedBand, 0.120, 0.1, slackOracle),
                    "probe routed band exceeded its strain limit");
            auto invalidRoutedBand = routedBand;
            invalidRoutedBand.pulleyLocalAxisAndRadius.z = 2.0f;
            numi::matter::NumiHumanTendonFEMLoadAdapter
                rejectedRoutedBandAdapter;
            require(!rejectedRoutedBandAdapter.initialize(runtime, {
                        .nodeLoads = nodeLoads,
                        .nodeAnchors = nodeAnchors,
                        .endpointReplacements = std::span(&replacement, 1u),
                        .contactSamples = std::span(&contactSample, 1u),
                        .contactContributions = contactContributions,
                        .contactRanges = contactRanges,
                        .femBodyContactSamples =
                            std::span(&femBodyContactSample, 1u),
                        .articularContactSamples = articularContactSamples,
                        .passiveRoutedBands =
                            std::span(&invalidRoutedBand, 1u),
                        .endpointCount = 2u,
                        .environmentCount = 1u,
                        .productionForceOwnerFraction = 0.1f,
                    }, {
                        .metallib = NUMI_MATTER_METALLIB,
                    }),
                    "probe malformed routed band did not fail closed");
            numi::matter::NumiHumanTendonFEMLoadAdapter adapter;
            require(adapter.initialize(runtime, {
                        .nodeLoads = nodeLoads,
                        .nodeAnchors = nodeAnchors,
                        .endpointReplacements = std::span(&replacement, 1u),
                        .contactSamples = std::span(&contactSample, 1u),
                        .contactContributions = contactContributions,
                        .contactRanges = contactRanges,
                        .femBodyContactSamples =
                            std::span(&femBodyContactSample, 1u),
                        .articularContactSamples = articularContactSamples,
                        .passiveRoutedBands = std::span(&routedBand, 1u),
                        .endpointCount = 2u,
                        .environmentCount = 1u,
                        .productionForceOwnerFraction = 0.1f,
                    }, {
                        .metallib = NUMI_MATTER_METALLIB,
                    }),
                    "probe tendon/FEM adapter did not initialize");
            const auto program = adapter.program();
            require(program.valid(), "probe tendon/FEM program is invalid");
            numi::matter::NumiHumanTendonFEMLoadAdapter baselineAdapter;
            require(baselineAdapter.initialize(runtime, {
                        .nodeLoads = nodeLoads,
                        .nodeAnchors = nodeAnchors,
                        .endpointReplacements = std::span(&replacement, 1u),
                        .contactSamples = std::span(&contactSample, 1u),
                        .contactContributions = contactContributions,
                        .contactRanges = contactRanges,
                        .femBodyContactSamples =
                            std::span(&femBodyContactSample, 1u),
                        .articularContactSamples =
                            std::span(&internalArticularContact, 1u),
                        .passiveRoutedBands = std::span(&routedBand, 1u),
                        .endpointCount = 2u,
                        .environmentCount = 1u,
                        .productionForceOwnerFraction = 0.1f,
                    }, {
                        .metallib = NUMI_MATTER_METALLIB,
                    }),
                    "probe baseline adapter did not initialize");
            const auto baselineProgram = baselineAdapter.program();
            require(baselineProgram.valid(),
                    "probe baseline program is invalid");

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(device != nil && queue != nil, "probe Metal queue is unavailable");
            std::vector<MRNumiHumanTendonTransferResultGPU> transfers(2u);
            for (std::uint32_t index = 0u; index < transfers.size(); ++index) {
                transfers[index].status = MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS;
                transfers[index].environment = 0u;
                transfers[index].bindingIndex = index;
                transfers[index].envelopeIndex = MR_INVALID_INDEX;
            }
            transfers[0u].terminalWorldForce = {10.0f, 0.0f, 0.0f, 0.0f};
            transfers[1u].terminalWorldForce = {-10.0f, 0.0f, 0.0f, 0.0f};
            std::array<MRNumiHumanTendonBindingGPU, 2u> bindings{};
            bindings[0u].muscleIndex = bindings[1u].muscleIndex = 0u;
            bindings[0u].endpointOrdinal = 0u;
            bindings[1u].endpointOrdinal = 1u;
            bindings[0u].bodyIndex = bindings[1u].bodyIndex = 0u;
            bindings[0u].mode = bindings[1u].mode =
                MR_NUMI_HUMAN_TENDON_TRANSFER_SOURCE_POINT;
            bindings[0u].envelopeIndex = bindings[1u].envelopeIndex =
                MR_INVALID_INDEX;
            bindings[0u].sourceLocalPoint = {0.0f, 0.0f, 0.01f, 0.0f};
            bindings[1u].sourceLocalPoint = {0.0f, 0.0f, 0.0f, 0.0f};
            id<MTLBuffer> bindingBuffer = [device
                newBufferWithBytes:bindings.data()
                length:bindings.size() * sizeof(bindings.front())
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> transferBuffer = [device
                newBufferWithBytes:transfers.data()
                length:transfers.size() * sizeof(transfers.front())
                options:MTLResourceStorageModeShared];
            MRNumiHumanStandStatusGPU stand{};
            stand.code = MR_NUMI_HUMAN_STAND_SUCCESS;
            stand.environment = 0u;
            stand.completedSteps = 1u;
            stand.failingIndex = MR_INVALID_INDEX;
            id<MTLBuffer> standBuffer = [device
                newBufferWithBytes:&stand
                length:sizeof(stand)
                options:MTLResourceStorageModeShared];
            std::array<MRArticulatedBodyPoseGPU, 2u> poses{};
            poses[0u].position.x = 0.001f;
            poses[0u].orientation = {0.0f, 0.0f, 0.0f, 1.0f};
            poses[1u].orientation = {0.0f, 0.0f, 0.0f, 1.0f};
            id<MTLBuffer> poseBuffer = [device
                newBufferWithBytes:poses.data() length:sizeof(poses)
                options:MTLResourceStorageModeShared];
            std::array<float, 24u> pointJacobians{};
            pointJacobians[0u] = pointJacobians[3u] =
                pointJacobians[6u] = pointJacobians[9u] = 1.0f;
            id<MTLBuffer> jacobianBuffer = [device
                newBufferWithBytes:pointJacobians.data()
                length:pointJacobians.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
            std::array<float, 2u> generalizedForces{};
            constexpr float sourceMuscleRow = 4.0f;
            id<MTLBuffer> generalizedForceBuffer = [device
                newBufferWithBytes:generalizedForces.data()
                length:generalizedForces.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> runtimeReactionBuffer = (__bridge id<MTLBuffer>)
                runtime.femConstraintReactionBuffer();
            id<MTLBuffer> reactionReadback = [device
                newBufferWithLength:4u * sizeof(nm_float4)
                options:MTLResourceStorageModeShared];
            require(bindingBuffer != nil && transferBuffer != nil &&
                        standBuffer != nil && poseBuffer != nil &&
                        jacobianBuffer != nil && generalizedForceBuffer != nil &&
                        runtimeReactionBuffer != nil && reactionReadback != nil,
                    "probe borrowed buffers are unavailable");

            const auto execute = [&](const auto& activeProgram,
                                     auto& activeAdapter,
                                     const std::uint32_t step,
                                     const bool accepted) {
                stand.code = accepted
                    ? MR_NUMI_HUMAN_STAND_SUCCESS
                    : MR_NUMI_HUMAN_STAND_NONFINITE_RESULT;
                stand.completedSteps = accepted ? step + 1u : step;
                std::memcpy(standBuffer.contents, &stand, sizeof(stand));
                std::memset(
                    generalizedForceBuffer.contents, 0,
                    generalizedForceBuffer.length);
                static_cast<float*>(generalizedForceBuffer.contents)[0u] =
                    sourceMuscleRow;
                id<MTLCommandBuffer> command = [queue commandBuffer];
                require(command != nil, "probe command buffer is unavailable");
                metalrobo::MetalNumiHumanTendonLoadPass pass{};
                pass.commandBuffer = (__bridge void*)command;
                pass.bindings = (__bridge void*)bindingBuffer;
                pass.transfers = (__bridge void*)transferBuffer;
                pass.generalizedForces = (__bridge void*)generalizedForceBuffer;
                pass.bodyPoses = (__bridge void*)poseBuffer;
                pass.pointJacobians = (__bridge void*)jacobianBuffer;
                pass.standStatuses = (__bridge void*)standBuffer;
                pass.stepIndex = step;
                pass.environmentCount = 1u;
                pass.endpointCount = 2u;
                pass.dofCount = 1u;
                pass.muscleCount = 1u;
                pass.generalizedForceStride = 1u;
                pass.generalizedForceOffset = 1u;
                pass.pointJacobianStride = 24u;
                pass.bodyJacobianPointOffset = 0u;
                pass.bodyPoseStride = 2u;
                pass.articulationFirstBody = 0u;
                if (!activeProgram.encodePreDynamics(
                        activeProgram.context, pass) ||
                    !activeProgram.encodePostValidation(
                        activeProgram.context, pass)) {
                    throw std::runtime_error(
                        "probe adapter rejected encoding: " +
                        activeAdapter.diagnostics().message
                    );
                }
                id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                require(blit != nil, "probe reaction blit is unavailable");
                [blit copyFromBuffer:runtimeReactionBuffer sourceOffset:0u
                            toBuffer:reactionReadback destinationOffset:0u
                                size:4u * sizeof(nm_float4)];
                [blit endEncoding];
                [command commit];
                [command waitUntilCompleted];
                require(command.status == MTLCommandBufferStatusCompleted,
                        "probe command did not complete");
            };

            execute(program, adapter, 0u, true);
            const auto accepted = runtime.snapshot();
            require(accepted.available && accepted.femNodes.size() == 4u,
                    "probe accepted snapshot is unavailable");
            const float acceptedDisplacement = std::abs(
                accepted.femNodes[3u].positionAndMass.x -
                initial.femNodes[3u].positionAndMass.x
            );
            require(std::isfinite(acceptedDisplacement) &&
                        acceptedDisplacement > 0.0f,
                    "probe tendon load did not deform the FEM node");
            const float acceptedContactDisplacement =
                accepted.femNodes[3u].positionAndMass.z -
                initial.femNodes[3u].positionAndMass.z;
            require(std::isfinite(acceptedContactDisplacement) &&
                        acceptedContactDisplacement > 0.0f,
                    "probe internal contact load did not repel the slave node");
            std::array<nm_float4, 4u> acceptedReactions{};
            std::memcpy(
                acceptedReactions.data(), reactionReadback.contents,
                acceptedReactions.size() * sizeof(acceptedReactions.front()));
            float acceptedReactionL1 = 0.0f;
            for (std::uint32_t node = 0u; node < 3u; ++node) {
                acceptedReactionL1 += std::sqrt(
                    acceptedReactions[node].x * acceptedReactions[node].x +
                    acceptedReactions[node].y * acceptedReactions[node].y +
                    acceptedReactions[node].z * acceptedReactions[node].z);
            }
            require(std::isfinite(acceptedReactionL1) &&
                        acceptedReactionL1 > 0.0f,
                    "probe FEM anchors published no bone reaction");
            float acceptedReactionX = 0.0f;
            for (std::uint32_t node = 0u; node < 3u; ++node)
                acceptedReactionX += acceptedReactions[node].x;
            const float expectedGeneralizedForce =
                0.1f * (transfers[0u].terminalWorldForce.x -
                        sourceMuscleRow) +
                acceptedReactionX + 0.9999f - 0.4999f +
                static_cast<float>(routedOracle.tensionNewtons);
            const float acceptedGeneralizedForce =
                static_cast<const float*>(
                    generalizedForceBuffer.contents)[1u];
            require(std::isfinite(acceptedGeneralizedForce) &&
                        std::abs(acceptedGeneralizedForce -
                            expectedGeneralizedForce) <= 1.0e-3f,
                    "probe full-muscle-row replacement did not preserve only the load-side reaction");
            execute(program, adapter, 1u, false);
            const auto rolledBack = runtime.snapshot();
            require(rolledBack.available &&
                        std::memcmp(
                            rolledBack.femNodes.data(), accepted.femNodes.data(),
                            accepted.femNodes.size() * sizeof(NMFEMNodeStateGPU)
                        ) == 0,
                    "probe rejected Human step did not roll Matter back");
            const auto rejectedDiagnostics = adapter.diagnostics();
            require(rejectedDiagnostics.articularAuditedStepCount == 1u &&
                        rejectedDiagnostics.articularClosedSampleCount == 1u,
                    "probe rejected Human step polluted accepted articular history");

            require(runtime.restore(initial).encoded,
                    "probe initial-state restore failed");
            execute(program, adapter, 0u, true);
            const auto replay = runtime.snapshot();
            const float replayGeneralizedForce =
                static_cast<const float*>(
                    generalizedForceBuffer.contents)[1u];
            require(replay.available &&
                        std::memcmp(
                            replay.femNodes.data(), accepted.femNodes.data(),
                            accepted.femNodes.size() * sizeof(NMFEMNodeStateGPU)
                        ) == 0 &&
                        std::memcmp(
                            reactionReadback.contents, acceptedReactions.data(),
                            acceptedReactions.size() *
                                sizeof(acceptedReactions.front())) == 0 &&
                        replayGeneralizedForce == acceptedGeneralizedForce,
                    "probe accepted tendon/FEM replay is not bitwise");
            require(runtime.restore(initial).encoded,
                    "probe baseline restore failed");
            execute(baselineProgram, baselineAdapter, 0u, true);
            const auto baseline = runtime.snapshot();
            const float baselineGeneralizedForce =
                static_cast<const float*>(
                    generalizedForceBuffer.contents)[1u];
            const float measuredArticularGeneralizedForce =
                acceptedGeneralizedForce - baselineGeneralizedForce;
            require(baseline.available &&
                        std::memcmp(
                            baseline.femNodes.data(), accepted.femNodes.data(),
                            accepted.femNodes.size() *
                                sizeof(NMFEMNodeStateGPU)) == 0 &&
                        std::isfinite(measuredArticularGeneralizedForce) &&
                        std::abs(measuredArticularGeneralizedForce - 0.9999f) <=
                            1.0e-4f,
                    "probe articular wrench A/B correction is invalid");
            const auto diagnostics = adapter.diagnostics();
            const std::string diagnosticFailure =
                "probe adapter diagnostics are incomplete: anchor_steps=" +
                std::to_string(diagnostics.anchorReactionAuditedStepCount) +
                " anchor_min_l1=" + std::to_string(
                    diagnostics.anchorReactionTrajectoryMinimumL1Newtons) +
                " anchor_max_l1=" + std::to_string(
                    diagnostics.anchorReactionTrajectoryMaximumL1Newtons) +
                " anchor_max_resultant=" + std::to_string(
                    diagnostics.anchorReactionTrajectoryMaximumResultantNewtons);
            require(diagnostics.initialized && diagnostics.encodedPassCount == 3u &&
                        diagnostics.abortCount == 0u &&
                        diagnostics.contactSampleCount == 1u &&
                        diagnostics.anchorReactionAuditedStepCount == 2u &&
                        diagnostics.anchorReactionTrajectoryMaximumL1Newtons >
                            0.0 &&
                        diagnostics.anchorReactionTrajectoryMaximumResultantNewtons >
                            0.0 &&
                        diagnostics.articularContactSampleCount == 2u &&
                        diagnostics.articularMechanicalSampleCount == 1u &&
                        diagnostics.articularInternalSameBodySampleCount == 1u &&
                        diagnostics.articularClosedSampleCount == 1u &&
                        std::abs(diagnostics.articularContactAreaSquareMeters -
                            1.0e-4) <= 1.0e-10 &&
                        std::abs(diagnostics.articularNormalForceNewtons -
                            0.9999) <= 1.0e-4 &&
                        std::abs(diagnostics.articularMaximumPressurePascals -
                            9999.0) <= 1.0 &&
                        std::abs(diagnostics.articularBodyForceL1Newtons -
                            1.9998) <= 2.0e-4 &&
                        diagnostics.articularForceResidualNewtons <= 1.0e-6 &&
                        diagnostics.articularMomentResidualNewtonMeters <=
                            1.0e-6 &&
                        diagnostics.articularAuditedStepCount == 1u &&
                        diagnostics.articularTrajectoryMinimumClosedSampleCount ==
                            1u &&
                        diagnostics.articularTrajectoryMaximumClosedSampleCount ==
                            1u &&
                        std::abs(diagnostics.articularStoredEnergyJoules -
                            0.000499900005) <= 1.0e-9 &&
                        std::abs(diagnostics.articularMaximumNormalStrain -
                            0.009999) <= 1.0e-6 &&
                        std::abs(diagnostics.articularMaximumClosureMeters -
                            0.0009999) <= 1.0e-7 &&
                        std::abs(
                            diagnostics.articularTrajectoryMinimumNormalForceNewtons -
                            0.9999) <= 1.0e-4 &&
                        std::abs(
                            diagnostics.articularTrajectoryMaximumNormalForceNewtons -
                            0.9999) <= 1.0e-4 &&
                        std::abs(
                            diagnostics.articularTrajectoryMaximumPressurePascals -
                            9999.0) <= 1.0 &&
                        std::abs(
                            diagnostics.articularTrajectoryMaximumStoredEnergyJoules -
                            0.000499900005) <= 1.0e-9 &&
                        std::abs(
                            diagnostics.articularTrajectoryMaximumNormalStrain -
                            0.009999) <= 1.0e-6 &&
                        std::abs(
                            diagnostics.articularTrajectoryMaximumClosureMeters -
                            0.0009999) <= 1.0e-7 &&
                        diagnostics.articularTrajectoryMaximumForceResidualNewtons <=
                            1.0e-6 &&
                        diagnostics.articularTrajectoryMaximumMomentResidualNewtonMeters <=
                            1.0e-6 &&
                        diagnostics.femBodyContactSampleCount == 1u &&
                        diagnostics.femBodyContactClosedSampleCount == 1u &&
                        diagnostics.femBodyContactAuditedStepCount == 1u &&
                        std::abs(
                            diagnostics.femBodyContactNormalForceNewtons -
                            0.4999) <= 1.0e-4 &&
                        std::abs(
                            diagnostics.femBodyContactMaximumPressurePascals -
                            4999.0) <= 1.0 &&
                        diagnostics.femBodyContactForceResidualNewtons <=
                            1.0e-6 &&
                        diagnostics
                            .femBodyContactMomentResidualNewtonMeters <=
                            1.0e-6 &&
                        std::abs(
                            diagnostics.femBodyContactStoredEnergyJoules -
                            0.000124950005) <= 1.0e-9 &&
                        std::abs(
                            diagnostics.femBodyContactMaximumNormalStrain -
                            0.004999) <= 1.0e-6 &&
                        std::abs(
                            diagnostics.femBodyContactMaximumClosureMeters -
                            0.0004999) <= 1.0e-7 &&
                        std::abs(
                            diagnostics
                                .femBodyContactMaximumTangentialSlipMeters -
                            0.01) <= 1.0e-6 &&
                        diagnostics.passiveRoutedBandCount == 1u &&
                        diagnostics
                            .passiveRoutedBandLatestTransactionAccepted &&
                        diagnostics.passiveRoutedBandMaximumTensionNewtons >
                            0.0 &&
                        diagnostics.passiveRoutedBandEndpointForceL1Newtons >
                            0.0 &&
                        std::abs(
                            diagnostics.passiveRoutedBandMaximumTensionNewtons -
                            routedOracle.tensionNewtons) <= 1.0e-3 &&
                        std::abs(
                            diagnostics.passiveRoutedBandMinimumStrain -
                            routedOracle.strain) <= 1.0e-6 &&
                        std::abs(
                            diagnostics.passiveRoutedBandMaximumStrain -
                            routedOracle.strain) <= 1.0e-6 &&
                        diagnostics.passiveRoutedBandForceResidualNewtons <=
                            1.0e-5 &&
                        diagnostics
                            .passiveRoutedBandMomentResidualNewtonMeters <=
                            1.0e-5 &&
                        std::abs(
                            diagnostics.passiveRoutedBandStoredEnergyJoules -
                            routedOracle.storedEnergyJoules) <= 1.0e-5 &&
                        std::abs(
                            diagnostics.passiveRoutedBandMaximumExtensionMeters -
                            (routedOracle.routeLengthMeters - 0.086)) <=
                            1.0e-6 &&
                        diagnostics.fingerprint != 0u,
                    diagnosticFailure.c_str());
            std::cout
                << "numi_human_tendon_fem_load=passed"
                << " device=\"" << initialized.device << "\""
                << " encoded_passes=" << diagnostics.encodedPassCount
                << " max_displacement_m=" << acceptedDisplacement
                << " contact_displacement_m="
                << acceptedContactDisplacement
                << " contact_samples=" << diagnostics.contactSampleCount
                << " articular_contact_samples="
                << diagnostics.articularContactSampleCount
                << " articular_mechanical_samples="
                << diagnostics.articularMechanicalSampleCount
                << " articular_internal_same_body_samples="
                << diagnostics.articularInternalSameBodySampleCount
                << " articular_closed_samples="
                << diagnostics.articularClosedSampleCount
                << " articular_contact_area_m2="
                << diagnostics.articularContactAreaSquareMeters
                << " articular_normal_force_n="
                << diagnostics.articularNormalForceNewtons
                << " articular_max_pressure_pa="
                << diagnostics.articularMaximumPressurePascals
                << " articular_body_force_l1_n="
                << diagnostics.articularBodyForceL1Newtons
                << " articular_force_residual_n="
                << diagnostics.articularForceResidualNewtons
                << " articular_moment_residual_nm="
                << diagnostics.articularMomentResidualNewtonMeters
                << " articular_stored_energy_j="
                << diagnostics.articularStoredEnergyJoules
                << " articular_max_normal_strain="
                << diagnostics.articularMaximumNormalStrain
                << " articular_max_closure_m="
                << diagnostics.articularMaximumClosureMeters
                << " articular_audited_steps="
                << diagnostics.articularAuditedStepCount
                << " articular_trajectory_min_closed_samples="
                << diagnostics.articularTrajectoryMinimumClosedSampleCount
                << " articular_trajectory_max_closed_samples="
                << diagnostics.articularTrajectoryMaximumClosedSampleCount
                << " rejected_step_excluded_from_history=true"
                << " articular_contact_generalized_force="
                << measuredArticularGeneralizedForce
                << " articular_contact_fem_state_ab=bitwise"
                << " malformed_contact_rejected=true"
                << " malformed_articular_contact_rejected=true"
                << " malformed_articular_reference_normal_rejected=true"
                << " malformed_articular_adjacency_rejected=true"
                << " malformed_fem_body_contact_rejected=true"
                << " malformed_routed_band_rejected=true"
                << " routed_band_compression_gate=zero"
                << " routed_band_quaternion_gate=fail_closed"
                << " routed_band_material_gate=fail_closed"
                << " routed_band_strain_limit=fail_closed"
                << " fem_body_contact_samples="
                << diagnostics.femBodyContactSampleCount
                << " fem_body_contact_closed_samples="
                << diagnostics.femBodyContactClosedSampleCount
                << " fem_body_contact_normal_force_n="
                << diagnostics.femBodyContactNormalForceNewtons
                << " fem_body_contact_force_residual_n="
                << diagnostics.femBodyContactForceResidualNewtons
                << " fem_body_contact_moment_residual_nm="
                << diagnostics.femBodyContactMomentResidualNewtonMeters
                << " fem_body_contact_max_tangent_glide_m="
                << diagnostics.femBodyContactMaximumTangentialSlipMeters
                << " passive_routed_band_count="
                << diagnostics.passiveRoutedBandCount
                << " passive_routed_band_tension_n="
                << diagnostics.passiveRoutedBandMaximumTensionNewtons
                << " passive_routed_band_strain="
                << diagnostics.passiveRoutedBandMaximumStrain
                << " passive_routed_band_energy_j="
                << diagnostics.passiveRoutedBandStoredEnergyJoules
                << " passive_routed_band_force_residual_n="
                << diagnostics.passiveRoutedBandForceResidualNewtons
                << " passive_routed_band_moment_residual_nm="
                << diagnostics.passiveRoutedBandMomentResidualNewtonMeters
                << " anchor_reaction_l1_n=" << acceptedReactionL1
                << " audited_anchor_reaction_min_l1_n="
                << diagnostics.anchorReactionTrajectoryMinimumL1Newtons
                << " audited_anchor_reaction_max_l1_n="
                << diagnostics.anchorReactionTrajectoryMaximumL1Newtons
                << " full_row_generalized_force=" << acceptedGeneralizedForce
                << " replay=bitwise rollback=verified"
                << " production_owner_fraction=0.1"
                << "\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "numi_human_tendon_fem_load=failed error=\""
                      << error.what() << "\"\n";
            return 1;
        }
    }
}
