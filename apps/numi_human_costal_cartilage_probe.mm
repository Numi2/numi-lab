#include "metalrobo/NumiHumanCartilage.hpp"
#include "numi/matter/matter.hpp"
#include "numi/matter/numi_human.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Vec3 = std::array<double, 3u>;

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::byte> readBytes(const char* path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("costal-cartilage payload did not open");
    const auto end = stream.tellg();
    if (end <= 0) throw std::runtime_error("costal-cartilage payload is empty");
    std::vector<std::byte> bytes(static_cast<std::size_t>(end));
    stream.seekg(0, std::ios::beg);
    stream.read(reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    if (!stream) throw std::runtime_error("costal-cartilage payload read failed");
    return bytes;
}

Vec3 subtract(const Vec3& a, const Vec3& b) {
    return {a[0u] - b[0u], a[1u] - b[1u], a[2u] - b[2u]};
}

double norm(const Vec3& value) {
    return std::sqrt(value[0u] * value[0u] + value[1u] * value[1u] +
                     value[2u] * value[2u]);
}

double signedVolume(const Vec3& a, const Vec3& b,
                    const Vec3& c, const Vec3& d) {
    const Vec3 ab = subtract(b, a);
    const Vec3 ac = subtract(c, a);
    const Vec3 ad = subtract(d, a);
    return (
        ab[0u] * (ac[1u] * ad[2u] - ac[2u] * ad[1u]) -
        ab[1u] * (ac[0u] * ad[2u] - ac[2u] * ad[0u]) +
        ab[2u] * (ac[0u] * ad[1u] - ac[1u] * ad[0u])
    ) / 6.0;
}

Vec3 position(const NMFEMNodeStateGPU& node) {
    return {node.positionAndMass.x, node.positionAndMass.y,
            node.positionAndMass.z};
}

constexpr std::array<std::uint8_t, 32u> expectedArchiveSha256{
    0x9f, 0xbc, 0x71, 0x3f, 0xff, 0xee, 0xe9, 0x24,
    0xa5, 0xa6, 0x57, 0xd9, 0x81, 0x3d, 0x84, 0xd7,
    0xeb, 0x95, 0x7b, 0xde, 0xd6, 0x3a, 0xdb, 0x85,
    0x49, 0x31, 0xdd, 0x5e, 0x3e, 0xb6, 0x1c, 0x97,
};

} // namespace

int main(const int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            require(argc == 2, "usage: probe COSTAL_CARTILAGE_PAYLOAD");
            const std::vector<std::byte> bytes = readBytes(argv[1]);
            metalrobo::NumiHumanCostalCartilagePayload payload;
            const auto decoded = metalrobo::decodeNumiHumanCostalCartilagePayload(
                bytes, expectedArchiveSha256, payload);
            if (!decoded.succeeded()) {
                throw std::runtime_error(
                    std::string("costal-cartilage payload rejected: ") +
                    metalrobo::numiHumanCartilageStatusName(decoded.status));
            }

            numi::matter::WorldSource source;
            source.environmentCount = 1u;
            source.frameTimestep = 1.0e-4;
            source.gravity = {0.0, 0.0, 0.0};
            // This one-step structural gate starts at identity with a small
            // load. Keep fixed solver work bounded so atlas-scale validation
            // remains practical on Apple Silicon while still exercising the
            // nonlinear material and accepted transaction.
            source.mixedSolver.newtonIterations = 3u;
            source.mixedSolver.fgmresRestart = 8u;
            source.mixedSolver.fgmresIterations = 12u;
            source.mixedSolver.lineSearchSteps = 4u;
            auto material = numi::matter::parseMatterFile(
                NUMI_HUMAN_COSTAL_CARTILAGE_MATERIAL);
            require(material.succeeded(), "costal-cartilage material did not parse");
            source.materials.push_back(std::move(material.material));

            numi::matter::ObjectSource object;
            object.name = "numi_human_exact_costal_cartilage_v2";
            object.materialIndex = 0u;
            object.representation = numi::matter::Representation::fem;
            object.mixedFEM = false;
            object.deformableSelfContact = false;
            object.characteristicLength = 0.0025;
            object.femNodes.reserve(payload.nodes.size());
            std::vector<NMNumiHumanTendonFEMNodeLoadGPU> nodeLoads(
                payload.nodes.size());
            std::vector<NMNumiHumanTendonFEMNodeAnchorGPU> nodeAnchors(
                payload.nodes.size());
            for (std::size_t index = 0u; index < payload.nodes.size(); ++index) {
                const auto& node = payload.nodes[index];
                object.femNodes.push_back({node.restPosition[0u],
                                           node.restPosition[1u],
                                           node.restPosition[2u]});
                std::fill_n(
                    nodeLoads[index].endpointIndex, 4u, NM_INVALID_INDEX);
                nodeAnchors[index].bodyIndex = NM_INVALID_INDEX;
                if ((node.flags &
                     metalrobo::NUMI_HUMAN_COSTAL_CARTILAGE_STERNAL_ATTACHMENT) != 0u) {
                    object.femFixedNodes.push_back(static_cast<std::uint32_t>(index));
                    nodeAnchors[index].bodyIndex = 0u;
                    nodeAnchors[index].flags =
                        NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE;
                    nodeAnchors[index].localPoint = {
                        node.restPosition[0u], node.restPosition[1u],
                        node.restPosition[2u], 0.0f};
                } else if ((node.flags &
                     metalrobo::NUMI_HUMAN_COSTAL_CARTILAGE_RIB_ATTACHMENT) != 0u) {
                    object.femFixedNodes.push_back(static_cast<std::uint32_t>(index));
                    nodeAnchors[index].bodyIndex = 1u;
                    nodeAnchors[index].flags =
                        NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE;
                    nodeAnchors[index].localPoint = {
                        node.restPosition[0u], node.restPosition[1u],
                        node.restPosition[2u], 0.0f};
                }
            }
            object.femContactNodes.reserve(payload.regions.size());
            for (const auto& region : payload.regions)
                object.femContactNodes.push_back(region.firstNode);
            object.tetrahedra.reserve(payload.tetrahedra.size());
            for (const auto& tetrahedron : payload.tetrahedra)
                object.tetrahedra.push_back({tetrahedron.node});
            const std::size_t fixedNodeCount = object.femFixedNodes.size();
            const std::size_t sternalFixedNodeCount = std::count_if(
                payload.nodes.begin(), payload.nodes.end(), [](const auto& node) {
                    return (node.flags & metalrobo::
                        NUMI_HUMAN_COSTAL_CARTILAGE_STERNAL_ATTACHMENT) != 0u;
                });
            const std::size_t ribFixedNodeCount = std::count_if(
                payload.nodes.begin(), payload.nodes.end(), [](const auto& node) {
                    return (node.flags & metalrobo::
                        NUMI_HUMAN_COSTAL_CARTILAGE_RIB_ATTACHMENT) != 0u;
                });
            require(fixedNodeCount == sternalFixedNodeCount + ribFixedNodeCount &&
                        sternalFixedNodeCount > 0u && ribFixedNodeCount > 0u,
                    "costal-cartilage two-sided attachment coverage is invalid");
            source.objects.push_back(std::move(object));

            constexpr std::size_t endpointCount = 1u;
            std::vector<NMNumiHumanTendonFEMEndpointReplacementGPU> replacements;
            std::vector<MRNumiHumanTendonTransferResultGPU> transfers(endpointCount);
            std::vector<MRNumiHumanTendonBindingGPU> bindings(endpointCount);
            for (std::size_t regionIndex = 0u;
                 regionIndex < payload.regions.size(); ++regionIndex) {
                const auto& region = payload.regions[regionIndex];
                Vec3 sternalCentroid{};
                Vec3 ribCentroid{};
                std::uint32_t sternalCount = 0u;
                std::uint32_t ribCount = 0u;
                for (std::uint32_t local = 0u; local < region.nodeCount; ++local) {
                    const std::uint32_t index = region.firstNode + local;
                    const auto& node = payload.nodes[index];
                    if ((node.flags & metalrobo::
                         NUMI_HUMAN_COSTAL_CARTILAGE_STERNAL_ATTACHMENT) != 0u) {
                        for (std::size_t axis = 0u; axis < 3u; ++axis)
                            sternalCentroid[axis] += node.restPosition[axis];
                        ++sternalCount;
                    }
                    if ((node.flags & metalrobo::
                         NUMI_HUMAN_COSTAL_CARTILAGE_RIB_ATTACHMENT) != 0u) {
                        for (std::size_t axis = 0u; axis < 3u; ++axis)
                            ribCentroid[axis] += node.restPosition[axis];
                        ++ribCount;
                    }
                }
                require(sternalCount == region.sternalAttachmentNodeCount &&
                            ribCount == region.ribAttachmentNodeCount,
                        "costal-cartilage runtime attachment coverage drifted");
                for (std::size_t axis = 0u; axis < 3u; ++axis) {
                    sternalCentroid[axis] /= sternalCount;
                    ribCentroid[axis] /= ribCount;
                }
                require(norm(subtract(ribCentroid, sternalCentroid)) > 1.0e-6,
                        "costal-cartilage attachment centroids coincide");
            }

            numi::matter::CompileOptions compileOptions;
            compileOptions.maximumRateExponent = 0u;
            auto compiled = numi::matter::compileWorld(source, compileOptions);
            if (!compiled.succeeded()) {
                std::string message = "costal-cartilage FEM world did not compile";
                for (const auto& diagnostic : compiled.diagnostics)
                    message += "; " + diagnostic.message;
                throw std::runtime_error(message);
            }
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
                    "costal-cartilage FEM runtime did not initialize");
            const auto initial = runtime.snapshot();
            require(initial.available && initial.femNodes.size() == payload.nodes.size(),
                    "costal-cartilage initial snapshot is unavailable");

            numi::matter::NumiHumanTendonFEMLoadAdapter adapter;
            require(adapter.initialize(runtime, {
                        .nodeLoads = nodeLoads,
                        .nodeAnchors = nodeAnchors,
                        .endpointReplacements = replacements,
                        .endpointCount = static_cast<std::uint32_t>(endpointCount),
                        .environmentCount = 1u,
                        .productionForceOwnerFraction = 0.0f,
                    }, {.metallib = NUMI_MATTER_METALLIB}),
                    "costal-cartilage passive two-way adapter did not initialize");
            const auto program = adapter.program();
            require(program.valid(), "costal-cartilage two-way program is invalid");

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(device != nil && queue != nil,
                    "costal-cartilage Metal queue is unavailable");
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
                newBufferWithBytes:&stand length:sizeof(stand)
                options:MTLResourceStorageModeShared];
            // A one-micron inter-body motion is deliberately small: this gate
            // qualifies attachment/reaction ownership in one 0.1 ms step, not
            // a physiological breathing-amplitude or static-load calibration.
            constexpr float prescribedRibTranslationM = 1.0e-6f;
            std::array<MRArticulatedBodyPoseGPU, 2u> poses{};
            poses[0u].orientation = {0.0f, 0.0f, 0.0f, 1.0f};
            poses[1u].position = {prescribedRibTranslationM, 0.0f, 0.0f, 0.0f};
            poses[1u].orientation = {0.0f, 0.0f, 0.0f, 1.0f};
            id<MTLBuffer> poseBuffer = [device
                newBufferWithBytes:poses.data()
                length:poses.size() * sizeof(poses.front())
                options:MTLResourceStorageModeShared];
            constexpr std::uint32_t dofCount = 6u;
            constexpr std::uint32_t pointJacobianStride = 4u * 2u * 3u * dofCount;
            std::array<float, pointJacobianStride> pointJacobians{};
            for (std::uint32_t body = 0u; body < 2u; ++body) {
                for (std::uint32_t point = 0u; point < 4u; ++point) {
                    const std::uint32_t base =
                        (4u * body + point) * 3u * dofCount;
                    pointJacobians[base + 3u * body] = 1.0f;
                    pointJacobians[base + dofCount + 3u * body + 1u] = 1.0f;
                    pointJacobians[base + 2u * dofCount + 3u * body + 2u] = 1.0f;
                }
            }
            id<MTLBuffer> jacobianBuffer = [device
                newBufferWithBytes:pointJacobians.data()
                length:pointJacobians.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
            std::array<float, 2u * dofCount> generalizedForces{};
            id<MTLBuffer> generalizedForceBuffer = [device
                newBufferWithBytes:generalizedForces.data()
                length:generalizedForces.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> runtimeReactionBuffer = (__bridge id<MTLBuffer>)
                runtime.femConstraintReactionBuffer();
            id<MTLBuffer> reactionReadback = [device
                newBufferWithLength:payload.nodes.size() * sizeof(nm_float4)
                options:MTLResourceStorageModeShared];
            require(bindingBuffer != nil && transferBuffer != nil &&
                        standBuffer != nil && poseBuffer != nil &&
                        jacobianBuffer != nil && generalizedForceBuffer != nil &&
                        runtimeReactionBuffer != nil && reactionReadback != nil,
                    "costal-cartilage borrowed buffers are unavailable");

            const auto execute = [&](const std::uint32_t step,
                                     const bool accepted) {
                stand.code = accepted ? MR_NUMI_HUMAN_STAND_SUCCESS
                                      : MR_NUMI_HUMAN_STAND_NONFINITE_RESULT;
                stand.completedSteps = accepted ? step + 1u : step;
                std::memcpy(standBuffer.contents, &stand, sizeof(stand));
                std::memset(generalizedForceBuffer.contents, 0,
                            generalizedForceBuffer.length);
                id<MTLCommandBuffer> command = [queue commandBuffer];
                require(command != nil,
                        "costal-cartilage command buffer is unavailable");
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
                pass.endpointCount = static_cast<std::uint32_t>(endpointCount);
                pass.dofCount = dofCount;
                pass.muscleCount = 1u;
                pass.generalizedForceStride = 2u * dofCount;
                pass.generalizedForceOffset = dofCount;
                pass.pointJacobianStride = pointJacobianStride;
                pass.bodyJacobianPointOffset = 0u;
                pass.bodyPoseStride = 2u;
                pass.articulationFirstBody = 0u;
                if (!program.encodePreDynamics(program.context, pass) ||
                    !program.encodePostValidation(program.context, pass)) {
                    throw std::runtime_error(
                        "costal-cartilage transaction rejected: " +
                        adapter.diagnostics().message);
                }
                id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                require(blit != nil,
                        "costal-cartilage reaction blit is unavailable");
                [blit copyFromBuffer:runtimeReactionBuffer sourceOffset:0u
                            toBuffer:reactionReadback destinationOffset:0u
                                size:payload.nodes.size() * sizeof(nm_float4)];
                [blit endEncoding];
                [command commit];
                [command waitUntilCompleted];
                require(command.status == MTLCommandBufferStatusCompleted,
                        "costal-cartilage command did not complete");
            };

            execute(0u, true);
            std::array<float, 2u * dofCount> acceptedGeneralizedForces{};
            std::memcpy(acceptedGeneralizedForces.data(),
                        generalizedForceBuffer.contents,
                        generalizedForceBuffer.length);
            double sternalBodyGeneralizedForceL1 = 0.0;
            double ribBodyGeneralizedForceL1 = 0.0;
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                sternalBodyGeneralizedForceL1 += std::abs(
                    acceptedGeneralizedForces[dofCount + axis]);
                ribBodyGeneralizedForceL1 += std::abs(
                    acceptedGeneralizedForces[dofCount + 3u + axis]);
            }
            require(std::isfinite(sternalBodyGeneralizedForceL1) &&
                        std::isfinite(ribBodyGeneralizedForceL1) &&
                        sternalBodyGeneralizedForceL1 > 0.0 &&
                        ribBodyGeneralizedForceL1 > 0.0,
                    "costal-cartilage body reaction scatter is incomplete");
            const auto accepted = runtime.snapshot();
            require(accepted.available && accepted.femNodes.size() == payload.nodes.size(),
                    "costal-cartilage accepted snapshot is unavailable");
            require(accepted.statuses.size() == 1u,
                    "costal-cartilage Matter status is unavailable");
            std::cerr << "costal_cartilage_status_preflight"
                      << " code=" << accepted.statuses[0u].code
                      << " object=" << accepted.statuses[0u].objectIndex
                      << " failing_index=" << accepted.statuses[0u].failingIndex
                      << " completed_microsteps="
                      << accepted.statuses[0u].completedMicrosteps
                      << " residual=" << accepted.statuses[0u].diagnostics.z
                      << " correction=" << accepted.statuses[0u].diagnostics.w
                      << "\n";
            require(accepted.statuses[0u].code == NM_STATUS_SUCCESS,
                    "costal-cartilage Matter step was rejected");
            double maximumDisplacement = 0.0;
            std::vector<double> regionMaximumDisplacement(payload.regions.size(), 0.0);
            for (std::size_t index = 0u; index < accepted.femNodes.size(); ++index) {
                const double displacement = norm(subtract(
                    position(accepted.femNodes[index]),
                    position(initial.femNodes[index])));
                require(std::isfinite(displacement),
                        "costal-cartilage displacement is nonfinite");
                maximumDisplacement = std::max(maximumDisplacement, displacement);
                regionMaximumDisplacement[payload.nodes[index].regionIndex] = std::max(
                    regionMaximumDisplacement[payload.nodes[index].regionIndex],
                    displacement);
            }
            const double minimumRegionDisplacement = *std::min_element(
                regionMaximumDisplacement.begin(), regionMaximumDisplacement.end());
            std::cerr << "costal_cartilage_deformation_preflight"
                      << " maximum_m=" << maximumDisplacement
                      << " minimum_region_maximum_m=" << minimumRegionDisplacement
                      << "\n";
            require(maximumDisplacement > 0.0 && maximumDisplacement < 0.01 &&
                        minimumRegionDisplacement > 0.0,
                    "costal-cartilage deformation coverage is invalid");
            double minimumJ = std::numeric_limits<double>::infinity();
            double maximumJ = 0.0;
            for (const auto& tetrahedron : payload.tetrahedra) {
                const auto& n = tetrahedron.node;
                const double restVolume = signedVolume(
                    position(initial.femNodes[n[0u]]), position(initial.femNodes[n[1u]]),
                    position(initial.femNodes[n[2u]]), position(initial.femNodes[n[3u]]));
                const double currentVolume = signedVolume(
                    position(accepted.femNodes[n[0u]]), position(accepted.femNodes[n[1u]]),
                    position(accepted.femNodes[n[2u]]), position(accepted.femNodes[n[3u]]));
                const double j = currentVolume / restVolume;
                require(std::isfinite(j), "costal-cartilage determinant is nonfinite");
                minimumJ = std::min(minimumJ, j);
                maximumJ = std::max(maximumJ, j);
            }
            require(minimumJ >= 0.50 && maximumJ <= 1.75,
                    "costal-cartilage determinant left material validity range");
            std::vector<nm_float4> acceptedReactions(payload.nodes.size());
            std::memcpy(acceptedReactions.data(), reactionReadback.contents,
                        acceptedReactions.size() * sizeof(acceptedReactions.front()));
            double sternalReactionL1 = 0.0;
            double ribReactionL1 = 0.0;
            std::vector<double> regionSternalReaction(payload.regions.size(), 0.0);
            std::vector<double> regionRibReaction(payload.regions.size(), 0.0);
            for (std::size_t index = 0u; index < payload.nodes.size(); ++index) {
                const Vec3 reaction{acceptedReactions[index].x,
                                    acceptedReactions[index].y,
                                    acceptedReactions[index].z};
                const double magnitude = norm(reaction);
                require(std::isfinite(magnitude),
                        "costal-cartilage reaction is nonfinite");
                if ((payload.nodes[index].flags & metalrobo::
                     NUMI_HUMAN_COSTAL_CARTILAGE_STERNAL_ATTACHMENT) != 0u) {
                    sternalReactionL1 += magnitude;
                    regionSternalReaction[payload.nodes[index].regionIndex] += magnitude;
                } else if ((payload.nodes[index].flags & metalrobo::
                     NUMI_HUMAN_COSTAL_CARTILAGE_RIB_ATTACHMENT) != 0u) {
                    ribReactionL1 += magnitude;
                    regionRibReaction[payload.nodes[index].regionIndex] += magnitude;
                }
            }
            require(sternalReactionL1 > 0.0 && ribReactionL1 > 0.0 &&
                        std::all_of(regionSternalReaction.begin(),
                                    regionSternalReaction.end(),
                                    [](const double value) { return value > 0.0; }) &&
                        std::all_of(regionRibReaction.begin(),
                                    regionRibReaction.end(),
                                    [](const double value) { return value > 0.0; }),
                    "costal-cartilage two-sided reactions are incomplete");

            execute(1u, false);
            const auto rolledBack = runtime.snapshot();
            require(rolledBack.available &&
                        std::memcmp(rolledBack.femNodes.data(), accepted.femNodes.data(),
                                    accepted.femNodes.size() * sizeof(NMFEMNodeStateGPU)) == 0,
                    "costal-cartilage rejected step did not roll back");
            require(runtime.restore(initial).encoded,
                    "costal-cartilage initial restore failed");
            execute(0u, true);
            const auto replay = runtime.snapshot();
            require(replay.available &&
                        std::memcmp(replay.femNodes.data(), accepted.femNodes.data(),
                                    accepted.femNodes.size() * sizeof(NMFEMNodeStateGPU)) == 0 &&
                        std::memcmp(reactionReadback.contents, acceptedReactions.data(),
                                    acceptedReactions.size() * sizeof(acceptedReactions.front())) == 0 &&
                        std::memcmp(generalizedForceBuffer.contents,
                                    acceptedGeneralizedForces.data(),
                                    generalizedForceBuffer.length) == 0,
                    "costal-cartilage accepted replay is not bitwise");
            const auto diagnostics = adapter.diagnostics();
            require(diagnostics.initialized && diagnostics.encodedPassCount == 3u &&
                        diagnostics.abortCount == 0u && diagnostics.fingerprint != 0u,
                    "costal-cartilage adapter diagnostics are incomplete");
            std::cout
                << "numi_human_costal_cartilage=passed"
                << " device=\"" << initialized.device << "\""
                << " source_regions=" << payload.regions.size()
                << " fem_nodes=" << payload.nodes.size()
                << " tetrahedra=" << payload.tetrahedra.size()
                << " sternal_fixed_nodes=" << sternalFixedNodeCount
                << " rib_fixed_nodes=" << ribFixedNodeCount
                << " prescribed_rib_translation_m=" << prescribedRibTranslationM
                << " max_displacement_m=" << maximumDisplacement
                << " min_J=" << minimumJ
                << " max_J=" << maximumJ
                << " sternal_reaction_l1_n=" << sternalReactionL1
                << " rib_reaction_l1_n=" << ribReactionL1
                << " sternal_body_generalized_force_l1_n="
                << sternalBodyGeneralizedForceL1
                << " rib_body_generalized_force_l1_n="
                << ribBodyGeneralizedForceL1
                << " replay=bitwise rollback=verified"
                << " production_owner_fraction=0"
                << "\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "numi_human_costal_cartilage=failed error=\""
                      << error.what() << "\"\n";
            return 1;
        }
    }
}
