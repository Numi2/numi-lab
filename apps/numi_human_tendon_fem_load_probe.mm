#include "numi/matter/matter.hpp"
#include "numi/matter/numi_human.hpp"

#import <Metal/Metal.h>

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
                load.endpointIndex = NM_INVALID_INDEX;
            }
            nodeLoads[3u].endpointIndex = 0u;
            nodeLoads[3u].flags = NM_NUMI_HUMAN_TENDON_FEM_NODE_LOAD_ACTIVE;
            nodeLoads[3u].scale.x = 0.1f;
            numi::matter::NumiHumanTendonFEMLoadAdapter adapter;
            require(adapter.initialize(runtime, {
                        .nodeLoads = nodeLoads,
                        .endpointCount = 2u,
                        .environmentCount = 1u,
                        .productionForceOwnerFraction = 0.0f,
                    }, {
                        .metallib = NUMI_MATTER_METALLIB,
                    }),
                    "probe tendon/FEM adapter did not initialize");
            const auto program = adapter.program();
            require(program.valid(), "probe tendon/FEM program is invalid");

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
            require(transferBuffer != nil && standBuffer != nil,
                    "probe borrowed buffers are unavailable");

            const auto execute = [&](const std::uint32_t step,
                                     const bool accepted) {
                stand.code = accepted
                    ? MR_NUMI_HUMAN_STAND_SUCCESS
                    : MR_NUMI_HUMAN_STAND_NONFINITE_RESULT;
                stand.completedSteps = accepted ? step + 1u : step;
                std::memcpy(standBuffer.contents, &stand, sizeof(stand));
                id<MTLCommandBuffer> command = [queue commandBuffer];
                require(command != nil, "probe command buffer is unavailable");
                metalrobo::MetalNumiHumanTendonLoadPass pass{};
                pass.commandBuffer = (__bridge void*)command;
                pass.transfers = (__bridge void*)transferBuffer;
                pass.standStatuses = (__bridge void*)standBuffer;
                pass.stepIndex = step;
                pass.environmentCount = 1u;
                pass.endpointCount = 2u;
                pass.dofCount = 1u;
                if (!program.encode(program.context, pass)) {
                    throw std::runtime_error(
                        "probe adapter rejected encoding: " +
                        adapter.diagnostics().message
                    );
                }
                [command commit];
                [command waitUntilCompleted];
                require(command.status == MTLCommandBufferStatusCompleted,
                        "probe command did not complete");
            };

            execute(0u, true);
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
            execute(1u, false);
            const auto rolledBack = runtime.snapshot();
            require(rolledBack.available &&
                        std::memcmp(
                            rolledBack.femNodes.data(), accepted.femNodes.data(),
                            accepted.femNodes.size() * sizeof(NMFEMNodeStateGPU)
                        ) == 0,
                    "probe rejected Human step did not roll Matter back");

            require(runtime.restore(initial).encoded,
                    "probe initial-state restore failed");
            execute(0u, true);
            const auto replay = runtime.snapshot();
            require(replay.available &&
                        std::memcmp(
                            replay.femNodes.data(), accepted.femNodes.data(),
                            accepted.femNodes.size() * sizeof(NMFEMNodeStateGPU)
                        ) == 0,
                    "probe accepted tendon/FEM replay is not bitwise");
            const auto diagnostics = adapter.diagnostics();
            require(diagnostics.initialized && diagnostics.encodedPassCount == 3u &&
                        diagnostics.abortCount == 0u && diagnostics.fingerprint != 0u,
                    "probe adapter diagnostics are incomplete");
            std::cout
                << "numi_human_tendon_fem_load=passed"
                << " device=\"" << initialized.device << "\""
                << " encoded_passes=" << diagnostics.encodedPassCount
                << " max_displacement_m=" << acceptedDisplacement
                << " replay=bitwise rollback=verified"
                << " production_owner_fraction=0"
                << "\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "numi_human_tendon_fem_load=failed error=\""
                      << error.what() << "\"\n";
            return 1;
        }
    }
}
