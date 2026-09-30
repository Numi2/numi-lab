#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#define NUMI_MATTER_METALLIB ""
#endif

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main(int argc, char** argv) {
    try {
        require((argc == 4 || argc == 5) &&
                (argc == 4 || std::string(argv[4]) == "--zero"),
                "usage: ventricular-source-step cooked.nmpkg "
                "cooked-tension.f32le accepted-nodes.bin [--zero]");
        const bool zeroInput = argc == 5;
        numi::matter::CompiledWorld world;
        std::string packageError;
        require(numi::matter::readPackage(argv[1], world, nullptr, &packageError),
                "cannot load full ventricular package: " + packageError);
        require(world.dispatch.environmentCount == 1u &&
                world.fem.tetrahedra.size() == 1097534u &&
                world.fem.nodes.size() == 218077u,
                "unexpected full source ventricular cooked dimensions");
        const std::filesystem::path tensionPath = argv[2];
        const std::size_t bytes = world.fem.tetrahedra.size() * sizeof(float);
        require(std::filesystem::is_regular_file(tensionPath) &&
                std::filesystem::file_size(tensionPath) == bytes,
                "cooked tension file does not cover every cell");
        std::vector<float> tensions(world.fem.tetrahedra.size());
        std::ifstream stream(tensionPath, std::ios::binary);
        stream.read(reinterpret_cast<char*>(tensions.data()),
                    static_cast<std::streamsize>(bytes));
        require(stream.good(), "cannot read cooked tension field");
        if (zeroInput) std::fill(tensions.begin(), tensions.end(), 0.0f);

        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "no Metal device");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLBuffer> tensionBuffer = [device
                newBufferWithBytes:tensions.data() length:bytes
                options:MTLResourceStorageModeShared];
            MRMetalWorldStatusGPU worldStatus{};
            worldStatus.code = MR_STEP_SUCCESS;
            id<MTLBuffer> statusBuffer = [device
                newBufferWithBytes:&worldStatus length:sizeof(worldStatus)
                options:MTLResourceStorageModeShared];
            require(queue != nil && tensionBuffer != nil && statusBuffer != nil,
                    "Metal queue or borrowed buffer allocation failed");

            numi::matter::Runtime runtime;
            const auto initialized = runtime.initialize(world, {
                .metallib = NUMI_MATTER_METALLIB,
                .environmentCount = 1u,
                .captureEvents = false,
                .captureDiagnostics = true,
                .automaticIdentification = false,
                .adaptiveTransfer = false,
            });
            require(initialized.encoded && runtime.valid(),
                    "runtime initialization: " + initialized.message);
            const auto before = runtime.snapshot();
            require(before.available && before.femNodes.size() == 218077u,
                    "initial snapshot: " + before.message);
            id<MTLCommandBuffer> command = [queue commandBuffer];
            require(command != nil, "no borrowed Metal command buffer");
            numi::matter::EncodeRequest request{};
            request.commandBuffer = (__bridge void*)command;
            request.environmentStatuses = (__bridge void*)statusBuffer;
            request.femActiveTensions = (__bridge void*)tensionBuffer;
            request.femActiveTensionCount =
                static_cast<std::uint32_t>(tensions.size());
            request.controlStep = 0u;
            request.physicsSubstep = 0u;
            request.physicsSubsteps = 1u;
            request.timestepSeconds = runtime.timestepSeconds();
            request.runAdaptiveTransfer = false;
            request.phase = numi::matter::EncodePhase::preDynamics;
            auto encoded = runtime.encode(request);
            require(encoded.encoded, "preDynamics: " + encoded.message);
            request.phase = numi::matter::EncodePhase::postCommit;
            encoded = runtime.encode(request);
            require(encoded.encoded, "postCommit: " + encoded.message);
            [command commit];
            [command waitUntilCompleted];
            require(command.status == MTLCommandBufferStatusCompleted,
                    "native Metal command did not complete: " +
                    std::string(command.error == nil ? "unknown" :
                        [[command.error localizedDescription] UTF8String]));
            const auto after = runtime.snapshot();
            require(after.available && after.femNodes.size() == before.femNodes.size() &&
                    after.statuses.size() == 1u,
                    "completion snapshot: " + after.message);
            const auto& status = after.statuses[0];
            std::ofstream acceptedNodes(argv[3],
                std::ios::binary | std::ios::trunc);
            require(acceptedNodes.good(), "cannot create accepted-state buffer");
            acceptedNodes.write(
                reinterpret_cast<const char*>(after.femNodes.data()),
                static_cast<std::streamsize>(after.femNodes.size() *
                    sizeof(NMFEMNodeStateGPU)));
            require(acceptedNodes.good(), "cannot write accepted-state buffer");
            std::size_t moved = 0u;
            double maximumDisplacement = 0.0;
            for (std::size_t index = 0u; index < after.femNodes.size(); ++index) {
                const auto& a = before.femNodes[index].positionAndMass;
                const auto& b = after.femNodes[index].positionAndMass;
                const double dx = double(b.x) - a.x;
                const double dy = double(b.y) - a.y;
                const double dz = double(b.z) - a.z;
                const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (distance > 0.0) ++moved;
                if (distance > maximumDisplacement) maximumDisplacement = distance;
            }
            const bool accepted = status.code == NM_STATUS_SUCCESS;
            std::printf("{\"device\":\"%s\",\"abi\":%u,"
                        "\"status_code\":%u,\"completed_microsteps\":%u,"
                        "\"fgmres_iterations\":%u,\"failing_index\":%u,"
                        "\"moved_nodes\":%zu,\"maximum_displacement_m\":%.12g,"
                        "\"accepted_native_steps\":%u,"
                        "\"zero_tension_input\":%s,"
                        "\"synthetic_density_kg_m3\":1050,"
                        "\"synthetic_fixed_nodes\":3,"
                        "\"heartbeat_qualified\":false}\n",
                        [[device name] UTF8String], NM_MATTER_ABI_VERSION,
                        status.code, status.completedMicrosteps,
                        status.fgmresIterations, status.failingIndex,
                        moved, maximumDisplacement, accepted ? 1u : 0u,
                        zeroInput ? "true" : "false");
            return accepted ? 0 : 2;
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ventricular source step: %s\n", error.what());
        return 1;
    }
}
