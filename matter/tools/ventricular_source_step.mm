#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/matter.hpp"
#include "metalrobo/engine_types.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
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
        require(argc >= 4, "usage: ventricular-source-step cooked.nmpkg "
                "cooked-tension.f32le accepted-nodes.bin [--zero] "
                "[--sequence-count N] [--capture-elements initial-elements.bin]");
        bool zeroInput = false;
        bool capture = false;
        bool sequenceCountSpecified = false;
        std::size_t sequenceCount = 1u;
        std::filesystem::path capturePath;
        for (int index = 4; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--zero") {
                require(!zeroInput, "duplicate --zero flag");
                zeroInput = true;
            } else if (option == "--sequence-count") {
                require(!sequenceCountSpecified && index + 1 < argc,
                        "--sequence-count requires one value and may appear once");
                const std::string value = argv[++index];
                std::size_t consumed = 0u;
                const auto parsed = std::stoul(value, &consumed, 10);
                require(consumed == value.size() && parsed >= 1u && parsed <= 64u,
                        "--sequence-count must be between 1 and 64");
                sequenceCount = static_cast<std::size_t>(parsed);
                sequenceCountSpecified = true;
            } else if (option == "--capture-elements") {
                require(!capture && index + 1 < argc,
                        "--capture-elements requires one path and may appear once");
                capturePath = argv[++index];
                capture = true;
            } else {
                require(false, "unknown option " + option);
            }
        }
        require(!capture || sequenceCount == 1u,
                "initial element-force capture is defined for one step");
        numi::matter::CompiledWorld world;
        std::string packageError;
        require(numi::matter::readPackage(argv[1], world, nullptr, &packageError),
                "cannot load full ventricular package: " + packageError);
        require(world.dispatch.environmentCount == 1u &&
                world.fem.tetrahedra.size() == 1097534u &&
                world.fem.nodes.size() == 218080u,
                "unexpected full source ventricular cooked dimensions");
        const std::filesystem::path tensionPath = argv[2];
        const std::size_t frameBytes = world.fem.tetrahedra.size() * sizeof(float);
        require(sequenceCount <= std::numeric_limits<std::size_t>::max() / frameBytes,
                "tension sequence byte count overflow");
        const std::size_t inputBytes = frameBytes * sequenceCount;
        require(std::filesystem::is_regular_file(tensionPath) &&
                std::filesystem::file_size(tensionPath) == inputBytes,
                "cooked tension file does not cover every requested frame");
        std::vector<float> tensions(inputBytes / sizeof(float));
        std::ifstream stream(tensionPath, std::ios::binary);
        stream.read(reinterpret_cast<char*>(tensions.data()),
                    static_cast<std::streamsize>(inputBytes));
        require(stream.good(), "cannot read cooked tension field");
        if (zeroInput) std::fill(tensions.begin(), tensions.end(), 0.0f);

        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "no Metal device");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            const std::size_t elementBytes = world.fem.tetrahedra.size() *
                sizeof(NMFEMElementVectorGPU);
            id<MTLBuffer> initialElements = capture ? [device
                newBufferWithLength:elementBytes
                options:MTLResourceStorageModeShared] : nil;
            MRMetalWorldStatusGPU worldStatus{};
            id<MTLBuffer> statusBuffer = [device
                newBufferWithBytes:&worldStatus length:sizeof(worldStatus)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> tensionBuffer = [device
                newBufferWithLength:frameBytes
                options:MTLResourceStorageModeShared];
            require(queue != nil && statusBuffer != nil,
                    "Metal queue or borrowed buffer allocation failed");
            require(tensionBuffer != nil,
                    "active-tension Metal buffer allocation failed");
            require(!capture || initialElements != nil,
                    "initial element-force buffer allocation failed");

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
            require(before.available && before.femNodes.size() == 218080u,
                    "initial snapshot: " + before.message);
            std::vector<std::uint32_t> statusCodes;
            std::vector<std::uint32_t> completedMicrosteps;
            std::vector<std::uint32_t> fgmresIterations;
            std::vector<std::uint32_t> failingIndices;
            statusCodes.reserve(sequenceCount);
            completedMicrosteps.reserve(sequenceCount);
            fgmresIterations.reserve(sequenceCount);
            failingIndices.reserve(sequenceCount);
            numi::matter::RuntimeStateSnapshot after;
            id<MTLBuffer> matterStatusBuffer =
                (__bridge id<MTLBuffer>)runtime.statusBuffer();
            const auto* matterStatuses = static_cast<const NMMatterStatusGPU*>(
                matterStatusBuffer.contents);
            require(matterStatusBuffer != nil && matterStatuses != nullptr,
                    "Matter status buffer is unavailable");
            const auto runStarted = std::chrono::steady_clock::now();
            for (std::size_t step = 0u; step < sequenceCount; ++step) {
                std::memcpy(tensionBuffer.contents,
                    tensions.data() + step * world.fem.tetrahedra.size(),
                    frameBytes);
                auto* stepStatus = static_cast<MRMetalWorldStatusGPU*>(
                    statusBuffer.contents);
                *stepStatus = {};
                stepStatus->code = MR_STEP_SUCCESS;
                id<MTLCommandBuffer> command = [queue commandBuffer];
                require(command != nil, "no borrowed Metal command buffer");
                numi::matter::EncodeRequest request{};
                request.commandBuffer = (__bridge void*)command;
                request.environmentStatuses = (__bridge void*)statusBuffer;
                request.femActiveTensions = (__bridge void*)tensionBuffer;
                request.femActiveTensionCount =
                    static_cast<std::uint32_t>(world.fem.tetrahedra.size());
                if (capture && step == 0u) {
                    request.femInitialElementForces = (__bridge void*)initialElements;
                    request.femInitialElementForceCount =
                        static_cast<std::uint32_t>(world.fem.tetrahedra.size());
                }
                request.controlStep = static_cast<std::uint32_t>(step);
                request.physicsSubstep = 0u;
                request.physicsSubsteps = 1u;
                request.timestepSeconds = runtime.timestepSeconds();
                request.runAdaptiveTransfer = false;
                request.phase = numi::matter::EncodePhase::preDynamics;
                auto encoded = runtime.encode(request);
                require(encoded.encoded, "preDynamics step " +
                        std::to_string(step) + ": " + encoded.message);
                request.femInitialElementForces = nullptr;
                request.femInitialElementForceCount = 0u;
                request.phase = numi::matter::EncodePhase::postCommit;
                encoded = runtime.encode(request);
                require(encoded.encoded, "postCommit step " +
                        std::to_string(step) + ": " + encoded.message);
                [command commit];
                [command waitUntilCompleted];
                require(command.status == MTLCommandBufferStatusCompleted,
                        "native Metal command step " + std::to_string(step) +
                        " did not complete: " +
                        std::string(command.error == nil ? "unknown" :
                            [[command.error localizedDescription] UTF8String]));
                if (capture && step == 0u) {
                    std::ofstream captured(capturePath,
                        std::ios::binary | std::ios::trunc);
                    require(captured.good(),
                            "cannot create initial element-force output");
                    captured.write(static_cast<const char*>(initialElements.contents),
                        static_cast<std::streamsize>(elementBytes));
                    require(captured.good(),
                            "cannot write initial element-force output");
                }
                const auto& stepResult = matterStatuses[0];
                statusCodes.push_back(stepResult.code);
                completedMicrosteps.push_back(stepResult.completedMicrosteps);
                fgmresIterations.push_back(stepResult.fgmresIterations);
                failingIndices.push_back(stepResult.failingIndex);
                require(stepResult.code == NM_STATUS_SUCCESS,
                        "native step " + std::to_string(step) +
                        " returned status " + std::to_string(stepResult.code));
            }
            const double wallSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - runStarted).count();
            after = runtime.snapshot();
            require(after.available && after.femNodes.size() == before.femNodes.size(),
                    "final snapshot: " + after.message);
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
            if (sequenceCount == 1u) {
                const auto& status = after.statuses[0];
                std::printf("{\"device\":\"%s\",\"abi\":%u,"
                            "\"status_code\":%u,\"completed_microsteps\":%u,"
                            "\"fgmres_iterations\":%u,\"failing_index\":%u,"
                            "\"moved_nodes\":%zu,\"maximum_displacement_m\":%.12g,"
                            "\"accepted_native_steps\":%u,\"wall_seconds\":%.9g,"
                            "\"zero_tension_input\":%s,"
                            "\"synthetic_density_kg_m3\":1050,"
                            "\"synthetic_fixed_nodes\":3,"
                            "\"point_only_lv_rv_nodes_split\":3,"
                            "\"initial_element_force_capture\":%s,"
                            "\"heartbeat_qualified\":false}\n",
                            [[device name] UTF8String], NM_MATTER_ABI_VERSION,
                            status.code, status.completedMicrosteps,
                            status.fgmresIterations, status.failingIndex,
                            moved, maximumDisplacement, 1u, wallSeconds,
                            zeroInput ? "true" : "false",
                            capture ? "true" : "false");
            } else {
                std::printf("{\"device\":\"%s\",\"abi\":%u,"
                            "\"status_code\":%u,\"sequence_count\":%zu,"
                            "\"accepted_native_steps\":%zu,"
                            "\"step_status_codes\":[",
                            [[device name] UTF8String], NM_MATTER_ABI_VERSION,
                            statusCodes.back(), sequenceCount, sequenceCount);
                for (std::size_t step = 0u; step < sequenceCount; ++step) {
                    std::printf("%s%u", step == 0u ? "" : ",", statusCodes[step]);
                }
                std::printf("],\"step_completed_microsteps\":[");
                for (std::size_t step = 0u; step < sequenceCount; ++step) {
                    std::printf("%s%u", step == 0u ? "" : ",",
                                completedMicrosteps[step]);
                }
                std::printf("],\"step_fgmres_iterations\":[");
                for (std::size_t step = 0u; step < sequenceCount; ++step) {
                    std::printf("%s%u", step == 0u ? "" : ",",
                                fgmresIterations[step]);
                }
                std::printf("],\"step_failing_indices\":[");
                for (std::size_t step = 0u; step < sequenceCount; ++step) {
                    std::printf("%s%u", step == 0u ? "" : ",", failingIndices[step]);
                }
                std::printf("],\"moved_nodes\":%zu,"
                            "\"maximum_displacement_m\":%.12g,"
                            "\"accepted_duration_s\":%.12g,"
                            "\"wall_seconds\":%.9g,"
                            "\"zero_tension_input\":%s,"
                            "\"synthetic_density_kg_m3\":1050,"
                            "\"synthetic_fixed_nodes\":3,"
                            "\"point_only_lv_rv_nodes_split\":3,"
                            "\"heartbeat_qualified\":false}\n",
                            moved, maximumDisplacement,
                            runtime.timestepSeconds() * sequenceCount,
                            wallSeconds,
                            zeroInput ? "true" : "false");
            }
            return 0;
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ventricular source step: %s\n", error.what());
        return 1;
    }
}
