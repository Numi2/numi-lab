#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class T> bool equal(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}
}

// Invoke the production initializer twice, including all typed failures and
// an out-of-range lane. No contact or integration implementation is replaced.
int main(int argc, char** argv) { @autoreleasepool { try {
    require(argc == 4, "Usage: event_status_probe METALLIB NEW_OUTPUT baseline|preserve");
    const bool preserve = std::string(argv[3]) == "preserve";
    require(preserve || std::string(argv[3]) == "baseline", "Invalid expectation");
    require(!std::filesystem::exists(argv[2]), "Output must be new");
    auto device = MTLCreateSystemDefaultDevice();
    require(device != nil, "Metal unavailable");
    NSError* error = nil;
    auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:
        [NSString stringWithUTF8String:argv[1]]] error:&error];
    require(library != nil, "Missing shader library");
    auto function = [library newFunctionWithName:@"mr_world_initialize_ccd_event_state"];
    require(function != nil, "Missing production initializer");
    auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline != nil, "Pipeline creation failed");
    auto queue = [device newCommandQueue];
    constexpr unsigned count = MR_STEP_STATUS_COUNT, allocated = count + 1;
    std::ofstream output(argv[2], std::ios::binary);
    unsigned preserved = 0, healthy = 0;
    for (const unsigned mode : {unsigned(MR_WORLD_CCD_HYBRID), unsigned(MR_WORLD_CCD_DISABLED)}) {
        MRMetalWorldContactDispatchGPU dispatch{};
        dispatch.environmentCount = count;
        dispatch.ccdMode = mode;
        dispatch.maxCCDAdvanceSolvePasses = 8;
        dispatch.timestepAndBias.x = .000625f;
        dispatch.ccdEventParameters.x = 1e-6f;
        std::array<MRCCDEventStateGPU, allocated> a, b;
        std::array<MRCCDImpactClusterGPU, allocated> cluster;
        std::array<MRMetalWorldContactStatusGPU, allocated> status;
        std::memset(a.data(), 0x33, sizeof(a));
        std::memset(b.data(), 0x44, sizeof(b));
        std::memset(cluster.data(), 0x55, sizeof(cluster));
        std::memset(status.data(), 0x22, sizeof(status));
        for (unsigned lane = 0; lane < allocated; ++lane) {
            status[lane].code = lane;
            status[lane].environment = lane;
        }
        auto buffer = [&](const void* data, std::size_t bytes) {
            auto result = [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
            require(result != nil, "Allocation failed");
            return result;
        };
        std::array<id<MTLBuffer>,5> buffers{
            buffer(&dispatch,sizeof(dispatch)), buffer(a.data(),sizeof(a)),
            buffer(b.data(),sizeof(b)), buffer(cluster.data(),sizeof(cluster)),
            buffer(status.data(),sizeof(status))};
        for (const unsigned substep : {7u, 99u}) {
            MRMetalWorldPassGPU pass{}; pass.physicsSubstep = substep;
            auto command = [queue commandBuffer];
            auto encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            for (unsigned i = 0; i < buffers.size(); ++i)
                [encoder setBuffer:buffers[i] offset:0 atIndex:i];
            [encoder setBytes:&pass length:sizeof(pass) atIndex:5];
            [encoder dispatchThreads:MTLSizeMake(allocated,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            require(command.status == MTLCommandBufferStatusCompleted, "GPU execution failed");
            const auto* actualA = static_cast<const MRCCDEventStateGPU*>(buffers[1].contents);
            const auto* actualB = static_cast<const MRCCDEventStateGPU*>(buffers[2].contents);
            const auto* actualC = static_cast<const MRCCDImpactClusterGPU*>(buffers[3].contents);
            const auto* actualS = static_cast<const MRMetalWorldContactStatusGPU*>(buffers[4].contents);
            for (unsigned lane = 0; lane < allocated; ++lane) {
                const bool unchanged = equal(actualA[lane],a[lane]) && equal(actualB[lane],b[lane]) &&
                    equal(actualC[lane],cluster[lane]) && equal(actualS[lane],status[lane]);
                if (lane == count || (preserve && lane != MR_STEP_SUCCESS)) {
                    require(unchanged, "Failed or out-of-range lane was modified"); ++preserved;
                } else {
                    require(!unchanged, "Initializer did not execute");
                    require(actualS[lane].code == lane, "Initializer changed status code");
                    require(actualA[lane].generation == substep * 8 && equal(actualA[lane],actualB[lane]), "Invalid event generation");
                    require(actualS[lane].zeroTimeCCDReplays == 0 && actualS[lane].ccdAdvanceCount == 0 &&
                        actualS[lane].firstFailingEventKeyLow == MR_INVALID_INDEX, "Invalid counter reset");
                    if (lane == 0) ++healthy;
                }
                output.write(reinterpret_cast<const char*>(&actualA[lane]),sizeof(actualA[lane]));
                output.write(reinterpret_cast<const char*>(&actualB[lane]),sizeof(actualB[lane]));
                output.write(reinterpret_cast<const char*>(&actualC[lane]),sizeof(actualC[lane]));
                output.write(reinterpret_cast<const char*>(&actualS[lane]),sizeof(actualS[lane]));
            }
        }
    }
    require(bool(output), "Output failed");
    std::cout << "{\"status\":\"passed\",\"preserved_checks\":" << preserved
        << ",\"healthy_checks\":" << healthy << "}\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } } }
