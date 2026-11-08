#include "metalrobo/engine_types.h"
#include "metalrobo/hybrid_renderer_types.h"
#include "metalrobo/visual_platform_types.h"
#include "metalrobo/world_compiler_types.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc != 2) return 2;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"mr_hybrid_reduce_object_tracks"];
        id<MTLComputePipelineState> pipeline = function ?
            [device newComputePipelineStateWithFunction:function error:&error] : nil;
        if (!pipeline) { std::cerr << "missing object-track pipeline\n"; return 3; }
        constexpr unsigned envs = 2, pixels = 4, history = 2, frame = 7, stride = 20, offset = 5;
        auto buffer = [&](NSUInteger bytes) {
            id<MTLBuffer> b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            std::memset(b.contents, 0, bytes);
            return b;
        };
        id<MTLBuffer> b[12];
        b[0] = buffer(envs * pixels * sizeof(float));
        b[1] = buffer(envs * pixels * sizeof(mr_uint4));
        b[2] = buffer(envs * pixels * sizeof(uint32_t));
        b[3] = buffer(sizeof(MRHybridObjectTrackBindingGPU));
        b[4] = buffer(envs * history * 2 * sizeof(mr_float4));
        b[5] = buffer(envs * sizeof(uint32_t));
        b[6] = buffer(envs * history * frame * sizeof(float));
        b[7] = buffer((offset + envs * stride) * sizeof(float));
        b[8] = buffer(envs * sizeof(MRBodyStateGPU));
        b[9] = buffer(envs * sizeof(MRHybridCameraStateGPU));
        b[10] = buffer(envs * sizeof(MRWorldInstanceHeaderGPU));
        b[11] = buffer(envs * sizeof(MRWorldSensorInstanceGPU));
        auto* binding = static_cast<MRHybridObjectTrackBindingGPU*>(b[3].contents);
        binding->identity.x = 42u;
        binding->transform = {1.f, 1.f, 1.f, 0.f};
        auto* roots = static_cast<MRBodyStateGPU*>(b[8].contents);
        auto* cameras = static_cast<MRHybridCameraStateGPU*>(b[9].contents);
        auto* instances = static_cast<MRWorldInstanceHeaderGPU*>(b[10].contents);
        auto* sensors = static_cast<MRWorldSensorInstanceGPU*>(b[11].contents);
        auto* resets = static_cast<uint32_t*>(b[5].contents);
        for (unsigned env = 0; env < envs; ++env) {
            roots[env].orientation.w = 1.f;
            cameras[env].currentOrientation.w = 1.f;
            cameras[env].currentPositionAndValidity.w = 1.f;
            instances[env].ranges.z = env;
            sensors[env].positionAndFocalScale.w = 1.f;
            sensors[env].intrinsics = {1.f, 1.f, 0.f, 0.f};
            resets[env] = 1u;
        }
        MRHybridObjectTrackUniformsGPU u{};
        u.counts = {envs, 1u, pixels, 1u};
        u.layout = {frame, history, 0u, envs};
        u.roots = {1u, 0u, 0u, 0u};
        u.output = {offset, 0u, stride, 0u};
        u.timing = {.02f, 1e5f, 0.f, 0.f};
        auto* depth = static_cast<float*>(b[0].contents);
        auto* ids = static_cast<mr_uint4*>(b[1].contents);
        auto* validity = static_cast<uint32_t*>(b[2].contents);
        auto* observed = static_cast<float*>(b[7].contents);
        auto* actorHistory = static_cast<float*>(b[6].contents);
        unsigned failures = 0;
        const uint32_t good = MR_VISUAL_VALIDITY_FRAME | MR_VISUAL_VALIDITY_DEPTH |
                              MR_VISUAL_VALIDITY_GEOMETRY;
        const uint32_t noDepth = MR_VISUAL_VALIDITY_FRAME | MR_VISUAL_VALIDITY_GEOMETRY;
        const uint32_t noFrame = MR_VISUAL_VALIDITY_DEPTH | MR_VISUAL_VALIDITY_GEOMETRY;
        auto run = [&](const char* name, uint32_t first, uint32_t second, bool visible) {
            std::memset(b[4].contents, 0, b[4].length);
            std::memset(b[6].contents, 0, b[6].length);
            std::fill_n(observed, offset + envs * stride, -5.f);
            for (unsigned env = 0; env < envs; ++env) {
                const auto base = env * pixels;
                for (unsigned p = 0; p < pixels; ++p) {
                    ids[base + p].y = 42u;
                    depth[base + p] = 5.f;
                    validity[base + p] = noDepth;
                }
                depth[base] = .5f;
                validity[base] = first;
                validity[base + 1] = second;
                ids[base + 3].y = 99u;
                validity[base + 3] = good;
            }
            id<MTLCommandQueue> q = [device newCommandQueue];
            id<MTLCommandBuffer> command = [q commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            for (unsigned i = 0; i < 12; ++i) [encoder setBuffer:b[i] offset:0 atIndex:i];
            [encoder setBytes:&u length:sizeof u atIndex:12];
            [encoder dispatchThreadgroups:MTLSizeMake(envs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) { ++failures; return; }
            unsigned bad = 0;
            for (unsigned i = 0; i < offset; ++i) bad += observed[i] != -5.f;
            for (unsigned env = 0; env < envs; ++env) {
                for (unsigned t = 0; t < history; ++t) {
                    const float* o = observed + offset + env * stride + t * frame;
                    const float* h = actorHistory + env * history * frame + t * frame;
                    const float expected[7] = {visible ? 1.f : 0.f, visible ? .25f : 0.f,
                                               visible ? .25f : 0.f, visible ? .5f : 0.f,
                                               0.f, 0.f, 0.f};
                    for (unsigned j = 0; j < 7; ++j) {
                        bad += !std::isfinite(o[j]) || std::abs(o[j] - expected[j]) > 1e-6f;
                        bad += !std::isfinite(h[j]) || std::abs(h[j] - expected[j]) > 1e-6f;
                    }
                }
                for (unsigned j = history * frame; j < stride; ++j)
                    bad += observed[offset + env * stride + j] != -5.f;
            }
            failures += bad;
            std::cout << name << " mismatches=" << bad << " env0_confidence=" << observed[offset]
                      << " env1_confidence=" << observed[offset + stride] << '\n';
        };
        run("valid_mixed", good, noDepth, true);
        run("no_depth", noDepth, noDepth, false);
        run("no_frame", noFrame, noFrame, false);
        run("valid_plus_missing_frame", good, noFrame, true);

        // A missing measurement invalidates the newest record; reacquisition
        // must not derive velocity across the missing interval.
        std::memset(b[4].contents, 0, b[4].length);
        std::memset(b[6].contents, 0, b[6].length);
        for (unsigned step = 0; step < 3; ++step) {
            std::fill_n(observed, offset + envs * stride, -5.f);
            for (unsigned env = 0; env < envs; ++env) {
                resets[env] = step == 0u ? 1u : 0u;
                const unsigned base = env * pixels;
                for (unsigned p = 0; p < pixels; ++p) {
                    ids[base + p].y = 42u;
                    depth[base + p] = 5.f;
                    validity[base + p] = noDepth;
                }
                depth[base] = .5f;
                validity[base] = step == 1u ? noDepth : good;
            }
            id<MTLCommandQueue> q = [device newCommandQueue];
            id<MTLCommandBuffer> command = [q commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            for (unsigned i = 0; i < 12; ++i) [encoder setBuffer:b[i] offset:0 atIndex:i];
            [encoder setBytes:&u length:sizeof u atIndex:12];
            [encoder dispatchThreadgroups:MTLSizeMake(envs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) { ++failures; continue; }
            const float expectedOld = step == 2u ? 0.f : 1.f;
            const float expectedNew = step == 1u ? 0.f : 1.f;
            unsigned bad = 0;
            for (unsigned env = 0; env < envs; ++env) {
                const float* o = observed + offset + env * stride;
                const float* h = actorHistory + env * history * frame;
                bad += o[0] != expectedOld || h[0] != expectedOld;
                bad += o[frame] != expectedNew || h[frame] != expectedNew;
                if (step == 2u) {
                    for (unsigned j = 4; j < 7; ++j)
                        bad += o[frame + j] != 0.f || h[frame + j] != 0.f;
                }
            }
            failures += bad;
            std::cout << "temporal_step=" << step << " mismatches=" << bad
                      << " old=" << observed[offset]
                      << " newest=" << observed[offset + frame] << '\n';
        }
        return failures ? 1 : 0;
    }
}
