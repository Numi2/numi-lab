#include "metalrobo/hybrid_renderer_types.h"
#include "metalrobo/task_program_types.h"
#include "metalrobo/visual_platform_types.h"
#include "metalrobo/world_compiler_types.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 2) {
            std::cerr << "usage: probe shader.metallib\n";
            return 2;
        }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        id<MTLLibrary> lib = [dev
            newLibraryWithURL:[NSURL
                                  fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                        error:&err];
        id<MTLFunction> fn = [lib newFunctionWithName:@"mr_hybrid_masked_depth_history"];
        id<MTLComputePipelineState> ps =
            fn ? [dev newComputePipelineStateWithFunction:fn error:&err] : nil;
        if (!ps) {
            std::cerr << "pipeline: "
                      << (err ? [[err description] UTF8String] : "missing") << '\n';
            return 3;
        }
        unsigned failures = 0;
        auto test = [&](unsigned w, unsigned h, bool direct) {
            const unsigned n = w * h, envs = 2, total = n * envs;
            id<MTLBuffer> b[16]{};
            auto buf = [&](NSUInteger len) {
                id<MTLBuffer> q = [dev newBufferWithLength:len
                                                   options:MTLResourceStorageModeShared];
                std::memset(q.contents, 0, len);
                return q;
            };
            b[0] = buf(total * sizeof(float));
            b[1] = buf(total * sizeof(mr_uint4));
            b[2] = buf(total * sizeof(uint32_t));
            b[3] = buf(sizeof(uint32_t));
            b[4] = buf(total * 3 * sizeof(float));
            b[5] = buf(envs * sizeof(uint32_t));
            b[6] = buf(16 * sizeof(float));
            b[7] = buf(total * sizeof(float));
            b[8] = buf(envs * sizeof(MRTaskStateGPU));
            b[10] = buf(total * sizeof(uint64_t));
            for (int i = 11; i <= 15; ++i)
                b[i] = buf(4096);
            auto depths = (float *)b[0].contents;
            auto ids = (mr_uint4 *)b[1].contents;
            auto val = (uint32_t *)b[2].contents;
            auto masks = (uint32_t *)b[5].contents;
            auto winners = (uint64_t *)b[10].contents;
            *(uint32_t *)b[3].contents = 42;
            masks[0] = masks[1] = 1;
            ((MRVisualPrimitiveGPUV2 *)b[12].contents)->identity.y = 42;
            auto sensor = (MRWorldSensorInstanceGPU *)b[15].contents;
            MRHybridMaskedDepthUniformsGPU u{};
            u.counts = {envs, h, w, 1};
            u.layout = {n, 1, 0, envs};
            u.history = {3, 1, 0, 1};
            u.range = {.1f, 5.f, 1.f / 4.9f, 0.f};
            u.curriculum.y = direct ? 1.f : 0.f;
            MRHybridRenderUniformsGPU ru{};
            ru.live.w = 1;
            ru.clearColorAndDepth.w = 5.f;
            ru.sensorRangeAndResponse = {.1f, 5.f, 0.f, 0.f};
            std::uint32_t fresh[envs] = {1u, 1u};
            u.curriculum.w = 1.f; // This invocation contains a newly sampled frame.
            auto run = [&](const char *phase) {
                id<MTLCommandQueue> q = [dev newCommandQueue];
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:ps];
                for (unsigned i = 0; i < 9; ++i)
                    [e setBuffer:b[i] offset:0 atIndex:i];
                [e setBytes:&u length:sizeof u atIndex:9];
                for (unsigned i = 10; i <= 15; ++i)
                    [e setBuffer:b[i] offset:0 atIndex:i];
                [e setBytes:&ru length:sizeof ru atIndex:16];
                [e setBytes:fresh length:sizeof fresh atIndex:17];
                [e dispatchThreadgroups:MTLSizeMake(envs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [e endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted) {
                    std::cerr << "command failed " << phase << '\n';
                    ++failures;
                }
                std::vector<float> out(total);
                std::memcpy(out.data(), b[7].contents, total * sizeof(float));
                return out;
            };
            for (unsigned i = 0; i < total; ++i) {
                depths[i] = .5f;
                ids[i].y = i < n ? 42u : 99u;
                val[i] = MR_VISUAL_VALIDITY_FRAME | MR_VISUAL_VALIDITY_GEOMETRY |
                         MR_VISUAL_VALIDITY_DEPTH;
                uint32_t bits;
                float d = .5f;
                std::memcpy(&bits, &d, sizeof bits);
                winners[i] = (uint64_t(bits) << 32);
            }
            if (direct) {
                for (unsigned i = n; i < total; ++i)
                    winners[i] = UINT64_MAX;
                ru.shadowBatch.w = 1;
            }
            auto valid = run("selected");
            unsigned badValid = 0, badUnselected = 0;
            for (unsigned i = 0; i < n; ++i)
                badValid += std::abs(valid[i] - (.4f / 4.9f)) > 1e-6f;
            for (unsigned i = n; i < total; ++i)
                badUnselected += std::abs(valid[i] - 1.f) > 1e-6f;
            failures += badValid + badUnselected;
            std::cout << "valid " << w << 'x' << h << " direct=" << direct
                      << " selected_bad=" << badValid
                      << " unselected_bad=" << badUnselected << '\n';
            if (!direct) {
                for (unsigned i = 0; i < total; ++i) {
                    depths[i] = .5f;
                    ids[i].y = 42;
                    val[i] = MR_VISUAL_VALIDITY_GEOMETRY | MR_VISUAL_VALIDITY_DEPTH;
                }
                auto missingFrame = run("missing-frame");
                unsigned falseNear = 0;
                for (float x : missingFrame)
                    falseNear += !std::isfinite(x) || x < 1.f - 1e-6f;
                failures += falseNear;
                std::cout << "missing_frame " << w << 'x' << h
                          << " false_near=" << falseNear << " total=" << total << '\n';
                for (unsigned i = 0; i < total; ++i) {
                    depths[i] = 5.f;
                    ids[i].y = 42;
                    val[i] = i < n ? 0u : MR_VISUAL_VALIDITY_FRAME;
                }
                depths[0] = depths[n] = .5f;
                val[0] = val[n] = MR_VISUAL_VALIDITY_FRAME | MR_VISUAL_VALIDITY_GEOMETRY |
                                  MR_VISUAL_VALIDITY_DEPTH;
                u.range.w = 1.f;
                auto mixed = run("mixed-frame-edge");
                unsigned badMissing = 0, badPresent = 0;
                float near = .4f / 4.9f;
                for (unsigned i = 0; i < n; ++i) {
                    const float expected = i == 0 ? near : 1.f;
                    badMissing +=
                        !std::isfinite(mixed[i]) || std::abs(mixed[i] - expected) > 1e-6f;
                }
                for (unsigned i = 0; i < n; ++i) {
                    const unsigned x = i % w, y = i / w;
                    const float expected = (x <= 1 && y <= 1) ? near : 1.f;
                    badPresent += !std::isfinite(mixed[n + i]) ||
                                  std::abs(mixed[n + i] - expected) > 1e-6f;
                }
                failures += badMissing + badPresent;
                std::cout << "mixed_frame " << w << 'x' << h
                          << " missing_bad=" << badMissing
                          << " present_bad=" << badPresent << '\n';
                u.range.w = 0.f;
            }
            for (unsigned i = 0; i < total; ++i) {
                depths[i] = 5.f;
                ids[i].y = 42;
                val[i] = MR_VISUAL_VALIDITY_FRAME | MR_VISUAL_VALIDITY_GEOMETRY;
                float d = .5f;
                uint32_t bits;
                std::memcpy(&bits, &d, sizeof bits);
                winners[i] = uint64_t(bits) << 32;
            }
            if (direct)
                sensor->noiseAndLatency.z = 1.f;
            u.corruption.w = 1.f;
            auto invalid = run("sensor-depth-invalid");
            unsigned falseNear = 0, nonfinite = 0;
            for (float x : invalid) {
                nonfinite += !std::isfinite(x);
                falseNear += x < 1.f - 1e-6f;
            }
            failures += falseNear + nonfinite;
            std::cout << "invalid " << w << 'x' << h << " direct=" << direct
                      << " false_near=" << falseNear << " nonfinite=" << nonfinite
                      << " total=" << total << '\n';
            // A sampled corrupted image is a measurement, not a new random
            // draw at each control tick. Exercise both a wrapped ring and a
            // one-slot ring on each materialized/direct path.
            sensor->noiseAndLatency.z = 0.f;
            u.corruption = {0.f, .25f, .12f, .04f};
            u.range.w = 0.f;
            auto *tasks = (MRTaskStateGPU *)b[8].contents;
            for (unsigned i=0; i<total; ++i) {
                depths[i]=.5f; ids[i].y=42;
                val[i]=MR_VISUAL_VALIDITY_FRAME | MR_VISUAL_VALIDITY_GEOMETRY | MR_VISUAL_VALIDITY_DEPTH;
                float d=.5f; uint32_t bits; std::memcpy(&bits,&d,sizeof bits);
                winners[i]=uint64_t(bits)<<32;
            }
            for (unsigned capacity : {3u, 1u}) {
                u.history.x=capacity;u.curriculum.w=1.f;
                masks[0]=masks[1]=1u; tasks[0].episode.x=tasks[1].episode.x=0u;
                auto first=run("hold-first");
                masks[0]=masks[1]=0u; tasks[0].episode.x=tasks[1].episode.x=1u;
                u.curriculum.w=0.f; auto held1=run("hold-one");
                tasks[0].episode.x=tasks[1].episode.x=2u;u.curriculum.w=1.f;
                auto sampled=run("hold-sample");
                tasks[0].episode.x=tasks[1].episode.x=3u;u.curriculum.w=0.f;
                auto held2=run("hold-two");
                const bool holdOK=first==held1 && sampled==held2;
                const bool changed=first!=sampled;
                failures += !holdOK || !changed;
                std::cout << "hold " << w << 'x' << h << " direct=" << direct
                          << " capacity=" << capacity << " held=" << holdOK
                          << " fresh_changed=" << changed << '\n';
                // Invalidated reset must never reuse the old episode's image;
                // the other environment continues to hold its last sample.
                masks[0]=1u; fresh[0]=0u;
                tasks[0].episode.x=0u;tasks[1].episode.x=4u;
                auto invalidReset=run("hold-reset");
                bool invalidOK=std::all_of(invalidReset.begin(),invalidReset.begin()+n,[](float x){return x==1.f;});
                bool neighborOK=std::equal(invalidReset.begin()+n,invalidReset.end(),held2.begin()+n);
                failures += !invalidOK || !neighborOK;
                std::cout << "hold_reset " << w << 'x' << h << " direct=" << direct
                          << " capacity=" << capacity << " invalid=" << invalidOK
                          << " neighbor_held=" << neighborOK << '\n';
                fresh[0]=1u;masks[0]=0u;u.curriculum.w=1.f;
                tasks[0].episode.x=1u;tasks[1].episode.x=5u;
                auto recovered=run("hold-recovered");
                bool recovery=std::any_of(recovered.begin(),recovered.begin()+n,[](float x){return x<1.f;});
                failures += !recovery;
                std::cout << "hold_recovery " << w << 'x' << h << " direct=" << direct
                          << " capacity=" << capacity << " recovered=" << recovery << '\n';
            }
        };
        test(4, 2, false);
        test(20, 20, false);
        test(4, 2, true);
        std::cout << "failures=" << failures << '\n';
        return failures ? 1 : 0;
    }
}
