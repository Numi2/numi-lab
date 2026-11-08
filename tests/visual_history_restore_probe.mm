#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/engine_types.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc != 2) return 2;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError* error = nil;
        NSString* path = [NSString stringWithUTF8String:argv[1]];
        id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"mr_hybrid_restore_failed_observation_history"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!device || !library || !function || !pipeline || !queue) {
            std::cerr << "Metal setup failed\n";
            return 3;
        }
        constexpr std::uint32_t envs = 4;
        for (const std::uint32_t trackCount : {0u, 2u, 18u}) {
            for (const std::uint32_t depthCount : {0u, 7u, 411u}) {
                for (const bool contactMode : {false, true}) {
                  for (const std::uint32_t sampledFrame : {0u, 1u}) {
                   for (const std::uint32_t oldValue : {0u, 1u}) {
                    const std::size_t trackWords = envs * trackCount * 4u + 16u;
                    const std::size_t depthWords = envs * depthCount + 16u;
                    const std::uint32_t actorCount = depthCount == 0u ? 0u : (trackCount == 0u ? 35u : 102u);
                    const std::size_t actorWords = envs * actorCount + 16u;
                    std::vector<std::uint32_t> actors(actorWords), oldActors(actorWords);
                    for (std::size_t i=0; i<actorWords; ++i) { actors[i]=0x55000000u+static_cast<std::uint32_t>(i); oldActors[i]=0x66000000u+static_cast<std::uint32_t>(i); }
                    std::vector<std::uint32_t> tracks(trackWords), oldTracks(trackWords);
                    std::vector<std::uint32_t> depths(depthWords), oldDepths(depthWords);
                    for (std::size_t i=0; i<trackWords; ++i) { tracks[i]=0x11000000u+static_cast<std::uint32_t>(i); oldTracks[i]=0x22000000u+static_cast<std::uint32_t>(i); }
                    for (std::size_t i=0; i<depthWords; ++i) { depths[i]=0x33000000u+static_cast<std::uint32_t>(i); oldDepths[i]=0x44000000u+static_cast<std::uint32_t>(i); }
                    std::array<MRMetalWorldStatusGPU,envs> worlds{};
                    std::array<MRMetalWorldContactStatusGPU,envs> contacts{};
                    worlds[1].code = MR_STEP_FACTORIZATION_FAILED;
                    contacts[2].code = MR_STEP_CONTACT_CAPACITY_OVERFLOW;
                    worlds[3].code = MR_STEP_FACTORIZATION_FAILED;
                    contacts[3].code = MR_STEP_CONTACT_CAPACITY_OVERFLOW;
                    auto buffer = [&](const void* bytes, std::size_t size) -> id<MTLBuffer> {
                        return [device newBufferWithBytes:bytes length:size options:MTLResourceStorageModeShared];
                    };
                    id<MTLBuffer> t=buffer(tracks.data(),trackWords*sizeof(std::uint32_t));
                    id<MTLBuffer> ot=buffer(oldTracks.data(),trackWords*sizeof(std::uint32_t));
                    id<MTLBuffer> d=buffer(depths.data(),depthWords*sizeof(std::uint32_t));
                    id<MTLBuffer> od=buffer(oldDepths.data(),depthWords*sizeof(std::uint32_t));
                    id<MTLBuffer> w=buffer(worlds.data(),sizeof(worlds));
                    id<MTLBuffer> c=buffer(contacts.data(),sizeof(contacts));
                    std::array<std::uint32_t, envs> fresh{{1u,1u,1u,1u}}, oldFresh{{oldValue,oldValue,oldValue,oldValue}};
                    id<MTLBuffer> frameFresh=buffer(fresh.data(),sizeof(fresh));
                    id<MTLBuffer> oldFrameFresh=buffer(oldFresh.data(),sizeof(oldFresh));
                    id<MTLBuffer> actor=buffer(actors.data(),actorWords*sizeof(std::uint32_t));
                    id<MTLBuffer> oldActor=buffer(oldActors.data(),actorWords*sizeof(std::uint32_t));
                    id<MTLCommandBuffer> command=[queue commandBuffer];
                    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
                    if (!t || !ot || !d || !od || !w || !c || !actor || !oldActor || !frameFresh || !oldFrameFresh || !command || !encoder) return 4;
                    [encoder setComputePipelineState:pipeline];
                    const std::array<id<MTLBuffer>,6> buffers{{t,ot,d,od,w,c}};
                    for (NSUInteger i=0;i<buffers.size();++i) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
                    const mr_uint4 counts{envs,trackCount,depthCount,contactMode ? 1u : 0u};
                    [encoder setBytes:&counts length:sizeof(counts) atIndex:6];
                    [encoder setBuffer:actor offset:0 atIndex:7];
                    [encoder setBuffer:oldActor offset:0 atIndex:8];
                    [encoder setBytes:&actorCount length:sizeof(actorCount) atIndex:9];
                    [encoder setBuffer:frameFresh offset:0 atIndex:10];
                    [encoder setBuffer:oldFrameFresh offset:0 atIndex:11];
                    [encoder setBytes:&sampledFrame length:sizeof(sampledFrame) atIndex:12];
                    [encoder dispatchThreadgroups:MTLSizeMake(envs,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
                    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
                    if (command.status != MTLCommandBufferStatusCompleted) return 5;
                    for (std::uint32_t env=0; env<envs; ++env) {
                        const bool failed = worlds[env].code != MR_STEP_SUCCESS ||
                            (contactMode && contacts[env].code != MR_STEP_SUCCESS);
                        for (std::uint32_t i=0; i<trackCount*4u; ++i) {
                            const std::size_t j=env*trackCount*4u+i;
                            if (static_cast<const std::uint32_t*>(t.contents)[j] != (failed ? oldTracks[j] : tracks[j])) return 6;
                        }
                        if (static_cast<const std::uint32_t*>(frameFresh.contents)[env] != (failed ? (sampledFrame ? 0u : oldValue) : 1u)) return 10;
                        for (std::uint32_t i=0; i<actorCount; ++i) {
                            const std::size_t j=env*actorCount+i;
                            if (static_cast<const std::uint32_t*>(actor.contents)[j] != (failed ? oldActors[j] : actors[j])) return 9;
                        }
                        for (std::uint32_t i=0; i<depthCount; ++i) {
                            const std::size_t j=env*depthCount+i;
                            if (static_cast<const std::uint32_t*>(d.contents)[j] != (failed ? oldDepths[j] : depths[j])) return 7;
                        }
                    }
                    if (std::memcmp(static_cast<const std::uint32_t*>(t.contents)+envs*trackCount*4u,
                            tracks.data()+envs*trackCount*4u,16u*sizeof(std::uint32_t)) != 0 ||
                        std::memcmp(static_cast<const std::uint32_t*>(d.contents)+envs*depthCount,
                            depths.data()+envs*depthCount,16u*sizeof(std::uint32_t)) != 0 ||
                        std::memcmp(static_cast<const std::uint32_t*>(actor.contents)+envs*actorCount,
                            actors.data()+envs*actorCount,16u*sizeof(std::uint32_t)) != 0) return 8;
                   }
                  }
                }
            }
        }
        std::cout << "72 mixed-status history and freshness layouts passed\n";
    }
    return 0;
}
