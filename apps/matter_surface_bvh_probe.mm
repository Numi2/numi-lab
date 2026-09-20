#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/shared.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
using Pair = std::array<std::uint32_t, 4>;
struct Fixture {
    const char* name;
    std::uint32_t count, capacity;
    unsigned mode; // 0 mixed random, 1 dense, 2 separated, 3 heterogeneous environments
};
float component(const nm_float4& v, unsigned axis) {
    return axis == 0 ? v.x : axis == 1 ? v.y : v.z;
}
std::uint32_t node(const NMContinuumSurfacePrimitiveGPU& p, unsigned axis) {
    return axis == 0 ? p.nodesAndObject.x : axis == 1 ? p.nodesAndObject.y : p.nodesAndObject.z;
}
// Independent exhaustive source-pair oracle. It deliberately does not traverse
// the hierarchy, use its bounds, or reuse its rejection helpers.
bool eligible(const NMMatterDispatchGPU& d,
              const NMContinuumSurfacePrimitiveGPU& a,
              const NMContinuumSurfacePrimitiveGPU& b,
              const std::vector<NMContinuumObjectGPU>& objects,
              const std::vector<NMFEMTopologyNodeGPU>& topology,
              unsigned environment) {
    for (unsigned axis=0; axis<3; ++axis)
        if (double(component(a.boundsMinimum,axis)) > double(component(b.boundsMaximum,axis)) ||
            double(component(b.boundsMinimum,axis)) > double(component(a.boundsMaximum,axis))) return false;
    const auto ao=a.nodesAndObject.w, bo=b.nodesAndObject.w;
    if ((objects[ao].flags|objects[bo].flags)&NM_OBJECT_DISABLE_DEFORMABLE_CONTACT) return false;
    if (ao!=bo) return true;
    if (objects[ao].flags&NM_OBJECT_DISABLE_SELF_CONTACT) return false;
    if (a.identity.y==NM_CONTINUUM_SURFACE_POINT && b.identity.y==NM_CONTINUUM_SURFACE_POINT) return false;
    for (unsigned i=0; i<3; ++i) for (unsigned j=0; j<3; ++j) {
        auto an=node(a,i), bn=node(b,j);
        if (an==NM_INVALID_INDEX || bn==NM_INVALID_INDEX) continue;
        if (an==bn) return false;
        if (an>=d.gridNodeCount && bn>=d.gridNodeCount &&
            topology[environment*d.femNodeCount+an-d.gridNodeCount].identity.x ==
            topology[environment*d.femNodeCount+bn-d.gridNodeCount].identity.x) return false;
    }
    return true;
}
}

int main() {
    @autoreleasepool {
        try {
            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            require(device!=nil,"Metal device unavailable");
            NSError* error=nil;
            id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_MATTER_METALLIB] error:&error];
            require(library!=nil,"Matter library unavailable");
            const char* names[]={"nm_contact_sort_surface_primitives","nm_contact_build_surface_bvh_leaves",
                "nm_contact_build_surface_bvh_level","nm_contact_count_deformable_candidates",
                "nm_contact_scan_deformable_candidate_counts","nm_contact_scatter_deformable_candidates"};
            std::array<id<MTLComputePipelineState>,6> pipelines;
            for (unsigned i=0; i<pipelines.size(); ++i) {
                auto name=[@"numi_matter_metal::" stringByAppendingString:[NSString stringWithUTF8String:names[i]]];
                id<MTLFunction> function=[library newFunctionWithName:name];
                require(function!=nil,"broadphase kernel missing");
                pipelines[i]=[device newComputePipelineStateWithFunction:function error:&error];
                require(pipelines[i]!=nil,"broadphase pipeline failed");
            }
            id<MTLCommandQueue> queue=[device newCommandQueue];
            const Fixture fixtures[]={
                {"empty",0,8,0},{"one",1,8,0},{"ragged-31",31,10000,0},
                {"ragged-32",32,10000,0},{"ragged-33",33,10000,0},
                {"ragged-257",257,100000,0},{"ragged-1025",1025,600000,0},
                {"exact-capacity",3,3,1},{"exact-prefix-overflow",4,3,1},
                {"dense-overflow",130,32,1},{"disabled-capacity",33,0,1},
                {"environment-isolation",33,3,3},{"large-separated",16385,8,2}
            };
            for (const auto fixture:fixtures) {
                @autoreleasepool {
                    NMMatterDispatchGPU dispatch{};
                    dispatch.environmentCount=3;
                    dispatch.gridNodeCount=fixture.count;
                    dispatch.femNodeCount=3*fixture.count;
                    dispatch.objectCount=4;
                    dispatch.deformableContactCapacity=fixture.capacity;
                    const auto n=fixture.count, environments=dispatch.environmentCount;
                    const std::uint32_t leaves=n==0 ? 0 : std::bit_ceil(n);
                    std::vector<NMContinuumSurfacePrimitiveGPU> primitives(environments*n);
                    std::vector<NMFEMTopologyNodeGPU> topology(environments*dispatch.femNodeCount);
                    std::vector<NMContinuumObjectGPU> objects(4);
                    if (fixture.mode==0) {
                        objects[1].flags=NM_OBJECT_DISABLE_SELF_CONTACT;
                        objects[2].flags=NM_OBJECT_DISABLE_DEFORMABLE_CONTACT;
                    }
                    for (unsigned e=0; e<environments; ++e) {
                        for (unsigned i=0; i<dispatch.femNodeCount; ++i)
                            topology[e*dispatch.femNodeCount+i].identity.x=i;
                        std::uint32_t random=0x6e756d69u+e;
                        for (unsigned i=0; i<n; ++i) {
                            random^=random<<13; random^=random>>17; random^=random<<5;
                            auto& p=primitives[e*n+i];
                            p.identity={i,NM_CONTINUUM_SURFACE_TRIANGLE,1,NM_TOPOLOGY_ACTIVE};
                            p.nodesAndObject={n+3*i,n+3*i+1,n+3*i+2,fixture.mode==0 ? i%4 : 0};
                            if (fixture.mode==0 && i%7==0) p.identity.w=0;
                            float x=0,y=0,z=0,extent=1;
                            if (fixture.mode==0) {
                                x=float(random&15u)*0.125f;
                                y=float((random>>4)&15u)*0.125f;
                                z=float((random>>8)&7u)*0.125f;
                                extent=0.125f; // exact face/edge touching is inclusive
                                if (i%3==0) {
                                    p.identity.y=NM_CONTINUUM_SURFACE_POINT;
                                    p.nodesAndObject={i,NM_INVALID_INDEX,NM_INVALID_INDEX,p.nodesAndObject.w};
                                } else if (i>0 && i%17==0) {
                                    topology[e*dispatch.femNodeCount+3*i].identity.x=3*(i-1);
                                } else if (i>0 && i%19==0) {
                                    p.nodesAndObject.x=n+3*(i-1);
                                }
                            } else if (fixture.mode==2 || (fixture.mode==3 && e>0)) {
                                x=float(i)*4.f;
                                if (fixture.mode==3 && e==2 && i==1) x=1.f;
                            }
                            p.boundsMinimum={x,y,z,1};
                            p.boundsMaximum={x+extent,y+extent,z+extent,0};
                        }
                    }
                    auto buffer=[&](const auto& values) {
                        const auto bytes=values.size()*sizeof(values[0]);
                        id<MTLBuffer> value=[device newBufferWithLength:std::max<std::size_t>({16,bytes,sizeof(values[0])})
                            options:MTLResourceStorageModeShared];
                        require(value!=nil,"probe allocation failed");
                        std::memset(value.contents,0,value.length);
                        if (bytes) std::memcpy(value.contents,values.data(),bytes);
                        return value;
                    };
                    auto scratch=[&](std::size_t count) {return buffer(std::vector<std::uint32_t>(count));};
                    auto primitiveBuffer=buffer(primitives), topologyBuffer=buffer(topology), objectBuffer=buffer(objects);
                    auto keysA=scratch(environments*n), keysB=scratch(environments*n);
                    auto indicesA=scratch(environments*n), indicesB=scratch(environments*n);
                    auto active=scratch(environments), counts=scratch(environments);
                    auto bounds=scratch(environments*4ull*leaves*4ull);
                    auto candidates=scratch(std::size_t(environments)*fixture.capacity*4);
                    auto statuses=buffer(std::vector<NMMatterStatusGPU>(environments));
                    std::vector<Pair> firstResult;
                    std::vector<std::uint32_t> firstCounts;
                    const auto initialPrimitives=primitives;
                    for (unsigned replay=0; replay<4; ++replay) {
                        if (replay==2 && fixture.mode==0) {
                            for (unsigned e=0; e<environments; ++e) for(unsigned i=0; i<n; ++i) {
                                auto& p=primitives[e*n+i];
                                if (i%2==0) {p.boundsMinimum.x+=32.f;p.boundsMaximum.x+=32.f;}
                                if (i%31==0) p.identity.w^=NM_TOPOLOGY_ACTIVE;
                            }
                        } else if (replay==3) primitives=initialPrimitives;
                        if (!primitives.empty()) std::memcpy(primitiveBuffer.contents,primitives.data(),primitives.size()*sizeof(primitives[0]));
                        std::memset(statuses.contents,0,statuses.length);
                        std::memset(candidates.contents,0xa5,candidates.length);
                        id<MTLCommandBuffer> command=[queue commandBuffer];
                        auto encoder=[command computeCommandEncoder];
                        auto launch=[&](unsigned pipeline, unsigned total, bool groups, const auto& bind) {
                            if (total==0) return;
                            [encoder setComputePipelineState:pipelines[pipeline]];
                            [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0];
                            bind();
                            if (groups) [encoder dispatchThreadgroups:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                            else [encoder dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
                        };
                        launch(0,environments,true,[&] {
                            [encoder setBuffer:primitiveBuffer offset:0 atIndex:1];
                            [encoder setBuffer:keysA offset:0 atIndex:2];[encoder setBuffer:keysB offset:0 atIndex:3];
                            [encoder setBuffer:indicesA offset:0 atIndex:4];[encoder setBuffer:indicesB offset:0 atIndex:5];
                            [encoder setBuffer:active offset:0 atIndex:6];
                        });
                        launch(1,environments*leaves,false,[&] {
                            [encoder setBytes:&leaves length:sizeof(leaves) atIndex:1];
                            [encoder setBuffer:primitiveBuffer offset:0 atIndex:2];[encoder setBuffer:indicesA offset:0 atIndex:3];
                            [encoder setBuffer:active offset:0 atIndex:4];[encoder setBuffer:bounds offset:0 atIndex:5];
                        });
                        for (std::uint32_t width=leaves/2; width; width/=2) launch(2,environments*width,false,[&] {
                            [encoder setBytes:&leaves length:sizeof(leaves) atIndex:1];
                            [encoder setBytes:&width length:sizeof(width) atIndex:2];
                            [encoder setBuffer:bounds offset:0 atIndex:3];
                        });
                        auto query=[&](id<MTLBuffer> values) {
                            [encoder setBytes:&leaves length:sizeof(leaves) atIndex:1];
                            [encoder setBuffer:primitiveBuffer offset:0 atIndex:2];[encoder setBuffer:indicesA offset:0 atIndex:3];
                            [encoder setBuffer:active offset:0 atIndex:4];[encoder setBuffer:bounds offset:0 atIndex:5];
                            [encoder setBuffer:topologyBuffer offset:0 atIndex:6];[encoder setBuffer:objectBuffer offset:0 atIndex:7];
                            [encoder setBuffer:values offset:0 atIndex:8];[encoder setBuffer:candidates offset:0 atIndex:9];
                            [encoder setBuffer:statuses offset:0 atIndex:10];
                        };
                        launch(3,environments*n,false,[&]{query(keysB);});
                        launch(4,environments,true,[&] {
                            [encoder setBuffer:active offset:0 atIndex:1];[encoder setBuffer:keysB offset:0 atIndex:2];
                            [encoder setBuffer:indicesB offset:0 atIndex:3];[encoder setBuffer:counts offset:0 atIndex:4];
                            [encoder setBuffer:statuses offset:0 atIndex:5];
                        });
                        launch(5,environments*n,false,[&]{query(indicesB);});
                        [encoder endEncoding];[command commit];[command waitUntilCompleted];
                        require(command.status==MTLCommandBufferStatusCompleted,"broadphase GPU failed");
                        const auto* order=static_cast<const std::uint32_t*>(indicesA.contents);
                        const auto* activeCounts=static_cast<const std::uint32_t*>(active.contents);
                        const auto* actualCounts=static_cast<const std::uint32_t*>(counts.contents);
                        const auto* status=static_cast<const NMMatterStatusGPU*>(statuses.contents);
                        const auto* output=static_cast<const Pair*>(candidates.contents);
                        for (unsigned e=0;e<environments;++e) {
                            std::vector<unsigned> sorted(order+e*n,order+e*n+activeCounts[e]);
                            auto canonical=sorted;std::sort(canonical.begin(),canonical.end());
                            std::vector<unsigned> expectedActive;
                            for (unsigned i=0;i<n;++i) if (primitives[e*n+i].identity.w&NM_TOPOLOGY_ACTIVE) expectedActive.push_back(i);
                            require(canonical==expectedActive,"sort lost or duplicated an active primitive");
                            std::vector<Pair> expected;
                            if (fixture.capacity && fixture.mode!=2)
                                for (unsigned a=0;a<sorted.size();++a) for(unsigned b=a+1;b<sorted.size();++b)
                                    if(eligible(dispatch,primitives[e*n+sorted[a]],primitives[e*n+sorted[b]],objects,topology,e))
                                        expected.push_back({sorted[a],sorted[b],std::uint32_t(expected.size()),NM_TOPOLOGY_ACTIVE});
                            if (fixture.mode==2) for(unsigned i=1;i<n;++i)
                                require(primitives[e*n+i-1].boundsMaximum.x < primitives[e*n+i].boundsMinimum.x,"analytic separated-box witness failed");
                            const bool overflow=expected.size()>fixture.capacity;
                            require(status[e].code==(overflow?NM_STATUS_CAPACITY_OVERFLOW:NM_STATUS_SUCCESS),"overflow or environment isolation mismatch");
                            require(actualCounts[e]==std::min<std::size_t>(expected.size(),fixture.capacity),"candidate count differs from exhaustive oracle");
                            if (!overflow) for (unsigned i=0;i<expected.size();++i)
                                require(output[e*fixture.capacity+i]==expected[i],"candidate membership/order differs from exhaustive oracle");
                        }
                        const std::vector<Pair> result(output,output+std::size_t(environments)*fixture.capacity);
                        const std::vector<std::uint32_t> resultCounts(actualCounts,actualCounts+environments);
                        if (replay==0) {firstResult=result;firstCounts=resultCounts;}
                        else if (replay!=2) require(result==firstResult && resultCounts==firstCounts,"broadphase replay/restore differs");
                        if (replay==0) std::printf("surface_bvh fixture=%s environments=%u gpu_seconds=%.9f result=pass\n",
                            fixture.name,environments,command.GPUEndTime-command.GPUStartTime);
                    }
                }
            }
            return 0;
        } catch(const std::exception& error) {
            std::fprintf(stderr,"surface_bvh failure=%s\n",error.what());
            return 1;
        }
    }
}
