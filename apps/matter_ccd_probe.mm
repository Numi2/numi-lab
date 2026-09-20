#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <simd/simd.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using V = simd_float4;
struct Case {
    std::string name;
    std::array<V, 9> data;
    bool hit;
    std::array<double, 3> axis; // independent fixed-plane FP64 miss witness
};
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
Case make(const char* name, bool edge, bool hit, std::array<V,4> start,
          std::array<V,4> finish, std::array<double,3> axis = {0,0,1}) {
    Case c{name, {}, hit, axis};
    c.data[0] = V{edge ? 1.f : 0.f, 1.e-5f, 0, 0};
    for (unsigned i=0; i<4; ++i) { c.data[1+i]=start[i]; c.data[5+i]=finish[i]; }
    return c;
}
// Independent FP64 certificate: all cross-feature projected differences at
// both sweep endpoints exceed physical thickness. Affine interpolation and
// convex combinations then prove separation at every time, not just samples.
double missMargin(const Case& c) {
    double gap = std::numeric_limits<double>::infinity();
    const unsigned firstCount = c.data[0].x == 1 ? 2 : 1;
    const double norm = std::sqrt(c.axis[0]*c.axis[0]+c.axis[1]*c.axis[1]+c.axis[2]*c.axis[2]);
    for (unsigned end=0; end<2; ++end)
        for (unsigned a=0; a<firstCount; ++a)
            for (unsigned b=firstCount; b<4; ++b) {
                double projection = 0;
                for (unsigned k=0; k<3; ++k)
                    projection += (double(c.data[1+4*end+a][k])-double(c.data[1+4*end+b][k]))*c.axis[k]/norm;
                gap = std::min(gap, projection);
            }
    return gap-double(c.data[0].y);
}
}
int main() {
    @autoreleasepool {
        try {
            std::vector<Case> cases;
            const std::array<V,4> vt = {V{0,0,.01f,0},V{-.1f,-.1f,0,0},V{.1f,-.1f,0,0},V{0,.1f,0,0}};
            auto end=vt; end[0].z=-.01f;
            cases.push_back(make("vertex-triangle-through",false,true,vt,end));
            end=vt; end[0].z=.02f;
            cases.push_back(make("vertex-triangle-separating",false,false,vt,end));
            cases.push_back(make("vertex-triangle-static",false,false,vt,vt));
            auto grazing=vt; grazing[0].z=0.5e-5f; end=grazing; end[0].x=.02f;
            cases.push_back(make("vertex-triangle-within-thickness",false,true,grazing,end));
            const std::array<V,4> ee = {V{-.1f,0,.01f,0},V{.1f,0,.01f,0},V{0,-.1f,0,0},V{0,.1f,0,0}};
            end=ee; end[0].z=end[1].z=-.01f;
            cases.push_back(make("edge-edge-through",true,true,ee,end));
            end=ee; end[0].z=end[1].z=.02f;
            cases.push_back(make("edge-edge-separating",true,false,ee,end));
            cases.push_back(make("edge-edge-static",true,false,ee,ee));
            grazing=ee; grazing[0].z=grazing[1].z=0.5e-5f; end=grazing; end[0].x+=.01f; end[1].x+=.01f;
            cases.push_back(make("edge-edge-within-thickness",true,true,grazing,end));
            end=ee; for(auto& p:end) p.x+=10.f;
            cases.push_back(make("edge-edge-coherent-translation",true,false,ee,end));
            cases.push_back(make("human-step7-p177-p487-e0-e2",true,false,
                {V{-.134162337f,.0598024838f,1.26404393f,0},V{-.137991428f,.0590338968f,1.26189733f,0},
                 V{-.130926281f,.0535416827f,1.25419676f,0},V{-.138660356f,.0612878054f,1.26552224f,0}},
                {V{-.134162620f,.0598024912f,1.26404393f,0},V{-.137998044f,.0590352193f,1.26189840f,0},
                 V{-.130926341f,.0535416193f,1.25419676f,0},V{-.138660491f,.0612879246f,1.26552224f,0}},
                {-.11288183,-.85436664,.50726259}));
            const auto originals=cases;
            // Coordinate permutations/reflections exercise each axis while
            // preserving the exact represented geometry and analytic witness.
            for(unsigned variant=1;variant<6;++variant) for(const auto& original:originals) {
                auto c=original; c.name+="-transform"+std::to_string(variant);
                for(unsigned k=0;k<3;++k) c.axis[k]=original.axis[(k+variant)%3]*(variant>=3?-1:1);
                for(unsigned i=1;i<9;++i) for(unsigned k=0;k<3;++k)
                    c.data[i][k]=original.data[i][(k+variant)%3]*(variant>=3?-1:1);
                cases.push_back(c);
            }
            std::vector<V> input;
            for(const auto& c:cases) {
                if(!c.hit) require(missMargin(c)>0,"FP64 sweep miss certificate failed");
                input.insert(input.end(),c.data.begin(),c.data.end());
            }
            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            require(device!=nil,"Metal device unavailable");
            NSError* error=nil;
            id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_MATTER_METALLIB] error:&error];
            require(library!=nil,"Matter metallib unavailable");
            id<MTLFunction> function=[library newFunctionWithName:@"numi_matter_metal::nm_contact_ccd_regression"];
            require(function!=nil,"production CCD regression kernel unavailable");
            id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
            require(pipeline!=nil,"CCD pipeline failed");
            id<MTLBuffer> source=[device newBufferWithBytes:input.data() length:input.size()*sizeof(V) options:MTLResourceStorageModeShared];
            id<MTLBuffer> result=[device newBufferWithLength:cases.size()*sizeof(V) options:MTLResourceStorageModeShared];
            require(source!=nil && result!=nil,"CCD buffer allocation failed");
            id<MTLCommandQueue> queue=[device newCommandQueue];
            id<MTLCommandBuffer> command=[queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            require(encoder!=nil,"CCD command encoder unavailable");
            const unsigned count=static_cast<unsigned>(cases.size());
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:source offset:0 atIndex:0];
            [encoder setBuffer:result offset:0 atIndex:1];
            [encoder setBytes:&count length:sizeof(count) atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(count,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            require(command.status==MTLCommandBufferStatusCompleted,"CCD GPU command failed");
            const auto* output=static_cast<const V*>(result.contents);
            unsigned failed=0;
            for(unsigned i=0;i<count;++i) {
                const V r=output[i];
                const bool ok=r.x==(cases[i].hit?1.f:0.f) && r.y==1.f && std::isfinite(r.z) && std::isfinite(r.w) && r.z>=0 && r.z<=1;
                std::printf("%s %s hit=%.0f certified=%.0f toi=%.9g separation=%.9g fp64_miss_margin=%.12g\n",ok?"PASS":"FAIL",cases[i].name.c_str(),r.x,r.y,r.z,r.w,cases[i].hit?0:missMargin(cases[i]));
                failed+=!ok;
            }
            std::printf("matter_ccd_cases=%u passed=%u failed=%u\n",count,count-failed,failed);
            return failed?1:0;
        } catch(const std::exception& error) {
            std::fprintf(stderr,"matter_ccd_probe: %s\n",error.what()); return 1;
        }
    }
}
