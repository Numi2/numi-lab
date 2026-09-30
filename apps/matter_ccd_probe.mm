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
            // Source-bound rejected PCL/ACL Newton correction from the native
            // eight-step Human run. The exact Float32 start has no strict
            // triangle crossing; the finish does. At least one feature CCD
            // must therefore report impact before the correction is applied.
            const unsigned kneeFirst=static_cast<unsigned>(cases.size());
            const std::array<V,6> kneeStart={
                V{.0519178249f,.156561702f,.45247823f,0},
                V{.0516432077f,.156714752f,.452109039f,0},
                V{.0517562628f,.1571358f,.452465057f,0},
                V{.0517320521f,.157598406f,.452421576f,0},
                V{.0517004244f,.157034218f,.452237159f,0},
                V{.0518251546f,.157020926f,.452557236f,0},
            };
            const std::array<V,6> kneeFinish={
                kneeStart[0],kneeStart[1],kneeStart[2],
                V{.0517288893f,.157598242f,.452422321f,0},
                V{.0516972058f,.157033727f,.452238292f,0},
                V{.0518222786f,.157020241f,.452558249f,0},
            };
            constexpr float kneeThickness=8.7e-7f;
            for(unsigned direction=0;direction<2;++direction)
                for(unsigned vertex=0;vertex<3;++vertex) {
                    const unsigned v=(direction==0?0:3)+vertex;
                    const unsigned t=direction==0?3:0;
                    auto c=make(("knee-vt-"+std::to_string(direction)+"-"+
                                 std::to_string(vertex)).c_str(),false,false,
                        {kneeStart[v],kneeStart[t],kneeStart[t+1],kneeStart[t+2]},
                        {kneeFinish[v],kneeFinish[t],kneeFinish[t+1],kneeFinish[t+2]});
                    c.data[0].y=kneeThickness;
                    cases.push_back(c);
                }
            constexpr std::array<std::array<unsigned,2>,3> edges={{{0,1},{1,2},{2,0}}};
            for(unsigned a=0;a<3;++a) for(unsigned b=0;b<3;++b) {
                const auto p=edges[a],q=edges[b];
                auto c=make(("knee-ee-"+std::to_string(a)+"-"+
                             std::to_string(b)).c_str(),true,false,
                    {kneeStart[p[0]],kneeStart[p[1]],
                     kneeStart[3+q[0]],kneeStart[3+q[1]]},
                    {kneeFinish[p[0]],kneeFinish[p[1]],
                     kneeFinish[3+q[0]],kneeFinish[3+q[1]]});
                c.data[0].y=kneeThickness;
                cases.push_back(c);
            }
            const unsigned stepSevenFirst=static_cast<unsigned>(cases.size());
            // Native attempted step seven, PCL primitive 314854 against ACL
            // primitive 463889. Replay both triangle directions because the
            // contact status records only the vertex index, not its side.
            const std::array<V,6> stepSevenStart={
                V{.0519175529f,.156560957f,.452478677f,0},
                V{.0521185212f,.156000853f,.452395082f,0},
                V{.0518133827f,.156181425f,.452050447f,0},
                V{.0520323887f,.156485274f,.452613503f,0},
                V{.0518768132f,.156482756f,.452299565f,0},
                V{.0518251061f,.157021731f,.452557057f,0},
            };
            const std::array<V,6> stepSevenFinish={
                V{.0519145131f,.156559274f,.452479839f,0},
                V{.0521184094f,.156000793f,.452395201f,0},
                V{.0518132709f,.156181365f,.452050537f,0},
                V{.0520265661f,.156481624f,.452616811f,0},
                V{.0518746786f,.156479746f,.452300638f,0},
                V{.0518273115f,.157022625f,.452556282f,0},
            };
            for(unsigned direction=0;direction<2;++direction)
                for(unsigned vertex=0;vertex<3;++vertex) {
                    const unsigned v=(direction==0?0:3)+vertex;
                    const unsigned t=direction==0?3:0;
                    auto c=make(("step7-pcl-acl-vt-"+
                                 std::to_string(direction)+"-"+
                                 std::to_string(vertex)).c_str(),false,false,
                        {stepSevenStart[v],stepSevenStart[t],
                         stepSevenStart[t+1],stepSevenStart[t+2]},
                        {stepSevenFinish[v],stepSevenFinish[t],
                         stepSevenFinish[t+1],stepSevenFinish[t+2]});
                    c.data[0].y=1.e-5f;
                    cases.push_back(c);
                }
            for(unsigned a=0;a<3;++a) for(unsigned b=0;b<3;++b) {
                const auto p=edges[a],q=edges[b];
                auto c=make(("step7-pcl-acl-ee-"+std::to_string(a)+"-"+
                             std::to_string(b)).c_str(),true,false,
                    {stepSevenStart[p[0]],stepSevenStart[p[1]],
                     stepSevenStart[3+q[0]],stepSevenStart[3+q[1]]},
                    {stepSevenFinish[p[0]],stepSevenFinish[p[1]],
                     stepSevenFinish[3+q[0]],stepSevenFinish[3+q[1]]});
                c.data[0].y=1.e-5f;
                c.hit=a==0 && b==0;
                cases.push_back(c);
            }
            std::vector<V> input;
            for(unsigned i=0;i<cases.size();++i) {
                const auto& c=cases[i];
                if(i<kneeFirst && !c.hit)
                    require(missMargin(c)>0,"FP64 sweep miss certificate failed");
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
            unsigned failed=0,kneeHits=0;
            for(unsigned i=0;i<count;++i) {
                const V r=output[i];
                const bool knee=i>=kneeFirst;
                const bool hitExpected=knee
                    ? (!cases[i].hit || r.x==1.f)
                    : r.x==(cases[i].hit?1.f:0.f);
                const bool ok=hitExpected &&
                    r.y==1.f && std::isfinite(r.z) && std::isfinite(r.w) &&
                    r.z>=0 && r.z<=1;
                if(knee && i<stepSevenFirst && r.x==1.f) ++kneeHits;
                std::printf("%s %s hit=%.0f certified=%.0f toi=%.9g separation=%.9g fp64_miss_margin=%.12g\n",ok?"PASS":"FAIL",cases[i].name.c_str(),r.x,r.y,r.z,r.w,knee||cases[i].hit?0:missMargin(cases[i]));
                failed+=!ok;
            }
            if(kneeHits==0) ++failed;
            std::printf("knee_feature_ccd_hits=%u required_minimum=1\n",kneeHits);
            std::printf("matter_ccd_cases=%u passed=%u failed=%u\n",count,count-failed,failed);
            return failed?1:0;
        } catch(const std::exception& error) {
            std::fprintf(stderr,"matter_ccd_probe: %s\n",error.what()); return 1;
        }
    }
}
