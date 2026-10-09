#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/numi_human_resting_bed_gpu.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
struct Result { mr_float4 pointGap; mr_float4 normal; mr_uint4 status; };
static_assert(sizeof(Result)==48);
static unsigned checks=0;
void check(bool ok,const char* label){++checks;if(!ok)throw std::runtime_error(label);}
using V=std::array<double,3>;
V sub(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
int main(int argc,char** argv){@autoreleasepool{try{
    check(argc==2,"metallib argument required");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();check(device!=nil,"Apple Metal device unavailable");
    NSError* error=nil;
    id<MTLLibrary> library=[device newLibraryWithFile:@(argv[1]) error:&error];
    check(library!=nil,"bed fixture metallib unavailable");
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"numi_human_bed_query_check"] error:&error];
    check(pipeline!=nil,"bed fixture pipeline unavailable");
    MRHumanRestingBedGPU bed{{3,3,1,1},{-.4f,-1.f,.2f,.3f}};
    std::vector<float> heights{.01f,.03f,.015f,.05f,.04f,.08f,.02f,.07f,.09f};
    std::vector<mr_float4> points;
    const auto node=[&](unsigned x,unsigned y)->V{return {
        std::fma(float(x),bed.originSpacing.z,bed.originSpacing.x),
        std::fma(float(y),bed.originSpacing.w,bed.originSpacing.y),heights[y*3+x]};};
    for(unsigned y=0;y<2;++y)for(unsigned x=0;x<2;++x)for(unsigned upper=0;upper<2;++upper)
        for(unsigned sample=0;sample<12;++sample) {
            const float f=upper?.62f+.02f*sample:.12f+.02f*sample;
            const float g=upper?.68f-.01f*sample:.19f+.01f*sample;
            const auto a=node(x,y),b=node(x+1,y+1);
            points.push_back({float(a[0]+f*(b[0]-a[0])),float(a[1]+g*(b[1]-a[1])),.12f,0});
        }
    const unsigned interior=unsigned(points.size());
    // Exact rounded nodes and closed outer boundary are valid.
    for(unsigned y=0;y<3;++y)for(unsigned x=0;x<3;++x){auto p=node(x,y);points.push_back({float(p[0]),float(p[1]),float(p[2]),0});}
    const unsigned validCount=unsigned(points.size());
    points.push_back({std::nextafter(bed.originSpacing.x,-INFINITY),-.7f,.1f,0});
    // A normal-range separation avoids Metal flushing nextafter(0,+inf) to zero.
    points.push_back({float(node(2,2)[0])+1e-6f,-.7f,.1f,0});
    points.push_back({-.2f,std::nextafter(bed.originSpacing.y,-INFINITY),.1f,0});
    points.push_back({-.2f,std::nextafter(float(node(2,2)[1]),INFINITY),.1f,0});
    points.push_back({NAN,-.7f,.1f,0});
    const uint32_t count=uint32_t(points.size());
    id<MTLBuffer> hb=[device newBufferWithBytes:heights.data() length:heights.size()*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> pb=[device newBufferWithBytes:points.data() length:points.size()*sizeof(mr_float4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> rb=[device newBufferWithLength:points.size()*sizeof(Result) options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue=[device newCommandQueue];
    const auto run=[&](){
        id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pipeline];[e setBytes:&bed length:sizeof(bed) atIndex:0];
        [e setBuffer:hb offset:0 atIndex:1];[e setBuffer:pb offset:0 atIndex:2];[e setBuffer:rb offset:0 atIndex:3];
        [e setBytes:&count length:sizeof(count) atIndex:4];
        [e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e endEncoding];[cb commit];[cb waitUntilCompleted];check(cb.status==MTLCommandBufferStatusCompleted,"bed GPU command failed");
    };
    run();const auto* results=static_cast<const Result*>(rb.contents);
    double maxGap=0,maxNormal=0;
    for(unsigned i=0;i<count;++i){
        const auto& r=results[i];
        if(bool(r.status.x)!=(i<validCount))std::cerr<<"domain_mismatch index="<<i
            <<" xyz="<<points[i].x<<","<<points[i].y<<","<<points[i].z
            <<" actual="<<r.status.x<<" expected="<<(i<validCount)<<std::endl;
        check(bool(r.status.x)==(i<validCount),"finite bed domain classification");
        if(i>=validCount)continue;
        check(r.status.y<8,"facet id in bounded grid");
        const unsigned cell=r.status.y/2,x=cell%2,y=cell/2;const bool upper=r.status.y%2;
        if(i<interior)check(upper==bool((i/12)%2),"expected interior facet side");
        const auto p0=upper?node(x+1,y+1):node(x,y);
        const auto p1=upper?node(x,y+1):node(x+1,y);
        const auto p2=upper?node(x+1,y):node(x,y+1);
        auto n=cross(sub(p1,p0),sub(p2,p0));const auto length=std::sqrt(dot(n,n));for(auto& z:n)z/=length;
        const auto& p=points[i];const double gap=dot(sub({p.x,p.y,p.z},p0),n);
        maxGap=std::max(maxGap,std::abs(gap-r.pointGap.w));
        maxNormal=std::max({maxNormal,std::abs(n[0]-r.normal.x),std::abs(n[1]-r.normal.y),std::abs(n[2]-r.normal.z)});
        check(std::abs(gap-r.pointGap.w)<2e-7,"gap equals actual rendered F32 triangle plane");
        check(maxNormal<2e-6,"normal equals triangle cross product");
        check(r.pointGap.x==float(p0[0])&&r.pointGap.y==float(p0[1])&&r.pointGap.z==float(p0[2]),"plane anchor is fixed mesh vertex");
    }
    // Malformed geometry is not silently clamped into a plausible bed.
    bed.counts.w=0;run();for(unsigned i=0;i<count;++i)check(results[i].status.x==0,"invalid bed ABI rejected");
    bed.counts.w=1;bed.originSpacing.z=0;run();for(unsigned i=0;i<count;++i)check(results[i].status.x==0,"zero grid spacing rejected");
    bed.originSpacing.z=.2f;static_cast<float*>(hb.contents)[4]=NAN;run();
    for(unsigned i=0;i<interior;++i)check(results[i].status.x==0,"nonfinite facet height rejected");
    std::cout<<"device="<<device.name.UTF8String<<" checks="<<checks<<" max_gap_error_m="<<maxGap
        <<" max_normal_error="<<maxNormal<<" status=passed scope=production_bed_query_not_full_human"<<std::endl;
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}}
