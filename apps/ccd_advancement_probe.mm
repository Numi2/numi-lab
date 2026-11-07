#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using V=std::array<double,3>;
V add(V a,V b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
V sub(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
V mul(V a,double b){return {a[0]*b,a[1]*b,a[2]*b};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
mr_float4 f4(float x,float y,float z,float w=0){return {x,y,z,w};}
float& component(mr_float4& v,unsigned i){
    switch(i){case 0:return v.x;case 1:return v.y;case 2:return v.z;default:return v.w;}
}
float component(const mr_float4& v,unsigned i){
    switch(i){case 0:return v.x;case 1:return v.y;case 2:return v.z;default:return v.w;}
}
void require(bool c,const std::string& s){if(!c)throw std::runtime_error(s);}
struct Pose { V p; std::array<double,4> q; };
Pose interpolate(const MRProjectedColliderGPU& a,const MRProjectedColliderGPU& b,double t){
    Pose p{}; double sign=0,n=0;
    for(int i=0;i<4;++i)sign+=double(component(a.rotation,i))*double(component(b.rotation,i));
    for(int i=0;i<4;++i){p.q[i]=(1-t)*component(a.rotation,i)+t*(sign<0?-1:1)*component(b.rotation,i);n+=p.q[i]*p.q[i];}
    for(auto& v:p.q)v/=std::sqrt(n);
    for(int i=0;i<3;++i)p.p[i]=(1-t)*component(a.centerAndRadius,i)+t*component(b.centerAndRadius,i);
    return p;
}
std::array<V,3> axes(Pose p){
    const auto q=p.q;
    const V u{q[0],q[1],q[2]};
    std::array<V,3> r{};
    for(int i=0;i<3;++i){V v{};v[i]=1;r[i]=add(v,mul(cross(u,add(cross(u,v),mul(v,q[3]))),2));}
    return r;
}
struct Case{
    std::string name;
    std::array<MRShapeGPU,2> shapes{};
    std::array<MRProjectedColliderGPU,2> start{},end{};
    bool hit=false;
    double missCertificate=0;
};
MRShapeGPU box(V half){
    MRShapeGPU s{};s.shapeType=MR_SHAPE_BOX;s.flags=MR_SHAPE_FLAG_ENABLE_CCD;
    s.dimensions=f4(half[0],half[1],half[2]);
    s.contactRestAndBoundingRadius=f4(.001,0,std::sqrt(dot(half,half)));
    s.localRotation=f4(0,0,0,1);return s;
}
MRProjectedColliderGPU projected(V p,double angle,const MRShapeGPU& s){
    MRProjectedColliderGPU r{};r.centerAndRadius=f4(p[0],p[1],p[2],component(s.contactRestAndBoundingRadius,2));
    r.rotation=f4(0,0,std::sin(angle/2),std::cos(angle/2));
    r.lowerAndHalfLength=f4(p[0]-1,p[1]-1,p[2]-1);
    r.upperAndContactOffset=f4(p[0]+1,p[1]+1,p[2]+1,.001);
    return r;
}
double gap(const Case& c,double t){
    const auto a=interpolate(c.start[0],c.end[0],t),b=interpolate(c.start[1],c.end[1],t);
    const auto ax=axes(a),bx=axes(b);
    if(c.shapes[1].shapeType==MR_SHAPE_PLANE){
        double radius=0;for(int i=0;i<3;++i)radius+=component(c.shapes[0].dimensions,i)*std::abs(dot(ax[i],bx[1]));
        return dot(sub(a.p,b.p),bx[1])-radius;
    }
    double best=-1.e100;
    auto test=[&](V n){const double n2=dot(n,n);if(n2<1.e-22)return;n=mul(n,1/std::sqrt(n2));
        double ra=0,rb=0;for(int i=0;i<3;++i){ra+=component(c.shapes[0].dimensions,i)*std::abs(dot(ax[i],n));rb+=component(c.shapes[1].dimensions,i)*std::abs(dot(bx[i],n));}
        best=std::max(best,std::abs(dot(sub(b.p,a.p),n))-ra-rb);};
    for(auto n:ax)test(n);for(auto n:bx)test(n);for(auto x:ax)for(auto y:bx)test(cross(x,y));
    return best;
}
double firstImpact(const Case& c){
    // FP64 OBB SAT/plane support, independent of the production GJK query.
    constexpr int samples=65536;
    double prev=0;
    for(int i=0;i<=samples;++i){const double t=double(i)/samples;
        if(gap(c,t)<=0){double lo=prev,hi=t;for(int j=0;j<45;++j){double mid=(lo+hi)/2;if(gap(c,mid)<=0)hi=mid;else lo=mid;}return hi;}
        prev=t;}
    return -1;
}
std::vector<Case> cases(double scale){
    const double pi=std::acos(-1.);
    std::vector<Case> out;
    auto plane=[&](std::string name,V a,V b,V half,double ra,double rb,bool hit,double cert){
        Case c;c.name=name;c.shapes[0]=box(half);c.shapes[1].shapeType=MR_SHAPE_PLANE;
        c.shapes[1].localRotation=f4(0,0,0,1);
        c.start={projected(a,ra,c.shapes[0]),projected({0,0,0},0,c.shapes[1])};
        c.end={projected(b,rb,c.shapes[0]),c.start[1]};c.hit=hit;c.missCertificate=cert;out.push_back(c);
    };
    plane("thin-plate-impact",{0,.12,0},{0,-.1,0},{.2,.002,.03},0,0,true,0);
    plane("tangential-plane-miss",{-2,.03,0},{2,.03,0},{.2,.002,.03},0,0,false,.028);
    plane("rotating-plane-hit",{0,.1998,0},{0,.1998,0},{.2,.002,.01},0,pi,true,0);
    plane("rotating-plane-near-miss",{0,.2002,0},{0,.2002,0},{.2,.002,.01},0,pi,false,.2002-std::hypot(.2,.002));
    auto pair=[&](std::string name,V a0,V a1,V b0,V b1,V ah,V bh,double r0,double r1,bool hit,double cert){
        Case c;c.name=name;c.shapes={box(ah),box(bh)};
        c.start={projected(a0,r0,c.shapes[0]),projected(b0,0,c.shapes[1])};
        c.end={projected(a1,r1,c.shapes[0]),projected(b1,0,c.shapes[1])};c.hit=hit;c.missCertificate=cert;out.push_back(c);
    };
    pair("tangential-box-miss",{-.4,.12,0},{.4,.12,0},{0,0,0},{0,0,0},{.1,.005,.02},{.1,.05,.05},0,0,false,.065);
    pair("common-motion-hit",{-.4,.02,0},{10.4,.02,0},{0,0,0},{10,0,0},{.1,.005,.02},{.1,.05,.05},0,0,true,0);
    pair("moving-catch",{-.4,.02,0},{.4,.02,0},{.4,0,0},{-.4,0,0},{.1,.005,.02},{.04,.05,.05},0,0,true,0);
    pair("rotating-edge-hit",{0,0,0},{0,0,0},{.18,.08,0},{.18,.08,0},{.2,.002,.01},{.005,.005,.01},0,pi/2,true,0);
    pair("rotating-edge-miss",{0,0,0},{0,0,0},{.205,.08,0},{.205,.08,0},{.2,.002,.01},{.005,.005,.01},0,pi/2,false,std::hypot(.205,.08)-std::hypot(.2,.002)-std::hypot(.005,.005));
    for(auto& c:out){
        for(auto& s:c.shapes){for(int j=0;j<3;++j)component(s.dimensions,j)*=scale;for(int j=0;j<3;++j)component(s.contactRestAndBoundingRadius,j)*=scale;}
        for(int i=0;i<2;++i){for(int j=0;j<3;++j){component(c.start[i].centerAndRadius,j)*=scale;component(c.end[i].centerAndRadius,j)*=scale;}
            // Match mr_world_project_swept_colliders at speculative scale 1:
            // translation plus the endpoint rotational chord, capped by speed.
            double translation2=0,cosine=0;
            for(int k=0;k<3;++k){const double d=component(c.end[i].centerAndRadius,k)-component(c.start[i].centerAndRadius,k);translation2+=d*d;}
            for(int k=0;k<4;++k)cosine+=double(component(c.start[i].rotation,k))*component(c.end[i].rotation,k);
            cosine=std::clamp(std::abs(cosine),0.,1.);
            const float margin=float(std::sqrt(translation2)+2*c.shapes[i].contactRestAndBoundingRadius.z*std::sqrt(std::max(0.,1-cosine*cosine)));
            c.start[i].statusAndFlags.z=std::bit_cast<std::uint32_t>(margin);
        }
        c.missCertificate*=scale;
    }
    // Same orientation path with the other quaternion representation.
    auto antipodal=out[2];antipodal.name="rotating-antipodal-hit";
    for(int j=0;j<4;++j)component(antipodal.end[0].rotation,j)=-component(antipodal.end[0].rotation,j);
    out.push_back(antipodal);
    return out;
}
struct Result{
    MRCCDPairGPU record{};MRMetalWorldContactStatusGPU status{};
    double gpuSeconds=0;bool replay=false;std::vector<double> gpuSamples;
};
Result run(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLComputePipelineState> pipeline,
           const Case& c,bool directional,int environments,int repeats,float scale){
    MRMetalWorldContactDispatchGPU d{};
    d.environmentCount=environments;d.shapeCount=2;d.eligiblePairCount=1;d.ccdCandidateCapacity=1;
    d.ccdEventCapacity=1;d.maxCCDEvents=1;d.ccdMode=MR_WORLD_CCD_HYBRID;
    d.maxConservativeAdvancementIterations=128;d.flags=directional?MR_METAL_WORLD_CONTACT_DIRECTIONAL_CCD:0;
    d.ccdParameters=f4(1.e-7f,1.e-5f*scale,1,1.e6f);
    const MRCompiledCollisionPairGPU pair{0,1,MR_COLLISION_PAIR_BOX_BOX,MR_INVALID_INDEX};
    std::vector<MRProjectedColliderGPU> start,end;
    std::vector<MRCCDEventStateGPU> events(environments);
    for(int i=0;i<environments;++i){start.insert(start.end(),c.start.begin(),c.start.end());end.insert(end.end(),c.end.begin(),c.end.end());events[i].time=f4(0,1,0,0);}
    const auto buffer=[&](const void* data,std::size_t bytes){return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];};
    std::vector<std::uint32_t> flags(environments,1);
    MRGeometryHeaderGPU header{};mr_float4 vertex{};MRMeshBVHNodeGPU node{};MRMeshTriangleGPU triangle{};
    std::array<id<MTLBuffer>,13> buffers{
        buffer(&d,sizeof d),buffer(c.shapes.data(),sizeof(c.shapes)),buffer(&pair,sizeof pair),
        buffer(flags.data(),flags.size()*sizeof(flags[0])),buffer(start.data(),start.size()*sizeof(start[0])),
        buffer(end.data(),end.size()*sizeof(end[0])),buffer(&header,sizeof header),buffer(&vertex,sizeof vertex),
        buffer(&node,sizeof node),buffer(&triangle,sizeof triangle),
        [device newBufferWithLength:environments*sizeof(MRCCDPairGPU) options:MTLResourceStorageModeShared],
        [device newBufferWithLength:environments*sizeof(MRMetalWorldContactStatusGPU) options:MTLResourceStorageModeShared],
        buffer(events.data(),events.size()*sizeof(events[0]))};
    for(auto b:buffers)require(b!=nil,"allocation failed");
    Result result;std::vector<unsigned char> previous,previousStatus;
    std::vector<double> timings;
    for(int iteration=0;iteration<repeats+2;++iteration){
        std::memset(buffers[10].contents,0,buffers[10].length);std::memset(buffers[11].contents,0,buffers[11].length);
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline];
        for(NSUInteger i=0;i<buffers.size();++i)[encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder dispatchThreads:MTLSizeMake(environments,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(32,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        require(command.status==MTLCommandBufferStatusCompleted,"GPU command failed");
        const auto* records=static_cast<const MRCCDPairGPU*>(buffers[10].contents);
        const auto* statuses=static_cast<const MRMetalWorldContactStatusGPU*>(buffers[11].contents);
        for(int i=1;i<environments;++i){auto a=records[0],b=records[i];b.environment=a.environment;b.stableKeyHigh=a.stableKeyHigh;require(std::memcmp(&a,&b,sizeof a)==0,"batch records differ");require(statuses[i].code==statuses[0].code,"batch status differs");}
        if(iteration==0){const auto* bytes=static_cast<const unsigned char*>(buffers[10].contents);previous.assign(bytes,bytes+buffers[10].length);
            const auto* statusBytes=static_cast<const unsigned char*>(buffers[11].contents);
            previousStatus.assign(statusBytes,statusBytes+buffers[11].length);}
        else {
            require(std::memcmp(previous.data(),buffers[10].contents,previous.size())==0,"CCD replay differs");
            require(std::memcmp(previousStatus.data(),buffers[11].contents,previousStatus.size())==0,"CCD status replay differs");
        }
        result.record=records[0];result.status=statuses[0];result.replay=true;
        if(iteration>=2)timings.push_back(command.GPUEndTime-command.GPUStartTime);
    }
    for(double t:timings)require(std::isfinite(t)&&t>0,"GPU timestamp unavailable");
    result.gpuSamples=timings;
    std::sort(timings.begin(),timings.end());result.gpuSeconds=timings[timings.size()/2];return result;
}
}
int main(int argc,char** argv){
 @autoreleasepool{
  try{
    require(argc>=2,"usage: ccd_advancement_probe METALLIB [scale] [environments] [repeats]");
    const double scale=argc>2?std::stod(argv[2]):1;
    const int environments=argc>3?std::stoi(argv[3]):256,repeats=argc>4?std::stoi(argv[4]):7;
    require(scale>0 && environments>0 && environments<=8192 && repeats>0,"invalid experiment size");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"Metal unavailable");
    NSError* error=nil;
    id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
    require(library!=nil,error?error.localizedDescription.UTF8String:"library unavailable");
    id<MTLFunction> function=[library newFunctionWithName:@"mr_world_resolve_ccd"];
    require(function!=nil,"production CCD kernel unavailable");
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline!=nil,error?error.localizedDescription.UTF8String:"pipeline unavailable");
    id<MTLCommandQueue> queue=[device newCommandQueue];
    int failures=0;std::cout<<std::setprecision(12)<<"{\"device\":\""<<device.name.UTF8String<<"\",\"scale\":"<<scale<<",\"environments\":"<<environments<<",\"repeats\":"<<repeats<<",\"cases\":[";
    bool first=true;int unit=0;
    for(const auto& c:cases(scale)){
        const double reference=firstImpact(c);
        const bool referenceValid=c.hit?reference>=0:c.missCertificate>2.e-5*scale && reference<0;
        require(referenceValid,c.name+": reference calibration failed");
        std::array<Result,2> r;
        // Alternate arm order deterministically between experimental units.
        const int order=(unit++)%2;
        r[order]=run(device,queue,pipeline,c,order!=0,environments,repeats,scale);
        r[1-order]=run(device,queue,pipeline,c,order==0,environments,repeats,scale);
        if(!first)std::cout<<",";first=false;
        std::cout<<"{\"name\":\""<<c.name<<"\",\"reference_toi\":"<<reference<<",\"miss_certificate\":"<<c.missCertificate<<",\"arms\":[";
        for(int arm=0;arm<2;++arm){
            const auto& v=r[arm];const bool hit=(v.record.flags&MR_CCD_PAIR_HAS_IMPACT)!=0;
            const bool unresolved=(v.record.flags&MR_CCD_PAIR_UNRESOLVED)!=0;
            const bool valid=c.hit?(hit || unresolved):!hit;
            const bool early= !hit || reference<0 || component(v.record.intervalAndDistance,0)<=reference+2.e-5;
            if(arm==1 && (!valid || !early || !v.replay))++failures;
            if(arm)std::cout<<",";
            std::cout<<"{\"directional\":"<<(arm?"true":"false")<<",\"flags\":"<<v.record.flags<<",\"status\":"<<v.status.code<<",\"iterations\":"<<component(v.record.normalAndIteration,3)<<",\"toi_lower\":"<<component(v.record.intervalAndDistance,0)<<",\"toi_upper\":"<<component(v.record.intervalAndDistance,1)<<",\"gpu_seconds_median\":"<<v.gpuSeconds<<",\"unresolved\":"<<(unresolved?"true":"false")<<",\"valid\":"<<(valid&&early?"true":"false")<<",\"replay\":"<<(v.replay?"true":"false")<<",\"gpu_seconds_samples\":[";
            for(std::size_t k=0;k<v.gpuSamples.size();++k){if(k)std::cout<<",";std::cout<<v.gpuSamples[k];}
            std::cout<<"]}";
        }
        std::cout<<"]}";
    }
    std::cout<<"],\"failures\":"<<failures<<",\"scope\":\"production CCD kernel; no integration, contact forces, friction or accepted simulation throughput\"}\n";
    return failures?1:0;
  }catch(const std::exception& e){std::cerr<<"ccd_advancement_probe: "<<e.what()<<"\n";return 1;}
 }
}
