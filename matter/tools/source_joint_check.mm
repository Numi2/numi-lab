#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/source_cylindrical_joint.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace numi_matter_joint;
namespace {
void require(bool ok,const std::string& why) { if(!ok) throw std::runtime_error(why); }
template<class T> std::array<T,21> values(Output<T> o) {
    return {o.forceA.x,o.forceA.y,o.forceA.z,o.momentA.x,o.momentA.y,o.momentA.z,
        o.forceB.x,o.forceB.y,o.forceB.z,o.momentB.x,o.momentB.y,o.momentB.z,
        o.gap.x,o.gap.y,o.gap.z,o.angularGap.x,o.angularGap.y,o.angularGap.z,
        o.connectorMoment.x,o.connectorMoment.y,o.connectorMoment.z};
}
template<class T> Input<T> perturb(Input<T> a,Direction<T> d,T h) {
    a.positionA=a.positionA+d.positionA*h;a.positionB=a.positionB+d.positionB*h;
    a.rotationA=multiply(exponential(d.rotationA*h),a.rotationA);
    a.rotationB=multiply(exponential(d.rotationB*h),a.rotationB);return a;
}
template<class T> V<T> readV(std::istream& in) { V<T> v;in>>v.x>>v.y>>v.z;return v; }
Input<double> readInput(std::istream& in) {
    Input<double> a;
    a.referenceA=readV<double>(in);a.referenceB=readV<double>(in);
    a.positionA=readV<double>(in);a.positionB=readV<double>(in);
    a.origin=readV<double>(in);a.axis=readV<double>(in);
    in>>a.rotationA.x>>a.rotationA.y>>a.rotationA.z>>a.rotationA.w;
    in>>a.rotationB.x>>a.rotationB.y>>a.rotationB.z>>a.rotationB.w;
    a.forceMultiplier=readV<double>(in);a.momentMultiplier=readV<double>(in);
    in>>a.forcePenalty>>a.momentPenalty>>a.translation>>a.rotation>>a.axialForce>>a.axialMoment;
    in>>a.prescribedTranslation>>a.prescribedRotation;return a;
}
Input<float> quantize(Input<double> a) {
    // Explicit scalar layout: both template instantiations have 38 scalar
    // fields followed by two uint flags; no vector SIMD padding.
    static_assert(sizeof(Input<float>)==40*sizeof(float));
    static_assert(offsetof(Input<double>,prescribedTranslation)==38*sizeof(double));
    std::array<double,38> x;std::memcpy(x.data(),&a,sizeof(x));
    std::array<float,38> y;for(unsigned i=0;i<38;++i)y[i]=float(x[i]);
    Input<float> b;std::memcpy(&b,y.data(),sizeof(y));
    b.prescribedTranslation=a.prescribedTranslation;b.prescribedRotation=a.prescribedRotation;return b;
}
void gpu(const std::vector<Input<double>>& source,const std::vector<Direction<double>>& directions) {
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"no Metal device");
    NSError* error=nil;
    id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_JOINT_CHECK_METALLIB] error:&error];
    require(library!=nil,"no joint operator library");
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"source_joint_check"] error:&error];
    require(pipeline!=nil,"no joint operator pipeline");
    std::vector<Input<float>> a;std::vector<Direction<float>> d;
    for(size_t i=0;i<source.size();++i) {
        a.push_back(quantize(source[i]));
        std::array<double,12> x;std::memcpy(x.data(),&directions[i],sizeof(x));
        std::array<float,12> y;for(unsigned j=0;j<12;++j)y[j]=float(x[j]);
        Direction<float> df;std::memcpy(&df,y.data(),sizeof(y));d.push_back(df);
    }
    auto buffer=[&](const void* p,size_t n){return [device newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];};
    id<MTLBuffer> ab=buffer(a.data(),a.size()*sizeof(a[0])),db=buffer(d.data(),d.size()*sizeof(d[0]));
    id<MTLBuffer> ob=[device newBufferWithLength:a.size()*sizeof(Output<float>) options:MTLResourceStorageModeShared];
    id<MTLBuffer> tb=[device newBufferWithLength:a.size()*sizeof(Output<float>) options:MTLResourceStorageModeShared];
    id<MTLBuffer> sb=[device newBufferWithLength:a.size()*sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    require(ab&&db&&ob&&tb&&sb&&encoder,"joint buffers unavailable");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:ab offset:0 atIndex:0];[encoder setBuffer:db offset:0 atIndex:1];
    [encoder setBuffer:ob offset:0 atIndex:2];[encoder setBuffer:tb offset:0 atIndex:3];[encoder setBuffer:sb offset:0 atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(a.size(),1,1) threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    require(command.status==MTLCommandBufferStatusCompleted,"joint Metal command failed");
    auto out=static_cast<const Output<float>*>(ob.contents),tangent=static_cast<const Output<float>*>(tb.contents);
    auto status=static_cast<const uint32_t*>(sb.contents);double worst=0;
    for(size_t i=0;i<a.size();++i) {
        Output<float> ref{},jac{};bool accepted=evaluate(a[i],d[i],ref,jac);
        require(status[i]==unsigned(accepted),"Metal admission mismatch");
        if(!accepted)continue;
        for(unsigned k=0;k<2;++k) {
            auto x=values(k?tangent[i]:out[i]),y=values(k?jac:ref);
            for(unsigned j=0;j<21;++j) {
                // Scale each physical block by the relevant authored penalty,
                // preserving a fixed FP32 arithmetic gate near cancellation.
                double scale=(j<12||j>=18)?std::max(a[i].forcePenalty,a[i].momentPenalty):1.0;
                double e=std::abs(double(x[j])-y[j])/std::max(scale,std::abs(double(y[j])));
                require(std::isfinite(e)&&e<2e-5,"Metal joint value mismatch");worst=std::max(worst,e);
            }
        }
    }
    std::cout<<"metal_rows="<<a.size()<<" max_scaled_error="<<worst<<" device="<<[device.name UTF8String]<<"\n";
}
}
int main(int argc,char** argv) { @autoreleasepool { try {
    bool cpu=argc==2&&std::string(argv[1])=="--cpu-only";
    bool rows=argc==4&&std::string(argv[1])=="--source-rows";
    require(argc==1||cpu||rows,"usage: source-joint-check [--cpu-only | --source-rows INPUT OUTPUT]");
    std::vector<Input<double>> inputs;std::vector<Direction<double>> directions;
    double worst=0,worstVirtualWork=0;
    for(unsigned i=0;i<96;++i) {
        Input<double> a{};a.referenceA={-1,2,4};a.referenceB={3,-2,1};a.origin={2,3,5};a.axis={.36,-.48,.8};
        a.rotationA=exponential(V<double>{.001*i,-.003*i,.002*i});
        a.rotationB=exponential(V<double>{-.002*i,.004*i,.003*i});
        a.positionA=a.referenceA+V<double>{.01*i,0,0};a.positionB=a.referenceB+V<double>{0,.02*i,-.01*i};
        a.forcePenalty=10000;a.momentPenalty=3000000;
        a.prescribedTranslation=i%2;a.prescribedRotation=(i/2)%2;a.translation=.02;a.rotation=-.6;
        a.axialForce=a.prescribedTranslation?0:3;a.axialMoment=a.prescribedRotation?0:7;
        a.forceMultiplier={1,2,3};a.momentMultiplier={2,-1,4};
        Direction<double> d{{.2,-.3,.1},{-.1,.3,.2},{.1,-.1,.2},{-.2,.1,.3}};
        Output<double> out{},jac{},hi{},lo{},unused{};
        require(evaluate(a,d,out,jac),"valid joint rejected");
        require(evaluate(perturb(a,d,1e-6),Direction<double>{},hi,unused)&&evaluate(perturb(a,d,-1e-6),Direction<double>{},lo,unused),"difference failed");
        auto x=values(hi),y=values(lo),z=values(jac);
        for(unsigned j=0;j<21;++j) {
            double fd=(x[j]-y[j])/2e-6,e=std::abs(fd-z[j])/std::max(1.0,std::abs(z[j]));
            worst=std::max(worst,e);require(e<2e-6,"joint tangent differs from central difference");
        }
        // Pull the complete source wrench (including both body couples) back
        // through a six-column generalized motion map. Check
        // f^T J qdot == (J^T f)^T qdot without imposing a conservative-energy
        // assumption on FEBio's penalty residual.
        auto pullbackInput=a;pullbackInput.forceMultiplier={};pullbackInput.momentMultiplier={};
        pullbackInput.axialForce=pullbackInput.axialMoment=0;
        Output<double> wrench{},wrenchTangent{};
        require(evaluate(pullbackInput,Direction<double>{},wrench,wrenchTangent),"wrench state rejected");
        std::array<MotionColumn<double>,6> columns{};
        std::array<double,6> rates{};
        for(unsigned c=0;c<columns.size();++c) {
            const double s=double(c+1)/7.0;
            columns[c]={{d.positionA.x*s,d.positionA.y*(1-s),d.positionA.z*s},
                        {d.rotationA.x*(1-s),d.rotationA.y*s,d.rotationA.z*(1-s)},
                        {d.positionB.x*(1-s),d.positionB.y*s,d.positionB.z*(1-s)},
                        {d.rotationB.x*s,d.rotationB.y*(1-s),d.rotationB.z*s}};
            rates[c]=double(int(c%3)-1)*.37+s;
        }
        MotionColumn<double> combined{};double pulledPower=0,powerScale=0;
        for(unsigned c=0;c<columns.size();++c) {
            combined.linearA=combined.linearA+columns[c].linearA*rates[c];
            combined.angularA=combined.angularA+columns[c].angularA*rates[c];
            combined.linearB=combined.linearB+columns[c].linearB*rates[c];
            combined.angularB=combined.angularB+columns[c].angularB*rates[c];
            const double coordinatePower=generalizedForce(wrench,columns[c])*rates[c];
            pulledPower+=coordinatePower;powerScale+=std::abs(coordinatePower);
        }
        const double directPower=wrench.forceA.x*combined.linearA.x+
            wrench.forceA.y*combined.linearA.y+wrench.forceA.z*combined.linearA.z+
            wrench.momentA.x*combined.angularA.x+wrench.momentA.y*combined.angularA.y+
            wrench.momentA.z*combined.angularA.z+wrench.forceB.x*combined.linearB.x+
            wrench.forceB.y*combined.linearB.y+wrench.forceB.z*combined.linearB.z+
            wrench.momentB.x*combined.angularB.x+wrench.momentB.y*combined.angularB.y+
            wrench.momentB.z*combined.angularB.z;
        const double virtualWorkError=std::abs(pulledPower-directPower)/
            std::max(1.0,powerScale);
        require(std::isfinite(virtualWorkError)&&virtualWorkError<5e-15,
                "source joint wrench pullback violates virtual-work identity");
        worstVirtualWork=std::max(worstVirtualWork,virtualWorkError);
        inputs.push_back(a);directions.push_back(d);
    }
    auto free=inputs[0];free.positionA=free.referenceA;free.rotationA={};free.rotationB=exponential(free.axis*1.1);
    free.positionB=free.origin+free.axis*7.0-rotate(free.rotationB,free.origin-free.referenceB);
    free.forceMultiplier={};free.momentMultiplier={};free.axialForce=free.axialMoment=0;
    free.prescribedTranslation=free.prescribedRotation=0;
    Output<double> o{},j{};require(evaluate(free,Direction<double>{},o,j),"free coordinate rejected");
    require(dot(o.forceA,o.forceA)<1e-15&&dot(o.connectorMoment,o.connectorMoment)<1e-15,"cylindrical free motion constrained");
    // Same connector represented in metres and millimetres. This exercises
    // both the residual and its derivative, including moment arms.
    for(unsigned i=0;i<96;++i) {
        auto a=inputs[i];auto d=directions[i];Output<double> base{},dbase{},scaled{},dscaled{};
        require(evaluate(a,d,base,dbase),"unit base rejected");
        const double s=1000;
        a.referenceA=a.referenceA*s;a.referenceB=a.referenceB*s;
        a.positionA=a.positionA*s;a.positionB=a.positionB*s;a.origin=a.origin*s;
        a.translation*=s;a.forcePenalty/=s;a.momentPenalty*=s;a.axialMoment*=s;a.momentMultiplier=a.momentMultiplier*s;
        d.positionA=d.positionA*s;d.positionB=d.positionB*s;
        require(evaluate(a,d,scaled,dscaled),"unit scaled joint rejected");
        for(unsigned k=0;k<2;++k) {
            auto x=values(k?dbase:base),y=values(k?dscaled:scaled);
            for(unsigned n=0;n<21;++n) {
                const bool length=(n>=3&&n<6)||(n>=9&&n<15)||n>=18;
                double expected=x[n]*(length?s:1);
                require(std::abs(y[n]-expected)<1e-8*std::max(1.0,std::abs(expected)),"length units change joint mechanics");
            }
        }
        inputs.push_back(a);directions.push_back(d);
    }
    auto bad=free;bad.axis={};require(!evaluate(bad,Direction<double>{},o,j),"zero axis accepted");
    inputs.push_back(bad);directions.push_back({});
    bad=free;bad.prescribedRotation=2;require(!evaluate(bad,Direction<double>{},o,j),"invalid flag accepted");
    inputs.push_back(bad);directions.push_back({});
    bad=free;bad.rotationA={0,0,0,0};require(!evaluate(bad,Direction<double>{},o,j),"zero quaternion accepted");
    inputs.push_back(bad);directions.push_back({});
    if(rows) {
        std::ifstream in(argv[2]);require(bool(in),"cannot read source rows");size_t count=0;in>>count;
        require(bool(in)&&count>0&&count<100000,"invalid source row count");
        std::ofstream out(argv[3]);require(bool(out),"cannot create source results");out<<std::setprecision(17);
        for(size_t i=0;i<count;++i) {
            auto a=readInput(in);require(bool(in)&&evaluate(a,Direction<double>{},o,j),"invalid source input row");
            for(auto v:values(o))out<<v<<' ';out<<'\n';inputs.push_back(a);directions.push_back({});
        }
        in>>std::ws;require(in.eof(),"extra source rows");require(bool(out),"source write failed");
    }
    std::cout<<"cpu_directional_checks=2016 max_relative_error="<<worst
             <<" virtual_work_checks=96 max_relative_error="<<worstVirtualWork<<"\n";
    if(!cpu)gpu(inputs,directions);
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} } }
