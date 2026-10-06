#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/numi_human_resting_common_field_math.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(bool condition,const std::string& message) { if(!condition) throw std::runtime_error(message); }
using E=numi_human_resting_common_field::Exponent;
void put(MRHumanRestingCommonFieldGPU& p,unsigned row,const E& e,float value) {
    const unsigned term=numi_human_resting_common_field::termIndex(e);
    check(term<120,"fixture polynomial exponent missing");
    mr_float4& packed=p.volumePolynomial[row][term/4];
    reinterpret_cast<float*>(&packed)[term%4]=value;
}
double evaluate(const MRHumanRestingCommonFieldGPU& p,unsigned row,const std::array<float,7>& x) {
    double result=0;
    for(unsigned term=0;term<120;++term) {
        double monomial=1;
        for(unsigned c=0;c<7;++c)
            for(unsigned power=0;power<numi_human_resting_common_field::kMonomialExponents[term][c];++power)
                monomial*=x[c];

        const float* packed=reinterpret_cast<const float*>(&p.volumePolynomial[row][term/4]);
        result+=double(packed[term%4])*monomial;
    }
    return result;
}
constexpr const char* structs=R"MSL(
struct MRHumanRestingCommonFieldGPU {
    float4 volumePolynomial[7][30];
    float4 sourceReferenceVolumes[2];
    float4 materialTargetVolumes;
    float4 trialLower[2];
    float4 trialUpper[2];
    float4 solver;
    uint4 countsAndFlags;
};
struct MRHumanRestingCommonCoordinateBoxGPU { float4 lower[2]; float4 upper[2]; };
struct MRHumanRestingCommonCoordinatesGPU { float4 first; float4 second; uint4 status; float4 diagnostics; };
)MSL";
constexpr const char* fixtureKernel=R"MSL(
kernel void common_field_solver_fixture(
    constant MRHumanRestingCommonFieldGPU& p [[buffer(0)]],
    device const MRHumanRestingCommonCoordinateBoxGPU* boxes [[buffer(1)]],
    device const float* targetInput [[buffer(2)]],
    device MRHumanRestingCommonCoordinatesGPU* output [[buffer(3)]],
    constant uint& mode [[buffer(4)]],constant uint& targetCount [[buffer(5)]],uint lane [[thread_position_in_grid]]) {
    if(lane>=targetCount)return;
    float target[7];for(uint i=0;i<7;++i)target[i]=targetInput[lane*7+i];
    if(mode==1u)target[5]=NAN;
    float x[7]={0,0,0,0,0,0,0},residual=INFINITY;
    uint iterations=0,box=0xffffffffu;
    const uint status=nmHumanRestingCommonSolve(p,boxes,target,x,iterations,box,residual);
    output[lane].first=status?float4(NAN):float4(x[0],x[1],x[2],x[3]);
    output[lane].second=status?float4(NAN):float4(x[4],x[5],x[6],0);
    output[lane].status=uint4(status,iterations,box,0);
    output[lane].diagnostics=float4(residual,0,0,0);
}
)MSL";
void run(id<MTLComputePipelineState> pipeline,id<MTLCommandQueue> queue,
    id<MTLBuffer> parameters,id<MTLBuffer> boxes,id<MTLBuffer> targets,id<MTLBuffer> output,
    id<MTLBuffer> mode,id<MTLBuffer> targetCount) {
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    check(command&&encoder,"Metal command creation failed");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:parameters offset:0 atIndex:0];
    [encoder setBuffer:boxes offset:0 atIndex:1];
    [encoder setBuffer:targets offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBuffer:mode offset:0 atIndex:4];
    [encoder setBuffer:targetCount offset:0 atIndex:5];
    const uint count=*static_cast<const uint*>(targetCount.contents);
    const NSUInteger width=std::min<NSUInteger>(64,pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    check(command.status==MTLCommandBufferStatusCompleted,"Metal common-field solver command failed");
}
std::array<float,7> readCoordinates(const MRHumanRestingCommonCoordinatesGPU& result) {
    return {result.first.x,result.first.y,result.first.z,result.first.w,
        result.second.x,result.second.y,result.second.z};
}
}

std::vector<std::uint8_t> readFile(const std::string& path,std::size_t expected) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    check(bool(in),std::string("phase fixture file missing: ")+path);
    const auto length=in.tellg();check(length>=0&&std::size_t(length)==expected,std::string("phase fixture file has wrong byte length: ")+path);
    std::vector<std::uint8_t> bytes(expected);in.seekg(0);in.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(bytes.size()));
    check(bool(in),std::string("phase fixture file could not be read: ")+path);return bytes;
}
void runPhaseFixture(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLComputePipelineState> pipeline,const std::string& dir) {
    constexpr unsigned boxCount=125,rowCount=1987;
    const auto polyBytes=readFile(dir+"/polynomials-f32-normalized.bin",7u*120u*sizeof(float));
    const auto referenceBytes=readFile(dir+"/source-reference-volumes-f32.bin",7u*sizeof(float));
    const auto targetBytes=readFile(dir+"/targets-m3-f32.bin",rowCount*7u*sizeof(float));
    const auto boxBytes=readFile(dir+"/certified-boxes-f32.bin",boxCount*sizeof(MRHumanRestingCommonCoordinateBoxGPU));
    MRHumanRestingCommonFieldGPU p{};
    std::memcpy(p.volumePolynomial,polyBytes.data(),polyBytes.size());
    float refs[7];std::memcpy(refs,referenceBytes.data(),sizeof(refs));
    p.sourceReferenceVolumes[0]={refs[0],refs[1],refs[2],refs[3]};p.sourceReferenceVolumes[1]={refs[4],refs[5],refs[6],0};
    p.countsAndFlags={boxCount,1,0,0};p.solver={2.0e-5f,0,0,0};
    std::vector<MRHumanRestingCommonCoordinateBoxGPU> boxes(boxCount);std::memcpy(boxes.data(),boxBytes.data(),boxBytes.size());
    float lo[7],hi[7];for(unsigned c=0;c<7;++c){lo[c]=INFINITY;hi[c]=-INFINITY;}
    for(const auto& box:boxes)for(unsigned c=0;c<7;++c){
        const float a=c<4?reinterpret_cast<const float*>(&box.lower[0])[c]:reinterpret_cast<const float*>(&box.lower[1])[c-4];
        const float b=c<4?reinterpret_cast<const float*>(&box.upper[0])[c]:reinterpret_cast<const float*>(&box.upper[1])[c-4];
        lo[c]=std::min(lo[c],a);hi[c]=std::max(hi[c],b);
    }
    p.trialLower[0]={lo[0],lo[1],lo[2],lo[3]};p.trialLower[1]={lo[4],lo[5],lo[6],0};
    p.trialUpper[0]={hi[0],hi[1],hi[2],hi[3]};p.trialUpper[1]={hi[4],hi[5],hi[6],0};
    std::vector<float> targets(rowCount*7u);std::memcpy(targets.data(),targetBytes.data(),targetBytes.size());
    const uint count=rowCount,mode=0;std::vector<id<MTLBuffer>> buffers(6);
    buffers[0]=[device newBufferWithBytes:&p length:sizeof(p) options:MTLResourceStorageModeShared];
    buffers[1]=[device newBufferWithBytes:boxes.data() length:boxes.size()*sizeof(boxes[0]) options:MTLResourceStorageModeShared];
    buffers[2]=[device newBufferWithBytes:targets.data() length:targets.size()*sizeof(float) options:MTLResourceStorageModeShared];
    buffers[3]=[device newBufferWithLength:rowCount*sizeof(MRHumanRestingCommonCoordinatesGPU) options:MTLResourceStorageModeShared];
    buffers[4]=[device newBufferWithBytes:&mode length:sizeof(mode) options:MTLResourceStorageModeShared];
    buffers[5]=[device newBufferWithBytes:&count length:sizeof(count) options:MTLResourceStorageModeShared];
    for(auto buffer:buffers)check(buffer!=nil,"phase fixture Metal buffer allocation failed");
    run(pipeline,queue,buffers[0],buffers[1],buffers[2],buffers[3],buffers[4],buffers[5]);
    const auto* results=static_cast<const MRHumanRestingCommonCoordinatesGPU*>(buffers[3].contents);
    unsigned maxIterations=0;float maxResidual=0.0f;
    for(unsigned row=0;row<rowCount;++row){
        check(results[row].status.x==0,"retained phase target failed GPU solve at row "+std::to_string(row)+" status "+std::to_string(results[row].status.x));
        check(results[row].status.z<boxCount,"retained phase root did not match certified union at row "+std::to_string(row));
        check(std::isfinite(results[row].diagnostics.x)&&results[row].diagnostics.x<=p.solver.x,"retained phase residual exceeds tolerance at row "+std::to_string(row));
        maxIterations=std::max(maxIterations,results[row].status.y);maxResidual=std::max(maxResidual,results[row].diagnostics.x);
    }
    std::cout<<"common_field_phase_fixture=PASS rows="<<rowCount<<" boxes="<<boxCount<<" max_iterations="<<maxIterations<<" max_normalized_residual="<<maxResidual<<"\n";
}
int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            check(device!=nil,"Metal device unavailable");
            id<MTLCommandQueue> queue=[device newCommandQueue];
            check(queue!=nil,"Metal command queue unavailable");
            NSError* error=nil;
            NSString* incPath=@(NUMI_COMMON_FIELD_MSLINC_PATH);
            NSString* include=[NSString stringWithContentsOfFile:incPath encoding:NSUTF8StringEncoding error:&error];
            check(include!=nil,"common-field shared Metal include could not be read");
            std::string source="#include <metal_stdlib>\nusing namespace metal;\n";
            source+=structs;source+="\n";source+=[include UTF8String];source+="\n";source+=fixtureKernel;
            NSString* shader=[NSString stringWithUTF8String:source.c_str()];
            id<MTLLibrary> library=[device newLibraryWithSource:shader options:nil error:&error];
            if(!library) throw std::runtime_error(error.localizedDescription.UTF8String);
            id<MTLFunction> function=[library newFunctionWithName:@"common_field_solver_fixture"];
            check(function!=nil,"shared common-field fixture kernel is absent");
            id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
            if(!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);

            MRHumanRestingCommonFieldGPU p{};
            p.countsAndFlags={1,1,0,0};p.solver={1.0e-6f,0,0,0};
            p.sourceReferenceVolumes[0]={1,1,1,1};p.sourceReferenceVolumes[1]={1,1,1,0};
            p.trialLower[0]={-.5f,-.5f,-.5f,-.5f};p.trialLower[1]={-.5f,-.5f,-.5f,0};
            p.trialUpper[0]={.5f,.5f,.5f,.5f};p.trialUpper[1]={.5f,.5f,.5f,0};
            std::array<float,7> expected{-.12f,-.06f,.04f,.09f,.03f,-.02f,.01f};
            for(unsigned row=0;row<7;++row) {
                put(p,row,E{0,0,0,0,0,0,0},1.0f);
                for(unsigned column=0;column<7;++column) {
                    const float a=row==column?1.25f:.0125f*float(1+((row+2*column)%4));
                    E linear{};linear[column]=1;put(p,row,linear,a);
                }
                E square{};square[row]=2;put(p,row,square,.12f);
                E cross{};cross[row]=1;cross[(row+1)%7]=1;put(p,row,cross,.08f);
                E cubic{};cubic[row]=1;cubic[(row+1)%7]=1;cubic[(row+2)%7]=1;put(p,row,cubic,.035f);
                reinterpret_cast<float*>(&p.sourceReferenceVolumes[0])[row]=float(2.0e-5+row*1.0e-6);
            }
            for(unsigned row=0;row<3;++row)reinterpret_cast<float*>(&p.sourceReferenceVolumes[1])[row]=float(3.0e-5+row*1.0e-6);
            p.materialTargetVolumes={4.0e-5f,4.1e-5f,4.2e-5f,0};
            std::array<float,7> target{};
            for(unsigned row=0;row<7;++row) {
                const double normalized=evaluate(p,row,expected);
                target[row]=float(normalized*reinterpret_cast<const float*>(&p.sourceReferenceVolumes[row<4?0:1])[row<4?row:row-4]);
            }
            MRHumanRestingCommonCoordinateBoxGPU box{};
            box.lower[0]={-.3f,-.3f,-.3f,-.3f};box.lower[1]={-.3f,-.3f,-.3f,0};
            box.upper[0]={.3f,.3f,.3f,.3f};box.upper[1]={.3f,.3f,.3f,0};
            std::array<id<MTLBuffer>,6> buffers{
                [device newBufferWithBytes:&p length:sizeof(p) options:MTLResourceStorageModeShared],
                [device newBufferWithBytes:&box length:sizeof(box) options:MTLResourceStorageModeShared],
                [device newBufferWithBytes:target.data() length:sizeof(target) options:MTLResourceStorageModeShared],
                [device newBufferWithLength:sizeof(MRHumanRestingCommonCoordinatesGPU) options:MTLResourceStorageModeShared],
                [device newBufferWithLength:sizeof(std::uint32_t) options:MTLResourceStorageModeShared],
                [device newBufferWithLength:sizeof(std::uint32_t) options:MTLResourceStorageModeShared]};
            for(auto buffer:buffers)check(buffer!=nil,"common-field test buffer allocation failed");
            std::memset(buffers[4].contents,0,buffers[4].length);
            *static_cast<std::uint32_t*>(buffers[5].contents)=1u;
            run(pipeline,queue,buffers[0],buffers[1],buffers[2],buffers[3],buffers[4],buffers[5]);
            auto result=*static_cast<MRHumanRestingCommonCoordinatesGPU*>(buffers[3].contents);
            check(result.status.x==0,"coupled common-field Newton solve did not converge");
            const auto solved=readCoordinates(result);
            for(unsigned i=0;i<7;++i)
                check(std::abs(solved[i]-expected[i])<3.0e-5f,"Metal coupled-coordinate solution differs from known seed");
            check(result.diagnostics.x<=p.solver.x,"Metal common-field residual exceeds configured tolerance");

            box.lower[0]={-.01f,-.01f,-.01f,-.01f};box.lower[1]={-.01f,-.01f,-.01f,0};
            box.upper[0]={.01f,.01f,.01f,.01f};box.upper[1]={.01f,.01f,.01f,0};
            std::memcpy(buffers[1].contents,&box,sizeof(box));
            run(pipeline,queue,buffers[0],buffers[1],buffers[2],buffers[3],buffers[4],buffers[5]);
            result=*static_cast<MRHumanRestingCommonCoordinatesGPU*>(buffers[3].contents);
            check(result.status.x==5,"solution outside certified-box union was not rejected");

            box.lower[0]={-.3f,-.3f,-.3f,-.3f};box.lower[1]={-.3f,-.3f,-.3f,0};
            box.upper[0]={.3f,.3f,.3f,.3f};box.upper[1]={.3f,.3f,.3f,0};
            std::memcpy(buffers[1].contents,&box,sizeof(box));
            *static_cast<std::uint32_t*>(buffers[4].contents)=1u;
            run(pipeline,queue,buffers[0],buffers[1],buffers[2],buffers[3],buffers[4],buffers[5]);
            result=*static_cast<MRHumanRestingCommonCoordinatesGPU*>(buffers[3].contents);
            check(result.status.x==1&&!std::isfinite(result.first.x),
                "nonfinite target was not rejected with an invalid coordinate result");
            if(argc==3&&std::string(argv[1])=="--phase-fixture")runPhaseFixture(device,queue,pipeline,argv[2]);
            std::cout<<"common_field_metal=PASS coupled_cross_terms=7x120 certified_union_guard=PASS nonfinite_rejection=PASS\n";
        } catch(const std::exception& e) {
            std::cerr<<"common_field_metal=FAIL "<<e.what()<<"\n";return 1;
        }
    }
    return 0;
}
