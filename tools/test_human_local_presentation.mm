#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <metalrobo/numi_human_resting_visual_gpu.h>
#include <metalrobo/numi_human_stand_gpu.h>
#include <numi/matter/human_respiration.h>
#include "../apps/NumiHumanRestingSurfaceAuditDiagnostic.hpp"
#include "../apps/NumiHumanRestingSkinInfluencePipeline.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

static void require(bool ok,const char* message) {
    if(!ok)throw std::runtime_error(message);
}
static std::string readText(const char* path) {
    std::ifstream file(path,std::ios::binary);
    require(file.good(),"pinned flattened Metal source is unavailable");
    return {std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
}
static id<MTLBuffer> sharedBuffer(id<MTLDevice> device,const void* bytes,std::size_t size) {
    require(size>0,"zero-sized Metal test buffer");
    id<MTLBuffer> buffer=[device newBufferWithLength:size options:MTLResourceStorageModeShared];
    require(buffer!=nil,"shared Metal test buffer allocation failed");
    std::memcpy(buffer.contents,bytes,size);
    return buffer;
}

int main(int argc,char** argv) { @autoreleasepool { try {
    require(argc==2,"usage: test_human_local_presentation PINNED_FLATTENED_HUMAN_PRESENTATION_METAL");
    static_assert(sizeof(MRVisualVertexGPUV2)==80);
    static_assert(sizeof(MRHumanRestingVertexMap)==48);
    static_assert(sizeof(MRHumanRestingInfluence)==48);
    static_assert(sizeof(MRHumanRestingCommonFieldVertexGPU)==112);
    static_assert(sizeof(MRHumanRestingCommonCoordinatesGPU)==64);
    static_assert(sizeof(MRBodyStateGPU)==128);
    static_assert(NM_MATTER_ABI_VERSION==41u);
    static_assert(alignof(NMHumanRespirationState)>=16&&sizeof(NMHumanRespirationState)%16==0);

    const auto source=readText(argv[1]);
    const bool identityV2=source.find("kernel void nm_human_resting_local_cardiac_layout_v2")!=std::string::npos;
    const bool identityV4=source.find("kernel void nm_human_resting_local_cardiac_layout_v4")!=std::string::npos;
    const bool identityV3=source.find("kernel void nm_human_resting_local_cardiac_layout_v3")!=std::string::npos;
    require((identityV2||identityV3||identityV4)&&
            source.find("kernel void nm_human_resting_local_cardiac_layout_v1")==std::string::npos,
            "pinned shader identity predates the required 017 v2 layout guard");
    require(source.find("m.deformationKind==12u")!=std::string::npos&&
            source.find("m.deformationWeight.w==2.0f")!=std::string::npos&&
            source.find("p=local;n=normal;")!=std::string::npos,
            "pinned shader lacks the torso-local cardiac vertex path");
    const char* identityName=identityV4?"nm_human_resting_local_cardiac_layout_v4":identityV3?"nm_human_resting_local_cardiac_layout_v3":
        "nm_human_resting_local_cardiac_layout_v2";

    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    require(device!=nil,"Metal device unavailable");
    NSError* error=nil;
    MTLCompileOptions* options=[MTLCompileOptions new];
    options.mathMode=MTLMathModeSafe;
    options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
    options.languageVersion=MTLLanguageVersion4_0;
    NSString* metalSource=[NSString stringWithUTF8String:source.c_str()];
    id<MTLLibrary> library=[device newLibraryWithSource:metalSource options:options error:&error];
    if(library==nil) {
        const char* detail=error.localizedDescription.UTF8String;
        throw std::runtime_error(detail?detail:"Metal shader compilation failed");
    }
    id<MTLFunction> identity=[library newFunctionWithName:[NSString stringWithUTF8String:identityName]];
    require(identity!=nil,"compiled shader lacks its 017 v2-or-newer layout identity");
    id<MTLFunction> function=numiHumanRestingMakeSkinInfluenceFunction(library,@"nm_human_resting_skin",{},&error);
    require(function!=nil,"compiled pinned source lacks nm_human_resting_skin");
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
    if(pipeline==nil)throw std::runtime_error(error.localizedDescription.UTF8String);

    constexpr float edge=1.0e-7f;
    const std::array<mr_float4,3> localPoints={mr_float4{0,0,0,1},mr_float4{edge,0,0,1},mr_float4{0,edge,0,1}};
    MRHumanRestingVertexMap maps[3]{};
    MRHumanRestingInfluence influences[3]{};
    MRVisualVertexGPUV2 vertices[3]{};
    MRBodyStateGPU body{};
    body.orientation={0,0,0,1};
    MRHumanRestingAnatomyGPU anatomy{};
    anatomy.bodyAndFlags={0,1,0,0};
    NMHumanRespirationState respiration{};
    float cardiacQ[4]{};
    MRHumanRestingCardiacWallVertexGPU wallMap[3]{};
    MRHumanRestingCardiacWallGPU wall{};
    mr_float4 wallQ{};
    MRHumanRestingCommonFieldVertexGPU commonMap[3]{};
    MRHumanRestingCommonCoordinatesGPU commonCoordinates{};
    for(unsigned i=0;i<3;++i) {
        maps[i].firstInfluence=i;
        maps[i].influenceCount=1;
        maps[i].deformationKind=12;
        maps[i].chamberIndex=i;
        influences[i].positionAndWeight=localPoints[i];
        influences[i].normal={0,0,1,0};
        influences[i].body={0,0,0,0};
    }
    const mr_uint4 dispatch={3,0,0,0};
    auto dispatchBuffer=sharedBuffer(device,&dispatch,sizeof(dispatch));
    auto mapBuffer=sharedBuffer(device,maps,sizeof(maps));
    auto influenceBuffer=sharedBuffer(device,influences,sizeof(influences));
    auto bodyBuffer=sharedBuffer(device,&body,sizeof(body));
    auto vertexBuffer=sharedBuffer(device,vertices,sizeof(vertices));
    auto respirationBuffer=sharedBuffer(device,&respiration,sizeof(respiration));
    auto anatomyBuffer=sharedBuffer(device,&anatomy,sizeof(anatomy));
    auto cardiacQBuffer=sharedBuffer(device,cardiacQ,sizeof(cardiacQ));
    auto wallMapBuffer=sharedBuffer(device,wallMap,sizeof(wallMap));
    auto wallBuffer=sharedBuffer(device,&wall,sizeof(wall));
    auto wallQBuffer=sharedBuffer(device,&wallQ,sizeof(wallQ));
    auto commonMapBuffer=sharedBuffer(device,commonMap,sizeof(commonMap));
    auto commonCoordinatesBuffer=sharedBuffer(device,&commonCoordinates,sizeof(commonCoordinates));
    id<MTLCommandQueue> queue=[device newCommandQueue];
    require(queue!=nil,"Metal command queue unavailable");
    auto runSkin=[&] {
        id<MTLCommandBuffer> command=[queue commandBuffer];
        require(command!=nil,"Metal command buffer allocation failed");
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        require(encoder!=nil,"Metal compute encoder allocation failed");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:dispatchBuffer offset:0 atIndex:0];
        [encoder setBuffer:mapBuffer offset:0 atIndex:1];
        [encoder setBuffer:influenceBuffer offset:0 atIndex:2];
        [encoder setBuffer:bodyBuffer offset:0 atIndex:3];
        [encoder setBuffer:vertexBuffer offset:0 atIndex:4];
        [encoder setBuffer:respirationBuffer offset:0 atIndex:5];
        [encoder setBuffer:anatomyBuffer offset:0 atIndex:6];
        [encoder setBuffer:cardiacQBuffer offset:0 atIndex:7];
        [encoder setBuffer:wallMapBuffer offset:0 atIndex:8];
        [encoder setBuffer:wallBuffer offset:0 atIndex:9];
        [encoder setBuffer:wallQBuffer offset:0 atIndex:10];
        [encoder setBuffer:commonMapBuffer offset:0 atIndex:11];
        [encoder setBuffer:commonCoordinatesBuffer offset:0 atIndex:12];
        [encoder dispatchThreads:MTLSizeMake(3,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if(command.status!=MTLCommandBufferStatusCompleted) {
            const char* detail=command.error.localizedDescription.UTF8String;
            throw std::runtime_error(detail?detail:"Metal skin dispatch failed");
        }
    };
    auto setCase=[&](float translation,float layoutWeight) {
        body.position={translation,0,0,1};
        std::memcpy(bodyBuffer.contents,&body,sizeof(body));
        for(unsigned i=0;i<3;++i)maps[i].deformationWeight.w=layoutWeight;
        std::memcpy(mapBuffer.contents,maps,sizeof(maps));
        runSkin();
    };
    auto output=[&] {return static_cast<const MRVisualVertexGPUV2*>(vertexBuffer.contents);};
    auto audit=[&] {
        const auto* out=output();
        const auto failure=numiHumanRestingSurfaceAudit::classifyRenderedTriangleAreaFailure(
            out[0].position,out[1].position,out[2].position);
        return failure;
    };
    auto requireLocal=[&](const char* label) {
        const auto* out=output();
        for(unsigned i=0;i<3;++i) {
            require(out[i].position.x==localPoints[i].x&&out[i].position.y==localPoints[i].y&&
                    out[i].position.z==localPoints[i].z&&out[i].position.w==1.0f,
                    label);
        }
        require(audit()==MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONE,
                "strict exact binary32 audit rejected the torso-local source triangle");
    };

    setCase(2.0f,2.0f);
    requireLocal("2 m body pose changed local cardiac source coordinates");
    MRVisualVertexGPUV2 localAt2[3];std::memcpy(localAt2,output(),sizeof(localAt2));
    setCase(20.0f,2.0f);
    requireLocal("20 m body pose changed local cardiac source coordinates");
    require(std::memcmp(localAt2,output(),sizeof(localAt2))==0,
            "torso-local cardiac vertex output changed with body translation");

    setCase(2.0f,0.0f);
    require(output()[0].position.x==output()[1].position.x&&
            audit() == MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA,
            "unchanged exact-area guard did not reject the collapsed 2 m legacy world triangle");
    setCase(20.0f,0.0f);
    require(output()[0].position.x==output()[1].position.x&&
            audit() == MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA,
            "unchanged exact-area guard did not reject the collapsed 20 m legacy world triangle");

    for(unsigned i=0;i<3;++i){maps[i].deformationKind=0u;maps[i].chamberIndex=0u;}
    setCase(2.0f,3.0f);requireLocal("2 m rigid owner frame lost source precision");
    setCase(20.0f,3.0f);requireLocal("20 m rigid owner frame lost source precision");

    std::cout<<"PASS pinned nm_human_resting_skin torso-local cardiac precision; exact area guard retained on "
             <<device.name.UTF8String<<" (source-only regression, not a physiology claim)\n";
    return 0;
} catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;} } }
