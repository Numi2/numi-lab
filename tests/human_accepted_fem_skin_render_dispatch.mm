#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include "NumiHumanAcceptedFEMSkinRenderPass.hpp"

using namespace numi::human;

struct EncoderContext {
    id<MTLComputeCommandEncoder> encoder;
};
static void useHeap(void*, void*) {}
static void setPipeline(void* c, void* p) {
    [static_cast<EncoderContext*>(c)->encoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)p];
}
static void setBuffer(void* c, void* b, std::size_t off, std::uint32_t i) {
    [static_cast<EncoderContext*>(c)->encoder setBuffer:(__bridge id<MTLBuffer>)b offset:off atIndex:i];
}
static void setBytes(void* c, const void* b, std::size_t n, std::uint32_t i) {
    [static_cast<EncoderContext*>(c)->encoder setBytes:b length:n atIndex:i];
}
static void dispatchThreads(void* c, std::size_t n, std::size_t width) {
    [static_cast<EncoderContext*>(c)->encoder dispatchThreads:MTLSizeMake(n,1,1)
        threadsPerThreadgroup:MTLSizeMake(width,1,1)];
}
static void dispatchThreadgroups(void* c, std::size_t n, std::size_t width) {
    [static_cast<EncoderContext*>(c)->encoder dispatchThreadgroups:MTLSizeMake(n,1,1)
        threadsPerThreadgroup:MTLSizeMake(width,1,1)];
}

static void require(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}
static bool near(float a, float b, float eps=2.0e-5f) {
    return std::isfinite(a) && std::fabs(a-b)<=eps;
}
static float norm3(const mr_float4& a) {
    return std::sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
}
static void set4(mr_float4& v, float x,float y,float z,float w) { v={x,y,z,w}; }

int main() {
    static_assert(sizeof(MRVisualVertexGPUV2)==80);
    static_assert(offsetof(MRVisualVertexGPUV2,position)==0);
    static_assert(offsetof(MRVisualVertexGPUV2,normalAndTangentSign)==16);
    static_assert(offsetof(MRVisualVertexGPUV2,tangent)==32);
    static_assert(offsetof(MRVisualVertexGPUV2,texcoord01)==48);
    static_assert(offsetof(MRVisualVertexGPUV2,color)==64);

    @autoreleasepool {
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        require(device!=nil,"Metal device unavailable");
        id<MTLCommandQueue> queue=[device newCommandQueue];
        require(queue!=nil,"Metal command queue allocation failed");

        using Row=AcceptedFEMSkinTransfer::GPUTransferRow;
        std::vector<Row> rows(3);
        rows[0].coarseVertex[0]=0; rows[0].coarseVertex[1]=1; rows[0].coarseVertex[2]=2;
        rows[0].weight[0]=1; rows[0].weight[1]=0; rows[0].weight[2]=0;
        rows[1].coarseVertex[0]=0; rows[1].coarseVertex[1]=1; rows[1].coarseVertex[2]=2;
        rows[1].weight[0]=0; rows[1].weight[1]=1; rows[1].weight[2]=0;
        rows[2].coarseVertex[0]=0; rows[2].coarseVertex[1]=1; rows[2].coarseVertex[2]=2;
        rows[2].weight[0]=0; rows[2].weight[1]=0; rows[2].weight[2]=1;

        AcceptedFEMSkinRenderPass pass;
        std::string error;
        require(pass.initialize(device,rows,3,&error),error.c_str());

        std::array<MRVisualVertexGPUV2,3> original{};
        set4(original[0].position,0.25f,0.0f,0.0f,1.0f);
        set4(original[1].position,1.25f,0.0f,0.0f,1.0f);
        set4(original[2].position,0.25f,1.0f,0.0f,1.0f);
        for(auto& v:original) {
            set4(v.normalAndTangentSign,0,0,1,1);
            set4(v.tangent,1,0,0,0);
            set4(v.texcoord01,0.1f,0.2f,0.3f,0.4f);
            set4(v.color,0.7f,0.6f,0.5f,1.0f);
        }
        id<MTLBuffer> mesh=[device newBufferWithBytes:original.data()
            length:sizeof(original) options:MTLResourceStorageModeShared];
        require(mesh!=nil,"renderer vertex buffer allocation failed");

        // The current mechanical baseline includes a nonzero remeshing offset.
        // With zero accepted FEM displacement, the presentation vertices must
        // remain unchanged; baseline only determines the normal/tangent frame.
        std::vector<AcceptedFEMSkinTransfer::Point> baseline={
            {0.25,0.0,0.0},{1.25,0.0,0.0},{0.25,1.0,0.0}};
        std::vector<std::array<float,4>> delta(3,{0,0,0,0});
        require(pass.updateAcceptedState(delta,baseline,&error),error.c_str());

        auto run=[&] {
            id<MTLCommandBuffer> cb=[queue commandBuffer];
            require(cb!=nil,"command buffer allocation failed");
            id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
            require(enc!=nil,"compute encoder allocation failed");
            EncoderContext context{enc};
            metalrobo::MetalHybridComputeEncoderCallbacks callbacks{};
            callbacks.context=&context;
            callbacks.useHeap=useHeap;
            callbacks.setPipeline=setPipeline;
            callbacks.setBuffer=setBuffer;
            callbacks.setBytes=setBytes;
            callbacks.dispatchThreads=dispatchThreads;
            callbacks.dispatchThreadgroups=dispatchThreadgroups;
            require(pass.encode(callbacks,(__bridge void*)mesh,0),
                    "presentation pass refused actual renderer buffer");
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            require(cb.status==MTLCommandBufferStatusCompleted,
                    "Metal dispatch did not complete successfully");
        };
        run();
        auto* actual=static_cast<const MRVisualVertexGPUV2*>([mesh contents]);
        require(actual!=nullptr,"shared renderer vertex buffer is not CPU visible");
        for(std::size_t i=0;i<original.size();++i) {
            require(std::memcmp(&actual[i].position,&original[i].position,sizeof(mr_float4))==0,
                    "zero displacement changed a remeshed skin position");
            require(near(norm3(actual[i].normalAndTangentSign),1.0f),
                    "zero-displacement normal is not unit length");
            require(near(norm3(actual[i].tangent),1.0f),
                    "zero-displacement tangent is not unit length");
            require(std::memcmp(&actual[i].texcoord01,&original[i].texcoord01,sizeof(mr_float4))==0,
                    "pass changed renderer UVs");
            require(std::memcmp(&actual[i].color,&original[i].color,sizeof(mr_float4))==0,
                    "pass changed renderer color");
        }

        std::memcpy([mesh contents],original.data(),sizeof(original));
        delta[1]={0,0,0.2f,0};
        delta[2]={0,0,0.4f,0};
        require(pass.updateAcceptedState(delta,baseline,&error),error.c_str());
        run();
        actual=static_cast<const MRVisualVertexGPUV2*>([mesh contents]);
        require(near(actual[0].position.x,0.25f)&&near(actual[0].position.y,0)&&near(actual[0].position.z,0),
                "one-hot row 0 position mismatch");
        require(near(actual[1].position.x,1.25f)&&near(actual[1].position.y,0)&&near(actual[1].position.z,0.2f),
                "one-hot row 1 position mismatch");
        require(near(actual[2].position.x,0.25f)&&near(actual[2].position.y,1)&&near(actual[2].position.z,0.4f),
                "one-hot row 2 position mismatch");
        for(std::size_t i=0;i<3;++i) {
            const auto& v=actual[i];
            require(near(norm3(v.normalAndTangentSign),1.0f),"deformed normal is not unit length");
            require(near(norm3(v.tangent),1.0f),"deformed tangent is not unit length");
            require(std::fabs(v.normalAndTangentSign.x*v.tangent.x+
                              v.normalAndTangentSign.y*v.tangent.y+
                              v.normalAndTangentSign.z*v.tangent.z)<5e-5f,
                    "deformed tangent is not orthogonal to normal");
        }
        std::printf("PASS Metal dispatch device=%s rows=%u coarse=%u vertexABI=%zu offset=%u zeroDelta=exact displacement=interpolated normals=unit\n",
                    [[device name] UTF8String],pass.rowCount(),pass.coarseCount(),sizeof(MRVisualVertexGPUV2),
                    static_cast<unsigned>(offsetof(MRVisualVertexGPUV2,normalAndTangentSign)));
    }
    return 0;
}
