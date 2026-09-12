#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/compensated_translation_gpu.h"
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct Case { MRCompensatedRootTranslationGPU initial; mr_float4 velocity; std::uint32_t count; };
std::vector<Case> cases() {
    std::vector<Case> c;
    c.push_back({mrCompensatedTranslationFromProjection({-0.0f,0.0f,-0.0f,0.0f}), {0,0,0,0}, 0});
    c.push_back({{{1,0,0,0},{0x1p-24f,0,0,0},{0x1p-50f,0,0,0}}, {0,0,0,0}, 0});
    c.push_back({{{1,0,0,0},{0x1p-24f,0,0,0},{-0x1p-50f,0,0,0}}, {0,0,0,0}, 0});
    c.push_back({{{-1,0,0,0},{-0x1p-24f,0,0,0},{-0x1p-50f,0,0,0}}, {0,0,0,0}, 0});
    c.push_back({{{100000000,0,0,0},{-100000000,0,0,0},{1,0,0,0}}, {0,0,0,0}, 0});
    for (unsigned i=0; i<64; ++i) {
        const float k = static_cast<float>(i+1);
        auto v = mrCompensatedTranslationFromProjection({-0.05f*k,0.1625f*k,1.94496f*k,0});
        c.push_back({v, {0.00003f*k,-0.000011f*k,-0.000451f*k,0.000025f}, 4096});
        c.push_back({v, {-0.00003f*k,0.000011f*k,0.000451f*k,0.000025f}, 4096});
    }
    return c;
}
MRCompensatedRootTranslationGPU evaluate(const Case& c) {
    auto result=c.initial;
    for (unsigned n=0;n<c.count;++n) result=mrCompensatedTranslationAdvance(result,c.velocity,c.velocity.w);
    return result;
}
float projected64(const Case& c,unsigned axis) {
    const auto* r=reinterpret_cast<const float*>(&c.initial.reference);
    const auto* d=reinterpret_cast<const float*>(&c.initial.displacement);
    const auto* e=reinterpret_cast<const float*>(&c.initial.correction);
    const auto* v=reinterpret_cast<const float*>(&c.velocity);
    if (d[axis] == 0.0f && e[axis] == 0.0f && c.count == 0u) return r[axis];
    return static_cast<float>((static_cast<double>(r[axis])+d[axis])+e[axis]+
        static_cast<double>(c.count)*static_cast<double>(v[axis])*static_cast<double>(c.velocity.w));
}
}
int main(int argc,char** argv) {
    try { @autoreleasepool {
        const bool cpu=argc==2 && std::strcmp(argv[1],"--cpu")==0;
        require(cpu || argc==2,"usage: compensated-translation-check --cpu | METALLIB");
        const auto c=cases(); std::vector<MRCompensatedRootTranslationGPU> expected;
        std::vector<mr_float4> projection; double maxError=0; unsigned checks=0;
        for (const auto& item:c) {
            require(mrCompensatedTranslationValid(item.initial),"invalid manufactured initial pair");
            const auto result=evaluate(item); require(mrCompensatedTranslationValid(result),"advance lost canonical pair");
            const auto p=mrCompensatedTranslationProjection(result);
            const auto pair=mrCompensatedTranslationPosition(result,{0,0,0,0});
            for (unsigned a=0;a<3;++a) {
                require(mrCompensatedBits(reinterpret_cast<const float*>(&p)[a])==mrCompensatedBits(projected64(item,a)),"projection disagrees with independent FP64 sum");
                require(mrCompensatedBits(reinterpret_cast<const float*>(&p)[a])==mrCompensatedBits(reinterpret_cast<const float*>(&pair.high)[a]),"paired geometry and q projection differ");
                const double physical=static_cast<double>(reinterpret_cast<const float*>(&result.displacement)[a])+reinterpret_cast<const float*>(&result.correction)[a];
                const double oracle=static_cast<double>(reinterpret_cast<const float*>(&item.initial.displacement)[a])+reinterpret_cast<const float*>(&item.initial.correction)[a]+
                    static_cast<double>(item.count)*reinterpret_cast<const float*>(&item.velocity)[a]*static_cast<double>(item.velocity.w);
                maxError=std::max(maxError,std::abs(physical-oracle)); checks+=2;
            }
            expected.push_back(result);projection.push_back(p);
        }
        require(maxError<1e-12,"relative-displacement accumulation error exceeded fixed numerical bound");
        auto bad=c[0].initial;bad.reference.w=-0.0f;require(!mrCompensatedTranslationValid(bad),"noncanonical reserved field admitted");++checks;
        bad=c[0].initial;bad.correction.x=std::numeric_limits<float>::quiet_NaN();require(!mrCompensatedTranslationValid(bad),"NaN pair admitted");++checks;
        if (cpu) {std::printf("{\"mode\":\"cpu\",\"passed\":true,\"checks\":%u,\"max_displacement_error_m\":%.17g}\n",checks,maxError);return 0;}
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"no Metal device");
        NSError* error=nil; id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];require(library!=nil,"cannot load production metallib");
        id<MTLFunction> fn=[library newFunctionWithName:@"mr_compensated_translation_check"];require(fn!=nil,"missing precision qualification kernel");
        id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:fn error:&error];require(pipeline!=nil,"cannot compile precision pipeline");
        id<MTLCommandQueue> queue=[device newCommandQueue];require(queue!=nil,"no command queue");
        std::vector<MRCompensatedRootTranslationGPU> init;std::vector<mr_float4> velocity;std::vector<unsigned> count;
        for(const auto& item:c){init.push_back(item.initial);velocity.push_back(item.velocity);count.push_back(item.count);}
        id<MTLBuffer> b[6]={
            [device newBufferWithBytes:init.data() length:init.size()*sizeof(init[0]) options:MTLResourceStorageModeShared],
            [device newBufferWithBytes:velocity.data() length:velocity.size()*sizeof(velocity[0]) options:MTLResourceStorageModeShared],
            [device newBufferWithBytes:count.data() length:count.size()*sizeof(count[0]) options:MTLResourceStorageModeShared],
            [device newBufferWithLength:init.size()*sizeof(init[0]) options:MTLResourceStorageModeShared],
            [device newBufferWithLength:init.size()*sizeof(mr_float4) options:MTLResourceStorageModeShared],
            [device newBufferWithLength:init.size()*sizeof(MRCompensatedPositionGPU) options:MTLResourceStorageModeShared]};
        for(auto item:b)require(item!=nil,"precision buffer allocation failed");
        id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:pipeline];for(unsigned i=0;i<6;++i)[enc setBuffer:b[i] offset:0 atIndex:i];
        const unsigned n=static_cast<unsigned>(c.size());[enc setBytes:&n length:sizeof(n) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
        require(cb.status==MTLCommandBufferStatusCompleted,"precision GPU command failed");
        require(std::memcmp(b[3].contents,expected.data(),expected.size()*sizeof(expected[0]))==0,"CPU/Metal translation words differ");
        require(std::memcmp(b[4].contents,projection.data(),projection.size()*sizeof(projection[0]))==0,"CPU/Metal projection words differ");
        const auto* pairs=static_cast<const MRCompensatedPositionGPU*>(b[5].contents);
        for(unsigned i=0;i<n;++i){const auto pair=mrCompensatedTranslationPosition(expected[i],{0,0,0,0});require(std::memcmp(&pairs[i],&pair,sizeof(pair))==0,"CPU/Metal geometry words differ");}
        std::printf("{\"mode\":\"metal\",\"passed\":true,\"cases\":%u,\"cpu_checks\":%u,\"max_displacement_error_m\":%.17g}\n",n,checks,maxError);return 0;
    }} catch(const std::exception& error){std::fprintf(stderr,"precision check failed: %s\n",error.what());return 1;}
}
