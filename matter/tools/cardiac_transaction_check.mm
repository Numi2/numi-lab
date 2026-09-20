#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/matter.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
__extension__ typedef unsigned __int128 Wide;
template<class T> id<MTLBuffer> buffer(id<MTLDevice> device, const std::vector<T>& values) {
    return [device newBufferWithBytes:values.data() length:values.size()*sizeof(T) options:MTLResourceStorageModeShared];
}
id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name) {
    NSError* error=nil;auto function=[library newFunctionWithName:name];
    auto result=[device newComputePipelineStateWithFunction:function error:&error];
    require(result!=nil,"production cardiac kernel unavailable");return result;
}
void complete(id<MTLCommandBuffer> cb) {
    [cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,"cardiac kernel command failed");
}
double reference(const NMVascularCompartmentGPU& node, NMVascularClockGPU clock) {
    const Wide ticks=(Wide(clock.high)<<64u)|clock.low;
    const double p=double((ticks%node.periodTicks)*node.periodMultiplier%node.periodTicks)/double(node.periodTicks)-node.waveform.y;
    const double start=node.elastance.z,end=node.elastance.w,pi=node.waveform.x;
    double a=0;
    if(node.identity.w==1u || node.identity.w==4u) {
        if(p<0)a=0;
        else if(p<=start)a=1-std::cos(pi*p/start);
        else if(p<=end)a=1+std::cos(pi*(p-start)/(end-start));
    } else {
        if(p<=start+end-1)a=1-std::cos(2*pi*(p-start+1)/end);
        else if(p>start)a=1-std::cos(2*pi*(p-start)/end);
    }
    return node.elastance.x+0.5*(node.elastance.y-node.elastance.x)*a;
}
void waveform(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    const std::uint64_t period=3ull<<44u;
    const std::array<double,10> phases={0,.005,.01,.1,.3,.4,.45,.6,.92,.97};
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=2u*phases.size();
    NMVascularLayoutGPU graph{};graph.counts.x=4;
    std::vector<NMVascularCompartmentGPU> nodes(4);
    for(unsigned i=0;i<2;++i){nodes[i].identity.w=i+1;nodes[i].periodTicks=period;nodes[i].periodMultiplier=1;nodes[i].waveform.x=3.14159f;
        nodes[i].elastance={10.f,100.f,i==0?.3f:.92f,i==0?.45f:.09f};}
    nodes[2]=nodes[0];nodes[2].identity.w=4;nodes[2].periodMultiplier=7;nodes[2].waveform={float(std::acos(-1.0)),.1f,0,0};
    nodes[3]=nodes[2];nodes[3].periodMultiplier=0xfffffffffffffffdull;
    std::vector<NMVascularClockGPU> clocks(dispatch.environmentCount);
    for(unsigned i=0;i<phases.size();++i){
        const auto rem=std::uint64_t(phases[i]*period);
        const Wide huge=Wide(period)*(Wide(1)<<60u)+rem;
        clocks[i]={rem,0};clocks[i+phases.size()]={std::uint64_t(huge),std::uint64_t(huge>>64u)};
    }
    auto nodeBuffer=buffer(device,nodes),clockBuffer=buffer(device,clocks);
    auto values=buffer(device,std::vector<float>(dispatch.environmentCount*4,0));
    auto statuses=buffer(device,std::vector<NMMatterStatusGPU>(dispatch.environmentCount));
    auto cb=[queue commandBuffer];auto encoder=[cb computeCommandEncoder];
    [encoder setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_vascular_prepare_elastance")];
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[encoder setBytes:&graph length:sizeof(graph) atIndex:1];
    [encoder setBuffer:nodeBuffer offset:0 atIndex:2];[encoder setBuffer:clockBuffer offset:0 atIndex:3];
    [encoder setBuffer:values offset:0 atIndex:4];[encoder setBuffer:statuses offset:0 atIndex:5];
    [encoder dispatchThreads:MTLSizeMake(dispatch.environmentCount*4,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [encoder endEncoding];complete(cb);
    const auto* actual=static_cast<const float*>(values.contents);
    const auto* status=static_cast<const NMMatterStatusGPU*>(statuses.contents);
    double worst=0;
    for(unsigned e=0;e<dispatch.environmentCount;++e){require(status[e].code==NM_STATUS_SUCCESS,"waveform rejected");
        for(unsigned n=0;n<4;++n){double error=std::abs(actual[e*4+n]-reference(nodes[n],clocks[e]));worst=std::max(worst,error);require(error<5e-4,"source waveform disagrees with independent FP64 evaluation");}}
    for(unsigned e=0;e<phases.size();++e)for(unsigned n=0;n<4;++n)
        require(actual[e*4+n]==actual[(e+phases.size())*4+n],"huge clock changed identical source phase");
    std::cout<<"cardiac_waveform=pass phase_cases="<<phases.size()<<" huge_clock_phase=bitwise rational_6_over_7=exact full_width_modular_multiplier=pass fp64_absolute_max="<<worst<<'\n';
}
void clockCarry(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=3;
    const std::uint64_t dt=7;
    std::vector<NMVascularClockGPU> clocks={{0xfffffffffffffffcul,12},{0xfffffffffffffffful,0xfffffffffffffffful},{123,0x100000001ul}};
    auto accepted=buffer(device,clocks),candidate=buffer(device,std::vector<NMVascularClockGPU>(3));
    auto statuses=buffer(device,std::vector<NMMatterStatusGPU>(3));
    auto cb=[queue commandBuffer];auto encoder=[cb computeCommandEncoder];
    [encoder setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_vascular_prepare_clock")];
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[encoder setBytes:&dt length:sizeof(dt) atIndex:1];
    [encoder setBuffer:accepted offset:0 atIndex:2];[encoder setBuffer:candidate offset:0 atIndex:3];[encoder setBuffer:statuses offset:0 atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(3,1,1) threadsPerThreadgroup:MTLSizeMake(3,1,1)];[encoder endEncoding];complete(cb);
    const auto* out=static_cast<const NMVascularClockGPU*>(candidate.contents);const auto* status=static_cast<const NMMatterStatusGPU*>(statuses.contents);
    require(out[0].low==3&&out[0].high==13&&status[0].code==0,"128-bit clock carry failed");
    require(out[1].low==clocks[1].low&&out[1].high==clocks[1].high&&status[1].code!=0,"clock overflow failed open");
    require(out[2].low==130&&out[2].high==clocks[2].high&&status[2].code==0,"clock overflow contaminated other environment");
    std::cout<<"cardiac_clock_kernel=pass carry=exact overflow=denied environment_isolation=pass\n";
}
void supportWorkingSetPivot(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=4;dispatch.objectCount=1;
    NMFGMRESLayoutGPU layout{};layout.supportContactCount=1;layout.supportBase=0;layout.unknownCount=4;
    NMHumanSupportDispatchGPU support{};support.contactCount=1;support.groundNormal={0,0,1,0};
    NMHumanSupportContactGPU contact{};contact.frictionSlopAndStabilization.x=1;auto contacts=buffer(device,std::vector<NMHumanSupportContactGPU>{contact});
    auto solution=buffer(device,std::vector<nm_float4>{{0,0,-1,0},{0,0,-1e-6f,0},{1,0,-1,0},{0,0,1,0}});
    const std::vector<nm_float4> initial={{0,0,0,0},{0,0,0,1e-8f},{0,0,0,1},{0,0,0,.25f}};
    auto histories=buffer(device,initial),working=buffer(device,std::vector<std::uint32_t>(4));
    auto coneWorking=buffer(device,std::vector<std::uint32_t>(4));
    auto coneTargets=buffer(device,std::vector<nm_float4>(4));
    auto changed=buffer(device,std::vector<std::uint32_t>(4));
    auto lines=buffer(device,std::vector<nm_float4>{{.8f,17,18,19},{.7f,17,18,19},{.9f,17,18,19},{.6f,17,18,19}});
    auto alpha=buffer(device,std::vector<float>(4,-1));
    auto statuses=buffer(device,std::vector<NMMatterStatusGPU>(4));
    auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_resolve_working_set")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];
    [e setBuffer:solution offset:0 atIndex:2];[e setBuffer:histories offset:0 atIndex:3];[e setBuffer:working offset:0 atIndex:4];[e setBuffer:changed offset:0 atIndex:5];[e setBuffer:statuses offset:0 atIndex:6];[e setBuffer:contacts offset:0 atIndex:7];[e setBuffer:coneWorking offset:0 atIndex:8];[e setBuffer:coneTargets offset:0 atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_limit_line_search")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];
    [e setBuffer:solution offset:0 atIndex:2];[e setBuffer:histories offset:0 atIndex:3];[e setBuffer:lines offset:0 atIndex:4];[e setBuffer:alpha offset:0 atIndex:5];[e setBuffer:statuses offset:0 atIndex:6];[e setBuffer:working offset:0 atIndex:7];[e setBuffer:changed offset:0 atIndex:8];[e setBuffer:contacts offset:0 atIndex:9];[e setBuffer:coneWorking offset:0 atIndex:10];[e setBuffer:coneTargets offset:0 atIndex:11];
    [e dispatchThreadgroups:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_apply_solution")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];
    [e setBuffer:solution offset:0 atIndex:2];[e setBuffer:alpha offset:0 atIndex:3];[e setBuffer:histories offset:0 atIndex:4];[e setBuffer:working offset:0 atIndex:5];[e setBuffer:changed offset:0 atIndex:6];[e setBuffer:statuses offset:0 atIndex:7];[e setBuffer:contacts offset:0 atIndex:8];[e setBuffer:coneWorking offset:0 atIndex:9];[e setBuffer:coneTargets offset:0 atIndex:10];
    [e dispatchThreads:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(4,1,1)];[e endEncoding];complete(cb);
    const auto* masks=static_cast<const std::uint32_t*>(working.contents);const auto* flags=static_cast<const std::uint32_t*>(changed.contents);
    const auto* out=static_cast<const float*>(alpha.contents);const auto* h=static_cast<const nm_float4*>(histories.contents);
    const auto* mirrored=static_cast<const nm_float4*>(lines.contents);const auto* status=static_cast<const NMMatterStatusGPU*>(statuses.contents);
    for(unsigned env=0;env<4;++env)require(status[env].code==NM_STATUS_SUCCESS,"support PDAS pivot rejected a finite case");
    for(unsigned env=0;env<3;++env){require(masks[env]==1&&flags[env]==1,"support normal-cone exit did not select a pending inactive row");require(std::memcmp(h+env,initial.data()+env,sizeof(nm_float4))==0,"discarded support direction changed its history");}
    require(masks[3]==0&&flags[3]==0,"support pivot contaminated an independent environment");
    require(h[3].x==0&&h[3].y==0&&h[3].z==0&&std::abs(h[3].w-.85f)<1e-7f,"feasible support row missed the shared object alpha");
    const std::array<float,4> expectedAlpha={.8f,.7f,.9f,.6f};
    for(unsigned env=0;env<4;++env)require(out[env]==expectedAlpha[env]&&mirrored[env].x==out[env]&&mirrored[env].y==17&&mirrored[env].z==18&&mirrored[env].w==19,"support PDAS lost the object-wide alpha");
    std::cout<<"cardiac_support_working_set_pivot=pass zero_and_tiny_positive=bounded exact_zero_with_tangent=inactive object_alpha=shared environment_isolation=pass\n";
}
void sharedAlpha(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    for(unsigned mode=0;mode<3;++mode) {
        NMMatterDispatchGPU dispatch{};dispatch.environmentCount=2;dispatch.rigidGeneralizedCapacity=1;dispatch.objectCount=mode==2?1:0;
        NMFGMRESLayoutGPU layout{};layout.supportContactCount=1;layout.supportBase=2;layout.vascularBase=4;layout.vascularUnknownCount=mode==1?0:1;
        NMHumanSupportDispatchGPU support{};support.contactCount=1;support.groundNormal={0,0,1,0};
        NMHumanSupportContactGPU contact{};contact.frictionSlopAndStabilization.x=10;auto contacts=buffer(device,std::vector<NMHumanSupportContactGPU>{contact});
        const std::vector<nm_float4> initial(2,nm_float4{1,2,0,4});
        auto candidate=buffer(device,std::vector<float>{2,2});auto histories=buffer(device,initial);
        auto solution=buffer(device,std::vector<nm_float4>{{8,0,0,0},{8,0,0,0},{8,12,16,0},{8,12,16,0}});
        auto working=buffer(device,std::vector<std::uint32_t>(2));auto changed=buffer(device,std::vector<std::uint32_t>(2));
        auto coneWorking=buffer(device,std::vector<std::uint32_t>(2));auto coneTargets=buffer(device,std::vector<nm_float4>(2));
        auto statuses=buffer(device,std::vector<NMMatterStatusGPU>(2));
        const std::vector<float> expectedAlpha = mode == 0
            ? std::vector<float>{0.0f, 0.25f}
            : mode == 1
                ? std::vector<float>{1.0f, 1.0f}
                : std::vector<float>{0.5f, 0.75f};
        auto alpha=buffer(device,expectedAlpha);
        auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
        [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_rigid_apply_candidate_solution")];
        [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];
        [e setBuffer:solution offset:0 atIndex:1];[e setBuffer:alpha offset:0 atIndex:2];[e setBuffer:candidate offset:0 atIndex:3];
        [e dispatchThreads:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(2,1,1)];
        [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_apply_solution")];
        [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];
        [e setBuffer:solution offset:0 atIndex:2];[e setBuffer:alpha offset:0 atIndex:3];[e setBuffer:histories offset:0 atIndex:4];[e setBuffer:working offset:0 atIndex:5];[e setBuffer:changed offset:0 atIndex:6];[e setBuffer:statuses offset:0 atIndex:7];[e setBuffer:contacts offset:0 atIndex:8];[e setBuffer:coneWorking offset:0 atIndex:9];[e setBuffer:coneTargets offset:0 atIndex:10];
        [e dispatchThreads:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(2,1,1)];[e endEncoding];complete(cb);
        const auto* out=static_cast<const float*>(candidate.contents);const auto* h=static_cast<const nm_float4*>(histories.contents);
        for(unsigned env=0;env<2;++env) {
            const float a=expectedAlpha[env];
            require(out[env]==2+8*a,"rigid candidate bypassed shared environment alpha");
            if(a==0)require(std::memcmp(h+env,initial.data()+env,sizeof(nm_float4))==0,"deferred support history changed");
            else require(h[env].x==1+8*a&&h[env].y==2+12*a&&h[env].z==0&&h[env].w==4+16*a,"support impulse bypassed shared environment alpha");
            require(static_cast<const NMMatterStatusGPU*>(statuses.contents)[env].code==0,"shared support alpha failed closed unexpectedly");
        }
    }
    std::cout<<"cardiac_shared_alpha=pass objectless_rigid_support=pass deferral=exact fractional=pass nonvascular_baseline=pass object_baseline=pass\n";
}
void supportConstrainedSharedAlpha(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=2;dispatch.objectCount=1;
    NMFGMRESLayoutGPU layout{};layout.supportContactCount=1;layout.supportBase=0;layout.unknownCount=2;
    NMHumanSupportDispatchGPU support{};support.contactCount=1;support.groundNormal={0,0,1,0};
    NMHumanSupportContactGPU contact{};contact.frictionSlopAndStabilization.x=1;auto contacts=buffer(device,std::vector<NMHumanSupportContactGPU>{contact});
    const std::vector<nm_float4> initial={{1,0,0,2},{0,1,0,1}};
    auto histories=buffer(device,initial),solution=buffer(device,std::vector<nm_float4>(2,nm_float4{9,9,9,0}));
    auto working=buffer(device,std::vector<std::uint32_t>{1,1}),changed=buffer(device,std::vector<std::uint32_t>(2));
    auto coneWorking=buffer(device,std::vector<std::uint32_t>(2)),coneTargets=buffer(device,std::vector<nm_float4>(2));
    auto lines=buffer(device,std::vector<nm_float4>{{.5f,2,3,4},{1,2,3,4}}),alpha=buffer(device,std::vector<float>(2,-1));
    auto statuses=buffer(device,std::vector<NMMatterStatusGPU>(2));
    auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_resolve_working_set")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:solution offset:0 atIndex:2];[e setBuffer:histories offset:0 atIndex:3];[e setBuffer:working offset:0 atIndex:4];[e setBuffer:changed offset:0 atIndex:5];[e setBuffer:statuses offset:0 atIndex:6];[e setBuffer:contacts offset:0 atIndex:7];[e setBuffer:coneWorking offset:0 atIndex:8];[e setBuffer:coneTargets offset:0 atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_limit_line_search")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:solution offset:0 atIndex:2];[e setBuffer:histories offset:0 atIndex:3];[e setBuffer:lines offset:0 atIndex:4];[e setBuffer:alpha offset:0 atIndex:5];[e setBuffer:statuses offset:0 atIndex:6];[e setBuffer:working offset:0 atIndex:7];[e setBuffer:changed offset:0 atIndex:8];[e setBuffer:contacts offset:0 atIndex:9];[e setBuffer:coneWorking offset:0 atIndex:10];[e setBuffer:coneTargets offset:0 atIndex:11];
    [e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_apply_solution")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:solution offset:0 atIndex:2];[e setBuffer:alpha offset:0 atIndex:3];[e setBuffer:histories offset:0 atIndex:4];[e setBuffer:working offset:0 atIndex:5];[e setBuffer:changed offset:0 atIndex:6];[e setBuffer:statuses offset:0 atIndex:7];[e setBuffer:contacts offset:0 atIndex:8];[e setBuffer:coneWorking offset:0 atIndex:9];[e setBuffer:coneTargets offset:0 atIndex:10];
    [e dispatchThreads:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(2,1,1)];[e endEncoding];complete(cb);
    const auto* s=static_cast<const nm_float4*>(solution.contents);const auto* h=static_cast<const nm_float4*>(histories.contents);const auto* mask=static_cast<const std::uint32_t*>(working.contents);
    require(s[0].x==-1&&s[0].y==0&&s[0].z==-2&&s[1].x==0&&s[1].y==-1&&s[1].z==-1,"constrained support solve did not use exact full -impulse");
    require(h[0].x==.5f&&h[0].y==0&&h[0].z==0&&h[0].w==1&&mask[0]==2,"partial object alpha did not scale full constrained impulse coherently");
    const nm_float4 zero{};require(std::memcmp(h+1,&zero,sizeof(zero))==0&&mask[1]==2,"full constrained step did not publish canonical zero");
    const auto* status=static_cast<const NMMatterStatusGPU*>(statuses.contents);require(status[0].code==0&&status[1].code==0,"constrained support shared-alpha solve failed");
    std::cout<<"cardiac_support_constrained_alpha=pass full_impulse=coherent partial_object_alpha=shared exact_zero=canonical state2=solved\n";
}
void supportConeIntersection(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=2;dispatch.objectCount=1;
    NMFGMRESLayoutGPU layout{};layout.supportContactCount=1;layout.supportBase=0;layout.unknownCount=2;
    NMHumanSupportDispatchGPU support{};support.contactCount=1;support.groundNormal={0,0,1,0};
    NMHumanSupportContactGPU contact{};contact.frictionSlopAndStabilization.x=.5f;auto contacts=buffer(device,std::vector<NMHumanSupportContactGPU>{contact});
    auto histories=buffer(device,std::vector<nm_float4>{{0,0,0,1},{.25f,0,0,1}}),solution=buffer(device,std::vector<nm_float4>{{1,0,0,0},{.1f,0,0,0}});
    auto working=buffer(device,std::vector<std::uint32_t>(2)),changed=buffer(device,std::vector<std::uint32_t>(2)),statuses=buffer(device,std::vector<NMMatterStatusGPU>(2));
    auto coneWorking=buffer(device,std::vector<std::uint32_t>(2)),coneTargets=buffer(device,std::vector<nm_float4>(2));
    auto lines=buffer(device,std::vector<nm_float4>{{.8f,2,3,4},{.4f,2,3,4}}),alpha=buffer(device,std::vector<float>(2,-1));
    auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_limit_line_search")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:solution offset:0 atIndex:2];[e setBuffer:histories offset:0 atIndex:3];[e setBuffer:lines offset:0 atIndex:4];[e setBuffer:alpha offset:0 atIndex:5];[e setBuffer:statuses offset:0 atIndex:6];[e setBuffer:working offset:0 atIndex:7];[e setBuffer:changed offset:0 atIndex:8];[e setBuffer:contacts offset:0 atIndex:9];[e setBuffer:coneWorking offset:0 atIndex:10];[e setBuffer:coneTargets offset:0 atIndex:11];
    [e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_apply_solution")];[e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:solution offset:0 atIndex:2];[e setBuffer:alpha offset:0 atIndex:3];[e setBuffer:histories offset:0 atIndex:4];[e setBuffer:working offset:0 atIndex:5];[e setBuffer:changed offset:0 atIndex:6];[e setBuffer:statuses offset:0 atIndex:7];[e setBuffer:contacts offset:0 atIndex:8];[e setBuffer:coneWorking offset:0 atIndex:9];[e setBuffer:coneTargets offset:0 atIndex:10];
    [e dispatchThreads:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(2,1,1)];[e endEncoding];complete(cb);
    const auto* a=static_cast<const float*>(alpha.contents);const auto* h=static_cast<const nm_float4*>(histories.contents);const auto* status=static_cast<const NMMatterStatusGPU*>(statuses.contents);
    require(a[0]==.5f&&h[0].x==.5f&&h[0].w==1,"shared support step did not land on the exact Coulomb-disk intersection");
    require(a[1]==.4f&&std::abs(h[1].x-.29f)<1e-7f&&h[1].w==1,"object alpha did not remain authoritative inside the support cone");
    require(status[0].code==0&&status[1].code==0,"support cone intersection rejected a finite feasible path");
    std::cout<<"cardiac_support_cone_intersection=pass shared_alpha=boundary object_alpha=interior no_dual_projection=true\n";
}
void supportConeScaleDrift(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=1;dispatch.objectCount=1;
    NMFGMRESLayoutGPU layout{};layout.supportContactCount=1;layout.supportBase=0;layout.unknownCount=1;
    NMHumanSupportDispatchGPU support{};support.contactCount=1;support.groundNormal={0,0,1,0};
    NMHumanSupportContactGPU contact{};contact.frictionSlopAndStabilization.x=1;auto contacts=buffer(device,std::vector<NMHumanSupportContactGPU>{contact});
    // Root 22's failing off-axis scale. The natural direction leaves the apex,
    // the resolver chooses a fixed-normal radial target, and another block's
    // exact shared alpha then interpolates toward that target.
    const nm_float4 initial={0,0,0,0};
    const float sharedAlpha=0.68359375f;
    auto histories=buffer(device,std::vector<nm_float4>{initial});
    auto solution=buffer(device,std::vector<nm_float4>{{-0.000241498041f,0.000424423342f,0.000244160008f,0}});
    auto working=buffer(device,std::vector<std::uint32_t>(1)),changed=buffer(device,std::vector<std::uint32_t>(1));
    auto coneWorking=buffer(device,std::vector<std::uint32_t>(1)),coneTargets=buffer(device,std::vector<nm_float4>(1));
    auto statuses=buffer(device,std::vector<NMMatterStatusGPU>(1));
    auto lines=buffer(device,std::vector<nm_float4>{{sharedAlpha,2,3,4}}),alpha=buffer(device,std::vector<float>(1,-1));
    auto encodeResolve=[&](id<MTLComputeCommandEncoder> e){
        [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_resolve_working_set")];
        [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];
        [e setBuffer:solution offset:0 atIndex:2];[e setBuffer:histories offset:0 atIndex:3];[e setBuffer:working offset:0 atIndex:4];[e setBuffer:changed offset:0 atIndex:5];[e setBuffer:statuses offset:0 atIndex:6];[e setBuffer:contacts offset:0 atIndex:7];[e setBuffer:coneWorking offset:0 atIndex:8];[e setBuffer:coneTargets offset:0 atIndex:9];
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    };
    auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];encodeResolve(e);[e endEncoding];complete(cb);
    const auto* target=static_cast<const nm_float4*>(coneTargets.contents);
    const double targetTangent=std::sqrt(double(target[0].x)*target[0].x+double(target[0].y)*target[0].y+double(target[0].z)*target[0].z);
    require(static_cast<const std::uint32_t*>(coneWorking.contents)[0]==1&&static_cast<const std::uint32_t*>(changed.contents)[0]==1,"small-scale boundary direction did not select a cone target");
    require(double(target[0].w)-targetTangent>0,"small-scale cone target has no representable interior slack");
    std::memset(changed.contents,0,changed.length);*static_cast<nm_float4*>(solution.contents)={9,8,7,0};
    cb=[queue commandBuffer];e=[cb computeCommandEncoder];encodeResolve(e);
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_limit_line_search")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:solution offset:0 atIndex:2];[e setBuffer:histories offset:0 atIndex:3];[e setBuffer:lines offset:0 atIndex:4];[e setBuffer:alpha offset:0 atIndex:5];[e setBuffer:statuses offset:0 atIndex:6];[e setBuffer:working offset:0 atIndex:7];[e setBuffer:changed offset:0 atIndex:8];[e setBuffer:contacts offset:0 atIndex:9];[e setBuffer:coneWorking offset:0 atIndex:10];[e setBuffer:coneTargets offset:0 atIndex:11];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_validate_final_line_search")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:histories offset:0 atIndex:2];[e setBuffer:lines offset:0 atIndex:3];[e setBuffer:alpha offset:0 atIndex:4];[e setBuffer:statuses offset:0 atIndex:5];[e setBuffer:changed offset:0 atIndex:6];[e setBuffer:contacts offset:0 atIndex:7];[e setBuffer:coneWorking offset:0 atIndex:8];[e setBuffer:coneTargets offset:0 atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_apply_solution")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:solution offset:0 atIndex:2];[e setBuffer:alpha offset:0 atIndex:3];[e setBuffer:histories offset:0 atIndex:4];[e setBuffer:working offset:0 atIndex:5];[e setBuffer:changed offset:0 atIndex:6];[e setBuffer:statuses offset:0 atIndex:7];[e setBuffer:contacts offset:0 atIndex:8];[e setBuffer:coneWorking offset:0 atIndex:9];[e setBuffer:coneTargets offset:0 atIndex:10];
    [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[e endEncoding];complete(cb);
    const auto* h=static_cast<const nm_float4*>(histories.contents);const auto* status=static_cast<const NMMatterStatusGPU*>(statuses.contents);
    const double tangent=std::sqrt(double(h[0].x)*h[0].x+double(h[0].y)*h[0].y+double(h[0].z)*h[0].z);
    require(status[0].code==0,"small-scale partial cone interpolation crossed the representable boundary");
    require(static_cast<const float*>(alpha.contents)[0]==sharedAlpha&&double(h[0].w)-tangent>0,"small-scale shared-alpha cone interpolation lost interior slack");
    std::cout<<"cardiac_support_cone_scale_drift=pass root22_scale=covered shared_alpha=interior strict_certificate=unchanged\n";
}
void supportConeFinalAlphaGate(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    const auto fromBits=[](const std::uint32_t bits){float value=0;std::memcpy(&value,&bits,sizeof(value));return value;};
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=3;dispatch.objectCount=1;
    NMFGMRESLayoutGPU layout{};layout.supportContactCount=1;layout.supportBase=0;layout.unknownCount=3;
    NMHumanSupportDispatchGPU support{};support.contactCount=1;support.groundNormal={0,0,1,0};
    NMHumanSupportContactGPU contact{};contact.frictionSlopAndStabilization.x=1;auto contacts=buffer(device,std::vector<NMHumanSupportContactGPU>{contact});
    const nm_float4 initial={fromBits(0xb8fd3aa3u),fromBits(0x395e8523u),0,fromBits(0x3980029au)};
    const nm_float4 target={fromBits(0xb84ee4cdu),fromBits(0x3a331303u),0,fromBits(0x3a338ab4u)};
    const float tinyAlpha=fromBits(0x32dad099u);
    const float roundingNoOpAlpha=fromBits(0x2b8cbcccu);
    auto histories=buffer(device,std::vector<nm_float4>{initial,initial,initial});
    auto solution=buffer(device,std::vector<nm_float4>{{9,8,7,0},{9,8,7,0},{9,8,7,0}});
    auto working=buffer(device,std::vector<std::uint32_t>(3));
    auto changed=buffer(device,std::vector<std::uint32_t>(3));
    auto coneWorking=buffer(device,std::vector<std::uint32_t>{1,1,1});
    auto coneTargets=buffer(device,std::vector<nm_float4>{target,target,target});
    auto statuses=buffer(device,std::vector<NMMatterStatusGPU>(3));
    auto lines=buffer(device,std::vector<nm_float4>{{tinyAlpha,2,3,4},{roundingNoOpAlpha,5,6,7},{1,8,9,10}});
    auto alpha=buffer(device,std::vector<float>(3,-1));
    auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_resolve_working_set")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:solution offset:0 atIndex:2];[e setBuffer:histories offset:0 atIndex:3];[e setBuffer:working offset:0 atIndex:4];[e setBuffer:changed offset:0 atIndex:5];[e setBuffer:statuses offset:0 atIndex:6];[e setBuffer:contacts offset:0 atIndex:7];[e setBuffer:coneWorking offset:0 atIndex:8];[e setBuffer:coneTargets offset:0 atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake(3,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_limit_line_search")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:solution offset:0 atIndex:2];[e setBuffer:histories offset:0 atIndex:3];[e setBuffer:lines offset:0 atIndex:4];[e setBuffer:alpha offset:0 atIndex:5];[e setBuffer:statuses offset:0 atIndex:6];[e setBuffer:working offset:0 atIndex:7];[e setBuffer:changed offset:0 atIndex:8];[e setBuffer:contacts offset:0 atIndex:9];[e setBuffer:coneWorking offset:0 atIndex:10];[e setBuffer:coneTargets offset:0 atIndex:11];
    [e dispatchThreadgroups:MTLSizeMake(3,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_validate_final_line_search")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:histories offset:0 atIndex:2];[e setBuffer:lines offset:0 atIndex:3];[e setBuffer:alpha offset:0 atIndex:4];[e setBuffer:statuses offset:0 atIndex:5];[e setBuffer:changed offset:0 atIndex:6];[e setBuffer:contacts offset:0 atIndex:7];[e setBuffer:coneWorking offset:0 atIndex:8];[e setBuffer:coneTargets offset:0 atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake(3,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_apply_solution")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:solution offset:0 atIndex:2];[e setBuffer:alpha offset:0 atIndex:3];[e setBuffer:histories offset:0 atIndex:4];[e setBuffer:working offset:0 atIndex:5];[e setBuffer:changed offset:0 atIndex:6];[e setBuffer:statuses offset:0 atIndex:7];[e setBuffer:contacts offset:0 atIndex:8];[e setBuffer:coneWorking offset:0 atIndex:9];[e setBuffer:coneTargets offset:0 atIndex:10];
    [e dispatchThreads:MTLSizeMake(3,1,1) threadsPerThreadgroup:MTLSizeMake(3,1,1)];[e endEncoding];complete(cb);
    const auto* status=static_cast<const NMMatterStatusGPU*>(statuses.contents);const auto* a=static_cast<const float*>(alpha.contents);const auto* line=static_cast<const nm_float4*>(lines.contents);const auto* h=static_cast<const nm_float4*>(histories.contents);const auto* cone=static_cast<const std::uint32_t*>(coneWorking.contents);
    require(status[0].code==NM_STATUS_CONTACT_FAILURE&&status[0].failingIndex==0&&status[0].diagnostics.z<0,"final shared-alpha cone gate did not reject the pinned FP32 counterexample");
    require(a[0]==0&&line[0].x==0&&line[0].y==2&&line[0].z==3&&line[0].w==4&&std::memcmp(h,&initial,sizeof(initial))==0,"failed cone gate mutated history or did not zero every shared-alpha owner");
    require(status[1].code==0&&a[1]==roundingNoOpAlpha&&line[1].x==roundingNoOpAlpha&&std::memcmp(h+1,&initial,sizeof(initial))==0&&cone[1]==1,"final cone gate imposed an arbitrary minimum alpha instead of validating the exact FP32 result");
    require(status[2].code==0&&a[2]==1&&line[2].x==1&&std::memcmp(h+2,&target,sizeof(target))==0&&cone[2]==2,"final cone gate rejected a full feasible target or contaminated its neighboring environment");
    std::cout<<"cardiac_support_cone_final_alpha=pass tiny_alpha=fail_closed smaller_rounding_noop=accepted preapply_history=unchanged environment_isolation=covered\n";
}
void supportConeWorkingSetPivot(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=4;dispatch.objectCount=1;
    NMFGMRESLayoutGPU layout{};layout.supportContactCount=1;layout.supportBase=0;layout.unknownCount=4;
    NMHumanSupportDispatchGPU support{};support.contactCount=1;support.groundNormal={0,0,1,0};
    NMHumanSupportContactGPU contact{};contact.frictionSlopAndStabilization.x=.5f;auto contacts=buffer(device,std::vector<NMHumanSupportContactGPU>{contact});
    const std::vector<nm_float4> initial={{0,0,0,0},{.5f,0,0,1},{.5f,0,0,1},{.25f,0,0,1}};
    auto histories=buffer(device,initial),solution=buffer(device,std::vector<nm_float4>{{1,0,1,0},{.2f,0,.2f,0},{.2f,0,.2f,0},{.1f,0,0,0}});
    auto working=buffer(device,std::vector<std::uint32_t>(4)),coneWorking=buffer(device,std::vector<std::uint32_t>(4));
    auto coneTargets=buffer(device,std::vector<nm_float4>(4)),changed=buffer(device,std::vector<std::uint32_t>(4));
    auto statuses=buffer(device,std::vector<NMMatterStatusGPU>(4));
    auto lines=buffer(device,std::vector<nm_float4>{{1,2,3,4},{.5f,2,3,4},{1,2,3,4},{.75f,2,3,4}}),alpha=buffer(device,std::vector<float>(4,-1));
    auto resolve=[&]{auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_resolve_working_set")];
        [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:solution offset:0 atIndex:2];[e setBuffer:histories offset:0 atIndex:3];[e setBuffer:working offset:0 atIndex:4];[e setBuffer:changed offset:0 atIndex:5];[e setBuffer:statuses offset:0 atIndex:6];[e setBuffer:contacts offset:0 atIndex:7];[e setBuffer:coneWorking offset:0 atIndex:8];[e setBuffer:coneTargets offset:0 atIndex:9];
        [e dispatchThreadgroups:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];complete(cb);};
    auto limitAndApply=[&]{auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_limit_line_search")];
        [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:solution offset:0 atIndex:2];[e setBuffer:histories offset:0 atIndex:3];[e setBuffer:lines offset:0 atIndex:4];[e setBuffer:alpha offset:0 atIndex:5];[e setBuffer:statuses offset:0 atIndex:6];[e setBuffer:working offset:0 atIndex:7];[e setBuffer:changed offset:0 atIndex:8];[e setBuffer:contacts offset:0 atIndex:9];[e setBuffer:coneWorking offset:0 atIndex:10];[e setBuffer:coneTargets offset:0 atIndex:11];
        [e dispatchThreadgroups:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_apply_solution")];[e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:solution offset:0 atIndex:2];[e setBuffer:alpha offset:0 atIndex:3];[e setBuffer:histories offset:0 atIndex:4];[e setBuffer:working offset:0 atIndex:5];[e setBuffer:changed offset:0 atIndex:6];[e setBuffer:statuses offset:0 atIndex:7];[e setBuffer:contacts offset:0 atIndex:8];[e setBuffer:coneWorking offset:0 atIndex:9];[e setBuffer:coneTargets offset:0 atIndex:10];
        [e dispatchThreads:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(4,1,1)];[e endEncoding];complete(cb);};
    resolve();
    const auto* firstModes=static_cast<const std::uint32_t*>(coneWorking.contents);const auto* firstFlags=static_cast<const std::uint32_t*>(changed.contents);const auto* firstTargets=static_cast<const nm_float4*>(coneTargets.contents);const auto* discarded=static_cast<const nm_float4*>(solution.contents);
    for(unsigned env=0;env<3;++env){const float margin=.5f*firstTargets[env].w-std::sqrt(firstTargets[env].x*firstTargets[env].x+firstTargets[env].y*firstTargets[env].y+firstTargets[env].z*firstTargets[env].z);require(firstModes[env]==1&&firstFlags[env]==1&&margin>0&&margin<1e-5f,"support cone pivot missed representable interior target slack");require(discarded[env].x==0&&discarded[env].y==0&&discarded[env].z==0,"support cone pivot did not discard the rejected direction");}
    require(firstTargets[0].w==1&&firstTargets[1].w==1.2f&&firstTargets[2].w==1.2f,"support cone pivot changed the solved normal");
    require(firstModes[3]==0&&firstFlags[3]==0&&discarded[3].x==.1f,"support cone pivot contaminated an interior environment");
    std::memset(changed.contents,0,4*sizeof(std::uint32_t));const std::vector<nm_float4> arbitrary={{9,8,7,0},{9,8,7,0},{9,8,7,0},{0,0,0,0}};std::memcpy(solution.contents,arbitrary.data(),arbitrary.size()*sizeof(nm_float4));resolve();limitAndApply();
    const auto* h=static_cast<const nm_float4*>(histories.contents);const auto* modes=static_cast<const std::uint32_t*>(coneWorking.contents);const auto* targets=static_cast<const nm_float4*>(coneTargets.contents);
    require(modes[0]==2&&modes[1]==1&&modes[2]==2&&modes[3]==0,"support cone target ignored shared-alpha completion state");
    require(std::memcmp(h+0,targets+0,sizeof(nm_float4))==0&&std::memcmp(h+2,targets+2,sizeof(nm_float4))==0,"full cone-target solve did not publish its exact selected impulse");
    require(h[1].w>initial[1].w&&h[1].w<targets[1].w&&.5f*h[1].w-std::sqrt(h[1].x*h[1].x+h[1].y*h[1].y+h[1].z*h[1].z)>=0,"partial shared alpha left the cone-target segment");
    std::vector<NMContactSampleGPU> samples(4);for(unsigned env=0;env<4;++env){samples[env].identity.w=NM_CONTACT_VALID;samples[env].impulseAndNormal={h[env].x,h[env].y,h[env].w,h[env].w};}
    std::vector<NMHumanSupportKKTGPU> natural(4);for(auto& k:natural){k.residual={3,4,5,0};k.projectionRow0={1,0,0,0};k.projectionRow1={0,1,0,0};k.projectionRow2={0,0,1,0};}
    auto sampleBuffer=buffer(device,samples),kkt=buffer(device,natural),residual=buffer(device,std::vector<nm_float4>(4,nm_float4{3,4,5,0}));NMMicrostepGPU micro{};micro.solverIteration=2;
    auto select=[&]{auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_select_working_set")];[e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBytes:&micro length:sizeof(micro) atIndex:2];[e setBuffer:sampleBuffer offset:0 atIndex:3];[e setBuffer:histories offset:0 atIndex:4];[e setBuffer:kkt offset:0 atIndex:5];[e setBuffer:residual offset:0 atIndex:6];[e setBuffer:working offset:0 atIndex:7];[e setBuffer:statuses offset:0 atIndex:8];[e setBuffer:coneWorking offset:0 atIndex:9];[e setBuffer:coneTargets offset:0 atIndex:10];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e dispatchThreads:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(4,1,1)];[e endEncoding];complete(cb);};
    select();require(static_cast<const std::uint32_t*>(coneWorking.contents)[0]==0&&static_cast<const std::uint32_t*>(coneWorking.contents)[1]==1&&static_cast<const std::uint32_t*>(coneWorking.contents)[2]==0,"completed cone target did not return to the natural law");
    std::memset(changed.contents,0,4*sizeof(std::uint32_t));std::memset(solution.contents,0,4*sizeof(nm_float4));resolve();for(unsigned env=0;env<4;++env)static_cast<nm_float4*>(lines.contents)[env].x=1;limitAndApply();
    h=static_cast<const nm_float4*>(histories.contents);samples=std::vector<NMContactSampleGPU>(4);for(unsigned env=0;env<4;++env){samples[env].identity.w=NM_CONTACT_VALID;samples[env].impulseAndNormal={h[env].x,h[env].y,h[env].w,h[env].w};}std::memcpy(sampleBuffer.contents,samples.data(),samples.size()*sizeof(NMContactSampleGPU));select();
    const auto* finalModes=static_cast<const std::uint32_t*>(coneWorking.contents);const auto* status=static_cast<const NMMatterStatusGPU*>(statuses.contents);for(unsigned env=0;env<4;++env){require(finalModes[env]==0,"cone-target pivot did not terminate back on the natural equation");require(status[env].code==0,"cone-target pivot rejected a finite case");}
    std::cout<<"cardiac_support_cone_working_set=pass apex_sliding=resolved warm_sliding=resolved shared_alpha=preserved full_direction=resolved environment_isolation=pass\n";
}
void supportWorkingSetRelease(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=3;
    NMFGMRESLayoutGPU layout{};layout.supportContactCount=1;layout.supportBase=0;layout.unknownCount=3;
    NMHumanSupportDispatchGPU support{};support.contactCount=1;support.groundNormal={0,0,1,0};
    NMHumanSupportContactGPU contact{};contact.frictionSlopAndStabilization.x=1;auto contacts=buffer(device,std::vector<NMHumanSupportContactGPU>{contact});
    NMMicrostepGPU micro{};micro.solverIteration=1;
    std::vector<NMContactSampleGPU> samples(3);for(auto& sample:samples)sample.identity.w=NM_CONTACT_VALID;
    samples[0].admissionVelocityAndNormal.w=-.1f;samples[1].admissionVelocityAndNormal.w=.1f;samples[2].admissionVelocityAndNormal.w=-.1f;
    samples[2].impulseAndNormal={.2f,0,.1f,.1f};
    std::vector<NMHumanSupportKKTGPU> initialKKT(3);for(auto& kkt:initialKKT){kkt.residual={1,2,3,0};kkt.projectionRow0={1,0,0,0};kkt.projectionRow1={0,1,0,0};kkt.projectionRow2={0,0,1,0};}
    auto sampleBuffer=buffer(device,samples),histories=buffer(device,std::vector<nm_float4>{{0,0,0,0},{0,0,0,0},{.2f,0,0,.1f}});
    auto kkt=buffer(device,initialKKT),residual=buffer(device,std::vector<nm_float4>(3,nm_float4{1,2,3,0}));
    auto working=buffer(device,std::vector<std::uint32_t>{2,2,1}),changed=buffer(device,std::vector<std::uint32_t>(3));
    auto coneWorking=buffer(device,std::vector<std::uint32_t>(3)),coneTargets=buffer(device,std::vector<nm_float4>(3));
    auto statuses=buffer(device,std::vector<NMMatterStatusGPU>(3));
    auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_select_working_set")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBytes:&micro length:sizeof(micro) atIndex:2];[e setBuffer:sampleBuffer offset:0 atIndex:3];[e setBuffer:histories offset:0 atIndex:4];[e setBuffer:kkt offset:0 atIndex:5];[e setBuffer:residual offset:0 atIndex:6];[e setBuffer:working offset:0 atIndex:7];[e setBuffer:statuses offset:0 atIndex:8];[e setBuffer:coneWorking offset:0 atIndex:9];[e setBuffer:coneTargets offset:0 atIndex:10];[e setBytes:&layout length:sizeof(layout) atIndex:30];
    [e dispatchThreads:MTLSizeMake(3,1,1) threadsPerThreadgroup:MTLSizeMake(3,1,1)];[e endEncoding];complete(cb);
    const auto* maskAfterSelect=static_cast<const std::uint32_t*>(working.contents);const auto* r=static_cast<const nm_float4*>(residual.contents);const auto* l=static_cast<const NMHumanSupportKKTGPU*>(kkt.contents);
    require(maskAfterSelect[0]==0&&r[0].x==1&&l[0].projectionRow2.z==1,"solved zero support failed to release on violated inactive inequality");
    require(maskAfterSelect[1]==2&&r[1].x==0&&l[1].projectionRow0.x==0,"feasible solved inactive support row released or retained a derivative");
    require(maskAfterSelect[2]==1&&r[2].x==-.2f&&r[2].z==-.1f&&l[2].projectionRow0.x==0,"pending support row bypassed its constrained full-impulse solve");
    auto solution=buffer(device,std::vector<nm_float4>{{0,0,-.1f,0},{7,8,9,0},{4,5,6,0}});
    cb=[queue commandBuffer];e=[cb computeCommandEncoder];[e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_resolve_working_set")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:solution offset:0 atIndex:2];[e setBuffer:histories offset:0 atIndex:3];[e setBuffer:working offset:0 atIndex:4];[e setBuffer:changed offset:0 atIndex:5];[e setBuffer:statuses offset:0 atIndex:6];[e setBuffer:contacts offset:0 atIndex:7];[e setBuffer:coneWorking offset:0 atIndex:8];[e setBuffer:coneTargets offset:0 atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake(3,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];complete(cb);
    const auto* flags=static_cast<const std::uint32_t*>(changed.contents);const auto* out=static_cast<const nm_float4*>(solution.contents);const auto* mask=static_cast<const std::uint32_t*>(working.contents);
    require(mask[0]==1&&flags[0]==1,"released support row did not reactivate through a fresh free solve");
    require(mask[1]==2&&flags[1]==0&&out[1].x==0&&out[1].y==0&&out[1].z==0,"support release contaminated a feasible inactive environment");
    require(mask[2]==1&&flags[2]==0&&out[2].x==-.2f&&out[2].y==0&&out[2].z==-.1f,"pending support row did not receive exact constrained direction");
    const auto* status=static_cast<const NMMatterStatusGPU*>(statuses.contents);for(unsigned env=0;env<3;++env)require(status[env].code==0,"support release/reactivation failed");
    std::cout<<"cardiac_support_release=pass exact_zero_and_violated_inequality=released pending_row=retained reactivation=fresh_solve environment_isolation=pass\n";
}
void supportMonolithicDeferral(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=2;dispatch.femNodeCount=1;dispatch.mpmActiveNodeCapacity=1;dispatch.rigidGeneralizedCapacity=1;
    NMFGMRESLayoutGPU layout{};layout.supportContactCount=1;layout.supportBase=8;layout.vascularUnknownCount=1;layout.vascularBase=10;layout.unknownCount=12;
    NMHumanSupportDispatchGPU support{};support.contactCount=1;support.groundNormal={0,0,1,0};
    NMHumanSupportContactGPU contact{};contact.frictionSlopAndStabilization.x=1;auto contacts=buffer(device,std::vector<NMHumanSupportContactGPU>{contact});
    const std::vector<nm_float4> before(12,nm_float4{.125f,.25f,.5f,1});auto direction=before;direction[8]={0,0,-1,0};direction[9]={0,0,1,0};
    auto solution=buffer(device,direction),histories=buffer(device,std::vector<nm_float4>{{0,0,0,0},{0,0,0,1}}),working=buffer(device,std::vector<std::uint32_t>(2)),changed=buffer(device,std::vector<std::uint32_t>(2)),statuses=buffer(device,std::vector<NMMatterStatusGPU>(2));
    auto coneWorking=buffer(device,std::vector<std::uint32_t>(2)),coneTargets=buffer(device,std::vector<nm_float4>(2));
    auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_resolve_working_set")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBuffer:solution offset:0 atIndex:2];[e setBuffer:histories offset:0 atIndex:3];[e setBuffer:working offset:0 atIndex:4];[e setBuffer:changed offset:0 atIndex:5];[e setBuffer:statuses offset:0 atIndex:6];[e setBuffer:contacts offset:0 atIndex:7];[e setBuffer:coneWorking offset:0 atIndex:8];[e setBuffer:coneTargets offset:0 atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];complete(cb);
    const auto* out=static_cast<const nm_float4*>(solution.contents);const auto* flags=static_cast<const std::uint32_t*>(changed.contents);const nm_float4 zero{};
    const std::array<unsigned,6> cleared={0,2,4,6,8,10};for(unsigned i=0;i<12;++i){const bool reset=std::find(cleared.begin(),cleared.end(),i)!=cleared.end();require(std::memcmp(out+i,reset?&zero:direction.data()+i,sizeof(nm_float4))==0,"support pivot did not discard exactly one environment's monolithic direction");}
    require(flags[0]==1&&flags[1]==0,"support monolithic deferral contaminated another environment");
    std::cout<<"cardiac_support_monolithic_deferral=pass fem_mpm_rigid_support_vascular=zero environment_isolation=pass\n";
}
void supportConeCertificate(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=4;
    NMFGMRESLayoutGPU layout{};layout.supportContactCount=2;layout.supportBase=0;layout.unknownCount=8;
    NMHumanSupportDispatchGPU support{};support.contactCount=2;support.groundNormal={0,0,1,0};
    NMMixedSolverGPU solver{};solver.regularization.x=1e-4f;solver.residualTolerances.x=1e-4f;
    std::vector<NMHumanSupportContactGPU> contacts(2);contacts[0].frictionSlopAndStabilization.x=0;contacts[1].frictionSlopAndStabilization.x=.5f;
    auto contactBuffer=buffer(device,contacts),residual=buffer(device,std::vector<nm_float4>(8));auto states=buffer(device,std::vector<NMFGMRESStateGPU>(4));auto statuses=buffer(device,std::vector<NMMatterStatusGPU>(4));
    auto histories=buffer(device,std::vector<nm_float4>{{0,0,0,1},{.5f,0,0,1},{1e-3f,0,0,1},{.5f,0,0,1},{0,0,0,1},{.501f,0,0,1},{0,0,0,-1},{0,0,0,1}});
    auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_certify")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&solver length:sizeof(solver) atIndex:1];[e setBuffer:residual offset:0 atIndex:2];[e setBuffer:states offset:0 atIndex:3];[e setBuffer:statuses offset:0 atIndex:4];[e setBuffer:histories offset:0 atIndex:5];[e setBytes:&support length:sizeof(support) atIndex:6];[e setBuffer:contactBuffer offset:0 atIndex:7];
    [e dispatchThreadgroups:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];complete(cb);
    const auto* status=static_cast<const NMMatterStatusGPU*>(statuses.contents);require(status[0].code==0,"valid zero-friction and Coulomb-boundary histories failed certification");require(status[1].code==NM_STATUS_CONTACT_FAILURE&&status[1].failingIndex==0,"mu=0 retained a tangential impulse");require(status[2].code==NM_STATUS_CONTACT_FAILURE&&status[2].failingIndex==1,"negative terminal Coulomb margin passed certification");require(status[3].code==NM_STATUS_CONTACT_FAILURE&&status[3].failingIndex==0,"negative normal tension passed certification");
    std::cout<<"cardiac_support_cone_certificate=pass mu_zero=exact coulomb_margin=terminal normal_tension=forbidden\n";
}
void deferredDirection(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=2;dispatch.femNodeCount=1;dispatch.rigidGeneralizedCapacity=1;dispatch.mpmActiveNodeCapacity=1;
    NMFGMRESLayoutGPU layout{};layout.supportContactCount=1;layout.supportBase=8;layout.vascularBase=10;layout.vascularUnknownCount=3;layout.unknownCount=16;
    NMVascularLayoutGPU graph{};graph.counts={2,1,0,0};graph.ranges.z=3;graph.offsets={0,2,3,3};graph.cavities.z=3; // ABI 29: empty pressure block follows all hydraulic rows.
    NMVascularConnectionGPU edge{};edge.identity={1,0,1,1};
    std::vector<nm_float4> before(16,nm_float4{.125f,.25f,.5f,1.f});before[12].x=-.5f;before[15].x=.1f;
    auto solution=buffer(device,before),candidate=buffer(device,std::vector<nm_float4>{{1,0,0,0},{1,0,0,0},{.2,0,0,0},{1,0,0,0},{1,0,0,0},{.2,0,0,0}});
    auto edges=buffer(device,std::vector<NMVascularConnectionGPU>{edge});auto masks=buffer(device,std::vector<std::uint32_t>(6));auto changed=buffer(device,std::vector<std::uint32_t>(2));auto statuses=buffer(device,std::vector<NMMatterStatusGPU>(2));
    std::vector<NMVascularUnknownGPU> unknowns(3);for(auto& u:unknowns)u.initialAndScaling={0,1,1,1e-5f};
    auto scaling=buffer(device,unknowns),nodes=buffer(device,std::vector<NMVascularCompartmentGPU>(2));
    auto coefficients=buffer(device,std::vector<float>(4,1));
    auto cavities=buffer(device,std::vector<NMVascularCavityGPU>(1)),cavityMap=buffer(device,std::vector<std::uint32_t>(2,NM_INVALID_INDEX));
    auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_vascular_resolve_working_set")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&graph length:sizeof(graph) atIndex:1];[e setBytes:&layout length:sizeof(layout) atIndex:30];
    [e setBuffer:edges offset:0 atIndex:2];[e setBuffer:candidate offset:0 atIndex:3];[e setBuffer:solution offset:0 atIndex:4];[e setBuffer:masks offset:0 atIndex:5];[e setBuffer:changed offset:0 atIndex:6];[e setBuffer:statuses offset:0 atIndex:7];[e setBuffer:scaling offset:0 atIndex:8];[e setBuffer:nodes offset:0 atIndex:9];[e setBuffer:coefficients offset:0 atIndex:10];
    [e setBuffer:cavities offset:0 atIndex:11];[e setBuffer:cavityMap offset:0 atIndex:12];
    [e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];complete(cb);
    const auto* out=static_cast<const nm_float4*>(solution.contents);const auto* flags=static_cast<const std::uint32_t*>(changed.contents);
    require(flags[0]==1&&flags[1]==0,"valve working-set deferral contaminated another environment");
    const std::array<unsigned,8> cleared={0,2,4,6,8,10,11,12};
    const nm_float4 zero{};
    for(unsigned i=0;i<16;++i){const bool reset=std::find(cleared.begin(),cleared.end(),i)!=cleared.end();
        require(std::memcmp(out+i,reset?&zero:before.data()+i,sizeof(nm_float4))==0,"discarded Newton direction reached another physical line limiter");}
    require(static_cast<const std::uint32_t*>(masks.contents)[2]==1,"crossing valve did not enter pending working set");
    std::cout<<"cardiac_working_set_deferral=pass mechanical_direction=zero all_blocks=pass environment_isolation=pass\n";
}
void pendingZeroRelease(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=3;
    NMFGMRESLayoutGPU layout{};layout.vascularUnknownCount=3;layout.unknownCount=9;
    NMVascularLayoutGPU graph{};graph.counts={2,1,0,0};graph.ranges.z=3;graph.offsets={0,2,3,3};graph.cavities.z=3; // ABI 29: empty pressure block follows all hydraulic rows.
    NMVascularConnectionGPU edge{};edge.identity={1,0,1,1};
    std::vector<NMVascularUnknownGPU> unknowns(3);for(auto& u:unknowns)u.initialAndScaling={0,1,1,1e-5f};
    auto scaling=buffer(device,unknowns),nodes=buffer(device,std::vector<NMVascularCompartmentGPU>(2));
    auto coefficients=buffer(device,std::vector<float>(6,1));
    auto cavities=buffer(device,std::vector<NMVascularCavityGPU>(1)),cavityMap=buffer(device,std::vector<std::uint32_t>(2,NM_INVALID_INDEX));
    auto candidate=buffer(device,std::vector<nm_float4>{{2,0,0,0},{1,0,0,0},{0,0,0,0},{2,0,0,0},{1,0,0,0},{0,0,0,0},{1,0,0,0},{2,0,0,0},{0,0,0,0}});
    std::vector<nm_float4> directions(9);directions[5].x=-.5f;
    auto solution=buffer(device,directions),edges=buffer(device,std::vector<NMVascularConnectionGPU>{edge});
    auto masks=buffer(device,std::vector<std::uint32_t>{0,0,1,0,0,0,0,0,1}),changed=buffer(device,std::vector<std::uint32_t>(3));
    auto statuses=buffer(device,std::vector<NMMatterStatusGPU>(3));
    auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
    [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_vascular_resolve_working_set")];
    [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&graph length:sizeof(graph) atIndex:1];[e setBytes:&layout length:sizeof(layout) atIndex:30];
    [e setBuffer:edges offset:0 atIndex:2];[e setBuffer:candidate offset:0 atIndex:3];[e setBuffer:solution offset:0 atIndex:4];[e setBuffer:masks offset:0 atIndex:5];[e setBuffer:changed offset:0 atIndex:6];[e setBuffer:statuses offset:0 atIndex:7];[e setBuffer:scaling offset:0 atIndex:8];[e setBuffer:nodes offset:0 atIndex:9];[e setBuffer:coefficients offset:0 atIndex:10];
    [e setBuffer:cavities offset:0 atIndex:11];[e setBuffer:cavityMap offset:0 atIndex:12];
    [e dispatchThreadgroups:MTLSizeMake(3,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];complete(cb);
    const auto* mask=static_cast<const std::uint32_t*>(masks.contents);const auto* flags=static_cast<const std::uint32_t*>(changed.contents);
    require(mask[2]==0&&flags[0]==1,"pending zero-flow valve trapped after empty positive-pressure free solve");
    require(mask[5]==1&&flags[1]==1,"newly crossing valve released before its constrained free solve");
    require(mask[8]==1&&flags[2]==0,"feasible closed valve released despite adverse pressure");
    const auto* status=static_cast<const NMMatterStatusGPU*>(statuses.contents);for(unsigned i=0;i<3;++i)require(status[i].code==0,"pending valve release failed");
    std::cout<<"cardiac_pending_zero_release=pass empty_free_solve=released new_bound=retained feasible_bound=retained\n";
}
struct EquationResult { std::vector<nm_float4> residual, tangent; std::vector<NMMatterStatusGPU> status; };
EquationResult equations(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library,
    const std::vector<NMVascularCompartmentGPU>& nodes,NMVascularConnectionGPU edge,
    const std::vector<nm_float4>& candidate,const std::vector<nm_float4>& direction) {
    NMMatterDispatchGPU dispatch{};dispatch.environmentCount=candidate.size()/3;
    NMVascularLayoutGPU graph{};graph.counts={2,1,0,0};graph.ranges.z=3;graph.offsets={0,2,3,3};graph.cavities.z=3; // ABI 29: empty pressure block follows all hydraulic rows.
    NMFGMRESLayoutGPU layout{};layout.vascularUnknownCount=3;layout.unknownCount=candidate.size();
    NMMicrostepGPU micro{};micro.time.x=.01f;const unsigned unmasked=0;
    std::vector<NMVascularUnknownGPU> unknowns(3);for(auto& u:unknowns)u.initialAndScaling={0,1,1,1e-5f};
    auto scaling=buffer(device,unknowns),nodeBuffer=buffer(device,nodes),edges=buffer(device,std::vector<NMVascularConnectionGPU>{edge});
    auto tissues=buffer(device,std::vector<NMVascularTissueGPU>(1)),exchanges=buffer(device,std::vector<NMVascularExchangeGPU>(1));
    auto indices=buffer(device,std::vector<unsigned>{0,0}),ranges=buffer(device,std::vector<NMVascularRangeGPU>{{0,1,0,0},{1,1,0,0}});
    auto dummyRanges=buffer(device,std::vector<NMVascularRangeGPU>(1)),values=buffer(device,candidate),directions=buffer(device,direction);
    auto residual=buffer(device,std::vector<nm_float4>(candidate.size())),tangent=buffer(device,std::vector<nm_float4>(candidate.size()));
    auto states=buffer(device,std::vector<NMFGMRESStateGPU>(dispatch.environmentCount)),statuses=buffer(device,std::vector<NMMatterStatusGPU>(dispatch.environmentCount));
    auto coefficients=buffer(device,std::vector<float>(dispatch.environmentCount*2,1));
    auto masks=buffer(device,std::vector<unsigned>(candidate.size()));
    // Non-cavity fixtures still bind every production argument with a valid
    // buffer; the empty cavity range prevents these dummy records being read.
    auto cavities=buffer(device,std::vector<NMVascularCavityGPU>(1)),faces=buffer(device,std::vector<NMVascularCavityFaceGPU>(1));
    auto cavityMap=buffer(device,std::vector<std::uint32_t>(2,NM_INVALID_INDEX));
    auto femCandidate=buffer(device,std::vector<NMFEMNodeStateGPU>(1)),femAccepted=buffer(device,std::vector<NMFEMNodeStateGPU>(1));
    auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
    for(unsigned derivative=0;derivative<2;++derivative) {
        [e setComputePipelineState:pipeline(device,library,derivative?@"numi_matter_metal::nm_vascular_operator":@"numi_matter_metal::nm_vascular_residual")];
        [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&graph length:sizeof(graph) atIndex:1];[e setBytes:&micro length:sizeof(micro) atIndex:2];
        const std::array<id<MTLBuffer>,19> bindings={scaling,nodeBuffer,edges,tissues,exchanges,indices,ranges,indices,dummyRanges,indices,dummyRanges,values,values,directions,derivative?tangent:residual,states,statuses,coefficients,masks};
        for(unsigned i=0;i<bindings.size();++i)[e setBuffer:bindings[i] offset:0 atIndex:i+3];
        [e setBytes:&unmasked length:sizeof(unmasked) atIndex:22];
        [e setBuffer:cavities offset:0 atIndex:23];[e setBuffer:faces offset:0 atIndex:24];[e setBuffer:cavityMap offset:0 atIndex:25];
        [e setBuffer:femCandidate offset:0 atIndex:26];[e setBuffer:femAccepted offset:0 atIndex:27];[e setBytes:&layout length:sizeof(layout) atIndex:30];
        [e dispatchThreads:MTLSizeMake(candidate.size(),1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [e endEncoding];complete(cb);
    EquationResult out;out.residual.resize(candidate.size());out.tangent.resize(candidate.size());out.status.resize(dispatch.environmentCount);
    std::memcpy(out.residual.data(),residual.contents,residual.length);std::memcpy(out.tangent.data(),tangent.contents,tangent.length);std::memcpy(out.status.data(),statuses.contents,statuses.length);return out;
}
void regionalLaws(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    std::vector<NMVascularCompartmentGPU> nodes(2);for(auto& n:nodes)n.compliance={2,0,1,0};
    nodes[0].identity.w=3;nodes[0].compliance.z=.25f;nodes[0].pressureParameters.x=1;nodes[0].waveform.x=float(std::acos(-1.0));
    NMVascularConnectionGPU edge{};edge.identity={1,0,1,0};edge.physical.x=1;
    const std::array<float,9> displacement={-.99f,-.5f,-.0001f,0,.0001f,.5f,.99f,-1,1};
    std::vector<nm_float4> candidate,direction;
    for(float d:displacement){candidate.insert(candidate.end(),{{2+d,0,0,0},{2,0,0,0},{0,0,0,0}});direction.insert(direction.end(),{{.01f,0,0,0},{0,0,0,0},{0,0,0,0}});}
    auto out=equations(device,queue,library,nodes,edge,candidate,direction);
    double worst=0;
    for(unsigned i=0;i<displacement.size();++i){
        if(i>=7){require(out.status[i].code!=0,"atan asymptote admitted by production residual");continue;}
        require(out.status[i].code==0,"interior atan state rejected");
        const double d=double(candidate[3*i].x)-2,pi=nodes[0].waveform.x,angle=.5*pi*d;
        const double t=std::tan(angle),pressure=8/pi*t,derivative=4*(1+t*t);
        const double pError=std::abs(out.residual[3*i+2].x-pressure)/std::max(1.,std::abs(pressure));
        const double jError=std::abs(out.tangent[3*i+2].x+derivative*.01)/std::max(1.,derivative*.01);
        worst=std::max({worst,pError,jError});require(pError<2e-5&&jError<2e-5,"atan residual/tangent disagrees with independent inverse compliance");
        const double recovered=2/pi*std::atan(pi*.25*pressure/2);
        require(std::abs(recovered-d)<1e-10,"atan inverse lost signed displacement");
    }
    // Canonical stateless Starling extension includes source-unassigned reverse
    // and equality cases explicitly. These are law tests, not calibration.
    for(unsigned law:{2u,3u}) {
        for(auto& n:nodes){n={};n.compliance={20,0,1,0};}edge.identity.w=law;edge.physical={2,0,0,law==3?1.f:0.f};
        const std::array<std::array<float,3>,7> rows={{{3,2,.5},{3,0,1},{0,2,0},{1,2,0},{3,1,1},{1,1,0},{0,-1,0}}};
        candidate.clear();direction.clear();
        for(auto r:rows){float q=std::max(r[0]-(law==3?std::max(r[1],1.f):r[1]),0.f)/2;
            candidate.insert(candidate.end(),{{20+r[0],0,0,0},{20+r[1],0,0,0},{q,0,0,0}});direction.insert(direction.end(),{{.1f,0,0,0},{.3f,0,0,0},{.2f,0,0,0}});}
        out=equations(device,queue,library,nodes,edge,candidate,direction);
        for(unsigned i=0;i<rows.size();++i){require(out.status[i].code==0,"canonical valve state rejected");
            require(std::abs(out.residual[3*i+2].x)<1e-6,"canonical valve failed min-law certificate");
            const double q=candidate[3*i+2].x;
            const double expected=q==0?.2: .4-.1+(law==3&&rows[i][1]<=1?0:.3);
            require(std::abs(out.tangent[3*i+2].x-expected)<2e-6,"canonical valve tangent or floor tie incorrect");}
    }
    std::cout<<"cardiac_regional_laws=pass atan_signed_domain=pass asymptotes=denied pressure_tangent_relative_max="<<worst<<" unilateral_linear=pass starling_reverse_and_floor_ties=pass\n";
}

}
int main(){@autoreleasepool{try{
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"Metal unavailable");
    require([[device name] rangeOfString:@"Apple"].location!=NSNotFound&&[[device name] rangeOfString:@"Paravirtual"].location==NSNotFound,"physical Apple Metal required");
    auto queue=[device newCommandQueue];NSError* error=nil;
    auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_MATTER_METALLIB] error:&error];require(library!=nil,"Matter library unavailable");
    regionalLaws(device,queue,library);waveform(device,queue,library);clockCarry(device,queue,library);supportWorkingSetPivot(device,queue,library);sharedAlpha(device,queue,library);supportConstrainedSharedAlpha(device,queue,library);supportConeIntersection(device,queue,library);supportConeScaleDrift(device,queue,library);supportConeFinalAlphaGate(device,queue,library);supportConeWorkingSetPivot(device,queue,library);supportWorkingSetRelease(device,queue,library);supportMonolithicDeferral(device,queue,library);supportConeCertificate(device,queue,library);deferredDirection(device,queue,library);pendingZeroRelease(device,queue,library);return 0;
}catch(const std::exception& error){std::cerr<<"cardiac_transaction_check=failed reason="<<error.what()<<'\n';return 1;}}}
