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
void sharedAlpha(id<MTLDevice> device,id<MTLCommandQueue> queue,id<MTLLibrary> library) {
    for(unsigned mode=0;mode<3;++mode) {
        NMMatterDispatchGPU dispatch{};dispatch.environmentCount=2;dispatch.rigidGeneralizedCapacity=1;dispatch.objectCount=mode==2?1:0;
        NMFGMRESLayoutGPU layout{};layout.supportContactCount=1;layout.supportBase=2;layout.vascularBase=4;layout.vascularUnknownCount=mode==1?0:1;
        NMHumanSupportDispatchGPU support{};support.contactCount=1;support.groundNormal={0,0,1,0};
        const std::vector<nm_float4> initial(2,nm_float4{1,2,3,4});
        auto candidate=buffer(device,std::vector<float>{2,2});auto histories=buffer(device,initial);
        auto solution=buffer(device,std::vector<nm_float4>{{8,0,0,0},{8,0,0,0},{8,12,16,0},{8,12,16,0}});
        auto alpha=buffer(device,std::vector<float>{0,.25});auto lines=buffer(device,std::vector<nm_float4>{{.5,0,0,0},{.75,0,0,0}});
        auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
        [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_rigid_apply_candidate_solution")];
        [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];
        [e setBuffer:solution offset:0 atIndex:1];[e setBuffer:lines offset:0 atIndex:2];[e setBuffer:candidate offset:0 atIndex:3];[e setBuffer:alpha offset:0 atIndex:4];
        [e dispatchThreads:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(2,1,1)];
        [e setComputePipelineState:pipeline(device,library,@"numi_matter_metal::nm_human_support_apply_solution")];
        [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBytes:&layout length:sizeof(layout) atIndex:30];[e setBytes:&support length:sizeof(support) atIndex:1];
        [e setBuffer:solution offset:0 atIndex:2];[e setBuffer:lines offset:0 atIndex:3];[e setBuffer:histories offset:0 atIndex:4];[e setBuffer:alpha offset:0 atIndex:5];
        [e dispatchThreads:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(2,1,1)];[e endEncoding];complete(cb);
        const auto* out=static_cast<const float*>(candidate.contents);const auto* h=static_cast<const nm_float4*>(histories.contents);
        for(unsigned env=0;env<2;++env) {
            const float a=mode==0?(env==0?0.f:.25f):(mode==1?1.f:(env==0?.5f:.75f));
            require(out[env]==2+8*a,"rigid candidate bypassed shared vascular alpha");
            if(a==0)require(std::memcmp(h+env,initial.data()+env,sizeof(nm_float4))==0,"deferred support history changed");
            else require(h[env].x==1+8*a&&h[env].y==2+12*a&&h[env].z==0&&h[env].w==7+16*a,"support impulse bypassed shared alpha");
        }
    }
    std::cout<<"cardiac_shared_alpha=pass objectless_rigid_support=pass deferral=exact fractional=pass nonvascular_baseline=pass object_baseline=pass\n";
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
    regionalLaws(device,queue,library);waveform(device,queue,library);clockCarry(device,queue,library);sharedAlpha(device,queue,library);deferredDirection(device,queue,library);pendingZeroRelease(device,queue,library);return 0;
}catch(const std::exception& error){std::cerr<<"cardiac_transaction_check=failed reason="<<error.what()<<'\n';return 1;}}}
