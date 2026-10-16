#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/shared.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using F4 = nm_float4;
static_assert(sizeof(F4) == 16, "shared Matter float4 ABI");
static_assert(sizeof(NMFEMNodeStateGPU) == 80, "production candidate-node ABI");

static F4 f4(float x=0.0f,float y=0.0f,float z=0.0f,float w=0.0f) {
    return F4{x,y,z,w};
}
static id<MTLBuffer> bytesBuffer(id<MTLDevice> device,const void* data,NSUInteger size) {
    id<MTLBuffer> b=[device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
    if (!b) throw std::runtime_error("newBufferWithBytes failed");
    return b;
}
static id<MTLBuffer> zeroBuffer(id<MTLDevice> device,NSUInteger size) {
    std::vector<std::uint8_t> z(size,0u); return bytesBuffer(device,z.data(),size);
}
static void poison(id<MTLBuffer> b,std::uint8_t byte=0xA5) {
    std::memset(b.contents,byte,b.length);
}
static void writeBytes(id<MTLBuffer> b,const void* p,std::size_t n) {
    if (n>b.length) throw std::runtime_error("host write exceeds buffer");
    std::memcpy(b.contents,p,n);
}
static id<MTLComputePipelineState> makePipeline(id<MTLDevice>d,id<MTLLibrary>l,const char*n) {
    NSString* fn=[NSString stringWithUTF8String:n];
    id<MTLFunction> f=[l newFunctionWithName:fn];
    if (!f) throw std::runtime_error(std::string("missing Metal function: ")+n);
    NSError* err=nil; id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:f error:&err];
    if(!p) throw std::runtime_error(std::string("pipeline failed: ")+n+" "+(err?[[err localizedDescription] UTF8String]:""));
    return p;
}
static void bindU(id<MTLComputeCommandEncoder> e,std::uint32_t v,NSUInteger i){[e setBytes:&v length:sizeof(v) atIndex:i];}
static void bindF(id<MTLComputeCommandEncoder> e,float v,NSUInteger i){[e setBytes:&v length:sizeof(v) atIndex:i];}
static void bindB(id<MTLComputeCommandEncoder> e,id<MTLBuffer>b,NSUInteger i){[e setBuffer:b offset:0 atIndex:i];}

template<class Binder>
static void dispatchThreads(id<MTLCommandQueue> q,id<MTLComputePipelineState> p,NSUInteger count,Binder bind) {
    id<MTLCommandBuffer> c=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[c computeCommandEncoder];
    [e setComputePipelineState:p]; bind(e);
    [e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e endEncoding]; [c commit]; [c waitUntilCompleted];
    if(c.status==MTLCommandBufferStatusError) throw std::runtime_error([[c.error localizedDescription] UTF8String]);
}
template<class Binder>
static void dispatchGroups(id<MTLCommandQueue> q,id<MTLComputePipelineState> p,NSUInteger groups,Binder bind) {
    id<MTLCommandBuffer> c=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[c computeCommandEncoder];
    [e setComputePipelineState:p]; bind(e);
    [e dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e endEncoding]; [c commit]; [c waitUntilCompleted];
    if(c.status==MTLCommandBufferStatusError) throw std::runtime_error([[c.error localizedDescription] UTF8String]);
}
static bool close(float a,float b,float absTol=2.0e-7f,float relTol=2.0e-5f) {
    return std::fabs(a-b)<=absTol+relTol*std::max(std::fabs(a),std::fabs(b));
}
static void require(bool b,const char*why){if(!b)throw std::runtime_error(why);}
static std::string statusName(std::uint32_t s) {
    switch(s){case 1:return "running";case 2:return "iteration_cap";case 3:return "zero_normal_residual";case 4:return "breakdown";case 5:return "invalid_input";default:return "quiet_or_unknown";}
}
struct EpochResult {
    std::uint32_t pass=0,status=0,iterations=0,error=0;
    std::uint32_t priorStatus=0,priorIterations=0,priorError=0;
    std::array<double,3> priorX{};
    double candidateMass=0,candidateSecondaryMask=0;
    double rhsNorm=0,gradientNorm=0,initialResidualNorm=0,finalResidualNorm=0;
    std::array<double,3> initialB{},initialATB{},initialX{},finalX{};
};

struct Scratch {
    id<MTLBuffer> dispatchBuffer;
    id<MTLBuffer> microstepBuffer;
    id<MTLBuffer> layoutBuffer;
    id<MTLBuffer> solverBuffer;
    id<MTLBuffer> nodeBuffer;
    id<MTLBuffer> stateBuffer;
    id<MTLBuffer> statusBuffer;
    id<MTLBuffer> trialStatusBuffer;
    id<MTLBuffer> lineSearchBuffer;
    id<MTLBuffer> correctionBuffer;
    id<MTLBuffer> environmentLineSearchBuffer;
    id<MTLBuffer> matrixBuffer;
    id<MTLBuffer> metricsBuffer;
    id<MTLBuffer> maskBuffer;
    id<MTLBuffer> rowEnvironmentBuffer;
    id<MTLBuffer> selectedBuffer;
    id<MTLBuffer> baseRhsBuffer;
    id<MTLBuffer> preparedRhsBuffer;
    id<MTLBuffer> atbBuffer;
    id<MTLBuffer> xBuffer;
    id<MTLBuffer> rBuffer;
    id<MTLBuffer> sBuffer;
    id<MTLBuffer> pBuffer;
    id<MTLBuffer> qBuffer;
    id<MTLBuffer> scalarBuffer;
    id<MTLBuffer> controlBuffer;
    id<MTLBuffer> errorBuffer;
    id<MTLBuffer> rightScaleBuffer;
};
static Scratch makeScratch(id<MTLDevice> device) {
    constexpr std::uint32_t E=2u,N=2u,U=8u;
    Scratch b{};
    b.dispatchBuffer=zeroBuffer(device,sizeof(NMMatterDispatchGPU));
    b.microstepBuffer=zeroBuffer(device,sizeof(NMMicrostepGPU));
    b.layoutBuffer=zeroBuffer(device,sizeof(NMFGMRESLayoutGPU));
    b.solverBuffer=zeroBuffer(device,sizeof(NMMixedSolverGPU));
    b.nodeBuffer=zeroBuffer(device,sizeof(NMFEMNodeStateGPU)*E*N);
    b.stateBuffer=zeroBuffer(device,sizeof(NMFGMRESStateGPU)*E);
    b.statusBuffer=zeroBuffer(device,sizeof(NMMatterStatusGPU)*E);
    b.trialStatusBuffer=zeroBuffer(device,sizeof(NMMatterStatusGPU)*E);
    b.lineSearchBuffer=zeroBuffer(device,sizeof(F4)*E);
    b.correctionBuffer=zeroBuffer(device,sizeof(F4)*E*N);
    b.environmentLineSearchBuffer=zeroBuffer(device,sizeof(float)*E);
    b.matrixBuffer=zeroBuffer(device,sizeof(float)*36u);
    b.metricsBuffer=zeroBuffer(device,sizeof(F4)*E);
    b.maskBuffer=zeroBuffer(device,sizeof(F4)*U);
    b.rowEnvironmentBuffer=zeroBuffer(device,sizeof(std::uint32_t)*U);
    b.selectedBuffer=zeroBuffer(device,sizeof(std::uint32_t)*E);
    b.baseRhsBuffer=zeroBuffer(device,sizeof(F4)*U);
    b.preparedRhsBuffer=zeroBuffer(device,sizeof(F4)*U);
    b.atbBuffer=zeroBuffer(device,sizeof(F4)*U);
    b.xBuffer=zeroBuffer(device,sizeof(F4)*U);
    b.rBuffer=zeroBuffer(device,sizeof(F4)*U);
    b.sBuffer=zeroBuffer(device,sizeof(F4)*U);
    b.pBuffer=zeroBuffer(device,sizeof(F4)*U);
    b.qBuffer=zeroBuffer(device,sizeof(F4)*U);
    b.scalarBuffer=zeroBuffer(device,sizeof(F4)*E);
    b.controlBuffer=zeroBuffer(device,sizeof(nm_uint4)*E);
    b.errorBuffer=zeroBuffer(device,sizeof(std::uint32_t)*E);
    b.rightScaleBuffer=zeroBuffer(device,sizeof(F4)*3u*E*N);
    return b;
}

static EpochResult runEpoch(
    id<MTLCommandQueue> queue,
    const std::map<std::string,id<MTLComputePipelineState>>& ps,
    std::uint32_t epoch,std::uint32_t cap,
    const NMMatterDispatchGPU& dispatchABI,
    const NMFGMRESLayoutGPU& layoutABI,
    const NMMixedSolverGPU& solver,
    const std::array<NMFEMNodeStateGPU,4>& nodes,
    const std::array<NMFGMRESStateGPU,2>& states,
    const std::array<NMMatterStatusGPU,2>& statuses,
    const std::array<float,36>& matrices,
    const std::array<F4,8>& baseRhs,
    float mu,Scratch& B,const EpochResult* previous) {
    constexpr std::uint32_t E=2u,N=2u,U=8u;
    auto dispatchBuffer=B.dispatchBuffer; auto microstepBuffer=B.microstepBuffer; auto layoutBuffer=B.layoutBuffer;
    auto solverBuffer=B.solverBuffer; auto nodeBuffer=B.nodeBuffer;
    auto stateBuffer=B.stateBuffer; auto statusBuffer=B.statusBuffer;
    auto trialStatusBuffer=B.trialStatusBuffer; auto lineSearchBuffer=B.lineSearchBuffer;
    auto correctionBuffer=B.correctionBuffer; auto environmentLineSearchBuffer=B.environmentLineSearchBuffer;
    auto matrixBuffer=B.matrixBuffer; auto metricsBuffer=B.metricsBuffer;
    auto maskBuffer=B.maskBuffer; auto rowEnvironmentBuffer=B.rowEnvironmentBuffer;
    auto selectedBuffer=B.selectedBuffer; auto baseRhsBuffer=B.baseRhsBuffer;
    auto preparedRhsBuffer=B.preparedRhsBuffer; auto atbBuffer=B.atbBuffer;
    auto xBuffer=B.xBuffer; auto rBuffer=B.rBuffer; auto sBuffer=B.sBuffer;
    auto pBuffer=B.pBuffer; auto qBuffer=B.qBuffer; auto scalarBuffer=B.scalarBuffer;
    auto controlBuffer=B.controlBuffer; auto errorBuffer=B.errorBuffer;
    auto rightScaleBuffer=B.rightScaleBuffer;
    writeBytes(dispatchBuffer,&dispatchABI,sizeof(dispatchABI));
    NMMicrostepGPU microstep{}; microstep.solverIteration=epoch+34u;
    writeBytes(microstepBuffer,&microstep,sizeof(microstep));
    writeBytes(layoutBuffer,&layoutABI,sizeof(layoutABI));
    writeBytes(solverBuffer,&solver,sizeof(solver));
    writeBytes(nodeBuffer,nodes.data(),sizeof(nodes));
    writeBytes(stateBuffer,states.data(),sizeof(states));
    writeBytes(statusBuffer,statuses.data(),sizeof(statuses));
    writeBytes(trialStatusBuffer,statuses.data(),sizeof(statuses));
    writeBytes(matrixBuffer,matrices.data(),sizeof(matrices));
    writeBytes(baseRhsBuffer,baseRhs.data(),sizeof(baseRhs));

    EpochResult out; out.pass=epoch;
    // Rebuild layout scratch every pass. On pass zero also poison private
    // solver state. On pass one leave the previous terminal CGLS state intact
    // until the production state/vector initializers consume it.
    for(auto b:{metricsBuffer,maskBuffer,rowEnvironmentBuffer,selectedBuffer,
                preparedRhsBuffer,atbBuffer}) poison(b);
    if(epoch==0u) {
        for(auto b:{xBuffer,rBuffer,sBuffer,pBuffer,qBuffer,scalarBuffer,
                    controlBuffer,errorBuffer}) poison(b);
    } else {
        require(previous!=nullptr,"missing previous solve for repeated initialization");
        const auto* oldControl=static_cast<const nm_uint4*>(controlBuffer.contents);
        const auto* oldX=static_cast<const F4*>(xBuffer.contents);
        const auto* oldError=static_cast<const std::uint32_t*>(errorBuffer.contents);
        require(oldControl[0].x==previous->status &&
                oldControl[0].y==previous->iterations &&
                oldControl[1].x==0u && oldError[0]==previous->error && oldError[1]==0u,
                "shared CGLS state no longer equals the previous pass terminal state");
        require(close(oldX[0].x,static_cast<float>(previous->finalX[0])) &&
                close(oldX[0].y,static_cast<float>(previous->finalX[1])) &&
                close(oldX[0].z,static_cast<float>(previous->finalX[2])),
                "second pass did not begin with the prior terminal solution in reused scratch");
        out.priorStatus=oldControl[0].x;
        out.priorIterations=oldControl[0].y;
        out.priorError=oldError[0];
        out.priorX={oldX[0].x,oldX[0].y,oldX[0].z};
        require(out.priorX!=std::array<double,3>{0,0,0},
                "first synthetic solve did not leave a nonzero solution to detect warm transfer");
    }
    std::array<F4,E> metrics{f4(1.0e-2f,0,1.0f,0),f4(5.0e-8f,4.0e-8f,0.0f,0.0f)};
    writeBytes(metricsBuffer,metrics.data(),sizeof(metrics));
    std::array<F4,E> lines{f4(1,0,0,0),f4(1,0,0,0)};
    std::array<float,E> envAlpha{1.0f,1.0f};
    writeBytes(lineSearchBuffer,lines.data(),sizeof(lines));
    writeBytes(environmentLineSearchBuffer,envAlpha.data(),sizeof(envAlpha));
    poison(correctionBuffer,0u);
    std::uint32_t trialIndex=23u, fallbackAllowed=1u;
    auto armijoBind=[&](id<MTLComputeCommandEncoder> e){
        bindB(e,dispatchBuffer,0);bindB(e,microstepBuffer,1);bindB(e,solverBuffer,2);
        bindB(e,metricsBuffer,3);bindB(e,lineSearchBuffer,4);bindB(e,trialStatusBuffer,5);
        bindB(e,statusBuffer,6);bindU(e,trialIndex,7);bindB(e,stateBuffer,8);
        bindB(e,correctionBuffer,9);bindB(e,environmentLineSearchBuffer,10);bindU(e,fallbackAllowed,11);
    };
    // A failed private trial at terminal exhaustion must remain a failure;
    // it must not arm the damped direction.
    auto unsafeTrial=statuses;
    unsafeTrial[0].code=NM_STATUS_NONLINEAR_SOLVER_FAILURE;
    writeBytes(trialStatusBuffer,unsafeTrial.data(),sizeof(unsafeTrial));
    metrics[0]=f4(1.0e-2f,1.01e-2f,1.0f/8388608.0f,0.0f);
    metrics[1]=f4(5.0e-8f,4.0e-8f,0.0f,2.0f);
    writeBytes(metricsBuffer,metrics.data(),sizeof(metrics));
    dispatchThreads(queue,ps.at("numi_matter_metal::nm_fem_armijo_decide"),E,armijoBind);
    auto* unsafeDecision=static_cast<F4*>(metricsBuffer.contents);
    auto* unsafeStatuses=static_cast<NMMatterStatusGPU*>(statusBuffer.contents);
    require(unsafeDecision[0].w!=3.0f &&
            unsafeStatuses[0].code==NM_STATUS_NONLINEAR_SOLVER_FAILURE,
            "unsafe exhausted private trial was incorrectly converted to fallback");
    writeBytes(statusBuffer,statuses.data(),sizeof(statuses));
    writeBytes(trialStatusBuffer,statuses.data(),sizeof(statuses));
    trialIndex=0u;
    metrics[0]=f4(1.0e-2f,8.0e-3f,1.0f,0.0f);
    metrics[1]=f4(5.0e-8f,4.0e-8f,0.0f,0.0f);
    writeBytes(metricsBuffer,metrics.data(),sizeof(metrics));
    // Actual production accepted and certified-noop branches.
    dispatchThreads(queue,ps.at("numi_matter_metal::nm_fem_armijo_decide"),E,armijoBind);
    const auto* decision=static_cast<const F4*>(metricsBuffer.contents);
    require(decision[0].w==1.0f && decision[1].w==2.0f,
            "production Armijo accepted/noop controls failed");
    metrics[0]=f4(1.0e-2f,1.01e-2f,1.0f/8388608.0f,0.0f);
    metrics[1]=decision[1];
    writeBytes(metricsBuffer,metrics.data(),sizeof(metrics));
    trialIndex=23u;
    dispatchThreads(queue,ps.at("numi_matter_metal::nm_fem_armijo_decide"),E,armijoBind);
    decision=static_cast<const F4*>(metricsBuffer.contents);
    require(decision[0].w==3.0f && decision[1].w==2.0f,
            "healthy exhausted Armijo did not select exactly env0");
    require(static_cast<const float*>(environmentLineSearchBuffer.contents)[0]==0.0f &&
            static_cast<const float*>(environmentLineSearchBuffer.contents)[1]==1.0f,
            "Armijo fallback alpha publication mismatch");
    require(static_cast<const NMMatterStatusGPU*>(statusBuffer.contents)[0].code==NM_STATUS_SUCCESS,
            "healthy exhausted Armijo changed environment status");
    dispatchThreads(queue,ps.at("numi_matter_metal::nm_fgmres_prepare_base_cgls_layout"),U,
        [&](id<MTLComputeCommandEncoder> e){
            bindB(e,dispatchBuffer,0);bindB(e,layoutBuffer,30);bindB(e,nodeBuffer,1);
            bindB(e,metricsBuffer,2);bindB(e,maskBuffer,3);bindB(e,rowEnvironmentBuffer,4);
            bindB(e,selectedBuffer,5);bindB(e,baseRhsBuffer,6);bindB(e,preparedRhsBuffer,7);
        });
    const auto* selected=static_cast<const std::uint32_t*>(selectedBuffer.contents);
    require(selected[0]==1u && selected[1]==0u,"production layout did not derive selected/quiet envs");
    const auto* rowEnv=static_cast<const std::uint32_t*>(rowEnvironmentBuffer.contents);
    const std::uint32_t expectedRows[U]={0u,0u,1u,1u,0u,0u,1u,1u};
    for(std::uint32_t i=0;i<U;++i)
        require(rowEnv[i]==expectedRows[i],"production layout row-environment offsets mismatch");
    const auto* mask=static_cast<const F4*>(maskBuffer.contents);
    require(mask[0].x==1.0f && mask[0].y==1.0f && mask[0].z==1.0f && mask[0].w==0.0f,
            "active candidate node was not free in production layout");
    const float expectedSecondaryMask=epoch==0u?1.0f:0.0f;
    require(mask[1].x==expectedSecondaryMask && mask[1].y==expectedSecondaryMask &&
            mask[1].z==expectedSecondaryMask && mask[1].w==0.0f,
            "second candidate node did not follow changed current candidate constraint");
    for(std::uint32_t i=2u;i<U;++i)
        require(mask[i].x==0.0f && mask[i].y==0.0f && mask[i].z==0.0f && mask[i].w==0.0f,
                "quiet environment or field row mask was nonzero");
    const auto* prepared=static_cast<const F4*>(preparedRhsBuffer.contents);
    require(close(prepared[0].x,baseRhs[0].x) && close(prepared[0].y,baseRhs[0].y) &&
            close(prepared[0].z,baseRhs[0].z),
            "production layout changed active RHS");
    for(std::uint32_t i=1u;i<U;++i)
        require(prepared[i].x==0.0f && prepared[i].y==0.0f && prepared[i].z==0.0f && prepared[i].w==0.0f,
                "production layout failed to suppress ineligible or zero-field RHS rows");

    const std::uint32_t transpose=1u;
    dispatchThreads(queue,ps.at("reinit_fixture_apply"),U,
        [&](id<MTLComputeCommandEncoder> e){
            bindB(e,matrixBuffer,0);bindB(e,preparedRhsBuffer,1);bindB(e,atbBuffer,2);
            bindU(e,U,3);bindU(e,N,4);bindU(e,transpose,5);
        });
    const std::uint32_t useRightScale=0u;
    dispatchThreads(queue,ps.at("numi_matter_metal::nm_fgmres_base_cgls_initialize_state"),E,
        [&](id<MTLComputeCommandEncoder> e){
            bindU(e,E,0);bindU(e,cap,1);bindB(e,selectedBuffer,2);
            bindB(e,scalarBuffer,3);bindB(e,controlBuffer,4);bindB(e,errorBuffer,5);
        });
    dispatchThreads(queue,ps.at("numi_matter_metal::nm_fgmres_base_cgls_initialize_vectors"),U,
        [&](id<MTLComputeCommandEncoder> e){
            bindU(e,U,0);bindU(e,E,1);bindB(e,maskBuffer,2);bindB(e,rowEnvironmentBuffer,3);
            bindB(e,selectedBuffer,4);bindB(e,preparedRhsBuffer,5);bindB(e,atbBuffer,6);
            bindB(e,xBuffer,7);bindB(e,rBuffer,8);bindB(e,sBuffer,9);bindB(e,pBuffer,10);
            bindB(e,qBuffer,11);bindB(e,errorBuffer,12);bindU(e,useRightScale,13);bindB(e,rightScaleBuffer,14);
        });
    const auto* ctl0=static_cast<const nm_uint4*>(controlBuffer.contents);
    const auto* scal0=static_cast<const F4*>(scalarBuffer.contents);
    const auto* err0=static_cast<const std::uint32_t*>(errorBuffer.contents);
    const auto* x0=static_cast<const F4*>(xBuffer.contents);
    const auto* r0=static_cast<const F4*>(rBuffer.contents);
    const auto* s0=static_cast<const F4*>(sBuffer.contents);
    const auto* p0=static_cast<const F4*>(pBuffer.contents);
    const auto* q0=static_cast<const F4*>(qBuffer.contents);
    require(ctl0[0].x==1u && ctl0[0].y==0u && ctl0[0].z==cap && ctl0[0].w==1u,
            "active production CGLS state was not reinitialized");
    require(ctl0[1].x==0u && ctl0[1].y==0u && ctl0[1].z==cap && ctl0[1].w==0u,
            "quiet production CGLS state was not reset");
    require(err0[0]==0u && err0[1]==0u,"production initializer retained/created error bits");
    require(scal0[0].x==0.0f && scal0[0].y==0.0f && scal0[0].z==0.0f && scal0[0].w==0.0f &&
            scal0[1].x==0.0f && scal0[1].y==0.0f && scal0[1].z==0.0f && scal0[1].w==0.0f,
            "production CGLS scalar state did not reset before measurement");
    for(std::uint32_t i=0;i<U;++i) {
        require(x0[i].x==0.0f && x0[i].y==0.0f && x0[i].z==0.0f && x0[i].w==0.0f,
                "production x did not reset to zero");
        require(q0[i].x==0.0f && q0[i].y==0.0f && q0[i].z==0.0f && q0[i].w==0.0f,
                "production q did not reset to zero");
    }
    require(close(r0[0].x,prepared[0].x) && close(r0[0].y,prepared[0].y) &&
            close(r0[0].z,prepared[0].z),"production residual did not reload current RHS");
    require(close(s0[0].x,static_cast<const F4*>(atbBuffer.contents)[0].x) &&
            close(s0[0].y,static_cast<const F4*>(atbBuffer.contents)[0].y) &&
            close(s0[0].z,static_cast<const F4*>(atbBuffer.contents)[0].z) &&
            close(p0[0].x,s0[0].x) && close(p0[0].y,s0[0].y) && close(p0[0].z,s0[0].z),
            "production gradient/search direction did not reload current A^T b");
    for(std::uint32_t i=1;i<U;++i)
        require(r0[i].x==0 && r0[i].y==0 && r0[i].z==0 &&
                s0[i].x==0 && s0[i].y==0 && s0[i].z==0 &&
                p0[i].x==0 && p0[i].y==0 && p0[i].z==0,
                "quiet env or field rows inherited CGLS vectors");

    dispatchGroups(queue,ps.at("numi_matter_metal::nm_fgmres_base_cgls_measure_initial"),E,
        [&](id<MTLComputeCommandEncoder> e){
            bindU(e,U,0);bindU(e,E,1);bindB(e,maskBuffer,2);bindB(e,rowEnvironmentBuffer,3);
            bindB(e,preparedRhsBuffer,4);bindB(e,rBuffer,5);bindB(e,sBuffer,6);
            bindB(e,scalarBuffer,7);bindB(e,controlBuffer,8);bindB(e,errorBuffer,9);
        });

    const auto* scalMeasured=static_cast<const F4*>(scalarBuffer.contents);
    require(scalMeasured[0].x>0.0f && scalMeasured[0].w>0.0f &&
            scalMeasured[1].x==0.0f && scalMeasured[1].y==0.0f &&
            scalMeasured[1].z==0.0f && scalMeasured[1].w==0.0f,
            "production measurement did not initialize active scalar state and preserve quiet state");
    require((epoch==0u && mask[1].x==1.0f) || (epoch==1u && mask[1].x==0.0f),
            "candidate constraint transition was not observed");

    out.candidateMass=nodes[0].positionAndMass.w;
    out.candidateSecondaryMask=mask[1].x;
    out.initialB={prepared[0].x,prepared[0].y,prepared[0].z};
    const F4 firstAtb=static_cast<const F4*>(atbBuffer.contents)[0];
    out.initialATB={firstAtb.x,firstAtb.y,firstAtb.z};
    out.initialX={x0[0].x,x0[0].y,x0[0].z};
    out.initialResidualNorm=std::sqrt(double(r0[0].x)*r0[0].x+double(r0[0].y)*r0[0].y+double(r0[0].z)*r0[0].z);
    out.gradientNorm=std::sqrt(double(s0[0].x)*s0[0].x+double(s0[0].y)*s0[0].y+double(s0[0].z)*s0[0].z);
    out.rhsNorm=out.initialResidualNorm;

    const float dampingSquared=mu*mu;
    for(std::uint32_t step=0;step<cap;++step) {
        const std::uint32_t forward=0u;
        dispatchThreads(queue,ps.at("reinit_fixture_apply"),U,
            [&](id<MTLComputeCommandEncoder> e){
                bindB(e,matrixBuffer,0);bindB(e,pBuffer,1);bindB(e,qBuffer,2);
                bindU(e,U,3);bindU(e,N,4);bindU(e,forward,5);
            });
        dispatchGroups(queue,ps.at("numi_matter_metal::nm_fgmres_base_cgls_form_alpha"),E,
            [&](id<MTLComputeCommandEncoder> e){
                bindU(e,U,0);bindU(e,E,1);bindB(e,maskBuffer,2);bindB(e,rowEnvironmentBuffer,3);
                bindB(e,qBuffer,4);bindB(e,scalarBuffer,5);bindB(e,controlBuffer,6);
                bindB(e,errorBuffer,7);bindB(e,pBuffer,8);bindF(e,dampingSquared,9);
            });
        dispatchThreads(queue,ps.at("numi_matter_metal::nm_fgmres_base_cgls_update_xr"),U,
            [&](id<MTLComputeCommandEncoder> e){
                bindU(e,U,0);bindU(e,E,1);bindB(e,maskBuffer,2);bindB(e,rowEnvironmentBuffer,3);
                bindB(e,scalarBuffer,4);bindB(e,controlBuffer,5);bindB(e,pBuffer,6);bindB(e,qBuffer,7);
                bindB(e,xBuffer,8);bindB(e,rBuffer,9);bindB(e,errorBuffer,10);
            });
        dispatchThreads(queue,ps.at("numi_matter_metal::nm_fgmres_base_cgls_complete_iteration"),E,
            [&](id<MTLComputeCommandEncoder> e){bindU(e,E,0);bindB(e,controlBuffer,1);bindB(e,errorBuffer,2);});
        const std::uint32_t transposeBack=1u;
        dispatchThreads(queue,ps.at("reinit_fixture_apply"),U,
            [&](id<MTLComputeCommandEncoder> e){
                bindB(e,matrixBuffer,0);bindB(e,rBuffer,1);bindB(e,sBuffer,2);
                bindU(e,U,3);bindU(e,N,4);bindU(e,transposeBack,5);
            });
        dispatchGroups(queue,ps.at("numi_matter_metal::nm_fgmres_base_cgls_form_beta"),E,
            [&](id<MTLComputeCommandEncoder> e){
                bindU(e,U,0);bindU(e,E,1);bindB(e,maskBuffer,2);bindB(e,rowEnvironmentBuffer,3);
                bindB(e,sBuffer,4);bindB(e,scalarBuffer,5);bindB(e,controlBuffer,6);
                bindB(e,errorBuffer,7);bindB(e,xBuffer,8);bindF(e,dampingSquared,9);
            });
        const auto* controlNow=static_cast<const nm_uint4*>(controlBuffer.contents);
        if(controlNow[0].x!=1u) break;
        dispatchThreads(queue,ps.at("numi_matter_metal::nm_fgmres_base_cgls_update_p"),U,
            [&](id<MTLComputeCommandEncoder> e){
                bindU(e,U,0);bindU(e,E,1);bindB(e,maskBuffer,2);bindB(e,rowEnvironmentBuffer,3);
                bindB(e,scalarBuffer,4);bindB(e,controlBuffer,5);bindB(e,sBuffer,6);bindB(e,pBuffer,7);bindB(e,errorBuffer,8);
            });
    }
    const auto* ctlFinal=static_cast<const nm_uint4*>(controlBuffer.contents);
    const auto* errorsFinal=static_cast<const std::uint32_t*>(errorBuffer.contents);
    const auto* finalX=static_cast<const F4*>(xBuffer.contents);
    const auto* finalR=static_cast<const F4*>(rBuffer.contents);
    out.status=ctlFinal[0].x;out.iterations=ctlFinal[0].y;out.error=errorsFinal[0];
    out.finalX={finalX[0].x,finalX[0].y,finalX[0].z};
    out.finalResidualNorm=std::sqrt(double(finalR[0].x)*finalR[0].x+double(finalR[0].y)*finalR[0].y+double(finalR[0].z)*finalR[0].z);
    require((out.status==2u||out.status==3u||(out.status==4u&&out.error==32u))&&out.iterations<=cap,
            "production recurrence did not end in a narrowly accepted terminal state");
    require(ctlFinal[1].x==0u&&ctlFinal[1].y==0u&&ctlFinal[1].z==cap&&ctlFinal[1].w==0u&&errorsFinal[1]==0u,
            "quiet environment status changed during recurrence");
    for(std::uint32_t i=1;i<U;++i) {
        require(finalX[i].x==0&&finalX[i].y==0&&finalX[i].z==0&&
                finalR[i].x==0&&finalR[i].y==0&&finalR[i].z==0,
                "quiet environment/field state changed during recurrence");
    }
    return out;
}

int main(int argc,char**argv) {
    @autoreleasepool {
        try {
            if(argc!=2) {std::fprintf(stderr,"usage: reinit_replay <metallib>\n");return 2;}
            id<MTLDevice> device=MTLCreateSystemDefaultDevice();if(!device)throw std::runtime_error("no Metal device");
            id<MTLCommandQueue> queue=[device newCommandQueue];if(!queue)throw std::runtime_error("no command queue");
            NSURL* url=[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
            NSError* libError=nil;id<MTLLibrary> lib=[device newLibraryWithURL:url error:&libError];
            if(!lib)throw std::runtime_error(std::string("metallib load failed: ")+(libError?[[libError localizedDescription] UTF8String]:""));
            const char* names[]={
                "reinit_fixture_apply",
                "numi_matter_metal::nm_fem_armijo_decide",
                "numi_matter_metal::nm_fgmres_prepare_base_cgls_layout",
                "numi_matter_metal::nm_fgmres_base_cgls_initialize_state",
                "numi_matter_metal::nm_fgmres_base_cgls_initialize_vectors",
                "numi_matter_metal::nm_fgmres_base_cgls_measure_initial",
                "numi_matter_metal::nm_fgmres_base_cgls_form_alpha",
                "numi_matter_metal::nm_fgmres_base_cgls_update_xr",
                "numi_matter_metal::nm_fgmres_base_cgls_complete_iteration",
                "numi_matter_metal::nm_fgmres_base_cgls_form_beta",
                "numi_matter_metal::nm_fgmres_base_cgls_update_p"};
            std::map<std::string,id<MTLComputePipelineState>> ps;
            for(const char*n:names)ps.emplace(n,makePipeline(device,lib,n));

            constexpr std::uint32_t E=2,N=2,U=8,cap=32;
            NMMatterDispatchGPU d{};d.abiVersion=NM_MATTER_ABI_VERSION;d.environmentCount=E;d.objectCount=1;d.femNodeCount=N;
            NMFGMRESLayoutGPU layout{};layout.unknownCount=U;
            NMMixedSolverGPU solver{};solver.residualTolerances=f4(1.0e-7f,0,0,0);solver.nonlinearIterations.w=24u;solver.globalization.x=0.1f;

            std::array<NMFEMNodeStateGPU,4> nodes{};
            std::array<NMFGMRESStateGPU,2> states{};
            std::array<NMMatterStatusGPU,2> statuses{};
            std::array<float,36> matrices{
                2.0e-3f,-0.75e-3f,0.0f, 0.5e-3f,3.0e-3f,0.0f, 0.0f,0.0f,1.25e-3f,
                1.8e-3f,0.15e-3f,0.0f, -0.2e-3f,2.6e-3f,0.0f, 0.0f,0.0f,1.3e-3f,
                1.6e-3f,0.2e-3f,0.0f, -0.3e-3f,2.7e-3f,0.0f, 0.0f,0.0f,1.1e-3f,
                1.2e-3f,-0.1e-3f,0.0f, 0.4e-3f,2.2e-3f,0.0f, 0.0f,0.0f,0.9e-3f};
            std::array<F4,8> rhs1{f4(1.25e-4f,-2.0e-4f,0.75e-4f),f4(),
                f4(3e-4f,-1e-4f,2e-4f),f4(-1e-4f,2e-4f,1e-4f),
                f4(),f4(),f4(),f4()};
            std::array<F4,8> rhs2{f4(-2.25e-4f,1.5e-4f,-1.0e-4f),f4(),
                f4(-4e-4f,1.5e-4f,2e-4f),f4(2e-4f,-2e-4f,1e-4f),
                f4(),f4(),f4(),f4()};
            std::array<EpochResult,2> results{};
            Scratch scratch=makeScratch(device);
            for(std::uint32_t epoch=0;epoch<2;++epoch) {
                // Candidate geometry/mass changes at the second solve. The
                // production layout consumes these exact candidate records.
                nodes[0].positionAndMass=f4(0.01f*epoch,-0.02f*epoch,0.03f*epoch,2.0f+epoch);
                nodes[0].velocityAndInverseMass=f4();
                nodes[0].restAndFixed=f4(0,0,0,0);
                nodes[0].deltaVelocity=f4();
                nodes[0].referenceDisplacementAndMode=f4();
                nodes[1].positionAndMass=f4(-0.02f*epoch,0.01f*epoch,0,3.0f);
                nodes[1].velocityAndInverseMass=f4();
                nodes[1].restAndFixed=f4(0,0,0,epoch==0u?0.0f:1.0f);
                nodes[1].deltaVelocity=f4();
                nodes[1].referenceDisplacementAndMode=f4();
                nodes[2].positionAndMass=f4(0,0,0,4.0f);
                nodes[2].velocityAndInverseMass=f4();
                nodes[2].restAndFixed=f4(0,0,0,0);
                nodes[2].deltaVelocity=f4();
                nodes[2].referenceDisplacementAndMode=f4();
                nodes[3].positionAndMass=f4(0,0,0,5.0f);
                nodes[3].velocityAndInverseMass=f4();
                nodes[3].restAndFixed=f4(0,0,0,0);
                nodes[3].deltaVelocity=f4();
                nodes[3].referenceDisplacementAndMode=f4();
                states[0].diagnostics=f4(1,1,1,0);states[0].nonlinear=f4();
                states[1].diagnostics=f4(1,1,0,0);states[1].nonlinear=f4();
                statuses[0]={};statuses[0].code=NM_STATUS_SUCCESS;statuses[0].environment=0;
                statuses[1]={};statuses[1].code=NM_STATUS_SUCCESS;statuses[1].environment=1;
                if(epoch==1) for(int i=0;i<9;++i) matrices[i]*=1.25f;
                results[epoch]=runEpoch(queue,ps,epoch,cap,d,layout,solver,nodes,states,statuses,matrices,epoch==0?rhs1:rhs2,1.0e-3f,scratch,epoch==0?nullptr:&results[epoch-1]);
            }
            require(results[0].initialB!=results[1].initialB,"successive Newton pass did not use changed RHS");
            require(results[0].initialATB!=results[1].initialATB,"successive Newton pass did not recompute changed A^T b");
            require(results[0].initialX==std::array<double,3>{0,0,0}&&results[1].initialX==std::array<double,3>{0,0,0},
                    "Newton pass inherited prior CGLS solution");
            require(results[0].finalX!=std::array<double,3>{0,0,0} &&
                    results[1].priorX==results[0].finalX &&
                    results[1].priorStatus==results[0].status &&
                    results[1].priorError==results[0].error &&
                    results[1].initialX==std::array<double,3>{0,0,0},
                    "second pass did not reset the reused prior terminal solution/state");
            require(results[0].candidateMass!=results[1].candidateMass &&
                    results[0].candidateSecondaryMask!=results[1].candidateSecondaryMask,
                    "current candidate did not change across repeated solve passes");
            std::cout<<std::setprecision(10)
              <<"{\"schema\":\"numi.cardboard.armijo-cgls-fallback-helper.v1\","
              <<"\"native_kernel_execution\":true,\"gpu_dispatch_performed\":true,\"physical_fem_transactions\":0,"
              <<"\"environment_count\":2,\"unknown_count\":8,\"selected_environment_words\":[1,0],\"row_environment\":[0,0,1,1,0,0,1,1],"
              <<"\"scope\":\"executed native Metal helper: actual production Armijo decision, runtime layout builder, initialization and CGLS recurrence; synthetic block-diagonal A/A^T; no FEM operator or full Newton/coupon qualification\","
              <<"\"passes\":[";
            for(std::size_t i=0;i<results.size();++i) {
                if(i)std::cout<<",";
                const auto&r=results[i];
                std::cout<<"{\"pass\":"<<r.pass<<",\"candidate_mass\":"<<r.candidateMass<<",\"candidate_secondary_mask\":"<<r.candidateSecondaryMask<<",\"prior_terminal_status\":"<<r.priorStatus<<",\"prior_terminal_iterations\":"<<r.priorIterations<<",\"prior_error_bits\":"<<r.priorError<<",\"prior_terminal_x\":["<<r.priorX[0]<<","<<r.priorX[1]<<","<<r.priorX[2]<<"],\"active_status\":\""<<statusName(r.status)<<"\",\"active_iterations\":"<<r.iterations
                 <<",\"active_error_bits\":"<<r.error<<",\"initial_b\":["<<r.initialB[0]<<","<<r.initialB[1]<<","<<r.initialB[2]<<"]"
                 <<",\"initial_AT_b\":["<<r.initialATB[0]<<","<<r.initialATB[1]<<","<<r.initialATB[2]<<"]"
                 <<",\"initial_x\":["<<r.initialX[0]<<","<<r.initialX[1]<<","<<r.initialX[2]<<"]"
                 <<",\"initial_residual_norm\":"<<r.initialResidualNorm<<",\"initial_gradient_norm\":"<<r.gradientNorm
                 <<",\"final_x\":["<<r.finalX[0]<<","<<r.finalX[1]<<","<<r.finalX[2]<<"],\"final_residual_norm\":"<<r.finalResidualNorm<<"}";
            }
            std::cout<<"],\"quiet_environment\":{\"selected\":false,\"private_vectors_zero_after_each_init_and_recurrence\":true,\"status_zero\":true},"
             <<"\"production_kernels_bound\":[\"fem_armijo_decide\",\"prepare_base_cgls_layout\",\"initialize_state\",\"initialize_vectors\",\"measure_initial\",\"form_alpha\",\"update_xr\",\"complete_iteration\",\"form_beta\",\"update_p\"],"
             <<"\"full_coupon_qualified\":false}\n";
        } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
    }
}
