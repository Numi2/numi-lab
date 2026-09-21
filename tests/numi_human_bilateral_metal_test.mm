#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/compensated_translation_gpu.h"
#include "metalrobo/numi_human_tendon_gpu.h"
#include "metalrobo/numi_human_joint_equality_gpu.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

// Production kernel, four coincident spherical-inertia bodies, three coaxial
// sibling joints and two coupled equality rows. Independent 2x2 reduced-mass
// oracle; no production Schur/factor helper is used by the expected result.
namespace {
constexpr unsigned nv=9,nq=10,bodies=4,points=16,neq=2,envs=2;
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
using V3=std::array<float,3>;
V3 cross(V3 a,V3 b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
unsigned exercise(id<MTLDevice> device,id<MTLComputePipelineState> pipeline,
                  id<MTLCommandQueue> queue,float h,unsigned sweeps,bool reverse,bool loaded,unsigned limitedDof) {
    const std::array<std::size_t,26> sizes={
        sizeof(MRWorldGPU),sizeof(MRArticulationGPU),nv*sizeof(MRDofPropertiesGPU),
        bodies*sizeof(MRBodyPropertiesGPU),sizeof(MRNumiHumanStandDispatchGPU),
        envs*nq*sizeof(float),envs*nv*sizeof(float),envs*bodies*sizeof(MRArticulatedBodyPoseGPU),
        envs*points*sizeof(MRArticulatedPointWorldGPU),envs*points*3*nv*sizeof(float),envs*nv*sizeof(float),
        sizeof(MRNumiHumanStandContactGPU),envs*bodies*MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS*nv*sizeof(float),envs*bodies*2*sizeof(mr_float4),
        envs*nv*nv*sizeof(float),envs*(4*nv+neq)*sizeof(float),
        envs*((neq+nv)*nv+neq*(neq+3+nv))*sizeof(float),envs*sizeof(MRNumiHumanStandStatusGPU),
        sizeof(MRNumiHumanTendonBindingGPU),sizeof(MRNumiHumanTendonTransferResultGPU),
        neq*sizeof(MRNumiHumanJointEqualityGPU),envs*sizeof(MRCompensatedRootTranslationGPU),
        envs*bodies*sizeof(mr_float4),envs*points*sizeof(mr_float4),nv*(nv+1)*sizeof(float),
        envs*3u*nv*sizeof(float)};
    std::array<id<MTLBuffer>,26> b;
    for(unsigned i=0;i<b.size();++i) {
        b[i]=[device newBufferWithLength:sizes[i] options:MTLResourceStorageModeShared];
        require(b[i]!=nil,"allocation failed");std::memset(b[i].contents,0,sizes[i]);
    }
    auto* world=static_cast<MRWorldGPU*>(b[0].contents);
    world->abiVersion=MR_ENGINE_ABI_VERSION;world->bodyCount=bodies;world->articulationCount=1;
    world->nq=nq;world->nv=nv;world->gravityAndTimestep={0,0,0,h};
    auto* art=static_cast<MRArticulationGPU*>(b[1].contents);
    art->bodyCount=bodies;art->rootType=MR_ROOT_FLOATING;art->nq=nq;art->nv=nv;
    auto* dofs=static_cast<MRDofPropertiesGPU*>(b[2].contents);
    for(unsigned d=0;d<nv;++d) {dofs[d].vIndex=d;dofs[d].qIndex=d<3?d:d<6?MR_INVALID_INDEX:d+1;}
    if (limitedDof != MR_INVALID_INDEX) {
        // Finite-range stop, not a structural lock. A limit on the shared
        // independent coordinate couples to both equality rows. Test a
        // dependent coordinate as well; both environments share the model.
        dofs[limitedDof].flags |= MR_DOF_FLAG_POSITION_LIMIT;
        dofs[limitedDof].limits = {-0.25f, 0.0f, 0.0f, 0.0f};
    }
    const std::array<float,4> inertia={0.2f,0.01f,2.0f,0.3f};
    auto* props=static_cast<MRBodyPropertiesGPU*>(b[3].contents);
    for(unsigned i=0;i<bodies;++i) {
        props[i].massAndInverseMass={1,1,0,0};props[i].inertiaRow0={inertia[i],0,0,0};
        props[i].inertiaRow1={0,inertia[i],0,0};props[i].inertiaRow2={0,0,inertia[i],0};
    }
    auto* d=static_cast<MRNumiHumanStandDispatchGPU*>(b[4].contents);
    d->abiVersion=MR_NUMI_HUMAN_STAND_ABI_VERSION;d->environmentCount=envs;d->stepCount=1;
    d->qStride=nq;d->vStride=nv;d->pointWorldStride=points;d->pointJacobianStride=points*3*nv;
    d->bodyPoseStride=bodies;d->generalizedForceStride=nv;d->contactIterationCount=sweeps;
    d->jointEqualityCount=neq;d->flags=MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES;
    if (limitedDof != MR_INVALID_INDEX)
        d->flags |= MR_NUMI_HUMAN_STAND_ENABLE_CONTACT;
    d->groundPointAndTimestep={0,0,0,h};d->groundNormal={0,0,1,0};d->targetRootOrientation={0,0,0,1};
    auto* eq=static_cast<MRNumiHumanJointEqualityGPU*>(b[20].contents);
    const float c1=0.7f,c2=-0.2f;
    eq[reverse?1:0].indices={8,7,7,6};eq[reverse?1:0].referencesAndCoefficients0={0,0,0,c1};
    eq[reverse?0:1].indices={9,8,7,6};eq[reverse?0:1].referencesAndCoefficients0={0,0,0,c2};
    auto* q=static_cast<float*>(b[5].contents);auto* v=static_cast<float*>(b[6].contents);
    auto* f=static_cast<float*>(b[10].contents);
    auto* pose=static_cast<MRArticulatedBodyPoseGPU*>(b[7].contents);
    auto* positions=static_cast<MRArticulatedPointWorldGPU*>(b[8].contents);
    auto* jac=static_cast<float*>(b[9].contents);
    std::array<std::array<double,4>,envs> expected{},old{};
    for(unsigned e=0;e<envs;++e) {
        q[e*nq+6]=1;
        v[e*nv+5]=e?-.2f:.1f;v[e*nv+6]=e?-.3f:.4f;
        v[e*nv+7]=loaded?c1*v[e*nv+6]:-.7f;
        v[e*nv+8]=loaded?c2*v[e*nv+6]:.9f;
        if(loaded) {f[e*nv+6]=2;f[e*nv+7]=-1;f[e*nv+8]=e?-3:3;}
        for(unsigned j=0;j<4;++j)old[e][j]=v[e*nv+5+j];
        for(unsigned i=0;i<bodies;++i) {
            pose[e*bodies+i].orientation={0,0,0,1};
            const std::array<V3,4> offsets={V3{0,0,0},V3{1,0,0},V3{0,1,0},V3{0,0,1}};
            for(unsigned p=0;p<4;++p) {
                const unsigned pi=e*points+4*i+p;
                positions[pi].position={offsets[p][0],offsets[p][1],offsets[p][2],0};
                for(unsigned k=0;k<3;++k)jac[pi*3*nv+k*nv+k]=1;
                for(unsigned k=3;k<nv;++k) {
                    V3 axis{0,0,0};if(k<6)axis[k-3]=1;else if(i>0&&k==5+i)axis[2]=1;
                    const auto velocity=cross(axis,offsets[p]);
                    for(unsigned c=0;c<3;++c)jac[pi*3*nv+c*nv+k]=velocity[c];
                }
            }
        }
        const std::array<double,3> t={1,double(c1),double(c2)};
        double a=inertia[0],bb=0,c=0,r0=inertia[0]*old[e][0],r1=0;
        for(unsigned j=0;j<3;++j) {
            const double m=inertia[j+1],momentum=m*(old[e][0]+old[e][j+1]);
            a+=m;bb+=m*t[j];c+=m*t[j]*t[j];r0+=momentum;
            r1+=t[j]*(momentum+double(h)*f[e*nv+6+j]);
        }
        const double determinant=a*c-bb*bb;
        expected[e][0]=(c*r0-bb*r1)/determinant;
        expected[e][1]=(a*r1-bb*r0)/determinant;
        if (limitedDof != MR_INVALID_INDEX) {
            const double multiplier = limitedDof == 6u ? 1.0 : limitedDof == 7u ? c1 : c2;
            if (multiplier * expected[e][1] > 0.0) {
                // With the independent joint stopped, angular momentum fixes
                // the floating root. This oracle does not use a Schur solve.
                expected[e][0] = r0 / a;
                expected[e][1] = 0.0;
            }
        }
        expected[e][2]=c1*expected[e][1];expected[e][3]=c2*expected[e][1];
    }
    id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    require(command!=nil&&encoder!=nil,"command creation failed");[encoder setComputePipelineState:pipeline];
    for(unsigned i=0;i<b.size();++i)[encoder setBuffer:b[i] offset:0 atIndex:i];
    [encoder dispatchThreadgroups:MTLSizeMake(envs,1,1) threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
    [encoder endEncoding];dispatch_semaphore_t finished=dispatch_semaphore_create(0);
    [command addCompletedHandler:^(id<MTLCommandBuffer>){dispatch_semaphore_signal(finished);}];[command commit];
    if(dispatch_semaphore_wait(finished,dispatch_time(DISPATCH_TIME_NOW,30*NSEC_PER_SEC))!=0)std::_Exit(2);
    require(command.status==MTLCommandBufferStatusCompleted,"Metal execution failed");
    auto* status=static_cast<MRNumiHumanStandStatusGPU*>(b[17].contents);unsigned checks=0;
    for(unsigned e=0;e<envs;++e) {
        require(status[e].code==MR_NUMI_HUMAN_STAND_SUCCESS&&status[e].completedSteps==1,"bilateral step failed");++checks;
        double oldMomentum=0,newMomentum=0,oldEnergy=0,newEnergy=0;
        for(unsigned j=0;j<4;++j) {
            const double error=std::abs(v[e*nv+5+j]-expected[e][j]);
            if(error>2e-5*(1+std::abs(expected[e][j])))std::cerr<<"row="<<j<<" env="<<e<<" sweeps="<<sweeps<<" limitedDof="<<limitedDof<<" got="<<v[e*nv+5+j]<<" expected="<<expected[e][j]<<'\n';
            require(error<2e-5*(1+std::abs(expected[e][j])),"coupled equality velocity differs from reduced-mass oracle");++checks;
            const double before=old[e][0]+(j?old[e][j]:0),after=v[e*nv+5]+(j?v[e*nv+5+j]:0);
            oldMomentum+=inertia[j]*before;newMomentum+=inertia[j]*after;
            oldEnergy+=.5*inertia[j]*before*before;newEnergy+=.5*inertia[j]*after*after;
        }
        require(std::abs(newMomentum-oldMomentum)<1e-5*(1+std::abs(oldMomentum)),"internal equality created angular momentum");++checks;
        require(std::isfinite(status[e].constraintImpulseWorkDiagnostics.x) &&
                    std::isfinite(status[e].constraintImpulseWorkDiagnostics.y) &&
                    std::isfinite(status[e].constraintImpulseWorkDiagnostics.z) &&
                    std::isfinite(status[e].constraintImpulseWorkDiagnostics.w) &&
                    std::isfinite(status[e].constraintImpulseAbsoluteWorkDiagnostics.x) &&
                    std::isfinite(status[e].constraintImpulseAbsoluteWorkDiagnostics.y) &&
                    std::isfinite(status[e].constraintImpulseAbsoluteWorkDiagnostics.z) &&
                    std::isfinite(status[e].constraintImpulseAbsoluteWorkDiagnostics.w),
                "constraint impulse work is non-finite");++checks;
        require(std::abs(status[e].constraintImpulseWorkDiagnostics.x)<2e-7 &&
                    std::abs(status[e].constraintImpulseWorkDiagnostics.y)<2e-7,
                "bilateral fixture reported contact work");++checks;
        if(!loaded) {
            require(newEnergy<=oldEnergy+1e-5,"unforced equality projection created energy");++checks;
            const double reportedWork =
                status[e].constraintImpulseWorkDiagnostics.z +
                status[e].constraintImpulseWorkDiagnostics.w;
            const double expectedWork = newEnergy - oldEnergy;
            require(std::abs(reportedWork-expectedWork)<
                        5e-5*(1+std::abs(expectedWork)),
                    "equality/limit impulse work differs from independent energy oracle");++checks;
            require(status[e].constraintImpulseAbsoluteWorkDiagnostics.z+
                        status[e].constraintImpulseAbsoluteWorkDiagnostics.w+2e-7>=
                        std::abs(reportedWork),
                    "absolute equality/limit work hid coupled-sweep activity");++checks;
        }
        require(std::abs(v[e*nv+7]-c1*v[e*nv+6])<1e-6&&std::abs(v[e*nv+8]-c2*v[e*nv+6])<1e-6,"published equality tangent residual");++checks;
        if (limitedDof != MR_INVALID_INDEX) {
            require(v[e*nv+limitedDof] <= 2e-6, "published finite-stop velocity violated"); ++checks;
            require(status[e].jointEqualityDiagnostics.y < 2e-6,
                    "limit correction invalidated a bilateral row before projection"); ++checks;
        }
        require(status[e].factorAndAssistance.z==0&&status[e].factorAndAssistance.w==0,"hidden assistance");++checks;
        require(std::abs(q[e*nq+7]-h*expected[e][1])<2e-7,"position integration differs from impulse solve");++checks;
    }
    return checks;
}
}
int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            require(argc==2,"expected production metallib path");id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            if(device==nil){std::cout<<"Metal unavailable; not executed\n";return 77;}
            NSError* error=nil;id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
            require(library!=nil,"library unavailable");id<MTLFunction> function=[library newFunctionWithName:@"mr_numi_human_stand_step"];
            require(function!=nil,"production kernel unavailable");id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
            require(pipeline!=nil,"pipeline unavailable");id<MTLCommandQueue> queue=[device newCommandQueue];require(queue!=nil,"queue unavailable");
            unsigned checks=0;
            for(float h:{1e-4f,5e-5f,1.25e-5f})for(unsigned sweeps:{1u,4u,64u})for(bool reverse:{false,true})for(bool loaded:{false,true})
                for(unsigned limitedDof:{MR_INVALID_INDEX,6u,7u,8u})
                    checks+=exercise(device,pipeline,queue,h,sweeps,reverse,loaded,limitedDof);
            std::cout<<"Human production bilateral block: "<<checks<<" checks passed; device="<<device.name.UTF8String<<'\n';return 0;
        }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    }
}
