#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/compensated_translation_gpu.h"
#include "metalrobo/numi_human_tendon_gpu.h"
#include "metalrobo/numi_human_joint_equality_gpu.h"
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

// Runs the production standing kernel, not a duplicate dynamics kernel.
// A free spherical-inertia parent and coaxial spherical-inertia child give
// M_angular=[[1.5,0.5],[0.5,0.5]], hence relative inertia 1/3. Their coincident
// centres remove centrifugal translation. Contact/anatomical validity is not
// asserted by this deliberately small numerical fixture.
namespace {
constexpr unsigned nv=7, nq=8, bodies=2, points=8, environments=2;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
using V3=std::array<float,3>;
V3 cross(V3 a,V3 b) { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
mr_float4 rotation(float angle) { return {0,0,std::sin(angle/2),std::cos(angle/2)}; }

std::size_t exercise(id<MTLDevice> device,id<MTLComputePipelineState> pipeline,
                     id<MTLCommandQueue> queue,float h,bool enabled) {
    // Even disabled array arguments require one ABI-sized element for Metal validation.
    const std::array<std::size_t,25> sizes={
        sizeof(MRWorldGPU),sizeof(MRArticulationGPU),nv*sizeof(MRDofPropertiesGPU),
        bodies*sizeof(MRBodyPropertiesGPU),sizeof(MRNumiHumanStandDispatchGPU),
        environments*nq*sizeof(float),environments*nv*sizeof(float),
        environments*bodies*sizeof(MRArticulatedBodyPoseGPU),
        environments*points*sizeof(MRArticulatedPointWorldGPU),
        environments*points*3*nv*sizeof(float),environments*nv*sizeof(float),
        sizeof(MRNumiHumanStandContactGPU),
        environments*bodies*6*nv*sizeof(float),environments*bodies*2*sizeof(mr_float4),
        environments*nv*nv*sizeof(float),environments*4*nv*sizeof(float),16,
        environments*sizeof(MRNumiHumanStandStatusGPU),
        sizeof(MRNumiHumanTendonBindingGPU),sizeof(MRNumiHumanTendonTransferResultGPU),
        sizeof(MRNumiHumanJointEqualityGPU),
        environments*sizeof(MRCompensatedRootTranslationGPU),
        environments*bodies*sizeof(mr_float4),environments*points*sizeof(mr_float4),
        nv*(nv+1)*sizeof(float)};
    std::array<id<MTLBuffer>,25> buffers;
    for (unsigned i=0;i<buffers.size();++i) {
        buffers[i]=[device newBufferWithLength:sizes[i] options:MTLResourceStorageModeShared];
        require(buffers[i]!=nil,"Metal buffer allocation failed");
        std::memset(buffers[i].contents,0,sizes[i]);
    }
    auto* world=static_cast<MRWorldGPU*>(buffers[0].contents);
    world->abiVersion=MR_ENGINE_ABI_VERSION;
    world->bodyCount=bodies; world->articulationCount=1; world->nq=nq; world->nv=nv;
    auto* articulation=static_cast<MRArticulationGPU*>(buffers[1].contents);
    articulation->bodyCount=bodies; articulation->rootType=MR_ROOT_FLOATING;
    articulation->nq=nq; articulation->nv=nv;
    auto* dofs=static_cast<MRDofPropertiesGPU*>(buffers[2].contents);
    for(unsigned i=0;i<nv;++i) { dofs[i].vIndex=i; dofs[i].qIndex=i<3?i:MR_INVALID_INDEX; }
    dofs[6].qIndex=7; dofs[6].drive.y=0.2f;
    auto* physical=static_cast<MRBodyPropertiesGPU*>(buffers[3].contents);
    for(unsigned i=0;i<bodies;++i) {
        const float inertia=i==0?1.0f:0.5f, mass=i==0?4.0f:2.0f;
        physical[i].massAndInverseMass={mass,1/mass,0,0};
        physical[i].inertiaRow0={inertia,0,0,0};
        physical[i].inertiaRow1={0,inertia,0,0};
        physical[i].inertiaRow2={0,0,inertia,0};
    }
    auto* dispatch=static_cast<MRNumiHumanStandDispatchGPU*>(buffers[4].contents);
    dispatch->abiVersion=MR_NUMI_HUMAN_STAND_ABI_VERSION;
    dispatch->environmentCount=environments; dispatch->stepCount=1;
    dispatch->qStride=nq; dispatch->vStride=nv; dispatch->pointWorldStride=points;
    dispatch->pointJacobianStride=points*3*nv; dispatch->bodyPoseStride=bodies;
    dispatch->generalizedForceStride=nv; dispatch->contactIterationCount=1;
    dispatch->groundPointAndTimestep={0,0,0,h}; dispatch->groundNormal={0,0,1,0};
    dispatch->targetRootOrientation={0,0,0,1};
    dispatch->flags=enabled?MR_NUMI_HUMAN_STAND_HAS_PASSIVE_JOINT_PROGRAM:0;
    auto* program=static_cast<float*>(buffers[24].contents);
    program[6*nv+6]=100000.0f; program[nv*nv+6]=0.1f;
    auto* q=static_cast<float*>(buffers[5].contents);
    auto* v=static_cast<float*>(buffers[6].contents);
    for(unsigned e=0;e<environments;++e) {
        q[e*nq+6]=1; q[e*nq+7]=e==0?0.2f:-0.12f;
        v[e*nv+6]=e==0?0.3f:-0.17f; v[e*nv+5]=-v[e*nv+6]/3;
    }
    std::size_t checks=0;
    for(unsigned step=0;step<64;++step) {
        std::array<double,environments> expectedV{},expectedQ{},oldEnergy{},momentum{};
        auto* poses=static_cast<MRArticulatedBodyPoseGPU*>(buffers[7].contents);
        auto* jacobian=static_cast<float*>(buffers[9].contents);
        std::memset(jacobian,0,sizes[9]);
        for(unsigned e=0;e<environments;++e) {
            double x=q[e*nq+7]-double(program[nv*nv+6]), u=v[e*nv+6];
            const double k=enabled?program[6*nv+6]:0.0, damping=dofs[6].drive.y;
            expectedV[e]=((1.0/3.0)*u-double(h)*k*x)/(1.0/3.0+h*damping+double(h)*h*k);
            expectedQ[e]=q[e*nq+7]+h*expectedV[e];
            oldEnergy[e]=0.5*k*x*x+0.5*(1.0/3.0)*u*u;
            momentum[e]=1.5*v[e*nv+5]+0.5*v[e*nv+6];
            float rootAngle=2*std::atan2(q[e*nq+5],q[e*nq+6]);
            for(unsigned body=0;body<bodies;++body) {
                float angle=rootAngle+(body==1?q[e*nq+7]:0);
                poses[e*bodies+body].orientation=rotation(angle);
                const std::array<V3,4> offsets={V3{0,0,0},V3{std::cos(angle),std::sin(angle),0},
                    V3{-std::sin(angle),std::cos(angle),0},V3{0,0,1}};
                for(unsigned probe=0;probe<4;++probe) {
                    unsigned base=(e*points+4*body+probe)*3*nv;
                    for(unsigned d=0;d<3;++d) jacobian[base+d*nv+d]=1;
                    for(unsigned d=3;d<nv;++d) {
                        V3 axis={0,0,0};
                        if(d<6) axis[d-3]=1;
                        else if(body==1) axis[2]=1;
                        V3 motion=cross(axis,offsets[probe]);
                        for(unsigned c=0;c<3;++c) jacobian[base+c*nv+d]=motion[c];
                    }
                }
            }
        }
        id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        require(command!=nil&&encoder!=nil,"Metal command creation failed");
        [encoder setComputePipelineState:pipeline];
        for(unsigned i=0;i<buffers.size();++i) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder dispatchThreadgroups:MTLSizeMake(environments,1,1)
               threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
        [encoder endEncoding];
        dispatch_semaphore_t complete=dispatch_semaphore_create(0);
        [command addCompletedHandler:^(id<MTLCommandBuffer>) { dispatch_semaphore_signal(complete); }];
        [command commit];
        if(dispatch_semaphore_wait(complete,dispatch_time(DISPATCH_TIME_NOW,30*NSEC_PER_SEC))!=0) {
            std::cerr<<"production Metal probe timed out; no result admitted\n";
            std::_Exit(2); // Do not release buffers potentially still owned by the GPU.
        }
        require(command.status==MTLCommandBufferStatusCompleted,"production Metal command failed");
        auto* status=static_cast<MRNumiHumanStandStatusGPU*>(buffers[17].contents);
        for(unsigned e=0;e<environments;++e) {
            if(status[e].code!=MR_NUMI_HUMAN_STAND_SUCCESS ||
               std::abs(v[e*nv+6]-expectedV[e])>2e-5*(1+std::abs(expectedV[e]))) {
                std::cerr<<"h="<<h<<" enabled="<<enabled<<" step="<<step<<" env="<<e
                         <<" status="<<status[e].code<<" failing_index="<<status[e].failingIndex
                         <<" expected_v="<<expectedV[e]<<" actual_v="<<v[e*nv+6]<<'\n';
            }
            require(status[e].code==MR_NUMI_HUMAN_STAND_SUCCESS&&status[e].completedSteps==1,
                    "production kernel did not complete the step");
            require(std::abs(v[e*nv+6]-expectedV[e])<=2e-5*(1+std::abs(expectedV[e])),
                    "Metal velocity differs from independent backward-Euler solution");
            require(std::abs(q[e*nq+7]-expectedQ[e])<=2e-6,
                    "Metal configuration differs from independent backward-Euler solution");
            require(std::abs(1.5*v[e*nv+5]+0.5*v[e*nv+6]-momentum[e])<=2e-5,
                    "internal passive joint created a net angular impulse");
            double x=q[e*nq+7]-double(program[nv*nv+6]),u=v[e*nv+6];
            double energy=0.5*(enabled?program[6*nv+6]:0.0)*x*x+0.5*(1.0/3.0)*u*u;
            require(energy<=oldEnergy[e]+2e-5*(1+oldEnergy[e]),"passive implicit step created energy");
            checks+=5;
        }
    }
    return checks;
}
}
int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            require(argc==2,"usage: passive-metal-test /path/NumiHumanStand.metallib");
            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            if(device==nil) { std::cout<<"gpu_available=false execution=not_run\n"; return 77; }
            NSError* error=nil;
            id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
            require(library!=nil,"cannot load production standing metallib");
            id<MTLFunction> function=[library newFunctionWithName:@"mr_numi_human_stand_step"];
            require(function!=nil,"production standing entry point is missing");
            id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
            require(pipeline!=nil,"cannot compile production standing pipeline");
            id<MTLCommandQueue> queue=[device newCommandQueue];
            require(queue!=nil,"cannot create Metal queue");
            std::size_t checks=0;
            for(float timestep:{1.25e-5f,1.0e-4f,1.0e-3f})
                for(bool enabled:{false,true}) checks+=exercise(device,pipeline,queue,timestep,enabled);
            std::cout<<"gpu_available=true device=\""<<device.name.UTF8String
                     <<"\" production_kernel=mr_numi_human_stand_step checks="<<checks
                     <<" status=passed scope=two_body_passive_integration_not_full_human\n";
            return 0;
        } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
    }
}
