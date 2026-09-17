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
#include <limits>
#include <stdexcept>
#include <vector>

// Runs the production standing kernel, not a duplicate dynamics kernel.
// A free spherical-inertia parent and coaxial spherical-inertia child give
// M_angular=[[1.5,0.5],[0.5,0.5]], hence relative inertia 1/3. Their coincident
// centres remove centrifugal translation. The contact cases use one point at
// the coincident centres with an analytic vertical/friction solution; this
// does not validate anatomical contact or a full Human trajectory.
namespace {
constexpr unsigned nv=7, nq=8, bodies=2, points=8, environments=2;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
using V3=std::array<float,3>;
V3 cross(V3 a,V3 b) { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
mr_float4 rotation(float angle) { return {0,0,std::sin(angle/2),std::cos(angle/2)}; }

std::size_t exercise(id<MTLDevice> device,id<MTLComputePipelineState> pipeline,
                     id<MTLCommandQueue> queue,float h,bool enabled,
                     float supportSeedScale=-1.0f,unsigned contactMode=0u) {
    const unsigned contactCount=supportSeedScale>=0.0f?1u:0u;
    // Even disabled arguments require one ABI-sized element for Metal validation.
    const std::array<std::size_t,25> sizes={
        sizeof(MRWorldGPU),sizeof(MRArticulationGPU),nv*sizeof(MRDofPropertiesGPU),
        bodies*sizeof(MRBodyPropertiesGPU),sizeof(MRNumiHumanStandDispatchGPU),
        environments*nq*sizeof(float),environments*nv*sizeof(float),
        environments*bodies*sizeof(MRArticulatedBodyPoseGPU),
        environments*points*sizeof(MRArticulatedPointWorldGPU),
        environments*points*3*nv*sizeof(float),environments*nv*sizeof(float),
        sizeof(MRNumiHumanStandContactGPU),
        environments*bodies*6*nv*sizeof(float),environments*bodies*2*sizeof(mr_float4),
        environments*nv*nv*sizeof(float),
        environments*(4*nv+12*contactCount)*sizeof(float),
        environments*(3*contactCount+nv)*nv*sizeof(float),
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
    world->gravityAndTimestep={0,0,contactCount?-9.81f:0.0f,h};
    auto* articulation=static_cast<MRArticulationGPU*>(buffers[1].contents);
    articulation->bodyCount=bodies; articulation->rootType=MR_ROOT_FLOATING;
    articulation->nq=nq; articulation->nv=nv;
    auto* dofs=static_cast<MRDofPropertiesGPU*>(buffers[2].contents);
    for(unsigned i=0;i<nv;++i) { dofs[i].vIndex=i; dofs[i].qIndex=i<3?i:MR_INVALID_INDEX; }
    dofs[6].qIndex=7; dofs[6].drive.y=0.2f;
    if(contactMode==3u) {
        // This interval is initially inactive but its response must be
        // prepared: later coupled rows may activate it. Poisoning the
        // arena detects cross-environment aliasing even for equal masses.
        dofs[6].flags |= MR_DOF_FLAG_POSITION_LIMIT;
        dofs[6].limits={-1.0f,1.0f,0.0f,0.0f};
        std::fill_n(static_cast<float*>(buffers[16].contents),
            sizes[16]/sizeof(float),std::numeric_limits<float>::quiet_NaN());
    }
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
    dispatch->generalizedForceStride=nv; dispatch->contactIterationCount=8;
    dispatch->supportContactCount=contactCount;
    dispatch->groundPointAndTimestep={0,0,0,h}; dispatch->groundNormal={0,0,1,0};
    dispatch->targetRootOrientation={0,0,0,1};
    dispatch->flags=(enabled?MR_NUMI_HUMAN_STAND_HAS_PASSIVE_JOINT_PROGRAM:0) |
        (contactCount?MR_NUMI_HUMAN_STAND_ENABLE_CONTACT:0);
    auto* contact=static_cast<MRNumiHumanStandContactGPU*>(buffers[11].contents);
    contact->bodyIndex=0; contact->pointQueryIndex=0;
    contact->frictionSlopAndStabilization={0.5f,0.002f,0.2f,
        contactCount?supportSeedScale*6.0f*9.81f:0.0f};
    auto* program=static_cast<float*>(buffers[24].contents);
    program[6*nv+6]=100000.0f; program[nv*nv+6]=0.1f;
    auto* q=static_cast<float*>(buffers[5].contents);
    auto* v=static_cast<float*>(buffers[6].contents);
    for(unsigned e=0;e<environments;++e) {
        q[e*nq+6]=1; q[e*nq+7]=e==0?0.2f:-0.12f;
        v[e*nv+6]=contactCount?0.0f:(e==0?0.3f:-0.17f);
        v[e*nv+5]=-v[e*nv+6]/3;
        if(contactCount) {
            v[e*nv]=(e==0?0.02f:-0.035f);
            v[e*nv+2]=contactMode==1u?0.1f:(contactMode==2u?-0.1f:0.0f);
        }
    }
    std::size_t checks=0;
    for(unsigned step=0;step<(contactCount?1u:64u);++step) {
        std::array<double,environments> expectedV{},expectedQ{},oldEnergy{},momentum{};
        std::array<double,environments> normalImpulse{},tangentImpulse{},expectedX{},expectedZ{};
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
            if(contactCount) {
                const double freeZ=v[e*nv+2]+double(h)*world->gravityAndTimestep.z;
                normalImpulse[e]=6.0*std::max(-freeZ,0.0);
                expectedZ[e]=std::max(freeZ,0.0);
                const double initialX=v[e*nv];
                tangentImpulse[e]=std::min(6.0*std::abs(initialX),0.5*normalImpulse[e]);
                expectedX[e]=std::copysign(std::max(std::abs(initialX)-tangentImpulse[e]/6.0,0.0),initialX);
            }
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
            std::_Exit(2);
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
            require(status[e].velocityDiagnosticOwners.x<nv &&
                        status[e].velocityDiagnosticOwners.y<nv &&
                        status[e].velocityDiagnosticOwners.z<nv &&
                        status[e].velocityDiagnosticOwners.w<nv,
                    "production kernel returned invalid velocity-diagnostic owners");
            require(status[e].velocityDiagnostics.x>=0.0f &&
                        status[e].velocityDiagnostics.y>=0.0f &&
                        status[e].velocityDiagnostics.z>=0.0f &&
                        status[e].velocityDiagnostics.w>=0.0f &&
                        std::abs(status[e].velocityDiagnostics.z/h-
                                 status[e].contactAndAcceleration.w)<=
                            2e-4f*(1.0f+status[e].contactAndAcceleration.w),
                    "smooth and impulse-equivalent velocity diagnostics are inconsistent");
            require(std::abs(status[e].velocityDiagnostics.w-
                             status[e].velocityDiagnostics.z)<=2e-6f,
                    "fixture without equality projection changed published delta-v");
            require(std::abs(v[e*nv+6]-expectedV[e])<=2e-5*(1+std::abs(expectedV[e])),
                    "Metal velocity differs from independent backward-Euler solution");
            require(std::abs(q[e*nq+7]-expectedQ[e])<=2e-6,
                    "Metal configuration differs from independent backward-Euler solution");
            require(std::abs(1.5*v[e*nv+5]+0.5*v[e*nv+6]-momentum[e])<=2e-5,
                    "internal passive joint created a net angular impulse");
            double x=q[e*nq+7]-double(program[nv*nv+6]),u=v[e*nv+6];
            double energy=0.5*(enabled?program[6*nv+6]:0.0)*x*x+0.5*(1.0/3.0)*u*u;
            require(energy<=oldEnergy[e]+2e-5*(1+oldEnergy[e]),"passive implicit step created energy");
            checks+=8;
            if(contactMode==3u) {
                const unsigned stride=(3u*contactCount+nv)*nv;
                const auto* response=static_cast<const float*>(buffers[16].contents)
                    +e*stride+3u*contactCount*nv;
                const double expectedResponse=1.0/(1.0/3.0+h*dofs[6].drive.y);
                require(std::isfinite(response[6]) &&
                    std::abs(response[6]-expectedResponse)<2e-5 &&
                    std::isfinite(response[5]) &&
                    std::abs(response[5]+expectedResponse/3.0)<2e-5,
                    "limit response column was not written in its own environment");
                ++checks;
            }
            if(contactCount) {
                const auto& measured=status[e];
                const bool matches=std::abs(v[e*nv]-expectedX[e])<2e-6 &&
                    std::abs(v[e*nv+2]-expectedZ[e])<2e-6 &&
                    std::abs(measured.contactAndAcceleration.z-normalImpulse[e])<1e-5 &&
                    std::abs(measured.constraintImpulseDiagnostics.y-tangentImpulse[e])<1e-5;
                if(!matches) {
                    std::cerr<<"support h="<<h<<" seed_scale="<<supportSeedScale<<" mode="<<contactMode
                             <<" env="<<e<<" vx="<<v[e*nv]<<" expected_vx="<<expectedX[e]
                             <<" vz="<<v[e*nv+2]<<" expected_vz="<<expectedZ[e]
                             <<" normal="<<measured.contactAndAcceleration.z<<" expected_normal="<<normalImpulse[e]
                             <<" tangent="<<measured.constraintImpulseDiagnostics.y<<" expected_tangent="<<tangentImpulse[e]<<'\n';
                }
                require(matches,"production support/friction solution differs from independent impulse oracle");
                require(measured.factorAndAssistance.z==0.0f&&measured.factorAndAssistance.w==0.0f,
                        "contact fixture received artificial root assistance");
                require(measured.contactAndAcceleration.z>=0.0f,
                        "unilateral contact pulled the body toward the plane");
                require(measured.constraintImpulseDiagnostics.y<=0.5f*measured.contactAndAcceleration.z+1e-6f,
                        "tangential impulse exceeded friction cone");
                checks+=4;
            }
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
            const auto passiveChecks=checks;
            for(float timestep:{1.25e-5f,1.0e-4f,1.0e-3f})
                for(float seed:{0.0f,1.0f,2.0f})
                    for(unsigned mode:{0u,1u,2u,3u})
                        checks+=exercise(device,pipeline,queue,timestep,false,seed,mode);
            std::cout<<"gpu_available=true device=\""<<device.name.UTF8String
                     <<"\" production_kernel=mr_numi_human_stand_step checks="<<checks
                     <<" passive_checks="<<passiveChecks<<" support_checks="<<(checks-passiveChecks)
                     <<" status=passed scope=two_body_passive_and_contact_not_full_human\n";
            return 0;
        } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
    }
}
