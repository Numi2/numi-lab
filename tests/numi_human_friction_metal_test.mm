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

// Actual production kernel. One floating rigid body with an offset ground
// contact has unequal tangent responses. The independent CPU oracle uses the
// analytic world-space contact response, not a shared production helper.
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
using V3 = std::array<float,3>;
V3 cross(V3 a,V3 b) {
    return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}
std::array<double,2> oracle(double a,double d,double vx,double vy,double radius) {
    if (radius==0) return {0,0};
    auto solve=[&](double s){return std::array<double,2>{-vx/(a+s),-vy/(d+s)};};
    auto p=solve(0);
    if(std::hypot(p[0],p[1])<=radius) return p;
    double lower=0,upper=std::hypot(vx,vy)/radius;
    for(unsigned i=0;i<100;++i) {
        double mid=(lower+upper)/2;
        p=solve(mid);
        if(std::hypot(p[0],p[1])>radius) lower=mid;else upper=mid;
    }
    return solve(upper);
}
unsigned exercise(id<MTLDevice> device,id<MTLComputePipelineState> pipeline,
                  id<MTLCommandQueue> queue,float friction,float speedSign,float seedScale,float angle) {
    constexpr unsigned nv=6,nq=7,points=5,contacts=1;
    const std::array<std::size_t,26> sizes={
        sizeof(MRWorldGPU),sizeof(MRArticulationGPU),nv*sizeof(MRDofPropertiesGPU),
        sizeof(MRBodyPropertiesGPU),sizeof(MRNumiHumanStandDispatchGPU),
        nq*sizeof(float),nv*sizeof(float),sizeof(MRArticulatedBodyPoseGPU),
        points*sizeof(MRArticulatedPointWorldGPU),points*3*nv*sizeof(float),nv*sizeof(float),
        sizeof(MRNumiHumanStandContactGPU),MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS*nv*sizeof(float),2*sizeof(mr_float4),
        nv*nv*sizeof(float),(4*nv+12*contacts)*sizeof(float),(3*contacts+nv)*nv*sizeof(float),
        sizeof(MRNumiHumanStandStatusGPU),sizeof(MRNumiHumanTendonBindingGPU),
        sizeof(MRNumiHumanTendonTransferResultGPU),sizeof(MRNumiHumanJointEqualityGPU),
        sizeof(MRCompensatedRootTranslationGPU),sizeof(mr_float4),points*sizeof(mr_float4),
        nv*(nv+1)*sizeof(float),3u*nv*sizeof(float)};
    std::array<id<MTLBuffer>,26> buffers;
    for(unsigned i=0;i<buffers.size();++i) {
        buffers[i]=[device newBufferWithLength:sizes[i] options:MTLResourceStorageModeShared];
        require(buffers[i]!=nil,"allocation failed");
        std::memset(buffers[i].contents,0,sizes[i]);
    }
    constexpr float h=0.0001f,mass=2.5f;
    auto* world=static_cast<MRWorldGPU*>(buffers[0].contents);
    world->abiVersion=MR_ENGINE_ABI_VERSION;world->bodyCount=1;world->articulationCount=1;
    world->nq=nq;world->nv=nv;world->gravityAndTimestep={0,0,0,h};
    auto* art=static_cast<MRArticulationGPU*>(buffers[1].contents);
    art->bodyCount=1;art->rootType=MR_ROOT_FLOATING;art->nq=nq;art->nv=nv;
    auto* dofs=static_cast<MRDofPropertiesGPU*>(buffers[2].contents);
    for(unsigned i=0;i<nv;++i){dofs[i].vIndex=i;dofs[i].qIndex=i<3?i:MR_INVALID_INDEX;}
    auto* body=static_cast<MRBodyPropertiesGPU*>(buffers[3].contents);
    body->massAndInverseMass={mass,1/mass,0,0};
    body->inertiaRow0={0.1f,0,0,0};body->inertiaRow1={0,1,0,0};body->inertiaRow2={0,0,1,0};
    auto* dispatch=static_cast<MRNumiHumanStandDispatchGPU*>(buffers[4].contents);
    dispatch->abiVersion=MR_NUMI_HUMAN_STAND_ABI_VERSION;
    dispatch->environmentCount=1;dispatch->stepCount=1;dispatch->qStride=nq;dispatch->vStride=nv;
    dispatch->pointWorldStride=points;dispatch->pointJacobianStride=points*3*nv;
    dispatch->bodyPoseStride=1;dispatch->generalizedForceStride=nv;
    dispatch->contactIterationCount=8;dispatch->supportContactCount=1;
    dispatch->groundPointAndTimestep={0,0,0,h};dispatch->groundNormal={0,0,1,0};
    dispatch->targetRootOrientation={0,0,0,1};dispatch->flags=MR_NUMI_HUMAN_STAND_ENABLE_CONTACT;
    const double cosine=std::cos(double(angle)),sine=std::sin(double(angle));
    const auto rotate=[&](double x,double y) {
        return std::array<double,2>{cosine*x-sine*y,sine*x+cosine*y};
    };
    const auto initialVelocity=rotate(2*speedSign,speedSign);
    auto* q=static_cast<float*>(buffers[5].contents);q[2]=0.5f;
    q[5]=std::sin(angle/2);q[6]=std::cos(angle/2);
    auto* v=static_cast<float*>(buffers[6].contents);
    v[0]=static_cast<float>(initialVelocity[0]);v[1]=static_cast<float>(initialVelocity[1]);v[2]=-1;
    auto* pose=static_cast<MRArticulatedBodyPoseGPU*>(buffers[7].contents);
    pose->orientation={0,0,q[5],q[6]};
    auto* positions=static_cast<MRArticulatedPointWorldGPU*>(buffers[8].contents);
    auto* jacobian=static_cast<float*>(buffers[9].contents);
    const std::array<V3,points> offsets={V3{0,0,0},{1,0,0},{0,1,0},{0,0,1},{0,0,-0.5f}};
    for(unsigned p=0;p<points;++p) {
        const auto xy=rotate(offsets[p][0],offsets[p][1]);
        const V3 worldOffset{static_cast<float>(xy[0]),static_cast<float>(xy[1]),offsets[p][2]};
        positions[p].position={worldOffset[0],worldOffset[1],0.5f+worldOffset[2],0};
        for(unsigned d=0;d<3;++d) jacobian[p*3*nv+d*nv+d]=1;
        for(unsigned d=3;d<6;++d) {
            V3 axis{0,0,0};axis[d-3]=1;const auto motion=cross(axis,worldOffset);
            for(unsigned c=0;c<3;++c) jacobian[p*3*nv+c*nv+d]=motion[c];
        }
    }
    auto* contact=static_cast<MRNumiHumanStandContactGPU*>(buffers[11].contents);
    contact->bodyIndex=0;contact->pointQueryIndex=4;
    contact->frictionSlopAndStabilization={friction,0.002f,0.2f,seedScale*mass/h};
    const double a=1/double(mass)+0.25/body->inertiaRow1.y;
    const double d=1/double(mass)+0.25/body->inertiaRow0.x;
    // Solve in the principal inertia frame, then rotate the analytic result.
    // The production kernel must handle the resulting off-diagonal world tensor.
    const auto p=oracle(a,d,2*speedSign,speedSign,double(friction)*mass);
    const auto worldImpulse=rotate(p[0],p[1]);
    const auto worldAngular=rotate(0.5*p[1]/body->inertiaRow0.x,-0.5*p[0]/body->inertiaRow1.y);
    const std::array<double,nv> expected={v[0]+worldImpulse[0]/mass,v[1]+worldImpulse[1]/mass,0,
        worldAngular[0],worldAngular[1],0};
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    require(command!=nil&&encoder!=nil,"command creation failed");
    [encoder setComputePipelineState:pipeline];
    for(unsigned i=0;i<buffers.size();++i) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
           threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
    [encoder endEncoding];
    dispatch_semaphore_t complete=dispatch_semaphore_create(0);
    [command addCompletedHandler:^(id<MTLCommandBuffer>){dispatch_semaphore_signal(complete);}];
    [command commit];
    if(dispatch_semaphore_wait(complete,dispatch_time(DISPATCH_TIME_NOW,30*NSEC_PER_SEC))!=0) {
        std::cerr<<"friction production kernel timed out\n";std::_Exit(2);
    }
    require(command.status==MTLCommandBufferStatusCompleted,"Metal command failed");
    auto* status=static_cast<MRNumiHumanStandStatusGPU*>(buffers[17].contents);
    require(status->code==MR_NUMI_HUMAN_STAND_SUCCESS&&status->completedSteps==1,
            "production step did not complete");
    require(status->velocityDiagnosticOwners.x<nv &&
                status->velocityDiagnosticOwners.y<nv &&
                status->velocityDiagnosticOwners.z<nv &&
                status->velocityDiagnosticOwners.w<nv,
            "velocity diagnostic owner is invalid");
    require(status->velocityDiagnostics.x>=0.0f &&
                status->velocityDiagnostics.y>=0.0f &&
                status->velocityDiagnostics.z>=0.0f &&
                status->velocityDiagnostics.w>=0.0f &&
                std::abs(status->velocityDiagnostics.z/h-
                         status->contactAndAcceleration.w)<=
                    2e-4f*(1.0f+status->contactAndAcceleration.w),
            "impulse-equivalent acceleration does not match pre-projection delta-v");
    require(std::abs(status->velocityDiagnostics.w-
                     status->velocityDiagnostics.z)<=2e-6f,
            "friction fixture without equality projection changed published delta-v");
    double maxError=0;
    for(unsigned i=0;i<nv;++i) maxError=std::max(maxError,std::abs(v[i]-expected[i]));
    if(maxError>3e-5) {
        std::cerr<<"ANISOTROPIC_ORACLE_MISMATCH mu="<<friction<<" seed="<<seedScale
                 <<" max_error="<<maxError<<" vx="<<v[0]<<" expected="<<expected[0]<<'\n';
        throw std::runtime_error("production tangent impulse violates maximum dissipation");
    }
    const double worldPx=mass*(v[0]-initialVelocity[0]),worldPy=mass*(v[1]-initialVelocity[1]);
    const double px=cosine*worldPx+sine*worldPy,py=-sine*worldPx+cosine*worldPy;
    require(std::hypot(px,py)<=double(friction)*mass+2e-5,"friction disk violated");
    require(std::abs(status->contactAndAcceleration.z-mass)<2e-5,"normal contact solution changed");
    require(status->factorAndAssistance.z==0&&status->factorAndAssistance.w==0,"hidden assistance");
    require(px*(2*speedSign+a*px)+py*(speedSign+d*py)<=2e-5,"positive sliding work");
    const double finalContactX = expected[0] - 0.5 * expected[4];
    const double finalContactY = expected[1] + 0.5 * expected[3];
    const double expectedNormalWork = 0.5 * mass * (-1.0 + expected[2]);
    const double expectedTangentWork = 0.5 * (
        worldImpulse[0] * (initialVelocity[0] + finalContactX) +
        worldImpulse[1] * (initialVelocity[1] + finalContactY)
    );
    require(std::abs(status->constraintImpulseWorkDiagnostics.x-
                     expectedNormalWork)<3e-5*(1+std::abs(expectedNormalWork)),
            "normal impulse work differs from analytic offset-contact oracle");
    require(std::abs(status->constraintImpulseWorkDiagnostics.y-
                     expectedTangentWork)<3e-5*(1+std::abs(expectedTangentWork)),
            "tangential impulse work differs from analytic offset-contact oracle");
    require(std::abs(status->constraintImpulseWorkDiagnostics.z)<2e-7 &&
                std::abs(status->constraintImpulseWorkDiagnostics.w)<2e-7,
            "contact fixture reported equality or source-limit work");
    require(status->constraintImpulseAbsoluteWorkDiagnostics.x+2e-7>=
                std::abs(status->constraintImpulseWorkDiagnostics.x) &&
                status->constraintImpulseAbsoluteWorkDiagnostics.y+2e-7>=
                std::abs(status->constraintImpulseWorkDiagnostics.y),
            "absolute offset-contact work hid coupled-sweep activity");
    // Independently check the cached I_world * J_angular columns, including
    // zero translational columns and non-diagonal rotated inertias.
    const auto* spatial=static_cast<const float*>(buffers[12].contents);
    for(unsigned column=0;column<nv;++column) {
        const double wx=column==3?1:0,wy=column==4?1:0,wz=column==5?1:0;
        const double localX=cosine*wx+sine*wy,localY=-sine*wx+cosine*wy;
        const auto weighted=rotate(body->inertiaRow0.x*localX,body->inertiaRow1.y*localY);
        const std::array<double,3> exact={weighted[0],weighted[1],body->inertiaRow2.z*wz};
        for(unsigned axis=0;axis<3;++axis)
            require(std::abs(spatial[6*nv+axis*nv+column]-exact[axis])<2e-6,
                    "cached world-inertia column differs from analytic rotation");
    }
    return 18+3*nv;
}
}
int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            require(argc==2,"usage: friction-metal-test /path/NumiHumanStand.metallib");
            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            if(device==nil){std::cout<<"gpu_available=false execution=not_run\n";return 77;}
            NSError* error=nil;
            id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
            require(library!=nil,"cannot load production metallib");
            id<MTLFunction> function=[library newFunctionWithName:@"mr_numi_human_stand_step"];
            require(function!=nil,"production entry point missing");
            id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
            require(pipeline!=nil,"pipeline creation failed");
            id<MTLCommandQueue> queue=[device newCommandQueue];require(queue!=nil,"queue unavailable");
            unsigned checks=0;
            for(float mu:{0.4f,0.0f,2.0f})
                for(float sign:{1.0f,-1.0f})
                    for(float seed:{0.0f,1.0f,2.0f})
                        for(float angle:{0.0f,0.37f,1.2f,-0.65f})
                            checks+=exercise(device,pipeline,queue,mu,sign,seed,angle);
            std::cout<<"gpu_available=true device=\""<<device.name.UTF8String
                     <<"\" checks="<<checks<<" status=passed scope=anisotropic_contact_not_full_human\n";
        } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
}
