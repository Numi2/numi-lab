// Directly exercise the dedicated ABA reaction output. Not a coupled step.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
static void check(bool b,const char* why){if(!b)throw std::runtime_error(why);}
int main(int argc,char**argv){@autoreleasepool{try{
 check(argc==2,"expected isolated slider ABA metallib");id<MTLDevice> dev=MTLCreateSystemDefaultDevice();check(dev!=nil,"no device");NSError* error=nil;
 id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];check(lib!=nil,"no library");
 id<MTLFunction> fn=[lib newFunctionWithName:@"mr_slider_midpoint_aba_reaction"];check(fn!=nil,"no reaction kernel");
 id<MTLComputePipelineState> pipe=[dev newComputePipelineStateWithFunction:fn error:&error];check(pipe!=nil,"no pipeline");
 id<MTLCommandQueue> queue=[dev newCommandQueue];check(queue!=nil,"no queue");
 MRWorldGPU world{};world.abiVersion=MR_ENGINE_ABI_VERSION;world.bodyCount=2;world.articulationCount=1;
 world.jointCount=1;world.nq=1;world.nv=1;world.gravityAndTimestep={0,0,-9.81f,.001f};
 MRArticulationGPU a{};a.rootBody=0;a.rootType=MR_ROOT_FIXED;a.firstBody=0;a.bodyCount=2;
 a.firstJoint=0;a.jointCount=1;a.nq=1;a.nv=1;
 MRBodyPropertiesGPU bodies[2]{};
 for(unsigned i=0;i<2;i++){bodies[i].articulationIndex=0;bodies[i].parentBody=i?0:MR_INVALID_INDEX;
  bodies[i].inboundJoint=i?0:MR_INVALID_INDEX;bodies[i].motionType=MR_MOTION_DYNAMIC;
  bodies[i].massAndInverseMass={.01f,100.f,0,0};
  bodies[i].inertiaRow0={.001f,0,0,0};bodies[i].inertiaRow1={0,.001f,0,0};bodies[i].inertiaRow2={0,0,.001f,0};}
 MRJointDescriptorGPU joint{};joint.parentBody=0;joint.childBody=1;joint.jointType=MR_JOINT_PRISMATIC;
 joint.nq=1;joint.nv=1;joint.axis0={0,0,1,0};joint.parentRotation={0,0,0,1};joint.childRotation={0,0,0,1};
 MRDofPropertiesGPU dof{};dof.articulationIndex=0;dof.jointIndex=0;
 MRABADispatchGPU dispatch{};dispatch.environmentCount=1;dispatch.flags=MR_ABA_HAS_BODY_WRENCHES;
 dispatch.qStride=1;dispatch.vStride=1;dispatch.effortStride=1;dispatch.wrenchStride=2;
 dispatch.accelerationStride=1;dispatch.nextVStride=1;dispatch.nextQStride=1;
 auto run=[&](float fx,float fz,float tx,float ty,float effort,float q,float expectedRx,float expectedRy,float expectedRz){
  MRABABodyWrenchGPU w[2]{};w[1].force={fx,0,fz,0};w[1].torque={tx,ty,0,0};float v=0,acc=0,nv=0,nq=0;
  MRABAStatusGPU st{};MRABABodyWrenchGPU reaction{};
  auto buffer=[&](const void* bytes,std::size_t n){id<MTLBuffer>b=[dev newBufferWithBytes:bytes length:n options:MTLResourceStorageModeShared];check(b!=nil,"buffer failed");return b;};
  id<MTLBuffer> bs[15]={buffer(&world,sizeof(world)),buffer(&a,sizeof(a)),buffer(&joint,sizeof(joint)),buffer(&dof,sizeof(dof)),
    buffer(bodies,sizeof(bodies)),buffer(&dispatch,sizeof(dispatch)),buffer(&q,sizeof(q)),buffer(&v,sizeof(v)),buffer(&effort,sizeof(effort)),
    buffer(w,sizeof(w)),buffer(&acc,sizeof(acc)),buffer(&nv,sizeof(nv)),buffer(&nq,sizeof(nq)),buffer(&st,sizeof(st)),buffer(&reaction,sizeof(reaction))};
  id<MTLCommandBuffer> cmd=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];check(enc!=nil,"no encoder");
  [enc setComputePipelineState:pipe];for(unsigned i=0;i<14;i++)[enc setBuffer:bs[i] offset:0 atIndex:i];
  [enc setBuffer:bs[14] offset:0 atIndex:24];[enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
  [enc endEncoding];[cmd commit];[cmd waitUntilCompleted];check(cmd.status==MTLCommandBufferStatusCompleted,"command failed");
  st=*static_cast<const MRABAStatusGPU*>(bs[13].contents);reaction=*static_cast<const MRABABodyWrenchGPU*>(bs[14].contents);
  check(st.code==MR_ABA_SUCCESS,"ABA failed");
  const float expectedA=-9.81f+(fz+effort)/.01f;
  check(std::abs(*static_cast<const float*>(bs[10].contents)-expectedA)<1e-4f,"acceleration mismatch");
  if(std::abs(reaction.torque.x-expectedRx)>1e-6f || std::abs(reaction.torque.y-expectedRy)>1e-6f || std::abs(reaction.torque.z-expectedRz)>1e-6f){
   std::cerr<<"reaction "<<reaction.torque.x<<","<<reaction.torque.y<<","<<reaction.torque.z<<" expected "<<expectedRx<<","<<expectedRy<<","<<expectedRz<<"\n";throw std::runtime_error("torque mismatch");}
  check(std::abs(reaction.force.x+fx)<1e-6f,"support force mismatch");
 };
 run(0,0,0,0,0,0,0,0,0);
 run(0,-.15f,.003f,-.002f,0,0,-.003f,.002f,0);
 run(0,.2f,-.004f,.001f,0,.04f,.004f,-.001f,0);
 run(.1f,0,.003f,-.002f,0,.04f,-.003f,.002f-.04f*.1f,0);
 std::cout<<"four dedicated ABA support-reaction cases passed\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}}
