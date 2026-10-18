
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include "metalrobo/numi_human_stand_gpu.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <stdexcept>
struct MRHumanSeparationGPU { mr_uint4 bodies,strides; mr_float4 parentAnchor; mr_uint4 event,rows[6]; };
static void check(bool b,const char* m){if(!b)throw std::runtime_error(m);}
static std::string read(const char* p){std::ifstream f(p);check(f.good(),"missing input");return {std::istreambuf_iterator<char>(f),{}};}
static std::string structure(const std::string& h,const char* n){
 auto end=h.find(std::string("} ")+n+";");check(end!=h.npos,"missing type");
 auto begin=h.rfind("typedef struct ",end);return h.substr(begin,end+std::strlen(n)+3-begin);
}
int main(int argc,char** argv){@autoreleasepool{try{
 check(argc==4,"usage: test KERNEL ENGINE_HEADER STAND_HEADER");
 auto engine=read(argv[2]);auto source=std::string("#include <metal_stdlib>\nusing namespace metal;\n#define MR_ALIGN16 alignas(16)\nusing mr_u32=uint;using mr_uint4=uint4;using mr_float4=float4;\n")+
 structure(engine,"MRArticulatedBodyPoseGPU")+"\n"+structure(engine,"MRArticulatedPointWorldGPU")+"\n"+
 structure(read(argv[3]),"MRNumiHumanStandContactGPU")+"\n"+read(argv[1]);
 auto device=MTLCreateSystemDefaultDevice();check(device!=nil,"Metal unavailable");NSError* error=nil;
 auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
 options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
 options.languageVersion=MTLLanguageVersion4_0;
 auto lib=[device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:options error:&error];
 if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
 auto pipe=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_separation_constraints"] error:&error];
 check(pipe!=nil,"pipeline unavailable");auto queue=[device newCommandQueue];
 constexpr unsigned nv=6,pointCount=14,probe=6;
 MRHumanSeparationGPU cfg{};cfg.bodies={10,11,10,probe};
 cfg.strides={2,pointCount,pointCount*3*nv,nv};cfg.event={100,99,1,0};cfg.parentAnchor={.01f,.02f,.03f,0};
 MRArticulatedBodyPoseGPU poses[2]{};poses[0].position={2,3,4,1};poses[1].position={2.01f,2.98f,4.03f,1};
 poses[0].orientation=poses[1].orientation={0,0,0,1};
 mr_float4 lows[2]={{.0000001f,0,0,0},{-.0000001f,0,0,0}};
 MRArticulatedPointWorldGPU points[pointCount]{};mr_float4 pointLow[pointCount]{};
 float jac[pointCount*3*nv]{};MRNumiHumanStandContactGPU contacts[7]{};
 for(unsigned axis=0;axis<3;++axis){
  jac[(3*probe+axis)*nv+axis]=1;
  jac[(3*(probe+4)+axis)*nv+axis]=1;
  jac[(3*(probe+4)+axis)*nv+3+axis]=1;
 }
 for(unsigned probeAxis=0;probeAxis<3;++probeAxis)
  for(unsigned axis=0;axis<3;++axis)
   jac[(3*(probe+1+probeAxis)+axis)*nv+axis]=1;
 jac[(3*(probe+1)+1)*nv+5]=1; // d(R e_x)/d yaw = +e_y
 jac[(3*(probe+2)+0)*nv+5]=-1; // d(R e_y)/d yaw = -e_x
 for(unsigned r=0;r<6;++r){
  cfg.rows[r]={r,r,r/2,r%2};contacts[r].bodyIndex=11;contacts[r].pointQueryIndex=r;
  contacts[r].frictionSlopAndStabilization={0,0,.2f,0};
 }contacts[6].bodyIndex=1234;contacts[6].planeNormal={1,0,0,0};
 auto alloc=[&](const void* a,NSUInteger n){return [device newBufferWithBytes:a length:n options:MTLResourceStorageModeShared];};
 id<MTLBuffer> buffers[]={alloc(&cfg,sizeof(cfg)),alloc(poses,sizeof(poses)),alloc(lows,sizeof(lows)),alloc(points,sizeof(points)),alloc(pointLow,sizeof(pointLow)),alloc(jac,sizeof(jac)),alloc(contacts,sizeof(contacts))};
 auto run=[&]{
  std::memcpy(buffers[0].contents,&cfg,sizeof(cfg));
  auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipe];
  for(unsigned i=0;i<7;++i)[e setBuffer:buffers[i] offset:0 atIndex:i];
  [e dispatchThreads:MTLSizeMake(6,1,1) threadsPerThreadgroup:MTLSizeMake(6,1,1)];
  [e endEncoding];[cb commit];[cb waitUntilCompleted];
  check(cb.status==MTLCommandBufferStatusCompleted,"Metal execution failed");
 };
 auto* outP=static_cast<MRArticulatedPointWorldGPU*>(buffers[3].contents);
 auto* outLow=static_cast<mr_float4*>(buffers[4].contents);
 auto* outJ=static_cast<float*>(buffers[5].contents);
 auto* outC=static_cast<MRNumiHumanStandContactGPU*>(buffers[6].contents);
 auto component=[](const mr_float4& v,unsigned axis){return axis==0?v.x:axis==1?v.y:v.z;};
 auto gap=[&](unsigned r){double g=0;for(unsigned a=0;a<3;++a)g+=(double(component(outP[r].position,a))+component(outLow[r],a)-component(outC[r].planePoint,a))*component(outC[r].planeNormal,a);return g;};
 run();
 for(unsigned r=0;r<6;++r) {
  unsigned axis=r/2;float sign=r%2?-1.f:1.f;
  check(std::abs(gap(r)-sign*(component(poses[1].position,axis)-component(poses[0].position,axis)+component(lows[1],axis)-component(lows[0],axis)-component(cfg.parentAnchor,axis)))<2e-7,"relative gap incorrect");
  for(unsigned a=0;a<3;++a)for(unsigned d=0;d<nv;++d)
   check(std::abs(outJ[(3*r+a)*nv+d]-((d==3+a?1.f:0.f)+(d==5?(a==0?.02f:a==1?-.01f:0.f):0.f)))<1e-7,"relative Jacobian not internal");
  check(gap(r)*gap(r^1)<=0,"bilateral row pair not opposed");
 }
 check(std::memcmp(outJ+3*probe*nv,jac+3*probe*nv,(pointCount-probe)*3*nv*sizeof(float))==0,"body probes overwritten");
 check(std::memcmp(outC+6,contacts+6,sizeof(contacts[6]))==0,"unrelated support overwritten");
 auto before=std::string(static_cast<char*>(buffers[6].contents),sizeof(contacts));
 cfg.event.y=100;run();for(unsigned r=0;r<6;++r)check(gap(r)>999999,"release left an active constraint");
 cfg.event.y=99;run();check(std::memcmp(before.data(),buffers[6].contents,sizeof(contacts))==0,"rejected attempt could not regenerate pre-cut constraints");
 cfg.event.y=100;run();auto released=std::string(static_cast<char*>(buffers[6].contents),sizeof(contacts));run();
 check(std::memcmp(released.data(),buffers[6].contents,sizeof(contacts))==0,"release not idempotent");
 // No source pose, velocity, mass, or source Jacobian may change at release.
 check(std::memcmp(buffers[1].contents,poses,sizeof(poses))==0,"release changed body state");
 check(std::memcmp(buffers[2].contents,lows,sizeof(lows))==0,"release changed compensated state");
 std::cout<<"PASS internal relative constraints, opposed gaps, zero common-translation reaction, event boundary, replay regeneration, isolation on "<<device.name.UTF8String<<"\n";
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}}
