#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/numi_human_resting_rope_gpu.h"
#include "metalrobo/numi_human_stand_gpu.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
static void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
static float component(const mr_float4& v,unsigned a){return a==0?v.x:a==1?v.y:v.z;}
static std::string read(const char* path){std::ifstream f(path);require(f.good(),"source unavailable");return {std::istreambuf_iterator<char>(f),{}};}
static std::string structure(const std::string& h,const char* name){
 const auto end=h.find(std::string("} ")+name+";");require(end!=std::string::npos,"missing structure");
 const auto begin=h.rfind("typedef struct ",end);require(begin!=std::string::npos,"missing structure start");
 return h.substr(begin,end+std::strlen(name)+3-begin);
}
int main(int argc,char** argv){@autoreleasepool{try{
 require(argc==5,"usage: test_human_rope_support KERNEL ROPE_HEADER ENGINE_HEADER STAND_HEADER");
 const auto engine=read(argv[3]);
 const auto source=std::string("#include <metal_stdlib>\nusing namespace metal;\n#define MR_ALIGN16 alignas(16)\nusing mr_u32=uint;using mr_uint4=uint4;using mr_float4=float4;\n")+
 structure(engine,"MRArticulatedBodyPoseGPU")+"\n"+structure(engine,"MRArticulatedPointWorldGPU")+"\n"+
 structure(read(argv[4]),"MRNumiHumanStandContactGPU")+"\n"+structure(read(argv[2]),"MRHumanRestingRopeGPU")+"\n"+read(argv[1]);
 auto device=MTLCreateSystemDefaultDevice();require(device!=nil,"Metal unavailable");NSError* error=nil;
 auto lib=[device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:nil error:&error];
 if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String);
 auto pipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_human_resting_rope_support"] error:&error];require(pipeline!=nil,"pipeline unavailable");
 MRHumanRestingRopeGPU cfg{};cfg.indices={1,0,7,7};cfg.strides={1,1,5,60};cfg.counts={1,1,4,1};
 cfg.anchorAndLength={1.9f,3.2f,6.3f,2};cfg.bodyLocalSite={.2f,.1f,.3f,0};
 MRArticulatedBodyPoseGPU body{};body.position={2,3,4,1};const float s=std::sqrt(.5f);body.orientation={0,0,s,s};
 mr_float4 low{};MRArticulatedPointWorldGPU points[5]{};mr_float4 pointLow[5]{};float j[60]{};
 // Four rigid probes: COM and rotated unit X/Y/Z, four DOFs Tx/Ty/Tz/Rz.
 const float probeOffset[4][3]={{0,0,0},{0,1,0},{-1,0,0},{0,0,1}};
 for(unsigned p=0;p<4;++p)for(unsigned a=0;a<3;++a){j[(3*(p+1)+a)*4+a]=1;
  j[(3*(p+1)+a)*4+3]=a==0?-probeOffset[p][1]:a==1?probeOffset[p][0]:0;}
 MRNumiHumanStandContactGPU contacts[2]{};contacts[0].bodyIndex=99;contacts[1].bodyIndex=7;contacts[1].pointQueryIndex=0;
 contacts[1].frictionSlopAndStabilization={0,.002f,.2f,0};
 auto alloc=[&](const void* bytes,NSUInteger length){return [device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];};
 id<MTLBuffer> buffers[7]={alloc(&cfg,sizeof(cfg)),alloc(&body,sizeof(body)),alloc(&low,sizeof(low)),alloc(points,sizeof(points)),alloc(pointLow,sizeof(pointLow)),alloc(j,sizeof(j)),alloc(contacts,sizeof(contacts))};
 auto queue=[device newCommandQueue];auto run=[&]{std::memcpy(buffers[0].contents,&cfg,sizeof(cfg));auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipeline];for(unsigned i=0;i<7;++i)[e setBuffer:buffers[i] offset:0 atIndex:i];[e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,"GPU command failed");};
 auto* output=static_cast<MRArticulatedPointWorldGPU*>(buffers[3].contents);auto* planes=static_cast<MRNumiHumanStandContactGPU*>(buffers[6].contents);auto* outJ=static_cast<float*>(buffers[5].contents);
 auto near=[](float a,float b){return std::abs(a-b)<2e-6f;};
 run();require(near(output[0].position.x,1.9f)&&near(output[0].position.y,3.2f)&&near(output[0].position.z,4.3f),"rigid site not transported");
 require(near(outJ[3],-.2f)&&near(outJ[7],-.1f)&&near(outJ[11],0),"rigid site angular Jacobian wrong");
 auto gap=[&]{float g=0;for(unsigned a=0;a<3;++a)g+=(component(output[0].position,a)-component(planes[1].planePoint,a))*component(planes[1].planeNormal,a);return g;};
 require(near(gap(),0)&&near(planes[1].planeNormal.z,1),"taut rope plane mismatch");
 cfg.anchorAndLength.w=2.5f;run();require(near(gap(),.5f),"slack rope must have positive gap");
 cfg.anchorAndLength.w=1.5f;run();require(near(gap(),-.5f),"overextension must have negative gap");
 require(std::memcmp(&planes[0],&contacts[0],sizeof(contacts[0]))==0,"unrelated floor contact changed");
 require(std::memcmp(outJ+12,j+12,sizeof(j)-12*sizeof(float))==0,"body probes changed");
 require(std::memcmp(&planes[1].frictionSlopAndStabilization,&contacts[1].frictionSlopAndStabilization,sizeof(mr_float4))==0,"read-only friction changed");
 cfg.anchorAndLength={2.6f,3.8f,6.3f,2};run();
 float normal[3]={planes[1].planeNormal.x,planes[1].planeNormal.y,planes[1].planeNormal.z};
 const float velocity[4]={.3f,-.2f,.1f,.4f};float power=0;
 for(unsigned a=0;a<3;++a)for(unsigned d=0;d<4;++d)power+=normal[a]*outJ[a*4+d]*velocity[d];
 const double eps=1e-4;double r0=0,r1=0;
 for(unsigned a=0;a<3;++a){double v=0;for(unsigned d=0;d<4;++d)v+=outJ[a*4+d]*velocity[d];
 double delta=component(cfg.anchorAndLength,a)-component(output[0].position,a);r0+=delta*delta;r1+=(delta-eps*v)*(delta-eps*v);}
 require(std::abs((std::sqrt(r0)-std::sqrt(r1))/eps-power)<1e-5,"gap derivative / virtual work mismatch");
 const auto saved=planes[1];run();require(std::memcmp(&saved,&planes[1],sizeof(saved))==0,"restored accepted pose did not replay");
 auto* mutableContacts=static_cast<MRNumiHumanStandContactGPU*>(buffers[6].contents);mutableContacts[1].frictionSlopAndStabilization.x=.5f;run();
 require(!std::isfinite(planes[1].planeNormal.x),"frictional tether silently admitted");
 std::cout<<"PASS rope rigid transport, affine J, slack/taut gap, virtual work, isolation, replay, invalid friction on "<<device.name.UTF8String<<"\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}}
