#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
int main(int argc,char**argv){@autoreleasepool{try{
 require(argc==3,"Usage: projection_probe METALLIB NEW_OUTPUT");require(!std::filesystem::exists(argv[2]),"Output must be new");
 constexpr unsigned count=128;std::ofstream out(argv[2],std::ios::binary);
 auto device=MTLCreateSystemDefaultDevice();require(device!=nil,"Metal unavailable");NSError* error=nil;
 auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];require(lib!=nil,"Missing library");
 auto function=[lib newFunctionWithName:@"mr_world_project_swept_colliders"];require(function!=nil,"Missing owner projection");
 auto pipeline=[device newComputePipelineStateWithFunction:function error:&error];require(pipeline!=nil,"Pipeline failed");auto queue=[device newCommandQueue];
 for(unsigned mode:{unsigned(MR_WORLD_CCD_SPECULATIVE),unsigned(MR_WORLD_CCD_HYBRID)}){
  MRMetalWorldContactDispatchGPU d{};d.environmentCount=count;d.shapeCount=2;d.bodyCount=1;d.bodyStateStride=1;d.articulationCount=1;d.vStride=1;
  d.ccdMode=mode;d.flags=MR_METAL_WORLD_CONTACT_HAS_FUTURE_KINEMATICS;d.timestepAndBias={.000625f,0,0,0};d.ccdParameters={1e-5f,1e-5f,1,1e4f};
  std::array<MRShapeGPU,2> shapes{};
  for(unsigned i=0;i<2;++i){auto&s=shapes[i];s.shapeType=MR_SHAPE_BOX;s.bodyIndex=0;s.localRotation={0,0,0,1};s.dimensions={.025f,.009f,.003f,0};s.contactRestAndBoundingRadius={.0005f,0,.02674f,0};s.flags=i?MR_SHAPE_FLAG_ENABLE_CCD:0;}
  MRArticulationGPU articulation{};articulation.bodyCount=1;articulation.nv=1;
  std::vector<MRBodyStateGPU>bodies(count);std::vector<MRArticulatedBodyPoseGPU>future(count);std::vector<float>v(count);
  std::vector<MRCCDEventStateGPU>events(count);std::vector<MRMetalWorldContactStatusGPU>status(count);
  for(unsigned i=0;i<count;++i){
   bodies[i].position={.1f+i*.001f,.02f,.3f,1};bodies[i].orientation={0,0,0,1};bodies[i].flagsAndIndices[0]=MR_MOTION_DYNAMIC;bodies[i].flagsAndIndices[1]=0;
   future[i].position={bodies[i].position.x+.003f,.025f,.302f,1};float a=.02f+i*.001f;future[i].orientation={0,std::sin(a/2),0,std::cos(a/2)};
   v[i]=.5f+i*.001f;events[i].time={0,.000625f,0,.000625f};status[i].environment=i;
  }
  std::vector<MRProjectedColliderGPU>projected(count*2),predicted(count*2);const MRGeometryHeaderGPU geometry{};
  auto buffer=[&](const void*p,std::size_t n){auto b=[device newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];require(b!=nil,"Allocation failed");return b;};
  std::array<id<MTLBuffer>,12>buffers{buffer(&d,sizeof(d)),buffer(shapes.data(),sizeof(shapes)),buffer(&articulation,sizeof(articulation)),
   buffer(bodies.data(),bodies.size()*sizeof(bodies[0])),buffer(bodies.data(),bodies.size()*sizeof(bodies[0])),buffer(future.data(),future.size()*sizeof(future[0])),buffer(v.data(),v.size()*4),
   buffer(&geometry,sizeof(geometry)),buffer(projected.data(),projected.size()*sizeof(projected[0])),buffer(predicted.data(),predicted.size()*sizeof(predicted[0])),
   buffer(status.data(),status.size()*sizeof(status[0])),buffer(events.data(),events.size()*sizeof(events[0]))};
  auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline];for(unsigned i=0;i<buffers.size();++i)[encoder setBuffer:buffers[i] offset:0 atIndex:i];
  [encoder dispatchThreads:MTLSizeMake(count*2,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];require(command.status==MTLCommandBufferStatusCompleted,"GPU failed");
  const auto*s=static_cast<const MRMetalWorldContactStatusGPU*>(buffers[10].contents);for(unsigned i=0;i<count;++i)require(s[i].code==MR_STEP_SUCCESS,"Projection rejected");
  out.write(static_cast<const char*>(buffers[8].contents),projected.size()*sizeof(projected[0]));out.write(static_cast<const char*>(buffers[9].contents),predicted.size()*sizeof(predicted[0]));
 }
 require(bool(out),"Output failed");std::cout<<"{\"status\":\"executed\",\"conditions\":128,\"paired_shape_flags\":2,\"modes\":2}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}}
