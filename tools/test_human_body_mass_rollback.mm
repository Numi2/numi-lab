#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include "metalrobo/numi_human_stand_gpu.h"
#include <fstream>
#include <iterator>
#include <string>
#include <iostream>
#include <cstring>
int main(int argc,char**argv){@autoreleasepool{
 if(argc!=2)return 2;
 std::ifstream file(argv[1]);std::string source{std::istreambuf_iterator<char>(file),{}};
 auto dev=MTLCreateSystemDefaultDevice();NSError*err=nil;
 MTLCompileOptions*options=[MTLCompileOptions new];options.fastMathEnabled=NO;options.languageVersion=MTLLanguageVersion4_0;
 auto lib=[dev newLibraryWithSource:@(source.c_str()) options:options error:&err];
 if(!lib){std::cerr<<err.localizedDescription.UTF8String;return 1;}
 auto pipe=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mr_numi_human_stand_reconcile"] error:&err];
 if(!pipe){std::cerr<<err.localizedDescription.UTF8String;return 1;}
 id<MTLBuffer> b[18];
 for(int i=0;i<18;++i){b[i]=[dev newBufferWithLength:65536 options:MTLResourceStorageModeShared];std::memset(b[i].contents,0,65536);}
 MRBodyPropertiesGPU accepted[3]{},candidate[3]{};
 for(unsigned i=0;i<3;++i){
  accepted[i].massAndInverseMass={float(5+i),1.0f/float(5+i),0,0};
  accepted[i].centerOfMass={.1f*float(i+1),.2f,.3f,0};
  accepted[i].inertiaRow0={float(2+i),.1f,.2f,0};
  accepted[i].inertiaRow1={.1f,float(3+i),.3f,0};
  accepted[i].inertiaRow2={.2f,.3f,float(4+i),0};
  accepted[i].inverseInertiaRow0={.5f,.01f,.02f,0};
  accepted[i].inverseInertiaRow1={.01f,.4f,.03f,0};
  accepted[i].inverseInertiaRow2={.02f,.03f,.3f,0};
  candidate[i]=accepted[i];
  candidate[i].massAndInverseMass={float(4+i),1.0f/float(4+i),0,0};
  candidate[i].inertiaRow0.x*=.8f;candidate[i].inertiaRow1.y*=.8f;candidate[i].inertiaRow2.z*=.8f;
  candidate[i].inverseInertiaRow0.x/=.8f;candidate[i].inverseInertiaRow1.y/=.8f;candidate[i].inverseInertiaRow2.z/=.8f;
 }
 auto queue=[dev newCommandQueue];
 for(int mode=0;mode<3;++mode){
  std::memcpy(b[15].contents,candidate,sizeof(candidate));std::memcpy(b[16].contents,accepted,sizeof(accepted));
  MRNumiHumanStandStatusGPU status{};status.environment=0;status.completedSteps=1;
  status.code=mode==0?MR_NUMI_HUMAN_STAND_SUCCESS:MR_NUMI_HUMAN_STAND_INVALID_DISPATCH;
  std::memcpy(b[5].contents,&status,sizeof(status));
  mr_uint4 shape={1,0,0,0},strides={0,0,0,0},transaction={mode==2?0u:1u,3,0,0};
  auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipe];
  for(int i=2;i<17;++i)[e setBuffer:b[i] offset:0 atIndex:i];
  [e setBytes:&shape length:sizeof(shape) atIndex:0];[e setBytes:&strides length:sizeof(strides) atIndex:1];
  [e setBytes:&transaction length:sizeof(transaction) atIndex:17];
  [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
  [e endEncoding];[cb commit];[cb waitUntilCompleted];
  if(cb.status!=MTLCommandBufferStatusCompleted){std::cerr<<"GPU failure\n";return 1;}
  const auto*expected=mode==1?accepted:candidate;
  if(std::memcmp(b[15].contents,expected,sizeof(accepted))){std::cerr<<"mass transaction mismatch mode "<<mode<<"\n";return 1;}
 }
 std::cout<<"PASS actual stand GPU reconciliation: three accepted masses/inertias retained, three rejected masses/inertias restored byte-for-byte, opt-out unchanged\n";
 return 0;
}}
