#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
int main(int argc,char**argv){ @autoreleasepool { try {
 constexpr unsigned N=10000;std::mt19937 rng(0xc0a15a7);std::uniform_real_distribution<float> uni(0,1),unit(-1,1);std::normal_distribution<float> normal(0,1);
 std::vector<MRShapeGPU> s(2*N);std::vector<MRProjectedColliderGPU> p(2*N);std::vector<mr_float4> output(N);
 auto makeq=[&](){mr_float4 q{normal(rng),normal(rng),normal(rng),normal(rng)};float l=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);q.x/=l;q.y/=l;q.z/=l;q.w/=l;return q;};
 for(unsigned i=0;i<N;i++){
  for(int j=0;j<2;j++){auto& x=s[2*i+j];x.shapeType=MR_SHAPE_BOX;x.dimensions={float(std::exp(std::log(.001f)+uni(rng)*std::log(700.f))),float(std::exp(std::log(.001f)+uni(rng)*std::log(700.f))),float(std::exp(std::log(.001f)+uni(rng)*std::log(700.f))),0};x.contactRestAndBoundingRadius.z=std::sqrt(x.dimensions.x*x.dimensions.x+x.dimensions.y*x.dimensions.y+x.dimensions.z*x.dimensions.z);p[2*i+j].rotation=makeq();}
  auto& a=p[2*i];auto& b=p[2*i+1];a.centerAndRadius={unit(rng)*10,unit(rng)*10,unit(rng)*10,0};float scale=(s[2*i].contactRestAndBoundingRadius.z+s[2*i+1].contactRestAndBoundingRadius.z);float fac=(i%4==0)?0.05f:(i%4==1)?0.5f:(i%4==2)?1.f:2.f;b.centerAndRadius={a.centerAndRadius.x+unit(rng)*scale*fac,a.centerAndRadius.y+unit(rng)*scale*fac,a.centerAndRadius.z+unit(rng)*scale*fac,0};if(i%8==0)b.rotation=a.rotation;
 }
 id<MTLDevice> dev=MTLCreateSystemDefaultDevice();NSError* err=nil;id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&err];if(!lib)throw std::runtime_error("library");id<MTLComputePipelineState> pipe=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"random_sat"] error:&err];if(!pipe)throw std::runtime_error("pipeline");id<MTLCommandQueue> queue=[dev newCommandQueue];id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];id<MTLBuffer> sb=[dev newBufferWithBytes:s.data() length:s.size()*sizeof(s[0]) options:MTLResourceStorageModeShared],pb=[dev newBufferWithBytes:p.data() length:p.size()*sizeof(p[0]) options:MTLResourceStorageModeShared],ob=[dev newBufferWithLength:output.size()*sizeof(output[0]) options:MTLResourceStorageModeShared],db=[dev newBufferWithLength:256 options:MTLResourceStorageModeShared];[enc setComputePipelineState:pipe];[enc setBuffer:sb offset:0 atIndex:0];[enc setBuffer:pb offset:0 atIndex:1];[enc setBuffer:ob offset:0 atIndex:2];[enc setBuffer:db offset:0 atIndex:3];[enc dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("gpu failure");std::memcpy(output.data(),ob.contents,output.size()*sizeof(output[0]));
 std::cout<<std::setprecision(9);for(unsigned i=0;i<N;i++){std::cout<<i;for(int j=0;j<2;j++){auto x=s[2*i+j].dimensions;auto c=p[2*i+j].centerAndRadius;auto q=p[2*i+j].rotation;std::cout<<' '<<x.x<<' '<<x.y<<' '<<x.z<<' '<<c.x<<' '<<c.y<<' '<<c.z<<' '<<q.x<<' '<<q.y<<' '<<q.z<<' '<<q.w;}auto r=output[i];std::cout<<' '<<r.x<<' '<<r.y<<' '<<r.z<<' '<<r.w<<'\n';}
 return 0;} catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}}
