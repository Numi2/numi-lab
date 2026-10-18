#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include <fstream>
#include <iterator>
#include <iostream>
#include <string>
#include <cmath>
int main(int argc,char**argv){@autoreleasepool{
 if(argc!=2)return 2;std::ifstream f(argv[1]);std::string source{std::istreambuf_iterator<char>(f),{}};
 source+=R"(
kernel void test_exact_bias(device const MRJointDescriptorGPU* j [[buffer(0)]],
device const MRArticulatedBodyPoseGPU* p [[buffer(1)]],device float4* out [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i)return;float3 a1,w1,a2,w2,ap,wp,az,wz;
 bool ok=standVelocityProductBiasChild(j[0],p[0],p[1],float3(0),float3(0,0,1),float3(0),float3(0),0,1,w1,a1);
 ok=ok&&standVelocityProductBiasChild(j[1],p[1],p[2],float3(0,0,1),float3(0,0,3),a1,w1,1.57079632679f,2,w2,a2);
 ok=ok&&standVelocityProductBiasChild(j[2],p[0],p[1],float3(0,0,1),float3(0,0,1),float3(0),float3(0),1,2,wp,ap);
 ok=ok&&standVelocityProductBiasChild(j[1],p[1],p[2],float3(0),float3(0),float3(0),float3(0),1.57079632679f,0,wz,az);
 out[0]=float4(a1,ok?1:0);out[1]=float4(a2,dot(float3(-1,1,0),a2));out[2]=float4(ap,-a2.x);out[3]=float4(az,length(wz));
})";
 auto dev=MTLCreateSystemDefaultDevice();NSError*err=nil;MTLCompileOptions*opt=[MTLCompileOptions new];opt.fastMathEnabled=NO;opt.languageVersion=MTLLanguageVersion4_0;
 auto lib=[dev newLibraryWithSource:@(source.c_str()) options:opt error:&err];if(!lib){std::cerr<<err.localizedDescription.UTF8String;return 1;}
 auto pipe=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"test_exact_bias"] error:&err];if(!pipe){std::cerr<<err.localizedDescription.UTF8String;return 1;}
 MRJointDescriptorGPU js[3]{};for(auto&j:js){j.jointType=MR_JOINT_REVOLUTE;j.nq=j.nv=1;j.axis0={0,0,1,0};j.parentRotation=j.childRotation={0,0,0,1};}js[0].childAnchor={-1,0,0,0};js[1].childAnchor={0,-1,0,0};js[2].jointType=MR_JOINT_PRISMATIC;js[2].axis0={1,0,0,0};
 MRArticulatedBodyPoseGPU ps[3]{};for(auto&p:ps)p.orientation={0,0,0,1};ps[1].position={1,0,0,1};ps[2].position={1,1,0,1};
 auto jb=[dev newBufferWithBytes:js length:sizeof(js) options:MTLResourceStorageModeShared];auto pb=[dev newBufferWithBytes:ps length:sizeof(ps) options:MTLResourceStorageModeShared];auto ob=[dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
 auto cb=[[dev newCommandQueue] commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipe];[e setBuffer:jb offset:0 atIndex:0];[e setBuffer:pb offset:0 atIndex:1];[e setBuffer:ob offset:0 atIndex:2];[e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];
 if(cb.status!=MTLCommandBufferStatusCompleted)return 1;auto o=(float*)ob.contents;float expected[16]={-1,0,0,1,-1,-9,0,-8,-1,4,0,1,0,0,0,0};bool pass=true;for(int i=0;i<16;i++){if(std::abs(o[i]-expected[i])>1e-5f){pass=false;std::cerr<<i<<":"<<o[i]<<" expected"<<expected[i]<<"\n";}}
 std::cout<<(pass?"PASS":"FAIL")<<" actual stand GPU helper: two-link Coriolis(-8,+1), rotating prismatic(-1,+4), zero-speed bias\n";return pass?0:1;
}}

