#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <array>
int main(int argc,char**argv) {
 @autoreleasepool {
 if(argc!=2)return 2;
 std::ifstream f(argv[1]);std::string h{std::istreambuf_iterator<char>(f),{}};
 if(h.empty())return 2;
 auto dev=MTLCreateSystemDefaultDevice();if(!dev)return 2;
 std::string source="#include <metal_stdlib>\nusing namespace metal;\n"+h+R"(
 kernel void boundary_test(device const float4* inputs [[buffer(0)]],
 device float4* outputs [[buffer(1)]],uint i [[thread_position_in_grid]]) {
 const float4 x=inputs[i]; const float dt=.002f, invMass=.5f;
 const float target=mrNumiHumanContactVelocityTarget(x.x,dt,.2f,x.z);
 const float target0=mrNumiHumanContactVelocityTarget(x.x,dt,.2f);
 const float impulse=max((target-x.y)/invMass,0.f);
 outputs[i]=float4(x.y+invMass*impulse,impulse,target,target0);
 })";
 MTLCompileOptions* opt=[MTLCompileOptions new];opt.fastMathEnabled=NO;opt.languageVersion=MTLLanguageVersion4_0;
 NSError* err=nil;auto lib=[dev newLibraryWithSource:@(source.c_str()) options:opt error:&err];
 if(!lib){std::cerr<<err.localizedDescription.UTF8String;return 1;}
 auto pipe=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"boundary_test"] error:&err];
 struct F{float gap,v,b,reserved;};
 // Touching stationary/moving planes, separating particle, speculative impact,
 // and translated copies test physical relative velocity and Galilean invariance.
 std::array<F,8> in{{{0,-1,0,0},{0,1,2,0},{0,3,2,0},{0,-3,-2,0},
 {.001f,-2,0,0},{.001f,5,7,0},{-.001f,0,0,0},{-.001f,7,7,0}}};
 auto ib=[dev newBufferWithBytes:in.data() length:sizeof(in) options:MTLResourceStorageModeShared];
 auto ob=[dev newBufferWithLength:8*16 options:MTLResourceStorageModeShared];
 auto cb=[[dev newCommandQueue] commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipe];[e setBuffer:ib offset:0 atIndex:0];[e setBuffer:ob offset:0 atIndex:1];[e dispatchThreads:MTLSizeMake(8,1,1) threadsPerThreadgroup:MTLSizeMake(8,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];
 if(cb.status!=MTLCommandBufferStatusCompleted)return 1;
 auto out=(const float(*)[4])ob.contents;
 auto near=[](float a,float b){return std::abs(a-b)<2e-5f;};
 bool pass=near(out[0][0],0)&&near(out[0][1],2)&&near(out[1][0],2)&&near(out[1][1],2)&&near(out[2][0],3)&&out[2][1]==0&&near(out[3][0],-2);
 pass=pass&&near(out[5][0]-7,out[4][0])&&near(out[5][1],out[4][1])&&near(out[7][0]-7,out[6][0])&&near(out[7][1],out[6][1]);
 for(unsigned i:{0u,4u,6u})pass=pass&&out[i][2]==out[i][3];
 std::cout<<(pass?"PASS":"FAIL")<<" actual Metal moving-boundary relative-velocity, unilateral separation, speculative contact, Galilean invariance, zero-boundary compatibility\n";
 return pass?0:1;
 }
}
