#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
struct Case {float scale,offset;mr_float4 half,q0,q1;float y0,y1,tol;};
mr_float4 f4(float a,float b,float c,float d=0){return {a,b,c,d};}
double support(mr_float4 q,mr_float4 e){double n=std::sqrt(double(q.x)*q.x+double(q.y)*q.y+double(q.z)*q.z+double(q.w)*q.w);double x=q.x/n,y=q.y/n,z=q.z/n,w=q.w/n;return e.x*std::abs(2*(x*y+z*w))+e.y*std::abs(1-2*(x*x+z*z))+e.z*std::abs(2*(y*z-x*w));}
int main(int argc,char**argv){@autoreleasepool{
 if(argc!=2)return 2;id<MTLDevice>d=MTLCreateSystemDefaultDevice();if(!d)return 3;NSError*err=nil;id<MTLLibrary>lib=[d newLibraryWithFile:[NSString stringWithUTF8String:argv[1]] error:&err];if(!lib){std::cerr<<err.localizedDescription.UTF8String;return 4;}
 id<MTLFunction>fn=[lib newFunctionWithName:@"numi_monotone_scalar_probe"];id<MTLComputePipelineState>pipe=[d newComputePipelineStateWithFunction:fn error:&err];if(!pipe){std::cerr<<err.localizedDescription.UTF8String;return 5;}
 constexpr int N=2048;std::mt19937_64 rng(0x20261108);std::uniform_real_distribution<double>U(0,1),S(-1,1);std::vector<Case> cases;std::vector<MRShapeGPU> shapes(2*N);std::vector<MRProjectedColliderGPU> start(2*N),end(2*N);std::vector<float> tol(N);
 auto quaternion=[&](){double a=S(rng),b=S(rng),c=S(rng),v=std::sqrt(a*a+b*b+c*c);if(v<1e-9){a=1;v=1;}double angle=3.141592653589793*S(rng),r=std::sin(angle/2)/v;return f4(a*r,b*r,c*r,std::cos(angle/2));};
 for(int i=0;i<N;++i){Case c{};c.scale=float(i%3==0?.1:(i%3==1?1:10));c.offset=float((i%8==0?10000:0)*c.scale);c.half=f4(float((.04+.2*U(rng))*c.scale),float((i%3==0?.0002:(.001+.03*U(rng)))*c.scale),float((.01+.04*U(rng))*c.scale));c.q0=quaternion();c.q1=quaternion();if(i%7==0)c.q1=f4(-c.q1.x,-c.q1.y,-c.q1.z,-c.q1.w);c.tol=1e-5f*c.scale;
  double gap=(i%5==0?1e-6:(i%5==1?-1e-6:(S(rng)*.05)))*c.scale;c.y0=float(c.offset+support(c.q0,c.half)+gap);c.y1=float(c.y0+S(rng)*.12*c.scale);cases.push_back(c);tol[i]=c.tol;
  auto&box=shapes[2*i];box.shapeType=MR_SHAPE_BOX;box.dimensions=c.half;box.contactRestAndBoundingRadius=f4(.001*c.scale,0,std::sqrt(c.half.x*c.half.x+c.half.y*c.half.y+c.half.z*c.half.z));auto&plane=shapes[2*i+1];plane.shapeType=MR_SHAPE_PLANE;plane.localRotation=f4(0,0,0,1);
  start[2*i].centerAndRadius=f4(0,c.y0,0);start[2*i].rotation=c.q0;end[2*i].centerAndRadius=f4(0,c.y1,0);end[2*i].rotation=c.q1;end[2*i].statusAndFlags.w=0x80000000u;
  start[2*i+1].centerAndRadius=end[2*i+1].centerAndRadius=f4(0,c.offset,0);start[2*i+1].rotation=end[2*i+1].rotation=f4(0,0,0,1);end[2*i+1].statusAndFlags.w=0x40000000u;
 }
 auto buffer=[&](const void*ptr,size_t len){return [d newBufferWithBytes:ptr length:len options:MTLResourceStorageModeShared];};id<MTLBuffer>b0=buffer(shapes.data(),shapes.size()*sizeof(shapes[0])),b1=buffer(start.data(),start.size()*sizeof(start[0])),b2=buffer(end.data(),end.size()*sizeof(end[0])),b3=buffer(tol.data(),tol.size()*sizeof(float)),b4=[d newBufferWithLength:N*sizeof(mr_float4) options:MTLResourceStorageModeShared];if(!b0||!b1||!b2||!b3||!b4)return 6;
 id<MTLCommandQueue>q=[d newCommandQueue];id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];[enc setComputePipelineState:pipe];[enc setBuffer:b0 offset:0 atIndex:0];[enc setBuffer:b1 offset:0 atIndex:1];[enc setBuffer:b2 offset:0 atIndex:2];[enc setBuffer:b3 offset:0 atIndex:3];[enc setBuffer:b4 offset:0 atIndex:4];[enc dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted){std::cerr<<cb.error.localizedDescription.UTF8String;return 7;}
 auto*out=(mr_float4*)b4.contents;std::cout<<std::setprecision(10)<<"{\"seed\":539365640,\"cases\":[";
 for(int i=0;i<N;++i){auto c=cases[i];auto o=out[i];if(i)std::cout<<",";std::cout<<"{\"id\":"<<i<<",\"scale\":"<<c.scale<<",\"offset\":"<<c.offset<<",\"half\":["<<c.half.x<<","<<c.half.y<<","<<c.half.z<<"],\"q0\":["<<c.q0.x<<","<<c.q0.y<<","<<c.q0.z<<","<<c.q0.w<<"],\"q1\":["<<c.q1.x<<","<<c.q1.y<<","<<c.q1.z<<","<<c.q1.w<<"],\"y0\":"<<c.y0<<",\"y1\":"<<c.y1<<",\"tolerance\":"<<c.tol<<",\"handled\":"<<int(o.x)<<",\"clear\":"<<int(o.y)<<",\"work\":"<<int(o.z)<<",\"lower\":"<<o.w<<"}";}
 std::cout<<"]}\n";
}}
