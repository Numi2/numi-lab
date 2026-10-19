#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include "numi/matter/shared.h"
struct F4{float x,y,z,w;};
static void req(bool b,const char*s){if(!b)throw std::runtime_error(s);}
static void set3(nm_float4&v,float x,float y,float z){v=nm_float4{x,y,z,0.f};}
static const char* extra=R"MSL(
namespace numi_matter_metal {
kernel void nm_vt_exact_trace(device const NMDeformableVTCCDAuditGPU* cases [[buffer(0)]],
 device float4* out [[buffer(1)]],uint idx [[thread_position_in_grid]]) {
 if(idx)return; const auto c=cases[0]; const float th=c.parameters.x;
 const float3 p0=c.start[0].xyz,p1=c.finish[0].xyz;
 const float3 a0=c.start[3].xyz,a1=c.finish[3].xyz;
 const float3 b0=c.start[4].xyz,b1=c.finish[4].xyz;
 const float3 d0=c.start[5].xyz,d1=c.finish[5].xyz;
 float t=0,sep=INFINITY;float3 n=float3(0),bary=float3(0),pt=float3(0);bool cert=false;
 bool hit=vertexTriangleCCD(p0,p1,a0,a1,b0,b1,d0,d1,th,t,n,bary,pt,sep,cert);
 out[0]=float4(hit?1.0f:0.0f,cert?1.0f:0.0f,t,sep);
 uint it=0;float tt=0,ss=INFINITY;float3 nn=float3(0),bb=float3(0),pp=float3(0);bool cc=false;
 device float4* trace=out+3;
 bool hh=vertexTriangleCCDBudgetCore<16u,true,true>(p0,p1,a0,a1,b0,b1,d0,d1,th,
   trace,0u,16u,it,tt,nn,bb,pp,ss,cc);
 out[1]=float4(hh?1.0f:0.0f,cc?1.0f:0.0f,tt,ss);
 uint wi=0;float wt=0,ws=INFINITY;float3 wn=float3(0),wb=float3(0),wp=float3(0);bool wc=false;
 bool wh=vertexTriangleCCDWithSeparatingPlane(p0,p1,a0,a1,b0,b1,d0,d1,th,wt,wn,wb,wp,ws,wc);
 out[2]=float4(wh?1.0f:0.0f,wc?1.0f:0.0f,wt,ws);
}
}
)MSL";
int main(int argc,const char**argv){@autoreleasepool{try{
 req(argc==2,"usage: vt_trace <MatterSource.metal>");
 NMDeformableVTCCDAuditGPU c{};c.parameters.x=9.99999975e-6f;
 set3(c.start[0],.0655745789f,-.224834442f,.0408293419f);
 set3(c.finish[0],.0655745789f,-.224834442f,.0408293419f);
 set3(c.start[3],.0656245798f,-.224834442f,.0408293381f);set3(c.finish[3],.0654995814f,-.224834442f,.0408293344f);
 set3(c.start[4],.0671207979f,-.218933985f,.0345707946f);set3(c.finish[4],.0669957921f,-.218933985f,.0345707908f);
 set3(c.start[5],.0676468164f,-.216995493f,.0438255593f);set3(c.finish[5],.0675218254f,-.216995493f,.0438255556f);
 std::ifstream f(argv[1],std::ios::binary);req(bool(f),"open MSL");std::string src{std::istreambuf_iterator<char>(f),{}};src+=extra;
 NSString*msl=[[NSString alloc]initWithBytes:src.data() length:src.size() encoding:NSUTF8StringEncoding];req(msl!=nil,"MSL UTF8");
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();req(d!=nil,"Metal device");NSError*e=nil;id<MTLLibrary>lib=[d newLibraryWithSource:msl options:nil error:&e];req(lib!=nil,e?[[e localizedDescription]UTF8String]:"MSL compile");
 id<MTLFunction>fn=[lib newFunctionWithName:@"numi_matter_metal::nm_vt_exact_trace"];req(fn!=nil,"kernel");id<MTLComputePipelineState>pipe=[d newComputePipelineStateWithFunction:fn error:&e];req(pipe!=nil,e?[[e localizedDescription]UTF8String]:"pipeline");
 id<MTLBuffer>ib=[d newBufferWithBytes:&c length:sizeof(c) options:MTLResourceStorageModeShared];
 id<MTLBuffer>ob=[d newBufferWithLength:19*sizeof(F4) options:MTLResourceStorageModeShared];id<MTLCommandQueue>q=[d newCommandQueue];id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];req(ib&&ob&&q&&cb&&enc,"resources");
 [enc setComputePipelineState:pipe];[enc setBuffer:ib offset:0 atIndex:0];[enc setBuffer:ob offset:0 atIndex:1];[enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];req(cb.status==MTLCommandBufferStatusCompleted,"dispatch");
 const auto*r=(const F4*)ob.contents;std::cout.precision(10);for(int i=0;i<3;i++)std::cout<<"summary "<<i<<" "<<r[i].x<<" "<<r[i].y<<" "<<r[i].z<<" "<<r[i].w<<"\n";for(int i=0;i<16;i++)std::cout<<"trace "<<i<<" "<<r[i+3].x<<" "<<r[i+3].y<<" "<<r[i+3].z<<" "<<r[i+3].w<<"\n";
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}}return 0;}
