#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "numi/matter/shared.h"
using V4=std::array<float,4>;
void req(bool b,const char*s){if(!b)throw std::runtime_error(s);}
void set3(nm_float4&v,float x,float y,float z){v=nm_float4{x,y,z,0.f};}
struct TestCase {std::string name; NMDeformableVTCCDAuditGPU q{};};
TestCase makeCase(const char* name, std::array<std::array<float,3>,4> a,
                  std::array<std::array<float,3>,4> b, float thickness=9.99999975e-6f) {
 TestCase c; c.name=name; c.q.parameters.x=thickness;
 set3(c.q.start[0],a[0][0],a[0][1],a[0][2]); set3(c.q.finish[0],b[0][0],b[0][1],b[0][2]);
 set3(c.q.start[1],a[1][0],a[1][1],a[1][2]); set3(c.q.finish[1],b[1][0],b[1][1],b[1][2]);
 set3(c.q.start[3],a[2][0],a[2][1],a[2][2]); set3(c.q.finish[3],b[2][0],b[2][1],b[2][2]);
 set3(c.q.start[4],a[3][0],a[3][1],a[3][2]); set3(c.q.finish[4],b[3][0],b[3][1],b[3][2]);
 return c;
}
const char* extra=R"MSL(
namespace numi_matter_metal {
kernel void nm_contact_eeccd_composed_probe(
    device const NMDeformableVTCCDAuditGPU* cases [[buffer(0)]],
    device float4* results [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
 if(index>=count)return;
 const NMDeformableVTCCDAuditGPU c=cases[index]; const uint o=7u*index;
 const float th=c.parameters.x;
 float oldToi=0.0f,oldSep=INFINITY;float3 oldN=float3(0),oldP=float3(0);float2 oldST=float2(0);bool oldCert=false;
 const bool oldHit=edgeEdgeCCD(c.start[0].xyz,c.finish[0].xyz,c.start[1].xyz,c.finish[1].xyz,
   c.start[3].xyz,c.finish[3].xyz,c.start[4].xyz,c.finish[4].xyz,
   th,oldToi,oldN,oldST,oldP,oldSep,oldCert);
 float newToi=0.0f,newSep=INFINITY;float3 newN=float3(0),newP=float3(0);float2 newST=float2(0);bool newCert=false;
 const bool newHit=edgeEdgeCCDWithSeparatingPlane(c.start[0].xyz,c.finish[0].xyz,c.start[1].xyz,c.finish[1].xyz,
   c.start[3].xyz,c.finish[3].xyz,c.start[4].xyz,c.finish[4].xyz,
   th,newToi,newN,newST,newP,newSep,newCert);
 float planeToi=0.0f,planeSep=INFINITY;float3 planeN=float3(0),planeP=float3(0);float2 planeST=float2(0);bool planeCert=false;
 const bool planeHit=edgeEdgeCCD(c.start[0].xyz,c.finish[0].xyz,c.start[1].xyz,c.finish[1].xyz,
   c.start[3].xyz,c.finish[3].xyz,c.start[4].xyz,c.finish[4].xyz,
   th,planeToi,planeN,planeST,planeP,planeSep,planeCert,true);
 float3 endN=float3(0),endP=float3(0);float2 endST=float2(0);float endSep=INFINITY;
 const bool endpoint=endpointEdgeEdgeContact(c.start[0].xyz,c.finish[0].xyz,c.start[1].xyz,c.finish[1].xyz,
   c.start[3].xyz,c.finish[3].xyz,c.start[4].xyz,c.finish[4].xyz,th,endN,endST,endP,endSep);
 bool exactParity=oldHit==newHit && oldCert==newCert;
 if(oldCert && newCert) exactParity=exactParity &&
   as_type<uint>(oldToi)==as_type<uint>(newToi) && as_type<uint>(oldSep)==as_type<uint>(newSep) &&
   all(as_type<uint3>(oldN)==as_type<uint3>(newN)) &&
   all(as_type<uint2>(oldST)==as_type<uint2>(newST)) &&
   all(as_type<uint3>(oldP)==as_type<uint3>(newP));
 results[o+6]=float4(exactParity?1.0f:0.0f,oldCert?1.0f:0.0f,newCert?1.0f:0.0f,0.0f);
 results[o+0]=float4(oldHit?1.0f:0.0f,oldCert?1.0f:0.0f,oldToi,oldSep);
 results[o+1]=float4(newHit?1.0f:0.0f,newCert?1.0f:0.0f,newToi,newSep);
 results[o+2]=float4(planeHit?1.0f:0.0f,planeCert?1.0f:0.0f,planeToi,planeSep);
 results[o+3]=float4(endpoint?1.0f:0.0f,endSep,endST.x,endST.y);
 results[o+4]=float4(newN,newN.x*newN.x+newN.y*newN.y+newN.z*newN.z);
 results[o+5]=float4(newST,newP.x,newP.y);
}
}
)MSL";
int main(int argc,const char**argv){@autoreleasepool{try{
 req(argc==2,"usage: ccd_edgeedge_probe <candidate.metal>");
 std::vector<TestCase> cases;
 cases.push_back(makeCase("wall-step4-pair433-1057",
  {{{-0.0319312029f,0.0981000513f,-0.0180372987f},{-0.0319677852f,0.0879923105f,-0.0179396551f},{-0.0319150016f,0.0981093124f,-0.0180420838f},{-0.0342148021f,0.0981037691f,-0.0182886906f}}},
  {{{-0.0319312029f,0.0981000513f,-0.0180372987f},{-0.0319680944f,0.087954022f,-0.0179257914f},{-0.0318989381f,0.0981044099f,-0.0180431698f},{-0.0341951363f,0.0980955809f,-0.0182928126f}}}));
 cases.push_back(makeCase("static-clear",
  {{{-0.1f,0.f,0.f},{0.1f,0.f,0.f},{0.f,0.1f,0.05f},{0.f,-0.1f,0.05f}}},
  {{{-0.1f,0.f,0.f},{0.1f,0.f,0.f},{0.f,0.1f,0.05f},{0.f,-0.1f,0.05f}}}));
 cases.push_back(makeCase("transient-crossing-endpoint-clear",
  {{{-0.1f,0.f,0.f},{0.1f,0.f,0.f},{0.f,-0.1f,0.01f},{0.f,0.1f,0.01f}}},
  {{{-0.1f,0.f,0.f},{0.1f,0.f,0.f},{0.f,-0.1f,-0.01f},{0.f,0.1f,-0.01f}}}));
 cases.push_back(makeCase("endpoint-contact",
  {{{0.f,0.f,0.f},{0.1f,0.f,0.f},{0.03f,0.03f,0.f},{0.03f,0.1f,0.f}}},
  {{{0.f,0.f,0.f},{0.1f,0.f,0.f},{0.03f,0.000005f,0.f},{0.03f,0.1f,0.f}}}));
 std::vector<NMDeformableVTCCDAuditGPU> inputs; for(auto&c:cases)inputs.push_back(c.q);
 std::ifstream f(argv[1],std::ios::binary);req(bool(f),"open MSL");std::string src{std::istreambuf_iterator<char>(f),{}};src+=extra;
 NSString*msl=[[NSString alloc]initWithBytes:src.data() length:src.size() encoding:NSUTF8StringEncoding];req(msl!=nil,"MSL UTF8");
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();req(d!=nil,"Metal device");NSError*e=nil;id<MTLLibrary>lib=[d newLibraryWithSource:msl options:nil error:&e];req(lib!=nil,e?[[e localizedDescription]UTF8String]:"MSL compile");
 id<MTLFunction>fn=[lib newFunctionWithName:@"numi_matter_metal::nm_contact_eeccd_composed_probe"];req(fn!=nil,"kernel missing");id<MTLComputePipelineState>pipe=[d newComputePipelineStateWithFunction:fn error:&e];req(pipe!=nil,e?[[e localizedDescription]UTF8String]:"pipeline");
 uint32_t count=(uint32_t)inputs.size();std::vector<V4>outs(inputs.size()*7);
 id<MTLBuffer>ib=[d newBufferWithBytes:inputs.data() length:inputs.size()*sizeof(inputs[0]) options:MTLResourceStorageModeShared];id<MTLBuffer>ob=[d newBufferWithBytes:outs.data() length:outs.size()*sizeof(V4) options:MTLResourceStorageModeShared];id<MTLBuffer>cbf=[d newBufferWithBytes:&count length:sizeof(count) options:MTLResourceStorageModeShared];
 id<MTLCommandQueue>q=[d newCommandQueue];id<MTLCommandBuffer>cb=[q commandBuffer];id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];req(ib&&ob&&cbf&&q&&cb&&enc,"Metal resources");[enc setComputePipelineState:pipe];[enc setBuffer:ib offset:0 atIndex:0];[enc setBuffer:ob offset:0 atIndex:1];[enc setBuffer:cbf offset:0 atIndex:2];[enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(count,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];req(cb.status==MTLCommandBufferStatusCompleted,cb.error?[[cb.error localizedDescription]UTF8String]:"GPU command");
 const auto*r=(const V4*)ob.contents;std::cout.precision(10);for(unsigned i=0;i<cases.size();i++){auto old=r[7*i],nw=r[7*i+1],pl=r[7*i+2],ep=r[7*i+3],n=r[7*i+4],v=r[7*i+5],par=r[7*i+6];std::cout<<cases[i].name<<" legacy="<<old[0]<<","<<old[1]<<","<<old[2]<<","<<old[3]<<" composed="<<nw[0]<<","<<nw[1]<<","<<nw[2]<<","<<nw[3]<<" plane="<<pl[0]<<","<<pl[1]<<","<<pl[2]<<","<<pl[3]<<" endpoint="<<ep[0]<<","<<ep[1]<<","<<ep[2]<<","<<ep[3]<<" legacyParity="<<par[0]<<" norm2="<<n[3]<<" params="<<v[0]<<","<<v[1]<<" point="<<v[2]<<","<<v[3]<<"\n";}
 return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}}}
