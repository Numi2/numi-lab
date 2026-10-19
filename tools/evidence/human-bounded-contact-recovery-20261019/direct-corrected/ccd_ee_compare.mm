#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using V4=std::array<float,4>;
using Case=std::array<V4,13>;
void req(bool x,const char*m){if(!x)throw std::runtime_error(m);}
float fromBits(uint32_t x){float y;std::memcpy(&y,&x,4);return y;}
void set(Case&c,unsigned block,unsigned i,float x,float y,float z){c[block+i]=V4{x,y,z,0.f};}
std::string read(const char*p){std::ifstream f(p,std::ios::binary);req(bool(f),"source read");return {std::istreambuf_iterator<char>(f),{}};}
void setPoint(Case&c,unsigned i,float x,float y,float z){set(c,1,i,x,y,z);}
void setFinish(Case&c,unsigned i,float x,float y,float z){set(c,5,i,x,y,z);}
void setVelocity(Case&c,unsigned i,float x,float y,float z){set(c,9,i,x,y,z);}
int main(int argc,const char**argv){@autoreleasepool{try{
 req(argc==3,"usage: ccd_ee_compare <MatterSource.metal> <append.metalinc>");
 std::string source=read(argv[1]);source+="\n";source+=read(argv[2]);
 NSString* text=[[NSString alloc]initWithBytes:source.data() length:source.size() encoding:NSUTF8StringEncoding];req(text!=nil,"MSL UTF8");
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();req(d!=nil,"Metal device");NSError*e=nil;
 id<MTLLibrary>lib=[d newLibraryWithSource:text options:nil error:&e];req(lib!=nil,e?[[e localizedDescription]UTF8String]:"MSL compile");
 id<MTLFunction>fn=[lib newFunctionWithName:@"numi_matter_metal::nm_contact_ee_wrapper_compare_probe"];req(fn!=nil,"probe function");
 id<MTLComputePipelineState>ps=[d newComputePipelineStateWithFunction:fn error:&e];req(ps!=nil,e?[[e localizedDescription]UTF8String]:"pipeline");
 std::vector<Case> cases;std::vector<const char*> names;
 Case exact{};exact[0]=V4{9.999999747378752e-6f,0.0002500000118743628f,1.f,0.f};
 const uint32_t finishBits[4][3]={{0x3d88c74cu,0xbe602c1du,0x3d0da2e5u},{0x3d860bd3u,0xbe664528u,0x3d273c33u},{0x3d7f6330u,0xbe64db73u,0x3d0de096u},{0x3d860d27u,0xbe6633e3u,0x3d26f376u}};
 const uint32_t velocityBits[4][3]={{0xbecd00b0u,0x3cddc2bdu,0x3ca8d497u},{0xbd96fbd3u,0xbd98ff06u,0x3ce72daau},{0xbd3fb4e4u,0x3c226017u,0xbd589fa4u},{0xbd4bc2deu,0x3cc3a281u,0xbdcc135fu}};
 for(unsigned i=0;i<4;i++){setFinish(exact,i,fromBits(finishBits[i][0]),fromBits(finishBits[i][1]),fromBits(finishBits[i][2]));setVelocity(exact,i,fromBits(velocityBits[i][0]),fromBits(velocityBits[i][1]),fromBits(velocityBits[i][2]));}
 cases.push_back(exact);names.push_back("captured-276-EE-edge2-edge0");
 auto make=[&](){Case c{};c[0]=V4{1.e-5f,0.f,0.f,0.f};return c;};
 Case clear=make();setPoint(clear,0,0,0,0);setPoint(clear,1,1,0,0);setPoint(clear,2,0,0,.02f);setPoint(clear,3,1,0,.02f);for(unsigned i=0;i<4;i++)setFinish(clear,i,clear[1+i][0],clear[1+i][1],clear[1+i][2]);cases.push_back(clear);names.push_back("static-clear");
 Case endpoint=make();setPoint(endpoint,0,0,0,0);setPoint(endpoint,1,1,0,0);setPoint(endpoint,2,0,0,.02f);setPoint(endpoint,3,1,0,.02f);for(unsigned i=0;i<4;i++)setFinish(endpoint,i,endpoint[1+i][0],endpoint[1+i][1],i>=2?.5e-5f:0.f);cases.push_back(endpoint);names.push_back("endpoint-contact");
 Case crossing=make();setPoint(crossing,0,-.1f,0,0);setPoint(crossing,1,.1f,0,0);setPoint(crossing,2,0,-.1f,-.02f);setPoint(crossing,3,0,.1f,-.02f);for(unsigned i=0;i<4;i++)setFinish(crossing,i,crossing[1+i][0],crossing[1+i][1],i>=2?.02f:0.f);cases.push_back(crossing);names.push_back("transient-crossing-endpoint-clear");
 Case miss=make();setPoint(miss,0,-.1f,0,0);setPoint(miss,1,.1f,0,0);setPoint(miss,2,.10002f,-.1f,-.02f);setPoint(miss,3,.10002f,.1f,-.02f);for(unsigned i=0;i<4;i++)setFinish(miss,i,miss[1+i][0],miss[1+i][1],i>=2?.02f:0.f);cases.push_back(miss);names.push_back("transient-near-miss");
 std::vector<V4> input;for(const auto&c:cases)for(const auto&v:c)input.push_back(v);std::vector<V4> zeros(cases.size()*2);
 id<MTLBuffer>ib=[d newBufferWithBytes:input.data() length:input.size()*sizeof(V4) options:MTLResourceStorageModeShared];
 id<MTLBuffer>ob=[d newBufferWithBytes:zeros.data() length:zeros.size()*sizeof(V4) options:MTLResourceStorageModeShared];
 uint32_t count=(uint32_t)cases.size();id<MTLBuffer>cbf=[d newBufferWithBytes:&count length:sizeof(count) options:MTLResourceStorageModeShared];
 id<MTLCommandQueue>q=[d newCommandQueue];id<MTLCommandBuffer>cmd=[q commandBuffer];id<MTLComputeCommandEncoder>enc=[cmd computeCommandEncoder];req(ib&&ob&&cbf&&q&&cmd&&enc,"Metal resources");
 [enc setComputePipelineState:ps];[enc setBuffer:ib offset:0 atIndex:0];[enc setBuffer:ob offset:0 atIndex:1];[enc setBuffer:cbf offset:0 atIndex:2];[enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(count,1,1)];[enc endEncoding];[cmd commit];[cmd waitUntilCompleted];req(cmd.status==MTLCommandBufferStatusCompleted,cmd.error?[[cmd.error localizedDescription]UTF8String]:"Metal run");
 const V4*out=(const V4*)ob.contents;req(out!=nullptr,"Metal readback");std::cout<<std::setprecision(10);
 for(unsigned i=0;i<count;i++)std::cout<<names[i]<<" legacy=["<<out[2*i][0]<<","<<out[2*i][1]<<","<<out[2*i][2]<<","<<out[2*i][3]<<"] fallback=["<<out[2*i+1][0]<<","<<out[2*i+1][1]<<","<<out[2*i+1][2]<<","<<out[2*i+1][3]<<"]\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}}}
