#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/world_compiler_types.h"
#include "metalrobo/hybrid_renderer_types.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
static void check(bool b,const char* s) {if(!b) throw std::runtime_error(s);}
int main(int argc,char** argv) {@autoreleasepool {try {
 check(argc==2,"metallib argument");
 id<MTLDevice> d=MTLCreateSystemDefaultDevice();check(d!=nil,"device");NSError* err=nil;
 id<MTLLibrary> lib=[d newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];check(lib!=nil,"library");
 id<MTLComputePipelineState> pipe=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mr_research_depth_select"] error:&err];check(pipe!=nil,"pipeline");
 auto alloc=^id<MTLBuffer>(NSUInteger n){return [d newBufferWithLength:n options:MTLResourceStorageModeShared];};
 constexpr uint32_t n=32, width=16;
 id<MTLBuffer> instance=alloc(n*sizeof(MRWorldInstanceHeaderGPU)),uniform=alloc(n*sizeof(MRHybridRenderUniformsGPU)),sensor=alloc(n*sizeof(MRWorldSensorInstanceGPU));
 id<MTLBuffer> pixel=alloc(n*4),geom=alloc(n*4),previous=alloc(n*4),sample=alloc(n*4),output=alloc(n*4);
 check(instance && uniform && sensor && pixel && geom && previous && sample && output,"buffers");
 std::memset(instance.contents,0,instance.length);std::memset(uniform.contents,0,uniform.length);std::memset(sensor.contents,0,sensor.length);
 auto* h=static_cast<MRWorldInstanceHeaderGPU*>(instance.contents);auto* u=static_cast<MRHybridRenderUniformsGPU*>(uniform.contents);auto* s=static_cast<MRWorldSensorInstanceGPU*>(sensor.contents);
 auto* px=static_cast<uint32_t*>(pixel.contents);auto* g=static_cast<float*>(geom.contents);auto* prev=static_cast<uint32_t*>(previous.contents);auto* sm=static_cast<uint32_t*>(sample.contents);auto* dst=static_cast<uint32_t*>(output.contents);
 for(uint32_t i=0;i<n;++i){const uint32_t e=i/width;h[i].identity={101+e,401+e,0,0};u[i].timing={e?2u:0u,0,e?1u:0u,0};u[i].sensorRangeAndResponse={.05f,8.f,.001f,0.f};s[i].noiseAndLatency.y=.005f;px[i]=(e?217u:73u)+(i%width);g[i]=e?.5f:.45f;sm[i]=1;prev[i]=0;}
 id<MTLCommandQueue> q=[d newCommandQueue];check(q!=nil,"queue");
 auto dispatch=^(uint32_t base,uint32_t count){id<MTLCommandBuffer> cmd=[q commandBuffer];id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];check(enc!=nil,"encoder");[enc setComputePipelineState:pipe];
  NSArray<id<MTLBuffer>>* buffers=@[instance,uniform,sensor,pixel,geom,previous,sample,output];for(NSUInteger j=0;j<buffers.count;++j)[enc setBuffer:buffers[j] offset:0 atIndex:j];
  [enc setBytes:&base length:sizeof(base) atIndex:8];[enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(count,1,1)];[enc endEncoding];[cmd commit];[cmd waitUntilCompleted];check(cmd.status==MTLCommandBufferStatusCompleted,"GPU command");};
 dispatch(0,n);for(uint32_t i=0;i<n;++i){check(dst[i]!=0xffffffffu,"initial invalid");prev[i]=dst[i];}
 for(uint32_t i=width;i<n;++i){sm[i]=0;g[i]=.7f;}
 dispatch(0,n);uint32_t mixed[n];std::memcpy(mixed,dst,sizeof(mixed));
 std::memset(dst,0,output.length);dispatch(0,width);dispatch(width,width);
 unsigned mismatch=0,heldMismatch=0;for(uint32_t i=0;i<n;++i){mismatch+=mixed[i]!=dst[i];if(i>=width)heldMismatch+=mixed[i]!=prev[i];}
 for(uint32_t i=0;i<width;++i)u[i].timing.x=1;
 dispatch(0,width);unsigned seedChanged=0;for(uint32_t i=0;i<width;++i)seedChanged+=mixed[i]!=dst[i];
 for(uint32_t i=width;i<n;++i)sm[i]=1;
 dispatch(width,width);unsigned freshChanged=0;for(uint32_t i=width;i<n;++i)freshChanged+=mixed[i]!=dst[i];
 std::cout<<"batch_vs_single_word_mismatches="<<mismatch<<" held_vs_previous_word_mismatches="<<heldMismatch<<" changed_seed_pixels="<<seedChanged<<" forced_fresh_changed_pixels="<<freshChanged<<'\n';
 check(mismatch==0 && heldMismatch==0 && seedChanged>0 && freshChanged>0,"depth reference mismatch or insensitive control");
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
