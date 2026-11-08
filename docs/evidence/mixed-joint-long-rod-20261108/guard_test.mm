#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <iostream>
int main(int argc,const char**argv){@autoreleasepool{
 if(argc!=2)return 2;
 constexpr uint32_t input[][6]={
  {0,1,0,1,2,1},{7,8,7,8,9,8},{8,9,8,7,8,7},{10,11,10,10,11,10},
  {0,1,0,2,3,2},{0,1,0,3,4,3},{7,8,7,10,11,10},{0,1,0,1,2,2},
  {0,0,0,2,2,1},{0,0,0,3,3,1},{4294967293u,4294967294u,4294967293u,4294967294u,4294967295u,4294967294u},
  {0,1,0,4294967294u,4294967295u,4294967294u}};
 constexpr uint32_t expected[]={1,1,1,1,0,0,0,0,1,0,1,0};
 const NSUInteger n=sizeof(expected)/sizeof(expected[0]);
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();if(!d)return 3;
 NSError*e=nil;id<MTLLibrary>lib=[d newLibraryWithFile:[NSString stringWithUTF8String:argv[1]] error:&e];if(!lib){std::cerr<<e.localizedDescription.UTF8String;return 4;}
 id<MTLFunction>fn=[lib newFunctionWithName:@"guard_check"];id<MTLComputePipelineState>pipe=[d newComputePipelineStateWithFunction:fn error:&e];if(!pipe){std::cerr<<e.localizedDescription.UTF8String;return 5;}
 id<MTLBuffer>in=[d newBufferWithBytes:input length:sizeof(input) options:MTLResourceStorageModeShared];
 id<MTLBuffer>out=[d newBufferWithLength:n*sizeof(uint32_t) options:MTLResourceStorageModeShared];
 id<MTLCommandQueue>queue=[d newCommandQueue];id<MTLCommandBuffer>cb=[queue commandBuffer];id<MTLComputeCommandEncoder>enc=[cb computeCommandEncoder];
 [enc setComputePipelineState:pipe];[enc setBuffer:in offset:0 atIndex:0];[enc setBuffer:out offset:0 atIndex:1];
 [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(n,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
 if(cb.status!=MTLCommandBufferStatusCompleted)return 6;
 auto*r=(uint32_t*)out.contents;bool ok=true;std::cout<<"{\"cases\":[";
 for(NSUInteger i=0;i<n;++i){if(i)std::cout<<",";std::cout<<"{\"index\":"<<i<<",\"actual\":"<<r[i]<<",\"expected\":"<<expected[i]<<"}";ok=ok&&r[i]==expected[i];}
 std::cout<<"],\"passed\":"<<(ok?"true":"false")<<"}\n";return ok?0:7;
}}
