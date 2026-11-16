#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdint>
struct Input { float A[1089],b[33],J[99],target[3]; };
struct Output { float x[33],lambda[3]; uint32_t status; };
static_assert(sizeof(Input)==4896 && sizeof(Output)==148);
int main(int argc,const char**argv){@autoreleasepool{
 if(argc!=4){fprintf(stderr,"usage: driver shader input output\n");return 1;}
 NSError* err=nil;NSString* source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&err];NSData* input=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]];
 if(!source||!input||input.length!=4*sizeof(Input)){fprintf(stderr,"invalid shader or input\n");return 2;}
 id<MTLDevice> dev=MTLCreateSystemDefaultDevice();if(!dev){fprintf(stderr,"no Metal device\n");return 3;}
 MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=NO;id<MTLLibrary> lib=[dev newLibraryWithSource:source options:options error:&err];if(!lib){fprintf(stderr,"shader: %s\n",err.localizedDescription.UTF8String);return 4;}
 id<MTLFunction> fn=[lib newFunctionWithName:@"rod_kkt"];id<MTLComputePipelineState> pipe=[dev newComputePipelineStateWithFunction:fn error:&err];if(!pipe){fprintf(stderr,"pipeline: %s\n",err.localizedDescription.UTF8String);return 5;}
 id<MTLBuffer> ib=[dev newBufferWithBytes:input.bytes length:input.length options:MTLResourceStorageModeShared];id<MTLBuffer> ob=[dev newBufferWithLength:4*sizeof(Output) options:MTLResourceStorageModeShared];id<MTLCommandQueue> queue=[dev newCommandQueue];id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];[enc setComputePipelineState:pipe];[enc setBuffer:ib offset:0 atIndex:0];[enc setBuffer:ob offset:0 atIndex:1];[enc dispatchThreads:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"command buffer: %s\n",cb.error.localizedDescription.UTF8String);return 6;}
 NSData* result=[NSData dataWithBytes:ob.contents length:4*sizeof(Output)];if(![result writeToFile:[NSString stringWithUTF8String:argv[3]] atomically:YES]){fprintf(stderr,"cannot save output\n");return 7;}printf("{\"device\":\"%s\",\"fast_math\":false,\"cases\":4}\n",dev.name.UTF8String);return 0;
}}
