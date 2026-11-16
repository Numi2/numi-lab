#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
int main(int argc,const char**argv){@autoreleasepool{
 if(argc!=4){fprintf(stderr,"usage: driver shader input output\n");return 1;}
 NSError* err=nil;NSString* source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&err];NSData* input=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]];
 if(!source||!input||input.length!=96*80){fprintf(stderr,"invalid shader or input\n");return 2;}
 id<MTLDevice> dev=MTLCreateSystemDefaultDevice();if(!dev){fprintf(stderr,"no Metal device\n");return 3;}
 MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=NO;id<MTLLibrary> lib=[dev newLibraryWithSource:source options:options error:&err];if(!lib){fprintf(stderr,"shader: %s\n",err.localizedDescription.UTF8String);return 4;}
 id<MTLFunction> fn=[lib newFunctionWithName:@"stretch"];id<MTLComputePipelineState> pipe=[dev newComputePipelineStateWithFunction:fn error:&err];if(!pipe){fprintf(stderr,"pipeline: %s\n",err.localizedDescription.UTF8String);return 5;}
 id<MTLBuffer> ib=[dev newBufferWithBytes:input.bytes length:input.length options:MTLResourceStorageModeShared];id<MTLBuffer> ob=[dev newBufferWithLength:96*16 options:MTLResourceStorageModeShared];id<MTLCommandQueue> queue=[dev newCommandQueue];id<MTLCommandBuffer> cb=[queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];[enc setComputePipelineState:pipe];[enc setBuffer:ib offset:0 atIndex:0];[enc setBuffer:ob offset:0 atIndex:1];[enc dispatchThreads:MTLSizeMake(96,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"command buffer: %s\n",cb.error.localizedDescription.UTF8String);return 6;}
 NSData* result=[NSData dataWithBytes:ob.contents length:96*16];if(![result writeToFile:[NSString stringWithUTF8String:argv[3]] atomically:YES]){fprintf(stderr,"cannot save output\n");return 7;}printf("{\"device\":\"%s\",\"fast_math\":false,\"cases\":96}\n",dev.name.UTF8String);return 0;
}}
