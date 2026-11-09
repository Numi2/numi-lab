// Exercises the actual shared rigid residual kernel. This is an operator test,
// not a coupled Matter transaction or a contact-force qualification.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/shared.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

static void check(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
int main(int argc, char** argv) { @autoreleasepool { try {
    check(argc == 2, "expected Matter metallib");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice(); check(device != nil, "no Metal device");
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:
        [NSString stringWithUTF8String:argv[1]]] error:&error];
    check(library != nil, "cannot open library");
    id<MTLFunction> function = [library newFunctionWithName:@"numi_matter_metal::nm_contact_subtract_rigid_inertia_residual"];
    check(function != nil, "residual function missing");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    check(pipeline != nil, "residual pipeline failed");
    id<MTLArgumentEncoder> arguments = [function newArgumentEncoderWithBufferIndex:4];
    check(arguments != nil, "contact argument encoder missing");
    id<MTLBuffer> argumentBuffer = [device newBufferWithLength:arguments.encodedLength options:MTLResourceStorageModeShared];
    check(argumentBuffer != nil, "argument buffer failed");
    [arguments setArgumentBuffer:argumentBuffer offset:0];
    id<MTLCommandQueue> queue = [device newCommandQueue]; check(queue != nil, "queue failed");
    NMMatterDispatchGPU dispatch{}; dispatch.environmentCount=1; dispatch.rigidGeneralizedCapacity=1;
    const std::uint32_t articulated=1;
    double maxError=0;
    for (const float h : {0.001f,0.002f}) for (const float mass : {.01f,.025f})
      for (const float gravity : {-9.81f,-3.7f,0.f}) for (const float perturbation : {0.f,.00007f}) {
        const float increment = .5f*h*gravity + perturbation;
        const float massAction = mass*increment;
        const float impulse = (.5f*h)*(mass*gravity);
        const float initial = .000013f;
        nm_float4 initialResidual{initial,0,0,0};
        id<MTLBuffer> massBuffer = [device newBufferWithBytes:&massAction length:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> candidate = [device newBufferWithBytes:&increment length:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> residual = [device newBufferWithBytes:&initialResidual length:sizeof(initialResidual) options:MTLResourceStorageModeShared];
        check(massBuffer && candidate && residual, "allocation failed");
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; check(encoder != nil, "encoder failed");
        [encoder setComputePipelineState:pipeline];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0];
        [encoder setBytes:&articulated length:sizeof(articulated) atIndex:1];
        [encoder setBuffer:massBuffer offset:0 atIndex:2];
        [encoder setBuffer:candidate offset:0 atIndex:3];
        [encoder setBuffer:argumentBuffer offset:0 atIndex:4];
        [encoder setBuffer:residual offset:0 atIndex:5];
        [encoder setBytes:&impulse length:sizeof(impulse) atIndex:6];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        check(command.status==MTLCommandBufferStatusCompleted, "GPU residual command failed");
        const auto value=static_cast<const nm_float4*>(residual.contents)->x;
        const double expected=double(initial)-double(massAction)+double(impulse);
        maxError=std::max(maxError,std::abs(double(value)-expected));
        check(std::isfinite(value) && std::abs(double(value)-expected)<1e-10, "incorrect half-step gravity residual");
      }
    std::cout << "24 rigid residual cases passed, maximum absolute error " << maxError << " N s\n";
    return 0;
} catch(const std::exception& e) {std::cerr << e.what() << "\n"; return 1;} } }
