#include "metalrobo/MetalWorld.hpp"
#import <Metal/Metal.h>

// Test-only callback: write a deterministic zero action for this control row
// on the borrowed command buffer, without committing or retaining it.
extern "C" bool measuredSurfaceProbeZeroAction(
    void* context, const metalrobo::MetalWorldDeviceActionPass& pass
) noexcept {
    if (context == nullptr || pass.commandBuffer == nullptr ||
        pass.taskActions == nullptr || pass.actionCount == 0u ||
        pass.environmentCount == 0u || pass.controlStep >= pass.controlStepCount ||
        pass.policyRevision != 1u) return false;
    id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    id<MTLBuffer> actions = (__bridge id<MTLBuffer>)pass.taskActions;
    const NSUInteger width = static_cast<NSUInteger>(pass.environmentCount) *
        static_cast<NSUInteger>(pass.actionCount) * sizeof(float);
    const NSUInteger offset = static_cast<NSUInteger>(pass.controlStep) * width;
    if (offset > actions.length || width > actions.length - offset) return false;
    id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
    if (encoder == nil) return false;
    [encoder fillBuffer:actions range:NSMakeRange(offset, width) value:0];
    [encoder endEncoding];
    return true;
}
