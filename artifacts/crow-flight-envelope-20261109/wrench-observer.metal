#include <metal_stdlib>
using namespace metal;
#include "metalrobo/flapping_wing_types.h"

kernel void mr_debug_wrench_before(
    constant const MRFlappingWingGPU* wings [[buffer(0)]],
    constant const MRCompiledFlappingWingDispatchGPU& d [[buffer(1)]],
    device const MRABABodyWrenchGPU* wrench [[buffer(2)]],
    device const MRArticulatedBodyPoseGPU* poses [[buffer(3)]],
    constant const MRAeroTailGPU& tail [[buffer(4)]],
    device float* diagnostic [[buffer(5)]],
    uint env [[thread_position_in_grid]]
) {
    if (env >= d.environmentCount || d.rootBodyIndex >= d.bodyStride) return;
    diagnostic[env * 18u + 15u] = wrench[env * d.bodyStride + d.rootBodyIndex].torque.z;
}

kernel void mr_debug_wrench_after(
    constant const MRFlappingWingGPU* wings [[buffer(0)]],
    constant const MRCompiledFlappingWingDispatchGPU& d [[buffer(1)]],
    device const MRABABodyWrenchGPU* wrench [[buffer(2)]],
    device const MRArticulatedBodyPoseGPU* poses [[buffer(3)]],
    constant const MRAeroTailGPU& tail [[buffer(4)]],
    device float* diagnostic [[buffer(5)]],
    uint env [[thread_position_in_grid]]
) {
    if (env >= d.environmentCount || d.rootBodyIndex >= d.bodyStride ||
        wings[0].bodyIndex >= d.bodyStride || wings[1].bodyIndex >= d.bodyStride ||
        tail.bodyIndex >= d.bodyStride) return;
    uint base = env * d.bodyStride;
    float3 root = poses[base + d.rootBodyIndex].position.xyz;
    float wingYaw = 0.0f;
    for (uint side = 0; side < 2; ++side) {
        uint body = base + wings[side].bodyIndex;
        wingYaw += wrench[body].torque.z +
            cross(poses[body].position.xyz - root, wrench[body].force.xyz).z;
    }
    uint t = base + tail.bodyIndex;
    float tailYaw = wrench[t].torque.z + cross(poses[t].position.xyz - root, wrench[t].force.xyz).z;
    diagnostic[env * 18u + 16u] = wingYaw;
    diagnostic[env * 18u + 17u] = wrench[base + d.rootBodyIndex].torque.z -
        diagnostic[env * 18u + 15u] + tailYaw;
}
