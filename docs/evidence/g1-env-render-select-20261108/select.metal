#include <metal_stdlib>
#include "metalrobo/engine_types.h"
using namespace metal;
// Diagnostic publication from two real rendered frame references and the
// MetalWorld-owned status buffers; not integrated into the rollout owner.
kernel void mr_research_rendered_select(
 device const uint* held [[buffer(0)]], device const uint* fresh [[buffer(1)]],
 device const MRMetalWorldStatusGPU* world [[buffer(2)]],
 device const MRMetalWorldContactStatusGPU* contact [[buffer(3)]],
 device const uint* sample [[buffer(4)]], device uint* result [[buffer(5)]],
 constant uint& stride [[buffer(6)]], constant uint& envs [[buffer(7)]],
 uint i [[thread_position_in_grid]]) {
 if(i>=stride*envs) return;
 const uint env=i/stride;
 const bool accept=world[env].code==MR_STEP_SUCCESS && contact[env].code==MR_STEP_SUCCESS;
 result[i]=accept && sample[env]?fresh[i]:held[i];
}
