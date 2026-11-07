#include <metal_stdlib>
#include "metalrobo/engine_types.h"
using namespace metal;
// Test-only: candidate has already been evaluated; failed environments
// select their exact pre-observation words before host publication.
kernel void mr_research_selective_restore(
 device uint* working [[buffer(0)]],
 device const uint* accepted [[buffer(1)]],
 device const MRMetalWorldStatusGPU* status [[buffer(2)]],
 device const MRMetalWorldContactStatusGPU* contact [[buffer(3)]],
 constant uint& stride [[buffer(4)]],
 constant uint& environments [[buffer(5)]],
 uint gid [[thread_position_in_grid]]) {
 if (gid >= stride*environments) return;
 uint env=gid/stride;
 if (status[env].code!=MR_STEP_SUCCESS || contact[env].code!=MR_STEP_SUCCESS)
  working[gid]=accepted[gid];
}
// Research fixture uses 50 Hz sensing and 20 ms accepted steps.
kernel void mr_research_accept_sensor_clock(
 device uint2* clocks [[buffer(0)]],
 device const MRMetalWorldStatusGPU* status [[buffer(1)]],
 device const MRMetalWorldContactStatusGPU* contact [[buffer(2)]],
 constant uint& environments [[buffer(3)]],
 uint env [[thread_position_in_grid]]) {
 if(env>=environments) return;
 if(status[env].code==MR_STEP_SUCCESS && contact[env].code==MR_STEP_SUCCESS) {
  uint n=clocks[env].x;
  clocks[env]=uint2(n+1,n);
 }
}
