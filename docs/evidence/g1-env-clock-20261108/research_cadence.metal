#include <metal_stdlib>
#include "metalrobo/engine_types.h"
using namespace metal;
// Candidate records are built before observation. Restore keeps the entire
// prior clock and frame metadata for every environment that did not commit.
kernel void mr_research_commit_visual_clock(
    device uint4* accepted [[buffer(0)]],
    device const uint4* candidate [[buffer(1)]],
    device const MRMetalWorldStatusGPU* world [[buffer(2)]],
    device const MRMetalWorldContactStatusGPU* contact [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint env [[thread_position_in_grid]]) {
    if (env < count && world[env].code == MR_STEP_SUCCESS &&
        contact[env].code == MR_STEP_SUCCESS) accepted[env] = candidate[env];
}
