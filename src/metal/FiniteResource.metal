// This kernel is part of MetalRobo.metallib. It reads accepted world state;
// it never writes q, bodies, contact, actions or a physics status.
#include "metalrobo/finite_resource_types.h"

inline float3 resourceRotate(float4 q, float3 p) {
    return p + 2.0f*cross(q.xyz,cross(q.xyz,p)+q.w*p);
}

kernel void mr_finite_resource_commit(
    device const MRFiniteResourceProgramGPU& program [[buffer(0)]],
    device const MRFiniteResourcePatchGPU* patches [[buffer(1)]],
    device MRFiniteResourceStateGPU* states [[buffer(2)]],
    device const MRBodyStateGPU* bodies [[buffer(3)]],
    device const float* q [[buffer(4)]],
    device const MRMetalWorldStatusGPU* statuses [[buffer(5)]],
    constant MRFiniteResourceDispatchGPU& dispatch [[buffer(6)]],
    uint environment [[thread_position_in_grid]]
) {
    if (environment>=dispatch.counts.x || dispatch.limits.y!=MR_FINITE_RESOURCE_ABI ||
        statuses[environment].code!=MR_STEP_SUCCESS) return;
    const uint bodyBase=environment*dispatch.counts.y;
    const float angle=q[environment*dispatch.counts.z+program.binding.y];
    const MRBodyStateGPU mouth=bodies[bodyBase+program.binding.x];
    if (!isfinite(angle) || !all(isfinite(mouth.position)) ||
        !all(isfinite(mouth.orientation))) return;
    const float3 point=mouth.position.xyz+resourceRotate(mouth.orientation,program.mouthClosed.xyz);
    // Validate all sources before changing any state. A malformed source must
    // not consume the jaw edge or advance the lifetime clock.
    for (uint patch=0;patch<program.binding.z;++patch) {
        const MRBodyStateGPU body=bodies[bodyBase+patches[patch].binding.x];
        if (!all(isfinite(body.position)) || !all(isfinite(body.orientation))) return;
    }
    MRFiniteResourceStateGPU state=states[environment];
    if (state.accounting.w==0u) {
        state={};
        for (uint patch=0;patch<program.binding.z;++patch)
            state.remaining[patch/4u][patch%4u]=patches[patch].binding.y;
        state.accounting.w=1u;
        state.jaw.x=angle;
    }
    const bool closingEdge=state.clock.z!=0u && angle<=program.mouthClosed.w && angle<state.jaw.x;
    if (angle>=program.opening.x) state.clock.z=1u;
    if (closingEdge) {
        // One accepted close consumes the edge even if the mouth misses food.
        // Overlapping volumes have a stable authored priority and one debit.
        state.clock.z=0u;
        for (uint patch=0;patch<program.binding.z;++patch) {
            const MRFiniteResourcePatchGPU source=patches[patch];
            const MRBodyStateGPU body=bodies[bodyBase+source.binding.x];
            const float3 center=body.position.xyz+resourceRotate(body.orientation,source.centerRadius.xyz);
            if (distance(point,center)>source.centerRadius.w) continue;
            const uint room=program.binding.w-state.accounting.x;
            const uint amount=min(min(source.binding.z,state.remaining[patch/4u][patch%4u]),room);
            if (amount==0u) continue;
            state.remaining[patch/4u][patch%4u]-=amount;
            state.accounting.x+=amount;
            state.accounting.y+=amount;
            state.accounting.z+=1u;
            break;
        }
    }
    state.jaw.x=angle;
    state.clock.x+=1u;
    if (state.clock.x==0u) state.clock.y+=1u;
    states[environment]=state;
}
