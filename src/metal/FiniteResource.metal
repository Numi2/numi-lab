// This kernel is part of MetalRobo.metallib. It reads accepted world state;
// it never writes q, bodies, contact, actions or a physics status.
#include "metalrobo/finite_resource_types.h"

inline float3 resourceRotate(float4 q, float3 p) {
    return p + 2.0f*cross(q.xyz,cross(q.xyz,p)+q.w*p);
}

// Author only mesh visibility, not pixels, geometry, physics or actor labels.
// The normal renderer retains projection, shading and depth/occlusion authority.
kernel void mr_finite_resource_visual_cues_v1(
    device const MRFiniteResourceStateGPU* states [[buffer(0)]],
    device const MRFiniteResourceCueGPU* cues [[buffer(1)]],
    device uint* visibility [[buffer(2)]],
    constant uint4& counts [[buffer(3)]], // environments, mesh instances, cues, ABI1
    uint index [[thread_position_in_grid]]) {
    if (counts.w!=1u || index>=counts.x*counts.y) return;
    const uint environment=index/counts.y, instance=index%counts.y;
    uint enabled=1u;
    for (uint cue=0;cue<counts.z;++cue) if (cues[cue].binding.x==instance) {
        const auto state=states[environment];const auto binding=cues[cue].binding;
        enabled=state.accounting.w==1u && state.clock.y==0u &&
            state.clock.x<binding.z && (state.clock.w&(1u<<binding.y))!=0u;
        break;
    }
    visibility[index]=enabled;
}

kernel void mr_finite_resource_commit_v4(
    device const MRFiniteResourceProgramGPU& program [[buffer(0)]],
    device const MRFiniteResourcePatchGPU* patches [[buffer(1)]],
    device MRFiniteResourceStateGPU* states [[buffer(2)]],
    device const MRBodyStateGPU* bodies [[buffer(3)]],
    device const float* q [[buffer(4)]],
    device const MRMetalWorldStatusGPU* statuses [[buffer(5)]],
    constant MRFiniteResourceDispatchGPU& dispatch [[buffer(6)]],
    device MRTaskStateGPU* taskStates [[buffer(7)]],
    device const uint* resetMasks [[buffer(8)]],
    constant MRFiniteResourceAllocationGPU& allocation [[buffer(9)]],
    constant ulong& allocationSeed [[buffer(10)]],
    uint environment [[thread_position_in_grid]]
) {
    if (environment>=dispatch.counts.x || dispatch.limits.y!=MR_FINITE_RESOURCE_KERNEL_ABI ||
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
        const uint selected=allocation.version ? mrResourceAllocationMask(allocationSeed,
            environment,program.binding.z,allocation.activePatchCount) : (1u<<program.binding.z)-1u;
        for (uint patch=0;patch<program.binding.z;++patch)
            state.remaining[patch/4u][patch%4u]=(selected & (1u<<patch)) ? patches[patch].binding.y : 0u;
        state.clock.w=allocation.version ? selected : 0u;
        state.accounting.w=1u;
        state.jaw.x=angle;
    }
    const bool resetting=resetMasks[dispatch.reset.y*dispatch.reset.x+environment]!=0u;
    const bool closingEdge=!resetting && state.clock.z!=0u && angle<=program.mouthClosed.w && angle<state.jaw.x;
    uint consumed=0u;
    if (resetting) state.clock.z=0u;
    else if (angle>=program.opening.x) state.clock.z=1u;
    // A close during the delay still consumes its physical edge. It cannot
    // bank an open-jaw gesture and receive food automatically when time elapses.
    const bool canConsume=!allocation.version || state.clock.y!=0u || state.clock.x>=allocation.availableAfterTicks;
    if (closingEdge) {
        // One accepted close consumes the edge even if the mouth misses food.
        // Overlapping volumes have a stable authored priority and one debit.
        state.clock.z=0u;
        for (uint patch=0;canConsume && patch<program.binding.z;++patch) {
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
            consumed=amount;
            break;
        }
    }
    state.jaw.x=angle;
    state.clock.x+=1u;
    if (state.clock.x==0u) state.clock.y+=1u;
    states[environment]=state;
    if (dispatch.limits.z!=0u) {
        float presence=0.0f, available=0.0f;
        for (uint patch=0;(!allocation.version || !allocation.concealProximity) && canConsume && patch<program.binding.z;++patch) {
            const MRFiniteResourcePatchGPU source=patches[patch];
            const MRBodyStateGPU body=bodies[bodyBase+source.binding.x];
            const float3 center=body.position.xyz+resourceRotate(body.orientation,source.centerRadius.xyz);
            const uint remaining=state.remaining[patch/4u][patch%4u];
            if (remaining>0u && distance(point,center)<=source.centerRadius.w) {
                presence=1.0f; available=float(remaining)/float(source.binding.y); break;
            }
        }
        taskStates[environment].nutrition=float4(
            float(state.accounting.x)/float(program.binding.w),float(consumed)*1.0e-9f,presence,available);
    }
}
