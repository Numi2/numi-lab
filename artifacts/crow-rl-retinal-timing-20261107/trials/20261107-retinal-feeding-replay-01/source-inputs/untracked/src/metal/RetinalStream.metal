#include <metal_stdlib>
using namespace metal;
#include "metalrobo/retinal_stream_types.h"
#include "metalrobo/task_program_types.h"

kernel void mr_retinal_prepare(
    device const MRTaskStateGPU* tasks [[buffer(0)]],
    device MRRetinalStateGPU* states [[buffer(1)]],
    device uint4* pending [[buffer(2)]],
    device atomic_uint* errors [[buffer(3)]],
    constant MRRetinalDispatchGPU& p [[buffer(4)]],
    uint env [[thread_position_in_grid]]) {
    if (env>=p.layout.x) return;
    MRRetinalStateGPU s=states[env];s.work=uint4(0);
    const uint step=tasks[env].episode.x,episode=tasks[env].episode.y;
    const uint ring=p.clock.z+1u;
    if (atomic_load_explicit(errors+env,memory_order_relaxed)!=0u) {states[env]=s;return;}
    if (tasks[env].status.x==0u) {atomic_store_explicit(errors+env,3u,memory_order_relaxed);states[env]=s;return;}
    const bool fresh=s.timeline.z==0u || s.timeline.y!=episode;
    if ((fresh && step!=0u) || (!fresh && (step<s.timeline.x || ulong(step)>ulong(s.timeline.x)+1ul))) {
        atomic_store_explicit(errors+env,1u,memory_order_relaxed);states[env]=s;return;
    }
    if (!fresh && step==s.timeline.x) {states[env]=s;return;}
    if (fresh) {
        s={};s.work.z=1u;
        for (uint slot=0;slot<ring;++slot) pending[env*ring+slot]=uint4(0);
    }
    const ulong sequence=ulong(step)*p.clock.x/p.clock.y;
    const ulong previous=fresh ? 0ul : ulong(s.timeline.x)*p.clock.x/p.clock.y;
    const bool capture=fresh || sequence!=previous;
    if (capture) {
        pending[env*ring+step%ring]=uint4(1,step,episode,0);
        s.work.x=1u;s.delivery.w+=1u;
    }
    if (step>=p.clock.z) {
        const uint slot=(step-p.clock.z)%ring;
        const uint4 frame=pending[env*ring+slot];
        if (frame.x!=0u && frame.y==step-p.clock.z && frame.z==episode) {
            s.work.y=1u;s.work.w=slot;s.delivery.xyz=frame.xyz;
        }
    }
    s.timeline=uint4(step,episode,1,0);states[env]=s;
}

kernel void mr_retinal_sample(
    device const float4* rgb [[buffer(0)]],
    device MRRetinalStateGPU* states [[buffer(1)]],
    device float4* pendingPixels [[buffer(2)]],
    device float4* visible [[buffer(3)]],
    device atomic_uint* errors [[buffer(4)]],
    constant MRRetinalDispatchGPU& p [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint pixels=p.dimensions.z*p.dimensions.w;
    if (index>=p.layout.x*pixels) return;
    const uint env=index/pixels,pixel=index%pixels,eye=p.output.z;
    const MRRetinalStateGPU s=states[env];
    const uint ring=p.clock.z+1u;
    const uint visibleIndex=(env*p.layout.y+eye)*pixels+pixel;
    if (s.work.z!=0u) visible[visibleIndex]=float4(0);
    if (atomic_load_explicit(errors+env,memory_order_relaxed)!=0u) return;
    if (s.work.x!=0u) {
        const uint bw=p.dimensions.x/p.dimensions.z,bh=p.dimensions.y/p.dimensions.w;
        const uint x=(pixel%p.dimensions.z)*bw,y=(pixel/p.dimensions.z)*bh;
        float3 color=0;bool finite=true;
        for (uint dy=0;dy<bh;++dy) for (uint dx=0;dx<bw;++dx) {
            const float3 sample=rgb[env*p.dimensions.x*p.dimensions.y+(y+dy)*p.dimensions.x+x+dx].xyz;
            finite=finite && all(isfinite(sample)) && all(sample>=0.0f);
            color+=sample;
        }
        if (!finite || !all(isfinite(color))) {atomic_store_explicit(errors+env,2u,memory_order_relaxed);return;}
        color/=float(bw*bh);
        // Fixed bounded response to nonnegative linear radiance, not a
        // per-frame normalization that could encode hidden global context.
        color=color/(1.0f+color);
        const uint slot=s.timeline.x%ring;
        const uint at=(((env*ring+slot)*p.layout.y+eye)*pixels+pixel);
        pendingPixels[at]=float4(color,1);
    }
    if (s.work.y!=0u) {
        const uint at=(((env*ring+s.work.w)*p.layout.y+eye)*pixels+pixel);
        visible[visibleIndex]=pendingPixels[at];
    }
}

kernel void mr_retinal_publish(
    device const MRRetinalStateGPU* states [[buffer(0)]],
    device const float4* visible [[buffer(1)]],
    device atomic_uint* errors [[buffer(2)]],
    device float* actor [[buffer(3)]],
    constant MRRetinalDispatchGPU& p [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint pixels=p.dimensions.z*p.dimensions.w;
    const uint features=p.layout.y*pixels*3u+2u;
    if (index>=p.layout.x*features) return;
    const uint env=index/features,feature=index%features;
    const MRRetinalStateGPU s=states[env];
    const bool valid=s.delivery.x!=0u && atomic_load_explicit(errors+env,memory_order_relaxed)==0u;
    float value=0;
    if (valid && feature<features-2u) value=visible[env*p.layout.y*pixels+feature/3u][feature%3u];
    if (valid && feature==features-2u) value=1;
    if (valid && feature==features-1u) value=float(s.timeline.x-s.delivery.y)*p.timing.x;
    const ulong base=ulong(p.output.x)|(ulong(p.output.y)<<32);
    actor[base+ulong(env)*p.layout.w+p.layout.z+feature]=value;
}
