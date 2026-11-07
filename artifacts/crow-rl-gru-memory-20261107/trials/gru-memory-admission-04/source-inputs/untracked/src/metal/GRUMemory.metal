#include <metal_stdlib>
#include "metalrobo/gru_memory_types.h"
#include "metalrobo/task_program_types.h"
using namespace metal;
// Input/clock validation finishes for the whole grid before any memory writes.
// A bad lane poisons publication without partially updating other lane memory.
kernel void mr_gru_memory_validate(device const float* input [[buffer(0)]],
    device const MRTaskStateGPU* tasks [[buffer(1)]], device const uint4* metadata [[buffer(2)]],
    device atomic_uint* errors [[buffer(5)]], constant MRGRUMemoryDispatchGPU& p [[buffer(10)]],
    uint env [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]]) {
    threadgroup atomic_uint bad;
    if (lane==0) atomic_store_explicit(&bad,0,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint problem=0;
    if (p.counts.w!=MR_GRU_MEMORY_ABI || tasks[env].status.x!=1u) problem|=1u;
    const uint4 old=metadata[env];
    if (old.z>1u || old.w>1u) problem|=1u;
    if (old.z && old.y==tasks[env].episode.y &&
        tasks[env].episode.x!=old.x &&
        (old.x==0xffffffffu || tasks[env].episode.x!=old.x+1u)) problem|=2u;
    for (uint i=lane;i<p.counts.y;i+=width) {
        const float value=input[p.inputBase+ulong(env)*p.mode.y+i];
        if (!isfinite(value) || abs(value)>p.limits.x) problem|=4u;
    }
    atomic_fetch_or_explicit(&bad,problem,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane==0) {
        const uint result=atomic_load_explicit(&bad,memory_order_relaxed);
        if (result) {
            atomic_fetch_or_explicit(&errors[env+1],result,memory_order_relaxed);
            atomic_fetch_or_explicit(&errors[0],result,memory_order_relaxed);
        }
    }
}
kernel void mr_gru_memory_prepare(device const MRTaskStateGPU* tasks [[buffer(1)]],
    device uint4* metadata [[buffer(2)]], device float* before [[buffer(3)]],
    device const float* proposed [[buffer(4)]], device atomic_uint* errors [[buffer(5)]],
    device uint* resetTrace [[buffer(9)]], constant MRGRUMemoryDispatchGPU& p [[buffer(10)]], uint env [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint width [[threads_per_threadgroup]]) {
    if (atomic_load_explicit(&errors[0],memory_order_relaxed)) return;
    const uint4 old=metadata[env];
    const bool reset=!old.z || old.y!=tasks[env].episode.y;
    const bool advance=!reset && old.x!=tasks[env].episode.x;
    for (uint h=lane;h<p.counts.z;h+=width) {
        const uint index=env*p.counts.z+h;
        if (reset) before[index]=0.0f;
        else if (advance) before[index]=proposed[index];
    }
    // All threads finish reading old metadata before the single writer.
    threadgroup_barrier(mem_flags::mem_device);
    if (lane==0) {
        // Repeating the first proposal of an episode must preserve its reset
        // trace as well as its numerical hidden state.
        const uint proposalReset=reset ? 1u : advance ? 0u : old.w;
        resetTrace[p.resetBase+env]=proposalReset;
        metadata[env]=uint4(tasks[env].episode.x,tasks[env].episode.y,1u,proposalReset);
    }
}
kernel void mr_gru_memory_cell(device const float* input [[buffer(0)]],
    device const MRTaskStateGPU* tasks [[buffer(1)]], device const uint4* metadata [[buffer(2)]],
    device const float* before [[buffer(3)]], device float* proposed [[buffer(4)]],
    device atomic_uint* errors [[buffer(5)]], device const float* parameters [[buffer(6)]],
    device float* output [[buffer(7)]], device float* trace [[buffer(8)]],
    device uint* resetTrace [[buffer(9)]], constant MRGRUMemoryDispatchGPU& p [[buffer(10)]],
    uint index [[thread_position_in_grid]]) {
    const uint I=p.counts.y,H=p.counts.z;
    if (index>=p.counts.x*H || atomic_load_explicit(&errors[0],memory_order_relaxed)) return;
    const uint env=index/H,h=index-env*H;
    const uint4 old=metadata[env];
    // Prepare already ran for real inference. Preview does not run it.
    const bool reset=!old.z || old.y!=tasks[env].episode.y;
    const bool useProposal=p.mode.x && !reset && old.x!=tasks[env].episode.x;
    const uint wi=0,wh=3*H*I,bi=wh+3*H*H,bh=bi+3*H;
    float3 x=float3(parameters[bi+h],parameters[bi+H+h],parameters[bi+2*H+h]);
    float3 recurrent=float3(parameters[bh+h],parameters[bh+H+h],parameters[bh+2*H+h]);
    for (uint i=0;i<I;++i) {
        const float value=input[p.inputBase+ulong(env)*p.mode.y+i];
        x.x=fma(parameters[wi+h*I+i],value,x.x);
        x.y=fma(parameters[wi+(H+h)*I+i],value,x.y);
        x.z=fma(parameters[wi+(2*H+h)*I+i],value,x.z);
    }
    for (uint j=0;j<H;++j) {
        const float value=reset ? 0.0f : useProposal ? proposed[env*H+j] : before[env*H+j];
        recurrent.x=fma(parameters[wh+h*H+j],value,recurrent.x);
        recurrent.y=fma(parameters[wh+(H+h)*H+j],value,recurrent.y);
        recurrent.z=fma(parameters[wh+(2*H+h)*H+j],value,recurrent.z);
    }
    const float r=1.0f/(1.0f+exp(-(x.x+recurrent.x)));
    const float z=1.0f/(1.0f+exp(-(x.y+recurrent.y)));
    const float n=tanh(x.z+r*recurrent.z);
    const float previous=reset ? 0.0f : useProposal ? proposed[index] : before[index];
    const float value=(1.0f-z)*n+z*previous;
    if (!isfinite(value) || abs(value)>1.000001f) {
        atomic_fetch_or_explicit(&errors[env+1],8u,memory_order_relaxed);
        atomic_fetch_or_explicit(&errors[0],8u,memory_order_relaxed);return;
    }
    trace[p.traceBase+index]=previous;
    // Real reset trace was written from the pre-prepare metadata.
    if (p.mode.x && h==0) resetTrace[p.resetBase+env]=reset ? 1u : useProposal ? 0u : old.w;
    if (!p.mode.x) proposed[index]=value;
    output[p.outputBase+index]=value;
}
