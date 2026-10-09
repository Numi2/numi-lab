#include <metal_stdlib>
using namespace metal;
#include "NumiHumanRestingBed.metalinc"
struct BedQueryResult { float4 pointGap; float4 normal; uint4 status; };
kernel void numi_human_bed_query_check(
    constant MRHumanRestingBedGPU& bed [[buffer(0)]],
    device const float* heights [[buffer(1)]],
    device const float4* points [[buffer(2)]],
    device BedQueryResult* output [[buffer(3)]],
    constant uint& count [[buffer(4)]],uint index [[thread_position_in_grid]]) {
    if(index>=count)return;
    const auto q=nmHumanRestingBedQuery(bed,heights,points[index].xyz);
    output[index]={float4(q.point,q.signedGap),float4(q.normal,0),
        uint4(q.valid?1u:0u,q.facet,0,0)};
}
