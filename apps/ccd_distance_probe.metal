// Test-only entry point over the exact production CCD query implementation.
#include "../src/metal/Collision.metal"
struct MRDistanceProbeCase { float4 rows[7]; };
struct MRDistanceProbeResult { float4 normalSeparation, pointA, pointB; uint4 status; };
kernel void mr_ccd_distance_probe(
    device const MRDistanceProbeCase* inputs [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRProjectedColliderGPU* current [[buffer(2)]],
    device const MRProjectedColliderGPU* future [[buffer(3)]],
    device const MRGeometryHeaderGPU* headers [[buffer(4)]],
    device const float4* vertices [[buffer(5)]],
    device const MRMeshBVHNodeGPU* nodes [[buffer(6)]],
    device const MRMeshTriangleGPU* triangles [[buffer(7)]],
    device MRDistanceProbeResult* results [[buffer(8)]],
    constant uint& count [[buffer(9)]],
    const uint index [[thread_position_in_grid]]) {
    if(index>=count)return;
    const uint a=2*index,b=a+1;ConvexQueryResult q={};
    const bool ok=ccdDistanceAtAlpha(a,b,shapes[a],shapes[b],current[a],current[b],future[a],future[b],
        headers,vertices,nodes,triangles,inputs[index].rows[1].w,q);
    MRDistanceProbeResult result={};result.normalSeparation=float4(q.normal,q.separation);
    result.pointA=float4(q.pointA,0);result.pointB=float4(q.pointB,0);
    result.status=uint4(q.status,ok,q.iterations,q.fallback);results[index]=result;
}
