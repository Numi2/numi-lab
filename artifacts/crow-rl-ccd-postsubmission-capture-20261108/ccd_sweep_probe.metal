#include "../src/metal/Collision.metal"
kernel void mr_ccd_sweep_probe(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRShapeGPU* shapes [[buffer(1)]],
    device const MRProjectedColliderGPU* current [[buffer(2)]],
    device const MRProjectedColliderGPU* future [[buffer(3)]],
    device const MRCompiledCollisionPairGPU* pairs [[buffer(4)]],
    device const MRCCDEventStateGPU* eventA [[buffer(5)]],
    device const MRCCDEventStateGPU* eventB [[buffer(6)]],
    device MRCCDPairGPU* records [[buffer(7)]],
    device float4* distances [[buffer(8)]],
    constant uint4& ids [[buffer(9)]],
    device const MRGeometryHeaderGPU* headers [[buffer(10)]],
    device const float4* vertices [[buffer(11)]],
    device const MRMeshBVHNodeGPU* nodes [[buffer(12)]],
    device const MRMeshTriangleGPU* triangles [[buffer(13)]],
    const uint index [[thread_position_in_grid]]) {
    const auto pair=pairs[ids.y];
    if(index<2) {
        const float dt=index==0?eventA[0].time.y:eventB[0].time.y;
        records[index]=resolveCCDPair(ids.x,ids.y,pair,shapes,current,future,
            headers,vertices,nodes,triangles,dispatch,max(dt,0.0f));
    }
    if(index<=256) {
        ConvexQueryResult q={};const float alpha=float(index)/256.0f;
        const bool ok=ccdDistanceAtAlpha(pair.colliderA,pair.colliderB,
            shapes[pair.colliderA],shapes[pair.colliderB],
            current[pair.colliderA],current[pair.colliderB],
            future[pair.colliderA],future[pair.colliderB],
            headers,vertices,nodes,triangles,alpha,q);
        distances[index]=float4(alpha,q.separation,float(q.status),ok?1.0f:0.0f);
    }
}
