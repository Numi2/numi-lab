#include "collision-sat.metal"
kernel void random_sat(device const MRShapeGPU* shapes [[buffer(0)]], device const MRProjectedColliderGPU* poses [[buffer(1)]],device float4* out [[buffer(2)]],device const uchar* dummy [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 const uint a=2*i,b=a+1;ConvexQueryResult q={},r={};
 device const MRGeometryHeaderGPU* h=(device const MRGeometryHeaderGPU*)dummy;
 device const float4* v=(device const float4*)dummy;
 device const MRMeshBVHNodeGPU* n=(device const MRMeshBVHNodeGPU*)dummy;
 device const MRMeshTriangleGPU* t=(device const MRMeshTriangleGPU*)dummy;
 bool ok=ccdDistanceAtAlpha(a,b,shapes[a],shapes[b],poses[a],poses[b],poses[a],poses[b],h,v,n,t,0.f,q);
 bool ok2=ccdDistanceAtAlpha(b,a,shapes[b],shapes[a],poses[b],poses[a],poses[b],poses[a],h,v,n,t,0.f,r);
 out[i]=float4(q.separation,r.separation,ok?1.f:0.f,ok2?1.f:0.f);
}
