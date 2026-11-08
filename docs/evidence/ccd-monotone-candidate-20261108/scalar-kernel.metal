kernel void numi_monotone_scalar_probe(
 device const MRShapeGPU* shapes [[buffer(0)]],
 device const MRProjectedColliderGPU* starts [[buffer(1)]],
 device const MRProjectedColliderGPU* ends [[buffer(2)]],
 device const float* tolerances [[buffer(3)]],
 device float4* output [[buffer(4)]],uint id [[thread_position_in_grid]]) {
 bool clear=false; float lower=0,upper=0;uint work=0;
 const bool handled=certifiedBoxPlaneSweep(shapes[2*id],shapes[2*id+1],starts[2*id],starts[2*id+1],ends[2*id],ends[2*id+1],tolerances[id],16,clear,lower,upper,work);
 output[id]=float4(handled?1.f:0.f,clear?1.f:0.f,float(work),lower);
}
