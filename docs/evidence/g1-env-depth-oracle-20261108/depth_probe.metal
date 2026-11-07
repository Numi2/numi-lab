#include "/Users/numi/numi-g1-env-clock-time-owner-20261108/src/metal/HybridRenderer.metal"
// Research-only proposal: resolve a fresh depth with the real sensor helpers,
// or retain the supplied previous frame word on a hold.
kernel void mr_research_depth_select(
 device const MRWorldInstanceHeaderGPU* instance [[buffer(0)]],
 constant MRHybridRenderUniformsGPU* uniforms [[buffer(1)]],
 device const MRWorldSensorInstanceGPU* sensors [[buffer(2)]],
 device const uint* globalPixels [[buffer(3)]],
 device const float* geometricDepth [[buffer(4)]],
 device const uint* previous [[buffer(5)]],
 device const uint* sample [[buffer(6)]],
 device uint* output [[buffer(7)]],
 constant uint& base [[buffer(8)]],
 uint tid [[thread_position_in_grid]]) {
 const uint i=base+tid;
 if (!sample[i]) {output[i]=previous[i]; return;}
 float z=0;
 bool valid=measureDepth(geometricDepth[i],sensors[i],frameSeed(instance[i],uniforms[i],globalPixels[i]),uniforms[i],z);
 output[i]=valid?as_type<uint>(z):0xffffffffu;
}
