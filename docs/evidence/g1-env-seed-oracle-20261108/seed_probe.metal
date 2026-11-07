// Include the exact research owner's renderer helper, not a copied formula.
#include "/Users/numi/numi-g1-env-clock-time-owner-20261108/src/metal/HybridRenderer.metal"
kernel void mr_research_seed_probe(
 device const MRWorldInstanceHeaderGPU* instance [[buffer(0)]],
 constant MRHybridRenderUniformsGPU* uniforms [[buffer(1)]],
 device const uint* pixels [[buffer(2)]],
 device uint* seeds [[buffer(3)]],
 constant uint& base [[buffer(4)]],
 uint tid [[thread_position_in_grid]]) {
 const uint i=base+tid;
 seeds[i]=frameSeed(instance[i],uniforms[i],pixels[i]);
}
