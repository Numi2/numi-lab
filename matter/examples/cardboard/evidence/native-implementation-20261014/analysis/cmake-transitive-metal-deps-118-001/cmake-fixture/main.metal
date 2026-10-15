#include <metal_stdlib>
using namespace metal;
#include "inc/value.metalinc"
kernel void depfile_probe(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = probe_value(); }
