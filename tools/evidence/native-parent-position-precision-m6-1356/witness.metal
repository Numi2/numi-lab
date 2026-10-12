#include <metal_stdlib>
using namespace metal;
inline void restingCompensatedPositionAdd(
    float3 term, thread float3& sum, thread float3& correction) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    const float3 adjusted=term-correction;
    const float3 next=sum+adjusted;
    correction=(next-sum)-adjusted;
    sum=next;
}

kernel void parent_position_witness(device const float4* inputs [[buffer(0)]],
    device float4* outputs [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    float3 sum=0, correction=0, control=0;
    for(uint j=0;j<21;++j) {
        float4 x=inputs[21*i+j];
        restingCompensatedPositionAdd(x.w*x.xyz,sum,correction);
        control=fma(x.www,x.xyz,control);
    }
    outputs[2*i]=float4(sum,0);
    outputs[2*i+1]=float4(control,0);
}
