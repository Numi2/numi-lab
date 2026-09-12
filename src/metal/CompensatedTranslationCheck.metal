#include <metal_stdlib>
using namespace metal;
#include "metalrobo/compensated_translation_gpu.h"

// Numerical qualification of the exact production helper. This is not an
// alternative dynamics path: input velocities are manufactured test vectors.
kernel void mr_compensated_translation_check(
    device const MRCompensatedRootTranslationGPU* initial [[buffer(0)]],
    device const float4* velocityAndDt [[buffer(1)]],
    device const uint* repetitions [[buffer(2)]],
    device MRCompensatedRootTranslationGPU* result [[buffer(3)]],
    device float4* projection [[buffer(4)]],
    device MRCompensatedPositionGPU* pairedPosition [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    auto state = initial[i];
    for (uint n=0u; n<repetitions[i]; ++n)
        state = mrCompensatedTranslationAdvance(state, velocityAndDt[i], velocityAndDt[i].w);
    result[i] = state;
    projection[i] = mrCompensatedTranslationProjection(state);
    pairedPosition[i] = mrCompensatedTranslationPosition(state, float4(0.0f));
}

#include "metalrobo/compensated_geometry_gpu.h"
struct MRGeometryCheckInput { float4 rotation,point,radii,normal; uint kind; uint padding[3]; };
kernel void mr_compensated_geometry_check(device const MRGeometryCheckInput* inputs [[buffer(0)]],
    device MRCompensatedPositionGPU* out [[buffer(1)]],constant uint& count [[buffer(2)]],uint index [[thread_position_in_grid]]) {
    if(index>=count)return;const auto v=inputs[index];
    out[3*index]=mrCompensatedQuaternionRotate(v.rotation,v.point);
    out[3*index+1]=mrCompensatedSupportOffset(v.rotation,v.point,float4(0,0,0,1),v.radii,v.normal,v.kind);
    auto p=mrCompensatedVector(float4(0));
    for(uint i=0;i<24;++i){const float4 a(.03125f+.0001220703125f*float(i),-.012f,-.083f,0),b(-.007f,.006f,.018f,0);
        p=mrCompensatedVectorAdd(p,mrCompensatedVectorAdd(mrCompensatedQuaternionRotate(v.rotation,a),mrCompensatedVectorNegate(mrCompensatedQuaternionRotate(v.rotation,b))));}
    out[3*index+2]=p;
}
