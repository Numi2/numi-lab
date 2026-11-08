#include <metal_stdlib>
using namespace metal;
#include "metalrobo/compensated_geometry_gpu.h"
struct Input {
    MRCompensatedScalar left, center, right, restLeft, restRight;
    MRCompensatedScalar stiffnessLeft, stiffnessRight, velocityOld, velocityNew, timestepOverMass;
};
struct Output { float naive; float high; float low; };
kernel void rod_precision_probe(device const Input *input [[buffer(0)]],
                                device Output *output [[buffer(1)]],
                                constant uint &count [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    Input x = input[i];
    auto extLeft = mrCompensatedAdd(mrCompensatedAdd(x.center, mrCompensatedNegate(x.left)),
                                    mrCompensatedNegate(x.restLeft));
    auto extRight = mrCompensatedAdd(mrCompensatedAdd(x.right, mrCompensatedNegate(x.center)),
                                     mrCompensatedNegate(x.restRight));
    auto force = mrCompensatedAdd(mrCompensatedMultiply(x.stiffnessLeft, extLeft),
                                   mrCompensatedNegate(mrCompensatedMultiply(x.stiffnessRight, extRight)));
    auto deltaVelocity = mrCompensatedAdd(x.velocityNew, mrCompensatedNegate(x.velocityOld));
    auto residual = mrCompensatedAdd(deltaVelocity, mrCompensatedMultiply(x.timestepOverMass, force));
    float naiveForce = x.stiffnessLeft.high * (x.center.high - x.left.high - x.restLeft.high) -
                       x.stiffnessRight.high * (x.right.high - x.center.high - x.restRight.high);
    output[i] = {x.velocityNew.high - x.velocityOld.high + x.timestepOverMass.high * naiveForce,
                 residual.high, residual.low};
}
