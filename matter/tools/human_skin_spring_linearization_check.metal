#include <metal_stdlib>
using namespace metal;
#include "../src/metal/human_skin_spring_linearization.metalinc"

struct NMHumanSkinSpringLinearizationCheckDispatchGPU {
    uint nv;
    float stiffness;
    uint reserved0;
    uint reserved1;
};

kernel void nm_human_skin_spring_linearization_check(
    constant NMHumanSkinSpringLinearizationCheckDispatchGPU& dispatch [[buffer(0)]],
    device const float* nodePoint [[buffer(1)]],
    device const float* bodyPoint [[buffer(2)]],
    device const float* pointJacobian [[buffer(3)]],
    device const float* pointHessian [[buffer(4)]],
    device const float* nodeDirection [[buffer(5)]],
    device const float* bodyDirection [[buffer(6)]],
    device float* output [[buffer(7)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0u) return;
    const uint nv = dispatch.nv;
    if (nv != 4u) {
        output[0] = NAN;
        return;
    }
    float energy = NAN;
    float3 nodeGradient = float3(NAN);
    float bodyGradient[4] = {NAN, NAN, NAN, NAN};
    float3 nodeHessianAction = float3(NAN);
    float bodyHessianAction[4] = {NAN, NAN, NAN, NAN};
    const bool valid = nmHumanSkinSpringLinearization(
        dispatch.stiffness,
        float3(nodePoint[0], nodePoint[1], nodePoint[2]),
        float3(bodyPoint[0], bodyPoint[1], bodyPoint[2]),
        nv, pointJacobian, pointHessian,
        float3(nodeDirection[0], nodeDirection[1], nodeDirection[2]),
        bodyDirection,
        energy, nodeGradient, bodyGradient,
        nodeHessianAction, bodyHessianAction);
    output[0] = valid ? energy : NAN;
    output[1] = nodeGradient.x;
    output[2] = nodeGradient.y;
    output[3] = nodeGradient.z;
    for (uint dof = 0u; dof < nv; ++dof)
        output[4u + dof] = bodyGradient[dof];
    output[8] = nodeHessianAction.x;
    output[9] = nodeHessianAction.y;
    output[10] = nodeHessianAction.z;
    for (uint dof = 0u; dof < nv; ++dof)
        output[11u + dof] = bodyHessianAction[dof];
    output[15] = valid ? 1.0f : 0.0f;
}
