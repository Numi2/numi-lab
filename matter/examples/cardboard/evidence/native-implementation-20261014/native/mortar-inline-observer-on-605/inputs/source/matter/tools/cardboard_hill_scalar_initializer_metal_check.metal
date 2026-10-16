#include <metal_stdlib>
using namespace metal;

#include "numi/matter/hill_scalar_initializer.h"
#include "numi/matter/hill_compensated_stress.h"
#include "numi/matter/hill_compensated_system.h"

// Compile-only smoke kernel: validates Metal address-space and math bindings
// for the shared initializer. It is not dispatched by the CPU parity check.
kernel void hill_scalar_initializer_compile_check(
    device const float* parameters [[buffer(0)]],
    device const float* acceptedInput [[buffer(1)]],
    device const float* trialInput [[buffer(2)]],
    device float* output [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    thread float accepted[13];
    thread float trial[6];
    thread float candidate[13];
    for (uint i = 0; i < 13; ++i) accepted[i] = acceptedInput[13 * tid + i];
    for (uint i = 0; i < 6; ++i) trial[i] = trialInput[6 * tid + i];
    const int status = nmHillScalarInitializeCandidate(parameters, accepted, trial, candidate);
    output[26 * tid] = float(status);
    for (uint i = 0; i < 13; ++i) output[26 * tid + 1 + i] = candidate[i];
}


// Compile-only smoke for the compensated trial-stress evaluator and its
// analytical Jacobian/directional action. This kernel is never dispatched.
kernel void hill_compensated_stress_compile_check(
    device const float* parameters [[buffer(0)]],
    device const float* acceptedInput [[buffer(1)]],
    device const float* deformationInput [[buffer(2)]],
    device const float* directionInput [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    thread float accepted[13];
    thread float deformation[9];
    thread float direction[9];
    thread float stress[6];
    thread float tangent[54];
    thread float directional[6];
    thread float systemResidual[13];
    thread float systemJacobian[169];
    thread float deformationAction[13];
    for (uint i = 0; i < 13; ++i) accepted[i] = acceptedInput[13 * tid + i];
    for (uint i = 0; i < 9; ++i) {
        deformation[i] = deformationInput[9 * tid + i];
        direction[i] = directionInput[9 * tid + i];
    }
    const device float* row = parameters + 20 * tid;
    const int tangentStatus = nmHillCompensatedTrialStressAndTangent(
        row, deformation, accepted, stress, tangent);
    const int directionStatus = nmHillCompensatedTrialStressDirectionalDerivative(
        row, deformation, direction, directional);
    const int systemStatus = nmHillCompensatedHillImplicitSystem(
        row, deformation, accepted, accepted, direction,
        systemResidual, systemJacobian, deformationAction);
    const uint base = 69 * tid;
    output[base] = float(tangentStatus);
    output[base + 1] = float(directionStatus);
    output[base + 68] = float(systemStatus);
    for (uint i = 0; i < 6; ++i) {
        output[base + 2 + i] = stress[i];
        output[base + 62 + i] = directional[i];
    }
    for (uint i = 0; i < 54; ++i) output[base + 8 + i] = tangent[i];
}
