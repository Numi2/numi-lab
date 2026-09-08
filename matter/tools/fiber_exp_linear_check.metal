#include "../src/metal/common.metalinc"

// Exercise the production scalar interpreter, including symbolic material
// stress and directional tangent programs. One row stores F and dF.
kernel void fiber_exp_linear_check(
    device const NMScalarProgramGPU* programs [[buffer(0)]],
    device const NMExpressionInstructionGPU* instructions [[buffer(1)]],
    device const float* parameters [[buffer(2)]],
    device const float* rows [[buffer(3)]],
    device float* results [[buffer(4)]],
    constant uint& programCount [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    float f[9], df[9], rate[9];
    const uint row = index / programCount;
    for (uint i = 0; i < 9; ++i) {
        f[i] = rows[row * 18 + i];
        df[i] = rows[row * 18 + 9 + i];
        rate[i] = 0.0f;
    }
    bool valid = true;
    float value = numi_matter_metal::evaluateScalarProgramWithCandidate(
        index % programCount, programs, instructions, parameters, parameters,
        parameters, f, df, rate, 1.0e-4f, 293.15f, valid);
    results[index] = valid ? value : NAN;
}
