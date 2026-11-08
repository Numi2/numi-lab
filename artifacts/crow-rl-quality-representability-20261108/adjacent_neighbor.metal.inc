// A continuous Newton direction can round back to the same FP32 point near
// stationarity. Search adjacent representable single/pair moves using the
// current generalized Hessian. Selection is distributed across SIMD32 and
// deterministic; the caller must fully reproject and validate the winner.
inline bool chooseAdjacentVelocity(
    device const MRUnifiedQualityDispatchGPU& dispatch,
    device const MRUnifiedQualityBlockGPU* blocks,
    device const float* dynamics,
    device const float* jacobian,
    device const float* derivatives,
    device float* hessian,
    threadgroup const float* velocity,
    threadgroup const float* gradient,
    threadgroup float* adjacentDeltas,
    threadgroup float* candidateVelocity,
    const uint lane
) {
    const uint nv = dispatch.generalizedVelocityCount;
    const uint changes = 2u * nv;
    for (uint entry = lane; entry < nv * nv; entry += kWidth) {
        hessian[entry] = directHessianEntry(
            dispatch, blocks, dynamics, jacobian, derivatives,
            entry / nv, entry % nv, 0.0f
        );
    }
    for (uint change = lane; change < changes; change += kWidth) {
        const float current = velocity[change / 2u];
        const float target = nextafter(
            current, (change & 1u) == 0u ? -INFINITY : INFINITY
        );
        adjacentDeltas[change] = isfinite(target) ? target - current : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    float bestMaximum = 0.0f;
    float bestSquared = 0.0f;
    for (uint row = 0u; row < nv; ++row) {
        bestMaximum = max(bestMaximum, abs(gradient[row]));
        bestSquared = fma(gradient[row], gradient[row], bestSquared);
    }
    uint bestIndex = MR_UNIFIED_QUALITY_INVALID_INDEX;
    const uint candidates = changes + changes * changes;
    for (uint index = lane; index < candidates; index += kWidth) {
        const bool pair = index >= changes;
        const uint first = pair ? (index - changes) / changes : index;
        const uint second = pair ? (index - changes) % changes : first;
        if (pair && (first >= second || first / 2u == second / 2u)) {
            continue;
        }
        const float firstDelta = adjacentDeltas[first];
        const float secondDelta = pair ? adjacentDeltas[second] : 0.0f;
        if (firstDelta == 0.0f || (pair && secondDelta == 0.0f)) {
            continue;
        }
        float maximum = 0.0f;
        float squared = 0.0f;
        for (uint row = 0u; row < nv; ++row) {
            float value = fma(
                hessian[row * nv + first / 2u], firstDelta, gradient[row]
            );
            if (pair) {
                value = fma(hessian[row * nv + second / 2u], secondDelta, value);
            }
            maximum = max(maximum, abs(value));
            squared = fma(value, value, squared);
        }
        if (isfinite(maximum) && isfinite(squared) &&
            (maximum < bestMaximum ||
             (maximum == bestMaximum && squared < bestSquared))) {
            bestMaximum = maximum;
            bestSquared = squared;
            bestIndex = index;
        }
    }
    const float selectedMaximum = simd_min(bestMaximum);
    const float selectedSquared = simd_min(
        bestMaximum == selectedMaximum ? bestSquared : INFINITY
    );
    const uint selected = simd_min(
        bestMaximum == selectedMaximum && bestSquared == selectedSquared
        ? bestIndex : MR_UNIFIED_QUALITY_INVALID_INDEX
    );
    if (selected == MR_UNIFIED_QUALITY_INVALID_INDEX) {
        return false;
    }
    const bool pair = selected >= changes;
    const uint first = pair ? (selected - changes) / changes : selected;
    const uint second = pair ? (selected - changes) % changes : first;
    for (uint dof = lane; dof < nv; dof += kWidth) {
        float value = velocity[dof];
        if (dof == first / 2u) value += adjacentDeltas[first];
        if (pair && dof == second / 2u) value += adjacentDeltas[second];
        candidateVelocity[dof] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return true;
}
