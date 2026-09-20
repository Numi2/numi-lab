#pragma once
#include "metalrobo/compensated_geometry_gpu.h"

// The body and support-offset expansions are derived by the same kinematic
// owner. Preserve both residuals until after subtracting the authored plane.
inline MRCompensatedScalar nmHumanSupportPairedSeparation(
    MRCompensatedPositionGPU body, MRCompensatedPositionGPU offset,
    mr_float4 ground, mr_float4 normal) {
    const auto point = mrCompensatedVectorAdd(body, offset);
    const auto relative = mrCompensatedVectorAdd(point,
        mrCompensatedVectorNegate(mrCompensatedVector(ground)));
    return mrCompensatedVectorDot(relative, mrCompensatedVector(normal));
}
